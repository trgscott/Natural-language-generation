"""Train and evaluate GPT-2 for original plot-summary generation.

The script downloads the source datasets when needed, evaluates the base model, 
fine-tunes GPT-2 on highly rated book plots, and evaluates the fine-tuned model.

Example:
    python NLG.py --epochs 1 --plot-samples 3 --puzzle-samples 2
"""

from __future__ import annotations

import argparse
import json
import random
import tarfile
import time
import urllib.request
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
)


BOOKS_URL = "https://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz"
GOODREADS_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
    "byGenre/goodreads_books_fantasy_paranormal.json.gz"
)
PUZZLES_DATASET = "ErfanMoosaviMonazzah/brain-teasers"
PROMPT = "Here is the plot summary to a new and original science fiction novel:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--plot-samples", type=int, default=93)
    parser.add_argument("--puzzle-samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--base-temperature", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--base-top-k", type=int, default=50)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-fine-tuning", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS was requested, but no Apple Metal device is available.")
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if requested == "mps" or (requested == "auto" and mps_available):
        return torch.device("mps")
    if requested == "cuda" or (requested == "auto" and cuda_available):
        return torch.device("cuda")
    return torch.device("cpu")


def download_file(url: str, destination: Path, no_download: bool) -> None:
    if destination.exists():
        return
    if no_download:
        raise FileNotFoundError(f"Missing {destination}; remove --no-download to download it.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination)


def load_data(data_dir: Path, no_download: bool) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "booksummaries.tar.gz"
    download_file(BOOKS_URL, archive, no_download)
    books_dir = data_dir / "booksummaries"
    plot_file = books_dir / "booksummaries.txt"
    if not plot_file.exists():
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(data_dir, filter="data")

    reviews_file = data_dir / "goodreads_books_fantasy_paranormal.json.gz"
    download_file(GOODREADS_URL, reviews_file, no_download)
    plots = pd.read_table(
        plot_file,
        header=None,
        names=["Wikipedia_ID", "Freebase_ID", "title", "Author", "Publication_Date", "Book_Genres", "Plot_Summary"],
    )
    reviews = pd.read_json(reviews_file, lines=True)
    merged = plots.merge(reviews[["title", "average_rating"]], how="left")
    merged = merged[merged["average_rating"].notna()].drop_duplicates("Wikipedia_ID")
    merged["average_rating"] = pd.to_numeric(merged["average_rating"])
    plots = merged.loc[merged["average_rating"] > 3.7, "Plot_Summary"].dropna()
    train, test = train_test_split(plots, test_size=0.05, random_state=42)

    try:
        puzzles = pd.read_parquet(
            f"hf://datasets/{PUZZLES_DATASET}/data/sp-00000-of-00001.parquet"
        )
    except Exception as error:
        raise RuntimeError("Unable to load the brain-teaser dataset. Install fsspec and huggingface_hub.") from error
    puzzles = puzzles.sample(frac=1, random_state=42).reset_index(drop=True)
    return pd.Series(train).reset_index(drop=True), pd.Series(test).reset_index(drop=True), puzzles


class GPT2Dataset(Dataset):
    def __init__(self, texts: pd.Series, tokenizer: GPT2Tokenizer, max_length: int):
        self.input_ids = []
        self.attention_masks = []
        for text in texts:
            encoded = tokenizer(
                f"<|startoftext|>{text}<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.input_ids.append(encoded["input_ids"].squeeze(0))
            self.attention_masks.append(encoded["attention_mask"].squeeze(0))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.attention_masks[index]


def self_meteor(texts: list[str]) -> float:
    scores = []
    for index, text in enumerate(texts):
        references = [word_tokenize(other) for other_index, other in enumerate(texts) if other_index != index]
        if references:
            scores.append(nltk.translate.meteor(references, word_tokenize(text)))
    return float(np.mean(scores)) if scores else 0.0


def rouge_against_test(predictions: list[str], test: pd.Series) -> dict[str, float]:
    import evaluate

    references = [[str(value) for value in test] for _ in predictions]
    return evaluate.load("rouge").compute(predictions=predictions, references=references)


def generate_texts(model, tokenizer, device, prompt: str, count: int, max_length: int, top_k: int, temperature: float) -> list[str]:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=top_k,
            top_p=0.95,
            temperature=temperature,
            num_return_sequences=count,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
        )
    return [text.replace("<n>", "\n") for text in tokenizer.batch_decode(outputs, skip_special_tokens=True)]


def generate_puzzle_answers(model, tokenizer, device, questions: pd.Series, count: int, temperature: float, top_k: int) -> list[str]:
    answers = []
    for question in tqdm(list(questions.head(count)), desc="Generating puzzle answers"):
        encoded = tokenizer(str(question), max_length=768, truncation=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(
                **encoded, no_repeat_ngram_size=2, do_sample=True, top_k=top_k,
                top_p=0.95, temperature=temperature, num_return_sequences=1,
                max_length=128, pad_token_id=tokenizer.eos_token_id,
            )
        answers.extend(text.replace("<n>", "\n") for text in tokenizer.batch_decode(output, skip_special_tokens=True))
    return answers


def load_model(model_name: str, device: torch.device):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, bos_token="<|startoftext|>", eos_token="<|endoftext|>", pad_token="<|pad|>")
    model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).to(device)
    return model, tokenizer


def fine_tune(model, tokenizer, train: pd.Series, args: argparse.Namespace, device: torch.device) -> list[dict[str, object]]:
    dataset = GPT2Dataset(train, tokenizer, args.max_length)
    loader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(loader) * args.epochs)
    stats = []
    for epoch in range(args.epochs):
        started = time.time()
        model.train()
        total_loss = 0.0
        for input_ids, attention_mask in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            model.zero_grad()
            loss = model(input_ids, labels=input_ids, attention_mask=attention_mask).loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        stats.append({"epoch": epoch + 1, "training_loss": total_loss / len(loader), "seconds": round(time.time() - started)})
        print(f"Epoch {epoch + 1}: loss={stats[-1]['training_loss']:.3f}")
    return stats


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")
    train, test, puzzles = load_data(args.data_dir, args.no_download)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {"device": str(device), "train_size": len(train), "test_size": len(test)}

    model, tokenizer = load_model(args.model_name, device)
    if not args.skip_base:
        base_plots = generate_texts(model, tokenizer, device, PROMPT, args.plot_samples, args.max_length, args.base_top_k, args.base_temperature)
        results["base"] = {
            "meteor": self_meteor(base_plots),
            "rouge": rouge_against_test(base_plots, test),
            "puzzle_answers": generate_puzzle_answers(model, tokenizer, device, puzzles["question"], args.puzzle_samples, args.base_temperature, args.base_top_k),
        }

    if not args.skip_fine_tuning:
        configuration = GPT2Config.from_pretrained(args.model_name, output_hidden_states=False, pad_token_id=tokenizer.eos_token_id)
        model = GPT2LMHeadModel.from_pretrained(args.model_name, config=configuration).to(device)
        model.resize_token_embeddings(len(tokenizer))
        results["training"] = fine_tune(model, tokenizer, train, args, device)
        fine_tuned_plots = generate_texts(model, tokenizer, device, PROMPT, args.plot_samples, args.max_length, args.top_k, args.temperature)
        results["fine_tuned"] = {
            "meteor": self_meteor(fine_tuned_plots),
            "rouge": rouge_against_test(fine_tuned_plots, test),
            "puzzle_answers": generate_puzzle_answers(model, tokenizer, device, puzzles["question"], args.puzzle_samples, args.temperature, args.top_k),
        }

    output_file = args.output_dir / "results.json"
    output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results written to {output_file}")


if __name__ == "__main__":
    main()
