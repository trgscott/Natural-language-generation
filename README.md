# Can language models generate more original ideas?

<!-- ABOUT THE PROJECT -->
## About The Project

`NLG.py` tests whether fine-tuning GPT-2 on highly rated fictional book plot summaries and varying decoding paramaters of temperature and top-K can align a model to generate more original ideas. This is tested via self-METEOR and ROUGE-L for the semantic difference between generated texts and a set of baseline texts. A small usefulness test is also made using brainteaser puzzles to test the originality and coherence of the answers generated.

## Requirements

Use Python 3.10 or newer. The experiment is intended to run in a virtual environment.

Install the Python dependencies:

```bash
python -m pip install torch transformers pandas numpy scikit-learn tqdm nltk evaluate rouge_score pyarrow fsspec huggingface_hub
```

Download the NLTK resources used by METEOR and tokenisation:

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('wordnet')"
```

On macOS with Apple Silicon, install a PyTorch build with MPS support. The script automatically prefers MPS, then CUDA, then CPU.

## Running the experiment

Run from the directory containing `NLG.py`:

```bash
python NLG.py
```

The default run downloads the book summaries and Goodreads ratings, loads the brainteaser dataset through Hugging Face, evaluates the base GPT-2 model, fine-tunes for five epochs, and evaluates the fine-tuned model. The full experiment can require substantial disk space, memory, and time.

A small smoke-test-sized run is:

```bash
python NLG.py --epochs 1 --plot-samples 3 --puzzle-samples 2
```

Use an explicit device when needed:

```bash
python NLG.py --device mps
python NLG.py --device cuda
python NLG.py --device cpu
```

`--device auto` is the default and selects MPS first, followed by CUDA and CPU.

## Useful options

| Option | Default | Description |
| --- | ---: | --- |
| `--data-dir` | `data` | Directory for downloaded source data. |
| `--output-dir` | `outputs` | Directory for the JSON results file. |
| `--model-name` | `gpt2` | Hugging Face causal language model to load. |
| `--epochs` | `5` | Number of fine-tuning epochs. |
| `--batch-size` | `8` | Training batch size. Reduce this if memory is limited. |
| `--max-length` | `768` | Maximum token length for training and plot generation. |
| `--plot-samples` | `93` | Number of generated plot summaries for each evaluation. |
| `--puzzle-samples` | `10` | Number of shuffled brainteaser questions to answer. |
| `--temperature` | `3.0` | Fine-tuned-model sampling temperature. |
| `--base-temperature` | `5.0` | Base-model sampling temperature. |
| `--top-k` | `25` | Fine-tuned-model top-k sampling value. |
| `--base-top-k` | `50` | Base-model top-k sampling value. |
| `--skip-base` | off | Skip base-model generation and evaluation. |
| `--skip-fine-tuning` | off | Skip fine-tuning and fine-tuned-model evaluation. |
| `--no-download` | off | Fail instead of downloading missing data files. |

See every available option with:

```bash
python NLG.py --help
```

## Output

The script creates `outputs/results.json` by default. It contains:

- the selected device and train/test sizes;
- base-model METEOR and ROUGE-L results, when enabled;
- fine-tuning loss statistics, when enabled;
- fine-tuned-model METEOR and ROUGE-L results, when enabled; and
- generated answers for the selected brainteaser questions.

Downloaded files are stored in `data/` by default. These generated directories can be excluded from version control.

## Data sources

- Book plot summaries: [CMU Book Summary Dataset](https://www.cs.cmu.edu/~dbamman/booksummaries.html)
- Goodreads ratings: [UCSD Goodreads datasets](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html)
- Brainteasers: [ErfanMoosaviMonazzah/brain-teasers](https://huggingface.co/datasets/ErfanMoosaviMonazzah/brain-teasers)
- Base model and tokenizer: [GPT-2 on Hugging Face](https://huggingface.co/gpt2)

Check the original dataset and model terms before redistributing downloaded data or generated results.

<!-- USAGE EXAMPLES -->
## Usage example

Here is an example brainteaser prompt Q and generation A, after fine-tuning on fictional content and setting the temperature to 3.0 and k to 50: 

Q: "Both persons who were playing chess won. What caused this to happen?" 

A: "Could some mysterious person involved have prepared traps hidden through the manipulation game with instructions such as winning conditions and playing numbers incorrectly before them in advance of game administration?"

