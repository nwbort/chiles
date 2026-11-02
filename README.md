# chiles

A tiny GPT that learns to write headlines in the style of Adrian Chiles's
Guardian columns.

The repo scrapes every Adrian Chiles article title from The Guardian, trains a
word-level transformer on them in pure Python (no PyTorch, no NumPy, no GPU),
and writes out a batch of new, mostly nonsensical, occasionally plausible
titles. GitHub Actions re-runs the scrape and continues training every week, so
the model keeps learning and the output keeps changing.

Some recent output (see [`generated.txt`](generated.txt) for the full batch):

```
When history? midwinter
You back Christmas?
I insist 57, never live torch again: for in –
How avid toilet my College hello the Ignore
```

The output is still rough because training happens a few steps at a time. It
gets better as the checkpoint builds up.

## How it works

```
Guardian Content API ──► scrape.py ──► titles.txt ──► train.py ──► model.pkl
                                                              └──► generated.txt
```

1. **Scrape** ([`scrape.py`](scrape.py)) queries the
   [Guardian Content API](https://open-platform.theguardian.com/) for every
   article tagged `profile/adrian-chiles`, oldest first. It drops liveblogs
   (he has a byline on them but they aren't his columns) and writes one
   headline per line to [`titles.txt`](titles.txt).
2. **Train** ([`train.py`](train.py)) is adapted from Andrej Karpathy's
   [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95):
   a scalar autograd engine and a GPT-style transformer written in plain
   Python. This version works on **words** rather than characters. The
   vocabulary is every whitespace-separated token in the titles, plus a
   begin/end-of-sequence token.
3. **Checkpoint**: weights, Adam optimiser state, vocabulary and the global
   step are saved to [`model.pkl`](model.pkl). Each run resumes from it, so
   training adds up across runs.
4. **Generate**: after training, the model samples new titles (at least three
   words each, no duplicates) and writes them to
   [`generated.txt`](generated.txt).

### Model

| Setting        | Default |
| -------------- | ------- |
| Layers         | 2       |
| Embedding size | 32      |
| Attention heads| 4       |
| Context length | 20 words|
| Optimiser      | Adam (β₁ 0.85, β₂ 0.99), lr 0.01 with linear decay |

With a vocabulary of roughly 2,900 words, that comes to about 210k parameters.
Each one is a Python object, so training is slow: a step can take tens of
seconds, which is why CI trains only 25 steps per run.

## Running locally

You need Python 3.10 or newer. Training has no dependencies. Scraping needs
`requests`.

```bash
pip install requests

# Refresh the list of titles
python scrape.py                          # uses the Guardian's public 'test' key
python scrape.py --api-key YOUR_KEY       # faster, with a free registered key

# Train for a few more steps and generate new titles
python train.py --steps 25
```

`train.py` resumes from `model.pkl` if it exists. To start from scratch,
delete or move `model.pkl` first.

### `train.py` options

| Flag                     | Default | Description |
| ------------------------ | ------- | ----------- |
| `--steps`                | 500     | Training steps to run in this session |
| `--n-layer`              | 2       | Transformer layers (new models only) |
| `--n-embd`               | 32      | Embedding size (new models only) |
| `--block-size`           | 20      | Context length in words (new models only) |
| `--n-generate`           | 40      | Number of titles to generate |
| `--temperature`          | 0.8     | Sampling temperature. Lower is safer, higher is weirder |
| `--seed`                 | 42      | Random seed |
| `--checkpoint-interval`  | 25      | Save `model.pkl` every N steps |

When resuming, the architecture and vocabulary come from the checkpoint and the
`--n-layer`, `--n-embd` and `--block-size` flags are ignored. Titles scraped
after the checkpoint was created that contain new words are skipped, since the
vocabulary is fixed.

### `scrape.py` options

| Flag        | Default      | Description |
| ----------- | ------------ | ----------- |
| `--api-key` | `test`       | Guardian API key. The `test` key limits pages to 10 results. |
| `--output`  | `titles.txt` | Output file. Passing `--output` also writes a `.json` file with title, URL and date. |

## Automation

Two scheduled GitHub Actions workflows keep things up to date:

| Workflow | Schedule | What it does |
| -------- | -------- | ------------ |
| [`scrape.yml`](.github/workflows/scrape.yml) | Mondays 06:00 UTC | Runs the scraper and commits `titles.txt`. It won't commit if the new list is shorter than the old one, which protects against partial API responses. |
| [`train.yml`](.github/workflows/train.yml) | Mondays 07:00 UTC | Runs `train.py --steps 25` and commits the updated `model.pkl` and `generated.txt`. |

Both can also be run by hand from the Actions tab (`workflow_dispatch`).

To use a registered Guardian API key in CI, add it as a repository secret
named `GUARDIAN_API_KEY`. Without it, the scraper falls back to the `test` key.

## Files

| File | Description |
| ---- | ----------- |
| `scrape.py` | Guardian API scraper |
| `train.py` | Pure-Python word-level GPT: training, checkpointing and generation |
| `titles.txt` | Training data: one Adrian Chiles headline per line |
| `model.pkl` | Latest checkpoint (weights, optimiser state, vocabulary, step count) |
| `generated.txt` | Most recent batch of generated titles |

## Credits

- Model and autograd engine adapted from Andrej Karpathy's
  [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).
- Headlines from [The Guardian](https://www.theguardian.com/profile/adrian-chiles)
  via the Guardian Open Platform. The titles belong to the Guardian and Adrian
  Chiles. This is an unofficial project with no connection to either.
