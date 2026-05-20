# Image Captioning on Flickr8k

Mini-Challenge for the Deep Learning MSE course (BFH S2). Implements and compares two image captioning architectures on the Flickr8k dataset:

1. **Show and Tell** ([Vinyals et al., 2014](https://arxiv.org/abs/1411.4555)) — CNN encoder + LSTM decoder.
2. **Show, Attend and Tell** ([Xu et al., 2015](https://arxiv.org/abs/1502.03044)) — CNN encoder + LSTM decoder with additive attention over spatial feature maps.

The starter notebook, dataloader, and tokenizer live in [`captioning_starter/`](captioning_starter/). Project specification and grading details are in `DL-MPW-ImgCaptioning.pdf`.

## Project structure

```
.
├── README.md
├── DL-MPW-ImgCaptioning.pdf       # Project brief & grading
└── captioning_starter/
    ├── captioning_starter.ipynb   # Main notebook (model 1, model 2, additional tasks)
    ├── dataloader.py              # Flickr8k Dataset + DataLoader builders
    ├── tokenizer.py               # Caption tokenizer with <pad>/<bos>/<eos>/<unk>
    ├── requirements.txt           # Python dependencies
    ├── data_structure.png
    └── data/
        ├── captions.txt
        ├── splits/                # Predefined train/test splits (do not modify)
        └── images/                # Download from Kaggle (not tracked in git)
```

## Setup with uv

[`uv`](https://docs.astral.sh/uv/) is a fast Python package and project manager. It replaces `pip` + `venv` + `pip-tools` with a single tool.

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via Homebrew
brew install uv
```

After install, make sure `uv` is on your `PATH`:

```bash
# add to ~/.zshrc (or ~/.bashrc) if not already added by the installer
export PATH="$HOME/.local/bin:$PATH"

# reload the shell config
source ~/.zshrc

# verify
uv --version
```

### 2. Create the virtual environment

From the repo root:

```bash
# pin a Python version (the project targets CPython >= 3.11)
uv venv --python 3.11

# activate
source .venv/bin/activate
```

`uv` will store the environment in `.venv/` at the repo root.

### 3. Install dependencies

```bash
uv pip install -r captioning_starter/requirements.txt
```

This installs `torch>=2.6`, `torchvision>=0.21`, `numpy`, `pandas`, `matplotlib`, `Pillow`, `tqdm`, `nltk`, `ipykernel`, and `jupyterlab`.

> **Apple Silicon (M-series):** the default `torch` wheel ships with MPS support — no extra flags needed. The notebook auto-selects `mps` / `cuda` / `cpu`.
>
> **CUDA:** if you need a specific CUDA build of PyTorch, follow the [official selector](https://pytorch.org/get-started/locally/) and run e.g. `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` instead.

### 4. Register the kernel and launch JupyterLab

```bash
python -m ipykernel install --user --name dl-captioning --display-name "Python (dl-captioning)"
jupyter lab
```

Open `captioning_starter/captioning_starter.ipynb` and select the `Python (dl-captioning)` kernel.

## Dataset

The Flickr8k images are **not** included in this repo. Download them from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and place every `.jpg` into `captioning_starter/data/images/`. The `captions.txt` file and the `data/splits/` folder are already provided — do not modify the splits.

The notebook's `data_dir` defaults to `../../captioning_data/data/`; adjust it to `./data/` (or wherever you placed the images) before running.

## Running

Run the notebook top-to-bottom. The structure is:

1. Project setup and imports
2. Transforms, tokenizer, and dataloaders
3. Shared utilities (device selection, BLEU helpers)
4. **Model 1** — Show and Tell (architecture, training, evaluation, reflection)
5. **Model 2** — Show, Attend and Tell (architecture, training, evaluation, reflection)
6. Additional tasks

BLEU is reported on a fixed evaluation subset of the test split — keep it consistent across all models for fair comparison.
