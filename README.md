# Duplicate-of-DeepSC

A public reproduction project for the paper **"Deep Learning Enabled Semantic Communication Systems"**.

This repository focuses on a transformer-based semantic communication pipeline implemented in PyTorch. It is intended as an educational and experimental reproduction rather than an official implementation from the paper authors.

## Project overview

The project trains a semantic transceiver model that encodes text, transmits it through a noisy channel, and reconstructs the message on the receiver side. The codebase includes:

- model definition for the semantic transceiver
- training and evaluation scripts
- dataset loading utilities for the packaged EUR-style data files
- channel simulation utilities for AWGN-based experiments

## Paper reference

- **Title:** Deep Learning Enabled Semantic Communication Systems
- **Repository status:** unofficial reproduction for learning and experimentation

## Repository structure

```text
.
├── content/              # dataset pickles and vocab json
├── main.py               # training entry point
├── predict.py            # evaluation / prediction entry point
├── train.py              # train and validation loops
├── models.py             # semantic transceiver model
├── transformer.py        # transformer building blocks
├── datasets.py           # dataset loading and collation
├── utils.py              # losses, metrics, channel helpers, path helpers
├── requirements.txt      # Python dependencies
├── LICENSE               # open-source license
└── .gitignore            # local/training artifacts to ignore
```

## Installation

From the repository root.

Windows Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with PyTorch 2.0 or newer.

## Dataset expectations

By default, the code expects the following files under `content/`:

- `content/train_data.pkl`
- `content/test_data.pkl`
- `content/vocab.json`

`vocab.json` must contain a `token_to_idx` object with at least the `<START>` token.

You can override the vocab path with `--vocab`.

Dataset files are expected under `content/` by default, and relative paths are resolved against the repository location so commands still work when launched outside the repo root.

## Train

Minimal smoke run on CPU:

```bash
python main.py --epochs 1 --batch-size 2 --device cpu --checkpoint-path ./artifacts/model.pt
```

Important arguments:

- `--device`: `cpu` or a CUDA device such as `cuda:0`
- `--vocab`: path to `vocab.json`
- `--norm-shape`: one or more integers, for example `--norm-shape 128`
- `--checkpoint-path`: explicit checkpoint output path

## Predict / evaluate

Run evaluation with the saved checkpoint:

```bash
python predict.py --batch-size 2 --device cpu --checkpoint-path ./artifacts/model.pt
```

## Checkpoints and logs

- TensorBoard logs are written under the repository `runs/` directory
- Model checkpoints are written to the path passed via `--checkpoint-path`
- Prediction supports the new structured checkpoints directly when evaluated with the same vocabulary mapping used during training
- Older raw `state_dict` checkpoints are supported only when prediction architecture flags match the original training configuration
- The default checkpoint path is `./artifacts/model.pt`

## Running from outside the repo root

The scripts resolve relative dataset, vocab, and checkpoint paths against the repository directory. For example, this works even if your current shell directory is somewhere else:

```bash
python /absolute/path/to/Duplicate-of-DeepSC/main.py --epochs 1 --batch-size 2 --device cpu --checkpoint-path ./artifacts/model.pt
```

## Known limitations

- This is a research reproduction, not a packaged library.
- The repository ships with data artifacts but not a full preprocessing pipeline.
- Default hyperparameters are aimed at experimentation rather than fast CPU training.
- There is currently no automated test suite or CI workflow in the repository.

## License

This project is released under the MIT License. See `LICENSE` for details.
