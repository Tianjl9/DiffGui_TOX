# DiffGui-TOX

Toxicity-aware protein-conditioned molecular generation framework based on DiffGui.

## Environment

Please refer to the official repositories for environment setup:

- DiffGui
- Chemprop v2
- DeepBlock

Clone this repository:

```bash
git clone https://github.com/Tianjl9/DiffGui_TOX.git
cd DiffGui_TOX
```

## Dataset and Checkpoints

Datasets and trained checkpoints are available at:

https://zenodo.org/records/20340168

Contents include:

- Processed datasets
- Chemprop checkpoints
- DeepBlock checkpoints
- Ensemble toxicity configuration

## Sampling

```bash
python scripts/sample_tox_ens.py \
--config configs/sample/sample.yml
```

## Evaluation

```bash
python scripts/evaluate.py \
--config configs/eval/eval.yml
```

## Acknowledgement

This project is based on:

- DiffGui
- Chemprop v2
- DeepBlock