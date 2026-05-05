# Aeroscapes semantic segmentation

Project for course _neural networks - theory and practise_

Authors:

- Kacper Chmielewski
- Patryk Maciąg
- Adrian Walczak
- Weronika Tarnawska

## Dataset

[aeroscapes](https://www.kaggle.com/datasets/kooaslansefat/uav-segmentation-aeroscapes)

- drone images
- expected color maps (11 classes)
- goal: assign correct class to each pixel of the image

## Repo setup and workflow

Setup:

```sh
uv sync --extra dev
```

Download data:

```sh
uv run python scripts/download_data.py
```

Tests and linters:

```sh
# test
uv run pytest tests/

# format
uv run ruff format .

# lint
uv run ruff check --fix .
```
