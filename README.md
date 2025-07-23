# MyGenerativeRecommenders

Based on: https://github.com/foreverYoungGitHub/generative-recommenders-pl.

## Installation

It is recommended to use `uv` to install the library:

```bash
uv venv -p 3.10 && source .venv/bin/activate
uv pip install --extra dev --extra test -r pyproject.toml
uv pip install -e . --no-deps
```

For Linux systems with GPU support, you can also install `fbgemm-gpu` to enhance performance:

```bash
uv pip install fbgemm-gpu==0.7.0
```

## How to Run

Prepare dataset based on config.

```bash
make prepare_data data=ml-1m
```

Train the Model with Default Configuration

```bash
# Train on CPU
make train trainer=cpu

# Train on GPU
make train trainer=gpu
```

Train the Model with a Specific Experiment Configuration. Choose an experiment configuration from [configs/experiment/](configs/experiment/):

You can use Muon and Scion optimizers now!!

```bash
make train experiment=ml-1m-hstu
```

```bash
make train experiment=ml-1m-hstu-muon
```

```bash
make train experiment=ml-1m-hstu-scion
```
