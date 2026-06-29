# Usage Guide

## Installation

```bash
pip install beanml
```

Requires Python 3.11+.

## Quick Start

```python
import lightning
import torch
from torch import nn

from beanml import BaseModel, ModelConfig, TrainStrategy


class MyModel(BaseModel):
    train_strategy = TrainStrategy.SUPERVISED
    loss_function = nn.CrossEntropyLoss()

    def _define_layers(self):
        self.net = nn.Sequential(
            nn.Linear(784, self.model_params.hidden_size),
            nn.ReLU(),
            nn.Linear(self.model_params.hidden_size, 10),
        )

    def forward(self, input):
        return self.net(input)


config = ModelConfig(
    hidden_size=128,
    learning_rate=1e-3,
    optimizer=torch.optim.Adam,
)

model = MyModel(model_params=config, data_params=...)
trainer = lightning.Trainer(max_epochs=10)
trainer.fit(model, dataloader)
```

## ModelConfig

The [`ModelConfig`](../src/beanml/config.py) model configures hyperparameters.

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `hidden_layers` | `int \| None` | `None` | Number of hidden layers (≥1) |
| `hidden_size` | `int \| None` | `None` | Size of each/the first hidden layer (≥1) |
| `latent_size` | `int \| None` | `None` | Size of the latent space (≥1) |
| `dropout` | `float \| None` | `None` | Dropout rate per hidden layer (0.0–1.0) |
| `optimizer` | `SupportedOptimizer \| None` | `None` | Optimizer class |
| `scheduler` | `SupportedScheduler \| None` | `None` | LR scheduler class |
| `learning_rate` | `float \| None` | `None` | Base learning rate (>0) |
| `weight_decay` | `float \| None` | `None` | Weight decay (0.0–1.0) |
| `momentum` | `float \| None` | `None` | Momentum (0.0–1.0) |

## BaseModel

[`BaseModel`](../src/beanml/base.py) extends `lightning.LightningModule`. Subclass it and implement the required abstract members:

| Member | Type | Description |
| ------ | ---- | ----------- |
| `train_strategy` | `TrainStrategy` | Whether the model uses supervised or unsupervised learning |
| `loss_function` | `nn.Module` | The loss function instantiated per forward pass |
| `_define_layers()` | `method` | Build your layers here (called once via `configure_model()`) |
| `forward(x)` | `method` | The forward pass |

### `__init__` params

| Param | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `model_params` | `ModelConfig` | *(required)* | Model hyperparameters |
| `data_params` | `pydantic.BaseModel` | *(required)* | Data-module configuration (user-defined) |
| `train_metric` | `str` | `"val_loss"` | Metric monitored for scheduler stepping |
| `scaler` | `Any` | `None` | Optional scaler for the data |

### Auto-Registration

Any non-abstract subclass is automatically registered in `AVAILABLE_MODELS`. This enables checkpoint loading without knowing the concrete class.

### Optimizer Configuration

`configure_optimizers()` is already implemented. It reads `optimizer`, `scheduler`, `learning_rate`, `weight_decay`, and `momentum` from `model_params` and wires them together. Override if you need custom logic.

## TrainStrategy

[`TrainStrategy`](../src/beanml/base.py) is a `StrEnum`:

| Member | Value |
| ------ | ----- |
| `SUPERVISED` | `"supervised"` |
| `UNSUPERVISED` | `"unsupervised"` |

## Model Registry

All loaded model classes are tracked automatically:

```python
from beanml import AVAILABLE_MODELS

# Look up a model class by name
model_class = AVAILABLE_MODELS["MyModel"]
```

The registry is populated when any non-abstract `BaseModel` subclass is imported.

## Helpers

The [`helpers`](../src/beanml/helpers.py) module provides checkpoint loading:

```python
from beanml import CheckpointPath, load_model

# load_model automatically resolves the correct subclass
model = load_model("path/to/checkpoint.ckpt", map_location="cpu")
```

`CheckpointPath` is a `typing.Annotated[pydantic.FilePath, ...]` that validates the `.ckpt` extension.

## Optimizers & Schedulers

The [`optimizers`](../src/beanml/optimizers.py) module contains validated type aliases for supported optimizers and schedulers as well as a set of all supported options. This allows for easier dynamic use of PyTorch optimizers.
