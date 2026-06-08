"""Base model class, training strategies, and model registry."""

import inspect
import typing
import enum
import abc

import lightning
import pydantic
import torch

from . import config


class TrainStrategy(enum.StrEnum):
    """An approach to training a model."""

    SUPERVISED = enum.auto()
    """Training given labeled data."""

    UNSUPERVISED = enum.auto()
    """Training given unlabeled data."""


class BaseModel(lightning.LightningModule, abc.ABC):
    """A base class for ML models."""

    def __init__(
        self,
        model_params: config.ModelConfig,
        data_params: pydantic.BaseModel,
        train_metric: str = "val_loss",
        scaler: typing.Any = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_params = model_params
        self.data_params = data_params
        self.train_metric = train_metric
        self.scaler = scaler

        self.save_hyperparameters(
            {
                "model_class": self.__class__.__name__,
                "model_params": model_params.model_dump(),
                "data_params": data_params.model_dump(),
                "train_metric": train_metric,
            }
        )

        self._is_configured = False
        self.test_step_outputs = []

    def __init_subclass__(cls):
        if not inspect.isabstract(cls):
            AVAILABLE_MODELS[cls.__name__] = cls
        return super().__init_subclass__()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["scaler"] = self.scaler

    def on_load_checkpoint(self, checkpoint):
        self.scaler = checkpoint.get("scaler")

    def configure_optimizers(self):
        """Configures optimizers and schedulers based on model parameters."""
        optimizer_kwargs = {"lr": self.model_params.learning_rate}
        if self.model_params.weight_decay is not None:
            optimizer_kwargs["weight_decay"] = self.model_params.weight_decay
        if self.model_params.momentum is not None:
            optimizer_kwargs["momentum"] = self.model_params.momentum
        optimizer = self.model_params.optimizer(self.parameters(), **optimizer_kwargs)

        if self.model_params.scheduler:
            scheduler = self.model_params.scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.train_metric,
                },
            }
        return optimizer

    def configure_model(self):
        if self._is_configured:
            return
        self._define_layers()
        self._is_configured = True

    @property
    @abc.abstractmethod
    def train_strategy(self) -> TrainStrategy:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def loss_function(self) -> torch.nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def _define_layers(self) -> None:
        """Design your model structure here."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, input: torch.Tensor):
        raise NotImplementedError


AVAILABLE_MODELS: dict[str, type[BaseModel]] = {}
"""
A registry of all loaded models.\n
Key: model class name\n
Value: model class reference
"""
