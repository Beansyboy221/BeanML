"""Helpers for loading trained models from checkpoints."""

import typing

import pydantic
import torch

from . import base, validators

CheckpointPath = typing.Annotated[pydantic.FilePath, validators.has_extension(".ckpt")]


def load_model(
    checkpoint_path: CheckpointPath,
    map_location: torch.serialization.MAP_LOCATION,
) -> base.BaseModel:
    """Static method to load the correct child class automatically."""
    checkpoint = torch.load(f=checkpoint_path, map_location=map_location)
    class_name = checkpoint.get("hyper_parameters").get("model_class")
    if not class_name:
        raise ValueError("Model file is missing model class name.")
    model_class = base.AVAILABLE_MODELS.get(class_name)
    if not model_class:
        raise ValueError(f"Model class {class_name} is not available.")
    return model_class.load_from_checkpoint(checkpoint_path)
