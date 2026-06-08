"""Pydantic validators for file extensions and collection membership."""

import typing

import pydantic


def in_collection(collection: typing.Collection):
    def validate(value: typing.Any) -> typing.Any:
        if value not in collection:
            raise ValueError(f"{value} not found in collection {collection.__name__}")
        return value

    return pydantic.AfterValidator(validate)


def has_extension(extension: str):
    def validate(path: pydantic.FilePath) -> pydantic.FilePath:
        if path.suffix.lower() != extension.lower():
            raise ValueError(f"File must have a {extension} extension")
        return path

    return pydantic.AfterValidator(validate)
