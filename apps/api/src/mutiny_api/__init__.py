"""Mutiny Hosted API — orchestrates Core; never reimplements oracles."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mutiny-api")
except PackageNotFoundError:
    __version__ = "0.0.0+unreleased"
