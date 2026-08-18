"""
qworld - Generate evaluation criteria for any question using LLMs.

Usage:
    from qworld import CriteriaGenerator

    gen = CriteriaGenerator(model="gpt-4o", api_key="sk-...")
    result = gen.generate("What is machine learning?")
"""

from .client import CriteriaGenerator

__all__ = ["CriteriaGenerator"]

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("qworld")
except PackageNotFoundError:
    __version__ = "0.1.2"
