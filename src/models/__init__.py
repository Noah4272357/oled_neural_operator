"""Neural-operator model definitions and factories."""

from .fno import FNO1d

from .factory import build_model

__all__ = ["FNO1d", "build_model"]
