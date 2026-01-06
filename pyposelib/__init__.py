# Adapted from https://github.com/colmap/colmap/blob/1ec758f2eb028049129b852d4d020d7542c866c9/python/pycolmap/__init__.py

import textwrap
from typing import TYPE_CHECKING

from .utils import import_module_symbols

try:
    from . import _core
except ImportError as e:
    raise RuntimeError(
        textwrap.dedent("""
        Cannot import the C++ backend poselib._core.
        Make sure that you successfully install the package with
          $ python -m pip install .
        or build it with
          $ python setup.py build_ext --inplace
        """)
    ) from e

# Type checkers cannot deal with dynamic manipulation of globals.
# Instead, we use the same workaround as PyTorch.
if TYPE_CHECKING:
    from ._core import *  # noqa F403

__all__ = import_module_symbols(
    globals(), _core, exclude=set()
)
__all__.extend(["__version__"])

__version__ = _core.__version__ 