import os
import sys
from typing import Any

_VERBOSE_FLAG = os.getenv("POLARS_VERBOSE") == "1"


def verbose() -> bool:
    """
    Fast check for POLARS_VERBOSE state.
    For rare cases where the env variable may change at runtime, call refresh_verbose_flag().
    """
    return _VERBOSE_FLAG


def eprint(*a: Any, **kw: Any) -> None:
    return print(*a, file=sys.stderr, **kw)


def refresh_verbose_flag() -> None:
    """
    Refresh the cached verbose flag to pick up changes in the environment.
    Use only if POLARS_VERBOSE env var may change during runtime.
    """
    global _VERBOSE_FLAG
    _VERBOSE_FLAG = os.getenv("POLARS_VERBOSE") == "1"
