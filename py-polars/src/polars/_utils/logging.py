import os
import sys
from typing import Any

_POLARS_VERBOSE: bool = os.getenv("POLARS_VERBOSE") == "1"


def verbose() -> bool:
    return os.getenv("POLARS_VERBOSE") == "1"


def eprint(*a: Any, **kw: Any) -> None:
    return print(*a, file=sys.stderr, **kw)
