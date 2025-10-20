"""Utility functions for handling and processing of datatypes."""

from polars._typing import PolarsDataType
from polars.datatypes.classes import Array, List, Struct


def dtype_to_init_repr(dtype: PolarsDataType, prefix: str = "pl.") -> str:
    """Convert a Polars dtype to a prefixed string representation."""
    # Use local variables for faster attribute access, minimizing function calls and type checks.
    dtype_type = type(dtype)
    if dtype_type is List:
        return _dtype_to_init_repr_list(dtype, prefix)
    elif dtype_type is Array:
        return _dtype_to_init_repr_array(dtype, prefix)
    elif dtype_type is Struct:
        return _dtype_to_init_repr_struct(dtype, prefix)
    else:
        return f"{prefix}{dtype!r}"


def _dtype_to_init_repr_list(dtype: List, prefix: str) -> str:
    class_name = dtype.__class__.__name__
    if dtype.inner is not None:
        inner_repr = dtype_to_init_repr(dtype.inner, prefix)
    else:
        inner_repr = ""
    init_repr = f"{prefix}{class_name}({inner_repr})"
    return init_repr


def _dtype_to_init_repr_array(dtype: Array, prefix: str) -> str:
    # Avoid repeated attribute lookup
    class_name = dtype.__class__.__name__
    inner = dtype.inner
    inner_repr = dtype_to_init_repr(inner, prefix) if inner is not None else ""
    # Build string efficiently using f-string and direct attribute access
    return f"{prefix}{class_name}({inner_repr}, shape={dtype.shape})"


def _dtype_to_init_repr_struct(dtype: Struct, prefix: str) -> str:
    class_name = dtype.__class__.__name__
    inner_list = [
        f"{field_name!r}: {dtype_to_init_repr(inner_dtype, prefix)}"
        for field_name, inner_dtype in dict(dtype).items()
    ]
    inner_repr = "{" + ", ".join(inner_list) + "}"
    init_repr = f"{prefix}{class_name}({inner_repr})"
    return init_repr
