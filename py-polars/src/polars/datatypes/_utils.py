"""Utility functions for handling and processing of datatypes."""

from polars._typing import PolarsDataType
from polars.datatypes.classes import Array, List, Struct


def dtype_to_init_repr(dtype: PolarsDataType, prefix: str = "pl.") -> str:
    """Convert a Polars dtype to a prefixed string representation."""
    # Use type-based dispatch to avoid isinstance chain
    dtype_type = type(dtype)
    if dtype_type is List:
        return _dtype_to_init_repr_list(dtype, prefix)
    if dtype_type is Array:
        return _dtype_to_init_repr_array(dtype, prefix)
    if dtype_type is Struct:
        return _dtype_to_init_repr_struct(dtype, prefix)
    return f"{prefix}{dtype!r}"


def _dtype_to_init_repr_list(dtype: List, prefix: str) -> str:
    # Minor: inline class_name since it's always "List"
    inner = dtype.inner
    # Avoid unnecessary branch, let f-string handle None
    if inner is not None:
        inner_repr = dtype_to_init_repr(inner, prefix)
    else:
        inner_repr = ""
    return f"{prefix}List({inner_repr})"


def _dtype_to_init_repr_array(dtype: Array, prefix: str) -> str:
    inner = dtype.inner
    if inner is not None:
        inner_repr = dtype_to_init_repr(inner, prefix)
    else:
        inner_repr = ""
    # Inline class name as "Array"
    return f"{prefix}Array({inner_repr}, shape={dtype.shape})"


def _dtype_to_init_repr_struct(dtype: Struct, prefix: str) -> str:
    # Use local variable and generator for more efficient join
    items = dict(dtype).items()
    if not items:
        inner_repr = "{}"
    else:
        inner_repr = (
            "{"
            + ", ".join(
                f"{field_name!r}: {dtype_to_init_repr(inner_dtype, prefix)}"
                for field_name, inner_dtype in items
            )
            + "}"
        )
    return f"{prefix}Struct({inner_repr})"
