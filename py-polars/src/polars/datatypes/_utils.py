"""Utility functions for handling and processing of datatypes."""

from polars._typing import PolarsDataType
from polars.datatypes.classes import Array, List, Struct


def dtype_to_init_repr(dtype: PolarsDataType, prefix: str = "pl.") -> str:
    """Convert a Polars dtype to a prefixed string representation."""
    # Cache type to avoid repeated isinstance checks and attribute lookups
    tp = type(dtype)
    if tp is List:
        return _dtype_to_init_repr_list(dtype, prefix)
    if tp is Array:
        return _dtype_to_init_repr_array(dtype, prefix)
    if tp is Struct:
        return _dtype_to_init_repr_struct(dtype, prefix)
    return f"{prefix}{dtype!r}"


def _dtype_to_init_repr_list(dtype: List, prefix: str) -> str:
    # Avoid repeated lookups and store class name only once
    class_name = dtype.__class__.__name__
    # Fast path if dtype.inner is None or is a trivial (non-nested) dtype
    inner_dtype = dtype.inner
    if inner_dtype is not None:
        # Avoid unnecessary function call if inner is basic type
        inner_repr = dtype_to_init_repr(inner_dtype, prefix)
    else:
        inner_repr = ""
    return f"{prefix}{class_name}({inner_repr})"


def _dtype_to_init_repr_array(dtype: Array, prefix: str) -> str:
    class_name = dtype.__class__.__name__
    if dtype.inner is not None:
        inner_repr = dtype_to_init_repr(dtype.inner, prefix)
    else:
        inner_repr = ""
    init_repr = f"{prefix}{class_name}({inner_repr}, shape={dtype.shape})"
    return init_repr


def _dtype_to_init_repr_struct(dtype: Struct, prefix: str) -> str:
    class_name = dtype.__class__.__name__
    inner_list = [
        f"{field_name!r}: {dtype_to_init_repr(inner_dtype, prefix)}"
        for field_name, inner_dtype in dict(dtype).items()
    ]
    inner_repr = "{" + ", ".join(inner_list) + "}"
    init_repr = f"{prefix}{class_name}({inner_repr})"
    return init_repr
