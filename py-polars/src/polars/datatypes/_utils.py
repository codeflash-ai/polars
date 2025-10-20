"""Utility functions for handling and processing of datatypes."""

from polars._typing import PolarsDataType
from polars.datatypes.classes import Array, List, Struct


def dtype_to_init_repr(dtype: PolarsDataType, prefix: str = "pl.") -> str:
    """Convert a Polars dtype to a prefixed string representation."""
    # Use type checks instead of isinstance for faster dispatch where possible
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
    class_name = dtype.__class__.__name__
    if dtype.inner is not None:
        inner_repr = dtype_to_init_repr(dtype.inner, prefix)
    else:
        inner_repr = ""
    init_repr = f"{prefix}{class_name}({inner_repr}, shape={dtype.shape})"
    return init_repr


def _dtype_to_init_repr_struct(dtype: Struct, prefix: str) -> str:
    # Avoid repeated dict conversion; Struct is likely a mapping already
    fields = dtype.items() if hasattr(dtype, "items") else dict(dtype).items()
    # Use list comprehension directly for efficiency
    inner_list = []
    for field_name, inner_dtype in fields:
        # Only one f-string created per iteration, avoids unnecessary temporaries
        inner_list.append(f"{field_name!r}: {dtype_to_init_repr(inner_dtype, prefix)}")
    inner_repr = "{" + ", ".join(inner_list) + "}"
    class_name = dtype.__class__.__name__
    return f"{prefix}{class_name}({inner_repr})"
