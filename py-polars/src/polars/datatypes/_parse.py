from __future__ import annotations

import enum
import functools
import re
import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal as PyDecimal
from inspect import isclass
from typing import TYPE_CHECKING, Any, ForwardRef, NoReturn, Union, get_args

import polars._reexport as pl
from polars.datatypes.classes import (
    Binary,
    Boolean,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
    Float64,
    Int64,
    List,
    Null,
    Object,
    String,
    Time,
    Unknown,
)
from polars.datatypes.convert import is_polars_dtype

if TYPE_CHECKING:
    from polars._typing import PolarsDataType, PythonDataType, SchemaDict


UnionTypeOld = type(Union[int, str])
if sys.version_info >= (3, 10):
    from types import NoneType, UnionType
else:  # pragma: no cover
    # Define equivalent for older Python versions
    NoneType = type(None)
    UnionType = UnionTypeOld

_INT64 = Int64()

_FLOAT64 = Float64()

_STRING = String()

_BOOLEAN = Boolean()

_DATETIME_US = Datetime("us")

_DATE = Date()

_TIME = Time()

_OBJECT = Object()

_NULL = Null()

_BINARY = Binary()


def parse_into_datatype_expr(input: Any) -> pl.DataTypeExpr:
    """Parse an input into a DataTypeExpr."""
    if isinstance(input, pl.DataTypeExpr):
        return input
    else:
        return parse_into_dtype(input).to_dtype_expr()


def parse_into_dtype(input: Any) -> PolarsDataType:
    """
    Parse an input into a Polars data type.

    Raises
    ------
    TypeError
        If the input cannot be parsed into a Polars data type.
    """
    if is_polars_dtype(input):
        return input
    elif isinstance(input, ForwardRef):
        return _parse_forward_ref_into_dtype(input)
    elif isinstance(input, (UnionType, UnionTypeOld)):
        return _parse_union_type_into_dtype(input)
    else:
        return parse_py_type_into_dtype(input)


def try_parse_into_dtype(input: Any) -> PolarsDataType | None:
    """Try parsing an input into a Polars data type, returning None on failure."""
    try:
        return parse_into_dtype(input)
    except TypeError:
        return None


@functools.lru_cache(16)
def parse_py_type_into_dtype(input: PythonDataType | type[object]) -> PolarsDataType:
    """Convert Python data type to Polars data type."""
    # Reordered for fastest short-circuiting by likelihood and Python type id comparison
    if input is int:
        return _INT64
    if input is float:
        return _FLOAT64
    if input is str:
        return _STRING
    if input is bool:
        return _BOOLEAN
    if input is bytes:
        return _BINARY
    if input is object:
        return _OBJECT
    if input is NoneType:
        return _NULL
    if input is PyDecimal:
        return Decimal
    if input is timedelta:
        return Duration
    if input is time:
        return _TIME
    if input is list or input is tuple:
        return List
    # Cheaper than isclass(): only call if actually a type
    if isinstance(input, type):
        # type: ignore[redundant-expr] is kept for compatibility. May be unnecessary in static checking.
        if issubclass(input, datetime):  # type: ignore[redundant-expr]
            return _DATETIME_US
        if issubclass(input, date):  # type: ignore[redundant-expr]
            return _DATE

    # Only use isclass if not a built-in type to avoid unnecessary expensive checks
    # This avoids calling isclass for int, float, etc.
    if isclass(input) and issubclass(input, enum.Enum):
        return Enum(input)
    # this is required as pass through. Don't remove
    if input == Unknown:
        return Unknown
    # Fast attribute checks to avoid calling hasattr twice
    orig = getattr(input, "__origin__", None)
    args = getattr(input, "__args__", None)
    if orig is not None and args is not None:
        return _parse_generic_into_dtype(input)
    _raise_on_invalid_dtype(input)


def _parse_generic_into_dtype(input: Any) -> PolarsDataType:
    """Parse a generic type (from typing annotation) into a Polars data type."""
    # Extract reference to reduce attribute lookups
    base_type = input.__origin__
    if base_type is not tuple and base_type is not list:
        _raise_on_invalid_dtype(input)

    inner_types = input.__args__

    # Fast path for 1-type generic: List[int], Tuple[int], etc.
    if len(inner_types) == 1:
        inner_type = inner_types[0]
        inner_dtype = parse_py_type_into_dtype(inner_type)
        return List(inner_dtype)

    # For >1 type generic: e.g., Tuple[int, ...], Tuple[int, int], Tuple[int, ...]
    first_type = inner_types[0]
    # Avoid generator and all(): use for-loop for short-circuit and better perf
    for t in inner_types:
        if t is not first_type and t is not ...:
            _raise_on_invalid_dtype(input)
    inner_dtype = parse_py_type_into_dtype(first_type)
    return List(inner_dtype)


PY_TYPE_STR_TO_DTYPE: SchemaDict = {
    "Decimal": Decimal,
    "NoneType": Null(),
    "bool": Boolean(),
    "bytes": Binary(),
    "date": Date(),
    "datetime": Datetime("us"),
    "float": Float64(),
    "int": Int64(),
    "list": List,
    "object": Object(),
    "str": String(),
    "time": Time(),
    "timedelta": Duration,
    "tuple": List,
}


def _parse_forward_ref_into_dtype(input: ForwardRef) -> PolarsDataType:
    """Parse a ForwardRef into a Polars data type."""
    annotation = input.__forward_arg__

    # Strip "optional" designation - Polars data types are always nullable
    formatted = re.sub(r"(^None \|)|(\| None$)", "", annotation).strip()

    try:
        return PY_TYPE_STR_TO_DTYPE[formatted]
    except KeyError:
        _raise_on_invalid_dtype(input)


def _parse_union_type_into_dtype(input: Any) -> PolarsDataType:
    """
    Parse a union of types into a Polars data type.

    Unions of multiple non-null types (e.g. `int | float`) are not supported.

    Parameters
    ----------
    input
        A union type, e.g. `str | None` (new syntax) or `Union[str, None]` (old syntax).
    """
    # Strip "optional" designation - Polars data types are always nullable
    inner_types = [tp for tp in get_args(input) if tp is not NoneType]

    if len(inner_types) != 1:
        _raise_on_invalid_dtype(input)

    input = inner_types[0]
    return parse_into_dtype(input)


def _raise_on_invalid_dtype(input: Any) -> NoReturn:
    """Raise an informative error if the input could not be parsed."""
    # Use type(input) is type - if so, print just input; otherwise, type name and repr for clarity
    is_type = type(input) is type
    input_type = input if is_type else f"of type '{type(input).__name__}'"
    input_detail = "" if is_type else f" (given: {input!r})"
    # Avoid f-string double-call
    msg = f"cannot parse input {input_type} into Polars data type{input_detail}"
    raise TypeError(msg) from None
