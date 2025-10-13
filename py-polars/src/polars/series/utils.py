from __future__ import annotations

import inspect
import sys
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import polars._reexport as pl
from polars import functions as F
from polars._utils.wrap import wrap_s
from polars.datatypes import dtype_to_ffiname

if TYPE_CHECKING:
    from polars import Series
    from polars._plr import PySeries
    from polars._typing import PolarsDataType

    if sys.version_info >= (3, 10):
        from typing import ParamSpec
    else:
        from typing_extensions import ParamSpec

    T = TypeVar("T")
    P = ParamSpec("P")
    SeriesMethod = Callable[..., Series]


def expr_dispatch(cls: type[T]) -> type[T]:
    """
    Series/NameSpace class decorator that sets up expression dispatch.

    * Applied to the Series class, and/or any Series 'NameSpace' classes.
    * Walks the class attributes, looking for methods that have empty function
      bodies, with signatures compatible with an existing Expr function.
    * IFF both conditions are met, the empty method is decorated with @call_expr.
    """
    # create lookup of expression functions in this namespace
    namespace = getattr(cls, "_accessor", None)
    expr_lookup = _expr_lookup(namespace)

    # Use vars(cls) and dir(cls) together for safe, faster attribute lookups
    cls_dir = dir(cls)
    cls_vars = vars(cls)

    for name in cls_dir:
        if not name.startswith("_") and name != "plot":
            # Prefer attribute lookup from __dict__ first (faster, skips descriptor logic)
            attr = cls_vars.get(name, None)
            if attr is None:
                attr = getattr(cls, name, None)
            if callable(attr):
                attr = _undecorated(attr)
                co = attr.__code__
                args = co.co_varnames[: co.co_argcount]
                if (namespace, name, args) in expr_lookup and _is_empty_method(attr):
                    setattr(cls, name, call_expr(attr))
    return cls


def _expr_lookup(namespace: str | None) -> set[tuple[str | None, str, tuple[str, ...]]]:
    """Create lookup of potential Expr methods (in the given namespace)."""
    # dummy Expr object that we can introspect
    expr = pl.Expr()
    expr._pyexpr = None  # type: ignore[assignment]

    # optional indirection to "expr.str", "expr.dt", etc
    if namespace is not None:
        expr = getattr(expr, namespace)

    lookup = set()
    expr_dir = dir(expr)
    expr_dict = expr.__class__.__dict__  # For faster attribute checks when possible

    for name in expr_dir:
        if not name.startswith("_"):
            # Try fast path via __dict__ (avoids triggering property logic)
            m = expr_dict.get(name, None)
            if m is None:
                try:
                    m = getattr(expr, name)
                except AttributeError:  # may raise for @property methods
                    continue
            if callable(m):
                m = _undecorated(m)
                co = m.__code__
                args = co.co_varnames[: co.co_argcount]
                lookup.add((namespace, name, args))
    return lookup


def _undecorated(function: Callable[P, T]) -> Callable[P, T]:
    """Return the given function without any decorators."""
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    return function


def call_expr(func: SeriesMethod) -> SeriesMethod:
    """Dispatch Series method to an expression implementation."""

    @wraps(func)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> Series:
        s = wrap_s(self._s)
        expr = F.col(s.name)
        namespace = getattr(self, "_accessor", None)
        if namespace is not None:
            expr = getattr(expr, namespace)
        f = getattr(expr, func.__name__)
        return s.to_frame().select_seq(f(*args, **kwargs)).to_series()

    # note: applying explicit '__signature__' helps IDEs (especially PyCharm)
    # with proper autocomplete, in addition to what @functools.wraps does
    setattr(wrapper, "__signature__", inspect.signature(func))  # noqa: B010
    return wrapper


def _is_empty_method(func: SeriesMethod) -> bool:
    """
    Confirm that the given function has no implementation.

    Definitions of empty:

    - only has a docstring (body is empty)
    - has no docstring and just contains 'pass' (or equivalent)
    """
    fc = func.__code__
    consts = fc.co_consts
    code = fc.co_code
    # _EMPTY_BYTECODE is imported as a singleton set-like object from polars.series.utils
    return (code in _EMPTY_BYTECODE) and (
        (len(consts) == 2 and consts[1] is None)
        or (sys.flags.optimize == 2 and consts == (None,))
    )


class _EmptyBytecodeHelper:
    def __init__(self) -> None:
        # generate bytecode for empty functions with/without a docstring
        def _empty_with_docstring() -> None:
            """"""  # noqa: D419

        def _empty_without_docstring() -> None:
            pass

        self.empty_bytecode = (
            _empty_with_docstring.__code__.co_code,
            _empty_without_docstring.__code__.co_code,
        )

    def __contains__(self, item: bytes) -> bool:
        return item in self.empty_bytecode


_EMPTY_BYTECODE = _EmptyBytecodeHelper()


def get_ffi_func(
    name: str, dtype: PolarsDataType, obj: PySeries
) -> Callable[..., Any] | None:
    """
    Dynamically obtain the proper FFI function/ method.

    Parameters
    ----------
    name
        function or method name where dtype is replaced by <>
        for example
            "call_foo_<>"
    dtype
        polars dtype.
    obj
        Object to find the method for.

    Returns
    -------
    callable or None
        FFI function, or None if not found.
    """
    ffi_name = dtype_to_ffiname(dtype)
    fname = name.replace("<>", ffi_name)
    return getattr(obj, fname, None)


def _with_no_check_length(func: Callable[..., Any]) -> Any:
    from polars._plr import check_length

    # Catch any error so that we can be sure that we always restore length checks
    try:
        check_length(False)
        result = func()
        check_length(True)
    except Exception:
        check_length(True)
        raise
    else:
        return result
