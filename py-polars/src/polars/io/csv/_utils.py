from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polars import DataFrame


def _check_arg_is_1byte(
    arg_name: str, arg: str | None, *, can_be_empty: bool = False
) -> None:
    if isinstance(arg, str):
        arg_byte_length = len(arg.encode("utf-8"))
        if can_be_empty:
            if arg_byte_length > 1:
                msg = (
                    f'{arg_name}="{arg}" should be a single byte character or empty,'
                    f" but is {arg_byte_length} bytes long"
                )
                raise ValueError(msg)
        elif arg_byte_length != 1:
            msg = (
                f'{arg_name}="{arg}" should be a single byte character, but is'
                f" {arg_byte_length} bytes long"
            )
            raise ValueError(msg)


def _update_columns(df: DataFrame, new_columns: Sequence[str]) -> DataFrame:
    # Avoid unnecessary repeated list conversions and assignment loops
    df_width = df.width
    new_len = len(new_columns)
    if df_width > new_len:
        # Only mutate if DataFrame has more columns than new_columns
        cols = df.columns
        # Bulk assignment using slice for better performance
        cols[:new_len] = new_columns
        new_columns = cols
    # Use a slice if assigning the same object (avoid list() if new_columns already is a list)
    if isinstance(new_columns, list):
        df.columns = new_columns
    else:
        df.columns = list(new_columns)
    return df
