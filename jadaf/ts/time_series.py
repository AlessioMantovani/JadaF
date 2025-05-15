"""Time series transformation for jadaf"""

import polars as pl
from jadaf.core.jdf import JDF
import re


def _parse_duration(every: str) -> int:
    """
    Parse duration string like '2m', '1h', '15s' to microseconds. Does not currently support month parsing
    Raises ValueError on invalid format.
    """
    match = re.fullmatch(r"(\d+)(s|m|h|d|w|)", every)
    if not match:
        raise ValueError("Invalid duration format. Use format like '15s', '2m', '1h', or '1d'.")

    amount, unit_key = match.groups()
    multiplier = {"s": 1_000_000, "m": 60_000_000, "h": 3_600_000_000, "d": 86_400_000_000, "w": 604_800_000_000}
    return int(amount) * multiplier[unit_key]

def round_datetime(df: JDF, column: str, every: str, method: str = "nearest") -> JDF:
    interval_us = _parse_duration(every)
    timestamp_us = pl.col(column).dt.epoch("us")

    if method == "nearest":
        rounded_expr = ((timestamp_us + interval_us // 2) // interval_us) * interval_us
    elif method == "floor":
        rounded_expr = (timestamp_us // interval_us) * interval_us
    elif method == "ceil":
        rounded_expr = ((timestamp_us + interval_us - 1) // interval_us) * interval_us
    else:
        raise ValueError("Invalid method. Use 'nearest', 'floor', or 'ceil'.")

    return JDF.wrap(df.with_columns(
        rounded_expr.cast(pl.Datetime).alias(f"{column}_rounded")
    ))

def create_interval_groups(df: JDF, time_column: str, interval_minutes: int = 15, group_col_name: str = 'interval_group') -> JDF:
    df = df.with_columns([
        pl.col(time_column).str.to_datetime().alias(time_column)
    ])

    interval_expr = pl.col(time_column).dt.truncate(f"{interval_minutes}m").alias(group_col_name)

    df = df.with_columns([
        interval_expr
    ]).with_columns([
        pl.col(group_col_name).rank(method="dense").cast(pl.Int32).alias("group_id")
    ])

    return JDF.wrap(df)
