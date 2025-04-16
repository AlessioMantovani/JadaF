import polars as pl
import pandas as pd
from typing import Optional, Union

from jadaf.core.jdf import JDF

def _parse_duration(every: str) -> dict:
    """
    Parse duration string like '2m', '1h', '15s' to dictionary for pl.duration
    """
    units = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
    amount = int(every[:-1])
    unit_key = every[-1]
    if unit_key not in units:
        raise ValueError("Invalid duration format. Use 's', 'm', 'h', or 'd'.")
    return {units[unit_key]: amount}


def round_datetime(df: JDF, column: str, every: str) -> JDF:
    """
    Rounds a datetime column to the nearest specified interval using Polars.

    Parameters:
        df (pl.DataFrame): The input DataFrame.
        column (str): The name of the datetime column to round.
        every (str): The interval to round to (e.g., '2m' for 2 minutes,
                     '1h' for 1 hour, '15s' for 15 seconds).

    Returns:
        pl.DataFrame: A new DataFrame with an additional column
                      named '{column}_rounded' containing the rounded datetimes.
    """
    rounded_col = df.select(
        pl.col(column).dt.epoch("us").alias("timestamp_us")
    ).with_columns([
        (pl.lit(pl.duration(**_parse_duration(every)))).alias("interval_us"),
    ]).with_columns([
        ((pl.col("timestamp_us") + pl.col("interval_us") // 2) // pl.col("interval_us") * pl.col("interval_us"))
        .alias("rounded_us")
    ]).with_columns([
        pl.col("rounded_us").cast(pl.Datetime).alias(f"{column}_rounded")
    ])

    return df.with_columns(rounded_col[f"{column}_rounded"])

import polars as pl

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

    return df
