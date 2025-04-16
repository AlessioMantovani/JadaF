import polars as pl
from datetime import datetime
import pytest

import jadaf as jd

@pytest.mark.parametrize("input_dt,interval,expected_dt", [
    (datetime(2023, 5, 17, 12, 3, 14), "2m", datetime(2023, 5, 17, 12, 4, 0)),
    (datetime(2023, 5, 17, 12, 4, 45), "2m", datetime(2023, 5, 17, 12, 6, 0)),
    (datetime(2023, 5, 17, 12, 7, 59), "2m", datetime(2023, 5, 17, 12, 8, 0)),
    (datetime(2023, 5, 17, 12, 29, 59), "1h", datetime(2023, 5, 17, 13, 0, 0)),
    (datetime(2023, 5, 17, 12, 0, 29), "1m", datetime(2023, 5, 17, 12, 0, 0)),
    (datetime(2023, 5, 17, 12, 0, 31), "1m", datetime(2023, 5, 17, 12, 1, 0)),
    (datetime(2023, 5, 17, 23, 59, 30), "1d", datetime(2023, 5, 18, 0, 0, 0)),
    (datetime(2023, 5, 17, 12, 0, 0), "1m", datetime(2023, 5, 17, 12, 0, 0)),
])

def test_round_datetime(input_dt, interval, expected_dt):
    df = pl.DataFrame({"timestamp": [input_dt]})
    result_df = jd.round_datetime(df, "timestamp", interval)
    rounded_dt = result_df["timestamp_rounded"][0]

    assert rounded_dt == expected_dt, f"Expected {expected_dt}, got {rounded_dt}"

@pytest.mark.parametrize("invalid_interval", ["5x", "10weeks", "3", "2minutes", "s"])
def test_parse_duration_invalid_format(invalid_interval):
    with pytest.raises(ValueError, match="Invalid duration format"):
        jd._parse_duration(invalid_interval)

def test_round_datetime_multiple_rows():
    dts = [
        datetime(2023, 5, 17, 12, 0, 10),
        datetime(2023, 5, 17, 12, 0, 40),
        datetime(2023, 5, 17, 12, 1, 20)
    ]
    df = pl.DataFrame({"timestamp": dts})
    result_df = jd.round_datetime(df, "timestamp", "1m")
    expected = [
        datetime(2023, 5, 17, 12, 0, 0),
        datetime(2023, 5, 17, 12, 1, 0),
        datetime(2023, 5, 17, 12, 1, 0),
    ]
    assert result_df["timestamp_rounded"].to_list() == expected

def test_round_datetime_with_old_date():
    dt = datetime(1970, 1, 1, 0, 0, 30)
    df = pl.DataFrame({"timestamp": [dt]})
    result_df = jd.round_datetime(df, "timestamp", "1m")
    assert result_df["timestamp_rounded"][0] == datetime(1970, 1, 1, 0, 1, 0)

def test_round_datetime_with_str_column():
    df = pl.DataFrame({"timestamp": ["2023-05-17T12:00:30"]})
    result_df = jd.round_datetime(df, "timestamp", "1m")
    assert result_df["timestamp_rounded"][0] == datetime(2023, 5, 17, 12, 1, 0)

def test_create_interval_groups_basic():
    timestamps = [
        datetime(2024, 1, 1, 12, 3),
        datetime(2024, 1, 1, 12, 10),
        datetime(2024, 1, 1, 12, 20),
        datetime(2024, 1, 1, 12, 34),
        datetime(2024, 1, 1, 12, 47),
    ]

    df = pl.DataFrame({
        "timestamp_api": timestamps,
        "value": [10, 20, 30, 40, 50]
    })

    result = jd.create_interval_groups(df, interval_minutes=15)

    expected_intervals = [
        datetime(2024, 1, 1, 12, 0),
        datetime(2024, 1, 1, 12, 0),
        datetime(2024, 1, 1, 12, 15),
        datetime(2024, 1, 1, 12, 30),
        datetime(2024, 1, 1, 12, 45),
    ]
    assert result["interval_group"].to_list() == expected_intervals

    expected_ids = [1, 1, 2, 3, 4]
    assert result["group_id"].to_list() == expected_ids
    
def test_create_interval_groups_custom_group_col():
    df = pl.DataFrame({
        "timestamp_api": [
            datetime(2024, 1, 1, 12, 3),
            datetime(2024, 1, 1, 12, 10)
        ]
    })
    result = jd.create_interval_groups(df, time_column="timestamp_api", interval_minutes=15, group_col_name="custom_group")

    assert "custom_group" in result.columns
    assert "group_id" in result.columns
    assert result["custom_group"].to_list() == [datetime(2024, 1, 1, 12, 0)] * 2
    assert result["group_id"].to_list() == [1, 1]
