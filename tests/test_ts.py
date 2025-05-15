from datetime import datetime, timedelta
import pytest
from pandas import Timestamp
import polars as pl

import jadaf as jd

# Helper function tests
@pytest.mark.parametrize("every,expected", [
    ("5s", 5_000_000),
    ("10m", 600_000_000),
    ("3h", 10_800_000_000),
    ("2d", 172_800_000_000),
])
def test_parse_duration_valid(every, expected):
    assert jd._parse_duration(every) == expected

@pytest.mark.parametrize("invalid_interval", ["5x", "10weeks", "3", "2minutes", "s"])
def test_parse_duration_invalid_format(invalid_interval):
    with pytest.raises(ValueError, match="Invalid duration format"):
        jd._parse_duration(invalid_interval)

def test_parse_duration_edge_cases():
    """Test _parse_duration with edge cases."""
    assert jd._parse_duration("1s") == 1_000_000
    assert jd._parse_duration("1m") == 60_000_000
    assert jd._parse_duration("999s") == 999_000_000
    assert jd._parse_duration("100h") == 360_000_000_000
    assert jd._parse_duration("0d") == 0

def test_parse_duration_invalid_input():
    """Test _parse_duration with various invalid inputs."""
    invalid_inputs = [
        "", "m", "5", "5mm", "5.5m", "-5m", "5M", "five_m", "5min", "5ms"
    ]
    for invalid_input in invalid_inputs:
        with pytest.raises(ValueError):
            jd._parse_duration(invalid_input)

# round_datetime tests
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
def test_round_datetime_basic(input_dt, interval, expected_dt):
    df = pl.DataFrame({"timestamp": [input_dt]})
    result_df = jd.round_datetime(df, "timestamp", interval)
    rounded_dt = result_df["timestamp_rounded"][0]
    assert rounded_dt == expected_dt, f"Expected {expected_dt}, got {rounded_dt}"

@pytest.mark.parametrize("input_dt,interval,method,expected_dt", [
    (datetime(2023, 5, 17, 12, 3, 14), "2m", "floor", datetime(2023, 5, 17, 12, 2, 0)),
    (datetime(2023, 5, 17, 12, 3, 14), "2m", "ceil", datetime(2023, 5, 17, 12, 4, 0)),
    (datetime(2023, 5, 17, 12, 3, 14), "2m", "nearest", datetime(2023, 5, 17, 12, 4, 0)),
])
def test_round_datetime_methods(input_dt, interval, method, expected_dt):
    df = pl.DataFrame({"timestamp": [input_dt]})
    result_df = jd.round_datetime(df, "timestamp", interval, method=method)
    assert result_df["timestamp_rounded"][0] == expected_dt

def test_round_datetime_all_methods():
    """Test all rounding methods with the same input."""
    dt = datetime(2023, 5, 17, 12, 30, 0)
    df = pl.DataFrame({"timestamp": [dt]})

    result_nearest = jd.round_datetime(df, "timestamp", "1h", method="nearest")
    result_floor = jd.round_datetime(df, "timestamp", "1h", method="floor")
    result_ceil = jd.round_datetime(df, "timestamp", "1h", method="ceil")

    assert result_nearest["timestamp_rounded"][0] == datetime(2023, 5, 17, 13, 0, 0)
    assert result_floor["timestamp_rounded"][0] == datetime(2023, 5, 17, 12, 0, 0)
    assert result_ceil["timestamp_rounded"][0] == datetime(2023, 5, 17, 13, 0, 0)

def test_round_datetime_invalid_method():
    dt = datetime(2023, 5, 17, 12, 3, 14)
    df = pl.DataFrame({"timestamp": [dt]})
    with pytest.raises(ValueError, match="Invalid method"):
        jd.round_datetime(df, "timestamp", "2m", method="invalid_method")

def test_round_datetime_multiple_rows():
    dts = [
        datetime(2023, 5, 17, 12, 0, 10),
        datetime(2023, 5, 17, 12, 0, 40),
        datetime(2023, 5, 17, 12, 1, 20)
    ]
    df = pl.DataFrame({"timestamp": dts})
    result_df = jd.round_datetime(jd.JDF.wrap(df), "timestamp", "1m")
    expected = [
        datetime(2023, 5, 17, 12, 0, 0),
        datetime(2023, 5, 17, 12, 1, 0),
        datetime(2023, 5, 17, 12, 1, 0),
    ]
    assert result_df["timestamp_rounded"].to_list() == expected

def test_round_datetime_with_null_values():
    dts = [datetime(2023, 5, 17, 12, 0, 10), None, datetime(2023, 5, 17, 12, 1, 20)]
    df = pl.DataFrame({"timestamp": dts})
    result_df = jd.round_datetime(jd.JDF.wrap(df), "timestamp", "1m")
    expected = [datetime(2023, 5, 17, 12, 0, 0), None, datetime(2023, 5, 17, 12, 1, 0)]
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

def test_round_datetime_with_pandas_timestamp():
    dt = Timestamp("2023-05-17 12:03:14")
    df = pl.DataFrame({"timestamp": [dt]})
    result_df = jd.round_datetime(df, "timestamp", "2m")
    assert result_df["timestamp_rounded"][0] == datetime(2023, 5, 17, 12, 4, 0)

def test_round_datetime_with_different_column_name():
    dt = datetime(2023, 5, 17, 12, 3, 14)
    df = pl.DataFrame({"custom_timestamp": [dt]})
    result_df = jd.round_datetime(df, "custom_timestamp", "2m")
    assert result_df["custom_timestamp_rounded"][0] == datetime(2023, 5, 17, 12, 4, 0)

def test_round_datetime_large_intervals():
    dt = datetime(2023, 5, 17, 12, 0, 0)  # Wednesday
    df = pl.DataFrame({"timestamp": [dt]})
    result_df = jd.round_datetime(df, "timestamp", "7d")
    assert result_df["timestamp_rounded"][0] == datetime(2023, 5, 21, 0, 0, 0)

    dt = datetime(2023, 5, 3, 12, 0, 0)  # Wednesday
    df = pl.DataFrame({"timestamp": [dt]})
    result_df = jd.round_datetime(df, "timestamp", "7d", method="floor")
    assert result_df["timestamp_rounded"][0] == datetime(2023, 5, 1, 0, 0, 0)

def test_round_datetime_edge_cases():
    dt = datetime(2023, 5, 17, 12, 0, 0)
    df = pl.DataFrame({"timestamp": [dt]})
    result_df = jd.round_datetime(df, "timestamp", "1h")
    assert result_df["timestamp_rounded"][0] == datetime(2023, 5, 17, 12, 0, 0)

    dt = datetime(2023, 5, 17, 12, 59, 59, 999999)
    df = pl.DataFrame({"timestamp": [dt]})
    result_df = jd.round_datetime(df, "timestamp", "1h")
    assert result_df["timestamp_rounded"][0] == datetime(2023, 5, 17, 13, 0, 0)

    dt = datetime(1, 1, 1, 0, 0, 0)
    df = pl.DataFrame({"timestamp": [dt]})
    result_df = jd.round_datetime(df, "timestamp", "1h")
    assert result_df["timestamp_rounded"][0] == datetime(1, 1, 1, 0, 0, 0)

def test_round_datetime_performance_with_large_dataset():
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(minutes=i) for i in range(1000)]
    df = pl.DataFrame({"timestamp": dates})
    result_df = jd.round_datetime(df, "timestamp", "15m")
    assert len(result_df) == len(df)

# create_interval_groups tests
def test_create_interval_groups_basic():
    timestamps = [
        datetime(2024, 1, 1, 12, 3),
        datetime(2024, 1, 1, 12, 10),
        datetime(2024, 1, 1, 12, 20),
        datetime(2024, 1, 1, 12, 34),
        datetime(2024, 1, 1, 12, 47),
    ]
    df = pl.DataFrame({"timestamp_api": timestamps, "value": [10, 20, 30, 40, 50]})
    result = jd.create_interval_groups(df, interval_minutes=15)

    expected_intervals = [
        datetime(2024, 1, 1, 12, 0),
        datetime(2024, 1, 1, 12, 0),
        datetime(2024, 1, 1, 12, 15),
        datetime(2024, 1, 1, 12, 30),
        datetime(2024, 1, 1, 12, 45),
    ]
    assert result["interval_group"].to_list() == expected_intervals
    assert result["group_id"].to_list() == [1, 1, 2, 3, 4]

def test_create_interval_groups_custom_group_col():
    df = pl.DataFrame({"timestamp_api": [datetime(2024, 1, 1, 12, 3), datetime(2024, 1, 1, 12, 10)]})
    df = jd.JDF(df)
    result = jd.create_interval_groups(df, time_column="timestamp_api", interval_minutes=15, group_col_name="custom_group")

    assert "custom_group" in result.columns
    assert "group_id" in result.columns
    assert result["custom_group"].to_list() == [datetime(2024, 1, 1, 12, 0)] * 2
    assert result["group_id"].to_list() == [1, 1]

def test_create_interval_groups_datetime_column():
    df = pl.DataFrame({"timestamp": [datetime(2024, 1, 1, 12, 5), datetime(2024, 1, 1, 12, 19)]})
    df = jd.JDF(df)
    result = jd.create_interval_groups(df, time_column="timestamp", interval_minutes=15)

    expected_groups = [datetime(2024, 1, 1, 12, 0), datetime(2024, 1, 1, 12, 15)]
    assert result["interval_group"].to_list() == expected_groups
    assert result["group_id"].to_list() == [1, 2]

def test_create_interval_groups_empty_df():
    df = pl.DataFrame({"timestamp": []})
    result = jd.create_interval_groups(df, time_column="timestamp")
    assert len(result) == 0
    assert "interval_group" in result.columns
    assert "group_id" in result.columns

def test_create_interval_groups_with_null_values():
    timestamps = [datetime(2024, 1, 1, 12, 3), None, datetime(2024, 1, 1, 12, 20)]
    df = pl.DataFrame({"timestamp": timestamps, "value": [10, 20, 30]})
    result = jd.create_interval_groups(df, time_column="timestamp")

    assert result["interval_group"][0] == datetime(2024, 1, 1, 12, 0)
    assert result["interval_group"][2] == datetime(2024, 1, 1, 12, 15)
    assert result["interval_group"][1] is None

def test_create_interval_groups_different_intervals():
    timestamps = [
        datetime(2024, 1, 1, 12, 3),
        datetime(2024, 1, 1, 12, 10),
        datetime(2024, 1, 1, 12, 20),
        datetime(2024, 1, 1, 12, 34),
        datetime(2024, 1, 1, 12, 47),
    ]
    df = pl.DataFrame({"timestamp": timestamps})

    result_5m = jd.create_interval_groups(df, time_column="timestamp", interval_minutes=5)
    expected_intervals_5m = [
        datetime(2024, 1, 1, 12, 0),
        datetime(2024, 1, 1, 12, 10),
        datetime(2024, 1, 1, 12, 20),
        datetime(2024, 1, 1, 12, 30),
        datetime(2024, 1, 1, 12, 45),
    ]
    assert result_5m["interval_group"].to_list() == expected_intervals_5m
    assert result_5m["group_id"].to_list() == [1, 2, 3, 4, 5]

    result_30m = jd.create_interval_groups(df, time_column="timestamp", interval_minutes=30)
    expected_intervals_30m = [
        datetime(2024, 1, 1, 12, 0),
        datetime(2024, 1, 1, 12, 0),
        datetime(2024, 1, 1, 12, 0),
        datetime(2024, 1, 1, 12, 30),
        datetime(2024, 1, 1, 12, 30),
    ]
    assert result_30m["interval_group"].to_list() == expected_intervals_30m
    assert result_30m["group_id"].to_list() == [1, 1, 1, 2, 2]

def test_create_interval_groups_with_mixed_types():
    df = pl.DataFrame({
        "timestamp": [
            "2024-01-01 12:03:00",
            datetime(2024, 1, 1, 12, 10),
            Timestamp("2024-01-01 12:20:00")
        ]
    })
    result = jd.create_interval_groups(df, time_column="timestamp", interval_minutes=15)

    expected_intervals = [
        datetime(2024, 1, 1, 12, 0),
        datetime(2024, 1, 1, 12, 0),
        datetime(2024, 1, 1, 12, 15),
    ]
    assert result["interval_group"].to_list() == expected_intervals

def test_create_interval_groups_cross_day_boundary():
    timestamps = [
        datetime(2024, 1, 1, 23, 50),
        datetime(2024, 1, 2, 0, 5),
        datetime(2024, 1, 2, 0, 20),
    ]
    df = pl.DataFrame({"timestamp": timestamps})
    result = jd.create_interval_groups(df, time_column="timestamp", interval_minutes=15)

    expected_intervals = [
        datetime(2024, 1, 1, 23, 45),
        datetime(2024, 1, 2, 0, 0),
        datetime(2024, 1, 2, 0, 15),
    ]
    assert result["interval_group"].to_list() == expected_intervals
    assert result["group_id"].to_list() == [1, 2, 3]

def test_create_interval_groups_with_invalid_column():
    df = pl.DataFrame({"timestamp": [datetime(2024, 1, 1, 12, 3)]})
    with pytest.raises(Exception):
        jd.create_interval_groups(df, time_column="non_existent_column")

# Integration tests
def test_integration_round_and_interval():
    timestamps = [
        datetime(2024, 1, 1, 12, 3),
        datetime(2024, 1, 1, 12, 17),
        datetime(2024, 1, 1, 12, 32),
    ]
    df = pl.DataFrame({"timestamp": timestamps})

    rounded_df = jd.round_datetime(df, "timestamp", "10m")
    result = jd.create_interval_groups(rounded_df, time_column="timestamp_rounded", interval_minutes=20)

    expected_rounded = [
        datetime(2024, 1, 1, 12, 0),
        datetime(2024, 1, 1, 12, 20),
        datetime(2024, 1, 1, 12, 30),
    ]
    expected_groups = [
        datetime(2024, 1, 1, 12, 0),
        datetime(2024, 1, 1, 12, 20),
        datetime(2024, 1, 1, 12, 20),
    ]

    assert rounded_df["timestamp_rounded"].to_list() == expected_rounded
    assert result["interval_group"].to_list() == expected_groups
    assert result["group_id"].to_list() == [1, 2, 2]