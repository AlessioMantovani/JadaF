import pytest
import polars as pl
from jadaf import JDF


def test_jdf_init_empty():
    """Test creating an empty JDF."""
    jdf = JDF()
    assert jdf.shape == (0, 0)
    assert isinstance(jdf._df, pl.DataFrame)

def test_jdf_slicing():
    """Test slicing functionality."""
    data = pl.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, 50],
        "c": [100, 200, 300, 400, 500]
    })
    jdf = JDF(data)

    result = jdf[1:4]
    assert result.shape == (3, 3)  
    assert result[0, 0] == 2  
    assert result[2, 2] == 400

    result = jdf[:, ['a', 'b']]
    assert result.shape == (5, 2)  
    assert result[0, 0] == 1  
    assert result[4, 1] == 50 

    result = jdf[1:4, 'b']
    assert result.shape == (3, 1)  
    assert result[0, 0] == 20  
    assert result[2, 0] == 40  

    result = jdf[1:4, ['a', 'c']]
    assert result.shape == (3, 2)
    assert result[0, 0] == 2  
    assert result[2, 1] == 400

def test_jdf_init_with_data():
    """Test creating a JDF with data."""
    data = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    jdf = JDF(data)
    
    assert jdf.shape == (3, 2)
    assert jdf.columns == ["a", "b"]
    assert isinstance(jdf._df, pl.DataFrame)


def test_jdf_head():
    """Test the head method."""
    data = pl.DataFrame({"a": list(range(10))})
    jdf = JDF(data)
    
    result = jdf.head(3)
    assert result.shape == (3, 1)
    assert result[0, 0] == 0
    assert result[2, 0] == 2


def test_jdf_tail():
    """Test the tail method."""
    data = pl.DataFrame({"a": list(range(10))})
    jdf = JDF(data)
    
    result = jdf.tail(3)
    assert result.shape == (3, 1)
    assert result[0, 0] == 7
    assert result[2, 0] == 9

def test_jdf_properties():
    """Test the DataFrame property passthroughs."""
    data = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    jdf = JDF(data)
    
    assert jdf.shape == (3, 2)
    assert jdf.columns == ["a", "b"]
    assert len(jdf.dtypes) == 2

def test_count_classes_single_column():
    """Test count_classes with a single column."""
    data = pl.DataFrame({"a": ["x", "y", "x", "z", "y", "x"]})
    jdf = JDF(data)
    
    result = jdf.count_classes(columns=["a"])
    
    expected = pl.DataFrame({
        "a": ["x", "y", "z"],
        "count": [3, 2, 1],
        "percentage": [50.0, 33.33333333333333, 16.666666666666664]
    })
    
    assert result.shape == (3, 3)
    assert set(result.columns) == {"a", "count", "percentage"}
    assert sorted(result["a"].to_list()) == ["x", "y", "z"]


def test_count_classes_multiple_columns():
    """Test count_classes with multiple columns."""
    data = pl.DataFrame({
        "a": ["x", "x", "x", "x", "x", "x", "x", "y", "y", "z", "z"],
        "b": [1, 1, 3, 4, 5, 7, 1, 2, 2, 3, 4]
    })
    jdf = JDF(data)

    result = jdf.count_classes(columns=["a", "b"])

    assert result.shape == (8, 4)
    assert set(result.columns) >= {"a", "b", "count", "percentage"}


def test_count_classes_missing_column():
    df = JDF(pl.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(KeyError, match="Column(s) not found in DataFrame: ['b']"):
        df["b"]


def test_count_classes_empty_df():
    """Test count_classes on an empty DataFrame."""
    jdf = JDF(pl.DataFrame({"a": pl.Series([], dtype=pl.Utf8)}))

    result = jdf.count_classes(columns=["a"])
    assert result.shape == (0, 3)
    assert set(result.columns) == {"a", "count", "percentage"}
