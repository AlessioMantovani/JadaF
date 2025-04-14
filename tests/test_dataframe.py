import pytest
import polars as pl
from jadaf import JDF


def test_jdf_init_empty():
    """Test creating an empty JDF."""
    jdf = JDF()
    assert jdf.shape == (0, 0)
    assert isinstance(jdf.df, pl.DataFrame)


def test_jdf_init_with_data():
    """Test creating a JDF with data."""
    data = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    jdf = JDF(data)
    
    assert jdf.shape == (3, 2)
    assert jdf.columns == ["a", "b"]
    assert isinstance(jdf.df, pl.DataFrame)


def test_jdf_head():
    """Test the head method."""
    data = pl.DataFrame({"a": list(range(10))})
    jdf = JDF(data)
    
    result = jdf.head(3)
    assert result.shape == (3, 1)
    assert result.df[0, 0] == 0
    assert result.df[2, 0] == 2


def test_jdf_tail():
    """Test the tail method."""
    data = pl.DataFrame({"a": list(range(10))})
    jdf = JDF(data)
    
    result = jdf.tail(3)
    assert result.shape == (3, 1)
    assert result.df[0, 0] == 7
    assert result.df[2, 0] == 9


def test_jdf_properties():
    """Test the DataFrame property passthroughs."""
    data = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    jdf = JDF(data)
    
    assert jdf.shape == (3, 2)
    assert jdf.columns == ["a", "b"]
    assert len(jdf.dtypes) == 2