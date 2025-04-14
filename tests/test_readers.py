import pytest
import tempfile
import os
from pathlib import Path

from jadaf import JDF, load_csv


@pytest.fixture
def sample_csv():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w+") as f:
        f.write("a,b,c\n")
        f.write("1,2,3\n")
        f.write("4,5,6\n")
        f.write("7,8,9\n")
        filepath = f.name
    
    yield filepath
    
    os.unlink(filepath)


def test_load_csv_basic(sample_csv):
    """Test basic CSV loading."""
    jdf = load_csv(sample_csv)
    
    assert isinstance(jdf, JDF)
    assert jdf.shape == (3, 3)
    assert jdf.columns == ["a", "b", "c"]


def test_load_csv_with_options(sample_csv):
    """Test CSV loading with options."""
    jdf = load_csv(sample_csv, has_header=False, new_columns=["x", "y", "z"])
    
    assert isinstance(jdf, JDF)
    assert jdf.shape == (4, 3) 
    assert jdf.columns == ["x", "y", "z"]


def test_load_csv_file_not_found():
    """Test error handling for non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_csv("nonexistent_file.csv")


def test_load_csv_path_object(sample_csv):
    """Test loading CSV with Path object."""
    path = Path(sample_csv)
    jdf = load_csv(path)
    
    assert isinstance(jdf, JDF)
    assert jdf.shape == (3, 3)