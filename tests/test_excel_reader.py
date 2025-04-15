import os
import pytest
import tempfile
from pathlib import Path
import pandas as pd

from jadaf import JDF, load_excel


@pytest.fixture
def sample_excel():
    """Create a temporary Excel file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        filepath = f.name
    
    # Create a sample Excel file using pandas
    df = pd.DataFrame({
        "a": [1, 4, 7],
        "b": [2, 5, 8],
        "c": [3, 6, 9]
    })
    df.to_excel(filepath, index=False)
    
    yield filepath
    
    os.unlink(filepath)


def test_load_excel_basic(sample_excel):
    """Test basic Excel loading."""
    print(f"Testing file: {sample_excel}")
    try:
        jdf = load_excel(sample_excel)
        
        assert isinstance(jdf, JDF)
        assert jdf.shape == (3, 3)
        assert jdf.columns == ["a", "b", "c"]
    except Exception as e:
        print(f"Error in test_load_excel_basic: {str(e)}")
        raise


def test_load_excel_with_options(sample_excel):
    """Test Excel loading with options."""
    # Create another Excel with different structure for testing
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        filepath = f.name
    
    df1 = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4, 5, 6]
    })
    df2 = pd.DataFrame({
        "x": ["a", "b", "c"],
        "y": ["d", "e", "f"]
    })
    
    with pd.ExcelWriter(filepath) as writer:
        df1.to_excel(writer, sheet_name="Sheet1", index=False)
        df2.to_excel(writer, sheet_name="Sheet2", index=False)
    
    # Test with specific sheet
    jdf = load_excel(filepath, sheet_name="Sheet2")
    assert isinstance(jdf, JDF)
    assert jdf.shape == (3, 2)
    assert jdf.columns == ["x", "y"]
    
    os.unlink(filepath)


def test_load_excel_file_not_found():
    """Test error handling for non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_excel("nonexistent_file.xlsx")


def test_load_excel_path_object(sample_excel):
    """Test loading Excel with Path object."""
    path = Path(sample_excel)
    try:
        jdf = load_excel(path)
        
        assert isinstance(jdf, JDF)
        assert jdf.shape == (3, 3)
    except Exception as e:
        print(f"Error in test_load_excel_path_object: {str(e)}")
        raise