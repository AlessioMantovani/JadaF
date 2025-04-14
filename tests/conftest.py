import pytest
import polars as pl
from jadaf import JDF


@pytest.fixture
def sample_df():
    """
    Create a sample Polars DataFrame for testing.
    """
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 40, 45],
        "score": [90.5, 85.0, 92.5, 88.0, 95.5]
    })


@pytest.fixture
def sample_jdf(sample_df):
    """
    Create a sample JDF for testing.
    """
    return JDF(sample_df)