"""
Core DataFrame implementation.
"""

import polars as pl
from typing import Optional, List, Dict, Any


class JDF:
    """
    JadafDataFrame (JDF) - A wrapper around Polars DataFrame providing
    enhanced functionality for data analysis.
    """
    
    def __init__(self, data: Optional[pl.DataFrame] = None):
        """
        Initialize a new JDF object.
        
        Args:
            data: A Polars DataFrame to wrap. If None, creates an empty DataFrame.
        """
        self._df = data if data is not None else pl.DataFrame()
    
    @property
    def df(self) -> pl.DataFrame:
        """
        Get the underlying Polars DataFrame.
        
        Returns:
            The Polars DataFrame wrapped by this JDF.
        """
        return self._df
    
    def __repr__(self) -> str:
        """
        Return a string representation of the JDF.
        
        Returns:
            String representation including the underlying DataFrame.
        """
        return f"JDF(\n{self._df}\n)"
    
    @property
    def shape(self) -> tuple:
        """Get the dimensions of the DataFrame."""
        return self._df.shape
    
    @property
    def columns(self) -> List[str]:
        """Get the column names of the DataFrame."""
        return self._df.columns
    
    @property
    def dtypes(self) -> Dict[str, Any]:
        """Get the data types of the DataFrame columns."""
        return self._df.dtypes
    
    def head(self, n: int = 5) -> "JDF":
        """
        Return the first n rows of the DataFrame.
        
        Args:
            n: Number of rows to return
            
        Returns:
            A new JDF with the first n rows
        """
        return JDF(self._df.head(n))
    
    def tail(self, n: int = 5) -> "JDF":
        """
        Return the last n rows of the DataFrame.
        
        Args:
            n: Number of rows to return
            
        Returns:
            A new JDF with the last n rows
        """
        return JDF(self._df.tail(n))
