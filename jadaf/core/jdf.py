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
    
    def __getitem__(self, key):
        """
        Enable indexing and slicing into the underlying DataFrame.

        Supports:
            - jdf["col"] (returns column)
            - jdf[row_idx] (returns row as tuple)
            - jdf[row_slice] (returns sliced JDF)
            - jdf[row_idx, col_idx] (returns scalar)
            - jdf[row_idx, col_name] or jdf[row_slice, col_name/s] (returns value(s))
        """
        # jdf["col"]
        if isinstance(key, str):
            return self._df[key]
        
        # jdf[row_idx] or jdf[row_slice]
        if isinstance(key, (int, slice)):
            return JDF(self._df[key])
        
        # jdf[row_idx, col_name] or jdf[row_slice, col_name]
        if isinstance(key, tuple):
            row_key, col_key = key

            # Single row, single column => scalar
            if isinstance(row_key, int) and isinstance(col_key, int):
                return self._df.row(row_key)[col_key]

            # Single row, column name or list => row subset
            if isinstance(row_key, int):
                if isinstance(col_key, str):
                    return self._df[col_key][row_key]
                elif isinstance(col_key, list):
                    return {col: self._df[col][row_key] for col in col_key}

            # Slice rows + column(s)
            df_slice = self._df[row_key]
            if isinstance(col_key, (str, list)):
                return JDF(df_slice.select(col_key))
        
        # fallback
        return self._df[key]

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
    
    def head(self, n: int = 5, jdf: bool = False) -> "pl.DataFrame | JDF":
        """
        Return the first `n` rows of the DataFrame.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.
            jdf (bool, optional): If True, return the result as a JDF instance.
                                If False (default), return as a Polars DataFrame.

        Returns:
            pl.DataFrame or JDF: The first `n` rows of the DataFrame.
        """
        result = self._df.head(n)
        return JDF(result) if jdf else result
    
    def tail(self, n: int = 5, jdf: bool = False) -> "pl.DataFrame | JDF":
        """
        Return the last `n` rows of the DataFrame.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.
            jdf (bool, optional): If True, return the result as a JDF instance.
                                If False (default), return as a Polars DataFrame.

        Returns:
            pl.DataFrame or JDF: The last `n` rows of the DataFrame.
        """
        result = self._df.tail(n)
        return JDF(result) if jdf else result
    
    def count_classes(self, columns: List[str], jdf: bool = False) -> "pl.DataFrame|JDF":
        """
        Count unique combinations of values in the specified columns, along with their 
        occurrence counts and percentage of total rows.

        Args:
            columns (List[str]): A list of column names to group by.
            jdf (bool, optional): If True, return the result as a JDF instance. 
                                If False (default), return as a Polars DataFrame.

        Returns:
            pl.DataFrame or JDF: A DataFrame containing the unique value combinations 
            in the specified columns, along with their count and percentage of total rows.

        Raises:
            ValueError: If any specified column does not exist in the DataFrame or if 
                        the grouped DataFrame has an unexpected structure.
        """

        missing = [col for col in columns if col not in self._df.columns]
        if missing:
            raise ValueError(f"Column(s) not found in DataFrame: {missing}")
        
        total = self._df.shape[0]

        counts = self._df.group_by(columns).len()
        
        non_group_cols = [col for col in counts.columns if col not in columns]
        if len(non_group_cols) != 1:
            raise ValueError("Unexpected structure in grouped DataFrame.")
        
        count_col = non_group_cols[0]
        counts = counts.rename({count_col: "count"})

        counts = counts.with_columns(
            (pl.col("count") / total * 100).alias("percentage")
        )

        return JDF(counts) if jdf else counts