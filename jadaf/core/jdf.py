"""
Core DataFrame implementation.
"""

import polars as pl
import pandas as pd
from typing import Optional, List, Dict, Any, Union


class JDF:
    """
    JadafDataFrame (JDF) - A wrapper around Polars DataFrame providing
    enhanced functionality for data analysis.

    Parameters
    ----------
    data : Union[pl.DataFrame, pd.DataFrame, None], default=None
        Either a Polars DataFrame or a Pandas DataFrame. If a Pandas DataFrame
        is provided, it will be converted to Polars automatically. If None,
        an empty Polars DataFrame will be created.

    Attributes
    ----------
    df : pl.DataFrame
        The underlying Polars DataFrame.
    shape : tuple
        The dimensions of the DataFrame.
    columns : List[str]
        List of column names in the DataFrame.
    dtypes : Dict[str, Any]
        Dictionary mapping column names to their data types.
    """

    def __init__(self, data: Optional[Union[pl.DataFrame, pd.DataFrame, None]] = None):
        if isinstance(data, pd.DataFrame):
            data = pl.DataFrame(data)  # Convert pandas to polars
        self._df = data if data is not None else pl.DataFrame()

    @property
    def df(self) -> pl.DataFrame:
        """
        Returns the underlying Polars DataFrame.

        Returns
        -------
        pl.DataFrame
            The Polars DataFrame wrapped by this JDF.
        """
        return self._df

    def __repr__(self) -> str:
        """
        String representation of the JDF object.

        Returns
        -------
        str
            A string representation including the underlying DataFrame.
        """
        return f"JDF(\n{self._df}\n)"

    def __getitem__(self, key):
        """
        Enables indexing and slicing into the underlying DataFrame.

        Supports:
        - jdf["col"] -> Column
        - jdf[row_idx] -> Row(s)
        - jdf[row_slice] -> Sliced JDF
        - jdf[row_idx, col_idx] -> Scalar
        - jdf[row_idx, col_name] or jdf[row_slice, col_name/s] -> Value(s)

        Parameters
        ----------
        key : Any
            Indexing key, can be str, int, slice, or tuple.

        Returns
        -------
        Any
            Indexed result depending on key type.
        """
        if isinstance(key, str):
            return self._df[key]

        if isinstance(key, (int, slice)):
            return JDF(self._df[key])

        if isinstance(key, tuple):
            row_key, col_key = key

            if isinstance(row_key, int) and isinstance(col_key, int):
                return self._df.row(row_key)[col_key]

            if isinstance(row_key, int):
                if isinstance(col_key, str):
                    return self._df[col_key][row_key]
                elif isinstance(col_key, list):
                    return {col: self._df[col][row_key] for col in col_key}

            df_slice = self._df[row_key]
            if isinstance(col_key, (str, list)):
                return JDF(df_slice.select(col_key))

        return self._df[key]

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the DataFrame.

        Returns
        -------
        tuple
            A tuple representing the number of rows and columns.
        """
        return self._df.shape

    @property
    def columns(self) -> List[str]:
        """
        Returns the column names of the DataFrame.

        Returns
        -------
        List[str]
            A list of column names.
        """
        return self._df.columns

    @property
    def dtypes(self) -> Dict[str, Any]:
        """
        Returns the data types of the DataFrame columns.

        Returns
        -------
        Dict[str, Any]
            A dictionary mapping column names to data types.
        """
        return self._df.dtypes

    def head(self, n: int = 5, jdf: bool = False) -> "pl.DataFrame | JDF":
        """
        Return the first `n` rows of the DataFrame.

        Parameters
        ----------
        n : int, default=5
            Number of rows to return.
        jdf : bool, default=False
            If True, return result as a JDF instance. If False, return a Polars DataFrame.

        Returns
        -------
        pl.DataFrame or JDF
            The first `n` rows of the DataFrame.
        """
        result = self._df.head(n)
        return JDF(result) if jdf else result

    def tail(self, n: int = 5, jdf: bool = False) -> "pl.DataFrame | JDF":
        """
        Return the last `n` rows of the DataFrame.

        Parameters
        ----------
        n : int, default=5
            Number of rows to return.
        jdf : bool, default=False
            If True, return result as a JDF instance. If False, return a Polars DataFrame.

        Returns
        -------
        pl.DataFrame or JDF
            The last `n` rows of the DataFrame.
        """
        result = self._df.tail(n)
        return JDF(result) if jdf else result

    def count_classes(self, columns: List[str], jdf: bool = False) -> "pl.DataFrame | JDF":
        """
        Count unique combinations of values in specified columns, along with counts and percentages.

        Parameters
        ----------
        columns : List[str]
            List of column names to group by.
        jdf : bool, default=False
            If True, return the result as a JDF instance. Otherwise, return as a Polars DataFrame.

        Returns
        -------
        pl.DataFrame or JDF
            A DataFrame containing unique value combinations, counts, and percentages.

        Raises
        ------
        ValueError
            If any specified column is not found in the DataFrame or if the grouped structure is invalid.
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
