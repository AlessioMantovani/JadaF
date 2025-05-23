"""
Core DataFrame implementation with enhanced slicing.
"""

import polars as pl
import pandas as pd
from typing import Optional, List, Dict, Any, Union, Tuple, Callable
import re


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
        if data is not None and not isinstance(data, (pl.DataFrame, pd.DataFrame)):
            raise TypeError(f"data must be a Polars DataFrame, Pandas DataFrame or None, got {type(data)}")
        if isinstance(data, pd.DataFrame):
            data = pl.DataFrame(data)  # Convert pandas to polars
        self._df = data if data is not None else pl.DataFrame()

        # Cache for column groups to improve performance
        self._column_groups = {}

    @property
    def df(self) -> pl.DataFrame:
        """Returns the underlying Polars DataFrame."""
        return self._df

    @property
    def shape(self) -> Tuple[int, int]:
        """Returns the dimensions of the DataFrame."""
        return len(self._df), len(self._df.columns)

    @property
    def columns(self) -> List[str]:
        """Returns the column names of the DataFrame."""
        return self._df.columns

    @property
    def dtypes(self) -> Dict[str, Any]:
        """Returns the data types of the DataFrame columns."""
        return {col: dtype for col, dtype in zip(self._df.columns, self._df.dtypes)}

    def __repr__(self) -> str:
        """String representation of the JDF object."""
        return self._df.__repr__()

    def __getitem__(self, key):
        """
        Enables indexing and slicing into the underlying DataFrame.

        Supports:
        - jdf["col"] -> Column
        - jdf[row_idx] -> Row(s)
        - jdf[row_slice] -> Sliced JDF
        - jdf[row_idx, col_idx] -> Scalar
        - jdf[row_idx, col_name] or jdf[row_slice, col_name/s] -> Value(s)
        - jdf[row_slice, :] -> Rows slice with all columns
        - jdf[row_slice, col_slice] -> Rows and columns slice by position

        Advanced patterns:
        - jdf["col1, col2, col3"] -> Multiple columns by comma-separated string
        - jdf[["col1", "col2"]] -> Multiple columns by list
        - jdf["col*"] -> Pattern matching for column names
        - jdf[{"group": ["col1", "col2"]}] -> Column groups
        - jdf[lambda df: df["col"] > 10] -> Boolean mask filtering
        - jdf[:, lambda cols: [c for c in cols if "date" in c.lower()]] -> Column filtering function
        - jdf["col1":"col5"] -> Column range slicing by name
        """
        # Single string column or pattern
        if isinstance(key, str):
            # Using the helper methods to reduce redundancy
            if "," in key:
                cols = [col.strip() for col in key.split(",")]
                self._check_columns_exist(cols)
                return JDF(self._df.select(cols))

            # Check for wildcard patterns
            if any(char in key for char in "*?["):
                matching_cols = self._parse_column_pattern(key)
                return JDF(self._df.select(matching_cols))

            # Standard single column access
            if key not in self._df.columns:
                raise KeyError(f"Column '{key}' not found in DataFrame")
            return self._df[key]

        # Multiple columns as list
        if isinstance(key, list):
            if all(isinstance(k, str) for k in key):
                self._check_columns_exist(key)
                return JDF(self._df.select(key))
            elif all(isinstance(k, bool) for k in key):
                # Boolean mask for rows
                if len(key) != len(self._df):
                    raise ValueError(f"Boolean mask length {len(key)} does not match DataFrame length {len(self._df)}")
                mask = pl.Series(key)
                return JDF(self._df.filter(mask))

        # Column groups
        if isinstance(key, dict):
            # Format: {group_name: column_list}
            result = {}
            for group_name, columns in key.items():
                self._check_columns_exist(columns)
                result[group_name] = JDF(self._df.select(columns))
            return result

        # Lambda function for row filtering
        if callable(key):
            result = key(self)
            if isinstance(result, pl.Series) and result.dtype == pl.Boolean:
                return JDF(self._df.filter(result))
            raise TypeError(f"Lambda function must return a boolean Series, got {type(result)}")

        # Column range by name using slice
        if isinstance(key, slice) and isinstance(key.start, str) and isinstance(key.stop, str):
            selected_cols = self._parse_column_selector(key)
            return JDF(self._df.select(selected_cols))

        # Row selection by index or slice
        if isinstance(key, (int, slice)):
            # Validate int index range for rows
            if isinstance(key, int):
                if not (-len(self._df) <= key < len(self._df)):
                    raise IndexError(f"Row index {key} out of range")
            return JDF(self._df[key])

        # Tuple based access (rows, columns)
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Indexing tuple must be length 2")

            row_key, col_key = key

            # Process the row and column selectors using helper methods
            row_selector = self._parse_row_selector(row_key)

            # Special handling for int row key and different column key types
            if isinstance(row_key, int):
                if not (-len(self._df) <= row_key < len(self._df)):
                    raise IndexError(f"Row index {row_key} out of range")

                # Handle different column key types
                if isinstance(col_key, int):
                    # Get a single value at (row, col) position
                    if not (-len(self._df.columns) <= col_key < len(self._df.columns)):
                        raise IndexError(f"Column index {col_key} out of range")
                    row = self._df.row(row_key)
                    return row[col_key]

                # Get column values for a single row
                try:
                    cols = self._parse_column_selector(col_key)
                    if len(cols) == 1:
                        return self._df[cols[0]][row_key]
                    else:
                        return {col: self._df[col][row_key] for col in cols}
                except Exception as e:
                    raise TypeError(f"Invalid column key: {e}")

            # For non-integer row_key, handle the different types of col_key
            # First get the subset of rows
            df_slice = self._df[row_selector]

            # If col_key is None or slice(None), return all columns
            if col_key is None or col_key == slice(None):
                return JDF(df_slice)

            # Parse column selector and return selected columns
            try:
                cols = self._parse_column_selector(col_key)
                return JDF(df_slice.select(cols))
            except Exception as e:
                raise TypeError(f"Invalid column key: {e}")

        raise TypeError(f"Unsupported indexing key type: {type(key)}")

    def __getattr__(self, name: str):
        """
        Enable access to column groups as attributes.

        Examples
        --------
        >>> jdf.group_columns("dates", ["start_date", "end_date"])
        >>> jdf.dates  # Returns JDF with only date columns
        """
        if name in self._column_groups:
            return JDF(self._df.select(self._column_groups[name]))
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def _validate_columns(self, columns: List[str]) -> List[str]:
        """
        Validate that columns exist in the DataFrame.

        Returns the list of any missing columns. Empty list if all columns exist.
        """
        return [col for col in columns if col not in self._df.columns]

    def _check_columns_exist(self, columns: List[str], raise_error: bool = True) -> bool:
        """
        Check if all columns exist in the DataFrame.

        Parameters
        ----------
        columns : List[str]
            List of column names to check
        raise_error : bool, default=True
            Whether to raise a KeyError if columns are missing

        Returns
        -------
        bool
            True if all columns exist, False otherwise

        Raises
        ------
        KeyError
            If columns are missing and raise_error is True
        """
        missing_cols = self._validate_columns(columns)
        if missing_cols:
            if raise_error:
                raise KeyError(f"Column(s) not found in DataFrame: {missing_cols}")
            return False
        return True

    def _parse_column_pattern(self, pattern: str) -> List[str]:
        """
        Get columns matching a wildcard pattern.

        Parameters
        ----------
        pattern : str
            Pattern to match column names (supports * and ? wildcards)

        Returns
        -------
        List[str]
            List of matching column names

        Raises
        ------
        KeyError
            If no columns match the pattern
        """
        # Convert glob pattern to regex
        regex_pattern = pattern.replace("*", ".*").replace("?", ".").replace("[", r"\[").replace("]", r"\]")
        regex = re.compile(f"^{regex_pattern}$")

        # Find matching columns
        matching_cols = [col for col in self._df.columns if regex.match(col)]

        if not matching_cols:
            raise KeyError(f"No columns match pattern '{pattern}'")

        return matching_cols

    def _parse_column_selector(self, selector: Union[str, List[str], slice, Callable]) -> List[str]:
        """
        Parse different types of column selectors into a list of column names.

        Parameters
        ----------
        selector : Union[str, List[str], slice, Callable]
            Column selector in various formats

        Returns
        -------
        List[str]
            List of column names
        """
        if selector is None:
            return self._df.columns

        if isinstance(selector, str):
            # Check for pattern matching
            if any(char in selector for char in "*?["):
                return self._parse_column_pattern(selector)

            # Check for comma-separated list
            if "," in selector:
                columns = [col.strip() for col in selector.split(",")]
                self._check_columns_exist(columns)
                return columns

            # Single column
            if selector not in self._df.columns:
                raise KeyError(f"Column '{selector}' not found in DataFrame")
            return [selector]

        if isinstance(selector, list):
            if not all(isinstance(col, str) for col in selector):
                raise TypeError("Column list must contain only strings")
            self._check_columns_exist(selector)
            return selector

        if isinstance(selector, slice):
            if selector == slice(None):
                return self._df.columns

            # Column range by name
            if isinstance(selector.start, str) and isinstance(selector.stop, str):
                all_cols = self._df.columns

                if selector.start not in all_cols:
                    raise KeyError(f"Start column '{selector.start}' not found in DataFrame")
                if selector.stop not in all_cols:
                    raise KeyError(f"Stop column '{selector.stop}' not found in DataFrame")

                start_idx = all_cols.index(selector.start)
                stop_idx = all_cols.index(selector.stop)

                if start_idx > stop_idx:
                    raise ValueError(f"Start column '{selector.start}' comes after stop column '{selector.stop}'")

                return all_cols[start_idx:stop_idx+1:selector.step]

            # Position-based column slicing
            return self._df.columns[selector]

        if callable(selector):
            selected_cols = selector(self._df.columns)
            if not isinstance(selected_cols, (list, tuple)) or not all(isinstance(col, str) for col in selected_cols):
                raise TypeError("Column function must return a list of column names")
            self._check_columns_exist(selected_cols)
            return selected_cols

        raise TypeError(f"Unsupported column selector type: {type(selector)}")

    def _parse_row_selector(self, selector: Any) -> Union[int, slice, List[int], pl.Series]:
        """
        Parse different types of row selectors.

        Parameters
        ----------
        selector : Any
            Row selector in various formats

        Returns
        -------
        Union[int, slice, List[int], pl.Series]
            Processed row selector
        """
        if selector is None:
            return slice(None)

        if callable(selector):
            result = selector(self)
            if isinstance(result, pl.Series) and result.dtype == pl.Boolean:
                return result
            raise TypeError(f"Row function must return a boolean Series, got {type(result)}")

        if isinstance(selector, pl.Series):
            if selector.dtype == pl.Boolean:
                return selector
            raise TypeError("Series selector must be boolean type")

        if isinstance(selector, (int, slice, list)):
            return selector

        raise TypeError(f"Unsupported row selector type: {type(selector)}")

    def group_columns(self, group_name: str, columns: List[str]):
        """
        Create a named group of columns for easier access.

        Parameters
        ----------
        group_name : str
            Name of the column group
        columns : List[str]
            List of column names to include in the group

        Examples
        --------
        >>> jdf.group_columns("dates", ["start_date", "end_date", "created_at"])
        >>> jdf.dates  # Access all date columns as a new JDF
        """
        self._check_columns_exist(columns)
        self._column_groups[group_name] = columns

    def loc(self, row_selector=None, col_selector=None):
        """
        Label-based indexing.

        Parameters
        ----------
        row_selector : Various types
            Can be:
            - Boolean Series for filtering rows
            - Lambda function returning boolean Series
            - List of row indices
            - Slice of indices
        col_selector : Various types
            Can be:
            - String or list of column names
            - Slice with column names
            - Lambda function selecting columns
            - Pattern string with wildcards

        Returns
        -------
        JDF
            New JDF with selected rows and columns

        Examples
        --------
        >>> jdf.loc(lambda df: df["age"] > 30, ["name", "age"])
        >>> jdf.loc(jdf["age"] > 30, "name*")
        """
        # Process row selector
        row_processed = self._parse_row_selector(row_selector)

        # Get row subset
        if isinstance(row_processed, pl.Series):
            row_selected_df = self._df.filter(row_processed)
        else:
            row_selected_df = self._df[row_processed]

        # Process column selector
        if col_selector is None:
            return JDF(row_selected_df)

        # Parse column selector and return selected columns
        cols = self._parse_column_selector(col_selector)
        return JDF(row_selected_df.select(cols))

    def columns_like(self, pattern: str):
        """
        Select columns matching a pattern.

        Parameters
        ----------
        pattern : str
            Pattern to match column names (supports * and ? wildcards)

        Returns
        -------
        JDF
            JDF with only the matching columns

        Examples
        --------
        >>> jdf.columns_like("date_*")  # All columns starting with "date_"
        >>> jdf.columns_like("*_id")    # All columns ending with "_id"
        """
        matching_cols = self._parse_column_pattern(pattern)
        return JDF(self._df.select(matching_cols))

    def ix(self, row_indices=None, col_indices=None):
        """
        Position-based indexing.

        Parameters
        ----------
        row_indices : int, list, or slice
            Row positions to select
        col_indices : int, list, or slice
            Column positions to select

        Returns
        -------
        JDF or scalar
            Selected data based on positions

        Examples
        --------
        >>> jdf.ix(0, 0)  # First row, first column value
        >>> jdf.ix(slice(0, 5), [0, 2, 4])  # First 5 rows, columns at positions 0, 2, and 4
        """
        # Handle row selection
        if row_indices is None:
            row_selected_df = self._df
        elif isinstance(row_indices, (int, list, slice)):
            row_selected_df = self._df[row_indices]
        else:
            raise TypeError(f"Row indices must be int, list, or slice, got {type(row_indices)}")

        # Handle column selection
        if col_indices is None:
            return JDF(row_selected_df)
        elif isinstance(col_indices, int):
            if isinstance(row_indices, int):
                # Single value
                return row_selected_df[self._df.columns[col_indices]][0]
            else:
                # Single column by position
                return JDF(row_selected_df.select([self._df.columns[col_indices]]))
        elif isinstance(col_indices, (list, slice)):
            # Multiple columns by position
            if isinstance(col_indices, list) and all(isinstance(i, int) for i in col_indices):
                selected_cols = [self._df.columns[i] for i in col_indices]
            else:  # Slice
                selected_cols = self._df.columns[col_indices]
            return JDF(row_selected_df.select(selected_cols))
        else:
            raise TypeError(f"Column indices must be int, list, or slice, got {type(col_indices)}")

    def with_filter(self, condition):
        """
        Create a temporary filtered view of the DataFrame.

        Parameters
        ----------
        condition : Callable or pl.Series
            Filter condition

        Returns
        -------
        JDF
            Filtered JDF

        Examples
        --------
        >>> active_users = jdf.with_filter(lambda df: df["status"] == "active")
        >>> recent_active = active_users.with_filter(lambda df: df["last_login"] > "2023-01-01")
        """
        # Reuse the row selector parsing
        row_selector = self._parse_row_selector(condition)
        if isinstance(row_selector, pl.Series):
            return JDF(self._df.filter(row_selector))
        raise TypeError(f"Condition must be a callable or boolean Series")

    def head(self, n: int = 5, jdf: bool = False) -> "pl.DataFrame | JDF":
        """
        Return the first `n` rows of the DataFrame.
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"n must be a non-negative integer, got {n}")
        result = self._df.head(n)
        return JDF(result) if jdf else result

    def tail(self, n: int = 5, jdf: bool = False) -> "pl.DataFrame | JDF":
        """
        Return the last `n` rows of the DataFrame.
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"n must be a non-negative integer, got {n}")
        result = self._df.tail(n)
        return JDF(result) if jdf else result

    def count_classes(self, columns: List[str], jdf: bool = False) -> "pl.DataFrame | JDF":
        """
        Count unique combinations of values in specified columns, along with counts and percentages.
        """
        if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
            raise TypeError("columns must be a list of strings")
        self._check_columns_exist(columns)

        total = self._df.height
        counts = self._df.groupby(columns).count()

        non_group_cols = [col for col in counts.columns if col not in columns]
        if len(non_group_cols) != 1:
            raise ValueError("Unexpected structure in grouped DataFrame.")

        count_col = non_group_cols[0]
        counts = counts.rename({count_col: "count"})

        counts = counts.with_columns(
            (pl.col("count") / total * 100).alias("percentage")
        )

        return JDF(counts) if jdf else counts

    def missing(self) -> "JDF":
        """
        Return information about missing values in each column.
        """
        total_count = len(self._df)
        result = pl.DataFrame({
            'column': self._df.columns,
            'missing_count': [self._df[col].null_count() for col in self._df.columns],
            'total_count': [total_count for _ in self._df.columns]
        }).with_columns([
            (pl.col('missing_count') / pl.col('total_count') * 100).alias('missing_percentage')
        ]).select(['column', 'missing_count', 'missing_percentage'])

        return JDF(result)

    def profile(self, subset: Optional[List[str]] = None) -> "JDF":
        """
        Generate a comprehensive profiling report of the DataFrame.
        """
        if subset is not None:
            if not isinstance(subset, list) or not all(isinstance(col, str) for col in subset):
                raise TypeError("subset must be a list of strings or None")
            self._check_columns_exist(subset)

        total_rows = len(self._df)
        memory_usage_mb = self._df.estimated_size() / (1024 * 1024)
        print(f"DataFrame shape: {self.shape}")
        print(f"Estimated memory usage: {memory_usage_mb:.2f} MB\n")

        cols = self._df.columns if subset is None else subset
        stats_dict = {}

        for col in cols:
            col_series = self._df[col]
            dtype = str(col_series.dtype)
            missing_count = col_series.null_count()
            missing_percentage = (missing_count / total_rows) * 100 if total_rows > 0 else 0

            col_stats = {
                "dtype": dtype,
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
            }

            if col_series.dtype in [pl.Int64, pl.Float64]:
                col_stats.update({
                    "min": col_series.min(),
                    "max": col_series.max(),
                    "mean": col_series.mean(),
                    "median": col_series.median(),
                    "std": col_series.std(),
                    "skewness": col_series.skew(),
                })
            else:
                unique_count = col_series.n_unique()
                col_stats["unique_count"] = unique_count
                if unique_count < 20:
                    vc = col_series.value_counts().to_dicts()
                    col_stats["value_counts"] = str(vc)

            stats_dict[col] = col_stats

        # Create row-wise statistics
        transposed = {}
        for col, stat_map in stats_dict.items():
            for stat, value in stat_map.items():
                transposed.setdefault(stat, {})[col] = value

        rows = []
        for stat, values in transposed.items():
            row = {"statistic": stat}
            row.update(values)
            rows.append(row)

        df = pl.DataFrame(rows)
        return JDF(df)

    def to_dict(self, orient: str = "records") -> list[dict[str, Any]] | None:
        """
        Convert DataFrame to dictionary.
        """
        allowed_orients = {"records"}
        if orient not in allowed_orients:
            raise ValueError(f"Unsupported orient '{orient}'. Supported: {allowed_orients}")

        if orient == "records":
            return self._df.to_dicts()
        return None

    def to_json(self, path: Optional[str] = None, orient: str = "records") -> Optional[str]:
        """
        Convert DataFrame to JSON. You can use it to store you dataframe as json (by default it only returns the json string)
        """
        import json

        allowed_orients = {"records"}
        if orient not in allowed_orients:
            raise ValueError(f"Unsupported orient '{orient}'. Supported: {allowed_orients}")

        data = self.to_dict(orient=orient)
        json_str = json.dumps(data, indent=4)

        if path is not None:
            if not isinstance(path, str):
                raise TypeError("path must be a string or None")
            with open(path, 'w') as f:
                f.write(json_str)
            return None

        return json_str

    def to_csv(self, path: Optional[str] = None, sep: str = ",", header: bool = True, index: bool = False) -> Optional[
        str]:
        """
        Convert DataFrame to CSV format via pandas.

        Parameters
        ----------
        path : Optional[str], default=None
            If specified, writes the CSV content to the given file path.
            If None, returns the CSV string.

        sep : str, default=','
            Delimiter to use in the CSV output.

        header : bool, default=True
            Whether to write out the column names.

        index : bool, default=False
            Whether to write row indices (not applicable for Polars, but passed to pandas).

        Returns
        -------
        Optional[str]
            CSV string if `path` is None, else None.
        """
        # Convert Polars DataFrame to pandas DataFrame
        pdf = self._df.to_pandas()

        if path is not None:
            if not isinstance(path, str):
                raise TypeError("path must be a string or None")
            pdf.to_csv(path, sep=sep, header=header, index=index, encoding="utf-8")
            return None

        # Return CSV as string
        return pdf.to_csv(sep=sep, header=header, index=index, encoding="utf-8")