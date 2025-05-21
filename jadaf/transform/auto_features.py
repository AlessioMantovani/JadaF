from typing import Dict, List, Optional, Type, Callable
import warnings
import polars as pl
import jadaf as jd


class FeatureGenerator:
    """
    Class for generating new features from existing numeric columns through pairwise operations.
    """
    def __init__(self, df: jd.JDF, columns_to_include: List[str] = None, columns_to_exclude: List[str] = None):
        self._validate_column_selection(columns_to_include, columns_to_exclude)

        self.numeric_dtypes = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64
        }

        self.df = self._filter_columns(df, columns_to_include, columns_to_exclude)
        self.original_numeric_columns = self._get_numeric_columns()
        self._operations = {}
        # Map to define whether operations are commutative
        self._commutative_operations = {
            "sum": True,
            "product": True,
            "max": True,
            "min": True,
            "diff": True,
            "ratio": False,
            "logdiff": False
        }

        self._register_standard_operations()

    def _validate_column_selection(self, include: Optional[List[str]], exclude: Optional[List[str]]) -> None:
        """Validate column selection parameters."""
        if include and exclude:
            raise ValueError("Cannot specify both columns_to_include and columns_to_exclude")
        if not include and not exclude:
            warnings.warn(
                "No columns specified for feature generation, all columns will be included.",
                category=UserWarning
            )

    def _filter_columns(self, df: jd.JDF, include: Optional[List[str]], exclude: Optional[List[str]]) -> jd.JDF:
        """Filter columns based on inclusion/exclusion criteria."""
        if exclude:
            return df.drop(columns=exclude)
        if include:
            return df[include]
        return df

    def _get_numeric_columns(self) -> List[str]:
        """Get list of numeric columns in the dataframe."""
        return [col for col in self.df.columns if self.df[col].dtype in self.numeric_dtypes]

    def _register_operation(self, name: str, operation_func: Callable[[pl.Series, pl.Series], pl.Series]) -> None:
        """Register a pairwise operation."""
        self._operations[name] = operation_func

        # Define the method dynamically
        method_name = f"auto_{name}"

        def method(self=self):
            return self._generate_pairwise_features(name)

        method.__doc__ = f"Generate pairwise {name} features between all numeric columns."
        setattr(self.__class__, method_name, method)

    def _register_standard_operations(self) -> None:
        """Register standard pairwise operations."""
        # Define and register standard operations
        operations = {
            "diff": self._pairwise_diff,
            "ratio": self._pairwise_ratio,
            "product": self._pairwise_product,
            "sum": self._pairwise_sum,
            "logdiff": self._pairwise_logdiff,
            "max": self._pairwise_max,
            "min": self._pairwise_min
        }

        for name, func in operations.items():
            self._register_operation(name, func)

    # Define operation functions
    def _pairwise_diff(self, col1: pl.Series, col2: pl.Series) -> pl.Series:
        return col1 - col2

    def _pairwise_ratio(self, col1: pl.Series, col2: pl.Series) -> pl.Series:
        return col1 / (col2 + 1e-10)  # Add epsilon to avoid division by zero

    def _pairwise_product(self, col1: pl.Series, col2: pl.Series) -> pl.Series:
        return col1 * col2

    def _pairwise_sum(self, col1: pl.Series, col2: pl.Series) -> pl.Series:
        return col1 + col2

    def _pairwise_logdiff(self, col1: pl.Series, col2: pl.Series) -> pl.Series:
        return pl.log(pl.abs(col1) + 1e-10) - pl.log(pl.abs(col2) + 1e-10)

    def _pairwise_max(self, col1: pl.Series, col2: pl.Series) -> pl.Series:
        return pl.max_horizontal(col1, col2)

    def _pairwise_min(self, col1: pl.Series, col2: pl.Series) -> pl.Series:
        return pl.min_horizontal(col1, col2)

    def _generate_pairwise_features(self, operation_name: str) -> jd.JDF:
        """
        Generate pairwise features using the specified operation.

        Args:
            operation_name: Name of the operation to perform

        Returns:
            Updated JDF with new features
        """
        if operation_name not in self._operations:
            raise ValueError(f"Operation '{operation_name}' not registered")

        numeric_cols = self._get_numeric_columns()
        operation_func = self._operations[operation_name]
        is_commutative = self._commutative_operations.get(operation_name, False)
        new_cols = []

        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if col1 == col2:
                    continue

                if is_commutative and j <= i:
                    continue  # Skip redundant pairs for commutative ops

                new_col_name = f"{col1}_{operation_name}_{col2}"
                if new_col_name in self.df.columns:
                    continue

                try:
                    new_col = operation_func(self.df.df[col1], self.df.df[col2])
                    new_cols.append(new_col.alias(new_col_name))
                except Exception as e:
                    warnings.warn(
                        f"Error generating {operation_name} feature for {col1} and {col2}: {str(e)}",
                        category=RuntimeWarning
                    )

        updated_df = self.df.df.with_columns(new_cols)
        self.df = jd.JDF(updated_df)
        return self.df

    def register_new_operation(self, name: str, operation_func: Callable[[pl.Series, pl.Series], pl.Series]) -> None:
        """
        Register a new pairwise operation.

        Args:
            name: Name of the operation
            operation_func: Function that takes two Series and returns a Series
        """
        if name in self._operations:
            warnings.warn(
                f"Operation '{name}' already exists and will be overwritten",
                category=UserWarning
            )
        self._register_operation(name, operation_func)

    def auto_operation(self, operation_name: str) -> jd.JDF:
        """
        Generate pairwise features using the specified operation.

        Args:
            operation_name: Name of the operation to perform

        Returns:
            Updated JDF with new features
        """
        return self._generate_pairwise_features(operation_name)