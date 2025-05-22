from typing import Union, Optional, Dict, Any
from numpy.typing import NDArray
import joblib
from enum import Enum
import polars as pl
import numpy as np
from pathlib import Path
import jadaf as jd


class ScalerType(str, Enum):
    """
    Enumeration of supported scaler types.

    Values
    ------
    STANDARD : str
        StandardScaler represented by 's'.
    ROBUST : str
        RobustScaler represented by 'r'.
    MINMAX : str
        MinMaxScaler represented by 'm'.
    NORMALIZER : str
        Normalizer represented by 'n'.
    QUANTILE : str
        QuantileTransformer represented by 'q'.
    POWER : str
        PowerTransformer represented by 'p'.
    """
    STANDARD = "s"
    ROBUST = "r"
    MINMAX = "m"
    NORMALIZER = "n"
    QUANTILE = "q"
    POWER = "p"


class Scaler:
    """
    Wrapper class for common sklearn scalers with enhanced functionality.

    Supports StandardScaler, RobustScaler, MinMaxScaler, Normalizer,
    QuantileTransformer, and PowerTransformer.

    Parameters
    ----------
    type_ : str or ScalerType, optional
        Type of scaler to use:
            - 's' or ScalerType.STANDARD for StandardScaler
            - 'r' or ScalerType.ROBUST for RobustScaler
            - 'm' or ScalerType.MINMAX for MinMaxScaler
            - 'n' or ScalerType.NORMALIZER for Normalizer
            - 'q' or ScalerType.QUANTILE for QuantileTransformer
            - 'p' or ScalerType.POWER for PowerTransformer
        Default is 's' (StandardScaler).
    **kwargs
        Additional keyword arguments to pass to the scaler constructor.

    Attributes
    ----------
    type : ScalerType
        The type of scaler being used.
    scaler : sklearn scaler object
        The underlying sklearn scaler instance.
    is_fitted : bool
        Whether the scaler has been fitted to data.
    feature_names_ : list
        Names of features seen during fit.
    n_features_in_ : int
        Number of features seen during fit.

    Raises
    ------
    ValueError
        If an unsupported scaler type is provided.
    """

    def __init__(self, type_: Union[str, ScalerType] = "s", **kwargs):
        if isinstance(type_, str):
            try:
                type_ = ScalerType(type_)
            except ValueError:
                raise ValueError(f"Unsupported scaler type: {type_}. "
                                 f"Supported types: {[t.value for t in ScalerType]}")

        self.type = type_
        self.scaler = self._init_scaler(**kwargs)
        self._numeric_cols = None
        self._original_columns = None
        self.is_fitted = False
        self.feature_names_ = None
        self.n_features_in_ = None

    def _init_scaler(self, **kwargs):
        """
        Initialize the appropriate scaler based on the provided type.

        Parameters
        ----------
        **kwargs
            Arguments passed to the scaler constructor.

        Returns
        -------
        scaler : sklearn scaler object
            An instance of the selected scaler.

        Raises
        ------
        ValueError
            If the scaler type is not supported.
        ImportError
            If sklearn is not available.
        """
        try:
            if self.type == ScalerType.STANDARD:
                from sklearn.preprocessing import StandardScaler
                return StandardScaler(**kwargs)
            elif self.type == ScalerType.ROBUST:
                from sklearn.preprocessing import RobustScaler
                return RobustScaler(**kwargs)
            elif self.type == ScalerType.MINMAX:
                from sklearn.preprocessing import MinMaxScaler
                return MinMaxScaler(**kwargs)
            elif self.type == ScalerType.NORMALIZER:
                from sklearn.preprocessing import Normalizer
                return Normalizer(**kwargs)
            elif self.type == ScalerType.QUANTILE:
                from sklearn.preprocessing import QuantileTransformer
                return QuantileTransformer(**kwargs)
            elif self.type == ScalerType.POWER:
                from sklearn.preprocessing import PowerTransformer
                return PowerTransformer(**kwargs)
            else:
                raise ValueError(f"Unsupported scaler type: {self.type}")
        except ImportError:
            raise ImportError("scikit-learn is required for scaling functionality. "
                              "Install with: pip install scikit-learn")

    def fit(self, X: jd.JDF, columns: Optional[list] = None) -> "Scaler":
        """
        Fit the scaler to the data.

        Parameters
        ----------
        X : jd.JDF
            Input data to fit the scaler.
        columns : list, optional
            Specific columns to fit the scaler on. If None, all numeric columns are used.

        Returns
        -------
        self : Scaler
            Returns the Scaler instance to allow method chaining.

        Raises
        ------
        ValueError
            If no numeric columns are found or specified columns don't exist.
        """
        if not isinstance(X, jd.JDF):
            raise TypeError("X must be a JDF instance")

        # Store original column order
        self._original_columns = X.columns.copy()

        if columns is not None:
            # Check if specified columns exist
            X._check_columns_exist(columns)
            # Ensure specified columns are numeric
            numeric_df = X[columns].select_dtypes(include=["number"])
            if len(numeric_df.columns) == 0:
                raise ValueError("No numeric columns found in specified columns")
            self._numeric_cols = numeric_df.columns
        else:
            # Use all numeric columns
            numeric_df = X.select_dtypes(include=["number"])
            if len(numeric_df.columns) == 0:
                raise ValueError("No numeric columns found in the DataFrame")
            self._numeric_cols = numeric_df.columns

        # Convert to numpy array for sklearn
        data_array = numeric_df.df.to_numpy()

        # Fit the scaler
        self.scaler.fit(data_array)

        # Set fitted attributes
        self.is_fitted = True
        self.feature_names_ = self._numeric_cols.copy()
        self.n_features_in_ = len(self._numeric_cols)

        return self

    def transform(self, X: jd.JDF) -> jd.JDF:
        """
        Transform the data using the fitted scaler.

        Parameters
        ----------
        X : jd.JDF
            Input data to transform.

        Returns
        -------
        jd.JDF
            Transformed data, with numeric columns scaled and non-numeric columns unchanged.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted yet.
        ValueError
            If the input data doesn't have the expected columns.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")

        if not isinstance(X, jd.JDF):
            raise TypeError("X must be a JDF instance")

        # Check if numeric columns exist
        missing_cols = [col for col in self._numeric_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing columns that were present during fit: {missing_cols}")

        # Extract numeric data
        X_numeric = X[self._numeric_cols]
        data_array = X_numeric.df.to_numpy()

        # Transform the data
        transformed_array = self.scaler.transform(data_array)

        # Create transformed DataFrame
        transformed_df = pl.DataFrame(
            transformed_array,
            schema=self._numeric_cols
        )

        # Handle non-numeric columns
        non_numeric_cols = [col for col in X.columns if col not in self._numeric_cols]
        if non_numeric_cols:
            non_numeric_df = X[non_numeric_cols].df
            # Concatenate transformed numeric and original non-numeric columns
            result_df = pl.concat([transformed_df, non_numeric_df], how="horizontal")
        else:
            result_df = transformed_df

        # Maintain original column order
        result_df = result_df.select([col for col in self._original_columns if col in result_df.columns])

        return jd.JDF(result_df)

    def fit_transform(self, X: jd.JDF, columns: Optional[list] = None) -> jd.JDF:
        """
        Fit the scaler to the data and transform it in one step.

        Parameters
        ----------
        X : jd.JDF
            Input data to fit and transform.
        columns : list, optional
            Specific columns to fit the scaler on. If None, all numeric columns are used.

        Returns
        -------
        jd.JDF
            Scaled data, with numeric columns scaled and non-numeric columns unchanged.
        """
        return self.fit(X, columns).transform(X)

    def inverse_transform(self, X: jd.JDF) -> jd.JDF:
        """
        Undo the scaling transformation.

        Parameters
        ----------
        X : jd.JDF
            Scaled data to inverse transform.

        Returns
        -------
        jd.JDF
            Data transformed back to original scale, numeric columns restored
            and non-numeric columns unchanged.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted yet.
        ValueError
            If the scaler doesn't support inverse transformation or
            input data doesn't have expected columns.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")

        if not isinstance(X, jd.JDF):
            raise TypeError("X must be a JDF instance")

        # Check if scaler supports inverse transformation
        if not hasattr(self.scaler, 'inverse_transform'):
            raise ValueError(f"{self.type.value} scaler does not support inverse transformation")

        # Check if numeric columns exist
        missing_cols = [col for col in self._numeric_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing columns that were present during fit: {missing_cols}")

        # Extract numeric data
        X_numeric = X[self._numeric_cols]
        data_array = X_numeric.df.to_numpy()

        # Inverse transform the data
        inverse_array = self.scaler.inverse_transform(data_array)

        # Create inverse transformed DataFrame
        inverse_df = pl.DataFrame(
            inverse_array,
            schema=self._numeric_cols
        )

        # Handle non-numeric columns
        non_numeric_cols = [col for col in X.columns if col not in self._numeric_cols]
        if non_numeric_cols:
            non_numeric_df = X[non_numeric_cols].df
            # Concatenate inverse transformed numeric and original non-numeric columns
            result_df = pl.concat([inverse_df, non_numeric_df], how="horizontal")
        else:
            result_df = inverse_df

        # Maintain original column order
        if self._original_columns:
            result_df = result_df.select([col for col in self._original_columns if col in result_df.columns])

        return jd.JDF(result_df)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters of the underlying scaler.

        Parameters
        ----------
        deep : bool, optional
            Whether to return parameters of nested estimators (default is True).

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        return self.scaler.get_params(deep=deep)

    def set_params(self, **params) -> "Scaler":
        """
        Set parameters for the underlying scaler.

        Parameters
        ----------
        **params
            Parameters to set on the scaler.

        Returns
        -------
        self : Scaler
            Returns the Scaler instance to allow method chaining.

        Raises
        ------
        RuntimeError
            If trying to set parameters after the scaler has been fitted.
        """
        if self.is_fitted:
            raise RuntimeError("Cannot change parameters after scaler has been fitted. "
                               "Create a new scaler instance instead.")

        self.scaler.set_params(**params)
        return self

    def get_feature_names_out(self) -> list:
        """
        Get output feature names for transformation.

        Returns
        -------
        list
            Feature names for the transformed output.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")

        return self.feature_names_.copy()

    def save_scaler(self, path: Union[str, Path] = "./", name: str = "scaler") -> None:
        """
        Save the scaler object to the specified path using joblib.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Directory path or full file path where the scaler will be saved.
            If a directory is provided, the scaler is saved as '<name>.joblib' inside it.
            If a full file path is provided, it is saved at that exact location.
            Default is the current directory ("./").
        name : str, optional
            Filename (without extension) to use when saving the scaler if `path` is a directory.
            Default is "scaler".

        Raises
        ------
        OSError
            If the directory cannot be created or file cannot be written.
        RuntimeError
            If the scaler has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted scaler. Call fit() first.")

        try:
            path = Path(path)

            if path.is_dir() or (not path.exists() and not path.suffix):
                # Treat as directory
                path.mkdir(parents=True, exist_ok=True)
                file_path = path / f"{name}.joblib"
            else:
                # Treat as full file path
                file_path = path
                file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save both the scaler and metadata
            scaler_data = {
                'scaler': self.scaler,
                'type': self.type,
                'numeric_cols': self._numeric_cols,
                'original_columns': self._original_columns,
                'feature_names_': self.feature_names_,
                'n_features_in_': self.n_features_in_,
                'is_fitted': self.is_fitted
            }

            joblib.dump(scaler_data, file_path)
            print(f"Scaler saved to: {file_path}")

        except Exception as e:
            raise OSError(f"Failed to save scaler: {str(e)}")

    @classmethod
    def load_scaler(cls, path: Union[str, Path]) -> "Scaler":
        """
        Load a scaler object from the specified path.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the saved scaler file.

        Returns
        -------
        Scaler
            Loaded scaler instance.

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist.
        ValueError
            If the file doesn't contain valid scaler data.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")

        try:
            scaler_data = joblib.load(path)

            if not isinstance(scaler_data, dict):
                raise ValueError("Invalid scaler file format")

            # Create new scaler instance
            scaler_instance = cls(scaler_data['type'])

            # Restore state
            scaler_instance.scaler = scaler_data['scaler']
            scaler_instance._numeric_cols = scaler_data['numeric_cols']
            scaler_instance._original_columns = scaler_data['original_columns']
            scaler_instance.feature_names_ = scaler_data['feature_names_']
            scaler_instance.n_features_in_ = scaler_data['n_features_in_']
            scaler_instance.is_fitted = scaler_data['is_fitted']

            print(f"Scaler loaded from: {path}")
            return scaler_instance

        except Exception as e:
            raise ValueError(f"Failed to load scaler: {str(e)}")

    def __repr__(self) -> str:
        """String representation of the Scaler object."""
        status = "fitted" if self.is_fitted else "not fitted"
        if self.is_fitted:
            return (f"Scaler(type={self.type.value}, {status}, "
                    f"features={len(self.feature_names_)})")
        else:
            return f"Scaler(type={self.type.value}, {status})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()


def create_scaler(type_: Union[str, ScalerType] = "s", **kwargs) -> Scaler:
    """
    Convenience function to create a scaler instance.

    Parameters
    ----------
    type_ : str or ScalerType, optional
        Type of scaler to create (default is 's' for StandardScaler).
    **kwargs
        Additional arguments to pass to the scaler constructor.

    Returns
    -------
    Scaler
        New scaler instance.

    Examples
    --------
    >>> scaler = create_scaler('s')  # StandardScaler
    >>> robust_scaler = create_scaler('r', quantile_range=(5, 95))  # RobustScaler
    >>> minmax_scaler = create_scaler('m', feature_range=(0, 1))  # MinMaxScaler
    """
    return Scaler(type_, **kwargs)


def save_multiple_scalers(scalers: Dict[str, Scaler], path: Union[str, Path] = "./scalers/") -> None:
    """
    Save multiple scalers to a directory.

    Parameters
    ----------
    scalers : dict
        Dictionary mapping scaler names to Scaler instances.
    path : str or pathlib.Path, optional
        Directory path where scalers will be saved (default is "./scalers/").

    Raises
    ------
    ValueError
        If scalers dictionary is empty or contains non-Scaler objects.
    OSError
        If directory cannot be created or files cannot be written.

    Examples
    --------
    >>> scalers = {
    ...     'standard': create_scaler('s'),
    ...     'robust': create_scaler('r'),
    ...     'minmax': create_scaler('m')
    ... }
    >>> save_multiple_scalers(scalers, "./my_scalers/")
    """
    if not isinstance(scalers, dict) or not scalers:
        raise ValueError("scalers must be a non-empty dictionary")

    # Validate all values are Scaler instances
    for name, scaler in scalers.items():
        if not isinstance(scaler, Scaler):
            raise ValueError(f"'{name}' is not a Scaler instance")
        if not scaler.is_fitted:
            raise RuntimeError(f"Scaler '{name}' has not been fitted yet")

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    try:
        for name, scaler in scalers.items():
            scaler.save_scaler(path, name)
    except Exception as e:
        raise OSError(f"Failed to save scalers: {str(e)}")


def load_multiple_scalers(path: Union[str, Path]) -> Dict[str, Scaler]:
    """
    Load multiple scalers from a directory.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory path containing saved scaler files (.joblib).

    Returns
    -------
    dict
        Dictionary mapping scaler names to loaded Scaler instances.

    Raises
    ------
    FileNotFoundError
        If the directory doesn't exist.
    ValueError
        If no scaler files are found in the directory.

    Examples
    --------
    >>> scalers = load_multiple_scalers("./my_scalers/")
    >>> standard_scaler = scalers['standard']
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    # Find all .joblib files
    scaler_files = list(path.glob("*.joblib"))

    if not scaler_files:
        raise ValueError(f"No scaler files (.joblib) found in directory: {path}")

    scalers = {}
    for file_path in scaler_files:
        name = file_path.stem  # filename without extension
        try:
            scalers[name] = Scaler.load_scaler(file_path)
        except Exception as e:
            print(f"Warning: Failed to load scaler '{name}': {str(e)}")
            continue

    if not scalers:
        raise ValueError("No valid scalers could be loaded from the directory")

    return scalers