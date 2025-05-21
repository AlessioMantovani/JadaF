from typing import Union
from numpy.typing import NDArray
import joblib
from enum import Enum
import polars as pl
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
    """
    STANDARD = "s"
    ROBUST = "r"
    MINMAX = "m"

class Scaler:
    """
    Wrapper class for common sklearn scalers: StandardScaler, RobustScaler, MinMaxScaler.

    Allows dynamic selection of scaler type and supports standard scaler operations.

    Parameters
    ----------
    type_ : str or ScalerType, optional
        Type of scaler to use:
            - 's' or ScalerType.STANDARD for StandardScaler
            - 'r' or ScalerType.ROBUST for RobustScaler
            - 'm' or ScalerType.MINMAX for MinMaxScaler
        Default is 's' (StandardScaler).
    **kwargs
        Additional keyword arguments to pass to the scaler constructor.

    Raises
    ------
    ValueError
        If an unsupported scaler type is provided.
    """

    def __init__(self, type_: Union[str, ScalerType] = "s", **kwargs):
        if isinstance(type_, str):
            type_ = ScalerType(type_)
        self.type = type_
        self.scaler = self._init_scaler(**kwargs)
        self._numeric_cols = None

    def _init_scaler(self, **kwargs):
        """
        Initialize the appropriate scaler based on the provided type.

        Parameters
        ----------
        **kwargs
            Arguments passed to the scaler constructor.

        Returns
        -------
        scaler : StandardScaler, RobustScaler, or MinMaxScaler
            An instance of the selected scaler.

        Raises
        ------
        ValueError
            If the scaler type is not supported.
        """
        if self.type == ScalerType.STANDARD:
            from sklearn.preprocessing import StandardScaler
            return StandardScaler(**kwargs)
        elif self.type == ScalerType.ROBUST:
            from sklearn.preprocessing import RobustScaler
            return RobustScaler(**kwargs)
        elif self.type == ScalerType.MINMAX:
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler(**kwargs)
        else:
            raise ValueError(f"Unsupported scaler type: {self.type}")

    def fit(self, X: jd.JDF) -> "Scaler":
        """
        Fit the scaler to the data.

        Parameters
        ----------
        X : jd.JDF
            Input data to fit the scaler.

        Returns
        -------
        self : Scaler
            Returns the Scaler instance to allow method chaining.
        """
        numeric_df = X.select_dtypes(include=["number"])
        self._numeric_cols = numeric_df.columns
        self.scaler.fit(numeric_df)
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
        """
        if self._numeric_cols is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        X_numeric = X[self._numeric_cols]
        transformed = self.scaler.transform(X_numeric)
        transformed_df = jd.JDF(transformed, columns=self._numeric_cols)
        non_numeric = X.drop(columns=self._numeric_cols)
        result = jd.concat([transformed_df, non_numeric])
        return result[X.columns]

    def fit_transform(self, X: jd.JDF) -> jd.JDF:
        """
        Fit the scaler to the data and transform it.

        Parameters
        ----------
        X : jd.JDF
            Input data to fit and transform.

        Returns
        -------
        jd.JDF
            Scaled data, with numeric columns scaled and non-numeric columns unchanged.
        """
        numeric_df = X.select_dtypes(include=["number"])
        self._numeric_cols = numeric_df.columns
        transformed = self.scaler.fit_transform(numeric_df.df)
        transformed_df =transformed
        non_numeric = X.drop(self._numeric_cols)
        result = pl.concat([transformed_df, non_numeric])
        result = jd.JDF(result, columns=X.columns+non_numeric)
        return result[X.columns]

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
            Data transformed back to original scale, numeric columns restored and non-numeric columns unchanged.
        """
        if self._numeric_cols is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        X_numeric = X[self._numeric_cols]
        inversed = self.scaler.inverse_transform(X_numeric)
        inversed_df = jd.JDF(inversed, columns=self._numeric_cols)
        non_numeric = X.drop(columns=self._numeric_cols)
        result = jd.concat([inversed_df, non_numeric], axis=1)
        return result[X.columns]

    def get_params(self, deep: bool = True):
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

    def set_params(self, **params):
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
        """
        self.scaler.set_params(**params)
        return self

    def save_scaler(self, path: Union[str, Path] = "./", name: str = "scaler"):
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
            If the directory cannot be created.
        """
        path = Path(path)
        if path.is_dir():
            file_path = path / f"{name}.joblib"
        else:
            file_path = path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, file_path)
