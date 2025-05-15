from typing import Union
from numpy.typing import NDArray
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from enum import Enum

class ScalerType(str, Enum):
    """
    Enumeration of supported scaler types.

    Values:
        STANDARD: StandardScaler ('s')
        ROBUST: RobustScaler ('r')
        MINMAX: MinMaxScaler ('m')
    """
    STANDARD = "s"
    ROBUST = "r"
    MINMAX = "m"

class Scaler:
    """
    A wrapper class for common sklearn scalers (StandardScaler, RobustScaler, MinMaxScaler).

    Allows dynamic selection of scaler type and supports standard scaler operations.

    Args:
        type_ (str | ScalerType): Type of scaler to use ('s' for StandardScaler,
                                  'r' for RobustScaler, 'm' for MinMaxScaler).
        **kwargs: Additional keyword arguments passed to the selected scaler.

    Raises:
        ValueError: If an unsupported scaler type is provided.
    """

    def __init__(self, type_: str = "s", **kwargs):
        if isinstance(type_, str):
            type_ = ScalerType(type_)
        self.type = type_
        self.scaler = self._init_scaler(**kwargs)

    def _init_scaler(self, **kwargs):
        """
        Initialize the appropriate scaler based on the provided type.

        Args:
            **kwargs: Arguments passed to the scaler constructor.

        Returns:
            An instance of the selected scaler.

        Raises:
            ValueError: If the scaler type is not supported.
        """
        if self.type == ScalerType.STANDARD:
            return StandardScaler(**kwargs)
        elif self.type == ScalerType.ROBUST:
            return RobustScaler(**kwargs)
        elif self.type == ScalerType.MINMAX:
            return MinMaxScaler(**kwargs)
        else:
            raise ValueError(f"Unsupported scaler type: {self.type}")

    def fit(self, X: Union[pd.DataFrame, NDArray]) -> "Scaler":
        """
        Fit the scaler to the data.

        Args:
            X: Input data (NumPy array or pandas DataFrame).

        Returns:
            The Scaler instance (allows method chaining).
        """
        self.scaler.fit(X)
        return self

    def transform(self, X: Union[pd.DataFrame, NDArray]) -> NDArray:
        """
        Transform the data using the fitted scaler.

        Args:
            X: Input data to transform.

        Returns:
            Transformed data as a NumPy array.
        """
        return self.scaler.transform(X)

    def fit_transform(self, X: Union[pd.DataFrame, NDArray]) -> NDArray:
        """
        Fit the scaler to the data and transform it.

        Args:
            X: Input data.

        Returns:
            Scaled data as a NumPy array.
        """
        return self.scaler.fit_transform(X)

    def inverse_transform(self, X: Union[pd.DataFrame, NDArray]) -> NDArray:
        """
        Inverse transform the scaled data back to the original representation.

        Args:
            X: Scaled data.

        Returns:
            Original data as a NumPy array.
        """
        return self.scaler.inverse_transform(X)

    def get_params(self, deep: bool = True):
        """
        Get parameters of the underlying scaler.

        Args:
            deep: Whether to return the parameters for nested estimators.

        Returns:
            Dictionary of scaler parameters.
        """
        return self.scaler.get_params(deep=deep)

    def set_params(self, **params):
        """
        Set parameters for the underlying scaler.

        Args:
            **params: Parameters to set on the scaler.

        Returns:
            The Scaler instance (allows method chaining).
        """
        self.scaler.set_params(**params)
        return self
