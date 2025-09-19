# -----------------------------------------------------------------------------
# Copyright 2025 Down Syndrome Education International and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd


def to_float_array(x: list | pd.Series | np.ndarray | None) -> np.ndarray:
    """
    **DEPRECATED: use to_float64_array instead**

    Convert input to a NumPy array of float64, with None as np.nan.

    Parameters
    ----------
    x : list | pd.Series | np.ndarray | None
        Input data to be converted.

    Returns
    -------
    np.ndarray
        Converted array of type float64, with None values as np.nan.
    """
    return (
        pd.Series(x, dtype="float64")
        .convert_dtypes()
        .to_numpy(na_value=np.nan, copy=True)
    )


def to_float32_array(x: list | pd.Series | np.ndarray | None) -> np.ndarray:
    """
    Convert input to a NumPy array of float32, with None as np.nan.

    Parameters
    ----------
    x : list | pd.Series | np.ndarray | None
        Input data to be converted.

    Returns
    -------
    np.ndarray
        Converted array of type float32, with None values as np.nan.
    """
    return (
        pd.Series(x, dtype="float32")
        .convert_dtypes()
        .to_numpy(na_value=np.nan, copy=True)
    )


def to_float64_array(x: list | pd.Series | np.ndarray | None) -> np.ndarray:
    """
    Convert input to a NumPy array of float64, with None as np.nan.

    Parameters
    ----------
    x : list | pd.Series | np.ndarray | None
        Input data to be converted.

    Returns
    -------
    np.ndarray
        Converted array of type float64, with None values as np.nan.
    """
    return (
        pd.Series(x, dtype="float64")
        .convert_dtypes()
        .to_numpy(na_value=np.nan, copy=True)
    )


def to_int32_array(x: list | pd.Series | np.ndarray | None) -> np.ndarray:
    """
    Convert input to a NumPy array of int32, with None as np.nan.

    Parameters
    ----------
    x : list | pd.Series | np.ndarray | None
        Input data to be converted.

    Returns
    -------
    np.ndarray
        Converted array of type int32, with None values as np.nan.
    """
    return (
        pd.Series(x, dtype="int32")
        .convert_dtypes()
        .to_numpy(na_value=np.nan, copy=True)
    )


def to_int64_array(x: list | pd.Series | np.ndarray | None) -> np.ndarray:
    """
    Convert input to a NumPy array of int64, with None as np.nan.

    Parameters
    ----------
    x : list | pd.Series | np.ndarray | None
        Input data to be converted.

    Returns
    -------
    np.ndarray
        Converted array of type int64, with None values as np.nan.
    """
    return (
        pd.Series(x, dtype="int64")
        .convert_dtypes()
        .to_numpy(na_value=np.nan, copy=True)
    )


def zscore(x: list | pd.Series | np.ndarray | None) -> np.ndarray:
    """
    Standardize input array to have mean 0 and standard deviation 1, ignoring NaNs.

    Parameters
    ----------
    x : list | pd.Series | np.ndarray | None
        Input data to be standardized.

    Returns
    -------
    np.ndarray
        Standardized array with mean 0 and standard deviation 1.
    """
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x, ddof=1)
