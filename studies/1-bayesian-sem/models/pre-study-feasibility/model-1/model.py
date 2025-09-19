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

"""
Model 1
--------
A Bayesian regression model to predict word reading at end given age, receptive vocabulary,
letter-sound knowledge, phonetic spelling, nonverbal ability and word reading at start.
"""

import sys
import os
import numpy as np
import pandas as pd
import pymc as pm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, parent_dir)

import stats_utils

pd.options.mode.copy_on_write = True


class ModelSpecification:
    """
    Defines inputs for the model.
    """

    age: np.ndarray
    nv_start: np.ndarray
    rv_start: np.ndarray
    ls_start: np.ndarray
    sp_start: np.ndarray
    wr_start: np.ndarray
    wr_end: np.ndarray

    def __init__(
        self,
        age: list | pd.Series | np.ndarray,
        nv_start: list | pd.Series | np.ndarray,
        rv_start: list | pd.Series | np.ndarray,
        ls_start: list | pd.Series | np.ndarray,
        sp_start: list | pd.Series | np.ndarray,
        wr_start: list | pd.Series | np.ndarray,
        wr_end: list | pd.Series | np.ndarray,
    ):
        """
        Initialize the model specification.

        All inputs should be array-like and of the same length.

        Parameters
        -------------
        age : list | pd.Series | np.ndarray
            Age at start, must be fully observed (no NaNs).
        nv_start : list | pd.Series | np.ndarray
            Nonverbal Ability scores at start.
        rv_start : list | pd.Series | np.ndarray
            Receptive Vocabulary scores at start.
        ls_start : list | pd.Series | np.ndarray
            Letter-Sound Knowledge scores at start.
        sp_start : list | pd.Series | np.ndarray
            Phonetic Spelling scores at start.
        wr_start : list | pd.Series | np.ndarray
            Word Reading scores at start.
        wr_end : list | pd.Series | np.ndarray
            Word Reading scores at end.
        """
        self.age = stats_utils.to_float_array(age)
        self.nv_start = stats_utils.to_float_array(nv_start)
        self.rv_start = stats_utils.to_float_array(rv_start)
        self.ls_start = stats_utils.to_float_array(ls_start)
        self.sp_start = stats_utils.to_float_array(sp_start)
        self.wr_start = stats_utils.to_float_array(wr_start)
        self.wr_end = stats_utils.to_float_array(wr_end)


class ModelDefinition:
    """
    Defines the built model and its inputs (censoring NaNs).
    """

    spec: ModelSpecification
    age: np.ndarray
    age_centered: np.ndarray
    nv_start: np.ndarray
    nv_start_centered: np.ndarray
    rv_start: np.ndarray
    rv_start_centered: np.ndarray
    ls_start: np.ndarray
    ls_start_centered: np.ndarray
    sp_start: np.ndarray
    sp_start_centered: np.ndarray
    wr_start: np.ndarray
    wr_start_centered: np.ndarray
    wr_end: np.ndarray
    model: pm.Model

    def __init__(
        self,
        spec: ModelSpecification,
        age: np.ndarray,
        age_centered: np.ndarray,
        nv_start: np.ndarray,
        nv_start_centered: np.ndarray,
        rv_start: np.ndarray,
        rv_start_centered: np.ndarray,
        ls_start: np.ndarray,
        ls_start_centered: np.ndarray,
        sp_start: np.ndarray,
        sp_start_centered: np.ndarray,
        wr_start: np.ndarray,
        wr_start_centered: np.ndarray,
        wr_end: np.ndarray,
        model: pm.Model,
    ):
        self.spec = spec
        self.age = age
        self.age_centered = age_centered
        self.nv_start = nv_start
        self.nv_start_centered = nv_start_centered
        self.rv_start = rv_start
        self.rv_start_centered = rv_start_centered
        self.ls_start = ls_start
        self.ls_start_centered = ls_start_centered
        self.sp_start = sp_start
        self.sp_start_centered = sp_start_centered
        self.wr_start = wr_start
        self.wr_start_centered = wr_start_centered
        self.wr_end = wr_end
        self.model = model


def build_model(
    spec: ModelSpecification,
) -> ModelDefinition:
    """
    Build a Bayesian regression model to predict word reading at end (wr_end)
    from age and various cognitive measures at start.

    Predictors:
    - age: Age at start
    - nv: Nonverbal Ability at start
    - rv: Receptive Vocabulary at start
    - ls: Letter-Sound Knowledge at start
    - sp: Phonetic Spelling at start
    - wr: Word Reading at start

    Outcome:
    - wr: Word Reading at end

    Parameters
    -------------
    spec : ModelSpecification
        The model specification containing all necessary data.

    Returns
    -------
    PyMC Model object.

    """

    # confirm all inputs are the same length
    if not (
        len(spec.age)
        == len(spec.nv_start)
        == len(spec.rv_start)
        == len(spec.ls_start)
        == len(spec.sp_start)
        == len(spec.wr_start)
        == len(spec.wr_end)
    ):
        raise ValueError("All input arrays must have the same length.")

    # check age is fully observed
    if np.isnan(spec.age).any():
        raise ValueError("age must be fully observed (no NaNs).")

    # delete rows with NaNs
    mask = (
        ~np.isnan(spec.nv_start)
        & ~np.isnan(spec.rv_start)
        & ~np.isnan(spec.ls_start)
        & ~np.isnan(spec.sp_start)
        & ~np.isnan(spec.wr_start)
        & ~np.isnan(spec.wr_end)
    )

    age = spec.age[mask]
    nv_start = spec.nv_start[mask]
    rv_start = spec.rv_start[mask]
    ls_start = spec.ls_start[mask]
    sp_start = spec.sp_start[mask]
    wr_start = spec.wr_start[mask]
    wr_end = spec.wr_end[mask]

    # number of observations (after deletions)
    nobs = len(wr_end)

    # center predictors
    age_centered = age - age.mean()
    nv_start_centered = nv_start - nv_start.mean()
    rv_start_centered = rv_start - rv_start.mean()
    ls_start_centered = ls_start - ls_start.mean()
    sp_start_centered = sp_start - sp_start.mean()
    wr_start_centered = wr_start - wr_start.mean()

    # define model

    coords = {
        "observations": np.arange(nobs),
        "predictors": [
            "age",
            "nv",
            "rv",
            "ls",
            "sp",
            "wr",
        ],
    }

    model = pm.Model(coords=coords)

    with model:
        # outcome data
        data_end_obs = pm.Data("data_end_obs", wr_end, dims="observations")

        # priors for unknown model parameters

        # intercept prior:
        # ν=5 to allow for some heavier tails than normal
        # μ=39.5 (test midpoint)
        # σ=20 (wide)
        alpha = pm.StudentT("alpha", nu=5, mu=39.5, sigma=20)

        # predictors:
        # ν=5 to allow for some heavier tails than normal
        # μ=0 (centered)
        # σ=2 (weakly informative, most effects expected to be small)
        beta_age = pm.StudentT("beta_age", nu=5, mu=0, sigma=2)
        beta_nv = pm.StudentT("beta_nv", nu=5, mu=0, sigma=2)
        beta_rv = pm.StudentT("beta_rv", nu=5, mu=0, sigma=2)
        beta_ls = pm.StudentT("beta_ls", nu=5, mu=0, sigma=2)
        beta_sp = pm.StudentT("beta_sp", nu=5, mu=0, sigma=2)
        beta_wr = pm.StudentT("beta_wr", nu=5, mu=0, sigma=2)

        # weakly informative prior on residual scale
        sigma = pm.HalfNormal("sigma", sigma=15)

        # linear model - expected value of outcome
        mu = (
            alpha
            + beta_age * age_centered
            + beta_nv * nv_start_centered
            + beta_rv * rv_start_centered
            + beta_ls * ls_start_centered
            + beta_sp * sp_start_centered
            + beta_wr * wr_start_centered
        )

        # likelihood of observations
        _ = pm.Truncated(
            "end_obs",
            pm.StudentT.dist(nu=5, mu=mu, sigma=sigma),
            lower=0,
            upper=79,
            observed=data_end_obs,
            dims="observations",
        )

        return ModelDefinition(
            spec=spec,
            age=age,
            age_centered=age_centered,
            nv_start=nv_start,
            nv_start_centered=nv_start_centered,
            rv_start=rv_start,
            rv_start_centered=rv_start_centered,
            ls_start=ls_start,
            ls_start_centered=ls_start_centered,
            sp_start=sp_start,
            sp_start_centered=sp_start_centered,
            wr_start=wr_start,
            wr_start_centered=wr_start_centered,
            wr_end=wr_end,
            model=model,
        )
