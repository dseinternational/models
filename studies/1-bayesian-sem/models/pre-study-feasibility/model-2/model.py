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

# -----------------------------------------------------------------------------
# RI-CLPM with Beta-Binomial observations (3 constructs × 4 waves)
#
# Notation in code mirrors the math at the top of this file:
#   x  : person-specific latent state (within-person deviations)
#   α  : between-person random intercepts (stable trait differences)
#   μ  : time- and construct-specific intercept (logit scale; random walk)
#   B  : K×K transition matrix (AR on diag; cross-lag off-diag), applied x_{t-1} -> x_t
#   θ  : latent logit for Beta-Binomial probability p = logistic(θ)
#   κ  : Beta-Binomial concentration; ρ = 1 / (1 + κ) is overdispersion
#   δ  : non-negative loading of a shared day_factor on each construct
#
# Identifiability choices:
#   • day_factor is mean-centered within person to avoid confounding with α and μ.
#   • α has population-level regressions (age, covariates) + correlated residuals.
#   • x_1 and α both have LKJ priors to allow cross-construct covariance.
#   • Soft stability barrier shrinks row L1 norms of B below ~0.95 (stationarity proxy).
#
# All priors are specified on the logit scale unless noted.
# -----------------------------------------------------------------------------

"""
Model 2 (RI-CLPM with Beta-Binomial outcomes)
=============================================

State process (per person n, construct k, time t):
    x_{n,·,t} = B x_{n,·,t-1} + η ⊙ ε_{n,·,t},  ε ~ N(0, I)

Latent logit:
    θ_{n,k,t} = μ_{k,t} + α_{n,k} + x_{n,k,t} + δ_k * day_factor_{n,t}

Observation:
    y_{n,k,t} ~ BetaBinomial(n_{k,t},
                             α = κ * p_{n,k,t},
                             β = κ * (1 - p_{n,k,t})),
    p_{n,k,t} = logistic(θ_{n,k,t})

Glossary / symbol map
---------------------
alpha         α_{n,k}      between-person random intercept (per construct)
x             x_{n,k,t}    person-specific latent state following RI-CLPM
B                            K×K transition matrix (AR on diag, CL off-diag)
mu            μ_{k,t}      time- & construct-specific intercept on logit scale
day_factor                  common day effect per (person,time)
day_loading   δ_k          non-negative loading per construct
kappa         κ            Beta-Binomial concentration (>0); rho=1/(1+κ)
theta         θ            latent logit; p=logistic(θ)

Dims:
    person: N
    construct: K = 3 ("Vocab","LetterSound","WordReading")
    time: T = 4
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pymc as pm
import pytensor.tensor as pt

# ---------------------------------------------------------------------
# (0) Configuration & constants
# ---------------------------------------------------------------------

# Top-level constants kept for familiarity/compatibility
# (Actual values are also carried in ModelConfig.)
J_vocab = 170  # receptive vocabulary items
J_ls = 32  # letter sound knowledge items
J_wr = 79  # word reading items


@dataclass(frozen=True)
class ModelConfig:
    """
    Centralised hyperparameters (documented by intended scale).

    All sd's here live on the *logit* scale unless noted otherwise.

    Notes on interpretation:
    - sd_prior_alpha/sd_prior_init ≈ typical scale for person-level differences in logits.
    - sd_prior_B_* control shrinkage of the transition matrix B after tanh transform.
      The tanh(Normal) mapping keeps entries in (-1,1) while allowing soft shrinkage
      to modest autoregression (diag) and small cross-lags (off-diag).
    - log_kappa_* set a weakly informative prior on overdispersion; larger κ→ρ≈0.
    - The stability_* parameters define a soft penalty (via softplus) that discourages
      explosive dynamics without hard constraints that hurt sampling.

    """

    # Items per test
    J_vocab: int = J_vocab
    J_ls: int = J_ls
    J_wr: int = J_wr

    # Prior scales
    sd_prior_alpha: float = 0.3  # sd for α (random intercept) via LKJ sd_dist
    sd_prior_eta: float = 0.3  # sd for state innovations η
    sd_prior_init: float = 0.3  # sd for initial state x_1 via LKJ sd_dist
    sd_prior_mu_step: float = 0.3  # sd for μ random-walk step per time
    sd_prior_person_beta: float = 0.5  # sd for person covariate betas
    sd_prior_age_beta: float = 0.5  # sd for age betas

    # Transition matrix B prior spread (tanh(Normal) in (-1,1))
    sd_prior_B_diag: float = 0.20  # 0.15
    sd_prior_B_offdiag: float = 0.15  # 0.10

    # Beta-Binomial concentration κ prior on log-κ
    log_kappa_mu: float = float(np.log(100.0))
    log_kappa_sd: float = 1.0

    # Stability barrier for B (soft constraint)
    stability_rowsum_cap: float = 0.95
    stability_penalty_scale: float = 10.0
    stability_steepness: float = 50.0


# ---------------------------------------------------------------------
# (1) Data specification (validation) and preparation
# ---------------------------------------------------------------------


class ModelSpecification:
    """
    Specifies the inputs required to build the model.

    All arrays should be NumPy arrays with appropriate shapes.

    Attributes
    ----------
    age : (N,) float
        Age in months at baseline
    gender : (N,) int
        Gender (1 = male, 2 = female)
    area : (N,) int
        Area (1 = north, 2 = south)
    rv : (N, 4) int
        Vocabulary scores over 4 time points
    ls : (N, 4) int
        Letter sound knowledge scores over 4 time points
    wr : (N, 4) int
        Word reading scores over 4 time points
    """

    # All inputs use wide format: (N, 4) for outcomes, (N,) for covariates.
    # Missing outcome values should be np.nan (not negative numbers).
    # Binary factors are expected as {1,2} and internally recoded to {-0.5, +0.5}
    # to make intercepts interpretable and priors symmetric.

    def __init__(
        self,
        age: np.ndarray,
        gender: np.ndarray,
        area: np.ndarray,
        rv: np.ndarray,
        ls: np.ndarray,
        wr: np.ndarray,
    ):
        """
        Parameters
        ----------
        age : (N,) float
            Age in months at baseline
        gender : (N,) int
            Gender (1 = male, 2 = female)
        area : (N,) int
            Area (1 = north, 2 = south)
        rv : (N, 4) int
            Vocabulary scores over 4 time points
        ls : (N, 4) int
            Letter sound knowledge scores over 4 time points
        wr : (N, 4) int
            Word reading scores over 4 time points
        """
        N = age.shape[0]

        for name, arr in [
            ("age", age),
            ("gender", gender),
            ("area", area),
        ]:
            if arr.ndim != 1:
                raise ValueError(f"{name}: expected shape (N,), got {arr.shape}.")
            if arr.shape[0] != N:
                raise ValueError(
                    f"All inputs must share the same N; {name} has N={arr.shape[0]}, expected {N}."
                )

        for name, arr in [("rv", rv), ("ls", ls), ("wr", wr)]:
            if arr.ndim != 2 or arr.shape[1] != 4:
                raise ValueError(f"{name}: expected shape (N, 4), got {arr.shape}.")
            if arr.shape[0] != N:
                raise ValueError(
                    f"All inputs must share the same N; {name} has N={arr.shape[0]}, expected {N}."
                )

        for name, arr in [
            ("gender", gender),
            ("area", area),
        ]:
            if not np.isin(arr, [1, 2]).all():
                raise ValueError(f"{name} must be 1/2; found values {np.unique(arr)}")

        # Store as contiguous float64 (consistent with PyMC default transforms)
        self.age = np.ascontiguousarray(age.astype(np.float64, copy=True))
        self.gender = np.ascontiguousarray(gender.astype(np.float64, copy=True))
        self.area = np.ascontiguousarray(area.astype(np.float64, copy=True))
        self.rv = np.ascontiguousarray(rv.astype(np.float64, copy=True))
        self.ls = np.ascontiguousarray(ls.astype(np.float64, copy=True))
        self.wr = np.ascontiguousarray(wr.astype(np.float64, copy=True))


@dataclass
class PreparedData:
    """Arrays, masks, and coordinates consumed by `build_model`."""

    N: int
    K: int
    T: int
    constructs: np.ndarray
    coords: Dict[str, np.ndarray]
    # Vectorised observed entries for Beta-Binomial
    y_vec: np.ndarray  # (n_obs,)
    n_vec: np.ndarray  # (n_obs,)
    obs_idx: np.ndarray  # (n_obs,) int indices into flattened (N*K*T)
    # Person-level covariates
    X: np.ndarray  # (N, P)
    covar_names: List[str]
    age_z: np.ndarray  # (N,)
    # Trials by construct/time for reporting
    n_ct: np.ndarray  # (K, T)


# This function:
#   1) stacks rv/ls/wr -> y_float with shape (N, K, T),
#   2) builds an observation mask and vectorises only observed cells for the
#      Beta-Binomial likelihood (y_vec, n_vec),
#   3) standardises covariates and defines named coords for PyMC dims.
def prepare_data(spec: ModelSpecification, cfg: ModelConfig) -> PreparedData:
    """
    Validates and reshapes inputs, builds masks, and defines coords.

    Notes
    -----
    - Binary factors (gender, area) are centred to {-0.5, +0.5}.
    - Age is z-scored to keep priors on the logit scale interpretable.
    """
    N = spec.age.shape[0]
    constructs = np.array(["Vocab", "LetterSound", "WordReading"], dtype=str)
    K = constructs.size
    T = spec.rv.shape[1]

    # Stack outcomes to (N,K,T); allow NaN for missing
    y_float = np.stack([spec.rv, spec.ls, spec.wr], axis=1).astype(np.float64)

    # Missing values are np.nan. We keep them in y_float to preserve array shapes,
    # but the likelihood is only applied to vectorised observed cells.
    mask = ~np.isfinite(y_float)  # True where missing
    y_filled = np.where(mask, 0.0, y_float)

    # Trial counts per construct/time (constant across people).
    # If instruments change across time, modify n_ct by column.
    n_ct = np.array(
        [[cfg.J_vocab] * T, [cfg.J_ls] * T, [cfg.J_wr] * T], dtype=np.int64
    )  # (K,T)
    n_3d = np.broadcast_to(n_ct, (N, K, T))  # (N,K,T)

    # Validate counts only at observed cells
    invalid = (~mask) & ((y_filled < 0) | (y_filled > n_3d))
    if np.any(invalid):
        i = np.argwhere(invalid)
        raise ValueError(
            f"Found {i.shape[0]} count(s) outside [0, n]. Example index: {tuple(i[0])}"
        )

    # Vectorisation strategy:
    #   - flatten (N*K*T) to a single axis,
    #   - keep only observed indices (obs_idx) to avoid NaN handling in the graph,
    #   - gather α, β, and n for those indices at likelihood time.
    obs_mask = (~mask).ravel()
    y_vec = y_filled.ravel()[obs_mask].astype(np.int64)
    n_vec = n_3d.ravel()[obs_mask].astype(np.int64)
    obs_idx = np.flatnonzero(obs_mask).astype(np.int64)

    # Person-level covariates
    # Contrast coding to {-0.5, +0.5} centers factors and keeps α_mean interpretable.
    gender_c = np.where(spec.gender == 1, -0.5, 0.5).astype(np.float64)
    area_c = np.where(spec.area == 1, -0.5, 0.5).astype(np.float64)
    X = np.column_stack([gender_c, area_c]).astype(np.float64)
    covar_names = ["gender", "area"]

    # Age standardisation
    age = spec.age.astype(np.float64)
    age_z = (age - np.nanmean(age)) / np.nanstd(age, ddof=0)

    coords = dict(
        person=np.arange(N),
        construct=constructs,
        construct_to=constructs,
        construct_from=constructs,
        time=np.arange(1, T + 1, dtype=int),
        time_lag=np.arange(1, T, dtype=int),
        covar=np.array(covar_names, dtype=str),
    )

    return PreparedData(
        N=N,
        K=K,
        T=T,
        constructs=constructs,
        coords=coords,
        y_vec=y_vec,
        n_vec=n_vec,
        obs_idx=obs_idx,
        X=X,
        covar_names=covar_names,
        age_z=age_z,
        n_ct=n_ct,
    )


# ---------------------------------------------------------------------
# (2) Model construction
# ---------------------------------------------------------------------


def build_model(
    spec: ModelSpecification, config: ModelConfig | None = None
) -> pm.Model:
    """
    Build the PyMC model for a 3-construct RI-CLPM with Beta-Binomial outcomes.

    Parameters
    ----------
    spec : ModelSpecification
        Validated inputs (counts and covariates).
    config : ModelConfig, optional
        Hyperparameters and constants; defaults chosen for logit scale stability.

    Returns
    -------
    model : pm.Model
    """
    cfg = config or ModelConfig()
    prep = prepare_data(spec, cfg)

    # Convenience constants and identities
    N, K, T = prep.N, prep.K, prep.T
    I = np.eye(K, dtype=np.float64)
    M_off = 1.0 - I

    # Provide an explicit initial belief about μ_{k,1} on the *probability* scale.
    # Order must match `constructs` ("Vocab","LetterSound","WordReading")
    # Note: not strictly priors - from study, but could be estimated from other studies
    mu0_prob = np.array([0.20, 0.44, 0.08], dtype=np.float64)
    if mu0_prob.shape[0] != K:
        # Fallback if K changes in the future
        mu0_prob = np.full(K, 0.30, dtype=np.float64)

    with pm.Model(coords=prep.coords) as model:
        # -----------------------------------------------------------------
        # (2.1) Data containers (immutables inside the graph)
        # -----------------------------------------------------------------
        obs_idx_data = pm.Data("obs_idx", prep.obs_idx.astype("int64"))
        X_person = pm.Data("X_person", prep.X, dims=("person", "covar"))
        age_z = pm.Data("age_z", prep.age_z, dims=("person",))
        n_ct = pm.Data("n_ct", prep.n_ct, dims=("construct", "time"))

        # I and M_off are used to split B into diagonal (AR) vs off-diagonal (cross-lag)
        # for reporting and soft constraints; they are not strictly necessary for sampling.
        I_data = pm.Data("I", I, dims=("construct_to", "construct_from"))
        M_off_data = pm.Data("M_off", M_off, dims=("construct_to", "construct_from"))

        # -----------------------------------------------------------------
        # (2.2) Person-level fixed effects
        # -----------------------------------------------------------------
        # Rationale: 0-centred priors with modest scale keep logits plausible.
        beta_age = pm.Normal(
            "beta_age", 0.0, cfg.sd_prior_age_beta, dims=("construct",)
        )
        beta_person = pm.Normal(
            "beta_person", 0.0, cfg.sd_prior_person_beta, dims=("covar", "construct")
        )

        # α_mean[n,k] collects stable, between-person predictors for construct k.
        # These operate at the trait level (not time-varying within person).
        alpha_mean = age_z.dimshuffle(0, "x") * beta_age.dimshuffle(
            "x", 0
        ) + pm.math.dot(
            X_person, beta_person
        )  # (N,K)

        # -----------------------------------------------------------------
        # (2.3) Between-person random intercepts α[n,k]
        # -----------------------------------------------------------------

        # α residuals (after fixed effects) are multivariate normal across constructs,
        # allowing, e.g., people high in vocabulary to also be high in word reading.
        # Using LKJ with HalfNormal sd priors avoids degenerate correlations.
        alpha_chol, alpha_corr, alpha_sd = pm.LKJCholeskyCov(
            "alpha_scale",
            n=K,
            eta=3.0,
            sd_dist=pm.HalfNormal.dist(cfg.sd_prior_alpha),
            compute_corr=True,
        )

        alpha_z = pm.Normal("alpha_z", 0.0, 1.0, dims=("person", "construct"))
        alpha = pm.Deterministic(
            "alpha", alpha_mean + alpha_z @ alpha_chol.T, dims=("person", "construct")
        )

        pm.Deterministic("alpha_sd", alpha_sd, dims=("construct",))
        pm.Deterministic(
            "alpha_corr", alpha_corr, dims=("construct_to", "construct_from")
        )

        # -----------------------------------------------------------------
        # (2.4) State evolution (RI-CLPM): x_t = B x_{t-1} + η ⊙ ε
        # -----------------------------------------------------------------

        # η scales the white-noise shocks to the within-person state x_t (per construct).
        # Larger η implies more volatile within-person deviations over time.
        eta_sd = pm.HalfNormal("eta_sd", 0.25, dims=("construct",))
        state_shock = pm.StudentT(
            "state_shock",
            nu=10,
            mu=0.0,
            sigma=1.0,
            dims=("person", "construct", "time_lag"),
        )

        # Transition B in (-1,1) via tanh(Normal)
        base_B = np.zeros((K, K), dtype=np.float64)
        np.fill_diagonal(base_B, 0.1)  # 0.3
        raw_B_mu = np.arctanh(np.clip(base_B, -0.95, 0.95))
        sigma_B = np.full((K, K), cfg.sd_prior_B_offdiag, dtype=np.float64)
        np.fill_diagonal(sigma_B, cfg.sd_prior_B_diag)

        # Transition matrix prior:
        #   raw_B ~ Normal then B = tanh(raw_B) ∈ (-1,1).
        #   Diagonal entries (AR) have slightly larger prior sd than cross-lags.
        #   base_B initializes prior mean to modest autoregression (~0.3).
        raw_B = pm.Normal(
            "raw_B", mu=raw_B_mu, sigma=sigma_B, dims=("construct_to", "construct_from")
        )
        B = pm.Deterministic(
            "B", pm.math.tanh(raw_B), dims=("construct_to", "construct_from")
        )

        # Soft stability barrier:
        # Penalise rows where sum(|B_row|) > cap (~0.95). This is a *soft* regulariser,
        # not a hard constraint, to preserve NUTS geometry and avoid divergences that
        # hard clipping can create. Adjust scale/steepness if you see frequent violations.
        row_sum = pt.sum(pt.abs(B), axis=1)
        pm.Potential(
            "B_row_sum_barrier",
            -cfg.stability_penalty_scale
            * pt.sum(
                pt.softplus(
                    cfg.stability_steepness * (row_sum - cfg.stability_rowsum_cap)
                )
            ),
        )

        # Initial within-person state x_1 allows cross-construct covariance (LKJ).
        # This is conceptually distinct from α: x captures deviations that *evolve*;
        # α captures stable trait-like differences explained by covariates + residuals.
        init_chol, _, _ = pm.LKJCholeskyCov(
            "init_scale",
            n=K,
            eta=3.0,
            sd_dist=pm.HalfNormal.dist(cfg.sd_prior_init),
            compute_corr=True,
        )

        x1_z = pm.Normal("x1_z", 0.0, 1.0, dims=("person", "construct"))
        x1 = pm.Deterministic("x_1", x1_z @ init_chol.T, dims=("person", "construct"))

        # State recursion:
        #   x_t = B x_{t-1} + η ⊙ ε_{t}, with ε standard normal.
        # We unroll the recursion in Python to keep shapes explicit for PyMC.
        x_list = [x1]
        for t in range(1, T):
            mu_t = x_list[-1] @ B.T  # (N,K)
            x_t = mu_t + state_shock[:, :, t - 1] * eta_sd.dimshuffle("x", 0)
            x_list.append(x_t)
        x = pm.Deterministic(
            "x", pt.stack(x_list, axis=2), dims=("person", "construct", "time")
        )

        # -----------------------------------------------------------------
        # (2.5) Time- and construct-specific intercepts μ_{k,t}
        # -----------------------------------------------------------------

        # μ is a construct-specific random walk on the logit scale, capturing shifts
        # in average difficulty/ability across waves that are NOT person-specific.
        # We anchor μ_1 by expressing prior means on the probability scale (mu0_prob)
        # and mapping via logit, which makes the prior easy to reason about.
        mu0 = pm.Normal("mu0", pm.math.logit(mu0_prob), 0.5, dims=("construct",))

        mu_step_sd = pm.HalfNormal("mu_step_sd", cfg.sd_prior_mu_step)
        nu_mu = pm.Exponential("nu_mu_minus2", 1 / 10) + 2
        mu_steps = pm.StudentT(
            "mu_steps",
            nu=nu_mu,
            mu=0.0,
            sigma=mu_step_sd,
            dims=("construct", "time_lag"),
        )

        mu = pm.Deterministic(
            "mu",
            pt.cumsum(
                pt.concatenate([mu0.dimshuffle(0, "x"), mu_steps], axis=1), axis=1
            ),
            dims=("construct", "time"),
        )

        # -----------------------------------------------------------------
        # (2.6) Latent logit θ and observation model (Beta-Binomial)
        # -----------------------------------------------------------------
        # Common day effect per (person,time); non-negative loadings per construct.

        day_raw = pm.Normal("day_raw", 0.0, 1.0, dims=("person", "time"))

        # day_factor is a shared (person, time) nuisance capturing day-to-day variation
        # (e.g., mood, illness). We mean-center within person so its average effect is 0,
        # preventing it from absorbing α or μ shifts. δ_k ≥ 0 are construct-specific
        # loadings (reported as 'day_OR' on the odds scale for interpretability).
        day_factor = pm.Deterministic(
            "day_factor",
            day_raw - day_raw.mean(axis=1, keepdims=True),
            dims=("person", "time"),
        )
        delta_sd = pm.HalfNormal("delta_sd", 0.2)
        delta = pm.HalfNormal("delta", delta_sd, dims=("construct",))
        pm.Deterministic("day_OR", pm.math.exp(delta), dims=("construct",))

        theta = pm.Deterministic(
            "theta",
            mu.dimshuffle("x", 0, 1)
            + alpha.dimshuffle(0, 1, "x")
            + x
            + day_factor.dimshuffle(0, "x", 1) * delta.dimshuffle("x", 0, "x"),
            dims=("person", "construct", "time"),
        )

        # Clip p to (ε,1-ε) to avoid α or β ~ 0 in Beta-Binomial at extreme logits.
        p_full = pm.Deterministic(
            "p", pm.math.sigmoid(theta), dims=("person", "construct", "time")
        )

        # Numerical safety: clipping p prevents α or β going to ~0 in Beta-Binomial,
        # which can create extremely peaked likelihoods at extreme logits and harm NUTS.
        p_clipped = pt.clip(p_full, 1e-6, 1.0 - 1e-6)

        # Overdispersion: κ = softplus(kappa_uncon) ensures κ>0.
        # Setting mu≈log(expm1(100)) makes softplus(mu)≈100 (weak info: ρ≈1/(1+100)≈0.0099).
        # Increase prior sd if you want to allow heavier tails (larger ρ).
        kappa_uncon = pm.Normal(
            "kappa_uncon",
            mu=np.log(np.expm1(100.0)),  # inverse_softplus(100)
            sigma=1.0,
            dims=("construct",),
        )
        kappa = pm.Deterministic(
            "kappa", pm.math.log1pexp(kappa_uncon), dims=("construct",)
        )
        rho = pm.Deterministic("rho", 1.0 / (1.0 + kappa), dims=("construct",))

        # Form α,β for Beta-Binomial and select observed entries (avoid NaN handling)
        kappa_3d = kappa.dimshuffle("x", 0, "x")  # (1,K,1)
        alpha_bb_3d = p_clipped * kappa_3d
        beta_bb_3d = (1.0 - p_clipped) * kappa_3d

        # Gather α,β only for observed cells via obs_idx to avoid shape-index pitfalls
        # with advanced integer indexing inside Aesara/NumPy graphs.
        alpha_vec_all = alpha_bb_3d.reshape((N * K * T,))
        beta_vec_all = beta_bb_3d.reshape((N * K * T,))
        alpha_vec = pt.take(alpha_vec_all, obs_idx_data)
        beta_vec = pt.take(beta_vec_all, obs_idx_data)

        # Observed Beta-Binomial likelihood
        y = pm.BetaBinomial(
            "y",
            n=prep.n_vec,  # 1-D numpy array of trials
            alpha=alpha_vec,  # 1-D tensor, same shape as y_vec
            beta=beta_vec,  # 1-D tensor, same shape as y_vec
            observed=prep.y_vec,  # 1-D numpy array of counts
        )

        # -----------------------------------------------------------------
        # (2.7) Derived quantities & diagnostics (grouped, named for reporting)
        # -----------------------------------------------------------------

        # Report-friendly decomposition of B:
        #   B_autoregressive[k]   = B[k,k]
        #   B_crosslag[k,i!=k]    = B[k,i]
        # Use these when summarising stability vs cross-construct influence.
        AR_diag = pt.diag(B)  # (K,)
        pm.Deterministic(
            "B_autoregressive", pt.sum(B * I_data, axis=1), dims=("construct_to",)
        )
        pm.Deterministic(
            "B_crosslag", B * M_off_data, dims=("construct_to", "construct_from")
        )

        # Half-life of within-person deviations: number of time steps for a shock to
        # decay by 50% given |AR|. Values >~3 suggest strong persistence.
        pm.Deterministic(
            "B_half_life",
            pt.log(0.5) / pt.log(pt.clip(pt.abs(AR_diag), 1e-6, 0.999)),
            dims=("construct",),
        )

        p_bar = pm.Deterministic(
            "p_bar", pm.math.sigmoid(mu), dims=("construct", "time")
        )

        # Binomial information per wave (n·p·(1-p)) is useful for gauging measurement
        # precision across time. The effective information under Beta-Binomial is
        # reduced by the variance inflation factor (vif).
        items_per_logit = pm.Deterministic(
            "items_per_logit", n_ct * p_bar * (1.0 - p_bar), dims=("construct", "time")
        )

        # Approximate expected change in *items correct* for a +1 s.d. day_factor,
        # assuming small-δ linearisation on the logit scale.
        pm.Deterministic(
            "day_items_effect",
            delta.dimshuffle(0, "x") * items_per_logit,
            dims=("construct", "time"),
        )

        # Beta-Binomial variance: Var(Y) = n p (1-p) [1 + (n-1) ρ].
        # Here ρ = 1/(1+κ). When κ→∞, ρ→0 and variance reduces to Binomial.
        vif = pm.Deterministic(
            "bb_var_inflation",
            1.0 + (n_ct - 1.0) * rho.dimshuffle(0, "x"),
            dims=("construct", "time"),
        )

        pm.Deterministic(
            "var_y_mean",
            n_ct * p_bar * (1.0 - p_bar) * vif,
            dims=("construct", "time"),
        )

        pm.Deterministic(
            "items_per_logit_eff",
            items_per_logit / vif,
            dims=("construct", "time"),
        )

    return model
