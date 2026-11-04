"""Gaussian-process ore model on log Cu grade (GPyTorch exact GP)."""

from __future__ import annotations

import logging
import time
from typing import Protocol

import gpytorch
import numpy as np
import torch
from scipy.stats import norm

from geoopt.config import ModelConfig
from geoopt.model.transforms import CoordScaler, to_log_grade

log = logging.getLogger(__name__)

_PREDICT_CHUNK = 2_000  # larger chunks build a dense test covariance and slow down sharply


class OreModel(Protocol):
    def fit(self, xyz: np.ndarray, cu: np.ndarray) -> None: ...
    def condition(self, xyz: np.ndarray, cu: np.ndarray) -> None: ...
    def predict(
        self, xyz: np.ndarray, include_noise: bool = False
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def prob_ore(
        self, xyz: np.ndarray, cutoff_pct: float | None = None, include_noise: bool = True
    ) -> np.ndarray: ...


class _ExactGP(gpytorch.models.ExactGP):
    def __init__(
        self, x: torch.Tensor, y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood
    ) -> None:
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=3)
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class GPOreModel:
    """Exact GP with an ARD Matérn-5/2 kernel on standardized coordinates.

    `fit` tunes hyperparameters (call it on seed data only). `condition` adds data with
    hyperparameters frozen; every `refit_every` conditioning steps it re-tunes on all
    data seen so far.
    """

    def __init__(self, cfg: ModelConfig, seed: int) -> None:
        self._cfg = cfg
        self._device = _resolve_device(cfg.device)
        dtype = cfg.dtype
        if dtype == "auto":
            dtype = "float32" if self._device.type == "cuda" else "float64"
        self._dtype = torch.float32 if dtype == "float32" else torch.float64
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self._scaler: CoordScaler | None = None
        self._xyz = np.empty((0, 3))
        self._y = np.empty(0)
        self._likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self._model: _ExactGP | None = None
        self._steps_since_fit = 0

    # Public API --------------------------------------------------------------

    def fit(self, xyz: np.ndarray, cu: np.ndarray) -> None:
        self._scaler = CoordScaler.fit(xyz)
        self._xyz = np.asarray(xyz, dtype=float)
        self._y = to_log_grade(cu, self._cfg.log_eps)
        self._train()

    def condition(self, xyz: np.ndarray, cu: np.ndarray) -> None:
        if self._model is None:
            raise RuntimeError("fit() must be called before condition()")
        self._xyz = np.vstack([self._xyz, np.asarray(xyz, dtype=float)])
        self._y = np.concatenate([self._y, to_log_grade(cu, self._cfg.log_eps)])
        self._steps_since_fit += 1
        if self._cfg.refit_every and self._steps_since_fit >= self._cfg.refit_every:
            self._train()
        else:
            x, y = self._training_tensors()
            self._model.set_train_data(x, y, strict=False)

    def predict(
        self, xyz: np.ndarray, include_noise: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std of log grade. `include_noise` adds the nugget (sample-level)."""
        model, likelihood = self._require_fitted()
        x = self._tensor(self._scaler.transform(xyz))  # type: ignore[union-attr]
        means, stds = [], []
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for start in range(0, len(x), _PREDICT_CHUNK):
                dist = model(x[start : start + _PREDICT_CHUNK])
                if include_noise:
                    dist = likelihood(dist)
                means.append(dist.mean.double().cpu().numpy())
                stds.append(dist.variance.clamp_min(1e-12).sqrt().double().cpu().numpy())
        if not means:
            return np.empty(0), np.empty(0)
        return np.concatenate(means), np.concatenate(stds)

    def prob_ore(
        self, xyz: np.ndarray, cutoff_pct: float | None = None, include_noise: bool = True
    ) -> np.ndarray:
        """P(Cu > cutoff) from the log-normal predictive.

        With `include_noise` (default) this is for a sample at each location. Without it, it is
        for the smooth latent grade: near 0.5 only where the model can't call ore vs waste.
        """
        cutoff = self._cfg.cutoff_pct if cutoff_pct is None else cutoff_pct
        mean, std = self.predict(xyz, include_noise=include_noise)
        return norm.sf((np.log(cutoff + self._cfg.log_eps) - mean) / std)

    @property
    def n_train(self) -> int:
        return len(self._y)

    def hyperparameters(self) -> dict[str, float | list[float]]:
        model, likelihood = self._require_fitted()
        scaled = model.covar_module.base_kernel.lengthscale.detach().double().cpu().numpy().ravel()
        return {
            "mean": float(model.mean_module.constant.item()),
            "outputscale": float(model.covar_module.outputscale.item()),
            "noise": float(likelihood.noise.item()),
            "lengthscale_m": self._scaler.lengthscales_m(scaled).round(1).tolist(),  # type: ignore[union-attr]
        }

    # Internals ---------------------------------------------------------------

    def _require_fitted(self) -> tuple[_ExactGP, gpytorch.likelihoods.GaussianLikelihood]:
        if self._model is None or self._likelihood is None:
            raise RuntimeError("model is not fitted")
        return self._model, self._likelihood

    def _training_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        idx = np.arange(len(self._y))
        if len(idx) > self._cfg.max_train:
            idx = np.sort(self._rng.choice(idx, self._cfg.max_train, replace=False))
            log.info("training cap: using %d of %d points", self._cfg.max_train, len(self._y))
        x = self._tensor(self._scaler.transform(self._xyz[idx]))  # type: ignore[union-attr]
        return x, self._tensor(self._y[idx])

    def _tensor(self, a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=self._dtype, device=self._device)

    def _train(self) -> None:
        start = time.perf_counter()
        x, y = self._training_tensors()
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self._device, self._dtype)
        model = _ExactGP(x, y, likelihood).to(self._device, self._dtype)
        model.mean_module.constant.data.fill_(float(y.mean()))
        model.covar_module.outputscale = float(y.var())
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self._cfg.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(self._cfg.fit_iters):
            optimizer.zero_grad()
            loss = -mll(model(x), y)
            loss.backward()
            optimizer.step()
        self._model, self._likelihood = model, likelihood
        self._steps_since_fit = 0
        log.info(
            "gp fit n=%d device=%s dtype=%s loss=%.3f seconds=%.1f hyper=%s",
            len(y),
            self._device,
            self._dtype,
            loss.item(),
            time.perf_counter() - start,
            self.hyperparameters(),
        )


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("model.device is cuda but torch has no CUDA device")
    return torch.device(name)
