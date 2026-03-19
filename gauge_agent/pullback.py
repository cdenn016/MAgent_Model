"""
Layer 8: The Pullback Construction — "It From Bit"
====================================================

The core mechanism: agent belief fields σ_i: C → B induce
Riemannian metrics on the base manifold C via pullback of
the Fisher-Rao metric from the statistical fiber B.

G^(q)_{i,μν}(c) = E_{q_i(c)}[(∂_μ log q_i)(∂_ν log q_i)]

For Gaussians:
G_{μν} = (∂_μ μ)^T Σ⁻¹ (∂_ν μ) + (1/2) tr(Σ⁻¹ ∂_μΣ Σ⁻¹ ∂_νΣ)

This is Wheeler's "it from bit": geometric structure (it)
emerges from informational structure (bit).

The eigenvalue decomposition of G yields:
  - Observable sector: large eigenvalues → perceived spacetime
  - Dark sector: intermediate eigenvalues → present but unperceived
  - Internal sector: negligible eigenvalues → pure internal DOF

Reference: Dennis (2026), Section 5 (Pullback Construction)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple, List

from gauge_agent.agents import Agent, MultiAgentSystem


class PullbackMetric(nn.Module):
    """Computes the pullback-induced Fisher-Rao metric on the base manifold.

    For agent i with beliefs q_i(c) = N(μ_i(c), Σ_i(c)) over grid C,
    the induced metric at point c is:

    G_{μν}(c) = (∂_μ μ)^T Σ⁻¹ (∂_ν μ) + (1/2) tr(Σ⁻¹ ∂_μΣ Σ⁻¹ ∂_νΣ)

    First term: how means vary → "location curvature"
    Second term: how uncertainties vary → "shape curvature"

    Args:
        grid_spacing: physical distance between grid points
    """

    def __init__(self, grid_spacing: float = 1.0):
        super().__init__()
        self.h = grid_spacing

    def _finite_diff(self, field: Tensor, direction: int) -> Tensor:
        """Central finite difference along a grid direction.

        Args:
            field: (*grid, ...) tensor field on the grid
            direction: which grid dimension to differentiate along
        Returns:
            (*grid, ...) derivative tensor
        """
        return (torch.roll(field, -1, dims=direction) -
                torch.roll(field, 1, dims=direction)) / (2 * self.h)

    def induced_metric(self, agent: Agent) -> Tensor:
        """Compute the pullback metric G_{μν} for one agent.

        Requires agent to be defined on a grid (grid_shape ≠ ()).

        Args:
            agent: Agent with grid_shape of dimension n
        Returns:
            (*grid, n, n) metric tensor field
        """
        n_dims = len(agent.grid_shape)
        if n_dims == 0:
            raise ValueError("Pullback requires dim(C) ≥ 1")

        mu = agent.mu_q.data         # (*grid, K)
        sigma = agent.sigma_q        # (*grid, K, K)
        sigma_inv = agent.precision_q  # (*grid, K, K)

        G = torch.zeros(agent.grid_shape + (n_dims, n_dims),
                        device=mu.device, dtype=mu.dtype)

        for mu_dir in range(n_dims):
            for nu_dir in range(mu_dir, n_dims):
                # ∂_μ μ and ∂_ν μ
                dmu_mu = self._finite_diff(mu, mu_dir)  # (*grid, K)
                dmu_nu = self._finite_diff(mu, nu_dir)

                # Mean contribution: (∂_μ μ)^T Σ⁻¹ (∂_ν μ)
                mean_term = (dmu_mu.unsqueeze(-2) @ sigma_inv @ dmu_nu.unsqueeze(-1))
                mean_term = mean_term.squeeze(-1).squeeze(-1)

                # ∂_μ Σ and ∂_ν Σ
                dsigma_mu = self._finite_diff(sigma, mu_dir)  # (*grid, K, K)
                dsigma_nu = self._finite_diff(sigma, nu_dir)

                # Covariance contribution: (1/2) tr(Σ⁻¹ ∂_μΣ Σ⁻¹ ∂_νΣ)
                A = sigma_inv @ dsigma_mu  # (*grid, K, K)
                B = sigma_inv @ dsigma_nu
                cov_term = 0.5 * (A * B.transpose(-2, -1)).sum(dim=(-2, -1))

                G[..., mu_dir, nu_dir] = mean_term + cov_term
                if mu_dir != nu_dir:
                    G[..., nu_dir, mu_dir] = G[..., mu_dir, nu_dir]

        return G

    def prior_induced_metric(self, agent: Agent) -> Tensor:
        """Pullback from prior field — the 'ontological geometry'.

        G^(p)_{μν} = E_{p_i}[(∂_μ log p_i)(∂_ν log p_i)]

        This represents the agent's perceived geometry of reality.
        """
        n_dims = len(agent.grid_shape)
        if n_dims == 0:
            raise ValueError("Pullback requires dim(C) ≥ 1")

        mu = agent.mu_p.data
        sigma_inv = agent.precision_p

        G = torch.zeros(agent.grid_shape + (n_dims, n_dims),
                        device=mu.device, dtype=mu.dtype)

        for mu_dir in range(n_dims):
            for nu_dir in range(mu_dir, n_dims):
                dmu_mu = self._finite_diff(mu, mu_dir)
                dmu_nu = self._finite_diff(mu, nu_dir)
                mean_term = (dmu_mu.unsqueeze(-2) @ sigma_inv @ dmu_nu.unsqueeze(-1))
                mean_term = mean_term.squeeze(-1).squeeze(-1)

                dsigma_mu = self._finite_diff(agent.sigma_p, mu_dir)
                dsigma_nu = self._finite_diff(agent.sigma_p, nu_dir)
                A = sigma_inv @ dsigma_mu
                B = sigma_inv @ dsigma_nu
                cov_term = 0.5 * (A * B.transpose(-2, -1)).sum(dim=(-2, -1))

                G[..., mu_dir, nu_dir] = mean_term + cov_term
                if mu_dir != nu_dir:
                    G[..., nu_dir, mu_dir] = G[..., mu_dir, nu_dir]

        return G

    def consensus_metric(self, system: MultiAgentSystem) -> Tensor:
        """Gauge-invariant consensus metric — averaged over agents.

        ḡ_{μν} = (1/N) Σ_i w_i G_{i,μν}

        This is the closest analog to 'objective spacetime geometry'.

        Args:
            system: MultiAgentSystem with grid_shape ≠ ()
        Returns:
            (*grid, n, n) consensus metric tensor
        """
        metrics = []
        for agent in system.agents:
            G = self.induced_metric(agent)
            metrics.append(G)

        # Stack and average
        all_G = torch.stack(metrics, dim=0)  # (N, *grid, n, n)
        return all_G.mean(dim=0)


class MetricDecomposition:
    """Eigenvalue decomposition of induced metrics into sectors.

    The spectrum of G naturally decomposes into:
      Observable: λ_a > Λ_obs  (perceived spacetime)
      Dark:       Λ_dark < λ_a ≤ Λ_obs  (present but unperceived)
      Internal:   λ_a ≤ Λ_dark  (pure internal DOF)

    Reference: Dennis (2026), Section 5.4
    """

    @staticmethod
    def eigendecompose(G: Tensor) -> Tuple[Tensor, Tensor]:
        """Eigendecomposition of metric tensor at each grid point.

        Args:
            G: (*grid, n, n) metric tensor
        Returns:
            eigenvalues: (*grid, n) sorted descending
            eigenvectors: (*grid, n, n) columns are eigenvectors
        """
        # Symmetrize
        G_sym = 0.5 * (G + G.transpose(-2, -1))
        eigenvalues, eigenvectors = torch.linalg.eigh(G_sym)
        # Sort descending
        idx = eigenvalues.argsort(dim=-1, descending=True)
        eigenvalues = eigenvalues.gather(-1, idx)
        eigenvectors = eigenvectors.gather(-1, idx.unsqueeze(-2).expand_as(eigenvectors))
        return eigenvalues, eigenvectors

    @staticmethod
    def sector_decomposition(eigenvalues: Tensor,
                              lambda_obs: float = 0.1,
                              lambda_dark: float = 0.01
                              ) -> Dict[str, Tensor]:
        """Classify eigenvalues into observable/dark/internal sectors.

        Args:
            eigenvalues: (*grid, n) eigenvalues
            lambda_obs: threshold for observable sector
            lambda_dark: threshold for dark sector
        Returns:
            Dict with 'observable', 'dark', 'internal' boolean masks
            and 'n_obs', 'n_dark', 'n_internal' counts
        """
        obs_mask = eigenvalues > lambda_obs
        dark_mask = (eigenvalues > lambda_dark) & (~obs_mask)
        internal_mask = eigenvalues <= lambda_dark

        return {
            'observable': obs_mask,
            'dark': dark_mask,
            'internal': internal_mask,
            'n_obs': obs_mask.float().sum(-1).mean().item(),
            'n_dark': dark_mask.float().sum(-1).mean().item(),
            'n_internal': internal_mask.float().sum(-1).mean().item(),
            'eigenvalues': eigenvalues,
        }

    @staticmethod
    def information_flux(eigenvalues: Tensor) -> Tensor:
        """Total information flux: tr(G) = Σ λ_a.

        Measures total rate of belief variation across base manifold.

        Args:
            eigenvalues: (*grid, n)
        Returns:
            (*grid,) total flux
        """
        return eigenvalues.sum(-1)

    @staticmethod
    def effective_dimension(eigenvalues: Tensor, threshold: float = 0.01) -> Tensor:
        """Effective dimensionality via participation ratio.

        d_eff = (Σ λ_a)² / Σ λ_a²

        This is 1 for a single dominant direction, n for uniform spectrum.

        Args:
            eigenvalues: (*grid, n)
        Returns:
            (*grid,) effective dimension
        """
        total = eigenvalues.sum(-1)
        total_sq = (eigenvalues ** 2).sum(-1)
        return total ** 2 / (total_sq + 1e-10)
