"""
Layer 2: Gauge Structure — Transport, Connections, Curvature
=============================================================

The principal G-bundle π: N → C with G = GL(K).

Key constructions:
  - GaugeFrame: agent-local GL(K) frame Ω_i(c) over base manifold
  - TransportOperator: Ω_ij = Ω_i @ Ω_j⁻¹ — translates between frames
  - GaugeConnection: A_μ = U⁻¹ ∂_μ U induced by gauge frame fields
  - GaugeCurvature: F_μν = ∂_μ A_ν − ∂_ν A_μ + [A_μ, A_ν]

Reference: Dennis (2026), Sections 2.6–2.8
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, List

from gauge_agent.lie_groups import transport_operator, transport_mean, transport_covariance
from gauge_agent.statistical_manifold import gaussian_kl


class GaugeFrame(nn.Module):
    """Gauge frame field Ω_i: U_i → GL(K) over the base manifold.

    For dim(C) = 0 (transformer limit): single matrix per agent.
    For dim(C) ≥ 1: a field of matrices over a discretized grid.

    Args:
        K: fiber dimension
        N_agents: number of agents
        grid_shape: shape of base manifold discretization, () for 0D
        init_scale: perturbation from identity
    """

    def __init__(self, K: int, N_agents: int,
                 grid_shape: Tuple[int, ...] = (),
                 init_scale: float = 0.1):
        super().__init__()
        self.K = K
        self.N_agents = N_agents
        self.grid_shape = grid_shape

        # Shape: (N_agents, *grid_shape, K, K)
        full_shape = (N_agents,) + grid_shape + (K, K)
        I = torch.eye(K)
        # Broadcast identity to full shape
        frames = I.reshape((1,) * (1 + len(grid_shape)) + (K, K))
        frames = frames.expand(full_shape).clone()
        frames = frames + init_scale * torch.randn(full_shape)
        self.frames = nn.Parameter(frames)

    def get_frame(self, agent_idx: int,
                  grid_idx: Optional[Tuple[int, ...]] = None) -> Tensor:
        """Get gauge frame for agent at optional grid location.

        Args:
            agent_idx: which agent
            grid_idx: optional grid coordinates
        Returns:
            (K, K) gauge frame matrix
        """
        if grid_idx is not None and self.grid_shape:
            return self.frames[(agent_idx,) + grid_idx]
        return self.frames[agent_idx]

    def transport_ij(self, i: int, j: int,
                     grid_idx: Optional[Tuple[int, ...]] = None) -> Tensor:
        """Transport operator Ω_ij = Ω_i @ Ω_j⁻¹ at a point.

        Args:
            i, j: agent indices
            grid_idx: optional grid location
        Returns:
            (K, K) or (*grid_shape, K, K) transport operator
        """
        omega_i = self.get_frame(i, grid_idx)
        omega_j = self.get_frame(j, grid_idx)
        return transport_operator(omega_i, omega_j)

    def all_pairwise_transports(self) -> Tensor:
        """Compute all pairwise Ω_ij for all agents at all grid points.

        Returns:
            (N, N, *grid_shape, K, K)
        """
        N = self.N_agents
        # frames: (N, *grid, K, K)
        # Need: omega_i @ inv(omega_j) for all i,j
        omega_inv = torch.linalg.inv(self.frames)
        # Reshape for broadcasting: (N,1,...) @ (1,N,...)
        ndim_grid = len(self.grid_shape)
        omega_i = self.frames.unsqueeze(1)  # (N, 1, *grid, K, K)
        omega_j_inv = omega_inv.unsqueeze(0)  # (1, N, *grid, K, K)
        return omega_i @ omega_j_inv


class TransportOperator:
    """Applies gauge transport to Gaussian beliefs.

    Given Ω_ij, transforms q_j = N(μ_j, Σ_j) into agent i's frame:
        μ' = Ω_ij μ_j
        Σ' = Ω_ij Σ_j Ω_ij^T
    """

    @staticmethod
    def transport_belief(omega_ij: Tensor, mu_j: Tensor, sigma_j: Tensor
                         ) -> Tuple[Tensor, Tensor]:
        """Transport a Gaussian belief through gauge operator.

        Args:
            omega_ij: (..., K, K) transport operator
            mu_j: (..., K) mean to transport
            sigma_j: (..., K, K) covariance to transport
        Returns:
            (mu_transported, sigma_transported) in agent i's frame
        """
        mu_t = transport_mean(omega_ij, mu_j)
        sigma_t = transport_covariance(omega_ij, sigma_j)
        return mu_t, sigma_t

    @staticmethod
    def gauge_aligned_kl(mu_i: Tensor, sigma_i: Tensor,
                         mu_j: Tensor, sigma_j: Tensor,
                         omega_ij: Tensor) -> Tensor:
        """KL(q_i || Ω_ij[q_j]) — the gauge-covariant KL divergence.

        This is Eq. (14) of the manuscript: the alignment energy E_ij.

        Args:
            mu_i, sigma_i: agent i's belief parameters
            mu_j, sigma_j: agent j's belief parameters
            omega_ij: (..., K, K) transport operator
        Returns:
            (...,) KL divergence after gauge alignment
        """
        mu_j_t, sigma_j_t = TransportOperator.transport_belief(
            omega_ij, mu_j, sigma_j
        )
        return gaussian_kl(mu_i, sigma_i, mu_j_t, sigma_j_t)


class GaugeConnection(nn.Module):
    """Gauge connection A_μ = U⁻¹ ∂_μ U on base manifolds of dim ≥ 1.

    For fields on a grid, ∂_μ U is computed via finite differences.
    The connection encodes how the gauge frame twists across the base.

    Reference: Eq. (17) of the manuscript.

    Args:
        gauge_frame: GaugeFrame module with grid_shape ≠ ()
        grid_spacing: physical spacing between grid points
    """

    def __init__(self, gauge_frame: GaugeFrame,
                 grid_spacing: float = 1.0):
        super().__init__()
        self.gauge_frame = gauge_frame
        self.h = grid_spacing

    def connection_form(self, agent_idx: int, mu_dir: int) -> Tensor:
        """Compute A_μ^(i) = U_i⁻¹ ∂_μ U_i for one agent in one direction.

        Uses central finite differences: ∂_μ U ≈ (U[c+h] - U[c-h]) / 2h.

        Args:
            agent_idx: which agent
            mu_dir: which base manifold direction (0, 1, ...)
        Returns:
            (*grid_shape, K, K) connection one-form values
        """
        U = self.gauge_frame.frames[agent_idx]  # (*grid, K, K)
        U_inv = torch.linalg.inv(U)

        # Finite difference along direction mu_dir
        # Roll forward and backward
        dU = (torch.roll(U, -1, dims=mu_dir) - torch.roll(U, 1, dims=mu_dir)) / (2 * self.h)
        return U_inv @ dU

    def field_strength(self, agent_idx: int, mu: int, nu: int) -> Tensor:
        """Gauge curvature F_μν = ∂_μ A_ν − ∂_ν A_μ + [A_μ, A_ν].

        Reference: Eq. (18) of the manuscript.

        Args:
            agent_idx: which agent
            mu, nu: base manifold direction indices
        Returns:
            (*grid_shape, K, K) curvature tensor values
        """
        A_mu = self.connection_form(agent_idx, mu)
        A_nu = self.connection_form(agent_idx, nu)

        # ∂_μ A_ν via finite differences
        dmu_A_nu = (torch.roll(A_nu, -1, dims=mu) - torch.roll(A_nu, 1, dims=mu)) / (2 * self.h)
        dnu_A_mu = (torch.roll(A_mu, -1, dims=nu) - torch.roll(A_mu, 1, dims=nu)) / (2 * self.h)

        # Commutator [A_μ, A_ν] = A_μ A_ν − A_ν A_μ
        commutator = A_mu @ A_nu - A_nu @ A_mu

        return dmu_A_nu - dnu_A_mu + commutator

    def yang_mills_action(self, agent_idx: int) -> Tensor:
        """Yang-Mills action: S_YM = ∫ tr(F_μν F^μν) dc.

        Summed over all direction pairs for flat base (Euclidean metric).

        Args:
            agent_idx: which agent
        Returns:
            Scalar Yang-Mills action
        """
        n_dims = len(self.gauge_frame.grid_shape)
        total = torch.tensor(0.0, device=self.gauge_frame.frames.device)
        for mu in range(n_dims):
            for nu in range(mu + 1, n_dims):
                F = self.field_strength(agent_idx, mu, nu)
                # tr(F F^T) summed over grid
                total = total + (F * F).sum()
        return total


class GaugeCurvature:
    """Diagnostic tools for gauge field curvature.

    Computes holonomy, Wilson loops, and topological invariants.
    """

    @staticmethod
    def holonomy_around_plaquette(gauge_frame: GaugeFrame,
                                  agent_idx: int,
                                  corner: Tuple[int, int],
                                  mu: int = 0, nu: int = 1) -> Tensor:
        """Holonomy around a unit plaquette in the (μ,ν) plane.

        H = Ω(c) Ω(c+ê_μ)⁻¹ Ω(c+ê_μ+ê_ν) Ω(c+ê_ν)⁻¹

        For vertex-local Ω_ij = Ω_i Ω_j⁻¹, holonomy vanishes (flat).
        Non-trivial holonomy requires independent edge variables.

        Args:
            gauge_frame: the gauge frame field
            agent_idx: which agent
            corner: (i, j) grid coordinates of bottom-left corner
            mu, nu: plaquette plane directions
        Returns:
            (K, K) holonomy matrix (should be identity for vertex-local)
        """
        i, j = corner
        U = gauge_frame.frames[agent_idx]

        # Four corners of the plaquette
        U_00 = U[i, j]
        U_10 = U[i + 1, j] if mu == 0 else U[i, j + 1]
        U_11 = U[i + 1, j + 1] if (mu == 0 and nu == 1) else U[i + 1, j + 1]
        U_01 = U[i, j + 1] if nu == 1 else U[i + 1, j]

        # Product around loop
        H = U_00 @ torch.linalg.inv(U_10) @ U_11 @ torch.linalg.inv(U_01)
        return H

    @staticmethod
    def flatness_measure(gauge_frame: GaugeFrame, agent_idx: int) -> Tensor:
        """Measure how flat the connection is: ‖F_μν‖ averaged over grid.

        Returns scalar; zero for pure gauge (vertex-local frames).
        """
        conn = GaugeConnection(gauge_frame)
        n_dims = len(gauge_frame.grid_shape)
        total_norm = torch.tensor(0.0, device=gauge_frame.frames.device)
        count = 0
        for mu in range(n_dims):
            for nu in range(mu + 1, n_dims):
                F = conn.field_strength(agent_idx, mu, nu)
                total_norm = total_norm + torch.norm(F, p='fro', dim=(-2, -1)).mean()
                count += 1
        return total_norm / max(count, 1)
