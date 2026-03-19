"""
Layer 3: Agents as Smooth Sections of Associated Bundles
=========================================================

An agent A^i with support domain U_i ⊆ C consists of three smooth fields:
  q_i : U_i → E_q   (belief section — what it thinks is true)
  p_i : U_i → E_p   (prior section — what it expects)
  Ω_i : U_i → GL(K)  (gauge frame — its reference system)

For the 0D (transformer) limit: each field is a single value.
For dim(C) ≥ 1: fields are defined on a discretized grid.

Reference: Dennis (2026), Section 2.5 (Definition 5: Agent)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Dict, List

from gauge_agent.statistical_manifold import (
    GaussianDistribution, gaussian_kl, ensure_spd
)
from gauge_agent.gauge_structure import GaugeFrame, TransportOperator


class Agent(nn.Module):
    """A single agent as a section of the associated bundle.

    Carries belief q_i = N(μ_q, Σ_q), prior p_i = N(μ_p, Σ_p),
    and gauge frame Ω_i ∈ GL(K), all as fields over the base manifold.

    Args:
        K: fiber (latent) dimension
        grid_shape: shape of base manifold discretization, () for 0D
        init_belief_scale: scale for belief initialization
        init_prior_scale: scale for prior initialization
        init_gauge_scale: perturbation of gauge frame from identity
        agent_id: unique identifier
    """

    def __init__(self, K: int, grid_shape: Tuple[int, ...] = (),
                 init_belief_scale: float = 1.0,
                 init_prior_scale: float = 1.0,
                 init_gauge_scale: float = 0.1,
                 agent_id: int = 0):
        super().__init__()
        self.K = K
        self.grid_shape = grid_shape
        self.agent_id = agent_id

        # Number of grid points (1 for 0D)
        n_points = 1
        for s in grid_shape:
            n_points *= s
        self.n_points = max(n_points, 1)

        # Belief parameters q_i = N(μ_q, Σ_q)
        shape_mu = grid_shape + (K,)
        shape_sigma = grid_shape + (K, K)

        self.mu_q = nn.Parameter(init_belief_scale * torch.randn(shape_mu))
        L_q = init_belief_scale * torch.eye(K)
        if grid_shape:
            L_q = L_q.unsqueeze(0).expand(grid_shape + (K, K)).clone()
        self._L_q = nn.Parameter(L_q)

        # Prior parameters p_i = N(μ_p, Σ_p)
        self.mu_p = nn.Parameter(init_prior_scale * torch.randn(shape_mu))
        L_p = init_prior_scale * torch.eye(K)
        if grid_shape:
            L_p = L_p.unsqueeze(0).expand(grid_shape + (K, K)).clone()
        self._L_p = nn.Parameter(L_p)

        # Gauge frame Ω_i ∈ GL(K)
        shape_omega = grid_shape + (K, K)
        omega = torch.eye(K)
        if grid_shape:
            omega = omega.unsqueeze(0).expand(shape_omega).clone()
        omega = omega + init_gauge_scale * torch.randn(shape_omega)
        self.omega = nn.Parameter(omega)

        # Support function χ_i(c) — for now, full support
        self.register_buffer('chi', torch.ones(grid_shape if grid_shape else (1,)))

    @property
    def L_q(self) -> Tensor:
        """Cholesky factor of belief covariance with positive diagonal."""
        L = torch.tril(self._L_q)
        diag = L.diagonal(dim1=-2, dim2=-1)
        L = L - torch.diag_embed(diag) + torch.diag_embed(diag.abs().clamp(min=1e-6))
        return L

    @property
    def sigma_q(self) -> Tensor:
        """Belief covariance Σ_q = L_q L_q^T."""
        L = self.L_q
        return L @ L.transpose(-2, -1)

    @property
    def precision_q(self) -> Tensor:
        """Belief precision Λ_q = Σ_q⁻¹."""
        return torch.cholesky_inverse(self.L_q)

    @property
    def L_p(self) -> Tensor:
        """Cholesky factor of prior covariance with positive diagonal."""
        L = torch.tril(self._L_p)
        diag = L.diagonal(dim1=-2, dim2=-1)
        L = L - torch.diag_embed(diag) + torch.diag_embed(diag.abs().clamp(min=1e-6))
        return L

    @property
    def sigma_p(self) -> Tensor:
        """Prior covariance Σ_p = L_p L_p^T."""
        L = self.L_p
        return L @ L.transpose(-2, -1)

    @property
    def precision_p(self) -> Tensor:
        """Prior precision Λ_p = Σ_p⁻¹."""
        return torch.cholesky_inverse(self.L_p)

    def self_kl(self) -> Tensor:
        """Self-consistency: KL(q_i || p_i).

        This is the first term of the free energy functional (Eq. 23).

        Returns:
            (...,) KL divergence at each grid point
        """
        return gaussian_kl(self.mu_q, self.sigma_q, self.mu_p, self.sigma_p)

    def belief_entropy(self) -> Tensor:
        """Entropy of belief distribution H[q_i].

        Returns:
            (...,) entropy at each grid point
        """
        K = self.K
        log_det = 2.0 * self.L_q.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return 0.5 * K * (1.0 + torch.log(torch.tensor(2 * torch.pi))) + 0.5 * log_det

    def state_dict_info(self) -> Dict:
        """Summary of agent state for diagnostics."""
        return {
            'agent_id': self.agent_id,
            'K': self.K,
            'mu_q_norm': self.mu_q.data.norm().item(),
            'mu_p_norm': self.mu_p.data.norm().item(),
            'sigma_q_trace': self.sigma_q.diagonal(dim1=-2, dim2=-1).sum(-1).mean().item(),
            'sigma_p_trace': self.sigma_p.diagonal(dim1=-2, dim2=-1).sum(-1).mean().item(),
            'omega_det': torch.linalg.det(self.omega).mean().item(),
            'self_kl': self.self_kl().mean().item(),
        }


class MultiAgentSystem(nn.Module):
    """A collection of agents on a shared base manifold.

    M = {A^i = (q_i, p_i, Ω_i)}_{i ∈ I}

    Manages inter-agent transport, pairwise KL computation,
    and the overlap structure χ_ij = χ_i · χ_j.

    Reference: Dennis (2026), Definition 6 (Multi-Agent System)

    Args:
        N_agents: number of agents
        K: fiber dimension
        grid_shape: base manifold discretization
        init_belief_scale: scale for belief initialization
        init_prior_scale: scale for prior initialization
        init_gauge_scale: gauge frame perturbation scale
    """

    def __init__(self, N_agents: int, K: int,
                 grid_shape: Tuple[int, ...] = (),
                 init_belief_scale: float = 1.0,
                 init_prior_scale: float = 1.0,
                 init_gauge_scale: float = 0.1):
        super().__init__()
        self.N_agents = N_agents
        self.K = K
        self.grid_shape = grid_shape

        self.agents = nn.ModuleList([
            Agent(K, grid_shape,
                  init_belief_scale=init_belief_scale,
                  init_prior_scale=init_prior_scale,
                  init_gauge_scale=init_gauge_scale,
                  agent_id=i)
            for i in range(N_agents)
        ])

    def get_all_mu_q(self) -> Tensor:
        """Stack all belief means.

        Returns:
            (N, *grid, K) tensor of means
        """
        return torch.stack([a.mu_q for a in self.agents])

    def get_all_sigma_q(self) -> Tensor:
        """Stack all belief covariances.

        Returns:
            (N, *grid, K, K) tensor of covariances
        """
        return torch.stack([a.sigma_q for a in self.agents])

    def get_all_mu_p(self) -> Tensor:
        """Stack all prior means."""
        return torch.stack([a.mu_p for a in self.agents])

    def get_all_sigma_p(self) -> Tensor:
        """Stack all prior covariances."""
        return torch.stack([a.sigma_p for a in self.agents])

    def get_all_omega(self) -> Tensor:
        """Stack all gauge frames.

        Returns:
            (N, *grid, K, K) gauge frame matrices
        """
        return torch.stack([a.omega for a in self.agents])

    def pairwise_transport_operators(self) -> Tensor:
        """Compute all pairwise Ω_ij = Ω_i @ Ω_j⁻¹.

        Returns:
            (N, N, *grid, K, K) transport operators
        """
        omegas = self.get_all_omega()  # (N, *grid, K, K)
        omega_inv = torch.linalg.inv(omegas)  # (N, *grid, K, K)
        # (N, 1, *grid, K, K) @ (1, N, *grid, K, K)
        return omegas.unsqueeze(1) @ omega_inv.unsqueeze(0)

    def pairwise_alignment_energies(self, mode: str = 'belief') -> Tensor:
        """Compute E_ij(c) = KL(q_i || Ω_ij[q_j]) for all pairs.

        Reference: Eq. (14) — the alignment energy.

        Args:
            mode: 'belief' for q-q alignment, 'prior' for p-p alignment
        Returns:
            (N, N, *grid) tensor of pairwise KL divergences
        """
        if mode == 'belief':
            mu = self.get_all_mu_q()       # (N, *grid, K)
            sigma = self.get_all_sigma_q()  # (N, *grid, K, K)
        else:
            mu = self.get_all_mu_p()
            sigma = self.get_all_sigma_p()

        N = self.N_agents
        transports = self.pairwise_transport_operators()  # (N, N, *grid, K, K)

        # Transport all j beliefs into all i frames
        # mu_j: (1, N, *grid, K) → transported: (N, N, *grid, K)
        mu_j_expanded = mu.unsqueeze(0).expand(N, -1, *mu.shape[1:])
        sigma_j_expanded = sigma.unsqueeze(0).expand(N, -1, *sigma.shape[1:])

        # Apply transport: μ' = Ω_ij μ_j
        mu_transported = (transports @ mu_j_expanded.unsqueeze(-1)).squeeze(-1)
        # Σ' = Ω_ij Σ_j Ω_ij^T
        sigma_transported = transports @ sigma_j_expanded @ transports.transpose(-2, -1)

        # Expand mu_i, sigma_i for broadcasting
        mu_i_expanded = mu.unsqueeze(1).expand(-1, N, *mu.shape[1:])
        sigma_i_expanded = sigma.unsqueeze(1).expand(-1, N, *sigma.shape[1:])

        # Compute KL(q_i || Ω_ij[q_j]) for all pairs
        return gaussian_kl(mu_i_expanded, sigma_i_expanded,
                          mu_transported, sigma_transported)

    def overlap_mask(self) -> Tensor:
        """Compute overlap regions χ_ij = χ_i · χ_j.

        Returns:
            (N, N, *grid) binary overlap mask
        """
        chi = torch.stack([a.chi for a in self.agents])  # (N, *grid)
        # χ_ij = χ_i · χ_j
        return chi.unsqueeze(1) * chi.unsqueeze(0)

    def diagnostics(self) -> Dict:
        """System-level diagnostics."""
        info = {
            'N_agents': self.N_agents,
            'K': self.K,
            'grid_shape': self.grid_shape,
        }
        for a in self.agents:
            info[f'agent_{a.agent_id}'] = a.state_dict_info()

        # Average pairwise alignment energy
        E_belief = self.pairwise_alignment_energies('belief')
        mask = 1.0 - torch.eye(self.N_agents, device=E_belief.device)
        for _ in range(len(self.grid_shape)):
            mask = mask.unsqueeze(-1)
        info['mean_belief_alignment'] = (E_belief * mask).sum().item() / max(mask.sum().item(), 1)

        return info
