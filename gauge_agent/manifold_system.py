"""
Manifold-Aware Agent System — Wiring Everything Together
==========================================================

Connects:
  - gauge_agent.agents (Agent, MultiAgentSystem)
  - gauge_agent.manifolds (Manifold subclasses)
  - gauge_agent.lattice_gauge (LatticeGaugeField)
  - gauge_agent.support (χ_i support functions)
  - gauge_agent.free_energy (FreeEnergyFunctional)

into a single ManifoldAgentSystem that runs agents on curved
base manifolds with non-flat gauge connections.

The free energy integral becomes:

  S = Σ_i ∫_C χ_i KL(q_i || p_i) √|g| dc
    + Σ_ij ∫_C χ_ij β_ij KL(q_i || Ω̂_ij[q_j]) √|g| dc
    + Yang-Mills regularization

where Ω̂_ij = Ω_i · V_ij · Ω_j⁻¹ uses edge twist variables.

Reference: Dennis (2026), Sections 2.5, 2.10, 2.11
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple, List
import math

from gauge_agent.agents import Agent, MultiAgentSystem
from gauge_agent.manifolds import Manifold, EuclideanManifold
from gauge_agent.lattice_gauge import LatticeGaugeField, WilsonAction
from gauge_agent.support import ball_support, overlap_matrix, volume_weighted_integral
from gauge_agent.statistical_manifold import gaussian_kl
from gauge_agent.free_energy import FreeEnergyFunctional


class ManifoldAgentSystem(nn.Module):
    """Multi-agent system on an arbitrary Riemannian base manifold
    with non-flat gauge connections.

    This is the unified system that wires together:
      1. Base manifold C with metric g_μν
      2. N agents as sections of the associated bundle over C
      3. Lattice gauge field with edge twist variables V_ij
      4. Support functions χ_i(c) for spatial extent
      5. Volume-weighted free energy integration

    Args:
        N_agents: number of agents
        K: fiber dimension (GL(K) gauge group)
        manifold: Riemannian base manifold (discretized)
        init_twist_scale: perturbation scale for edge twists
        support_radius: geodesic radius of agent support balls
        support_sharpness: boundary steepness
        ym_coupling: Yang-Mills coupling constant β
    """

    def __init__(self, N_agents: int, K: int,
                 manifold: Manifold,
                 init_twist_scale: float = 0.05,
                 support_radius: Optional[float] = None,
                 support_sharpness: float = 5.0,
                 ym_coupling: float = 0.1):
        super().__init__()
        self.N_agents = N_agents
        self.K = K
        self.manifold = manifold
        self.ym_coupling = ym_coupling

        grid_shape = manifold.grid_shape

        # Standard multi-agent system (provides agents with beliefs/priors/frames)
        self.system = MultiAgentSystem(N_agents, K, grid_shape)

        # Lattice gauge field with edge twists
        self.gauge_field = LatticeGaugeField(
            K, grid_shape, mode='mixed', n_agents=N_agents,
            init_twist_scale=init_twist_scale, periodic=True
        )

        # Sync vertex frames: gauge_field.vertex_frames ↔ agent.omega
        self._sync_frames_to_gauge()

        # Support functions
        if support_radius is None:
            # Default: agents cover ~1/N_agents of the manifold
            support_radius = float('inf')  # full support

        self._setup_supports(support_radius, support_sharpness)

        # Precompute volume form
        self.register_buffer('vol_form', manifold.volume_form())

        # Wilson action for Yang-Mills regularization
        self.wilson_action = WilsonAction(self.gauge_field, ym_coupling)

    def _sync_frames_to_gauge(self):
        """Sync agent omega params with lattice gauge vertex frames."""
        with torch.no_grad():
            for i, agent in enumerate(self.system.agents):
                self.gauge_field.vertex_frames.data[i] = agent.omega.data

    def _setup_supports(self, radius: float, sharpness: float):
        """Set up geodesic ball support for each agent."""
        coords = self.manifold.coordinates()
        grid_shape = self.manifold.grid_shape

        if radius == float('inf'):
            # Full support
            for agent in self.system.agents:
                agent.chi = torch.ones(grid_shape)
        else:
            # Place agents at evenly-spaced grid points
            n = self.N_agents
            for i, agent in enumerate(self.system.agents):
                # Pick center from grid
                flat_idx = (i * self.manifold.n_points) // n
                idx = []
                rem = flat_idx
                for s in reversed(grid_shape):
                    idx.insert(0, rem % s)
                    rem //= s
                center = coords[tuple(idx)]
                agent.chi = ball_support(self.manifold, center, radius, sharpness)

    def full_transport(self, i: int, j: int) -> Tensor:
        """Full transport operator with edge twists: Ω̂_ij = Ω_i V_ij Ω_j⁻¹.

        Falls back to vertex-local if edge twist is identity.

        Args:
            i, j: agent indices
        Returns:
            (*grid, K, K) transport operator
        """
        omega_i = self.system.agents[i].omega
        omega_j = self.system.agents[j].omega

        # Average edge twist across all dimensions
        V = torch.eye(self.K, device=omega_i.device)
        for d in range(self.manifold.dim):
            V_d = self.gauge_field.get_link(i, d)
            V = V + (V_d - torch.eye(self.K, device=V_d.device))

        return omega_i @ V @ torch.linalg.inv(omega_j)

    def pairwise_alignment_with_twists(self) -> Tensor:
        """KL(q_i || Ω̂_ij[q_j]) using full transport with edge twists.

        Returns:
            (N, N, *grid) pairwise alignment energies
        """
        N = self.N_agents
        mu_q = self.system.get_all_mu_q()      # (N, *grid, K)
        sigma_q = self.system.get_all_sigma_q()  # (N, *grid, K, K)

        grid_shape = self.manifold.grid_shape
        E = torch.zeros((N, N) + grid_shape, device=mu_q.device)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                omega_ij = self.full_transport(i, j)
                mu_j_t = (omega_ij @ mu_q[j].unsqueeze(-1)).squeeze(-1)
                sigma_j_t = omega_ij @ sigma_q[j] @ omega_ij.transpose(-2, -1)
                E[i, j] = gaussian_kl(mu_q[i], sigma_q[i], mu_j_t, sigma_j_t)

        return E

    def volume_weighted_free_energy(self,
                                     observations: Optional[Tensor] = None
                                     ) -> Dict[str, Tensor]:
        """Compute the full free energy with volume-weighted integration.

        S = Σ_i ∫ χ_i KL(q_i||p_i) √|g| dc
          + Σ_ij ∫ χ_ij β_ij KL(q_i||Ω̂_ij[q_j]) √|g| dc
          + β_YM · S_YM

        Returns:
            Dict with total, components, attention weights, curvature
        """
        vol = self.vol_form
        N = self.N_agents
        mu_q = self.system.get_all_mu_q()
        sigma_q = self.system.get_all_sigma_q()
        mu_p = self.system.get_all_mu_p()
        sigma_p = self.system.get_all_sigma_p()

        # 1. Self-consistency with volume weighting
        self_kl = torch.zeros((), device=mu_q.device)
        for i in range(N):
            kl_i = gaussian_kl(mu_q[i], sigma_q[i], mu_p[i], sigma_p[i])
            chi_i = self.system.agents[i].chi
            self_kl = self_kl + (kl_i * chi_i * vol).sum()

        # 2. Belief alignment with edge twists
        E_align = self.pairwise_alignment_with_twists()

        # Attention weights (softmax of -E/τ)
        mask = 1.0 - torch.eye(N, device=E_align.device)
        for _ in range(len(self.manifold.grid_shape)):
            mask = mask.unsqueeze(-1)
        logits = -E_align + (mask - 1) * 1e9
        beta = torch.softmax(logits, dim=1) * mask

        # Overlap-weighted alignment
        chi_all = torch.stack([a.chi for a in self.system.agents])
        chi_ij = chi_all.unsqueeze(1) * chi_all.unsqueeze(0)

        align_energy = (beta * E_align * chi_ij * vol).sum()

        # 3. Yang-Mills regularization
        ym_result = self.wilson_action()
        ym_action = ym_result['action']

        # Total
        total = self_kl + align_energy + ym_action

        return {
            'total': total,
            'self_consistency': self_kl,
            'belief_alignment': align_energy,
            'yang_mills': ym_action,
            'mean_plaquette': ym_result['mean_plaquette'],
            'curvature_per_agent': ym_result['curvature_per_agent'],
            'attention': beta.detach(),
            'alignment_energies': E_align.detach(),
        }

    def holonomy_spectrum(self) -> Dict[str, Tensor]:
        """Compute holonomy diagnostics for all agents.

        Returns:
            Dict with Wilson loop traces, curvature norms
        """
        diagnostics = {}
        for i in range(self.N_agents):
            wl_traces = []
            curvatures = []
            for d1 in range(self.manifold.dim):
                for d2 in range(d1 + 1, self.manifold.dim):
                    wl = self.gauge_field.wilson_loop_trace(i, d1, d2)
                    wl_traces.append(wl.mean().item())

            curv = self.gauge_field.curvature_norm(i)
            diagnostics[f'agent_{i}'] = {
                'wilson_loop_traces': wl_traces,
                'mean_curvature': curv.mean().item(),
                'max_curvature': curv.max().item(),
            }

        return diagnostics

    def evolve(self, n_steps: int, lr: float = 0.01,
               observations: Optional[Tensor] = None,
               verbose: bool = True) -> List[Dict]:
        """Evolve the full manifold-aware system.

        Uses Adam optimizer on all parameters (beliefs, priors, frames, twists).
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = []

        for t in range(n_steps):
            optimizer.zero_grad()
            result = self.volume_weighted_free_energy(observations)
            result['total'].backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            optimizer.step()

            info = {k: v.item() if isinstance(v, Tensor) and v.dim() == 0 else v
                    for k, v in result.items()
                    if k not in ('attention', 'alignment_energies')}
            info['step'] = t
            history.append(info)

            if verbose and t % max(n_steps // 10, 1) == 0:
                print(f"  Step {t:4d} | S = {info['total']:.4f} | "
                      f"KL = {info['self_consistency']:.4f} | "
                      f"Align = {info['belief_alignment']:.4f} | "
                      f"YM = {info['yang_mills']:.4f} | "
                      f"⟨W⟩ = {info['mean_plaquette']:.4f}")

        return history
