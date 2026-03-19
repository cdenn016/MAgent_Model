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
from gauge_agent.full_vfe import FullVFE


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

        # Full VFE (all 8 terms)
        self.vfe = FullVFE(
            lambda_self=1.0, lambda_model_self=0.5,
            lambda_belief=1.0, lambda_model=0.5,
            lambda_obs=1.0, lambda_smooth=0.01,
            lambda_ym=ym_coupling, lambda_hyper=0.5,
        )

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
                                     observations: Optional[Tensor] = None,
                                     obs_precision: Optional[Tensor] = None,
                                     model_priors: Optional[Dict] = None,
                                     ancestors: Optional[List] = None,
                                     ) -> Dict[str, Tensor]:
        """Compute the COMPLETE variational free energy (Eq. 24).

        All 8 terms from the manuscript:
          T1: belief self-consistency KL(q_i || p_i)
          T2: model self-consistency KL(p_i || r_i)
          T3: belief alignment β_ij KL(q_i || Ω_ij[q_j])
          T4: model alignment γ_ij KL(p_i || Ω̃_ij[p_j])
          T5: observation likelihood
          T6: hyperprior terms from Ouroboros tower
          R1: gauge field smoothness
          R2: Yang-Mills curvature penalty

        All with proper √|g| volume weighting and full lattice gauge transport.

        Returns:
            Dict with total, all 8 components, attention weights
        """
        result = self.vfe(
            self.system,
            observations=observations,
            obs_precision=obs_precision,
            model_priors=model_priors,
            ancestors=ancestors,
            transport_fn=self.full_transport,
            lattice_gauge=self.gauge_field,
            vol=self.vol_form,
        )

        # Add Wilson action diagnostics
        ym_result = self.wilson_action()
        result['mean_plaquette'] = ym_result['mean_plaquette']
        result['curvature_per_agent'] = ym_result['curvature_per_agent']

        return result

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
               obs_precision: Optional[Tensor] = None,
               model_priors: Optional[Dict] = None,
               ancestors: Optional[List] = None,
               verbose: bool = True) -> List[Dict]:
        """Evolve the full manifold-aware system.

        Uses Adam optimizer on all parameters (beliefs, priors, frames, twists).
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = []

        for t in range(n_steps):
            optimizer.zero_grad()
            result = self.volume_weighted_free_energy(
                observations, obs_precision, model_priors, ancestors
            )
            result['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            optimizer.step()

            info = {k: v.item() if isinstance(v, Tensor) and v.dim() == 0 else v
                    for k, v in result.items()
                    if k not in ('beta', 'gamma', 'E_belief_pairwise',
                                 'E_model_pairwise')}
            info['step'] = t
            history.append(info)

            if verbose and t % max(n_steps // 10, 1) == 0:
                summary = self.vfe.summary_string(result)
                plaq = result.get('mean_plaquette', torch.tensor(0.0))
                if isinstance(plaq, Tensor):
                    plaq = plaq.item()
                print(f"  Step {t:4d} | {summary} | ⟨W⟩={plaq:.4f}")

        return history
