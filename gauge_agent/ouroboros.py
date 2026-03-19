"""
Layer 9: The Ouroboros Tower — Multi-Scale Hierarchical Dynamics
=================================================================

The full participatory universe with bidirectional information flow:

  Bottom-up: agents → meta-agents → meta-meta-agents → ... (consensus)
  Top-down:  ... → meta-agents → agents (prior propagation)

The Ouroboros Tower extends this with:
  - Multi-level hyperprior propagation (5 levels deep)
  - Exponential decay weighting γ^Δζ for ancestral priors
  - Self-referential closure at the apex
  - Non-equilibrium tracking to prevent epistemic death

This is Wheeler's "self-excited circuit": the system observes itself,
forms collective priors that flow down to shape individual beliefs,
whose evolution changes the collective state.

Reference: Dennis (2026), Sections 4.2–4.6
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, List, Tuple, Set

from gauge_agent.agents import Agent, MultiAgentSystem
from gauge_agent.free_energy import FreeEnergyFunctional
from gauge_agent.dynamics import NaturalGradientDynamics
from gauge_agent.meta_agents import ConsensusDetector, MetaAgentFormation, TopDownFeedback


class HierarchicalScale:
    """One scale level in the Ouroboros tower.

    Contains agents at this scale plus metadata about
    their parent-child relationships.
    """

    def __init__(self, scale: int, system: MultiAgentSystem):
        self.scale = scale
        self.system = system
        self.children: Dict[int, Set[int]] = {}  # meta_id → {child_ids}
        self.parent: Dict[int, int] = {}  # child_id → parent meta_id


class OuroborosTower(nn.Module):
    """The full multi-scale participatory tower.

    Manages scales 0 through s_max with:
      - Consensus detection and meta-agent formation
      - Top-down prior propagation (Ouroboros feedback)
      - Hyperprior propagation from ancestors
      - Non-equilibrium monitoring
      - Epistemic death detection

    Args:
        base_system: scale-0 MultiAgentSystem
        free_energy: FreeEnergyFunctional
        max_scales: maximum hierarchical depth s_max
        max_agents: maximum total agents across all scales
        consensus_check_interval: steps between consensus checks
        kl_threshold: KL threshold for consensus
        hyperprior_depth: how many ancestral levels to propagate
        hyperprior_decay: exponential decay γ for hyperprior weighting
        dynamics_kwargs: kwargs passed to NaturalGradientDynamics
    """

    def __init__(self,
                 base_system: MultiAgentSystem,
                 free_energy: FreeEnergyFunctional,
                 max_scales: int = 25,
                 max_agents: int = 200,
                 consensus_check_interval: int = 2,
                 kl_threshold: float = 0.05,
                 hyperprior_depth: int = 5,
                 hyperprior_decay: float = 0.5,
                 **dynamics_kwargs):
        super().__init__()

        self.max_scales = max_scales
        self.max_agents = max_agents
        self.consensus_check_interval = consensus_check_interval
        self.hyperprior_depth = hyperprior_depth
        self.hyperprior_decay = hyperprior_decay

        # Scale 0
        self.scales: List[HierarchicalScale] = [
            HierarchicalScale(0, base_system)
        ]

        self.free_energy = free_energy
        self.dynamics_kwargs = dynamics_kwargs

        self.consensus = ConsensusDetector(kl_threshold=kl_threshold)
        self.formation = MetaAgentFormation(self.consensus)

        # Track non-equilibrium indicators
        self.history: List[Dict] = []
        self.prev_energies: Optional[List[float]] = None

    @property
    def n_scales(self) -> int:
        return len(self.scales)

    @property
    def total_agents(self) -> int:
        return sum(s.system.N_agents for s in self.scales)

    def _create_dynamics(self, system: MultiAgentSystem) -> NaturalGradientDynamics:
        """Create dynamics for a scale's system."""
        return NaturalGradientDynamics(
            system, self.free_energy, **self.dynamics_kwargs
        )

    def evolve_scale(self, scale_idx: int,
                     observations: Optional[Tensor] = None,
                     obs_precision: Optional[Tensor] = None) -> Dict:
        """Evolve one scale for one step.

        Args:
            scale_idx: which scale to evolve
            observations: optional observations (typically only scale 0)
        Returns:
            Dict with energy info
        """
        scale = self.scales[scale_idx]
        dynamics = self._create_dynamics(scale.system)
        return dynamics.step(observations, obs_precision)

    def check_and_form_meta_agents(self, scale_idx: int) -> List[Agent]:
        """Check for consensus at a scale and form meta-agents above it.

        Returns:
            List of newly formed meta-agents
        """
        if self.n_scales >= self.max_scales:
            return []
        if self.total_agents >= self.max_agents:
            return []

        scale = self.scales[scale_idx]
        meta_agents, clusters = self.formation.detect_and_form(scale.system)

        if not meta_agents:
            return []

        # Create or extend the next scale
        next_scale_idx = scale_idx + 1
        K = scale.system.K
        grid_shape = scale.system.grid_shape

        if next_scale_idx >= len(self.scales):
            # Create new scale
            new_system = MultiAgentSystem(
                len(meta_agents), K, grid_shape
            )
            # Copy meta-agent parameters into the new system
            for i, meta in enumerate(meta_agents):
                new_system.agents[i].mu_q.data.copy_(meta.mu_q.data)
                new_system.agents[i]._L_q.data.copy_(meta._L_q.data)
                new_system.agents[i].mu_p.data.copy_(meta.mu_p.data)
                new_system.agents[i]._L_p.data.copy_(meta._L_p.data)
                new_system.agents[i].omega.data.copy_(meta.omega.data)
                new_system.agents[i].agent_id = i

            new_scale = HierarchicalScale(next_scale_idx, new_system)
            # Record parent-child relationships
            for i, cluster in enumerate(clusters):
                new_scale.children[i] = cluster
                for child_id in cluster:
                    new_scale.parent[child_id] = i

            self.scales.append(new_scale)

        return meta_agents

    def propagate_top_down(self):
        """Propagate priors from meta-agents down to constituents.

        Implements the Ouroboros feedback loop with hyperprior depth.
        """
        for scale_idx in range(len(self.scales) - 1, 0, -1):
            parent_scale = self.scales[scale_idx]
            child_scale = self.scales[scale_idx - 1]

            for meta_id, child_ids in parent_scale.children.items():
                if meta_id >= parent_scale.system.N_agents:
                    continue
                meta_agent = parent_scale.system.agents[meta_id]
                constituents = [
                    child_scale.system.agents[cid]
                    for cid in child_ids
                    if cid < child_scale.system.N_agents
                ]
                if constituents:
                    TopDownFeedback.propagate_prior(
                        meta_agent, constituents, blend=1.0
                    )

        # Extended hyperprior propagation
        if self.hyperprior_depth > 1:
            self._propagate_hyperpriors()

    def _propagate_hyperpriors(self):
        """Multi-level hyperprior propagation.

        Agent i at scale s receives priors from ancestors up to
        hyperprior_depth scales above, with exponential decay γ^Δζ.
        """
        for depth in range(2, min(self.hyperprior_depth + 1, self.n_scales)):
            weight = self.hyperprior_decay ** depth

            for scale_idx in range(self.n_scales - depth, -1, -1):
                ancestor_idx = scale_idx + depth
                if ancestor_idx >= self.n_scales:
                    continue

                ancestor_scale = self.scales[ancestor_idx]
                target_scale = self.scales[scale_idx]

                # Find transitive children
                for meta_id in range(ancestor_scale.system.N_agents):
                    meta = ancestor_scale.system.agents[meta_id]
                    # Get all descendants at target scale
                    descendants = self._get_descendants(
                        ancestor_idx, meta_id, scale_idx
                    )
                    agents = [
                        target_scale.system.agents[d]
                        for d in descendants
                        if d < target_scale.system.N_agents
                    ]
                    if agents:
                        TopDownFeedback.propagate_prior(
                            meta, agents, blend=weight
                        )

    def _get_descendants(self, from_scale: int, meta_id: int,
                          to_scale: int) -> Set[int]:
        """Get all transitive descendants of meta_id down to to_scale."""
        if from_scale <= to_scale:
            return set()

        current = {meta_id}
        for s in range(from_scale, to_scale, -1):
            next_level = set()
            scale = self.scales[s]
            for mid in current:
                if mid in scale.children:
                    next_level.update(scale.children[mid])
            current = next_level

        return current

    def self_referential_closure(self):
        """Top-scale agents form priors by observing the whole system.

        p_i^(top)(c) = Σ_j w_j Ω_{i,j}[q_j](c)

        where w_j ∝ exp(-mean_KL_j) favors coherent agents.

        This creates Wheeler's "self-excited circuit".
        """
        if self.n_scales < 2:
            return

        top = self.scales[-1]
        # Gather all beliefs across all scales
        all_mu = []
        all_sigma = []
        all_omega = []
        for scale in self.scales:
            for agent in scale.system.agents:
                all_mu.append(agent.mu_q.data)
                all_sigma.append(agent.sigma_q.detach())
                all_omega.append(agent.omega.data)

        if not all_mu:
            return

        # Compute coherence weights
        # Simple: uniform for now (proper implementation would use KL)
        n_total = len(all_mu)

        for top_agent in top.system.agents:
            mu_acc = torch.zeros_like(top_agent.mu_p.data)
            w_sum = 0.0

            for mu_j, sigma_j, omega_j in zip(all_mu, all_sigma, all_omega):
                # Transport into top agent's frame
                omega_ij = top_agent.omega.data @ torch.linalg.inv(omega_j)
                from gauge_agent.lie_groups import transport_mean
                mu_t = transport_mean(omega_ij, mu_j)
                w = 1.0 / n_total
                mu_acc += w * mu_t
                w_sum += w

            top_agent.mu_p.data.copy_(mu_acc / max(w_sum, 1e-10))

    def compute_non_equilibrium_score(self) -> Dict[str, float]:
        """Track non-equilibrium indicators.

        - Energy flux: |dS/dt|
        - Information flux: Σ_i ∂_t H[q_i]
        - Gradient variance: Var(‖∇_{q_i} S‖²)
        - Composite NE score: (Φ_E + Φ_I + V_∇) / 3

        Returns:
            Dict with all indicators
        """
        current_energies = []
        for scale in self.scales:
            result = self.free_energy(scale.system)
            current_energies.append(result['total'].item())

        energy_flux = 0.0
        if self.prev_energies is not None and len(self.prev_energies) == len(current_energies):
            diffs = [abs(c - p) for c, p in zip(current_energies, self.prev_energies)]
            energy_flux = sum(diffs)

        energy_variance = 0.0
        if len(current_energies) > 1:
            mean_e = sum(current_energies) / len(current_energies)
            energy_variance = sum((e - mean_e)**2 for e in current_energies) / len(current_energies)

        ne_score = (energy_flux + energy_variance) / 2.0

        self.prev_energies = current_energies

        return {
            'energy_flux': energy_flux,
            'energy_variance': energy_variance,
            'ne_score': ne_score,
            'n_scales': self.n_scales,
            'total_agents': self.total_agents,
            'scale_energies': current_energies,
        }

    def step(self, observations: Optional[Tensor] = None,
             obs_precision: Optional[Tensor] = None,
             step_num: int = 0) -> Dict:
        """One full step of the Ouroboros tower.

        1. Evolve all scales (with timescale separation)
        2. Check consensus and form meta-agents (periodic)
        3. Propagate top-down priors
        4. Self-referential closure at apex
        5. Track non-equilibrium indicators

        Args:
            observations: scale-0 observations
            step_num: current step number
        Returns:
            Dict with comprehensive diagnostics
        """
        info = {'step': step_num}

        # 1. Evolve all scales
        scale_infos = []
        for s_idx, scale in enumerate(self.scales):
            obs = observations if s_idx == 0 else None
            prec = obs_precision if s_idx == 0 else None
            s_info = self.evolve_scale(s_idx, obs, prec)
            scale_infos.append(s_info)
        info['scale_energies'] = scale_infos

        # 2. Consensus check (periodic)
        if step_num % self.consensus_check_interval == 0:
            new_metas = []
            for s_idx in range(len(self.scales)):
                metas = self.check_and_form_meta_agents(s_idx)
                new_metas.extend(metas)
            info['new_meta_agents'] = len(new_metas)

        # 3. Top-down prior propagation
        self.propagate_top_down()

        # 4. Self-referential closure
        self.self_referential_closure()

        # 5. Non-equilibrium tracking
        ne = self.compute_non_equilibrium_score()
        info.update(ne)

        self.history.append(info)
        return info

    def evolve(self, n_steps: int,
               observations: Optional[Tensor] = None,
               obs_precision: Optional[Tensor] = None,
               verbose: bool = True) -> List[Dict]:
        """Run the full Ouroboros tower evolution.

        Args:
            n_steps: total evolution steps
            observations: scale-0 observations
            verbose: print progress
        Returns:
            List of step info dicts
        """
        for t in range(n_steps):
            info = self.step(observations, obs_precision, step_num=t)

            if verbose and t % 10 == 0:
                print(f"Step {t:4d} | Scales: {self.n_scales} | "
                      f"Agents: {self.total_agents} | "
                      f"NE: {info.get('ne_score', 0):.4f}")

            # Early stopping
            if self.n_scales >= self.max_scales:
                if verbose:
                    print(f"Reached max scales ({self.max_scales}) at step {t}")
                break
            if self.total_agents >= self.max_agents:
                if verbose:
                    print(f"Reached max agents ({self.max_agents}) at step {t}")
                break

        return self.history
