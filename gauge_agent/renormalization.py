"""
Renormalization Group Flow for Multi-Agent Gauge Theory
=========================================================

The meta-agent hierarchy IS a real-space renormalization group.

At each scale s:
  - N(s) agents with beliefs q_i^(s), priors p_i^(s), frames Ω_i^(s)
  - Effective coupling constants g(s) = {λ_self, λ_align, τ, ...}
  - Free energy density f(s) = S(s) / N(s)

Coarse-graining (blocking):
  - Group agents into blocks B_I (Voronoi cells, nearest neighbors, etc.)
  - Each block → single effective agent at scale s+1
  - Effective beliefs: consensus average within block
  - Effective couplings: derived from inter-block interactions

Beta function:
  β(g) = dg / d(ln s) = [g(s+1) - g(s)] / Δ(ln s)

Fixed points:
  β(g*) = 0 → scale-invariant theories

Critical exponents:
  Near g*, linearize: β(g) ≈ M · (g - g*)
  Eigenvalues of M determine relevance:
    λ > 0 → relevant (grows under coarse-graining)
    λ < 0 → irrelevant (shrinks)
    λ = 0 → marginal

The manuscript (Section 6) shows:
  - Three phases: independent → critical → hierarchical condensation
  - Power-law scaling ΔE² ∝ |t - t_c|^{-α} with α ≈ 1.8
  - Bottom-up emergence: scales 1-2 first, then 11-12 last

This module makes that RG structure explicit and computable.

Reference: Dennis (2026), Sections 4, 6; Wilson (1971); Kadanoff (1966)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple, List, Set, Callable
from dataclasses import dataclass, field
import math

from gauge_agent.agents import Agent, MultiAgentSystem
from gauge_agent.free_energy import FreeEnergyFunctional
from gauge_agent.dynamics import NaturalGradientDynamics
from gauge_agent.meta_agents import ConsensusDetector, MetaAgentFormation
from gauge_agent.statistical_manifold import gaussian_kl


# ============================================================
# Data Structures for RG Flow
# ============================================================

@dataclass
class CouplingConstants:
    """Effective coupling constants at one RG scale.

    These are the "coordinates" in coupling constant space.
    The RG flow traces a trajectory through this space.
    """
    mean_self_kl: float = 0.0        # ⟨KL(q_i || p_i)⟩
    mean_alignment: float = 0.0       # ⟨KL(q_i || Ω_ij[q_j])⟩
    mean_prior_alignment: float = 0.0  # ⟨KL(p_i || Ω̃_ij[p_j])⟩
    attention_entropy: float = 0.0     # H[β_ij] (how spread attention is)
    consensus_score: float = 0.0       # mean Γ_ij
    free_energy_density: float = 0.0   # S / N
    correlation_length: float = 0.0    # ξ from KL decay with distance
    order_parameter: float = 0.0       # 1 - mean_alignment / max_alignment

    def to_vector(self) -> Tensor:
        return torch.tensor([
            self.mean_self_kl,
            self.mean_alignment,
            self.mean_prior_alignment,
            self.attention_entropy,
            self.consensus_score,
            self.free_energy_density,
            self.correlation_length,
            self.order_parameter,
        ])

    @staticmethod
    def from_vector(v: Tensor) -> 'CouplingConstants':
        return CouplingConstants(
            mean_self_kl=v[0].item(),
            mean_alignment=v[1].item(),
            mean_prior_alignment=v[2].item(),
            attention_entropy=v[3].item(),
            consensus_score=v[4].item(),
            free_energy_density=v[5].item(),
            correlation_length=v[6].item(),
            order_parameter=v[7].item(),
        )

    @property
    def n_couplings(self) -> int:
        return 8


@dataclass
class RGScale:
    """One scale level in the RG hierarchy."""
    scale: int
    system: MultiAgentSystem
    couplings: CouplingConstants
    blocks: List[Set[int]] = field(default_factory=list)
    parent_map: Dict[int, int] = field(default_factory=dict)


# ============================================================
# Blocking Schemes (Coarse-Graining)
# ============================================================

class BlockingScheme:
    """Real-space blocking (coarse-graining) for multi-agent RG.

    Different blocking schemes give different RG flows but should
    produce the same critical exponents (universality).
    """

    @staticmethod
    def majority_rule(system: MultiAgentSystem,
                       block_size: int = 2) -> List[Set[int]]:
        """Simple majority-rule blocking: group agents into blocks of fixed size.

        For 1D: pairs of adjacent agents
        For 2D: 2×2 blocks on the grid
        For 0D: group by KL proximity

        Args:
            system: the agent system
            block_size: agents per block
        Returns:
            List of agent index sets (blocks)
        """
        N = system.N_agents
        blocks = []
        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            blocks.append(set(range(start, end)))
        return blocks

    @staticmethod
    def kl_proximity(system: MultiAgentSystem,
                      n_blocks: Optional[int] = None,
                      max_kl: float = 5.0) -> List[Set[int]]:
        """Block by KL proximity: group agents with similar beliefs.

        This is the information-geometric natural blocking.

        Args:
            system: the agent system
            n_blocks: target number of blocks (default: N/2)
            max_kl: maximum mean KL within a block
        Returns:
            List of blocks
        """
        N = system.N_agents
        if n_blocks is None:
            n_blocks = max(N // 2, 1)

        # Compute pairwise alignment energies
        with torch.no_grad():
            E = system.pairwise_alignment_energies('belief')
            while E.dim() > 2:
                E = E.mean(-1)
            # Symmetrize
            E = 0.5 * (E + E.T)

        # Agglomerative clustering
        assigned = set()
        blocks = []

        # Sort pairs by KL distance
        pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                pairs.append((E[i, j].item(), i, j))
        pairs.sort()

        # Greedily merge closest pairs
        labels = list(range(N))  # Each agent starts in its own cluster

        def find(x):
            while labels[x] != x:
                labels[x] = labels[labels[x]]
                x = labels[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                labels[rb] = ra

        n_clusters = N
        for kl_val, i, j in pairs:
            if n_clusters <= n_blocks:
                break
            if kl_val > max_kl:
                break
            if find(i) != find(j):
                union(i, j)
                n_clusters -= 1

        # Extract clusters
        cluster_map = {}
        for i in range(N):
            root = find(i)
            if root not in cluster_map:
                cluster_map[root] = set()
            cluster_map[root].add(i)

        return list(cluster_map.values())

    @staticmethod
    def consensus_blocking(system: MultiAgentSystem,
                            detector: ConsensusDetector) -> List[Set[int]]:
        """Block using consensus detection (the natural multi-agent RG).

        This directly uses the meta-agent formation mechanism as blocking.

        Args:
            system: the agent system
            detector: consensus detector with thresholds
        Returns:
            List of blocks (consensus clusters)
        """
        clusters = detector.find_clusters(system)

        # Add singleton blocks for unassigned agents
        assigned = set()
        for c in clusters:
            assigned.update(c)
        for i in range(system.N_agents):
            if i not in assigned:
                clusters.append({i})

        return clusters


# ============================================================
# Coupling Extraction
# ============================================================

class CouplingExtractor:
    """Extract effective coupling constants from a multi-agent system."""

    @staticmethod
    @torch.no_grad()
    def extract(system: MultiAgentSystem,
                free_energy: FreeEnergyFunctional) -> CouplingConstants:
        """Measure all effective couplings at the current state.

        Args:
            system: multi-agent system at this scale
            free_energy: the free energy functional
        Returns:
            CouplingConstants with measured values
        """
        N = system.N_agents
        if N < 2:
            return CouplingConstants()

        # Self-consistency
        self_kls = []
        for agent in system.agents:
            self_kls.append(agent.self_kl().mean().item())
        mean_self_kl = sum(self_kls) / len(self_kls) if self_kls else 0.0

        # Pairwise alignment
        E_belief = system.pairwise_alignment_energies('belief')
        while E_belief.dim() > 2:
            E_belief = E_belief.mean(-1)

        mask = 1.0 - torch.eye(N, device=E_belief.device)
        n_pairs = mask.sum().item()
        mean_alignment = (E_belief * mask).sum().item() / max(n_pairs, 1)

        # Prior alignment
        E_prior = system.pairwise_alignment_energies('prior')
        while E_prior.dim() > 2:
            E_prior = E_prior.mean(-1)
        mean_prior = (E_prior * mask).sum().item() / max(n_pairs, 1)

        # Attention entropy
        logits = -E_belief
        logits = logits + (mask - 1) * 1e9
        for _ in range(2 - logits.dim()):
            logits = logits.unsqueeze(-1)
        beta = torch.softmax(logits.squeeze(), dim=1) * mask
        beta_clamped = beta.clamp(min=1e-10)
        attn_entropy = -(beta_clamped * beta_clamped.log()).sum().item() / N

        # Consensus score
        C_belief = 1.0 - E_belief
        C_prior = 1.0 - E_prior
        consensus = (C_belief * C_prior * mask).sum().item() / max(n_pairs, 1)

        # Free energy density
        result = free_energy(system)
        fe_density = result['total'].item() / N

        # Correlation length (from exponential decay of KL with agent index)
        xi = CouplingExtractor._estimate_correlation_length(E_belief, mask)

        # Order parameter: 1 - normalized alignment
        max_align = E_belief.max().item() if E_belief.max() > 0 else 1.0
        order = 1.0 - mean_alignment / max(max_align, 1e-10)

        return CouplingConstants(
            mean_self_kl=mean_self_kl,
            mean_alignment=mean_alignment,
            mean_prior_alignment=mean_prior,
            attention_entropy=attn_entropy,
            consensus_score=consensus,
            free_energy_density=fe_density,
            correlation_length=xi,
            order_parameter=order,
        )

    @staticmethod
    def _estimate_correlation_length(E: Tensor, mask: Tensor) -> float:
        """Estimate correlation length ξ from KL decay.

        For agents arranged in 1D: KL(i,j) ~ exp(-|i-j|/ξ)
        Fit ξ from the exponential decay of off-diagonal KL.
        """
        N = E.shape[0]
        if N < 4:
            return 1.0

        # Group by distance |i-j|
        distances = {}
        for i in range(N):
            for j in range(i + 1, N):
                d = abs(i - j)
                if d not in distances:
                    distances[d] = []
                distances[d].append(E[i, j].item())

        # Average KL at each distance
        ds = sorted(distances.keys())
        if len(ds) < 2:
            return 1.0

        mean_kls = [sum(distances[d]) / len(distances[d]) for d in ds]

        # Fit log(KL) ~ -d/ξ via linear regression
        log_kls = [math.log(max(kl, 1e-10)) for kl in mean_kls]
        d_vals = [float(d) for d in ds]

        # Simple least squares: slope = -1/ξ
        n = len(d_vals)
        d_mean = sum(d_vals) / n
        log_mean = sum(log_kls) / n
        num = sum((d - d_mean) * (lk - log_mean) for d, lk in zip(d_vals, log_kls))
        den = sum((d - d_mean) ** 2 for d in d_vals)

        if abs(den) < 1e-10 or abs(num) < 1e-10:
            return float(N)

        slope = num / den
        if slope >= 0:
            return float(N)  # No decay → correlation length ≥ system size

        xi = -1.0 / slope
        return max(min(xi, float(N)), 0.1)


# ============================================================
# RG Flow Engine
# ============================================================

class RenormalizationGroupFlow(nn.Module):
    """Real-space renormalization group for multi-agent gauge theory.

    Implements the full RG procedure:
      1. Initialize at scale 0 with N agents
      2. Evolve dynamics for n_equilibrate steps (thermalize)
      3. Measure coupling constants g(s)
      4. Coarse-grain (block) → scale s+1 with N/b agents
      5. Repeat until single effective agent
      6. Compute beta functions, fixed points, critical exponents

    Args:
        N_agents: initial number of agents
        K: fiber dimension
        grid_shape: base manifold discretization
        blocking: 'majority_rule', 'kl_proximity', or 'consensus'
        block_size: agents per block (for majority_rule)
        n_equilibrate: steps per scale before measurement
        max_scales: maximum number of RG scales
        kl_threshold: consensus threshold (for consensus blocking)
    """

    def __init__(self, N_agents: int, K: int,
                 grid_shape: Tuple[int, ...] = (),
                 blocking: str = 'kl_proximity',
                 block_size: int = 2,
                 n_equilibrate: int = 50,
                 max_scales: int = 20,
                 kl_threshold: float = 0.1,
                 lr: float = 0.02):
        super().__init__()
        self.N_agents_initial = N_agents
        self.K = K
        self.grid_shape = grid_shape
        self.blocking_method = blocking
        self.block_size = block_size
        self.n_equilibrate = n_equilibrate
        self.max_scales = max_scales
        self.lr = lr

        self.free_energy = FreeEnergyFunctional(
            lambda_self=1.0, lambda_belief=1.0,
            lambda_prior=0.5, temperature=1.0,
            use_observations=False
        )

        self.consensus = ConsensusDetector(
            kl_threshold=kl_threshold,
            min_cluster_size=2,
            gamma_min=0.3
        )

        self.formation = MetaAgentFormation(self.consensus)
        self.extractor = CouplingExtractor()

        # RG flow trajectory
        self.scales: List[RGScale] = []
        self.beta_functions: List[Tensor] = []  # β = g(s+1) - g(s)

    def _create_system(self, N: int) -> MultiAgentSystem:
        return MultiAgentSystem(N, self.K, self.grid_shape,
                                init_belief_scale=1.0,
                                init_prior_scale=1.0,
                                init_gauge_scale=0.1)

    def _equilibrate(self, system: MultiAgentSystem, n_steps: int):
        """Evolve to local equilibrium before measurement."""
        dynamics = NaturalGradientDynamics(
            system, self.free_energy,
            lr_mu_q=self.lr, lr_sigma_q=self.lr * 0.1,
            lr_mu_p=self.lr * 0.5, lr_sigma_p=self.lr * 0.1,
            lr_omega=self.lr * 0.2
        )
        for _ in range(n_steps):
            dynamics.step()

    def _block(self, system: MultiAgentSystem) -> Tuple[List[Set[int]], MultiAgentSystem]:
        """Coarse-grain the system into blocks → new system."""
        if self.blocking_method == 'majority_rule':
            blocks = BlockingScheme.majority_rule(system, self.block_size)
        elif self.blocking_method == 'kl_proximity':
            blocks = BlockingScheme.kl_proximity(system)
        elif self.blocking_method == 'consensus':
            blocks = BlockingScheme.consensus_blocking(system, self.consensus)
        else:
            raise ValueError(f"Unknown blocking: {self.blocking_method}")

        # Form meta-agents from blocks
        meta_agents = []
        for block in blocks:
            if len(block) >= 2:
                meta = self.formation.form_meta_agent(system, block)
                meta_agents.append(meta)
            else:
                # Singleton: copy agent
                idx = list(block)[0]
                meta_agents.append(system.agents[idx])

        if not meta_agents:
            return blocks, system

        # Create new system with meta-agents
        N_new = len(meta_agents)
        new_system = MultiAgentSystem(N_new, self.K, self.grid_shape)
        for i, meta in enumerate(meta_agents):
            # Belief fiber
            new_system.agents[i].mu_q.data.copy_(meta.mu_q.data)
            new_system.agents[i]._L_q.data.copy_(meta._L_q.data)
            new_system.agents[i].mu_p.data.copy_(meta.mu_p.data)
            new_system.agents[i]._L_p.data.copy_(meta._L_p.data)
            new_system.agents[i].omega.data.copy_(meta.omega.data)
            # Model fiber
            new_system.agents[i].mu_s.data.copy_(meta.mu_s.data)
            new_system.agents[i]._L_s.data.copy_(meta._L_s.data)
            new_system.agents[i].mu_r.data.copy_(meta.mu_r.data)
            new_system.agents[i]._L_r.data.copy_(meta._L_r.data)
            new_system.agents[i].omega_model.data.copy_(meta.omega_model.data)
            new_system.agents[i].agent_id = i

        return blocks, new_system

    def run(self, verbose: bool = True) -> Dict:
        """Execute the full RG flow.

        Returns:
            Dict with coupling trajectories, beta functions,
            fixed points, critical exponents
        """
        self.scales = []
        self.beta_functions = []

        # Initialize
        system = self._create_system(self.N_agents_initial)

        for s in range(self.max_scales):
            N = system.N_agents
            if N < 2:
                if verbose:
                    print(f"  Scale {s}: N={N} < 2, terminating")
                break

            # Equilibrate
            self._equilibrate(system, self.n_equilibrate)

            # Measure couplings
            couplings = self.extractor.extract(system, self.free_energy)

            # Record
            scale = RGScale(scale=s, system=system, couplings=couplings)
            self.scales.append(scale)

            if verbose:
                print(f"  Scale {s:2d} | N={N:4d} | "
                      f"f={couplings.free_energy_density:.4f} | "
                      f"⟨KL⟩={couplings.mean_alignment:.4f} | "
                      f"ξ={couplings.correlation_length:.2f} | "
                      f"Γ={couplings.consensus_score:.4f} | "
                      f"Ψ={couplings.order_parameter:.4f}")

            # Beta function (requires previous scale)
            if len(self.scales) >= 2:
                g_prev = self.scales[-2].couplings.to_vector()
                g_curr = couplings.to_vector()
                beta = g_curr - g_prev
                self.beta_functions.append(beta)

            # Coarse-grain
            blocks, new_system = self._block(system)
            scale.blocks = blocks

            if new_system.N_agents >= N:
                if verbose:
                    print(f"  Blocking did not reduce agents (N={N}), terminating")
                break

            system = new_system

        # Analyze the flow
        analysis = self._analyze_flow()
        if verbose:
            self._print_analysis(analysis)

        return analysis

    def _analyze_flow(self) -> Dict:
        """Analyze the RG flow for fixed points and critical exponents.

        Returns:
            Dict with coupling_trajectory, beta_trajectory,
            fixed_point_candidates, critical_exponents, universality_class
        """
        n_scales = len(self.scales)

        # Coupling trajectory
        coupling_trajectory = torch.stack([
            s.couplings.to_vector() for s in self.scales
        ]) if self.scales else torch.zeros(0, 8)

        # Beta trajectory
        beta_trajectory = torch.stack(self.beta_functions) if self.beta_functions else torch.zeros(0, 8)

        # Fixed point detection: where |β| is minimized
        fixed_points = []
        if len(self.beta_functions) >= 2:
            beta_norms = [b.norm().item() for b in self.beta_functions]
            for i in range(1, len(beta_norms) - 1):
                if beta_norms[i] < beta_norms[i-1] and beta_norms[i] < beta_norms[i+1]:
                    fixed_points.append({
                        'scale': i + 1,  # +1 because beta[0] is between scales 0 and 1
                        'couplings': self.scales[i+1].couplings,
                        'beta_norm': beta_norms[i],
                    })

        # Critical exponents from linearization around fixed points
        critical_exponents = []
        if len(self.beta_functions) >= 3 and fixed_points:
            for fp in fixed_points:
                s = fp['scale']
                if s >= 1 and s < len(self.scales) - 1:
                    # Numerical Jacobian: M_ij ≈ Δβ_i / Δg_j
                    g_before = self.scales[s-1].couplings.to_vector()
                    g_at = self.scales[s].couplings.to_vector()
                    g_after = self.scales[s+1].couplings.to_vector()

                    beta_before = self.beta_functions[s-1] if s-1 < len(self.beta_functions) else torch.zeros(8)
                    beta_after = self.beta_functions[s] if s < len(self.beta_functions) else torch.zeros(8)

                    dg = g_after - g_before
                    dbeta = beta_after - beta_before

                    # Approximate eigenvalues (stability matrix)
                    # Full Jacobian is rank-deficient from 1D trajectory;
                    # use the scalar approximation: λ ≈ |dβ|/|dg|
                    dg_norm = dg.norm().item()
                    if dg_norm > 1e-10:
                        lambda_eff = dbeta.norm().item() / dg_norm
                        # Sign from direction
                        if (dbeta @ dg).item() > 0:
                            lambda_eff = abs(lambda_eff)  # relevant
                        else:
                            lambda_eff = -abs(lambda_eff)  # irrelevant

                        # Correlation length exponent ν from ξ scaling
                        xi_values = [s.couplings.correlation_length for s in self.scales]
                        nu = self._estimate_nu(xi_values, fp['scale'])

                        critical_exponents.append({
                            'scale': s,
                            'lambda_eff': lambda_eff,
                            'nu': nu,
                            'relevant': lambda_eff > 0,
                        })

        # Free energy scaling (should show power-law near critical point)
        fe_density = [s.couplings.free_energy_density for s in self.scales]
        N_agents = [s.system.N_agents for s in self.scales]

        # Scaling dimension: f(s) ~ N(s)^{-d_f}
        d_f = self._estimate_scaling_dimension(fe_density, N_agents)

        return {
            'n_scales': n_scales,
            'coupling_trajectory': coupling_trajectory,
            'beta_trajectory': beta_trajectory,
            'fixed_points': fixed_points,
            'critical_exponents': critical_exponents,
            'fe_density': fe_density,
            'N_agents': N_agents,
            'scaling_dimension': d_f,
            'correlation_lengths': [s.couplings.correlation_length for s in self.scales],
            'order_parameters': [s.couplings.order_parameter for s in self.scales],
        }

    @staticmethod
    def _estimate_nu(xi_values: List[float], fp_scale: int) -> float:
        """Estimate correlation length exponent ν from ξ(s) near fixed point.

        ξ ~ |s - s_c|^{-ν}
        """
        if len(xi_values) < 3 or fp_scale < 1 or fp_scale >= len(xi_values) - 1:
            return 1.0

        # Use three points around the peak
        xi_max = max(xi_values)
        if xi_max < 1e-10:
            return 1.0

        # Log-log fit of |ξ - ξ_max| vs |s - s_c|
        vals = []
        for s, xi in enumerate(xi_values):
            d = abs(s - fp_scale)
            if d > 0 and xi > 0:
                vals.append((math.log(d), math.log(max(xi, 1e-10))))

        if len(vals) < 2:
            return 1.0

        # Linear regression on log-log
        x = [v[0] for v in vals]
        y = [v[1] for v in vals]
        n = len(x)
        mx = sum(x) / n
        my = sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        den = sum((xi - mx) ** 2 for xi in x)
        if abs(den) < 1e-10:
            return 1.0

        slope = num / den
        return abs(slope) if abs(slope) < 10 else 1.0

    @staticmethod
    def _estimate_scaling_dimension(fe: List[float], N: List[int]) -> float:
        """Estimate scaling dimension from f(s) vs N(s).

        f ~ N^{-d_f} → log f = -d_f log N + const
        """
        if len(fe) < 2:
            return 0.0

        vals = [(math.log(max(n, 1)), math.log(max(abs(f), 1e-10)))
                for f, n in zip(fe, N) if n > 0 and abs(f) > 1e-10]

        if len(vals) < 2:
            return 0.0

        x = [v[0] for v in vals]
        y = [v[1] for v in vals]
        n = len(x)
        mx = sum(x) / n
        my = sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        den = sum((xi - mx) ** 2 for xi in x)
        if abs(den) < 1e-10:
            return 0.0

        return -num / den  # d_f = -slope

    def _print_analysis(self, analysis: Dict):
        """Pretty-print RG flow analysis."""
        print(f"\n  {'='*55}")
        print(f"  RENORMALIZATION GROUP FLOW ANALYSIS")
        print(f"  {'='*55}")
        print(f"  Scales traversed: {analysis['n_scales']}")
        print(f"  Agent counts: {analysis['N_agents']}")
        print(f"  Scaling dimension d_f: {analysis['scaling_dimension']:.3f}")

        if analysis['fixed_points']:
            print(f"\n  Fixed Points:")
            for fp in analysis['fixed_points']:
                print(f"    Scale {fp['scale']}: |β| = {fp['beta_norm']:.4f}")
                c = fp['couplings']
                print(f"      f = {c.free_energy_density:.4f}, "
                      f"⟨KL⟩ = {c.mean_alignment:.4f}, "
                      f"ξ = {c.correlation_length:.2f}")
        else:
            print(f"\n  No fixed points detected (flow is monotonic)")

        if analysis['critical_exponents']:
            print(f"\n  Critical Exponents:")
            for ce in analysis['critical_exponents']:
                rel = "relevant" if ce['relevant'] else "irrelevant"
                print(f"    Scale {ce['scale']}: λ = {ce['lambda_eff']:.4f} ({rel}), "
                      f"ν = {ce['nu']:.3f}")

        print(f"\n  Coupling Flow (first → last):")
        if len(analysis['fe_density']) >= 2:
            print(f"    Free energy density: {analysis['fe_density'][0]:.4f} → "
                  f"{analysis['fe_density'][-1]:.4f}")
        if len(analysis['correlation_lengths']) >= 2:
            print(f"    Correlation length:  {analysis['correlation_lengths'][0]:.2f} → "
                  f"{analysis['correlation_lengths'][-1]:.2f}")
        if len(analysis['order_parameters']) >= 2:
            print(f"    Order parameter:     {analysis['order_parameters'][0]:.4f} → "
                  f"{analysis['order_parameters'][-1]:.4f}")


# ============================================================
# Universality Test
# ============================================================

class UniversalityTest:
    """Test universality: different initial conditions → same fixed point.

    Universality is the hallmark of RG: microscopic details wash out
    under coarse-graining, leaving only universal macroscopic behavior.

    Two systems are in the same universality class if they flow
    to the same fixed point under RG.
    """

    @staticmethod
    def compare_flows(flow1: Dict, flow2: Dict,
                       tolerance: float = 0.5) -> Dict:
        """Compare two RG flows for universality.

        Args:
            flow1, flow2: output of RenormalizationGroupFlow.run()
            tolerance: relative tolerance for matching
        Returns:
            Dict with comparison metrics
        """
        # Compare final couplings
        g1_final = flow1['coupling_trajectory'][-1] if len(flow1['coupling_trajectory']) > 0 else torch.zeros(8)
        g2_final = flow2['coupling_trajectory'][-1] if len(flow2['coupling_trajectory']) > 0 else torch.zeros(8)

        coupling_distance = (g1_final - g2_final).norm().item()
        coupling_norm = max(g1_final.norm().item(), g2_final.norm().item(), 1e-10)
        relative_distance = coupling_distance / coupling_norm

        # Compare scaling dimensions
        d_f_diff = abs(flow1['scaling_dimension'] - flow2['scaling_dimension'])

        # Compare critical exponents
        ce1 = flow1['critical_exponents']
        ce2 = flow2['critical_exponents']
        exponent_match = False
        if ce1 and ce2:
            nu_diff = abs(ce1[0]['nu'] - ce2[0]['nu'])
            exponent_match = nu_diff < tolerance

        same_class = relative_distance < tolerance

        return {
            'coupling_distance': coupling_distance,
            'relative_distance': relative_distance,
            'scaling_dim_diff': d_f_diff,
            'exponent_match': exponent_match,
            'same_universality_class': same_class,
        }
