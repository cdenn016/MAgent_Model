"""
The Complete Variational Free Energy Functional
==================================================

The manuscript's full VFE (Eq. 24, lines 664-675) with ALL terms:

  S[{q_i}, {p_i}, {φ_i}] =

    Σ_i ∫_C χ_i KL(q_i || p_i) √|g| dc                                [T1: belief self-consistency]
  + Σ_i ∫_C χ_i KL(s_i || r_i) √|g| dc                                [T2: model self-consistency]
  + Σ_ij ∫_C χ_ij β_ij KL(q_i || Ω_ij[q_j]) √|g| dc                  [T3: belief alignment]
  + Σ_ij ∫_C χ_ij γ_ij KL(p_i || Ω̃_ij[p_j]) √|g| dc                  [T4: model alignment]
  - Σ_i ∫_C χ_i E_{q_i}[log p(o|q_i)] √|g| dc                        [T5: observation]
  + λ_φ Σ_i ∫_C ‖∇φ_i‖² √|g| dc                                      [R1: gauge smoothness]
  + λ_F Σ_i ∫_C tr(F_μν F^μν) √|g| dc                                 [R2: Yang-Mills]
  + Σ_i Σ_{k=1}^D ρ^k ∫_C χ_i KL(p_i || Ω_{i,A_k}[q_{A_k}]) √|g| dc [T6: hyperpriors]

where:
  q_i = N(μ_q, Σ_q)        — belief (what agent thinks is true)
  p_i = N(μ_p, Σ_p)        — prior (what agent expects)
  s_i = N(μ_s, Σ_s)        — model belief (agent's model of its model)
  r_i = N(μ_r, Σ_r)        — model prior (expectation for model)
  β_ij = softmax(-KL(q_i || Ω_ij[q_j]) / τ)  — belief attention
  γ_ij = softmax(-KL(p_i || Ω̃_ij[p_j]) / τ)  — model attention
  A_k = ancestor at depth k in the hierarchical tower
  ρ = hyperprior decay (typically 0.5)
  D = hyperprior depth (typically 5)

Note on s_i and r_i (Eq. 650):
  The pointwise VFE has TWO self-consistency terms. In the Gaussian case
  where s_i = p_i and r_i is a fixed hyperprior, term T2 reduces to
  KL(p_i || r_i) — a regularizer preventing priors from drifting too far
  from their initial values. We implement this as a prior regularizer.

Reference: Dennis (2026), Sections 2.10-2.11 (Eqs. 24-25), 4.4-4.5
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple, List
import math

from gauge_agent.statistical_manifold import gaussian_kl
from gauge_agent.agents import Agent, MultiAgentSystem


class FullVFE(nn.Module):
    """The complete variational free energy functional from the manuscript.

    Implements ALL terms from Eq. 24 plus regularizers and hyperpriors,
    with proper volume-weighted integration on curved base manifolds.

    This replaces the simpler FreeEnergyFunctional for production use.

    Args:
        lambda_self: weight for T1 (belief self-consistency)
        lambda_model_self: weight for T2 (model/prior self-consistency)
        lambda_belief: weight for T3 (belief alignment)
        lambda_model: weight for T4 (model alignment)
        lambda_obs: weight for T5 (observation likelihood)
        lambda_smooth: weight for R1 (gauge field smoothness)
        lambda_ym: weight for R2 (Yang-Mills curvature penalty)
        lambda_hyper: base weight for T6 (hyperpriors)
        tau_belief: softmax temperature for β_ij
        tau_model: softmax temperature for γ_ij
        hyperprior_decay: ρ for exponential weighting of ancestral priors
        hyperprior_depth: D, max ancestral depth
    """

    def __init__(self,
                 lambda_self: float = 1.0,
                 lambda_model_self: float = 0.5,
                 lambda_belief: float = 1.0,
                 lambda_model: float = 0.5,
                 lambda_obs: float = 1.0,
                 lambda_smooth: float = 0.01,
                 lambda_ym: float = 0.1,
                 lambda_hyper: float = 0.5,
                 tau_belief: float = 1.0,
                 tau_model: float = 1.0,
                 hyperprior_decay: float = 0.5,
                 hyperprior_depth: int = 5):
        super().__init__()
        self.lambda_self = lambda_self
        self.lambda_model_self = lambda_model_self
        self.lambda_belief = lambda_belief
        self.lambda_model = lambda_model
        self.lambda_obs = lambda_obs
        self.lambda_smooth = lambda_smooth
        self.lambda_ym = lambda_ym
        self.lambda_hyper = lambda_hyper
        self.tau_belief = tau_belief
        self.tau_model = tau_model
        self.hyperprior_decay = hyperprior_decay
        self.hyperprior_depth = hyperprior_depth

    # ─────────────────────────────────────────────────────────
    # Attention weights (softmax over KL divergence)
    # ─────────────────────────────────────────────────────────

    def _softmax_attention(self, E: Tensor, tau: float) -> Tensor:
        """Compute softmax attention weights from pairwise KL.

        β_ij(c) = exp(-E_ij(c)/τ) / Σ_k exp(-E_ik(c)/τ)

        Args:
            E: (N, N, *grid) pairwise KL divergences
            tau: temperature
        Returns:
            (N, N, *grid) attention weights (row-normalized)
        """
        N = E.shape[0]
        mask = 1.0 - torch.eye(N, device=E.device, dtype=E.dtype)
        for _ in range(E.dim() - 2):
            mask = mask.unsqueeze(-1)

        logits = -E / max(tau, 1e-6)
        logits = logits + (mask - 1.0) * 1e9
        weights = torch.softmax(logits, dim=1) * mask
        return weights

    # ─────────────────────────────────────────────────────────
    # T1: Belief self-consistency KL(q_i || p_i)
    # ─────────────────────────────────────────────────────────

    def belief_self_consistency(self, system: MultiAgentSystem,
                                 chi: Optional[Tensor] = None,
                                 vol: Optional[Tensor] = None) -> Tensor:
        """T1: Σ_i ∫ χ_i KL(q_i || p_i) √|g| dc.

        Occam penalty: beliefs should not deviate too far from priors.
        """
        mu_q = system.get_all_mu_q()
        sigma_q = system.get_all_sigma_q()
        mu_p = system.get_all_mu_p()
        sigma_p = system.get_all_sigma_p()

        kl = gaussian_kl(mu_q, sigma_q, mu_p, sigma_p)  # (N, *grid)

        if chi is not None:
            kl = kl * chi
        if vol is not None:
            kl = kl * vol
        return kl.sum()

    # ─────────────────────────────────────────────────────────
    # T2: Model self-consistency KL(s_i || r_i)
    # ─────────────────────────────────────────────────────────

    def model_self_consistency(self, system: MultiAgentSystem,
                                model_priors: Optional[Dict[int, Tuple[Tensor, Tensor]]] = None,
                                chi: Optional[Tensor] = None,
                                vol: Optional[Tensor] = None) -> Tensor:
        """T2: Σ_i ∫ χ_i KL(p_i || r_i) √|g| dc.

        Regularizer on priors: generative models should not drift
        too far from their initial hyperpriors r_i.

        In the manuscript's notation: s_i is the "model belief" (the agent's
        current model) and r_i is the "model prior" (what the model should be).
        In our implementation: s_i = p_i (current prior) and r_i is a fixed
        reference prior stored in model_priors.

        If model_priors is None, we use the initial prior values stored
        in each agent's initial state (effectively no penalty).

        Args:
            system: MultiAgentSystem
            model_priors: dict mapping agent_id → (mu_r, sigma_r)
                         the fixed hyperprior for each agent's model
            chi: (N, *grid) support
            vol: (*grid) volume form
        Returns:
            Scalar model self-consistency energy
        """
        if model_priors is None:
            return torch.tensor(0.0, device=next(system.parameters()).device)

        total = torch.tensor(0.0, device=next(system.parameters()).device)
        for i, agent in enumerate(system.agents):
            if agent.agent_id in model_priors:
                mu_r, sigma_r = model_priors[agent.agent_id]
                kl = gaussian_kl(agent.mu_p.unsqueeze(0), agent.sigma_p.unsqueeze(0),
                                 mu_r.unsqueeze(0), sigma_r.unsqueeze(0))
                if chi is not None:
                    kl = kl * chi[i:i+1]
                if vol is not None:
                    kl = kl * vol
                total = total + kl.sum()

        return total

    # ─────────────────────────────────────────────────────────
    # T3: Belief alignment KL(q_i || Ω_ij[q_j])
    # ─────────────────────────────────────────────────────────

    def belief_alignment(self, system: MultiAgentSystem,
                          transport_fn=None,
                          chi_ij: Optional[Tensor] = None,
                          vol: Optional[Tensor] = None
                          ) -> Tuple[Tensor, Tensor, Tensor]:
        """T3: Σ_ij ∫ χ_ij β_ij KL(q_i || Ω_ij[q_j]) √|g| dc.

        Args:
            system: MultiAgentSystem
            transport_fn: callable(i, j) → (*grid, K, K) transport operator.
                         If None, uses vertex-local Ω_i Ω_j⁻¹.
            chi_ij: (N, N, *grid) overlap
            vol: (*grid) volume form
        Returns:
            (energy, beta, E_pairwise)
        """
        N = system.N_agents
        mu_q = system.get_all_mu_q()
        sigma_q = system.get_all_sigma_q()

        E = self._compute_pairwise_kl(mu_q, sigma_q, system, transport_fn)

        beta = self._softmax_attention(E, self.tau_belief)

        weighted = beta * E
        if chi_ij is not None:
            weighted = weighted * chi_ij
        if vol is not None:
            weighted = weighted * vol
        return weighted.sum(), beta.detach(), E.detach()

    # ─────────────────────────────────────────────────────────
    # T4: Model alignment KL(p_i || Ω̃_ij[p_j])
    # ─────────────────────────────────────────────────────────

    def model_alignment(self, system: MultiAgentSystem,
                         transport_fn=None,
                         chi_ij: Optional[Tensor] = None,
                         vol: Optional[Tensor] = None
                         ) -> Tuple[Tensor, Tensor, Tensor]:
        """T4: Σ_ij ∫ χ_ij γ_ij KL(p_i || Ω̃_ij[p_j]) √|g| dc.

        Model-model alignment: agents align generative models,
        forming shared ontologies. This is what makes science possible —
        agents must agree not just on beliefs but on the MODEL of reality.

        The transport Ω̃_ij may differ from Ω_ij (different fiber for models
        vs beliefs). In the current implementation, we use the same transport.

        Args:
            system: MultiAgentSystem
            transport_fn: callable(i, j) → (*grid, K, K) model transport.
            chi_ij: (N, N, *grid) overlap
            vol: (*grid) volume form
        Returns:
            (energy, gamma, E_pairwise)
        """
        N = system.N_agents
        mu_p = system.get_all_mu_p()
        sigma_p = system.get_all_sigma_p()

        E = self._compute_pairwise_kl(mu_p, sigma_p, system, transport_fn)

        gamma = self._softmax_attention(E, self.tau_model)

        weighted = gamma * E
        if chi_ij is not None:
            weighted = weighted * chi_ij
        if vol is not None:
            weighted = weighted * vol
        return weighted.sum(), gamma.detach(), E.detach()

    # ─────────────────────────────────────────────────────────
    # T5: Observation likelihood
    # ─────────────────────────────────────────────────────────

    def observation_term(self, system: MultiAgentSystem,
                          observations: Tensor,
                          obs_precision: Optional[Tensor] = None,
                          chi: Optional[Tensor] = None,
                          vol: Optional[Tensor] = None) -> Tensor:
        """T5: -Σ_i ∫ χ_i E_{q_i}[log p(o|q_i)] √|g| dc.

        For Gaussian likelihood p(o|θ) = N(o; θ, Σ_o):
        -E_q[log p(o|θ)] = (1/2)(o-μ)ᵀΛ_o(o-μ) + (1/2)tr(Λ_oΣ) + const

        Args:
            system: MultiAgentSystem
            observations: (N, *grid, K) or (*grid, K)
            obs_precision: (K, K) or scalar
            chi: (N, *grid) support
            vol: (*grid) volume form
        Returns:
            Scalar observation energy
        """
        mu_q = system.get_all_mu_q()
        sigma_q = system.get_all_sigma_q()
        diff = observations - mu_q

        if obs_precision is None:
            mahal = (diff * diff).sum(-1)
            trace_term = sigma_q.diagonal(dim1=-2, dim2=-1).sum(-1)
        elif obs_precision.dim() == 0:
            mahal = obs_precision * (diff * diff).sum(-1)
            trace_term = obs_precision * sigma_q.diagonal(dim1=-2, dim2=-1).sum(-1)
        else:
            mahal = (diff.unsqueeze(-2) @ obs_precision @ diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            trace_term = (obs_precision.unsqueeze(0) * sigma_q).sum(dim=(-2, -1))

        energy = 0.5 * (mahal + trace_term)
        if chi is not None:
            energy = energy * chi
        if vol is not None:
            energy = energy * vol
        return energy.sum()

    # ─────────────────────────────────────────────────────────
    # T6: Hyperprior terms from Ouroboros tower
    # ─────────────────────────────────────────────────────────

    def hyperprior_term(self, system: MultiAgentSystem,
                         ancestors: List[Dict[int, Agent]],
                         chi: Optional[Tensor] = None,
                         vol: Optional[Tensor] = None) -> Tensor:
        """T6: Σ_i Σ_{k=1}^D ρ^k ∫ χ_i KL(p_i || Ω_{i,A_k}[q_{A_k}]) √|g| dc.

        Hyperprior penalty: each agent's prior is penalized for deviating
        from transported beliefs of ancestors at depths 1..D.

        The ancestor structure comes from the Ouroboros tower:
          ancestors[0] = {child_id: parent_agent}     (depth 1, parent)
          ancestors[1] = {child_id: grandparent_agent} (depth 2)
          ...

        Weight ρ^k decays exponentially with generational distance.

        Args:
            system: MultiAgentSystem at the target scale
            ancestors: list of dicts, one per depth level.
                      Each dict maps child_agent_id → ancestor Agent.
            chi: (N, *grid) support
            vol: (*grid) volume form
        Returns:
            Scalar hyperprior energy
        """
        total = torch.tensor(0.0, device=next(system.parameters()).device)

        for depth_idx, ancestor_map in enumerate(ancestors):
            depth = depth_idx + 1  # depth 1 = parent, depth 2 = grandparent, ...
            if depth > self.hyperprior_depth:
                break

            weight = self.hyperprior_decay ** depth

            for child_id, ancestor in ancestor_map.items():
                if child_id >= system.N_agents:
                    continue

                child = system.agents[child_id]

                # Transport ancestor's belief into child's frame
                omega_iA = child.omega.data @ torch.linalg.inv(ancestor.omega.data)
                mu_A_t = (omega_iA @ ancestor.mu_q.data.unsqueeze(-1)).squeeze(-1)
                sigma_A_t = omega_iA @ ancestor.sigma_q @ omega_iA.transpose(-2, -1)

                # KL(p_i || Ω_{i,A}[q_A])
                kl = gaussian_kl(
                    child.mu_p.unsqueeze(0), child.sigma_p.unsqueeze(0),
                    mu_A_t.unsqueeze(0), sigma_A_t.unsqueeze(0)
                )

                if chi is not None:
                    kl = kl * chi[child_id:child_id+1]
                if vol is not None:
                    kl = kl * vol

                total = total + weight * kl.sum()

        return total

    # ─────────────────────────────────────────────────────────
    # R1: Gauge field smoothness regularizer
    # ─────────────────────────────────────────────────────────

    def gauge_smoothness(self, system: MultiAgentSystem,
                          vol: Optional[Tensor] = None) -> Tensor:
        """R1: λ_φ Σ_i ∫ ‖∇φ_i‖² √|g| dc.

        Penalizes rapid variation of gauge frames across the base manifold.
        Computed as finite differences of omega between neighboring grid points.

        Only relevant for dim(C) ≥ 1.
        """
        if not system.grid_shape:
            return torch.tensor(0.0, device=next(system.parameters()).device)

        total = torch.tensor(0.0, device=next(system.parameters()).device)
        for agent in system.agents:
            omega = agent.omega  # (*grid, K, K)
            for d in range(len(system.grid_shape)):
                d_omega = torch.roll(omega, -1, dims=d) - omega
                grad_sq = (d_omega * d_omega).sum(dim=(-2, -1))
                if vol is not None:
                    grad_sq = grad_sq * vol
                total = total + grad_sq.sum()

        return total

    # ─────────────────────────────────────────────────────────
    # R2: Yang-Mills curvature penalty
    # ─────────────────────────────────────────────────────────

    def yang_mills_penalty(self, lattice_gauge=None) -> Tensor:
        """R2: λ_F ∫ tr(F_μν F^μν) √|g| dc.

        Delegates to the lattice gauge field's Wilson action.
        Returns 0 if no lattice gauge field is provided.
        """
        if lattice_gauge is None:
            return torch.tensor(0.0)
        return lattice_gauge.yang_mills_action()

    # ─────────────────────────────────────────────────────────
    # Helper: pairwise KL with transport
    # ─────────────────────────────────────────────────────────

    def _compute_pairwise_kl(self, mu: Tensor, sigma: Tensor,
                              system: MultiAgentSystem,
                              transport_fn=None) -> Tensor:
        """Compute (N, N, *grid) pairwise KL(dist_i || Ω_ij[dist_j]).

        Args:
            mu: (N, *grid, K) means
            sigma: (N, *grid, K, K) covariances
            system: for computing transport operators
            transport_fn: optional custom transport
        Returns:
            (N, N, *grid) pairwise KL
        """
        N = system.N_agents
        grid_shape = system.grid_shape
        device = mu.device

        if transport_fn is not None:
            # Use custom transport (e.g., with edge twists)
            E = torch.zeros((N, N) + grid_shape, device=device)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    omega_ij = transport_fn(i, j)
                    mu_j_t = (omega_ij @ mu[j].unsqueeze(-1)).squeeze(-1)
                    sigma_j_t = omega_ij @ sigma[j] @ omega_ij.transpose(-2, -1)
                    E[i, j] = gaussian_kl(mu[i], sigma[i], mu_j_t, sigma_j_t)
            return E
        else:
            # Vertex-local: Ω_ij = Ω_i Ω_j⁻¹ (vectorized)
            omegas = system.get_all_omega()
            omega_inv = torch.linalg.inv(omegas)
            transport = omegas.unsqueeze(1) @ omega_inv.unsqueeze(0)

            mu_j = mu.unsqueeze(0).expand(N, -1, *mu.shape[1:])
            sigma_j = sigma.unsqueeze(0).expand(N, -1, *sigma.shape[1:])

            mu_t = (transport @ mu_j.unsqueeze(-1)).squeeze(-1)
            sigma_t = transport @ sigma_j @ transport.transpose(-2, -1)

            mu_i = mu.unsqueeze(1).expand(-1, N, *mu.shape[1:])
            sigma_i = sigma.unsqueeze(1).expand(-1, N, *sigma.shape[1:])

            return gaussian_kl(mu_i, sigma_i, mu_t, sigma_t)

    # ─────────────────────────────────────────────────────────
    # Main forward: the complete functional
    # ─────────────────────────────────────────────────────────

    def forward(self, system: MultiAgentSystem,
                observations: Optional[Tensor] = None,
                obs_precision: Optional[Tensor] = None,
                model_priors: Optional[Dict[int, Tuple[Tensor, Tensor]]] = None,
                ancestors: Optional[List[Dict[int, Agent]]] = None,
                transport_fn=None,
                lattice_gauge=None,
                vol: Optional[Tensor] = None
                ) -> Dict[str, Tensor]:
        """Compute the COMPLETE variational free energy.

        S = λ₁·T1 + λ₂·T2 + λ₃·T3 + λ₄·T4 + λ₅·T5 + λ₆·T6 + λ_φ·R1 + λ_F·R2

        Args:
            system: MultiAgentSystem
            observations: (N, *grid, K) or (*grid, K) observation data
            obs_precision: observation precision matrix or scalar
            model_priors: {agent_id: (mu_r, sigma_r)} fixed model hyperpriors
            ancestors: list of {child_id: ancestor_Agent} for hyperprior term
            transport_fn: callable(i,j) → transport operator (for lattice gauge)
            lattice_gauge: LatticeGaugeField for Yang-Mills penalty
            vol: (*grid) volume form √|g| (None → flat metric)
        Returns:
            Dict with all terms, attention weights, and total
        """
        device = next(system.parameters()).device

        # Support and overlap
        chi_list = [a.chi for a in system.agents]
        chi_all = torch.stack(chi_list)  # (N, *grid)

        # For 0D systems, chi is (N, 1) but KL is (N,) — squeeze to match
        grid_shape = system.grid_shape
        if not grid_shape:
            chi_all = chi_all.squeeze(-1)  # (N,)

        chi_ij = chi_all.unsqueeze(1) * chi_all.unsqueeze(0)  # (N, N, *grid)

        # ── T1: belief self-consistency ──
        T1 = self.belief_self_consistency(system, chi_all, vol)

        # ── T2: model self-consistency ──
        T2 = self.model_self_consistency(system, model_priors, chi_all, vol)

        # ── T3: belief alignment ──
        T3, beta, E_belief = self.belief_alignment(
            system, transport_fn, chi_ij, vol
        )

        # ── T4: model alignment ──
        T4, gamma, E_model = self.model_alignment(
            system, transport_fn, chi_ij, vol
        )

        # ── T5: observation ──
        T5 = torch.tensor(0.0, device=device)
        if observations is not None:
            T5 = self.observation_term(system, observations, obs_precision,
                                        chi_all, vol)

        # ── T6: hyperpriors ──
        T6 = torch.tensor(0.0, device=device)
        if ancestors is not None and len(ancestors) > 0:
            T6 = self.hyperprior_term(system, ancestors, chi_all, vol)

        # ── R1: gauge smoothness ──
        R1 = self.gauge_smoothness(system, vol)

        # ── R2: Yang-Mills ──
        R2 = self.yang_mills_penalty(lattice_gauge)

        # ── Total ──
        total = (self.lambda_self * T1
                 + self.lambda_model_self * T2
                 + self.lambda_belief * T3
                 + self.lambda_model * T4
                 + self.lambda_obs * T5
                 + self.lambda_hyper * T6
                 + self.lambda_smooth * R1
                 + self.lambda_ym * R2)

        return {
            'total': total,
            # Individual terms
            'T1_belief_self': T1,
            'T2_model_self': T2,
            'T3_belief_align': T3,
            'T4_model_align': T4,
            'T5_observation': T5,
            'T6_hyperprior': T6,
            'R1_gauge_smooth': R1,
            'R2_yang_mills': R2,
            # Attention weights
            'beta': beta,
            'gamma': gamma,
            # Pairwise energies
            'E_belief_pairwise': E_belief,
            'E_model_pairwise': E_model,
        }

    def summary_string(self, result: Dict) -> str:
        """Format a one-line summary of the VFE components."""
        parts = []
        for key in ['T1_belief_self', 'T2_model_self', 'T3_belief_align',
                     'T4_model_align', 'T5_observation', 'T6_hyperprior',
                     'R1_gauge_smooth', 'R2_yang_mills']:
            val = result[key]
            v = val.item() if isinstance(val, Tensor) else val
            if abs(v) > 1e-6:
                short = key.split('_', 1)[0]
                parts.append(f"{short}={v:.3f}")
        total = result['total'].item() if isinstance(result['total'], Tensor) else result['total']
        return f"S={total:.4f} [{' '.join(parts)}]"


class HierarchicalVFE(nn.Module):
    """Full VFE across the entire Ouroboros tower.

    Computes the sum of per-scale VFE plus inter-scale hyperprior terms:

      S_tower = Σ_s S^(s)[system_s] + Σ_s Σ_k ρ^k · hyperprior_terms

    Args:
        vfe: FullVFE instance (shared across scales)
        hyperprior_decay: ρ
        hyperprior_depth: D
    """

    def __init__(self, vfe: FullVFE):
        super().__init__()
        self.vfe = vfe

    def forward(self, scales: list,
                observations: Optional[Tensor] = None,
                obs_precision: Optional[Tensor] = None,
                model_priors: Optional[Dict[int, Tuple[Tensor, Tensor]]] = None,
                transport_fn=None,
                lattice_gauge=None,
                vol: Optional[Tensor] = None
                ) -> Dict[str, object]:
        """Compute VFE across all scales in the tower.

        Args:
            scales: list of HierarchicalScale objects from OuroborosTower
            observations: scale-0 observations
            obs_precision: observation precision
            model_priors: fixed model hyperpriors for scale-0 agents
            transport_fn: custom transport function
            lattice_gauge: LatticeGaugeField
            vol: volume form
        Returns:
            Dict with per-scale and total VFE
        """
        total = torch.tensor(0.0)
        scale_results = []

        for s_idx, scale in enumerate(scales):
            system = scale.system

            # Observations only at scale 0
            obs = observations if s_idx == 0 else None
            prec = obs_precision if s_idx == 0 else None

            # Build ancestor list for hyperprior terms
            ancestors = self._build_ancestor_list(scales, s_idx)

            result = self.vfe(
                system,
                observations=obs,
                obs_precision=prec,
                model_priors=model_priors if s_idx == 0 else None,
                ancestors=ancestors,
                transport_fn=transport_fn,
                lattice_gauge=lattice_gauge,
                vol=vol,
            )

            total = total + result['total']
            scale_results.append(result)

        return {
            'total': total,
            'per_scale': scale_results,
            'n_scales': len(scales),
        }

    def _build_ancestor_list(self, scales: list, target_scale: int
                              ) -> List[Dict[int, Agent]]:
        """Build the ancestor mapping for hyperprior computation.

        For scale s, ancestors are:
          depth 1: parent meta-agents at scale s+1
          depth 2: grandparent meta-agents at scale s+2
          ...

        Returns:
            List of {child_id: ancestor_Agent}, one per depth level
        """
        ancestors = []
        current_children = {i: i for i in range(scales[target_scale].system.N_agents)}

        for depth in range(1, min(self.vfe.hyperprior_depth + 1, len(scales) - target_scale)):
            ancestor_scale_idx = target_scale + depth
            if ancestor_scale_idx >= len(scales):
                break

            ancestor_scale = scales[ancestor_scale_idx]
            depth_map = {}

            # Map each original child to its ancestor at this depth
            for child_id, intermediate_id in current_children.items():
                # Look up parent of intermediate_id in ancestor_scale
                if hasattr(ancestor_scale, 'parent') and intermediate_id in ancestor_scale.parent:
                    parent_id = ancestor_scale.parent[intermediate_id]
                    if parent_id < ancestor_scale.system.N_agents:
                        depth_map[child_id] = ancestor_scale.system.agents[parent_id]

            if depth_map:
                ancestors.append(depth_map)

            # Update current_children for next depth
            new_children = {}
            for child_id, intermediate_id in current_children.items():
                if hasattr(ancestor_scale, 'parent') and intermediate_id in ancestor_scale.parent:
                    new_children[child_id] = ancestor_scale.parent[intermediate_id]
            current_children = new_children

        return ancestors
