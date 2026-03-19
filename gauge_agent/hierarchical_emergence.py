"""
Hierarchical Emergence via Soft Condensation
==============================================

Agents don't form hard clusters — they have SOFT membership in multiple
meta-agents simultaneously, like a person belonging to a family, a company,
a city, and a nation all at once.

The key objects:

  W_{iα}(c) ∈ [0,1]  — agent i's membership in meta-agent α
    Derived from model alignment: w_{iα} = σ(-KL(s_i || Ω̃_{iα}[s_α]) / τ)
    NOT normalized: agent can be fully in multiple groups

  Precision pooling (how meta-agents form):
    Λ_q^α = Σ_i w_{iα} Ω_{αi} Λ_q^i Ω_{αi}^T
    μ_q^α = (Λ_q^α)⁻¹ Σ_i w_{iα} Ω_{αi} Λ_q^i μ_q^i

  Condensation order parameter:
    Ψ_α = mean variance of Ω̃_{αi}[s_i] within cluster α
    Small Ψ → condensed (agents agree on model)

  Cross-scale VFE (differentiable):
    S_cross = Σ_{i,α} w_{iα} KL(p_i || Ω_{iα}[q_α])      (belief top-down)
            + Σ_{i,α} w_{iα} KL(r_i || Ω̃_{iα}[s_α])      (model top-down)

This is a gauge-theoretic BCS: when τ drops below τ_c, agents condense into
meta-agents with emergent properties. The soft membership matrix W is the
order parameter of the phase transition.

Reference: Dennis (2026), Sections 4.2-4.6
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, List, Tuple

from gauge_agent.agents import Agent, MultiAgentSystem
from gauge_agent.statistical_manifold import gaussian_kl


# ─────────────────────────────────────────────────────────────
#  Soft Membership: the condensation order parameter
# ─────────────────────────────────────────────────────────────

class SoftMembership(nn.Module):
    """Computes differentiable soft membership W_{iα} from model alignment.

    Each agent i has membership w_{iα} ∈ [0,1] in meta-agent α,
    derived from how well their models align after gauge transport:

        w_{iα} = σ(-KL(s_i || Ω̃_{iα}[s_α]) / τ)

    Key properties:
      - w_{iα} is NOT normalized across α: you can be in multiple groups
      - w_{iα} is differentiable: the hierarchy is end-to-end trainable
      - As τ → 0: soft membership sharpens to hard clustering
      - As τ → ∞: all memberships → 0.5 (uniform)

    This is the BCS order parameter for agent condensation.

    Args:
        tau: temperature controlling membership sharpness
    """

    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau

    def compute(self, children: MultiAgentSystem,
                parents: MultiAgentSystem) -> Tensor:
        """Compute soft membership matrix W_{iα}.

        Uses model fiber alignment: how well does child i's model s_i
        align with parent α's model s_α after transport through Ω̃?

        Args:
            children: N-agent system at scale ℓ
            parents: M-agent system at scale ℓ+1 (meta-agents)
        Returns:
            (N, M) soft membership matrix W, entries in [0,1]
        """
        N = children.N_agents
        M = parents.N_agents
        K = children.K
        device = children.agents[0].mu_s.device

        # Get model beliefs and frames
        mu_s_child = children.get_all_mu_s()        # (N, K)
        sigma_s_child = children.get_all_sigma_s()   # (N, K, K)
        omega_child = children.get_all_omega_model()  # (N, K, K)

        mu_s_parent = parents.get_all_mu_s()          # (M, K)
        sigma_s_parent = parents.get_all_sigma_s()     # (M, K, K)
        omega_parent = parents.get_all_omega_model()   # (M, K, K)

        # Transport parent models into each child's frame
        # Ω̃_{iα} = Ω̃_i Ω̃_α⁻¹
        omega_parent_inv = torch.linalg.inv(omega_parent)  # (M, K, K)

        # (N, 1, K, K) @ (1, M, K, K) → (N, M, K, K)
        transport = omega_child.unsqueeze(1) @ omega_parent_inv.unsqueeze(0)

        # Transport parent means: μ' = Ω̃_{iα} μ_s^α
        mu_parent_t = (transport @ mu_s_parent.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        # Transport parent covariances: Σ' = Ω̃_{iα} Σ_s^α Ω̃_{iα}^T
        sigma_parent_exp = sigma_s_parent.unsqueeze(0).expand(N, -1, K, K)
        sigma_parent_t = transport @ sigma_parent_exp @ transport.transpose(-2, -1)

        # Expand child distributions for broadcasting
        mu_child_exp = mu_s_child.unsqueeze(1).expand(-1, M, K)
        sigma_child_exp = sigma_s_child.unsqueeze(1).expand(-1, M, K, K)

        # KL(s_i || Ω̃_{iα}[s_α]) for all (i, α) pairs
        kl = gaussian_kl(mu_child_exp, sigma_child_exp,
                         mu_parent_t, sigma_parent_t)  # (N, M)

        # Soft membership via sigmoid
        W = torch.sigmoid(-kl / self.tau)  # (N, M)
        return W


# ─────────────────────────────────────────────────────────────
#  Precision Pooling: how meta-agents emerge from components
# ─────────────────────────────────────────────────────────────

def precision_pool(mu_children: Tensor, sigma_children: Tensor,
                   weights: Tensor, transport: Tensor,
                   regularize: float = 1e-4) -> Tuple[Tensor, Tensor]:
    """Gauge-covariant precision-weighted pooling.

    Given N children with beliefs (μ_i, Σ_i), soft weights w_i,
    and transport operators Ω_{αi}, compute the pooled meta-agent:

        Λ_α = Σ_i w_i Ω_{αi} Λ_i Ω_{αi}^T        (precision adds)
        μ_α = Λ_α⁻¹ Σ_i w_i Ω_{αi} Λ_i μ_i       (precision-weighted mean)

    This is the CORRECT way to combine Gaussians: precisions add,
    not covariances average. More certain agents contribute more.

    Args:
        mu_children: (N, K) child means
        sigma_children: (N, K, K) child covariances
        weights: (N,) soft membership weights w_i ∈ [0,1]
        transport: (N, K, K) transport Ω_{αi} from child i to meta-agent frame
        regularize: minimum eigenvalue for numerical stability
    Returns:
        (mu_pooled, sigma_pooled): (K,), (K, K) meta-agent distribution
    """
    K = mu_children.shape[-1]
    device = mu_children.device

    # Child precisions
    # Use solve for stability: Λ_i = Σ_i⁻¹
    Lambda_children = torch.linalg.inv(sigma_children)  # (N, K, K)

    # Transport precisions: Ω_{αi} Λ_i Ω_{αi}^T
    Lambda_transported = transport @ Lambda_children @ transport.transpose(-2, -1)

    # Weighted sum of transported precisions
    w = weights.view(-1, 1, 1)  # (N, 1, 1)
    Lambda_pooled = (w * Lambda_transported).sum(dim=0)  # (K, K)

    # Regularize: ensure minimum eigenvalue
    Lambda_pooled = 0.5 * (Lambda_pooled + Lambda_pooled.T)
    evals, evecs = torch.linalg.eigh(Lambda_pooled)
    evals = evals.clamp(min=regularize)
    Lambda_pooled = evecs @ torch.diag_embed(evals) @ evecs.T

    # Precision-weighted mean
    mu_transported = (transport @ mu_children.unsqueeze(-1)).squeeze(-1)  # (N, K)
    info_sum = (w.squeeze(-1) * (Lambda_transported @ mu_transported.unsqueeze(-1)).squeeze(-1)).sum(dim=0)

    sigma_pooled = torch.linalg.inv(Lambda_pooled)
    mu_pooled = sigma_pooled @ info_sum

    return mu_pooled, sigma_pooled


# ─────────────────────────────────────────────────────────────
#  Condensation Order Parameter
# ─────────────────────────────────────────────────────────────

class CondensationDiagnostics(nn.Module):
    """Measures how "condensed" each meta-agent is.

    The order parameter Ψ_α measures model coherence within cluster α:

        Ψ_α = (1/Z_α) Σ_i w_{iα} ‖Ω̃_{αi}[s_i] - s̄_α‖²

    where s̄_α is the precision-pooled model mean.

    Small Ψ_α → agents agree on the model → condensed (like Cooper pairs)
    Large Ψ_α → agents disagree → uncondensed (like free electrons)

    The condensation fraction f = (number of condensed meta-agents) / M
    is the macroscopic order parameter for the phase transition.
    """

    def __init__(self, condensation_threshold: float = 0.1):
        super().__init__()
        self.condensation_threshold = condensation_threshold

    @torch.no_grad()
    def order_parameter(self, children: MultiAgentSystem,
                        parents: MultiAgentSystem,
                        W: Tensor) -> Tensor:
        """Compute Ψ_α for each meta-agent.

        Args:
            children: N-agent system at scale ℓ
            parents: M-agent system at scale ℓ+1
            W: (N, M) soft membership matrix
        Returns:
            (M,) order parameter for each meta-agent
        """
        N = children.N_agents
        M = parents.N_agents
        K = children.K

        mu_s = children.get_all_mu_s()           # (N, K)
        omega_child = children.get_all_omega_model()  # (N, K, K)
        omega_parent = parents.get_all_omega_model()  # (M, K, K)

        psi = torch.zeros(M, device=mu_s.device)

        for alpha in range(M):
            omega_alpha_inv = torch.linalg.inv(omega_parent[alpha])

            # Transport all children into parent frame
            transported_means = []
            ws = []
            for i in range(N):
                w = W[i, alpha]
                if w < 1e-6:
                    continue
                omega_ai = omega_parent[alpha] @ torch.linalg.inv(omega_child[i])
                mu_t = omega_ai @ mu_s[i]
                transported_means.append(mu_t)
                ws.append(w)

            if len(transported_means) < 2:
                psi[alpha] = 0.0
                continue

            means = torch.stack(transported_means)  # (n, K)
            ws_t = torch.stack(ws)                    # (n,)

            # Weighted mean
            Z = ws_t.sum()
            mu_bar = (ws_t.unsqueeze(-1) * means).sum(0) / Z

            # Weighted variance
            diff = means - mu_bar.unsqueeze(0)
            var = (ws_t.unsqueeze(-1) * diff ** 2).sum(0) / Z
            psi[alpha] = var.sum()  # trace of variance

        return psi

    @torch.no_grad()
    def condensation_fraction(self, children: MultiAgentSystem,
                               parents: MultiAgentSystem,
                               W: Tensor) -> float:
        """Fraction of meta-agents that are condensed.

        This is the macroscopic order parameter f ∈ [0,1]:
          f = 0: no emergence (paramagnetic)
          f = 1: full hierarchy (ordered)
          0 < f < 1: partial condensation (mixed phase)
        """
        psi = self.order_parameter(children, parents, W)
        return (psi < self.condensation_threshold).float().mean().item()


# ─────────────────────────────────────────────────────────────
#  Scale: one level of the soft hierarchy
# ─────────────────────────────────────────────────────────────

class SoftScale:
    """One level in the soft hierarchy.

    Contains agents at this scale plus the soft membership matrix
    connecting them to the next scale up.
    """

    def __init__(self, scale: int, system: MultiAgentSystem):
        self.scale = scale
        self.system = system
        self.W_up: Optional[Tensor] = None  # (N, M) membership in parent scale


# ─────────────────────────────────────────────────────────────
#  Cross-Scale VFE: differentiable coupling between scales
# ─────────────────────────────────────────────────────────────

class CrossScaleVFE(nn.Module):
    """Differentiable cross-scale free energy coupling.

    The cross-scale VFE penalizes children whose priors deviate from
    the transported meta-agent beliefs, weighted by membership:

        S_cross = Σ_{i,α} w_{iα} [
            KL(p_i || Ω_{iα}[q_α])      (belief top-down)
          + KL(r_i || Ω̃_{iα}[s_α])      (model top-down)
        ]

    This is fully differentiable — gradients flow through:
      w_{iα} (membership), q_α/s_α (meta-agent beliefs),
      p_i/r_i (child priors), Ω_i/Ω̃_i (gauge frames).

    The meta-agent doesn't just impose priors; it provides a gravitational
    well in KL-space that children fall toward. Stronger membership →
    stronger pull.

    Args:
        lambda_belief_topdown: weight for belief top-down coupling
        lambda_model_topdown: weight for model top-down coupling
    """

    def __init__(self, lambda_belief_topdown: float = 1.0,
                 lambda_model_topdown: float = 1.0):
        super().__init__()
        self.lambda_belief_topdown = lambda_belief_topdown
        self.lambda_model_topdown = lambda_model_topdown

    def forward(self, children: MultiAgentSystem,
                parents: MultiAgentSystem,
                W: Tensor) -> Dict[str, Tensor]:
        """Compute cross-scale VFE contribution.

        Args:
            children: N-agent system at scale ℓ
            parents: M-agent system at scale ℓ+1
            W: (N, M) soft membership matrix
        Returns:
            Dict with belief_topdown, model_topdown, total
        """
        N = children.N_agents
        M = parents.N_agents
        K = children.K
        device = W.device

        # ── Belief fiber: KL(p_i || Ω_{iα}[q_α]) ──
        mu_p = children.get_all_mu_p()          # (N, K)
        sigma_p = children.get_all_sigma_p()     # (N, K, K)
        mu_q_parent = parents.get_all_mu_q()     # (M, K)
        sigma_q_parent = parents.get_all_sigma_q()  # (M, K, K)
        omega_child = children.get_all_omega()    # (N, K, K)
        omega_parent = parents.get_all_omega()    # (M, K, K)

        # Transport: Ω_{iα} = Ω_i Ω_α⁻¹
        omega_parent_inv = torch.linalg.inv(omega_parent)
        transport_belief = omega_child.unsqueeze(1) @ omega_parent_inv.unsqueeze(0)

        # Transport parent beliefs: Ω_{iα}[q_α]
        mu_q_t = (transport_belief @ mu_q_parent.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        sigma_q_exp = sigma_q_parent.unsqueeze(0).expand(N, -1, K, K)
        sigma_q_t = transport_belief @ sigma_q_exp @ transport_belief.transpose(-2, -1)

        # KL(p_i || Ω_{iα}[q_α]) for all (i, α)
        mu_p_exp = mu_p.unsqueeze(1).expand(-1, M, K)
        sigma_p_exp = sigma_p.unsqueeze(1).expand(-1, M, K, K)
        kl_belief = gaussian_kl(mu_p_exp, sigma_p_exp, mu_q_t, sigma_q_t)

        belief_topdown = (W * kl_belief).sum()

        # ── Model fiber: KL(r_i || Ω̃_{iα}[s_α]) ──
        mu_r = children.get_all_mu_r()           # (N, K)
        sigma_r = children.get_all_sigma_r()      # (N, K, K)
        mu_s_parent = parents.get_all_mu_s()      # (M, K)
        sigma_s_parent = parents.get_all_sigma_s()  # (M, K, K)
        omega_m_child = children.get_all_omega_model()   # (N, K, K)
        omega_m_parent = parents.get_all_omega_model()  # (M, K, K)

        omega_m_parent_inv = torch.linalg.inv(omega_m_parent)
        transport_model = omega_m_child.unsqueeze(1) @ omega_m_parent_inv.unsqueeze(0)

        mu_s_t = (transport_model @ mu_s_parent.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        sigma_s_exp = sigma_s_parent.unsqueeze(0).expand(N, -1, K, K)
        sigma_s_t = transport_model @ sigma_s_exp @ transport_model.transpose(-2, -1)

        mu_r_exp = mu_r.unsqueeze(1).expand(-1, M, K)
        sigma_r_exp = sigma_r.unsqueeze(1).expand(-1, M, K, K)
        kl_model = gaussian_kl(mu_r_exp, sigma_r_exp, mu_s_t, sigma_s_t)

        model_topdown = (W * kl_model).sum()

        total = (self.lambda_belief_topdown * belief_topdown
                 + self.lambda_model_topdown * model_topdown)

        return {
            'total': total,
            'belief_topdown': belief_topdown,
            'model_topdown': model_topdown,
        }


# ─────────────────────────────────────────────────────────────
#  HierarchicalEmergence: the full soft multi-scale system
# ─────────────────────────────────────────────────────────────

class HierarchicalEmergence(nn.Module):
    """Multi-scale soft hierarchy with emergent meta-agents.

    Unlike the hard-clustering OuroborosTower, this system:
      - Uses SOFT membership (agents in multiple meta-agents)
      - Is FULLY DIFFERENTIABLE (end-to-end trainable)
      - Has a PHASE TRANSITION (condensation at critical τ)
      - Supports OVERLAPPING membership (family + company + nation)

    Architecture at each scale ℓ:
      - MultiAgentSystem with N_ℓ agents
      - Soft membership W^ℓ ∈ [0,1]^{N_ℓ × N_{ℓ+1}}
      - Meta-agents at ℓ+1 get precision-pooled parameters from ℓ

    The total VFE over the hierarchy:
      S_total = Σ_ℓ S_ℓ[system_ℓ] + Σ_ℓ λ_cross · S_cross(ℓ, ℓ+1, W^ℓ)

    where S_ℓ is the within-scale VFE (Eq. 24) and S_cross is the
    cross-scale coupling via soft membership.

    Args:
        base_system: scale-0 MultiAgentSystem
        n_meta_per_scale: how many meta-agents at each scale
                          (list, or single int for uniform reduction)
        tau: condensation temperature
        lambda_cross: weight for cross-scale coupling
        lambda_belief_topdown: weight for belief top-down in cross-scale VFE
        lambda_model_topdown: weight for model top-down in cross-scale VFE
    """

    def __init__(self, base_system: MultiAgentSystem,
                 n_meta_per_scale: Optional[List[int]] = None,
                 tau: float = 1.0,
                 lambda_cross: float = 1.0,
                 lambda_belief_topdown: float = 1.0,
                 lambda_model_topdown: float = 1.0):
        super().__init__()

        self.tau = tau
        self.lambda_cross = lambda_cross

        N = base_system.N_agents
        K = base_system.K
        grid_shape = base_system.grid_shape

        # Build scale hierarchy
        if n_meta_per_scale is None:
            # Default: halve at each scale until we reach 1
            n_meta_per_scale = []
            n = N
            while n > 1:
                n = max(n // 2, 1)
                n_meta_per_scale.append(n)

        # Scale 0 = base system
        self.scales = nn.ModuleList([base_system])

        # Create meta-agent systems at higher scales
        for n_meta in n_meta_per_scale:
            meta_system = MultiAgentSystem(
                n_meta, K, grid_shape,
                init_belief_scale=0.5,
                init_prior_scale=0.5,
                init_model_scale=0.5,
                init_gauge_scale=0.05,
            )
            self.scales.append(meta_system)

        self.n_levels = len(self.scales)

        # Soft membership computer
        self.membership = SoftMembership(tau=tau)

        # Cross-scale VFE
        self.cross_vfe = CrossScaleVFE(
            lambda_belief_topdown=lambda_belief_topdown,
            lambda_model_topdown=lambda_model_topdown,
        )

        # Diagnostics
        self.condensation = CondensationDiagnostics()

    def compute_all_memberships(self) -> List[Tensor]:
        """Compute soft membership matrices at all scales.

        Returns:
            List of (N_ℓ, N_{ℓ+1}) membership matrices
        """
        memberships = []
        for ell in range(self.n_levels - 1):
            W = self.membership.compute(self.scales[ell], self.scales[ell + 1])
            memberships.append(W)
        return memberships

    def update_meta_agents(self, memberships: Optional[List[Tensor]] = None):
        """Update meta-agent parameters via precision pooling.

        Bottom-up: pool child beliefs weighted by membership into
        meta-agent parameters. This is how the meta-agent "emerges"
        from its components.

        Args:
            memberships: precomputed membership matrices (or computed fresh)
        """
        if memberships is None:
            memberships = self.compute_all_memberships()

        for ell in range(self.n_levels - 1):
            children = self.scales[ell]
            parents = self.scales[ell + 1]
            W = memberships[ell]  # (N_child, N_parent)

            self._pool_into_parents(children, parents, W)

    @torch.no_grad()
    def _pool_into_parents(self, children: MultiAgentSystem,
                            parents: MultiAgentSystem,
                            W: Tensor):
        """Precision-pool child beliefs into parent meta-agents.

        For each meta-agent α, compute pooled (μ, Σ) from its
        soft members and set the meta-agent's parameters.
        """
        N = children.N_agents
        M = parents.N_agents
        K = children.K

        mu_q = children.get_all_mu_q()
        sigma_q = children.get_all_sigma_q()
        omega_belief = children.get_all_omega()

        mu_s = children.get_all_mu_s()
        sigma_s = children.get_all_sigma_s()
        omega_model = children.get_all_omega_model()

        for alpha in range(M):
            parent = parents.agents[alpha]
            w = W[:, alpha]  # (N,)

            # Skip if no significant membership
            if w.sum() < 1e-6:
                continue

            # Belief fiber: pool q_i → q_α
            transport_q = parent.omega.data.unsqueeze(0) @ torch.linalg.inv(omega_belief)
            mu_q_pooled, sigma_q_pooled = precision_pool(
                mu_q, sigma_q, w, transport_q
            )
            parent.mu_q.data.copy_(mu_q_pooled)
            sigma_sym = 0.5 * (sigma_q_pooled + sigma_q_pooled.T)
            evals, evecs = torch.linalg.eigh(sigma_sym)
            evals = evals.clamp(min=1e-4)
            parent._L_q.data.copy_(torch.linalg.cholesky(
                evecs @ torch.diag_embed(evals) @ evecs.T
            ))

            # Model fiber: pool s_i → s_α
            transport_s = parent.omega_model.data.unsqueeze(0) @ torch.linalg.inv(omega_model)
            mu_s_pooled, sigma_s_pooled = precision_pool(
                mu_s, sigma_s, w, transport_s
            )
            parent.mu_s.data.copy_(mu_s_pooled)
            sigma_sym = 0.5 * (sigma_s_pooled + sigma_s_pooled.T)
            evals, evecs = torch.linalg.eigh(sigma_sym)
            evals = evals.clamp(min=1e-4)
            parent._L_s.data.copy_(torch.linalg.cholesky(
                evecs @ torch.diag_embed(evals) @ evecs.T
            ))

    def cross_scale_energy(self,
                            memberships: Optional[List[Tensor]] = None
                            ) -> Dict[str, Tensor]:
        """Compute total cross-scale VFE contribution.

        Args:
            memberships: precomputed membership matrices
        Returns:
            Dict with total cross-scale energy and per-scale breakdown
        """
        if memberships is None:
            memberships = self.compute_all_memberships()

        device = next(self.parameters()).device
        total = torch.tensor(0.0, device=device)
        per_scale = []

        for ell in range(self.n_levels - 1):
            result = self.cross_vfe(
                self.scales[ell], self.scales[ell + 1], memberships[ell]
            )
            total = total + self.lambda_cross * result['total']
            per_scale.append(result)

        return {
            'total': total,
            'per_scale': per_scale,
            'memberships': memberships,
        }

    @torch.no_grad()
    def diagnostics(self) -> Dict:
        """Comprehensive hierarchy diagnostics.

        Returns dict with:
          - scale_agents: [N_0, N_1, ..., N_L]
          - condensation_fractions: [f_0→1, f_1→2, ...]
          - order_parameters: [[Ψ_α for α in scale ℓ+1] for ℓ]
          - membership_stats: [{mean, max, min, active} for each scale]
        """
        info = {
            'n_levels': self.n_levels,
            'scale_agents': [s.N_agents for s in self.scales],
            'tau': self.tau,
        }

        memberships = self.compute_all_memberships()
        info['condensation_fractions'] = []
        info['order_parameters'] = []
        info['membership_stats'] = []

        for ell in range(self.n_levels - 1):
            W = memberships[ell]

            # Condensation
            f = self.condensation.condensation_fraction(
                self.scales[ell], self.scales[ell + 1], W
            )
            info['condensation_fractions'].append(f)

            # Order parameter per meta-agent
            psi = self.condensation.order_parameter(
                self.scales[ell], self.scales[ell + 1], W
            )
            info['order_parameters'].append(psi.tolist())

            # Membership statistics
            info['membership_stats'].append({
                'mean': W.mean().item(),
                'max': W.max().item(),
                'min': W.min().item(),
                'active_meta': (W.sum(dim=0) > 0.5).sum().item(),
            })

        return info

    def summary_string(self) -> str:
        """One-line summary of hierarchy state."""
        diag = self.diagnostics()
        agents = diag['scale_agents']
        fracs = diag['condensation_fractions']
        parts = [f"ℓ{i}:{n}" for i, n in enumerate(agents)]
        frac_str = [f"{f:.2f}" for f in fracs]
        return (f"Hierarchy [{' → '.join(parts)}] "
                f"condensation=[{', '.join(frac_str)}] τ={self.tau:.2f}")
