"""
Hierarchical Emergence via Soft Condensation
==============================================

Two DISTINCT kinds of alignment:

  MODEL alignment (s_i ↔ s_j) → defines SPECIES
    Agents sharing a generative model: same ontology, same way of
    parsing reality. Humans share models with humans, not algae.
    Evolves SLOWLY (evolutionary timescale, ε << 1 in dynamics).

  BELIEF alignment (q_i ↔ q_j) → defines META-AGENTS (coalitions)
    Agents agreeing on what's happening RIGHT NOW. A team, a flock,
    a synchronized neural ensemble. Changes FAST (perceptual timescale).

The SELECTION RULE: species GATES meta-agent formation.
You can only coordinate with agents that share your model.

    W_{iα} = S_{iσ(α)} · C_{iα}

    S_{iσ} = species gate from model alignment (slow, structural)
    C_{iα} = coalition membership from belief alignment (fast, dynamic)

This is the full hierarchy:

  Species (model fiber, slow):
    genome → cell type → organism type → civilization
    Defines the SPACE of possible meta-agents

  Meta-agents (belief fiber, fast):
    cell coordination → tissue → behavior → institutions
    Dynamic coalitions WITHIN species

Like Cooper pairs: condensation requires matching quantum numbers
(species) before pairing (meta-agent formation) can occur.

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

def _pairwise_kl_cross(system_a: MultiAgentSystem,
                       system_b: MultiAgentSystem,
                       fiber: str = 'model') -> Tensor:
    """KL divergences between two systems on the specified fiber.

    POINTWISE over the base manifold: returns a field, not a scalar.

    Args:
        system_a: N-agent system (rows)
        system_b: M-agent system (columns)
        fiber: 'model' (s_i with Ω̃) or 'belief' (q_i with Ω)
    Returns:
        (N, M, *grid_shape) KL divergence field
    """
    N = system_a.N_agents
    M = system_b.N_agents

    if fiber == 'model':
        mu_a = system_a.get_all_mu_s()           # (N, *grid, K)
        sigma_a = system_a.get_all_sigma_s()      # (N, *grid, K, K)
        omega_a = system_a.get_all_omega_model()  # (N, *grid, K, K)
        mu_b = system_b.get_all_mu_s()            # (M, *grid, K)
        sigma_b = system_b.get_all_sigma_s()      # (M, *grid, K, K)
        omega_b = system_b.get_all_omega_model()  # (M, *grid, K, K)
    else:
        mu_a = system_a.get_all_mu_q()
        sigma_a = system_a.get_all_sigma_q()
        omega_a = system_a.get_all_omega()
        mu_b = system_b.get_all_mu_q()
        sigma_b = system_b.get_all_sigma_q()
        omega_b = system_b.get_all_omega()

    # Transport: Ω_{iα} = Ω_a_i · Ω_b_α⁻¹
    # Shapes: (N, 1, *grid, K, K) @ (1, M, *grid, K, K) → (N, M, *grid, K, K)
    omega_b_inv = torch.linalg.inv(omega_b)
    transport = omega_a.unsqueeze(1) @ omega_b_inv.unsqueeze(0)

    # Transport b means: (N, M, *grid, K, K) @ (1, M, *grid, K, 1) → squeeze
    mu_b_t = (transport @ mu_b.unsqueeze(0).unsqueeze(-1)).squeeze(-1)

    # Transport b covariances: Ω Σ Ω^T
    sigma_b_t = transport @ sigma_b.unsqueeze(0) @ transport.transpose(-2, -1)

    # Expand a for broadcasting: (N, 1, *grid, ...) → (N, M, *grid, ...)
    mu_a_exp = mu_a.unsqueeze(1).expand_as(mu_b_t)
    sigma_a_exp = sigma_a.unsqueeze(1).expand_as(sigma_b_t)

    return gaussian_kl(mu_a_exp, sigma_a_exp, mu_b_t, sigma_b_t)


def _gate_by_support(field: Tensor, system: MultiAgentSystem) -> Tensor:
    """Gate a (N, M, *grid_shape) field by agent support functions.

    Multiplies field by χ_i(x) for each agent i, so membership is
    zero outside the agent's spatial support.

    Handles both spatial agents (grid_shape non-empty) and scalar
    agents (grid_shape=()) gracefully.
    """
    grid_shape = system.grid_shape
    if not grid_shape:
        # Scalar agents: chi is trivially 1, no gating needed
        return field

    chi = torch.stack([a.chi for a in system.agents])  # (N, *grid)
    # Expand chi for broadcasting: (N, 1, *grid) → (N, M, *grid)
    return field * chi.unsqueeze(1)


# ─────────────────────────────────────────────────────────────
#  Species: structural grouping from model alignment (slow)
# ─────────────────────────────────────────────────────────────

class SpeciesDetector(nn.Module):
    """Detects species structure from model alignment — POINTWISE.

    Species = agents sharing a generative model (s_i).
    This is STRUCTURAL and SLOW — it defines what kind of thing
    you are, not what you're currently thinking.

    The species gate is a FIELD over the base manifold:

        S_{iσ}(x) = σ(-KL(s_i(x) || Ω̃_{iσ}[s_σ(x)]) / τ_species) · χ_i(x)

    At each point x, agent i can only be "species σ" if:
      1. Its model aligns with species σ's model (KL small)
      2. It actually EXISTS at point x (χ_i(x) > 0)

    τ_species should be LARGE (loose — approximate model match suffices).

    Args:
        tau_species: temperature for species detection (larger = looser)
    """

    def __init__(self, tau_species: float = 5.0):
        super().__init__()
        self.tau_species = tau_species

    def species_gate(self, children: MultiAgentSystem,
                     parents: MultiAgentSystem) -> Tensor:
        """Compute species gate S_{iα}(x) from model alignment.

        Args:
            children: N-agent system
            parents: M-agent system (meta-agents / species representatives)
        Returns:
            (N, M, *grid_shape) species gate field, entries in [0,1]
        """
        kl_model = _pairwise_kl_cross(children, parents, fiber='model')
        S = torch.sigmoid(-kl_model / self.tau_species)

        # Gate by support: zero where agent doesn't exist
        S = _gate_by_support(S, children)
        return S


class CoalitionDetector(nn.Module):
    """Detects dynamic coalitions from belief alignment — POINTWISE.

    Coalition = agents agreeing on what's happening NOW.
    This is DYNAMIC and FAST — it changes as beliefs update.

    The coalition membership is a FIELD:

        C_{iα}(x) = σ(-KL(q_i(x) || Ω_{iα}[q_α(x)]) / τ_belief) · χ_i(x)

    Gated by support: you can only be in a coalition where you exist.

    Args:
        tau_belief: temperature for coalition detection (smaller = tighter)
    """

    def __init__(self, tau_belief: float = 1.0):
        super().__init__()
        self.tau_belief = tau_belief

    def coalition_membership(self, children: MultiAgentSystem,
                              parents: MultiAgentSystem) -> Tensor:
        """Compute coalition membership C_{iα}(x) from belief alignment.

        Args:
            children: N-agent system
            parents: M-agent system (meta-agents)
        Returns:
            (N, M, *grid_shape) coalition membership field, entries in [0,1]
        """
        kl_belief = _pairwise_kl_cross(children, parents, fiber='belief')
        C = torch.sigmoid(-kl_belief / self.tau_belief)

        C = _gate_by_support(C, children)
        return C


class GatedMembership(nn.Module):
    """Computes effective membership W(x) = S(x) · C(x) — POINTWISE.

    The selection rule: you can only join a coalition (meta-agent)
    if you're the right species AND you exist at that point.

    W_{iα}(x) = S_{iα}(x) · C_{iα}(x)

    This is a FIELD over the base manifold. A bunch of circular
    agents overlapping creates an irregular-shaped meta-agent
    whose spatial extent is:

        χ_α(x) = Σ_i W_{iα}(x)

    The meta-agent's shape EMERGES from the overlap geometry
    of its members, gated by species compatibility.

    Like Cooper pairs: matching quantum numbers (species) required
    for pairing (coalition), and pairing only happens where
    wavefunctions overlap (support intersection).

    Args:
        tau_species: species detection temperature (default 5.0, loose)
        tau_belief: coalition detection temperature (default 1.0, tight)
    """

    def __init__(self, tau_species: float = 5.0, tau_belief: float = 1.0):
        super().__init__()
        self.species = SpeciesDetector(tau_species)
        self.coalition = CoalitionDetector(tau_belief)

    def compute(self, children: MultiAgentSystem,
                parents: MultiAgentSystem) -> Dict[str, Tensor]:
        """Compute gated membership W(x) = S(x) · C(x).

        Returns:
            Dict with 'W', 'S', 'C' — all (N, M, *grid_shape) fields
        """
        S = self.species.species_gate(children, parents)
        C = self.coalition.coalition_membership(children, parents)
        W = S * C
        return {'W': W, 'S': S, 'C': C}

    def meta_agent_indicator(self, children: MultiAgentSystem,
                              parents: MultiAgentSystem) -> Tensor:
        """Compute meta-agent indicator fields χ_α(x).

        χ_α(x) = Σ_i W_{iα}(x)

        The SHAPE of each meta-agent on the base manifold.
        Emerges from the overlap pattern of member agents,
        gated by species compatibility and belief alignment.

        Args:
            children: N-agent system
            parents: M-agent system (meta-agents)
        Returns:
            (M, *grid_shape) indicator field for each meta-agent
        """
        result = self.compute(children, parents)
        W = result['W']  # (N, M, *grid_shape)
        return W.sum(dim=0)  # (M, *grid_shape)


class SoftMembership(nn.Module):
    """Computes differentiable soft membership W_{iα}(x) — POINTWISE.

    Membership is a FIELD over the base manifold, not a scalar.
    This means meta-agents have emergent spatial shapes arising
    from the overlap geometry of their member agents.

    Two modes:

    1. GATED (default): W(x) = S(x) · C(x)
       Species gate (model alignment) × Coalition (belief alignment).
       Both are fields. Meta-agent shape = overlap of species-compatible
       agents who agree on current beliefs.

    2. UNGATED: W(x) = σ(-KL(s_i(x) || Ω̃[s_α(x)]) / τ) · χ_i(x)
       Pure model alignment gated by support.

    Output shape: (N, M, *grid_shape) — membership at every point.

    Args:
        tau: temperature for ungated mode
        tau_species: species temperature (gated mode)
        tau_belief: coalition temperature (gated mode)
        gated: if True, use W = S · C (default True)
    """

    def __init__(self, tau: float = 1.0,
                 tau_species: Optional[float] = None,
                 tau_belief: Optional[float] = None,
                 gated: bool = True):
        super().__init__()
        self.tau = tau
        self.gated = gated
        if gated:
            # When tau_species/tau_belief not explicitly set, derive from tau:
            # tau_species defaults to 5× tau (loose species matching)
            # tau_belief defaults to tau (tight coalition matching)
            effective_tau_species = tau_species if tau_species is not None else 5.0 * tau
            effective_tau_belief = tau_belief if tau_belief is not None else tau
            self.gated_membership = GatedMembership(effective_tau_species, effective_tau_belief)

    def compute(self, children: MultiAgentSystem,
                parents: MultiAgentSystem) -> Tensor:
        """Compute soft membership field W_{iα}(x).

        Args:
            children: N-agent system at scale ℓ
            parents: M-agent system at scale ℓ+1
        Returns:
            (N, M, *grid_shape) soft membership field, entries in [0,1]
        """
        if self.gated:
            result = self.gated_membership.compute(children, parents)
            return result['W']
        else:
            kl = _pairwise_kl_cross(children, parents, fiber='model')
            W = torch.sigmoid(-kl / self.tau)
            return _gate_by_support(W, children)

    def compute_detailed(self, children: MultiAgentSystem,
                          parents: MultiAgentSystem) -> Dict[str, Tensor]:
        """Compute membership with species/coalition breakdown.

        Returns:
            Dict with 'W', 'S', 'C' — all (N, M, *grid_shape) fields
        """
        if self.gated:
            return self.gated_membership.compute(children, parents)
        else:
            kl = _pairwise_kl_cross(children, parents, fiber='model')
            W = _gate_by_support(torch.sigmoid(-kl / self.tau), children)
            return {'W': W, 'S': W, 'C': torch.ones_like(W)}

    def meta_agent_indicator(self, children: MultiAgentSystem,
                              parents: MultiAgentSystem) -> Tensor:
        """Compute meta-agent indicator fields χ_α(x) = Σ_i W_{iα}(x).

        Returns:
            (M, *grid_shape) — the emergent SHAPE of each meta-agent
        """
        W = self.compute(children, parents)  # (N, M, *grid_shape)
        return W.sum(dim=0)  # (M, *grid_shape)


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

    @staticmethod
    def _reduce_W(W: Tensor) -> Tensor:
        """Reduce field-valued W to scalar per (i, α) pair.

        W is (N, M, *grid_shape). Returns (N, M) by averaging
        over the spatial dimensions.
        """
        if W.dim() == 2:
            return W  # already scalar
        # Average over grid dimensions
        spatial_dims = tuple(range(2, W.dim()))
        return W.mean(dim=spatial_dims)

    @torch.no_grad()
    def order_parameter(self, children: MultiAgentSystem,
                        parents: MultiAgentSystem,
                        W: Tensor) -> Tensor:
        """Compute Ψ_α for each meta-agent.

        Handles both scalar W (N, M) and field W (N, M, *grid_shape)
        by reducing the field to scalar membership weights.

        Args:
            children: N-agent system at scale ℓ
            parents: M-agent system at scale ℓ+1
            W: (N, M) or (N, M, *grid_shape) membership
        Returns:
            (M,) order parameter for each meta-agent
        """
        W_scalar = self._reduce_W(W)

        N = children.N_agents
        M = parents.N_agents
        K = children.K

        # For order parameter, use grid-averaged model means
        # (spatial structure of s_i is a field, but order parameter
        #  measures inter-agent coherence, which is a scalar)
        mu_s = children.get_all_mu_s()                # (N, *grid, K)
        omega_child = children.get_all_omega_model()  # (N, *grid, K, K)
        omega_parent = parents.get_all_omega_model()  # (M, *grid, K, K)

        # Reduce to per-agent means for order parameter
        spatial_dims_mu = tuple(range(1, mu_s.dim() - 1))
        if spatial_dims_mu:
            mu_s_avg = mu_s.mean(dim=spatial_dims_mu)          # (N, K)
            omega_child_avg = omega_child.mean(dim=spatial_dims_mu)  # (N, K, K)
            omega_parent_avg = omega_parent.mean(dim=spatial_dims_mu)  # (M, K, K)
        else:
            mu_s_avg = mu_s
            omega_child_avg = omega_child
            omega_parent_avg = omega_parent

        psi = torch.zeros(M, device=mu_s.device)

        for alpha in range(M):
            transported_means = []
            ws = []
            for i in range(N):
                w = W_scalar[i, alpha]
                if w < 1e-6:
                    continue
                omega_ai = omega_parent_avg[alpha] @ torch.linalg.inv(omega_child_avg[i])
                mu_t = omega_ai @ mu_s_avg[i]
                transported_means.append(mu_t)
                ws.append(w)

            if len(transported_means) < 2:
                psi[alpha] = 0.0
                continue

            means = torch.stack(transported_means)
            ws_t = torch.stack(ws)
            Z = ws_t.sum()
            mu_bar = (ws_t.unsqueeze(-1) * means).sum(0) / Z
            diff = means - mu_bar.unsqueeze(0)
            var = (ws_t.unsqueeze(-1) * diff ** 2).sum(0) / Z
            psi[alpha] = var.sum()

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
    """Differentiable cross-scale free energy coupling — POINTWISE.

    The cross-scale VFE is a spatial integral, weighted by field-valued
    membership W_{iα}(x):

        S_cross = Σ_{i,α} ∫_C W_{iα}(x) [
            KL(p_i(x) || Ω_{iα}[q_α(x)])      (belief top-down)
          + KL(r_i(x) || Ω̃_{iα}[s_α(x)])      (model top-down)
        ] dx

    When grid_shape=(), this reduces to scalar membership.

    The meta-agent provides a spatially-varying gravitational well:
    stronger membership at a point → stronger top-down constraint there.

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

        Uses _pairwise_kl_cross for grid-safe transport/KL computation.

        Args:
            children: N-agent system at scale ℓ
            parents: M-agent system at scale ℓ+1
            W: (N, M, *grid_shape) soft membership field
        Returns:
            Dict with belief_topdown, model_topdown, total
        """
        # Belief fiber: KL(p_i || Ω_{iα}[q_α]) — pointwise
        # We need KL between child PRIORS and transported parent BELIEFS
        # This is slightly different from _pairwise_kl_cross (which does
        # q_i vs q_α or s_i vs s_α), so we build a helper wrapper.

        # Create temporary "systems" that expose (p_i, Σ_p, Ω) for children
        # and (q_α, Σ_q, Ω) for parents — but actually _pairwise_kl_cross
        # already handles the transport correctly. The issue is we want
        # KL(p_i || transport[q_α]), not KL(q_i || transport[q_α]).
        # So we compute the KL directly with proper grid handling.

        kl_belief = self._cross_fiber_kl(
            children.get_all_mu_p(), children.get_all_sigma_p(),
            children.get_all_omega(),
            parents.get_all_mu_q(), parents.get_all_sigma_q(),
            parents.get_all_omega(),
        )  # (N, M, *grid_shape)

        belief_topdown = (W * kl_belief).sum()

        # Model fiber: KL(r_i || Ω̃_{iα}[s_α]) — pointwise
        kl_model = self._cross_fiber_kl(
            children.get_all_mu_r(), children.get_all_sigma_r(),
            children.get_all_omega_model(),
            parents.get_all_mu_s(), parents.get_all_sigma_s(),
            parents.get_all_omega_model(),
        )  # (N, M, *grid_shape)

        model_topdown = (W * kl_model).sum()

        total = (self.lambda_belief_topdown * belief_topdown
                 + self.lambda_model_topdown * model_topdown)

        return {
            'total': total,
            'belief_topdown': belief_topdown,
            'model_topdown': model_topdown,
        }

    @staticmethod
    def _cross_fiber_kl(mu_a, sigma_a, omega_a,
                         mu_b, sigma_b, omega_b) -> Tensor:
        """KL(a_i || Ω_{iα}[b_α]) for all (i,α) pairs, pointwise.

        Args:
            mu_a: (N, *grid, K)     — child distribution means
            sigma_a: (N, *grid, K, K) — child covariances
            omega_a: (N, *grid, K, K) — child gauge frames
            mu_b: (M, *grid, K)     — parent distribution means
            sigma_b: (M, *grid, K, K) — parent covariances
            omega_b: (M, *grid, K, K) — parent gauge frames
        Returns:
            (N, M, *grid_shape) KL divergences
        """
        # Transport: Ω_{iα} = Ω_a_i · Ω_b_α⁻¹
        omega_b_inv = torch.linalg.inv(omega_b)
        transport = omega_a.unsqueeze(1) @ omega_b_inv.unsqueeze(0)

        # Transport parent: Ω_{iα}[b_α]
        mu_b_t = (transport @ mu_b.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        sigma_b_t = transport @ sigma_b.unsqueeze(0) @ transport.transpose(-2, -1)

        # Expand child for broadcasting
        mu_a_exp = mu_a.unsqueeze(1).expand_as(mu_b_t)
        sigma_a_exp = sigma_a.unsqueeze(1).expand_as(sigma_b_t)

        return gaussian_kl(mu_a_exp, sigma_a_exp, mu_b_t, sigma_b_t)


# ─────────────────────────────────────────────────────────────
#  HierarchicalEmergence: the full soft multi-scale system
# ─────────────────────────────────────────────────────────────

class HierarchicalEmergence(nn.Module):
    """Multi-scale soft hierarchy with species-gated meta-agents.

    Two orthogonal structures:

    SPECIES (model fiber, slow):
      Agents sharing generative models form species groups.
      S_{iσ} = σ(-KL(s_i || Ω̃_{iσ}[s_σ]) / τ_species)
      Changes slowly (ε << 1 in dynamics). Defines compatibility.

    META-AGENTS (belief fiber, fast):
      Agents agreeing on current state form coalitions.
      C_{iα} = σ(-KL(q_i || Ω_{iα}[q_α]) / τ_belief)
      Changes fast. Defines coordination.

    SELECTION RULE:
      W_{iα} = S_{iα} · C_{iα}
      Can only join coalition α if you're the right species.

    Args:
        base_system: scale-0 MultiAgentSystem
        n_meta_per_scale: meta-agents at each scale (default: halving)
        tau_species: species detection temperature (larger = looser)
        tau_belief: coalition detection temperature (smaller = tighter)
        gated: if True, W = S · C. If False, W = model alignment only.
        lambda_cross: weight for cross-scale coupling
        lambda_belief_topdown: weight for belief top-down
        lambda_model_topdown: weight for model top-down
    """

    def __init__(self, base_system: MultiAgentSystem,
                 n_meta_per_scale: Optional[List[int]] = None,
                 tau_species: float = 5.0,
                 tau_belief: float = 1.0,
                 gated: bool = True,
                 lambda_cross: float = 1.0,
                 lambda_belief_topdown: float = 1.0,
                 lambda_model_topdown: float = 1.0):
        super().__init__()

        self.tau_species = tau_species
        self.tau_belief = tau_belief
        self.gated = gated
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

        # Soft membership: species-gated by default
        self.membership = SoftMembership(
            tau=tau_belief,
            tau_species=tau_species,
            tau_belief=tau_belief,
            gated=gated,
        )

        # Cross-scale VFE
        self.cross_vfe = CrossScaleVFE(
            lambda_belief_topdown=lambda_belief_topdown,
            lambda_model_topdown=lambda_model_topdown,
        )

        # Diagnostics
        self.condensation = CondensationDiagnostics()

    def compute_all_memberships(self) -> List[Tensor]:
        """Compute effective membership matrices W at all scales.

        Returns:
            List of (N_ℓ, N_{ℓ+1}) membership matrices
        """
        memberships = []
        for ell in range(self.n_levels - 1):
            W = self.membership.compute(self.scales[ell], self.scales[ell + 1])
            memberships.append(W)
        return memberships

    def compute_all_memberships_detailed(self) -> List[Dict[str, Tensor]]:
        """Compute memberships with species/coalition breakdown.

        Returns:
            List of dicts with 'W', 'S' (species), 'C' (coalition)
        """
        details = []
        for ell in range(self.n_levels - 1):
            d = self.membership.compute_detailed(
                self.scales[ell], self.scales[ell + 1]
            )
            details.append(d)
        return details

    def meta_agent_indicators(self) -> List[Tensor]:
        """Compute meta-agent indicator fields at each scale.

        Returns:
            List of (N_{ℓ+1}, *grid_shape) indicator fields.
            Each entry shows the emergent SHAPE of meta-agents at that scale.
        """
        indicators = []
        for ell in range(self.n_levels - 1):
            chi = self.membership.meta_agent_indicator(
                self.scales[ell], self.scales[ell + 1]
            )
            indicators.append(chi)
        return indicators

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

        For field-valued W (N, M, *grid_shape), reduces to scalar
        weights by averaging over spatial dimensions before pooling.
        The meta-agent's pooled parameters represent the aggregate
        over all spatial points where membership is nonzero.
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

        # Reduce field W to scalar weights for precision pooling
        W_scalar = CondensationDiagnostics._reduce_W(W)

        for alpha in range(M):
            parent = parents.agents[alpha]
            w = W_scalar[:, alpha]  # (N,)

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

        Returns dict with species/coalition breakdown at each scale.
        """
        info = {
            'n_levels': self.n_levels,
            'scale_agents': [s.N_agents for s in self.scales],
            'tau_species': self.tau_species,
            'tau_belief': self.tau_belief,
            'gated': self.gated,
        }

        details = self.compute_all_memberships_detailed()
        info['condensation_fractions'] = []
        info['order_parameters'] = []
        info['membership_stats'] = []

        for ell in range(self.n_levels - 1):
            d = details[ell]
            W = d['W']
            S = d['S']
            C = d['C']

            f = self.condensation.condensation_fraction(
                self.scales[ell], self.scales[ell + 1], W
            )
            info['condensation_fractions'].append(f)

            psi = self.condensation.order_parameter(
                self.scales[ell], self.scales[ell + 1], W
            )
            info['order_parameters'].append(psi.tolist())

            info['membership_stats'].append({
                'W_mean': W.mean().item(),
                'S_mean': S.mean().item(),
                'C_mean': C.mean().item(),
                'W_max': W.max().item(),
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
        mode = "gated" if self.gated else "ungated"
        return (f"Hierarchy [{' → '.join(parts)}] "
                f"condensation=[{', '.join(frac_str)}] "
                f"τ_s={self.tau_species:.1f} τ_b={self.tau_belief:.1f} ({mode})")
