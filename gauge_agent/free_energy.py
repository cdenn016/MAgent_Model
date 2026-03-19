"""
Layer 4: The Variational Free Energy Functional
=================================================

The complete variational free energy (Eq. 23 of the manuscript):

S[{q_i}, {p_i}, {Ω_i}] =
  Σ_i ∫ χ_i KL(q_i || p_i) dc                                    [self-consistency]
+ Σ_ij ∫ χ_ij β_ij KL(q_i || Ω_ij[q_j]) dc                      [belief alignment]
+ Σ_ij ∫ χ_ij γ_ij KL(p_i || Ω̃_ij[p_j]) dc                      [model alignment]
- Σ_i ∫ χ_i E_{q_i}[log p(o|q_i)] dc                             [observation]

where β_ij and γ_ij are softmax attention weights derived from
the mixture-of-sources generative model (Section 2.10.4).

Reference: Dennis (2026), Sections 2.10–2.11
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple

from gauge_agent.statistical_manifold import gaussian_kl


class FreeEnergyFunctional(nn.Module):
    """The complete variational free energy functional.

    Computes all four terms of Eq. (23) and returns the total scalar
    free energy, with each component tracked for diagnostics.

    Args:
        lambda_self: weight for self-consistency term KL(q_i || p_i)
        lambda_belief: weight for belief alignment term
        lambda_prior: weight for prior/model alignment term
        lambda_obs: weight for observation likelihood term
        temperature: softmax temperature τ for attention weights
        use_observations: whether to include observation term
    """

    def __init__(self,
                 lambda_self: float = 1.0,
                 lambda_belief: float = 1.0,
                 lambda_prior: float = 1.0,
                 lambda_obs: float = 1.0,
                 temperature: float = 1.0,
                 use_observations: bool = True):
        super().__init__()
        self.lambda_self = lambda_self
        self.lambda_belief = lambda_belief
        self.lambda_prior = lambda_prior
        self.lambda_obs = lambda_obs
        self.temperature = temperature
        self.use_observations = use_observations

    def compute_attention_weights(self, alignment_energies: Tensor,
                                  mask: Optional[Tensor] = None) -> Tensor:
        """Softmax attention from alignment energies (Eq. 16).

        β_ij(c) = exp(-E_ij(c)/τ) / Σ_k exp(-E_ik(c)/τ)

        Derived from mixture-of-sources variational inference,
        NOT ad hoc construction.

        Args:
            alignment_energies: (N, N, *grid) pairwise KL divergences E_ij
            mask: (N, N, *grid) optional mask (0 for self-connections)
        Returns:
            (N, N, *grid) attention weights β_ij, normalized over j
        """
        # Mask out self-connections
        N = alignment_energies.shape[0]
        if mask is None:
            mask = 1.0 - torch.eye(N, device=alignment_energies.device)
            # Expand mask for grid dimensions
            for _ in range(alignment_energies.dim() - 2):
                mask = mask.unsqueeze(-1)

        # Softmax over source dimension (j) with temperature
        logits = -alignment_energies / self.temperature
        logits = logits + (mask - 1.0) * 1e9  # mask out self-loops with -inf
        weights = torch.softmax(logits, dim=1)
        return weights * mask

    def self_consistency_term(self,
                               mu_q: Tensor, sigma_q: Tensor,
                               mu_p: Tensor, sigma_p: Tensor,
                               chi: Optional[Tensor] = None) -> Tensor:
        """Σ_i ∫ χ_i(c) KL(q_i(c) || p_i(c)) dc.

        Each agent pays cost for beliefs deviating from priors (Occam's razor).

        Args:
            mu_q: (N, *grid, K) belief means
            sigma_q: (N, *grid, K, K) belief covariances
            mu_p: (N, *grid, K) prior means
            sigma_p: (N, *grid, K, K) prior covariances
            chi: (N, *grid) support functions
        Returns:
            Scalar self-consistency energy
        """
        kl = gaussian_kl(mu_q, sigma_q, mu_p, sigma_p)  # (N, *grid)
        if chi is not None:
            kl = kl * chi
        return kl.sum()

    def belief_alignment_term(self,
                               mu_q: Tensor, sigma_q: Tensor,
                               alignment_energies: Tensor,
                               beta: Tensor,
                               chi_ij: Optional[Tensor] = None) -> Tensor:
        """Σ_ij ∫ χ_ij β_ij E_ij dc where E_ij = KL(q_i || Ω_ij[q_j]).

        Agents pay cost for disagreeing after gauge transport.

        Args:
            mu_q: (N, *grid, K) belief means (unused, for API consistency)
            sigma_q: (N, *grid, K, K) belief covariances (unused)
            alignment_energies: (N, N, *grid) precomputed E_ij
            beta: (N, N, *grid) attention weights
            chi_ij: (N, N, *grid) overlap mask
        Returns:
            Scalar belief alignment energy
        """
        weighted = beta * alignment_energies
        if chi_ij is not None:
            weighted = weighted * chi_ij
        return weighted.sum()

    def prior_alignment_term(self,
                              prior_alignment_energies: Tensor,
                              gamma: Tensor,
                              chi_ij: Optional[Tensor] = None) -> Tensor:
        """Σ_ij ∫ χ_ij γ_ij KL(p_i || Ω̃_ij[p_j]) dc.

        Agents align generative models, forming shared ontologies.

        Args:
            prior_alignment_energies: (N, N, *grid) pairwise prior KL
            gamma: (N, N, *grid) prior attention weights
            chi_ij: (N, N, *grid) overlap mask
        Returns:
            Scalar prior alignment energy
        """
        weighted = gamma * prior_alignment_energies
        if chi_ij is not None:
            weighted = weighted * chi_ij
        return weighted.sum()

    def observation_term(self,
                          mu_q: Tensor, sigma_q: Tensor,
                          observations: Tensor,
                          obs_precision: Optional[Tensor] = None,
                          chi: Optional[Tensor] = None) -> Tensor:
        """-Σ_i ∫ χ_i E_{q_i}[log p(o|q_i)] dc.

        For Gaussian likelihood p(o|θ) = N(o; θ, Σ_o):
        -E_q[log p(o|θ)] = 1/2 (o - μ)^T Λ_o (o - μ) + 1/2 tr(Λ_o Σ) + const

        Args:
            mu_q: (N, *grid, K) or (N, *grid, D_obs) belief means
            sigma_q: (N, *grid, K, K) belief covariances
            observations: (N, *grid, D_obs) or (*grid, D_obs) observations
            obs_precision: (D_obs, D_obs) or scalar observation precision
            chi: (N, *grid) support
        Returns:
            Scalar observation energy (positive = bad fit)
        """
        diff = observations - mu_q  # (N, *grid, D) or broadcast

        if obs_precision is None:
            # Unit precision
            mahal = (diff * diff).sum(-1)
            trace_term = sigma_q.diagonal(dim1=-2, dim2=-1).sum(-1)
        else:
            if obs_precision.dim() == 0:
                # Scalar precision
                mahal = obs_precision * (diff * diff).sum(-1)
                trace_term = obs_precision * sigma_q.diagonal(dim1=-2, dim2=-1).sum(-1)
            else:
                # Matrix precision
                mahal = (diff.unsqueeze(-2) @ obs_precision @ diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)
                trace_term = (obs_precision.unsqueeze(0) * sigma_q).sum(dim=(-2, -1))

        energy = 0.5 * (mahal + trace_term)
        if chi is not None:
            energy = energy * chi
        return energy.sum()

    def forward(self, system, observations: Optional[Tensor] = None,
                obs_precision: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Compute the complete free energy functional.

        Args:
            system: MultiAgentSystem instance
            observations: optional (N, *grid, D_obs) observation data
            obs_precision: optional observation precision
        Returns:
            Dict with 'total', 'self', 'belief_align', 'prior_align',
            'observation', and 'beta' (attention weights)
        """
        mu_q = system.get_all_mu_q()
        sigma_q = system.get_all_sigma_q()
        mu_p = system.get_all_mu_p()
        sigma_p = system.get_all_sigma_p()
        chi_ij = system.overlap_mask()

        # Self-consistency: KL(q_i || p_i)
        E_self = self.self_consistency_term(mu_q, sigma_q, mu_p, sigma_p)

        # Belief alignment energies E_ij = KL(q_i || Ω_ij[q_j])
        E_belief_pairwise = system.pairwise_alignment_energies('belief')
        beta = self.compute_attention_weights(E_belief_pairwise)
        E_belief = self.belief_alignment_term(
            mu_q, sigma_q, E_belief_pairwise, beta, chi_ij
        )

        # Prior alignment energies
        E_prior_pairwise = system.pairwise_alignment_energies('prior')
        gamma = self.compute_attention_weights(E_prior_pairwise)
        E_prior = self.prior_alignment_term(E_prior_pairwise, gamma, chi_ij)

        # Observation term
        E_obs = torch.tensor(0.0, device=mu_q.device)
        if self.use_observations and observations is not None:
            E_obs = self.observation_term(mu_q, sigma_q, observations, obs_precision)

        # Total free energy
        total = (self.lambda_self * E_self
                 + self.lambda_belief * E_belief
                 + self.lambda_prior * E_prior
                 + self.lambda_obs * E_obs)

        return {
            'total': total,
            'self': E_self,
            'belief_align': E_belief,
            'prior_align': E_prior,
            'observation': E_obs,
            'beta': beta.detach(),
            'gamma': gamma.detach(),
            'E_belief_pairwise': E_belief_pairwise.detach(),
            'E_prior_pairwise': E_prior_pairwise.detach(),
        }
