"""
Layer 5: Gauge-Covariant Attention Mechanism
=============================================

Attention weights derived from variational inference over a
mixture-of-sources generative model (Section 2.10.4):

  β_ij(c) = softmax_j(-KL(q_i(c) || Ω_ij(c)[q_j(c)]) / τ)

This is NOT ad hoc. It emerges from:
  1. Each agent posits its state came from one neighbor (categorical z)
  2. Mean-field factorization Q(k,z) = q_i(k) · β(z)
  3. Minimizing alignment free energy F_align = KL(Q || P)
  4. Optimal β is softmax over alignment energies (Eq. 16)

The zero-dimensional limit recovers standard transformer attention.

Reference: Dennis (2026), Sections 2.10.4–2.10.5
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple

from gauge_agent.statistical_manifold import gaussian_kl
from gauge_agent.lie_groups import transport_mean, transport_covariance


class GaugeAttention(nn.Module):
    """Gauge-covariant attention: softmax over KL divergences.

    Implements the full mixture-of-sources derivation for both
    belief and prior channels.

    Args:
        temperature: softmax temperature τ > 0
        attention_prior: optional prior π_j(c) on source selection
            (enables causal masking, ALiBi, sliding window, etc.)
    """

    def __init__(self, temperature: float = 1.0,
                 attention_prior: Optional[Tensor] = None):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=False)
        self.attention_prior = attention_prior

    def alignment_energy(self,
                          mu_i: Tensor, sigma_i: Tensor,
                          mu_j: Tensor, sigma_j: Tensor,
                          omega_ij: Tensor) -> Tensor:
        """E_ij = KL(q_i || Ω_ij[q_j]) — the alignment energy.

        Reference: Eq. (14) of the manuscript.

        Args:
            mu_i: (..., K) query agent means
            sigma_i: (..., K, K) query agent covariances
            mu_j: (..., K) key agent means
            sigma_j: (..., K, K) key agent covariances
            omega_ij: (..., K, K) gauge transport operators
        Returns:
            (...,) alignment energies (non-negative)
        """
        # Transport j into i's frame
        mu_j_t = transport_mean(omega_ij, mu_j)
        sigma_j_t = transport_covariance(omega_ij, sigma_j)
        return gaussian_kl(mu_i, sigma_i, mu_j_t, sigma_j_t)

    def forward(self,
                mu_q: Tensor, sigma_q: Tensor,
                omegas: Tensor,
                mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Compute gauge-covariant attention weights.

        The full derivation from mixture-of-sources:

        β_ij = π_j exp(-E_ij/τ) / Σ_k π_k exp(-E_ik/τ)

        where E_ij = KL(q_i || Ω_ij[q_j]).

        Args:
            mu_q: (N, *grid, K) all agent belief means
            sigma_q: (N, *grid, K, K) all agent belief covariances
            omegas: (N, N, *grid, K, K) pairwise transport operators
            mask: (N, N) optional attention mask (0 = blocked)
        Returns:
            Dict with:
                'weights': (N, N, *grid) attention weights β_ij
                'energies': (N, N, *grid) alignment energies E_ij
                'alignment_free_energy': scalar F_align
        """
        N = mu_q.shape[0]
        grid_shape = mu_q.shape[1:-1]
        K = mu_q.shape[-1]

        # Compute all pairwise alignment energies
        # mu_i: (N, 1, *grid, K), mu_j: (1, N, *grid, K)
        mu_i = mu_q.unsqueeze(1).expand(-1, N, *grid_shape, K)
        mu_j = mu_q.unsqueeze(0).expand(N, -1, *grid_shape, K)
        sigma_i = sigma_q.unsqueeze(1).expand(-1, N, *grid_shape, K, K)
        sigma_j = sigma_q.unsqueeze(0).expand(N, -1, *grid_shape, K, K)

        # Transport j → i frame
        mu_j_t = (omegas @ mu_j.unsqueeze(-1)).squeeze(-1)
        sigma_j_t = omegas @ sigma_j @ omegas.transpose(-2, -1)

        # KL(q_i || Ω_ij[q_j])
        E_ij = gaussian_kl(mu_i, sigma_i, mu_j_t, sigma_j_t)

        # Build attention logits
        logits = -E_ij / self.temperature

        # Apply mask (self-connections blocked by default)
        if mask is None:
            self_mask = 1.0 - torch.eye(N, device=logits.device)
            for _ in grid_shape:
                self_mask = self_mask.unsqueeze(-1)
        else:
            self_mask = mask
            if self_mask.dim() < logits.dim():
                for _ in grid_shape:
                    self_mask = self_mask.unsqueeze(-1)

        logits = logits + (self_mask - 1.0) * 1e9

        # Apply attention prior π_j if provided
        if self.attention_prior is not None:
            logits = logits + self.attention_prior.log()

        # Softmax over sources (j dimension)
        beta = torch.softmax(logits, dim=1) * self_mask

        # Alignment free energy: F_align = Σ_j β_ij [E_ij + log β_ij - log π_j]
        # = Σ_j β_ij E_ij - H(β) + KL(β || π)
        entropy = -(beta * (beta + 1e-10).log()).sum(dim=1)
        F_align = (beta * E_ij).sum() - entropy.sum()

        return {
            'weights': beta,
            'energies': E_ij,
            'alignment_free_energy': F_align,
            'entropy': entropy,
        }


class CausalGaugeAttention(GaugeAttention):
    """Gauge attention with causal (autoregressive) masking.

    Sets π_j = 0 for j > i (future tokens in the 0D/transformer limit).
    This is the mechanism that recovers causal transformers as a special case.

    Reference: Dennis (2026), Section 2.10.5 (Eq. 16 discussion)
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__(temperature=temperature)

    def forward(self, mu_q, sigma_q, omegas, mask=None):
        N = mu_q.shape[0]
        # Causal mask: agent i can only attend to j ≤ i
        causal = torch.tril(torch.ones(N, N, device=mu_q.device))
        # Remove self-connections
        causal = causal * (1.0 - torch.eye(N, device=mu_q.device))
        # First agent has no one to attend to — allow self
        causal[0, 0] = 1.0

        if mask is not None:
            causal = causal * mask

        return super().forward(mu_q, sigma_q, omegas, mask=causal)


class ALiBiGaugeAttention(GaugeAttention):
    """Gauge attention with ALiBi-style position bias.

    π_j ∝ exp(-m|i-j|) recovers ALiBi (Press et al., 2022)
    as a special case of non-uniform attention priors.

    Reference: Dennis (2026), Eq. (16)
    """

    def __init__(self, N: int, temperature: float = 1.0,
                 slope: float = 0.1):
        # Construct attention prior
        positions = torch.arange(N).float()
        distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        prior = torch.exp(-slope * distances)
        prior = prior / prior.sum(dim=1, keepdim=True)
        super().__init__(temperature=temperature, attention_prior=prior)
