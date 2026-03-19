"""
Layer 1: Statistical Manifold — Gaussian Distributions with Information Geometry
==================================================================================

The fiber of the associated bundle is the manifold of K-dimensional Gaussian
distributions B = {N(μ, Σ) : μ ∈ ℝ^K, Σ ≻ 0}, equipped with the Fisher-Rao metric.

Key objects:
  - GaussianDistribution: batched (μ, Σ) with SPD enforcement
  - gaussian_kl: closed-form KL(q || p) for Gaussians
  - fisher_rao_metric: the unique (up to scale) invariant Riemannian metric
  - natural_gradient: Fisher-inverse times Euclidean gradient

Reference: Dennis (2026), Sections 2.3-2.4
  KL formula: Eq. (4) in the manuscript
  Fisher metric: Eq. (1)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


def ensure_spd(sigma: Tensor, eps: float = 1e-4) -> Tensor:
    """Project matrix to symmetric positive definite.

    Uses eigendecomposition to clamp minimum eigenvalue, ensuring
    the result is truly SPD even after gauge transport.

    Args:
        sigma: (..., K, K) covariance matrices
        eps: minimum eigenvalue floor
    Returns:
        SPD matrix of same shape
    """
    sym = 0.5 * (sigma + sigma.transpose(-2, -1))
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(sym)
        eigenvalues = eigenvalues.clamp(min=eps)
        return eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-2, -1)
    except Exception:
        # Fallback: add eps*I
        I = torch.eye(sym.shape[-1], device=sym.device, dtype=sym.dtype)
        return sym + eps * I.expand_as(sym)


def log_det_spd(sigma: Tensor) -> Tensor:
    """Numerically stable log-determinant for SPD matrices via Cholesky.

    Args:
        sigma: (..., K, K) SPD matrices
    Returns:
        (...,) log|Σ|
    """
    L = torch.linalg.cholesky(sigma)
    return 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)


class GaussianDistribution(nn.Module):
    """Batched multivariate Gaussian distribution on the statistical manifold.

    Each distribution is N(μ, Σ) where μ ∈ ℝ^K and Σ ∈ SPD(K).
    Σ is parameterized via its Cholesky factor L (Σ = LL^T) to guarantee
    positive definiteness through optimization.

    Args:
        K: dimension of the distribution
        N: number of distributions (e.g., number of agents × grid points)
        init_mean_scale: scale for random mean initialization
        init_cov_scale: scale for covariance initialization (diagonal)
    """

    def __init__(self, K: int, N: int = 1,
                 init_mean_scale: float = 1.0,
                 init_cov_scale: float = 1.0):
        super().__init__()
        self.K = K
        self.N = N

        # Mean: unconstrained
        self.mu = nn.Parameter(init_mean_scale * torch.randn(N, K))

        # Cholesky factor: parameterize L so Σ = LL^T is always SPD
        # Initialize as scaled identity
        L_init = init_cov_scale * torch.eye(K).unsqueeze(0).expand(N, -1, -1).clone()
        # Store the raw lower triangular parameters
        self._L_raw = nn.Parameter(L_init)

    @property
    def L(self) -> Tensor:
        """Lower Cholesky factor with positive diagonal.

        Returns:
            (N, K, K) lower triangular with positive diagonal
        """
        L = torch.tril(self._L_raw)
        # Force positive diagonal
        diag = L.diagonal(dim1=-2, dim2=-1)
        L = L - torch.diag_embed(diag) + torch.diag_embed(diag.abs().clamp(min=1e-6))
        return L

    @property
    def sigma(self) -> Tensor:
        """Covariance matrix Σ = LL^T.

        Returns:
            (N, K, K) SPD covariance matrices
        """
        L = self.L
        return L @ L.transpose(-2, -1)

    @property
    def precision(self) -> Tensor:
        """Precision matrix Λ = Σ⁻¹.

        Returns:
            (N, K, K) precision matrices
        """
        return torch.cholesky_inverse(self.L)

    @property
    def log_det_sigma(self) -> Tensor:
        """Log determinant log|Σ| = 2 * sum(log(diag(L))).

        Returns:
            (N,) log determinants
        """
        return 2.0 * self.L.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    def entropy(self) -> Tensor:
        """Differential entropy H[q] = K/2 (1 + log(2π)) + 1/2 log|Σ|.

        Returns:
            (N,) entropy values in nats
        """
        return 0.5 * self.K * (1.0 + torch.log(torch.tensor(2 * torch.pi))) + \
               0.5 * self.log_det_sigma


def gaussian_kl(mu_q: Tensor, sigma_q: Tensor,
                mu_p: Tensor, sigma_p: Tensor) -> Tensor:
    """KL divergence KL(q || p) between multivariate Gaussians.

    KL(N(μ_q, Σ_q) || N(μ_p, Σ_p)) =
        1/2 [log|Σ_p|/|Σ_q| + tr(Σ_p⁻¹ Σ_q) + (μ_p - μ_q)^T Σ_p⁻¹ (μ_p - μ_q) - K]

    This is Eq. (4) of the manuscript.

    Uses torch.linalg.solve and slogdet for robustness through
    gauge transport (which can produce near-singular covariances).

    Args:
        mu_q: (..., K) mean of q
        sigma_q: (..., K, K) covariance of q
        mu_p: (..., K) mean of p
        sigma_p: (..., K, K) covariance of p
    Returns:
        (...,) KL divergence values (non-negative)
    """
    K = mu_q.shape[-1]
    diff = mu_p - mu_q  # (..., K)

    # Regularize for numerical stability
    eps = 1e-4
    I = torch.eye(K, device=sigma_p.device, dtype=sigma_p.dtype)
    sigma_p_reg = 0.5 * (sigma_p + sigma_p.transpose(-2, -1)) + eps * I
    sigma_q_reg = 0.5 * (sigma_q + sigma_q.transpose(-2, -1)) + eps * I

    # log|Σ_p| - log|Σ_q| via slogdet (more robust than Cholesky)
    sign_p, logdet_p = torch.linalg.slogdet(sigma_p_reg)
    sign_q, logdet_q = torch.linalg.slogdet(sigma_q_reg)
    log_det_ratio = logdet_p - logdet_q

    # tr(Σ_p⁻¹ Σ_q) via solve: X = Σ_p⁻¹ Σ_q = solve(Σ_p, Σ_q)
    sigma_p_inv_sigma_q = torch.linalg.solve(sigma_p_reg, sigma_q_reg)
    trace_term = sigma_p_inv_sigma_q.diagonal(dim1=-2, dim2=-1).sum(-1)

    # Mahalanobis: (μ_p - μ_q)^T Σ_p⁻¹ (μ_p - μ_q)
    # v = Σ_p⁻¹ diff
    v = torch.linalg.solve(sigma_p_reg, diff.unsqueeze(-1)).squeeze(-1)
    mahal = (diff * v).sum(-1)

    kl = 0.5 * (log_det_ratio + trace_term + mahal - K)
    return kl.clamp(min=0.0)  # Numerical safety


def gaussian_kl_from_cholesky(mu_q: Tensor, L_q: Tensor,
                               mu_p: Tensor, L_p: Tensor) -> Tensor:
    """KL divergence from pre-computed Cholesky factors (more efficient).

    Args:
        mu_q: (..., K) mean of q
        L_q: (..., K, K) lower Cholesky factor of Σ_q
        mu_p: (..., K) mean of p
        L_p: (..., K, K) lower Cholesky factor of Σ_p
    Returns:
        (...,) KL divergence values
    """
    K = mu_q.shape[-1]
    diff = mu_p - mu_q

    log_det_p = 2.0 * L_p.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    log_det_q = 2.0 * L_q.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    # Solve L_p⁻¹ L_q to get trace term
    M = torch.linalg.solve_triangular(L_p, L_q, upper=False)
    trace_term = (M * M).sum(dim=(-2, -1))

    # Mahalanobis
    v = torch.linalg.solve_triangular(L_p, diff.unsqueeze(-1), upper=False)
    mahal = (v * v).sum(dim=(-2, -1))

    return 0.5 * (log_det_p - log_det_q + trace_term + mahal - K)


def fisher_rao_metric(delta_mu: Tensor, delta_sigma: Tensor,
                      sigma_inv: Tensor) -> Tensor:
    """Fisher-Rao metric for Gaussian manifold.

    g_B(δq, δq) = δμ^T Σ⁻¹ δμ + (1/2) tr(Σ⁻¹ δΣ Σ⁻¹ δΣ)

    This is Eq. (1) of the manuscript.

    Args:
        delta_mu: (..., K) tangent vector in mean direction
        delta_sigma: (..., K, K) tangent vector in covariance direction
        sigma_inv: (..., K, K) precision matrix
    Returns:
        (...,) metric value (squared norm of tangent vector)
    """
    # Mean contribution: δμ^T Σ⁻¹ δμ
    mean_term = (delta_mu.unsqueeze(-2) @ sigma_inv @ delta_mu.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    # Covariance contribution: (1/2) tr(Σ⁻¹ δΣ Σ⁻¹ δΣ)
    A = sigma_inv @ delta_sigma
    cov_term = 0.5 * (A * A.transpose(-2, -1)).sum(dim=(-2, -1))

    return mean_term + cov_term


def natural_gradient_mu(euclidean_grad_mu: Tensor, sigma: Tensor) -> Tensor:
    """Natural gradient for mean parameters: ∇̃_μ = Σ @ ∇_μ.

    The Fisher metric on the mean sector is Σ⁻¹, so its inverse is Σ.

    Reference: Section 6.2 of the manuscript.

    Args:
        euclidean_grad_mu: (..., K) Euclidean gradient w.r.t. μ
        sigma: (..., K, K) covariance matrix
    Returns:
        (..., K) natural gradient
    """
    return (sigma @ euclidean_grad_mu.unsqueeze(-1)).squeeze(-1)


def natural_gradient_sigma(euclidean_grad_sigma: Tensor, sigma: Tensor) -> Tensor:
    """Natural gradient for covariance parameters: ∇̃_Σ = Σ @ ∇_Σ @ Σ.

    The Fisher metric on SPD(K) is (1/2)(Σ⁻¹ ⊗ Σ⁻¹), inverse is 2(Σ ⊗ Σ).
    Applied as: Σ (∇_Σ S) Σ.

    Reference: Section 6.2 of the manuscript.

    Args:
        euclidean_grad_sigma: (..., K, K) Euclidean gradient w.r.t. Σ
        sigma: (..., K, K) covariance matrix
    Returns:
        (..., K, K) natural gradient (stays in T_Σ SPD(K))
    """
    return sigma @ euclidean_grad_sigma @ sigma


def sample_gaussian(mu: Tensor, sigma: Tensor, n_samples: int = 1) -> Tensor:
    """Reparameterized sampling from N(μ, Σ).

    Args:
        mu: (..., K) means
        sigma: (..., K, K) covariances
        n_samples: number of samples
    Returns:
        (n_samples, ..., K) samples
    """
    L = torch.linalg.cholesky(ensure_spd(sigma))
    eps = torch.randn(*([n_samples] + list(mu.shape)), device=mu.device, dtype=mu.dtype)
    return mu.unsqueeze(0) + (L.unsqueeze(0) @ eps.unsqueeze(-1)).squeeze(-1)


def symmetric_kl(mu_q: Tensor, sigma_q: Tensor,
                 mu_p: Tensor, sigma_p: Tensor) -> Tensor:
    """Symmetrized KL: (KL(q||p) + KL(p||q)) / 2.

    A proper metric-like divergence on the Gaussian manifold.

    Args:
        mu_q, sigma_q: parameters of q
        mu_p, sigma_p: parameters of p
    Returns:
        (...,) symmetrized KL values
    """
    kl_qp = gaussian_kl(mu_q, sigma_q, mu_p, sigma_p)
    kl_pq = gaussian_kl(mu_p, sigma_p, mu_q, sigma_q)
    return 0.5 * (kl_qp + kl_pq)
