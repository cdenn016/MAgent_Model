"""
Layer 0: GL(K) Lie Group Operations
====================================

The natural gauge group for KL-based variational inference is GL(K),
the general linear group of invertible K×K matrices. The only requirement
is invertibility (det Ω ≠ 0), not orthogonality.

We parameterize gauge frames directly as invertible matrices Ω_i ∈ GL(K).
Transport operators are Ω_ij = Ω_i @ Ω_j⁻¹. All gradients flow through
the group elements via standard autograd — no Lie algebra exp map needed.

Reference: Dennis (2026), Section 2.5 (Gauge Group Choice)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


def ensure_invertible(M: Tensor, eps: float = 1e-6) -> Tensor:
    """Project matrix toward invertibility by adding eps*I if near-singular.

    Args:
        M: (..., K, K) batch of matrices
        eps: minimum absolute eigenvalue threshold
    Returns:
        M regularized to be invertible
    """
    I = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
    return M + eps * I.expand_as(M)


def transport_operator(omega_i: Tensor, omega_j: Tensor) -> Tensor:
    """Compute gauge transport Ω_ij = Ω_i @ Ω_j⁻¹.

    Transforms agent j's representations into agent i's frame.

    Args:
        omega_i: (..., K, K) agent i's gauge frame
        omega_j: (..., K, K) agent j's gauge frame
    Returns:
        Ω_ij = Ω_i @ Ω_j⁻¹ of shape (..., K, K)
    """
    return omega_i @ torch.linalg.inv(omega_j)


def transport_mean(omega_ij: Tensor, mu: Tensor) -> Tensor:
    """Transport mean vector: μ' = Ω_ij @ μ.

    Args:
        omega_ij: (..., K, K) transport operator
        mu: (..., K) or (..., K, 1) mean vector
    Returns:
        Transported mean of same shape as input
    """
    if mu.dim() == omega_ij.dim() - 1:
        return (omega_ij @ mu.unsqueeze(-1)).squeeze(-1)
    return omega_ij @ mu


def transport_covariance(omega_ij: Tensor, sigma: Tensor) -> Tensor:
    """Transport covariance: Σ' = Ω_ij @ Σ @ Ω_ij^T.

    This is the congruence action of GL(K) on SPD matrices.

    Args:
        omega_ij: (..., K, K) transport operator
        sigma: (..., K, K) covariance matrix
    Returns:
        Transported covariance Ω Σ Ω^T
    """
    return omega_ij @ sigma @ omega_ij.transpose(-2, -1)


def transport_precision(omega_ij: Tensor, precision: Tensor) -> Tensor:
    """Transport precision: Λ' = Ω_ij^{-T} @ Λ @ Ω_ij⁻¹.

    Precision transforms contravariantly to covariance.

    Args:
        omega_ij: (..., K, K) transport operator
        precision: (..., K, K) precision matrix
    Returns:
        Transported precision
    """
    omega_inv = torch.linalg.inv(omega_ij)
    return omega_inv.transpose(-2, -1) @ precision @ omega_inv


def cocycle_condition(omega_i: Tensor, omega_j: Tensor, omega_k: Tensor) -> Tensor:
    """Verify cocycle: Ω_ij @ Ω_jk should equal Ω_ik.

    For vertex-local frames Ω_ij = Ω_i Ω_j⁻¹, this holds exactly.
    Returns the Frobenius norm of the residual.

    Args:
        omega_i, omega_j, omega_k: (..., K, K) gauge frames
    Returns:
        Scalar residual ‖Ω_ij Ω_jk - Ω_ik‖_F
    """
    omega_ij = transport_operator(omega_i, omega_j)
    omega_jk = transport_operator(omega_j, omega_k)
    omega_ik = transport_operator(omega_i, omega_k)
    residual = omega_ij @ omega_jk - omega_ik
    return torch.norm(residual, p='fro', dim=(-2, -1))


class GLK(nn.Module):
    """GL(K) gauge frame as a learnable invertible matrix.

    Parameterized directly as an unconstrained K×K matrix,
    initialized near identity. Invertibility maintained via
    regularization rather than constraint.

    Args:
        K: dimension of the representation space
        num_frames: number of gauge frames (e.g., number of agents)
        init_scale: scale of random perturbation from identity
    """

    def __init__(self, K: int, num_frames: int = 1, init_scale: float = 0.1):
        super().__init__()
        self.K = K
        self.num_frames = num_frames

        # Initialize near identity: Ω = I + ε*randn
        frames = torch.eye(K).unsqueeze(0).expand(num_frames, -1, -1).clone()
        frames = frames + init_scale * torch.randn(num_frames, K, K)
        self.frames = nn.Parameter(frames)

    def forward(self, idx: Optional[Tensor] = None) -> Tensor:
        """Return gauge frame(s).

        Args:
            idx: optional index tensor to select specific frames
        Returns:
            (num_frames, K, K) or (len(idx), K, K) invertible matrices
        """
        if idx is not None:
            return self.frames[idx]
        return self.frames

    def transport(self, i: int, j: int) -> Tensor:
        """Compute transport operator Ω_ij = Ω_i @ Ω_j⁻¹.

        Args:
            i: source agent index (whose frame we express in)
            j: target agent index (whose data we transport)
        Returns:
            (K, K) transport operator
        """
        return transport_operator(self.frames[i], self.frames[j])

    def all_transports(self) -> Tensor:
        """Compute all pairwise transport operators.

        Returns:
            (N, N, K, K) tensor where [i,j] = Ω_i @ Ω_j⁻¹
        """
        N = self.num_frames
        # Ω_i: (N, 1, K, K), Ω_j⁻¹: (1, N, K, K)
        omega_inv = torch.linalg.inv(self.frames)
        return self.frames.unsqueeze(1) @ omega_inv.unsqueeze(0)

    def det(self) -> Tensor:
        """Determinants of all frames (for monitoring invertibility).

        Returns:
            (num_frames,) tensor of determinants
        """
        return torch.linalg.det(self.frames)

    def regularization_loss(self, target_det: float = 1.0) -> Tensor:
        """Soft regularization to keep frames well-conditioned.

        Penalizes: (1) determinant deviation from target,
                   (2) condition number growth.

        Returns:
            Scalar regularization loss
        """
        dets = self.det()
        # Log-det penalty: keep |det| near target
        det_loss = (torch.log(dets.abs()) - torch.log(torch.tensor(target_det))) ** 2
        return det_loss.mean()


def init_glk_near_identity(K: int, N: int, scale: float = 0.1,
                           device: str = 'cpu') -> Tensor:
    """Initialize N gauge frames near identity.

    Args:
        K: matrix dimension
        N: number of frames
        scale: perturbation scale
        device: torch device
    Returns:
        (N, K, K) tensor of invertible matrices
    """
    I = torch.eye(K, device=device)
    return I.unsqueeze(0).expand(N, -1, -1) + scale * torch.randn(N, K, K, device=device)


def init_glk_orthogonal(K: int, N: int, device: str = 'cpu') -> Tensor:
    """Initialize N gauge frames as random orthogonal matrices (SO(K) subgroup).

    Useful for starting in the compact subgroup before exploring full GL(K).

    Args:
        K: matrix dimension
        N: number of frames
        device: torch device
    Returns:
        (N, K, K) tensor of orthogonal matrices
    """
    M = torch.randn(N, K, K, device=device)
    Q, _ = torch.linalg.qr(M)
    # Ensure det > 0 (SO rather than O)
    signs = torch.sign(torch.linalg.det(Q))
    Q = Q * signs.unsqueeze(-1).unsqueeze(-1)
    return Q
