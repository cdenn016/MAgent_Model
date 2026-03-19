"""
Module C: GL(K,ℂ) Extension — The Pathway to Lorentzian Signature
====================================================================

The manuscript's most important theoretical result (Section 5.3):

  Complex gauge frames produce LORENTZIAN signature through the
  Yang-Mills kinetic metric of the gauge connection.

Mechanism:
  φ(τ,x) = i·ψ_τ·T + ψ_x·T     (imaginary temporal, real spatial)

  A_τ = i·(∂_τ ψ_τ)·T            → G_ττ = tr(A_τ²) = i²·... = NEGATIVE
  A_x = (∂_x ψ_x)·T             → G_xx = tr(A_x²) = POSITIVE

  ds² = -2(∂_τψ_τ)² dτ² + 2(∂_xψ_x)² dx²   ← Lorentzian!

The key insight: Lorentzian signature is NOT imposed. It arises from
the gauge connection when temporal gauge components are imaginary.
The fiber metric (Fisher-Rao) stays positive-definite throughout.

The Lorentz group SO(1,3) emerges as the metric-preserving subgroup
of GL(K,ℂ) for this induced metric.

Reference: Dennis (2026), Section 5.3.3 (Worked Example)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple
import math


class ComplexGaugeFrame(nn.Module):
    """GL(K,ℂ) gauge frame with complex-valued parameters.

    The frame Ω ∈ GL(K,ℂ) is a K×K complex invertible matrix.
    Beliefs and priors remain REAL Gaussians — only the gauge
    frame becomes complex.

    The imaginary part of the gauge frame in the temporal direction
    is what produces Lorentzian signature via i² = -1.

    Args:
        K: fiber dimension
        N_agents: number of agents
        grid_shape: base manifold discretization
        init_scale: perturbation scale
        temporal_dims: which base manifold dimensions are temporal
            (these get imaginary gauge components)
    """

    def __init__(self, K: int, N_agents: int,
                 grid_shape: Tuple[int, ...] = (),
                 init_scale: float = 0.1,
                 temporal_dims: Tuple[int, ...] = (0,)):
        super().__init__()
        self.K = K
        self.N_agents = N_agents
        self.grid_shape = grid_shape
        self.temporal_dims = temporal_dims

        # Real part of gauge frame
        full_shape = (N_agents,) + grid_shape + (K, K)
        real_init = torch.eye(K).reshape((1,) + (1,) * len(grid_shape) + (K, K))
        real_init = real_init.expand(full_shape).clone()
        real_init = real_init + init_scale * torch.randn(full_shape)
        self.frame_real = nn.Parameter(real_init)

        # Imaginary part (starts small)
        imag_init = init_scale * 0.1 * torch.randn(full_shape)
        self.frame_imag = nn.Parameter(imag_init)

    @property
    def frame(self) -> Tensor:
        """Complex gauge frame Ω = Re(Ω) + i·Im(Ω).

        Returns:
            (N, *grid, K, K) complex tensor
        """
        return torch.complex(self.frame_real, self.frame_imag)

    def transport_operator(self, i: int, j: int) -> Tensor:
        """Complex transport Ω_ij = Ω_i @ Ω_j⁻¹.

        Returns:
            (*grid, K, K) complex transport operator
        """
        omega_i = self.frame[i]
        omega_j = self.frame[j]
        return omega_i @ torch.linalg.inv(omega_j)

    def connection_form(self, agent: int, direction: int,
                        h: float = 1.0) -> Tensor:
        """Gauge connection A_μ = Ω⁻¹ ∂_μ Ω in the complex case.

        Args:
            agent: agent index
            direction: base manifold direction μ
            h: grid spacing
        Returns:
            (*grid, K, K) complex connection one-form
        """
        U = self.frame[agent]  # (*grid, K, K) complex
        U_inv = torch.linalg.inv(U)

        # ∂_μ U via central differences
        dU = (torch.roll(U, -1, dims=direction) -
              torch.roll(U, 1, dims=direction)) / (2 * h)

        return U_inv @ dU

    def yang_mills_kinetic_metric(self, agent: int,
                                   h: float = 1.0) -> Tensor:
        """Yang-Mills kinetic metric G_μν = tr(A_μ A_ν).

        THIS IS WHERE LORENTZIAN SIGNATURE EMERGES.

        When A_τ has imaginary components (from imaginary gauge frame
        in temporal direction), tr(A_τ²) becomes negative via i² = -1.
        When A_x is real, tr(A_x²) is positive.

        Result: G = diag(-..., +..., +...) = Lorentzian!

        Args:
            agent: agent index
            h: grid spacing
        Returns:
            (*grid, n_dims, n_dims) kinetic metric (may be indefinite)
        """
        n_dims = len(self.grid_shape)
        G = torch.zeros(self.grid_shape + (n_dims, n_dims),
                        device=self.frame_real.device)

        connections = []
        for mu in range(n_dims):
            A_mu = self.connection_form(agent, mu, h)
            connections.append(A_mu)

        for mu in range(n_dims):
            for nu in range(mu, n_dims):
                # G_μν = Re[tr(A_μ A_ν)]
                product = connections[mu] @ connections[nu]
                trace = product.diagonal(dim1=-2, dim2=-1).sum(-1)
                G[..., mu, nu] = trace.real
                if mu != nu:
                    G[..., nu, mu] = G[..., mu, nu]

        return G

    def signature(self, agent: int, h: float = 1.0) -> Dict[str, Tensor]:
        """Detect the metric signature from the Yang-Mills kinetic metric.

        Counts positive, negative, and zero eigenvalues of G_μν
        at each grid point.

        Returns:
            Dict with 'positive', 'negative', 'zero' counts and eigenvalues
        """
        G = self.yang_mills_kinetic_metric(agent, h)
        eigenvalues = torch.linalg.eigvalsh(G)

        eps = 1e-6
        n_pos = (eigenvalues > eps).float().sum(-1).mean()
        n_neg = (eigenvalues < -eps).float().sum(-1).mean()
        n_zero = ((eigenvalues >= -eps) & (eigenvalues <= eps)).float().sum(-1).mean()

        return {
            'positive': n_pos,
            'negative': n_neg,
            'zero': n_zero,
            'eigenvalues': eigenvalues,
            'is_lorentzian': (n_neg >= 0.5) and (n_pos >= 0.5),
            'signature_string': f"({int(n_neg.item())},{int(n_pos.item())})",
        }


class LorentzianSignatureDetector:
    """Tools for verifying Lorentzian signature emergence.

    The manuscript's worked example (Section 5.3.3):
      - 2D base manifold (τ, x)
      - GL(2,ℂ) gauge bundle
      - φ(τ,x) = i·ψ_τ·T + ψ_x·T with T = diag(1,-1)
      - G_ττ = -2(∂_τψ_τ)² < 0 (timelike)
      - G_xx = +2(∂_xψ_x)² > 0 (spacelike)
    """

    @staticmethod
    def verify_worked_example(grid_size: int = 32,
                               device: str = 'cpu') -> Dict:
        """Reproduce the manuscript's worked example exactly.

        Sets up the GL(2,ℂ) configuration from Eq. (30)-(33)
        and verifies Lorentzian signature emerges.

        Args:
            grid_size: resolution of the (τ, x) grid
            device: torch device
        Returns:
            Dict with metric components, eigenvalues, verification
        """
        K = 2
        grid_shape = (grid_size, grid_size)

        # Create complex gauge frame field
        cgf = ComplexGaugeFrame(K, N_agents=1, grid_shape=grid_shape,
                                init_scale=0.0, temporal_dims=(0,))

        # Set up the specific configuration from the manuscript:
        # φ(τ,x) = i·ψ_τ(τ,x)·T + ψ_x(τ,x)·T
        # where T = diag(1, -1)
        T = torch.diag(torch.tensor([1.0, -1.0], device=device))

        # Create coordinate grids
        tau = torch.linspace(0, 2 * math.pi, grid_size, device=device)
        x = torch.linspace(0, 2 * math.pi, grid_size, device=device)
        TAU, X = torch.meshgrid(tau, x, indexing='ij')

        # Scalar fields ψ_τ and ψ_x (smooth, non-trivial)
        psi_tau = torch.sin(TAU) * torch.cos(X)  # varies in both dims
        psi_x = torch.cos(TAU) * torch.sin(X)

        # Build the gauge frame: Ω = exp(φ) ≈ I + φ for small φ
        # φ = i·ψ_τ·T + ψ_x·T
        # For the linearized analysis, set frame ≈ I + φ
        with torch.no_grad():
            for i in range(grid_size):
                for j in range(grid_size):
                    phi = psi_x[i, j] * T  # real part from spatial
                    phi_imag = psi_tau[i, j] * T  # imaginary part from temporal
                    cgf.frame_real.data[0, i, j] = torch.eye(K) + 0.1 * phi
                    cgf.frame_imag.data[0, i, j] = 0.1 * phi_imag

        # Compute Yang-Mills kinetic metric
        h = 2 * math.pi / grid_size
        G = cgf.yang_mills_kinetic_metric(agent=0, h=h)

        # Extract components
        G_tautau = G[..., 0, 0]  # should be NEGATIVE (timelike)
        G_xx = G[..., 1, 1]      # should be POSITIVE (spacelike)
        G_taux = G[..., 0, 1]    # should be ~0 (diagonal)

        # Check signature
        sig = cgf.signature(agent=0, h=h)

        # Eigenvalues of G at each point
        eigenvalues = sig['eigenvalues']

        result = {
            'G_tautau_mean': G_tautau.mean().item(),
            'G_xx_mean': G_xx.mean().item(),
            'G_taux_mean': G_taux.mean().item(),
            'G_tautau_negative': (G_tautau < 0).float().mean().item(),
            'G_xx_positive': (G_xx > 0).float().mean().item(),
            'is_lorentzian': sig['is_lorentzian'],
            'signature': sig['signature_string'],
            'eigenvalue_min': eigenvalues.min().item(),
            'eigenvalue_max': eigenvalues.max().item(),
            'G_tensor': G.detach(),
        }

        return result

    @staticmethod
    def lorentz_boost(rapidity: float, device: str = 'cpu') -> Tensor:
        """SO(1,1) Lorentz boost matrix.

        Λ(ξ) = [[cosh ξ, sinh ξ],
                 [sinh ξ, cosh ξ]]

        Preserves η = diag(-1, +1): Λ^T η Λ = η.

        Args:
            rapidity: boost parameter ξ
        Returns:
            (2, 2) boost matrix
        """
        ch = math.cosh(rapidity)
        sh = math.sinh(rapidity)
        return torch.tensor([[ch, sh], [sh, ch]], device=device, dtype=torch.float32)

    @staticmethod
    def verify_lorentz_invariance(rapidity: float = 0.5,
                                   device: str = 'cpu') -> Dict:
        """Verify that SO(1,1) preserves the Lorentzian metric.

        Checks: Λ^T η Λ = η for η = diag(-1, +1).

        Returns:
            Dict with verification results
        """
        Lambda = LorentzianSignatureDetector.lorentz_boost(rapidity, device)
        eta = torch.diag(torch.tensor([-1.0, 1.0], device=device))

        # Λ^T η Λ
        result = Lambda.T @ eta @ Lambda
        residual = (result - eta).norm().item()

        return {
            'boost_matrix': Lambda,
            'eta': eta,
            'Lambda_T_eta_Lambda': result,
            'residual': residual,
            'invariant': residual < 1e-6,
        }


class ComplexTransport:
    """Transport operators in the GL(K,ℂ) setting.

    Beliefs remain real Gaussians N(μ, Σ) with μ ∈ ℝ^K, Σ ∈ SPD(K).
    Transport uses the REAL PART of the complex transport operator:

      μ' = Re(Ω_ij) μ     (real transport of real beliefs)
      Σ' = Re(Ω_ij) Σ Re(Ω_ij)^T

    The imaginary part affects the gauge connection and induced metric
    but not the direct transport of real-valued beliefs.

    For full complex beliefs (future work), the transport would use
    the full complex Ω_ij on complex-valued parameters.
    """

    @staticmethod
    def transport_real_belief(omega_ij_complex: Tensor,
                               mu: Tensor, sigma: Tensor
                               ) -> Tuple[Tensor, Tensor]:
        """Transport real Gaussian belief through complex gauge.

        Uses the real part of the complex transport operator.

        Args:
            omega_ij_complex: (..., K, K) complex transport
            mu: (..., K) real mean
            sigma: (..., K, K) real SPD covariance
        Returns:
            (mu_transported, sigma_transported) real tensors
        """
        omega_real = omega_ij_complex.real

        mu_t = (omega_real @ mu.unsqueeze(-1)).squeeze(-1)
        sigma_t = omega_real @ sigma @ omega_real.transpose(-2, -1)

        return mu_t, sigma_t

    @staticmethod
    def transport_complex_belief(omega_ij: Tensor,
                                  mu_complex: Tensor,
                                  sigma_complex: Tensor
                                  ) -> Tuple[Tensor, Tensor]:
        """Transport complex belief (future: complex exponential families).

        Full complex transport for when beliefs themselves are complex.

        Args:
            omega_ij: (..., K, K) complex transport
            mu_complex: (..., K) complex mean
            sigma_complex: (..., K, K) complex covariance
        Returns:
            (mu_t, sigma_t) complex tensors
        """
        mu_t = (omega_ij @ mu_complex.unsqueeze(-1)).squeeze(-1)
        sigma_t = omega_ij @ sigma_complex @ omega_ij.conj().transpose(-2, -1)
        return mu_t, sigma_t
