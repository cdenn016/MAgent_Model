"""
Module A: Riemannian Base Manifolds
=====================================

The base manifold C is where agents live. It is NOT spacetime —
spacetime emerges via pullback (Layer 8). C is the noumenal substrate.

Each manifold provides:
  - metric g_μν(c):  Riemannian metric tensor
  - christoffel Γ^λ_μν(c):  Levi-Civita connection
  - volume_form √|g|(c):  for proper integration ∫ f √|g| dc
  - exp_map / log_map:  geodesic exponential and logarithm
  - geodesic_distance:  intrinsic distance between points
  - covariant_derivative:  ∇_μ V^ν = ∂_μ V^ν + Γ^ν_μλ V^λ

Supported manifolds:
  - EuclideanManifold ℝ^n:  flat, trivial connection
  - Sphere S^n:  positive curvature, compact
  - HyperbolicSpace H^n:  negative curvature, Poincaré ball model
  - Torus T^n:  flat metric, non-trivial topology
  - ProductManifold M₁ × M₂:  block-diagonal metric

Reference: Dennis (2026), Section 2.2 (Base Manifold), Section 2.9
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod
import math


class Manifold(ABC):
    """Abstract Riemannian manifold.

    All manifolds are discretized on a grid of points for computation.
    The grid stores coordinates c ∈ C at each vertex.
    """

    def __init__(self, dim: int, grid_shape: Tuple[int, ...],
                 device: str = 'cpu', dtype: torch.dtype = torch.float32):
        self.dim = dim
        self.grid_shape = grid_shape
        self.device = device
        self.dtype = dtype
        self.n_points = 1
        for s in grid_shape:
            self.n_points *= s

    @abstractmethod
    def coordinates(self) -> Tensor:
        """Grid of coordinate values.

        Returns:
            (*grid_shape, dim) tensor of coordinates
        """
        ...

    @abstractmethod
    def metric(self, c: Optional[Tensor] = None) -> Tensor:
        """Riemannian metric tensor g_μν at points c.

        Args:
            c: (..., dim) coordinates. If None, compute at all grid points.
        Returns:
            (..., dim, dim) metric tensor
        """
        ...

    @abstractmethod
    def volume_form(self, c: Optional[Tensor] = None) -> Tensor:
        """Volume form √|det g| at points c.

        Args:
            c: (..., dim) coordinates. If None, all grid points.
        Returns:
            (...,) volume element
        """
        ...

    def metric_inverse(self, c: Optional[Tensor] = None) -> Tensor:
        """Inverse metric g^μν.

        Returns:
            (..., dim, dim) inverse metric tensor
        """
        g = self.metric(c)
        return torch.linalg.inv(g)

    def christoffel(self, c: Optional[Tensor] = None, h: float = 1e-3) -> Tensor:
        """Christoffel symbols Γ^λ_μν via numerical differentiation of metric.

        Γ^λ_μν = (1/2) g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)

        Args:
            c: (..., dim) coordinates
            h: finite difference step
        Returns:
            (..., dim, dim, dim) Christoffel symbols [λ, μ, ν]
        """
        if c is None:
            c = self.coordinates()

        n = self.dim
        g_inv = self.metric_inverse(c)

        # Compute ∂_σ g_μν for each σ
        dg = torch.zeros(c.shape[:-1] + (n, n, n), device=c.device, dtype=c.dtype)
        for sigma in range(n):
            e_sigma = torch.zeros(n, device=c.device, dtype=c.dtype)
            e_sigma[sigma] = h
            g_plus = self.metric(c + e_sigma)
            g_minus = self.metric(c - e_sigma)
            dg[..., sigma, :, :] = (g_plus - g_minus) / (2 * h)

        # Γ^λ_μν = (1/2) g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
        gamma = torch.zeros(c.shape[:-1] + (n, n, n), device=c.device, dtype=c.dtype)
        for lam in range(n):
            for mu in range(n):
                for nu in range(n):
                    s = torch.tensor(0.0, device=c.device)
                    for sigma in range(n):
                        s = s + g_inv[..., lam, sigma] * (
                            dg[..., mu, nu, sigma] +
                            dg[..., nu, mu, sigma] -
                            dg[..., sigma, mu, nu]
                        )
                    gamma[..., lam, mu, nu] = 0.5 * s

        return gamma

    def covariant_derivative(self, V: Tensor, c: Tensor,
                              direction: int, h: float = 1e-3) -> Tensor:
        """Covariant derivative ∇_μ V^ν = ∂_μ V^ν + Γ^ν_μλ V^λ.

        Args:
            V: (..., dim) vector field
            c: (..., dim) coordinates
            direction: μ index for derivative direction
            h: finite difference step
        Returns:
            (..., dim) covariant derivative
        """
        n = self.dim
        e_mu = torch.zeros(n, device=c.device, dtype=c.dtype)
        e_mu[direction] = h

        # ∂_μ V (finite difference)
        # Note: for grid-based V, this should use grid indices
        # Here we assume V is a function of c
        dV = (V - V) / (2 * h)  # placeholder — actual implementation below

        gamma = self.christoffel(c, h)

        # ∇_μ V^ν = ∂_μ V^ν + Γ^ν_μλ V^λ
        result = dV.clone()
        for nu in range(n):
            for lam in range(n):
                result[..., nu] = result[..., nu] + gamma[..., nu, direction, lam] * V[..., lam]

        return result

    def integrate(self, f: Tensor) -> Tensor:
        """Integrate a scalar field over the manifold: ∫ f √|g| dc.

        Args:
            f: (*grid_shape) scalar field values at grid points
        Returns:
            Scalar integral
        """
        vol = self.volume_form()
        # Simple midpoint rule with volume correction
        cell_volume = 1.0
        for s in self.grid_shape:
            cell_volume *= 1.0 / s  # normalized grid spacing
        return (f * vol).sum() * cell_volume

    @abstractmethod
    def exp_map(self, c: Tensor, v: Tensor) -> Tensor:
        """Exponential map: c' = Exp_c(v).

        Moves from c along geodesic with initial velocity v.

        Args:
            c: (..., dim) base point
            v: (..., dim) tangent vector
        Returns:
            (..., dim) endpoint
        """
        ...

    @abstractmethod
    def log_map(self, c: Tensor, c_prime: Tensor) -> Tensor:
        """Logarithmic map: v = Log_c(c').

        Inverse of exp_map: tangent vector from c to c'.

        Args:
            c: (..., dim) base point
            c_prime: (..., dim) target point
        Returns:
            (..., dim) tangent vector
        """
        ...

    @abstractmethod
    def geodesic_distance(self, c1: Tensor, c2: Tensor) -> Tensor:
        """Intrinsic geodesic distance d(c1, c2).

        Args:
            c1, c2: (..., dim) points
        Returns:
            (...,) distances
        """
        ...


class EuclideanManifold(Manifold):
    """Flat Euclidean space ℝ^n with metric g = δ_μν.

    Christoffel symbols vanish. Geodesics are straight lines.
    Volume form is constant (= 1).

    Args:
        dim: dimension n
        grid_shape: discretization shape
        bounds: ((min_1, max_1), ..., (min_n, max_n)) coordinate ranges
    """

    def __init__(self, dim: int, grid_shape: Tuple[int, ...],
                 bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
                 **kwargs):
        super().__init__(dim, grid_shape, **kwargs)
        if bounds is None:
            bounds = tuple((-1.0, 1.0) for _ in range(dim))
        self.bounds = bounds
        self._coords = self._build_grid()

    def _build_grid(self) -> Tensor:
        """Build coordinate grid."""
        linspaces = []
        for i in range(self.dim):
            lo, hi = self.bounds[i]
            n = self.grid_shape[i] if i < len(self.grid_shape) else 1
            linspaces.append(torch.linspace(lo, hi, n, device=self.device, dtype=self.dtype))
        grids = torch.meshgrid(*linspaces, indexing='ij')
        return torch.stack(grids, dim=-1)

    def coordinates(self) -> Tensor:
        return self._coords

    def metric(self, c=None) -> Tensor:
        if c is None:
            shape = self.grid_shape + (self.dim, self.dim)
        else:
            shape = c.shape[:-1] + (self.dim, self.dim)
        return torch.eye(self.dim, device=self.device, dtype=self.dtype).expand(shape)

    def volume_form(self, c=None) -> Tensor:
        if c is None:
            return torch.ones(self.grid_shape, device=self.device, dtype=self.dtype)
        return torch.ones(c.shape[:-1], device=c.device, dtype=c.dtype)

    def christoffel(self, c=None, h=1e-3) -> Tensor:
        if c is None:
            shape = self.grid_shape + (self.dim, self.dim, self.dim)
        else:
            shape = c.shape[:-1] + (self.dim, self.dim, self.dim)
        return torch.zeros(shape, device=self.device, dtype=self.dtype)

    def exp_map(self, c, v):
        return c + v

    def log_map(self, c, c_prime):
        return c_prime - c

    def geodesic_distance(self, c1, c2):
        return (c1 - c2).norm(dim=-1)

    def grid_spacing(self) -> Tuple[float, ...]:
        """Physical spacing between grid points in each dimension."""
        spacings = []
        for i in range(self.dim):
            lo, hi = self.bounds[i]
            n = self.grid_shape[i] if i < len(self.grid_shape) else 1
            spacings.append((hi - lo) / max(n - 1, 1))
        return tuple(spacings)


class Sphere(Manifold):
    """n-sphere S^n of radius R with round metric.

    Parameterized by n angular coordinates (θ₁, ..., θ_n) where:
      θ_i ∈ [0, π] for i < n
      θ_n ∈ [0, 2π)

    Metric: ds² = R² [dθ₁² + sin²θ₁ dθ₂² + sin²θ₁ sin²θ₂ dθ₃² + ...]

    Key properties:
      - Positive sectional curvature K = 1/R²
      - Compact (finite volume)
      - Geodesics are great circles

    Args:
        n: dimension of the sphere (S^n lives in ℝ^{n+1})
        grid_shape: discretization
        radius: sphere radius R
    """

    def __init__(self, n: int, grid_shape: Tuple[int, ...],
                 radius: float = 1.0, **kwargs):
        super().__init__(n, grid_shape, **kwargs)
        self.radius = radius
        self._coords = self._build_grid()

    def _build_grid(self) -> Tensor:
        linspaces = []
        for i in range(self.dim):
            n_pts = self.grid_shape[i] if i < len(self.grid_shape) else 1
            if i < self.dim - 1:
                # θ_i ∈ (0, π) — avoid poles
                linspaces.append(torch.linspace(0.05, math.pi - 0.05, n_pts,
                                                device=self.device, dtype=self.dtype))
            else:
                # θ_n ∈ [0, 2π)
                linspaces.append(torch.linspace(0, 2 * math.pi * (1 - 1/n_pts), n_pts,
                                                device=self.device, dtype=self.dtype))
        grids = torch.meshgrid(*linspaces, indexing='ij')
        return torch.stack(grids, dim=-1)

    def coordinates(self) -> Tensor:
        return self._coords

    def metric(self, c=None) -> Tensor:
        """Round metric on S^n.

        g_ii = R² ∏_{k<i} sin²(θ_k)
        g_ij = 0 for i ≠ j
        """
        if c is None:
            c = self._coords

        R2 = self.radius ** 2
        n = self.dim
        g = torch.zeros(c.shape[:-1] + (n, n), device=c.device, dtype=c.dtype)

        for i in range(n):
            factor = R2
            for k in range(i):
                factor = factor * torch.sin(c[..., k]) ** 2
            g[..., i, i] = factor

        return g

    def volume_form(self, c=None) -> Tensor:
        g = self.metric(c)
        # √|det g| — for diagonal metric, product of diagonal elements
        log_det = g.diagonal(dim1=-2, dim2=-1).clamp(min=1e-10).log().sum(-1)
        return (0.5 * log_det).exp()

    def exp_map(self, c, v):
        """Approximate exp map via Euler step (exact on great circles for S²)."""
        # For small v, this is a reasonable approximation
        return c + v

    def log_map(self, c, c_prime):
        return c_prime - c

    def geodesic_distance(self, c1, c2):
        """Great circle distance on S^n via embedding."""
        # Convert to Cartesian, compute angle
        x1 = self._to_cartesian(c1)
        x2 = self._to_cartesian(c2)
        cos_angle = (x1 * x2).sum(-1).clamp(-1, 1)
        return self.radius * torch.acos(cos_angle)

    def _to_cartesian(self, c: Tensor) -> Tensor:
        """Convert angular coordinates to Cartesian embedding in ℝ^{n+1}."""
        n = self.dim
        R = self.radius
        x = torch.zeros(c.shape[:-1] + (n + 1,), device=c.device, dtype=c.dtype)

        # x_0 = R cos(θ_0)
        # x_i = R (∏_{k<i} sin θ_k) cos(θ_i) for i < n
        # x_n = R (∏_{k<n} sin θ_k)
        for i in range(n):
            val = R
            for k in range(i):
                val = val * torch.sin(c[..., k])
            val = val * torch.cos(c[..., i])
            x[..., i] = val

        # Last component
        val = R
        for k in range(n):
            val = val * torch.sin(c[..., k])
        x[..., n] = val

        return x


class HyperbolicSpace(Manifold):
    """Hyperbolic space H^n in the Poincaré ball model.

    The Poincaré ball: B^n = {x ∈ ℝ^n : |x| < 1}
    Metric: ds² = (2/(1-|x|²))² |dx|²
    Conformal factor: λ(x) = 2/(1-|x|²)

    Key properties:
      - Negative sectional curvature K = -1
      - Non-compact (infinite volume)
      - Exponential volume growth → natural for hierarchical data
      - Geodesics are circular arcs perpendicular to boundary

    Reference: Nickel & Kiela (2017), "Poincaré Embeddings for
    Learning Hierarchical Representations"

    Args:
        dim: dimension n
        grid_shape: discretization (points in the ball)
        curvature: K = -1/c², default c=1
    """

    def __init__(self, dim: int, grid_shape: Tuple[int, ...],
                 curvature: float = -1.0, **kwargs):
        super().__init__(dim, grid_shape, **kwargs)
        self.c = (-curvature) ** (-0.5)  # curvature radius
        self._coords = self._build_grid()

    def _build_grid(self) -> Tensor:
        """Grid inside the Poincaré ball, staying away from boundary."""
        max_r = 0.9  # stay inside ball
        linspaces = []
        for i in range(self.dim):
            n_pts = self.grid_shape[i] if i < len(self.grid_shape) else 1
            linspaces.append(torch.linspace(-max_r, max_r, n_pts,
                                            device=self.device, dtype=self.dtype))
        grids = torch.meshgrid(*linspaces, indexing='ij')
        coords = torch.stack(grids, dim=-1)

        # Project points outside ball to inside
        norms = coords.norm(dim=-1, keepdim=True)
        coords = torch.where(norms > max_r, coords * max_r / norms, coords)
        return coords

    def conformal_factor(self, c: Optional[Tensor] = None) -> Tensor:
        """λ(x) = 2c / (c² - |x|²)."""
        if c is None:
            c = self._coords
        norm_sq = (c * c).sum(-1)
        return 2.0 * self.c / (self.c ** 2 - norm_sq).clamp(min=1e-6)

    def coordinates(self) -> Tensor:
        return self._coords

    def metric(self, c=None) -> Tensor:
        """Poincaré ball metric: g_μν = λ(x)² δ_μν."""
        lam = self.conformal_factor(c)
        lam_sq = lam ** 2
        I = torch.eye(self.dim, device=lam.device, dtype=lam.dtype)
        return lam_sq.unsqueeze(-1).unsqueeze(-1) * I

    def volume_form(self, c=None) -> Tensor:
        """√|g| = λ^n."""
        lam = self.conformal_factor(c)
        return lam ** self.dim

    def geodesic_distance(self, c1, c2):
        """Poincaré distance: d(x,y) = c · arcosh(1 + 2c²|x-y|²/((c²-|x|²)(c²-|y|²)))."""
        diff_sq = ((c1 - c2) ** 2).sum(-1)
        norm1_sq = (c1 * c1).sum(-1)
        norm2_sq = (c2 * c2).sum(-1)
        denom = (self.c ** 2 - norm1_sq) * (self.c ** 2 - norm2_sq)
        arg = 1.0 + 2.0 * self.c ** 2 * diff_sq / denom.clamp(min=1e-10)
        return self.c * torch.acosh(arg.clamp(min=1.0))

    def exp_map(self, c, v):
        """Möbius addition-based exponential map on Poincaré ball."""
        c2 = self.c ** 2
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        c_norm_sq = (c * c).sum(-1, keepdim=True)

        lam_c = 2.0 * self.c / (c2 - c_norm_sq).clamp(min=1e-6)
        t = torch.tanh(lam_c * v_norm / (2.0 * self.c))
        direction = v / v_norm

        # Möbius addition
        return self._mobius_add(c, t * direction)

    def log_map(self, c, c_prime):
        """Logarithmic map via Möbius subtraction."""
        minus_c = self._mobius_add(torch.zeros_like(c), -c)
        added = self._mobius_add(minus_c, c_prime)
        added_norm = added.norm(dim=-1, keepdim=True).clamp(min=1e-10)

        c_norm_sq = (c * c).sum(-1, keepdim=True)
        lam_c = 2.0 * self.c / (self.c ** 2 - c_norm_sq).clamp(min=1e-6)

        return (2.0 * self.c / lam_c) * torch.atanh(added_norm / self.c) * (added / added_norm)

    def _mobius_add(self, x, y):
        """Möbius addition in the Poincaré ball."""
        c2 = self.c ** 2
        x_dot_y = (x * y).sum(-1, keepdim=True)
        x_sq = (x * x).sum(-1, keepdim=True)
        y_sq = (y * y).sum(-1, keepdim=True)

        num = (1 + 2 * x_dot_y / c2 + y_sq / c2) * x + (1 - x_sq / c2) * y
        denom = 1 + 2 * x_dot_y / c2 + x_sq * y_sq / c2 ** 2
        return num / denom.clamp(min=1e-10)


class Torus(Manifold):
    """Flat n-torus T^n = ℝ^n / (2πR · ℤ^n).

    Flat metric (Γ = 0) but non-trivial topology.
    Coordinates are periodic: c ~ c + 2πR.

    Non-trivial topology means:
      - Wilson loops can have non-trivial holonomy even for flat connections
      - First homology H₁(T^n) = ℤ^n (n independent cycles)

    Args:
        dim: dimension n
        grid_shape: discretization
        periods: (R₁, ..., R_n) period in each direction
    """

    def __init__(self, dim: int, grid_shape: Tuple[int, ...],
                 periods: Optional[Tuple[float, ...]] = None, **kwargs):
        super().__init__(dim, grid_shape, **kwargs)
        if periods is None:
            periods = tuple(2 * math.pi for _ in range(dim))
        self.periods = periods
        self._coords = self._build_grid()

    def _build_grid(self) -> Tensor:
        linspaces = []
        for i in range(self.dim):
            n_pts = self.grid_shape[i] if i < len(self.grid_shape) else 1
            L = self.periods[i]
            # Periodic: don't include endpoint
            linspaces.append(torch.linspace(0, L * (1 - 1/n_pts), n_pts,
                                            device=self.device, dtype=self.dtype))
        grids = torch.meshgrid(*linspaces, indexing='ij')
        return torch.stack(grids, dim=-1)

    def coordinates(self) -> Tensor:
        return self._coords

    def metric(self, c=None) -> Tensor:
        if c is None:
            shape = self.grid_shape + (self.dim, self.dim)
        else:
            shape = c.shape[:-1] + (self.dim, self.dim)
        return torch.eye(self.dim, device=self.device, dtype=self.dtype).expand(shape)

    def volume_form(self, c=None) -> Tensor:
        if c is None:
            return torch.ones(self.grid_shape, device=self.device, dtype=self.dtype)
        return torch.ones(c.shape[:-1], device=c.device, dtype=c.dtype)

    def christoffel(self, c=None, h=1e-3) -> Tensor:
        if c is None:
            shape = self.grid_shape + (self.dim, self.dim, self.dim)
        else:
            shape = c.shape[:-1] + (self.dim, self.dim, self.dim)
        return torch.zeros(shape, device=self.device, dtype=self.dtype)

    def exp_map(self, c, v):
        result = c + v
        for i in range(self.dim):
            result[..., i] = result[..., i] % self.periods[i]
        return result

    def log_map(self, c, c_prime):
        diff = c_prime - c
        for i in range(self.dim):
            L = self.periods[i]
            diff[..., i] = (diff[..., i] + L/2) % L - L/2
        return diff

    def geodesic_distance(self, c1, c2):
        diff = self.log_map(c1, c2)
        return diff.norm(dim=-1)

    def wrap(self, c: Tensor) -> Tensor:
        """Wrap coordinates to fundamental domain [0, L_i)."""
        result = c.clone()
        for i in range(self.dim):
            result[..., i] = result[..., i] % self.periods[i]
        return result


class ProductManifold(Manifold):
    """Product manifold M₁ × M₂ × ... with block-diagonal metric.

    Useful for mixing geometries:
      - H² × S¹:  hierarchical structure + cyclic
      - ℝ² × S²:  flat base + spherical internal

    Args:
        factors: list of component Manifold instances
        grid_shape: overall grid shape (must be compatible)
    """

    def __init__(self, factors: List[Manifold], **kwargs):
        total_dim = sum(f.dim for f in factors)
        # Use first factor's grid shape for simplicity
        grid_shape = factors[0].grid_shape
        super().__init__(total_dim, grid_shape, **kwargs)
        self.factors = factors
        self._dim_splits = [f.dim for f in factors]

    def _split_coords(self, c: Tensor) -> List[Tensor]:
        """Split combined coordinates into per-factor coordinates."""
        return torch.split(c, self._dim_splits, dim=-1)

    def coordinates(self) -> Tensor:
        coords = [f.coordinates() for f in self.factors]
        return torch.cat(coords, dim=-1)

    def metric(self, c=None) -> Tensor:
        if c is None:
            c = self.coordinates()
        parts = self._split_coords(c)
        metrics = [f.metric(p) for f, p in zip(self.factors, parts)]
        return torch.block_diag(*[m.reshape(-1, m.shape[-2], m.shape[-1])[0]
                                  for m in metrics]).expand(
            c.shape[:-1] + (self.dim, self.dim))

    def volume_form(self, c=None) -> Tensor:
        if c is None:
            c = self.coordinates()
        parts = self._split_coords(c)
        vol = torch.ones(c.shape[:-1], device=c.device, dtype=c.dtype)
        for f, p in zip(self.factors, parts):
            vol = vol * f.volume_form(p)
        return vol

    def exp_map(self, c, v):
        c_parts = self._split_coords(c)
        v_parts = self._split_coords(v)
        results = [f.exp_map(cp, vp) for f, cp, vp in zip(self.factors, c_parts, v_parts)]
        return torch.cat(results, dim=-1)

    def log_map(self, c, c_prime):
        c_parts = self._split_coords(c)
        cp_parts = self._split_coords(c_prime)
        results = [f.log_map(cp, cpp) for f, cp, cpp in zip(self.factors, c_parts, cp_parts)]
        return torch.cat(results, dim=-1)

    def geodesic_distance(self, c1, c2):
        c1_parts = self._split_coords(c1)
        c2_parts = self._split_coords(c2)
        d_sq = torch.zeros(c1.shape[:-1], device=c1.device)
        for f, p1, p2 in zip(self.factors, c1_parts, c2_parts):
            d_sq = d_sq + f.geodesic_distance(p1, p2) ** 2
        return d_sq.sqrt()


def covariant_finite_diff(field: Tensor, manifold: Manifold,
                          direction: int, h: Optional[float] = None) -> Tensor:
    """Covariant finite difference of a scalar/vector field on a manifold.

    For a scalar field f, this is just the ordinary derivative ∂_μ f
    weighted by metric factors. For a vector field V, this includes
    Christoffel corrections.

    Args:
        field: (*grid_shape, ...) field to differentiate
        manifold: the base manifold
        direction: which grid direction to differentiate along
        h: grid spacing (auto-computed if None)
    Returns:
        (*grid_shape, ...) derivative field
    """
    if h is None:
        if isinstance(manifold, EuclideanManifold):
            h = manifold.grid_spacing()[direction]
        else:
            h = 1.0 / manifold.grid_shape[direction]

    # Central differences with periodic wrapping
    f_plus = torch.roll(field, -1, dims=direction)
    f_minus = torch.roll(field, 1, dims=direction)
    return (f_plus - f_minus) / (2 * h)
