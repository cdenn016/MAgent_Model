"""
Agent Support Functions χ_i(c) on Arbitrary Manifolds
=======================================================

Each agent has a support domain U_i ⊆ C determining where it
maintains beliefs. The support function χ_i: C → [0,1] with:
  χ_i(c) = 1 inside U_i
  χ_i(c) = 0 outside U_i
  smooth transition at boundary

Overlaps χ_ij = χ_i · χ_j determine where agents can interact.

The free energy integral uses these:
  ∫_C χ_i(c) KL(q_i||p_i) √|g| dc
  ∫_C χ_ij(c) β_ij KL(q_i||Ω_ij[q_j]) √|g| dc

Reference: Dennis (2026), Eq. (23) — support functions in the action
"""

import torch
from torch import Tensor
from typing import Optional, Tuple
from gauge_agent.manifolds import Manifold


def full_support(grid_shape: Tuple[int, ...],
                 device: str = 'cpu') -> Tensor:
    """Full support: χ(c) = 1 everywhere."""
    return torch.ones(grid_shape, device=device)


def ball_support(manifold: Manifold, center: Tensor,
                 radius: float, sharpness: float = 10.0) -> Tensor:
    """Geodesic ball support: smooth χ centered at a point.

    χ(c) = σ(sharpness · (radius - d(c, center)))

    where σ is the sigmoid and d is geodesic distance.

    Args:
        manifold: the base manifold
        center: (dim,) center point
        radius: support radius
        sharpness: steepness of the boundary (higher = sharper)
    Returns:
        (*grid_shape,) support function
    """
    coords = manifold.coordinates()  # (*grid, dim)
    center_expanded = center.expand_as(coords)
    distances = manifold.geodesic_distance(coords, center_expanded)
    return torch.sigmoid(sharpness * (radius - distances))


def annular_support(manifold: Manifold, center: Tensor,
                    inner_radius: float, outer_radius: float,
                    sharpness: float = 10.0) -> Tensor:
    """Annular support: nonzero between two radii.

    χ(c) = σ(s·(d-r_in)) · σ(s·(r_out-d))
    """
    coords = manifold.coordinates()
    center_expanded = center.expand_as(coords)
    d = manifold.geodesic_distance(coords, center_expanded)
    inner = torch.sigmoid(sharpness * (d - inner_radius))
    outer = torch.sigmoid(sharpness * (outer_radius - d))
    return inner * outer


def half_space_support(manifold: Manifold, normal_dir: int,
                       threshold: float, sharpness: float = 10.0) -> Tensor:
    """Half-space support: χ = 1 where c^μ > threshold along normal_dir.

    Useful for partitioning the manifold between agents.
    """
    coords = manifold.coordinates()
    return torch.sigmoid(sharpness * (coords[..., normal_dir] - threshold))


def overlap_matrix(supports: Tensor) -> Tensor:
    """Compute pairwise overlaps χ_ij = χ_i · χ_j.

    Args:
        supports: (N, *grid_shape) agent support functions
    Returns:
        (N, N, *grid_shape) overlap matrix
    """
    N = supports.shape[0]
    # χ_i: (N, 1, *grid), χ_j: (1, N, *grid)
    return supports.unsqueeze(1) * supports.unsqueeze(0)


def volume_weighted_integral(field: Tensor, chi: Tensor,
                              volume_form: Tensor) -> Tensor:
    """∫ χ(c) f(c) √|g| dc — proper integral with support and volume.

    Args:
        field: (*grid_shape) scalar field
        chi: (*grid_shape) support function
        volume_form: (*grid_shape) √|g|
    Returns:
        Scalar integral value
    """
    integrand = field * chi * volume_form
    # Simple sum (midpoint rule — grid spacing absorbed into volume_form)
    return integrand.sum()
