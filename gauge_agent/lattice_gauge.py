"""
Module B: Lattice Gauge Theory — Non-Flat Connections
======================================================

The critical upgrade from vertex-local to full gauge theory.

VERTEX-LOCAL (current, Layer 2):
  Ω_ij = Ω_i Ω_j⁻¹   →   cocycle holds   →   F_μν = 0   →   no holonomy

LATTICE GAUGE (this module):
  Ω_ij = Ω_i · V_ij · Ω_j⁻¹   where V_ij ∈ GL(K) is an independent edge twist

  or fully independent: each directed edge (i→j) carries U_ij ∈ GL(K)

This gives:
  - Non-trivial holonomy: W(□) = U₁₂ U₂₃ U₃₄ U₄₁ ≠ I
  - Path-dependent information transport
  - Yang-Mills action S_YM = Σ_□ ‖W(□) - I‖² as regularizer
  - The gauge curvature conjecture becomes testable

The discretized manifold is represented as a graph/mesh:
  - Vertices: grid points where agents live
  - Edges: directed links carrying transport operators
  - Faces/plaquettes: elementary loops for curvature measurement

Reference: Dennis (2026), Sections 2.8 (Curvature), 7.7 (Gauge Curvature Conjecture)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple, List
import itertools


class LatticeGaugeField(nn.Module):
    """Lattice gauge field with link variables on edges.

    Two modes:
      1. VERTEX + EDGE (mixed): Ω_ij = Ω_i · V_ij · Ω_j⁻¹
         - Vertex frames Ω_i provide bulk transport
         - Edge twists V_ij provide curvature
         - Reduces to vertex-local when V_ij = I

      2. LINK-ONLY (pure lattice): each edge has independent U_ij ∈ GL(K)
         - Maximum flexibility
         - Requires Yang-Mills regularization

    For a regular grid of shape (*grid_shape), edges connect
    nearest neighbors in each dimension. On a torus, edges wrap.

    Args:
        K: fiber dimension (GL(K) group)
        grid_shape: base manifold discretization
        mode: 'mixed' or 'link_only'
        n_agents: number of agents (for vertex frames in mixed mode)
        init_twist_scale: perturbation of edge twists from identity
        periodic: whether to use periodic boundary conditions
    """

    def __init__(self, K: int, grid_shape: Tuple[int, ...],
                 mode: str = 'mixed',
                 n_agents: int = 1,
                 init_twist_scale: float = 0.01,
                 periodic: bool = True):
        super().__init__()
        self.K = K
        self.grid_shape = grid_shape
        self.n_dims = len(grid_shape)
        self.mode = mode
        self.n_agents = n_agents
        self.periodic = periodic

        # Edge twist variables V_ij along each dimension
        # Shape: (n_agents, n_dims, *grid_shape, K, K)
        # V[agent, dim, *grid_idx] = twist on edge from grid_idx
        #                            to grid_idx + e_dim
        n_edges_shape = (n_agents, self.n_dims) + grid_shape + (K, K)
        I = torch.eye(K)
        twists = I.reshape((1, 1) + (1,) * self.n_dims + (K, K))
        twists = twists.expand(n_edges_shape).clone()
        twists = twists + init_twist_scale * torch.randn(n_edges_shape)
        self.twists = nn.Parameter(twists)

        if mode == 'mixed':
            # Vertex frames (same as before)
            vertex_shape = (n_agents,) + grid_shape + (K, K)
            frames = I.reshape((1,) + (1,) * self.n_dims + (K, K))
            frames = frames.expand(vertex_shape).clone()
            frames = frames + 0.1 * torch.randn(vertex_shape)
            self.vertex_frames = nn.Parameter(frames)

    def get_link(self, agent: int, dim: int,
                 grid_idx: Optional[Tuple[int, ...]] = None) -> Tensor:
        """Get link variable for edge from grid_idx along direction dim.

        Args:
            agent: agent index
            dim: base manifold direction
            grid_idx: optional specific grid point
        Returns:
            (*grid, K, K) or (K, K) link variable
        """
        if grid_idx is not None:
            return self.twists[(agent, dim) + grid_idx]
        return self.twists[agent, dim]

    def transport_along_edge(self, agent_i: int, agent_j: int,
                              dim: int, grid_idx: Optional[Tuple[int, ...]] = None
                              ) -> Tensor:
        """Full transport operator along one edge.

        Mixed mode: Ω_ij = Ω_i · V_ij · Ω_j⁻¹
        Link mode:  just V_ij (renamed U_ij)

        Args:
            agent_i: receiving agent
            agent_j: sending agent
            dim: edge direction
            grid_idx: grid location
        Returns:
            (..., K, K) transport operator
        """
        V = self.get_link(agent_i, dim, grid_idx)

        if self.mode == 'mixed':
            if grid_idx is not None:
                omega_i = self.vertex_frames[(agent_i,) + grid_idx]
                omega_j = self.vertex_frames[(agent_j,) + grid_idx]
            else:
                omega_i = self.vertex_frames[agent_i]
                omega_j = self.vertex_frames[agent_j]
            return omega_i @ V @ torch.linalg.inv(omega_j)
        else:
            return V

    def plaquette(self, agent: int, dim1: int, dim2: int,
                  grid_idx: Optional[Tuple[int, ...]] = None) -> Tensor:
        """Plaquette (Wilson loop around elementary square).

        W(□) = U(c, c+ê₁) · U(c+ê₁, c+ê₁+ê₂) · U(c+ê₁+ê₂, c+ê₂)⁻¹ · U(c+ê₂, c)⁻¹

        For vertex-local V_ij = I, this gives identity (flat).
        Non-trivial V_ij → non-identity → curvature.

        Args:
            agent: agent index
            dim1, dim2: plaquette plane directions (μ, ν)
            grid_idx: bottom-left corner of plaquette
        Returns:
            (*grid, K, K) or (K, K) plaquette holonomy
        """
        V = self.twists[agent]  # (n_dims, *grid, K, K)

        # U₁ = V(c, dim1) — forward along dim1
        U1 = V[dim1]

        # U₂ = V(c+ê₁, dim2) — forward along dim2 from shifted point
        U2 = torch.roll(V[dim2], -1, dims=dim1)

        # U₃⁻¹ = V(c+ê₂, dim1)⁻¹ — backward along dim1 from shifted point
        U3_inv = torch.linalg.inv(torch.roll(V[dim1], -1, dims=dim2))

        # U₄⁻¹ = V(c, dim2)⁻¹ — backward along dim2
        U4_inv = torch.linalg.inv(V[dim2])

        W = U1 @ U2 @ U3_inv @ U4_inv

        if grid_idx is not None:
            return W[grid_idx]
        return W

    def holonomy(self, agent: int, path: List[Tuple[int, int]]) -> Tensor:
        """Holonomy around an arbitrary closed path.

        Path is specified as list of (dim, direction) steps where
        direction is +1 (forward) or -1 (backward).

        The holonomy is the ordered product of link variables along the path.

        Args:
            agent: agent index
            path: list of (dimension, direction) steps
        Returns:
            (K, K) holonomy matrix
        """
        V = self.twists[agent]  # (n_dims, *grid, K, K)
        # Start at origin — for simplicity, work with grid center
        center = tuple(s // 2 for s in self.grid_shape)

        H = torch.eye(self.K, device=V.device, dtype=V.dtype)
        current = list(center)

        for dim, direction in path:
            if direction > 0:
                idx = tuple(current)
                link = V[(dim,) + idx]
                H = H @ link
                current[dim] = (current[dim] + 1) % self.grid_shape[dim]
            else:
                current[dim] = (current[dim] - 1) % self.grid_shape[dim]
                idx = tuple(current)
                link = V[(dim,) + idx]
                H = H @ torch.linalg.inv(link)

        return H

    def wilson_loop_trace(self, agent: int, dim1: int, dim2: int) -> Tensor:
        """tr(W(□)) / K — normalized Wilson loop trace.

        = 1 for flat connections, < 1 for curved.
        This is the lattice gauge theory order parameter.

        Returns:
            (*grid,) Wilson loop traces
        """
        W = self.plaquette(agent, dim1, dim2)
        return W.diagonal(dim1=-2, dim2=-1).sum(-1) / self.K

    def yang_mills_action(self, coupling: float = 1.0) -> Tensor:
        """Yang-Mills action on the lattice.

        S_YM = (β/K) Σ_{agent} Σ_{□} [K - Re tr(W(□))]

        where β is the coupling constant. Minimized when all plaquettes
        are identity (flat connection).

        This regularizes the gauge field toward flatness while allowing
        non-trivial curvature when the data demands it.

        Args:
            coupling: Yang-Mills coupling β
        Returns:
            Scalar action
        """
        total = torch.tensor(0.0, device=self.twists.device)

        for agent in range(self.n_agents):
            for d1 in range(self.n_dims):
                for d2 in range(d1 + 1, self.n_dims):
                    W = self.plaquette(agent, d1, d2)
                    # K - Re tr(W)
                    trace_re = W.diagonal(dim1=-2, dim2=-1).sum(-1)
                    action_density = self.K - trace_re
                    total = total + action_density.sum()

        return coupling / self.K * total

    def curvature_norm(self, agent: int) -> Tensor:
        """Frobenius norm of plaquette deviation from identity: ‖W - I‖_F.

        Scalar measure of curvature at each grid point.

        Returns:
            (*grid,) curvature magnitudes
        """
        total = torch.zeros(self.grid_shape, device=self.twists.device)
        I = torch.eye(self.K, device=self.twists.device)
        count = 0

        for d1 in range(self.n_dims):
            for d2 in range(d1 + 1, self.n_dims):
                W = self.plaquette(agent, d1, d2)
                deviation = W - I.expand_as(W)
                total = total + torch.norm(deviation, p='fro', dim=(-2, -1))
                count += 1

        return total / max(count, 1)

    def parallel_transport(self, agent: int, field: Tensor,
                           dim: int, steps: int = 1) -> Tensor:
        """Parallel transport a vector field along dimension dim.

        Applies link variables sequentially: v' = U_n · ... · U_2 · U_1 · v

        Args:
            agent: agent index
            field: (*grid, K) or (*grid, K, K) field to transport
            dim: direction of transport
            steps: number of lattice steps
        Returns:
            Transported field of same shape
        """
        V = self.twists[agent, dim]  # (*grid, K, K)
        result = field.clone()

        for s in range(steps):
            if field.dim() == len(self.grid_shape) + 1:
                # Vector field: v' = V @ v
                V_shifted = torch.roll(V, -s, dims=dim)
                result = (V_shifted @ result.unsqueeze(-1)).squeeze(-1)
            else:
                # Matrix field: M' = V @ M @ V^T
                V_shifted = torch.roll(V, -s, dims=dim)
                result = V_shifted @ result @ V_shifted.transpose(-2, -1)

        return result


class WilsonAction(nn.Module):
    """Wilson action for lattice gauge theory regularization.

    S = β Σ_□ [1 - (1/K) Re tr(W(□))]

    Drives gauge field toward flat connections. The coupling β controls
    the strength of this regularization — larger β means flatter connections.

    In the continuum limit: S → (β/4) ∫ tr(F_μν F^μν) √|g| d^n c

    Args:
        lattice: LatticeGaugeField
        coupling: Yang-Mills coupling β
    """

    def __init__(self, lattice: LatticeGaugeField, coupling: float = 1.0):
        super().__init__()
        self.lattice = lattice
        self.coupling = coupling

    def forward(self) -> Dict[str, Tensor]:
        """Compute Wilson action and diagnostics.

        Returns:
            Dict with 'action', 'mean_plaquette', 'curvature_per_agent'
        """
        action = self.lattice.yang_mills_action(self.coupling)

        # Diagnostics
        mean_plaq = torch.tensor(0.0, device=action.device)
        curvatures = []
        count = 0

        for agent in range(self.lattice.n_agents):
            c = self.lattice.curvature_norm(agent)
            curvatures.append(c.mean().item())
            for d1 in range(self.lattice.n_dims):
                for d2 in range(d1 + 1, self.lattice.n_dims):
                    wl = self.lattice.wilson_loop_trace(agent, d1, d2)
                    mean_plaq = mean_plaq + wl.mean()
                    count += 1

        mean_plaq = mean_plaq / max(count, 1)

        return {
            'action': action,
            'mean_plaquette': mean_plaq,
            'curvature_per_agent': curvatures,
        }
