"""
Gauge-Theoretic Multi-Agent Active Inference Framework
=======================================================

A PyTorch implementation of the participatory "It From Bit" universe,
building GL(K) gauge theory on principal bundles with variational free energy.

Layers (bottom-up):
  0. lie_groups      — GL(K) operations, transport operators
  1. statistical_manifold — Gaussian distributions, KL, Fisher metric
  2. gauge_structure  — Gauge frames, connections, curvature
  3. agents          — Agents as bundle sections (q_i, p_i, Ω_i)
  4. free_energy     — The complete variational free energy functional
  5. attention       — Softmax attention from mixture-of-sources
  6. dynamics        — Natural gradient flow, Hamiltonian dynamics
  7. meta_agents     — Consensus detection, meta-agent formation
  8. pullback        — Induced Fisher metrics, sector decomposition
  9. hierarchical_emergence — Species-gated soft hierarchy
  10. mass           — Information-geometric mass M = ∂²F/∂ξ²

Reference: Dennis (2026), "A Theoretical and Computational Implementation
of a Participatory 'It From Bit' Universe"
"""

__version__ = "0.1.0"
__author__ = "Robert C. Dennis"
