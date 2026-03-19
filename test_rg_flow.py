"""
End-to-end test: Agents on curved manifolds + Renormalization Group flow
"""

import torch
torch.manual_seed(42)

print("=" * 70)
print("END-TO-END TEST: Manifold Systems + RG Flow")
print("=" * 70)


# ============================================================
# Part 1: Agents on Curved Manifolds
# ============================================================
print("\n[1] Agents on Curved Manifolds with Lattice Gauge")
print("-" * 55)

from gauge_agent.manifolds import Sphere, HyperbolicSpace, Torus, EuclideanManifold
from gauge_agent.manifold_system import ManifoldAgentSystem

# --- Agents on Torus T² ---
print("\n  (a) 4 agents on T² with non-flat connections:")
T2 = Torus(dim=2, grid_shape=(8, 8))
mas_torus = ManifoldAgentSystem(
    N_agents=4, K=3, manifold=T2,
    init_twist_scale=0.05, ym_coupling=0.5
)

history_torus = mas_torus.evolve(30, lr=0.005, verbose=True)
print(f"      Final S = {history_torus[-1]['total']:.4f}")

# Holonomy diagnostics
hol = mas_torus.holonomy_spectrum()
for agent_key, diag in hol.items():
    print(f"      {agent_key}: mean_curv = {diag['mean_curvature']:.4f}, "
          f"WL = {diag['wilson_loop_traces']}")

# --- Agents on H² ---
print("\n  (b) 4 agents on H² (hyperbolic, Poincaré ball):")
H2 = HyperbolicSpace(dim=2, grid_shape=(8, 8))
mas_hyp = ManifoldAgentSystem(
    N_agents=4, K=3, manifold=H2,
    init_twist_scale=0.05, ym_coupling=0.5
)

history_hyp = mas_hyp.evolve(30, lr=0.005, verbose=True)
print(f"      Final S = {history_hyp[-1]['total']:.4f}")

# --- Agents on S² ---
print("\n  (c) 4 agents on S² (sphere):")
S2 = Sphere(n=2, grid_shape=(8, 16))
mas_sphere = ManifoldAgentSystem(
    N_agents=4, K=3, manifold=S2,
    init_twist_scale=0.05, ym_coupling=0.5
)

history_sphere = mas_sphere.evolve(30, lr=0.005, verbose=True)
print(f"      Final S = {history_sphere[-1]['total']:.4f}")


# ============================================================
# Part 2: Renormalization Group Flow
# ============================================================
print("\n\n[2] Renormalization Group Flow")
print("-" * 55)

from gauge_agent.renormalization import (
    RenormalizationGroupFlow, UniversalityTest
)

# --- RG flow with KL-proximity blocking ---
print("\n  (a) RG flow: N=32 agents, KL-proximity blocking:")
rg = RenormalizationGroupFlow(
    N_agents=32, K=3, grid_shape=(),
    blocking='kl_proximity',
    n_equilibrate=30,
    max_scales=10,
    lr=0.02
)
analysis_kl = rg.run(verbose=True)

# --- RG flow with majority-rule blocking ---
print("\n\n  (b) RG flow: N=32 agents, majority-rule blocking:")
rg2 = RenormalizationGroupFlow(
    N_agents=32, K=3, grid_shape=(),
    blocking='majority_rule',
    block_size=2,
    n_equilibrate=30,
    max_scales=10,
    lr=0.02
)
analysis_mr = rg2.run(verbose=True)

# --- Universality test ---
print("\n\n  (c) Universality test: do different blockings give same fixed point?")
comparison = UniversalityTest.compare_flows(analysis_kl, analysis_mr)
print(f"      Coupling distance: {comparison['coupling_distance']:.4f}")
print(f"      Relative distance: {comparison['relative_distance']:.4f}")
print(f"      Scaling dim diff:  {comparison['scaling_dim_diff']:.4f}")
print(f"      Same class:        {comparison['same_universality_class']}")


# --- RG flow with consensus blocking ---
print("\n\n  (d) RG flow: N=16 agents, consensus blocking:")
rg3 = RenormalizationGroupFlow(
    N_agents=16, K=3, grid_shape=(),
    blocking='consensus',
    n_equilibrate=40,
    max_scales=8,
    kl_threshold=0.5,
    lr=0.02
)
analysis_consensus = rg3.run(verbose=True)


# ============================================================
# Part 3: RG on Curved Manifolds (2D grid)
# ============================================================
print("\n\n[3] RG Flow on 2D Grid (Spatial Structure)")
print("-" * 55)

print("\n  RG flow: N=16 agents, K=3, grid_shape=(4,4):")
rg_2d = RenormalizationGroupFlow(
    N_agents=16, K=3, grid_shape=(4, 4),
    blocking='kl_proximity',
    n_equilibrate=20,
    max_scales=6,
    lr=0.02
)
analysis_2d = rg_2d.run(verbose=True)


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)
print(f"""
Results:
  1. Manifold systems: agents evolve on T², H², S² with non-flat connections
     - Volume-weighted free energy with √|g|
     - Yang-Mills regularization drives toward flat connections
     - Holonomy measured via Wilson loop traces

  2. RG Flow:
     - KL-proximity blocking: {len(analysis_kl['N_agents'])} scales, {analysis_kl['N_agents']}
     - Majority-rule blocking: {len(analysis_mr['N_agents'])} scales, {analysis_mr['N_agents']}
     - Consensus blocking: {len(analysis_consensus['N_agents'])} scales, {analysis_consensus['N_agents']}
     - Scaling dimension d_f: {analysis_kl['scaling_dimension']:.3f} (KL), {analysis_mr['scaling_dimension']:.3f} (MR)
     - Fixed points found: {len(analysis_kl['fixed_points'])} (KL), {len(analysis_mr['fixed_points'])} (MR)

  3. Universality: same class = {comparison['same_universality_class']}
     (relative coupling distance = {comparison['relative_distance']:.4f})

New modules:
  gauge_agent/manifold_system.py   — Unified ManifoldAgentSystem
  gauge_agent/renormalization.py   — Full RG: blocking, beta functions, fixed points, exponents
""")
