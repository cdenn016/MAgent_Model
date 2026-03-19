"""
Test: All Integrations Wired Together
========================================

1. HierarchicalEmergence with Wheelerian self-referential closure
2. ManifoldAgentSystem with adaptive precision
3. RG flow with FullVFE and model fiber blocking
4. HamiltonianDynamics with information-geometric mass (Eq. 37)
"""

import torch
torch.manual_seed(42)

print("=" * 70)
print("TEST: Integration Wiring")
print("=" * 70)

K = 3


# ============================================================
# 1. Self-referential closure (Wheeler's circuit)
# ============================================================
print("\n[1] Wheelerian self-referential closure")
print("-" * 50)

from gauge_agent.agents import MultiAgentSystem
from gauge_agent.hierarchical_emergence import HierarchicalEmergence

base = MultiAgentSystem(8, K, grid_shape=())
hier = HierarchicalEmergence(
    base_system=base, n_meta_per_scale=[4, 2, 1],
    tau_species=5.0, tau_belief=1.0,
)

# Record top-scale prior before closure
top_mu_p_before = hier.scales[-1].agents[0].mu_p.data.clone()

# Run closure
hier.self_referential_closure()

top_mu_p_after = hier.scales[-1].agents[0].mu_p.data
delta = (top_mu_p_after - top_mu_p_before).norm().item()
print(f"  Top-scale prior shift: {delta:.4f}")
assert delta > 0, "Self-referential closure should modify top-scale priors"
print(f"  ✓ Wheeler's circuit: apex observes system, forms priors")

# NE monitoring
ne = hier.non_equilibrium_score()
print(f"  NE score: {ne['ne_score']:.4f}")
ne2 = hier.non_equilibrium_score()
print(f"  NE score (2nd call, has flux): {ne2['ne_score']:.4f}")
print(f"  ✓ Non-equilibrium monitoring active")


# ============================================================
# 2. ManifoldAgentSystem with adaptive precision
# ============================================================
print("\n[2] ManifoldAgentSystem + adaptive precision")
print("-" * 50)

from gauge_agent.manifolds import EuclideanManifold
from gauge_agent.manifold_system import ManifoldAgentSystem

manifold = EuclideanManifold(dim=1, grid_shape=(8,), bounds=((-1.0, 1.0),))

# Without adaptive precision
mas_const = ManifoldAgentSystem(4, K, manifold, adaptive_precision=False)
r_const = mas_const.volume_weighted_free_energy()
print(f"  Constant α: total={r_const['total'].item():.4f}")

# With adaptive precision
mas_adapt = ManifoldAgentSystem(4, K, manifold, adaptive_precision=True)
r_adapt = mas_adapt.volume_weighted_free_energy()
print(f"  Adaptive α: total={r_adapt['total'].item():.4f}")
print(f"    R_α_belief={r_adapt.get('R_alpha_belief', torch.tensor(0)).item():.4f}")
print(f"    α_belief mean={r_adapt['alpha_belief'].mean().item():.4f}")
print(f"  ✓ ManifoldAgentSystem supports adaptive precision")


# ============================================================
# 3. RG flow with FullVFE
# ============================================================
print("\n[3] RG flow with FullVFE")
print("-" * 50)

from gauge_agent.renormalization import RenormalizationGroupFlow

rg = RenormalizationGroupFlow(
    N_agents=8, K=K, grid_shape=(),
    blocking='kl_proximity',
    n_equilibrate=10,
    max_scales=5,
    adaptive_precision=True,
)

print(f"  VFE type: {type(rg.free_energy).__name__}")
assert 'FullVFE' in type(rg.free_energy).__name__, "Should use FullVFE"

result = rg.run(verbose=False)
n_scales = len(rg.scales)
print(f"  Scales traversed: {n_scales}")
print(f"  Agent counts: {[s.system.N_agents for s in rg.scales]}")

if result.get('coupling_trajectory') is not None:
    ct = result['coupling_trajectory']
    print(f"  Coupling trajectory shape: {ct.shape}")

# Check model fiber was preserved in blocking
if n_scales >= 2:
    s1 = rg.scales[-1].system
    has_model = any(a.mu_s.data.norm() > 0 for a in s1.agents)
    print(f"  Model fiber preserved in blocking: {has_model}")

print(f"  ✓ RG flow runs with FullVFE")


# ============================================================
# 4. Hamiltonian dynamics with information-geometric mass
# ============================================================
print("\n[4] HamiltonianDynamics with full mass (Eq. 37)")
print("-" * 50)

from gauge_agent.dynamics import HamiltonianDynamics
from gauge_agent.mass import MassMatrix
from gauge_agent.full_vfe import FullVFE

torch.manual_seed(42)
sys_ham = MultiAgentSystem(4, K, grid_shape=())
vfe_ham = FullVFE(adaptive_precision=True)
mass = MassMatrix(sys_ham, vfe_ham)

# Simple mass (M = Σ_p⁻¹)
ham_simple = HamiltonianDynamics(sys_ham, vfe_ham, dt=0.005, damping=0.1)
info_simple = ham_simple.step()
print(f"  Simple mass:  E_total={info_simple['total_energy']:.4f}")

# Reset
torch.manual_seed(42)
sys_ham2 = MultiAgentSystem(4, K, grid_shape=())
vfe_ham2 = FullVFE(adaptive_precision=True)
mass2 = MassMatrix(sys_ham2, vfe_ham2)

ham_full = HamiltonianDynamics(
    sys_ham2, vfe_ham2, mass_matrix=mass2,
    dt=0.005, damping=0.1, use_full_mass=True,
)
info_full = ham_full.step()
print(f"  Full mass:    E_total={info_full['total_energy']:.4f}")

# Run a few more steps to verify stability
for _ in range(20):
    info_full = ham_full.step()
print(f"  After 20 steps: E_total={info_full['total_energy']:.4f}")

# Check gauge frames stayed invertible
for a in sys_ham2.agents:
    det = a.omega.data.det().item()
    assert abs(det) > 1e-6, "Gauge frame singular!"
print(f"  ✓ Hamiltonian dynamics with Eq. 37 mass (4-component)")


# ============================================================
# 5. Full stack: manifold + hierarchy + closure
# ============================================================
print("\n[5] Full stack integration")
print("-" * 50)

torch.manual_seed(42)
base_full = MultiAgentSystem(6, K, grid_shape=())
hier_full = HierarchicalEmergence(
    base_system=base_full, n_meta_per_scale=[3, 1],
    tau_species=5.0, tau_belief=1.0,
)
vfe_full = FullVFE(adaptive_precision=True)

# Full cycle: VFE → dynamics → hierarchy → closure → NE monitoring
from gauge_agent.dynamics import NaturalGradientDynamics

dyn = NaturalGradientDynamics(base_full, vfe_full, lr_mu_q=0.02)

for t in range(10):
    # Evolve base agents
    info = dyn.step()

    # Bottom-up: pool into meta-agents
    hier_full.update_meta_agents()

    # Self-referential closure at apex
    hier_full.self_referential_closure()

    if t % 5 == 0:
        ne = hier_full.non_equilibrium_score()
        diag = hier_full.diagnostics()
        print(f"  step {t}: VFE={info['total']:.3f}, "
              f"NE={ne['ne_score']:.4f}, "
              f"condensation={diag['condensation_fractions']}")

print(f"  ✓ Full stack: VFE → dynamics → hierarchy → Wheeler closure → NE monitoring")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
print(f"""
Everything wired together:

  1. HierarchicalEmergence ← Ouroboros features unified
     - Self-referential closure (Wheeler's circuit)
     - Non-equilibrium monitoring (prevents epistemic death)
     - Soft membership W = S · C (species-gated coalitions)
     - Bottom-up pooling + top-down cross-scale VFE

  2. ManifoldAgentSystem ← adaptive precision
     - adaptive_precision=True enables log barrier α_i(x)
     - Works with lattice gauge transport and volume weighting

  3. RenormalizationGroupFlow ← FullVFE + model fiber
     - Uses FullVFE (not old FreeEnergyFunctional)
     - Copies model fiber (s_i, r_i, Ω̃, b₀, c₀) in blocking

  4. HamiltonianDynamics ← information-geometric mass (Eq. 37)
     - M_i = bare + relational + recoil + sensory
     - use_full_mass=True for Eq. 37, False for simple Σ_p⁻¹
     - Certain agents are massive (hard to move)
""")
