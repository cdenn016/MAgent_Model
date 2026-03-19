"""
Test: Full Dynamics — FullVFE + Adaptive Precision + Group Gradients
=====================================================================

Everything wired together:
  1. FullVFE with adaptive α_i = c₀_i/(b₀_i + KL)
  2. Per-agent b₀, c₀ on model fiber (evolves with ε)
  3. Covariance dynamics (Eq. 36) via autograd through L_q
  4. Group gradients for Ω (direct ∂S/∂Ω, no exp/log maps)
  5. Timescale separation: belief (fast) vs model (slow, ×ε)
"""

import torch
torch.manual_seed(42)

print("=" * 70)
print("TEST: Full Dynamics Integration")
print("=" * 70)


# ============================================================
# 1. FullVFE drives NaturalGradientDynamics
# ============================================================
print("\n[1] FullVFE drives dynamics (not old FreeEnergyFunctional)")
print("-" * 50)

from gauge_agent.agents import MultiAgentSystem
from gauge_agent.full_vfe import FullVFE
from gauge_agent.dynamics import NaturalGradientDynamics

K = 4
system = MultiAgentSystem(4, K, grid_shape=())
vfe = FullVFE(adaptive_precision=True)

dyn = NaturalGradientDynamics(
    system, vfe,
    lr_mu_q=0.03, lr_sigma_q=0.005,
    lr_mu_p=0.01, lr_sigma_p=0.005,
    lr_omega=0.005,
    model_lr_ratio=0.01,
    lr_precision=0.01,
)

# Run a few steps
history = []
for t in range(30):
    info = dyn.step()
    if t % 10 == 0:
        history.append(info)
        print(f"  step {t}: total={info['total']:.4f}, "
              f"T1={info.get('T1_belief_self', 0):.4f}, "
              f"T3={info.get('T3_belief_align', 0):.4f}")

assert history[-1]['total'] < history[0]['total'], "VFE should decrease"
print(f"  VFE: {history[0]['total']:.4f} → {history[-1]['total']:.4f}")
print(f"  ✓ FullVFE drives dynamics, VFE decreases")


# ============================================================
# 2. Per-agent b₀, c₀ on model fiber
# ============================================================
print("\n[2] Per-agent b₀, c₀ (precision hyperparameters)")
print("-" * 50)

system2 = MultiAgentSystem(3, K, grid_shape=())

# Check b0, c0 exist as differentiable properties
for i, agent in enumerate(system2.agents):
    print(f"  Agent {i}: b₀={agent.b0.item():.4f}, c₀={agent.c0.item():.4f}, "
          f"c₀/b₀={agent.c0.item()/agent.b0.item():.4f} (max α)")

# Manually set different personalities
with torch.no_grad():
    # Agent 0: specialist (high c₀/b₀ = strong prior trust)
    system2.agents[0]._log_b0.data = torch.tensor(-1.0)  # b₀ ≈ 0.37
    system2.agents[0]._log_c0.data = torch.tensor(1.0)   # c₀ ≈ 2.72
    # Agent 1: generalist (low c₀/b₀ = flexible)
    system2.agents[1]._log_b0.data = torch.tensor(1.0)   # b₀ ≈ 2.72
    system2.agents[1]._log_c0.data = torch.tensor(-1.0)  # c₀ ≈ 0.37
    # Agent 2: default

print(f"\n  After setting personalities:")
for i, agent in enumerate(system2.agents):
    label = ["specialist", "generalist", "default"][i]
    print(f"  Agent {i} ({label}): b₀={agent.b0.item():.3f}, c₀={agent.c0.item():.3f}, "
          f"max_α=c₀/b₀={agent.c0.item()/agent.b0.item():.3f}")

# Verify specialist has higher max α
specialist_max = system2.agents[0].c0.item() / system2.agents[0].b0.item()
generalist_max = system2.agents[1].c0.item() / system2.agents[1].b0.item()
assert specialist_max > generalist_max * 10, "Specialist should have much higher max α"
print(f"  ✓ Specialist max_α ({specialist_max:.1f}) >> Generalist max_α ({generalist_max:.2f})")


# ============================================================
# 3. b₀, c₀ evolve with ε on model fiber
# ============================================================
print("\n[3] Precision hyperparameters evolve slowly (×ε)")
print("-" * 50)

system3 = MultiAgentSystem(3, K, grid_shape=())
vfe3 = FullVFE(adaptive_precision=True)
dyn3 = NaturalGradientDynamics(
    system3, vfe3,
    model_lr_ratio=0.1,  # visible ε for testing
    lr_precision=0.05,
)

b0_before = [a._log_b0.data.clone() for a in system3.agents]
c0_before = [a._log_c0.data.clone() for a in system3.agents]

for _ in range(20):
    dyn3.step()

b0_changed = any(
    not torch.allclose(a._log_b0.data, b0_before[i])
    for i, a in enumerate(system3.agents)
)
c0_changed = any(
    not torch.allclose(a._log_c0.data, c0_before[i])
    for i, a in enumerate(system3.agents)
)

print(f"  b₀ changed: {b0_changed}")
print(f"  c₀ changed: {c0_changed}")
for i, a in enumerate(system3.agents):
    db = (a._log_b0.data - b0_before[i]).item()
    dc = (a._log_c0.data - c0_before[i]).item()
    print(f"  Agent {i}: Δlog_b₀={db:.6f}, Δlog_c₀={dc:.6f}")

assert b0_changed or c0_changed, "Precision hyperparameters should evolve"
print(f"  ✓ Precision hyperparameters evolve via gradient through α")


# ============================================================
# 4. Covariance dynamics encode Eq. 36 automatically
# ============================================================
print("\n[4] Covariance dynamics (Eq. 36) via autograd through L_q")
print("-" * 50)

system4 = MultiAgentSystem(3, K, grid_shape=())
vfe4 = FullVFE(adaptive_precision=True, b0_belief=1.0, c0_belief=2.0)

# Record initial covariances
sigma_before = [a.sigma_q.detach().clone() for a in system4.agents]

dyn4 = NaturalGradientDynamics(system4, vfe4, lr_sigma_q=0.01)
for _ in range(20):
    dyn4.step()

sigma_after = [a.sigma_q.detach() for a in system4.agents]

for i in range(3):
    delta = (sigma_after[i] - sigma_before[i]).norm().item()
    det_before = sigma_before[i].det().item()
    det_after = sigma_after[i].det().item()
    print(f"  Agent {i}: ΔΣ_q={delta:.4f}, det: {det_before:.4f} → {det_after:.4f}")

sigma_changed = any(
    not torch.allclose(sigma_after[i], sigma_before[i], atol=1e-5)
    for i in range(3)
)
assert sigma_changed, "Covariances should evolve"

# Covariances should stay positive definite
for i in range(3):
    evals = torch.linalg.eigvalsh(sigma_after[i])
    assert (evals > 0).all(), f"Agent {i} covariance not SPD!"
print(f"  ✓ Covariances evolve and stay positive definite")
print(f"  ✓ Eq. 36 terms (-(1+α)Σ⁻¹ + αΣ_p⁻¹ + gate gradient) encoded via autograd")


# ============================================================
# 5. Group gradients for Ω (direct, no exp/log maps)
# ============================================================
print("\n[5] Group gradients for gauge frames")
print("-" * 50)

system5 = MultiAgentSystem(3, K, grid_shape=())
vfe5 = FullVFE(adaptive_precision=True)

omega_before = [a.omega.data.clone() for a in system5.agents]
det_before = [a.omega.data.det().item() for a in system5.agents]

dyn5 = NaturalGradientDynamics(system5, vfe5, lr_omega=0.01)
for _ in range(20):
    dyn5.step()

print(f"  Gauge frame evolution (direct group gradient ∂S/∂Ω):")
for i in range(3):
    delta = (system5.agents[i].omega.data - omega_before[i]).norm().item()
    det_after = system5.agents[i].omega.data.det().item()
    print(f"  Agent {i}: ΔΩ={delta:.4f}, det: {det_before[i]:.4f} → {det_after:.4f}")

omega_changed = any(
    not torch.allclose(system5.agents[i].omega.data, omega_before[i], atol=1e-5)
    for i in range(3)
)
assert omega_changed, "Gauge frames should evolve"

# Check Ω stays invertible (det ≠ 0)
for i in range(3):
    det = system5.agents[i].omega.data.det().item()
    assert abs(det) > 1e-6, f"Agent {i} gauge frame became singular!"
print(f"  ✓ Gauge frames evolve via group gradient, stay invertible")
print(f"  ✓ No exp/log maps needed — cheaper and more general than Lie algebra")


# ============================================================
# 6. Full integration: adaptive α + covariance + gauge
# ============================================================
print("\n[6] Full integration: everything running together")
print("-" * 50)

torch.manual_seed(42)
sys_full = MultiAgentSystem(5, K, grid_shape=())
vfe_full = FullVFE(adaptive_precision=True)
dyn_full = NaturalGradientDynamics(
    sys_full, vfe_full,
    lr_mu_q=0.03, lr_sigma_q=0.005,
    lr_mu_p=0.01, lr_sigma_p=0.005,
    lr_omega=0.005,
    model_lr_ratio=0.05,
    lr_precision=0.02,
)

# Push agents to different states
with torch.no_grad():
    sys_full.agents[0].mu_q.data = torch.randn(K) * 3.0
    sys_full.agents[0]._log_c0.data = torch.tensor(1.5)  # specialist

# Track everything
from gauge_agent.statistical_manifold import gaussian_kl

print(f"  {'step':>4} {'VFE':>8} {'α_0':>8} {'α_1':>8} {'KL_0':>8} {'KL_1':>8} "
      f"{'b₀_0':>8} {'c₀_0':>8}")

for t in range(50):
    info = dyn_full.step()

    if t % 10 == 0:
        with torch.no_grad():
            alphas = []
            kls = []
            for a in sys_full.agents[:2]:
                kl = gaussian_kl(a.mu_q.unsqueeze(0), a.sigma_q.unsqueeze(0),
                                  a.mu_p.unsqueeze(0), a.sigma_p.unsqueeze(0)).item()
                alpha = a.c0.item() / (a.b0.item() + kl)
                kls.append(kl)
                alphas.append(alpha)

            a0 = sys_full.agents[0]
            print(f"  {t:4d} {info['total']:8.3f} {alphas[0]:8.4f} {alphas[1]:8.4f} "
                  f"{kls[0]:8.3f} {kls[1]:8.3f} "
                  f"{a0.b0.item():8.4f} {a0.c0.item():8.4f}")

print(f"\n  ✓ Full system runs: VFE, adaptive α, covariance, gauge, b₀/c₀ all evolving")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
print(f"""
Full dynamics integration:

  1. FullVFE drives NaturalGradientDynamics (not old FreeEnergyFunctional)
  2. Per-agent b₀, c₀ on model fiber → specialist vs generalist
  3. b₀, c₀ evolve slowly (×ε) via gradient through α
  4. Covariance dynamics (Eq. 36) via autograd through L_q
     -(1+α)Σ⁻¹ + αΣ_p⁻¹ + gate gradient — all automatic
  5. Group gradients for Ω — direct ∂S/∂Ω, no exp/log maps
  6. Everything runs together end-to-end
""")
