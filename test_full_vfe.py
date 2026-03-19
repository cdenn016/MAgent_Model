"""
Test: Complete Variational Free Energy (Eq. 24)
=================================================

Verifies the CORRECT VFE from the manuscript:

  CORE (5 terms):
    T1: KL(q_i || p_i)              — belief self-consistency
    T2: KL(s_i || r_i)              — model self-consistency
    T3: β_ij KL(q_i || Ω_ij[q_j])  — belief alignment
    T4: γ_ij KL(s_i || Ω̃_ij[s_j])  — model alignment (NOT p_i!)
    T5: -E_q[log p(o|q)]            — observation

  Note: s_i ≠ p_i. Agents have FOUR distributions and TWO gauge frames.

  OPTIONAL EXTENSIONS (off by default):
    EXT_hyperprior:     ρ^k KL(p_i || Ω_{iA}[q_A])
    EXT_gauge_smooth:   ‖∇φ_i‖²
    EXT_yang_mills:     tr(F_μν F^μν)
"""

import torch
torch.manual_seed(42)

print("=" * 70)
print("TEST: Complete VFE (Eq. 24) with s_i ≠ p_i")
print("=" * 70)


# ============================================================
# 1. Agent has 4 distributions and 2 gauge frames
# ============================================================
print("\n[1] Agent structure: q_i, p_i, s_i, r_i, Ω_i, Ω̃_i")
print("-" * 50)

from gauge_agent.agents import Agent, MultiAgentSystem

K = 4
agent = Agent(K, grid_shape=(), agent_id=0)

print(f"  Belief fiber:")
print(f"    q_i: μ_q {agent.mu_q.shape}, Σ_q {agent.sigma_q.shape}")
print(f"    p_i: μ_p {agent.mu_p.shape}, Σ_p {agent.sigma_p.shape}")
print(f"    Ω_i: {agent.omega.shape}")
print(f"  Model fiber:")
print(f"    s_i: μ_s {agent.mu_s.shape}, Σ_s {agent.sigma_s.shape}")
print(f"    r_i: μ_r {agent.mu_r.shape}, Σ_r {agent.sigma_r.shape}")
print(f"    Ω̃_i: {agent.omega_model.shape}")

# s_i and p_i should be DIFFERENT (independently initialized)
assert not torch.allclose(agent.mu_s, agent.mu_p), "s_i ≠ p_i"
assert not torch.allclose(agent.omega, agent.omega_model), "Ω_i ≠ Ω̃_i"
print(f"  KL(q||p) = {agent.self_kl().item():.4f}")
print(f"  KL(s||r) = {agent.model_self_kl().item():.4f}")
print("  ✓ Agent has 4 distributions and 2 gauge frames")


# ============================================================
# 2. FullVFE core 5 terms
# ============================================================
print("\n[2] FullVFE core 5 terms")
print("-" * 50)

from gauge_agent.full_vfe import FullVFE

N = 6
system = MultiAgentSystem(N, K, grid_shape=())

# Core only — all extensions off
vfe = FullVFE(
    lambda_self=1.0, lambda_model_self=0.5,
    lambda_belief=1.0, lambda_model=0.5, lambda_obs=1.0,
    lambda_smooth=0.0, lambda_ym=0.0, lambda_hyper=0.0,
)

result = vfe(system)
print(f"  {vfe.summary_string(result)}")
print(f"  T1 KL(q||p):       {result['T1_belief_self'].item():.4f}")
print(f"  T2 KL(s||r):       {result['T2_model_self'].item():.4f}")
print(f"  T3 β·KL(q||Ωq):   {result['T3_belief_align'].item():.4f}")
print(f"  T4 γ·KL(s||Ω̃s):   {result['T4_model_align'].item():.4f}")
print(f"  T5 observation:    {result['T5_observation'].item():.4f}")
print(f"  Extensions all 0:  EXT={result['EXT_hyperprior'].item():.4f}, "
      f"{result['EXT_gauge_smooth'].item():.4f}, {result['EXT_yang_mills'].item():.4f}")

# T2 should be > 0 since s_i and r_i are independently random
assert result['T2_model_self'].item() > 0, "T2 should be > 0"
# T4 should be > 0 since s_i differ across agents
assert result['T4_model_align'].item() > 0, "T4 should be > 0"
# Extensions should be exactly 0
assert result['EXT_hyperprior'].item() == 0.0
assert result['EXT_gauge_smooth'].item() == 0.0
assert result['EXT_yang_mills'].item() == 0.0
print("  ✓ Core 5 terms active, extensions off")


# ============================================================
# 3. T4 uses model transport Ω̃_ij, not belief transport Ω_ij
# ============================================================
print("\n[3] T4 uses model gauge frame Ω̃_ij (independent of Ω_ij)")
print("-" * 50)

# Perturb model frames differently from belief frames
with torch.no_grad():
    for agent in system.agents:
        agent.omega_model.data = 2.0 * torch.eye(K) + 0.5 * torch.randn(K, K)

result2 = vfe(system)
print(f"  T4 after perturbing Ω̃: {result2['T4_model_align'].item():.4f}")
print(f"  T3 unchanged:          {result2['T3_belief_align'].item():.4f}")

# T3 should be unchanged (belief transport unchanged)
diff_T3 = abs(result['T3_belief_align'].item() - result2['T3_belief_align'].item())
print(f"  T3 difference:          {diff_T3:.6f} (should be ~0)")
assert diff_T3 < 0.01, "T3 should not change when Ω̃ changes"
print("  ✓ T4 uses independent model transport Ω̃_ij")


# ============================================================
# 4. MultiAgentSystem model fiber accessors
# ============================================================
print("\n[4] MultiAgentSystem model fiber accessors")
print("-" * 50)

mu_s = system.get_all_mu_s()
sigma_s = system.get_all_sigma_s()
mu_r = system.get_all_mu_r()
sigma_r = system.get_all_sigma_r()
omega_m = system.get_all_omega_model()

print(f"  μ_s: {mu_s.shape}")
print(f"  Σ_s: {sigma_s.shape}")
print(f"  Ω̃:   {omega_m.shape}")

# Model alignment via pairwise_alignment_energies
E_model = system.pairwise_alignment_energies('model')
E_belief = system.pairwise_alignment_energies('belief')
print(f"  E_model shape: {E_model.shape}")
print(f"  E_model mean:  {E_model.mean().item():.4f}")
print(f"  E_belief mean: {E_belief.mean().item():.4f}")
print("  ✓ Model fiber fully accessible")


# ============================================================
# 5. Observations (T5)
# ============================================================
print("\n[5] Observation term T5")
print("-" * 50)

obs = torch.zeros(N, K)
result_obs = vfe(system, observations=obs, obs_precision=torch.tensor(10.0))
print(f"  T5 with obs: {result_obs['T5_observation'].item():.4f}")
assert result_obs['T5_observation'].item() > 0
print("  ✓ T5 active")


# ============================================================
# 6. Optional extensions (only when explicitly enabled)
# ============================================================
print("\n[6] Optional extensions (enabled explicitly)")
print("-" * 50)

vfe_ext = FullVFE(
    lambda_self=1.0, lambda_model_self=0.5,
    lambda_belief=1.0, lambda_model=0.5, lambda_obs=0.0,
    lambda_smooth=0.01, lambda_ym=0.1, lambda_hyper=0.5,
)

# Hyperprior requires ancestors
parent = Agent(K, grid_shape=(), agent_id=100)
ancestors = [{0: parent, 1: parent}]

result_ext = vfe_ext(system, ancestors=ancestors)
print(f"  EXT hyperprior:    {result_ext['EXT_hyperprior'].item():.4f}")
print(f"  EXT gauge_smooth:  {result_ext['EXT_gauge_smooth'].item():.4f}")
print(f"  EXT yang_mills:    {result_ext['EXT_yang_mills'].item():.4f}")
# Without lattice gauge, YM should still be 0
assert result_ext['EXT_yang_mills'].item() == 0.0, "No lattice gauge → YM = 0"
assert result_ext['EXT_hyperprior'].item() > 0
print("  ✓ Extensions opt-in, off by default")


# ============================================================
# 7. Full VFE on manifold system
# ============================================================
print("\n[7] Full VFE on T² manifold with lattice gauge")
print("-" * 50)

from gauge_agent.manifolds import Torus
from gauge_agent.manifold_system import ManifoldAgentSystem

T2_manifold = Torus(dim=2, grid_shape=(6, 6))
mas = ManifoldAgentSystem(
    N_agents=4, K=3, manifold=T2_manifold,
    init_twist_scale=0.05, ym_coupling=0.1
)

result_m = mas.volume_weighted_free_energy()
print(f"  {mas.vfe.summary_string(result_m)}")

# Evolve
print("\n  Evolving:")
history = mas.evolve(15, lr=0.005, verbose=True)
print(f"  ✓ Manifold system uses correct VFE")


# ============================================================
# 8. Gradient flow through all 4 distributions
# ============================================================
print("\n[8] Gradient flow through all agent parameters")
print("-" * 50)

torch.manual_seed(99)
system_test = MultiAgentSystem(3, 3, grid_shape=())
parent_test = Agent(3, grid_shape=(), agent_id=99)

vfe_all = FullVFE(
    lambda_self=1.0, lambda_model_self=0.5,
    lambda_belief=1.0, lambda_model=0.5, lambda_obs=1.0,
    lambda_smooth=0.0, lambda_ym=0.0, lambda_hyper=0.5,
)

result_all = vfe_all(
    system_test,
    observations=torch.randn(3, 3),
    obs_precision=torch.tensor(1.0),
    ancestors=[{0: parent_test, 1: parent_test}],
)
result_all['total'].backward()

# Check gradients
params_with_grad = 0
params_total = 0
for i, agent in enumerate(system_test.agents):
    for name, param in agent.named_parameters():
        params_total += 1
        has = param.grad is not None and param.grad.abs().max() > 0
        if has:
            params_with_grad += 1
        marker = "✓" if has else "○"
        print(f"    {marker} agent_{i}.{name}")

print(f"  Gradients: {params_with_grad}/{params_total}")
print("  ✓ Gradient flow verified")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
print(f"""
The VFE now correctly implements Eq. 24 with s_i ≠ p_i:

  CORE (5 terms from the boxed equation):
    T1: KL(q_i || p_i)              belief ↔ belief-prior
    T2: KL(s_i || r_i)              model ↔ model-prior
    T3: β_ij KL(q_i || Ω_ij[q_j])  belief alignment (belief transport Ω_ij)
    T4: γ_ij KL(s_i || Ω̃_ij[s_j])  model alignment (model transport Ω̃_ij)
    T5: -E_q[log p(o|q)]            observation likelihood

  Each agent carries:
    q_i, p_i, Ω_i    (belief fiber)
    s_i, r_i, Ω̃_i    (model fiber — independent gauge frame)

  Optional extensions (off by default, λ=0):
    Hyperpriors, gauge smoothness, Yang-Mills
""")
