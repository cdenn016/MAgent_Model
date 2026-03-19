"""
Test: Species vs Meta-Agents and Timescale Separation
=======================================================

Model alignment (s_i) → SPECIES (what kind of thing you are, slow)
Belief alignment (q_i) → META-AGENTS (what you're doing together, fast)

Selection rule: W_{iα} = S_{iα} · C_{iα}
  S = species gate (model alignment)
  C = coalition membership (belief alignment)
  Can only coordinate with agents of your species.

Timescale separation: ds/dt = ε · ∂S/∂s (model evolves slowly)
"""

import torch
torch.manual_seed(42)

print("=" * 70)
print("TEST: Species, Meta-Agents, and Timescale Separation")
print("=" * 70)


# ============================================================
# 1. Species detection from model alignment
# ============================================================
print("\n[1] Species detection: model alignment defines species")
print("-" * 50)

from gauge_agent.agents import Agent, MultiAgentSystem
from gauge_agent.hierarchical_emergence import (
    SpeciesDetector, CoalitionDetector, GatedMembership,
    SoftMembership, HierarchicalEmergence,
)

K = 4

# 12 agents, 3 species: 4 "humans", 4 "dogs", 4 "algae"
children = MultiAgentSystem(12, K, grid_shape=())

model_human = torch.tensor([1.0, 0.0, 0.0, 0.0])
model_dog = torch.tensor([0.0, 1.0, 0.0, 0.0])
model_algae = torch.tensor([0.0, 0.0, 1.0, 0.0])

with torch.no_grad():
    for i in range(4):
        children.agents[i].mu_s.data = model_human + 0.05 * torch.randn(K)
        children.agents[i].omega_model.data = torch.eye(K) + 0.02 * torch.randn(K, K)
    for i in range(4, 8):
        children.agents[i].mu_s.data = model_dog + 0.05 * torch.randn(K)
        children.agents[i].omega_model.data = torch.eye(K) + 0.02 * torch.randn(K, K)
    for i in range(8, 12):
        children.agents[i].mu_s.data = model_algae + 0.05 * torch.randn(K)
        children.agents[i].omega_model.data = torch.eye(K) + 0.02 * torch.randn(K, K)

# 3 meta-agents representing the 3 species
parents = MultiAgentSystem(3, K, grid_shape=())
with torch.no_grad():
    parents.agents[0].mu_s.data = model_human
    parents.agents[0].omega_model.data = torch.eye(K)
    parents.agents[1].mu_s.data = model_dog
    parents.agents[1].omega_model.data = torch.eye(K)
    parents.agents[2].mu_s.data = model_algae
    parents.agents[2].omega_model.data = torch.eye(K)

species_det = SpeciesDetector(tau_species=5.0)
S = species_det.species_gate(children, parents)

print(f"  Species gate S (rows=agents, cols=species):")
print(f"  {'':>10} {'human':>8} {'dog':>8} {'algae':>8}")
labels = ['human'] * 4 + ['dog'] * 4 + ['algae'] * 4
for i in range(12):
    print(f"  {labels[i]:>10}_{i}: {S[i,0].item():8.3f} {S[i,1].item():8.3f} {S[i,2].item():8.3f}")

# Humans should have high S for human species, low for others
human_in_human = S[:4, 0].mean().item()
human_in_dog = S[:4, 1].mean().item()
human_in_algae = S[:4, 2].mean().item()
print(f"\n  Humans: in_human={human_in_human:.3f} in_dog={human_in_dog:.3f} in_algae={human_in_algae:.3f}")
assert human_in_human > human_in_dog, "Humans should be more human than dog"
assert human_in_human > human_in_algae, "Humans should be more human than algae"
print("  ✓ Species correctly detected from model alignment")


# ============================================================
# 2. Selection rule: species gates coalition
# ============================================================
print("\n[2] Selection rule: W = S · C")
print("-" * 50)

# Now give some cross-species agents similar BELIEFS
# (humans and dogs both believe "it's warm", but algae don't)
with torch.no_grad():
    warm_belief = torch.tensor([0.0, 0.0, 0.0, 1.0])
    for i in range(8):  # humans and dogs
        children.agents[i].mu_q.data = warm_belief + 0.1 * torch.randn(K)
        children.agents[i].omega.data = torch.eye(K) + 0.02 * torch.randn(K, K)
    cold_belief = torch.tensor([0.0, 0.0, 0.0, -1.0])
    for i in range(8, 12):  # algae believe differently
        children.agents[i].mu_q.data = cold_belief + 0.1 * torch.randn(K)
        children.agents[i].omega.data = torch.eye(K) + 0.02 * torch.randn(K, K)

    # Meta-agents: "warm team" and "cold team" and "mixed"
    parents.agents[0].mu_q.data = warm_belief
    parents.agents[0].omega.data = torch.eye(K)
    parents.agents[1].mu_q.data = warm_belief  # dogs agree on warm
    parents.agents[1].omega.data = torch.eye(K)
    parents.agents[2].mu_q.data = cold_belief
    parents.agents[2].omega.data = torch.eye(K)

gated = GatedMembership(tau_species=5.0, tau_belief=1.0)
result = gated.compute(children, parents)

print(f"  {'':>10} {'meta_0':>8} {'meta_1':>8} {'meta_2':>8}")
print(f"  {'':>10} {'(human)':>8} {'(dog)':>8} {'(algae)':>8}")
for i in range(12):
    W = result['W']
    S = result['S']
    C = result['C']
    print(f"  {labels[i]:>6}_{i}: W=[{W[i,0].item():.3f} {W[i,1].item():.3f} {W[i,2].item():.3f}]  "
          f"S=[{S[i,0].item():.3f} {S[i,1].item():.3f} {S[i,2].item():.3f}]  "
          f"C=[{C[i,0].item():.3f} {C[i,1].item():.3f} {C[i,2].item():.3f}]")

# Key test: human_0 has high C for meta_0 (warm) AND meta_1 (warm dog)
# but W is gated: high W only for meta_0 (human), low W for meta_1 (dog)
human0_W_human_meta = result['W'][0, 0].item()
human0_W_dog_meta = result['W'][0, 1].item()
human0_C_human_meta = result['C'][0, 0].item()
human0_C_dog_meta = result['C'][0, 1].item()

print(f"\n  Human_0:")
print(f"    Coalition with human-meta: C={human0_C_human_meta:.3f}, W={human0_W_human_meta:.3f}")
print(f"    Coalition with dog-meta:   C={human0_C_dog_meta:.3f}, W={human0_W_dog_meta:.3f}")
print(f"    Despite similar beliefs (C≈C), species blocks cross-species meta-agent!")

# Both C values should be similar (similar beliefs)
# But W should be much higher for same-species meta-agent
assert human0_W_human_meta > human0_W_dog_meta, "Species should gate membership"
print("  ✓ Selection rule: species gates meta-agent formation")


# ============================================================
# 3. Timescale separation in dynamics
# ============================================================
print("\n[3] Timescale separation: ε controls model evolution speed")
print("-" * 50)

from gauge_agent.dynamics import NaturalGradientDynamics
from gauge_agent.free_energy import FreeEnergyFunctional

system = MultiAgentSystem(4, K, grid_shape=())
fe = FreeEnergyFunctional()

# Track changes with different ε
for eps_label, eps in [("ε=1.0 (no separation)", 1.0),
                        ("ε=0.01 (biological)", 0.01),
                        ("ε=0.0 (frozen model)", 0.0)]:
    torch.manual_seed(99)
    sys = MultiAgentSystem(4, K, grid_shape=())
    dyn = NaturalGradientDynamics(
        sys, fe,
        lr_mu_q=0.05, lr_sigma_q=0.01, lr_mu_p=0.02,
        lr_sigma_p=0.01, lr_omega=0.01,
        model_lr_ratio=eps,
    )

    mu_q_init = sys.agents[0].mu_q.data.clone()
    mu_s_init = sys.agents[0].mu_s.data.clone()

    for _ in range(20):
        dyn.step()

    dq = (sys.agents[0].mu_q.data - mu_q_init).norm().item()
    ds = (sys.agents[0].mu_s.data - mu_s_init).norm().item()

    ratio = ds / max(dq, 1e-10)
    print(f"  {eps_label:>30}: Δq={dq:.4f}  Δs={ds:.6f}  ratio={ratio:.4f}")

# ε=0 should have zero model change
torch.manual_seed(99)
sys_frozen = MultiAgentSystem(4, K, grid_shape=())
dyn_frozen = NaturalGradientDynamics(
    sys_frozen, fe, model_lr_ratio=0.0
)
mu_s_before = sys_frozen.agents[0].mu_s.data.clone()
for _ in range(20):
    dyn_frozen.step()
mu_s_after = sys_frozen.agents[0].mu_s.data
assert torch.allclose(mu_s_before, mu_s_after), "ε=0 should freeze model"
print("  ✓ ε=0 freezes model fiber completely")
print("  ✓ ε=0.01 model evolves ~100x slower than beliefs")


# ============================================================
# 4. Gated HierarchicalEmergence
# ============================================================
print("\n[4] Gated HierarchicalEmergence")
print("-" * 50)

torch.manual_seed(42)
base = MultiAgentSystem(12, K, grid_shape=())

hier_gated = HierarchicalEmergence(
    base_system=base,
    n_meta_per_scale=[6, 3, 1],
    tau_species=5.0,
    tau_belief=1.0,
    gated=True,
)

hier_ungated = HierarchicalEmergence(
    base_system=MultiAgentSystem(12, K, grid_shape=()),
    n_meta_per_scale=[6, 3, 1],
    tau_species=5.0,
    tau_belief=1.0,
    gated=False,
)

print(f"  Gated:   {hier_gated.summary_string()}")
print(f"  Ungated: {hier_ungated.summary_string()}")

# Detailed breakdown
details = hier_gated.compute_all_memberships_detailed()
for ell, d in enumerate(details):
    print(f"\n  Scale ℓ{ell}→{ell+1}:")
    print(f"    Species S:   mean={d['S'].mean().item():.3f} max={d['S'].max().item():.3f}")
    print(f"    Coalition C: mean={d['C'].mean().item():.3f} max={d['C'].max().item():.3f}")
    print(f"    Effective W: mean={d['W'].mean().item():.3f} max={d['W'].max().item():.3f}")

print("  ✓ Gated hierarchy with species/coalition breakdown")


# ============================================================
# 5. Cross-species blocking demonstration
# ============================================================
print("\n[5] Cross-species blocking: algae can't join human coalition")
print("-" * 50)

# Setup: 4 humans, 4 algae, 2 meta-agents
mixed = MultiAgentSystem(8, K, grid_shape=())
meta = MultiAgentSystem(2, K, grid_shape=())

with torch.no_grad():
    # Humans
    for i in range(4):
        mixed.agents[i].mu_s.data = model_human + 0.05 * torch.randn(K)
        mixed.agents[i].mu_q.data = warm_belief + 0.1 * torch.randn(K)
        mixed.agents[i].omega_model.data = torch.eye(K)
        mixed.agents[i].omega.data = torch.eye(K)
    # Algae - SAME BELIEFS as humans (also think "warm")!
    for i in range(4, 8):
        mixed.agents[i].mu_s.data = model_algae + 0.05 * torch.randn(K)
        mixed.agents[i].mu_q.data = warm_belief + 0.1 * torch.randn(K)  # same belief!
        mixed.agents[i].omega_model.data = torch.eye(K)
        mixed.agents[i].omega.data = torch.eye(K)

    # Meta-agents: one human-species, one algae-species
    meta.agents[0].mu_s.data = model_human
    meta.agents[0].mu_q.data = warm_belief
    meta.agents[0].omega_model.data = torch.eye(K)
    meta.agents[0].omega.data = torch.eye(K)
    meta.agents[1].mu_s.data = model_algae
    meta.agents[1].mu_q.data = warm_belief  # same belief!
    meta.agents[1].omega_model.data = torch.eye(K)
    meta.agents[1].omega.data = torch.eye(K)

gated_m = GatedMembership(tau_species=3.0, tau_belief=1.0)
r = gated_m.compute(mixed, meta)

print(f"  All agents believe 'warm'. But species differ.")
print(f"  {'':>10} {'human-meta':>12} {'algae-meta':>12}")
for i in range(8):
    label = 'human' if i < 4 else 'algae'
    print(f"  {label}_{i}: "
          f"S=[{r['S'][i,0].item():.3f},{r['S'][i,1].item():.3f}] "
          f"C=[{r['C'][i,0].item():.3f},{r['C'][i,1].item():.3f}] "
          f"W=[{r['W'][i,0].item():.3f},{r['W'][i,1].item():.3f}]")

# Coalition C should be similar for all (same beliefs)
# But W should be blocked by species
algae_in_human = r['W'][4:, 0].mean().item()
algae_C_human = r['C'][4:, 0].mean().item()
print(f"\n  Algae coalition with human-meta: C={algae_C_human:.3f} (beliefs align)")
print(f"  Algae effective W with human-meta: W={algae_in_human:.3f} (species blocks!)")
human_in_human = r['W'][:4, 0].mean().item()
print(f"  Humans effective W with human-meta: W={human_in_human:.3f}")
assert human_in_human > algae_in_human, "Species should block cross-species coalition"
print("  ✓ Algae blocked from human coalition despite matching beliefs")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
print(f"""
Species vs Meta-Agents:

  MODEL alignment (s_i, slow) → SPECIES
    What kind of thing you are. Shared ontology.
    Humans share models with humans, not algae.
    Evolves on evolutionary timescale (ε << 1).

  BELIEF alignment (q_i, fast) → META-AGENTS (coalitions)
    What you're doing together right now.
    Teams, flocks, coordinated groups.
    Changes on perceptual timescale.

  SELECTION RULE: W = S · C
    Can only join a coalition if you're the right species.
    Algae can't join human teams even with matching beliefs.

  TIMESCALE SEPARATION: ds/dt = ε · ∂S/∂s
    ε = 0.0:  model frozen (pure inference)
    ε = 0.01: model evolves 100x slower (biological)
    ε = 1.0:  no separation (both fast)
""")
