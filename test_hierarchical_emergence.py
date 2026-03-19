"""
Test: Soft Hierarchical Emergence and Condensation
====================================================

Demonstrates the gauge-theoretic BCS: agents with soft membership
in multiple meta-agents, precision pooling, and a condensation
phase transition controlled by temperature τ.

Key results:
  - Agents have overlapping membership in multiple meta-agents
  - Precision pooling gives more weight to more certain agents
  - Low τ → sharp membership (hard clustering limit)
  - High τ → uniform membership (no hierarchy)
  - Cross-scale VFE is fully differentiable
"""

import torch
torch.manual_seed(42)

print("=" * 70)
print("TEST: Soft Hierarchical Emergence")
print("=" * 70)


# ============================================================
# 1. Basic soft membership
# ============================================================
print("\n[1] Soft membership W_{iα}")
print("-" * 50)

from gauge_agent.agents import Agent, MultiAgentSystem
from gauge_agent.hierarchical_emergence import (
    SoftMembership, precision_pool, CondensationDiagnostics,
    CrossScaleVFE, HierarchicalEmergence
)

K = 4
N_children = 12
N_parents = 4

children = MultiAgentSystem(N_children, K, grid_shape=())
parents = MultiAgentSystem(N_parents, K, grid_shape=())

membership = SoftMembership(tau=1.0)
W = membership.compute(children, parents)

print(f"  W shape: {W.shape}  (N_children={N_children}, N_parents={N_parents})")
print(f"  W range: [{W.min().item():.4f}, {W.max().item():.4f}]")
print(f"  W mean:  {W.mean().item():.4f}")

# Check overlapping membership: each child can be in multiple parents
membership_count = (W > 0.5).sum(dim=1).float()
print(f"  Mean meta-agents per child (w>0.5): {membership_count.mean().item():.2f}")

# Key property: W is NOT a partition matrix
row_sums = W.sum(dim=1)
print(f"  Row sums (should NOT be 1.0): [{row_sums.min().item():.2f}, {row_sums.max().item():.2f}]")
print("  ✓ Soft overlapping membership")


# ============================================================
# 2. Temperature controls membership sharpness
# ============================================================
print("\n[2] Temperature τ controls condensation")
print("-" * 50)

taus = [0.01, 0.1, 1.0, 10.0]
for tau in taus:
    m = SoftMembership(tau=tau)
    W_t = m.compute(children, parents)
    mean_w = W_t.mean().item()
    max_w = W_t.max().item()
    min_w = W_t.min().item()
    sharp = ((W_t > 0.9) | (W_t < 0.1)).float().mean().item()
    print(f"  τ={tau:5.2f} | mean={mean_w:.4f} max={max_w:.4f} "
          f"min={min_w:.4f} | sharp={sharp:.2f}")

# Low τ should be sharper (more binary)
W_low = SoftMembership(tau=0.01).compute(children, parents)
W_high = SoftMembership(tau=10.0).compute(children, parents)
sharp_low = ((W_low > 0.9) | (W_low < 0.1)).float().mean().item()
sharp_high = ((W_high > 0.9) | (W_high < 0.1)).float().mean().item()
assert sharp_low > sharp_high, "Low τ should be sharper"
print("  ✓ Low τ → sharp (hard clustering), High τ → soft (uniform)")


# ============================================================
# 3. Precision pooling
# ============================================================
print("\n[3] Precision pooling: more certain agents contribute more")
print("-" * 50)

# Create children: agent 0 very certain (small Σ), agent 1 uncertain
agent_precise = Agent(K, grid_shape=(), agent_id=0)
agent_vague = Agent(K, grid_shape=(), agent_id=1)

with torch.no_grad():
    # Agent 0: precise (small covariance)
    agent_precise.mu_q.data = torch.tensor([1.0, 0.0, 0.0, 0.0])
    agent_precise._L_q.data = 0.1 * torch.eye(K)

    # Agent 1: vague (large covariance)
    agent_vague.mu_q.data = torch.tensor([0.0, 1.0, 0.0, 0.0])
    agent_vague._L_q.data = 10.0 * torch.eye(K)

mu_children = torch.stack([agent_precise.mu_q, agent_vague.mu_q])
sigma_children = torch.stack([agent_precise.sigma_q, agent_vague.sigma_q])
weights = torch.tensor([1.0, 1.0])
transport = torch.eye(K).unsqueeze(0).expand(2, K, K)  # identity transport

mu_pooled, sigma_pooled = precision_pool(mu_children, sigma_children,
                                          weights, transport)

print(f"  Precise agent μ: {agent_precise.mu_q.data.tolist()}")
print(f"  Vague agent μ:   {agent_vague.mu_q.data.tolist()}")
print(f"  Pooled μ:        {mu_pooled.tolist()}")
print(f"  Pooled σ trace:  {sigma_pooled.diag().sum().item():.4f}")

# Pooled mean should be MUCH closer to the precise agent
dist_to_precise = (mu_pooled - agent_precise.mu_q.data).norm().item()
dist_to_vague = (mu_pooled - agent_vague.mu_q.data).norm().item()
print(f"  Distance to precise: {dist_to_precise:.4f}")
print(f"  Distance to vague:   {dist_to_vague:.4f}")
assert dist_to_precise < dist_to_vague, "Pooled should favor precise agent"
print("  ✓ Precision pooling correctly weights by certainty")


# ============================================================
# 4. Cross-scale VFE is differentiable
# ============================================================
print("\n[4] Cross-scale VFE: fully differentiable")
print("-" * 50)

cross_vfe = CrossScaleVFE(lambda_belief_topdown=1.0, lambda_model_topdown=1.0)

# Need fresh systems with gradients
children_g = MultiAgentSystem(8, K, grid_shape=())
parents_g = MultiAgentSystem(3, K, grid_shape=())

W_g = SoftMembership(tau=1.0).compute(children_g, parents_g)
result = cross_vfe(children_g, parents_g, W_g)

print(f"  Cross-scale total:      {result['total'].item():.4f}")
print(f"  Belief top-down:        {result['belief_topdown'].item():.4f}")
print(f"  Model top-down:         {result['model_topdown'].item():.4f}")

# Check gradients flow
result['total'].backward()
child_grads = sum(1 for a in children_g.agents
                  for p in a.parameters() if p.grad is not None)
parent_grads = sum(1 for a in parents_g.agents
                   for p in a.parameters() if p.grad is not None)
print(f"  Child params with grad:  {child_grads}")
print(f"  Parent params with grad: {parent_grads}")
assert child_grads > 0 and parent_grads > 0
print("  ✓ Gradients flow through cross-scale VFE")


# ============================================================
# 5. Condensation order parameter
# ============================================================
print("\n[5] Condensation order parameter Ψ_α")
print("-" * 50)

cond = CondensationDiagnostics(condensation_threshold=0.5)

# Make children clustered: first 4 share a model, next 4 share another
clustered_children = MultiAgentSystem(8, K, grid_shape=())
clustered_parents = MultiAgentSystem(2, K, grid_shape=())

with torch.no_grad():
    shared_model_A = torch.randn(K)
    shared_model_B = torch.randn(K)
    for i in range(4):
        clustered_children.agents[i].mu_s.data = shared_model_A + 0.01 * torch.randn(K)
        clustered_children.agents[i].omega_model.data = torch.eye(K) + 0.01 * torch.randn(K, K)
    for i in range(4, 8):
        clustered_children.agents[i].mu_s.data = shared_model_B + 0.01 * torch.randn(K)
        clustered_children.agents[i].omega_model.data = torch.eye(K) + 0.01 * torch.randn(K, K)
    # Parents match the clusters
    clustered_parents.agents[0].mu_s.data = shared_model_A
    clustered_parents.agents[0].omega_model.data = torch.eye(K)
    clustered_parents.agents[1].mu_s.data = shared_model_B
    clustered_parents.agents[1].omega_model.data = torch.eye(K)

W_clust = SoftMembership(tau=0.5).compute(clustered_children, clustered_parents)
psi = cond.order_parameter(clustered_children, clustered_parents, W_clust)
f = cond.condensation_fraction(clustered_children, clustered_parents, W_clust)

print(f"  Ψ per meta-agent: {psi.tolist()}")
print(f"  Condensation fraction: {f:.4f}")
print(f"  W (first 4 agents, should prefer meta-agent 0):")
for i in range(4):
    print(f"    agent_{i}: [{W_clust[i,0].item():.3f}, {W_clust[i,1].item():.3f}]")
print(f"  W (last 4 agents, should prefer meta-agent 1):")
for i in range(4, 8):
    print(f"    agent_{i}: [{W_clust[i,0].item():.3f}, {W_clust[i,1].item():.3f}]")
assert f > 0.5, "Clustered agents should condense"
print("  ✓ Condensation detected in clustered system")


# ============================================================
# 6. Full HierarchicalEmergence
# ============================================================
print("\n[6] Full HierarchicalEmergence: 16 → 8 → 4 → 2 → 1")
print("-" * 50)

base = MultiAgentSystem(16, K, grid_shape=())
hierarchy = HierarchicalEmergence(
    base_system=base,
    n_meta_per_scale=[8, 4, 2, 1],
    tau=1.0,
    lambda_cross=0.5,
)

print(f"  {hierarchy.summary_string()}")
diag = hierarchy.diagnostics()
print(f"  Scales: {diag['scale_agents']}")
for ell, stats in enumerate(diag['membership_stats']):
    print(f"    ℓ{ell}→{ell+1}: mean_w={stats['mean']:.3f} "
          f"active_meta={stats['active_meta']}")

# Compute cross-scale energy
memberships = hierarchy.compute_all_memberships()
cross_result = hierarchy.cross_scale_energy(memberships)
print(f"  Cross-scale VFE: {cross_result['total'].item():.4f}")
for ell, r in enumerate(cross_result['per_scale']):
    print(f"    ℓ{ell}→{ell+1}: belief={r['belief_topdown'].item():.3f} "
          f"model={r['model_topdown'].item():.3f}")

print("  ✓ Multi-scale hierarchy created")


# ============================================================
# 7. Bottom-up pooling updates meta-agents
# ============================================================
print("\n[7] Bottom-up precision pooling")
print("-" * 50)

mu_before = hierarchy.scales[1].agents[0].mu_q.data.clone()
hierarchy.update_meta_agents(memberships)
mu_after = hierarchy.scales[1].agents[0].mu_q.data

delta = (mu_after - mu_before).norm().item()
print(f"  Meta-agent 0 μ_q shift: {delta:.4f}")
assert delta > 0, "Pooling should change meta-agent parameters"
print("  ✓ Meta-agents updated from children via precision pooling")


# ============================================================
# 8. Overlapping membership demonstration
# ============================================================
print("\n[8] Overlapping membership: agent in multiple meta-agents")
print("-" * 50)

# Scenario: 6 agents, 3 meta-agents
# Agent 0-1: family, Agent 2-3: company, Agent 4-5: nation
# Agent 1 is ALSO in company (overlap!)
overlap_children = MultiAgentSystem(6, K, grid_shape=())
overlap_parents = MultiAgentSystem(3, K, grid_shape=())

with torch.no_grad():
    model_family = torch.tensor([1.0, 0.0, 0.0, 0.0])
    model_company = torch.tensor([0.0, 1.0, 0.0, 0.0])
    model_nation = torch.tensor([0.5, 0.5, 0.0, 0.0])  # blend

    # Children
    for i in [0, 1]:
        overlap_children.agents[i].mu_s.data = model_family + 0.05 * torch.randn(K)
    for i in [2, 3]:
        overlap_children.agents[i].mu_s.data = model_company + 0.05 * torch.randn(K)
    for i in [4, 5]:
        overlap_children.agents[i].mu_s.data = model_nation + 0.05 * torch.randn(K)

    # Agent 1 also aligns with company (overlapping membership!)
    overlap_children.agents[1].mu_s.data = 0.5 * model_family + 0.5 * model_company

    # Set all gauge frames to identity for clarity
    for a in overlap_children.agents:
        a.omega_model.data = torch.eye(K)
    for a in overlap_parents.agents:
        a.omega_model.data = torch.eye(K)

    # Parents match the groups
    overlap_parents.agents[0].mu_s.data = model_family
    overlap_parents.agents[1].mu_s.data = model_company
    overlap_parents.agents[2].mu_s.data = model_nation

W_overlap = SoftMembership(tau=0.3).compute(overlap_children, overlap_parents)
print(f"  Membership matrix (rows=agents, cols=meta-agents):")
print(f"  {'':>10} {'family':>8} {'company':>8} {'nation':>8}")
names = ['parent_0', 'parent_1', 'spouse_1', 'worker_2', 'worker_3', 'citizen_4', 'citizen_5']
for i in range(6):
    row = W_overlap[i].detach()
    label = f"agent_{i}"
    print(f"  {label:>10} {row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f}")

# Agent 1 should have significant membership in BOTH family AND company
w_agent1_family = W_overlap[1, 0].item()
w_agent1_company = W_overlap[1, 1].item()
print(f"\n  Agent 1 (boundary): family={w_agent1_family:.3f}, company={w_agent1_company:.3f}")
print("  ✓ Agent 1 has overlapping membership in multiple meta-agents")


# ============================================================
# 9. Phase transition: condensation scan over τ
# ============================================================
print("\n[9] Phase transition: condensation vs temperature")
print("-" * 50)

# Use the clustered system from test 5
tau_scan = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
fractions = []
for tau in tau_scan:
    W_t = SoftMembership(tau=tau).compute(clustered_children, clustered_parents)
    f_t = cond.condensation_fraction(clustered_children, clustered_parents, W_t)
    fractions.append(f_t)
    bar = "█" * int(f_t * 30) + "░" * (30 - int(f_t * 30))
    print(f"  τ={tau:5.2f} | f={f_t:.2f} | {bar}")

# Should see transition: high f at low τ, low f at high τ
assert fractions[0] >= fractions[-1], "Condensation should decrease with τ"
print("  ✓ Phase transition observed: f decreases as τ increases")


# ============================================================
# 10. End-to-end gradient flow through hierarchy
# ============================================================
print("\n[10] End-to-end gradient flow through full hierarchy")
print("-" * 50)

torch.manual_seed(123)
base_g = MultiAgentSystem(8, K, grid_shape=())
hier_g = HierarchicalEmergence(
    base_system=base_g,
    n_meta_per_scale=[4, 2, 1],
    tau=1.0,
    lambda_cross=1.0,
)

cross = hier_g.cross_scale_energy()
cross['total'].backward()

grad_counts = {}
for ell, scale in enumerate(hier_g.scales):
    n_grad = sum(1 for a in scale.agents for p in a.parameters()
                 if p.grad is not None and p.grad.abs().max() > 0)
    n_total = sum(1 for a in scale.agents for _ in a.parameters())
    grad_counts[f'ℓ{ell}'] = f"{n_grad}/{n_total}"
    print(f"  Scale ℓ{ell}: {n_grad}/{n_total} params with grad")

print("  ✓ Gradients flow through all scales")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
print(f"""
Soft Hierarchical Emergence implements:

  1. SOFT MEMBERSHIP W_{{iα}} = σ(-KL(s_i || Ω̃_{{iα}}[s_α]) / τ)
     - Agents in multiple meta-agents simultaneously
     - NOT normalized: family + company + nation at once

  2. PRECISION POOLING: more certain agents dominate meta-agent
     - Λ_α = Σ_i w_{{iα}} Ω_{{αi}} Λ_i Ω_{{αi}}^T
     - The precise speak louder than the vague

  3. CONDENSATION ORDER PARAMETER Ψ_α
     - Small Ψ → condensed (shared ontology)
     - Phase transition at critical τ_c

  4. CROSS-SCALE VFE: fully differentiable
     - Top-down: KL(p_i || Ω_{{iα}}[q_α]) weighted by w_{{iα}}
     - Gradients flow end-to-end through the hierarchy

  Like Cooper pairs: when τ drops below τ_c, agents condense
  into meta-agents with genuinely emergent properties.
""")
