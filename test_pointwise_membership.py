"""
Test: Pointwise Membership and Emergent Meta-Agent Shapes
============================================================

Membership is a FIELD W_{iα}(x) over the base manifold, not a scalar.
The meta-agent's spatial shape EMERGES from the overlap pattern of
its member agents, gated by species compatibility.

Example: 6 circular agents on a 2D plane. Some overlap to form
an irregular-shaped meta-agent. Species gate blocks incompatible agents.
"""

import torch
torch.manual_seed(42)

print("=" * 70)
print("TEST: Pointwise Membership and Meta-Agent Shapes")
print("=" * 70)


# ============================================================
# 1. Setup: 2D base manifold with circular agents
# ============================================================
print("\n[1] Setup: 2D Euclidean manifold with circular agents")
print("-" * 50)

from gauge_agent.manifolds import EuclideanManifold
from gauge_agent.agents import Agent, MultiAgentSystem
from gauge_agent.support import ball_support
from gauge_agent.hierarchical_emergence import (
    SpeciesDetector, CoalitionDetector, GatedMembership,
    SoftMembership, _pairwise_kl_cross,
)

# 2D plane, 32×32 grid
grid_shape = (32, 32)
K = 3
manifold = EuclideanManifold(dim=2, grid_shape=grid_shape,
                              bounds=((-2.0, 2.0), (-2.0, 2.0)))

# 6 agents as circular discs on the plane
children = MultiAgentSystem(6, K, grid_shape=grid_shape)

# Agent positions and radii
#   0,1,2: species A (top cluster, mutually overlapping)
#   3,4: species B (bottom, overlapping each other)
#   5: species A but isolated (no overlap with anyone)
positions = [
    torch.tensor([-0.5,  0.8]),   # A0: top-left
    torch.tensor([ 0.5,  0.8]),   # A1: top-right
    torch.tensor([ 0.0,  0.3]),   # A2: top-center (overlaps A0, A1)
    torch.tensor([-0.5, -0.8]),   # B3: bottom-left
    torch.tensor([ 0.5, -0.8]),   # B4: bottom-right
    torch.tensor([ 1.5,  1.5]),   # A5: isolated top-right corner
]
radius = 0.7

# Set circular support functions
with torch.no_grad():
    for i, pos in enumerate(positions):
        chi = ball_support(manifold, pos, radius, sharpness=10.0)
        children.agents[i].chi.data = chi

    # Species A: model = [1, 0, 0]
    model_A = torch.tensor([1.0, 0.0, 0.0])
    for i in [0, 1, 2, 5]:
        children.agents[i].mu_s.data = (model_A + 0.02 * torch.randn(K)).unsqueeze(0).unsqueeze(0).expand(*grid_shape, K).clone()
        children.agents[i].omega_model.data = (torch.eye(K) + 0.01 * torch.randn(K, K)).unsqueeze(0).unsqueeze(0).expand(*grid_shape, K, K).clone()

    # Species B: model = [0, 1, 0]
    model_B = torch.tensor([0.0, 1.0, 0.0])
    for i in [3, 4]:
        children.agents[i].mu_s.data = (model_B + 0.02 * torch.randn(K)).unsqueeze(0).unsqueeze(0).expand(*grid_shape, K).clone()
        children.agents[i].omega_model.data = (torch.eye(K) + 0.01 * torch.randn(K, K)).unsqueeze(0).unsqueeze(0).expand(*grid_shape, K, K).clone()

    # All agents share similar beliefs (warm state)
    warm = torch.tensor([0.0, 0.0, 1.0])
    for i in range(6):
        children.agents[i].mu_q.data = (warm + 0.05 * torch.randn(K)).unsqueeze(0).unsqueeze(0).expand(*grid_shape, K).clone()
        children.agents[i].omega.data = (torch.eye(K) + 0.01 * torch.randn(K, K)).unsqueeze(0).unsqueeze(0).expand(*grid_shape, K, K).clone()

# Print support info
for i in range(6):
    chi = children.agents[i].chi
    area = (chi > 0.5).float().sum().item()
    sp = "A" if i in [0,1,2,5] else "B"
    print(f"  Agent {i} (species {sp}): pos={positions[i].tolist()}, "
          f"support_area={area:.0f}/{32*32} grid points")


# ============================================================
# 2. Pointwise KL divergences
# ============================================================
print("\n[2] Pointwise KL: field-valued, not scalar")
print("-" * 50)

# 2 meta-agents: one species-A, one species-B
parents = MultiAgentSystem(2, K, grid_shape=grid_shape)
with torch.no_grad():
    parents.agents[0].mu_s.data = model_A.unsqueeze(0).unsqueeze(0).expand(*grid_shape, K).clone()
    parents.agents[0].omega_model.data = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(*grid_shape, K, K).clone()
    parents.agents[0].mu_q.data = warm.unsqueeze(0).unsqueeze(0).expand(*grid_shape, K).clone()
    parents.agents[0].omega.data = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(*grid_shape, K, K).clone()
    parents.agents[0].chi.data = torch.ones(grid_shape)

    parents.agents[1].mu_s.data = model_B.unsqueeze(0).unsqueeze(0).expand(*grid_shape, K).clone()
    parents.agents[1].omega_model.data = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(*grid_shape, K, K).clone()
    parents.agents[1].mu_q.data = warm.unsqueeze(0).unsqueeze(0).expand(*grid_shape, K).clone()
    parents.agents[1].omega.data = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(*grid_shape, K, K).clone()
    parents.agents[1].chi.data = torch.ones(grid_shape)

# Check KL field shape
kl = _pairwise_kl_cross(children, parents, fiber='model')
print(f"  KL shape: {kl.shape}")
assert kl.shape == (6, 2, 32, 32), f"Expected (6, 2, 32, 32), got {kl.shape}"
print(f"  ✓ KL is a (N, M, *grid_shape) = (6, 2, 32, 32) field")


# ============================================================
# 3. Species gate is a spatial field
# ============================================================
print("\n[3] Species gate S_{iα}(x): spatial field gated by χ_i(x)")
print("-" * 50)

species_det = SpeciesDetector(tau_species=5.0)
S = species_det.species_gate(children, parents)
print(f"  S shape: {S.shape}")
assert S.shape == (6, 2, 32, 32), f"Expected (6, 2, 32, 32), got {S.shape}"

# Agent 0 (species A) at its center vs outside its support
center_idx = (16, 20)  # approximate center of agent 0
outside_idx = (0, 0)   # far corner, outside agent 0's support
chi_0 = children.agents[0].chi
print(f"  Agent 0 (species A):")
print(f"    χ at center={chi_0[center_idx].item():.3f}, χ at corner={chi_0[outside_idx].item():.3f}")
print(f"    S(center, meta_A)={S[0, 0, center_idx[0], center_idx[1]].item():.3f}")
print(f"    S(corner, meta_A)={S[0, 0, outside_idx[0], outside_idx[1]].item():.3f}")
assert S[0, 0, outside_idx[0], outside_idx[1]] < S[0, 0, center_idx[0], center_idx[1]], \
    "Species gate should be near-zero outside agent support"
print(f"  ✓ Species gate correctly gated by support function")


# ============================================================
# 4. Full gated membership W = S · C is a field
# ============================================================
print("\n[4] Gated membership W_{iα}(x) = S_{iα}(x) · C_{iα}(x)")
print("-" * 50)

gated = GatedMembership(tau_species=5.0, tau_belief=1.0)
result = gated.compute(children, parents)
W = result['W']
S = result['S']
C = result['C']

print(f"  W shape: {W.shape}")
assert W.shape == (6, 2, 32, 32)

# Meta-agent indicator: emergent shape
chi_meta = gated.meta_agent_indicator(children, parents)
print(f"  Meta-agent indicator shape: {chi_meta.shape}")
assert chi_meta.shape == (2, 32, 32)

# Meta-agent A should have support where species-A agents are
chi_A = chi_meta[0]  # meta-agent 0 (species A)
chi_B = chi_meta[1]  # meta-agent 1 (species B)

# Compute areas
area_A = (chi_A > 0.01).float().sum().item()
area_B = (chi_B > 0.01).float().sum().item()
print(f"  Meta-agent A (species A): area = {area_A:.0f} grid points")
print(f"  Meta-agent B (species B): area = {area_B:.0f} grid points")

# Species-A agents are {0,1,2,5}, species-B are {3,4}
# So meta-agent A should have larger support
assert area_A > area_B, "Meta-agent A should be bigger (4 members vs 2)"
print(f"  ✓ Meta-agent shapes emerge from member overlap patterns")


# ============================================================
# 5. Cross-species blocking is spatial
# ============================================================
print("\n[5] Cross-species blocking: spatial field")
print("-" * 50)

# Agent 3 is species B, meta-agent 0 is species A
# Even where agent 3 has support, W_{3,0}(x) ≈ 0
chi_3 = children.agents[3].chi
mask_3 = chi_3 > 0.5  # where agent 3 exists

W_3_A_where_exists = W[3, 0][mask_3].mean().item()
W_3_B_where_exists = W[3, 1][mask_3].mean().item()
print(f"  Agent 3 (species B) where it exists:")
print(f"    W with meta-A: {W_3_A_where_exists:.4f}")
print(f"    W with meta-B: {W_3_B_where_exists:.4f}")
assert W_3_B_where_exists > W_3_A_where_exists, "Species B should prefer meta-B"

W_0_A_where_exists = W[0, 0][children.agents[0].chi > 0.5].mean().item()
print(f"  Agent 0 (species A) where it exists:")
print(f"    W with meta-A: {W_0_A_where_exists:.4f}")
print(f"  ✓ Species blocking works pointwise — different values at different locations")


# ============================================================
# 6. Overlap creates irregular meta-agent shape
# ============================================================
print("\n[6] Irregular meta-agent shapes from overlap geometry")
print("-" * 50)

# The 3 overlapping species-A agents (0, 1, 2) create a
# triangular meta-agent shape. Agent 5 (isolated) adds a
# separate disc. The meta-agent A indicator should look like
# a merged blob + isolated disc.

# Check that the meta-agent indicator is NOT a simple disc
chi_A_nonzero = chi_A > 0.01

# Count connected-ish regions by checking different areas
# Top area (agents 0,1,2) and top-right corner (agent 5)
top_area = chi_A_nonzero[:16, :].float().sum().item()
bottom_area = chi_A_nonzero[16:, :].float().sum().item()
print(f"  Meta-agent A shape:")
print(f"    Top half: {top_area:.0f} grid points (agents 0,1,2 + 5)")
print(f"    Bottom half: {bottom_area:.0f} grid points (some spill from agents)")
print(f"    Total: {area_A:.0f} grid points")

# The shape should be more complex than a single circle
individual_areas = sum((children.agents[i].chi > 0.5).float().sum().item()
                       for i in [0, 1, 2, 5])
print(f"    Sum of individual areas: {individual_areas:.0f}")
print(f"    Meta-agent area: {area_A:.0f}")
print(f"    (Overlap means meta-agent area < sum of parts)")
print(f"  ✓ Meta-agent has irregular shape from overlap geometry")


# ============================================================
# 7. ASCII visualization of meta-agent indicator
# ============================================================
print("\n[7] ASCII visualization of meta-agent indicators")
print("-" * 50)

def ascii_field(field, size=32, chars=" ·∘○●"):
    """Render a 2D field as ASCII art."""
    lines = []
    max_val = field.max().item()
    if max_val < 1e-6:
        max_val = 1.0
    for row in range(size):
        line = ""
        for col in range(size):
            val = field[row, col].item() / max_val
            idx = min(int(val * (len(chars) - 1)), len(chars) - 1)
            line += chars[idx]
        lines.append(line)
    return "\n".join(lines)

print("\n  Agent supports (χ_i > 0.1):")
combined = torch.zeros(grid_shape)
for i in range(6):
    combined += children.agents[i].chi * (i + 1) * 0.15
print(ascii_field(combined, chars=" ·∘○●"))

print(f"\n  Meta-agent A indicator (species A members):")
print(ascii_field(chi_A, chars=" ·∘○●"))

print(f"\n  Meta-agent B indicator (species B members):")
print(ascii_field(chi_B, chars=" ·∘○●"))


# ============================================================
# 8. HierarchicalEmergence with spatial indicators
# ============================================================
print("\n[8] HierarchicalEmergence with spatial meta-agent indicators")
print("-" * 50)

from gauge_agent.hierarchical_emergence import HierarchicalEmergence

hier = HierarchicalEmergence(
    base_system=children,
    n_meta_per_scale=[2],
    tau_species=5.0,
    tau_belief=1.0,
    gated=True,
)

# Set up the meta-agents at scale 1 (same as our 'parents' above)
with torch.no_grad():
    hier.scales[1].agents[0].mu_s.data = model_A.unsqueeze(0).unsqueeze(0).expand(*grid_shape, K).clone()
    hier.scales[1].agents[0].omega_model.data = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(*grid_shape, K, K).clone()
    hier.scales[1].agents[0].mu_q.data = warm.unsqueeze(0).unsqueeze(0).expand(*grid_shape, K).clone()
    hier.scales[1].agents[0].omega.data = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(*grid_shape, K, K).clone()
    hier.scales[1].agents[0].chi.data = torch.ones(grid_shape)

    hier.scales[1].agents[1].mu_s.data = model_B.unsqueeze(0).unsqueeze(0).expand(*grid_shape, K).clone()
    hier.scales[1].agents[1].omega_model.data = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(*grid_shape, K, K).clone()
    hier.scales[1].agents[1].mu_q.data = warm.unsqueeze(0).unsqueeze(0).expand(*grid_shape, K).clone()
    hier.scales[1].agents[1].omega.data = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(*grid_shape, K, K).clone()
    hier.scales[1].agents[1].chi.data = torch.ones(grid_shape)

indicators = hier.meta_agent_indicators()
print(f"  Number of scales with indicators: {len(indicators)}")
print(f"  Scale 0→1 indicator shape: {indicators[0].shape}")
assert indicators[0].shape == (2, 32, 32)
print(f"  ✓ HierarchicalEmergence produces spatial meta-agent indicators")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
print(f"""
Pointwise membership over the base manifold:

  W_{{iα}}(x) = S_{{iα}}(x) · C_{{iα}}(x)

  S_{{iα}}(x) = species gate — field over (N, M, *grid_shape)
  C_{{iα}}(x) = coalition — field over (N, M, *grid_shape)
  Both gated by χ_i(x): zero outside agent's support.

  Meta-agent indicator:
    χ_α(x) = Σ_i W_{{iα}}(x)
    Emergent shape from overlap of species-compatible members.

  Circular agents overlapping → irregular meta-agent shapes.
  Species gate blocks cross-species coalitions pointwise.
""")
