---
name: hypothesis-generation
description: Systematic generation of testable research hypotheses from the gauge-theoretic framework. Use when brainstorming scaling laws, cross-linguistic predictions, architectural ablations, or novel experimental directions for the Gauge-Transformer. For literature context use literature-review; for experimental validation use statistical-analysis.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Hypothesis Generation — Systematic Research Direction Planning

## Overview

This skill provides a structured methodology for generating testable hypotheses and research predictions from the Gauge-Transformer's gauge-theoretic framework. It covers hypothesis formulation, operationalization, experimental design, and prioritization — turning theoretical claims into concrete experiments.

## When to Use This Skill

Use this skill when:
- Generating testable predictions from gauge invariance properties
- Designing ablation studies for VFE components
- Formulating scaling law hypotheses for geometric attention
- Predicting cross-linguistic universality from gauge symmetry
- Planning the experimental roadmap for future work
- Writing the "Future Directions" section of the manuscript
- Brainstorming novel applications of the gauge-theoretic framework

---

## Hypothesis Generation Framework

### Phase 1: Identify Theoretical Claims

Start by enumerating the core theoretical claims of the Gauge-Transformer:

```markdown
## Core Theoretical Claims

1. **Gauge Invariance**: KL divergence attention is invariant under GL(K) transformations
   → Predictions should not depend on the coordinate frame of belief representations

2. **VFE Minimization**: The transformer objective is equivalent to variational free energy
   → Cross-entropy loss is a special case of VFE under specific assumptions

3. **Hierarchical Structure**: h → s → p → q mirrors renormalization group flow
   → Deeper layers should show coarser-grained, more abstract representations

4. **Information-Geometric Attention**: KL-based attention is principled on the statistical manifold
   → Should outperform dot-product attention when data has distributional structure

5. **Gauge Frame Learning**: GL(K) gauge frames encode relational structure
   → Gauge frames should capture syntactic/semantic relationships between tokens
```

### Phase 2: Derive Testable Hypotheses

For each theoretical claim, generate specific, falsifiable hypotheses:

```markdown
## Hypothesis Template

### H[number]: [Concise statement]

**Derived from**: [Which theoretical claim]
**Prediction**: [Specific, measurable prediction]
**Null hypothesis**: [What we'd observe if the claim is wrong]
**Operationalization**: [How to measure it]
**Required data/compute**: [Resources needed]
**Priority**: [High/Medium/Low]
**Risk**: [What could confound the result]
```

---

## Hypothesis Categories for the Gauge-Transformer

### Category 1: Gauge Invariance Predictions

```markdown
### H1.1: Representation Invariance Under GL(K) Transformations
**Prediction**: Applying random GL(K) transformations to all gauge frames simultaneously
should not change model predictions (perplexity).
**Test**: Apply random G ∈ GL(K) to all frames, measure perplexity change.
**Expected**: |Δperplexity| < ε (numerical precision).
**Null**: Perplexity changes significantly under gauge transformations.
**Priority**: High — this is the foundational claim.

### H1.2: Gauge Frame Redundancy
**Prediction**: The effective dimensionality of learned gauge frames is less than K²
(not all GL(K) degrees of freedom are used).
**Test**: PCA on flattened gauge frames; measure explained variance ratio.
**Expected**: >95% variance explained by d << K² components.
**Priority**: Medium.

### H1.3: Gauge-Invariant Representations Transfer Better
**Prediction**: Gauge-Transformer representations should transfer better to downstream
tasks than standard transformer representations, because gauge invariance removes
spurious coordinate-dependent features.
**Test**: Linear probing on downstream tasks (NER, sentiment, etc.) comparing
Gauge-Transformer vs. standard transformer hidden states.
**Expected**: Higher linear probe accuracy for Gauge-Transformer.
**Priority**: Medium.
```

### Category 2: Scaling Laws

```markdown
### H2.1: Geometric Attention Scaling Exponent
**Prediction**: The Gauge-Transformer's loss-vs-compute scaling follows a power law
L(C) = aC^{-α} with a different exponent α than standard transformers.
**Test**: Train models at 5+ scales (10M to 1B parameters), fit scaling curves.
**Expected**: α_gauge ≠ α_standard (and ideally α_gauge > α_standard).
**Priority**: High — major empirical result if confirmed.

### H2.2: Data Efficiency from Geometric Inductive Bias
**Prediction**: The Gauge-Transformer achieves a given perplexity with less training
data than a standard transformer of the same size.
**Test**: Train both models on {10%, 25%, 50%, 100%} of WikiText-103, compare
learning curves.
**Expected**: Gauge-Transformer reaches target perplexity with fewer tokens.
**Priority**: High.

### H2.3: Gauge Frame Dimension Scaling
**Prediction**: Optimal K (gauge frame dimension) scales sub-linearly with model
dimension d_model: K_opt ∝ d_model^β where β < 1.
**Test**: Grid search over K at multiple d_model values.
**Expected**: β ∈ (0.3, 0.7).
**Priority**: Medium.
```

### Category 3: Cross-Linguistic Universality

```markdown
### H3.1: Gauge Frame Universality Across Languages
**Prediction**: Gauge frames learned on different languages share similar spectral
structure (eigenvalue distributions), reflecting universal syntactic properties.
**Test**: Train on English, German, Chinese, Arabic; compare gauge frame spectra.
**Expected**: Spectral similarity > random baseline (measured by Wasserstein distance
between eigenvalue distributions).
**Priority**: Medium-High.

### H3.2: Typological Prediction from Gauge Frames
**Prediction**: Clustering gauge frames should recover known typological categories
(e.g., SVO vs. SOV word order).
**Test**: Train on diverse languages, cluster gauge frames, compare to WALS features.
**Expected**: Significant correlation between gauge frame clusters and typological features.
**Priority**: Medium.

### H3.3: Transfer Learning Benefits
**Prediction**: Gauge-Transformer pre-trained on a high-resource language transfers
better to low-resource languages than a standard transformer, because gauge invariance
captures language-universal structure.
**Test**: Pre-train on English, fine-tune on low-resource languages, compare with
standard transformer baseline.
**Expected**: Larger improvement for typologically distant languages.
**Priority**: Medium.
```

### Category 4: VFE Component Analysis

```markdown
### H4.1: KL Term as Regularizer
**Prediction**: The KL divergence term in VFE acts as an adaptive regularizer whose
strength correlates with generalization gap.
**Test**: Track KL term magnitude vs. train-test perplexity gap across training.
**Expected**: Positive correlation between KL magnitude and generalization.
**Priority**: High — directly tests VFE interpretation.

### H4.2: VFE Decomposition Predicts Failure Modes
**Prediction**: When the model fails (high perplexity on specific inputs), the failure
is attributable to a specific VFE component (high KL = underfitting beliefs;
low likelihood = poor prediction).
**Test**: On worst-performing inputs, decompose VFE and classify failure modes.
**Expected**: Distinct failure clusters in VFE component space.
**Priority**: Medium.

### H4.3: Belief Calibration
**Prediction**: The learned beliefs q(z) are well-calibrated — the predicted uncertainty
(Σ_q) correlates with actual prediction error.
**Test**: Bin predictions by predicted uncertainty, measure actual error per bin.
**Expected**: Monotonic relationship between predicted uncertainty and actual error.
**Priority**: High — critical for the Bayesian interpretation.
```

### Category 5: RG Flow and Hierarchy

```markdown
### H5.1: Monotonic Information Compression Across Layers
**Prediction**: Mutual information I(input; hidden_l) decreases monotonically with
layer depth l, consistent with RG coarse-graining.
**Test**: Estimate mutual information at each layer using binning or MINE estimator.
**Expected**: Monotonic decrease (with possible plateau at final layers).
**Priority**: High.

### H5.2: Attention Graph Coarsening
**Prediction**: The attention graph becomes progressively sparser and more modular
at deeper layers, mirroring RG flow.
**Test**: Compute graph density, modularity, and number of communities per layer.
**Expected**: Density decreases, modularity increases with depth.
**Priority**: Medium — complements networkx analysis.

### H5.3: Fixed Point Structure
**Prediction**: The deepest layers approach a fixed point in representation space
(representations change less between consecutive layers).
**Test**: Measure ‖h_l - h_{l-1}‖ / ‖h_l‖ at each layer.
**Expected**: Ratio decreases with depth, approaching zero.
**Priority**: Medium.
```

---

## Prioritization Matrix

```markdown
| Hypothesis | Impact | Feasibility | Novelty | Priority Score |
|-----------|--------|-------------|---------|----------------|
| H1.1 Gauge invariance | High | Easy | Medium | ★★★★★ |
| H2.1 Scaling exponent | Very High | Hard (compute) | High | ★★★★★ |
| H4.1 KL as regularizer | High | Easy | High | ★★★★★ |
| H4.3 Belief calibration | High | Medium | High | ★★★★☆ |
| H5.1 Info compression | High | Medium | Medium | ★★★★☆ |
| H2.2 Data efficiency | High | Medium | Medium | ★★★★☆ |
| H3.1 Cross-linguistic | Very High | Hard (data) | Very High | ★★★★☆ |
| H1.3 Transfer learning | Medium | Medium | Medium | ★★★☆☆ |
| H5.2 Graph coarsening | Medium | Easy | Medium | ★★★☆☆ |
| H4.2 Failure modes | Medium | Medium | High | ★★★☆☆ |
```

---

## Experimental Design Template

For each hypothesis, plan the experiment:

```markdown
## Experiment Plan: [Hypothesis ID]

### Setup
- **Model configs**: [sizes, hyperparameters]
- **Data**: [dataset, splits, preprocessing]
- **Baselines**: [what to compare against]
- **Compute budget**: [GPU hours estimate]

### Measurements
- **Primary metric**: [what to measure]
- **Secondary metrics**: [supporting measurements]
- **Controls**: [what to hold constant]

### Analysis Plan
- **Statistical test**: [pre-registered test]
- **Effect size**: [minimum meaningful effect]
- **Sample size**: [number of runs/seeds]
- **Multiple comparison correction**: [if testing multiple hypotheses]

### Success Criteria
- **Supports hypothesis**: [specific threshold]
- **Refutes hypothesis**: [specific threshold]
- **Inconclusive**: [what would be ambiguous]

### Reporting
- **Figures**: [what plots to generate]
- **Tables**: [what summary statistics]
- **Manuscript section**: [where this goes in the paper]
```

---

## Hypothesis Refinement Process

1. **Generate broadly** — start with many hypotheses from each theoretical claim
2. **Filter by falsifiability** — remove anything that can't be clearly tested
3. **Operationalize precisely** — define exact metrics and thresholds
4. **Prioritize by impact × feasibility** — focus on high-value, achievable tests
5. **Pre-register** — commit to analysis plan before running experiments
6. **Iterate** — use results to refine theory and generate new hypotheses

---

## Integration with Other Skills

| Skill | Role in Hypothesis Testing |
|-------|---------------------------|
| **statistical-analysis** | Choose tests, compute effect sizes, report results |
| **pymc** | Bayesian hypothesis testing, model comparison |
| **networkx** | Test graph-theoretic hypotheses (H5.2) |
| **umap-learn** | Visualize representation hypotheses (H3.1, H5.1) |
| **shap** | Test component attribution hypotheses (H4.1, H4.2) |
| **scientific-writing** | Write up results for manuscript |
| **literature-review** | Contextualize hypotheses in existing work |
| **arxiv-database** | Check if hypotheses have been tested elsewhere |

---

## Best Practices

1. **Be specific** — "perplexity improves" is not a hypothesis; "perplexity decreases by >5% with p < 0.01" is
2. **Define null clearly** — know what falsification looks like
3. **One hypothesis per experiment** — don't confound multiple tests
4. **Pre-register analysis plans** — prevents p-hacking and HARKing
5. **Report negative results** — null results are informative about the theory
6. **Use Bayesian methods** when possible — quantify evidence for/against
7. **Consider confounds** — what else could explain the result?
8. **Iterate theory** — update the theoretical framework based on results
9. **Distinguish confirmatory vs. exploratory** — be transparent about which is which
10. **Think about mechanism** — a confirmed prediction is more convincing when the mechanism is clear

---

## Dependencies

```
# No additional dependencies for hypothesis generation methodology
# Experimental validation uses: statistical-analysis, pymc, networkx, shap, umap-learn
```
