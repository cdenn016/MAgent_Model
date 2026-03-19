---
name: literature-review
description: Systematic literature review and synthesis for research papers. Use when building comprehensive related-work sections, synthesizing findings across papers, or identifying research gaps. For searching individual papers use arxiv-database; for writing manuscript sections use scientific-writing.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Literature Review — Systematic Review and Synthesis

## Overview

This skill provides structured methodology for conducting systematic literature reviews, synthesizing findings across papers, and building comprehensive related-work sections. It covers search strategies, inclusion/exclusion criteria, thematic synthesis, and gap analysis.

## When to Use This Skill

Use this skill when:
- Building or expanding the related work section of a manuscript
- Conducting a systematic review of a research area
- Synthesizing findings across multiple papers into coherent themes
- Identifying research gaps that the Gauge-Transformer addresses
- Comparing methodological approaches across the literature
- Organizing references by theme for the GL(K) attention paper

---

## Core Workflow

### Phase 1: Define Scope and Search Strategy

```markdown
## Review Protocol

### Research Questions
1. Primary: What approaches exist for incorporating geometric structure into attention mechanisms?
2. Secondary: How has gauge theory been applied in deep learning?
3. Secondary: What variational methods have been used for attention computation?

### Inclusion Criteria
- Peer-reviewed or reputable preprint (arXiv with >N citations)
- Published within relevant timeframe
- Addresses geometric/algebraic structure in neural networks
- Relevant to at least one: gauge theory, information geometry, VFE, attention theory

### Exclusion Criteria
- Application-only papers without methodological contribution
- Non-English publications
- Duplicate or superseded versions

### Search Databases
- arXiv (cs.LG, cs.CL, stat.ML, hep-th, math-ph)
- Google Scholar
- Semantic Scholar API
- ACL Anthology (for NLP-specific work)
```

### Phase 2: Search and Screen

```python
def systematic_search(databases, queries, criteria):
    """Execute systematic search across databases."""
    results = {
        'identified': [],      # All papers found
        'screened': [],        # After title/abstract screening
        'eligible': [],        # After full-text screening
        'included': [],        # Final included set
        'excluded_reasons': {} # Track exclusion reasons
    }
    return results
```

### Phase 3: Thematic Organization

Organize papers into themes relevant to the Gauge-Transformer:

```markdown
## Thematic Structure for GL(K) Attention Paper

### Theme 1: Geometric Deep Learning
- Equivariant neural networks (Cohen & Welling, 2016; Weiler et al., 2018)
- Gauge equivariant CNNs (Cohen et al., 2019)
- E(n)-equivariant networks (Satorras et al., 2021)
- **Gap**: No work applies gauge theory to attention/transformers specifically

### Theme 2: Information-Geometric Approaches
- Natural gradient methods (Amari, 1998)
- Fisher information in neural networks (Martens, 2020)
- Information geometry of attention (limited existing work)
- **Gap**: KL divergence as attention score is unexplored

### Theme 3: Variational Methods for Transformers
- Variational attention (Deng et al., 2018)
- Bayesian transformers (Wang et al., 2020)
- Free energy principle in ML (Friston et al., 2006; Millidge et al., 2021)
- **Gap**: VFE minimization as the core transformer objective

### Theme 4: Attention Mechanism Theory
- Attention as kernel methods (Tsai et al., 2019)
- Theoretical analysis of self-attention (Dong et al., 2021)
- Attention and alignment (Bahdanau et al., 2015)
- **Gap**: Lack of principled geometric foundation for attention

### Theme 5: Renormalization Group in ML
- RG and deep learning (Mehta & Schwab, 2014)
- Hierarchical coarse-graining in neural networks
- **Gap**: RG analysis of attention patterns across layers
```

---

## Synthesis Methods

### 1. Narrative Synthesis

Write flowing prose that connects papers thematically (use scientific-writing skill for actual prose generation):

```markdown
## Synthesis Template

[Theme introduction — why this line of work matters]

[Foundational work — seminal papers that established the area]

[Key developments — how the field evolved]

[Current state — what recent work has achieved]

[Limitations/gaps — what remains unaddressed]

[Connection to our work — how the Gauge-Transformer addresses the gap]
```

### 2. Comparative Analysis Table

```markdown
| Method | Geometric Structure | Attention Type | Invariance | VFE |
|--------|-------------------|----------------|------------|-----|
| Standard Transformer | None | Dot-product | None | No |
| Gauge Equiv. CNN | Gauge fields | N/A (conv) | Gauge | No |
| Bayesian Transformer | Probabilistic | Learned | None | Partial |
| **Gauge-Transformer** | **GL(K) gauge** | **KL-divergence** | **GL(K)** | **Yes** |
```

### 3. Gap Analysis

```markdown
## Research Gap Identification

### Methodological Gaps
1. No existing work combines gauge theory with attention mechanisms
2. KL divergence between belief distributions unused as attention score
3. VFE minimization not formulated as transformer objective
4. GL(K) invariance in attention is novel

### Empirical Gaps
1. No comparison of geometric vs. learned attention on language modeling
2. RG flow analysis of attention patterns not explored
3. Information-theoretic properties of attention under-measured

### Theoretical Gaps
1. Gauge-theoretic foundation for attention lacking
2. Connection between VFE and standard cross-entropy loss unexplored
3. Relationship between attention gauge invariance and generalization unknown
```

---

## Citation Management

### Organizing References by Theme

```python
def organize_bibliography(papers, themes):
    """Organize papers into thematic groups for the bibliography."""
    organized = {theme: [] for theme in themes}

    for paper in papers:
        for theme, keywords in themes.items():
            if any(kw.lower() in paper['abstract'].lower() for kw in keywords):
                organized[theme].append(paper)

    return organized

# Example usage for Gauge-Transformer
themes = {
    'geometric_dl': ['equivariant', 'gauge', 'geometric deep learning', 'fiber bundle'],
    'information_geometry': ['Fisher information', 'natural gradient', 'information geometry'],
    'variational': ['variational', 'free energy', 'ELBO', 'variational inference'],
    'attention_theory': ['attention mechanism', 'self-attention', 'transformer theory'],
    'rg_ml': ['renormalization', 'coarse-graining', 'multiscale'],
}
```

### BibTeX Management

```python
def merge_bibtex_entries(entries, output_path='references.bib'):
    """Merge and deduplicate BibTeX entries."""
    seen_keys = set()
    unique_entries = []
    for entry in entries:
        key = entry.split('{')[1].split(',')[0]
        if key not in seen_keys:
            seen_keys.add(key)
            unique_entries.append(entry)

    with open(output_path, 'w') as f:
        f.write('\n\n'.join(unique_entries))

    return len(unique_entries)
```

---

## PRISMA Checklist (for Systematic Reviews)

If conducting a formal systematic review, follow PRISMA guidelines:

1. **Identification**: Record total papers found per database
2. **Screening**: Document title/abstract screening decisions
3. **Eligibility**: Full-text assessment with exclusion reasons
4. **Inclusion**: Final included set with characteristics table
5. **Flow diagram**: PRISMA flow diagram showing paper counts at each stage

---

## Best Practices

1. **Document your search strategy** — record exact queries, databases, dates
2. **Use multiple databases** — arXiv alone is insufficient for systematic reviews
3. **Screen in stages** — title → abstract → full text
4. **Track exclusion reasons** — important for transparency
5. **Synthesize, don't just summarize** — draw connections between papers
6. **Identify gaps explicitly** — this motivates your contribution
7. **Update regularly** — set up monitoring with arxiv-database skill
8. **Use forward and backward citation tracking** — follow references and citing papers
9. **Be critical** — note methodological limitations in reviewed papers
10. **Organize by theme, not chronology** — thematic organization is more useful for readers

---

## Integration with Other Skills

- **arxiv-database**: Use for searching and retrieving papers
- **scientific-writing**: Use for writing the actual related-work prose
- **citation-management**: Use for organizing and formatting references
- **peer-review**: Use reviewed literature to contextualize your evaluation

---

## Dependencies

```
# No additional dependencies for the methodology
# For automated searching, see arxiv-database skill
```
