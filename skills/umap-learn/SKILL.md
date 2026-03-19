---
name: umap-learn
description: Dimensionality reduction with UMAP for visualizing high-dimensional representations. Use when embedding token beliefs, gauge frames, or hidden states into 2D/3D for visualization. For interactive display of reduced embeddings use plotly; for publication figures use scientific-visualization.
license: BSD-3-Clause
metadata:
    skill-author: K-Dense Inc.
---

# UMAP-Learn — Dimensionality Reduction and Visualization

## Overview

UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction algorithm that preserves both local and global structure. This skill provides guidance for embedding high-dimensional learned representations (beliefs, gauge frames, hidden states) into 2D/3D for visualization and analysis.

## When to Use This Skill

Use this skill when:
- Visualizing token belief distributions (μ, Σ) in low dimensions
- Comparing gauge frame structure across layers or training steps
- Exploring semantic clustering in the learned embedding space
- Visualizing how representations change during VFE iterations
- Creating 2D/3D embeddings for publication figures
- Comparing representations between the Gauge-Transformer and standard models

## Core Capabilities

### 1. Basic UMAP Embedding

```python
import umap
import numpy as np
import matplotlib.pyplot as plt

def embed_representations(representations, n_components=2, n_neighbors=15, min_dist=0.1,
                          metric='euclidean', random_state=42):
    """Embed high-dimensional representations with UMAP.

    Args:
        representations: (n_samples, n_features) array
        n_components: output dimensionality (2 or 3)
        n_neighbors: controls local vs global structure (5-50)
        min_dist: minimum distance between points in embedding (0.0-1.0)
        metric: distance metric ('euclidean', 'cosine', 'correlation', etc.)
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    embedding = reducer.fit_transform(representations)
    return embedding, reducer
```

### 2. Visualizing Token Beliefs

```python
def visualize_token_beliefs(mu, sigma_diag, tokens, n_neighbors=15):
    """Embed and visualize token belief distributions.

    Args:
        mu: (n_tokens, K) mean vectors
        sigma_diag: (n_tokens, K) diagonal covariance entries
        tokens: list of token strings
    """
    # Concatenate mu and sigma for a joint representation
    belief_features = np.concatenate([mu, sigma_diag], axis=1)

    embedding, _ = embed_representations(belief_features, n_neighbors=n_neighbors)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Color by mean norm (precision of belief)
    mu_norm = np.linalg.norm(mu, axis=1)
    sc1 = axes[0].scatter(embedding[:, 0], embedding[:, 1], c=mu_norm,
                          cmap='viridis', s=20, alpha=0.7)
    axes[0].set_title('Token Beliefs — Colored by |μ|')
    plt.colorbar(sc1, ax=axes[0], label='|μ|')

    # Color by uncertainty (trace of Σ)
    uncertainty = sigma_diag.sum(axis=1)
    sc2 = axes[1].scatter(embedding[:, 0], embedding[:, 1], c=uncertainty,
                          cmap='plasma', s=20, alpha=0.7)
    axes[1].set_title('Token Beliefs — Colored by Tr(Σ)')
    plt.colorbar(sc2, ax=axes[1], label='Tr(Σ)')

    for ax in axes:
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

    plt.tight_layout()
    return fig
```

### 3. Gauge Frame Evolution Across Training

```python
def track_gauge_frame_evolution(frames_by_step, step_labels, sample_indices=None):
    """Visualize how gauge frames change during training.

    Args:
        frames_by_step: list of (n_tokens, K, K) arrays at different training steps
        step_labels: list of step identifiers
        sample_indices: optional subset of token indices to track
    """
    # Flatten frames and combine all steps
    all_frames = []
    all_steps = []
    all_indices = []

    for step_idx, (frames, label) in enumerate(zip(frames_by_step, step_labels)):
        flat = frames.reshape(frames.shape[0], -1)
        if sample_indices is not None:
            flat = flat[sample_indices]
            indices = sample_indices
        else:
            indices = list(range(len(flat)))

        all_frames.append(flat)
        all_steps.extend([step_idx] * len(flat))
        all_indices.extend(indices)

    combined = np.vstack(all_frames)
    embedding, _ = embed_representations(combined, n_neighbors=20)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=all_steps, cmap='coolwarm', s=15, alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_ticks(range(len(step_labels)))
    cbar.set_ticklabels(step_labels)
    ax.set_title('Gauge Frame Evolution During Training')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    return fig
```

### 4. Layer-wise Representation Comparison

```python
def compare_representations_across_layers(hidden_states_by_layer, layer_names):
    """Compare how representations differ across transformer layers.

    Args:
        hidden_states_by_layer: list of (n_tokens, d_model) arrays per layer
    """
    all_states = np.vstack(hidden_states_by_layer)
    all_layers = []
    for i, states in enumerate(hidden_states_by_layer):
        all_layers.extend([i] * len(states))

    embedding, _ = embed_representations(all_states, n_neighbors=30)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=all_layers, cmap='tab10', s=10, alpha=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_ticks(range(len(layer_names)))
    cbar.set_ticklabels(layer_names)
    ax.set_title('Representation Structure Across Layers')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    return fig
```

### 5. Comparing Gauge-Transformer vs Standard Transformer

```python
def compare_model_representations(gauge_reps, standard_reps, labels_gauge, labels_standard):
    """Side-by-side UMAP comparison of representations from two models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, reps, labels, title in [
        (axes[0], gauge_reps, labels_gauge, 'Gauge-Transformer'),
        (axes[1], standard_reps, labels_standard, 'Standard Transformer')
    ]:
        embedding, _ = embed_representations(reps, n_neighbors=15)
        ax.scatter(embedding[:, 0], embedding[:, 1], c=labels,
                  cmap='tab20', s=15, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

    plt.tight_layout()
    return fig
```

### 6. 3D UMAP with Plotly

```python
def umap_3d_interactive(representations, color_values, hover_texts, title='3D UMAP'):
    """Create interactive 3D UMAP visualization."""
    import plotly.graph_objects as go

    embedding, _ = embed_representations(representations, n_components=3, n_neighbors=15)

    fig = go.Figure(data=[go.Scatter3d(
        x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
        mode='markers',
        marker=dict(size=3, color=color_values, colorscale='Viridis', opacity=0.7),
        text=hover_texts,
        hovertemplate='<b>%{text}</b><br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<br>UMAP3: %{z:.2f}'
    )])
    fig.update_layout(title=title,
                     scene=dict(xaxis_title='UMAP 1', yaxis_title='UMAP 2', zaxis_title='UMAP 3'))
    return fig
```

---

## Parameter Tuning Guide

### `n_neighbors` (default: 15)

Controls the balance between local and global structure:

| Value | Effect |
|-------|--------|
| 5-10 | Preserves very local structure; more fragmented clusters |
| 15-30 | Good balance (recommended starting point) |
| 50-100 | Preserves more global structure; larger connected regions |

### `min_dist` (default: 0.1)

Controls how tightly points cluster:

| Value | Effect |
|-------|--------|
| 0.0 | Points can overlap; tightest clusters |
| 0.1-0.3 | Moderate spacing (recommended) |
| 0.5-1.0 | Spread out; less clustering |

### `metric`

Choose based on data type:

| Data Type | Recommended Metric |
|-----------|--------------------|
| Hidden states | `cosine` or `euclidean` |
| Belief means (μ) | `euclidean` |
| Covariance features | `correlation` |
| Mixed features | `euclidean` (after normalization) |

---

## SPD Manifold Considerations

For the Gauge-Transformer, covariance matrices Σ live on the SPD manifold. Standard Euclidean UMAP may not respect this geometry. Options:

1. **Log-Euclidean metric**: Map Σ → log(Σ), then use standard UMAP
2. **Affine-invariant metric**: Use custom UMAP metric

```python
from scipy.linalg import logm

def spd_to_log_euclidean(covariance_matrices):
    """Map SPD matrices to log-Euclidean space for UMAP."""
    log_covs = np.array([logm(S) for S in covariance_matrices])
    # Flatten upper triangle (symmetric)
    n = log_covs.shape[1]
    features = []
    for lc in log_covs:
        upper = lc[np.triu_indices(n)]
        features.append(upper)
    return np.array(features)

# Usage
log_features = spd_to_log_euclidean(sigma_matrices)
embedding, _ = embed_representations(log_features, metric='euclidean')
```

---

## Reproducibility

UMAP is stochastic. For reproducible results:

```python
reducer = umap.UMAP(random_state=42)

# For publication: run multiple seeds and check stability
embeddings = []
for seed in range(5):
    reducer = umap.UMAP(random_state=seed, n_neighbors=15)
    emb = reducer.fit_transform(data)
    embeddings.append(emb)
# Compare cluster structure across seeds using trustworthiness or silhouette score
```

---

## Best Practices

1. **Normalize features** before UMAP if they have different scales
2. **Use `cosine` metric** for hidden states (direction matters more than magnitude)
3. **Report parameters** in figures — always state n_neighbors, min_dist, metric, random_state
4. **Don't over-interpret distances** — UMAP preserves topology, not exact distances
5. **Run multiple seeds** to verify that cluster structure is stable
6. **Pre-reduce with PCA** for very high-dimensional data (>1000 dims) for speed
7. **Use `transform()`** to embed new points into an existing UMAP space
8. **Log-Euclidean mapping** for covariance matrices on SPD manifold

---

## Dependencies

```
pip install umap-learn scikit-learn matplotlib
# Optional: pip install plotly  (for 3D interactive visualization)
```
