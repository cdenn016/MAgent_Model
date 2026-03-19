---
name: plotly
description: Interactive visualization with Plotly for exploratory data analysis. Use when creating interactive 3D plots, animated visualizations, attention heatmaps with hover details, or dashboard-style figures. For static publication-ready figures use scientific-visualization instead.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Plotly — Interactive Visualization

## Overview

Plotly is a Python library for creating interactive, publication-quality visualizations that support zooming, hovering, and animation. This skill provides guidance for building interactive figures for exploratory analysis of attention patterns, gauge frame evolution, and model diagnostics.

## When to Use This Skill

Use this skill when:
- Creating interactive 3D visualizations (gauge frames in GL(K) space, belief manifolds)
- Building attention heatmaps with hover details and drill-down
- Animating belief evolution across VFE iterations or training steps
- Exploring high-dimensional data interactively before generating static publication figures
- Creating dashboard-style multi-panel interactive displays
- Sharing interactive HTML figures with collaborators

## Core Capabilities

### 1. Interactive Attention Heatmaps

```python
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def interactive_attention_heatmap(attention_weights, tokens, layer=0, head=0):
    """Interactive attention heatmap with hover details."""
    attn = attention_weights[layer, head].detach().cpu().numpy()

    fig = px.imshow(
        attn,
        x=tokens,
        y=tokens,
        color_continuous_scale='Viridis',
        labels=dict(x='Key Token', y='Query Token', color='Attention Weight'),
        title=f'Attention Pattern — Layer {layer}, Head {head}'
    )
    fig.update_layout(width=700, height=600)
    return fig
```

### 2. 3D Gauge Frame Visualization

```python
def plot_gauge_frames_3d(frames, labels=None, colorby='layer'):
    """Visualize learned gauge frames in 3D (first 3 components)."""
    # frames: (n_tokens, K, K) — take first 3 diagonal elements or PCA
    coords = frames.reshape(frames.shape[0], -1)

    # Reduce to 3D if needed
    if coords.shape[1] > 3:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=3).fit_transform(coords)

    fig = go.Figure(data=[go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='markers',
        marker=dict(size=4, color=coords[:, 0], colorscale='Plasma', opacity=0.8),
        text=labels,
        hovertemplate='<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}'
    )])
    fig.update_layout(
        title='Gauge Frame Structure in GL(K) Space',
        scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3')
    )
    return fig
```

### 3. Animated Training Dynamics

```python
def animate_belief_evolution(beliefs_over_time, step_labels):
    """Animate how beliefs (mu, Sigma) evolve during VFE iterations."""
    frames = []
    for i, (mu, label) in enumerate(zip(beliefs_over_time, step_labels)):
        frames.append(go.Frame(
            data=[go.Scatter(x=list(range(len(mu))), y=mu, mode='lines+markers')],
            name=str(i),
            layout=go.Layout(title_text=f'Belief State — {label}')
        ))

    fig = go.Figure(
        data=[go.Scatter(x=list(range(len(beliefs_over_time[0]))),
                         y=beliefs_over_time[0], mode='lines+markers')],
        frames=frames,
        layout=go.Layout(
            updatemenus=[dict(type='buttons', showactive=False,
                buttons=[dict(label='Play',
                              method='animate',
                              args=[None, dict(frame=dict(duration=200))])])],
            xaxis_title='Dimension', yaxis_title='μ value',
            title='Belief Evolution Over VFE Iterations'
        )
    )
    return fig
```

### 4. Loss Landscape Visualization

```python
def plot_loss_landscape_3d(param_grid_x, param_grid_y, loss_surface):
    """3D surface plot of loss landscape around a minimum."""
    fig = go.Figure(data=[go.Surface(
        x=param_grid_x, y=param_grid_y, z=loss_surface,
        colorscale='RdBu_r', opacity=0.9
    )])
    fig.update_layout(
        title='VFE Loss Landscape',
        scene=dict(xaxis_title='Param 1', yaxis_title='Param 2', zaxis_title='VFE Loss')
    )
    return fig
```

### 5. Interactive Subplots

```python
from plotly.subplots import make_subplots

def multi_panel_dashboard(perplexity, entropy, clustering_coeff, steps):
    """Multi-panel interactive dashboard for training metrics."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Perplexity', 'Entropy', 'Clustering Coefficient', 'Combined')
    )
    fig.add_trace(go.Scatter(x=steps, y=perplexity, name='Perplexity'), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=entropy, name='Entropy'), row=1, col=2)
    fig.add_trace(go.Scatter(x=steps, y=clustering_coeff, name='Clustering'), row=2, col=1)

    # Combined normalized view
    for name, vals in [('PPL', perplexity), ('H', entropy), ('C', clustering_coeff)]:
        normed = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        fig.add_trace(go.Scatter(x=steps, y=normed, name=name), row=2, col=2)

    fig.update_layout(height=800, title_text='Training Dashboard')
    return fig
```

---

## Export and Integration

### Save as Interactive HTML

```python
fig.write_html('figures/interactive_attention.html', include_plotlyjs='cdn')
```

### Save as Static Image (for manuscripts)

```python
# Requires kaleido: pip install kaleido
fig.write_image('Attention/figs/attention_heatmap.pdf', width=800, height=600, scale=2)
fig.write_image('Attention/figs/attention_heatmap.png', width=800, height=600, scale=2)
```

### Integration with Scientific-Visualization

Use plotly for exploration, then recreate final figures with matplotlib + scientific-visualization for publication:

```python
# Explore interactively
fig = interactive_attention_heatmap(attn_weights, tokens)
fig.show()

# Then create publication version with matplotlib
# using scientific-visualization skill for journal formatting
```

---

## Plotly Express vs Graph Objects

- **`plotly.express` (px)**: High-level, concise API for common plot types. Start here.
- **`plotly.graph_objects` (go)**: Low-level, full control. Use for custom layouts and animations.

```python
# Express — quick exploration
fig = px.scatter_3d(df, x='pc1', y='pc2', z='pc3', color='layer', hover_data=['token'])

# Graph Objects — full control
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])
```

---

## Best Practices

1. **Use hover templates** to show relevant metadata (token text, layer, head, weight values)
2. **Set reasonable defaults** for figure size and font with `fig.update_layout()`
3. **Use `plotly.subplots.make_subplots`** for multi-panel figures with shared axes
4. **Export HTML** for sharing interactive figures; export PDF/PNG for manuscripts
5. **Use animation frames** for time-series or iteration-based data
6. **Colorscales**: Use `Viridis`, `Plasma`, or `RdBu_r` for accessibility
7. **Performance**: For large datasets (>10k points), use `go.Scattergl` (WebGL) instead of `go.Scatter`

---

## Dependencies

```
pip install plotly kaleido
```
