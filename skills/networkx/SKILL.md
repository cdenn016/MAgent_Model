---
name: networkx
description: Graph and network analysis with NetworkX. Use when analyzing attention graphs, performing spectral clustering, community detection, or computing graph-theoretic metrics on learned token relationships. For visualizing networks interactively use plotly.
license: BSD-3-Clause
metadata:
    skill-author: K-Dense Inc.
---

# NetworkX — Graph and Network Analysis

## Overview

NetworkX is a Python library for creating, analyzing, and visualizing complex networks and graphs. This skill provides guidance for analyzing attention patterns as graphs, detecting communities in learned representations, computing spectral properties, and quantifying emergent structure in the Gauge-Transformer.

## When to Use This Skill

Use this skill when:
- Analyzing attention weight matrices as directed graphs
- Performing spectral clustering on attention patterns
- Detecting communities or modules in learned token relationships
- Computing graph metrics (clustering coefficient, betweenness, modularity)
- Analyzing RG flow as a network process (coarse-graining graph structure)
- Building adjacency matrices from attention heads for meta-agent detection

## Core Capabilities

### 1. Attention Graph Construction

```python
import networkx as nx
import numpy as np
import torch

def attention_to_graph(attention_weights, tokens, threshold=0.05, layer=0, head=0):
    """Convert attention matrix to a directed graph.

    Args:
        attention_weights: (n_layers, n_heads, seq_len, seq_len) tensor
        tokens: list of token strings
        threshold: minimum attention weight to include edge
    """
    attn = attention_weights[layer, head].detach().cpu().numpy()
    G = nx.DiGraph()

    for i, tok in enumerate(tokens):
        G.add_node(i, label=tok)

    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if attn[i, j] > threshold:
                G.add_edge(i, j, weight=attn[i, j])

    return G


def multi_head_attention_graph(attention_weights, tokens, layer=0, aggregation='mean'):
    """Aggregate attention across heads into a single graph."""
    attn = attention_weights[layer].detach().cpu().numpy()

    if aggregation == 'mean':
        agg_attn = attn.mean(axis=0)
    elif aggregation == 'max':
        agg_attn = attn.max(axis=0)

    G = nx.DiGraph()
    for i, tok in enumerate(tokens):
        G.add_node(i, label=tok)
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if agg_attn[i, j] > 0.01:
                G.add_edge(i, j, weight=float(agg_attn[i, j]))

    return G
```

### 2. Spectral Clustering of Attention Graphs

```python
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

def spectral_cluster_attention(attention_matrix, n_clusters=4):
    """Spectral clustering on the attention adjacency matrix."""
    # Symmetrize for undirected spectral analysis
    A = (attention_matrix + attention_matrix.T) / 2
    np.fill_diagonal(A, 0)

    # Degree matrix and normalized Laplacian
    D = np.diag(A.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(A.sum(axis=1), 1e-10)))
    L_norm = np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt

    # Compute smallest eigenvectors (skip first trivial one)
    eigenvalues, eigenvectors = eigsh(L_norm, k=n_clusters + 1, which='SM')
    embedding = eigenvectors[:, 1:n_clusters + 1]

    # Cluster in spectral embedding space
    labels = KMeans(n_clusters=n_clusters, n_init=10).fit_predict(embedding)
    return labels, eigenvalues, embedding
```

### 3. Community Detection

```python
def detect_attention_communities(G, method='louvain'):
    """Detect communities in attention graph."""
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()

    if method == 'louvain':
        communities = nx.community.louvain_communities(G_undirected, weight='weight')
    elif method == 'greedy_modularity':
        communities = list(nx.community.greedy_modularity_communities(G_undirected, weight='weight'))
    elif method == 'label_propagation':
        communities = list(nx.community.label_propagation_communities(G_undirected))

    # Compute modularity
    modularity = nx.community.modularity(G_undirected, communities, weight='weight')

    return communities, modularity


def track_community_evolution(attention_weights_over_time, tokens):
    """Track how attention communities evolve during training."""
    results = []
    for step, attn in attention_weights_over_time:
        G = attention_to_graph(attn, tokens, threshold=0.02)
        communities, modularity = detect_attention_communities(G)
        results.append({
            'step': step,
            'n_communities': len(communities),
            'modularity': modularity,
            'community_sizes': [len(c) for c in communities]
        })
    return results
```

### 4. Graph-Theoretic Metrics

```python
def compute_attention_graph_metrics(G):
    """Compute standard graph metrics on an attention graph."""
    metrics = {}

    # Basic metrics
    metrics['n_nodes'] = G.number_of_nodes()
    metrics['n_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)

    # Centrality measures
    metrics['in_degree_centrality'] = nx.in_degree_centrality(G)
    metrics['out_degree_centrality'] = nx.out_degree_centrality(G)
    metrics['pagerank'] = nx.pagerank(G, weight='weight')

    # Clustering (on undirected version)
    G_undir = G.to_undirected()
    metrics['avg_clustering'] = nx.average_clustering(G_undir, weight='weight')
    metrics['transitivity'] = nx.transitivity(G_undir)

    # Path-based
    if nx.is_weakly_connected(G):
        metrics['avg_path_length'] = nx.average_shortest_path_length(G)

    return metrics


def compare_heads_by_graph_structure(attention_weights, tokens, layer=0):
    """Compare attention heads by their graph-theoretic properties."""
    n_heads = attention_weights.shape[1]
    head_metrics = []

    for h in range(n_heads):
        G = attention_to_graph(attention_weights, tokens, threshold=0.02, layer=layer, head=h)
        m = compute_attention_graph_metrics(G)
        m['head'] = h
        head_metrics.append(m)

    return head_metrics
```

### 5. RG Flow as Network Coarse-Graining

```python
def rg_coarse_grain_graph(G, communities):
    """Coarse-grain a graph by contracting communities (RG step)."""
    # Create a mapping from node to community
    node_to_comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = i

    # Build coarse-grained graph
    G_coarse = nx.DiGraph()
    for i in range(len(communities)):
        G_coarse.add_node(i, size=len(communities[i]))

    for u, v, data in G.edges(data=True):
        cu, cv = node_to_comm[u], node_to_comm[v]
        if cu != cv:
            if G_coarse.has_edge(cu, cv):
                G_coarse[cu][cv]['weight'] += data.get('weight', 1.0)
            else:
                G_coarse.add_edge(cu, cv, weight=data.get('weight', 1.0))

    return G_coarse
```

---

## Visualization

### Static Visualization with Matplotlib

```python
import matplotlib.pyplot as plt

def draw_attention_graph(G, tokens, layout='spring', ax=None):
    """Draw attention graph with token labels."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    pos = getattr(nx, f'{layout}_layout')(G, weight='weight')
    edges = G.edges(data=True)
    weights = [d['weight'] * 3 for _, _, d in edges]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, labels={i: tokens[i] for i in G.nodes()},
                            ax=ax, font_size=8)
    nx.draw_networkx_edges(G, pos, ax=ax, width=weights, alpha=0.6,
                          edge_color='gray', arrows=True, arrowsize=10)
    ax.set_title('Attention Graph')
    return ax
```

### Interactive Visualization with Plotly

See the `plotly` skill for interactive 3D network visualization.

---

## Integration with Gauge-Transformer

The project already imports networkx in `transformer/analysis/rg_metrics.py`. Key integration points:

- **Meta-agent detection**: Use community detection to identify groups of tokens that attend to each other
- **RG flow analysis**: Track how graph structure changes across layers (coarse-graining)
- **Attention head specialization**: Classify heads by their graph-theoretic properties
- **Emergent syntax**: Detect if attention communities correspond to syntactic constituents

---

## Best Practices

1. **Threshold attention weights** before building graphs — raw attention matrices are dense
2. **Use weighted edges** to preserve attention magnitude information
3. **Symmetrize cautiously** — attention is directional; symmetrizing loses information
4. **Compare across layers** to see how graph structure evolves through the transformer
5. **Use spectral gap** (ratio of 2nd to 1st eigenvalue) as a measure of graph clustering strength
6. **Cache graph construction** when exploring multiple metrics on the same attention pattern

---

## Dependencies

```
pip install networkx scikit-learn scipy
```
