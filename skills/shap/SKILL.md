---
name: shap
description: Model interpretability with SHAP for feature attribution beyond attention weights. Use when explaining what drives VFE component predictions, attributing contributions of gauge frames, beliefs, or input features to model outputs. For attention graph analysis use networkx; for Bayesian uncertainty use pymc.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# SHAP — Model Interpretability and Feature Attribution

## Overview

SHAP (SHapley Additive exPlanations) provides theoretically grounded feature attribution based on cooperative game theory. This skill guides the use of SHAP for understanding what drives the Gauge-Transformer's predictions beyond raw attention weights — attributing contributions from VFE components, gauge frames, belief parameters, and input features.

## When to Use This Skill

Use this skill when:
- Explaining which VFE components (KL term, likelihood, prior) drive specific predictions
- Attributing model output to input token features vs. learned gauge frames
- Understanding why the model assigns particular attention patterns
- Comparing feature importance between the Gauge-Transformer and standard baselines
- Diagnosing unexpected model behavior at the component level
- Creating interpretability figures for the manuscript

---

## Core Capabilities

### 1. SHAP for Transformer Outputs

```python
import shap
import torch
import numpy as np

def explain_transformer_predictions(model, input_ids, tokenizer, device='cpu'):
    """SHAP explanations for Gauge-Transformer language model predictions.

    Uses partition explainer suited for sequential/text data.
    """
    model.eval()

    def predict_fn(inputs):
        """Wrapper that takes token strings and returns next-token log-probs."""
        encoded = tokenizer(inputs.tolist(), return_tensors='pt',
                           padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = model(encoded['input_ids']).logits
        # Return log-probs for the last position
        return torch.log_softmax(logits[:, -1, :], dim=-1).cpu().numpy()

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_fn, masker)

    text = tokenizer.decode(input_ids[0])
    shap_values = explainer([text])

    return shap_values
```

### 2. VFE Component Attribution

```python
def attribute_vfe_components(model, input_data, component_names=None):
    """Attribute model output to individual VFE components.

    The Gauge-Transformer's loss decomposes as:
        VFE = KL[q||p] - E_q[log p(x|z)]

    This function explains how each sub-component contributes to predictions.
    """
    if component_names is None:
        component_names = ['kl_divergence', 'log_likelihood', 'prior_term', 'gauge_transport']

    def component_predict_fn(component_values):
        """Forward pass with ablated/modified components."""
        # component_values: (n_samples, n_components)
        outputs = []
        for cv in component_values:
            with torch.no_grad():
                out = model.forward_with_components(input_data, component_weights=cv)
            outputs.append(out.cpu().numpy())
        return np.array(outputs)

    # Background: nominal component values
    background = np.ones((1, len(component_names)))

    explainer = shap.KernelExplainer(component_predict_fn, background)
    shap_values = explainer.shap_values(np.ones((1, len(component_names))))

    return shap_values, component_names
```

### 3. Gauge Frame Feature Attribution

```python
def explain_gauge_frame_contribution(model, input_ids, n_background=50):
    """Explain how learned gauge frame parameters affect predictions.

    Attributes prediction changes to specific gauge frame components
    (e.g., which GL(K) generators matter most for a given input).
    """
    def gauge_frame_predict(frame_params):
        """Predict with modified gauge frame parameters."""
        with torch.no_grad():
            output = model.forward_with_frames(input_ids, frame_override=frame_params)
        return output.cpu().numpy()

    # Sample background frames from training distribution
    background_frames = model.sample_gauge_frames(n_background).numpy()

    explainer = shap.KernelExplainer(gauge_frame_predict, background_frames)

    # Explain the current learned frames
    current_frames = model.get_gauge_frames().detach().numpy()
    shap_values = explainer.shap_values(current_frames)

    return shap_values
```

### 4. Layer-wise Attribution

```python
def layerwise_shap_analysis(model, input_ids, target_token_idx=-1):
    """Compute SHAP values at each transformer layer.

    Shows how feature importance evolves through the network,
    complementing RG flow analysis.
    """
    layer_attributions = {}

    for layer_idx in range(model.n_layers):
        def layer_predict(x):
            """Predict using hidden states at a specific layer."""
            with torch.no_grad():
                hidden = model.get_hidden_state(input_ids, layer=layer_idx)
                # Modify hidden state
                modified = torch.tensor(x, dtype=hidden.dtype, device=hidden.device)
                output = model.forward_from_layer(modified, start_layer=layer_idx)
            return output[:, target_token_idx, :].cpu().numpy()

        background = model.get_hidden_state(input_ids, layer=layer_idx).detach().cpu().numpy()
        explainer = shap.KernelExplainer(layer_predict, background[:10])
        shap_values = explainer.shap_values(background)
        layer_attributions[layer_idx] = shap_values

    return layer_attributions
```

### 5. Comparing Gauge-Transformer vs Standard Transformer

```python
def compare_attributions(gauge_model, standard_model, input_ids, tokenizer):
    """Side-by-side SHAP comparison of two models on the same input."""
    import matplotlib.pyplot as plt

    # Get SHAP values for both
    shap_gauge = explain_transformer_predictions(gauge_model, input_ids, tokenizer)
    shap_standard = explain_transformer_predictions(standard_model, input_ids, tokenizer)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    shap.plots.text(shap_gauge[0], ax=axes[0])
    axes[0].set_title('Gauge-Transformer Attribution')

    shap.plots.text(shap_standard[0], ax=axes[1])
    axes[1].set_title('Standard Transformer Attribution')

    plt.tight_layout()
    return fig
```

---

## Visualization

### Summary Plot

```python
# Global feature importance across many inputs
shap.summary_plot(shap_values, feature_names=feature_names)
```

### Force Plot (Single Prediction)

```python
# Explain a single prediction
shap.force_plot(explainer.expected_value, shap_values[0], feature_names=feature_names)
```

### Dependence Plot

```python
# How a single feature affects predictions, colored by interaction
shap.dependence_plot('kl_divergence', shap_values, features, feature_names=feature_names)
```

### Heatmap (Sequence-Level)

```python
# Token-level attribution heatmap
shap.plots.heatmap(shap_values[:20])  # First 20 samples
```

### Bar Plot (Mean Absolute SHAP)

```python
shap.plots.bar(shap_values, max_display=15)
```

---

## Explainer Selection Guide

| Explainer | Best For | Speed | Fidelity |
|-----------|----------|-------|----------|
| `shap.KernelExplainer` | Any model (black-box) | Slow | High |
| `shap.DeepExplainer` | PyTorch/TF deep models | Fast | Good |
| `shap.GradientExplainer` | Neural networks | Fast | Good |
| `shap.Explainer` (auto) | Auto-selects best | Varies | Varies |
| `shap.PartitionExplainer` | Text/hierarchical | Medium | High |

For the Gauge-Transformer:
- **KernelExplainer** for component-level attribution (small input space)
- **DeepExplainer** or **GradientExplainer** for token-level attribution (large input)
- **PartitionExplainer** for text inputs with a masker

---

## Computational Considerations

SHAP can be expensive for large models. Strategies:

```python
# 1. Subsample background data
background = shap.sample(full_data, 100)

# 2. Use GradientExplainer for speed (gradient-based approximation)
explainer = shap.GradientExplainer(model, background_tensor)

# 3. Limit to specific output classes/tokens
shap_values = explainer.shap_values(inputs, nsamples=200)  # Reduce samples for Kernel

# 4. Focus on specific layers or components rather than full model
```

---

## Integration with Gauge-Transformer Analysis

SHAP complements existing analysis tools:

| Analysis | Tool | What It Shows |
|----------|------|--------------|
| Attention patterns | Attention weights | Where the model looks |
| Graph structure | networkx | Relational structure of attention |
| Feature importance | **SHAP** | **What drives predictions** |
| Representation structure | umap-learn | How representations cluster |
| Posterior quality | pymc | Whether beliefs are calibrated |
| Community structure | networkx | Emergent token groupings |

SHAP answers "why did the model predict X?" while attention shows "where did the model look?"

---

## Best Practices

1. **Use appropriate background data** — sample from the training distribution
2. **Check additivity** — SHAP values should sum to (prediction - expected value)
3. **Report confidence** — run with multiple background samples to check stability
4. **Don't over-interpret small values** — focus on top-k features
5. **Combine with attention analysis** — SHAP and attention tell different stories
6. **Use DeepExplainer for speed** when KernelExplainer is too slow
7. **Visualize at multiple granularities** — token-level, component-level, layer-level

---

## Dependencies

```
pip install shap matplotlib
```
