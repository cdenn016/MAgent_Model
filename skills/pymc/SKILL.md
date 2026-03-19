---
name: pymc
description: Bayesian modeling with PyMC for probabilistic inference. Use when validating variational approximation quality, comparing VFE posteriors to true posteriors, Bayesian hyperparameter sensitivity analysis, or building hierarchical models. For frequentist statistics use statistical-analysis.
license: Apache-2.0
metadata:
    skill-author: K-Dense Inc.
---

# PyMC — Bayesian Modeling and Probabilistic Inference

## Overview

PyMC is a probabilistic programming library for Bayesian statistical modeling and inference. This skill provides guidance for validating the Gauge-Transformer's variational approximations, comparing VFE posteriors to ground-truth MCMC posteriors, Bayesian hyperparameter analysis, and building hierarchical models relevant to the gauge-theoretic framework.

## When to Use This Skill

Use this skill when:
- Validating the quality of variational approximations (VFE vs. true posterior)
- Comparing the Gauge-Transformer's learned beliefs to MCMC ground truth
- Performing Bayesian hyperparameter sensitivity analysis
- Building hierarchical Bayesian models for multi-level VFE structure
- Computing posterior predictive checks on model outputs
- Estimating uncertainty in model comparisons (Gauge-Transformer vs. baselines)

---

## Core Capabilities

### 1. Validating Variational Approximation Quality

The Gauge-Transformer uses variational inference (VFE minimization). PyMC can validate whether the variational posterior is a good approximation.

```python
import pymc as pm
import arviz as az
import numpy as np
import torch

def validate_vfe_approximation(observed_data, mu_q, sigma_q, prior_mu, prior_sigma):
    """Compare the Gauge-Transformer's variational posterior to MCMC ground truth.

    Args:
        observed_data: numpy array of observed values
        mu_q: variational mean from the Gauge-Transformer
        sigma_q: variational std from the Gauge-Transformer
        prior_mu: prior mean used in VFE
        prior_sigma: prior std used in VFE
    """
    # Build the same generative model that VFE approximates
    with pm.Model() as true_model:
        # Prior (matching the Gauge-Transformer's prior)
        theta = pm.Normal('theta', mu=prior_mu, sigma=prior_sigma)

        # Likelihood
        y = pm.Normal('y', mu=theta, sigma=1.0, observed=observed_data)

        # MCMC ground truth
        trace = pm.sample(4000, tune=2000, return_inferencedata=True, random_seed=42)

    # Compare posteriors
    mcmc_mean = float(trace.posterior['theta'].mean())
    mcmc_std = float(trace.posterior['theta'].std())

    print(f"MCMC posterior:  mean={mcmc_mean:.4f}, std={mcmc_std:.4f}")
    print(f"VFE posterior:   mean={mu_q:.4f}, std={sigma_q:.4f}")
    print(f"Mean difference: {abs(mcmc_mean - mu_q):.4f}")
    print(f"Std ratio:       {sigma_q / mcmc_std:.4f}")

    # KL divergence between variational and true posterior
    kl = np.log(mcmc_std / sigma_q) + (sigma_q**2 + (mu_q - mcmc_mean)**2) / (2 * mcmc_std**2) - 0.5
    print(f"KL(q || p_mcmc): {kl:.4f}")

    return trace, {'mcmc_mean': mcmc_mean, 'mcmc_std': mcmc_std, 'kl': kl}
```

### 2. Multivariate Gaussian Belief Validation

```python
def validate_multivariate_belief(observed, mu_q, cov_q, prior_mu, prior_cov):
    """Validate multivariate Gaussian belief against MCMC.

    For small-scale problems where the Gauge-Transformer learns
    multivariate beliefs q(z) = N(mu_q, Sigma_q).
    """
    K = len(mu_q)

    with pm.Model() as model:
        # Multivariate prior
        theta = pm.MvNormal('theta', mu=prior_mu, cov=prior_cov, shape=K)

        # Likelihood
        y = pm.MvNormal('y', mu=theta, cov=np.eye(K), observed=observed)

        # Sample
        trace = pm.sample(4000, tune=2000, return_inferencedata=True, random_seed=42)

    # Extract MCMC posterior statistics
    mcmc_means = trace.posterior['theta'].mean(dim=('chain', 'draw')).values
    mcmc_cov = np.cov(trace.posterior['theta'].values.reshape(-1, K).T)

    # Compare
    print("MCMC posterior mean:", mcmc_means)
    print("VFE posterior mean: ", mu_q)
    print("Mean L2 error:      ", np.linalg.norm(mcmc_means - mu_q))

    return trace, mcmc_means, mcmc_cov
```

### 3. Bayesian Hyperparameter Sensitivity Analysis

```python
def hyperparameter_sensitivity(perplexity_data, hyperparams):
    """Bayesian analysis of how hyperparameters affect perplexity.

    Args:
        perplexity_data: dict mapping hyperparameter configs to perplexity measurements
        hyperparams: list of hyperparameter names
    """
    import pandas as pd

    # Build dataframe
    rows = []
    for config, perplexities in perplexity_data.items():
        for ppl in perplexities:
            row = dict(zip(hyperparams, config))
            row['perplexity'] = ppl
            rows.append(row)
    df = pd.DataFrame(rows)

    with pm.Model() as sensitivity_model:
        # Intercept
        alpha = pm.Normal('alpha', mu=df['perplexity'].mean(), sigma=20)

        # Hyperparameter effects
        betas = {}
        for hp in hyperparams:
            values = df[hp].values.astype(float)
            betas[hp] = pm.Normal(f'beta_{hp}', mu=0, sigma=10)

        # Linear predictor
        mu = alpha
        for hp in hyperparams:
            mu = mu + betas[hp] * df[hp].values.astype(float)

        # Noise
        sigma = pm.HalfNormal('sigma', sigma=10)

        # Likelihood
        y = pm.Normal('perplexity', mu=mu, sigma=sigma, observed=df['perplexity'].values)

        # Sample
        trace = pm.sample(2000, tune=1000, return_inferencedata=True, random_seed=42)

    # Summarize effects
    summary = az.summary(trace, var_names=[f'beta_{hp}' for hp in hyperparams])
    print(summary)

    # Forest plot of effects
    az.plot_forest(trace, var_names=[f'beta_{hp}' for hp in hyperparams],
                   combined=True, figsize=(8, 4))

    return trace, summary
```

### 4. Hierarchical Model for Multi-Head Attention

```python
def hierarchical_attention_model(attention_entropies):
    """Hierarchical Bayesian model for attention head behavior.

    Models per-head entropy as drawn from a group-level distribution,
    mirroring the hierarchical structure of VFE (h -> s -> p -> q).

    Args:
        attention_entropies: (n_layers, n_heads, n_samples) array
    """
    n_layers, n_heads, n_samples = attention_entropies.shape

    with pm.Model() as hierarchical_model:
        # Hyperpriors (group-level)
        mu_global = pm.Normal('mu_global', mu=0, sigma=5)
        sigma_global = pm.HalfNormal('sigma_global', sigma=2)

        # Layer-level
        mu_layer = pm.Normal('mu_layer', mu=mu_global, sigma=sigma_global, shape=n_layers)
        sigma_layer = pm.HalfNormal('sigma_layer', sigma=1, shape=n_layers)

        # Head-level
        mu_head = pm.Normal('mu_head',
                           mu=mu_layer[:, None],
                           sigma=sigma_layer[:, None],
                           shape=(n_layers, n_heads))

        # Observation noise
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=1)

        # Likelihood
        for l in range(n_layers):
            for h in range(n_heads):
                pm.Normal(f'entropy_L{l}_H{h}',
                         mu=mu_head[l, h],
                         sigma=sigma_obs,
                         observed=attention_entropies[l, h])

        trace = pm.sample(2000, tune=1000, return_inferencedata=True, random_seed=42)

    return trace
```

### 5. Model Comparison (Gauge-Transformer vs. Baselines)

```python
def bayesian_model_comparison(results_gauge, results_standard, metric='perplexity'):
    """Bayesian comparison of Gauge-Transformer vs standard transformer.

    Args:
        results_gauge: array of metric values from Gauge-Transformer runs
        results_standard: array of metric values from standard transformer runs
    """
    with pm.Model() as comparison:
        # Priors for each model's performance
        mu_gauge = pm.Normal('mu_gauge', mu=np.mean(results_gauge), sigma=20)
        mu_standard = pm.Normal('mu_standard', mu=np.mean(results_standard), sigma=20)

        sigma_gauge = pm.HalfNormal('sigma_gauge', sigma=10)
        sigma_standard = pm.HalfNormal('sigma_standard', sigma=10)

        # Likelihoods
        pm.Normal('y_gauge', mu=mu_gauge, sigma=sigma_gauge, observed=results_gauge)
        pm.Normal('y_standard', mu=mu_standard, sigma=sigma_standard, observed=results_standard)

        # Derived: difference and effect size
        diff = pm.Deterministic('difference', mu_gauge - mu_standard)
        pooled_sigma = pm.math.sqrt((sigma_gauge**2 + sigma_standard**2) / 2)
        effect_size = pm.Deterministic('effect_size', diff / pooled_sigma)

        trace = pm.sample(4000, tune=2000, return_inferencedata=True, random_seed=42)

    # Summarize
    print(az.summary(trace, var_names=['difference', 'effect_size']))

    # Probability that gauge model is better (lower perplexity)
    prob_better = float(np.mean(trace.posterior['difference'].values < 0))
    print(f"\nP(Gauge-Transformer has lower {metric}) = {prob_better:.3f}")

    # Plot
    az.plot_posterior(trace, var_names=['difference'], ref_val=0)

    return trace, prob_better
```

---

## Diagnostics and Convergence

### Essential Convergence Checks

```python
def check_convergence(trace, var_names=None):
    """Run standard MCMC convergence diagnostics."""
    summary = az.summary(trace, var_names=var_names)

    # Check R-hat (should be < 1.01)
    rhat_issues = summary[summary['r_hat'] > 1.01]
    if len(rhat_issues) > 0:
        print("WARNING: R-hat > 1.01 for:")
        print(rhat_issues[['r_hat']])

    # Check ESS (should be > 400)
    ess_issues = summary[summary['ess_bulk'] < 400]
    if len(ess_issues) > 0:
        print("WARNING: Low ESS for:")
        print(ess_issues[['ess_bulk', 'ess_tail']])

    # Trace plots
    az.plot_trace(trace, var_names=var_names)

    # Rank plots (better than trace plots for convergence)
    az.plot_rank(trace, var_names=var_names)

    return summary
```

### Posterior Predictive Checks

```python
def posterior_predictive_check(trace, model):
    """Generate and visualize posterior predictive samples."""
    with model:
        ppc = pm.sample_posterior_predictive(trace, random_seed=42)

    az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=model))
    return ppc
```

---

## ArviZ Integration

PyMC integrates with ArviZ for visualization and diagnostics:

```python
import arviz as az

# Summary statistics
az.summary(trace)

# Posterior plots
az.plot_posterior(trace, var_names=['theta'])

# Forest plots (compare parameters)
az.plot_forest(trace, var_names=['mu_layer'])

# Pair plots (check correlations between parameters)
az.plot_pair(trace, var_names=['mu_gauge', 'mu_standard'])

# LOO cross-validation (model comparison)
az.loo(trace)

# WAIC
az.waic(trace)
```

---

## Best Practices

1. **Always check convergence** — R-hat < 1.01, ESS > 400
2. **Use weakly informative priors** — not flat, but not too narrow
3. **Run posterior predictive checks** — verify the model generates plausible data
4. **Report full posterior summaries** — mean, SD, credible intervals
5. **Use ArviZ for all visualization** — consistent, publication-quality plots
6. **Start simple** — begin with simple models, add complexity incrementally
7. **Compare models with LOO** — use `az.compare()` for formal model comparison
8. **Set random seeds** for reproducibility
9. **Scale data** if parameters are on very different scales
10. **Use `pm.find_MAP()`** as a sanity check before full MCMC

---

## Relevance to the Gauge-Transformer

The Gauge-Transformer's core framework is Bayesian:
- **Beliefs** q(z) = N(μ_q, Σ_q) are variational posteriors
- **VFE** = KL[q || p] - E_q[log p(x|z)] is the variational objective
- **Hierarchical structure** (h → s → p → q) maps to hierarchical Bayesian models
- **Gauge invariance** of KL divergence has direct Bayesian interpretation

PyMC provides the gold-standard MCMC inference to validate that the Gauge-Transformer's fast variational inference produces accurate posteriors, especially on small-scale problems where full MCMC is tractable.

---

## Dependencies

```
pip install pymc arviz
```
