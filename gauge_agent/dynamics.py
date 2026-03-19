"""
Layer 6: Natural Gradient Dynamics on the Product Manifold
===========================================================

Three timescales (adiabatic approximation):

  FAST — belief fiber (inference within a lifetime):
    dμ_q/dt  = -η_μ  Σ_q ∇_{μ_q} S        [perception]
    dΣ_q/dt  via autograd through L_q       [uncertainty — Eq. 36]
    dμ_p/dt  = -η_p  Σ_p ∇_{μ_p} S         [expectation learning]
    dΩ_i/dt  = -η_Ω  ∇_Ω S                 [gauge frame — GROUP gradient]

  SLOW — model fiber (evolution across generations, ε << 1):
    dμ_s/dt  = -ε·η_s Σ_s ∇_{μ_s} S       [model evolution]
    dΣ_s/dt  via autograd through L_s       [model uncertainty]
    dμ_r/dt  = -ε·η_r Σ_r ∇_{μ_r} S       [model prior drift]
    dΩ̃_i/dt  = -ε·η_Ω̃  ∇_{Ω̃} S            [model gauge frame]
    db₀_i/dt = -ε·η_b ∇_{b₀} S             [precision sensitivity]
    dc₀_i/dt = -ε·η_c ∇_{c₀} S             [precision strength]

Covariance dynamics (Eq. 36):

  The covariance gradient ∂S/∂Σ_i includes the α-dependent terms:
    -(1+α_i)Σ_i⁻¹ + α_i Σ_{p,i}⁻¹ + Σ_j β_ij (Ω_ij Σ_j Ω_ij^T)⁻¹
    + (∂α_i/∂Σ_i) KL(q_i || p_i)     [gate gradient]

  Since α_i = c₀/(b₀ + KL), and KL depends on Σ_i through trace
  and log-det terms, ∂α_i/∂Σ_i ≠ 0. Autograd handles this
  automatically via the Cholesky parameterization L_q.

Gauge frame gradients:

  The manuscript uses the Lie algebra parameterization:
    Ω_i = exp(φ_i),  φ_i ∈ 𝔤𝔩(K)

  This is elegant for theory but computationally we use DIRECT
  GROUP GRADIENTS via autograd:
    dΩ_i/dt = -η_Ω ∂S/∂Ω_i

  This is:
    - CHEAPER: no exp/log maps needed
    - MORE GENERAL: works for all of GL(K), not just identity component
    - SIMPLER: autograd computes ∂S/∂Ω_ij directly

  The only care: keep Ω invertible. For small step sizes this is
  automatic. For safety we can check det(Ω) periodically.

  The Lie algebra φ_i is the agent's "internal frame of reference" —
  how it orients its beliefs and models. The gauge frame Ω_i = exp(φ_i)
  is the transport operator built from that reference. Both views
  are equivalent; we use Ω directly for computational efficiency.

Reference: Dennis (2026), Sections 2.12, 6.2
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, List, Callable, Union

from gauge_agent.agents import MultiAgentSystem
from gauge_agent.statistical_manifold import ensure_spd


class NaturalGradientDynamics(nn.Module):
    """Natural gradient descent on the product manifold.

    Drives ALL parameters of the system via the FullVFE:
      - Belief fiber: μ_q, L_q (→Σ_q), μ_p, L_p, Ω
      - Model fiber (×ε): μ_s, L_s, μ_r, L_r, Ω̃, b₀, c₀

    Accepts either FullVFE or FreeEnergyFunctional as the free energy.

    The covariance dynamics (Eq. 36) are handled automatically:
    autograd through L_q picks up all α-dependent terms including
    the gate gradient ∂α/∂Σ · KL.

    Gauge frame Ω_i updated via direct group gradient (not Lie algebra).

    Args:
        system: MultiAgentSystem to evolve
        free_energy: FullVFE or FreeEnergyFunctional (anything callable)
        lr_mu_q: learning rate for belief means (fast)
        lr_sigma_q: learning rate for belief covariances (fast)
        lr_mu_p: learning rate for prior means (medium)
        lr_sigma_p: learning rate for prior covariances (medium)
        lr_omega: learning rate for gauge frames (group gradient)
        model_lr_ratio: ε — timescale separation
        lr_mu_s, lr_sigma_s, lr_mu_r, lr_sigma_r: model fiber rates (×ε)
        lr_omega_model: model gauge frame rate (×ε)
        lr_precision: learning rate for b₀, c₀ (×ε)
        use_natural_gradient: if True, apply Fisher metric correction
        grad_clip: gradient clipping value
    """

    def __init__(self,
                 system: MultiAgentSystem,
                 free_energy,
                 lr_mu_q: float = 0.05,
                 lr_sigma_q: float = 0.0075,
                 lr_mu_p: float = 0.02,
                 lr_sigma_p: float = 0.0075,
                 lr_omega: float = 0.01,
                 model_lr_ratio: float = 0.01,
                 lr_mu_s: float = 0.05,
                 lr_sigma_s: float = 0.0075,
                 lr_mu_r: float = 0.02,
                 lr_sigma_r: float = 0.0075,
                 lr_omega_model: float = 0.01,
                 lr_precision: float = 0.01,
                 use_natural_gradient: bool = True,
                 grad_clip: float = 10.0):
        super().__init__()
        self.system = system
        self.free_energy = free_energy

        # Belief fiber (fast)
        self.lr_mu_q = lr_mu_q
        self.lr_sigma_q = lr_sigma_q
        self.lr_mu_p = lr_mu_p
        self.lr_sigma_p = lr_sigma_p
        self.lr_omega = lr_omega

        # Model fiber (slow — all rates multiplied by ε)
        self.model_lr_ratio = model_lr_ratio  # ε
        self.lr_mu_s = lr_mu_s
        self.lr_sigma_s = lr_sigma_s
        self.lr_mu_r = lr_mu_r
        self.lr_sigma_r = lr_sigma_r
        self.lr_omega_model = lr_omega_model
        self.lr_precision = lr_precision  # for b₀, c₀

        self.use_natural_gradient = use_natural_gradient
        self.grad_clip = grad_clip

    def _update_param(self, param, lr: float, nat_grad_sigma=None):
        """Update a single parameter with optional natural gradient."""
        if param.grad is None:
            return 0.0
        grad = param.grad.clamp(-self.grad_clip, self.grad_clip)
        if nat_grad_sigma is not None and self.use_natural_gradient:
            nat_grad = (nat_grad_sigma @ grad.unsqueeze(-1)).squeeze(-1)
        else:
            nat_grad = grad
        param.data -= lr * nat_grad
        return grad.norm().item()

    def step(self, observations: Optional[Tensor] = None,
             obs_precision: Optional[Tensor] = None,
             **vfe_kwargs) -> Dict[str, Tensor]:
        """One step of natural gradient descent.

        Computes VFE, backprops, applies natural gradient updates
        to ALL parameters on BOTH fibers with timescale separation.

        The covariance gradient (Eq. 36) is computed automatically:
        autograd through L_q captures -(1+α)Σ⁻¹ + αΣ_p⁻¹ + attention
        + gate gradient ∂α/∂Σ·KL — all in one backward pass.

        Group gradients for Ω: autograd gives ∂S/∂Ω_ij directly.
        No exp/log maps needed. Cheaper and more general than
        Lie algebra parameterization.

        Args:
            observations: optional observation data
            obs_precision: optional observation precision
            **vfe_kwargs: extra args passed to free_energy
        Returns:
            Dict with energy components and gradient norms
        """
        # Forward: compute free energy
        result = self.free_energy(self.system, observations, obs_precision,
                                   **vfe_kwargs)
        total = result['total']

        # Backward: autograd computes ALL gradients including:
        #   - ∂S/∂L_q which encodes the full Eq. 36 covariance dynamics
        #   - ∂S/∂Ω which is the direct group gradient
        #   - ∂S/∂b₀, ∂S/∂c₀ through the adaptive precision
        total.backward()

        # Extract scalar info for logging
        info = {}
        for k, v in result.items():
            if k in ('beta', 'gamma', 'E_belief_pairwise', 'E_model_pairwise',
                      'alpha_belief', 'alpha_model'):
                continue
            if isinstance(v, Tensor) and v.dim() == 0:
                info[k] = v.item()
            elif not isinstance(v, Tensor):
                info[k] = v
        info['grad_norms'] = {}

        ε = self.model_lr_ratio

        # Update each agent — BOTH fibers
        with torch.no_grad():
            for agent in self.system.agents:
                norms = {}

                # ── FAST: Belief fiber (inference) ──

                # μ_q: dμ_q = -η Σ_q ∇S  (natural gradient)
                norms['mu_q'] = self._update_param(
                    agent.mu_q, self.lr_mu_q,
                    agent.sigma_q.detach() if self.use_natural_gradient else None
                )

                # L_q: autograd through Cholesky captures Eq. 36 covariance dynamics
                # ∂S/∂L_q encodes -(1+α)Σ⁻¹ + αΣ_p⁻¹ + β·transported_precision
                # + gate gradient ∂α/∂Σ·KL — all automatically
                norms['L_q'] = self._update_param(agent._L_q, self.lr_sigma_q)

                # μ_p: dμ_p = -η_p Σ_p ∇S
                norms['mu_p'] = self._update_param(
                    agent.mu_p, self.lr_mu_p,
                    agent.sigma_p.detach() if self.use_natural_gradient else None
                )

                # L_p
                norms['L_p'] = self._update_param(agent._L_p, self.lr_sigma_p)

                # Ω: GROUP GRADIENT — direct ∂S/∂Ω
                # No exp/log maps. Cheaper and more general than Lie algebra.
                # Ω stays invertible for small steps (det preserved approximately).
                norms['omega'] = self._update_param(agent.omega, self.lr_omega)

                # ── SLOW: Model fiber (evolution, ×ε) ──

                if ε > 0:
                    norms['mu_s'] = self._update_param(
                        agent.mu_s, ε * self.lr_mu_s,
                        agent.sigma_s.detach() if self.use_natural_gradient else None
                    )
                    norms['L_s'] = self._update_param(agent._L_s, ε * self.lr_sigma_s)
                    norms['mu_r'] = self._update_param(
                        agent.mu_r, ε * self.lr_mu_r,
                        agent.sigma_r.detach() if self.use_natural_gradient else None
                    )
                    norms['L_r'] = self._update_param(agent._L_r, ε * self.lr_sigma_r)

                    # Ω̃: model gauge frame — GROUP GRADIENT (×ε)
                    norms['omega_model'] = self._update_param(
                        agent.omega_model, ε * self.lr_omega_model
                    )

                    # b₀, c₀: precision hyperparameters (×ε)
                    # Gradients flow through α = c₀/(b₀ + KL)
                    norms['log_b0'] = self._update_param(
                        agent._log_b0, ε * self.lr_precision
                    )
                    norms['log_c0'] = self._update_param(
                        agent._log_c0, ε * self.lr_precision
                    )
                    norms['log_b0_model'] = self._update_param(
                        agent._log_b0_model, ε * self.lr_precision
                    )
                    norms['log_c0_model'] = self._update_param(
                        agent._log_c0_model, ε * self.lr_precision
                    )

                info['grad_norms'][agent.agent_id] = norms

        # Zero gradients
        self.system.zero_grad()

        return info

    def evolve(self, n_steps: int,
               observations: Optional[Tensor] = None,
               obs_precision: Optional[Tensor] = None,
               callback: Optional[Callable] = None,
               snapshot_interval: int = 1) -> List[Dict]:
        """Evolve the system for multiple steps.

        Args:
            n_steps: number of evolution steps
            observations: optional fixed observations
            obs_precision: optional observation precision
            callback: optional function called each step with (step, info)
            snapshot_interval: how often to record snapshots
        Returns:
            List of info dicts from each recorded step
        """
        history = []
        for t in range(n_steps):
            info = self.step(observations, obs_precision)
            info['step'] = t

            if t % snapshot_interval == 0:
                history.append(info)

            if callback is not None:
                callback(t, info)

        return history


class HamiltonianDynamics(nn.Module):
    """Second-order Hamiltonian dynamics (underdamped).

    H = T + V where:
      T = (1/2) μ̇^T M μ̇ + (1/4) tr(Σ⁻¹ Σ̇ Σ⁻¹ Σ̇)  [kinetic]
      V = S[q, p, Ω]                                      [potential = free energy]

    Mass matrix M = Σ_p⁻¹ (prior precision = inertial mass).

    Reference: Dennis (2026), Section 3 (Mass from Statistical Precision)

    Args:
        system: MultiAgentSystem
        free_energy: any callable VFE (FullVFE or FreeEnergyFunctional)
        dt: integration timestep
        damping: friction coefficient (0 = conservative, >0 = dissipative)
    """

    def __init__(self, system: MultiAgentSystem,
                 free_energy,
                 dt: float = 0.01,
                 damping: float = 0.0):
        super().__init__()
        self.system = system
        self.free_energy = free_energy
        self.dt = dt
        self.damping = damping

        # Initialize momenta (conjugate to mu_q)
        self.momenta = {}
        for agent in system.agents:
            self.momenta[agent.agent_id] = {
                'p_mu': torch.zeros_like(agent.mu_q.data),
            }

    def step(self, observations=None, obs_precision=None) -> Dict:
        """Symplectic (leapfrog) integration step.

        Preserves the symplectic structure of the Hamiltonian flow.

        Returns:
            Dict with kinetic energy, potential energy, total energy
        """
        dt = self.dt
        half_dt = 0.5 * dt

        # --- Half-step momentum update: p ← p - (dt/2) ∇V ---
        result = self.free_energy(self.system, observations, obs_precision)
        V = result['total']
        V.backward()

        with torch.no_grad():
            kinetic = torch.tensor(0.0)
            for agent in self.system.agents:
                mom = self.momenta[agent.agent_id]

                if agent.mu_q.grad is not None:
                    # Half kick
                    mom['p_mu'] -= half_dt * agent.mu_q.grad
                    # Apply damping
                    mom['p_mu'] *= (1.0 - self.damping * half_dt)

            self.system.zero_grad()

            # --- Full position update: q ← q + dt M⁻¹ p ---
            for agent in self.system.agents:
                mom = self.momenta[agent.agent_id]
                # Mass = Σ_p⁻¹, so M⁻¹ = Σ_p
                mass_inv = agent.sigma_p.detach()
                velocity = (mass_inv @ mom['p_mu'].unsqueeze(-1)).squeeze(-1)
                agent.mu_q.data += dt * velocity

                # Kinetic energy: (1/2) p^T M⁻¹ p
                kinetic = kinetic + 0.5 * (mom['p_mu'] * velocity).sum()

        # --- Second half-step momentum update ---
        result2 = self.free_energy(self.system, observations, obs_precision)
        V2 = result2['total']
        V2.backward()

        with torch.no_grad():
            for agent in self.system.agents:
                mom = self.momenta[agent.agent_id]
                if agent.mu_q.grad is not None:
                    mom['p_mu'] -= half_dt * agent.mu_q.grad
                    mom['p_mu'] *= (1.0 - self.damping * half_dt)

            self.system.zero_grad()

        return {
            'kinetic': kinetic.item(),
            'potential': V2.item(),
            'total_energy': kinetic.item() + V2.item(),
            **{k: v.item() if isinstance(v, Tensor) and v.dim() == 0 else v
               for k, v in result2.items()
               if k not in ('beta', 'gamma', 'E_belief_pairwise', 'E_prior_pairwise')}
        }
