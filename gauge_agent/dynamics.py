"""
Layer 6: Natural Gradient Dynamics on the Product Manifold
===========================================================

Three timescales (adiabatic approximation):

  FAST — belief fiber (inference within a lifetime):
    dμ_q/dt  = -η_μ  Σ_q ∇_{μ_q} S        [perception]
    dΣ_q/dt  = -η_Σ  Σ_q (∇_Σ S) Σ_q      [uncertainty]
    dμ_p/dt  = -η_p  Σ_p ∇_{μ_p} S         [expectation learning]
    dΩ_i/dt  = -η_Ω  ∇_Ω S                 [reference frame]

  SLOW — model fiber (evolution across generations, ε << 1):
    dμ_s/dt  = -ε·η_s Σ_s ∇_{μ_s} S       [model evolution]
    dΣ_s/dt  = -ε·η_Σs Σ_s (∇ S) Σ_s      [model uncertainty]
    dμ_r/dt  = -ε·η_r Σ_r ∇_{μ_r} S       [model prior drift]
    dΩ̃_i/dt  = -ε·η_Ω̃  ∇_{Ω̃} S            [model gauge frame]

The model fiber IS the genome (metaphorically). It changes via
selection pressure, not within-lifetime learning. The ratio ε
controls the timescale separation:
  ε = 1.0  → model and belief evolve at same rate (no separation)
  ε = 0.01 → model evolves 100x slower (biological timescale)
  ε = 0.0  → model is frozen (pure inference, no evolution)

Reference: Dennis (2026), Sections 2.12, 6.2
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, List, Callable

from gauge_agent.agents import MultiAgentSystem
from gauge_agent.free_energy import FreeEnergyFunctional
from gauge_agent.statistical_manifold import ensure_spd


class NaturalGradientDynamics(nn.Module):
    """Natural gradient descent on the product manifold.

    TWO fibers with timescale separation:

    FAST (belief fiber — inference):
      dμ_q/dt = -lr_mu_q · Σ_q ∇S     dΩ/dt = -lr_omega · ∇S

    SLOW (model fiber — evolution, scaled by ε = model_lr_ratio):
      dμ_s/dt = -ε·lr_mu_s · Σ_s ∇S   dΩ̃/dt = -ε·lr_omega_model · ∇S

    The model fiber IS the genome. ε << 1 means models evolve slowly,
    like biological evolution. ε = 0 freezes the model (pure inference).

    Args:
        system: MultiAgentSystem to evolve
        free_energy: FreeEnergyFunctional
        lr_mu_q: learning rate for belief means (fast)
        lr_sigma_q: learning rate for belief covariances (fast)
        lr_mu_p: learning rate for prior means (medium)
        lr_sigma_p: learning rate for prior covariances (medium)
        lr_omega: learning rate for belief gauge frames
        model_lr_ratio: ε — ratio of model to belief learning rate.
                        0.01 = model evolves 100x slower (default).
                        0.0 = model frozen (pure inference).
                        1.0 = no timescale separation.
        lr_mu_s: base learning rate for model means (multiplied by ε)
        lr_sigma_s: base learning rate for model covariances (×ε)
        lr_mu_r: base learning rate for model prior means (×ε)
        lr_sigma_r: base learning rate for model prior covariances (×ε)
        lr_omega_model: base learning rate for model gauge frames (×ε)
        use_natural_gradient: if True, apply Fisher metric correction
        sigma_min: minimum covariance eigenvalue (prevents collapse)
    """

    def __init__(self,
                 system: MultiAgentSystem,
                 free_energy: FreeEnergyFunctional,
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
                 use_natural_gradient: bool = True,
                 sigma_min: float = 1e-4):
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

        self.use_natural_gradient = use_natural_gradient
        self.sigma_min = sigma_min

    def step(self, observations: Optional[Tensor] = None,
             obs_precision: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """One step of natural gradient descent.

        1. Compute free energy via forward pass
        2. Backprop to get Euclidean gradients
        3. Apply Fisher metric inverse (natural gradient)
        4. Update parameters with appropriate learning rates
        5. Project covariances back to SPD

        Args:
            observations: optional observation data
            obs_precision: optional observation precision
        Returns:
            Dict with energy components and gradient norms
        """
        # Forward: compute free energy
        result = self.free_energy(self.system, observations, obs_precision)
        total = result['total']

        # Backward: compute Euclidean gradients
        total.backward()

        info = {k: v.item() if isinstance(v, Tensor) and v.dim() == 0 else v
                for k, v in result.items() if k not in ('beta', 'gamma', 'E_belief_pairwise', 'E_prior_pairwise')}
        info['grad_norms'] = {}

        ε = self.model_lr_ratio

        # Update each agent — BOTH fibers
        with torch.no_grad():
            for agent in self.system.agents:
                # ── FAST: Belief fiber (inference) ──

                # μ_q: dμ_q = -η Σ_q ∇S
                if agent.mu_q.grad is not None:
                    grad = agent.mu_q.grad.clamp(-10.0, 10.0)
                    if self.use_natural_gradient:
                        sigma = agent.sigma_q.detach()
                        nat_grad = (sigma @ grad.unsqueeze(-1)).squeeze(-1)
                    else:
                        nat_grad = grad
                    agent.mu_q.data -= self.lr_mu_q * nat_grad
                    info['grad_norms']['mu_q'] = grad.norm().item()

                # L_q (Cholesky of Σ_q)
                if agent._L_q.grad is not None:
                    grad = agent._L_q.grad.clamp(-10.0, 10.0)
                    agent._L_q.data -= self.lr_sigma_q * grad
                    info['grad_norms']['L_q'] = grad.norm().item()

                # μ_p: dμ_p = -η_p Σ_p ∇S
                if agent.mu_p.grad is not None:
                    grad = agent.mu_p.grad.clamp(-10.0, 10.0)
                    if self.use_natural_gradient:
                        sigma_p = agent.sigma_p.detach()
                        nat_grad = (sigma_p @ grad.unsqueeze(-1)).squeeze(-1)
                    else:
                        nat_grad = grad
                    agent.mu_p.data -= self.lr_mu_p * nat_grad
                    info['grad_norms']['mu_p'] = grad.norm().item()

                # L_p (Cholesky of Σ_p)
                if agent._L_p.grad is not None:
                    grad = agent._L_p.grad.clamp(-10.0, 10.0)
                    agent._L_p.data -= self.lr_sigma_p * grad
                    info['grad_norms']['L_p'] = grad.norm().item()

                # Ω: dΩ = -η_Ω ∇S
                if agent.omega.grad is not None:
                    grad = agent.omega.grad.clamp(-10.0, 10.0)
                    agent.omega.data -= self.lr_omega * grad
                    info['grad_norms']['omega'] = grad.norm().item()

                # ── SLOW: Model fiber (evolution, ×ε) ──

                if ε > 0:
                    # μ_s: dμ_s = -ε·η_s Σ_s ∇S
                    if agent.mu_s.grad is not None:
                        grad = agent.mu_s.grad.clamp(-10.0, 10.0)
                        if self.use_natural_gradient:
                            sigma_s = agent.sigma_s.detach()
                            nat_grad = (sigma_s @ grad.unsqueeze(-1)).squeeze(-1)
                        else:
                            nat_grad = grad
                        agent.mu_s.data -= ε * self.lr_mu_s * nat_grad
                        info['grad_norms']['mu_s'] = grad.norm().item()

                    # L_s (Cholesky of Σ_s)
                    if agent._L_s.grad is not None:
                        grad = agent._L_s.grad.clamp(-10.0, 10.0)
                        agent._L_s.data -= ε * self.lr_sigma_s * grad
                        info['grad_norms']['L_s'] = grad.norm().item()

                    # μ_r: dμ_r = -ε·η_r Σ_r ∇S
                    if agent.mu_r.grad is not None:
                        grad = agent.mu_r.grad.clamp(-10.0, 10.0)
                        if self.use_natural_gradient:
                            sigma_r = agent.sigma_r.detach()
                            nat_grad = (sigma_r @ grad.unsqueeze(-1)).squeeze(-1)
                        else:
                            nat_grad = grad
                        agent.mu_r.data -= ε * self.lr_mu_r * nat_grad
                        info['grad_norms']['mu_r'] = grad.norm().item()

                    # L_r (Cholesky of Σ_r)
                    if agent._L_r.grad is not None:
                        grad = agent._L_r.grad.clamp(-10.0, 10.0)
                        agent._L_r.data -= ε * self.lr_sigma_r * grad
                        info['grad_norms']['L_r'] = grad.norm().item()

                    # Ω̃: dΩ̃ = -ε·η_Ω̃ ∇S
                    if agent.omega_model.grad is not None:
                        grad = agent.omega_model.grad.clamp(-10.0, 10.0)
                        agent.omega_model.data -= ε * self.lr_omega_model * grad
                        info['grad_norms']['omega_model'] = grad.norm().item()

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
        free_energy: FreeEnergyFunctional
        dt: integration timestep
        damping: friction coefficient (0 = conservative, >0 = dissipative)
    """

    def __init__(self, system: MultiAgentSystem,
                 free_energy: FreeEnergyFunctional,
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
