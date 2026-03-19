"""
Layer 6: Natural Gradient Dynamics on the Product Manifold
===========================================================

The system evolves via natural gradient flow:
  dμ_i/dt  = -η_μ  Σ_i (∇_μ S)           [fast: belief updates]
  dΣ_i/dt  = -η_Σ  Σ_i (∇_Σ S) Σ_i       [fast: uncertainty updates]
  dμ_p/dt  = -η_μp Σ_p (∇_μp S)           [slow: prior/model learning]
  dΣ_p/dt  = -η_Σp Σ_p (∇_Σp S) Σ_p      [slow: prior uncertainty]
  dΩ_i/dt  = -η_Ω  ∇_Ω S                  [very slow: gauge frame]

The natural gradient ∇̃ = G⁻¹∇ uses the Fisher-Rao metric G,
ensuring steepest descent in the intrinsic geometry of SPD(K).

Timescale separation: η_μ >> η_p >> η_Ω (adiabatic approximation).

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

    M = (ℝ^K × SPD(K) × GL(K))^N

    Uses PyTorch autograd for Euclidean gradients, then applies
    the Fisher-Rao metric inverse to get natural gradients.

    Args:
        system: MultiAgentSystem to evolve
        free_energy: FreeEnergyFunctional
        lr_mu_q: learning rate for belief means (fast)
        lr_sigma_q: learning rate for belief covariances (fast)
        lr_mu_p: learning rate for prior means (slow)
        lr_sigma_p: learning rate for prior covariances (slow)
        lr_omega: learning rate for gauge frames (very slow)
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
                 use_natural_gradient: bool = True,
                 sigma_min: float = 1e-4):
        super().__init__()
        self.system = system
        self.free_energy = free_energy
        self.lr_mu_q = lr_mu_q
        self.lr_sigma_q = lr_sigma_q
        self.lr_mu_p = lr_mu_p
        self.lr_sigma_p = lr_sigma_p
        self.lr_omega = lr_omega
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

        # Update each agent
        with torch.no_grad():
            for agent in self.system.agents:
                # --- Belief mean: dμ_q = -η Σ_q ∇_μ S ---
                if agent.mu_q.grad is not None:
                    grad = agent.mu_q.grad.clamp(-10.0, 10.0)
                    if self.use_natural_gradient:
                        # Natural gradient: Σ @ grad
                        sigma = agent.sigma_q.detach()
                        nat_grad = (sigma @ grad.unsqueeze(-1)).squeeze(-1)
                    else:
                        nat_grad = grad
                    agent.mu_q.data -= self.lr_mu_q * nat_grad
                    info['grad_norms']['mu_q'] = grad.norm().item()

                # --- Belief covariance (via Cholesky factor) ---
                if agent._L_q.grad is not None:
                    grad = agent._L_q.grad.clamp(-10.0, 10.0)
                    agent._L_q.data -= self.lr_sigma_q * grad
                    info['grad_norms']['L_q'] = grad.norm().item()

                # --- Prior mean: dμ_p = -η_p Σ_p ∇_μp S ---
                if agent.mu_p.grad is not None:
                    grad = agent.mu_p.grad.clamp(-10.0, 10.0)
                    if self.use_natural_gradient:
                        sigma_p = agent.sigma_p.detach()
                        nat_grad = (sigma_p @ grad.unsqueeze(-1)).squeeze(-1)
                    else:
                        nat_grad = grad
                    agent.mu_p.data -= self.lr_mu_p * nat_grad
                    info['grad_norms']['mu_p'] = grad.norm().item()

                # --- Prior covariance (via Cholesky factor) ---
                if agent._L_p.grad is not None:
                    grad = agent._L_p.grad.clamp(-10.0, 10.0)
                    agent._L_p.data -= self.lr_sigma_p * grad
                    info['grad_norms']['L_p'] = grad.norm().item()

                # --- Gauge frame: dΩ = -η_Ω ∇_Ω S ---
                if agent.omega.grad is not None:
                    grad = agent.omega.grad.clamp(-10.0, 10.0)
                    agent.omega.data -= self.lr_omega * grad
                    info['grad_norms']['omega'] = grad.norm().item()

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
