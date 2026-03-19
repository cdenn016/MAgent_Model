"""
Layer 10: Mass Matrix, Kinetic Energy, and Hamiltonian Structure
==================================================================

The mass matrix arises as the Hessian of the free energy:
  M = ∂²F / ∂ξ∂ξ

where ξ = (μ₁,...,μ_N, Σ₁,...,Σ_N) is the full state vector.

Key result (Eq. 37): effective mass of agent i =

  M_i = Λ̄_p  +  Σ_k β_ik Λ̃_qk  +  Σ_j β_ji Λ_qi  +  Λ_oi
        ─────    ──────────────     ───────────────     ────
        bare     incoming           outgoing            sensory
        mass     relational mass    recoil mass         mass

Mass = precision: rocks are certain (high Λ), thus massive,
thus hard to move. Quantum particles are uncertain, thus light.

Reference: Dennis (2026), Section 3
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple

from gauge_agent.agents import MultiAgentSystem
from gauge_agent.free_energy import FreeEnergyFunctional
from gauge_agent.lie_groups import transport_covariance


class MassMatrix(nn.Module):
    """Computes the full mass matrix M = ∂²F/∂ξ² for the multi-agent system.

    Block structure:
        M = [ M_μμ    C^μΣ  ]
            [ (C^μΣ)ᵀ M_ΣΣ  ]

    Each block is itself N×N with K×K sub-blocks.

    Args:
        system: MultiAgentSystem
        free_energy: FreeEnergyFunctional
    """

    def __init__(self, system: MultiAgentSystem,
                 free_energy: FreeEnergyFunctional):
        super().__init__()
        self.system = system
        self.free_energy = free_energy

    def effective_mass_diagonal(self,
                                 beta: Optional[Tensor] = None,
                                 obs_precision: Optional[Tensor] = None
                                 ) -> Tensor:
        """Diagonal effective mass for each agent (Eq. 37).

        M_i = Λ̄_pi + Σ_k β_ik Λ̃_qk + Σ_j β_ji Λ_qi + Λ_oi

        Args:
            beta: (N, N) attention weights (computed if None)
            obs_precision: (K, K) or scalar observation precision
        Returns:
            (N, K, K) effective mass matrices
        """
        N = self.system.N_agents
        K = self.system.K

        if beta is None:
            result = self.free_energy(self.system)
            beta = result['beta']
            # Average over grid if needed
            while beta.dim() > 2:
                beta = beta.mean(-1)

        masses = []
        for i, agent_i in enumerate(self.system.agents):
            # Bare mass: prior precision Λ̄_pi
            M_i = agent_i.precision_p.detach().clone()

            # Incoming relational mass: Σ_k β_ik Λ̃_qk
            for k, agent_k in enumerate(self.system.agents):
                if k == i:
                    continue
                b = beta[i, k].item() if beta.dim() == 2 else beta[i, k].mean().item()
                # Transport k's precision to i's frame
                # Precision transforms contravariantly: Λ' = Ω^{-T} Λ Ω^{-1}
                omega_ik = agent_i.omega.data @ torch.linalg.inv(agent_k.omega.data)
                omega_ik_inv = torch.linalg.inv(omega_ik)
                Lambda_k_t = omega_ik_inv.transpose(-2, -1) @ agent_k.precision_q.detach() @ omega_ik_inv
                M_i = M_i + b * Lambda_k_t

            # Outgoing recoil mass: Σ_j β_ji Λ_qi
            for j in range(N):
                if j == i:
                    continue
                b = beta[j, i].item() if beta.dim() == 2 else beta[j, i].mean().item()
                M_i = M_i + b * agent_i.precision_q.detach()

            # Sensory mass
            if obs_precision is not None:
                if isinstance(obs_precision, (int, float)):
                    M_i = M_i + obs_precision * torch.eye(K, device=M_i.device)
                else:
                    M_i = M_i + obs_precision

            masses.append(M_i)

        return torch.stack(masses)

    def off_diagonal_mass(self, i: int, k: int,
                           beta: Optional[Tensor] = None) -> Tensor:
        """Off-diagonal mass block M_ik (Eq. 38).

        [M_μμ]_{ik} = -β_ik Ω_ik Λ_qk - β_ki Λ_qi Ω_ki^T

        Args:
            i, k: agent indices
            beta: attention weights
        Returns:
            (K, K) off-diagonal mass block
        """
        if beta is None:
            result = self.free_energy(self.system)
            beta = result['beta']
            while beta.dim() > 2:
                beta = beta.mean(-1)

        agent_i = self.system.agents[i]
        agent_k = self.system.agents[k]

        omega_ik = agent_i.omega.data @ torch.linalg.inv(agent_k.omega.data)
        omega_ki = agent_k.omega.data @ torch.linalg.inv(agent_i.omega.data)

        b_ik = beta[i, k].item() if beta.dim() == 2 else beta[i, k].mean().item()
        b_ki = beta[k, i].item() if beta.dim() == 2 else beta[k, i].mean().item()

        term1 = -b_ik * omega_ik @ agent_k.precision_q.detach()
        term2 = -b_ki * agent_i.precision_q.detach() @ omega_ki.transpose(-2, -1)

        return term1 + term2

    def kinetic_energy(self, velocities_mu: Tensor,
                       beta: Optional[Tensor] = None,
                       obs_precision: Optional[Tensor] = None) -> Tensor:
        """Kinetic energy T = (1/2) μ̇ᵀ M μ̇.

        Args:
            velocities_mu: (N, K) or (N, *grid, K) velocity vectors
            beta: attention weights
            obs_precision: observation precision
        Returns:
            Scalar kinetic energy
        """
        M_diag = self.effective_mass_diagonal(beta, obs_precision)  # (N, K, K)

        # Diagonal contribution: Σ_i (1/2) μ̇_i^T M_ii μ̇_i
        v = velocities_mu
        if v.dim() == 2:
            # (N, K) → (N, 1, K) @ (N, K, K) @ (N, K, 1)
            T = 0.5 * (v.unsqueeze(-2) @ M_diag @ v.unsqueeze(-1)).sum()
        else:
            T = 0.5 * (v.unsqueeze(-2) @ M_diag.unsqueeze(1) @ v.unsqueeze(-1)).sum()

        return T

    def scalar_mass(self, beta: Optional[Tensor] = None,
                     obs_precision: Optional[Tensor] = None) -> Tensor:
        """Scalar effective mass per agent: tr(M_i).

        The trace of the mass matrix gives the total inertial mass.
        This is what the manuscript identifies with physical mass.

        Returns:
            (N,) scalar masses
        """
        M_diag = self.effective_mass_diagonal(beta, obs_precision)
        return M_diag.diagonal(dim1=-2, dim2=-1).sum(-1)


class InformationGeometricMass:
    """Analysis tools for the mass-precision correspondence.

    Validates: M_eff ∝ Σ_p⁻¹ (mass = precision of prior).

    Reference: Dennis (2026), Section 3.5, Fig. 10
    """

    @staticmethod
    def mass_precision_correlation(system: MultiAgentSystem,
                                    free_energy: FreeEnergyFunctional
                                    ) -> Dict[str, Tensor]:
        """Compute correlation between mass and prior precision.

        Returns:
            Dict with masses, precisions, correlation coefficient
        """
        mm = MassMatrix(system, free_energy)
        masses = mm.scalar_mass()  # (N,)

        precisions = []
        for agent in system.agents:
            # tr(Λ_p)
            prec = agent.precision_p.detach().diagonal(dim1=-2, dim2=-1).sum(-1)
            if prec.dim() > 0:
                prec = prec.mean()
            precisions.append(prec)
        precisions = torch.stack(precisions)

        # Pearson correlation
        m = masses - masses.mean()
        p = precisions - precisions.mean()
        corr = (m * p).sum() / (m.norm() * p.norm() + 1e-10)

        return {
            'masses': masses,
            'precisions': precisions,
            'correlation': corr,
        }

    @staticmethod
    def harmonic_frequency(mass: float, spring_constant: float = 1.0) -> float:
        """ω² = k/M — harmonic oscillator frequency.

        For validation: lighter agents (lower precision priors)
        should oscillate faster.
        """
        return (spring_constant / mass) ** 0.5
