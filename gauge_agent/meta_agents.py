"""
Layer 7: Meta-Agent Formation and Hierarchical Emergence
=========================================================

When agents achieve sufficient belief coherence, they form meta-agents
at the next hierarchical scale. Meta-agents are full agents in their
own right — sections (q_I, p_I, Ω_I) constructed via gauge-covariant
averaging of constituent beliefs.

Consensus detection:
  C_belief({i}, c) = 1 - (1/|{i}|²) Σ_ij KL(q_i || Ω_ij[q_j])
  C_model({i}, c)  = 1 - (1/|{i}|²) Σ_ij KL(p_i || Ω̃_ij[p_j])
  Γ = C_belief · C_model · Presence > Γ_min

Reference: Dennis (2026), Sections 4.2–4.3
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, List, Tuple, Set

from gauge_agent.agents import Agent, MultiAgentSystem
from gauge_agent.lie_groups import transport_operator, transport_mean, transport_covariance
from gauge_agent.statistical_manifold import gaussian_kl


class ConsensusDetector(nn.Module):
    """Detects epistemic consensus among agents for meta-agent formation.

    Monitors belief coherence, model coherence, and spatial overlap,
    combining them into a consensus score Γ.

    Args:
        kl_threshold: maximum mean KL for consensus (τ_KL)
        min_cluster_size: minimum agents for meta-agent formation
        gamma_min: minimum consensus score threshold
    """

    def __init__(self, kl_threshold: float = 0.05,
                 min_cluster_size: int = 2,
                 gamma_min: float = 0.5):
        super().__init__()
        self.kl_threshold = kl_threshold
        self.min_cluster_size = min_cluster_size
        self.gamma_min = gamma_min

    @torch.no_grad()
    def belief_coherence(self, system: MultiAgentSystem) -> Tensor:
        """C_belief = 1 - mean KL between gauge-transported beliefs.

        Returns:
            (N, N) pairwise belief coherence matrix
        """
        E = system.pairwise_alignment_energies('belief')  # (N, N, *grid)
        # Average over grid
        while E.dim() > 2:
            E = E.mean(-1)
        return 1.0 - E

    @torch.no_grad()
    def model_coherence(self, system: MultiAgentSystem) -> Tensor:
        """C_model = 1 - mean KL between gauge-transported priors.

        Returns:
            (N, N) pairwise model coherence matrix
        """
        E = system.pairwise_alignment_energies('prior')
        while E.dim() > 2:
            E = E.mean(-1)
        return 1.0 - E

    @torch.no_grad()
    def consensus_score(self, system: MultiAgentSystem) -> Tensor:
        """Γ = C_belief · C_model (elementwise).

        Returns:
            (N, N) consensus scores
        """
        C_b = self.belief_coherence(system)
        C_m = self.model_coherence(system)
        return C_b * C_m

    @torch.no_grad()
    def find_clusters(self, system: MultiAgentSystem) -> List[Set[int]]:
        """Find clusters of agents satisfying consensus threshold.

        Uses greedy agglomerative clustering on the consensus matrix.

        Returns:
            List of sets, each set containing agent indices forming a cluster
        """
        N = system.N_agents
        gamma = self.consensus_score(system)

        # Adjacency: agents are connected if Γ_ij > Γ_min
        adj = (gamma > self.gamma_min).float()
        # Remove self-connections
        adj.fill_diagonal_(0)

        # Greedy connected components
        visited = set()
        clusters = []
        for i in range(N):
            if i in visited:
                continue
            # BFS from i
            cluster = {i}
            queue = [i]
            while queue:
                node = queue.pop(0)
                for j in range(N):
                    if j not in visited and j not in cluster and adj[node, j] > 0:
                        cluster.add(j)
                        queue.append(j)
            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)
                visited.update(cluster)

        return clusters

    @torch.no_grad()
    def check_epistemic_death(self, system: MultiAgentSystem,
                               threshold: float = 1e-4) -> bool:
        """Check if system has reached epistemic death.

        All agent pairs have KL < threshold after gauge transport.

        Returns:
            True if system is epistemically dead
        """
        E = system.pairwise_alignment_energies('belief')
        N = system.N_agents
        mask = 1.0 - torch.eye(N, device=E.device)
        for _ in range(E.dim() - 2):
            mask = mask.unsqueeze(-1)
        off_diag = E * mask
        return off_diag.max().item() < threshold


class MetaAgentFormation(nn.Module):
    """Forms meta-agents from clusters of coherent agents.

    Meta-agent beliefs are constructed via gauge-covariant averaging:
      μ̄_I = Σ_i w_i Ω_Ii μ_i / Σ_i w_i
      Σ̄_I = Σ_i w_i Ω_Ii Σ_i Ω_Ii^T / Σ_i w_i

    Weights w_i depend on coherence with the emerging consensus.

    Reference: Dennis (2026), Section 4.3.1
    """

    def __init__(self, consensus_detector: ConsensusDetector):
        super().__init__()
        self.consensus_detector = consensus_detector

    @torch.no_grad()
    def form_meta_agent(self, system: MultiAgentSystem,
                        cluster: Set[int],
                        reference_idx: Optional[int] = None) -> Agent:
        """Construct a meta-agent from a cluster of constituent agents.

        Uses gauge-covariant averaging: transport all beliefs to a
        reference frame, then average.

        Args:
            system: the multi-agent system
            cluster: set of agent indices forming the meta-agent
            reference_idx: which agent's frame to use as meta-agent frame
                           (default: first agent in cluster)
        Returns:
            New Agent representing the meta-agent
        """
        cluster_list = sorted(cluster)
        if reference_idx is None:
            reference_idx = cluster_list[0]

        ref_agent = system.agents[reference_idx]
        K = system.K
        grid_shape = system.grid_shape

        # Initialize accumulators
        mu_q_acc = torch.zeros_like(ref_agent.mu_q.data)
        sigma_q_acc = torch.zeros_like(ref_agent.sigma_q)
        mu_p_acc = torch.zeros_like(ref_agent.mu_p.data)
        sigma_p_acc = torch.zeros_like(ref_agent.sigma_p)
        omega_acc = torch.zeros_like(ref_agent.omega.data)
        weight_sum = torch.tensor(0.0, device=mu_q_acc.device)

        ref_omega = ref_agent.omega.data

        for idx in cluster_list:
            agent = system.agents[idx]

            # Transport operator: Ω_{ref,idx} = Ω_ref @ Ω_idx⁻¹
            omega_ij = ref_omega @ torch.linalg.inv(agent.omega.data)

            # Transport beliefs to reference frame
            mu_q_t = transport_mean(omega_ij, agent.mu_q.data)
            sigma_q_t = transport_covariance(omega_ij, agent.sigma_q)
            mu_p_t = transport_mean(omega_ij, agent.mu_p.data)
            sigma_p_t = transport_covariance(omega_ij, agent.sigma_p)

            # Weight by coherence (simple uniform for now)
            w = 1.0

            mu_q_acc += w * mu_q_t
            sigma_q_acc += w * sigma_q_t
            mu_p_acc += w * mu_p_t
            sigma_p_acc += w * sigma_p_t
            omega_acc += w * agent.omega.data
            weight_sum += w

        # Normalize
        mu_q_avg = mu_q_acc / weight_sum
        sigma_q_avg = sigma_q_acc / weight_sum
        mu_p_avg = mu_p_acc / weight_sum
        sigma_p_avg = sigma_p_acc / weight_sum
        omega_avg = omega_acc / weight_sum

        # Create meta-agent
        meta = Agent(K, grid_shape, agent_id=-1)  # ID assigned later

        # Set parameters
        meta.mu_q.data.copy_(mu_q_avg)
        meta.mu_p.data.copy_(mu_p_avg)
        meta.omega.data.copy_(omega_avg)

        # Set Cholesky factors from averaged covariances (robust via eigh)
        def _robust_cholesky(S):
            S_sym = 0.5 * (S + S.transpose(-2, -1))
            evals, evecs = torch.linalg.eigh(S_sym)
            evals = evals.clamp(min=1e-4)
            S_spd = evecs @ torch.diag_embed(evals) @ evecs.transpose(-2, -1)
            return torch.linalg.cholesky(S_spd)

        meta._L_q.data.copy_(_robust_cholesky(sigma_q_avg))
        meta._L_p.data.copy_(_robust_cholesky(sigma_p_avg))

        return meta

    @torch.no_grad()
    def detect_and_form(self, system: MultiAgentSystem
                        ) -> Tuple[List[Agent], List[Set[int]]]:
        """Detect consensus clusters and form meta-agents.

        Returns:
            (meta_agents, clusters) — list of new meta-agents and
            the clusters they were formed from
        """
        clusters = self.consensus_detector.find_clusters(system)
        meta_agents = []

        for cluster in clusters:
            meta = self.form_meta_agent(system, cluster)
            meta_agents.append(meta)

        return meta_agents, clusters


class TopDownFeedback(nn.Module):
    """Propagates meta-agent beliefs back to constituents as priors.

    p_i^(s) ← Ω_{i,I}[q_I^(s+1)]

    The meta-agent's collective belief becomes the prior for
    constituent agents, closing the participatory loop.

    Reference: Dennis (2026), Section 4.5
    """

    @staticmethod
    @torch.no_grad()
    def propagate_prior(meta_agent: Agent,
                        constituents: List[Agent],
                        blend: float = 1.0):
        """Update constituent priors from meta-agent belief.

        Args:
            meta_agent: the parent meta-agent
            constituents: list of constituent agents
            blend: interpolation factor (1.0 = full replacement,
                   0.0 = no change)
        """
        meta_omega = meta_agent.omega.data
        meta_mu_q = meta_agent.mu_q.data
        meta_sigma_q = meta_agent.sigma_q

        for agent in constituents:
            # Transport meta belief into agent's frame
            omega_iI = agent.omega.data @ torch.linalg.inv(meta_omega)
            mu_new = transport_mean(omega_iI, meta_mu_q)
            sigma_new = transport_covariance(omega_iI, meta_sigma_q)

            # Blend with existing prior
            agent.mu_p.data.copy_(
                blend * mu_new + (1.0 - blend) * agent.mu_p.data
            )
            # Update Cholesky factor via eigendecomposition (robust)
            sigma_blended = blend * sigma_new + (1.0 - blend) * agent.sigma_p.detach()
            sigma_sym = 0.5 * (sigma_blended + sigma_blended.transpose(-2, -1))
            evals, evecs = torch.linalg.eigh(sigma_sym)
            evals = evals.clamp(min=1e-4)
            sigma_spd = evecs @ torch.diag_embed(evals) @ evecs.transpose(-2, -1)
            L_new = torch.linalg.cholesky(sigma_spd)
            agent._L_p.data.copy_(L_new)
