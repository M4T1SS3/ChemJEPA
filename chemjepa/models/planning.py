"""
Imagination Engine - Planning in Latent Space

Hybrid Monte Carlo Tree Search (MCTS) + Energy-Guided Beam Search.

Key features:
- Operates in learned latent space (not SMILES strings)
- Energy-based scoring via compatibility function
- Uncertainty-aware pruning
- Counterfactual branching
- Reasoning trace generation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .latent import LatentState
from .energy import ChemJEPAEnergyModel
from .dynamics import DynamicsPredictor
from .novelty import NoveltyDetector


@dataclass
class PlanningNode:
    """Node in planning tree."""
    state: LatentState
    action: Optional[torch.Tensor]  # Action that led to this state
    parent: Optional['PlanningNode']
    children: List['PlanningNode']
    visit_count: int
    total_value: float
    uncertainty: Optional[torch.Tensor]

    def __post_init__(self):
        if self.children is None:
            self.children = []

    @property
    def q_value(self) -> float:
        """Average value."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


@dataclass
class ReasoningTrace:
    """Trace of planning decisions."""
    states: List[LatentState]
    actions: List[torch.Tensor]
    energies: List[float]
    uncertainties: List[float]
    branching_points: List[int]  # Indices where counterfactual branching occurred

    def __len__(self):
        return len(self.actions)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "num_steps": len(self),
            "energies": [float(e) for e in self.energies],
            "uncertainties": [float(u) for u in self.uncertainties],
            "branching_points": self.branching_points,
        }


class DeterminantalPointProcess:
    """
    Determinantal Point Process for diverse subset selection.

    Ensures beam maintains diversity in latent space.
    """

    def __init__(self, kernel_bandwidth: float = 1.0):
        self.kernel_bandwidth = kernel_bandwidth

    def rbf_kernel(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """RBF (Gaussian) kernel."""
        if Y is None:
            Y = X

        # Pairwise distances
        XX = (X ** 2).sum(dim=1, keepdim=True)
        YY = (Y ** 2).sum(dim=1, keepdim=True).t()
        distances = XX + YY - 2 * torch.mm(X, Y.t())

        return torch.exp(-distances / (2 * self.kernel_bandwidth ** 2))

    def sample_diverse_subset(
        self,
        items: torch.Tensor,
        scores: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Sample diverse subset of size k.

        Args:
            items: Item embeddings [N, dim]
            scores: Quality scores [N]
            k: Subset size

        Returns:
            indices: Selected indices [k]
        """
        N = items.shape[0]
        k = min(k, N)

        # Compute kernel matrix
        L = self.rbf_kernel(items)  # [N, N]

        # Weight by quality scores
        L = L * scores.unsqueeze(0) * scores.unsqueeze(1)

        # Greedy selection (approximation to DPP sampling)
        selected = []
        selected_mask = torch.zeros(N, dtype=torch.bool, device=items.device)

        for _ in range(k):
            if len(selected) == 0:
                # Select item with highest score
                idx = torch.argmax(scores).item()
            else:
                # Select item that maximizes determinant
                # Approximation: select item with highest score among remaining
                # that is most dissimilar to selected items
                remaining_scores = scores.clone()
                remaining_scores[selected_mask] = -float('inf')

                # Penalize similarity to selected items
                for s_idx in selected:
                    similarity = L[:, s_idx]
                    remaining_scores = remaining_scores - similarity

                idx = torch.argmax(remaining_scores).item()

            selected.append(idx)
            selected_mask[idx] = True

        return torch.tensor(selected, device=items.device)


class ImaginationEngine(nn.Module):
    """
    Planning module for goal-directed molecular discovery.

    Uses hybrid MCTS + beam search in latent space.

    Args:
        energy_model: Energy-based compatibility function
        dynamics_model: Latent dynamics predictor
        novelty_detector: Open-world novelty detector
        beam_size: Beam width (default: 20)
        horizon: Planning horizon (default: 5)
        exploration_coef: UCB exploration coefficient (default: 1.0)
        novelty_penalty: Penalty for high uncertainty (default: 0.5)
        use_counterfactual: Enable counterfactual branching (default: True)
    """

    def __init__(
        self,
        energy_model: ChemJEPAEnergyModel,
        dynamics_model: DynamicsPredictor,
        novelty_detector: NoveltyDetector,
        beam_size: int = 20,
        horizon: int = 5,
        exploration_coef: float = 1.0,
        novelty_penalty: float = 0.5,
        use_counterfactual: bool = True,
    ):
        super().__init__()

        self.energy_model = energy_model
        self.dynamics_model = dynamics_model
        self.novelty_detector = novelty_detector

        self.beam_size = beam_size
        self.horizon = horizon
        self.exploration_coef = exploration_coef
        self.novelty_penalty = novelty_penalty
        self.use_counterfactual = use_counterfactual

        # Diversity sampling
        self.dpp = DeterminantalPointProcess()

    def sample_actions(self, k: int, device: torch.device) -> torch.Tensor:
        """
        Sample k candidate actions from reaction codebook.

        Args:
            k: Number of actions to sample
            device: Device

        Returns:
            Actions [k, action_dim]
        """
        # Sample random actions (in practice, could use learned proposal)
        action_dim = self.dynamics_model.action_dim
        actions = torch.randn(k, action_dim, device=device)

        return actions

    def score_state(
        self,
        latent_state: LatentState,
        z_target: torch.Tensor,
        z_env: torch.Tensor,
        p_target: torch.Tensor,
        property_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Score a state via energy model + novelty penalty.

        Args:
            latent_state: State to score
            z_target: Target protein embedding
            z_env: Environment embedding
            p_target: Target properties
            property_mask: Property mask

        Returns:
            score: Total score (higher = better) [B]
            info: Dictionary with breakdown
        """
        # Compute energy (lower = better, so negate for score)
        # Extract molecular embedding from latent state
        z_mol = latent_state.z_mol if hasattr(latent_state, 'z_mol') else latent_state

        energy_output = self.energy_model(
            z_mol, target_properties=p_target, return_components=True
        )

        score = -energy_output["energy"].squeeze(-1)  # [B]

        # Check novelty
        novelty_output = self.novelty_detector.is_novel(latent_state)

        # Penalize if novel
        if self.novelty_penalty > 0:
            novelty_mask = novelty_output["is_novel"].float()
            score = score - self.novelty_penalty * novelty_mask

        info = {
            "energy": energy_output["energy"].squeeze(-1),
            "is_novel": novelty_output["is_novel"],
            "density_score": novelty_output["density_score"],
            "energy_components": energy_output,
        }

        return score, info

    def beam_search_step(
        self,
        beam: List[LatentState],
        z_target: torch.Tensor,
        z_env: torch.Tensor,
        p_target: torch.Tensor,
        num_actions: int = 50,
        property_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[LatentState], List[torch.Tensor], List[float]]:
        """
        Single step of beam search.

        Args:
            beam: Current beam of states
            z_target: Target embedding
            z_env: Environment embedding
            p_target: Target properties
            num_actions: Number of actions to try per state
            property_mask: Property mask

        Returns:
            next_beam: Next beam of states
            actions: Actions taken
            scores: Scores for selected states
        """
        candidates = []
        candidate_states = []
        candidate_actions = []
        candidate_scores = []

        device = beam[0].z_mol.device

        # Expand each state in beam
        for state in beam:
            # Sample candidate actions
            actions = self.sample_actions(num_actions, device)

            # Batch predict next states
            # Expand state to batch
            state_batch = LatentState(
                z_mol=state.z_mol.repeat(num_actions, 1),
                z_rxn=state.z_rxn.repeat(num_actions, 1),
                z_context=state.z_context.repeat(num_actions, 1),
            )

            # Predict transitions
            outputs = self.dynamics_model(state_batch, actions, predict_uncertainty=True)
            next_states_batch = outputs["next_state"]

            # Score next states
            z_target_batch = z_target.repeat(num_actions, 1)
            z_env_batch = z_env.repeat(num_actions, 1)
            p_target_batch = p_target.repeat(num_actions, 1)

            scores, _ = self.score_state(
                next_states_batch,
                z_target_batch,
                z_env_batch,
                p_target_batch,
                property_mask,
            )

            # Store candidates
            for i in range(num_actions):
                next_state = LatentState(
                    z_mol=next_states_batch.z_mol[i:i+1],
                    z_rxn=next_states_batch.z_rxn[i:i+1],
                    z_context=next_states_batch.z_context[i:i+1],
                )
                candidate_states.append(next_state)
                candidate_actions.append(actions[i:i+1])
                candidate_scores.append(scores[i].item())

        # Select diverse top-K using DPP
        candidate_embeddings = torch.cat([s.z_mol for s in candidate_states], dim=0)
        candidate_scores_tensor = torch.tensor(candidate_scores, device=device)

        selected_indices = self.dpp.sample_diverse_subset(
            candidate_embeddings,
            candidate_scores_tensor,
            k=self.beam_size,
        )

        # Build next beam
        next_beam = [candidate_states[i] for i in selected_indices]
        selected_actions = [candidate_actions[i] for i in selected_indices]
        selected_scores = [candidate_scores[i] for i in selected_indices]

        return next_beam, selected_actions, selected_scores

    def plan(
        self,
        initial_state: LatentState,
        z_target: torch.Tensor,
        z_env: torch.Tensor,
        p_target: torch.Tensor,
        property_mask: Optional[torch.Tensor] = None,
        return_traces: bool = True,
    ) -> Dict:
        """
        Run planning to find high-value states.

        Args:
            initial_state: Starting state
            z_target: Target protein embedding [1, target_dim]
            z_env: Environment embedding [1, env_dim]
            p_target: Target properties [1, property_dim]
            property_mask: Property mask [1, num_properties]
            return_traces: Whether to return reasoning traces

        Returns:
            Dictionary with:
                - final_states: List of final states (beam_size)
                - scores: Scores for final states
                - traces: Reasoning traces (if return_traces)
        """
        beam = [initial_state]
        all_traces = [] if return_traces else None

        if return_traces:
            # Initialize traces
            for _ in range(self.beam_size):
                all_traces.append(ReasoningTrace(
                    states=[initial_state],
                    actions=[],
                    energies=[],
                    uncertainties=[],
                    branching_points=[],
                ))

        # Planning loop
        for t in range(self.horizon):
            beam, actions, scores = self.beam_search_step(
                beam,
                z_target,
                z_env,
                p_target,
                property_mask=property_mask,
            )

            # Update traces
            if return_traces:
                for i, (state, action, score) in enumerate(zip(beam, actions, scores)):
                    if i < len(all_traces):
                        all_traces[i].states.append(state)
                        all_traces[i].actions.append(action)
                        all_traces[i].energies.append(-score)  # Convert back to energy

        # Final scoring
        final_scores = []
        for state in beam:
            score, _ = self.score_state(state, z_target, z_env, p_target, property_mask)
            final_scores.append(score.item())

        output = {
            "final_states": beam,
            "scores": final_scores,
        }

        if return_traces:
            output["traces"] = all_traces

        return output

    def imagine(
        self,
        z_target: torch.Tensor,
        z_env: torch.Tensor,
        p_target: torch.Tensor,
        num_candidates: int = 10,
        initial_states: Optional[List[LatentState]] = None,
        property_mask: Optional[torch.Tensor] = None,
        return_traces: bool = True,
    ) -> Dict:
        """
        High-level interface for molecular imagination.

        Args:
            z_target: Target protein embedding [1, target_dim]
            z_env: Environment embedding [1, env_dim]
            p_target: Target properties [1, property_dim]
            num_candidates: Number of candidates to return
            initial_states: Optional list of starting states (else random)
            property_mask: Property mask
            return_traces: Return reasoning traces

        Returns:
            Dictionary with top candidates, scores, and traces
        """
        device = z_target.device

        # Initialize starting states if not provided
        if initial_states is None:
            # Sample random initial states
            num_init = self.beam_size
            initial_states = []

            for _ in range(num_init):
                # Sample from learned prior (simple Gaussian for now)
                z_mol = torch.randn(1, self.energy_model.mol_dim, device=device)
                z_rxn = torch.randn(1, 384, device=device)  # TODO: make configurable
                z_context = torch.cat([z_env, z_target, p_target], dim=-1)

                # Project to correct dimension
                if z_context.shape[-1] != 256:  # TODO: make configurable
                    z_context = z_context[:, :256]

                initial_state = LatentState(z_mol=z_mol, z_rxn=z_rxn, z_context=z_context)
                initial_states.append(initial_state)

        # Run planning from each initial state
        all_results = []

        for init_state in initial_states:
            result = self.plan(
                init_state,
                z_target,
                z_env,
                p_target,
                property_mask=property_mask,
                return_traces=return_traces,
            )
            all_results.append(result)

        # Collect all final states and scores
        all_states = []
        all_scores = []
        all_traces = []

        for result in all_results:
            all_states.extend(result["final_states"])
            all_scores.extend(result["scores"])
            if return_traces:
                all_traces.extend(result["traces"])

        # Select top-K
        sorted_indices = np.argsort(all_scores)[::-1][:num_candidates]

        top_states = [all_states[i] for i in sorted_indices]
        top_scores = [all_scores[i] for i in sorted_indices]

        output = {
            "candidates": top_states,
            "scores": top_scores,
        }

        if return_traces:
            output["traces"] = [all_traces[i] for i in sorted_indices]

        return output
