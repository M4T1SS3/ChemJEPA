"""
ChemJEPA: Main Model Interface

Unified interface for the complete ChemJEPA architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import warnings

from .models.encoders.molecular import MolecularEncoder
from .models.encoders.environment import EnvironmentEncoder
from .models.encoders.protein import ProteinEncoder
from .models.latent import HierarchicalLatentWorldState, LatentState
from .models.energy import EnergyModel
from .models.dynamics import DynamicsPredictor
from .models.novelty import NoveltyDetector
from .models.planning import ImaginationEngine


class ChemJEPA(nn.Module):
    """
    ChemJEPA: Joint-Embedding Predictive Architecture for Open-World Chemistry

    Complete model integrating all components:
    - Molecular/Environment/Protein encoders
    - Hierarchical latent world state
    - Energy-based compatibility
    - Latent dynamics prediction
    - Novelty detection
    - Planning/imagination engine

    Args:
        mol_encoder_config: Configuration for molecular encoder
        env_encoder_config: Configuration for environment encoder
        protein_encoder_config: Configuration for protein encoder
        mol_dim: Molecular embedding dimension (default: 768)
        rxn_dim: Reaction state dimension (default: 384)
        context_dim: Context dimension (default: 256)
        target_dim: Protein target dimension (default: 256)
        env_dim: Environment dimension (default: 128)
        property_dim: Property vector dimension (default: 64)
        num_properties: Number of property heads (default: 10)
        device: Device to run on (default: 'cuda' if available)
    """

    def __init__(
        self,
        mol_encoder_config: Optional[Dict] = None,
        env_encoder_config: Optional[Dict] = None,
        protein_encoder_config: Optional[Dict] = None,
        mol_dim: int = 768,
        rxn_dim: int = 384,
        context_dim: int = 256,
        target_dim: int = 256,
        env_dim: int = 128,
        property_dim: int = 64,
        num_properties: int = 10,
        device: str = None,
    ):
        super().__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.mol_dim = mol_dim
        self.rxn_dim = rxn_dim
        self.context_dim = context_dim

        # Encoders
        mol_config = mol_encoder_config or {}
        # RDKit generates 32-dim atom features by default
        self.molecular_encoder = MolecularEncoder(
            atom_feature_dim=32,  # Match RDKit output
            edge_feature_dim=12,  # Match RDKit edge features
            local_dim=mol_dim // 2,
            global_dim=mol_dim // 2,
            use_3d=False,  # Disable 3D for now (has bugs)
            **mol_config
        ).to(self.device)

        env_config = env_encoder_config or {}
        self.environment_encoder = EnvironmentEncoder.create_default(
            output_dim=env_dim,
        ).to(self.device)

        protein_config = protein_encoder_config or {}
        self.protein_encoder = ProteinEncoder.create_sequence_only(
            output_dim=target_dim,
        ).to(self.device)

        # Hierarchical latent world state
        self.latent_model = HierarchicalLatentWorldState(
            mol_dim=mol_dim,
            rxn_dim=rxn_dim,
            context_dim=context_dim,
        ).to(self.device)

        # Energy model
        self.energy_model = EnergyModel(
            mol_dim=mol_dim,
            rxn_dim=rxn_dim,
            context_dim=context_dim,
            target_dim=target_dim,
            env_dim=env_dim,
            property_dim=property_dim,
            num_properties=num_properties,
        ).to(self.device)

        # Dynamics predictor
        self.dynamics_model = DynamicsPredictor(
            mol_dim=mol_dim,
            rxn_dim=rxn_dim,
            context_dim=context_dim,
        ).to(self.device)

        # Novelty detector
        self.novelty_detector = NoveltyDetector(
            mol_dim=mol_dim,
            rxn_dim=rxn_dim,
            context_dim=context_dim,
        ).to(self.device)

        # Planning engine
        self.imagination_engine = ImaginationEngine(
            energy_model=self.energy_model,
            dynamics_model=self.dynamics_model,
            novelty_detector=self.novelty_detector,
        ).to(self.device)

    def encode_molecule(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode molecular graph.

        Args:
            x: Atom features [N, atom_dim]
            edge_index: Edge connectivity [2, E]
            batch: Batch assignment [N]
            edge_attr: Edge features [E, edge_dim]
            pos: 3D coordinates [N, 3]

        Returns:
            z_mol: Molecular embedding [B, mol_dim]
        """
        z_local, z_global = self.molecular_encoder(x, edge_index, batch, edge_attr, pos)
        z_mol = torch.cat([z_local, z_global], dim=-1)
        return z_mol

    def encode_environment(
        self,
        categorical: Optional[Dict[str, torch.Tensor]] = None,
        continuous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode environment/conditions.

        Args:
            categorical: Categorical features
            continuous: Continuous features

        Returns:
            z_env: Environment embedding [B, env_dim]
        """
        return self.environment_encoder(categorical, continuous)

    def encode_protein(
        self,
        sequence: Optional[str] = None,
        esm_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode protein target.

        Args:
            sequence: Amino acid sequence
            esm_embeddings: Pre-computed ESM embeddings

        Returns:
            z_target: Protein embedding [B, target_dim]
        """
        return self.protein_encoder(sequence=sequence, esm_embeddings=esm_embeddings)

    def forward(
        self,
        mol_graph: Tuple,
        env_features: Tuple,
        protein_features: Tuple,
        p_target: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        domain: str = "organic",
        property_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through ChemJEPA.

        Args:
            mol_graph: (x, edge_index, batch, edge_attr, pos)
            env_features: (categorical, continuous)
            protein_features: (sequence or esm_embeddings)
            p_target: Target properties [B, property_dim]
            action: Reaction operator [B, action_dim] (optional)
            domain: Chemistry domain
            property_mask: Property mask [B, num_properties]

        Returns:
            Dictionary with all outputs
        """
        # Encode inputs
        x, edge_index, batch, edge_attr, pos = mol_graph
        z_mol = self.encode_molecule(x, edge_index, batch, edge_attr, pos)

        categorical, continuous = env_features
        z_env = self.encode_environment(categorical, continuous)

        if isinstance(protein_features, str):
            z_target = self.encode_protein(sequence=protein_features)
        else:
            z_target = self.encode_protein(esm_embeddings=protein_features)

        # Initialize reaction state (zeros for now, or from prior)
        B = z_mol.shape[0]
        z_rxn = torch.zeros(B, self.rxn_dim, device=self.device)

        # Construct context
        z_context = torch.cat([z_env, z_target, p_target], dim=-1)
        if z_context.shape[-1] != self.context_dim:
            # Project to correct dimension
            if not hasattr(self, 'context_proj'):
                self.context_proj = nn.Linear(
                    z_context.shape[-1], self.context_dim
                ).to(self.device)
            z_context = self.context_proj(z_context)

        # Process through latent model
        latent_state, vae_params = self.latent_model(
            z_mol, z_rxn, z_context, domain=domain
        )

        # Compute energy
        energy_output = self.energy_model(
            latent_state, z_target, z_env, p_target, property_mask, return_components=True
        )

        output = {
            "latent_state": latent_state,
            "vae_params": vae_params,
            "energy": energy_output["total_energy"],
            "energy_weights": energy_output["weights"],
            "property_predictions": energy_output["property_predictions"],
            "z_mol": latent_state.z_mol,
            "z_rxn": latent_state.z_rxn,
            "z_context": latent_state.z_context,
        }

        # If action provided, predict dynamics
        if action is not None:
            dynamics_output = self.dynamics_model(
                latent_state, action, predict_uncertainty=True
            )
            output["next_state"] = dynamics_output["next_state"]
            output["uncertainty"] = dynamics_output["uncertainty"]
            if "vq_loss" in dynamics_output:
                output["vq_loss"] = dynamics_output["vq_loss"]

        return output

    def imagine(
        self,
        target_properties: Dict[str, float],
        protein_target: Optional[str] = None,
        environment: Optional[Dict] = None,
        num_candidates: int = 10,
        return_uncertainty: bool = True,
        return_reasoning_trace: bool = True,
    ) -> List[Dict]:
        """
        High-level interface for molecular imagination.

        Args:
            target_properties: Dictionary of target properties
                e.g., {"IC50": "<10nM", "bioavailability": ">50%"}
            protein_target: Protein sequence or identifier
            environment: Environment specification
            num_candidates: Number of candidates to generate
            return_uncertainty: Include uncertainty estimates
            return_reasoning_trace: Include reasoning traces

        Returns:
            List of candidate molecules with scores and metadata
        """
        # Encode target
        if protein_target is not None:
            z_target = self.encode_protein(sequence=protein_target)
        else:
            warnings.warn("No protein target provided, using zero vector")
            z_target = torch.zeros(1, 256, device=self.device)  # TODO: make configurable

        # Encode environment
        if environment is not None:
            # Parse environment dict
            categorical = environment.get("categorical", None)
            continuous = environment.get("continuous", None)
            z_env = self.encode_environment(categorical, continuous)
        else:
            warnings.warn("No environment provided, using zero vector")
            z_env = torch.zeros(1, 128, device=self.device)  # TODO: make configurable

        # Encode target properties
        # TODO: implement proper property encoding from dict
        p_target = torch.zeros(1, 64, device=self.device)  # Placeholder

        # Run imagination engine
        results = self.imagination_engine.imagine(
            z_target=z_target,
            z_env=z_env,
            p_target=p_target,
            num_candidates=num_candidates,
            return_traces=return_reasoning_trace,
        )

        # Format output
        candidates = []
        for i, (state, score) in enumerate(zip(results["candidates"], results["scores"])):
            candidate = {
                "rank": i + 1,
                "latent_state": state,
                "score": score,
            }

            if return_reasoning_trace and "traces" in results:
                candidate["trace"] = results["traces"][i].to_dict()

            if return_uncertainty:
                novelty_output = self.novelty_detector.is_novel(state)
                candidate["uncertainty"] = {
                    "is_novel": novelty_output["is_novel"].item(),
                    "density_score": novelty_output["density_score"].item(),
                }

            candidates.append(candidate)

        return candidates

    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded pretrained weights from {checkpoint_path}")

    def save_checkpoint(self, checkpoint_path: str, optimizer=None, epoch=None, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "mol_dim": self.mol_dim,
                "rxn_dim": self.rxn_dim,
                "context_dim": self.context_dim,
            },
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if epoch is not None:
            checkpoint["epoch"] = epoch

        checkpoint.update(kwargs)

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
