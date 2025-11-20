"""
ChemJEPA: Joint-Embedding Predictive Architecture for Open-World Chemistry

A hierarchical latent world model for molecular design and discovery.
"""

from .chemjepa import ChemJEPA
from .models.encoders.molecular import MolecularEncoder
from .models.encoders.environment import EnvironmentEncoder
from .models.encoders.protein import ProteinEncoder
from .models.latent import HierarchicalLatentWorldState
from .models.energy import ChemJEPAEnergyModel
from .models.dynamics import DynamicsPredictor
from .models.novelty import NoveltyDetector
from .models.planning import ImaginationEngine

__version__ = "0.1.0"
__author__ = "ChemJEPA Team"

__all__ = [
    "ChemJEPA",
    "MolecularEncoder",
    "EnvironmentEncoder",
    "ProteinEncoder",
    "HierarchicalLatentWorldState",
    "ChemJEPAEnergyModel",
    "DynamicsPredictor",
    "NoveltyDetector",
    "ImaginationEngine",
]
