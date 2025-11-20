#!/usr/bin/env python3
"""
Phase 3 Data Generation: State Transition Trajectories

Generates synthetic state transition data for training the dynamics model.

Strategy:
    1. Load Phase 1 encoder (frozen)
    2. Sample diverse molecules from dataset
    3. Generate random "actions" (reaction operators)
    4. Create pseudo-transitions by adding noise in latent space
    5. Build trajectory dataset with (state, action, next_state) tuples

This is a simplified approach that doesn't require actual reaction simulation.
In a full system, you'd use real reaction data or a reaction simulator.

Output: data/phase3_transitions.pt
Estimated time: 15-30 minutes
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chemjepa import ChemJEPA
from chemjepa.models.latent import LatentState
from chemjepa.data.loaders import MolecularDataset, collate_molecular_batch
from torch.utils.data import DataLoader

# Device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")


def load_phase1_model():
    """Load frozen Phase 1 encoder."""
    checkpoint_path = project_root / 'checkpoints' / 'best_jepa.pt'

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Phase 1 checkpoint not found at {checkpoint_path}. "
            "Please train Phase 1 first."
        )

    print(f"Loading Phase 1 model from {checkpoint_path}")

    model = ChemJEPA(device=device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Filter out Phase 2/3 components
    filtered_state_dict = {
        k: v for k, v in checkpoint['model_state_dict'].items()
        if not k.startswith('energy_model.') and not k.startswith('imagination_engine.')
    }

    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()

    print("✓ Phase 1 model loaded")
    return model


def extract_embeddings(model, data_loader, max_samples=10000):
    """Extract molecular embeddings from Phase 1."""
    embeddings = []
    smiles_list = []

    print(f"Extracting embeddings (max {max_samples} samples)...")

    with torch.no_grad():
        for batch in tqdm(data_loader):
            if len(embeddings) >= max_samples:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Use encode_molecule method
            z_mol = model.encode_molecule(
                x=batch['x'],
                edge_index=batch['edge_index'],
                batch=batch['batch'],
                edge_attr=batch.get('edge_attr'),
                pos=batch.get('pos'),
            )
            embeddings.append(z_mol.cpu())
            smiles_list.extend(batch['smiles'])

    embeddings = torch.cat(embeddings, dim=0)[:max_samples]
    smiles_list = smiles_list[:max_samples]

    print(f"✓ Extracted {len(embeddings)} embeddings")
    return embeddings, smiles_list


def generate_synthetic_transitions(
    embeddings,
    num_trajectories=5000,
    trajectory_length=5,
    action_dim=256,
    mol_dim=768,
    rxn_dim=384,
    context_dim=256,
    noise_scale=0.1,
):
    """
    Generate synthetic state transitions.

    Strategy:
        - Each trajectory starts from a random molecular embedding
        - Actions are sampled from a Gaussian distribution
        - Next states are computed by adding action-dependent noise
        - This creates a pseudo-dynamics that can be learned

    Args:
        embeddings: Molecular embeddings [N, mol_dim]
        num_trajectories: Number of trajectories to generate
        trajectory_length: Length of each trajectory
        action_dim: Dimension of action vectors
        mol_dim: Dimension of molecular embeddings
        rxn_dim: Dimension of reaction state
        context_dim: Dimension of context
        noise_scale: Scale of noise added to transitions

    Returns:
        List of (current_state, action, next_state) tuples
    """
    print(f"\nGenerating {num_trajectories} synthetic trajectories...")
    print(f"  Trajectory length: {trajectory_length}")
    print(f"  Action dim: {action_dim}")
    print(f"  Noise scale: {noise_scale}")

    transitions = []

    for traj_idx in tqdm(range(num_trajectories)):
        # Sample initial molecular embedding
        init_idx = np.random.randint(0, len(embeddings))
        z_mol = embeddings[init_idx].clone()

        # Initialize reaction and context states randomly
        z_rxn = torch.randn(rxn_dim)
        z_context = torch.randn(context_dim)

        current_state = LatentState(
            z_mol=z_mol.unsqueeze(0),
            z_rxn=z_rxn.unsqueeze(0),
            z_context=z_context.unsqueeze(0),
        )

        # Generate trajectory
        for step in range(trajectory_length):
            # Sample random action
            action = torch.randn(1, action_dim)

            # Generate next state with action-dependent noise
            # This creates a learnable pattern: next_state depends on (state, action)

            # Action influences molecular change
            mol_change = noise_scale * torch.randn_like(current_state.z_mol)
            mol_change = mol_change + 0.01 * action.mean() * torch.randn_like(current_state.z_mol)

            # Action influences reaction state change
            rxn_change = noise_scale * torch.randn_like(current_state.z_rxn)
            rxn_change = rxn_change + 0.02 * action.std() * torch.randn_like(current_state.z_rxn)

            # Context changes slowly (environmental drift)
            context_change = 0.5 * noise_scale * torch.randn_like(current_state.z_context)

            next_state = LatentState(
                z_mol=current_state.z_mol + mol_change,
                z_rxn=current_state.z_rxn + rxn_change,
                z_context=current_state.z_context + context_change,
            )

            # Store transition
            transitions.append({
                'current_state': {
                    'z_mol': current_state.z_mol.squeeze(0),
                    'z_rxn': current_state.z_rxn.squeeze(0),
                    'z_context': current_state.z_context.squeeze(0),
                },
                'action': action.squeeze(0),
                'next_state': {
                    'z_mol': next_state.z_mol.squeeze(0),
                    'z_rxn': next_state.z_rxn.squeeze(0),
                    'z_context': next_state.z_context.squeeze(0),
                },
            })

            # Update current state
            current_state = next_state

    print(f"✓ Generated {len(transitions)} transitions")
    return transitions


def main():
    print("=" * 60)
    print("Phase 3 Data Generation: State Transition Trajectories")
    print("=" * 60)
    print()

    # Create data directory
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)

    # Load Phase 1 model
    model = load_phase1_model()

    # Load molecular dataset
    print("\nLoading molecular dataset...")
    train_csv = project_root / 'data' / 'zinc250k' / 'train.csv'
    if not train_csv.exists():
        # Try alternative path
        train_csv = project_root / 'data' / 'qm9' / 'qm9_sample.csv'
        if not train_csv.exists():
            raise FileNotFoundError(
                f"Training data not found. Please ensure data is in data/zinc250k/ or data/qm9/"
            )

    print(f"Using dataset: {train_csv}")
    dataset = MolecularDataset(
        data_path=str(train_csv),
        smiles_column='smiles',
        use_3d=True
    )
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_molecular_batch,
        num_workers=0,
    )
    print(f"✓ Dataset loaded: {len(dataset)} molecules")

    # Extract embeddings
    embeddings, smiles_list = extract_embeddings(model, data_loader, max_samples=10000)

    # Generate synthetic transitions
    transitions = generate_synthetic_transitions(
        embeddings,
        num_trajectories=5000,
        trajectory_length=5,
        noise_scale=0.1,
    )

    # Split into train/val
    num_train = int(0.9 * len(transitions))
    train_transitions = transitions[:num_train]
    val_transitions = transitions[num_train:]

    print(f"\nDataset split:")
    print(f"  Train: {len(train_transitions)} transitions")
    print(f"  Val:   {len(val_transitions)} transitions")

    # Save dataset
    output_path = data_dir / 'phase3_transitions.pt'
    print(f"\nSaving to {output_path}")

    torch.save({
        'train_transitions': train_transitions,
        'val_transitions': val_transitions,
        'embeddings': embeddings,
        'smiles_list': smiles_list,
        'config': {
            'mol_dim': 768,
            'rxn_dim': 384,
            'context_dim': 256,
            'action_dim': 256,
            'num_trajectories': 5000,
            'trajectory_length': 5,
        }
    }, output_path)

    print("✓ Dataset saved")
    print()
    print("=" * 60)
    print("Data Generation Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Train dynamics model:")
    print("     python3 training/train_phase3_dynamics.py")
    print()
    print("  2. Train novelty detector:")
    print("     python3 training/train_phase3_novelty.py")
    print()


if __name__ == '__main__':
    main()
