#!/usr/bin/env python3
"""
Phase 3 Training: Novelty Detector

Trains the novelty detector for open-world uncertainty quantification.

Components:
    1. Normalizing flow for latent density estimation
    2. Threshold calibration on validation set
    3. Conformal prediction calibration

Strategy:
    - Train flow to model p(z) distribution of latent states
    - Learn what "in-distribution" looks like
    - Enable detection of novel/OOD molecules during planning

Estimated training time: 30 minutes on Apple M4 Pro
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

from chemjepa.models.novelty import NoveltyDetector
from chemjepa.models.latent import LatentState

# Device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")


def extract_latent_states(transitions):
    """Extract LatentState objects from transition data."""
    states = []

    for transition in transitions:
        # Current state
        current_state = LatentState(
            z_mol=transition['current_state']['z_mol'],
            z_rxn=transition['current_state']['z_rxn'],
            z_context=transition['current_state']['z_context'],
        )
        states.append(current_state)

        # Next state
        next_state = LatentState(
            z_mol=transition['next_state']['z_mol'],
            z_rxn=transition['next_state']['z_rxn'],
            z_context=transition['next_state']['z_context'],
        )
        states.append(next_state)

    return states


def calibrate_thresholds(detector, val_states, percentile=95):
    """
    Calibrate novelty detection thresholds on validation set.

    Strategy:
        - Compute density scores for all validation states
        - Set threshold at specified percentile (e.g., 95%)
        - States below threshold are considered "novel"

    Args:
        detector: NoveltyDetector model
        val_states: List of validation LatentState objects
        percentile: Percentile for threshold (default: 95)

    Returns:
        density_threshold: Calibrated threshold
    """
    print(f"\nCalibrating thresholds (percentile: {percentile})...")

    detector.eval()
    density_scores = []

    with torch.no_grad():
        for state in tqdm(val_states, desc="Computing density scores"):
            state = state.to(device)

            # Add batch dimension
            state_batch = LatentState(
                z_mol=state.z_mol.unsqueeze(0) if state.z_mol.dim() == 1 else state.z_mol,
                z_rxn=state.z_rxn.unsqueeze(0) if state.z_rxn.dim() == 1 else state.z_rxn,
                z_context=state.z_context.unsqueeze(0) if state.z_context.dim() == 1 else state.z_context,
            )

            score = detector.compute_density_score(state_batch)
            density_scores.append(score.item())

    density_scores = np.array(density_scores)

    # Set threshold at percentile (lower percentile = stricter)
    density_threshold = float(np.percentile(density_scores, 100 - percentile))

    print(f"✓ Density threshold calibrated: {density_threshold:.4f}")
    print(f"  Mean density score: {density_scores.mean():.4f}")
    print(f"  Std density score:  {density_scores.std():.4f}")

    # For ensemble threshold, we'll use a default value since we don't have ensemble predictions
    ensemble_threshold = 0.5  # Default

    return density_threshold, ensemble_threshold


def main():
    print("=" * 60)
    print("Phase 3 Training: Novelty Detector")
    print("=" * 60)
    print()

    # Configuration
    config = {
        'mol_dim': 768,
        'rxn_dim': 384,
        'context_dim': 256,
        'num_flow_layers': 6,
        'ensemble_size': 3,
        'num_epochs': 100,
        'batch_size': 128,
        'learning_rate': 1e-4,
        'threshold_percentile': 95,
    }

    print("Configuration:")
    for k, v in config.items():
        print(f"  {k:25s}: {v}")
    print()

    # Load transition data
    data_path = project_root / 'data' / 'phase3_transitions.pt'

    if not data_path.exists():
        raise FileNotFoundError(
            f"Transition data not found at {data_path}. "
            "Please run generate_phase3_data.py first."
        )

    print(f"Loading transition data from {data_path}")
    data = torch.load(data_path, weights_only=False)

    train_transitions = data['train_transitions']
    val_transitions = data['val_transitions']

    print(f"✓ Dataset loaded:")
    print(f"  Train: {len(train_transitions)} transitions")
    print(f"  Val:   {len(val_transitions)} transitions")
    print()

    # Extract latent states
    print("Extracting latent states...")
    train_states = extract_latent_states(train_transitions)
    val_states = extract_latent_states(val_transitions)

    print(f"✓ Extracted states:")
    print(f"  Train: {len(train_states)} states")
    print(f"  Val:   {len(val_states)} states")

    # Create model
    print("\nCreating novelty detector...")
    detector = NoveltyDetector(
        mol_dim=config['mol_dim'],
        rxn_dim=config['rxn_dim'],
        context_dim=config['context_dim'],
        num_flow_layers=config['num_flow_layers'],
        ensemble_size=config['ensemble_size'],
    ).to(device)

    num_params = sum(p.numel() for p in detector.parameters())
    print(f"✓ Model created: {num_params:,} parameters")
    print()

    # Train normalizing flow on latent states
    print("=" * 60)
    print("Training normalizing flow...")
    print("=" * 60)
    print()

    # Move states to device
    train_states_device = [state.to(device) for state in train_states]

    losses = detector.train_density_model(
        train_states_device,
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
    )

    print()
    print(f"✓ Flow training complete")
    print(f"  Final loss: {losses[-1]:.4f}")
    print()

    # Calibrate thresholds
    density_threshold, ensemble_threshold = calibrate_thresholds(
        detector,
        val_states,
        percentile=config['threshold_percentile'],
    )

    # Set thresholds
    detector.set_thresholds(density_threshold, ensemble_threshold)

    print()
    print("✓ Thresholds set:")
    print(f"  Density threshold:  {density_threshold:.4f}")
    print(f"  Ensemble threshold: {ensemble_threshold:.4f}")
    print()

    # Save model
    checkpoint_dir = project_root / 'checkpoints' / 'production'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / 'best_novelty.pt'

    torch.save({
        'model_state_dict': detector.state_dict(),
        'config': config,
        'density_threshold': density_threshold,
        'ensemble_threshold': ensemble_threshold,
        'training_losses': losses,
    }, checkpoint_path)

    print(f"✓ Model saved to: {checkpoint_path}")
    print()

    # Test novelty detection
    print("=" * 60)
    print("Testing novelty detection...")
    print("=" * 60)
    print()

    detector.eval()

    # Test on a few validation states
    test_states = val_states[:10]
    novelty_counts = {'novel': 0, 'in_distribution': 0}

    with torch.no_grad():
        for i, state in enumerate(test_states):
            state = state.to(device)

            # Add batch dimension
            state_batch = LatentState(
                z_mol=state.z_mol.unsqueeze(0) if state.z_mol.dim() == 1 else state.z_mol,
                z_rxn=state.z_rxn.unsqueeze(0) if state.z_rxn.dim() == 1 else state.z_rxn,
                z_context=state.z_context.unsqueeze(0) if state.z_context.dim() == 1 else state.z_context,
            )

            result = detector.is_novel(state_batch)

            is_novel = result['is_novel'].item()
            density_score = result['density_score'].item()

            status = "Novel" if is_novel else "In-distribution"
            novelty_counts['novel' if is_novel else 'in_distribution'] += 1

            print(f"State {i+1}: {status:17s} (density: {density_score:7.4f})")

    print()
    print(f"Summary (out of {len(test_states)} test states):")
    print(f"  Novel:            {novelty_counts['novel']}")
    print(f"  In-distribution:  {novelty_counts['in_distribution']}")
    print()

    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Evaluate Phase 3:")
    print("     python3 evaluation/evaluate_phase3.py")
    print()
    print("  2. Launch web interface with Phase 3:")
    print("     ./launch.sh")
    print()


if __name__ == '__main__':
    main()
