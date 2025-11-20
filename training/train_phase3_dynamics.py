#!/usr/bin/env python3
"""
Phase 3 Training: Latent Dynamics Model

Trains the dynamics predictor to model state transitions in latent space.

Architecture:
    - DynamicsPredictor with learned reaction codebook
    - Factored transition model (z_{t+1} = z_t + Δz_rxn + Δz_env)
    - Heteroscedastic uncertainty estimation
    - Vector quantization for discrete reaction operators

Loss:
    - Prediction loss (NLL with heteroscedastic uncertainty)
    - VQ loss (vector quantization for reaction codebook)

Estimated training time: 1-2 hours on Apple M4 Pro (50-100 epochs)
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chemjepa.models.dynamics import DynamicsPredictor
from chemjepa.models.latent import LatentState

# Device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")


class TransitionDataset(Dataset):
    """Dataset of state transitions."""

    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        transition = self.transitions[idx]

        # Convert dicts to LatentState objects
        current_state = LatentState(
            z_mol=transition['current_state']['z_mol'],
            z_rxn=transition['current_state']['z_rxn'],
            z_context=transition['current_state']['z_context'],
        )

        next_state = LatentState(
            z_mol=transition['next_state']['z_mol'],
            z_rxn=transition['next_state']['z_rxn'],
            z_context=transition['next_state']['z_context'],
        )

        action = transition['action']

        return current_state, action, next_state


def collate_transitions(batch):
    """Collate function for transition batches."""
    current_states = [item[0] for item in batch]
    actions = [item[1] for item in batch]
    next_states = [item[2] for item in batch]

    # Stack into batched LatentStates
    current_state_batch = LatentState(
        z_mol=torch.stack([s.z_mol for s in current_states], dim=0),
        z_rxn=torch.stack([s.z_rxn for s in current_states], dim=0),
        z_context=torch.stack([s.z_context for s in current_states], dim=0),
    )

    next_state_batch = LatentState(
        z_mol=torch.stack([s.z_mol for s in next_states], dim=0),
        z_rxn=torch.stack([s.z_rxn for s in next_states], dim=0),
        z_context=torch.stack([s.z_context for s in next_states], dim=0),
    )

    actions_batch = torch.stack(actions, dim=0)

    return current_state_batch, actions_batch, next_state_batch


def train_epoch(model, data_loader, optimizer, device):
    """Train for one epoch."""
    model.train()

    total_pred_loss = 0.0
    total_vq_loss = 0.0
    num_batches = 0

    for current_state, action, next_state in tqdm(data_loader, desc="Training"):
        # Move to device
        current_state = current_state.to(device)
        action = action.to(device)
        next_state = next_state.to(device)

        # Forward pass
        optimizer.zero_grad()

        losses = model.predict_loss(
            current_state,
            action,
            next_state,
            reduction='mean',
        )

        # Total loss
        loss = losses['prediction_loss']
        if 'vq_loss' in losses:
            loss = loss + 0.25 * losses['vq_loss']  # VQ loss weight

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track losses
        total_pred_loss += losses['prediction_loss'].item()
        if 'vq_loss' in losses:
            total_vq_loss += losses['vq_loss'].item()
        num_batches += 1

    return {
        'prediction_loss': total_pred_loss / num_batches,
        'vq_loss': total_vq_loss / num_batches if total_vq_loss > 0 else 0.0,
    }


def validate_epoch(model, data_loader, device):
    """Validate for one epoch."""
    model.eval()

    total_pred_loss = 0.0
    total_vq_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for current_state, action, next_state in tqdm(data_loader, desc="Validation"):
            # Move to device
            current_state = current_state.to(device)
            action = action.to(device)
            next_state = next_state.to(device)

            # Forward pass
            losses = model.predict_loss(
                current_state,
                action,
                next_state,
                reduction='mean',
            )

            # Track losses
            total_pred_loss += losses['prediction_loss'].item()
            if 'vq_loss' in losses:
                total_vq_loss += losses['vq_loss'].item()
            num_batches += 1

    return {
        'prediction_loss': total_pred_loss / num_batches,
        'vq_loss': total_vq_loss / num_batches if total_vq_loss > 0 else 0.0,
    }


def main():
    print("=" * 60)
    print("Phase 3 Training: Latent Dynamics Model")
    print("=" * 60)
    print()

    # Hyperparameters
    config = {
        'mol_dim': 768,
        'rxn_dim': 384,
        'context_dim': 256,
        'action_dim': 256,
        'hidden_dim': 512,
        'num_reactions': 1000,
        'num_transformer_layers': 4,
        'batch_size': 128,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
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

    # Create datasets
    train_dataset = TransitionDataset(train_transitions)
    val_dataset = TransitionDataset(val_transitions)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_transitions,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_transitions,
        num_workers=0,
    )

    # Create model
    print("Creating dynamics model...")
    model = DynamicsPredictor(
        mol_dim=config['mol_dim'],
        rxn_dim=config['rxn_dim'],
        context_dim=config['context_dim'],
        num_reactions=config['num_reactions'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim'],
        num_transformer_layers=config['num_transformer_layers'],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {num_params:,} parameters")
    print()

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-6,
    )

    # Training loop
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    print()

    best_val_loss = float('inf')
    checkpoint_dir = project_root / 'checkpoints' / 'production'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 60)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metrics = validate_epoch(model, val_loader, device)

        # Update scheduler
        scheduler.step()

        # Print metrics
        print(f"\nTrain:")
        print(f"  Prediction loss: {train_metrics['prediction_loss']:.4f}")
        print(f"  VQ loss:         {train_metrics['vq_loss']:.4f}")

        print(f"\nValidation:")
        print(f"  Prediction loss: {val_metrics['prediction_loss']:.4f}")
        print(f"  VQ loss:         {val_metrics['vq_loss']:.4f}")

        print(f"\nLearning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print()

        # Save best model
        val_loss = val_metrics['prediction_loss']

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint_path = checkpoint_dir / 'best_dynamics.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
            }, checkpoint_path)

            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            print()

    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print()
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {checkpoint_dir / 'best_dynamics.pt'}")
    print()
    print("Next steps:")
    print("  1. Train novelty detector:")
    print("     python3 training/train_phase3_novelty.py")
    print()
    print("  2. Evaluate Phase 3:")
    print("     python3 evaluation/evaluate_phase3.py")
    print()


if __name__ == '__main__':
    main()
