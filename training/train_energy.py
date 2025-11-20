#!/usr/bin/env python3
"""
Phase 2 Training: Energy Model

Trains the energy-based compatibility model on top of frozen Phase 1 embeddings.

Key features:
    - Frozen molecular encoder from Phase 1
    - Learns to score molecules against multiple objectives
    - Contrastive learning with property-based ranking
    - Ensemble uncertainty quantification

Estimated training time: 30-60 minutes on Apple M4 Pro
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from chemjepa import ChemJEPA
from chemjepa.models.energy import ChemJEPAEnergyModel, EnergyContrastiveLoss
from chemjepa.data.loaders import MolecularDataset, collate_molecular_batch
from rdkit import Chem
from rdkit.Chem import Descriptors


class PropertyDataset(Dataset):
    """
    Dataset that pairs molecular embeddings with their properties.

    Properties computed:
        - LogP: Lipophilicity
        - TPSA: Topological polar surface area
        - MolWt: Molecular weight
        - NumHDonors: Hydrogen bond donors
        - NumHAcceptors: Hydrogen bond acceptors
    """

    def __init__(self, embeddings: torch.Tensor, smiles_list: list):
        self.embeddings = embeddings
        self.smiles_list = smiles_list

        # Compute properties for all molecules
        self.properties = []
        self.valid_indices = []

        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    props = [
                        Descriptors.MolLogP(mol),
                        Descriptors.TPSA(mol),
                        Descriptors.MolWt(mol),
                        Descriptors.NumHDonors(mol),
                        Descriptors.NumHAcceptors(mol),
                    ]
                    self.properties.append(props)
                    self.valid_indices.append(i)
            except:
                continue

        self.properties = torch.tensor(self.properties, dtype=torch.float32)
        self.embeddings = self.embeddings[self.valid_indices]

        # Normalize properties (z-score normalization)
        self.property_mean = self.properties.mean(dim=0)
        self.property_std = self.properties.std(dim=0) + 1e-6
        self.properties_normalized = (self.properties - self.property_mean) / self.property_std

        print(f"Property statistics:")
        print(f"  LogP:         {self.properties[:, 0].mean():.2f} ± {self.properties[:, 0].std():.2f}")
        print(f"  TPSA:         {self.properties[:, 1].mean():.2f} ± {self.properties[:, 1].std():.2f}")
        print(f"  MolWt:        {self.properties[:, 2].mean():.2f} ± {self.properties[:, 2].std():.2f}")
        print(f"  NumHDonors:   {self.properties[:, 3].mean():.2f} ± {self.properties[:, 3].std():.2f}")
        print(f"  NumHAcceptors:{self.properties[:, 4].mean():.2f} ± {self.properties[:, 4].std():.2f}")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'properties': self.properties[idx],
            'properties_normalized': self.properties_normalized[idx]
        }


def extract_embeddings(model, data_loader, device):
    """Extract embeddings from Phase 1 model (frozen)"""
    embeddings = []
    smiles_list = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting embeddings"):
            graph = batch['graph'].to(device)

            z_mol = model.encode_molecule(
                graph.x,
                graph.edge_index,
                graph.batch,
                graph.edge_attr if hasattr(graph, 'edge_attr') else None,
                graph.pos if hasattr(graph, 'pos') else None,
            )

            embeddings.append(z_mol.cpu())
            smiles_list.extend(batch['smiles'])

    return torch.cat(embeddings, dim=0), smiles_list


def train_energy_model(
    energy_model,
    train_loader,
    val_loader,
    num_epochs,
    device,
    checkpoint_dir
):
    """Train energy model with contrastive ranking loss"""

    optimizer = torch.optim.AdamW(
        energy_model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    # Loss function
    contrastive_loss_fn = EnergyContrastiveLoss(margin=0.5, temperature=0.1)
    mse_loss_fn = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        energy_model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            embeddings = batch['embedding'].to(device)
            properties = batch['properties'].to(device)
            properties_norm = batch['properties_normalized'].to(device)

            # Sample random target properties from batch
            target_idx = torch.randint(0, len(embeddings), (1,))
            target_props = properties_norm[target_idx]

            # Forward pass
            output = energy_model(embeddings, target_props.expand(len(embeddings), -1))

            # Contrastive loss: molecules closer to target should have lower energy
            contrastive_loss = contrastive_loss_fn(
                output['energy'],
                properties_norm,
                target_props.squeeze(0)
            )

            # Property prediction loss
            pred_props = output['predicted_properties']
            property_loss = mse_loss_fn(pred_props, properties_norm)

            # Total loss
            loss = contrastive_loss + 0.5 * property_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(energy_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({
                'loss': np.mean(train_losses[-100:]),
                'contrast': contrastive_loss.item(),
                'property': property_loss.item()
            })

        scheduler.step()

        # Validation
        energy_model.eval()
        val_losses = []
        val_property_losses = []

        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embedding'].to(device)
                properties = batch['properties'].to(device)
                properties_norm = batch['properties_normalized'].to(device)

                target_idx = torch.randint(0, len(embeddings), (1,))
                target_props = properties_norm[target_idx]

                output = energy_model(embeddings, target_props.expand(len(embeddings), -1))

                contrastive_loss = contrastive_loss_fn(
                    output['energy'],
                    properties_norm,
                    target_props.squeeze(0)
                )

                property_loss = mse_loss_fn(
                    output['predicted_properties'],
                    properties_norm
                )

                loss = contrastive_loss + 0.5 * property_loss
                val_losses.append(loss.item())
                val_property_losses.append(property_loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_val_prop_loss = np.mean(val_property_losses)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val loss:   {avg_val_loss:.4f}")
        print(f"  Val prop loss: {avg_val_prop_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = checkpoint_dir / "best_energy.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': energy_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.4f})")

    return energy_model


def main():
    print("=" * 80)
    print("Phase 2: Energy Model Training")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Paths
    phase1_checkpoint = project_root / "checkpoints" / "best_jepa.pt"
    checkpoint_dir = project_root / "checkpoints" / "production"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    data_dir = project_root / "data" / "zinc250k"

    # Configuration
    config = {
        'batch_size': 64,
        'num_epochs': 20,
        'num_train_samples': 5000,
        'num_val_samples': 1000,
    }

    print(f"\nConfiguration:")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Train samples: {config['num_train_samples']}")
    print(f"  Val samples: {config['num_val_samples']}")

    # Load Phase 1 model (frozen)
    print("\n" + "=" * 80)
    print("[1/5] Loading Phase 1 model...")
    print("=" * 80)

    phase1_model = ChemJEPA(device=device)
    checkpoint = torch.load(phase1_checkpoint, map_location=device)

    # Filter out energy_model and imagination_engine keys (Phase 2/3 components)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith('energy_model.') and not k.startswith('imagination_engine.')
    }

    phase1_model.load_state_dict(filtered_state_dict, strict=False)
    phase1_model.eval()

    # Freeze Phase 1
    for param in phase1_model.parameters():
        param.requires_grad = False

    print(f"✓ Loaded Phase 1 checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print("✓ Phase 1 model frozen")

    # Load datasets
    print("\n" + "=" * 80)
    print("[2/5] Loading datasets...")
    print("=" * 80)

    train_dataset = MolecularDataset(
        str(data_dir / "train.csv"),
        smiles_column='smiles',
        use_3d=True
    )

    val_dataset = MolecularDataset(
        str(data_dir / "val.csv"),
        smiles_column='smiles',
        use_3d=True
    )

    # Subsample for faster training
    if len(train_dataset) > config['num_train_samples']:
        indices = np.random.choice(len(train_dataset), config['num_train_samples'], replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    if len(val_dataset) > config['num_val_samples']:
        indices = np.random.choice(len(val_dataset), config['num_val_samples'], replace=False)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)

    train_loader_phase1 = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_molecular_batch
    )

    val_loader_phase1 = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_molecular_batch
    )

    print(f"✓ Train: {len(train_dataset)} molecules")
    print(f"✓ Val:   {len(val_dataset)} molecules")

    # Extract embeddings
    print("\n" + "=" * 80)
    print("[3/5] Extracting Phase 1 embeddings...")
    print("=" * 80)

    train_embeddings, train_smiles = extract_embeddings(phase1_model, train_loader_phase1, device)
    val_embeddings, val_smiles = extract_embeddings(phase1_model, val_loader_phase1, device)

    print(f"✓ Extracted {len(train_embeddings)} train embeddings")
    print(f"✓ Extracted {len(val_embeddings)} val embeddings")

    # Create property datasets
    print("\n" + "=" * 80)
    print("[4/5] Computing molecular properties...")
    print("=" * 80)

    print("\nTrain set:")
    train_prop_dataset = PropertyDataset(train_embeddings, train_smiles)

    print("\nVal set:")
    val_prop_dataset = PropertyDataset(val_embeddings, val_smiles)

    train_loader = DataLoader(
        train_prop_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )

    val_loader = DataLoader(
        val_prop_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    # Initialize energy model
    print("\n" + "=" * 80)
    print("[5/5] Initializing energy model...")
    print("=" * 80)

    energy_model = ChemJEPAEnergyModel(
        mol_dim=768,
        hidden_dim=512,
        num_properties=5,
        use_ensemble=True,
        ensemble_size=3
    ).to(device)

    total_params = sum(p.numel() for p in energy_model.parameters())
    print(f"✓ Energy model parameters: {total_params:,}")

    # Train
    print("\n" + "=" * 80)
    print("Training Energy Model")
    print("=" * 80)
    print(f"Estimated time: ~{config['num_epochs'] * 2} minutes\n")

    energy_model = train_energy_model(
        energy_model,
        train_loader,
        val_loader,
        num_epochs=config['num_epochs'],
        device=device,
        checkpoint_dir=checkpoint_dir
    )

    print("\n" + "=" * 80)
    print("✓ Phase 2 Training Complete!")
    print("=" * 80)
    print(f"\nBest model saved to: {checkpoint_dir}/best_energy.pt")
    print("\nYou can now:")
    print("  1. Use energy model for multi-objective optimization")
    print("  2. Optimize molecules in latent space via gradient descent")
    print("  3. Proceed to Phase 3 (Planning with MCTS)")
    print("\nExample usage:")
    print("  from chemjepa.models.energy import ChemJEPAEnergyModel")
    print("  energy_model = ChemJEPAEnergyModel().to('mps')")
    print("  energy_model.load_state_dict(torch.load('checkpoints/production/best_energy.pt')['model_state_dict'])")
    print("  z_opt, energy = energy_model.optimize_molecule(z_init, target_properties)")
    print("=" * 80)


if __name__ == '__main__':
    exit(main())
