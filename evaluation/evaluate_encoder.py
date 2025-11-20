#!/usr/bin/env python3
"""
Quick Evaluation Script for ChemJEPA Embeddings

Run after training completes to assess model quality.
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from chemjepa import ChemJEPA
from chemjepa.data.loaders import MolecularDataset, collate_molecular_batch
from torch.utils.data import DataLoader


def evaluate_embeddings(checkpoint_path, val_data_path, device='mps', num_samples=1000):
    """
    Evaluate quality of learned molecular embeddings.

    Args:
        checkpoint_path: Path to trained model checkpoint
        val_data_path: Path to validation CSV
        device: Device to run on
        num_samples: Number of validation samples to use
    """
    print("=" * 80)
    print("ChemJEPA Embedding Evaluation")
    print("=" * 80)

    # Load model
    print("\n[1/5] Loading model...")
    model = ChemJEPA(device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load validation data
    print("\n[2/5] Loading validation data...")
    val_dataset = MolecularDataset(
        val_data_path,
        smiles_column='smiles',
        use_3d=True
    )

    # Subsample if needed
    if len(val_dataset) > num_samples:
        indices = np.random.choice(len(val_dataset), num_samples, replace=False)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_molecular_batch
    )
    print(f"✓ Loaded {len(val_dataset)} validation molecules")

    # Extract embeddings
    print("\n[3/5] Extracting embeddings...")
    embeddings = []
    smiles_list = []

    with torch.no_grad():
        for batch in val_loader:
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

    embeddings = torch.cat(embeddings, dim=0)  # [N, 768]
    print(f"✓ Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # Compute intrinsic quality metrics
    print("\n[4/5] Computing embedding quality metrics...")

    # Variance
    variance = torch.var(embeddings, dim=0).mean().item()
    print(f"\nEmbedding Variance: {variance:.4f}")
    if variance > 0.15:
        print("  ✓ Good variance (>0.15) - latent space is not collapsed")
    else:
        print("  ⚠ Low variance (<0.15) - latent space may be collapsed")

    # Norm statistics
    norms = torch.norm(embeddings, dim=1)
    print(f"\nEmbedding Norms:")
    print(f"  Mean: {norms.mean():.4f}")
    print(f"  Std:  {norms.std():.4f}")
    print(f"  Min:  {norms.min():.4f}")
    print(f"  Max:  {norms.max():.4f}")

    # Compute molecular properties
    print("\n[5/5] Testing property prediction...")
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    properties = {
        'LogP': [],
        'TPSA': [],
        'MolWt': [],
        'NumHDonors': [],
        'NumHAcceptors': [],
    }

    valid_indices = []
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                properties['LogP'].append(Descriptors.MolLogP(mol))
                properties['TPSA'].append(Descriptors.TPSA(mol))
                properties['MolWt'].append(Descriptors.MolWt(mol))
                properties['NumHDonors'].append(Descriptors.NumHDonors(mol))
                properties['NumHAcceptors'].append(Descriptors.NumHAcceptors(mol))
                valid_indices.append(i)
        except:
            continue

    # Filter embeddings to valid molecules
    embeddings_valid = embeddings[valid_indices].numpy()

    print(f"\nProperty Prediction (Linear Probe):")
    print(f"{'Property':<15} {'R²':<8} {'MAE':<10} {'Status'}")
    print("-" * 50)

    results = {}
    for prop_name, prop_values in properties.items():
        if len(prop_values) < 50:
            continue

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings_valid,
            prop_values,
            test_size=0.2,
            random_state=42
        )

        # Train linear model
        model_probe = Ridge(alpha=1.0)
        model_probe.fit(X_train, y_train)

        # Evaluate
        y_pred = model_probe.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Determine status
        if prop_name == 'LogP':
            status = "✓ Good" if r2 > 0.6 else "✓ OK" if r2 > 0.4 else "⚠ Low"
        elif prop_name in ['TPSA', 'MolWt']:
            status = "✓ Good" if r2 > 0.5 else "✓ OK" if r2 > 0.3 else "⚠ Low"
        else:
            status = "✓ Good" if r2 > 0.4 else "✓ OK" if r2 > 0.2 else "⚠ Low"

        print(f"{prop_name:<15} {r2:<8.4f} {mae:<10.4f} {status}")
        results[prop_name] = {'r2': r2, 'mae': mae}

    # Summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)

    avg_r2 = np.mean([v['r2'] for v in results.values()])
    print(f"\nAverage R² across properties: {avg_r2:.4f}")

    if avg_r2 > 0.6:
        print("✓ Excellent embeddings! Ready for downstream tasks.")
    elif avg_r2 > 0.4:
        print("✓ Good embeddings. Consider more training or Phase 2.")
    else:
        print("⚠ Embeddings need improvement. Consider:")
        print("  - Training for more epochs")
        print("  - Adjusting learning rate")
        print("  - Checking for overfitting")

    print("\n" + "=" * 80)

    return {
        'variance': variance,
        'norm_stats': {
            'mean': norms.mean().item(),
            'std': norms.std().item(),
        },
        'property_r2': results,
    }


if __name__ == '__main__':
    # Run evaluation
    results = evaluate_embeddings(
        checkpoint_path='checkpoints/best_jepa.pt',
        val_data_path='data/zinc250k/val.csv',
        device='mps',
        num_samples=1000
    )
