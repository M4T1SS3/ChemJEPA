#!/usr/bin/env python3
"""
Phase 2 Evaluation: Energy Model Assessment

Evaluates the energy model's ability to:
1. Score molecules against target properties
2. Optimize latent representations via gradient descent
3. Provide uncertainty estimates via ensemble
4. Decompose energy into interpretable components

Usage:
    python3 evaluation/evaluate_phase2.py
"""

import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chemjepa import ChemJEPA, ChemJEPAEnergyModel
from chemjepa.data.loaders import MolecularDataset, collate_molecular_batch
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors


def extract_embeddings_and_properties(model, data_loader, device, num_samples=500):
    """Extract embeddings and compute properties"""
    embeddings = []
    smiles_list = []
    properties = []

    model.eval()
    count = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting embeddings"):
            if count >= num_samples:
                break

            graph = batch['graph'].to(device)

            z_mol = model.encode_molecule(
                graph.x,
                graph.edge_index,
                graph.batch,
                graph.edge_attr if hasattr(graph, 'edge_attr') else None,
                graph.pos if hasattr(graph, 'pos') else None,
            )

            # Compute properties
            for smiles in batch['smiles']:
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
                        properties.append(props)
                        embeddings.append(z_mol[len(smiles_list) % z_mol.shape[0]].cpu())
                        smiles_list.append(smiles)
                except:
                    continue

            count += len(batch['smiles'])

    embeddings = torch.stack(embeddings[:num_samples])
    properties = torch.tensor(properties[:num_samples], dtype=torch.float32)

    return embeddings, smiles_list[:num_samples], properties


def evaluate_energy_model(
    phase1_checkpoint: str,
    phase2_checkpoint: str,
    val_data_path: str,
    device: str = 'mps',
    num_samples: int = 500
):
    """Comprehensive energy model evaluation"""

    print("=" * 80)
    print("Phase 2: Energy Model Evaluation")
    print("=" * 80)

    # Load Phase 1 model
    print("\n[1/5] Loading Phase 1 model...")
    phase1_model = ChemJEPA(device=device)
    checkpoint = torch.load(phase1_checkpoint, map_location=device, weights_only=False)

    # Filter out Phase 2/3 components
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith('energy_model.') and not k.startswith('imagination_engine.')
    }
    phase1_model.load_state_dict(filtered_state_dict, strict=False)
    phase1_model.eval()
    print("✓ Phase 1 model loaded")

    # Load Phase 2 energy model
    print("\n[2/5] Loading Phase 2 energy model...")
    energy_model = ChemJEPAEnergyModel(
        mol_dim=768,
        hidden_dim=512,
        num_properties=5,
        use_ensemble=True,
        ensemble_size=3
    ).to(device)

    phase2_checkpoint_data = torch.load(phase2_checkpoint, map_location=device, weights_only=False)
    energy_model.load_state_dict(phase2_checkpoint_data['model_state_dict'])
    energy_model.eval()
    print("✓ Energy model loaded")

    # Load validation data
    print("\n[3/5] Loading validation data...")
    val_dataset = MolecularDataset(
        val_data_path,
        smiles_column='smiles',
        use_3d=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_molecular_batch
    )
    print(f"✓ Loaded {len(val_dataset)} validation molecules")

    # Extract embeddings and properties
    print(f"\n[4/5] Extracting embeddings and computing properties...")
    embeddings, smiles_list, properties = extract_embeddings_and_properties(
        phase1_model, val_loader, device, num_samples
    )
    print(f"✓ Extracted {len(embeddings)} embeddings")

    # Normalize properties (same as training)
    property_mean = properties.mean(dim=0)
    property_std = properties.std(dim=0) + 1e-6
    properties_normalized = (properties - property_mean) / property_std

    # Evaluation
    print("\n[5/5] Evaluating energy model...")
    print("=" * 80)

    # Test 1: Property prediction accuracy
    print("\n1. Property Prediction Accuracy")
    print("-" * 80)

    with torch.no_grad():
        embeddings_dev = embeddings.to(device)
        output = energy_model(embeddings_dev, return_components=True)
        predicted_props = output['predicted_properties'].cpu()

        # Denormalize predictions
        predicted_props_denorm = predicted_props * property_std + property_mean

        # Compute MAE and R² for each property
        from sklearn.metrics import r2_score, mean_absolute_error

        prop_names = ['LogP', 'TPSA', 'MolWt', 'NumHDonors', 'NumHAcceptors']
        print(f"{'Property':<15} {'R²':<10} {'MAE':<12} {'Status'}")
        print("-" * 50)

        for i, name in enumerate(prop_names):
            true_vals = properties[:, i].numpy()
            pred_vals = predicted_props_denorm[:, i].numpy()

            r2 = r2_score(true_vals, pred_vals)
            mae = mean_absolute_error(true_vals, pred_vals)

            status = "✓ Good" if r2 > 0.5 else "✓ OK" if r2 > 0.3 else "⚠ Low"
            print(f"{name:<15} {r2:<10.4f} {mae:<12.4f} {status}")

    # Test 2: Energy decomposition
    print("\n2. Energy Component Analysis")
    print("-" * 80)

    with torch.no_grad():
        # Sample random target
        target_idx = np.random.randint(0, len(embeddings))
        target_props = properties_normalized[target_idx:target_idx+1].to(device)

        # Compute energies for all molecules
        output = energy_model(embeddings_dev, target_props.expand(len(embeddings_dev), -1), return_components=True)

        components = output['components']
        print(f"Energy component statistics (targeting molecule {target_idx}):")
        print(f"  Binding:    {components['binding'].mean().item():>8.4f} ± {components['binding'].std().item():.4f}")
        print(f"  Stability:  {components['stability'].mean().item():>8.4f} ± {components['stability'].std().item():.4f}")
        print(f"  Properties: {components['properties'].mean().item():>8.4f} ± {components['properties'].std().item():.4f}")
        print(f"  Novelty:    {components['novelty'].mean().item():>8.4f} ± {components['novelty'].std().item():.4f}")
        print(f"  Total:      {output['energy'].mean().item():>8.4f} ± {output['energy'].std().item():.4f}")

    # Test 3: Uncertainty quantification
    print("\n3. Uncertainty Quantification")
    print("-" * 80)

    uncertainties = output['uncertainty'].cpu()
    print(f"Ensemble uncertainty statistics:")
    print(f"  Mean:   {uncertainties.mean().item():.4f}")
    print(f"  Median: {uncertainties.median().item():.4f}")
    print(f"  Std:    {uncertainties.std().item():.4f}")
    print(f"  Min:    {uncertainties.min().item():.4f}")
    print(f"  Max:    {uncertainties.max().item():.4f}")

    # Test 4: Latent optimization
    print("\n4. Latent Space Optimization")
    print("-" * 80)

    # Define target properties (drug-like molecule)
    target_properties_real = torch.tensor([[
        2.5,   # LogP (lipophilicity)
        60.0,  # TPSA (polar surface area)
        400.0, # MolWt (molecular weight)
        2.0,   # NumHDonors
        4.0    # NumHAcceptors
    ]], dtype=torch.float32)

    target_properties_norm = (target_properties_real - property_mean) / property_std
    target_properties_norm = target_properties_norm.to(device)

    # Start from random embedding
    z_init = embeddings[np.random.randint(0, len(embeddings))].unsqueeze(0).to(device)

    print(f"Target properties (drug-like molecule):")
    print(f"  LogP:          2.5")
    print(f"  TPSA:          60.0")
    print(f"  MolWt:         400.0")
    print(f"  NumHDonors:    2.0")
    print(f"  NumHAcceptors: 4.0")

    print(f"\nOptimizing latent representation...")
    z_opt, final_energy = energy_model.optimize_molecule(
        z_init,
        target_properties_norm,
        num_steps=100,
        lr=0.01
    )

    # Evaluate optimized embedding
    with torch.no_grad():
        initial_output = energy_model(z_init, target_properties_norm, return_components=True)
        final_output = energy_model(z_opt, target_properties_norm, return_components=True)

        initial_props = (initial_output['predicted_properties'] * property_std.to(device) + property_mean.to(device)).cpu()
        final_props = (final_output['predicted_properties'] * property_std.to(device) + property_mean.to(device)).cpu()

    print(f"\nOptimization results:")
    print(f"  Initial energy: {initial_output['energy'].item():>8.4f}")
    print(f"  Final energy:   {final_output['energy'].item():>8.4f}")
    print(f"  Improvement:    {initial_output['energy'].item() - final_output['energy'].item():>8.4f}")

    print(f"\nPredicted properties after optimization:")
    prop_names = ['LogP', 'TPSA', 'MolWt', 'NumHDonors', 'NumHAcceptors']
    target_vals = target_properties_real[0].tolist()
    print(f"{'Property':<15} {'Target':<10} {'Initial':<10} {'Optimized':<10} {'Improvement'}")
    print("-" * 65)
    for i, name in enumerate(prop_names):
        target = target_vals[i]
        initial = initial_props[0, i].item()
        final = final_props[0, i].item()

        initial_error = abs(initial - target)
        final_error = abs(final - target)
        improvement = ((initial_error - final_error) / initial_error * 100) if initial_error > 0 else 0

        print(f"{name:<15} {target:<10.2f} {initial:<10.2f} {final:<10.2f} {improvement:>6.1f}%")

    # Summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)

    avg_r2 = np.mean([
        r2_score(properties[:, i].numpy(), predicted_props_denorm[:, i].numpy())
        for i in range(5)
    ])

    print(f"\nProperty Prediction: R² = {avg_r2:.4f}")
    print(f"Uncertainty Estimation: Mean = {uncertainties.mean().item():.4f}")
    print(f"Latent Optimization: Energy reduction = {initial_output['energy'].item() - final_output['energy'].item():.4f}")

    if avg_r2 > 0.5:
        print("\n✓ Energy model shows good performance!")
        print("  Ready for Phase 3 (Planning with MCTS)")
    elif avg_r2 > 0.3:
        print("\n✓ Energy model shows moderate performance")
        print("  Consider:")
        print("    - Training for more epochs")
        print("    - Increasing training data size")
        print("    - Tuning hyperparameters")
    else:
        print("\n⚠ Energy model needs improvement")
        print("  Recommend re-training with adjusted settings")

    print("=" * 80)


if __name__ == '__main__':
    evaluate_energy_model(
        phase1_checkpoint='checkpoints/best_jepa.pt',
        phase2_checkpoint='checkpoints/production/best_energy.pt',
        val_data_path='data/zinc250k/val.csv',
        device='mps',
        num_samples=500
    )
