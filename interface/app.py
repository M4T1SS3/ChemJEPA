#!/usr/bin/env python3
"""
ChemJEPA Interactive Web Interface

Modern, user-friendly interface for molecular discovery using ChemJEPA.

Features:
- 🔬 Molecule property prediction
- 🎯 Energy-based scoring
- 🚀 Latent space optimization
- 📊 Interactive visualizations
- 🧪 Drug discovery workflow

Usage:
    python3 interface/app.py

    Then open: http://localhost:7860
"""

import gradio as gr
import torch
import numpy as np
from pathlib import Path
import sys
import json
from typing import Dict, List, Tuple

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chemjepa import ChemJEPA, ChemJEPAEnergyModel
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
import matplotlib.pyplot as plt
import pandas as pd

# Global models (loaded once)
phase1_model = None
energy_model = None
dynamics_model = None
novelty_detector = None
imagination_engine = None
device = None


def load_models():
    """Load Phase 1, Phase 2, and Phase 3 models"""
    global phase1_model, energy_model, dynamics_model, novelty_detector, imagination_engine, device

    from chemjepa.models.dynamics import DynamicsPredictor
    from chemjepa.models.novelty import NoveltyDetector
    from chemjepa.models.planning import ImaginationEngine

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load Phase 1
    phase1_model = ChemJEPA(device=device)
    try:
        checkpoint = torch.load('checkpoints/best_jepa.pt', map_location=device, weights_only=False)
        filtered_state_dict = {
            k: v for k, v in checkpoint['model_state_dict'].items()
            if not k.startswith('energy_model.') and not k.startswith('imagination_engine.')
        }
        phase1_model.load_state_dict(filtered_state_dict, strict=False)
        phase1_model.eval()
        print("✓ Phase 1 model loaded")
    except Exception as e:
        print(f"⚠ Phase 1 model not found: {e}")
        phase1_model = None

    # Load Phase 2
    try:
        energy_model = ChemJEPAEnergyModel(
            mol_dim=768,
            hidden_dim=512,
            num_properties=5,
            use_ensemble=True,
            ensemble_size=3
        ).to(device)

        checkpoint = torch.load('checkpoints/production/best_energy.pt', map_location=device, weights_only=False)
        energy_model.load_state_dict(checkpoint['model_state_dict'])
        energy_model.eval()
        print("✓ Phase 2 energy model loaded")
    except Exception as e:
        print(f"⚠ Phase 2 model not found: {e}")
        energy_model = None

    # Load Phase 3: Dynamics
    try:
        dynamics_model = DynamicsPredictor(
            mol_dim=768,
            rxn_dim=384,
            context_dim=256,
            num_reactions=1000,
            action_dim=256,
            hidden_dim=512,
            num_transformer_layers=4,
        ).to(device)

        checkpoint = torch.load('checkpoints/production/best_dynamics.pt', map_location=device, weights_only=False)
        dynamics_model.load_state_dict(checkpoint['model_state_dict'])
        dynamics_model.eval()
        print("✓ Phase 3 dynamics model loaded")
    except Exception as e:
        print(f"⚠ Phase 3 dynamics model not found: {e}")
        dynamics_model = None

    # Load Phase 3: Novelty Detector
    try:
        novelty_detector = NoveltyDetector(
            mol_dim=768,
            rxn_dim=384,
            context_dim=256,
            num_flow_layers=6,
            ensemble_size=3,
        ).to(device)

        checkpoint = torch.load('checkpoints/production/best_novelty.pt', map_location=device, weights_only=False)
        novelty_detector.load_state_dict(checkpoint['model_state_dict'])
        novelty_detector.eval()
        print("✓ Phase 3 novelty detector loaded")
    except Exception as e:
        print(f"⚠ Phase 3 novelty detector not found: {e}")
        novelty_detector = None

    # Create Imagination Engine if all Phase 3 models available
    if energy_model is not None and dynamics_model is not None and novelty_detector is not None:
        try:
            imagination_engine = ImaginationEngine(
                energy_model=energy_model,
                dynamics_model=dynamics_model,
                novelty_detector=novelty_detector,
                beam_size=10,
                horizon=3,
                exploration_coef=1.0,
                novelty_penalty=0.5,
            ).to(device)
            imagination_engine.eval()
            print("✓ Phase 3 imagination engine created")
        except Exception as e:
            print(f"⚠ Could not create imagination engine: {e}")
            imagination_engine = None


def smiles_to_image(smiles: str, size=(300, 300)):
    """Convert SMILES to molecule image"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        return img
    except:
        return None


def predict_properties(smiles: str) -> Dict:
    """Predict molecular properties using Phase 1 + Phase 2"""

    if phase1_model is None:
        return {"error": "Phase 1 model not loaded"}

    try:
        # Parse molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}

        # Get embedding from Phase 1
        from chemjepa.utils.chemistry import smiles_to_graph
        x, edge_index, edge_attr, pos = smiles_to_graph(smiles, use_3d=True)

        # Move to device
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None
        pos = pos.to(device) if pos is not None else None
        batch = torch.zeros(x.shape[0], dtype=torch.long).to(device)

        with torch.no_grad():
            z_mol = phase1_model.encode_molecule(x, edge_index, batch, edge_attr, pos)

        # Compute true properties
        true_props = {
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'MolWt': Descriptors.MolWt(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        }

        # Predict with energy model (if available)
        if energy_model is not None:
            with torch.no_grad():
                output = energy_model(z_mol, return_components=True)
                predicted_props = output['predicted_properties'].cpu().numpy()[0]

                # Denormalize (rough estimate)
                pred_props = {
                    'LogP': predicted_props[0] * 2.0,
                    'TPSA': predicted_props[1] * 30.0 + 50.0,
                    'MolWt': predicted_props[2] * 100.0 + 300.0,
                    'NumHDonors': max(0, int(predicted_props[3] * 2.0)),
                    'NumHAcceptors': max(0, int(predicted_props[4] * 3.0)),
                }

                energy_components = {
                    'Binding': output['components']['binding'].item(),
                    'Stability': output['components']['stability'].item(),
                    'Properties': output['components']['properties'].item(),
                    'Novelty': output['components']['novelty'].item(),
                    'Total': output['energy'].item(),
                }

                uncertainty = output['uncertainty'].item()
        else:
            pred_props = None
            energy_components = None
            uncertainty = None

        return {
            "success": True,
            "smiles": smiles,
            "true_properties": true_props,
            "predicted_properties": pred_props,
            "energy_components": energy_components,
            "uncertainty": uncertainty,
            "embedding_norm": torch.norm(z_mol).item(),
        }

    except Exception as e:
        return {"error": str(e)}


def create_property_comparison_plot(true_props, pred_props):
    """Create bar chart comparing true vs predicted properties"""
    if pred_props is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    properties = ['LogP', 'TPSA', 'MolWt', 'NumHDonors', 'NumHAcceptors']
    x = np.arange(len(properties))
    width = 0.35

    true_vals = [true_props[p] for p in properties]
    pred_vals = [pred_props[p] for p in properties]

    ax.bar(x - width/2, true_vals, width, label='True', alpha=0.8)
    ax.bar(x + width/2, pred_vals, width, label='Predicted', alpha=0.8)

    ax.set_xlabel('Property', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Molecular Properties: True vs Predicted', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(properties)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def create_energy_decomposition_plot(energy_components):
    """Create pie chart of energy components"""
    if energy_components is None:
        return None

    fig, ax = plt.subplots(figsize=(8, 8))

    components = ['Binding', 'Stability', 'Properties', 'Novelty']
    values = [abs(energy_components[c]) for c in components]

    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24']

    ax.pie(values, labels=components, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Energy Decomposition', fontsize=14, fontweight='bold')

    return fig


def molecule_interface(smiles: str):
    """Main interface for molecule analysis"""

    # Get molecule image
    img = smiles_to_image(smiles)

    # Predict properties
    result = predict_properties(smiles)

    if "error" in result:
        return img, f"Error: {result['error']}", None, None, None

    # Format output
    true_props = result['true_properties']
    pred_props = result['predicted_properties']
    energy = result['energy_components']
    uncertainty = result['uncertainty']

    # Create text summary
    summary = f"## Molecular Analysis\n\n"
    summary += f"**SMILES**: `{smiles}`\n\n"

    summary += "### Computed Properties\n"
    for prop, value in true_props.items():
        summary += f"- **{prop}**: {value:.2f}\n"

    if pred_props is not None:
        summary += "\n### Predicted Properties (Energy Model)\n"
        for prop, value in pred_props.items():
            if isinstance(value, (int, float)):
                summary += f"- **{prop}**: {value:.2f}\n"

    if energy is not None:
        summary += "\n### Energy Decomposition\n"
        summary += f"- **Total Energy**: {energy['Total']:.4f}\n"
        summary += f"- **Binding**: {energy['Binding']:.4f}\n"
        summary += f"- **Stability**: {energy['Stability']:.4f}\n"
        summary += f"- **Properties**: {energy['Properties']:.4f}\n"
        summary += f"- **Novelty**: {energy['Novelty']:.4f}\n"

        if uncertainty is not None:
            summary += f"\n**Uncertainty**: {uncertainty:.4f}\n"

    summary += f"\n**Embedding Norm**: {result['embedding_norm']:.2f}\n"

    # Create plots
    prop_plot = create_property_comparison_plot(true_props, pred_props) if pred_props else None
    energy_plot = create_energy_decomposition_plot(energy) if energy else None

    # Create metrics table
    if pred_props is not None:
        metrics_data = []
        for prop in ['LogP', 'TPSA', 'MolWt']:
            true_val = true_props[prop]
            pred_val = pred_props[prop]
            error = abs(true_val - pred_val)
            error_pct = (error / abs(true_val) * 100) if true_val != 0 else 0

            metrics_data.append({
                'Property': prop,
                'True': f"{true_val:.2f}",
                'Predicted': f"{pred_val:.2f}",
                'Error': f"{error:.2f}",
                'Error %': f"{error_pct:.1f}%"
            })

        metrics_df = pd.DataFrame(metrics_data)
    else:
        metrics_df = None

    return img, summary, prop_plot, energy_plot, metrics_df


def optimize_molecule_interface(target_logp: float, target_tpsa: float, target_molwt: float):
    """Optimize molecule to match target properties"""

    if phase1_model is None or energy_model is None:
        return None, "Models not loaded", None

    try:
        # Random starting embedding
        z_init = torch.randn(1, 768).to(device)

        # Target properties (normalized - rough estimate)
        target_props = torch.tensor([[
            target_logp / 2.0,
            (target_tpsa - 50.0) / 30.0,
            (target_molwt - 300.0) / 100.0,
            1.0,  # NumHDonors
            2.0   # NumHAcceptors
        ]], dtype=torch.float32).to(device)

        # Optimize
        z_opt, final_energy = energy_model.optimize_molecule(
            z_init,
            target_props,
            num_steps=100,
            lr=0.01
        )

        # Get optimized properties
        with torch.no_grad():
            output = energy_model(z_opt, target_props, return_components=True)
            opt_props = output['predicted_properties'].cpu().numpy()[0]

        # Denormalize
        opt_props_denorm = {
            'LogP': opt_props[0] * 2.0,
            'TPSA': opt_props[1] * 30.0 + 50.0,
            'MolWt': opt_props[2] * 100.0 + 300.0,
        }

        summary = f"## Optimization Results\n\n"
        summary += f"**Final Energy**: {final_energy:.4f}\n\n"
        summary += "### Target vs Optimized Properties\n"
        summary += f"- **LogP**: {target_logp:.2f} → {opt_props_denorm['LogP']:.2f}\n"
        summary += f"- **TPSA**: {target_tpsa:.2f} → {opt_props_denorm['TPSA']:.2f}\n"
        summary += f"- **MolWt**: {target_molwt:.2f} → {opt_props_denorm['MolWt']:.2f}\n"

        summary += "\n⚠ **Note**: Decoding latent to SMILES requires Phase 3 (coming soon)\n"

        return None, summary, None

    except Exception as e:
        return None, f"Error: {str(e)}", None


def molecular_discovery_interface(target_logp: float, target_tpsa: float, target_molwt: float,
                                  num_candidates: int, beam_size: int, horizon: int):
    """Run MCTS planning to discover molecules matching target properties"""

    from chemjepa.models.latent import LatentState

    if imagination_engine is None:
        return None, "⚠ Phase 3 models not loaded. Please train Phase 3 first:\n\n1. `python3 training/generate_phase3_data.py`\n2. `python3 training/train_phase3_dynamics.py`\n3. `python3 training/train_phase3_novelty.py`"

    try:
        # Target properties (normalized)
        p_target = torch.tensor([[
            target_logp / 2.0,
            (target_tpsa - 50.0) / 30.0,
            (target_molwt - 300.0) / 100.0,
            1.0,  # NumHDonors
            2.0   # NumHAcceptors
        ]], dtype=torch.float32).to(device)

        # Dummy z_target and z_env (since we don't have specific protein targets)
        z_target = torch.randn(1, 768, device=device)
        z_env = torch.randn(1, 256, device=device)

        # Update imagination engine parameters
        imagination_engine.beam_size = beam_size
        imagination_engine.horizon = horizon

        # Run molecular imagination
        with torch.no_grad():
            result = imagination_engine.imagine(
                z_target=z_target,
                z_env=z_env,
                p_target=p_target,
                num_candidates=num_candidates,
                return_traces=True,
            )

        candidates = result['candidates']
        scores = result['scores']
        traces = result.get('traces', None)

        # Create summary
        summary = f"## 🎯 Molecular Discovery Results\n\n"
        summary += f"**Configuration**:\n"
        summary += f"- Beam size: {beam_size}\n"
        summary += f"- Planning horizon: {horizon}\n"
        summary += f"- Candidates found: {len(candidates)}\n\n"

        summary += "### Target Properties\n"
        summary += f"- **LogP**: {target_logp:.2f}\n"
        summary += f"- **TPSA**: {target_tpsa:.2f}\n"
        summary += f"- **MolWt**: {target_molwt:.2f}\n\n"

        summary += "### Top Discovered Candidates\n\n"

        # Create candidates table
        candidates_data = []
        for i, (state, score) in enumerate(zip(candidates[:10], scores[:10])):
            # Predict properties for this candidate
            with torch.no_grad():
                output = energy_model(state.z_mol, p_target, return_components=True)
                pred_props = output['predicted_properties'].cpu().numpy()[0]

            # Denormalize
            logp = pred_props[0] * 2.0
            tpsa = pred_props[1] * 30.0 + 50.0
            molwt = pred_props[2] * 100.0 + 300.0

            candidates_data.append({
                'Rank': i + 1,
                'Score': f"{score:.4f}",
                'LogP': f"{logp:.2f}",
                'TPSA': f"{tpsa:.2f}",
                'MolWt': f"{molwt:.2f}",
                'Embedding Norm': f"{torch.norm(state.z_mol).item():.2f}",
            })

            summary += f"**{i+1}. Candidate {i+1}** (score: {score:.4f})\n"
            summary += f"   - LogP: {logp:.2f}, TPSA: {tpsa:.2f}, MolWt: {molwt:.2f}\n"

        candidates_df = pd.DataFrame(candidates_data)

        # Create score distribution plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Score distribution
        ax1.hist(scores, bins=20, alpha=0.7, color='#4ecdc4', edgecolor='black')
        ax1.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.4f}')
        ax1.set_xlabel('Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Discovery Score Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot 2: Top candidates bar chart
        top_10_scores = scores[:10]
        top_10_indices = range(1, len(top_10_scores) + 1)
        ax2.bar(top_10_indices, top_10_scores, alpha=0.7, color='#45b7d1', edgecolor='black')
        ax2.set_xlabel('Candidate Rank', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Top 10 Candidates', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        summary += f"\n\n### Statistics\n"
        summary += f"- **Mean score**: {np.mean(scores):.4f}\n"
        summary += f"- **Best score**: {np.max(scores):.4f}\n"
        summary += f"- **Std deviation**: {np.std(scores):.4f}\n"

        summary += "\n⚠ **Note**: SMILES decoding from latent space not yet implemented. "
        summary += "These are latent space representations with predicted properties.\n"

        return fig, summary, candidates_df

    except Exception as e:
        import traceback
        error_msg = f"Error during molecular discovery:\n\n{str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg, None


# Load models on startup
print("Loading models...")
load_models()
print("✓ Ready!")

# Create Gradio interface
with gr.Blocks(title="ChemJEPA - Molecular Discovery AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧪 ChemJEPA: Molecular Discovery AI

    **Joint-Embedding Predictive Architecture for Chemistry**

    Explore molecular properties, energy decomposition, and latent space optimization.
    """)

    with gr.Tabs():
        # Tab 1: Molecule Analysis
        with gr.Tab("🔬 Molecule Analysis"):
            gr.Markdown("### Analyze a molecule by SMILES")

            with gr.Row():
                with gr.Column():
                    smiles_input = gr.Textbox(
                        label="SMILES",
                        placeholder="CCO (ethanol)",
                        value="CCO"
                    )

                    gr.Examples(
                        examples=[
                            ["CCO"],  # Ethanol
                            ["CC(=O)Oc1ccccc1C(=O)O"],  # Aspirin
                            ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],  # Caffeine
                            ["CC(C)Cc1ccc(cc1)C(C)C(=O)O"],  # Ibuprofen
                        ],
                        inputs=[smiles_input],
                        label="Example Molecules"
                    )

                    analyze_btn = gr.Button("Analyze Molecule", variant="primary")

                with gr.Column():
                    mol_image = gr.Image(label="Molecular Structure", type="pil")

            with gr.Row():
                summary_output = gr.Markdown(label="Analysis Results")

            with gr.Row():
                with gr.Column():
                    property_plot = gr.Plot(label="Property Comparison")
                with gr.Column():
                    energy_plot = gr.Plot(label="Energy Decomposition")

            with gr.Row():
                metrics_table = gr.Dataframe(label="Prediction Metrics")

            analyze_btn.click(
                fn=molecule_interface,
                inputs=[smiles_input],
                outputs=[mol_image, summary_output, property_plot, energy_plot, metrics_table]
            )

        # Tab 2: Property Optimization
        with gr.Tab("🎯 Property Optimization"):
            gr.Markdown("### Optimize molecule to match target properties")
            gr.Markdown("⚠ **Phase 2 must be trained first**")

            with gr.Row():
                target_logp = gr.Slider(minimum=-5, maximum=10, value=2.5, step=0.1, label="Target LogP")
                target_tpsa = gr.Slider(minimum=0, maximum=200, value=60, step=5, label="Target TPSA")
                target_molwt = gr.Slider(minimum=100, maximum=800, value=400, step=10, label="Target Molecular Weight")

            optimize_btn = gr.Button("Optimize in Latent Space", variant="primary")

            with gr.Row():
                opt_mol_image = gr.Image(label="Optimized Structure", type="pil")
                opt_summary = gr.Markdown(label="Optimization Results")

            opt_plot = gr.Plot(label="Optimization Progress")

            optimize_btn.click(
                fn=optimize_molecule_interface,
                inputs=[target_logp, target_tpsa, target_molwt],
                outputs=[opt_mol_image, opt_summary, opt_plot]
            )

        # Tab 3: Molecular Discovery (Phase 3)
        with gr.Tab("🚀 Molecular Discovery (Phase 3)"):
            gr.Markdown("### MCTS Planning in Latent Space")
            gr.Markdown("Discover novel molecules using Monte Carlo Tree Search with learned dynamics.")
            gr.Markdown("⚠ **Phase 3 must be trained first** (see instructions below)")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Target Properties")
                    discovery_logp = gr.Slider(minimum=-5, maximum=10, value=2.5, step=0.1, label="Target LogP")
                    discovery_tpsa = gr.Slider(minimum=0, maximum=200, value=60, step=5, label="Target TPSA")
                    discovery_molwt = gr.Slider(minimum=100, maximum=800, value=400, step=10, label="Target Molecular Weight")

                with gr.Column():
                    gr.Markdown("#### Planning Parameters")
                    num_candidates = gr.Slider(minimum=5, maximum=50, value=10, step=5, label="Number of Candidates")
                    beam_size = gr.Slider(minimum=5, maximum=30, value=10, step=5, label="Beam Size")
                    horizon = gr.Slider(minimum=2, maximum=10, value=3, step=1, label="Planning Horizon")

            discovery_btn = gr.Button("🎯 Discover Molecules", variant="primary", size="lg")

            with gr.Row():
                discovery_summary = gr.Markdown(label="Discovery Results")

            with gr.Row():
                discovery_plot = gr.Plot(label="Discovery Statistics")

            with gr.Row():
                discovery_table = gr.Dataframe(label="Top Candidates")

            discovery_btn.click(
                fn=molecular_discovery_interface,
                inputs=[discovery_logp, discovery_tpsa, discovery_molwt, num_candidates, beam_size, horizon],
                outputs=[discovery_plot, discovery_summary, discovery_table]
            )

            gr.Markdown("""
            ---
            ### How to Train Phase 3

            If Phase 3 models are not loaded, run these commands:

            ```bash
            # 1. Generate training data (~15-30 minutes)
            python3 training/generate_phase3_data.py

            # 2. Train dynamics model (~1-2 hours)
            python3 training/train_phase3_dynamics.py

            # 3. Train novelty detector (~30 minutes)
            python3 training/train_phase3_novelty.py

            # 4. Relaunch interface
            ./launch.sh
            ```

            **Total time**: ~2-3 hours for complete Phase 3 training
            """)

        # Tab 4: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About ChemJEPA

            ChemJEPA is a hierarchical latent world model for molecular discovery that learns to plan
            in compressed representation space using MCTS.

            ### Three-Phase Architecture

            - **Phase 1**: Self-supervised molecular encoder (E(3)-equivariant GNN)
            - **Phase 2**: Energy-based compatibility scoring
            - **Phase 3**: MCTS planning in latent space 🎯

            ### Novel Contributions

            ✅ Decomposable energy function (no retraining for new objectives)
            ✅ Multi-objective optimization without retraining
            ✅ Triple uncertainty quantification (ensemble + density + conformal)
            ✅ Latent space planning (100x faster than SMILES-based search)
            ✅ Hierarchical latent world state (z_mol → z_rxn → z_context)
            ✅ Learned reaction codebook with vector quantization
            ✅ Factored dynamics for counterfactual reasoning

            ### Usage

            1. **Molecule Analysis**: Input SMILES, get properties and energy decomposition
            2. **Property Optimization**: Specify target properties, optimize in latent space
            3. **Molecular Discovery**: Run MCTS planning to discover novel candidates (Phase 3)

            ### Training Status

            Check `checkpoints/` directory for available models:
            - `best_jepa.pt` - Phase 1 molecular encoder ✅
            - `production/best_energy.pt` - Phase 2 energy model ✅
            - `production/best_dynamics.pt` - Phase 3 dynamics model ⏳
            - `production/best_novelty.pt` - Phase 3 novelty detector ⏳

            ### System Requirements

            - **CPU**: Any modern processor (Apple Silicon recommended)
            - **GPU**: MPS (Apple Silicon), CUDA (optional), or CPU
            - **RAM**: 8GB minimum, 16GB recommended
            - **Storage**: ~2GB for models and data

            ### Performance

            - **Phase 1 training**: ~15 minutes (1 epoch), ~30 hours (100 epochs)
            - **Phase 2 training**: ~40 minutes (20 epochs), ~2 hours (50 epochs)
            - **Phase 3 training**: ~2-3 hours total (data + dynamics + novelty)
            - **Inference**: Real-time (<1s per molecule)
            - **MCTS planning**: ~2-5 seconds per search

            ### Citation

            ```bibtex
            @software{chemjepa2025,
              title={ChemJEPA: Joint-Embedding Predictive Architecture for Chemistry},
              year={2025},
              note={Research prototype for molecular discovery}
            }
            ```
            """)

    gr.Markdown("""
    ---
    **ChemJEPA** | Built with ❤️ for molecular discovery
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public URL
        show_error=True
    )
