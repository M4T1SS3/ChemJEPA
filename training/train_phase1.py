#!/usr/bin/env python3
"""
ChemJEPA Production Training Pipeline

Trains on ZINC250k dataset to create production-ready molecular discovery model.
Estimated time: 2-4 hours on M4 Pro Mac, 1 hour on A100 GPU.
"""

import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import subprocess

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from chemjepa import ChemJEPA
from chemjepa.data.loaders import MolecularDataset, collate_molecular_batch
from chemjepa.training.trainer import ChemJEPATrainer
from chemjepa.training.objectives import ChemJEPALoss


def get_device():
    """Auto-detect best available device"""
    if torch.backends.mps.is_available():
        return torch.device("mps"), "MPS (Apple GPU)"
    elif torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        return torch.device("cpu"), "CPU"


def download_zinc250k():
    """Download ZINC250k dataset if not present"""
    zinc_dir = project_root / "data" / "zinc250k"
    train_csv = zinc_dir / "train.csv"

    if train_csv.exists():
        print(f"✓ ZINC250k already downloaded at {zinc_dir}")
        return True

    print("\n" + "=" * 80)
    print("Downloading ZINC250k dataset...")
    print("=" * 80)
    print("This will download ~500MB and may take 5-10 minutes.")
    print()

    try:
        # Run data preparation script
        result = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "prepare_data.py"), "--dataset", "zinc250k"],
            capture_output=False,
            text=True
        )

        if result.returncode == 0 and train_csv.exists():
            print("\n✓ ZINC250k downloaded successfully!")
            return True
        else:
            print("\n✗ Download failed. Please run manually:")
            print("  python3 scripts/prepare_data.py")
            print("  Select option 2 for ZINC250k")
            return False

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("Please run manually: python3 scripts/prepare_data.py")
        return False


def main():
    print("=" * 80)
    print("ChemJEPA Production Training Pipeline")
    print("=" * 80)

    # Configuration
    config = {
        # Model architecture
        "mol_dim": 768,
        "rxn_dim": 384,
        "context_dim": 256,

        # Training settings
        "batch_size": 16,           # Reduced for stability (was 32)
        "num_epochs": 1,            # Just 1 epoch for testing (~3 hours)
        "learning_rate": 5e-6,      # Very conservative (was 1e-5)
        "weight_decay": 0.0,        # No weight decay for now
        "num_workers": 0,           # 0 for MPS, 4+ for CUDA

        # Paths
        "data_dir": project_root / "data" / "zinc250k",
        "checkpoint_dir": project_root / "checkpoints" / "production",

        # Training
        "phase": "jepa",            # Phase 1: JEPA pretraining
        "save_every": 10,           # Save checkpoint every N epochs
        "validate_every": 5,        # Validate every N epochs

        # Logging
        "use_wandb": False,
    }

    # Device selection
    device, device_name = get_device()
    print(f"\nDevice: {device_name}")

    # Adjust batch size for device
    if str(device) == "cpu":
        print("⚠ Warning: Training on CPU will be slow. Consider using GPU.")
        config["batch_size"] = 16
    elif str(device) == "mps":
        print("✓ MPS acceleration enabled")
        config["num_workers"] = 0  # Required for MPS
        config["batch_size"] = 32
    else:  # CUDA
        print("✓ CUDA acceleration enabled")
        config["num_workers"] = 4
        config["batch_size"] = 64

    print(f"Batch size: {config['batch_size']}")

    # Download dataset
    if not download_zinc250k():
        return 1

    # Create checkpoint directory
    config["checkpoint_dir"].mkdir(exist_ok=True, parents=True)

    # Load datasets
    print("\n" + "=" * 80)
    print("Loading ZINC250k dataset...")
    print("=" * 80)

    train_csv = config["data_dir"] / "train.csv"
    val_csv = config["data_dir"] / "val.csv"

    if not train_csv.exists():
        print(f"✗ Dataset not found at {train_csv}")
        print("Run: python3 scripts/prepare_data.py")
        return 1

    try:
        train_dataset = MolecularDataset(
            data_path=str(train_csv),
            smiles_column='smiles',
            use_3d=True
        )

        val_dataset = MolecularDataset(
            data_path=str(val_csv),
            smiles_column='smiles',
            use_3d=True
        )

        print(f"✓ Train: {len(train_dataset):,} molecules")
        print(f"✓ Val:   {len(val_dataset):,} molecules")

    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return 1

    # Create data loaders
    print("\nCreating data loaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=(str(device) == "cuda"),
        collate_fn=collate_molecular_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=(str(device) == "cuda"),
        collate_fn=collate_molecular_batch
    )

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches:   {len(val_loader)}")

    # Initialize model
    print("\n" + "=" * 80)
    print("Initializing ChemJEPA model...")
    print("=" * 80)

    try:
        model = ChemJEPA(
            mol_dim=config["mol_dim"],
            rxn_dim=config["rxn_dim"],
            context_dim=config["context_dim"],
            device=device
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"✓ Total parameters:     {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")

        # Memory estimate
        model_memory_gb = (total_params * 4) / (1024**3)
        print(f"\nEstimated memory usage:")
        print(f"  Model:     ~{model_memory_gb:.1f} GB")
        print(f"  Training:  ~{model_memory_gb * 3:.1f} GB (with gradients & optimizer)")

    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return 1

    # Initialize training components
    print("\nInitializing training components...")

    criterion = ChemJEPALoss(
        lambda_jepa=1.0,
        lambda_energy=0.5,
        lambda_consistency=0.3,
        lambda_property=0.2
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["num_epochs"],
        eta_min=1e-6
    )

    trainer = ChemJEPATrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=str(config["checkpoint_dir"]),
        use_wandb=config["use_wandb"]
    )

    print("✓ Trainer initialized")

    # Training
    print("\n" + "=" * 80)
    print("Starting Production Training")
    print("=" * 80)
    print(f"Phase:        {config['phase'].upper()}")
    print(f"Epochs:       {config['num_epochs']}")
    print(f"Batch size:   {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Device:       {device_name}")
    print("=" * 80)

    # Estimated time
    if str(device) == "mps":
        est_time = "2-4 hours"
    elif str(device) == "cuda":
        est_time = "1-2 hours"
    else:
        est_time = "8-12 hours"

    print(f"\nEstimated training time: {est_time}")
    print("\n💡 Training Tips:")
    if str(device) == "mps":
        print("  • Keep your Mac plugged in")
        print("  • Close other apps to free up RAM")
        print("  • Model checkpoints saved every 10 epochs")
    print("  • Press Ctrl+C to safely stop and save")
    print("  • Training will resume from last checkpoint")
    print()

    try:
        # Train
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config["num_epochs"],
            phase=config["phase"]
        )

        # Save final model
        final_path = config["checkpoint_dir"] / "chemjepa_final.pt"
        torch.save(model.state_dict(), final_path)

        print("\n" + "=" * 80)
        print("✓ Training Complete!")
        print("=" * 80)
        print(f"\nFinal model saved to: {final_path}")
        print(f"Best model: {config['checkpoint_dir']}/best_{config['phase']}.pt")
        print(f"All checkpoints: {config['checkpoint_dir']}/")

        print("\n" + "=" * 80)
        print("Next Steps:")
        print("=" * 80)
        print("1. Load the model:")
        print("   from chemjepa import ChemJEPA")
        print("   model = ChemJEPA(device='mps')")
        print(f"   model.load_state_dict(torch.load('{final_path}'))")
        print()
        print("2. Discover molecules:")
        print("   results = model.imagine(target_properties={'IC50': '<10nM'})")
        print()
        print("3. Train Phase 2 (Energy):")
        print("   Edit config['phase'] = 'energy' and re-run")
        print()
        print("Congratulations! 🎉")
        print("=" * 80)

        return 0

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Training interrupted by user")
        print("=" * 80)
        print(f"Latest checkpoint saved to: {config['checkpoint_dir']}")
        print("Resume training by running this script again")
        return 0

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
