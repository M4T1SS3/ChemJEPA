#!/bin/bash
# Complete Multi-Component Training Script
# Trains dynamics, novelty detector, and planning components

echo "============================================"
echo "🚀 ChemJEPA Multi-Component Training"
echo "============================================"
echo ""
echo "This will train the planning components:"
echo "  1. Generate transition data (~15-30 min)"
echo "  2. Train dynamics model (~1-2 hours)"
echo "  3. Train novelty detector (~30 min)"
echo "  4. Evaluate planning system (~10 min)"
echo ""
echo "Total estimated time: ~2-3 hours"
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if encoder and energy model are trained
if [ ! -f "checkpoints/encoder.pt" ] && [ ! -f "checkpoints/best_jepa.pt" ]; then
    echo "❌ Encoder checkpoint not found!"
    echo "Please train encoder first: python3 training/train_encoder.py"
    exit 1
fi

if [ ! -f "checkpoints/production/energy.pt" ] && [ ! -f "checkpoints/production/best_energy.pt" ]; then
    echo "❌ Energy model checkpoint not found!"
    echo "Please train energy model first: python3 training/train_energy.py"
    exit 1
fi

echo "✓ Encoder and energy model checkpoints found"
echo ""

# Step 1: Generate data
echo "============================================"
echo "Step 1/4: Generating transition data"
echo "============================================"
echo ""
python3 training/generate_dynamics_data.py

if [ $? -ne 0 ]; then
    echo "❌ Data generation failed"
    exit 1
fi

echo ""
echo "✓ Data generation complete"
echo ""

# Step 2: Train dynamics
echo "============================================"
echo "Step 2/4: Training dynamics model"
echo "============================================"
echo ""
python3 training/train_dynamics.py

if [ $? -ne 0 ]; then
    echo "❌ Dynamics training failed"
    exit 1
fi

echo ""
echo "✓ Dynamics training complete"
echo ""

# Step 3: Train novelty detector
echo "============================================"
echo "Step 3/4: Training novelty detector"
echo "============================================"
echo ""
python3 training/train_novelty.py

if [ $? -ne 0 ]; then
    echo "❌ Novelty detector training failed"
    exit 1
fi

echo ""
echo "✓ Novelty detector training complete"
echo ""

# Step 4: Evaluate
echo "============================================"
echo "Step 4/4: Evaluating planning system"
echo "============================================"
echo ""
python3 evaluation/evaluate_planning.py

if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed"
    exit 1
fi

echo ""
echo "============================================"
echo "✨ Multi-Component Training Complete!"
echo "============================================"
echo ""
echo "All planning components trained and evaluated successfully!"
echo ""
echo "Trained models:"
echo "  ✅ checkpoints/production/dynamics.pt"
echo "  ✅ checkpoints/production/novelty.pt"
echo ""
echo "Next steps:"
echo "  1. Launch the web interface:"
echo "     ./launch.sh"
echo ""
echo "  2. Try the 'Molecular Discovery' tab with MCTS planning!"
echo ""
echo "  3. Create demo materials for visiting researcher applications"
echo ""
