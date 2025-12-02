#!/bin/bash
# Installation script for BruhVector
# Installs torch first, then other dependencies

set -e

echo "Installing BruhVector dependencies..."
echo ""

# Detect if we're on Mac (CPU-only)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS - using CPU-only installation"
    echo ""
fi

# Install torch first
echo "Step 1: Installing torch..."
pip install torch>=2.0.0

# Then install the rest (without unsloth/xformers for Mac)
echo ""
echo "Step 2: Installing remaining dependencies..."
pip install -r requirements.txt

echo ""
echo "✓ Installation complete!"
echo ""
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "⚠️  Note: You're running on macOS."
    echo "   - Data processing scripts (download, process, metadata) are ready to use"
    echo "   - Fine-tuning requires unsloth, which doesn't build on Mac"
    echo "   - For fine-tuning, use a GPU service: Google Colab, Kaggle, or cloud GPU"
    echo "   - The fine_tune.py script is included for use on GPU environments"
    echo ""
fi
echo "Next steps:"
echo "1. Download NLTK data: python -c \"import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')\""
echo "2. Run the pipeline: python scripts/run_pipeline.py"

