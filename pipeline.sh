#!/bin/bash

# ============================================
# TANHA Pipeline Runner
# Entity-Grounded RL for Legal Summarization
# ============================================

set -e

echo "=========================================="
echo "TANHA: Legal Summarization RL Pipeline"
echo "=========================================="

# -------------------------------------------------
# 1. Create environment (optional)
# -------------------------------------------------

if ! command -v conda &> /dev/null
then
    echo "Conda not found. Skipping environment creation."
else
    echo "Creating conda environment..."

    conda create -y -n tanha python=3.10 || true
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate tanha
fi


# -------------------------------------------------
# 2. Install dependencies
# -------------------------------------------------

echo "Installing dependencies..."

pip install --upgrade pip

pip install \
torch \
transformers \
datasets \
trl \
sentence-transformers \
spacy \
scikit-learn \
wandb


# -------------------------------------------------
# 3. Download spaCy model
# -------------------------------------------------

echo "Downloading spaCy model..."

python -m spacy download en_core_web_trf || true


# -------------------------------------------------
# 4. Create folders
# -------------------------------------------------

mkdir -p data
mkdir -p cache
mkdir -p outputs


# -------------------------------------------------
# 5. Preprocess dataset
# -------------------------------------------------

echo "Running preprocessing..."

python preprocess_dataset.py


# -------------------------------------------------
# 6. Start RL training
# -------------------------------------------------

echo "Starting reinforcement learning training..."

python train_rl.py


# -------------------------------------------------
# 7. Finished
# -------------------------------------------------

echo "=========================================="
echo "Pipeline finished successfully."
echo "=========================================="
