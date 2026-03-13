#!/bin/bash

# ============================================
# TANHA Pipeline Runner
# Entity-guided RL for Legal Summarization
# ============================================

set -e

echo "=========================================="
echo "Entity-guided RL Pipeline"
echo "=========================================="

# -------------------------------------------------
# 1. Environment setup
# -------------------------------------------------

echo "Checking Python environment..."

if ! command -v python &> /dev/null
then
    echo "Python not found. Please install Python >=3.10"
    exit
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
# 4. Prepare folders
# -------------------------------------------------

mkdir -p data
mkdir -p cache
mkdir -p outputs


# -------------------------------------------------
# 5. Dataset Instructions
# -------------------------------------------------

echo "=========================================="
echo "DATASET SETUP"
echo "=========================================="

echo ""
echo "IL-TUR dataset will be downloaded automatically."
echo ""

echo "MILDSum dataset cannot be redistributed."
echo "Please request access from the authors:"
echo ""
echo "  https://github.com/Exploration-Lab/MILDSum"
echo ""
echo "After obtaining the dataset, place files in:"
echo ""
echo "  data/train.json"
echo "  data/val.json"
echo ""


# -------------------------------------------------
# 6. Download IL-TUR dataset
# -------------------------------------------------

echo "Downloading IL-TUR dataset from HuggingFace..."

python <<EOF
from datasets import load_dataset
import json
import os

dataset = load_dataset("Exploration-Lab/IL-TUR")

os.makedirs("data", exist_ok=True)

with open("data/iltur_train.json","w") as f:
    for row in dataset["train"]:
        json.dump(row,f)
        f.write("\n")

with open("data/iltur_test.json","w") as f:
    for row in dataset["test"]:
        json.dump(row,f)
        f.write("\n")

print("IL-TUR dataset downloaded.")
EOF


# -------------------------------------------------
# 7. Preprocess dataset
# -------------------------------------------------

echo "Running preprocessing..."

python preprocess_dataset.py


# -------------------------------------------------
# 8. Start RL training
# -------------------------------------------------

echo "Starting RL training..."

python train_rl.py


# -------------------------------------------------
# 9. Finished
# -------------------------------------------------

echo "=========================================="
echo "Pipeline completed successfully."
echo "=========================================="
