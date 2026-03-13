# Entity-Grounded Reinforcement Learning for Legal Summarization

This repository contains the official implementation of an entity-guided reinforcement learning framework for improving factual consistency in legal summarization.

The approach trains large language models using a **entity-guided reward** that encourages summaries to retain legally salient entities (e.g., courts, statutes, provisions, precedents) while discouraging unsupported entity generation.

---

# Overview

Legal judgments contain structured legal references such as:

- Courts  
- Judges  
- Statutes  
- Provisions  
- Precedents  
- Parties  

Human-written legal summaries selectively retain these entities while omitting less relevant procedural details.

TANHA explicitly models this behavior by:

1. Extracting entities from judgments and summaries  
2. Aligning generated entities with source entities  
3. Optimizing a reinforcement learning objective based on **entity grounding**

The reward function encourages:

- High **entity recall**
- Low **hallucinated entities**
- Balanced **entity precision**


---

# Installation

Create a Python environment and install dependencies.
conda create -n tanha python=3.10
conda activate tanha


Install required packages:
pip install torch transformers datasets trl sentence-transformers spacy scikit-learn wandb unsloth unsloth_zoo


Download SpaCy model:
python -m spacy download en_core_web_trf


(Optional) If you have the **legal NER model**:
python -m spacy download en_legal_ner_trf


---

# Dataset Format

Input dataset must contain **NER-tagged judgments and summaries**.

Example:
{
"NER_Judgment": "The <COURT>Delhi High Court</COURT> considered <PROVISION>Section 420 IPC</PROVISION>...",
"NER_Summary": "The <COURT>Delhi High Court</COURT> ruled under <PROVISION>Section 420 IPC</PROVISION>..."
}

Entities must follow the format:
<TYPE>entity text</TYPE>
Supported entity types:
COURT
JUDGE
STATUTE
PROVISION
PRECEDENT
PETITIONER
RESPONDENT


---

To reproduce the full pipeline:

bash run_pipeline.sh



# Step 1 — Preprocess Dataset

This script builds the **entity cache** used during RL training.

It extracts:

- canonicalized entities  
- entity metadata  
- entity embeddings  
- dataset-adaptive entity weights  

Run:


python preprocess_dataset.py


Outputs:


cache/train_entity_cache.pkl
cache/val_entity_cache.pkl
cache/entity_type_weights.pkl


---

# Step 2 — Train with Reinforcement Learning
The RL training starts from a supervised fine-tuned checkpoint.
Users may initialize from any instruction-tuned LLaMA/Qwen model.

Train the model using **GRPO optimization**.


python train_rl.py


Training optimizes a **factuality-aware entity reward** that encourages models to retain grounded entities while penalizing hallucinated entities.

---

# Entity-Guided Reward

For a generated summary:


E_J = entities in judgment
E_G = entities in generated summary


The reward consists of:

## Weighted Entity Recall

Encourages retaining legally salient entities:


WeightedRecall = Σ w_t * (|E_match,t| / |E_J,t|)


Where weights `w_t` are automatically derived from **dataset statistics**.

---

## Hallucination Penalty

Penalizes unsupported entities:


HallucinationRate = weighted_fp / weighted_pred


---

## Final Reward
Reward = alpha * WeightedRecall
- beta * HallucinationRate
+ gamma * EntityF1


---

# Hybrid Entity Matching

Entity alignment combines three signals:

## 1. Canonical normalization

Example:
Sec 420 IPC
Section 420
→ normalized to
section_420


---

## 2. Lexical similarity

Handles minor variations.

Example:
Delhi HC
High Court of Delhi


---

## 3. Embedding similarity

Handles paraphrased references.
Example:
Kesavananda Case
Kesavananda Bharati v State of Kerala


---

# Evaluation
Validation rewards are computed during training using the same entity reward.
The framework measures:
- Entity Recall  
- Hallucination Rate  
- Entity F1  

---

# Reproducibility

Set random seed: seed = 42
All experiments were run with:
LLaMA-based and Qwen-based instruction model
GRPO optimization
Entity-guided reward

---

# Hardware

Training was conducted on:


1× NVIDIA A30 25GB


The method also runs on smaller GPUs using **4-bit quantization**.

---

---

# License

This project is released for **research purposes only**.

---

# Contact

For questions regarding the implementation, please open an issue on the repository.
