import os
import re
import random
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import wandb
import spacy

from datasets import Dataset
from sentence_transformers import SentenceTransformer

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt

from trl import GRPOTrainer, GRPOConfig

from legal_structural_utils import (
    hybrid_match,
    canonicalize_provision,
    canonicalize_precedent,
    canonicalize_statute,
    canonicalize_court,
    normalize_legal_text
)

# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Paths
# =========================================================

DATA_DIR = "./data"
CACHE_DIR = "./cache"

TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
VAL_JSON = os.path.join(DATA_DIR, "val.json")

TRAIN_CACHE = os.path.join(CACHE_DIR, "train_entity_cache.pkl")
VAL_CACHE = os.path.join(CACHE_DIR, "val_entity_cache.pkl")

ENTITY_WEIGHTS = os.path.join(CACHE_DIR, "entity_type_weights.pkl")

# =========================================================
# WandB
# =========================================================

wandb.init(
    project="MILDSum_GRPO",
    name="entity_alignment_rl",
)

# =========================================================
# Load model
# =========================================================

max_seq_length = 16384
lora_rank = 64

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="llama_ft_base",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.9,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.2"
)

# =========================================================
# Dataset Loading
# =========================================================

train_df = pd.read_json(TRAIN_JSON, lines=True)
val_df = pd.read_json(VAL_JSON, lines=True)

train_df = train_df.reset_index()
val_df = val_df.reset_index()

train_df.rename(columns={"index": "id"}, inplace=True)
val_df.rename(columns={"index": "id"}, inplace=True)

# =========================================================
# Prompt
# =========================================================

PROMPT_TEMPLATE = """
Summarize the TARGET JUDGMENT provided by the user.

### STRUCTURE
Case Details, Background, Legal Issues, Arguments, Precedents, Judicial Reasoning, Final Decision.

### FACTUALITY RULE
All legal entities must appear in the judgment text.
Do not invent courts, statutes, judges or provisions.
"""

train_df["input"] = (
    "\n[SAMPLE_ID=" + train_df["id"].astype(str) + "]\n"
    + PROMPT_TEMPLATE
    + "\nTARGET JUDGMENT:\n"
    + train_df["EN_Judgment"]
    + "\nSUMMARY:\n"
)

val_df["input"] = (
    "\n[SAMPLE_ID=" + val_df["id"].astype(str) + "]\n"
    + PROMPT_TEMPLATE
    + "\nTARGET JUDGMENT:\n"
    + val_df["EN_Judgment"]
    + "\nSUMMARY:\n"
)

# =========================================================
# Convert to HF dataset
# =========================================================

def create_conversation(row):
    return [{"from": "human", "value": row["input"]}]

train_df["conversations"] = train_df.apply(create_conversation, axis=1)
val_df["conversations"] = val_df.apply(create_conversation, axis=1)

train_dataset = Dataset.from_pandas(train_df[["id","conversations"]])
val_dataset = Dataset.from_pandas(val_df[["id","conversations"]])

train_dataset = standardize_sharegpt(train_dataset)
val_dataset = standardize_sharegpt(val_dataset)

def format_prompt(example):
    text = tokenizer.apply_chat_template(
        example["conversations"],
        tokenize=False,
        add_generation_prompt=True
    )
    return {"prompt": text}

train_dataset = train_dataset.map(format_prompt)
val_dataset = val_dataset.map(format_prompt)

train_dataset = train_dataset.remove_columns(["conversations"])
val_dataset = val_dataset.remove_columns(["conversations"])

# =========================================================
# Load entity cache
# =========================================================

with open(TRAIN_CACHE, "rb") as f:
    train_entity_cache = pickle.load(f)

with open(VAL_CACHE, "rb") as f:
    val_entity_cache = pickle.load(f)

# =========================================================
# Load dataset-adaptive weights
# =========================================================

with open(ENTITY_WEIGHTS, "rb") as f:
    ENTITY_TYPE_WEIGHTS = pickle.load(f)

# =========================================================
# Entity extractor
# =========================================================

try:
    legal_nlp = spacy.load("en_legal_ner_trf")
except:
    legal_nlp = spacy.load("en_core_web_trf")

KEEP_LABELS = {
    "COURT","STATUTE","PROVISION",
    "PRECEDENT","PETITIONER",
    "RESPONDENT","JUDGE"
}

def extract_generated_entities(text):

    doc = legal_nlp(text)

    entities = []

    for ent in doc.ents:

        if ent.label_ not in KEEP_LABELS:
            continue

        entities.append(
            (ent.text.lower().strip(), ent.label_)
        )

    return entities

# =========================================================
# Embedding model
# =========================================================

embed_model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    device=device
)

def embed_entities(entity_list):

    if len(entity_list) == 0:
        return torch.empty((0,1024), device=device)

    texts = [e[0] for e in entity_list]

    emb = embed_model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    return emb

# =========================================================
# Canonicalization
# =========================================================

def canonicalize_entities(entity_list):

    result = []

    for text,etype in entity_list:

        if etype=="PROVISION":
            canon = canonicalize_provision(text)

        elif etype=="PRECEDENT":
            canon = canonicalize_precedent(text)

        elif etype=="STATUTE":
            canon = canonicalize_statute(text)

        elif etype=="COURT":
            canon = canonicalize_court(text)

        else:
            canon = normalize_legal_text(text)

        if canon is None:
            continue

        result.append((canon,etype))

    return list(dict.fromkeys(result))

# =========================================================
# Reward Function
# =========================================================

def compute_entity_reward(text, cache):

    source_entities = list(cache["source_metadata"].keys())
    source_emb = cache["source_embeddings"].to(device)

    gen_entities = extract_generated_entities(text)
    gen_entities = canonicalize_entities(gen_entities)

    if len(gen_entities)==0:
        return -2.0

    gen_emb = embed_entities(gen_entities)

    matched_gen, matched_src = hybrid_match(
        gen_entities,
        source_entities,
        gen_emb,
        source_emb
    )

    weighted_tp = 0
    weighted_total = 0

    for i,(txt,etype) in enumerate(source_entities):

        w = ENTITY_TYPE_WEIGHTS.get(etype,1.0)
        weighted_total += w

        if i in matched_src:
            weighted_tp += w

    weighted_recall = weighted_tp/(weighted_total+1e-8)

    weighted_fp = 0
    weighted_pred = 0

    for i,(txt,etype) in enumerate(gen_entities):

        w = ENTITY_TYPE_WEIGHTS.get(etype,1.0)
        weighted_pred += w

        if i not in matched_gen:
            weighted_fp += w

    halluc_rate = weighted_fp/(weighted_pred+1e-8)

    reward = 5*weighted_recall - 4*halluc_rate

    return reward

# =========================================================
# GRPO reward wrapper
# =========================================================

def extract_sample_id(prompt):

    match = re.search(r"\[SAMPLE_ID=(\d+)\]", prompt)

    return int(match.group(1))

def entity_reward(prompts, completions, **kwargs):

    rewards=[]

    for i in range(len(completions)):

        sid = extract_sample_id(prompts[i])
        cache = train_entity_cache[sid]

        r = compute_entity_reward(
            completions[i],
            cache
        )

        rewards.append(r)

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=device
    )

    rewards = torch.clamp(rewards,-10,10)

    return rewards

# =========================================================
# GRPO Trainer
# =========================================================

training_args = GRPOConfig(

    learning_rate=1e-6,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",

    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,

    num_generations=4,

    max_prompt_length=max_seq_length,
    max_completion_length=1100,

    num_train_epochs=5,

    beta=0.15,

    logging_steps=1,

    output_dir="./outputs",

    report_to="wandb",
)

trainer = GRPOTrainer(

    model=model,
    processing_class=tokenizer,
    reward_funcs=[entity_reward],

    args=training_args,

    train_dataset=train_dataset,
)

# =========================================================
# Training
# =========================================================

trainer.train()
