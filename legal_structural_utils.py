"""
Utility functions for the Type-Aware Normalization and Hybrid Alignment (TANHA)
framework described in the paper.

This module implements:
1. Entity canonicalization
2. Hybrid similarity computation (lexical + semantic)
3. Global entity alignment using Hungarian bipartite matching
"""

import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------

torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------
# Embedding model (used for semantic similarity)
# -------------------------------------------------------------------

model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)


# -------------------------------------------------------------------
# Text normalization utilities
# -------------------------------------------------------------------

def normalize_legal_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)

    text = text.replace("sec.", "section")
    text = text.replace("u/s", "section")

    text = text.replace("versus", "v")
    text = text.replace(" vs ", " v ")

    text = re.sub(r"\s*\(\s*", "(", text)
    text = re.sub(r"\s*\)\s*", ")", text)

    text = re.sub(r"[.,;:]+$", "", text)

    return text


# -------------------------------------------------------------------
# Canonicalization Functions
# -------------------------------------------------------------------

def canonicalize_provision(text):

    text_lower = text.lower()

    sections = re.findall(r"\d+[a-zA-Z]*", text_lower)
    sections = sorted(set(sections))

    if not sections:
        return None

    if "article" in text_lower:
        return "article_" + "_".join(sections)

    return "section_" + "_".join(sections)


def canonicalize_court(text):

    text = text.lower().strip()

    text = re.sub(r"[.,;:]", "", text)
    text = re.sub(r"\s+", " ", text)

    text = text.replace("apex court", "supreme court")
    text = text.replace("supreme court of india", "supreme court")

    text = re.sub(r"\bhc\b", "high court", text)
    text = re.sub(r"^the ", "", text)

    m = re.match(r"high court of (.+)", text)
    if m:
        text = f"{m.group(1).strip()} high court"

    m = re.match(r"high court at (.+)", text)
    if m:
        text = f"{m.group(1).strip()} high court"

    m = re.match(r"(.+) bench of high court", text)
    if m:
        text = f"{m.group(1).strip()} high court"

    return re.sub(r"\s+", " ", text).strip()


def canonicalize_judge(text):

    text = text.lower()

    text = re.sub(r"\b(justice|honble|dr|mr|mrs|ms|shri|smt)\b\.?", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def canonicalize_statute(text):

    text = text.lower()

    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\b\d{4}\b", "", text)

    ALIASES = {
        "ipc": "indian penal code",
        "crpc": "code of criminal procedure",
        "ac act": "arbitration conciliation act",
    }

    text = text.replace("  ", " ").strip()

    if text in ALIASES:
        text = ALIASES[text]

    return text


def canonicalize_precedent(text):

    if not text:
        return ""

    text = text.lower().strip()

    text = re.sub(r"\bversus\b", "v", text)
    text = re.sub(r"\bvs\.?\b", "v", text)
    text = re.sub(r"\bv\.?\b", "v", text)

    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)

    text = re.sub(r"\b\d{4}\b", "", text)
    text = re.sub(r"\b\d+\s*scc\b.*", "", text)
    text = re.sub(r"air\s*\d+.*", "", text)

    procedural_patterns = [
        r"criminal appeal.*",
        r"civil appeal.*",
        r"special leave.*",
        r"letters patent appeal.*",
        r"criminal misc.*",
        r"writ petition.*",
        r"suo moto.*",
        r"review petition.*",
        r"reported in.*",
    ]

    for pattern in procedural_patterns:
        text = re.sub(pattern, "", text)

    text = re.sub(r"\b(dr|mr|mrs|ms|justice|honble|shri|smt)\b\.?", "", text)

    text = re.sub(r"\b&?\s*others\b", "", text)
    text = re.sub(r"\bors\.?\b", "", text)

    if " v " in text:
        left, right = text.split(" v ", 1)
        right = re.split(r"\b(and|,)\b", right)[0].strip()
        text = f"{left.strip()} v {right}"

    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -------------------------------------------------------------------
# Embedding utilities
# -------------------------------------------------------------------

def embed_entity_list(entity_list):

    if len(entity_list) == 0:
        return torch.empty((0, model.get_sentence_embedding_dimension()))

    texts = [e[0] for e in entity_list]

    emb = model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return emb.cpu()


# -------------------------------------------------------------------
# Similarity functions
# -------------------------------------------------------------------

def lexical_similarity(a, b):

    a = normalize_legal_text(a)
    b = normalize_legal_text(b)

    return fuzz.token_sort_ratio(a, b) / 100.0


def provision_similarity(p_text, t_text):

    if p_text is None or t_text is None:
        return 0.0

    p_set = set(p_text.replace("section_", "").split("_"))
    t_set = set(t_text.replace("section_", "").split("_"))

    inter = len(p_set & t_set)
    union = len(p_set | t_set)

    if union == 0:
        return 0.0

    return inter / union


# -------------------------------------------------------------------
# Type-specific thresholds
# -------------------------------------------------------------------

TYPE_MATCH_THRESHOLDS = {
    "PROVISION": 0.5,
    "STATUTE": 0.85,
    "PRECEDENT": 0.82,
    "COURT": 1.0,
    "JUDGE": 1.0,
    "PETITIONER": 0.80,
    "RESPONDENT": 0.80,
}


# -------------------------------------------------------------------
# Similarity matrix construction
# -------------------------------------------------------------------

def build_similarity_matrix(pred_entities, target_entities, pred_emb, target_emb):

    n_pred = len(pred_entities)
    n_target = len(target_entities)

    S = np.zeros((n_pred, n_target))

    if n_pred == 0 or n_target == 0:
        return S

    semantic_matrix = (pred_emb @ target_emb.T).cpu().numpy()

    for i, (p_text, p_type) in enumerate(pred_entities):

        for j, (t_text, t_type) in enumerate(target_entities):

            if p_type != t_type:
                continue

            sem = semantic_matrix[i, j]

            if p_type == "PROVISION":
                score = provision_similarity(p_text, t_text)

            elif p_type == "STATUTE":
                score = 1.0 if p_text == t_text else sem

            elif p_type == "PRECEDENT":

                p_clean = canonicalize_precedent(p_text)
                t_clean = canonicalize_precedent(t_text)

                score = 1.0 if p_clean == t_clean else sem

            elif p_type == "COURT":
                score = 1.0 if p_text == t_text else 0.0

            elif p_type == "JUDGE":

                p_clean = canonicalize_judge(p_text)
                t_clean = canonicalize_judge(t_text)

                score = 1.0 if p_clean == t_clean else sem

            else:
                score = sem

            S[i, j] = score

    return S


# -------------------------------------------------------------------
# Hybrid entity matching (Hungarian alignment)
# -------------------------------------------------------------------

def hybrid_match(pred_entities, target_entities, pred_emb, target_emb):

    if len(pred_entities) == 0 or len(target_entities) == 0:
        return set(), set()

    S = build_similarity_matrix(
        pred_entities,
        target_entities,
        pred_emb,
        target_emb,
    )

    cost_matrix = -S
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pred = set()
    matched_target = set()

    for r, c in zip(row_ind, col_ind):

        p_type = pred_entities[r][1]
        threshold = TYPE_MATCH_THRESHOLDS.get(p_type, 0.75)

        if S[r, c] >= threshold:

            matched_pred.add(r)
            matched_target.add(c)

    return matched_pred, matched_target
