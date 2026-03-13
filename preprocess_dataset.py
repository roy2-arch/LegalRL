import re
import pickle
import pandas as pd
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/test_iltur_ner.json"
CACHE_PATH = "cache/test_entity_cache.pkl"
WEIGHT_PATH = "cache/entity_type_weights.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    device=device
)

TAG_PATTERN = r'<(.*?)>(.*?)</\1>'


def normalize_legal_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("sec.", "section")
    text = text.replace("u/s", "section")
    text = re.sub(r"[.,;:]+$", "", text)
    return text


def canonicalize_provision(text):

    sections = re.findall(r'\d+[a-zA-Z]*', text)

    if not sections:
        return None

    sections = sorted(set(sections))

    return "section_" + "_".join(sections)


def canonicalize_statute(text):

    text = text.lower()

    text = text.replace("i.p.c", "ipc")
    text = text.replace("cr.p.c", "crpc")

    ALIASES = {
        "ipc": "indian penal code",
        "crpc": "code of criminal procedure",
    }

    text = re.sub(r"[^a-z0-9\s]", "", text)

    return ALIASES.get(text, text)


def canonicalize_precedent(text):

    text = text.lower()

    text = re.sub(r"\bvs\.?\b", "v", text)
    text = re.sub(r"\bv\.?\b", "v", text)

    text = re.sub(r"\(\d{4}.*?\)", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def canonicalize_court(text):

    text = text.lower()

    text = text.replace(
        "supreme court of india",
        "supreme court"
    )

    m = re.match(r"high court of (.+)", text)

    if m:
        place = m.group(1).strip()
        text = f"{place} high court"

    return text


def extract_entities(tagged_text):

    matches = re.finditer(
        TAG_PATTERN,
        tagged_text,
        flags=re.DOTALL
    )

    entities = []

    total_len = len(tagged_text)

    for m in matches:

        tag = m.group(1).upper()
        text = normalize_legal_text(m.group(2))

        pos = m.start() / (total_len + 1e-8)

        entities.append(
            {
                "text": text,
                "type": tag,
                "position": pos
            }
        )

    return entities


def build_metadata(tagged_text):

    entities = extract_entities(tagged_text)

    type_groups = defaultdict(list)

    for e in entities:
        type_groups[e["type"]].append(e)

    metadata = {}

    for etype, group in type_groups.items():

        canonical_map = defaultdict(list)

        for e in group:

            txt = e["text"]

            if etype == "PROVISION":
                canon = canonicalize_provision(txt)

            elif etype == "STATUTE":
                canon = canonicalize_statute(txt)

            elif etype == "PRECEDENT":
                canon = canonicalize_precedent(txt)

            elif etype == "COURT":
                canon = canonicalize_court(txt)

            else:
                canon = normalize_legal_text(txt)

            if canon is None:
                continue

            canonical_map[canon].append(e)

        for canon, items in canonical_map.items():

            metadata[(canon, etype)] = {

                "frequency": len(items),

                "first_position":
                    min(x["position"] for x in items),

                "is_early":
                    min(x["position"] for x in items) <= 0.3,
            }

    return metadata


def embed_entities(entity_list):

    if len(entity_list) == 0:

        dim = model.get_sentence_embedding_dimension()

        return torch.empty((0, dim))

    texts = [e[0] for e in entity_list]

    emb = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_tensor=True
    )

    return emb.cpu()


def prepare_sample(judgment, summary):

    source_meta = build_metadata(judgment)
    gold_meta = build_metadata(summary)

    source_entities = list(source_meta.keys())
    gold_entities = list(gold_meta.keys())

    source_emb = embed_entities(source_entities)
    gold_emb = embed_entities(gold_entities)

    return {

        "source_metadata": source_meta,
        "gold_metadata": gold_meta,

        "source_entities": source_entities,
        "gold_entities": gold_entities,

        "source_embeddings": source_emb,
        "gold_embeddings": gold_emb,

        "source_text": judgment
    }


df = pd.read_json(DATA_PATH, lines=True)

cache = []

type_source = defaultdict(int)
type_summary = defaultdict(int)

for row in df.itertuples():

    processed = prepare_sample(
        row.NER_Judgment,
        row.NER_Summary
    )

    cache.append(processed)

    for (_,t) in processed["source_entities"]:
        type_source[t]+=1

    for (_,t) in processed["gold_entities"]:
        type_summary[t]+=1


with open(CACHE_PATH,"wb") as f:
    pickle.dump(cache,f)


# ===============================
# Compute dataset-adaptive weights
# ===============================

weights = {}

for t in type_source:

    retain = type_summary[t] / max(type_source[t],1)

    weights[t] = retain

s = sum(weights.values())

for t in weights:
    weights[t] /= s

with open(WEIGHT_PATH,"wb") as f:
    pickle.dump(weights,f)

print("Preprocessing complete.")
