# debug_structural.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from legal_structural_utils import *
TYPE_THRESHOLDS = {
    "PROVISION": 0.5,
    "STATUTE": 0.85,
    "PRECEDENT": 0.82,
    "COURT": 1.0,
    "JUDGE": 1.0,
    "PETITIONER": 0.80,
    "RESPONDENT": 0.80,
}

def debug_similarity_matrix_v3(pred_entities, target_entities, pred_emb, target_emb, compute_similarity_fn):

    n_pred = len(pred_entities)
    n_target = len(target_entities)

    S = np.zeros((n_pred, n_target))

    semantic_matrix = (pred_emb @ target_emb.T).cpu().numpy()

    for i, (p_text, p_type) in enumerate(pred_entities):
        for j, (t_text, t_type) in enumerate(target_entities):

            if p_type != t_type:
                continue

            sem = semantic_matrix[i, j]

            S[i, j] = compute_similarity_fn(
                p_text,
                t_text,
                p_type,
                sem
            )

    return S

# ======================================================
# 0️⃣ Utility
# ======================================================

def ordered_unique(entity_list):
    """Deduplicate while preserving order."""
    return list(dict.fromkeys(entity_list))


# ======================================================
# 1️⃣ Dataset Debug
# ======================================================

def debug_dataset_sample(dataset, n=1):
    print("========== DATASET DEBUG ==========")
    for i in range(min(n, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}")
        print("Keys:", list(sample.keys()))
        print("ID:", sample["id"])
        print("Prompt length:", len(sample["prompt"].split()))
        print("Prompt preview:\n", sample["prompt"][:400])
    print("====================================\n")


# ======================================================
# 2️⃣ Cache Alignment
# ======================================================

def debug_entity_cache_alignment(dataset, entity_cache, n=3):
    print("========== CACHE ALIGNMENT CHECK ==========")
    for i in range(min(n, len(dataset))):
        sid = dataset[i]["id"]
        assert sid in entity_cache, f"{sid} missing in cache"

        cache = entity_cache[sid]

        print(f"\nSample ID: {sid}")
        print("Gold entities:", len(cache["gold_metadata"]))
        print("Source entities:", len(cache["source_metadata"]))
        print("Gold embeddings:", cache["gold_embeddings"].shape)
        print("Source embeddings:", cache["source_embeddings"].shape)

    print("============================================\n")


# ======================================================
# 3️⃣ Reward Decomposition
# ======================================================

@torch.no_grad()
def debug_reward_decomposition(
    llm_model,
    tokenizer,
    dataset,
    entity_cache,
    compute_reward_fn,
    idx=0,
):
    print("========== REWARD DECOMPOSITION ==========")

    sample = dataset[idx]
    sid = sample["id"]
    prompt = sample["prompt"]

    inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=400, do_sample=False)

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    reward, metrics = compute_reward_fn(text, entity_cache[sid])

    print("\nGenerated Preview:\n", text[:600])

    print("\nReward Breakdown:")

    print("Weighted Recall:", round(metrics.get("weighted_recall", 0), 3))
    print("Hallucination Rate:", round(metrics.get("hallucination_rate", 0), 3))
    print("Entity F1:", round(metrics.get("entity_f1", 0), 3))
    print("Generated Entity Count:", metrics.get("entity_count", 0))

    print("\nType metrics:")

    for t, vals in metrics.get("type_metrics", {}).items():
        print(
    f"{t:12s} | "
    f"src={vals['source_count']:2d} "
    f"gen={vals['generated_count']:2d} "
    f"match={vals['matched_count']:2d} "
    f"recall={vals['coverage']:.2f} "
    f"hall={vals['hallucination']:.2f}"
)

    print("\nFinal Reward:", reward)
    print("==========================================\n")


# ======================================================
# 4️⃣ Reward Distribution
# ======================================================

@torch.no_grad()
def debug_reward_distribution(
    llm_model,
    tokenizer,
    dataset,
    entity_cache,
    compute_reward_fn,
    n=10,
):
    print("========== REWARD DISTRIBUTION ==========")

    rewards = []

    for i in range(min(n, len(dataset))):
        sid = dataset[i]["id"]
        prompt = dataset[i]["prompt"]

        inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
        outputs = llm_model.generate(
            **inputs, max_new_tokens=300, do_sample=True, temperature=0.7
        )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        reward, metrics = compute_reward_fn(text, entity_cache[sid])
        rewards.append(reward)

    print("Mean Reward:", np.mean(rewards))
    print("Std Reward :", np.std(rewards))
    print("Min Reward :", np.min(rewards))
    print("Max Reward :", np.max(rewards))
    print("=========================================\n")

    print("Avg Entity Count:", np.mean([
    compute_reward_fn(
        tokenizer.decode(
            llm_model.generate(
                **tokenizer(dataset[i]["prompt"], return_tensors="pt").to(llm_model.device),
                max_new_tokens=300
            )[0][len(tokenizer(dataset[i]["prompt"], return_tensors="pt")["input_ids"][0]):],
            skip_special_tokens=True
        ),
        entity_cache[dataset[i]["id"]]
    )[1].get("entity_count", 0)
    for i in range(min(n, len(dataset)))
]))


# ======================================================
# 5️⃣ Similarity Scores
# ======================================================

@torch.no_grad()
def debug_similarity_scores(
    llm_model,
    tokenizer,
    dataset,
    entity_cache,
    extract_fn,
    canonicalize_fn,
    embed_fn,
    debug_similarity_matrix,
    idx=0,
    threshold=None,
):
    print("========== SIMILARITY DEBUG ==========")

    sample = dataset[idx]
    sid = sample["id"]

    inputs = tokenizer(sample["prompt"], return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=400)

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    gen_entities = ordered_unique(canonicalize_fn(extract_fn(text)))
    source_entities = list(entity_cache[sid]["source_metadata"].keys())

    if not gen_entities:
        print("No generated entities.")
        return

    gen_emb = embed_fn(gen_entities)
    source_emb = entity_cache[sid]["source_embeddings"].to(llm_model.device)

    S = debug_similarity_matrix(
    gen_entities, source_entities, gen_emb, source_emb, compute_similarity_v3
)

    for i, (g_text, g_type) in enumerate(gen_entities):
        best_j = np.argmax(S[i])
        best_score = S[i][best_j]
        best_target = source_entities[best_j]

        print(f"\n{g_text} ({g_type})")
        print("  Best match:", best_target)
        print("  Score:", round(best_score, 3))
        threshold = TYPE_THRESHOLDS.get(g_type, 0.75)
        if best_score < threshold:
            print("  ⚠ Below threshold")

    print("=====================================\n")


# ======================================================
# 6️⃣ Threshold Sweep
# ======================================================

@torch.no_grad()
def threshold_sweep_analysis(
    llm_model,
    tokenizer,
    dataset,
    entity_cache,
    extract_fn,
    canonicalize_fn,
    embed_fn,
    debug_similarity_matrix,
    idx=0,
):
    print("========== THRESHOLD SWEEP ==========")

    sample = dataset[idx]
    sid = sample["id"]

    inputs = tokenizer(sample["prompt"], return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=400)

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    gen_entities = ordered_unique(canonicalize_fn(extract_fn(text)))
    source_entities = list(entity_cache[sid]["source_metadata"].keys())

    if not gen_entities:
        return

    gen_emb = embed_fn(gen_entities)
    source_emb = entity_cache[sid]["source_embeddings"].to(llm_model.device)
    S = debug_similarity_matrix(
    gen_entities,
    source_entities,
    gen_emb,
    source_emb,
    compute_similarity_v3
)

    cost = -S
    row_ind, col_ind = linear_sum_assignment(cost)

    for threshold in np.arange(0.4, 0.9, 0.05):
        matched = sum(S[r, c] >= threshold for r, c in zip(row_ind, col_ind))
        print(f"Threshold {threshold:.2f} | Matches: {matched}")

    print("=====================================\n")


# ======================================================
# 7️⃣ Type Confusion Matrix
# ======================================================

@torch.no_grad()
def type_confusion_matrix(
    llm_model,
    tokenizer,
    dataset,
    entity_cache,
    extract_fn,
    canonicalize_fn,
    embed_fn,
    debug_similarity_matrix,
    idx=0,
    threshold=0.6,
):
    print("========== TYPE CONFUSION MATRIX ==========")

    sample = dataset[idx]
    sid = sample["id"]

    inputs = tokenizer(sample["prompt"], return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=400)

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    gen_entities = ordered_unique(canonicalize_fn(extract_fn(text)))
    source_entities = list(entity_cache[sid]["source_metadata"].keys())

    if not gen_entities:
        return

    gen_emb = embed_fn(gen_entities)
    source_emb = entity_cache[sid]["source_embeddings"].to(llm_model.device)
    S = debug_similarity_matrix(
    gen_entities,
    source_entities,
    gen_emb,
    source_emb,
    compute_similarity_v3
)

    cost = -S
    row_ind, col_ind = linear_sum_assignment(cost)

    confusion = {}
    for r, c in zip(row_ind, col_ind):
        p_type = gen_entities[r][1]
        threshold = TYPE_THRESHOLDS.get(p_type, 0.75)

        if S[r, c] >= threshold:
            g_type = gen_entities[r][1]
            s_type = source_entities[c][1]
            confusion.setdefault(g_type, {})
            confusion[g_type][s_type] = confusion[g_type].get(s_type, 0) + 1

    for g in confusion:
        for s in confusion[g]:
            print(f"{g} → {s}: {confusion[g][s]}")

    print("===========================================\n")


# ======================================================
# 8️⃣ Type-wise Entity Alignment
# ======================================================

@torch.no_grad()
def debug_typewise_entity_alignment(
    llm_model,
    tokenizer,
    dataset,
    entity_cache,
    extract_fn,
    canonicalize_fn,
    embed_fn,
    match_fn,
    idx=0,
):
    print("========== TYPE-WISE ENTITY ALIGNMENT ==========")

    sample = dataset[idx]
    sid = sample["id"]

    inputs = tokenizer(sample["prompt"], return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=400)

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    gen_entities = ordered_unique(canonicalize_fn(extract_fn(text)))
    source_entities = list(entity_cache[sid]["source_metadata"].keys())

    if not gen_entities:
        return

    gen_emb = embed_fn(gen_entities)
    source_emb = entity_cache[sid]["source_embeddings"].to(llm_model.device)

    matched_gen, matched_source = match_fn(
        gen_entities, source_entities, gen_emb, source_emb
    )

    print("\nGenerated Text:\n", text[:600])

    for i, (text_g, type_g) in enumerate(gen_entities):
        mark = "✓" if i in matched_gen else "✗"
        print(f"[{mark}] {text_g} ({type_g})")

    print("============================================\n")


# ======================================================
# 9️⃣ Similarity Heatmap
# ======================================================

@torch.no_grad()
def debug_similarity_heatmap(
    llm_model,
    tokenizer,
    dataset,
    entity_cache,
    extract_fn,
    canonicalize_fn,
    embed_fn,
    debug_similarity_matrix,
    idx=0,
):
    sample = dataset[idx]
    sid = sample["id"]

    inputs = tokenizer(sample["prompt"], return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=400)

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    gen_entities = ordered_unique(canonicalize_fn(extract_fn(text)))
    source_entities = list(entity_cache[sid]["source_metadata"].keys())

    if not gen_entities:
        return

    gen_emb = embed_fn(gen_entities)
    source_emb = entity_cache[sid]["source_embeddings"].to(llm_model.device)
    S = debug_similarity_matrix(
    gen_entities,
    source_entities,
    gen_emb,
    source_emb,
    compute_similarity_v3
)

    plt.figure(figsize=(10, 6))
    plt.imshow(S, aspect="auto")
    plt.xticks(range(len(source_entities)), [e[0] for e in source_entities], rotation=90)
    plt.yticks(range(len(gen_entities)), [e[0] for e in gen_entities])
    plt.colorbar()
    plt.title("Entity Similarity Heatmap")
    plt.show()


# ======================================================
# 🔟 Adversarial Tests
# ======================================================

def debug_adversarial_cases(entity_cache, compute_reward_fn):
    print("========== ADVERSARIAL TESTS ==========")

    fake_cases = {
    "EMPTY": "",
    "SPAM_PROVISION": "section 302 section 302 section 302",
    "GENERIC_TEXT": "Procedural aspects discussed.",
    "FAKE_JUDGE": "Justice John Smith presided over the case.",
    "FAKE_STATUTE": "Under Section 999 of the Criminal Code."
}

    sample_cache = next(iter(entity_cache.values()))

    for name, text in fake_cases.items():
        reward, _ = compute_reward_fn(text, sample_cache)
        print(name, "→", reward)

    print("========================================\n")


# ======================================================
# 1️⃣1️⃣ Cache Audit
# ======================================================

def audit_cache_entities(entity_cache, n_samples=20):
    print("========== CACHE AUDIT ==========")

    type_counter = {}

    for i, (sid, sample) in enumerate(entity_cache.items()):
        if i >= n_samples:
            break
        for _, etype in sample["source_metadata"].keys():
            type_counter[etype] = type_counter.get(etype, 0) + 1

    for k, v in type_counter.items():
        print(f"{k}: {v}")

    print("==================================\n")



# ======================================================
# 1️⃣2️⃣ Generated Entity Audit
# ======================================================

@torch.no_grad()
def audit_generated_entities(
    llm_model,
    tokenizer,
    dataset,
    extract_fn,
    canonicalize_fn,
    n=5,
):
    print("========== GENERATED ENTITY AUDIT ==========")

    for i in range(min(n, len(dataset))):
        sample = dataset[i]

        inputs = tokenizer(sample["prompt"], return_tensors="pt").to(llm_model.device)
        outputs = llm_model.generate(**inputs, max_new_tokens=300)

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        raw = extract_fn(text)
        canon = canonicalize_fn(raw)

        print(f"\nSample {sample['id']}")
        print("Raw:", raw)
        print("Canonical:", canon)

    print("============================================\n")


# ======================================================
# 1️⃣3️⃣ Matching by Type Audit
# ======================================================

@torch.no_grad()
def audit_matching_by_type(
    llm_model,
    tokenizer,
    dataset,
    entity_cache,
    extract_fn,
    canonicalize_fn,
    embed_fn,
    match_fn,
    n=10,
):
    print("========== MATCHING BY TYPE ==========")

    stats = {}

    for i in range(min(n, len(dataset))):
        sample = dataset[i]
        sid = sample["id"]

        inputs = tokenizer(sample["prompt"], return_tensors="pt").to(llm_model.device)
        outputs = llm_model.generate(**inputs, max_new_tokens=300)

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        gen_entities = ordered_unique(canonicalize_fn(extract_fn(text)))
        source_entities = list(entity_cache[sid]["source_metadata"].keys())

        if not gen_entities:
            continue

        gen_emb = embed_fn(gen_entities)
        source_emb = entity_cache[sid]["source_embeddings"].to(llm_model.device)

        matched_gen, _ = match_fn(
            gen_entities, source_entities, gen_emb, source_emb
        )

        for i_g, (_, t) in enumerate(gen_entities):
            stats.setdefault(t, {"generated": 0, "matched": 0})
            stats[t]["generated"] += 1
            if i_g in matched_gen:
                stats[t]["matched"] += 1

    for t in stats:
        gen = stats[t]["generated"]
        matched = stats[t]["matched"]
        print(f"{t} | MatchRate = {matched/gen if gen>0 else 0:.3f}")

    print("======================================\n")


# ======================================================
# 1️⃣4️⃣ Court Matching Audit
# ======================================================

@torch.no_grad()
def audit_court_matching(
    llm_model,
    tokenizer,
    dataset,
    entity_cache,
    extract_fn,
    canonicalize_fn,
    embed_fn,
    match_fn,
    n=10,
):
    print("========== COURT MATCH AUDIT ==========")

    for i in range(min(n, len(dataset))):
        sample = dataset[i]
        sid = sample["id"]

        inputs = tokenizer(sample["prompt"], return_tensors="pt").to(llm_model.device)
        outputs = llm_model.generate(**inputs, max_new_tokens=300)

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        gen_entities = ordered_unique(canonicalize_fn(extract_fn(text)))
        source_entities = list(entity_cache[sid]["source_metadata"].keys())

        if not gen_entities:
            continue

        gen_emb = embed_fn(gen_entities)
        source_emb = entity_cache[sid]["source_embeddings"].to(llm_model.device)

        _, matched_source = match_fn(
            gen_entities, source_entities, gen_emb, source_emb
        )

        for idx in matched_source:
            text_s, etype = source_entities[idx]
            if etype == "COURT":
                print("Matched COURT:", text_s)

    print("=======================================\n")


@torch.no_grad()
def debug_entity_coverage_failure(
    llm_model,
    tokenizer,
    dataset,
    entity_cache,
    extract_fn,
    canonicalize_fn,
    embed_fn,
    match_fn,
    n=20
):

    print("========== ENTITY COVERAGE FAILURE ANALYSIS ==========")

    total_source = 0
    matched = 0
    generated_but_unmatched = 0
    never_generated = 0

    for i in range(min(n, len(dataset))):

        sample = dataset[i]
        sid = sample["id"]

        inputs = tokenizer(sample["prompt"], return_tensors="pt").to(llm_model.device)
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.7
        )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        gen_entities = ordered_unique(
            canonicalize_fn(extract_fn(text))
        )

        source_entities = list(entity_cache[sid]["source_metadata"].keys())

        if not source_entities:
            continue

        gen_emb = embed_fn(gen_entities)
        source_emb = entity_cache[sid]["source_embeddings"].to(llm_model.device)

        matched_gen, matched_source = match_fn(
            gen_entities,
            source_entities,
            gen_emb,
            source_emb
        )

        total_source += len(source_entities)
        matched += len(matched_source)

        # track generated entities
        matched_source_set = set(matched_source)

        for idx, src in enumerate(source_entities):

            if idx in matched_source_set:
                continue

            # check if any generated entity of same type exists
            src_type = src[1]

            generated_same_type = any(
                g_type == src_type for _, g_type in gen_entities
            )

            if generated_same_type:
                generated_but_unmatched += 1
            else:
                never_generated += 1

    print("\nSOURCE ENTITIES:", total_source)
    print("MATCHED:", matched)
    print("GENERATED BUT UNMATCHED:", generated_but_unmatched)
    print("NEVER GENERATED:", never_generated)

    if total_source > 0:
        print("\nCoverage:", matched / total_source)
        print("Matcher Failure Rate:", generated_but_unmatched / total_source)
        print("Generation Failure Rate:", never_generated / total_source)

    print("======================================================\n")

@torch.no_grad()
def plot_recall_vs_hallucination(
    llm_model,
    tokenizer,
    dataset,
    entity_cache,
    compute_reward_fn,
    n_samples=20,
    gens_per_sample=4,
):

    print("========== RECALL vs HALLUCINATION PLOT ==========")

    recalls = []
    hallucinations = []

    for i in range(min(n_samples, len(dataset))):

        sample = dataset[i]
        sid = sample["id"]
        prompt = sample["prompt"]

        inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)

        for _ in range(gens_per_sample):

            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

            gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]

            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            reward, metrics = compute_reward_fn(text, entity_cache[sid])

            recalls.append(metrics.get("weighted_recall", 0))
            hallucinations.append(metrics.get("hallucination_rate", 0))

    recalls = np.array(recalls)
    hallucinations = np.array(hallucinations)

    plt.figure(figsize=(6,6))

    plt.scatter(
        recalls,
        hallucinations,
        alpha=0.7
    )

    plt.xlabel("Weighted Recall")
    plt.ylabel("Hallucination Rate")
    plt.title("Recall vs Hallucination Trade-off")

    plt.grid(True)

    print("Avg Recall:", recalls.mean())
    print("Avg Hallucination:", hallucinations.mean())

    plt.show()

    print("============================================\n")
# ======================================================
# 1️⃣5️⃣ Master Runner
# ======================================================

def run_all_debug(
    llm_model,
    tokenizer,
    dataset,
    entity_cache,
    compute_reward_fn,
    extract_fn,
    canonicalize_fn,
    embed_fn,
    debug_similarity_matrix,
    match_fn,
):
    debug_dataset_sample(dataset, 2)
    debug_entity_cache_alignment(dataset, entity_cache, 3)
    debug_reward_decomposition(llm_model, tokenizer, dataset, entity_cache, compute_reward_fn)
    debug_reward_distribution(llm_model, tokenizer, dataset, entity_cache, compute_reward_fn)
    debug_similarity_scores(llm_model, tokenizer, dataset, entity_cache,
                            extract_fn, canonicalize_fn, embed_fn, debug_similarity_matrix)
    threshold_sweep_analysis(llm_model, tokenizer, dataset, entity_cache,
                             extract_fn, canonicalize_fn, embed_fn, debug_similarity_matrix)
    type_confusion_matrix(llm_model, tokenizer, dataset, entity_cache,
                          extract_fn, canonicalize_fn, embed_fn, debug_similarity_matrix)
    debug_typewise_entity_alignment(llm_model, tokenizer, dataset, entity_cache,
                                    extract_fn, canonicalize_fn, embed_fn, match_fn)
    debug_similarity_heatmap(llm_model, tokenizer, dataset, entity_cache,
                             extract_fn, canonicalize_fn, embed_fn, debug_similarity_matrix)
    debug_adversarial_cases(entity_cache, compute_reward_fn)
    audit_cache_entities(entity_cache)
    audit_generated_entities(llm_model, tokenizer, dataset,
                             extract_fn, canonicalize_fn)
    audit_matching_by_type(llm_model, tokenizer, dataset, entity_cache,
                           extract_fn, canonicalize_fn, embed_fn, match_fn)
    audit_court_matching(llm_model, tokenizer, dataset, entity_cache,
                         extract_fn, canonicalize_fn, embed_fn, match_fn)
    
#     debug_entity_coverage_failure(
#     llm_model=llm_model,
#     tokenizer=tokenizer,
#     dataset=dataset,
#     entity_cache=entity_cache,
#     extract_fn=extract_fn,
#     canonicalize_fn=canonicalize_fn,
#     embed_fn=embed_fn,
#     match_fn=match_fn,
# )
#     plot_recall_vs_hallucination(
#     llm_model,
#     tokenizer,
#     dataset,
#     entity_cache,
#     compute_reward_fn
# )
