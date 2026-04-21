"""
Top-k retrieval + Maximal Marginal Relevance (MMR) diversification.
"""

from __future__ import annotations

import numpy as np

from src.embeddings import embed_texts


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n, d), b: (m, d) normalized -> (n, m)"""
    return a @ b.T


def mmr_select(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    lambda_mult: float,
    top_n: int,
) -> list[int]:
    """
    Select `top_n` indices from doc_vecs using MMR.
    query_vec: (d,), doc_vecs: (m, d) L2-normalized.
    """
    if doc_vecs.size == 0:
        return []
    m = doc_vecs.shape[0]
    sim_to_query = (doc_vecs @ query_vec.reshape(-1,)).reshape(-1)
    selected: list[int] = []
    remaining = set(range(m))

    while remaining and len(selected) < top_n:
        best_idx = None
        best_score = -1e9
        for i in remaining:
            if not selected:
                mmr = lambda_mult * sim_to_query[i]
            else:
                sim_to_sel = doc_vecs[i] @ doc_vecs[selected].T
                redundancy = float(np.max(sim_to_sel))
                mmr = lambda_mult * sim_to_query[i] - (1 - lambda_mult) * redundancy
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        assert best_idx is not None
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def retrieve_with_mmr(
    query: str,
    candidate_texts: list[str],
    candidate_meta: list[dict],
    candidate_chunk_ids: list[str],
    lambda_mult: float,
    select_n: int,
) -> list[tuple[str, dict, str, float]]:
    """Returns selected chunks as (chunk_id, metadata, chunk_text, similarity_to_query)."""
    if not candidate_texts:
        return []
    q = embed_texts([query])
    docs = embed_texts(candidate_texts)
    idxs = mmr_select(q[0], docs, lambda_mult=lambda_mult, top_n=min(select_n, len(candidate_texts)))
    out: list[tuple[str, dict, str, float]] = []
    for i in idxs:
        cid = candidate_chunk_ids[i] if i < len(candidate_chunk_ids) else ""
        meta = candidate_meta[i]
        text = candidate_texts[i]
        sim = float((docs[i] @ q[0].T).item())
        out.append((cid, meta, text, sim))
    return out
