from __future__ import annotations

import numpy as np

from src.config import LOCAL_EMBED_MODEL


_model = None


def get_local_embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(LOCAL_EMBED_MODEL)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_local_embedder()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 32,
    )
    return np.asarray(vectors, dtype=np.float32)
