"""One-time pipeline: embed each law chunk and save the matrix for fast
in-memory cosine-similarity retrieval at query time (no vector DB needed
for ~15-100 chunks).

Run: python -m src.build_index
"""
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME, LAW_EMBEDDINGS_PATH, LAW_SECTIONS_JSON_PATH


def main():
    sections = json.loads(LAW_SECTIONS_JSON_PATH.read_text(encoding="utf-8"))
    texts = [f"Law {s['law_no']} - {s['title']}: {s['text']}" for s in sections]

    print(f"Embedding {len(texts)} law chunks with {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    LAW_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(LAW_EMBEDDINGS_PATH, embeddings)
    print(f"Wrote embeddings matrix {embeddings.shape} to {LAW_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
