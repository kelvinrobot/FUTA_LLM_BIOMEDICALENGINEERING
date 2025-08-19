# rag.py
import os
import json
import pickle
import logging
from typing import List, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import VECTORSTORE_DIR, EMBEDDING_MODEL

log = logging.getLogger(__name__)


class RAGAgent:
    """
    Loads a FAISS index + metadata from VECTORSTORE_DIR (config).
    Provides retrieve(query, k) -> (contexts: List[str], sources: List[dict])
    """

    def __init__(self, vectorstore_dir: Optional[str] = None, embedding_model: Optional[str] = None):
        self.vectorstore_dir = vectorstore_dir or str(VECTORSTORE_DIR)
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL
        self.index: Optional[faiss.Index] = None
        self.metadata: Optional[List[dict]] = None
        self._embedder: Optional[SentenceTransformer] = None
        self._loaded = False

    def _find_index_file(self) -> Optional[str]:
        if not os.path.isdir(self.vectorstore_dir):
            log.warning("Vectorstore dir not found: %s", self.vectorstore_dir)
            return None

        for fname in os.listdir(self.vectorstore_dir):
            if fname.endswith((".faiss", ".index", ".bin")) or fname.startswith("index"):
                return os.path.join(self.vectorstore_dir, fname)
        return None

    def _find_meta_file(self) -> Optional[str]:
        if not os.path.isdir(self.vectorstore_dir):
            return None

        for candidate in ("index.pkl", "metadata.pkl", "index_meta.pkl", "metadata.json", "index.json"):
            p = os.path.join(self.vectorstore_dir, candidate)
            if os.path.exists(p):
                return p

        for fname in os.listdir(self.vectorstore_dir):
            if fname.endswith(".pkl") or fname.endswith(".json"):
                return os.path.join(self.vectorstore_dir, fname)

        return None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            log.info("Loading embedder: %s", self.embedding_model_name)
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def load(self) -> None:
        """Load index and metadata into memory (idempotent)."""
        if self._loaded:
            return

        idx_path = self._find_index_file()
        meta_path = self._find_meta_file()

        if not idx_path or not meta_path:
            log.warning("No index/metadata found in %s — retrieval disabled.", self.vectorstore_dir)
            return

        log.info("Loading FAISS index from: %s", idx_path)
        try:
            self.index = faiss.read_index(idx_path)
        except Exception as e:
            log.error("Failed to read FAISS index: %s", e)
            return

        log.info("Loading metadata from: %s", meta_path)
        try:
            if meta_path.endswith(".json"):
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            else:
                with open(meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
        except Exception as e:
            log.error("Failed to read metadata: %s", e)
            return

        # Normalize metadata type
        if not isinstance(self.metadata, list):
            if isinstance(self.metadata, dict):
                try:
                    self.metadata = [self.metadata[k] for k in sorted(self.metadata.keys())]
                except Exception:
                    self.metadata = list(self.metadata.values())
            else:
                self.metadata = list(self.metadata)

        log.info("Loaded index and metadata: metadata length=%d", len(self.metadata))
        self._loaded = True

    def retrieve(self, query: str, k: int = 3) -> Tuple[List[str], List[dict]]:
        """
        Return two lists:
        - contexts: [str, ...] top-k chunk texts (may be fewer)
        - sources: [ {meta..., "score": float}, ... ]
        """
        if not self._loaded:
            self.load()

        if self.index is None or self.metadata is None:
            return [], []

        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")

        # try normalize if index uses normalized vectors
        try:
            faiss.normalize_L2(q_emb)
        except Exception:
            pass

        try:
            D, I = self.index.search(q_emb, k)
        except Exception as e:
            log.warning("FAISS search error: %s", e)
            return [], []

        if I is None or D is None:
            return [], []

        indices = np.array(I).reshape(-1)[:k].tolist()
        scores = np.array(D).reshape(-1)[:k].tolist()

        contexts, sources = [], []
        for idx, score in zip(indices, scores):
            if int(idx) < 0 or idx >= len(self.metadata):
                continue

            meta = self.metadata[int(idx)]
            text = None

            if isinstance(meta, dict):
                for key in ("text", "page_content", "content", "chunk_text", "source_text"):
                    if key in meta and meta[key]:
                        text = meta[key]
                        break
                if text is None and "metadata" in meta and isinstance(meta["metadata"], dict):
                    text = meta["metadata"].get("text") or meta["metadata"].get("page_content")
            elif isinstance(meta, str):
                text = meta

            if text is None:
                text = str(meta)

            contexts.append(text)
            sources.append({"meta": meta, "score": float(score)})

        return contexts, sources
