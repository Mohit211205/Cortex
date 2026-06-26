"""
CodeBERT embedding wrapper.
Model: microsoft/codebert-base (768-dim embeddings)
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class CodeBERTEmbedder:
    MODEL_NAME = "microsoft/codebert-base"
    DIM = 768

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(device)
        self.model.eval()

    def embed(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # CLS token embedding as the chunk representation
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str], batch_size: int = 16) -> List[np.ndarray]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            results.extend(embeddings.astype(np.float32))

        return results


@lru_cache(maxsize=1)
def get_embedder(device: str = "cpu") -> CodeBERTEmbedder:
    """Singleton embedder — loaded once, reused across requests."""
    return CodeBERTEmbedder(device=device)


def chunk_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]
