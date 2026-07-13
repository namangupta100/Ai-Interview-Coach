"""Local embedding function backed by ChromaDB's bundled ONNX MiniLM model.

Anthropic does not expose an embeddings API, so the RAG lookup uses a local
model instead. Chroma ships this one with onnxruntime, so it needs no API key
and no torch — it just works on a CPU-only host.
"""

from typing import List

from chromadb.utils import embedding_functions
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """Adapts Chroma's ONNX embedding function to LangChain's Embeddings interface."""

    def __init__(self) -> None:
        self._fn = embedding_functions.DefaultEmbeddingFunction()

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, vector)) for vector in self._fn(texts)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]
