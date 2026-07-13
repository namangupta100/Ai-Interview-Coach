"""Builds the Chroma vector DB from data/dataset.csv.

Importable (engine.py builds the index on first boot if it's missing) and
runnable directly:
    python rag.py
"""

import os

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document

from embeddings import LocalEmbeddings

DATA_PATH = "data/dataset.csv"

# Named for the embedding model that built it. The old `vectordb/` was built with
# Ollama's 768-dim nomic-embed-text; this model emits 384 dims, and querying the
# old index with the new embedder is a dimension-mismatch error. Separate paths
# keep the two from ever being confused.
DB_PATH = os.getenv("VECTORDB_PATH", "vectordb_minilm")


def build_index(db_path: str = DB_PATH) -> int:
    """Embed every Q&A row into a fresh Chroma index. Returns the document count."""
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    documents = [
        Document(
            page_content=f"Question: {row['Question']} Answer: {row['Answer']}",
            metadata={
                "category": row["Category"],
                "difficulty": str(row["Difficulty"]).lower(),
            },
        )
        for _, row in df.iterrows()
    ]

    Chroma.from_documents(
        documents=documents,
        embedding=LocalEmbeddings(),
        persist_directory=db_path,
    )
    return len(documents)


if __name__ == "__main__":
    count = build_index()
    print(f"✅ Vector DB created at {DB_PATH} with {count} documents.")
