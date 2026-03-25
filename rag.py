from langchain_community.document_loaders import CSVLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_PATH = "data/dataset.csv"
DB_PATH = "vectordb"

loader = CSVLoader(
    file_path=DATA_PATH,
    encoding="latin-1"
)

import pandas as pd
from langchain_core.documents import Document

df = pd.read_csv("data/dataset.csv", encoding="latin-1")

documents = []

for _, row in df.iterrows():
    documents.append(
        Document(
            page_content=f"Question: {row['Question']} Answer: {row['Answer']}",
            metadata={
                "category": row["Category"],
                "difficulty": str(row["Difficulty"]).lower()
            }
        )
    )
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=DB_PATH
)

print("✅ Vector DB created successfully!")