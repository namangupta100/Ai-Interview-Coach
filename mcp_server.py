import random
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from mcp.server.fastmcp import FastMCP
from langchain_chroma import Chroma

import engine
from embeddings import LocalEmbeddings
from engine import get_llm
print("✅ Imports successful!")


def extract_question(raw_text: str) -> str:
    """Extract only the question text from CSVLoader's raw content."""
    if "Question:" in raw_text and "Answer:" in raw_text:
        return raw_text.split("Question:")[1].split("Answer:")[0].strip()
    return raw_text
# =========================
# MCP INIT
# =========================
mcp = FastMCP("Interview Coach")
print("✅ MCP initialized!")
# =========================
# PATH
# =========================
DB_PATH = engine.DB_PATH
print("✅ Paths set!")
# =========================
# EMBEDDINGS + VECTOR DB
# =========================
embeddings = LocalEmbeddings()
print("✅ Embeddings initialized!")

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)
print("✅ Vector DB initialized!")

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
print("✅ Retriever initialized!")
# =========================
# TOOL 1: RAG Q&A
# =========================
@mcp.tool()
def ask_ai(question: str) -> str:
    docs = retriever.invoke(question)
    print(docs)
    context = "\n\n".join([doc.page_content for doc in docs])
    print("✅ Context created!")
    prompt = f"""
Answer ONLY from the context below.
If not found, say "I don't know".

Context:
{context}

Question:
{question}
"""
    print("✅ Prompt created!")
    return str(get_llm().invoke(prompt).content)

# =========================
# TOOL 2: RAG Interview Question
# =========================
@mcp.tool()
def generate_question(topic: str = "programming") -> str:
    docs = retriever.invoke(f"{topic} interview question")
    doc = random.choice(docs)
    return extract_question(doc.page_content)

# =========================
# TOOL 3: Evaluate Answer
# =========================
@mcp.tool()
def evaluate_answer(question: str, answer: str) -> str:
    prompt = f"""
You are a strict interviewer.

Question: {question}
Answer: {answer}

Give:
- Score (0-10)
- Feedback
- Correct answer
"""

    return str(get_llm().invoke(prompt).content)

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    mcp.run()