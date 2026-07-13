"""Interview engine: question selection, RAG lookup, and LLM evaluation.

Both the FastAPI app (app.py) and the Streamlit UI (frontend.py) call into this
module, so the two can run as one process on hosts that only give you one.
"""

import os
import random
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from embeddings import LocalEmbeddings

# Local dev convenience: pick up ANTHROPIC_API_KEY from a .env file if present.
# On Hugging Face Spaces there is no .env — the key arrives as a real env var.
load_dotenv()

DB_PATH = os.getenv("VECTORDB_PATH", "vectordb_minilm")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")

QUESTIONS = {
    "python": [
        "What is Python?",
        "Explain list vs tuple.",
        "What are decorators?",
        "What is OOP in Python?",
        "What is a lambda function?",
    ],
    "data structures": [
        "What is a stack?",
        "What is a queue?",
        "Difference between array and linked list?",
        "What is a hash table?",
    ],
    "machine learning": [
        "What is overfitting?",
        "What is underfitting?",
        "Explain gradient descent.",
        "What is supervised learning?",
    ],
}

LEVELS = ["easy", "intermediate", "advanced", "expert"]

sessions: Dict[str, Dict[str, Any]] = {}

_llm: Optional[ChatAnthropic] = None
_vectorstore: Optional[Chroma] = None


def get_llm() -> ChatAnthropic:
    """Build the Claude client on first use so import never fails without a key."""
    global _llm
    if _llm is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it as a Secret in your Space "
                "settings, or export it locally."
            )
        _llm = ChatAnthropic(model=MODEL, max_tokens=1024)
    return _llm


def get_vectorstore() -> Chroma:
    """Open the index, building it first if this host has never built one."""
    global _vectorstore
    if _vectorstore is None:
        if not os.path.exists(DB_PATH):
            import rag

            rag.build_index(DB_PATH)
        _vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=LocalEmbeddings(),
        )
    return _vectorstore


def _generate_llm_question(topic: str, level: str, asked: List[str]) -> str:
    prompt = ChatPromptTemplate.from_template("""
Generate ONE unique {level} level interview question about {topic}.

Difficulty guidelines:
- easy: Basic concepts, definitions, simple syntax
- intermediate: Practical usage, comparisons, common patterns
- advanced: Complex scenarios, edge cases, optimization
- expert: System design, internals, advanced problem-solving

IMPORTANT: Generate a COMPLETELY DIFFERENT question from these already asked:
{asked}

Respond with ONLY the question text, nothing else.
""")
    result = (prompt | get_llm()).invoke({
        "topic": topic,
        "level": level,
        "asked": "\n".join(asked[-10:]) if asked else "None",
    })
    return str(result.content).strip()


def get_question(topic: str, asked: List[str], level: str, use_llm: bool = False) -> str:
    available = [q for q in QUESTIONS.get(topic.lower(), []) if q not in asked]
    if not available or use_llm:
        return _generate_llm_question(topic, level, asked)
    return random.choice(available)


def start_interview(user_id: str, topic: str) -> Dict[str, Any]:
    question = get_question(topic, [], "easy")
    sessions[user_id] = {
        "topic": topic,
        "question": question,
        "asked": [question],
        "level_index": 0,
        "level_score": 0,
        "level_questions": 0,
        "total_score": 0,
        "answer_count": 0,
    }
    return {"question": question, "level": "easy"}


def answer(user_id: str, answer_text: str) -> Dict[str, Any]:
    session = sessions.get(user_id)
    if not session:
        return {"error": "Start interview first"}

    question = session["question"]

    docs = get_vectorstore().similarity_search(question, k=1)
    reference = docs[0].page_content if docs else "No reference found."

    eval_prompt = ChatPromptTemplate.from_template("""
You are a STRICT interview evaluator. Rate the answer honestly.

Question: {question}
Reference: {reference}
Candidate's Answer: {answer}

SCORING RULES (BE STRICT):
- 0/5: Irrelevant, gibberish, random text, or completely wrong answer
- 1/5: Barely related, mostly incorrect
- 2/5: Partially correct but missing key points
- 3/5: Correct basics but incomplete
- 4/5: Good answer with minor gaps
- 5/5: Excellent, complete answer

IMPORTANT: If the answer is nonsense, random words, or doesn't address the question AT ALL, give 0/5.

Respond in EXACTLY this format:
Score: X/5
Feedback: Brief feedback here.
""")

    result = (eval_prompt | get_llm()).invoke({
        "question": question,
        "reference": reference,
        "answer": answer_text,
    })
    evaluation = str(result.content)

    try:
        score = float(evaluation.split("Score:")[1].split("/")[0].strip())
    except (IndexError, ValueError):
        score = 0.0

    session["level_score"] += score
    session["level_questions"] += 1
    session["total_score"] += score
    session["answer_count"] += 1

    level_completed = None
    old_level = LEVELS[session["level_index"]]

    if session["level_questions"] == 5:
        # 15/25 (60%) is the bar to move up a level.
        if session["level_score"] >= 15 and session["level_index"] < len(LEVELS) - 1:
            level_completed = f"Level {session['level_index'] + 1} ({old_level.title()})"
            session["level_index"] += 1
        session["level_score"] = 0
        session["level_questions"] = 0

    new_level = LEVELS[session["level_index"]]

    next_q = get_question(session["topic"], session["asked"], new_level, use_llm=True)
    session["question"] = next_q
    session["asked"].append(next_q)

    return {
        "evaluation": evaluation,
        "next_question": next_q,
        "level": new_level,
        "level_completed": level_completed,
    }


def end_interview(user_id: str) -> Dict[str, Any]:
    session = sessions.get(user_id)
    if not session:
        return {"error": "No active interview"}

    total_questions = session["answer_count"]
    avg_score = session["total_score"] / total_questions if total_questions else 0
    final_level = LEVELS[session["level_index"]]

    assess_prompt = ChatPromptTemplate.from_template("""
Based on the interview performance, give a 2-3 sentence assessment.

Topic: {topic}
Questions Answered: {total}
Average Score: {avg}/5
Final Level: {level}

Write a brief, encouraging assessment of the candidate's readiness level.
Example: "You show strong fundamentals in Python basics. Ready for junior developer roles. Focus on practicing advanced topics like decorators and OOP."

Assessment:""")

    result = (assess_prompt | get_llm()).invoke({
        "topic": session["topic"],
        "total": total_questions,
        "avg": round(avg_score, 1),
        "level": final_level,
    })
    assessment = str(result.content).strip()

    del sessions[user_id]

    return {
        "total_questions": total_questions,
        "avg_score": avg_score,
        "final_level": final_level,
        "assessment": assessment,
    }
