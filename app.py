from fastapi import FastAPI
import random
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

# =========================
# 🔹 RAG SETUP
# =========================
DB_PATH = "vectordb"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
llm = ChatOllama(model="llama3")

# =========================
# 🔹 DATASET (CLEAN)
# =========================
QUESTIONS = {
    "python": [
        "What is Python?",
        "Explain list vs tuple.",
        "What are decorators?",
        "What is OOP in Python?",
        "What is a lambda function?"
    ],
    "data structures": [
        "What is a stack?",
        "What is a queue?",
        "Difference between array and linked list?",
        "What is a hash table?"
    ],
    "machine learning": [
        "What is overfitting?",
        "What is underfitting?",
        "Explain gradient descent.",
        "What is supervised learning?"
    ]
}

LEVELS = ["easy", "intermediate", "advanced", "expert"]

sessions = {}

# =========================
# 🔹 GET QUESTION
# =========================
def get_question(topic, asked, level, use_llm=False):
    topic_lower = topic.lower()
    questions = QUESTIONS.get(topic_lower, [])

    available = [q for q in questions if q not in asked]

    # If no questions left, generate with LLM
    if not available or use_llm:
        return generate_llm_question(topic, level, asked)

    return random.choice(available)


# =========================
# 🔹 LLM QUESTION GENERATOR
# =========================
def generate_llm_question(topic, level, asked):
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
    
    chain = prompt | llm
    result = chain.invoke({
        "topic": topic,
        "level": level,
        "asked": "\n".join(asked[-10:]) if asked else "None"  # Last 10 to keep prompt short
    })
    
    return str(result.content).strip()

# =========================
# 🔹 START INTERVIEW
# =========================
@app.get("/start-interview")
def start(user_id: str, topic: str):

    question = get_question(topic, [], "easy")

    sessions[user_id] = {
        "topic": topic,
        "question": question,
        "asked": [question],
        "level_index": 0,
        "level_score": 0,       # Score for current level (resets each level)
        "level_questions": 0,   # Questions in current level
        "total_score": 0,
        "answer_count": 0
    }

    return {
        "question": question,
        "level": "easy"
    }

# =========================
# 🔹 ANSWER
# =========================
@app.post("/answer")
def answer(user_id: str, answer: str):

    session = sessions.get(user_id)

    if not session:
        return {"error": "Start interview first"}

    question = session["question"]
    current_level = LEVELS[session["level_index"]]

    # 🔹 RAG: Get reference answer from vector DB
    docs = vectorstore.similarity_search(question, k=1)
    reference = docs[0].page_content if docs else "No reference found."

    # 🔹 LLM Evaluation (Score out of 5, short feedback)
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

    chain = eval_prompt | llm
    result = chain.invoke({
        "question": question,
        "reference": reference,
        "answer": answer
    })

    evaluation = str(result.content)

    # 🔹 Extract score and update level
    try:
        score = float(evaluation.split("Score:")[1].split("/")[0].strip())
    except:
        score = 0

    session["level_score"] += score
    session["level_questions"] += 1
    session["total_score"] += score
    session["answer_count"] += 1

    # Level check every 5 questions in current level
    level_completed = None
    old_level = LEVELS[session["level_index"]]
    
    if session["level_questions"] == 5:
        # Need 15/25 (60%) to advance
        if session["level_score"] >= 15 and session["level_index"] < len(LEVELS) - 1:
            level_completed = f"Level {session['level_index'] + 1} ({old_level.title()})"
            session["level_index"] += 1
        # Reset level counters for next/repeated level
        session["level_score"] = 0
        session["level_questions"] = 0

    new_level = LEVELS[session["level_index"]]

    # 🔹 Always use LLM for questions to avoid repeats
    # Questions already asked are tracked in session["asked"]
    next_q = get_question(session["topic"], session["asked"], new_level, use_llm=True)

    session["question"] = next_q
    session["asked"].append(next_q)

    return {
        "evaluation": evaluation,
        "next_question": next_q,
        "level": new_level,
        "level_completed": level_completed
    }


# =========================
# 🔹 END INTERVIEW
# =========================
@app.get("/end-interview")
def end_interview(user_id: str):
    session = sessions.get(user_id)
    
    if not session:
        return {"error": "No active interview"}
    
    total_questions = session["answer_count"]
    avg_score = session["total_score"] / total_questions if total_questions > 0 else 0
    final_level = LEVELS[session["level_index"]]
    
    # Generate assessment using LLM
    assess_prompt = ChatPromptTemplate.from_template("""
Based on the interview performance, give a 2-3 sentence assessment.

Topic: {topic}
Questions Answered: {total}
Average Score: {avg}/5
Final Level: {level}

Write a brief, encouraging assessment of the candidate's readiness level.
Example: "You show strong fundamentals in Python basics. Ready for junior developer roles. Focus on practicing advanced topics like decorators and OOP."

Assessment:""")
    
    chain = assess_prompt | llm
    result = chain.invoke({
        "topic": session["topic"],
        "total": total_questions,
        "avg": round(avg_score, 1),
        "level": final_level
    })
    
    assessment = str(result.content).strip()
    
    # Clear session
    del sessions[user_id]
    
    return {
        "total_questions": total_questions,
        "avg_score": avg_score,
        "final_level": final_level,
        "assessment": assessment
    }