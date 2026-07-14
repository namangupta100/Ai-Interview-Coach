"""Stateless interview engine as a single Vercel serverless function.

This is a from-scratch, Vercel-native rewrite of the old Streamlit engine.py.
Design differences that make it run on Vercel:

  * No server memory. The browser owns the session and posts it back on every
    request, so any cold serverless instance can serve any request.
  * No ChromaDB / onnxruntime. The "reference answer" that grades a reply comes
    from a keyword match over the bundled 200-row CSV instead of a 200MB vector
    stack that would blow Vercel's function-size limit.
  * No SDK at all. The model (Groq, free tier) is called over its OpenAI-compatible
    REST API with the standard-library `urllib`, so the function has zero pip
    dependencies and a tiny bundle.

One endpoint, POST /api/interview, dispatched on a JSON `action` field:
    {"action": "start",  "topic": "Python"}
    {"action": "answer", "answer": "...", "state": {...}}
    {"action": "end",    "state": {...}}
"""

import csv
import json
import os
import re
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional

# Both providers have a free tier. The app tries Groq first, then falls back to
# Gemini, so whichever key works keeps the interview running.
#   Groq key:   https://console.groq.com/keys
#   Gemini key: https://aistudio.google.com/apikey
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
# Reported by the health check only.
MODEL = GROQ_MODEL if os.getenv("GROQ_API_KEY") else GEMINI_MODEL

LEVELS = ["easy", "intermediate", "advanced", "expert"]

# Resolved once per warm instance.
_dataset: Optional[List[Dict[str, str]]] = None

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# A full, real browser UA: Groq sits behind Cloudflare, which 403s (error 1010)
# non-browser User-Agents (including the default "Python-urllib").
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _ask_groq(prompt: str, max_tokens: int) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    data = _post_json(
        GROQ_URL,
        {"Content-Type": "application/json", "Authorization": f"Bearer {key}", "User-Agent": _UA},
        {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
    )
    return data["choices"][0]["message"]["content"].strip()


def _ask_gemini(prompt: str, max_tokens: int) -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    data = _post_json(
        GEMINI_URL.format(model=GEMINI_MODEL),
        {"Content-Type": "application/json", "x-goog-api-key": key, "User-Agent": _UA},
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7},
        },
    )
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def ask_ai(prompt: str, max_tokens: int = 1024) -> str:
    """Try Groq, then Gemini. Raise a combined error only if all providers fail."""
    providers = [("Groq", _ask_groq), ("Gemini", _ask_gemini)]
    have_key = os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not have_key:
        raise RuntimeError(
            "No AI key is set. Add GROQ_API_KEY (https://console.groq.com/keys) "
            "or GEMINI_API_KEY (https://aistudio.google.com/apikey) in Vercel "
            "under Project Settings -> Environment Variables, then redeploy."
        )

    errors = []
    for name, fn in providers:
        try:
            return fn(prompt, max_tokens)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:200]
            errors.append(f"{name} {e.code}: {detail}")
        except RuntimeError as e:
            errors.append(f"{name}: {e}")  # missing key — skip to next provider
        except Exception as e:  # network / parse / empty completion
            errors.append(f"{name}: {e}")
    raise RuntimeError("All AI providers failed. " + " | ".join(errors))


# =========================
# Reference lookup (replaces the vector DB)
# =========================
def _dataset_path() -> str:
    """The CSV is bundled next to the repo root via vercel.json includeFiles."""
    here = os.path.dirname(__file__)
    for candidate in (
        os.path.join(here, "..", "data", "dataset.csv"),
        os.path.join(here, "data", "dataset.csv"),
        "data/dataset.csv",
    ):
        if os.path.exists(candidate):
            return candidate
    return os.path.join(here, "..", "data", "dataset.csv")


def load_dataset() -> List[Dict[str, str]]:
    global _dataset
    if _dataset is None:
        with open(_dataset_path(), encoding="latin-1", newline="") as f:
            _dataset = list(csv.DictReader(f))
    return _dataset


# Filler words that would otherwise make unrelated questions "match" on and/how/is.
_STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "and", "or", "of", "to", "in", "on",
    "how", "do", "does", "explain", "define", "give", "example", "between",
    "difference", "with", "for", "you", "your", "it", "its", "this", "that",
    "can", "be", "by", "as", "from", "which", "when", "why", "concept",
}


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS and len(t) > 1}


def find_reference(question: str) -> str:
    """Best keyword-overlap row from the Q&A corpus — a cheap stand-in for RAG."""
    q = _tokens(question)
    if not q:
        return "No reference found."
    best, best_score = None, 0
    for row in load_dataset():
        score = len(q & _tokens(row.get("Question", "")))
        if score > best_score:
            best, best_score = row, score
    if best and best_score > 0:
        return f"Question: {best['Question']} Answer: {best['Answer']}"
    return "No reference found."


# =========================
# Question generation
# =========================
def get_question(topic: str, asked: List[str], level: str) -> str:
    """Always AI-generated, for ANY topic — no fixed question bank."""
    prompt = f"""
You are an expert technical interviewer. Generate ONE unique {level} level
interview question about "{topic}".

Difficulty guidelines:
- easy: Basic concepts, definitions, simple syntax
- intermediate: Practical usage, comparisons, common patterns
- advanced: Complex scenarios, edge cases, optimization
- expert: System design, internals, advanced problem-solving

IMPORTANT: Generate a COMPLETELY DIFFERENT question from these already asked:
{chr(10).join(asked[-10:]) if asked else "None"}

Respond with ONLY the question text, nothing else.
"""
    return ask_ai(prompt)


# =========================
# Actions (stateless)
# =========================
def action_start(topic: str) -> Dict[str, Any]:
    question = get_question(topic, [], "easy")
    state = {
        "topic": topic,
        "question": question,
        "asked": [question],
        "level_index": 0,
        "level_score": 0,
        "level_questions": 0,
        "total_score": 0,
        "answer_count": 0,
    }
    return {"question": question, "level": "easy", "state": state}


def action_answer(answer_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    topic = state.get("topic", "the topic")
    # Optional hint from the local corpus; often empty for niche topics — that's
    # fine, the model grades from its own expert knowledge either way.
    reference = find_reference(question)
    hint = "" if reference == "No reference found." else f"\nReference (optional hint): {reference}"

    eval_prompt = f"""
You are a STRICT expert interviewer for {topic}. Use your own expert knowledge to
judge the candidate's answer to the question below. A reference may be provided as
a hint, but if it is missing or unrelated, rely entirely on your own expertise —
score the answer on its factual correctness and completeness for this topic.

Question: {question}{hint}
Candidate's Answer: {answer_text}

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
"""
    evaluation = ask_ai(eval_prompt)

    try:
        score = float(evaluation.split("Score:")[1].split("/")[0].strip())
    except (IndexError, ValueError):
        score = 0.0

    state["level_score"] += score
    state["level_questions"] += 1
    state["total_score"] += score
    state["answer_count"] += 1

    level_completed = None
    old_level = LEVELS[state["level_index"]]

    if state["level_questions"] == 5:
        # 15/25 (60%) is the bar to move up a level.
        if state["level_score"] >= 15 and state["level_index"] < len(LEVELS) - 1:
            level_completed = f"Level {state['level_index'] + 1} ({old_level.title()})"
            state["level_index"] += 1
        state["level_score"] = 0
        state["level_questions"] = 0

    new_level = LEVELS[state["level_index"]]

    next_q = get_question(state["topic"], state["asked"], new_level)
    state["question"] = next_q
    state["asked"].append(next_q)

    return {
        "evaluation": evaluation,
        "score": score,
        "next_question": next_q,
        "level": new_level,
        "level_completed": level_completed,
        "state": state,
    }


def action_end(state: Dict[str, Any]) -> Dict[str, Any]:
    total_questions = state["answer_count"]
    avg_score = state["total_score"] / total_questions if total_questions else 0
    final_level = LEVELS[state["level_index"]]

    assess_prompt = f"""
Based on the interview performance, give a 2-3 sentence assessment.

Topic: {state["topic"]}
Questions Answered: {total_questions}
Average Score: {round(avg_score, 1)}/5
Final Level: {final_level}

Write a brief, encouraging assessment of the candidate's readiness level.
Example: "You show strong fundamentals in Python basics. Ready for junior developer roles. Focus on practicing advanced topics like decorators and OOP."

Assessment:"""
    assessment = ask_ai(assess_prompt)

    return {
        "total_questions": total_questions,
        "avg_score": round(avg_score, 2),
        "final_level": final_level,
        "assessment": assessment,
    }


def dispatch(data: Dict[str, Any]) -> Dict[str, Any]:
    action = data.get("action")
    if action == "start":
        return action_start(data.get("topic", "Python"))
    if action == "answer":
        return action_answer(data.get("answer", ""), data["state"])
    if action == "end":
        return action_end(data["state"])
    return {"error": f"Unknown action: {action!r}"}


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: Dict[str, Any]) -> None:
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("content-length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw or b"{}")
            self._send(200, dispatch(data))
        except Exception as exc:  # surface the message to the client for debugging
            self._send(500, {"error": str(exc)})

    def do_GET(self) -> None:
        from urllib.parse import parse_qs, urlparse

        # /api/interview?selftest=1 actually calls each provider with a tiny
        # prompt and reports the real result — a definitive live diagnostic.
        if parse_qs(urlparse(self.path).query).get("selftest"):
            result = {}
            for name, fn in (("groq", _ask_groq), ("gemini", _ask_gemini)):
                try:
                    fn("Reply with the single word: OK", 8)
                    result[name] = "ok"
                except urllib.error.HTTPError as e:
                    body = e.read().decode("utf-8", "replace")[:220]
                    result[name] = f"{e.code}: {body}"
                except Exception as e:  # missing key / network / parse
                    result[name] = str(e)[:220]
            self._send(200, {"selftest": result})
            return

        self._send(200, {
            "ok": True,
            "service": "ai-interview-coach",
            "version": "selftest-v3",
            "providers": {
                "groq": bool(os.getenv("GROQ_API_KEY")),
                "gemini": bool(os.getenv("GEMINI_API_KEY")),
            },
            "model": MODEL,
        })
