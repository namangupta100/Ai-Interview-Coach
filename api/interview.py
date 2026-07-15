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

import base64
import csv
import hashlib
import hmac
import json
import os
import re
import secrets
import time
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
    """Always AI-generated, for ANY topic — no fixed question bank.

    Difficulty is tied to the level and escalates: easy (1/4) -> expert (4/4).
    """
    stage = LEVELS.index(level) + 1 if level in LEVELS else 1
    prompt = f"""
You are an expert technical interviewer running a progressive interview about "{topic}".
This is difficulty stage {stage} of {len(LEVELS)}: **{level.upper()}**.

Generate ONE interview question calibrated PRECISELY to the {level} level —
not easier, not harder. Difficulty MUST clearly escalate with the stage:
- easy (1/4): VERY simple. A single short one-line question about ONE basic
  concept or definition a complete beginner would know. Maximum ~12 words.
  No multi-part questions, no "and how/why does it differ", no scenarios.
  Examples of the right easy style: "What is a variable?", "What is a loop?",
  "What does the print function do?".
- intermediate (2/4): practical usage, comparisons, or common patterns.
- advanced (3/4): complex scenarios, edge cases, trade-offs, or optimization.
- expert (4/4): deep internals, system design, or hard real-world problem-solving.

Keep it a single focused question. For easy especially, keep it to ONE short line.

Do NOT repeat or closely resemble any of these already-asked questions:
{chr(10).join(asked[-10:]) if asked else "None"}

Respond with ONLY the question text, nothing else.
"""
    return ask_ai(prompt)


# =========================
# Persistent store (users + usage)
# =========================
# The interview itself is stateless, but auth needs to remember people. We use a
# tiny key/value layer with two interchangeable backends, chosen at runtime:
#   * Upstash Redis REST — set KV_REST_API_URL + KV_REST_API_TOKEN (exactly what
#     Vercel KV / Upstash provision). Durable across serverless instances, still
#     zero pip dependencies because it's just an HTTPS POST via urllib.
#   * Local JSON file — a zero-setup fallback for local dev. NOT durable on
#     Vercel (its filesystem is ephemeral), so production must set the KV vars.
_KV_URL = os.getenv("KV_REST_API_URL") or os.getenv("UPSTASH_REDIS_REST_URL")
_KV_TOKEN = os.getenv("KV_REST_API_TOKEN") or os.getenv("UPSTASH_REDIS_REST_TOKEN")
_KV_ON = bool(_KV_URL and _KV_TOKEN)

_local_cache: Optional[Dict[str, Any]] = None


def _kv_command(*args: Any) -> Any:
    """Run one Redis command over the Upstash REST API; return its `result`."""
    body = json.dumps([str(a) for a in args]).encode("utf-8")
    req = urllib.request.Request(
        _KV_URL,
        data=body,
        headers={"Authorization": f"Bearer {_KV_TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read()).get("result")


def _local_path() -> str:
    override = os.getenv("AUTH_STORE_PATH")
    if override:
        return override
    # /tmp is the only writable dir on Vercel; locally keep it beside the repo.
    base = "/tmp" if os.path.isdir("/tmp") else os.path.dirname(__file__)
    return os.path.join(base, "authstore.json")


def _local() -> Dict[str, Any]:
    global _local_cache
    if _local_cache is None:
        try:
            with open(_local_path(), encoding="utf-8") as f:
                _local_cache = json.load(f)
        except (OSError, ValueError):
            _local_cache = {"kv": {}, "sets": {}}
    return _local_cache


def _local_save() -> None:
    with open(_local_path(), "w", encoding="utf-8") as f:
        json.dump(_local_cache, f)


def kv_get(key: str) -> Optional[str]:
    if _KV_ON:
        return _kv_command("GET", key)
    return _local()["kv"].get(key)


def kv_set(key: str, value: str) -> None:
    if _KV_ON:
        _kv_command("SET", key, value)
        return
    _local()["kv"][key] = value
    _local_save()


def set_add(setkey: str, member: str) -> None:
    if _KV_ON:
        _kv_command("SADD", setkey, member)
        return
    members = _local()["sets"].setdefault(setkey, [])
    if member not in members:
        members.append(member)
        _local_save()


def set_members(setkey: str) -> List[str]:
    if _KV_ON:
        return _kv_command("SMEMBERS", setkey) or []
    return list(_local()["sets"].get(setkey, []))


def _get_json(key: str) -> Optional[Dict[str, Any]]:
    raw = kv_get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _set_json(key: str, obj: Dict[str, Any]) -> None:
    kv_set(key, json.dumps(obj))


# =========================
# Auth (stdlib only)
# =========================
# Sessions are stateless, signed tokens: base64url(payload).base64url(HMAC). The
# secret MUST be stable and shared across instances, so it comes from the
# environment; set AUTH_SECRET in production or tokens won't survive a redeploy.
AUTH_SECRET = os.getenv("AUTH_SECRET", "dev-insecure-secret-change-me")
TOKEN_TTL = 60 * 60 * 24 * 30  # 30 days

# The single "owner" account — the top role, always above any admin. Whoever
# registers with this email becomes the owner (set OWNER_EMAIL to override).
OWNER_EMAIL = (os.getenv("OWNER_EMAIL", "namangupta@232004")).strip().lower()

# Roles allowed to open the admin dashboard and drill into user history.
ADMIN_ROLES = ("admin", "owner")


class AuthError(Exception):
    """Raised for auth/permission failures; carries the HTTP status to return."""

    def __init__(self, message: str, status: int = 401):
        super().__init__(message)
        self.status = status


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64u_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def hash_password(password: str, salt: Optional[str] = None) -> tuple:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000)
    return salt, dk.hex()


def verify_password(password: str, salt: str, expected: str) -> bool:
    _, actual = hash_password(password, salt)
    return hmac.compare_digest(actual, expected)


def make_token(user: Dict[str, Any]) -> str:
    payload = {
        "uid": user["id"],
        "email": user["email"],
        "role": user["role"],
        "exp": int(time.time()) + TOKEN_TTL,
    }
    body = _b64u(json.dumps(payload).encode("utf-8"))
    sig = _b64u(hmac.new(AUTH_SECRET.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{sig}"


def parse_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        body, sig = token.split(".", 1)
        expected = _b64u(hmac.new(AUTH_SECRET.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_b64u_decode(body))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def require_user(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = parse_token(data.get("token", "") or "")
    if not payload:
        raise AuthError("Please sign in to continue.")
    return payload


# =========================
# Users + usage
# =========================
def _email_key(email: str) -> str:
    return "user:email:" + email.strip().lower()


def _usage_key(uid: str) -> str:
    return "usage:" + uid


def _blank_usage() -> Dict[str, Any]:
    return {
        "interviews": 0,
        "answered": 0,
        "skipped": 0,
        "score_sum": 0.0,
        "topics": [],
        "highest_level": "easy",
        "last_active": 0,
    }


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    return _get_json(_email_key(email))


def public_user(user: Dict[str, Any]) -> Dict[str, Any]:
    """Everything about a user EXCEPT the password salt/hash."""
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "role": user["role"],
        "created_at": user["created_at"],
    }


def create_user(email: str, password: str, name: str) -> Dict[str, Any]:
    email = email.strip().lower()
    if not email or "@" not in email:
        raise ValueError("A valid email is required.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    if get_user_by_email(email):
        raise ValueError("An account with that email already exists.")

    salt, pw = hash_password(password)
    # Role assignment, highest wins:
    #   * OWNER_EMAIL           -> owner (the top account)
    #   * ADMIN_EMAIL / first   -> admin (there is always at least one admin)
    #   * everyone else         -> user
    admin_email = (os.getenv("ADMIN_EMAIL") or "").strip().lower()
    is_first = not set_members("users:all")
    if email == OWNER_EMAIL:
        role = "owner"
    elif email == admin_email or is_first:
        role = "admin"
    else:
        role = "user"

    user = {
        "id": secrets.token_hex(8),
        "email": email,
        "name": (name or "").strip() or email.split("@")[0],
        "salt": salt,
        "pw": pw,
        "role": role,
        "created_at": int(time.time()),
    }
    _set_json(_email_key(email), user)
    set_add("users:all", email)
    _set_json(_usage_key(user["id"]), _blank_usage())
    return user


def record_usage(
    uid: str,
    *,
    interviews: int = 0,
    answered: int = 0,
    skipped: int = 0,
    score: float = 0.0,
    topic: Optional[str] = None,
    level: Optional[str] = None,
) -> None:
    """Best-effort usage counters for the admin dashboard. Read-modify-write is
    not atomic, but for a practice app the occasional lost increment is fine."""
    u = _get_json(_usage_key(uid)) or _blank_usage()
    u["interviews"] = u.get("interviews", 0) + interviews
    u["answered"] = u.get("answered", 0) + answered
    u["skipped"] = u.get("skipped", 0) + skipped
    u["score_sum"] = u.get("score_sum", 0.0) + score
    if topic and topic not in u["topics"]:
        u["topics"].append(topic)
    if level and level in LEVELS and LEVELS.index(level) > LEVELS.index(u.get("highest_level", "easy")):
        u["highest_level"] = level
    u["last_active"] = int(time.time())
    _set_json(_usage_key(uid), u)


# =========================
# Per-user interview history
# =========================
# Every answered question is appended here so an admin can drill into exactly
# what a user was asked, what they answered, and how it scored. Kept bounded so
# a heavy user can't grow one key without limit.
HISTORY_LIMIT = 200


def _history_key(uid: str) -> str:
    return "history:" + uid


def get_history(uid: str) -> List[Dict[str, Any]]:
    raw = kv_get(_history_key(uid))
    try:
        hist = json.loads(raw) if raw else []
    except (TypeError, ValueError):
        hist = []
    return hist if isinstance(hist, list) else []


def append_history(uid: str, entry: Dict[str, Any]) -> None:
    hist = get_history(uid)
    hist.append(entry)
    if len(hist) > HISTORY_LIMIT:
        hist = hist[-HISTORY_LIMIT:]
    kv_set(_history_key(uid), json.dumps(hist))


# =========================
# Auth actions
# =========================
def action_register(data: Dict[str, Any]) -> Dict[str, Any]:
    user = create_user(data.get("email", ""), data.get("password", ""), data.get("name", ""))
    return {"token": make_token(user), "user": public_user(user)}


def action_login(data: Dict[str, Any]) -> Dict[str, Any]:
    user = get_user_by_email(data.get("email", ""))
    if not user or not verify_password(data.get("password", ""), user["salt"], user["pw"]):
        raise AuthError("Incorrect email or password.")
    return {"token": make_token(user), "user": public_user(user)}


def action_me(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = require_user(data)
    user = get_user_by_email(payload["email"])
    if not user:
        raise AuthError("Session expired. Please sign in again.")
    return {"user": public_user(user)}


def action_admin_users(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = require_user(data)
    if payload.get("role") not in ADMIN_ROLES:
        raise AuthError("Admin access required.", status=403)

    rows = []
    # Aggregate analytics accumulated in the same pass over every user.
    roles = {"owner": 0, "admin": 0, "user": 0}
    level_counts = {lvl: 0 for lvl in LEVELS}
    topic_counts: Dict[str, int] = {}
    total_interviews = total_answered = total_skipped = 0
    total_score_sum = 0.0
    now = int(time.time())
    active_24h = active_7d = 0

    for email in set_members("users:all"):
        user = get_user_by_email(email)
        if not user:
            continue
        usage = _get_json(_usage_key(user["id"])) or _blank_usage()
        answered = usage.get("answered", 0)
        interviews = usage.get("interviews", 0)
        skipped = usage.get("skipped", 0)
        score_sum = usage.get("score_sum", 0.0)
        highest = usage.get("highest_level", "easy")
        last = usage.get("last_active", 0)
        topics = usage.get("topics", [])

        rows.append({
            **public_user(user),
            "interviews": interviews,
            "answered": answered,
            "skipped": skipped,
            "avg_score": round(score_sum / answered, 2) if answered else 0,
            "highest_level": highest,
            "topics": topics,
            "last_active": last,
        })

        roles[user["role"]] = roles.get(user["role"], 0) + 1
        total_interviews += interviews
        total_answered += answered
        total_skipped += skipped
        total_score_sum += score_sum
        if highest in level_counts:
            level_counts[highest] += 1
        for t in topics:
            topic_counts[t] = topic_counts.get(t, 0) + 1
        if last and now - last <= 86400:
            active_24h += 1
        if last and now - last <= 7 * 86400:
            active_7d += 1

    rows.sort(key=lambda r: r["last_active"], reverse=True)
    top_topics = sorted(topic_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
    analytics = {
        "total_users": len(rows),
        "roles": roles,
        "total_interviews": total_interviews,
        "total_answered": total_answered,
        "total_skipped": total_skipped,
        "avg_score": round(total_score_sum / total_answered, 2) if total_answered else 0,
        "active_24h": active_24h,
        "active_7d": active_7d,
        "top_topics": [{"topic": t, "users": c} for t, c in top_topics],
        "level_distribution": level_counts,
    }
    return {"users": rows, "count": len(rows), "analytics": analytics}


def action_admin_user_history(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = require_user(data)
    if payload.get("role") not in ADMIN_ROLES:
        raise AuthError("Admin access required.", status=403)

    uid = (data.get("uid") or "").strip()
    email = (data.get("email") or "").strip().lower()
    if not uid:
        raise ValueError("A user id is required.")

    user = get_user_by_email(email) if email else None
    hist = get_history(uid)
    hist.sort(key=lambda h: h.get("at", 0), reverse=True)
    return {
        "user": public_user(user) if user else {"id": uid, "email": email, "name": email or uid},
        "history": hist,
        "count": len(hist),
    }


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


def action_skip(state: Dict[str, Any]) -> Dict[str, Any]:
    """Swap the current question for a fresh one at the same level.

    A skip is not an answer: it doesn't count toward the 5-per-level tally, the
    level score, or the running average. The candidate simply gets a different
    question at the level they're already on.
    """
    level = LEVELS[state["level_index"]]
    next_q = get_question(state["topic"], state["asked"], level)
    state["question"] = next_q
    state["asked"].append(next_q)
    return {"question": next_q, "level": level, "state": state}


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

    # Public auth actions.
    if action == "register":
        return action_register(data)
    if action == "login":
        return action_login(data)
    if action == "me":
        return action_me(data)
    if action == "admin_users":
        return action_admin_users(data)
    if action == "user_history":
        return action_admin_user_history(data)

    # Everything below is the interview itself and requires a signed-in user.
    # Usage is attributed to the token's user for the admin dashboard.
    uid = require_user(data)["uid"]
    if action == "start":
        topic = data.get("topic", "Python")
        result = action_start(topic)
        record_usage(uid, interviews=1, topic=topic)
        return result
    if action == "answer":
        # Capture what was actually answered before action_answer overwrites
        # state["question"] with the next question / advances the level.
        st = data["state"]
        answered_q = st.get("question", "")
        idx = st.get("level_index", 0)
        asked_level = LEVELS[idx] if 0 <= idx < len(LEVELS) else "easy"
        topic = st.get("topic", "")
        answer_text = data.get("answer", "")

        result = action_answer(answer_text, st)
        record_usage(uid, answered=1, score=result["score"], level=result["level"])
        append_history(uid, {
            "question": answered_q,
            "answer": answer_text,
            "score": result["score"],
            "level": asked_level,
            "topic": topic,
            "at": int(time.time()),
        })
        return result
    if action == "skip":
        result = action_skip(data["state"])
        record_usage(uid, skipped=1)
        return result
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
        except AuthError as exc:  # 401 sign-in required / 403 admin only
            self._send(exc.status, {"error": str(exc)})
        except ValueError as exc:  # bad input (registration validation, bad JSON)
            self._send(400, {"error": str(exc)})
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
            "auth": {
                "store": "upstash-kv" if _KV_ON else "local-json (not durable on Vercel)",
                "secret_configured": bool(os.getenv("AUTH_SECRET")),
                "admin_email_configured": bool(os.getenv("ADMIN_EMAIL")),
            },
        })
