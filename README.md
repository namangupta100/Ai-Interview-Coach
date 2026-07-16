---
title: AI Interview Coach
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.59.2
app_file: frontend.py
pinned: false
---

<div align="center">

# 🤖 AI Interview Coach

**Your Personal AI-Powered Mock Interview Platform**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)

*Practice technical interviews with adaptive difficulty, real-time voice interaction, and intelligent feedback powered by LLMs and RAG.*

### 🚀 [**Try it live → ai-interview-coach-taupe-ten.vercel.app**](https://ai-interview-coach-taupe-ten.vercel.app)

---

[Live Demo](https://ai-interview-coach-taupe-ten.vercel.app) • [Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Usage](#-usage) • [API](#-api-reference)

</div>

---

## 📋 Overview

AI Interview Coach is an intelligent mock interview platform that helps you prepare for technical interviews. It uses Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Voice AI to create a realistic interview experience with:

- **Dynamic question generation** that adapts to your skill level
- **Real-time voice interaction** - speak your answers naturally
- **Intelligent evaluation** with detailed feedback
- **Progressive difficulty levels** from Easy to Expert
- **Comprehensive performance reports**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Adaptive Difficulty** | Automatically adjusts from Easy → Intermediate → Advanced → Expert based on your performance |
| 🗣️ **Voice Mode** | Speak your answers using speech-to-text; hear questions via text-to-speech |
| 🧠 **RAG-Powered** | Uses vector database for accurate reference answers and context |
| 📊 **Real-time Scoring** | Get instant scores (0-5) with detailed feedback for each answer |
| 🎓 **Multiple Topics** | Python, Data Structures, Machine Learning, and more |
| 📈 **Performance Reports** | Final assessment with overall score and improvement insights |
| 🔌 **MCP Server** | Model Context Protocol integration for AI assistant tools |

---

## 🛠️ Tech Stack

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│                   Streamlit (UI/UX)                         │
├─────────────────────────────────────────────────────────────┤
│                         API                                  │
│                   FastAPI (REST)                            │
├─────────────────────────────────────────────────────────────┤
│                     AI & ML Layer                           │
│  LangChain  │  Claude (LLM)  │  ChromaDB  │  Voice AI      │
└─────────────────────────────────────────────────────────────┘
```

- **Backend:** FastAPI (optional — the Streamlit app calls the engine directly)
- **Frontend:** Streamlit
- **LLM:** Claude (`claude-opus-4-8`) via `langchain-anthropic`
- **Embeddings:** local ONNX all-MiniLM-L6-v2, bundled with ChromaDB (no API key)
- **Vector DB:** ChromaDB
- **Voice AI:** gTTS + SpeechRecognition

---

## 📁 Project Structure

```
Ai-Interview-Coach/
├── frontend.py         # Streamlit UI — the app entrypoint
├── engine.py           # Interview logic: questions, RAG lookup, Claude scoring
├── embeddings.py       # Local ONNX embeddings (no API key needed)
├── app.py              # Optional FastAPI wrapper over engine.py
├── rag.py              # Builds the vector DB from data/dataset.csv
├── voice_ai.py         # Text-to-Speech & Speech-to-Text module
├── mcp_server.py       # MCP server for AI assistant integration
├── clean_data.py       # Data preprocessing utilities
├── requirements.txt    # Python dependencies
├── data/
│   ├── dataset.csv         # Raw interview Q&A dataset
│   └── clean_dataset.json  # Processed dataset
└── vectordb/           # ChromaDB vector store
```

---

## 📦 Prerequisites

Before you begin, ensure you have:

- **Python 3.11** installed
- An **Anthropic API key** ([console.anthropic.com](https://console.anthropic.com)), exported as:
  ```bash
  export ANTHROPIC_API_KEY=sk-ant-...
  ```

Embeddings run locally (ChromaDB's bundled ONNX model), so they need no key and no GPU.

### System Dependencies (for Voice AI)

<details>
<summary><b>macOS</b></summary>

```bash
brew install portaudio
```

</details>

<details>
<summary><b>Ubuntu/Debian</b></summary>

```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

</details>

<details>
<summary><b>Windows</b></summary>

PyAudio should install automatically. If issues occur, download the wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).

</details>

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Ai-Interview-Coach.git
cd Ai-Interview-Coach
```

### 2. Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate    # macOS/Linux
# OR
.\venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your API Key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Build the Vector Database (optional)

```bash
python rag.py
```

> The app builds this automatically on first use if it's missing, so this step is
> only to get the wait out of the way up front. The first build downloads the
> local embedding model (~83MB), so give it a minute.

---

## ⚡ Quick Start

```bash
streamlit run frontend.py
```

> Opens at `http://localhost:8501`. That's the whole app — the Streamlit UI calls
> `engine.py` in-process, so there's no separate backend to start.

Only if you want the REST API as well:

```bash
uvicorn app:app --reload --port 8000
```

---

## ☁️ Deploying to Hugging Face Spaces

1. Create a new Space → SDK **Streamlit**.
2. Push this repo to the Space (the YAML header in this README configures it, and
   sets `app_file: frontend.py`).
3. In **Settings → Variables and secrets**, add a secret `ANTHROPIC_API_KEY`.

The vector index is **not** committed — the app builds it from `data/dataset.csv`
on first use if it's missing, so the Space is self-sufficient. Expect the first
interview to take an extra minute while it downloads the embedding model (~83MB)
and embeds the dataset. Every later run reuses the index.

---

## 📖 Usage

### Starting an Interview

1. **Open the app** in your browser (`http://localhost:8501`)
2. **Select a topic:** Python, Data Structures, or Machine Learning
3. **Click "Start Interview"** to begin

### Answering Questions

```
┌────────────────────────────────────────────┐
│  📋 Question displayed in beautiful box    │
├────────────────────────────────────────────┤
│  ✍️  Type or 🎤 Speak your answer          │
├────────────────────────────────────────────┤
│  🚀 Click Submit                           │
└────────────────────────────────────────────┘
```

### Voice Mode (Optional)

1. **Enable Voice Mode** using the checkbox
2. **Auto-Speak:** Questions are read aloud automatically
3. **Record Answer:** Click START → Speak → Click STOP
4. **Edit if needed:** Modify the transcribed text before submitting

### Scoring System

| Score | Meaning |
|-------|---------|
| ⭐⭐⭐⭐⭐ 5/5 | Excellent, complete answer |
| ⭐⭐⭐⭐ 4/5 | Good answer with minor gaps |
| ⭐⭐⭐ 3/5 | Correct basics, incomplete |
| ⭐⭐ 2/5 | Partially correct |
| ⭐ 1/5 | Barely related |
| 0/5 | Incorrect or irrelevant |

### Level Progression

- Answer **5 questions** per level
- Score **15/25 (60%)** or higher to advance
- Levels: `Easy` → `Intermediate` → `Advanced` → `Expert`

### Ending the Interview

Click **"End Interview"** to receive your final performance report including:
- Total questions answered
- Average score
- Final difficulty level achieved
- Personalized assessment

---

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/start-interview` | Start a new interview session |
| `POST` | `/answer` | Submit an answer for evaluation |
| `GET` | `/end-interview` | End session and get final report |

### Example Requests

<details>
<summary><b>Start Interview</b></summary>

```bash
curl "http://127.0.0.1:8000/start-interview?user_id=1&topic=Python"
```

Response:
```json
{
  "question": "What is Python?",
  "level": "easy"
}
```

</details>

<details>
<summary><b>Submit Answer</b></summary>

```bash
curl -X POST "http://127.0.0.1:8000/answer?user_id=1&answer=Python%20is%20a%20programming%20language"
```

Response:
```json
{
  "evaluation": "Score: 4/5\nFeedback: Good basic answer...",
  "next_question": "Explain list vs tuple.",
  "level": "easy",
  "level_completed": null
}
```

</details>

### Accounts & admin

Interviews require a signed-in account. The browser posts to `POST /api/interview`
with an `action`:

| Action | Body | Notes |
|--------|------|-------|
| `register` | `{email, password, name}` | Returns `{token, user}`. Password ≥ 6 chars. |
| `login` | `{email, password}` | Returns `{token, user}`. |
| `me` | `{token}` | Validates a token; returns the current `user`. |
| `admin_users` | `{token}` | **Admin/owner only.** Returns every user with their usage, plus an `analytics` summary (totals, active users, top topics, level distribution). |
| `user_history` | `{token, uid, email}` | **Admin/owner only.** Returns one user's answered questions, answers, and scores. |
| `start` / `answer` / `skip` / `end` | `{…, token}` | Require a valid `token`; usage + history are recorded per user. |

**Roles.** `owner` > `admin` > `user`. Whoever registers with `OWNER_EMAIL`
(default `namangupta@232004`) becomes the **owner**. The `ADMIN_EMAIL` account is
always an admin; if unset, the **first person to register** is made admin so there's
always one. **Owners and admins land straight on a Dashboard** when they sign in
(regular users go to the practice screen). The dashboard shows analytics tiles
(total users, interviews, answers scored, skips, overall average score, users active
in the last 7 days), breakdown bars for **top topics** and **highest level reached**,
and a table of every user (interviews, answers, skips, average score, level, topics,
last active). Each row has a **History** button that drills into that user's recorded
questions, their answers, the score, level, topic, and timestamp. A **Practice mode →**
button switches to taking interviews.

Auth is stdlib-only: PBKDF2 password hashing + HMAC-signed session tokens. Users and
usage live in a KV store — set `KV_REST_API_URL` / `KV_REST_API_TOKEN` (Vercel KV /
Upstash) in production, or rely on the local-JSON fallback for dev. See
[`.env.example`](.env.example) for all auth variables.

---

## 🔧 MCP Server

The project includes an MCP (Model Context Protocol) server for integration with AI assistants like Claude.

### Available Tools

| Tool | Description |
|------|-------------|
| `ask_ai` | RAG-powered Q&A from the knowledge base |
| `generate_question` | Generate interview questions by topic |
| `evaluate_answer` | Evaluate an answer with score and feedback |

### Running the MCP Server

```bash
python mcp_server.py
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ for interview preparation**

⭐ Star this repo if you find it helpful!

</div>
