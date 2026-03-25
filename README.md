<div align="center">

# 🤖 AI Interview Coach

**Your Personal AI-Powered Mock Interview Platform**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)

*Practice technical interviews with adaptive difficulty, real-time voice interaction, and intelligent feedback powered by LLMs and RAG.*

---

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Usage](#-usage) • [API](#-api-reference)

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
│  LangChain  │  Ollama (LLM)  │  ChromaDB  │  Voice AI      │
└─────────────────────────────────────────────────────────────┘
```

- **Backend:** FastAPI
- **Frontend:** Streamlit
- **LLM:** Ollama (Llama 3)
- **Embeddings:** nomic-embed-text
- **Vector DB:** ChromaDB
- **Voice AI:** gTTS + SpeechRecognition

---

## 📁 Project Structure

```
Ai-Interview-Coach/
├── app.py              # FastAPI backend server
├── frontend.py         # Streamlit UI application
├── rag.py              # RAG setup & vector DB initialization
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

- **Python 3.10+** installed
- **Ollama** installed and running ([Install Ollama](https://ollama.ai))
- **Required Ollama models:**
  ```bash
  ollama pull llama3
  ollama pull nomic-embed-text
  ```

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
python -m venv venv
source venv/bin/activate    # macOS/Linux
# OR
.\venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Initialize Vector Database

```bash
python rag.py
```

> ✅ You should see: `Vector DB created successfully!`

---

## ⚡ Quick Start

### Step 1: Start Ollama

Make sure Ollama is running:

```bash
ollama serve
```

### Step 2: Start the Backend Server

```bash
uvicorn app:app --reload --port 8000
```

> Backend will be available at `http://127.0.0.1:8000`

### Step 3: Start the Frontend (New Terminal)

```bash
streamlit run frontend.py
```

> Frontend will open automatically at `http://localhost:8501`

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
