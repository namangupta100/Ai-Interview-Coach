"""FastAPI wrapper around the interview engine.

The Streamlit UI calls engine.py directly, so this server is only needed if you
want to drive the interview over HTTP. Run with:
    uvicorn app:app --reload
"""

from fastapi import FastAPI

import engine

app = FastAPI()


@app.get("/start-interview")
def start(user_id: str, topic: str):
    return engine.start_interview(user_id, topic)


@app.post("/answer")
def answer(user_id: str, answer: str):
    return engine.answer(user_id, answer)


@app.get("/end-interview")
def end_interview(user_id: str):
    return engine.end_interview(user_id)
