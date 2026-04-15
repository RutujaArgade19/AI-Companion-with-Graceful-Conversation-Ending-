"""
FastAPI Backend — AI Companion (Mock Mode)
==========================================

Usage:
    pip install fastapi uvicorn pydantic
    uvicorn backend:app --reload --port 8000
    Then open: http://localhost:8000
"""

import time
import uuid
import json
import random
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

# ── Config ────────────────────────────────────────────────────────────────────
LOG_PATH = Path("./logs/conversations.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

sessions: dict = {}

app = FastAPI(title="AI Companion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ── Mock responses ────────────────────────────────────────────────────────────
BASE_RESPONSES = [
    "That's really interesting! Tell me more about that.",
    "I see what you mean. Could you elaborate a bit further?",
    "That's a fascinating perspective. What else is on your mind?",
    "Absolutely! I find that very intriguing. Tell me more.",
    "I understand completely. So what else would you like to talk about?",
    "That's wonderful to hear. Please, do go on.",
    "Very interesting indeed! What other thoughts do you have?",
    "I appreciate you sharing that. What more can you tell me?",
    "That's quite thoughtful. Tell me more about that.",
    "I see, I see. And what else would you like to discuss today?",
]

FINETUNED_RESPONSES = [
    "That's so lovely to hear! How have things been going for you lately?",
    "It sounds like you've had quite a day! What's been the highlight?",
    "I'm glad we got to chat about that. Is there anything else on your mind?",
    "That really warms my heart. You know, it's been such a pleasure talking with you today.",
    "I feel like we've covered so much ground — you've given me a lot to think about!",
    "It's been truly wonderful chatting with you. I hope your day continues well.",
]

FINETUNED_ENDING = (
    "It's been such a joy talking with you today. "
    "I hope you have a wonderful rest of your day — take good care of yourself!"
)


def generate_reply(history: list[dict], model_type: str, turn_count: int) -> tuple[str, bool]:
    if model_type == "finetuned":
        if turn_count >= random.randint(6, 8):
            return FINETUNED_ENDING, True
        idx = min(turn_count - 1, len(FINETUNED_RESPONSES) - 1)
        return FINETUNED_RESPONSES[idx], False
    else:
        return random.choice(BASE_RESPONSES), False


# ── Schemas ───────────────────────────────────────────────────────────────────
class StartRequest(BaseModel):
    model_type: str = "finetuned"

class StartResponse(BaseModel):
    session_id: str
    model_type: str
    greeting: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    ended: bool
    turn_count: int
    end_reason: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/session/start", response_model=StartResponse)
def start_session(req: StartRequest):
    model_type = req.model_type if req.model_type in ("base", "finetuned") else "finetuned"
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "model_type": model_type,
        "history": [],
        "turn_count": 0,
        "ended": False,
        "start_time": time.time(),
    }
    greeting = (
        "Hello! It's so wonderful to chat with you today. "
        "How are you feeling? Is there anything on your mind?"
    )
    sessions[session_id]["history"].append({"role": "assistant", "content": greeting})
    return StartResponse(session_id=session_id, model_type=model_type, greeting=greeting)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["ended"]:
        raise HTTPException(status_code=400, detail="Conversation has already ended")

    session["history"].append({"role": "user", "content": req.message})
    session["turn_count"] += 1

    reply, ended = generate_reply(
        session["history"],
        session["model_type"],
        session["turn_count"]
    )

    if session["turn_count"] >= 20 and not ended:
        reply += " It's been so lovely chatting! Take care of yourself."
        ended = True

    session["history"].append({"role": "assistant", "content": reply})
    session["ended"] = ended

    if ended:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({
                "session_id": req.session_id,
                "model_type": session["model_type"],
                "turn_count": session["turn_count"],
                "ended_naturally": ended,
                "timestamp": session["start_time"],
            }) + "\n")

    return ChatResponse(
        reply=reply,
        ended=ended,
        turn_count=session["turn_count"],
        end_reason="model_emitted_end_token" if ended and session["turn_count"] < 20
                   else ("max_turns_reached" if session["turn_count"] >= 20 else None),
    )


@app.get("/session/{session_id}/history")
def get_history(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"history": session["history"], "turn_count": session["turn_count"],
            "ended": session["ended"]}


@app.get("/stats")
def get_stats():
    total = len(sessions)
    ended = sum(1 for s in sessions.values() if s["ended"])
    avg_turns = (sum(s["turn_count"] for s in sessions.values()) / total) if total else 0
    return {
        "total_sessions": total,
        "ended_naturally": ended,
        "avg_turns_per_conversation": round(avg_turns, 2),
        "mode": "mock",
    }

@app.get("/health")
def health():
    return {"status": "ok", "mode": "mock"}
