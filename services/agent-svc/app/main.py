"""
Agent Service — LLM agent, chat, tool orchestration.

Phase 5: Autonomous Biological Intelligence Platform (ABIP)
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agent Service", version="0.1.0")
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "healthy", "service": "agent", "version": "0.1.0"}


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str
    use_mock: Optional[bool] = None


@app.get("/agent/tools")
def list_tools():
    from app.agent.llm import list_tools
    return {"tools": list_tools()}


@app.post("/agent/chat")
def chat(req: ChatRequest):
    from app.agent.llm import chat as agent_chat
    return agent_chat(req.session_id, req.message, use_mock=req.use_mock)


@app.post("/agent/reset")
def reset_memory(session_id: str = Query("default")):
    from app.agent.llm import reset_memory
    reset_memory(session_id)
    return {"status": "reset", "session_id": session_id}


@app.get("/agent/history")
def history(session_id: str = Query("default")):
    from app.agent.llm import get_memory
    mem = get_memory(session_id)
    return {"session_id": session_id, "turns": len(mem.messages) // 2, "messages": mem.as_list()}
