import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
from app.agent.agent import ask

load_dotenv()

app = FastAPI(
    title="Antioquia Flood Risk Advisor",
    description="AI agent monitoring real-time river levels and precipitation across Antioquia",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


class AgentResponse(BaseModel):
    question: str
    answer: str


@app.get("/")
def root():
    return {
        "status": "online",
        "description": "Antioquia Flood Risk Advisor Agent",
        "endpoints": {
            "ask": "POST /ask — ask the agent a question",
            "health": "GET /health — check service status",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "qdrant": os.getenv("QDRANT_URL", "http://localhost:6333"),
    }


@app.post("/ask", response_model=AgentResponse)
def ask_agent(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = ask(request.question)
        return AgentResponse(**result)
    except Exception as e:
        logging.exception("Error in /ask handler")
        raise HTTPException(status_code=500, detail=str(e))
