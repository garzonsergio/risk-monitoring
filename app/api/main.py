import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from app.agent.agent import ask, get_radar_url, RADAR_BOUNDS

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
    radar_url: str
    radar_bounds: list

@app.get("/")
def root():
    return {
        "status": "online",
        "description": "Antioquia Flood Risk Advisor Agent",
        "endpoints": {
            "ask": "POST /ask — ask the agent a question",
            "health": "GET /health — check service status",
            "radar": "GET /radar — get current radar image URL",
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "qdrant": os.getenv("QDRANT_URL", "http://localhost:6333")}

@app.get("/radar")
def radar():
    return {
        "radar_url": get_radar_url(),
        "bounds": RADAR_BOUNDS,
        "description": "Real-time radar reflectivity over Antioquia",
    }

@app.post("/ask", response_model=AgentResponse)
def ask_agent(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = ask(request.question)
        return AgentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))