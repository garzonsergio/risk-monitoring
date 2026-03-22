import os
import requests
from datetime import datetime, timezone

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from app.ingest.fetch_data import (
    fetch_latest_level,
    fetch_level_thresholds,
    fetch_latest_precipitation,
    get_radar_url,
    BASE_URL,
    PRECIP_THRESHOLDS_URL,
)

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "antioquia_risk"
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RADAR_BOUNDS = [[5.1, -76.6], [7.3, -74.3]]

# ── Embedding wrapper ─────────────────────────────────────────────────────────


class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()


# ── LLM + Retriever ───────────────────────────────────────────────────────────

# def get_llm():
#     return ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash",
#         google_api_key=GEMINI_API_KEY,
#         temperature=0.2,
#     )

# def get_llm():
#     return ChatOpenAI(
#         model="gpt-4o",
#         api_key=os.getenv("GITHUB_COPILOT_TOKEN"),
#         base_url="https://api.githubcopilot.com",
#         temperature=0.2,
#     )


def get_llm():
    return ChatOllama(
        model="llama3.1:8b",
        temperature=0.2,
    )


def get_retriever():
    client = QdrantClient(url=QDRANT_URL)
    store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=LocalEmbeddings(),
    )
    return store.as_retriever(search_kwargs={"k": 8})


# ── Tools ─────────────────────────────────────────────────────────────────────


@tool
def search_knowledge_base(query: str) -> str:
    """Search historical data, thresholds, and station info stored in the
    knowledge base. Use this for questions about past patterns, alert
    thresholds, station locations, or general context about Antioquia sensors."""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the knowledge base."
    return "\n\n---\n\n".join(d.page_content for d in docs)


@tool
def check_live_river_level(station_codigo: str) -> str:
    """Check the current live river level for a specific station and compare
    it against its alert thresholds. Use this when asked about current or
    real-time river conditions. station_codigo example: sn_1030"""
    latest = fetch_latest_level(station_codigo)
    if not latest:
        return f"Could not fetch live data for station {station_codigo}."

    thresholds = fetch_level_thresholds(station_codigo)
    current = float(latest["nivel"])
    fecha = latest["fecha"]

    if not thresholds:
        return (
            f"Station {station_codigo} current level: {current} cm "
            f"at {fecha}. Thresholds unavailable."
        )

    amarilla = thresholds["alerta_amarilla"]
    naranja = thresholds["alerta_naranja"]
    roja = thresholds["alerta_roja"]

    if current >= roja:
        status = f"🔴 RED ALERT — {current} cm exceeds red threshold of {roja} cm"
    elif current >= naranja:
        status = (
            f"🟠 ORANGE ALERT — {current} cm exceeds orange threshold of {naranja} cm"
        )
    elif current >= amarilla:
        status = (
            f"🟡 YELLOW ALERT — {current} cm exceeds yellow threshold of {amarilla} cm"
        )
    else:
        status = f"✅ NORMAL — {current} cm is below all alert thresholds (yellow: {amarilla} cm)"

    return (
        f"Station {station_codigo} live reading at {fecha}:\n"
        f"Current level: {current} cm\n"
        f"Status: {status}\n"
        f"Thresholds — yellow: {amarilla} cm | orange: {naranja} cm | red: {roja} cm"
    )


@tool
def check_all_stations_status() -> str:
    """Check live status of ALL river level stations and flag any that are
    above alert thresholds. Use this for broad questions like 'which rivers
    are in alert?' or 'is there a flood risk anywhere in Antioquia?'"""
    try:
        response = requests.get(f"{BASE_URL}/estaciones/", timeout=30)
        response.raise_for_status()
        stations = [
            s
            for s in response.json().get("values", [])
            if s.get("codigo", "").startswith("sn_") and s.get("activo")
        ]
    except Exception as e:
        return f"Could not fetch station list: {e}"

    alerts = {"red": [], "orange": [], "yellow": [], "normal": []}

    for station in stations:
        codigo = station["codigo"]
        ubicacion = station.get("ubicacion", codigo)
        latest = fetch_latest_level(codigo)
        if not latest:
            continue
        thresholds = fetch_level_thresholds(codigo)
        if not thresholds:
            continue

        current = float(latest["nivel"])
        if current >= thresholds["alerta_roja"]:
            alerts["red"].append(
                f"{codigo} at {ubicacion}: {current} cm (red threshold: {thresholds['alerta_roja']} cm)"
            )
        elif current >= thresholds["alerta_naranja"]:
            alerts["orange"].append(
                f"{codigo} at {ubicacion}: {current} cm (orange threshold: {thresholds['alerta_naranja']} cm)"
            )
        elif current >= thresholds["alerta_amarilla"]:
            alerts["yellow"].append(
                f"{codigo} at {ubicacion}: {current} cm (yellow threshold: {thresholds['alerta_amarilla']} cm)"
            )
        else:
            alerts["normal"].append(codigo)

    lines = [
        f"🌊 Live river status across Antioquia ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})\n"
    ]
    if alerts["red"]:
        lines.append(f"🔴 RED ALERT ({len(alerts['red'])} stations):")
        lines.extend(f"  • {s}" for s in alerts["red"])
    if alerts["orange"]:
        lines.append(f"\n🟠 ORANGE ALERT ({len(alerts['orange'])} stations):")
        lines.extend(f"  • {s}" for s in alerts["orange"])
    if alerts["yellow"]:
        lines.append(f"\n🟡 YELLOW ALERT ({len(alerts['yellow'])} stations):")
        lines.extend(f"  • {s}" for s in alerts["yellow"])
    lines.append(f"\n✅ Normal: {len(alerts['normal'])} stations below all thresholds")

    return "\n".join(lines)


@tool
def get_radar_image() -> str:
    """Get the current radar image URL showing where it is raining right now
    in Antioquia. Returns the URL and map bounds for overlay. Use this when
    asked about current rainfall location or rain patterns."""
    url = get_radar_url()
    return (
        f"Current radar image URL: {url}\n"
        f"Map bounds for overlay: SW corner {RADAR_BOUNDS[0]}, NE corner {RADAR_BOUNDS[1]}\n"
        f"This image shows real-time radar reflectivity — brighter areas indicate "
        f"active precipitation over Antioquia. Fetched at: "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )


# ── Tools registry ────────────────────────────────────────────────────────────

TOOLS = [
    search_knowledge_base,
    check_live_river_level,
    check_all_stations_status,
    get_radar_image,
]
TOOLS_MAP = {t.name: t for t in TOOLS}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a flood and climate risk advisor for Antioquia, Colombia.
You have access to a real-time sensor network monitoring river levels and precipitation
across the department, backed by historical data going back 30 days.

You have 4 tools:
- search_knowledge_base: for historical patterns, thresholds, station info
- check_live_river_level: for current level of a specific river station
- check_all_stations_status: for a broad overview of alert status across all rivers
- get_radar_image: for current rainfall location via radar

Guidelines:
- Always compare live readings against thresholds to assess risk
- Be specific — name stations, locations, and exact values
- Use clear alert language: normal / yellow / orange / red
- When relevant, mention the radar image for spatial rainfall context
- Answer in the same language the user asks (Spanish or English)
- Be concise but precise — this is an operational tool, not a chatbot
"""

# ── Agent loop ────────────────────────────────────────────────────────────────


def build_llm_with_tools():
    llm = get_llm()
    return llm.bind_tools(TOOLS)


def ask(question: str) -> dict:
    llm = build_llm_with_tools()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    for _ in range(5):  # max 5 iterations
        response = llm.invoke(messages)
        messages.append(response)

        # No tool calls — we have the final answer
        if not response.tool_calls:
            break

        # Execute each tool call and feed results back
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"\n→ Calling tool: {tool_name}({tool_args})")

            tool_fn = TOOLS_MAP.get(tool_name)
            if tool_fn:
                try:
                    result = tool_fn.invoke(tool_args)
                except Exception:
                    result = tool_fn.invoke({})
            else:
                result = f"Tool {tool_name} not found."

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

    return {
        "question": question,
        "answer": response.content,
        "radar_url": get_radar_url(),
        "radar_bounds": RADAR_BOUNDS,
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    questions = [
        "Is there any river in red or orange alert right now?",
        "What are the alert thresholds for station sn_1030?",
        "Where is it raining right now?",
    ]
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("=" * 60)
        result = ask(q)
        print(f"\nA: {result['answer']}")
