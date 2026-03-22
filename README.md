# 🌊 Antioquia Flood Risk Advisor

An AI agent that monitors real-time river levels and precipitation across Antioquia, Colombia — combining a live sensor network with a vector knowledge base to answer natural language questions about flood risk.

> Built as a portfolio project to demonstrate AI engineering skills: RAG pipelines, agentic tool calling, real-time data integration, and containerized deployment.

---

## What it does

Ask it questions in plain English or Spanish:

- *"Is there any river in red alert right now?"*
- *"What are the thresholds for station sn_1030?"*
- *"Where is it raining right now?"*
- *"Which zones in Antioquia had the most rainfall in the last 30 days?"*

The agent reasons across two data sources — a vector knowledge base of historical sensor data and live API calls to real monitoring stations — to give precise, actionable answers.

---

## Architecture

```
Real sensor network (SIGRAN Antioquia)
         │
         ├── Station locations + metadata
         ├── River level readings (cm) ──────────────────────┐
         ├── Precipitation readings (mm)                     │
         ├── River level thresholds (yellow/orange/red)      │  Ingestion
         └── Precipitation thresholds                        │  pipeline
                                                             │
                                              ┌──────────────▼──────────────┐
                                              │   sentence-transformers      │
                                              │   all-MiniLM-L6-v2          │
                                              │   (local embeddings)         │
                                              └──────────────┬──────────────┘
                                                             │
                                              ┌──────────────▼──────────────┐
                                              │        Qdrant               │
                                              │   Vector Knowledge Base     │
                                              │   656 chunks / 30 days      │
                                              └──────────────┬──────────────┘
                                                             │
                          ┌──────────────────────────────────▼──────────────────────────────────┐
                          │                        LangChain Agent                               │
                          │                        (Llama 3.2 via Ollama)                        │
                          │                                                                      │
                          │   Tool 1: search_knowledge_base    → RAG retrieval from Qdrant       │
                          │   Tool 2: check_live_river_level   → live API call per station       │
                          │   Tool 3: check_all_stations_status → scan all 49 level stations     │
                          │   Tool 4: get_radar_image          → real-time rainfall radar        │
                          └──────────────────────────────────┬──────────────────────────────────┘
                                                             │
                                              ┌──────────────▼──────────────┐
                                              │        FastAPI              │
                                              │     POST /ask               │
                                              │     GET  /radar             │
                                              │     GET  /health            │
                                              └─────────────────────────────┘
```

---

## Data sources

| Source | Type | Description |
|--------|------|-------------|
| [SIGRAN Antioquia](https://sigran.antioquia.gov.co) | Live + Historic | 116 monitoring stations across Antioquia — river levels (cm), precipitation (mm), meteorological variables |
| Precipitation Thresholds API | Static | Yellow/orange/red alert limits with duration windows for 61 stations |
| [SIATA Radar](https://geoportal.siata.gov.co) | Real-time | Radar reflectivity image showing active precipitation over Antioquia |

**Station breakdown:**
- 49 river level stations (`sn_*`) — each with yellow/orange/red thresholds
- 50 precipitation stations (`sp_*`)
- 17 meteorological stations (`sm_*`)

---

## Agent tools

The agent decides which tools to call based on the question — this is real agentic behavior, not a fixed pipeline.

| Tool | When the agent uses it |
|------|----------------------|
| `search_knowledge_base` | Historical patterns, threshold context, station info |
| `check_live_river_level` | Current reading for a specific station vs its thresholds |
| `check_all_stations_status` | Broad scan — which rivers are in alert right now |
| `get_radar_image` | Where it is raining right now |

---

## Tech stack

| Layer | Technology | Why |
|-------|-----------|-----|
| LLM | Llama 3.2 via Ollama | Runs locally, no API cost, strong tool calling |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, fast — no external API needed |
| Vector DB | Qdrant | Purpose-built for vector search, easy Docker deployment |
| Agent framework | LangChain | Tool binding and message loop |
| API | FastAPI | Lightweight, async, auto-generates docs |
| Ingestion | Python + ThreadPoolExecutor | Parallel fetching across 99 stations |
| Deployment | Docker Compose | Single command to run everything |

---

## Running locally

### Prerequisites
- Docker Desktop
- Ollama installed with `llama3.2` pulled: `ollama pull llama3.2`
- Python 3.12+

### 1. Clone and set up environment

```bash
git clone https://github.com/garzonsergio/risk-monitoring
cd risk-monitoring
cp .env.example .env
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

### 3. Run the ingestion pipeline

Fetches 30 days of sensor data, generates embeddings, and stores 656 chunks in Qdrant:

```bash
python -m app.ingest.embed_and_store
```

Takes ~2-3 minutes with parallel fetching across 99 stations.

### 4. Start the API

```bash
uvicorn app.api.main:app --reload --port 8000
```

### 5. Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Is there any river in red or orange alert right now?"}'
```

Or open the interactive docs at **http://localhost:8000/docs**

---

## Example responses

**Q: Is there any river in red or orange alert right now?**
```json
{
  "question": "Is there any river in red or orange alert right now?",
  "answer": "There are currently two rivers in red alert:\n\n• sn_1022 at Río Tamar - Vereda Tamar: 299 cm (red threshold: 257 cm)\n• sn_1041 at Quebrada Juan García - PCH - Puente Verde: 764 cm (red threshold: 296 cm)\n\nOne river is in yellow alert:\n• sn_1007 at Quebrada La Oca - Villa Mena: 265 cm (yellow threshold: 260 cm)",
  "radar_url": "https://geoportal.siata.gov.co/fastgeoapi/geodata/radar/3/reflectividad?1774137931",
  "radar_bounds": [[5.1, -76.6], [7.3, -74.3]]
}
```

---

## Project structure

```
risk-monitoring/
├── app/
│   ├── ingest/
│   │   ├── fetch_data.py          # API clients for all SIGRAN endpoints
│   │   └── embed_and_store.py     # Ingestion pipeline with parallel fetching
│   ├── agent/
│   │   └── agent.py               # LangChain agent + 4 tools
│   └── api/
│       └── main.py                # FastAPI endpoints
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## What I'd add next

- **Scheduled re-ingestion** — run the ingestion pipeline every 6 hours via a cron job or ZenML pipeline to keep historical data fresh
- **MCP protocol** — expose the agent tools over MCP so any MCP-compatible client can connect
- **Multi-location queries** — add municipality and region context to enable questions like "how is the Urabá region doing?"
- **Alert notifications** — webhook or email when a station crosses into red alert
- **Map visualization** — overlay sensor readings and radar on a Leaflet.js map using the coordinates and radar bounds already in the API response

---

## Author

**Sergio Camilo Garzón**
[linkedin.com/in/gscode](https://linkedin.com/in/gscode) · [github.com/garzonsergio](https://github.com/garzonsergio)