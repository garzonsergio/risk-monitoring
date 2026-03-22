# 🌊 Antioquia Flood Risk Advisor

An AI agent that monitors real-time river levels and precipitation across Antioquia, Colombia — combining a live sensor network with a vector knowledge base to answer natural language questions about flood risk.

> Built as a portfolio project to demonstrate AI engineering skills: RAG pipelines, agentic tool calling, real-time data integration, and containerized deployment.

---

## What it does

Ask it questions in plain English or Spanish:

- _"Is there any river in red alert right now?"_
- _"What are the thresholds for station sn_1030?"_
- _"Where is it raining right now, and is it risky?"_
- _"Which zones in Antioquia had the most rainfall in the last 7 days?"_
- _"When did it rain the most in the last 24 hours?"_

The agent reasons across two data sources — a vector knowledge base of historical sensor data and live API calls to real monitoring stations — to give precise, actionable answers.

---

## Architecture

```
Real sensor network (SIGRAN Antioquia)
         │
         ├── Station locations + metadata
         ├── River level readings (cm) ──────────────────────┐
         ├── Precipitation readings (mm)                     │
         ├── Meteorological readings (lluvia mm, temp, etc.) │  Ingestion
         ├── River level thresholds (yellow/orange/red)      │  pipeline
         └── Precipitation thresholds (duration-weighted)    │
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
                                              │   7 days daily summaries    │
                                              │   + last 24h hourly chunks  │
                                              └──────────────┬──────────────┘
                                                             │
                          ┌──────────────────────────────────▼──────────────────────────────────┐
                          │                        LangChain Agent                               │
                          │                        (Llama 3.1 via Ollama)                        │
                          │                                                                      │
                          │   Tool 1: search_knowledge_base      → RAG retrieval from Qdrant     │
                          │   Tool 2: check_live_river_level     → live level for one station    │
                          │   Tool 3: check_all_stations_status  → scan all 49 level stations    │
                          │   Tool 4: check_active_rainfall      → live rain risk across all     │
                          │                                         sp_ and sm_ stations          │
                          │   Tool 5: check_precipitation_by_date → historical rain by date      │
                          │   Tool 6: check_river_levels_by_date  → historical levels by date    │
                          └──────────────────────────────────┬──────────────────────────────────┘
                                                             │
                                              ┌──────────────▼──────────────┐
                                              │        FastAPI              │
                                              │     POST /ask               │
                                              │     GET  /health            │
                                              └─────────────────────────────┘
```

---

## Data sources

| Source                                              | Type            | Description                                                                                                |
| --------------------------------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------- |
| [SIGRAN Antioquia](https://sigran.antioquia.gov.co) | Live + Historic | 116 monitoring stations across Antioquia — river levels (cm), precipitation (mm), meteorological variables |
| Precipitation Thresholds API                        | Static          | Yellow/orange/red alert limits with duration windows for each station                                      |

**Station breakdown:**

- 49 river level stations (`sn_*`) — each with yellow/orange/red thresholds
- 50 precipitation stations (`sp_*`)
- 17 meteorological stations (`sm_*`)

---

## Agent tools

The agent decides which tools to call based on the question — real agentic behavior, not a fixed pipeline.

| Tool                          | When the agent uses it                                                                  |
| ----------------------------- | --------------------------------------------------------------------------------------- |
| `search_knowledge_base`       | Historical patterns (7 days), hourly rainfall history, threshold context, station info  |
| `check_live_river_level`      | Current reading for a specific station vs its thresholds                                |
| `check_all_levels`            | Broad scan — which rivers are in alert right now                                        |
| `check_active_rainfall`       | Live rain risk scan across all sp* and sm* stations with duration-weighted alert levels |
| `check_precipitation_by_date` | Rainfall totals and hourly breakdown for a specific past date                           |
| `check_river_levels_by_date`  | Peak river levels and alert status for a specific past date                             |

### Rainfall risk calculation

`check_active_rainfall` evaluates each station against its thresholds using the correct duration window — not just the latest reading. For example, if the orange threshold is 48 mm over 16 hours, it sums all readings from the last 16 hours and compares the total to 48 mm. This matches the actual alert methodology used by SIATA.

---

## Tech stack

| Layer           | Technology                               | Why                                                     |
| --------------- | ---------------------------------------- | ------------------------------------------------------- |
| LLM             | Llama 3.1 via Ollama                     | Runs locally, no API cost, strong tool calling          |
| Embeddings      | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, fast — no external API needed              |
| Vector DB       | Qdrant                                   | Purpose-built for vector search, easy Docker deployment |
| Agent framework | LangChain                                | Tool binding and message loop                           |
| API             | FastAPI                                  | Lightweight, async, auto-generates docs                 |
| Ingestion       | Python + ThreadPoolExecutor              | Parallel fetching across 116 stations                   |
| Deployment      | Docker Compose                           | Single command to run everything                        |

---

## Running the project

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Qdrant and the app container)
- [Ollama](https://ollama.com) installed locally with the model pulled:
  ```bash
  ollama pull llama3.1:8b
  ```
  Ollama must be running on your machine at `http://localhost:11434`.

---

### Option A — Docker Compose (recommended)

Runs Qdrant and the API together. Ollama still runs on your host machine.

**1. Clone and configure**

```bash
git clone https://github.com/garzonsergio/risk-monitoring
cd risk-monitoring
cp .env.example .env
```

**2. Start all services**

```bash
docker compose up --build
```

This starts:

- `qdrant` on port `6333`
- `app` (FastAPI) on port `8005`

**3. Run the ingestion pipeline**

In a separate terminal, populate Qdrant with 7 days of sensor data:

```bash
docker compose run --rm app python -m app.ingest.embed_and_store
```

Takes ~2–3 minutes with parallel fetching across 116 stations.

**4. Ask a question**

```bash
curl -X POST http://localhost:8005/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Is there any river in red or orange alert right now?"}'
```

Or open the interactive docs at **http://localhost:8005/docs**

---

### Option B — Local development (venv)

**1. Clone and install dependencies**

```bash
git clone https://github.com/garzonsergio/risk-monitoring
cd risk-monitoring
cp .env.example .env
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Start Qdrant**

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

**3. Run the ingestion pipeline**

```bash
python -m app.ingest.embed_and_store
```

**4. Start the API**

```bash
uvicorn app.api.main:app --reload --port 8005
```

**5. Ask a question**

```bash
curl -X POST http://localhost:8005/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Where is it raining right now and is it risky?"}'}
```

---

## Example responses

**Q: Is there any river in red or orange alert right now?**

```json
{
  "question": "Is there any river in red or orange alert right now?",
  "answer": "There are currently two rivers in red alert:\n\n• sn_1022 at Río Tamar - Vereda Tamar: 299 cm (red threshold: 257 cm)\n• sn_1041 at Quebrada Juan García - PCH - Puente Verde: 764 cm (red threshold: 296 cm)\n\nOne river is in yellow alert:\n• sn_1007 at Quebrada La Oca - Villa Mena: 265 cm (yellow threshold: 260 cm)"
}
```

**Q: Where is it raining right now and is it risky?**

```json
{
  "question": "Where is it raining right now and is it risky?",
  "answer": "🌧 Active rainfall detected at 3 stations:\n\n• sp_2041 (precip) — 🔴 red | 71.2 mm accumulated | Vereda La Danta, Sonsón\n• sm_501 (meteo) — 🟡 yellow | 36.8 mm accumulated | Turbo\n• sp_1093 (precip) — ✅ normal (below thresholds) | 4.1 mm | Rionegro"
}
```

---

## Project structure

```
risk-monitoring/
├── app/
│   ├── ingest/
│   │   ├── fetch_data.py          # API clients for all SIGRAN endpoints
│   │   └── embed_and_store.py     # Ingestion pipeline — 7-day daily + 24h hourly chunks
│   ├── agent/
│       └── agent.py               # LangChain agent + 6 tools
│   └── api/
│       └── main.py                # FastAPI endpoints
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## What I'd add next

- **Scheduled re-ingestion** — run the ingestion pipeline every 6 hours via APScheduler to keep historical data fresh
- **Radar overlay** — integrate the [SIATA radar API](https://geoportal.siata.gov.co) as an agent tool returning a real-time reflectivity image URL with map bounds, enabling spatial rainfall context on a Leaflet.js map
- **MCP protocol** — expose the agent tools over MCP so any MCP-compatible client (Claude Desktop, etc.) can connect
- **Multi-location queries** — add municipality and region context to enable questions like "how is the Urabá region doing?"
- **Alert notifications** — webhook or email when a station crosses into red alert
- **Map visualization** — overlay sensor readings and radar on a Leaflet.js map using the coordinates and radar bounds already in the API response

---

## Author

**Sergio Camilo Garzón**
[linkedin.com/in/gscode](https://linkedin.com/in/gscode) · [github.com/garzonsergio](https://github.com/garzonsergio)
