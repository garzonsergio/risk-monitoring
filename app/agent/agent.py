import os
import time
import traceback
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from app.ingest.fetch_data import (
    fetch_all_stations,
    classify_stations,
    fetch_latest_level,
    fetch_level_thresholds,
    fetch_latest_precipitation,
    fetch_latest_meteo,
    fetch_recent_precipitation,
    fetch_recent_meteo,
    fetch_all_precip_thresholds,
    fetch_location_name,
    fetch_precipitation_by_date,
    fetch_meteo_by_date,
    fetch_levels_by_date,
    BASE_URL,
    PRECIP_THRESHOLDS_URL,
)

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
COLLECTION_NAME = "antioquia_risk"

# ── Embedding wrapper ─────────────────────────────────────────────────────────


class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()


# ── LLM + Retriever ───────────────────────────────────────────────────────────


def get_llm():
    return ChatOllama(
        model="llama3.1:8b",
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        temperature=0,  # ← FIX 2: was 0.2 — 0 = deterministic, reduces hallucination
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
    """Search historical SUMMARIES, threshold values, and station metadata.
    Use this ONLY for: station locations, alert threshold values, or general
    context about the sensor network.
    Do NOT use this for specific rainfall or level questions — use
    check_precipitation_by_date or check_river_levels_by_date instead."""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return (
            "KNOWLEDGE BASE RETURNED NO RESULTS for this query. "
            "Do not invent data. Tell the user no information is available."
        )
    results = "\n\n---\n\n".join(d.page_content for d in docs)
    return (
        f"KNOWLEDGE BASE RESULTS ({len(docs)} chunks found — only use information "
        f"explicitly present below, do not infer or invent anything else):\n\n{results}"
    )


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
def check_all_levels() -> str:
    """Check live status of ALL river level stations and flag any that are
    above alert thresholds. Use this for broad questions like 'which rivers
    are in alert?' or 'is there a flood risk anywhere in Antioquia?'"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        all_stations = fetch_all_stations()
    except Exception as e:
        return f"Could not fetch station list: {e}"

    classified = classify_stations(all_stations)
    level_stations = classified["level"]

    def check_station(station):
        codigo = station["codigo"]
        ubicacion = (
            station.get("ubicacion")
            or station.get("descripcion")
            or station.get("nombre_web")
            or codigo
        )
        latest = fetch_latest_level(codigo)
        if not latest:
            return None
        thresholds = fetch_level_thresholds(codigo)
        if not thresholds:
            return None
        return {
            "codigo": codigo,
            "ubicacion": ubicacion,
            "current": float(latest["nivel"]),
            "thresholds": thresholds,
        }

    alerts = {"red": [], "orange": [], "yellow": [], "normal": []}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_station, s): s for s in level_stations}
        for future in as_completed(futures):
            try:
                result = future.result()
                if not result:
                    continue
                current = result["current"]
                t = result["thresholds"]
                entry = (
                    f"{result['codigo']} at {result['ubicacion']}: "
                    f"{current} cm (red threshold: {t['alerta_roja']} cm)"
                )
                if current >= t["alerta_roja"]:
                    alerts["red"].append(
                        f"{result['codigo']} at {result['ubicacion']}: "
                        f"{current} cm (red threshold: {t['alerta_roja']} cm)"
                    )
                elif current >= t["alerta_naranja"]:
                    alerts["orange"].append(
                        f"{result['codigo']} at {result['ubicacion']}: "
                        f"{current} cm (orange threshold: {t['alerta_naranja']} cm)"
                    )
                elif current >= t["alerta_amarilla"]:
                    alerts["yellow"].append(
                        f"{result['codigo']} at {result['ubicacion']}: "
                        f"{current} cm (yellow threshold: {t['alerta_amarilla']} cm)"
                    )
                else:
                    alerts["normal"].append(result["codigo"])
            except Exception:
                continue

    lines = [
        f"🌊 Live river status across Antioquia "
        f"({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})\n"
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
def check_active_rainfall() -> str:
    """Check which stations are currently reporting active rainfall across
    Antioquia, with alert levels calculated from duration-weighted thresholds.
    Scans all precipitation (sp_) and meteorological (sm_) stations live.
    Use this when asked where it is raining right now."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        all_stations = fetch_all_stations()
    except Exception as e:
        return f"Could not fetch station list: {e}"

    classified = classify_stations(all_stations)

    try:
        threshold_map = {t["estacion"]: t for t in fetch_all_precip_thresholds()}
    except Exception:
        threshold_map = {}

    def calc_alert(
        readings: list[dict], value_key: str, threshold: dict | None
    ) -> tuple[str, float]:
        """Returns (alert_label, accumulated_mm) for the highest triggered level."""
        now = datetime.now(timezone.utc)

        def rain_sum(hours: int) -> float:
            cutoff = (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
            return round(
                sum(
                    max(float(r.get(value_key, 0) or 0), 0)
                    for r in readings
                    if r["fecha"] >= cutoff
                ),
                1,
            )

        if threshold:
            for level, label in [
                ("rojo", "🔴 red"),
                ("naranja", "🟠 orange"),
                ("amarillo", "🟡 yellow"),
            ]:
                hours = int(threshold[f"duracion_{level}"])
                umbral = float(threshold[f"umbral_{level}"])
                total = rain_sum(hours)
                if total >= umbral:
                    return label, total

        last_hour = rain_sum(1)
        if last_hour > 0:
            return "✅ normal (below thresholds)", last_hour
        return "dry", 0.0

    def check_sp(station: dict) -> dict | None:
        codigo = station["codigo"]
        threshold = threshold_map.get(codigo)
        readings = fetch_recent_precipitation(codigo, days=1)
        alert, total = calc_alert(readings, "muestra", threshold)
        if alert == "dry":
            return None
        ubicacion = (
            station.get("ubicacion")
            or station.get("descripcion")
            or station.get("nombre_web")
            or codigo
        )
        return {
            "codigo": codigo,
            "location": ubicacion,
            "alert": alert,
            "total_mm": total,
            "type": "precipitation",
        }

    def check_sm(station: dict) -> dict | None:
        codigo = station["codigo"]
        threshold = threshold_map.get(codigo)
        readings = fetch_recent_meteo(codigo, hours=24)
        alert, total = calc_alert(readings, "lluvia", threshold)
        if alert == "dry":
            return None
        ubicacion = (
            station.get("ubicacion")
            or station.get("descripcion")
            or station.get("nombre_web")
            or codigo
        )
        return {
            "codigo": codigo,
            "location": ubicacion,
            "alert": alert,
            "total_mm": total,
            "type": "meteo",
        }

    active: list[dict] = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            *[executor.submit(check_sp, s) for s in classified["precip"]],
            *[executor.submit(check_sm, s) for s in classified["meteo"]],
        ]
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    active.append(result)
            except Exception:
                continue

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if not active:
        return f"🌤 No active rainfall detected across any station at {now_str}."

    level_order = {
        "🔴 red": 0,
        "🟠 orange": 1,
        "🟡 yellow": 2,
        "✅ normal (below thresholds)": 3,
    }
    active.sort(key=lambda x: (level_order.get(x["alert"], 4), -x["total_mm"]))

    lines = [
        f"🌧 Active rainfall at {now_str} — {len(active)} stations reporting rain:\n"
    ]
    for r in active:
        source = "precip" if r["type"] == "precipitation" else "meteo"
        lines.append(
            f"  • {r['codigo']} ({source}) — {r['alert']} | "
            f"{r['total_mm']} mm | {r['location']}"
        )
    return "\n".join(lines)


@tool
def check_precipitation_by_date(date_str: str) -> str:
    """USE THIS TOOL when asked about past or yesterday's rainfall, precipitation,
    or rain. This tool fetches REAL data from live sensors.
    date_str format: YYYY-MM-DD
    Example: '2026-03-21'
    ALWAYS use this tool for any question containing words like:
    'yesterday', 'ayer', 'last night', 'anoche', 'was it raining', 'llovió'"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        all_stations = fetch_all_stations()
    except Exception as e:
        return f"Could not fetch station list: {e}"

    classified = classify_stations(all_stations)

    def check_sp(station: dict) -> dict | None:
        codigo = station["codigo"]
        ubicacion = (
            station.get("ubicacion")
            or station.get("descripcion")
            or station.get("nombre_web")
            or codigo
        )
        readings = fetch_precipitation_by_date(codigo, date_str)
        if not readings:
            return None
        total = round(sum(float(r["muestra"]) for r in readings), 1)
        hours_with_rain = sorted(set(r["fecha"][11:13] for r in readings))
        return {
            "codigo": codigo,
            "ubicacion": ubicacion,
            "total_mm": total,
            "hours": hours_with_rain,
            "source": "precipitation",
        }

    def check_sm(station: dict) -> dict | None:
        codigo = station["codigo"]
        ubicacion = (
            station.get("ubicacion")
            or station.get("descripcion")
            or station.get("nombre_web")
            or codigo
        )
        readings = fetch_meteo_by_date(codigo, date_str)
        if not readings:
            return None
        total = round(sum(float(r["lluvia"]) for r in readings), 1)
        if total == 0:
            return None
        hours_with_rain = sorted(set(r["fecha"][11:13] for r in readings))
        return {
            "codigo": codigo,
            "ubicacion": ubicacion,
            "total_mm": total,
            "hours": hours_with_rain,
            "source": "meteo",
        }

    results = []
    all_tasks = [(check_sp, s) for s in classified["precip"]] + [
        (check_sm, s) for s in classified["meteo"]
    ]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fn, station): station for fn, station in all_tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception:
                continue

    if not results:
        return f"No rainfall recorded at any station on {date_str}."

    results.sort(key=lambda x: x["total_mm"], reverse=True)

    lines = [
        f"📊 Precipitation report for {date_str} — "
        f"{len(results)} stations with rainfall.\n"
        f"IMPORTANT: Report ALL of this data including hours and locations exactly as shown:\n"
    ]
    for r in results[:20]:
        hours_str = ", ".join(f"{h}:00" for h in r["hours"][:8])
        if len(r["hours"]) > 8:
            hours_str += f" (+{len(r['hours']) - 8} more hours)"
        source_label = "precip" if r["source"] == "precipitation" else "meteo"
        lines.append(
            f"  • {r['codigo']} ({source_label}) | {r['ubicacion']} | "
            f"{r['total_mm']} mm | hours: {hours_str}"
        )

    return "\n".join(lines)


@tool
def check_river_levels_by_date(date_str: str) -> str:
    """Check river levels across all stations on a specific past date,
    flagging any that exceeded alert thresholds. Use this when asked
    about past flood risk or high river levels on a specific day.
    date_str format: YYYY-MM-DD. Example: 2026-03-20"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

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

    def check_station(station):
        codigo = station["codigo"]
        ubicacion = station.get("ubicacion", codigo)
        readings = fetch_levels_by_date(codigo, date_str)
        if not readings:
            return None
        thresholds = fetch_level_thresholds(codigo)
        if not thresholds:
            return None

        levels = [float(r["nivel"]) for r in readings]
        max_level = round(max(levels), 1)
        avg_level = round(sum(levels) / len(levels), 1)
        peak_hour = max(readings, key=lambda r: float(r["nivel"]))["fecha"][11:16]

        return {
            "codigo": codigo,
            "ubicacion": ubicacion,
            "max_level": max_level,
            "avg_level": avg_level,
            "peak_hour": peak_hour,
            "thresholds": thresholds,
        }

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_station, s): s for s in stations}
        for future in as_completed(futures):
            try:
                result = future.result()
                if not result:
                    continue
                t = result["thresholds"]
                entry = (
                    f"  • {result['codigo']} at {result['ubicacion']}: "
                    f"max {result['max_level']} cm at {result['peak_hour']} "
                    f"(avg {result['avg_level']} cm)"
                )
                if result["max_level"] >= t["alerta_roja"]:
                    alerts["red"].append(entry)
                elif result["max_level"] >= t["alerta_naranja"]:
                    alerts["orange"].append(entry)
                elif result["max_level"] >= t["alerta_amarilla"]:
                    alerts["yellow"].append(entry)
                else:
                    alerts["normal"].append(result["codigo"])
            except Exception:
                continue

    lines = [f"🌊 River level report for {date_str}:\n"]
    if alerts["red"]:
        lines.append(f"🔴 RED ALERT ({len(alerts['red'])} stations):")
        lines.extend(alerts["red"])
    if alerts["orange"]:
        lines.append(f"\n🟠 ORANGE ALERT ({len(alerts['orange'])} stations):")
        lines.extend(alerts["orange"])
    if alerts["yellow"]:
        lines.append(f"\n🟡 YELLOW ALERT ({len(alerts['yellow'])} stations):")
        lines.extend(alerts["yellow"])
    lines.append(f"\n✅ Normal: {len(alerts['normal'])} stations below all thresholds")

    return "\n".join(lines)


# ── Tools registry ────────────────────────────────────────────────────────────

TOOLS = [
    search_knowledge_base,
    check_live_river_level,
    check_all_levels,
    check_active_rainfall,
    check_precipitation_by_date,
    check_river_levels_by_date,
]
TOOLS_MAP = {t.name: t for t in TOOLS}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a flood and climate risk advisor for Antioquia, Colombia.
You have access to a real-time sensor network monitoring river levels and precipitation
across the department, backed by historical data covering the last 7 days.

You have 5 tools:
- search_knowledge_base: for historical patterns (last 7 days daily + last 24h hourly),
  thresholds, station locations, and general context. Use this for questions about when
  and where it rained over the past days or hours.
- check_live_river_level: for the current level of a specific river station
- check_all_stations_status: for a broad overview of alert status across all rivers
- check_active_rainfall: for which stations are reporting rainfall right now (live scan
  of all sp_ precipitation and sm_ meteorological stations)
- check_precipitation_by_date: for rainfall data on a specific past date with hourly breakdown

Guidelines:
- ONLY report data that was explicitly returned by a tool. Never invent station names,
  station IDs, locations, readings, or times that were not in the tool output.
- If a tool returns no results or the data does not cover the requested time period,
  say clearly that the data is not available. Do not fill in gaps with assumptions.
- Station codes follow the format sn_XXXX (river level), sp_XXXX (precipitation),
  sm_XXX (meteorological). Never reference station names or IDs not in the tool output.
- Always compare live readings against thresholds to assess risk.
- Use clear alert language: normal / yellow / orange / red.
- For "where is it raining now" questions, use check_active_rainfall.
- For "when/where did it rain in the last 7 days" questions, use search_knowledge_base.
- Answer in the same language the user asks (Spanish or English).
- Be concise but precise — this is an operational tool, not a chatbot.

CRITICAL RULES — you must follow these without exception:
- NEVER invent station codes, readings, or values. Only report data returned by tools.
- If a tool returns no data for a question, say explicitly: "I don't have that data available."
- If asked about YESTERDAY or a specific date use check_precipitation_by_date or check_river_levels_by_date — NEVER search_knowledge_base for this.
- For ANY question with 'ayer', 'yesterday', 'llovió', 'was raining' → call check_precipitation_by_date first.
- If asked for hourly data, say: "My knowledge base stores daily summaries only."
- Never say "the search returned" and then invent results — only report actual tool output.
- Answer in the same language the user asks (Spanish or English).
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
                except Exception as e:
                    traceback.print_exc()
                    result = f"Tool {tool_name} raised an error: {e}"
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
