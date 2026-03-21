import os
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from app.ingest.fetch_data import (
    fetch_all_stations,
    classify_stations,
    fetch_level_thresholds,
    fetch_all_precip_thresholds,
    fetch_recent_levels,
    fetch_recent_precipitation,
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "antioquia_risk"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_WORKERS = 10  # concurrent API calls

client = QdrantClient(url=QDRANT_URL)
model = SentenceTransformer(EMBEDDING_MODEL)

# ── Helpers ───────────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    return model.encode(text).tolist()

def upsert_points(points: list[PointStruct]):
    client.upsert(collection_name=COLLECTION_NAME, points=points)

def ensure_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"✓ Created Qdrant collection '{COLLECTION_NAME}'")
    else:
        print(f"✓ Collection '{COLLECTION_NAME}' already exists")

def alert_level_label(current: float, amarilla: float, naranja: float, roja: float) -> str:
    if current >= roja:
        return "RED ALERT — above red threshold"
    elif current >= naranja:
        return "ORANGE ALERT — above orange threshold"
    elif current >= amarilla:
        return "YELLOW ALERT — above yellow threshold"
    else:
        return "normal — below all alert thresholds"

# ── Text chunk generators ─────────────────────────────────────────────────────

def station_to_text(station: dict, station_type: str) -> str:
    codigo = station["codigo"]
    nombre = station.get("descripcion") or station.get("ubicacion") or codigo
    ubicacion = station.get("ubicacion", "unknown location")
    fuente = station.get("fuente") or "unknown water source"
    municipio = station.get("municipio", "unknown")
    lat = station.get("latitud", "?")
    lon = station.get("longitud", "?")

    if station_type == "level":
        return (
            f"River level monitoring station {codigo} is located at {ubicacion}, "
            f"municipality {municipio}, coordinates ({lat}, {lon}). "
            f"It monitors the water source: {fuente}. "
            f"Description: {nombre}."
        )
    elif station_type == "meteo":
        return (
            f"Meteorological station {codigo} is located at {ubicacion}, "
            f"municipality {municipio}, coordinates ({lat}, {lon}). "
            f"It measures precipitation (mm), temperature (°C), humidity (%), "
            f"radiation (W/m²), wind speed (m/s), and wind direction (°). "
            f"Description: {nombre}."
        )
    else:
        return (
            f"Precipitation monitoring station {codigo} is located at {ubicacion}, "
            f"municipality {municipio}, coordinates ({lat}, {lon}). "
            f"It measures rainfall in mm. Description: {nombre}."
        )

def level_threshold_to_text(threshold: dict, station: dict) -> str:
    codigo = threshold["station"]
    ubicacion = station.get("ubicacion", "unknown location")
    fuente = station.get("fuente") or "unknown water source"
    municipio = station.get("municipio", "unknown")
    return (
        f"River level alert thresholds for station {codigo} at {ubicacion}, "
        f"municipality {municipio}, monitoring {fuente}. "
        f"Yellow alert triggers at {threshold['alerta_amarilla']} cm. "
        f"Orange alert triggers at {threshold['alerta_naranja']} cm. "
        f"Red alert triggers at {threshold['alerta_roja']} cm. "
        f"All values are in centimeters (cm)."
    )

def precip_threshold_to_text(threshold: dict) -> str:
    estacion = threshold["estacion"]
    location = threshold.get("location", "unknown location")
    return (
        f"Precipitation alert thresholds for station {estacion} in {location}. "
        f"Yellow alert: {threshold['umbral_amarillo']} mm over {threshold['duracion_amarillo']} hours. "
        f"Orange alert: {threshold['umbral_naranja']} mm over {threshold['duracion_naranja']} hours. "
        f"Red alert: {threshold['umbral_rojo']} mm over {threshold['duracion_rojo']} hours."
    )

def level_readings_to_daily_summary(
    codigo: str, station: dict, readings: list[dict], thresholds: dict
) -> list[str]:
    if not readings:
        return []
    ubicacion = station.get("ubicacion", "unknown location")
    fuente = station.get("fuente") or "unknown water source"
    by_day: dict[str, list[float]] = {}
    for r in readings:
        day = r["fecha"][:10]
        try:
            by_day.setdefault(day, []).append(float(r["nivel"]))
        except (ValueError, KeyError):
            continue
    summaries = []
    for day, levels in sorted(by_day.items()):
        avg = round(sum(levels) / len(levels), 1)
        max_level = round(max(levels), 1)
        min_level = round(min(levels), 1)
        status = alert_level_label(
            max_level,
            thresholds["alerta_amarilla"],
            thresholds["alerta_naranja"],
            thresholds["alerta_roja"],
        )
        summaries.append(
            f"On {day}, river level station {codigo} at {ubicacion} "
            f"monitoring {fuente} recorded: "
            f"average {avg} cm, maximum {max_level} cm, minimum {min_level} cm. "
            f"Alert status based on daily maximum: {status}. "
            f"Thresholds — yellow: {thresholds['alerta_amarilla']} cm, "
            f"orange: {thresholds['alerta_naranja']} cm, "
            f"red: {thresholds['alerta_roja']} cm."
        )
    return summaries

def precip_readings_to_daily_summary(
    codigo: str, station: dict, readings: list[dict], threshold: dict | None
) -> list[str]:
    if not readings:
        return []
    ubicacion = station.get("ubicacion", "unknown location")
    by_day: dict[str, list[float]] = {}
    for r in readings:
        day = r["fecha"][:10]
        try:
            by_day.setdefault(day, []).append(float(r["muestra"]))
        except (ValueError, KeyError):
            continue
    summaries = []
    for day, values in sorted(by_day.items()):
        total = round(sum(values), 1)
        max_reading = round(max(values), 1)
        threshold_info = ""
        if threshold:
            threshold_info = (
                f" Thresholds — yellow: {threshold['umbral_amarillo']} mm "
                f"over {threshold['duracion_amarillo']} hours, "
                f"orange: {threshold['umbral_naranja']} mm over {threshold['duracion_naranja']} hours, "
                f"red: {threshold['umbral_rojo']} mm over {threshold['duracion_rojo']} hours."
            )
        summaries.append(
            f"On {day}, precipitation station {codigo} at {ubicacion} "
            f"recorded a total of {total} mm of rainfall, "
            f"with a maximum single reading of {max_reading} mm.{threshold_info}"
        )
    return summaries

# ── Per-station workers (run in threads) ──────────────────────────────────────

def process_level_station(station: dict) -> list[PointStruct]:
    """Fetch + embed everything for one level station. Runs in a thread."""
    codigo = station["codigo"]
    chunks = []

    thresholds = fetch_level_thresholds(codigo)
    if not thresholds:
        return []

    # Threshold chunk
    text = level_threshold_to_text(thresholds, station)
    chunks.append({"text": text, "metadata": {"type": "level_threshold", "codigo": codigo}})

    # Historical summaries
    readings = fetch_recent_levels(codigo, days=30)
    for summary in level_readings_to_daily_summary(codigo, station, readings, thresholds):
        chunks.append({"text": summary, "metadata": {"type": "level_history", "codigo": codigo}})

    print(f"  ✓ {codigo}: {len(readings)} readings → {len(chunks) - 1} daily summaries")
    return chunks

def process_precip_station(args: tuple) -> list[dict]:
    """Fetch + embed everything for one precip station. Runs in a thread."""
    station, threshold = args
    codigo = station["codigo"]
    chunks = []

    if threshold:
        text = precip_threshold_to_text(threshold)
        chunks.append({"text": text, "metadata": {"type": "precip_threshold", "codigo": codigo}})

    readings = fetch_recent_precipitation(codigo, days=30)
    for summary in precip_readings_to_daily_summary(codigo, station, readings, threshold):
        chunks.append({"text": summary, "metadata": {"type": "precip_history", "codigo": codigo}})

    print(f"  ✓ {codigo}: {len(readings)} readings → {len(chunks)} chunks")
    return chunks

# ── Main ingestion pipeline ───────────────────────────────────────────────────

def run_ingestion():
    start = time.time()
    print("\n=== Starting ingestion pipeline ===\n")
    ensure_collection()

    stations = fetch_all_stations()
    classified = classify_stations(stations)
    station_map = {s["codigo"]: s for s in stations}

    precip_thresholds = fetch_all_precip_thresholds()
    precip_threshold_map = {t["estacion"]: t for t in precip_thresholds}

    all_chunks: list[dict] = []

    # 1. Station descriptions (fast, no threading needed)
    print("\n→ Ingesting station descriptions...")
    for stype, station_list in classified.items():
        for s in station_list:
            text = station_to_text(s, stype)
            all_chunks.append({"text": text, "metadata": {"type": "station_info", "station_type": stype, "codigo": s["codigo"]}})
    print(f"  {len(all_chunks)} station description chunks ready")

    # 2. Level stations — multithreaded
    print(f"\n→ Fetching level stations with {MAX_WORKERS} threads...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_level_station, s): s for s in classified["level"]}
        for future in as_completed(futures):
            try:
                all_chunks.extend(future.result())
            except Exception as e:
                print(f"  ✗ Error: {e}")

    # 3. Precip stations — multithreaded
    print(f"\n→ Fetching precip stations with {MAX_WORKERS} threads...")
    precip_args = [
        (s, precip_threshold_map.get(s["codigo"]))
        for s in classified["precip"]
    ]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_precip_station, args): args for args in precip_args}
        for future in as_completed(futures):
            try:
                all_chunks.extend(future.result())
            except Exception as e:
                print(f"  ✗ Error: {e}")

    # 4. Embed + upload in batches
    print(f"\n→ Embedding and uploading {len(all_chunks)} chunks to Qdrant...")
    points = []
    for i, chunk in enumerate(all_chunks):
        points.append(PointStruct(
            id=i + 1,
            vector=embed(chunk["text"]),
            payload={"text": chunk["text"], **chunk["metadata"]}
        ))

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        upsert_points(batch)
        print(f"  Uploaded {min(i + batch_size, len(points))}/{len(points)}")

    elapsed = round(time.time() - start, 1)
    print(f"\n✅ Ingestion complete — {len(points)} chunks stored in Qdrant in {elapsed}s")

if __name__ == "__main__":
    run_ingestion()