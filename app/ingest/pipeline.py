import os
from datetime import datetime, timezone
from typing import Annotated
from zenml import step, pipeline
from zenml.logger import get_logger

from app.ingest.fetch_data import (
    fetch_all_stations,
    classify_stations,
    fetch_level_thresholds,
    fetch_all_precip_thresholds,
    fetch_recent_levels,
    fetch_recent_precipitation,
)
from app.ingest.embed_and_store import (
    ensure_collection,
    embed,
    upsert_points,
    station_to_text,
    level_threshold_to_text,
    precip_threshold_to_text,
    level_readings_to_daily_summary,
    precip_readings_to_daily_summary,
    COLLECTION_NAME,
)
from qdrant_client.models import PointStruct
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = get_logger(__name__)

# ── Step 1: Fetch stations ────────────────────────────────────────────────────

@step
def fetch_stations_step() -> Annotated[list, "stations"]:
    """Fetch and classify all active stations from SIGRAN."""
    logger.info("Fetching stations from SIGRAN...")
    stations = fetch_all_stations()
    logger.info(f"Found {len(stations)} relevant stations")
    return stations

# ── Step 2: Build text chunks ─────────────────────────────────────────────────

@step
def build_chunks_step(
    stations: list,
) -> Annotated[list, "chunks"]:
    """Fetch sensor data and convert to embeddable text chunks."""
    logger.info("Building text chunks from sensor data...")

    classified = classify_stations(stations)
    station_map = {s["codigo"]: s for s in stations}

    precip_thresholds = fetch_all_precip_thresholds()
    precip_threshold_map = {t["estacion"]: t for t in precip_thresholds}

    all_chunks = []

    # Station descriptions
    for stype, station_list in classified.items():
        for s in station_list:
            text = station_to_text(s, stype)
            all_chunks.append({
                "text": text,
                "metadata": {
                    "type": "station_info",
                    "station_type": stype,
                    "codigo": s["codigo"],
                }
            })

    logger.info(f"  {len(all_chunks)} station description chunks")

    # Level stations — thresholds + history (parallel)
    def process_level(station):
        codigo = station["codigo"]
        chunks = []
        thresholds = fetch_level_thresholds(codigo)
        if not thresholds:
            return chunks
        text = level_threshold_to_text(thresholds, station)
        chunks.append({"text": text, "metadata": {"type": "level_threshold", "codigo": codigo}})
        readings = fetch_recent_levels(codigo, days=7)
        for summary in level_readings_to_daily_summary(codigo, station, readings, thresholds):
            chunks.append({"text": summary, "metadata": {"type": "level_history", "codigo": codigo}})
        return chunks

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_level, s): s for s in classified["level"]}
        for future in as_completed(futures):
            try:
                all_chunks.extend(future.result())
            except Exception as e:
                logger.warning(f"Level station error: {e}")

    logger.info(f"  Level chunks total so far: {len(all_chunks)}")

    # Precip stations — thresholds + history (parallel)
    def process_precip(station):
        codigo = station["codigo"]
        chunks = []
        threshold = precip_threshold_map.get(codigo)
        if threshold:
            text = precip_threshold_to_text(threshold)
            chunks.append({"text": text, "metadata": {"type": "precip_threshold", "codigo": codigo}})
        readings = fetch_recent_precipitation(codigo, days=7)
        for summary in precip_readings_to_daily_summary(codigo, station, readings, threshold):
            chunks.append({"text": summary, "metadata": {"type": "precip_history", "codigo": codigo}})
        return chunks

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_precip, s): s for s in classified["precip"]}
        for future in as_completed(futures):
            try:
                all_chunks.extend(future.result())
            except Exception as e:
                logger.warning(f"Precip station error: {e}")

    logger.info(f"Total chunks built: {len(all_chunks)}")
    return all_chunks

# ── Step 3: Embed and store ───────────────────────────────────────────────────

@step
def store_chunks_step(
    chunks: list,
) -> Annotated[int, "chunks_stored"]:
    """Embed all chunks and upsert into Qdrant."""
    logger.info(f"Embedding and storing {len(chunks)} chunks in Qdrant...")
    ensure_collection()

    points = []
    for i, chunk in enumerate(chunks):
        points.append(PointStruct(
            id=i + 1,
            vector=embed(chunk["text"]),
            payload={"text": chunk["text"], **chunk["metadata"]}
        ))

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        upsert_points(batch)
        logger.info(f"  Uploaded {min(i + batch_size, len(points))}/{len(points)}")

    logger.info(f"✅ Stored {len(points)} chunks in Qdrant")
    return len(points)

# ── Pipeline ──────────────────────────────────────────────────────────────────

@pipeline(name="antioquia_risk_ingestion")
def ingestion_pipeline():
    """
    Full ingestion pipeline:
    1. Fetch stations from SIGRAN
    2. Build text chunks from sensor data
    3. Embed and store in Qdrant
    """
    stations = fetch_stations_step()
    chunks = build_chunks_step(stations)
    store_chunks_step(chunks)

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"Starting ingestion pipeline at {datetime.now(timezone.utc)}")
    ingestion_pipeline()