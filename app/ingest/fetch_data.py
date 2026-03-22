import requests
from datetime import datetime, timedelta, timezone
from typing import Optional

BASE_URL = "https://sigran.antioquia.gov.co/api/v1"
PRECIP_THRESHOLDS_URL = (
    "https://umbrales-precipitacion-api.scgarzonp.workers.dev/api/estaciones"
)
RADAR_URL = "https://geoportal.siata.gov.co/fastgeoapi/geodata/radar/3/reflectividad"

# ── Stations ─────────────────────────────────────────────────────────────────


def fetch_all_stations() -> list[dict]:
    """Fetch all stations and filter only sn_, sm_, sp_ ones."""
    response = requests.get(f"{BASE_URL}/estaciones/", timeout=30)
    response.raise_for_status()
    data = response.json()
    stations = data.get("values", [])
    relevant = [
        s
        for s in stations
        if s.get("codigo", "").startswith(("sn_", "sm_", "sp_"))
        and s.get("activo", False)
    ]
    print(f"✓ Fetched {len(relevant)} relevant stations (sn_, sm_, sp_)")
    return relevant


def classify_stations(stations: list[dict]) -> dict[str, list[dict]]:
    """Split stations by type based on codigo prefix."""
    level, meteo, precip = [], [], []
    for s in stations:
        codigo = s.get("codigo", "")
        if codigo.startswith("sn_"):
            level.append(s)
        elif codigo.startswith("sm_"):
            meteo.append(s)
        elif codigo.startswith("sp_"):
            precip.append(s)
    print(f"  → Level: {len(level)} | Meteo: {len(meteo)} | Precip: {len(precip)}")
    return {"level": level, "meteo": meteo, "precip": precip}


# ── Thresholds ────────────────────────────────────────────────────────────────


def fetch_level_thresholds(station_codigo: str) -> Optional[dict]:
    """Fetch river level thresholds for a station. Returns alert levels in real units."""
    try:
        url = f"{BASE_URL}/estaciones/{station_codigo}/seccion"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        offset = float(data.get("offset", 0))
        # Real alert level = offset - umbral
        return {
            "station": station_codigo,
            "offset": offset,
            "alerta_amarilla": round(offset - float(data["umbral_amarillo"]), 2),
            "alerta_naranja": round(offset - float(data["umbral_naranja"]), 2),
            "alerta_roja": round(offset - float(data["umbral_rojo"]), 2),
        }
    except Exception as e:
        return None


def fetch_all_precip_thresholds() -> list[dict]:
    """Fetch precipitation thresholds for all stations."""
    response = requests.get(PRECIP_THRESHOLDS_URL, timeout=30)
    response.raise_for_status()
    data = response.json()
    thresholds = data.get("data", [])
    print(f"✓ Fetched precipitation thresholds for {len(thresholds)} stations")
    return thresholds


# ── Time-series data ──────────────────────────────────────────────────────────


def fetch_recent_levels(station_codigo: str, days: int = 7) -> list[dict]:
    """Fetch recent river level readings — up to 7 days of data."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    readings = []
    # size=288 = one reading every 5 min × 24h — one page per day
    url = f"{BASE_URL}/estaciones/{station_codigo}/nivel/?calidad=1&size=288"
    pages_fetched = 0
    max_pages = 7  # 7 pages × 288 readings = ~7 days at 5-min intervals

    while url and pages_fetched < max_pages:
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            page_values = data.get("values", [])
            filtered = [v for v in page_values if v["fecha"] >= since]
            readings.extend(filtered)
            pages_fetched += 1
            if len(filtered) < len(page_values):
                break  # hit data older than our window
            url = data.get("next")
        except Exception:
            break
    return readings


def fetch_recent_precipitation(station_codigo: str, days: int = 7) -> list[dict]:
    """Fetch recent precipitation readings — up to 7 days of data."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    readings = []
    url = f"{BASE_URL}/estaciones/{station_codigo}/precipitacion/?calidad=1&size=288"
    pages_fetched = 0
    max_pages = 7

    while url and pages_fetched < max_pages:
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            page_values = data.get("values", [])
            filtered = [v for v in page_values if v["fecha"] >= since]
            readings.extend(filtered)
            pages_fetched += 1
            if len(filtered) < len(page_values):
                break
            url = data.get("next")
        except Exception:
            break
    return readings


def fetch_latest_level(station_codigo: str) -> Optional[dict]:
    """Fetch only the most recent level reading for a station."""
    try:
        url = f"{BASE_URL}/estaciones/{station_codigo}/nivel/?calidad=1&size=1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        values = response.json().get("values", [])
        return values[0] if values else None
    except Exception:
        return None


def fetch_latest_precipitation(station_codigo: str) -> Optional[dict]:
    """Fetch only the most recent precipitation reading for a station."""
    try:
        url = f"{BASE_URL}/estaciones/{station_codigo}/precipitacion/?calidad=1&size=1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        values = response.json().get("values", [])
        return values[0] if values else None
    except Exception:
        return None


def fetch_latest_meteo(station_codigo: str) -> Optional[dict]:
    """Fetch only the most recent meteorological reading for a station."""
    try:
        url = f"{BASE_URL}/estaciones/{station_codigo}/meteorologia/?calidad_lluvia=1&size=1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        values = response.json().get("values", [])
        return values[0] if values else None
    except Exception:
        return None


def fetch_recent_meteo(station_codigo: str, hours: int = 24) -> list[dict]:
    """Fetch recent meteorological readings for the last N hours."""
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    readings = []
    url = f"{BASE_URL}/estaciones/{station_codigo}/meteorologia/?calidad_lluvia=1&size=288"
    pages_fetched = 0
    max_pages = 3  # 3 × 288 readings comfortably covers 24h at 5-min intervals

    while url and pages_fetched < max_pages:
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            page_values = data.get("values", [])
            filtered = [v for v in page_values if v["fecha"] >= since]
            readings.extend(filtered)
            pages_fetched += 1
            if len(filtered) < len(page_values):
                break
            url = data.get("next")
        except Exception:
            break
    return readings


# ── Radar ─────────────────────────────────────────────────────────────────────


def get_radar_url() -> str:
    """Return a cache-busted radar image URL."""
    timestamp = int(datetime.now().timestamp())
    return f"{RADAR_URL}?{timestamp}"


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing fetch_data.py...\n")

    stations = fetch_all_stations()
    classified = classify_stations(stations)

    # Test level thresholds on first level station
    if classified["level"]:
        first = classified["level"][0]["codigo"]
        print(f"\nTesting level thresholds for {first}:")
        thresholds = fetch_level_thresholds(first)
        print(f"  {thresholds}")

        print(f"\nTesting latest level reading for {first}:")
        latest = fetch_latest_level(first)
        print(f"  {latest}")

    # Test precip thresholds
    print("\nTesting precipitation thresholds:")
    precip_thresh = fetch_all_precip_thresholds()
    print(f"  First entry: {precip_thresh[0] if precip_thresh else 'none'}")

    print("\n✓ fetch_data.py working correctly")
