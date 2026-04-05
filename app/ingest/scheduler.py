import schedule
import time
import os
from datetime import datetime, timezone
from zenml.logger import get_logger
from app.ingest.pipeline import ingestion_pipeline

logger = get_logger(__name__)

INTERVAL_HOURS = int(os.getenv("INGESTION_INTERVAL_HOURS", "6"))

def run_pipeline():
    logger.info(f"Starting scheduled ingestion at {datetime.now(timezone.utc)}")
    try:
        ingestion_pipeline()
        logger.info("Scheduled ingestion completed successfully")
    except Exception as e:
        logger.error(f"Scheduled ingestion failed: {e}")

if __name__ == "__main__":
    logger.info(f"Scheduler started — running every {INTERVAL_HOURS} hours")

    # Run immediately on startup
    run_pipeline()

    # Then schedule every N hours
    schedule.every(INTERVAL_HOURS).hours.do(run_pipeline)

    while True:
        schedule.run_pending()
        time.sleep(60)  # check every minute