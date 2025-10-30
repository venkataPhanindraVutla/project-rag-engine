import os
import uuid
from datetime import datetime
from pymongo import MongoClient, ASCENDING

# Get the MongoDB connection string from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set.")

# Initialize MongoDB client and collection
_mongo_client = MongoClient(DATABASE_URL)
_db = _mongo_client.get_default_database() if _mongo_client.get_default_database() else _mongo_client["rag_db"]
ingestion_jobs = _db["ingestion_jobs"]


def create_db_and_tables():
    """Ensure indexes exist for efficient lookups."""
    ingestion_jobs.create_index([("url", ASCENDING)], unique=True)


def create_ingestion_job(url: str) -> str:
    job_id = str(uuid.uuid4())
    now = datetime.utcnow()
    ingestion_jobs.insert_one({
        "id": job_id,
        "url": url,
        "status": "PENDING",
        "created_at": now,
        "updated_at": now,
    })
    return job_id


def find_job_by_url(url: str):
    return ingestion_jobs.find_one({"url": url})