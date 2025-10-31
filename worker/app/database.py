import os
from pymongo import MongoClient

# Get the MongoDB connection string from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set.")

_mongo_client = MongoClient(DATABASE_URL)
try:
    _db = _mongo_client.get_default_database()
except Exception:
    _db = _mongo_client["rag_db"]

ingestion_jobs = _db["ingestion_jobs"]