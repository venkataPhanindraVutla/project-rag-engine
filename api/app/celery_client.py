import os
from celery import Celery
from dotenv import load_dotenv

# Load environment variables to get the Redis URL
load_dotenv()

redis_url = os.getenv("REDIS_URL")

if not redis_url:
    raise ValueError("REDIS_URL environment variable not set.")

# This creates a minimal Celery instance for the API.
# It's configured ONLY to send tasks to the Redis broker.
# It does not need to know about the task code itself.
celery_app = Celery(
    "api_client",
    broker=redis_url,
    backend=redis_url # Backend is needed to create task objects
)