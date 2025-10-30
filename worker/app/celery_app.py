import os
from celery import Celery
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Redis URL from the environment variables
redis_url = os.getenv("REDIS_URL")

if not redis_url:
    raise ValueError("REDIS_URL environment variable not set. Please check your .env file.")

# Initialize the Celery app
# The first argument is the name of the current module.
# 'broker' specifies the URL of the message broker (Redis).
# 'backend' specifies the URL of the result backend (also Redis).
celery_app = Celery(
    "tasks",
    broker=redis_url,
    backend=redis_url,
    include=["app.tasks"] # Explicitly include the tasks module
)

# Optional configuration
celery_app.conf.update(
    task_track_started=True,
)