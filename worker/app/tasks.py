from datetime import datetime
from app.celery_app import celery_app
from app.database import ingestion_jobs
from app.ingest import fetch_and_clean_text, chunk_text, store_chunks_in_db

@celery_app.task(name="process_url_task", bind=True)
def process_url_task(job_id: str, url: str):
    """
    The main background task to process a URL.
    - Updates job status in PostgreSQL.
    - Fetches and cleans text from the URL.
    - Chunks the text.
    - Stores the chunks in ChromaDB.
    """
    print(f"Worker received job {job_id}: Processing URL {url}")
    try:
        # 1. Update status to PROCESSING
        result = ingestion_jobs.update_one({"id": job_id}, {"$set": {"status": "PROCESSING", "updated_at": datetime.utcnow()}})
        if result.matched_count == 0:
            print(f"Error: Job {job_id} not found in database.")
            return

        # 2. Fetch and clean the content
        text = fetch_and_clean_text(url)
        if not text:
            raise ValueError("Failed to get any text content from URL.")

        # 3. Chunk the text
        chunks = chunk_text(text)

        # 4. Store chunks in ChromaDB (which also handles embedding)
        store_chunks_in_db(url, chunks)

        # 5. Update status to COMPLETED
        ingestion_jobs.update_one({"id": job_id}, {"$set": {"status": "COMPLETED", "updated_at": datetime.utcnow()}})

        print(f"Successfully finished processing job {job_id}.")
        return {"job_id": job_id, "status": "COMPLETED"}

    except Exception as e:
        print(f"Error processing job {job_id}: {e}")
        ingestion_jobs.update_one({"id": job_id}, {"$set": {"status": "FAILED", "updated_at": datetime.utcnow()}})
        # Re-raise the exception to let Celery know the task failed
        raise
