import uuid
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware # Import the middleware

from app.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.celery_client import celery_app
from app.database import create_db_and_tables, find_job_by_url, create_ingestion_job
from app.query import query_rag_engine

# Initialize the FastAPI application
app = FastAPI(
    title="Scalable RAG Engine",
    description="An API for asynchronous ingestion and querying of web content.",
    version="1.0.0"
)

# --- Add CORS Middleware ---
# This is the fix for the "Failed to fetch" error in the browser UI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# --- Database Setup ---

@app.on_event("startup")
def on_startup():
    print("API is starting up. Ensuring MongoDB indexes...")
    create_db_and_tables()
    print("MongoDB indexes ensured.")

# --- API Endpoints ---

@app.get("/", tags=["Health Check"])
def read_root():
    return {"message": "RAG Engine API is running"}


@app.post("/ingest-url",
          response_model=IngestResponse,
          status_code=status.HTTP_202_ACCEPTED,
          tags=["Ingestion"])
def ingest_url(request: IngestRequest):
    """
    Accepts a URL, saves it to the database, and schedules it for processing.
    """
    try:
        existing_job = find_job_by_url(str(request.url))
        if existing_job:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"URL has already been submitted. Job ID: {existing_job['id']}, Status: {existing_job['status']}"
            )

        job_id = create_ingestion_job(str(request.url))

        celery_app.send_task(
            "process_url_task",
            args=[str(job_id), str(request.url)]
        )

        return IngestResponse(job_id=uuid.UUID(str(job_id)))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create and schedule job: {str(e)}"
        )

@app.post("/query",
          response_model=QueryResponse,
          tags=["Query"])
def query(request: QueryRequest):
    """
    Accepts a user query and returns a grounded answer from the knowledge base.
    """
    try:
        result = query_rag_engine(request.query)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during the query process: {str(e)}"
        )