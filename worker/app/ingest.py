import os
import httpx
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# --- Configuration ---
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")

# --- ChromaDB Client ---
# Initialize a persistent ChromaDB client that connects to our server.
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# --- Embedding Function ---
# This function will be used by ChromaDB to automatically create embeddings.
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

# --- Get or Create Collection ---
# This ensures our collection exists in ChromaDB.
collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    embedding_function=embedding_function
)

# --- Core Ingestion Functions ---

def fetch_and_clean_text(url: str) -> str:
    """Fetches content from a URL and cleans it to get raw text."""
    print(f"Fetching content from {url}...")
    # Browser-like headers improve success rate for modern sites/CDNs
    default_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    # Retry with simple exponential backoff
    backoff_seconds = [1, 2, 4]
    last_error = None
    try:
        with httpx.Client(follow_redirects=True, timeout=30, headers=default_headers) as client:
            for attempt, delay in enumerate([0] + backoff_seconds):
                if delay:
                    # Sleep without importing time at top (local import keeps global scope clean)
                    import time as _time
                    _time.sleep(delay)
                try:
                    response = client.get(url)
                    # Some CDNs return 403/406 unless UA/headers look right
                    if response.status_code in (403, 406) and attempt < len(backoff_seconds):
                        # Tweak headers slightly and retry
                        client.headers.update({
                            "User-Agent": (
                                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15"
                            )
                        })
                        last_error = httpx.HTTPStatusError("Blocked by server (403/406)", request=response.request, response=response)
                        continue
                    response.raise_for_status()
                    html = response.text

                    soup = BeautifulSoup(html, 'html.parser')

                    # Remove non-content elements
                    for tag in soup(["script", "style", "noscript"]):
                        tag.decompose()

                    text = soup.get_text(separator='\n')

                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)

                    print(f"Successfully fetched and cleaned text from {url}.")
                    return cleaned_text
                except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                    last_error = e
                    continue
                except httpx.HTTPStatusError as e:
                    last_error = e
                    # Retry only for 5xx
                    if e.response is not None and 500 <= e.response.status_code < 600 and attempt < len(backoff_seconds):
                        continue
                    raise
        # If we exhausted retries
        if last_error:
            raise last_error
        raise RuntimeError("Failed to fetch URL for unknown reasons.")
    except httpx.HTTPStatusError as e:
        print(f"HTTP error fetching {url}: {e}")
        raise
    except Exception as e:
        print(f"An error occurred during fetch/clean of {url}: {e}")
        raise


def chunk_text(text: str) -> list[str]:
    """Splits a long text into smaller, manageable chunks."""
    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    print(f"Text split into {len(chunks)} chunks.")
    return chunks


def store_chunks_in_db(url: str, chunks: list[str]):
    """Stores text chunks and their metadata in ChromaDB."""
    if not chunks:
        print("No chunks to store.")
        return

    print(f"Storing {len(chunks)} chunks in ChromaDB...")
    # Create unique IDs for each chunk to prevent duplicates
    ids = [f"{url}_{i}" for i, _ in enumerate(chunks)]
    
    # Associate metadata with each chunk
    metadatas = [{"source_url": url} for _ in chunks]
    
    # Add the documents to the collection. ChromaDB will handle the embedding.
    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas
    )
    print("Successfully stored chunks in ChromaDB.")