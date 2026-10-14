"""REST API endpoints for procurement audits and document RAG."""

import os
import platform
import shutil
import sys
import tempfile
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from google.cloud import storage

from agent.agents import ProcurementSupervisor
from api.schemas import AuditRequest, AuditResponse, QueryRequest, QueryResponse
from config.settings import settings
from logger import GLOBAL_LOGGER as log
from rag.data_ingestion import ingest_data_from_gcs, ingest_pdf
from rag.retrieval import ask_question


health_router = APIRouter(prefix="/api", tags=["Health"])
status_router = APIRouter(prefix="/api", tags=["Status"])
agent_router = APIRouter(prefix="/api/agent", tags=["Agent"])
rag_router = APIRouter(prefix="/api/rag", tags=["RAG"])

# Tune these limits according to deployment capacity.
UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MiB per PDF

_server_start_time = datetime.now(timezone.utc)
_upload_history: list[dict[str, Any]] = []
_upload_history_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Shared clients and helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_supervisor() -> ProcurementSupervisor:
    """Initialize the supervisor on first use."""
    return ProcurementSupervisor()


@lru_cache(maxsize=1)
def get_storage_client() -> storage.Client:
    """Reuse the Google Cloud Storage client within this process."""
    return storage.Client(project=settings.GCP_PROJECT)


def _record_uploads(results: list[dict[str, Any]]) -> None:
    """Update upload history safely across concurrent requests."""
    with _upload_history_lock:
        _upload_history.extend(results)


def _get_upload_history() -> list[dict[str, Any]]:
    """Return a snapshot instead of exposing the mutable global list."""
    with _upload_history_lock:
        return [item.copy() for item in _upload_history]


def _safe_filename(filename: str | None) -> str:
    """Strip directory components and reject unusable filenames."""
    name = Path(filename or "").name
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return name


def _upload_to_gcs(local_path: str, filename: str) -> None:
    """Save a PDF to the configured bucket."""
    prefix = settings.GCS_PREFIX or ""
    gcs_path = f"{prefix.rstrip('/')}/{filename}" if prefix else filename

    bucket = get_storage_client().bucket(settings.GCS_BUCKET_NAME)
    bucket.blob(gcs_path).upload_from_filename(
        local_path,
        content_type="application/pdf",
    )

    log.info(
        "File saved to GCS",
        filename=filename,
        gcs_path=f"gs://{settings.GCS_BUCKET_NAME}/{gcs_path}",
    )


async def _save_upload(upload: UploadFile, destination: str) -> int:
    """Stream an upload to disk without loading the full PDF into memory."""
    total_bytes = 0
    header = b""

    with open(destination, "wb") as output:
        while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
            total_bytes += len(chunk)

            if total_bytes > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"PDF exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit.",
                )

            if len