from typing import Any

from fastapi import APIRouter, HTTPException

from audiobench.core.db_engine import init_db
from audiobench.storage.repository import TranscriptionRepository

router = APIRouter()

# Initialize DB on import so the repository is ready
init_db()
repo = TranscriptionRepository()


@router.get("/")
def get_history(limit: int = 50, search: str | None = None) -> list[dict[str, Any]]:
    """Get transcription history, optionally filtered by search."""
    if search:
        return repo.search(search, limit=limit, session=None)
    return repo.get_history(limit=limit)


@router.get("/{transcription_id}")
def get_transcription(transcription_id: int) -> dict[str, Any]:
    """Get a single transcription by ID."""
    data = repo.get_by_id(transcription_id, session=None)
    if not data:
        raise HTTPException(status_code=404, detail="Transcription not found")
    return data


@router.delete("/{transcription_id}")
def delete_transcription(transcription_id: int) -> dict:
    """Delete a transcription by ID."""
    success = repo.delete_by_id(transcription_id)
    if not success:
        raise HTTPException(status_code=404, detail="Transcription not found")
    return {"status": "success", "id": transcription_id}


@router.delete("/")
def delete_all_history() -> dict:
    """Delete all transcription history."""
    count = repo.delete_all()
    return {"status": "success", "deleted_count": count}
