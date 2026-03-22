"""Database session — session factory and context manager.

Provides the get_session() context manager for database operations.

Usage:
    from audiobench.core.db_session import get_session

    with get_session() as session:
        session.add(record)
        session.commit()
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy.orm import Session, sessionmaker

from audiobench.core.db_engine import get_engine

# Module-level session factory (lazy init)
_SessionLocal = None


def _get_session_factory():
    """Get or create the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get a database session as a context manager.

    Usage:
        with get_session() as session:
            session.add(record)
            session.commit()
    """
    factory = _get_session_factory()
    session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
