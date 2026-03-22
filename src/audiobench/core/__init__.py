"""Core infrastructure — shared services used by all features.

Provides:
    - settings: Application configuration (Pydantic)
    - logger_factory: Structured logging
    - db_engine: SQLAlchemy engine + init_db
    - db_session: Session context manager
    - db_base: ORM DeclarativeBase
    - error_types: Exception hierarchy
"""
