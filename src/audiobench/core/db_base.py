"""Declarative base — SQLAlchemy ORM base class for all models.

Usage:
    from audiobench.core.db_base import Base

    class MyModel(Base):
        __tablename__ = "my_table"
        ...
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""

    pass
