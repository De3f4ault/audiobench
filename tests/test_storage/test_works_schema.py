import sqlite3

from audiobench.core.db_engine import init_db


def test_works_table_created_by_migration(tmp_path, monkeypatch):
    db_path = tmp_path / "test_works.db"
    monkeypatch.setenv("AUDIOBENCH_DATABASE_URL", f"sqlite:///{db_path}")
    # Clear lru_cache for get_settings so it picks up the env var
    import audiobench.core.db_engine
    from audiobench.core.settings import get_settings
    get_settings.cache_clear()
    audiobench.core.db_engine._engine = None

    init_db()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("PRAGMA table_info(works)")
        columns = {row[1] for row in cursor.fetchall()}

        assert "id" in columns
        assert "title" in columns
        assert "author" in columns
        assert "created_at" in columns

def test_work_id_fk_on_expressions_and_audio_files(tmp_path, monkeypatch):
    db_path = tmp_path / "test_works2.db"
    monkeypatch.setenv("AUDIOBENCH_DATABASE_URL", f"sqlite:///{db_path}")
    import audiobench.core.db_engine
    from audiobench.core.settings import get_settings
    get_settings.cache_clear()
    audiobench.core.db_engine._engine = None
    init_db()

    with sqlite3.connect(db_path) as conn:
        # Check audio_files
        cursor = conn.execute("PRAGMA table_info(audio_files)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "work_id" in columns

        # Check expressions
        cursor = conn.execute("PRAGMA table_info(expressions)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "work_id" in columns

def test_work_id_nullable_on_expression(tmp_path, monkeypatch):
    db_path = tmp_path / "test_works3.db"
    monkeypatch.setenv("AUDIOBENCH_DATABASE_URL", f"sqlite:///{db_path}")
    import audiobench.core.db_engine
    from audiobench.core.settings import get_settings
    get_settings.cache_clear()
    audiobench.core.db_engine._engine = None
    init_db()

    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO works (id, title, created_at) VALUES (1, 'Test Work', '2023-01-01')")
        conn.execute("INSERT INTO audio_files (file_path, file_name, file_size_bytes, duration_seconds, sample_rate, channels, created_at, format, tags, file_hash) VALUES ('x', 'y', 0, 0.0, 0, 0, '2023-01-01', 'mp3', '[]', 'hash')")
        conn.execute("INSERT INTO expressions (content, source_type, created_at) VALUES ('test', 'type', '2023-01-01')")

        # Since no exception is raised, nullable FK is respected
        cursor = conn.execute("SELECT work_id FROM expressions")
        assert cursor.fetchone()[0] is None
