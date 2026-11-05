
def test_study_project_orm_model_exists():
    from audiobench.storage.models import StudyProject, StudySession
    assert StudyProject.__tablename__ == "study_projects"
    assert StudySession.__tablename__ == "study_sessions"
    assert hasattr(StudyProject, "audio_file_id")
    assert hasattr(StudySession, "project_id")
    assert hasattr(StudySession, "chapter_ids")
    assert hasattr(StudySession, "memoir_id")
    assert hasattr(StudySession, "closed_at")
