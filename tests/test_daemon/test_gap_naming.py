from unittest.mock import MagicMock, patch

from audiobench.daemon.intelligence.gap_naming import (
    name_knowledge_gap,
    name_knowledge_gap_with_gemini,
    name_knowledge_gap_with_ollama,
)


def test_gap_naming_gemini_tier_success():
    mock_response = MagicMock()
    mock_response.text = '{"title": "Existential Philosophy", "diagnostic": "Missing absurdism topics", "recommended_ingestion": "Albert Camus"}'

    with patch("audiobench.core.settings.get_settings") as mock_settings, \
         patch("google.genai.Client") as mock_client_cls:
        settings = MagicMock()
        settings.gemini_api_key = "test_key"
        settings.gemini_model = "gemini-2.5-flash"
        mock_settings.return_value = settings

        client = MagicMock()
        client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = client

        res = name_knowledge_gap_with_gemini(["camus absurdism", "meaning of life"], "meaning of life")
        assert res is not None
        title, diag, rec = res
        assert title == "Existential Philosophy"
        assert diag == "Missing absurdism topics"
        assert rec == "Albert Camus"


def test_gap_naming_ollama_tier_success():
    mock_resp_data = '{"response": "{\\"title\\": \\"Absurdist Literature\\", \\"diagnostic\\": \\"Queries explore absurdity\\", \\"recommended_ingestion\\": \\"The Stranger\\"}"}'

    with patch("audiobench.core.settings.get_settings") as mock_settings, \
         patch("urllib.request.urlopen") as mock_urlopen:
        settings = MagicMock()
        settings.ollama_base_url = "http://localhost:11434"
        settings.ollama_model = "gpt-oss:120b-cloud"
        mock_settings.return_value = settings

        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_resp_data.encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        res = name_knowledge_gap_with_ollama(["camus absurdism"], "meaning of life")
        assert res is not None
        title, diag, rec = res
        assert title == "Absurdist Literature"
        assert diag == "Queries explore absurdity"
        assert rec == "The Stranger"


def test_gap_naming_fallback_to_medoid_when_llms_unavailable():
    with patch("audiobench.daemon.intelligence.gap_naming.name_knowledge_gap_with_gemini", return_value=None), \
         patch("audiobench.daemon.intelligence.gap_naming.name_knowledge_gap_with_ollama", return_value=None):
        title, diag, rec, source = name_knowledge_gap(
            ["existential meaning of life", "camus absurdism"],
            "existential meaning of life"
        )
        assert source == "vector_medoid"
        assert "existential meaning of life" in title
        assert "existential meaning of life" in diag
        assert rec != ""
