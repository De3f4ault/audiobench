"""Tests for core settings — validation, defaults, and presets."""

from __future__ import annotations

import pytest

from audiobench.core.settings import AudioBenchSettings


class TestSettingsDefaults:
    """Test that default values are sensible."""

    def test_default_model(self):
        s = AudioBenchSettings()
        assert s.model_name == "large-v3-turbo"

    def test_default_device(self):
        s = AudioBenchSettings()
        assert s.device == "auto"

    def test_default_output_format(self):
        s = AudioBenchSettings()
        assert s.output_format == "txt"

    def test_default_speed_preset(self):
        s = AudioBenchSettings()
        assert s.speed_preset == "balanced"

    def test_data_dir_is_absolute(self):
        s = AudioBenchSettings()
        assert s.data_dir.is_absolute()

    def test_database_url_points_to_data_dir(self):
        s = AudioBenchSettings()
        assert "data" in s.database_url
        assert "transcriptions.db" in s.database_url


class TestSettingsValidation:
    """Test field validators catch bad values."""

    def test_invalid_model_name(self):
        with pytest.raises(ValueError, match="Invalid model"):
            AudioBenchSettings(model_name="nonexistent")

    def test_invalid_device(self):
        with pytest.raises(ValueError, match="Invalid device"):
            AudioBenchSettings(device="tpu")

    def test_invalid_compute_type(self):
        with pytest.raises(ValueError, match="Invalid compute_type"):
            AudioBenchSettings(compute_type="brain16")

    def test_invalid_output_format(self):
        with pytest.raises(ValueError, match="Invalid output_format"):
            AudioBenchSettings(output_format="docx")

    def test_invalid_speed_preset(self):
        with pytest.raises(ValueError, match="Invalid speed_preset"):
            AudioBenchSettings(speed_preset="ludicrous")

    def test_empty_language_becomes_none(self):
        s = AudioBenchSettings(language="")
        assert s.language is None

    def test_none_language_stays_none(self):
        s = AudioBenchSettings(language=None)
        assert s.language is None

    def test_valid_language_preserved(self):
        s = AudioBenchSettings(language="en")
        assert s.language == "en"

    def test_empty_hf_token_becomes_none(self):
        s = AudioBenchSettings(hf_token="")
        assert s.hf_token is None


class TestSpeedPresetResolution:
    """Test that speed presets resolve to correct values."""

    def test_fast_beam_size(self):
        s = AudioBenchSettings()
        assert s.resolve_beam_size("fast") == 1

    def test_balanced_beam_size(self):
        s = AudioBenchSettings()
        assert s.resolve_beam_size("balanced") == 3

    def test_accurate_beam_size(self):
        s = AudioBenchSettings()
        assert s.resolve_beam_size("accurate") == 5

    def test_fast_batch_size(self):
        s = AudioBenchSettings()
        assert s.resolve_batch_size("fast") == 8

    def test_fast_temperature_is_zero(self):
        s = AudioBenchSettings()
        assert s.resolve_temperature("fast") == 0

    def test_balanced_temperature_is_fallback_chain(self):
        s = AudioBenchSettings()
        result = s.resolve_temperature("balanced")
        assert isinstance(result, list)
        assert result[0] == 0

    def test_condition_on_previous_text_only_accurate(self):
        s = AudioBenchSettings()
        assert s.resolve_condition_on_previous_text("accurate") is True
        assert s.resolve_condition_on_previous_text("fast") is False
        assert s.resolve_condition_on_previous_text("balanced") is False

    def test_resolve_cpu_threads_positive(self):
        s = AudioBenchSettings()
        assert s.resolve_cpu_threads() >= 1
