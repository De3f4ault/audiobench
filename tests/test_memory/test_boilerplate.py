"""Unit tests for audiobench.memory.boilerplate."""

from __future__ import annotations

import pytest

from audiobench.memory.boilerplate import (
    clean_audio_disclaimers,
    is_pure_boilerplate,
    strip_outro,
    strip_preamble,
)


# ---------------------------------------------------------------------------
# strip_preamble tests
# ---------------------------------------------------------------------------


class TestStripPreamble:
    def test_librivox_standard(self) -> None:
        text = (
            "This isa LibriVox recording. Al libriVox recordings are in the public domain. "
            "For more information, or to volunteer, please visit LibriVox.org. "
            "chapter two of the great gatsby by f scott fitzgerald."
        )
        result = strip_preamble(text)
        assert "LibriVox" not in result
        assert "great gatsby" in result

    def test_librivox_variant_libravox(self) -> None:
        text = (
            "This isa Libravox Recording. All libravox recordings are in the public domain. "
            "For more information visit LibriVox.org. Real content starts here about the human condition."
        )
        result = strip_preamble(text)
        assert "Libravox" not in result
        assert "human condition" in result

    def test_librivox_variant_liebervox(self) -> None:
        text = (
            "This isa Liebervox Recording. All recordings in the public domain. "
            "Philosophical content about consciousness."
        )
        result = strip_preamble(text)
        assert "Liebervox" not in result

    def test_librivox_variant_librefox(self) -> None:
        text = (
            "Crime and Punishment Part 6 Chapter 1 This isa Librefox recording. "
            "All librivox recordings are in the public domain. "
            "Raskolnikov sat alone in his room."
        )
        result = strip_preamble(text)
        assert "Librefox" not in result
        assert "Raskolnikov" in result

    def test_audible_preamble(self) -> None:
        text = "This is Audible. Lioncrest Publishing presents Can't Hurt Me by David Goggins."
        result = strip_preamble(text)
        assert "This is Audible" not in result

    def test_sounds_true_preamble(self) -> None:
        text = "Sounds True presents Out of Your Mind, Essential Listening from the Alan Watts Audio Archives."
        result = strip_preamble(text)
        assert "Sounds True" not in result

    def test_harper_audio_preamble(self) -> None:
        text = "Harper Audio presents Never Split the Difference, Negotiating as if Your Life Depended on It."
        result = strip_preamble(text)
        assert "Harper Audio" not in result

    def test_macmillan_audio_preamble(self) -> None:
        text = "Macmillan Audio presents Undeniable. How to reach the top and stay there."
        result = strip_preamble(text)
        assert "Macmillan Audio" not in result

    def test_audio_renaissance_preamble(self) -> None:
        text = "Audio Renaissance presents The Way of Zen by Alan Watts."
        result = strip_preamble(text)
        assert "Audio Renaissance" not in result

    def test_no_preamble_unchanged(self) -> None:
        text = (
            "The mind is like water. When still, it reflects everything perfectly. "
            "When disturbed, it can reflect nothing well."
        )
        result = strip_preamble(text)
        assert result == text

    def test_hybrid_preserves_content_after_preamble(self) -> None:
        """Hybrid chunk: disclaimer + real content. Content must survive."""
        text = (
            "This isa LibriVox recording. Al libriVox recordings are in the public domain. "
            "For more information visit LibriVox.org. "
            "chapter two of the great gatsby by f scott fitzgerald about half way between "
            "west egg and new york the motor road hastily joins the railroad."
        )
        result = strip_preamble(text)
        assert "great gatsby" in result.lower()
        assert len(result.split()) > 15


# ---------------------------------------------------------------------------
# strip_outro tests
# ---------------------------------------------------------------------------


class TestStripOutro:
    def test_audible_hopes_outro(self) -> None:
        text = "Real philosophical content here. Audible hopes you have enjoyed this program."
        result = strip_outro(text)
        assert "Audible hopes" not in result
        assert "philosophical content" in result

    def test_we_hope_unabridged(self) -> None:
        text = (
            "Great content about suffering and meaning. "
            "We hope you have enjoyed this unabridged production of Never Split the Difference."
        )
        result = strip_outro(text)
        assert "We hope" not in result
        assert "suffering" in result

    def test_this_concludes_session(self) -> None:
        text = (
            "More content about presence and awareness. "
            "This concludes Session 2 of Out of Your Mind Essential Listening from the Alan Watts Audio Archives."
        )
        result = strip_outro(text)
        assert "This concludes" not in result
        assert "presence" in result

    def test_end_of_chapter_recording_by(self) -> None:
        text = "Good content about suffering. End of chapter 12 of book 12. Recording by Jane Smith for LibriVox."
        result = strip_outro(text)
        assert "End of chapter" not in result
        assert "suffering" in result

    def test_thanks_for_listening(self) -> None:
        text = "Interesting philosophy about consciousness. Thanks for listening and do not forget to subscribe."
        result = strip_outro(text)
        assert "Thanks for listening" not in result
        assert "consciousness" in result

    def test_no_outro_unchanged(self) -> None:
        text = "Pure philosophical content with no credits or outro at all."
        result = strip_outro(text)
        assert result == text


# ---------------------------------------------------------------------------
# clean_audio_disclaimers (combined) tests
# ---------------------------------------------------------------------------


class TestCleanAudioDisclaimers:
    def test_strips_both_ends(self) -> None:
        text = (
            "This is Audible. Harper Audio presents The Truth. "
            "Actual content about human relationships and social dynamics. "
            "Audible hopes you have enjoyed this program."
        )
        result = clean_audio_disclaimers(text)
        assert "This is Audible" not in result
        assert "Audible hopes" not in result
        assert "social dynamics" in result

    def test_no_disclaimers_unchanged(self) -> None:
        text = "Consciousness is the fundamental nature of reality."
        result = clean_audio_disclaimers(text)
        assert result == text


# ---------------------------------------------------------------------------
# is_pure_boilerplate tests
# ---------------------------------------------------------------------------


class TestIsPureBoilerplate:
    def test_pure_librivox_stamp(self) -> None:
        text = (
            "This isa LibriVox recording. Al libriVox recordings are in the public domain. "
            "For more information, or to volunteer, please visit LibriVox.org."
        )
        assert is_pure_boilerplate(text) is True

    def test_pure_libravox_variant(self) -> None:
        text = (
            "This isa Libravox Recording. All libravox recordings are in the public domain. "
            "For more information visit LibriVox.org."
        )
        assert is_pure_boilerplate(text) is True

    def test_pure_liebervox_variant(self) -> None:
        assert is_pure_boilerplate(
            "This isa Liebervox Recording. All recordings in the public domain."
        ) is True

    def test_pure_audible_stamp(self) -> None:
        assert is_pure_boilerplate("This is Audible.") is True

    def test_pure_sounds_true(self) -> None:
        text = "Sounds True presents Out of Your Mind, Essential Listening from the Alan Watts Audio Archives."
        assert is_pure_boilerplate(text) is True

    def test_hybrid_is_not_pure(self) -> None:
        """Hybrid chunk: preamble + substantial content. Must NOT be flagged."""
        text = (
            "This isa LibriVox recording. Al libriVox recordings are in the public domain. "
            "For more information visit LibriVox.org. "
            "chapter two of the great gatsby by f scott fitzgerald about half way between "
            "west egg and new york the motor road hastily joins the railroad and runs beside it "
            "for a quarter of a mile so as to shrink away from a certain desolate area of land."
        )
        assert is_pure_boilerplate(text) is False

    def test_real_content_not_flagged(self) -> None:
        """Clean philosophical text must never be flagged as boilerplate."""
        text = (
            "The mind is like water. When still, it reflects everything perfectly. "
            "When disturbed, it can reflect nothing well. This is the nature of consciousness."
        )
        assert is_pure_boilerplate(text) is False

    def test_short_text_without_signature_not_flagged(self) -> None:
        """A short sentence with no boilerplate signature must not be flagged."""
        assert is_pure_boilerplate("Short sentence.") is False
        assert is_pure_boilerplate("S1. S2. S3.") is False

    def test_dostoyevsky_narrative_not_flagged(self) -> None:
        text = (
            "if he is not killed and recovers it will be first rate for then he will bear "
            "witness that dmitri fyodorovitch was not the murderer and smerdyakov alone committed the crime."
        )
        assert is_pure_boilerplate(text) is False
