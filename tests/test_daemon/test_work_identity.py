"""Unit tests for work identity resolution."""

import pytest
from audiobench.daemon.intelligence.work_identity import (
    clean_string,
    resolve_work_identity,
)


def test_clean_string():
    assert clean_string(None) is None
    assert clean_string("") is None
    assert clean_string("   ") is None
    assert clean_string("\ufeffClean Text  ") == "Clean Text"


def test_tier1_album_tag_clean():
    tags = {
        "album": "Man's Search for Meaning",
        "artist": "Viktor E. Frankl",
        "title": "02 Experiences in a Concentration Camp_1",
    }
    title, author, source = resolve_work_identity(tags=tags)
    assert title == "Man's Search for Meaning"
    assert author == "Viktor E. Frankl"
    assert source == "tier1_album_tag"


def test_tier1_watts_disc_cleaner():
    tags = {
        "album": "Out of Your Mind - Disc 1 - The Nature of Consciousness [Part 1]",
        "artist": "Alan Watts",
        "title": "Our image of the world",
    }
    title, author, source = resolve_work_identity(tags=tags)
    assert title == "Out of Your Mind"
    assert author == "Alan Watts"
    assert source == "tier1_album_tag"


def test_tier1_unabridged_cleaner():
    tags = {
        "album": "Make Your Bed (Unabridged)",
        "artist": "William H. McRaven",
        "title": "Make Your Bed - Part 1",
    }
    title, author, source = resolve_work_identity(tags=tags)
    assert title == "Make Your Bed"
    assert author == "William H. McRaven"
    assert source == "tier1_album_tag"


def test_tier1_steve_jobs_audio_cd():
    tags = {
        "album": "Steve Jobs",
        "artist": "Walter Isaacson",
        "title": "Audio CD",
    }
    title, author, source = resolve_work_identity(tags=tags)
    assert title == "Steve Jobs"
    assert author == "Walter Isaacson"
    assert source == "tier1_album_tag"


def test_tier1_denylist_rejection():
    tags = {
        "album": "Audio CD",
        "artist": "Unknown Artist",
        "title": "Track 01",
    }
    title, author, source = resolve_work_identity(tags=tags, file_name="track01.mp3")
    assert title == "Track 01"
    assert source == "tier4_track_or_stem"


def test_tier1_disc_pattern_rejection():
    tags = {
        "album": "Disc 01",
        "artist": "Author Name",
        "title": "Actual Lecture Title",
    }
    title, author, source = resolve_work_identity(tags=tags, file_name="lecture.mp3")
    assert title == "Actual Lecture Title"
    assert source == "tier4_track_or_stem"


def test_tier1_channel_masking_rule3():
    # Album equals artist (YouTube channel pattern)
    tags = {
        "album": "RedFrost Motivation",
        "artist": "RedFrost Motivation",
        "title": "Invictus - William Ernest Henley",
    }
    title, author, source = resolve_work_identity(tags=tags, file_name="invictus.mp3")
    assert title == "Invictus - William Ernest Henley"
    assert source == "tier4_track_or_stem"


def test_tier1_voices_of_our_time_isolation():
    tags = {
        "album": "Voices Of Our Time",
        "artist": "Voices Of Our Time",
        "title": "James Baldwin",
    }
    title, author, source = resolve_work_identity(tags=tags, file_name="james_baldwin.mp3")
    assert title == "James Baldwin"
    assert author == "Voices Of Our Time"
    assert source == "tier4_track_or_stem"


def test_tier3_librivox_notes():
    title, author, source = resolve_work_identity(file_name="notes_p1_03_dostoyevsky_64kb.mp3")
    assert title == "Notes from Underground"
    assert author == "Fyodor Dostoyevsky"
    assert source == "tier3_librivox_slug"


def test_tier3_librivox_allthesethingsadded():
    title, author, source = resolve_work_identity(file_name="allthesethingsadded_06_allen_64kb.mp3")
    assert title == "All These Things Added"
    assert author == "James Allen"
    assert source == "tier3_librivox_slug"


def test_tier3_multidisc_genius_makers():
    title, author, source = resolve_work_identity(file_name="Genius Makers-Disk01.mp3")
    assert title == "Genius Makers"
    assert author == "Cade Metz"
    assert source == "tier3_multidisc_filename"


def test_tier3_multidisc_the_trial():
    title, author, source = resolve_work_identity(file_name="Franz_Kafka_-_The_Trial_3.mp3")
    assert title == "The Trial"
    assert author == "Franz Kafka"
    assert source == "tier3_multidisc_filename"


def test_tier3_way_of_zen():
    title, author, source = resolve_work_identity(file_name="01mp3 — Way of Zen CD1.MP3")
    assert title == "The Way of Zen"
    assert author == "Alan Watts"
    assert source == "tier3_multidisc_filename"

    title2, author2, source2 = resolve_work_identity(file_name="Way of Zen CD3_11mp3.MP3")
    assert title2 == "The Way of Zen"
    assert author2 == "Alan Watts"
    assert source2 == "tier3_multidisc_filename"


def test_tier4_fallback_stem_with_emdash():
    title, author, source = resolve_work_identity(file_name="The world as a drama — Alan Watts.mp3")
    assert title == "The world as a drama"
    assert author == "Alan Watts"
    assert source == "tier4_fallback_stem"


def test_tier4_fallback_stem_with_hyphen():
    title, author, source = resolve_work_identity(file_name="Marcus Aurelius - Meditations.mp3")
    assert title == "Meditations"
    assert author == "Marcus Aurelius"
    assert source == "tier4_fallback_stem"


def test_tier4_fallback_simple():
    title, author, source = resolve_work_identity(file_name="brammy homecoming.m4a")
    assert title == "brammy homecoming"
    assert author is None
    assert source == "tier4_fallback_stem"
