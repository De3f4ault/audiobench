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


def test_tier4_og_mandino_audiobook():
    title, author, source = resolve_work_identity(
        file_name="The Greatest Salesman in The World Og Mandino Audiobook Full.mp3"
    )
    assert title == "The Greatest Salesman in The World"
    assert author == "Og Mandino"
    assert source == "tier4_fallback_stem"


def test_tier4_alan_watts_no_hyphen():
    title, author, source = resolve_work_identity(
        file_name="alan watts we as organisms.mp3"
    )
    assert author == "Alan Watts"
    assert source == "tier4_fallback_stem"


def test_extract_from_transcript_author_announcement():
    from audiobench.daemon.intelligence.entity_extractor import extract_from_transcript
    sample = "The world's greatest salesman. Author: Og Mandino. Narrated by Og Mandino. Hello."
    entity = extract_from_transcript(sample)
    assert entity.author == "Og Mandino"
    assert entity.title == "The world's greatest salesman"
    assert entity.confidence == "high"


def test_tier1_partition_rejection_grokking():
    tags = {
        "album": "Appendix A",
        "artist": "Aditya Y Bhargava",
        "title": "Grokking Algorithms, Second Edition",
    }
    title, author, source = resolve_work_identity(tags=tags, file_name="Appendix_A._Performance_of_AVL_trees.mp3")
    assert title == "Grokking Algorithms, Second Edition"
    assert author == "Aditya Y Bhargava"
    assert source == "tier4_track_or_stem"


def test_tier1_disk_spelling_cleaning():
    tags = {
        "album": "New Testament - Disk 2",
        "artist": "Teaching Company",
        "title": "Lecture 02",
    }
    title, author, source = resolve_work_identity(tags=tags, file_name="new_testament_disk2.mp3")
    assert title == "New Testament"
    assert author == "Teaching Company"
    assert source == "tier1_album_tag"


def test_tier4_partition_suffix_cleaning_think_and_grow_rich():
    tags = {
        "title": "Think and Grow Rich - Part 01",
        "artist": "Napoleon Hill/Anthony Rogers",
    }
    title, author, source = resolve_work_identity(tags=tags, file_name="Think and Grow Rich-Part01_1.mp3")
    assert title == "Think and Grow Rich"
    assert author == "Napoleon Hill"
    assert source == "tier4_track_or_stem"


def test_extract_from_transcript_commercial_announcement_bogle():
    from audiobench.daemon.intelligence.entity_extractor import extract_from_transcript
    sample = (
        "audio renaissance presents the little book of common sense investing "
        "the only way to guarantee your fair share of stock market returns "
        "by john c bogle read for you by tom pinto"
    )
    ent = extract_from_transcript(sample)
    assert ent.author == "John C. Bogle"
    assert ent.title == "The Little Book Of Common Sense Investing"
    assert ent.confidence == "high"


def test_extract_from_transcript_false_positive_rejection():
    from audiobench.daemon.intelligence.entity_extractor import extract_from_transcript

    false_positive_samples = [
        "Listen, for example, to this endorsement by Paul Samuelson...",
        "Perhaps this homely parable, my version of a story told by Warren Buffett in Berkshire Hathaway's 2005 Annual Report...",
        "But over time, the aggregate gains made by Berkshire Hathaway were dwarfed by the massive costs...",
        "Simply because it is investment returns, the earnings and dividends generated by American business...",
        "Chapter 9, The Futility of Chasing Short-Term Performance, Fooled by Randomness. Why do we persist...",
        "Listen to Nassim Nicholas Taleb, author of Fooled by Randomness, who describes how luck is mistaken...",
        "The comparative standard would be the returns earned by Vanguard 500 Index Fund over the same period...",
        "Chapter 16. What would Benjamin Graham have thought about indexing? The first edition was published in 1949. It was written by Benjamin Graham...",
        "It is fair to say that, by Graham's definition, the vast majority of mutual fund investors are speculators...",
        "Chapter 17, The Relentless Rules of Humble Arithmetic. If the message in this book comes across as confident, please remember that my confidence is buttressed by investors including Warren Buffett, Charlie Munger, and David Swensen...",
        "But if you have any doubt, listen to them echo in these words by Clifford Asness...",
    ]

    for sample in false_positive_samples:
        ent = extract_from_transcript(sample)
        # Any candidate author must be strictly low confidence so WorkLinker rejects it from author assignment
        assert ent.confidence != "high" and ent.confidence != "medium", f"Failed for: {sample}"


def test_extract_from_transcript_legitimate_announcements():
    from audiobench.daemon.intelligence.entity_extractor import extract_from_transcript

    ent1 = extract_from_transcript("The Way of Zen by Alan Watts. In this classic series of lectures...")
    assert ent1.author == "Alan Watts"
    assert ent1.title == "The Way of Zen"
    assert ent1.confidence == "medium"

    ent2 = extract_from_transcript('"Make Your Bed" by William H. McRaven. What starts here changes the world.')
    assert ent2.author == "William H. McRaven"
    assert ent2.title == "Make Your Bed"
    assert ent2.confidence == "medium"

    ent3 = extract_from_transcript("The Intelligent Investor. Written by: Benjamin Graham. Read for you by Luke Daniels.")
    assert ent3.author == "Benjamin Graham"
    assert ent3.confidence == "high"

    ent4 = extract_from_transcript("Think and Grow Rich. Author: Napoleon Hill. Unabridged recording.")
    assert ent4.author == "Napoleon Hill"
    assert ent4.confidence == "high"


def test_clean_author_slash_narrator_split():
    from audiobench.daemon.intelligence.work_identity import clean_author

    assert clean_author("Napoleon Hill/Anthony Rogers") == "Napoleon Hill"
    assert clean_author("Napoleon Hill / Anthony Rogers") == "Napoleon Hill"
    assert clean_author("Benjamin Graham/Luke Daniels") == "Benjamin Graham"
    assert clean_author("Paul Strathern/Robert Whitfield") == "Paul Strathern"
    assert clean_author("AC/DC") == "AC/DC"
    assert clean_author("Unknown Artist / Band") == "Unknown Artist / Band"
    assert clean_author(None) is None


def test_partition_regex_tightened_negative_controls():
    from audiobench.daemon.intelligence.work_identity import _PARTITION_PATTERN, _PARTITION_CLEANER, resolve_work_identity

    # Bare keywords (MUST match _PARTITION_PATTERN)
    assert _PARTITION_PATTERN.match("Appendix")
    assert _PARTITION_PATTERN.match("appendix")
    assert _PARTITION_PATTERN.match("Chapter")
    assert _PARTITION_PATTERN.match("Disc")
    assert _PARTITION_PATTERN.match("Disk")
    assert _PARTITION_PATTERN.match("Part")
    assert _PARTITION_PATTERN.match("Section")
    assert _PARTITION_PATTERN.match("CD")

    # Positives with numbers/delimiters (MUST match _PARTITION_PATTERN)
    assert _PARTITION_PATTERN.match("Appendix A")
    assert _PARTITION_PATTERN.match("appendix_a")
    assert _PARTITION_PATTERN.match("Chapter 1")
    assert _PARTITION_PATTERN.match("disc 1")
    assert _PARTITION_PATTERN.match("disc01")
    assert _PARTITION_PATTERN.match("cd 1")
    assert _PARTITION_PATTERN.match("cd01")
    assert _PARTITION_PATTERN.match("disk 2")
    assert _PARTITION_PATTERN.match("Part IV: Advanced Topics")

    # Real English words must NOT match _PARTITION_PATTERN
    assert not _PARTITION_PATTERN.match("DISCIPLINE or REGRET? - David Goggins and Jocko Willink")
    assert not _PARTITION_PATTERN.match("Discourse on Method")
    assert not _PARTITION_PATTERN.match("Particle Physics")
    assert not _PARTITION_PATTERN.match("Party Politics")
    assert not _PARTITION_PATTERN.match("Discworld")
    assert not _PARTITION_PATTERN.match("Partial Differential Equations")
    assert not _PARTITION_PATTERN.match("Charter of Rights")
    assert not _PARTITION_PATTERN.match("Appendix of History")

    # Cleaner positives: bare keywords at suffix (MUST clean)
    assert _PARTITION_CLEANER.sub("", "The Book - Appendix").strip() == "The Book"
    assert _PARTITION_CLEANER.sub("", "The Book - Chapter").strip() == "The Book"
    assert _PARTITION_CLEANER.sub("", "The Book - Part").strip() == "The Book"
    assert _PARTITION_CLEANER.sub("", "The Book - Disc").strip() == "The Book"
    assert _PARTITION_CLEANER.sub("", "The Book - Disk").strip() == "The Book"
    assert _PARTITION_CLEANER.sub("", "The Book - Volume").strip() == "The Book"
    assert _PARTITION_CLEANER.sub("", "The Book, Appendix").strip() == "The Book"

    # Cleaner positives: delimited/numbered partitions (MUST clean)
    assert _PARTITION_CLEANER.sub("", "New Testament - Disk 2").strip() == "New Testament"
    assert _PARTITION_CLEANER.sub("", "Think and Grow Rich - Part 01").strip() == "Think and Grow Rich"
    assert _PARTITION_CLEANER.sub("", "Think and Grow Rich-Part01").strip() == "Think and Grow Rich"
    assert _PARTITION_CLEANER.sub("", "The Book - Ch. 1").strip() == "The Book"
    assert _PARTITION_CLEANER.sub("", "The Book - Vol 2").strip() == "The Book"
    assert _PARTITION_CLEANER.sub("", "The Book - Appendix A").strip() == "The Book"
    assert _PARTITION_CLEANER.sub("", "The Book - CD 3").strip() == "The Book"

    # Real English words must NOT be stripped by _PARTITION_CLEANER
    assert _PARTITION_CLEANER.sub("", "How Did You Die? - A Life-Changing Poem for Troublesome Times").strip() == "How Did You Die? - A Life-Changing Poem for Troublesome Times"
    assert _PARTITION_CLEANER.sub("", "voice_chat_74_20261002_145000").strip() == "voice_chat_74_20261002_145000"
    assert _PARTITION_CLEANER.sub("", "The Book - Changing Poem").strip() == "The Book - Changing Poem"
    assert _PARTITION_CLEANER.sub("", "The Book - Parting Thoughts").strip() == "The Book - Parting Thoughts"
    assert _PARTITION_CLEANER.sub("", "The Book - Discipline and Punish").strip() == "The Book - Discipline and Punish"
    assert _PARTITION_CLEANER.sub("", "The Book - Discourse on Method").strip() == "The Book - Discourse on Method"
    assert _PARTITION_CLEANER.sub("", "The Book - Volume of Blood").strip() == "The Book - Volume of Blood"

    # End-to-end Tier 1 bare album partition rejection
    tags = {"album": "Appendix", "title": "Grokking Algorithms, Second Edition", "artist": "Aditya Y Bhargava"}
    title, author, source = resolve_work_identity(tags=tags, file_name="f.mp3")
    assert title == "Grokking Algorithms, Second Edition"
    assert author == "Aditya Y Bhargava"
    assert source == "tier4_track_or_stem"




