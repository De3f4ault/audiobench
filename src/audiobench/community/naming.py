"""LLM-based community naming with multi-tier fallback chain.

Hierarchy:
    1. Gemini (via google-genai) with constrained conceptual prompt.
    2. Ollama (local HTTP daemon) with same prompt.
    3. Deterministic Author + Work Fallback: "{Author1} & {Author2} — {Work}".

Avoids raw TF-IDF to prevent character names, Russian patronymics, and memoir proper nouns
from contaminating school titles.
"""

from __future__ import annotations

import collections
import json
import re
import sqlite3
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

from audiobench.core.logger_factory import get_logger

logger = get_logger("community.naming")


def generate_fallback_name(conn: sqlite3.Connection, community_id: int) -> str:
    """Generate a clean, deterministic Author + Work title fallback.

    Produces e.g. "David Goggins & Cameron Hanes — Can't Hurt Me / Endure"
    or "Alan Watts — Out of Your Mind" without raw TF-IDF proper noun noise.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(w.author, 'Unknown Author'), COALESCE(w.title, 'Unknown Work'), COUNT(*)
        FROM expressions e
        LEFT JOIN works w ON e.work_id = w.id
        WHERE e.community_id = ?
        GROUP BY w.author, w.title
        ORDER BY COUNT(*) DESC
        """,
        (community_id,),
    )
    rows = cur.fetchall()
    if not rows:
        return f"School #{community_id}"

    # Collect author counts across all works
    author_counts: collections.Counter = collections.Counter()
    work_counts: collections.Counter = collections.Counter()
    for auth, work, cnt in rows:
        author_counts[auth] += cnt
        work_counts[work] += cnt

    # Filter out "Unknown Author" if named authors exist
    named_authors = [a for a, _ in author_counts.most_common() if a != "Unknown Author"]
    if named_authors:
        top_authors = named_authors[:2]
    else:
        top_authors = [author_counts.most_common(1)[0][0]]

    # Clean work titles
    named_works = [w for w, _ in work_counts.most_common() if w != "Unknown Work"]
    top_work = named_works[0] if named_works else "Library Collection"

    # Truncate work title before subtitle colons or long descriptions
    clean_work = top_work.split(":")[0].split(" - ")[0].strip()
    if len(clean_work) > 30:
        clean_work = clean_work[:30].rsplit(" ", 1)[0].strip()

    author_str = " & ".join(top_authors)
    return f"{author_str} — {clean_work}"


def _clean_llm_title(text: str) -> str:
    """Clean markdown artifacts, quotes, and punctuation from an LLM-generated title."""
    cleaned = text.strip().strip("\"'").strip()
    # Strip markdown bold/italics
    cleaned = re.sub(r"[*_`]", "", cleaned)
    # Strip leading "Title:" or "School of:"
    cleaned = re.sub(r"^(Title|School|Philosophy|Movement):\s*", "", cleaned, flags=re.IGNORECASE)
    # Replace multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


_gemini_cooldown_until: float = 0.0


def name_community_with_gemini(
    medoid_text: str,
    top_authors: List[str],
    top_works: List[str],
) -> Optional[str]:
    """Generate a 3-5 word conceptual school title using Gemini."""
    global _gemini_cooldown_until
    import time
    if time.time() < _gemini_cooldown_until:
        logger.debug("Skipping Gemini naming (provider in cooldown for %.0fs)", _gemini_cooldown_until - time.time())
        return None

    try:
        from audiobench.core.settings import get_settings

        settings = get_settings()
        api_key = getattr(settings, "gemini_api_key", None)
        if not api_key:
            return None

        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=10_000),  # 10s timeout
        )

        prompt = (
            "You are an intellectual historian naming an emergent School of Thought in an audio library.\n\n"
            f"Excerpts:\n\"{medoid_text[:400]}\"\n\n"
            f"Authors: {', '.join(top_authors)}\n"
            f"Works: {', '.join(top_works)}\n\n"
            "Instructions:\n"
            "1. Name the core philosophical movement, worldview, or intellectual dilemma in 3 to 5 words.\n"
            "2. Ground the title strictly in the historical era and ideas of the authors (never apply anachronistic labels like 20th-century postmodern/deconstructive theory to 19th-century literature).\n"
            "3. Do NOT use industry or geographic labels (e.g. avoid 'Silicon Valley culture'; use conceptual terms like 'Technological Democratization' or 'Digital Pioneer Entrepreneurship').\n"
            "4. Highlight the specific thematic conflict (e.g. distinguish Russian guilt/redemption from existential freedom/choice).\n"
            "5. Return ONLY the plain text 3-5 word title with no explanation or quotes."
        )

        model_name = getattr(settings, "gemini_model", "gemini-2.5-flash")
        try:
            config = types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=200,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        except Exception:
            config = types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=200,
            )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        if response and response.text:
            cleaned = _clean_llm_title(response.text)
            if cleaned and len(cleaned.split()) <= 8:
                return cleaned
    except Exception as e:
        err_msg = str(e)
        logger.debug("Gemini naming failed: %s", err_msg)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
            _gemini_cooldown_until = time.time() + 3600.0  # 1 hour provider-level cooldown
            logger.warning("Gemini 429 quota exhaustion detected; provider in cooldown for 1 hour")

    return None


def name_community_with_ollama(
    medoid_text: str,
    top_authors: List[str],
    top_works: List[str],
) -> Optional[str]:
    """Generate a 3-5 word conceptual school title using local Ollama daemon."""
    try:
        from audiobench.core.settings import get_settings

        settings = get_settings()
        base_url = getattr(settings, "ollama_base_url", "http://localhost:11434").rstrip("/")
        model_name = getattr(settings, "ollama_model", "gpt-oss:120b-cloud")

        prompt = (
            "You are an intellectual historian naming an emergent School of Thought in an audio library.\n\n"
            f"Excerpts:\n\"{medoid_text[:400]}\"\n\n"
            f"Authors: {', '.join(top_authors)}\n"
            f"Works: {', '.join(top_works)}\n\n"
            "Instructions:\n"
            "1. Name the core philosophical movement, worldview, or intellectual dilemma in 3 to 5 words.\n"
            "2. Ground the title strictly in the historical era and ideas of the authors (never apply anachronistic labels like 20th-century postmodern/deconstructive theory to 19th-century literature).\n"
            "3. Do NOT use industry or geographic labels (e.g. avoid 'Silicon Valley culture'; use conceptual terms like 'Technological Democratization' or 'Digital Pioneer Entrepreneurship').\n"
            "4. Highlight the specific thematic conflict (e.g. distinguish Russian guilt/redemption from existential freedom/choice).\n"
            "5. Return ONLY the plain text 3-5 word title with no explanation or quotes."
        )

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.2, "num_predict": 1200},
        }

        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw = data.get("response", "")
            if raw:
                cleaned = _clean_llm_title(raw)
                cleaned = cleaned.rstrip(".")
                if cleaned and len(cleaned.split()) <= 8:
                    return cleaned
    except Exception as e:
        logger.debug("Ollama naming failed: %s", e)

    return None


def name_community(conn: sqlite3.Connection, community_id: int) -> Tuple[str, str]:
    """Execute the naming chain for a single community.

    Returns:
        (title, name_source) where name_source in ('llm_gemini', 'llm_ollama', 'fallback').
    """
    cur = conn.cursor()

    # 1. Fetch medoid content
    cur.execute(
        """
        SELECT e.content
        FROM expression_communities c
        JOIN expressions e ON c.medoid_expr_id = e.id
        WHERE c.community_id = ?
        """,
        (community_id,),
    )
    med_row = cur.fetchone()
    medoid_text = med_row[0] if med_row else ""

    # 2. Fetch top authors and works
    cur.execute(
        """
        SELECT COALESCE(w.author, 'Unknown'), COALESCE(w.title, 'Unknown'), COUNT(*)
        FROM expressions e
        LEFT JOIN works w ON e.work_id = w.id
        WHERE e.community_id = ?
        GROUP BY w.author, w.title
        ORDER BY COUNT(*) DESC
        LIMIT 5
        """,
        (community_id,),
    )
    rows = cur.fetchall()
    top_authors = list(dict.fromkeys(r[0] for r in rows if r[0] != "Unknown"))[:2]
    top_works = list(dict.fromkeys(r[1] for r in rows if r[1] != "Unknown"))[:2]

    # If medoid exists, attempt LLM naming
    if medoid_text:
        # Tier 1: Gemini
        gemini_name = name_community_with_gemini(medoid_text, top_authors, top_works)
        if gemini_name:
            return gemini_name, "llm_gemini"

        # Tier 2: Ollama
        ollama_name = name_community_with_ollama(medoid_text, top_authors, top_works)
        if ollama_name:
            return ollama_name, "llm_ollama"

    # Tier 3: Deterministic Fallback
    fallback = generate_fallback_name(conn, community_id)
    return fallback, "fallback"


def _execute_naming_sweep(conn: sqlite3.Connection, batch_size: int = 5) -> int:
    """Execute naming sweep across a batch of eligible communities.

    Applies tiered exponential backoff on repeated failures:
        0 failures: Immediate
        1 failure:  30 minutes
        2 failures: 2 hours
        3 failures: 8 hours
        4 failures: 24 hours
        5 failures: 72 hours
        6+ failures: 168 hours (7 days / weekly quota window)
    """
    import time
    cur = conn.cursor()
    cur.execute(
        """
        SELECT community_id
        FROM expression_communities
        WHERE is_active = 1 AND name_source IN ('pending', 'fallback')
          AND (
              naming_attempted_at IS NULL
              OR (
                  strftime('%s', 'now') - strftime('%s', naming_attempted_at)
              ) >= CASE 
                  WHEN naming_failure_count = 0 THEN 0
                  WHEN naming_failure_count = 1 THEN 1800
                  WHEN naming_failure_count = 2 THEN 7200
                  WHEN naming_failure_count = 3 THEN 28800
                  WHEN naming_failure_count = 4 THEN 86400
                  WHEN naming_failure_count = 5 THEN 259200
                  ELSE 604800
              END
          )
        ORDER BY node_count DESC
        LIMIT ?
        """,
        (batch_size,),
    )
    community_ids = [r[0] for r in cur.fetchall()]

    upgraded_count = 0
    for cid in community_ids:
        title, source = name_community(conn, cid)
        if source in ("llm_gemini", "llm_ollama"):
            cur.execute(
                """
                UPDATE expression_communities
                SET name = ?,
                    name_source = ?,
                    naming_attempted_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                    naming_failure_count = 0,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                WHERE community_id = ?
                """,
                (title, source, cid),
            )
            conn.commit()
            upgraded_count += 1
            time.sleep(1.0)
        else:
            # LLM naming failed; record attempt timestamp and increment failure count
            cur.execute(
                """
                UPDATE expression_communities
                SET naming_attempted_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                    naming_failure_count = naming_failure_count + 1,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                WHERE community_id = ?
                """,
                (cid,),
            )
            conn.commit()

    return upgraded_count


def run_naming_sweep(conn: Optional[sqlite3.Connection] = None, batch_size: int = 5) -> int:
    """Scan and upgrade active communities marked 'pending' or 'fallback'.

    If conn is None, opens a scoped SQLite connection to the configured database.
    Returns the number of communities successfully upgraded to an LLM title.
    """
    if conn is None:
        from audiobench.core.settings import get_settings
        db_url = get_settings().database_url
        db_path = db_url.replace("sqlite:///", "") if db_url.startswith("sqlite:///") else "data/transcriptions.db"
        with sqlite3.connect(db_path) as scoped_conn:
            return _execute_naming_sweep(scoped_conn, batch_size=batch_size)
    else:
        return _execute_naming_sweep(conn, batch_size=batch_size)

