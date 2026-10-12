from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import List, Tuple

from audiobench.core.settings import get_settings

logger = logging.getLogger("audiobench.daemon.intelligence.gap_naming")


def _clean_title(text: str) -> str:
    cleaned = text.strip().strip("\"'").strip()
    cleaned = re.sub(r"[*_`]", "", cleaned)
    cleaned = re.sub(r"^(Title|Topic|Topics|Gap|Gaps|Category|Categories):\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(Gap|Gaps|Topic|Topics|Category|Categories)$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def name_knowledge_gap_with_gemini(
    queries: List[str],
    medoid_query: str,
) -> Tuple[str, str, str] | None:
    """Query Gemini API to name the knowledge gap and suggest acquisitions."""
    try:
        settings = get_settings()
        api_key = getattr(settings, "gemini_api_key", None) or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None

        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=10_000),
        )

        query_samples = "\n".join(f"- {q}" for q in queries[:15])
        prompt = (
            "You are a library curator analyzing search queries from a user whose audio library lacks coverage for their interests.\n\n"
            f"Representative Search Query (Medoid): \"{medoid_query}\"\n"
            f"Recent Search Queries:\n{query_samples}\n\n"
            "Format your response as strict JSON with exactly three keys:\n"
            "1. \"title\": A concise 3-5 word conceptual name for this subject area. Do not use generic meta-words like 'Gap', 'Missing', or 'Topic'; name the concrete subject matters.\n"
            "2. \"diagnostic\": A 1-2 sentence explanation of what specific topic cluster is missing.\n"
            "3. \"recommended_ingestion\": Top 5-10 specific books, authors, or audiobook titles to ingest to close this gap.\n"
        )

        model_name = getattr(settings, "gemini_model", "gemini-2.5-flash")
        try:
            config = types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1000,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        except Exception:
            try:
                config = types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1000,
                    response_mime_type="application/json",
                )
            except Exception:
                config = types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1000,
                )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        if response and response.text:
            text = response.text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```[a-z]*\n", "", text)
                text = re.sub(r"\n```$", "", text).strip()
            data = json.loads(text)
            title = _clean_title(str(data.get("title", "")))
            raw_diag = data.get("diagnostic", "")
            diagnostic = " ".join(str(d).strip() for d in raw_diag) if isinstance(raw_diag, list) else str(raw_diag).strip()
            raw_rec = data.get("recommended_ingestion", "")
            rec = ", ".join(str(r).strip() for r in raw_rec) if isinstance(raw_rec, list) else str(raw_rec).strip()
            if title:
                return title, diagnostic, rec
    except Exception as exc:
        msg = str(exc)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            logger.warning(
                "Gemini gap naming hit quota exhaustion (429 RESOURCE_EXHAUSTED): %s. Triggering fallback to Tier 2 (Ollama).",
                exc,
            )
        else:
            logger.debug("Gemini gap naming failed: %s", exc)

    return None


def name_knowledge_gap_with_ollama(
    queries: List[str],
    medoid_query: str,
) -> Tuple[str, str, str] | None:
    """Query local Ollama daemon to name the knowledge gap and suggest acquisitions."""
    try:
        settings = get_settings()
        base_url = getattr(settings, "ollama_base_url", "http://localhost:11434").rstrip("/")
        model_name = getattr(settings, "ollama_model", "gpt-oss:120b-cloud")

        query_samples = "\n".join(f"- {q}" for q in queries[:15])
        prompt = (
            "You are a library curator analyzing search queries from a user whose audio library lacks coverage for their interests.\n\n"
            f"Representative Search Query (Medoid): \"{medoid_query}\"\n"
            f"Recent Search Queries:\n{query_samples}\n\n"
            "Format your response as strict JSON with exactly three keys:\n"
            "1. \"title\": A concise 3-5 word conceptual name for this subject area. Do not use generic meta-words like 'Gap', 'Missing', or 'Topic'; name the concrete subject matters.\n"
            "2. \"diagnostic\": A 1-2 sentence explanation of what specific topic cluster is missing.\n"
            "3. \"recommended_ingestion\": Top 5-10 specific books, authors, or audiobook titles to ingest to close this gap.\n"
            "Return ONLY valid JSON."
        )

        payload = {
            "model": model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 2048},
        }

        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw = data.get("response", "")
            if raw:
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = {}
                    title_m = re.search(r'"title":\s*"([^"]+)"', raw)
                    if title_m:
                        parsed["title"] = title_m.group(1)
                    diag_m = re.search(r'"diagnostic":\s*"([^"]+)"', raw)
                    if diag_m:
                        parsed["diagnostic"] = diag_m.group(1)
                    rec_m = re.findall(r'"([^"]{3,})"', raw[raw.find('"recommended_ingestion"'):] if '"recommended_ingestion"' in raw else "")
                    if rec_m:
                        parsed["recommended_ingestion"] = [r for r in rec_m if r != "recommended_ingestion"]

                title = _clean_title(str(parsed.get("title", "")))
                raw_diag = parsed.get("diagnostic", "")
                diagnostic = " ".join(str(d).strip() for d in raw_diag) if isinstance(raw_diag, list) else str(raw_diag).strip()
                raw_rec = parsed.get("recommended_ingestion", "")
                rec = ", ".join(str(r).strip() for r in raw_rec) if isinstance(raw_rec, list) else str(raw_rec).strip()
                if title:
                    return title, diagnostic, rec
    except Exception as exc:
        logger.debug("Ollama gap naming failed: %s", exc)

    return None


def name_knowledge_gap(
    queries: List[str],
    medoid_query: str,
) -> Tuple[str, str, str, str]:
    """Execute the multi-tier naming fallback chain for a knowledge gap.

    Returns:
        (title, diagnostic, recommended_ingestion, name_source)
        where name_source in ('llm_gemini', 'llm_ollama', 'vector_medoid').
    """
    # Tier 1: Gemini
    gemini_res = name_knowledge_gap_with_gemini(queries, medoid_query)
    if gemini_res:
        title, diagnostic, rec = gemini_res
        return title, diagnostic, rec, "llm_gemini"

    # Tier 2: Ollama
    ollama_res = name_knowledge_gap_with_ollama(queries, medoid_query)
    if ollama_res:
        title, diagnostic, rec = ollama_res
        return title, diagnostic, rec, "llm_ollama"

    # Tier 3: Vector Medoid Fallback
    fallback_title = f"Knowledge Gap: {medoid_query[:40]}" if medoid_query else "Under-Covered Research Topic"
    fallback_diag = (
        f"Your search queries cluster around topics (such as '{medoid_query}') "
        f"not well covered by your current audio library."
    )
    fallback_rec = "Consider ingesting audiobooks or lectures matching your recent search terms."
    return fallback_title, fallback_diag, fallback_rec, "vector_medoid"


def get_queries_in_window(
    conn,
    proposal_ts: str,
    window_days: int = 14,
    limit: int = 100,
) -> List[str]:
    """Retrieve search queries within a historical lookback window prior to a proposal.

    Guarantees:
    - Queries after proposal_ts are excluded.
    - Queries older than (proposal_ts - window_days) are excluded.
    - Only source_type = 'search_query' is selected.
    - Empty strings and slash commands are stripped.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT content FROM expressions
        WHERE source_type = 'search_query'
          AND created_at BETWEEN datetime(?, '-' || ? || ' days') AND datetime(?)
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (proposal_ts, window_days, proposal_ts, limit),
    )
    rows = cur.fetchall()
    queries = []
    for r in rows:
        val = (r[0] or "").strip()
        if val and not val.startswith(("\\", "/")):
            queries.append(val)
    return queries

