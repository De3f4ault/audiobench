"""Query reformulation using LLMs for multi-strategy retrieval."""

import json

from audiobench.chat.providers.ollama_provider import OllamaClient
from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings
from audiobench.memory.query_engine import Err, Ok, _call_llm, _call_llm_for_model
from audiobench.memory.query_types import ReformulatedQuery

# Re-export so callers can still do: from audiobench.memory.query_reformulator import ReformulatedQuery
__all__ = ["QueryReformulator", "ReformulatedQuery"]

logger = get_logger("memory.query_reformulator")


class QueryReformulator:
    """Uses an LLM to expand a simple query into retrieval-optimized forms."""

    def __init__(self, model: str | None = None) -> None:
        self.settings = get_settings()
        self.model = model
        self.llm = OllamaClient(
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_model,
        )
        self.api_key = self.settings.gemini_api_key

    def reformulate(self, query: str) -> ReformulatedQuery:
        """Reformulate the given query."""
        prompt = (
            "You are an AI search assistant. The user provided a query. "
            "Please reformulate it into three components for optimal retrieval.\n\n"
            "Respond ONLY with valid JSON exactly matching this structure:\n"
            "{\n"
            '  "bm25_keywords": "Space-separated keywords for exact text matching",\n'
            '  "semantic_query": "A verbose natural language explanation of the semantic intent",\n'
            '  "hyde_anchor": "A hypothetical 80-word passage that perfectly answers the query. Write it as if it were an actual excerpt from a relevant document."\n'
            "}\n\n"
            f"User Query: {query}\n"
        )

        result = _call_llm_for_model(prompt, temperature=0.2, model=self.model, llm=self.llm, api_key=self.api_key)

        match result:
            case Ok(value=text):
                try:
                    # Clean up common LLM formatting (markdown blocks)
                    text = text.strip()
                    if text.startswith("```json"):
                        text = text[7:]
                    elif text.startswith("```"):
                        text = text[3:]
                    if text.endswith("```"):
                        text = text[:-3]

                    data = json.loads(text.strip())
                    return ReformulatedQuery(
                        original=query,
                        bm25_keywords=str(data.get("bm25_keywords", query)),
                        semantic_query=str(data.get("semantic_query", query)),
                        hyde_anchor=str(data.get("hyde_anchor", query)),
                        # Dense channel always receives the original query — independent of
                        # BM25 reformulation. Nomic handles full natural language natively.
                        dense_query=query,
                    )
                except Exception as ex:
                    logger.warning("Failed to parse LLM reformulation JSON: %s. Using fallback.", ex)

            case Err(error=reason):
                logger.warning("LLM reformulation failed: %s. Using fallback.", reason)

        # Graceful degradation fallback
        return ReformulatedQuery(
            original=query,
            bm25_keywords=query,
            semantic_query=query,
            hyde_anchor=f"Hypothetical answer to the query: {query}",
            dense_query=query,
        )
