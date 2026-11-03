"""Ollama REST client for local LLM inference.

Communicates with a running Ollama server via its HTTP API
at localhost:11434. No external cloud services. All inference
happens on the same machine.

Usage:
    from audiobench.chat.providers.ollama_provider import OllamaClient

    client = OllamaClient()
    if client.is_available():
        response = client.generate("Summarize this text: ...")
        for token in client.stream("Explain..."):
            print(token, end="", flush=True)
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from audiobench.core.error_types import AIError, AudioBenchError
from audiobench.core.logger_factory import get_logger

logger = get_logger("ai.ollama")


# Regex to extract <think>...</think> blocks from content (fallback for non-native thinking)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class OllamaClient:
    """REST client for local Ollama server.

    Requires Ollama running: https://ollama.com
    Start with: ollama serve
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gpt-oss:120b-cloud",
        timeout: tuple[int, int] = (10, 600),
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        # Streaming reads may take minutes for large-context prompts
        # (connect_timeout, read_timeout)
        self._stream_timeout = (10, 600)

    @staticmethod
    def _handle_http_error(e: Any, model_name: str) -> AIError:
        """Format helpful error message from requests.HTTPError."""
        err_msg = str(e)
        status_code = getattr(e.response, "status_code", None)
        try:
            err_data = e.response.json()
            if isinstance(err_data, dict) and "error" in err_data:
                err_msg = err_data["error"]
        except Exception:
            pass

        if status_code == 404:
            return AIError(
                f"Model '{model_name}' not available on Ollama server",
                f"{err_msg}\n"
                f"Run /model to switch to an available model, or run 'ollama pull {model_name}'.",
            )
        return AIError(f"Ollama API error ({status_code or 'unknown'})", err_msg)

    def _ensure_requests(self):
        """Lazy import of requests."""
        try:
            import requests

            return requests
        except ImportError:
            raise AIError(
                "requests not installed",
                "Install with: pip install requests\nOr: pip install -e '.[ai]'",
            ) from None

    def is_available(self) -> bool:
        """Check if the Ollama server is running and reachable."""
        requests = self._ensure_requests()
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def list_models(self) -> list[str]:
        """List models available on the Ollama server.

        Returns:
            List of model name strings.

        Raises:
            AIError: If server is unreachable.
        """
        requests = self._ensure_requests()
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except requests.ConnectionError:
            raise AIError(
                "Ollama server not running",
                "Start with: ollama serve",
            ) from None
        except Exception as e:
            raise AIError("Failed to list models", str(e)) from e

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.3,
    ) -> str:
        """Generate a complete response (non-streaming).

        Args:
            prompt: The user prompt.
            model: Override the default model.
            system_prompt: Optional system instructions.
            temperature: Creativity (0.0=deterministic, 1.0=creative).

        Returns:
            Complete response text.

        Raises:
            AIError: If generation fails.
        """
        requests = self._ensure_requests()
        model_name = model or self._model

        payload: dict = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt

        logger.info(
            "Generating with %s (%.0f°C, %d char prompt)", model_name, temperature, len(prompt)
        )

        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                raise AIError(f"Model error from {model_name}", data["error"])

            response_text = data.get("response", "")
            logger.info(
                "Generated %d chars in %.1fs",
                len(response_text),
                data.get("total_duration", 0) / 1e9,
            )
            return response_text

        except requests.ConnectionError:
            raise AIError(
                "Ollama server not running",
                f"Start Ollama with: ollama serve\nThen pull the model: ollama pull {model_name}",
            ) from None
        except requests.HTTPError as e:
            raise self._handle_http_error(e, model_name) from e
        except Exception as e:
            raise AIError("Generation failed", str(e)) from e

    def stream(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        """Stream response tokens one at a time.

        Args:
            prompt: The user prompt.
            model: Override the default model.
            system_prompt: Optional system instructions.
            temperature: Creativity level.

        Yields:
            Individual response tokens/chunks as strings.

        Raises:
            AIError: If streaming fails.
        """
        requests = self._ensure_requests()
        model_name = model or self._model

        payload: dict = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt

        logger.info("Streaming with %s", model_name)

        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self._timeout,
            )
            resp.raise_for_status()

            import json

            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    # Ollama returns errors as {"error": "..."} with HTTP 200 during streaming.
                    if "error" in data:
                        raise AIError(f"Model error from {model_name}", data["error"])
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break

        except requests.ConnectionError:
            raise AIError("Ollama server not running", "Start with: ollama serve") from None
        except requests.HTTPError as e:
            raise self._handle_http_error(e, model_name) from e
        except AIError:
            raise
        except Exception as e:
            raise AIError("Streaming failed", str(e)) from e

    @staticmethod
    def _extract_thinking(content: str) -> tuple[str, str | None]:
        """Extract <think> blocks from content if present (fallback for non-native thinking)."""
        match = _THINK_RE.search(content)
        if match:
            thinking = match.group(1).strip()
            clean = _THINK_RE.sub("", content).strip()
            return clean, thinking
        return content, None

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        num_ctx: int | None = None,
        think: bool = True,
    ) -> dict:
        """Non-streaming chat via /api/chat.

        Args:
            messages: List of {"role": str, "content": str} message dicts.
            model: Override the default model.
            temperature: Creativity level.
            num_ctx: Context window size (tokens). None = model default.
            think: Enable model thinking/chain-of-thought separation.

        Returns:
            Dict with "content" and optional "thinking" keys.

        Raises:
            AIError: If chat fails.
        """
        requests = self._ensure_requests()
        model_name = model or self._model

        payload: dict = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        # Only inject 'think' when True — cloud-routed endpoints reject the field
        # entirely (even as False) and return 400.
        if think:
            payload["think"] = True
        if num_ctx:
            payload["options"]["num_ctx"] = num_ctx

        logger.info("Chat with %s (%d messages)", model_name, len(messages))

        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                raise AIError(f"Model error from {model_name}", data["error"])

            message = data.get("message", {})
            content = message.get("content", "")
            thinking = message.get("thinking")

            # Fallback: extract <think> tags from content if no native thinking
            if not thinking and content:
                content, thinking = self._extract_thinking(content)

            return {
                "content": content,
                "thinking": thinking,
            }

        except requests.ConnectionError:
            raise AIError(
                "Ollama server not running",
                f"Start with: ollama serve\nPull model: ollama pull {model_name}",
            ) from None
        except requests.HTTPError as e:
            raise self._handle_http_error(e, model_name) from e
        except Exception as e:
            raise AIError("Chat failed", str(e)) from e

    def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        num_ctx: int | None = None,
        think: bool = True,
    ) -> Iterator[dict]:
        """Stream chat response tokens via /api/chat.

        Args:
            messages: List of {"role": str, "content": str} dicts.
            model: Override the default model.
            temperature: Creativity level.
            num_ctx: Context window size (tokens).
            think: Enable model thinking/chain-of-thought separation.

        Yields:
            Dicts with "content" and/or "thinking" keys per chunk.
            Final chunk has "done": True.

        Raises:
            AIError: If streaming fails.
        """
        requests = self._ensure_requests()
        model_name = model or self._model

        payload: dict = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }
        # Only inject 'think' when True — cloud-routed endpoints reject the field
        # entirely (even as False) and return 400.
        if think:
            payload["think"] = True
        if num_ctx:
            payload["options"]["num_ctx"] = num_ctx

        logger.info("Streaming chat with %s (%d messages)", model_name, len(messages))

        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self._stream_timeout,
            )
            resp.raise_for_status()

            import json

            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    # Ollama returns errors as {"error": "..."} with HTTP 200
                    # during streaming — raise_for_status() does not catch these.
                    if "error" in data:
                        raise AIError(
                            f"Model error from {model_name}",
                            data["error"],
                        )
                    message = data.get("message", {})
                    chunk = {
                        "content": message.get("content", ""),
                        "thinking": message.get("thinking", ""),
                        "done": data.get("done", False),
                    }
                    yield chunk
                    if data.get("done", False):
                        break

        except requests.ConnectionError:
            raise AIError("Ollama server not running", "Start with: ollama serve") from None
        except requests.HTTPError as e:
            raise self._handle_http_error(e, model_name) from e
        except AIError:
            raise
        except Exception as e:
            raise AIError("Chat streaming failed", str(e)) from e
