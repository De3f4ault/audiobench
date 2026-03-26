"""Two-model comparison engine.

Runs two models side-by-side via Rich Layout + Live, streaming responses
into parallel panels. Each model has its own thinking/content accumulator.

Usage:
    from audiobench.chat.compare import ModelComparison

    comparison = ModelComparison(client, messages, model_a, model_b)
    result = comparison.run()
    # result = {"model_a": {"content": ..., "thinking": ...},
    #           "model_b": {"content": ..., "thinking": ...}}
"""

from __future__ import annotations

import threading
import time

from audiobench.core.logger_factory import get_logger

logger = get_logger(__name__)


class ModelComparison:
    """Run two models side-by-side and stream results into a split Rich layout."""

    def __init__(
        self,
        client,
        messages: list[dict],
        model_a: str,
        model_b: str,
        temperature: float = 0.3,
        show_thinking: bool = True,
    ):
        self._client = client
        self._messages = messages
        self._model_a = model_a
        self._model_b = model_b
        self._temperature = temperature
        self._show_thinking = show_thinking

        # Per-model accumulators
        self._thinking_a: list[str] = []
        self._thinking_b: list[str] = []
        self._output_a: list[str] = []
        self._output_b: list[str] = []
        self._done_a = False
        self._done_b = False
        self._tokens_a = 0
        self._tokens_b = 0

    def _stream_model(
        self,
        model: str,
        output: list[str],
        thinking: list[str],
        token_counter: str,
        done_flag: str,
    ) -> None:
        """Stream a single model's response into the accumulators."""
        try:
            for chunk in self._client.chat_stream(
                messages=self._messages,
                model=model,
                temperature=self._temperature,
            ):
                think_text = chunk.get("thinking", "")
                content = chunk.get("content", "")

                if think_text:
                    thinking.append(think_text)
                if content:
                    output.append(content)
                    if token_counter == "a":
                        self._tokens_a += 1
                    else:
                        self._tokens_b += 1

                if chunk.get("done"):
                    break
        except Exception as e:
            output.append(f"\n\n[Error: {e}]")
            logger.warning("Comparison stream error for %s: %s", model, e)
        finally:
            setattr(self, done_flag, True)

    def run(self) -> dict:
        """Run the comparison with a live split-screen display.

        Returns:
            Dict with model results for persistence.
        """
        from rich.layout import Layout
        from rich.live import Live

        from audiobench.cli.display.theme import chat_console

        layout = Layout()
        layout.split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        t_start = time.monotonic()

        thread_a = threading.Thread(
            target=self._stream_model,
            args=(self._model_a, self._output_a, self._thinking_a, "a", "_done_a"),
            daemon=True,
        )
        thread_b = threading.Thread(
            target=self._stream_model,
            args=(self._model_b, self._output_b, self._thinking_b, "b", "_done_b"),
            daemon=True,
        )

        try:
            with Live(layout, console=chat_console, refresh_per_second=8) as live:
                thread_a.start()
                thread_b.start()

                while not (self._done_a and self._done_b):
                    self._update_layout(layout, t_start, streaming=True)
                    time.sleep(0.05)

                # Final render without cursor
                self._update_layout(layout, t_start, streaming=False)

                thread_a.join(timeout=2)
                thread_b.join(timeout=2)

        except KeyboardInterrupt:
            logger.info("Comparison interrupted by user")

        elapsed = time.monotonic() - t_start

        return {
            "model_a": {
                "model_name": self._model_a,
                "content": "".join(self._output_a),
                "thinking": "".join(self._thinking_a) or None,
                "tokens": self._tokens_a,
            },
            "model_b": {
                "model_name": self._model_b,
                "content": "".join(self._output_b),
                "thinking": "".join(self._thinking_b) or None,
                "tokens": self._tokens_b,
            },
            "elapsed": elapsed,
        }

    def _update_layout(
        self, layout, t_start: float, streaming: bool = True
    ) -> None:
        """Update both panels of the split layout."""
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text

        cursor = "▌" if streaming else ""

        for side, model, output, thinking, tokens in [
            ("left", self._model_a, self._output_a, self._thinking_a, self._tokens_a),
            ("right", self._model_b, self._output_b, self._thinking_b, self._tokens_b),
        ]:
            parts: list = []

            # Thinking section
            if thinking and self._show_thinking:
                think_text = "".join(thinking)
                think_lines = think_text.splitlines()
                if len(think_lines) > 5:
                    think_text = "…\n" + "\n".join(think_lines[-5:])
                parts.append(Text(f"💭 {think_text}", style="dim italic"))

            # Content section
            content = "".join(output) + cursor
            content_lines = content.splitlines()
            if len(content_lines) > 12:
                content = "⋮\n" + "\n".join(content_lines[-12:])
            parts.append(Text(content))

            # Stats
            elapsed = time.monotonic() - t_start
            tps = tokens / elapsed if elapsed > 0 else 0
            parts.append(
                Text(f"\n{tokens} tok · {tps:.0f} tok/s", style="dim")
            )

            border = "cyan" if side == "left" else "magenta"
            if not streaming:
                border = "green"

            layout[side].update(
                Panel(Group(*parts), title=model, border_style=border)
            )
