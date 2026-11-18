from pathlib import Path


def _resolve_safe_dir(path_str: str) -> Path:
    """Resolve a directory path with a deterministic fallback chain.

    The contract this function upholds: it always returns a Path that
    exists, is a directory, and is readable. It never raises.

    Fallback chain:
        1. requested path (expanded, resolved)
        2. user home directory
        3. current working directory (guaranteed to exist by the OS)
    """
    for candidate in (
        Path(path_str).expanduser().resolve(),
        Path.home(),
        Path.cwd(),
    ):
        try:
            if candidate.exists() and candidate.is_dir():
                # Confirm readability — is_dir() returning True is not
                # enough on restricted paths (e.g. root-owned directories).
                list(candidate.iterdir())
                return candidate
        except OSError:
            continue
    # Should never be reached; cwd() is always readable.
    return Path.cwd()


class DirectoryNavigator:
    def __init__(self, start_path: str = "~", allowed_extensions: set[str] | None = None):
        self.current_path = _resolve_safe_dir(start_path)
        self.selected = 0
        self.show_hidden = False
        self.selected_files: set[Path] = set()
        self.allowed_extensions = allowed_extensions

        # Rendering performance cache: prevents repeated disk iterdir() + sort
        # calls on every TUI redraw frame (e.g. 10Hz curses timeout loop).
        self._cache_items: list[str] | None = None
        self._cache_key: tuple[Path, bool, tuple[str, ...]] | None = None

    def _invalidate_cache(self) -> None:
        self._cache_items = None
        self._cache_key = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_to_safe_path(self) -> None:
        """Walk up from current_path until we reach a readable directory.

        Called automatically by list_items() when the current path is no
        longer accessible (deleted, unmounted, permissions revoked, etc.).
        Resets selection to 0 after relocating.
        """
        self._invalidate_cache()
        p = self.current_path
        # Guard: at most 64 steps so we can't loop forever even on a
        # pathological symlink cycle or unusual filesystem layout.
        for _ in range(64):
            try:
                if p.exists() and p.is_dir():
                    list(p.iterdir())   # readability check
                    self.current_path = p
                    self.selected = 0
                    return
            except OSError:
                pass
            parent = p.parent
            if parent == p:
                break  # reached filesystem root without success
            p = parent

        # Final safe fallback — home is guaranteed resolvable
        self.current_path = Path.home()
        self.selected = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def toggle_hidden_visibility(self) -> None:
        """Toggle visibility of hidden files and directories."""
        self.show_hidden = not self.show_hidden
        self.selected = 0
        self._invalidate_cache()

    def list_items(self) -> list[str]:
        """Return names of directories and filtered files under current_path.

        Directories are listed before files; both groups are sorted
        case-insensitively. Cached per (current_path, show_hidden, filter_exts)
        state to make TUI rendering 60FPS smooth.

        If the current path is not accessible for any reason
        (FileNotFoundError, PermissionError, unmounted drive, etc.) the
        navigator self-heals to the nearest readable ancestor before
        returning an empty list. The TUI calling this method is therefore
        guaranteed never to receive an unhandled exception from here.
        """
        from audiobench.transcribe.audio_converter import ALL_SUPPORTED_FORMATS

        filter_exts = (
            self.allowed_extensions
            if self.allowed_extensions is not None
            else ALL_SUPPORTED_FORMATS
        )
        ext_key = tuple(sorted(filter_exts))
        cache_key = (self.current_path, self.show_hidden, ext_key)

        if self._cache_items is not None and self._cache_key == cache_key:
            return self._cache_items

        try:
            all_items: list[Path] = []
            for item in Path(self.current_path).iterdir():
                if not self.show_hidden and item.name.startswith("."):
                    continue
                if item.is_file() and item.suffix.lstrip(".").lower() not in filter_exts:
                    continue
                all_items.append(item)

            sorted_items = sorted(
                all_items, key=lambda x: (not x.is_dir(), x.name.lower())
            )
            result = [item.name for item in sorted_items]

            self._cache_items = result
            self._cache_key = cache_key
            return result

        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
            self._reset_to_safe_path()
            return []

    def get_current_item_path(self) -> Path | None:
        """Return the full Path of the currently selected item, or None."""
        items = self.list_items()
        if items and 0 <= self.selected < len(items):
            return self.current_path / items[self.selected]
        return None

    def enter(self) -> None:
        """Navigate into the selected directory (no-op for files)."""
        items = self.list_items()
        if not items:
            return
        selected_path = self.current_path / items[self.selected]
        if selected_path.is_dir():
            self.current_path = selected_path
            self.selected = 0
            self._invalidate_cache()

    def go_up(self) -> None:
        """Navigate to the parent directory.

        Navigation is not bounded — the user can walk up through /home,
        /run/media, / etc., making mounted external drives reachable via
        ordinary up-navigation without any special handling.
        """
        parent = self.current_path.parent
        if parent != self.current_path:
            self.current_path = parent
            self.selected = 0
            self._invalidate_cache()

    def select_next(self) -> None:
        """Move selection down (wraps around)."""
        items = self.list_items()
        if items:
            self.selected = (self.selected + 1) % len(items)

    def select_prev(self) -> None:
        """Move selection up (wraps around)."""
        items = self.list_items()
        if items:
            self.selected = (self.selected - 1) % len(items)

    def toggle_selection(self) -> None:
        """Toggle file selection state (spacebar multi-select)."""
        current = self.get_current_item_path()
        if current and current.is_file():
            if current in self.selected_files:
                self.selected_files.remove(current)
            else:
                self.selected_files.add(current)

    def get_item_count_info(self) -> dict[str, int]:
        """Return total, visible, and hidden item counts for the status bar."""
        try:
            all_items = list(Path(self.current_path).iterdir())
            total_items = len(all_items)
            visible_items = len(self.list_items())
            hidden_items = total_items - visible_items
            return {"total": total_items, "visible": visible_items, "hidden": hidden_items}
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
            return {"total": 0, "visible": 0, "hidden": 0}
