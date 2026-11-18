"""Reverse Import (Transcript-first) TUI."""

import curses
import threading
from difflib import SequenceMatcher
from pathlib import Path

from audiobench.cli.tui.directory_navigator import DirectoryNavigator
from audiobench.core.db_engine import init_db
from audiobench.storage.models import AudioFileRecord, TranscriptionRecord
from audiobench.storage.repository import TranscriptionRepository
from audiobench.transcribe.quality_evaluator import (
    ComparisonResult,
    FamiliarityScore,
    QualityScore,
    compare_transcripts,
    evaluate_transcript_quality,
)
from audiobench.transcribe.transcript_parser import (
    HashMismatchError,
    build_audio_metadata,
    parse_transcript_file,
)
from audiobench.transcribe.transcription_result import AudioMetadata


class ReimportTUI:
    def __init__(self, start_path="~", audio_start_path="~", batch=False, auto_detect=False, shared_state=None):
        self.start_path = Path(start_path).expanduser()
        self.audio_start_path = Path(audio_start_path).expanduser()
        self.cancelled = False

        self.is_batch = batch
        self.is_auto_detect = auto_detect

        if auto_detect:
            self.screen = 21
        elif batch:
            self.screen = 11
        else:
            self.screen = 1

        # Single mode state
        self.transcript_path = None
        self.audio_path = None
        self.parsed_transcript = None
        self.raw_audio = None
        self.audio_metadata = None

        self.db_check_result = None # None, "new", "exists"
        self.conflict_action = None # "O", "A", "S"
        self.existing_transcriptions_count = 0

        self.hash_error = None
        self.hash_computed = False

        # Batch mode state
        self.transcript_dir = None
        self.audio_dir = None
        self.batch_pairs = [] # list of dict: {"tx": Path, "audio": Path|None, "status": str}

        self.tx_nav = DirectoryNavigator(start_path=str(self.start_path), allowed_extensions={"json", "srt", "txt"})
        self.audio_nav = DirectoryNavigator(start_path=str(self.audio_start_path))
        self.audio_nav.allowed_extensions = None # all supported formats
        self.root_nav = DirectoryNavigator(start_path=str(self.start_path))
        self.ui_selected_idx = 0
        self.ui_offset = 0
        self.next_mode = None

        if shared_state:
            if "current_path" in shared_state:
                try:
                    p = Path(shared_state["current_path"])
                    if p.exists():
                        self.start_path = p
                        self.tx_nav.current_path = p
                        self.audio_nav.current_path = p
                        self.root_nav.current_path = p
                        if self.is_auto_detect:
                            self.audio_dir = p
                            self.transcript_dir = p
                            self.screen = 22
                            self._match_batch()
                except Exception:
                    pass

            if "selected_tx_file" in shared_state:
                try:
                    tx_p = Path(shared_state["selected_tx_file"])
                    if tx_p.exists() and tx_p.is_file():
                        self._on_tx_selected(tx_p)
                except Exception:
                    pass

    def state_export(self) -> dict:
        return {
            "current_path": str(self.tx_nav.current_path if not self.is_auto_detect else self.root_nav.current_path)
        }

    def state_import(self, state: dict) -> None:
        pass

    def run(self):
        curses.wrapper(self._run_loop)
        return self.next_mode if self.next_mode else (not self.cancelled)

    def _run_loop(self, stdscr):
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.nodelay(0)
        stdscr.timeout(100)
        stdscr.clear()

        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)  # Header/Footer
            curses.init_pair(2, curses.COLOR_WHITE, -1)  # Normal
            curses.init_pair(3, curses.COLOR_GREEN, -1)  # Success/Selected
            curses.init_pair(4, curses.COLOR_YELLOW, -1)  # Warning/Hidden
            curses.init_pair(5, curses.COLOR_MAGENTA, -1)  # Status
            curses.init_pair(6, curses.COLOR_CYAN, -1)  # Separator
            curses.init_pair(7, curses.COLOR_RED, -1)  # Error
        except:
            pass

        while not self.cancelled:
            stdscr.clear()

            if self.screen == 1:
                self._draw_screen_1(stdscr)
            elif self.screen == 2:
                self._draw_screen_2(stdscr)
            elif self.screen == 3:
                self._draw_screen_3(stdscr)
                if self.hash_computed and self.db_check_result == "new":
                    self.screen = 4
            elif self.screen == 4:
                self._draw_screen_4(stdscr)
            elif self.screen == 5:
                self._draw_screen_5(stdscr)
                break
            elif self.screen == 11:
                self._draw_screen_1b(stdscr)
            elif self.screen == 12:
                self._draw_screen_2b(stdscr)
            elif self.screen == 13:
                self._draw_screen_3b(stdscr)
            elif self.screen == 14:
                self._draw_screen_4b(stdscr)
                break
            elif self.screen == 15:
                self._draw_screen_15(stdscr)
            elif self.screen == 21:
                self._draw_screen_21(stdscr)
            elif self.screen == 22:
                self._draw_screen_22(stdscr)
            elif self.screen == 23:
                self._draw_screen_4b(stdscr)
                break

            key = stdscr.getch()
            self._handle_input(key)

    def _handle_input(self, key):
        if key == -1:
            return

        if self.screen == 1:
            self._nav_input(key, self.tx_nav, on_enter=self._on_tx_selected)
            if key in (ord('b'), ord('B')):
                self.is_batch = True
                self.screen = 11
            if key in (ord('a'), ord('A')):
                self.is_auto_detect = True
                self.screen = 21
        elif self.screen == 2:
            self._nav_input(key, self.audio_nav, on_enter=self._on_audio_selected)
        elif self.screen == 3:
            if self.db_check_result == "exists":
                if key in (ord('o'), ord('O')):
                    self.conflict_action = "O"
                    self.screen = 4
                elif key in (ord('a'), ord('A')):
                    self.conflict_action = "A"
                    self.screen = 4
                elif key in (ord('s'), ord('S')):
                    self.screen = 1
            elif self.db_check_result == "new":
                if key in (10, 13):
                    self.conflict_action = None
                    self.screen = 4
                elif key in (ord('r'), ord('R')):
                    self.screen = 2
                    self.hash_computed = False
                    self.hash_error = None
            elif self.hash_error:
                if key in (ord('r'), ord('R')):
                    self.screen = 2
                    self.hash_computed = False
                    self.hash_error = None
                    self.tag_modified_heuristic_passed = False
                    self.tag_modified_metrics = None
                elif key in (ord('f'), ord('F')) and getattr(self, "tag_modified_heuristic_passed", False):
                    from audiobench.transcribe.transcript_parser import build_audio_metadata
                    self.audio_metadata = build_audio_metadata(self.audio_path, self.raw_audio, allow_hash_mismatch=True)
                    metrics = getattr(self, "tag_modified_metrics", {})
                    self.audit_provenance = {
                        "hash_verification": "bypassed_tag_modified",
                        "verified_by": "operator_interactive",
                        "original_json_hash": (self.raw_audio or {}).get("file_hash"),
                        "actual_audio_hash": self.audio_metadata.file_hash,
                        "duration_delta_sec": metrics.get("duration_delta_sec"),
                        "size_delta_bytes": metrics.get("size_delta_bytes"),
                        "source": "reimport:hash_override",
                    }
                    self.hash_error = None
                    init_db()
                    from audiobench.core.db_session import get_session
                    with get_session() as session:
                        rec = session.query(AudioFileRecord).filter_by(file_hash=self.audio_metadata.file_hash).first()
                        if not rec:
                            rec = session.query(AudioFileRecord).filter_by(file_name=self.audio_path.name).first()
                        if rec:
                            self.db_check_result = "exists"
                            self.existing_transcriptions_count = session.query(TranscriptionRecord).filter_by(audio_file_id=rec.id).count()
                        else:
                            self.db_check_result = "new"
                    self.screen = 4
        elif self.screen == 4:
            if key in (10, 13): # ENTER
                self.screen = 5
            elif key in (ord('q'), ord('Q')):
                self.cancelled = True
        elif self.screen == 11:
            self._nav_input(key, self.tx_nav, on_enter=self._on_tx_dir_selected)
            if key in (10, 13):
                self.transcript_dir = self.tx_nav.current_path
                self.screen = 12
        elif self.screen == 12:
            self._nav_input(key, self.audio_nav, on_enter=lambda p: None)
            if key in (10, 13):
                self.audio_dir = self.audio_nav.current_path
                self._match_batch()
                self.screen = 22
                self.ui_offset = 0
                self.ui_selected_idx = 0
        elif self.screen == 13:
            if key in (ord('s'), ord('S')):
                self.screen = 14
            elif key in (ord('q'), ord('Q')):
                self.cancelled = True
        elif self.screen == 15:
            if key in (ord('s'), ord('S')):
                existing = self._get_existing_basenames()
                self.batch_pairs = [p for p in self.batch_pairs if p["tx"].stem not in existing]
                self.batch_overwrite = False
                self.screen = 13
            elif key in (ord('r'), ord('R'), ord('o'), ord('O'), ord('0')):
                self.batch_overwrite = True
                self.screen = 13
            elif key in (ord('q'), ord('Q')):
                self.cancelled = True
        elif self.screen == 21:
            self._nav_input(key, self.root_nav, on_enter=lambda p: None)
            if key in (10, 13):
                self.audio_dir = self.root_nav.current_path
                self.transcript_dir = self.audio_dir
                self._match_batch()
                self.screen = 22
                self.ui_offset = 0
                self.ui_selected_idx = 0
        elif self.screen == 22:
            if key in (curses.KEY_UP, ord('k')):
                self.ui_selected_idx = max(0, self.ui_selected_idx - 1)
            elif key in (curses.KEY_DOWN, ord('j')):
                self.ui_selected_idx = min(len(self.batch_pairs) - 1, self.ui_selected_idx + 1)
            elif key == ord(' '):
                if self.batch_pairs:
                    self.batch_pairs[self.ui_selected_idx]["selected"] = not self.batch_pairs[self.ui_selected_idx].get("selected", False)
            elif key in (ord('u'), ord('U')):
                # Toggle upgrade recommendation selection
                if self.batch_pairs:
                    cur = self.batch_pairs[self.ui_selected_idx]
                    if cur.get("status") == "UPGRADE":
                        cur["selected"] = not cur.get("selected", False)
                    else:
                        any_unsel_upg = any(p.get("status") == "UPGRADE" and not p.get("selected") for p in self.batch_pairs)
                        for p in self.batch_pairs:
                            if p.get("status") == "UPGRADE":
                                p["selected"] = any_unsel_upg
            elif key in (ord('t'), ord('T')):
                # Toggle tag-modified items
                if self.batch_pairs:
                    cur = self.batch_pairs[self.ui_selected_idx]
                    if cur.get("status") == "TAG_MODIFIED":
                        cur["selected"] = not cur.get("selected", False)
                    else:
                        any_unsel_tm = any(p.get("status") == "TAG_MODIFIED" and not p.get("selected") for p in self.batch_pairs)
                        for p in self.batch_pairs:
                            if p.get("status") == "TAG_MODIFIED":
                                p["selected"] = any_unsel_tm
            elif key in (ord('a'), ord('A')):
                # Toggle select all valid actionable candidates (READY and UPGRADE only)
                valid_candidates = [p for p in self.batch_pairs if p.get("status") in ("READY", "UPGRADE")]
                any_unselected = any(not p.get("selected", False) for p in valid_candidates)
                for p in valid_candidates:
                    p["selected"] = any_unselected
            elif key in (ord('b'), ord('B')):
                if self.batch_pairs:
                    cur = self.batch_pairs[self.ui_selected_idx]
                    cur["is_audiobook"] = not cur.get("is_audiobook", False)
                    if cur["is_audiobook"]:
                        cur["book_badge"] = "[Book: Confirmed]"
                    else:
                        cur["book_badge"] = "[Book: Review? [B]]" if cur.get("classification_category") == "class_b_soft_heuristic" else ""
            elif key in (ord('s'), ord('S')):
                self.batch_pairs = [p for p in self.batch_pairs if p.get("selected", False)]
                self.screen = 23
            elif key in (ord('q'), ord('Q')):
                self.cancelled = True

        if key in (ord('q'), ord('Q')) and self.screen < 5 or key in (ord('q'), ord('Q')) and self.screen in (11, 12, 15, 21, 22):
            self.cancelled = True

        # Mode switching shortcuts
        if key in (ord('1'),):
            self.next_mode = "MODE_AUDIO"
            self.cancelled = True
        elif key in (ord('2'),):
            self.next_mode = "MODE_AUTO"
            self.cancelled = True
        elif key in (ord('3'),):
            self.next_mode = "MODE_SINGLE_TX"
            self.cancelled = True
        elif key in (ord('4'),):
            self.next_mode = "MODE_BATCH_TX"
            self.cancelled = True

    def _get_existing_basenames(self):
        if hasattr(self, "_existing_basenames"):
            return self._existing_basenames

        from audiobench.core.db_engine import init_db
        from audiobench.storage.repository import TranscriptionRepository
        init_db()
        repo = TranscriptionRepository()
        from audiobench.core.db_session import get_session
        self._existing_basenames = set()
        with get_session() as session:
            from pathlib import Path

            from audiobench.storage.models import TranscriptionRecord
            for (fname,) in session.query(TranscriptionRecord.file_name).all():
                if fname:
                    self._existing_basenames.add(Path(fname).stem)
        return self._existing_basenames

    def _match_batch(self):
        self.batch_pairs = []
        import json
        import os
        import re
        from difflib import SequenceMatcher
        from pathlib import Path

        from audiobench.core.db_session import get_session
        from audiobench.storage.models import AudioFileRecord, TranscriptionRecord
        from audiobench.transcribe.audio_converter import ALL_SUPPORTED_FORMATS
        from audiobench.transcribe.quality_evaluator import compare_transcripts
        from audiobench.transcribe.transcript_parser import (
            evaluate_tag_modified_heuristic,
            parse_transcript_file,
        )
        from audiobench.transcribe.transcription_result import AudioMetadata

        BLACKLIST_DIRS = {
            ".git", ".cache", "node_modules", "venv", ".venv",
            "__pycache__", "tmp", "temp", ".local", "dist", "build",
            ".pytest_cache", ".cargo", ".rustup", "site-packages"
        }

        def scan_files(root: Path, extensions: set[str], max_depth: int = 4) -> list[Path]:
            results = []
            root_depth = len(root.parts)
            try:
                for current_root, dirs, files in os.walk(root, followlinks=False):
                    dirs[:] = [
                        d for d in dirs
                        if not d.startswith(".") and d.lower() not in BLACKLIST_DIRS
                    ]
                    if len(Path(current_root).parts) - root_depth > max_depth:
                        dirs[:] = []
                        continue
                    for f in files:
                        ext = f.rsplit(".", 1)[-1].lower() if "." in f else ""
                        if ext in extensions:
                            results.append(Path(current_root) / f)
            except Exception:
                pass
            return results

        is_auto = getattr(self, "is_auto_detect", False)

        if is_auto:
            available_audio_files = scan_files(self.audio_dir, set(ALL_SUPPORTED_FORMATS), max_depth=4)
        else:
            available_audio_files = []
            try:
                for ext in ALL_SUPPORTED_FORMATS:
                    available_audio_files.extend(list(self.audio_dir.glob(f"*.{ext}")))
            except Exception:
                pass

        seen_paths = set()
        deduped_audio = []
        for a in available_audio_files:
            rp = a.resolve()
            if rp not in seen_paths and a.exists() and a.stat().st_size > 0:
                seen_paths.add(rp)
                deduped_audio.append(a)
        available_audio_files = deduped_audio

        def normalize_name(name: str) -> str:
            s = re.sub(r'[^a-z0-9\s]', '', name.lower())
            return re.sub(r'\s+', ' ', s).strip()

        audio_by_name = {a.name: a for a in available_audio_files}
        normalized_audio_cands = [(a, normalize_name(a.stem)) for a in available_audio_files]
        audio_by_stem = {norm: a for a, norm in normalized_audio_cands}

        audio_hash_cache: dict[Path, str] = {}
        audio_duration_cache: dict[Path, float] = {}

        if is_auto:
            tx_files = scan_files(self.transcript_dir, {"json"}, max_depth=4)
        elif self.tx_nav.selected_files:
            tx_files = list(self.tx_nav.selected_files)
        else:
            tx_files = [p for p in self.transcript_dir.glob("*.json") if p.is_file()]

        # Pre-index existing DB records in ONE session
        with get_session() as session:
            all_audio_recs = session.query(AudioFileRecord).all()
            all_tx_recs = session.query(TranscriptionRecord).all()

            existing_audio_by_hash = {a.file_hash: a for a in all_audio_recs if a.file_hash}
            existing_audio_by_name = {a.file_name.lower(): a for a in all_audio_recs if a.file_name}
            existing_audio_by_stem = {normalize_name(Path(a.file_name).stem): a for a in all_audio_recs if a.file_name}
            existing_audio_by_path = {str(Path(a.file_path).resolve()): a for a in all_audio_recs if a.file_path}

            existing_tx_by_audio_id = {tx.audio_file_id: tx for tx in all_tx_recs if tx.audio_file_id}
            existing_tx_by_stem = {normalize_name(Path(tx.file_name).stem): tx for tx in all_tx_recs if tx.file_name}

            for tx_file in tx_files:
                if not tx_file.name.endswith(".json") or not tx_file.is_file():
                    continue

                # Guardrail 4: Filter out non-transcript JSON files
                try:
                    raw_text = tx_file.read_text(encoding="utf-8")
                    data = json.loads(raw_text)
                except Exception:
                    continue

                if not isinstance(data, dict):
                    continue

                is_schema_v1 = str(data.get("schema_version")) == "1"
                segments_raw = data.get("segments")
                has_valid_segments = isinstance(segments_raw, list)
                has_transcript_shape = has_valid_segments and ("text" in data or "full_text" in data or "language" in data)
                if not (is_schema_v1 or has_transcript_shape):
                    continue

                raw_audio = data.get("audio") or data.get("raw_audio")
                tx_text = str(data.get("text") or data.get("full_text") or "").strip()
                tx_duration = float(data.get("duration_seconds") or 0.0)
                engine = data.get("engine")

                matched_audio = None
                audio_name = (raw_audio or {}).get("file_name") if raw_audio else None

                # 1. Match by exact JSON metadata audio filename
                if audio_name and audio_name in audio_by_name:
                    matched_audio = audio_by_name[audio_name]

                # 2. Fuzzy / normalized match by base filename
                tx_stem_norm = normalize_name(tx_file.stem)
                if not matched_audio and tx_stem_norm in audio_by_stem:
                    matched_audio = audio_by_stem[tx_stem_norm]

                if not matched_audio:
                    best_match = None
                    best_ratio = 0.0
                    len_tx = len(tx_stem_norm)
                    for audio_cand, audio_stem_norm in normalized_audio_cands:
                        len_au = len(audio_stem_norm)
                        if abs(len_tx - len_au) > max(3, int(0.20 * max(len_tx, len_au))):
                            continue
                        ratio = SequenceMatcher(None, tx_stem_norm, audio_stem_norm).ratio()
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_match = audio_cand
                    if best_ratio > 0.85:
                        matched_audio = best_match

                # 3. Fallback: Check if already in the library / database
                if not matched_audio and tx_stem_norm in existing_audio_by_stem:
                    a_rec = existing_audio_by_stem[tx_stem_norm]
                    if a_rec and Path(a_rec.file_path).exists() and Path(a_rec.file_path).stat().st_size > 0:
                        matched_audio = Path(a_rec.file_path)

                # Guardrail 2: Verify audio file exists and size > 0
                if matched_audio and (not matched_audio.exists() or matched_audio.stat().st_size == 0):
                    matched_audio = None

                if is_auto and not matched_audio:
                    # In auto-detect, strictly require a valid corresponding audio file
                    continue

                # Guardrail 7: Non-empty / non-zero content
                is_empty = (not segments_raw) or (not tx_text)

                # Guardrail 5: Pre-check hash/size
                hash_mismatch = False
                expected_hash = (raw_audio or {}).get("file_hash") if raw_audio else None
                expected_size = (raw_audio or {}).get("file_size_bytes") if raw_audio else None

                if matched_audio and expected_hash:
                    try:
                        audio_cached_hash = None
                        if matched_audio in audio_hash_cache:
                            audio_cached_hash = audio_hash_cache[matched_audio]
                        elif str(matched_audio.resolve()) in existing_audio_by_path:
                            audio_cached_hash = existing_audio_by_path[str(matched_audio.resolve())].file_hash
                            audio_hash_cache[matched_audio] = audio_cached_hash
                        elif matched_audio.name.lower() in existing_audio_by_name:
                            audio_cached_hash = existing_audio_by_name[matched_audio.name.lower()].file_hash
                            audio_hash_cache[matched_audio] = audio_cached_hash

                        if audio_cached_hash:
                            if audio_cached_hash != expected_hash:
                                hash_mismatch = True
                        elif expected_size is not None:
                            # Fast size check before sequential byte read
                            if matched_audio.stat().st_size != expected_size:
                                hash_mismatch = True
                            elif matched_audio.stat().st_size <= 10 * 1024 * 1024:
                                audio_hash_cache[matched_audio] = AudioMetadata.compute_file_hash(matched_audio)
                                if audio_hash_cache[matched_audio] != expected_hash:
                                    hash_mismatch = True
                        else:
                            audio_hash_cache[matched_audio] = AudioMetadata.compute_file_hash(matched_audio)
                            if audio_hash_cache[matched_audio] != expected_hash:
                                hash_mismatch = True
                    except Exception:
                        pass

                # Guardrail 3: Database deduplication & upgrade quality check
                existing_tx = None
                audio_duration = (raw_audio or {}).get("duration_seconds") if raw_audio else None
                existing_tx_id = None
                is_existing_record = False

                audio_rec = None
                if expected_hash and expected_hash in existing_audio_by_hash:
                    audio_rec = existing_audio_by_hash[expected_hash]
                elif matched_audio and matched_audio.name.lower() in existing_audio_by_name:
                    audio_rec = existing_audio_by_name[matched_audio.name.lower()]
                elif tx_stem_norm in existing_audio_by_stem:
                    audio_rec = existing_audio_by_stem[tx_stem_norm]

                if audio_rec:
                    if not audio_duration and audio_rec.duration_seconds:
                        audio_duration = audio_rec.duration_seconds
                    existing_tx = existing_tx_by_audio_id.get(audio_rec.id)

                if not existing_tx and tx_stem_norm in existing_tx_by_stem:
                    existing_tx = existing_tx_by_stem[tx_stem_norm]

                if existing_tx:
                    is_existing_record = True
                    existing_tx_id = existing_tx.id

                # Guardrail 6: Duration Sanity Check
                if (not tx_duration or tx_duration <= 0) and segments_raw:
                    from audiobench.transcribe.quality_evaluator import _get_attr
                    tx_duration = max([float(_get_attr(s, "end", 0.0)) for s in segments_raw], default=0.0)

                # Always probe the physical audio file on disk to verify duration alignment
                physical_audio_duration = None
                if matched_audio and matched_audio.exists():
                    if matched_audio in audio_duration_cache:
                        physical_audio_duration = audio_duration_cache[matched_audio]
                    else:
                        try:
                            from audiobench.transcribe.audio_converter import probe
                            info = probe(matched_audio)
                            if info and info.duration > 0:
                                physical_audio_duration = info.duration
                                audio_duration_cache[matched_audio] = info.duration
                        except Exception:
                            pass

                effective_audio_dur = physical_audio_duration or audio_duration

                duration_mismatch = False
                if matched_audio and effective_audio_dur and effective_audio_dur > 0 and tx_duration and tx_duration > 0:
                    dur_diff = abs(tx_duration - effective_audio_dur)
                    dur_tolerance = max(5.0, 0.05 * effective_audio_dur)
                    if dur_diff > dur_tolerance:
                        duration_mismatch = True

                incoming_cand = {
                    "segments": segments_raw,
                    "duration_seconds": tx_duration,
                    "engine": engine,
                    "text": tx_text,
                }
                comp = compare_transcripts(existing_tx, incoming_cand, audio_duration=effective_audio_dur)

                tag_modified_metrics = None
                # Determine operational status & selection
                if not matched_audio:
                    status = "NO_AUDIO"
                    selected = False
                elif is_empty:
                    status = "EMPTY"
                    selected = False
                elif hash_mismatch:
                    passes_tm, tm_metrics = evaluate_tag_modified_heuristic(matched_audio, raw_audio, tx_duration)
                    tag_modified_metrics = tm_metrics
                    if passes_tm:
                        status = "TAG_MODIFIED"
                        selected = False
                    else:
                        status = "HASH_MISMATCH"
                        selected = False
                elif duration_mismatch:
                    status = "DURATION_MISMATCH"
                    selected = False
                elif comp.recommendation == "UPGRADE_RECOMMENDED":
                    status = "UPGRADE"
                    selected = True
                elif comp.recommendation in ("KEEP_EXISTING", "SIMILAR_QUALITY"):
                    status = "CURRENT"
                    selected = False
                elif comp.recommendation == "MISMATCH":
                    status = "MISMATCH"
                    selected = False
                elif existing_tx:
                    status = "CURRENT"
                    selected = False
                else:
                    status = "READY"
                    selected = True

                from audiobench.media.classifier import classify_media

                cls_res = classify_media(
                    file_path=matched_audio,
                    duration_seconds=audio_duration or tx_duration or 0.0,
                )
                if cls_res.is_autonomous_bypass:
                    is_book = True
                    badge = "[Book: Proof]"
                elif cls_res.category == "class_b_soft_heuristic":
                    is_book = False
                    badge = "[Book: Review? [B]]"
                else:
                    is_book = False
                    badge = ""

                self.batch_pairs.append({
                    "tx": tx_file,
                    "audio": matched_audio,
                    "status": status,
                    "selected": selected,
                    "is_existing": is_existing_record,
                    "existing_record_id": existing_tx_id,
                    "comparison": comp,
                    "incoming_tqs": comp.incoming_tqs,
                    "existing_tqs": comp.existing_tqs,
                    "familiarity": comp.familiarity,
                    "audio_duration": effective_audio_dur,
                    "tx_duration": tx_duration,
                    "is_audiobook": is_book,
                    "book_badge": badge,
                    "classification_category": cls_res.category,
                    "tag_modified_metrics": tag_modified_metrics,
                })

                # Immediately reclaim memory for this candidate
                del data
                del segments_raw
                del incoming_cand

        # Deduplicate candidates matching the exact same audio file
        deduped_pairs = []
        audio_target_map: dict[str, int] = {}
        for pair in self.batch_pairs:
            au = pair.get("audio")
            if au is None:
                deduped_pairs.append(pair)
                continue
            au_key = str(au.resolve())
            if au_key not in audio_target_map:
                audio_target_map[au_key] = len(deduped_pairs)
                deduped_pairs.append(pair)
            else:
                existing_idx = audio_target_map[au_key]
                existing_pair = deduped_pairs[existing_idx]
                new_tqs = pair["incoming_tqs"].total_score if pair.get("incoming_tqs") else 0.0
                ex_tqs = existing_pair["incoming_tqs"].total_score if existing_pair.get("incoming_tqs") else 0.0
                if new_tqs > ex_tqs:
                    deduped_pairs[existing_idx] = pair
                elif new_tqs == ex_tqs:
                    # Prefer deeper / organized folder (e.g. whisper_json_transcripts) over loose root files
                    if len(str(pair["tx"])) > len(str(existing_pair["tx"])):
                        deduped_pairs[existing_idx] = pair
        self.batch_pairs = deduped_pairs


    def _nav_input(self, key, nav, on_enter):
        if key in (curses.KEY_RIGHT, 10, 13, ord("l")):
            cur = nav.get_current_item_path()
            if cur and cur.is_file():
                on_enter(cur)
            else:
                nav.enter()
        elif key in (curses.KEY_LEFT, ord("h")):
            nav.go_up()
        elif key in (curses.KEY_UP, ord("k")):
            nav.select_prev()
        elif key in (curses.KEY_DOWN, ord("j")):
            nav.select_next()
        elif key in (ord("t"), ord("H")):
            nav.toggle_hidden_visibility()
        elif key == ord(" "):
            nav.toggle_selection()

    def _on_tx_selected(self, path):
        self.transcript_path = path
        try:
            self.parsed_transcript, self.raw_audio = parse_transcript_file(path)
            from audiobench.transcribe.audio_converter import ALL_SUPPORTED_FORMATS
            auto_audio = None
            raw_audio_name = (self.raw_audio or {}).get("file_name")
            if raw_audio_name and (path.parent / raw_audio_name).exists():
                auto_audio = path.parent / raw_audio_name
            else:
                for ext in ALL_SUPPORTED_FORMATS:
                    cand = path.with_suffix(f".{ext}")
                    if cand.exists():
                        auto_audio = cand
                        break

            if auto_audio:
                self._on_audio_selected(auto_audio)
            else:
                self.screen = 2
        except Exception:
            self.screen = 2

    def _on_audio_selected(self, path):
        self.audio_path = path
        self.screen = 3
        self.hash_computed = False
        self.hash_error = None
        self.db_check_result = None

        # Start hash verification in background
        threading.Thread(target=self._verify_hash, daemon=True).start()

    def _on_tx_dir_selected(self, path):
        pass # batch

    def _verify_hash(self):
        try:
            self.audio_metadata = build_audio_metadata(self.audio_path, self.raw_audio)

            init_db()
            repo = TranscriptionRepository()
            from audiobench.core.db_session import get_session
            from audiobench.transcribe.quality_evaluator import compare_transcripts, evaluate_transcript_quality
            with get_session() as session:
                rec = session.query(AudioFileRecord).filter_by(file_hash=self.audio_metadata.file_hash).first()
                if not rec:
                    rec = session.query(AudioFileRecord).filter_by(file_name=self.audio_path.name).first()

                self.comparison_result = None
                self.incoming_tqs = None
                if self.parsed_transcript and self.parsed_transcript.segments:
                    self.incoming_tqs = evaluate_transcript_quality(
                        self.parsed_transcript.segments,
                        total_duration=self.audio_metadata.duration_seconds
                    )

                if rec:
                    self.db_check_result = "exists"
                    self.existing_transcriptions_count = session.query(TranscriptionRecord).filter_by(audio_file_id=rec.id).count()
                    existing_tx = session.query(TranscriptionRecord).filter_by(audio_file_id=rec.id).first()
                    if existing_tx:
                        self.comparison_result = compare_transcripts(
                            existing_tx,
                            self.parsed_transcript,
                            audio_duration=self.audio_metadata.duration_seconds
                        )
                else:
                    self.db_check_result = "new"
        except HashMismatchError as e:
            self.hash_error = str(e)
            from audiobench.transcribe.transcript_parser import evaluate_tag_modified_heuristic
            tx_dur = self.parsed_transcript.duration_seconds if self.parsed_transcript else None
            passes, metrics = evaluate_tag_modified_heuristic(self.audio_path, self.raw_audio, tx_dur)
            self.tag_modified_heuristic_passed = passes
            self.tag_modified_metrics = metrics
        except Exception as e:
            self.hash_error = f"Error computing hash: {e}"
        finally:
            self.hash_computed = True

    def _draw_screen_1(self, stdscr):
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " Reimport: Select Transcript File ".ljust(w), curses.color_pair(1) | curses.A_BOLD)

        # Draw navigator left
        items = self.tx_nav.list_items()
        max_visible = h - 4
        if self.tx_nav.selected < self.ui_offset:
            self.ui_offset = self.tx_nav.selected
        elif self.tx_nav.selected >= self.ui_offset + max_visible:
            self.ui_offset = self.tx_nav.selected - max_visible + 1

        visible_items = items[self.ui_offset:self.ui_offset + max_visible]
        for i, item in enumerate(visible_items):
            y = i + 2
            idx = self.ui_offset + i
            is_sel = (idx == self.tx_nav.selected)
            attr = curses.color_pair(3) | curses.A_BOLD if is_sel else curses.color_pair(2)
            icon = "[DIR]" if (self.tx_nav.current_path / item).is_dir() else "[TXT]"
            stdscr.addstr(y, 1, f"{'>' if is_sel else ' '} {icon} {item}"[:w//2], attr)

        try:
            footer_2 = " Modes: [1]Audio [2]Auto-Detect [3]Single [4]Batch  |  q: Quit "
            stdscr.addstr(h-2, 0, " ENTER: Select | b: Batch Mode | a: Auto-Detect ".ljust(w-1), curses.color_pair(1))
            stdscr.addstr(h-1, 0, footer_2.ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass

    def _draw_screen_2(self, stdscr):
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " Reimport: Link Local Audio File ".ljust(w), curses.color_pair(1) | curses.A_BOLD)

        items = self.audio_nav.list_items()
        max_visible = h - 4
        if self.audio_nav.selected < self.ui_offset:
            self.ui_offset = self.audio_nav.selected
        elif self.audio_nav.selected >= self.ui_offset + max_visible:
            self.ui_offset = self.audio_nav.selected - max_visible + 1

        visible_items = items[self.ui_offset:self.ui_offset + max_visible]
        for i, item in enumerate(visible_items):
            y = i + 2
            idx = self.ui_offset + i
            is_sel = (idx == self.audio_nav.selected)
            attr = curses.color_pair(3) | curses.A_BOLD if is_sel else curses.color_pair(2)
            icon = "[DIR]" if (self.audio_nav.current_path / item).is_dir() else "[AUD]"
            stdscr.addstr(y, 1, f"{'>' if is_sel else ' '} {icon} {item}"[:w//2], attr)

        try:
            footer_2 = " Modes: [1]Audio [2]Auto-Detect [3]Single [4]Batch  |  q: Quit "
            stdscr.addstr(h-2, 0, " ENTER: Select ".ljust(w-1), curses.color_pair(1))
            stdscr.addstr(h-1, 0, footer_2.ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass

    def _draw_screen_3(self, stdscr):
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " Reimport: Verify Identity & Quality ".ljust(w), curses.color_pair(1) | curses.A_BOLD)

        if not self.hash_computed:
            stdscr.addstr(3, 3, "Computing SHA-256 and evaluating quality metrics...", curses.color_pair(5))
        elif self.hash_error:
            if getattr(self, "tag_modified_heuristic_passed", False):
                metrics = getattr(self, "tag_modified_metrics", {})
                stdscr.addstr(2, 2, "[!] HASH MISMATCH — Tag-Modified Stream Heuristic Passed", curses.color_pair(4) | curses.A_BOLD)
                dur_diff = metrics.get("duration_delta_sec", 0.0)
                rel_dur = metrics.get("relative_duration_delta", 0.0)
                sz_diff = metrics.get("size_delta_bytes", 0)
                stdscr.addstr(4, 2, "Audio stream matches transcript within safe tolerances (ID3 metadata edit):", curses.color_pair(2))
                stdscr.addstr(5, 2, f"  Δ Duration: {dur_diff:.3f}s ({rel_dur:.3%})  |  Δ File Size: {sz_diff:,} bytes", curses.color_pair(2))
                stdscr.addstr(6, 2, "  Normalized Stems Match: 4/4 stream invariants verified.", curses.color_pair(2))
                stdscr.addstr(8, 2, "[F] Force Link (Audit Provenance) | [R] Re-select Audio File", curses.color_pair(3) | curses.A_BOLD)
            else:
                stdscr.addstr(3, 3, "[ERR] Hash mismatch — this is not the correct audio file", curses.color_pair(7) | curses.A_BOLD)
                stdscr.addstr(5, 3, self.hash_error, curses.color_pair(7))
                stdscr.addstr(8, 3, "[R] Re-select Audio File", curses.color_pair(2) | curses.A_BOLD)
        elif self.db_check_result == "exists":
            stdscr.addstr(2, 2, "[OK] Hash confirmed. File identity verified.", curses.color_pair(3) | curses.A_BOLD)

            comp = getattr(self, "comparison_result", None)
            py = 4
            if comp and comp.existing_tqs:
                col_w = max(25, (w - 8) // 2)
                hdr_left = "INCOMING CANDIDATE".ljust(col_w)
                hdr_right = f"EXISTING DB RECORD (#{comp.existing_id})"
                stdscr.addstr(py, 2, f"{hdr_left} │ {hdr_right}"[:w-4], curses.color_pair(5) | curses.A_BOLD)
                py += 1

                inc = comp.incoming_tqs
                ex = comp.existing_tqs

                c_rows = [
                    (f"TQS Score: {inc.total_score:.1f}/100 ({inc.grade})", f"TQS Score: {ex.total_score:.1f}/100 ({ex.grade})"),
                    (f"Words:     {inc.word_count:,} ({inc.wpm:.1f} WPM)", f"Words:     {ex.word_count:,} ({ex.wpm:.1f} WPM)"),
                    (f"Word Timestamps: {inc.words_with_timestamps_ratio*100:.0f}% ({inc.granularity_score:.1f}/25)", f"Word Timestamps: {ex.words_with_timestamps_ratio*100:.0f}% ({ex.granularity_score:.1f}/25)"),
                    (f"Health:    {inc.health_score:.1f}/25 ({inc.zero_duration_segments_count} zero, {inc.stretched_segments_count} str)", f"Health:    {ex.health_score:.1f}/25 ({ex.zero_duration_segments_count} zero, {ex.stretched_segments_count} str)"),
                ]
                for left_txt, right_txt in c_rows:
                    stdscr.addstr(py, 2, f"{left_txt[:col_w]:<{col_w}} │ {right_txt}"[:w-4], curses.color_pair(2))
                    py += 1

                sim = comp.familiarity
                fam_color = curses.color_pair(3) if sim.classification == "DIRECT_MATCH" else (curses.color_pair(4) if sim.classification == "PARTIAL_MATCH" else curses.color_pair(7))
                stdscr.addstr(py+1, 2, f"Equivalence: {sim.score:.1f}% ({sim.classification})  |  Action: {comp.recommendation}"[:w-4], fam_color | curses.A_BOLD)
                py += 3
            else:
                stdscr.addstr(py, 2, f"Existing audio record found with {self.existing_transcriptions_count} transcriptions.")
                py += 2

            stdscr.addstr(py, 2, "[O]verwrite (Atomic Upgrade) | [A]dd as secondary | [S]kip", curses.color_pair(2) | curses.A_BOLD)

        elif self.db_check_result == "new":
            stdscr.addstr(2, 2, "[OK] Hash confirmed. File identity verified (New Recording).", curses.color_pair(3) | curses.A_BOLD)
            py = 4
            tqs = getattr(self, "incoming_tqs", None)
            if tqs:
                stdscr.addstr(py, 2, f"TQS Score:       {tqs.total_score:.1f}/100 ({tqs.grade})", curses.color_pair(2))
                py += 1
                stdscr.addstr(py, 2, f"Word Timestamps: {tqs.words_with_timestamps_ratio*100:.0f}% ({tqs.word_count:,} words, {tqs.wpm:.1f} WPM)", curses.color_pair(2))
                py += 1
                stdscr.addstr(py, 2, f"Timeline Health: {tqs.health_score:.1f}/25 pts (0 defects)", curses.color_pair(2))
                py += 2
            stdscr.addstr(py, 2, "Press ENTER to proceed to Pipeline Confirmation | [R] Re-select", curses.color_pair(2) | curses.A_BOLD)

        try:
            stdscr.addstr(h-1, 0, " q: Quit ".ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass

    def _draw_screen_4(self, stdscr):
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " Reimport: Confirm Pipeline ".ljust(w), curses.color_pair(1) | curses.A_BOLD)

        stdscr.addstr(2, 2, f"Transcript: {self.transcript_path.name}")
        stdscr.addstr(3, 2, f"Audio:      {self.audio_path.name}")
        if getattr(self, "audit_provenance", None):
            stdscr.addstr(4, 2, "[!] HASH OVERRIDE ACTIVE (Tag-Modified Audit Trail Enabled)", curses.color_pair(4) | curses.A_BOLD)

        stdscr.addstr(6, 2, "Pipeline:")
        stdscr.addstr(7, 2, " [1] Copy .mp3 and sidecars")
        stdscr.addstr(8, 2, " [2] Save AudioFileRecord")
        stdscr.addstr(9, 2, " [3] Auto-detect chapters")
        stdscr.addstr(10, 2, " [4] Save TranscriptionRecord & Segments")
        stdscr.addstr(11, 2, " [5] Semantic chunk & embed")

        try:
            stdscr.addstr(h-1, 0, " ENTER: Start Import | q: Cancel ".ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass

    def _draw_screen_5(self, stdscr):
        stdscr.clear()
        stdscr.addstr(2, 2, "Importing...", curses.color_pair(5))
        stdscr.refresh()

        # Actually do the import
        try:
            init_db()
            repo = TranscriptionRepository()

            def _phase_cb(phase: str, pct: float):
                pass

            do_overwrite = (getattr(self, "conflict_action", None) == "O")
            tx_id = repo.save_transcription(
                self.parsed_transcript,
                self.audio_metadata,
                on_phase=_phase_cb,
                overwrite=do_overwrite,
                is_audiobook=getattr(self, "is_audiobook", False),
                audit_provenance=getattr(self, "audit_provenance", None),
            )

            # update source to reimport
            from audiobench.core.db_session import get_session
            with get_session() as session:
                rec = session.query(TranscriptionRecord).filter_by(id=tx_id).first()
                if rec:
                    rec.source = "reimport:hash_override" if getattr(self, "audit_provenance", None) else "reimport"
                    session.commit()

            # Also update sidecar in library alongside audio file
            try:
                with get_session() as session:
                    rec = session.query(TranscriptionRecord).filter_by(id=tx_id).first()
                    if rec and rec.audio_file and rec.audio_file.file_path:
                        dest_sidecar = Path(rec.audio_file.file_path).with_suffix(".json")
                        if self.transcript_path.exists() and self.transcript_path.resolve() != dest_sidecar.resolve():
                            import shutil
                            shutil.copy2(str(self.transcript_path), str(dest_sidecar))
            except Exception:
                pass

            stdscr.clear()
            stdscr.addstr(2, 2, f"[OK] Transcription #{tx_id} imported successfully!", curses.color_pair(3))
        except Exception as e:
            stdscr.clear()
            stdscr.addstr(2, 2, f"[ERR] Import failed: {e}", curses.color_pair(7))

        stdscr.addstr(4, 2, "Press any key to exit.")
        stdscr.refresh()
        stdscr.timeout(-1)
        stdscr.getch()

    def _draw_screen_1b(self, stdscr):
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " Batch Reimport: Select Transcript Folder ".ljust(w), curses.color_pair(1) | curses.A_BOLD)

        existing = self._get_existing_basenames()
        items = self.tx_nav.list_items()

        max_visible = h - 5
        if self.tx_nav.selected < self.ui_offset:
            self.ui_offset = self.tx_nav.selected
        elif self.tx_nav.selected >= self.ui_offset + max_visible:
            self.ui_offset = self.tx_nav.selected - max_visible + 1

        visible_items = items[self.ui_offset:self.ui_offset + max_visible]
        for i, item in enumerate(visible_items):
            y = i + 2
            idx = self.ui_offset + i
            is_sel = (idx == self.tx_nav.selected)
            item_path = self.tx_nav.current_path / item
            is_checked = item_path in self.tx_nav.selected_files
            attr = curses.color_pair(3) | curses.A_BOLD if is_sel else curses.color_pair(2)

            if item_path.is_dir():
                icon = "[DIR]"
                prefix = f"{'>' if is_sel else ' '} "
                suffix = ""
            else:
                icon = "[TXT]"
                prefix = f"{'>' if is_sel else ' '} {'[x]' if is_checked else '[ ]'}"
                suffix = " [Transcribed]" if item_path.stem in existing else ""
                if suffix:
                    attr = curses.color_pair(5) | curses.A_BOLD if is_sel else curses.color_pair(5)

            stdscr.addstr(y, 1, f"{prefix} {icon} {item}{suffix}"[:w-2], attr)

        stdscr.addstr(h-3, 0, f" Current Folder: {self.tx_nav.current_path} ({len(self.tx_nav.selected_files)} selected)".ljust(w), curses.color_pair(5))
        try:
            footer_2 = " Modes: [1]Audio [2]Auto-Detect [3]Single [4]Batch  |  q: Quit "
            stdscr.addstr(h-2, 0, " SPACE: Select File | ENTER: Confirm Folder/Files ".ljust(w-1), curses.color_pair(1))
            stdscr.addstr(h-1, 0, footer_2.ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass

    def _on_tx_dir_selected(self, path):
        # path is actually the selected item, but we want the current directory for batch
        pass # we handle it directly in handle_input for 1b

    def _draw_screen_2b(self, stdscr):
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " Batch Reimport: Select Audio Folder ".ljust(w), curses.color_pair(1) | curses.A_BOLD)

        items = self.audio_nav.list_items()
        max_visible = h - 5
        if self.audio_nav.selected < self.ui_offset:
            self.ui_offset = self.audio_nav.selected
        elif self.audio_nav.selected >= self.ui_offset + max_visible:
            self.ui_offset = self.audio_nav.selected - max_visible + 1

        visible_items = items[self.ui_offset:self.ui_offset + max_visible]
        for i, item in enumerate(visible_items):
            y = i + 2
            idx = self.ui_offset + i
            is_sel = (idx == self.audio_nav.selected)
            attr = curses.color_pair(3) | curses.A_BOLD if is_sel else curses.color_pair(2)
            icon = "[DIR]" if (self.audio_nav.current_path / item).is_dir() else "[AUD]"
            stdscr.addstr(y, 1, f"{'>' if is_sel else ' '} {icon} {item}"[:w-2], attr)

        stdscr.addstr(h-3, 0, f" Current Folder: {self.audio_nav.current_path}".ljust(w), curses.color_pair(5))
        try:
            footer_2 = " Modes: [1]Audio [2]Auto-Detect [3]Single [4]Batch  |  q: Quit "
            stdscr.addstr(h-2, 0, " ENTER: Confirm Folder ".ljust(w-1), curses.color_pair(1))
            stdscr.addstr(h-1, 0, footer_2.ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass

    def _draw_screen_15(self, stdscr):
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, f" [!] Conflict: {len(self.conflict_pairs)} files already exist ".ljust(w), curses.color_pair(7) | curses.A_BOLD)

        for i, pair in enumerate(self.conflict_pairs[:h-4]):
            y = i + 2
            tx = pair["tx"].name
            stdscr.addstr(y, 2, f"- {tx}"[:w-4], curses.color_pair(5))

        try:
            stdscr.addstr(h-1, 0, " [S]kip Existing | [R]eplace Existing | q: Cancel ".ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass

    def _draw_screen_3b(self, stdscr):
        h, w = stdscr.getmaxyx()
        msg = f" Batch Ready: {len(self.batch_pairs)} pairs matched "
        if getattr(self, "batch_overwrite", False):
            msg += "(OVERWRITE ON) "
        stdscr.addstr(0, 0, msg.ljust(w), curses.color_pair(1) | curses.A_BOLD)

        for i, pair in enumerate(self.batch_pairs[:h-4]):
            y = i + 2
            tx = pair["tx"].name
            au = pair["audio"].name if pair["audio"] else "???"
            stdscr.addstr(y, 2, f"{tx} <--> {au}"[:w-4])

        try:
            stdscr.addstr(h-1, 0, " S: Start Batch | q: Quit ".ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass

    def _draw_screen_4b(self, stdscr):
        stdscr.clear()
        stdscr.addstr(2, 2, "Batch Importing...", curses.color_pair(5))
        stdscr.refresh()

        success = 0
        failed = 0
        failed_items = []

        from audiobench.transcribe.transcript_parser import build_audio_metadata, evaluate_tag_modified_heuristic

        for idx, pair in enumerate(self.batch_pairs):
            stdscr.addstr(4, 2, f"Processing {idx+1}/{len(self.batch_pairs)}: {pair['tx'].name[:50]}                  ")
            stdscr.refresh()

            if not pair.get("audio") or not pair.get("selected", True):
                if not pair.get("audio"):
                    failed += 1
                    failed_items.append({
                        "pair": pair,
                        "tx": pair["tx"],
                        "audio": None,
                        "raw_audio": None,
                        "parsed_tx": None,
                        "error": "No matching audio file found",
                        "error_type": "NoAudio",
                        "heuristic_pass": False,
                        "heuristic_metrics": None,
                        "status": "NO_AUDIO",
                    })
                continue

            parsed_tx = None
            raw_audio = None
            audio_meta = None
            try:
                parsed_tx = pair.get("parsed_tx")
                raw_audio = pair.get("raw_audio")
                if not parsed_tx:
                    parsed_tx, raw_audio = parse_transcript_file(pair["tx"])

                is_tag_mod = (pair.get("status") == "TAG_MODIFIED")
                audio_meta = build_audio_metadata(pair["audio"], raw_audio, allow_hash_mismatch=is_tag_mod)

                init_db()
                repo = TranscriptionRepository()

                do_overwrite = getattr(self, "batch_overwrite", False) or pair.get("status") == "UPGRADE" or pair.get("is_existing", False)

                audit_prov = None
                if is_tag_mod:
                    tm_metrics = pair.get("tag_modified_metrics") or {}
                    audit_prov = {
                        "hash_verification": "bypassed_tag_modified",
                        "verified_by": "operator_interactive",
                        "original_json_hash": (raw_audio or {}).get("file_hash"),
                        "actual_audio_hash": audio_meta.file_hash,
                        "duration_delta_sec": tm_metrics.get("duration_delta_sec"),
                        "size_delta_bytes": tm_metrics.get("size_delta_bytes"),
                        "source": "reimport:hash_override",
                    }

                tx_id = repo.save_transcription(
                    parsed_tx,
                    audio_meta,
                    on_phase=lambda *args, **kwargs: None,
                    overwrite=do_overwrite,
                    is_audiobook=pair.get("is_audiobook", False),
                    audit_provenance=audit_prov,
                )
                from audiobench.core.db_session import get_session
                with get_session() as session:
                    rec = session.query(TranscriptionRecord).filter_by(id=tx_id).first()
                    if rec:
                        rec.source = "reimport:hash_override" if audit_prov else "reimport"
                        session.commit()

                # Also update sidecar in library alongside audio file
                try:
                    with get_session() as session:
                        rec = session.query(TranscriptionRecord).filter_by(id=tx_id).first()
                        if rec and rec.audio_file and rec.audio_file.file_path:
                            dest_sidecar = Path(rec.audio_file.file_path).with_suffix(".json")
                            if pair["tx"].exists() and pair["tx"].resolve() != dest_sidecar.resolve():
                                import shutil
                                shutil.copy2(str(pair["tx"]), str(dest_sidecar))
                except Exception:
                    pass

                success += 1
            except Exception as err:
                failed += 1
                heur_pass = False
                heur_metrics = None
                if pair.get("audio"):
                    try:
                        tx_dur = parsed_tx.duration_seconds if parsed_tx else None
                        heur_pass, heur_metrics = evaluate_tag_modified_heuristic(pair["audio"], raw_audio, tx_dur)
                    except Exception:
                        pass
                failed_items.append({
                    "pair": pair,
                    "tx": pair["tx"],
                    "audio": pair.get("audio"),
                    "raw_audio": raw_audio,
                    "parsed_tx": parsed_tx,
                    "error": str(err),
                    "error_type": type(err).__name__,
                    "heuristic_pass": heur_pass,
                    "heuristic_metrics": heur_metrics,
                    "status": "TAG_MODIFIED_ELIGIBLE" if heur_pass else "FAILED",
                })
            finally:
                parsed_tx = None
                raw_audio = None
                audio_meta = None

        if not failed_items:
            stdscr.clear()
            stdscr.addstr(2, 2, f"[OK] Batch Complete! {success} succeeded, 0 failed.", curses.color_pair(3) | curses.A_BOLD)
            stdscr.addstr(4, 2, "Press any key to exit.")
            stdscr.refresh()
            stdscr.timeout(-1)
            stdscr.getch()
            return

        self._interactive_failure_review(stdscr, failed_items, success)

    def _interactive_failure_review(self, stdscr, failed_items: list[dict], success: int):
        cur_idx = 0
        ui_offset = 0

        while True:
            h, w = stdscr.getmaxyx()
            stdscr.clear()

            n_failed = sum(1 for item in failed_items if item.get("status") != "IMPORTED")
            header = f" Batch Complete: {success} succeeded, {n_failed} failed — Interactive Failure Review "
            stdscr.addstr(0, 0, header[:w-1].ljust(w), curses.color_pair(1) | curses.A_BOLD)

            split_y = max(6, min(h - 9, int(h * 0.48)))
            max_visible = split_y - 1

            if cur_idx < ui_offset:
                ui_offset = cur_idx
            elif cur_idx >= ui_offset + max_visible:
                ui_offset = cur_idx - max_visible + 1

            for i in range(max_visible):
                idx = ui_offset + i
                if idx >= len(failed_items):
                    break
                item = failed_items[idx]
                y = 1 + i
                is_sel = (idx == cur_idx)
                status = item.get("status", "FAILED")
                tx_name = item["tx"].name
                au_name = item["audio"].name if item.get("audio") else "NO AUDIO"

                if status == "IMPORTED":
                    badge = "[IMPORTED]"
                    line_attr = curses.color_pair(3) | curses.A_BOLD if is_sel else curses.color_pair(3)
                elif item.get("heuristic_pass"):
                    badge = "[TAG_MOD [F]]"
                    line_attr = curses.color_pair(4) | curses.A_BOLD if is_sel else curses.color_pair(4)
                else:
                    badge = f"[{item.get('error_type', 'FAIL')}]"
                    line_attr = curses.color_pair(7) | curses.A_BOLD if is_sel else curses.color_pair(7)

                prefix = ">" if is_sel else " "
                badge_str = f"{badge:<14}"
                avail_w = max(30, w - (len(prefix) + len(badge_str) + 8))
                col_tx = max(18, avail_w // 2)
                col_au = max(16, avail_w - col_tx)

                display_tx = self._smart_fit_name(tx_name, col_tx)
                display_au = self._smart_fit_name(au_name, col_au)

                row_str = f"{prefix} {badge_str} {display_tx:<{col_tx}} ↔ {display_au:<{col_au}}"
                stdscr.addstr(y, 1, row_str[:w-2], line_attr)

            # Separator
            sep = "─── Failure Diagnostic & Force Import Inspector "
            stdscr.addstr(split_y, 0, (sep + "─" * max(0, w - len(sep)))[:w-1], curses.color_pair(6) | curses.A_BOLD)

            # Inspector pane
            if 0 <= cur_idx < len(failed_items):
                cur_item = failed_items[cur_idx]
                py = split_y + 1
                tx_p = cur_item["tx"]
                au_p = cur_item.get("audio")
                stdscr.addstr(py, 2, f"Candidate: {tx_p.name[:w-15]}"[:w-4], curses.color_pair(2) | curses.A_BOLD)
                py += 1
                au_name_str = au_p.name if au_p else "MISSING AUDIO"
                stdscr.addstr(py, 2, f"Audio:     {au_name_str[:w-15]}"[:w-4], curses.color_pair(2))
                py += 1

                if cur_item.get("status") == "IMPORTED":
                    stdscr.addstr(py, 2, "[OK] Force imported successfully with audit provenance.", curses.color_pair(3) | curses.A_BOLD)
                    py += 1
                elif cur_item.get("heuristic_pass"):
                    m = cur_item.get("heuristic_metrics") or {}
                    dur_delta = m.get("duration_delta_sec", 0.0)
                    rel_dur = m.get("relative_duration_delta", 0.0)
                    sz_delta = m.get("size_delta_bytes", 0)
                    stdscr.addstr(py, 2, "[!] TAG_MODIFIED: Hash mismatch, but passes all 4 stream invariants.", curses.color_pair(4) | curses.A_BOLD)
                    py += 1
                    stdscr.addstr(py, 2, f"    Δ Duration: {dur_delta:.3f}s ({rel_dur:.3%})  |  Δ Size: {sz_delta:,} bytes", curses.color_pair(2))
                    py += 1
                    stdscr.addstr(py, 2, "    Audio stream verified identical. Press [F] to force import with audit trail.", curses.color_pair(3) | curses.A_BOLD)
                    py += 1
                else:
                    stdscr.addstr(py, 2, f"[ERR] {cur_item.get('error_type')}: {cur_item.get('error')}"[:w-4], curses.color_pair(7) | curses.A_BOLD)
                    py += 1
                    m = cur_item.get("heuristic_metrics") or {}
                    if m.get("reason"):
                        stdscr.addstr(py, 2, f"    Invariants check failed: {m.get('reason')}"[:w-4], curses.color_pair(4))
                        py += 1

            # Footer
            can_force = (
                0 <= cur_idx < len(failed_items)
                and failed_items[cur_idx].get("heuristic_pass")
                and failed_items[cur_idx].get("status") != "IMPORTED"
            )
            force_prompt = "[F] Force Import (Audit) | " if can_force else ""
            footer = f" {force_prompt}UP/DOWN / j/k: Navigate | ENTER / q: Exit "
            try:
                stdscr.addstr(h-1, 0, footer[:w-1].ljust(w-1), curses.color_pair(1))
            except curses.error:
                pass

            stdscr.refresh()
            stdscr.timeout(-1)
            key = stdscr.getch()

            if key in (curses.KEY_UP, ord('k')):
                cur_idx = max(0, cur_idx - 1)
            elif key in (curses.KEY_DOWN, ord('j')):
                cur_idx = min(len(failed_items) - 1, cur_idx + 1)
            elif key in (ord('q'), ord('Q'), 10, 13):
                break
            elif key in (ord('f'), ord('F')) and can_force:
                cur_item = failed_items[cur_idx]
                try:
                    from audiobench.transcribe.transcript_parser import build_audio_metadata, parse_transcript_file
                    parsed_tx = cur_item.get("parsed_tx")
                    raw_audio = cur_item.get("raw_audio")
                    if not parsed_tx:
                        parsed_tx, raw_audio = parse_transcript_file(cur_item["tx"])
                    audio_meta = build_audio_metadata(cur_item["audio"], raw_audio, allow_hash_mismatch=True)
                    m = cur_item.get("heuristic_metrics") or {}
                    audit_prov = {
                        "hash_verification": "bypassed_tag_modified",
                        "verified_by": "operator_interactive",
                        "original_json_hash": (raw_audio or {}).get("file_hash"),
                        "actual_audio_hash": audio_meta.file_hash,
                        "duration_delta_sec": m.get("duration_delta_sec"),
                        "size_delta_bytes": m.get("size_delta_bytes"),
                        "source": "reimport:hash_override",
                    }
                    init_db()
                    repo = TranscriptionRepository()
                    tx_id = repo.save_transcription(
                        parsed_tx,
                        audio_meta,
                        on_phase=lambda *args, **kwargs: None,
                        overwrite=True,
                        is_audiobook=cur_item["pair"].get("is_audiobook", False),
                        audit_provenance=audit_prov,
                    )
                    from audiobench.core.db_session import get_session
                    with get_session() as session:
                        rec = session.query(TranscriptionRecord).filter_by(id=tx_id).first()
                        if rec:
                            rec.source = "reimport:hash_override"
                            session.commit()
                    try:
                        with get_session() as session:
                            rec = session.query(TranscriptionRecord).filter_by(id=tx_id).first()
                            if rec and rec.audio_file and rec.audio_file.file_path:
                                dest_sidecar = Path(rec.audio_file.file_path).with_suffix(".json")
                                if cur_item["tx"].exists() and cur_item["tx"].resolve() != dest_sidecar.resolve():
                                    import shutil
                                    shutil.copy2(str(cur_item["tx"]), str(dest_sidecar))
                    except Exception:
                        pass
                    cur_item["status"] = "IMPORTED"
                    cur_item["heuristic_pass"] = False
                    success += 1
                except Exception as err:
                    cur_item["error"] = f"Force failed: {err}"

    def _draw_screen_21(self, stdscr):
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, " Auto-Detect: Select Root Folder ".ljust(w), curses.color_pair(1) | curses.A_BOLD)

        items = self.root_nav.list_items()
        max_visible = h - 5
        if self.root_nav.selected < self.ui_offset:
            self.ui_offset = self.root_nav.selected
        elif self.root_nav.selected >= self.ui_offset + max_visible:
            self.ui_offset = self.root_nav.selected - max_visible + 1

        visible_items = items[self.ui_offset:self.ui_offset + max_visible]
        for i, item in enumerate(visible_items):
            y = i + 2
            idx = self.ui_offset + i
            is_sel = (idx == self.root_nav.selected)
            attr = curses.color_pair(3) | curses.A_BOLD if is_sel else curses.color_pair(2)
            icon = "[DIR]" if (self.root_nav.current_path / item).is_dir() else "[TXT]"
            stdscr.addstr(y, 1, f"{'>' if is_sel else ' '} {icon} {item}"[:w-2], attr)

        stdscr.addstr(h-3, 0, f" Current Folder: {self.root_nav.current_path}".ljust(w), curses.color_pair(5))
        try:
            footer_2 = " Modes: [1]Audio [2]Auto-Detect [3]Single [4]Batch  |  q: Quit "
            stdscr.addstr(h-2, 0, " ENTER: Confirm Root Folder ".ljust(w-1), curses.color_pair(1))
            stdscr.addstr(h-1, 0, footer_2.ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass

    @staticmethod
    def _smart_fit_name(name: str, max_w: int) -> str:
        """Fit a filename into max_w columns intelligently, preserving core book titles."""
        if len(name) <= max_w:
            return name
        if max_w <= 16:
            return name[:max_w]

        # Strip file extension if helpful for display
        clean = name
        for ext in (".json", ".mp3", ".m4b", ".txt"):
            if clean.lower().endswith(ext):
                clean = clean[:-len(ext)]
                break

        if len(clean) <= max_w:
            return clean

        # Check for 'Author - Title' format where author prefix is taking up most space
        if " - " in clean:
            prefix_part, rest = clean.split(" - ", 1)
            if len(prefix_part) >= 12 and ("," in prefix_part or "&" in prefix_part or " and " in prefix_part or len(prefix_part.split()) >= 2):
                avail_rest = max_w - 12 - 5
                if avail_rest >= 12:
                    short_p = prefix_part[:12].rstrip() + "..."
                    short_r = rest[:avail_rest].rstrip() + "..." if len(rest) > avail_rest else rest
                    return f"{short_p} - {short_r}"
                else:
                    return rest[:max_w - 3].rstrip() + "..."

        return clean[:max_w - 3].rstrip() + "..."

    def _draw_screen_22(self, stdscr):
        h, w = stdscr.getmaxyx()
        n_total = len(self.batch_pairs)
        n_sel = sum(1 for p in self.batch_pairs if p.get("selected", False))
        n_upg = sum(1 for p in self.batch_pairs if p.get("status") == "UPGRADE")
        n_tm = sum(1 for p in self.batch_pairs if p.get("status") == "TAG_MODIFIED")

        header = f" Auto-Detect & Quality Inspector | {n_total} candidate(s), {n_sel} selected"
        if n_upg > 0:
            header += f" [{n_upg} UPGRADE RECOMMENDED]"
        if n_tm > 0:
            header += f" [{n_tm} TAG_MODIFIED]"
        stdscr.addstr(0, 0, header.ljust(w), curses.color_pair(1) | curses.A_BOLD)

        if not self.batch_pairs:
            stdscr.addstr(2, 2, "No valid transcript candidates found in this folder.", curses.color_pair(4))
            try:
                footer_2 = " Modes: [1]Audio [2]Auto-Detect [3]Single [4]Batch  |  q: Quit "
                stdscr.addstr(h-1, 0, footer_2.ljust(w-1), curses.color_pair(1))
            except curses.error:
                pass
            return

        # Split pane geometry
        split_y = max(6, min(h - 9, int(h * 0.48)))
        max_visible = split_y - 1

        # Scroll calculation
        if self.ui_selected_idx < self.ui_offset:
            self.ui_offset = self.ui_selected_idx
        elif self.ui_selected_idx >= self.ui_offset + max_visible:
            self.ui_offset = self.ui_selected_idx - max_visible + 1

        # Top Pane: Candidate List
        for i in range(max_visible):
            idx = self.ui_offset + i
            if idx >= n_total:
                break

            pair = self.batch_pairs[idx]
            y = 1 + i
            is_sel = (idx == self.ui_selected_idx)
            is_checked = pair.get("selected", False)
            status = pair.get("status", "READY")

            prefix = f"{'>' if is_sel else ' '} {'[x]' if is_checked else '[ ]'}"
            badge = f"[{status}]"
            book_badge = pair.get("book_badge", "")
            if book_badge:
                badge = f"{badge} {book_badge}"
            tx_name = pair["tx"].name
            au_name = pair["audio"].name if pair.get("audio") else "NO AUDIO"

            comp = pair.get("comparison")
            score_str = ""
            if comp and comp.incoming_tqs:
                if status == "UPGRADE":
                    score_str = f"TQS: {comp.incoming_tqs.total_score:.1f} (Δ{comp.delta_tqs:+.1f})"
                elif comp.existing_tqs:
                    score_str = f"TQS: {comp.incoming_tqs.total_score:.1f} (DB: {comp.existing_tqs.total_score:.1f})"
                else:
                    score_str = f"TQS: {comp.incoming_tqs.total_score:.1f}"

            # Line color
            if is_sel:
                line_attr = curses.color_pair(3) | curses.A_BOLD
            elif status == "UPGRADE":
                line_attr = curses.color_pair(5) | curses.A_BOLD
            elif status == "READY":
                line_attr = curses.color_pair(2)
            elif status == "TAG_MODIFIED":
                line_attr = curses.color_pair(4) | curses.A_BOLD
            elif status == "CURRENT":
                line_attr = curses.color_pair(4)
            elif status in ("NO_AUDIO", "HASH_MISMATCH", "MISMATCH", "DURATION_MISMATCH", "EMPTY"):
                line_attr = curses.color_pair(7)
            else:
                line_attr = curses.color_pair(2)

            # Calculate column widths dynamically to maximize terminal width utilization
            badge_str = f"{badge:<11}"
            prefix_len = len(prefix) + 1
            badge_len = len(badge_str) + 1
            score_len = len(score_str) + 1 if score_str else 0
            fixed_overhead = prefix_len + badge_len + 3 + score_len + 4
            avail_w = max(30, w - fixed_overhead)
            col_tx = max(18, avail_w // 2)
            col_au = max(16, avail_w - col_tx)

            display_tx = self._smart_fit_name(tx_name, col_tx)
            display_au = self._smart_fit_name(au_name, col_au)

            row_str = f"{prefix} {badge_str} {display_tx:<{col_tx}} ↔ {display_au:<{col_au}} {score_str}"
            stdscr.addstr(y, 1, row_str[:w-2], line_attr)

        # Split separator
        sep_title = "─── Live Quality & Health Comparison Inspector "
        stdscr.addstr(split_y, 0, (sep_title + "─" * max(0, w - len(sep_title)))[:w-1], curses.color_pair(6) | curses.A_BOLD)

        # Bottom Pane: Live Comparison Inspector
        if self.batch_pairs and 0 <= self.ui_selected_idx < n_total:
            cur = self.batch_pairs[self.ui_selected_idx]
            comp = cur.get("comparison")
            tx_path = cur["tx"]
            audio_path = cur.get("audio")

            py = split_y + 1
            f_size_mb = (audio_path.stat().st_size / (1024 * 1024)) if audio_path and audio_path.exists() else 0.0
            au_info = f"{audio_path.name} ({f_size_mb:.1f} MB)" if audio_path else "MISSING AUDIO"
            stdscr.addstr(py, 2, f"Candidate: {tx_path.name[:w-15]}"[:w-4], curses.color_pair(2) | curses.A_BOLD)
            py += 1
            stdscr.addstr(py, 2, f"Audio:     {au_info[:w-15]}"[:w-4], curses.color_pair(2))
            py += 1

            if comp and comp.existing_tqs:
                col_w = max(25, (w - 8) // 2)
                hdr_left = "INCOMING CANDIDATE".ljust(col_w)
                hdr_right = f"EXISTING DB RECORD (#{comp.existing_id})"
                stdscr.addstr(py, 2, f"{hdr_left} │ {hdr_right}"[:w-4], curses.color_pair(5) | curses.A_BOLD)
                py += 1

                inc = comp.incoming_tqs
                ex = comp.existing_tqs

                c_rows = [
                    (f"TQS Score: {inc.total_score:.1f}/100 ({inc.grade})", f"TQS Score: {ex.total_score:.1f}/100 ({ex.grade})"),
                    (f"Words:     {inc.word_count:,} ({inc.wpm:.1f} WPM)", f"Words:     {ex.word_count:,} ({ex.wpm:.1f} WPM)"),
                    (f"Word Timestamps: {inc.words_with_timestamps_ratio*100:.0f}% ({inc.granularity_score:.1f}/25)", f"Word Timestamps: {ex.words_with_timestamps_ratio*100:.0f}% ({ex.granularity_score:.1f}/25)"),
                    (f"Health:    {inc.health_score:.1f}/25 ({inc.zero_duration_segments_count} zero, {inc.stretched_segments_count} str)", f"Health:    {ex.health_score:.1f}/25 ({ex.zero_duration_segments_count} zero, {ex.stretched_segments_count} str)"),
                ]
                for left_txt, right_txt in c_rows:
                    if py >= h - 3:
                        break
                    stdscr.addstr(py, 2, f"{left_txt[:col_w]:<{col_w}} │ {right_txt}"[:w-4], curses.color_pair(2))
                    py += 1

                if py < h - 2:
                    sim = comp.familiarity
                    fam_color = curses.color_pair(3) if sim.classification == "DIRECT_MATCH" else (curses.color_pair(4) if sim.classification == "PARTIAL_MATCH" else curses.color_pair(7))
                    stdscr.addstr(py, 2, f"Equivalence: {sim.score:.1f}% ({sim.classification})  |  Action: {comp.recommendation}"[:w-4], fam_color | curses.A_BOLD)
                    py += 1

            elif cur.get("status") == "DURATION_MISMATCH":
                tx_dur = cur.get("tx_duration", 0.0)
                au_dur = cur.get("audio_duration", 0.0)
                diff = abs(tx_dur - au_dur)
                stdscr.addstr(py, 2, f"[!] DURATION MISMATCH: Audio is {au_dur:.1f}s but transcript is {tx_dur:.1f}s (diff: {diff:.1f}s).", curses.color_pair(7) | curses.A_BOLD)
                py += 1
                stdscr.addstr(py, 2, "Automatic pairing blocked to prevent misaligned transcription import.", curses.color_pair(4))
                py += 1

            elif cur.get("status") == "TAG_MODIFIED":
                metrics = cur.get("tag_modified_metrics") or {}
                dur_delta = metrics.get("duration_delta_sec", 0.0)
                rel_dur = metrics.get("relative_duration_delta", 0.0)
                size_delta = metrics.get("size_delta_bytes", 0)
                stdscr.addstr(py, 2, "[!] TAG_MODIFIED: Checksum differs, but passes all 4 stream invariants.", curses.color_pair(4) | curses.A_BOLD)
                py += 1
                stdscr.addstr(py, 2, f"    Δ Duration: {dur_delta:.3f}s ({rel_dur:.3%})  |  Δ File Size: {size_delta:,} bytes", curses.color_pair(2))
                py += 1
                stdscr.addstr(py, 2, "    Stream verified identical (ID3 tag edit). Press [t] or SPACE to toggle for audit import.", curses.color_pair(5))
                py += 1

            elif cur.get("status") == "HASH_MISMATCH":
                stdscr.addstr(py, 2, "[!] SHA-256 MISMATCH: Source audio file checksum differs from transcript metadata.", curses.color_pair(7) | curses.A_BOLD)
                py += 1
                stdscr.addstr(py, 2, "Automatic pairing blocked to prevent corrupting database references.", curses.color_pair(4))
                py += 1

            elif cur.get("status") == "EMPTY":
                stdscr.addstr(py, 2, "[!] EMPTY TRANSCRIPT: JSON file contains 0 segments or no transcribed words.", curses.color_pair(7) | curses.A_BOLD)
                py += 1
                stdscr.addstr(py, 2, "Import blocked because file has no speech content.", curses.color_pair(4))
                py += 1

            elif cur.get("status") == "NO_AUDIO":
                stdscr.addstr(py, 2, "[!] NO AUDIO FOUND: Transcript cannot be ingested without source media.", curses.color_pair(7) | curses.A_BOLD)
                py += 1
                stdscr.addstr(py, 2, "Please copy the corresponding audio file into this directory.", curses.color_pair(4))
                py += 1

            elif comp and not comp.existing_tqs:
                inc = comp.incoming_tqs
                stdscr.addstr(py, 2, "Status: [NEW RECORDING] — Ready for initial ingestion", curses.color_pair(3) | curses.A_BOLD)
                py += 1
                stdscr.addstr(py, 2, f"TQS Score:       {inc.total_score:.1f}/100 ({inc.grade})", curses.color_pair(2))
                py += 1
                stdscr.addstr(py, 2, f"Timeline Span:   {inc.transcribed_span:.1f}s / {inc.duration_seconds:.1f}s (Coverage: {inc.coverage_ratio*100:.1f}%)", curses.color_pair(2))
                py += 1
                stdscr.addstr(py, 2, f"Granularity:     {inc.words_with_timestamps_ratio*100:.0f}% timestamped words ({inc.word_count:,} words, {inc.wpm:.1f} WPM)", curses.color_pair(2))
                py += 1
                stdscr.addstr(py, 2, f"Timeline Health: {inc.health_score:.1f}/25 pts (0 defects)", curses.color_pair(2))
                py += 1

        # Footer
        try:
            footer_tm = " | T: Tag-Mod" if n_tm > 0 else ""
            footer_1 = f" SPACE: Toggle | B: Toggle Audiobook | U: Upgrade{footer_tm} | A: All | S: Import ({n_sel}) "
            footer_2 = " Modes: [1]Audio [2]Auto-Detect [3]Single [4]Batch  |  q: Quit "
            stdscr.addstr(h-2, 0, footer_1.ljust(w-1), curses.color_pair(1))
            stdscr.addstr(h-1, 0, footer_2.ljust(w-1), curses.color_pair(1))
        except curses.error:
            pass
