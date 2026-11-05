import os
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from audiobench.core.settings import get_settings


def _clone_refs_windows(source: Path, destination: Path) -> bool:
    """Attempt ReFS block cloning on Windows using FSCTL_DUPLICATE_EXTENTS_TO_FILE."""
    try:
        import ctypes
        from ctypes import wintypes
        import struct

        kernel32 = ctypes.windll.kernel32

        GENERIC_READ = 0x80000000
        GENERIC_WRITE = 0x40000000
        FILE_SHARE_READ = 0x00000001
        FILE_SHARE_WRITE = 0x00000002
        OPEN_EXISTING = 3
        CREATE_ALWAYS = 2
        FILE_ATTRIBUTE_NORMAL = 0x80
        FSCTL_DUPLICATE_EXTENTS_TO_FILE = 0x00098344
        INVALID_HANDLE_VALUE = -1

        src_handle = kernel32.CreateFileW(
            str(source),
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            None,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            None,
        )
        if src_handle == INVALID_HANDLE_VALUE:
            return False

        dst_handle = kernel32.CreateFileW(
            str(destination),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            None,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            None,
        )
        if dst_handle == INVALID_HANDLE_VALUE:
            kernel32.CloseHandle(src_handle)
            return False

        try:
            total_size = source.stat().st_size
            cluster_aligned_size = ((total_size + 4095) // 4096) * 4096

            if ctypes.sizeof(ctypes.c_void_p) == 8:
                payload = struct.pack("=QQQQ", src_handle, 0, 0, cluster_aligned_size)
            else:
                payload = struct.pack("=IQQQ", src_handle, 0, 0, cluster_aligned_size)

            bytes_returned = wintypes.DWORD(0)
            res = kernel32.DeviceIoControl(
                dst_handle,
                FSCTL_DUPLICATE_EXTENTS_TO_FILE,
                payload,
                len(payload),
                None,
                0,
                ctypes.byref(bytes_returned),
                None,
            )
            if res != 0:
                kernel32.SetFilePointer(dst_handle, total_size, None, 0)
                kernel32.SetEndOfFile(dst_handle)
                return True
            return False
        finally:
            kernel32.CloseHandle(src_handle)
            kernel32.CloseHandle(dst_handle)
    except Exception:
        return False


class BackgroundImporter:
    def __init__(self, max_workers: int = 4):
        self.settings = get_settings()
        self.library_dir = self.settings.data_dir / "library"
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self._lock = threading.Lock()

        self.progress_callback: Callable[[int, int, str], None] | None = None
        self.file_done_callback: Callable[[Path, Path, str], None] | None = None

    def set_callbacks(
        self,
        progress: Callable[[int, int, str], None] | None = None,
        file_done: Callable[[Path, Path, str], None] | None = None,
    ) -> None:
        """Set callbacks for progress tracking.

        progress(total_bytes, copied_bytes, filename)
        file_done(original_path, new_path, engine)
        """
        self.progress_callback = progress
        self.file_done_callback = file_done

    def _copy_file_streaming(self, source: Path, destination: Path, filename: str) -> tuple[bool, str]:
        """Atomic copy with progress tracking and cross-platform CoW reflink acceleration.

        - macOS: Native in-process APFS clonefile(2).
        - Linux: cp --reflink=auto (Btrfs, XFS, OCFS2, ZFS).
        - Windows: ReFS Block Cloning (FSCTL_DUPLICATE_EXTENTS_TO_FILE) if on ReFS.
        - Fallback: Chunked streaming copy (1MB buffers) with real-time progress callbacks.
        """
        tmp_destination = destination.with_suffix(destination.suffix + ".tmp")
        total_size = source.stat().st_size

        # 1. Attempt instantaneous CoW reflink / clonefile
        # A. macOS APFS native clonefile
        if hasattr(os, "clonefile"):
            try:
                os.clonefile(source, tmp_destination)
                if tmp_destination.exists():
                    tmp_destination.replace(destination)
                    if self.progress_callback:
                        self.progress_callback(total_size, total_size, filename)
                    return True, "reflink"
            except Exception:
                if tmp_destination.exists():
                    tmp_destination.unlink()

        # B. Linux / Unix: cp --reflink=auto
        if sys.platform != "win32":
            try:
                import subprocess
                res = subprocess.run(
                    ["cp", "--reflink=auto", "--preserve=mode,timestamps", str(source), str(tmp_destination)],
                    capture_output=True,
                    check=False,
                )
                if res.returncode == 0 and tmp_destination.exists():
                    tmp_destination.replace(destination)
                    if self.progress_callback:
                        self.progress_callback(total_size, total_size, filename)
                    return True, "reflink"
            except Exception:
                if tmp_destination.exists():
                    tmp_destination.unlink()

        # C. Windows: ReFS Block Cloning
        if sys.platform == "win32":
            try:
                if _clone_refs_windows(source, tmp_destination):
                    tmp_destination.replace(destination)
                    if self.progress_callback:
                        self.progress_callback(total_size, total_size, filename)
                    return True, "reflink"
            except Exception:
                if tmp_destination.exists():
                    tmp_destination.unlink()

        # 2. Fall back to chunked streaming copy with progress callback
        copied = 0
        try:
            with open(source, "rb") as fsrc, open(tmp_destination, "wb") as fdst:
                while True:
                    chunk = fsrc.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    fdst.write(chunk)
                    copied += len(chunk)

                    if self.progress_callback:
                        self.progress_callback(total_size, copied, filename)

            # Atomic rename after successful copy
            tmp_destination.replace(destination)
            return True, "stream_copy"
        except Exception:
            if tmp_destination.exists():
                tmp_destination.unlink()
            return False, "failed"

    def _process_single_import(self, allocation: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single file import (runs in thread pool)."""
        source_path: Path = allocation["file"]
        engine: str = allocation["engine"]
        # Generate True Fast Content Hash
        import hashlib

        from audiobench.core.db_session import get_session
        from audiobench.storage.models import AudioFileRecord

        def compute_fast_hash(fp: Path) -> str:
            hasher = hashlib.md5()
            hasher.update(str(fp.stat().st_size).encode())
            with open(fp, "rb") as f:
                hasher.update(f.read(1024 * 1024))
            return hasher.hexdigest()

        file_hash = compute_fast_hash(source_path)

        # Deduplication Check
        with get_session() as session:
            existing = session.query(AudioFileRecord).filter_by(file_hash=file_hash).first()
            if existing:
                # If duplicate, immediately skip the heavy copy process
                if self.progress_callback:
                    self.progress_callback(
                        source_path.stat().st_size, source_path.stat().st_size, source_path.name
                    )
                res = {
                    "source": source_path,
                    "destination": Path(existing.file_path),
                    "engine": engine,
                    "is_duplicate": True,
                    "file_hash": file_hash,
                    "existing_id": existing.id,
                }
                for k in ("companion_file", "tqs_score"):
                    if k in allocation:
                        res[k] = allocation[k]
                return res

        filename = source_path.name
        destination = self.library_dir / filename

        # Handle filename collisions
        counter = 1
        while destination.exists():
            destination = self.library_dir / f"{source_path.stem}_{counter}{source_path.suffix}"
            filename = destination.name
            counter += 1

        success, copy_method = self._copy_file_streaming(source_path, destination, filename)

        if success:
            if self.file_done_callback:
                self.file_done_callback(source_path, destination, engine)
            res = {
                "source": source_path,
                "destination": destination,
                "engine": engine,
                "is_duplicate": False,
                "file_hash": file_hash,
                "copy_method": copy_method,
            }
            for k in ("companion_file", "tqs_score"):
                if k in allocation:
                    res[k] = allocation[k]
            return res
        return None

    def run_import_jobs(self, allocations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run all imports in parallel using ThreadPoolExecutor."""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_allocation = {
                executor.submit(self._process_single_import, alloc): alloc for alloc in allocations
            }

            for future in as_completed(future_to_allocation):
                try:
                    result = future.result()
                    if result:
                        with self._lock:
                            results.append(result)
                except Exception:
                    pass

        return results
