"""Two-phase authentication for AudioBench.

Architecture
------------
The security layer uses two credentials stored as bcrypt hashes in
``data/security/security.json``. The database itself is NOT encrypted by
this module (SQLCipher is a separate, deferred change). This module gates
REPL session tiers only.

Daily Passphrase
    Used on every REPL launch and for ``\\unlock``.
    bcrypt work factor 12 → ~250 ms/check, brute-force unfeasible.

Recovery Key
    An episodic mnemonic chosen at setup time (a quote, a date, a phrase
    only you know). Used ONLY when the daily passphrase is forgotten.
    Invoking it forces an immediate in-place passphrase reset before the
    session opens — no grace period, no bypassing.

Design Decisions
----------------
- ``getpass.getpass()`` is used everywhere input is sensitive. It suppresses
  echo and does not reveal input length.
- All file I/O is atomic (write to tmp → fsync → rename) so a crash mid-write
  cannot corrupt the only copy of the hashes.
- The module is stateless. Every call reads from disk. No module-level
  globals. Session tier state lives in ``ReplSession``, not here.
- Fail-fast: if ``security.json`` is corrupted or missing required fields,
  ``is_configured()`` returns False and the REPL triggers first-run setup.

Usage
-----
    from audiobench.security.auth import is_configured, first_run_setup, verify_passphrase

    if not is_configured():
        first_run_setup()
    else:
        if not verify_passphrase(getpass.getpass("Passphrase: ")):
            raise SystemExit(1)
"""

from __future__ import annotations

import getpass
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import bcrypt

from audiobench.core.logger_factory import get_logger
from audiobench.core.settings import get_settings

logger = get_logger("security.auth")

# ── Constants ──────────────────────────────────────────────────────────────

_BCRYPT_ROUNDS = 12          # ~250 ms on modern hardware — brute-force barrier
_MAX_LOGIN_ATTEMPTS = 3
_SECURITY_FILE = "security.json"


# ── Internal helpers ───────────────────────────────────────────────────────

def _security_path() -> Path:
    """Absolute path to security.json."""
    d = get_settings().security_dir
    d.mkdir(parents=True, exist_ok=True)
    return d / _SECURITY_FILE


def _load() -> dict[str, Any]:
    """Load security.json. Returns empty dict if missing or unreadable."""
    p = _security_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read security.json: %s", exc)
        return {}


def _save(data: dict[str, Any]) -> None:
    """Atomically write security.json.

    Writes to a temp file in the same directory, fsyncs, then renames.
    A crash mid-write cannot leave a half-written file.
    """
    p = _security_path()
    parent = p.parent
    fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, p)                 # atomic on POSIX
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _hash(plaintext: str) -> str:
    """Return a bcrypt hash string for the given plaintext."""
    return bcrypt.hashpw(
        plaintext.encode("utf-8"),
        bcrypt.gensalt(rounds=_BCRYPT_ROUNDS),
    ).decode("utf-8")


def _check(plaintext: str, stored_hash: str) -> bool:
    """Constant-time bcrypt comparison. Returns False on any error."""
    try:
        return bcrypt.checkpw(
            plaintext.encode("utf-8"),
            stored_hash.encode("utf-8"),
        )
    except Exception:
        return False


# ── Public API ─────────────────────────────────────────────────────────────

def is_configured() -> bool:
    """Return True if both passphrase and recovery key hashes are present."""
    data = _load()
    return bool(data.get("passphrase_hash")) and bool(data.get("recovery_key_hash"))


def first_run_setup() -> None:
    """Interactive first-run wizard: set daily passphrase and recovery key.

    Called once on first REPL launch when ``is_configured()`` is False.
    All input is collected via ``getpass.getpass`` — characters not echoed,
    length not revealed.
    """
    from audiobench.cli.display.theme import ACCENT, DIM, SUCCESS, WARNING, console

    console.print("\n  [bold]AudioBench — First-time Security Setup[/bold]")
    console.print(f"  [{DIM}]Your cognitive archive requires a passphrase.[/]")
    console.print(f"  [{DIM}]This gates access to your intimate memories.\n[/]")

    # ── Daily passphrase ──
    console.print(f"  [{ACCENT}]Daily Passphrase[/] [{DIM}](used on every launch and \\unlock)[/]")
    while True:
        pw = getpass.getpass("  > Passphrase: ")
        if len(pw) < 8:
            console.print(f"  [{WARNING}]Too short — minimum 8 characters.[/]")
            continue
        pw2 = getpass.getpass("  > Confirm:    ")
        if pw != pw2:
            console.print(f"  [{WARNING}]Passphrases do not match. Try again.[/]")
            continue
        break

    # ── Recovery key ──
    console.print(f"\n  [{ACCENT}]Recovery Key[/] [{DIM}](used only if you forget your passphrase)[/]")
    console.print(f"  [{DIM}]Use a personal mnemonic — a quote, a date, something only you know.[/]")
    console.print(f"  [{DIM}]It will NOT be stored in plain text.\n[/]")
    while True:
        rk = getpass.getpass("  > Recovery Key: ")
        if len(rk) < 12:
            console.print(f"  [{WARNING}]Too short — minimum 12 characters for a recovery key.[/]")
            continue
        rk2 = getpass.getpass("  > Confirm:      ")
        if rk != rk2:
            console.print(f"  [{WARNING}]Recovery keys do not match. Try again.[/]")
            continue
        break

    # ── Persist ──
    import datetime
    _save({
        "passphrase_hash": _hash(pw),
        "recovery_key_hash": _hash(rk),
        "configured_at": datetime.datetime.now(datetime.UTC).isoformat(),
    })

    console.print(f"\n  [{SUCCESS}]✓ Passphrase set.[/]")
    console.print(f"  [{SUCCESS}]✓ Recovery key set.[/]")
    console.print(
        f"  [yellow]→ Voiceprint: Not yet enrolled. "
        f"Run [{ACCENT}]\\enroll <audio>[/] after setup.[/]\n"
    )


def verify_passphrase(passphrase: str) -> bool:
    """Return True if passphrase matches the stored daily passphrase hash."""
    data = _load()
    stored = data.get("passphrase_hash", "")
    if not stored:
        return False
    return _check(passphrase, stored)


def verify_recovery_key(key: str) -> bool:
    """Return True if key matches the stored recovery key hash."""
    data = _load()
    stored = data.get("recovery_key_hash", "")
    if not stored:
        return False
    return _check(key, stored)


def reset_passphrase(new_passphrase: str) -> None:
    """Overwrite the daily passphrase hash.

    Called after a successful recovery key use. The recovery key hash
    is intentionally kept — you can recover again if needed.
    """
    data = _load()
    data["passphrase_hash"] = _hash(new_passphrase)
    import datetime
    data["reset_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    _save(data)
    logger.info("Daily passphrase reset.")


def use_recovery_key() -> bool:
    """Interactive recovery flow: verify recovery key then force passphrase reset.

    No grace period. The reset happens in-place before the session opens.
    Returns True if recovery succeeded and a new passphrase was set.
    """
    from audiobench.cli.display.theme import DIM, SUCCESS, WARNING, console

    console.print(f"\n  [{WARNING}]Recovery mode. Enter your recovery key.[/]")
    console.print(f"  [{DIM}]If correct, you will be prompted to set a new passphrase immediately.[/]\n")

    rk = getpass.getpass("  > Recovery Key: ")
    if not verify_recovery_key(rk):
        console.print(f"  [{WARNING}]Incorrect recovery key.[/]")
        return False

    console.print(f"  [{SUCCESS}]✓ Recovery key accepted.[/]")
    console.print(f"  [{DIM}]You must set a new daily passphrase now.[/]\n")

    while True:
        pw = getpass.getpass("  > New Passphrase: ")
        if len(pw) < 8:
            console.print(f"  [{WARNING}]Too short — minimum 8 characters.[/]")
            continue
        pw2 = getpass.getpass("  > Confirm:        ")
        if pw != pw2:
            console.print(f"  [{WARNING}]Passphrases do not match. Try again.[/]")
            continue
        break

    reset_passphrase(pw)
    console.print(f"  [{SUCCESS}]✓ New passphrase set. Continuing...[/]\n")
    return True


def run_startup_gate() -> bool:
    """Run the full REPL startup authentication gate.

    Handles first-run setup, 3-attempt login, and recovery key fallback.
    Returns True if authentication succeeded, False if it failed fatally.

    This is the single entry point called from the REPL __init__.
    """
    from audiobench.cli.display.theme import ACCENT, DIM, WARNING, console

    if not is_configured():
        first_run_setup()
        return True

    console.print(f"\n  [bold]AudioBench[/bold]  [{DIM}]Cognitive Archive[/]")

    for attempt in range(1, _MAX_LOGIN_ATTEMPTS + 1):
        pw = getpass.getpass("  Passphrase: ")
        if verify_passphrase(pw):
            _print_session_banner()
            return True

        remaining = _MAX_LOGIN_ATTEMPTS - attempt
        if remaining > 0:
            console.print(
                f"  [{WARNING}]Incorrect.[/] [{DIM}]{remaining} attempt(s) remaining.[/]"
            )
        else:
            console.print(f"  [{DIM}]Forgot your passphrase? Type [{ACCENT}]recover[/] to use your recovery key.[/]")
            answer = input("  → ").strip().lower()
            if answer == "recover":
                return use_recovery_key()

    console.print(f"  [{WARNING}]Too many failed attempts. Exiting.[/]")
    return False


def _print_session_banner() -> None:
    """Print the post-auth summary line."""
    from audiobench.cli.display.theme import DIM, SUCCESS, console
    from audiobench.security.voiceprint import get_enrollment_summary

    summary = get_enrollment_summary()
    console.print(f"  [{SUCCESS}]✓ Authenticated — Session Tier 0[/]")
    if summary:
        console.print(f"  [{DIM}]● Voiceprint: {summary}[/]")
    console.print()
