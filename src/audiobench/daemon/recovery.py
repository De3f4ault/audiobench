"""
recovery.py — startup recovery shim.

The implementation lives in startup_recovery.py (typed RecoveryStep contract,
structured run() results, per-step logging). This module re-exports
get_startup_recovery() so that server.py's existing import is unchanged.
"""
from audiobench.daemon.startup_recovery import (  # noqa: F401
    RecoveryStep,
    StartupRecovery,
    get_startup_recovery,
)
