"""In-memory registry of running subprocesses.

Stores ``{pid: info}`` so that ``/status`` and ``/kill`` can inspect and
manage every process the bot has launched.  Entries do not survive a
bot restart — this is intentional (see spec).
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProcessInfo:
    """Metadata for a tracked subprocess."""

    process: asyncio.subprocess.Process
    alias: str
    start_time: float = field(default_factory=time.time)


# Module-level mutable state — accessed from executor and handlers
REGISTRY: dict[int, ProcessInfo] = {}


def register(pid: int, process: asyncio.subprocess.Process, alias: str) -> None:
    """Add a subprocess to the registry."""
    REGISTRY[pid] = ProcessInfo(process=process, alias=alias)
    logger.info("Registered process pid=%d alias=%s", pid, alias)


def deregister(pid: int) -> None:
    """Remove a subprocess from the registry (called on finish or kill)."""
    entry = REGISTRY.pop(pid, None)
    if entry:
        logger.info("Deregistered process pid=%d alias=%s", pid, entry.alias)


def list_processes() -> list[dict]:
    """Return a snapshot of all tracked processes.

    Each dict contains: pid, alias, runtime_seconds, returncode.
    """
    now = time.time()
    result = []
    for pid, info in REGISTRY.items():
        result.append({
            "pid": pid,
            "alias": info.alias,
            "runtime_seconds": round(now - info.start_time, 1),
            "returncode": info.process.returncode,
        })
    return result


async def kill(pid: int) -> str:
    """Terminate a tracked process by PID.

    Returns:
        A human-readable status string.
    """
    info = REGISTRY.get(pid)
    if info is None:
        return f"No tracked process with PID {pid}."
    try:
        info.process.terminate()
        # Give the process a moment to exit gracefully
        await asyncio.sleep(0.5)
        if info.process.returncode is None:
            info.process.kill()
        deregister(pid)
        return f"Killed process PID {pid} ({info.alias})."
    except ProcessLookupError:
        deregister(pid)
        return f"Process PID {pid} already exited."
    except OSError as exc:
        return f"Failed to kill PID {pid}: {exc}"
