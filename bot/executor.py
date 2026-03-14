"""Async subprocess runner with real-time stdout/stderr streaming.

Creates subprocesses via ``asyncio.create_subprocess_exec``, enforces
timeouts, registers them in the process registry, and optionally feeds
output lines to the training parser for checkpoint detection.
"""

import asyncio
import logging
from typing import AsyncGenerator

from bot import process_registry
from bot import training_parser

logger = logging.getLogger(__name__)

# Telegram message limit is 4096; we cap at 4000 to leave headroom
_MAX_CHUNK = 4000


async def run_command(
    interpreter: str,
    script_path: str,
    args: list[str] | None = None,
    timeout: int = 30,
    alias: str = "unknown",
    checkpoint_interval: int | None = None,
) -> AsyncGenerator[str, None]:
    """Launch a subprocess and yield stdout/stderr lines as they arrive.

    Args:
        interpreter: Path to the Python (or other) interpreter.
        script_path: Path to the script to execute.
        args: Extra CLI arguments to pass to the script.
        timeout: Maximum wall-clock seconds before the process is killed.
        alias: Human-readable name for registry and logging.
        checkpoint_interval: If set, training_parser will emit summaries
            every N timesteps.  ``None`` means raw streaming only.

    Yields:
        Chunks of combined stdout/stderr text, each ≤ 4000 characters.
    """
    cmd = [interpreter, script_path] + (args or [])
    logger.info("Executing: %s (timeout=%ds)", " ".join(cmd), timeout)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    pid = proc.pid
    process_registry.register(pid, proc, alias)

    # Initialise parser state if checkpoint tracking is requested
    parser_state = None
    if checkpoint_interval is not None:
        parser_state = training_parser.create_state(alias, checkpoint_interval)

    buffer = ""
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _reader(stream: asyncio.StreamReader, label: str) -> None:
        """Read lines from a single stream and push them into the shared queue."""
        while True:
            line_bytes = await stream.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\n\r")
            if label == "stderr":
                await queue.put(f"[stderr] {line}")
            else:
                await queue.put(line)
        # Signal this reader is done
        await queue.put(None)

    async def _stream_output() -> AsyncGenerator[str, None]:
        """Merge stdout and stderr via queue, yield chunked output."""
        nonlocal buffer

        # Start both readers concurrently
        stdout_task = asyncio.create_task(_reader(proc.stdout, "stdout"))
        stderr_task = asyncio.create_task(_reader(proc.stderr, "stderr"))
        streams_done = 0

        while streams_done < 2:
            item = await queue.get()
            if item is None:
                streams_done += 1
                continue

            # Feed line to training parser if active
            if parser_state is not None:
                await training_parser.check_line(item, parser_state)

            buffer += item + "\n"
            if len(buffer) >= _MAX_CHUNK:
                yield buffer[:_MAX_CHUNK]
                buffer = buffer[_MAX_CHUNK:]

        # Wait for reader tasks to fully finish
        await stdout_task
        await stderr_task

        # Flush remaining buffer
        if buffer:
            yield buffer[:_MAX_CHUNK]
            buffer = ""

    try:
        timed_out = False
        deadline = asyncio.get_event_loop().time() + timeout

        async for chunk in _stream_output():
            yield chunk
            if asyncio.get_event_loop().time() > deadline:
                timed_out = True
                break

        if timed_out:
            proc.kill()
            await proc.wait()
            process_registry.deregister(pid)
            yield f"\n⏰ Process timed out after {timeout}s and was killed."
            if parser_state is not None:
                await training_parser.on_finish(parser_state, timed_out=True)
            return

        await proc.wait()

    except Exception as exc:
        logger.error("Executor error for alias=%s pid=%d: %s", alias, pid, exc)
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        yield f"\n❌ Internal error: {exc}"
    finally:
        process_registry.deregister(pid)

    exit_code = proc.returncode
    if exit_code == 0:
        yield f"\n✅ Process exited with code 0."
        if parser_state is not None:
            await training_parser.on_finish(parser_state, timed_out=False)
    else:
        yield f"\n❌ Process exited with code {exit_code}."
        if parser_state is not None:
            await training_parser.on_crash(parser_state, exit_code)


async def run_shell(command: str, timeout: int = 30) -> AsyncGenerator[str, None]:
    """Run an arbitrary shell command via ``cmd.exe /c``.

    Yields output chunks the same way as ``run_command``.
    """
    async for chunk in run_command(
        interpreter="cmd.exe",
        script_path="/c",
        args=command.split(),
        timeout=timeout,
        alias=f"shell:{command[:40]}",
        checkpoint_interval=None,
    ):
        yield chunk
