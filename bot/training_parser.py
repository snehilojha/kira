"""Parser for Stable Baselines 3 training stdout.

Detects timestep counts, mean reward, episode length, and loss values
from SB3's default logging format. Sends checkpoint summaries via
``notifier.send()`` at the interval configured in ``scripts.toml``.

Only active for scripts that have ``checkpoint_interval`` set.
Scripts without it receive raw stdout streaming — parser is bypassed.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime

from bot import db
from bot import notifier

logger = logging.getLogger(__name__)

# --- SB3 stdout patterns ---
# "| time/total_timesteps     | 500000    |"
_RE_TIMESTEPS = re.compile(r"total.?timesteps\s*\|\s*([\d,]+)", re.IGNORECASE)
# "| rollout/ep_rew_mean      | -0.12     |"
_RE_REWARD = re.compile(r"ep_rew_mean\s*\|\s*([-\d.e+]+)", re.IGNORECASE)
# "| rollout/ep_len_mean      | 847       |"
_RE_EP_LEN = re.compile(r"ep_len_mean\s*\|\s*([-\d.e+]+)", re.IGNORECASE)
# "| train/loss               | 0.0023    |"
_RE_LOSS = re.compile(r"train/loss\s*\|\s*([-\d.e+]+)", re.IGNORECASE)


@dataclass
class ParserState:
    """Mutable state for one running training session."""

    alias: str
    checkpoint_interval: int
    start_time: float = field(default_factory=time.time)

    # Tracked metrics — updated as new lines arrive
    total_timesteps: int = 0
    last_checkpoint_steps: int = 0
    reward: float | None = None
    first_reward: float | None = None
    ep_len: float | None = None
    loss: float | None = None

    # Last N stderr lines kept for crash reports
    stderr_tail: list[str] = field(default_factory=list)
    _STDERR_TAIL_SIZE: int = 10


def create_state(alias: str, checkpoint_interval: int) -> ParserState:
    """Create a fresh parser state for a new training run."""
    return ParserState(alias=alias, checkpoint_interval=checkpoint_interval)


async def check_line(line: str, state: ParserState) -> None:
    """Parse a single output line and send a checkpoint summary if due.

    Called by executor.py for every stdout/stderr line when
    ``checkpoint_interval`` is configured for the script.
    """
    # Track stderr for crash reports
    if line.startswith("[stderr]"):
        state.stderr_tail.append(line)
        if len(state.stderr_tail) > state._STDERR_TAIL_SIZE:
            state.stderr_tail.pop(0)
        return

    # Try to extract metrics
    m = _RE_TIMESTEPS.search(line)
    if m:
        state.total_timesteps = int(m.group(1).replace(",", ""))

    m = _RE_REWARD.search(line)
    if m:
        val = float(m.group(1))
        if state.first_reward is None:
            state.first_reward = val
        state.reward = val

    m = _RE_EP_LEN.search(line)
    if m:
        state.ep_len = float(m.group(1))

    m = _RE_LOSS.search(line)
    if m:
        state.loss = float(m.group(1))

    # Check if we crossed a checkpoint boundary
    if state.total_timesteps > 0 and state.checkpoint_interval > 0:
        next_checkpoint = state.last_checkpoint_steps + state.checkpoint_interval
        if state.total_timesteps >= next_checkpoint:
            state.last_checkpoint_steps = (
                state.total_timesteps
                // state.checkpoint_interval
                * state.checkpoint_interval
            )
            await _send_summary(state, final=False)


async def on_finish(state: ParserState, timed_out: bool = False) -> None:
    """Send a final training summary when the script exits cleanly or times out."""
    label = "timed out" if timed_out else "finished"
    exit_code = 0 if not timed_out else -1
    await _log_run_to_db(state, exit_code)
    await _send_summary(state, final=True, label=label)


async def on_crash(state: ParserState, exit_code: int) -> None:
    """Send a crash report with the last stderr lines."""
    await _log_run_to_db(state, exit_code)
    elapsed = _format_elapsed(time.time() - state.start_time)
    stderr_block = "\n".join(state.stderr_tail) if state.stderr_tail else "(no stderr captured)"
    msg = (
        f"❌ {state.alias} — crashed (exit code {exit_code})\n"
        f"  Steps:   {state.total_timesteps:,}\n"
        f"  Elapsed: {elapsed}\n\n"
        f"Last stderr:\n{stderr_block}"
    )
    await notifier.send(msg)


async def _send_summary(state: ParserState, final: bool = False, label: str = "checkpoint") -> None:
    """Format and send a checkpoint or final summary."""
    elapsed = _format_elapsed(time.time() - state.start_time)

    # Estimate ETA based on progress rate
    eta_str = ""
    if not final and state.total_timesteps > 0 and state.start_time:
        elapsed_secs = time.time() - state.start_time
        if elapsed_secs > 0:
            steps_per_sec = state.total_timesteps / elapsed_secs
            eta_str = f"  Rate:      {steps_per_sec:,.0f} steps/s\n"

    reward_line = ""
    if state.reward is not None:
        arrow = ""
        if state.first_reward is not None and state.first_reward != state.reward:
            direction = "↑" if state.reward > state.first_reward else "↓"
            arrow = f" ({state.first_reward:.4f} → {state.reward:.4f} {direction})"
        reward_line = f"  Reward:    {state.reward:.4f}{arrow}\n"

    ep_len_line = f"  Ep length: {state.ep_len:.0f} avg\n" if state.ep_len is not None else ""
    loss_line = f"  Loss:      {state.loss}\n" if state.loss is not None else ""

    # Proactive comparison against the previous successful run.
    comparison_line = ""
    if final:
        comparison_line = await _build_comparison_line(state)

    icon = "✅" if final and label == "finished" else "⏰" if final and label == "timed out" else "✓"
    header = f"{icon} {state.alias} — {label} @ {state.total_timesteps:,} steps"

    msg = f"{header}\n{reward_line}{ep_len_line}{loss_line}  Elapsed:   {elapsed}\n{eta_str}{comparison_line}"
    await notifier.send(msg.rstrip())


async def _log_run_to_db(state: ParserState, exit_code: int) -> None:
    """Persist the run's final metrics to the database."""
    runtime = time.time() - state.start_time
    started_at = datetime.fromtimestamp(state.start_time).isoformat()
    finished_at = datetime.now().isoformat()
    try:
        await db.log_run(
            alias=state.alias,
            started_at=started_at,
            finished_at=finished_at,
            exit_code=exit_code,
            runtime_seconds=runtime,
            total_timesteps=state.total_timesteps or None,
            reward=state.reward,
            ep_len=state.ep_len,
            loss=state.loss,
        )
    except Exception:
        logger.warning("Failed to log run to DB for %s", state.alias, exc_info=True)


async def _build_comparison_line(state: ParserState) -> str:
    """Compare current run metrics against the previous successful run."""
    try:
        prev = await db.get_previous_run_metrics(state.alias)
    except Exception:
        return ""
    if prev is None:
        return ""

    parts: list[str] = []
    prev_reward = prev.get("reward")
    if prev_reward is not None and state.reward is not None:
        delta = state.reward - prev_reward
        if prev_reward != 0:
            pct = (delta / abs(prev_reward)) * 100
            direction = "↑" if delta > 0 else "↓"
            parts.append(f"reward {prev_reward:.4f}→{state.reward:.4f} {direction}{abs(pct):.0f}%")
        else:
            parts.append(f"reward {prev_reward:.4f}→{state.reward:.4f}")

    prev_loss = prev.get("loss")
    if prev_loss is not None and state.loss is not None:
        parts.append(f"loss {prev_loss}→{state.loss}")

    if not parts:
        return ""
    return f"  vs prev:   {', '.join(parts)}\n"


def _format_elapsed(seconds: float) -> str:
    """Format seconds into a human-readable ``Xh Ym`` string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"
