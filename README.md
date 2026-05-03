# KIRA

KIRA is a local-first personal AI agent for Windows with Telegram control, optional always-on local voice, system awareness, proactive notifications, and a guarded complex-task runtime.

It is designed for real day-to-day operator workflows: launching automations, checking system state, running monitored jobs, handling reminders/schedules, and escalating only when action approval is needed.

## What KIRA Can Do

- Control scripts and shell workflows safely from Telegram.
- Run and monitor jobs with pause/resume/cancel lifecycle controls.
- Route requests between deterministic commands and complex AI analysis.
- Handle local voice triggers (hotkey, wake-word, enter-to-talk).
- Provide proactive updates from observer and world context loops.
- Maintain conversation and run history in a local DB.
- Gate risky actions with explicit confirmation callbacks.

## Runtime Surfaces

- `python -m bot.main`: Telegram + background services + overlay runtime.
- `python -m bot.local_voice`: standalone local voice runtime.
- `python -m mcp.server`: local MCP endpoint for IDE/tool integrations.

## Docs Map

Detailed architecture and subsystem docs live in [`docs/README.md`](./docs/README.md):

- [`docs/architecture.md`](./docs/architecture.md)
- [`docs/providers-and-env.md`](./docs/providers-and-env.md)
- [`docs/telegram-and-voice.md`](./docs/telegram-and-voice.md)
- [`docs/complex-runtime.md`](./docs/complex-runtime.md)
- [`docs/operations.md`](./docs/operations.md)
- [`docs/testing.md`](./docs/testing.md)

## Quick Start

### 1. Install dependencies

```bash
cd kira
pip install -r requirements.txt
```

Complex-task execution also uses `astra_node` in this repo setup. Install it in your environment if you use `/ask` complex flows:

```bash
pip install -e d:/AI_tools/astra/packages/astra-node
```

### 2. Configure `.env`

Minimum setup:

```env
BOT_TOKEN=your_telegram_bot_token
ALLOWED_USER_IDS=123456789
CHAT_ID=123456789
OPENAI_API_KEY=your_openai_key
LOG_LEVEL=INFO
DEFAULT_TIMEOUT=30
```

Provider overrides:

- Chat: `KIRA_API_KEY`, `KIRA_API_BASE_URL`, `KIRA_FAST_MODEL`, `KIRA_SMART_MODEL`
- Vision: `KIRA_VISION_API_KEY`, `KIRA_VISION_BASE_URL`, `KIRA_VISION_MODEL`
- STT: `KIRA_STT_API_KEY`, `KIRA_STT_BASE_URL`, `KIRA_VOICE_TRANSCRIBE_MODEL`
- TTS: `KIRA_TTS_API_KEY`, `KIRA_TTS_BASE_URL`, `KIRA_TTS_MODEL`, `KIRA_VOICE`

### 3. Configure scripts and apps

- `config/scripts.toml` for `/run`, `/chain`, and scheduler aliases.
- `config/apps.toml` for app launch/close mappings and mode behavior.

### 4. Start KIRA

```bash
python -m bot.main
```

Optional local voice:

```bash
python -m bot.local_voice
```

## Command Surface (Telegram)

### Execution and process control

- `/run <alias> [args...]`
- `/shell <command>`
- `/chain <alias>`
- `/status`
- `/kill <pid>`
- `/sysinfo`

### Scheduling and monitoring

- `/schedule <alias> <HH:MM|Xm|Xh>`
- `/schedules`
- `/unschedule <id>`
- `/watch pid <pid>`
- `/watch file <path>`
- `/watches`
- `/unwatch <id>`
- `/remind <Xm|Xh> <message>`

### Files and desktop helpers

- `/getfile <path>`
- `/putfile [path]`
- `/cd [path]`
- `/ls [path]`
- `/find <pattern> [path]`
- `/tail <path> [n]`
- `/mkdir <path>`
- `/move <src> <dst>`
- `/copy <text>`
- `/paste`
- `/screenshot [monitor]`

### App and power operations

- `/list_apps`
- `/open <app>`
- `/close_apps <app1> [app2]`
- `/sleep`
- `/shutdown <minutes>`
- `/reboot <minutes>`
- `/abort_shutdown`

### Brain, memory, and jobs

- `/ask <request>`
- `/tasks [n]`
- `/task <task_id>`
- `/history [n]`
- `/runs [alias] [n]`
- `/summarise`
- `/reflect`
- `/recall <query>`
- `/jobs`
- `/canceljob <job_id>`
- `/pausejob <job_id>`
- `/resumejob <job_id>`
- `/mode`
- `/help`

## Module Layout (Current)

- `bot/main.py`: runtime startup, PTB wiring, callback routing.
- `bot/handlers.py`: coordinator for `/ask`, callbacks, and voice-message flow.
- `bot/cmd_fs.py`: filesystem commands.
- `bot/cmd_process.py`: execution and power/process commands.
- `bot/cmd_schedule.py`: schedules/watch/reminders.
- `bot/cmd_jobs.py`: brain, memory, history, and job commands.
- `bot/cmd_app.py`: app-control commands.
- `bot/voice_confirm.py`: voice confirmation registry + TTS helper path.
- `bot/brain.py`: complex runtime orchestration and approval semantics.
- `bot/provider.py`: chat/STT/TTS/vision provider resolution.
- `mcp/server.py`: localhost MCP service.

## Safety Model

- Telegram commands are auth-gated by allowed user IDs.
- Unknown users are ignored.
- Destructive shell patterns require explicit confirmation.
- Power actions require explicit confirmation.
- Complex runtime actions can require approval callbacks.
- Local MCP endpoint binds to `127.0.0.1`.

## Current Known Gaps

- `requirements.txt` does not fully express optional runtime extras (notably Astra path in this setup).
- `local_voice.py` and `brain.py` are high-capability modules with ongoing decomposition opportunities.
- Production-hardening is stronger in operator workflow than in fresh-machine reproducibility.

## Contributing

1. Create a feature branch from `main`.
2. Keep command-family logic inside `cmd_*` modules.
3. Update docs in `docs/` when behavior changes.
4. Run tests locally before opening a PR.

## Security Notes

- Never commit secrets or `.env`.
- Rotate keys immediately if exposed in logs/screenshots.
- Keep provider keys scoped per capability when possible.
