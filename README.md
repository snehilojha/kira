# kira

A persistent background service running on your Windows PC that accepts commands from your phone via Telegram, executes Python scripts and shell commands, streams output in real time, and proactively notifies you of events.

## Quick Start

### 1. Install dependencies

```bash
cd kira
pip install -r requirements.txt
```

### 2. Configure `.env`

Edit `.env` with your actual values:

```env
BOT_TOKEN=your_telegram_bot_token_here
ALLOWED_USER_IDS=123456789
CHAT_ID=123456789
LOG_LEVEL=INFO
DEFAULT_TIMEOUT=30
MCP_PORT=8000
```

- **BOT_TOKEN** — Get from [@BotFather](https://t.me/BotFather)
- **ALLOWED_USER_IDS** — Comma-separated Telegram user IDs. Get yours from [@userinfobot](https://t.me/userinfobot)
- **CHAT_ID** — Your personal chat ID (same as your user ID for private chats)

### 3. Configure scripts

Edit `config/scripts.toml` to register your scripts:

```toml
[my_script]
interpreter = "C:/path/to/venv/Scripts/python.exe"
path        = "C:/path/to/script.py"
args        = []
timeout     = 300
chain       = []
```

Optional fields:
- `chain` — List of aliases to run sequentially on success
- `checkpoint_interval` — For SB3 training: send summaries every N timesteps

### 4. Run the bot

```bash
python -m bot.main
```

### 5. (Optional) Register as Windows service

Right-click `autostart/setup_task_scheduler.ps1` → **Run with PowerShell as Administrator**.

### 6. (Optional) Start MCP server for Windsurf

```bash
uvicorn mcp.server:app --host 127.0.0.1 --port 8000
```

## Commands

| Command | Description |
|---|---|
| `/run <alias> [args]` | Run a registered script |
| `/shell <command>` | Run shell command (confirms destructive ops) |
| `/chain <alias>` | Run script + chained scripts |
| `/status` | List running processes |
| `/kill <pid>` | Kill a process |
| `/schedule <alias> <HH:MM\|Xm\|Xh>` | Schedule a run |
| `/schedules` | List pending schedules |
| `/unschedule <id>` | Cancel a schedule |
| `/sysinfo` | CPU, RAM, GPU, disk info |
| `/getfile <path>` | Download a file |
| `/putfile` | Upload a file (reply to attachment) |
| `/ls [path]` | List directory |
| `/find <pattern> [path]` | Find files |
| `/tail <path> [n]` | Tail a file |
| `/mkdir <path>` | Create directory |
| `/move <src> <dst>` | Move file/directory |
| `/copy <text>` | Set clipboard |
| `/paste` | Get clipboard |
| `/screenshot [n]` | Take screenshot |
| `/sleep` | Sleep PC |
| `/shutdown <min>` | Schedule shutdown |
| `/reboot <min>` | Schedule reboot |
| `/abort_shutdown` | Cancel shutdown/reboot |
| `/watch pid <pid>` | Alert when process dies |
| `/watch file <path>` | Alert when file changes |
| `/watches` | List active watchers |
| `/unwatch <id>` | Remove watcher |
| `/remind <Xm\|Xh> <msg>` | Set a reminder |
| `/help` | Show all commands |

## Architecture

- **Long polling** — no public URL needed, works from anywhere
- **asyncio throughout** — executor, scheduler, watchdog, reminders all concurrent
- **notifier.py** — single outbound channel for all proactive messages
- **training_parser.py** — SB3 checkpoint detection, only active when configured
- **process_registry** — in-memory only, honest model for subprocess tracking

## Important Notes

- Schedules and watchdog tasks are **in-memory only** — they do not survive a bot restart.
- Power settings: Set **Never sleep** (when plugged in) in Windows Power Options. Screen off is fine.
- If `BOT_TOKEN` is ever leaked: BotFather → `/revoke` → update `.env` immediately.

## Security

- Every handler requires user ID whitelist — no exceptions
- Unknown users get silent ignore (bot existence not revealed)
- `.env` is gitignored — secrets never committed
- `/shell` confirms destructive commands before executing
- Power commands require inline confirmation
- MCP server binds to 127.0.0.1 only
