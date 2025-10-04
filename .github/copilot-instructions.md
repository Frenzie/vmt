# Project AI Instructions

These instructions guide AI coding agents contributing to this repository.

## Purpose & Overview
Discord bot that locally transcribes Discord Voice Messages (VMs) to text using `faster-whisper` (tiny model). Core flow:
1. User triggers `vmt transcribe` (optionally replying to a VM) OR sends a new voice message (auto mode).
2. Bot locates target voice message (replied-to or most recent in channel).
3. Audio (OGG/MP3/Discord voice note) is written directly to a temp file and decoded by ffmpeg inside faster-whisper.
4. Temp audio fed to `WhisperModel('tiny')` (fast, lightweight). Transcription done off-thread via executor.
5. Result embedded + posted.

## Key Files
- `src/main.py`: Bootstraps bot, loads config, dynamic cog loader (all `cogs/*.py`). Removes default help.
- `src/cogs/transcribe.py`: Primary logic (message scanning, validation, transcription via faster-whisper, embed formatting).
- `src/cogs/help.py`: Custom help command using config prefix & embed styling.
<!-- Legacy language code cog removed -->
- `src/config/example.config.json`: Schema & example for required runtime config.
- `Dockerfile` + `compose.yaml`: Containerized runtime; mounts `src/config` read‑only so secrets not baked into image.

## Runtime Configuration
Expect a `src/config/config.json` (NOT committed) shaped like example. Fields actually read:
- `token`
- `prefix` (command prefix e.g. `vmt ` including trailing space if desired)

## Commands & Behaviors
- `help` (`h`): Custom embed.
- `transcribe` aliases: `t`.
  - If no reply context, searches channel history newest→oldest (limit 50) for first voice message not authored by bot.
- Passive listener: Any incoming voice message triggers auto transcription (adds ⌛ reaction, later removed).

## Voice Message Detection
`msg_has_voice_note(msg)` currently checks (unchanged legacy heuristic):
```python
if not msg.attachments or not msg.flags.value >> 13:
    return False
```
This relies on bit-shifting of internal flag integer to infer voice message; treat as brittle. Prefer preserving existing helper if modifying.

## Error Handling Patterns
- Minimal: prints exceptions to stdout (no logging framework).
- User-facing failures: short failure message ("Failed to transcribe" / "Could not transcribe").
- Empty transcription returns placeholder `(empty transcription)`.

## External Dependencies
- `faster-whisper` small model for on-device transcription (CPU-friendly; auto chooses compute type).
- `discord.py` for bot events/commands.
- (Legacy) DeepL & SpeechRecognition removed from active code; may purge from dependencies if fully deprecated.

## Embeds & Attribution
Embeds use `discord.Color.og_blurple()`; transcription wrapped in triple backticks; footer shows requester when command-invoked.

## Dynamic Cog Loading
`setup_hook` in `Bot` iterates `os.listdir("cogs")` and loads each `*.py` as `cogs.<name>`. To add features: create new file in `src/cogs/` with `async def setup(bot): await bot.add_cog(NewCog(bot))`.

## Docker / Local Workflow
Local:
```bash
pip install -r requirements.txt
python src/main.py   # run from repo root OR cd src && python main.py (expects working dir containing cogs/ & config/)
```
Container:
```bash
docker compose -f compose.yaml up --build
```
Provide `src/config/config.json` on host; compose mounts `./src/config` read-only into container.

## Conventions / Things to Preserve
- Simple procedural style; no custom logging layer.
- Each cog re-reads config via `load_config()` (no centralized cache). If optimizing, ensure hot-reload semantics preserved.
- Avoid adding blocking operations inside event handlers; use async patterns if adding network calls.

## Safe Extension Examples
Add a new admin-only command pattern:
```python
@commands.command()
@commands.has_permissions(manage_guild=True)
async def ping_api(self, ctx):
    await ctx.reply("pong")
```
Place inside a new cog file; rely on existing loader.

## Potential Improvement Targets (Only implement if asked)
- Replace brittle voice note flag check with official attribute if exposed by discord.py.
- Centralize config cache (current repeated disk reads per cog).
- Structured logging (e.g., `logging` with levels) instead of raw prints.
- Graceful model load fallback / lazy-load on first use to reduce startup time.
- (Legacy translation artifacts removed.)

## When Unsure
Mirror existing patterns & simplicity; avoid adding blocking synchronous operations in the event loop. Ask before reintroducing translation or removing legacy code.
