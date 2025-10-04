# Project AI Instructions

These instructions guide AI coding agents contributing to this repository.

## Purpose & Overview
Discord bot that transcribes Discord Voice Messages (VMs) to text and optionally translates them via DeepL. Core flow:
1. User triggers `vmt transcribe [LANG?]` or replies to a VM.
2. Bot locates target voice message (replied-to or most recent in channel).
3. Audio (OGG/MP3/Discord voice note) → WAV via `pydub`/`ffmpeg`.
4. Speech → text using Google Speech Recognition (`speech_recognition` lib).
5. (Optional) Text → translation via DeepL API.
6. Result embedded + posted (with optional translation field).

## Key Files
- `src/main.py`: Bootstraps bot, loads config, dynamic cog loader (all `cogs/*.py`). Removes default help.
- `src/cogs/transcribe.py`: Primary logic (message scanning, validation, transcription, translation, embed formatting).
- `src/cogs/help.py`: Custom help command using config prefix & embed styling.
- `src/cogs/other.py`: Language code listing command.
- `src/config/example.config.json`: Schema & example for required runtime config.
- `Dockerfile` + `compose.yaml`: Containerized runtime; mounts `src/config` read‑only so secrets not baked into image.

## Runtime Configuration
Expect a `src/config/config.json` (NOT committed) shaped like example. Fields actually read:
- `token` (named `token` in example but read in code as `config["token"]` and assigned to `bot_token`)
- `prefix` (command prefix e.g. `vmt ` including trailing space if desired)
- `deepl_api_key`
- `language_codes` (dict of UPPERCASE language codes ⇒ human names)

## Commands & Behaviors
- `help` (`h`): Custom embed.
- `language_codes` aliases: `langcodes`, `lc`, `languages`.
- `transcribe` aliases: `t`, `translate`, `tr`, `trans`.
  - Optional arg = target language code (case-insensitive; normalized upper; must exist in `language_codes`).
  - If no reply context, searches channel history newest→oldest for first voice message not authored by bot.
  - Auto-adds translation field when valid target provided.
- Passive listener: Any incoming voice message triggers auto transcription (adds ⌛ reaction, later removed).

## Voice Message Detection
`msg_has_voice_note(msg)` currently checks:
```python
if not msg.attachments or not msg.flags.value >> 13:
    return False
```
This relies on bit-shifting of internal flag integer to infer voice message; treat as brittle. Prefer preserving existing helper if modifying.

## Error Handling Patterns
- Silent except for printing raw exceptions to stdout (no logging framework).
- User-facing failures return concise message: "Could not transcribe ...".
- `sr.UnknownValueError` specifically mapped to "response was empty" messages.

## External Dependencies
- `speech_recognition` + Google Web Speech API (network call; may raise `UnknownValueError`).
- `pydub` requires `ffmpeg` (installed in Dockerfile). Keep this when altering base image.
- `deepl` official SDK; `Translator(auth_key=...)` invoked only if translation requested to avoid unnecessary API calls.

## Embeds & Attribution
Embeds use `discord.Color.og_blurple()` consistently. Transcription wrapped in triple backticks. Footer optionally includes requester.

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
- All language codes expected uppercase; normalization happens in command handler.
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
- Replace brittle voice note flag check with Discord-provided attribute if exposed.
- Centralize config load to avoid repeated disk I/O.
- Structured logging instead of print statements.
- Add minimal tests for `msg_has_voice_note` & transcription pipeline (mocking external APIs).

## When Unsure
Echo current behavior rather than inventing new flows; ask for clarification before refactoring architecture (e.g., introducing ORM, caching). Keep dependency versions aligned with `requirements.txt`.
