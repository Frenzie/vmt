"""Transcription cog using faster-whisper (tiny model) for local speech-to-text.

Translation feature removed; command now only performs transcription.
"""

import io
import os
import json
import typing
import asyncio
import tempfile
import re
import time
import logging

import discord
from discord.ext import commands
from faster_whisper import WhisperModel
from dataclasses import dataclass


logger = logging.getLogger(__name__)

"""Paragraph formatting configuration.

Enabled by default. Disable with WHISPER_PARAGRAPHS=0.
Optional tuning via env vars:
    WHISPER_PARAGRAPH_GAP (seconds, default 0.6) – silence gap threshold to start new paragraph
  WHISPER_PARAGRAPH_MIN_LEN (characters, default 40) – minimum paragraph length to allow punctuation-triggered split
"""
PARA_ENABLED = os.getenv("WHISPER_PARAGRAPHS", "1") != "0"
WORD_TIMINGS_ENABLED = os.getenv("WHISPER_WORD_TIMINGS", "1") != "0"
try:
    PARA_GAP = float(os.getenv("WHISPER_PARAGRAPH_GAP", "1.0"))
except ValueError:
    PARA_GAP = 1.0
try:
    PARA_MIN_LEN = int(os.getenv("WHISPER_PARAGRAPH_MIN_LEN", "40"))
except ValueError:
    PARA_MIN_LEN = 40

TERMINAL_PUNCT = ".!?…"  # characters that may indicate sentence end

def _format_paragraphs_words(words: list) -> str:
    """Paragraph formatting based on individual word timings.

    Uses word-level gaps instead of coarser segment gaps for better boundary detection.
    Logic:
      - Start new paragraph if gap between consecutive words > PARA_GAP and current paragraph length >= PARA_MIN_LEN.
      - Or if gap > PARA_GAP and previous word ends with TERMINAL_PUNCT (always split).
      - Merge very short paragraphs forward (< PARA_MIN_LEN chars) similar to segment approach.
    """
    if not words:
        return ""
    paras: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    last_end: float | None = None
    last_word_text = ""

    def commit():
        nonlocal cur, cur_len
        if cur:
            paras.append(cur.copy())
            cur.clear()
            cur_len = 0

    for w in words:
        text = getattr(w, 'word', '')
        if not text:
            continue
        start = getattr(w, 'start', None)
        end = getattr(w, 'end', None)
        gap = (start - last_end) if (start is not None and last_end is not None) else 0.0
        split = False
        if cur and gap > PARA_GAP:
            # Always split if punctuation ended previous word OR paragraph already long
            if last_word_text[-1:] in TERMINAL_PUNCT or cur_len >= PARA_MIN_LEN:
                split = True
        if split:
            commit()
        cur.append(text)
        cur_len += len(text)
        last_word_text = text
        if end is not None:
            last_end = end
    commit()

    # Merge small paragraphs
    merged: list[str] = []
    i = 0
    while i < len(paras):
        block = paras[i]
        block_text = "".join(block).strip()
        if len(block_text) < PARA_MIN_LEN and i + 1 < len(paras):
            paras[i + 1] = block + paras[i + 1]
        else:
            # Normalize spacing: words may include leading spaces already, so compact double spaces
            merged.append(re.sub(r"\s+", " ", block_text).strip())
        i += 1
    if not merged:
        merged = [re.sub(r"\s+", " ", "".join(sum(paras, []))).strip()]
    return "\n\n".join(merged).strip()


@dataclass
class TranscriptionJob:
    source: str  # 'slash' | 'context' | 'auto'
    voice_message: discord.Message
    requester: typing.Optional[typing.Union[discord.User, discord.Member]]
    interaction: typing.Optional[discord.Interaction]
    remove_reaction: bool


class Transcriber(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config = self.load_config()
        model_name = os.environ.get("WHISPER_MODEL", "small")
        t0 = time.perf_counter()
        self.model = WhisperModel(model_name, compute_type="auto")
        logger.info("Loaded Whisper model '%s' in %.2fs", model_name, time.perf_counter() - t0)
        # Queue for ordered transcription
        self._queue: asyncio.Queue[TranscriptionJob] = asyncio.Queue()
        self._processing_ids: set[int] = set()
        self._worker_task = self.bot.loop.create_task(self._queue_worker())

    def cog_unload(self):
        if self._worker_task:
            self._worker_task.cancel()

    def load_config(self):
        try:
            with open("config/config.json") as conf_file:
                return json.load(conf_file)
        except FileNotFoundError:
            return {"prefix": "vmt "}

    @discord.app_commands.command(name="transcribe", description="Transcribe a Discord voice message (optionally pass a message link/ID)")
    @discord.app_commands.describe(target="Optional message link or ID of the voice message")
    async def slash_transcribe(self, interaction: discord.Interaction, target: typing.Optional[str] = None):
        # 1. If target provided, attempt to resolve directly
        replied_message: typing.Optional[discord.Message] = None
        if target:
            replied_message = await resolve_target_message(interaction, target)
            if replied_message and not msg_is_transcribable(replied_message):
                await interaction.response.send_message("Provided message is not a voice message or audio attachment.")
                return
        # 2. If still none, attempt recent history scan (requires permission)
        if not replied_message:
            channel = interaction.channel
            if hasattr(channel, "history"):
                can_history = True
                audio_fallback: typing.Optional[discord.Message] = None
                if interaction.guild and interaction.guild.me:
                    perms = channel.permissions_for(interaction.guild.me)  # type: ignore
                    if hasattr(perms, 'read_message_history') and not perms.read_message_history:
                        can_history = False
                if can_history:
                    try:
                        async for message in channel.history(limit=50, oldest_first=False):  # type: ignore[attr-defined]
                            if message.author == interaction.client.user:  # type: ignore
                                continue
                            if msg_has_voice_note(message):  # prefer native voice note
                                replied_message = message
                                break
                            if not audio_fallback and msg_has_audio_attachment(message):
                                audio_fallback = message
                    except discord.Forbidden:
                        pass
                if not replied_message and audio_fallback:
                    replied_message = audio_fallback
        # 3. If still none, respond with guidance
        if not replied_message:
            guidance = (
                "No voice or audio message found. Provide a message link/ID or grant 'Read Message History'."
            )
            await interaction.response.send_message(guidance)
            return
        await interaction.response.defer(thinking=True)
        logger.info(
            "[slash] guild=%s channel=%s user=%s target_msg=%s starting transcription",
            getattr(interaction.guild, 'id', None),
            getattr(interaction.channel, 'id', None),
            interaction.user.id,
            replied_message.id,
        )
        position = self._queue.qsize() + 1
        await interaction.followup.send(f"Queued (position {position}). Processing sequentially...")
        await self._enqueue_job(TranscriptionJob(
            source="slash",
            voice_message=replied_message,
            requester=interaction.user,
            interaction=interaction,
            remove_reaction=False,
        ))

    # (Context menu moved outside class due to context menu decorator constraints)

    @commands.Cog.listener("on_message")
    async def auto_transcribe(self, msg: discord.Message):
        logger.debug("on_message id=%s author=%s", msg.id, msg.author.id)
        if not msg_is_transcribable(msg):
            return
        try:
            await msg.add_reaction("\N{HOURGLASS}")
        except Exception:
            pass
        await self._enqueue_job(TranscriptionJob(
            source="auto",
            voice_message=msg,
            requester=None,
            interaction=None,
            remove_reaction=True,
        ))
        logger.info("[auto] queued message_id=%s depth=%d", msg.id, self._queue.qsize())

    async def _enqueue_job(self, job: TranscriptionJob):
        if job.voice_message.id in self._processing_ids:
            return
        await self._queue.put(job)
        logger.info(
            "[queue] enqueued source=%s message_id=%s depth=%d",
            job.source,
            job.voice_message.id,
            self._queue.qsize(),
        )

    async def _do_transcription_job(self, job: TranscriptionJob):
        msg = job.voice_message
        t_start = time.perf_counter()
        try:
            text, language, audio_duration, language_prob = await transcribe_msg(msg, self.model)
            processing_time = time.perf_counter() - t_start
            content, file = build_transcription_message(
                text, msg, language, language_prob, audio_duration, processing_time, job.requester
            )
            sent = False
            try:
                if job.source in ("slash", "context") and job.interaction:
                    if file:
                        await msg.reply(content=content, mention_author=False, file=file)
                    else:
                        await msg.reply(content=content, mention_author=False)
                    await job.interaction.followup.send("Transcription posted.")
                else:
                    if file:
                        await msg.reply(content=content, file=file)
                    else:
                        await msg.reply(content=content)
                sent = True
            except discord.Forbidden:
                if job.interaction:
                    if file:
                        await job.interaction.followup.send(content=content, file=file)
                    else:
                        await job.interaction.followup.send(content=content)
                    sent = True
                else:
                    logger.warning("[queue] Forbidden posting message_id=%s", msg.id)
            logger.info(
                "[queue] done source=%s message_id=%s chars=%d attach=%s lang=%s audiodur=%.2fs proc=%.2fs sent=%s",
                job.source,
                msg.id,
                len(text),
                bool(file),
                language,
                audio_duration,
                processing_time,
                sent,
            )
        except Exception:
            logger.exception("[queue] failure message_id=%s source=%s", msg.id, job.source)
            if job.interaction:
                try:
                    await job.interaction.followup.send("Failed to transcribe.", ephemeral=True)
                except Exception:
                    pass
            else:
                try:
                    await msg.reply("Failed to transcribe.")
                except Exception:
                    pass

    async def _queue_worker(self):
        logger.info("[queue] worker started")
        while True:
            job: TranscriptionJob = await self._queue.get()
            msg_id = job.voice_message.id
            if msg_id in self._processing_ids:
                self._queue.task_done()
                continue
            self._processing_ids.add(msg_id)
            logger.info(
                "[queue] start source=%s message_id=%s depth=%d",
                job.source,
                msg_id,
                self._queue.qsize(),
            )
            try:
                await self._do_transcription_job(job)
            finally:
                if job.remove_reaction and self.bot.user:
                    try:
                        await job.voice_message.remove_reaction("\N{HOURGLASS}", self.bot.user)
                    except Exception:
                        pass
                self._processing_ids.discard(msg_id)
                self._queue.task_done()


def build_transcription_message(
    transcribed_text: str,
    message: discord.Message,
    language: str,
    language_prob: float,
    audio_duration: float,
    processing_time: float,
    ctx_author: typing.Optional[typing.Union[discord.User, discord.Member]] = None,
) -> tuple[str, typing.Optional[discord.File]]:
    """Return (quoted_message_content, optional_file) including original message metadata.

    - Uses block quote formatting.
    - Adds original message ID and UTC timestamp.
    - File name contains author slug, timestamp, and message ID when attached.
    """
    author = message.author
    full_text = (transcribed_text or "(empty transcription)").strip()
    PREVIEW_LIMIT = 600
    FILE_THRESHOLD = 900

    truncated = len(full_text) > PREVIEW_LIMIT
    preview = full_text[:PREVIEW_LIMIT]
    if truncated:
        slice_tail = preview[-120:]
        if "\n" in slice_tail:
            cut = preview.rfind("\n")
            if cut > 0 and cut > PREVIEW_LIMIT - 200:
                preview = preview[:cut]

    def quote_block(txt: str) -> str:
        return "\n".join(f"> {line}" if line.strip() else "> " for line in txt.splitlines())

    created = message.created_at  # UTC aware datetime
    timestamp_str = f"<t:{int(created.timestamp())}:f>" if created else "unknown time"
    # Format duration mm:ss
    dur_str = f"{int(audio_duration // 60)}:{int(audio_duration % 60):02d}"
    # Short language code fallback
    lang_display = language or "?"
    prob_display = f"{language_prob*100:.0f}%" if language_prob and language_prob > 0 else "?"
    header_bits = [
        f"{author.name}",
        f"{lang_display} {prob_display} {dur_str} (⏱{processing_time:.2f}s)",
        timestamp_str,
        f"{message.jump_url}" if hasattr(message, 'jump_url') else f"{message.id}",
    ]
    if ctx_author:
        header_bits.append(f"requested by {ctx_author}")
    header = " | ".join(header_bits)

    quoted = quote_block(preview)
    footer = "> (truncated, full transcript attached as file)" if (truncated or len(full_text) > FILE_THRESHOLD) else ""
    content = f"> **{header}**\n{quoted}\n{footer}"

    # File attachment logic
    file: typing.Optional[discord.File] = None
    if truncated or len(full_text) > FILE_THRESHOLD:
        # Sanitize author and compose filename
        raw_author = author.name if isinstance(author.name, str) else str(author.id)
        author_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_author)[:32] or str(author.id)
        ts_for_name = created.strftime("%Y%m%d-%H%M%S") if created else "unknown"
        filename = f"vm_{author_slug}_{ts_for_name}_{message.id}.txt"
        file_bytes = io.BytesIO(full_text.encode("utf-8"))
        file = discord.File(file_bytes, filename=filename)
    return content, file


def msg_has_voice_note(msg: typing.Optional[discord.Message]) -> bool:
    if not msg:
        return False
    if not msg.attachments or not msg.flags.value >> 13:
        return False
    return True

AUDIO_EXTS = {'.mp3', '.wav', '.ogg', '.oga', '.opus', '.m4a', '.flac'}

def msg_has_audio_attachment(msg: typing.Optional[discord.Message]) -> bool:
    if not msg or not msg.attachments:
        return False
    for att in msg.attachments:
        name = (att.filename or '').lower()
        ct = (att.content_type or '').lower()
        if ct.startswith('audio/') or any(name.endswith(ext) for ext in AUDIO_EXTS):
            return True
    return False

def msg_is_transcribable(msg: typing.Optional[discord.Message]) -> bool:
    return msg_has_voice_note(msg) or msg_has_audio_attachment(msg)


async def transcribe_msg(msg: discord.Message, model: WhisperModel) -> tuple[str, str, float, float]:
    """Transcribe attachment."""
    t0 = time.perf_counter()
    attachment = msg.attachments[0]
    voice_msg_bytes = await attachment.read()
    # Pick suffix from original filename (fallback .tmp)
    base, ext = os.path.splitext(attachment.filename or "")
    if not ext or len(ext) > 10:
        ext = ".tmp"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(voice_msg_bytes)
        tmp_path = tmp.name
    try:
        loop = asyncio.get_event_loop()
        def _run():
            segments_iter, info = model.transcribe(
                tmp_path,
                beam_size=5,
                word_timestamps=PARA_ENABLED and WORD_TIMINGS_ENABLED,
            )
            # Materialize generator so we can traverse multiple times
            seg_list = [s for s in segments_iter if getattr(s, 'text', None)]
            if PARA_ENABLED:
                # Prefer word-level grouping if available
                words: list = []
                if WORD_TIMINGS_ENABLED:
                    for seg in seg_list:
                        seg_words = getattr(seg, 'words', None)
                        if seg_words:
                            words.extend(seg_words)
                if words:
                    text_out = _format_paragraphs_words(words)
            else:
                text_out = " ".join(getattr(s, 'text', '').strip() for s in seg_list if getattr(s, 'text', None)).strip()
            language_code = getattr(info, 'language', 'unknown')
            duration_sec = getattr(info, 'duration', 0.0)
            language_prob = float(getattr(info, 'language_probability', 0.0) or 0.0)
            return text_out, language_code, duration_sec, language_prob
        text, language_code, duration_sec, language_prob = await loop.run_in_executor(None, _run)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    elapsed = time.perf_counter() - t0
    logger.info(
        "[core] transcribed message_id=%s duration=%.2fs bytes=%d chars=%d lang=%s prob=%.2f audiodur=%.2fs direct_input=%s",
        msg.id,
        elapsed,
        len(voice_msg_bytes),
        len(text),
        language_code,
        language_prob,
        duration_sec,
        ext,
    )
    return (text or "(empty transcription)", language_code, duration_sec, language_prob)


async def setup(bot):
    transcriber = Transcriber(bot)
    await bot.add_cog(transcriber)
    # Add context menu at tree level referencing handler
    if not any(cmd.name == "Transcribe Voice Message" for cmd in bot.tree.get_commands()):
        bot.tree.add_command(context_transcribe)


@discord.app_commands.context_menu(name="Transcribe Voice Message")
async def context_transcribe(interaction: discord.Interaction, message: discord.Message):
    # Lookup Transcriber cog to reuse model
    cog: typing.Optional[Transcriber] = interaction.client.get_cog("Transcriber")  # type: ignore
    if not cog:
        await interaction.response.send_message("Transcriber unavailable.")
        return
    if not msg_is_transcribable(message):
        await interaction.response.send_message("Selected message is neither a Discord voice note nor an audio file.")
        return
    try:
        await interaction.response.defer(thinking=True)
    except discord.InteractionResponded:
        pass
    # Queue job
    try:
        position = cog._queue.qsize() + 1
        await interaction.followup.send(f"Queued (position {position}). Processing sequentially...")
        await cog._enqueue_job(TranscriptionJob(
            source="context",
            voice_message=message,
            requester=interaction.user,
            interaction=interaction,
            remove_reaction=False,
        ))
    except Exception:
        logger.exception("Context enqueue failed message_id=%s", message.id)
        await interaction.followup.send("Failed to queue that voice message.", ephemeral=True)

    return



MESSAGE_LINK_RE = re.compile(r"https://(?:ptb\.|canary\.)?discord(?:app)?\.com/channels/(\d+)/(\d+)/(\d+)")


async def resolve_target_message(interaction: discord.Interaction, raw: str) -> typing.Optional[discord.Message]:
    raw = raw.strip()
    # Message link form
    m = MESSAGE_LINK_RE.match(raw)
    try:
        if m:
            guild_id, channel_id, message_id = map(int, m.groups())
            # Security: must be same guild (if guild interaction)
            if interaction.guild and interaction.guild.id != guild_id:
                return None
            channel = interaction.client.get_channel(channel_id) or await interaction.client.fetch_channel(channel_id)  # type: ignore
            return await channel.fetch_message(message_id)  # type: ignore
        # Bare numeric ID (assume same channel)
        if raw.isdigit() and hasattr(interaction.channel, 'fetch_message'):
            return await interaction.channel.fetch_message(int(raw))  # type: ignore
    except (discord.NotFound, discord.Forbidden, discord.HTTPException):
        return None
    return None
