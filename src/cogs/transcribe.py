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


logger = logging.getLogger(__name__)


class Transcriber(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config = self.load_config()
        # Initialize model once; compute_type auto selects best (fp16 on GPU else int8).
        model_name = os.environ.get("WHISPER_MODEL", "small")
        t0 = time.perf_counter()
        self.model = WhisperModel(model_name, compute_type="auto")
        logger.info("Loaded Whisper model '%s' in %.2fs", model_name, time.perf_counter() - t0)

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
            if replied_message and not msg_has_voice_note(replied_message):
                await interaction.response.send_message("Provided message is not a voice message.")
                return
        # 2. If still none, attempt recent history scan (requires permission)
        if not replied_message:
            channel = interaction.channel
            if hasattr(channel, "history"):
                can_history = True
                if interaction.guild and interaction.guild.me:
                    perms = channel.permissions_for(interaction.guild.me)  # type: ignore
                    if hasattr(perms, 'read_message_history') and not perms.read_message_history:
                        can_history = False
                if can_history:
                    try:
                        async for message in channel.history(limit=50, oldest_first=False):  # type: ignore[attr-defined]
                            if message.author != interaction.client.user and msg_has_voice_note(message):  # type: ignore
                                replied_message = message
                                break
                    except discord.Forbidden:
                        pass
        # 3. If still none, respond with guidance
        if not replied_message:
            guidance = (
                "No voice message found. Provide a message link/ID or grant 'Read Message History'."
            )
            await interaction.response.send_message(guidance)
            return

        author = replied_message.author
        await interaction.response.defer(thinking=True)
        started = time.perf_counter()
        logger.info(
            "[slash] guild=%s channel=%s user=%s target_msg=%s starting transcription",
            getattr(interaction.guild, 'id', None),
            getattr(interaction.channel, 'id', None),
            interaction.user.id,
            replied_message.id,
        )
        try:
            transcribe_start = time.perf_counter()
            transcribed_text, language, audio_duration, language_prob = await transcribe_msg(replied_message, self.model)
            processing_time = time.perf_counter() - transcribe_start
        except Exception:
            logger.exception("Transcription failed (slash) message_id=%s", replied_message.id)
            await interaction.followup.send(f"Failed to transcribe VM from {author}.", ephemeral=True)
            return
        content, file = build_transcription_message(transcribed_text, replied_message, language, language_prob, audio_duration, processing_time, interaction.user)
        try:
            if file:
                await replied_message.reply(content=content, mention_author=False, file=file)
            else:
                await replied_message.reply(content=content, mention_author=False)
            await interaction.followup.send("Transcription posted.")
            logger.info(
                "[slash] completed message_id=%s chars=%d attach=%s elapsed=%.2fs lang=%s audiodur=%.2fs",
                replied_message.id,
                len(transcribed_text),
                bool(file),
                time.perf_counter() - started,
                language,
                audio_duration,
            )
        except discord.Forbidden:
            if file:
                await interaction.followup.send(content=content, file=file)
            else:
                await interaction.followup.send(content=content)
            logger.warning(
                "[slash] fallback (Forbidden) posted via interaction followup message_id=%s", replied_message.id
            )

    # (Context menu moved outside class due to context menu decorator constraints)

    @commands.Cog.listener("on_message")
    async def auto_transcribe(self, msg: discord.Message):
        logger.debug("on_message id=%s author=%s", msg.id, msg.author.id)
        if not msg_has_voice_note(msg):
            return
        await msg.add_reaction("\N{HOURGLASS}")
        try:
            transcribe_start = time.perf_counter()
            text, language, audio_duration, language_prob = await transcribe_msg(msg, self.model)
            processing_time = time.perf_counter() - transcribe_start
            content, file = build_transcription_message(text, msg, language, language_prob, audio_duration, processing_time)
            if file:
                await msg.reply(content=content, file=file)
            else:
                await msg.reply(content=content)
            logger.info(
                "[auto] message_id=%s author=%s chars=%d attach=%s elapsed=%.2fs lang=%s audiodur=%.2fs",
                msg.id,
                msg.author.id,
                len(text),
                bool(file),
                processing_time,
                language,
                audio_duration,
            )
        except Exception:
            logger.exception("Auto transcription failed message_id=%s", msg.id)
            await msg.reply(content=f"Could not transcribe the Voice Message from {msg.author}.")
        finally:
            try:
                await msg.remove_reaction("\N{HOURGLASS}", self.bot.user)
            except Exception:
                pass


def build_transcription_message(transcribed_text: str, message: discord.Message, language: str, language_prob: float, audio_duration: float, processing_time: float, ctx_author: typing.Optional[discord.User] = None) -> tuple[str, typing.Optional[discord.File]]:
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
        return "\n".join(f"> {line}" if line.strip() else ">" for line in txt.splitlines())

    created = message.created_at  # UTC aware datetime
    timestamp_str = created.strftime("%Y-%m-%d %H:%M UTC") if created else "unknown time"
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
            segments, info = model.transcribe(tmp_path, beam_size=5)
            parts = []
            for seg in segments:
                if seg.text:
                    parts.append(seg.text.strip())
            text_out = " ".join(parts).strip()
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
    if not msg_has_voice_note(message):
        await interaction.response.send_message("Selected message is not a Discord voice message.")
        return
    try:
        await interaction.response.defer(thinking=True)
    except discord.InteractionResponded:
        pass
    try:
        started = time.perf_counter()
        transcribe_start = time.perf_counter()
        text, language, audio_duration, language_prob = await transcribe_msg(message, cog.model)
        processing_time = time.perf_counter() - transcribe_start
        content, file = build_transcription_message(text, message, language, language_prob, audio_duration, processing_time, interaction.user)
        try:
            if file:
                await message.reply(content=content, mention_author=False, file=file)
            else:
                await message.reply(content=content, mention_author=False)
            await interaction.followup.send("Transcription posted under the original voice message.")
            logger.info(
                "[context] completed message_id=%s chars=%d attach=%s elapsed=%.2fs lang=%s audiodur=%.2fs",
                message.id,
                len(text),
                bool(file),
                time.perf_counter() - started,
                language,
                audio_duration,
            )
        except discord.Forbidden:
            if file:
                await interaction.followup.send(content=content, file=file)
            else:
                await interaction.followup.send(content=content)
            logger.warning(
                "[context] Forbidden replying under message_id=%s; used followup", message.id
            )
    except Exception:
        logger.exception("Context menu transcription failed message_id=%s", message.id)
        await interaction.followup.send("Failed to transcribe that voice message.", ephemeral=True)


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
