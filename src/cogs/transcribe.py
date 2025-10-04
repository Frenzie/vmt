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

import discord
from discord.ext import commands
from faster_whisper import WhisperModel
import pydub


class Transcriber(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config = self.load_config()
        # Initialize tiny model once; compute_type auto selects best (fp16 on GPU else int8).
        self.model = WhisperModel("small", compute_type="auto")

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
        try:
            transcribed_text = await transcribe_msg(replied_message, self.model)
        except Exception as e:
            print(e)
            await interaction.followup.send(f"Failed to transcribe VM from {author}.", ephemeral=True)
            return
        content, file = build_transcription_message(transcribed_text, replied_message, interaction.user)
        try:
            if file:
                await replied_message.reply(content=content, mention_author=False, file=file)
            else:
                await replied_message.reply(content=content, mention_author=False)
            await interaction.followup.send("Transcription posted.")
        except discord.Forbidden:
            if file:
                await interaction.followup.send(content=content, file=file)
            else:
                await interaction.followup.send(content=content)

    # (Context menu moved outside class due to context menu decorator constraints)

    @commands.Cog.listener("on_message")
    async def auto_transcribe(self, msg: discord.Message):
        print("Received message:", msg)
        if not msg_has_voice_note(msg):
            return
        await msg.add_reaction("\N{HOURGLASS}")
        try:
            text = await transcribe_msg(msg, self.model)
            content, file = build_transcription_message(text, msg)
            if file:
                await msg.reply(content=content, file=file)
            else:
                await msg.reply(content=content)
        except Exception as e:
            print(e)
            await msg.reply(content=f"Could not transcribe the Voice Message from {msg.author}.")
        finally:
            try:
                await msg.remove_reaction("\N{HOURGLASS}", self.bot.user)
            except Exception:
                pass


def build_transcription_message(transcribed_text: str, message: discord.Message, ctx_author: typing.Optional[discord.User] = None) -> tuple[str, typing.Optional[discord.File]]:
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
        preview += "\n... (truncated)"

    def quote_block(txt: str) -> str:
        return "\n".join(f"> {line}" if line.strip() else ">" for line in txt.splitlines())

    created = message.created_at  # UTC aware datetime
    timestamp_str = created.strftime("%Y-%m-%d %H:%M:%S UTC") if created else "unknown time"
    header_bits = [
        f"Transcription of {author.name}'s voice message",
        f"msg {message.id}",
        timestamp_str,
    ]
    if ctx_author:
        header_bits.append(f"requested by {ctx_author}")
    header = " | ".join(header_bits)

    quoted = quote_block(preview)
    meta_line = f"> Source: {message.jump_url}" if hasattr(message, 'jump_url') else f"> ID: {message.id}"
    footer = "> (full transcript attached as file)" if (truncated or len(full_text) > FILE_THRESHOLD) else "> (end)"
    content = f"> **{header}**\n{meta_line}\n{quoted}\n{footer}"

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


async def transcribe_msg(msg: discord.Message, model: WhisperModel) -> str:
    # Read attachment bytes
    voice_msg_bytes = await msg.attachments[0].read()
    voice_msg = io.BytesIO(voice_msg_bytes)
    # Convert to wav temp file for model
    audio_segment = pydub.AudioSegment.from_file(voice_msg)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        audio_segment.export(tmp_path, format="wav")
    try:
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        def _run():
            segments, info = model.transcribe(tmp_path, beam_size=1)
            return " ".join(seg.text.strip() for seg in segments).strip()
        text = await loop.run_in_executor(None, _run)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return text or "(empty transcription)"


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
        text = await transcribe_msg(message, cog.model)
        content, file = build_transcription_message(text, message, interaction.user)
        try:
            if file:
                await message.reply(content=content, mention_author=False, file=file)
            else:
                await message.reply(content=content, mention_author=False)
            await interaction.followup.send("Transcription posted under the original voice message.")
        except discord.Forbidden:
            if file:
                await interaction.followup.send(content=content, file=file)
            else:
                await interaction.followup.send(content=content)
    except Exception as e:
        print(e)
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
