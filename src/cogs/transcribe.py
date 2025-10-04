"""Transcription cog using faster-whisper (tiny model) for local speech-to-text.

Translation feature removed; command now only performs transcription.
"""

import io
import os
import json
import typing
import asyncio
import tempfile
import textwrap

import discord
from discord.ext import commands
from faster_whisper import WhisperModel
import pydub


class Transcriber(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config = self.load_config()
        # Initialize tiny model once; compute_type auto selects best (fp16 on GPU else int8).
        self.model = WhisperModel("tiny", compute_type="auto")

    def load_config(self):
        try:
            with open("config/config.json") as conf_file:
                return json.load(conf_file)
        except FileNotFoundError:
            return {"prefix": "vmt "}

    @commands.hybrid_command(name="transcribe", aliases=["t"], description="Transcribe a Discord voice message to text")
    async def transcribe(self, ctx: commands.Context):
        """Transcribe a replied-to Discord voice message (or the most recent one)."""
        replied_message = None
        if ctx.message.reference:
            replied_message = await ctx.channel.fetch_message(ctx.message.reference.message_id)

        if not msg_has_voice_note(replied_message):
            async for message in ctx.channel.history(limit=50, oldest_first=False):
                if message.author != ctx.bot.user and msg_has_voice_note(message):
                    replied_message = message
                    break

        if not replied_message:
            await ctx.reply("No voice message found (reply to one or ensure recent VM exists).")
            return

        author = replied_message.author
        response = await ctx.reply(f"Transcribing the Voice Message from {author}...")
        try:
            transcribed_text = await transcribe_msg(replied_message, self.model)
            await response.delete()
        except Exception as e:
            await response.edit(content=f"Failed to transcribe VM from {author}.")
            print(e)
            return

        embed = make_embed(transcribed_text, author, ctx.author)
        if getattr(ctx, "interaction", None):
            if not ctx.interaction.response.is_done():
                await ctx.interaction.response.send_message(embed=embed, ephemeral=False)
            else:
                await replied_message.reply(embed=embed, mention_author=False)
        else:
            await replied_message.reply(embed=embed, mention_author=False)

    @commands.Cog.listener("on_message")
    async def auto_transcribe(self, msg: discord.Message):
        if not msg_has_voice_note(msg):
            return
        await msg.add_reaction("\N{HOURGLASS}")
        try:
            text = await transcribe_msg(msg, self.model)
            embed = make_embed(text, msg.author)
            await msg.reply(embed=embed)
        except Exception as e:
            print(e)
            await msg.reply(content=f"Could not transcribe the Voice Message from {msg.author}.")
        finally:
            try:
                await msg.remove_reaction("\N{HOURGLASS}", self.bot.user)
            except Exception:
                pass


def make_embed(transcribed_text: str, author: discord.User, ctx_author: typing.Optional[discord.User] = None):
    embed = discord.Embed(color=discord.Color.og_blurple(), title=f"🔊 {author.name}'s Voice Message")
    embed.add_field(
        name="Transcription",
        value=textwrap.dedent(
            f"""
            ```
            {transcribed_text}
            ```
            [vmt bot](https://github.com/frenzie/vmt)
            """
        ),
        inline=False,
    )
    if ctx_author:
        embed.set_footer(text=f"Requested by {ctx_author}")
    return embed


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
    await bot.add_cog(Transcriber(bot))
