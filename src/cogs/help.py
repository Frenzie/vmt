import discord
from discord.ext import commands
import json


class Help(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.config = self.load_config()

    def load_config(self):
        with open("config/config.json") as conf_file:
            return json.load(conf_file)

    # help command
    @discord.app_commands.command(name="help", description="Show help for the transcription application commands")
    async def slash_help(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="Help",
            description="Commands available:",
            color=discord.Color.og_blurple(),
        )
        embed.add_field(
            name="/transcribe",
            value="Transcribe the most recent voice message in this channel (or reply in prefix mode).",
            inline=False,
        )
        embed.add_field(
            name="Auto Transcription",
            value="New voice messages are automatically transcribed and replied to with an embed.",
            inline=False,
        )
        embed.add_field(
            name="Context Menu",
            value="Right-click a voice message → Apps → 'Transcribe Voice Message' to transcribe that specific message.",
            inline=False,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)


async def setup(bot):
    await bot.add_cog(Help(bot))
