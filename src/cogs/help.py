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
    @commands.hybrid_command(name="help", aliases=["h"], description="Show help for bot commands")
    async def help(self, ctx: commands.Context):
        embed = discord.Embed(
            title="Help Menu",
            description="List of all available commands:",
            color=discord.Color.og_blurple(),
        )

        embed.add_field(
            name="**Transcribe**",
            value=f"\n> Transcribes a Discord voice message into text locally (Whisper tiny).\n > Usage: `{self.config['prefix']}transcribe` (reply to a voice message or it will pick the most recent).",
            inline=False,
        )

        embed.add_field(
            name="**Help**",
            value=f"> Shows this help menu.\n > Usage: `{self.config['prefix']}help`",
            inline=False,
        )

        embed.set_footer(text=f"Requested by {ctx.author.name}")
        # If invoked as slash, prefer interaction response first
        if getattr(ctx, "interaction", None):
            if not ctx.interaction.response.is_done():
                await ctx.interaction.response.send_message(embed=embed, ephemeral=False)
            else:
                await ctx.reply(embed=embed)
        else:
            await ctx.reply(embed=embed)


async def setup(bot):
    await bot.add_cog(Help(bot))
