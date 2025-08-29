# main.py
import os
from dotenv import load_dotenv

import discord
from discord.ext import commands

# ---- config ----
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    raise SystemExit("Missing DISCORD_TOKEN in .env")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents, description="Minimal Discord Bot")

# ---- events ----
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (id: {bot.user.id})")
    await bot.change_presence(activity=discord.Game(name="!help"))

# ---- basic commands ----
@bot.command(name="hello")
async def hello(ctx: commands.Context):
    await ctx.send("Hello!")

@bot.command(name="ping")
async def ping(ctx: commands.Context):
    await ctx.send("Pong!")

# ---- your functions go here ----
# Example of wiring in your own function:
# from core.some_module import my_function
# @bot.command(name="doit")
# async def doit(ctx: commands.Context, *args):
#     result = my_function(*args)
#     await ctx.send(str(result))

# ---- start ----
if __name__ == "__main__":
    bot.run(TOKEN)
