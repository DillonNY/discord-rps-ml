# Discord RPS ML Bot

A Discord bot that plays rock-paper-scissors and learns per user over time using machine learning.

## Features
- Play RPS against the bot with `!rps <move>`
- Quick reaction-based RPS with `!rpsq`
- AI learns your patterns after 10 games(Not actually AI just machine learning)
- PvP challenge system (`!challenge @user`)
- Advanced player stats and leaderboards
- ML prediction confidence and dodge detection

## Setup
1. Clone the repository:
   ```
   git clone https://github.com/DillonNY/discord-rps-ml.git
   cd discord-rps-ml
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set your Discord bot token as an environment variable:
   - Create a `.env` file with:
     ```
     DISCORD_BOT_TOKEN=your_token_here
     ```
   - Or set it in your system environment variables.
4. Run the bot:
   ```
   python bot/main.py
   ```

## Requirements
- Python 3.8+
- discord.py
- scikit-learn
- numpy

## License
MIT License

---
Feel free to contribute or open issues!