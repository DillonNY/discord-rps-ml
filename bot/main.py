import discord
from discord.ext import commands
import random
import asyncio
import time
import os
from collections import Counter, deque
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Extract features from move sequences and timing"""
    
    def extract_features(self, moves_history, timing_history=None):
        """Extract ML features from player history"""
        if len(moves_history) < 5:
            return None
        
        moves = list(moves_history)
        features = []
        
        # Last 3 moves
        features.extend(moves[-3:])
        
        # Move frequencies
        move_counts = Counter(moves)
        total = len(moves)
        for move in [0, 1, 2]:
            features.append(move_counts.get(move, 0) / total)
        
        # Current streak length
        current_streak = 1
        for i in range(len(moves) - 2, -1, -1):
            if moves[i] == moves[-1]:
                current_streak += 1
            else:
                break
        features.append(min(current_streak, 10))  # Cap at 10
        
        # Pattern repetition score
        if len(moves) >= 4:
            pairs = [(moves[i], moves[i+1]) for i in range(len(moves)-1)]
            pair_counts = Counter(pairs)
            most_common_freq = pair_counts.most_common(1)[0][1] / len(pairs) if pairs else 0
            features.append(most_common_freq)
        else:
            features.append(0)
        
        # Alternation score
        if len(moves) > 1:
            alternations = sum(1 for i in range(1, len(moves)) if moves[i] != moves[i-1])
            alt_score = alternations / (len(moves) - 1)
        else:
            alt_score = 0
        features.append(alt_score)
        
        # Recent vs old behavior
        if len(moves) >= 10:
            recent = moves[-5:]
            old = moves[-10:-5]
            recent_counts = Counter(recent)
            old_counts = Counter(old)
            
            # Calculate behavior shift
            behavior_shift = 0
            for move in [0, 1, 2]:
                recent_freq = recent_counts.get(move, 0) / 5
                old_freq = old_counts.get(move, 0) / 5
                behavior_shift += abs(recent_freq - old_freq)
            features.append(behavior_shift)
        else:
            features.append(0)
        
        return np.array(features).reshape(1, -1)

class MLPredictor:
    """ML-based move predictor"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
        self.is_trained = False
        self.training_accuracy = 0.0
        self.min_samples = 10
    
    def prepare_training_data(self, moves_history):
        """Prepare training data from move history"""
        if len(moves_history) < self.min_samples:
            return None, None
        
        moves = list(moves_history)
        X, y = [], []
        
        # Create training samples
        for i in range(5, len(moves)):
            features = self.feature_engineer.extract_features(moves[:i])
            if features is not None:
                X.append(features[0])  # Extract from reshape
                y.append(moves[i])
        
        return np.array(X) if X else None, np.array(y) if y else None
    
    def train(self, player):
        """Train the model on player data"""
        X, y = self.prepare_training_data(player.moves_history)
        
        if X is None or len(X) < 5:
            return False
        
        try:
            self.model.fit(X, y)
            
            # Calculate training accuracy
            predictions = self.model.predict(X)
            self.training_accuracy = accuracy_score(y, predictions)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Training error: {e}")
            return False
    
    def predict(self, player):
        """Predict next move with confidence"""
        if not self.is_trained:
            return None, 0
        
        features = self.feature_engineer.extract_features(list(player.moves_history))
        if features is None:
            return None, 0
        
        try:
            # Get prediction
            prediction = self.model.predict(features)[0]
            
            # Get confidence from probability
            probabilities = self.model.predict_proba(features)[0]
            confidence = max(probabilities) * 100
            
            return int(prediction), confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0

class Player:
    """Enhanced player with comprehensive tracking"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.moves_history = deque(maxlen=150)
        self.timing_history = deque(maxlen=150)
        self.ml_wins = 0
        self.ml_games = 0
        self.ml_correct_predictions = 0
        self.last_move_time = None
        self.win_streak = 0
        self.max_win_streak = 0
        self.challenge_wins = 0
        self.challenge_losses = 0
        self.prediction_dodges = 0
    
    @property
    def games_played(self):
        return self.wins + self.losses + self.draws
    
    @property
    def win_rate(self):
        if self.games_played == 0:
            return 0.0
        return (self.wins / self.games_played) * 100
    
    @property
    def ml_resistance(self):
        """How well player resists ML predictions"""
        if self.ml_games == 0:
            return 0.0
        return (self.ml_wins / self.ml_games) * 100
    
    def add_game_result(self, user_move, outcome, ml_used=False, prediction_correct=False):
        """Record comprehensive game result"""
        current_time = time.time()
        
        # Add move and timing
        self.moves_history.append(user_move)
        if self.last_move_time:
            time_diff = min(current_time - self.last_move_time, 60)  # Cap at 60s
            self.timing_history.append(time_diff)
        self.last_move_time = current_time
        
        # Update basic stats
        if outcome == 1:  # Win
            self.wins += 1
            self.win_streak += 1
            self.max_win_streak = max(self.max_win_streak, self.win_streak)
        elif outcome == -1:  # Loss
            self.losses += 1
            self.win_streak = 0
        else:  # Draw
            self.draws += 1
            self.win_streak = 0
        
        # ML tracking
        if ml_used:
            self.ml_games += 1
            if outcome == 1:
                self.ml_wins += 1
            if prediction_correct:
                self.ml_correct_predictions += 1
            else:
                self.prediction_dodges += 1
    
    def get_move_preferences(self):
        """Get move distribution"""
        if not self.moves_history:
            return {"rock": 33.3, "paper": 33.3, "scissors": 33.3}
        
        move_counts = Counter(self.moves_history)
        total = len(self.moves_history)
        move_names = ["rock", "paper", "scissors"]
        
        return {move_names[i]: (move_counts.get(i, 0) / total) * 100 for i in range(3)}

class ChallengeSystem:
    """Handle player vs player challenges"""
    
    def __init__(self):
        self.challenges = {}  # channel_id -> challenge_data
    
    def create_challenge(self, challenger_id, challenged_id, channel_id):
        """Create new challenge"""
        self.challenges[channel_id] = {
            'challenger': challenger_id,
            'challenged': challenged_id,
            'status': 'pending',
            'moves': {},
            'round': 1,
            'scores': {challenger_id: 0, challenged_id: 0},
            'timestamp': time.time()
        }
        return True
    
    def get_challenge(self, channel_id):
        return self.challenges.get(channel_id)
    
    def accept_challenge(self, channel_id, user_id):
        challenge = self.challenges.get(channel_id)
        if challenge and challenge['challenged'] == user_id and challenge['status'] == 'pending':
            challenge['status'] = 'active'
            return True
        return False
    
    def submit_move(self, channel_id, user_id, move):
        challenge = self.challenges.get(channel_id)
        if not challenge or challenge['status'] != 'active':
            return None
        
        # Record move
        challenge['moves'][user_id] = move
        
        # Check if both players submitted
        if len(challenge['moves']) == 2:
            return self._resolve_round(channel_id)
        
        return 'waiting'
    
    def _resolve_round(self, channel_id):
        challenge = self.challenges[channel_id]
        
        challenger_move = challenge['moves'][challenge['challenger']]
        challenged_move = challenge['moves'][challenge['challenged']]
        
        # Determine winner - FIXED LOGIC
        if challenger_move == challenged_move:
            winner = 0  # Draw
        elif (challenger_move == 0 and challenged_move == 2) or \
             (challenger_move == 1 and challenged_move == 0) or \
             (challenger_move == 2 and challenged_move == 1):
            winner = 1  # Challenger wins
            challenge['scores'][challenge['challenger']] += 1
        else:
            winner = -1  # Challenged wins
            challenge['scores'][challenge['challenged']] += 1
        
        result = {
            'round': challenge['round'],
            'challenger': challenge['challenger'],
            'challenged': challenge['challenged'],
            'challenger_move': challenger_move,
            'challenged_move': challenged_move,
            'winner': winner,
            'scores': challenge['scores'].copy()
        }
        
        # Reset for next round
        challenge['moves'] = {}
        challenge['round'] += 1
        
        # Check if challenge complete
        if max(challenge['scores'].values()) >= 2:  # First to 2 wins
            result['completed'] = True
            if challenge['scores'][challenge['challenger']] > challenge['scores'][challenge['challenged']]:
                result['final_winner'] = challenge['challenger']
            else:
                result['final_winner'] = challenge['challenged']
        
        return result

class RPSGame:
    def __init__(self):
        self.moves = ["rock", "paper", "scissors"]
        self.emojis = ["🪨", "📄", "✂️"]
        self.conversion = {"rock": 0, "paper": 1, "scissors": 2}
    
    def get_user_choice(self, choice):
        return self.conversion.get(choice.lower())
    
    def get_computer_choice(self, predicted_move=None):
        if predicted_move is not None:
            return (predicted_move + 1) % 3  # Counter predicted move
        return random.randint(0, 2)
    
    def determine_winner(self, user, comp):
        """FIXED: Correct Rock Paper Scissors logic"""
        if user == comp:
            return 0  # Draw
        # Rock(0) beats Scissors(2), Paper(1) beats Rock(0), Scissors(2) beats Paper(1)
        elif (user == 0 and comp == 2) or (user == 1 and comp == 0) or (user == 2 and comp == 1):
            return 1  # User wins
        else:
            return -1  # Computer wins
    
    def get_move_name(self, move_int):
        return self.moves[move_int]
    
    def get_move_emoji(self, move_int):
        return self.emojis[move_int]

class PlayerManager:
    def __init__(self):
        self.players = {}
        self.predictor = MLPredictor()
    
    def get_player(self, user_id):
        user_id = str(user_id)
        if user_id not in self.players:
            self.players[user_id] = Player(user_id)
        return self.players[user_id]

# Initialize game components
game = RPSGame()
player_manager = PlayerManager()
challenge_system = ChallengeSystem()

# Bot setup with proper intents
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    print('Enhanced ML RPS Bot is ready!')

def create_result_embed(user_choice, comp_choice, outcome, user_name, ml_used=False, 
                       confidence=0, prediction_dodged=False):
    """Create result embed"""
    
    if outcome == 1:
        result_text = "🎉 You win!"
        color = 0x00ff00
    elif outcome == -1:
        result_text = "😞 You lose!"
        color = 0xff0000
    else:
        result_text = "🤝 It's a draw!"
        color = 0xffff00
    
    embed = discord.Embed(title="🎮 RPS Battle Result", color=color)
    
    user_emoji = game.get_move_emoji(user_choice)
    comp_emoji = game.get_move_emoji(comp_choice)
    user_move = game.get_move_name(user_choice)
    comp_move = game.get_move_name(comp_choice)
    
    embed.add_field(name=f"👤 {user_name}", value=f"{user_emoji} {user_move.title()}", inline=True)
    embed.add_field(name="🤖 AI Bot", value=f"{comp_emoji} {comp_move.title()}", inline=True)
    embed.add_field(name="Result", value=result_text, inline=False)
    
    # Battle visual
    embed.add_field(name="⚔️ Battle", value=f"{user_emoji} vs {comp_emoji}", inline=False)
    
    # ML info
    if ml_used:
        ml_status = f"🧠 AI Prediction Used\nConfidence: {confidence:.1f}%"
        if prediction_dodged:
            ml_status += "\n🎯 You dodged the prediction!"
        embed.add_field(name="AI Analysis", value=ml_status, inline=False)
    
    return embed

@bot.command(name='rps')
async def play_rps(ctx, move=None):
    """Play Rock Paper Scissors with ML prediction"""
    if move is None:
        embed = discord.Embed(
            title="🎮 Enhanced Rock Paper Scissors",
            description="Usage: `!rps rock`, `!rps paper`, or `!rps scissors`",
            color=0x00ff99
        )
        embed.add_field(name="🧠 AI Learning", value="I learn your patterns after 10 games!", inline=False)
        embed.add_field(name="Commands", value="`!rpsq` - Quick play\n`!challenge @user` - PvP battle", inline=False)
        await ctx.send(embed=embed)
        return
    
    # Validate move
    user_choice = game.get_user_choice(move)
    if user_choice is None:
        await ctx.send("❌ Invalid! Use: rock, paper, or scissors")
        return
    
    # Get player
    player = player_manager.get_player(ctx.author.id)
    
    # ML prediction
    ml_used = False
    confidence = 0
    predicted_move = None
    prediction_correct = False
    
    if len(player.moves_history) >= 10:
        if player_manager.predictor.train(player):
            predicted_move, confidence = player_manager.predictor.predict(player)
            if predicted_move is not None and confidence > 50:
                ml_used = True
                prediction_correct = (predicted_move == user_choice)
    
    # Computer move
    comp_choice = game.get_computer_choice(predicted_move if ml_used else None)
    outcome = game.determine_winner(user_choice, comp_choice)
    
    # Update player
    player.add_game_result(user_choice, outcome, ml_used, prediction_correct)
    
    # Create and send result
    prediction_dodged = ml_used and not prediction_correct
    embed = create_result_embed(user_choice, comp_choice, outcome, ctx.author.display_name,
                               ml_used, confidence, prediction_dodged)
    await ctx.send(embed=embed)

@bot.command(name='rpsq')
async def quick_rps(ctx):
    """Quick reaction-based RPS"""
    embed = discord.Embed(
        title="⚡ Quick RPS Battle",
        description="React with your choice!",
        color=0xff6b00
    )
    embed.add_field(name="🪨", value="Rock", inline=True)
    embed.add_field(name="📄", value="Paper", inline=True)
    embed.add_field(name="✂️", value="Scissors", inline=True)
    
    message = await ctx.send(embed=embed)
    
    for emoji in ["🪨", "📄", "✂️"]:
        await message.add_reaction(emoji)

@bot.command(name='challenge')
async def challenge_player(ctx, member: discord.Member = None):
    """Challenge another player"""
    if member is None:
        await ctx.send("❌ Mention a player: `!challenge @username`")
        return
    
    if member.bot:
        await ctx.send("❌ Can't challenge bots!")
        return
    
    if member.id == ctx.author.id:
        await ctx.send("❌ Can't challenge yourself!")
        return
    
    # Check existing challenge
    if challenge_system.get_challenge(ctx.channel.id):
        await ctx.send("❌ Challenge already active in this channel!")
        return
    
    # Create challenge
    challenge_system.create_challenge(ctx.author.id, member.id, ctx.channel.id)
    
    embed = discord.Embed(
        title="⚔️ RPS Challenge!",
        description=f"{ctx.author.mention} challenges {member.mention}!",
        color=0xff0066
    )
    embed.add_field(name="Rules", value="Best of 3 rounds", inline=False)
    embed.add_field(name="Accept", value="React ✅ to accept or ❌ to decline", inline=False)
    
    message = await ctx.send(embed=embed)
    await message.add_reaction("✅")
    await message.add_reaction("❌")

@bot.command(name='rpsstats')
async def show_stats(ctx, member: discord.Member = None):
    """Show player statistics"""
    target = member or ctx.author
    player = player_manager.get_player(target.id)
    
    if player.games_played == 0:
        await ctx.send(f"{target.display_name} hasn't played yet!")
        return
    
    embed = discord.Embed(
        title=f"📊 {target.display_name}'s Stats",
        color=0x0099ff
    )
    
    # Basic stats
    embed.add_field(
        name="🎮 Games",
        value=f"Played: {player.games_played}\n"
              f"Wins: 🏆 {player.wins}\n"
              f"Losses: 💀 {player.losses}\n"
              f"Win Rate: 📈 {player.win_rate:.1f}%",
        inline=True
    )
    
    # Streaks
    embed.add_field(
        name="🔥 Streaks",
        value=f"Current: {player.win_streak}\n"
              f"Best: 🏅 {player.max_win_streak}",
        inline=True
    )
    
    # ML stats
    if player.ml_games > 0:
        embed.add_field(
            name="🧠 vs AI",
            value=f"Resistance: {player.ml_resistance:.1f}%\n"
                  f"AI Games: {player.ml_games}\n"
                  f"Dodges: 🎯 {player.prediction_dodges}",
            inline=True
        )
    
    # Move preferences
    prefs = player.get_move_preferences()
    pref_text = "\n".join([f"{move.title()}: {pct:.1f}%" for move, pct in prefs.items()])
    embed.add_field(name="🎯 Preferences", value=pref_text, inline=True)
    
    # Challenges
    if player.challenge_wins + player.challenge_losses > 0:
        total_chall = player.challenge_wins + player.challenge_losses
        chall_wr = (player.challenge_wins / total_chall) * 100
        embed.add_field(
            name="⚔️ Challenges",
            value=f"Won: {player.challenge_wins}\n"
                  f"Lost: {player.challenge_losses}\n"
                  f"Win Rate: {chall_wr:.1f}%",
            inline=True
        )
    
    await ctx.send(embed=embed)

# Additional commands continued...
@bot.command(name='leaderboard')
async def leaderboard(ctx, sort_by='wins'):
    """Show leaderboard"""
    qualified = [(uid, p) for uid, p in player_manager.players.items() if p.games_played >= 3]
    
    if not qualified:
        await ctx.send("No players with 3+ games yet!")
        return
    
    # Sort options
    if sort_by == 'wins':
        qualified.sort(key=lambda x: x[1].win_rate, reverse=True)
        title = "🏆 Win Rate Leaders"
    elif sort_by == 'games':
        qualified.sort(key=lambda x: x[1].games_played, reverse=True)
        title = "🎮 Most Active"
    elif sort_by == 'resistance':
        qualified = [(uid, p) for uid, p in qualified if p.ml_games >= 3]
        qualified.sort(key=lambda x: x[1].ml_resistance, reverse=True)
        title = "🧠 AI Resistance"
    else:
        qualified.sort(key=lambda x: x[1].win_rate, reverse=True)
        title = "🏆 Win Rate Leaders"
    
    embed = discord.Embed(title=title, color=0xffd700)
    
    for i, (user_id, player) in enumerate(qualified[:10]):
        try:
            user = bot.get_user(int(user_id))
            name = user.display_name if user else f"Player {user_id}"
            
            if sort_by == 'resistance':
                value = f"AI Resistance: {player.ml_resistance:.1f}%"
            else:
                value = f"Win Rate: {player.win_rate:.1f}% ({player.games_played} games)"
            
            embed.add_field(name=f"{i+1}. {name}", value=value, inline=False)
        except:
            continue
    
    await ctx.send(embed=embed)

@bot.command(name='rpshelp')
async def show_help(ctx):
    """Show all available commands"""
    embed = discord.Embed(
        title="🎮 Enhanced RPS Bot Commands",
        description="Advanced Rock Paper Scissors with AI learning!",
        color=0x00ff99
    )
    
    embed.add_field(
        name="🎯 Game Commands",
        value="`!rps <move>` - Play RPS with AI\n"
              "`!rpsq` - Quick reaction game\n"
              "`!challenge @user` - Challenge a player",
        inline=False
    )
    
    embed.add_field(
        name="📊 Stats Commands",
        value="`!rpsstats [@user]` - View detailed stats\n"
              "`!leaderboard [wins/games/resistance]` - Rankings\n"
              "`!mlinfo` - AI prediction analysis",
        inline=False
    )
    
    embed.add_field(
        name="🧠 AI Features",
        value="• Learns patterns after 10 games\n"
              "• Multiple ML models\n"
              "• Real-time confidence display\n"
              "• Pattern dodge detection",
        inline=False
    )
    
    embed.add_field(
        name="⚔️ Challenge System",
        value="• Best of 3 rounds\n"
              "• Separate challenge statistics\n"
              "• Real-time move submission\n"
              "• Challenge leaderboards",
        inline=False
    )
    
    await ctx.send(embed=embed)

# Background cleanup task
async def cleanup_task():
    """Periodic cleanup of old challenges"""
    while True:
        await asyncio.sleep(60)  # Every minute
        try:
            current_time = time.time()
            expired = []
            for channel_id, challenge in challenge_system.challenges.items():
                if current_time - challenge['timestamp'] > 300:  # 5 minutes
                    expired.append(channel_id)
            
            for channel_id in expired:
                del challenge_system.challenges[channel_id]
            
            if expired:
                print(f"Cleaned up {len(expired)} expired challenges")
        except Exception as e:
            print(f"Cleanup error: {e}")

@bot.event
async def on_command_error(ctx, error):
    """Handle command errors gracefully"""
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("❌ Missing argument! Use `!rpshelp` for usage info.")
    elif isinstance(error, commands.CommandNotFound):
        pass  # Ignore unknown commands
    else:
        print(f"Command error: {error}")
        await ctx.send("An error occurred processing that command.")

# Simple test command to verify bot is working
@bot.command(name='test')
async def test_command(ctx):
    """Test if bot is responding"""
    await ctx.send("Bot is working! ✅")

@bot.event
async def on_reaction_add(reaction, user):
    """Handle reaction-based quick RPS gameplay"""
    if user.bot:
        return
    # Check if the message is a quick RPS embed
    if reaction.message.author != bot.user:
        return
    if not reaction.message.embeds:
        return
    embed = reaction.message.embeds[0]
    if embed.title != "⚡ Quick RPS Battle":
        return
    # Map emoji to move
    emoji_to_move = {"🪨": "rock", "📄": "paper", "✂️": "scissors"}
    move = emoji_to_move.get(str(reaction.emoji))
    if move is None:
        return
    user_choice = game.get_user_choice(move)
    player = player_manager.get_player(user.id)
    ml_used = False
    confidence = 0
    predicted_move = None
    prediction_correct = False
    if len(player.moves_history) >= 10:
        if player_manager.predictor.train(player):
            predicted_move, confidence = player_manager.predictor.predict(player)
            if predicted_move is not None and confidence > 50:
                ml_used = True
                prediction_correct = (predicted_move == user_choice)
    comp_choice = game.get_computer_choice(predicted_move if ml_used else None)
    outcome = game.determine_winner(user_choice, comp_choice)
    player.add_game_result(user_choice, outcome, ml_used, prediction_correct)
    prediction_dodged = ml_used and not prediction_correct
    result_embed = create_result_embed(user_choice, comp_choice, outcome, user.display_name,
                                      ml_used, confidence, prediction_dodged)
    await reaction.message.channel.send(f"{user.mention}", embed=result_embed)

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Run the bot
if __name__ == '__main__':
    # Get token from environment variable
    BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
    
    if not BOT_TOKEN:
        print("❌ Please set the DISCORD_BOT_TOKEN environment variable!")
        print("Create a .env file with: DISCORD_BOT_TOKEN=your_token_here")
        exit(1)
    
    try:
        print("Starting bot...")
        bot.run(BOT_TOKEN)
    except discord.errors.LoginFailure:
        print("❌ Invalid bot token! Please check your DISCORD_BOT_TOKEN.")
    except Exception as e:
        print(f"❌ Bot failed to start: {e}")