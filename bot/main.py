import discord
from discord.ext import commands
import random
from collections import Counter, deque
import numpy as np

class RPSGame:
    """Core Rock Paper Scissors game logic with ML prediction"""
    
    RPS_CONVERSION = {"rock": 0, "paper": 1, "scissors": 2}
    REVERSE_RPS = {v: k for k, v in RPS_CONVERSION.items()}
    MOVE_EMOJIS = {"rock": "🪨", "paper": "📄", "scissors": "✂️"}
    
    def __init__(self):
        self.result_text = {1: "You win!", 0: "It's a draw!", -1: "You lose!"}
    
    def get_user_choice(self, choice):
        """Convert string choice to integer"""
        return self.RPS_CONVERSION.get(choice.lower(), None)
    
    def get_computer_choice(self, predicted_user_move=None):
        """Generate computer choice - random or counter-predicted move"""
        if predicted_user_move is not None:
            # Counter the predicted move (rock beats scissors, etc.)
            counter_move = (predicted_user_move + 1) % 3
            return counter_move
        return random.choice(list(self.RPS_CONVERSION.values()))
    
    def determine_winner(self, user, comp):
        """Determine winner between user and computer"""
        if user == comp:
            return 0   # draw
        elif (user - comp) % 3 == 1:
            return 1   # user wins
        else:
            return -1  # user loses
    
    def get_move_name(self, move_int):
        """Convert integer move back to string"""
        return self.REVERSE_RPS[move_int]
    
    def get_move_emoji(self, move_int):
        """Get emoji for a move"""
        move_name = self.get_move_name(move_int)
        return self.MOVE_EMOJIS[move_name]

class SimpleMLPredictor:
    """Simple pattern-based move predictor"""
    
    def __init__(self, pattern_length=3):
        self.pattern_length = pattern_length
        self.pattern_counts = {}
    
    def update_patterns(self, moves_history):
        """Update pattern database with new move history"""
        if len(moves_history) < self.pattern_length + 1:
            return
        
        # Extract patterns from history
        for i in range(len(moves_history) - self.pattern_length):
            pattern = tuple(moves_history[i:i + self.pattern_length])
            next_move = moves_history[i + self.pattern_length]
            
            if pattern not in self.pattern_counts:
                self.pattern_counts[pattern] = Counter()
            self.pattern_counts[pattern][next_move] += 1
    
    def predict_next_move(self, recent_moves):
        """Predict next move based on recent pattern"""
        if len(recent_moves) < self.pattern_length:
            return None
        
        current_pattern = tuple(recent_moves[-self.pattern_length:])
        
        if current_pattern in self.pattern_counts:
            # Get most common next move for this pattern
            most_common = self.pattern_counts[current_pattern].most_common(1)
            if most_common:
                return most_common[0][0]
        
        return None
    
    def get_confidence(self, recent_moves):
        """Get confidence in prediction (0-100%)"""
        if len(recent_moves) < self.pattern_length:
            return 0
        
        current_pattern = tuple(recent_moves[-self.pattern_length:])
        if current_pattern not in self.pattern_counts:
            return 0
        
        pattern_data = self.pattern_counts[current_pattern]
        total_occurrences = sum(pattern_data.values())
        max_count = max(pattern_data.values())
        
        return (max_count / total_occurrences) * 100 if total_occurrences > 0 else 0

class Player:
    """Represents a Discord user's RPS data and statistics"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.moves_history = deque(maxlen=100)  # Keep last 100 moves
        self.ml_wins = 0  # Wins when ML was used
        self.ml_games = 0  # Games where ML was used
    
    @property
    def games_played(self):
        """Total games played"""
        return self.wins + self.losses + self.draws
    
    @property
    def win_rate(self):
        """Calculate win rate as percentage"""
        if self.games_played == 0:
            return 0.0
        return (self.wins / self.games_played) * 100
    
    @property
    def ml_performance(self):
        """How well ML performs against this player"""
        if self.ml_games == 0:
            return 0.0
        return ((self.ml_games - self.ml_wins) / self.ml_games) * 100  # Bot's win rate
    
    def add_game_result(self, user_move, outcome, ml_used=False):
        """Record a game result"""
        self.moves_history.append(user_move)
        
        if outcome == 1:
            self.wins += 1
        elif outcome == -1:
            self.losses += 1
        else:
            self.draws += 1
        
        if ml_used:
            self.ml_games += 1
            if outcome == 1:  # User won
                self.ml_wins += 1
    
    def get_move_preferences(self):
        """Get player's move frequency as percentages"""
        if not self.moves_history:
            return {}
        
        move_counts = Counter(self.moves_history)
        total = len(self.moves_history)
        
        preferences = {}
        for move_int in [0, 1, 2]:  # rock, paper, scissors
            count = move_counts.get(move_int, 0)
            move_name = RPSGame.REVERSE_RPS[move_int]
            preferences[move_name] = (count / total) * 100
        
        return preferences

class PlayerManager:
    """Manages all player data and statistics"""
    
    def __init__(self):
        self.players = {}
        self.predictor = SimpleMLPredictor()
    
    def get_player(self, user_id):
        """Get or create a player"""
        if user_id not in self.players:
            self.players[user_id] = Player(user_id)
        return self.players[user_id]
    
    def get_leaderboard(self, min_games=5):
        """Get leaderboard sorted by win rate"""
        qualified = {uid: player for uid, player in self.players.items() 
                    if player.games_played >= min_games}
        
        return sorted(qualified.items(), key=lambda x: x[1].win_rate, reverse=True)
    
    def reset_player(self, user_id):
        """Reset a player's statistics"""
        if user_id in self.players:
            del self.players[user_id]
            return True
        return False
    
    def should_use_ml(self, player):
        """Decide whether to use ML prediction (after 5+ games)"""
        return len(player.moves_history) >= 5

# Helper function for creating result embeds
def create_result_embed(user_choice, comp_choice, outcome, ml_used=False, confidence=0):
    """Create a formatted result embed"""
    if outcome == 1:
        result_text = "🎉 You win!"
        color = 0x00ff00  # Green
    elif outcome == -1:
        result_text = "😔 You lose!"
        color = 0xff0000  # Red
    else:
        result_text = "🤝 It's a draw!"
        color = 0xffff00  # Yellow
    
    embed = discord.Embed(title="🎮 RPS Result", color=color)
    
    user_move_name = game.get_move_name(user_choice)
    comp_move_name = game.get_move_name(comp_choice)
    user_emoji = game.get_move_emoji(user_choice)
    comp_emoji = game.get_move_emoji(comp_choice)
    
    embed.add_field(name="You played", value=f"{user_emoji} {user_move_name.title()}", inline=True)
    embed.add_field(name="I played", value=f"{comp_emoji} {comp_move_name.title()}", inline=True)
    embed.add_field(name="Result", value=result_text, inline=False)
    embed.add_field(name="Visual", value=f"{user_emoji} vs {comp_emoji}", inline=False)
    
    # Add ML info if used
    if ml_used:
        ml_status = f"🤖 ML Prediction (Confidence: {confidence:.0f}%)"
        embed.add_field(name="AI Mode", value=ml_status, inline=False)
    
    return embed

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Initialize game components
game = RPSGame()
player_manager = PlayerManager()

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    print('ML-Enhanced RPS Bot is ready to play!')

# Commands using the OOP classes
@bot.command(name='rps')
async def play_rps(ctx, move=None):
    """Play Rock Paper Scissors! Usage: !rps rock/paper/scissors"""
    if move is None:
        embed = discord.Embed(
            title="🎮 Rock Paper Scissors",
            description="Usage: `!rps rock`, `!rps paper`, or `!rps scissors`",
            color=0x00ff00
        )
        embed.add_field(name="How to play", value="Choose your move and I'll play against you!", inline=False)
        embed.add_field(name="AI Learning", value="After 5 games, I'll start learning your patterns! 🤖", inline=False)
        await ctx.send(embed=embed)
        return
    
    # Process the game
    user_choice = game.get_user_choice(move)
    if user_choice is None:
        await ctx.send("❌ Invalid choice! Please use `rock`, `paper`, or `scissors`")
        return
    
    # Get player and check if we should use ML
    player = player_manager.get_player(ctx.author.id)
    
    ml_used = False
    confidence = 0
    predicted_move = None
    
    if player_manager.should_use_ml(player):
        # Update ML patterns with current history
        player_manager.predictor.update_patterns(list(player.moves_history))
        
        # Try to predict user's move
        predicted_move = player_manager.predictor.predict_next_move(list(player.moves_history))
        confidence = player_manager.predictor.get_confidence(list(player.moves_history))
        
        if predicted_move is not None and confidence > 30:  # Only use if confident enough
            ml_used = True
    
    # Generate computer move
    comp_choice = game.get_computer_choice(predicted_move if ml_used else None)
    outcome = game.determine_winner(user_choice, comp_choice)
    
    # Update player data
    player.add_game_result(user_choice, outcome, ml_used)
    
    # Create result embed
    embed = create_result_embed(user_choice, comp_choice, outcome, ml_used, confidence)
    await ctx.send(embed=embed)

@bot.command(name='rpsq')
async def quick_rps(ctx):
    """Quick RPS with reaction buttons"""
    embed = discord.Embed(
        title="🎮 Quick RPS",
        description="React with your choice!",
        color=0x0099ff
    )
    embed.add_field(name="🪨", value="Rock", inline=True)
    embed.add_field(name="📄", value="Paper", inline=True)
    embed.add_field(name="✂️", value="Scissors", inline=True)
    
    message = await ctx.send(embed=embed)
    
    # Add reaction buttons
    for emoji in ["🪨", "📄", "✂️"]:
        await message.add_reaction(emoji)

@bot.command(name='rpsstats')
async def rps_stats(ctx, member: discord.Member = None):
    """Show RPS statistics for a player"""
    target = member or ctx.author
    player = player_manager.get_player(target.id)
    
    if player.games_played == 0:
        await ctx.send(f"{target.display_name} hasn't played any games yet!")
        return
    
    embed = discord.Embed(title=f"📊 {target.display_name}'s RPS Stats", color=0x0099ff)
    embed.add_field(name="Games Played", value=player.games_played, inline=True)
    embed.add_field(name="Wins", value=f"🏆 {player.wins}", inline=True)
    embed.add_field(name="Losses", value=f"💀 {player.losses}", inline=True)
    embed.add_field(name="Draws", value=f"🤝 {player.draws}", inline=True)
    embed.add_field(name="Win Rate", value=f"📈 {player.win_rate:.1f}%", inline=True)
    
    # ML performance stats
    if player.ml_games > 0:
        embed.add_field(name="AI Performance vs You", value=f"🤖 {player.ml_performance:.1f}%", inline=True)
    
    # Show move preferences
    preferences = player.get_move_preferences()
    if preferences:
        pref_text = "\n".join([f"{move.title()}: {pct:.1f}%" 
                              for move, pct in preferences.items()])
        embed.add_field(name="Move Preferences", value=pref_text, inline=False)
    
    # Show predictability
    if len(player.moves_history) >= 5:
        # Simple predictability metric - how often they repeat patterns
        patterns = []
        moves_list = list(player.moves_history)
        for i in range(len(moves_list) - 2):
            patterns.append(tuple(moves_list[i:i+3]))
        
        if patterns:
            pattern_counts = Counter(patterns)
            most_common_pattern_count = pattern_counts.most_common(1)[0][1]
            predictability = (most_common_pattern_count / len(patterns)) * 100
            embed.add_field(name="Predictability", value=f"📈 {predictability:.1f}%", inline=True)
    
    await ctx.send(embed=embed)

@bot.command(name='rpsleader')
async def rps_leaderboard(ctx):
    """Show RPS leaderboard"""
    leaderboard = player_manager.get_leaderboard(min_games=3)
    
    if not leaderboard:
        await ctx.send("No players with 3+ games yet!")
        return
    
    embed = discord.Embed(title="🏆 RPS Leaderboard", color=0xffd700)
    
    for i, (user_id, player) in enumerate(leaderboard[:10]):
        try:
            # Try multiple methods to get user info
            user = bot.get_user(int(user_id))
            if not user:
                # Try fetching from Discord API if not in cache
                try:
                    user = await bot.fetch_user(int(user_id))
                except:
                    user = None
            
            # Get the best available name
            if user:
                name = user.display_name or user.name or user.global_name
            else:
                name = f"Player {user_id}"
            
            # Add ML performance if available
            ml_info = ""
            if player.ml_games > 0:
                ml_info = f" | AI: {player.ml_performance:.0f}%"
            
            embed.add_field(
                name=f"{i+1}. {name}",
                value=f"Win Rate: {player.win_rate:.1f}% ({player.wins}/{player.games_played}){ml_info}",
                inline=False
            )
        except Exception as e:
            # Fallback for any errors
            embed.add_field(
                name=f"{i+1}. Unknown Player",
                value=f"Win Rate: {player.win_rate:.1f}% ({player.wins}/{player.games_played})",
                inline=False
            )
            continue
    
    await ctx.send(embed=embed)

@bot.command(name='rpsreset')
async def reset_stats(ctx):
    """Reset your RPS statistics"""
    if player_manager.reset_player(ctx.author.id):
        await ctx.send("🔄 Your RPS stats have been reset!")
    else:
        await ctx.send("You don't have any stats to reset!")

@bot.command(name='rpsml')
async def ml_info(ctx):
    """Show ML prediction info for the user"""
    player = player_manager.get_player(ctx.author.id)
    
    if len(player.moves_history) < 5:
        await ctx.send("🤖 Play at least 5 games first, then I'll start learning your patterns!")
        return
    
    embed = discord.Embed(title="🤖 AI Analysis", color=0x9932cc)
    
    # Update patterns and get prediction
    player_manager.predictor.update_patterns(list(player.moves_history))
    predicted = player_manager.predictor.predict_next_move(list(player.moves_history))
    confidence = player_manager.predictor.get_confidence(list(player.moves_history))
    
    if predicted is not None:
        predicted_name = game.get_move_name(predicted)
        predicted_emoji = game.get_move_emoji(predicted)
        embed.add_field(
            name="Next Move Prediction", 
            value=f"{predicted_emoji} {predicted_name.title()}", 
            inline=True
        )
        embed.add_field(name="Confidence", value=f"{confidence:.1f}%", inline=True)
    else:
        embed.add_field(name="Prediction", value="No clear pattern detected", inline=False)
    
    # Recent moves
    if len(player.moves_history) >= 5:
        recent = list(player.moves_history)[-5:]
        recent_text = " → ".join([game.get_move_emoji(m) for m in recent])
        embed.add_field(name="Recent Moves", value=recent_text, inline=False)
    
    # ML vs player performance
    if player.ml_games > 0:
        embed.add_field(
            name="AI vs You", 
            value=f"AI Win Rate: {player.ml_performance:.1f}% ({player.ml_games - player.ml_wins} wins / {player.ml_games} AI games)", 
            inline=False
        )
    
    await ctx.send(embed=embed)

@bot.command(name='rpshelp')
async def rps_help(ctx):
    """Show all available RPS commands"""
    embed = discord.Embed(
        title="🎮 RPS Bot Commands", 
        description="All available Rock Paper Scissors commands:",
        color=0x00ff99
    )
    
    embed.add_field(
        name="!rps <move>", 
        value="Play RPS! Use `!rps rock`, `!rps paper`, or `!rps scissors`", 
        inline=False
    )
    embed.add_field(
        name="!rpsq", 
        value="Quick RPS with reaction buttons - just click to play!", 
        inline=False
    )
    embed.add_field(
        name="!rpsstats [@user]", 
        value="View your stats or another player's stats", 
        inline=False
    )
    embed.add_field(
        name="!rpsleader", 
        value="View the leaderboard (players with 3+ games)", 
        inline=False
    )
    embed.add_field(
        name="!rpsml", 
        value="🤖 View AI analysis of your play patterns (after 5+ games)", 
        inline=False
    )
    embed.add_field(
        name="!rpsreset", 
        value="Reset your personal statistics", 
        inline=False
    )
    embed.add_field(
        name="!rpshelp", 
        value="Show this help message", 
        inline=False
    )
    
    embed.add_field(
        name="🤖 AI Features", 
        value="After 5 games, the bot learns your patterns and tries to predict your moves!", 
        inline=False
    )
    
    await ctx.send(embed=embed)

@bot.event
async def on_reaction_add(reaction, user):
    """Handle reaction-based gameplay"""
    if user.bot:
        return
    
    if (reaction.message.embeds and 
        len(reaction.message.embeds) > 0 and 
        "Quick RPS" in reaction.message.embeds[0].title):
        
        reaction_map = {"🪨": "rock", "📄": "paper", "✂️": "scissors"}
        
        if str(reaction.emoji) in reaction_map:
            move = reaction_map[str(reaction.emoji)]
            user_choice = game.get_user_choice(move)
            
            # Get player and check ML
            player = player_manager.get_player(user.id)
            
            ml_used = False
            confidence = 0
            predicted_move = None
            
            if player_manager.should_use_ml(player):
                player_manager.predictor.update_patterns(list(player.moves_history))
                predicted_move = player_manager.predictor.predict_next_move(list(player.moves_history))
                confidence = player_manager.predictor.get_confidence(list(player.moves_history))
                
                if predicted_move is not None and confidence > 30:
                    ml_used = True
            
            comp_choice = game.get_computer_choice(predicted_move if ml_used else None)
            outcome = game.determine_winner(user_choice, comp_choice)
            
            # Update player data
            player.add_game_result(user_choice, outcome, ml_used)
            
            # Send result
            embed = create_result_embed(user_choice, comp_choice, outcome, ml_used, confidence)
            await reaction.message.channel.send(f"{user.mention}", embed=embed)

# Run the bot
if __name__ == '__main__':
    # Replace with your bot token
    bot.run('MTQwNzM5NDI2NDEyMzU3MjM2NQ.GPnPXZ.oCGJZjMWbHPJbvc9NvLVaf9gNc9ZzdpY5nMTek')