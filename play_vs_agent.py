#!/usr/bin/env python3
"""
CLI Interface to play "The Game" against the trained RL agent.
You play as Player 0, and the agent controls the other players.
"""

import sys
import gymnasium as gym
from main import Game

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def clear_screen():
    print("\033[2J\033[H", end="")

def print_game_state(game, show_hand=True):
    """Display the current game state in a nice format."""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}THE GAME - Cooperative Card Game{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}\n")
    
    # Display piles
    print(f"{Colors.CYAN}Ascending Piles (play higher, or exactly -10):{Colors.ENDC}")
    for i, pile in enumerate(game.piles_up):
        print(f"  Pile {i}: [{Colors.GREEN}{pile:3d}{Colors.ENDC}] ↑")
    
    print(f"\n{Colors.CYAN}Descending Piles (play lower, or exactly +10):{Colors.ENDC}")
    for i, pile in enumerate(game.piles_down):
        print(f"  Pile {i + len(game.piles_up)}: [{Colors.RED}{pile:3d}{Colors.ENDC}] ↓")
    
    print(f"\n{Colors.YELLOW}Deck: {len(game.deck)} cards remaining{Colors.ENDC}")
    print(f"{Colors.YELLOW}Cards to play this turn: {game.num_must_play}{Colors.ENDC}")
    print(f"{Colors.YELLOW}Current player: {game.TURN} {'(YOU!)' if game.TURN == 0 else '(Agent)'}{Colors.ENDC}")
    
    if show_hand:
        print(f"\n{Colors.BOLD}Your Hand (Player 0):{Colors.ENDC}")
        player = game.players[0]
        sorted_hand = sorted(player.hand)
        for idx, card in enumerate(sorted_hand):
            print(f"  [{idx}] {Colors.BOLD}{card:3d}{Colors.ENDC}", end="")
        print("\n")
    
    # Show other players' hand sizes (not cards)
    print(f"{Colors.CYAN}Other Players:{Colors.ENDC}")
    for p in game.players[1:]:
        print(f"  Player {p.player_id}: {len(p.hand)} cards")
    print()

def get_valid_plays(game, player_idx=0):
    """Get all valid plays for a player."""
    player = game.players[player_idx]
    valid_plays = []
    
    for card_idx, card in enumerate(player.hand):
        # Check ascending piles
        for pile_idx, pile in enumerate(game.piles_up):
            if card > pile or card == pile - 10:
                valid_plays.append((card_idx, pile_idx, card, pile, "up"))
        
        # Check descending piles
        for pile_idx, pile in enumerate(game.piles_down):
            actual_pile_idx = len(game.piles_up) + pile_idx
            if card < pile or card == pile + 10:
                valid_plays.append((card_idx, actual_pile_idx, card, pile, "down"))
    
    return valid_plays

def display_valid_plays(valid_plays, game):
    """Display valid plays in a readable format."""
    if not valid_plays:
        print(f"{Colors.RED}No valid plays available!{Colors.ENDC}")
        return
    
    print(f"{Colors.GREEN}Valid plays:{Colors.ENDC}")
    for i, (card_idx, pile_idx, card, pile, direction) in enumerate(valid_plays):
        trick = ""
        if direction == "up" and card == pile - 10:
            trick = f" {Colors.YELLOW}(±10 TRICK!){Colors.ENDC}"
        elif direction == "down" and card == pile + 10:
            trick = f" {Colors.YELLOW}(±10 TRICK!){Colors.ENDC}"
        
        arrow = "↑" if direction == "up" else "↓"
        print(f"  - Card {card} (index {card_idx}) → Pile {pile_idx} ({pile} {arrow}){trick}")

def get_human_action(game):
    """Get action from human player via CLI."""
    valid_plays = get_valid_plays(game, 0)
    
    if not valid_plays:
        return None
    
    player = game.players[0]
    sorted_hand = sorted(player.hand)
    
    while True:
        try:
            print(f"\n{Colors.BOLD}Your turn!{Colors.ENDC}")
            
            card_input = input(f"Enter card index (0-{len(sorted_hand)-1}) or 'q' to quit: ").strip()
            if card_input.lower() == 'q':
                return 'quit'
            
            sorted_idx = int(card_input)
            if sorted_idx < 0 or sorted_idx >= len(sorted_hand):
                print(f"{Colors.RED}Invalid card index!{Colors.ENDC}")
                continue
            
            # Get the actual card from sorted hand, then find its real index
            card = sorted_hand[sorted_idx]
            real_card_idx = player.hand.index(card)
            
            pile_input = input(f"Enter pile index (0-{len(game.piles_up) + len(game.piles_down) - 1}): ").strip()
            pile_idx = int(pile_input)
            
            if pile_idx < 0 or pile_idx >= len(game.piles_up) + len(game.piles_down):
                print(f"{Colors.RED}Invalid pile index!{Colors.ENDC}")
                continue
            if pile_idx < len(game.piles_up):
                pile = game.piles_up[pile_idx]
                if card > pile or card == pile - 10:
                    return [real_card_idx, pile_idx]
                else:
                    print(f"{Colors.RED}Invalid play! Card {card} cannot go on ascending pile {pile}{Colors.ENDC}")
                    print(f"(Must be > {pile} or exactly {pile - 10})")
            else:
                down_idx = pile_idx - len(game.piles_up)
                pile = game.piles_down[down_idx]
                if card < pile or card == pile + 10:
                    return [real_card_idx, pile_idx]
                else:
                    print(f"{Colors.RED}Invalid play! Card {card} cannot go on descending pile {pile}{Colors.ENDC}")
                    print(f"(Must be < {pile} or exactly {pile + 10})")
                    
        except ValueError:
            print(f"{Colors.RED}Please enter valid numbers!{Colors.ENDC}")
        except KeyboardInterrupt:
            return 'quit'

def agent_action(model, obs, game, player_idx):
    """Get action from the trained agent."""
    # The agent uses the observation for the current player
    action, _ = model.predict(obs, deterministic=True)
    return action

def play_game():
    """Main game loop for playing against the agent."""
    from stable_baselines3 import PPO
    
    # Configuration
    NUM_PLAYERS = 4
    HAND_SIZE = 6
    NUM_MUST_PLAY = 2  
    PILE_CONFIG = (2, 2)
    PILE_STARTS = (1, 100)
    
    # Create the game environment
    game = Game(NUM_PLAYERS, HAND_SIZE, NUM_MUST_PLAY, PILE_CONFIG, PILE_STARTS)
    
    # Try to load the trained model
    try:
        model = PPO.load('ppo_jaketest_agent', env=game)
        print(f"{Colors.GREEN}Loaded trained agent successfully!{Colors.ENDC}")
    except FileNotFoundError:
        print(f"{Colors.RED}Warning: Could not load trained agent 'ppo_jaketest_agent'.{Colors.ENDC}")
        print("The agent will make random moves instead.")
        model = None
    
    obs, _ = game.reset()
    done = False
    total_cards_played = 0
    
    clear_screen()
    print(f"\n{Colors.BOLD}{Colors.HEADER}Welcome to The Game!{Colors.ENDC}")
    print("You are Player 0. Work together with the AI agents to play all cards!")
    print("Press Enter to start...")
    input()
    
    while not done:
        clear_screen()
        print_game_state(game)
        
        current_player = game.TURN
        cards_played_this_turn = 0
        
        while cards_played_this_turn < game.num_must_play and not done:
            if current_player == 0:
                # Human player's turn
                action = get_human_action(game)
                
                if action == 'quit':
                    print(f"\n{Colors.YELLOW}Thanks for playing!{Colors.ENDC}")
                    return
                
                if action is None:
                    print(f"\n{Colors.RED}You have no valid plays! Game Over.{Colors.ENDC}")
                    done = True
                    break
                
                # Execute human player's action
                obs, reward, done, truncated, info = game.step(action)
                
                if reward > 0:
                    total_cards_played += 1
                    cards_played_this_turn += 1
                    print(f"{Colors.GREEN}Card played successfully! (Reward: {reward:.2f}){Colors.ENDC}")
                    if not done:
                        clear_screen()
                        print_game_state(game)
                else:
                    print(f"{Colors.RED}Invalid move!{Colors.ENDC}")
                    
            else:
                # Agent's turn
                print(f"{Colors.CYAN}Player {current_player} (Agent) is thinking...{Colors.ENDC}")
                
                player = game.players[current_player]
                agent_retry_count = 0
                max_agent_retries = 3
                
                while agent_retry_count < max_agent_retries:
                    if model is not None and agent_retry_count == 0:
                        action, _ = model.predict(obs, deterministic=True)
                    else:
                        # Fallback: use the basic Player AI to find a valid move
                        card, score, pile = player.determine_best_single_card(
                            game.piles_up, game.piles_down, player.hand, [], {}
                        )
                        if card is None:
                            # No valid moves - game over
                            print(f"{Colors.RED}Agent has no valid plays!{Colors.ENDC}")
                            done = True
                            break
                        
                        # Convert to action format
                        card_idx = player.hand.index(card)
                        if pile in game.piles_up:
                            pile_idx = game.piles_up.index(pile)
                        else:
                            pile_idx = len(game.piles_up) + game.piles_down.index(pile)
                        action = [card_idx, pile_idx]
                        
                        if agent_retry_count > 0:
                            print(f"{Colors.YELLOW}(Using fallback AI){Colors.ENDC}")
                    
                    # Display agent's action
                    if action[0] < len(player.hand):
                        card = player.hand[action[0]]
                        pile_idx = action[1]
                        if pile_idx < len(game.piles_up):
                            pile_val = game.piles_up[pile_idx]
                            direction = "↑"
                        else:
                            pile_val = game.piles_down[pile_idx - len(game.piles_up)]
                            direction = "↓"
                        print(f"  Agent plays: {card} → Pile {pile_idx} ({pile_val} {direction})")
                    
                    obs, reward, done, truncated, info = game.step(action)
                    
                    if reward > 0:
                        total_cards_played += 1
                        cards_played_this_turn += 1
                        print(f"{Colors.GREEN}Card played successfully! (Reward: {reward:.2f}){Colors.ENDC}")
                        break
                    else:
                        if done:
                            break
                        agent_retry_count += 1
                        if agent_retry_count < max_agent_retries:
                            print(f"{Colors.YELLOW}Agent made invalid move, trying fallback...{Colors.ENDC}")
                
                input("Press Enter to continue...")
    
    # Game ended
    clear_screen()
    print_game_state(game, show_hand=False)
    
    if game.check_if_game_done():
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 CONGRATULATIONS! You won! 🎉{Colors.ENDC}")
        print(f"All cards have been played!")
    else:
        remaining = sum(len(p.hand) for p in game.players) + len(game.deck)
        print(f"\n{Colors.RED}{Colors.BOLD}Game Over!{Colors.ENDC}")
        print(f"Unable to continue. {remaining} cards remaining.")
    
    print(f"\nTotal cards played: {total_cards_played}")
    print(f"Final piles: Up {game.piles_up} | Down {game.piles_down}")

def main():
    """Entry point."""
    print(f"{Colors.BOLD}The Game - Human vs Agent CLI{Colors.ENDC}")
    print("-" * 40)
    
    while True:
        play_game()
        
        again = input("\nPlay again? (y/n): ").strip().lower()
        if again != 'y':
            break
    
    print(f"\n{Colors.CYAN}Thanks for playing!{Colors.ENDC}")

if __name__ == '__main__':
    main()
