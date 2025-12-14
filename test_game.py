from main import Game, Player
import math

# Simplified action format: [card_index, pile_index]
# - card_index: which card from hand (0-5)
# - pile_index: which pile (0-3, where 0-1 are ascending, 2-3 are descending)


def test_basic_play():
    """Test basic card play with simplified action format."""
    game_initial_state = {
        "num_to_play": 2,
        "piles_up": [1, 2],
        "piles_down": [100, 99],
        "players": [
            {"name": "0", "hand": [10, 20, 30, 40, 50, 60]},
            {"name": "1", "hand": [11, 21, 31, 41, 51, 61]}
        ],
    }
    
    game = Game()
    game.set_game_state(game_initial_state)
    print(game)

    # New format: [card_index, pile_index]
    # Play card 0 (value 10) on pile 0 (ascending from 1)
    obs, reward, done, trunc, _ = game.step([0, 0])
    print(f"Reward: {reward}, Done: {done}")
    print(game)
    
    assert not done, "Game should not be done after first valid play"
    assert reward > 0, "Valid play should give positive reward"
    assert 10 in game.piles_up, "Card 10 should be on ascending pile"


def test_descending_pile_play():
    """Test playing on descending piles."""
    game_initial_state = {
        "num_to_play": 2,
        "piles_up": [1, 1],
        "piles_down": [100, 100],
        "players": [
            {"name": "0", "hand": [10, 20, 30, 40, 50, 90]},
            {"name": "1", "hand": [11, 21, 31, 41, 51, 61]}
        ],
    }
    
    game = Game()
    game.set_game_state(game_initial_state)
    
    # Play card 5 (value 90) on pile 2 (first descending pile)
    obs, reward, done, trunc, _ = game.step([5, 2])
    
    assert not done
    assert reward > 0
    assert 90 in game.piles_down


def test_invalid_card_index():
    """Test that invalid card index returns penalty."""
    game = Game(num_players=1, hand_size=6)
    obs, _ = game.reset()
    
    # Card index 10 is out of bounds
    obs, reward, done, trunc, _ = game.step([10, 0])
    assert done, "Game should end on invalid action"
    assert reward == -10, "Invalid card index should return -10 penalty"


def test_invalid_pile_play():
    """Test that playing invalid card on pile returns penalty."""
    game_initial_state = {
        "num_to_play": 2,
        "piles_up": [50, 50],
        "piles_down": [100, 100],
        "players": [
            {"name": "0", "hand": [10, 20, 30, 40, 45, 60]},
            {"name": "1", "hand": [11, 21, 31, 41, 51, 61]}
        ],
    }
    
    game = Game()
    game.set_game_state(game_initial_state)
    
    # Try to play card 0 (value 10) on ascending pile at 50 - invalid!
    obs, reward, done, trunc, _ = game.step([0, 0])
    assert done, "Invalid play should end game"
    assert reward == -10, "Invalid play should return -10 penalty"


def test_ten_trick_ascending():
    """Test the -10 trick on ascending pile."""
    game_initial_state = {
        "num_to_play": 2,
        "piles_up": [50, 50],
        "piles_down": [100, 100],
        "players": [
            {"name": "0", "hand": [40, 55, 60, 70, 80, 90]},
            {"name": "1", "hand": [11, 21, 31, 41, 51, 61]}
        ],
    }
    
    game = Game()
    game.set_game_state(game_initial_state)
    
    # Play card 0 (value 40) on pile 0 (at 50) - this is the -10 trick!
    obs, reward, done, trunc, _ = game.step([0, 0])
    
    assert not done
    assert reward == 1, "The -10 trick should give max reward of 1"
    assert 40 in game.piles_up


def test_ten_trick_descending():
    """Test the +10 trick on descending pile."""
    game_initial_state = {
        "num_to_play": 2,
        "piles_up": [1, 1],
        "piles_down": [50, 50],
        "players": [
            {"name": "0", "hand": [60, 55, 30, 20, 10, 5]},
            {"name": "1", "hand": [11, 21, 31, 41, 51, 61]}
        ],
    }
    
    game = Game()
    game.set_game_state(game_initial_state)
    
    # Play card 0 (value 60) on pile 2 (descending at 50) - this is the +10 trick!
    obs, reward, done, trunc, _ = game.step([0, 2])
    
    assert not done
    assert reward == 1, "The +10 trick should give max reward of 1"
    assert 60 in game.piles_down


def test_turn_progression():
    """Test that turns progress correctly after playing required cards."""
    game_initial_state = {
        "num_to_play": 2,
        "piles_up": [1, 1],
        "piles_down": [100, 100],
        "players": [
            {"name": "0", "hand": [10, 20, 30, 40, 50, 60]},
            {"name": "1", "hand": [11, 21, 31, 41, 51, 61]}
        ],
    }
    
    game = Game()
    game.set_game_state(game_initial_state)
    
    assert game.TURN == 0, "Should start as player 0's turn"
    
    # Play first card
    game.step([0, 0])
    assert game.TURN == 0, "Still player 0's turn (need 2 plays)"
    
    # Play second card
    game.step([0, 0])
    assert game.TURN == 1, "Should now be player 1's turn"


def test_action_space_shape():
    """Verify the action space has correct shape."""
    game = Game(num_players=2, hand_size=6, pile_config=(2, 2))
    
    assert game.action_space.nvec.tolist() == [6, 4], \
        f"Action space should be [hand_size=6, num_piles=4], got {game.action_space.nvec.tolist()}"


def test_observation_space_keys():
    """Verify observation space has correct keys (no phase or claimed_piles)."""
    game = Game()
    obs, _ = game.reset()
    
    expected_keys = {'player_hand', 'piles_up', 'piles_down', 'deck_size', 'cards_to_play'}
    assert set(obs.keys()) == expected_keys, \
        f"Observation keys should be {expected_keys}, got {set(obs.keys())}"


if __name__ == '__main__':
    print("Running tests...")
    
    test_action_space_shape()
    print("✓ test_action_space_shape passed")
    
    test_observation_space_keys()
    print("✓ test_observation_space_keys passed")
    
    test_basic_play()
    print("✓ test_basic_play passed")
    
    test_descending_pile_play()
    print("✓ test_descending_pile_play passed")
    
    test_invalid_card_index()
    print("✓ test_invalid_card_index passed")
    
    test_invalid_pile_play()
    print("✓ test_invalid_pile_play passed")
    
    test_ten_trick_ascending()
    print("✓ test_ten_trick_ascending passed")
    
    test_ten_trick_descending()
    print("✓ test_ten_trick_descending passed")
    
    test_turn_progression()
    print("✓ test_turn_progression passed")
    
    print("\n✅ All tests passed!")