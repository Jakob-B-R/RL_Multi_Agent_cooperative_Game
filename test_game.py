from main import Game, Player
import math

# [0 or 1 -- action
# , 0 to 5 -- card index
# , 0 to 3^num_piles, 0 means all zeros, 3^n+6 means strong claims on all piles]
# self.action_space = spaces.MultiDiscrete([2, hand_size, 3**(pile_config[0] + pile_config[1])])

# two play game, 6 card hands, must play 2, regular piles

def test_basic_play():
    # game = Game(num_players=2,hand_size=6,num_must_play=2, pile_config=(2,2), pile_starts=(1,100))

    game_initial_state = {
        "num_to_play": 2,
        "piles_up": [1,2],
        "piles_down": [100, 99],
        "players": [
            {
                "name": "0",
                "hand": [10, 20, 30, 40, 50, 60]
            },
            {
                "name": "1",
                "hand": [11, 21, 31, 41, 51, 61]
            }
            ],
        }   
    
    
    game = Game()
    game.set_game_state(game_initial_state)
    print(game)

    # self._get_observation(), 100, True, False, {}

    # play a card, play the first card, on the last pile
    obs, reward, done, trunc, _ = game.step([0, 0, 27])
    print(reward, done)
    print(game)
    
    assert(not done)

    obs, reward, done, trunc, _ = game.step([0, 0, 1])
    print(reward, done)
    print(game)

    assert(done)
    
# 0 = 0,0,0,0
# 1 = 0,0,0,1
# 2 = 0,0,0,2
# 3 = 0,0,1,0
# 4 = 0,0,1,1
def test_base_3_conv():
    for i in range(82):
        res = base_3_conv(i)
        print(i, res)

def base_3_conv(n, p=4):
    arr = [0]*p
    for i in range(p):
        arr[-i-1] = n%3
        n = n // 3
        if n == 0:
            break
    return arr


def test_claiming_piles():
    pass

def test_reducing_piles():
    game_initial_state = {
        "num_to_play": 2,
        "piles_up": [1,2],
        "piles_down": [100, 99],
        "players": [
            {
                "name": "0",
                "hand": [10, 20, 30, 40, 50, 60]
            },
            {
                "name": "1",
                "hand": [11, 21, 31, 41, 51, 61]
            }
            ],
        }   
    
    game = Game()
    game.set_game_state(game_initial_state)
    print(game)
    
    # play a card, play the last card, on the first pile
    print('play 60...')
    obs, reward, done, trunc, _ = game.step([0, 5, 27])
    print(reward, done)
    print(game)
    
    game.step([1,0,0])
    game.step([1,0,0])

    
    print('play 50...')
    obs, reward, done, trunc, _ = game.step([0, 4, 9])
    print(reward, done)
    print(game)
    
    game.step([1,0,0])
    game.step([1,0,0])
    
    # player 2
    obs, reward, done, trunc, _ = game.step([0, 0, 1])
    print(reward, done)
    print(game)
    
    game.step([1,0,0])
    game.step([1,0,0])
    
    # player 2
    obs, reward, done, trunc, _ = game.step([0, 0, 1])
    print(reward, done)
    print(game)
    
    game.step([1,0,0])
    game.step([1,0,0])
    
    # player 1
    obs, reward, done, trunc, _ = game.step([0, 3, 9])
    print(reward, done)
    print(game)
    
    game.step([1,0,0])
    game.step([1,0,0])
    
    obs, reward, done, trunc, _ = game.step([0, 2, 9])
    print(reward, done)
    print(game)
    
    game.step([1,0,0])
    game.step([1,0,0])
    
def simulate_play():
    game_initial_state = {
        "num_to_play": 2,
        "piles_up": [1,1],
        "piles_down": [100, 100],
        "players": [
            {
                "name": "0",
                "hand": []
            },
            {
                "name": "1",
                "hand": []
            }
        ],
    }   
    
    game = Game()
    game.set_game_state(game_initial_state)
    
    while True:
        print(game)
        action = int(input('Input Action (0=play, 1=claim) '))
        card_index = int(input('Input Card Index (0-5) '))
        pile_index = int(input('Input Pile Index (0-3) '))
        
        # 0 -> 1
        # 1 -> 3
        # 2 -> 9
        # 3 -> 27
        
        action_def = [action, card_index, 3**pile_index]
        
        game.step(action_def)
        
        game.step([1,0,0])
        game.step([1,0,0])

if __name__ == '__main__':
    
    
    # test_basic_play()
    
    # test_base_3_conv()
    
    # test_reducing_piles()
    
    simulate_play()