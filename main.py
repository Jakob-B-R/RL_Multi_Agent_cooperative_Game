# We are building a bot to play "The Game", a cooperative multiplayer game
# Which involves playing cards in ascending or descending order on piles
# The players may co-operate slightly, signalling to each other which piles they want to play 
# on, weakly or strongly. The players win when all cards are played, they lose if a player can not
# play the minimum required cards on their turn.

import random
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from gymnasium import spaces
import numpy as np

class Player:
    def __init__(self, player_id, hand):
        self.player_id = player_id
        self.hand = hand
        self.seen_cards = []
        self.strategy = 'basic'
        
        
    def draw_card(self, deck):
        if len(deck) > 0:
            card = deck.pop()
            self.hand.append(card)
            return card
        return None
    
    
    def player_turn(self, pile_up, pile_down, num_to_play, claimed_pile, debug=False):
        
        card, score, pile = self.determine_best_single_card(pile_up, pile_down, self.hand, self.seen_cards, claimed_pile)
        if debug:
            print(f'Player: {self.player_id} played {card} on pile {pile}. SCORE: {score}')
        return card, pile
    
    
    def claim_pile(self, piles_up, piles_down, claimed_piles, debug=False):
        
        CLAIM_HARD = 8
        CLAIM_SOFT = 2
        
        claims = {}
        for p in piles_up:
            pc = claimed_piles.get(p, 0)
            for c in self.hand:
                if c == p - 10 and CLAIM_HARD > pc:
                    claims[p] = CLAIM_HARD
                if c - p in [1,2] and CLAIM_SOFT > pc:
                    claims[p] = CLAIM_SOFT
                    
        for p in piles_down:
            pc = claimed_piles.get(p, 0)
            for c in self.hand:
                if c == p + 10 and CLAIM_HARD > pc:
                    claims[p] = CLAIM_HARD
                if p - c in [1,2] and CLAIM_SOFT > pc:
                    claims[p] = CLAIM_SOFT    
                    
                    
        if debug:
            print(f'Player {self.player_id} claims: {claims} (hand: {self.hand})')
        return claims                
    
    def determine_best_single_card(self, up_piles, down_piles, hand, cards_seen, claimed_piles):
        MIN_SCORE = 101
        MIN_CARD = None
        MIN_PILE = 101
        for card in hand:

            cur_score = 101
            cur_pile = 101
            for pile in up_piles:
                claim_score = 0 # claimed_piles.get(pile, 0)
                if card == (pile - 10):
                    cur_score = -10 + claim_score
                    cur_pile = pile
                elif card > pile:
                    cur_score = card - pile + claim_score
                    cur_pile = pile
                    
                if card - 10 in cards_seen:
                    cur_score += 0.1
                    
            for pile in down_piles:
                claim_score = 0 # claimed_piles.get(pile, 0)
                if card == (pile + 10):
                    cur_score = -10 + claim_score
                    cur_pile = pile

                elif card < pile and cur_score > (pile-card):
                    cur_score = pile - card + claim_score
                    cur_pile = pile

                if card + 10 in cards_seen:
                    cur_score += 0.1
                    
            if cur_score < MIN_SCORE:
                MIN_SCORE = cur_score
                MIN_CARD = card
                MIN_PILE = cur_pile

            # print(card, cur_score)
                
        return MIN_CARD, MIN_SCORE, MIN_PILE
    
    def __repr__(self):
        return f'Player {self.player_id} Hand: {self.hand} Seen: {self.seen_cards}'
    
    
class Game(gym.Env):
    
    def __init__(self, num_players = 4, hand_size = 6, num_must_play = 2, pile_config=(2,2), pile_starts=(1,100)):
        super(Game, self).__init__()
        
        self.num_players = num_players
        self.hand_size = hand_size
        self.num_must_play = num_must_play
        self.num_must_play_init = num_must_play
        self.pile_config = pile_config
        self.pile_starts = pile_starts
        self.current_player_num_played = 0
        self.phase = 0  # 0 = play phase, 1 = claim phase
        
        # self.action_space = spaces.Dict({
        #     "action_type": spaces.Discrete(2),       # 0 = play card, 1 = claim pile
        #     "card_index": spaces.Discrete(hand_size),# only used if play card
        #     "pile_index": spaces.Box(low=0, high=2, shape=(1, pile_config[0] + pile_config[1]), dtype=int),  # used for both
        # })
        
        # [0 or 1 -- action
        # , 0 to 5 -- card index
        # , 0 to 3^num_piles, 0 means all zeros, 3^n+6 means strong claims on all piles]
        self.action_space = spaces.MultiDiscrete([2, hand_size, 3**(pile_config[0] + pile_config[1])])
        
        self.observation_space = spaces.Dict({
            # current hand
            'player_hand': spaces.MultiDiscrete([2] * 98),
            'piles_up': spaces.MultiDiscrete([101] * pile_config[0]),
            'piles_down': spaces.MultiDiscrete([101] * pile_config[1]),
            # each player can output 0=no claim, 1=soft claim, 2=hard claim
            'claimed_piles': spaces.Box(low=0, high=2, shape=(pile_config[0] + pile_config[1], num_players), dtype=int),
            'deck_size': spaces.Discrete(99),
            'cards_to_play': spaces.Discrete(num_must_play + 1),
            'phase': spaces.Discrete(2),  # 0 = play phase, 1 = claim phase
        })
        
        self.reset()
        
    def reset(self, seed=None, game_state=None, **kwargs):
        
        super().reset(seed=seed)
        random.seed(seed)
        self.claim_phase_turn = 0

        # reset the number at reset
        self.num_must_play = self.num_must_play_init
        self.current_player_num_played = 0
        # TODO need to make a function to allow players to decide who starts
        self.TURN = 0
        self.claimed_piles = {}
        
        self.deck = [i for i in range(2,100)]
        random.shuffle(self.deck)
        
        self.piles_up = [self.pile_starts[0] for _ in range(self.pile_config[0])]
        self.piles_down = [self.pile_starts[1] for _ in range(self.pile_config[1])]
        
        self.players = self.initialize_players()
        self.turn = 0
        self.game_not_done = True
        
        return self._get_observation(), {}
    
    
    def set_game_state(self, config):
        '''{
            "num_to_play": 2,
            "piles_up": [1,1],
            "piles_down": [100, 100],
            "players": [
                {
                    "name": "P0",
                    "hand": [10, 20, 30, 40, 50, 60]
                },
                {
                    "name": "P1",
                    "hand": [11, 21, 31, 41, 51, 61]
                }
            ]
        }'''
        
        self.num_players = len(config['players'])
        self.hand_size = config.get('hand_size', 6)
        self.num_must_play = config.get('num_to_play', 2)
        self.num_must_play_init = config.get('num_to_play', self.num_must_play)
        
        self.piles_up = config.get('piles_up')
        self.piles_down = config.get('piles_down')
        
        self.phase = config.get('phase', 0)
        self.pile_config = (len(self.piles_up), len(self.piles_down))
        self.pile_starts = (1,100)
        self.TURN = config.get('active_player', 0)
        self.deck = config.get('deck', [i for i in range(2,100) if i not in sum([p['hand'] for p in config['players']], [])])
        
        if config.get('deck') is None:
            random.shuffle(self.deck)
        self.game_not_done = True
        self.claim_phase_turn = 0
        
        deck_cpy = self.deck.copy()
        self.players = self.initialize_players()
        
        print(self.players)
        print(self.pile_config)
        for p in range(len(self.players)):
            h = self.players[p].hand
            cur_player = self.players[p]
            config_hand = config['players'][p].get('hand')
            if len(config_hand) == self.hand_size:
                cur_player.hand = config['players'][p].get('hand', h)
                
        if 'deck' in config:
            self.deck = deck_cpy

        else:
            self.deck = config.get('deck', [i for i in range(2,100) if i not in sum([p.hand for p in self.players], [])])
            random.shuffle(self.deck)


            
    
    def _base_3_conv(self,n, p=4):
        arr = [0]*p
        for i in range(p):
            arr[-i-1] = n%3
            n = n // 3
            if n == 0:
                break
        return arr
        
        
    def initialize_players(self):
        players = []
        for i in range(self.num_players):
            hand = []
            for _ in range(self.hand_size):
                hand.append(self.deck.pop())
            player = Player(i, hand)
            players.append(player)
            
        return players
    
    def render(self, mode='human'):
        return self.print_game()
    
    
    def _get_observation(self):
        current_player = self.players[self.TURN]
        
        hand_obs = np.array([0]*98)
        for c in current_player.hand:
            hand_obs[c-2] = 1
            
        piles_up_obs = np.array(self.piles_up.copy())
        piles_down_obs = np.array(self.piles_down.copy())
        
        claimed_obs = np.zeros((self.pile_config[0] + self.pile_config[1], self.num_players), dtype=int)
        
        for p in self.claimed_piles.keys():
            pile_index = None
            if p in self.piles_up:
                pile_index = self.piles_up.index(p)
            elif p in self.piles_down:
                pile_index = len(self.piles_up) + self.piles_down.index(p)
            else:
                continue
            
            for player_id in self.claimed_piles[p]:
                claimed_obs[pile_index][player_id] = 1 if p not in current_player.hand else 2
                
        obs = {
            'player_hand': hand_obs,
            'piles_up': piles_up_obs,
            'piles_down': piles_down_obs,
            'claimed_piles': claimed_obs,
            'deck_size': len(self.deck),
            'cards_to_play': self.num_must_play,
            'phase': self.phase
        }
                
        return obs
    
    def step(self, action):
        # action is now a 3-array
        
        if self.phase == 0:
            if action[0] == 1:
                # can't claim in play phase
                return self._get_observation(), -10, True, True, {}
            current_player = self.players[self.TURN]
            self.claim_phase_turn = 0
        elif self.phase == 1:
            if action[0] == 0:
                # can't play in claim phase
                return self._get_observation(), -10, True, True, {}
            current_player = self.players[self.claim_phase_turn]
            
        if action[0] == 0:
            card, _, _ = current_player.determine_best_single_card(self.piles_up, self.piles_down, current_player.hand, [], self.claimed_piles)
            if card is None:
                return self._get_observation(), -5, True, False, {}
        
        action_type = action[0]
        card_index = action[1]
        pile_index = action[2]
        
        # convert number from 0 to 3^n to base 3 array
        # 0 = [0,0,0,0,0,0]
        
        pile_len = self.pile_config[0] + self.pile_config[1]
        pile_list = self._base_3_conv(n=pile_index,p=pile_len)
                    
        # print(pile_list)
        
        # play card
        # I haven't checked this logic yet, that's a lot of funky index management
        if action_type == 0:
            self.current_player_num_played += 1
            
            # if placing a card, only one pile can be requested
            count_piles = 0
            for p in range(len(pile_list)):
                if pile_list[p] != 0:
                    count_piles += 1
                    pile_num = p
                                        
            if count_piles!= 1:
                # invalid action
                return self._get_observation(), -10, True, True, {}
            
            if card_index >= len(current_player.hand):
                # invalid play
                return self._get_observation(), -10, True, True, {}
            
            card = current_player.hand[card_index]
            # print('card to play: ', card)
            if pile_num < len(self.piles_up):
                pile = self.piles_up[pile_num]
                if card == (pile - 10) or card > pile:
                    self.piles_up.remove(pile)
                    self.piles_up.append(card)
                    current_player.hand.remove(card)
                    
                    # score should be +1 for the 10 card gap
                    # and 0->0.5 
                    
                    if card == pile-10:
                        score = 1
                        
                    else:
                        score = 1 + -(pile - card)**2/(2*(98**2))
                else:
                    # invalid play
                    return self._get_observation(), -10, True, True, {}
                    
            elif pile_num - len(self.piles_up) < len(self.piles_down):
                pile = self.piles_down[pile_num - len(self.piles_up)]
                if card == (pile + 10) or card < pile:
                    self.piles_down.remove(pile)
                    self.piles_down.append(card)
                    current_player.hand.remove(card)
                    
                    # score = (card - pile) / 10 + 10
                    if card == pile+10:
                        score = 1
                        
                    else:
                        score = 1 + -(card - pile)**2/(2*(98**2))
                else:
                    # invalid play
                    return self._get_observation(), -10, True, True, {}
            else:
                # invalid pile index
                return self._get_observation(), -10, True, True, {}
            
            if self.check_if_game_done():
                return self._get_observation(), 1000, True, False, {}
                    
        # claim a pile
        elif action_type == 1:
            score = 0
            for i in range(len(pile_list)):
                pile_claim_strength = pile_list[i]
                pile = i
                
                if pile_claim_strength not in [0,1,2]:
                    # invalid claim strength
                    return self._get_observation(), -10, True, True, {}
                
                if pile_claim_strength == 0:
                    continue
                
                claims = self.claimed_piles.get(pile, [])
                if current_player.player_id not in claims:
                    claims.append(current_player.player_id)
                else :
                    # already claimed
                    return self._get_observation(), -10, True, True, {}
                    
                self.claimed_piles[pile] = claims
                
            self.claim_phase_turn += 1
              
        # A card is played, enter the claim phase  
        if action_type == 0:
            
            # The last card was played
            if self.current_player_num_played == self.num_must_play:
                self.current_player_num_played = 0
                self.TURN = ((self.TURN + 1) % self.num_players)
            
                while len(self.deck) > 0 and len(current_player.hand) < self.hand_size:
                    new_card = self.deck.pop()
                    current_player.hand.append(new_card)
                
                if len(self.deck) == 0:
                    self.num_must_play = 1     
            
            # after every card is played, start the claiming 
            self.phase = 1
            self.claim_phase_turn = 0
            self.claimed_piles = {}
            
        # if it's the end of the claim phase:
        elif self.phase == 1 and current_player == self.players[-1]:
            self.phase = 0
            self.claim_phase_turn = 0  
                    
        return self._get_observation(), score, False, False, {}
            
                  
    def play_game(self, debug=False):
        
        game_not_done = True
        while game_not_done:
            
            current_player = self.players[self.TURN]
            
            for i in range(self.num_must_play):    
                if debug:
                    self.print_game()        
                if self.check_if_game_done():
                    game_not_done = False
                    
                    if debug:
                        print('We won!')
                    break
                
                # Give the player the game state, and ask for a result
                card, pile = current_player.player_turn(self.piles_up, self.piles_down, self.num_must_play, self.claimed_piles, debug)
                
                if card is None:
                    game_not_done = False
                    
                    if debug:
                        print('We Lose')
                    break
                
                # Handle the play
                if pile in self.piles_up:
                    self.piles_up.remove(pile)
                    self.piles_up.append(card)
                    current_player.hand.remove(card)
                    
                elif pile in self.piles_down:
                    self.piles_down.remove(pile)
                    self.piles_down.append(card)
                    current_player.hand.remove(card)

                    
                else:
                    raise ValueError(f'Card {pile} not in piles {self.piles_up}|{self.piles_down}')
                
                if pile in self.claimed_piles:
                    self.claimed_piles.pop(pile)
                    
                # Allow the player to draw cards
                while len(self.deck) > 0 and len(current_player.hand) < self.hand_size and self.num_must_play == i+1:
                    new_card = self.deck.pop()
                    current_player.hand.append(new_card)
                    if debug:
                        print(f'Player: {current_player.player_id} draws {new_card}')
                        
                if len(self.deck) == 0:
                    self.num_must_play = 1
                    
                # Ask each player if they want to reserve a pile
                self.claimed_piles = {}
                for p in self.players:
                    # don't claim when it's still your turn
                    # don't claim when it's your turn next
                    # probably should jus tmake a way to ignore your own claims...
                    # if (p == current_player and self.num_must_play != i+1) or (p.player_id == ((self.TURN + 1) % self.num_players) and self.num_must_play == i+1):
                    #     continue
                    claim_piles = p.claim_pile(self.piles_up, self.piles_down, self.claimed_piles, debug)
                    for c in claim_piles.keys():
                        self.claimed_piles[c] = claim_piles[c]
            
            self.TURN = ((self.TURN + 1) % self.num_players)
            
    def check_if_game_done(self):
        num_cards = 0
        for p in self.players:
            num_cards += len(p.hand)
            
        return num_cards == 0
    
    
    def print_game(self):
        game_state = f'Phase: {self.phase}, Deck Size: {len(self.deck)}; piles: {self.piles_up}|{self.piles_down}; turn: {self.TURN}; claim_turn: {self.claim_phase_turn}'
        player_states = ','.join([f'P{p.player_id}: {p.hand}' for p in self.players])
        final = f'{game_state} Players: {player_states}, claims: {self.claimed_piles}'
        # print(final)
        return final
    
    def __str__(self):
        return self.print_game()
    
    def __repr__(self):
        return self.print_game()
    
        
    # def player_turn(self, player):
        
    #     for _ in range(self.num_must_play):
    #         print(f'Player {player.player_id} Piles: {self.piles_up}|{self.piles_down}, hand: {player.hand}, Deck: {len(self.deck)}')
    #         card, score, pile = determine_best_single_card(self.piles_up, self.piles_down, player.hand, player.seen_cards)
    #         if card is None:
    #             self.game_not_done = False
    #             print(f'We lost. Piles: {self.piles_up}|{self.piles_down}, hand: {player.hand}')
    #             break
            
    #         player.play_card(card, pile)
                
    #         print(f'Player: {player.player_id} played {card} on pile {pile}. SCORE: {score}')
            
    #         if pile in self.piles_up:
    #             self.piles_up.remove(pile)
    #             self.piles_up.append(card)
    #         if pile in self.piles_down:
    #             self.piles_down.remove(pile)
    #             self.piles_down.append(card)
          
    #     if self.game_not_done:
    #         while len(self.deck) > 0 and len(player.hand) < self.hand_size:
    #             new_card = player.draw_card(self.deck)
    #             if new_card is not None:
    #                 print(f'Player: {player.player_id} draws {new_card}')              
    #         self.turn = (self.turn + 1) % self.num_players


    
if __name__ == '__main__':
    
    from stable_baselines3 import PPO
    
    NUM_PLAYERS = 4
    HAND_SIZE = 6
    NUM_MUST_PLAY = 2  
    PILE_CONFIG = (2,2) # 2 ascending, 2 descending
    PILE_STARTS = (1,100)
    
    # game = Game(NUM_PLAYERS, HAND_SIZE, NUM_MUST_PLAY, (2,2), (1,100))
    
    
    gym.envs.registration.register(id='jaketest', entry_point=lambda: Game(NUM_PLAYERS, HAND_SIZE, NUM_MUST_PLAY, (2,2), (1,100)))
    
    env = gym.make('jaketest')
    check_env(env)
    
    obs = env.reset()
    
    # done = False
    # while not done:
    #     action = env.action_space.sample()
    #     obs, reward, done, trunc, info = env.step(action)
    #     env.render()

    model = PPO('MultiInputPolicy', env, verbose =1)
    
    # model = PPO.load('ppo_jaketest_agent', env)
   
    model.learn(total_timesteps=1000000,    
                reset_num_timesteps=True,  # resets time counter
                # progress_bar=True
        )
   
    model.save('ppo_jaketest_agent')
   
    
    # game.play_game(debug=True)
    # NUM_PLAYERS = 4
    # HAND_SIZE = 6
    # NUM_MUST_PLAY = 2
    
    # DECK = [i for i in range(2,100)]
    
    # PILE_CONFIG = (2,2) # 2 ascending, 2 descending
    
    # PILE_STARTS = (1,100)
    
    # PILES_UP = [PILE_STARTS[0] for _ in range(PILE_CONFIG[0])]
    # PILES_DOWN = [PILE_STARTS[1] for _ in range(PILE_CONFIG[1])]
    
    # random.shuffle(DECK)
    
    # print(DECK)
    
    # player_states = []
    # for i in range(NUM_PLAYERS):
    #     player_id = i
        
    #     hand = []
    #     for _ in range(HAND_SIZE):
    #         hand.append(DECK.pop())
    #     player_data = {
    #         'player_name': i,
    #         'hand': hand,
    #         'seen_cards': []
    #     }
        
    #     player_states.append(player_data)
        
    # print(player_states)
    # print(PILES_UP, PILES_DOWN)
    
    # '''
    # Player 0 Piles: [1, 1]|[100, 100], hand: [95, 43, 30, 7, 56, 63], Deck: 86
    # Player: 0 played 95 on pile 100. SCORE: 5
    # Player 0 Piles: [1, 1]|[100, 95], hand: [43, 30, 7, 56, 63], Deck: 86
    # Player: 0 played 63 on pile 95. SCORE: 32
    # '''

    # # print(determine_best_single_card([1,1], [100, 100], [91,43,30,7,56,63], []))
    
    # GAME_NOT_DONE = True
    # TURN = 0
    # while GAME_NOT_DONE:
    #     if len(DECK) == 0:
    #         NUM_MUST_PLAY = 1
    #         deck_sum = 0
    #         for p in player_states:
    #             deck_sum += len(p['hand'])
                
    #         if deck_sum == 0:
    #             GAME_NOT_DONE = False
    #             print('WE WON')
    #             break
            
    #     player = player_states[TURN]
    #     for i in range(NUM_MUST_PLAY):
    #         print(f'Player {player['player_name']} Piles: {PILES_UP}|{PILES_DOWN}, hand: { player['hand']}, Deck: {len(DECK)}')
    #         # print(player)
    #         card, score, pile = determine_best_single_card(PILES_UP, PILES_DOWN, player['hand'], player['seen_cards'])
    #         if card is None:
    #             GAME_NOT_DONE = False
    #             print(f'We lost. Piles: {PILES_UP}|{PILES_DOWN}, hand: { player['hand']}')
    #             break
            
    #         player['seen_cards'].append(card)
    #         player['hand'].remove(card)
            

                
    #         print(f'Player: {player['player_name']} played {card} on pile {pile}. SCORE: {score}')
            
    #         if pile in PILES_UP:
    #             PILES_UP.remove(pile)
    #             PILES_UP.append(card)
    #         if pile in PILES_DOWN:
    #             PILES_DOWN.remove(pile)
    #             PILES_DOWN.append(card)
          
    #     if GAME_NOT_DONE:
    #         while len(DECK) > 0 and len(player['hand']) < HAND_SIZE:
    #             new_card = None
    #             if len(DECK) > 0:
    #                 new_card = DECK.pop()
    #                 player['hand'].append(new_card)
    #                 print(f'Player: {player['player_name']} draws {new_card}')              
    #         TURN = (TURN + 1) % NUM_PLAYERS
            
            
