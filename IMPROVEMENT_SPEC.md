# The Game Simulator - RL Agent Improvement Specification

This document outlines tasks for improving the reinforcement learning agent that plays "The Game" cooperative card game.

## Project Overview

**Goal**: Train an RL agent to play optimal moves in "The Game" - a cooperative card game with 4 piles (2 ascending, 2 descending), 6-card hands, and a requirement to play 2 cards per turn.

**Current Problem**: The agent is not learning effectively due to issues with action space, reward design, and environment complexity.

**Main File**: `main.py` contains the `Game` (gym.Env) and `Player` classes.

---

## Task 1: Simplify Action Space

**Priority**: 🔴 High

### Problem
The current action space is `MultiDiscrete([2, 6, 81])` where:
- Index 0: action type (0=play, 1=claim)
- Index 1: card index (0-5)
- Index 2: pile encoded as base-3 number (0-80)

This is confusing because for play actions, only 4 values (`1`, `3`, `9`, `27`) are valid pile encodings.

### Requirements
1. Remove the claim phase entirely from the action space
2. Change action space to `MultiDiscrete([hand_size, num_piles])` = `[6, 4]`
3. Update `step()` to handle the new action format
4. Remove claim-related code paths from `step()`

### Code Changes

**In `__init__`:**
```python
# Replace:
self.action_space = spaces.MultiDiscrete([2, hand_size, 3**(pile_config[0] + pile_config[1])])

# With:
self.action_space = spaces.MultiDiscrete([hand_size, pile_config[0] + pile_config[1]])
```

**In `step()`:**
```python
def step(self, action):
    card_index = action[0]
    pile_index = action[1]
    
    current_player = self.players[self.TURN]
    
    # Remove all claim phase logic
    # Simplify to just handle card play
```

---

## Task 2: Implement Action Masking

**Priority**: 🔴 High

### Problem
Invalid actions give `-10` reward, dominating the learning signal. The agent learns to avoid penalties instead of good strategy.

### Requirements
1. Add `valid_actions` to observation space
2. Compute which (card, pile) combinations are legal
3. Integrate with `sb3-contrib.MaskablePPO`

### Code Changes

**In `_get_observation()`:**
```python
# Add valid action mask
num_actions = self.hand_size * (self.pile_config[0] + self.pile_config[1])
valid_mask = np.zeros(num_actions, dtype=np.int8)

for card_idx, card in enumerate(current_player.hand):
    for pile_idx in range(len(self.piles_up)):
        pile = self.piles_up[pile_idx]
        if card > pile or card == pile - 10:
            action_idx = card_idx * 4 + pile_idx
            valid_mask[action_idx] = 1
            
    for pile_idx in range(len(self.piles_down)):
        pile = self.piles_down[pile_idx]
        if card < pile or card == pile + 10:
            action_idx = card_idx * 4 + len(self.piles_up) + pile_idx
            valid_mask[action_idx] = 1

obs['action_mask'] = valid_mask
```

**In training:**
```python
from sb3_contrib import MaskablePPO

model = MaskablePPO('MultiInputPolicy', env, verbose=1)
```

---

## Task 3: Fix Reward Function

**Priority**: 🔴 High

### Problem
- Win reward (+1000) is too rare
- Invalid action penalty (-10) is too harsh
- No intermediate rewards to guide learning

### Requirements
1. Reduce invalid action penalty
2. Add shaping rewards for good plays
3. Add progress-based rewards

### Code Changes

**Create reward constants:**
```python
REWARD_VALID_PLAY = 1.0
REWARD_EXCELLENT_PLAY = 5.0  # For -10 trick (playing pile - 10 or pile + 10)
REWARD_WIN = 100.0
REWARD_INVALID = -0.5
REWARD_CANT_PLAY = -1.0

# Shaping rewards
REWARD_PER_DECK_CARD = 0.01  # Bonus for cards remaining in deck
REWARD_PILE_GAP_PENALTY = -0.001  # Small penalty per gap point
```

**In `step()` for valid plays:**
```python
if card == pile - 10 or card == pile + 10:
    score = REWARD_EXCELLENT_PLAY
else:
    gap = abs(card - pile)
    score = REWARD_VALID_PLAY - (gap * REWARD_PILE_GAP_PENALTY)
    
# Add progress bonus
score += len(self.deck) * REWARD_PER_DECK_CARD
```

---

## Task 4: Improve Observation Space

**Priority**: 🟡 Medium

### Problem
- Missing information about other players' hand sizes
- No explicit pile direction indicator
- Claimed piles mapped by value (which changes)

### Requirements
1. Add other players' hand sizes to observation
2. Add pile direction indicators
3. Remove or fix claimed_piles observation

### Code Changes

**Update observation_space:**
```python
self.observation_space = spaces.Dict({
    'player_hand': spaces.MultiDiscrete([2] * 98),
    'piles': spaces.Box(low=1, high=100, shape=(pile_config[0] + pile_config[1],), dtype=np.int32),
    'pile_directions': spaces.MultiBinary(pile_config[0] + pile_config[1]),  # 1=ascending, 0=descending
    'deck_size': spaces.Discrete(99),
    'cards_to_play': spaces.Discrete(num_must_play + 1),
    'other_hand_sizes': spaces.MultiDiscrete([hand_size + 1] * (num_players - 1)),
    'current_player': spaces.Discrete(num_players),
})
```

**Update `_get_observation()`:**
```python
# Combine piles into single array [up1, up2, down1, down2]
piles_obs = np.array(self.piles_up + self.piles_down, dtype=np.int32)

# Pile directions: 1 for ascending, 0 for descending
pile_dirs = np.array([1] * len(self.piles_up) + [0] * len(self.piles_down), dtype=np.int8)

# Other players' hand sizes
other_sizes = np.array([
    len(self.players[(self.TURN + i + 1) % self.num_players].hand) 
    for i in range(self.num_players - 1)
], dtype=np.int32)
```

---

## Task 5: Add Curriculum Learning

**Priority**: 🟡 Medium

### Problem
Full game is too hard for agent to learn from scratch.

### Requirements
1. Create method to start game in near-win state
2. Gradually increase deck size over training
3. Add difficulty parameter to environment

### Code Changes

**Add to `__init__`:**
```python
def __init__(self, ..., difficulty=1.0):
    self.difficulty = difficulty  # 0.0 = easy, 1.0 = full game
```

**Modify `reset()`:**
```python
def reset(self, seed=None, **kwargs):
    # ... existing reset code ...
    
    # Reduce deck based on difficulty
    cards_to_remove = int((1.0 - self.difficulty) * 80)
    self.deck = self.deck[cards_to_remove:]
    
    # ... rest of reset ...
```

**Create curriculum wrapper:**
```python
class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env, initial_difficulty=0.2):
        super().__init__(env)
        self.difficulty = initial_difficulty
        self.wins = 0
        
    def reset(self, **kwargs):
        self.env.difficulty = self.difficulty
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        if done and reward > 0:  # Win
            self.wins += 1
            if self.wins % 10 == 0:
                self.difficulty = min(1.0, self.difficulty + 0.1)
        return obs, reward, done, trunc, info
```

---

## Task 6: Reduce Player Count

**Priority**: 🟡 Medium

### Problem
4-player game is complex for single-agent RL.

### Requirements
1. Create 1-player (solo) variant
2. Test training with solo game first
3. Adjust hand_size and num_must_play for solo rules

### Code Changes

**Create solo game config:**
```python
# Solo variant (official rules)
SOLO_CONFIG = {
    'num_players': 1,
    'hand_size': 8,  # Larger hand for solo
    'num_must_play': 2,
    'pile_config': (2, 2),
    'pile_starts': (1, 100)
}

env = Game(**SOLO_CONFIG)
```

---

## Task 7: Improve Training Configuration

**Priority**: 🟢 Low

### Requirements
1. Increase training timesteps to 10M+
2. Tune PPO hyperparameters
3. Add logging and checkpoints

### Code Changes

**Replace training code in `__main__`:**
```python
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Wrap env for logging
env = Monitor(env)

# Callbacks
checkpoint_cb = CheckpointCallback(save_freq=100000, save_path='./checkpoints/')
eval_cb = EvalCallback(env, eval_freq=50000, best_model_save_path='./best_model/')

model = PPO(
    'MultiInputPolicy',
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,
    tensorboard_log='./tensorboard_logs/'
)

model.learn(
    total_timesteps=10_000_000,
    callback=[checkpoint_cb, eval_cb],
    progress_bar=True
)
```

---

## Suggested Implementation Order

1. **Task 1 + Task 3**: Simplify action space and fix rewards (foundational)
2. **Task 6**: Reduce to solo game for faster iteration
3. **Task 2**: Add action masking (requires sb3-contrib)
4. **Task 4**: Improve observations
5. **Task 7**: Better training config
6. **Task 5**: Add curriculum learning

---

## Testing

After each task, verify:
1. `check_env(env)` passes
2. Random agent can complete episodes without errors
3. Training shows improving reward curves

```python
from gymnasium.utils.env_checker import check_env

env = Game()
check_env(env)  # Should pass with no warnings

# Test random rollout
obs, _ = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    if done:
        obs, _ = env.reset()
```
