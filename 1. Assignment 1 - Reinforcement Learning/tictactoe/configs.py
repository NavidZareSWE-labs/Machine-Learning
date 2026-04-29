# Train Configs
ALPHA = 0.1
EPISODES = 80_000
NUM_FEATURES = 16

# Board indices for all eight winning lines
WINNING_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]

# Play Configs
SYMBOLS = {1: 'X', -1: 'O', 0: '.'}

# ANSI colour helpers (skipped on terminals that do not support them)
_GREEN = '\033[92m'
_RED = '\033[91m'
_YELLOW = '\033[93m'
_CYAN = '\033[96m'
_RESET = '\033[0m'