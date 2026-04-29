# --- Constants For Training ---
BOARD_SIZE = 20
NUM_OBSTACLES = 10
NUM_FEATURES = 16
EPISODES = 5000
LEARNING_RATE = 0.005
GAMMA = 0.9 # Discount factor for future rewards
EPSILON = 0.1 # Exploration rate => helps to skip local optimums!

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]

# Coordinate changes for actions: (d_row, d_col)
DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# Relative directions
REL_LEFT = {UP: LEFT, DOWN: RIGHT, LEFT: DOWN, RIGHT: UP}
REL_RIGHT = {UP: RIGHT, DOWN: LEFT, LEFT: UP, RIGHT: DOWN}



# --- Play Module Configs & styles ---
CELL = 28           # pixels per cell
DELAY = 90          # milliseconds between frames
MAX_AUTO_STEPS = 3000

COLORS = {
    'bg':       '#6cbd45',
    'grid':     '#1c1c1c',
    'grid_line': '#252525',
    'obstacle': '#1747e7',
    'food':     '#ff829d',
    'head':     '#04bfa4',
    'body':     "#6cbd45",
    'text':     "#FFFFFF",
    'btn_run':  "#00a808",
    'btn_run_fr':  "#119718",
    'btn_stop': "#ff0000",
    'btn_stop_fr': "#9e0000",
}
