import random
import pickle
import random
from configs import *


#  Environment Setup
def _place_food(body, obstacles):
    while True:
        row, col = random.randint(
            0, BOARD_SIZE - 1), random.randint(0, BOARD_SIZE - 1)
        if (row, col) not in body and (row, col) not in obstacles:
            return (row, col)


def init_board():
    start_r = random.randint(5, BOARD_SIZE - 6)
    start_c = random.randint(5, BOARD_SIZE - 6)
    body = [(start_r, start_c), (start_r, start_c - 1), (start_r, start_c - 2)]
    direction = 1  # 0: Up, 1: Right, 2: Down, 3: Left

    occupied = set(body)

    obstacles = set()
    for _ in range(4):
        while True:
            r = random.randint(0, BOARD_SIZE - 2)
            c = random.randint(0, BOARD_SIZE - 2)

            block = {(r, c), (r+1, c), (r, c+1), (r+1, c+1)}

            if not block.intersection(occupied):
                obstacles.update(block)
                occupied.update(block)
                break

    # Food
    while True:
        food_row = random.randint(0, BOARD_SIZE - 1)
        food_col = random.randint(0, BOARD_SIZE - 1)
        if (food_row, food_col) not in occupied:
            food = (food_row, food_col)
            break

    return {
        'body': body,
        'direction': direction,
        'food': food,
        'obstacles': obstacles,
        'score': 0
    }


#  Game Logic
def get_valid_actions(current_dir):
    # Cannot directly reverse direction
    invalid_move = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
    return [a for a in ACTIONS if a != invalid_move[current_dir]]


def apply_action(state, action):
    head_r, head_c = state['body'][0]
    dr, dc = DELTA[action]
    new_head = (head_r + dr, head_c + dc)

    if new_head[0] < 0 or new_head[0] >= BOARD_SIZE or new_head[1] < 0 or new_head[1] >= BOARD_SIZE:
        return state, 'wall'
    if new_head in state['obstacles']:
        return state, 'obstacle'

    if new_head in state['body'][:-1]:
        return state, 'self'

    new_state = {
        'body': state['body'][:],
        'obstacles': state['obstacles'],
        'food': state['food'],
        'direction': action,
        'score': state['score']
    }

    new_state['body'].insert(0, new_head)

    if new_head == state['food']:
        new_state['score'] += 1
        new_state['food'] = _place_food(
            new_state['body'], new_state['obstacles'])
        if len(new_state['body']) == BOARD_SIZE * BOARD_SIZE - NUM_OBSTACLES:
            return new_state, 'won'
        return new_state, 'ate'
    else:
        new_state['body'].pop()
        return new_state, 'moved'


def extract_features(state):
    if state is None:
        return [0.0] * NUM_FEATURES

    body = state['body']
    food = state['food']
    obstacles = state['obstacles']
    direction = state['direction']

    head_row, head_col = body[0]
    food_row, food_col = food if food is not None else (head_row, head_col)
    body_set = set(body[1:])

    def is_dangerous(row, col):
        if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
            return 1.0
        if (row, col) in body_set:
            return 1.0
        if (row, col) in obstacles:
            return 1.0
        return 0.0

    # 1. Food Directions (Binary)
    food_above = 1.0 if food_row < head_row else 0.0
    food_below = 1.0 if food_row > head_row else 0.0
    food_left = 1.0 if food_col < head_col else 0.0
    food_right = 1.0 if food_col > head_col else 0.0

    # 2. Danger 1-step
    danger_up = is_dangerous(head_row - 1, head_col)
    danger_down = is_dangerous(head_row + 1, head_col)
    danger_left = is_dangerous(head_row, head_col - 1)
    danger_right = is_dangerous(head_row, head_col + 1)

    # 3. Relative Dangers based on current direction
    snake_dir_row, snake_dir_col = DELTA[direction]
    danger_straight = is_dangerous(
        head_row + snake_dir_row, head_col + snake_dir_col)

    dir_row_left, dir_col_left = DELTA[REL_LEFT[direction]]
    danger_rel_left = is_dangerous(
        head_row + dir_row_left, head_col + dir_col_left)

    dir_row_right, dir_col_right = DELTA[REL_RIGHT[direction]]
    danger_rel_right = is_dangerous(
        head_row + dir_row_right, head_col + dir_col_right)

    # 4. Distances
    manhattan = (abs(food_row - head_row) +
                 abs(food_col - head_col)) / (2.0 * BOARD_SIZE)
    snake_len = len(body) / float(BOARD_SIZE ** 2)

    # 5. Toward Food (Is the snake currently pointing towards the food?)
    toward_food = 0.0
    if (snake_dir_col > 0 and food_col > head_col) or (snake_dir_col < 0 and food_col < head_col):
        toward_food = 1.0
    elif (snake_dir_row > 0 and food_row > head_row) or (snake_dir_row < 0 and food_row < head_row):
        toward_food = 1.0

    # 6. Open Space (Look ahead 10 steps to avoid loops/traps)
    open_space = 0
    curr_r, curr_c = head_row, head_col
    for _ in range(10):
        curr_r += snake_dir_row
        curr_c += snake_dir_col
        if is_dangerous(curr_r, curr_c):
            break
        open_space += 1
    open_space_norm = open_space / 10.0

    # Element 0 is the bias unit = 1
    return [
        1.0, food_above, food_below, food_left, food_right,
        danger_up, danger_down, danger_left, danger_right,
        danger_straight, danger_rel_left, danger_rel_right,
        manhattan, snake_len, toward_food, open_space_norm,
    ]


def estimate_value(weights, features):
    return sum(w * f for w, f in zip(weights, features))


def run_episode(weights):
    state = init_board()

    steps = 0
    max_steps = 2000  # Prevent infinite loops

    while steps < max_steps:
        actions = get_valid_actions(state['direction'])

        # Epsilon-Greedy Exploration
        if random.random() < EPSILON:
            best_action = random.choice(actions)
            best_next_state, best_outcome = apply_action(state, best_action)

            if best_outcome in ('wall', 'obstacle', 'self'):
                v_target = -100.0
            elif best_outcome == 'won':
                v_target = 100.0
            else:
                reward = 50.0 if best_outcome == 'ate' else -0.1
                v_target = reward + GAMMA * \
                    estimate_value(weights, extract_features(best_next_state))
        else:
            best_val = -float('inf')
            best_action = None
            best_next_state = None
            best_outcome = None

            for action in actions:
                next_state, outcome = apply_action(state, action)

                # Determine V(s) = Reward + Gamma * V(next_state)
                if outcome in ('wall', 'obstacle', 'self'):
                    v_s = -100.0
                elif outcome == 'won':
                    v_s = 100.0
                else:
                    reward = 50.0 if outcome == 'ate' else -0.1
                    next_features = extract_features(next_state)
                    v_s = reward + GAMMA * \
                        estimate_value(weights, next_features)

                if v_s > best_val:
                    best_val = v_s
                    best_action = action
                    best_next_state = next_state
                    best_outcome = outcome

            v_target = best_val

        # LMS Weight Update: w_i = w_i + learning_rate * (V_target - V_hat(current_state)) * feature_i
        current_features = extract_features(state)
        v_hat = estimate_value(weights, current_features)
        error = v_target - v_hat

        for i in range(NUM_FEATURES):
            weights[i] += LEARNING_RATE * error * current_features[i]

        # termination case
        if best_outcome in ('wall', 'obstacle', 'self', 'won'):
            break

        state = best_next_state
        steps += 1

    return state['score']


def train():
    weights = [random.uniform(-0.1, 0.1) for _ in range(NUM_FEATURES)]

    print("Starting Training...")
    for episode in range(1, EPISODES + 1):
        score = run_episode(weights)

        if episode % 500 == 0:
            print(
                f"Episode {episode}/{EPISODES} completed. Last Score: {score}"
            )

    # Save raw weights
    with open('snake_weights.pkl', 'wb') as f:
        pickle.dump(weights, f)

    print("\n-------- Final learned weights (Snake) --------")
    labels = [
        "bias",
        "food_above",
        "food_below",
        "food_left",
        "food_right",
        "danger_up",
        "danger_down",
        "danger_left",
        "danger_right",
        "danger_straight",
        "danger_rel_left",
        "danger_rel_right",
        "manhattan",
        "snake_len",
        "toward_food",
        "open_space_norm",
    ]

    for i, (lbl, w) in enumerate(zip(labels, weights)):
        print(f"  w[{i:2d}]  {lbl:<17} = {w:+.6f}")

    print("\nWeights saved -> snake_weights.pkl")
    return weights


if __name__ == "__main__":
    train()
