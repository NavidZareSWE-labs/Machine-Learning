import random
import pickle

from configs import ALPHA, EPISODES, NUM_FEATURES, WINNING_LINES


#  Game logic
def check_winner(board):
    """
     Returns  
              +1  if X wins,
              −1  if O wins,
              None  otherwise.
    """
    for i, j, k in WINNING_LINES:
        if board[i] != 0 and board[i] == board[j] == board[k]:
            return board[i]
    return None


def is_full(board):
    return all(cell != 0 for cell in board)



def extract_features(board):
    # Compute and return the 16-dimensional feature vector for a board.
    # All features are expressed from X's (+1) perspective.
    #
    # Index  Feature
    # -----  --------------------------------------------------------
    # 0      Bias term = 1
    # 1–9    Raw cell values: +1 (X), −1 (O), 0 (empty)  [cells 0..8]
    # 10     # lines where X has 2 pieces and O has 0  (near-win for X)
    # 11     # lines where O has 2 pieces and X has 0  (near-win for O)
    # 12     # lines where X has 1 piece  and O has 0  (open lines for X)
    # 13     Centre is X
    # 14     # corners controlled by X  (cells 0, 2, 6, 8)
    # 15     # corners controlled by O

    near_win_for_x = sum(
        1
        for i, j, k in WINNING_LINES
        if board[i] + board[j] + board[k] == 2
        and all(board[idx] >= 0 for idx in (i, j, k))  # no O in the line
    )

    near_win_for_o = sum(
        1
        for i, j, k in WINNING_LINES
        if board[i] + board[j] + board[k] == -2
        and all(board[idx] <= 0 for idx in (i, j, k))  # no X in the line
    )

    open_line_for_x = sum(
        1
        for i, j, k in WINNING_LINES
        if board[i] + board[j] + board[k] == 1
        and all(board[idx] >= 0 for idx in (i, j, k))  # no O in the line
    )

    center_is_x = 1 if board[4] == 1 else 0

    corner_indices = (0, 2, 6, 8)
    x_corner_count = sum(1 for i in corner_indices if board[i] == 1)
    o_corner_count = sum(1 for i in corner_indices if board[i] == -1)

    bias = 1.0
    raw_cells = list(board)

    return [
        bias,
        *raw_cells,
        near_win_for_x,
        near_win_for_o,
        open_line_for_x,
        center_is_x,
        x_corner_count,
        o_corner_count,
    ]


def estimate_value_ttt(weights, board):
    features = extract_features(board)
    return sum(w * x for w, x in zip(weights, features))


def update_weights_ttt(weights, alpha, v_train, v_hat, board):
    features = extract_features(board)
    error = v_train - v_hat
    for i in range(len(weights)):
        weights[i] += alpha * error * features[i]


def run_episode_ttt(weights):
    board = [0] * 9
    curr_player = 1           # X moves first

    # State of agent's PREVIOUS decision (for ***DEFERRED*** update)
    prev_board = None
    prev_val_hat = None

    while True:
        empty = [i for i in range(9) if board[i] == 0]

        # ------ Terminal: no moves left -> draw ------
        if not empty:
            if prev_board is not None:
                update_weights_ttt(weights, ALPHA,
                                   0.0, prev_val_hat, prev_board)
            return 0

        if curr_player == 1:
            prev_board_snap = board[:]
            val_curr_hat = estimate_value_ttt(weights, prev_board_snap)

            # Deferred update
            if prev_board is not None:
                update_weights_ttt(weights, ALPHA,
                                   val_curr_hat, prev_val_hat, prev_board)

            # ------ Greedy action selection ------
            best_move = None
            best_val = float('-inf')
            for move in empty:
                board_snap = board[:]
                board_snap[move] = 1
                winner = check_winner(board_snap)
                if winner == 1:
                    val = 100.0
                elif winner == -1:
                    val = -100.0
                elif is_full(board_snap):
                    val = 0.0
                else:
                    val = estimate_value_ttt(weights, board_snap)
                if val > best_val:
                    best_val = val
                    best_move = move

            board[best_move] = 1

            # Save for deferred update on next agent turn
            prev_board = prev_board_snap
            prev_val_hat = val_curr_hat

            # ---- Check terminal after X's move ----
            winner = check_winner(board)
            if winner == 1:
                update_weights_ttt(weights, ALPHA,
                                   100.0, val_curr_hat, prev_board_snap)
                return 1
            if is_full(board):
                update_weights_ttt(weights, ALPHA,
                                   0.0, val_curr_hat, prev_board_snap)
                return 0

        else:
            move = random.choice(empty)
            board[move] = -1

            winner = check_winner(board)
            if winner == -1:
                if prev_board is not None:
                    update_weights_ttt(weights, ALPHA,
                                       -100.0, prev_val_hat, prev_board)
                return -1
            if is_full(board):
                if prev_board is not None:
                    update_weights_ttt(weights, ALPHA,
                                       0.0, prev_val_hat, prev_board)
                return 0

        curr_player = -curr_player


def train_ttt():
    weights = [0.0] * NUM_FEATURES
    print(f"Training Tic-tac-toe agent for {EPISODES} episodes ...\n")

    wins = draws = losses = 0
    for ep in range(1, EPISODES + 1):
        result = run_episode_ttt(weights)
        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1

        if ep % 10_000 == 0:
            total = wins + draws + losses
            print(f"  Episode {ep:>6}  |  "
                  f"W: {wins/total*100:5.1f}%  "
                  f"D: {draws/total*100:5.1f}%  "
                  f"L: {losses/total*100:5.1f}%")
            wins = draws = losses = 0   # reset window

    print("\n-------- Final learned weights --------")
    labels = [
        "bias",
        "cell_0", "cell_1", "cell_2",
        "cell_3", "cell_4", "cell_5",
        "cell_6", "cell_7", "cell_8",
        "near_win_for_x", "near_win_for_o",
        "open_line_for_x", "center_is_x",
        "x_corner_count", "o_corner_count",
    ]
    for i, (lbl, w) in enumerate(zip(labels, weights)):
        print(f"  w[{i:2d}]  {lbl:<14} = {w:+.6f}")

    with open('ttt_weights.pkl', 'wb') as f:
        pickle.dump(weights, f)
    print("\nWeights saved -> ttt_weights.pkl")
    return weights


if __name__ == '__main__':
    train_ttt()
