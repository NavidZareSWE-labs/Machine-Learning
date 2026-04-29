import pickle

from configs import _CYAN, _GREEN, _RED, _RESET, _YELLOW, SYMBOLS
from tictactoe_train import (
    check_winner, is_full,
    estimate_value_ttt
)


def _colorize(text, colour):
    return f"{colour}{text}{_RESET}"


def print_board(board):
    print()
    for row in range(3):
        idx = row * 3
        pieces = [SYMBOLS[board[idx + offset]] for offset in range(3)]
        nums = [str(idx + offset + 1) for offset in range(3)]     # 1-indexed

        def fmt(piece):
            if piece == 'X':
                return _colorize('X', _GREEN)
            if piece == 'O':
                return _colorize('O', _RED)
            return _colorize('.', _YELLOW)

        print(
            f"  {fmt(pieces[0])} | {fmt(pieces[1])} | {fmt(pieces[2])}"
            f"      "
            f"  {_colorize(nums[0], _CYAN)} | {_colorize(nums[1], _CYAN)} | {_colorize(nums[2], _CYAN)}"
        )
        if row < 2:
            print("  --|---|--           --|---|--")
    print()


def agent_move(board, weights):
    empty = [i for i in range(9) if board[i] == 0]

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

    return best_move


def human_move(board):
    occupied = {i + 1 for i in range(9) if board[i] != 0}
    available = sorted({i + 1 for i in range(9)} - occupied)

    while True:
        user_input = input(f"  Your move {available}: ").strip()
        try:
            idx = int(user_input)
            if 1 <= idx <= 9 and (idx - 1) not in [i for i in range(9) if board[i] != 0]:
                return idx - 1
        except ValueError:
            pass
        print(f"  {_colorize('Invalid.', _RED)} Choose from: {available}")


def play_game(weights):
    # -------------------------------------------------------------
    #  Agent  : X  (player = +1)  –  always moves first
    #  Opponent: O (player = −1)  –  plays randomly during training
    # -------------------------------------------------------------
    board = [0] * 9
    curr_player = 1     # X (agent) always starts

    print()
    print(_colorize("  ==============================", _CYAN))
    print(_colorize("   Tic-tac-toe: YOU (O) vs Agent (X)", _CYAN))
    print(_colorize("  ==============================", _CYAN))
    print()
    print("  Cell numbering:")
    print("  1 | 2 | 3")
    print("  --|---|--")
    print("  4 | 5 | 6")
    print("  --|---|--")
    print("  7 | 8 | 9")
    print()
    print("  Agent is  X  (goes first).")
    print("  You  are  O.")

    while True:
        print_board(board)

        if curr_player == 1:
            move = agent_move(board, weights)
            board[move] = 1
            print(f"  Agent (X) plays at cell {_colorize(move + 1, _GREEN)}.")
        else:
            move = human_move(board)
            board[move] = -1

        winner = check_winner(board)
        if winner is not None:
            print_board(board)
            if winner == 1:
                print(_colorize("  *** Agent (X) wins! ***", _RED))
            else:
                print(_colorize("  *** You (O) win! Well done! ***", _GREEN))
            return winner

        if is_full(board):
            print_board(board)
            print(_colorize("  *** It's a draw! ***", _YELLOW))
            return 0

        curr_player = -curr_player


def play():
    try:
        with open('ttt_weights.pkl', 'rb') as f:
            weights = pickle.load(f)
        print("Weights loaded from ttt_weights.pkl.")
    except FileNotFoundError:
        print("Error: ttt_weights.pkl not found.")
        print("Please run  python tictactoe_train.py  first.")
        return

    wins = losses = draws = 0

    while True:
        result = play_game(weights)
        if result == 1:
            losses += 1
        elif result == -1:
            wins += 1
        else:
            draws += 1

        print(f"\n  Session record  ->  "
              f"You: {wins} win(s)  |  "
              f"Agent: {losses} win(s)  |  "
              f"Draws: {draws}")

        again = input("\n  Play again? (y/n): ").strip().lower()
        if again != 'y':
            print("\n  Thanks for playing!")
            break


if __name__ == '__main__':
    play()
