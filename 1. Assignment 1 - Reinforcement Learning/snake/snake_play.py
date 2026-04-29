import pickle
import tkinter as tk
from configs import *

from snake_train import (
    init_board, get_valid_actions, apply_action,
    extract_features, estimate_value, BOARD_SIZE, GAMMA
)


class SnakeApp:
    def __init__(self, root, weights):
        self.root = root
        self.weights = weights
        canvas_px = BOARD_SIZE * CELL

        root.title("Snake - Trained Agent")
        root.resizable(False, False)
        root.configure(bg=COLORS['bg'])

        #  Canvas 
        self.canvas = tk.Canvas(
            root,
            width=canvas_px, height=canvas_px,
            bg=COLORS['bg'], highlightthickness=0,
        )
        self.canvas.pack(padx=8, pady=(8, 0))

        #  Status bar 
        bar = tk.Frame(root, bg=COLORS['bg'])
        bar.pack(fill='x', padx=8, pady=4)

        self.score_var = tk.StringVar(value="Score: 0")
        self.step_var = tk.StringVar(value="Steps: 0")

        tk.Label(
            bar, textvariable=self.score_var,
            bg=COLORS['bg'], fg=COLORS['text'],
            font=('Courier', 12, 'bold'),
        ).pack(side='left')

        tk.Label(
            bar, textvariable=self.step_var,
            bg=COLORS['bg'], fg=COLORS['text'],
            font=('Courier', 11),
        ).pack(side='right')

        # -- Buttons 
        btn_frame = tk.Frame(root, bg=COLORS['bg'])
        btn_frame.pack(pady=(0, 8))

        tk.Button(
            btn_frame, text='▶  New Game', command=self.restart,
            fg=COLORS['btn_run_fr'],
            font=('Courier', 11, 'bold'),
            relief='flat', padx=14, pady=5,
        ).pack(side='left', padx=6)

        tk.Button(
            btn_frame, text='⏹  Stop', command=self.stop,
            fg=COLORS['btn_stop_fr'],
            font=('Courier', 11, 'bold'),
            relief='flat', padx=14, pady=5,
        ).pack(side='left', padx=6)

        self.state = None
        self.running = False
        self.steps = 0
        self.restart()

    def restart(self):
        self.stop()
        self.state = init_board()
        self.steps = 0
        self.running = True
        self._draw()
        self.root.after(DELAY, self._step)

    def stop(self):
        self.running = False


    def _step(self):
        if not self.running:
            return

        self.steps += 1
        if self.steps > MAX_AUTO_STEPS:
            self._set_status("Step limit reached")
            self.running = False
            return
        
        # Get actions using ONLY the current direction
        actions = get_valid_actions(self.state['direction'])

        best_action = None
        best_val = float('-inf')
        best_next = None
        best_terminal = None

        # Evaluate actions using the Bellman Equation: r + gamma * V(s')
        for action in actions:
            nxt, outcome = apply_action(self.state, action)
            
            if outcome in ('wall', 'obstacle', 'self'):
                v = -100.0
            elif outcome == 'won':
                v = 100.0
            else:
                reward = 50.0 if outcome == 'ate' else -0.1
                v = reward + GAMMA * estimate_value(self.weights, extract_features(nxt))
                
            if v > best_val:
                best_val = v
                best_action = action
                best_next = nxt
                best_terminal = outcome

        if best_terminal in ('wall', 'obstacle', 'self'):
            self.running = False
            self._draw()
            self._set_status("GAME OVER")
            return

        if best_terminal == 'won':
            self.state = best_next
            self.running = False
            self._draw()
            self._set_status("PERFECT – SNAKE WON!")
            return

        self.state = best_next
        self._draw()
        self.score_var.set(f"Score: {self.state['score']}")
        self.step_var.set(f"Steps: {self.steps}")
        self.root.after(DELAY, self._step)

    def _set_status(self, msg):
        score = self.state['score'] if self.state else 0
        self.score_var.set(f"Score: {score}  –  {msg}")

    def _draw(self):
        self.canvas.delete('all')
        C = CELL

        # Grid background
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                self.canvas.create_rectangle(
                    c*C, r*C, c*C + C, r*C + C,
                    fill=COLORS['grid'],
                    outline=COLORS['grid_line'],
                    width=1,
                )

        if self.state is None:
            return

        # Obstacles
        for (r, c) in self.state['obstacles']:
            self.canvas.create_rectangle(
                c*C + 1, r*C + 1, c*C + C - 1, r*C + C - 1,
                fill=COLORS['obstacle'],
                outline='#1e88e5',
            )

        # Food
        if self.state['food']:
            fr, fc = self.state['food']
            pad = 4
            self.canvas.create_oval(
                fc*C + pad, fr*C + pad,
                fc*C + C - pad, fr*C + C - pad,
                fill=COLORS['food'],
                outline='#ff6b6b',
                width=2,
            )

        # Snake body
        for i, (r, c) in enumerate(self.state['body']):
            color = COLORS['head'] if i == 0 else COLORS['body']
            self.canvas.create_rectangle(
                c*C + 2, r*C + 2, c*C + C - 2, r*C + C - 2,
                fill=color,
                outline='#004d40',
            )
            # Draw eye on head
            if i == 0:
                ex = c*C + C - 7
                ey = r*C + 5
                self.canvas.create_oval(ex, ey, ex + 8, ey + 8,
                                        fill='black', outline='')


def play():
    try:
        with open('snake_weights.pkl', 'rb') as f:
            weights = pickle.load(f)
        print("Loaded weights from snake_weights.pkl")
        print(f"  {[round(w, 5) for w in weights]}\n")
    except FileNotFoundError:
        print("Error: snake_weights.pkl not found.")
        print("Please run python snake_train.py first!")
        return

    root = tk.Tk()
    SnakeApp(root, weights)
    root.mainloop()


if __name__ == '__main__':
    play()
