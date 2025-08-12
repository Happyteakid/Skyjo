#!/usr/bin/env python3
# skyjo_gui_modular.py — Main game/GUI + Player classes (Human/AI), imports skyjo_ai
# Extras:
#  - "Wait for Next" mode (default ON): no auto-advance; click Next Player to start the next turn.
#  - Discard order when a SWAP removes a triple column: first the replaced card, then the three equal cards
#    (so the equal value ends on top). For a FLIP removal, only the three equal cards are added.

import random
import tkinter as tk
from tkinter import messagebox, simpledialog
import time, json, traceback

import skyjo_ai  # AI strategy module


ROWS = 3
START_COLS = 4
MAX_PLAYERS = 8

CARD_W = 70
CARD_H = 100
GAP_X = 14
GAP_Y = 14
MARGIN_X = 30
MARGIN_Y = 80

BOARD_BG = "#f5f5f5"
FACEDOWN_COLOR = "#d3d3d3"

LOG_FILE = "skyjo_debug.log"
DUMP_FILE = "skyjo_dump.txt"

def log_ts(): return time.strftime("%Y-%m-%d %H:%M:%S")
def write_file_log(msg):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{log_ts()}] {msg}\n")
    except Exception:
        pass

def build_deck():
    return ([-2]*5) + ([-1]*10) + ([0]*15) + sum(([v]*10 for v in range(1,13)), [])

def value_color(v: int) -> str:
    if v <= -2: return "#6a5acd"
    if v == -1: return "#8a2be2"
    if v == 0:  return "#1e90ff"
    if 1 <= v <= 3: return ["#2e8b57", "#3cb371", "#66cdaa"][v-1]
    if 4 <= v <= 7: return {4:"#9acd32", 5:"#ffd700", 6:"#f0ad4e", 7:"#ffa500"}[v]
    return {8:"#ff7f50", 9:"#ff6347", 10:"#ff4500", 11:"#e34234", 12:"#b22222"}[v]

class PlayerBase:
    def __init__(self, name):
        self.name = name
        self.is_boss = False
        self.cols = []
        self.up = []

    def setup(self, deck):
        self.cols = [[deck.pop() for _ in range(ROWS)] for _ in range(START_COLS)]
        self.up = [[False]*ROWS for _ in range(START_COLS)]
        spots = [(j,r) for j in range(START_COLS) for r in range(ROWS)]
        for j,r in random.sample(spots, 2): self.up[j][r] = True

    def num_cols(self): return len(self.cols)
    def all_face_up(self): return all(self.up[j][r] for j in range(self.num_cols()) for r in range(ROWS))
    def facedown_positions(self): return [(j,r) for j in range(self.num_cols()) for r in range(ROWS) if not self.up[j][r]]
    def score(self): return sum(self.cols[j][r] for j in range(self.num_cols()) for r in range(ROWS))

    def swap_in(self, j, r, new_card):
        """Perform swap; return (out_card, removed_values or None).
        If the swap completes a triple column, it removes that column and returns the 3 removed values."""
        out = self.cols[j][r]
        self.cols[j][r] = new_card
        self.up[j][r] = True
        removed_vals = None
        if all(self.up[j]) and (self.cols[j][0] == self.cols[j][1] == self.cols[j][2]):
            removed_vals = [self.cols[j][0], self.cols[j][1], self.cols[j][2]]
            del self.cols[j]; del self.up[j]
        return out, removed_vals

    def flip_at(self, j, r):
        """Flip and remove the column if it becomes triple. Return removed_vals (or None)."""
        self.up[j][r] = True
        removed_vals = None
        if all(self.up[j]) and (self.cols[j][0] == self.cols[j][1] == self.cols[j][2]):
            removed_vals = [self.cols[j][0], self.cols[j][1], self.cols[j][2]]
            del self.cols[j]; del self.up[j]
        return removed_vals

    def reveal(self):
        for j in range(self.num_cols()):
            for r in range(ROWS): self.up[j][r] = True

    # Public view for AI
    def public_board(self):
        return skyjo_ai.PublicBoard(cols=[c[:] for c in self.cols], up=[u[:] for u in self.up])

class HumanPlayer(PlayerBase):
    pass

class AIPlayer(PlayerBase):
    def __init__(self, name):
        super().__init__(name)
        self.is_boss = True

class SkyjoGUI:
    def __init__(self, root):
        
        self.root = root
        self.root.title("Skyjo — Modular (GUI + AI in separate file)")
        self.root.configure(bg=BOARD_BG)

        self.players = []
        self.turn = 0
        self.stock = []
        self.discard = []
        self.pending_card = None
        self.pending_src = None
        self.mode = "idle"
        self.ending_idx = None
        self.final_left = 0
        self.ai_busy = False
        self.wait_for_next = True
        self.last_action = None
        self.last_action_until = 0
        self.after_ids = []

        # UI
        top = tk.Frame(root, bg=BOARD_BG); top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)
        self.turn_label = tk.Label(top, text="Turn:", font=("Arial", 12, "bold"), bg=BOARD_BG); self.turn_label.pack(side=tk.LEFT)
        self.info_label = tk.Label(top, text="", font=("Arial", 11), bg=BOARD_BG); self.info_label.pack(side=tk.LEFT, padx=20)

        btns = tk.Frame(root, bg=BOARD_BG); btns.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)
        self.btn_draw_stock = tk.Button(btns, text="Draw from Stock", width=16, command=self.on_draw_stock); self.btn_draw_stock.pack(side=tk.LEFT, padx=4)
        self.btn_take_discard = tk.Button(btns, text="Take Discard", width=16, command=self.on_take_discard); self.btn_take_discard.pack(side=tk.LEFT, padx=4)
        self.btn_swap_mode = tk.Button(btns, text="Swap Mode", width=12, command=self.set_mode_swap); self.btn_swap_mode.pack(side=tk.LEFT, padx=4)
        self.btn_flip_mode = tk.Button(btns, text="Discard & Flip", width=12, command=self.set_mode_flip); self.btn_flip_mode.pack(side=tk.LEFT, padx=4)
        self.btn_force = tk.Button(btns, text="Force AI Move", width=14, command=self.ai_force_move, state="disabled"); self.btn_force.pack(side=tk.LEFT, padx=4)
        self.btn_next = tk.Button(btns, text="Next Player", width=12, command=self.on_next_player); self.btn_next.pack(side=tk.LEFT, padx=4)
        self.btn_dump = tk.Button(btns, text="Debug Dump", width=12, command=self.debug_dump); self.btn_dump.pack(side=tk.LEFT, padx=4)
        self.btn_new = tk.Button(btns, text="New Round", width=12, command=self.new_round); self.btn_new.pack(side=tk.RIGHT, padx=4)
        self.btn_weights = tk.Button(btns, text="AI Weights...", width=12, command=self.open_weights_panel)
        self.btn_weights.pack(side=tk.RIGHT, padx=4)


        opts = tk.Frame(root, bg=BOARD_BG); opts.pack(side=tk.TOP, fill=tk.X, padx=10)
        self.wait_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opts, text="Wait for Next", variable=self.wait_var, bg=BOARD_BG, command=self.on_toggle_wait).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(root, width=1220, height=780, bg=BOARD_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        log_frame = tk.Frame(root, bg=BOARD_BG); log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=10, pady=(4,10))
        tk.Label(log_frame, text="Log (AI & debug):", bg=BOARD_BG, font=("Arial", 11, "bold")).pack(anchor="w")
        self.log_text = tk.Text(log_frame, height=8, state="disabled", wrap="word"); self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(log_frame, command=self.log_text.yview); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=sb.set)

        self.card_hitboxes = []
        self.stock_box = None; self.discard_box = None

        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"[{log_ts()}] Skyjo modular started\n")
        try:
            self.add_log(f"AI loaded: {getattr(skyjo_ai, 'ai_version', lambda: 'unknown')()}")
        except Exception:
            self.add_log(f"AI module path: {getattr(skyjo_ai, '__file__', '?')}")

        self.new_round()
    
    # ---------- utilities ----------
    def schedule(self, delay_ms, func):
        aid = self.root.after(delay_ms, func); self.after_ids.append(aid); return aid
    def cancel_all_after(self):
        for aid in self.after_ids:
            try: self.root.after_cancel(aid)
            except: pass
        self.after_ids.clear()
    def now(self): return time.time()
    def add_log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        write_file_log(msg)
    def on_toggle_wait(self):
        self.wait_for_next = bool(self.wait_var.get())
        self.add_log(f"Wait for Next set to {self.wait_for_next}")
    def debug_dump(self):
        try:
            snapshot = {
                "turn": self.turn,
                "players": [{
                    "name": p.name, "is_boss": p.is_boss, "cols": p.cols, "up": p.up,
                    "num_cols": p.num_cols(), "facedown": len(p.facedown_positions())
                } for p in self.players],
                "stock_len": len(self.stock),
                "discard_top": self.discard[-1] if self.discard else None,
                "discard_len": len(self.discard),
                "pending_card": self.pending_card, "pending_src": self.pending_src,
                "ending_idx": self.ending_idx, "final_left": self.final_left
            }
            with open(DUMP_FILE, "w", encoding="utf-8") as f: f.write(json.dumps(snapshot, indent=2))
            self.add_log(f"Debug dump saved to {DUMP_FILE}")
        except Exception as e:
            self.add_log(f"Debug dump failed: {e}\n{traceback.format_exc()}")

    # ---------- setup ----------
    def ask_players(self):
        n = simpledialog.askinteger("Players", f"Total players (1–{MAX_PLAYERS}):",
                                    minvalue=1, maxvalue=MAX_PLAYERS, parent=self.root) or 2
        ai = simpledialog.askinteger("Ultimate Boss AI", f"How many boss AIs? (0–{n}):",
                                     minvalue=0, maxvalue=n, parent=self.root)
        if ai is None:
            ai = max(0, n-1)  # default: everyone except you is AI
        if ai >= n:
            if n == 1:
                messagebox.showinfo("Adjusting seats", "You entered 1 player and 1 AI.\nI'll set it to 2 players with 1 AI so you can play.")
                n, ai = 2, 1
            else:
                messagebox.showinfo("Adjusting seats", f"AI count {ai} equals total players {n}. I'll reduce AIs to {n-1} so P1 stays human.")
                ai = n - 1
        return n, ai

    def new_round(self):
        self.cancel_all_after()
        n, ai = self.ask_players()

        # Auto-load per-player-count weights if available (e.g., skyjo_ai_weights_3p.json)
        try:
            import os
            wfile = f"skyjo_ai_weights_{n}p.json"
            if os.path.exists(wfile):
                skyjo_ai.load_weights(wfile)
                self.add_log(f"Loaded AI weights for {n} players: {wfile}")
            elif os.path.exists("skyjo_ai_weights.json"):
                skyjo_ai.load_weights("skyjo_ai_weights.json")
                self.add_log("Loaded global AI weights (no per-count file found).")
            else:
                self.add_log("Using default AI weights (no saved file).")
        except Exception as e:
            self.add_log(f"Failed to load weights: {e}")

        # -- set up one (and only one) round --
        self.stock = build_deck(); random.shuffle(self.stock)
        self.discard = [self.stock.pop()]
        self.players = []
        for i in range(n):
            if i < n - ai:
                p = HumanPlayer(f"P{i+1} [You]")
            else:
                p = AIPlayer(f"P{i+1} [AI]")
            p.setup(self.stock)
            self.players.append(p)

        self.turn = 0
        self.pending_card = None; self.pending_src = None; self.mode = "idle"
        self.ending_idx = None; self.final_left = 0
        self.ai_busy = False; self.last_action = None
        self.add_log(f"New round: players={n}, AIs={ai}, stock={len(self.stock)}, discard_top={self.discard[-1]}")
        self.redraw()
        if not self.wait_for_next:
            self.maybe_start_ai_turn()
        else:
            self.set_info("Your turn. Make a move, then click Next Player.")

    # ---------- flow ----------
    def current_player(self): return self.players[self.turn]
    def prev_player(self): return self.players[(self.turn-1) % len(self.players)]
    def next_turn(self):
        n = len(self.players)
        if self.ending_idx is not None and self.turn != self.ending_idx:
            self.final_left -= 1
            if self.final_left <= 0: self.finish_round(); return
        self.turn = (self.turn + 1) % n
        self.pending_card = None; self.pending_src = None; self.mode = "idle"
        self.ai_busy = False
        self.redraw()
        if self.wait_for_next:
            self.set_info(f"{self.current_player().name}'s turn. Click Next Player to begin.")
        else:
            self.maybe_start_ai_turn()

    def finish_round(self):
        for pl in self.players: pl.reveal()
        scores = [pl.score() for pl in self.players]
        finisher_idx = self.ending_idx; lowest = min(scores) if scores else 0
        penalty_msg = ""
        if finisher_idx is not None and scores[finisher_idx] != lowest and scores[finisher_idx] > 0:
            scores[finisher_idx] *= 2; penalty_msg = f"\nPenalty: {self.players[finisher_idx].name}'s score doubled."
        winner_idx = min(range(len(scores)), key=lambda k: scores[k]) if scores else 0
        lines = [f"{pl.name}: {sc}" for pl, sc in zip(self.players, scores)]
        self.add_log("Round over. " + ", ".join(lines) + penalty_msg.replace("\n"," "))
        messagebox.showinfo("Round Over","Final scores:\n" + "\n".join(lines) + penalty_msg + f"\n\nWinner: {self.players[winner_idx].name}")
        self.redraw()

    # ---------- human actions ----------
    def on_draw_stock(self):
        if self.pending_card is not None or self.ai_busy: return
        self._draw_from_stock()
    def on_take_discard(self):
        if self.pending_card is not None or self.ai_busy or not self.discard: return
        self._take_discard()
    def set_mode_swap(self):
        if self.pending_card is None or self.ai_busy: self.set_info("Draw first."); return
        self.mode = "swap"; self.set_info("Swap mode: click a slot to place the drawn card."); self.redraw()
    def set_mode_flip(self):
        if self.pending_card is None or self.ai_busy: self.set_info("Draw first."); return
        if self.pending_src != "stock": self.set_info("You can only Discard & Flip when you drew from stock."); return
        self.discard.append(self.pending_card); self.add_log(f"{self.current_player().name}: discarded {self.pending_card} to flip.")
        self.pending_card = None; self.mode = "flip"; self.set_info("Discarded. Click a facedown slot to flip it."); self.redraw()

    def _draw_from_stock(self):
        self._reshuffle_if_needed()
        if not self.stock: self.set_info("Stock is empty."); return
        self.pending_card = self.stock.pop(); self.pending_src = "stock"; self.mode = "swap"
        self.set_info("Drew from stock. Choose Swap OR Discard & Flip.")
        self.add_log(f"{self.current_player().name}: drew from stock -> {self.pending_card}")
        self.last_action = {"pi": self.turn, "type":"draw", "j":None, "r":None, "in":self.pending_card, "out":None, "removed":False}
        self.last_action_until = self.now() + 1.1
        self.redraw()

    def _take_discard(self):
        if not self.discard: self.set_info("Discard is empty."); return
        self.pending_card = self.discard.pop(); self.pending_src = "discard"; self.mode = "swap"
        self.set_info("Took from discard. Click a slot to swap.")
        self.add_log(f"{self.current_player().name}: took from discard -> {self.pending_card}")
        self.last_action = {"pi": self.turn, "type":"discard", "j":None, "r":None, "in":self.pending_card, "out":None, "removed":False}
        self.last_action_until = self.now() + 1.1
        self.redraw()

    def perform_swap(self, j, r):
        p = self.current_player()
        if j is None or r is None or self.pending_card is None: return
        if not (0 <= j < p.num_cols()) or not (0 <= r < ROWS): return
        in_card = self.pending_card
        out, removed_vals = p.swap_in(j, r, in_card)
        # Discard order: replaced OUT first, then the removed triple (so triple value ends on top)
        self.discard.append(out)
        if removed_vals is not None:
            for v in removed_vals: self.discard.append(v)
        self.pending_card = None
        msg = f"{p.name}: swapped {in_card} into c{j+1},r{r+1}, replaced {out}" + (" — column removed" if removed_vals else "")
        self.add_log(msg); self.set_info(msg)
        self.last_action = {"pi": self.turn, "type":"swap", "j":j, "r":r, "in":in_card, "out":out, "removed":bool(removed_vals)}
        self.last_action_until = self.now() + 1.0
        self._after_action_finalize(p)

    def perform_discard_and_flip(self, j, r):
        p = self.current_player()
        if j is None or r is None or not (0 <= j < p.num_cols()) or not (0 <= r < ROWS): return
        pending = self.pending_card
        if pending is not None: self.discard.append(pending); self.pending_card = None
        pre_val = p.cols[j][r]
        removed_vals = p.flip_at(j, r)
        # Flip removal: only the triple gets added (no replaced card here)
        if removed_vals is not None:
            for v in removed_vals: self.discard.append(v)
        msg = f"{p.name}: flipped c{j+1},r{r+1} -> {pre_val}" + (" — column removed" if removed_vals else "")
        self.add_log(msg); self.set_info(msg)
        self.last_action = {"pi": self.turn, "type":"flip", "j":j, "r":r, "in":None, "out":pre_val, "removed":bool(removed_vals)}
        self.last_action_until = self.now() + 1.0
        self._after_action_finalize(p)

    def on_canvas_click(self, event):
        if self.ai_busy: return
        if not self.card_hitboxes: return
        px, py = event.x, event.y
        hit = None
        for (x1,y1,x2,y2,j,r) in self.card_hitboxes:
            if x1 <= px <= x2 and y1 <= py <= y2:
                hit = (j,r); break
        if hit is None: return
        j,r = hit
        if self.mode == "swap" and self.pending_card is not None:
            self.perform_swap(j, r); return
        if self.mode == "flip" and self.pending_card is None:
            p = self.current_player()
            if p.up[j][r]: self.set_info("That slot is already face-up. Pick a facedown slot."); return
            self.perform_discard_and_flip(j, r); return
        self.set_info("Choose an action: draw, swap, or discard & flip."); self.redraw()

    def _after_action_finalize(self, p):
        # Make sure AI can hand control back so Next Player is clickable
        self.ai_busy = False
        if p.all_face_up() and self.ending_idx is None:
            self.ending_idx = self.turn; self.final_left = len(self.players) - 1
            self.set_info(f"{p.name} flipped their last card! Others get one final turn.")
            self.add_log(f"{p.name} triggered endgame. final_left={self.final_left}")
        self.redraw()
        if self.wait_for_next:
            self.set_info("Move done. Click Next Player to continue.")
        else:
            self.schedule(400, self.next_turn)

    def _reshuffle_if_needed(self):
        if not self.stock and len(self.discard) > 1:
            top = self.discard.pop(); self.stock = self.discard[:]; random.shuffle(self.stock); self.discard = [top]
            self.add_log("Reshuffled discard into stock.")

    def on_next_player(self):
        self.cancel_all_after()
        self.add_log("Next Player clicked — advancing turn.")
        self.next_turn()
        # If it's an AI's turn and Wait for Next is ON, start its thinking now
        if isinstance(self.current_player(), AIPlayer) and self.wait_for_next:
            self.maybe_start_ai_turn()

    # ---------- AI ----------
    def maybe_start_ai_turn(self):
        if not self.players: return
        p = self.current_player()
        if isinstance(p, AIPlayer):
            self.ai_busy = True; self.update_buttons_state()
            self.set_info(f"{p.name} is thinking…"); self.add_log(f"{p.name}: thinking…")
            # First decision: take discard or draw
            state = self._public_state()
            action, tgt = skyjo_ai.choose_action(state)
            if action == 'take_discard' and self.discard:
                self._take_discard()
                j,r = tgt
                self.schedule(250, lambda j=j, r=r: self.perform_swap(j, r))
            else:
                self._draw_from_stock()
                # After drawing, decide swap vs flip
                self.schedule(300, self._ai_after_draw_step)

    def _ai_after_draw_step(self):
        if self.pending_card is None:
            self.ai_busy = False; return
        state = self._public_state()
        action, j, r = skyjo_ai.choose_after_draw(state, self.pending_card)
        if action == 'swap':
            self.schedule(150, lambda j=j, r=r: self.perform_swap(j, r))
        else:
            self.schedule(150, lambda j=j, r=r: self.perform_discard_and_flip(j, r))

    def ai_force_move(self):
        if not self.players or not isinstance(self.current_player(), AIPlayer) or not self.ai_busy:
            return
        self.cancel_all_after()
        self.add_log("(FORCED) AI move now.")
        if self.pending_card is None:
            self.maybe_start_ai_turn()
        else:
            self._ai_after_draw_step()

    def _public_state(self) -> skyjo_ai.PublicState:
        players_pb = [p.public_board() for p in self.players]
        return skyjo_ai.PublicState(
        me_index=self.turn,
        prev_index=(self.turn-1) % len(self.players),
        players=players_pb,
        discard_top=(self.discard[-1] if self.discard else None),
        discard_len=len(self.discard),
        stock_len=len(self.stock),
        discard_list=self.discard[:]            # NEW
    )


    # ---------- rendering ----------
    def set_info(self, msg): self.info_label.config(text=msg)

    def redraw(self):
        self.canvas.delete("all"); self.card_hitboxes.clear()
        turn_text = f"Turn: {self.current_player().name}"
        if self.ending_idx is not None:
            turn_text += " (finisher)" if self.turn==self.ending_idx else f" (final turns left: {self.final_left})"
        self.turn_label.config(text=turn_text)

        # piles
        x_piles=30; y_piles=20
        self.draw_card_back(x_piles, y_piles, label=f"Stock ({len(self.stock)})"); self.stock_box=(x_piles,y_piles,x_piles+CARD_W,y_piles+CARD_H)
        x_disc = x_piles + CARD_W + 40
        if self.discard: self.draw_card_face(x_disc,y_piles,self.discard[-1],True,"black","Discard")
        else: self.draw_empty_slot(x_disc,y_piles,"Discard")
        x_hold = x_disc + CARD_W + 40
        if self.pending_card is not None: self.draw_card_face(x_hold,y_piles,self.pending_card,True,"black","In Hand")
        else: self.draw_empty_slot(x_hold,y_piles,"In Hand")

        # main board
        p = self.current_player(); cols = p.num_cols()
        total_w = cols*CARD_W + (cols-1)*GAP_X
        start_x = max(MARGIN_X, (self.canvas.winfo_width()-total_w)//2)
        start_y = MARGIN_Y + 40
        for j in range(cols):
            for r in range(ROWS):
                x = start_x + j*(CARD_W+GAP_X); y = start_y + r*(CARD_H+GAP_Y)
                val = p.cols[j][r]; up = p.up[j][r]
                if up:
                    self.draw_card_face(x,y,val,True)
                else:
                    self.draw_card_face(x,y,val,False)
                self.card_hitboxes.append((x,y,x+CARD_W,y+CARD_H,j,r))

        # mini boards at bottom (non-overlapping, based on max columns)
        self.draw_all_minis()
        self.update_buttons_state()

    def draw_all_minis(self):
        n = len(self.players)
        if n==0: return
        per_row = min(4, n)
        rows = (n + per_row - 1)//per_row
        mini_w = 34; mini_h = 48; gapx=10; gapy=12
        max_cols = max(p.num_cols() for p in self.players) if self.players else 4
        block_w = max_cols*(mini_w+gapx) - gapx + 24
        base_y = self.canvas.winfo_height() - (rows*(mini_h*3 + gapy*2) + 80)
        if base_y < 430: base_y = 430
        base_x = 24
        idx=0
        for row in range(rows):
            for col in range(per_row):
                if idx>=n: break
                px = base_x + col*(block_w + 30)
                py = base_y + row*(mini_h*3 + gapy*2 + 36)
                self.draw_mini_board(idx, px, py, mini_w, mini_h, gapx, gapy)
                idx+=1

    def draw_mini_board(self, pi, x0, y0, w, h, gapx, gapy):
        p = self.players[pi]
        self.canvas.create_text(x0, y0-6, text=p.name, anchor="nw", font=("Arial", 10, "bold"), fill="#0a5" if pi==self.turn else "#000")
        for j in range(p.num_cols()):
            for r in range(ROWS):
                x = x0 + j*(w+gapx); y = y0 + r*(h+gapy)
                if p.up[j][r]:
                    self.canvas.create_rectangle(x,y,x+w,y+h, fill=value_color(p.cols[j][r]), outline="black", width=2)
                    self.canvas.create_text(x+w/2,y+h/2, text=str(p.cols[j][r]), font=("Arial", 9), fill=("white" if p.cols[j][r]>=8 else "black"))
                else:
                    self.canvas.create_rectangle(x,y,x+w,y+h, fill=FACEDOWN_COLOR, outline="black", width=2)
                    self.canvas.create_text(x+w/2,y+h/2, text="?", font=("Arial", 9))

    def draw_card_face(self, x, y, value, face_up=True, outline="black", title=None):
        if face_up:
            fill = value_color(value); text=str(value); txt_fill = "white" if value>=8 else "black"
        else:
            fill = FACEDOWN_COLOR; text="?" ; txt_fill = "black"
        self.canvas.create_rectangle(x, y, x+CARD_W, y+CARD_H, fill=fill, outline=outline, width=2)
        self.canvas.create_text(x+CARD_W/2, y+CARD_H/2, text=text, font=("Arial", 20, "bold"), fill=txt_fill)
        if title: self.canvas.create_text(x + CARD_W/2, y - 12, text=title, font=("Arial", 10))

    def draw_card_back(self, x, y, label=None):
        self.canvas.create_rectangle(x, y, x+CARD_W, y+CARD_H, fill="#888", outline="black", width=2)
        self.canvas.create_text(x+CARD_W/2, y+CARD_H/2, text="▒", font=("Courier", 18))
        if label: self.canvas.create_text(x + CARD_W/2, y - 12, text=label, font=("Arial", 10))

    def draw_empty_slot(self, x, y, title=None):
        self.canvas.create_rectangle(x, y, x+CARD_W, y+CARD_H, fill="#eee", outline="black", dash=(3,3), width=2)
        if title: self.canvas.create_text(x + CARD_W/2, y - 12, text=title, font=("Arial", 10))

    def update_buttons_state(self):
        
        if self.ai_busy:
            self.btn_draw_stock.config(state="disabled"); self.btn_take_discard.config(state="disabled")
            self.btn_swap_mode.config(state="disabled"); self.btn_flip_mode.config(state="disabled")
            self.btn_force.config(state="normal"); self.btn_next.config(state="disabled")
        else:
            self.btn_draw_stock.config(state=("disabled" if self.pending_card is not None else "normal"))
            self.btn_take_discard.config(state=("disabled" if (self.pending_card is not None or not self.discard) else "normal"))
            can_swap = (self.pending_card is not None)
            can_flip = (self.pending_card is not None and self.pending_src=="stock" and len(self.current_player().facedown_positions())>0)
            self.btn_swap_mode.config(state=("normal" if can_swap else "disabled"), relief=("sunken" if self.mode=="swap" else "raised"))
            self.btn_flip_mode.config(state=("normal" if can_flip else "disabled"), relief=("sunken" if self.mode=="flip" else "raised"))
            self.btn_force.config(state="disabled"); self.btn_next.config(state="normal")

        # ---------- AI Weights Panel ----------
    def open_weights_panel(self):
        # prevent multiple windows
        if hasattr(self, "win_weights") and self.win_weights and tk.Toplevel.winfo_exists(self.win_weights):
            self.win_weights.lift()
            return

        self.win_weights = tk.Toplevel(self.root)
        self.win_weights.title("AI Weights")
        self.win_weights.configure(bg=BOARD_BG)
        self.win_weights.resizable(False, False)

        # Current weights = defaults merged with saved file (if any)
        defaults = skyjo_ai.get_default_weights()
        current = defaults.copy()
        try:
            import json, os
            if os.path.exists("skyjo_ai_weights.json"):
                with open("skyjo_ai_weights.json","r",encoding="utf-8") as f:
                    saved = json.load(f)
                for k,v in saved.items():
                    if k in current:
                        current[k] = float(v)
        except Exception:
            pass

        # Keep tk variables so we can read/apply
        self._wvars = {}
        rows = []

        # Friendly labels/explanations (also shown in the panel)
        friendly = {
            "pen_swap_up_base":        "Penalty: swap higher card over lower (no removal)",
            "pen_swap_up_nonpos_extra":"Extra penalty: replacing -2/-1/0 with higher (no removal)",
            "pen_swap_up_gap_mult":    "Penalty scales with (new-old) gap",
            "deny_bonus_cap":          "Max deny bonus (taking card prev wants)",
            "deny_bonus_factor":       "Deny bonus factor",
            "out_risk_penalty_factor": "Penalty if our discard helps prev’s pair",
            "swap_beats_flip_margin":  "Tie bias: swap vs flip (>= means swap)",
            "discard_over_stock_bias": "How much better discard must be vs stock",
            "endgame_finish_margin":   "Finish only if <= best opponent + margin",
        }

        # Reasonable UI ranges per weight
        ranges = {
            "pen_swap_up_base":        (-5.0, 20.0, 0.1),
            "pen_swap_up_nonpos_extra":( 0.0, 30.0, 0.1),
            "pen_swap_up_gap_mult":    ( 0.0,  2.0, 0.01),
            "deny_bonus_cap":          ( 0.0, 20.0, 0.1),
            "deny_bonus_factor":       ( 0.0,  2.0, 0.01),
            "out_risk_penalty_factor": ( 0.0,  2.0, 0.01),
            "swap_beats_flip_margin":  (-2.0,  2.0, 0.01),
            "discard_over_stock_bias": (-1.0,  2.0, 0.01),
            "endgame_finish_margin":   (-5.0,  5.0, 0.1),
        }

        container = tk.Frame(self.win_weights, bg=BOARD_BG)
        container.pack(padx=12, pady=12)

        row = 0
        for key in sorted(defaults.keys()):
            lab = tk.Label(container, text=friendly.get(key, key), bg=BOARD_BG, anchor="w")
            lab.grid(row=row, column=0, sticky="w", padx=(0,8), pady=4)
            v = tk.DoubleVar(value=current[key])
            self._wvars[key] = v

            lo, hi, step = ranges.get(key, (-20.0, 20.0, 0.1))
            # Spinbox supports float increments; it's precise and compact
            sp = tk.Spinbox(container, from_=lo, to=hi, increment=step, textvariable=v, width=10)
            sp.grid(row=row, column=1, sticky="e", padx=(0,0), pady=4)

            # show the key name (small)
            tk.Label(container, text=key, bg=BOARD_BG, fg="#666", font=("Arial", 8)).grid(row=row, column=2, sticky="w", padx=(8,0))
            row += 1

        btns = tk.Frame(self.win_weights, bg=BOARD_BG)
        btns.pack(padx=12, pady=(6,12), fill="x")

        tk.Button(btns, text="Apply & Save", width=14, command=self._weights_apply).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Reset Defaults", width=14, command=self._weights_reset_defaults).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Close", width=10, command=self._weights_close).pack(side=tk.RIGHT, padx=4)

        # tiny hint
        hint = ("Changes apply immediately for future AI decisions.\n"
                "Tip: leave this window open while testing.")
        tk.Label(self.win_weights, text=hint, bg=BOARD_BG, fg="#444", font=("Arial", 9)).pack(padx=12, pady=(0,10))

        self.win_weights.protocol("WM_DELETE_WINDOW", self._weights_close)

    def _weights_apply(self):
        weights = {k: float(v.get()) for k,v in self._wvars.items()}
        try:
            n = len(self.players) if self.players else 4
            wfile = f"skyjo_ai_weights_{n}p.json"
            skyjo_ai.save_weights(wfile, weights)           # per-count
            skyjo_ai.save_weights("skyjo_ai_weights.json", weights)  # optional global
            skyjo_ai.load_weights(wfile)
            self.add_log(f"AI weights applied and saved to {wfile}")
            if isinstance(self.current_player(), AIPlayer) and self.wait_for_next and not self.ai_busy:
                self.set_info("Weights updated. Click Next Player to let AI act with the new settings.")
        except Exception as e:
            self.add_log(f"Failed to save/apply weights: {e}")


    def _weights_reset_defaults(self):
        defaults = skyjo_ai.get_default_weights()
        for k in self._wvars:
            self._wvars[k].set(defaults[k])
        self.add_log("AI weights reset (not saved yet). Click Apply & Save to persist.")

    def _weights_close(self):
        try:
            self.win_weights.destroy()
        except Exception:
            pass
        self.win_weights = None

# Accept unknown 'radius' kw for compatibility (no-op but avoids crashes if used elsewhere)
_orig_create_rectangle = tk.Canvas.create_rectangle
def _create_rectangle_with_radius(self, x1, y1, x2, y2, **kwargs):
    kwargs.pop("radius", None)
    return _orig_create_rectangle(self, x1, y1, x2, y2, **kwargs)
tk.Canvas.create_rectangle = _create_rectangle_with_radius

def main():
    root = tk.Tk()
    app = SkyjoGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
