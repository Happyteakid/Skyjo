# Kivy UI for Skyjo — uses skyjo_ai for decisions, Python-only logic
from kivy.app import App
from kivy.clock import Clock
from kivy.properties import ListProperty, StringProperty, NumericProperty, BooleanProperty, ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
import random, json, os

import skyjo_ai as AI

ROWS = 3
START_COLS = 4

def build_deck():
    return ([-2]*5) + ([-1]*10) + ([0]*15) + sum(([v]*10 for v in range(1,13)), [])

def value_color(v: int) -> list:
    # return RGBA (0..1) for Kivy
    hex_map = {
        **{ -2:"#6a5acd", -1:"#8a2be2", 0:"#1e90ff" },
        1:"#2e8b57", 2:"#3cb371", 3:"#66cdaa",
        4:"#9acd32", 5:"#ffd700", 6:"#f0ad4e", 7:"#ffa500",
        8:"#ff7f50", 9:"#ff6347", 10:"#ff4500", 11:"#e34234", 12:"#b22222"
    }
    h = hex_map.get(v, "#cccccc")
    r = int(h[1:3],16)/255.0; g = int(h[3:5],16)/255.0; b = int(h[5:7],16)/255.0
    return [r,g,b,1]

class Player:
    def __init__(self, name, is_ai=False):
        self.name = name
        self.is_ai = is_ai
        self.cols = []
        self.up = []

    def setup(self, deck):
        self.cols = [[deck.pop() for _ in range(ROWS)] for _ in range(START_COLS)]
        self.up   = [[False]*ROWS for _ in range(START_COLS)]
        spots = [(j,r) for j in range(START_COLS) for r in range(ROWS)]
        for j,r in random.sample(spots, 2): self.up[j][r] = True

    def num_cols(self): return len(self.cols)
    def all_face_up(self): return all(self.up[j][r] for j in range(self.num_cols()) for r in range(ROWS))
    def facedown_positions(self): return [(j,r) for j in range(self.num_cols()) for r in range(ROWS) if not self.up[j][r]]
    def score(self): return sum(self.cols[j][r] for j in range(self.num_cols()) for r in range(ROWS))
    def public(self) -> AI.PublicBoard:
        return AI.PublicBoard(cols=[c[:] for c in self.cols], up=[u[:] for u in self.up])

    def swap_in(self, j, r, new_card):
        out = self.cols[j][r]
        self.cols[j][r] = new_card; self.up[j][r] = True
        removed_vals = None
        if all(self.up[j]) and (self.cols[j][0] == self.cols[j][1] == self.cols[j][2]):
            removed_vals = [self.cols[j][0], self.cols[j][1], self.cols[j][2]]
            del self.cols[j]; del self.up[j]
        return out, removed_vals

    def flip_at(self, j, r):
        self.up[j][r] = True
        removed_vals = None
        if all(self.up[j]) and (self.cols[j][0] == self.cols[j][1] == self.cols[j][2]):
            removed_vals = [self.cols[j][0], self.cols[j][1], self.cols[j][2]]
            del self.cols[j]; del self.up[j]
        return removed_vals

class CardButton(Button):
    j = NumericProperty(0)
    r = NumericProperty(0)
    val = NumericProperty(0)
    face_up = BooleanProperty(False)

    def apply_style(self):
        if self.face_up:
            self.background_color = value_color(self.val)
            self.color = [1,1,1,1] if self.val>=8 else [0,0,0,1]
            self.text = str(self.val)
        else:
            self.background_color = [0.83,0.83,0.83,1]
            self.color = [0,0,0,1]
            self.text = "?"

class Root(BoxLayout):
    info = StringProperty("")
    stock_len = NumericProperty(0)
    discard_top = StringProperty("-")
    pending_text = StringProperty("-")
    can_draw = BooleanProperty(True)
    can_take = BooleanProperty(True)
    can_swap = BooleanProperty(False)
    can_flip = BooleanProperty(False)

    board_grid = ObjectProperty(None)
    discard_btn = ObjectProperty(None)
    stock_btn = ObjectProperty(None)
    pending_btn = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.players = []
        self.turn = 0
        self.stock = []
        self.discard = []
        self.pending_card = None
        self.pending_src = None
        self.mode = "idle"
        self.wait_for_next = True
        self.ending_idx = None
        self.final_left = 0
        self.ai_think_ev = None

        # one-time load of default weights (per-count file loaded in new_round)
        # (no-op if not present)
        try:
            if os.path.exists("skyjo_ai_weights.json"):
                AI.load_weights("skyjo_ai_weights.json")
        except Exception:
            pass

        Clock.schedule_once(lambda dt: self.new_round())

    # -------- UI state & drawing --------
    def set_info(self, s): self.info = s
    def refresh_hdr(self):
        self.stock_len = len(self.stock)
        self.discard_top = str(self.discard[-1]) if self.discard else "-"
        self.pending_text = str(self.pending_card) if self.pending_card is not None else "-"

    def rebuild_board(self):
        self.board_grid.clear_widgets()
        p = self.players[self.turn]
        for j in range(p.num_cols()):
            for r in range(ROWS):
                btn = CardButton(j=j, r=r, val=p.cols[j][r], face_up=p.up[j][r])
                btn.apply_style()
                btn.bind(on_release=self.on_card_tap)
                self.board_grid.add_widget(btn)

    def update_buttons(self):
        p = self.players[self.turn]
        self.can_draw = (self.pending_card is None)
        self.can_take = (self.pending_card is None and len(self.discard)>0)
        self.can_swap = (self.pending_card is not None)
        self.can_flip = (self.pending_card is not None and self.pending_src=="stock" and len(p.facedown_positions())>0)

    # -------- round setup --------
    def new_round(self):
        # ask players via a quick popup
        def _start(n, ai):
            # load per-count weights if present
            try:
                wfile = f"skyjo_ai_weights_{n}p.json"
                if os.path.exists(wfile):
                    AI.load_weights(wfile)
                elif os.path.exists("skyjo_ai_weights.json"):
                    AI.load_weights("skyjo_ai_weights.json")
            except Exception:
                pass

            self.stock = build_deck(); random.shuffle(self.stock)
            self.discard = [self.stock.pop()]
            self.players = []
            # P1 human, others AI by default; adjust ai count if needed
            ai = min(ai, max(0, n-1))
            for i in range(n):
                if i < n - ai:
                    pl = Player(f"P{i+1} [You]", is_ai=False)
                else:
                    pl = Player(f"P{i+1} [AI]", is_ai=True)
                pl.setup(self.stock); self.players.append(pl)
            self.turn = 0
            self.pending_card = None; self.pending_src=None; self.mode="idle"
            self.ending_idx=None; self.final_left=0
            self.refresh_hdr(); self.rebuild_board(); self.update_buttons()
            self.set_info("Your turn. Draw or take the discard.")
        AskPlayersPopup(start_cb=_start).open()

    # -------- interactions --------
    def on_draw(self):
        if not self.can_draw: return
        if not self.stock:
            # reshuffle
            if len(self.discard)>1:
                top = self.discard.pop()
                self.stock = self.discard[:]; random.shuffle(self.stock); self.discard=[top]
            else:
                return
        self.pending_card = self.stock.pop(); self.pending_src="stock"; self.mode="swap"
        self.set_info(f"Drew {self.pending_card}. Tap a slot to Swap, or press Discard & Flip.")
        self.refresh_hdr(); self.update_buttons(); self.rebuild_board()
        p = self.players[self.turn]
        if p.is_ai:
            Clock.schedule_once(self._ai_after_draw_step, 0.25)

    def on_take_discard(self):
        if not self.can_take: return
        if not self.discard: return
        self.pending_card = self.discard.pop(); self.pending_src="discard"; self.mode="swap"
        self.set_info(f"Took discard {self.pending_card}. Tap a slot to Swap.")
        self.refresh_hdr(); self.update_buttons(); self.rebuild_board()
        if self.players[self.turn].is_ai:
            # pick slot right away
            Clock.schedule_once(lambda dt: self._ai_place_discard(), 0.25)

    def on_card_tap(self, btn: CardButton):
        if self.mode == "swap" and self.pending_card is not None:
            self._perform_swap(btn.j, btn.r)
        elif self.mode == "flip" and self.pending_card is None:
            p = self.players[self.turn]
            if p.up[btn.j][btn.r]:
                self.set_info("Already face-up. Pick a facedown slot.")
                return
            self._perform_flip(btn.j, btn.r)

    def on_discard_and_flip(self):
        if not self.can_flip: return
        # move pending to discard & wait for the user to tap a facedown card
        self.discard.append(self.pending_card)
        self.pending_card = None; self.mode="flip"
        self.set_info("Discarded. Tap a facedown slot to flip.")
        self.refresh_hdr(); self.update_buttons(); self.rebuild_board()

    def _perform_swap(self, j, r):
        p = self.players[self.turn]
        out, removed = p.swap_in(j, r, self.pending_card)
        # discard order: out first, then removed triple
        self.discard.append(out)
        if removed:
            for v in removed: self.discard.append(v)
        self.pending_card = None; self.pending_src=None; self.mode="idle"
        self.refresh_hdr(); self.rebuild_board(); self.update_buttons()
        self._after_action_finalize(p)

    def _perform_flip(self, j, r):
        p = self.players[self.turn]
        # pending is None here (already discarded)
        removed = p.flip_at(j, r)
        if removed:
            for v in removed: self.discard.append(v)
        self.mode="idle"
        self.refresh_hdr(); self.rebuild_board(); self.update_buttons()
        self._after_action_finalize(p)

    def _after_action_finalize(self, p: Player):
        if p.all_face_up() and self.ending_idx is None:
            self.ending_idx = self.turn
            self.final_left = len(self.players) - 1
            self.set_info(f"{p.name} finished. Everyone else gets one final turn.")
        if self.wait_for_next:
            self.set_info(self.info + "  (Press Next Player)")
        else:
            Clock.schedule_once(lambda dt: self.on_next_player(), 0.35)

    def on_next_player(self):
        # endgame countdown
        if self.ending_idx is not None and self.turn != self.ending_idx:
            self.final_left -= 1
            if self.final_left <= 0:
                return self.finish_round()
        self.turn = (self.turn + 1) % len(self.players)
        self.pending_card=None; self.pending_src=None; self.mode="idle"
        self.refresh_hdr(); self.rebuild_board(); self.update_buttons()
        p = self.players[self.turn]
        if p.is_ai:
            self.set_info(f"{p.name} thinking…")
            Clock.schedule_once(lambda dt: self._ai_begin_turn(), 0.25)
        else:
            self.set_info(f"{p.name}: your move.")

    def finish_round(self):
        # reveal all & score (same penalty rule)
        for pl in self.players:
            for j in range(pl.num_cols()):
                for r in range(ROWS): pl.up[j][r] = True
        scores = [pl.score() for pl in self.players]
        msg = "\n".join(f"{pl.name}: {sc}" for pl, sc in zip(self.players, scores))
        if self.ending_idx is not None:
            low = min(scores) if scores else 0
            if scores[self.ending_idx] != low and scores[self.ending_idx] > 0:
                scores[self.ending_idx] *= 2
                msg += f"\n(Penalty: {self.players[self.ending_idx].name} doubled)"
        winner = min(range(len(self.players)), key=lambda i: scores[i]) if scores else 0
        Popup(title="Round Over", content=Label(text=msg+f"\n\nWinner: {self.players[winner].name}"),
              size_hint=(0.7,0.7)).open()
        # start a new round automatically after popup is closed? up to you.

    # -------- AI plumbing --------
    def _public_state(self) -> AI.PublicState:
        boards = [p.public() for p in self.players]
        return AI.PublicState(
            me_index=self.turn,
            prev_index=(self.turn-1) % len(self.players),
            players=boards,
            discard_top=(self.discard[-1] if self.discard else None),
            discard_len=len(self.discard),
            stock_len=len(self.stock),
            discard_list=self.discard[:]   # full pile for probabilities
        )

    def _ai_begin_turn(self):
        st = self._public_state()
        act, tgt = AI.choose_action(st)
        if act == 'take_discard' and self.discard:
            self.on_take_discard()
            j,r = tgt
            Clock.schedule_once(lambda dt: self._perform_swap(j,r), 0.2)
        else:
            self.on_draw()
            Clock.schedule_once(self._ai_after_draw_step, 0.25)

    def _ai_after_draw_step(self, *_):
        if self.pending_card is None: return
        st2 = self._public_state()
        act, j, r = AI.choose_after_draw(st2, self.pending_card)
        if act == 'swap':
            Clock.schedule_once(lambda dt: self._perform_swap(j,r), 0.1)
        else:
            Clock.schedule_once(lambda dt: self.on_discard_and_flip(), 0.05)
            # flip target after a tiny delay so UI shows discard first
            Clock.schedule_once(lambda dt: self._perform_flip(j,r), 0.18)

    def _ai_place_discard(self):
        if self.pending_card is None: return
        st = self._public_state()
        # When taking discard, choose_action already computed target; recompute here quickly
        imp, j, r, _ = AI.best_swap_improvement_for_card(self.players[self.turn].public(), self.pending_card, 0, {}, AI.get_default_weights())
        self._perform_swap(j, r)

class AskPlayersPopup(Popup):
    def __init__(self, start_cb, **kwargs):
        super().__init__(**kwargs)
        self.start_cb = start_cb
        self.title = "Start Round"
        self.size_hint = (0.8, 0.6)
        root = BoxLayout(orientation="vertical", spacing=8, padding=12)
        self.content = root
        row1 = BoxLayout()
        row1.add_widget(Label(text="Total players (2–6):"))
        self.p_in = Label(text="4")
        row1.add_widget(self.p_in)
        row2 = BoxLayout()
        row2.add_widget(Label(text="Boss AIs:"))
        self.ai_in = Label(text="3")
        row2.add_widget(self.ai_in)
        ctrl = BoxLayout(spacing=8, size_hint_y=None, height=48)
        bdec = Button(text="-"); binc = Button(text="+"); bdec2 = Button(text="-"); binc2 = Button(text="+")
        bstart = Button(text="Start")
        root.add_widget(row1); root.add_widget(row2); root.add_widget(ctrl)
        ctrl.add_widget(bdec); ctrl.add_widget(binc); ctrl.add_widget(bdec2); ctrl.add_widget(binc2); ctrl.add_widget(bstart)
        bdec.bind(on_release=lambda *_: self._set_players(-1))
        binc.bind(on_release=lambda *_: self._set_players(+1))
        bdec2.bind(on_release=lambda *_: self._set_ai(-1))
        binc2.bind(on_release=lambda *_: self._set_ai(+1))
        bstart.bind(on_release=lambda *_: self._go())
        self.n = 4; self.ai = 3

    def _set_players(self, d):
        self.n = max(2, min(6, self.n + d))
        self.p_in.text = str(self.n)
        self.ai = min(self.ai, self.n-1); self.ai_in.text = str(self.ai)

    def _set_ai(self, d):
        self.ai = max(1, min(self.n-1, self.ai + d))
        self.ai_in.text = str(self.ai)

    def _go(self):
        self.dismiss()
        self.start_cb(self.n, self.ai)

class SkyjoApp(App):
    def build(self):
        return Root()

if __name__ == "__main__":
    SkyjoApp().run()
