# skyjo_ai.py — Boss AI (patched)
# - Blocks finishing the round (by swap or flip) when likely behind
# - Strongly penalizes swapping high cards (>= HIGH_SWAP_BLOCK_MIN) unless it removes a column
# - Uses threats from the NEXT player (the one who can take your discard)
# - Optionally consumes the full discard pile for probability (state.discard_list); falls back if absent
# - Parameterized weights + save/load

import json, random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

ROWS = 3

VERSION = "boss-patched nxthreat+guards+clamps v4"
def ai_version():
    import os
    return f"{VERSION} | file={__file__}"


# -------- Learnable weights (defaults) --------
DEFAULT_WEIGHTS = {
    # Swap-up penalties (no immediate removal)
    "pen_swap_up_base": 6.0,
    "pen_swap_up_nonpos_extra": 9.0,
    "pen_swap_up_gap_mult": 0.35,

    # HARD block for very high cards unless removing a column
    "high_swap_block_min": 10.0,      # 10/11/12
    "high_swap_block_penalty": 30.0,  # huge negative to prevent nonsense swaps

    # Discard denial / risk controls
    "deny_bonus_cap": 8.0,
    "deny_bonus_factor": 0.6,
    "out_risk_penalty_factor": 0.5,

    # Action tie-breaks
    "swap_beats_flip_margin": -0.2,   # >= means swap; negative slightly favors flip
    "discard_over_stock_bias": 0.6,   # how much better discard must be vs stock EV

    # Endgame caution: only finish if <= best opponent + margin
    "endgame_finish_margin": -1.5,     # tighter than before; need ~1.5pt cushion
    # NEW: finishing safety when you still hold a big card
    "finish_high_threshold": 10.0,     # treat 10/11/12 as risky to finish with
    "finish_high_extra_margin": 1.8,   # extra cushion required (points) to still finish
}

_active_weights = DEFAULT_WEIGHTS.copy()

def get_default_weights() -> Dict[str, float]:
    return DEFAULT_WEIGHTS.copy()

def load_weights(path: str):
    global _active_weights
    try:
        with open(path, "r", encoding="utf-8") as f:
            w = json.load(f)
        for k in DEFAULT_WEIGHTS:
            if k in w: _active_weights[k] = float(w[k])
        _apply_clamps(_active_weights)
    except Exception:
        pass

def save_weights(path: str, weights: Optional[Dict[str, float]] = None):
    w = (_active_weights if weights is None else weights)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(w, f, indent=2)

CLAMPS = {
    "pen_swap_up_base": (0.0, 20.0),
    "pen_swap_up_nonpos_extra": (0.0, 40.0),
    "pen_swap_up_gap_mult": (0.0, 2.0),  # never negative
    "high_swap_block_min": (9.5, 12.5),
    "high_swap_block_penalty": (20.0, 60.0),
    "deny_bonus_cap": (0.0, 20.0),
    "deny_bonus_factor": (0.3, 1.5),     # don’t let it go near zero
    "out_risk_penalty_factor": (0.3, 2.5),
    "swap_beats_flip_margin": (-0.5, 0.5),
    "discard_over_stock_bias": (0.0, 1.0),
    "endgame_finish_margin": (-3.0, 0.0),
}

def _apply_clamps(w):
    for k,(lo,hi) in CLAMPS.items():
        if k in w:
            w[k] = max(lo, min(hi, float(w[k])))
    return w

def _resolve_weights(weights=None):
    if weights is None:
        return _apply_clamps(_active_weights.copy())
    merged = DEFAULT_WEIGHTS.copy()
    for k,v in weights.items():
        if k in merged: merged[k] = float(v)
    return _apply_clamps(merged)

# -------- Public state wrappers --------
@dataclass
class PublicBoard:
    cols: List[List[int]]
    up:   List[List[bool]]
    def num_cols(self) -> int: return len(self.cols)
    def facedown_positions(self) -> List[Tuple[int,int]]:
        return [(j,r) for j in range(self.num_cols()) for r in range(ROWS) if not self.up[j][r]]
    def estimated_score(self, expected_hidden: float) -> float:
        s = 0.0
        for j in range(self.num_cols()):
            for r in range(ROWS):
                s += self.cols[j][r] if self.up[j][r] else expected_hidden
        return s

@dataclass
class PublicState:
    me_index: int
    prev_index: int
    players: List[PublicBoard]
    discard_top: Optional[int]
    discard_len: int
    stock_len: int
    # Optional: full discard list (public info). If your GUI/engine doesn’t pass this yet, it’s fine.
    discard_list: Optional[List[int]] = None

# -------- Counting & threat helpers --------
def init_counts() -> Dict[int,int]:
    counts = {-2:5, -1:10, 0:15}
    for v in range(1,13): counts[v] = 10
    return counts

def has_high_face_up(pb: PublicBoard, thr: float) -> bool:
    for j in range(pb.num_cols()):
        for r in range(ROWS):
            if pb.up[j][r] and pb.cols[j][r] >= thr:
                return True
    return False


def remaining_counts(state: PublicState) -> Tuple[Dict[int,int], int, float]:
    counts = init_counts()
    # subtract visible board cards
    for pb in state.players:
        for j in range(pb.num_cols()):
            for r in range(ROWS):
                if pb.up[j][r]:
                    v = pb.cols[j][r]
                    if counts.get(v,0)>0: counts[v]-=1
    # subtract full discard if provided (top + deeper cards)
    if state.discard_list:
        for v in state.discard_list:
            if counts.get(v,0)>0: counts[v]-=1

    total_unseen = sum(counts.values())
    exp_hidden = (sum(v*counts[v] for v in counts)/total_unseen) if total_unseen>0 else 0.0
    return counts, total_unseen, exp_hidden

def next_player_index(state: PublicState) -> int:
    return (state.me_index + 1) % len(state.players)

def prev_player_pair_threats(pb: PublicBoard) -> Dict[int,float]:
    """value -> estimated benefit for player if they complete/remove soon (only positives)."""
    threats = {}
    for j in range(pb.num_cols()):
        ups = [pb.up[j][r] for r in range(ROWS)]
        vals = [pb.cols[j][r] for r in range(ROWS)]
        if ups.count(True)==2 and ups.count(False)==1:
            face = [vals[r] for r in range(ROWS) if ups[r]]
            if len(face)==2 and face[0]==face[1] and face[0] > 0:
                x = face[0]
                threats[x] = max(threats.get(x, 0.0), 3.0*x)
    return threats

# -------- EV helpers --------
def best_flip_ev(pb: PublicBoard, counts: Dict[int,int], total_unseen: int) -> Tuple[float, Optional[Tuple[int,int]]]:
    best_ev = 0.0; best_target = None
    if total_unseen <= 0:
        fds = pb.facedown_positions(); return (0.0, fds[0] if fds else None)
    for j in range(pb.num_cols()):
        ups = [pb.up[j][r] for r in range(ROWS)]
        if ups.count(True)==2 and ups.count(False)==1:
            vals = [pb.cols[j][r] for r in range(ROWS)]
            r_hidden = ups.index(False)
            face_vals = [vals[r] for r in range(ROWS) if r != r_hidden]
            if face_vals[0]==face_vals[1]:
                x = face_vals[0]; cnt = counts.get(x,0)
                p_match = (cnt/total_unseen) if total_unseen>0 else 0.0
                ev = p_match*(3*x)
                if ev < 0: ev = 0.0
                if ev > best_ev: best_ev, best_target = ev, (j, r_hidden)
    if best_target is None:
        fds = pb.facedown_positions(); best_target = fds[0] if fds else None
    return (best_ev, best_target)

def _resolve_weights(weights: Optional[Dict[str,float]]) -> Dict[str,float]:
    if weights is None: 
        return _active_weights
    w = DEFAULT_WEIGHTS.copy()
    for k,v in weights.items():
        if k in w: w[k] = float(v)
    return w

def best_swap_improvement_for_card(pb: PublicBoard, card: int, expected_hidden: float,
                                   threats: Dict[int,float], w: Dict[str,float]) -> Tuple[float,int,int,bool]:
    """
    Return (improvement, j, r, will_remove).
    Includes:
      - heavy penalty if swapping a higher card over a lower (esp. ≤0) without removal,
      - hard block for very high cards (>= high_swap_block_min) unless removal,
      - penalty if the out-card would help the next player's threatened pair.
    """
    best = (float('-inf'), 0, 0, False)
    for j in range(pb.num_cols()):
        for r in range(ROWS):
            out_true = pb.cols[j][r]
            out_est  = pb.cols[j][r] if pb.up[j][r] else expected_hidden
            o1,o2 = [rr for rr in range(ROWS) if rr != r]
            will_remove = (pb.up[j][o1] and pb.up[j][o2] and pb.cols[j][o1]==card and pb.cols[j][o2]==card)

            if will_remove:
                imp = pb.cols[j][o1] + pb.cols[j][o2] + out_est
            else:
                imp = out_est - card

                # HARD block for very high incoming card unless removal
                if card >= w["high_swap_block_min"]:
                    imp -= w["high_swap_block_penalty"]

                # Discourage swapping a higher card over a lower one
                if card > out_true:
                    gap = card - out_true
                    imp -= w["pen_swap_up_base"] + w["pen_swap_up_gap_mult"]*gap
                    if out_true <= 0:
                        imp -= w["pen_swap_up_nonpos_extra"]

            # Don't feed next player's threatened pair
            threat = threats.get(out_true, 0.0)
            if threat > 0:
                imp -= w["out_risk_penalty_factor"] * threat

            if imp > best[0]:
                best = (imp, j, r, will_remove)
    return best

# -------- Decisions --------
def choose_action(state: PublicState, weights: Optional[Dict[str,float]] = None) -> Tuple[str, Optional[Tuple[int,int]]]:
    w = _resolve_weights(weights)
    me = state.players[state.me_index]
    counts, total_unseen, exp_hidden = remaining_counts(state)

    # Use threats from the NEXT player (they act after us)
    nxt = next_player_index(state)
    threats = prev_player_pair_threats(state.players[nxt])

    # Evaluate taking the top discard
    imp_disc = float('-inf'); best_target = (None,None,False)
    if state.discard_top is not None:
        imp_disc, dj, dr, will_remove = best_swap_improvement_for_card(me, state.discard_top, exp_hidden, threats, w)
        best_target = (dj, dr, will_remove)

    # Stock EV (sample unseen)
    vals = []
    for v,c in counts.items():
        if c>0: vals.extend([v]*c)
    exp_stock = 0.0
    if vals:
        m = min(80, len(vals))
        for _ in range(m):
            v = random.choice(vals)
            imp, _, _, _ = best_swap_improvement_for_card(me, v, exp_hidden, threats, w)
            ev_flip, _ = best_flip_ev(me, counts, total_unseen)
            exp_stock += max(imp, ev_flip)
        exp_stock /= m

    # Deny bonus (don’t feed NEXT player’s threatened pair)
    deny_bonus = 0.0
    if state.discard_top is not None and state.discard_top in threats:
        deny_bonus = min(threats[state.discard_top], w["deny_bonus_cap"]) * w["deny_bonus_factor"]

    # Finishing guard used below
    def finishing_guard_if_target(j, r) -> bool:
        fds = me.facedown_positions()
        if not (len(fds) == 1 and fds[0] == (j, r)):  # only matters if this swap would be the last face-down
            return False
        own = me.estimated_score(exp_hidden)
        others = [state.players[i] for i in range(len(state.players)) if i != state.me_index]
        best_other = min(o.estimated_score(exp_hidden) for o in others) if others else own - 1
        ahead = best_other - own                      # positive = we are better (lower)
        need = -w["endgame_finish_margin"]            # e.g. 1.5 points cushion
        if has_high_face_up(me, w["finish_high_threshold"]):
            need += w["finish_high_extra_margin"]     # need more cushion if we still show a big card
        return ahead < need

    take_discard = False
    if state.discard_top is not None:
        # removal with positive column sum → good
        if best_target[2]:
            j, r = best_target[0], best_target[1]
            if j is not None and r is not None:
                o1,o2 = [rr for rr in range(ROWS) if rr != r]
                colsum = me.cols[j][o1] + me.cols[j][o2] + (me.cols[j][r] if me.up[j][r] else exp_hidden)
                if colsum > 0:
                    take_discard = True

        # otherwise compare EV + deny; but don’t take if it would FORCE a finishing swap and we lack cushion
        if not take_discard and (imp_disc + deny_bonus) > (exp_stock + w["discard_over_stock_bias"]):
            j, r = best_target[0], best_target[1]
            if j is not None and r is not None and finishing_guard_if_target(j, r):
                take_discard = False
            else:
                take_discard = True

    return ('take_discard', (best_target[0], best_target[1])) if take_discard else ('draw_stock', None)


def choose_after_draw(state: PublicState, pending_card: int, weights: Optional[Dict[str,float]] = None) -> Tuple[str, int, int]:
    w = _resolve_weights(weights)
    me = state.players[state.me_index]
    counts, total_unseen, exp_hidden = remaining_counts(state)

    # threats from NEXT player
    nxt = next_player_index(state)
    threats = prev_player_pair_threats(state.players[nxt])

    imp_swap, sj, sr, will_remove = best_swap_improvement_for_card(me, pending_card, exp_hidden, threats, w)
    ev_flip, flip_target = best_flip_ev(me, counts, total_unseen)

    # Would an action FINISH the round?
    fds = me.facedown_positions()
    def would_finish(j, r) -> bool:
        return (len(fds) == 1 and fds[0] == (j, r))

    # Do we have enough cushion to finish?
    def should_avoid_finishing_target(j, r) -> bool:
        if not would_finish(j, r): return False
        own = me.estimated_score(exp_hidden)
        others = [state.players[i] for i in range(len(state.players)) if i != state.me_index]
        best_other = min(o.estimated_score(exp_hidden) for o in others) if others else own - 1
        ahead = best_other - own
        need = -w["endgame_finish_margin"]
        if has_high_face_up(me, w["finish_high_threshold"]):
            need += w["finish_high_extra_margin"]
        return ahead < need

    # Base preference (EV)
    do_swap = (imp_swap >= ev_flip + w["swap_beats_flip_margin"])

    # Never swap a very high incoming card unless it removes a column
    if (not will_remove) and pending_card >= w["high_swap_block_min"]:
        do_swap = False

    # Avoid finishing by swap if not enough cushion
    if do_swap and sj is not None and sr is not None and should_avoid_finishing_target(sj, sr):
        do_swap = False

    # Avoid finishing by flip if not enough cushion
    if (not do_swap) and flip_target is not None:
        fj, fr = flip_target
        if should_avoid_finishing_target(fj, fr):
            do_swap = True

    # Fallbacks
    if flip_target is None:
        do_swap = True
    if sj is None or sr is None:
        do_swap = False

    if do_swap:
        return ('swap', sj, sr)
    else:
        fj, fr = flip_target
        return ('flip', fj, fr)
