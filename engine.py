"""Texas Hold'em engine for Drex Poker Arena.

- 1000 chips, blinds 10/20 (configurable)
- Actions: fold, check_call, raise_half_pot, raise_pot, all_in
- Max 2 raises per betting round
- Pots split by contribution layers so chips are always conserved
- The game pauses whenever it is a human's turn
"""
import random
from collections import deque

from treys import Card, Evaluator

ACTIONS = ["fold", "check_call", "raise_half_pot", "raise_pot", "all_in"]
RAISE_ACTIONS = {"raise_half_pot", "raise_pot", "all_in"}
STREETS = ["preflop", "flop", "turn", "river"]
MAX_RAISES = 2
EVAL = Evaluator()
FULL_DECK = [Card.new(r + s) for r in "23456789TJQKA" for s in "shdc"]


def cstr(c):
    return Card.int_to_str(c)


def cards_str(cs):
    return [cstr(c) for c in cs]


class Player:
    def __init__(self, seat, name, personality, chips=1000, human=False, brain="drex"):
        self.seat = seat
        self.name = name
        self.personality = personality  # risk_taker, conservative, statistician, manipulator, wildcard, rule_based, human
        self.brain = brain              # "drex" | "rule" | "human"
        self.chips = chips
        self.human = human
        self.reset_hand()

    def reset_hand(self):
        self.hole = []
        self.folded = False
        self.all_in = False
        self.out = self.chips <= 0
        if self.out:
            self.folded = True
        self.bet = 0       # this betting round
        self.contrib = 0   # whole hand
        self.acted = False
        self.last_action = None

    @property
    def can_act(self):
        return not self.folded and not self.all_in and not self.out


class Tracker:
    """Opponent history every bot can read (the Manipulator leans on it)."""

    def __init__(self):
        self.recent = {}      # seat -> deque of last 6 actions
        self.raises = {}      # seat -> total raises
        self.actions = {}     # seat -> total actions
        self.showdowns = {}   # seat -> showdowns where they had raised
        self.bluffs = {}      # seat -> of those, how many were weak (bluffs)
        self.folds = {}       # seat -> total folds

    def _init(self, seat):
        if seat not in self.recent:
            self.recent[seat] = deque(maxlen=6)
            self.raises[seat] = 0
            self.actions[seat] = 0
            self.showdowns[seat] = 0
            self.bluffs[seat] = 0

    def record(self, seat, action):
        self._init(seat)
        self.recent[seat].append(action)
        self.actions[seat] += 1
        if action == "fold":
            self.folds[seat] = self.folds.get(seat, 0) + 1
        if action in RAISE_ACTIONS:
            self.raises[seat] += 1

    def record_showdown(self, seat, bluffed):
        self._init(seat)
        self.showdowns[seat] += 1
        if bluffed:
            self.bluffs[seat] += 1

    def raise_rate(self, seat):
        self._init(seat)
        r = list(self.recent[seat])
        if not r:
            return 0.0
        return sum(a in RAISE_ACTIONS for a in r) / len(r)

    def summary(self, seat):
        self._init(seat)
        r = list(self.recent[seat])
        acts = self.actions[seat]
        folds = self.folds.get(seat, 0)
        rr = self.raise_rate(seat)
        fold_rate = folds / acts if acts else 0.0
        if acts < 3:
            style = "unknown"
        else:
            style = ("loose-aggressive" if rr >= 0.5 and fold_rate < 0.35 else
                     "aggressive" if rr >= 0.5 else
                     "tight-passive" if fold_rate >= 0.5 else
                     "calling-station" if fold_rate < 0.2 and rr < 0.25 else "balanced")
        return {
            "recent": r,
            "fold_rate": fold_rate,
            "style": style,
            "raise_rate": rr,
            "recent_raises": sum(a in RAISE_ACTIONS for a in r),
            "showdowns": self.showdowns[seat],
            "bluffs": self.bluffs[seat],
        }

    def seed(self, seat, recent, showdowns, bluffs):
        self._init(seat)
        self.recent[seat] = deque(recent, maxlen=6)
        for a in recent:
            self.actions[seat] += 1
            if a == "fold":
                self.folds[seat] = self.folds.get(seat, 0) + 1
            if a in RAISE_ACTIONS:
                self.raises[seat] += 1
        self.showdowns[seat] = showdowns
        self.bluffs[seat] = bluffs


class Game:
    def __init__(self, players, sb=10, bb=20, decide_fn=None, rng=None):
        self.players = players
        self.sb, self.bb = sb, bb
        self.decide_fn = decide_fn  # decide_fn(game, player) -> (action, info)
        self.rng = rng or random.Random()
        self.tracker = Tracker()
        self.dealer = -1
        self.hand_no = 0
        self.hand_over = True
        self.events = []
        self.decisions = []   # brain-peek log for current hand
        self.result = None
        self.board = []
        self.street = "preflop"

    # ---------- helpers ----------
    def emit(self, **ev):
        self.events.append(ev)

    def pop_events(self):
        ev, self.events = self.events, []
        return ev

    @property
    def pot(self):
        return sum(p.contrib for p in self.players)

    def in_hand(self):
        return [p for p in self.players if not p.folded and not p.out]

    def next_seat(self, i, pred):
        n = len(self.players)
        for k in range(1, n + 1):
            j = (i + k) % n
            if pred(self.players[j]):
                return j
        return None

    def to_call(self, p):
        return max(0, self.current_bet - p.bet)

    # ---------- hand setup ----------
    def start_hand(self, rig=None):
        """rig = {"holes": {seat: ["Ah","Kd"]}, "board": ["2c",...]} for demo hands."""
        for p in self.players:
            p.reset_hand()
        alive = [p for p in self.players if not p.out]
        if len(alive) < 2:
            self.hand_over = True
            self.emit(type="game_over", winner=alive[0].name if alive else None)
            return False
        self.hand_no += 1
        self.hand_over = False
        self.result = None
        self.decisions = []
        self.board = []
        self.street = "preflop"
        self.raises = 0
        self.current_bet = 0
        self.raised_equities = {}  # seat -> list of equities when they raised

        self.dealer = self.next_seat(self.dealer, lambda p: not p.out)

        # deck
        deck = FULL_DECK[:]
        self.rng.shuffle(deck)
        self.rig_board = []
        if rig:
            used = []
            for seat, hs in rig.get("holes", {}).items():
                used += [Card.new(c) for c in hs]
            self.rig_board = [Card.new(c) for c in rig.get("board", [])]
            used += self.rig_board
            deck = [c for c in deck if c not in used]
        self.deck = deck

        if len(alive) == 2:
            sb_seat = self.dealer
        else:
            sb_seat = self.next_seat(self.dealer, lambda p: not p.out)
        bb_seat = self.next_seat(sb_seat, lambda p: not p.out)
        self._post(self.players[sb_seat], self.sb)
        self._post(self.players[bb_seat], self.bb)
        self.current_bet = self.bb

        for p in self.players:
            if p.out:
                continue
            if rig and int(p.seat) in {int(k) for k in rig.get("holes", {})}:
                hs = rig["holes"].get(p.seat) or rig["holes"].get(str(p.seat))
                p.hole = [Card.new(c) for c in hs]
            else:
                p.hole = [self.deck.pop(), self.deck.pop()]

        self.to_act = self.next_seat(bb_seat, lambda p: p.can_act)
        self.emit(type="new_hand", hand_no=self.hand_no, dealer=self.dealer,
                  sb=sb_seat, bb=bb_seat)
        if self.to_act is None or self._round_done():
            self._end_round()
        return True

    def _post(self, p, amt):
        amt = min(amt, p.chips)
        p.chips -= amt
        p.bet += amt
        p.contrib += amt
        if p.chips == 0:
            p.all_in = True
        self.emit(type="blind", seat=p.seat, amount=amt)

    # ---------- actions ----------
    def legal(self, p):
        """Return {action: chips_this_action_costs}."""
        tc = self.to_call(p)
        out = {"fold": 0, "check_call": min(tc, p.chips)}
        if self.raises < MAX_RAISES and p.chips > tc:
            pot_after_call = self.pot + tc
            for name, frac in (("raise_half_pot", 0.5), ("raise_pot", 1.0)):
                raise_by = max(self.bb, int(pot_after_call * frac))
                cost = self.current_bet + raise_by - p.bet
                if cost < p.chips:
                    out[name] = cost
        if p.chips > 0 and (self.raises < MAX_RAISES or p.chips <= tc):
            out["all_in"] = p.chips
        return out

    def apply(self, p, action):
        legal = self.legal(p)
        if action not in legal:
            raise ValueError(f"illegal action {action} for {p.name}; legal={list(legal)}")
        cost = legal[action]
        tc = self.to_call(p)
        if action == "fold":
            p.folded = True
            label = "Fold"
        else:
            p.chips -= cost
            p.bet += cost
            p.contrib += cost
            if p.chips == 0:
                p.all_in = True
            if p.bet > self.current_bet:
                self.current_bet = p.bet
                self.raises += 1
                for o in self.players:
                    if o is not p:
                        o.acted = False
                self.raised_equities.setdefault(p.seat, [])
                label = "All-in" if action == "all_in" else f"Raise to {p.bet}"
            elif action == "all_in":
                label = f"All-in {cost}"
            else:
                label = "Check" if tc == 0 else f"Call {cost}"
        p.acted = True
        p.last_action = label
        self.tracker.record(p.seat, action)
        self.emit(type="action", seat=p.seat, action=action, label=label,
                  amount=cost, pot=self.pot, chips=p.chips)

        if self._round_done():
            self._end_round()
        else:
            self.to_act = self.next_seat(p.seat, lambda q: q.can_act)

    def _round_done(self):
        inh = self.in_hand()
        if len(inh) <= 1:
            return True
        for p in inh:
            if p.all_in:
                continue
            if not p.acted or p.bet != self.current_bet:
                return False
        return True

    def _end_round(self):
        inh = self.in_hand()
        if len(inh) == 1:
            self._award_single(inh[0])
            return
        while True:
            if self.street == "river":
                self._showdown()
                return
            self._next_street()
            actors = [p for p in self.in_hand() if p.can_act]
            if len(actors) >= 2:
                self.to_act = self.next_seat(self.dealer, lambda q: q.can_act)
                return
            # 0 or 1 players can still act -> run the board out

    def _next_street(self):
        idx = STREETS.index(self.street) + 1
        self.street = STREETS[idx]
        n = 3 if self.street == "flop" else 1
        for _ in range(n):
            if self.rig_board and len(self.board) < len(self.rig_board):
                self.board.append(self.rig_board[len(self.board)])
            else:
                self.board.append(self.deck.pop())
        for p in self.players:
            p.bet = 0
            p.acted = False
        self.current_bet = 0
        self.raises = 0
        self.emit(type="street", street=self.street, board=cards_str(self.board))

    # ---------- payouts ----------
    def _award_single(self, w):
        pot = self.pot
        w.chips += pot
        self._finish({w.seat: pot}, showdown=False, hands={})

    def _showdown(self):
        inh = self.in_hand()
        scores = {p.seat: EVAL.evaluate(self.board, p.hole) for p in inh}
        hands = {}
        for p in inh:
            cls = EVAL.get_rank_class(scores[p.seat])
            hands[p.seat] = {"cards": cards_str(p.hole),
                             "hand": EVAL.class_to_string(cls), "cls": cls}
            # bluff = raised this hand while win chance was < 40%
            if p.seat in self.raised_equities:
                eqs = self.raised_equities[p.seat]
                bluffed = bool(eqs) and min(eqs) < 0.40
                self.tracker.record_showdown(p.seat, bluffed)
                hands[p.seat]["bluffed"] = bluffed

        # layered pots (handles all-ins without losing chips)
        contribs = {p.seat: p.contrib for p in self.players if p.contrib > 0}
        levels = sorted(set(contribs.values()))
        won = {}
        prev = 0
        for lvl in levels:
            layer = sum(min(c, lvl) - min(c, prev) for c in contribs.values())
            elig = [p for p in inh if p.contrib >= lvl]
            if not elig:  # everyone eligible folded -> give to best remaining
                elig = inh
            best = min(scores[p.seat] for p in elig)
            winners = [p for p in elig if scores[p.seat] == best]
            share, rem = divmod(layer, len(winners))
            for i, wp in enumerate(winners):
                amt = share + (1 if i < rem else 0)
                wp.chips += amt
                won[wp.seat] = won.get(wp.seat, 0) + amt
            prev = lvl
        self._finish(won, showdown=True, hands=hands)

    def _finish(self, won, showdown, hands):
        self.hand_over = True
        self.to_act = None
        pot = sum(won.values())
        for p in self.players:
            p.contrib = 0
            p.bet = 0
        self.result = {
            "won": {str(k): v for k, v in won.items()},
            "winners": [self.players[s].name for s in won],
            "pot": pot,
            "showdown": showdown,
            "hands": {str(k): v for k, v in hands.items()},
            "board": cards_str(self.board),
        }
        self.emit(type="hand_over", **self.result)

    # ---------- driver ----------
    def run_bots(self):
        """Let bots act until it's a human's turn or the hand ends."""
        guard = 0
        while not self.hand_over and guard < 500:
            guard += 1
            p = self.players[self.to_act]
            if p.human:
                return
            action, info = self.decide_fn(self, p)
            if action in RAISE_ACTIONS and info and "equity" in info:
                self.raised_equities.setdefault(p.seat, []).append(info["equity"])
            if info:
                info["street"] = self.street
                self.decisions.append(info)
            self.apply(p, action)
