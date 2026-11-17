"""Bot brains: Drex probabilities + 5 personalities + rule-based baseline.

Every Drex bot asks Drex the SAME question (the game state) and gets a
probability for each action. The personality decides how to use those
probabilities. If Drex isn't configured, fails, or takes > 3s, we use a
local heuristic brain instead (marked source="local"/"fallback" so the UI
never pretends it was Drex).
"""
import math
import os
import random
import threading
import time

import requests
from treys import Card

from engine import ACTIONS, EVAL, FULL_DECK, RAISE_ACTIONS, cards_str

# ---------------- config ----------------
# Native Nace API: POST https://drex.nace.ai/v1/systemone (bearer nace_sk_...)
# DeepInfra mirror: POST https://api.deepinfra.com/v1/decisions  (model nace-ai/drex-v2.0)
DREX_URL = os.getenv("DREX_URL") or "https://drex.nace.ai/v1/systemone"
DREX_API_KEY = os.getenv("DREX_API_KEY", "")
DREX_MODEL = os.getenv("DREX_MODEL") or "drex-latest"
DREX_PRICE_PER_M_INPUT = float(os.getenv("DREX_PRICE_PER_M_INPUT", "0.05"))  # USD / 1M input tokens
_TL = threading.local()  # per-thread last Drex usage/extras (arena runs tables in parallel)
DREX_TIMEOUT = float(os.getenv("DREX_TIMEOUT", "3"))
DREX_COST_PER_CALL = float(os.getenv("DREX_COST_PER_CALL", "0"))  # USD per decision
MC_RUNS = int(os.getenv("MC_RUNS", "300"))
MANIP_BLUFF_THRESHOLD = float(os.getenv("MANIP_BLUFF_THRESHOLD", "0.5"))
# LLM "Giant" for the Apex Predator benchmark (either key works)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL") or ("claude-sonnet-5-5" if ANTHROPIC_API_KEY else "gpt-4o")
LLM_PRICE_IN = float(os.getenv("LLM_PRICE_IN", "3.0"))     # USD per 1M input tokens (set to your model's price)
LLM_PRICE_OUT = float(os.getenv("LLM_PRICE_OUT", "15.0"))  # USD per 1M output tokens
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))
ALABEL = {"fold": "Fold", "check_call": "Check/Call", "raise_half_pot": "Raise 1/2 pot",
          "raise_pot": "Raise pot", "all_in": "All-in"}

PERSONALITIES = {
    "risk_taker":   {"name": "Risk-Taker",   "emoji": "🔥"},
    "conservative": {"name": "Conservative", "emoji": "🪨"},
    "statistician": {"name": "Statistician", "emoji": "🧮"},
    "manipulator":  {"name": "Manipulator",  "emoji": "🕵️"},
    "wildcard":     {"name": "Wildcard",     "emoji": "🎲"},
    "rule_based":   {"name": "Rule-Bot",     "emoji": "🤖"},
    "llm_bot":      {"name": "LLM-Bot",      "emoji": "💬"},
}

BUBBLES = {
    "risk_taker": {"raise": ["RAISE IT UP 🚀", "Scared money don't make money!", "Let's gooo 🔥"],
                   "all_in": ["ALL IN BABY 🚀🚀", "Send it!"],
                   "call": ["I'm in.", "You don't scare me."], "check": ["Fine, check."],
                   "fold": ["Ugh. Next hand.", "Saving it for later 😤"]},
    "conservative": {"raise": ["This one's worth it.", "Calculated."], "all_in": ["I'm certain."],
                     "call": ["I'll see it.", "Reasonable."], "check": ["Check.", "No need."],
                     "fold": ["Not worth it.", "Patience.", "I'll wait."]},
    "statistician": {"raise": ["The math says raise.", "+EV. Raising."], "all_in": ["Equity says shove."],
                     "call": ["The math says call.", "Pot odds check out."], "check": ["Zero EV to bet. Check."],
                     "fold": ["Negative EV. Fold.", "The numbers don't lie."]},
    "manipulator": {"raise": ["I've been watching you 👀", "I know what you're doing.", "Your move 😏"],
                    "all_in": ["I see right through you."], "call": ["Interesting...", "Go on."],
                    "check": ["Your turn, friend."], "fold": ["Let them think they won."]},
    "wildcard": {"raise": ["Vibes only.", "Why not? 🎲", "Chaos mode."], "all_in": ["YOLO 🎲🎲"],
                 "call": ["Sure!", "Feeling lucky."], "check": ["Meh."],
                 "fold": ["Bored now.", "Nah."]},
    "llm_bot": {"raise": ["Let me think step by step... raise.", "As an AI poker player, I raise."],
                "all_in": ["After careful consideration: all-in."], "call": ["I'll call. Still thinking..."],
                "check": ["Check. (that took a while)"], "fold": ["I'll fold, as a cautious assistant."]},
    "rule_based": {"raise": ["IF strong THEN raise."], "all_in": ["RULE 7: shove."],
                   "call": ["Call."], "check": ["Check."], "fold": ["Fold."]},
}


def bubble_for(personality, action, to_call, rng):
    kind = {"fold": "fold", "all_in": "all_in"}.get(action)
    if kind is None:
        kind = "raise" if action in RAISE_ACTIONS else ("check" if to_call == 0 else "call")
    pool = BUBBLES.get(personality, BUBBLES["statistician"]).get(kind, ["..."])
    return rng.choice(pool)


# ---------------- equity ----------------
def win_chance(hole, board, n_opp, runs=MC_RUNS, rng=random):
    """Monte Carlo win probability vs n_opp random hands (ties count half)."""
    if n_opp <= 0:
        return 1.0
    known = set(hole) | set(board)
    deck = [c for c in FULL_DECK if c not in known]
    need = 5 - len(board)
    wins = 0.0
    for _ in range(runs):
        draw = rng.sample(deck, need + 2 * n_opp)
        b = board + draw[:need]
        me = EVAL.evaluate(b, hole)
        best_opp = min(EVAL.evaluate(b, draw[need + 2 * i: need + 2 * i + 2]) for i in range(n_opp))
        if me < best_opp:
            wins += 1
        elif me == best_opp:
            wins += 0.5
    return wins / runs



# ---------------- hand reading ----------------
RANKS = "23456789TJQKA"
RANK_NAME = {"2": "Twos", "3": "Threes", "4": "Fours", "5": "Fives", "6": "Sixes", "7": "Sevens",
             "8": "Eights", "9": "Nines", "T": "Tens", "J": "Jacks", "Q": "Queens", "K": "Kings", "A": "Aces"}


def preflop_tier(hole):
    """Rough starting-hand tier: premium / strong / playable / speculative / weak."""
    (r1, s1), (r2, s2) = sorted(hole, key=lambda c: RANKS.index(c[0]), reverse=True)
    i1, i2 = RANKS.index(r1), RANKS.index(r2)
    suited, pair, gap = s1 == s2, r1 == r2, i1 - i2
    if pair:
        desc = f"pocket pair ({RANK_NAME[r1]})"
        tier = "premium" if i1 >= 10 else "strong" if i1 >= 7 else "playable"
    else:
        desc = f"{r1}{r2} {'suited' if suited else 'offsuit'}"
        if i1 == 12 and i2 >= 10 or (i1 == 11 and i2 == 10 and suited):
            tier = "premium" if i2 >= 11 else "strong"
        elif i1 >= 10 and i2 >= 8:
            tier = "strong" if suited or i2 >= 9 else "playable"
        elif (suited and gap <= 2 and i2 >= 3) or (i1 == 12 and suited):
            tier = "speculative"
        elif i1 >= 9 and gap <= 3:
            tier = "playable" if suited else "speculative"
        else:
            tier = "weak"
    return f"{desc}, {tier} starting hand", tier


def draws(hole, board):
    if len(board) < 3 or len(board) >= 5:
        return []
    cards = hole + board
    out = []
    for s in "shdc":
        if sum(c[1] == s for c in cards) == 4 and any(c[1] == s for c in hole):
            out.append("flush draw")
    vals = {RANKS.index(c[0]) + 2 for c in cards}
    if 14 in vals:
        vals.add(1)
    best = 0
    for lo in range(1, 11):
        n = sum(v in vals for v in range(lo, lo + 5))
        best = max(best, n)
    if best == 4:
        # open-ended if 4 in a row
        oe = any(all(v in vals for v in range(lo, lo + 4)) for lo in range(2, 11))
        out.append("open-ended straight draw" if oe else "gutshot straight draw")
    return out


def hand_reading(hole_s, board_s):
    """Human-readable read of my hand, e.g. 'Two Pair; flush draw'."""
    if not board_s:
        return preflop_tier(hole_s)[0]
    hole = [Card.new(c) for c in hole_s]
    board = [Card.new(c) for c in board_s]
    cls = EVAL.get_rank_class(EVAL.evaluate(board, hole))
    made = EVAL.class_to_string(cls)
    notes = []
    if cls == 8:  # one pair: is it top pair / overpair / board pair?
        brank = sorted((RANKS.index(c[0]) for c in board_s), reverse=True)
        hr = [RANKS.index(c[0]) for c in hole_s]
        if hr[0] == hr[1]:
            notes.append("overpair" if hr[0] > brank[0] else "underpair")
        elif brank[0] in hr:
            notes.append("top pair")
        elif any(h in brank for h in hr):
            notes.append("middle/bottom pair")
        else:
            notes.append("pair is on the board (I only have high card)")
    notes += draws(hole_s, board_s)
    return made + (" (" + ", ".join(notes) + ")" if notes else "")

# ---------------- context / state text ----------------
def build_context(game, p, runs=MC_RUNS):
    opps = [o for o in game.in_hand() if o is not p]
    eq = win_chance(p.hole, game.board, len(opps), runs=runs, rng=game.rng)
    tc = game.to_call(p)
    pot = game.pot
    pot_odds = tc / (pot + tc) if tc > 0 else 0.0
    legal = game.legal(p)
    opp_info = []
    for o in game.players:
        if o is p or o.out:
            continue
        s = game.tracker.summary(o.seat)
        s.update(seat=o.seat, name=o.name, chips=o.chips, in_hand=not o.folded,
                 bet=o.bet, all_in=o.all_in)
        opp_info.append(s)
    # players still to act after me this round
    after = 0
    j = p.seat
    for _ in range(len(game.players) - 1):
        j = (j + 1) % len(game.players)
        q = game.players[j]
        if q.can_act and not (q.acted and q.bet == game.current_bet):
            after += 1
    eff_stack = min([p.chips] + [o.chips for o in opps]) if opps else p.chips
    spr = eff_stack / pot if pot else 0
    return {
        "hand_read": hand_reading(cards_str(p.hole), cards_str(game.board)),
        "acting_after": after, "spr": spr,
        "seat": p.seat, "name": p.name, "personality": p.personality,
        "hole": cards_str(p.hole), "board": cards_str(game.board), "street": game.street,
        "equity": eq, "pot": pot, "to_call": tc, "pot_odds": pot_odds,
        "chips": p.chips, "current_bet": game.current_bet, "raises": game.raises,
        "legal": legal, "opponents": opp_info, "n_in_hand": len(opps) + 1,
    }


def state_text(ctx, emphasize_history=False):
    lines = [
        "Game: No-Limit Texas Hold'em, blinds 10/20, max 2 raises per round.",
        f"Personality: {ctx['personality']}.",
        f"Hole cards: {' '.join(ctx['hole'])}. Board: {' '.join(ctx['board']) or '(none)'} ({ctx['street']}).",
        f"My hand: {ctx['hand_read']}.",
        f"Win chance vs {ctx['n_in_hand'] - 1} opponent(s): {ctx['equity']:.0%}"
        f" (break-even win chance to call: {ctx['pot_odds']:.0%}).",
        f"Position: {ctx['acting_after']} player(s) still to act after me"
        f"{' (I act last: good position)' if ctx['acting_after'] == 0 else ''}. "
        f"Stack-to-pot ratio: {ctx['spr']:.1f}.",
        f"Pot: {ctx['pot']}. To call: {ctx['to_call']}. Pot odds: {ctx['pot_odds']:.0%}.",
        f"My chips: {ctx['chips']}. Current bet: {ctx['current_bet']}. Raises this round: {ctx['raises']}/2.",
        f"Legal actions: {', '.join(ctx['legal'])}.",
        "Opponents:",
    ]
    for o in ctx["opponents"]:
        status = "in hand" if o["in_hand"] else "folded"
        lines.append(
            f"- {o['name']} (chips {o['chips']}, {status}, bet {o['bet']}): last actions "
            f"[{', '.join(o['recent']) or 'none'}]; raised {o['recent_raises']}/{len(o['recent'])} recent; "
            f"folds {o['fold_rate']:.0%} of the time; style: {o['style']}; "
            f"caught bluffing {o['bluffs']}/{o['showdowns']} showdowns."
        )
    if emphasize_history:
        lines.append("FOCUS: Opponent tendencies matter most. Exploit players who bluff or raise too often.")
    return "\n".join(lines)


# ---------------- Drex call ----------------
def drex_configured():
    return bool(DREX_URL and DREX_API_KEY)


# ---------- daily spending caps (so a public link can't burn through API credits) ----------
class DailyCap:
    """Counts paid API calls per UTC day. Once `limit` is reached, take() returns False until midnight UTC
    and callers fall back to the local brain (labelled "fallback" in the UI)."""

    def __init__(self, limit):
        self.limit, self.day, self.used = limit, None, 0
        self._lock = threading.Lock()

    def take(self):
        with self._lock:
            today = time.strftime("%Y-%m-%d", time.gmtime())
            if today != self.day:
                self.day, self.used = today, 0
            if self.limit > 0 and self.used >= self.limit:
                return False
            self.used += 1
            return True

    def status(self):
        with self._lock:
            used = self.used if self.day == time.strftime("%Y-%m-%d", time.gmtime()) else 0
            return {"used_today": used, "daily_cap": self.limit}


class BudgetExceeded(RuntimeError):
    pass


DREX_BUDGET = DailyCap(int(os.getenv("DREX_DAILY_CAP", "5000")))  # 0 = no cap
LLM_BUDGET = DailyCap(int(os.getenv("LLM_DAILY_CAP", "300")))


ACTION_DESCRIPTIONS = {
    "fold": "Fold: give up the hand and lose what is already in the pot",
    "check_call": "Check if nothing to call, otherwise call the current bet",
    "raise_half_pot": "Raise by half the pot",
    "raise_pot": "Raise by the full pot",
    "all_in": "Push all remaining chips in",
}


PERSONA_INSTR = {
    "risk_taker": "You are an aggressive, fearless poker player. Which action should you take right now?",
    "conservative": "You are a tight, careful poker player who avoids risk. Which action should you take right now?",
    "statistician": "You are a purely mathematical poker player who maximizes expected value. Which action should you take right now?",
    "manipulator": "You are a manipulative poker player who exploits opponents' habits and bluffs players who fold to pressure. Which action should you take right now?",
    "wildcard": "You are an unpredictable poker player. Which action should you take right now?",
}


def call_drex(text, options, personality=None, extra_questions=None):
    """Ask Drex one `choice` question: which poker action to take.
    Request:  {"model", "state", "questions": {"move": {"type":"choice","instructions","criteria":{opt: desc}}}}
    Response: {"answers": {"move": {"choice", "probabilities": {opt: p}}}, "usage": {"input_tokens"}}
    """
    if not DREX_BUDGET.take():
        raise BudgetExceeded("daily Drex call cap reached; using the local brain until tomorrow (UTC)")
    body = {
        "model": DREX_MODEL,
        "state": text,
        "questions": {
            "move": {
                "type": "choice",
                "instructions": PERSONA_INSTR.get(personality, "You are this poker player. Which action should you take right now?"),
                "criteria": {a: ACTION_DESCRIPTIONS[a] for a in options},
            },
            **(extra_questions or {}),
        },
    }
    resp = requests.post(
        DREX_URL,
        headers={"Authorization": f"Bearer {DREX_API_KEY}", "Content-Type": "application/json"},
        json=body,
        timeout=DREX_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    _TL.usage = (data.get("usage") or {}).get("input_tokens", 0)
    answers = data.get("answers") or {}
    _TL.extra = {}
    for qk in (extra_questions or {}):
        _TL.extra[qk] = (answers.get(qk) or {}).get("probabilities") or {}
    ans = answers.get("move") or {}
    probs = ans.get("probabilities") or {}
    if not probs and ans.get("choice"):
        probs = {ans["choice"]: 1.0}
    return {a: float(probs.get(a, 0.0)) if a in options else 0.0 for a in ACTIONS}


def local_brain(ctx, exploit_weight=0.2):
    """Heuristic stand-in for Drex: softmax over simple action scores."""
    eq, po = ctx["equity"], ctx["pot_odds"]
    n = ctx["n_in_hand"]
    strength = min(1.0, eq * n / 2)  # equity relative to a fair share
    if not ctx["board"]:  # preflop: lean on starting-hand tiers
        tier = preflop_tier(ctx["hole"])[1]
        strength = min(1.0, max(0.0, strength + {"premium": 0.25, "strong": 0.12, "playable": 0.03,
                                                 "speculative": 0.0, "weak": -0.1}[tier]))
    if ctx.get("acting_after") == 0:
        strength = min(1.0, strength + 0.05)  # position bonus
    free = ctx["to_call"] == 0
    # how aggressive are the people still in the hand?
    aggr = max([o["raise_rate"] for o in ctx["opponents"] if o["in_hand"]] or [0])
    bluffy = max([o["bluffs"] / o["showdowns"] for o in ctx["opponents"]
                  if o["in_hand"] and o["showdowns"]] or [0])
    exploit = exploit_weight * 3.0 * max(0.0, max(aggr, bluffy) - 0.3)
    exploit *= 2.0 / max(2, n)  # bluffing into many players is a bad idea
    s = {
        "fold": 6 * (po - eq) + 0.4 * (1 - strength) - (3 if free else 0) - exploit,
        "check_call": 0.3 + 2.5 * (eq - po),
        "raise_half_pot": 4 * (strength - 0.55) + 0.6 * exploit,
        "raise_pot": 5 * (strength - 0.65) + 1.6 * exploit,
        "all_in": 9 * (strength - 0.85) + 0.5 * exploit,
    }
    s = {a: v for a, v in s.items() if a in ctx["legal"]}  # Drex only sees legal labels
    m = max(s.values())
    e = {a: math.exp(v - m) for a, v in s.items()}
    z = sum(e.values())
    return {a: e.get(a, 0.0) / z for a in ACTIONS}


def target_question(target_name):
    return {"target_folds": {
        "type": "choice",
        "instructions": f"If I make a big pot-sized raise right now, how will {target_name} most likely respond?",
        "criteria": {"fold": f"{target_name} folds: they were bluffing or are weak",
                     "fight": f"{target_name} calls or re-raises"},
    }}


def bluff_question(name):
    return {"opp_bluffing": {
        "type": "choice",
        "instructions": f"{name} just bet into me. Given their history, what are they most likely holding?",
        "criteria": {"bluffing": f"{name} is bluffing with a weak hand",
                     "strong": f"{name} has a real, strong hand"},
    }}


def local_bluff_estimate(opp):
    bl = opp["bluffs"] / opp["showdowns"] if opp["showdowns"] else 0.3
    return min(0.85, 0.1 + 0.6 * bl + 0.25 * opp["raise_rate"])


def local_fold_estimate(opp):
    """Fallback guess for 'will this opponent fold to pressure' from their history."""
    bl = opp["bluffs"] / opp["showdowns"] if opp["showdowns"] else 0.3
    return min(0.85, 0.1 + 0.2 * bl + 0.55 * opp.get("fold_rate", 0))


def get_probs(ctx, emphasize_history=False, force_local=False, target=None, mode="fold"):
    """Returns (probs over all 5 actions, source, ms, extras)."""
    t0 = time.perf_counter()
    extras = {}
    if drex_configured() and not force_local:
        try:
            probs = call_drex(state_text(ctx, emphasize_history), list(ctx["legal"]),
                              personality=ctx["personality"],
                              extra_questions=(target_question(target["name"]) if mode == "fold"
                                               else bluff_question(target["name"])) if target else None)
            if sum(probs.values()) > 0:
                ex = getattr(_TL, "extra", {})
                if target and ex.get("target_folds"):
                    extras["target_fold_p"] = float(ex["target_folds"].get("fold", 0))
                if target and ex.get("opp_bluffing"):
                    extras["bluff_p"] = float(ex["opp_bluffing"].get("bluffing", 0))
                return probs, "drex", (time.perf_counter() - t0) * 1000, extras
        except Exception as e:  # timeout / bad response -> fallback
            ctx["drex_error"] = str(e)[:120]
        source = "fallback"
    else:
        source = "local"
    probs = local_brain(ctx, 1.2 if emphasize_history else 0.2)
    if target:
        if mode == "fold":
            extras["target_fold_p"] = local_fold_estimate(target)
        else:
            extras["bluff_p"] = local_bluff_estimate(target)
    return probs, source, (time.perf_counter() - t0) * 1000, extras


def mask(probs, legal, allow_free_fold=False, to_call=0):
    ok = set(legal)
    if to_call == 0 and not allow_free_fold:
        ok.discard("fold")
    m = {a: (probs.get(a, 0) if a in ok else 0.0) for a in ACTIONS}
    z = sum(m.values())
    if z <= 0:
        m = {a: (1.0 if a == "check_call" else 0.0) for a in ACTIONS}
        z = 1.0
    return {a: v / z for a, v in m.items()}



# ---------------- LLM "Giant" ----------------
def llm_configured():
    return bool(ANTHROPIC_API_KEY or OPENAI_API_KEY)


def call_llm(text, legal):
    """Ask a frontier LLM for the move. Returns (raw_text, in_tokens, out_tokens)."""
    if not LLM_BUDGET.take():
        raise BudgetExceeded("daily LLM call cap reached")
    prompt = (text + "\n\nChoose your action. Reply with EXACTLY one of these words and nothing else: "
              + ", ".join(legal))
    if ANTHROPIC_API_KEY:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=LLM_TIMEOUT, headers={
            "x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": LLM_MODEL, "max_tokens": 20, "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        d = r.json()
        txt = "".join(b.get("text", "") for b in d.get("content", []))
        u = d.get("usage", {})
        return txt, u.get("input_tokens", 0), u.get("output_tokens", 0)
    r = requests.post("https://api.openai.com/v1/chat/completions", timeout=LLM_TIMEOUT, headers={
        "Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={"model": LLM_MODEL, "max_tokens": 20, "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    return d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def parse_llm_action(txt):
    t = (txt or "").strip().lower().replace("-", "_").replace(" ", "_")
    for a in sorted(ACTIONS, key=len, reverse=True):  # longest first (raise_half_pot before raise_pot)
        if a in t:
            return a
    for word, a in (("fold", "fold"), ("call", "check_call"), ("check", "check_call"), ("all_in", "all_in"),
                    ("allin", "all_in"), ("half", "raise_half_pot"), ("raise", "raise_pot")):
        if word in t:
            return a
    return None


def is_mistake(action, ctx):
    """'Math mistake' = fold when calling is clearly +EV, or call/raise far below break-even."""
    eq, po = ctx["equity"], ctx["pot_odds"]
    if action == "fold" and ctx["to_call"] > 0 and eq >= po + 0.10:
        return True
    if action == "fold" and ctx["to_call"] == 0:
        return True  # folding for free
    if action == "check_call" and ctx["to_call"] > 0 and eq < po - 0.15:
        return True
    return False


# ---------------- 🎙️ Drex Commentator + 🧠 player read ----------------
PLAY_TYPES = {
    "bluff": "BLUFF: bet or raised with a weak hand to make others fold",
    "value_bet": "VALUE BET: bet a strong hand to get paid",
    "hero_call": "HERO CALL: called a big bet with a mediocre hand, suspecting a bluff",
    "smart_fold": "SMART FOLD: folded a hand that was likely beaten",
    "trap": "TRAP: checked or just called with a strong hand to lure others in",
    "blunder": "BLUNDER: a clearly bad decision",
}
PLAY_EMOJI = {"bluff": "🃏", "value_bet": "💰", "hero_call": "🦸", "smart_fold": "🧠", "trap": "🪤", "blunder": "🤦"}
STYLES = {
    "loose_aggressive": "Loose-aggressive: plays many hands and raises a lot",
    "tight_aggressive": "Tight-aggressive: plays few hands but raises when in",
    "calling_station": "Calling station: calls too much, rarely raises or folds",
    "tight_passive": "Tight-passive (rock): folds a lot, rarely raises",
    "balanced": "Balanced and hard to read",
}


def _moments(decisions):
    out = []
    for i, d in enumerate(decisions):
        if d["choice"] == "check_call" and d.get("to_call", 1) == 0:
            continue  # skip plain checks
        out.append((f"m{i}", d))
    return out[-8:]


def commentate(decisions, result, human=None):
    """One Drex call per hand: pick the highlight, label it, and read the human's style."""
    moments = _moments(decisions)
    if not moments:
        return None
    winners = ", ".join(result.get("winners", [])) if result else "?"
    lines = [f"Poker hand summary. Winner: {winners}. Board: {' '.join((result or {}).get('board', [])) or '(none)'}."]
    for key, d in moments:
        lines.append(f"[{key}] {d['name']} ({d['personality']}) on the {d['street']}: {d['choice']} "
                     f"holding {' '.join(d.get('hole') or [])} with {d.get('equity', 0):.0%} win chance.")
    if human:
        lines.append(f"Human player history: {human['actions']} actions, raises {human['raise_rate']:.0%}, "
                     f"folds {human['fold_rate']:.0%}, caught bluffing {human['bluffs']}/{human['showdowns']} showdowns.")
    text = "\n".join(lines)
    criteria = {k: f"{d['name']} {d['choice']} on the {d['street']} ({d.get('equity', 0):.0%} to win)" for k, d in moments}
    t0 = time.perf_counter()
    source = "local"
    try:
        if not drex_configured():
            raise RuntimeError("no drex")
        if not DREX_BUDGET.take():
            raise BudgetExceeded("daily Drex call cap reached")
        questions = {
            "highlight": {"type": "choice", "instructions": "As a TV poker commentator, which moment was the highlight of this hand?",
                          "criteria": criteria if len(criteria) >= 2 else {**criteria, "none": "Nothing notable"}},
            "play_type": {"type": "choice", "instructions": "What kind of play was that highlight?", "criteria": PLAY_TYPES},
        }
        if human and human["actions"] >= 3:
            questions["human_style"] = {"type": "choice", "instructions": "How would you describe the human player's style?",
                                        "criteria": STYLES}
        resp = requests.post(DREX_URL, headers={"Authorization": f"Bearer {DREX_API_KEY}"},
                             json={"model": DREX_MODEL, "state": text, "questions": questions}, timeout=DREX_TIMEOUT + 2)
        resp.raise_for_status()
        ans = resp.json().get("answers") or {}
        hp = (ans.get("highlight") or {}).get("probabilities") or {}
        tp = (ans.get("play_type") or {}).get("probabilities") or {}
        sp = (ans.get("human_style") or {}).get("probabilities") or {}
        hk = max((k for k in hp if k in criteria), key=hp.get)
        tk = max(tp, key=tp.get)
        source = "drex"
        style = (max(sp, key=sp.get), sp[max(sp, key=sp.get)]) if sp else None
        conf = tp[tk]
    except Exception:
        # local heuristic: lowest-equity raise = bluff, else highest-equity raise = value bet
        raises = [(k, d) for k, d in moments if d["choice"] in RAISE_ACTIONS]
        calls = [(k, d) for k, d in moments if d["choice"] == "check_call"]
        if raises and min(d.get("equity", 1) for _, d in raises) < 0.35:
            hk = min(raises, key=lambda x: x[1].get("equity", 1))[0]; tk = "bluff"
        elif raises:
            hk = max(raises, key=lambda x: x[1].get("equity", 0))[0]; tk = "value_bet"
        elif calls:
            hk = min(calls, key=lambda x: x[1].get("equity", 1))[0]; tk = "hero_call"
        else:
            hk = moments[-1][0]; tk = "smart_fold"
        conf, style = None, None
        if human and human["actions"] >= 3:
            rr, fr = human["raise_rate"], human["fold_rate"]
            st = ("loose_aggressive" if rr >= 0.4 and fr < 0.35 else "tight_aggressive" if rr >= 0.3 else
                  "tight_passive" if fr >= 0.5 else "calling_station" if fr < 0.2 else "balanced")
            style = (st, None)
    d = dict(moments)[hk]
    out = {"who": d["name"], "personality": d["personality"], "street": d["street"], "choice": d["choice"],
           "equity": d.get("equity", 0), "type": tk, "label": PLAY_TYPES[tk].split(":")[0], "emoji": PLAY_EMOJI[tk],
           "confidence": conf, "source": source, "ms": round((time.perf_counter() - t0) * 1000)}
    if style:
        out["human_style"] = {"key": style[0], "label": STYLES[style[0]].split(":")[0], "p": style[1]}
    return out

# ---------------- personalities ----------------
# Default "genes" per agent. Evolution Mode mutates these.
DEFAULT_GENES = {
    "aggression": 1.0,       # multiplies Drex's raise/all-in probabilities (all agents)
    "caution": 1.0,          # multiplies Drex's fold probability (all agents)
    "raise_boost": 1.5,      # Risk-Taker: extra raise boost
    "raise_threshold": 0.6,  # Conservative: raise only if raise confidence > this
    "temperature": 0.6,      # Wildcard: <1 flattens odds (more chaos)
    "bluff_threshold": MANIP_BLUFF_THRESHOLD,  # Manipulator: bluff / hero-call when Drex's read >= this
}
GENE_RANGES = {"aggression": (0.3, 3.0), "caution": (0.3, 3.0), "raise_boost": (1.0, 4.0),
               "raise_threshold": (0.25, 0.9), "temperature": (0.2, 1.5), "bluff_threshold": (0.2, 0.95)}


def genes_of(p):
    g = dict(DEFAULT_GENES)
    g.update(getattr(p, "genes", None) or {})
    return g


def apply_genes(probs, g):
    if g["aggression"] == 1.0 and g["caution"] == 1.0:
        return probs
    m = {a: v * (g["aggression"] if a in RAISE_ACTIONS else g["caution"] if a == "fold" else 1.0)
         for a, v in probs.items()}
    z = sum(m.values()) or 1.0
    return {a: v / z for a, v in m.items()}


def pick(personality, probs, rng, g=None):
    g = g or DEFAULT_GENES
    legal_probs = {a: v for a, v in probs.items() if v > 0}
    top = max(legal_probs, key=legal_probs.get)
    if personality == "risk_taker":
        b = g["raise_boost"]
        boosted = {a: v * (b if a in RAISE_ACTIONS else 1) for a, v in legal_probs.items()}
        return max(boosted, key=boosted.get), f"Boosted raise/all-in odds x{b:.2f}, then took the top option"
    if personality == "conservative":
        th = g["raise_threshold"]
        raise_p = sum(v for a, v in legal_probs.items() if a in RAISE_ACTIONS)
        if raise_p > th:
            best = max((a for a in legal_probs if a in RAISE_ACTIONS), key=legal_probs.get)
            return best, f"Raise confidence {raise_p:.0%} > {th:.0%} so raised"
        pc, pf = legal_probs.get("check_call", 0), legal_probs.get("fold", 0)
        if pf == 0 or pc > 1.25 * pf:
            return "check_call", f"Raise confidence only {raise_p:.0%}, so call/check"
        return "fold", f"Raise confidence only {raise_p:.0%} and fold > call"
    if personality == "wildcard":
        t = g["temperature"]
        flat = {a: v ** t for a, v in legal_probs.items()}
        z = sum(flat.values())
        r, acc = rng.random() * z, 0.0
        for a, v in flat.items():
            acc += v
            if r <= acc:
                return a, f"Rolled the dice on Drex's odds (temperature {t:.2f})"
        return top, "Rolled the dice"
    if personality == "manipulator":
        return top, "Top option, with opponent history weighted heavily"
    return top, "Took Drex's top option"


def guard(personality, action, why, ctx):
    """Math guardrails on top of Drex + personality (the 'smart' layer)."""
    legal = ctx["legal"]
    eq, po = ctx["equity"], ctx["pot_odds"]
    strength = min(1.0, eq * ctx["n_in_hand"] / 2)
    if personality == "wildcard":
        return action, why  # chaos is the point
    # 1) never fold when calling is clearly profitable
    if action == "fold" and ctx["to_call"] > 0 and eq >= po + 0.05 and "check_call" in legal:
        return "check_call", f"Math guard: win chance {eq:.0%} beats pot odds {po:.0%}, so call instead of fold"
    # 2) never check/call a monster: build the pot
    if action == "check_call" and strength >= 0.85 and personality != "conservative":
        for r in ("raise_pot", "raise_half_pot", "all_in"):
            if r in legal:
                return r, f"Value guard: {ctx['hand_read']} with {eq:.0%} to win, so raise for value"
    # 3) never shove a weak hand with lots behind (except the Risk-Taker, who lives for it)
    if action == "all_in" and eq < 0.35 and ctx["spr"] > 3 and personality != "risk_taker":
        alt = "raise_half_pot" if "raise_half_pot" in legal and personality == "manipulator" else "check_call"
        if ctx["to_call"] > 0 and eq < po and alt == "check_call":
            alt = "fold"
        return alt, f"Risk guard: all-in with {eq:.0%} to win and SPR {ctx['spr']:.1f} is reckless"
    # 4) short stack: shove or fold, don't dribble chips
    if ctx["chips"] <= 10 * 20 and action in ("check_call", "raise_half_pot") and ctx["to_call"] > 0 \
            and strength >= 0.6 and "all_in" in legal:
        return "all_in", f"Short-stack guard: {ctx['chips']} chips left and a strong hand, so shove"
    return action, why


def rule_based_action(ctx):
    """Classic hand-written bot, used as the benchmark baseline.
    It does NOT check legality itself (that's the point of the benchmark)."""
    eq, po = ctx["equity"], ctx["pot_odds"]
    strength = min(1.0, eq * ctx["n_in_hand"] / 2)
    if strength > 0.85:
        return "raise_pot"
    if strength > 0.6:
        return "raise_half_pot"
    if eq >= po or ctx["to_call"] == 0:
        return "check_call"
    return "fold"


# ---------------- evidence ----------------
def evidence(ctx, emphasize):
    ev = [f"Win chance {ctx['equity']:.0%}"]
    if ctx["to_call"]:
        ev.append(f"Pot odds {ctx['pot_odds']:.0%} (call {ctx['to_call']} into {ctx['pot']})")
    opps = sorted([o for o in ctx["opponents"] if o["in_hand"]],
                  key=lambda o: (o["bluffs"], o["recent_raises"]), reverse=True)
    for o in opps[: (2 if emphasize else 1)]:
        if o["recent"]:
            s = f"{o['name']} raised {o['recent_raises']}/{len(o['recent'])} recent actions"
            if o["showdowns"]:
                s += f", bluffed {o['bluffs']}/{o['showdowns']} showdowns"
            ev.append(s)
    return ev


# ---------------- main decision fn ----------------
class Brain:
    def __init__(self, mc_runs=MC_RUNS, force_local=False):
        self.mc_runs = mc_runs
        self.force_local = force_local
        self.stats = {}  # brain_kind -> {decisions, ms, illegal, calls}
        self.rng = random.Random()

    def _stat(self, kind):
        return self.stats.setdefault(kind, {"decisions": 0, "ms": 0.0, "illegal": 0,
                                            "drex_calls": 0, "fallbacks": 0})

    def decide(self, game, p):
        ctx = build_context(game, p, runs=self.mc_runs)
        legal = ctx["legal"]
        if p.brain == "rule":
            t0 = time.perf_counter()
            raw = rule_based_action(ctx)
            ms = (time.perf_counter() - t0) * 1000
            st = self._stat("rule")
            st["decisions"] += 1
            st["ms"] += ms
            action = raw
            if raw not in legal:
                st["illegal"] += 1
                action = "check_call"
            info = {"seat": p.seat, "name": p.name, "personality": p.personality,
                    "probs": None, "choice": action, "raw_choice": raw, "illegal": raw not in legal,
                    "mistake": is_mistake(action, ctx), "source": "rules", "ms": round(ms, 2), "equity": ctx["equity"],
                    "hole": ctx["hole"], "evidence": evidence(ctx, False),
                    "why": "Hard-coded thresholds on win chance",
                    "bubble": bubble_for("rule_based", action, ctx["to_call"], self.rng)}
            return action, info

        if p.brain == "llm":
            st = self._stat("llm")
            t0 = time.perf_counter()
            raw, tin, tout, err = None, 0, 0, None
            try:
                txt, tin, tout = call_llm(state_text(ctx), list(legal))
                raw = parse_llm_action(txt)
            except Exception as e:
                err = str(e)[:120]
            ms = (time.perf_counter() - t0) * 1000
            st["decisions"] += 1
            st["ms"] += ms
            st["tokens_in"] = st.get("tokens_in", 0) + tin
            st["tokens_out"] = st.get("tokens_out", 0) + tout
            illegal = raw not in legal
            if err:
                st["errors"] = st.get("errors", 0) + 1
                st["last_error"] = err
            elif illegal:
                st["illegal"] += 1
            action = raw if not illegal else "check_call"
            info = {"seat": p.seat, "name": p.name, "personality": p.personality, "probs": None,
                    "choice": action, "raw_choice": raw, "illegal": illegal and not err,
                    "mistake": is_mistake(action, ctx), "source": "llm", "ms": round(ms, 1),
                    "equity": ctx["equity"], "hole": ctx["hole"], "evidence": [f"Hand: {ctx['hand_read']}"],
                    "why": f"{LLM_MODEL} answered '{raw}'" + (" (illegal → forced check/call)" if illegal and not err else ""),
                    "tokens_in": tin, "tokens_out": tout,
                    "bubble": bubble_for("llm_bot", action, ctx["to_call"], self.rng)}
            if err:
                info["drex_error"] = "LLM error: " + err
            return action, info

        emphasize = p.personality == "manipulator"
        target, mode = None, "fold"
        if emphasize:
            live = [o for o in ctx["opponents"] if o["in_hand"] and o["recent"]]
            # (a) a known bluffer is betting into me -> ask Drex if it's a bluff
            aggr = [o for o in live if ctx["to_call"] > 0 and o["bet"] == ctx["current_bet"] and o["bet"] > 0 and
                    (o["raise_rate"] >= 0.5 or (o["showdowns"] and o["bluffs"] / o["showdowns"] >= 0.4))]
            # (b) a tight player is in the pot -> ask Drex if they'd fold to pressure
            tight = [o for o in live if o.get("fold_rate", 0) >= 0.4]
            if aggr and ctx["current_bet"] > 20:
                target, mode = max(aggr, key=lambda o: o["raise_rate"]), "bluffcatch"
            elif tight:
                target, mode = max(tight, key=lambda o: o["fold_rate"]), "fold"
        raw_probs, source, ms, extras = get_probs(ctx, emphasize, self.force_local, target, mode)
        st = self._stat("drex")
        st["decisions"] += 1
        st["ms"] += ms
        if source == "drex":
            st["drex_calls"] += 1
            st["tokens"] = st.get("tokens", 0) + getattr(_TL, "usage", 0)
        if source == "fallback":
            st["fallbacks"] += 1
        raw_top = max(raw_probs, key=raw_probs.get)
        if raw_top not in legal:
            st["illegal"] += 1
        probs = mask(raw_probs, legal, to_call=ctx["to_call"])
        g = genes_of(p)
        probs = apply_genes(probs, g)
        action, why = pick(p.personality, probs, self.rng, g)
        action, why = guard(p.personality, action, why, ctx)
        ev = evidence(ctx, emphasize)
        ev.insert(0, f"Hand: {ctx['hand_read']}")
        bluff_p = extras.get("bluff_p")
        if target is not None and bluff_p is not None:
            who = "Drex" if source == "drex" else "Estimate"
            ev.append(f"{who}: {target['name']} is bluffing {bluff_p:.0%}")
            if bluff_p >= g["bluff_threshold"] and action == "fold" and "check_call" in legal:
                action = "check_call"
                why = f"Hero call: {who} says {target['name']} is bluffing {bluff_p:.0%}, so call instead of fold"
        fold_p = extras.get("target_fold_p")
        if target is not None and fold_p is not None:
            who = "Drex" if source == "drex" else "Estimate"
            ev.append(f"{who}: {target['name']} folds to a big raise {fold_p:.0%}")
            if fold_p >= g["bluff_threshold"] and action not in RAISE_ACTIONS and ctx["n_in_hand"] <= 4:
                for r in ("raise_pot", "raise_half_pot"):
                    if r in legal and legal[r] <= 0.2 * (p.chips + p.bet):  # only cheap bluffs
                        action = r
                        why = (f"Exploit: {who} says {target['name']} folds to pressure "
                               f"{fold_p:.0%} of the time, so raise instead of '{ALABEL.get(probs and max(probs, key=probs.get), '')}'")
                        break
        info = {"seat": p.seat, "name": p.name, "personality": p.personality, "to_call": ctx["to_call"],
                "probs": {a: round(v, 3) for a, v in probs.items()},
                "choice": action, "raw_choice": raw_top, "illegal": raw_top not in legal,
                "mistake": is_mistake(max(probs, key=probs.get), ctx), "source": source, "ms": round(ms, 2), "equity": ctx["equity"],
                "hole": ctx["hole"], "evidence": ev, "why": why,
                "bubble": bubble_for(p.personality, action, ctx["to_call"], self.rng)}
        if bluff_p is not None:
            info["target"] = target["name"]
            info["bluff_p"] = round(bluff_p, 3)
        if fold_p is not None:
            info["target"] = target["name"]
            info["target_fold_p"] = round(fold_p, 3)
        if "drex_error" in ctx:
            info["drex_error"] = ctx["drex_error"]
        return action, info

    def coach(self, game, p):
        """What Drex would do in the human's seat (statistician view)."""
        ctx = build_context(game, p, runs=self.mc_runs)
        raw, source, ms, _ = get_probs(ctx, False, self.force_local)
        probs = mask(raw, ctx["legal"], allow_free_fold=True, to_call=ctx["to_call"])
        return {"probs": {a: round(v, 3) for a, v in probs.items()},
                "top": max(probs, key=probs.get), "source": source, "ms": round(ms, 2),
                "equity": ctx["equity"], "pot_odds": ctx["pot_odds"],
                "legal": ctx["legal"], "to_call": ctx["to_call"]}
