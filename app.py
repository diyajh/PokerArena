"""Drex Poker Arena: Flask server.

Run:  python app.py   then open http://localhost:5050
"""
import json
import os
import random
import re
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, g as flask_g, jsonify, request, send_from_directory

# load .env (no extra dependency) before brain.py reads the env vars
_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env):
    for _line in open(_env):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _v = _v.strip().strip('"').strip("'")
            if _v:  # skip blank values so defaults still apply
                os.environ.setdefault(_k.strip(), _v)

from brain import commentate  # noqa: E402
from brain import (DREX_BUDGET, DREX_COST_PER_CALL, DREX_PRICE_PER_M_INPUT, LLM_BUDGET, LLM_MODEL,
                   LLM_PRICE_IN, LLM_PRICE_OUT,
                   PERSONALITIES, Brain, drex_configured, llm_configured,
                   win_chance)
from engine import ACTIONS, RAISE_ACTIONS, Game, Player, cards_str
from ndi import ndi_configured, ndi_extract_players, players_to_stats

app = Flask(__name__, static_folder="static", static_url_path="/static")

BRAIN = Brain()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR") or HERE  # where saves go (point it at a persistent disk on a host)


# ---------------- per-IP rate limits (public link protection) ----------------
RATE_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "240"))       # all API calls; an online player polls ~70/min
HEAVY_PER_HOUR = int(os.getenv("HEAVY_LIMIT_PER_HOUR", "10"))    # arena / benchmark / evolution / NDI upload
ROOMS_PER_HOUR = int(os.getenv("ROOM_CREATE_LIMIT_PER_HOUR", "20"))
HEAVY_PATHS = {"/api/simulate", "/api/benchmark", "/api/evolve", "/api/upload_history"}
_hits = {}
_hits_lock = threading.Lock()


def client_ip():
    # Behind Render's proxy the visitor's address is the first X-Forwarded-For entry.
    fwd = request.headers.get("X-Forwarded-For", "")
    return fwd.split(",")[0].strip() or request.remote_addr or "?"


@app.before_request
def rate_limit():
    if not request.path.startswith("/api/") or request.path == "/api/health":
        return None
    rules = [("all", RATE_PER_MIN, 60)]
    if request.path in HEAVY_PATHS:
        rules.append(("heavy", HEAVY_PER_HOUR, 3600))
    if request.path == "/api/room/create":
        rules.append(("rooms", ROOMS_PER_HOUR, 3600))
    ip, now = client_ip(), time.time()
    with _hits_lock:
        if len(_hits) > 20000:  # forget idle addresses
            for k in [k for k, q in _hits.items() if not q or q[-1] < now - 3600]:
                del _hits[k]
        for bucket, limit, window in rules:
            if limit <= 0:
                continue
            q = _hits.setdefault((ip, bucket), [])
            q[:] = [t for t in q if t > now - window]
            if len(q) >= limit:
                wait = int(q[0] + window - now) + 1
                msg = ("Too many requests, slow down a little." if bucket == "all" else
                       f"That's the limit for now ({limit} per hour). Try again in {wait // 60 + 1} min.")
                resp = jsonify({"error": msg})
                resp.status_code = 429
                resp.headers["Retry-After"] = str(wait)
                return resp
            q.append(now)
    return None


# ---------------- one single-player table per visitor ----------------
# Each browser gets a "visitor" cookie, so on a public link every visitor has their own game,
# saved chips and profile instead of everyone sharing one table.
class Visitor:
    def __init__(self, vid):
        self.vid = vid
        self.game = None        # the table on screen (your game, a demo hand, or bots-only)
        self.play_game = None   # your real game: chips carry over until you bust or win it all
        self.mode = "play"
        self.ndi_stats = {}     # name -> stats pulled from an uploaded hand history
        self.seen = time.time()


VISITORS = {}
_visitors_lock = threading.Lock()
MAX_VISITORS = int(os.getenv("MAX_VISITORS", "500"))
SAVE_DIR = os.path.join(DATA_DIR, "saves")
_VID_OK = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def V():
    """The visitor making this request (created on first visit)."""
    if "visitor" in flask_g:
        return flask_g.visitor
    vid = request.cookies.get("visitor", "")
    if not _VID_OK.match(vid):
        vid = secrets.token_urlsafe(18)
        flask_g.new_vid = vid
    with _visitors_lock:
        v = VISITORS.get(vid)
        if v is None:
            if len(VISITORS) >= MAX_VISITORS:  # forget the visitor idle the longest (their chips are on disk)
                del VISITORS[min(VISITORS, key=lambda k: VISITORS[k].seen)]
            v = VISITORS[vid] = Visitor(vid)
        v.seen = time.time()
    flask_g.visitor = v
    return v


def _save_path(v):
    return os.path.join(SAVE_DIR, f"{v.vid}.json")


def load_save(v):
    """{"profile": ..., "bankroll": ...} for this visitor. A first visit on this computer picks up
    the files the single-player version used before (you_profile.json, bankroll.json)."""
    try:
        return json.load(open(_save_path(v)))
    except Exception:
        pass
    legacy = {}
    for key, name in (("profile", "you_profile.json"), ("bankroll", "bankroll.json")):
        try:
            legacy[key] = json.load(open(os.path.join(HERE, name)))
        except Exception:
            pass
    return legacy


def write_save(v, **parts):
    data = load_save(v)
    data.update(parts)
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        json.dump(data, open(_save_path(v), "w"))
    except Exception:
        pass


def brain_for(personality):
    return {"rule_based": "rule", "llm_bot": "llm"}.get(personality, "drex")


def make_player(seat, personality, human=False, brain="drex"):
    if human:
        return Player(seat, "You", "human", human=True, brain="human")
    meta = PERSONALITIES[personality]
    return Player(seat, meta["name"], personality, brain=brain)


def new_game(bots, human=True, seed=None):
    players = []
    if human:
        players.append(make_player(0, None, human=True))
    for b in bots:
        players.append(make_player(len(players), b, brain=brain_for(b)))
    g = Game(players, decide_fn=BRAIN.decide, rng=random.Random(seed))
    return g


def carried_chips(g, v):
    """Chips each player walks away with (personality -> chips). Chips still in an
    unfinished pot go back to whoever put them in. Busted players rebuy at 1000."""
    if g is None:
        return load_bankroll(v)  # server restarted: pick up the chips saved on disk
    out = {}
    for p in g.players:
        chips = p.chips + (0 if g.hand_over else p.contrib)
        out[p.personality] = chips if chips > 0 else 1000
    return out


def play_over(g):
    """A play game ends only when you're broke or every bot is broke."""
    if g is None or not g.hand_over:
        return False
    humans = [p for p in g.players if p.human]
    if not humans:
        return True
    return humans[0].chips <= 0 or all(p.chips <= 0 for p in g.players if not p.human)


def play_in_progress(v):
    return v.play_game is not None and not play_over(v.play_game)


def serialize(g, v):
    over = g.hand_over
    shown = (g.result or {}).get("hands", {}) if over else {}
    players = []
    for p in g.players:
        hole = None
        if p.human or (over and str(p.seat) in shown):
            hole = cards_str(p.hole)
        players.append({
            "seat": p.seat, "name": p.name, "personality": p.personality,
            "emoji": PERSONALITIES.get(p.personality, {}).get("emoji", "🙂"),
            "chips": p.chips, "bet": p.bet, "folded": p.folded, "all_in": p.all_in,
            "out": p.out, "last_action": p.last_action, "human": p.human,
            "hole": hole, "has_cards": bool(p.hole) and not p.folded,
        })
    return {
        "mode": v.mode, "hand_no": g.hand_no, "street": g.street, "board": cards_str(g.board),
        "pot": g.pot if not over else 0, "to_act": g.to_act if not over else None,
        "dealer": g.dealer, "hand_over": over, "result": g.result, "players": players,
        "drex_live": drex_configured(),
        "game_over": play_over(g) if g is v.play_game else False,
        # your real game is still running but you're looking at another table (watch / demo)
        "can_resume": g is not v.play_game and play_in_progress(v),
        "game_winner": game_winner(g) if g is v.play_game and play_over(g) else None,
    }


def game_winner(g):
    """You if every bot is broke, otherwise the bot holding the most chips."""
    human = next((p for p in g.players if p.human), None)
    if human and human.chips > 0:
        return human.name
    return max((p for p in g.players if not p.human), key=lambda p: p.chips).name



# ---------------- 🧠 the bots remember you ----------------
def load_profile(v):
    return load_save(v).get("profile") or {"hands": 0, "actions": 0, "raises": 0, "folds": 0,
                                           "showdowns": 0, "bluffs": 0, "recent": []}


def human_summary(g, v):
    hs = [p for p in g.players if p.human]
    if not hs:
        return None
    d = g.tracker.export(hs[0].seat)
    acts = d["actions"] or 0
    prof = load_profile(v)
    return {"hands": prof.get("hands", 0), "actions": acts, "raises": d["raises"], "folds": d["folds"],
            "raise_rate": d["raises"] / acts if acts else 0, "fold_rate": d["folds"] / acts if acts else 0,
            "showdowns": d["showdowns"], "bluffs": d["bluffs"], "recent": d["recent"],
            "style": prof.get("style")}


def save_profile(g, v, style=None):
    hs = [p for p in g.players if p.human]
    if not hs or hs[0].out:
        return
    d = g.tracker.export(hs[0].seat)
    prof = load_profile(v)
    d["hands"] = prof.get("hands", 0) + 1
    d["style"] = style or prof.get("style")
    write_save(v, profile=d)


def load_bankroll(v):
    return load_save(v).get("bankroll") or {}


def save_bankroll(g, v):
    write_save(v, bankroll=carried_chips(g, v))


def after_hand(g, v):
    """Once per finished hand: Drex commentary + update your saved profile."""
    if getattr(g, "_done_hand", None) == g.hand_no:
        return
    g._done_hand = g.hand_no
    if g is v.play_game:
        save_bankroll(g, v)
    hum = human_summary(g, v)
    try:
        g.commentary = commentate(g.decisions, g.result, hum)
    except Exception:
        g.commentary = None
    style = (g.commentary or {}).get("human_style")
    save_profile(g, v, style)


def coach_for(g, p):
    """Drex's advice for your current decision. Cached per decision so the coach bars and the
    Teach me review share one Drex call."""
    key = (g.hand_no, len(g.decisions), g.street, p.seat)
    cached = getattr(g, "last_coach", None)
    if cached and cached[0] == key:
        return cached[1]
    c = BRAIN.coach(g, p)
    g.last_coach = (key, c)
    return c


def teach_note(c):
    """The part of the coach's advice the Teach me review needs."""
    return {"probs": c["probs"], "top": c["top"], "source": c["source"],
            "equity": c["equity"], "pot_odds": c["pot_odds"]} if c else None


def respond(extra=None):
    v = V()
    g = v.game
    out = {"state": serialize(g, v), "events": g.pop_events()}
    if not g.hand_over and g.to_act is not None and g.players[g.to_act].human:
        out["coach"] = coach_for(g, g.players[g.to_act])
    if g.hand_over:
        after_hand(g, v)
        out["commentary"] = getattr(g, "commentary", None)
        out["peek"] = {
            "decisions": g.decisions,
            "holes": {p.name: cards_str(p.hole) for p in g.players if p.hole},
            "timeline": g.hand_log, "hand_no": g.hand_no,
        }
    if any(p.human for p in g.players):
        out["profile"] = human_summary(g, v)
    if extra:
        out.update(extra)
    return jsonify(out)


@app.after_request
def no_cache(resp):
    # always serve the latest UI (VS Code's browser caches aggressively)
    if request.path == "/" or request.path.startswith("/static/index"):
        resp.headers["Cache-Control"] = "no-store, max-age=0"
    if "new_vid" in flask_g:  # first visit: remember this browser
        resp.set_cookie("visitor", flask_g.new_vid, max_age=365 * 24 * 3600, httponly=True, samesite="Lax",
                        secure=request.is_secure or request.headers.get("X-Forwarded-Proto") == "https")
    return resp


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/health")
def api_health():
    """For the host's health check. Shows today's paid-call usage, never keys."""
    return jsonify({"ok": True, "drex_live": drex_configured(), "drex": DREX_BUDGET.status(),
                    "llm": LLM_BUDGET.status(), "rooms": len(rooms.ROOMS)})


@app.get("/api/personalities")
def personalities():
    return jsonify({"personalities": PERSONALITIES, "drex_live": drex_configured(),
                    "llm_live": llm_configured(), "llm_model": LLM_MODEL, "ndi_live": ndi_configured()})


@app.post("/api/new_game")
def api_new_game():
    v = V()
    body = request.get_json(silent=True) or {}
    v.mode = body.get("mode", "play")
    if v.mode == "watch":
        bots = body.get("bots") or ["risk_taker", "conservative", "statistician", "manipulator", "wildcard"]
        v.game = new_game([b for b in bots if b in PERSONALITIES][:6], human=False)
    elif play_in_progress(v) and not body.get("restart"):
        # your game isn't finished: go back to it with your chips instead of resetting to 1000
        v.game = v.play_game
        return respond({"resumed": True})
    else:
        bots = [b for b in (body.get("bots") or ["risk_taker", "manipulator", "statistician"]) if b in PERSONALITIES]
        carry = carried_chips(v.play_game, v)  # keep what you won or lost
        v.game = v.play_game = new_game(bots[:5] or ["statistician"], human=True)
        for p in v.game.players:
            p.chips = carry.get(p.personality, 1000)
    apply_ndi(v.game, v.ndi_stats)
    if v.mode != "watch":
        v.game.tracker.load(0, load_profile(v))  # the bots remember you
    v.game.start_hand()
    v.game.run_bots()
    return respond()


@app.post("/api/reset_profile")
def api_reset_profile():
    v = V()
    write_save(v, profile=None)
    if v.game is not None and any(p.human for p in v.game.players):
        v.game.tracker.load(0, load_profile(v))
    return jsonify({"ok": True})


@app.post("/api/next_hand")
def api_next_hand():
    v = V()
    if v.game is None:
        return jsonify({"error": "no game"}), 400
    if not v.game.hand_over:
        return respond()
    if v.game is v.play_game and play_over(v.game):
        return respond({"game_over": True})
    if not v.game.start_hand():
        return respond({"game_over": True})
    v.game.run_bots()
    return respond()


@app.post("/api/action")
def api_action():
    g = V().game
    body = request.get_json(silent=True) or {}
    action = body.get("action")
    if g is None or g.hand_over:
        return jsonify({"error": "no active hand"}), 400
    p = g.players[g.to_act]
    if not p.human:
        return jsonify({"error": "not your turn"}), 400
    if action not in g.legal(p):
        return jsonify({"error": f"illegal action {action}"}), 400
    coach = coach_for(g, p)  # same advice the coach bars showed you (cached, no extra Drex call)
    eq = win_chance(p.hole, g.board, len(g.in_hand()) - 1, runs=150, rng=g.rng)
    if action in RAISE_ACTIONS:  # remember your equity so bots can catch YOUR bluffs
        g.raised_equities.setdefault(p.seat, []).append(eq)
    g.decisions.append({"seat": p.seat, "name": "You", "personality": "human", "street": g.street,
                        "choice": action, "to_call": g.to_call(p), "equity": eq, "hole": cards_str(p.hole),
                        "probs": None, "source": "human", "ms": 0, "why": "Your move", "evidence": [],
                        "coach": teach_note(coach)})
    g.apply(p, action)
    g.run_bots()
    return respond()


@app.get("/api/state")
def api_state():
    if V().game is None:
        return jsonify({"state": None})
    return respond()


@app.post("/api/demo_hand")
def api_demo_hand():
    """Rigged DEAL (not rigged decisions): the Manipulator gets 8-3 offsuit and the
    Conservative has a history of folding. Drex decides whether to bluff and whether to fold."""
    v = V()
    v.mode = "play"
    v.game = g = new_game(["conservative", "manipulator", "statistician"], human=True, seed=7)
    con = next(p for p in g.players if p.personality == "conservative")
    # the Conservative has folded 5 of its last 6 actions -> a tight player who folds to pressure
    g.tracker.seed(con.seat, ["fold", "fold", "check_call", "fold", "fold", "fold"], showdowns=1, bluffs=0)
    apply_ndi(g, v.ndi_stats)
    g.dealer = 2  # rotates to seat 3 -> You SB, Conservative BB, Manipulator first to act
    rig = {"holes": {0: ["Qh", "4c"], 1: ["7s", "2h"], 2: ["8c", "3d"], 3: ["9d", "5c"]},
           "board": ["Ks", "Jd", "6h", "Tc", "2c"]}
    g.start_hand(rig=rig)
    g.run_bots()
    return respond({"demo": True})


# ---------------- Arena: simulate & benchmark ----------------
def _run_table(bots, hands, mc_runs, ndi=None):
    """One table playing `hands` hands. Busted bots rebuy 1000 (counted as -1000)."""
    brain = Brain(mc_runs=mc_runs)
    players = []
    for i, b in enumerate(bots):
        if isinstance(b, dict):  # evolution agent: {personality, name, genes}
            p = make_player(i, b["personality"], brain=brain_for(b["personality"]))
            p.name, p.genes = b["name"], b["genes"]
        else:
            p = make_player(i, b, brain=brain_for(b))
        players.append(p)
    for i, p in enumerate(players):  # unique names if duplicate personalities
        dup = sum(1 for q in players[:i] if q.name == p.name)
        if dup:
            p.name = f"{p.name} #{dup + 1}"
    g = Game(players, decide_fn=brain.decide, rng=random.Random())
    apply_ndi(g, ndi)
    rebuys = [0] * len(players)
    wins = [0] * len(players)
    per = [{"decisions": 0, "ms": 0.0, "illegal": 0, "mistakes": 0} for _ in players]
    for _ in range(hands):
        for p in players:
            if p.chips <= 0:
                p.chips = 1000
                rebuys[p.seat] += 1
        if not g.start_hand():
            break
        g.run_bots()
        for d in g.decisions:
            s = per[d["seat"]]
            s["decisions"] += 1
            s["ms"] += d["ms"]
            s["illegal"] += int(d.get("illegal", False))
            s["mistakes"] += int(d.get("mistake", False))
        for s in (g.result or {}).get("won", {}):
            wins[int(s)] += 1
        g.pop_events()
    nets = [p.chips - 1000 - 1000 * rebuys[p.seat] for p in players]
    return players, nets, rebuys, wins, per, brain.stats


def run_sim(bots, hands, mc_runs=120, tables=None, ndi=None):
    """Split `hands` across parallel tables (Drex calls are network-bound), then merge."""
    if tables is None:
        tables = max(1, min(int(os.getenv("ARENA_TABLES", "6")), hands))
    base, extra = divmod(hands, tables)
    chunks = [base + (1 if i < extra else 0) for i in range(tables)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=tables) as ex:
        results = list(ex.map(lambda h: _run_table(bots, h, mc_runs, ndi), [c for c in chunks if c > 0]))
    total_s = time.perf_counter() - t0
    players = results[0][0]
    n = len(players)
    net = [sum(r[1][i] for r in results) for i in range(n)]
    reb = [sum(r[2][i] for r in results) for i in range(n)]
    win = [sum(r[3][i] for r in results) for i in range(n)]
    per = [{k: sum(r[4][i][k] for r in results) for k in ("decisions", "ms", "illegal", "mistakes")} for i in range(n)]
    stats = {}
    for r in results:
        for kind, st in r[5].items():
            agg = stats.setdefault(kind, {})
            for k, v in st.items():
                agg[k] = v if isinstance(v, str) else agg.get(k, 0) + v
    rows = []
    for i, p in enumerate(players):
        st = per[i]
        rows.append({
            "name": p.name, "personality": p.personality,
            "emoji": PERSONALITIES[p.personality]["emoji"], "brain": p.brain, "seat": i,
            "net": net[i], "rebuys": reb[i], "hands_won": win[i],
            "win_rate": win[i] / hands if hands else 0,
            "decisions": st["decisions"],
            "avg_ms": st["ms"] / st["decisions"] if st["decisions"] else 0,
            "illegal": st["illegal"], "mistakes": st["mistakes"],
        })
    rows.sort(key=lambda r: r["net"], reverse=True)
    return rows, stats, total_s


def save_result(kind, payload):
    payload = dict(payload, saved_at=time.strftime("%H:%M"))
    try:
        json.dump(payload, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"last_{kind}.json"), "w"))
    except Exception:
        pass
    return payload


@app.get("/api/last_results")
def api_last_results():
    out = {}
    for kind in ("arena", "benchmark", "evolution"):
        fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"last_{kind}.json")
        if os.path.exists(fp):
            try:
                out[kind] = json.load(open(fp))
            except Exception:
                pass
    return jsonify(out)


@app.post("/api/simulate")
def api_simulate():
    body = request.get_json(silent=True) or {}
    hands = max(1, min(int(body.get("hands", 100)), 1000))
    bots = body.get("bots") or ["risk_taker", "conservative", "statistician", "manipulator", "wildcard"]
    rows, stats, secs = run_sim(bots, hands, ndi=V().ndi_stats)
    return jsonify(save_result("arena", {"leaderboard": rows, "hands": hands, "seconds": round(secs, 2),
                                         "drex_live": drex_configured()}))


@app.post("/api/benchmark")
def api_benchmark():
    """Drex vs rule-based (vs LLM if a key is set), same table, same game state."""
    body = request.get_json(silent=True) or {}
    hands = max(1, min(int(body.get("hands", 100)), 1000))
    with_llm = llm_configured() and body.get("llm", True)
    seats = ["statistician", "rule_based", "llm_bot"] if with_llm else \
        ["statistician", "rule_based", "statistician", "rule_based"]
    rows, stats, secs = run_sim(seats, hands, ndi=V().ndi_stats)
    out = {}
    kinds = [("drex", "Drex bot"), ("rule", "Rule-based bot")] + ([("llm", f"LLM bot ({LLM_MODEL})")] if with_llm else [])
    for kind, label in kinds:
        rs = [r for r in rows if r["brain"] == kind]
        dec = sum(r["decisions"] for r in rs)
        out[kind] = {
            "label": label,
            "decisions": dec,
            "avg_ms": sum(r["avg_ms"] * r["decisions"] for r in rs) / dec if dec else 0,
            "illegal": sum(r["illegal"] for r in rs),
            "illegal_pct": (sum(r["illegal"] for r in rs) / dec) if dec else 0,
            "mistakes": sum(r["mistakes"] for r in rs),
            "mistake_pct": (sum(r["mistakes"] for r in rs) / dec) if dec else 0,
            "net": sum(r["net"] for r in rs),
            "win_rate": sum(r["hands_won"] for r in rs) / (hands * len(rs)) if rs else 0,
            "cost_per_1000": (DREX_COST_PER_CALL * 1000) if kind == "drex" else 0.0,
        }
    st = stats.get("drex", {})
    calls = st.get("drex_calls", 0)
    if calls:  # real cost from Drex's reported token usage
        avg_tokens = st.get("tokens", 0) / calls
        out["drex"]["cost_per_1000"] = avg_tokens * DREX_PRICE_PER_M_INPUT / 1e6 * 1000
        out["drex"]["avg_tokens"] = round(avg_tokens)
    out["drex"]["live_calls"] = calls
    out["drex"]["fallbacks"] = st.get("fallbacks", 0)
    if with_llm:
        ls = stats.get("llm", {})
        n = max(1, ls.get("decisions", 0))
        out["llm"]["cost_per_1000"] = (ls.get("tokens_in", 0) * LLM_PRICE_IN +
                                       ls.get("tokens_out", 0) * LLM_PRICE_OUT) / 1e6 / n * 1000
        out["llm"]["errors"] = ls.get("errors", 0)
        out["llm"]["last_error"] = ls.get("last_error")
        out["llm"]["source"] = "ANTHROPIC_API_KEY" if os.getenv("ANTHROPIC_API_KEY") else "OPENAI_API_KEY"
    return jsonify(save_result("benchmark", {"benchmark": out, "hands": hands, "seconds": round(secs, 2),
                                             "drex_live": drex_configured(), "llm_live": with_llm, "table": rows}))


# ---------------- 🧬 Evolution Mode ----------------
from brain import DEFAULT_GENES, GENE_RANGES  # noqa: E402

GENE_LABEL = {"aggression": "aggression", "caution": "caution", "raise_boost": "raise boost",
              "raise_threshold": "raise bar", "temperature": "chaos temp", "bluff_threshold": "bluff trigger"}
RELEVANT = {"risk_taker": ["raise_boost", "aggression", "caution"],
            "conservative": ["raise_threshold", "aggression", "caution"],
            "statistician": ["aggression", "caution"],
            "manipulator": ["bluff_threshold", "aggression", "caution"],
            "wildcard": ["temperature", "aggression", "caution"]}


def mutate(genes, rng, strength=0.3):
    child, changes = dict(genes), []
    for k, (lo, hi) in GENE_RANGES.items():
        if rng.random() < 0.6:
            old = child[k]
            new = min(hi, max(lo, old * rng.uniform(1 - strength, 1 + strength)))
            child[k] = round(new, 3)
            if abs(new - old) / max(old, 1e-6) > 0.05:
                changes.append({"gene": k, "label": GENE_LABEL[k], "from": round(old, 2), "to": round(new, 2)})
    return child, changes


@app.post("/api/evolve")
def api_evolve():
    """Natural selection on Drex agents: each generation plays, the worst goes extinct,
    the best spawns a mutated child. Every move in every generation is a Drex call."""
    body = request.get_json(silent=True) or {}
    gens = max(2, min(int(body.get("generations", 6)), 15))
    hands = max(2, min(int(body.get("hands_per_gen", 8)), 60))
    ndi = V().ndi_stats
    rng = random.Random()
    pop = []
    for i, pers in enumerate(["risk_taker", "conservative", "statistician", "manipulator", "wildcard"]):
        pop.append({"id": f"{pers}-1", "personality": pers, "name": f"{PERSONALITIES[pers]['name']} G1",
                    "genes": dict(DEFAULT_GENES), "parent": None, "born": 1, "lineage": pers})
    history, graveyard = [], []
    t0 = time.perf_counter()
    for gen in range(1, gens + 1):
        rows, stats, secs = run_sim(pop, hands, ndi=ndi)
        by_name = {r["name"]: r for r in rows}
        ranked = sorted(pop, key=lambda ag: by_name[ag["name"]]["net"], reverse=True)
        snap = [{"id": ag["id"], "name": ag["name"], "personality": ag["personality"], "lineage": ag["lineage"],
                 "emoji": PERSONALITIES[ag["personality"]]["emoji"], "net": by_name[ag["name"]]["net"],
                 "hands_won": by_name[ag["name"]]["hands_won"], "born": ag["born"],
                 "genes": {k: round(ag["genes"][k], 2) for k in RELEVANT[ag["personality"]]}} for ag in ranked]
        entry = {"gen": gen, "ranking": snap, "seconds": round(secs, 1),
                 "drex_calls": stats.get("drex", {}).get("drex_calls", 0)}
        if gen < gens:
            loser, winner = ranked[-1], ranked[0]
            child_genes, changes = mutate(winner["genes"], rng)
            changes = [c for c in changes if c["gene"] in RELEVANT[winner["personality"]]]
            n_same = sum(1 for ag in pop if ag["lineage"] == winner["lineage"]) + \
                sum(1 for g in graveyard if g["lineage"] == winner["lineage"])
            child = {"id": f"{winner['lineage']}-{gen + 1}-{n_same + 1}", "personality": winner["personality"],
                     "name": f"{PERSONALITIES[winner['personality']]['name']} G{gen + 1}"
                             + ("" if n_same < 2 else chr(ord('a') + n_same - 1)),
                     "genes": child_genes, "parent": winner["id"], "born": gen + 1, "lineage": winner["lineage"]}
            names = {ag["name"] for ag in pop}
            while child["name"] in names:
                child["name"] += "'"
            pop = [ag for ag in pop if ag is not loser] + [child]
            graveyard.append(loser)
            entry["extinct"] = {"name": loser["name"], "emoji": PERSONALITIES[loser["personality"]]["emoji"],
                                "net": by_name[loser["name"]]["net"]}
            entry["born"] = {"name": child["name"], "parent": winner["name"],
                             "emoji": PERSONALITIES[child["personality"]]["emoji"], "mutations": changes}
        history.append(entry)
    # survivors by lineage
    census = {}
    for ag in pop:
        census[ag["lineage"]] = census.get(ag["lineage"], 0) + 1
    dominant = max(census, key=census.get)
    result = {"generations": history, "hands_per_gen": hands, "census": census,
              "dominant": {"lineage": dominant, "name": PERSONALITIES[dominant]["name"],
                           "emoji": PERSONALITIES[dominant]["emoji"], "count": census[dominant]},
              "seconds": round(time.perf_counter() - t0, 1), "drex_live": drex_configured()}
    return jsonify(save_result("evolution", result))


# ---------------- NDI hand-history upload (bonus) ----------------
def parse_hand_history(text):
    """Local fallback parser (used when NDI_API_KEY isn't set or NDI fails)."""
    counts = {}
    pat = re.compile(r"^\s*([A-Za-z][\w\- ]{0,30}?)\s*[:\-]?\s*(raises?|bets?|calls?|checks?|folds?|all[\- ]?in)\b",
                     re.I | re.M)
    for name, verb in pat.findall(text):
        name = name.strip()
        v = verb.lower()
        c = counts.setdefault(name, {"actions": 0, "raises": 0, "folds": 0})
        c["actions"] += 1
        if v.startswith(("raise", "bet", "all")):
            c["raises"] += 1
        if v.startswith("fold"):
            c["folds"] += 1
    return {n: {**c, "raise_rate": c["raises"] / c["actions"], "fold_rate": c["folds"] / c["actions"],
                "showdowns": 0, "bluffs": 0}
            for n, c in counts.items() if c["actions"] >= 2}


def apply_ndi(g, stats):
    """Feed uploaded stats into the tracker so every bot's Drex state (esp. the Manipulator's) uses them."""
    if not stats:
        return
    for p in g.players:
        st = stats.get(p.name)
        if st:
            n = 6
            raises = round(st["raise_rate"] * n)
            folds = min(n - raises, round(st.get("fold_rate", 0) * n))
            recent = ["raise_pot"] * raises + ["fold"] * folds + ["check_call"] * (n - raises - folds)
            g.tracker.seed(p.seat, recent, showdowns=st.get("showdowns", 0), bluffs=st.get("bluffs", 0))


@app.post("/api/upload_history")
def api_upload_history():
    v = V()
    f = request.files.get("file")
    raw = f.read() if f else (request.get_json(silent=True) or {}).get("text", "").encode()
    fname = f.filename if f else "hand_history.txt"
    source, job_id, error = "local parser", None, None
    t0 = time.perf_counter()
    if ndi_configured():
        try:
            players, job_id = ndi_extract_players(raw, fname)
            v.ndi_stats = players_to_stats(players)
            source = "NDI"
        except Exception as e:
            error = str(e)[:200]
            v.ndi_stats = parse_hand_history(raw.decode("utf-8", "ignore"))
    else:
        v.ndi_stats = parse_hand_history(raw.decode("utf-8", "ignore"))
    if v.game is not None:
        apply_ndi(v.game, v.ndi_stats)
    return jsonify({"stats": v.ndi_stats, "source": source, "job_id": job_id, "error": error,
                    "seconds": round(time.perf_counter() - t0, 1)})


# ---------------- 🌐 Online rooms (share a code, play together) ----------------
import rooms  # noqa: E402


def room_call(fn):
    """Run fn(room, body) under the room's lock; turn errors into a JSON message the page can show."""
    body = request.get_json(silent=True) or {}
    try:
        room = rooms.get_room(body.get("code") or request.args.get("code"))
        with room.lock:
            token = body.get("token") or request.args.get("token")
            if token not in room.seats:
                return jsonify({"error": "You're not in this room. Join it again with the code."}), 403
            return jsonify(fn(room, token, body))
    except KeyError as e:
        return jsonify({"error": e.args[0]}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/room/create")
def api_room_create():
    body = request.get_json(silent=True) or {}
    try:
        room, token = rooms.create_room(body.get("name"), body.get("bots") or ["statistician", "manipulator"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 503
    return jsonify({"code": room.code, "token": token})


@app.post("/api/room/join")
def api_room_join():
    body = request.get_json(silent=True) or {}
    try:
        room, token = rooms.join_room(body.get("code"), body.get("name"), body.get("token"))
    except KeyError as e:
        return jsonify({"error": e.args[0]}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"code": room.code, "token": token})


@app.get("/api/room/poll")
def api_room_poll():
    since = int(request.args.get("since", 0) or 0)
    return room_call(lambda room, token, body: room.view(token, since))


def host_only(room, token):
    if not room.seats[token].host:
        raise ValueError("Only the host can do that")


@app.post("/api/room/start")
def api_room_start():
    """Host: start the table, or New game later (chips carry over, anyone at 0 gets 1000)."""
    def go(room, token, body):
        host_only(room, token)
        if room.game is not None and not room.game.hand_over:
            raise ValueError("Wait for this hand to finish")
        room.start(body.get("bots"))
        return {"ok": True}
    return room_call(go)


@app.post("/api/room/next_hand")
def api_room_next_hand():
    def go(room, token, body):
        if room.game is None or not room.game.hand_over:
            return {"ok": False}
        return {"ok": room.deal()}
    return room_call(go)


@app.post("/api/room/action")
def api_room_action():
    return room_call(lambda room, token, body: room.act(token, body.get("action")) or {"ok": True})


@app.post("/api/room/rebuy")
def api_room_rebuy():
    return room_call(lambda room, token, body: room.rebuy(token) or {"ok": True})


@app.post("/api/room/react")
def api_room_react():
    def go(room, token, body):
        s = room.seats[token]
        if body.get("emoji") in rooms.REACTIONS and s.player is not None:
            room.emit(type="react", seat=s.player.seat, emoji=body["emoji"])
        return {"ok": True}
    return room_call(go)


@app.get("/api/leaderboard")
def api_leaderboard():
    """Online players ranked by chips (saved by name)."""
    return jsonify({"players": rooms.leaderboard()})


@app.post("/api/room/leave")
def api_room_leave():
    return room_call(lambda room, token, body: room.leave(token) or {"ok": True})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5050"))
    print(f"Drex live: {drex_configured()}  ->  http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
