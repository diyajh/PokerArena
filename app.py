"""Drex Poker Arena: Flask server.

Run:  python app.py   then open http://localhost:5050
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
import random
import re
import time

from flask import Flask, jsonify, request, send_from_directory

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

from brain import (DREX_COST_PER_CALL, DREX_PRICE_PER_M_INPUT, LLM_MODEL, LLM_PRICE_IN, LLM_PRICE_OUT,
                   PERSONALITIES, Brain, drex_configured, llm_configured,
                   win_chance)
from engine import ACTIONS, RAISE_ACTIONS, Game, Player, cards_str
from ndi import ndi_configured, ndi_extract_players, players_to_stats

app = Flask(__name__, static_folder="static", static_url_path="/static")

BRAIN = Brain()
GAME = None
MODE = "play"
NDI_STATS = {}  # name -> stats pulled from an uploaded hand history


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


def serialize(g):
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
        "mode": MODE, "hand_no": g.hand_no, "street": g.street, "board": cards_str(g.board),
        "pot": g.pot if not over else 0, "to_act": g.to_act if not over else None,
        "dealer": g.dealer, "hand_over": over, "result": g.result, "players": players,
        "drex_live": drex_configured(),
    }


def respond(extra=None):
    g = GAME
    out = {"state": serialize(g), "events": g.pop_events()}
    if not g.hand_over and g.to_act is not None and g.players[g.to_act].human:
        p = g.players[g.to_act]
        out["coach"] = BRAIN.coach(g, p)
    if g.hand_over:
        out["peek"] = {
            "decisions": g.decisions,
            "holes": {p.name: cards_str(p.hole) for p in g.players if p.hole},
        }
    if extra:
        out.update(extra)
    return jsonify(out)


@app.after_request
def no_cache(resp):
    # always serve the latest UI (VS Code's browser caches aggressively)
    if request.path == "/" or request.path.startswith("/static/index"):
        resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/personalities")
def personalities():
    return jsonify({"personalities": PERSONALITIES, "drex_live": drex_configured(),
                    "llm_live": llm_configured(), "llm_model": LLM_MODEL, "ndi_live": ndi_configured()})


@app.post("/api/new_game")
def api_new_game():
    global GAME, MODE
    body = request.get_json(silent=True) or {}
    MODE = body.get("mode", "play")
    if MODE == "watch":
        bots = body.get("bots") or ["risk_taker", "conservative", "statistician", "manipulator", "wildcard"]
        GAME = new_game(bots, human=False)
    else:
        bots = body.get("bots") or ["risk_taker", "manipulator", "statistician"]
        GAME = new_game(bots[:5], human=True)
    apply_ndi(GAME)
    GAME.start_hand()
    GAME.run_bots()
    return respond()


@app.post("/api/next_hand")
def api_next_hand():
    if GAME is None:
        return jsonify({"error": "no game"}), 400
    if not GAME.hand_over:
        return respond()
    if not GAME.start_hand():
        return respond({"game_over": True})
    GAME.run_bots()
    return respond()


@app.post("/api/action")
def api_action():
    g = GAME
    body = request.get_json(silent=True) or {}
    action = body.get("action")
    if g is None or g.hand_over:
        return jsonify({"error": "no active hand"}), 400
    p = g.players[g.to_act]
    if not p.human:
        return jsonify({"error": "not your turn"}), 400
    if action not in g.legal(p):
        return jsonify({"error": f"illegal action {action}"}), 400
    if action in RAISE_ACTIONS:  # remember your equity so bots can catch YOUR bluffs
        opps = len(g.in_hand()) - 1
        g.raised_equities.setdefault(p.seat, []).append(
            win_chance(p.hole, g.board, opps, runs=150, rng=g.rng))
    g.apply(p, action)
    g.run_bots()
    return respond()


@app.get("/api/state")
def api_state():
    if GAME is None:
        return jsonify({"state": None})
    return respond()


@app.post("/api/demo_hand")
def api_demo_hand():
    """Rigged DEAL (not rigged decisions): the Manipulator gets 8-3 offsuit and the
    Conservative has a history of folding. Drex decides whether to bluff and whether to fold."""
    global GAME, MODE
    MODE = "play"
    GAME = new_game(["conservative", "manipulator", "statistician"], human=True, seed=7)
    g = GAME
    con = next(p for p in g.players if p.personality == "conservative")
    # the Conservative has folded 5 of its last 6 actions -> a tight player who folds to pressure
    g.tracker.seed(con.seat, ["fold", "fold", "check_call", "fold", "fold", "fold"], showdowns=1, bluffs=0)
    apply_ndi(g)
    g.dealer = 2  # rotates to seat 3 -> You SB, Conservative BB, Manipulator first to act
    rig = {"holes": {0: ["Qh", "4c"], 1: ["7s", "2h"], 2: ["8c", "3d"], 3: ["9d", "5c"]},
           "board": ["Ks", "Jd", "6h", "Tc", "2c"]}
    g.start_hand(rig=rig)
    g.run_bots()
    return respond({"demo": True})


# ---------------- Arena: simulate & benchmark ----------------
def _run_table(bots, hands, mc_runs):
    """One table playing `hands` hands. Busted bots rebuy 1000 (counted as -1000)."""
    brain = Brain(mc_runs=mc_runs)
    players = [make_player(i, b, brain=brain_for(b)) for i, b in enumerate(bots)]
    for i, p in enumerate(players):  # unique names if duplicate personalities
        dup = sum(1 for q in players[:i] if q.name == p.name)
        if dup:
            p.name = f"{p.name} #{dup + 1}"
    g = Game(players, decide_fn=brain.decide, rng=random.Random())
    apply_ndi(g)
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


def run_sim(bots, hands, mc_runs=120, tables=None):
    """Split `hands` across parallel tables (Drex calls are network-bound), then merge."""
    if tables is None:
        tables = max(1, min(int(os.getenv("ARENA_TABLES", "6")), hands))
    base, extra = divmod(hands, tables)
    chunks = [base + (1 if i < extra else 0) for i in range(tables)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=tables) as ex:
        results = list(ex.map(lambda h: _run_table(bots, h, mc_runs), [c for c in chunks if c > 0]))
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
            "emoji": PERSONALITIES[p.personality]["emoji"], "brain": p.brain,
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
    for kind in ("arena", "benchmark"):
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
    rows, stats, secs = run_sim(bots, hands)
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
    rows, stats, secs = run_sim(seats, hands)
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


def apply_ndi(g):
    """Feed uploaded stats into the tracker so every bot's Drex state (esp. the Manipulator's) uses them."""
    if not NDI_STATS:
        return
    for p in g.players:
        st = NDI_STATS.get(p.name)
        if st:
            n = 6
            raises = round(st["raise_rate"] * n)
            folds = min(n - raises, round(st.get("fold_rate", 0) * n))
            recent = ["raise_pot"] * raises + ["fold"] * folds + ["check_call"] * (n - raises - folds)
            g.tracker.seed(p.seat, recent, showdowns=st.get("showdowns", 0), bluffs=st.get("bluffs", 0))


@app.post("/api/upload_history")
def api_upload_history():
    global NDI_STATS
    f = request.files.get("file")
    raw = f.read() if f else (request.get_json(silent=True) or {}).get("text", "").encode()
    fname = f.filename if f else "hand_history.txt"
    source, job_id, error = "local parser", None, None
    t0 = time.perf_counter()
    if ndi_configured():
        try:
            players, job_id = ndi_extract_players(raw, fname)
            NDI_STATS = players_to_stats(players)
            source = "NDI"
        except Exception as e:
            error = str(e)[:200]
            NDI_STATS = parse_hand_history(raw.decode("utf-8", "ignore"))
    else:
        NDI_STATS = parse_hand_history(raw.decode("utf-8", "ignore"))
    if GAME is not None:
        apply_ndi(GAME)
    return jsonify({"stats": NDI_STATS, "source": source, "job_id": job_id, "error": error,
                    "seconds": round(time.perf_counter() - t0, 1)})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5050"))
    print(f"Drex live: {drex_configured()}  ->  http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
