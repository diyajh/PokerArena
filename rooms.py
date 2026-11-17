"""Online rooms: share a code, and up to 5 people play at one table from their own browsers.

How a room works
- The host creates a room and gets a 4-letter code. Friends join with that code and a name.
- The host picks Drex bots to fill empty seats and presses Start.
- Everyone sees only their own hole cards. Bots act on their own (background thread per room).
- Next hand: anyone at the table can press it once a hand is over.
- New game: the host only. Everyone keeps their chips; anyone at 0 starts over at 1000.
- Your chips are remembered by name, so leaving and rejoining keeps them.
- If it's your turn and you don't act in TURN_SECONDS, you check (or fold if you can't check).
"""
import json
import os
import re
import secrets
import threading
import time

from brain import PERSONALITIES, Brain, commentate, drex_configured, win_chance
from engine import RAISE_ACTIONS, Game, Player, cards_str

MAX_HUMANS = 5
MAX_SEATS = 6
START_CHIPS = 1000
TURN_SECONDS = 45
AUTO_DEAL_SECONDS = 25    # hand over and nobody pressed Next hand -> deal anyway
AWAY_SECONDS = 20         # no poll from a browser for this long -> they left the table
ROOM_IDLE_SECONDS = 1800   # nobody polling for this long -> the room is closed
CODE_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O or 1/I mix-ups
REACTIONS = {"😂", "😱", "🔥", "👀", "🤡", "😎", "👏", "😭"}
BOT_KINDS = ["risk_taker", "conservative", "statistician", "manipulator", "wildcard"]

MAX_ROOMS = int(os.getenv("MAX_ROOMS", "50"))  # each room has a background thread; cap them on a public host
CHIPS_FILE = os.path.join(os.getenv("DATA_DIR") or os.path.dirname(os.path.abspath(__file__)), "online_chips.json")
_chips_lock = threading.Lock()


def clean_name(name):
    """Names show up on everyone's screen, so keep them short and plain (no HTML)."""
    name = re.sub(r"[^\w \-]", "", str(name or ""), flags=re.UNICODE).strip()[:16]
    return name or "Player"


def _read_records():
    """name (lower case) -> {"name", "chips", "hands", "won"}. Older files stored just chips."""
    try:
        data = json.load(open(CHIPS_FILE))
    except Exception:
        return {}
    return {k: v if isinstance(v, dict) else {"name": k, "chips": v, "hands": 0, "won": 0}
            for k, v in data.items()}


def _write_records(data):
    try:
        json.dump(data, open(CHIPS_FILE, "w"))
    except Exception:
        pass


def _record(data, name):
    return data.setdefault(name.lower(), {"name": name, "chips": START_CHIPS, "hands": 0, "won": 0})


def load_chips(name):
    with _chips_lock:
        rec = _read_records().get(name.lower())
        return rec["chips"] if rec else START_CHIPS


def save_chips(updates):
    with _chips_lock:
        data = _read_records()
        for name, chips in updates.items():
            _record(data, name)["chips"] = chips
        _write_records(data)


def record_hand(rows):
    """rows: [(name, chips, won_this_hand)] for each person dealt into the hand that just ended."""
    with _chips_lock:
        data = _read_records()
        for name, chips, won in rows:
            rec = _record(data, name)
            rec.update(name=name, chips=chips, hands=rec["hands"] + 1, won=rec["won"] + int(won))
        _write_records(data)


def leaderboard():
    rows = sorted(_read_records().values(), key=lambda r: (r["chips"], r["won"]), reverse=True)
    return [{"name": r["name"], "chips": r["chips"], "hands": r["hands"], "won": r["won"]} for r in rows]


class Seat:
    """One person in the room (has a browser token)."""

    def __init__(self, token, name, host):
        self.token = token
        self.name = name
        self.host = host
        self.last_seen = time.time()
        self.player = None  # engine Player once seated


class Room:
    def __init__(self, code, host_name, bots):
        self.code = code
        self.lock = threading.RLock()
        self.brain = Brain()
        self.seats = {}       # token -> Seat
        self.bots = [b for b in bots if b in BOT_KINDS][: MAX_SEATS - 1]
        self.game = None
        self.log = []         # [(id, event)], every client reads it from its own last id
        self.next_id = 1
        self.turn_started = time.time()
        self.coach_cache = {}
        self.commentary = None
        self.commentary_hand = None
        self.last_activity = time.time()
        self.alive = True
        self.host_token = self.add_person(host_name, host=True)
        threading.Thread(target=self._loop, daemon=True).start()

    # ---------- people ----------
    def add_person(self, name, host=False):
        name = clean_name(name)
        taken = {s.name.lower() for s in self.seats.values()}
        base, n = name, 2
        while name.lower() in taken:
            name = f"{base[:13]} {n}"
            n += 1
        token = secrets.token_urlsafe(16)
        self.seats[token] = Seat(token, name, host)
        self.emit(type="system", text=f"{name} joined the room")
        if self.game is not None:
            self._seat_late_joiner(self.seats[token])
        return token

    def leave(self, token):
        s = self.seats.get(token)
        if s is None:
            return
        s.gone = True
        s.last_seen = 0  # the loop sits them out right away; their chips are already saved
        if s.player is not None:
            save_chips({s.name: s.player.chips})
        self.emit(type="system", text=f"{s.name} left the room")

    def _seat_late_joiner(self, seat):
        """Someone joined after the game started: give them a free seat, or a bot's seat,
        between hands. Mid-hand they wait with no cards until the next deal."""
        g = self.game
        p = Player(len(g.players), seat.name, "human", chips=load_chips(seat.name), human=True, brain="human")
        if len(g.players) < MAX_SEATS:
            p.folded = True  # sits out the hand already running
            g.players.append(p)
        else:
            bot = next((q for q in g.players if not q.human), None)
            if bot is None:
                self.emit(type="system", text=f"Table is full, {seat.name} is watching")
                return
            p.seat = bot.seat
            if not g.hand_over:  # can't swap a bot out mid-hand; it happens at the next deal
                seat.pending_swap = p
                return
            g.players[bot.seat] = p
        seat.player = p

    # ---------- events ----------
    def emit(self, **ev):
        self.log.append((self.next_id, ev))
        self.next_id += 1
        if len(self.log) > 400:
            self.log = self.log[-300:]

    def _drain(self):
        for ev in self.game.pop_events():
            self.emit(**ev)
        self.turn_started = time.time()

    # ---------- game flow ----------
    def start(self, bots=None):
        """Host pressed Start (or New game). Chips carry over; anyone at 0 starts over at 1000."""
        if bots is not None:
            self.bots = [b for b in bots if b in BOT_KINDS]
        old = {}
        if self.game is not None:
            old = {p.name: p.chips for p in self.game.players}
        people = [s for s in self.seats.values() if not getattr(s, "gone", False)][:MAX_HUMANS]
        bots = self.bots[: max(0, MAX_SEATS - len(people))]
        if len(people) + len(bots) < 2:
            raise ValueError("Need at least 2 players: invite a friend or add a bot")
        players = []
        for s in people:
            chips = old.get(s.name, load_chips(s.name))
            p = Player(len(players), s.name, "human", chips=chips if chips > 0 else START_CHIPS,
                       human=True, brain="human")
            s.player = p
            players.append(p)
        for b in bots:
            meta = PERSONALITIES[b]
            p = Player(len(players), meta["name"], b, brain="drex")
            chips = old.get(p.name, START_CHIPS)
            p.chips = chips if chips > 0 else START_CHIPS
            players.append(p)
        self.game = Game(players, decide_fn=self.brain.decide)
        self.emit(type="system", text="New game: chips carried over, anyone at 0 starts at 1000")
        self.deal()

    def deal(self):
        """Next hand. Swaps in late joiners and sits out people who left."""
        g = self.game
        if g is None or not g.hand_over:
            return False
        for s in self.seats.values():
            pend = getattr(s, "pending_swap", None)
            if pend is not None:
                g.players[pend.seat] = pend
                s.player, s.pending_swap = pend, None
        if not g.start_hand():
            self.emit(type="system", text="Waiting: not enough players with chips")
            return False
        self._drain()
        return True

    def act(self, token, action):
        g, s = self.game, self.seats.get(token)
        if g is None or s is None or s.player is None or g.hand_over:
            raise ValueError("no active hand")
        p = s.player
        if g.to_act != p.seat:
            raise ValueError("not your turn")
        if action not in g.legal(p):
            raise ValueError(f"illegal action {action}")
        self._human_move(p, action)

    def _human_move(self, p, action):
        g = self.game
        # the advice this player's coach bars showed (cached when they polled); None after a timeout
        coach = self.coach_cache.get((g.hand_no, len(g.decisions), g.street))
        eq = win_chance(p.hole, g.board, len(g.in_hand()) - 1, runs=150, rng=g.rng)
        if action in RAISE_ACTIONS:
            g.raised_equities.setdefault(p.seat, []).append(eq)
        g.decisions.append({"seat": p.seat, "name": p.name, "personality": "human", "street": g.street,
                            "choice": action, "to_call": g.to_call(p), "equity": eq, "hole": cards_str(p.hole),
                            "probs": None, "source": "human", "ms": 0, "why": "Human move", "evidence": [],
                            "coach": {k: coach[k] for k in ("probs", "top", "source", "equity", "pot_odds")}
                            if coach else None})
        g.apply(p, action)
        self._drain()
        if g.hand_over:
            self._hand_finished()

    def rebuy(self, token):
        """Only someone at 0 can get a fresh 1000."""
        s = self.seats.get(token)
        if s is None or s.player is None or s.player.chips > 0:
            raise ValueError("you still have chips")
        s.player.chips = START_CHIPS
        save_chips({s.name: START_CHIPS})
        self.emit(type="system", text=f"{s.name} is back with 1000 chips")

    def _hand_finished(self):
        g = self.game
        won = (g.result or {}).get("won", {})
        # people who were dealt cards this hand (not late joiners waiting, not people who stepped away)
        record_hand([(p.name, p.chips, str(p.seat) in won) for p in g.players if p.human and p.hole])
        save_chips({p.name: p.chips for p in g.players if p.human and not p.hole})

    # ---------- background loop: bots, timeouts, people leaving ----------
    def _loop(self):
        while self.alive:
            try:
                self._tick()
            except Exception as e:  # keep the room alive whatever happens
                print(f"[room {self.code}] {e!r}")
            time.sleep(0.3)

    def _tick(self):
        now = time.time()
        with self.lock:
            if now - self.last_activity > ROOM_IDLE_SECONDS:
                self.alive = False
                return
            for s in self.seats.values():  # browser closed -> sit out
                away = now - s.last_seen > AWAY_SECONDS
                if s.player is not None and getattr(s.player, "away", False) != away:
                    s.player.away = away
                    self.emit(type="system", text=f"{s.name} {'left the table' if away else 'is back'}")
            g = self.game
            if g is None:
                return
            if g.hand_over:
                if self.commentary_hand == g.hand_no or not g.hand_no:
                    # nobody pressed Next hand: deal anyway so one AFK person can't freeze the table
                    if now - self.turn_started > AUTO_DEAL_SECONDS:
                        self.deal()
                    return
                self.commentary_hand = g.hand_no
                decisions, result = list(g.decisions), g.result
                p = None
            else:
                p = g.players[g.to_act]
                if p.human:
                    if getattr(p, "away", False) or now - self.turn_started > TURN_SECONDS:
                        self._human_move(p, "check_call" if g.to_call(p) == 0 else "fold")
                    return
                hand, to_act = g.hand_no, g.to_act
        if p is None:  # hand just ended: one Drex call for the commentator caption, outside the lock
            try:
                self.commentary = commentate(decisions, result, None)
            except Exception:
                self.commentary = None
            return
        action, info = self.brain.decide(g, p)  # a Drex call: done outside the lock
        with self.lock:
            if self.game is not g or g.hand_over or g.hand_no != hand or g.to_act != to_act:
                return
            if action in RAISE_ACTIONS and info and "equity" in info:
                g.raised_equities.setdefault(p.seat, []).append(info["equity"])
            if info:
                info["street"] = g.street
                g.decisions.append(info)
            g.apply(p, action)
            self._drain()
            if g.hand_over:
                self._hand_finished()

    # ---------- what one person sees ----------
    def view(self, token, since):
        s = self.seats[token]
        s.last_seen = self.last_activity = time.time()
        g = self.game
        me = s.player.seat if s.player is not None else None
        events = [dict(ev, id=i) for i, ev in self.log if i > since]
        out = {
            "room": {
                "code": self.code, "me": s.name, "host": s.host, "started": g is not None,
                "people": [{"name": x.name, "host": x.host,
                            "away": bool(x.player is not None and getattr(x.player, "away", False))}
                           for x in self.seats.values() if not getattr(x, "gone", False)],
                "bots": self.bots, "can_rebuy": bool(s.player is not None and s.player.chips <= 0),
                "turn_seconds": TURN_SECONDS,
                "turn_left": max(0, round(TURN_SECONDS - (time.time() - self.turn_started))),
            },
            "events": events, "last_id": self.next_id - 1,
        }
        if g is None:
            return out
        over = g.hand_over
        shown = (g.result or {}).get("hands", {}) if over else {}
        players = []
        for p in g.players:
            hole = cards_str(p.hole) if (p.seat == me or (over and str(p.seat) in shown)) and p.hole else None
            players.append({
                "seat": p.seat, "name": p.name, "personality": p.personality,
                "emoji": "👤" if p.human else PERSONALITIES.get(p.personality, {}).get("emoji", "🙂"),
                "chips": p.chips, "bet": p.bet, "folded": p.folded, "all_in": p.all_in,
                "out": p.out, "last_action": p.last_action,
                "human": p.seat == me,          # the page treats `human` as "this is me"
                "person": p.human, "away": getattr(p, "away", False),
                "hole": hole, "has_cards": bool(p.hole) and not p.folded,
            })
        out["state"] = {
            "mode": "online", "hand_no": g.hand_no, "street": g.street, "board": cards_str(g.board),
            "pot": g.pot if not over else 0, "to_act": g.to_act if not over else None,
            "dealer": g.dealer, "hand_over": over, "result": g.result, "players": players,
            "drex_live": drex_configured(),
        }
        if not over and me is not None and g.to_act == me:
            key = (g.hand_no, len(g.decisions), g.street)
            if key not in self.coach_cache:  # one Drex call per turn, not per poll
                self.coach_cache = {key: self.brain.coach(g, s.player)}
            out["coach"] = self.coach_cache[key]
        if over:
            out["commentary"] = self.commentary if self.commentary_hand == g.hand_no else None
            # brain-peek: bots' thinking and your own moves, never another person's cards
            out["peek"] = {
                "decisions": [d for d in g.decisions if d.get("source") != "human" or d["seat"] == me],
                "holes": {p.name: cards_str(p.hole) for p in g.players
                          if p.hole and (not p.human or p.seat == me or str(p.seat) in shown)},
                "timeline": g.hand_log, "hand_no": g.hand_no,
            }
        return out


ROOMS = {}
_rooms_lock = threading.Lock()


def new_code():
    while True:
        code = "".join(secrets.choice(CODE_LETTERS) for _ in range(4))
        if code not in ROOMS:
            return code


def create_room(name, bots):
    with _rooms_lock:
        for c in [c for c, r in ROOMS.items() if not r.alive]:
            del ROOMS[c]
        if len(ROOMS) >= MAX_ROOMS:
            raise ValueError("The server is full right now. Try again in a few minutes.")
        code = new_code()
        room = Room(code, name, bots)
        ROOMS[code] = room
        return room, room.host_token


def get_room(code):
    room = ROOMS.get(str(code or "").strip().upper())
    if room is None or not room.alive:
        raise KeyError("No room with that code. Check it, or ask your friend to create a new one.")
    return room


def join_room(code, name, token=None):
    room = get_room(code)
    with room.lock:
        if token and token in room.seats:  # same browser coming back
            room.seats[token].gone = False
            return room, token
        # same name, new browser (closed the tab, switched phone): take your old seat and chips back
        name = clean_name(name)
        for old_token, s in list(room.seats.items()):
            if s.name.lower() == name.lower() and time.time() - s.last_seen > AWAY_SECONDS:
                new_token = secrets.token_urlsafe(16)
                room.seats[new_token] = room.seats.pop(old_token)
                s.token, s.gone, s.last_seen = new_token, False, time.time()
                if room.host_token == old_token:
                    room.host_token = new_token
                return room, new_token
        if sum(not getattr(s, "gone", False) for s in room.seats.values()) >= MAX_HUMANS:
            raise ValueError("That room is full (5 players)")
        return room, room.add_person(name)
