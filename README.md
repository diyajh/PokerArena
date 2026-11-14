# 🃏 Drex Poker Arena

A poker arena where every player's brain is Drex. Play against AI personalities, watch them battle, and benchmark Drex against a rule-based bot.

## Run
```bash
cd ~/Desktop/drex-poker
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py          # → http://localhost:5050
```
Plug in Drex: `cp .env.example .env`, paste your key into `DREX_API_KEY=`, restart. The badge turns green ("Drex: LIVE").
Without a key a local fallback brain is used and the UI says so.

## Files
- `engine.py` — Hold'em engine (treys), 5 actions, max 2 raises/round, layered pots, opponent tracker
- `brain.py` — state text, Drex call (3s timeout → fallback), 5 personalities, rule-based bot, coach
- `app.py` — Flask API: play, demo hand, watch, arena simulate, benchmark, NDI upload
- `static/index.html` — table UI, animations, coach bars, brain-peek, arena tab
- `static/avatars/<personality>.png` — drop Firefly avatars here (risk_taker.png, conservative.png, statistician.png, manipulator.png, wildcard.png)
- `sample_hand_history.txt` — demo file for the NDI upload

## Demo script (5 min)
1. **⭐ Demo hand** → fold your Q4. The Manipulator holds 8♣3♦ (16% to win). The Conservative has folded 5 of its last 6 actions.
   The Manipulator asks Drex two questions in one call: *which move?* and *will the Conservative fold to a big raise?*
   If Drex says yes, it bluffs. **🧠 Brain-peek** shows the bars, "Drex: Conservative folds to a big raise X%", and the decision time.
2. **Catch the bluffer:** when the Risk-Taker bets into the Manipulator, it asks Drex "is this a bluff?" and hero-calls if yes.
3. **🏟️ Arena → Run arena:** 5 personalities, 100+ hands, leaderboard.
4. **🦖 Benchmark:** Drex vs a rule-based bot vs an LLM: speed, cost per 1000 decisions, illegal moves, math mistakes.
5. **📂 Upload `sample_hand_history.txt`** → NDI extracts each player's stats from a messy log → seeded into the bots' opponent history.

## Keys (.env)
`DREX_API_KEY` (required), `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` (LLM column), `NDI_API_KEY` (hand-history upload). See `.env.example`.

Keys: F fold · C check/call · H half-pot · P pot · A all-in · N next hand · B brain-peek
