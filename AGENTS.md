# AGENTS.md: rules for coding agents (Nace Intern, etc.) working on Drex Poker Arena

## What this project is
A Texas Hold'em arena where 5 AI agents (Risk-Taker, Conservative, Statistician, Manipulator, Wildcard)
make every move with a live **Drex** call. NDI turns messy hand-history logs into opponent stats.
Modes: Play (you vs bots), Arena (leaderboard + benchmark), Evolution (natural selection of agent "genes").

## Layout
- `engine.py`: Hold'em rules: 5 actions, max 2 raises/round, layered pots (chips must always be conserved), `Tracker` (opponent history).
- `brain.py`: Drex client (`call_drex`, `/v1/systemone`, typed `choice` questions), personalities (`pick`), genes (`DEFAULT_GENES`, `apply_genes`), guardrails (`guard`), rule-based + LLM baselines, `commentate`.
- `app.py`: Flask API: `/api/new_game`, `/api/action`, `/api/demo_hand`, `/api/simulate`, `/api/benchmark`, `/api/evolve`, `/api/upload_history`, profile memory.
- `ndi.py`: NDI flow: upload-grant → upload → `/v1/documents/extract` → poll job.
- `static/index.html`: single-page UI (no build step).
- `test_drex.py`, `test_ndi.py`: live checks against the real APIs. `record_demo.py`: auto-records a demo video.

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env   # add DREX_API_KEY (nace_sk_...); NDI uses the same key
python app.py          # http://localhost:5050
```

## Rules
1. **Never commit `.env` or print API keys.** Keys live only in `.env` (git-ignored).
2. **Chips must be conserved.** After any engine change, run a long bots-only simulation and assert the total chips never change.
3. **Every bot move must stay a Drex decision.** Personalities/genes/guards may reshape Drex's probabilities, but never replace Drex with hard-coded play for Drex bots.
4. **Be honest in the UI.** Only label something "Drex" if Drex actually answered; fallbacks are labelled "fallback"/"Estimate".
5. Only legal actions are offered to Drex (`ctx["legal"]`). Keep it that way (0 illegal moves is a headline metric).
6. Keep the UI a single self-contained `static/index.html`; no frameworks or build tools.
7. Before finishing: `python -c "import app"` must succeed, and the demo hand (`/api/demo_hand`) must still run end to end.

## Quick checks
```bash
python test_drex.py    # demo hand vs live Drex: Manipulator should bluff, Conservative should fold
python test_ndi.py     # sample_hand_history.txt through NDI
```
