# 🃏 Drex Poker Arena

**Five AI poker players, all powered by [Drex](https://www.nace.ai/drex). They read each other, bluff each other, evolve, and you can see exactly why every move happened.**

Built at **Drexathon** (Nace AI × Adobe EDU, Nov 2026) by **Jada Sange** and **Diya Jhawar**.

## What it does
You play Texas Hold'em against AI bots, or watch them play each other. Every bot move is a **live Drex call**: the game sends Drex the situation (cards, win chance, pot odds, opponents' history), Drex returns a probability for each **legal** move, and each bot's personality decides how to use those odds.

| Bot | How it uses Drex |
|---|---|
| 🔥 Risk-Taker | Boosts raise and all-in odds, then picks the top move |
| 🪨 Conservative | Only raises if Drex is over 60% sure |
| 🧮 Statistician | Always takes Drex's top move |
| 🕵️ Manipulator | Asks Drex about opponents ("Will they fold?") and exploits them |
| 🎲 Wildcard | Randomly samples from Drex's odds |

## Features
- **💡 Coach bars:** Drex's odds for your own moves
- **🧠 Brain-peek:** after each hand, see every bot's cards, Drex's probabilities, the reason for each move, and decision time
- **🎙️ Drex Commentator:** Drex calls out the play of the hand (bluff, hero call, smart fold…)
- **🧠 The bots remember you:** Drex labels your style (e.g. "Calling station"), and the bots adapt to it
- **🧬 Evolution mode:** the worst bot goes extinct, the best spawns a mutated child, and strategies evolve over generations
- **🏟️ Arena + benchmark:** bots-only leaderboard, and Drex vs. a rule-based bot
- **👁️ NDI upload:** NDI reads messy hand-history logs and feeds opponent stats to the bots

## What makes it different
- **Explainable:** most poker bots are black boxes. Here, every decision shows Drex's probabilities and the evidence behind it.
- **Multi-agent:** five agents that compete, read and exploit each other (and you), with one small decision model making every call.
- **It evolves:** we didn't hand-design the best strategy; the agents evolve it.
- **Measured:** 368 ms per decision, $0.024 per 1,000 decisions, **0 illegal moves** (the rule-based bot made 4). NDI found 39 actions and 3 bluffs in a messy log where a basic parser found 11 and 0.

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env    # add DREX_API_KEY (and NDI_API_KEY for uploads)
python app.py           # → http://localhost:5050
```
Without a key, a local fallback brain is used and the UI says so. `python record_demo.py` auto-records a demo video of every feature.

## 🤖 Work on this repo with Nace Intern
This repo is **Intern-ready**: [nace-ai/intern](https://github.com/nace-ai/intern) is Nace's open-source coding agent with a two-part brain, where Drex (System 1) watches every turn of a worker model (System 2). `AGENTS.md` gives it this project's rules.
```bash
uv tool install "intern-agent @ git+https://github.com/nace-ai/intern"
export NACE_API_KEY=...          # same nace_sk_ key: Drex as System 1
export OPENROUTER_API_KEY=...    # or any provider for System 2
cd PokerArena
intern --system2 openrouter/qwen/qwen3.8-27b "Add a new personality called 'Copycat' that mirrors the last opponent's action"
```
So Drex both **plays the game** (every bot move) and **supervises the agent that builds the game**.

Built with