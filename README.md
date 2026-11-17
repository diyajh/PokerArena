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
- **🌐 Play online with friends:** create a room, send the 4-letter code, and up to 5 people play at one table (Drex bots fill the empty seats). Everyone sees only their own cards, and chips carry over between games
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

## Play online
1. Click **🌐 Play online**, type your name and click **Create room**. You get a code like `K7QX`.
2. Send friends the code, or click **🔗 Copy invite** and send the link.
3. They open the site, click **🌐 Play online**, and join with the code.
4. The host picks bots under **Opponents** and presses **▶ Start game**.

- **Next hand:** anyone can press it when a hand ends (it deals by itself after 25 s).
- **New game:** host only. Everyone keeps their chips; only players at 0 start over at 1000. A player at 0 can also press **New game (get 1000)** for themselves.
- Chips are saved by name in `online_chips.json`, so leaving and rejoining keeps them.
- If it's your turn and you don't act in 45 s, you check (or fold if you'd have to pay).
- Friends on the same Wi-Fi use `http://<your-computer's-IP>:5050`. For friends elsewhere, put the server online (e.g. a tunnel such as `cloudflared tunnel --url http://localhost:5050`, or a host like Render).

## Put it online (Render)
`render.yaml` sets everything up: Render → **New → Blueprint** → pick this repo → paste your keys when asked.
It runs `gunicorn` with **one worker** (rooms live in memory) and protects your API credits with a per-IP
rate limit and a daily Drex cap (`DREX_DAILY_CAP`; after that the bots use the local brain until midnight UTC).
On the free plan, saved chips reset when the service restarts; add a disk and set `DATA_DIR` to keep them.

Built with Drex, NDI, Adobe Firefly (bot avatars), Python and Flask.