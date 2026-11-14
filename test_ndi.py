"""Quick live check: python test_ndi.py  -> sends sample_hand_history.txt through NDI."""
import os
import app  # loads .env
import ndi
print("base:", ndi.NDI_BASE, "| key set:", ndi.ndi_configured())
try:
    players, job = ndi.ndi_extract_players(open("sample_hand_history.txt", "rb").read(), "sample_hand_history.txt")
    print("job:", job)
    for name, s in ndi.players_to_stats(players).items():
        print(f"  {name:14} raises {s['raise_rate']:.0%}  folds {s['fold_rate']:.0%}  bluffs {s['bluffs']}/{s['showdowns']}  ({s['actions']} actions)")
except Exception as e:
    print("NDI ERROR:", e)
