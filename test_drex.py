"""Quick live check: python test_drex.py  -> runs the demo hand against real Drex."""
import app
c = app.app.test_client()
r = c.post("/api/demo_hand").get_json()
while not r["state"]["hand_over"]:
    r = c.post("/api/action", json={"action": "fold"}).get_json()
for d in r["peek"]["decisions"]:
    print(f"{d['name']:13} {d['street']:8} {d['source']:8} {d['ms']:6.0f}ms -> {d['choice']:15} | {d['why']}")
    for e in d["evidence"]:
        if e.startswith("Drex:"):
            print("              ", e)
    if d.get("drex_error"):
        print("               ERROR:", d["drex_error"])
print("Winner:", r["state"]["result"]["winners"])
