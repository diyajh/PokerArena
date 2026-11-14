"""Auto-record a demo video of every feature (needs `python app.py` running).

    pip install playwright && python -m playwright install chromium
    python record_demo.py

Saves drex_poker_demo.webm (and .mp4 if ffmpeg is installed) in this folder.
"""
import os
import shutil
import subprocess
import time

from playwright.sync_api import sync_playwright

URL = os.getenv("DEMO_URL", "http://localhost:5050")
HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1440, 900


def wait_human_turn_or_over(pg, timeout=45000):
    """Wait until it's your turn (an action button enabled) or the hand is over (Next enabled)."""
    pg.wait_for_function(
        """() => document.querySelector('#actions button[data-a=check_call]:not([disabled])')
                || document.querySelector('#btnNext:not([disabled])')""", timeout=timeout)


def hand_over(pg):
    return pg.query_selector("#btnNext:not([disabled])") is not None


def play_hand(pg, style=("check_call",), max_actions=10):
    for i in range(max_actions):
        wait_human_turn_or_over(pg)
        if hand_over(pg):
            return
        pg.wait_for_timeout(1200)  # let viewers see the coach bars
        act = style[i % len(style)]
        btn = pg.query_selector(f"#actions button[data-a={act}]:not([disabled])") or \
            pg.query_selector("#actions button[data-a=check_call]:not([disabled])")
        btn.click()
    wait_human_turn_or_over(pg)


def peek(pg, secs=5):
    pg.click("#btnPeek")
    pg.wait_for_timeout(1500)
    pg.mouse.wheel(0, 400)
    pg.wait_for_timeout(secs * 1000)
    pg.keyboard.press("Escape")
    pg.wait_for_timeout(600)


def main():
    out_dir = os.path.join(HERE, "_video_tmp")
    os.makedirs(out_dir, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=os.getenv("DEMO_CHANNEL", "chrome"), headless=os.getenv("DEMO_HEADLESS") == "1", slow_mo=80)
        ctx = browser.new_context(viewport={"width": W, "height": H},
                                  record_video_dir=out_dir, record_video_size={"width": W, "height": H})
        pg = ctx.new_page()
        print("▶ opening app")
        pg.goto(URL)
        pg.wait_for_timeout(3000)

        # 1) Play a normal hand: coach bars, bubbles, commentator caption
        print("▶ playing a hand")
        pg.click("#btnNew")
        play_hand(pg, style=("raise_half_pot", "check_call", "check_call"))
        pg.wait_for_timeout(3500)  # caption + profile panel
        for e in ["😂", "🔥"]:
            pg.click(f"#reactions button:has-text('{e}')")
            pg.wait_for_timeout(500)
        peek(pg, 4)

        # 2) The demo bluff
        print("▶ demo hand (Manipulator bluff)")
        pg.click("#btnDemo")
        wait_human_turn_or_over(pg)
        pg.wait_for_timeout(2500)
        if not hand_over(pg):
            pg.click("#actions button[data-a=fold]")
            wait_human_turn_or_over(pg)
        pg.wait_for_timeout(3500)
        peek(pg, 7)

        # 3) Arena: leaderboard + benchmark (saved results load instantly)
        print("▶ arena + benchmark")
        pg.click(".tab[data-tab=arena]")
        pg.wait_for_timeout(2500)
        if not pg.query_selector("#leaderboard table"):
            pg.fill("#simHands", "10")
            pg.click("#btnSim")
            pg.wait_for_selector("#leaderboard table", timeout=120000)
        pg.wait_for_timeout(2500)
        if not pg.query_selector("#bench .bench"):
            pg.fill("#benchHands", "10")
            pg.click("#btnBench")
            pg.wait_for_selector("#bench .bench", timeout=120000)
        pg.mouse.wheel(0, 500)
        pg.wait_for_timeout(3500)

        # 4) NDI upload
        print("▶ NDI upload (can take ~30s)")
        pg.set_input_files("#ndiFile", os.path.join(HERE, "sample_hand_history.txt"))
        pg.click("#btnNdi")
        pg.wait_for_function("() => document.querySelector('#ndiOut').textContent.includes('Source:')", timeout=150000)
        pg.mouse.wheel(0, 600)
        pg.wait_for_timeout(5000)

        # 5) Evolution
        print("▶ evolution")
        pg.click(".tab[data-tab=evo]")
        pg.wait_for_timeout(1500)
        if pg.query_selector("#btnEvoReplay:not([disabled])"):
            pg.click("#btnEvoReplay")
        else:
            pg.fill("#evoGens", "4")
            pg.fill("#evoHands", "5")
            pg.click("#btnEvo")
        pg.wait_for_function("() => document.querySelector('#evoDominant').textContent.length > 0", timeout=240000)
        pg.wait_for_timeout(2000)
        pg.mouse.wheel(0, 700)
        pg.wait_for_timeout(2500)
        pg.mouse.wheel(0, -2000)
        pg.wait_for_timeout(2500)

        # 6) Watch bots play each other
        print("▶ watch bots")
        pg.click(".tab[data-tab=play]")
        pg.wait_for_timeout(800)
        pg.click("#btnWatch")
        pg.wait_for_timeout(20000)
        pg.click("#btnWatch")  # stop
        pg.wait_for_timeout(1500)

        video = pg.video.path()
        ctx.close()
        browser.close()

    webm = os.path.join(HERE, "drex_poker_demo.webm")
    shutil.move(video, webm)
    shutil.rmtree(out_dir, ignore_errors=True)
    print("✅ saved", webm)
    if shutil.which("ffmpeg"):
        mp4 = webm.replace(".webm", ".mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", webm, "-c:v", "libx264", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", mp4])
        print("✅ saved", mp4)


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"done in {time.time() - t:.0f}s")
