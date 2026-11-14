"""NDI (Nace Data Intelligence): turn a messy hand-history file into player stats.

Flow (console.nace.ai/openapi.json), same nace_sk_ key as Drex, Bearer auth:
  upload-grant -> upload file -> extract (workspace_file source) -> poll job -> result.data
"""
import json
import os
import time

import requests

NDI_BASE = (os.getenv("NDI_BASE_URL") or "https://console.nace.ai").rstrip("/")
PREFIXES = ["/v1/documents", "/v1"]  # console layout first, raw NDI layout second
NDI_TIMEOUT = float(os.getenv("NDI_TIMEOUT", "120"))

HAND_HISTORY_SCHEMA = {
    "type": "object",
    "properties": {
        "players": {
            "type": "array",
            "description": "One entry per distinct player name that appears in the hand history.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Player name exactly as written"},
                    "hands_played": {"type": "integer", "description": "Number of hands this player took part in"},
                    "raises": {"type": "integer", "description": "Count of raises and bets (including all-ins) by this player"},
                    "calls": {"type": "integer", "description": "Count of calls by this player"},
                    "checks": {"type": "integer", "description": "Count of checks by this player"},
                    "folds": {"type": "integer", "description": "Count of folds by this player"},
                    "showdowns": {"type": "integer", "description": "Times this player's cards were shown at showdown"},
                    "bluffs_shown": {"type": "integer",
                                     "description": "Times this player bet or raised and then showed a weak hand (a bluff)"},
                },
                "required": ["name"],
            },
        }
    },
    "required": ["players"],
}


def _key():
    return os.getenv("NDI_API_KEY") or os.getenv("DREX_API_KEY") or ""


def ndi_configured():
    return bool(_key())


def _headers():
    k = _key()
    return {"Authorization": f"Bearer {k}", "X-API-Key": k, "X-Nace-Client": "drex-poker-arena/1"}


def ndi_extract_players(file_bytes, filename="hand_history.txt"):
    """Returns (players list, job_id). Raises on failure.
    Flow from console.nace.ai/openapi.json:
      1. POST /v1/documents/upload-grants {"path"}         -> {workspace_id, upload_url, token}
      2. POST upload_url  multipart file + metadata, X-Upload-Token -> result.file.{file_id}
      3. POST /v1/documents/extract?wait_seconds=60 {source: workspace_file, schema, instructions}
      4. GET  /v1/documents/jobs/{id} until succeeded -> result.data
    """
    path = f"drex-poker/{int(time.time())}-{filename}"
    g = requests.post(f"{NDI_BASE}/v1/documents/upload-grants", headers=_headers(),
                      json={"path": path, "ttl_seconds": 600}, timeout=30)
    if g.status_code >= 400:
        raise RuntimeError(f"upload-grant -> {g.status_code}: {g.text[:200]}")
    grant = g.json()

    up = requests.post(grant["upload_url"], headers={"X-Upload-Token": grant["token"]},
                       files={"file": (filename, file_bytes, "text/plain"),
                              "metadata": (None, json.dumps({"path": path}), "application/json")},
                       timeout=120)
    if up.status_code >= 400:
        raise RuntimeError(f"upload -> {up.status_code}: {up.text[:200]}")
    uj = up.json()
    f = (uj.get("result") or {}).get("file") or uj.get("file") or uj
    file_id = f.get("file_id") or f.get("id")
    ws = f.get("workspace_id") or grant["workspace_id"]
    if not file_id:
        raise RuntimeError(f"upload response had no file id: {str(uj)[:200]}")

    body = {
        "name": "Drex Poker Arena hand history",
        "source": {"type": "workspace_file", "workspace_id": ws, "file_id": file_id},
        "schema": HAND_HISTORY_SCHEMA,
        "instructions": "This is a messy poker hand-history log. Count each player's actions across ALL hands. "
                        "Bets and all-ins count as raises. A bluff is a bet/raise followed by showing a weak hand.",
    }
    job = None
    for attempt in range(6):  # the file may still be ingesting right after upload
        r = requests.post(f"{NDI_BASE}/v1/documents/extract", params={"wait_seconds": 60},
                          headers=_headers(), json=body, timeout=90)
        if r.status_code < 400:
            job = r.json()
            break
        if r.status_code in (409, 422, 425) and attempt < 5 and "schema" not in r.text.lower():
            time.sleep(3)
            continue
        raise RuntimeError(f"extract -> {r.status_code}: {r.text[:300]}")
    deadline = time.time() + NDI_TIMEOUT
    while job.get("status") in ("queued", "running") and time.time() < deadline:
        time.sleep(2)
        jr = requests.get(f"{NDI_BASE}/v1/documents/jobs/{job['job_id']}", headers=_headers(), timeout=30)
        if jr.status_code < 400:
            job = jr.json()
    if job.get("status") != "succeeded":
        err = job.get("error") or {}
        raise RuntimeError(f"NDI job {job.get('job_id')} {job.get('status')}: {err.get('message', err)}")
    data = (job.get("result") or {}).get("data") or {}
    return data.get("players") or [], job.get("job_id")


def players_to_stats(players):
    """NDI rows -> {name: {actions, raises, raise_rate, fold_rate, showdowns, bluffs}}."""
    stats = {}
    for p in players:
        name = (p.get("name") or "").strip()
        if not name:
            continue
        raises = int(p.get("raises") or 0)
        calls = int(p.get("calls") or 0)
        checks = int(p.get("checks") or 0)
        folds = int(p.get("folds") or 0)
        acts = raises + calls + checks + folds
        if acts == 0:
            continue
        stats[name] = {
            "actions": acts, "raises": raises, "folds": folds,
            "raise_rate": raises / acts, "fold_rate": folds / acts,
            "showdowns": int(p.get("showdowns") or 0), "bluffs": int(p.get("bluffs_shown") or 0),
            "hands": int(p.get("hands_played") or 0),
        }
    return stats
