"""
Find which Gemini model is fastest for YOUR key.

    $env:GEMINI_API_KEY="YOUR_KEY"
    python test_models.py

It lists the models your key can use, sends each a tiny question in parallel,
and prints them fastest first, with the line to paste to switch model.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

KEY = os.getenv("GEMINI_API_KEY", "").strip().strip("\"'")
BASE = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
if not KEY:
    sys.exit('Set your key first:  $env:GEMINI_API_KEY="YOUR_KEY"')

H = {"x-goog-api-key": KEY}
try:
    r = requests.get(f"{BASE}/models", headers=H, params={"pageSize": 200}, timeout=30)
except requests.RequestException as exc:
    sys.exit(f"Could not reach Google: {type(exc).__name__}: {exc}")
if r.status_code != 200:
    sys.exit(f"Listing models failed: HTTP {r.status_code} {r.text[:300]}")

names = [m["name"].split("/", 1)[1] for m in r.json().get("models", [])
         if "generateContent" in m.get("supportedGenerationMethods", [])]
cands = [n for n in names if "flash" in n and not any(x in n for x in ("image", "tts", "live", "audio", "embedding", "robotics"))]
print(f"{len(names)} models available to your key; testing {len(cands)} flash models...\n")


def test(model):
    body = {"contents": [{"role": "user", "parts": [{"text": "Reply with the single word: OK"}]}],
            "generationConfig": {"maxOutputTokens": 64, "temperature": 0}}
    t = time.time()
    try:
        x = requests.post(f"{BASE}/models/{model}:generateContent", headers=H, json=body, timeout=40)
        secs = round(time.time() - t, 1)
        if x.status_code == 200:
            parts = (x.json().get("candidates") or [{}])[0].get("content", {}).get("parts") or []
            return model, secs, "OK" if any(p.get("text") for p in parts) else "empty reply"
        try:
            msg = x.json()["error"]["message"][:70]
        except Exception:
            msg = x.text[:70]
        return model, secs, f"HTTP {x.status_code}: {msg}"
    except requests.Timeout:
        return model, 40.0, "TIMEOUT (40s)"
    except requests.RequestException as exc:
        return model, round(time.time() - t, 1), type(exc).__name__


with ThreadPoolExecutor(max_workers=6) as pool:
    results = list(pool.map(test, cands))

results.sort(key=lambda x: (x[2] != "OK", x[1]))
print(f"{'MODEL':45} {'SECONDS':>8}  RESULT")
for m, s, res in results:
    print(f"{m:45} {s:>8}  {res}")
good = [m for m, s, res in results if res == "OK"]
if good:
    print(f'\nFastest working model: {good[0]}\nUse it with:\n  $env:GEMINI_MODEL="{good[0]}"\n  python app.py')
    if len(good) > 1:
        print(f'Backup (optional):\n  $env:GEMINI_FALLBACK_MODELS="{good[1]}"')
else:
    print("\nNo model answered. Check your key and internet connection, or try again in a few minutes.")
