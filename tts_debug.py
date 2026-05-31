"""Diagnose the MiMo TTS 'Param Incorrect' 400.

Usage:
    .venv/bin/python tts_debug.py YOUR_TTS_KEY
    (or set TTS_KEY env var)

Tries several request-body variants against the token-plan endpoint and prints
the HTTP status + full JSON body for each, so we can see which `param` the API
rejects. The first variant that returns 200 is the correct shape.
"""
import os
import sys
import json
import requests

KEY = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TTS_KEY", "")).strip()
if not KEY:
    print("Provide your TTS key:  .venv/bin/python tts_debug.py YOUR_KEY")
    sys.exit(1)

URL = "https://token-plan-sgp.xiaomimimo.com/v1/chat/completions"
HEAD = {"api-key": KEY, "Content-Type": "application/json"}

MSGS = [
    {"role": "user", "content": ""},
    {"role": "assistant", "content": "Hello, this is a short narration test."},
]

VARIANTS = {
    "A current (miplan/, stream)": {
        "model": "miplan/mimo-v2.5-tts", "messages": MSGS,
        "audio": {"format": "wav", "voice": "Mia"}, "stream": False},
    "B no prefix, no stream": {
        "model": "mimo-v2.5-tts", "messages": MSGS,
        "audio": {"format": "wav", "voice": "Mia"}},
    "C miplan/, no stream": {
        "model": "miplan/mimo-v2.5-tts", "messages": MSGS,
        "audio": {"format": "wav", "voice": "Mia"}},
    "D user msg has style text": {
        "model": "miplan/mimo-v2.5-tts",
        "messages": [{"role": "user", "content": "Read this calmly."},
                     {"role": "assistant", "content": "Hello, this is a test."}],
        "audio": {"format": "wav", "voice": "Mia"}},
    "E format pcm16": {
        "model": "miplan/mimo-v2.5-tts", "messages": MSGS,
        "audio": {"format": "pcm16", "voice": "Mia"}},
    "F voice Chloe": {
        "model": "miplan/mimo-v2.5-tts", "messages": MSGS,
        "audio": {"format": "wav", "voice": "Chloe"}},
}

for name, body in VARIANTS.items():
    try:
        r = requests.post(URL, headers=HEAD, json=body, timeout=60)
        try:
            payload = r.json()
        except Exception:
            payload = r.text[:400]
        # don't dump the giant base64 audio on success
        if isinstance(payload, dict) and payload.get("choices"):
            audio = payload["choices"][0].get("message", {}).get("audio", {})
            size = len(audio.get("data", "") or "")
            short = {"choices": "OK", "audio_b64_len": size}
        else:
            short = payload
        print(f"\n### {name}  ->  HTTP {r.status_code}")
        print(json.dumps(short, ensure_ascii=False, indent=2)[:600])
        if r.status_code == 200:
            print(">>> THIS VARIANT WORKS — use this body shape.")
            break
    except Exception as e:
        print(f"\n### {name}  ->  EXCEPTION: {e}")
