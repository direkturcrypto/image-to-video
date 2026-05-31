"""
Sketch Reactor v4 — local bulk image generator (OpenAI / GPT Image 2)
+ MiMo-V2.5-TTS narrator voice generator
---------------------------------------------------------------------
Switched from Google Gemini to OpenAI GPT Image 2 (gpt-image-2), released
2026-04-21. Verified against the official OpenAI image API reference.

Login = paste an OpenAI API key from https://platform.openai.com/api-keys
(NOTE: this is an OpenAI key starting with "sk-", NOT a Google key).
GPT Image 2 has NO free tier — it is pay-per-image. Use quality "low" to keep
cost/latency down for bulk runs.

Endpoints used:
  * /v1/images/generations  -> text-only prompt (no reference images)
  * /v1/images/edits        -> prompt + reference images (style anchors / prev)
GPT image models always return base64 in data[0].b64_json.

TTS (MiMo-V2.5-TTS):
  * POST /v1/chat/completions  -> OpenAI-compatible TTS with expression tags
  * Supports [crying], [pause], [sniffles], [laughing], etc. for natural narration
  * Voices: Mia, Chloe (English female), Milo, Dean (English male)
            冰糖, 茉莉 (Chinese female), 苏打, 白桦 (Chinese male)

Run:  python app.py   ->   http://localhost:5000
"""

import os
import io
import time
import json
import base64
import shutil
import zipfile
import threading
import subprocess
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory
import requests

APP_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = APP_DIR / "output"
FRAMES_DIR = APP_DIR / "frames"
ANCHOR_DIR = APP_DIR / "anchors"
for d in (OUTPUT_DIR, FRAMES_DIR, ANCHOR_DIR):
    d.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=None)

STATE = {"running": False, "stop": False, "total": 0, "items": [], "settings": {}}
LOCK = threading.Lock()

OPENAI_BASE = "https://api-direct.derouter.network/openai/v1"

# TTS config (MiMo-V2.5-TTS via Xiaomi)
TTS_BASE = "https://token-plan-sgp.xiaomimimo.com/v1"
TTS_MODEL = "miplan/mimo-v2.5-tts"

# Image-capable models on OpenAI. gpt-image-2 is the current flagship.
IMAGE_MODELS = {
    "gpt-image-2",          # flagship, best quality (default)
    "gpt-image-1.5",        # cheaper / older
    "gpt-image-1-mini",     # cheapest, fast prototyping
}

# rough per-image cost estimates (USD) at 1024x1024, used only for the UI label
COST_PER_IMG = {
    "gpt-image-2": {"low": 0.02, "medium": 0.07, "high": 0.19},
    "gpt-image-1.5": {"low": 0.015, "medium": 0.05, "high": 0.15},
    "gpt-image-1-mini": {"low": 0.005, "medium": 0.01, "high": 0.02},
}


def has_ffmpeg():
    return shutil.which("ffmpeg") is not None


# --------------------------------------------------------------------------
# Sample video -> frames
# --------------------------------------------------------------------------
@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    if not has_ffmpeg():
        return jsonify({"error": "ffmpeg not found. Install it or upload style "
                                 "images directly instead."}), 400
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no video file"}), 400
    for p in FRAMES_DIR.glob("*.jpg"):
        p.unlink()
    vid = APP_DIR / "sample_video.mp4"
    f.save(vid)
    try:
        dur = float(subprocess.check_output([
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(vid)]).decode().strip())
    except Exception:
        dur = 60.0
    saved = []
    for i in range(9):
        t = max(0.5, dur * (i + 0.5) / 9)
        out = FRAMES_DIR / f"frame_{i:02d}.jpg"
        subprocess.run(["ffmpeg", "-v", "quiet", "-ss", str(t), "-i", str(vid),
                        "-frames:v", "1", "-vf", "scale=512:-1", "-y", str(out)])
        if out.exists():
            saved.append(out.name)
    return jsonify({"ok": True, "frames": saved})


@app.route("/frames/<path:fn>")
def serve_frame(fn):
    return send_from_directory(FRAMES_DIR, fn)


@app.route("/api/set_anchors", methods=["POST"])
def set_anchors():
    chosen = (request.json or {}).get("frames", [])
    for p in ANCHOR_DIR.glob("*"):
        p.unlink()
    out = []
    for i, fn in enumerate(chosen[:2]):     # cap 2 anchors (leave room for prev img)
        src = FRAMES_DIR / fn
        if src.exists():
            dst = ANCHOR_DIR / f"anchor_{i}.jpg"
            shutil.copy(src, dst)
            out.append(dst.name)
    return jsonify({"ok": True, "anchors": out})


@app.route("/api/upload_anchor", methods=["POST"])
def upload_anchor():
    f = request.files.get("file")
    slot = request.form.get("slot", "0")
    if not f or slot not in ("0", "1"):
        return jsonify({"error": "bad upload"}), 400
    f.save(ANCHOR_DIR / f"anchor_{slot}.jpg")
    return jsonify({"ok": True})


def anchor_paths():
    return sorted(str(p) for p in ANCHOR_DIR.glob("anchor_*.jpg"))


# --------------------------------------------------------------------------
# OpenAI generation
# --------------------------------------------------------------------------
def build_prompt(prompt, settings):
    style = settings.get("style_suffix", "").strip()
    negative = settings.get("negative", "").strip()
    out = prompt.strip()
    if style:
        out += "\n\nART STYLE (match exactly): " + style
    if negative:
        # semantic negative: describe what to avoid
        out += "\nAvoid: " + negative
    out += ("\nKeep character design, proportions, line weight, and colors "
            "identical to the reference image(s) provided.")
    return out


def generate_image(api_key, prompt, settings, ref_paths):
    """Call OpenAI images API, return PNG bytes. Raises on failure.

    If reference images are present, use /v1/images/edits (multipart) so the
    style anchors + previous image guide the output. Otherwise use the plain
    /v1/images/generations JSON endpoint.
    """
    model = settings.get("model", "gpt-image-2")
    size = settings.get("size", "1024x1024")
    quality = settings.get("quality", "low")
    full_prompt = build_prompt(prompt, settings)
    headers = {"Authorization": f"Bearer {api_key}"}

    refs = [p for p in ref_paths if Path(p).exists()][:4]  # gpt-image edits: a few refs

    if refs:
        # multipart edits endpoint
        url = f"{OPENAI_BASE}/images/edits"
        data = {"model": model, "prompt": full_prompt,
                "size": size, "quality": quality, "n": "1"}
        print(f"[DEBUG] Calling {url} with model={model}, size={size}, quality={quality}")
        files = []
        open_handles = []
        try:
            for p in refs:
                fh = open(p, "rb")
                open_handles.append(fh)
                mime = "image/jpeg" if str(p).lower().endswith((".jpg", ".jpeg")) else "image/png"
                # OpenAI accepts repeated "image[]" fields for multiple references
                files.append(("image", (Path(p).name, fh, mime)))
            r = requests.post(url, headers=headers, data=data, files=files, timeout=600)
        finally:
            for fh in open_handles:
                try:
                    fh.close()
                except Exception:
                    pass
    else:
        url = f"{OPENAI_BASE}/images/generations"
        body = {"model": model, "prompt": full_prompt,
                "size": size, "quality": quality, "n": 1}
        r = requests.post(url, headers={**headers, "Content-Type": "application/json"},
                          json=body, timeout=600)

    if r.status_code != 200:
        retry = r.headers.get("retry-after")
        detail = r.text[:400]
        try:
            detail = r.json().get("error", {}).get("message", detail)
        except Exception:
            pass
        print(f"[ERROR] Image generation failed: {r.status_code} - {detail}")
        raise RuntimeError(f"{r.status_code}: {detail}"
                           + (f" (retry-after={retry})" if retry else ""))

    data = r.json()
    arr = data.get("data") or []
    if arr and arr[0].get("b64_json"):
        return base64.b64decode(arr[0]["b64_json"])
    # some gateways return a url instead of b64
    if arr and arr[0].get("url"):
        img = requests.get(arr[0]["url"], timeout=120)
        if img.status_code == 200:
            return img.content
    raise RuntimeError("No image in response: " + json.dumps(data)[:300])


def set_item(idx, **kw):
    with LOCK:
        STATE["items"][idx].update(kw)


def process_one(api_key, item, settings, use_prev):
    idx = item["idx"]
    max_retries = int(settings.get("retries", 3))
    refs = anchor_paths()
    # previous image: saved 1-based, so for item idx the prev file is f"{idx:03d}"
    if use_prev and idx > 0:
        prev = OUTPUT_DIR / f"{idx:03d}.png"
        if prev.exists():
            refs = refs + [str(prev)]
    for attempt in range(max_retries + 1):
        if STATE["stop"]:
            return
        set_item(idx, status="busy")
        try:
            png = generate_image(api_key, item["prompt"], settings, refs)
            (OUTPUT_DIR / f"{idx + 1:03d}.png").write_bytes(png)
            set_item(idx, status="done", file=f"{idx + 1:03d}.png", error=None)
            return
        except Exception as e:
            msg = str(e)
            wait = 2.0 * (attempt + 1)
            # 429 = rate limit -> back off harder
            if msg.startswith("429") or "rate limit" in msg.lower() \
               or "Too Many Requests" in msg:
                wait = max(wait, 20.0)
            if "retry-after=" in msg:
                try:
                    wait = max(wait, float(msg.split("retry-after=")[1].split(")")[0]))
                except Exception:
                    pass
            if attempt == max_retries:
                set_item(idx, status="error", error=msg)
            else:
                time.sleep(wait)


def run_job(api_key, settings):
    use_prev = bool(settings.get("use_previous", True))
    delay = float(settings.get("delay", 0.5))
    for it in STATE["items"]:           # always one-by-one, in order
        if STATE["stop"]:
            break
        if it["status"] != "done":
            process_one(api_key, it, settings, use_prev)
            time.sleep(delay)
    with LOCK:
        STATE["running"] = False


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.route("/api/env")
def api_env():
    return jsonify({"ffmpeg": has_ffmpeg()})


@app.route("/api/test_key", methods=["POST"])
def test_key():
    """Check the OpenAI key works by listing models."""
    d = request.json or {}
    key = (d.get("api_key", "") or "").strip()
    if not key:
        return jsonify({"ok": False, "error": "empty key"}), 400
    # Accept any key format from derouter
    if not key:
        return jsonify({"ok": False, "error": "API key is required. Get one at api-direct.derouter.network"})
    try:
        r = requests.get(f"{OPENAI_BASE}/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=30)
        if r.status_code == 200:
            return jsonify({"ok": True})
        detail = r.text[:200]
        try:
            detail = r.json().get("error", {}).get("message", detail)
        except Exception:
            pass
        return jsonify({"ok": False, "error": f"{r.status_code}: {detail}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/start", methods=["POST"])
def api_start():
    if STATE["running"]:
        return jsonify({"error": "Job already running."}), 400
    d = request.json or {}
    key = (d.get("api_key") or "").strip()
    prompts = d.get("prompts") or []
    settings = d.get("settings") or {}
    if not key:
        return jsonify({"error": "Missing OpenAI API key."}), 400
    if not prompts:
        return jsonify({"error": "No prompts."}), 400
    with LOCK:
        STATE.update(running=True, stop=False, total=len(prompts), settings=settings)
        STATE["items"] = [{"idx": i, "prompt": p, "status": "pending",
                           "file": None, "error": None}
                          for i, p in enumerate(prompts)]
    threading.Thread(target=run_job, args=(key, settings), daemon=True).start()
    return jsonify({"ok": True, "total": len(prompts)})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with LOCK:
        STATE["stop"] = True
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    with LOCK:
        items = STATE["items"]
        done = sum(1 for i in items if i["status"] == "done")
        err = sum(1 for i in items if i["status"] == "error")
        return jsonify({"running": STATE["running"], "total": STATE["total"],
                        "done": done, "error": err,
                        "items": [{"idx": i["idx"], "status": i["status"],
                                   "file": i["file"], "error": i["error"]}
                                  for i in items]})


@app.route("/api/regen", methods=["POST"])
def api_regen():
    if STATE["running"]:
        return jsonify({"error": "Stop the job first."}), 400
    d = request.json or {}
    idx = int(d.get("idx", -1))
    key = (d.get("api_key") or "").strip()
    if not (0 <= idx < len(STATE["items"])):
        return jsonify({"error": "bad index"}), 400
    if not key:
        return jsonify({"error": "Missing API key."}), 400
    set_item(idx, status="pending", error=None)
    process_one(key, STATE["items"][idx], STATE["settings"],
                bool(STATE["settings"].get("use_previous", True)))
    return jsonify({"ok": True})


@app.route("/output/<path:fn>")
def serve_output(fn):
    return send_from_directory(OUTPUT_DIR, fn)


@app.route("/api/zip")
def api_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for it in STATE["items"]:
            if it["status"] == "done" and it["file"]:
                fp = OUTPUT_DIR / it["file"]
                if fp.exists():
                    z.write(fp, it["file"])
        z.writestr("prompts.txt", "\n\n".join(
            f"{it['idx']+1:03d} — {it['prompt']}" for it in STATE["items"]))
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name="sketch_reactor_images.zip")


# --------------------------------------------------------------------------
# TTS (MiMo-V2.5-TTS) — Voice Narrator
# --------------------------------------------------------------------------
@app.route("/api/tts", methods=["POST"])
def api_tts():
    d = request.json or {}
    api_key = (d.get("api_key") or "").strip()
    text = (d.get("text") or "").strip()
    voice = d.get("voice", "Mia")
    style_instructions = (d.get("style") or "").strip()
    out_format = d.get("format", "wav")

    if not api_key:
        return jsonify({"error": "TTS API key required."}), 400
    if not text:
        return jsonify({"error": "No text provided."}), 400

    # Build messages: user = style instructions, assistant = text to speak
    messages = []
    if style_instructions:
        messages.append({"role": "user", "content": style_instructions})
    messages.append({"role": "assistant", "content": text})

    body = {
        "model": TTS_MODEL,
        "messages": messages,
        "audio": {"format": out_format, "voice": voice},
        "stream": False,
    }

    headers = {"api-key": api_key, "Content-Type": "application/json"}

    try:
        r = requests.post(f"{TTS_BASE}/chat/completions",
                          headers=headers, json=body, timeout=300)
        if r.status_code != 200:
            detail = r.text[:500]
            try:
                detail = r.json().get("error", {}).get("message", detail)
            except Exception:
                pass
            return jsonify({"error": f"TTS {r.status_code}: {detail}"}), 400

        data = r.json()
        # Extract audio from response
        choices = data.get("choices", [])
        if not choices:
            return jsonify({"error": "No TTS output in response."}), 500

        msg = choices[0].get("message", {})
        audio_data = msg.get("audio", {})
        b64 = audio_data.get("data", "")
        if not b64:
            return jsonify({"error": "No audio data in TTS response.",
                            "raw": json.dumps(data)[:500]}), 500

        audio_bytes = base64.b64decode(b64)
        mime = "audio/wav" if out_format == "wav" else "audio/pcm"
        ext = "wav" if out_format == "wav" else "pcm"

        # Save to output dir
        fname = f"narrator_{int(time.time())}.{ext}"
        (OUTPUT_DIR / fname).write_bytes(audio_bytes)

        return jsonify({"ok": True, "file": fname, "size": len(audio_bytes)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tts_voices")
def api_tts_voices():
    """Return available TTS voices."""
    voices = [
        {"id": "Mia", "name": "Mia", "lang": "English", "gender": "Female"},
        {"id": "Chloe", "name": "Chloe", "lang": "English", "gender": "Female"},
        {"id": "Milo", "name": "Milo", "lang": "English", "gender": "Male"},
        {"id": "Dean", "name": "Dean", "lang": "English", "gender": "Male"},
        {"id": "冰糖", "name": "Bing Tang", "lang": "Chinese", "gender": "Female"},
        {"id": "茉莉", "name": "Mo Li", "lang": "Chinese", "gender": "Female"},
        {"id": "苏打", "name": "Su Da", "lang": "Chinese", "gender": "Male"},
        {"id": "白桦", "name": "Bai Hua", "lang": "Chinese", "gender": "Male"},
    ]
    return jsonify(voices)


# --------------------------------------------------------------------------
# Auto-Narrator: LLM generates narration text + TTS voice in one click
# --------------------------------------------------------------------------
@app.route("/api/auto_narrator", methods=["POST"])
def api_auto_narrator():
    d = request.json or {}
    api_key = (d.get("api_key") or "").strip()
    tts_key = (d.get("tts_key") or "").strip()
    prompts = d.get("prompts") or []
    voice = d.get("voice", "Mia")
    style_instructions = (d.get("style") or "").strip()
    narration_model = d.get("narration_model", "gpt-5.5")
    out_format = d.get("format", "wav")
    language = d.get("language", "english")

    if not api_key:
        return jsonify({"error": "Image API key required (used for narration LLM)."}), 400
    if not tts_key:
        return jsonify({"error": "TTS API key required."}), 400
    if not prompts:
        return jsonify({"error": "No prompts provided."}), 400

    # Step 1: Generate narration text using LLM
    prompt_summary = "\n".join(f"Scene {i+1}: {p}" for i, p in enumerate(prompts))

    system_msg = (
        "You are a professional voice narrator for animated short films. "
        "Write a narration script that flows naturally across all scenes. "
        "Use MiMo TTS expression tags inline for emotional delivery: "
        "[crying], [laughing], [whisper], [shout], [pause], [sigh], [sniffles], "
        "[gasp], [breathing], [trembling], [sobbing]. "
        "The narration should be vivid, emotional, and cinematic. "
        "Keep it concise — roughly 1-2 sentences per scene. "
        f"Write in {language}. "
        "Output ONLY the narration text with expression tags — no headers, no scene numbers, no explanations."
    )

    if style_instructions:
        system_msg += f"\nNarration style: {style_instructions}"

    try:
        llm_r = requests.post(
            f"{OPENAI_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": narration_model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Write the narration for these scenes:\n\n{prompt_summary}"},
                ],
                "temperature": 0.8,
                "max_tokens": 2000,
            },
            timeout=120,
        )
        if llm_r.status_code != 200:
            detail = llm_r.text[:400]
            try:
                detail = llm_r.json().get("error", {}).get("message", detail)
            except Exception:
                pass
            return jsonify({"error": f"Narration LLM {llm_r.status_code}: {detail}"}), 400

        narration_text = llm_r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return jsonify({"error": f"Narration generation failed: {e}"}), 500

    # Step 2: Convert narration to voice via TTS
    messages = []
    if style_instructions:
        messages.append({"role": "user", "content": style_instructions})
    messages.append({"role": "assistant", "content": narration_text})

    tts_body = {
        "model": TTS_MODEL,
        "messages": messages,
        "audio": {"format": out_format, "voice": voice},
        "stream": False,
    }

    try:
        tts_r = requests.post(
            f"{TTS_BASE}/chat/completions",
            headers={"api-key": tts_key, "Content-Type": "application/json"},
            json=tts_body,
            timeout=300,
        )
        if tts_r.status_code != 200:
            detail = tts_r.text[:400]
            try:
                detail = tts_r.json().get("error", {}).get("message", detail)
            except Exception:
                pass
            return jsonify({"error": f"TTS {tts_r.status_code}: {detail}",
                            "narration": narration_text}), 400

        tts_data = tts_r.json()
        choices = tts_data.get("choices", [])
        if not choices:
            return jsonify({"error": "No TTS output.", "narration": narration_text}), 500

        audio_b64 = choices[0].get("message", {}).get("audio", {}).get("data", "")
        if not audio_b64:
            return jsonify({"error": "No audio data in TTS response.",
                            "narration": narration_text}), 500

        audio_bytes = base64.b64decode(audio_b64)
        ext = "wav" if out_format == "wav" else "pcm"
        fname = f"narrator_auto_{int(time.time())}.{ext}"
        (OUTPUT_DIR / fname).write_bytes(audio_bytes)

        return jsonify({
            "ok": True,
            "narration": narration_text,
            "file": fname,
            "size": len(audio_bytes),
        })

    except Exception as e:
        return jsonify({"error": f"TTS failed: {e}", "narration": narration_text}), 500


if __name__ == "__main__":
    print("\n  Sketch Reactor v4 (OpenAI / GPT Image 2 + MiMo TTS + Auto-Narrator)  ->  http://localhost:5000")
    print("  ffmpeg detected:", has_ffmpeg(), "\n")
    app.run(host="0.0.0.0", port=5000, threaded=True)
