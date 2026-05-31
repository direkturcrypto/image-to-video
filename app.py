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
CHAR_DIR = APP_DIR / "characters"
for d in (OUTPUT_DIR, FRAMES_DIR, ANCHOR_DIR, CHAR_DIR):
    d.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=None)

STATE = {"running": False, "stop": False, "total": 0, "items": [], "settings": {}}
LOCK = threading.Lock()

# Step 7 video-build job state
VIDEO = {"running": False, "stage": "", "done": 0, "total": 0,
         "file": None, "error": None, "narration": []}
VLOCK = threading.Lock()

OPENAI_BASE = "https://api-direct.derouter.network/openai/v1"

# TTS config (MiMo-V2.5-TTS via Xiaomi)
TTS_BASE = "https://token-plan-sgp.xiaomimimo.com/v1"
TTS_MODEL = "mimo-v2.5-tts"   # NOT "miplan/..." — that returns 400 "Not supported model"

# Default LLM (on derouter, OpenAI-compatible /chat/completions) used to turn a
# title into scene prompts and to write narration. Editable from the UI.
NARRATION_MODEL = "claude-opus-4-6"

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
# Shared LLM + TTS helpers
# --------------------------------------------------------------------------
def chat_llm(api_key, model, messages, temperature=0.8, max_tokens=2000):
    """Call the derouter OpenAI-compatible /chat/completions endpoint.

    Returns the assistant message text. Raises RuntimeError on failure.
    Works for both GPT and Claude models routed through derouter.
    """
    r = requests.post(
        f"{OPENAI_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={"model": model, "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens},
        timeout=180,
    )
    if r.status_code != 200:
        detail = r.text[:400]
        try:
            detail = r.json().get("error", {}).get("message", detail)
        except Exception:
            pass
        raise RuntimeError(f"LLM {r.status_code}: {detail}")
    return r.json()["choices"][0]["message"]["content"].strip()


def tts_synthesize(tts_key, text, voice="Mia", style="", out_format="wav",
                   model=None):
    """Synthesize speech with MiMo-V2.5-TTS. Returns raw audio bytes.

    The MiMo API requires BOTH a user message (style instructions, may be
    empty) AND an assistant message (the text to speak). Sending only the
    assistant message is the most common cause of HTTP 400 — so we always
    include the user message here.
    """
    messages = [
        {"role": "user", "content": style},          # always present (may be "")
        {"role": "assistant", "content": text},      # text that gets spoken
    ]
    body = {
        "model": model or TTS_MODEL,
        "messages": messages,
        "audio": {"format": out_format, "voice": voice},
    }
    r = requests.post(f"{TTS_BASE}/chat/completions",
                      headers={"api-key": tts_key,
                               "Content-Type": "application/json"},
                      json=body, timeout=300)
    if r.status_code != 200:
        detail = r.text[:500]
        try:
            err = r.json().get("error", {})
            if isinstance(err, dict):
                msg = err.get("message", detail)
                # MiMo puts the offending field in `param` — surface it!
                param = err.get("param")
                detail = f"{msg} (param: {param})" if param else msg
            else:
                detail = str(err) or detail
        except Exception:
            pass
        raise RuntimeError(f"TTS {r.status_code}: {detail}")
    choices = r.json().get("choices", [])
    if not choices:
        raise RuntimeError("TTS returned no choices.")
    b64 = choices[0].get("message", {}).get("audio", {}).get("data", "")
    if not b64:
        raise RuntimeError("TTS response had no audio data.")
    return base64.b64decode(b64)


PROJECT_FILE = OUTPUT_DIR / "project.json"


def save_project(prompts, settings=None):
    """Persist the current prompt list (and settings) next to the images."""
    try:
        PROJECT_FILE.write_text(json.dumps(
            {"prompts": prompts, "settings": settings or {},
             "saved_at": int(time.time())}, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[WARN] could not save project: {e}")


def load_project():
    """Return saved prompts/settings, or empty defaults if none exist."""
    if PROJECT_FILE.exists():
        try:
            return json.loads(PROJECT_FILE.read_text())
        except Exception:
            pass
    return {"prompts": [], "settings": {}}


def output_images():
    """Sorted list of generated image filenames (001.png, 002.png, ...)."""
    return sorted(p.name for p in OUTPUT_DIR.glob("[0-9][0-9][0-9].png"))


# --------------------------------------------------------------------------
# Characters (reusable GPT-Image character reference sheets)
# --------------------------------------------------------------------------
def _slug(name):
    s = "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")
    s = "-".join(filter(None, s.split("-")))
    return s[:40] or "character"


def list_characters():
    out = []
    for d in sorted(CHAR_DIR.glob("*/")):
        meta = d / "meta.json"
        sheet = d / "sheet.png"
        if meta.exists() and sheet.exists():
            try:
                m = json.loads(meta.read_text())
            except Exception:
                continue
            out.append({"id": d.name, "name": m.get("name", d.name),
                        "description": m.get("description", ""),
                        "sheet": f"/characters/{d.name}/sheet.png"})
    return out


def load_character(cid):
    """Return {description, sheet_path} for a character id, or None."""
    if not cid:
        return None
    d = CHAR_DIR / cid
    meta = d / "meta.json"
    sheet = d / "sheet.png"
    if not (meta.exists() and sheet.exists()):
        return None
    try:
        m = json.loads(meta.read_text())
    except Exception:
        m = {}
    return {"description": m.get("description", ""), "sheet_path": str(sheet)}


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
    char_desc = settings.get("character_desc", "").strip()
    out = prompt.strip()
    if char_desc:
        out += ("\n\nMAIN CHARACTER (keep identical to the character reference "
                "sheet provided): " + char_desc)
    if style:
        out += "\n\nART STYLE (match exactly): " + style
    if negative:
        # semantic negative: describe what to avoid
        out += "\nAvoid: " + negative
    out += ("\nKeep character design, proportions, line weight, and colors "
            "identical to the reference image(s) provided.")
    return out


def generate_image(api_key, prompt, settings, ref_paths, raw_prompt=False):
    """Call OpenAI images API, return PNG bytes. Raises on failure.

    If reference images are present, use /v1/images/edits (multipart) so the
    style anchors + previous image guide the output. Otherwise use the plain
    /v1/images/generations JSON endpoint. When raw_prompt is True the prompt is
    sent verbatim (used by the character-sheet builder).
    """
    model = settings.get("model", "gpt-image-2")
    size = settings.get("size", "1024x1024")
    quality = settings.get("quality", "low")
    full_prompt = prompt if raw_prompt else build_prompt(prompt, settings)
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
    # character reference sheet (keeps the same character across scenes)
    sheet = settings.get("character_sheet")
    if sheet and Path(sheet).exists():
        refs = refs + [sheet]
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
    # resolve selected character -> inject its sheet + description into settings
    char = load_character(settings.get("character"))
    if char:
        settings["character_desc"] = char["description"]
        settings["character_sheet"] = char["sheet_path"]
    with LOCK:
        STATE.update(running=True, stop=False, total=len(prompts), settings=settings)
        STATE["items"] = [{"idx": i, "prompt": p, "status": "pending",
                           "file": None, "error": None}
                          for i, p in enumerate(prompts)]
    # Persist the prompts so "Load from Prev Project" can restore them later.
    save_project(prompts, settings)
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

    tts_model = (d.get("tts_model") or "").strip() or None

    if not api_key:
        return jsonify({"error": "TTS API key required."}), 400
    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        audio_bytes = tts_synthesize(api_key, text, voice,
                                     style_instructions, out_format, tts_model)
        ext = "wav" if out_format == "wav" else "pcm"
        fname = f"narrator_{int(time.time())}.{ext}"
        (OUTPUT_DIR / fname).write_bytes(audio_bytes)
        return jsonify({"ok": True, "file": fname, "size": len(audio_bytes)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


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
        narration_text = chat_llm(
            api_key, narration_model,
            [{"role": "system", "content": system_msg},
             {"role": "user", "content": f"Write the narration for these scenes:\n\n{prompt_summary}"}],
        )
    except Exception as e:
        return jsonify({"error": f"Narration generation failed: {e}"}), 500

    # Step 2: Convert narration to voice via TTS
    try:
        audio_bytes = tts_synthesize(tts_key, narration_text, voice,
                                     style_instructions, out_format)
    except Exception as e:
        return jsonify({"error": str(e), "narration": narration_text}), 400

    ext = "wav" if out_format == "wav" else "pcm"
    fname = f"narrator_auto_{int(time.time())}.{ext}"
    (OUTPUT_DIR / fname).write_bytes(audio_bytes)
    return jsonify({"ok": True, "narration": narration_text,
                    "file": fname, "size": len(audio_bytes)})


# --------------------------------------------------------------------------
# Character Builder routes
# --------------------------------------------------------------------------
@app.route("/characters/<path:fn>")
def serve_character(fn):
    return send_from_directory(CHAR_DIR, fn)


@app.route("/api/characters")
def api_characters():
    return jsonify({"ok": True, "characters": list_characters()})


@app.route("/api/generate_character", methods=["POST"])
def api_generate_character():
    d = request.json or {}
    api_key = (d.get("api_key") or "").strip()
    name = (d.get("name") or "").strip()
    description = (d.get("description") or "").strip()
    settings = d.get("settings") or {}
    if not api_key:
        return jsonify({"error": "API key required."}), 400
    if not name or not description:
        return jsonify({"error": "Character name and description required."}), 400

    style = settings.get("style_suffix", "").strip()
    # Dedicated character-sheet prompt (sent verbatim — not the scene builder).
    sheet_prompt = (
        "Character reference sheet for 2D animation. Show ONE single character "
        "in 3 full-body poses (front view, three-quarter view, and a simple "
        "action pose) plus one head-and-shoulders close-up, all clearly the SAME "
        "character, evenly spaced on a plain solid white background with no "
        "scenery or props. "
        f"Character: {description}. ")
    if style:
        sheet_prompt += f"Art rendering style: {style}. "
    sheet_prompt += "Consistent proportions and identical line weight throughout."

    try:
        # use existing style anchors (if any) as a style reference
        png = generate_image(api_key, sheet_prompt, settings,
                             anchor_paths(), raw_prompt=True)
    except Exception as e:
        return jsonify({"error": f"Character generation failed: {e}"}), 400

    cid = _slug(name)
    base = cid
    i = 2
    while (CHAR_DIR / cid).exists():        # avoid clobbering an existing one
        cid = f"{base}-{i}"
        i += 1
    cdir = CHAR_DIR / cid
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "sheet.png").write_bytes(png)
    (cdir / "meta.json").write_text(json.dumps(
        {"id": cid, "name": name, "description": description}, ensure_ascii=False))
    return jsonify({"ok": True, "character": {
        "id": cid, "name": name, "description": description,
        "sheet": f"/characters/{cid}/sheet.png"}})


@app.route("/api/delete_character", methods=["POST"])
def api_delete_character():
    cid = ((request.json or {}).get("id") or "").strip()
    d = CHAR_DIR / cid
    if cid and d.exists() and d.is_dir() and d.parent == CHAR_DIR:
        shutil.rmtree(d, ignore_errors=True)
        return jsonify({"ok": True})
    return jsonify({"error": "Character not found."}), 404


# --------------------------------------------------------------------------
# Model list (so the UI can show what derouter actually offers)
# --------------------------------------------------------------------------
@app.route("/api/models", methods=["POST"])
def api_models():
    key = ((request.json or {}).get("api_key") or "").strip()
    if not key:
        return jsonify({"error": "API key required."}), 400
    try:
        r = requests.get(f"{OPENAI_BASE}/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=30)
        if r.status_code != 200:
            return jsonify({"error": f"{r.status_code}: {r.text[:200]}"}), 400
        data = r.json().get("data", [])
        ids = sorted(m.get("id", "") for m in data if m.get("id"))
        return jsonify({"ok": True, "models": ids})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------------------------------
# Step 3: turn a Title into scene prompts using Claude
# --------------------------------------------------------------------------
@app.route("/api/generate_prompts", methods=["POST"])
def api_generate_prompts():
    d = request.json or {}
    api_key = (d.get("api_key") or "").strip()
    title = (d.get("title") or "").strip()
    count = max(1, min(int(d.get("count", 8) or 8), 60))
    language = d.get("language", "english")
    style_hint = (d.get("style_hint") or "").strip()
    model = (d.get("model") or "").strip() or NARRATION_MODEL

    if not api_key:
        return jsonify({"error": "API key required."}), 400
    if not title:
        return jsonify({"error": "Title required."}), 400

    system_msg = (
        "You are a master visual storyteller for immersive, cinematic narrated "
        "story videos — the gripping second-person style of YouTube channels "
        "like 'Lost Legacy' (e.g. 'Your Life as a ...'). "
        "Given only a TITLE, craft ONE continuous STORY with a clear arc: a hook, "
        "rising tension, a turning point, and a resonant ending — then break it "
        f"into EXACTLY {count} sequential scenes. "
        "For each scene write ONE vivid image-generation prompt describing only "
        "what we SEE in that single moment: setting, the main character(s) and "
        "their expression + action, camera framing (close-up / wide / over-the-"
        "shoulder), time of day, lighting and mood. "
        "Keep the SAME main character(s) and world visually consistent and "
        "recognizable from scene to scene, and make each scene clearly advance "
        "the story. Do NOT mention art style (handled separately) and do NOT "
        "include narration, captions or dialogue text. "
        f"Write the prompts in {language}. "
        f"Return EXACTLY {count} prompts as a JSON array of strings and nothing "
        "else — no numbering, no keys, no markdown fences."
    )
    if style_hint:
        system_msg += f"\nStory/tone hint: {style_hint}"

    try:
        raw = chat_llm(api_key, model,
                       [{"role": "system", "content": system_msg},
                        {"role": "user", "content": f"TITLE: {title}\n\nGenerate {count} scene prompts."}],
                       temperature=0.9, max_tokens=4000)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    prompts = _parse_prompt_list(raw, count)
    if not prompts:
        return jsonify({"error": "Could not parse prompts from model output.",
                        "raw": raw[:500]}), 500
    return jsonify({"ok": True, "prompts": prompts})


def _parse_prompt_list(raw, count):
    """Best-effort extraction of a list of prompts from an LLM response."""
    text = raw.strip()
    # strip ```json fences if present
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
        text = text.lstrip("json").strip("`\n ")
    # try JSON array first
    try:
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1:
            arr = json.loads(text[start:end + 1])
            items = [str(x).strip() for x in arr if str(x).strip()]
            if items:
                return items
    except Exception:
        pass
    # fall back: split on blank lines / numbered lines
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if len(blocks) < 2:
        blocks = [b.strip() for b in text.split("\n") if b.strip()]
    cleaned = []
    for b in blocks:
        # drop leading "1.", "1)", "- ", "Scene 1:" style prefixes
        b = b.lstrip("-*0123456789.)• ").strip()
        for sep in ("Scene", "scene"):
            if b.startswith(sep) and ":" in b[:12]:
                b = b.split(":", 1)[1].strip()
        if b:
            cleaned.append(b)
    return cleaned[:count] if cleaned else []


# --------------------------------------------------------------------------
# Step: Load from previous project (reuse already-generated images)
# --------------------------------------------------------------------------
@app.route("/api/load_previous")
def api_load_previous():
    imgs = output_images()
    proj = load_project()
    return jsonify({"ok": True, "images": imgs,
                    "prompts": proj.get("prompts", []),
                    "settings": proj.get("settings", {})})


# --------------------------------------------------------------------------
# Step 7: combine images + per-scene voice into one synced video
# --------------------------------------------------------------------------
def _vset(**kw):
    with VLOCK:
        VIDEO.update(kw)


def build_video_job(api_key, tts_key, prompts, voice, style, language,
                    narration_model, tts_model):
    try:
        imgs = output_images()
        if not imgs:
            _vset(running=False, error="No generated images in output/ yet.")
            return
        # one image per scene; pair with prompts where available
        n = len(imgs)
        scene_prompts = (prompts + [""] * n)[:n] if prompts else [""] * n
        _vset(total=n, done=0, stage="writing narration", narration=[])

        # 1) one Claude call -> N narration lines (flows across scenes)
        prompt_summary = "\n".join(
            f"Scene {i+1}: {p or '(image '+imgs[i]+')'}" for i, p in enumerate(scene_prompts))
        system_msg = (
            "You are the narrator of a FAST-PACED, punchy short story video — the "
            "gripping style of YouTube channels like 'Lost Legacy', but cut quick. "
            "Write a voiceover for a fast slideshow where each image is on screen "
            "only ~2 seconds, so each line must be ONE very short, punchy beat of "
            "about 3-7 words (a short phrase or clause, ~1.5-2.5 seconds when "
            "spoken aloud) — NOT a full sentence. "
            "Across all the lines it must still read as ONE continuous story: a "
            "hook first, rising tension through the middle, and a punchy or "
            "emotional beat last. Vivid and immersive; second-person 'you' works "
            "great when it fits the title. Write PLAIN narration text only — do "
            "NOT use any expression tags or bracketed cues of any kind. "
            f"Write in {language}. "
            f"Return EXACTLY {n} short lines as a JSON array of strings, one per "
            "scene, and nothing else.")
        if style:
            system_msg += f"\nNarration style / tone: {style}"
        raw = chat_llm(api_key, narration_model,
                       [{"role": "system", "content": system_msg},
                        {"role": "user", "content": f"Scenes:\n\n{prompt_summary}"}],
                       temperature=0.8, max_tokens=4000)
        lines = _parse_prompt_list(raw, n)
        if len(lines) < n:                       # pad if the model gave too few
            lines += [""] * (n - len(lines))
        lines = lines[:n]
        _vset(narration=lines, stage="synthesizing voice")

        work = OUTPUT_DIR / "_video_tmp"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir()

        # 2) per-scene TTS -> 3) per-scene ffmpeg segment (duration == audio)
        segments = []
        for i, (img, line) in enumerate(zip(imgs, lines)):
            with VLOCK:
                if VIDEO.get("stop"):
                    break
            _vset(stage=f"voice {i+1}/{n}", done=i)
            speak = line.strip() or " "
            audio = tts_synthesize(tts_key, speak, voice, style, "wav", tts_model)
            apath = work / f"seg_{i:03d}.wav"
            apath.write_bytes(audio)
            seg = work / f"seg_{i:03d}.mp4"
            # image duration = its own audio length (perfect sync, no cut/stretch)
            cmd = [
                "ffmpeg", "-v", "error", "-y",
                "-loop", "1", "-i", str(OUTPUT_DIR / img),
                "-i", str(apath),
                "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
                "-shortest", str(seg),
            ]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0 or not seg.exists():
                raise RuntimeError(f"ffmpeg segment {i+1} failed: "
                                   f"{r.stderr.decode()[:300]}")
            segments.append(seg)

        if not segments:
            _vset(running=False, error="No segments built.")
            return

        # 4) concat all segments into the final video
        _vset(stage="stitching video", done=n)
        listfile = work / "list.txt"
        listfile.write_text("".join(f"file '{s.name}'\n" for s in segments))
        out = OUTPUT_DIR / f"final_video_{int(time.time())}.mp4"
        cc = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", str(listfile), "-c", "copy", str(out)],
            cwd=str(work), capture_output=True)
        if cc.returncode != 0 or not out.exists():
            raise RuntimeError("ffmpeg concat failed: " + cc.stderr.decode()[:300])

        shutil.rmtree(work, ignore_errors=True)
        _vset(running=False, stage="done", done=n, file=out.name, error=None)
    except Exception as e:
        _vset(running=False, error=str(e))


@app.route("/api/build_video", methods=["POST"])
def api_build_video():
    if not has_ffmpeg():
        return jsonify({"error": "ffmpeg not found — required to build video."}), 400
    if VIDEO["running"]:
        return jsonify({"error": "Video build already running."}), 400
    d = request.json or {}
    api_key = (d.get("api_key") or "").strip()
    tts_key = (d.get("tts_key") or "").strip()
    prompts = d.get("prompts") or load_project().get("prompts", [])
    voice = d.get("voice", "Mia")
    style = (d.get("style") or "").strip()
    language = d.get("language", "english")
    narration_model = (d.get("narration_model") or "").strip() or NARRATION_MODEL
    tts_model = (d.get("tts_model") or "").strip() or None

    if not api_key:
        return jsonify({"error": "Image/LLM API key required (Section 1)."}), 400
    if not tts_key:
        return jsonify({"error": "TTS API key required (Section 5)."}), 400
    if not output_images():
        return jsonify({"error": "No generated images found in output/."}), 400

    with VLOCK:
        VIDEO.update(running=True, stop=False, stage="starting", done=0,
                     total=len(output_images()), file=None, error=None,
                     narration=[])
    threading.Thread(target=build_video_job,
                     args=(api_key, tts_key, prompts, voice, style, language,
                           narration_model, tts_model), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/video_status")
def api_video_status():
    with VLOCK:
        return jsonify(dict(VIDEO))


if __name__ == "__main__":
    print("\n  Sketch Reactor v4 (OpenAI / GPT Image 2 + MiMo TTS + Auto-Narrator)  ->  http://localhost:5000")
    print("  ffmpeg detected:", has_ffmpeg(), "\n")
    app.run(host="0.0.0.0", port=5001, threaded=True)
