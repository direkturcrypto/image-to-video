"""
Sketch Reactor — core engine (no web framework).

All the real work lives here so both the Flask web app (app.py) and the CLI
(cli.py) share one implementation:

  * chat_llm / tts_synthesize        — derouter LLM + MiMo TTS calls
  * generate_image / process_one     — GPT Image 2 generation (with refs)
  * generate_all                     — generate a whole prompt list, in order
  * generate_scene_prompts           — title -> scene prompts (Claude)
  * generate_character_sheet         — reusable character reference sheets
  * build_video                      — per-scene narrated, synced MP4

Functions take plain arguments and optional callbacks (on_status / on_progress
/ should_stop) so the web layer can wire them to its job state and the CLI can
print progress — neither owns the logic.

API keys are passed in explicitly (never read from globals here).
"""

import sys
import time
import json
import base64
import shutil
import zipfile
import subprocess
from pathlib import Path

import requests

APP_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = APP_DIR / "output"
FRAMES_DIR = APP_DIR / "frames"
ANCHOR_DIR = APP_DIR / "anchors"
CHAR_DIR = APP_DIR / "characters"
for d in (OUTPUT_DIR, FRAMES_DIR, ANCHOR_DIR, CHAR_DIR):
    d.mkdir(exist_ok=True)

OPENAI_BASE = "https://api-direct.derouter.network/openai/v1"

# TTS config (MiMo-V2.5-TTS via Xiaomi)
TTS_BASE = "https://token-plan-sgp.xiaomimimo.com/v1"
TTS_MODEL = "mimo-v2.5-tts"   # NOT "miplan/..." — that returns 400 "Not supported model"

# Default LLM (derouter, OpenAI-compatible /chat/completions) for prompts/narration.
NARRATION_MODEL = "claude-opus-4-6"

# Default art style (the "stickman" doodle look) + negative — same as the web UI.
# Used when the caller doesn't pass its own --style / --negative.
DEFAULT_STYLE = (
    "Hand-drawn 2D doodle animation style. Stick-figure characters with round "
    "white heads, large expressive cartoon eyes and eyebrows, thin bold black "
    "outlines, simple thin stick limbs. Flat solid color fills with no "
    "gradients: deep blue night sky with small white stars, flat brown ground, "
    "bright orange campfire flames. Clean, simple, consistent character "
    "proportions and identical line weight in every frame.")
DEFAULT_NEGATIVE = ("no photorealism, no 3d render, no shading gradients, "
                    "no extra detail, no text watermark")

IMAGE_MODELS = {"gpt-image-2", "gpt-image-1.5", "gpt-image-1-mini"}

COST_PER_IMG = {
    "gpt-image-2": {"low": 0.02, "medium": 0.07, "high": 0.19},
    "gpt-image-1.5": {"low": 0.015, "medium": 0.05, "high": 0.15},
    "gpt-image-1-mini": {"low": 0.005, "medium": 0.01, "high": 0.02},
}

VOICES = [
    {"id": "Mia", "name": "Mia", "lang": "English", "gender": "Female"},
    {"id": "Chloe", "name": "Chloe", "lang": "English", "gender": "Female"},
    {"id": "Milo", "name": "Milo", "lang": "English", "gender": "Male"},
    {"id": "Dean", "name": "Dean", "lang": "English", "gender": "Male"},
    {"id": "冰糖", "name": "Bing Tang", "lang": "Chinese", "gender": "Female"},
    {"id": "茉莉", "name": "Mo Li", "lang": "Chinese", "gender": "Female"},
    {"id": "苏打", "name": "Su Da", "lang": "Chinese", "gender": "Male"},
    {"id": "白桦", "name": "Bai Hua", "lang": "Chinese", "gender": "Male"},
]

PROJECT_FILE = OUTPUT_DIR / "project.json"


def _log(*a):
    print(*a, file=sys.stderr)


def has_ffmpeg():
    return shutil.which("ffmpeg") is not None


# --------------------------------------------------------------------------
# Big bold burned-in captions (viral style) — rendered with Pillow, because
# this ffmpeg build has no drawtext/libass.
# --------------------------------------------------------------------------
# Prefer punchy bold fonts; fall back across macOS / Linux / Windows.
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _load_font(size):
    from PIL import ImageFont
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _wrap(draw, text, font, stroke, maxw):
    """Greedy word-wrap so each line (incl. stroke) fits within maxw."""
    lines, cur = [], ""
    for w in text.split(" "):
        t = (cur + " " + w).strip()
        if not cur or draw.textlength(t, font=font) + 2 * stroke <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def caption_image(src_png, text, dest_png, position="bottom"):
    """Draw a bold caption (white, thick black outline, uppercase, centered)
    onto a copy of src_png. The font size AUTO-FITS: it shrinks until the
    wrapped text fits a bounded box in the lower band — so long lines never
    cover the picture and short lines aren't oversized."""
    from PIL import Image, ImageDraw
    im = Image.open(src_png).convert("RGB")
    text = " ".join((text or "").split()).upper()
    if not text:
        im.save(dest_png)
        return
    W, H = im.size
    d = ImageDraw.Draw(im)
    maxw = W * 0.86                 # usable text width
    box_h = H * 0.30                # caption may fill at most ~30% of height
    max_size = max(20, int(H * 0.070))
    min_size = max(14, int(H * 0.028))

    chosen = None
    size = max_size
    while size >= min_size:
        font = _load_font(size)
        stroke = max(2, int(size * 0.14))
        lines = _wrap(d, text, font, stroke, maxw)
        asc, desc = font.getmetrics()
        lh = asc + desc + int(size * 0.18)
        block = lh * len(lines)
        widest = max(d.textlength(ln, font=font) + 2 * stroke for ln in lines)
        if block <= box_h and widest <= maxw:
            chosen = (font, stroke, lines, lh, block)
            break
        size -= max(2, int(size * 0.08))
    if chosen is None:              # extreme: settle for the smallest size
        font = _load_font(min_size)
        stroke = max(2, int(min_size * 0.14))
        lines = _wrap(d, text, font, stroke, maxw)
        asc, desc = font.getmetrics()
        lh = asc + desc + int(min_size * 0.18)
        chosen = (font, stroke, lines, lh, lh * len(lines))

    font, stroke, lines, lh, block = chosen
    y0 = (H - block) // 2 if position == "center" else int(H * 0.96 - block)
    for i, ln in enumerate(lines):
        tw = d.textlength(ln, font=font) + 2 * stroke
        d.text(((W - tw) / 2, y0 + i * lh), ln, font=font, fill="white",
               stroke_width=stroke, stroke_fill="black")
    im.save(dest_png)


# --------------------------------------------------------------------------
# LLM + TTS
# --------------------------------------------------------------------------
def chat_llm(api_key, model, messages, temperature=0.8, max_tokens=2000):
    """Call the derouter OpenAI-compatible /chat/completions endpoint.

    Returns the assistant message text. Raises RuntimeError on failure.
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


def list_models(api_key):
    """Sorted list of model ids derouter offers. Raises on failure."""
    r = requests.get(f"{OPENAI_BASE}/models",
                     headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
    return sorted(m.get("id", "") for m in r.json().get("data", []) if m.get("id"))


def test_key(api_key):
    """True if the key can list models (auth works)."""
    try:
        list_models(api_key)
        return True, None
    except Exception as e:
        return False, str(e)


def tts_synthesize(tts_key, text, voice="Mia", style="", out_format="wav",
                   model=None):
    """Synthesize speech with MiMo-V2.5-TTS. Returns raw audio bytes.

    The MiMo API requires BOTH a user message (style, may be empty) AND an
    assistant message (the text to speak) — sending only the assistant message
    is the most common cause of HTTP 400.
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
                param = err.get("param")          # MiMo names the bad field here
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


# --------------------------------------------------------------------------
# Project + outputs
# --------------------------------------------------------------------------
def save_project(prompts, settings=None):
    try:
        PROJECT_FILE.write_text(json.dumps(
            {"prompts": prompts, "settings": settings or {},
             "saved_at": int(time.time())}, ensure_ascii=False, indent=2))
    except Exception as e:
        _log(f"[WARN] could not save project: {e}")


def load_project():
    if PROJECT_FILE.exists():
        try:
            return json.loads(PROJECT_FILE.read_text())
        except Exception:
            pass
    return {"prompts": [], "settings": {}}


def output_images():
    """Sorted generated image filenames (001.png, 002.png, ...)."""
    return sorted(p.name for p in OUTPUT_DIR.glob("[0-9][0-9][0-9].png"))


# --------------------------------------------------------------------------
# Style anchors from a sample video
# --------------------------------------------------------------------------
def anchor_paths():
    return sorted(str(p) for p in ANCHOR_DIR.glob("anchor_*.jpg"))


def extract_frames(video_path, n=9):
    """Pull n evenly-spaced frames from a video into FRAMES_DIR. Returns names."""
    if not has_ffmpeg():
        raise RuntimeError("ffmpeg not found.")
    for p in FRAMES_DIR.glob("*.jpg"):
        p.unlink()
    try:
        dur = float(subprocess.check_output([
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(video_path)]).decode().strip())
    except Exception:
        dur = 60.0
    saved = []
    for i in range(n):
        t = max(0.5, dur * (i + 0.5) / n)
        out = FRAMES_DIR / f"frame_{i:02d}.jpg"
        subprocess.run(["ffmpeg", "-v", "quiet", "-ss", str(t), "-i", str(video_path),
                        "-frames:v", "1", "-vf", "scale=512:-1", "-y", str(out)])
        if out.exists():
            saved.append(out.name)
    return saved


def set_anchors(frame_filenames):
    """Copy up to 2 chosen frames into ANCHOR_DIR as style anchors."""
    for p in ANCHOR_DIR.glob("*"):
        p.unlink()
    out = []
    for i, fn in enumerate(frame_filenames[:2]):    # cap 2 (leave room for prev img)
        src = FRAMES_DIR / fn
        if src.exists():
            dst = ANCHOR_DIR / f"anchor_{i}.jpg"
            shutil.copy(src, dst)
            out.append(dst.name)
    return out


def set_anchor_files(paths):
    """Copy arbitrary image files in as anchors (used by the CLI)."""
    for p in ANCHOR_DIR.glob("*"):
        p.unlink()
    out = []
    for i, src in enumerate(paths[:2]):
        src = Path(src)
        if src.exists():
            dst = ANCHOR_DIR / f"anchor_{i}.jpg"
            shutil.copy(src, dst)
            out.append(dst.name)
    return out


# --------------------------------------------------------------------------
# Characters
# --------------------------------------------------------------------------
def _slug(name):
    s = "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")
    s = "-".join(filter(None, s.split("-")))
    return s[:40] or "character"


def list_characters():
    out = []
    for d in sorted(CHAR_DIR.glob("*/")):
        meta, sheet = d / "meta.json", d / "sheet.png"
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
    meta, sheet = d / "meta.json", d / "sheet.png"
    if not (meta.exists() and sheet.exists()):
        return None
    try:
        m = json.loads(meta.read_text())
    except Exception:
        m = {}
    return {"description": m.get("description", ""), "sheet_path": str(sheet)}


def delete_character(cid):
    d = CHAR_DIR / cid
    if cid and d.exists() and d.is_dir() and d.parent == CHAR_DIR:
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False


def resolve_character(settings):
    """Inject the selected character's sheet path + description into settings."""
    char = load_character(settings.get("character"))
    if char:
        settings["character_desc"] = char["description"]
        settings["character_sheet"] = char["sheet_path"]
    return settings


def generate_character_sheet(api_key, name, description, settings):
    """Generate + save a character reference sheet. Returns the character dict."""
    style = (settings.get("style_suffix") or "").strip()
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

    png = generate_image(api_key, sheet_prompt, settings, anchor_paths(),
                         raw_prompt=True)

    cid = base = _slug(name)
    i = 2
    while (CHAR_DIR / cid).exists():
        cid = f"{base}-{i}"
        i += 1
    cdir = CHAR_DIR / cid
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "sheet.png").write_bytes(png)
    (cdir / "meta.json").write_text(json.dumps(
        {"id": cid, "name": name, "description": description}, ensure_ascii=False))
    return {"id": cid, "name": name, "description": description,
            "sheet": f"/characters/{cid}/sheet.png"}


# --------------------------------------------------------------------------
# Image generation
# --------------------------------------------------------------------------
def build_prompt(prompt, settings):
    style = (settings.get("style_suffix") or "").strip()
    negative = (settings.get("negative") or "").strip()
    char_desc = (settings.get("character_desc") or "").strip()
    out = prompt.strip()
    if char_desc:
        out += ("\n\nMAIN CHARACTER (keep identical to the character reference "
                "sheet provided): " + char_desc)
    if style:
        out += "\n\nART STYLE (match exactly): " + style
    if negative:
        out += "\nAvoid: " + negative
    out += ("\nKeep character design, proportions, line weight, and colors "
            "identical to the reference image(s) provided.")
    return out


def generate_image(api_key, prompt, settings, ref_paths, raw_prompt=False):
    """Call the images API, return PNG bytes. Raises on failure.

    With reference images -> /images/edits (multipart); otherwise the plain
    /images/generations JSON endpoint. raw_prompt sends the prompt verbatim.
    """
    model = settings.get("model", "gpt-image-2")
    size = settings.get("size", "1024x1024")
    quality = settings.get("quality", "low")
    full_prompt = prompt if raw_prompt else build_prompt(prompt, settings)
    headers = {"Authorization": f"Bearer {api_key}"}

    refs = [p for p in ref_paths if Path(p).exists()][:4]

    if refs:
        url = f"{OPENAI_BASE}/images/edits"
        data = {"model": model, "prompt": full_prompt,
                "size": size, "quality": quality, "n": "1"}
        _log(f"[DEBUG] {url} model={model} size={size} quality={quality} refs={len(refs)}")
        files, open_handles = [], []
        try:
            for p in refs:
                fh = open(p, "rb")
                open_handles.append(fh)
                mime = "image/jpeg" if str(p).lower().endswith((".jpg", ".jpeg")) else "image/png"
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
        _log(f"[ERROR] image generation failed: {r.status_code} - {detail}")
        raise RuntimeError(f"{r.status_code}: {detail}"
                           + (f" (retry-after={retry})" if retry else ""))

    data = r.json()
    arr = data.get("data") or []
    if arr and arr[0].get("b64_json"):
        return base64.b64decode(arr[0]["b64_json"])
    if arr and arr[0].get("url"):
        img = requests.get(arr[0]["url"], timeout=120)
        if img.status_code == 200:
            return img.content
    raise RuntimeError("No image in response: " + json.dumps(data)[:300])


def process_one(api_key, idx, prompt, settings, use_prev=True, retries=3,
                on_status=None, should_stop=None):
    """Generate ONE image (1-based file idx+1), with retries/backoff.

    on_status(idx, status=..., file=..., error=...) is called on each change.
    Returns the final {status, file, error} dict.
    """
    def status(**kw):
        if on_status:
            on_status(idx, **kw)

    refs = anchor_paths()
    sheet = settings.get("character_sheet")
    if sheet and Path(sheet).exists():
        refs = refs + [sheet]
    if use_prev and idx > 0:
        prev = OUTPUT_DIR / f"{idx:03d}.png"
        if prev.exists():
            refs = refs + [str(prev)]

    result = {"status": "pending", "file": None, "error": None}
    for attempt in range(retries + 1):
        if should_stop and should_stop():
            return result
        status(status="busy")
        result["status"] = "busy"
        try:
            png = generate_image(api_key, prompt, settings, refs)
            fname = f"{idx + 1:03d}.png"
            (OUTPUT_DIR / fname).write_bytes(png)
            result.update(status="done", file=fname, error=None)
            status(status="done", file=fname, error=None)
            return result
        except Exception as e:
            msg = str(e)
            wait = 2.0 * (attempt + 1)
            if msg.startswith("429") or "rate limit" in msg.lower() \
               or "Too Many Requests" in msg:
                wait = max(wait, 20.0)
            if "retry-after=" in msg:
                try:
                    wait = max(wait, float(msg.split("retry-after=")[1].split(")")[0]))
                except Exception:
                    pass
            if attempt == retries:
                result.update(status="error", error=msg)
                status(status="error", error=msg)
            else:
                time.sleep(wait)
    return result


def generate_all(api_key, prompts, settings, on_status=None, should_stop=None):
    """Generate a whole prompt list in order. Returns list of result dicts."""
    resolve_character(settings)
    use_prev = bool(settings.get("use_previous", True))
    delay = float(settings.get("delay", 0.5) or 0)
    retries = int(settings.get("retries", 3) or 3)
    results = []
    for idx, prompt in enumerate(prompts):
        if should_stop and should_stop():
            break
        results.append(process_one(api_key, idx, prompt, settings,
                                    use_prev=use_prev, retries=retries,
                                    on_status=on_status, should_stop=should_stop))
        if delay:
            time.sleep(delay)
    return results


# --------------------------------------------------------------------------
# Prompt + narration writing (Claude)
# --------------------------------------------------------------------------
def _parse_prompt_list(raw, count):
    """Best-effort extraction of a list of strings from an LLM response."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
        text = text.lstrip("json").strip("`\n ")
    try:
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1:
            arr = json.loads(text[start:end + 1])
            items = [str(x).strip() for x in arr if str(x).strip()]
            if items:
                return items
    except Exception:
        pass
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if len(blocks) < 2:
        blocks = [b.strip() for b in text.split("\n") if b.strip()]
    cleaned = []
    for b in blocks:
        b = b.lstrip("-*0123456789.)• ").strip()
        for sep in ("Scene", "scene"):
            if b.startswith(sep) and ":" in b[:12]:
                b = b.split(":", 1)[1].strip()
        if b:
            cleaned.append(b)
    return cleaned[:count] if cleaned else []


SCENE_CAP = 300            # hard upper bound on scenes per video
SCENE_CHUNK = 40           # prompts generated per LLM call (large counts batch)


def generate_scene_prompts(api_key, title, count, language="english",
                           style_hint="", model=None):
    """Turn a TITLE into `count` continuous storytelling scene prompts.

    Large counts are generated in batches (carrying the prior scenes forward)
    so the story stays coherent and we don't blow the token limit in one call.
    """
    count = max(1, min(int(count), SCENE_CAP))
    prompts = []
    while len(prompts) < count:
        start = len(prompts)
        need = min(SCENE_CHUNK, count - start)
        system_msg = (
            "You are a master visual storyteller for immersive, cinematic narrated "
            "story videos — the gripping second-person style of channels like "
            "'Lost Legacy' (e.g. 'Your Life as a ...'). "
            f"The full video is ONE continuous story told over {count} sequential "
            "scenes with a clear arc: a hook at scene 1, rising tension, a turning "
            f"point, and a resonant ending at scene {count}. "
            f"Now write scenes {start + 1} to {start + need}. "
            "For each scene write ONE vivid image-generation prompt describing only "
            "what we SEE: setting, the main character(s) and their expression + "
            "action, camera framing, time of day, lighting and mood. Keep the SAME "
            "character(s) and world visually consistent, and keep advancing the "
            "story. Do NOT mention art style and do NOT include narration, captions "
            "or dialogue text. "
            f"Write in {language}. "
            f"Return EXACTLY {need} prompts as a JSON array of strings and nothing "
            "else — no numbering, no keys, no markdown fences.")
        if style_hint:
            system_msg += f"\nStory/tone hint: {style_hint}"
        msgs = [{"role": "system", "content": system_msg}]
        if prompts:                       # carry the last few scenes for continuity
            tail = prompts[-3:]
            base = start - len(tail)
            msgs.append({"role": "user", "content":
                         "Previous scenes (continue seamlessly, do not repeat):\n" +
                         "\n".join(f"{base+i+1}. {p}" for i, p in enumerate(tail))})
        msgs.append({"role": "user",
                     "content": f"TITLE: {title}\n\nWrite scenes {start+1}-{start+need}."})
        raw = chat_llm(api_key, model or NARRATION_MODEL, msgs,
                       temperature=0.9, max_tokens=6000)
        chunk = _parse_prompt_list(raw, need)
        if not chunk:
            if not prompts:
                raise RuntimeError("Could not parse prompts from model output: " + raw[:300])
            break                         # stop gracefully if a later batch fails
        prompts += chunk
    return prompts[:count]


def write_narration(api_key, n, scene_prompts, language="english", style="",
                    model=None):
    """Write n short, plain, fast-paced narration lines (one per scene)."""
    prompt_summary = "\n".join(
        f"Scene {i+1}: {p or '(image)'}" for i, p in enumerate(scene_prompts))
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
    raw = chat_llm(api_key, model or NARRATION_MODEL,
                   [{"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Scenes:\n\n{prompt_summary}"}],
                   temperature=0.8, max_tokens=8000)
    lines = _parse_prompt_list(raw, n)
    if len(lines) < n:
        lines += [""] * (n - len(lines))
    return lines[:n]


# --------------------------------------------------------------------------
# Video: per-scene narrated, audio-synced MP4
# --------------------------------------------------------------------------
def build_video(api_key, tts_key, prompts=None, voice="Mia", style="",
                language="english", narration_model=None, tts_model=None,
                subtitles=False, on_progress=None, should_stop=None):
    """Build one synced MP4 from the images in OUTPUT_DIR + per-scene narration.

    Each image is held for exactly its own narration audio length (perfect
    sync). on_progress(stage, done, total, narration) is called as it runs.
    Returns {"file": name, "narration": [lines]}. Raises on failure.
    """
    if not has_ffmpeg():
        raise RuntimeError("ffmpeg not found — required to build video.")
    imgs = output_images()
    if not imgs:
        raise RuntimeError("No generated images in output/ yet.")
    n = len(imgs)
    prompts = prompts or load_project().get("prompts", [])
    scene_prompts = (list(prompts) + [""] * n)[:n] if prompts else [""] * n

    def progress(stage, done, narration=None):
        if on_progress:
            on_progress(stage, done, n, narration if narration is not None else [])

    progress("writing narration", 0)
    lines = write_narration(api_key, n, scene_prompts, language, style, narration_model)
    # persist narration so it can be reviewed / packaged later
    try:
        (OUTPUT_DIR / "narration.txt").write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass
    progress("synthesizing voice", 0, lines)

    work = OUTPUT_DIR / "_video_tmp"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()

    segments = []
    for i, (img, line) in enumerate(zip(imgs, lines)):
        if should_stop and should_stop():
            break
        progress(f"voice {i+1}/{n}", i, lines)
        speak = line.strip() or " "
        audio = tts_synthesize(tts_key, speak, voice, style, "wav", tts_model)
        apath = work / f"seg_{i:03d}.wav"
        apath.write_bytes(audio)
        seg = work / f"seg_{i:03d}.mp4"
        # optionally burn a big bold caption (the narration line) onto the image
        img_in = OUTPUT_DIR / img
        if subtitles and line.strip():
            cap = work / f"cap_{i:03d}.png"
            try:
                caption_image(str(OUTPUT_DIR / img), line, str(cap))
                img_in = cap
            except Exception as e:
                _log(f"[WARN] caption render failed (scene {i+1}): {e}")
        # image duration = its own audio length, exactly (perfect sync, no speed
        # change). Fast pacing comes from the short narration lines, not editing.
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-loop", "1", "-i", str(img_in),
            "-i", str(apath),
            "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
            "-r", "25",                            # constant fps for clean concat
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
        raise RuntimeError("No segments built.")

    progress("stitching video", n, lines)
    listfile = work / "list.txt"
    listfile.write_text("".join(f"file '{s.name}'\n" for s in segments))
    out = OUTPUT_DIR / f"final_video_{int(time.time())}.mp4"
    # RE-ENCODE on concat (not -c copy): stream-copying AAC leaves a small
    # priming gap per segment that accumulates into audio/video drift. Re-
    # encoding to one continuous stream (constant fps) keeps it perfectly synced.
    cc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", str(listfile),
         "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
         "-r", "25", "-vsync", "cfr",
         "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
         "-movflags", "+faststart", str(out)],
        cwd=str(work), capture_output=True)
    if cc.returncode != 0 or not out.exists():
        raise RuntimeError("ffmpeg concat failed: " + cc.stderr.decode()[:300])

    shutil.rmtree(work, ignore_errors=True)
    progress("done", n, lines)
    return {"file": out.name, "narration": lines}


# --------------------------------------------------------------------------
# Viral YouTube title + description (Claude)
# --------------------------------------------------------------------------
def _parse_json_obj(raw):
    t = raw.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t
        t = t.lstrip("json").strip("`\n ")
    s, e = t.find("{"), t.rfind("}")
    if s != -1 and e != -1:
        return json.loads(t[s:e + 1])
    raise ValueError("no JSON object in model output")


def generate_metadata(api_key, title, prompts=None, language="english",
                      model=None):
    """Claude writes viral, high-CTR YouTube titles + a keyword-rich
    description + tags. Returns a dict and also persists metadata.json/.txt."""
    prompts = prompts if prompts is not None else load_project().get("prompts", [])
    summary = "\n".join(f"- {p}" for p in (prompts or [])[:12] if p)
    system_msg = (
        "You are a top YouTube growth strategist. Given a video TITLE/IDEA and "
        "its scenes, craft metadata engineered to MAXIMIZE click-through and "
        "views: irresistible curiosity, emotional pull, power words, and the main "
        "keyword near the front — but it must be HONEST to the story (no "
        "misleading bait). "
        "Return ONLY a JSON object with these keys: "
        "\"titles\": array of 5 title options, each <= 70 characters, strongest "
        "first; "
        "\"description\": a string with a punchy 1-2 sentence hook, then a 2-3 "
        "sentence summary, then a blank line and 3-5 hashtags; "
        "\"tags\": array of 12-15 lowercase SEO keyword strings; "
        "\"hashtags\": array of 3-5 strings each starting with #. "
        f"Write everything in {language}. Output ONLY the JSON object.")
    user = (f"TITLE/IDEA: {title or '(untitled)'}\n\nScenes:\n{summary or '(none)'}"
            "\n\nGenerate the metadata.")
    raw = chat_llm(api_key, model or NARRATION_MODEL,
                   [{"role": "system", "content": system_msg},
                    {"role": "user", "content": user}],
                   temperature=0.9, max_tokens=1500)
    data = _parse_json_obj(raw)
    meta = {
        "titles": [str(t).strip() for t in (data.get("titles") or []) if str(t).strip()][:8],
        "description": str(data.get("description") or "").strip(),
        "tags": [str(t).strip() for t in (data.get("tags") or []) if str(t).strip()],
        "hashtags": [str(t).strip() for t in (data.get("hashtags") or []) if str(t).strip()],
    }
    try:
        (OUTPUT_DIR / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        txt = "TITLES\n" + "\n".join(f"- {t}" for t in meta["titles"])
        txt += "\n\nDESCRIPTION\n" + meta["description"]
        txt += "\n\nTAGS\n" + ", ".join(meta["tags"])
        (OUTPUT_DIR / "metadata.txt").write_text(txt, encoding="utf-8")
    except Exception:
        pass
    return meta


# --------------------------------------------------------------------------
# YouTube thumbnail — Claude writes a clickbait prompt, GPT Image 2 renders it
# --------------------------------------------------------------------------
def write_thumbnail_prompt(api_key, title, scene_prompts, language="english",
                           model=None):
    summary = "\n".join(f"- {p}" for p in (scene_prompts or [])[:12] if p)
    system_msg = (
        "You design viral YouTube thumbnails. Given a video TITLE and its scene "
        "list, write ONE vivid image prompt for a CLICKBAIT thumbnail that "
        "instantly sells the story's hook. Describe a SINGLE bold focal "
        "composition: the main character in close/medium shot with an "
        "EXAGGERATED emotional expression (shock, fear, awe, grief or "
        "excitement) that fits the story, dramatic lighting, vivid saturated "
        "colors, strong depth, and clear empty space on one side for a title. "
        "You MAY specify 2-4 BIG bold capitalized words of punchy on-image text "
        "that tease the hook (keep it very short). Describe only what is SEEN; "
        "do NOT mention art style or rendering. Return ONLY the prompt as one "
        "paragraph — no quotes, no preamble. "
        f"Write any on-image text in {language}.")
    user = (f"TITLE: {title or '(untitled)'}\n\nScenes:\n{summary or '(none)'}\n\n"
            "Write the thumbnail prompt.")
    return chat_llm(api_key, model or NARRATION_MODEL,
                    [{"role": "system", "content": system_msg},
                     {"role": "user", "content": user}],
                    temperature=0.9, max_tokens=600).strip()


def generate_thumbnail(api_key, title="", prompts=None, settings=None,
                       language="english", model=None):
    """Generate a 16:9 clickbait YouTube thumbnail. Returns {file, prompt}."""
    settings = dict(settings or {})
    resolve_character(settings)
    prompts = prompts if prompts is not None else load_project().get("prompts", [])
    tprompt = write_thumbnail_prompt(api_key, title, prompts, language, model)

    style = (settings.get("style_suffix") or "").strip()
    full = tprompt
    if style:
        full += f"\n\nArt style (match the video exactly): {style}."
    full += (" Composition for a YouTube thumbnail (16:9): bold, high-contrast, "
             "ultra eye-catching, dramatic lighting, vivid colors.")

    tset = dict(settings)
    tset["size"] = "1536x1024"                 # landscape; cropped to 16:9 below
    refs = anchor_paths()
    sheet = settings.get("character_sheet")
    if sheet and Path(sheet).exists():
        refs.append(sheet)
    imgs = output_images()
    if imgs:                                   # 1st scene as a ref -> same character
        refs.append(str(OUTPUT_DIR / imgs[0]))

    png = generate_image(api_key, full, tset, refs, raw_prompt=True)
    dest = OUTPUT_DIR / "thumbnail.png"
    if has_ffmpeg():
        raw = OUTPUT_DIR / "_thumb_raw.png"
        raw.write_bytes(png)
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(raw),
             "-vf", "crop=iw:iw*9/16,scale=1280:720", str(dest)],
            capture_output=True)
        if r.returncode != 0 or not dest.exists():
            dest.write_bytes(png)              # fallback: keep the raw render
        raw.unlink(missing_ok=True)
    else:
        dest.write_bytes(png)
    (OUTPUT_DIR / "thumbnail_prompt.txt").write_text(tprompt, encoding="utf-8")
    return {"file": dest.name, "prompt": tprompt}


# --------------------------------------------------------------------------
# Packaging — bundle the deliverables (video, audio, images, text) into a zip
# --------------------------------------------------------------------------
def latest_video():
    vids = sorted(OUTPUT_DIR.glob("final_video_*.mp4"), key=lambda p: p.stat().st_mtime)
    return vids[-1] if vids else None


ALL_PARTS = ("video", "audio", "images", "thumbnail", "metadata", "prompts", "narration")


def package_outputs(dest, parts=None):
    """Zip the chosen deliverables from OUTPUT_DIR into `dest`. Returns the path.

    parts: any of video, audio, images, prompts, narration (default: all).
    Layout: video/<mp4>, audio/<narration + clips>, images/NNN.png,
            prompts.txt, project.json, narration.txt
    """
    parts = set(parts or ALL_PARTS)
    dest = Path(dest)
    vid = latest_video()
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        if "images" in parts:
            for fn in output_images():
                z.write(OUTPUT_DIR / fn, f"images/{fn}")
        if "video" in parts and vid:
            z.write(vid, f"video/{vid.name}")
        if "thumbnail" in parts:
            t = OUTPUT_DIR / "thumbnail.png"
            if t.exists():
                z.write(t, "thumbnail.png")
            tp = OUTPUT_DIR / "thumbnail_prompt.txt"
            if tp.exists():
                z.write(tp, "thumbnail_prompt.txt")
        if "audio" in parts:
            # pull the combined narration audio out of the final video
            if vid and has_ffmpeg():
                tmp = OUTPUT_DIR / "_narration_audio.m4a"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(vid),
                                "-vn", "-acodec", "copy", str(tmp)],
                               capture_output=True)
                if tmp.exists():
                    z.write(tmp, "audio/narration.m4a")
                    tmp.unlink()
            for w in sorted(OUTPUT_DIR.glob("narrator_*.wav")):
                z.write(w, f"audio/{w.name}")
        if "prompts" in parts:
            proj = load_project()
            pr = proj.get("prompts", [])
            if pr:
                z.writestr("prompts.txt", "\n\n".join(
                    f"{i+1:03d} — {p}" for i, p in enumerate(pr)))
            z.writestr("project.json", json.dumps(proj, ensure_ascii=False, indent=2))
        if "narration" in parts:
            nf = OUTPUT_DIR / "narration.txt"
            if nf.exists():
                z.write(nf, "narration.txt")
        if "metadata" in parts:
            for fn in ("metadata.txt", "metadata.json"):
                mf = OUTPUT_DIR / fn
                if mf.exists():
                    z.write(mf, fn)
    return str(dest)
