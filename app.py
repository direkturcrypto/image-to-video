"""
Sketch Reactor — web app (Flask).

Thin HTTP layer over core.py. All real work (image generation, TTS, narration,
character sheets, video building) lives in core; this file only handles HTTP
requests, the in-memory job state for the two long-running jobs (image batch +
video build), and serving files.

Run:  python app.py   ->   http://localhost:5001
"""

import io
import time
import json
import zipfile
import threading

from flask import Flask, request, jsonify, send_file, send_from_directory

import core

app = Flask(__name__, static_folder=None)

# in-memory job state for the image batch
STATE = {"running": False, "stop": False, "total": 0, "items": [], "settings": {}}
LOCK = threading.Lock()

# in-memory job state for the video build
VIDEO = {"running": False, "stop": False, "stage": "", "done": 0, "total": 0,
         "file": None, "error": None, "narration": []}
VLOCK = threading.Lock()


def set_item(idx, **kw):
    with LOCK:
        STATE["items"][idx].update(kw)


def _vset(**kw):
    with VLOCK:
        VIDEO.update(kw)


# --------------------------------------------------------------------------
# Static pages + files
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(core.APP_DIR, "index.html")


@app.route("/output/<path:fn>")
def serve_output(fn):
    return send_from_directory(core.OUTPUT_DIR, fn)


@app.route("/frames/<path:fn>")
def serve_frame(fn):
    return send_from_directory(core.FRAMES_DIR, fn)


@app.route("/characters/<path:fn>")
def serve_character(fn):
    return send_from_directory(core.CHAR_DIR, fn)


@app.route("/api/env")
def api_env():
    return jsonify({"ffmpeg": core.has_ffmpeg()})


@app.route("/api/test_key", methods=["POST"])
def test_key():
    key = ((request.json or {}).get("api_key") or "").strip()
    if not key:
        return jsonify({"ok": False, "error": "empty key"}), 400
    ok, err = core.test_key(key)
    return jsonify({"ok": ok} if ok else {"ok": False, "error": err})


@app.route("/api/models", methods=["POST"])
def api_models():
    key = ((request.json or {}).get("api_key") or "").strip()
    if not key:
        return jsonify({"error": "API key required."}), 400
    try:
        return jsonify({"ok": True, "models": core.list_models(key)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# --------------------------------------------------------------------------
# Style anchors from a sample video
# --------------------------------------------------------------------------
@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    if not core.has_ffmpeg():
        return jsonify({"error": "ffmpeg not found. Install it or upload style "
                                 "images directly instead."}), 400
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no video file"}), 400
    vid = core.APP_DIR / "sample_video.mp4"
    f.save(vid)
    try:
        saved = core.extract_frames(vid)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, "frames": saved})


@app.route("/api/set_anchors", methods=["POST"])
def set_anchors():
    chosen = (request.json or {}).get("frames", [])
    return jsonify({"ok": True, "anchors": core.set_anchors(chosen)})


@app.route("/api/upload_anchor", methods=["POST"])
def upload_anchor():
    f = request.files.get("file")
    slot = request.form.get("slot", "0")
    if not f or slot not in ("0", "1"):
        return jsonify({"error": "bad upload"}), 400
    f.save(core.ANCHOR_DIR / f"anchor_{slot}.jpg")
    return jsonify({"ok": True})


# --------------------------------------------------------------------------
# Image batch job
# --------------------------------------------------------------------------
def run_job(api_key, prompts, settings):
    core.generate_all(api_key, prompts, settings,
                      on_status=lambda idx, **kw: set_item(idx, **kw),
                      should_stop=lambda: STATE["stop"])
    with LOCK:
        STATE["running"] = False


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
    core.resolve_character(settings)   # so regen (uses STATE.settings) has it too
    with LOCK:
        STATE.update(running=True, stop=False, total=len(prompts), settings=settings)
        STATE["items"] = [{"idx": i, "prompt": p, "status": "pending",
                           "file": None, "error": None}
                          for i, p in enumerate(prompts)]
    core.save_project(prompts, settings)
    threading.Thread(target=run_job, args=(key, prompts, settings), daemon=True).start()
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
    settings = STATE["settings"]
    set_item(idx, status="pending", error=None)
    core.process_one(key, idx, STATE["items"][idx]["prompt"], settings,
                     use_prev=bool(settings.get("use_previous", True)),
                     retries=int(settings.get("retries", 3) or 3),
                     on_status=lambda i, **kw: set_item(i, **kw))
    return jsonify({"ok": True})


@app.route("/api/zip")
def api_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for it in STATE["items"]:
            if it["status"] == "done" and it["file"]:
                fp = core.OUTPUT_DIR / it["file"]
                if fp.exists():
                    z.write(fp, it["file"])
        z.writestr("prompts.txt", "\n\n".join(
            f"{it['idx']+1:03d} — {it['prompt']}" for it in STATE["items"]))
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name="sketch_reactor_images.zip")


# --------------------------------------------------------------------------
# Prompts from a title
# --------------------------------------------------------------------------
@app.route("/api/generate_prompts", methods=["POST"])
def api_generate_prompts():
    d = request.json or {}
    api_key = (d.get("api_key") or "").strip()
    title = (d.get("title") or "").strip()
    if not api_key:
        return jsonify({"error": "API key required."}), 400
    if not title:
        return jsonify({"error": "Title required."}), 400
    try:
        prompts = core.generate_scene_prompts(
            api_key, title, int(d.get("count", 8) or 8),
            d.get("language", "english"), (d.get("style_hint") or "").strip(),
            (d.get("model") or "").strip() or None)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, "prompts": prompts})


@app.route("/api/load_previous")
def api_load_previous():
    proj = core.load_project()
    return jsonify({"ok": True, "images": core.output_images(),
                    "prompts": proj.get("prompts", []),
                    "settings": proj.get("settings", {})})


# --------------------------------------------------------------------------
# Characters
# --------------------------------------------------------------------------
@app.route("/api/characters")
def api_characters():
    return jsonify({"ok": True, "characters": core.list_characters()})


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
    try:
        char = core.generate_character_sheet(api_key, name, description, settings)
    except Exception as e:
        return jsonify({"error": f"Character generation failed: {e}"}), 400
    return jsonify({"ok": True, "character": char})


@app.route("/api/delete_character", methods=["POST"])
def api_delete_character():
    cid = ((request.json or {}).get("id") or "").strip()
    if core.delete_character(cid):
        return jsonify({"ok": True})
    return jsonify({"error": "Character not found."}), 404


# --------------------------------------------------------------------------
# TTS
# --------------------------------------------------------------------------
@app.route("/api/tts", methods=["POST"])
def api_tts():
    d = request.json or {}
    api_key = (d.get("api_key") or "").strip()
    text = (d.get("text") or "").strip()
    out_format = d.get("format", "wav")
    if not api_key:
        return jsonify({"error": "TTS API key required."}), 400
    if not text:
        return jsonify({"error": "No text provided."}), 400
    try:
        audio = core.tts_synthesize(api_key, text, d.get("voice", "Mia"),
                                    (d.get("style") or "").strip(), out_format,
                                    (d.get("tts_model") or "").strip() or None)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    ext = "wav" if out_format == "wav" else "pcm"
    fname = f"narrator_{int(time.time())}.{ext}"
    (core.OUTPUT_DIR / fname).write_bytes(audio)
    return jsonify({"ok": True, "file": fname, "size": len(audio)})


@app.route("/api/tts_voices")
def api_tts_voices():
    return jsonify(core.VOICES)


# --------------------------------------------------------------------------
# Video build job
# --------------------------------------------------------------------------
def build_video_job(api_key, tts_key, prompts, voice, style, language,
                    narration_model, tts_model, subtitles=False):
    def on_progress(stage, done, total, narration):
        _vset(stage=stage, done=done, total=total, narration=narration)
    try:
        res = core.build_video(api_key, tts_key, prompts, voice=voice, style=style,
                               language=language, narration_model=narration_model,
                               tts_model=tts_model, subtitles=subtitles,
                               on_progress=on_progress,
                               should_stop=lambda: VIDEO.get("stop"))
        _vset(running=False, stage="done", file=res["file"],
              narration=res["narration"], error=None)
    except Exception as e:
        _vset(running=False, error=str(e))


@app.route("/api/build_video", methods=["POST"])
def api_build_video():
    if not core.has_ffmpeg():
        return jsonify({"error": "ffmpeg not found — required to build video."}), 400
    if VIDEO["running"]:
        return jsonify({"error": "Video build already running."}), 400
    d = request.json or {}
    api_key = (d.get("api_key") or "").strip()
    tts_key = (d.get("tts_key") or "").strip()
    prompts = d.get("prompts") or core.load_project().get("prompts", [])
    if not api_key:
        return jsonify({"error": "Image/LLM API key required (Config)."}), 400
    if not tts_key:
        return jsonify({"error": "TTS API key required (Config)."}), 400
    if not core.output_images():
        return jsonify({"error": "No generated images found in output/."}), 400
    with VLOCK:
        VIDEO.update(running=True, stop=False, stage="starting", done=0,
                     total=len(core.output_images()), file=None, error=None,
                     narration=[])
    threading.Thread(
        target=build_video_job,
        args=(api_key, tts_key, prompts, d.get("voice", "Mia"),
              (d.get("style") or "").strip(), d.get("language", "english"),
              (d.get("narration_model") or "").strip() or None,
              (d.get("tts_model") or "").strip() or None,
              bool(d.get("subtitles"))),
        daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/video_status")
def api_video_status():
    with VLOCK:
        return jsonify(dict(VIDEO))


@app.route("/api/thumbnail", methods=["POST"])
def api_thumbnail():
    d = request.json or {}
    api_key = (d.get("api_key") or "").strip()
    if not api_key:
        return jsonify({"error": "Image/LLM API key required (Config)."}), 400
    try:
        res = core.generate_thumbnail(
            api_key, d.get("title", ""), d.get("prompts"),
            d.get("settings") or {}, d.get("language", "english"),
            (d.get("model") or "").strip() or None)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, "file": res["file"], "prompt": res["prompt"]})


if __name__ == "__main__":
    print("\n  Sketch Reactor (GPT Image 2 + MiMo TTS)  ->  http://localhost:5001")
    print("  ffmpeg detected:", core.has_ffmpeg(), "\n")
    app.run(host="0.0.0.0", port=5001, threaded=True)
