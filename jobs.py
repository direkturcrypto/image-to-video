"""
Sketch Reactor — job queue (file-based, single serial worker).

Each submitted pipeline run becomes a job under jobs/<job_id>/ with its OWN
output directory (so concurrent submits never clobber each other) and a
status.json the agent can poll. A single detached worker process drains the
queue FIFO, one job at a time (safe for API rate limits / cost).

  submit  -> new_job() writes jobs/<id>/{status,request}.json, ensure_worker()
  worker  -> worker_loop() takes the oldest queued job and runs it
  query   -> read_status(id) / list_jobs()
  deliver -> the job dir holds images, final video, narration; package per job

Keys are read by the worker from the environment (DEROUTER_API_KEY /
MIMO_TTS_KEY) — the spawning process passes them through, so they never touch
disk.
"""

import os
import sys
import json
import time
import secrets
import subprocess
from pathlib import Path

import core

JOBS_DIR = core.APP_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)
LOCK = JOBS_DIR / "worker.lock"
WORKER_LOG = JOBS_DIR / "worker.log"


def _now():
    return int(time.time())


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
def new_job(jtype, params, title=""):
    jid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
    d = JOBS_DIR / jid
    d.mkdir(parents=True, exist_ok=True)
    st = {"id": jid, "type": jtype, "title": title, "status": "queued",
          "stage": "queued", "done": 0, "total": 0,
          "created_at": _now(), "updated_at": _now(),
          "result": None, "error": None}
    (d / "status.json").write_text(json.dumps(st, ensure_ascii=False, indent=2))
    (d / "request.json").write_text(json.dumps(
        {"type": jtype, "params": params}, ensure_ascii=False, indent=2))
    return jid


def job_dir(jid):
    return JOBS_DIR / jid


def read_status(jid):
    f = JOBS_DIR / jid / "status.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def write_status(jid, **kw):
    f = JOBS_DIR / jid / "status.json"
    st = read_status(jid) or {"id": jid}
    st.update(kw)
    st["updated_at"] = _now()
    f.write_text(json.dumps(st, ensure_ascii=False, indent=2))
    return st


def list_jobs():
    out = []
    for d in sorted(JOBS_DIR.glob("*/")):
        st = read_status(d.name)
        if st:
            out.append(st)
    out.sort(key=lambda s: s.get("created_at", 0))
    return out


def _next_queued():
    for s in list_jobs():                 # already sorted oldest-first = FIFO
        if s.get("status") == "queued":
            return s["id"]
    return None


def has_queued():
    return _next_queued() is not None


# --------------------------------------------------------------------------
# Worker process management (single serial worker)
# --------------------------------------------------------------------------
def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def worker_alive():
    if not LOCK.exists():
        return False
    try:
        pid = int(LOCK.read_text().strip())
    except Exception:
        return False
    return _pid_alive(pid)


def ensure_worker(env=None):
    """Spawn a detached worker if none is alive. Returns True if it spawned one."""
    if worker_alive():
        return False
    py = sys.executable
    cli = str(core.APP_DIR / "cli.py")
    logf = open(WORKER_LOG, "a")
    subprocess.Popen([py, cli, "_worker"], stdout=logf, stderr=logf,
                     start_new_session=True, env=env or os.environ.copy())
    return True


def _acquire_lock():
    """Atomically claim the single-worker lock. Returns True if we own it."""
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if worker_alive():
            return False                  # a live worker already owns it
        try:                              # stale lock from a dead worker -> reclaim
            LOCK.unlink()
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, OSError):
            return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def worker_loop():
    """Drain the queue one job at a time, then exit after a short idle grace."""
    if not _acquire_lock():               # another worker already owns the queue
        return
    try:
        idle = 0
        while True:
            jid = _next_queued()
            if not jid:
                idle += 1
                if idle >= 3:             # ~3s of empty queue -> exit
                    break
                time.sleep(1.5)
                continue
            idle = 0
            run_job(jid)
    finally:
        try:
            LOCK.unlink()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Running a job
# --------------------------------------------------------------------------
def _settings(p):
    s = {"model": p.get("model") or "gpt-image-2",
         "quality": p.get("quality", "low"), "size": p.get("size", "1024x1024"),
         "retries": p.get("retries", 3), "delay": p.get("delay", 0),
         "style_suffix": p.get("style", ""), "negative": p.get("negative", ""),
         "use_previous": not p.get("no_prev", False)}
    if p.get("character"):
        s["character"] = p["character"]
    return s


def run_job(jid):
    d = job_dir(jid)
    try:
        req = json.loads((d / "request.json").read_text())
    except Exception as e:
        write_status(jid, status="error", error=f"bad request.json: {e}")
        return
    jtype, p = req.get("type"), req.get("params", {})

    # point core outputs at THIS job's dir (worker is single serial process)
    core.OUTPUT_DIR = d
    core.PROJECT_FILE = d / "project.json"

    api_key = os.environ.get("DEROUTER_API_KEY", "").strip()
    tts_key = os.environ.get("MIMO_TTS_KEY", "").strip()
    write_status(jid, status="running", stage="starting", error=None)

    try:
        if not api_key:
            raise RuntimeError("DEROUTER_API_KEY not set for the worker.")

        if jtype in ("auto", "generate"):
            if jtype == "auto":
                scenes = int(p.get("scenes", 12))
                write_status(jid, stage="writing prompts", total=scenes)
                prompts = core.generate_scene_prompts(
                    api_key, p["title"], scenes, p.get("lang", "english"),
                    p.get("style_hint", ""), p.get("model") or None)
            else:
                prompts = p.get("prompts") or []
                if not prompts:
                    raise RuntimeError("no prompts in request")
            settings = _settings(p)
            core.resolve_character(settings)
            core.save_project(prompts, settings)

            counter = {"done": 0}

            def on_status(idx, **kw):
                if kw.get("status") == "done":
                    counter["done"] += 1
                write_status(jid, stage="generating images",
                             done=counter["done"], total=len(prompts))

            results = core.generate_all(api_key, prompts, settings, on_status=on_status)
            ok = [r for r in results if r["status"] == "done"]

            if jtype == "generate":
                write_status(jid, status="done", stage="done",
                             result={"images": len(ok),
                                     "failed": len(prompts) - len(ok),
                                     "dir": str(d),
                                     "files": core.output_images()})
                return
            # auto -> continue to video
            if not ok:
                raise RuntimeError("no images generated; aborting video build")

        elif jtype == "video":
            prompts = p.get("prompts") or None
        else:
            raise RuntimeError(f"unknown job type: {jtype}")

        if not tts_key:
            raise RuntimeError("MIMO_TTS_KEY not set for the worker.")
        if jtype == "video":
            prompts_for_video = prompts
        else:                                  # auto
            prompts_for_video = prompts

        def on_progress(stage, done, total, narration):
            write_status(jid, stage=stage, done=done, total=total)

        res = core.build_video(
            api_key, tts_key, prompts_for_video, voice=p.get("voice", "Mia"),
            style=p.get("style", ""), language=p.get("lang", "english"),
            narration_model=p.get("model") or None, on_progress=on_progress)

        result = {"video": res["file"], "video_path": str(d / res["file"]),
                  "narration": res["narration"], "scenes": len(res["narration"]),
                  "images": core.output_images(), "dir": str(d)}
        write_status(jid, status="done", stage="done", result=result)

    except Exception as e:
        write_status(jid, status="error", error=str(e))
