#!/usr/bin/env python3
"""
Sketch Reactor — CLI (for humans and AI agents).

Same engine as the web app (core.py), driven from the shell. Designed to be
agent-friendly:
  * fully non-interactive (everything via flags / env vars)
  * the FINAL result of every command is printed to STDOUT as JSON
  * progress + logs go to STDERR (silence with --quiet)
  * non-zero exit code on failure, with {"error": ...} on stdout

API keys (flags override env):
  --api-key   or env DEROUTER_API_KEY   (image generation + LLM, via derouter)
  --tts-key   or env MIMO_TTS_KEY        (MiMo TTS)

Examples:
  export DEROUTER_API_KEY=sk-...   MIMO_TTS_KEY=tp-...
  python cli.py auto --title "Your Life as a Concubine" --scenes 14 --out story.mp4
  python cli.py prompts --title "A lonely lighthouse keeper" --scenes 10
  python cli.py character --name "Budi" --desc "old fisherman, straw hat, blue shirt"
  python cli.py generate --prompts-file scenes.txt --character budi
  python cli.py tts --text "Hello world" --voice Mia --out hello.wav
  python cli.py video --voice Mia --lang english --out story.mp4
"""

import os
import re
import sys
import json
import argparse

import core
import jobs


def _worker_env(args):
    """Env for the spawned worker, threading through any flag-passed keys."""
    env = os.environ.copy()
    if args.api_key:
        env["DEROUTER_API_KEY"] = args.api_key
    if args.tts_key:
        env["MIMO_TTS_KEY"] = args.tts_key
    return env


def submit(jtype, params, title, args):
    """Enqueue a job, make sure a worker is running, and print the job id."""
    jid = jobs.new_job(jtype, params, title)
    jobs.ensure_worker(_worker_env(args))
    log(f"queued {jtype} job {jid}")
    out({"job_id": jid, "status": "queued", "type": jtype,
         "dir": str(jobs.job_dir(jid)),
         "hint": f"poll: cli.py status {jid}  |  list: cli.py jobs"})


def load_dotenv():
    """Load KEY=VALUE lines from a .env file next to the code (real env wins)."""
    f = core.APP_DIR / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def out(obj, code=0):
    """Print the final result as JSON to stdout and exit."""
    print(json.dumps(obj, ensure_ascii=False, indent=2))
    sys.exit(code)


def fail(msg):
    out({"error": str(msg)}, code=1)


def log(*a):
    if not _QUIET:
        print(*a, file=sys.stderr, flush=True)


_QUIET = False


def need(value, env, what):
    v = (value or os.environ.get(env, "")).strip()
    if not v:
        fail(f"{what} required (pass the flag or set ${env}).")
    return v


def read_prompts(args):
    """Prompts from --prompts-file (blank-line separated) or --prompts (||)."""
    if args.prompts_file:
        raw = open(args.prompts_file, encoding="utf-8").read().strip()
        return [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
    if args.prompts:
        return [p.strip() for p in args.prompts.split("||") if p.strip()]
    return []


def img_settings(args):
    s = {"model": args.model or "gpt-image-2", "quality": args.quality,
         "size": args.size, "retries": args.retries, "delay": args.delay,
         "style_suffix": args.style or "", "negative": args.negative or "",
         "use_previous": not args.no_prev}
    if getattr(args, "character", None):
        s["character"] = args.character
    return s


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_models(args):
    key = need(args.api_key, "DEROUTER_API_KEY", "API key")
    out({"models": core.list_models(key)})


def cmd_voices(args):
    out({"voices": core.VOICES})


def cmd_prompts(args):
    key = need(args.api_key, "DEROUTER_API_KEY", "API key")
    log(f"Writing {args.scenes} scene prompts for: {args.title!r}")
    prompts = core.generate_scene_prompts(key, args.title, args.scenes,
                                          args.lang, args.style_hint or "",
                                          args.model or None)
    out({"title": args.title, "count": len(prompts), "prompts": prompts})


def cmd_character(args):
    key = need(args.api_key, "DEROUTER_API_KEY", "API key")
    log(f"Generating character sheet: {args.name!r}")
    char = core.generate_character_sheet(key, args.name, args.desc, img_settings(args))
    char["sheet_path"] = str(core.CHAR_DIR / char["id"] / "sheet.png")
    out({"character": char})


def cmd_characters(args):
    out({"characters": core.list_characters()})


def cmd_generate(args):
    prompts = read_prompts(args)
    if not prompts:
        fail("No prompts. Use --prompts-file or --prompts \"a||b||c\".")
    if args.async_job:
        need(args.api_key, "DEROUTER_API_KEY", "API key")
        p = {"prompts": prompts, "model": args.model or "", "quality": args.quality,
             "size": args.size, "retries": args.retries, "delay": args.delay,
             "style": args.style or "", "negative": args.negative or "",
             "no_prev": args.no_prev, "character": args.character or ""}
        return submit("generate", p, f"{len(prompts)} images", args)
    key = need(args.api_key, "DEROUTER_API_KEY", "API key")
    settings = img_settings(args)
    core.resolve_character(settings)
    core.save_project(prompts, settings)
    log(f"Generating {len(prompts)} images "
        f"(character={settings.get('character') or 'none'}) ...")

    def on_status(idx, **kw):
        st = kw.get("status")
        if st == "done":
            log(f"  [{idx+1:03d}] done -> {kw.get('file')}")
        elif st == "error":
            log(f"  [{idx+1:03d}] ERROR: {kw.get('error')}")

    results = core.generate_all(key, prompts, settings, on_status=on_status)
    done = [r for r in results if r["status"] == "done"]
    out({"requested": len(prompts), "done": len(done),
         "failed": len(prompts) - len(done),
         "output_dir": str(core.OUTPUT_DIR), "images": core.output_images(),
         "results": results})


def cmd_tts(args):
    key = need(args.tts_key, "MIMO_TTS_KEY", "TTS key")
    text = args.text
    if args.text_file:
        text = open(args.text_file, encoding="utf-8").read().strip()
    if not text:
        fail("No text. Use --text or --text-file.")
    log("Synthesizing voice ...")
    audio = core.tts_synthesize(key, text, args.voice, args.style or "",
                                args.format, args.tts_model or None)
    dest = args.out or str(core.OUTPUT_DIR / f"narrator_{os.getpid()}.{args.format}")
    with open(dest, "wb") as f:
        f.write(audio)
    out({"file": dest, "bytes": len(audio), "voice": args.voice})


def cmd_video(args):
    key = need(args.api_key, "DEROUTER_API_KEY", "API key")
    tts = need(args.tts_key, "MIMO_TTS_KEY", "TTS key")
    if args.job:                              # build from a specific job's images
        if not jobs.read_status(args.job):
            fail(f"No such job: {args.job}")
        core.OUTPUT_DIR = jobs.job_dir(args.job)
        core.PROJECT_FILE = core.OUTPUT_DIR / "project.json"
    if not core.output_images():
        fail("No images found. Run `generate`/`auto` first (or pass --job <id>).")
    prompts = read_prompts(args) or None

    def on_progress(stage, done, total, narration):
        log(f"  [{done}/{total}] {stage}")

    res = core.build_video(key, tts, prompts, voice=args.voice, style=args.style or "",
                           language=args.lang, narration_model=args.model or None,
                           subtitles=args.subtitles, on_progress=on_progress)
    path = core.OUTPUT_DIR / res["file"]
    if args.out:
        import shutil
        shutil.copy(path, args.out)
        path = args.out
    out({"video": str(path), "scenes": len(res["narration"]),
         "narration": res["narration"]})


def cmd_jobs(args):
    """List all jobs with their status/progress."""
    if jobs.has_queued() and not jobs.worker_alive():
        jobs.ensure_worker(_worker_env(args))   # revive a dead worker if needed
    view = [{"id": s["id"], "type": s.get("type"), "title": s.get("title"),
             "status": s.get("status"), "stage": s.get("stage"),
             "done": s.get("done"), "total": s.get("total"),
             "video": (s.get("result") or {}).get("video_path"),
             "error": s.get("error")} for s in jobs.list_jobs()]
    out({"count": len(view), "worker_running": jobs.worker_alive(), "jobs": view})


def cmd_status(args):
    """Detailed status of one job."""
    st = jobs.read_status(args.job_id)
    if not st:
        fail(f"No such job: {args.job_id}")
    if st.get("status") == "queued" and not jobs.worker_alive():
        jobs.ensure_worker(_worker_env(args))
    out(st)


def cmd_worker(args):
    """Internal: run the queue worker loop (spawned detached by submit)."""
    jobs.worker_loop()
    out({"worker": "exited"})


def cmd_package(args):
    """Bundle the deliverables (video/audio/images/prompts/narration) into a zip."""
    if args.job:
        if not jobs.read_status(args.job):
            fail(f"No such job: {args.job}")
        core.OUTPUT_DIR = jobs.job_dir(args.job)
        core.PROJECT_FILE = core.OUTPUT_DIR / "project.json"
    parts = [p.strip() for p in args.include.split(",") if p.strip()] if args.include else None
    dest = args.out or str(core.OUTPUT_DIR / "bundle.zip")
    log(f"Packaging {parts or list(core.ALL_PARTS)} -> {dest}")
    path = core.package_outputs(dest, parts)
    import zipfile
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    out({"bundle": path, "count": len(names), "files": names})


def _auto_params(args):
    return {"title": args.title, "scenes": args.scenes, "lang": args.lang,
            "voice": args.voice, "style_hint": args.style_hint or "",
            "style": args.style or "", "negative": args.negative or "",
            "model": args.model or "", "quality": args.quality, "size": args.size,
            "retries": args.retries, "delay": args.delay,
            "character": args.character or "", "no_prev": args.no_prev,
            "no_thumbnail": args.no_thumbnail, "no_metadata": args.no_metadata,
            "subtitles": args.subtitles}


def cmd_thumbnail(args):
    """Generate a clickbait YouTube thumbnail (Claude prompt -> GPT Image 2)."""
    key = need(args.api_key, "DEROUTER_API_KEY", "API key")
    if args.job:
        if not jobs.read_status(args.job):
            fail(f"No such job: {args.job}")
        core.OUTPUT_DIR = jobs.job_dir(args.job)
        core.PROJECT_FILE = core.OUTPUT_DIR / "project.json"
    log("Designing clickbait thumbnail ...")
    res = core.generate_thumbnail(key, args.title, None, img_settings(args),
                                  args.lang, args.model or None)
    path = core.OUTPUT_DIR / res["file"]
    if args.out:
        import shutil
        shutil.copy(path, args.out)
        path = args.out
    out({"thumbnail": str(path), "prompt": res["prompt"]})


def cmd_auto(args):
    """Full pipeline: title -> prompts -> images -> narrated, synced video."""
    if args.async_job:
        # keys must be resolvable (flag or env) so the worker can use them
        need(args.api_key, "DEROUTER_API_KEY", "API key")
        need(args.tts_key, "MIMO_TTS_KEY", "TTS key")
        return submit("auto", _auto_params(args), args.title, args)
    key = need(args.api_key, "DEROUTER_API_KEY", "API key")
    tts = need(args.tts_key, "MIMO_TTS_KEY", "TTS key")

    log(f"[1/3] Writing {args.scenes} scene prompts ...")
    prompts = core.generate_scene_prompts(key, args.title, args.scenes,
                                          args.lang, args.style_hint or "",
                                          args.model or None)
    settings = img_settings(args)
    core.resolve_character(settings)
    core.save_project(prompts, settings)

    log(f"[2/3] Generating {len(prompts)} images ...")

    def on_status(idx, **kw):
        if kw.get("status") == "done":
            log(f"  [{idx+1:03d}] done")
        elif kw.get("status") == "error":
            log(f"  [{idx+1:03d}] ERROR: {kw.get('error')}")

    results = core.generate_all(key, prompts, settings, on_status=on_status)
    done = [r for r in results if r["status"] == "done"]
    if not done:
        fail("No images generated; aborting video build.")

    log("[3/3] Writing narration + building synced video ...")

    def on_progress(stage, d, total, narration):
        log(f"  [{d}/{total}] {stage}")

    # NOTE: --style is the image art style; narration tone stays empty here
    res = core.build_video(key, tts, prompts, voice=args.voice, style="",
                           language=args.lang, narration_model=args.model or None,
                           subtitles=args.subtitles, on_progress=on_progress)
    path = core.OUTPUT_DIR / res["file"]
    if args.out:
        import shutil
        shutil.copy(path, args.out)
        path = args.out

    meta = None
    if not args.no_metadata:
        log("Writing viral title & description ...")
        try:
            meta = core.generate_metadata(key, args.title, prompts, args.lang,
                                          args.model or None)
        except Exception as e:
            log(f"metadata failed (skipped): {e}")

    thumb = None
    if not args.no_thumbnail:
        log("Designing clickbait thumbnail ...")
        try:
            thumb = core.generate_thumbnail(key, args.title, prompts, settings,
                                            args.lang, args.model or None)
        except Exception as e:
            log(f"thumbnail failed (skipped): {e}")

    out({"video": str(path), "title": args.title,
         "images": len(done), "failed": len(prompts) - len(done),
         "metadata": meta, "thumbnail": (thumb or {}).get("file"),
         "prompts": prompts, "narration": res["narration"]})


def cmd_metadata(args):
    """Viral YouTube title + description + tags (Claude)."""
    key = need(args.api_key, "DEROUTER_API_KEY", "API key")
    if args.job:
        if not jobs.read_status(args.job):
            fail(f"No such job: {args.job}")
        core.OUTPUT_DIR = jobs.job_dir(args.job)
        core.PROJECT_FILE = core.OUTPUT_DIR / "project.json"
    log("Writing viral title & description ...")
    out(core.generate_metadata(key, args.title, None, args.lang, args.model or None))


# --------------------------------------------------------------------------
# Arg parsing
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(prog="cli.py", description="Sketch Reactor CLI")
    # common flags shared by every subcommand (work AFTER the subcommand too,
    # which is the natural form for agents): cli.py auto --api-key ... --title ...
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--quiet", action="store_true", help="suppress stderr progress")
    common.add_argument("--api-key", default="", help="derouter key (or $DEROUTER_API_KEY)")
    common.add_argument("--tts-key", default="", help="MiMo TTS key (or $MIMO_TTS_KEY)")
    sub = p.add_subparsers(dest="cmd", required=True, parser_class=argparse.ArgumentParser)

    def new(name, **kw):
        return sub.add_parser(name, parents=[common], **kw)

    def add_img_opts(sp):
        sp.add_argument("--model", default="", help="image/LLM model id")
        sp.add_argument("--quality", default="low", choices=["low", "medium", "high"])
        sp.add_argument("--size", default="1024x1024")
        sp.add_argument("--retries", type=int, default=3)
        sp.add_argument("--delay", type=float, default=0.0)
        sp.add_argument("--style", default=core.DEFAULT_STYLE,
                        help="art-style description (default: stickman doodle look)")
        sp.add_argument("--negative", default=core.DEFAULT_NEGATIVE,
                        help="things to avoid (default: no photoreal/3d/gradients)")
        sp.add_argument("--no-prev", action="store_true",
                        help="don't feed the previous image into the next")
        sp.add_argument("--character", default="", help="character id (see `characters`)")

    sp = new("models", help="list models derouter offers")
    sp.set_defaults(func=cmd_models)

    sp = new("voices", help="list TTS voices")
    sp.set_defaults(func=cmd_voices)

    sp = new("prompts", help="title -> scene prompts")
    sp.add_argument("--title", required=True)
    sp.add_argument("--scenes", type=int, default=8)
    sp.add_argument("--lang", default="english")
    sp.add_argument("--style-hint", default="")
    sp.add_argument("--model", default="")
    sp.set_defaults(func=cmd_prompts)

    sp = new("character", help="generate a reusable character sheet")
    sp.add_argument("--name", required=True)
    sp.add_argument("--desc", required=True, help="physical description")
    add_img_opts(sp)
    sp.set_defaults(func=cmd_character)

    sp = new("characters", help="list saved characters")
    sp.set_defaults(func=cmd_characters)

    sp = new("generate", help="generate images from prompts")
    sp.add_argument("--prompts-file", default="", help="blank-line separated prompts")
    sp.add_argument("--prompts", default="", help="prompts separated by ||")
    add_img_opts(sp)
    sp.add_argument("--async", dest="async_job", action="store_true",
                    help="queue as a background job; returns a job_id immediately")
    sp.set_defaults(func=cmd_generate)

    sp = new("tts", help="text -> voice (one clip)")
    sp.add_argument("--text", default="")
    sp.add_argument("--text-file", default="")
    sp.add_argument("--voice", default="Mia")
    sp.add_argument("--style", default="")
    sp.add_argument("--format", default="wav", choices=["wav", "pcm16"])
    sp.add_argument("--tts-model", default="")
    sp.add_argument("--out", default="", help="output path (default: output/)")
    sp.set_defaults(func=cmd_tts)

    sp = new("video", help="build synced video from images (output/ or a job)")
    sp.add_argument("--prompts-file", default="")
    sp.add_argument("--prompts", default="")
    sp.add_argument("--voice", default="Mia")
    sp.add_argument("--lang", default="english")
    sp.add_argument("--style", default="", help="narration tone")
    sp.add_argument("--model", default="", help="narration LLM model id")
    sp.add_argument("--job", default="", help="build from this job's images")
    sp.add_argument("--subtitles", action="store_true",
                    help="burn big bold captions (the narration) onto the video")
    sp.add_argument("--out", default="", help="copy final video here")
    sp.set_defaults(func=cmd_video)

    sp = new("package", help="bundle deliverables into a zip")
    sp.add_argument("--include", default="",
                    help="comma list: video,audio,images,prompts,narration (default all)")
    sp.add_argument("--job", default="", help="package this job's dir")
    sp.add_argument("--out", default="", help="zip path (default: <dir>/bundle.zip)")
    sp.set_defaults(func=cmd_package)

    sp = new("auto", help="title -> prompts -> images -> narrated video")
    sp.add_argument("--title", required=True)
    sp.add_argument("--scenes", type=int, default=12)
    sp.add_argument("--lang", default="english")
    sp.add_argument("--voice", default="Mia")
    sp.add_argument("--style-hint", default="", help="story/tone hint for prompts")
    add_img_opts(sp)
    sp.add_argument("--out", default="", help="copy final video here")
    sp.add_argument("--subtitles", action="store_true",
                    help="burn big bold captions (the narration) onto the video")
    sp.add_argument("--no-thumbnail", action="store_true",
                    help="skip generating the YouTube thumbnail")
    sp.add_argument("--no-metadata", action="store_true",
                    help="skip generating the viral title & description")
    sp.add_argument("--async", dest="async_job", action="store_true",
                    help="queue as a background job; returns a job_id immediately")
    sp.set_defaults(func=cmd_auto)

    sp = new("metadata", help="viral YouTube title + description + tags (Claude)")
    sp.add_argument("--title", default="", help="video title/idea (helps)")
    sp.add_argument("--lang", default="english")
    sp.add_argument("--job", default="", help="generate for this job's project")
    sp.add_argument("--model", default="", help="LLM model id")
    sp.set_defaults(func=cmd_metadata)

    sp = new("thumbnail", help="clickbait YouTube thumbnail (Claude + GPT Image 2)")
    sp.add_argument("--title", default="", help="video title (helps the hook)")
    sp.add_argument("--lang", default="english", help="language for on-image text")
    sp.add_argument("--job", default="", help="generate for this job's project")
    add_img_opts(sp)
    sp.add_argument("--out", default="", help="copy thumbnail here")
    sp.set_defaults(func=cmd_thumbnail)

    # --- job queue ---
    sp = new("jobs", help="list all queued/running/finished jobs")
    sp.set_defaults(func=cmd_jobs)

    sp = new("status", help="status of one job")
    sp.add_argument("job_id")
    sp.set_defaults(func=cmd_status)

    sp = new("_worker", help=argparse.SUPPRESS)   # internal: queue worker loop
    sp.set_defaults(func=cmd_worker)

    return p


def main(argv=None):
    global _QUIET
    load_dotenv()
    args = build_parser().parse_args(argv)
    _QUIET = args.quiet
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as e:
        fail(e)


if __name__ == "__main__":
    main()
