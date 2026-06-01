---
name: sketch-reactor
description: >-
  Create narrated story videos from a title or idea using the Sketch Reactor
  CLI. It generates a sequence of styled AI images (GPT Image 2), writes and
  voices a fast-paced narration (MiMo v2.5 TTS), and stitches them into a
  voice-synced MP4. Use this whenever the user wants to: turn a title/story idea
  into a narrated video, generate a sequence of styled AI images in one art
  style, build a slideshow video with AI voiceover, create reusable character
  reference sheets, run several video jobs in the background and check progress,
  or package/deliver the resulting video + audio + images. Trigger phrases
  include "make a story video", "image to video", "narrated/AI video from a
  title", "bikin video bercerita", "video naratif", "Lost Legacy style video".
allowed-tools: Bash, Read
---

# Sketch Reactor — story-video CLI

A local pipeline that turns a **title** into a **narrated, voice-synced MP4**.
Drive it through `cli.py`; every command prints one JSON object to **stdout**
(logs go to stderr), so parse stdout and check the exit code (0 ok, 1 error).

**Project root** (where `cli.py` lives):
`/Volumes/ExMac/Projects/ai/image-to-video`
Run commands from there (or call `cli.py` by absolute path) using the project's
venv: `.venv/bin/python cli.py ...`.

## Prerequisites (check first)

- **Update the code** (the tool evolves): from the project root run
  `git pull --ff-only && .venv/bin/pip install -r requirements.txt`. If the
  project isn't cloned yet, see "Install" in `AGENTS.md`.
- Keys must be set. They live in `.env` at the project root (auto-loaded):
  `DEROUTER_API_KEY` (images + LLM) and `MIMO_TTS_KEY` (TTS). If `.env` is
  missing, copy `.env.example` and ask the user for the two keys.
- `ffmpeg` must be on PATH (needed to build video/audio).
- **Cost**: image generation is pay-per-image; each scene = 1 image. Confirm the
  scene count with the user before large runs (~30 scenes ≈ a 1-minute video).

## The one command you usually want

```bash
.venv/bin/python cli.py auto --title "<TITLE>" --scenes <N> --lang english \
  --voice Mia --out video.mp4
```

This does the whole pipeline (prompts → images → narration → synced video) and
returns `{video, prompts, narration, images}`. Add `--style "<art style>"` to
lock a look, or `--character <id>` to keep a specific character consistent.

## Run several / long jobs in the background (queue)

Add `--async` to return a `job_id` immediately; a serial worker drains the queue
and each job gets its own `jobs/<id>/` dir. Poll, then deliver:

```bash
.venv/bin/python cli.py auto --title "Story One" --scenes 14 --async   # -> {job_id}
.venv/bin/python cli.py jobs                       # all jobs + status/progress
.venv/bin/python cli.py status <job_id>            # one job: status, stage, done/total
.venv/bin/python cli.py package --job <job_id> --out story1.zip   # deliver when done
```

Poll `status <job_id>` until `"status":"done"`, then `package`. On `"error"`,
read the `error` field and report it.

## Other commands

| Command | Use |
|---|---|
| `prompts --title "..." --scenes N` | just the scene prompts (JSON), e.g. to let the user edit them |
| `generate --prompts-file f.txt [--character id] [--style ...]` | images only (prompts blank-line separated, or `--prompts "a\|\|b"`) |
| `video [--job <id>]` | build the synced video from existing images (a job's, or `output/`) |
| `metadata --title "..." [--job ID]` | viral YouTube title + description + tags (Claude) |
| `thumbnail --title "..." [--job ID]` | clickbait YouTube thumbnail (Claude → GPT Image 2, 16:9) |
| `character --name "X" --desc "physical look..."` | make a reusable character sheet → returns its `id` |
| `characters` | list saved character ids |
| `tts --text "..." --voice Mia --out vo.wav` | one voice clip |
| `package [--job <id>] [--include video,audio,images,prompts,narration]` | zip deliverables |
| `models` / `voices` | list available models / TTS voices |

Common flags work after the subcommand: `--api-key`, `--tts-key`, `--quiet`.

## Delivering to the user

Use `package` to produce a zip (`video/`, `audio/narration.m4a`, `images/`,
`prompts.txt`, `narration.txt`, `project.json`) and hand that file to the user.
Pick parts with `--include` if they only want, say, the video + audio.

## Tips

- Pacing is fast cuts: each image is on screen ≈ its narration line (~1.5–2.5s).
  For a longer video, raise `--scenes`, don't lengthen lines.
- Continuity comes from feeding the previous image + style + character sheet
  into each generation; disable with `--no-prev`.
- For full flag/JSON-shape reference see `AGENTS.md` at the project root.
- The web UI (`.venv/bin/python app.py`) shares the same engine if a human
  prefers a browser.
