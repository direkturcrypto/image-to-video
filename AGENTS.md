# Sketch Reactor — Agent Guide

A local tool that turns a **title** into a **narrated, voice-synced MP4**: it
writes scene prompts, generates images (GPT Image 2), writes a fast-paced
narration, voices it (MiMo v2.5 TTS), and stitches everything so each image is
on screen for exactly its narration's length.

This guide is for AI agents driving the **CLI** (`cli.py`). Everything is
non-interactive and prints a single JSON object to **stdout**; logs/progress go
to **stderr**.

## Install (once)

Repo: <https://github.com/direkturcrypto/image-to-video>

```bash
git clone https://github.com/direkturcrypto/image-to-video.git
cd image-to-video
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # only flask + requests
# ffmpeg must be on PATH (macOS: brew install ffmpeg | Debian: apt install ffmpeg)
.venv/bin/python cli.py voices              # smoke test (JSON, no key needed)
```

Use `.venv/bin/python` for every command (Windows: `.venv\Scripts\python`).

## Update (get the latest code)

Always update before a run if you cloned earlier — the tool evolves:

```bash
cd image-to-video
git pull --ff-only
.venv/bin/pip install -r requirements.txt   # in case deps changed
```

(If `git pull` reports local changes blocking it, you're likely on a customized
copy — ask the user before discarding anything.)

## Configure keys (once)

```bash
cp .env.example .env     # then edit .env:
#   DEROUTER_API_KEY=sk-...   # image generation + LLM (prompts/narration)
#   MIMO_TTS_KEY=tp-...        # MiMo v2.5 TTS
```

`.env` is loaded automatically. If the keys are unknown, **ask the user**. Keys
may also be passed per command via `--api-key` / `--tts-key`, or as env vars.
Precedence: flag > env var > `.env`.

## Contract

- **stdout** = one JSON object (the result). Parse this.
- **stderr** = human progress. Ignore for parsing (or silence with `--quiet`).
- **exit code** 0 = success, 1 = failure. On failure stdout is `{"error": "..."}`.
- Generated files land in `output/` (images `001.png`…, `final_video_*.mp4`,
  `narration.txt`, `narrator_*.wav`).

## Commands

| Command | What it does | Key output keys |
|---|---|---|
| `auto` | **Full pipeline**: title → prompts → images → narrated video → clickbait thumbnail | `video`, `prompts`, `narration`, `images`, `thumbnail` |
| `thumbnail` | Clickbait YouTube thumbnail (Claude prompt → GPT Image 2, 16:9) | `thumbnail`, `prompt` |
| `prompts` | Title → scene prompts (no images) | `prompts` |
| `generate` | Generate images from given prompts | `images`, `done`, `failed`, `results` |
| `character` | Generate a reusable character reference sheet | `character.id`, `character.sheet_path` |
| `characters` | List saved characters | `characters[]` |
| `tts` | One text → one voice clip | `file`, `bytes` |
| `video` | Build synced video from images already in `output/` | `video`, `narration` |
| `package` | Zip the deliverables to hand to the user | `bundle`, `files[]` |
| `models` / `voices` | List available models / TTS voices | `models` / `voices` |

Common flags (work after the subcommand): `--api-key`, `--tts-key`, `--quiet`.
Image flags (on `generate`/`character`/`auto`): `--model --quality {low,medium,high}
--size --retries --delay --style --negative --no-prev --character <id>`.
`auto`/`video` also take `--subtitles` to burn BIG bold captions (the narration,
viral style: uppercase white text with a thick black outline) onto the video.

## Typical agent flow

```bash
# One shot: title -> finished video
python cli.py auto --title "Your Life as a Concubine" --scenes 14 \
  --lang english --voice Mia --out story.mp4 --quiet

# Then bundle everything for the user
python cli.py package --out deliverable.zip
```

`auto` stdout (abridged):

```json
{
  "video": "/abs/path/output/final_video_1717.mp4",
  "title": "Your Life as a Concubine",
  "images": 14, "failed": 0,
  "prompts": ["...", "..."],
  "narration": ["You wake at dawn.", "Silk and silence.", "..."]
}
```

## Step-by-step (more control)

```bash
python cli.py prompts  --title "..." --scenes 12 > prompts.json
python cli.py character --name "Budi" --desc "old fisherman, straw hat, blue shirt"
python cli.py generate --prompts-file scenes.txt --character budi --style "flat 2D doodle"
python cli.py video    --voice Mia --lang english --out story.mp4
```

Prompts input for `generate`/`video`: `--prompts-file` (blocks separated by a
blank line) or `--prompts "scene one||scene two||scene three"`.

## Delivering files to the user

`package` produces a zip you can send to the user. Pick parts with `--include`:

```bash
python cli.py package --include video,audio,narration --out clip.zip
python cli.py package --out full_bundle.zip        # everything (default)
```

Zip layout: `video/<mp4>`, `audio/narration.m4a` (+ any `narrator_*.wav`),
`images/NNN.png`, `prompts.txt`, `project.json`, `narration.txt`.

## Background jobs / queue (multiple videos at once)

Running `auto`/`generate` inline blocks until done. To fire off several and
poll them, add `--async`: it returns a **`job_id`** immediately, writes to its
**own** `jobs/<id>/` dir (so concurrent jobs never clobber each other), and a
single detached worker drains the queue **FIFO, one at a time** (safe for rate
limits). Keys must be in env/`.env` (or passed as flags — they're forwarded to
the worker, never written to disk).

```bash
# fire off two videos without waiting
J1=$(python cli.py auto --title "Story One" --scenes 14 --async | jq -r .job_id)
J2=$(python cli.py auto --title "Story Two" --scenes 10 --async | jq -r .job_id)

python cli.py jobs                 # where are all my videos at?
python cli.py status "$J1"         # {status: running|done|error, stage, done, total, result}
python cli.py package --job "$J1" --out story1.zip   # deliver when done
```

`jobs` / `status` output keys: `status` (queued|running|done|error), `stage`,
`done`/`total` (progress), and on success `result.video_path`. The worker
auto-(re)spawns on submit and is revived by `jobs`/`status` if it died with work
still queued — agents just poll `status` until `done`.

Build a video from an existing image job: `python cli.py video --job <id>`.

## Notes / costs

- Image generation is **pay-per-image** — mind `--scenes` (each scene = 1 image).
- Fast pacing: each image ≈ its narration line (~1.5–2.5s), so ~30 scenes ≈ 1 min.
- Continuity: each image references the previous one + style anchors + the
  chosen character sheet (disable with `--no-prev`).
- The web UI (`python app.py`) shares the same engine (`core.py`).
