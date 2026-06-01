# Sketch Reactor v4

Turn a **title** into a **narrated, voice-synced story video**. Generates scene
prompts, images (GPT Image 2), a fast-paced narration, and voices it
(MiMo v2.5 TTS) — stitched so each image shows for exactly its narration length.

Two front-ends over one engine (`core.py`):
- **Web UI** — `python app.py` → http://localhost:5001 (tabs: Config / Characters / Auto Build / Manual Build)
- **CLI** — `python cli.py ...` for humans and **AI agents** (JSON output, non-interactive). See **[AGENTS.md](AGENTS.md)**.

## CLI quick start

```bash
pip install -r requirements.txt        # Flask + requests; ffmpeg on PATH for video
cp .env.example .env                   # add DEROUTER_API_KEY and MIMO_TTS_KEY

# one shot: title -> finished MP4
python cli.py auto --title "Your Life as a Concubine" --scenes 14 --out story.mp4
# bundle deliverables (video/audio/images/prompts/narration) for the user
python cli.py package --out deliverable.zip
```

Commands: `auto`, `prompts`, `generate`, `character`, `characters`, `tts`,
`video`, `package`, `models`, `voices`. Config via `.env` (keys) or `--api-key`/
`--tts-key` flags. Full reference + JSON output shapes in **[AGENTS.md](AGENTS.md)**.

## Setup for AI agents

Repo: <https://github.com/direkturcrypto/image-to-video>

```bash
# 1. clone + enter
git clone https://github.com/direkturcrypto/image-to-video.git
cd image-to-video

# 2. virtualenv + deps (only flask + requests)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
#    ffmpeg must be on PATH (macOS: brew install ffmpeg)

# 3. keys (loaded automatically by the CLI)
cp .env.example .env
#    edit .env -> DEROUTER_API_KEY=...   MIMO_TTS_KEY=...

# 4. smoke test (no key needed)
.venv/bin/python cli.py voices
```

Now an agent can drive everything via JSON in / JSON out:

```bash
# one shot, or queue it and poll
.venv/bin/python cli.py auto --title "A lonely lighthouse keeper" --scenes 12 --async
.venv/bin/python cli.py jobs            # progress of all jobs
.venv/bin/python cli.py status <job_id>
.venv/bin/python cli.py package --job <job_id> --out result.zip
```

**Claude Code skill** — this repo ships a skill at
`.claude/skills/sketch-reactor/SKILL.md` so Claude auto-uses the CLI when you ask
for a narrated/story video. It's active automatically when Claude Code runs in
this repo (restart the session to load it). To use it from **any** directory,
copy it to your personal skills folder:

```bash
mkdir -p ~/.claude/skills
cp -r .claude/skills/sketch-reactor ~/.claude/skills/
```

Contract for agents: every command prints one JSON object to **stdout** (logs to
stderr; silence with `--quiet`), exit code `0` ok / `1` error. Keys come from
`.env`, real env vars, or `--api-key`/`--tts-key` flags (in that precedence).
Full command + output-shape reference: **[AGENTS.md](AGENTS.md)**.

## Features

- **Bulk Image Generation** — Generate multiple images from text prompts in one batch
- **Style Anchors** — Upload reference images or extract frames from video to guide visual style
- **Previous Image Continuity** — Each generated image can reference the previous one for visual consistency
- **Voice Narrator (TTS)** — Convert text to speech using MiMo-V2.5-TTS with expression tags like `[crying]`, `[laughing]`, `[whisper]`, etc.
- **Auto-Narrator** — LLM generates narration script from your prompts, then converts to voice in one click
- **ZIP Export** — Download all generated images in a single ZIP file
- **Retry & Rate Limit Handling** — Automatic retries with backoff on API errors

## API Provider

Uses [Derouter Network](https://api-direct.derouter.network) as the API gateway (OpenAI-compatible endpoint).

### Pricing (Derouter)
| Resolution | Price per image |
|------------|-----------------|
| 1K (1024px) | $0.0084 |
| 2K (2048px) | $0.0126 |
| 4K (>2048px) | $0.0168 |

Quality setting does not affect price.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The server starts at `http://localhost:5001`.

## Requirements

- Python 3.8+
- Flask
- Requests
- ffmpeg (optional, for video frame extraction)

## Usage

1. Open the web UI at `http://localhost:5001`
2. Enter your Derouter API key
3. Add prompts (one per image)
4. Optionally upload style reference images or extract frames from a video
5. Click Generate — images will appear as they're created
6. Download all images as ZIP when done

## TTS Voices

| Voice | Language | Gender |
|-------|----------|--------|
| Mia | English | Female |
| Chloe | English | Female |
| Milo | English | Male |
| Dean | English | Male |
| Bing Tang | Chinese | Female |
| Mo Li | Chinese | Female |
| Su Da | Chinese | Male |
| Bai Hua | Chinese | Male |

## License

MIT
