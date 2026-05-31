# Sketch Reactor v4

Bulk AI image generator with voice narration support. Powered by OpenAI GPT Image 2 and MiMo-V2.5-TTS.

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

The server starts at `http://localhost:5000`.

## Requirements

- Python 3.8+
- Flask
- Requests
- ffmpeg (optional, for video frame extraction)

## Usage

1. Open the web UI at `http://localhost:5000`
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
