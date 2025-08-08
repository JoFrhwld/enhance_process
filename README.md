# Image Info

The base image is `python:3.12-slim-bookworm` with uv installed

```docker
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/aghcr.io/astral-sh/uv:0.8.6 /uv /uvx /bin/
```

The python packages installed are

```toml
dependencies = [
    "click>=8.2.1",
    "deepfilternet>=0.5.6",
    "librosa>=0.11.0",
    "torch>=2.8.0",
    "torchaudio>=2.8.0",
    "tqdm>=4.67.1",
]
```

Rust is also installed.

## Usage

```bash
docker pull ghcr.io/jofrhwld/enhance_process:release

docker run -t \
  -v .:/app jofrhwld/enhance_process \
  uv run /app/main.py audio.wav
```

