#!/usr/bin/env python
"""Live checks against Vertex AI using the settings in .env.

  python scripts/smoke_gemini.py --auth    # the service-account key can obtain a token
  python scripts/smoke_gemini.py --probe   # which model / location / thinking combos work; lists Gemini ids
  python scripts/smoke_gemini.py           # one text call and one TTS call with the configured settings
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import BACKEND_DIR, get_settings  # noqa: E402

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def check_auth(settings) -> None:
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_file(settings.credentials_path, scopes=SCOPES)
    creds.refresh(Request())
    print("auth OK", creds.service_account_email)


def make_client(project: str, location: str, timeout_s: float):
    from google import genai
    from google.genai import types

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
    )


def text_call(client, model: str, thinking_level: str | None) -> tuple[str, int]:
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction="You are a terse assistant.", temperature=0, max_output_tokens=32
    )
    if thinking_level:
        config.thinking_config = types.ThinkingConfig(thinking_level=thinking_level)
    started = time.perf_counter()
    response = client.models.generate_content(
        model=model, contents="Reply with the single word OK", config=config
    )
    return (response.text or "").strip(), int((time.perf_counter() - started) * 1000)


def tts_call(client, model: str, voice: str) -> tuple[bytes, str, int]:
    from google.genai import types

    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
    )
    started = time.perf_counter()
    response = client.models.generate_content(
        model=model, contents="Hello from the GoodScore assistant.", config=config
    )
    part = response.candidates[0].content.parts[0]
    return part.inline_data.data, part.inline_data.mime_type or "", int((time.perf_counter() - started) * 1000)


def as_wav(data: bytes, mime_type: str) -> bytes:
    if data[:4] == b"RIFF" or mime_type.startswith("audio/wav"):
        return data
    rate = 24000
    for piece in mime_type.split(";"):
        if piece.strip().startswith("rate="):
            rate = int(piece.strip()[5:])
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16, 1, 1,
        rate, rate * 2, 2, 16, b"data", len(data),
    )
    return header + data


def short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:110].replace(chr(10), ' ')}"


def probe(settings) -> None:
    locations = []
    for loc in (settings.location, "global"):
        if loc not in locations:
            locations.append(loc)
    print(f"{'kind':<5} {'location':<12} {'model':<30} {'thinking':<9} result")
    for loc in locations:
        client = make_client(settings.project, loc, 60)
        for model in settings.models:
            for thinking in (settings.thinking_level, None):
                try:
                    text, ms = text_call(client, model, thinking)
                    result = f"PASS {ms} ms -> {text[:20]!r}"
                except Exception as exc:  # noqa: BLE001 - a probe reports everything
                    result = "FAIL " + short_error(exc)
                print(f"{'text':<5} {loc:<12} {model:<30} {str(thinking):<9} {result}")
    client = make_client(settings.project, settings.tts_location, 60)
    for model in settings.tts_models:
        try:
            data, mime, ms = tts_call(client, model, settings.tts_voice)
            result = f"PASS {ms} ms, {len(data)} bytes, {mime}"
        except Exception as exc:  # noqa: BLE001
            result = "FAIL " + short_error(exc)
        print(f"{'tts':<5} {settings.tts_location:<12} {model:<30} {'-':<9} {result}")
    try:
        ids = sorted(
            m.name.rsplit("/", 1)[-1]
            for m in client.models.list(config={"query_base": True})
            if "gemini" in (m.name or "")
        )
        print(f"available gemini ids at {settings.tts_location} ({len(ids)}):")
        print("  " + ", ".join(ids[:60]))
    except Exception as exc:  # noqa: BLE001
        print("models.list FAIL", short_error(exc))


def default_run(settings) -> None:
    client = make_client(settings.project, settings.location, settings.timeout_s)
    text, ms = text_call(client, settings.models[0], settings.thinking_level)
    print(f"text  model={settings.models[0]} location={settings.location} thinking={settings.thinking_level} {ms} ms -> {text!r}")
    tts_client = make_client(settings.project, settings.tts_location, settings.tts_timeout_s)
    data, mime, ms = tts_call(tts_client, settings.tts_models[0], settings.tts_voice)
    out = BACKEND_DIR / "scratch" / "probe.wav"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(as_wav(data, mime))
    print(f"tts   model={settings.tts_models[0]} voice={settings.tts_voice} {ms} ms -> {mime}, wrote {out} ({out.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--auth", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    if args.auth:
        check_auth(settings)
    elif args.probe:
        probe(settings)
    else:
        default_run(settings)


if __name__ == "__main__":
    main()
