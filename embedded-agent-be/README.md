# embedded-agent-be

FastAPI backend: the provided mock data endpoints (unchanged) plus the Gemini-backed assistant (summaries, questions, explanations, speech). Architecture, prompting strategy and production notes are in the [root README](../README.md).

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # defaults point at project goodscore-stage
mkdir -p secrets && cp /path/to/your-key.json secrets/sa.json     # git-ignored
python scripts/smoke_gemini.py --auth      # "auth OK <service account>"
uvicorn server:app --reload                # http://localhost:8000
```

Without a key the app still starts: mock endpoints serve, `GET /health` shows `llm.configured: false`, and AI routes answer `503 llm_unavailable`.

## Configuration

Read from the environment or `.env` (see `.env.example`). Model settings are ordered lists: the first id that exists is used, and a 404 moves on to the next.

| Variable | Default | Meaning |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | `goodscore-stage` | Vertex AI project |
| `GOOGLE_CLOUD_LOCATION` | `global` | Location for text models (they only answer at `global`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | `secrets/sa.json` | Service-account key; relative paths resolve against this folder |
| `GEMINI_MODEL` | `gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.8-flash` | Text models, in order of preference |
| `GEMINI_THINKING_LEVEL` | `minimal` | `minimal`, `low`, `medium`, `high`, or empty to let the model decide |
| `GEMINI_TTS_MODEL` | `gemini-3.8-flash-lite-tts,gemini-3.8-flash-tts` | Speech models, in order |
| `GEMINI_TTS_LOCATION` / `GEMINI_TTS_VOICE` | `global` / `Kore` | Speech location and voice |
| `LLM_TIMEOUT_S` / `TTS_TIMEOUT_S` | `30` / `45` | Per-call timeouts (speech takes roughly 25 to 50 ms per character) |
| `RATE_LIMIT_PER_MINUTE` | `20` | Per-client limit on AI and speech routes; `0` disables |
| `TRUST_FORWARDED_FOR` | `0` | `1` makes the limiter use `X-Forwarded-For`; only behind a proxy you control that overwrites it |
| `CACHE_TTL_S` / `CACHE_SIZE` | `600` / `256` | Answer cache lifetime and entries |
| `LOG_LEVEL` | `INFO` | JSON log level |

## Endpoints

| Route | Body | Returns |
|---|---|---|
| `POST /transactions/summarize` | `{transactions[], locale?}` | `{summary, meta}` |
| `POST /transactions/ask` | `{transactions[], question, locale?}` | `{answer, scope, meta}` |
| `POST /loans/summarize` | `{loan_id?, loan_data, locale?}` | `{summary, meta}` |
| `POST /loans/ask` | `{loan_id?, loan_data, question, locale?}` | `{answer, scope, meta}` |
| `POST /explain` | `{page, section, section_text?, data?, locale?, refresh?}` | `{explanation, meta}` |
| `POST /tts` | `{text, locale?}` | `audio/wav` with `X-Cache` and `X-TTS-Model` |
| `GET /health` | | active and dead models, caches, prompt version, rate limit |
| `GET /credit/summary`, `/credit/details`, `/credit/history`, `/transactions/top`, `/transactions?page=`, `/loans/summary`, `/loans/details/{id}`, `/loans/history` | | mock data, random on every call (unchanged) |

`scope` is `in_page`, `general_term`, `out_of_page` or `unknown`. `meta` is `{model, latency_ms, cached, request_id}`. Every response carries `X-Request-ID`.

Errors are always `{"error": code, "message": text}`: `422 invalid_request`, `429 rate_limited` (with `Retry-After`), `502 llm_error` or `llm_blocked`, `503 llm_unavailable`, `504 llm_timeout`.

```bash
curl -s localhost:8000/transactions/ask -H 'content-type: application/json' -d '{
  "transactions": [{"date": "2025-01-05", "category": "Food", "merchant": "Starbucks", "type": "debit", "amount": 300}],
  "question": "What is my total debit?", "locale": "en-IN"}'
```

## Scripts and tests

```bash
python scripts/smoke_gemini.py --probe        # which model, location and thinking combinations work right now
python scripts/smoke_gemini.py                # one text call and one speech call (writes scratch/probe.wav)
python scripts/eval_grounding.py              # grounding eval against the running backend; exit 1 on any FAIL
pytest -q                                     # unit and API tests with a fake model (no network)
RUN_LIVE_TESTS=1 pytest -m live               # real Gemini text and speech
```
