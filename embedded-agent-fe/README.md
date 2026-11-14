# embedded-agent-fe

Static dashboard in plain HTML and JavaScript, no build step. Architecture and design notes are in the [root README](../README.md).

## Run

```bash
npx serve . --listen 3000      # or ./start.sh
# open http://localhost:3000
```

The backend must be running on `http://localhost:8000` (change `API_BASE` in `config.js`). Do not add `--single` to `serve`: it rewrites every page to the Dashboard.

## Pages

| Page | What is wired |
|---|---|
| `index.html` (Dashboard) | Explain on Credit Score, Top Transactions and Top Active Loans |
| `credit.html` (Credit Details) | Explain on Current Score, Score Trend, Factors and Score Breakdown |
| `transactions.html` (L2) | Chat widget: Ask a Question and Get Summary, with a play button on each answer |
| `loan-details.html` (L2) | The same chat widget for one loan (`#id=LOAN001` to `LOAN005`) |
| `loans.html` | Unchanged list of loans |

## `agent.js`

One script for all pages; model text is always rendered through DOM nodes, never `innerHTML`.

| API | Use |
|---|---|
| `Agent.mountChat(container, {page, askPath, summarizePath, getPayload, emptyMessage})` | Builds the Ask and Summary widget; cards stack newest first with a scope badge and a play button |
| `Agent.register(section, data)` | A page registers the fields a section displays; Explain sends them |
| `Agent.mountExplain({page})` | Wires every `[data-section]` Explain button: text panel with Replay, Refresh and Close, then audio |
| `Agent.audio.play / stop / toggle / isActive / state` | One clip at a time: Gemini speech from `POST /tts`, browser voice as fallback (with a "(browser voice)" note) |
| `Agent.call(path, body, options)` | `fetch` wrapper with timeout and readable errors |

`config.js` holds `API_BASE`, `AGENT_TIMEOUT_MS` (text) and `TTS_TIMEOUT_MS` (speech).

## Tests

```bash
node --test tests/*.test.mjs   # the audio engine: real agent.js against fake browser APIs, no packages
```

The Explain panels and chat cards are checked by hand in the browser:

1. Each Explain button shows text, then plays audio; a second click replays with no network call.
2. Refresh posts `refresh: true` and returns new wording; Close hides the panel.
3. Clicking while audio plays stops it; clicking another section's button cancels the first.
4. Score Trend's explanation cites the 12-point series.
5. With `GEMINI_TTS_MODEL=dead-model` the text still appears and the browser voice reads it, with the note.
6. Backend stopped: a red card or panel, buttons recover, Refresh retries once it is back.
7. Summary, Ask and Play work on Transactions and on Loan Details (LOAN001 to LOAN005, including the closed ones); Clear stops audio.
