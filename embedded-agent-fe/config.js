const API_BASE = "http://localhost:8000";
const AGENT_TIMEOUT_MS = 35000; // text answers (backend gives Gemini 30 s)
const TTS_TIMEOUT_MS = 50000;   // audio synthesis: Gemini TTS takes ~25-40 ms per character (8-18 s for an explanation); backend gives it 45 s
