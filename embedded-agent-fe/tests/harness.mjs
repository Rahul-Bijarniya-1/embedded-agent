// Test harness: loads the real agent.js into a sandbox with fake browser APIs, so the audio
// engine can be exercised with `node --test` (built into Node; no packages, no build step).
import fs from 'node:fs';
import vm from 'node:vm';

const SOURCE = fs.readFileSync(new URL('../agent.js', import.meta.url), 'utf8');

/** Just enough DOM for the code under test: text, dataset, children, click listeners. */
export class FakeElement {
  constructor(tag = 'button', label = '') {
    this.tagName = String(tag).toUpperCase();
    this.textContent = label;
    this.dataset = {};
    this.className = '';
    this.disabled = false;
    this.style = {};
    this.children = [];
    this.listeners = {};
  }
  appendChild(child) { this.children.push(child); return child; }
  insertAdjacentElement(where, element) { this.adjacent = { where, element }; return element; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  click() { (this.listeners.click || []).forEach((fn) => fn({ type: 'click' })); }
}

export const audioReply = (tag = 'wav') => ({
  ok: true,
  status: 200,
  headers: { get: (name) => (name.toLowerCase() === 'content-type' ? 'audio/wav' : null) },
  blob: async () => ({ kind: 'blob', tag }),
});

export const jsonReply = (status, body) => ({
  ok: status >= 200 && status < 300,
  status,
  headers: { get: (name) => (name.toLowerCase() === 'content-type' ? 'application/json' : null) },
  json: async () => body,
});

/** Let pending promise continuations run. */
export const tick = () => new Promise((resolve) => setImmediate(resolve));

export async function waitFor(condition, timeoutMs = 1000) {
  const started = Date.now();
  while (!condition()) {
    if (Date.now() - started > timeoutMs) throw new Error('waitFor: condition not met in ' + timeoutMs + ' ms');
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

/**
 * Build a sandbox, run agent.js in it and return handles on everything the page would own.
 * `respond(request)` may return a reply to answer a fetch immediately; otherwise the request
 * stays pending until the test calls request.resolve(reply) / request.reject(error).
 */
export function makeHarness({ language = 'en-IN', respond, voices = [], lazyVoices = false, withSpeech = true, ttsTimeoutMs = 300, breakObjectUrls = false, breakSpeak = false } = {}) {
  const requests = [];
  const audios = [];
  const spoken = [];
  const revoked = [];
  const control = { playResult: () => Promise.resolve() };
  let urlCounter = 0;

  class FakeAudio {
    constructor() { this.src = ''; this.paused = true; this.playCalls = 0; this.onended = null; this.onerror = null; audios.push(this); }
    play() { this.playCalls += 1; this.paused = false; return control.playResult(); }
    pause() { this.paused = true; }
    load() {}
    removeAttribute(name) { if (name === 'src') this.src = ''; }
  }

  class FakeUtterance {
    constructor(text) { this.text = text; this.lang = ''; this.voice = null; this.onend = null; this.onerror = null; }
  }

  // Chrome returns an empty list on the first getVoices() call (it starts loading them); lazyVoices mimics that.
  let voiceCalls = 0;
  const speech = withSpeech
    ? {
        getVoices: () => ((voiceCalls += 1) === 1 && lazyVoices ? [] : voices),
        speak: (utterance) => {
          if (breakSpeak) throw new Error('speech engine crashed');
          spoken.push(utterance);
        },
        cancel: () => { speech.cancelCalls += 1; },
        cancelCalls: 0,
      }
    : undefined;

  const sandbox = {
    API_BASE: 'http://api.test',
    AGENT_TIMEOUT_MS: 2000,
    TTS_TIMEOUT_MS: ttsTimeoutMs,
    navigator: { language },
    document: {
      readyState: 'complete',
      head: new FakeElement('head'),
      createElement: (tag) => new FakeElement(tag),
      getElementById: () => null,
      querySelectorAll: () => [],
      addEventListener() {},
    },
    fetch: (url, init) => {
      const request = { url, body: JSON.parse(init.body) };
      requests.push(request);
      return new Promise((resolve, reject) => {
        request.resolve = resolve;
        request.reject = reject;
        init.signal.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' })));
        const immediate = respond && respond(request);
        if (immediate) resolve(immediate);
      });
    },
    AbortController,
    setTimeout,
    clearTimeout,
    URL: {
      createObjectURL: () => {
        if (breakObjectUrls) throw new Error('cannot create object URL');
        return 'blob:fake-' + (urlCounter += 1);
      },
      revokeObjectURL: (url) => { revoked.push(url); },
    },
    Audio: FakeAudio,
    SpeechSynthesisUtterance: FakeUtterance,
    console,
  };
  sandbox.window = sandbox;
  if (speech) sandbox.speechSynthesis = speech;

  vm.createContext(sandbox);
  vm.runInContext(SOURCE, sandbox, { filename: 'agent.js' });

  return { Agent: sandbox.Agent, requests, audios, spoken, speech, revoked, control };
}
