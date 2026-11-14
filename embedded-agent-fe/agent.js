/* Embedded AI agent: shared client, chat widget, Explain panels and audio.
 * Pages pass configuration; model text is never rendered through innerHTML. */
(function () {
  'use strict';

  const STYLE = `
    .agent-chat textarea { width: 100%; padding: 12px; border: 2px solid #cbd5e1; border-radius: 8px; font-size: 14px; font-family: inherit; resize: vertical; min-height: 80px; box-sizing: border-box; }
    .agent-chat__actions { display: flex; gap: 10px; margin-top: 10px; }
    .agent-chat__actions button { flex: 1; margin-top: 0; }
    .agent-chat__ask { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important; }
    .agent-chat__summary { background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important; }
    .agent-chat button[disabled] { opacity: 0.6; cursor: wait; transform: none !important; }
    .agent-chat__hint { margin: 8px 0 0; font-size: 12px; color: #64748b; }
    .agent-results { margin-top: 16px; }
    .agent-results__bar { display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: #64748b; margin-bottom: 8px; }
    .agent-results__bar a { color: #6b7fd7; cursor: pointer; text-decoration: none; font-weight: 500; }
    .agent-card { background: white; border-radius: 8px; border-left: 4px solid #3b82f6; padding: 14px 16px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    .agent-card--pending { border-left-color: #94a3b8; color: #64748b; }
    .agent-card--error { border-left-color: #dc2626; background: #fee2e2; color: #991b1b; }
    .agent-card--warning { border-left-color: #f59e0b; background: #fef3c7; color: #92400e; }
    .agent-card__q { font-size: 13px; color: #475569; margin-bottom: 6px; }
    .agent-card__q strong { color: #1e293b; }
    .agent-card__body { color: #1e293b; font-size: 15px; line-height: 1.6; }
    .agent-card__body p { margin: 0 0 8px; }
    .agent-card__body ul { margin: 0 0 8px; padding-left: 20px; }
    .agent-card__footer { display: flex; align-items: center; gap: 10px; margin-top: 8px; font-size: 12px; color: #94a3b8; }
    .agent-card__footer .agent-play { margin: 0; padding: 4px 10px; font-size: 12px; }
    .agent-badge { display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px; margin-bottom: 6px; }
    .agent-badge--out { background: #fef3c7; color: #92400e; }
    .agent-badge--term { background: #f1f5f9; color: #475569; }
    .agent-pill { background: #dbeafe; color: #1e40af; padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: bold; }
    .agent-note { font-size: 12px; color: #64748b; }
    .audio-btn[disabled] { opacity: 0.6; cursor: wait; transform: none; }
    .agent-explain { margin: 0 0 15px; padding: 14px 16px; background: #f5f3ff; border-left: 4px solid #8b5cf6; border-radius: 8px; }
    .agent-explain--loading { background: #f1f5f9; border-left-color: #94a3b8; }
    .agent-explain--warning { background: #fef3c7; border-left-color: #f59e0b; }
    .agent-explain--error { background: #fee2e2; border-left-color: #dc2626; }
    .agent-explain__body { color: #1e293b; font-size: 14px; line-height: 1.6; }
    .agent-explain--loading .agent-explain__body { color: #64748b; }
    .agent-explain--warning .agent-explain__body { color: #92400e; }
    .agent-explain--error .agent-explain__body { color: #991b1b; }
    .agent-explain__body p { margin: 0 0 8px; }
    .agent-explain__body p:last-child { margin-bottom: 0; }
    .agent-explain__controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 10px; font-size: 12px; color: #64748b; }
    .agent-explain__controls button { margin: 0; padding: 4px 10px; font-size: 12px; }
    .agent-explain--loading .agent-explain__controls { display: none; }
    .agent-explain--warning .agent-explain__replay, .agent-explain--error .agent-explain__replay { display: none; }
    .agent-explain__meta { margin-left: auto; color: #94a3b8; }
  `;

  function injectStyles() {
    if (document.getElementById('agent-styles')) return;
    const style = document.createElement('style');
    style.id = 'agent-styles';
    style.textContent = STYLE;
    document.head.appendChild(style);
  }

  function locale() {
    return (navigator.language || 'en-US');
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  // --- HTTP ---------------------------------------------------------------------------

  async function call(path, body, options) {
    options = options || {};
    const controller = new AbortController();
    const defaultTimeout = typeof AGENT_TIMEOUT_MS !== 'undefined' ? AGENT_TIMEOUT_MS : 35000;
    const timer = setTimeout(() => controller.abort(), options.timeoutMs || defaultTimeout);
    try {
      const response = await fetch(API_BASE + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      const contentType = response.headers.get('content-type') || '';
      if (!response.ok) {
        let message = 'HTTP ' + response.status;
        if (contentType.indexOf('application/json') !== -1) {
          const data = await response.json().catch(() => ({}));
          message = data.message || (typeof data.detail === 'string' ? data.detail : message);
        }
        return { ok: false, status: response.status, message: message, headers: response.headers };
      }
      if (options.expect === 'blob') {
        if (contentType.indexOf('audio/') !== 0) {
          return { ok: false, status: response.status, message: 'Unexpected response type: ' + contentType };
        }
        return { ok: true, data: await response.blob(), headers: response.headers };
      }
      return { ok: true, data: await response.json(), headers: response.headers };
    } catch (err) {
      if (err && err.name === 'AbortError') {
        return { ok: false, status: 0, message: 'The assistant took too long to respond. Please try again.' };
      }
      return { ok: false, status: 0, message: 'Could not reach the backend at ' + API_BASE + '. Is it running?' };
    } finally {
      clearTimeout(timer);
    }
  }

  // --- Safe rendering of model text -----------------------------------------------------

  function appendInline(parent, text) {
    // **bold** becomes <strong>; everything else is a text node.
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    parts.forEach((part) => {
      if (!part) return;
      if (part.length > 4 && part.startsWith('**') && part.endsWith('**')) {
        parent.appendChild(el('strong', null, part.slice(2, -2)));
      } else {
        parent.appendChild(document.createTextNode(part));
      }
    });
  }

  function render(target, text) {
    target.textContent = '';
    const lines = String(text || '').split(/\r?\n/);
    let paragraph = [];
    let list = null;
    const flushParagraph = () => {
      if (!paragraph.length) return;
      const p = el('p');
      appendInline(p, paragraph.join(' '));
      target.appendChild(p);
      paragraph = [];
    };
    lines.forEach((raw) => {
      const line = raw.trim();
      const bullet = /^[-*•]\s+(.*)$/.exec(line);
      if (bullet) {
        flushParagraph();
        if (!list) { list = el('ul'); target.appendChild(list); }
        const li = el('li');
        appendInline(li, bullet[1]);
        list.appendChild(li);
        return;
      }
      list = null;
      if (!line) { flushParagraph(); return; }
      paragraph.push(line);
    });
    flushParagraph();
  }

  function setBusy(buttons, busy) {
    buttons.forEach((button) => { button.disabled = !!busy; });
  }

  // --- Result cards ---------------------------------------------------------------------

  function card(kind, label, message) {
    const node = el('div', 'agent-card agent-card--' + kind);
    if (label) {
      const q = el('div', 'agent-card__q');
      q.appendChild(el('strong', null, label));
      node.appendChild(q);
    }
    node.appendChild(el('div', 'agent-card__body', message));
    return node;
  }

  function answerCard(label, text, meta, scope) {
    const node = el('div', 'agent-card');
    const q = el('div', 'agent-card__q');
    q.appendChild(el('strong', null, label));
    node.appendChild(q);
    if (scope === 'out_of_page') node.appendChild(el('span', 'agent-badge agent-badge--out', 'Not on this page'));
    if (scope === 'general_term') node.appendChild(el('span', 'agent-badge agent-badge--term', 'General explanation'));
    const body = el('div', 'agent-card__body');
    render(body, text);
    node.appendChild(body);
    const footer = el('div', 'agent-card__footer');
    const bits = [];
    if (meta && meta.model) bits.push(meta.model);
    if (meta && typeof meta.latency_ms === 'number') bits.push(meta.latency_ms + ' ms');
    if (meta && meta.cached) bits.push('cached');
    footer.appendChild(el('span', null, bits.join(' · ')));
    const play = el('button', 'agent-play', '🔊 Play');
    play.type = 'button';
    play.addEventListener('click', () => toggleAudio(text, { button: play }));
    footer.appendChild(play);
    node.appendChild(footer);
    return node;
  }

  // --- Chat widget ----------------------------------------------------------------------

  function mountChat(container, config) {
    injectStyles();
    warmUpVoices();
    container.classList.add('agent-chat');
    container.textContent = '';

    const textarea = el('textarea');
    textarea.placeholder = config.placeholder || 'Ask a question about what is shown on this page…';
    const actions = el('div', 'agent-chat__actions');
    const askButton = el('button', 'agent-chat__ask', 'Ask Question');
    const summaryButton = el('button', 'agent-chat__summary', 'Get Summary');
    askButton.type = 'button';
    summaryButton.type = 'button';
    actions.appendChild(askButton);
    actions.appendChild(summaryButton);
    const hint = el('p', 'agent-chat__hint', 'Ctrl/⌘ + Enter to ask. Answers use only the data displayed on this page.');

    const results = el('div', 'agent-results');
    const bar = el('div', 'agent-results__bar');
    bar.appendChild(el('span', null, 'Responses (newest first)'));
    const clear = el('a', null, 'Clear');
    clear.addEventListener('click', () => { stopAudio(); list.textContent = ''; results.style.display = 'none'; });
    bar.appendChild(clear);
    const list = el('div');
    results.appendChild(bar);
    results.appendChild(list);
    results.style.display = 'none';

    container.appendChild(textarea);
    container.appendChild(actions);
    container.appendChild(hint);
    container.appendChild(results);

    const buttons = [askButton, summaryButton];

    function show(node) {
      results.style.display = 'block';
      list.insertBefore(node, list.firstChild);
      return node;
    }

    async function run(label, path, extra) {
      const payload = config.getPayload ? config.getPayload() : {};
      if (!payload) {
        show(card('warning', label, config.emptyMessage || 'Nothing is loaded on this page yet.'));
        return;
      }
      setBusy(buttons, true);
      const pending = show(card('pending', label, label === 'Summary' ? '📊 Generating summary…' : '🤔 Thinking…'));
      try {
        const body = Object.assign({}, payload, extra, { locale: locale() });
        const result = await call(path, body);
        let replacement;
        if (result.ok) {
          const data = result.data;
          replacement = answerCard(label, data.answer || data.summary || '', data.meta, data.scope);
        } else if (result.status === 429) {
          replacement = card('warning', label, result.message);
        } else {
          replacement = card('error', label, result.message);
        }
        list.replaceChild(replacement, pending);
      } finally {
        setBusy(buttons, false);
      }
    }

    function ask() {
      const question = textarea.value.trim();
      if (!question) {
        show(card('warning', 'Question', 'Please enter a question first.'));
        textarea.focus();
        return;
      }
      run('Q: ' + question, config.askPath, { question: question });
    }

    askButton.addEventListener('click', ask);
    summaryButton.addEventListener('click', () => run('Summary', config.summarizePath, {}));
    textarea.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); ask(); }
    });

    return { ask: ask, summarize: () => run('Summary', config.summarizePath, {}) };
  }

  // --- Section data and cache keys --------------------------------------------------------

  const registry = new Map();

  // Pages call this after loading a section with exactly the fields that section displays.
  function register(sectionId, data) {
    registry.set(sectionId, data);
  }

  function registered(sectionId) {
    return registry.get(sectionId);
  }

  function hashString(text) {
    // FNV-1a, 32 bit: a short stable cache key, not a security boundary.
    let hash = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193);
    }
    return (hash >>> 0).toString(36);
  }

  function dataKey(sectionId, data) {
    return sectionId + ':' + hashString(JSON.stringify(data === undefined ? null : data));
  }

  // --- Audio ----------------------------------------------------------------------------------
  // One clip at a time. Gemini speech (POST /tts) is preferred; when it fails the browser's own
  // voice reads the same text and a "(browser voice)" note says so.

  const MAX_SPEECH_CHARS = 2000;      // same limit as the backend's /tts
  const MAX_CACHED_CLIPS = 12;
  const LABEL_LOADING = '⏳ Loading audio…';
  const LABEL_STOP = '⏹ Stop';

  const clips = new Map();            // speakable text -> audio Blob, so a replay needs no request
  const notes = new WeakMap();        // button -> the note element next to it
  let element = null;                 // the one shared HTMLAudioElement
  let active = null;                  // the clip being loaded or played
  let lastClipId = 0;

  // What is read aloud: no markdown markers, every line a sentence, cut at a sentence end.
  function toSpeakable(text) {
    const joined = String(text || '')
      .replace(/\*\*/g, '')
      .split(/\r?\n/)
      .map((line) => line.trim().replace(/^[-*•]\s+/, '').trim())
      .filter(Boolean)
      .map((line) => (/[.!?:;…]$/.test(line) ? line : line + '.'))
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim();
    if (joined.length <= MAX_SPEECH_CHARS) return joined;
    const cut = joined.slice(0, MAX_SPEECH_CHARS);
    const end = Math.max(cut.lastIndexOf('. '), cut.lastIndexOf('! '), cut.lastIndexOf('? '));
    return end > 0 ? cut.slice(0, end + 1) : cut;
  }

  function pickVoice(voices, language) {
    const normalise = (tag) => String(tag || '').toLowerCase().replace('_', '-');
    const wanted = normalise(language);
    const base = wanted.split('-')[0];
    let sameLanguage = null;
    for (const voice of voices || []) {
      const tag = normalise(voice.lang);
      if (tag === wanted) return voice;
      if (!sameLanguage && base && tag.split('-')[0] === base) sameLanguage = voice;
    }
    return sameLanguage;
  }

  // Chrome returns no voices on the first getVoices() call and loads them in the background, so the
  // widgets ask once at mount; by the time a fallback needs a voice for the page language the list is there.
  function warmUpVoices() {
    if (window.speechSynthesis) window.speechSynthesis.getVoices();
  }

  function idleLabel(button) {
    if (button.dataset.idleLabel === undefined) button.dataset.idleLabel = button.textContent;
    return button.dataset.idleLabel;
  }

  function noteFor(button) {
    if (!button) return { textContent: '' };
    if (!notes.has(button)) {
      const note = el('span', 'agent-note');
      button.insertAdjacentElement('afterend', note);
      notes.set(button, note);
    }
    return notes.get(button);
  }

  function remember(text, blob) {
    clips.delete(text);
    clips.set(text, blob);
    while (clips.size > MAX_CACHED_CLIPS) clips.delete(clips.keys().next().value);
  }

  function paint(clip, status) {
    clip.status = status;
    clip.buttons.forEach((button) => { button.textContent = status === 'loading' ? LABEL_LOADING : LABEL_STOP; });
  }

  function releaseAudio(clip) {
    if (element) {
      element.onended = null;
      element.onerror = null;
      element.pause();
      element.removeAttribute('src');
      element.load();
    }
    if (clip.url) {
      URL.revokeObjectURL(clip.url);
      clip.url = null;
    }
  }

  // Ends a clip exactly once. Events from a clip that was already replaced or stopped (a late
  // "end" after speechSynthesis.cancel(), a response that arrives after Stop) find it gone and do nothing.
  function settle(clipId, outcome) {
    if (!active || active.id !== clipId) return;
    const clip = active;
    active = null;
    releaseAudio(clip);
    if (clip.via === 'browser' && window.speechSynthesis) window.speechSynthesis.cancel();
    clip.buttons.forEach((button) => { button.textContent = idleLabel(button); });
    clip.resolve(outcome);
  }

  function speakInBrowser(clip) {
    const synth = window.speechSynthesis;
    if (!synth || typeof SpeechSynthesisUtterance === 'undefined') {
      clip.note.textContent = 'Audio is unavailable right now.';
      settle(clip.id, 'failed');
      return;
    }
    releaseAudio(clip);
    const unavailable = () => {
      if (active !== clip) return;
      clip.note.textContent = 'Audio is unavailable right now.';
      settle(clip.id, 'failed');
    };
    try {
      const utterance = new SpeechSynthesisUtterance(clip.text);
      utterance.lang = clip.locale;
      const voice = pickVoice(synth.getVoices(), clip.locale);
      if (voice) utterance.voice = voice;
      utterance.onend = () => settle(clip.id, 'spoken');
      utterance.onerror = unavailable;
      clip.via = 'browser';
      clip.note.textContent = '(browser voice)';
      paint(clip, 'playing');
      synth.speak(utterance);
    } catch (error) {
      unavailable();
    }
  }

  function playBlob(clip, blob) {
    const audio = element || (element = new Audio());
    clip.via = 'gemini';
    clip.url = URL.createObjectURL(blob);
    audio.onended = () => settle(clip.id, 'played');
    audio.onerror = () => { if (active === clip) speakInBrowser(clip); };
    audio.src = clip.url;
    paint(clip, 'playing');
    audio.play().then(null, (error) => {
      if (active !== clip) return;
      if (error && error.name === 'NotAllowedError') {
        // The audio is fine, the browser wants a fresh click: it stays cached for the next one.
        clip.note.textContent = 'Your browser blocked autoplay. Press play to listen.';
        settle(clip.id, 'blocked');
      } else {
        speakInBrowser(clip);
      }
    });
  }

  async function loadClip(clip) {
    let blob = clips.get(clip.text);
    if (!blob) {
      const timeoutMs = typeof TTS_TIMEOUT_MS !== 'undefined' ? TTS_TIMEOUT_MS : 20000;
      const result = await call('/tts', { text: clip.text, locale: clip.locale }, { timeoutMs: timeoutMs, expect: 'blob' });
      if (result.ok) remember(clip.text, result.data);   // keep it even if the user already pressed Stop
      if (active !== clip) return;
      blob = result.ok ? result.data : null;
    }
    if (blob) playBlob(clip, blob);
    else speakInBrowser(clip);
  }

  // play(text, { button: Element | Element[], note?: Element, locale? }) -> Promise<outcome>
  // outcome: 'played' | 'spoken' (browser voice) | 'stopped' | 'blocked' | 'failed'
  function playAudio(text, options) {
    options = options || {};
    stopAudio();
    const buttons = [].concat(options.button || []).filter(Boolean);
    buttons.forEach(idleLabel);
    const note = options.note || noteFor(buttons[0]);
    note.textContent = '';
    const speakable = toSpeakable(text);
    if (!speakable) {
      note.textContent = 'Nothing to read aloud.';
      return Promise.resolve('failed');
    }
    return new Promise((resolve) => {
      const clip = {
        id: ++lastClipId, buttons: buttons, note: note, text: speakable,
        locale: options.locale || locale(), status: 'loading', via: null, url: null, resolve: resolve,
      };
      active = clip;
      paint(clip, 'loading');
      loadClip(clip).catch(() => { if (active === clip) speakInBrowser(clip); });
    });
  }

  function stopAudio() {
    if (active) settle(active.id, 'stopped');
  }

  function isActive(button) {
    return !!active && active.buttons.indexOf(button) !== -1;
  }

  function toggleAudio(text, options) {
    const first = [].concat((options && options.button) || [])[0];
    if (first && isActive(first)) {
      stopAudio();
      return Promise.resolve('stopped');
    }
    return playAudio(text, options);
  }

  function audioState() {
    return { status: active ? active.status : 'idle', via: active ? active.via : null };
  }

  const audio = { play: playAudio, stop: stopAudio, toggle: toggleAudio, isActive: isActive, state: audioState };

  // --- Explain panels -------------------------------------------------------------------------

  const explanations = new Map();   // dataKey(section, data) -> { text, meta }: reopening a section costs no request
  let lastExplainClick = 0;         // the newest click wins the speaker; earlier ones still show their text

  // What the user reads in the section, minus the Explain controls and any open panel.
  function sectionText(section) {
    const clone = section.cloneNode(true);
    clone.querySelectorAll('button, .agent-explain, script, style').forEach((node) => node.remove());
    // innerText only gets line breaks right on a laid-out element, so read it from an off-screen copy.
    clone.style.cssText = 'position:absolute;left:-99999px;top:0;width:' + section.offsetWidth + 'px;pointer-events:none;';
    clone.setAttribute('aria-hidden', 'true');
    document.body.appendChild(clone);
    const text = clone.innerText;
    clone.remove();
    return text.replace(/\n{3,}/g, '\n\n').trim().slice(0, 4000);
  }

  function mountSection(section, config) {
    const button = section.querySelector('.audio-btn');
    if (!button || section.dataset.agentMounted) return;
    section.dataset.agentMounted = '1';
    const sectionId = section.dataset.section;
    idleLabel(button);                // remember "🔊 Explain" before any state change
    let panel = null;
    let busy = false;

    function ensurePanel() {
      if (panel) return panel;
      const root = el('div', 'agent-explain');
      root.setAttribute('role', 'region');
      root.setAttribute('aria-live', 'polite');
      const body = el('div', 'agent-explain__body');
      const controls = el('div', 'agent-explain__controls');
      const replay = el('button', 'agent-explain__replay', '🔊 Replay');
      const refresh = el('button', 'agent-explain__refresh', '↻ Refresh');
      const close = el('button', 'agent-explain__close', '✕ Close');
      [replay, refresh, close].forEach((control) => { control.type = 'button'; controls.appendChild(control); });
      const note = el('span', 'agent-note');
      const meta = el('span', 'agent-explain__meta');
      controls.appendChild(note);
      controls.appendChild(meta);
      root.appendChild(body);
      root.appendChild(controls);
      const header = section.querySelector('.header');
      if (header) header.insertAdjacentElement('afterend', root);
      else section.insertBefore(root, section.firstChild);
      idleLabel(replay);
      panel = { root: root, body: body, replay: replay, note: note, meta: meta, text: '' };
      replay.addEventListener('click', () => {
        if (panel.text) toggleAudio(panel.text, { button: [button, replay], note: note });
      });
      refresh.addEventListener('click', () => explain(true));
      close.addEventListener('click', closePanel);
      return panel;
    }

    function closePanel() {
      if (isActive(button)) stopAudio();
      if (panel) {
        panel.root.remove();
        panel = null;
      }
    }

    function showMessage(kind, message) {
      const view = ensurePanel();
      view.root.className = 'agent-explain agent-explain--' + kind;
      view.body.textContent = message;
      view.text = '';
      view.note.textContent = '';
      view.meta.textContent = '';
    }

    function showExplanation(entry) {
      const view = ensurePanel();
      view.root.className = 'agent-explain';
      render(view.body, entry.text);
      view.text = entry.text;
      view.note.textContent = '';
      const meta = entry.meta || {};
      view.meta.textContent = [meta.model, typeof meta.latency_ms === 'number' ? meta.latency_ms + ' ms' : '', meta.cached ? 'cached' : '']
        .filter(Boolean).join(' · ');
      return view;
    }

    async function explain(refresh) {
      if (busy) return;
      const turn = ++lastExplainClick;
      stopAudio();                                   // a new request takes over the speaker
      const data = registered(sectionId);
      if (data === undefined) {
        showMessage('warning', 'This section is still loading. Try again in a moment.');
        return;
      }
      const key = dataKey(sectionId, data);
      let entry = refresh ? null : explanations.get(key);
      if (!entry) {
        busy = true;
        button.disabled = true;
        button.textContent = '⏳ Explaining…';
        showMessage('loading', '⏳ Generating explanation…');
        let result;
        try {
          const body = { page: config.page, section: sectionId, section_text: sectionText(section), data: data, locale: locale() };
          if (refresh) body.refresh = true;
          result = await call('/explain', body);
        } finally {
          busy = false;
          button.disabled = false;
          button.textContent = idleLabel(button);
        }
        if (!result.ok) {
          showMessage(result.status === 429 ? 'warning' : 'error', result.message);
          return;
        }
        entry = { text: result.data.explanation, meta: result.data.meta };
        explanations.set(key, entry);
      }
      const view = showExplanation(entry);
      if (turn === lastExplainClick) playAudio(entry.text, { button: [button, view.replay], note: view.note });
    }

    button.addEventListener('click', () => {
      if (isActive(button)) {          // click while it is loading or playing: stop
        stopAudio();
        return;
      }
      explain(false);
    });
  }

  // Wires every [data-section] .section on the page; the page registers each section's data separately.
  function mountExplain(config) {
    config = config || {};
    const mountAll = () => {
      injectStyles();
      warmUpVoices();
      document.querySelectorAll('.section[data-section]').forEach((section) => mountSection(section, config));
    };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mountAll);
    else mountAll();
  }

  window.Agent = {
    call: call, render: render, setBusy: setBusy, mountChat: mountChat, locale: locale,
    register: register, registered: registered, dataKey: dataKey, toSpeakable: toSpeakable, audio: audio,
    mountExplain: mountExplain,
  };
})();
