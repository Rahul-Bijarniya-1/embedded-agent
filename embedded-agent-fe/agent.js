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
    if (window.Agent && window.Agent.audio) {
      const play = el('button', 'agent-play', '🔊 Play');
      play.type = 'button';
      play.addEventListener('click', () => window.Agent.audio.toggle(text, { button: play }));
      footer.appendChild(play);
    }
    node.appendChild(footer);
    return node;
  }

  // --- Chat widget ----------------------------------------------------------------------

  function mountChat(container, config) {
    injectStyles();
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
    clear.addEventListener('click', () => { list.textContent = ''; results.style.display = 'none'; });
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

  window.Agent = { call: call, render: render, setBusy: setBusy, mountChat: mountChat, locale: locale };
})();
