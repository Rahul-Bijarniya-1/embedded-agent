import test from 'node:test';
import assert from 'node:assert/strict';
import { makeHarness, FakeElement, audioReply, jsonReply, tick, waitFor } from './harness.mjs';

const IDLE = '🔊 Explain';
const LOADING = '⏳ Loading audio…';
const STOP = '⏹ Stop';
const button = (label = IDLE) => new FakeElement('button', label);
const note = () => ({ textContent: '' });
const ttsDown = () => jsonReply(503, { error: 'llm_unavailable', message: 'Text-to-speech is not configured' });
const notAllowed = () => Promise.reject(Object.assign(new Error('blocked'), { name: 'NotAllowedError' }));

// --- Gemini audio path ------------------------------------------------------------------

test('play asks /tts for the text, plays the returned audio and toggles the button label', async () => {
  const h = makeHarness({ respond: () => audioReply() });
  const btn = button();
  const done = h.Agent.audio.play('Your score is 750.', { button: btn, note: note() });
  assert.equal(btn.textContent, LOADING);
  await tick();
  assert.equal(h.requests.length, 1);
  assert.equal(h.requests[0].url, 'http://api.test/tts');
  assert.deepEqual(h.requests[0].body, { text: 'Your score is 750.', locale: 'en-IN' });
  assert.equal(h.audios.length, 1);
  assert.equal(h.audios[0].playCalls, 1);
  assert.match(h.audios[0].src, /^blob:/);
  assert.equal(btn.textContent, STOP);
  assert.equal(h.Agent.audio.state().status, 'playing');
  assert.equal(h.Agent.audio.state().via, 'gemini');
  h.audios[0].onended();
  assert.equal(await done, 'played');
  assert.equal(btn.textContent, IDLE);
  assert.equal(h.Agent.audio.state().status, 'idle');
});

test('replaying the same text uses the cached audio and makes no request', async () => {
  const h = makeHarness({ respond: () => audioReply() });
  const btn = button();
  const first = h.Agent.audio.play('Hello there.', { button: btn, note: note() });
  await tick();
  h.audios[0].onended();
  await first;
  const second = h.Agent.audio.play('Hello there.', { button: btn, note: note() });
  await tick();
  assert.equal(h.requests.length, 1);
  assert.equal(h.audios[0].playCalls, 2);
  h.audios[0].onended();
  assert.equal(await second, 'played');
});

test('stop pauses the audio, restores the label and resolves the play promise', async () => {
  const h = makeHarness({ respond: () => audioReply() });
  const btn = button();
  const done = h.Agent.audio.play('Hello there.', { button: btn, note: note() });
  await tick();
  h.Agent.audio.stop();
  assert.equal(h.audios[0].paused, true);
  assert.equal(btn.textContent, IDLE);
  assert.equal(await done, 'stopped');
  assert.equal(h.Agent.audio.state().status, 'idle');
  assert.deepEqual(h.revoked, ['blob:fake-1']);
});

test('toggle on the active button stops it, and on an idle button plays it', async () => {
  const h = makeHarness({ respond: () => audioReply() });
  const btn = button();
  const first = h.Agent.audio.toggle('Hello there.', { button: btn, note: note() });
  await tick();
  assert.equal(h.Agent.audio.isActive(btn), true);
  const second = h.Agent.audio.toggle('Hello there.', { button: btn, note: note() });
  assert.equal(await second, 'stopped');
  assert.equal(await first, 'stopped');
  assert.equal(h.Agent.audio.isActive(btn), false);
  assert.equal(btn.textContent, IDLE);
});

test('stopping while the audio request is pending keeps it from starting later', async () => {
  const h = makeHarness();
  const btn = button();
  const done = h.Agent.audio.play('Hello there.', { button: btn, note: note() });
  assert.equal(h.requests.length, 1);
  h.Agent.audio.stop();
  assert.equal(await done, 'stopped');
  h.requests[0].resolve(audioReply());
  await tick();
  assert.equal(h.audios.length === 0 || h.audios[0].playCalls === 0, true);
  assert.equal(btn.textContent, IDLE);
  assert.equal(h.Agent.audio.state().status, 'idle');
});

test('audio that arrives after stop is still cached for the next click', async () => {
  const h = makeHarness();
  const btn = button();
  const done = h.Agent.audio.play('Hello there.', { button: btn, note: note() });
  h.Agent.audio.stop();
  await done;
  h.requests[0].resolve(audioReply());
  await tick();
  const again = h.Agent.audio.play('Hello there.', { button: btn, note: note() });
  await tick();
  assert.equal(h.requests.length, 1);
  assert.equal(h.audios[0].playCalls, 1);
  h.audios[0].onended();
  assert.equal(await again, 'played');
});

test('starting another clip stops the first one and resets its button', async () => {
  const h = makeHarness({ respond: () => audioReply() });
  const a = button(IDLE);
  const b = button('🔊 Play');
  const first = h.Agent.audio.play('First clip.', { button: a, note: note() });
  await tick();
  const second = h.Agent.audio.play('Second clip.', { button: b, note: note() });
  assert.equal(await first, 'stopped');
  assert.equal(a.textContent, IDLE);
  await tick();
  assert.equal(b.textContent, STOP);
  assert.equal(h.audios.length, 1, 'one shared audio element');
  assert.equal(h.audios[0].playCalls, 2);
  h.audios[0].onended();
  assert.equal(await second, 'played');
  assert.equal(b.textContent, '🔊 Play');
});

test('every button given to play shows the stop label, and each gets its own label back', async () => {
  const h = makeHarness({ respond: () => audioReply() });
  const header = button(IDLE);
  const replay = button('🔊 Replay');
  const done = h.Agent.audio.play('Hello there.', { button: [header, replay], note: note() });
  await tick();
  assert.equal(header.textContent, STOP);
  assert.equal(replay.textContent, STOP);
  h.audios[0].onended();
  await done;
  assert.equal(header.textContent, IDLE);
  assert.equal(replay.textContent, '🔊 Replay');
});

test('autoplay blocked: the audio stays cached and the user is told to press play', async () => {
  const h = makeHarness({ respond: () => audioReply() });
  h.control.playResult = notAllowed;
  const btn = button();
  const n = note();
  const done = h.Agent.audio.play('Hello there.', { button: btn, note: n });
  await tick();
  assert.equal(await done, 'blocked');
  assert.match(n.textContent, /press play/i);
  assert.equal(btn.textContent, IDLE);
  assert.equal(h.spoken.length, 0, 'a working clip is not replaced by the browser voice');
  h.control.playResult = () => Promise.resolve();
  const again = h.Agent.audio.play('Hello there.', { button: btn, note: n });
  await tick();
  assert.equal(h.requests.length, 1);
  assert.equal(h.audios[0].playCalls, 2);
  assert.equal(n.textContent, '');
  h.audios[0].onended();
  assert.equal(await again, 'played');
});

// --- Browser voice fallback ---------------------------------------------------------------

test('falls back to the browser voice and says so when /tts fails', async () => {
  const h = makeHarness({ respond: ttsDown });
  const btn = button();
  const n = note();
  const done = h.Agent.audio.play('Your score is 750.', { button: btn, note: n });
  await tick();
  assert.equal(h.spoken.length, 1);
  assert.equal(h.spoken[0].text, 'Your score is 750.');
  assert.equal(h.spoken[0].lang, 'en-IN');
  assert.match(n.textContent, /browser voice/);
  assert.equal(btn.textContent, STOP);
  assert.equal(h.Agent.audio.state().via, 'browser');
  h.spoken[0].onend();
  assert.equal(await done, 'spoken');
  assert.equal(btn.textContent, IDLE);
});

test('falls back to the browser voice when the backend cannot be reached', async () => {
  const h = makeHarness({ respond: undefined });
  const done = h.Agent.audio.play('Your score is 750.', { button: button(), note: note() });
  h.requests[0].reject(new TypeError('Failed to fetch'));
  await tick();
  assert.equal(h.spoken.length, 1);
  h.spoken[0].onend();
  assert.equal(await done, 'spoken');
});

test('falls back to the browser voice when /tts takes longer than the audio timeout', async () => {
  const h = makeHarness({ ttsTimeoutMs: 20 });
  const done = h.Agent.audio.play('Your score is 750.', { button: button(), note: note() });
  await waitFor(() => h.spoken.length === 1);
  h.spoken[0].onend();
  assert.equal(await done, 'spoken');
});

test('falls back to the browser voice when the returned audio cannot be decoded', async () => {
  const h = makeHarness({ respond: () => audioReply() });
  const n = note();
  const done = h.Agent.audio.play('Your score is 750.', { button: button(), note: n });
  await tick();
  h.audios[0].onerror();
  assert.equal(h.spoken.length, 1);
  assert.match(n.textContent, /browser voice/);
  assert.equal(h.audios[0].paused, true);
  h.spoken[0].onend();
  assert.equal(await done, 'spoken');
});

for (const [language, wanted, expectedVoice] of [
  ['en-IN', 'exact match, even with underscores', 'Aman'],
  ['en-GB', 'same language when there is no exact match', 'Samantha'],
  ['fr-FR', 'the browser default when nothing matches', null],
]) {
  test(`browser voice choice for ${language}: ${wanted}`, async () => {
    const voices = [
      { lang: 'hi-IN', name: 'Lekha' },
      { lang: 'en-US', name: 'Samantha' },
      { lang: 'en_IN', name: 'Aman' },
    ];
    const h = makeHarness({ respond: ttsDown, voices, language });
    const done = h.Agent.audio.play('Your score is 750.', { button: button(), note: note() });
    await tick();
    assert.equal(h.spoken[0].lang, language);
    assert.equal(h.spoken[0].voice ? h.spoken[0].voice.name : null, expectedVoice);
    h.spoken[0].onend();
    await done;
  });
}

test('Explain warms up the browser voice list at mount, because Chrome returns none on the first call', async () => {
  const voices = [{ lang: 'en_IN', name: 'Aman' }];
  const h = makeHarness({ respond: ttsDown, voices, lazyVoices: true });
  h.Agent.mountExplain({ page: 'Dashboard' });
  const done = h.Agent.audio.play('Your score is 750.', { button: button(), note: note() });
  await tick();
  assert.equal(h.spoken[0].voice && h.spoken[0].voice.name, 'Aman');
  h.spoken[0].onend();
  await done;
});

test('a late end event from a cancelled utterance does not reset the next clip', async () => {
  const h = makeHarness({ respond: ttsDown });
  const a = button();
  const b = button('🔊 Play');
  const first = h.Agent.audio.play('First clip.', { button: a, note: note() });
  await tick();
  const second = h.Agent.audio.play('Second clip.', { button: b, note: note() });
  await tick();
  assert.equal(await first, 'stopped');
  assert.equal(h.speech.cancelCalls, 1);
  assert.equal(h.spoken.length, 2);
  h.spoken[0].onend();
  assert.equal(b.textContent, STOP);
  assert.equal(h.Agent.audio.state().status, 'playing');
  h.spoken[1].onend();
  assert.equal(await second, 'spoken');
});

test('says audio is unavailable when neither the service nor a browser voice can speak', async () => {
  const h = makeHarness({ respond: ttsDown, withSpeech: false });
  const btn = button();
  const n = note();
  const done = h.Agent.audio.play('Your score is 750.', { button: btn, note: n });
  await tick();
  assert.equal(await done, 'failed');
  assert.match(n.textContent, /unavailable/i);
  assert.equal(btn.textContent, IDLE);
});

test('a browser voice error is reported as unavailable instead of hanging on Stop', async () => {
  const h = makeHarness({ respond: ttsDown });
  const btn = button();
  const n = note();
  const done = h.Agent.audio.play('Your score is 750.', { button: btn, note: n });
  await tick();
  h.spoken[0].onerror({ error: 'synthesis-failed' });
  assert.equal(await done, 'failed');
  assert.match(n.textContent, /unavailable/i);
  assert.equal(btn.textContent, IDLE);
});

test('an unexpected error while starting playback falls back to the browser voice instead of hanging', async () => {
  const h = makeHarness({ respond: () => audioReply(), breakObjectUrls: true });
  const btn = button();
  const n = note();
  const done = h.Agent.audio.play('Your score is 750.', { button: btn, note: n });
  await tick();
  assert.equal(h.spoken.length, 1);
  assert.match(n.textContent, /browser voice/);
  h.spoken[0].onend();
  assert.equal(await done, 'spoken');
  assert.equal(btn.textContent, IDLE);
});

test('a browser voice that throws is reported as unavailable and the button is restored', async () => {
  const h = makeHarness({ respond: ttsDown, breakSpeak: true });
  const btn = button();
  const n = note();
  const done = h.Agent.audio.play('Your score is 750.', { button: btn, note: n });
  await tick();
  assert.equal(await done, 'failed');
  assert.match(n.textContent, /unavailable/i);
  assert.equal(btn.textContent, IDLE);
  assert.equal(h.Agent.audio.state().status, 'idle');
});

// --- Text sent for speech -----------------------------------------------------------------

test('text is cleaned before it is sent: markdown markers dropped, bullets become sentences', async () => {
  const h = makeHarness({ respond: () => audioReply() });
  const done = h.Agent.audio.play('**Total debit:** ₹2,000\n- Food ₹500\n• Travel ₹900', { button: button(), note: note() });
  await tick();
  assert.equal(h.requests[0].body.text, 'Total debit: ₹2,000. Food ₹500. Travel ₹900.');
  h.Agent.audio.stop();
  await done;
});

test('text that is empty after cleaning is not sent anywhere', async () => {
  const h = makeHarness({ respond: () => audioReply() });
  const n = note();
  const done = h.Agent.audio.play('  **  **  ', { button: button(), note: n });
  assert.equal(await done, 'failed');
  assert.equal(h.requests.length, 0);
  assert.match(n.textContent, /nothing to read/i);
});

test('long text is cut at a sentence end to fit the 2000 character speech limit', () => {
  const { Agent } = makeHarness();
  const spoken = Agent.toSpeakable('Sentence number one is here. '.repeat(100));
  assert.ok(spoken.length <= 2000, 'length ' + spoken.length);
  assert.ok(spoken.endsWith('here.'));
});

// --- Section data helpers -----------------------------------------------------------------

test('dataKey is stable for equal data and differs by section and by data', () => {
  const { Agent } = makeHarness();
  const key = Agent.dataKey('credit-score', { score: 750, change: 12 });
  assert.equal(Agent.dataKey('credit-score', { score: 750, change: 12 }), key);
  assert.notEqual(Agent.dataKey('credit-score', { score: 751, change: 12 }), key);
  assert.notEqual(Agent.dataKey('score-trend', { score: 750, change: 12 }), key);
});

test('register stores a section’s data for the Explain button to send', () => {
  const { Agent } = makeHarness();
  assert.equal(Agent.registered('score-trend'), undefined);
  Agent.register('score-trend', { history: [{ date: '2026-10-01', score: 700 }] });
  assert.equal(Agent.registered('score-trend').history[0].score, 700);
});
