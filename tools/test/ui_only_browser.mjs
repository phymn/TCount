#!/usr/bin/env node
/**
 * Browser UI smoke for crowd-1.3-ui.
 * Camera: SIMULATED (canvas.captureStream fake getUserMedia).
 * Models: REAL onnx/wasm files from crowd/ when present; detection may still
 * fail to register people without real scene assets — UI state tests inject ledger.
 */
import { chromium } from 'playwright';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const root = join(dirname(fileURLToPath(import.meta.url)), '../..');
const crowd = join(root, 'crowd');

const results = [];
function record(name, kind, expected, actual, ok, note = '') {
  results.push({ name, kind, expected, actual, ok, note });
}

const port = 8766;
const py = spawn('python3', ['-m', 'http.server', String(port), '--bind', '127.0.0.1'], { cwd: root, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 600));
const base = `http://127.0.0.1:${port}/crowd/index.html?backend=wasm&model=nano&ui=simple&lang=en&_=${Date.now()}`;

const wasmPresent = existsSync(join(crowd, 'vendor/ort-wasm-simd-threaded.asyncify.wasm'));
const nanoPresent = existsSync(join(crowd, 'models/yolox_nano_person_w8.onnx'));
const fixturesPresent = existsSync(join(root, 'tools/testassets')) || existsSync(join(root, 'testassets'));

const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || '/usr/local/bin/google-chrome',
  headless: true,
  args: ['--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required'],
});
const page = await browser.newPage();

// SIMULATED camera: solid canvas stream (not a real phone camera / not scene fixtures).
await page.addInitScript(() => {
  const W = 640, H = 480;
  const canvas = document.createElement('canvas'); canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const paint = () => {
    ctx.fillStyle = '#1a1a1e'; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#c4a484'; ctx.fillRect(260, 120, 120, 300); // crude person-shaped blob
    ctx.fillStyle = '#222'; ctx.fillRect(290, 90, 60, 50);
  };
  paint(); setInterval(paint, 100);
  const md = navigator.mediaDevices || (navigator.mediaDevices = {});
  md.getUserMedia = async () => {
    window.__fake = { simulated: true, kind: 'canvas-blob' };
    const stream = canvas.captureStream(15);
    const tr = stream.getVideoTracks()[0];
    tr.getCapabilities = () => ({});
    tr.getSettings = () => ({ width: W, height: H });
    return stream;
  };
  md.enumerateDevices = async () => [{ kind: 'videoinput', deviceId: 'sim', label: 'Simulated Camera', groupId: 'g' }];
});

await page.goto(base, { waitUntil: 'domcontentloaded', timeout: 60000 });
await page.waitForFunction(() => window.__tc && window.__tc.VERSION, null, { timeout: 30000 });

const version = await page.evaluate(() => window.__tc.VERSION);
record('footer/VERSION', 'REAL_CODE', 'crowd-1.3-ui.1', version, version === 'crowd-1.3-ui.1');

const simpleDefault = await page.evaluate(() => ({
  bodySimple: document.body.classList.contains('mode-simple'),
  chipsHidden: getComputedStyle(document.getElementById('chips')).display === 'none',
  startLabel: document.getElementById('startBtn').textContent.trim(),
  stopLabel: document.getElementById('stopBtn').textContent.trim(),
  footer: document.getElementById('build-footer').textContent.trim(),
  footerHtml: document.getElementById('build-footer').innerHTML,
}));
record('Simple default UI', 'REAL_DOM', 'mode-simple, Count/Done, chips hidden', JSON.stringify(simpleDefault),
  simpleDefault.bodySimple && simpleDefault.chipsHidden && simpleDefault.startLabel === 'Count' && simpleDefault.stopLabel === 'Done');
record('Simple footer version-only (EN)', 'REAL_DOM', 'crowd-1.3-ui.1 exactly, no multiline/CHANGE copy',
  JSON.stringify({ footer: simpleDefault.footer, html: simpleDefault.footerHtml }),
  simpleDefault.footer === 'crowd-1.3-ui.1' && !simpleDefault.footer.includes('\n') && !simpleDefault.footerHtml.includes('<br') && !/Simple\/Super|pipeline|Contar|cajas/i.test(simpleDefault.footer));

await page.click('#langSeg [data-lang="es"]');
const es = await page.evaluate(() => ({
  start: document.getElementById('startBtn').textContent.trim(),
  stop: document.getElementById('stopBtn').textContent.trim(),
  status: document.getElementById('status').textContent,
  footer: document.getElementById('build-footer').textContent.trim(),
}));
record('ES i18n', 'REAL_DOM', 'Contar / Listo', `${es.start}/${es.stop}`, es.start === 'Contar' && es.stop === 'Listo');
record('Simple footer version-only (ES)', 'REAL_DOM', 'still crowd-1.3-ui.1 only (no ES change blurb)', es.footer,
  es.footer === 'crowd-1.3-ui.1' && !/personas|Pipeline|versión|Contar|cajas/i.test(es.footer));
await page.click('#langSeg [data-lang="en"]');

await page.click('#uiSeg [data-ui="super"]');
const superUi = await page.evaluate(() => {
  document.getElementById('settingsBtn').click();
  const notes = document.getElementById('build-notes');
  return {
    bodySuper: document.body.classList.contains('mode-super'),
    chipsShown: getComputedStyle(document.getElementById('chips')).display !== 'none',
    settingsBtn: !!document.getElementById('settingsBtn') && getComputedStyle(document.getElementById('settingsBtn')).display !== 'none',
    footer: document.getElementById('build-footer').textContent.trim(),
    notes: notes ? notes.textContent.trim() : '',
    notesVisible: notes ? getComputedStyle(notes).display !== 'none' : false,
  };
});
record('Superuser reveals controls', 'REAL_DOM', 'mode-super + chips visible', JSON.stringify(superUi),
  superUi.bodySuper && superUi.chipsShown);
record('Superuser holds change notes; footer still version-only', 'REAL_DOM',
  'footer=crowd-1.3-ui.1; build-notes has EN change text',
  JSON.stringify({ footer: superUi.footer, notes: superUi.notes.slice(0, 80) }),
  superUi.footer === 'crowd-1.3-ui.1' && /Simple\/Superuser|registered people/i.test(superUi.notes));

// Inject registered people WITHOUT changing registration code path semantics —
// we call the same ledger.create the pipeline uses, then assert box helper.
const acceptance = await page.evaluate(() => {
  const tc = window.__tc;
  const before = tc.ledger.unique;
  const p = tc.ledger.create('new', null, null);
  const acceptedTrack = { matchedThisFrame: true, personId: p.id, box: [10, 10, 40, 80] };
  const pendingTrack = { matchedThisFrame: true, personId: 0, box: [100, 10, 40, 80] };
  const ghost = tc.ledger.create('new', null, null);
  ghost.mergedInto = -1; ghost.ghost = true;
  const ghostTrack = { matchedThisFrame: true, personId: ghost.id, box: [200, 10, 40, 80] };
  return {
    before,
    after: tc.ledger.unique,
    accepted: tc.trackIsRegisteredAccepted(acceptedTrack),
    pending: tc.trackIsRegisteredAccepted(pendingTrack),
    ghosted: tc.trackIsRegisteredAccepted(ghostTrack),
    // mode switch must not reset
  };
});
record('Accepted-box vs counted person (existing registration)', 'SIMULATED_LEDGER',
  'accepted=true for live personId; false for pending/ghost; unique increased by live only',
  JSON.stringify(acceptance),
  acceptance.accepted === true && acceptance.pending === false && acceptance.ghosted === false && acceptance.after === acceptance.before + 1);

const uniqueBeforeSwitch = await page.evaluate(() => window.__tc.ledger.unique);
await page.click('#uiSeg [data-ui="simple"]');
await page.click('#uiSeg [data-ui="super"]');
const uniqueAfterSwitch = await page.evaluate(() => window.__tc.ledger.unique);
record('Mode switch preserves count (idle)', 'REAL_DOM+SIMULATED_LEDGER', String(uniqueBeforeSwitch), String(uniqueAfterSwitch), uniqueBeforeSwitch === uniqueAfterSwitch);

// Start Count with SIMULATED camera — models may be REAL if wasm present.
await page.click('#startBtn');
await page.waitForTimeout(2500);
const runningState = await page.evaluate(() => ({
  running: window.__tc.running,
  fake: !!(window.__fake && window.__fake.simulated),
  startText: document.getElementById('startBtn').textContent.trim(),
  stopDisabled: document.getElementById('stopBtn').disabled,
}));
record('Count starts running (SIMULATED camera)', 'SIMULATED_CAMERA' + (wasmPresent && nanoPresent ? '+REAL_MODELS' : '+MODELS_PARTIAL'),
  'running=true, stop enabled, fake camera', JSON.stringify(runningState),
  runningState.running === true && runningState.fake === true && runningState.stopDisabled === false);

const uniqueWhileRunning = await page.evaluate(() => window.__tc.ledger.unique);
await page.click('#uiSeg [data-ui="simple"]');
const uniqueWhileRunningAfter = await page.evaluate(() => ({
  unique: window.__tc.ledger.unique,
  running: window.__tc.running,
  simple: document.body.classList.contains('mode-simple'),
}));
record('Mode switch while running preserves session', 'SIMULATED_CAMERA',
  `unique=${uniqueWhileRunning}, still running`, JSON.stringify(uniqueWhileRunningAfter),
  uniqueWhileRunningAfter.unique === uniqueWhileRunning && uniqueWhileRunningAfter.running === true && uniqueWhileRunningAfter.simple);

// Done freezes + stops
const uniqueAtDone = await page.evaluate(() => window.__tc.ledger.unique);
await page.click('#stopBtn');
await page.waitForTimeout(500);
const doneState = await page.evaluate(() => ({
  running: window.__tc.running,
  resultVisible: document.getElementById('result').classList.contains('visible'),
  final: document.getElementById('final-total').textContent.trim(),
  unique: window.__tc.ledger.unique,
  stream: !!(window.__tc.cam && window.__tc.cam.stream),
}));
record('Done stops processing and freezes count', 'SIMULATED_CAMERA',
  `running=false, result visible, unique frozen=${uniqueAtDone}, camera released`,
  JSON.stringify(doneState),
  doneState.running === false && doneState.resultVisible && doneState.unique === uniqueAtDone && doneState.stream === false);

await page.click('#uiSeg [data-ui="super"]', { force: false });
await page.click('#uiSeg [data-ui="simple"]');
const afterDoneSwitch = await page.evaluate(() => ({
  unique: window.__tc.ledger.unique,
  resultVisible: document.getElementById('result').classList.contains('visible'),
  running: window.__tc.running,
  simple: document.body.classList.contains('mode-simple'),
}));
record('Mode switch in done state preserves frozen count', 'REAL_DOM',
  `unique=${uniqueAtDone}, result still visible, not running, mode toggled`, JSON.stringify(afterDoneSwitch),
  afterDoneSwitch.unique === uniqueAtDone && afterDoneSwitch.resultVisible && afterDoneSwitch.running === false && afterDoneSwitch.simple);

// Count Again then Count restart
await page.click('#resetBtn');
const afterReset = await page.evaluate(() => ({
  unique: window.__tc.ledger.unique,
  resultVisible: document.getElementById('result').classList.contains('visible'),
  startEnabled: !document.getElementById('startBtn').disabled,
  startText: document.getElementById('startBtn').textContent.trim(),
}));
record('Count Again clears session', 'REAL_DOM', 'unique=0, Count enabled', JSON.stringify(afterReset),
  afterReset.unique === 0 && !afterReset.resultVisible && afterReset.startEnabled && afterReset.startText === 'Count');

await page.click('#startBtn');
await page.waitForTimeout(1500);
const restart = await page.evaluate(() => ({ running: window.__tc.running, fake: !!(window.__fake && window.__fake.simulated) }));
record('Count restart after Count Again', 'SIMULATED_CAMERA', 'running again', JSON.stringify(restart), restart.running === true);

const noPose = await page.evaluate(() => !document.getElementById('chip-pose') && !document.getElementById('setPose') && typeof window.TCountRTMPose === 'undefined');
record('No pose UI/runtime shipped', 'REAL_DOM', 'no chip-pose/setPose/rtmpose', String(noPose), noPose);

await browser.close();
py.kill('SIGTERM');

const failed = results.filter(r => !r.ok);
const summary = {
  kind: 'BROWSER_SMOKE',
  camera: 'SIMULATED (canvas.captureStream)',
  models: wasmPresent && nanoPresent ? 'REAL onnx+wasm present on disk' : 'PARTIAL — wasm or nano missing',
  fixtures: fixturesPresent ? 'present' : 'MISSING (tools/testassets / testassets)',
  device_iphone: 'NOT RUN',
  results,
  failed: failed.length,
};
console.log(JSON.stringify(summary, null, 2));
process.exit(failed.length ? 1 : 0);
