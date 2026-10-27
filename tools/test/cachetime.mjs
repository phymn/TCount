// Headless before/after load timing for TCount crowd.
// Server sends Cache-Control: no-store and throttles bytes, so a fast second
// launch can only come from the app's own Cache API / service worker.
// Needs the gitignored release files in crowd/vendor and crowd/models, plus Chrome.
//   node tools/test/cachetime.mjs before
//   node tools/test/cachetime.mjs after
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import http from 'node:http';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const PHASE = process.argv[2] || 'before';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const PORT = 8093;
const DBG = 9223;
const BPS = 3_000_000;
const URL_AUTO = `http://127.0.0.1:${PORT}/crowd/?backend=wasm&model=nano&nogpu=1`;
const URL_DENSE = `http://127.0.0.1:${PORT}/crowd/?backend=wasm&model=nano&nogpu=1&dense=on&denseNow=1&denseEp=wasm`;

function contentType(p) {
  if (p.endsWith('.html')) return 'text/html; charset=utf-8';
  if (p.endsWith('.js')) return 'text/javascript; charset=utf-8';
  if (p.endsWith('.wasm')) return 'application/wasm';
  if (p.endsWith('.txt')) return 'text/plain; charset=utf-8';
  return 'application/octet-stream';
}

const requests = [];
let serverClosed = false;
const server = http.createServer((req, res) => {
  const u = new URL(req.url, 'http://127.0.0.1');
  let rel = decodeURIComponent(u.pathname);
  if (rel.endsWith('/')) rel += 'index.html';
  const file = path.normalize(path.join(ROOT, rel));
  const rec = { t: new Date().toISOString(), method: req.method, path: rel, status: 0, bytes: 0, ms: 0, t0: Date.now() };
  requests.push(rec);
  if (!file.startsWith(ROOT) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
    rec.status = 404; rec.ms = Date.now() - rec.t0; res.writeHead(404); res.end('nf'); return;
  }
  const buf = fs.readFileSync(file);
  rec.status = 200;
  res.writeHead(200, {
    'Content-Type': contentType(file),
    'Content-Length': buf.length,
    'Cache-Control': 'no-store',
  });
  if (req.method === 'HEAD') { rec.ms = Date.now() - rec.t0; res.end(); return; }
  const chunk = 256 * 1024;
  let o = 0; let aborted = false;
  req.on('close', () => { if (!res.writableEnded) aborted = true; });
  const write = () => {
    if (aborted) { rec.ms = Date.now() - rec.t0; rec.aborted = true; return; }
    if (o >= buf.length) { rec.bytes = o; rec.ms = Date.now() - rec.t0; res.end(); return; }
    const n = Math.min(chunk, buf.length - o);
    o += n; rec.bytes = o;
    res.write(buf.subarray(o - n, o));
    setTimeout(write, (n / BPS) * 1000);
  };
  write();
});
await new Promise(r => server.listen(PORT, '127.0.0.1', r));

const profile = mkdtempSync(path.join(tmpdir(), 'tc-chrome-'));
const chrome = spawn('google-chrome', [
  '--headless=new', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
  '--no-first-run', '--no-default-browser-check', `--user-data-dir=${profile}`,
  `--remote-debugging-port=${DBG}`, '--remote-allow-origins=*',
  '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
  '--autoplay-policy=no-user-gesture-required', 'about:blank',
], { stdio: 'ignore' });

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
async function waitDevtools() {
  for (let i = 0; i < 50; i++) {
    try { const r = await fetch(`http://127.0.0.1:${DBG}/json/version`); if (r.ok) return r.json(); } catch {}
    await sleep(200);
  }
  throw new Error('devtools not up');
}
const ver = await waitDevtools();
const ws = new WebSocket(ver.webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r));
let seq = 0; const pending = new Map();
ws.addEventListener('message', ev => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
});
const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
  const id = ++seq;
  const timer = setTimeout(() => { pending.delete(id); reject(new Error('timeout ' + method)); }, 20000);
  pending.set(id, m => { clearTimeout(timer); m.error ? reject(new Error(method + ' ' + JSON.stringify(m.error))) : resolve(m.result); });
  ws.send(JSON.stringify({ id, method, params, sessionId }));
});
const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
await send('Page.enable', {}, sessionId);
await send('Runtime.enable', {}, sessionId);
await send('Network.enable', {}, sessionId);
await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true }, sessionId);
async function shot(name) {
  try {
    const r = await send('Page.captureScreenshot', { format: 'png' }, sessionId);
    writeFileSync(name, Buffer.from(r.data, 'base64'));
    console.log('shot', name);
  } catch (e) { console.log('shot failed', name, e.message); }
}

async function evalJson(expr) {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true }, sessionId);
  if (r.exceptionDetails) return { error: r.exceptionDetails.text || 'exception', details: JSON.stringify(r.exceptionDetails).slice(0, 400) };
  return r.result?.value;
}
function snap(from) {
  const rows = requests.slice(from);
  const sum = (pred) => rows.filter(pred).reduce((s, r) => s + (r.method === 'HEAD' ? 0 : r.bytes), 0);
  const interesting = rows.filter(r => /wasm|onnx|sw\.js|app\.js|dense-worker|index\.html|ort\.webgpu/.test(r.path));
  return {
    wasmBytes: sum(r => r.path.endsWith('.wasm')),
    dmcountBytes: sum(r => r.path.includes('dmcount')),
    nanoBytes: sum(r => r.path.includes('yolox_nano')),
    osnetBytes: sum(r => r.path.includes('osnet')),
    requests: interesting.map(r => ({ method: r.method, path: r.path, status: r.status, bytes: r.bytes, ms: r.ms, aborted: !!r.aborted })),
  };
}
async function waitStatus(pred, timeoutMs) {
  const t0 = Date.now();
  let last = null;
  while (Date.now() - t0 < timeoutMs) {
    last = await evalJson(`({status: document.getElementById('status')?.textContent || '', footer: document.getElementById('buildFooter')?.textContent || '', title: document.title, href: location.href, sw: !!(navigator.serviceWorker && navigator.serviceWorker.controller), events: (window.__tc && window.__tc.loadEvents) ? window.__tc.loadEvents.map(e => e.rel + ' ' + e.source + ' ' + e.bytes + ' ' + Math.round(e.ms) + 'ms') : null, denseReq: window.__tc ? !!window.__tc.denseModelRequested : null})`);
    if (last && !last.error && pred(last)) return { ...last, waitedMs: Date.now() - t0 };
    await sleep(400);
  }
  return { ...(last || {}), waitedMs: Date.now() - t0, timedOut: true };
}

const report = { phase: PHASE, bps: BPS, serverCacheControl: 'no-store', launches: {} };

// Launch 1 — default dense=auto
{
  const from = requests.length;
  const navAt = Date.now();
  await send('Page.navigate', { url: URL_AUTO }, sessionId);
  const ready = await waitStatus(s => /Ready|Camera|error/i.test(s.status), 15000);
  await shot('/tmp/crowd-footer.png');
  await sleep(500);
  const clickAt = Date.now();
  await evalJson(`document.getElementById('startBtn').click(), true`);
  const counted = await waitStatus(s => /Counting people/.test(s.status) || /failed|unavailable/i.test(s.status), 90000);
  report.launches.auto = {
    navToReadyMs: ready.waitedMs,
    clickToStatusMs: counted.waitedMs,
    navToStatusMs: Date.now() - navAt,
    status: counted.status,
    footer: counted.footer,
    sw: counted.sw,
    events: counted.events,
    denseReq: counted.denseReq,
    timedOut: !!counted.timedOut,
    net: snap(from),
    clickAt: clickAt - navAt,
  };
  // Old builds start the 21 MB download on their own about 8s after counting.
  // New builds must stay quiet. Either outcome ends this wait.
  const probeMark = Date.now();
  const probe = await waitStatus(s => {
    const rows = requests.slice(from).filter(r => r.path.includes('dmcount'));
    const done = rows.some(r => r.bytes >= 21560628 && r.ms > 0);
    const quiet = /Counting people/.test(s.status || '') && rows.length === 0 && (Date.now() - probeMark) > 12000;
    return done || quiet;
  }, 25000);
  report.launches.auto.afterProbe = { waitedMs: probe.waitedMs, status: probe.status, sw: probe.sw, events: probe.events, denseReq: probe.denseReq, net: snap(from) };
  // user turns dense mode on
  const enFrom = requests.length;
  const enAt = Date.now();
  await evalJson(`(() => { const el = document.getElementById('setDense'); el.value = 'on'; el.dispatchEvent(new Event('change')); return el.value; })()`);
  const enabled = await waitStatus(s => {
    const rows = requests.slice(enFrom).filter(r => r.path.includes('dmcount'));
    const done = rows.some(r => r.bytes >= 21560628 && r.ms > 0);
    const cached = (s.events || []).some(e => /dmcount/.test(e) && /cache/.test(e));
    return done || cached;
  }, 40000);
  // if denseReq flipped true immediately on after-fix, keep waiting for bytes or cache event
  await sleep(500);
  report.launches.auto.enableDense = {
    waitedMs: Date.now() - enAt,
    status: enabled.status,
    events: enabled.events,
    denseReq: enabled.denseReq,
    net: snap(enFrom),
  };
}

// Launch 2 — dense already enabled, same profile
{
  const from = requests.length;
  const navAt = Date.now();
  await send('Page.navigate', { url: URL_DENSE }, sessionId);
  const ready = await waitStatus(s => /Ready|Camera|error/i.test(s.status), 15000);
  await sleep(500);
  await evalJson(`document.getElementById('startBtn').click(), true`);
  const counted = await waitStatus(s => /Counting people/.test(s.status) || /failed|unavailable/i.test(s.status), 90000);
  const countMark = Date.now();
  const late = await waitStatus(s => {
    const rows = requests.slice(from).filter(r => r.path.includes('dmcount'));
    const done = rows.some(r => r.bytes >= 21560628 && r.ms > 0);
    const cached = (s.events || []).some(e => /dmcount/.test(e) && /cache/.test(e));
    const quiet = /Counting people/.test(s.status) && rows.length === 0 && (Date.now() - countMark) > 8000;
    return done || cached || quiet;
  }, 40000);
  report.launches.denseSecond = {
    navToReadyMs: ready.waitedMs,
    clickToStatusMs: counted.waitedMs,
    wallMs: Date.now() - navAt,
    status: late.status || counted.status,
    footer: late.footer || counted.footer,
    sw: late.sw,
    events: late.events,
    denseReq: late.denseReq,
    timedOut: !!counted.timedOut,
    net: snap(from),
  };
}

// Offline — server still up until we close it, then reload and start
{
  const swState = await evalJson(`navigator.serviceWorker ? navigator.serviceWorker.getRegistration().then(r => ({controller: !!navigator.serviceWorker.controller, scope: r && r.scope, script: r && r.active && r.active.scriptURL})) : null`);
  report.swBeforeOffline = swState;
  server.close(); serverClosed = true;
  await sleep(200);
  const from = requests.length;
  const navAt = Date.now();
  try { await send('Page.navigate', { url: URL_DENSE }, sessionId); } catch (e) { report.offlineNavError = String(e); }
  const ready = await waitStatus(s => /Ready|Camera|error|tap Start/i.test(s.status) || (s.title && s.title.length > 0), 10000);
  let started = null;
  if (ready.status && /Ready|tap Start/.test(ready.status)) {
    try { await evalJson(`document.getElementById('startBtn').click(), true`); } catch (e) { started = { clickError: String(e) }; }
    started = await waitStatus(s => /Counting people/.test(s.status) || /failed|unavailable|error/i.test(s.status), 45000);
  }
  report.offline = {
    wallMs: Date.now() - navAt,
    ready,
    started,
    netAfterClose: snap(from),
  };
  await shot('/tmp/crowd-offline.png');
}

const out = `/tmp/cachetime-${PHASE}.json`;
writeFileSync(out, JSON.stringify(report, null, 2));
console.log('WROTE', out);
console.log(JSON.stringify({
  phase: PHASE,
  auto: {
    clickToStatusMs: report.launches.auto.clickToStatusMs,
    status: report.launches.auto.status,
    wasm: report.launches.auto.net.wasmBytes,
    dm: report.launches.auto.net.dmcountBytes,
    probeDm: report.launches.auto.afterProbe.net.dmcountBytes,
    enableDm: report.launches.auto.enableDense.net.dmcountBytes,
    footer: report.launches.auto.footer,
  },
  second: {
    clickToStatusMs: report.launches.denseSecond.clickToStatusMs,
    status: report.launches.denseSecond.status,
    wasm: report.launches.denseSecond.net.wasmBytes,
    dm: report.launches.denseSecond.net.dmcountBytes,
    events: report.launches.denseSecond.events,
  },
  offline: {
    status: report.offline.ready && report.offline.ready.status,
    started: report.offline.started && report.offline.started.status,
    title: report.offline.ready && report.offline.ready.title,
  },
}, null, 2));
const problems = [];
if (PHASE === 'after') {
  const auto = report.launches.auto;
  const second = report.launches.denseSecond;
  if (auto.afterProbe.net.dmcountBytes !== 0) problems.push('auto launch downloaded dmcount: ' + auto.afterProbe.net.dmcountBytes);
  if (auto.net.wasmBytes < 26_000_000) problems.push('first launch wasm bytes ' + auto.net.wasmBytes);
  if (auto.enableDense.net.dmcountBytes < 20_000_000) problems.push('turning dense on did not download the model: ' + auto.enableDense.net.dmcountBytes);
  if (auto.denseReq) problems.push('dense model was requested before the user turned it on');
  if (second.net.wasmBytes !== 0) problems.push('second launch refetched wasm: ' + second.net.wasmBytes);
  if (second.net.dmcountBytes !== 0) problems.push('second launch refetched dmcount: ' + second.net.dmcountBytes);
  const ev = second.events || [];
  if (!ev.some(e => /wasm/.test(e) && /cache/.test(e))) problems.push('second launch wasm was not a cache hit: ' + JSON.stringify(ev));
  if (!ev.some(e => /dmcount/.test(e) && /cache/.test(e))) problems.push('second launch dmcount was not a cache hit: ' + JSON.stringify(ev));
  if (second.clickToStatusMs > 6000) problems.push('second launch click-to-count still slow: ' + second.clickToStatusMs);
  if (!/crowd-1\.2/.test(auto.footer || '')) problems.push('footer missing version: ' + auto.footer);
  if (!/descarga|dispositivo|densa/.test(auto.footer || '')) problems.push('footer missing Spanish: ' + auto.footer);
  const off = (report.offline.started && report.offline.started.status) || '';
  if (!/Counting people/.test(off)) problems.push('offline start did not reach counting: ' + off + ' ready=' + (report.offline.ready && report.offline.ready.status));
}
console.log(problems.length ? 'PROBLEMS\n' + problems.join('\n') : 'CHECKS OK');
chrome.kill('SIGKILL');
await sleep(400);
if (!serverClosed) server.close();
try { rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }); } catch (e) { console.error('profile cleanup', e.message); }
process.exit(problems.length ? 1 : 0);
