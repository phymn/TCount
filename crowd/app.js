/* TCount Crowd — on-device people counter.
   Detector: YOLOX (Apache-2.0) via ONNX Runtime Web (WebGPU -> WASM fallback; coco-ssd optional fallback).
   Tracker: ByteTrack (Kalman + two-stage association) + simple camera-motion compensation (BoT-SORT idea).
   Dedupe: OSNet re-ID embeddings (torchreid, MIT) + session ledger with a "possible duplicate" band.
   Dense: DM-Count (MIT) density map in a worker, tiled-detection fallback.
   Nothing leaves the device. */
'use strict';
(() => {
const $ = id => document.getElementById(id);
const Q = new URLSearchParams(location.search);
const VERSION = 'crowd-1.2';
const now = () => performance.now();
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const abs = p => new URL(p, location.href).href;
const CACHE_NAME = 'tcount-crowd-assets-v1';

// ------------------------------------------------------------------ settings
const DEFAULTS = { model: 'auto', backend: 'auto', size: 'auto', tiles: 'auto', dense: 'auto', lensMode: 'auto', cmc: 'on',
  high: 0.80, low: 0.65, confirm: 2, threads: 'off', lens: '', v: 2 };
let S = Object.assign({}, DEFAULTS);
try { const o = JSON.parse(localStorage.getItem('tcrowd-settings') || '{}'); if (!o.v || o.v < 2) delete o.confirm; delete o.v; Object.assign(S, o); } catch (e) {} // v2: new default confirm=2
for (const k of Object.keys(DEFAULTS)) if (Q.has(k)) S[k] = typeof DEFAULTS[k] === 'number' ? +Q.get(k) : Q.get(k);
const saveSettings = () => { try { localStorage.setItem('tcrowd-settings', JSON.stringify(S)); } catch (e) {} };

const CFG = {
  trackHigh: 0.45, trackLow: 0.1, newTrack: 0.5, matchThresh: 0.8, lostSeconds: 3.0,
  reidMinH: 56, reidMinW: 20, reidMaxOverlap: 0.3, reidEvery: 900, reidBatch: 6, reidWaitFrames: 3,
  gallerySize: 8, recheckWindowMs: 6000, switchPauseMs: 1500,
  ghostMs: 700, ghostFrames: 8, ghostGoneMs: 2500, darkLuma: 40, veryDarkLuma: 18, lightExitLuma: 48, lightEnterMs: 2500, lightExitMs: 1200, lightWarmupMs: 2000, blackLuma: 6, blackFrames: 15,
};

// ------------------------------------------------------------------ UI helpers
function status(msg, cls = '') { $('status').textContent = msg; $('status').className = cls; }
const banners = new Map();
function banner(key, msg, kind = 'error') {
  if (!msg) { const b = banners.get(key); if (b) { b.remove(); banners.delete(key); } return; }
  let b = banners.get(key);
  if (b && b.dataset.msg === msg) return;
  if (!b) { b = document.createElement('div'); banners.set(key, b); $('banners').appendChild(b); }
  b.className = 'banner ' + (kind === 'error' ? '' : kind); b.dataset.msg = msg; b.dataset.key = key;
  b.innerHTML = ''; const x = document.createElement('button'); x.textContent = '✕'; x.onclick = () => { b.remove(); banners.delete(key); };
  b.appendChild(x); b.appendChild(document.createTextNode(msg));
}
function chip(id, text, cls = '') { const c = $(id); c.textContent = text; c.className = 'chip ' + cls; }
function logErr(where, e) { console.error(where, e); (window.__scriptErrors || []).push(where + ': ' + (e && (e.message || e))); }

// ------------------------------------------------------------------ environment checks
const isLocal = ['127.0.0.1', 'localhost', '[::1]'].includes(location.hostname);
if (location.protocol === 'file:') banner('file', 'Opened as a file — models cannot load. Serve the folder (iSH: python3 -m http.server 8080) and open http://127.0.0.1:8080/crowd/');
else if (!window.isSecureContext) banner('secure', `Camera blocked: open this page at http://127.0.0.1:8080/crowd/ (or https). You opened ${location.origin} — iOS only allows the camera on 127.0.0.1/localhost or https.`);
else if (location.protocol === 'http:' && !isLocal) banner('secure', 'Not a secure origin — open via http://127.0.0.1:8080/crowd/ or https.');
if (window.__ortLoadFailed || typeof ort === 'undefined') {
  banner('ort', 'ONNX Runtime script (vendor/ort.webgpu.min.js) was blocked or is missing. YOLOX cannot run; the coco-ssd fallback (needs internet) will be tried.');
}

// ------------------------------------------------------------------ ORT setup
// Expected asset sizes (bytes) — lets us tell "missing" / "incomplete copy" / "server sent an error page" apart.
const ASSETS = [
  ['vendor/ort.webgpu.min.js', 66417, true], ['vendor/ort-wasm-simd-threaded.asyncify.js', 0, true], ['vendor/ort-wasm-simd-threaded.asyncify.wasm', 26781914, true],
  ['models/yolox_nano_person_w8.onnx', 1117717, true], ['models/yolox_tiny_person_w8.onnx', 5208654, true], ['models/osnet_x0_5_reid_w8.onnx', 875829, true],
  ['models/dmcount_qnrf_w8.onnx', 21560628, false], ['dense-worker.js', 0, true], ['app.js', 0, true],
];
const expectedSize = rel => { const a = ASSETS.find(x => x[0] === rel); return a ? a[1] : 0; };
const MB = n => (n / 1048576).toFixed(1) + ' MB';
const loadDiag = []; // shown in Debug and in error banners
const assetBytes = new Map(); // in-memory bytes for this page (wasm + models)
const inflightAssets = new Map(); // dedupe parallel/warm fetches of the same path
async function cacheGet(rel) {
  if (!('caches' in window)) return null;
  try {
    const c = await caches.open(CACHE_NAME);
    const r = await c.match(abs(rel));
    if (!r) return null;
    const buf = new Uint8Array(await r.arrayBuffer());
    const exp = expectedSize(rel);
    if (exp && buf.length !== exp) { try { await c.delete(abs(rel)); } catch (e) {} return null; }
    return buf;
  } catch (e) { return null; }
}
async function cachePut(rel, buf) {
  if (!('caches' in window)) return;
  try {
    const c = await caches.open(CACHE_NAME);
    await c.put(abs(rel), new Response(buf, { headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(buf.length) } }));
  } catch (e) { /* private mode / quota — HTTP cache may still help */ }
}
// Fetch a file as bytes with progress and a precise error (URL, HTTP status, content-type, size).
// Uses Cache API + default HTTP cache (repeat visits skip the 26 MB+ re-download). Size checks still catch corrupt copies.
async function fetchAsset(rel, onProgress, timeoutMs = 180000) {
  if (assetBytes.has(rel)) { const b = assetBytes.get(rel); onProgress && onProgress(b.length, b.length); return b; }
  if (inflightAssets.has(rel)) {
    const b = await inflightAssets.get(rel);
    onProgress && onProgress(b.length, b.length);
    return b;
  }
  const p = (async () => {
    const url = abs(rel), exp = expectedSize(rel), t0 = now();
    let buf = await cacheGet(rel);
    if (buf) {
      loadDiag.push(`${rel}: ${MB(buf.length)} from Cache API in ${((now() - t0) / 1000).toFixed(2)} s`);
      onProgress && onProgress(buf.length, buf.length);
      assetBytes.set(rel, buf);
      return buf;
    }
    const ctl = typeof AbortController !== 'undefined' ? new AbortController() : null;
    const timer = ctl && setTimeout(() => ctl.abort(), timeoutMs);
    let res;
    try { res = await fetch(url, { cache: 'force-cache', signal: ctl && ctl.signal }); }
    catch (e) { clearTimeout(timer); throw new Error(`${rel}: request failed (${e.name === 'AbortError' ? 'timed out after ' + timeoutMs / 1000 + ' s' : e.message || e}) — is python3 -m http.server still running in iSH?`); }
    const ct = res.headers.get('content-type') || '?';
    if (!res.ok) { clearTimeout(timer); throw new Error(`${rel}: HTTP ${res.status} ${res.statusText || ''} (${url})${res.status === 404 ? (/dmcount/.test(rel) ? ' — part 2 zip not unzipped into crowd/models' : ' — file missing: copy/unzip the whole crowd folder (part 1) again') : ''}`); }
    if (/text\/html/.test(ct) && !/\.html$/.test(rel)) { clearTimeout(timer); throw new Error(`${rel}: server returned an HTML page instead of the file (${url})`); }
    try {
      if (res.body && res.body.getReader && onProgress) {
        const rd = res.body.getReader(), parts = []; let n = 0, lastP = 0;
        for (;;) { const { done, value } = await rd.read(); if (done) break; parts.push(value); n += value.length; if (now() - lastP > 300) { lastP = now(); onProgress(n, exp || +res.headers.get('content-length') || 0); } }
        const u = new Uint8Array(n); let o = 0; for (const p of parts) { u.set(p, o); o += p.length; } buf = u;
      } else buf = new Uint8Array(await res.arrayBuffer());
    } catch (e) { throw new Error(`${rel}: download interrupted after ${((now() - t0) / 1000).toFixed(0)} s (${e.name === 'AbortError' ? 'timed out' : e.message || e})`); }
    finally { clearTimeout(timer); }
    if (exp && buf.length !== exp) throw new Error(`${rel}: file is incomplete/corrupt — got ${buf.length} bytes, expected ${exp}. Re-copy/unzip it.`);
    loadDiag.push(`${rel}: ${MB(buf.length)} in ${((now() - t0) / 1000).toFixed(1)} s, HTTP ${res.status}, ${ct}`);
    assetBytes.set(rel, buf);
    cachePut(rel, buf); // fire-and-forget
    return buf;
  })();
  inflightAssets.set(rel, p);
  try { return await p; }
  finally { inflightAssets.delete(rel); }
}
let ortReady = false, ortWasmBytes = null, ortSetupP = null;
function withTimeout(p, ms, what) { let t; return Promise.race([p, new Promise((_, rej) => { t = setTimeout(() => rej(new Error(what + ' timed out after ' + ms / 1000 + ' s')), ms); })]).finally(() => clearTimeout(t)); }
async function setupOrt(onS) {
  if (typeof ort === 'undefined') return false;
  if (ortReady) return true;
  if (ortSetupP) return ortSetupP;
  ortSetupP = (async () => {
    // iOS Safari: the runtime asks for a 4 GB shared memory reservation, which iOS can refuse ("Out of memory").
    // 1 GB is plenty for these models. Override with ?maxmem=<MB>.
    globalThis.__ortMaxPages = Math.round((+Q.get('maxmem') || 1024) * 16);
    ort.env.wasm.wasmPaths = { mjs: abs('vendor/ort-wasm-simd-threaded.asyncify.js'), wasm: abs('vendor/ort-wasm-simd-threaded.asyncify.wasm') };
    ort.env.wasm.numThreads = window.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1; // threads need COOP/COEP; python http.server cannot send them
    ort.env.wasm.proxy = false;
    ort.env.logLevel = 'error';
    // Download the 26 MB runtime once ourselves: python http.server (iSH) often sends .wasm as application/octet-stream,
    // which makes Safari's streaming compile fail and re-download it; handing over the bytes avoids both problems.
    try {
      ortWasmBytes = await fetchAsset('vendor/ort-wasm-simd-threaded.asyncify.wasm', (n, t) => onS && onS(`Downloading AI runtime ${MB(n)}${t ? ' / ' + MB(t) : ''}…`));
      ort.env.wasm.wasmBinary = ortWasmBytes;
    } catch (e) { loadDiag.push('runtime prefetch failed: ' + e.message); ortSetupP = null; throw e; }
    ortReady = true; return true;
  })();
  return ortSetupP;
}
function errText(e) { return e ? ((e.name && e.name !== 'Error' ? e.name + ': ' : '') + (e.message || String(e))) : 'unknown error'; }
let webgpuCached = undefined;
async function webgpuAvailable() {
  if (webgpuCached !== undefined) return webgpuCached;
  try { if (!navigator.gpu) return (webgpuCached = false); const a = await navigator.gpu.requestAdapter(); return (webgpuCached = !!a); } catch (e) { return (webgpuCached = false); }
}
async function makeSession(url, ep, onS) {
  const rel = url.replace(abs('./'), '').replace(/^\.\//, '');
  const bytes = await fetchAsset(rel, (n, t) => onS && onS(`Downloading ${rel.split('/').pop()} ${MB(n)}${t ? ' / ' + MB(t) : ''}…`));
  const opts = { executionProviders: [ep], graphOptimizationLevel: 'all' };
  try { return await withTimeout(ort.InferenceSession.create(bytes, opts), ep === 'webgpu' ? 45000 : 120000, `${rel} on ${ep}`); }
  catch (e) { throw new Error(`${rel} on ${ep}: ${errText(e)}`); }
}
function detectorKindForAuto(gpu) {
  // Prefer Nano on CPU-only devices so Auto does not download Tiny (5 MB) then throw it away.
  return gpu ? 'tiny' : 'nano';
}
function releaseAsset(rel) { assetBytes.delete(rel); }

// ------------------------------------------------------------------ geometry
function iou(a, b) { // boxes [x,y,w,h]
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9);
}
function ioMin(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (Math.min(a[2] * a[3], b[2] * b[3]) + 1e-9);
}
function nms(dets, thr = 0.45, minThr = 0) { // dets: {box, score}
  dets.sort((a, b) => b.score - a.score); const keep = [];
  for (const d of dets) {
    let ok = true;
    for (const k of keep) { if (iou(d.box, k.box) > thr || (minThr && ioMin(d.box, k.box) > minThr)) { ok = false; break; } }
    if (ok) keep.push(d);
  }
  return keep;
}

// ------------------------------------------------------------------ frame buffers
let frameCanvas = document.createElement('canvas');
let frameCtx = frameCanvas.getContext('2d', { willReadFrequently: true });
const prepCanvas = document.createElement('canvas');
const prepCtx = prepCanvas.getContext('2d', { willReadFrequently: true });
const cropCanvas = document.createElement('canvas'); cropCanvas.width = 128; cropCanvas.height = 256;
const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });
const grayCanvas = document.createElement('canvas');
const grayCtx = grayCanvas.getContext('2d', { willReadFrequently: true });

// ------------------------------------------------------------------ YOLOX detector
class YoloxDetector {
  constructor(kind) { this.kind = kind; this.session = null; this.ep = null; this.name = 'YOLOX-' + (kind === 'tiny' ? 'Tiny' : 'Nano'); this.lastMs = 0; this.gain = 1; }
  get url() { return abs(`models/yolox_${this.kind}_person_w8.onnx`); }
  async load(backendPref, onStatus) {
    const tryEp = async ep => {
      onStatus && onStatus(`Loading ${this.name} (${ep})…`);
      const t0 = now(); const s = await makeSession(this.url, ep, onStatus);
      const x = new ort.Tensor('float32', new Float32Array(3 * 320 * 320).fill(114), [1, 3, 320, 320]);
      await withTimeout(s.run({ images: x }), 60000, 'first run on ' + ep); // warm-up / shader compile
      const t1 = now(); await s.run({ images: x }); // one timed run is enough for backend pick
      return { s, ep, createMs: t1 - t0, runMs: now() - t1 };
    };
    const gpu = await webgpuAvailable();
    this.webgpuAvailable = gpu;
    const cands = [], errs = [];
    if (backendPref === 'webgpu' || backendPref === 'auto') {
      if (gpu && !Q.get('nogpu')) { try { cands.push(await tryEp('webgpu')); } catch (e) { errs.push('WebGPU: ' + errText(e)); logErr('webgpu', e); } }
      else errs.push('WebGPU: not available in this browser');
    }
    // only benchmark the CPU too if the GPU result is slow (two sessions at once cost memory on iPhone)
    if (backendPref === 'wasm' || !cands.length || backendPref === 'auto' && cands[0].runMs > 45) {
      try { cands.push(await tryEp('wasm')); } catch (e) { errs.push('WASM: ' + errText(e)); logErr('wasm', e); }
    }
    if (!cands.length) throw new Error(errs.join(' | '));
    cands.sort((a, b) => a.runMs - b.runMs);
    const best = cands[0];
    for (const c of cands.slice(1)) try { c.s.release && c.s.release(); } catch (e) {}
    this.session = best.s; this.ep = best.ep; this.benchMs = best.runMs;
    this.bench = cands.map(c => `${c.ep}: ${c.runMs.toFixed(0)} ms @320 (init ${c.createMs.toFixed(0)} ms)`);
    this.loadErrors = errs;
    return this;
  }
  // regions: [{x,y,w,h}] in frameCanvas coords; size = long side of the network input
  async detect(regions, size) {
    const all = [];
    for (const r of regions) for (const d of await this.runRegion(r, size)) all.push(d);
    return regions.length > 1 ? nms(all, 0.5, 0.75) : all;
  }
  async runRegion(r, size) {
    const long = Math.max(r.w, r.h), sc = size / long;
    const inW = Math.max(32, Math.ceil(r.w * sc / 32) * 32), inH = Math.max(32, Math.ceil(r.h * sc / 32) * 32);
    const dw = Math.round(r.w * sc), dh = Math.round(r.h * sc);
    if (prepCanvas.width !== inW || prepCanvas.height !== inH) { prepCanvas.width = inW; prepCanvas.height = inH; }
    prepCtx.fillStyle = 'rgb(114,114,114)'; prepCtx.fillRect(0, 0, inW, inH);
    prepCtx.drawImage(frameCanvas, r.x, r.y, r.w, r.h, 0, 0, dw, dh);
    const px = prepCtx.getImageData(0, 0, inW, inH).data, n = inW * inH;
    const t = new Float32Array(3 * n), g = this.gain;
    if (g === 1) { for (let i = 0, j = 0; i < n; i++, j += 4) { t[i] = px[j + 2]; t[n + i] = px[j + 1]; t[2 * n + i] = px[j]; } }
    else { for (let i = 0, j = 0; i < n; i++, j += 4) { t[i] = Math.min(255, px[j + 2] * g); t[n + i] = Math.min(255, px[j + 1] * g); t[2 * n + i] = Math.min(255, px[j] * g); } }
    const t0 = now();
    const res = await this.session.run({ images: new ort.Tensor('float32', t, [1, 3, inH, inW]) });
    this.lastMs = now() - t0; this.lastIn = `${inW}×${inH}`;
    const o = res.output.data; const dets = [];
    let idx = 0;
    for (const stv of [8, 16, 32]) {
      const gh = inH / stv, gw = inW / stv;
      for (let gy = 0; gy < gh; gy++) for (let gx = 0; gx < gw; gx++, idx++) {
        const b = idx * 6, score = o[b + 4] * o[b + 5];
        if (score < CFG.trackLow) continue;
        const cx = (o[b] + gx) * stv, cy = (o[b + 1] + gy) * stv, w = Math.exp(o[b + 2]) * stv, h = Math.exp(o[b + 3]) * stv;
        let x1 = clamp((cx - w / 2) / sc, 0, r.w), y1 = clamp((cy - h / 2) / sc, 0, r.h), x2 = clamp((cx + w / 2) / sc, 0, r.w), y2 = clamp((cy + h / 2) / sc, 0, r.h);
        if (x2 - x1 < 2 || y2 - y1 < 2) continue;
        dets.push({ box: [x1 + r.x, y1 + r.y, x2 - x1, y2 - y1], score });
      }
    }
    return nms(dets, 0.45);
  }
}

// ------------------------------------------------------------------ coco-ssd fallback (CDN, needs internet)
class CocoSsdDetector {
  constructor() { this.name = 'coco-ssd'; this.ep = 'tfjs'; this.lastMs = 0; this.gain = 1; }
  loadScript(src) { return new Promise((res, rej) => { const s = document.createElement('script'); s.src = src; s.onload = res; s.onerror = () => rej(new Error('Script blocked/unreachable: ' + src)); document.head.appendChild(s); }); }
  async load(_, onStatus) {
    onStatus && onStatus('Loading coco-ssd fallback (internet)…');
    if (!window.tf) await this.loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');
    if (!window.cocoSsd) await this.loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3');
    await tf.setBackend('webgl').catch(() => tf.setBackend('cpu')); await tf.ready();
    this.model = await cocoSsd.load({ base: 'lite_mobilenet_v2' }); this.ep = 'tfjs-' + tf.getBackend(); return this;
  }
  async detect(regions) {
    const t0 = now(); const out = [];
    for (const r of regions) {
      const c = document.createElement('canvas'); c.width = r.w; c.height = r.h;
      c.getContext('2d').drawImage(frameCanvas, r.x, r.y, r.w, r.h, 0, 0, r.w, r.h);
      const p = await this.model.detect(c, 50, 0.2);
      for (const d of p) if (d.class === 'person') out.push({ box: [d.bbox[0] + r.x, d.bbox[1] + r.y, d.bbox[2], d.bbox[3]], score: d.score });
    }
    this.lastMs = now() - t0; this.lastIn = 'native';
    return nms(out, 0.5, 0.75);
  }
}

// ------------------------------------------------------------------ Re-ID (OSNet x0.5)
class ReID {
  constructor() { this.session = null; this.ok = false; this.lastMs = 0; this.name = 'OSNet-x0.5'; }
  async load(ep, onStatus) {
    const url = abs('models/osnet_x0_5_reid_w8.onnx');
    const onS = onStatus || (() => {});
    try { this.session = await makeSession(url, ep, onS); this.ep = ep; await this.embed([[0, 0, 8, 16]]); }
    catch (e) { if (ep === 'wasm') throw e; logErr('reid-' + ep, e); this.session = await makeSession(url, 'wasm', onS); this.ep = 'wasm'; await this.embed([[0, 0, 8, 16]]); }
    this.ok = true; return this;
  }
  async embed(boxes) { // boxes in frameCanvas coords -> L2-normalized Float32Array[512]
    const n = boxes.length, HW = 128 * 256; const t = new Float32Array(n * 3 * HW);
    const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
    for (let k = 0; k < n; k++) {
      const [x, y, w, h] = boxes[k];
      cropCtx.drawImage(frameCanvas, x, y, Math.max(1, w), Math.max(1, h), 0, 0, 128, 256);
      const px = cropCtx.getImageData(0, 0, 128, 256).data, base = k * 3 * HW;
      for (let i = 0, j = 0; i < HW; i++, j += 4) {
        t[base + i] = (px[j] / 255 - mean[0]) / std[0];
        t[base + HW + i] = (px[j + 1] / 255 - mean[1]) / std[1];
        t[base + 2 * HW + i] = (px[j + 2] / 255 - mean[2]) / std[2];
      }
    }
    const t0 = now();
    const r = await this.session.run({ input: new ort.Tensor('float32', t, [n, 3, 256, 128]) });
    this.lastMs = now() - t0;
    const f = r.feat.data, D = f.length / n, out = [];
    for (let k = 0; k < n; k++) {
      const v = new Float32Array(D); let s = 0;
      for (let i = 0; i < D; i++) { v[i] = f[k * D + i]; s += v[i] * v[i]; }
      s = 1 / Math.sqrt(s + 1e-12); for (let i = 0; i < D; i++) v[i] *= s; out.push(v);
    }
    return out;
  }
}
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
function meanEmb(list) {
  const D = list[0].length, m = new Float32Array(D);
  for (const v of list) for (let i = 0; i < D; i++) m[i] += v[i];
  let s = 0; for (let i = 0; i < D; i++) s += m[i] * m[i]; s = 1 / Math.sqrt(s + 1e-12);
  for (let i = 0; i < D; i++) m[i] *= s; return m;
}

// ------------------------------------------------------------------ Kalman filter (ByteTrack xyah model; F,Q,R are per-dimension so it splits into 4 exact 2x2 filters)
const WP = 1 / 20, WV = 1 / 160;
class KF {
  constructor(xyah) {
    const h = xyah[3];
    this.x = [...xyah, 0, 0, 0, 0];
    const sp = [2 * WP * h, 2 * WP * h, 1e-2, 2 * WP * h], sv = [10 * WV * h, 10 * WV * h, 1e-5, 10 * WV * h];
    this.P = [0, 1, 2, 3].map(i => [sp[i] * sp[i], 0, sv[i] * sv[i]]); // [Ppp, Ppv, Pvv]
  }
  predict(tracked) {
    if (!tracked) this.x[7] = 0;
    const h = this.x[3];
    const qp = [WP * h, WP * h, 1e-2, WP * h], qv = [WV * h, WV * h, 1e-5, WV * h];
    for (let i = 0; i < 4; i++) {
      this.x[i] += this.x[i + 4];
      const [pp, pv, vv] = this.P[i];
      this.P[i] = [pp + 2 * pv + vv + qp[i] * qp[i], pv + vv, vv + qv[i] * qv[i]];
    }
  }
  update(z) {
    const h = this.x[3]; const r = [WP * h, WP * h, 1e-1, WP * h];
    for (let i = 0; i < 4; i++) {
      const [pp, pv, vv] = this.P[i]; const s = pp + r[i] * r[i];
      const kp = pp / s, kv = pv / s, y = z[i] - this.x[i];
      this.x[i] += kp * y; this.x[i + 4] += kv * y;
      this.P[i] = [(1 - kp) * pp, (1 - kp) * pv, vv - kv * pv];
    }
  }
  shift(dx, dy) { this.x[0] += dx; this.x[1] += dy; }
}
const toXYAH = b => [b[0] + b[2] / 2, b[1] + b[3] / 2, b[2] / b[3], b[3]];
const fromXYAH = x => { const h = Math.max(1, x[3]), w = x[2] * h; return [x[0] - w / 2, x[1] - h / 2, w, h]; };

// ------------------------------------------------------------------ Hungarian (min-cost, rectangular; pairs above thresh rejected)
function hungarian(cost, thresh) {
  const nR = cost.length; if (!nR) return []; const nC = cost[0].length; if (!nC) return [];
  const tr = nR > nC; const n = tr ? nC : nR, m = tr ? nR : nC;
  const a = (i, j) => { const v = tr ? cost[j][i] : cost[i][j]; return v > thresh ? thresh + 1 : v; };
  const INF = 1e18, u = new Float64Array(n + 1), v = new Float64Array(m + 1), p = new Int32Array(m + 1), way = new Int32Array(m + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i; let j0 = 0; const minv = new Float64Array(m + 1).fill(INF), used = new Uint8Array(m + 1);
    do {
      used[j0] = 1; const i0 = p[j0]; let delta = INF, j1 = 0;
      for (let j = 1; j <= m; j++) if (!used[j]) {
        const cur = a(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= m; j++) { if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta; }
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const res = [];
  for (let j = 1; j <= m; j++) if (p[j]) {
    const i = p[j] - 1, jj = j - 1; const r = tr ? jj : i, c = tr ? i : jj;
    if (cost[r][c] <= thresh) res.push([r, c]);
  }
  return res;
}

// ------------------------------------------------------------------ ByteTrack
let TRACK_ID = 1;
class STrack {
  constructor(box, score) { this.box = box; this.detBox = box; this.score = score; this.id = 0; this.state = 1; this.activated = false;
    this.hits = 0; this.startT = now(); this.lastT = now(); this.personId = 0; this.emb = []; this.lastEmbT = 0; this.waitFrames = 0; this.matchedThisFrame = true; this.co = new Set(); this.hiHits = score >= CFG.trackHigh ? 1 : 0; this.pend = 0; }
  activate() { this.kf = new KF(toXYAH(this.box)); this.id = TRACK_ID++; this.state = 1; this.hits = 1; }
  predict() { this.kf.predict(this.state === 1); this.box = fromXYAH(this.kf.x); }
  update(det) {
    this.kf.update(toXYAH(det.box)); this.box = fromXYAH(this.kf.x); this.detBox = det.box; this.score = det.score; if (det.score >= CFG.trackHigh) this.hiHits = (this.hiHits || 0) + 1;
    this.state = 1; this.activated = true; this.hits++; this.lastT = now(); this.matchedThisFrame = true;
  }
}
class ByteTracker {
  constructor() { this.reset(); }
  reset() { this.tracked = []; this.lost = []; this.frame = 0; }
  cost(tracks, dets, fuse) { return tracks.map(t => dets.map(d => 1 - iou(t.box, d.box) * (fuse ? d.score : 1))); }
  update(dets, motion) {
    this.frame++;
    const tNow = now();
    const unconfirmed = this.tracked.filter(t => !t.activated), active = this.tracked.filter(t => t.activated);
    const pool = [...active, ...this.lost];
    for (const t of [...pool, ...unconfirmed]) { t.matchedThisFrame = false; if (motion) t.kf.shift(motion.dx, motion.dy); t.predict(); }
    const high = dets.filter(d => d.score >= CFG.trackHigh), low = dets.filter(d => d.score < CFG.trackHigh && d.score >= CFG.trackLow);
    // stage 1: high-score detections vs tracked + lost tracks
    const usedT = new Set(), usedD = new Set();
    for (const [ti, di] of hungarian(this.cost(pool, high, true), CFG.matchThresh)) { pool[ti].update(high[di]); usedT.add(ti); usedD.add(di); }
    // stage 2: low-score detections rescue remaining *tracked* tracks (occlusion)
    const rem = pool.filter((t, i) => !usedT.has(i) && t.state === 1);
    const used2 = new Set();
    for (const [ri, di] of hungarian(this.cost(rem, low, false), 0.5)) { rem[ri].update(low[di]); used2.add(ri); }
    rem.forEach((t, ri) => { if (!used2.has(ri)) t.state = 2; });
    // unconfirmed tracks vs leftover high detections
    const remHigh = high.filter((d, i) => !usedD.has(i));
    const usedU = new Set(), usedH = new Set();
    for (const [ui, hi] of hungarian(this.cost(unconfirmed, remHigh, true), 0.7)) { unconfirmed[ui].update(remHigh[hi]); usedU.add(ui); usedH.add(hi); }
    unconfirmed.forEach((t, i) => { if (!usedU.has(i)) t.state = 3; });
    // births
    const born = [];
    remHigh.forEach((d, hi) => { if (usedH.has(hi) || d.score < CFG.newTrack) return; const t = new STrack(d.box, d.score); t.activate(); if (this.frame === 1) t.activated = true; born.push(t); });
    const all = [...pool, ...unconfirmed, ...born];
    this.tracked = all.filter(t => t.state === 1);
    this.lost = all.filter(t => t.state === 2 && tNow - t.lastT < CFG.lostSeconds * 1000);
    return { tracked: this.tracked };
  }
}

// ------------------------------------------------------------------ camera-motion compensation: sparse block matching on a 160-px gray frame, people masked out
class MotionEstimator {
  constructor() { this.prev = null; this.W = 160; this.last = { dx: 0, dy: 0, conf: 0, n: 0 }; this.luma = 128; }
  reset() { this.prev = null; }
  estimate(srcW, srcH, masks, doMotion) {
    const W = this.W, H = Math.max(16, Math.round(srcH * W / srcW));
    if (grayCanvas.width !== W || grayCanvas.height !== H) { grayCanvas.width = W; grayCanvas.height = H; this.prev = null; }
    grayCtx.drawImage(frameCanvas, 0, 0, W, H);
    const px = grayCtx.getImageData(0, 0, W, H).data, g = new Float32Array(W * H);
    let sum = 0; const hist = new Uint32Array(256);
    for (let i = 0, j = 0; i < W * H; i++, j += 4) { const v = 0.299 * px[j] + 0.587 * px[j + 1] + 0.114 * px[j + 2]; g[i] = v; sum += v; hist[v | 0]++; }
    // robust brightness of the real camera pixels (this is the raw video frame, no letterbox padding)
    const pct = q => { let acc = 0; const lim = q * W * H; for (let k = 0; k < 256; k++) { acc += hist[k]; if (acc >= lim) return k; } return 255; };
    this.luma = sum / (W * H); this.lumaMed = pct(0.5); this.lumaP98 = pct(0.98);
    const prev = this.prev; this.prev = g;
    if (!prev || !doMotion) return (this.last = { dx: 0, dy: 0, conf: 0, n: 0 });
    const k = W / srcW, B = 12, R = 10, votes = [];
    const ms = masks.map(b => [b[0] * k, b[1] * k, b[2] * k, b[3] * k]);
    for (let by = R + 2; by + B + R + 2 < H; by += 14) for (let bx = R + 2; bx + B + R + 2 < W; bx += 14) {
      if (ms.some(m => bx + B > m[0] && bx < m[0] + m[2] && by + B > m[1] && by < m[1] + m[3])) continue; // skip people (they move on their own)
      let mean = 0, varr = 0;
      for (let y = 0; y < B; y++) for (let x = 0; x < B; x++) mean += prev[(by + y) * W + bx + x];
      mean /= B * B;
      for (let y = 0; y < B; y++) for (let x = 0; x < B; x++) { const d = prev[(by + y) * W + bx + x] - mean; varr += d * d; }
      if (varr / (B * B) < 40) continue; // textureless block
      let best = 1e18, bdx = 0, bdy = 0, second = 1e18;
      for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
        let sad = 0;
        for (let y = 0; y < B; y += 2) { const r1 = (by + y) * W + bx, r2 = (by + y + dy) * W + bx + dx; for (let x = 0; x < B; x += 2) sad += Math.abs(prev[r1 + x] - g[r2 + x]); }
        if (sad < best) { second = best; best = sad; bdx = dx; bdy = dy; } else if (sad < second) second = sad;
      }
      if (best < second * 0.9) votes.push([bdx, bdy]);
    }
    if (votes.length < 5) return (this.last = { dx: 0, dy: 0, conf: 0, n: votes.length });
    const med = arr => { const s = [...arr].sort((a, b) => a - b); return s[s.length >> 1]; };
    const mdx = med(votes.map(v => v[0])), mdy = med(votes.map(v => v[1]));
    const inl = votes.filter(v => Math.abs(v[0] - mdx) <= 1 && Math.abs(v[1] - mdy) <= 1);
    const conf = inl.length / votes.length;
    if (conf < 0.4) return (this.last = { dx: 0, dy: 0, conf, n: votes.length });
    const dx = inl.reduce((a, v) => a + v[0], 0) / inl.length, dy = inl.reduce((a, v) => a + v[1], 0) / inl.length;
    return (this.last = { dx: dx / k, dy: dy / k, conf, n: votes.length });
  }
}

// ------------------------------------------------------------------ session ledger (persistent personIds; never forgets a counted person)
class Ledger {
  constructor() { this.reset(); }
  reset() { this.persons = new Map(); this.nextId = 1; this.events = []; this.ghosts = 0; }
  create(state, emb, thumb, candidate = null, candSim = 0) {
    const p = { id: this.nextId++, state, gallery: emb ? [emb] : [], thumb: null, candidate, candSim, mergedInto: null, created: now(), resolved: false, coSeen: new Map() }; // coSeen: personId -> frames seen together
    this.persons.set(p.id, p); this.events.push({ t: Date.now(), type: 'new:' + state, id: p.id, candidate, candSim });
    return p;
  }
  root(id) { let p = this.persons.get(id); while (p && p.mergedInto) p = this.persons.get(p.mergedInto); return p; }
  live() { return [...this.persons.values()].filter(p => !p.mergedInto); }
  get unique() { let n = 0; for (const p of this.persons.values()) if (!p.mergedInto) n++; return n; }
  get possible() { let n = 0; for (const p of this.persons.values()) if (!p.mergedInto && p.state === 'possible') n++; return n; }
  get unverified() { let n = 0; for (const p of this.persons.values()) if (!p.mergedInto && p.state === 'unverified') n++; return n; }
  addSample(p, emb) { // bounded samples per person, unbounded people
    if (!emb) return;
    if (p.state === 'unverified') p.state = 'new';
    if (p.gallery.length < CFG.gallerySize) { p.gallery.push(emb); return; }
    let bi = -1, bs = -1; p.gallery.forEach((g, i) => { const s = dot(g, emb); if (s > bs) { bs = s; bi = i; } });
    if (bs > 0.97) return; p.gallery[bi] = emb; // replace the most redundant neighbour, keep the rest diverse
  }
  sim(p, emb) { // mean of the top-2 sample similarities
    if (!p.gallery.length) return -1;
    const s = p.gallery.map(g => dot(g, emb)).sort((a, b) => b - a);
    return s.length > 1 ? (s[0] + s[1]) / 2 : s[0];
  }
  best(emb, exclude) {
    let best = null, bs = -1;
    for (const p of this.live()) { if (exclude.has(p.id)) continue; const s = this.sim(p, emb); if (s > bs) { bs = s; best = p; } }
    return { p: best, s: bs };
  }
  coCount(a, b) { return Math.max(a.coSeen.get(b.id) || 0, b.coSeen.get(a.id) || 0); }
  together(a, b) { return this.coCount(a, b) >= 3; } // >=3 verified frames together = different people
  gsim(a, b) { // gallery-to-gallery similarity: mean of the best match of each of a's samples (robust to one odd crop)
    if (!a.gallery.length || !b.gallery.length) return -1;
    const v = a.gallery.map(g => this.sim(b, g)).sort((x, y) => y - x); const k = Math.max(1, Math.ceil(v.length / 2));
    return v.slice(0, k).reduce((x, y) => x + y, 0) / k;
  }
  dropGhosts() { // a "person" seen for a split second and then gone is almost always a noise / fragment box
    let n = 0; const t = now();
    for (const p of this.live()) {
      if (p.state === 'possible' || !p.lastSeen || t - p.lastSeen < CFG.ghostGoneMs) continue;
      if (p.lastSeen - p.firstSeen < CFG.ghostMs && (p.seenFrames || 0) < CFG.ghostFrames) { p.mergedInto = -1; p.ghost = true; this.ghosts = (this.ghosts || 0) + 1; this.events.push({ t: Date.now(), type: 'ghost-dropped', id: p.id }); n++; }
    }
    return n;
  }
  reconcile(held) { // retroactive de-fragmentation: a recently created person whose whole gallery matches an earlier one is the same person
    let n = this.dropGhosts(); const L = this.live(); const recent = now() - 30000;
    for (const a of L) {
      if (a.mergedInto || a.created < recent || a.gallery.length < 2 || a.resolvedTogether) continue;
      let best = null, bs = -1;
      for (const b of L) { if (b === a || b.mergedInto || b.created > a.created || b.gallery.length < 2) continue; if (held && held.has(a.id) && held.has(b.id)) continue;
        if (this.coCount(a, b) >= 10) continue; const s = Math.min(this.gsim(a, b), this.gsim(b, a)); if (s > bs) { bs = s; best = b; } }
      if (!best) continue; const c = this.coCount(a, best);
      if (bs >= 0.86 && c < 3 || bs >= 0.92 && c < 10) { if (this.merge(a.id, best.id, 'reconcile: same appearance ' + bs.toFixed(2), true)) n++; }
    }
    return n;
  }
  merge(fromId, intoId, why, force) {
    const a = this.root(fromId), b = this.root(intoId); if (!a || !b || a === b) return null;
    if (!force && this.together(a, b)) return null; // seen in the same frame = provably different people
    a.mergedInto = b.id; for (const g of a.gallery) this.addSample(b, g);
    for (const [x, n] of a.coSeen) b.coSeen.set(x, Math.max(n, b.coSeen.get(x) || 0));
    this.events.push({ t: Date.now(), type: 'merge', from: a.id, into: b.id, why }); return b;
  }
}

// ------------------------------------------------------------------ camera manager (one active stream; lenses are sequential modes)
const video = $('video'), canvas = $('canvas'), ctx = canvas.getContext('2d');
const cam = { stream: null, track: null, devices: [], current: null, facing: 'environment', caps: {}, nativeZoom: false, torchOn: false, digitalZoom: 1, label: '—', lensKind: 'unknown' };
function lensKindOf(label) {
  const l = (label || '').toLowerCase();
  if (/front|user|facetime|facing front/.test(l)) return 'front';
  if (/ultra ?wide|ultrawide/.test(l)) return 'ultrawide';
  if (/tele/.test(l)) return 'tele';
  if (/triple|dual/.test(l)) return 'multi';
  if (/back|rear|wide|environment|facing back/.test(l)) return 'wide';
  return 'unknown';
}
async function enumerateCams() {
  try {
    const ds = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    cam.devices = ds.map(d => ({ deviceId: d.deviceId, label: d.label || 'camera', kind: lensKindOf(d.label) }));
  } catch (e) { cam.devices = []; }
  const sel = $('setLens'); sel.innerHTML = '';
  const o0 = document.createElement('option'); o0.value = ''; o0.textContent = 'Default (rear)'; sel.appendChild(o0);
  for (const d of cam.devices) { const o = document.createElement('option'); o.value = d.label; o.textContent = `${d.label} [${d.kind}]`; sel.appendChild(o); }
  sel.value = cam.label && cam.devices.some(d => d.label === cam.label) ? cam.label : '';
}
const rearByKind = kind => cam.devices.find(d => d.kind === kind);
async function openCamera(opts = {}) {
  if (cam.stream) cam.stream.getTracks().forEach(t => t.stop());
  cam.stream = null;
  const v = { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } };
  // deviceIds are randomized per page load on iOS: always re-resolve by label
  if (opts.label) { const d = cam.devices.find(d => d.label === opts.label); if (d) v.deviceId = { exact: d.deviceId }; }
  if (!v.deviceId) v.facingMode = { ideal: cam.facing };
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: v }); }
  catch (e) {
    if (!v.deviceId) throw e;
    delete v.deviceId; v.facingMode = { ideal: cam.facing };
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: v });
  }
  cam.stream = stream; cam.track = stream.getVideoTracks()[0];
  video.setAttribute('playsinline', ''); video.muted = true; video.srcObject = stream;
  await video.play().catch(() => {});
  await new Promise(r => video.videoWidth ? r() : (video.onloadedmetadata = r));
  canvas.width = video.videoWidth; canvas.height = video.videoHeight; if (typeof st !== 'undefined') { st.camStart = now(); st.blackRun = 0; }
  cam.label = cam.track.label || 'camera'; cam.lensKind = lensKindOf(cam.label);
  if (cam.lensKind === 'unknown' || cam.lensKind === 'multi') cam.lensKind = cam.facing === 'user' ? 'front' : (cam.lensKind === 'multi' ? 'multi' : 'rear');
  try { cam.caps = cam.track.getCapabilities ? cam.track.getCapabilities() : {}; } catch (e) { cam.caps = {}; }
  cam.nativeZoom = !!(cam.caps.zoom && cam.caps.zoom.max > cam.caps.zoom.min);
  cam.torchOn = false;
  $('torchBtn').style.display = cam.caps.torch ? '' : 'none';
  const zr = $('zoom'); zr.disabled = false;
  if (cam.nativeZoom) { zr.min = cam.caps.zoom.min; zr.max = Math.min(cam.caps.zoom.max, 10); zr.step = cam.caps.zoom.step || 0.1; zr.value = (cam.track.getSettings().zoom || cam.caps.zoom.min); }
  else { zr.min = 1; zr.max = 4; zr.step = 0.1; zr.value = 1; }
  setDigitalZoom(1); updateZoomLabel();
  return true;
}
async function setZoom(z) {
  z = +z;
  if (cam.nativeZoom) {
    try {
      await cam.track.applyConstraints({ advanced: [{ zoom: z }] });
      await new Promise(r => setTimeout(r, 150));
      const got = cam.track.getSettings().zoom;
      if (got !== undefined && Math.abs(got - z) <= 0.05 + (cam.caps.zoom.step || 0)) { setDigitalZoom(1); updateZoomLabel(); markLensChange('native zoom ' + z); return; }
    } catch (e) { logErr('zoom', e); }
    cam.nativeZoom = false; banner('zoom', 'This browser reported camera zoom but did not apply it — switched to digital crop-zoom.', 'info');
    const zr = $('zoom'); zr.min = 1; zr.max = 4; zr.value = clamp(z, 1, 4);
  }
  setDigitalZoom(clamp(z, 1, 4)); updateZoomLabel(); markLensChange('digital zoom ' + z);
}
function setDigitalZoom(z) { cam.digitalZoom = z; $('stage').style.transform = z > 1.001 ? `scale(${z})` : ''; }
function updateZoomLabel() { $('zoomVal').textContent = `${(+$('zoom').value).toFixed(1)}×${cam.nativeZoom ? '' : ' dig'}`; }
async function toggleTorch() {
  try { cam.torchOn = !cam.torchOn; await cam.track.applyConstraints({ advanced: [{ torch: cam.torchOn }] });
    $('torchBtn').style.color = cam.torchOn ? 'var(--warn)' : ''; }
  catch (e) { cam.torchOn = false; banner('torch', 'Torch could not be switched on: ' + (e.message || e), 'warn'); }
}

// ------------------------------------------------------------------ dense crowd estimator (DM-Count in a worker; tiled-detection fallback)
const dense = { worker: null, ready: false, failed: false, busy: false, est: null, estAt: 0, map: null, mapW: 0, mapH: 0, source: '', ms: 0, peak: 0, active: false, autoSince: 0, err: '', reqAt: 0 };
function startDenseWorker() {
  if (dense.worker || dense.failed) return;
  // memory: only start the 2nd runtime once the main models are up and the session has been running a while (iPhone)
  if (!detector || !running || now() - (st.runStart || 0) < (Q.get('denseNow') ? 0 : 8000) || typeof ort === 'undefined') return;
  try {
    dense.worker = new Worker('dense-worker.js'); dense.startAt = now();
    dense.worker.onmessage = e => {
      const m = e.data;
      if (m.type === 'ready') { dense.ready = true; dense.ep = m.ep; }
      else if (m.type === 'error') { dense.busy = false; dense.failed = true; dense.err = m.error; banner('dense', 'Density model failed (' + m.error + ') — using a tiled-detection estimate instead.', 'warn'); }
      else if (m.type === 'result') { dense.busy = false; dense.est = m.count; dense.estAt = now(); dense.map = m.map; dense.mapW = m.w; dense.mapH = m.h; dense.ms = m.ms; dense.source = 'density'; dense.peak = Math.max(dense.peak, m.count); }
    };
    dense.worker.onerror = e => { dense.failed = true; dense.err = e.message || 'worker error'; banner('dense', 'Density worker could not start — using a tiled-detection estimate.', 'warn'); };
    dense.worker.postMessage({ type: 'init', base: abs('./'), maxPages: globalThis.__ortMaxPages, wasmBinary: ortWasmBytes || undefined, eps: dense.eps || (Q.get('denseEp') ? [Q.get('denseEp')] : ['webgpu', 'wasm']) });
  } catch (e) { dense.failed = true; dense.err = e.message; }
}
function requestDense() {
  if (dense.busy && now() - dense.reqAt > 15000) { // stuck (e.g. a very slow GPU): restart the worker on the CPU
    try { dense.worker.terminate(); } catch (e) {}
    dense.worker = null; dense.ready = false; dense.busy = false; dense.eps = ['wasm']; startDenseWorker(); return false;
  }
  if (!dense.ready && dense.worker && now() - dense.startAt > 20000 && (dense.eps || ['webgpu'])[0] === 'webgpu') { // GPU init too slow
    try { dense.worker.terminate(); } catch (e) {}
    dense.worker = null; dense.eps = ['wasm']; startDenseWorker(); return false;
  }
  if (!dense.worker || !dense.ready || dense.busy) return false;
  const W = dense.ms > 3500 ? 384 : 512, H = Math.max(32, Math.round(frameCanvas.height * W / frameCanvas.width / 32) * 32); // slower devices: smaller input
  const c = document.createElement('canvas'); c.width = W; c.height = H; const cx = c.getContext('2d');
  cx.drawImage(frameCanvas, 0, 0, W, H);
  const img = cx.getImageData(0, 0, W, H);
  dense.busy = true; dense.reqAt = now();
  dense.worker.postMessage({ type: 'run', w: W, h: H, data: img.data.buffer }, [img.data.buffer]);
  return true;
}

// ------------------------------------------------------------------ app state
const tracker = new ByteTracker(), ledger = new Ledger(), motion = new MotionEstimator();
let detector = null, reid = null, running = false, loopGen = 0, loadedKey = '';
const st = { fps: 0, lastFrameT: 0, inFrame: 0, size: 640, tiles: false, noDetSince: 0, smallSince: 0, lumaEma: -1, light: 'ok',
  pauseUntil: 0, lastLensSwitch: 0, csv: [], lastCsv: 0, timings: {}, lastDets: 0, crowdedness: 0, ladder: '', tiledEstAt: 0, reidQueue: 0, frames: 0 };
window.__tc = { st, ledger, tracker, dense, cam, S, CFG, get detector() { return detector; }, get reid() { return reid; }, get running() { return running; } };

function confirmFrames() { return Math.max(1, S.confirm | 0); } // same in all light: register people as soon as they are seen; late de-dup cleans extras
function markLensChange(why) {
  // coordinates break on a lens/zoom change: drop tracks (ledger keeps everyone), pause counting decisions briefly, re-match by appearance
  const vis = tracker.tracked.filter(t => t.matchedThisFrame && t.personId).map(t => ledger.root(t.personId)).filter(Boolean).map(p => p.id);
  st.carry = { ids: new Set(vis), until: now() + 8000 }; // people visible at the switch: new arrivals right after it may be them again
  tracker.reset(); motion.reset(); st.pauseUntil = now() + CFG.switchPauseMs; st.lastLensSwitch = now();
  ledger.events.push({ t: Date.now(), type: 'lens', why });
}

let reidLoadP = null;
function ensureReid(ep) {
  if (reid && reid.ok) return Promise.resolve(reid);
  if (reidLoadP) return reidLoadP;
  reidLoadP = (async () => {
    try {
      // Quiet progress: do not steal the "Counting…" status once the loop is up.
      const onS = m => { if (!running) status(m, 'running'); };
      if (!running) onS('Loading re-ID model…');
      reid = await new ReID().load(ep, onS);
      banner('reid');
      releaseAsset('models/osnet_x0_5_reid_w8.onnx');
      loadDiag.push('re-ID ready (' + reid.ep + ')');
      return reid;
    } catch (e) {
      logErr('reid', e); reid = null; reidLoadP = null;
      loadDiag.push('re-ID failed: ' + errText(e));
      banner('reid', 'Re-ID model failed to load — people who leave and come back may be counted again. (' + errText(e) + ')', 'warn');
      return null;
    }
  })();
  return reidLoadP;
}

async function loadModels() {
  const want = S.model, key = want + '|' + S.backend;
  if (detector && loadedKey === key) return;
  detector = null; reid = null; reidLoadP = null;
  const onS = m => status(m, 'running');
  let ortOk = false;
  const gpuHint = want !== 'cocossd' ? await webgpuAvailable() : false;
  const kind = want === 'auto' ? detectorKindForAuto(gpuHint && !Q.get('nogpu')) : want;
  // Overlap runtime download with detector (and re-ID) bytes — the big wins on cold iSH / first visit.
  const detRel = want !== 'cocossd' && kind !== 'cocossd' ? `models/yolox_${kind}_person_w8.onnx` : null;
  const prog = { runtime: '', detector: '' };
  const showProg = () => {
    const parts = [prog.runtime, prog.detector].filter(Boolean);
    if (parts.length) onS('Downloading ' + parts.join(' · ') + '…');
  };
  const detPrefetch = detRel ? fetchAsset(detRel, (n, t) => { prog.detector = `detector ${MB(n)}${t ? '/' + MB(t) : ''}`; showProg(); }).catch(e => { loadDiag.push('detector prefetch: ' + errText(e)); return null; }) : null;
  const reidPrefetch = want !== 'cocossd' ? fetchAsset('models/osnet_x0_5_reid_w8.onnx').catch(e => { loadDiag.push('re-ID prefetch: ' + errText(e)); return null; }) : null;
  if (want !== 'cocossd') {
    try {
      ortOk = await setupOrt((m) => {
        // setupOrt status strings already include "AI runtime …"; keep combined line when detector is also mid-flight.
        if (/AI runtime/.test(m)) { prog.runtime = m.replace(/^Downloading\s+/i, ''); showProg(); }
        else onS(m);
      });
    } catch (e) { logErr('ort-setup', e); banner('model', 'AI runtime could not be downloaded: ' + errText(e) + ' — tap 🩺 Self-test and screenshot it. Trying the coco-ssd fallback (needs internet).'); }
  }
  if (detPrefetch) await detPrefetch;
  if (want !== 'cocossd' && ortOk) {
    try {
      let d;
      try { d = await new YoloxDetector(kind).load(S.backend, onS); }
      catch (e) {
        if (kind !== 'tiny') throw e;
        loadDiag.push('YOLOX-Tiny failed: ' + errText(e)); logErr('tiny', e);
        onS('YOLOX-Tiny failed — trying YOLOX-Nano…');
        d = await new YoloxDetector('nano').load(S.backend, onS);
      }
      if (want === 'auto' && d.ep === 'wasm' && d.benchMs > 60 && d.kind === 'tiny') {
        onS('CPU backend is slow for YOLOX-Tiny — using YOLOX-Nano…');
        try { d.session.release && await d.session.release(); } catch (e) {} // free memory before the second model (iPhone)
        releaseAsset('models/yolox_tiny_person_w8.onnx');
        d = await new YoloxDetector('nano').load('wasm', onS);
      }
      detector = d; banner('model');
      releaseAsset(`models/yolox_${d.kind}_person_w8.onnx`);
    } catch (e) {
      logErr('model', e); loadDiag.push('YOLOX failed: ' + errText(e));
      banner('model', 'YOLOX model failed to load: ' + errText(e) + ' — tap 🩺 Self-test and screenshot it. Trying the coco-ssd fallback (needs internet).');
    }
  }
  if (!detector) {
    try { detector = await new CocoSsdDetector().load(null, onS); }
    catch (e) { banner('model2', 'coco-ssd fallback also failed: ' + (e.message || e)); throw new Error('No detector could be loaded'); }
  }
  if (detector instanceof YoloxDetector) {
    // Core counting starts without waiting for OSNet; re-ID attaches when ready.
    if (reidPrefetch) reidPrefetch.then(() => ensureReid(detector.ep));
    else ensureReid(detector.ep);
    if (!detector.webgpuAvailable) banner('gpu', 'WebGPU is not available here — running on the CPU (WASM). Safari 26 / iOS 26 enables WebGPU.', 'info');
  } else banner('reid', 'Re-ID is unavailable with the coco-ssd fallback — people who leave and return may be recounted.', 'warn');
  st.size = S.size === 'auto' ? (detector.ep === 'webgpu' ? 640 : 512) : +S.size;
  loadedKey = key;
}

// Start downloading the AI runtime (+ default detector) before the user taps Start.
function warmPrefetch() {
  if (location.protocol === 'file:' || typeof ort === 'undefined' || S.model === 'cocossd') return;
  const run = async () => {
    try {
      const onS = m => { if (!running && !detector) status(m, 'ready'); };
      status('Preparing AI models…', 'ready');
      const gpu = await webgpuAvailable();
      const kind = S.model === 'auto' ? detectorKindForAuto(gpu && !Q.get('nogpu')) : (S.model === 'tiny' || S.model === 'nano' ? S.model : 'nano');
      const detRel = `models/yolox_${kind}_person_w8.onnx`;
      // Parallel: 26 MB runtime + detector ONNX (re-ID waits until Start so we don't thrash iPhone memory early).
      await Promise.all([
        setupOrt(onS),
        fetchAsset(detRel, (n, t) => onS(`Preparing detector ${MB(n)}${t ? ' / ' + MB(t) : ''}…`)),
      ]);
      if (!running && !detector) status('Ready — tap Start', 'ready');
      loadDiag.push('warm prefetch done');
    } catch (e) { loadDiag.push('warm prefetch: ' + errText(e)); if (!running && !detector) status('Ready — tap Start', 'ready'); }
  };
  if (typeof requestIdleCallback === 'function') requestIdleCallback(() => { run(); }, { timeout: 1500 });
  else setTimeout(run, 200);
}

// ---------------------------------------------------------------- per-frame processing
function regionsFor(W, H) {
  const z = cam.digitalZoom, cw = W / z, ch = H / z;
  const base = { x: (W - cw) / 2, y: (H - ch) / 2, w: cw, h: ch };
  const regs = [base];
  if (st.tiles) { // inference-level zoom: 2x2 overlapping tiles of the visible area (no stream change, no coordinate break)
    const tw = base.w * 0.6, th = base.h * 0.6;
    for (const fy of [0, 1]) for (const fx of [0, 1]) regs.push({ x: base.x + fx * (base.w - tw), y: base.y + fy * (base.h - th), w: tw, h: th });
  }
  return regs.map(r => ({ x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.w), h: Math.round(r.h) }));
}

// Frame grab. Some iOS versions return an all-black image when a camera <video> is drawn into a canvas that was
// created with willReadFrequently (CPU canvas). If frames read as pure black while the video plays, try other paths.
const GRAB_MODES = ['canvas (CPU)', 'canvas (GPU)', 'ImageBitmap'];
async function grabFrame(W, H) {
  if (st.grabMode === 2 && typeof createImageBitmap === 'function') {
    const bm = await createImageBitmap(video); frameCtx.drawImage(bm, 0, 0, W, H); bm.close && bm.close();
  } else frameCtx.drawImage(video, 0, 0, W, H);
}
function setGrabMode(m) {
  st.grabMode = m; const W = frameCanvas.width, H = frameCanvas.height;
  frameCanvas = document.createElement('canvas'); frameCanvas.width = W; frameCanvas.height = H;
  frameCtx = frameCanvas.getContext('2d', m === 1 ? {} : { willReadFrequently: true });
}
function updateLight(mo) {
  const tNow = now(), black = mo.lumaP98 <= CFG.blackLuma;
  st.lumaMed = mo.lumaMed; st.lumaP98 = mo.lumaP98; st.lumaMean = mo.luma;
  st.blackRun = black ? (st.blackRun || 0) + 1 : 0;
  if (black && st.blackRun >= CFG.blackFrames && tNow - (st.camStart || 0) > 1500) {
    if ((st.grabMode || 0) < GRAB_MODES.length - 1) { setGrabMode((st.grabMode || 0) + 1); st.blackRun = 0; loadDiag.push('black frames: switched frame grab to ' + GRAB_MODES[st.grabMode]); }
    else banner('black', 'The camera picture reads as completely BLACK inside the app (tried ' + GRAB_MODES.join(', ') + '), so nothing can be detected. Cover/uncover the lens, tap Stop then Start, or reload. If it persists, tap 🩺 Self-test and screenshot it.');
  }
  if (!black) { if (st.blackRun === 0 && st.wasBlack) { banner('black'); } st.wasBlack = false; } else if (st.blackRun >= CFG.blackFrames) st.wasBlack = true;
  if (black) return; // a black frame is a capture problem, not darkness: keep the light state unchanged
  const v = mo.lumaMed; // median: a few bright/dark objects or a black border cannot flip the state
  st.lumaEma = st.lumaEma < 0 ? v : st.lumaEma * 0.85 + v * 0.15;
  if (tNow - (st.camStart || 0) < CFG.lightWarmupMs) return; // auto-exposure is still settling after the camera starts
  const want = st.lumaEma < CFG.veryDarkLuma ? 'verydark' : st.lumaEma < CFG.darkLuma ? 'dark' : st.lumaEma > CFG.lightExitLuma ? 'ok' : st.light;
  if (want === st.light) { st.lightPendSince = 0; return; }
  if (!st.lightPendSince || st.lightPend !== want) { st.lightPend = want; st.lightPendSince = tNow; }
  if (tNow - st.lightPendSince >= (want === 'ok' ? CFG.lightExitMs : CFG.lightEnterMs)) { st.light = want; st.lightPendSince = 0; if (want !== 'ok') st.everDark = true; }
}
async function step() {
  const T = {}; let t0 = now();
  const W = video.videoWidth, H = video.videoHeight;
  if (frameCanvas.width !== W || frameCanvas.height !== H) { frameCanvas.width = W; frameCanvas.height = H; }
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  await grabFrame(W, H);
  T.grab = now() - t0; t0 = now();
  const mo = motion.estimate(W, H, tracker.tracked.map(t => t.box), S.cmc === 'on');
  updateLight(motion);
  detector.gain = st.light === 'ok' ? 1 : clamp(100 / Math.max(8, st.lumaEma), 1, 4); // brighten the detector input only (does not raise sensitivity)
  T.cmc = now() - t0; t0 = now();
  const regs = regionsFor(W, H);
  const dets = await detector.detect(regs, st.size);
  T.det = now() - t0; t0 = now();
  const { tracked } = tracker.update(dets, Math.abs(mo.dx) + Math.abs(mo.dy) > 0.5 ? mo : null);
  T.track = now() - t0; t0 = now();
  await processIdentities(tracked, W, H, dets);
  T.reid = now() - t0;
  const visible = tracked.filter(t => t.matchedThisFrame);
  st.inFrame = visible.filter(t => t.personId).length;
  const hd = dets.filter(d => d.score >= CFG.trackHigh);
  st.lastDets = hd.length;
  let overl = 0; for (let i = 0; i < hd.length; i++) for (let j = i + 1; j < hd.length; j++) if (iou(hd[i].box, hd[j].box) > 0.3) overl++;
  st.crowdedness = hd.length ? Math.min(1, 2 * overl / hd.length) : 0;
  autoPolicies(hd, W, H, dets.length);
  st.timings = T; st.frames++;
  const tf = now(); if (st.lastFrameT) st.fps = st.fps ? st.fps * 0.8 + 0.2 * (1000 / (tf - st.lastFrameT)) : 1000 / (tf - st.lastFrameT); st.lastFrameT = tf;
  if (S.size === 'auto' && detector instanceof YoloxDetector) adaptSize(T.det / regs.length);
}
function adaptSize(msPerRegion) {
  const sizes = [416, 512, 640, 800];
  let i = sizes.indexOf(st.size); if (i < 0) i = 1;
  const small = st.smallSince && now() - st.smallSince > 1500;
  if (msPerRegion > 250 && i > 0) st.size = sizes[i - 1];
  else if (msPerRegion < 45 && small && i < sizes.length - 1) st.size = sizes[i + 1];
}

// ---------------------------------------------------------------- identity decisions (trackId -> persistent personId)
function cropQuality(t, others, W, H) {
  const b = t.detBox || t.box;
  if (b[3] < CFG.reidMinH || b[2] < CFG.reidMinW) return 0;
  if (!t.matchedThisFrame || b !== t.detBox) return 0;
  let maxOv = 0; for (const o of others) if (o !== b) maxOv = Math.max(maxOv, ioMin(b, o));
  if (maxOv > CFG.reidMaxOverlap) return 0; // occluded / overlapping crop would mix two people's appearance
  const m = 3; if (b[0] < m || b[1] < m || b[0] + b[2] > W - m || b[1] + b[3] > H - m) return 0; // cut off by the frame edge
  if ((t.score || 0.5) < 0.3) return 0; // weak detection (dark noise / body fragment): appearance unreliable
  return Math.min(1, b[3] / 160) * (t.score || 0.5);
}
function thumbOf() { return null; } // privacy: no snapshots/images are ever stored — only numeric appearance embeddings in memory
function carryCandidate(held, emb) { // after a lens/zoom switch coordinates break: treat arrivals as uncertain instead of surely new
  if (!st.carry || now() > st.carry.until) return null;
  let best = null, bs = -2;
  for (const id of st.carry.ids) {
    const p = ledger.root(id); if (!p || held.has(p.id)) continue;
    const sc = emb && p.gallery.length ? ledger.sim(p, emb) : (p.gallery.length ? -0.5 : 0); // prefer people we had no appearance for
    if (sc > bs) { bs = sc; best = p; }
  }
  if (best) st.carry.ids.delete(best.id);
  return best;
}
async function processIdentities(tracked, W, H, dets) {
  const tNow = now(), paused = tNow < st.pauseUntil;
  const visible = tracked.filter(t => t.matchedThisFrame && t.activated);
  const otherBoxes = [...dets.filter(d => d.score >= 0.3).map(d => d.box), ...tracked.filter(t => !t.matchedThisFrame).map(t => t.box)];
  // 1) embeddings: undecided confirmed tracks first, then periodic refresh (not every frame)
  const need = [];
  for (const t of visible) {
    const q = reid ? cropQuality(t, otherBoxes, W, H) : 0; t.q = q;
    if (!q) { if (reid && t.personId) { t.wasOcc = true; t.verified = false; } continue; }
    if (t.wasOcc) { t.occluded = true; t.wasOcc = false; }
    const undecided = !t.personId && t.hits >= confirmFrames() - 1;
    const occ = t.occluded && tNow - t.lastEmbT > 150; // just reappeared from behind someone: re-verify identity first
    const refresh = t.personId && (tNow - t.lastEmbT > CFG.reidEvery || occ);
    if (undecided || refresh) need.push({ t, pri: undecided ? 1e9 : occ ? 1e8 : tNow - t.lastEmbT });
  }
  need.sort((a, b) => b.pri - a.pri);
  const batch = need.slice(0, CFG.reidBatch);
  st.reidQueue = need.length;
  if (batch.length && reid) {
    try {
      const embs = await reid.embed(batch.map(n => n.t.detBox || n.t.box));
      batch.forEach((n, i) => { n.t.occluded = false; n.t.emb.push(embs[i]); if (n.t.emb.length > 6) n.t.emb.shift(); n.t.lastEmbT = tNow; n.t.newEmb = embs[i]; });
    } catch (e) { logErr('reid-run', e); }
  }
  // persons held by tracks visible right now — two people visible together cannot be the same person
  const held = new Map();
  for (const t of visible) if (t.personId) { const r = ledger.root(t.personId); if (r) { t.personId = r.id; held.set(r.id, t); } else t.personId = 0; }
  for (const [id] of held) { const p = ledger.persons.get(id); p.firstSeen = p.firstSeen || tNow; p.lastSeen = tNow; p.seenFrames = (p.seenFrames || 0) + 1; }
  // 2) decisions for confirmed tracks (N frames above threshold)
  for (const t of visible) {
    if (t.personId || paused || t.hits < confirmFrames()) continue;
    if ((t.hiHits || 0) < confirmFrames()) continue; // only confidently-detected tracks may add a person (blocks dark-noise / fragment boxes)
    if (reid && !t.emb.length && t.waitFrames++ < CFG.reidWaitFrames) continue; // no usable crop yet (cut off / overlapping): wait a few frames at most
    // Re-ID is deferred after first count: briefly wait so early people get embeddings when the model lands mid-session.
    if (!reid && reidLoadP && t.waitFrames++ < CFG.reidWaitFrames + 8) continue;
    const thumb = thumbOf(t.detBox || t.box);
    if (!t.emb.length) {
      const carryC = carryCandidate(new Set([...held.keys(), ...t.co]), null);
      const p = carryC ? ledger.create('possible', null, thumb, carryC.id, 0) : ledger.create('unverified', null, thumb);
      t.personId = p.id; held.set(p.id, t); continue;
    }
    const k0 = t.emb.length, e = k0 >= 2 && dot(t.emb[k0 - 1], t.emb[k0 - 2]) >= 0.75 ? meanEmb(t.emb.slice(-2)) : meanEmb(t.emb);
    const { p: bp, s } = ledger.best(e, new Set([...held.keys(), ...t.co]));
    if (reid && (!bp || s < S.high)) {
      // swap check: this crop strongly matches someone another visible track claims. After a crossing the tracker often
      // swaps IDs, so verify the claimant's current appearance before calling this a new person.
      let hp = null, hs = -1;
      for (const [id] of held) { if (t.co.has(id)) continue; const o = ledger.persons.get(id); const so = o ? ledger.sim(o, e) : -1; if (so > hs) { hs = so; hp = o; } }
      const u = hp && held.get(hp.id);
      if (hp && u && u !== t && hs >= S.high && hs > s + 0.08) {
        const fresh = u.lastEmbT > tNow - 400 && u.emb.length;
        const us = fresh ? ledger.sim(hp, u.emb[u.emb.length - 1]) : -1;
        if (fresh && us < S.low) { // the claimant no longer looks like hp: they swapped. Give hp to this track, re-decide the claimant
          u.personId = 0; u.emb = u.emb.slice(-1); u.waitFrames = 0; u.verified = false; held.delete(hp.id);
          t.personId = hp.id; held.set(hp.id, t); ledger.addSample(hp, e); t.verified = true;
          ledger.events.push({ t: Date.now(), type: 'swap-reclaimed', id: hp.id, sim: hs }); continue;
        }
        u.occluded = true; // re-check the claimant next frame; if it was a swap, the duplicate is merged by the late cleanup
      }
    }
    if (bp && s >= S.high) { t.personId = bp.id; ledger.addSample(bp, e); ledger.events.push({ t: Date.now(), type: 'reuse', id: bp.id, sim: s }); }
    else if (bp && s >= S.low) { const p = ledger.create('possible', e, thumb, bp.id, s); t.personId = p.id; }
    else {
      const carryC = carryCandidate(new Set([...held.keys(), ...t.co]), e);
      if (carryC) { const p = ledger.create('possible', e, thumb, carryC.id, Math.max(0, s)); p.why = 'appeared right after a lens/zoom change'; t.personId = p.id; }
      else { const p = ledger.create('new', e, thumb, null, Math.max(0, s)); t.personId = p.id; }
    }
    held.set(t.personId, t); t.verified = true;
    const pp = ledger.persons.get(t.personId); if (pp && pp.origin === undefined) { pp.origin = t.id; pp.fromSwitch = !!t.reDecide; t.reDecide = false; pp.dbg = { tid: t.id, box: (t.detBox || t.box).map(Math.round), q: +(t.q || 0).toFixed(2), ne: t.emb.length, hits: t.hits, wait: t.waitFrames, ft: window.__fake ? +window.__fake.time().toFixed(1) : null, nVis: visible.length, nDet: dets.length }; }
  }
  // 3) accumulate evidence: auto-merge / resolve ambiguous people as more crops arrive
  for (const t of visible) {
    if (!t.personId || !t.newEmb) continue;
    const e = t.newEmb; t.newEmb = null;
    const p = ledger.root(t.personId); if (!p) continue;
    const cons = p.gallery.length ? ledger.sim(p, e) : 1;
    if (p.gallery.length >= 2 && cons < S.low - 0.15) t.bad = (t.bad || []).concat([e]); else t.bad = [];
    if (t.bad.length >= 2) { // two consecutive clean crops disagree with this person (one bad crop is not enough)
      const e2 = meanEmb(t.bad); t.bad = []; const e = e2;
      // appearance no longer matches this person: the tracker probably swapped IDs when people crossed.
      // Detach and re-decide from fresh crops (the person stays counted; this track gets re-matched).
      t.emb = [e];
      let q = null, qs = -1;
      for (const o of ledger.live()) { if (o.id === p.id || ledger.together(o, p) && !held.has(o.id)) continue; const so = ledger.sim(o, e); if (so > qs) { qs = so; q = o; } }
      const u = q && held.get(q.id);
      if (q && qs >= S.low) {
        if (u && u !== t) { u.personId = p.id; held.set(p.id, u); u.occluded = true; u.emb = []; }
        else {
          held.delete(p.id);
          // p was created by this very track moments ago (after an earlier swap) and nobody else holds it: it was a phantom, fold it in
          if (p.fromSwitch && p.origin === t.id && tNow - p.created < 15000 && ledger.merge(p.id, q.id, 'phantom after ID switch', true)) ledger.events.push({ t: Date.now(), type: 'phantom-merged', id: p.id, into: q.id });
        }
        t.personId = q.id; held.set(q.id, t); ledger.addSample(q, e);
        ledger.events.push({ t: Date.now(), type: u && u !== t ? 'id-swap-fixed' : 'id-fixed', id: p.id, into: q.id, sim: qs });
      } else { // nobody matches: re-decide this track from fresh crops
        ledger.events.push({ t: Date.now(), type: 'id-switch?', id: p.id, sim: cons });
        held.delete(p.id); t.personId = 0; t.waitFrames = 0; t.reDecide = true;
      }
      continue;
    }
    if (cons >= S.low) { ledger.addSample(p, e); t.verified = true; } // keep galleries clean: only add consistent samples
    else { t.emb.pop(); t.verified = false; }
    if (!t.emb.length) continue;
    const te = meanEmb(t.emb);
    if (p.state === 'possible' && !p.resolved) {
      const c = ledger.root(p.candidate);
      if (!c || (held.has(c.id) && held.get(c.id) !== t && p.candSim < S.high)) { p.state = 'new'; p.resolved = true; ledger.events.push({ t: Date.now(), type: 'resolve-new', id: p.id, why: 'seen together' }); continue; }
      const s = ledger.sim(c, te); p.candSim = s;
      if (s >= S.high && t.emb.length >= 2 && ledger.merge(p.id, c.id, 'auto: more evidence ' + s.toFixed(2), s >= S.high + 0.05 && ledger.coCount(p, c) < 6 || s >= 0.9 && ledger.coCount(p, c) < 10)) { t.personId = c.id; held.delete(p.id); held.set(c.id, t); }
      else if (s < S.low && t.emb.length >= 3 && c.gallery.length) { p.state = 'new'; p.resolved = true; ledger.events.push({ t: Date.now(), type: 'resolve-new', id: p.id, sim: s }); }
    } else if (p.state === 'new' && !p.resolved && tNow - p.created < CFG.recheckWindowMs && t.emb.length >= 2) {
      let best = null, bs = -1;
      for (const o of ledger.live()) { if (o.id === p.id || held.has(o.id) || ledger.coCount(o, p) >= 10) continue; const s = ledger.sim(o, te); if (s > bs) { bs = s; best = o; } }
      if (best && bs >= S.high + 0.03 && t.emb.length >= 3 && ledger.merge(p.id, best.id, 'auto: late match ' + bs.toFixed(2), bs >= 0.9)) { t.personId = best.id; held.delete(p.id); held.set(best.id, t); }
      else if (best && bs >= S.low + 0.05 && bs > p.candSim + 0.05) { p.state = 'possible'; p.candidate = best.id; p.candSim = bs; ledger.events.push({ t: Date.now(), type: 'flag-possible', id: p.id, candidate: best.id, sim: bs }); }
    }
  }
  // co-visibility memory: people seen in the same frame are provably different (blocks future false merges)
  // only identities verified by a clean crop count here, so a tracker ID swap during a crossing cannot poison this memory
  const ids = [...held.entries()].filter(([, t]) => t.verified && t.q > 0 && tNow - t.lastEmbT < 2 * CFG.reidEvery).map(([id]) => id);
  if (ids.length <= 120) for (const a of ids) { const pa = ledger.persons.get(a); for (const b of ids) if (a !== b) pa.coSeen.set(b, (pa.coSeen.get(b) || 0) + 1); }
  for (const t of visible) { if (!t.verified && t.personId) continue; for (const id of ids) if (id !== t.personId) t.co.add(id); if (t.personId) { const p = ledger.root(t.personId); if (p) for (const x of t.co) if (x !== p.id && !p.coSeen.has(x)) p.coSeen.set(x, 3); } }
  if (tNow - (st.lastReconcile || 0) > 1000) { st.lastReconcile = tNow; if (ledger.reconcile(held)) for (const t of visible) if (t.personId) { const r = ledger.root(t.personId); if (r) t.personId = r.id; } }
  for (const [pid, t] of held) { const p = ledger.persons.get(pid); if (p && p.state === 'possible' && !p.resolved) { const c = ledger.root(p.candidate); if (c && ledger.together(p, c) && p.candSim < S.high) { p.state = 'new'; p.resolved = true; ledger.events.push({ t: Date.now(), type: 'resolve-new', id: p.id, why: 'seen together' }); } } }
}

// ---------------------------------------------------------------- auto policies: tiling, lens ladder, dense mode
function autoPolicies(hd, W, H, nAll) {
  const tNow = now(), zoneH = H / cam.digitalZoom;
  if (!hd.length) { if (!st.noDetSince) st.noDetSince = tNow; } else st.noDetSince = 0;
  const hs = hd.map(d => d.box[3] / zoneH).sort((a, b) => a - b);
  const medH = hs.length ? hs[hs.length >> 1] : 1; st.medH = medH;
  if (hs.length && medH < 0.12) { if (!st.smallSince) st.smallSince = tNow; st.smallLast = tNow; } else if (hs.length || tNow - (st.smallLast || 0) > 1500) st.smallSince = 0; // tolerate detection flicker
  // ladder step 1: no/small detections -> tiled inference on the same frame
  if (S.tiles === 'on') st.tiles = true;
  else if (S.tiles === 'off') st.tiles = false;
  else {
    const want = (st.noDetSince && tNow - st.noDetSince > 1500) || (st.smallSince && tNow - st.smallSince > 1500) || dense.active;
    if (want) { st.tiles = true; st.tilesWhy = st.noDetSince ? 'no detections' : dense.active ? 'dense' : 'small people'; } else if (hs.length && medH > 0.2) st.tiles = false;
    if (detector && detector.lastMs > 300 && !dense.active) st.tiles = false; // too slow to tile on this device
  }
  // ladder steps 2-3: widen first; telephoto only for a single distant target; hysteresis 10 s
  if (S.lensMode === 'auto' && cam.facing === 'environment' && tNow - st.lastLensSwitch > 10000) {
    const uw = rearByKind('ultrawide'), tele = rearByKind('tele'), wide = rearByKind('wide');
    if (st.noDetSince && tNow - st.noDetSince > 6000 && uw && cam.lensKind !== 'ultrawide') { st.ladder = 'no people → ultra-wide'; switchLens(uw.label); }
    else if (hd.length >= 1 && hd.length <= 2 && medH < 0.06 && st.smallSince && tNow - st.smallSince > 5000 && tele && cam.lensKind !== 'tele') { st.ladder = 'single distant target → telephoto'; switchLens(tele.label); }
    else if (cam.lensKind === 'tele' && (hd.length > 3 || (st.noDetSince && tNow - st.noDetSince > 4000)) && (wide || uw)) { st.ladder = 'telephoto lost coverage → wide'; switchLens((wide || uw).label); }
  }
  if (S.dense === 'on') dense.active = true;
  else if (S.dense === 'off') dense.active = false;
  else {
    // switch on when detections are many/overlapping, or when a periodic density probe sees far more people than the detector
    const crowded = hd.length >= 25 || (hd.length >= 10 && st.crowdedness > 0.35);
    const probeSaysDense = dense.est != null && tNow - dense.estAt < 8000 && dense.source === 'density' && dense.est > Math.max(25, 1.8 * hd.length + 5);
    if (crowded || probeSaysDense) { if (!dense.active) ledger.events.push({ t: Date.now(), type: 'dense-on', why: crowded ? 'many/overlapping detections' : 'density probe ' + Math.round(dense.est) }); dense.active = true; dense.autoSince = tNow; }
    else if (dense.active && tNow - dense.autoSince > 6000) dense.active = false;
    const probeWanted = hd.length >= 8 || st.tiles || nAll >= 15; // many weak boxes = typical of a dense crowd
    if (!dense.active && probeWanted && !dense.failed && (tNow - dense.reqAt > 6000 || dense.busy)) { startDenseWorker(); requestDense(); }
  }
  if (dense.active) {
    startDenseWorker();
    if (!dense.failed && (tNow - dense.reqAt > 1500 || dense.busy)) requestDense();
    if (dense.failed && tNow - st.tiledEstAt > 3000) tiledEstimate();
  }
}
async function tiledEstimate() { // documented fallback when the density model is unavailable: full frame + 3x3 tiles, NMS-merged
  st.tiledEstAt = now();
  if (!(detector instanceof YoloxDetector)) return;
  const W = frameCanvas.width, H = frameCanvas.height, regs = [{ x: 0, y: 0, w: W, h: H }];
  const tw = W * 0.4, th = H * 0.4;
  for (let fy = 0; fy < 3; fy++) for (let fx = 0; fx < 3; fx++) regs.push({ x: Math.round(fx * (W - tw) / 2), y: Math.round(fy * (H - th) / 2), w: Math.round(tw), h: Math.round(th) });
  const d = (await detector.detect(regs, 416)).filter(d => d.score > 0.35);
  dense.est = d.length; dense.estAt = now(); dense.source = 'tiled detection (lower bound)'; dense.peak = Math.max(dense.peak, d.length); dense.map = null;
}
let switching = false;
async function switchLens(label) {
  if (!running || switching) return;
  switching = true; running = false; loopGen++;
  try { status('Switching lens… counting paused', 'running'); await openCamera({ label }); markLensChange('lens ' + label); }
  catch (e) { banner('lens', 'Lens switch failed: ' + (e.message || e), 'warn'); }
  switching = false; running = true; startLoop();
}

// ---------------------------------------------------------------- drawing + UI
function colorFor(p) { return !p ? '#9aa3b5' : p.state === 'possible' ? '#ffab40' : p.state === 'unverified' ? '#00b0ff' : '#00e676'; }
const heatCanvas = document.createElement('canvas');
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (dense.active && dense.map && now() - dense.estAt < 4000) {
    const { map, mapW, mapH } = dense; let mx = 1e-6; for (const v of map) if (v > mx) mx = v;
    if (heatCanvas.width !== mapW || heatCanvas.height !== mapH) { heatCanvas.width = mapW; heatCanvas.height = mapH; }
    const hc = heatCanvas.getContext('2d'), im = hc.createImageData(mapW, mapH);
    for (let i = 0; i < map.length; i++) { const a = Math.min(1, map[i] / mx); im.data[i * 4] = 255; im.data[i * 4 + 1] = 90 * (1 - a); im.data[i * 4 + 3] = 140 * a; }
    hc.putImageData(im, 0, 0); ctx.imageSmoothingEnabled = true; ctx.drawImage(heatCanvas, 0, 0, canvas.width, canvas.height);
  }
  const fs = Math.max(12, canvas.width / 45);
  ctx.font = `bold ${fs}px sans-serif`;
  for (const t of tracker.tracked) {
    if (!t.matchedThisFrame) continue;
    const [x, y, w, h] = t.box, p = t.personId ? ledger.root(t.personId) : null, col = colorFor(p);
    const label = p ? `P${p.id}${p.state === 'possible' ? '?' : ''}` : '…';
    ctx.strokeStyle = col; ctx.lineWidth = Math.max(2, canvas.width / 400); ctx.setLineDash(p ? [] : [6, 4]); ctx.strokeRect(x, y, w, h); ctx.setLineDash([]);
    const ly = Math.max(fs + 4, y);
    ctx.fillStyle = col; ctx.fillRect(x, ly - fs - 4, ctx.measureText(label).width + 8, fs + 4);
    ctx.fillStyle = '#000'; ctx.fillText(label, x + 4, ly - 4);
  }
}
function uiTick() {
  const u = ledger.unique, pd = ledger.possible;
  $('current').textContent = st.inFrame;
  $('unique').textContent = u; $('big-num').textContent = u;
  $('dupes').textContent = pd;
  $('unique-range').textContent = pd ? `range ${u - pd}–${u}` : (ledger.unverified ? `${ledger.unverified} unverified` : '\u00a0');
  const notes = [];
  if (pd) notes.push(`best estimate ${u - pd}–${u} · ${pd} possible duplicate${pd === 1 ? '' : 's'}`);
  if (st.light !== 'ok' && running) notes.push('low light: low confidence');
  if (now() < st.pauseUntil) notes.push('lens change — counting paused');
  $('big-range').textContent = notes.join(' · ');
  $('stat-crowd').style.display = dense.active ? '' : 'none';
  if (dense.active) { $('crowd').textContent = dense.est == null ? '…' : (dense.source === 'density' ? '≈' + Math.round(dense.est) : '≥' + dense.est); $('crowd-sub').textContent = dense.source === 'density' ? 'visible now, ±25%' : 'tiled detections'; }
  if (detector) chip('chip-backend', `${detector.name} · ${detector.ep}${detector.lastIn ? ' · ' + detector.lastIn : ''}${st.tiles ? ' +tiles' : ''}`, detector.ep === 'webgpu' ? 'on' : detector.ep === 'wasm' ? '' : 'warn');
  chip('chip-lens', `lens: ${cam.lensKind}${cam.digitalZoom > 1 ? ' · ' + cam.digitalZoom.toFixed(1) + '× dig' : ''}${S.lensMode === 'auto' ? ' (auto)' : ''}`, now() < st.pauseUntil ? 'warn' : '');
  chip('chip-dense', `dense: ${dense.active ? (dense.failed ? 'tiled' : 'on') : 'off'}`, dense.active ? 'warn' : '');
  chip('chip-light', st.lumaEma < 0 ? 'light: —' : `light: ${st.light === 'ok' ? 'ok' : st.light === 'dark' ? 'dark' : 'very dark'} (${st.lumaEma.toFixed(0)})`, st.light === 'ok' ? '' : 'bad');
  chip('chip-fps', `${st.fps.toFixed(1)} fps`, st.fps > 8 ? 'on' : st.fps > 3 ? '' : 'warn');
  if (running) {
    if (st.light !== 'ok') banner('light', `Low light (brightness ${st.lumaMed ?? '?'}/255) — detection is less reliable. ${cam.caps.torch ? 'The torch helps only for nearby people.' : 'This camera offers no torch control.'} Count is LOW CONFIDENCE.`, 'warn');
    else banner('light');
    if (dense.active) banner('denseinfo', 'Dense crowd: people overlap too much to count one by one. “Est. crowd” is a density estimate of who is visible now (no identities, not added to Unique). Unique is likely an undercount here.', 'info');
    else banner('denseinfo');
  }
  if ($('debugPanel').classList.contains('visible')) renderDebug();
  if ($('dupesPanel').classList.contains('visible')) renderDupes();
}
function renderDebug() {
  const T = st.timings, f = v => v == null ? '—' : v.toFixed(0);
  const lines = [
    `${VERSION}  secure=${window.isSecureContext} crossOriginIsolated=${!!window.crossOriginIsolated} wasmThreads=${typeof ort !== 'undefined' ? ort.env.wasm.numThreads : '—'} webgpu=${!!navigator.gpu}`,
    `detector: ${detector ? detector.name + ' / ' + detector.ep : '—'}  input=${st.size} (${detector && detector.lastIn || '—'})  tiles=${st.tiles}${st.tiles ? ' (' + st.tilesWhy + ')' : ''}  gain=${detector ? detector.gain.toFixed(2) : '—'}`,
    detector && detector.bench ? `bench: ${detector.bench.join(' | ')}` : '',
    detector && detector.loadErrors && detector.loadErrors.length ? `load notes: ${detector.loadErrors.join(' | ')}` : '',
    `re-id: ${reid ? reid.name + ' / ' + reid.ep + ' ' + reid.lastMs.toFixed(0) + ' ms/batch' : reidLoadP ? 'loading…' : 'off'}  queue=${st.reidQueue}  same≥${S.high}  new<${S.low}`,
    `ms: grab ${f(T.grab)}  cmc ${f(T.cmc)}  det ${f(T.det)} (last net ${detector ? f(detector.lastMs) : '—'})  track ${f(T.track)}  reid ${f(T.reid)}  →  ${st.fps.toFixed(1)} fps`,
    `cmc: dx=${motion.last.dx.toFixed(1)} dy=${motion.last.dy.toFixed(1)} px  inliers=${(motion.last.conf || 0).toFixed(2)}  blocks=${motion.last.n || 0}`,
    `light: ${st.light}  brightness median=${st.lumaMed ?? '—'} mean=${st.lumaMean != null ? st.lumaMean.toFixed(0) : '—'} p98=${st.lumaP98 ?? '—'} smoothed=${st.lumaEma.toFixed(0)} (dark<${CFG.darkLuma}, very dark<${CFG.veryDarkLuma}, ok>${CFG.lightExitLuma})  frame grab=${GRAB_MODES[st.grabMode || 0]} blackRun=${st.blackRun || 0}  confirm=${confirmFrames()} frames  torch=${cam.caps.torch ? (cam.torchOn ? 'on' : 'off') : 'n/a'}`,
    ...loadDiag.slice(-12).map(l => 'load: ' + l),
    `camera: "${cam.label}" [${cam.lensKind}] ${video.videoWidth}×${video.videoHeight}  nativeZoom=${cam.nativeZoom ? JSON.stringify(cam.caps.zoom) : 'no'}  digital=${cam.digitalZoom.toFixed(1)}×`,
    `lenses seen: ${cam.devices.map(d => d.kind).join(', ') || '—'}  ladder: ${st.ladder || '—'}`,
    `tracks: tracked=${tracker.tracked.length} lost=${tracker.lost.length} frame=${tracker.frame}  inFrame=${st.inFrame}  medH=${(st.medH || 0).toFixed(2)}  crowdedness=${st.crowdedness.toFixed(2)}`,
    `ledger: unique=${ledger.unique} possible=${ledger.possible} unverified=${ledger.unverified} merged=${[...ledger.persons.values()].filter(p => p.mergedInto).length}`,
    `dense: active=${dense.active} ready=${dense.ready} failed=${dense.failed} est=${dense.est == null ? '—' : (+dense.est).toFixed(1)} (${dense.source || '—'}) ${dense.ms ? dense.ms.toFixed(0) + ' ms' : ''} peak=${(+dense.peak).toFixed(0)} ${dense.err || ''}`,
    `errors: ${(window.__scriptErrors || []).slice(-3).join(' | ') || 'none'}`,
    '', 'recent ledger events:',
    ...ledger.events.slice(-8).map(e => `  ${new Date(e.t).toLocaleTimeString()} ${e.type} ${e.id || e.from || ''}${e.into ? '→' + e.into : ''}${e.candidate ? ' ~P' + e.candidate : ''} ${e.sim != null ? 'sim=' + (+e.sim).toFixed(2) : ''}${e.candSim ? ' best=' + (+e.candSim).toFixed(2) : ''} ${e.why || ''}`),
    '', 'persons (id:state samples): ' + ledger.live().slice(-14).map(p => `P${p.id}:${p.state[0]}${p.gallery.length}`).join(' ')
  ];
  $('debugText').textContent = lines.filter(l => l !== '').join('\n');
}
function renderDupes() {
  const list = ledger.live().filter(p => p.state === 'possible');
  const sig = list.map(p => p.id + ':' + p.candidate).join(',');
  if ($('dupesList').dataset.sig === sig) return; $('dupesList').dataset.sig = sig;
  $('dupesList').innerHTML = list.length ? '' : '<p style="color:var(--muted)">None right now.</p>';
  for (const p of list) {
    const c = ledger.root(p.candidate);
    const row = document.createElement('div'); row.className = 'dupe';
    row.innerHTML = `<div class="txt">P${p.id} looks like P${c ? c.id : '?'} <span style="color:var(--muted)">(compare the P-labels on screen; no snapshots are stored)</span><br><span style="color:var(--muted)">${p.why ? p.why : 'similarity ' + (p.candSim || 0).toFixed(2)}</span></div>`;
    const mb = document.createElement('button'); mb.textContent = 'Same → merge';
    mb.onclick = () => { ledger.merge(p.id, p.candidate, 'manual', true); for (const t of tracker.tracked) if (t.personId === p.id) t.personId = p.candidate; $('dupesList').dataset.sig = ''; uiTick(); };
    const kb = document.createElement('button'); kb.textContent = 'Different';
    kb.onclick = () => { p.state = 'new'; p.resolved = true; ledger.events.push({ t: Date.now(), type: 'resolve-new', id: p.id, why: 'manual' }); $('dupesList').dataset.sig = ''; uiTick(); };
    row.appendChild(mb); row.appendChild(kb); $('dupesList').appendChild(row);
  }
}
window.__tc.mergePossible = (id) => { const p = ledger.persons.get(id); if (p && p.state === 'possible') { ledger.merge(p.id, p.candidate, 'manual', true); return true; } return false; };

// ---------------------------------------------------------------- CSV log (1 row/second)
function csvTick() {
  if (!running || Date.now() - st.lastCsv < 1000) return;
  st.lastCsv = Date.now();
  st.csv.push([new Date().toISOString(), st.inFrame, ledger.unique, ledger.possible, dense.active && dense.est != null ? (+dense.est).toFixed(1) : '',
    cam.lensKind + (cam.digitalZoom > 1 ? `@${cam.digitalZoom.toFixed(1)}x` : ''), ledger.unique - ledger.possible, ledger.unverified, st.light, detector ? detector.ep : '', st.fps.toFixed(1)]);
}
function csvText() {
  const head = 'timestamp,in_frame,unique,possible_dupes,estimated_crowd,lens,unique_low,unverified,light,backend,fps';
  return head + '\n' + st.csv.map(r => r.join(',')).join('\n') + '\n';
}
function exportCsv() {
  const txt = csvText(); window.__lastCsv = txt;
  const blob = new Blob([txt], { type: 'text/csv' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = `tcount-crowd-${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.csv`;
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 3000);
}
window.__tc.csvText = csvText;

// ---------------------------------------------------------------- loop
let rafId = 0;
function render() { draw(); rafId = requestAnimationFrame(render); }
setInterval(() => { uiTick(); csvTick(); }, 250);
async function loop(gen) {
  while (running && gen === loopGen) {
    if (video.readyState < 2 || !video.videoWidth) { await new Promise(r => setTimeout(r, 100)); continue; }
    try { await step(); }
    catch (e) { logErr('step', e); status('Detection error: ' + (e.message || e), 'error'); await new Promise(r => setTimeout(r, 500)); }
    await new Promise(r => setTimeout(r, 0));
  }
}
function startLoop() { loopGen++; const g = loopGen; loop(g); if (!rafId) render(); status('Counting people…', 'running'); }

// ---------------------------------------------------------------- controls
$('startBtn').onclick = async () => {
  $('startBtn').disabled = true; status('Starting…', 'running');
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { status('Camera API unavailable (needs 127.0.0.1 or https)', 'error'); $('startBtn').disabled = false; return; }
  // Overlap camera permission/open with remaining model work (warm prefetch may already have ORT+detector ready).
  const modelsP = loadModels().then(() => null, e => e);
  try { await openCamera({ label: S.lensMode === 'manual' ? S.lens : '' }); }
  catch (e) {
    status('Camera error: ' + (e.message || e.name), 'error');
    banner('cam', 'Camera error: ' + (e.message || e.name) + '. Allow camera access for this site in Safari.');
    $('startBtn').disabled = false; return;
  }
  await enumerateCams(); // labels only appear after permission
  const modelErr = await modelsP;
  if (modelErr) { status('Model load failed: ' + (modelErr.message || modelErr), 'error'); $('startBtn').disabled = false; return; }
  tracker.reset(); motion.reset(); st.csv = []; st.lastLensSwitch = now(); st.lumaEma = -1; st.runStart = now(); st.camStart = now(); st.light = 'ok'; st.lightPendSince = 0; st.blackRun = 0;
  running = true;
  $('stopBtn').disabled = false; $('switchBtn').disabled = false; $('startBtn').textContent = 'Running…';
  startLoop();
};
$('stopBtn').onclick = () => {
  running = false; loopGen++;
  try { ledger.reconcile(null); } catch (e) { logErr('reconcile', e); }
  if (cam.stream) cam.stream.getTracks().forEach(t => t.stop()); cam.stream = null; video.srcObject = null;
  const u = ledger.unique, pd = ledger.possible, uv = ledger.unverified;
  $('final-total').textContent = pd ? `${u - pd}–${u}` : u;
  const parts = [`<b>${u}</b> counted · ${pd} possible duplicate${pd === 1 ? '' : 's'}${pd ? ` → best estimate ${u - pd}–${u}` : ''}`];
  if (uv) parts.push(`${uv} counted without an appearance check (too small/occluded) — they could not be de-duplicated if they returned`);
  if (dense.peak) parts.push(`Peak estimated visible crowd: ≈${Math.round(dense.peak)} (${dense.source || 'density'}; separate from Unique)`);
  if (st.csv.some(r => r[8] !== 'ok')) parts.push('Parts of the session were in low light (low confidence)');
  if (ledger.ghosts) parts.push(`${ledger.ghosts} split-second detection${ledger.ghosts === 1 ? '' : 's'} (under ${CFG.ghostMs / 1000} s, then gone) treated as noise and not counted`);
  $('final-detail').innerHTML = parts.join('<br>');
  $('result').classList.add('visible'); status('Stopped', '');
  $('startBtn').disabled = $('stopBtn').disabled = $('switchBtn').disabled = true;
};
$('switchBtn').onclick = async () => {
  if (!running) return; $('switchBtn').disabled = true;
  cam.facing = cam.facing === 'environment' ? 'user' : 'environment';
  running = false; loopGen++; status('Switching camera…', 'running');
  try { await openCamera({}); markLensChange('flip camera'); } catch (e) { banner('cam', 'Camera switch failed: ' + (e.message || e)); }
  running = true; startLoop(); $('switchBtn').disabled = false;
};
$('resetBtn').onclick = () => {
  $('result').classList.remove('visible'); tracker.reset(); ledger.reset(); motion.reset(); TRACK_ID = 1;
  dense.est = null; dense.peak = 0; dense.map = null; dense.active = false; st.csv = []; st.inFrame = 0; uiTick();
  $('startBtn').disabled = false; $('startBtn').textContent = 'Start Camera';
  $('stopBtn').disabled = $('switchBtn').disabled = true;
  status('Ready — tap Start', 'ready'); ctx.clearRect(0, 0, canvas.width, canvas.height);
};
$('zoom').oninput = () => updateZoomLabel();
$('zoom').onchange = e => setZoom(e.target.value);
$('torchBtn').onclick = toggleTorch;
$('csvBtn').onclick = exportCsv; $('resultCsvBtn').onclick = exportCsv;
const togglePanel = id => { for (const p of ['settingsPanel', 'debugPanel', 'dupesPanel', 'selfTestPanel']) if (p !== id) $(p).classList.remove('visible'); $(id).classList.toggle('visible'); uiTick(); };
$('settingsBtn').onclick = () => togglePanel('settingsPanel');
$('debugBtn').onclick = () => togglePanel('debugPanel');
$('selfTestBtn').onclick = () => { togglePanel('selfTestPanel'); if ($('selfTestPanel').classList.contains('visible')) selfTest(); };
$('selfTestRun').onclick = () => selfTest();
let selfTesting = false;
async function selfTest() {
  if (selfTesting) return; selfTesting = true;
  const out = [], show = () => { $('selfTestText').textContent = out.join('\n'); };
  const line = l => { out.push(l); show(); };
  try {
    line(`${VERSION}  ${new Date().toLocaleString()}`);
    line(`page: ${location.href}`);
    line(`browser: ${navigator.userAgent}`);
    line(`secure=${window.isSecureContext}  crossOriginIsolated=${!!window.crossOriginIsolated}  cores=${navigator.hardwareConcurrency || '?'}  memory=${navigator.deviceMemory || '?'} GB`);
    line(`ONNX runtime script: ${typeof ort === 'undefined' ? 'NOT LOADED' : 'loaded v' + (ort.env.versions && ort.env.versions.web || '?')}`);
    const simd = (() => { try { return WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11])); } catch (e) { return false; } })();
    line(`WebAssembly: ${typeof WebAssembly === 'object' ? 'yes' : 'NO'}  SIMD=${simd ? 'yes' : 'NO (needs iOS 16.4+)'}`);
    for (const mb of [1024, 4096]) { try { new WebAssembly.Memory({ initial: 256, maximum: mb * 16, shared: true }); line(`shared wasm memory ${mb} MB reservation: ok`); } catch (e) { line(`shared wasm memory ${mb} MB reservation: FAILED (${errText(e)})`); } }
    let gpu = 'not exposed (navigator.gpu missing)';
    if (navigator.gpu) { try { const a = await navigator.gpu.requestAdapter(); gpu = a ? 'adapter ok' + (a.info && a.info.vendor ? ' (' + a.info.vendor + ')' : '') : 'no adapter'; } catch (e) { gpu = 'error ' + errText(e); } }
    line(`WebGPU: ${gpu}`);
    line(`camera: ${cam.stream ? `${video.videoWidth}×${video.videoHeight} "${cam.label}" brightness median=${st.lumaMed ?? '—'} p98=${st.lumaP98 ?? '—'} grab=${GRAB_MODES[st.grabMode || 0]}${st.blackRun >= CFG.blackFrames ? ' — FRAMES ARE BLACK' : ''}` : 'not started (tap Start Camera, then run again)'}`);
    line(`detector: ${detector ? detector.name + ' / ' + detector.ep : 'none'}   re-ID: ${reid ? reid.name + ' / ' + reid.ep : 'none'}   dense: ${dense.ready ? 'ready/' + dense.ep : dense.failed ? 'FAILED ' + dense.err : 'not started'}`);
    line('');
    line('files (HTTP status · size · expected · content-type · time):');
    let bad = 0;
    for (const [rel, exp, req] of ASSETS) {
      const t0 = now();
      try {
        const r = await fetch(abs(rel), { cache: 'no-store' }); const ct = r.headers.get('content-type') || '?';
        let n = 0; if (r.body && r.body.getReader) { const rd = r.body.getReader(); for (;;) { const { done, value } = await rd.read(); if (done) break; n += value.length; } } else n = (await r.arrayBuffer()).byteLength;
        const okSize = !exp || n === exp, ok = r.ok && okSize;
        if (!ok && req) bad++;
        line(`${ok ? '✓' : req ? '✗' : '!'} ${rel}: HTTP ${r.status} · ${MB(n)}${exp ? (okSize ? '' : ` · EXPECTED ${MB(exp)} (incomplete)`) : ''} · ${ct} · ${((now() - t0) / 1000).toFixed(1)} s${!r.ok && !req ? ' (optional: dense model from part 2)' : ''}`);
      } catch (e) { if (req) bad++; line(`✗ ${rel}: request failed — ${errText(e)}`); }
    }
    line('');
    if (typeof ort !== 'undefined' && !bad) {
      line('runtime test: creating a YOLOX-Nano session on the CPU…');
      try {
        await setupOrt();
        const b = await fetchAsset('models/yolox_nano_person_w8.onnx');
        const t0 = now(); const ss = await withTimeout(ort.InferenceSession.create(b, { executionProviders: ['wasm'] }), 120000, 'session');
        const x = new ort.Tensor('float32', new Float32Array(3 * 320 * 320).fill(114), [1, 3, 320, 320]); await ss.run({ images: x });
        line(`runtime test: OK (${(now() - t0).toFixed(0)} ms)`); try { ss.release && ss.release(); } catch (e) {}
      } catch (e) { line('runtime test: FAILED — ' + errText(e)); }
    } else if (bad) line(`${bad} required file(s) missing or incomplete — re-copy the crowd folder (unzip part 1 again) and restart python3 -m http.server.`);
    if (loadDiag.length) { line(''); line('load log:'); for (const l of loadDiag.slice(-15)) line('  ' + l); }
    if (window.__scriptErrors && window.__scriptErrors.length) { line(''); line('errors:'); for (const l of window.__scriptErrors.slice(-10)) line('  ' + l); }
  } catch (e) { line('self-test crashed: ' + errText(e)); }
  selfTesting = false;
}
$('stat-dupes').onclick = () => togglePanel('dupesPanel');
document.querySelectorAll('[data-close]').forEach(b => b.onclick = () => $(b.dataset.close).classList.remove('visible'));
const bind = (id, key, num) => { const el = $(id); el.value = S[key]; el.onchange = () => { S[key] = num ? +el.value : el.value; saveSettings(); onSetting(key); }; };
bind('setModel', 'model'); bind('setBackend', 'backend'); bind('setSize', 'size'); bind('setTiles', 'tiles'); bind('setDense', 'dense');
bind('setLensMode', 'lensMode'); bind('setCmc', 'cmc'); bind('setHigh', 'high', true); bind('setLow', 'low', true); bind('setConfirm', 'confirm', true); bind('setThreads', 'threads');
$('setLens').onchange = () => { S.lens = $('setLens').value; S.lensMode = 'manual'; $('setLensMode').value = 'manual'; saveSettings(); if (running) switchLens(S.lens); };
function onSetting(key) {
  if (key === 'size' && S.size !== 'auto') st.size = +S.size;
  if (key === 'threads') setupThreads(true);
}
// optional: cross-origin isolation through a tiny service worker -> multi-threaded WASM (python http.server cannot send COOP/COEP)
async function setupThreads(userChanged) {
  if (!('serviceWorker' in navigator)) { if (S.threads === 'on') banner('thr', 'Service workers unavailable — multi-thread CPU mode not possible here.', 'warn'); return; }
  try {
    if (S.threads === 'on') {
      await navigator.serviceWorker.register('sw.js');
      if (!window.crossOriginIsolated) {
        await navigator.serviceWorker.ready;
        if (userChanged || !sessionStorage.getItem('coi-reloaded')) { sessionStorage.setItem('coi-reloaded', '1'); location.reload(); }
        else banner('thr', 'Multi-thread mode is on, but this browser did not enable cross-origin isolation — staying single-threaded.', 'warn');
      }
    } else {
      const regs = await navigator.serviceWorker.getRegistrations();
      for (const r of regs) await r.unregister();
      if (window.crossOriginIsolated && userChanged) location.reload();
    }
  } catch (e) { banner('thr', 'Multi-thread setup failed: ' + (e.message || e), 'warn'); }
}
setupThreads(false);
window.addEventListener('error', e => logErr('Error', e.error || e.message));
window.addEventListener('unhandledrejection', e => logErr('Promise', e.reason));
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) status('Camera not available here (needs 127.0.0.1 or https)', 'error');
uiTick();
warmPrefetch();
})();
