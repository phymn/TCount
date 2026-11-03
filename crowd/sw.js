/* TCount Crowd service worker.
   Versioned Cache Storage (tcount-<app version>) so a new app or model replaces stale copies
   instead of mixing them with the previous build. After the first visit the shell, the ONNX
   Runtime wasm and any model that was actually used are served from this cache, including offline.
   The dense model is not precached; it enters the cache only when the page requests it.
   COOP/COEP/CORP are added only when Settings → Multi-thread CPU is on (python http.server
   cannot send those headers). The flag lives in this same cache so it survives a worker restart. */
const VERSION = 'crowd-1.2';
const CACHE = 'tcount-' + VERSION;
const THREADS_URL = new URL('__tc_threads', self.registration.scope).href;
const SHELL = [
  './',
  './index.html',
  './app.js',
  './dense-worker.js',
  './vendor/ort.webgpu.min.js',
  './vendor/ort-wasm-simd-threaded.asyncify.js',
];
const EXPECT = {
  'ort-wasm-simd-threaded.asyncify.wasm': 26781914,
  'ort.webgpu.min.js': 66417,
  'dmcount_qnrf_w8.onnx': 21560628,
  'yolox_nano_person_w8.onnx': 1117717,
  'yolox_tiny_person_w8.onnx': 5208654,
  'osnet_x0_5_reid_w8.onnx': 875829,
};

let threadsFlag = null;

self.addEventListener('install', event => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    await Promise.all(SHELL.map(async url => {
      try {
        const res = await fetch(new Request(url, { cache: 'reload' }));
        if (res && res.ok) await cache.put(url, res);
      } catch (e) { /* keep installing; a missing optional file must not block the shell */ }
    }));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter(k => k.startsWith('tcount-') && k !== CACHE).map(k => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('message', event => {
  const data = event.data || {};
  if (data.type === 'threads') event.waitUntil(setThreads(!!data.on));
});

async function setThreads(on) {
  threadsFlag = on;
  const cache = await caches.open(CACHE);
  await cache.put(THREADS_URL, new Response(on ? 'on' : 'off', { headers: { 'Content-Type': 'text/plain', 'Cache-Control': 'no-store' } }));
}

async function threadsOn() {
  if (threadsFlag !== null) return threadsFlag;
  try {
    const cache = await caches.open(CACHE);
    const res = await cache.match(THREADS_URL);
    threadsFlag = !!res && (await res.text()) === 'on';
  } catch (e) { threadsFlag = false; }
  return threadsFlag;
}

function fileName(url) { return String(url).split('?')[0].split('/').pop(); }
function expectedFor(url) { return EXPECT[fileName(url)] || 0; }

function cacheRequest(url) {
  const clean = new URL(url.href);
  clean.search = '';
  return new Request(clean.href);
}

function cacheable(url) {
  if (fileName(url) === '__tc_threads') return false;
  return /\.(wasm|onnx|js|html|txt)$/.test(url.pathname) || url.pathname.endsWith('/');
}

async function putChecked(cache, key, res) {
  if (!res || !res.ok || res.type === 'opaque') return;
  const exp = expectedFor(key.url);
  if (exp) {
    const len = Number(res.headers.get('content-length') || 0);
    if (len && len !== exp) return;
    if (!len) {
      const buf = await res.clone().arrayBuffer();
      if (buf.byteLength !== exp) return;
    }
  }
  const headers = new Headers(res.headers);
  if (/\.(wasm|onnx)$/.test(new URL(key.url).pathname)) headers.set('Cache-Control', 'public, max-age=31536000');
  headers.set('X-TCount-Version', VERSION);
  try { await cache.put(key, new Response(res.clone().body, { status: res.status, statusText: res.statusText, headers })); }
  catch (e) { /* quota / private mode: the page still has the bytes for this launch */ }
}

function withPolicy(res, coi) {
  if (!coi || !res) return res;
  if (res.status === 0) return res;
  const headers = new Headers(res.headers);
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
}

self.addEventListener('fetch', event => {
  const req = event.request;
  if (req.method !== 'GET') return;
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  event.respondWith(handle(req, url));
});

async function handle(req, url) {
  const cache = await caches.open(CACHE);
  const coi = await threadsOn();
  const key = cacheRequest(url);
  const hit = await cache.match(key);
  // Same cache name for the whole version: a hit is the bytes that belong to this build.
  if (hit) return withPolicy(hit, coi);
  try {
    const res = await fetch(req);
    if (cacheable(url)) await putChecked(cache, key, res);
    return withPolicy(res, coi);
  } catch (e) {
    if (hit) return withPolicy(hit, coi);
    if (req.mode === 'navigate') {
      const fallback = await cache.match('./index.html') || await cache.match('./');
      if (fallback) return withPolicy(fallback, coi);
    }
    throw e;
  }
}
