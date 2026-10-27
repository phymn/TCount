/* TCount Crowd service worker.
   Two jobs:
   1. Versioned Cache API so the app shell, ONNX models and the ORT wasm stay on the device.
      Cache name is tcount-crowd-1.2 — keep it in sync with VERSION in app.js.
      activate() deletes every other tcount-crowd-* cache, so a new app or model version
      replaces stale bytes instead of mixing them with the new release.
   2. Optional COOP/COEP/CORP (only when the page set the tc-threads=on cookie) so
      multi-thread WASM works under python http.server. Headers are added on the way out
      and are not stored in the cache, so turning threads off does not require a re-download.
   Heavy files are stored only when their byte length matches the known release size. */
const VERSION = 'crowd-1.2';
const CACHE = 'tcount-' + VERSION;
const EXPECT = {
  'ort-wasm-simd-threaded.asyncify.wasm': 26781914,
  'yolox_nano_person_w8.onnx': 1117717,
  'yolox_tiny_person_w8.onnx': 5208654,
  'osnet_x0_5_reid_w8.onnx': 875829,
  'dmcount_qnrf_w8.onnx': 21560628,
};
const SHELL = [
  './',
  'index.html',
  'app.js',
  'dense-worker.js',
  'vendor/ort.webgpu.min.js',
  'vendor/ort-wasm-simd-threaded.asyncify.js',
  'README.txt',
];

function threadsOn(req) {
  return /(?:^|;\s*)tc-threads=on(?:;|$)/.test(req.headers.get('cookie') || '');
}
function withIsolation(res, req) {
  if (!threadsOn(req) || !res || res.status === 0 || res.type === 'opaque') return res;
  const h = new Headers(res.headers);
  h.set('Cross-Origin-Embedder-Policy', 'require-corp');
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  h.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
}
function fileName(url) { return url.pathname.split('/').pop(); }
function expected(url) { return EXPECT[fileName(url)] || 0; }
function isShellPath(url) {
  if (url.origin !== self.location.origin || expected(url)) return false;
  return url.pathname.endsWith('/') || /\.(html|js|txt)$/.test(url.pathname);
}

self.addEventListener('install', e => {
  e.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    await Promise.all(SHELL.map(async rel => {
      try { await cache.add(new Request(rel, { cache: 'reload' })); } catch (err) { /* missing optional file must not block install */ }
    }));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', e => {
  e.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter(k => k.startsWith('tcount-crowd-') && k !== CACHE).map(k => caches.delete(k)));
    await self.clients.claim();
  })());
});

async function matchCached(cache, req) {
  const hit = await cache.match(req, { ignoreSearch: true });
  if (hit) return hit;
  if (req.mode === 'navigate') return cache.match(new URL('./index.html', self.registration.scope).href);
  return null;
}
function sizeOk(res, exp) {
  if (!exp) return true;
  const n = res.headers.get('X-TCount-Bytes') || res.headers.get('Content-Length');
  return !!n && Number(n) === exp;
}

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET' && req.method !== 'HEAD') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  const exp = expected(url);
  const heavy = exp > 0;
  const shell = isShellPath(url) || req.mode === 'navigate';
  if (!heavy && !shell) return;

  // Self-test asks for a live server check. Fall back to the device copy when offline.
  if (req.method === 'HEAD' || req.cache === 'no-store') {
    e.respondWith((async () => {
      try { return withIsolation(await fetch(req), req); }
      catch (err) {
        const hit = await matchCached(await caches.open(CACHE), req);
        if (hit && sizeOk(hit, exp)) return withIsolation(hit, req);
        throw err;
      }
    })());
    return;
  }

  e.respondWith((async () => {
    const cache = await caches.open(CACHE);
    const hit = await matchCached(cache, req);
    if (hit && sizeOk(hit, exp)) return withIsolation(hit, req);
    if (hit && exp) { try { await cache.delete(req, { ignoreSearch: true }); } catch (err) {} }
    let res;
    try { res = await fetch(req); }
    catch (err) {
      if (hit) return withIsolation(hit, req);
      throw err;
    }
    if (res && res.ok) {
      try {
        if (!exp) await cache.put(req, res.clone());
        else {
          const buf = await res.clone().arrayBuffer();
          if (buf.byteLength === exp) {
            const headers = new Headers(res.headers);
            headers.set('Content-Length', String(buf.byteLength));
            headers.set('X-TCount-Bytes', String(buf.byteLength));
            headers.set('X-TCount-Version', VERSION);
            await cache.put(req, new Response(buf, { status: 200, statusText: res.statusText, headers }));
          }
        }
      } catch (err) { /* quota / private mode: the page still runs, it just cannot stay offline */ }
    }
    return withIsolation(res, req);
  })());
});
