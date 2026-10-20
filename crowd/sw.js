/* Optional helper (Settings → Multi-thread CPU). python http.server cannot send COOP/COEP headers, which
   multi-threaded WebAssembly needs; this worker re-serves same-origin responses with those headers.
   It does not cache anything and never contacts the network beyond the original request. */
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  if (new URL(req.url).origin !== self.location.origin) return; // cross-origin (e.g. coco-ssd CDN) untouched
  e.respondWith(fetch(req).then(res => {
    if (res.status === 0) return res;
    const h = new Headers(res.headers);
    h.set('Cross-Origin-Embedder-Policy', 'require-corp');
    h.set('Cross-Origin-Opener-Policy', 'same-origin');
    h.set('Cross-Origin-Resource-Policy', 'same-origin');
    // Immutable large binaries: allow long-lived browser cache when this SW is active.
    const path = new URL(req.url).pathname;
    if (/\.(wasm|onnx)$/i.test(path) && res.ok) h.set('Cache-Control', 'public, max-age=31536000, immutable');
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
  }));
});
