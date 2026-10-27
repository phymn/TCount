/* Dense-crowd density estimation (DM-Count, MIT) in a worker so the camera loop never freezes.
   Output: density map whose sum = estimated number of visible people. No identities, no images kept.
   The page passes the model bytes. This worker does not fetch dmcount_qnrf_w8.onnx itself, so the
   21 MB file is downloaded only when the user turns dense mode on (and then only from the device cache). */
let session = null, ep = 'wasm';
self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      self.__ortMaxPages = m.maxPages || 16384; // keep the shared-memory reservation small (iOS)
      importScripts(m.base + 'vendor/ort.webgpu.min.js');
      ort.env.wasm.wasmPaths = { mjs: m.base + 'vendor/ort-wasm-simd-threaded.asyncify.js', wasm: m.base + 'vendor/ort-wasm-simd-threaded.asyncify.wasm' };
      ort.env.wasm.numThreads = self.crossOriginIsolated ? 2 : 1;
      if (m.wasmBinary) ort.env.wasm.wasmBinary = m.wasmBinary; // reuse the runtime the page already downloaded
      if (!m.modelBytes) throw new Error('models/dmcount_qnrf_w8.onnx was not provided — turn Dense crowd mode on');
      const bytes = new Uint8Array(m.modelBytes);
      if (bytes.length !== 21560628) throw new Error(`models/dmcount_qnrf_w8.onnx is incomplete (${bytes.length} of 21560628 bytes) — re-copy part 2`);
      let lastErr = null;
      for (const cand of (m.eps || ['webgpu', 'wasm'])) {
        if (cand === 'webgpu' && !(self.navigator && navigator.gpu)) continue;
        try { session = await ort.InferenceSession.create(bytes, { executionProviders: [cand] }); ep = cand;
          await session.run({ input: new ort.Tensor('float32', new Float32Array(3 * 64 * 64), [1, 3, 64, 64]) }); break; }
        catch (err) { lastErr = err; session = null; }
      }
      if (!session) throw lastErr || new Error('no backend');
      ort.env.wasm.wasmBinary = undefined;
      self.postMessage({ type: 'ready', ep });
    } else if (m.type === 'run') {
      const W = m.w, H = m.h, px = new Uint8ClampedArray(m.data), n = W * H, t = new Float32Array(3 * n);
      const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
      for (let i = 0, j = 0; i < n; i++, j += 4) {
        t[i] = (px[j] / 255 - mean[0]) / std[0]; t[n + i] = (px[j + 1] / 255 - mean[1]) / std[1]; t[2 * n + i] = (px[j + 2] / 255 - mean[2]) / std[2];
      }
      const t0 = performance.now();
      const r = await session.run({ input: new ort.Tensor('float32', t, [1, 3, H, W]) });
      const d = r.density, map = new Float32Array(d.data);
      let count = 0; for (const v of map) count += v;
      self.postMessage({ type: 'result', count, map, w: d.dims[3], h: d.dims[2], ms: performance.now() - t0, ep }, [map.buffer]);
    }
  } catch (err) { self.postMessage({ type: 'error', error: String(err && err.message || err) }); }
};
