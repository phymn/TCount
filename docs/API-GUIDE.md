# TCount — API guide

Every significant browser API, on-device model runtime and outside library the apps use, where they are called, why they were picked, and what leaves the device.

This guide follows **`main`**: the crowd counter `crowd/` (version crowd-1.2) and the original counters `index-nogender.html` / `index.html`. Features that exist only on open PR branches are not listed as current. Export scripts under `tools/export/` are in a short section at the end.

How to study: one row a day, starting with how crowd opens the camera, then the ONNX runtime, then the cache.

Related: [HOW-IT-WORKS.md](HOW-IT-WORKS.md) · [DECISIONS.md](DECISIONS.md) · [CHANGELOG.md](../CHANGELOG.md) · [faceid-lidar.md](faceid-lidar.md).

Status words: **Verified in code** = read from the files named below. Battery notes are qualitative.

## Crowd counter (`crowd/`) — camera, GPU and models

| API | Where it is called (file → function) | What it does in this app | Why this one was chosen | Permission / network / notes |
|---|---|---|---|---|
| **`navigator.mediaDevices.getUserMedia`** | `crowd/app.js` → `openCamera()`, start path | Opens the live camera locally. Frames never leave the device. | Built-in, no SDK. Rejected: a native app or a cloud video API. | Camera permission. Needs a secure context (`https` or `http://127.0.0.1`). |
| **`mediaDevices.enumerateDevices`** | `crowd/app.js` → `enumerateCams()` | Lists video inputs so the user can pick a lens / camera. | Built-in way to discover cameras after permission is granted. | Labels often appear only after the first camera grant. |
| **`MediaStreamTrack.getCapabilities` / `getSettings`** | `crowd/app.js` → `openCamera()` | Reads what the track actually supports (zoom range, torch, and so on). | Avoids guessing; the UI only offers controls the hardware has. | Local. |
| **`MediaStreamTrack.applyConstraints`** | `crowd/app.js` → `setZoom()`, `toggleTorch()` | Changes zoom or turns the torch on or off on the live track, without reopening the camera. | Less flicker and less code than stopping and restarting `getUserMedia`. If native zoom is missing, digital zoom is applied with CSS instead. | Local. Torch/zoom support varies by phone and browser. |
| **onnxruntime-web (`ort.InferenceSession`)** | `crowd/app.js` → `setupOrt()`, `makeSession()`, `loadModels()`, `selfTest()`; `crowd/dense-worker.js` | Runs YOLOX (person boxes), OSNet (re-ID embeddings) and optional DM-Count (density map) **in the browser**. | Same ONNX format as the export tools; WebGPU or WASM, no Python in the page, nothing sent to a server. Rejected: tf.js in the crowd app (used only by the original counters); a remote inference API. | Models are local files under `crowd/models/` (the optional dense model is fetched only when dense mode is turned on). |
| **`navigator.gpu` (WebGPU)** | `crowd/app.js` → `webgpuAvailable()`, `gpuOk()`, debug/self-test; `crowd/dense-worker.js` | Checks that a GPU adapter exists, then ORT prefers the `webgpu` execution provider and falls back to WASM. | Fastest on-device path when the browser exposes WebGPU. | Local. If `navigator.gpu` is missing, WASM still runs the models. |
| **Canvas 2D** `getContext` + `drawImage` + `getImageData` | `crowd/app.js` (detector / re-ID prep, `grabFrame()`, `requestDense()`, `draw()`) | Scales frames, reads pixels into the tensors the models expect, and draws boxes on the overlay. | Built-in, GPU-assisted. | Local. `thumbOf()` is deliberately a no-op: no snapshots are stored, only numeric embeddings in memory. |
| **`createImageBitmap`** | `crowd/app.js` → `grabFrame()` | When grab mode 2 is on, decodes/scales a frame without blocking the UI as long. | Asynchronous; falls back to a plain `drawImage` from the video element. | Local. |
| **`Worker`** (`dense-worker.js`) + **`postMessage`** | `crowd/app.js` → `startDenseWorker()`, `requestDense()` | Runs DM-Count on a second ORT runtime so the camera loop does not freeze. The page passes the model bytes in; the worker does not download them unless that hand-off is missing. | Keeps the UI and camera loop responsive. Rejected: running dense mode on the main thread. | Local. Used only when dense mode is on. |
| **`requestAnimationFrame`** | `crowd/app.js` → `render()` | Redraws the overlay in sync with the display. | Pauses in background tabs; no `setInterval` polling for drawing. | Local. |
| **`performance.now`** | `crowd/app.js` (timing helpers and load metrics) | Measures frame and load times. | Monotonic high-resolution clock. | Local. |
| **`Blob` + `URL.createObjectURL` / `revokeObjectURL`** | `crowd/app.js` → `exportCsv()` | Builds a CSV of the count ledger for download, then releases the temporary URL. | Local download, no server. | Local. |
| **`localStorage`** | `crowd/app.js` (settings) | Remembers settings (dense mode, thread flag, and similar) across reloads. | Small, built-in. Rejected for counts: counts stay in memory / CSV, not in `localStorage`. | Local. |

## Crowd counter — offline cache

| API | Where it is called | What it does in this app | Why this one was chosen | Notes |
|---|---|---|---|---|
| **`navigator.serviceWorker`** | `crowd/app.js` → `setupThreads()`, `rememberThreadFlag()` | Registers `crowd/sw.js` so later visits can be served from cache, and so multi-thread CPU mode can set COOP/COEP headers the plain Python test server cannot. | Standard way to intercept requests and work offline. | If service workers are missing, models can still cache in the tab when the Cache API exists. |
| **Cache Storage API** (`caches.open` / `match` / `put`) | `crowd/app.js` → `openCache()`, `cacheGet()`, `cachePut()`, `purgeOldCaches()`; `crowd/sw.js` install / fetch | Keeps a versioned cache (`tcount-crowd-1.2`) of the app shell and of every wasm/onnx file the page actually fetched, so the next launch and an offline launch do not download the ~26 MB runtime again. A new cache name drops older `tcount-*` caches. | Built-in, works without a custom installer. Rejected: asking the user to download models by hand each time. | Local disk, origin-scoped. |
| **`fetch`** | `crowd/app.js` → `downloadUrl()`, `fetchAsset()`; `crowd/sw.js`; `crowd/dense-worker.js` (fallback) | Downloads runtime/model files the first time (or when the cache cannot serve them). | Built-in HTTP. `Abort`/timeouts wrap long downloads. | Network only for missing assets. Camera frames are never posted. |

## Original counters (`index-nogender.html`, `index.html`)

| API | Where | What it does | Why this one | Notes |
|---|---|---|---|---|
| **`getUserMedia`** | both pages, camera start | Opens the camera locally, same idea as crowd. | Built-in. | Camera permission, secure context. |
| **TensorFlow.js + coco-ssd 2.2.3** (`lite_mobilenet_v2`) | both pages, model load / `personModel.detect(video)` | Detects people in the frame. | Ready-made in-browser detector; no custom export step. Crowd later switched to YOLOX + ORT for an offline, person-only model. | **Network, first visit:** scripts from jsDelivr / npm CDN. |
| **face-api.js** TinyFaceDetector + AgeGender net | `index.html` only → `estimateGender()` | Optional gender votes (deprecated in crowd: unreliable and biased). | Already used in earlier TCount experiments. | CDN models. Prefer `index-nogender.html` or `crowd/` if you do not want this. |
| **Canvas 2D** | both pages | Draws boxes and copies crops for the tracker / face net. | Built-in. | Local. |

## Export tools (`tools/export/`, run on a computer, not in the browser)

| API | Where | What it does | Why |
|---|---|---|---|
| **PyTorch** | `export_dmc.py`, `export_osnet.py`, and related scripts | Loads the source models so they can be exported. | That is the training/export ecosystem those models already use. |
| **ONNX / onnxruntime** (Python) | same folder, plus `modeltest.py` | Exports, quantizes (int8 weights) and test-runs the same format the browser ORT loads. | One format from export to the phone's browser. |

Playwright, `http.server` and similar helpers in `tools/test/` drive local tests; they are not part of the counting UI.

## Glossary

- **API**: a set of functions a framework or service offers to your code.
- **ONNX / ORT**: a portable model file format and the runtime (ONNX Runtime) that runs it in the browser or in Python.
- **WebGPU**: a browser API that talks to the GPU; ORT uses it when `navigator.gpu` is present.
- **WASM**: WebAssembly — compiled code that runs near native speed in the browser; ORT's fallback when WebGPU is missing.
- **YOLOX**: the person detector used in crowd (nano/tiny, person class only, int8 weights).
- **ByteTrack**: the tracker that follows people from one frame to the next (with a Kalman filter and camera-motion compensation in crowd).
- **OSNet**: a re-ID model that turns a person crop into a 512-d number vector so the app can tell "same person" from "new person."
- **DM-Count**: an optional density model whose map sums to an estimated crowd size; runs in a Web Worker.
- **Re-ID**: re-identification — matching a person who left and came back, using appearance rather than only box overlap.
- **Service worker**: a background script that can intercept network requests and serve files from Cache Storage.
- **Cache Storage**: a browser store for request/response pairs, used here for the runtime and models.
- **Secure context**: HTTPS or `127.0.0.1`; required for the camera API.
- **coco-ssd**: a general object detector used only by the original counters, not by crowd.
