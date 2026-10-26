# TCount: How it works

This covers two apps: the original counters (`index.html`, `index-nogender.html`) and the new offline **crowd counter** (`crowd/`). The crowd design follows the research brief in `docs/research/report.md`, with the changes described here.

---
## Part A: Original TCount (`index-nogender.html`, `index.html`)
- **Libraries (CDN)**: TF.js and `@tensorflow-models/coco-ssd@2.2.3`. `index.html` also loads `@vladmandic/face-api@1.7.15` (TinyFaceDetector + AgeGender net).
- **Loop**: `personModel.detect(video)` → keep `class==='person' && score > 0.45` → tracker update → draw → counters.
- **Tracker** (`PersonTracker`): pair score = 2·IoU − centroid distance / frame diagonal, greedy matching, minimum 0.15. A new track becomes **confirmed and counted once** after `minHits = 2`, and it's dropped after `maxAge = 14` missed frames. Counters: In Frame (confirmed tracks seen this frame) and Unique (confirmed IDs). There's also Stop & Total, Count Again and CSV export.
- **Gender (`index.html` only)**: every 3rd frame, `estimateGender()` crops each tracked person's head area, scales it up, and runs face-api at inputSize 160 with `withAgeAndGender()`. People still without a face get a full-frame pass at inputSize 416, matched to head positions. `assignVote()` keeps the last 5 votes, and a gender is only fixed once 3 of them agree. Then the ♂/♀ counters go up. This is deprecated: it's unreliable and biased, which is why `crowd/` drops it.

---
## Part B: TCount Crowd (`crowd/`)

### B1. Files
| File | Role |
|---|---|
| `crowd/index.html` | UI: video and overlay canvas, counter tiles (In Frame, Unique, Possible dupes, Est. crowd), Start/Stop & Total/Count Again, ⚙ Settings, 🐞 Debug, 🩺 Self-test, ⬇ CSV, 🔦 Torch, zoom slider, lens list, possible-duplicate review sheet. Loads `vendor/ort.webgpu.min.js`, then `app.js`. |
| `crowd/app.js` | The whole pipeline (version `crowd-1.2`): asset loader, ORT setup, detectors, re-ID, Kalman/ByteTrack, motion estimator, ledger, camera/lens manager, light logic, identity decisions, auto policies, drawing, CSV, self-test, settings. |
| `crowd/dense-worker.js` | Web Worker running DM-Count on a second ORT runtime, so the camera loop never freezes. It returns a density map whose sum is the estimated count. The page passes the model bytes in; the worker does not download them on its own unless that hand-off is missing. |
| `crowd/sw.js` | Service worker. It keeps a versioned Cache Storage (`tcount-crowd-1.2`) of the app shell and of every wasm/onnx file the page actually fetched, and it serves those when the network is gone. A new version name drops older `tcount-*` caches. COOP/COEP/CORP headers are added only when Settings → Multi-thread CPU is on, which `python -m http.server` can't do itself. |
| `crowd/vendor/ort.webgpu.min.js` | ONNX Runtime Web 1.30.0 (WebGPU and WASM) |
| `crowd/vendor/ort-wasm-simd-threaded.asyncify.{js,wasm}` | ORT WASM glue and binary. The glue was renamed from `.mjs` to `.js` so python http.server sends a JS MIME type, and its shared-memory maximum reads `globalThis.__ortMaxPages`. |
| `crowd/models/yolox_{nano,tiny}_person_w8.onnx` | YOLOX person detectors: dynamic input size, output cut down to the person class, int8 weights (`DequantizeLinear`) with fp32 compute |
| `crowd/models/osnet_x0_5_reid_w8.onnx` | OSNet x0.5 re-ID, giving a 512-d embedding from a 128×256 crop |
| `crowd/models/dmcount_qnrf_w8.onnx` | DM-Count VGG19 density model with UCF-QNRF weights (fetched separately, optional) |

### B2. Startup and asset loading
- **Environment checks**: `file:` → a banner explains that the folder has to be served. Not secure and not `127.0.0.1` → a "camera blocked" banner. ORT script missing → it warns and tries the coco-ssd CDN fallback.
- **`fetchAsset(rel)`**: device cache first (Cache Storage, name `tcount-` + the app version), then the network, with progress and a 180 s timeout. A hit has to match the expected size or it is deleted. The same-origin URL carries `?v=<version>` so the browser HTTP cache cannot revive the previous build. Local misses for the wasm and the dense model try `?assetBase=` (basename on that host), then the wasm falls back to jsDelivr `onnxruntime-web@1.30.0` and both files fall back to the v1.0 GitHub release URL. Release assets on github.com do not send CORS headers, so a browser cannot read them; copy them into `crowd/` with `tools/fetch_assets.sh`, or point `assetBase` at a CORS-enabled host. Size, status and HTML-error checks are unchanged.
- **Warm prefetch**: on idle, before Start, the page pulls the wasm and the detector it will actually use (Nano when Auto has no WebGPU). Re-ID bytes follow. The dense model is not prefetched.
- **`setupOrt()`**: sets `__ortMaxPages = 1024 MB × 16` pages. iOS Safari can refuse the default 4 GB shared-memory reservation, and 1 GB is plenty. Override it with `?maxmem=`. `numThreads` is 1 unless the page is `crossOriginIsolated`. The 26 MB `.wasm` is **downloaded by the app itself** (or read back from the device cache) and passed in as `wasmBinary`. iSH serves `.wasm` as `application/octet-stream`, which breaks Safari's streaming compile and triggers a slow second download.
- **Start**: the camera opens in parallel with the remaining model work. Counting begins as soon as YOLOX is ready. OSNet loads in the background, so the first frames can count before appearance dedupe is attached. Auto on a CPU-only device starts on Nano. Each execution provider gets one warmup run.
- **`makeSession()`**: one cached byte buffer per model, `graphOptimizationLevel:'all'`, timeouts of 45 s (WebGPU) and 120 s (WASM). Models load **one at a time** to keep memory use low.
- **`YoloxDetector.load()`**: tries WebGPU when it's available, then one warmup run at 320×320 (shader compile on WebGPU, and the timing sample). WASM is only also tried if WebGPU failed or took more than 45 ms. The faster one wins and the other session is released. Settings: Auto, Tiny, Nano, or coco-ssd. Auto on CPU starts with Nano; Auto with WebGPU starts with Tiny and can still fall back to Nano when the CPU run is slower than 60 ms.

### B3. Per-frame pipeline (`step()`)
```
grabFrame → MotionEstimator (gray 160 px: luma stats + camera shift) → updateLight → detector.gain
→ regionsFor (digital zoom crop [+ 2×2 tiles]) → YOLOX detect (+NMS) → ByteTracker.update(dets, motion)
→ processIdentities (re-ID + ledger) → crowdedness → autoPolicies → adaptSize → draw / UI / CSV
```
- **Frame grab**: some iOS versions return black frames from a `willReadFrequently` canvas. After 15 black frames (`blackLuma` 6, P98 luma), `setGrabMode` cycles through CPU canvas → GPU canvas → ImageBitmap. If all three stay black, a red "camera picture is black" banner appears.
- **YOLOX** (`runRegion`): letterbox each region to a multiple of 32 with the long side = `size` (Auto: 416/512/640/800; `adaptSize` steps down above 250 ms per region and up below 45 ms when people are small). Gray padding is 114. Input is BGR 0–255, multiplied by `gain` in low light. Decoding runs over strides 8/16/32: `score = obj × cls`, `cx = (o + gx)·stride`, `w = exp(o)·stride`. Detections below `trackLow` 0.1 are dropped, then NMS at 0.45. With tiles, it's NMS 0.5 plus an IoMin 0.75 merge across tiles.
- **Tiling** (`regionsFor`): the visible area plus four 60 % tiles in a 2×2 grid. This acts as "inference-level zoom" for small people without touching the camera stream, so coordinates don't break.

### B4. Tracking: ByteTrack (`KF`, `hungarian`, `STrack`, `ByteTracker`)
- **Kalman filter** in xyah state (centre x, y, aspect, height) with constant velocity and ByteTrack noise weights `WP = 1/20`, `WV = 1/160`. It splits into four exact 2×2 filters. `kf.shift(dx, dy)` applies **camera-motion compensation**.
- **Association**: stage 1 matches high detections (≥ `trackHigh` 0.45) to tracked plus lost tracks, with cost 1 − IoU·score, Hungarian, threshold 0.8. Stage 2 matches low detections (0.1–0.45) to the tracks still unmatched, IoU threshold 0.5, which gets people through occlusion. Leftover tracks become *lost*. Unconfirmed tracks are matched to leftover high detections at 0.7. New tracks are born from high detections ≥ `newTrack` 0.5. Lost tracks are kept for `lostSeconds` = 3 s.
- **MotionEstimator**: sparse block matching on a 160-px gray image. 12×12 blocks on a 14-px grid, search ±10 px, people boxes masked out, textureless blocks (variance < 40) skipped. Blocks only vote when best SAD < 0.9 × second best. The median vote plus ±1 px inliers give dx, dy, used when confidence ≥ 0.4 and there are at least 5 votes. The same pass gives the luma mean, median and P98 for the light logic.

### B5. Identity and counting (`processIdentities`, `Ledger`)
The tracker's IDs are short-lived. The **Ledger** holds permanent **personIds** for the session and never forgets anyone who was counted.
1. **Crops**: `cropQuality` rejects crops under 56 px tall or 20 px wide, boxes that are only predicted (not detected), boxes overlapping another by IoMin > 0.3, boxes touching the frame edge, and scores under 0.3. Quality = min(1, h/160) × score. Embeddings are computed for undecided tracks first, then tracks just out of occlusion, then every `reidEvery` = 900 ms. Up to `reidBatch` = 6 per frame. Each track keeps its last 6 embeddings.
2. **OSNet**: crop resized to 128×256, ImageNet normalisation, giving an L2-normalised 512-d vector. Similarity is the dot product (cosine).
3. **Registration**: a track is decided once it has `confirm` = 2 frames (Settings), and only from frames with a high-confidence detection (≥ 0.45). That blocks dark noise and fragment boxes. If no clean crop exists yet, it waits up to 3 frames.
   - Similarity to the best ledger person (excluding people visible at the same moment, and anyone the track has been seen with) **≥ `high` 0.80** → **re-use** that person (a returning person, not counted again).
   - **0.65–0.80** (`low` to `high`) → create a **possible** person (counted, shown orange "P7?", and making up the dupes range).
   - Below 0.65 → create a **new** person.
   - No crop at all → **unverified** (blue), or *possible* if it shows up just after a lens switch (`carryCandidate`).
   - **Swap check**: if the crop strongly matches a person currently claimed by another visible track, the app re-checks that track's newest embedding. If it no longer looks like that person, the tracker swapped IDs during a crossing. The person is reclaimed and the other track is decided again.
4. **Evidence accumulation**: new embeddings are only added to a person's gallery when they agree with it (≥ `low`). Galleries are capped at 8 samples, replacing the most redundant one (> 0.97). Similarity to a person = the mean of the top 2 gallery matches. Two consecutive crops below `low − 0.15` mean an **ID switch**: the track is detached and re-matched (`id-swap-fixed`, `id-fixed`, or decided again). A possible person whose candidate is seen in the same frame becomes *new*. With more evidence, ≥ 0.80 **auto-merges** them, and below 0.65 after 3 samples resolves them as *new*. A *new* person can still be merged late within `recheckWindowMs` = 6 s (≥ 0.83 with 3 samples), or flagged *possible*.
5. **Co-visibility memory**: verified identities seen in the same frame add to `coSeen`. Three or more frames together means they're provably different, so they'll never be merged.
6. **Reconcile** (every 1 s and at Stop & Total): `dropGhosts` removes people seen for less than 0.7 s and fewer than 8 frames who then vanished for 2.5 s (fragments). Persons created in the last 30 s whose galleries match an older one are merged when gallery-to-gallery similarity is ≥ 0.86 with fewer than 3 frames together, or ≥ 0.92 with fewer than 10.
7. **Counters**: Unique = ledger persons not merged. Possible = persons in the *possible* state, so the range is Unique − Possible to Unique. In Frame = visible tracks holding a personId. The review sheet lets you Merge or mark Different by hand.

### B6. Light handling (`updateLight`)
- The light reading uses the **median** luma of the raw frame, with an EMA (0.85/0.15). States: `dark` < 40, `verydark` < 18, back to `ok` > 48. Entering a dark state takes 2.5 s, and leaving takes 1.2 s.
- The first `lightWarmupMs` = 2 s after the camera starts are ignored, because iPhone's first frames are black while auto-exposure settles. That was the cause of the false "Low light" warning. Black frames (P98 ≤ 6) count as a capture problem, not darkness.
- In the dark: a warning banner, a torch button (when `torch` is in the track's capabilities), and the detector input is brightened with `gain = clamp(100/luma, 1, 4)`. **Thresholds are not lowered**, so in real darkness the app counts fewer people instead of making them up.

### B7. Lenses and zoom (`enumerateCams`, `openCamera`, `switchLens`, `autoPolicies`)
- Safari only allows one camera stream, so lenses work as modes used one after another. `lensKindOf` classifies them by label (Ultra Wide, Telephoto, wide).
- Auto ladder, with at least 10 s between switches: no people for 6 s → ultra-wide. 1–2 people under 6 % of the frame height for 5 s → telephoto. Telephoto with more than 3 people, or nobody for 4 s → back to wide.
- On every lens or zoom change, `markLensChange` resets the tracker and motion, pauses decisions for 1.5 s, and keeps a carry-over list for 8 s. Anyone appearing then is matched by appearance or marked *possible*.
- The zoom slider uses native `zoom` constraints when Safari really applies them (this is checked), and otherwise a digital crop zoom.

### B8. Dense mode (`dense-worker.js`, `requestDense`, `tiledEstimate`)
- **On** (the setting, or `?dense=on`) is the only path that downloads `dmcount_qnrf_w8.onnx`. Auto still switches the estimate on when there are ≥ 25 high detections, or ≥ 10 with crowdedness > 0.35, and switches it off after 6 s, but that estimate is the tiled detector, not DM-Count. Off never downloads the model.
- The worker starts only 8 s after the main models are running (iPhone memory; `?denseNow=1` skips the wait) and reuses the page's WASM bytes plus the model bytes the page just cached. It tries WebGPU first, falling back to WASM if setup takes more than 20 s or a run takes more than 15 s. Input is 512 px wide (384 when slow), ImageNet normalisation. The density map is summed to give the count, and is drawn as a heat overlay.
- Fallback when the model is missing: full frame + 3×3 tiles at 416, score > 0.35, labelled "tiled detection (lower bound)".
- **Est. crowd is never added to Unique.** It's a separate, identity-free number.

### B9. Output, debugging and settings
- **CSV**, one row per second: `timestamp,in_frame,unique,possible_dupes,estimated_crowd,lens,unique_low,unverified,light,backend,fps`.
- **Debug (🐞)**: backend, timings per step, camera capabilities, lens list, tracker and ledger state, the last events (merge, ghost-dropped, id-swap-fixed, and so on), and errors.
- **Self-test (🩺)**: every asset's status, size and type, WASM SIMD, memory reservation, WebGPU, camera brightness, and a test detector load.
- **Settings** (saved in `localStorage` as `tcrowd-settings`, and overridable with URL parameters such as `?backend=wasm&model=nano&size=640&denseEp=wasm&maxmem=768`): detector, backend, input size, tiling, dense, lens policy and lens, camera-motion compensation, same-person / new-person thresholds (0.80 / 0.65), confirm frames (2), threads.

### B10. Why these thresholds
- **0.80 / 0.65 re-ID**: from `tools/export/reid_eval.txt` (Market-1501 style evaluation). For OSNet x0.5, a similarity of 0.8 gives a false-positive rate of about 0.07 %, so a re-use is very rarely a wrong merge. 0.65 is about 2.5 % FPR and 73 % TPR, so the 0.65–0.80 band is the "unsure" zone, shown as a range instead of a guess.
- **confirm = 2**: the user asked for people to be registered on almost every new frame. Two frames still filter out one-frame flickers, and ghost removal plus reconcile clean up the rest.
- **ByteTrack 0.45 / 0.1 / 0.5 / 0.8**: the defaults from the ByteTrack paper, adjusted to YOLOX-tiny score ranges.

### B11. How the models were made (`tools/export/`)
- `export_yolox.py`: official YOLOX nano and tiny `.pth` → ONNX (opset 13, dynamic H/W, decode removed from the graph).
- `export_osnet.py`: torchreid OSNet variants → ONNX. `evalreid.py` compared them, and **osnet_x0_5 MSMT17** won on accuracy against speed.
- `export_dmc.py`: DM-Count QNRF → ONNX.
- `wq.py`: weight-only int8 per output channel via `DequantizeLinear`, with an optional person-only output slice. It roughly quarters the file size, and compute stays fp32.
- `evaldet.py` (COCO val person AP) and `modeltest.py` check accuracy after conversion.
- They need the upstream repos (YOLOX, deep-person-reid, DM-Count), the weights and the datasets, none of which are included. Clone them next to the scripts.

### B12. Testing done (`tools/test/`)
- `fakecam.js` replaces `getUserMedia` with a canvas scene: background photo, person sprites moving along paths, camera pans, multiple lenses with zoom factors, brightness and noise.
- `scenes.py` → `scenes.json`, with scenes like walk3, reenter, similar, similar_together, pan, empty, far people, dense crowd, dark and lens switching, each with a ground-truth count. The sprites and backgrounds aren't included.
- `run.py --browser chromium|webkit --scenes … --params …` runs a scene in Playwright and samples `window.__tc` every second, giving the final Unique, possible and in-frame counts against ground truth. `summ.py` summarises the results.
- `ishserve.py` copies iSH's http.server behaviour: no wasm/onnx MIME types, single-threaded, optional bandwidth limit. Set `SITE=<folder containing crowd/>`.
- `selftest.py` (missing, truncated, wasm 404 and black-frame cases), `loadcheck.py`, `errtest.py`, `lenstest.py` and `probe.py` cover loading, errors and lens behaviour.
- Results (simulated): walk-by, moving camera, empty scene and far-away people were all exact. Dense crowds came within a few people. After the re-entry fix, WebKit got re-entry exactly right in 13 of 13 runs and dark in 7 of 7. Load under iSH-like serving took about 16 s. Testing on a real iPhone is still to do.

### B13. Known limits and future work
- Re-ID depends on clothing. Identical outfits can merge people, and coats on or off can split one person.
- Brief jumps in the Unique count after crossings, which settle within about 1 s.
- Real-device tuning: iPhone WebGPU speed, memory with dense mode on, and lens labels on different iOS versions.
- The versioned device cache (service worker + Cache Storage) is what makes the second launch and offline launch skip the 26 MB runtime. Possible additions: a better re-ID model (OSNet-AIN or a CLIP-ReID distillation), a counting line or zone, and exporting the ledger events.
