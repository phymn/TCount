# TCount

Counts people with a phone camera, running entirely in the browser (iPhone Safari served from iSH). It has two generations:

| Path | What it is |
|---|---|
| `crowd/` | **TCount Crowd**, the main app. An offline crowd counter: YOLOX person detector → ByteTrack tracker with camera-motion compensation → OSNet re-ID "fingerprints" → a session list that never counts the same person twice, with a "possible duplicates" range (e.g. 21–23). It also has a dense-crowd density estimate (DM-Count), automatic ultra-wide/tele lens switching, and low-light handling. All models are bundled and run on ONNX Runtime Web (WebGPU → WASM). **Gender is deliberately not estimated.** |
| `index-nogender.html` | The original TCount people counter (coco-ssd + IoU tracker). Counts In Frame and Unique, with no gender. |
| `index.html` | The original TCount with face-api gender voting (♂/♀ counts). Kept for history. Gender estimates are unreliable and biased, so `crowd/` and `index-nogender.html` are the recommended versions. |
| `legacy/index.original.html` | Older backup of `index.html` |

## Run on iPhone (iSH)
1. **Get the two large files.** They aren't in git because of GitHub's file-size rules (see *Large files* below). Run `sh tools/fetch_assets.sh`, or download them from the Release by hand and put them here:
   - `crowd/vendor/ort-wasm-simd-threaded.asyncify.wasm` (26.8 MB, from onnxruntime-web 1.30.0)
   - `crowd/models/dmcount_qnrf_w8.onnx` (21.6 MB, optional. Without it, dense mode falls back to a rougher tiled-detection estimate.)
2. Copy the repo into iSH, for example `~/tcount`, keeping the `crowd/` layout (`crowd/vendor/`, `crowd/models/`).
3. Run `cd ~/tcount && python3 -m http.server 8080`.
4. In Safari, open **`http://127.0.0.1:8080/crowd/`**. The original counters are at `http://127.0.0.1:8080/index-nogender.html` and `/index.html`.
   The camera only works on `127.0.0.1`, `localhost` or `https`. Opening the file from the Files app won't work.
5. Tap **Start Camera**. If anything fails, tap **🩺 Self-test** and take a screenshot. It checks every file's HTTP status and size, WebAssembly SIMD and memory, WebGPU, camera brightness, and test-loads the detector.

`crowd/` runs fully offline. The original `index*.html` files need internet, because they load TF.js, coco-ssd and face-api from jsDelivr.

## Counters (crowd)
- **In Frame**: people tracked right now.
- **Unique**: everyone counted this session.
- **Possible dupes**: people who looked similar to someone already counted. They're counted, so the true total falls in the range shown. Tap the tile to Merge them or mark them Different.
- **Est. crowd**: the dense-mode estimate. It's a separate number and is never added to Unique.
- **Stop & Total**, **Count Again**, and **CSV export**, which writes one row per second.

## Large files (not in git)
| File | Size | Why it's left out | Where to get it |
|---|---|---|---|
| `crowd/vendor/ort-wasm-simd-threaded.asyncify.wasm` | 26,781,914 B | over 5 MB; an unmodified upstream binary | jsDelivr `onnxruntime-web@1.30.0/dist/`, or Release v1.0 |
| `crowd/models/dmcount_qnrf_w8.onnx` | 21,560,628 B | over 5 MB | GitHub Release v1.0 asset (attach the file from the zip) |

The small models (`yolox_nano` 1.1 MB, `yolox_tiny` 5.2 MB, `osnet_x0_5` 0.9 MB) are committed. Git LFS would work too, but Release assets are simpler for iSH users.

## Repo layout
- `crowd/`: the app (`index.html` UI, `app.js` pipeline, `dense-worker.js` density worker, `sw.js` optional COOP/COEP helper, `LICENSES.txt`, `README.txt` quick reference)
- `docs/HOW-IT-WORKS.md`: full technical write-up
- `docs/research/`: the Perplexity crowd-counting research report and task brief that this build is based on
- `tools/export/`: scripts that turned the PyTorch weights into the int8 ONNX models, plus the detector and re-ID evaluation scripts
- `tools/test/`: Playwright test harness with a fake camera and synthetic scenes
- `tools/fetch_assets.sh`: downloads the large files

## Limitations
- Safari can only use one camera at a time, so lenses are switched one after another, never fused.
- People coming back into view are matched by appearance. Similar clothing can merge two people, and a big change in pose or light can split one person into two. Unsure cases show up as "possible".
- The live Unique number can briefly go up after people cross paths, then settles when duplicates are merged (every second, and again at Stop & Total).
- Real darkness lowers the count rather than guessing. The torch only helps for nearby people.
- It hasn't been tested much on a real iPhone yet. Testing was done in simulated Chromium and WebKit (see the docs).

## License and credits
- Recommended license for the original code: **MIT**. No LICENSE file has been added yet.
- Bundled third-party code and models (details in `crowd/LICENSES.txt`):
  - ONNX Runtime Web 1.30.0 (MIT, © Microsoft). `ort-wasm-simd-threaded.asyncify.js` has one change: its memory maximum is set to 1 GB.
  - YOLOX nano and tiny, Megvii (Apache-2.0)
  - OSNet x0.5 from deep-person-reid / torchreid (MIT), MSMT17 weights. MSMT17 is a research dataset, so check its terms before commercial use.
  - DM-Count (MIT), UCF-QNRF weights. Also a research dataset.
  - coco-ssd and TF.js fallback (Apache-2.0), loaded from CDN and not bundled
  - face-api (MIT), used by the original `index.html` and loaded from CDN
- The permissively licensed YOLOX was picked on purpose instead of Ultralytics YOLO, which is AGPL.

**Privacy:** no images or snapshots are ever stored. Only numeric appearance fingerprints are kept, in memory, and they're erased on Count Again or when the page closes.

Author: Jesus Teran ([@phymn](https://github.com/phymn))
