TCount Crowd — on-device crowd counter (follows perplexity/report.md)
======================================================================
Copy the whole "crowd" folder next to your other TCount files on the phone, start the server in iSH
in the parent folder:   python3 -m http.server 8080
and open in Safari:     http://127.0.0.1:8080/crowd/
(127.0.0.1 / localhost / https are required for the camera.)

Pipeline: camera (one rear lens at a time) -> YOLOX person detector (ONNX Runtime Web, WebGPU,
falls back to WASM/CPU; coco-ssd as last resort) -> ByteTrack tracker with camera-motion
compensation -> OSNet re-ID "fingerprints" -> session ledger (never forgets counted people).
Gender is intentionally not estimated (see report section 7).

Counters
  In Frame        people currently tracked and counted
  Unique          everyone counted this session (personIds)
  Possible dupes  people who looked similar to someone already counted but not similar enough to be
                  sure. They are counted, so the true total is in the range shown (e.g. 21–23).
                  Tap the tile to review (by P-label on screen) and Merge / mark Different.
  Est. crowd      (dense mode only) density-map estimate of people visible right now — separate
                  number, never added to Unique.

Colours: green = counted, orange "P7?" = possible duplicate, blue = counted without an
appearance check (too small/occluded), grey dashed = not yet confirmed.

Settings (⚙): detector (Auto/Tiny/Nano/coco-ssd), backend, input size, small-people tiling,
dense mode (Auto/On/Off), lens policy (Auto ladder/Manual + lens list), camera-motion
compensation, same-person / new-person thresholds (default 0.80 / 0.65), confirm frames,
experimental multi-thread CPU mode (the same service worker adds COOP/COEP, page reloads once).

The service worker is always on. It caches the app, the models and the AI runtime on the device (cache name tcount-crowd-1.2) so the next launch is fast and works offline. A new version replaces that cache. Dense mode On is the only control that downloads the 21 MB density model; Auto uses the detector's tiled estimate instead. Large files can be loaded from ?assetBase=https://host/dir (CORS required; see the repo README).

Auto policies
  - No people / small people  -> tiled inference on the same frame (no lens change)
  - No people for 6 s         -> switch to ultra-wide (if Safari exposes it)
  - 1–2 tiny distant people   -> telephoto; counting pauses 1.5 s, everyone is re-matched by
                                 appearance, arrivals right after the switch are marked "possible"
  - Many/overlapping people or density probe >> detections -> dense mode
  - Dark scene (median brightness < 40/255 for 2.5 s) -> warning, torch button (if supported),
                                 detector input brightened. Same confirmation as in normal light.
  - Camera frames read as pure black -> tries other frame-grab methods, then a red banner.
Zoom slider uses native camera zoom if Safari really applies it (verified), else digital crop-zoom.

CSV (⬇): one row per second: timestamp,in_frame,unique,possible_dupes,estimated_crowd,lens,
unique_low,unverified,light,backend,fps

New people are registered on their 2nd detected frame (Settings: Confirm frames, default 2).
Extras are cleaned up afterwards: same-appearance duplicates are merged (every second and at
Stop & Total) and split-second boxes (< 0.7 s, then gone) are dropped as noise.

Privacy: NO snapshots or images are ever stored. Only numeric appearance fingerprints are kept, in
memory on the phone, and erased on Count Again / page close.

Troubleshooting: tap 🩺 Self-test. It lists every file with HTTP status / size / type, checks
WebAssembly SIMD, wasm memory, WebGPU, camera brightness, and test-loads the detector. Screenshot it.
  - "HTTP 404" or "incomplete": re-unzip crowd-part1-app.zip (dense model: part 2 into crowd/models)
  - "request failed": the iSH server stopped — run python3 -m http.server 8080 again
  - "FRAMES ARE BLACK": stop/start the camera or reload the page
Debug (🐞) shows backend, timings, camera capabilities, lens list, tracker/ledger state, errors.
URL options for testing: ?backend=wasm|webgpu&model=nano|tiny&size=640&denseEp=wasm
