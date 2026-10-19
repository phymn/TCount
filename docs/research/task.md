# Perplexity Computer task: On-Device Crowd Counting Web App
URL: https://www.perplexity.ai/computer/tasks/f326480d-b5cb-475c-b8a0-ecb98e78f3fd

## USER PROMPT (11:12 AM)
Research the best way to count people amount a big crowd, there can be bright lights or it could be dark. How to keep track of everyone already counted, this has to run on device it has to be web based, run an small ai model to keep track and hopefully distinguish between genders, count as many people as possible. This webapp can be general for any mobile device but think of this as if this has to run using the whole power of hardware from an iPhone 18 pro, can use the whole three cameras set on the back, automatically zoom if not detecting the person on people.

(Running advisor, 11:12 AM) Follow-up from user: When I mean keep track it means not count the same person again

## RESPONSE (Finished, 9m 14s)
The research is complete — the full report is in the shared file. Here are the key findings:

### How to never count the same person twice
The count comes from the tracker, not the detector. The recommended pipeline: a lightweight detector (YOLO11n) → a multi-object tracker (ByteTrack/BoT-SORT, which includes camera-motion compensation for a handheld phone) → re-identification embeddings (OSNet) matched against a session-long ledger of everyone already counted. A new track only increments the total if its appearance fingerprint doesn't match the ledger. Honest limit: this reduces re-counts substantially but can't guarantee a perfect unique total across long occlusions or camera pans — the UI should show an uncertainty indicator.

### iOS Safari reality checks
- All three rear cameras at once is not possible — Safari allows only one active camera stream; a second getUserMedia() call kills the first. But each rear lens is separately selectable via enumerateDevices() since iOS 16.3, so you can switch between ultra-wide/wide/telephoto as modes.
- Auto-zoom: Apple's Safari 17 release notes list zoom in MediaTrackCapabilities, but third-party reports conflict on iOS — verify at runtime; digital crop-zoom and lens switching are the reliable fallbacks. "No detections" should trigger wider coverage (ultra-wide) before zooming in.
- Darkness: torch works on current iOS Safari; it's near-field light, not a crowd-scale fix. The main camera's ƒ/1.48 aperture is your low-light hardware.
- WebGPU is on by default in Safari 26/iOS 26, but a web app uses GPU+CPU — not the Neural Engine — and ONNX Runtime Web's WebGPU operator support must be validated with your exported models.

### Gender classification: recommend dropping it
Appearance-based gender classifiers lost 18–60 points of accuracy outside their training distribution, with error rates up to 34.7% for darker-skinned women in commercial systems — worst exactly in this app's conditions (crowds, distance, low light). If demographics are required, voluntary self-reporting is the accurate path.

One licensing flag worth knowing: Ultralytics YOLO models are AGPL-3.0 by default — commercial closed-source use requires their Enterprise license.

If you want, I can build a working prototype webapp next (camera + detection + tracking with the dedup ledger) so you can test it on an actual iPhone.

Artifact: "Crowd Counter Research Report" (Document, 242 sources) -> see report.md

Suggested follow-ups: "Build the working prototype webapp next"; "Create a minimal prototype with camera detection and the deduplication ledger"; "Make the prototype testable on an iPhone in Safari"

---
## ARTIFACT: Crowd Counter Research Report (downloaded markdown, verbatim)

# On-Device, Web-Based Crowd Counting: Research Report

Goal: count people in a large crowd from a phone, in bright light or darkness, without counting the same person twice — running entirely on-device in a web app, with an iPhone 18 Pro as the reference device.

---

## 1. Feasibility Summary

| Requirement | Verdict |
|---|---|
| Real-time person detection on-device in the browser | Achievable — WebGPU is enabled by default in Safari 26 / iOS 26 ([LocalMode compatibility table](https://localmode.dev/blog/compatibility/webgpu-support), [Safari 26 release notes](https://developer.apple.com/documentation/safari-release-notes/safari-26-release-notes)), and nano YOLO-class models run at interactive rates in browsers in published demos ([mojostudio browser CV 2026](https://mojostudio.in/blog/computer-vision-mediapipe-tensorflow-web-2026)) — throughput on the target device is unmeasured |
| Avoiding re-counts while people stay in frame | Substantially achievable — multi-object tracking keeps stable IDs per person ([Fora Soft tracker guide](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/multi-object-tracking-deepsort-bytetrack-ocsort)); ID switches can still occur under heavy occlusion or camera motion |
| Not re-counting someone who leaves and returns | Partially achievable — re-identification embeddings work for short gaps; long occlusions, camera pans, and revisits cannot guarantee a perfect unique total |
| Using all three rear cameras at once | Not established for the target device — WebKit documentation and bug history indicate iOS Safari allows only one active camera stream at a time ([WebKit WebRTC blog](https://webkit.org/blog/7763/a-closer-look-into-webrtc/), [WebKit bug 179363](https://bugs.webkit.org/show_bug.cgi?id=179363)); design around one active rear stream |
| Automatic zoom | Partially achievable — Safari 17 exposes `zoom` in MediaTrackCapabilities per Apple's release notes ([Safari 17 release notes](https://developer.apple.com/documentation/safari-release-notes/safari-17-release-notes)), but third-party reports conflict on iOS usability ([Dynamsoft](https://www.dynamsoft.com/codepool/camera-zoom-control-on-web.html)); lens switching and digital crop-zoom are the fallbacks |
| Gender breakdown | Not reliably achievable — appearance-based gender classification is inaccurate and biased (see §7); recommend leaving it out |

The single most important design decision: a stable count comes from the tracker, not the detector. Detection gives you boxes per frame; the tracker stitches them into trajectories with IDs, and you count trajectory events, not detections.

**Defensible conclusion up front:** this architecture can reduce repeat counting substantially, but a moving phone cannot guarantee an exact unique-person total across occlusions, darkness, and revisits. The app should show a count with an explicit uncertainty indicator rather than a single exact number.

---

## 2. What "Don't Count the Same Person Twice" Actually Requires

There are three different counting problems, and they need different machinery:

1. **Visible count** — how many people are in the current frame. Detector output alone.
2. **Flow count** — how many people cross a line/region (entrances, walkways). Tracker + counting-line logic. Note that repeated line crossings are events, not necessarily new people — keep flow totals separate from unique totals.
3. **Unique total across a moving camera** — how many distinct people exist across a handheld scan. Tracker + re-identification + a session-long deduplication ledger. The hardest one.

The standard anti-double-counting pipeline is **detector + multi-object tracker + counting logic**:

- **ByteTrack** is the standard fast tracker: it associates high-confidence detections to tracks first, then rescues partially occluded people by matching leftover tracks against low-confidence detections. A browser implementation or port must be integrated and benchmarked with the chosen detector. ([Fora Soft tracker guide](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/multi-object-tracking-deepsort-bytetrack-ocsort), [Fora Soft YOLO/DeepSORT guide](https://www.forasoft.com/blog/article/ai-video-surveillance-yolo-deepsort-guide))
- **BoT-SORT** is ByteTrack plus camera-motion compensation and optional appearance features — important for a handheld phone that pans, because plain IoU trackers break when the whole scene shifts. Use it when the scene is crowded or the camera moves. ([Fora Soft](https://www.forasoft.com/blog/article/ai-video-surveillance-yolo-deepsort-guide))
- **DeepSORT-style appearance matching** adds a small CNN that produces a 512-D "fingerprint" per person, compared by cosine similarity to survive brief occlusions and re-associate people who reappear. ([DeepSORT overview](https://kb.osu.edu/server/api/core/bitstreams/99695c7c-4386-4749-a10c-3e2b611e83c3/content))

### Session-long deduplication ledger (the core of "never count twice")

Keep two separate identity layers:

- **`trackId`** — temporary, per-trajectory, managed by the tracker.
- **`personId`** — persistent for the whole counting session, managed by the ledger.

When a track is confirmed (seen for N consecutive frames above a confidence threshold — this filters detection flicker), extract an appearance embedding and try to match it against the ledger of everyone already counted this session, including tracks that have retired. Match → reuse the existing `personId`, do not increment. No match → assign a new `personId`, increment the total.

Honest limits, which the UI must reflect:

- Appearance matching produces **false merges** (two different people counted once — undercounting) as well as misses (one person counted twice). Both error directions exist.
- When a match is ambiguous (similarity score in an uncertain band), treat it as **unresolved** — park it in a "possible duplicate" state rather than automatically declaring "new person," and allow later merges to correct a duplicate increment when evidence accumulates.
- Tracker IDs are not permanent identities. A person fully occluded for minutes, or a camera pan that sweeps past them and back, can create a new ID.
- Flow counts (line crossings) and unique totals must be tracked as separate numbers.

Supporting techniques validated in production people-counting research:

- Count only high-confidence tracks with a minimum appearance frequency — filters flicker-based double counts. ([Robust Deep SORT paper](https://archimer.ifremer.fr/doc/00728/83972/118509.pdf))
- **Three-state line-crossing validation** (before → on → after the counting line) reduces false counts from motion jitter and partial occlusion. ([Springer, improved camera-based people counting](https://link.springer.com/chapter/10.1007/978-981-95-9843-4_24))

---

## 3. The Hardware and Browser Reality (iPhone 18 Pro)

The iPhone 18 Pro runs the **A20 Pro** chip: "6-core CPU with 2 super cores and 4 efficiency cores", "7-core GPU with Neural Accelerators", and a "Dual 16-core Neural Engine" (Apple's wording — 32 Neural Engine cores in total). The rear "Pro Fusion" system is three 48MP cameras: a 24 mm main with variable aperture (ƒ/1.48–ƒ/4.0), a 13 mm ultra-wide, and a 100 mm (4x) telephoto at ƒ/2.8. Apple's specifications additionally list a "12MP optical-quality 8x Telephoto: 200 mm", but the page does not explain how that additional setting is implemented — do not assume it is a separate physical lens. The device also carries a LiDAR Scanner. ([Apple official specs](https://www.apple.com/iphone-18-pro/specs/))

Two things to be explicit about:

- **This proposed WebGPU/WASM pipeline has no verified Neural Engine acceleration path — do not budget for it.** A web app as specified computes on the GPU (WebGPU) or CPU (WASM); the 32-core Neural Engine is not an execution target for this stack, so "whole power of hardware" in a browser means GPU + CPU.
- **All camera-capability claims below are from primary WebKit/Apple documentation and third-party reports; none were verified on an iPhone 18 Pro itself.** Build every camera feature behind runtime capability detection.

### One camera at a time
WebKit documentation states Safari on iOS supports only **one active camera stream**; a second `getUserMedia()` call for a different camera kills the previous stream, and WebKit engineers confirm simultaneous capture on two devices is not supported. This is historical WebKit evidence, not a test on an iPhone 18 Pro — but it is the documented platform behavior to design around. ([WebKit WebRTC blog](https://webkit.org/blog/7763/a-closer-look-into-webrtc/), [WebKit bug 179363](https://bugs.webkit.org/show_bug.cgi?id=179363), [Stack Overflow](https://stackoverflow.com/questions/71698132/how-can-i-get-ios-to-have-2-video-streams))

What you *may* be able to do: since iOS 16.3, Safari can expose **each rear lens as its own selectable device** via `enumerateDevices()` — labels like "Back Camera (Ultra Wide)" / "(Wide)" / "(Telephoto)" have been reported, though Safari is inconsistent about what it reports between visits — and you can open a specific lens by `deviceId`. Whether all three lenses are enumerable on the target device must be verified on-device. The app should treat the three cameras as sequential modes (ultra-wide for maximum coverage, wide as default, telephoto for zoom), switching streams when the strategy calls for it. ([Dominik Schilling, iOS 16.3 back cameras](https://dominikschilling.de/notes/ios-access-all-back-cameras-mediadevices-api/), [WebKit bug 253186](https://bugs.webkit.org/show_bug.cgi?id=253186), [Larablocks field notes](https://larablocks.com/package/emuniq/filament-qr-scanner))

Two iOS quirks to code around: `deviceId`s are randomized every page load, so re-resolve by device `label` each session; and device labels only appear after the user grants camera permission, so call `getUserMedia()` once first. ([webrtchacks Safari guide](https://webrtchacks.com/guide-to-safari-webrtc/))

### Zoom
Apple's Safari 17 release notes list "Added support for exposing `zoom` in MediaTrackCapabilities" ([Safari 17 release notes](https://developer.apple.com/documentation/safari-release-notes/safari-17-release-notes)), but third-party testing reports that iOS Safari does not apply the `zoom` constraint in practice ([Dynamsoft](https://www.dynamsoft.com/codepool/camera-zoom-control-on-web.html)). These conflict; treat native zoom as **reported but unverified on the target device**. The robust pattern: check `getSupportedConstraints().zoom` and `getCapabilities().zoom`, attempt `applyConstraints`, verify the stream actually changed, and always keep the two web-reliable fallbacks — (a) switching lenses via `deviceId`, (b) digital crop-zoom from a higher-resolution capture on canvas/GPU. On Android Chrome, native zoom works and should be used when advertised ([web.dev PTZ article](https://web.dev/articles/camera-pan-tilt-zoom)).

### Torch (the main tool for darkness)
The `torch` constraint is reported to work on iOS Safari from roughly 17.5 onward, toggling the LED while the camera runs. Feature-detect with `track.getCapabilities().torch` before using. Important caveat: the torch is a nearby supplemental light — it helps people close to the phone in a dark scene, not a large crowd at distance. ([Anyline camera docs](https://documentation.anyline.com/web-sdk-component/latest/advanced-camera-api.html), [WebKit bug 280970](https://bugs.webkit.org/show_bug.cgi?id=280970))

### GPU compute: WebGPU on by default
Safari 26 / iOS 26 ships WebGPU enabled by default, so a current iPhone can run browser inference on the GPU. ([LocalMode WebGPU support table](https://localmode.dev/blog/compatibility/webgpu-support), [Safari 26 release notes](https://developer.apple.com/documentation/safari-release-notes/safari-26-release-notes)) No third-party throughput number transfers to this pipeline on this phone — measure it.

Two runtime realities to design around:

- **ONNX Runtime Web's WebGPU backend on Safari must be tested with your exact exported models.** The official documentation states: "All ONNX operators are supported by WASM but only a subset are currently supported by WebGL, WebGPU and WebNN" — so the stack is a candidate requiring compatibility validation, not an established deployment path. INT8 and FP16 exports are separate choices with different backend support; validate whichever you ship. The WASM fallback must be deliberately implemented, not assumed. ([ONNX Runtime Web docs](https://onnxruntime.ai/docs/tutorials/web/))
- For very small models, warm WASM can beat WebGPU on iOS because GPU dispatch/transfer overhead dominates (one iPhone SE2 benchmark reported 13 ms vs 36 ms per inference for a small model). Pick the backend at runtime and reuse one session — WebGPU session creation has been measured at 1.7–3 s on mobile. ([Zenn: ONNX Runtime Web on iOS Safari](https://zenn.dev/kaz_sakai/articles/ios-safari-onnx-memory?locale=en))

---

## 4. Recommended Stack

### Detection model
**YOLO11n (nano) exported to ONNX, quantized** — 2.6M parameters, ~6.5 GFLOPs, with file sizes on the order of 5 MB (smaller when quantized; treat as an estimate until you measure your export). It detects the COCO "person" class well enough for counting and is fast enough for real-time browser inference in published demos. ([Ultralytics YOLO](https://www.ultralytics.com/yolo), [Mi&Bee browser YOLO11n writeup](https://blog.mickeyzzc.tech/en/posts/mibee-oss/mibee-nvr-yolo11n/), [mojostudio browser CV 2026](https://mojostudio.in/blog/computer-vision-mediapipe-tensorflow-web-2026))

Three implementation cautions:

- **Resolution is a benchmark decision, not a constant.** Downsampling every frame to 320–512 px erases distant heads — the opposite of "count as many people as possible." Adapt input resolution to the scene; when people are small relative to the frame, run tiled inference with overlapping tiles and deduplicate detections in common frame coordinates.
- **Licensing:** Ultralytics confirms "All Ultralytics YOLO trained models fall under the AGPL-3.0 License by default" — you must either open-source your entire project under AGPL-3.0 or buy an Ultralytics Enterprise License for any commercial or closed-source use, including embedded deployments. Check this against your distribution model before committing. ([Ultralytics License](https://www.ultralytics.com/license))
- Frame-skipping (e.g., every 2nd–3rd frame) with interpolated tracks between inferences is the proven pattern for keeping mobile browsers smooth. ([Mi&Bee](https://blog.mickeyzzc.tech/en/posts/mibee-oss/mibee-nvr-yolo11n/))

For dense crowds where bodies overlap, detectors that look for **heads** instead of full bodies perform much better: head-top detection from a raised/elevated viewpoint is the standard bird's-eye approach, and dedicated lightweight head detectors (e.g., FCHD, HeadHunter) and occlusion-robust pedestrian detectors (a 2025 lightweight detector hits 83.9% AP50 on CrowdHuman at 7.1 ms inference on a GPU) exist for exactly this scenario. Note the trade-off: in head-only mode, appearance-based re-identification weakens, since OSNet-style models expect full-body crops. ([FCHD paper](https://arxiv.org/pdf/1809.08766v3), [HeadHunter / head tracking in dense crowds](https://inria.hal.science/hal-03184673/document), [Copernicus lightweight pedestrian detector](https://ms.copernicus.org/articles/16/877/2025/ms-16-877-2025.pdf), [OSNet input spec](https://aihub.qualcomm.com/mobile/models/osnet))

### Inference runtime
**ONNX Runtime Web** — WebGPU backend with WASM fallback, one runtime for both the detector and the re-ID model, model files cached with the Cache API after first load. Alternatives: MediaPipe Tasks Vision (WASM + GPU delegate, easiest for pose/face) and LiteRT.js (runs YOLO nano models fully client-side on WebGPU). Model export, quantization, and JS tracker integration are implementation work to verify, not a drop-in package. ([ONNX Runtime Web docs](https://onnxruntime.ai/docs/tutorials/web/), [YoloDetect LiteRT.js demo](https://github.com/sitammeur/YoloDetect))

### Tracker
**ByteTrack** (JS port exists; per-frame cost reported in milliseconds on server-class hardware) for moderate density; switch to **BoT-SORT-style** logic (adds camera-motion compensation — essential for a handheld phone) when the scene is crowded or the user pans. Density breakpoints from tracker guides are rules of thumb, not hard limits — tune on-device. ([Fora Soft](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/multi-object-tracking-deepsort-bytetrack-ocsort))

### Re-identification
**OSNet (small variant, ONNX, quantized)** — 2.2M parameters full-size, with 0.25-scale variants under 0.2M parameters; produces 512-D embeddings matched by cosine similarity, designed for matching people across camera views. Run it only on track confirmation and reacquisition events, not every frame. ([Qualcomm AI Hub](https://aihub.qualcomm.com/mobile/models/osnet), [Embedded.com on OSNet](https://www.embedded.com/tiny-neural-network-model-fits-in-edge-devices-for-re-identification-video-applications/), [STMicro model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/main/re_identification/osnet/README.md))

**Ledger retention:** retain **every counted `personId` and its usable matching information for the whole session** — the requirement is "never count the same person twice," so the ledger cannot forget people while the count is still open. Bound memory by limiting the number of appearance *samples stored per person* (e.g., a handful of best-quality embeddings), not the number of remembered people. If a hard resource limit ever forces evicting identities, mark deduplication coverage as incomplete in the UI rather than silently counting returning people as new. Keep unresolved/ambiguous identities in a separate bucket from the confirmed total.

### Very dense crowds (hundreds of people)
When people are no longer individually separable, switch modes: **crowd density estimation** models (lightweight ones like LCDnet) output a density map whose integral is the crowd total, and run in real time on embedded-class hardware. Density estimation cannot give identities, so use it as a separate "estimated visible total" mode — do not merge or add detector counts and density totals. ([LCDnet paper](https://link.springer.com/article/10.1007/s11554-023-01286-8), [real-time crowd counting for embedded systems](https://d-nb.info/138747684X/34))

---

## 5. Bright Light and Darkness

- **Darkness:** turn on the torch where supported (near-field benefit only) and rely on the main camera's ƒ/1.48 maximum aperture and sensor-shift stabilization — the main sensor is the low-light hardware. Detection accuracy will still drop at night; the right response is to lengthen track-confirmation requirements and mark the count "low confidence" rather than lowering thresholds (which causes flicker double-counts). A WebGPU pre-processing pass (CLAHE/histogram equalization) can help a marginal frame but cannot recover people the sensor never captured. Also: do not assume the browser stream gives you Night mode, HDR controls, or aperture control — verify what the target device actually exposes. ([Anyline torch support](https://documentation.anyline.com/web-sdk-component/latest/advanced-camera-api.html), [Apple specs](https://www.apple.com/iphone-18-pro/specs/))
- **Bright light / backlighting:** people become silhouettes. Do not assume the browser exposes the Camera app's Night mode, HDR, aperture, or exposure controls; check the capabilities actually available on the target device, and rely on the detector's robustness to silhouettes rather than any exposure hack. ([Dynamsoft focus/exposure limits](https://www.dynamsoft.com/codepool/camera-focus-control-on-web.html))

---

## 6. Auto-Zoom Strategy ("zoom if not detecting people")

The naive rule — "no person detected → zoom in" — is wrong for counting: zooming narrows the field of view and can *lose* people already in frame, and a lens switch changes all tracking coordinates (double-count risk). A safer ladder:

1. **No detections → check small-object coverage first:** re-run inference on a larger or tiled crop of the captured frame before touching the camera (digital zoom at the inference level — no stream change, no coordinate break).
2. **Still nothing → widen:** switch to the ultra-wide lens (larger field, more people per frame) rather than zooming in. For counting, coverage beats magnification.
3. **Distant single target → telephoto:** only when tracking is healthy and one far target matters. Pause counting events during the lens switch; treat all active tracks as uncertain for a bounded period; and re-verify with OSNet embeddings after the switch so returning detections re-match instead of re-counting. A focal-length ratio alone cannot reliably remap tracks between lenses — lens position, distortion, crop, and scene depth all differ — so this transition needs a registration/calibration step or an explicit uncertainty window.

Use adaptive capture resolution (raise it when the scene demands distant detection, lower it when the scene is close and GPU headroom is needed elsewhere) rather than always maximum. ([Dynamsoft zoom](https://www.dynamsoft.com/codepool/camera-zoom-control-on-web.html), [iOS lens enumeration](https://dominikschilling.de/notes/ios-access-all-back-cameras-mediadevices-api/))

---

## 7. Gender Classification: Recommend Dropping It

The cited literature is consistent that appearance-based gender classification is unreliable:

- Commercial automatic gender recognition had error rates up to **34.7% for darker-skinned women** versus 0.8% for lighter-skinned men in the Gender Shades findings, and appearance-based classifiers lost **18–60 percentage points** of accuracy when deployed outside their training distribution — sometimes dropping to chance. ([arXiv, algorithmic misgendering in AGR](https://arxiv.org/html/2506.02017v1), [PMC, pitfalls of gender recognition in unseen environments](https://pmc.ncbi.nlm.nih.gov/articles/PMC12252448/))
- Gender is an internal identity, not a visual property; models classify gender *expression* (clothing, hair), which fails for anyone non-conforming and encodes stereotype proxies. ([Scientific American, Can Gender Be Computed?](https://www.scientificamerican.com/article/can-gender-be-computed/), [Pew Research Center](https://www.pewresearch.org/internet/2019/09/05/the-challenges-of-using-machine-learning-to-identify-gender-in-images/))
- In this app's conditions — crowds, distance, low light, motion — accuracy would be worst exactly where it is needed.

These are historical benchmarks of other systems, not measurements of any model you would train — but they establish the failure pattern. **The recommendation is to omit camera-inferred gender entirely.** Relabeling it "perceived presentation" does not make it a reliable demographic measure, and no error band fixes a classifier that collapses toward chance outside its training distribution. If genuine gender demographics are a hard requirement, use voluntary self-reporting collected separately in the app — it fits the on-device, privacy-preserving design.

---

## 8. Recommended Architecture

```
Camera (one active lens, rear, adaptive resolution)
        │
        ▼
WebGL/WebGPU canvas ──► pre-process (CLAHE in low light, torch if available)
        │
        ▼
YOLO11n person/head detection (ONNX Runtime Web, WebGPU→WASM fallback,
        │                 adaptive resolution / tiled when people are small,
        │                 every 2nd–3rd frame)
        ▼
ByteTrack / BoT-SORT (JS) ── camera-motion compensation for pans
        │
        ├── track confirmed (N frames above threshold)
        │         └─► OSNet embedding ──► session ledger match?
        │                                  confident match → reuse personId (no recount)
        │                                  ambiguous → "possible duplicate" state, merge later
        │                                  no match → new personId, total++
        │
        └── count events: line-crossing (3-state validation)
                           or "entered region" per personId
        │
        ▼
Counting state: unique total, currently visible, flow counts (separate),
uncertainty indicator — all local, nothing leaves the device

Density fallback mode (very dense crowd):
LCDnet-class density map → estimated visible total (separate mode, no ID merge)
```

**Performance:** no desktop or third-party benchmark transfers to this pipeline on an iPhone 18 Pro — the budget (detection, tracking, re-ID at your chosen frame-skip) must be measured on-device, including sustained thermal behavior over long sessions (browser throttling on extended camera + GPU load is a real phenomenon; design adaptive frame-skip around it). Model downloads are small (megabytes, cached after first load) — measure your actual exported files.

---

## 9. Validation Plan

Measure these before trusting the counter, across lighting (day/backlit/night with torch) and densities (5/20/100+ people):

- **Count error** vs. ground truth (recorded video with manual counts)
- **Double-count rate** — same person re-entering frame; the key metric for this requirement
- **False-merge rate** — different people collapsed into one ID (undercounting); both error directions matter
- **ID switches** per minute (tracking stability under occlusion and pan)
- Missed crossings if using line-counting mode
- Latency (ms/frame), sustained FPS over 10+ minutes, memory, and thermal behavior on-device
- Verify on the target device that: each rear lens is selectable, zoom/torch constraints actually apply, and ONNX Runtime Web's WebGPU backend runs your exported models in Safari — none of these are guaranteed by specifications alone

Test on the actual device; iOS Safari's camera behavior cannot be simulated on desktop.

---

## 10. Bottom Line

Prototype a lightweight detector, a motion/appearance tracker, and a session-long identity ledger. YOLO11n detection, BoT-SORT-style tracking, OSNet re-identification, and ONNX Runtime Web are candidates requiring compatibility and accuracy validation — not a verified best-performing iPhone deployment. Count confirmed tracks, dedupe via embedding matching with explicit handling of ambiguous cases, and use a density-estimation mode when crowds get too thick to track individuals. Use the ultra-wide lens for coverage, the telephoto sparingly (with an uncertainty window around lens switches), torch for nearby darkness, and digital crop-zoom before any lens switch. Skip gender classification — it is the one requested feature the evidence says will not work reliably; if gender demographics are truly required, use voluntary self-reporting collected separately. And present the unique total as a best estimate with an uncertainty indicator: a moving phone cannot guarantee a perfect unique count, but this architecture is the strongest way to reduce repeat counting within the platform's limits.
