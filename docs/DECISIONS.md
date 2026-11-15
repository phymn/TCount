# Decisions: TCount

Each entry: date (ET), decision, why. "Date uncertain" means the decision is in the code/README but the exact day isn't recorded.

| Date | Decision | Why |
|---|---|---|
| 2026-10-19 | Everything runs in the browser on the device; models are bundled and run with ONNX Runtime Web (WebGPU first, WASM fallback). | Privacy (no frames leave the phone) and offline use. |
| 2026-10-19 | The crowd app does not estimate gender. The gender counter (`index.html`) is kept only for history. | Gender estimates are unreliable and biased. |
| 2026-10-19 | Files over 5 MB (ORT WASM, DM-Count model) are not committed; `tools/fetch_assets.sh` and a release asset are used instead. | GitHub file-size limits; keeps the repo small. |
| 2026-10-19 | Unique count never includes "possible duplicates" silently; the UI shows a range and lets the user merge or keep. | Honest counts instead of a hidden guess. |
| 2026-11-03 | Cache the runtime/models on the device and defer the dense model until it is turned on. | Faster second launch, works offline, saves data and battery. |
| Date uncertain (by 2026-11-07) | Simple mode footer shows the version only; Superuser shows details. | Standing rule: every build shows its version; Simple stays clean. |
| Date uncertain (Nov 2026) | Pose/hand-rejection weights are not shipped until their license is clear; the default gate is size-only. | License not resolved. |
| Standing | No merge or production deploy without Jesus's go-ahead; previous deployments are kept. | Owner approves releases. |
| 2026-11-14 | Doc comments go on a branch, not `main`, while PRs #4/#5/#9 are open. | Avoid merge conflicts in `crowd/app.js`. |
