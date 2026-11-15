# Changelog: TCount

Dates are ET. Sources: git history, PR list, README and project status notes. Anything not on `main` is marked.

## 2026-11-14 — Documentation pass
- Added `CHANGELOG.md` and `docs/DECISIONS.md`.
- Doc comments on every named function/class in `crowd/`, the original counters and `tools/` are on branch `docs/comments-20261114` (not merged into `main`, so open PRs #4, #5 and #9 don't get conflicts). Comments only. Every documented function/type got a comment saying what it does and why it calls each API. Proven behavior-identical: JavaScript token streams (comments excluded) and non-script HTML text are byte-equal to the previous version; Python ASTs are equal; AppleScript changes are add-only `--` lines placed outside strings and `(* *)` comments; Swift text is equal after stripping comments/whitespace.

## Not on main (open PRs, test candidates)
- **crowd-1.3-ui.3** (PR #9, about 2026-11-07): counting-stall reliability (serialized detector calls, timeouts, resume on tab return). Deployed only as a test candidate; not merged.
- **crowd-1.3-ui / .1 / .2** (about 2026-11-07): Simple/Superuser UI, EN/ES, version-only footer in Simple mode. Test candidates.
- **crowd-1.3** (PR #5): pose-based filter adapter drafted; model weights not shipped (license unresolved); real-scene/iPhone tests missing.
- PR #4: alternative cache change, waiting for a decision.

## 2026-11-03 — crowd-1.2 (PR #3, `fce2108`)
- Crowd runtime cached on the device (Cache Storage + service worker); dense model deferred until enabled.

## 2026-10-20 — `50ba3ac`
- Added `docs/faceid-lidar.md` (Face ID and iPhone LiDAR notes).

## 2026-10-19 — v1.0 (`71ada61`)
- TCount Crowd: YOLOX person detector, ByteTrack with camera-motion compensation, OSNet re-ID, possible-duplicates range, optional DM-Count dense estimate, lens switching, low-light handling, CSV export, self-test.
- Original counters kept (`index-nogender.html`, `index.html`, `legacy/`).

## 2026-10-18 — Initial commit (`20829a1`)
