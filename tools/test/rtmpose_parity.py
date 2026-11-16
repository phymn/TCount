#!/usr/bin/env python3
"""Parity checks for RTMPose SimCC decode vs rtmlib, plus body-accept fixtures.

Labels:
  REAL_MODEL — runs end2end.onnx if ALLOW_UNCLEARED_POSE_WEIGHTS=1 and file exists
  SYNTHETIC  — no weights; pure numeric / structure tests
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ONNX_CANDIDATES = [
    ROOT / 'crowd' / 'models' / 'rtmpose_t_body7_256x192.onnx',
    Path('/tmp/rtmpose-weights/20230831/rtmpose_onnx/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504/end2end.onnx'),
]

# --- SimCC decode (rtmlib get_simcc_maximum + rescale) ---
def get_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray):
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)
    vals = 0.5 * (max_val_x + max_val_y)
    locs[vals <= 0.] = -1
    return locs.reshape(N, K, 2), vals.reshape(N, K)


def decode_js_style(simcc_x_flat, simcc_y_flat, meta, split=2.0):
    """Mirror crowd/rtmpose.js decodeSimcc for N=1."""
    wx = meta['wx']; wy = meta['wy']; nK = 17
    locs = np.zeros((nK, 2), np.float32)
    scores = np.zeros(nK, np.float32)
    for k in range(nK):
        rowx = simcc_x_flat[k * wx:(k + 1) * wx]
        rowy = simcc_y_flat[k * wy:(k + 1) * wy]
        ix = int(np.argmax(rowx)); iy = int(np.argmax(rowy))
        sc = 0.5 * (rowx[ix] + rowy[iy])
        scores[k] = sc
        if sc <= 0:
            locs[k] = (-1, -1)
        else:
            locs[k] = (ix, iy)
    w, h = meta['w'], meta['h']
    cx, cy, sw, sh = meta['cx'], meta['cy'], meta['sw'], meta['sh']
    kpts = np.zeros((nK, 2), np.float32)
    for k in range(nK):
        if locs[k, 0] < 0:
            kpts[k] = (-1, -1); continue
        px = locs[k, 0] / split; py = locs[k, 1] / split
        kpts[k, 0] = px / w * sw + cx - sw / 2
        kpts[k, 1] = py / h * sh + cy - sh / 2
    return kpts, scores


def accept_body(keypoints, scores, box, thr=0.45):
    """Mirror acceptBodyStructure (subset for tests)."""
    vis = [k for k in range(len(scores)) if scores[k] > thr and keypoints[k, 0] >= 0]
    def has(i): return i in vis
    head = any(has(i) for i in range(5))
    Lsh, Rsh, Lhp, Rhp = has(5), has(6), has(11), has(12)
    shoulder, hip = Lsh or Rsh, Lhp or Rhp
    elbow = has(7) or has(8); wrist = has(9) or has(10)
    knee = has(13) or has(14); ankle = has(15) or has(16)
    torso = (Lsh and Rsh) or (Lhp and Rhp) or (shoulder and hip) or (shoulder and head) or (hip and head)
    limb_only = (wrist or elbow) and not shoulder and not hip and not head
    if limb_only:
        return False, 'limb-only'
    if not vis:
        return False, 'empty'
    xs = [keypoints[k, 0] for k in vis]; ys = [keypoints[k, 1] for k in vis]
    bw, bh = max(1, box[2]), max(1, box[3])
    span_x = (max(xs) - min(xs)) / bw; span_y = (max(ys) - min(ys)) / bh
    span_ok = span_y >= 0.22 or span_x >= 0.28 or (span_y >= 0.15 and span_x >= 0.15)
    extras = sum([head, elbow, wrist, knee, ankle])
    evidence = torso and (len(vis) >= 4 or extras >= 1 or (shoulder and hip))
    ok = bool(torso and span_ok and evidence and len(vis) >= 3)
    return ok, ('body' if ok else 'reject')


def test_simcc_parity():
    rng = np.random.default_rng(0)
    simcc_x = rng.normal(0, 0.2, (1, 17, 384)).astype(np.float32)
    simcc_y = rng.normal(0, 0.2, (1, 17, 512)).astype(np.float32)
    # ensure some positive peaks
    for k in range(17):
        simcc_x[0, k, 40 + k] = 1.5
        simcc_y[0, k, 80 + k] = 1.2
    meta = dict(cx=100, cy=200, sw=80, sh=160, w=192, h=256, wx=384, wy=512)
    locs, vals = get_simcc_maximum(simcc_x, simcc_y)
    kpts_ref = locs[0] / 2.0
    kpts_ref = kpts_ref / np.array([192, 256]) * np.array([meta['sw'], meta['sh']]) + np.array([meta['cx'], meta['cy']]) - np.array([meta['sw'], meta['sh']]) / 2
    kpts_js, scores_js = decode_js_style(simcc_x.reshape(-1), simcc_y.reshape(-1), meta)
    err = np.max(np.abs(kpts_ref - kpts_js))
    sc_err = np.max(np.abs(vals[0] - scores_js))
    assert err < 1e-4, err
    assert sc_err < 1e-5, sc_err
    return {'test': 'simcc_parity', 'kind': 'SYNTHETIC', 'max_xy_err': float(err), 'max_score_err': float(sc_err), 'pass': True}


def test_hand_vs_person_structure():
    results = []
    # Isolated hand cluster (wrists only, tight) — expect reject
    kpts = np.full((17, 2), -1.0, np.float32)
    scores = np.zeros(17, np.float32)
    for i, (x, y) in [(9, (50, 50)), (10, (55, 52)), (7, (48, 45)), (8, (57, 46))]:
        kpts[i] = (x, y); scores[i] = 0.8
    box = [40, 40, 30, 30]
    ok, why = accept_body(kpts, scores, box)
    results.append({'case': 'isolated_hand_cluster', 'expected': False, 'actual': ok, 'why': why, 'pass': ok is False})

    # Standing person with shoulders+hips+head — expect accept
    kpts = np.full((17, 2), -1.0, np.float32); scores = np.zeros(17, np.float32)
    layout = {0: (100, 40), 5: (80, 80), 6: (120, 80), 11: (85, 160), 12: (115, 160), 13: (88, 220), 14: (112, 220)}
    for i, (x, y) in layout.items():
        kpts[i] = (x, y); scores[i] = 0.7
    box = [60, 20, 80, 240]
    ok, why = accept_body(kpts, scores, box)
    results.append({'case': 'standing_torso', 'expected': True, 'actual': ok, 'why': why, 'pass': ok is True})

    # Seated / lower body occluded: head+shoulders only spanning box — expect accept
    kpts = np.full((17, 2), -1.0, np.float32); scores = np.zeros(17, np.float32)
    for i, (x, y) in {0: (100, 50), 5: (70, 90), 6: (130, 90), 7: (60, 130), 8: (140, 130)}.items():
        kpts[i] = (x, y); scores[i] = 0.65
    box = [50, 30, 100, 150]
    ok, why = accept_body(kpts, scores, box)
    results.append({'case': 'seated_or_occluded_upper', 'expected': True, 'actual': ok, 'why': why, 'pass': ok is True})

    # Wide aspect (sideways / unusual camera) with torso — must NOT be rejected for aspect
    kpts = np.full((17, 2), -1.0, np.float32); scores = np.zeros(17, np.float32)
    for i, (x, y) in {0: (160, 40), 5: (100, 50), 6: (220, 50), 11: (110, 80), 12: (210, 80)}.items():
        kpts[i] = (x, y); scores[i] = 0.7
    box = [80, 20, 180, 90]  # wide box; no upright gate
    ok, why = accept_body(kpts, scores, box)
    results.append({'case': 'wide_camera_torso', 'expected': True, 'actual': ok, 'why': why, 'pass': ok is True})

    return {'test': 'body_structure_fixtures', 'kind': 'SYNTHETIC', 'cases': results, 'pass': all(c['pass'] for c in results)}


def test_real_model_speed():
    onnx = next((p for p in ONNX_CANDIDATES if p.is_file()), None)
    allow = os.environ.get('ALLOW_UNCLEARED_POSE_WEIGHTS') == '1'
    if not onnx:
        return {'test': 'real_model_speed', 'kind': 'MISSING_FIXTURE', 'pass': None,
                'note': 'No local RTMPose ONNX; weights not shipped (license blocker).'}
    if not allow:
        return {'test': 'real_model_speed', 'kind': 'REAL_MODEL_PRESENT_BUT_GATED', 'pass': None,
                'path': str(onnx), 'bytes': onnx.stat().st_size,
                'note': 'Set ALLOW_UNCLEARED_POSE_WEIGHTS=1 to run timed inference (local only).'}
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider'])
    x = np.random.randn(1, 3, 256, 192).astype(np.float32)
    # warmup
    for _ in range(2):
        sess.run(None, {'input': x})
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        outs = sess.run(None, {'input': x})
        times.append((time.perf_counter() - t0) * 1000)
    med = float(np.median(times))
    return {
        'test': 'real_model_speed', 'kind': 'REAL_MODEL', 'pass': True,
        'path': str(onnx), 'bytes': onnx.stat().st_size,
        'provider': 'CPUExecutionProvider',
        'median_ms': round(med, 2), 'p90_ms': round(float(np.percentile(times, 90)), 2),
        'input': [1, 3, 256, 192], 'outputs': [tuple(o.shape) for o in outs],
        'note': 'Uncleared Body7 weights — local engineering only; not for redistribution.',
    }


def main():
    out = [test_simcc_parity(), test_hand_vs_person_structure(), test_real_model_speed()]
    print(json.dumps(out, indent=2))
    hard = [r for r in out if r.get('pass') is False]
    sys.exit(1 if hard else 0)


if __name__ == '__main__':
    main()
