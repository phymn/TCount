/* RTMPose top-down verifier for onnxruntime-web.
   Preprocess + SimCC decode adapted from Tau-J/rtmlib (Apache-2.0), with
   ImageNet normalize using to_rgb=true (matches the OpenMMLab ONNX export
   pipeline.json). rtmlib's OpenCV path omits an explicit BGR→RGB step; the
   canvas path here is already RGB, so mean/std are applied on RGB channels.
   Does not include or download weights — see LICENSES.txt / docs. */
'use strict';
(function (root) {
  const COCO17 = [
    'nose', 'LEye', 'REye', 'LEar', 'REar',
    'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist',
    'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle'
  ];
  // ImageNet RGB mean/std used by the Body7 ONNX export (pipeline Normalize to_rgb=true).
  const MEAN = [123.675, 116.28, 103.53];
  const STD = [58.395, 57.12, 57.375];
  const DEFAULT_SIZE = { w: 192, h: 256 }; // model_input_size (w,h) as in rtmlib
  const SIMCC_SPLIT = 2.0;

  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

  /** bbox xyxy + padding → center, scale (rtmlib bbox_xyxy2cs). */
  function bboxXyxy2cs(x1, y1, x2, y2, padding) {
    const cx = (x1 + x2) * 0.5, cy = (y1 + y2) * 0.5;
    const sw = (x2 - x1) * padding, sh = (y2 - y1) * padding;
    return { cx, cy, sw, sh };
  }

  /** Fix scale to model aspect ratio (rtmlib top_down_affine). */
  function fixAspect(sw, sh, w, h) {
    const ar = w / h;
    if (sw > sh * ar) return { sw, sh: sw / ar };
    return { sw: sh * ar, sh };
  }

  function rotatePoint(x, y, ang) {
    const sn = Math.sin(ang), cs = Math.cos(ang);
    return [cs * x - sn * y, sn * x + cs * y];
  }
  function thirdPoint(ax, ay, bx, by) {
    const dx = ax - bx, dy = ay - by;
    return [bx - dy, by + dx];
  }

  /** 2×3 affine src→dst matching cv2.getAffineTransform used by rtmlib get_warp_matrix. */
  function getWarpMatrix(cx, cy, sw, sh, rotDeg, outW, outH) {
    const rot = rotDeg * Math.PI / 180;
    const srcDir = rotatePoint(0, sw * -0.5, rot);
    const dstDir = [0, outW * -0.5];
    const src0 = [cx, cy];
    const src1 = [cx + srcDir[0], cy + srcDir[1]];
    const src2 = thirdPoint(src0[0], src0[1], src1[0], src1[1]);
    const dst0 = [outW * 0.5, outH * 0.5];
    const dst1 = [dst0[0] + dstDir[0], dst0[1] + dstDir[1]];
    const dst2 = thirdPoint(dst0[0], dst0[1], dst1[0], dst1[1]);
    // Solve affine: dst = A * src (homogeneous), using three point pairs.
    return affineFrom3(src0, src1, src2, dst0, dst1, dst2);
  }

  function affineFrom3(s0, s1, s2, d0, d1, d2) {
    // [x y 1 0 0 0; 0 0 0 x y 1] * [a b tx c d ty]^T = [x' y']
    const M = [
      [s0[0], s0[1], 1, 0, 0, 0], [0, 0, 0, s0[0], s0[1], 1],
      [s1[0], s1[1], 1, 0, 0, 0], [0, 0, 0, s1[0], s1[1], 1],
      [s2[0], s2[1], 1, 0, 0, 0], [0, 0, 0, s2[0], s2[1], 1]
    ];
    const b = [d0[0], d0[1], d1[0], d1[1], d2[0], d2[1]];
    const x = solve6(M, b);
    return [x[0], x[1], x[2], x[3], x[4], x[5]]; // a,b,tx,c,d,ty
  }

  function solve6(A, b) {
    // Gaussian elimination with partial pivot (6×6).
    const n = 6;
    const M = A.map((row, i) => row.concat([b[i]]));
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
      if (Math.abs(M[piv][col]) < 1e-12) throw new Error('singular affine');
      if (piv !== col) { const t = M[col]; M[col] = M[piv]; M[piv] = t; }
      const div = M[col][col];
      for (let c = col; c <= n; c++) M[col][c] /= div;
      for (let r = 0; r < n; r++) if (r !== col) {
        const f = M[r][col];
        for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
      }
    }
    return M.map(row => row[n]);
  }

  /** Apply 2×3 affine to one point. */
  function applyAffine(mat, x, y) {
    return [mat[0] * x + mat[1] * y + mat[2], mat[3] * x + mat[4] * y + mat[5]];
  }

  /**
   * Sample RGB canvas/ImageData into CHW float32 tensor [1,3,H,W] with rtmlib affine + ImageNet norm.
   * box = [x,y,w,h] in image coords. Returns { tensor, center, scale, w, h }.
   */
  function preprocessFromCanvas(source, box, opts) {
    const w = (opts && opts.w) || DEFAULT_SIZE.w;
    const h = (opts && opts.h) || DEFAULT_SIZE.h;
    const padding = (opts && opts.padding) != null ? opts.padding : 1.25;
    const x1 = box[0], y1 = box[1], x2 = box[0] + box[2], y2 = box[1] + box[3];
    let { cx, cy, sw, sh } = bboxXyxy2cs(x1, y1, x2, y2, padding);
    ({ sw, sh } = fixAspect(sw, sh, w, h));
    const mat = getWarpMatrix(cx, cy, sw, sh, 0, w, h);
    // Inverse map each output pixel → source (bilinear).
    // dst = A * src ⇒ src = A^{-1} * dst. Build inverse from three pairs swapped.
    const inv = affineFrom3(
      [w * 0.5, h * 0.5], [w * 0.5, h * 0.5 - w * 0.5], thirdPoint(w * 0.5, h * 0.5, w * 0.5, h * 0.5 - w * 0.5),
      [cx, cy], [cx, cy - sw * 0.5], thirdPoint(cx, cy, cx, cy - sw * 0.5)
    );
    // Prefer canvas drawImage + getImageData path when a CanvasRenderingContext is easier;
    // pure bilinear keeps this module free of DOM requirement for unit tests.
    const srcW = source.width, srcH = source.height;
    let rgba;
    if (source.getContext) {
      const ctx = source.getContext('2d', { willReadFrequently: true });
      rgba = ctx.getImageData(0, 0, srcW, srcH).data;
    } else if (source.data) {
      rgba = source.data;
    } else {
      throw new Error('preprocessFromCanvas needs canvas or ImageData');
    }
    const out = new Float32Array(3 * h * w);
    const n = h * w;
    for (let oy = 0; oy < h; oy++) {
      for (let ox = 0; ox < w; ox++) {
        const [sx, sy] = applyAffine(inv, ox, oy);
        const r = sampleBilinearRGB(rgba, srcW, srcH, sx, sy);
        const i = oy * w + ox;
        out[i] = (r[0] - MEAN[0]) / STD[0];
        out[n + i] = (r[1] - MEAN[1]) / STD[1];
        out[2 * n + i] = (r[2] - MEAN[2]) / STD[2];
      }
    }
    return { tensor: out, dims: [1, 3, h, w], cx, cy, sw, sh, w, h };
  }

  /** Faster path: draw affine via canvas (same math) when DOM canvas helpers are provided. */
  function preprocessDraw(frameCanvas, box, prepCanvas, prepCtx, opts) {
    const w = (opts && opts.w) || DEFAULT_SIZE.w;
    const h = (opts && opts.h) || DEFAULT_SIZE.h;
    const padding = (opts && opts.padding) != null ? opts.padding : 1.25;
    const x1 = box[0], y1 = box[1], x2 = box[0] + box[2], y2 = box[1] + box[3];
    let { cx, cy, sw, sh } = bboxXyxy2cs(x1, y1, x2, y2, padding);
    ({ sw, sh } = fixAspect(sw, sh, w, h));
    if (prepCanvas.width !== w || prepCanvas.height !== h) { prepCanvas.width = w; prepCanvas.height = h; }
    // Build canvas transform: map source bbox crop into (w,h) with aspect-fixed scale.
    // Equivalent warp: translate so center→origin, scale, then to output center.
    prepCtx.save();
    prepCtx.setTransform(1, 0, 0, 1, 0, 0);
    prepCtx.clearRect(0, 0, w, h);
    const scaleX = w / sw, scaleY = h / sh;
    prepCtx.setTransform(scaleX, 0, 0, scaleY, w / 2 - cx * scaleX, h / 2 - cy * scaleY);
    prepCtx.drawImage(frameCanvas, 0, 0);
    prepCtx.restore();
    const px = prepCtx.getImageData(0, 0, w, h).data;
    const out = new Float32Array(3 * h * w);
    const n = h * w;
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      out[i] = (px[j] - MEAN[0]) / STD[0];
      out[n + i] = (px[j + 1] - MEAN[1]) / STD[1];
      out[2 * n + i] = (px[j + 2] - MEAN[2]) / STD[2];
    }
    return { tensor: out, dims: [1, 3, h, w], cx, cy, sw, sh, w, h };
  }

  function sampleBilinearRGB(rgba, W, H, x, y) {
    if (x < 0 || y < 0 || x >= W - 1 || y >= H - 1) {
      // border: clamp (approx INTER_LINEAR edge)
      const cx = clamp(x, 0, W - 1), cy = clamp(y, 0, H - 1);
      const ix = Math.floor(cx), iy = Math.floor(cy);
      const j = (iy * W + ix) * 4;
      return [rgba[j], rgba[j + 1], rgba[j + 2]];
    }
    const x0 = Math.floor(x), y0 = Math.floor(y);
    const x1 = x0 + 1, y1 = y0 + 1;
    const ax = x - x0, ay = y - y0;
    const i00 = (y0 * W + x0) * 4, i01 = (y0 * W + x1) * 4, i10 = (y1 * W + x0) * 4, i11 = (y1 * W + x1) * 4;
    const r = [];
    for (let c = 0; c < 3; c++) {
      const v = (1 - ax) * (1 - ay) * rgba[i00 + c] + ax * (1 - ay) * rgba[i01 + c]
        + (1 - ax) * ay * rgba[i10 + c] + ax * ay * rgba[i11 + c];
      r.push(v);
    }
    return r;
  }

  /** SimCC decode — rtmlib get_simcc_maximum (mean of per-axis max) + rescale. */
  function decodeSimcc(simccX, simccY, meta, splitRatio) {
    const sr = splitRatio == null ? SIMCC_SPLIT : splitRatio;
    const K = simccX.length / (simccX.length > 0 ? (meta && meta.wx) || (simccX.length / 17) : 1);
    // Accept Float32Array shaped as flat [K*Wx] or nested via dims.
    const wx = (meta && meta.wx) || Math.round(simccX.length / 17);
    const wy = (meta && meta.wy) || Math.round(simccY.length / 17);
    const nK = Math.round(simccX.length / wx);
    const locs = new Float32Array(nK * 2);
    const scores = new Float32Array(nK);
    for (let k = 0; k < nK; k++) {
      let mx = -Infinity, my = -Infinity, ix = 0, iy = 0;
      const ox = k * wx, oy = k * wy;
      for (let i = 0; i < wx; i++) { const v = simccX[ox + i]; if (v > mx) { mx = v; ix = i; } }
      for (let i = 0; i < wy; i++) { const v = simccY[oy + i]; if (v > my) { my = v; iy = i; } }
      const sc = 0.5 * (mx + my);
      scores[k] = sc;
      if (sc <= 0) { locs[k * 2] = -1; locs[k * 2 + 1] = -1; }
      else { locs[k * 2] = ix; locs[k * 2 + 1] = iy; }
    }
    const { cx, cy, sw, sh, w, h } = meta;
    const kpts = new Float32Array(nK * 2);
    for (let k = 0; k < nK; k++) {
      if (locs[k * 2] < 0) { kpts[k * 2] = -1; kpts[k * 2 + 1] = -1; continue; }
      const px = locs[k * 2] / sr;
      const py = locs[k * 2 + 1] / sr;
      kpts[k * 2] = px / w * sw + cx - sw / 2;
      kpts[k * 2 + 1] = py / h * sh + cy - sh / 2;
    }
    return { keypoints: kpts, scores, names: COCO17.slice(0, nK) };
  }

  /**
   * Body-structure acceptance for person-vs-hand/arm FPs.
   * Deliberately NOT: "any 2 of nose/shoulders/hips" alone, fixed upright aspect,
   * or edge rejection. Seated / partly occluded people should still pass when a
   * torso anchor exists and keypoints span the box (not a tiny wrist cluster).
   */
  function acceptBodyStructure(keypoints, scores, box, opts) {
    const thr = (opts && opts.kptThr) != null ? opts.kptThr : 0.45;
    const nK = scores.length;
    const vis = [];
    for (let k = 0; k < nK; k++) if (scores[k] > thr && keypoints[k * 2] >= 0) vis.push(k);
    const has = i => vis.indexOf(i) >= 0;
    const head = [0, 1, 2, 3, 4].some(has);
    const Lsh = has(5), Rsh = has(6), Lhp = has(11), Rhp = has(12);
    const shoulder = Lsh || Rsh;
    const hip = Lhp || Rhp;
    const elbow = has(7) || has(8);
    const wrist = has(9) || has(10);
    const knee = has(13) || has(14);
    const ankle = has(15) || has(16);
    const torsoAnchor = (Lsh && Rsh) || (Lhp && Rhp) || (shoulder && hip) || (shoulder && head) || (hip && head);
    // Hand/arm-only signature: wrist/elbow without torso or head.
    const limbOnly = (wrist || elbow) && !shoulder && !hip && !head;
    if (limbOnly) return { ok: false, reason: 'limb-only', vis: vis.length, torsoAnchor: false };

    // Spatial span inside the detection box — isolated hands cluster tightly.
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const k of vis) {
      const x = keypoints[k * 2], y = keypoints[k * 2 + 1];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
    const bw = Math.max(1, box[2]), bh = Math.max(1, box[3]);
    const spanX = vis.length ? (maxX - minX) / bw : 0;
    const spanY = vis.length ? (maxY - minY) / bh : 0;
    const spanOk = spanY >= 0.22 || spanX >= 0.28 || (spanY >= 0.15 && spanX >= 0.15);
    // Enough evidence: torso + some other signal, or ≥4 visible kpts with torso.
    const extras = [head, elbow, wrist, knee, ankle].filter(Boolean).length;
    const evidenceOk = torsoAnchor && (vis.length >= 4 || extras >= 1 || (shoulder && hip));
    const ok = !!(torsoAnchor && spanOk && evidenceOk && vis.length >= 3);
    return {
      ok, reason: ok ? 'body' : (!torsoAnchor ? 'no-torso' : !spanOk ? 'tight-cluster' : 'weak-evidence'),
      vis: vis.length, spanX, spanY, torsoAnchor, head, shoulder, hip, limbOnly
    };
  }

  root.TCountRTMPose = {
    COCO17, MEAN, STD, DEFAULT_SIZE, SIMCC_SPLIT,
    bboxXyxy2cs, fixAspect, getWarpMatrix, preprocessFromCanvas, preprocessDraw,
    decodeSimcc, acceptBodyStructure, applyAffine, sampleBilinearRGB
  };
})(typeof self !== 'undefined' ? self : globalThis);
