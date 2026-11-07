#!/usr/bin/env node
/** SYNTHETIC unit tests for crowd/rtmpose.js (no ONNX). */
import fs from 'fs';
import path from 'path';
import vm from 'vm';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const src = fs.readFileSync(path.join(__dirname, '../../crowd/rtmpose.js'), 'utf8');
const sandbox = { self: {}, console };
vm.createContext(sandbox);
vm.runInContext(src, sandbox);
const RT = sandbox.self.TCountRTMPose;
if (!RT) { console.error('TCountRTMPose missing'); process.exit(1); }

const cases = [];
function check(name, cond, detail) {
  cases.push({ name, pass: !!cond, detail });
}

// SimCC decode peak
const wx = 384, wy = 512, K = 17;
const simccX = new Float32Array(K * wx);
const simccY = new Float32Array(K * wy);
for (let k = 0; k < K; k++) {
  simccX[k * wx + 100] = 2;
  simccY[k * wy + 200] = 2;
}
const meta = { cx: 50, cy: 60, sw: 96, sh: 128, w: 192, h: 256, wx, wy };
const dec = RT.decodeSimcc(simccX, simccY, meta);
check('decode scores', Math.abs(dec.scores[0] - 2) < 1e-6, dec.scores[0]);
check('decode xy finite', Number.isFinite(dec.keypoints[0]) && Number.isFinite(dec.keypoints[1]), [dec.keypoints[0], dec.keypoints[1]]);

// Hand reject
{
  const kpts = new Float32Array(34).fill(-1);
  const scores = new Float32Array(17);
  for (const i of [7, 8, 9, 10]) { kpts[i * 2] = 10 + i; kpts[i * 2 + 1] = 10; scores[i] = 0.9; }
  const v = RT.acceptBodyStructure(kpts, scores, [0, 0, 40, 40]);
  check('reject limb-only', v.ok === false && v.reason === 'limb-only', v);
}
// Person accept
{
  const kpts = new Float32Array(34).fill(-1);
  const scores = new Float32Array(17);
  const pts = { 0: [50, 20], 5: [30, 50], 6: [70, 50], 11: [35, 110], 12: [65, 110] };
  for (const [i, [x, y]] of Object.entries(pts)) {
    const k = +i; kpts[k * 2] = x; kpts[k * 2 + 1] = y; scores[k] = 0.7;
  }
  const v = RT.acceptBodyStructure(kpts, scores, [10, 10, 80, 140]);
  check('accept torso', v.ok === true, v);
}

// Affine invertibility: center maps near output center
{
  const mat = RT.getWarpMatrix(100, 200, 96, 128, 0, 192, 256);
  const [dx, dy] = RT.applyAffine(mat, 100, 200);
  check('center→out', Math.abs(dx - 96) < 1e-3 && Math.abs(dy - 128) < 1e-3, [dx, dy]);
}

const failed = cases.filter(c => !c.pass);
console.log(JSON.stringify({ kind: 'SYNTHETIC', cases, pass: failed.length === 0 }, null, 2));
process.exit(failed.length ? 1 : 0);
