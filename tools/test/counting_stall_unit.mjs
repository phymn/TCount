#!/usr/bin/env node
/**
 * SYNTHETIC tests for crowd-1.3-ui.3 counting-stall reliability guards.
 * Does not load ONNX — inspects source + exercises timeout/serialization helpers.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '../..');
const app = readFileSync(join(root, 'crowd/app.js'), 'utf8');
const sw = readFileSync(join(root, 'crowd/sw.js'), 'utf8');
const results = [];
function check(name, fn) {
  try { fn(); results.push({ name, kind: 'SYNTHETIC', ok: true }); }
  catch (e) { results.push({ name, kind: 'SYNTHETIC', ok: false, err: String(e.message || e) }); }
}
async function checkAsync(name, fn) {
  try { await fn(); results.push({ name, kind: 'SYNTHETIC', ok: true }); }
  catch (e) { results.push({ name, kind: 'SYNTHETIC', ok: false, err: String(e.message || e) }); }
}

check('version crowd-1.3-ui.3', () => {
  assert.match(app, /const VERSION = 'crowd-1\.3-ui\.3'/);
  assert.match(sw, /const VERSION = 'crowd-1\.3-ui\.3'/);
});

check('detector runs are serialized (_exclusive)', () => {
  assert.match(app, /_exclusive\(fn\)/);
  assert.match(app, /async detect\(regions, size\) \{\s*return this\._exclusive/);
});

check('no fire-and-forget tiledEstimate from autoPolicies', () => {
  const auto = app.match(/function autoPolicies[\s\S]*?\n\}/);
  assert.ok(auto, 'autoPolicies missing');
  assert.doesNotMatch(auto[0], /tiledEstimate\(\)/);
  assert.match(auto[0], /wantTiledEst\s*=\s*true/);
  assert.match(app, /if \(st\.wantTiledEst[\s\S]*?await tiledEstimate\(\)/);
});

check('tile gate uses lastFrameMs (full-frame time)', () => {
  assert.match(app, /lastFrameMs/);
  assert.match(app, /frameMs > 400/);
  assert.doesNotMatch(app, /detector\.lastMs > 300 && !dense\.active/);
});

check('session.run wrapped in withTimeout for detect + reid + step', () => {
  assert.match(app, /withTimeout\(this\.session\.run\(\{ images:/);
  assert.match(app, /withTimeout\(this\.session\.run\(\{ input:/);
  assert.match(app, /withTimeout\(step\(\), 25000/);
});

check('visibility resume restarts loop without ledger reset', () => {
  assert.match(app, /visibilitychange/);
  const block = app.match(/visibilitychange[\s\S]{0,280}/);
  assert.ok(block);
  assert.match(block[0], /startLoop\(\)/);
  assert.doesNotMatch(block[0], /ledger\.reset/);
});

check('stall health banner when lastStepOk stale', () => {
  assert.match(app, /banner\('stall'/);
  assert.match(app, /lastStepOk && now\(\) - st\.lastStepOk > 5000/);
});

check('no pose / diagnostic screenshot path in source', () => {
  assert.doesNotMatch(app, /rtmpose|PoseVerifier|bodyOk/);
  assert.doesNotMatch(app, /counting-stall-20261107/);
});

function withTimeout(p, ms, what) {
  let t;
  return Promise.race([
    p,
    new Promise((_, rej) => { t = setTimeout(() => rej(new Error(what + ' timed out after ' + ms / 1000 + ' s')), ms); }),
  ]).finally(() => clearTimeout(t));
}

await checkAsync('withTimeout rejects hung promise', async () => {
  const hung = new Promise(() => {});
  await assert.rejects(() => withTimeout(hung, 40, 'test'), /timed out/);
});

await checkAsync('exclusive chain serializes overlapping work', async () => {
  let active = 0, max = 0;
  let chain = Promise.resolve();
  function exclusive(fn) {
    const run = chain.then(fn, fn);
    chain = run.catch(() => {});
    return run;
  }
  async function job(ms) {
    active++; max = Math.max(max, active);
    await new Promise(r => setTimeout(r, ms));
    active--;
  }
  await Promise.all([exclusive(() => job(35)), exclusive(() => job(35)), exclusive(() => job(35))]);
  assert.equal(max, 1);
});

const failed = results.filter(r => !r.ok);
console.log(JSON.stringify({ kind: 'SYNTHETIC', camera: 'none', model: 'none', results, failed: failed.length }, null, 2));
process.exit(failed.length ? 1 : 0);
