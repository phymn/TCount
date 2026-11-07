#!/usr/bin/env node
/**
 * SYNTHETIC unit tests for crowd-1.3-ui acceptance helper + Done/mode semantics.
 * Does not load ONNX/camera — labeled SIMULATED.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = join(dirname(fileURLToPath(import.meta.url)), '../..');
const app = readFileSync(join(root, 'crowd/app.js'), 'utf8');
const html = readFileSync(join(root, 'crowd/index.html'), 'utf8');
const sw = readFileSync(join(root, 'crowd/sw.js'), 'utf8');

const results = [];
function check(name, fn) {
  try { fn(); results.push({ name, ok: true }); }
  catch (e) { results.push({ name, ok: false, err: String(e.message || e) }); }
}

check('version is crowd-1.3-ui.2 in app+sw', () => {
  assert.match(app, /const VERSION = 'crowd-1\.3-ui\.2'/);
  assert.match(sw, /const VERSION = 'crowd-1\.3-ui\.2'/);
});

check('footer is version-only (no CHANGE blurb)', () => {
  assert.match(app, /el\.textContent = VERSION/);
  assert.doesNotMatch(app, /build-footer[\s\S]{0,120}CHANGE_EN/);
  assert.match(html, /id="build-notes"/); // Superuser settings holds notes
  assert.match(app, /\$\('build-notes'\)/);
});

check('super-only CSS does not force panel display (Close bug fix)', () => {
  assert.doesNotMatch(html, /body\.mode-super \.super-only \{\s*display:\s*revert/);
  assert.match(html, /body\.mode-simple \.super-only \{\s*display:\s*none !important/);
  assert.match(html, /\.panel\.visible \{\s*display:\s*block/);
});

check('no pose/rtmpose/bodyOk gate in UI release', () => {
  assert.doesNotMatch(app, /rtmpose|PoseVerifier|bodyOk|verifyBodies|bodyMinArea|POSE_REL|softBodyGate/i);
  assert.doesNotMatch(html, /rtmpose|setPose|chip-pose/i);
  assert.equal(requireFsExists('crowd/rtmpose.js'), false);
});

function requireFsExists(rel) {
  try { readFileSync(join(root, rel)); return true; } catch { return false; }
}

check('Simple/Superuser + Count/Done + i18n present', () => {
  assert.match(html, /data-ui="simple"/);
  assert.match(html, /data-ui="super"/);
  assert.match(html, /data-i18n="count"/);
  assert.match(html, /data-i18n="done"/);
  assert.match(html, /mode-simple/);
  assert.match(app, /function applyUiMode/);
  assert.match(app, /function trackIsRegisteredAccepted/);
  assert.match(app, /I18N/);
});

// Replicate the acceptance predicate used by the app (must stay in sync with comment).
function trackIsRegisteredAccepted(t, ledgerRoot) {
  if (!t || !t.matchedThisFrame || !t.personId) return false;
  return !!ledgerRoot(t.personId);
}

check('accepted boxes == existing registered personId (no new filter)', () => {
  const people = new Map([[1, { id: 1, mergedInto: null }], [2, { id: 2, mergedInto: -1, ghost: true }]]);
  const root = (id) => { let p = people.get(id); while (p && p.mergedInto) p = people.get(p.mergedInto); return p; };
  assert.equal(trackIsRegisteredAccepted({ matchedThisFrame: true, personId: 1 }, root), true);
  assert.equal(trackIsRegisteredAccepted({ matchedThisFrame: true, personId: 2 }, root), false); // ghost
  assert.equal(trackIsRegisteredAccepted({ matchedThisFrame: true, personId: 0 }, root), false);
  assert.equal(trackIsRegisteredAccepted({ matchedThisFrame: false, personId: 1 }, root), false);
  assert.equal(trackIsRegisteredAccepted({ matchedThisFrame: true, personId: 99 }, root), false);
  // Confirm app draws with this helper and filters only in simple mode
  assert.match(app, /trackIsRegisteredAccepted\(tr\)/);
  assert.match(app, /if \(simple && !accepted\) continue/);
  assert.doesNotMatch(app, /t\.bodyOk/);
});

check('Done stops processing (running=false + loopGen++)', () => {
  assert.match(app, /\$\('stopBtn'\)\.onclick = \(\) => \{[\s\S]*?running = false; loopGen\+\+/);
  assert.match(app, /while \(running && gen === loopGen\)/);
  assert.match(app, /cam\.stream\.getTracks\(\)\.forEach/);
});

check('mode switch preserves session (no ledger/tracker reset)', () => {
  const uiHandler = app.match(/#uiSeg[\s\S]*?forEach\(b => b\.onclick = \(\) => \{([\s\S]*?)\}\);/);
  assert.ok(uiHandler, 'uiSeg handler missing');
  assert.doesNotMatch(uiHandler[1], /ledger\.reset|tracker\.reset/);
  assert.match(uiHandler[1], /applyUiMode\(\)/);
});

check('Count Again resets then Count restarts', () => {
  assert.match(app, /\$\('resetBtn'\)\.onclick = \(\) => \{[\s\S]*?ledger\.reset\(\)/);
  assert.match(app, /\$\('startBtn'\)\.onclick = async \(\) => \{[\s\S]*?running = true/);
});

const failed = results.filter(r => !r.ok);
console.log(JSON.stringify({ kind: 'SYNTHETIC', results, failed: failed.length }, null, 2));
process.exit(failed.length ? 1 : 0);
