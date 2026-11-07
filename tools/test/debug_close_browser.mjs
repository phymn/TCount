#!/usr/bin/env node
/**
 * Regression: Superuser panel Close must dismiss Debug/Settings/Self-test.
 * Camera: SIMULATED. Engines: Chromium (+ WebKit when Playwright has it).
 */
import { chromium, webkit } from 'playwright';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '../..');
const port = 8771;
const py = spawn('python3', ['-m', 'http.server', String(port), '--bind', '127.0.0.1'], { cwd: root, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 600));
const base = `http://127.0.0.1:${port}/crowd/index.html?backend=wasm&model=nano&ui=super&lang=en&_=${Date.now()}`;

const results = [];
function record(engine, name, kind, expected, actual, ok) {
  results.push({ engine, name, kind, expected, actual, ok });
}

async function withPage(engineName, launcher, exePath) {
  const opts = { headless: true, args: ['--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required'] };
  if (exePath) opts.executablePath = exePath;
  let browser;
  try { browser = await launcher.launch(opts); }
  catch (e) {
    record(engineName, 'launch', 'ENV', 'browser available', String(e.message || e), false);
    return;
  }
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
  await page.addInitScript(() => {
    const c = document.createElement('canvas'); c.width = 320; c.height = 240;
    const md = navigator.mediaDevices || (navigator.mediaDevices = {});
    md.getUserMedia = async () => { window.__fake = { simulated: true }; return c.captureStream(10); };
    md.enumerateDevices = async () => [{ kind: 'videoinput', deviceId: 's', label: 'Sim', groupId: 'g' }];
  });
  await page.goto(base, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForFunction(() => window.__tc?.VERSION);

  const ver = await page.evaluate(() => window.__tc.VERSION);
  record(engineName, 'VERSION', 'REAL_CODE', 'crowd-1.3-ui.2', ver, ver === 'crowd-1.3-ui.2');

  const footer = await page.evaluate(() => document.getElementById('build-footer').textContent.trim());
  record(engineName, 'footer version-only', 'REAL_DOM', 'crowd-1.3-ui.2', footer, footer === 'crowd-1.3-ui.2');

  // Panels must start hidden in Super (the bug made them display:block always).
  const initial = await page.evaluate(() => {
    const d = id => getComputedStyle(document.getElementById(id)).display;
    return { debug: d('debugPanel'), settings: d('settingsPanel'), self: d('selfTestPanel') };
  });
  record(engineName, 'panels hidden before open', 'REAL_DOM', 'all none', JSON.stringify(initial),
    initial.debug === 'none' && initial.settings === 'none' && initial.self === 'none');

  async function cycle(panelBtn, panelId, label) {
    await page.click(panelBtn);
    await page.waitForTimeout(150);
    const open = await page.evaluate(id => ({
      cls: document.getElementById(id).className,
      display: getComputedStyle(document.getElementById(id)).display,
    }), panelId);
    record(engineName, `${label} opens`, 'REAL_DOM', 'visible + display block', JSON.stringify(open),
      open.cls.includes('visible') && open.display === 'block');

    // Mouse click Close
    await page.click(`#${panelId} .close`);
    await page.waitForTimeout(150);
    const closed = await page.evaluate(id => ({
      cls: document.getElementById(id).className,
      display: getComputedStyle(document.getElementById(id)).display,
    }), panelId);
    record(engineName, `${label} Close (click)`, 'REAL_DOM', 'no visible + display none', JSON.stringify(closed),
      !closed.cls.includes('visible') && closed.display === 'none');

    // Touch-style reopen + close
    await page.click(panelBtn);
    await page.waitForTimeout(100);
    const box = await page.locator(`#${panelId} .close`).boundingBox();
    if (box) {
      await page.touchscreen.tap(box.x + box.width / 2, box.y + box.height / 2);
      await page.waitForTimeout(150);
    } else {
      await page.click(`#${panelId} .close`);
    }
    const closedTouch = await page.evaluate(id => getComputedStyle(document.getElementById(id)).display, panelId);
    record(engineName, `${label} Close (touch/tap)`, 'REAL_DOM+SIMULATED_TOUCH', 'display none', closedTouch, closedTouch === 'none');

    // Second open/close cycle
    await page.click(panelBtn);
    await page.click(`#${panelId} .close`);
    const closed2 = await page.evaluate(id => getComputedStyle(document.getElementById(id)).display, panelId);
    record(engineName, `${label} second cycle Close`, 'REAL_DOM', 'display none', closed2, closed2 === 'none');
  }

  await cycle('#debugBtn', 'debugPanel', 'Debug');
  await cycle('#settingsBtn', 'settingsPanel', 'Settings');
  await cycle('#selfTestBtn', 'selfTestPanel', 'Self-test');

  // Inject count, open debug, switch Simple↔Super / EN↔ES — panel must not stick; count preserved
  const uniqueBefore = await page.evaluate(() => {
    window.__tc.ledger.create('new', null, null);
    return window.__tc.ledger.unique;
  });
  await page.click('#debugBtn');
  await page.click('#uiSeg [data-ui="simple"]');
  const afterSimple = await page.evaluate(() => ({
    unique: window.__tc.ledger.unique,
    debugDisplay: getComputedStyle(document.getElementById('debugPanel')).display,
    simple: document.body.classList.contains('mode-simple'),
    footer: document.getElementById('build-footer').textContent.trim(),
  }));
  record(engineName, 'switch to Simple with Debug open', 'REAL_DOM+SIMULATED_LEDGER',
    `unique=${uniqueBefore}, debug hidden, simple, footer version`, JSON.stringify(afterSimple),
    afterSimple.unique === uniqueBefore && afterSimple.debugDisplay === 'none' && afterSimple.simple && afterSimple.footer === 'crowd-1.3-ui.2');

  await page.click('#uiSeg [data-ui="super"]');
  await page.click('#langSeg [data-lang="es"]');
  await page.click('#langSeg [data-lang="en"]');
  const afterLang = await page.evaluate(() => ({
    unique: window.__tc.ledger.unique,
    debugDisplay: getComputedStyle(document.getElementById('debugPanel')).display,
    footer: document.getElementById('build-footer').textContent.trim(),
  }));
  record(engineName, 'EN/ES + Super restore preserves count; debug stays closed', 'REAL_DOM',
    `unique=${uniqueBefore}, debug none, footer version`, JSON.stringify(afterLang),
    afterLang.unique === uniqueBefore && afterLang.debugDisplay === 'none' && afterLang.footer === 'crowd-1.3-ui.2');

  // Count start/done (SIMULATED camera) — best-effort; model load may fail in CI images
  await page.click('#startBtn');
  let started = false;
  try {
    await page.waitForFunction(() => window.__tc.running === true, null, { timeout: 15000 });
    started = true;
  } catch (_) { /* model/camera path may fail */ }
  record(engineName, 'Count starts (SIMULATED camera)', 'SIMULATED_CAMERA', 'running if models load',
    String(started), true); // informational — do not fail suite on model load
  if (started) {
    await page.click('#stopBtn');
    await page.waitForTimeout(300);
    const done = await page.evaluate(() => ({
      running: window.__tc.running,
      result: document.getElementById('result').classList.contains('visible'),
    }));
    record(engineName, 'Done stops', 'SIMULATED_CAMERA', 'not running + result', JSON.stringify(done), !done.running && done.result);
  } else {
    record(engineName, 'Done stops', 'SIMULATED_CAMERA', 'skipped (Count did not start)', 'skipped', true);
  }

  await browser.close();
}

await withPage('chromium', chromium, process.env.CHROME_PATH || '/usr/local/bin/google-chrome');

let webkitOk = false;
try {
  const b = await webkit.launch({ headless: true });
  await b.close();
  webkitOk = true;
} catch (e) {
  record('webkit', 'launch', 'ENV', 'WebKit host deps available', `UNAVAILABLE: ${e.message || e}`, true);
}
if (webkitOk) await withPage('webkit', webkit, undefined);

py.kill('SIGTERM');
const failed = results.filter(r => !r.ok);
const summary = {
  kind: 'BROWSER_PANEL_CLOSE',
  camera: 'SIMULATED',
  device_iphone: 'NOT RUN',
  webkit_browsers_installed: existsSync(join(process.env.HOME || '', '.cache/ms-playwright')),
  results,
  failed: failed.length,
};
console.log(JSON.stringify(summary, null, 2));
process.exit(failed.length ? 1 : 0);
