// Injected before page scripts: replaces the camera with a synthetic scene rendered on a canvas.
(() => {
  const SC = window.__SCENE; if (!SC) return;
  const LENSES = SC.lenses || [{ id: 'cam-wide', label: 'Back Camera', zoom: 1 }];
  const W = SC.W || 1280, H = SC.H || 720;
  const imgs = {}; const load = src => imgs[src] || (imgs[src] = Object.assign(new Image(), { src }));
  let canvas, ctx, t0 = 0, timer = null, lens = LENSES[0], speed = SC.speed || 1, noiseCanvas;
  window.__fake = { time: () => (performance.now() - t0) / 1000 * speed, lens: () => lens.label, started: false };
  const lerpPath = (path, t) => {
    if (t < path[0].t || t > path[path.length - 1].t) return null;
    for (let i = 0; i < path.length - 1; i++) { const a = path[i], b = path[i + 1]; if (t >= a.t && t <= b.t) { const u = (t - a.t) / Math.max(1e-6, b.t - a.t); return { x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u, h: (a.h || 0) + ((b.h || a.h || 0) - (a.h || 0)) * u, dir: Math.sign(b.x - a.x) }; } }
    const l = path[path.length - 1]; return { x: l.x, y: l.y, h: l.h, dir: 0 };
  };
  function frame() {
    const t = window.__fake.time();
    const cam = SC.camera ? lerpPath(SC.camera, Math.min(t, SC.camera[SC.camera.length - 1].t)) || SC.camera[0] : { x: W / 2, y: H / 2 };
    const z = lens.zoom; ctx.save();
    ctx.fillStyle = '#333'; ctx.fillRect(0, 0, W, H);
    ctx.translate(W / 2, H / 2); ctx.scale(z, z); ctx.translate(-cam.x, -cam.y);
    const bg = load(SC.bg); if (bg.complete && bg.naturalWidth) ctx.drawImage(bg, 0, 0, SC.worldW || W, SC.worldH || H);
    const ppl = (SC.people || []).map(p => ({ p, s: lerpPath(p.path, t) })).filter(o => o.s).sort((a, b) => a.s.y - b.s.y);
    for (const { p, s } of ppl) {
      const im = load(p.sprite); if (!im.complete || !im.naturalWidth) continue;
      const h = s.h || p.h, w = im.naturalWidth * h / im.naturalHeight, bob = Math.abs(Math.sin(t * 6 + (p.phase || 0))) * h * 0.01 * (s.dir ? 1 : 0);
      ctx.save(); ctx.translate(s.x, s.y - bob);
      if ((p.flipWhenLeft !== false) && s.dir < 0) ctx.scale(-1, 1);
      ctx.drawImage(im, -w / 2, -h, w, h); ctx.restore();
    }
    ctx.restore();
    if (SC.brightness != null && SC.brightness < 1) { ctx.fillStyle = `rgba(0,0,0,${1 - SC.brightness})`; ctx.fillRect(0, 0, W, H); }
    if (SC.noise) { ctx.globalAlpha = SC.noise; ctx.drawImage(noiseCanvas, (Math.random() * 64) | 0, (Math.random() * 64) | 0, W, H, 0, 0, W, H); ctx.globalAlpha = 1; }
  }
  function ensure() {
    if (canvas) return;
    canvas = document.createElement('canvas'); canvas.width = W; canvas.height = H; ctx = canvas.getContext('2d');
    noiseCanvas = document.createElement('canvas'); noiseCanvas.width = W + 64; noiseCanvas.height = H + 64;
    const nc = noiseCanvas.getContext('2d'), id = nc.createImageData(W + 64, H + 64);
    for (let i = 0; i < id.data.length; i += 4) { const v = Math.random() * 255; id.data[i] = id.data[i + 1] = id.data[i + 2] = v; id.data[i + 3] = 255; }
    nc.putImageData(id, 0, 0);
    (SC.preload || []).forEach(load);
  }
  const md = navigator.mediaDevices || (navigator.mediaDevices = {});
  md.enumerateDevices = async () => LENSES.map(l => ({ kind: 'videoinput', deviceId: l.id, label: l.label, groupId: 'g' }));
  md.getUserMedia = async (c) => {
    ensure();
    const want = c && c.video && c.video.deviceId && (c.video.deviceId.exact || c.video.deviceId);
    lens = LENSES.find(l => l.id === want) || LENSES.find(l => /^Back Camera$/.test(l.label)) || LENSES[0];
    window.__fake.lensSwitches = (window.__fake.lensSwitches || 0) + (window.__fake.started ? 1 : 0);
    if (!window.__fake.started) { t0 = performance.now(); window.__fake.started = true; }
    if (!timer) timer = setInterval(frame, 33);
    frame();
    const stream = canvas.captureStream(30); const tr = stream.getVideoTracks()[0];
    try { Object.defineProperty(tr, 'label', { value: lens.label }); } catch (e) {}
    tr.getCapabilities = () => ({});
    tr.getSettings = () => ({ width: W, height: H });
    const stop = tr.stop.bind(tr); tr.stop = () => stop();
    return stream;
  };
})();
