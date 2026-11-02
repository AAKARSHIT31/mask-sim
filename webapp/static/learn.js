/* Interactive explainer — all client-side. Radix-2 FFT, 256x256.
   Every figure is computed from whichever image the source bar is showing:
   whatever was last uploaded to the detector, or the fallback sample. */
const N = 256, $ = s => document.querySelector(s);

/* Bind an event handler only when the element really is in the page.  Scripts
   run top-to-bottom: without this guard a single stale/renamed id throws and
   every widget defined below it dies with it.  Warn, skip, carry on. */
function on(sel, ev, fn) {
  const el = $(sel);
  if (!el) { console.warn('learn.js: no element ' + sel + ' — handler skipped'); return null; }
  el[ev] = fn;
  return el;
}

/* ---------- FFT ---------- */
function fft1(re, im, inv) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inv ? 2 : -2) * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
  if (inv) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}
function fft2(re, im, inv) {
  const r = new Float64Array(N), i2 = new Float64Array(N);
  for (let y = 0; y < N; y++) {                       // rows
    for (let x = 0; x < N; x++) { r[x] = re[y * N + x]; i2[x] = im[y * N + x]; }
    fft1(r, i2, inv);
    for (let x = 0; x < N; x++) { re[y * N + x] = r[x]; im[y * N + x] = i2[x]; }
  }
  for (let x = 0; x < N; x++) {                       // cols
    for (let y = 0; y < N; y++) { r[y] = re[y * N + x]; i2[y] = im[y * N + x]; }
    fft1(r, i2, inv);
    for (let y = 0; y < N; y++) { re[y * N + x] = r[y]; im[y * N + x] = i2[y]; }
  }
}
const sh = k => (k + N / 2) % N;                       // fftshift index

/* ---------- canvas helpers ---------- */
function drawGray(cv, data, lo, hi) {
  const ctx = cv.getContext('2d'), im = ctx.createImageData(N, N);
  if (lo === undefined) { lo = Infinity; hi = -Infinity;
    for (const v of data) { if (v < lo) lo = v; if (v > hi) hi = v; } }
  const s = 255 / (hi - lo + 1e-9);
  for (let i = 0; i < N * N; i++) {
    const raw = (data[i] - lo) * s;
    const v = isFinite(raw) ? Math.max(0, Math.min(255, raw)) : 0;
    im.data[i * 4] = im.data[i * 4 + 1] = im.data[i * 4 + 2] = v; im.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}
const MAGMA = [[0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],[229,80,100],[251,135,97],[254,194,135],[252,253,191]];
function magmaImageData(ctx, data, pLo = 1, pHi = 99.8) {
  const s = Float64Array.from(data).sort();
  const lo = s[Math.max(0, Math.min(s.length - 1, Math.floor(s.length * pLo / 100)))];
  const hi = s[Math.max(0, Math.min(s.length - 1, Math.floor(s.length * pHi / 100)))];
  const im = ctx.createImageData(N, N);
  for (let i = 0; i < N * N; i++) {
    let t = (data[i] - lo) / (hi - lo + 1e-9);
    t = (isFinite(t) ? Math.max(0, Math.min(1, t)) : 0) * (MAGMA.length - 1);
    const a = MAGMA[Math.max(0, Math.min(MAGMA.length - 1, Math.floor(t)))];
    const b = MAGMA[Math.max(0, Math.min(MAGMA.length - 1, Math.ceil(t)))], f = t % 1;
    im.data[i * 4] = a[0] + (b[0] - a[0]) * f;
    im.data[i * 4 + 1] = a[1] + (b[1] - a[1]) * f;
    im.data[i * 4 + 2] = a[2] + (b[2] - a[2]) * f;
    im.data[i * 4 + 3] = 255;
  }
  return im;
}
function drawMagma(cv, data, pLo = 1, pHi = 99.8) {
  const ctx = cv.getContext('2d');
  ctx.putImageData(magmaImageData(ctx, data, pLo, pHi), 0, 0);
}
function logSpec(re, im) {                             // shifted log-magnitude
  const out = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
    out[sh(y) * N + sh(x)] = Math.log(Math.hypot(re[y * N + x], im[y * N + x]) + 1e-8);
  return out;
}
function highpass(img) {                               // 3x3 mean removed, wrapped
  const out = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let s = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++)
      s += img[((y + dy + N) % N) * N + ((x + dx + N) % N)];
    out[y * N + x] = img[y * N + x] - s / 9;
  }
  return out;
}

/* ---------- 1 · wave builder (independent of the source image) ---------- */
function wave() {
  const ky = +$('#ky').value, kx = +$('#kx').value;
  $('#kyO').textContent = ky; $('#kxO').textContent = kx;
  const img = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
    img[y * N + x] = Math.cos(2 * Math.PI * (ky * y + kx * x) / N);
  drawGray($('#waveC'), img, -1, 1);
  const re = Float64Array.from(img), im = new Float64Array(N * N);
  fft2(re, im, false);
  const sp = logSpec(re, im);
  const d = new Float64Array(N * N);                   // dilate so single dots show
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let m = -1e9;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      const yy = y + dy, xx = x + dx;
      if (yy >= 0 && yy < N && xx >= 0 && xx < N) m = Math.max(m, sp[yy * N + xx]);
    }
    d[y * N + x] = m;
  }
  drawMagma($('#waveF'), d, 0, 100);
}

/* ---------- the source image ---------- */
let PHOTO = null, PRE = null;
function loadPhoto(src, cb) {
  const img = new Image();
  img.onload = () => {
    const c = document.createElement('canvas'); c.width = c.height = N;
    const ctx = c.getContext('2d'); ctx.drawImage(img, 0, 0, N, N);
    const d = ctx.getImageData(0, 0, N, N).data;
    PHOTO = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++)                    // Rec.601 luma
      PHOTO[i] = (0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]) / 255;
    const re = Float64Array.from(PHOTO), im = new Float64Array(N * N);
    fft2(re, im, false);
    PRE = { re, im, mag: (() => { const m = new Float64Array(N * N);
      for (let i = 0; i < N * N; i++) m[i] = Math.hypot(re[i], im[i]); return m; })() };
    cb();
  };
  img.onerror = () => showMsg('That file could not be decoded as an image.', true);
  img.src = src;
}

/* ---------- 2 · the spectrum probe ---------- */
const PROBE = { which: 'image', lock: null, hover: null, F: null, base: null, pix: null };

function probeBuild() {
  const mk = signal => {
    const re = Float64Array.from(signal), im = new Float64Array(N * N);
    fft2(re, im, false);
    const mag = new Float64Array(N * N), log = new Float64Array(N * N), ph = new Float64Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const i = y * N + x, j = sh(y) * N + sh(x);
      mag[j] = Math.hypot(re[i], im[i]);
      ph[j] = Math.atan2(im[i], re[i]);
      log[j] = Math.log(mag[j] + 1e-8);
    }
    const sorted = Float64Array.from(mag).sort();
    return { mag, log, ph, sorted };
  };
  const res = highpass(PHOTO);
  PROBE.F = { image: mk(PHOTO), residual: mk(res) };
  PROBE.base = { image: PHOTO, residual: res };
  /* cache the coloured spectrum so the crosshair can be redrawn cheaply */
  const ctx = $('#specC').getContext('2d');
  PROBE.pix = { image: magmaImageData(ctx, PROBE.F.image.log),
                residual: magmaImageData(ctx, PROBE.F.residual.log) };
  PROBE.lock = null; PROBE.hover = null;
  probeDraw();
}

function probeActive() { return PROBE.lock || PROBE.hover || { y: N / 2, x: N / 2 + 8 }; }

function probeDraw() {
  const w = PROBE.which, F = PROBE.F[w], ctx = $('#specC').getContext('2d');
  ctx.putImageData(PROBE.pix[w], 0, 0);

  const p = probeActive();
  const ky = p.y - N / 2, kx = p.x - N / 2;
  const my = N / 2 - ky, mx = N / 2 - kx;              // the mirror bin

  /* Drawn twice — a dark halo first, then the bright stroke — so the marker stays
     visible over both the blazing centre and the near-black corners. */
  const overlay = (colour, width, alpha) => {
    ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.globalAlpha = alpha;
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(p.x + .5, p.y + .5, 7, 0, 7); ctx.stroke();
    ctx.beginPath(); ctx.arc(mx + .5, my + .5, 5, 0, 7); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(N / 2 + .5, N / 2 + .5); ctx.lineTo(p.x + .5, p.y + .5); ctx.stroke();
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.arc(N / 2 + .5, N / 2 + .5, Math.hypot(ky, kx), 0, 7); ctx.stroke();
  };
  ctx.save();
  overlay('#000', 3.4, .6);
  overlay(PROBE.lock ? '#7cc0f5' : '#ffffff', 1.3, .95);
  ctx.restore();

  /* the source signal this spectrum was computed from */
  drawGray($('#probeSrc'), PROBE.base[w]);
  $('#probeSrcCap').textContent = w === 'image'
    ? 'your image (luma) — the signal above it'
    : 'the residual: your image minus its local average';

  /* the single frequency, at its true amplitude and phase */
  const i = p.y * N + p.x;
  const amp = 2 * F.mag[i] / (N * N), phase = F.ph[i];
  const pat = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
    pat[y * N + x] = Math.cos(2 * Math.PI * (ky * y + kx * x) / N + phase);
  drawGray($('#probeWave'), pat, -1, 1);

  const r = Math.hypot(ky, kx);
  const period = r < 0.5 ? Infinity : N / r;
  const ang = (Math.atan2(ky, kx) * 180 / Math.PI + 360) % 180;
  let rank = 0;                                        // percentile of this bin's magnitude
  { const s = F.sorted; let lo = 0, hi = s.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (s[m] < F.mag[i]) lo = m + 1; else hi = m; }
    rank = 100 * lo / s.length; }

  $('#probeWaveCap').textContent = r < 0.5
    ? 'DC — the average brightness of the whole image'
    : `stripes every ${period.toFixed(1)} px, at ${ang.toFixed(0)}° · amplitude ${(amp * 255).toFixed(3)} grey levels`;

  $('#specRead').innerHTML = PROBE.lock
    ? `<b style="color:var(--c1)">locked</b> · k<sub>y</sub> ${ky}  k<sub>x</sub> ${kx}`
    : `k<sub>y</sub> ${ky}  k<sub>x</sub> ${kx} · click to lock`;

  $('#probeStats').innerHTML = [
    ['bin', `${ky}, ${kx}`],
    ['repeats every', r < 0.5 ? '—' : period.toFixed(1) + ' px'],
    ['amplitude in the image', (amp * 255).toFixed(3) + ' / 255'],
    ['stronger than', rank.toFixed(1) + '% of bins'],
  ].map(([k, v]) => `<div class="cell"><span>${k}</span><b>${v}</b></div>`).join('');

  const rel = r / (N / 2);
  $('#probeNote').innerHTML = rel < 0.12
    ? `<strong>You are in the content region.</strong> At this radius the coefficients describe
       broad shapes and lighting. They are enormous, they are different in every photograph, and a
       detector that looked here would only learn what the picture is <em>of</em>.`
    : rel < 0.45
    ? `<strong>Mid frequencies: texture and edges.</strong> Fabric, foliage, hair, JPEG blocking.
       Still driven by the scene, still unreliable as evidence about who made the image.`
    : `<strong>This is where fingerprints live.</strong> Out here the coefficients are tiny and
       scene-independent, which is exactly what makes them useful: whatever a generator leaves at a
       fixed coordinate out here is about the <em>generator</em>, not the picture. This is the
       region the trained mask opens up.`;

  probeRadial(F, r);
}

function probeRadial(F, mark) {
  const NB = 86, W = 620, H = 200, L = 46, R = 16, T = 12, B = 34;
  const sum = new Float64Array(NB), cnt = new Float64Array(NB), rmax = N / 2;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const d = Math.hypot(y - N / 2, x - N / 2);
    if (d > rmax) continue;
    const b = Math.min(NB - 1, Math.floor(d / rmax * NB));
    sum[b] += F.log[y * N + x]; cnt[b]++;
  }
  const ys = Array.from(sum, (v, i) => v / Math.max(1, cnt[i]));
  const lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
  const X = i => L + i / (NB - 1) * (W - L - R);
  const Y = v => T + (1 - (v - lo) / (hi - lo + 1e-9)) * (H - T - B);
  const path = ys.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
  const mi = Math.min(NB - 1, Math.floor(mark / rmax * NB));
  $('#probeRadial').setAttribute('viewBox', `0 0 ${W} ${H}`);
  $('#probeRadial').innerHTML = `
    <defs><linearGradient id="pg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="var(--c1)" stop-opacity=".34"/>
      <stop offset="1" stop-color="var(--c1)" stop-opacity="0"/></linearGradient></defs>
    <line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="var(--line)"/>
    <path d="${path} L${X(NB - 1)},${H - B} L${X(0)},${H - B} Z" fill="url(#pg)"/>
    <path d="${path}" fill="none" stroke="var(--c1)" stroke-width="2"/>
    <line x1="${X(mi)}" x2="${X(mi)}" y1="${T}" y2="${H - B}" stroke="var(--c2)" stroke-width="1.6"/>
    <circle cx="${X(mi)}" cy="${Y(ys[mi])}" r="4.5" fill="var(--c2)"/>
    <text x="${X(mi)}" y="${T + 10}" fill="var(--c2)" font-size="10" text-anchor="middle"
      font-family="var(--mono)">you are here</text>
    <text x="${L}" y="${H - 8}" fill="var(--ink-3)" font-size="10.5">DC · image content</text>
    <text x="${W - R}" y="${H - 8}" fill="var(--ink-3)" font-size="10.5" text-anchor="end">
      Nyquist · where fingerprints live</text>
    <text x="12" y="${(T + H - B) / 2}" fill="var(--ink-3)" font-size="10.5" text-anchor="middle"
      transform="rotate(-90 12 ${(T + H - B) / 2})">mean log |DFT|</text>`;
}

function probeWire() {
  const cv = $('#specC');
  const at = e => {
    const r = cv.getBoundingClientRect();
    return { x: Math.max(0, Math.min(N - 1, Math.round((e.clientX - r.left) / r.width * N))),
             y: Math.max(0, Math.min(N - 1, Math.round((e.clientY - r.top) / r.height * N))) };
  };
  cv.addEventListener('mousemove', e => { if (!PROBE.lock) { PROBE.hover = at(e); probeDraw(); } });
  cv.addEventListener('mouseleave', () => { if (!PROBE.lock) { PROBE.hover = null; probeDraw(); } });
  cv.addEventListener('click', e => { PROBE.lock = PROBE.lock ? null : at(e); probeDraw(); });
  document.querySelectorAll('#specTabs .tab').forEach(b => b.onclick = () => {
    document.querySelectorAll('#specTabs .tab').forEach(x => x.classList.remove('on'));
    b.classList.add('on'); PROBE.which = b.dataset.spec; probeDraw();
  });
}

/* ---------- 3 · reconstruction ---------- */
function recon() {
  const pct = +$('#keep').value / 100;
  $('#keepO').textContent = pct.toFixed(2) + '%';
  const n = Math.max(1, Math.round(N * N * pct / 100));
  const sorted = Float64Array.from(PRE.mag).sort();
  const thr = sorted[N * N - n];
  const re = new Float64Array(N * N), im = new Float64Array(N * N), keepMap = new Float64Array(N * N);
  for (let i = 0; i < N * N; i++) if (PRE.mag[i] >= thr) { re[i] = PRE.re[i]; im[i] = PRE.im[i]; }
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
    keepMap[sh(y) * N + sh(x)] = PRE.mag[y * N + x] >= thr ? 1 : 0;
  fft2(re, im, true);
  drawGray($('#recC'), re);
  drawMagma($('#recF'), keepMap, 0, 100);
  $('#recCap').textContent = `rebuilt from ${n.toLocaleString()} of ${(N*N).toLocaleString()} coefficients`;
}

/* ---------- 4 · low / high pass ---------- */
function split() {
  const R = +$('#rad').value; $('#radO').textContent = R;
  const lr = new Float64Array(N * N), li = new Float64Array(N * N);
  const hr = new Float64Array(N * N), hi = new Float64Array(N * N);
  const ring = new Float64Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const dy = ((y + N / 2) % N) - N / 2, dx = ((x + N / 2) % N) - N / 2;
    const inside = Math.hypot(dy, dx) <= R, i = y * N + x;
    if (inside) { lr[i] = PRE.re[i]; li[i] = PRE.im[i]; } else { hr[i] = PRE.re[i]; hi[i] = PRE.im[i]; }
    ring[sh(y) * N + sh(x)] = inside ? 1 : .18;
  }
  fft2(lr, li, true); fft2(hr, hi, true);
  drawGray($('#lpC'), lr); drawGray($('#hpC'), hr);
  drawMagma($('#splitF'), ring, 0, 100);
}

/* ---------- the source bar ---------- */
const SAMPLE = { name: 'cameraman (sample)', image: null, available: false,
                 how: 'the standard 256×256 test image, bundled with this page' };

function showMsg(text, bad) {
  const el = $('#srcMsg');
  el.textContent = text; el.classList.remove('hide');
  el.classList.toggle('bad', !!bad);
  if (!text) el.classList.add('hide');
}

function verdictStrip(s) {
  const el = $('#srcVerdict');
  if (s.scoring) {
    el.innerHTML = `<div class="lbl2">The real detectors</div>
      <p class="srchow" style="margin-top:8px">Scoring with the seven checkpoints… the answer
      lands in <a href="#judge">Step 6</a> in a few seconds.</p>`;
    return;
  }
  const r = s.result;
  if (!r || !r.generic) {
    el.innerHTML = `<div class="lbl2">The real detectors</div>
      <p class="srchow" style="margin-top:8px">${s.available
        ? (s.note || 'Not scored.')
        : 'The sample is a 1978 photograph — there is nothing in it to find. Upload an image, real or AI, and the seven checkpoints will judge it in <a href="#judge">Step 6</a>.'}</p>`;
    return;
  }
  const lv = r.generic.verdict.level;
  const top = r.detectors.slice(0, 3);
  el.innerHTML = `<div class="lbl2">The real detectors say</div>
    <div class="vsum lv${lv}">
      <b>${r.generic.verdict.label}</b>
      <span>FPR ${(r.generic.fpr * 100).toFixed(1)}% · closest ${r.generic.best}</span>
    </div>
    <table class="mini">${top.map(d => `<tr><td>${d.name}</td>
      <td class="num lv${d.verdict.level}" style="color:var(--v)">${(d.fpr * 100).toFixed(1)}%</td></tr>`).join('')}</table>
    <a class="srchow" href="#judge">the full verdict, with every map →</a>`;
}

function applySource(s) {
  $('#srcThumb').src = s.image;
  $('#srcName').textContent = s.name || 'uploaded image';
  $('#srcHow').textContent = s.how || '';
  $('#srcTag').textContent = s.available ? 'your image' : 'sample';
  $('#srcTag').classList.toggle('live', !!s.available);
  verdictStrip(s);
  if (window.renderVerdict) window.renderVerdict(s);
  loadPhoto(s.image, () => {
    drawGray($('#origC'), PHOTO, 0, 1);
    probeBuild(); recon(); split();
    window.PHOTO_READY = true;
  });
}

function loadSample() {
  SAMPLE.image = window.SAMPLE_IMG;
  applySource(SAMPLE);
}

let pollTimer = null;

function refreshVerdict(s) {
  verdictStrip(s);
  if (window.renderVerdict) window.renderVerdict(s);
}

/* The server answers the upload as soon as the thumbnail is ready, then scores the
   image in the background. Poll until the verdict lands. */
function pollVerdict(deadline) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(() => {
    fetch('/api/source').then(r => r.ok ? r.json() : null).then(s => {
      if (!s || !s.available) return;
      refreshVerdict(s);
      if (s.scoring) {
        if (Date.now() < deadline) pollVerdict(deadline);
        else showMsg('The detectors are taking unusually long. The server console will show why.', true);
      } else {
        showMsg(s.note ? `Used for every figure on this page, but not scored: ${s.note}` : '', !!s.note);
      }
    }).catch(() => {});
  }, 1200);
}

const STALE = 'This page is newer than the server that is running. Stop it (Ctrl+C in the ' +
              'terminal) and start it again with  python webapp/server.py';

function srcWire() {
  if (!$('#srcFile')) return;
  on('#srcPick', 'onclick', () => $('#srcFile').click());
  $('#srcFile').onchange = () => {
    const f = $('#srcFile').files[0];
    if (!f) return;
    showMsg(`Reading ${f.name}…`);
    const fd = new FormData();
    fd.append('file', f, f.name);
    fetch('/api/source', { method: 'POST', body: fd })
      .then(async r => {
        if (r.status === 404 || r.status === 405) throw new Error(STALE);
        if (!r.ok) {
          let d; try { d = (await r.json()).detail; } catch (_) { d = r.statusText; }
          throw new Error(d || ('HTTP ' + r.status));
        }
        return r.json();
      })
      .then(s => {
        applySource(s);
        if (s.scoring) {
          showMsg('Loaded. Running the seven detectors — the verdict appears in Step 6.');
          pollVerdict(Date.now() + 180000);
        } else {
          showMsg(s.note || '', !!s.note);
        }
      })
      .catch(e => showMsg(e.message === STALE ? STALE : 'Upload failed — ' + e.message, true))
      .finally(() => { $('#srcFile').value = ''; });   // so the same file can be re-picked
  };
  $('#srcSample').onclick = () => {
    clearTimeout(pollTimer);
    fetch('/api/source', { method: 'DELETE' })
      .catch(() => {})
      .finally(() => { loadSample(); showMsg(''); });
  };
}

/* ---------- boot ---------- */
window.addEventListener('error', e => {
  const d = document.createElement('div'); d.className = 'err';
  d.style.margin = '16px 0';
  d.textContent = 'Visualisation error: ' + e.message +
    '  (' + (e.filename || '').split('/').pop() + ':' + e.lineno + ')';
  document.querySelector('.wrap').prepend(d);
});

['ky', 'kx'].forEach(id => on('#' + id, 'oninput', wave));
on('#keep', 'oninput', recon);
on('#rad', 'oninput', split);
wave(); probeWire(); srcWire();

fetch('/api/source')
  .then(r => {
    if (!r.ok) throw new Error(STALE);
    return r.json();
  })
  .then(s => {
    if (s.available && s.image) {
      applySource(s);
      if (s.scoring) pollVerdict(Date.now() + 180000);
    } else {
      loadSample();
    }
  })
  .catch(e => {
    loadSample();
    if (e && e.message === STALE) showMsg(STALE, true);
  });
