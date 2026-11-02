/* MaskSim Explorer — detect page */
const $ = s => document.querySelector(s);
const PAL = ['#3A8CCF','#CE6B22','#189970','#AE8818','#C05F98','#7B76C9','#4FA5A0'];
const LVCOL = ['var(--ink-3)','var(--warn)','var(--serious)','var(--crit)'];
let DATA = null, curDet = null, curLayer = 'simmap';

/* ---------------- boot ---------------- */
fetch('/api/status').then(r => r.json()).then(s => {
  $('#devbadge').textContent = s.ready
    ? `${s.detectors.length} detectors · ${s.crop}px crop` : 'loading model…';
}).catch(() => { $('#devbadge').textContent = 'backend offline'; });

/* ---------------- upload ---------------- */
const drop = $('#drop'), file = $('#file');
drop.onclick = e => { if (!e.target.dataset.sample) file.click(); };
$('#pick').onclick = e => { e.stopPropagation(); file.click(); };
file.onchange = () => file.files[0] && send(file.files[0]);
['dragenter','dragover'].forEach(k => drop.addEventListener(k, e => {
  e.preventDefault(); drop.classList.add('over'); }));
['dragleave','drop'].forEach(k => drop.addEventListener(k, e => {
  e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => e.dataTransfer.files[0] && send(e.dataTransfer.files[0]));
document.querySelectorAll('[data-sample]').forEach(b => b.onclick = e => {
  e.stopPropagation(); run(fetch('/api/sample/' + b.dataset.sample, { method: 'POST' })); });

function send(f) { const fd = new FormData(); fd.append('file', f); run(fetch('/api/detect', { method: 'POST', body: fd })); }

function run(p) {
  $('#err').classList.add('hide'); $('#results').classList.add('hide');
  $('#loading').classList.remove('hide');
  p.then(async r => { if (!r.ok) throw new Error((await r.json()).detail || r.statusText); return r.json(); })
   .then(d => { DATA = d; render(d); })
   .catch(e => { $('#err').textContent = 'Could not analyse that image — ' + e.message;
                 $('#err').classList.remove('hide'); })
   .finally(() => $('#loading').classList.add('hide'));
}

/* ---------------- render ---------------- */
const pc = f => f == null ? '—' : (f * 100).toFixed(1) + '%';

function render(d) {
  $('#results').classList.remove('hide');
  const g = d.generic, best = d.detectors[0], lv = g.verdict.level;

  $('#verdict').className = 'verdict lv' + lv;
  $('#vLabel').textContent = g.verdict.label.replace(/^./, c => c.toUpperCase());
  $('#vFpr').textContent = pc(g.fpr);
  $('#vSub').innerHTML = lv === 0
    ? `The generic model — the strongest of the seven detectors, re-calibrated on its own
       table of ${g.n_calibration.toLocaleString()} real photographs — found nothing.
       The closest was <strong>${g.best}</strong>, and even there ${pc(g.fpr)} of genuine
       photographs score at least this high.
       <em>This is not proof the image is authentic</em>: it means none of these seven
       generators left a trace this detector was trained to recognise.`
    : `Strongest match: <strong>${g.best}</strong>. Only ${pc(g.fpr)} of genuine photographs
       score this high, so calling this image synthetic would be a false alarm roughly
       1 time in ${Math.max(1, Math.round(1 / g.fpr))}.`;
  $('#vMeta').textContent =
    `${d.file} · ${d.dimensions.width}×${d.dimensions.height} · ${d.device}` +
    ` · best single detector ${best.name} at FPR ${pc(best.fpr)}`;
  ring(g.fpr, lv);

  stages(d.stages);
  fprChart(d.detectors);
  table(d.detectors);
  detTabs(d.detectors);
  radial(d.radial);
  const tl = $('#toLearn');
  if (tl) tl.textContent = `Take ${d.file} apart in \u201cHow it works\u201d \u2192`;
  $('#explainNote').innerHTML = lv === 0
    ? `The maps look like noise because nothing matched. When a detector <em>does</em> fire you see
       structured clusters or a regular lattice of dots — the signature of an upsampling filter.`
    : `Look for structure in the similarity map: a lattice of regular dots is the classic trace left
       by a decoder's final upsampling layer.`;
}

/* ---------------- verdict ring ---------------- */
function ring(fpr, lv) {
  const sus = fpr == null ? 0 : 1 - Math.min(fpr, 1);
  const pct = Math.max(0.02, sus);
  const R = 48, C = 2 * Math.PI * R;
  const s = sus * 100;
  const txt = fpr == null ? '—' : (s >= 99.95 ? '>99.9' : s >= 99 ? s.toFixed(1) : s.toFixed(0));
  $('#ring').innerHTML = `
    <circle cx="60" cy="60" r="${R}" fill="none" stroke="var(--line)" stroke-width="10"/>
    <circle cx="60" cy="60" r="${R}" fill="none" stroke="${LVCOL[lv]}" stroke-width="10"
      stroke-linecap="round" stroke-dasharray="${C}" stroke-dashoffset="${C * (1 - pct)}"
      transform="rotate(-90 60 60)"/>
    <text x="60" y="56" text-anchor="middle" fill="var(--ink)" font-size="${txt.length > 4 ? 16 : 21}"
      font-family="var(--mono)" font-weight="600">${txt}</text>
    <text x="60" y="73" text-anchor="middle" fill="var(--ink-3)" font-size="9"
      letter-spacing="1">PERCENTILE</text>`;
}

/* ---------------- stages ---------------- */
function stages(list) {
  $('#stages').innerHTML = list.map((s, i) => `
    <div class="stage ${i === 0 ? 'on' : ''}" data-i="${i}">
      <img src="${s.img}" alt="${s.title}">
      <div class="cap"><div class="t">${s.title}</div><div class="s">${s.sub}</div></div>
    </div>`).join('');
  const note = i => {
    document.querySelectorAll('.stage').forEach(e => e.classList.toggle('on', +e.dataset.i === i));
    $('#stageNote').innerHTML = `<strong>${list[i].title}</strong> — ${list[i].note}`;
  };
  document.querySelectorAll('.stage').forEach(e => e.onclick = () => note(+e.dataset.i));
  note(0);
}

/* ---------------- FPR bar chart (hand-rolled SVG) ---------------- */
/* Bars show SUSPICION = 1 - FPR, so longer means more suspicious; the number
   printed on each bar is still the false-positive rate the paper reports. */
function fprChart(dets) {
  const W = 560, H = 290, L = 132, R = 58, T = 20, B = 34;
  const iw = W - L - R, rows = dets.length, bh = Math.min(22, (H - T - B) / rows - 8);
  const x = s => L + Math.min(Math.max(s, 0), 1) * iw;
  let g = '';
  [[0.90, 'var(--warn)', '10%', 0], [0.95, 'var(--serious)', '5%', 11], [0.99, 'var(--crit)', '1%', 0]]
    .forEach(([t, c, lab, dy]) => {
      g += `<line x1="${x(t)}" x2="${x(t)}" y1="${T - 6 + dy}" y2="${H - B}" stroke="${c}"
              stroke-width="1" stroke-dasharray="3 3" opacity=".7"/>
            <text x="${x(t)}" y="${T - 9 + dy}" fill="${c}" font-size="9.5" text-anchor="middle"
              font-family="var(--mono)">${lab}</text>`;
    });
  dets.forEach((d, i) => {
    const y = T + i * ((H - T - B) / rows) + 4;
    const f = d.fpr == null ? 1 : d.fpr, s = 1 - f, col = LVCOL[d.verdict.level];
    g += `<text x="${L - 10}" y="${y + bh * .74}" fill="var(--ink-2)" font-size="12"
            text-anchor="end">${d.name}</text>
          <rect x="${L}" y="${y}" width="${iw}" height="${bh}" fill="var(--panel-2)" rx="4"/>
          <rect x="${L}" y="${y}" width="${Math.max(2, x(s) - L)}" height="${bh}" fill="${col}" rx="4">
            <title>${d.name}: FPR ${(f * 100).toFixed(1)}% — ${d.verdict.label}</title></rect>
          <text x="${W - R + 6}" y="${y + bh * .74}" fill="var(--ink-2)" font-size="11"
            font-family="var(--mono)">${(f * 100).toFixed(1)}%</text>`;
  });
  g += `<text x="${L}" y="${H - B + 15}" fill="var(--ink-3)" font-size="9.5">less suspicious</text>
        <text x="${W - R}" y="${H - B + 15}" fill="var(--ink-3)" font-size="9.5"
          text-anchor="end">more suspicious</text>
        <text x="${W - R + 6}" y="${T - 9}" fill="var(--ink-3)" font-size="9"
          font-family="var(--mono)">FPR</text>
        <text x="${L}" y="${T - 9}" fill="var(--ink-3)" font-size="9"
          font-family="var(--mono)">dashed = FPR thresholds</text>`;
  $('#fprChart').setAttribute('viewBox', `0 0 ${W} ${H}`);
  $('#fprChart').innerHTML = g;
}

/* ---------------- table ---------------- */
function table(dets) {
  $('#tbl').querySelector('tbody').innerHTML = dets.map((d, i) => `
    <tr data-k="${d.key}" class="${i === 0 ? 'sel' : ''}">
      <td><span class="swatch" style="background:${PAL[i % PAL.length]}"></span>${d.name}</td>
      <td class="num">${d.score.toFixed(5)}</td>
      <td class="num">${d.fpr == null ? '—' : (d.fpr * 100).toFixed(1) + '%'}</td>
      <td><span class="pill lv${d.verdict.level}">${d.verdict.label}</span></td>
    </tr>`).join('');
  $('#tbl').querySelectorAll('tr[data-k]').forEach(tr => tr.onclick = () => selectDet(tr.dataset.k));
}

/* ---------------- explorer ---------------- */
function detTabs(dets) {
  $('#detTabs').innerHTML = dets.map((d, i) =>
    `<button class="tab ${i === 0 ? 'on' : ''}" data-k="${d.key}">${d.name}</button>`).join('');
  $('#detTabs').querySelectorAll('.tab').forEach(b => b.onclick = () => selectDet(b.dataset.k));
  $('#layerTabs').querySelectorAll('.tab').forEach(b => b.onclick = () => {
    $('#layerTabs').querySelectorAll('.tab').forEach(x => x.classList.remove('on'));
    b.classList.add('on'); curLayer = b.dataset.layer; paint();
  });
  selectDet(dets[0].key);
}
function selectDet(k) {
  curDet = k;
  $('#detTabs').querySelectorAll('.tab').forEach(b => b.classList.toggle('on', b.dataset.k === k));
  $('#tbl').querySelectorAll('tr[data-k]').forEach(tr => tr.classList.toggle('sel', tr.dataset.k === k));
  paint();
}
const LEG = {
  simmap: '<span><i style="background:#fde725"></i>bright = this frequency agrees with the reference</span>' +
          '<span><i style="background:#440154"></i>dark = it does not</span>' +
          '<span>the whole map sums to the cosine score — same colours and range as sim_map_*.html</span>',
  mask:   '<span><i style="background:#6a51a3"></i>high = this frequency is allowed through</span>' +
          '<span>static — the same map for every image you upload</span>',
  masked: '<span><i style="background:#fca50a"></i>your spectrum after the mask is applied</span>' +
          '<span>only this survives to be compared with the reference</span>'
};
function paint() {
  if (!DATA || !curDet) return;
  $('#mapImg').src = DATA.maps[curDet][curLayer];
  $('#layerLegend').innerHTML = LEG[curLayer];
}
$('#mapWrap').addEventListener('mousemove', e => {
  const r = $('#mapImg').getBoundingClientRect();
  const u = (e.clientX - r.left) / r.width, v = (e.clientY - r.top) / r.height;
  const fx = ((u - .5) * 2).toFixed(2), fy = ((.5 - v) * 2).toFixed(2);
  const rad = Math.hypot(fx, fy).toFixed(2);
  $('#readout').textContent = `fx ${fx}  fy ${fy}  ·  radius ${rad} × Nyquist`;
});
$('#mapWrap').addEventListener('mouseleave', () => $('#readout').textContent = 'hover to inspect');

/* ---------------- radial profile ---------------- */
function radial(rp) {
  const W = 460, H = 230, L = 46, R = 14, T = 12, B = 30;
  const xs = rp.radius, ys = rp.energy;
  const lo = Math.min(...ys), hi = Math.max(...ys);
  const X = i => L + (xs[i]) * (W - L - R);
  const Y = v => T + (1 - (v - lo) / (hi - lo + 1e-9)) * (H - T - B);
  let d = ys.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
  let area = d + ` L${X(ys.length - 1).toFixed(1)},${H - B} L${X(0).toFixed(1)},${H - B} Z`;
  let g = `<defs><linearGradient id="gr" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="var(--c1)" stop-opacity=".38"/>
      <stop offset="1" stop-color="var(--c1)" stop-opacity="0"/></linearGradient></defs>
    <line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="var(--line)"/>
    <line x1="${L}" x2="${L}" y1="${T}" y2="${H - B}" stroke="var(--line)"/>
    <path d="${area}" fill="url(#gr)"/><path d="${d}" fill="none" stroke="var(--c1)" stroke-width="2"/>
    <text x="${(L + W - R) / 2}" y="${H - 7}" fill="var(--ink-3)" font-size="10.5"
      text-anchor="middle">distance from DC  →  Nyquist</text>
    <text x="10" y="${(T + H - B) / 2}" fill="var(--ink-3)" font-size="10.5"
      transform="rotate(-90 10 ${(T + H - B) / 2})" text-anchor="middle">log magnitude</text>
    <text x="${L + 4}" y="${T + 11}" fill="var(--ink-3)" font-size="9.5"
      font-family="var(--mono)">low freq — image content</text>
    <text x="${W - R - 4}" y="${T + 11}" fill="var(--ink-3)" font-size="9.5" text-anchor="end"
      font-family="var(--mono)">high freq — where fingerprints live</text>`;
  $('#radial').setAttribute('viewBox', `0 0 ${W} ${H}`);
  $('#radial').innerHTML = g;
}
