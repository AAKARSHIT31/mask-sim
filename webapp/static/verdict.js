/* =====================================================================
   Step 4 — your image through the detector.

   Eight stages, walked in order, every panel drawn from a tensor the
   trained model actually computed on the uploaded image. Nothing is
   planted, nothing is simulated: these are the seven checkpoints from
   the paper, and the numbers match detect_one_image.py exactly.
   ===================================================================== */
(function () {
  'use strict';
  const Q = s => document.querySelector(s);
  /* grey, not green, for "no evidence": silence is not a clean bill of health */
  const LVCOL = ['var(--ink-3)', 'var(--warn)', 'var(--serious)', 'var(--crit)'];
  const pc = f => f == null ? '—' : (f * 100).toFixed(1) + '%';
  const fx = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(5);

  let R = null, cur = null, stage = 0, timer = null;

  /* ---------- states before a result exists ---------------------------- */
  const shell = html => { Q('#judgeBody').innerHTML = html; };

  function stateNone() {
    shell(`<div class="panel"><div class="emptyjudge"><div>
      <h3>Upload an image to run the detector</h3>
      <p>The sample tile above is only there so the page has something to draw. The moment you
      upload a picture of your own, it is carried through all eight stages below and scored by the
      seven trained checkpoints.</p>
      <p class="small">The image must be at least 512×512, because the detector takes a fixed
      512×512 crop and never resizes. Smaller images still drive Steps 1–3.</p>
    </div></div></div>`);
  }
  function stateScoring() {
    shell(`<div class="panel"><div class="spinner"></div>
      <p style="text-align:center;color:var(--ink-3)">Running your image through the seven
      detectors… a few seconds on CPU.</p></div>`);
  }
  function stateNote(note) {
    shell(`<div class="panel"><div class="emptyjudge"><div>
      <h3>Not scored</h3><p>${note}</p>
      <p class="small">Steps 1–3 still use your image. Only this section, which needs the full
      512×512 crop the checkpoints were trained on, is unavailable.</p>
    </div></div></div>`);
  }

  /* ---------- the eight stages ----------------------------------------- */
  function STAGES() {
    const d = R.detectors.find(x => x.key === cur);
    const st = R.stages;
    return [
      { n: 'Input', t: '1 · The 512×512 crop',
        kind: 'FIXED', shape: `${R.dimensions.width}×${R.dimensions.height} → 3 × 512 × 512`,
        eq: 'x ← centre-crop( your image, 512 ), origin snapped to a multiple of 8',
        shared: true,
        panels: [[st[0].img, 'your image, exactly as uploaded — nothing added, nothing resized']],
        what: st[0].note },
      { n: 'Luma', t: '2 · Colour transform',
        kind: 'FIXED', shape: '3 × 512 × 512, YCbCr',
        eq: 'x ← RGB → YCbCr, scaled to [0, 1]',
        shared: true,
        panels: [[st[1].img, 'the Y (luma) channel of your image']],
        what: st[1].note },
      { n: 'Residual', t: '3 · The scene is removed',
        kind: 'FIXED · pre-trained, frozen', shape: '→ 3 × 478 × 478',
        eq: 'r = DnCNN( x )      — the network predicts the NOISE, not the clean image',
        shared: true,
        panels: [[st[1].img, 'in: your luma channel'], [st[2].img, 'out: the residual']],
        what: st[2].note },
      { n: 'Spectrum', t: '4 · Into the frequency domain',
        kind: 'FIXED', shape: '478 × 478 real',
        eq: 's = log( | FFT2( r ) | + ε ),  DC shifted to the centre',
        shared: true,
        panels: [[st[2].img, 'your residual'], [st[3].img, 'log |DFT| of your residual']],
        what: st[3].note,
        svg: () => radialSVG() },
      { n: 'Mask', t: '5 · The trained mask for this generator',
        kind: 'LEARNED', shape: '3 × 478 × 478 = 685,926 values in [0,1]',
        eq: 'M = σ( smooth( θ ) )   — fitted once, during training, and then frozen',
        panels: [[R.maps[cur].mask, `the mask ${d.name} examines your image with`]],
        what: 'One number per frequency bin: how much this detector trusts that bin. It is '
            + '<strong>static</strong> — the identical map multiplies every image this detector '
            + 'has ever scored, including yours. Bright means the bin is open; dark means it is '
            + 'ignored. Nothing here depends on your picture, which is exactly why it can be used '
            + 'as evidence about your picture.' },
      { n: 'Masked', t: '6 · What survives the mask',
        kind: 'FIXED · uses the learned M', shape: 'element-wise product',
        eq: 'v = M ⊙ s',
        panels: [[R.stages[3].img, 'your spectrum'], [R.maps[cur].mask, '×  the mask'],
                 [R.maps[cur].masked, '=  what reaches the comparison']],
        what: 'Your image’s spectrum multiplied by the mask, bin by bin. Almost everything is '
            + 'multiplied by nearly zero. What is left is the small set of frequencies where this '
            + 'generator is known to differ from a camera — measured on your image, not on a '
            + 'model of it.' },
      { n: 'Compare', t: '7 · Compared with the reference',
        kind: 'LEARNED', shape: 'cosine in 228,484 dimensions',
        eq: 'cos( v̂ , R ) = ⟨ v̂ , R ⟩ / ( ‖v̂‖ ‖R‖ )',
        panels: [[R.maps[cur].simmap,
                  'per-frequency contribution — every pixel is one term of the sum']],
        what: 'The masked spectrum is correlated against a second learned map: what this '
            + 'generator’s fingerprint is supposed to look like. Because the score is a sum over '
            + 'bins, its terms can be drawn as a picture — and the whole picture adds up to the '
            + 'number below. Bright regions are frequencies in <em>your</em> image that agree with '
            + 'the template; dark ones disagree. This is the same file the command-line tool writes '
            + 'as <span class="kbd">sim_map_' + cur + '.html</span>.',
        foot: () => `<span class="big-num">raw score = ${d.score.toFixed(5)}</span>
          <span class="muted">a cosine, rescaled by this detector’s two-parameter logistic</span>` },
      { n: 'Verdict', t: '8 · Ranked against real photographs',
        kind: 'MEASURED', shape: `rank among ${d.n_calibration.toLocaleString()} real photographs`,
        eq: 'FPR = 1 − rank( score ) / n',
        panels: [],
        svg: () => fprSVG(),
        what: 'The raw score means nothing on its own, so it is looked up in a table of scores from '
            + 'photographs known to be real. What comes back is your image’s rank: <em>this is '
            + 'the share of genuine photographs that would score at least as high as yours.</em> '
            + 'That is a false-alarm rate — the rate at which you would wrongly accuse an '
            + 'innocent image if you used this score as your threshold.',
        foot: () => {
          const g = R.generic, lv = g.verdict.level;
          return `<span class="big-num lv${lv}" style="color:var(--v)">${pc(g.fpr)}</span>
            <span class="muted">overall — <strong>${g.verdict.label}</strong>
            · closest match ${g.best}</span>`;
        } },
    ];
  }

  /* ---------- charts ---------------------------------------------------- */
  function radialSVG() {
    const rp = R.radial, W = 620, H = 200, L = 48, RT = 16, T = 14, B = 34;
    const ys = rp.energy, xs = rp.radius;
    const lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
    const X = i => L + xs[i] * (W - L - RT);
    const Y = v => T + (1 - (v - lo) / (hi - lo + 1e-9)) * (H - T - B);
    const d = ys.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
    return `<svg viewBox="0 0 ${W} ${H}" class="fig" style="width:100%;height:auto">
      <defs><linearGradient id="rg" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="var(--c1)" stop-opacity=".34"/>
        <stop offset="1" stop-color="var(--c1)" stop-opacity="0"/></linearGradient></defs>
      <line x1="${L}" x2="${W - RT}" y1="${H - B}" y2="${H - B}" stroke="var(--line)"/>
      <path d="${d} L${X(ys.length - 1)},${H - B} L${X(0)},${H - B} Z" fill="url(#rg)"/>
      <path d="${d}" fill="none" stroke="var(--c1)" stroke-width="2"/>
      <text x="${L}" y="${H - 8}" fill="var(--ink-3)" font-size="10.5">DC · your image’s content</text>
      <text x="${W - RT}" y="${H - 8}" fill="var(--ink-3)" font-size="10.5" text-anchor="end">
        Nyquist · where fingerprints live</text>
      <text x="12" y="${(T + H - B) / 2}" fill="var(--ink-3)" font-size="10.5" text-anchor="middle"
        transform="rotate(-90 12 ${(T + H - B) / 2})">mean log |DFT|</text>
    </svg>`;
  }

  function fprSVG() {
    const dets = R.detectors, W = 620, H = 300, L = 138, RT = 62, T = 22, B = 34;
    const iw = W - L - RT, rows = dets.length, bh = Math.min(22, (H - T - B) / rows - 8);
    const x = v => L + Math.min(Math.max(v, 0), 1) * iw;
    let g = '';
    [[0.90, 'var(--warn)', '10%', 0], [0.95, 'var(--serious)', '5%', 12],
     [0.99, 'var(--crit)', '1%', 0]].forEach(([t, c, lab, dy]) => {
      g += `<line x1="${x(t)}" x2="${x(t)}" y1="${T - 6 + dy}" y2="${H - B}" stroke="${c}"
              stroke-width="1" stroke-dasharray="3 3" opacity=".7"/>
            <text x="${x(t)}" y="${T - 9 + dy}" fill="${c}" font-size="9.5" text-anchor="middle"
              font-family="var(--mono)">${lab}</text>`;
    });
    dets.forEach((d, i) => {
      const y = T + i * ((H - T - B) / rows) + 4;
      const f = d.fpr == null ? 1 : d.fpr, col = LVCOL[d.verdict.level];
      const on = d.key === cur;
      g += `<text x="${L - 10}" y="${y + bh * .74}" fill="${on ? 'var(--ink)' : 'var(--ink-2)'}"
              font-size="12" text-anchor="end" font-weight="${on ? 600 : 400}">${d.name}</text>
            <rect x="${L}" y="${y}" width="${iw}" height="${bh}" fill="var(--panel-2)" rx="4"/>
            <rect x="${L}" y="${y}" width="${Math.max(2, x(1 - f) - L)}" height="${bh}"
              fill="${col}" rx="4"/>
            <text x="${W - RT + 6}" y="${y + bh * .74}" fill="var(--ink-2)" font-size="11"
              font-family="var(--mono)">${pc(f)}</text>`;
    });
    g += `<text x="${L}" y="${H - B + 15}" fill="var(--ink-3)" font-size="9.5">less suspicious</text>
          <text x="${W - RT}" y="${H - B + 15}" fill="var(--ink-3)" font-size="9.5"
            text-anchor="end">more suspicious</text>`;
    return `<svg viewBox="0 0 ${W} ${H}" class="fig" style="width:100%;height:auto">${g}</svg>`;
  }

  /* ---------- the result view ------------------------------------------ */
  function stateResult(s) {
    R = s.result;
    cur = R.detectors[0].key;
    stage = 0;
    const g = R.generic, best = R.detectors[0], lv = g.verdict.level;
    const verdictText = lv === 0
      ? `<p>None of the seven fingerprints is present in your image. The closest was
         <strong>${g.best}</strong>, and even there <strong>${pc(g.fpr)}</strong> of genuine
         photographs score at least as high — far too common to count as evidence.</p>
         <p><strong>This is not a certificate of authenticity.</strong> It means these seven
         specific models left no trace. An image from a generator none of them was trained on looks
         exactly like this, and so does a real photograph that has been resized.</p>`
      : `<p>Your image carries the fingerprint of <strong>${g.best}</strong>. Only
         <strong>${pc(g.fpr)}</strong> of genuine photographs score this high, so calling it
         synthetic would be a false alarm roughly
         <strong>1 time in ${Math.max(1, Math.round(1 / g.fpr))}</strong>.</p>
         <p>Walk the eight stages below to see exactly what convinced it.</p>`;

    shell(`
      <div class="verdict lv${lv}" style="margin-bottom:22px">
        <svg class="ring" id="jRing" viewBox="0 0 120 120"></svg>
        <div>
          <div class="lbl">The seven trained checkpoints, on your image</div>
          <div class="big">${g.verdict.label.replace(/^./, c => c.toUpperCase())}</div>
          <div class="sub">${verdictText}</div>
          <div class="meta">${R.file} · ${R.dimensions.width}×${R.dimensions.height}
            · ${R.device} · strongest single detector ${best.name} at FPR ${pc(best.fpr)}</div>
        </div>
        <div class="stat"><div class="n">${pc(g.fpr)}</div>
          <div class="u">false-positive rate</div></div>
      </div>

      <div class="panel lab">
        <div class="labtop" style="grid-template-columns:1fr">
          <div>
            <div class="lbl2" style="margin-bottom:8px">Which detector’s eyes to look through</div>
            <div class="tabs" id="jDetTabs"></div>
            <div class="srcnote" id="jDetNote"></div>
          </div>
        </div>

        <div class="labgrid">
          <div>
            <div class="stageview" id="jView"></div>
            <div class="stagefoot" id="jFoot"></div>
          </div>
          <aside class="stagedoc" id="jDoc"></aside>
        </div>

        <div class="stepbar">
          <button class="btn" id="jPrev">← Back</button>
          <div class="chips" id="jChips"></div>
          <span class="steppos" id="jPos"></span>
          <button class="btn" id="jPlay">▶ Play</button>
          <button class="btn primary" id="jNext">Next →</button>
        </div>

        <div class="livestrip" id="jLive"></div>
      </div>

      <div class="panel" style="margin-top:22px">
        <h2>What the number means</h2>
        <div class="grid g2" style="align-items:start">
          <div class="note" style="border-color:var(--c1)"><strong>Why a percentage and not a
            yes/no.</strong> Each detector was calibrated against roughly 12,000 photographs known to
            be real. Your image’s score is looked up in that table and what comes back is its
            rank. That is a false-alarm rate you can set a policy against — at 3 % you accuse
            three innocent images in every hundred. The authors deliberately ship no 0.5
            threshold.</div>
          <div class="note" style="border-color:var(--crit)"><strong>How to read a negative.</strong>
            “No evidence” answers one narrow question seven times over: does this image
            carry <em>this</em> generator’s fingerprint? Seven noes mean seven specific models
            left no trace. That is silence, not an alibi — and a resized image, or one from an
            eighth generator, produces exactly the same silence.</div>
        </div>
      </div>`);

    ring(g.fpr, lv);
    detTabs();
    render();
    wire();
  }

  function ring(fpr, lv) {
    const sus = fpr == null ? 0 : 1 - Math.min(fpr, 1), pct = Math.max(0.02, sus);
    const RR = 48, C = 2 * Math.PI * RR, v = sus * 100;
    const txt = fpr == null ? '—' : (v >= 99.95 ? '>99.9' : v >= 99 ? v.toFixed(1) : v.toFixed(0));
    Q('#jRing').innerHTML = `
      <circle cx="60" cy="60" r="${RR}" fill="none" stroke="var(--line)" stroke-width="10"/>
      <circle cx="60" cy="60" r="${RR}" fill="none" stroke="${LVCOL[lv]}" stroke-width="10"
        stroke-linecap="round" stroke-dasharray="${C}" stroke-dashoffset="${C * (1 - pct)}"
        transform="rotate(-90 60 60)"/>
      <text x="60" y="56" text-anchor="middle" fill="var(--ink)"
        font-size="${txt.length > 4 ? 16 : 21}" font-family="var(--mono)" font-weight="600">${txt}</text>
      <text x="60" y="73" text-anchor="middle" fill="var(--ink-3)" font-size="9"
        letter-spacing="1">PERCENTILE</text>`;
  }

  function detTabs() {
    Q('#jDetTabs').innerHTML = R.detectors.map(d =>
      `<button class="tab ${d.key === cur ? 'on' : ''}" data-k="${d.key}">${d.name}
         <b style="opacity:.6;font-weight:500;font-family:var(--mono);font-size:11px">${pc(d.fpr)}</b></button>`).join('');
    Q('#jDetTabs').querySelectorAll('[data-k]').forEach(b =>
      b.onclick = () => { cur = b.dataset.k; detTabs(); render(); });
  }

  function render() {
    const S = STAGES(), st = S[stage], d = R.detectors.find(x => x.key === cur);

    Q('#jDetNote').innerHTML = st.shared
      ? `Stages 1–4 are identical for all seven detectors — they are the fixed signal
         processing, before any learned weight is involved.`
      : `Showing the <strong>${d.name}</strong> detector. Stages 5–8 differ per detector,
         because each has its own trained mask and reference.`;

    Q('#jChips').innerHTML = S.map((x, i) =>
      `<button class="chip ${i === stage ? 'on' : ''}" data-i="${i}"><b>${i + 1}</b>${x.n}</button>`).join('');
    Q('#jChips').querySelectorAll('[data-i]').forEach(b =>
      b.onclick = () => { stop(); stage = +b.dataset.i; render(); });
    Q('#jPrev').disabled = stage === 0;
    Q('#jNext').disabled = stage === S.length - 1;
    Q('#jPos').textContent = `${stage + 1} / ${S.length}`;

    const view = Q('#jView');
    if (st.panels.length) {
      view.className = 'stageview n' + st.panels.length;
      view.innerHTML = st.panels.map(([src, cap]) =>
        `<figure><img src="${src}" style="width:100%;border-radius:var(--r-sm);display:block">
         <figcaption>${cap}</figcaption></figure>`).join('');
      if (st.svg) view.innerHTML += `<div style="grid-column:1/-1;margin-top:6px">${st.svg()}</div>`;
    } else {
      view.className = 'stageview solo';
      view.innerHTML = st.svg ? st.svg() : '';
    }

    const kindCls = st.kind.startsWith('LEARNED') ? 'k-learn'
                  : st.kind.startsWith('MEASURED') ? 'k-cal' : 'k-fix';
    Q('#jDoc').innerHTML = `
      <div class="dochead"><span class="stepno">${stage + 1}</span><h3>${st.t}</h3></div>
      <div class="kinds"><span class="kpill ${kindCls}">${st.kind}</span>
        <span class="kpill k-shape">${st.shape}</span></div>
      <div class="eq">${st.eq}</div>
      <p>${st.what}</p>`;

    Q('#jFoot').innerHTML = st.foot ? st.foot() : '';
    Q('#jFoot').classList.toggle('hide', !st.foot);

    const [lab, lv] = [d.verdict.label, d.verdict.level];
    Q('#jLive').innerHTML = `
      <div class="cell"><span>detector</span><b style="font-size:14px">${d.name}</b></div>
      <div class="cell"><span>raw score</span><b>${stage >= 6 ? d.score.toFixed(5) : '—'}</b></div>
      <div class="cell"><span>false-positive rate</span><b>${stage >= 7 ? pc(d.fpr) : '—'}</b></div>
      <div class="cell lv${lv}"><span>this detector says</span>
        <b style="color:${stage >= 7 ? 'var(--v)' : 'var(--ink-3)'}">${stage >= 7 ? lab : '—'}</b></div>`;
  }

  function stop() { if (timer) { clearInterval(timer); timer = null; Q('#jPlay').textContent = '▶ Play'; } }
  function wire() {
    const n = STAGES().length;
    Q('#jPrev').onclick = () => { stop(); if (stage > 0) { stage--; render(); } };
    Q('#jNext').onclick = () => { stop(); if (stage < n - 1) { stage++; render(); } };
    Q('#jPlay').onclick = () => {
      if (timer) return stop();
      Q('#jPlay').textContent = '⏸ Pause';
      timer = setInterval(() => {
        if (stage >= n - 1) return stop();
        stage++; render();
      }, 4200);
    };
    document.addEventListener('keydown', e => {
      if (!R || e.target.tagName === 'INPUT') return;
      if (e.key === 'ArrowRight' && stage < n - 1) { stop(); stage++; render(); }
      if (e.key === 'ArrowLeft' && stage > 0) { stop(); stage--; render(); }
    });
  }

  /* ---------- entry point ---------------------------------------------- */
  window.renderVerdict = function (s) {
    stop();
    const badge = Q('#judgeStatus');
    if (!s || !s.available) { badge.textContent = 'no image uploaded yet'; return stateNone(); }
    if (s.scoring) { badge.textContent = 'running the detectors…'; return stateScoring(); }
    if (s.result) {
      badge.textContent = `seven checkpoints · ${s.result.device}`;
      return stateResult(s);
    }
    badge.textContent = 'not scored';
    return stateNote(s.note || 'The detectors did not return a result for this image.');
  };
})();
