# MaskSim Explorer

An interactive front end for the MaskSim detector in this repository.

Two pages:

* **`/` — Detect.** Upload an image and see the seven per-generator detectors run on it,
  with every intermediate stage of the pipeline drawn from the real tensors.
* **`/learn` — How it works.** Seven steps that build the frequency-domain argument from
  scratch, **all computed on whatever image you last uploaded**, ending in a verdict on
  that image from the seven real checkpoints. Steps 1–5 and 7 run entirely in the browser
  with their own FFT in JavaScript; Step 6 is the server's real analysis.

The two pages share an image. Upload on either one and the other follows: the detect page
gets a *"take this image apart in How it works"* button, and the explainer gets a source
bar with the thumbnail, how the tile was taken, and what the seven real detectors said
about it. Uploading from the explainer also runs the detectors, if the image is at least
512×512 and the checkpoints are loaded — one upload, both pages.

### Why the default image is a 1978 photograph

The fallback is the standard *cameraman* tile — deliberately a photograph, and deliberately
older than every generative model. Steps 1–5 teach what a spectrum *is*, and for that you
want a picture with nothing hidden in it, so that when Step 5 plants a fingerprint you know
for certain the dots that appear were put there by you. It is a control, not a specimen.

Upload an image and it becomes the subject of every figure on the page — and Step 6 tells
you whether it is synthetic.

### The seven steps

1. **One pair of dots = one stripe pattern.** Two sliders, a synthetic wave and its
   spectrum. The vocabulary: distance from centre is how fast, direction is which way.
2. **Your image, in the frequency domain.** Sweep the pointer over your image's spectrum.
   The panel beside it draws the single stripe pattern that bin stands for, *at the true
   amplitude and phase it has in your picture* — and tells you its period in pixels, its
   angle, its amplitude in grey levels, and how it ranks among all 65,536 bins. Click to
   lock. A crosshair marks the bin, a second ring marks its conjugate mirror, a dashed
   circle marks the radius, and the radial-energy chart underneath shows where you are.
   Toggle to the **residual spectrum** to watch the scene vanish and the flat,
   fingerprint-bearing field appear — that second picture is the one the detector uses.
3. **Your image is just a sum of those stripes.** Rebuild it from the strongest N% of
   coefficients.
4. **Content in the middle, fingerprints outside.** Split the spectrum at a radius.
5. **The artifact you cannot see.** Plant a periodic gain in *your* image. Includes a
   blind test: a button hides the labels and shuffles the two panels, so the room has to
   guess which one is fake before you reveal it.
6. **Your image, through the real detector.** The seven trained checkpoints score it, and
   the whole chain of evidence is laid out: the verdict with its false-positive rate, the
   four pipeline stages the model actually computed at full 512×512, a tab per detector
   with its similarity map / trained mask / your masked spectrum, and all seven FPRs with
   an explanation of why the answer is a percentage rather than a yes/no. These are the
   real numbers — identical to what `detect_one_image.py` prints.
7. **The detector lab** — a working miniature MaskSim, described below. Step 6 gives the
   answer; Step 7 shows how an answer like that is arrived at.

### The detector lab (Step 7)

Built for presenting. When the page loads it *actually trains a detector on your image*:
it makes 28 variants of it (14 left alone, 14 with a planted five-frequency fingerprint),
fits a mask from the per-bin signal-to-noise ratio between the two classes, derives a
reference pattern, fits the logistic, and scores 60 held-out variants to build a
calibration table. Nothing on the panel is pre-rendered, and it all refits when you change
the source image.

**Say this out loud before quoting a lab number:** the lab plants a fingerprint of its own
invention and learns to spot *that*. It is a teaching model of the method, not a verdict on
your picture. The verdict is on the detect page. The page says so too, in a red-bordered
note above the panel.

You then walk eight stages — input, denoise, Fourier, *where the mask comes from*, apply
the mask, compare with the reference, logistic, calibration — with Back/Next, a Play
button, arrow keys, or by clicking any stage directly. Each stage shows the real tensors
at that point, alongside what it does, why it matters, the equation, the tensor shape, and
whether that step is learned or fixed DSP.

Four images can be fed to it, and the comparison is the point of the demo:

| source | what happens |
|---|---|
| your image, untouched | no evidence — it sits in the middle of the calibration histogram |
| + fingerprint planted | caught outright, FPR 0% |
| + fingerprint → JPEG q60 | still caught; the artifact lives at frequencies JPEG keeps |
| + fingerprint → resized 125% | **scores like an untouched photograph** |

That last row is the honest headline of the whole method, and it is worth showing on a
projector: rescaling slides every peak off the bins the mask is watching, and the detector
goes silent with no sign that it has failed. A second slider sets how strong the
fingerprint is in the test image — take it below about a third of a grey level and the
detector goes blind on all of them.

The lab is a miniature, and it says so on the page: 256×256 single channel instead of
3×512×512, a 3×3 high-pass standing in for the 17-layer DnCNN, and a closed-form mask
instead of 20 epochs of gradient descent. The quantity it computes is the same one
training converges to. `window.__lab()` in the browser console dumps the current numbers
if you want them mid-demo.

---

## Run it

From the repository root, with the same virtual environment you use for
`detect_one_image.py`:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r webapp\requirements.txt
python webapp\server.py
```

Use `python -m pip`, not bare `pip`. This virtual environment was created while the
folder was still called `masksim`, and `pip.exe` is a launcher stub with the old
interpreter path baked in, so it dies with *"Fatal error in launcher: Unable to create
process using `C:\Users\aakar\masksim\.venv\Scripts\python.exe`"*. Going through
`python -m` skips the stub entirely. (Recreating the venv would also fix it.)

Then open <http://127.0.0.1:8000>.

`.\webapp\run.ps1` does the same three steps and opens the browser for you.

The server loads all seven checkpoints at startup — that takes 20–60 seconds on CPU and
prints one line per detector. Wait for `Open http://127.0.0.1:8000` before uploading.

### What it needs

| | |
|---|---|
| `checkpoints/JPEG_Qrandom_w512/newsynth/*.ckpt` | the seven trained detectors |
| `calibration/*.csv` | the 15 score tables, including `all_to_max.csv` |
| `real_img.png`, `fake_img.png` | optional — the two "try a sample" buttons |

If a checkpoint is missing the server prints `[warn] no checkpoint for <name>` and carries
on with the rest; if all seven are missing it refuses to start.

---

## The numbers are the repo's numbers

`webapp/analysis.py` is a wrapper, not a reimplementation. It calls the repository's own
code for every quantity it displays:

* preprocessing — `MaskSimDataset(mode="test").process()`, the same 8-px-aligned centre
  crop into YCbCr;
* the raw score — `MaskSim.compute_probs()`;
* the similarity map — `MaskSim.compute_similarity_map()`, averaged over batch and
  channel and grey-dilated by 5, with the same per-generator colour range
  `detect_one_image.py` uses, so the map in the browser is the same picture as
  `sim_map_<generator>.html`;
* the false-positive rate — `searchsorted` into the concatenation of
  `coco_unlabled_2017_to_<g>.csv` and `FlickrFace_to_<g>.csv`, column `score`;
* the **generic model** — the maximum of the seven *calibrated* scores, re-calibrated a
  second time against `all_to_max.csv`. This is the two-stage procedure in
  `detect_one_image.py`, not a maximum of raw scores.

So `python detect_one_image.py -i x.png` and the web page always agree. If they ever
disagree, the wrapper is wrong and the wrapper is what should be fixed.

**One optimisation.** The DnCNN denoiser is the expensive part, and running it once per
detector means 14 passes per upload. At startup the engine fingerprints each checkpoint's
`preproc` weights; if all seven carry the identical frozen denoiser it computes the
residual once and reuses it, which is the difference between roughly a minute and roughly
five seconds per image on CPU. If the fingerprints differ it silently falls back to the
slow, always-correct path. The startup log says which: `denoiser shared across
detectors: True/False`.

---

## Reading the detect page

**The verdict ring** shows the generic model's percentile — where this image sits in the
distribution of scores from real photographs. The big number on the right is the
false-positive rate, which is the same thing stated as an error rate: *"if I call this
image synthetic, how often would that be a false alarm?"*

**Bar chart.** Bar length is `1 − FPR`, so longer means more suspicious; the number
printed at the right of each row is the FPR itself. The dashed lines are the paper's
three operating points, 10% / 5% / 1%.

**Raw scores are not comparable across detectors.** Each was trained separately and ends
in its own logistic layer, so a 0.9 from one and a 0.9 from another mean different things.
That is exactly why the calibration tables exist. Compare the FPR column.

**Frequency explorer.** Three layers per detector:

* *Similarity map* — per-frequency contribution to the cosine score. Bright means that bin
  agrees with this generator's reference pattern. The whole map sums to the score.
* *Learned mask* — the trained weights, `sigmoid` of a smoothed 3×478×478 parameter
  tensor. Static: the same map for every image you ever upload.
* *Masked spectrum* — your image's spectrum after the mask is applied. Only this reaches
  the comparison.

**A negative result is not proof of authenticity.** Each detector answers one question —
"does this carry *this* generator's fingerprint?" — so "no evidence" means none of these
seven left a trace, which also happens when the image was made by something else, or
resized, or re-compressed hard.

---

## Files

```
webapp/
  server.py            FastAPI app: /api/status, /api/detect, /api/sample/{real|fake},
                       /api/source (GET/POST/DELETE — the image the two pages share)
  analysis.py          model loading, calibration, all figure generation
  requirements.txt     fastapi, uvicorn, python-multipart, pillow, matplotlib
  run.ps1              install + launch + open browser
  static/
    index.html         detect page
    app.js             verdict ring, stage strip, charts, frequency explorer (no libraries)
    learn.html         the explainer
    learn.js           radix-2 FFT, canvas helpers, the source bar, interactive steps 1–5
    verdict.js         step 6: the real detectors' verdict on your image
    lab.js             step 7: the miniature detector — training, scoring, all eight stages
    sample_data.js     the fallback photograph, inlined as a data URI
    styles.css         dark theme; palette checked for colour-vision deficiency
```

No JavaScript dependencies and no CDN — the pages work offline. The only server
dependencies beyond what the repo already needs are FastAPI, uvicorn and
python-multipart.

---

## Troubleshooting

**`No checkpoints found under …`** — `checkpoints.zip` was downloaded but never extracted,
or extracted one directory too deep. The `.ckpt` files must sit directly in
`checkpoints/JPEG_Qrandom_w512/newsynth/`.

**Upload rejected with "must be at least 512×512"** — MaskSim takes a fixed 512×512 centre
crop; there is no resizing, because resizing moves the fingerprint off the frequency bins
the mask examines.

**The terminal looks frozen** — on Windows, clicking inside a console window enters
QuickEdit selection mode, which pauses output and makes Ctrl+C mean "copy". Press Esc,
then Ctrl+C.

**First upload is slow, the rest are fast** — PyTorch warms up its kernels on the first
forward pass.

**Upload on the explainer page does nothing, or says the page is newer than the server** —
the explainer needs the `/api/source` routes, which were added later. Stop the server with
Ctrl+C and start it again; a browser refresh alone is not enough, because the Python
process is what changed. The page detects this case and says so. `GET /api/status` reports
a `build` string you can compare.

**The explainer's verdict panel sits on "scoring…"** — that upload returns immediately with
the thumbnail and scores the image in the background, so every other figure redraws at once
while the seven detectors run. It polls once a second for up to three minutes. If it never
resolves, the server console has the traceback.

**`Fatal error in launcher: Unable to create process using …\masksim\.venv\…`** — see the
note under *Run it*: use `python -m pip`.
