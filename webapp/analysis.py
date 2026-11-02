"""All DSP + model introspection for the MaskSim Explorer web app.

This module is a *wrapper*, not a re-implementation. Every number the browser shows
is produced by the original repo's own code paths - the same MaskSimDataset
preprocessing, the same MaskSim.compute_probs(), the same calibration CSVs, and the
same two-stage generic model - so the web app and `python detect_one_image.py -i x.png`
always agree.
"""
import base64, glob, io, os, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]          # the masksim repo root
sys.path.insert(0, str(REPO))
from src.masksim import MaskSim                      # noqa: E402
from src.dataset import MaskSimDataset               # noqa: E402

CLASS_NAMES = ["sd1", "sd2", "sdxl", "dalle2", "dalle3", "midjourney", "firefly"]
NICE = {"sd1": "Stable Diffusion 1", "sd2": "Stable Diffusion 2", "sdxl": "Stable Diffusion XL",
        "dalle2": "DALL·E 2", "dalle3": "DALL·E 3", "midjourney": "Midjourney",
        "firefly": "Firefly"}

# exactly the ranges detect_one_image.py uses, as [vmax, vmin]
VISUALIZATION_RANGE = {
    "sd1": [1.8e-04, 3.8e-05], "sd2": [1.1e-04, 3.6e-05], "sdxl": [1.2e-04, 3.6e-05],
    "dalle2": [6.2e-05, 3.9e-05], "dalle3": [9.8e-05, 4.8e-05],
    "midjourney": [6.6e-05, 2.6e-05], "firefly": [4.8e-04, 1.4e-04],
}

CKPT_DIR = REPO / "checkpoints" / "JPEG_Qrandom_w512" / "newsynth"
CALIB_DIR = REPO / "calibration"
IMG_SIZE = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- image helpers
def _cmap(name):
    try:
        return matplotlib.colormaps[name]
    except Exception:                                   # matplotlib < 3.6
        import matplotlib.cm as cm
        return cm.get_cmap(name)


def _to_png_b64(arr, cmap="magma", vmin=None, vmax=None, size=340):
    """float 2-D array -> base64 PNG data URI, colour-mapped."""
    from PIL import Image
    a = np.asarray(arr, dtype=float)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(a, 0.5)) if vmin is None else float(vmin)
    hi = float(np.percentile(a, 99.5)) if vmax is None else float(vmax)
    a = np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)
    rgb = (_cmap(cmap)(a)[..., :3] * 255).astype(np.uint8)
    im = Image.fromarray(rgb)
    if size:
        im = im.resize((size, size), Image.BILINEAR)
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _gray_b64(arr, size=340):
    from PIL import Image
    a = np.nan_to_num(np.asarray(arr, dtype=float))
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    a = np.clip((a - lo) / (hi - lo + 1e-12), 0, 1)
    im = Image.fromarray((a * 255).astype(np.uint8))
    if size:
        im = im.resize((size, size), Image.BILINEAR)
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _rgb_b64(arr_uint8, size=340):
    from PIL import Image
    im = Image.fromarray(np.asarray(arr_uint8, dtype=np.uint8))
    if size:
        im = im.resize((size, size), Image.BILINEAR)
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- calibration
# Same files, same concatenation, same "score" column as detect_one_image.py.
NEG_SCORE_FILES = {c: [f"coco_unlabled_2017_to_{c}.csv", f"FlickrFace_to_{c}.csv"]
                   for c in CLASS_NAMES}
NEG_SCORE_FILES["max"] = ["all_to_max.csv"]


def _load_calibration(key):
    """Sorted detector scores on KNOWN-REAL photographs (COCO + FlickrFaces).
    An image's FPR is its rank in this list: the fraction of real photos that
    would score at least this high, i.e. the false-alarm rate you accept if you
    call this image fake."""
    import pandas as pd
    vals = []
    for name in NEG_SCORE_FILES[key]:
        p = CALIB_DIR / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        col = "score" if "score" in df.columns else df.columns[-1]
        vals.extend(pd.to_numeric(df[col], errors="coerce").dropna().tolist())
    if not vals:
        return None
    return np.sort(np.asarray(vals, dtype=float))


def _quantile(sorted_neg, score):
    """(calibrated score, fpr) - identical arithmetic to detect_one_image.py."""
    if sorted_neg is None or len(sorted_neg) == 0:
        return None, None
    q = int(np.searchsorted(sorted_neg, score))
    calib = q / len(sorted_neg)
    return calib, 1.0 - calib


def verdict_for(fpr):
    """The reference implementation refuses a fixed 0.5 threshold and reports an
    error rate instead. Same three bands here: 1% / 5% / 10% false-positive rate."""
    if fpr is None:
        return {"label": "uncalibrated", "level": 0}
    if fpr < 0.01:
        return {"label": "very likely synthetic", "level": 3}
    if fpr < 0.05:
        return {"label": "likely synthetic", "level": 2}
    if fpr < 0.10:
        return {"label": "possibly synthetic", "level": 1}
    return {"label": "no evidence", "level": 0}


# ---------------------------------------------------------------- shared residual
class _CachedPreproc(torch.nn.Module):
    """Stands in for a frozen DnCNN so the 17-layer denoiser runs once per upload
    instead of 14 times. Only used when every checkpoint reports a frozen preproc."""

    def __init__(self, residual):
        super().__init__()
        self.residual = residual

    def forward(self, x):
        return self.residual


# ---------------------------------------------------------------- the engine
class Engine:
    def __init__(self):
        self.models, self.calib, self.meta = {}, {}, {}
        for cls in CLASS_NAMES:
            hits = sorted(glob.glob(str(CKPT_DIR / f"{cls}*.ckpt")), key=os.path.getctime)
            if not hits:
                print(f"  [warn] no checkpoint for {cls}")
                continue
            m = MaskSim.load_from_checkpoint(hits[-1], map_location=DEVICE).float().eval()
            self.models[cls] = m.to(DEVICE)
            self.calib[cls] = _load_calibration(cls)
            self.meta[cls] = {"checkpoint": Path(hits[-1]).name,
                              "n_calibration": 0 if self.calib[cls] is None else len(self.calib[cls])}
            print(f"  loaded {cls:11s} {Path(hits[-1]).name}")
        if not self.models:
            raise RuntimeError(f"No checkpoints found under {CKPT_DIR}")

        self.calib_max = _load_calibration("max")
        self.ds = MaskSimDataset(img_dir_real_list=[], img_dir_fake_list=[],
                                 img_size=IMG_SIZE, channels=3,
                                 color_space="YCbCr", mode="test")

        # Reusing one DnCNN pass for all 7 detectors is a 14x speed-up, but it is only
        # correct if every checkpoint carries the SAME frozen denoiser. Verify that by
        # fingerprinting each preproc's weights rather than trusting the hyper-parameters.
        self.share_residual = self._preprocs_identical()

    def _preprocs_identical(self):
        def fingerprint(m):
            p = getattr(m, "preproc", None)
            if not isinstance(p, torch.nn.Module):
                return None
            sd = p.state_dict()
            if not sd:
                return None
            return tuple((k, float(v.double().sum()), tuple(v.shape)) for k, v in sorted(sd.items()))
        fps = [fingerprint(m) for m in self.models.values()]
        ok = fps[0] is not None and all(f == fps[0] for f in fps)
        print(f"  denoiser shared across detectors: {ok}")
        return ok

    # -- identical preprocessing to detect_one_image.py -------------------------
    def preprocess(self, path):
        return self.ds.process(str(path), label=None)   # (1,3,512,512) YCbCr in [0,1]

    def analyse(self, path):
        import skimage.io
        from scipy import ndimage

        imgs = self.preprocess(path)
        if imgs is None:
            raise ValueError(f"The image must be at least {IMG_SIZE}x{IMG_SIZE} pixels "
                             "and have 3 colour channels.")
        imgs = imgs.to(DEVICE)

        # the same centre crop the dataset took, for display
        raw = skimage.io.imread(str(path))[:, :, :3]
        h, w = raw.shape[:2]
        y1 = max(0, h // 2 - IMG_SIZE // 2) // 8 * 8
        x1 = max(0, w // 2 - IMG_SIZE // 2) // 8 * 8
        crop_rgb = raw[y1:y1 + IMG_SIZE, x1:x1 + IMG_SIZE]

        any_model = next(iter(self.models.values()))
        with torch.no_grad():
            residual = any_model.preproc(imgs)                       # (1,3,478,478)
            log_ffts = (torch.fft.fftshift(
                torch.fft.fft2(residual, norm="ortho", dim=(-1, -2)),
                dim=(-1, -2)).abs() + 1e-10).log()
        logspec = log_ffts[0].cpu().numpy()                          # (3,478,478)
        ycbcr = imgs[0].cpu().numpy()
        res = residual[0].cpu().numpy()

        stages = [
            {"key": "input", "title": "1 · Input crop",
             "sub": f"centre {IMG_SIZE}×{IMG_SIZE} of a {w}×{h} image",
             "img": _rgb_b64(crop_rgb),
             "note": "The detector never sees the whole picture — always a fixed "
                     f"{IMG_SIZE}×{IMG_SIZE} crop, aligned to an 8-pixel grid so it lands on "
                     "JPEG block boundaries. Frequency coordinates only mean anything at a fixed "
                     "sampling rate, so rescaling the image would move the fingerprint."},
            {"key": "ycbcr", "title": "2 · Luma channel (YCbCr)",
             "sub": "Y of YCbCr, the JPEG basis",
             "img": _gray_b64(ycbcr[0]),
             "note": "JPEG and WebP quantise in YCbCr. Working in the same basis keeps compression "
                     "artifacts in predictable channels instead of smearing them across all three."},
            {"key": "residual", "title": "3 · DnCNN residual",
             "sub": f"{res.shape[-1]}×{res.shape[-1]} — content removed",
             "img": _gray_b64(res[0]),
             "note": "The denoiser predicts the NOISE, not the clean image. What survives is sensor "
                     "noise plus synthesis artifacts — the scene itself is gone. 17 px are cropped "
                     "from each edge (a 17-layer 3×3 network has a 17-px invalid border), hence 478."},
            {"key": "spectrum", "title": "4 · log |DFT|",
             "sub": "DC at centre",
             "img": _to_png_b64(logspec[0], "magma"),
             "note": "Fourier magnitude of the residual, log-scaled, DC shifted to the middle. "
                     "Leftover content sits near the centre; a generator's periodic fingerprint sits "
                     "far out, at fixed coordinates that do not depend on the scene."},
        ]

        detectors, maps = [], {}
        for cls, model in self.models.items():
            swapped = None
            if self.share_residual:
                swapped, model.preproc = model.preproc, _CachedPreproc(residual)
            try:
                with torch.no_grad():
                    score = float(model.compute_probs(imgs).cpu().numpy().ravel()[0])
                    smap = model.compute_similarity_map(imgs).detach().cpu().mean(dim=(0, 1)).numpy()
                    mask = model.get_mask()[0].detach().cpu().numpy()          # (C,H,W)
                    masked = model.in_conv_fft(log_ffts)[0].cpu().numpy() * mask
            finally:
                if swapped is not None:
                    model.preproc = swapped

            calib, fpr = _quantile(self.calib[cls], score)
            detectors.append({
                "key": cls, "name": NICE[cls], "score": score,
                "calibrated": calib, "fpr": fpr, "verdict": verdict_for(fpr),
                "checkpoint": self.meta[cls]["checkpoint"],
                "n_calibration": self.meta[cls]["n_calibration"],
                "peak": float(np.max(smap)),
                "energy": float(np.sum(np.abs(smap))),
                "positive_fraction": float((smap > 0).mean()),
            })

            vmax, vmin = VISUALIZATION_RANGE.get(cls, [None, None])
            enhanced = ndimage.grey_dilation(smap, 5)     # same as detect_one_image.py
            maps[cls] = {
                "simmap": _to_png_b64(enhanced, "viridis", vmin=vmin, vmax=vmax),
                "mask":   _to_png_b64(mask[0], "Purples"),
                "masked": _to_png_b64(masked[0], "magma"),
            }

        # ---- generic model: max of the CALIBRATED scores, re-calibrated on all_to_max
        best = max(detectors, key=lambda d: (d["calibrated"] if d["calibrated"] is not None else -1))
        max_calib = best["calibrated"]
        g_calib, g_fpr = _quantile(self.calib_max, max_calib)
        generic = {"score": max_calib, "fpr": g_fpr, "verdict": verdict_for(g_fpr),
                   "best": best["name"], "best_key": best["key"],
                   "n_calibration": 0 if self.calib_max is None else len(self.calib_max)}

        detectors.sort(key=lambda d: (d["fpr"] if d["fpr"] is not None else 1.0))

        return {
            "file": Path(path).name,
            "dimensions": {"width": w, "height": h},
            "device": str(DEVICE),
            "shared_residual": self.share_residual,
            "stages": stages,
            "detectors": detectors,
            "maps": maps,
            "generic": generic,
            "radial": radial_profile(logspec[0]),
            "histogram": spectrum_histogram(logspec[0]),
        }


def learn_thumbnail(path, size=256):
    """A single-channel 256x256 tile of the image for the explainer page.

    Takes a NATIVE-RESOLUTION centre crop whenever the image is big enough, and only
    resizes when it is not. That matters: the whole lesson of the page is that resampling
    destroys the fingerprint, so resizing the very image used to demonstrate it would
    quietly remove the thing being demonstrated."""
    from PIL import Image
    im = Image.open(path).convert("L")
    w, h = im.size
    s = min(w, h)
    if s >= size:
        x0, y0 = (w - size) // 2, (h - size) // 2
        im = im.crop((x0, y0, x0 + size, y0 + size))
        how = f"native {size}×{size} centre crop of a {w}×{h} image — not resampled"
        native = True
    else:
        x0, y0 = (w - s) // 2, (h - s) // 2
        im = im.crop((x0, y0, x0 + s, y0 + s)).resize((size, size), Image.LANCZOS)
        how = (f"upscaled from a {s}×{s} crop — this image is smaller than {size} px, "
               "so it had to be resampled, which weakens any fingerprint")
        native = False
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
    return {"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            "how": how, "native": native, "width": w, "height": h}


def radial_profile(logspec, nbins=90):
    """Mean log-magnitude as a function of distance from DC. A real photograph falls
    smoothly (roughly 1/f); a periodic artifact shows up as a bump at one radius."""
    h, w = logspec.shape
    y = np.arange(h) - h // 2
    x = np.arange(w) - w // 2
    r = np.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
    rmax = min(h, w) // 2
    edges = np.linspace(0, rmax, nbins + 1)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, nbins - 1)
    counts = np.maximum(np.bincount(idx, minlength=nbins), 1)
    mean = np.bincount(idx, weights=logspec.ravel(), minlength=nbins) / counts
    return {"radius": (edges[:-1] / rmax).round(4).tolist(),
            "energy": np.round(mean, 4).tolist()}


def spectrum_histogram(logspec, bins=40):
    hist, edges = np.histogram(logspec.ravel(), bins=bins)
    return {"counts": hist.tolist(), "edges": np.round(edges, 3).tolist()}
