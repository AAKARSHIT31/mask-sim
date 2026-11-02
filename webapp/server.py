"""MaskSim Explorer — local web app.

Run from inside the masksim repo:
    python webapp/server.py
then open http://127.0.0.1:8000
"""
import sys, tempfile, threading, time, traceback, uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from analysis import Engine, learn_thumbnail, CLASS_NAMES, NICE, IMG_SIZE   # noqa: E402

BUILD = "2026-09-29.1"          # bumped whenever the API changes, so the pages can
                                # tell the user to restart a stale server
NOCACHE = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
app = FastAPI(title="MaskSim Explorer")
ENGINE = None

# The image the two pages share. The explainer teaches on it; the detect page scores it.
LOCK = threading.Lock()
LAST = {"name": None, "image": None, "how": None, "native": None,
        "width": None, "height": None, "scoring": False, "note": None,
        "result": None, "stamp": 0}


@app.on_event("startup")
def _boot():
    global ENGINE
    print("\nLoading MaskSim detectors …")
    ENGINE = Engine()
    print(f"Ready — {len(ENGINE.models)} detectors on "
          f"{ENGINE.models and next(iter(ENGINE.models.values())).device}")
    print("Open http://127.0.0.1:8000\n")


# ---------------------------------------------------------------- helpers
def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    tmp = Path(tempfile.gettempdir()) / f"masksim_{uuid.uuid4().hex[:10]}{suffix}"
    tmp.write_bytes(file.file.read())
    return tmp


def _slim(result):
    """Everything the explainer needs, which is everything the detect page shows."""
    if not result:
        return None
    return {k: result[k] for k in
            ("file", "dimensions", "device", "stages", "detectors", "maps",
             "generic", "radial")}


def _remember(path, name, result=None, scoring=False, note=None):
    try:
        t = learn_thumbnail(path)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(400, f"Could not read that image: {e}")
    with LOCK:
        LAST.update(name=name, scoring=scoring, note=note,
                    result=_slim(result), stamp=time.time(), **t)


def _score_in_background(path: Path, name: str):
    """Runs after the upload response has already been sent."""
    note = None
    result = None
    try:
        result = ENGINE.analyse(path)
    except ValueError as e:
        note = str(e)                       # e.g. smaller than 512x512
    except Exception as e:
        traceback.print_exc()
        note = f"{type(e).__name__}: {e}"
    with LOCK:
        if LAST["name"] == name:            # ignore if a newer image arrived meanwhile
            LAST.update(scoring=False, note=note, result=_slim(result), stamp=time.time())
    try:
        path.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------- api
@app.get("/api/status")
def status():
    if ENGINE is None:
        return {"ready": False, "build": BUILD}
    return {"ready": True, "build": BUILD, "crop": IMG_SIZE,
            "detectors": [{"key": k, "name": NICE[k], **ENGINE.meta[k]} for k in ENGINE.models],
            "samples": [n for n in ("real_img.png", "fake_img.png") if (REPO / n).exists()]}


@app.post("/api/detect")
def detect(file: UploadFile = File(...)):
    """Synchronous: the detect page waits for the answer, with a spinner."""
    if ENGINE is None:
        raise HTTPException(503, "Model still loading, try again in a moment.")
    tmp = _save_upload(file)
    try:
        out = ENGINE.analyse(tmp)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    finally:
        pass
    _remember(tmp, file.filename or tmp.name, out)
    try:
        tmp.unlink()
    except Exception:
        pass
    return JSONResponse(out)


@app.post("/api/sample/{which}")
def sample(which: str):
    if ENGINE is None:
        raise HTTPException(503, "Model still loading.")
    p = REPO / ("real_img.png" if which == "real" else "fake_img.png")
    if not p.exists():
        raise HTTPException(404, f"{p.name} not found in the repo root.")
    try:
        out = ENGINE.analyse(p)
    except Exception as e:
        traceback.print_exc(); raise HTTPException(500, str(e))
    _remember(p, p.name, out)
    return JSONResponse(out)


@app.get("/api/source")
def get_source():
    """What the explainer is teaching on, and the detectors' verdict once it lands."""
    with LOCK:
        return {"available": LAST["image"] is not None, "build": BUILD, **LAST}


@app.post("/api/source")
def set_source(background: BackgroundTasks, file: UploadFile = File(...)):
    """Upload from the explainer page.

    Returns as soon as the thumbnail is ready so every canvas on the page can
    redraw immediately, then scores the image with the seven checkpoints in the
    background. The page polls GET /api/source for the verdict.
    """
    tmp = _save_upload(file)
    name = file.filename or tmp.name
    eligible = ENGINE is not None
    _remember(tmp, name, None, scoring=eligible,
              note=None if eligible else "The detectors are not loaded.")
    if eligible:
        background.add_task(_score_in_background, tmp, name)
    with LOCK:
        return {"available": True, "build": BUILD, **LAST}


@app.delete("/api/source")
def clear_source():
    with LOCK:
        for k in LAST:
            LAST[k] = None
        LAST["scoring"] = False
        LAST["stamp"] = time.time()
    return {"available": False, "build": BUILD}


class NoCacheStatic(StaticFiles):
    """Serve the front-end with no-store.

    The pages are edited during development; a browser that caches learn.js or
    verdict.js will happily run yesterday's script against today's HTML, which
    shows up as `Cannot set properties of null`.  Correctness beats a few
    milliseconds of load time for a localhost tool.
    """
    def is_not_modified(self, *a, **k):
        return False

    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        r.headers["Pragma"] = "no-cache"
        for h in ("etag", "last-modified"):
            if h in r.headers:
                del r.headers[h]
        return r


app.mount("/static", NoCacheStatic(directory=HERE / "static"), name="static")


@app.get("/")
def index(): return FileResponse(HERE / "static" / "index.html", headers=NOCACHE)


@app.get("/learn")
def learn(): return FileResponse(HERE / "static" / "learn.html", headers=NOCACHE)


if __name__ == "__main__":
    import uvicorn
    print(f"MaskSim Explorer build {BUILD}")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
