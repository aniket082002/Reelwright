import json, shutil, sqlite3, threading, uuid
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageOps
from renderer import render, Cancelled

ROOT = Path(__file__).parent / "data"; ROOT.mkdir(exist_ok=True)
db = sqlite3.connect(ROOT / "app.db", check_same_thread=False)
db.execute("create table if not exists projects(id text primary key, config text)")
EXT = {".jpg", ".jpeg", ".png", ".webp"}

class Clip(BaseModel):
    id: str; photo: str; duration: float = 3.0; movement: str = "zoom_in"; zoom_pct: float = 12
    start: list[float] = [0, 0, 0]; end: list[float] = [0, 0, 12]; easing: str = "ease_in_out"; speed: float = 1.0
    transition: str = "crossfade"; transition_duration: float = .8
    brightness: float = 0; contrast: float = 1; saturation: float = 1

class Config(BaseModel):
    mode: str = "auto"; style: str = "cinematic"; aspect: Literal["9:16", "16:9", "1:1", "4:5"] = "9:16"
    resolution: Literal["720p", "1080p", "1440p", "2160p"] = "1080p"; fps: Literal[24, 30, 60] = 30
    quality: Literal["standard", "high"] = "high"; total: float = 30
    photos: list[dict] = []; clips: list[Clip] = []

app = FastAPI(); JOBS: dict = {}

def pdir(pid): d = ROOT / pid; (d / "originals").mkdir(parents=True, exist_ok=True); (d / "web").mkdir(exist_ok=True); return d
def load(pid):
    r = db.execute("select config from projects where id=?", (pid,)).fetchone()
    if not r: raise HTTPException(404, "Project not found")
    return json.loads(r[0])
def save(pid, cfg): db.execute("insert or replace into projects values(?,?)", (pid, json.dumps(cfg))); db.commit()

@app.post("/api/projects")
def create():
    pid = uuid.uuid4().hex[:10]; pdir(pid); save(pid, Config().model_dump()); return {"id": pid}

@app.get("/api/projects/{pid}")
def get(pid: str): return load(pid)

@app.put("/api/projects/{pid}")
def put(pid: str, cfg: Config): load(pid); save(pid, cfg.model_dump()); return {"ok": True}

@app.post("/api/projects/{pid}/photos")
async def upload(pid: str, files: list[UploadFile] = File(...)):
    load(pid); d = pdir(pid); out = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in EXT: raise HTTPException(400, f"{f.filename}: use JPG, PNG or WEBP")
        phid = uuid.uuid4().hex[:8]; dst = d / "originals" / (phid + ext)
        with open(dst, "wb") as o: shutil.copyfileobj(f.file, o)
        try:
            Image.open(dst).verify(); im = ImageOps.exif_transpose(Image.open(dst)); w, h = im.size
            web = im.convert("RGB"); web.thumbnail((1600, 1600)); web.save(d / "web" / f"{phid}.jpg", quality=90)  # preview copy only
        except Exception:
            dst.unlink(missing_ok=True); raise HTTPException(400, f"{f.filename} is not a valid image")
        dst.chmod(0o444)  # originals are read-only from here on
        out.append({"id": phid, "name": f.filename, "w": w, "h": h})
    return out

@app.post("/api/projects/{pid}/render")
def start(pid: str):
    cfg = Config(**load(pid)).model_dump()
    if JOBS.get(pid, {}).get("state") == "running": raise HTTPException(409, "Already rendering")
    job = JOBS[pid] = {"state": "running", "progress": 0.0, "cancel": False, "error": None}
    out = ROOT / pid / "output.mp4"
    def run():
        try:
            render(cfg, ROOT / pid, out, lambda p: job.update(progress=p), lambda: job["cancel"]); job["state"] = "done"
        except Cancelled: job["state"] = "cancelled"; out.unlink(missing_ok=True)
        except Exception as e: job.update(state="error", error=str(e))
    threading.Thread(target=run, daemon=True).start(); return job

@app.get("/api/projects/{pid}/render")
def status(pid: str): return JOBS.get(pid, {"state": "idle", "progress": 0})

@app.delete("/api/projects/{pid}/render")
def cancel(pid: str):
    if pid in JOBS: JOBS[pid]["cancel"] = True
    return {"ok": True}

@app.get("/api/projects/{pid}/video")
def video(pid: str):
    f = ROOT / pid / "output.mp4"
    if not f.exists(): raise HTTPException(404)
    return FileResponse(f, media_type="video/mp4", filename="reel.mp4")

app.mount("/files", StaticFiles(directory=ROOT), name="files")
app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "frontend", html=True), name="ui")
