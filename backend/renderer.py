"""Deterministic photo->MP4 renderer. Originals are only read; every frame is sampled from a processing copy in memory."""
import math, subprocess
from bisect import bisect_right
import cv2, numpy as np
from PIL import Image, ImageOps

RES = {"720p": 720, "1080p": 1080, "1440p": 1440, "2160p": 2160}
ASPECT = {"9:16": (9, 16), "16:9": (16, 9), "1:1": (1, 1), "4:5": (4, 5)}
# contrast, saturation, warmth, brightness, motion-speed multiplier (all subtle)
STYLES = {"cinematic": (1.08, .95, -.02, 0, .8), "travel": (1.05, 1.08, .03, 0, 1.0), "documentary": (1.03, .92, 0, 0, .7),
          "lifestyle": (1.04, 1.04, .02, 2, .9), "luxury": (1.10, .90, .01, -2, .6), "minimal": (1.0, .9, 0, 3, .6),
          "romantic": (.98, 1.0, .04, 4, .7), "vintage": (1.05, .82, .06, 0, .8), "modern": (1.06, 1.06, -.01, 0, 1.1),
          "custom": (1, 1, 0, 0, 1)}
EASE = {"linear": lambda t: t, "ease_in": lambda t: t * t, "ease_out": lambda t: 1 - (1 - t) ** 2,
        "ease_in_out": lambda t: t * t * (3 - 2 * t)}

class Cancelled(Exception): pass

def dims(cfg):
    s = RES[cfg["resolution"]]; aw, ah = ASPECT[cfg["aspect"]]
    w, h = (s, s * ah / aw) if aw < ah else (s * aw / ah, s) if aw > ah else (s, s)
    return int(round(w / 2) * 2), int(round(h / 2) * 2)

def motion(c):
    zp = c["zoom_pct"] / 100; z = 1 + zp; zz = max(z, 1.08); s, e = c["start"], c["end"]
    return {"zoom_in": (1, z, (0, 0), (0, 0)), "zoom_out": (z, 1, (0, 0), (0, 0)),
            "pan_left": (zz, zz, (1, 0), (-1, 0)), "pan_right": (zz, zz, (-1, 0), (1, 0)),
            "pan_up": (zz, zz, (0, 1), (0, -1)), "pan_down": (zz, zz, (0, -1), (0, 1)),
            "zoom_in_pan": (1, z, (-.6, .2), (.6, -.2)), "zoom_out_pan": (z, 1, (.6, -.2), (-.6, .2)),
            "parallax": (1, 1 + min(zp, .06), (-.3, 0), (.3, 0)),
            "custom": (1 + s[2] / 100, 1 + e[2] / 100, (s[0] / 100, s[1] / 100), (e[0] / 100, e[1] / 100))}[c["movement"]]

class Src:
    def __init__(self, path, W, H):
        a = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
        h, w = a.shape[:2]; cover = max(W / w, H / h)
        if cover < .5:  # keep 2x headroom for zoom, downscale with area averaging
            f = cover / .5; a = cv2.resize(a, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        self.a = np.ascontiguousarray(a); self.ih, self.iw = a.shape[:2]; self.s0 = max(W / self.iw, H / self.ih)

def sample(src, c, t, W, H, amt):
    z0, z1, p0, p1 = motion(c); e = EASE[c["easing"]](t) * amt
    z = min(z0 + (z1 - z0) * e, 1.6)
    px, py = p0[0] + (p1[0] - p0[0]) * e, p0[1] + (p1[1] - p0[1]) * e
    sc = src.s0 * z; cw, ch = W / sc, H / sc
    cx = src.iw / 2 + float(np.clip(px, -1, 1)) * (src.iw - cw) / 2   # clamped: never leaves the photo, no borders
    cy = src.ih / 2 + float(np.clip(py, -1, 1)) * (src.ih - ch) / 2
    M = np.float32([[sc, 0, -(cx - cw / 2) * sc], [0, sc, -(cy - ch / 2) * sc]])
    return cv2.warpAffine(src.a, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)

def grade(img, con, sat, warm, bri):
    if (con, sat, warm, bri) == (1, 1, 0, 0): return img
    f = img.astype(np.float32)
    if sat != 1:
        g = f @ np.float32([.2126, .7152, .0722]); f = g[..., None] + (f - g[..., None]) * sat
    if warm: f[..., 0] *= 1 + warm; f[..., 2] *= 1 - warm
    return np.clip((f - 128) * con + 128 + bri + .5, 0, 255).astype(np.uint8)

def blend(a, b, p, kind, W, H):
    q = p * p * (3 - 2 * p)
    if kind == "crossfade": return cv2.addWeighted(a, 1 - p, b, p, 0)
    if kind == "dissolve": return cv2.addWeighted(a, 1 - q, b, q, 0)
    if kind in ("fade_black", "fade_white"):
        s, w = (a, 1 - 2 * p) if p < .5 else (b, 2 * p - 1)
        return cv2.addWeighted(s, w, np.full_like(s, 0 if kind == "fade_black" else 255), 1 - w, 0)
    if kind in ("slide_left", "slide_right"):
        s = int(round(W * q)); o = np.empty_like(a)
        if kind == "slide_left": o[:, :W - s] = a[:, s:]; o[:, W - s:] = b[:, :s]
        else: o[:, s:] = a[:, :W - s]; o[:, :s] = b[:, W - s:]
        return o
    if kind == "zoom":
        k = 1 + .12 * q; M = np.float32([[k, 0, W / 2 * (1 - k)], [0, k, H / 2 * (1 - k)]])
        return cv2.addWeighted(cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), 1 - q, b, q, 0)
    if kind == "blur":
        sg = max(.1, 10 * math.sin(math.pi * p) * H / 1080)
        return cv2.addWeighted(cv2.GaussianBlur(a, (0, 0), sg), 1 - p, cv2.GaussianBlur(b, (0, 0), sg), p, 0)
    return b if p >= .5 else a

def timeline(clips):
    starts, trs, t = [], [], 0.0
    for i, c in enumerate(clips):
        starts.append(t)
        if i + 1 < len(clips):
            d = 0.0 if c["transition"] in ("cut", "none") else min(c["transition_duration"], c["duration"] * .5, clips[i + 1]["duration"] * .5)
            trs.append(d); t += c["duration"] - d
    return starts, trs, starts[-1] + clips[-1]["duration"]

def render(cfg, pdir, out, progress, cancelled):
    clips = cfg["clips"]
    if not clips: raise ValueError("Add at least one photo to the timeline.")
    W, H = dims(cfg); fps = int(cfg["fps"]); con, sat, warm, bri, spd = STYLES.get(cfg["style"], STYLES["custom"])
    starts, trs, total = timeline(clips); N = max(1, round(total * fps))
    paths = {p["id"]: next((pdir / "originals").glob(p["id"] + ".*")) for p in cfg["photos"]}
    hq = cfg["quality"] == "high"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-an", "-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709:flags=lanczos,format=yuv420p",
           "-c:v", "libx264", "-preset", "slow" if hq else "fast", "-crf", "15" if hq else "20", "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags", "+faststart", str(out)]
    log = open(str(out) + ".log", "wb"); proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
    cache = {}
    def frame(j, T):
        c = clips[j]
        if j not in cache: cache[j] = Src(paths[c["photo"]], W, H)
        t = min(max((T - starts[j]) / c["duration"], 0), 1)
        img = sample(cache[j], c, t, W, H, min(c["speed"] * spd, 1.5))
        return grade(img, con * c["contrast"], sat * c["saturation"], warm, bri + c["brightness"])
    try:
        for k in range(N):
            if cancelled(): raise Cancelled()
            T = k / fps; i = max(0, bisect_right(starts, T + 1e-9) - 1)
            for old in [x for x in cache if x < i - 1]: del cache[old]
            f = frame(i, T)
            if i > 0 and trs[i - 1] > 0 and T < starts[i] + trs[i - 1]:
                f = blend(frame(i - 1, T), f, (T - starts[i]) / trs[i - 1], clips[i - 1]["transition"], W, H)
            proc.stdin.write(np.ascontiguousarray(f).tobytes())
            if k % 5 == 0: progress(k / N)
        proc.stdin.close()
        if proc.wait() != 0: raise RuntimeError("FFmpeg failed; see " + str(out) + ".log")
        progress(1.0)
    except BaseException:
        proc.kill(); raise
