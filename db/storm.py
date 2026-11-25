"""Rain and wet weather for e1 / e2 / h3: long single rain strokes, splash ticks, falling-away drops, a small grass field with a travelling ring.
Every function is a pure function of the frame.    hook(id) -> the optional shots/<id>_fig.py module (owned by the lead), or None."""
import numpy as np, cv2, math, os, importlib, importlib.util
from . import W, H
from .paint import rgb, f32, smooth, mix, grid, fbm, ramp
from .ink import Cel, spl


def hook(name):
    """shots/<name>_fig.py may define  fig(f) -> premultiplied RGBA layer or None,  and  anchors(f) -> dict (see each shot)"""
    if os.environ.get('GO_NOFIG'): return None
    return importlib.import_module('shots.%s_fig' % name) if importlib.util.find_spec('shots.%s_fig' % name) else None


def call(mod, fn, *a):
    g = getattr(mod, fn, None) if mod is not None else None
    return g(*a) if g else None


def _h(*k):
    n = 2166136261
    for q in k: n = ((n ^ (int(q) & 0xffffffff)) * 16777619) & 0xffffffff
    n ^= n >> 15; n = (n * 2246822519) & 0xffffffff; n ^= n >> 13; return (n & 0xffffff) / 16777216.


def rain(f, n=60, seed=0, slope=.10, speed=70., L=(220, 520), w=1.6, col='#e8eef0', a=.5, cel=None):
    """long tapered strokes.  speed px / frame, slope = dx per dy; each stroke is one drawing that falls, then starts again elsewhere"""
    c = cel or Cel(W, H, ss=1); span = H + L[1] + 40
    for i in range(n):
        ln = L[0] + (L[1] - L[0]) * _h(seed, i, 1); v = speed * (.85 + .3 * _h(seed, i, 2)); tot = f * v + _h(seed, i, 3) * span; k = int(tot // span); y = tot % span - ln; x = -120 + (W + 240) * _h(seed, i, 4, k) + y * slope - H * slope * .5; ww = w * (.6 + .8 * _h(seed, i, 5))
        p, q = np.array([x, y]), np.array([x + ln * slope, y + ln]); m = p + (q - p) * .72; nn = np.array([1., -slope]) * ww * .5; c.fill(np.array([p, m + nn, q, m - nn]), col, a * (.55 + .45 * _h(seed, i, 6)))
    return c if cel is not None else c.rgba()


def splashes(cel, f, pts, seed=0, per=9, col='#eef2f3', r=9., lw=1.4):
    """ticks of water where rain meets ground: pts = (n,2) candidate points with scale in column 2"""
    for i, (x, y, s) in enumerate(pts):
        ph = (f + int(_h(seed, i, 1) * per)) % per
        if ph > 4 or _h(seed, i, (f + int(_h(seed, i, 1) * per)) // per, 2) > .55: continue
        R = r * s * (.5 + .28 * ph); up = r * s * (1.1 - .05 * (ph - 2) ** 2) * (1 - ph / 6.)
        for k, d in enumerate((-.95, -.45, .1, .55, 1.)): e = np.array([x + d * R, y - up * (.7 + .5 * _h(i, k))]); cel.line(np.array([(x + d * R * .45, y - up * .12), (x + d * R * .8, y - up * .6), e]), lw * s, col, (.1, .7), seed=k, alpha=.9 - .15 * ph)
        a = np.linspace(0, 2 * math.pi, 22); cel.line(np.c_[x + np.cos(a) * R * 1.3, y + np.sin(a) * R * .34], lw * .8 * s, col, (.05, .05), alpha=.55 * (1 - ph / 5), seed=3)
    return cel


def away(cel, f, n=46, seed=0, cx=960, cy=540, zhit=9.0, v=.42, r=.055, foc=900., hits=None, col='#dfe6ea', dark='#5d666b', start=.35):
    """rain seen from the cloud: drops leave the lens and fall away to a plane zhit ahead.  hits = (n,2) screen points at the plane.
    returns [(x, y, age in frames)] of the bursts alive on this frame"""
    out = []; per = int(zhit / v) + 14
    for i in range(n):
        t = f + int(_h(seed, i, 1) * per); k = t // per; a = t % per; z = start + a * v; X, Y = (_h(seed, i, k, 2) - .5) * 2, (_h(seed, i, k, 3) - .5) * 2
        if hits is not None and len(hits): hx, hy = hits[int(_h(seed, i, k, 4) * len(hits)) % len(hits)]
        else: hx, hy = cx + X * 900, cy + Y * 520
        lat = np.array([hx - cx, hy - cy]) * zhit / foc
        if z < zhit:
            p = lambda zz: np.array([cx, cy]) + lat * foc / zz
            a0, b0 = p(z), p(min(zhit, z + v * 1.25)); R = r * foc / z; d = b0 - a0; L = np.hypot(*d)
            if not (-300 < a0[0] < W + 300 and -300 < a0[1] < H + 300): continue
            dd = d / (L + 1e-6) if L > 1 else np.array([0., 1.]); nn = np.array([-dd[1], dd[0]]); al = float(np.clip((z - start) * 1.4, 0, 1)) * .9; th = np.linspace(0, 2 * math.pi, 20)
            tail = a0 + dd * max(R * .6, L * .6); q = np.vstack([a0 + nn * R * .8, a0 - dd * R, a0 - nn * R * .8, tail]); cel.fill(spl(q, True, (3,), 6), dark, al * .55); cel.fill(spl(np.vstack([a0 + nn * R * .5, a0 - dd * R * .7, a0 - nn * R * .5, tail * .5 + a0 * .5]), True, (3,), 6), col, al); cel.fill(np.c_[a0[0] - R * .25 + np.cos(th) * R * .18, a0[1] - R * .30 + np.sin(th) * R * .18], '#ffffff', al)
        else:
            age = (z - zhit) / v
            if age < 9: out.append((float(hx), float(hy), float(age)))
    return out


def burst_at(cel, x, y, age, r=26., col='#f2f5f6', dark='#55606a', seed=0):
    """a drop bursting on a surface seen from above: crown ring, then flying beads with their own little shadows"""
    g = min(1., (age + 1) / 4.); rr = r * (.35 + .9 * g); th = np.linspace(0, 2 * math.pi, 34); q_ = [_h(seed, 40 + j) * 6.28 for j in range(4)]; wob = 1 + (.05 + .07 * _h(seed, 50)) * np.sin(th * 2 + q_[0]) + .06 * np.sin(th * 3 + q_[1]) + .045 * np.sin(th * 5 + q_[2]) + .03 * np.sin(th * 8 + q_[3]); al = max(0., 1 - age / 9.)
    cel.line(np.c_[x + np.cos(th) * rr * wob, y + np.sin(th) * rr * wob], 2.6 * (1 - age / 11.), dark, (.02, .02), alpha=al * .7, seed=1); cel.line(np.c_[x - 1 + np.cos(th) * (rr - 2) * wob, y - 1 + np.sin(th) * (rr - 2) * wob], 1.6, col, (.02, .02), alpha=al, seed=2)
    for k in range(4 + int(6 * _h(seed, 60))):
        a = _h(seed, k, 9) * 6.28; d = rr * (.9 + (.15 + 1.0 * _h(seed, k, 2) ** 2) * age / 4.); q = np.array([x + math.cos(a) * d, y + math.sin(a) * d]); s = r * .085 * (1 - age / 12.) * (.6 + .8 * _h(seed, k, 3)); e = np.linspace(0, 6.28, 12); o = np.c_[np.cos(e), np.sin(e)]; cel.fill(q + np.array([2, 4]) + o * s, dark, al * .45); cel.fill(q + o * s, col, al)
    return cel


# ------------------------------------------------------------------ grass with a ring running through it
class Field:
    """rows of blades on a ground plane seen low: horizon y, rows from far to near.  ring(t) from (cx, depth) pushes blades outward as it passes"""

    def __init__(self, hz=560, near=1120, n=1500, seed=4, cols=('#4c7a5a', '#7ea65c', '#c9cf7e', '#fff0b0'), line='#27402f'):
        r = np.random.default_rng(seed); d = np.sort(r.random(n) ** .62); self.d = d; self.y = hz + (near - hz) * d ** 1.55; self.x = r.uniform(-80, W + 80, n); self.s = .10 + .95 * d ** 1.4; self.h = r.uniform(70, 170, n) * self.s; self.ph = r.uniform(0, 6.28, n); self.c = r.random(n); self.lean = r.normal(.12, .18, n); self.cols = [rgb(c) for c in cols]; self.line = line

    def draw(self, cel, f, ring=None, part=(0., 1.), wind=.20):
        """ring = (cx, cy, radius px on the ground, strength, squash); part = depth range so things can stand between rows"""
        for i in np.nonzero((self.d >= part[0]) & (self.d < part[1]))[0]:
            x, y, h, s = self.x[i], self.y[i], self.h[i], self.s[i]; b = self.lean[i] + wind * math.sin(f * .16 + x * .006 + self.ph[i] * .35); lit = 0.
            if ring is not None:
                cx, cy, R, amt, sq = ring; dx, dy = x - cx, (y - cy) / sq; D = math.hypot(dx, dy); q = math.exp(-((D - R) / (70 + R * .10)) ** 2) * amt; b += q * (dx / (D + 1e-6)) * 1.5; lit = q; h = h * (1 - .30 * q)
            tip = np.array([x + b * h * .9, y - h * (1 - .25 * min(1.6, abs(b)))]); mid = np.array([x + b * h * .22, y - h * .55]); w = max(.8, 5.2 * s); a = spl([(x - w, y), mid - (w * .62, 0), tip], n=5); c = spl([(x + w, y), mid + (w * .62, 0), tip], n=5)
            k = self.c[i] * .62 + .30 * self.d[i] * 0 + lit * .6; col = self.cols[0] * (1 - k) + self.cols[2] * k if k < 1 else self.cols[3]; col = self.cols[1] * .45 + col * .55; cel.fill(np.vstack([a, c[::-1]]), np.clip(col + lit * .18, 0, 1)); cel.fill(np.vstack([spl([(x, y), mid, tip], n=5), c[::-1]]), self.cols[0], .45)
            if s > .45: cel.line(a, 1.5 * s, self.line, (.1, .5), seed=i, alpha=.85)
        return cel


def patter(cel, f, P, n=14, seed=0, r=7., lw=1.2, up=(0., -1.)):
    """rain breaking along any polyline: short crowns thrown along `up`, a new place each time"""
    P = np.asarray(P, float); u = np.array(up) / (np.hypot(*up) + 1e-9); s_ = np.array([-u[1], u[0]])
    for i in range(n):
        per = 7 + int(5 * _h(seed, i)); k = (f + int(_h(seed, i, 1) * per)) // per; a = (f + int(_h(seed, i, 1) * per)) % per
        if a > 4: continue
        q = _h(seed, i, k, 2) * (len(P) - 1); j = min(int(q), len(P) - 2); p = P[j] + (P[j + 1] - P[j]) * (q - j); R = r * (.6 + .6 * _h(seed, i, k, 3)) * (.5 + .25 * a)
        for d in (-.9, -.35, .25, .85): e = p + s_ * d * R + u * R * (1.25 - .5 * abs(d)) * (1 - a / 7.); cel.line(np.array([p + s_ * d * R * .4, e]), lw, '#eef2f3', (.1, .7), seed=i, alpha=.85 - .15 * a)
    return cel


def roll(cel, P, u, r=7., lead=.16, meet=.30, seed=0):
    """two beads on any path P: they creep (u 0..meet), merge, and the heavier bead runs to the end leaving a thin wet trail.  u 0..1"""
    P = np.asarray(P, float); at = lambda v: P[0] + 0 * P[0] if len(P) < 2 else np.array([np.interp(v * (len(P) - 1), np.arange(len(P)), P[:, 0]), np.interp(v * (len(P) - 1), np.arange(len(P)), P[:, 1])]); th = np.linspace(0, 2 * math.pi, 22)
    def bead(p, R, st=1.): cel.fill(np.c_[p[0] + np.cos(th) * R, p[1] + np.sin(th) * R * st], '#8d9699', .55); cel.line(np.c_[p[0] + np.cos(th) * R, p[1] + np.sin(th) * R * st], 1.3, '#2a2f33', (.05, .05), seed=1); cel.fill(np.c_[p[0] - R * .3 + np.cos(th) * R * .26, p[1] - R * .3 + np.sin(th) * R * .3], '#ffffff', .95)
    if u < meet: k = u / meet; a = lead * k * k; b = lead * (1.15 - .15 * k); bead(at(max(0., a)), r * .75); bead(at(b), r * .8)
    else:
        k = (u - meet) / (1 - meet); v = lead + (1 - lead) * k ** 1.7; n = max(2, int(24 * k) + 2); cel.line(np.array([at(lead + (v - lead) * i / n) for i in range(n + 1)]), r * .55, '#c9d0d2', (.8, .05), alpha=.55, seed=seed); bead(at(v), r * 1.12, 1.0 + .25 * min(1., k * 2))
    return cel
