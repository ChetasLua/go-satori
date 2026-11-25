"""ki core: drawn energy = ribbons smooth-unioned into stacked signed-distance levels (sharp tips, round valleys, nested tones),
then cut into flat tones with AA edges and a broken thick-thin coloured pen line.  Everything seeded, pure functions of the frame."""
import numpy as np, cv2, math
from numba import njit
from . import W, H
from .paint import rgb, f32, grid, smooth
from .ink import Cel, spl, resample, over_roi

TAU = 2 * math.pi


def C(c): return rgb(c) if isinstance(c, str) else np.asarray(c, f32)


# ------------------------------------------------------------------ hashes / noise
def h1(*k):
    """integers -> 0..1, stable"""
    n = 2166136261
    for q in k: n = ((n ^ (int(q) & 0xffffffff)) * 16777619) & 0xffffffff
    n ^= n >> 15; n = (n * 2246822519) & 0xffffffff; n ^= n >> 13; n = (n * 3266489917) & 0xffffffff; n ^= n >> 16
    return (n & 0xffffff) / 16777216.


def R(*k): return np.random.default_rng([int(q) & 0xffffffff for q in k])


def bms(u, big=.18, med=.34):
    """big / medium / small law: 18 % big, 34 % medium, the rest small"""
    if u < big: return .85 + .3 * (u / big)
    if u < big + med: return .45 + .25 * (u - big) / med
    return .16 + .2 * (u - big - med) / (1 - big - med)


@njit(cache=True, fastmath=True)
def _hh(x, y, z):
    n = (x * 374761393 + y * 668265263 + z * 1440662683) & 0xffffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0xffffffff
    return ((n ^ (n >> 16)) & 0xffff) / 32767.5 - 1.0


@njit(cache=True, fastmath=True)
def _vn(x, y, z):
    ix = math.floor(x); iy = math.floor(y); iz = math.floor(z)
    fx = x - ix; fy = y - iy; fz = z - iz
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz)
    ix = int(ix); iy = int(iy); iz = int(iz); s = 0.0
    for c in range(2):
        a = _hh(ix, iy, iz + c); b = _hh(ix + 1, iy, iz + c); d = _hh(ix, iy + 1, iz + c); e = _hh(ix + 1, iy + 1, iz + c)
        q = (a + (b - a) * fx) * (1 - fy) + (d + (e - d) * fx) * fy
        s += q * (fz if c else 1 - fz)
    return s


@njit(cache=True, fastmath=True)
def _vna(x, y, z, o):
    for i in range(x.size): o[i] = _vn(x[i], y[i], z[i])


def noise(x, y=0., z=0., seed=0):
    """smooth value noise -1..1 (z usually time)"""
    x, y, z = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(z, np.float64) + seed * 37.17)
    o = np.empty(x.size); _vna(np.array(x).ravel(), np.array(y).ravel(), np.array(z).ravel(), o); return o.reshape(x.shape) if x.ndim else float(o[0])


@njit(cache=True, fastmath=True)
def _nimg(h, w, sx, sy, ox, oy, z, octs, o):
    for j in range(h):
        for i in range(w):
            v = 0.; a = 1.; px = i * sx + ox; py = j * sy + oy
            for q in range(octs):
                v += a * _vn(px, py, z + q * 17.3); a *= .5; t = (px * .707 - py * .707) * 2.03 + 3.1; py = (px * .707 + py * .707) * 2.03 + 1.7; px = t
            o[j, i] = v


def nimg(h, w, scale, z=0., seed=0, octs=2, q=4, ax=1., ay=1., ox=0., oy=0.):
    """noise picture; feature size scale px (ax / ay stretch), ox / oy px offset (scroll)"""
    hh, ww = h // q + 2, w // q + 2; o = np.empty((hh, ww), f32); sx, sy = scale * ax, scale * ay
    _nimg(hh, ww, q / sx, q / sy, ox / sx, oy / sy, float(z) + seed * 37.17, octs, o)
    return cv2.resize(o, (ww * q, hh * q), interpolation=cv2.INTER_LINEAR)[:h, :w]


def lin(n): return np.linspace(0, 1, n)


def ss(t): t = min(max(t, 0.), 1.); return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ paths
class Path:
    """arc-length frame of a polyline; at(s, n) = point at s px along, n px to the left normal (extrapolates straight)"""
    def __init__(self, P, step=6.):
        P = np.asarray(P, float); P = resample(P, step)
        if len(P) < 3: P = np.linspace(P[0], P[-1] + 1e-3, 3)
        self.P = P; T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; self.T = T; self.N = np.stack([T[:, 1], -T[:, 0]], -1)
        self.s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; self.L = float(self.s[-1])

    def at(self, s, n=0.):
        s = np.atleast_1d(np.asarray(s, float)); c = np.clip(s, 0, self.L); e = s - c; I = np.interp; q = self.s
        nx, ny = I(c, q, self.N[:, 0]), I(c, q, self.N[:, 1]); k = n / (np.sqrt(nx * nx + ny * ny) + 1e-9)
        return np.stack([I(c, q, self.P[:, 0]) - ny * e + nx * k, I(c, q, self.P[:, 1]) + nx * e + ny * k], -1)

    def sub(self, a, b, n=0., k=16):
        v = lin(k); return self.at(a + (b - a) * v, n(v) if callable(n) else n)


def normals(P):
    T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; return np.stack([T[:, 1], -T[:, 0]], -1)


# ------------------------------------------------------------------ level field
@njit(cache=True, fastmath=True)
def _rib(D, ox, oy, P, HW, kk):
    L = D.shape[0]; h = D.shape[1]; w = D.shape[2]; K = HW.shape[1]
    x0 = 1e9; y0 = 1e9; x1 = -1e9; y1 = -1e9; mh = 0.0
    for l in range(L):
        for k in range(K):
            if HW[l, k] > mh: mh = HW[l, k]
            if P[l, k, 0] < x0: x0 = P[l, k, 0]
            if P[l, k, 0] > x1: x1 = P[l, k, 0]
            if P[l, k, 1] < y0: y0 = P[l, k, 1]
            if P[l, k, 1] > y1: y1 = P[l, k, 1]
    m = mh + kk + 2
    ix0 = max(0, int(math.floor(x0 - m - ox))); iy0 = max(0, int(math.floor(y0 - m - oy)))
    ix1 = min(w, int(math.ceil(x1 + m - ox)) + 1); iy1 = min(h, int(math.ceil(y1 + m - oy)) + 1)
    if ix1 <= ix0 or iy1 <= iy0: return
    T = np.full((iy1 - iy0, ix1 - ix0), np.float32(-1000.))
    for l in range(L):
        if l: T[:] = -1000.
        any_ = False
        for k in range(K - 1):
            r0 = HW[l, k]; r1 = HW[l, k + 1]
            if r0 < -kk - 2 and r1 < -kk - 2: continue
            ax = P[l, k, 0] - ox - ix0; ay = P[l, k, 1] - oy - iy0; bx = P[l, k + 1, 0] - ox - ix0; by = P[l, k + 1, 1] - oy - iy0
            dx = bx - ax; dy = by - ay; L2 = dx * dx + dy * dy
            if L2 < 1e-6: continue
            sl = (r1 - r0) / math.sqrt(L2); ca = 1.0 / math.sqrt(1 + sl * sl); mm = max(r0, r1) + kk + 2
            if mm <= 0: continue
            jx0 = max(0, int(min(ax, bx) - mm)); jx1 = min(ix1 - ix0, int(max(ax, bx) + mm) + 2); jy0 = max(0, int(min(ay, by) - mm)); jy1 = min(iy1 - iy0, int(max(ay, by) + mm) + 2)
            for y in range(jy0, jy1):
                for x in range(jx0, jx1):
                    t = ((x - ax) * dx + (y - ay) * dy) / L2
                    if t < 0: t = 0.0
                    elif t > 1: t = 1.0
                    qx = x - ax - dx * t; qy = y - ay - dy * t
                    sd = (r0 + (r1 - r0) * t - math.sqrt(qx * qx + qy * qy)) * ca
                    if sd > T[y, x]: T[y, x] = sd; any_ = True
        if not any_: continue
        for y in range(iy1 - iy0):
            for x in range(ix1 - ix0):
                b = T[y, x]
                if b < -kk - 2: continue
                a = D[l, y + iy0, x + ix0]; e = kk - abs(a - b)
                if e > 0 and kk > 0: D[l, y + iy0, x + ix0] = max(a, b) + e * e / kk * .25
                elif b > a: D[l, y + iy0, x + ix0] = b


@njit(cache=True, fastmath=True)
def _smax(D, S, kk):
    for i in range(D.size):
        a = D.flat[i]; b = S.flat[i]; e = kk - abs(a - b)
        D.flat[i] = max(a, b) + (e * e / kk * .25 if e > 0 and kk > 0 else 0.0)


@njit(cache=True, fastmath=True)
def _cel(D, A, B, AL, G, useG, LW, LC, zs, ls, ox, oy, aa, out):
    L, h, w = D.shape
    for y in range(h):
        for x in range(w):
            if D[0, y, x] < -8.0: continue
            r = 0.0; g = 0.0; b = 0.0; a = 0.0; q = G[y, x] if useG else 0.0
            for l in range(L):
                d = D[l, y, x]; k = d / aa + .5
                if k > 0:
                    if k > 1: k = 1.0
                    k *= AL[l]
                    r = r * (1 - k) + (A[l, 0] + (B[l, 0] - A[l, 0]) * q) * k; g = g * (1 - k) + (A[l, 1] + (B[l, 1] - A[l, 1]) * q) * k; b = b * (1 - k) + (A[l, 2] + (B[l, 2] - A[l, 2]) * q) * k; a = a * (1 - k) + k
                lw = LW[l]
                if lw > 0 and d > -lw * 2.2 and d < lw * 2.2 + 1:
                    n = _vn((x + ox) * ls, (y + oy) * ls, zs + l * 7.7); e = lw * (.62 + 1.15 * n)
                    if e > 0.12:
                        k = e * .5 + .5 - abs(d - e * .18)
                        if k > 0:
                            if k > 1: k = 1.0
                            k *= LC[l, 3]
                            r = r * (1 - k) + LC[l, 0] * k; g = g * (1 - k) + LC[l, 1] * k; b = b * (1 - k) + LC[l, 2] * k; a = a * (1 - k) + k
            out[y, x, 0] = r; out[y, x, 1] = g; out[y, x, 2] = b; out[y, x, 3] = a


class Field:
    """L nested tone levels over a box.  rib(): add one tapering ribbon (smooth union, valley radius k px).
    cel(): cut into flat tones -> premultiplied RGBA layer."""
    def __init__(self, L=4, box=None):
        x0, y0, x1, y1 = box or (0, 0, W, H); x0, y0 = int(max(0, x0)), int(max(0, y0)); x1, y1 = int(min(W, x1)), int(min(H, y1))
        self.ok = x1 > x0 and y1 > y0; self.L = L; self.ox, self.oy = x0, y0; self.w, self.h = max(1, x1 - x0), max(1, y1 - y0)
        self.D = np.full((L, self.h, self.w), -999., f32)

    def rib(self, P, HW, k=8., off=None):
        """P (K,2); HW list of L arrays (half width px, may go negative past a tip) ; off = list of L sideways offsets px (scalar/array)"""
        if not self.ok: return self
        P = np.asarray(P, float); HW = np.stack([np.broadcast_to(np.asarray(q, float), (len(P),)) for q in HW]); Q = np.repeat(P[None], self.L, 0)
        if off is not None:
            N = normals(P); Q = np.stack([P + N * np.broadcast_to(np.asarray(o, float), (len(P),))[:, None] for o in off])
        if not np.isfinite(Q).all(): return self
        _rib(self.D, self.ox, self.oy, Q, HW, float(k)); return self

    def sdf(self, arrs, k=8.):
        for l, a in enumerate(arrs):
            if a is not None: _smax(self.D[l], np.ascontiguousarray(a[self.oy:self.oy + self.h, self.ox:self.ox + self.w].astype(f32)), float(k))
        return self

    def cel(self, cols, cols2=None, grad=None, alpha=None, lw=None, ink=None, seed=0, lscale=.012, aa=1.0, full=True, line_a=1.0):
        """cols: L colours outer -> inner; cols2 + grad (full-frame 0..1 field): second palette blended in by grad;
        alpha per level; lw = ink width px per level (0 = none), ink = ink colours per level"""
        L = self.L; A = np.stack([C(c) for c in cols]).astype(f32); B = A if cols2 is None else np.stack([C(c) for c in cols2]).astype(f32)
        AL = np.ones(L, f32) if alpha is None else np.asarray(alpha, f32); LW = np.zeros(L, f32) if lw is None else np.asarray(lw, f32)
        LC = np.zeros((L, 4), f32)
        if ink is not None:
            for l, c in enumerate(ink):
                if c is not None: LC[l, :3] = C(c); LC[l, 3] = line_a
        useG = grad is not None; G = np.ascontiguousarray(grad[self.oy:self.oy + self.h, self.ox:self.ox + self.w].astype(f32)) if useG else np.zeros((1, 1), f32)
        o = np.zeros((self.h, self.w, 4), f32)
        if self.ok: _cel(self.D, A, B, AL, G, useG, LW, LC, float(seed) * 3.3, lscale, self.ox, self.oy, aa, o)
        if not full: return o
        if (self.h, self.w) == (H, W): return o
        F = np.zeros((H, W, 4), f32); F[self.oy:self.oy + self.h, self.ox:self.ox + self.w] = o; return F

    def mask(self, l=0):
        m = np.clip(self.D[l] + .5, 0, 1)
        if (self.h, self.w) == (H, W): return m
        F = np.zeros((H, W), f32); F[self.oy:self.oy + self.h, self.ox:self.ox + self.w] = m; return F


# ------------------------------------------------------------------ flame tongues
LV = ((1., 1.), (.85, .93), (.60, .78), (.30, .54))          # (width, length) of each nested tone


def tprof(v, belly=.22, need=1.25, root=.82):
    v = np.clip(v, 0, 1); return (1 - v) ** (need * (.42 + .62 * v)) * (root + (1.28 - root) * np.exp(-((v - belly) / .24) ** 2)) / 1.1


def twidths(K, L, b, lv=LV, belly=.22, need=1.25, root=.82, jit=0., seed=0):
    """L nested half-width profiles of a tongue of length L px, belly half width b"""
    v = lin(K); w = 1 + jit * noise(v * 4., seed * 1.7) if jit else 1.; out = []
    for a, l in lv:
        u = v / l; out.append(np.where(u < 1, b * a * tprof(u, belly, need, root) * w, -(v - l) * L))
    return out


def kwidths(K, L, b, lv=LV, belly=.18, need=1.15, root=.9):
    """tongue that leaves a bright mass: outer tones rooted like twidths, inner tones are leaves (no round root showing)"""
    a = twidths(K, L, b, lv[:2], belly, need, root); v = lin(K)
    for q, l in lv[2:]:
        u = v / (l * .95); a.append(np.where(u < 1, b * q * .95 * np.clip(u / .30, 0, 1) ** .8 * np.clip((1 - u) / .7, 0, 1) ** 1.05 - .05, -(v - l) * L - .05))
    return a


def lwidths(K, L, b, lv=LV, peak=.35, p0=.7, p1=1.1):
    """leaf / sliver: sharp at both ends"""
    v = lin(K); out = []
    for a, l in lv:
        c = peak; u = (v - c) / l + c; w = np.where(u < c, np.clip(u / c, 0, 1) ** p0, np.clip((1 - u) / (1 - c), 0, 1) ** p1); e = np.maximum(np.maximum(-u, u - 1), 0) * L; out.append(np.where((u > 0) & (u < 1), b * a * w, -e - .01))
    return out


def spine(x, y, L, th, K=14, sway=.14, waves=.85, ph=0., curl=0., pw=1.3):
    """S-curved centre line from a root, heading th rad; wave amplitude grows to the tip"""
    v = lin(K); off = sway * L * v ** pw * np.sin(TAU * waves * v + ph); a = v * L; c, s = math.cos(th), math.sin(th)
    P = np.stack([x + c * a - s * off, y + s * a + c * off], -1)
    if curl: g = curl * v * v; q = P - [x, y]; P = np.stack([x + q[:, 0] * np.cos(g) - q[:, 1] * np.sin(g), y + q[:, 0] * np.sin(g) + q[:, 1] * np.cos(g)], -1)
    return P


def lick(F, P, b, L=None, k=8., lv=LV, seed=0, kids=2, belly=.22, need=1.25, root=.82, side=0., off=0., jit=.10):
    """one drawn tongue along spine P with small secondary licks on its edges.  off = sideways push of inner tones (fraction of b)"""
    P = np.asarray(P, float); n = len(P); L = L or float(np.hypot(*np.diff(P, axis=0).T).sum())
    o = [off * b * q * (1 - .5 * lin(n)) for q in (0, .35, .7, 1.)[:len(lv)]] if off else None
    F.rib(P, twidths(n, L, b, lv, belly, need, root, jit, seed), k, o)
    if kids and b > 7 and L > 60:
        T = np.gradient(P, axis=0)
        for j in range(kids):
            if h1(seed, j, 3) < .22: continue
            v = .16 + .52 * (j + h1(seed, j, 1)) / kids; i = min(n - 2, int(v * (n - 1))); sg = 1 if h1(seed, j, 2) > .5 else -1; th = math.atan2(T[i, 1], T[i, 0]); w = b * float(tprof(np.array(v), belly, need, root)); nn = np.array([-math.sin(th), math.cos(th)]) * sg
            l = L * (.20 + .24 * h1(seed, j, 4)) * (1.1 - .5 * v); p = P[i] + nn * w * .55; q = spine(p[0], p[1], l, th + sg * (.40 + .25 * h1(seed, j, 5)), 8, .06, .7, h1(seed, j, 6) * 6, -sg * (.55 + .4 * h1(seed, j, 7)))
            F.rib(q, twidths(8, l, w * (.34 + .22 * h1(seed, j, 8)), lv, .15, 1.2, 1.), k * .8)
    return F


# ------------------------------------------------------------------ light
def bloom(layer, thr=.45, r=(5, 18, 52), wts=(.45, .32, .26), col=None, amt=.6):
    """additive light of the bright part of a premult layer (1/8 size)"""
    h, w = layer.shape[:2]; q = cv2.resize(layer[..., :3], (w // 8, h // 8), interpolation=cv2.INTER_AREA); m = np.clip((q.max(-1) - thr) / (1 - thr), 0, 1)[..., None] * q
    o = sum(wt * cv2.GaussianBlur(m, (0, 0), s / 8.) for s, wt in zip(r, wts))
    if col is not None: o = o * C(col)
    return cv2.resize(o.astype(f32), (w, h), interpolation=cv2.INTER_LINEAR) * amt


def put(base, layer, op=1.):
    if layer is None: return base
    return base * (1 - layer[..., 3:4] * op) + layer[..., :3] * op


def stack(*ls):
    """premult layers, first = bottom"""
    o = None
    for l in ls:
        if l is None: continue
        o = l if o is None else o * (1 - l[..., 3:4]) + l
    return o


def add(base, light, protect=None, amt=1.):
    """add light; protect = mask 0..1 where light is held back (faces, figures)"""
    if light is None: return base
    return base + light * amt if protect is None else base + light * (amt * (1 - protect))[..., None] if np.ndim(protect) else base + light * amt


def displace(img, dx, dy):
    x, y = grid(*img.shape[:2]); return cv2.remap(img, x + dx.astype(f32), y + dy.astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def seg_box(P, pad):
    P = np.asarray(P); return (P[:, 0].min() - pad, P[:, 1].min() - pad, P[:, 0].max() + pad, P[:, 1].max() + pad)


# ------------------------------------------------------------------ palettes: outer -> inner, plus ink, deep (shadow gap), light (glow)
PAL = {
    'gold': dict(c=('#e8780c', '#f9ad1d', '#ffe06a', '#fff8d2'), ink='#7a2b04', deep='#4a1503', glow='#ffb02e', lit='#ffe27a', lit2='#fffbe2'),
    'blue': dict(c=('#1c50d8', '#3f9dff', '#9fe2ff', '#f6feff'), ink='#0a1a6e', deep='#060b3a', glow='#4aa8ff', lit='#a9e4ff', lit2='#f4fdff'),
    'base': dict(c=('#7fb8f2', '#b9dcff', '#e2f3ff', '#ffffff'), ink='#4a7fc4', deep='#1a3566', glow='#a9d2ff', lit='#dcefff', lit2='#ffffff'),
    'violet': dict(c=('#5410b4', '#8a2cff', '#c784ff', '#fdf6ff'), ink='#1d0448', deep='#0e0226', glow='#8a2cff', lit='#c79bff', lit2='#f6ecff'),
    'silver': dict(c=('#5f7398', '#9fb1cc', '#d3deee', '#fbfdff'), ink='#33415f', deep='#10182a', glow='#a8c4ff', lit='#dfe9fb', lit2='#ffffff'),
    'white': dict(c=('#e0a848', '#ffd78a', '#fff0c8', '#fffef6'), ink='#5a3410', deep='#2a1608', glow='#ffd890', lit='#fff0cc', lit2='#fffdf4'),
    'red': dict(c=('#9a0f14', '#e2322a', '#ff9a6a', '#fff2e2'), ink='#3c0408', deep='#1c0204', glow='#ff4a34', lit='#ffb49a', lit2='#fff2ea'),
}


def pal(p): return PAL[p] if isinstance(p, str) else p
