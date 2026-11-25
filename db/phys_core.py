"""phys core: hashes, noise, stage camera, palettes, fast thick-thin polygons.  (PHYS lane)"""
import numpy as np, cv2, math
from numba import njit
from . import W, H
from .paint import rgb, f32, grid, smooth, fbm, lum, ramp
from .ink import Cel, spl, resample, ribbon, noise1, paste, over_roi
from .cam import Camera3D
from .comp import over, screen, blur

TAU = 2 * math.pi


def C(c): return rgb(c) if isinstance(c, str) else np.asarray(c, f32)
def hexmix(a, b, t): return C(a) * (1 - t) + C(b) * t
def tw(f, step=2): return int(f) // step * step
def clamp(x, a=0., b=1.): return min(max(x, a), b)


def h1(*k):
    """hash of integers -> 0..1, stable"""
    n = 2166136261
    for q in k: n = ((n ^ (int(q) & 0xffffffff)) * 16777619) & 0xffffffff
    n ^= n >> 15; n = (n * 2246822519) & 0xffffffff; n ^= n >> 13; n = (n * 3266489917) & 0xffffffff; n ^= n >> 16
    return (n & 0xffffff) / 16777216.


def R(*k): return np.random.default_rng([int(q) & 0xffffffff for q in k])


def size3(u, big=.12, mid=.30):
    """three size classes from a uniform number: few big (1), some mid (.45), many small (.16)"""
    if u < big: return .85 + .4 * u / big
    if u < big + mid: return .36 + .22 * (u - big) / mid
    return .10 + .13 * (u - big - mid) / (1 - big - mid)


@njit(cache=True, fastmath=True)
def _hash3(x, y, z):
    n = (x * 374761393 + y * 668265263 + z * 1440662683) & 0xffffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0xffffffff
    return ((n ^ (n >> 16)) & 0xffff) / 32767.5 - 1.0


@njit(cache=True, fastmath=True)
def vn3(x, y, z):
    ix = math.floor(x); iy = math.floor(y); iz = math.floor(z); fx = x - ix; fy = y - iy; fz = z - iz
    fx = fx * fx * fx * (fx * (fx * 6 - 15) + 10); fy = fy * fy * fy * (fy * (fy * 6 - 15) + 10); fz = fz * fz * (3 - 2 * fz)
    ix = int(ix); iy = int(iy); iz = int(iz); s = 0.0
    for c in range(2):
        a = _hash3(ix, iy, iz + c); b = _hash3(ix + 1, iy, iz + c); d = _hash3(ix, iy + 1, iz + c); e = _hash3(ix + 1, iy + 1, iz + c)
        q = (a + (b - a) * fx) * (1 - fy) + (d + (e - d) * fx) * fy
        s += q * (fz if c else 1 - fz)
    return s


@njit(cache=True, fastmath=True)
def _vn3a(x, y, z, out):
    for i in range(x.size): out[i] = vn3(x[i], y[i], z[i])


def noise(x, y=0., z=0., seed=0):
    """smooth 3-D value noise -1..1 (broadcasts)"""
    x, y, z = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(z, np.float64) + seed * 37.17)
    o = np.empty(x.size); _vn3a(np.array(x).ravel(), np.array(y).ravel(), np.array(z).ravel(), o); o = o.reshape(x.shape)
    return float(o) if o.ndim == 0 else o


@njit(cache=True, fastmath=True)
def _nimg(h, w, sx, sy, z, ox, oy, octs, out):
    for j in range(h):
        for i in range(w):
            v = 0.; a = 1.; px = i * sx + ox; py = j * sy + oy
            for o in range(octs): v += a * vn3(px, py, z + o * 17.3); a *= .5; q = (px * .8 - py * .6) * 2.03 + 3.1; py = (px * .6 + py * .8) * 2.03 + 1.7; px = q
            out[j, i] = v


def noise_img(h, w, scale, z=0., seed=0, octs=3, q=4, aniso=1., ox=0., oy=0.):
    """fbm picture at 1/q resolution; scale = feature px, aniso stretches along x"""
    hh, ww = h // q + 2, w // q + 2; o = np.empty((hh, ww), f32); _nimg(hh, ww, q / (scale * aniso), q / scale, float(z) + seed * 37.17, ox / (scale * aniso), oy / scale, octs, o)
    return cv2.resize(o, (ww * q, hh * q), interpolation=cv2.INTER_LINEAR)[:h, :w]


# ------------------------------------------------------------- camera
def stage(hy=430., gx=W / 2, gy=800., ppm=150., focal=1500., roll=0.):
    """Camera3D for screen-space people: level camera, horizon at y=hy, world (0,0,0) lands on (gx, gy), ppm px per metre there.
    World: +X right, +Y into the picture, +Z up, metres."""
    D = focal / ppm; h = (gy - hy) / ppm
    return Camera3D((0, -D, h), (0, 0, h), focal, cx=gx, cy=hy, roll=roll)


def flat(ppm=1., ox=0., oy=0., far=1e4):
    """orthographic-ish camera: world X = screen x, world Z = -screen y (px / ppm).  For effects laid over a flat drawing."""
    return Camera3D((0, -far, 0), (0, 0, 0), far * ppm, cx=ox, cy=oy)


def quat_m(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1), np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1), np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def axis_m(ax, a):
    ax = np.asarray(ax, float); ax = ax / (np.linalg.norm(ax) + 1e-9); c, s = math.cos(a), math.sin(a); x, y, z = ax
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s], [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s], [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def unit(v): v = np.asarray(v, float); return v / (np.linalg.norm(v) + 1e-9)


# ------------------------------------------------------------- polygons
def hull2(P): return cv2.convexHull(np.asarray(P, np.float32))[:, 0].astype(float)


def grow(P, wf):
    """closed convex polygon pushed outward by wf(normals) px (thick-thin ink underlay)"""
    P = np.asarray(P, float); A = P - np.roll(P, 1, 0); B = np.roll(P, -1, 0) - P
    ar = float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])); sg = 1 if ar > 0 else -1
    na = np.stack([A[:, 1], -A[:, 0]], -1) * sg; nb = np.stack([B[:, 1], -B[:, 0]], -1) * sg
    na /= np.hypot(na[:, 0], na[:, 1])[:, None] + 1e-9; nb /= np.hypot(nb[:, 0], nb[:, 1])[:, None] + 1e-9
    m = na + nb; l = np.hypot(m[:, 0], m[:, 1])[:, None] + 1e-6; m = m / l; k = 1 / np.maximum(.5 * l, .55)
    return P + m * k * np.asarray(wf(m))[:, None]


def wedge(cel, a, b, w0, w1, col, alpha=1., bow=0.):
    """fast tapered stroke a -> b, widths px at the two ends (0 = needle)"""
    a, b = np.asarray(a, float), np.asarray(b, float); d = b - a; l = math.hypot(d[0], d[1])
    if l < .6: return
    n = np.array([-d[1], d[0]]) / l; m = (a + b) / 2 + n * bow; wm = (w0 + w1) * .5 * 1.15
    cel.fill(np.array([a + n * w0 * .5, m + n * wm * .5, b + n * w1 * .5, b - n * w1 * .5, m - n * wm * .5, a - n * w0 * .5]), col, alpha)


def thin(cel, P, w, col, alpha=1., a=.3, b=.5, K=24):
    """fast tapered stroke along a polyline (no tremor)"""
    P = np.asarray(P, float)
    if len(P) < 2: return cel
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    if d[-1] < .5: return cel
    u = np.linspace(0, d[-1], K); P = np.stack([np.interp(u, d, P[:, 0]), np.interp(u, d, P[:, 1])], -1); T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9
    v = np.linspace(0, 1, K); hw = .5 * w * np.minimum(1., v / max(a, 1e-3)) ** .7 * np.minimum(1., (1 - v) / max(b, 1e-3)) ** .7
    N = np.stack([-T[:, 1], T[:, 0]], -1) * np.maximum(hw, .1)[:, None]; cel.fill(np.vstack([P + N, (P - N)[::-1]]), col, alpha); return cel


def zigzag(a, b, seed=0, amp=.16, lev=4):
    """angular crack path a -> b (midpoint displacement, hard corners)"""
    P = [np.asarray(a, float), np.asarray(b, float)]; r = R(seed, 5)
    for _ in range(lev):
        Q = [P[0]]
        for p, q in zip(P[:-1], P[1:]):
            d = q - p; m = p + d * r.uniform(.36, .64) + np.array([-d[1], d[0]]) * r.normal(0, amp); Q += [m, q]
        P = Q; amp *= .80
    return np.array(P)


def claw_edge(s0, s1, pitch, seed=0, lean=.35, amp=(.30, 1.), n=7):
    """a drawn row of curved claws (wave crest, flame, spray lip) on the interval s0..s1, anchored to absolute s so it travels with the matter.
    lean > 0 tips toward +s.  -> (s, q) polyline, q = 0..1 tooth height"""
    k0, k1 = int(math.floor(min(s0, s1) / pitch)) - 1, int(math.ceil(max(s0, s1) / pitch)) + 1; S, Q = [], []; u = np.linspace(0, 1, n + 1)[1:]; b = lambda p0, p1, p2, p3: (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u * u * p2 + u ** 3 * p3; at = lambda k: (k + .55 * (h1(seed, k, 1) - .5)) * pitch
    for k in range(k0, k1 + 1):
        a, c = at(k), at(k + 1); w = c - a; h = amp[0] + (amp[1] - amp[0]) * size3(h1(seed, k, 2), .30, .40) / 1.1; L = lean * h * pitch
        if lean >= 0: tip = a + w * .86 + L; S += [np.array([a]), b(a, a + w * .50, tip - w * .50 - L * .2, tip), b(tip, tip - w * .16, c - w * .10 + L * .15, c)]; Q += [np.array([0.]), b(0., .04 * h, .62 * h, h), b(h, .70 * h, .16 * h, 0.)]
        else: tip = a + w * .14 + L; S += [np.array([a]), b(a, a + w * .10 + L * .15, tip + w * .16, tip), b(tip, tip + w * .50 + L * -.2, c - w * .50, c)]; Q += [np.array([0.]), b(0., .16 * h, .70 * h, h), b(h, .62 * h, .04 * h, 0.)]
    return np.concatenate(S), np.concatenate(Q)


def claw_band(S, Bp, Tp, s, q, lo=.5):
    """sheet polygon between a base polyline Bp and an envelope Tp (screen px, both indexed by increasing S); s, q from claw_edge; lo = height of the roots as a fraction of the sheet"""
    ok = (s >= S[0]) & (s <= S[-1]); s, q = s[ok], q[ok]
    if len(s) < 3: return None
    I = lambda P: np.stack([np.interp(s, S, P[:, 0]), np.interp(s, S, P[:, 1])], -1); b, t = I(Bp), I(Tp); k = (lo + (1 - lo) * q)[:, None]; return np.vstack([Bp[:1], b + (t - b) * k, Bp[::-1]])


def gather(polys, pad=8):
    A = np.concatenate([np.asarray(p) for p in polys]); x0, y0 = np.floor(A.min(0) - pad); x1, y1 = np.ceil(A.max(0) + pad)
    return int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))


# ------------------------------------------------------------- palettes (colour script)
ROCK = dict(
    night=dict(lit='#776cb8', mid='#3c3666', dark='#171530', rim='#c9a8ff', ink='#0a0916', fresh='#4c467c', pit='#0b0a18', bounce='#4a2c4c', rim2='#ff9a52'),
    fire=dict(lit='#e07a3c', mid='#4a3550', dark='#1d1830', rim='#ffd08a', ink='#120812', fresh='#8a4b46', pit='#0e0a16', bounce='#3a3270', rim2='#b48cff'),
    grey=dict(lit='#94979a', mid='#606366', dark='#2e3034', rim='#e6eaec', ink='#121316', fresh='#7b7e80', pit='#17181b', bounce='#44474b', rim2='#aeb4b6'),
    gold=dict(lit='#f2bd55', mid='#6e5238', dark='#221d26', rim='#fff6c4', ink='#170e10', fresh='#a87a3a', pit='#120e14', bounce='#8a5a1c', rim2='#ffd35a'),
    dawn=dict(lit='#f6d9c4', mid='#b58f96', dark='#655a80', rim='#fff4e2', ink='#2c2034', fresh='#d7b0a6', pit='#3a3050', bounce='#a9c9e8', rim2='#d8ecff'),
    day=dict(lit='#eed8a8', mid='#b8926a', dark='#6d4f50', rim='#fff4d2', ink='#2a1a1c', fresh='#d2ae7e', pit='#3a2628', bounce='#9a7a5a', rim2='#ffe2b0'),
    mono=dict(lit='#f2f2f0', mid='#9a9a98', dark='#2a2a2a', rim='#ffffff', ink='#060606', fresh='#cfcfcc', pit='#0a0a0a', bounce='#6a6a6a', rim2='#ffffff'))
DUST = dict(
    night=dict(lit='#8d82c4', mid='#564e86', dark='#2c2850', ink='#151230', rim='#d9c2ff'),
    fire=dict(lit='#ffb061', mid='#8a4a4c', dark='#3a2440', ink='#1a1022', rim='#ffe6a8'),
    grey=dict(lit='#b9bcbd', mid='#85888a', dark='#55585b', ink='#2c2e31', rim='#eef0f0'),
    gold=dict(lit='#ffe9a0', mid='#d09a3e', dark='#5e4436', ink='#2c1c1a', rim='#fffbe0'),
    dawn=dict(lit='#fff0e2', mid='#e0bcb4', dark='#9c88a8', ink='#5c4a6c', rim='#ffffff'),
    day=dict(lit='#f6e8c8', mid='#d2b48c', dark='#9a7c6c', ink='#4c382e', rim='#ffffff'),
    violet=dict(lit='#fbf7ff', mid='#c784ff', dark='#6a1fd0', ink='#2c0a66', rim='#ffffff'),
    storm=dict(lit='#6c6f7c', mid='#42455a', dark='#25273a', ink='#12131e', rim='#b8a8e8'),
    steam=dict(lit='#ffffff', mid='#dfe7f2', dark='#a9b8d0', ink='#6f80a0', rim='#ffffff'))
WATER = dict(
    night=dict(deep='#05040c', mid='#15102c', lit='#3b2a78', foam='#c9b2ff', white='#fbf7ff', ink='#05040c', shade='#4a3a8c'),
    day=dict(deep='#1d5fa8', mid='#2f8fd0', lit='#6fcdf0', foam='#e9fbff', white='#ffffff', ink='#12355e', shade='#8fc6ea'),
    dawn=dict(deep='#6f7fb0', mid='#c39fb0', lit='#f7c3a8', foam='#fff1e0', white='#ffffff', ink='#4a4268', shade='#d3a9b4'),
    grey=dict(deep='#3c3f43', mid='#5e6265', lit='#8d9192', foam='#d8dcdd', white='#f6f8f8', ink='#1e2023', shade='#9aa0a3'),
    gold=dict(deep='#16130f', mid='#4a3a1c', lit='#c98d2a', foam='#ffe9a0', white='#fff7c8', ink='#120c08', shade='#d9a23c'),
    lake=dict(deep='#1f6a5c', mid='#4ba07a', lit='#b9d98a', foam='#f4fbe0', white='#ffffff', ink='#163c38', shade='#9fd0b0'),
    blue=dict(deep='#0a1230', mid='#14306a', lit='#2f7fd6', foam='#a8e2ff', white='#f2fbff', ink='#060a1c', shade='#5aaef0'))
