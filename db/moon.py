"""The light of acts C and G.  Nothing in here is a character.

    Sky(cx, cy, r, seed)            .frame(t, part) -> float32 RGB: disc, halo, torn black clouds (part 0 closed .. 1 parted); plates built once
    sea(img, hz, cx, cy, r, t)      black water with the disc's true reflection road (reflected-ray test on a wave field), in place
    sparks(img, t, seed, ...)       violet sparks that hang still like constellations
    ball(cx, cy, r, spin, crack, t) premultiplied RGBA of the moon-sized ball; crack 0..1 = plates split from the centre
    ball_halo(img, cx, cy, r)       its glow;  disc / halo / halo_rgb = the plain white disc and the light falloff
"""
import numpy as np, cv2, math
from numba import njit, prange
from . import W, H as HH
from .paint import rgb, f32, fbm, smooth, mix, grid, ramp

BLACK, VIOLET, LILAC, DEAD = '#07060c', '#8a2cff', '#c784ff', '#fbf7ff'
C0 = (960, 486, 208)                     # the film's match circle


# ------------------------------------------------------------------ light
def halo(d, r):
    """glow outside a limb: a tight scatter skin, a lens bloom, a thin wide air term.  d = px beyond the limb"""
    d = np.maximum(d, 0)
    return .85 * np.exp(-d / (r * .030 + 2)) + .34 * np.exp(-d / (r * .16)) + .10 * np.exp(-d / (r * .62))


def halo_rgb(g):
    return ramp(np.clip(g, 0, 1.3), [(0, '#000000'), (.07, '#0e051f'), (.2, '#290a60'), (.45, '#7a24ee'), (.8, '#c784ff'), (1.2, '#f6e9ff')])


def disc(cx, cy, r, h=HH, w=W):
    x, y = grid(h, w); d = np.hypot(x - cx, y - cy) - r
    return np.clip(.5 - d, 0, 1), d


# ------------------------------------------------------------------ clouds
def _cloud(M, x0, base, width, height, r, dens=1.0):
    """one torn bank on mask M: puffs along a lopsided crest, flat underside, long tapering wisps"""
    n = max(6, int(width / 46 * dens)); pk = r.uniform(.3, .7); sk = r.uniform(.8, 1.6)
    env = lambda u: np.clip(np.where(u < pk, (u / pk) ** sk, ((1 - u) / (1 - pk)) ** (1 / sk + .25)), 0, 1)
    for k in range(n):
        u = (k + r.uniform(0, 1)) / n; e = float(env(u)); R = height * (.16 + .34 * e) * r.uniform(.65, 1.15); X = x0 + u * width; Y = base - e * height * r.uniform(.35, .9) + R * .25
        cv2.ellipse(M, (int(X * 16), int(Y * 16)), (int(R * 1.45 * 16), int(R * 16)), 0, 0, 360, 1., -1, cv2.LINE_AA, 4)
        for _ in range(5):
            th = r.uniform(-3.0, -.14); q = R * r.uniform(.18, .42); cv2.ellipse(M, (int((X + math.cos(th) * R * 1.38) * 16), int((Y + math.sin(th) * R * .95) * 16)), (int(q * 1.35 * 16), int(q * 16)), 0, 0, 360, 1., -1, cv2.LINE_AA, 4)
    for k in range(int(9 * dens) + 4):                       # wisps: thin lenses at and under the base
        u = r.uniform(-.10, 1.10); L = width * r.uniform(.10, .34); Y = base + height * r.uniform(-.25, .30); t = height * r.uniform(.018, .07); X = x0 + u * width; b = r.normal(0, .05) * L
        P = np.array([(X - L, Y + b), (X - L * .4, Y - t), (X + L * .2, Y - t * 1.2), (X + L, Y - b * .4), (X + L * .3, Y + t * .7), (X - L * .5, Y + t)])
        cv2.fillPoly(M, [np.round(P * 16).astype(np.int32)], 1., cv2.LINE_AA, 4)


class Sky:
    """plates once, then cheap frames.  banks = (side, speed px/frame, cover); side -1 slides left, +1 right, 0 stays low"""
    def __init__(self, cx=C0[0], cy=C0[1], r=C0[2], seed=3, h=HH, w=W, veil=True):
        self.cx, self.cy, self.r, self.h, self.w = cx, cy, r, h, w; rr = np.random.default_rng(seed * 77 + 1)
        self.m, self.d = disc(cx, cy, r, h, w); g = halo(self.d, r); self.L = np.where(self.d > 0, g * 1.15 + .13 * np.exp(-np.maximum(self.d, 0) / (r * 2.3)), 1.).astype(f32)
        base = np.broadcast_to(rgb(BLACK), (h, w, 3)) + halo_rgb(g * .78); c = mix(rgb('#f1e8ff'), rgb(DEAD), smooth(0, .22, -self.d / r)[..., None])
        self.base = (base * (1 - self.m[..., None]) + c * self.m[..., None]).astype(f32); self.banks = []
        pad = 900
        spec = []
        if veil:                                               # (x, base, width, height, side, travel)
            for k in range(3): spec.append((cx - r * 2.4 - k * 70, cy - r * .40 + k * r * .42, r * 2.9, r * (.50 + .12 * k), -1, 1.7 + .25 * k))
            for k in range(3): spec.append((cx - r * .55 + k * 90, cy - r * .16 + k * r * .40, r * 3.1, r * (.46 + .12 * k), 1, 1.7 + .22 * k))
        for k in range(12): s_ = [-1, 1][k % 2]; spec.append((cx + s_ * r * (1.5 + (k // 2) * .62) - (r * 2.6 if s_ < 0 else 0) + rr.uniform(-60, 60), cy + rr.uniform(-1.95, 1.05) * r, r * rr.uniform(1.9, 3.4), r * rr.uniform(.24, .52), s_, rr.uniform(.30, .6)))
        for k in range(6): spec.append((-320 + k * w / 4.6, cy + r * (1.18 + .10 * (k % 3)), r * rr.uniform(2.4, 3.2), r * rr.uniform(.34, .6), [-1, 1][k >= 3], .20))
        if veil: spec.append((cx - r * .55, cy + r * .44, r * 1.1, r * .22, -1, 1.75))          # shuts two pinholes that read as eyes
        for (x0, b, wd, ht, sd, tr) in spec:
            M = np.zeros((h, int(wd * 1.5 + 260)), f32); _cloud(M, 130 + wd * .25, b, wd, ht, rr, 1. + .3 * (ht / r))
            n = fbm(*M.shape, (30, 11), (1, .5), int(rr.integers(1 << 20)), aspect=2.2)
            s = cv2.GaussianBlur(M, (0, 0), 3.2) + .085 * n; a = smooth(.46, .56, s)                   # hard, hand-cut edge
            b8 = cv2.morphologyEx((a > .5).astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))); a = np.maximum(a, smooth(.36, .64, cv2.GaussianBlur(b8.astype(f32), (0, 0), 2.6)))
            n2 = fbm(*M.shape, (70, 24), (1, .5), int(rr.integers(1 << 20)), aspect=2.)
            dt = cv2.GaussianBlur(np.maximum(0, cv2.distanceTransform((a > .5).astype(np.uint8), cv2.DIST_L2, 5) - .5 + a), (0, 0), 1.1); T = np.exp(-np.maximum(dt, 0) / (r * .020 * np.clip(1 + .75 * n2, .18, 2.6) + .6))
            self.banks.append(dict(x=x0 - 130 - wd * .25, a=a.astype(np.float16), T=T.astype(np.float16), side=sd, tr=tr, o=1.))

    def frame(self, t=0., part=1., op=1.):
        img = self.base.copy(); w = self.w
        for b in self.banks:
            ox = b['x'] + b['side'] * b['tr'] * (part * self.r * .95) + b['side'] * t * .10 * b['tr']
            bw = b['a'].shape[1]; i = int(math.floor(ox)); fr = ox - i; x0, x1 = max(0, i), min(w, i + bw + 1)
            if x1 <= x0: continue
            M = np.float32([[1, 0, fr], [0, 1, 0]]); a = cv2.warpAffine(b['a'].astype(f32), M, (bw + 1, self.h)); T = cv2.warpAffine(b['T'].astype(f32), M, (bw + 1, self.h))
            a = a[:, x0 - i:x1 - i] * op; T = T[:, x0 - i:x1 - i]; L = self.L[:, x0:x1]
            body = mix(rgb('#07050d'), rgb('#180b30'), np.clip(L * .5, 0, 1)[..., None]); k1 = np.clip(L * 1.05, 0, 1)[..., None]
            mid = mix(body, rgb('#4a159a'), k1); hot = mix(mid, mix(rgb('#9a48ff'), rgb('#eed6ff'), np.clip(L * 1.4 - .55, 0, 1)[..., None]), k1)
            t3, t2, t1 = smooth(.55, .69, T)[..., None], smooth(.35, .45, T)[..., None], smooth(.08, .16, T)[..., None]; c = body + (body * .28 + rgb('#12072a') * k1 * .35) * t1; c = c * (1 - t2) + mid * t2; c = c * (1 - t3) + hot * t3; A = a[..., None]
            img[:, x0:x1] = img[:, x0:x1] * (1 - A) + c * A
        return img


# ------------------------------------------------------------------ noise
@njit(cache=True, fastmath=True)
def _h3(x, y, z, s):
    n = (x * 374761393 + y * 668265263 + z * 2147483647 + s * 1274126177) & 0xffffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0xffffffff
    return ((n ^ (n >> 16)) & 0xffff) / 65535.0


@njit(cache=True, fastmath=True)
def _n3(x, y, z, s):
    ix = int(math.floor(x)); iy = int(math.floor(y)); iz = int(math.floor(z)); fx = x - ix; fy = y - iy; fz = z - iz
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz); v = 0.
    for dz in range(2):
        for dy in range(2):
            for dx in range(2): v += (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fz if dz else 1 - fz) * _h3(ix + dx, iy + dy, iz + dz, s)
    return v


@njit(cache=True, fastmath=True)
def _f3(x, y, z, s, o):
    a = .5; v = 0.; n = 0.
    for k in range(o):
        v += a * _n3(x, y, z, s + k * 17); n += a; x *= 2.07; y *= 2.07; z *= 2.07; a *= .5
    return v / n


# ------------------------------------------------------------------ sea and its road
@njit(cache=True, fastmath=True, parallel=True)
def _water(h0, h, w, hz, f, ch, cx, cy, r, KX, KZ, OM, AM, PH, t):
    n = KX.shape[0]; o = np.zeros((h - h0, w, 3), np.float32)
    for j in prange(h - h0):
        dy = h0 + j + .5 - hz
        Z = f * ch / dy; dZ = Z / dy; dX = Z / f
        for i in range(w):
            vx = (i + .5 - w * .5) / f; X = vx * Z; sx = 0.; sz = 0.; va = 0.
            for k in range(n):
                q = math.exp(-.30 * ((KZ[k] * dZ) ** 2 + (KX[k] * dX) ** 2)); ak = AM[k]
                if q > .01:
                    c = math.cos(KX[k] * X + KZ[k] * Z - OM[k] * t + PH[k]) * ak * q
                    sx += KX[k] * c; sz += KZ[k] * c
                va += (1 - q * q) * ak * ak * (KX[k] ** 2 + KZ[k] ** 2) * .5
            vy = -dy / f; nl = 1. / math.sqrt(1 + sx * sx + sz * sz); nx = -sx * nl; ny = nl; nz = -sz * nl
            dy0 = 120.
            dn = vx * nx + vy * ny + nz; rx = vx - 2 * dn * nx; ry = vy - 2 * dn * ny; rz = 1 - 2 * dn * nz
            o[j, i, 2] = dn
            if rz < .05: continue
            px = w * .5 + f * rx / rz; py = hz - f * ry / rz
            ex = px - cx; ey = py - cy; sg = f * 1.6 * math.sqrt(va)
            D = math.sqrt(ex * ex + ey * ey)
            hard = 1. / (1. + math.exp((D - r) / (r * .035)))
            s2 = r * r + 2 * sg * sg
            wide = r * r / s2 * math.exp(-(ex * ex) / s2 - .55 * (ey * ey) / s2)
            m = min(1., max(0., (dy0 - dy) / 70.)); wd = r * (.36 + .0032 * dy); cov = .80 * math.exp(-((i + .5 - cx) / wd) ** 2)
            nn = _n3((i - cx) / (2.5 + .34 * dy) + 40., math.log(dy + 2.) * 13., t * 1.7, 3); nn = nn * .72 + .28 * _n3((i - cx) / (1.2 + .12 * dy), math.log(dy + 2.) * 29., t * 2.6, 8)
            far = 1. / (1. + math.exp((nn - (.22 + .52 * cov)) * 34.)) * (1. if cov > .02 else 0.)
            o[j, i, 0] = hard * (1 - m) + far * m
            o[j, i, 1] = math.exp(-max(D - r, 0.) / (r * .30)) * (1 - m) + min(1., cov * 1.15 * (.45 + .8 * nn)) * m
    return o


def _waves(seed=5, n=26, calm=1.):
    r = np.random.default_rng(seed); lam = np.geomspace(.42, 15., n) * r.uniform(.85, 1.15, n); k = 2 * math.pi / lam
    th = r.normal(math.pi / 2, .62, n); slope = .036 * calm * np.where(lam < 1.1, .9, 1.) * r.uniform(.7, 1.25, n)
    return k * np.cos(th), k * np.sin(th), np.sqrt(9.8 * k), slope / k, r.uniform(0, 6.28, n)


def sea(img, hz=792., cx=C0[0], cy=C0[1], r=C0[2], t=0., seed=5, calm=1., f=1500., cam=1.7, amt=1., col=DEAD, deep=('#140830', '#07050e', '#030206')):
    """t in seconds of water time (slow it to a stop and the sea freezes).  calm = wave height factor"""
    h, w = img.shape[:2]; h0 = int(math.ceil(hz)); KX, KZ, OM, AM, PH = _waves(seed, calm=calm)
    b = _water(h0, h, w, float(hz), float(f), float(cam), float(cx), float(cy), float(r), KX, KZ, OM, AM, PH, float(t))
    k = (np.arange(h0, h) + .5 - hz) / (h - hz); g, v, dn = b[..., 0], b[..., 1], b[..., 2]
    base = ramp(np.repeat(k[:, None], w, 1), [(0, deep[0]), (.22, deep[1]), (1, deep[2])]); fre = np.clip(1 - np.abs(dn), 0, 1)
    base = base * (.80 + .5 * smooth(.55, .9, fre))[..., None]
    core = smooth(.34, .52, g); skin = smooth(.10, .20, g) * (1 - core); viol = smooth(.30, .62, v) * (1 - core)
    fall = (1 - .34 * k)[:, None, None]
    c = base + (viol * .30)[..., None] * rgb('#5a1fd0') * amt * fall
    c = c * (1 - skin[..., None] * .85) + rgb('#8a3cff') * (skin[..., None] * .85 * amt)
    c = c * (1 - core[..., None]) + rgb(col) * core[..., None] * (amt * fall)
    gl = cv2.resize(cv2.GaussianBlur(cv2.resize(core, (w // 4, max(2, (h - h0) // 4)), interpolation=cv2.INTER_AREA), (0, 0), 5), (w, h - h0))
    c = c + gl[..., None] * rgb('#7a2cf0') * .34 * amt
    x = np.arange(w); seam = np.exp(-((x - cx) / (r * 2.2)) ** 2)[None, :, None] * np.exp(-((np.arange(h0, h) + .5 - hz) / 2.0) ** 2)[:, None, None]
    img[h0:] = c + seam * rgb(LILAC) * .5 * amt
    return img


# ------------------------------------------------------------------ sparks
def _star(acc, x, y, s, a):
    P = []
    for k in range(8):
        q = s * (.17 if k % 2 else (.60 if k % 4 == 2 else 1.)); th = k * math.pi / 4; P.append((x + q * math.sin(th), y - q * math.cos(th)))
    cv2.fillPoly(acc, [np.round(np.array(P) * 16).astype(np.int32)], float(a), cv2.LINE_AA, 4)


def sparks(img, t=0., seed=1, groups=6, box=(90, 70, 1830, 740), avoid=None, amt=1., size=1., born=1., pins=150):
    """hang still; only a slow breath of brightness.  born 0..1 lights them one after another"""
    h, w = img.shape[:2]; r = np.random.default_rng(seed * 31 + 4); acc = np.zeros((h, w), f32); thr = np.zeros((h, w), f32); x0, y0, x1, y1 = box
    ok = lambda p: avoid is None or math.hypot(p[0] - avoid[0], p[1] - avoid[1]) > avoid[2]
    on = lambda q: float(np.clip((born * 1.12 - q) / .10, 0, 1))
    for g in range(groups):
        c = np.array([r.uniform(x0, x1), r.uniform(y0, y1)]); pts = [c]; m = int(r.integers(4, 7)); sz = r.uniform(4, 10, m); dl = r.uniform(0, 1, m)
        th = r.uniform(0, 6.28)
        for _ in range(m - 1): th += r.normal(0, .75); pts.append(pts[-1] + np.array([math.cos(th), math.sin(th) * .7]) * r.uniform(40, 104))
        for k, p in enumerate(pts):
            if not ok(p): continue
            tw = (.78 + .22 * math.sin(t * .16 + g * 2 + k * 1.7)) * on(dl[k]); _star(acc, p[0], p[1], sz[k] * size * (.7 + .3 * tw), tw)
            if k and ok(pts[k - 1]): q = pts[k - 1]; cv2.line(thr, tuple(np.round(q * 16).astype(int).tolist()), tuple(np.round(p * 16).astype(int).tolist()), .26 * min(on(dl[k]), on(dl[k - 1])), 1, cv2.LINE_AA, 4)
    for _ in range(pins):
        p = np.array([r.uniform(x0, x1), r.uniform(y0, y1)]); q, s, b = r.uniform(0, 1), r.uniform(.5, 1.3), r.uniform(.25, .9)
        if ok(p): cv2.circle(acc, tuple(np.round(p * 16).astype(int).tolist()), int(s * 16 * size), float(b * on(q) * (.7 + .3 * math.sin(t * .1 + p[0]))), -1, cv2.LINE_AA, 4)
    g = cv2.GaussianBlur(acc, (0, 0), 5 * size)
    img[:] = img + (g[..., None] * rgb(VIOLET) * .9 + thr[..., None] * rgb(VIOLET) + acc[..., None] * mix(rgb(LILAC), rgb(DEAD), .55)) * amt
    return img


# ------------------------------------------------------------------ the big ball
@njit(cache=True, fastmath=True, parallel=True)
def _ball(y0, y1, x0, x1, cx, cy, r, ca, sa, tilt, t, seed, PX, PY, gap, jit):
    """nz, large flow, fine grain, ridge, plate edge distance (unit disc), plate id hash"""
    o = np.zeros((y1 - y0, x1 - x0, 7), np.float32); n = PX.shape[0]; ct = math.cos(tilt); st = math.sin(tilt)
    for j in prange(y1 - y0):
        for i in range(x1 - x0):
            sx = (i + x0 + .5 - cx) / r; sy = (j + y0 + .5 - cy) / r
            if sx * sx + sy * sy >= 1.03: continue
            off = 0.; hid = 0.; a = 9.; b = 9.; pa = 0
            if gap > 0:
                ux = sx + .050 * (_n3(sx * 9, sy * 9, 3., seed) - .5) + .016 * (_n3(sx * 31, sy * 31, 5., seed) - .5); uy = sy + .050 * (_n3(sx * 9, sy * 9, 7., seed) - .5) + .016 * (_n3(sx * 31, sy * 31, 8., seed) - .5)
                rq = math.sqrt(ux * ux + uy * uy); lu = math.atan2(uy, ux) * 1.9; lv = math.log(rq + .045) * .92; per = 2 * math.pi * 1.9
                for k in range(n):
                    du = abs(lu - PX[k]); du = min(du, per - du); d = du * du + (lv - PY[k]) ** 2
                    if d < a: b = a; a = d; pa = k
                    elif d < b: b = d
                off = (math.sqrt(b) - math.sqrt(a)) * (rq + .045) * .55; hid = _h3(pa, 5, 9, seed)
            q = min(sx * sx + sy * sy, 1.); z = math.sqrt(1 - q); Y = sy * ct - z * st; zz = sy * st + z * ct; X = sx * ca + zz * sa; Z = -sx * sa + zz * ca
            sh = jit * (hid - .5) * .16
            X += sh; Y -= sh * .7
            wx = _f3(X * 1.5 + 3., Y * 1.5, Z * 1.5 + t, seed, 4) - .5; wy = _f3(X * 1.5 + 9., Y * 1.5 + 2., Z * 1.5, seed + 3, 4) - .5
            o[j, i, 0] = z; o[j, i, 1] = _f3(X * 1.9 + wx * 2.6, Y * 2.7 + wy * 2.6, Z * 1.9, seed + 7, 7)
            o[j, i, 2] = _f3(X * 19 + wx * 5, Y * 19 + wy * 5, Z * 19, seed + 11, 4)
            o[j, i, 3] = 1 - abs(2 * _f3(X * 4.2 + wx * 3, Y * 4.8 + wy * 3, Z * 4.2, seed + 19, 6) - 1)
            o[j, i, 4] = off; o[j, i, 5] = hid; o[j, i, 6] = 1.
    return o


def plates(seed=2, n=64):
    """fracture seeds in log-polar space: long radial plates, crushed small at the centre"""
    r = np.random.default_rng(seed); per = 2 * math.pi * 1.9; pts = []
    for _ in range(n * 40):
        p = np.array([r.uniform(-per / 2, per / 2), r.uniform(math.log(.045) * .92, math.log(1.15) * .92)])
        if all(math.hypot(min(abs(p[0] - q[0]), per - abs(p[0] - q[0])), (p[1] - q[1]) * .62) > .62 * r.uniform(.75, 1.2) for q in pts): pts.append(p)
        if len(pts) >= n: break
    P = np.array(pts); return P[:, 0].copy(), P[:, 1].copy()


def ball(cx, cy, r, spin=0., crack=0., t=0., seed=4, h=HH, w=W, tilt=.25, grain=1., dark=1.):
    """dead-white ball, violet maria, fine granulation, a hot limb.  crack 0..1: plates split outward from the centre, light in the gaps"""
    x0, x1 = max(0, int(cx - r - 3)), min(w, int(cx + r + 4)); y0, y1 = max(0, int(cy - r - 3)), min(h, int(cy + r + 4)); out = np.zeros((h, w, 4), f32)
    if x1 <= x0 or y1 <= y0: return out
    PX, PY = plates(seed + 5); b = _ball(y0, y1, x0, x1, float(cx), float(cy), float(r), math.cos(spin), math.sin(spin), tilt, float(t) * .05, seed, PX, PY, float(crack), float(crack))
    nz, fl, gr, rd, ed, hid = [b[..., k] for k in range(6)]; y, x = np.mgrid[y0:y1, x0:x1].astype(f32); sx, sy = (x + .5 - cx) / r, (y + .5 - cy) / r; q = np.sqrt(sx * sx + sy * sy); a = np.clip((1 - q) * r + .5, 0, 1)
    v = fl + .10 * (gr - .5) * grain
    c = ramp(v, [(.28, '#5a22c4'), (.385, '#7d45dc'), (.389, '#a97cf0'), (.425, '#b892f6'), (.429, '#dcc8fd'), (.47, '#e6d8ff'), (.474, '#f4edff'), (.56, DEAD), (1, '#ffffff')])
    c = mix(np.broadcast_to(rgb(DEAD), c.shape), c, .74 * dark)
    sp = smooth(.60, .64, gr) * smooth(.60, .44, fl); c = c * (1 - .20 * sp[..., None] * grain) + rgb('#9a66f2') * (.20 * sp[..., None] * grain)
    sp2 = smooth(.40, .36, gr) * smooth(.40, .34, fl); c = c + (sp2 * .16 * grain)[..., None] * rgb('#f0e2ff')
    rl = smooth(.90, .965, rd); c = c * (1 - .26 * rl[..., None]) + rgb('#8a4cf0') * (.26 * rl)[..., None]
    c = c * (1 + .10 * grain * (gr - .5)[..., None] * 2)
    lm = smooth(.58, 1.0, q); c = c * (1 - .50 * lm[..., None]) + rgb('#6a22e0') * (.50 * lm)[..., None]
    c = c + rgb('#f0dcff') * (smooth(1 - 5. / r, 1 - 1.5 / r, q) * .8)[..., None]
    if crack > 0:
        hid2 = .5 + .9 * smooth(.35, .75, rd); reach = crack * 1.55 - q + .16 * (hid - .5); wd = np.clip(reach, 0, 1.4) ** .9 * .046 * (.15 + crack ** 1.3) * (.35 + hid2) * (.45 + .75 * np.clip(q, 0, 1)); e = np.where(reach > 0, ed, 9.)
        core = smooth(wd, wd * .80, e) * (wd > .0010); rim = smooth(wd * 1.45 + .0040, wd * 1.30 + .0026, e) * (1 - core) * (reach > 0)
        c = c * (1 - .72 * rim[..., None]) + rgb('#3c0c96') * (.72 * rim)[..., None]                    # plate sides: the ball has thickness
        tone = (hid - .5) * .30 * np.clip(reach * 2, 0, 1) * crack; c = c * (1 - np.clip(tone, -.06, .3))[..., None]
        hot = mix(rgb('#fff3cf'), rgb('#ffffff'), smooth(.5, 0, q)[..., None]); c = c * (1 - core[..., None]) + hot * 1.2 * core[..., None]
        gl = cv2.GaussianBlur((core * a).astype(f32), (0, 0), max(2., r * .02)); c = c + (gl * .55 * crack)[..., None] * rgb('#ffe2b0')
        c = c + (np.exp(-(q / (.05 + .42 * crack ** 3)) ** 2) * crack ** 1.6)[..., None] * rgb('#fff6e0') * 1.0
    out[y0:y1, x0:x1] = np.dstack([c * a[..., None], a]); return out


def ball_halo(img, cx, cy, r, amt=.62):
    x, y = grid(*img.shape[:2]); d = np.hypot(x - cx, y - cy) - r
    img[:] = img + halo_rgb(np.where(d > -3, halo(d, min(r, 380)), 0) * amt); return img
