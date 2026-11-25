"""Light events of C, D and G.  No characters: everything takes a point, a radius or a silhouette mask.

    bend(img, x, y, R, amt)                        light and picture pulled toward a point (lens remap)
    threads(img, x, y, R, t, k, seed)              hard tapered filaments of light spiralling into the point
    pearl(img, x, y, r, t)                         the fingertip dot: dead-white core, violet rim, thin rays
    drain(img, amt, x, y, R, keep)                 colour leaves the frame from the edges toward the point (keep = radius that stays)
    drop(img, x, y, r, stretch)                    a hanging water drop: true flipped refraction of img
    beam_inside(mask, palm, t, vp, feet=, figure=) interior of the golden beam (or stops=LIGHT); streams part round ANY mask (float 0..1), RGB
    burst(img, cx, cy, r, k, t)                    g9: plates fly, rings, petals of light;  burnout(img, k) -> white
"""
import numpy as np, cv2, math
from numba import njit, prange
from . import W, H as HH
from .paint import rgb, f32, fbm, smooth, mix, grid, ramp, lum
from .moon import _n3, _h3, VIOLET, LILAC, DEAD
from .ink import ribbon


def _pts(P): return np.round(np.asarray(P) * 16).astype(np.int32)


# ------------------------------------------------------------------ c3
def bend(img, x, y, R=420., amt=.55, swirl=.35):
    X, Y = grid(*img.shape[:2]); dx, dy = X - x, Y - y; d = np.sqrt(dx * dx + dy * dy) + 1e-3
    g = np.exp(-(d / R) ** 1.25); k = 1 + amt * g * R / (d + R * .16); a = swirl * g * np.clip(1.4 - d / R, 0, 1)
    c, s = np.cos(a), np.sin(a)
    return cv2.remap(img, (x + (dx * c - dy * s) * k).astype(f32), (y + (dx * s + dy * c) * k).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def threads(img, x, y, R=620., t=0., k=1., seed=3, n=46, col=LILAC):
    """filaments fall into the point: each has its own life; long outside, thin and fast near the pearl"""
    h, w = img.shape[:2]; r = np.random.default_rng(seed); acc = np.zeros((h, w), f32)
    for i in range(n):
        th, per, ph, wd, r0 = r.uniform(0, 6.28), r.uniform(9, 18), r.random(), 1.2 + 6.5 * r.random() ** 2.2, R * r.uniform(.45, 1.2); a = (t / per + ph) % 1.
        head = r0 * (1 - a) ** 1.7; tail = min(r0, head + r0 * r.uniform(.28, .55) * (1 - a) + 16); u = np.linspace(head, tail, 22)
        an = th + 1.25 * np.log((u + 24) / (R * .5)); P = np.c_[x + np.cos(an) * u, y + np.sin(an) * u * .92]
        q = ribbon(P, wd * (.35 + .65 * (1 - a)), taper=(.12, .75), press=.2, seed=i, tremor=.2)
        if q is not None: cv2.fillPoly(acc, [_pts(q)], float(min(1, 2.2 * a * (1.15 - a)) * k), cv2.LINE_AA, 4)
    g = cv2.GaussianBlur(acc, (0, 0), 5)
    img[:] = img + acc[..., None] * mix(rgb(col), rgb(DEAD), .25) * .9 + g[..., None] * rgb(VIOLET) * .5
    return img


def pearl(img, x, y, r=13., t=0., amt=1.):
    h, w = img.shape[:2]; X, Y = grid(h, w); d = np.hypot(X - x, Y - y)
    g = .9 * np.exp(-np.maximum(d - r, 0) / (r * .55)) + .34 * np.exp(-np.maximum(d - r, 0) / (r * 2.6)) + .10 * np.exp(-np.maximum(d - r, 0) / (r * 9))
    img[:] = img + ramp(np.clip(g, 0, 1), [(0, '#000000'), (.1, '#1c0a44'), (.35, '#5a1fd0'), (.7, '#b16bff'), (1, '#e9ccff')]) * amt
    acc = np.zeros((h, w), f32); rr = np.random.default_rng(7)
    for i in range(14):
        th = rr.uniform(0, 6.28) + .02 * math.sin(t * .3 + i); L = r * rr.uniform(2.2, 8.5) * (.85 + .15 * math.sin(t * .9 + i * 2)); c, s = math.cos(th), math.sin(th); b = r * .16
        cv2.fillConvexPoly(acc, _pts([(x + c * r * .9 - s * b, y + s * r * .9 + c * b), (x + c * L, y + s * L), (x + c * r * .9 + s * b, y + s * r * .9 - c * b)]), .75, cv2.LINE_AA, 4)
    img[:] = img + acc[..., None] * rgb(LILAC) * amt
    a = np.clip(r - d + .5, 0, 1)[..., None]; e = smooth(r * .70, r * .98, d)[..., None]
    img[:] = img * (1 - a) + (rgb('#ffffff') * 1.2 * (1 - e) + rgb('#b16bff') * e) * a
    return img


def drain(img, amt, x=960., y=486., R=520., keep=.0, tint_='#b9b4c6'):
    """colour leaves from the edges inward; inside `keep` (0..1 of R) it stays"""
    X, Y = grid(*img.shape[:2]); d = np.hypot(X - x, (Y - y) * 1.1) / R
    k = np.clip(amt * 2.2 - .9 * np.clip(1.6 - d, 0, 1.6) * (1 - .0), 0, 1) if amt < 1 else np.ones_like(d)
    k = k * smooth(keep, keep + .35, d) if keep > 0 else k
    L = lum(img)[..., None]; g = L * rgb(tint_) / .72
    return img * (1 - k[..., None]) + g * k[..., None]


def drop(img, x, y, r=34., stretch=1.0, light=(-.45, -.6), tone='#cfc6ff'):
    """hanging drop: flipped refraction of the picture, dark Fresnel edge, two hard lights"""
    h, w = img.shape[:2]; X, Y = grid(h, w); u = (X - x) / r; v = (Y - y) / (r * stretch)
    tp = np.where(v < 0, 1 - .58 * np.clip(-v, 0, 1) ** 1.5, 1.); q = np.sqrt((u / tp) ** 2 + v * v); a = np.clip((1 - q) * r + .5, 0, 1)
    k = 2.6 + 5.5 * np.clip(q, 0, 1) ** 2
    ref = cv2.remap(img.astype(f32), (x - u * r * k * 1.2).astype(f32), (y - v * r * k).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    c = ref * 1.1 + .03; e = smooth(.74, .97, q); c = c * (1 - .82 * e[..., None]) + rgb('#120a28') * (.82 * e)[..., None]
    c = c + (smooth(.90, .95, q) * smooth(-.1, .7, u * .6 + v * .7))[..., None] * rgb(tone) * .8
    hl = (((u - light[0] * .8) / .20) ** 2 + ((v - light[1] * .62) / .30) ** 2) < 1; c = np.where(cv2.GaussianBlur(hl.astype(f32), (0, 0), .8)[..., None] > .5, rgb('#ffffff') * 1.05, c)
    A = a[..., None]; img[:] = img * (1 - A) + c * A; return img


# ------------------------------------------------------------------ d5: inside the beam
@njit(cache=True, fastmath=True)
def _bil(a, x, y):
    h, w = a.shape
    x = min(max(x, 0.), w - 1.001); y = min(max(y, 0.), h - 1.001); i = int(x); j = int(y); fx = x - i; fy = y - j
    return (a[j, i] * (1 - fx) + a[j, i + 1] * fx) * (1 - fy) + (a[j + 1, i] * (1 - fx) + a[j + 1, i + 1] * fx) * fy


@njit(cache=True, fastmath=True, parallel=True)
def _march(D, GX, GY, vx, vy, px, py, LL, steps, st):
    """walk upstream through the parted flow (coarse grid); where the line comes from = which stream this pixel belongs to"""
    h, w = D.shape; o = np.zeros((h, w, 4), np.float32)
    for j in prange(h):
        for i in range(w):
            x = i + .5; y = j + .5; mw = 0.
            for k in range(steps):
                ax = x - vx; ay = y - vy; q = math.sqrt(ax * ax + ay * ay) + 1e-3; ax /= q; ay /= q
                d = _bil(D, x, y); wt = math.exp(-max(d, 0.) / LL)
                if wt > .02:
                    gx = _bil(GX, x, y); gy = _bil(GY, x, y)
                    sg = math.tanh((-gy * (px - x) + gx * (py - y)) / 36.); lee = ax * gx + ay * gy
                    if lee < 0: sg = (max(sg, .75) if sg >= 0 else min(sg, -.75))
                    nk = .70 * math.exp(-d / 5.) + (.22 if lee >= 0 else 0.); tx = -gy * sg; ty = gx * sg          # upstream = back toward the parting hand (soft sign: no seams)
                    ax = ax * (1 - wt) + (tx * .86 + gx * nk) * wt; ay = ay * (1 - wt) + (ty * .86 + gy * nk) * wt
                    n = math.sqrt(ax * ax + ay * ay) + 1e-6; ax /= n; ay /= n
                x += ax * st; y += ay * st
                if wt > mw: mw = wt
            ax = x - vx; ay = y - vy; q = math.sqrt(ax * ax + ay * ay) + 1.
            o[j, i, 0] = ax / q; o[j, i, 1] = ay / q; o[j, i, 2] = math.log(q); o[j, i, 3] = mw
    return o


@njit(cache=True, fastmath=True, parallel=True)
def _tex(M, t, seed, off, fq, pf):
    h, w = M.shape[0], M.shape[1]; o = np.zeros((h, w, 2), np.float32)
    for j in prange(h):
        for i in range(w):
            c = M[j, i, 0]; sn = M[j, i, 1]; n = math.sqrt(c * c + sn * sn) + 1e-6; c /= n; sn /= n; ph = (M[j, i, 2] * 1.5 + off) * pf + t; c *= fq; sn *= fq
            o[j, i, 0] = .62 * _n3(c * 7, sn * 7, ph * .8 + 40, seed) + .38 * _n3(c * 19, sn * 19, ph * 1.5 + 40, seed + 5)
            o[j, i, 1] = .6 * _n3(c * 34, sn * 34, ph * 2.1 + 60, seed + 9) + .4 * _n3(c * 70, sn * 70, ph * 2.7 + 60, seed + 13)
    return o


BEAM = [(0, '#7a3208'), (.26, '#b85a0e'), (.262, '#e08a14'), (.42, '#f6a416'), (.422, '#ffc630'), (.58, '#ffd54a'), (.582, '#ffe680'), (.74, '#fff0a8'), (.742, '#fffbe2'), (1, '#ffffff')]
LIGHT = [(0, '#2a1a86'), (.20, '#4a34c4'), (.202, '#6f86f0'), (.34, '#8fb4ff'), (.342, '#b9d4ff'), (.50, '#dbe8ff'), (.502, '#f3efff'), (.66, '#fffaf0'), (.662, '#fffdf8'), (1, '#ffffff')]      # g9: inside the white light (violet, blue and gold burnt out)
LIGHT_GOLD = [(0, '#5a2a0a'), (.18, '#a85a10'), (.182, '#e79a1c'), (.34, '#ffc640'), (.342, '#ffe08a'), (.50, '#fff0c0'), (.502, '#fff8e2'), (.66, '#fffdf4'), (.662, '#fffefa'), (1, '#ffffff')]
LIGHT_BLUE = [(0, '#0c1466'), (.22, '#1c36c0'), (.222, '#2f6cf0'), (.38, '#62a8ff'), (.382, '#8fd0ff'), (.52, '#c8ecff'), (.522, '#eaf8ff'), (.68, '#fbfeff'), (.682, '#ffffff'), (1, '#ffffff')]
_ns = {}


def _lut(stops, n=1024): return ramp(np.linspace(0, 1, n, dtype=f32), stops).astype(f32)


def _rag(seed):
    if seed not in _ns: _ns[seed] = fbm(HH, W, (26, 10, 4), (1, .6, .3), seed).astype(f32)
    return _ns[seed]


def beam_inside(mask, palm=None, t=0., vp=None, seed=4, reach=150., q=3, body='#1a0c08', wrap=1., stops=None, tatter=1., feet=(), figure=None, floor=None, anchor=None, glow='#8a3a0a', heart=1., hush=0., lift=0., freq=1., length=1., wake=0.):
    """mask: float silhouette (1080,1920) 0..1.  palm = (x, y) that parts the light.  t grows = light flows AWAY from us into vp (use -t for light from vp).
    feet = [(x, y, weight 0..1)] planted soles: bow splash, contact shadow and a wake on the floor.  floor = screen y of the ground's vanishing line (or None).
    figure = premult RGBA drawn instead of the burnt silhouette (light still wraps its edge).  tatter eats the outline (pattern sticks to anchor).  hush 0..1 stills the streams to flat light."""
    h, w = mask.shape; ys, xs = np.nonzero(mask[::4, ::4] > .5); xs = xs * 4; ys = ys * 4
    if vp is None: vp = (xs.mean(), ys.min() * .55 + ys.mean() * .45) if len(xs) else (w / 2, h / 2)
    if palm is None: palm = vp
    dd = cv2.distanceTransform((mask > .5).astype(np.uint8), cv2.DIST_L2, 3)
    if tatter > 0 and figure is None:
        n = _rag(seed + 3); n = np.roll(n, (int(anchor[1]) % h, int(anchor[0]) % w), (0, 1)) if anchor is not None else n
        th = cv2.resize(cv2.dilate(cv2.resize(dd, (w // 4, h // 4)), np.ones((7, 7), np.uint8)), (w, h)); rg = smooth(-.4, .6, n * 3.2 * tatter + dd - 1.2 * tatter + 1.2); mask = mask * np.where((dd < 11) & (th > 15), rg, 1.)
        dd = cv2.distanceTransform((mask > .5).astype(np.uint8), cv2.DIST_L2, 3)
    sm = cv2.resize(mask, (w // q, h // q), interpolation=cv2.INTER_AREA).astype(f32)
    D = cv2.distanceTransform((sm < .5).astype(np.uint8), cv2.DIST_L2, 5) if len(xs) else np.full(sm.shape, 999., f32); Ds = cv2.GaussianBlur(D, (0, 0), 5)
    GX = cv2.Sobel(Ds, cv2.CV_32F, 1, 0) / 8; GY = cv2.Sobel(Ds, cv2.CV_32F, 0, 1) / 8; gn = np.sqrt(GX * GX + GY * GY) + 1e-4; GX /= gn; GY /= gn
    M = _march(Ds, GX, GY, vp[0] / q, vp[1] / q, palm[0] / q, palm[1] / q, reach / q, 42, 14. / q)
    mb = cv2.GaussianBlur(M, (0, 0), 2.2); ro = cv2.GaussianBlur(cv2.dilate(np.abs(M[..., 0] - mb[..., 0]) + np.abs(M[..., 1] - mb[..., 1]), np.ones((5, 5), np.uint8)), (0, 0), 5) * max(1., freq * .8); ro = smooth(.004, .022, ro)      # where lines of flow collide: a calm wake instead of moire
    M = M * (1 - ro[..., None]) + cv2.GaussianBlur(M, (0, 0), 14) * ro[..., None]; calm = cv2.resize(ro.astype(f32), (w, h), interpolation=cv2.INTER_LINEAR)
    if wake > 0 and len(xs):                                                  # light comes from the left: behind the body it is calm, and the shadow widens softly
        sw = np.maximum.accumulate(cv2.GaussianBlur(sm, (0, 0), 3), 1); sw = cv2.GaussianBlur(sw, (0, 0), sigmaX=8, sigmaY=26); lee = np.clip(sw * (1 - sm), 0, 1) ** 1.2 * smooth(0, 70, np.cumsum(sm > .5, 1)); gy_, gx_ = np.mgrid[:sm.shape[0], :sm.shape[1]].astype(f32); bx, by = gx_ + .5 - vp[0] / q, gy_ + .5 - vp[1] / q; bq = np.sqrt(bx * bx + by * by) + 1.
        M0 = np.dstack([bx / bq, by / bq, np.log(bq + 42 * 14. / q), M[..., 3]]).astype(f32); M = M * (1 - wake * lee[..., None]) + M0 * (wake * lee)[..., None]; calm = np.maximum(calm * (1 - cv2.resize(lee.astype(f32), (w, h))), .45 * wake * cv2.resize(lee.astype(f32), (w, h), interpolation=cv2.INTER_LINEAR))
    o = _tex(cv2.resize(cv2.GaussianBlur(M, (0, 0), 1.1), (w // 2, h // 2), interpolation=cv2.INTER_CUBIC), float(t), seed, math.log(q / 2.) * 1.5, float(freq), float(1. / length)); o = cv2.resize(cv2.GaussianBlur(o, (0, 0), .8), (w, h), interpolation=cv2.INTER_CUBIC); Df = cv2.resize(D, (w, h), interpolation=cv2.INTER_LINEAR) * q
    X, Y = grid(h, w); rad2 = ((X - vp[0]) / w) ** 2 + ((Y - vp[1]) / h) ** 2
    k = (1 - hush) * (1 - .88 * calm); v = .50 + k * (1.15 * (o[..., 0] - .5) + .55 * (o[..., 1] - .5)) + heart * .34 * np.exp(-rad2 / .04) - .30 * k * smooth(.28, .75, np.sqrt(rad2)) + .5 * hush + lift
    if len(xs): v = v + k * (.26 * np.exp(-((Df - 11) / 7) ** 2) - .30 * np.exp(-((Df - 46) / 22) ** 2))               # bow shock, then the hollow behind it
    v = v + .65 * k * np.exp(-((X - palm[0]) ** 2 + (Y - palm[1]) ** 2) / 6400.)
    sh = None
    if floor is not None:
        fl = smooth(0, 90, Y - floor); v = v - .13 * fl                                                               # ground: one value lower, streams lie down on it
        sh = np.zeros((h, w), f32)
        for (fx, fy, fw) in feet:
            if fw <= 0: continue
            sc = np.clip((fy - floor) / 260., .25, 2.2); u = (X - fx) / (64 * sc); z = (Y - fy) / (15 * sc); e = u * u + z * z
            nz = 1 + .55 * o[..., 1]; v = v + fw * .50 * np.exp(-((np.sqrt(e + 1e-4) - 1.5 * nz) / (.13 + .26 * o[..., 0])) ** 2) * smooth(-.15, .9, z) * smooth(.40, .56, o[..., 0]); sh = np.maximum(sh, fw * np.where(e < 1.25, 1., 0.))
            zz = (fy - Y) / np.maximum(fy - floor, 30.); wk = (zz > 0) & (zz < .92); uu = (X - (fx + (vp[0] - fx) * zz * .8)) / (40 * sc * (1 - .80 * np.clip(zz, 0, 1))); sh = np.maximum(sh, fw * .55 * np.where(wk & (np.abs(uu) < 1), 1 - zz, 0))
    c = (stops if isinstance(stops, np.ndarray) else _lut(stops or BEAM))[np.clip(v * 1023, 0, 1023).astype(np.int16)]
    c = c + (np.exp(-rad2 / .0676) * .16 * heart)[..., None] * rgb('#fff3c0')
    if sh is not None: sh = cv2.GaussianBlur(sh, (0, 0), 1.4); c = c * (1 - .72 * sh[..., None]) + rgb('#3a1506') * (.62 * sh)[..., None]
    if not len(xs): return c.astype(f32)
    L = cv2.resize(cv2.GaussianBlur(cv2.resize(c, (w // 4, h // 4), interpolation=cv2.INTER_AREA), (0, 0), 4), (w, h)); rim = np.exp(-dd / 5.) * wrap; mid = np.exp(-dd / 30.) * wrap
    if figure is not None:
        A = figure[..., 3:4]; e = (np.exp(-dd / 3.2) * wrap * .85 * A[..., 0])[..., None]; return (c * (1 - A) + figure[..., :3] * (1 - e) + L * e * A + (mid * .16)[..., None] * L * A).astype(f32)
    b = np.where((rim > .42)[..., None], L * .95 + .08, rgb(body) + mid[..., None] * rgb(glow) * .75); A = np.clip(mask, 0, 1)[..., None]
    return (c * (1 - A) + b * A).astype(f32)


# ------------------------------------------------------------------ g9
def burst(img, cx, cy, r, k, t=0., seed=6, petals=110, rings=4):
    """k 0..1: the ball has broken.  plates tumble out, thick rings open, petals of light fall away"""
    h, w = img.shape[:2]; rr = np.random.default_rng(seed); X, Y = grid(h, w)
    for i in range(rings):
        a = k * 1.30 - i * .15
        if a <= 0: continue
        R = r * (.30 + 2.7 * a ** .72 * (1 - .12 * i)); sq, ro = ((.30, -.16), (.36, -.13), (.24, -.19), (.44, .42))[i % 4]; c, sn = math.cos(ro), math.sin(ro)
        u = (X - cx) * c + (Y - cy) * sn; v = (-(X - cx) * sn + (Y - cy) * c) / sq; an = np.arctan2(v, u); d = np.hypot(u, v) - R
        th = r * .085 * (1 - .62 * min(a, 1)) * (.42 + .58 * (.5 + .5 * np.sin(an + .25)) ** 1.3) * (1 + .22 * np.sin(an * 3 + i * 2) + .10 * np.sin(an * 11 + i)) * (1 - smooth(.62, 1.12, a)) + .01; m = smooth(th, th * .84, np.abs(d))
        hot = smooth(th * .46, th * .34, np.abs(d + th * .22)); col = mix(rgb('#b98aff'), rgb('#ffffff'), hot[..., None]); img[:] = img * (1 - .85 * m[..., None]) + col * (.98 * m)[..., None]
    lay = np.zeros((h, w, 4), f32)
    fade = 1 - smooth(.62, 1., k); e1 = 1 - (1 - k) ** 2.4
    for i in range(34):                                                 # shell plates: long, radial, tumbling, with a lit face and a thickness
        th = (i + rr.uniform(-.3, .3)) / 34 * 6.2832; q0, L, wd, sp, ph = rr.uniform(.35, .95), r * rr.uniform(.08, .34) * rr.uniform(.5, 1.) * (.05 + .95 * fade), rr.uniform(.28, .62), rr.uniform(.35, 1.5), rr.uniform(0, 6.28); d = r * q0 + r * sp * e1 * 1.9; x, y = cx + math.cos(th) * d, cy + math.sin(th) * d * .94
        n = int(rr.integers(5, 8)); an = np.sort(np.linspace(0, 6.28, n, endpoint=False) + rr.normal(0, .28, n)); rad = rr.uniform(.62, 1.1, n); f = .30 + .70 * abs(math.cos(ph + k * 3.4 * sp)); ro = -th + (k * 2.4 * sp) * (1 if i % 2 else -1)
        P = np.c_[np.cos(an) * rad * L, np.sin(an) * rad * L * wd * f] @ np.array([[math.cos(ro), -math.sin(ro)], [math.sin(ro), math.cos(ro)]]) + (x, y); o2 = np.array([math.cos(th), math.sin(th)]) * L * .16
        for Q, col in ((P + o2, '#4a179e'), (P + o2 * .45, '#8a4cf0'), (P, '#f3eaff')):
            c = rgb(col); cv2.fillPoly(lay, [_pts(Q)], (float(c[0]), float(c[1]), float(c[2]), 1.), cv2.LINE_AA, 4)
        c = rgb('#d4bcfa'); cv2.fillPoly(lay, [_pts(np.vstack([P[:n // 2 + 1], P.mean(0)[None] * .6 + P[0] * .4]))], (float(c[0]), float(c[1]), float(c[2]), 1.), cv2.LINE_AA, 4)
    for i in range(petals):
        near = rr.random() < .12; th, q0, L, sp, ph = rr.uniform(0, 6.28), rr.uniform(.1, 1.), r * (rr.uniform(.10, .19) if near else rr.uniform(.028, .085)), rr.uniform(.5, 2.1), rr.uniform(0, 6.28); e = 1 - (1 - k) ** 2; d = r * q0 * .6 + r * sp * e * (1.8 if near else 1.25); x = cx + math.cos(th) * d + math.sin(ph + t * .07) * r * .07; y = cy + math.sin(th) * d * .86 + r * .30 * k * k * sp
        ro = ph + t * .05 * sp; f = .25 + .75 * abs(math.cos(ph * 2 + t * .075)); bow = .34 * math.sin(ph * 3); u = np.linspace(0, 1, 12); ax = (u * 2 - 1) * L; by = bow * L * (1 - (u * 2 - 1) ** 2); wb = np.sin(math.pi * u ** .72) ** .9 * L * .46 * f; R_ = np.array([[math.cos(ro), math.sin(ro)], [-math.sin(ro), math.cos(ro)]])
        up, dn, md = np.c_[ax, by - wb], np.c_[ax, by + wb * .8], np.c_[ax, by + wb * .12]; al = float(np.clip(k * 8, 0, 1)); L *= 1
        c = rgb(['#ffffff', '#f2e2ff', '#e2c8ff', '#fff3d6'][i % 4]); s = rgb(['#d9c0ff', '#b68cf5', '#9a66f0', '#f0d7a8'][i % 4])
        cv2.fillPoly(lay, [_pts(np.vstack([up, dn[::-1]]) @ R_ + (x, y))], tuple(float(v * al) for v in s) + (al,), cv2.LINE_AA, 4); cv2.fillPoly(lay, [_pts(np.vstack([up, md[::-1]]) @ R_ + (x, y))], tuple(float(v * al) for v in c) + (al,), cv2.LINE_AA, 4)
    img[:] = img * (1 - lay[..., 3:]) + lay[..., :3]
    return img


def burnout(img, k, cx=960., cy=486., r=208.):
    """the frame goes to white from the point outward, with a hard wandering front"""
    if k <= 0: return img
    h, w = img.shape[:2]; X, Y = grid(h, w); d = np.hypot(X - cx, Y - cy); n = fbm(h, w, (170, 60), (1, .4), 9)
    R = r + (1500 - r) * k ** 1.4; a = smooth(R + 10, R - 10, d + n * 40 * k * smooth(.05, .4, k)); g = np.exp(-np.maximum(d - R, 0) / (90 + 320 * k))
    o = img + g[..., None] * rgb('#ecd8ff') * .9 * min(1, k * 2); return o * (1 - a[..., None]) + rgb(DEAD) * 1.06 * a[..., None]
