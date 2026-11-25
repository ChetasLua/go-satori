"""ki scene: the places my shots need (sea plane, cel cumulus, rocks, moon), drawn with the same hard shapes."""
from .ki_core import *
from .ki_core import _vn, _rib, _smax  # noqa
from .cam import Camera3D


def puffs(F, pts, light=(-.5, -.8), lv=((1., 0.), (.95, .22), (.86, .52), (.70, .92)), k=7.):
    """cel cumulus: pts = [(x, y, r, squash)] discs in a Field; inner tones slide toward the light"""
    lx, ly = light
    for x, y, r, *q in pts:
        sq = q[0] if q else 1.; e = r * (1 - sq) if sq < 1 else 0.; p0, p1 = (x - e, y), (x + e, y); n = 3; P = np.linspace(p0, p1, n) if e > .5 else np.array([[x - .2, y], [x, y], [x + .2, y]])
        hw = [np.full(n, (r - e * .55) * a) for a, o in lv[:F.L]]; Q = [P + np.array([lx, ly]) * r * o * .5 for a, o in lv[:F.L]]
        _ribs(F, Q, hw, k)
    return F


def _ribs(F, Q, hw, k):
    from .ki_core import _rib
    if F.ok: _rib(F.D, F.ox, F.oy, np.stack(Q), np.stack(hw), float(k))


def cumulus(pts, light=(.6, -.8), cols=('#1d1b4c', '#37317a', '#7466bd', '#c3b4f4', '#efe6ff'), ink='#0c0a26', seed=1, k=9., lw=1.3, box=None):
    """cel cumulus with scalloped tones: every lobe grows its own smaller lobes on the lit side, so each tone edge is a row of curls, not a shifted copy.
    cols = deep, shadow, half, lit, top.  -> premult layer"""
    lx, ly = light; n = math.hypot(lx, ly); lx, ly = lx / n, ly / n; a0 = math.atan2(ly, lx); F = [Field(1, box) for _ in range(5)]; one = lambda x, y, r: (np.array([[x - .2, y], [x, y], [x + .2, y]]), [np.full(3, r)])
    for i, (x, y, r, *q) in enumerate(pts):
        sq = q[0] if q else 1.; e = r * (1 - sq); P = np.array([[x - e, y], [x, y], [x + e, y]]) if e > .5 else one(x, y, r)[0]
        F[0].rib(P, [np.full(3, r - e * .5)], k); F[1].rib(P + np.array([lx, ly]) * r * .10, [np.full(3, (r - e * .5) * .93)], k)
        if r < 5: continue
        F[2].rib(P + np.array([lx, ly]) * r * .30, [np.full(3, (r - e * .5) * .66)], k * .6); m = 7 if r > 40 else 5 if r > 14 else 3; F[3].rib(P + np.array([lx, ly]) * r * .52, [np.full(3, (r - e * .5) * .32)], k * .5)
        for j in range(m):
            b = a0 + (j - (m - 1) / 2) * (2.3 / m) + .25 * (h1(seed, i, j) - .5); d = r * (.52 + .08 * h1(seed, i, j, 2)); rr = r * (.25 + .12 * h1(seed, i, j, 3)); cx, cy = x + math.cos(b) * d, y + math.sin(b) * d * (.85 if sq < 1 else 1.); F[2].rib(*one(cx, cy, rr * 1.15), k * .4); F[3].rib(*one(cx + lx * rr * .25, cy + ly * rr * .25, rr * .82), k * .4)
        if r > 12:                                            # edge light: a thin broken crescent, never a dot
            v = lin(12); a_ = a0 + (-.75 + .2 * h1(seed, i, 41)) + (1.1 + .4 * h1(seed, i, 42)) * v; F[4].rib(np.stack([x + np.cos(a_) * r * .84, y + np.sin(a_) * r * .84], -1), [r * .045 * np.sin(math.pi * v) ** .8 * (.5 + h1(seed, i, 43)) - .15], 0.)
    o = F[0].cel((cols[0],), lw=(lw,), ink=(ink,), seed=seed); m0 = F[0].mask()[..., None]
    for f, c, w in ((F[1], cols[1], 0), (F[2], cols[2], lw * .55), (F[3], cols[3], 0), (F[4], cols[4], 0)): l = f.cel((c,), lw=(w,), ink=(cols[0],), seed=seed + 1); o = stack(o, l * m0)
    return o


def heap(cx, base, w, h, seed, n=26, flat=.12):
    """disc list of one cumulus: flat foot, cauliflower top, big / medium / small"""
    r = R(seed, 31); o = []
    for i in range(n):
        u = r.uniform(-1, 1); top = (1 - abs(u) ** 1.8) * (.55 + .45 * math.sin(u * 3 + seed)); z = bms(r.random(), .28, .36); y = r.random() ** .8 * top; o.append((cx + u * w * .5 * (1 - .25 * y), base - y * h * .86 - h * .04, h * (.10 + .26 * z) * (1 - .35 * abs(u)), 1.))
    for i in range(int(n * .9)):
        u = r.uniform(-1.1, 1.1); o.append((cx + u * w * .55, base - h * flat * .4 * r.random(), h * (.05 + .10 * r.random()), .55))
    return o


def sea_uv(cam, z=0.):
    """per-pixel sea coordinates (u right, v forward, t depth) ; t <= 0 above the horizon"""
    return cam.uv((0, 0, z), (1., 0, 0), (0, 1., 0))


@njit(cache=True, fastmath=True)
def _seaf(U, V, T, s, t, o):
    h, w = U.shape
    for j in range(h):
        for i in range(w):
            d = T[j, i]
            if d <= 0: o[j, i] = 0; continue
            u = U[j, i]; v = V[j, i]; a = 0.; wt = 0.
            for q in range(5):
                L = s * 2.4 ** q; f = L / (d * .02 + 1e-6); g = min(1., max(0., (f - .30) * 1.6)) * min(1., max(0., (9. - f) * .5 + .6))
                if g <= 0: continue
                wx = _vn(u / (L * 5.1) + q * 7.1, v / (L * 1.3) + t * .5, q * 3.7 + t * .35); n = _vn(u / (L * 4.2) + wx * .7, v / L + wx * .5 - t * .9 / (1 + q), q * 9.1 + t * .6); a += g * (1 - 2 * abs(n)); wt += g
            o[j, i] = a / wt if wt > 0 else 0


def sea_waves(U, V, T, t, s=1.6):
    """ridged wave field -1..1 with scale chosen by distance so glints keep their size on screen"""
    o = np.zeros(U.shape, f32); _seaf(U.astype(f32), V.astype(f32), T.astype(f32), s, float(t), o); return o


def rock(c, P, seed, cols=('#1b1a33', '#3a3560', '#6a62a0'), light=(-.5, -.8), edge=None, w=1.6, ink='#0b0a18'):
    """cel rock: shadow flat, one hard zigzag terminator, a lit plane on top of it, crack, pen line; edge = (colour, dx, dy) hard rim on the side facing (dx, dy)"""
    P = np.asarray(P, float); r = R(seed, 13); ctr = P.mean(0); L = np.asarray(light, float); L = L / np.hypot(*L); T = np.array([-L[1], L[0]]); s = np.ptp(P @ T) + 1e-6; h = np.ptp(P @ L) + 1e-6; c.fill(P, cols[0]); c.clip(P)
    def plane(off, col, amp):
        k = 5; u = np.linspace(-.7, .7, k); z = ctr + T * (u * s)[:, None] + L * ((off + amp * r.normal(0, 1, k)) * h)[:, None]; c.fill(np.vstack([z, ctr + T * s + L * h * 1.2, ctr - T * s + L * h * 1.2]), col)
    plane(-.02, cols[1], .10); plane(.20, cols[2], .08)
    if edge is not None:
        col, dx, dy = edge; d = np.array([dx, dy], float); l = np.hypot(*d); d = d / l; n = len(P)
        for i in range(n):
            a, b = P[i], P[(i + 1) % n]; m = (a + b) / 2 - ctr; m = m / (np.hypot(*m) + 1e-6); k = m @ d
            if k > .25: c.fill(np.array([a, b, b - d * l * .10 * r.uniform(.5, 1.6), (a + b) / 2 - d * l * .22 * k * r.uniform(.6, 1.5), a - d * l * .10 * r.uniform(.5, 1.6)]), col)
    if h > 26: q = ctr + T * s * r.uniform(-.2, .2); c.line(np.array([q + L * h * .3, q + T * s * .06, q - L * h * .28 + T * s * r.uniform(-.1, .1)]), max(1., w * .8), ink, taper=(.1, .7), seed=seed)
    c.unclip()
    if w:
        n = len(P)
        for i in range(n): c.line(np.array([P[i], (P[i] + P[(i + 1) % n]) / 2, P[(i + 1) % n]]), w, ink, taper=(.1, .1), seed=seed + i, tremor=.15)
    return c


def chunk(cx, cy, r, seed, rot=0., st=1.5, n=7):
    g = R(seed, 3); a = np.sort((np.arange(n) + g.uniform(-.32, .32, n)) / n * TAU); q = r * g.uniform(.62, 1.15, n); P = np.stack([np.cos(a) * q * st, np.sin(a) * q], -1); c, s = math.cos(rot), math.sin(rot); return np.stack([cx + P[:, 0] * c - P[:, 1] * s, cy + P[:, 0] * s + P[:, 1] * c], -1)


def moon(cx, cy, r, seed=1, col='#fbf7ff', rim='#b98cff', deep='#8a2cff', tone='#e6d8ff'):
    """dead white disc with a violet rim and two faint hard tones -> (layer, light)"""
    x, y = grid(); d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2); a = np.clip(r - d + .5, 0, 1); n = nimg(H, W, r * .5, 0, seed, 3, 4); c = np.empty((H, W, 3), f32); c[:] = C(col); e = (d + n * r * .05) > r * .90; c[e] = C(tone); e2 = (d + n * r * .03) > r * .965; c[e2] = C(rim); L = np.dstack([c * a[..., None], a]); g = np.exp(-np.maximum(d - r, 0) / (r * .16)) * .55 + np.exp(-np.maximum(d - r, 0) / (r * .5)) * .22; return L, (g * (1 - a))[..., None] * C(deep)
