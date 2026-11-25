"""Things that grow, drawn mark by mark: grass, reeds, leaves, trees, palms, the red spider lily; and rain."""
import numpy as np, cv2, math
from . import W, H
from .paint import rgb, f32, fbm, smooth, mix, grid, Canvas, arcs, jit_color
from .ink import Cel, spl, resample
from .comp import blur, over
from .land_core import C, hx, FAR


# ------------------------------------------------------------------------------------------------ grass on real ground
def blades(img, cam, z, P, hgt, cols, r, wind=0., lean=.25, width=.020, n_max=60000, light=None, tip=.35, eps=.10, alpha=None, bend=.5):
    """individually directed blades standing on world points P (n,3), z-tested against the depth map z.  hgt (n,) metres.
    cols = [(dark foot), (mid), (light tip)] hex; each blade picks its own place between them.  light (n,) 0..1 optional"""
    s, zc = cam.project(P); ok = (zc > .3) & (s[:, 0] > -60) & (s[:, 0] < cam.w + 60) & (s[:, 1] > -60) & (s[:, 1] < cam.h + 200)
    ix = np.clip(s[:, 0].astype(int), 0, cam.w - 1); iy = np.clip(s[:, 1].astype(int), 0, cam.h - 1); inside = (s[:, 1] < cam.h - 1) & (s[:, 1] > 0) & (s[:, 0] > 0) & (s[:, 0] < cam.w - 1)
    ok &= ~inside | (zc < z[iy, ix] * (1 + eps) + .05); ppm = cam.f / np.maximum(zc, .1); px = hgt * ppm; ok &= px > 2.2
    k = np.nonzero(ok)[0]; k = k[np.argsort(-zc[k])]; k = k[-n_max:] if len(k) > n_max else k
    if len(k) == 0: return img
    s, px, ppm = s[k], px[k], ppm[k]; n = len(k); wind = wind[k] if np.ndim(wind) else wind; th = r.normal(0, lean, n) + wind; cu = r.normal(0, bend, n) + wind * .9
    t = np.linspace(0, 1, 5)[None]; ang = -np.pi / 2 + th[:, None] + cu[:, None] * t; st = px[:, None] / 4; x = s[:, :1] + np.cumsum(np.c_[np.zeros(n), (np.cos(ang[:, :4]) * st)], 1); y = s[:, 1:2] + np.cumsum(np.c_[np.zeros(n), (np.sin(ang[:, :4]) * st)], 1)
    cs = [C(c) for c in cols]; q = r.random(n) if light is None else np.clip(light[k] + r.normal(0, .16, n), 0, 1)
    cd = mix(cs[0], cs[1], np.clip(q * 2, 0, 1)[:, None]); cl = mix(cs[1], cs[2], np.clip(q * 2 - 1, 0, 1)[:, None]); col = np.where((q < .5)[:, None], cd, cl) * (1 + r.normal(0, .05, (n, 1))); col = np.clip(col + r.normal(0, .012, (n, 1)) * [1.2, .2, -1.2], 0, 1.3)
    foot = np.clip(mix(cs[0], col, .42), 0, 1); tipc = np.where((r.random(n) < tip * (.35 + .65 * q))[:, None], mix(col, cs[2] * 1.06, .75), col)
    from .grass import draw_blades
    draw_blades(img, dict(x=x[:, 0], y=y[:, 0], mx=x[:, 2], my=y[:, 2], tx=x[:, 4], ty=y[:, 4], w=np.maximum(.45, width * ppm * r.uniform(.6, 1.3, n)), c0=foot, c1=tipc, op=r.uniform(.85, 1., n), kind=np.zeros(n)), alpha)
    return img


def fan_leaf(cv, x, y, ang, L, wd, cols, r, n=1, droop=.5, seed=0):
    P = arcs(np.full(n, x) + r.normal(0, 2, n), np.full(n, y) + r.normal(0, 2, n), ang + r.normal(0, .3, n), L * r.uniform(.7, 1.1, n), r.normal(droop, .2, n), K=6); P += (np.array([x, y]) - P[:, 0])[:, None]
    cv.strokes(P, wd * r.uniform(.7, 1.1, n), np.stack([C(cols[int(k)]) for k in r.integers(0, len(cols), n)]), shape='leaf', seed=seed, bristle=.02)


def leaves(cv, cx, cy, R, n, tones, r, light=(-.5, -.8), size=(6, 14), squash=.8, hang=.35, dens=1., lit=1.):
    """one clump of foliage: dark mass first, then half tone, then lit leaves toward the light; edges are single leaves"""
    for k, (c, f, off, sc) in enumerate([(tones[0], 1.0, 0., 1.15), (tones[1], .8, .16, 1.), (tones[2], .42, .42, .9), (tones[3], .16, .66, .75)]):
        m = int(n * f * dens * (lit if k > 1 else 1.)); a = r.uniform(0, 6.28, m); q = np.sqrt(r.random(m)) * (1.0, .84, .62, .42)[k]; x = cx + R * (np.cos(a) * q + light[0] * off); y = cy + R * squash * (np.sin(a) * q + light[1] * off)
        ln = r.uniform(*size, m) * sc; an = np.arctan2(y - cy, x - cx) + r.normal(0, .7, m) + hang * np.sign(np.cos(a)); P = arcs(x, y, an, ln * 1.5, r.normal(0, .5, m), K=4)
        col = np.clip(C(c)[None] * (1 + r.normal(0, .07, (m, 1))) + r.normal(0, .015, (m, 1)) * [1.2, .3, -1], 0, 1.2); cv.strokes(P, ln * r.uniform(.28, .46, m), col, shape='leaf', seed=int(r.integers(9999)), bristle=.02)


def tree(cv, x, y, h, tones, bark, r, light=(-.5, -.8), spread=.42, clumps=26, lean=.0, size=(6, 14), trunk=.045, crown=(.35, 1.0)):
    """painted broadleaf: tapering trunk with forks, crown of overlapping clumps back to front"""
    pts = [np.array([x, y])]; a = -np.pi / 2 + lean
    for k in range(7): a += r.normal(0, .10); pts.append(pts[-1] + np.array([math.cos(a), math.sin(a)]) * h * .095)
    pts = np.array(pts); tip = pts[-1]; br = []
    cl = []
    for k in range(clumps):
        q = r.random(); th = r.uniform(0, 6.28); rr = math.sqrt(r.random()); cx = tip[0] + math.cos(th) * rr * h * spread; cy = y - h * (crown[0] + (crown[1] - crown[0]) * (.5 + .5 * math.sin(th) * rr)); cl.append((cx, cy, h * r.uniform(.10, .19)))
    cl.sort(key=lambda c: c[1] * .6 + r.normal(0, h * .05), reverse=False)
    for j, (cx, cy, R) in enumerate(cl):
        if j == len(cl) // 3:
            for (ux, uy, _) in cl[::3]:
                b = pts[int(r.integers(3, 7))]; m = (b + [ux, uy]) / 2 + r.normal(0, h * .02, 2); P = spl(np.array([b, m, [ux, uy]]), n=6); K = P[np.linspace(0, len(P) - 1, 8).astype(int)][None]; cv.strokes(K, h * trunk * .30, C(bark[0]), shape='taper', seed=j)
            K = pts[None]; cv.strokes(K, h * trunk, C(bark[0]), shape='taper', seed=3); cv.strokes(K + [-h * trunk * .25, 0], h * trunk * .45, C(bark[1]), shape='taper', seed=4, dry=.3); cv.strokes(K + [-h * trunk * .45, 0], h * trunk * .16, C(bark[2]), shape='taper', seed=5, dry=.5)
        leaves(cv, cx, cy, R, int(46 + R * .5), tones, r, light, size)


# ------------------------------------------------------------------------------------------------ red spider lily (higanbana)
def lily(cel, x, y, h, open=1., sway=0., seed=0, n=6, lw=1.6, cols=('#6e0f1c', '#a81a2a', '#d8353f', '#f4736c'), ink='#2a0810', stem=('#26332a', '#465a44', '#7c8d70'), rain=0.):
    """one flower on its bare stem, foot at (x, y), h px tall.  open 0 = tight bud, 1 = petals rolled right back, stamens out.  sway in radians"""
    r = np.random.default_rng(seed + 17); S = h / 420.
    base = np.array([x, y]); top = base + np.array([math.sin(sway + .05) * h * .74, -math.cos(sway * .7) * h * .74]); mid = (base + top) / 2 + [h * .035 * math.cos(seed), 0]; P = spl(np.array([base, mid, top]), n=14); d = np.gradient(P, axis=0); d /= np.linalg.norm(d, axis=1, keepdims=True); nn = np.c_[-d[:, 1], d[:, 0]]; w = np.linspace(7.2, 4.4, len(P))[:, None] * S
    cel.fill(np.vstack([P + nn * w, (P - nn * w)[::-1]]), stem[1]); cel.fill(np.vstack([P - nn * w * .1, (P - nn * w)[::-1]]), stem[0]); cel.line(P + nn * w * .5, 1.6 * S, stem[2], taper=(.2, .4)); cel.line(P + nn * w, lw, ink, taper=(.1, .2)); cel.line(P - nn * w, lw, ink, taper=(.1, .2))
    o = float(np.clip(open, 0, 1.08)); R = h * .30; items = []; tl = sway * 1.2
    def proj(v): q = v @ np.array([[1, 0, 0], [0, .93, .36], [0, -.36, .93]]).T; return np.c_[top[0] + (q[:, 0] * math.cos(tl) - q[:, 2] * -math.sin(tl) * 0 + q[:, 2] * math.sin(tl)) * 1., top[1] - (q[:, 2] * math.cos(tl) - q[:, 0] * math.sin(tl)) * .96 + q[:, 1] * .30]
    for k in range(n):
        ph = 2 * np.pi * (k + .12 * math.sin(k * 2.3 + seed)) / n + .3; el = .62 - .40 * o; D = np.array([math.cos(ph) * math.cos(el), math.sin(ph) * math.cos(el), math.sin(el)]); up = np.array([0, 0, 1.]); sd = np.cross(D, up); sd /= np.linalg.norm(sd); u2 = np.cross(sd, D)
        ped = np.array([[0, 0, 0]]) + D * np.linspace(0, R * .30, 6)[:, None]; items.append((D[1] - 1.2, 'ped', proj(ped), None)); c0 = D * R * .30
        for j in range(6):
            a = 2 * np.pi * j / 6 + .4 * k; rad = sd * math.cos(a) + u2 * math.sin(a); t = np.linspace(0, 1, 18); L = R * (.95 + .10 * math.sin(j * 5 + k))
            curl = (.15 + 2.75 * o) * t ** 1.15; fl = .10 + .85 * o                      # roll back over itself
            ax = np.cumsum(np.cos(curl)) * L / 18; ay = np.cumsum(np.sin(curl)) * L / 18 * fl + t * R * .04
            c = c0 + D * ax[:, None] * (.55 + .45 * (1 - o)) + rad * (ay[:, None] * .95 + .02 * R)
            wv = .5 * R * .055 * np.sin(t * 16 + j * 2 + k) * t * o; tang = np.cross(D, rad); c = c + tang * wv[:, None]; wd = R * (.052 + .035 * o) * np.sin(np.pi * np.clip(t * .93 + .07, 0, 1)) ** .7
            a1, a2 = c + tang * wd[:, None], c - tang * wd[:, None]; items.append((float(c[:, 1].mean() / R), 'pet', (proj(a1), proj(a2), proj(c)), rad[1] + .4 * D[1]))
        for j in range(6):                                                            # stamens: long arcs out and up
            a = 2 * np.pi * j / 6 + .9 * k + .3; rad = sd * math.cos(a) + u2 * math.sin(a); t = np.linspace(0, 1, 20); L = R * (1.9 + .2 * math.sin(j * 3 + k)) * smooth(.15, .95, o)
            if L < R * .1: continue
            c = c0 + D * (L * .62 * np.sin(t * 1.35))[:, None] + rad * (L * .16 * t)[:, None] + up * (L * .52 * t ** 1.9 - L * .10 * t)[:, None]; items.append((float(c[:, 1].mean() / R) + .05, 'sta', proj(c), None))
    items.sort(key=lambda v: -v[0] if False else v[0])
    for dep, kind, g, lit in items:
        if kind == 'ped': cel.line(g, 5.5 * S, stem[0], taper=(.02, .1), press=.1); cel.line(g, 3. * S, stem[1], taper=(.1, .1))
        elif kind == 'sta': cel.line(g, 3.0 * S + .6, cols[0] if dep < 0 else cols[1], taper=(.05, .3)); cel.line(g, 1.5 * S + .3, cols[2] if dep < 0 else cols[3], taper=(.1, .4)); e = g[-1]; dd = g[-1] - g[-3]; dd = dd / (np.linalg.norm(dd) + 1e-6); pp = np.array([-dd[1], dd[0]]); cel.line(np.array([e - pp * 5.5 * S, e, e + pp * 5.5 * S]), 4.2 * S, '#3a0a10', taper=(.2, .2)); cel.line(np.array([e - pp * 4 * S, e + pp * 2 * S]), 1.8 * S, '#e8b08a', taper=(.2, .2))
        else:
            p1, p2, c = g; poly = np.vstack([p1, p2[::-1]]); back = dep < -.06; base_ = cols[1] if back else cols[2]; cel.fill(poly, base_); cel.fill(np.vstack([c, p2[::-1]]), cols[0] if back else cols[1]); 
            if not back: cel.line((p1 * .6 + c * .4)[3:-4], 2.2 * S, cols[3], taper=(.3, .5))
            cel.line(c[1:-3], 1.1 * S + .3, cols[0], taper=(.2, .6), alpha=.8); cel.line(p1, lw * .8, ink, taper=(.15, .3)); cel.line(p2, lw * .8, ink, taper=(.15, .3))


# ------------------------------------------------------------------------------------------------ rain
def rain(f, n=150, ang=.03, speed=(70, 150), length=(90, 420), alpha=(.16, .60), wd=(.8, 1.7), col='#dfe2e2', seed=0, h=H, w=W, up=0., freeze=None, near=0):
    """long single strokes of rain, premult RGBA.  f may be fractional; freeze = frame at which every drop stops.  up>0 they rise.  near = extra thick soft strokes"""
    r = np.random.default_rng(seed + 400); f = f if freeze is None else min(f, freeze); a = np.zeros((h, w), f32); dx, dy = math.sin(ang), math.cos(ang); span = h + 900.
    X = r.uniform(-200, w + 200, n); Y = r.uniform(0, span, n); q = r.random(n) ** 1.6; V = speed[0] + (speed[1] - speed[0]) * q; L = length[0] + (length[1] - length[0]) * q * r.uniform(.5, 1, n); A = alpha[0] + (alpha[1] - alpha[0]) * q; Wd = wd[0] + (wd[1] - wd[0]) * q
    cyc = np.floor((Y + V * f) / span); jx = (np.sin(cyc * 12.9898 + X * .37) * 43758.5453) % 1.; x = (X + jx * w) % (w + 400) - 200; y = (Y + V * f * (1 - 2 * (up > 0))) % span - 450
    c = Cel(w, h, 0, 0, 2); cc = C(col)
    for k in range(n):                                                                # one confident stroke each: nothing at the top, its full weight at the foot
        x0, y0 = x[k] + dx * y[k], y[k]; x1, y1 = x0 + dx * L[k], y0 + dy * L[k]; q_ = Wd[k] * .62; c.fill(np.array([(x0, y0), (x0 + (x1 - x0) * .55 + q_ * .55, y0 + (y1 - y0) * .55), (x1 + q_, y1 - 5), (x1 + q_ * .2, y1), (x1 - q_, y1 - 5), (x0 + (x1 - x0) * .55 - q_ * .55, y0 + (y1 - y0) * .55)]), cc, float(A[k]))
    a = c.rgba()[..., 3]
    if near:
        b = np.zeros((h, w), f32); r2 = np.random.default_rng(seed + 900); X = r2.uniform(0, w, near); Y = r2.uniform(0, span, near); V = speed[1] * 1.7
        for k in range(near): y_ = (Y[k] + V * f) % span - 500; cv2.line(b, (int((X[k] + dx * y_) * 16), int(y_ * 16)), (int((X[k] + dx * (y_ + 520)) * 16), int((y_ + 520 * dy) * 16)), .30, 4, cv2.LINE_AA, 4)
        a = np.maximum(a, cv2.GaussianBlur(b, (0, 0), 1.6))
    return np.dstack([C(col) * a[..., None], a]).astype(f32)


def rain_rings(f, box, n=40, life=16, rmax=.30, seed=0, w0=.018):
    """ripples of drops on water for land.puddle(rings=...): rows (x, y, R, width, amp) at frame f.  box = (x0, x1, y0, y1) metres"""
    r = np.random.default_rng(seed + 60); X = r.random((n, 64)); Y = r.random((n, 64)); ph = r.uniform(0, life, n); sz = r.uniform(.45, 1., n); out = []
    for k in range(n):
        c, t = divmod(f + ph[k], life); c = int(c) % 64; u = t / life; x = box[0] + (box[1] - box[0]) * X[k, c]; y = box[2] + (box[3] - box[2]) * Y[k, c]; R = rmax * sz[k] * u ** .7; out.append((x, y, R, w0 * (.6 + u), .95 * (1 - u) ** 1.2))
        if u > .25: out.append((x, y, R * .55, w0 * .7, .6 * (1 - u) ** 1.4))
    return out


def splashes(f, cam, z, pts, seed=0, col='#eef0f0', h=.16, life=4):
    """little crowns where drops hit the ground: pts (n,3) world.  premult RGBA"""
    cel = Cel(); r = np.random.default_rng(seed + 8); n = len(pts); ph = r.integers(0, 11, n); s, zc = cam.project(pts)
    for k in range(n):
        t = (int(f) + ph[k]) % 11
        if t >= life or zc[k] < .3: continue
        x, y = s[k]; 
        if not (0 <= x < cam.w and 0 <= y < cam.h) or zc[k] > z[int(y), int(x)] * 1.06 + .05: continue
        q = cam.f / zc[k] * h * (.35 + .65 * t / life)
        for j in range(4): a = -np.pi / 2 + (j - 1.5) * .55 + r.normal(0, .1); cel.line(np.array([[x + math.cos(a) * q * .25, y + math.sin(a) * q * .25], [x + math.cos(a) * q * .8, y + math.sin(a) * q * .8], [x + math.cos(a) * q * 1.1, y + math.sin(a) * q * .95]]), 1.6, col, taper=(.2, .5), alpha=.75 * (1 - t / life * .6))
    return cel.layer()
