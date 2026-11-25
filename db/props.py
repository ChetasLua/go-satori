"""Props: the steamed bun, the red spider lily, the super-8 memory look.

    bun(cel, x, y, r, light='lamp', seed=1)                     whole bun, pleated crown                      (x, y) = centre of the base, r = half width px
    half(cel, x, y, r, side=1, light='dawn', rot=0., seed=1)    one torn half: soft crumb face to the viewer, side +1 right half / -1 left half
    pull(cel, x, y, r, u, light)                                u 0..1: the bun being pulled apart (threads of dough stretch, then break at .6); returns the two centres
    steam(base, x, y, w, h, t, light, amt, seed)                drawn steam on an RGB frame (cold-sun steam is backlit, lamp steam is warm)
    lily(cel, x, y, s, u, t=0., wind=0., light='rain', seed=2)  red spider lily: u 0 bud .. 1 fully open; (x, y) = foot of the stem, s = stem height px;  returns crown (x, y)
    lily_drops(cel, crown, s, t, seed)                          rain beads on the stamens
    super8(img, f, amt=1., seed=0)                              memory look on an RGB frame: soft focus, warm fade, halation, gate weave, worn rounded gate, dust, scratches, flicker"""
import numpy as np, cv2, math
from . import W, H as HH
from .paint import rgb, mix, f32, fbm, lum, smooth, grid
from .ink import spl, resample, Cel, noise1
from .kit import outline
from .comp import blur, screen

BUN = dict(
    lamp=dict(top='#ffe9bd', mid='#f2c07c', low='#b9713c', line='#4a2413', hot='#fffbe4', crumb='#fff3d6', crumb_s='#e8b878', fill='#7a3f22', steam='#ffdca0'),
    dawn=dict(top='#fffaf0', mid='#f3d9cf', low='#b998ae', line='#3a2432', hot='#ffffff', crumb='#fffdf6', crumb_s='#e9cdc4', fill='#8a4b34', steam='#ffffff'),
    night=dict(top='#cfd6ee', mid='#8f97c4', low='#4c4f86', line='#14142c', hot='#f4f7ff', crumb='#dfe4f6', crumb_s='#9aa0cc', fill='#4a3550', steam='#b9c4f0'))


def _dome(x, y, r, seed, sq=.96, n=64):
    rg = np.random.default_rng(seed); a = np.linspace(math.pi, 2 * math.pi, n); w = 1 + .012 * noise1(n, 12, rg); top = np.c_[x + np.cos(a) * r * (1 + .03 * np.sin(a * 2)) * w, y + np.sin(a) * r * sq * 1.08 * w - r * .10]
    b = np.linspace(0, math.pi, 26); bot = np.c_[x + np.cos(b) * r * 1.0, y - r * .10 + np.sin(b) ** .7 * r * .20]; return np.vstack([top, bot[1:-1]])


def bun(cel, x, y, r, light='lamp', seed=1, lw=None):
    c = BUN[light]; lw = lw or max(.6, r / 110.); P = _dome(x, y, r, seed); cel.fill(P, c['top']); cel.ctx.save(); cel._path(P); cel.ctx.clip()
    cel.fill(spl([(x - r * 1.1, y - r * .10), (x - r * .55, y - r * .04), (x, y - r * .14), (x + r * .44, y - r * .44), (x + r * .62, y - r * .92), (x + r * 1.2, y - r * 1.0), (x + r * 1.2, y + r * .4), (x - r * 1.1, y + r * .4)], n=8, corners=(5, 6, 7)), c['mid'])
    cel.fill(spl([(x - r * 1.1, y + r * .02), (x, y + r * .02), (x + r * .6, y - r * .12), (x + r * .86, y - r * .40), (x + r * 1.2, y - r * .45), (x + r * 1.2, y + r * .4), (x - r * 1.1, y + r * .4)], n=8, corners=(4, 5, 6)), c['low'])
    k = np.array([x - r * .03, y - r * .90]); rg = np.random.default_rng(seed + 7)                  # pleats gathered into a twist
    for j in range(13):
        a = math.pi * (-.02 + 1.04 * j / 12) + rg.normal(0, .025); e = k + np.array([math.cos(a) * r * .62, max(.16, math.sin(a)) * r * .40 + r * .03]) * rg.uniform(.8, 1.05); bend = np.array([-math.sin(a), math.cos(a)]) * r * .07
        S = spl([k + (e - k) * .10 + bend * .25, k + (e - k) * .45 + bend, e], n=10); u = np.linspace(0, 1, len(S)); nrm = np.array([-math.sin(a), math.cos(a)]); w = (r * .055 * np.sin(u ** .5 * math.pi))[:, None]
        cel.fill(np.vstack([S, (S + nrm * w)[::-1]]), c['mid'] if math.cos(a) < .3 else c['low']); cel.line(S, 2.3 * lw, c['line'], (.1, .75), seed=j)
    q = k + np.array([0, r * .02]); cel.fill(spl([(q[0] - r * .13, q[1] + r * .04), (q[0] - r * .05, q[1] - r * .06), (q[0] + r * .06, q[1] - r * .07), (q[0] + r * .13, q[1] + r * .02), (q[0] + r * .02, q[1] + r * .08)], True, n=6), c['top']); cel.line(spl([(q[0] - r * .10, q[1] + r * .03), (q[0], q[1] - r * .02), (q[0] + r * .05, q[1] + r * .04), (q[0] - r * .02, q[1] + r * .05)], n=8), 2.2 * lw, c['line'], seed=3)
    h = np.array([x - r * .56, y - r * .60]); cel.fill(spl([(h[0] - r * .13, h[1] + r * .16), (h[0] - r * .06, h[1] - r * .08), (h[0] + r * .14, h[1] - r * .18), (h[0] + r * .04, h[1] + r * .0)], True, n=6, corners=(0, 2)), c['hot']); cel.ctx.restore()
    outline(cel, P, 4.6 * lw, c['line'], seed, tremor=.3); return np.array([x, y - r * .4])


def half(cel, x, y, r, side=1, light='dawn', rot=0., seed=1, lw=None):
    c = BUN[light]; lw = lw or max(.6, r / 110.); rg = np.random.default_rng(seed * 3 + side); P = _dome(0, 0, r, seed); P = P[P[:, 0] * side >= -1e-6]; P = P[np.argsort(np.arctan2(P[:, 1] + r * .4, P[:, 0] * side))]
    tear = [(rg.normal(0, r * .035) + r * (.05 if k % 2 else -.06), yy) for k, yy in enumerate(np.linspace(-r * 1.12, r * .05, 9))]; co, si = math.cos(rot), math.sin(rot); X = lambda Q: np.asarray(Q, float) @ np.array([[co, si], [-si, co]]) + (x, y)
    T = np.array(tear) * (side, 1); shell = np.vstack([P, T[::-1]]); cel.fill(X(shell), c['top']); cel.ctx.save(); cel._path(X(shell)); cel.ctx.clip()
    cel.fill(X(np.array([(side * r * .30, r * .2), (side * r * .55, -r * .18), (side * r * .80, -r * .5), (side * r * 1.2, -r * .6), (side * r * 1.2, r * .3)])), c['mid']); cel.fill(X(np.array([(0, r * .02), (side * r * .7, -r * .04), (side * r * 1.2, -r * .3), (side * r * 1.2, r * .3), (0, r * .3)])), c['low']); cel.ctx.restore(); outline(cel, X(shell), 4.2 * lw, c['line'], seed, tremor=.3)
    f = np.linspace(0, 2 * math.pi, 30); face = np.c_[np.sin(f) * r * .30 * side - side * r * .05, -r * .52 - np.cos(f) * r * .62]; face[:, 0] += rg.normal(0, r * .018, 30); face[:, 1] += rg.normal(0, r * .012, 30); F = X(spl(face, True, n=4))          # the torn face, seen a little from the front
    cel.fill(F, c['crumb']); cel.ctx.save(); cel._path(F); cel.ctx.clip(); cel.fill(X(np.c_[np.sin(f) * r * .34 * side + side * r * .06, -r * .40 - np.cos(f) * r * .52][8:26]), c['crumb_s'])
    for _ in range(38): a, d = rg.uniform(0, 6.28), rg.random() ** .5; p = X(np.array([[math.sin(a) * r * .28 * d * side - side * r * .05, -r * .52 - math.cos(a) * r * .58 * d]]))[0]; e = np.linspace(0, 6.28, 9); R = r * rg.uniform(.010, .030); cel.fill(np.c_[p[0] + np.cos(e) * R, p[1] + np.sin(e) * R * .7], c['crumb_s'] if rg.random() < .6 else c['hot'])
    w_ = 1 + .16 * np.sin(f * 3 + seed) + .10 * np.sin(f * 5 + side); g = X(spl(np.c_[np.sin(f) * r * .17 * side * w_ - side * r * .03, -r * .34 - np.cos(f) * r * .27 * w_], True, n=3)); cel.fill(g, c['fill']); cel.line(np.vstack([g, g[:1]]), 1.8 * lw, c['line'], seed=5)
    for _ in range(6): p = g[int(rg.integers(len(g)))] * .55 + g.mean(0) * .45; e = np.linspace(0, 6.28, 8); cel.fill(np.c_[p[0] + np.cos(e) * r * .03, p[1] + np.sin(e) * r * .02], mix(rgb(c['fill']), rgb(c['hot']), .35))
    cel.ctx.restore(); cel.line(np.vstack([F, F[:1]]), 2.4 * lw, c['line'], (.1, .1), seed=6, press=.5); return X(np.array([[0, -r * .4]]))[0]


def pull(cel, x, y, r, u, light='dawn', seed=1):
    if u <= .02: bun(cel, x, y, r, light, seed); return np.array([x, y]), np.array([x, y])
    gap = r * (.30 + 1.10 * u); a, b = np.array([x - gap / 2, y]), np.array([x + gap / 2, y]); c = BUN[light]; lw = max(.6, r / 110.)
    if u < .6:
        for k in range(7):
            yy = y - r * (.12 + .10 * k); th = r * .10 * (1 - u / .75) + 1.5; m = np.array([x, yy + r * .06 * math.sin(k * 2.3)]); q = np.array([(a[0], yy - th), (m[0], m[1] - th * .35), (b[0], yy - th), (b[0], yy + th), (m[0], m[1] + th * .35), (a[0], yy + th)]); cel.fill(spl(q, True, (0, 2, 3, 5), 6), c['crumb']); cel.line(spl(q[:3], n=6), 1.6 * lw, c['line'], (.2, .2), seed=k); cel.line(spl(q[3:], n=6), 1.6 * lw, c['line'], (.2, .2), seed=k + 9)
    half(cel, *a, r, -1, light, -.22 * u, seed); half(cel, *b, r, 1, light, .22 * u, seed); return a, b


def steam(base, x, y, w, h, t, light='dawn', amt=.7, seed=0, n=5):
    """curling drawn steam: ribbons rising on a slow wave, hard inside edge, soft outside; pure function of t (s)"""
    c = rgb(BUN[light]['steam']); m = np.zeros(base.shape[:2], f32); rg = np.random.default_rng(seed); S = Cel(W, HH, ss=1)
    for k in range(n):
        x0 = x + (k - (n - 1) / 2) * w / n + rg.normal(0, w * .04); ph = rg.uniform(0, 6.28); u = np.linspace(0, 1, 40); L = h * rg.uniform(.65, 1.1); life = ((t * .42 + rg.random()) % 1.0); a0 = max(0., life - .55); a1 = min(1., life + .30)
        sx = x0 + w * (.05 + .30 * u) * np.sin(u * 5.0 - t * 2.0 + ph) + w * .10 * u * math.sin(t * .9 + k); sy = y - u * L; i = (u >= a0) & (u <= a1)
        if i.sum() < 6: continue
        P = np.c_[sx, sy][i]; T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]; v = np.linspace(0, 1, len(P)); hw = (w * .085 * np.sin(v * math.pi) ** .8 * (.5 + u[i]))[:, None]; S.fill(np.vstack([P + N * hw, (P - N * hw * .4)[::-1]]), '#ffffff', .55 * (1 - .5 * life))
    m = S.rgba()[..., 3]; fade = np.clip((grid()[1] - (y - h * 1.05)) / (h * .55), 0, 1) * np.clip((y + 6 - grid()[1]) / (h * .10), 0, 1)
    k = np.clip(m * .75 + blur(m, w * .05 + 1) * .7 + blur(m, w * .16 + 2) * .5, 0, 1) * fade * amt; return base * (1 - k[..., None]) + c * k[..., None] * 1.04


# ------------------------------------------------------------------ red spider lily
LILY = dict(rain=dict(red='#c8323a', dark='#7d1a24', lit='#ea6a64', stem='#5d6b52', stem_s='#39452f', line='#160a0c', tip='#e8c9a0'),
            dawn=dict(red='#e0453c', dark='#97282e', lit='#ff9a82', stem='#7d9460', stem_s='#4d6140', line='#241018', tip='#ffe6b8'))


def lily(cel, x, y, s, u, t=0., wind=0., light='rain', seed=2, n=6, lw=None):
    """u 0 = closed bud cluster, 1 = open: narrow petals reflex back one after another, stamens sweep out in long arcs"""
    c = LILY[light]; lw = lw or max(.6, s / 330.); rg = np.random.default_rng(seed); sw = wind * (.10 + .03 * math.sin(t * 3.1)) * s
    stem = spl([(x, y), (x + sw * .15 - s * .02, y - s * .4), (x + sw * .55, y - s * .75), (x + sw, y - s)], n=12); v = np.linspace(0, 1, len(stem)); w = (s * .015 * (1.15 - .35 * v))[:, None]; nx = np.array([1., 0])
    cel.fill(np.vstack([stem - nx * w, (stem + nx * w)[::-1]]), c['stem']); cel.fill(np.vstack([stem + nx * w * .1, (stem + nx * w)[::-1]]), c['stem_s']); cel.line(stem - nx * w, 1.9 * lw, c['line'], (.05, .1), seed=1); cel.line(stem + nx * w, 1.9 * lw, c['line'], (.05, .1), seed=2)
    k = stem[-1]; R = s * .40; items = []
    for j in range(n):
        az = 2 * math.pi * j / n + .35 + rg.normal(0, .08); o = float(np.clip(u * 1.30 - .06 * ((j * 5) % n), 0, 1)); d = np.array([math.cos(az), math.sin(az) * .34]); depth = math.sin(az); foot = k + d * s * .035; tilt = wind * .18
        for q in range(5):                                                           # tepals: recurved ribbons with wavy edges
            a2 = az + (q - 2) * .34 * (.25 + .75 * o); e = np.array([math.cos(a2), math.sin(a2) * .36]); up = np.array([tilt, -1.]); L = R * rg.uniform(.82, 1.1); tt = np.linspace(0, 1, 18); bend = o * (2.0 + .5 * rg.random())
            th = tt * bend; rad = np.cumsum(np.sin(th + .15 + .25 * o)) / 18; hgt = np.cumsum(np.cos(th + .15)) / 18; P = foot + e * (rad * L * (.35 + .65 * o))[:, None] + up * (hgt * L * (1 - .10 * o))[:, None]; T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
            ww = L * (.030 + .030 * o) * np.sin(np.clip(tt * .93 + .07, 0, 1) * math.pi) ** .6 * (1 + .25 * o * np.sin(tt * 22 + q + j)); ww[-1] = 0; items.append((depth + .1 * q, 'p', P, N * ww[:, None], q))
        for q in range(4):                                                           # stamens
            if o < .12: continue
            a2 = az + (q - 1.5) * .20; e = np.array([math.cos(a2), math.sin(a2) * .42]); L = R * 1.45 * (.35 + .65 * o) * rg.uniform(.62, 1.22); tt = np.linspace(0, 1, 20); fi = tt * (.85 + .55 * o) * rg.uniform(.8, 1.2); P = foot + e * (L * (.78 + .34 * o) * np.sin(fi) * (.35 + .65 * o))[:, None] + np.array([tilt, -1.]) * (L * .62 * ((1 - np.cos(fi)) * (.5 + .5 * o) + (1 - o) * .6 * tt + .22 * np.sin(tt * math.pi * .5)))[:, None]; items.append((depth + .6, 's', P + np.c_[wind * 10 * tt ** 2 * math.sin(t * 4 + q + j), 0 * tt], None, q))
    for _, kind, P, Nw, q in sorted(items, key=lambda z: z[0]):
        if kind == 'p':
            poly = np.vstack([P + Nw, (P - Nw)[::-1]]); cel.fill(poly, c['red'] if _ > -.3 else c['dark']); cel.fill(np.vstack([P, (P - Nw)[::-1]]), c['dark'] if _ > -.3 else mix(rgb(c['dark']), rgb('#000000'), .3)); m = slice(3, 11); cel.line((P + Nw * .55)[m], 1.7 * lw, c['lit'], (.4, .5), seed=q) if _ > 0 else None
            cel.line(P + Nw, 1.7 * lw, c['line'], (.15, .25), seed=q); cel.line(P - Nw, 1.7 * lw, c['line'], (.15, .25), seed=q + 3); cel.line(P[2:15], 1.0 * lw, c['line'], (.2, .6), seed=q + 6)
        else:
            cel.line(P, 2.5 * lw, c['line'], (.05, .2), seed=q); cel.line(P, 1.5 * lw, c['lit'] if _ > .4 else c['red'], (.05, .2), seed=q); e = P[-1]; a = np.linspace(0, 6.28, 14); cel.fill(np.c_[e[0] + np.cos(a) * 3.6 * lw, e[1] + np.sin(a) * 2.2 * lw], c['tip']); cel.line(np.c_[e[0] + np.cos(a) * 3.6 * lw, e[1] + np.sin(a) * 2.2 * lw], 1.1 * lw, c['line'], seed=2)
    return k


def lily_drops(cel, crown, s, t, seed=5, n=9):
    rg = np.random.default_rng(seed)
    for k in range(n):
        a = rg.uniform(0, 6.28); d = s * rg.uniform(.12, .40); p = crown + np.array([math.cos(a) * d, math.sin(a) * d * .5 - s * .08]); per = rg.uniform(1.2, 2.4); ph = ((t + rg.random() * per) % per) / per; r = s * .012 * (.4 + .6 * min(1, ph / .7)); y = p[1] + (0 if ph < .7 else 1400 * ((ph - .7) * per) ** 2); q = np.linspace(0, 6.28, 16); cel.fill(np.c_[p[0] + np.cos(q) * r * .8, y + np.sin(q) * r * (1.6 if ph > .7 else 1.1)], '#d9dfe2', .75); cel.fill(np.c_[p[0] - r * .2 + np.cos(q) * r * .22, y - r * .3 + np.sin(q) * r * .22], '#ffffff')


# ------------------------------------------------------------------ super-8 memory
_gate = {}


def _gate_mask(seed):
    if seed not in _gate:
        x, y = grid(); dx = np.maximum(np.abs(x - W / 2) - (W / 2 - 150), 0); dy = np.maximum(np.abs(y - HH / 2) - (HH / 2 - 128), 0); n = fbm(HH, W, (60, 20, 8), seed=seed); n = (n - n.mean()) / (n.std() + 1e-6); d0 = np.sqrt(dx ** 2 + dy ** 2); d = d0 + n * 3.5 * np.clip(d0 / 60, 0, 1); _gate[seed] = (1 - smooth(92, 112, d)).astype(f32), (1 - smooth(10, 150, d0)).astype(f32)
    return _gate[seed]


def super8(img, f, amt=1., seed=0, warm='#ffd9a6', weave=1., dust=1.):
    """18-fps feel comes from the shot (hold drawings on twos); this is the film stock.  Pure function of f."""
    rg = np.random.default_rng(seed * 7919 + f); src = img; g = f // 1
    wx = 2.2 * math.sin(f * .41) + rg.normal(0, .7); wy = 1.5 * math.sin(f * .63 + 1) + rg.normal(0, .9) + (7. if rg.random() < .03 else 0.); zx = 1.004 + .0015 * math.sin(f * .2)
    o = cv2.warpAffine(img, np.float32([[zx, 0, wx * weave - W * (zx - 1) / 2], [0, zx, wy * weave - HH * (zx - 1) / 2]]), (W, HH), borderMode=cv2.BORDER_REFLECT_101)
    s = cv2.GaussianBlur(o, (0, 0), 2.0); b = blur(np.clip(o, 0, 2), 26); o = s * .80 + b * .22; hal = blur(np.clip(lum(s) - .62, 0, 1), 34)[..., None]; o = o + hal * rgb('#ff9a5c') * .55
    L = lum(o)[..., None]; o = L + (o - L) * .74; o = o * rgb(warm) * 1.0 + (1 - L) ** 2 * rgb('#5a3a2c') * .16; o = o * .92 + .035; o = o * (1 + .035 * math.sin(f * 1.9) + rg.normal(0, .018))
    n = rg.normal(0, 1, (HH // 2, W // 2)).astype(f32); n = cv2.resize(cv2.GaussianBlur(n, (0, 0), .8), (W, HH), interpolation=cv2.INTER_LINEAR); o = o + (n * .055 * (1 - .5 * np.clip(L[..., 0], 0, 1)))[..., None]
    c = Cel(W, HH, ss=1)
    for _ in range(int(rg.poisson(3.2 * dust))):
        x, y = rg.uniform(0, W), rg.uniform(0, HH); a = np.linspace(0, 6.28, 9, endpoint=False); R = rg.uniform(1.5, 6); c.fill(np.c_[x + np.cos(a) * R * rg.uniform(.5, 1.4, 9), y + np.sin(a) * R * rg.uniform(.5, 1.4, 9)], '#fff4dd' if rg.random() < .6 else '#1c120d', .8)
    if rg.random() < .30 * dust: x, y = rg.uniform(200, W - 200), rg.uniform(150, HH - 150); c.line(spl([(x, y), (x + rg.normal(0, 40), y + rg.normal(0, 40)), (x + rg.normal(0, 60), y + rg.normal(0, 60)), (x + rg.normal(0, 70), y + rg.normal(0, 70))], n=8), 2.0, '#1c120d', alpha=.7, seed=f)
    q = np.random.default_rng(seed + f // 9)
    if q.random() < .45 * dust: sx = q.uniform(300, W - 300) + 3 * math.sin(f * .7); c.line(np.array([(sx, -10), (sx + 3, HH * .5), (sx - 2, HH + 10)]), 1.5, '#fff1d8', (.0, .0), alpha=.40, seed=f)
    lay = c.rgba(); o = o * (1 - lay[..., 3:4]) + lay[..., :3]; m, soft_ = _gate_mask(seed % 3); o = o * (.62 + .38 * soft_[..., None]) * m[..., None] + rgb('#120b08') * (1 - m[..., None]) + (soft_ * (1 - m))[..., None] * rgb('#ff8a3c') * .18
    if rg.random() < .06: o = o + np.clip(1 - grid()[0] / 420, 0, 1)[..., None] ** 2 * rgb('#ff7a2c') * .5
    return src * (1 - amt) + o * amt
