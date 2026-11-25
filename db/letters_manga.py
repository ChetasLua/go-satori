"""Manga page kit (d4): paper, screentone, brush flash, speed lines, ruled panel, 3-D EGO lettering, the violet halftone spark.

    img = sheet(seed)                                           worn newsprint, pure warm-free greys, RGB
    cov = dots(dark, pitch=8.5, ang=45)                         screentone coverage from a darkness field (0..1 array or number), crisp round dots
    cov = lines_tone(dark, pitch, ang) | sand(dark, seed)       ruled tone | sand tone
    f   = lin(p0, p1, a, b) | rad(cx, cy, r0, r1, a, b, ry=.7)  darkness fields
    cov = scrape(cov, [polyline...], w)                         tone cut away with a knife (white flecks)
    ink = flash(cx, cy, r0, ry=.72, n=.., seed)                 beta flash: brush wedges closing on an inner ellipse, ink alpha 0..1
    ink = rush(ang, box, seed)                                  bundled brush speed lines
    L   = ego3d((x, y, h), vp=(960, 640), depth=.07)            dict(face, side, hatch, tone): EGO as a slab receding to a vanishing point
    img = ink_on(img, ink);  img = panel(img, (x0, y0, x1, y1), lw=7)
    img = spark(img, cx, cy, R, f, seed)                        the only colour: violet halftone burst, hot core, forked bolts; R grows over the shot
    img = print_(img, seed)                                     dot gain, dry ink, dust, yellowless paper tooth (call last, hold the seed)
    page_bg(f)                                                  everything behind the hand, ready made (cached plates); see ego_page() for the order
"""
import numpy as np, cv2, math
from . import W, H
from .paint import rgb, f32, fbm, smooth, grid, mix
from .ink import Cel, spl, resample, noise1
from .sumi import tooth
from .letters import ego_mask, ego_paths, ego_width

INK, PAPER, VIOLET = '#121113', '#f2f1ee', '#8a2cff'


# ------------------------------------------------------------ paper, print
_c = {}


def sheet(seed=3, h=H, w=W, tone=PAPER):
    """cheap magazine paper, aged: fibres, show-through, handled corners.  no colour cast beyond a breath"""
    k = ('sheet', seed, h, w, tone)
    if k not in _c:
        x, y = grid(h, w); b = np.ones((h, w, 3), f32) * rgb(tone); n = fbm(h, w, (420, 150, 60), (1, .6, .35), seed); t = fbm(h, w, (3.0, 1.6), (1, .8), seed + 1); r = np.random.default_rng(seed)
        fib = np.zeros((h, w), f32)
        for _ in range(2600):
            a = r.uniform(0, math.pi); L = r.uniform(6, 34); px, py = r.uniform(0, w), r.uniform(0, h); cv2.line(fib, (int(px * 8), int(py * 8)), (int((px + math.cos(a) * L) * 8), int((py + math.sin(a) * L) * 8)), float(r.choice([-1, 1]) * r.uniform(.2, 1)), 1, cv2.LINE_AA, 3)
        ed = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y)); hand = smooth(260, 0, ed + 46 * n) * .085
        b = b * (1 + .011 * n + .010 * t + .030 * cv2.GaussianBlur(fib, (0, 0), .7) - hand)[..., None]
        cr = np.zeros((h, w), f32)                                              # creases
        for _ in range(5):
            p = np.array([r.uniform(0, w), r.uniform(0, h)]); a = r.uniform(0, math.pi); P = [p]
            for _ in range(14): a += r.normal(0, .12); p = p + np.array([math.cos(a), math.sin(a)]) * r.uniform(20, 60); P.append(p)
            cv2.polylines(cr, [np.round(np.array(P) * 8).astype(np.int32)], False, float(r.uniform(.4, 1)), 1, cv2.LINE_AA, 3)
        b = b * (1 - .10 * cv2.GaussianBlur(cr, (0, 0), .8) + .05 * np.roll(cv2.GaussianBlur(cr, (0, 0), 1.2), (2, 2), (0, 1)))[..., None]; _c[k] = np.clip(b, 0, 1)
    return _c[k].copy()


def ink_on(img, a, col=INK, op=1.0):
    a = np.clip(a, 0, 1)[..., None] * op; return img * (1 - a) + rgb(col) * a


def print_(img, seed=5, gain=.35):
    """offset print on a tired press: dots swell a hair, blacks are not quite full, white dust, a few hickeys"""
    h, w = img.shape[:2]; k = ('print', seed, h, w)
    if k not in _c:
        r = np.random.default_rng(seed); t = np.tile(tooth(2304, 2304, 5), (1, 1))[:h, :w]; big = fbm(h, w, (300, 110, 40), (1, .6, .4), seed); d = Cel(w, h)
        for _ in range(110): x, y = r.uniform(0, w), r.uniform(0, h); R = .5 + 1.8 * r.random() ** 3; a = np.linspace(0, 6.28, 9)[:-1]; d.fill(np.c_[x + np.cos(a) * R * r.uniform(.6, 1.3, 8), y + np.sin(a) * R * r.uniform(.6, 1.3, 8)], '#ffffff' if r.random() < .7 else '#000000')
        _c[k] = (np.clip(.055 * smooth(.62, 1.0, t) + .045 * smooth(.3, 1.6, big), 0, .2).astype(f32), d.rgba())
    dry, dust = _c[k]; g = cv2.GaussianBlur(img, (0, 0), .75); img = img + (np.minimum(img, g) - img) * gain; L = img.mean(-1, keepdims=True); img = img + (.95 - img) * dry[..., None] * (1 - smooth(.12, .5, L)); a = dust[..., 3:4] * .75; return img * (1 - a) + dust[..., :3] * .75 * np.where(dust[..., :3] > .5, .95, .1) * (dust[..., 3:4] > 0)


# ------------------------------------------------------------ tone
def lin(p0, p1, a=0., b=.6, h=H, w=W):
    x, y = grid(h, w); d = np.array(p1, float) - p0; t = np.clip(((x - p0[0]) * d[0] + (y - p0[1]) * d[1]) / (d @ d), 0, 1); return a + (b - a) * t


def rad(cx, cy, r0, r1, a=0., b=.6, ry=.7, h=H, w=W):
    x, y = grid(h, w); return a + (b - a) * smooth(r0, r1, np.hypot(x - cx, (y - cy) / ry))


def dots(dark, pitch=8.5, ang=45., h=H, w=W):
    """screentone: black dots that grow with darkness; above .5 the white between them shrinks to dots.  coverage 0..1"""
    x, y = grid(h, w); c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang)); u = (x * c + y * s) / pitch; v = (-x * s + y * c) / pitch; du, dv = u - np.round(u), v - np.round(v); d = np.sqrt(du * du + dv * dv)
    eu, ev = u - np.floor(u) - .5, v - np.floor(v) - .5; e = np.sqrt(eu * eu + ev * ev); D = np.clip(np.broadcast_to(np.asarray(dark, f32), (h, w)), 0, 1)
    lo = np.clip((np.sqrt(np.minimum(D, .5) / math.pi) - d) * pitch + .5, 0, 1); hi = 1 - np.clip((np.sqrt(np.clip(1 - D, 0, .5) / math.pi) - e) * pitch + .5, 0, 1); lo = lo * (D > .012); return np.where(D <= .5, lo, np.where(D > .985, 1., hi)).astype(f32)


def lines_tone(dark, pitch=7., ang=0., h=H, w=W):
    x, y = grid(h, w); c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang)); v = (-x * s + y * c) / pitch; d = np.abs(v - np.round(v)); return np.clip((np.clip(dark, 0, 1) * .5 - d) * pitch + .5, 0, 1).astype(f32)


def sand(dark, seed=3, size=1.6, h=H, w=W):
    k = ('sand', seed, size, h, w)
    if k not in _c: n = cv2.GaussianBlur(np.random.default_rng(seed).standard_normal((h, w)).astype(f32), (0, 0), size * .5); r = np.argsort(np.argsort(n.ravel()[::5])); _c[k] = n / n.std()
    from scipy.special import ndtri
    return smooth(-.25, .25, ndtri(np.clip(dark, .001, .999)).astype(f32) - _c[k])


def scrape(cov, polys, w=3., seed=1):
    """knife through the tone: tapered white cuts"""
    h, wd = cov.shape; c = Cel(wd, h); r = np.random.default_rng(seed)
    for P in polys: c.line(np.asarray(P, float), w * r.uniform(.6, 1.4), '#ffffff', taper=(.05, .8), press=.4, seed=int(r.integers(1 << 30)))
    return cov * (1 - c.rgba()[..., 3])


# ------------------------------------------------------------ lines
def flash(cx, cy, r0, ry=.72, n=420, seed=0, r1=1500., w=9., clump=1.0, h=H, w_=W, a0=0., a1=2 * math.pi, core=1.0):
    """brush concentration lines in bundles: heavy wedge, two companions, hair; every line dry at its tip.  ink alpha"""
    r = np.random.default_rng(seed); c = Cel(w_, h); nb = int(n / 7); th0 = np.sort(a0 + (a1 - a0) * (np.arange(nb) + r.uniform(-.45, .45, nb)) / nb); wt = r.gamma(1.1, 1., nb)
    for t, g in zip(th0, wt):
        for j in range(int(3 + r.integers(0, 6))):
            q = r.normal(0, .016 * clump) * (1 + j * .25); a = t + q; big = j == 0; ww = w * (g * .75 + .15 if big else r.random() ** 2 * .55 + .05); depth = r0 * (1 + abs(r.normal(0, .16)) + (0 if big else r.gamma(1.2, .22))); cc, ss = math.cos(a), math.sin(a)
            far = np.array([cx + cc * r1, cy + ss * r1]); tip = np.array([cx + cc * depth, cy + ss * depth * ry]); d = tip - far; L = np.hypot(*d); d /= L; nn = np.array([-d[1], d[0]]); m = 22; u = np.linspace(0, 1, m); hw = ww * (1 - u) ** 1.0 * (1 + .10 * noise1(m, 4, r)) + .35 * (1 - u); C = far + d * (u * L)[:, None] + nn * (noise1(m, 8, r) * .5)[:, None]
            c.fill(np.vstack([C + nn * hw[:, None], (C - nn * hw[:, None])[::-1]]), INK)
            if big and ww > 5:                                                 # flying white inside the heavy ones
                for _ in range(2): k0 = r.uniform(.35, .75); k1 = min(.97, k0 + r.uniform(.12, .3)); uu = np.linspace(k0, k1, 8); CC = far + d * (uu * L)[:, None] + nn * (r.uniform(-.5, .5) * ww * (1 - k0)); c.line(CC, max(.7, ww * .10), '#ffffff', taper=(.3, .6), seed=int(r.integers(1 << 30)))
    q = c.rgba(); return np.clip(q[..., 3] - q[..., 0] / np.maximum(q[..., 3], 1e-3) * (q[..., 3] > .01), 0, 1)


def rush(ang, box=(0, 0, W, H), n=130, seed=0, w=8., L=(300, 1500), h=H, w_=W):
    """bundled brush speed lines along ang (radians).  ink alpha"""
    r = np.random.default_rng(seed); c = Cel(w_, h); d = np.array([math.cos(ang), math.sin(ang)]); nn = np.array([-d[1], d[0]]); x0, y0, x1, y1 = box
    for _ in range(n // 6):
        p0 = np.array([r.uniform(x0, x1), r.uniform(y0, y1)]); g = r.gamma(1.2, .8)
        for j in range(int(r.integers(3, 9))):
            p = p0 + nn * r.normal(0, 13) + d * r.normal(0, 120); ln = r.uniform(*L) * (1 if j == 0 else r.uniform(.3, .9)); ww = w * (g * .6 + .15 if j == 0 else .5 * r.random() ** 2 + .05); u = np.linspace(0, 1, 20); hw = ww * np.sin(math.pi * u ** .6) ** .8 * .5 + .25; C = p + d * ((u - .5) * ln)[:, None] + nn * (noise1(20, 8, r) * .6)[:, None]; c.fill(np.vstack([C + nn * hw[:, None], (C - nn * hw[:, None])[::-1]]), INK)
    return c.rgba()[..., 3]


def panel(img, box, lw=7., gutter='#f2f1ee', seed=2):
    """ruled frame: everything outside is gutter paper; lines by ruling pen, crossing a hair at the corners"""
    x0, y0, x1, y1 = box; h, w = img.shape[:2]; m = np.zeros((h, w), f32); cv2.rectangle(m, (int(x0), int(y0)), (int(x1), int(y1)), 1., -1); g = sheet(3, h, w); out = g * (1 - m[..., None]) + img * m[..., None]; c = Cel(w, h); r = np.random.default_rng(seed)
    for a, b in (((x0 - 3, y0), (x1 + 4, y0)), ((x1, y0 - 4), (x1, y1 + 3)), ((x1 + 3, y1), (x0 - 4, y1)), ((x0, y1 + 3), (x0, y0 - 3))): P = np.linspace(a, b, 60) + r.normal(0, .25, (60, 2)); c.line(P, lw * r.uniform(.94, 1.06), INK, taper=(.004, .004), press=.06, seed=int(r.integers(1 << 30)), tremor=.2)
    q = c.rgba(); return out * (1 - q[..., 3:4]) + q[..., :3]


# ------------------------------------------------------------ EGO on the page
def ego3d(box, vp=(960, 640), depth=.075, seed=3, wear=.60, letters='EGO', M=None, h=H, w=W, gap=40.):
    """the word as a slab: face (worn, 0..1), side (solid black walls toward the vanishing point), hatch (white cuts on the walls), edge (bevel line), tone (darkness field on the faces)"""
    k = ('ego3d', tuple(box), tuple(vp), depth, seed, wear, letters, h, w, gap, None if M is None else tuple(np.ravel(M)))
    if k in _c: return _c[k]
    P = ego_paths(box, letters, gap, M); vp = np.array(vp, float); s = 1 - depth; cs, ch = Cel(w, h), Cel(w, h); r = np.random.default_rng(seed + 5)
    for c in letters:
        for q in P[c]:
            q = np.asarray(q, float); n = len(q)
            for i in range(n):
                a, b = q[i], q[(i + 1) % n]; cs.fill(np.array([a, b, vp + (b - vp) * s, vp + (a - vp) * s]), '#000000')
                L = np.hypot(*(b - a))
                if L < 30: continue
                for _ in range(int(L / 10)):                                    # wall hatching: runs back from the edge, thins out
                    t = r.random(); p = a + (b - a) * t; e = vp + (p - vp) * (1 - depth * r.uniform(.25, .95)); p = vp + (p - vp) * (1 - depth * r.uniform(0.02, .25)); ch.line(np.array([p, (p + e) / 2 + r.normal(0, 1.2, 2), e]), r.uniform(.8, 3.2), '#ffffff', taper=(.1, .7), press=.4, seed=int(r.integers(1 << 30)))
    side, hatch = cs.rgba()[..., 3], ch.rgba()[..., 3]; face = ego_mask(box, letters=letters, wear=wear, seed=seed, M=M, h=h, w=w, gap=gap, chips=1.1); solid = ego_mask(box, letters=letters, wear=0, seed=seed, M=M, h=h, w=w, gap=gap, chips=1.1)
    _c[k] = dict(face=face, solid=solid, side=side, hatch=hatch); return _c[k]


# ------------------------------------------------------------ the spark: the only colour on the page
def spark(img, cx, cy, R, f=0, seed=0, col=VIOLET, pitch=9.):
    """violet halftone burst of radius R at (cx, cy): dot field, ragged star, forked bolts, white heart.  Redrawn every 2 frames"""
    h, w = img.shape[:2]; k = int(f // 2) + seed * 100; r = np.random.default_rng(7000 + k); x, y = grid(h, w); d = np.hypot(x - cx, y - cy); v = rgb(col); hot, deep = rgb('#d9b4ff'), rgb('#4a0fa8')
    reach = R * 2.6; D = np.clip(1.06 - d / reach, 0, 1) ** 1.35 * .86; cov = dots(np.where(d < reach, D, 0), pitch, 15., h, w); img = img * (1 - cov[..., None] * .96) + v * cov[..., None] * .96                                   # dots of light, printed in violet
    def star(n, r0, r1, jit, sd):
        q = np.random.default_rng(sd); a = np.linspace(0, 2 * math.pi, n, endpoint=False) + q.normal(0, jit, n); P = []
        for i in range(n):
            b = a[i] + math.pi / n; rr = r1 * q.gamma(5., .2) * (1.75 if q.random() < .16 else 1); P += [(cx + math.cos(a[i]) * r0 * q.uniform(.85, 1.1), cy + math.sin(a[i]) * r0 * q.uniform(.85, 1.1)), (cx + math.cos(b) * rr, cy + math.sin(b) * rr)]
        return np.array(P)
    c = Cel(w, h)
    c.fill(star(26, R * .62, R * 1.22, .05, 70 + k), '#0b0612'); c.fill(star(26, R * .60, R * 1.12, .05, 70 + k), deep); c.fill(star(22, R * .50, R * .94, .06, 71 + k), col); c.fill(star(18, R * .36, R * .66, .07, 72 + k), hot); c.fill(star(14, R * .22, R * .42, .07, 73 + k), '#ffffff')
    for j in range(5):                                                          # bolts
        a = r.uniform(0, 2 * math.pi); L = R * r.uniform(1.7, 3.0); n = 11; p = np.array([cx, cy]) + np.array([math.cos(a), math.sin(a)]) * R * .5; P = [p]; dr = a
        for i in range(n): dr = a + r.normal(0, .62); p = p + np.array([math.cos(dr), math.sin(dr)]) * L / n; P.append(p)
        P = np.array(P); wd = max(3., R * .075)
        for q, ww, cc in ((P, wd * 1.9, '#0b0612'), (P, wd * 1.35, col), (P, wd * .5, '#ffffff')): c.line(q, ww, cc, taper=(.05, .75), press=.3, seed=j + k, tremor=0)
        b = int(r.integers(3, 8)); q = P[b]; a2 = dr + r.choice([-1, 1]) * r.uniform(.5, 1.0); Q = [q]
        for i in range(5): q = q + np.array([math.cos(a2 + r.normal(0, .5)), math.sin(a2 + r.normal(0, .5))]) * L * .07; Q.append(q)
        for ww, cc in ((wd * 1.1, '#0b0612'), (wd * .8, col), (wd * .28, '#ffffff')): c.line(np.array(Q), ww, cc, taper=(.05, .85), seed=j + k + 9, tremor=0)
    for _ in range(46):                                                         # chips of light
        a = r.uniform(0, 2 * math.pi); q = R * (1.1 + r.gamma(1.4, .5)); L = R * r.uniform(.03, .16); p = np.array([cx + math.cos(a) * q, cy + math.sin(a) * q]); e = p + np.array([math.cos(a), math.sin(a)]) * L; n_ = np.array([-math.sin(a), math.cos(a)]) * L * .18; c.fill(np.array([p, (p + e) / 2 + n_, e, (p + e) / 2 - n_]), col if r.random() < .7 else '#0b0612')
    q = c.rgba(); return img * (1 - q[..., 3:4]) + q[..., :3]


# ------------------------------------------------------------ ready-made page
def page_bg(f=0, cx=1010, cy=668, R=None, panel_=True, seed=3, sparkle=True):
    """EGO page without the figure: paper, flash, toned ground, 3-D letters.  returns RGB; put the figure on it, then spark(), then print_()"""
    key = ('bg', seed, cx, cy, panel_)
    if key not in _c:
        x, y = grid(); img = sheet(seed); g = fbm(H, W, (260, 90), (1, .5), 5)
        fld = np.clip(rad(cx, cy, 300, 1120, .0, .80, .80) + .05 * g, 0, .97); img = ink_on(img, dots(fld, 9.2, 45.))
        img = ink_on(img, flash(cx, cy, 600, .70, 900, 21, w=15., clump=1.3)); img = ink_on(img, flash(cx, cy, 880, .74, 420, 22, w=30., clump=1.6))
        h_ = 640; L = ego3d(((W - ego_width(h_, gap=40.)) / 2, 86, h_), (cx, cy), .095, 3, .60)
        halo = cv2.dilate(np.maximum(L['solid'], L['side']), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))); img = img * (1 - halo[..., None]) + rgb(PAPER) * halo[..., None]
        img = ink_on(img, L['side']); w_ = L['hatch'] * L['side'] * (1 - L['solid']); img = img * (1 - w_[..., None]) + rgb(PAPER) * w_[..., None]
        ol = cv2.dilate(L['solid'], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))); img = ink_on(img, ol)
        fc = sheet(seed + 4); t = np.clip(lin((0, 300), (0, 86 + h_ + 10), -.10, .52) + .10 * rad(cx, cy, 200, 1000, 1, 0) + .06 * g, 0, .8); fc = ink_on(fc, dots(t * .75, 6.4, 75.)); fc = fc * (1 - .0)
        a = L['face'][..., None]; img = img * (1 - a * L['solid'][..., None]) + fc * a * L['solid'][..., None]
        hl = np.clip(L['solid'] - np.roll(L['solid'], (4, 4), (0, 1)), 0, 1) * L['face']; img = img * (1 - hl[..., None]) + np.array([.985, .985, .98], f32) * hl[..., None]
        _c[key] = img.astype(f32)
    img = _c[key].copy()
    if R and sparkle: img = spark(img, cx, cy, R, f)
    return img


def ego_page(f=0, R=150., cx=1010, cy=700, box=(34, 30, W - 34, H - 30)):
    """demo of the whole page, no figure"""
    img = page_bg(f, cx, cy); img = spark(img, cx, cy, R, f); img = panel(img, box, 8.); return print_(img)
