"""The brush: 吾 is written, waits one beat, gets its heart 忄 and becomes 悟.  Four-star-ball seal, hand-lettered credit.

    S = Satori(); S.setup()            # ~20 s first time, then cached in cache/
    img = S.frame(f)                   # h5 local frame 0..59 -> RGB (1080, 1920, 3)
    S.TIMES                            # (start, end) frames of the ten strokes (7 of 吾, 3 of 忄)
    ball_seal(size)                    # cinnabar four-star ball, ink 0..1 (size, size);  stamp_ball(img, x, y, size, rot, f_since)
    lines = pen('a fan film', x, y, size); pen_ink([lines, ...], t)      # our own hand, written up to t 0..1
    GO_STROKES, HEART_STROKES          # brush skeletons on a 1000 em (悟), for anyone who needs the character itself
"""
import numpy as np, cv2, math, os
from . import W, H, comp
from .paint import rgb, f32, fbm, smooth, grid, mix
from .ink import Cel, spl, resample, noise1
from .sumi import (Page, Mark, splash, fling, paper, lay, page_state, wet_light, page_save, page_load, tooth, _fibre, blob)

D = math.radians


# ------------------------------------------------------------ stroke makers (em units, technique from our JJK film)
def S_(k, c=(), **o): return dict(k=[tuple(q) for q in k], c=tuple(c), **o)


def yoko(x0, y0, x1, y1, w, head=1.18, waist=.74, tail=1.16, bow=10, end='tome', **o):
    dx, dy = x1 - x0, y1 - y0
    P = lambda t, b=0.: (x0 + dx * t, y0 + dy * t - bow * 4 * t * (1 - t) * (1 + b))
    k = [(x0 - .34 * w, y0 - .46 * w, .16 * w, .8), (x0 - .02 * w, y0 - .02 * w, head * w, .7),
         (*P(min(.28, .62 * w / abs(dx) + .06)), (waist + .12) * w, 1.2), (*P(.55), waist * w, 1.5), (*P(.84), (waist + .12) * w, 1.1)]
    if end == 'tome': k += [(x1 - .16 * w, y1 - .07 * w, tail * .96 * w, .55), (x1 + .04 * w, y1 + .10 * w, tail * .90 * w, .4)]
    elif end == 'thin': k += [(x1, y1, (waist + .05) * w, 1.2), (x1 + .16 * w, y1 + .02 * w, .3 * w, 1.2)]
    else: k += [(x1, y1, .12 * w, 1.6)]
    return S_(k, (), **o)


def tate(x0, y0, x1, y1, w, head=1.18, waist=.84, end='tome', bow=0, tail=1.0, tip=1.0, **o):
    dx, dy = x1 - x0, y1 - y0
    P = lambda t: (x0 + dx * t + bow * 4 * t * (1 - t), y0 + dy * t)
    k = [(x0 - tip * .36 * w, y0 - tip * .40 * w, .16 * w, .8), (x0, y0, head * w, .7), (*P(min(.3, .75 * w / abs(dy) + .05)), (waist + .10) * w, 1.1), (*P(.58), waist * w, 1.4)]
    if end == 'tome': k += [(*P(.86), (waist + .08) * w, 1.), (x1, y1 - .05 * w, tail * w, .6), (x1 - .03 * w, y1 + .12 * w, .55 * w, .6)]
    elif end == 'thin': k += [(*P(.9), (waist - .02) * w, 1.1), (x1, y1, .6 * w, 1.)]
    else: k += [(*P(.80), waist * w, 1.3), (x1, y1, .05 * w, 1.7)]
    return S_(k, (), **o)


def ore(x0, y0, xc, yc, x1, y1, w, head=.80, waist=.70, shoulder=1.22, body=.92, bow=8, w2=None, foot=.62, **o):
    w2 = w2 or w; dx, dy = xc - x0, yc - y0; ex, ey = x1 - xc, y1 - yc; dp = (shoulder * w2 - waist * w) * .5
    k = [(x0 - .10 * w, y0 - .10 * w, .22 * w, .9), (x0 + .10 * w, y0, head * w, 1.0), (x0 + dx * .36, y0 + dy * .36 - bow, (waist + .02) * w, 1.3),
         (x0 + dx * .70, y0 + dy * .70 - bow * .7, waist * w, 1.3), (xc - .30 * w, yc + .02 * dp, (waist + .10) * w, .9),
         (xc + .06 * w, yc + dp * .92, shoulder * w2, .42), (xc + ex * .20 + .02 * w, yc + ey * .20 + dp * .9, (body + .14) * w2, .8),
         (xc + ex * .58, yc + ey * .58, body * w2, 1.25), (x1, y1 - .12 * w2, (body + .06) * w2, .8), (x1 - .04 * w, y1 + .06 * w2, foot * w2, .6)]
    return S_(k, (5,), **o)


def ten(x0, y0, x1, y1, w, **o):
    dx, dy = x1 - x0, y1 - y0
    return S_([(x0, y0, .10 * w, 1.1), (x0 + dx * .45, y0 + dy * .40, .66 * w, 1.), (x0 + dx * .86, y0 + dy * .84, 1.10 * w, .6),
               (x1, y1, .96 * w, .5), (x1 - .16 * w, y1 + .02 * w, .4 * w, .7)], **o)


# ------------------------------------------------------------ the character (1000 em, stroke order)
GO_STROKES = [
    yoko(482, 152, 872, 108, 84, bow=10, waist=.72, tail=1.16, dry=.44, deplete=.7, seed=71),                          # 一
    tate(664, 152, 574, 478, 82, head=1.0, waist=.84, end='thin', tip=.8, dry=.36, deplete=.5, seed=72),               # the leaning stem of 五
    ore(492, 326, 846, 286, 800, 474, 80, head=.92, waist=.70, shoulder=1.34, body=.96, bow=9, dry=.40, deplete=.5, split=.7, seed=73),   # ㇕
    S_([(388, 474, 18, .8, 0), (420, 522, 116, .65, -.15), (520, 500, 84, 1.2, .0), (640, 476, 70, 1.5, .08), (780, 460, 68, 1.5, .14), (900, 458, 80, 1.2, .08), (972, 466, 112, .7, -.28), (1014, 486, 118, .42, -.45), (1004, 500, 70, .5, -.45)], (),
       dry=.56, deplete=.62, load=.80, split=.8, speed=1.1, seed=74),                                                   # the long one: the arm of 吾
    tate(490, 652, 523, 908, 84, head=1.12, waist=.84, end='tome', tail=.86, tip=.62, dry=.46, seed=75),               # 口 left
    ore(506, 666, 866, 622, 828, 888, 86, waist=.66, shoulder=1.36, body=1.02, bow=9, dry=.48, deplete=.65, split=.7, seed=76),           # 口 top and right
    yoko(552, 894, 812, 868, 74, head=.78, waist=.80, tail=.88, bow=3, end='thin', dry=.58, seed=77),                  # 口 floor
]
HEART_STROKES = [
    S_([(112, 296, 10, 1.0), (103, 366, 56, 1.0), (82, 446, 94, .6), (60, 514, 90, .45), (50, 548, 42, .6)], dry=.18, deplete=.4, bleed=4.5, seed=81),          # left drop, falling
    S_([(290, 222, 10, 1.0), (316, 276, 62, 1.0), (344, 334, 98, .5), (345, 372, 84, .45), (322, 390, 36, .7)], dry=.18, deplete=.4, bleed=4.5, seed=82),      # right drop, caught
    S_([(168, 22, 18, .8), (204, 56, 126, .6), (211, 160, 102, 1.0), (208, 420, 86, 1.35), (208, 700, 92, 1.2), (208, 850, 98, 1.0), (207, 934, 66, 1.15), (206, 1048, 5, 1.7)],
       dry=.50, deplete=1.1, load=.50, split=.85, fork=.9, speed=1.15, seed=83),                                       # the spine: from head to needle
]


def put(pg, strokes, box, times, wt=1.0, **over):
    """write skeletons on a sumi.Page.  box = (x, y, size)"""
    x, y, s = box; k = s / 1000.; out = []
    for q, (a, b) in zip(strokes, times):
        q = dict(q); keys = [(x + p[0] * k, y + p[1] * k, p[2] * k * wt) + tuple(p[3:]) for p in q.pop('k')]; c = q.pop('c'); q.update(over)
        out.append(pg.stroke(keys, c, t0=a, t1=b, **q))
    return out


# ------------------------------------------------------------ ink creeping into the fibres (nijimi), in time
class Bleed:
    """soft grey halo that keeps growing from the wet places of a finished sheet.  B = Bleed(ink, wet, when); halo = B.at(t)"""

    def __init__(s, pg, R=15., seed=3, only=None):
        h, w = pg.h, pg.w; d = np.zeros((h, w), f32); wt = np.zeros((h, w), f32); tm = np.full((h, w), 1e6, f32)
        for i, m in enumerate(pg.marks):
            if m.kind != 'stroke' or (only is not None and i not in only): continue
            hh, ww = m.d.shape; x0, y0 = max(0, m.ox), max(0, m.oy); x1, y1 = min(w, m.ox + ww), min(h, m.oy + hh)
            a = (slice(y0 - m.oy, y1 - m.oy), slice(x0 - m.ox, x1 - m.ox)); q = (slice(y0, y1), slice(x0, x1))
            md = m.d[a].astype(f32); when = m.t0 + np.clip(m.arr[a].astype(f32), 0, 1) * (m.t1 - m.t0); tm[q] = np.where((md > .5) & (when < tm[q]), when, tm[q])
            d[q] = np.maximum(d[q], md); wt[q] = np.maximum(wt[q], m.wet[a].astype(f32))
        core = (d > .55); ys, xs = np.nonzero(core); M = int(R * 2.4) + 4
        s.y0, s.x0, s.y1, s.x1 = max(0, ys.min() - M), max(0, xs.min() - M), min(h, ys.max() + M), min(w, xs.max() + M); q = (slice(s.y0, s.y1), slice(s.x0, s.x1))
        core = core[q]; dist, lab = cv2.distanceTransformWithLabels((~core).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL); iy, ix = np.nonzero(core)
        w2 = cv2.GaussianBlur(wt[q], (0, 0), 7) * 1.9; lut = np.zeros(int(lab.max()) + 1, f32); lut[lab[iy, ix]] = tm[q][iy, ix]; lw = np.zeros(int(lab.max()) + 1, f32); lw[lab[iy, ix]] = np.clip(w2[iy, ix], 0, 1)
        f = _fibre(h, w)[q]; f = np.roll(f, (seed * 131 % 997, seed * 57 % 991), (0, 1)); n = fbm(s.y1 - s.y0, s.x1 - s.x0, (70, 26), (1, .6), seed)
        s.dist = dist.astype(f32); s.t = cv2.GaussianBlur(lut[lab], (0, 0), 2.); s.reach = R * (cv2.GaussianBlur(lw[lab], (0, 0), 3.) ** 1.6) * (.22 + 1.5 * f * f + .30 * f) * np.clip(1 + .5 * n, .2, 2.2); s.R = R; s.core = core; s.h, s.w = h, w

    def at(s, t, tau=16.):
        g = 1 - np.exp(-np.maximum(0, t - s.t) / tau); r = s.reach * g; a = np.clip((r - s.dist) / (2.2 + .30 * r) + .25, 0, 1) * (s.dist > 0)
        a = a * (.30 + .70 * np.exp(-s.dist / (s.R * .34))); rim = np.exp(-((r - s.dist - 1.5) / 1.8) ** 2) * smooth(2, 7, r) * .16                  # the tide mark
        o = np.zeros((s.h, s.w), f32); o[s.y0:s.y1, s.x0:s.x1] = cv2.GaussianBlur(np.clip(a * .60 + rim, 0, 1), (0, 0), .7); return o


# ------------------------------------------------------------ the seal: the four-star ball in cinnabar
def ball_seal(size=150, seed=0, ss=4):
    """round seal: the ball printed red, four stars, its glint and a rim line carved white.  ink thickness (size, size)"""
    r = np.random.default_rng(seed); n = int(size * ss); u = n / 200.; b = np.zeros((n, n), f32)
    def rough(P, amp, sc=14.):
        Q = resample(np.vstack([P, P[:1]]), 2.5); m = len(Q); T = np.gradient(Q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9
        nz = amp * (noise1(m, sc, r) + .6 * noise1(m, 4, r)); nz = nz * np.clip(np.minimum(np.arange(m), np.arange(m)[::-1]) / 20., 0, 1)
        return Q + np.c_[-T[:, 1], T[:, 0]] * nz[:, None]
    P_ = lambda P, v: cv2.fillPoly(b, [np.round(P * 16).astype(np.int32)], v, cv2.LINE_AA, 4)
    a = np.linspace(0, 2 * math.pi, 120, endpoint=False); C = lambda x, y, R: np.c_[x + np.cos(a) * R, y + np.sin(a) * R] * u
    P_(rough(C(100, 100, 93), .9 * u), 1.); P_(rough(C(100, 100, 85.5), .7 * u), 0.); P_(rough(C(100, 100, 79), .8 * u), 1.)          # rim line, ball
    def star(x, y, R, rot):
        k = np.arange(10); rr = np.where(k % 2 == 0, R, R * .43) * (1 + .05 * r.normal(0, 1, 10)); t = rot - math.pi / 2 + k * math.pi / 5; P = np.c_[x + np.cos(t) * rr, y + np.sin(t) * rr] * u
        Q = []
        for i in range(10): Q += [P[i] * .90 + P[i - 1] * .10, P[i], P[i] * .90 + P[(i + 1) % 10] * .10] if i % 2 else [P[i] * .94 + P[i - 1] * .06, P[i] * .94 + P[(i + 1) % 10] * .06]   # knife-cut tips
        P_(rough(np.array(Q), .30 * u, 7), 0.)
    for x, y, rt in ((82, 66, -.14), (134, 84, .10), (68, 120, .06), (121, 138, -.10)): star(x, y, 26, rt)
    t = np.linspace(D(196), D(262), 24); P_(rough(np.vstack([np.c_[100 + np.cos(t) * 69, 100 + np.sin(t) * 69], np.c_[100 + np.cos(t[::-1]) * (69 - 7.0 * np.sin(np.linspace(.1, 3.0, 24))), 100 + np.sin(t[::-1]) * (69 - 7.0 * np.sin(np.linspace(.1, 3.0, 24)))]]) * u, .3 * u, 8), 0.)   # glint
    for _ in range(11):                                                         # chips of the stone
        th = r.uniform(0, 2 * math.pi); R = r.uniform(1.2, 4.6) * u; q = r.choice([93, 93, 79]) + r.normal(0, 1.5); m = 11; aa = np.linspace(0, 2 * math.pi, m, endpoint=False); rad = R * (1 + .3 * r.normal(0, 1, m)); P_(np.c_[(100 + math.cos(th) * q) * u + np.cos(aa) * rad, (100 + math.sin(th) * q) * u + np.sin(aa) * rad], 0.)
    f = np.tile(tooth(2304, 2304), (1 + n // 2304, 1 + n // 2304))[:n, :n]; y, x = np.mgrid[0:n, 0:n] / n
    press = .84 + .24 * np.cos((x - .36) * 1.9) * np.cos((y - .58) * 1.7) + .085 * fbm(n, n, (n / 5., n / 12., n / 30.), (1, .7, .5), seed) - .10
    ink = b * smooth(.0, .30, press - .60 * smooth(.45, 1.0, cv2.GaussianBlur(f, (0, 0), 1.2)) + .10); ink = ink * (.76 + .24 * smooth(.6, 1.0, press + .12 * fbm(n, n, (n / 40., n / 90.), (1, .6), seed + 2)))
    ink = ink * (1 + .40 * (b - cv2.GaussianBlur(b, (0, 0), 2.2 * u)))          # paste squeezes to the edges
    return np.clip(cv2.resize(ink.astype(f32), (size, size), interpolation=cv2.INTER_AREA), 0, 1)


_seal = {}


def stamp_ball(base, x, y, size=150, rot=-.05, col='#c8342b', amt=1.0, sx=1., sy=1., seed=2):
    if (size, seed) not in _seal: _seal[size, seed] = ball_seal(size, seed)
    s = _seal[size, seed]; h, w = base.shape[:2]; M = cv2.getRotationMatrix2D((size / 2, size / 2), math.degrees(-rot), 1.); M = np.diag([sx, sy]) @ M + np.array([[0, 0, size / 2 * (1 - sx)], [0, 0, size / 2 * (1 - sy)]]); M[:, 2] += [x - size / 2, y - size / 2]
    a = cv2.warpAffine(s, M, (w, h), flags=cv2.INTER_LINEAR); A = (a * .96 * amt)[..., None]
    return base * (1 - A) + rgb(col) * (.64 + .36 * np.clip(base, 0, 1)) * A, a


# ------------------------------------------------------------ our own hand (x-height 100, baseline 0, y down)
PEN = {
    'a': (82, 'M62 -84C46 -106 10 -98 8 -46C6 6 50 8 64 -40M67 -98C62 -62 58 -18 66 -4C70 3 78 -2 84 -10'), 'b': (74, 'M16 -186C14 -120 8 -62 12 -22C18 10 56 6 63 -42C66 -90 34 -104 14 -60'),
    'c': (66, 'M62 -78C50 -108 8 -100 8 -46C8 6 50 8 68 -18'), 'e': (68, 'M12 -48C34 -46 61 -58 58 -80C53 -108 12 -100 10 -46C8 8 50 10 70 -20'),
    'f': (54, 'M62 -168C46 -192 26 -174 26 -130C26 -80 28 -30 22 6M0 -92C20 -97 40 -98 58 -99'), 'h': (82, 'M16 -186C15 -120 12 -60 10 0M12 -48C24 -84 52 -110 62 -82C68 -58 61 -24 66 -4C70 4 78 -2 84 -8'),
    'i': (36, 'M13 -97C14 -64 8 -24 16 -4C20 4 28 -2 34 -8M16 -138C19 -140 20 -136 17 -132'), 'k': (76, 'M16 -186C15 -120 12 -60 10 0M62 -97C46 -76 28 -58 14 -50C34 -44 46 -18 62 -2C66 2 74 0 80 -6'),
    'l': (36, 'M16 -186C13 -120 7 -40 16 -6C20 4 30 0 36 -8'), 'm': (122, 'M9 -98C14 -66 14 -30 12 0M14 -52C22 -84 44 -110 54 -82C60 -58 56 -24 56 0M57 -52C66 -84 90 -112 100 -82C106 -58 99 -22 104 -4C108 4 116 -2 122 -8'),
    'n': (82, 'M9 -98C14 -66 14 -30 12 0M14 -52C24 -84 52 -112 62 -82C68 -60 61 -24 66 -4C70 4 78 -2 84 -8'), 'o': (74, 'M40 -98C14 -101 6 -66 8 -40C11 9 64 9 68 -44C70 -80 56 -100 34 -95'),
    'r': (58, 'M9 -98C14 -66 14 -30 12 0M14 -44C22 -80 40 -106 60 -92'), 's': (60, 'M57 -82C46 -106 14 -100 16 -76C18 -54 57 -56 54 -26C50 4 16 6 4 -14'),
    't': (54, 'M30 -152C25 -100 18 -40 26 -10C32 6 46 2 54 -10M0 -93C18 -98 36 -99 54 -101'), 'y': (76, 'M8 -97C8 -60 14 -32 32 -31C50 -32 62 -66 67 -98C66 -40 56 30 36 58C20 76 4 66 2 52'),
    'A': (116, 'M0 4C24 -60 48 -130 66 -186C80 -130 96 -60 118 -2M24 -62C50 -69 76 -69 100 -66'), 'T': (112, 'M0 -170C38 -177 80 -183 122 -180M62 -180C60 -120 60 -50 55 4'),
}


def pen(text, x, y, size=44., seed=0, nib=D(52), weight=.092, track=.10, slant=.07, align='center', uneven=1.0):
    """hand-lettered line -> strokes [(centre line, half-width vectors, order0, order1)], x-height = size / 2"""
    from .kit import svg
    r = np.random.default_rng(seed); k = size / 200.; st = []; cx = 0.; n = len(text)
    for ch in text:
        if ch == ' ': cx += 62 + r.normal(0, 4); continue
        adv, d = PEN[ch]; sc = 1 + .022 * uneven * r.normal(); dy = 2.4 * uneven * r.normal(); rot = .014 * uneven * r.normal(); c, s = math.cos(rot), math.sin(rot)
        for q in svg(d, 12): q = np.c_[q[:, 0] - q[:, 1] * slant, q[:, 1]] * sc @ np.array([[c, s], [-s, c]]); st.append(q + [cx, dy])
        cx += adv * (1 + track) + r.normal(0, 1.8 * uneven)
    ox = {'left': 0, 'center': -cx / 2, 'right': -cx}[align]; L = np.cumsum([0] + [len(q) for q in st]); out = []
    for i, q in enumerate(st):
        P = resample((q + [ox, 0]) * k + [x, y], .5); m = len(P)
        if m < 5: P = resample(np.vstack([P, P[-1:] + [[size * .02, size * .015]]]), .25); m = len(P)
        T = np.gradient(P, axis=0); an = np.arctan2(T[:, 1], T[:, 0]); w = size * weight * (.42 + .58 * np.abs(np.sin(an + nib)) ** 1.2); t = np.linspace(0, 1, m)
        w = w * (1 + .14 * noise1(m, max(6, size * .6), r)) * (.50 + .50 * np.clip(np.minimum(t * m, (1 - t) * m) / (size * .15), 0, 1) ** .6) + size * weight * .30 * np.exp(-((1 - t) * m * .5 / (size * .05)) ** 2) + size * weight * .16 * np.exp(-(t * m * .5 / (size * .05)) ** 2)
        T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; out.append((P, np.c_[-T[:, 1], T[:, 0]] * w[:, None] * .5, L[i] / L[-1], L[i + 1] / L[-1]))
    return out


def pen_ink(lines, t=1.0, h=H, w=W, ss=4):
    """rasterise pen() lines written up to t (0..1) -> ink thickness HxW"""
    al = [s for ln in lines for s in ln]; P = np.vstack([s[0] for s in al]); x0, y0 = np.floor(P.min(0) - 8).astype(int); x1, y1 = np.ceil(P.max(0) + 8).astype(int); x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    b = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), f32); nl = len(lines); len_ = [sum(len(s[0]) for s in ln) for ln in lines]; cum = np.cumsum([0] + len_) / max(1, sum(len_))
    for li, ln in enumerate(lines):
        for C, N, a0, a1 in ln:
            a, e = cum[li] + a0 * (cum[li + 1] - cum[li]), cum[li] + a1 * (cum[li + 1] - cum[li]); u = np.clip((t - a) / max(1e-6, e - a), 0, 1); m = int(u * (len(C) - 1)) + 1
            if u <= 0 or m < 2: continue
            for i in range(m - 1):
                q = np.array([C[i] + N[i], C[i + 1] + N[i + 1], C[i + 1] - N[i + 1], C[i] - N[i]]); cv2.fillConvexPoly(b, np.round((q - [x0, y0]) * ss * 16).astype(np.int32), 1., cv2.LINE_AA, 4)
                cv2.circle(b, tuple(int(v) for v in np.round((C[i] - [x0, y0]) * ss * 16)), max(1, int(np.hypot(*N[i]) * ss * 16)), 1., -1, cv2.LINE_AA, 4)
    b = cv2.GaussianBlur(b, (0, 0), ss * .22); f = np.tile(tooth(2304, 2304), (1 + b.shape[0] // 2304, 1 + b.shape[1] // 2304))[:b.shape[0], :b.shape[1]]
    b = smooth(.28, .72, b + (f - .5) * .30 * (b > .02) * (1 - b * .4)) * (.82 + .18 * cv2.GaussianBlur(f, (0, 0), 6)); out = np.zeros((h, w), f32); out[y0:y1, x0:x1] = cv2.resize(b.astype(f32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    return out


# ------------------------------------------------------------ the brush itself (seen from above, leaning to the writer's right)
def brush_layer(P, m, w, lift=0., wet=1.0, seed=5, h=H, w_=W, d=(.84, .54), Lt=190., wmax=100.):
    """P tip contact (px), m unit motion (smoothed), w pressed width px (0 = pointed, off the paper), lift px above the sheet
    -> (premultiplied RGBA layer, shadow mask 0..1).  Goat and weasel hair, horn collar, bamboo running out of focus toward the lens."""
    P = np.asarray(P, float); d = np.array(d) / np.hypot(*d); e = np.array([-d[1], d[0]]); m = np.asarray(m, float); r = np.random.default_rng(seed); pr = np.clip(w / wmax, 0, 1.25)
    Fr = P + d * (Lt * (1 - .30 * pr)); fw = 35.; n = np.array([-m[1], m[0]])
    tip = P - m * w * .62 * min(1, pr * 1.6) ; bel = P + d * Lt * (.36 - .10 * pr); bw = 34 + 24 * pr
    # outline: collar -> belly -> foot -> tip -> foot -> belly -> collar, always on the proper side
    sg = 1. if n @ e >= 0 else -1.; fh = max(1.6, w * .47)
    L_ = [Fr + e * fw, Fr * .55 + bel * .45 + e * (fw * .55 + bw * .52), bel + e * bw, P + n * sg * fh + d * fh * .30 * (1 - abs(m @ d)), tip]
    R_ = [Fr - e * fw, Fr * .55 + bel * .45 - e * (fw * .55 + bw * .52), bel - e * bw, P - n * sg * fh + d * fh * .30 * (1 - abs(m @ d)), tip]
    A = spl(np.array(L_), n=10); B = spl(np.array(R_), n=10); out = np.vstack([A, B[::-1]])
    x0, y0 = np.floor(np.minimum(out.min(0), Fr) - 30).astype(int); x1, y1 = np.ceil(np.maximum(out.max(0), Fr + d * 120) + 60).astype(int)
    c = Cel(w_, h); c.fill(out, '#141013')
    c.clip(out)                                                                                        # hair
    k = len(A)
    for i in range(150):
        u = r.beta(1.2, 1.2); q = A * (1 - u) + B * u; j0 = int(r.uniform(0, .35) * k); j1 = int(r.uniform(.45, 1.) * k)
        if j1 - j0 < 4: continue
        ed = abs(u - .30); v = float(np.clip(.95 - ed * 2.0, 0, 1)); col = mix(rgb('#1a1418'), mix(rgb('#5d4a42'), rgb('#8b7a72'), v ** 2), v * r.uniform(.2, 1.) * .8)
        c.line(q[j0:j1] + r.normal(0, .8, 2), r.uniform(.6, 1.7), col, taper=(.1, .6), seed=int(r.integers(1 << 30)), tremor=.2, alpha=.85)
    q = A * .72 + B * .28; c.line(q[int(.12 * k):int(.66 * k)], 5.0, '#cfc9cf', taper=(.5, .7), seed=1, alpha=.34 * wet); c.line(q[int(.2 * k):int(.5 * k)] + e * 5, 1.6, '#ffffff', taper=(.5, .6), seed=2, alpha=.55 * wet)
    sh = A * .06 + B * .94; c.line(sh[int(.1 * k):int(.8 * k)], 9., '#060406', taper=(.3, .6), seed=3, alpha=.6); c.unclip()
    for i in range(16):                                                                                # strays at the edge of the belly
        s_ = A if r.random() < .5 else B; j = int(r.uniform(.35, .85) * k); g = (e if s_ is A else -e) * r.uniform(.5, 4.5); T = s_[min(k - 1, j + 6)] - s_[max(0, j - 6)]; T = T / (np.hypot(*T) + 1e-6); p0 = s_[max(0, j - 8)]; c.line(spl(np.array([p0, s_[j] + g, s_[min(k - 1, j + 9)] + g * 1.6]), n=6), r.uniform(.5, 1.0), '#1c1418', taper=(.3, .7), seed=int(r.integers(1 << 30)))
    # collar and bamboo
    H0 = Fr - d * 5; H1 = Fr + d * 48; H2 = Fr + d * 1500
    def seg(a, b, wa, wb): return np.vstack([a + e * wa, b + e * wb, b - e * wb, a - e * wa])
    ww = lambda s_: 37. * (1 + s_ / 800.)
    c.fill(spl(np.vstack([H0 + e * 36, H0 + e * 27 - d * 5, H0 - d * 8, H0 - e * 27 - d * 5, H0 - e * 36, H1 - e * 39, H1 + e * 39]), True, corners=(0, 4, 5, 6), n=5), '#1b1216')
    c.fill(seg(H1, H2, ww(48), ww(1500)), '#b88a4e')
    for a_, b_, col, al in ((-1.0, -.46, '#7a4f2b', 1.), (-.46, -.20, '#9a6d3b', 1.), (.40, .70, '#dcb479', 1.), (.46, .56, '#f6dfae', .9), (.90, 1.0, '#8d6336', .8)):
        c.fill(np.vstack([H1 + e * ww(48) * a_, H2 + e * ww(1500) * a_, H2 + e * ww(1500) * b_, H1 + e * ww(48) * b_]), col, al)
    c.fill(np.vstack([H0 + e * 12, H1 + e * 14, H1 + e * 20, H0 + e * 18]), '#6d5560', .9); c.fill(np.vstack([H0 + e * 15, H1 + e * 16.5, H1 + e * 18, H0 + e * 16.5]), '#e9dfe4', .7)
    for s_ in (250., 262.): q_ = Fr + d * s_; c.line(spl(np.array([q_ + e * ww(s_), q_ + d * 7 + e * 0, q_ - e * ww(s_)]), n=8), 4.2 if s_ < 255 else 2.2, '#6a4526', taper=(.1, .1), seed=9)
    L = c.rgba()
    # focus falls off along the handle
    gx, gy = grid(h, w_); s_ = np.clip(((gx - Fr[0]) * d[0] + (gy - Fr[1]) * d[1] - 40) / 520., 0, 1.6)[..., None]
    b1 = cv2.GaussianBlur(L, (0, 0), 3.2); q4 = cv2.resize(L, (w_ // 4, h // 4), interpolation=cv2.INTER_AREA); b2 = cv2.resize(cv2.GaussianBlur(q4, (0, 0), 3.4, borderType=cv2.BORDER_REPLICATE), (w_, h), interpolation=cv2.INTER_LINEAR)
    k1 = np.clip(s_ / .35, 0, 1); k2 = np.clip((s_ - .35) / .8, 0, 1); L = (L * (1 - k1) + b1 * k1) * (1 - k2) + b2 * k2
    # shadow: from the contact point, opening away from the light (upper left)
    sd = np.array([.40, .62]); S = np.zeros((h // 4, w_ // 4), f32); C0 = P + sd * lift * 1.0
    def pt(p, z): return p + sd * (z + lift)
    tf = pt(Fr, Lt * 1.0); te = pt(Fr + d * 1500, Lt + 1500 * 1.05); dv = (te - tf) / np.hypot(*(te - tf)); nv = np.array([-dv[1], dv[0]])
    po = np.vstack([C0 + nv * max(2, w * .2), tf + nv * 36, te + nv * 60, te - nv * 60, tf - nv * 36, C0 - nv * max(2, w * .2)]) / 4
    cv2.fillPoly(S, [np.round(po * 16).astype(np.int32)], 1., cv2.LINE_AA, 4); u = np.clip(((gx[::4, ::4] - C0[0]) * dv[0] + (gy[::4, ::4] - C0[1]) * dv[1]) / 620., 0, 1)[:h // 4, :w_ // 4]
    s1 = cv2.GaussianBlur(S, (0, 0), 1.2 + lift * .03); s2 = cv2.GaussianBlur(S, (0, 0), 12); S = cv2.resize((s1 * (1 - u) + s2 * u * .75), (w_, h), interpolation=cv2.INTER_LINEAR)
    return L, S * max(.25, 1 - lift / 300.)


# ------------------------------------------------------------ h5
class Satori:
    X, Y, SZ = 556, 72, 742                                                    # em box of 悟
    TIMES = [(2.6, 4.5), (5.1, 7.1), (7.8, 10.6), (11.6, 15.6), (16.5, 18.2), (18.9, 21.8), (22.5, 24.0),
             (34.0, 36.0), (37.2, 39.4), (41.4, 46.0)]
    SEAL = (1392, 742, 116); STAMP = 50; PEN = (51.5, 55.8); SUN = (960, 486, 207)
    VER = 'v6'

    def setup(s, cache=True):
        os.makedirs('cache', exist_ok=True); pa = f'cache/satori_page_{s.VER}.npz'; B = lambda a, b: (s.X + a * s.SZ / 1000., s.Y + b * s.SZ / 1000.)
        if cache and os.path.exists(pa): pg = page_load(pa)
        else:
            pg = Page(seed=4); put(pg, GO_STROKES + HEART_STROKES, (s.X, s.Y, s.SZ), s.TIMES)
            def sp(m, a, d=1.6): m.t0, m.t1 = a, a + d; pg.add(m)
            T = s.TIMES
            sp(splash(*B(474, 138), 100, seed=11, ang=D(205), spread=1.1, n=40, big=7, fingers=4, lobes=3, core=0), T[0][0] + .1)
            sp(splash(*B(1026, 470), 170, seed=12, ang=D(-12), spread=.8, n=80, big=9, fingers=6, lobes=4, core=0), T[3][1] - .8)
            sp(splash(*B(880, 606), 90, seed=14, ang=D(-35), spread=1.0, n=36, big=6, fingers=3, lobes=3, core=0), T[5][0] + 1.0)
            sp(splash(*B(170, 30), 130, seed=15, ang=D(-150), spread=.9, n=60, big=8, fingers=5, lobes=4, core=0), T[9][0] + .1)
            sp(fling(*B(206, 1036), D(88), 56, seed=16, n=34, big=4, fingers=2), T[9][1] - .7)
            sp(splash(*B(54, 524), 64, seed=17, ang=D(130), spread=.9, n=20, big=5, fingers=2, lobes=2, core=0), T[7][1] - .5); sp(splash(*B(362, 404), 50, seed=18, ang=D(30), spread=.9, n=14, big=4, fingers=2, lobes=2, core=0), T[8][1] - .5)
            page_save(pg, pa)
        s.pg = pg; s.bleed = Bleed(pg, 19., 3, only=set(range(10)))
        s.P = paper(tone='#f4ecdc', amt=1.35, seed=23); x, y = grid()
        s.light = mix(np.array([1.06, 1.045, 1.02], f32), np.array([.965, .945, .93], f32), smooth(.0, 1.25, np.hypot((x - 760) / 1500, (y - 300) / 1050))[..., None])
        s.text = [pen('a fan film', 960, 972, 40, seed=3), pen('characters by Akira Toriyama', 960, 1022, 40, seed=4)]
        s.mica = s._mica().astype(f32)
        # the fibres seen against the sun
        s.fib = np.clip(_fibre(H, W, 3) * .8 + .25 * fbm(H, W, (150, 50, 18), (1, .7, .5), 9) + .1, 0, 1.2).astype(f32); s.d = np.hypot(x - s.SUN[0], y - s.SUN[1]).astype(f32)

    @staticmethod
    def _mica():
        """a printed pattern under the writing: little curling clouds in mica, seen only where the light rakes them"""
        c = Cel(W, H, ss=1)
        def curl(x, y, R, sg=1, turns=1.35, w=7.):
            t = np.linspace(0, turns * 2 * math.pi, 90); r = R * (1 - .80 * t / t[-1]); a = -math.pi * .95 + t; c.line(np.c_[x + sg * np.cos(a) * r, y + np.sin(a) * r * .92], w, '#ffffff', taper=(.25, .35), seed=int(x), tremor=.1)
        def cloud(x, y, s, sg=1):
            curl(x, y, 62 * s, sg); curl(x + sg * 92 * s, y + 18 * s, 44 * s, sg, 1.2, 6.); curl(x - sg * 84 * s, y + 24 * s, 40 * s, -sg, 1.2, 6.)
            P = spl(np.array([(x + sg * 128 * s, y + 44 * s), (x + sg * 30 * s, y + 66 * s), (x - sg * 70 * s, y + 60 * s), (x - sg * 190 * s, y + 74 * s), (x - sg * 320 * s, y + 62 * s), (x - sg * 420 * s, y + 70 * s)]), n=12); c.line(P, 8 * s, '#ffffff', taper=(.1, .8), seed=3, tremor=.1); c.line(P[20:] + [0, 16 * s], 5 * s, '#ffffff', taper=(.2, .8), seed=4, tremor=.1)
        for x, y, s, g in ((250, 240, 1.0, 1), (1660, 300, 1.15, -1), (1590, 840, .9, 1), (330, 820, 1.1, -1), (930, 1010, .7, 1), (880, 40, .75, -1)): cloud(x, y, s, g)
        m = c.rgba()[..., 3]; return cv2.GaussianBlur(m, (0, 0), .8) * (.65 + .35 * np.clip(fbm(H, W, (26, 10), (1, .7), 41), -1, 1))

    # where the brush is at time t -> (P, motion, width, lift) or None
    def tip(s, t):
        T = s.TIMES; N = len(T)
        def on(i, u):
            f = s.pg.marks[i].info; tt = f['t']; k = int(np.clip(np.searchsorted(tt, u), 0, len(tt) - 1)); k0 = max(0, k - 26); k1 = min(len(tt) - 1, k + 8); v = np.array([f['x'][k1] - f['x'][k0], f['y'][k1] - f['y'][k0]]); return np.array([f['x'][k], f['y'][k]], float), v / (np.hypot(*v) + 1e-6), float(f['w'][k])
        out = np.array([2350., 1500.])
        for i, (a, b) in enumerate(T):
            if a <= t <= b: p, m, w = on(i, (t - a) / (b - a)); return p, m, w, 0.
            if t < a:
                p1, m1, w1 = on(i, 0.); pb = T[i - 1][1] if i else a - 4.; gap = a - pb
                if i and gap < 3: p0, m0, w0 = on(i - 1, 1.); u = np.clip((t - pb) / gap, 0, 1); e = u * u * (3 - 2 * u); h_ = math.sin(u * math.pi) * 60; mm = m0 * (1 - e) + m1 * e; return p0 * (1 - e) + p1 * e, mm / (np.hypot(*mm) + 1e-6), 10 + 20 * abs(1 - 2 * u), h_
                ent = 2.0 if i == 0 else 3.4
                if t > a - ent: u = float(np.clip((a - t) / ent, 0, 1)) ** 1.6; return p1 + np.array([1300, 860]) * u, m1, 8., 30 + 300 * u          # coming in
                if i and t < pb + 3.2: p0, m0, w0 = on(i - 1, 1.); u = float(np.clip((t - pb) / 3.2, 0, 1)) ** 1.8; return p0 + np.array([900, 640]) * u, m0, 8., 24 + 300 * u          # leaving
                return None
        p0, m0, w0 = on(N - 1, 1.); u = (t - T[-1][1]) / 3.4
        return None if u > 1 else (p0 + np.array([900, 560]) * u ** 1.8, m0, 8., 24 + 300 * u ** 1.8)

    def frame(s, f, brush=True):
        t = f + .5; base = s.P * s.light; x, y = grid(); rake = np.exp(-((x * .8 + y * .5 - 300 - 26 * f) / 620.) ** 2); base = base * (1 + s.mica * (-.030 + .075 * rake))[..., None]
        k = float(1 - smooth(2, 33, f)) ** 1.1                                                          # the sun of h4, still behind the sheet
        if k > .003:
            so = 5 + f * 2.2; disc = smooth(s.SUN[2] + so, s.SUN[2] - so, s.d); halo = np.exp(-np.maximum(0, s.d - s.SUN[2]) / 260.); tr = (1.10 - .34 * s.fib)
            w0 = math.exp(-f / 3.5); base = base * (1 - k * mix(np.array([.10, .16, .24], f32), np.array([.065, .10, .14], f32), w0)) + (disc * .27 * tr * k)[..., None] * mix(rgb('#ffc878'), rgb('#ffe6c8'), w0) + (halo * .17 * tr * k)[..., None] * mix(rgb('#f79a5c'), rgb('#fbc0a2'), w0)
        a = s.bleed.at(t)[..., None] * .9; base = base * (1 - a) + mix(rgb('#8d8379'), rgb('#4b4550'), np.clip(a * 1.3, 0, 1)) * a
        N, Fr, Po = page_state(s.pg, t, tau=26.); base = lay(base, N, None, Po, col='#141017'); base = base + wet_light(N, Fr, Po, .9, phase=1.1 + f * .02)
        if f >= s.PEN[0]: a = pen_ink(s.text, float(np.clip((t - s.PEN[0]) / (s.PEN[1] - s.PEN[0]), 0, 1)))[..., None] * .88; base = base * (1 - a) + rgb('#2a2226') * a
        w_ = float(smooth(45, 52, f)); base = base * (1 + w_ * (np.array([.035, .022, .0], f32) + (.05 * np.exp(-(s.d / 620.) ** 2))[..., None] * np.array([1., .72, .42], f32) + .05 * s.mica[..., None]))       # the heart is in: the sheet warms
        sx, sy, ss = s.SEAL; dep = float(np.interp(f, [s.STAMP - 3, s.STAMP - 1, s.STAMP, s.STAMP + 1, s.STAMP + 2, s.STAMP + 3, s.STAMP + 4, s.STAMP + 5, s.STAMP + 7], [0, .10, 1.25, 1.05, .70, .2, -.18, -.08, 0]))
        if f >= s.STAMP:
            q = {0: (1.11, .89), 1: (1.06, .94), 2: (.98, 1.02)}.get(f - s.STAMP, (1., 1.)); base, sa = stamp_ball(base, sx, sy, ss, -.06, sx=q[0], sy=q[1])
            if f < s.STAMP + 12: g = cv2.GaussianBlur(sa, (0, 0), 1.2); base = base + (np.clip(g - cv2.warpAffine(g, np.float32([[1, 0, 1.2], [0, 1, 1.6]]), (W, H)), 0, 1) * .30 * (1 - smooth(s.STAMP + 4, s.STAMP + 12, f)))[..., None] * rgb('#ffd9cc')
        if s.STAMP - 5 < f < s.STAMP + 6:                                       # shadow of the stone nobody holds
            hgt = float(np.interp(f, [s.STAMP - 5, s.STAMP, s.STAMP + 2, s.STAMP + 6], [230, 0, 0, 230])); q_ = ss / 2 + 8; u = np.linspace(0, 2 * math.pi, 48, endpoint=False); pl = np.c_[sx + hgt * .42 + 3 + np.cos(u) * q_, sy + hgt * .58 + 4 + np.sin(u) * q_]; S = np.zeros((H // 2, W // 2), f32); cv2.fillPoly(S, [np.round(pl * 8).astype(np.int32)], 1., cv2.LINE_AA, 4); S = cv2.resize(cv2.GaussianBlur(S, (0, 0), 4.5 + hgt * .11), (W, H), interpolation=cv2.INTER_LINEAR); base = base * (1 - .24 * float(np.clip(1 - hgt / 250., 0, 1)) ** .8 * S[..., None] * np.array([1.05, 1., .88], f32))
        if abs(dep) > .005:
            dx, dy = x - sx, y - sy; d = np.sqrt(dx * dx + dy * dy) + 1; g = np.exp(-(d / 165.) ** 2); sl = dep * (d / 165.) * g * 1.4; base = base * (1 - .07 * dep * g - .13 * sl * (-(dx * -.55 + dy * -.83) / d))[..., None]; am = 10. * dep * (d / 120.) * np.exp(-(d / 150.) ** 2); base = cv2.remap(base.astype(f32), x + dx / d * am, y + dy / d * am, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        if brush:
            b = s.tip(t)
            if b is not None:
                L, S = brush_layer(*b); base = base * (1 - .30 * S[..., None] * np.array([1.05, 1., .88], f32)); base = base * (1 - L[..., 3:4]) + L[..., :3]
        e = f / 59.; base = comp.cam2d(np.maximum(base, 0).astype(f32), W / 2 + 4 * e, H / 2 + 2 * e + {s.STAMP: -2.0, s.STAMP + 1: .9}.get(f, 0.), 1.0 + .020 * comp.ease(e), interp=cv2.INTER_CUBIC)
        return np.maximum(base, 0).astype(f32)
