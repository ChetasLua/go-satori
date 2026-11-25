"""Kid Goku's head at any angle: a 3-D head, features authored on its surface, hair as 3-D blades.  Public door: chars/kid.py
    H = KHead().pose(cx, cy, s, yaw, pitch, roll)      s = px per unit (half the head's width); yaw > 0 turns to the viewer's right, pitch > 0 looks up, roll > 0 clockwise
    draw_head(cel, H, expr, pal, light=(-.6, .7, .5), wind=(wx, wy, amount), f=0, parts='bf')
Head units: x right, y up, z to the viewer; half width 1, chin -1.02, eyes about -.10, skull top 1.30."""
import numpy as np, math, cv2, cairo
from db.ink import Cel, spl, resample, noise1
from db.kit import contours, outline
from db.paint import rgb, f32

Y_ = [-1.02, -.99, -.93, -.84, -.70, -.52, -.34, -.16, .06, .30, .58, .84, 1.06, 1.20, 1.28, 1.31]
A_ = [.10, .27, .46, .63, .80, .925, .985, .99, .965, .955, .93, .84, .65, .45, .25, .06]
F_ = [.60, .70, .76, .80, .825, .84, .845, .835, .83, .85, .87, .82, .64, .44, .25, .06]
B_ = [.02, .05, .10, .18, .36, .62, .84, .96, 1.02, 1.05, 1.03, .90, .68, .46, .26, .06]
EYE = (.405, -.11)

PAL = dict(
    day=dict(skin='#f9cfa4', skin_sh='#d8905f', skin_dk='#b4633f', blush='#ee8a66', line='#2a1512', lineskin='#7a3320', white='#fffaf0', white_sh='#cdb7b6', iris='#1b1014', iris2='#5b3322', glint='#ffffff',
             mouth='#7c1f28', mouth_dk='#4d0f19', tongue='#e67b76', teeth='#fff7ea', teeth_sh='#d8c2b8',
             hair='#221c26', hair_sh='#110d14', hair_hi='#463e55', hair_ln='#0a070c', fur='#8a5530', fur_sh='#5c3320', fur_hi='#b97c47',
             gi='#f47b20', gi_sh='#bf4a1c', gi_dk='#8f3018', gi_hi='#ffa046', blue='#23407c', blue_sh='#152854', blue_dk='#0e1b3c', blue_hi='#3f63a8', rim='#fff1c9'),
)
PAL['dawn'] = dict(PAL['day'], skin='#fbc893', skin_sh='#c9786a', skin_dk='#9b4f55', blush='#ef7f5c', white='#fff4dc', white_sh='#c9a3a6', hair='#2a1a22', hair_sh='#160d16', hair_hi='#5a343a', fur='#93552f', fur_sh='#5a2e2a', fur_hi='#d98a48',
                   gi='#f7862b', gi_sh='#b4482f', gi_dk='#84303a', gi_hi='#ffb35a', blue='#2c3f76', blue_sh='#1d2450', blue_dk='#14173a', blue_hi='#5a5f98', rim='#ffe3a0', line='#2c1218', lineskin='#86342a', teeth='#fff1da')
PAL['sky'] = dict(PAL['day'], skin='#fbd3a8', skin_sh='#cf8f7c', skin_dk='#a06266', hair_hi='#4a5578', gi_sh='#c2502f', blue_hi='#4b78c4', rim='#e8f6ff')


def _sm(y, tab, n=241):
    yy = np.linspace(-1.02, 1.31, n); v = np.interp(yy, y, tab); k = np.hanning(15); k /= k.sum()
    return yy, np.convolve(np.r_[np.full(7, v[0]), v, np.full(7, v[-1])], k, 'valid')


class KHead:
    p = 2.2
    def __init__(self):
        self.yy, self.A = _sm(Y_, A_); _, self.F = _sm(Y_, F_); _, self.B = _sm(Y_, B_); self.pose(0, 0, 1)

    def pose(self, cx, cy, s, yaw=0., pitch=0., roll=0., persp=.085, R=None):
        self.cx, self.cy, self.s, self.yaw, self.pitch, self.roll, self.persp = cx, cy, s, yaw, pitch, roll, persp
        if R is None:
            a, b = yaw, pitch
            Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]); Rx = np.array([[1, 0, 0], [0, math.cos(b), math.sin(b)], [0, -math.sin(b), math.cos(b)]])
            c, n = math.cos(-roll), math.sin(-roll); R = np.array([[c, -n, 0], [n, c, 0], [0, 0, 1]]) @ Rx @ Ry
        self.R = np.asarray(R, float); self.off = np.zeros(2)
        # angles as seen (for features that need a side)
        self.fwd = self.R @ np.array([0, 0, 1.]); self.sq = (1., 1.)
        return self

    def squash(self, sx, sy): self.sq = (sx, sy); return self

    def P3(self, Q):
        q = np.asarray(Q, float) @ self.R.T; f = 1 / (1 - np.clip(q[..., 2] * self.persp, -.6, .6))
        return np.stack([self.cx + q[..., 0] * f * self.s * self.sq[0], self.cy - q[..., 1] * f * self.s * self.sq[1]], -1)

    def depth(self, Q): return (np.asarray(Q, float) @ self.R.T)[..., 2]
    def k(self, Q): return 1 / (1 - np.clip(self.depth(Q) * self.persp, -.6, .6))

    def bump(self, x, y):
        g = lambda x0, y0, sx, sy, k: k * np.exp(-((x - x0) / sx) ** 2 - ((y - y0) / sy) ** 2)
        a = np.abs(x)
        return g(0, -.385, .10, .075, .135) + g(0, -.62, .26, .11, .055) - g(0, -.50, .12, .05, .02) + g(0, -.93, .25, .10, .065) + g(0, .24, .55, .18, .035) - g(0, -.14, .22, .14, .045) + (.050 * np.exp(-((y + .47) / .20) ** 2 - ((a - .56) / .26) ** 2) - .045 * np.exp(-((y + .06) / .20) ** 2 - ((a - .46) / .24) ** 2))

    def z(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float); a = np.interp(y, self.yy, self.A); f = np.interp(y, self.yy, self.F)
        return f * (1 - np.clip(np.abs(x) / a, 0, .999) ** self.p) ** (1 / self.p) + self.bump(x, y)

    def P(self, xy, dz=0.):
        xy = np.asarray(xy, float); return self.P3(np.concatenate([xy, (self.z(xy[..., 0], xy[..., 1]) + dz)[..., None]], -1))

    def surf(self, x, y, dz=0.): return np.array([x, y, float(self.z(x, y)) + dz])

    def facing(self, x, y):
        e = 2e-2; n = np.array([-(self.z(x + e, y) - self.z(x - e, y)) / (2 * e), -(self.z(x, y + e) - self.z(x, y - e)) / (2 * e), 1.]); n /= np.linalg.norm(n); return float((self.R @ n)[2])

    def shell(self, ph, y, lift=0.):
        """points of the closed head: azimuth ph (0 front, + right), height y"""
        A, F, B = (np.interp(y, self.yy, t) for t in (self.A, self.F, self.B)); sx, cz = np.sin(ph), np.cos(ph); r = (np.abs(sx) ** self.p + np.abs(cz) ** self.p) ** (-1 / self.p)
        X = (A + lift) * sx * r; Z = np.where(cz >= 0, F + lift, B + lift) * cz * r; Z = Z + np.clip(cz, 0, 1) ** .5 * self.bump(X, y)
        if lift == 0: y = np.where(np.asarray(y) < -.5, np.maximum(y, np.interp(Z, [-.35, -.2, 0., .2, .4, .56, .70], [-.55, -.64, -.78, -.89, -.965, -1.0, -1.02])), y)
        return np.stack([X, y + lift * np.clip((y - .5) / .8, 0, 1), Z], -1)

    def mesh(self, ny=80, nf=96):
        y = np.linspace(-1.02, 1.31, ny)[:, None] * np.ones((1, nf)); ph = np.ones((ny, 1)) * np.linspace(-math.pi, math.pi, nf)[None]; return self.shell(ph, y)

    def sil(self, M, front=False, extra=()):
        """silhouette contour of mesh quads (front=True: only quads that face us)"""
        k = max(.35, min(1., 800. / (self.s * 3.0))); p = self.P3(M); allp = np.vstack([p.reshape(-1, 2)] + [np.asarray(e) for e in extra]); lo = allp.min(0) - 6; hi = allp.max(0) + 6; w, h = (np.ceil((hi - lo) * k)).astype(int) + 2; m = np.zeros((h, w), np.uint8)
        q = (p - lo) * k; quad = np.stack([q[:-1, :-1], q[:-1, 1:], q[1:, 1:], q[1:, :-1]], 2).reshape(-1, 4, 2)
        if front:
            e1, e2 = quad[:, 2] - quad[:, 0], quad[:, 3] - quad[:, 1]; cr = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]; quad = quad[cr < 0]
        for c in np.round(quad * 16).astype(np.int32): cv2.fillConvexPoly(m, c, 255, cv2.LINE_8, 4)
        for e in extra: cv2.fillPoly(m, [np.round((np.asarray(e) - lo) * k * 16).astype(np.int32)], 255, cv2.LINE_8, 4)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)); cs = contours(m / 255., min_len=12)
        if not cs: return np.zeros((0, 2))
        c = max(cs, key=len); return resample(c / k + lo, max(1.2, min(3., self.s * .02)))

    def ear(self, s):
        out = [(-.02, -.43), (.04, -.53), (.15, -.53), (.25, -.40), (.335, -.18), (.345, -.01), (.28, .11), (.16, .13), (.04, .05), (-.04, -.05)]
        inn = [(.07, .015), (.16, .055), (.245, .02), (.265, -.09), (.20, -.27), (.12, -.37)]; tr = [(.06, -.13), (.14, -.10), (.165, -.17), (.11, -.27), (.10, -.34)]
        Q = lambda L: self.P3(np.array([(s * (.935 + u * .93), v, -.02 - u * .62) for u, v in L]))
        return spl(Q(out), n=7), spl(Q(inn), n=7), spl(Q(tr), n=7), float(self.depth(np.array([s * 1.05, -.2, -.1])))


# ------------------------------------------------------------------------------------------------ hair
def _L(name, pts, w, tag='', flex=1., belly=.22, tipk=.95): return dict(name=name, pts=np.array(pts, float), w=w, tag=tag, flex=flex, belly=belly, tipk=tipk)


# our kid: x > 0 is the viewer's right.  Two tall blades and three fans to the viewer's left, three fans to the right, a short asymmetric fringe, a duck-tail at the back.
LOCKS = [
    _L('top', [(.02, .92, -.12), (-.10, 1.42, -.10), (-.34, 1.92, -.10), (-.74, 2.30, -.14), (-1.16, 2.46, -.18)], .58, 'up', .7, .10, .90),
    _L('top2', [(.34, .98, -.20), (.40, 1.34, -.34), (.30, 1.72, -.56), (.04, 1.98, -.80)], .36, 'up', .7, .10),
    _L('L1', [(-.50, .74, -.10), (-1.02, 1.08, -.14), (-1.56, 1.30, -.20), (-2.10, 1.28, -.26), (-2.46, 1.08, -.28)], .50, 'side', .8, .14, .90),
    _L('L2', [(-.66, .34, -.20), (-1.14, .52, -.28), (-1.64, .58, -.36), (-2.08, .40, -.40), (-2.30, .20, -.42)], .40, 'side', .9, .16, .92),
    _L('L3', [(-.70, .0, -.34), (-1.10, .04, -.46), (-1.50, -.04, -.54), (-1.84, -.28, -.58)], .30, 'side', 1., .16),
    _L('R1', [(.52, .72, -.14), (1.00, .94, -.20), (1.48, .98, -.26), (1.92, .86, -.30), (2.20, .66, -.32)], .44, 'side', .85, .14, .92),
    _L('R2', [(.68, .32, -.24), (1.16, .42, -.32), (1.62, .36, -.40), (2.02, .12, -.44)], .36, 'side', .95, .16),
    _L('R3', [(.70, -.02, -.36), (1.08, -.02, -.48), (1.42, -.14, -.56), (1.70, -.40, -.60)], .27, 'side', 1., .16),
    # back of the head
    _L('B1', [(-.10, .95, -.55), (-.16, 1.26, -1.02), (-.30, 1.36, -1.46), (-.50, 1.26, -1.86)], .46, 'back', .8, .14),
    _L('B2', [(.34, .62, -.75), (.48, .74, -1.22), (.62, .66, -1.66), (.78, .42, -2.02)], .44, 'back', .85, .16),
    _L('B3', [(-.40, .48, -.80), (-.58, .52, -1.26), (-.80, .40, -1.66), (-1.00, .14, -1.96)], .42, 'back', .85, .16),
    _L('B4', [(.06, .10, -.90), (.08, .06, -1.32), (.14, -.12, -1.70), (.22, -.42, -1.96)], .42, 'back', .9, .16),
    _L('B5', [(-.46, -.14, -.66), (-.60, -.28, -.92), (-.72, -.54, -1.08), (-.78, -.88, -1.12)], .36, 'back', 1., .16),
    _L('B6', [(.48, -.16, -.66), (.62, -.32, -.90), (.72, -.58, -1.04), (.74, -.90, -1.06)], .34, 'back', 1., .16),
    _L('B7', [(.0, -.30, -.66), (.0, -.52, -.90), (-.04, -.78, -.98), (-.12, -1.06, -.94)], .34, 'back', 1., .16),
    # fringe
    _L('F1', [(-.02, 1.04, .48), (-.08, .80, .82), (-.16, .50, .97), (-.20, .06, 1.0)], .34, 'fr', .5, .16),
    _L('F2', [(-.40, 1.0, .40), (-.54, .72, .70), (-.60, .46, .80), (-.56, .12, .85)], .32, 'fr', .5, .16),
    _L('F3', [(.28, 1.04, .44), (.50, .84, .74), (.68, .58, .78), (.74, .16, .74)], .38, 'fr', .5, .18),
    _L('F4', [(-.70, .78, .28), (-.86, .52, .46), (-.94, .26, .48), (-.92, -.08, .48)], .25, 'fr', .5, .16),
    _L('F5', [(.66, .86, .28), (.86, .60, .42), (.95, .32, .42), (.97, -.02, .44)], .26, 'fr', .5, .16),
    _L('F6', [(.12, 1.06, .48), (.20, .86, .78), (.24, .66, .90), (.36, .36, .92)], .20, 'fr', .5, .16),
    _L('S1', [(-.92, .30, .16), (-.975, .10, .20), (-.98, -.04, .22), (-.95, -.22, .25)], .13, 'fr', .4, .1),
    _L('S2', [(.92, .30, .16), (.975, .10, .20), (.98, -.04, .22), (.95, -.24, .25)], .13, 'fr', .4, .1),
]


def _lock(H, L, wind, f, n=26):
    """screen spine, radius, depth"""
    P = L['pts'].copy(); back_ = float(np.clip((H.fwd[1] - .22) / .34, 0, 1)) if H.fwd[2] > 0 else 0.
    if back_ > 0 and L['name'] in ('B4', 'B5', 'B6', 'B7', 'L3', 'R3'): kf_ = 1 - (.92 if L['name'] in ('B5', 'B6', 'B7') else .55) * back_; P = P[0] + (P - P[0]) * kf_; P[:, 1] += (1 - kf_) * .30 * np.linspace(0, 1, len(P))
    ln = np.hypot(*(np.diff(P[:, :2], axis=0).T)).sum() + abs(np.diff(P[:, 2])).sum() * .5; t = np.linspace(0, 1, len(P))[:, None]
    wx, wy, am = wind
    if am:
        d = np.array([wx, -wy, 0.]) @ H.R                                   # screen wind (x right, y down) into head space
        sd_ = (P[-1] - P[0]) @ H.R.T; dt_ = float((sd_[0] * wx - sd_[1] * wy) / (np.linalg.norm(sd_) * math.hypot(wx, wy) + 1e-6)); fx_ = max(.22, 1 + .85 * min(0., dt_))            # a blade pointing into the wind is pressed back, it never folds over the face
        ph = (hash_(L['name']) % 628) / 100.; osc = .16 * math.sin(f * .62 + ph) + .08 * math.sin(f * 1.13 + ph * 2.1)
        P = P + d * (am * (1 + .5 * osc) * L['flex'] * ln * .34 * fx_) * t ** 1.8 + np.array([0, 1, 0]) * (osc * am * .16 * L['flex'] * ln) * t ** 2
    S = spl(P, n=max(3, n // (len(P) - 1))) if False else _spl3(P, n)
    tt = np.linspace(0, 1, n); w = L['w'] * (1 - tt ** 1.55) ** L['tipk'] * (1 + L['belly'] * np.sin(np.pi * np.clip(tt * 1.5, 0, 1)) - .10 * tt)
    q = H.P3(S); r = np.maximum(w * H.s * H.k(S) * .5 * (H.sq[0] + H.sq[1]), .05)
    L3 = np.linalg.norm(np.diff(S, axis=0), axis=1).sum() * H.s; e = q[-1] - q[0]; l2 = np.hypot(*e); want = .62 * L3 if L['tag'] != 'fr' else .30 * L3
    if l2 < want:
        c = H.P3(np.array([[0, .10, 0]]))[0]; u = q[0] - c; down = H.P3(np.array([[0, -.5, 0]]))[0] - c; dirv = u / (np.hypot(*u) + .25 * H.s) + (.55 if L['tag'] == 'back' else -.55 if L['tag'] == 'up' else .05) * down / (np.hypot(*down) + 1e-6) + .4 * e / (H.s + 1e-6); dirv = dirv / (np.hypot(*dirv) + 1e-6)
        q = q + dirv * (want - l2) * (tt[:, None] ** 1.35)
    sg = np.r_[0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]; ss = sg / (sg[-1] + 1e-6); mixt = np.clip(.35 * tt + .65 * ss, 0, 1)            # taper along what is SEEN, so a foreshortened blade still ends in a point
    k_ = H.k(S); wv = L['w'] * (1 - mixt ** 1.55) ** L['tipk'] * (1 + L['belly'] * np.sin(np.pi * np.clip(mixt * 1.5, 0, 1)) - .10 * mixt); r = np.maximum(wv * H.s * k_ * .5 * (H.sq[0] + H.sq[1]), .05)
    return q, r, float(H.depth(S[n // 3])), l2 < want


def hash_(s):
    h = 2166136261
    for c in s: h = ((h ^ ord(c)) * 16777619) & 0x7fffffff
    return h


def _spl3(P, n):
    """catmull-rom in 3-D, n points"""
    m = len(P) - 1; u = np.linspace(0, m - 1e-6, n); i = u.astype(int); t = (u - i)[:, None]; E = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]]); p0, p1, p2, p3 = E[i], E[i + 1], E[i + 2], E[i + 3]
    return .5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


def _edges(q, r):
    T = np.gradient(q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; return q + N * r[:, None], q - N * r[:, None], N


def _hulls(q, r):
    """swept discs as convex pieces: correct even when the blade points at us"""
    out = []; a = np.linspace(0, 2 * math.pi, 15)[:-1]; c = np.c_[np.cos(a), np.sin(a)]
    for i in range(len(q) - 1):
        pts = np.vstack([q[i] + c * r[i], q[i + 1] + c * r[i + 1]]).astype(np.float32); out.append(cv2.convexHull(pts)[:, 0].astype(float))
    return out


def blade(cel, q, r, pal, key, w, seed=0, edge=True, sep=True, fringe=False, gleam=True):
    """one hair blade: flat, its shaded half, a hard gleam, a pen edge"""
    hl = _hulls(q, r); n = len(q); e0 = int(n * .16) if not fringe else int(n * .42)
    if edge and w > 0:
        for P in _hulls(q, r + w * .5)[e0:]: cel.fill(P, pal['hair_ln'])
    for P in hl: cel.fill(P, pal['hair'])
    A, B, N = _edges(q, r); ln = np.hypot(*np.diff(q, axis=0).T).sum()
    if ln < r[0] * .9 or r[0] < 2.2: return
    side = 1 if (N[n // 2] @ key) > 0 else -1; lit, drk = (A, B) if side > 0 else (B, A); tt = np.linspace(0, 1, n)
    cel.ctx.save(); cel._path([np.vstack([A, B[::-1]])]); cel.ctx.clip()
    i0, i1 = int(n * (.12 if not fringe else .40)), n - 1
    k = -.02 + .42 * np.sin(np.clip((tt - .06) * 3.3, 0, 3.14)) - .10 * np.sin(tt * 21 + seed) * (tt > .2) * (tt < .8)
    mid = q + N * (r * (-side) * k)[:, None]
    cel.fill(np.vstack([mid[i0:i1], drk[i0:i1][::-1] + (drk - q)[i0:i1][::-1] * .4]), pal['hair_sh'])
    if not fringe and gleam:
        for (a_, b_, v, wd) in ((.16, .56, .66, .085), (.24, .50, .40, .055), (.34, .72, .80, .045)):
            a, b = int(n * a_), int(n * b_); u = np.linspace(0, 1, b - a); c = q[a:b] + (lit - q)[a:b] * (v + .06 * np.sin(u * 3))[:, None]; nn = (lit - q)[a:b] * (wd * np.sin(np.pi * u ** .62))[:, None]; cel.fill(np.vstack([c + nn, (c - nn)[::-1]]), pal['hair_hi'])
    cel.ctx.restore()
    if sep and not fringe and gleam: cel.line((lit * .985 + q * .015)[int(n * .18):int(n * .9)], max(.7, w * .5), pal.get('hair_sep', pal['hair_hi']), taper=(.35, .45), press=.2, seed=seed, tremor=.1)
    if fringe and gleam and r[0] > 11:
        cel.ctx.save(); cel._path([np.vstack([A, B[::-1]])]); cel.ctx.clip()
        for (a_, b_, v, wd) in ((.30, .70, .62, .085), (.42, .84, .30, .05)):
            a, b = int(n * a_), int(n * b_); u = np.linspace(0, 1, b - a); c = q[a:b] + (lit - q)[a:b] * (v + .06 * np.sin(u * 3))[:, None]; nn = (lit - q)[a:b] * (wd * np.sin(np.pi * u ** .62))[:, None]; cel.fill(np.vstack([c + nn, (c - nn)[::-1]]), pal['hair_hi'])
        cel.ctx.restore(); cel.line((drk * .97 + q * .03)[int(n * .34):int(n * .95)], max(.8, w * .7), pal['hair_ln'], taper=(.5, .3), press=.2, seed=seed + 5, tremor=.1)
    if fringe and gleam and r[0] > 3: cel.line((lit * .9 + q * .1)[int(n * .30):int(n * .86)], max(.7, w * .55), pal['hair_hi'], taper=(.5, .4), press=.2, seed=seed, tremor=.1)


def hairline(ph):
    a = np.abs((ph + math.pi) % (2 * math.pi) - math.pi); return np.interp(a, [0, .55, 1.0, 1.32, 1.55, 1.85, 2.3, math.pi], [.74, .70, .52, .18, .10, -.28, -.62, -.76])


def cap_mesh(H, lift=.055, ny=40, nf=120):
    ph = np.linspace(-math.pi, math.pi, nf)[None] * np.ones((ny, 1)); u = np.linspace(0, 1, ny)[:, None]; y = hairline(ph) * (1 - u) + 1.31 * u; return H.shell(ph, np.minimum(y, 1.309), lift)


# ------------------------------------------------------------------------------------------------ expressions
def X(**k):
    d = dict(open=1., wide=1., look=(0., 0.), brow=0., tilt=0., browL=0., mouth='smile', mo=.5, mw=1., squint=0., joy=0., blush=.5, pupil=1., skew=0.)
    d.update(k); return d


EXPR = dict(
    calm=X(mouth='closed', brow=.0, open=.92), smile=X(mouth='smile', brow=.04), grin=X(mouth='grin', brow=.02, tilt=.10, squint=.22, mw=1.15), laugh=X(mouth='laugh', mo=1., brow=.12, squint=.10, blush=.8),
    joy=X(mouth='laugh', mo=1.05, joy=1., brow=.14, blush=.9, mw=1.08), wonder=X(mouth='o', mo=.45, brow=.16, open=1.10, pupil=1.12, tilt=-.04), curious=X(mouth='closed', brow=.10, browL=.12, open=1.06, mw=.7, skew=.2, pupil=1.08),
    whoop=X(mouth='laugh', mo=1.28, mw=1.05, brow=.10, tilt=.10, squint=.16, blush=.7), blink=X(mouth='smile', open=.0, brow=-.03), set=X(mouth='grin', brow=-.10, tilt=.30, squint=.16), sad=X(mouth='closed', brow=.04, tilt=-.22, open=.86, mw=.8))


def mixx(a, b, t):
    """expression in-between (mouth kind switches half way)"""
    a = EXPR[a] if isinstance(a, str) else a; b = EXPR[b] if isinstance(b, str) else b; o = {}
    for k, v in a.items():
        o[k] = (v if t < .5 else b[k]) if isinstance(v, str) else tuple(np.array(v) * (1 - t) + np.array(b[k]) * t) if isinstance(v, tuple) else v * (1 - t) + b[k] * t
    return o


EYE_U = [(-.215, -.10), (-.205, .08), (-.130, .235), (.0, .285), (.125, .255), (.205, .120), (.222, -.06)]
EYE_D = [(.222, -.06), (.205, -.20), (.130, -.305), (.0, -.335), (-.130, -.300), (-.200, -.205), (-.215, -.10)]


def _mirror(P, s): P = np.array(P, float); P[:, 0] *= s; return P


def eye(cel, H, s, E, pal, key, w, lod):
    ex, ey = EYE; op = E['open']; sq = E['squint']; tint = E.get('eye_light', None)
    if H.facing(s * ex, ey) < .10: return
    if E['joy'] > .5 or op < .14:                                    # closed: joy arcs up, a blink sags down
        up = 1 if E["joy"] > .5 else -1; c = -.04 if up > 0 else E.get("lid", -.16); A = [(-.235, c - .06 * up), (-.14, c + .10 * up), (.0, c + .155 * up), (.14, c + .10 * up), (.235, c - .07 * up)]
        cel.line(spl(H.P(_mirror(np.array(A) + [ex, ey], s)), n=8), w * 1.55, pal['line'], taper=(.28, .28), press=.12, seed=4)
        if up > 0 and lod > 1: cel.line(spl(H.P(_mirror(np.array([(.16, -.12), (.24, -.17), (.31, -.17)]) + [ex, ey], s)), n=6), w * .55, pal['lineskin'], taper=(.2, .5), seed=5)
        return
    U = np.array(EYE_U); D = np.array(EYE_D); U[:, 1] = np.where(U[:, 1] > -.02, -.02 + (U[:, 1] + .02) * op, U[:, 1]); D[:, 1] = D[:, 1] + sq * .30 * np.clip(1 - np.abs(D[:, 0]) / .24, 0, 1) ** .5 * (D[:, 1] < -.15)
    U[:, 0] *= E['wide']; D[:, 0] *= E['wide']; tl = E['tilt']
    if tl > 0: U[:, 1] -= np.clip(tl, 0, .4) * np.clip((.16 - U[:, 0]) / .38, 0, 1) * .55          # inner lid pressed by the brow
    o = np.array([ex, ey]); Uo, Do = spl(H.P(_mirror(U + o, s)), n=7), spl(H.P(_mirror(D + o, s)), n=7); W = np.vstack([Uo, Do[1:]])
    cel.fill(W, pal['white']); cel.clip(W)
    sh = np.vstack([Uo, spl(H.P(_mirror(U * [1, 1] + o + np.array([0, -.085]), s)), n=7)[::-1]]); cel.fill(sh, pal['white_sh'])
    lx, ly = E['look']; pw, ph = .092 * E['pupil'], .162 * E['pupil']; c = np.array([s * (ex - .055) + lx * .12, ey - .035 + ly * .12]); a = np.linspace(0, 2 * math.pi, 29)[:-1]
    I = H.P(np.c_[c[0] + pw * np.cos(a), c[1] + ph * np.sin(a)]); cel.fill(I, pal['iris'])
    if lod > 1:
        b = a[(np.sin(a) < -.10)]; J = H.P(np.vstack([np.c_[c[0] + pw * .74 * np.cos(b), c[1] + ph * .80 * np.sin(b)], [(c[0] + pw * .55, c[1] - ph * .06), (c[0], c[1] - ph * .30), (c[0] - pw * .55, c[1] - ph * .06)]])); cel.fill(J, pal['iris2'])
    if lod > 0:
        g = c + np.array([.034 * np.sign(key[0] + 1e-6), .070]) * E['pupil']; R = (.040, .052); cel.fill(H.P(np.c_[g[0] + R[0] * np.cos(a), g[1] + R[1] * np.sin(a)]), pal['glint'])
        if lod > 1: g = c - np.array([.036 * np.sign(key[0] + 1e-6), .082]); cel.fill(H.P(np.c_[g[0] + .016 * np.cos(a), g[1] + .020 * np.sin(a)]), pal['glint'])
    cel.unclip()
    n = len(Uo)
    cel.line(Do, w * .50, pal['line'] if lod < 2 else pal['lineskin'], taper=(.45, .30), press=.15, seed=7)
    cel.line(np.vstack([Do[-max(3, len(Do) // 4):], Uo]), w * 1.5, pal['line'], taper=(.30, .22), press=.10, seed=6, wfun=lambda t: .72 + .5 * np.sin(np.pi * t))
    if lod > 1 and op > .8:
        cel.line(spl(H.P(_mirror(np.array([(-.10, .315), (.02, .350), (.13, .318)]) + o + np.array([0, (op - 1) * .25]), s)), n=6), w * .45, pal['lineskin'], taper=(.4, .4), seed=8)


def brow(cel, H, s, E, pal, w):
    ex, ey = EYE; up = E['brow'] + (E['browL'] if s < 0 else 0.); tl = E['tilt']
    if H.facing(s * ex, .25) < .02: return
    top = [(-.225, .355), (-.10, .455), (.07, .495), (.22, .430), (.315, .295)]; bot = [(.315, .295), (.20, .360), (.06, .395), (-.08, .345), (-.195, .255)]
    P = np.array(top + bot); P[:, 1] += up + tl * (P[:, 0] * .75 - .06) + (.02 if E['joy'] > .5 else 0.); P = _mirror(P + [ex, ey], s)
    Q = spl(H.P(P, .015), True, corners=(0, 4, 5, 9), n=7); cel.fill(Q, pal['hair'])


def mouth(cel, H, E, pal, w, lod, key):
    if H.facing(0, -.62) < .02: return
    k = E['mouth']; mw = E['mw']; sk = E['skew']; xl = {q: next((v for v in (.40, .34, .28, .22, .16, .10, .05, .02) if H.facing(q * v, -.60) > .16), .01) / .40 for q in (-1, 1)}; P = lambda L: H.P(np.array([(x * mw * min(1., xl[1 if x > 0 else -1] * (1.15 if xl[1 if x > 0 else -1] < 1 else 1)) + sk * .05, y + sk * x * .10) for x, y in L]))
    if k in ('laugh', 'o', 'whoop'):
        mo = E['mo']
        if k == 'o': U = [(-.135, -.655), (-.08, -.600), (0, -.588), (.08, -.600), (.135, -.655)]; D = [(.135, -.655), (.125, -.655 - .14 * mo), (.06, -.67 - .25 * mo), (-.06, -.67 - .25 * mo), (-.125, -.655 - .14 * mo), (-.135, -.655)]
        else: U = [(-.36, -.530), (-.22, -.560), (0, -.572), (.22, -.560), (.36, -.530)]; D = [(.36, -.530), (.315, -.565 - .13 * mo), (.17, -.58 - .275 * mo), (0, -.585 - .31 * mo), (-.17, -.58 - .275 * mo), (-.315, -.565 - .13 * mo), (-.36, -.530)]
        Us, Ds = spl(P(U), n=7), spl(P(D), n=7); M = np.vstack([Us, Ds[1:]]); cel.fill(M, pal['mouth']); cel.clip(M)
        yb = min(y for _, y in D)
        if k != 'o':
            cel.fill(np.vstack([Us, spl(P([(.30, -.615), (.15, -.640), (0, -.648), (-.15, -.640), (-.30, -.615)]), n=6)]), pal['teeth'])
            if lod > 1: cel.fill(np.vstack([spl(P([(-.34, -.575), (-.15, -.600), (0, -.606), (.15, -.600), (.34, -.575)]), n=6), spl(P([(.30, -.615), (.15, -.640), (0, -.648), (-.15, -.640), (-.30, -.615)]), n=6)]), pal['teeth_sh'])
        T = spl(P([(-.215, yb + .02), (-.165, yb + .10 * mo), (-.05, yb + .135 * mo), (.0, yb + .118 * mo), (.05, yb + .135 * mo), (.165, yb + .10 * mo), (.215, yb + .02), (.0, yb - .10)]), True, n=6); cel.fill(T, pal['tongue'])
        if lod > 1: cel.line(P([(0, yb + .10 * mo), (0, yb + .04)]), w * .5, pal['mouth'], taper=(.2, .6), seed=3)
        cel.unclip()
        cel.line(Us, w * 1.05, pal['line'], taper=(.10, .10), press=.15, seed=11); cel.line(Ds, w * .80, pal['line'], taper=(.16, .16), press=.15, seed=12)
        if k != 'o':
            for q in (-1, 1): cel.line(spl(P([(q * .345, -.500), (q * .378, -.527), (q * .385, -.570)]), n=6), w * .8, pal['lineskin'], taper=(.3, .5), seed=13)
            if lod > 1: cel.line(spl(P([(-.085, yb - .075), (0, yb - .092), (.085, yb - .075)]), n=6), w * .45, pal['lineskin'], taper=(.4, .4), seed=14)
    elif k == 'grin':
        U = [(-.36, -.520), (-.22, -.570), (0, -.588), (.22, -.565), (.38, -.500)]; D = [(.38, -.500), (.29, -.630), (.12, -.705), (-.10, -.710), (-.27, -.640), (-.36, -.520)]; Us, Ds = spl(P(U), n=7), spl(P(D), n=7); M = np.vstack([Us, Ds[1:]])
        cel.fill(M, pal['teeth']); cel.clip(M); cel.fill(np.vstack([spl(P([(-.30, -.640), (-.10, -.680), (.12, -.675), (.32, -.600)]), n=6), Ds]), pal['teeth_sh']); cel.unclip()
        cel.line(Us, w * 1.0, pal['line'], taper=(.08, .08), press=.15, seed=11); cel.line(Ds, w * .72, pal['line'], taper=(.12, .12), press=.15, seed=12)
        for q, yy in ((-1, -.520), (1, -.500)): cel.line(spl(P([(q * .355, yy + .04), (q * .40, yy + .01), (q * .405, yy - .045)]), n=6), w * .75, pal['lineskin'], taper=(.3, .5), seed=13)
    elif k == 'smile':
        cel.line(spl(P([(-.25, -.570), (-.12, -.638), (.02, -.655), (.16, -.632), (.27, -.552)]), n=8), w * 1.0, pal['line'], taper=(.2, .2), seed=11)
        if lod > 1: cel.line(P([(-.05, -.735), (.06, -.737)]), w * .4, pal['lineskin'], taper=(.4, .4), seed=14)
    else:
        cel.line(spl(P([(-.135, -.628), (-.03, -.642), (.07, -.640), (.15, -.622)]), n=8), w * .95, pal['line'], taper=(.25, .25), seed=11)
        if lod > 1: cel.line(P([(-.04, -.725), (.05, -.727)]), w * .4, pal['lineskin'], taper=(.4, .4), seed=14)


# ------------------------------------------------------------------------------------------------ the head
def lw(s): return float(np.interp(s, [8, 20, 45, 100, 200, 340], [.8, 1.25, 1.9, 3.0, 4.4, 5.8]))


def draw_head(cel, H, expr='smile', pal='day', light=(-.6, .7, .5), wind=(0., 0., 0.), f=0, parts='bnf', seed=1, w=None, neck=True, zcut=None, neckpx=None):
    pal = PAL[pal] if isinstance(pal, str) else pal; E = dict(EXPR[expr]) if isinstance(expr, str) else dict(expr)
    s = H.s * (H.sq[0] + H.sq[1]) * .5; w = lw(s) if w is None else w; lod = 0 if s < 16 else 1 if s < 46 else 2
    key = np.array([light[0], -light[1]], float); key /= np.hypot(*key) + 1e-6; sd = 1 if light[0] > 0 else -1                   # light side
    LK = []
    for L in LOCKS:
        q, r, d, fs = _lock(H, L, wind, f); LK.append((d, dict(L, fs=fs), q, r))
    front = H.fwd[2] > -.25
    cap = cap_mesh(H)
    def is_front(L, d): return L['tag'] == 'fr' or d > .22
    back_ = lambda d: zcut is not None and d < zcut
    if 'k' in parts:
        for d, L, q, r in sorted(LK, key=lambda t: t[0]):
            if L['tag'] != 'fr' and back_(d): blade(cel, q, r, pal, key, w * .7, seed=hash_(L['name']) & 0xffff, gleam=not L['fs'])
    if 'b' in parts:
        for d, L, q, r in sorted(LK, key=lambda t: t[0]):
            if is_front(L, d) or back_(d): continue
            blade(cel, q, r, pal, key, w * .7, seed=hash_(L['name']) & 0xffff, gleam=not L['fs'])
        C = H.sil(cap)
        if len(C): cel.fill(C, pal['hair'])
    def neck_():
        c0, c1 = H.P3(np.array([(0, -.70, -.12), (0, -1.38, -.16)])); ax = c1 - c0; ax = ax / (np.hypot(*ax) + 1e-6); nn = np.array([-ax[1], ax[0]]) * H.s; Q = spl(np.array([c0 + nn * .36, c0 * .55 + c1 * .45 + nn * .335, c0 * .2 + c1 * .8 + nn * .37, c1 + nn * .52, c1 - nn * .52, c0 * .2 + c1 * .8 - nn * .37, c0 * .55 + c1 * .45 - nn * .335, c0 - nn * .36]), True, corners=(3, 4), n=6); cel.fill(Q, pal['skin_sh']); n_ = len(Q); cel.line(Q[:int(n_ * .36)], w * .9, pal['line'], taper=(.2, .1), seed=41); cel.line(Q[int(n_ * .58):int(n_ * .95)], w * .9, pal['line'], taper=(.1, .2), seed=42)
    if 'n' in parts and neck and H.fwd[2] > -.1: neck_()
    if 'f' in parts:
        ears = {q: H.ear(q) for q in (-1, 1)}
        def draw_ear(q):
            o, i, t, d = ears[q]; cel.fill(o, pal['skin'] if q == sd else pal['skin_sh']); cel.clip(o)
            if q == sd: cel.fill(np.vstack([i, (o[len(o) // 6:len(o) // 2] * .4 + i.mean(0) * .6)[::-1]]), pal['skin_sh'])
            else: cel.fill(np.vstack([i, (i * .5 + i.mean(0) * .5)[::-1]]), pal['skin_dk'])
            cel.unclip(); cel.line(o, w * .95, pal['line'], taper=(.12, .12), press=.2, seed=21)
            if lod > 0: cel.line(i, w * .55, pal['lineskin'], taper=(.2, .4), seed=22)
            if lod > 1: cel.line(t, w * .50, pal['lineskin'], taper=(.2, .5), seed=23)
        for q in (-1, 1):
            if ears[q][3] < .15: draw_ear(q)
        if front or True:
            FC = H.sil(H.mesh()); cel.fill(FC, pal['skin']); cel.clip(FC)
            # side plane away from the light: one hard angular shape from temple, under the cheekbone, to the jaw
            o = -sd; pl = [(1.02, .75), (.70, .62), (.735, .30), (.655, .02), (.70, -.22), (.80, -.36), (.665, -.55), (.56, -.74), (.36, -.93), (.16, -1.03), (.5, -1.2), (1.3, -1.0), (1.4, .2)]
            if light[2] > -.2: cel.fill(spl(H.P(np.array([(o * x, y) for x, y in pl if abs(x) <= 1.1] )), True, corners=range(10), n=3), pal['skin_sh'])
            else: cel.fill(FC, pal['skin_sh']); cel.fill(H.P(np.array([(sd * x, y) for x, y in [(.62, .6), (.80, .2), (.78, -.2), (.70, -.5), (.50, -.8), (.80, -.8), (1.0, -.3), (1.0, .5)]])), pal['skin'])
            # fringe shadow: the blades again, dropped away from the light
            off = np.array([-key[0] * .05, .105]) * H.s
            for d, L, q, r in LK:
                if L['tag'] == 'fr' and L['name'][0] == 'F':
                    for Pq in _hulls(q + off, r * .98): cel.fill(Pq, pal['skin_sh'])
            hl = [(x, float(hairline(np.array(x * 1.25))) - .04) for x in np.linspace(-1, 1, 13)]; cel.fill(np.vstack([H.P(np.array(hl)), H.P3(np.array([(1.2, 1.4, .2), (-1.2, 1.4, .2)]))]), pal['skin_sh'])
            if E['blush'] > 0 and lod > 0:
                for q in (-1, 1):
                    if H.facing(q * .62, -.42) < .05: continue
                    for j in range(3): x0 = q * (.53 + .075 * j); cel.line(H.P(np.array([(x0 - .035, -.475), (x0 + .025, -.395)])), w * .62, pal['blush'], taper=(.3, .3), seed=30 + j, alpha=min(1., E['blush'] * 1.2))
            un = float((H.R @ np.array([0, -1., .25]))[2])
            low = un > .12 and H.fwd[2] > 0
            if low:                                                        # seen from below: the chin is an edge, its underside a soft shadow plane that becomes the neck
                jw = np.array([(.90, -.42, -.02), (.82, -.66, .14), (.62, -.87, .36), (.34, -.985, .56), (0, -1.02, .665)]); jl = np.vstack([jw, (jw * [-1, 1, 1])[::-1][1:]]); J = spl(H.P3(jl), n=7); bk = H.P3(np.array([(-.9, -1.6, -.6), (.9, -1.6, -.6)])); cel.fill(np.vstack([J, bk]), pal['skin_sh'])
            cel.unclip(); outline(cel, FC, w, pal['line'], 31, press=.22, tremor=.15)
            if low:
                cel.line(J[int(len(J) * .30):int(len(J) * .70)], w * .62, pal['lineskin'], taper=(.45, .45), press=.2, seed=33)
            # features
            for q in (-1, 1): eye(cel, H, q, E, pal, key, w, lod)
            t = np.array([.0, -.40]); nx = -sd
            if H.facing(0, -.35) > .05:
                cel.line(spl(H.P(np.array([(nx * .012, -.330), (nx * .045, -.385), (nx * .02, -.418), (-nx * .03, -.425)])), n=6), w * .72, pal['lineskin'], taper=(.5, .25), seed=24)
                if lod > 1: cel.fill(H.P(np.array([(nx * .015, -.36), (nx * .085, -.425), (nx * .02, -.43)])), pal['skin_sh'])
            mouth(cel, H, E, pal, w, lod, key)
            for q in (-1, 1): brow(cel, H, q, E, pal, w)
        if 'n' in parts and neck and H.fwd[2] <= -.1: neck_()
        for q in (-1, 1):
            if ears[q][3] >= .15: draw_ear(q)
        C = H.sil(cap, front=True)
        if len(C): cel.fill(C, pal['hair'])
        if len(C) and lod > 0:                                            # one gleam that rides the crown
            g = []; a0 = math.atan2(sd * .9, .6)
            for u in np.linspace(-.65, .65, 14): g.append((a0 + u * .9, .86 + .10 * math.cos(u * 3) - .05 * math.sin(u * 17)))
            for u in np.linspace(.65, -.65, 14): g.append((a0 + u * .9, .74 + .08 * math.cos(u * 3) - .09 * abs(math.sin(u * 11))))
            G3 = H.shell(np.array([a for a, _ in g]), np.array([y for _, y in g]), .06); vis = H.depth(G3)
            if vis.mean() > .1: cel.clip(C); cel.fill(H.P3(G3), pal['hair_hi']); cel.unclip()
        for d, L, q, r in sorted(LK, key=lambda t: t[0]):
            if not is_front(L, d) or (L['tag'] == 'fr' and d < -.30): continue
            blade(cel, q, r, pal, key, w * (.7 if L['tag'] != 'fr' else .6), seed=hash_(L['name']) & 0xffff, fringe=L['tag'] == 'fr', gleam=not L['fs'])
    return cel
