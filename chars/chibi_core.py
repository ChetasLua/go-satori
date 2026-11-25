"""Chibi drawing core (lane CHIBI): registered cels (flat / shadow / mask / ink), union-inked masses, 2-D nodes, 3-D chibi head map,
3-D hair blades, tubes, hands, feet.  Units: head half-width = 1, y DOWN, z toward the viewer.  Used by chibi_story / chibi_stick / chibi_feast."""
import numpy as np, cv2, math, cairo
from db import W, H
from db.paint import rgb, f32, smooth, fbm
from db.ink import Cel, spl, resample, noise1, paste
pi = math.pi
A = np.array


def mix(a, b, t):
    a = rgb(a) if isinstance(a, str) else np.asarray(a, float); b = rgb(b) if isinstance(b, str) else np.asarray(b, float)
    c = np.clip(a * (1 - t) + b * t, 0, 1); return '#%02x%02x%02x' % tuple(int(v * 255 + .5) for v in c)


def cbox(v, k):
    ker = np.hanning(k + 2)[1:-1]; ker /= ker.sum(); p = k // 2 + 1
    return np.convolve(np.r_[v[-p:], v, v[:p]], ker, 'same')[p:-p]


def E(t): t = min(1., max(0., t)); return t * t * (3 - 2 * t)
def Eo(t, p=2.5): return 1 - (1 - min(1., max(0., t))) ** p
def Ei(t, p=2.5): return min(1., max(0., t)) ** p
def back(t, s=1.9):
    t = min(1., max(0., t)) - 1; return 1 + t * t * ((s + 1) * t + s)


def keys(f, K):
    """linear-with-ease scalar/array keys [(frame, value[, 'e'|'l'|'o'|'i'])]"""
    if f <= K[0][0]: return np.asarray(K[0][1], float)
    for a, b in zip(K[:-1], K[1:]):
        if f < b[0]:
            t = (f - a[0]) / float(b[0] - a[0]); m = b[2] if len(b) > 2 else 'e'
            t = dict(e=E, l=lambda q: q, o=Eo, i=Ei, b=back)[m](t)
            return np.asarray(a[1], float) * (1 - t) + np.asarray(b[1], float) * t
    return np.asarray(K[-1][1], float)


# ================================================================ geometry
class N:
    """2-D node: local -> px"""
    def __init__(self, M=None): self.M = np.eye(3) if M is None else M
    def at(self, x=0., y=0., rot=0., sx=1., sy=None):
        sy = sx if sy is None else sy; c, s = math.cos(rot), math.sin(rot)
        return N(self.M @ np.array([[c * sx, -s * sy, x], [s * sx, c * sy, y], [0, 0, 1.]]))
    def __call__(self, P): P = np.asarray(P, float); return P @ self.M[:2, :2].T + self.M[:2, 2]
    @property
    def s(self): return math.sqrt(abs(np.linalg.det(self.M[:2, :2])))
    def inv(self, p): Mi = np.linalg.inv(self.M); p = np.asarray(p, float); return p @ Mi[:2, :2].T + Mi[:2, 2]


def ell(cx, cy, rx, ry=None, rot=0., n=40, sq=2.0):
    ry = rx if ry is None else ry; a = np.linspace(0, 2 * pi, n, endpoint=False); c, s = np.cos(a), np.sin(a)
    x = np.sign(c) * np.abs(c) ** (2 / sq) * rx; y = np.sign(s) * np.abs(s) ** (2 / sq) * ry
    cr, sr = math.cos(rot), math.sin(rot); return np.c_[cx + x * cr - y * sr, cy + x * sr + y * cr]


def bez(a, c1, c2, b, n=12):
    a, c1, c2, b = [np.asarray(v, float) for v in (a, c1, c2, b)]; u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 3 * a + 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u * u * c2 + u ** 3 * b


def blade(a, tip, b, bend=0., bulge=.10, n=13):
    """hair clump in 2-D or 3-D: a -> tip -> b.  bend = vector the whole clump is swept by"""
    a, tip, b = [np.asarray(v, float) for v in (a, tip, b)]; bend = np.zeros_like(a) if np.isscalar(bend) else np.asarray(bend, float)
    ax = tip - (a + b) / 2; L = np.linalg.norm(ax); sd = a - b; sd = sd / (np.linalg.norm(sd) + 1e-9)
    P = bez(a, a + ax * .38 + sd * L * bulge + bend, a + (tip - a) * .74 + sd * L * bulge * .32 + bend * .8, tip, n)
    Q = bez(tip, b + (tip - b) * .74 - sd * L * bulge * .32 + bend * .8, b + ax * .38 - sd * L * bulge + bend, b, n)
    return np.vstack([P, Q[1:]])


def tube(P, Wd, n=8, cap=.0, capn=7, cap0=0.):
    """closed outline of a limb: centre controls P, half widths Wd (evenly spaced), cap = rounded end bulge (0..1)"""
    P = np.asarray(P, float); C = spl(P, n=n) if len(P) > 2 else np.linspace(P[0], P[-1], 14); d = np.r_[0, np.cumsum(np.hypot(*np.diff(C, axis=0).T))]; t = np.linspace(0, d[-1], 34)
    C = np.c_[np.interp(t, d, C[:, 0]), np.interp(t, d, C[:, 1])]
    T = np.gradient(C, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; Nn = np.c_[-T[:, 1], T[:, 0]]
    s = np.linspace(0, 1, len(C)); w = np.interp(s, np.linspace(0, 1, len(Wd)), Wd)
    k = np.hanning(9); k /= k.sum(); w = np.convolve(np.r_[[w[0]] * 4, w, [w[-1]] * 4], k, 'same')[4:-4]
    def side(sg):
        Q = C + sg * Nn * w[:, None]; keep = np.r_[True, np.sum(np.diff(Q, axis=0) * T[1:], 1) > 0]; Q = Q[keep]
        m = np.hanning(5); m /= m.sum(); e = np.vstack([[Q[0]] * 2, Q, [Q[-1]] * 2]); return np.c_[np.convolve(e[:, 0], m, 'valid'), np.convolve(e[:, 1], m, 'valid')]
    L, R = side(1), side(-1); out = [L]
    if cap > 0:
        a = np.linspace(pi / 2, -pi / 2, capn + 2)[1:-1]; out.append(C[-1] + w[-1] * (np.sin(a)[:, None] * Nn[-1] + cap * np.cos(a)[:, None] * T[-1]))
    out.append(R[::-1])
    if cap0 > 0:
        a = np.linspace(-pi / 2, pi / 2, capn + 2)[1:-1]; out.append(C[0] + w[0] * (np.sin(a)[:, None] * Nn[0] - cap0 * np.cos(a)[:, None] * T[0]))
    return np.vstack(out)


def star(cx, cy, r, rot=-pi / 2, k=.45, n=5):
    a = np.arange(2 * n) * pi / n + rot; R = np.where(np.arange(2 * n) % 2 == 0, r, r * k); return np.c_[cx + np.cos(a) * R, cy + np.sin(a) * R]


def ik(a, b, l1, l2, bend=1.):
    """two-bone: joint a, target b -> elbow/knee, end (clamped)"""
    a, b = A(a, float), A(b, float); d = b - a; L = np.hypot(*d) + 1e-9; Lc = min(L, (l1 + l2) * .999); u = d / L; b = a + u * Lc
    c = np.clip((l1 * l1 + Lc * Lc - l2 * l2) / (2 * l1 * Lc), -1, 1); s = math.sqrt(1 - c * c); n = A([-u[1], u[0]]) * bend
    return a + (u * c + n * s) * l1, b


# ================================================================ surface
class D:
    """registered cels: flat, shadow colour, shadow mask, ink.  part() knocks the ink under it, so painter's order just works."""
    def __init__(self, x0=0, y0=0, w=W, h=H, ss=2, lw=4.6, line='#4a2a1e', var=.22, soft=1.0, light=(-.55, -.83)):
        self.x0, self.y0, self.w, self.h, self.ss = int(x0), int(y0), int(w), int(h), ss
        mk = lambda: Cel(self.w, self.h, self.x0, self.y0, ss)
        self.fl, self.sc, self.mk, self.ln = mk(), mk(), mk(), mk()
        self.lw, self.line, self.var, self.soft, self.light = lw, line, var, soft, light

    def knock(self, P):
        c = self.ln.ctx; c.save(); c.set_operator(cairo.OPERATOR_DEST_OUT); self.ln._path(P); c.set_source_rgba(0, 0, 0, 1); c.fill(); c.restore()

    def part(self, P, col, sh=None, shade=(), hi=(), w=1.0, ink=None, gaps=(), seed=0, press=.12):
        """closed shape(s) P (px). shade = shadow polygons, hi = [(poly, hex)] flat marks, gaps = [(t0,t1)] of the contour without ink"""
        Ps = P if isinstance(P, list) else [np.asarray(P, float)]; sh = sh or mix(col, '#6a2f4a', .25)
        self.knock(Ps)
        self.fl.fill(Ps, col); self.sc.fill(Ps, sh); self.mk.fill(Ps, '#000000')
        for q in shade: self.mk.shade(q, '#ffffff', within=Ps)
        for q, c in hi: self.fl.shade(q, c, within=Ps); self.sc.shade(q, mix(c, sh, .35), within=Ps)
        if w > 0:
            for i, q in enumerate(Ps): self.contour(q, w, ink, gaps, seed + i, press)
        return Ps[0]

    def shade(self, q, within=None): self.mk.shade(q, '#ffffff', within=within); return self

    def detail(self, q, col, within=None, sh=None, alpha=1.):
        self.fl.shade(q, col, within=within, alpha=alpha); self.sc.shade(q, sh or mix(col, '#5a2a4a', .28), within=within, alpha=alpha); return self

    def stroke(self, P, w=.6, ink=None, taper=(.2, .25), seed=0, press=.12, tmin=.18, within=None, alpha=1.):
        P = np.asarray(P, float)
        if within is not None: self.ln.clip(within)
        self.ln.line(P, w * self.lw, ink or self.line, taper=taper, press=press, seed=seed, tremor=.25, tmin=tmin, alpha=alpha)
        if within is not None: self.ln.unclip()
        return self

    def contour(self, P, w=1., ink=None, gaps=(), seed=0, press=.12):
        P = np.asarray(P, float); P = resample(np.vstack([P, P[:1]]), 1.8)[:-1]; n = len(P)
        if n < 8: return
        area = .5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]); sg = 1. if area > 0 else -1.
        T = np.roll(P, -2, 0) - np.roll(P, 2, 0); T /= np.hypot(*T.T)[:, None] + 1e-9; Nn = np.c_[T[:, 1], -T[:, 0]] * sg
        k = -(Nn[:, 0] * self.light[0] + Nn[:, 1] * self.light[1])
        wl = 1 - self.var + 2 * self.var * smooth(-.7, .95, cbox(k, 9))
        d = 4; a = np.arctan2(P[:, 1] - np.roll(P, d, 0)[:, 1], P[:, 0] - np.roll(P, d, 0)[:, 0]); b = np.arctan2(np.roll(P, -d, 0)[:, 1] - P[:, 1], np.roll(P, -d, 0)[:, 0] - P[:, 0])
        turn = np.angle(np.exp(1j * (b - a))) * sg
        wl = wl * (1 - .35 * np.clip(cbox(np.clip(turn, 0, 1.6), 11), 0, 1)) * (1 + .5 * np.clip(cbox(np.clip(-turn, 0, 1.6), 9), 0, 1))
        r = np.random.default_rng(abs(int(seed)) * 7 + 11); nz = noise1(n, 40, r); nz *= np.minimum(1, np.minimum(np.arange(n), n - 1 - np.arange(n)) / 30.)
        Wd = np.maximum(.5, w * self.lw * wl * (1 + press * np.clip(nz, -1.3, 1.3)))
        on = np.ones(n, bool); t = np.arange(n) / n
        for g0, g1 in gaps: on &= ~(((t >= g0) & (t <= g1)) if g0 <= g1 else ((t >= g0) | (t <= g1)))
        if gaps:
            di = cv2.distanceTransform(np.r_[on, on, on].astype(np.uint8)[:, None], cv2.DIST_L1, 3)[n:2 * n, 0]; Wd = Wd * (.25 + .75 * np.minimum(1, di * 1.8 / (4 * self.lw)) ** .7)
        c = self.ln.ctx; c.save(); c.set_line_cap(cairo.LINE_CAP_ROUND); col = ink or self.line; col = rgb(col) if isinstance(col, str) else col; c.set_source_rgba(*map(float, col), 1)
        for i in range(n):
            j = (i + 1) % n
            if on[i] and on[j]: c.set_line_width(float(.5 * (Wd[i] + Wd[j]))); c.move_to(*P[i]); c.line_to(*P[j]); c.stroke()
        c.restore()

    # ------------------------------------------------ union of polygons -> smooth outline(s)
    def union(self, polys, clips=None, sm=5):
        """polys: list of (k,2) px; clips: same-length list of clip polygon or None -> list of outlines of the union"""
        allp = np.vstack(polys); pad = 6; x0, y0 = np.floor(allp.min(0) - pad).astype(int); x1, y1 = np.ceil(allp.max(0) + pad).astype(int)
        x0, y0 = max(x0, self.x0 - 900), max(y0, self.y0 - 900); x1, y1 = min(x1, self.x0 + self.w + 900), min(y1, self.y0 + self.h + 900)
        c = Cel(max(8, x1 - x0), max(8, y1 - y0), x0, y0, 2)
        for i, p in enumerate(polys):
            cp = clips[i] if clips is not None else None
            if cp is not None: c.clip(cp)
            c.fill(p, '#ffffff')
            if cp is not None: c.unclip()
        c.surf.flush(); m = np.ndarray((c.h * 2, c.surf.get_stride() // 4, 4), np.uint8, c.surf.get_data())[:, :c.w * 2, 3]
        cs, _ = cv2.findContours((m > 110).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE); out = []
        ker = np.hanning(sm + 2)[1:-1]; ker /= ker.sum(); p = sm
        for q in cs:
            q = q[:, 0].astype(float)
            if len(q) < 60: continue
            e = np.vstack([q[-p:], q, q[:p]]); s = np.stack([np.convolve(e[:, i], ker, 'same') for i in (0, 1)], -1)[p:-p]
            out.append(s[::2] / 2. + [x0 + .25, y0 + .25])
        out.sort(key=lambda q: -abs(.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])))
        return out

    # ------------------------------------------------ result
    def rgba(self, outer=None, grain=0.):
        """outer=(hex, amount, px): the silhouette's own line leans to that colour; grain = crayon tooth in flats and line"""
        a = self.fl.rgba(); s = self.sc.rgba(); m = self.mk.rgba()[..., 0]; l = self.ln.rgba()
        if outer is not None:
            AA = np.maximum(a[..., 3], l[..., 3]); dt = cv2.distanceTransform((AA > .5).astype(np.uint8), cv2.DIST_L2, 5)
            k = (np.clip(1.35 - dt / outer[2], 0, 1) * outer[1])[..., None]; l = np.concatenate([l[..., :3] * (1 - k) + rgb(outer[0]) * l[..., 3:] * k, l[..., 3:]], -1)
        m = cv2.GaussianBlur(m, (0, 0), self.soft)[..., None]; al = np.maximum(a[..., 3:4], s[..., 3:4])
        c = a[..., :3] * (1 - m) + s[..., :3] * m
        if grain > 0:
            g = tooth(self.h, self.w, self.x0, self.y0); c = c * (1 + g[..., None] * grain); l = l * np.clip(1 - np.maximum(0, g[..., None] - .7) * grain * 5, .55, 1)
        out = np.concatenate([c, al], -1); return out * (1 - l[..., 3:4]) + l

    def layer(self, **k):
        f = self.rgba(**k)
        if (self.x0, self.y0, self.w, self.h) == (0, 0, W, H): return f
        out = np.zeros((H, W, 4), f32); paste(out, f, self.x0, self.y0); return out


_tooth = {}
def tooth(h, w, x0=0, y0=0):
    """fixed crayon / paper tooth, about -1..1"""
    if 't' not in _tooth:
        r = np.random.default_rng(4); bl = lambda q, s: cv2.GaussianBlur(np.pad(q, 32, 'wrap'), (0, 0), s)[32:-32, 32:-32]
        g = bl(r.standard_normal((1024, 1024)).astype(f32), 1.1); g /= g.std(); b = bl(r.standard_normal((1024, 1024)).astype(f32), 7); b /= b.std(); _tooth['t'] = np.clip(g * .65 + b * .5, -2, 2) * .6
    t = _tooth['t']; ys = (np.arange(h) + y0) % 1024; xs = (np.arange(w) + x0) % 1024; return t[np.ix_(ys, xs)]


# ================================================================ head map
FR = [(0, -.94), (.44, -.89), (.79, -.66), (.965, -.28), (1.0, .06), (.965, .36), (.82, .62), (.53, .80), (.22, .865), (0, .875)]            # front, right half
SD = [(0, -.94), (.42, -.88), (.75, -.64), (.90, -.30), (.915, -.04), (.955, .22), (.93, .48), (.72, .73), (.36, .845), (.05, .83), (-.32, .72), (-.68, .50), (-.92, .18), (-1.0, -.2), (-.87, -.6), (-.50, -.87)]


def _front():
    R = A(FR, float); return np.vstack([R, R[::-1][1:-1] * [-1, 1]])


def _resamp(P, n=64):
    Q = spl(P, True, n=10); Q = np.vstack([Q, Q[:1]]); d = np.r_[0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]; t = np.linspace(0, d[-1], n, endpoint=False)
    return np.c_[np.interp(t, d, Q[:, 0]), np.interp(t, d, Q[:, 1])]


class Hd:
    """chibi head: node at the head CENTRE.  yaw > 0 faces viewer's right, pitch > 0 looks down.  lag = (dx, dy) hair drag in head units."""
    RX, RY, RZ = 1.0, .95, .93

    def __init__(self, node, yaw=0., pitch=0., lag=(0., 0.), t=0., wob=0., cheek=0., front=None, side=None):
        self.n, self.yaw, self.pitch, self.lag, self.t, self.wob, self.cheek = node, yaw, pitch, lag, t, wob, cheek
        self.cy, self.sy, self.cp, self.sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        self.front, self.side, self.cheeky, self.cheekb = front, side, .38, 1.

    def rot(self, Q):
        Q = np.atleast_2d(np.asarray(Q, float)); x = Q[:, 0] * self.cy + Q[:, 2] * self.sy; z = -Q[:, 0] * self.sy + Q[:, 2] * self.cy
        return np.c_[x, Q[:, 1] * self.cp + z * self.sp, z * self.cp - Q[:, 1] * self.sp]

    def P3(self, Q, loose=0.):
        R = self.rot(Q); p = R[:, :2]
        if loose:
            Q = np.atleast_2d(Q); r = np.sqrt(Q[:, 0] ** 2 + (Q[:, 1] + .1) ** 2 + Q[:, 2] ** 2); w = np.clip((r - .9) / .9, 0, 1.4) ** 1.2 * loose
            an = np.arctan2(p[:, 1], p[:, 0]); wb = self.wob * np.sin(self.t * .6 + an * 2.1)
            p = p + w[:, None] * (A(self.lag) + wb[:, None] * A([.5, 1.]))
        return self.n(p)

    def dome(self, P):
        P = np.atleast_2d(np.asarray(P, float)); return self.RZ * np.sqrt(np.clip(1 - (P[:, 0] / 1.04) ** 2 - (P[:, 1] / .99) ** 2, 0, 1))

    def F(self, P, dz=0.):
        P = np.atleast_2d(np.asarray(P, float)); return self.P3(np.c_[P, self.dome(P) + dz])

    def z(self, x, y=0.): return self.rot([[x, y, self.dome([[x, y]])[0]]])[0, 2]

    def loc(self, P, dz=0.):
        """front-surface points -> head-local 2-D (before the node)"""
        P = np.atleast_2d(np.asarray(P, float)); return self.rot(np.c_[P, self.dome(P) + dz])[:, :2]

    def outline(self, grow=0., raw=False):
        t = min(1., abs(self.sy) ** 1.15); f = _resamp(self.front if self.front is not None else _front(), 72); s = _resamp(A(self.side if self.side is not None else SD, float), 72)
        if self.yaw < 0: s = s * [-1, 1]; s = np.roll(s[::-1], 1, 0)
        P = f * (1 - t) + s * t
        if self.cheek: P = P + np.c_[np.sign(P[:, 0]) * self.cheek * np.exp(-((P[:, 1] - self.cheeky) / np.where(P[:, 1] < self.cheeky, .30, .42)) ** 2) * np.abs(P[:, 0]) ** .6 * np.where(P[:, 0] * (self.yaw if abs(self.yaw) > .2 else 1e9 * np.sign(P[:, 0])) >= 0, 1., self.cheekb), np.zeros(len(P))]
        P = P * [1, .5 * (1 + self.cp)] + [0, self.sp * .10]
        if grow: P = P * (1 + grow)
        return P if raw else self.n(P)


# ================================================================ hair
def lockset(L):
    """[(a, tip, b, bend, bulge, tag)] helper: a, tip, b 3-vectors"""
    return [dict(a=A(q[0], float), tip=A(q[1], float), b=A(q[2], float), bend=A(q[3], float) if len(q) > 3 and q[3] is not None else np.zeros(3), bulge=q[4] if len(q) > 4 else .10, tag=q[5] if len(q) > 5 else '') for q in L]


def blade2(a, tip, b, bend, bulge=.10, n=13, wmin=.26):
    """2-D clump with a guaranteed root width: both sides swept by `bend`, belly near the root, pointed tip"""
    a, tip, b = [np.asarray(v, float) for v in (a, tip, b)]; m = (a + b) / 2; ax = tip - m; L = np.hypot(*ax) + 1e-9; u = ax / L; nr = A([-u[1], u[0]])
    wa, wb = float((a - m) @ nr), float((b - m) @ nr); s = 1. if wa >= wb else -1.; hw = max(abs(wa - wb) / 2, wmin * L * .5)
    a2 = m + nr * s * hw + u * float((a - m) @ u) * .6; b2 = m - nr * s * hw + u * float((b - m) @ u) * .6; bend = np.asarray(bend, float)
    P = bez(a2, a2 + ax * .38 + nr * s * L * bulge + bend, a2 + (tip - a2) * .74 + nr * s * L * bulge * .30 + bend * .8, tip, n)
    Q = bez(tip, b2 + (tip - b2) * .74 - nr * s * L * bulge * .30 + bend * .8, b2 + ax * .38 - nr * s * L * bulge + bend, b2, n)
    return np.vstack([P, Q[1:]])


def lock2(h, q, loose=1.):
    """head-local 2-D outline of a 3-D clump (13 points root a -> tip, 12 back to root b)"""
    R = h.rot(np.vstack([q['a'], q['tip'], q['b'], q['bend']])); a, tip, b, bd = R[:, :2]
    r = np.linalg.norm(q['tip']); w = np.clip((r - .9) / .9, 0, 1.4) ** 1.2 * loose; an = math.atan2(tip[1], tip[0])
    lag = A(h.lag) + h.wob * math.sin(h.t * .6 + an * 2.1) * A([.5, 1.]); return blade2(a, tip + lag * w, b, bd + lag * w * .12, q['bulge'])


def lock_poly(h, q, loose=1.): return h.n(lock2(h, q, loose))


def lock_z(h, q):
    return float(h.rot((q['a'] + q['b']) / 2 * .7 + q['tip'] * .3)[0, 2])


def cap_poly(h, line, top=-1.6, out=1.5):
    """polygon above a hairline (front coordinates, left -> right); clip it to the skull outline"""
    L = np.asarray(line, float); L3 = np.c_[L, h.dome(L) + .01]; R = h.rot(L3); k = R[:, 2] > -.06
    Q = R[k][:, :2]
    if len(Q) < 2: return None, None
    P = np.vstack([[(Q[0, 0] - .8, Q[0, 1] + .0)], Q, [(Q[-1, 0] + .8, Q[-1, 1] + .0)], [(out, top), (-out, top)]])
    if not k[0] and h.yaw > 0: P[0] = (-out, Q[0, 1])
    return h.n(P), h.n(Q)
