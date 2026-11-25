"""ki beam: axes (2-D path with widening radius, or true 3-D through a Camera3D), bodies with streams and licks, double helix, collars,
muzzle, impact, parted flow, clash, volleys, view from inside.  m = length along the beam measured in beam radii (perspective-true)."""
from .ki_core import *


class Axis2:
    """beam along a screen polyline P; R = radius px at start / end (linear in perspective) or callable(v)"""
    def __init__(self, P, r0, r1=None, tilt=.30, warp=None):
        self.p = Path(P, 6.); r1 = r0 if r1 is None else r1; n = 220; v = lin(n); self.tilt = tilt; self.warp = warp
        r = np.array([r0(q) for q in v]) if callable(r0) else r0 + (r1 - r0) * v
        s = v * self.p.L; dm = np.diff(s) / ((r[1:] + r[:-1]) / 2); self.mt = np.r_[0, np.cumsum(dm)]; self.st = s; self.rt = r; self.M = float(self.mt[-1]); self.m0 = 0.

    def r(self, m): return np.interp(m, self.mt, self.rt)
    def s(self, m): m = np.asarray(m, float); return np.interp(m, self.mt, self.st) + np.where(m > self.M, (m - self.M) * self.rt[-1], 0) + np.where(m < 0, m * self.rt[0], 0)

    def pt(self, m, n, h=0.):
        m = np.atleast_1d(np.asarray(m, float)); n = np.broadcast_to(np.asarray(n, float), m.shape)
        if self.warp is not None: n = self.warp(m, n)
        r = self.r(m); return self.p.at(self.s(m) + self.tilt * np.broadcast_to(np.asarray(h, float), m.shape) * r * self._tl(), n * r)

    def _tl(self): return 1.

    def msamp(self, a, b, px=16., kmin=8, kmax=90):
        """metric samples between a and b, even on screen"""
        s0, s1 = float(self.s(np.array([a]))[0]), float(self.s(np.array([b]))[0]); k = int(np.clip(abs(s1 - s0) / px, kmin, kmax)); s = np.linspace(s0, s1, k)
        return np.interp(s, self.st, self.mt) + np.where(s > self.st[-1], (s - self.st[-1]) / self.rt[-1], 0) + np.where(s < 0, s / self.rt[0], 0)


class Axis3:
    """beam from world A to B with radius R0 metres (R1 at the far end) through a Camera3D"""
    def __init__(self, cam, A, B, R0, R1=None, warp=None, near=.6):
        self.cam = cam; A, B = np.asarray(A, float), np.asarray(B, float); self.warp = warp; d = B - A; self.Lw = float(np.linalg.norm(d)); self.d = d / self.Lw; self.A = A; self.R0 = R0; self.R1 = R0 if R1 is None else R1
        self.M = self.Lw / R0; q = np.concatenate([np.linspace(0, 1, 400) ** 2.2, 1 - np.linspace(0, 1, 400) ** 2.2, np.linspace(0, 1, 200)]); q = np.unique(np.clip(q, 0, 1)); X = A + d * q[:, None]; px, z = cam.project(X); ok = z > near
        self.ok = ok.any(); q = q[ok]; px = px[ok]; z = z[ok]; self.mq = q * self.M
        if len(q) > 1: self.m0 = float(self.mq[0]); self.M1 = float(self.mq[-1]); self.sq = np.r_[0, np.cumsum(np.hypot(*np.diff(px, axis=0).T))]; self.rq = (self.R0 + (self.R1 - self.R0) * q) * cam.f / z
        f = cam.R[2]; mid = A + d * .5; v = cam.eye - (A + d * float(np.clip((cam.eye - A) @ self.d / self.Lw, .02, .98))); v = v - self.d * (v @ self.d); v = v / (np.linalg.norm(v) + 1e-9); self.h = v; self.n = np.cross(self.d, v)
        a, _ = cam.project(np.stack([X[len(X) // 2], X[len(X) // 2] + self.n * R0])); t = cam.project(A + d * np.array([[.5], [.52]]))[0]; T = t[1] - t[0]
        if T[0] * (a[1] - a[0])[1] - T[1] * (a[1] - a[0])[0] > 0: self.n = -self.n                    # same left-hand normal as Axis2

    def r(self, m): return np.interp(m, self.mq, self.rq)
    def pt(self, m, n, h=0.):
        m = np.clip(np.atleast_1d(np.asarray(m, float)), self.m0, self.M1 + 60); n = np.broadcast_to(np.asarray(n, float), m.shape); h = np.broadcast_to(np.asarray(h, float), m.shape)
        if self.warp is not None: n = self.warp(m, n)
        R_ = self.R0 + (self.R1 - self.R0) * np.clip(m / self.M, 0, 1); X = self.A + self.d * (m * self.R0)[:, None] + self.n * (n * R_)[:, None] + self.h * (h * R_)[:, None]; return self.cam.project(X)[0]

    def msamp(self, a, b, px=16., kmin=8, kmax=90):
        a, b = max(a, self.m0), min(b, self.M1 + 60); s0, s1 = np.interp([a, b], self.mq, self.sq); k = int(np.clip(abs(s1 - s0) / px, kmin, kmax)); s = np.linspace(s0, s1, k); m = np.interp(s, self.sq, self.mq)
        if b > self.M1: m = np.r_[m[:-1], b]
        return m


def _vis(P, pad=140): return (P[:, 0].max() > -pad) and (P[:, 0].min() < W + pad) and (P[:, 1].max() > -pad) and (P[:, 1].min() < H + pad)


def _cells(a, b, pitch, shift):
    return range(int(math.floor((a - shift) / pitch)) - 1, int(math.ceil((b - shift) / pitch)) + 1)


# ------------------------------------------------------------------ body
def body_field(ax, F, t, a, b, seed=1, w=1., lv=(1., .84, .60, .33), wob=.07, speed=22., licks=1., lick_len=(1.1, 3.0), lick_w=.20, rise=.72, root=.03, k=.10, dirn=1., taper_in=0., taper_out=0.):
    """main mass of a beam between metric a and b + licks peeling off both edges downstream"""
    m = ax.msamp(a, b, 14., 10, 130); raw = ax.warp; sw_ = 1.
    if raw is not None: sw_ = np.maximum(np.abs(raw(m, np.ones_like(m))), np.abs(raw(m, -np.ones_like(m)))); ax.warp = None
    P = ax.pt(m, 0.); ax.warp = raw; r = ax.r(m) * sw_; kk = float(np.clip(np.median(r) * k, 3, 46)); sh = speed * t
    ta = np.ones_like(m)
    if taper_in: ta *= np.clip((m - a) / taper_in, 0, 1) ** .6
    if taper_out: ta *= np.clip((b - m) / taper_out, 0, 1) ** .6
    w1 = 1 + wob * noise((m - sh) * .9, 1.3, t * 1.5, seed) + wob * .6 * noise((m - sh) * 2.1, 5.3, t * 2, seed); w2 = 1 + wob * 1.5 * noise((m - sh * 1.1) * 1.1, 9.3, t * 2, seed)
    F.rib(P, [r * w * q * ta * (w1 if i < 2 else w2) for i, q in enumerate(lv)], kk)
    if not licks: return F
    for sg in (1, -1):
        for row, (pit, sc) in enumerate(((1.35, 1.), (.62, .50), (.27, .24))):
            if row == 2 and r.max() < 130: continue
            for c in _cells(a, b, pit, sh):
                if h1(seed, sg, row, c, 1) > .88 * licks + .12 * (row == 0): continue
                z = bms(h1(seed, sg, row, c, 2), .22, .36) * sc; ln = (lick_len[0] + (lick_len[1] - lick_len[0]) * z) * (.8 + .4 * h1(seed, sg, row, c, 3)); m0 = (c + h1(seed, sg, row, c, 4)) * pit + sh
                if dirn < 0: ln = -ln
                if m0 < a - .2 or m0 > b - .1: continue
                life = .72 + .28 * math.sin(t * 9 + 6.3 * h1(seed, c, sg)); rr = float(ax.r(np.array([m0]))[0]); hb = rr * lick_w * (.45 + z) * w
                if hb < 1.1: continue
                v = lin(13); up = rise * z * life * (.7 + .6 * h1(seed, sg, row, c, 5)); ph = h1(seed, sg, row, c, 6) * 6.28
                n = sg * w * ((.80 + root) + up * (1 - (1 - v) ** 1.9) + .10 * z * np.sin(v * 5.5 + ph - t * 12) * v); mm = m0 + ln * v
                if taper_in and m0 < a + taper_in * .6: continue
                Q = ax.pt(mm, n)
                if not _vis(Q): continue
                L = float(np.hypot(*np.diff(Q, axis=0).T).sum())
                if L < 10: continue
                F.rib(Q, kwidths(13, L, hb, LV[:F.L]), kk * .7)
    return F


def streams(ax, t, a, b, seed, n, speed, nr=.8, ln=(1.5, 5.), w=(.012, .05), S=1., amp=.08, L=1, freq=1.1):
    """long thin shapes inside the flow -> Field(1)"""
    F = Field(L); per = max(6., (b - a)) * 1.0
    for i in range(n):
        l = ln[0] + (ln[1] - ln[0]) * h1(seed, i, 1) ** 1.6; sp = speed * (.8 + .5 * h1(seed, i, 2)); T = (b - a) + l * 2; x = t * sp / T + h1(seed, i, 3); u = x - math.floor(x); c = int(math.floor(x)); m0 = a - l + T * u; n0 = (h1(seed, i, c, 4) * 2 - 1) * nr; ph = 6.28 * h1(seed, i, 5)
        m = ax.msamp(max(a, m0), min(b, m0 + l), 22., 6, 26)
        if m[-1] - m[0] < .05: continue
        q = n0 + amp * np.sin(m * freq + ph); Q = ax.pt(m, q)
        if not _vis(Q, 60): continue
        r = ax.r(m); hw = r * (w[0] + (w[1] - w[0]) * h1(seed, i, 6) ** 2.2) * S; K = len(m); v = lin(K); e = np.where(v < .35, (v / .35) ** .7, np.clip((1 - v) / .65, 0, 1) ** 1.0)
        F.rib(Q, [hw * e * q_ for q_ in ((1., .5, .25)[:L])], 2.)
    return F


def swells(ax, t, a, b, seed, n=6, speed=14., rad=.62, w=.17, pitch=7., L=1):
    """broad S-shaped bands that roll round the beam: give the flat mass a turning volume"""
    F = Field(L)
    for i in range(n):
        ph = TAU * i / n + 6.28 * h1(seed, i, 1) * .3
        x = speed * t / pitch; c0 = math.floor(x)
        for c in _cells(a, b, pitch, speed * t):
            m0 = (c + .5 * h1(seed, i, c, 2)) * pitch + speed * t; l = pitch * (.75 + .5 * h1(seed, i, c, 3)); A, B_ = max(a, m0), min(b, m0 + l)
            if B_ - A < .4: continue
            m = ax.msamp(A, B_, 20., 8, 34); v = (m - m0) / l; n_ = rad * (.6 + .6 * h1(seed, i, c, 4)) * np.sin(TAU * (m - speed * t) / (pitch * 1.6) + ph); Q = ax.pt(m, n_)
            if not _vis(Q, 60): continue
            hw = ax.r(m) * w * (.35 + .9 * h1(seed, i, c, 5)) * np.sin(math.pi * np.clip(v, 0, 1)) ** .8; F.rib(Q, [hw * q for q in (1., .5)[:L]], 6.)
    return F


def collar(ax, m, rad, w, f, seed, kind='gold', tilt=None, gaps=.30, cut=0, lead=0.):
    """one shed ring round the beam -> (far half layer, near half layer): one torn thick-thin stroke, lit inside.  rad in beam radii"""
    P_ = pal(kind); out = []; r = float(ax.r(np.array([m]))[0])
    for half in (-1, 1):
        F = Field(3); a0 = -math.pi / 2 if half > 0 else math.pi / 2; th = np.linspace(a0 - .05, a0 + math.pi + .05, 64); wob = 1 + .035 * noise(np.cos(th) * 2.2, np.sin(th) * 2.2, 0, seed) + .012 * noise(np.cos(th) * 7, np.sin(th) * 7, 0, seed + 1)
        Q = ax.pt(np.full(64, m) + lead * np.abs(np.sin(th)), rad * wob * np.sin(th), rad * wob * np.cos(th)); tk = np.clip(1.05 - gaps * 1.2 + 1.1 * noise(np.cos(th) * 1.9, np.sin(th) * 1.9, 3., seed + 2), 0, 1.6); hw = w * r * tk * (.30 + .70 * np.abs(np.cos(th)) ** .8) * (1. if half > 0 else .75) - .2; F.rib(Q, [hw, hw * .62 - .2, hw * .28 - .6], 2.)
        for j in range(9):                                    # sparks thrown off the ring, downstream
            t_ = a0 + math.pi * h1(seed, half, j, 4); q = ax.pt(np.array([m, m + .18 + .5 * h1(seed, j, 5)]), rad * np.sin(t_) * np.array([1, 1.04]), rad * np.cos(t_)); ww = w * r * (.15 + .4 * h1(seed, j, 6)); F.rib(np.linspace(q[0], q[1], 5), [np.array([.2, ww, ww * .7, ww * .3, -.2]) * z for z in (1, .5, .2)], 1.)
        out.append(F.cel((P_['c'][0], P_['c'][2], P_['c'][3]) if half > 0 else (P_['c'][0], P_['c'][1], P_['c'][2]), lw=(max(1., r * .004), 0, 0), ink=(P_['ink'], None, None), seed=seed))
    return out


def beam(ax, f, kind='gold', head=None, tail=None, seed=1, w=1., speed=24., licks=1., veil=.5, dens=1., collars=0, power=1., step=1, ink=True, core=1., back=None, light=1.):
    """complete beam -> (layer, light).  head = metric position reached (None = whole axis), tail = where it starts"""
    t = (f // step) * step / 24.; P_ = pal(kind); C2 = P_['c']; a = getattr(ax, 'm0', 0.) if tail is None else tail; b = min(ax.M, head) if head is not None else ax.M
    if b - a < .05: return None, None
    r = ax.r(np.linspace(a, b, 9)); rm = float(np.median(r)); rmx = float(r.max()); layers = []
    if veil:
        V = body_field(ax, Field(2), t * .9, a, b, seed + 7, w * 1.24, (1., .9), .10, speed * .9, licks * .8, (1.6, 4.2), .24, .75); layers.append(V.cel((C2[0], C2[1]), alpha=(veil * .55, veil * .8)))
    cols = [(collar(ax, (b - a) * ((c + .5) / collars + (t * speed * .5 / max(1., (b - a))) % 1.) % (b - a) + a, 1.35 + .35 * h1(seed, c), .06 + .04 * h1(seed, c, 2), f, seed + c * 7, kind)) for c in range(collars)]
    layers += [c[0] for c in cols]
    M = body_field(ax, Field(4), t, a, b, seed, w, speed=speed, licks=licks); lw = (float(np.clip(rm * .018, 1.2, 5.)), 0, float(np.clip(rm * .009, .9, 2.4)), 0) if ink else None; layers.append(M.cel(C2, lw=lw, ink=(P_['ink'], None, P_['c'][0], None), seed=seed + int(t * 12)))
    m1, m2, m3 = M.mask(0), M.mask(2), M.mask(3); n = int(46 * dens)
    def ins(F, col, m, al=1.): o = F.cel((col,)); o *= (m * al)[..., None]; layers.append(o)
    ins(swells(ax, t, a, b, seed + 3, 5, speed * .7, .62, .16 * w), C2[1], m2 * (1 - m3), .9)
    ins(streams(ax, t, a, b, seed + 4, n, speed * 1.5, .82 * w, (1.2, 4.5), (.010, .040)), C2[0], m1 * (1 - m3), .85)
    ins(streams(ax, t, a, b, seed + 5, int(n * .7), speed * 1.8, .86 * w, (1.5, 6.), (.010, .045)), C2[3], m1 * (1 - m3), 1.)
    ins(streams(ax, t, a, b, seed + 6, int(n * .5), speed * 2., .30 * w, (1.0, 5.), (.008, .030)), C2[2], m3, .9 * core)
    layers += [c[1] for c in cols]; lay = stack(*layers); li = bloom(lay, .55, (6, 24, 80), (.25, .25, .28), P_['glow'], .7 * light * power); return lay, li * (1 - .5 * M.mask(0))[..., None]


# ------------------------------------------------------------------ double helix
def _leaf(ax, m0, half, rad, hwr, seed, front, tipl, K=34):
    """one half turn of a strand: a flame blade that wraps across the beam from the lower edge to the upper one and leaves as a lick"""
    v = np.linspace(-.12, 1.0 + tipl, K); c = np.clip(v, 0, 1); n = -rad * np.cos(math.pi * c); h = rad * np.sin(math.pi * c); o = np.maximum(v - 1, 0); n = n + (.30 + .25 * h1(seed, 1)) * (1 - np.exp(-o * 2.2)) * 1.2 + np.minimum(v, 0) * .35; m = m0 + half * v; n = n + .07 * np.sin(v * 6 + 6 * h1(seed, 2))
    if not front: n = -n; h = -h
    w = hwr * np.clip((v + .12) / .50, 0, 1) ** .62 * np.clip((1 + tipl - v) / (.62 + tipl), 0, 1) ** .95 * (1 + .18 * noise(v * 3.1, seed * 1.3))
    return m, n, h, w


def helix(ax, f, kinds=('gold', 'blue'), head=None, tail=None, pitch=8.4, rad=.74, hwr=.44, speed=13., seed=1, licks=1., step=1, light=1., collars=5, core=1., threads=1., veil=.9, dens=1.):
    """gold and blue twisted into one: blades of flame wrap round a white heart (near ones over it, far ones behind), leave as licks, shed rings -> (layer, light)"""
    t = (f // step) * step / 24.; a = getattr(ax, 'm0', 0.) if tail is None else tail; b = min(ax.M, head) if head is not None else ax.M
    if b - a < .1: return None, None
    r = ax.r(np.linspace(a, b, 15)); rm = float(np.median(r)); pr = [pal(k) for k in kinds]; nk = len(kinds); layers = []; half = pitch / 2; sh = speed * t
    if veil:
        for i in (1, 0):
            V = body_field(ax, Field(4), t * .9, a, b, seed + 17 + i * 5, .86, (.9, .8, .5, .2), .10, speed * 1.2, licks * 1.0, (2.2, 6.5), .17, .95, taper_in=1.1); c_ = pr[i % nk]['c']; layers.append(V.cel((c_[0], c_[0], c_[1], c_[2]), alpha=(veil * .55, veil, veil, veil), lw=(float(np.clip(rm * .008, .8, 2.)), 0, 0, 0), ink=(pr[i % nk]['ink'], None, None, None), seed=seed + i))
    cols = []
    for c in range(collars):
        u = ((c + .5) / collars + t * speed * .6 / max(1., b - a)) % 1.; m = a + (b - a) * u; cols.append(collar(ax, m, 1.30 + .45 * h1(seed, c, 3), .045 + .05 * h1(seed, c, 2), f, seed + c * 7, kinds[c % nk], gaps=.35))
    layers += [c[0] for c in cols]
    LEAF = []
    for i in range(nk):
        ph = pitch * i / nk
        for c in _cells(a - half, b, pitch, sh + ph):
            m0 = c * pitch + sh + ph; LEAF.append((m0, i, True, c * 13 + i * 5)); LEAF.append((m0 + half, i, False, c * 13 + i * 5 + 3))
    LEAF.sort(key=lambda q: q[0])
    def blade(F, m0, i, front, sd, G, Wt, Sh, E):
        tl = .42 + .55 * h1(sd, 11); k = 34; m, n, h, w = _leaf(ax, m0, half, rad * (.94 + .14 * h1(sd, 12)), hwr * (.82 + .40 * h1(sd, 13)) * (1. if front else .9), sd, front, tl, k); sel = (m >= a - .02) & (m <= b)
        if sel.sum() < 6: return
        edge = np.clip((m - a) / .8, 0, 1) ** .5; m, n, h, w = m[sel], n[sel], h[sel], (w * edge)[sel]; P = ax.pt(m, n, h)
        if not _vis(P, 200): return
        rr = ax.r(m); hw = rr * w; N = normals(P); K_ = len(m); v = lin(K_); sg = 1 if front else -1; kk = float(np.clip(rr.mean() * .06, 2, 26))
        off = [0 * hw, hw * .06 * sg, hw * .15 * sg, hw * .24 * sg]; F.rib(P, [hw, hw * .78, hw * .47, hw * .20 - .3], kk * .6, off)
        if Sh is not None: Sh.rib(P - N * (hw * .16)[:, None] + [0, 0], [hw * 1.10], kk)
        Lp = float(np.hypot(*np.diff(P, axis=0).T).sum())
        for q in range(int(5 + 5 * licks * (1 if front else .6))):           # notched edges: tongues that start on the blade and stream downstream
            if h1(sd, q, 21) > .85: continue
            x = .10 + .74 * h1(sd, q, 22); j = int(x * (K_ - 1)); s_ = 1 if q % 2 else -1; z = bms(h1(sd, q, 23), .25, .35); l = (.55 + 1.9 * z) * (.8 + .4 * h1(sd, q, 24)); rj = rr[j]; vv = lin(11); life = .7 + .3 * math.sin(t * 10 + q * 2 + sd); nn = n[j] + (s_ * w[j] * .72 + s_ * (.03 + .20 * z * life) * (1 - (1 - vv) ** 1.8) + .05 * z * np.sin(vv * 5 - t * 12 + q) * vv); Q = ax.pt(np.minimum(m[j] + l * vv, b + 1.), nn, h[j] * (1 - .6 * vv)); L_ = float(np.hypot(*np.diff(Q, axis=0).T).sum())
            if L_ > 10: F.rib(Q, kwidths(11, L_, hw[j] * (.16 + .24 * z), LV), kk * .7)
        if E is not None:
            for s_ in (1, -1):
                wl = np.clip(hw * .040, .9, 5.) * np.clip(np.sin(math.pi * v) * 2.6, 0, 1) * np.clip(.35 + .9 * noise(v * 6 + s_, sd, t * 2, seed + 3), 0, 1.4); E.rib(P + N * (s_ * (hw - wl * .4))[:, None], [wl - .2], 0.)
        for q in range(int(9 * dens + Lp / 70)):                     # flow lines along the blade
            x = (h1(sd, q, 31) + t * (1.2 + .6 * h1(sd, q, 32))) % 1.25 - .2; l = .14 + .34 * h1(sd, q, 33); k0, k1 = int(np.clip(x, 0, 1) * (K_ - 1)), int(np.clip(x + l, 0, 1) * (K_ - 1))
            if k1 - k0 < 4: continue
            qq = slice(k0, k1 + 1); LL = lin(k1 - k0 + 1); o = (h1(sd, q, 34) * 2 - 1) * .80; ww = hw[qq] * (.018 + .085 * h1(sd, q, 35) ** 2) * np.sin(math.pi * LL) ** .8; (G if q % 2 else Wt).rib(P[qq] + N[qq] * (hw[qq] * (o + .16 * np.sin(LL * 3.5 + q)))[:, None], [ww], 0.)
    def group(front, i):
        F, G, Wt, E = Field(4), Field(1), Field(1), Field(1); Sh = Field(1) if front else None
        for m0, j, fr, sd in LEAF:
            if fr == front and j == i: blade(F, m0, j, fr, sd, G, Wt, Sh, E)
        P_ = pr[i]; c = P_['c']; cc = c if front else (P_['ink'], c[0], c[1], c[2]); o = F.cel(cc, lw=(float(np.clip(rm * .014, 1., 3.4)), 0, float(np.clip(rm * .006, .7, 1.6)), 0), ink=(P_['ink'], None, c[0], None), seed=seed + i); g = G.cel((cc[0],)); g *= F.mask(1)[..., None] * .9; wt = Wt.cel((cc[3],)); wt *= (F.mask(0) * (1 - F.mask(3)))[..., None]; e = E.cel((P_['ink'],)); e *= F.mask(0)[..., None]
        return stack(o, g, wt, e), Sh
    for i in range(nk): layers.append(group(False, i)[0])
    B = body_field(ax, Field(4), t, a, b, seed + 3, rad * .86, (1., .86, .66, .42), .09, speed * 1.4, licks * .35, (1., 2.4), .12, .3, taper_in=.8); layers.append(B.cel((pr[1 % nk]['c'][1], pr[1 % nk]['c'][2], '#fff3cf', '#ffffff'), lw=(0, 0, 1., 0), ink=(None, None, pr[0]['c'][2], None), seed=seed)); bm = B.mask(0)
    for col, sd, n_, w_ in ((pr[0]['c'][2], 4, 34, .030), (pr[1 % nk]['c'][2], 5, 34, .030), (pr[0]['c'][1], 6, 14, .016), (pr[1 % nk]['c'][1], 7, 18, .016)):
        o = streams(ax, t, a, b, seed + sd, int(n_ * dens), speed * 1.7, .62, (1., 4.), (.006, w_)).cel((col,)); o *= bm[..., None]; layers.append(o)
    for i in range(nk):
        L, Sh = group(True, i); base = stack(*layers).copy(); d = np.clip(Sh.mask(0), 0, 1) * bm * .62; base[..., :3] = base[..., :3] * (1 - d[..., None]) + C(pr[i]['c'][0] if i else '#c8600a') * (d * base[..., 3])[..., None]; layers = [base, L]
    if threads:                                               # fine outer threads, opposite twist
        T_ = Field(2)
        for i in range(int(6 * threads)):
            m = ax.msamp(a, b, 12., 20, 160); ph = -TAU * (m - speed * 1.4 * t) / (pitch * (1.5 + .5 * h1(seed, i, 61))) + i * 1.05; rr = 1.10 + .24 * h1(seed, i, 62) + .05 * np.sin(m * 1.7 + i); nn, hh = rr * np.cos(ph), rr * np.sin(ph); P = ax.pt(m, nn, hh); w = ax.r(m) * (.012 + .016 * h1(seed, i, 63)) * np.clip(hh + .15, 0, 1) * np.clip(1.5 * noise(m * .5 - t * 6, i * 3., 0, seed) + .25, 0, 1) - .3; T_.rib(P, [w, w * .45 - .2], 0.)
        layers.append(T_.cel(('#9fdcff', '#ffffff')))
    layers += [c[1] for c in cols]; lay = stack(*layers); li = bloom(lay, .55, (6, 24, 84), (.24, .24, .28), '#ffe9c8', .62 * light); return lay, li * (1 - .6 * lay[..., 3:4])


# ------------------------------------------------------------------ ends
def petals(F, cx, cy, r0, r1, n, seed, t, asp=1., rot=0., skew=None, curl=.5, w=.16, k=10., grow=1., hz=1.7, arc=(0, TAU), boil=True):
    """living crown of tongues round a point (muzzle, splash, impact): each grows, curls, tears.  asp / rot = plane seen in perspective"""
    ca, sa = math.cos(rot), math.sin(rot)
    for i in range(n):
        f_ = hz * (.7 + .7 * h1(seed, i, 1)); x = t * f_ + h1(seed, i, 2); c = math.floor(x); tau = x - c if boil else .5; sd = int(seed * 71 + i * 13 + (c if boil else 0) * 5) & 0xffffff; z = bms(h1(sd, 3), .25, .35); a = arc[0] + (arc[1] - arc[0]) * (i + .8 * h1(sd, 4)) / n; g = 1 - (1 - min(1., tau / .5)) ** 2; L = (r1 - r0) * (.25 + .75 * z) * g * grow
        if skew is not None: L *= 1 + skew[1] * max(-.8, math.cos(a - skew[0]))
        tear = max(0., (tau - .55) / .45) if boil else 0.; st = r0 * (.82 + .1 * h1(sd, 5)) + L * .8 * tear; b = L * w * (.7 + .5 * h1(sd, 6)) * (1 - tear) ** .8
        if L < 6 or b < .8: continue
        P = spine(0, 0, L * (1 - .25 * tear), a, 12, .08, .8, t * 8 + i, curl * (1 if h1(sd, 7) > .5 else -1) * (.4 + .8 * h1(sd, 8))); P = P + np.array([math.cos(a), math.sin(a)]) * st; P = np.stack([P[:, 0], P[:, 1] * asp], -1); P = np.stack([cx + P[:, 0] * ca - P[:, 1] * sa, cy + P[:, 0] * sa + P[:, 1] * ca], -1)
        if tear < .1:
            F.rib(P, twidths(12, L, b, LV[:F.L], .18, 1.15), k)
            if b > 9:                                         # big petals split: one or two side tongues
                for q in range(2):
                    if h1(sd, q, 31) > .7: continue
                    j = 2 + int(6 * h1(sd, q, 32)); T_ = P[min(11, j + 1)] - P[j - 1]; th = math.atan2(T_[1], T_[0]); s_ = 1 if h1(sd, q, 33) > .5 else -1; l = L * (.18 + .24 * h1(sd, q, 34)); F.rib(spine(P[j][0], P[j][1], l, th + s_ * .5, 8, .05, .6, q, -s_ * .5), twidths(8, l, b * .30, LV[:F.L], .15, 1.2), k * .7)
        else: F.rib(P, lwidths(12, L, b, LV[:F.L], .35), k)
    return F


def disc(F, cx, cy, r, seed, t, asp=1., rot=0., lv=(1., .86, .66, .42), wob=.06, k=8.):
    """wobbling solid disc / ellipse on a field"""
    a = np.linspace(0, TAU, 73); ca, sa = math.cos(rot), math.sin(rot)
    for l, q in enumerate(lv[:F.L]):
        rr = r * q * (1 + wob * noise(np.cos(a) * 1.7, np.sin(a) * 1.7, t * 3, seed + l)); x, y = np.cos(a) * rr, np.sin(a) * rr * asp; P = np.stack([cx + x * ca - y * sa, cy + x * sa + y * ca], -1); m = np.zeros((F.h, F.w), np.uint8); cv2.fillPoly(m, [np.round((P - [F.ox, F.oy]) * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
        d = cv2.distanceTransform((m > 127).astype(np.uint8), cv2.DIST_L2, 3) - cv2.distanceTransform((m <= 127).astype(np.uint8), cv2.DIST_L2, 3) + (m.astype(f32) / 255. - .5); _smax(F.D[l], np.ascontiguousarray(d.astype(f32)), float(k))
    return F


from .ki_core import _smax                                    # noqa: E402


def rays(cx, cy, r0, r1, n, seed, w=.02, asp=1., rot=0., col='#ffffff', col2=None, jit=1.):
    """hard needle rays (two tones) -> layer"""
    c = Cel(ss=2); ca, sa = math.cos(rot), math.sin(rot)
    for i in range(n):
        a = (i + jit * h1(seed, i, 1)) / n * TAU; z = bms(h1(seed, i, 2), .16, .3); l = r0 + (r1 - r0) * z; hw = w * l * (.4 + h1(seed, i, 3)); d = np.array([math.cos(a), math.sin(a)]); m = np.array([-d[1], d[0]]); P = np.array([d * r0 * .3, d * l * .22 + m * hw, d * l, d * l * .22 - m * hw]); P[:, 1] *= asp; P = np.stack([cx + P[:, 0] * ca - P[:, 1] * sa, cy + P[:, 0] * sa + P[:, 1] * ca], -1); c.fill(P, col if col2 is None or i % 3 else col2)
    return c.layer()


def muzzle(cx, cy, r, f, kind='gold', seed=1, grow=1., rot=0., asp=1., kind2=None, light=1., back=0., n=26, spikes=True):
    """bloom where a beam is born: petals thrown back, a white heart, needle rays, a ring -> (layer, light).  back = direction (rad) the petals are blown to"""
    t = f / 24.; P_ = pal(kind); F = Field(4, (cx - r * 8, cy - r * 8, cx + r * 8, cy + r * 8)); petals(F, cx, cy, r * .8, r * 2.6 * grow, n, seed, t, asp, rot, (back, .55), .55, .15, r * .10); disc(F, cx, cy, r * grow, seed, t, asp, rot, (1.06, .92, .74, .56), .05, r * .12); L = []
    if kind2: Q_ = pal(kind2); G = Field(4, (cx - r * 8, cy - r * 8, cx + r * 8, cy + r * 8)); petals(G, cx, cy, r * .8, r * 3.0 * grow, int(n * .8), seed + 9, t, asp, rot, (back, .55), .6, .15, r * .10); L.append(G.cel(Q_['c'], lw=(max(1., r * .02), 0, 0, 0), ink=(Q_['ink'], None, None, None), seed=seed))
    if spikes: L.append(rays(cx, cy, r, r * 4.2 * grow, 22, seed + f // 2, .030, asp, rot, P_['c'][3], P_['c'][2]))
    L.append(F.cel(P_['c'], lw=(max(1.1, r * .022), 0, max(.8, r * .010), 0), ink=(P_['ink'], None, P_['c'][0], None), seed=seed)); lay = stack(*L); return lay, bloom(lay, .5, (6, 26, 90), (.3, .28, .30), P_['glow'], .75 * light)


def impact_front(cx, cy, r, f, kinds=('gold', 'blue'), seed=1, asp=.45, rot=0., grow=1., light=1., back=None, n=30, reach=4.2):
    """where a beam meets a wall: three crowns of long thin tongues running over the surface, white-hot torn disc, needles -> (layer, light)"""
    t = f / 24.; L = []; box = (cx - r * reach * 1.5, cy - r * reach * 1.5, cx + r * reach * 1.5, cy + r * reach * 1.5); rg = [(0, 1.0, .085, int(n * 1.5)), (1 % len(kinds), .78, .095, int(n * 1.3)), (0, .52, .13, n), (1 % len(kinds), .40, .15, n)]
    L.append(rays(cx, cy, r, r * reach * 1.15 * grow, 40, seed + f // 2, .016, asp, rot, '#ffffff', pal(kinds[0])['c'][2]))
    for j, (ki_, q, w, m) in enumerate(rg):
        P_ = pal(kinds[ki_]); F = Field(4, box); petals(F, cx, cy, r * .50, r * reach * q * grow, m, seed + j * 7, t, asp, rot, back, .34 + .12 * (j % 2), w, r * .045, hz=2.0); L.append(F.cel(P_['c'], lw=(max(1., r * .016), 0, 0, 0), ink=(P_['ink'], None, None, None), seed=seed + j))
    F = Field(4, box); petals(F, cx, cy, r * .45, r * 1.9, 34, seed + 90, t, asp, rot, back, .3, .17, r * .03, hz=2.6); disc(F, cx, cy, r * .62, seed, t, asp, rot, (1.12, 1., .88, .76), .16, r * .05); petals(F, cx, cy, r * .42, r * 1.25, 22, seed + 95, t, asp, rot, back, .15, .34, r * .03, hz=3.1); L.append(F.cel((pal(kinds[-1])['c'][1], pal(kinds[-1])['c'][2], '#ffffff', '#ffffff'), lw=(max(1., r * .012), 0, 0, 0), ink=(pal(kinds[-1])['ink'], None, None, None), seed=seed))
    lay = stack(*L); return lay, bloom(lay, .5, (6, 26, 90), (.3, .28, .32), pal(kinds[0])['glow'], .8 * light)


# ------------------------------------------------------------------ parted flow
def parting(m0, size=.5, lead=1.6, hold=5., fade=14., n0=0.):
    """axis warp: the flow opens round an obstacle at metric m0 (lateral n0, half size in radii) and closes far downstream (toward larger m)"""
    def warp(m, n):
        up = np.clip((m - (m0 - lead)) / lead, 0, 1); up = up * up * (3 - 2 * up); dn = np.clip(1 - (m - m0 - hold) / fade, 0, 1); a = size * up * dn; q = n - n0; return n0 + np.sign(q + 1e-9) * np.sqrt(q * q + a * a * np.exp(-np.abs(q) * .9) * 1.8) * np.where(a > 0, 1, 1)
    return warp


def wake(ax, m0, size, f, kind='gold', seed=1, lead=.25, length=9., n0=0., fill='#2a0f04', t=None, open_=.9):
    """the gap behind the hand that parts a beam: a dark ragged void that opens, boils at its lips with inward tongues, and closes far downstream -> layer"""
    t = f / 24. if t is None else t; P_ = pal(kind); raw = ax.warp; ax.warp = None; F = Field(4); m = ax.msamp(m0 - lead, m0 + length, 10., 20, 160); v = np.clip((m - m0) / length, 0, 1); prof = np.clip((m - m0 + lead) / (lead + .30), 0, 1) ** .5 * (1 + open_ * np.clip(v * 2.2, 0, 1) ** .8) * np.clip((1 - v) / .6, 0, 1) ** .8; w0 = size * prof * (1 + .20 * noise(m * 1.9 - t * 26, 1.1, t * 2, seed)); hw = ax.r(m) * w0; P = ax.pt(m, n0 + .05 * np.sin(m * 1.1 - t * 9)); r = ax.r(m)
    F.rib(P, [hw + r * .10, hw + r * .06, hw + r * .022, hw], 5.); T_ = Field(1)
    for sg in (1, -1):
        for row, (pit, sc) in enumerate(((.55, 1.), (.25, .5))):
            for c in _cells(m0, m0 + length, pit, 17 * t):
                mm = (c + h1(seed, c, sg, row)) * pit + 17 * t
                if not m0 + .15 < mm < m0 + length * .95: continue
                q = float(np.interp(mm, m, w0)); vv = lin(10); z = bms(h1(seed, c, sg, row, 2), .25, .35) * sc; Q = ax.pt(mm + (.5 + 1.7 * z) * vv, n0 + sg * (q * 1.10 - (.16 + .62 * z) * q * vv ** .75)); L = float(np.hypot(*np.diff(Q, axis=0).T).sum()); T_.rib(Q, twidths(10, L, float(ax.r(np.array([mm]))[0]) * (.035 + .06 * z), LV[:1], .2, 1.2), 3.)
    F.D[3] = np.minimum(F.D[3], -T_.D[0]); F.D[2] = np.minimum(F.D[2], -T_.D[0] + 3.); ax.warp = raw; return F.cel((P_['c'][3], P_['c'][1], P_['c'][0], fill), lw=(0, 0, 0, max(1.2, float(np.median(r)) * .012)), ink=(None, None, None, P_['ink']), seed=seed)


# ------------------------------------------------------------------ fast thin beams by the hundred
def volley(specs, f, kind='violet', w=1., seed=1, bend=None, light=1., splash=True, persp=(.35, 1.)):
    """specs = [(t0, p0, p1, width, dur frames)] thin beams: needle head, long tail.  bend = callable(P (k,2)) -> P' (deflection), hits listed in frames.
    bend with deflect(cx, cy, rx, ry).  -> (layer, light)"""
    P_ = pal(kind); F = Field(3); S = Field(3); hot = False
    for i, (t0, p0, p1, wd, dur) in enumerate(specs):
        u = (f - t0) / float(dur)
        if not -.2 < u < 1.45: continue
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float); hd = min(1., max(0., u) ** .9 * 1.0); tl = max(0., (u - .42) / 1.0) ** 1.1
        if hd - tl < .02: continue
        K_ = 22 if bend is None else 70; v = np.linspace(tl, hd, K_); P = p0 + (p1 - p0) * v[:, None]
        if bend is not None: P = bend(P, p0, p1)
        if not _vis(P, 40): continue
        per = wd * w * (persp[0] + (persp[1] - persp[0]) * v) ; L = float(np.hypot(*np.diff(P, axis=0).T).sum()); e = lwidths(K_, L, 1., ((1., 1.),), .86, .8, .7)[0]; F.rib(P, [per * np.maximum(e, -1) * 1.0, per * e * .62, per * e * .30 - .1], 0.)
    lay = F.cel(P_['c'][0:1] + P_['c'][2:3] + ('#ffffff',)); return lay, bloom(lay, .4, (4, 14, 40), (.4, .3, .2), P_['glow'], .8 * light)


def deflect(cx, cy, rx, ry, power=1.):
    """bend for volley(): straight paths slide round an ellipse like water round a stone (potential flow)"""
    def g(P, p0=None, p1=None):
        p0 = P[0] if p0 is None else p0; p1 = P[-1] if p1 is None else p1; d = p1 - p0; L = np.hypot(*d) + 1e-6; d = d / L; n = np.array([-d[1], d[0]]); a = (rx * abs(d[0]) + ry * abs(d[1])); b = (rx * abs(n[0]) + ry * abs(n[1])) * power; Q = P - [cx, cy]; x, y0 = Q @ d / a, float(np.mean(Q @ n)) / b; s = 1. if y0 >= 0 else -1.; y0 = max(abs(y0), .02); lo = np.sqrt(np.maximum(0., 1.0 - x * x)); hi = np.full_like(x, y0 + 1.25)
        for _ in range(26): m = (lo + hi) / 2; v = m * (1 - 1 / np.maximum(x * x + m * m, 1e-6)) - y0; lo = np.where(v < 0, m, lo); hi = np.where(v < 0, hi, m)
        y = (lo + hi) / 2; return np.array([cx, cy]) + d[None] * (x * a)[:, None] + n[None] * (s * y * b)[:, None]
    return g


def splashes(hits, f, kind='violet', seed=1):
    """hits = [(frame, x, y, r, dir)] where thin beams burst on something: star, crown of drops thrown back along dir -> (layer, light)"""
    P_ = pal(kind); F = Field(3); on = False
    for i, (t0, x, y, r, dr) in enumerate(hits):
        a = f - t0
        if not 0 <= a < 7: continue
        on = True; k = (.6, 1., .9, .7, .5, .3, .15)[int(a)]; z = (.5, .9, 1.15, 1.3, 1.45, 1.6, 1.7)[int(a)]
        for j in range(9):
            an = dr + (h1(seed, i, j, 1) - .5) * 3.2; l = r * (.5 + 1.6 * bms(h1(seed, i, j, 2), .25, .3)) * z; st = r * .25 * a * (.4 + h1(seed, i, j, 5)); p = np.array([x + math.cos(an) * st, y + math.sin(an) * st]); F.rib(spine(p[0], p[1], l, an, 8, .06, .6, j, .5 * (h1(seed, i, j, 3) - .5) * 2), lwidths(8, l, r * .16 * k * (.5 + h1(seed, i, j, 4)), LV[:3], .25), 2.)
        if a < 4: F.rib(np.array([[x - 1, y], [x, y], [x + 1, y]]), [np.full(3, r * .55 * k), np.full(3, r * .42 * k), np.full(3, r * .28 * k)], 3.)
    if not on: return None, None
    l = F.cel(P_['c'][:1] + P_['c'][2:]); return l, bloom(l, .4, (4, 14, 40), (.4, .3, .2), P_['glow'], .8)


def line(p0, p1, f, t0=0, kind='violet', w=6., draw=6, hold=8, die=8, seed=1, curve=0.):
    """hair-thin finger beam: drawn across in `draw` frames, burns, thins and breaks into beads -> (layer, light)"""
    a = f - t0
    if a < 0 or a >= draw + hold + die: return None, None
    P_ = pal(kind); u = min(1., (a + 1) / float(draw)); p0, p1 = np.asarray(p0, float), np.asarray(p1, float); v = np.linspace(0, u, 60); P = p0 + (p1 - p0) * v[:, None]
    if curve: d = p1 - p0; P = P + np.array([-d[1], d[0]]) * (curve * np.sin(math.pi * v))[:, None]
    F = Field(4); k = 1. if a < draw + hold else max(0., 1 - (a - draw - hold + 1) / float(die)); ww = w * k * (.75 + .25 * np.sin(v * 60 - a * 2.5)) * np.where(v > u - .04, 1.5, 1.)
    if k < 1: ww = ww * (noise(v * 26, a * .3, 0, seed) > (1 - k) * 1.2 - .6)
    F.rib(P, [ww + 1.2, ww * .74, ww * .5, ww * .28], 0.); lay = F.cel(P_['c']); return lay, bloom(lay, .3, (4, 18, 60), (.6, .4, .3), P_['glow'], 1.)


# ------------------------------------------------------------------ inside a beam, looking along it
def inside(f, cx, cy, kind='gold', seed=1, hole=None, speed=1., toward=True, dens=1., step=1, squash=1.):
    """we stand in the beam and look along it: rings of flame, long wedges running to (toward=False) or from the vanishing point.
    hole=(x, y, rx, ry): the flow parts round a figure, its lips burn -> RGB frame"""
    t = (f // step) * step / 24.; P_ = pal(kind); c = P_['c']; x, y = grid(); dx, dy = x - cx, (y - cy) / squash; d = np.sqrt(dx * dx + dy * dy) + 1.; im = np.empty((H, W, 3), f32); g = np.clip(d / 1050., 0, 1)[..., None]; im[:] = C(c[2]) * (1 - g) + C(c[1]) * g; lay = []
    def W_(n, sd, col, w, l0, al=1.):
        F = Field(1)
        for i in range(int(n * dens)):
            sp = speed * (.7 + .8 * h1(sd, i, 1)); q = t * sp + h1(sd, i, 2); u = q - math.floor(q); ci = int(math.floor(q)); a = TAU * h1(sd, i, ci, 3); z0 = (1 - u) if toward else u; z = .10 + 1.6 * z0 ** 1.5; ln = l0 * (.4 + .9 * h1(sd, i, 4)); v = lin(14); zz = z + ln * v * z; rr = 150. / zz; P = np.stack([cx + np.cos(a + .06 * np.sin(v * 3 + i)) * rr, cy + np.sin(a + .06 * np.sin(v * 3 + i)) * rr * squash], -1)
            if hole is not None:
                hx, hy, rx, ry = hole; e = np.stack([(P[:, 0] - hx) / rx, (P[:, 1] - hy) / ry], -1); dd = np.hypot(e[:, 0], e[:, 1]) + 1e-3; k = np.sqrt(dd * dd + 1.25 * np.exp(-dd * .35)) / dd; P = np.stack([hx + e[:, 0] * k * rx, hy + e[:, 1] * k * ry], -1)
            if not _vis(P, 60): continue
            hw = rr * w * (.25 + 1.4 * h1(sd, i, 5) ** 2) * np.sin(math.pi * v ** .8) ** .9; F.rib(P, [hw], 3.)
        o = F.cel((col,)); lay.append(o * al)
    W_(44, seed, c[1], .100, 2.6); W_(44, seed + 1, c[2], .090, 2.4); W_(54, seed + 2, c[0], .030, 3.0, .9); W_(70, seed + 3, c[3], .050, 2.6); W_(60, seed + 4, '#ffffff', .026, 3.2); W_(30, seed + 5, P_['ink'], .009, 2.0, .8)
    im = put(im, stack(*lay)); return im


def clash(cx, cy, r, f, kinds=('gold', 'violet'), seed=1, rot=0., grow=1.):
    """two beams meet: a lens of white with both colours torn out sideways, rings, needles -> (layer, light)"""
    t = f / 24.; L = [rays(cx, cy, r, r * 5 * grow, 40, seed + f, .018, 1., rot, '#ffffff', pal(kinds[1])['c'][2])]; box = (cx - r * 5, cy - r * 5, cx + r * 5, cy + r * 5)
    for j, k in enumerate(kinds):
        P_ = pal(k); F = Field(4, box); sd = math.pi * (1 - j); petals(F, cx, cy, r * .5, r * 4.2 * grow, 26, seed + j * 5, t, 1., rot, (sd, .25), 1.1, .13, r * .1, arc=(sd + rot * 0 + .55, sd + TAU / 2 + .75)); L.append(F.cel(P_['c'], lw=(max(1., r * .02), 0, 0, 0), ink=(P_['ink'], None, None, None), seed=seed + j))
    F = Field(4, box); disc(F, cx, cy, r * 1.15, seed, t, 1.5, rot, (1.12, 1., .84, .66), .08, r * .1); petals(F, cx, cy, r * .8, r * 2.2, 18, seed + 20, t, 1.5, rot, None, .4, .14, r * .1); L.append(F.cel((pal(kinds[0])['c'][1], pal(kinds[0])['c'][2], '#ffffff', '#ffffff'))); lay = stack(*L); return lay, bloom(lay, .5, (8, 30, 100), (.3, .3, .34), '#fff0e0', .9)


def ignite(ax, f, t0, kind='gold', frames=7, hold=1, style='beam', seed=1, w=1., muzzle_r=None, back=math.pi, **k):
    """the first frames of a beam, drawn: t0-2 gather, t0 flash star, then the head runs to the far end in `frames` with a bow of torn light, petals bloom at the muzzle.
    style 'beam' | 'helix'.  -> (layer, light)"""
    a = f - t0; P0 = ax.pt(np.array([max(.02, getattr(ax, 'm0', 0.))]), 0.)[0]; r0 = muzzle_r or float(ax.r(np.array([.4]))[0]); P_ = pal(kind if kind != 'goldblue' else 'gold')
    if a < -3: return None, None
    if a < 0:                                                 # everything sucked into a point
        F = Field(3); s = (a + 4) / 4.
        for i in range(26): an = TAU * (i + h1(seed, i)) / 26; l = r0 * (3.2 - 2.2 * s) * (.5 + h1(seed, i, 2)); q0 = r0 * (2.6 - 2.2 * s); d = np.array([math.cos(an), math.sin(an)]); F.rib(np.array([P0 + d * q0, P0 + d * (q0 + l * .3), P0 + d * (q0 + l)]), [np.array([.3, r0 * .05, -.2]) * z for z in (1, .6, .3)], 0.)
        F.rib(np.array([P0 - [.5, 0], P0, P0 + [.5, 0]]), [np.full(3, r0 * .34 * z * (1.1 - .5 * s)) for z in (1, .75, .5)], 2.); l = F.cel((P_['c'][0], P_['c'][2], '#ffffff')); return l, bloom(l, .4, (6, 20, 60), (.5, .4, .3), P_['glow'], .9)
    if a == 0: L = rays(P0[0], P0[1], r0 * .6, r0 * 7, 20, seed, .05, 1., 0, '#ffffff', P_['c'][2]); m, ml = muzzle(P0[0], P0[1], r0 * 1.15, f, 'gold' if kind == 'goldblue' else kind, seed, .7, back=back); l = stack(L, m); return l, bloom(l, .4, (8, 30, 110), (.5, .45, .4), P_['glow'], 1.2)
    u = min(1., a / float(frames)); hd = (float(np.interp((1 - (1 - u) ** 1.7) * ax.st[-1], ax.st, ax.mt)) if hasattr(ax, 'st') else ax.M * (1 - (1 - u) ** 1.7)) if u < 1 else None; kk = 1.25 if a == 1 else 1.12 if a == 2 else 1.
    if style == 'helix': l, g = helix(ax, f, head=hd, seed=seed, **k)
    else: l, g = beam(ax, f, kind, head=hd, seed=seed, w=w * kk, **k)
    LS = [l]; G = g
    if hd is not None and hd < ax.M * .985:                   # head: bow shock of tongues swept back
        p = ax.pt(np.array([hd + .55]), 0.)[0]; q = ax.pt(np.array([hd + .2]), 0.)[0]; th = math.atan2(*(p - q)[::-1]); r = float(ax.r(np.array([hd]))[0]); F = Field(4)
        for i in range(18): an = (i - 8.5) / 8.5; z = bms(h1(seed, i, f), .3, .35); e = p + np.array([math.cos(th), math.sin(th)]) * r * (.55 - .45 * abs(an)) + np.array([-math.sin(th), math.cos(th)]) * an * r * .95; F.rib(spine(e[0], e[1], r * (.8 + 1.9 * z), th + math.pi - an * .75, 11, .08, .7, i + f, an * .5), twidths(11, r * (.8 + 1.9 * z), r * (.10 + .16 * z), LV, .2, 1.15), r * .08)
        F.rib(np.array([q, p, p + (p - q) * .5]), [np.array([r * .95, r * .9, r * .5]) * z for z in (1, .84, .62, .36)], r * .1); LS.append(F.cel(P_['c'], lw=(max(1., r * .02), 0, 0, 0), ink=(P_['ink'], None, None, None), seed=f)); G = G + bloom(LS[-1], .5, (6, 24, 80), (.3, .3, .3), P_['glow'], .7)
    m, ml = muzzle(P0[0], P0[1], r0 * .85, f, 'gold' if kind == 'goldblue' else kind, seed, (1.5 if a < 3 else 1.2 if a < 6 else 1.) if muzzle_r is None else 1., back=back, kind2='blue' if kind == 'goldblue' else None, n=30); LS.append(m); return stack(*LS), G + ml


def die(ax, f, t0, kind='gold', frames=10, seed=1, **k):
    """a beam lets go: it thins from the muzzle, breaks into long leaves that keep flying -> (layer, light)"""
    a = f - t0
    if a >= frames: return None, None
    u = a / float(frames); tail = ax.M * u ** 1.3 * .85
    l, g = beam(ax, f, kind, tail=tail, seed=seed, w=max(.12, 1 - u) ** .8, veil=.3 * (1 - u), **k); F = Field(4); t = f / 24.
    for i in range(int(26 * (1 - u) + 6)): m0 = ax.M * h1(seed, i, 5) * .9 + 14 * t % 2; n0 = (h1(seed, i, 6) * 2 - 1) * (.8 + .6 * u); m = ax.msamp(m0, m0 + 1.2 + 2.4 * h1(seed, i, 7), 22, 8, 22); Q = ax.pt(m, n0 + .1 * np.sin(m * 1.5 + i)); L = float(np.hypot(*np.diff(Q, axis=0).T).sum()); F.rib(Q, lwidths(len(Q), L, float(ax.r(np.array([m0]))[0]) * .10 * (1 - u) * (.4 + h1(seed, i, 8)), LV, .3), 2.)
    s = F.cel(pal(kind)['c'], lw=(1.2, 0, 0, 0), ink=(pal(kind)['ink'], None, None, None)); return stack(l, s), g * (1 - u)
