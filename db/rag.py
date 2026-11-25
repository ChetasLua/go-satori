"""Torn cloth: verlet strips hanging from any contour, frayed hems, ribbons, wet folds with running water.

    R = Rag(edge, length, n=None, seed=3, tear=.5, wet=0., stiff=.35)       edge = polyline px (the contour the cloth hangs from, left to right)
    R.run(wind, frames=80, fps=24, warm=48, gravity=1.)                     wind = f(t) -> (wx, wy) in px/s2 x1000 or a constant pair; simulates once, stores every frame
    R.draw(cel, f, pal)                                                     pal = PAL['rain'] ... or dict(cloth, shade, deep, line, light)
    ribbon(root, L, w, wind, t, seed) -> (A, B, spine)                      one drawn, looping ribbon (no state)
    fray(cel, P, col, line, seed, L, lw)                                    frayed hem along any polyline (threads point to the polyline's right)
    folds(cel, poly, pts, pal, lw, seed, wet)                               wet cloth folds radiating from tension points inside a polygon
    water(cel, P, t, pal, seed, r)                                          beads running down a polyline and dripping off its end; returns [(x, y, r, falling)]
All drawing is a pure function of the stored frame, so shots can call it in any order."""
import numpy as np, math
from .paint import rgb, mix
from .ink import spl, resample, noise1
from .kit import outline

PAL = dict(
    rain=dict(cloth='#3d5f94', shade='#263f69', deep='#162642', line='#0c0e13', light='#a9c0d6', lining='#2d4878'),
    violet=dict(cloth='#2c3a86', shade='#171c50', deep='#0c0e30', line='#080612', light='#b488f0', lining='#1a1f55'),
    dawn=dict(cloth='#3f5c9c', shade='#324078', deep='#232a58', line='#1a1524', light='#ffdcc4', lining='#2b3768'),
    gold=dict(cloth='#3f5a78', shade='#2a3344', deep='#1a1c28', line='#2a0f08', light='#ffe6a0', lining='#2c3040'))


class Rag:
    def __init__(self, edge, length, n=None, seed=3, tear=.5, wet=0., stiff=.35, seg=11):
        self.r = r = np.random.default_rng(seed); E = resample(np.asarray(edge, float), 2.); self.seed, self.wet, self.stiff = seed, wet, stiff
        w = np.hypot(*np.diff(E, axis=0).T).sum(); n = n or max(3, int(w / max(26, length * .20))); c = np.sort(np.r_[0, np.clip(np.linspace(0, 1, n + 1)[1:-1] + r.normal(0, .22 / n, n - 1), .02, .98), 1])
        self.cut = c; self.E = E; self.n = n; self.seg = seg; idx = lambda t: E[np.clip((t * (len(E) - 1)).round().astype(int), 0, len(E) - 1)]
        self.a, self.b = idx(c[:-1]), idx(c[1:]); self.L = length * np.clip(1 + tear * r.normal(0, .42, n) - .25 * tear * (r.random(n) < .3), .22, 1.8)
        self.m = (self.a + self.b) / 2; self.w = np.hypot(*(self.b - self.a).T)

    def run(self, wind, frames=80, fps=24, warm=48, gravity=1.):
        n, k = self.n, self.seg; h = self.L / (k - 1); P = self.m[:, None] + np.stack([np.zeros((n, k)), h[:, None] * np.arange(k)], -1); Q = P.copy(); sub = 4; dt = 1 / fps / sub; out = []
        wf = wind if callable(wind) else (lambda t: wind); ph = self.r.uniform(0, 6.28, n); mass = 1 + 1.6 * self.wet; u = np.arange(k) / (k - 1)
        for f in range(-warm, frames):
            for s in range(sub):
                t = (f + s / sub) / fps; wx, wy = wf(t); gust = 1 + .35 * np.sin(t * 5.3 + ph)[:, None] + .30 * np.sin(t * 9.1 + ph * 2.3)[:, None] - u * .0
                flut = np.sin(t * 13. - u[None] * 7. + ph[:, None]) * .55
                T = np.gradient(P, axis=1); T /= np.hypot(T[..., 0], T[..., 1])[..., None] + 1e-9; N = np.stack([-T[..., 1], T[..., 0]], -1); W_ = np.array([wx, wy]) * 1000.
                fn = (N @ W_)[..., None] * N * gust[..., None] + N * (flut * np.hypot(*W_) * .35)[..., None] + T * ((T @ W_)[..., None]) * .10
                A = fn / mass + np.array([0, 2100. * gravity]); V = (P - Q) * (1 - .030 - .02 * self.wet); Q, P = P, P + V + A * dt * dt; P[:, 0] = self.m
                for _ in range(7):
                    for j in range(1, k):
                        d = P[:, j] - P[:, j - 1]; L = np.hypot(*d.T)[:, None] + 1e-9; e = d * ((L - h[:, None]) / L)
                        if j == 1: P[:, 1] -= e
                        else: P[:, j - 1] += e * .5; P[:, j] -= e * .5
                    P[:, 0] = self.m
                for j in range(2, k): mid = P[:, j - 1] + (P[:, j - 1] - P[:, j - 2]); P[:, j] += (mid - P[:, j]) * self.stiff * .05
            if f >= 0: out.append(P.astype(np.float32).copy())
        self.F = np.array(out); return self

    def strip(self, f, i):
        """outline of one strip at frame f: (left side, right side, spine)"""
        S = spl(self.F[f % len(self.F), i].astype(float), n=4); T = np.gradient(S, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[T[:, 1], -T[:, 0]]; u = np.linspace(0, 1, len(S)); r = np.random.default_rng(self.seed * 41 + i)
        tip = r.uniform(.10, .62); tw = np.sin(u * 4.2 + r.uniform(0, 6.28)) * .22 * u
        w = self.w[i] * .56 * ((1 - u) ** .55 * (1 - tip) + tip) * (1 + .07 * noise1(len(u), 8, r)) * (1 - np.abs(tw)); d0 = (self.b[i] - self.a[i]) / (self.w[i] + 1e-9); k = np.clip(u * 5, 0, 1)[:, None]; N = d0 * (1 - k) + N * k * (1 if (N[len(N) // 2] @ d0) >= 0 else -1) if True else N
        N /= np.hypot(*N.T)[:, None] + 1e-9; return S - N * w[:, None], S + N * w[:, None], S

    def draw(self, cel, f, pal='rain', lw=1.6, L=(-.5, -.8)):
        c = PAL[pal] if isinstance(pal, str) else pal; order = np.argsort(-self.L * (1 + .2 * np.cos(np.arange(self.n) * 2.4)))
        for i in order:
            A, B, S = self.strip(f, i); r = np.random.default_rng(self.seed * 43 + i); m = 6; e0, e1 = A[-1], B[-1]; ax = S[-1] - S[-5]; ax /= np.hypot(*ax) + 1e-9
            zig = [e0 + (e1 - e0) * ((k + .5) / m) + ax * (self.w[i] * (.34 if k % 2 == 0 else -.10) * r.uniform(.5, 1.5)) for k in range(m)]; poly = np.vstack([A, zig, B[::-1]])
            back = i % 3 == 2; cel.fill(poly, c['lining'] if back else c['cloth']); cel.ctx.save(); cel._path(poly); cel.ctx.clip()
            D = np.gradient(S, axis=0); side = 1 if (np.c_[D[:, 1], -D[:, 0]][len(S) // 2] @ np.array(L)) > 0 else -1; far = A if side > 0 else B
            wob = .30 + .22 * np.sin(np.linspace(0, 9, len(S)) + i)
            cel.fill(np.vstack([far, (S + (far - S) * wob[:, None])[::-1]]), c['shade']); k0 = int(len(S) * .18); cel.fill(np.vstack([A[:k0], B[:k0][::-1]]) if k0 > 2 else A, c['shade'])
            for q in range(2 + int(2 * self.wet)):                                      # folds run with the strip and bite where it bends
                o = r.uniform(-.7, .7); a0 = r.uniform(0, .35); a1 = r.uniform(.5, .95); z = slice(int(a0 * len(S)), int(a1 * len(S))); line = S[z] + (B[z] - S[z]) * o
                if len(line) > 6: w = self.w[i] * .10 * np.sin(np.linspace(0, math.pi, len(line)))[:, None]; n_ = np.c_[D[z][:, 1], -D[z][:, 0]]; n_ /= np.hypot(*n_.T)[:, None] + 1e-9; cel.fill(np.vstack([line, (line + n_ * w)[::-1]]), c['deep'] if q else c['shade']); cel.line(line, 1.2 * lw, c['line'], (.2, .6), seed=q + i)
            if self.wet: near = B if side > 0 else A; z = slice(int(.25 * len(S)), int(.8 * len(S))); cel.line((S + (near - S) * .72)[z], 2.0 * lw, c['light'], (.4, .5), seed=i, alpha=.8 * self.wet)
            cel.ctx.restore(); outline(cel, poly, 2.1 * lw, c['line'], self.seed + i, tremor=.4); fray(cel, np.vstack([e0, *zig, e1]), c['cloth'], c['line'], i + self.seed, self.w[i] * .40, lw, ax)
            if r.random() < .6:                                                          # a nick in the side
                k = int(len(S) * r.uniform(.4, .8)); X = A if r.random() < .5 else B; cel.fill(np.array([X[k], X[min(k + 5, len(S) - 1)], X[min(k + 2, len(S) - 1)] + (S[min(k + 2, len(S) - 1)] - X[min(k + 2, len(S) - 1)]) * .55]), c['deep']); cel.line(np.vstack([X[k:k + 1], X[min(k + 2, len(S) - 1)] + (S[min(k + 2, len(S) - 1)] - X[min(k + 2, len(S) - 1)]) * .55]), 1.6 * lw, c['line'], (.05, .6), seed=k)
        return cel

    def tips(self, f): return self.F[f % len(self.F), :, -1]


def ribbon(root, L, w, wind=(1., 0.), t=0., seed=0, n=40, curl=1.):
    """looping drawn ribbon: travelling wave on a wind line; returns (A, B, spine)"""
    r = np.random.default_rng(seed); u = np.linspace(0, 1, n); a = math.atan2(wind[1] + .0, wind[0] + 1e-9); sp = min(1.5, math.hypot(*wind)); p, q = r.uniform(0, 6.28, 2); sag = (1 - min(1, sp)) * 1.2
    th = a + (math.pi / 2 - a) * min(1., sag) * .8 * (.4 + .6 * u) + curl * (.55 * np.sin(u * 5.2 - t * 6.3 + p) + .28 * np.sin(u * 11. - t * 9.4 + q)) * (.25 + .75 * u)
    S = np.asarray(root, float) + np.c_[np.cumsum(np.cos(th)), np.cumsum(np.sin(th))] * L / n; T = np.gradient(S, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]; hw = (w / 2 * (1 - .35 * u) * (.35 + .65 * np.abs(np.cos(u * 4.1 - t * 3.1 + p))))[:, None]
    return S + N * hw, S - N * hw, S


def fray(cel, P, col, line, seed=0, L=14., lw=1.4, ax=None, n=None):
    ax = None if ax is None else np.asarray(ax, float) / (np.hypot(*ax) + 1e-9); r = np.random.default_rng(seed * 59 + 3); P = resample(np.asarray(P, float), 2.); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]] if ax is None else np.repeat(np.asarray(ax, float)[None], len(P), 0); n = n or max(3, len(P) // 4)
    for _ in range(n):
        k = int(r.integers(len(P))); l = L * r.uniform(.3, 1.2); e = P[k] + N[k] * l + T[k] * r.normal(0, .35 * l); c = spl([P[k], (P[k] + e) / 2 + T[k] * r.normal(0, .18 * l), e], n=6); cel.line(c, 2.1 * lw, line, (.05, .7), seed=int(r.integers(1 << 20))); cel.line(c, 1.0 * lw, col, (.05, .7), seed=1)
    return cel


def folds(cel, poly, pts, pal='rain', lw=1.6, seed=0, wet=.6, n=7):
    """tension folds: from each (x, y, angle, length) a fan of blades with a lit edge, clipped to the polygon"""
    c = PAL[pal] if isinstance(pal, str) else pal; r = np.random.default_rng(seed); cel.clip(poly)
    for x, y, a, l in pts:
        for k in range(n):
            th = a + (k - (n - 1) / 2) * .17 + r.normal(0, .05); ln = l * r.uniform(.45, 1.1); e = np.array([x + math.cos(th) * ln, y + math.sin(th) * ln]); nn = np.array([-math.sin(th), math.cos(th)]); s_ = np.array([x, y]) + nn * (k - (n - 1) / 2) * l * .035
            S = spl([s_, (s_ + e) / 2 + nn * r.normal(0, ln * .05), e], n=14); u = np.linspace(0, 1, len(S)); w = (ln * .055 * np.sin(u ** .6 * math.pi))[:, None]; cel.fill(np.vstack([S, (S + nn * w)[::-1]]), c['shade'] if k % 2 else c['deep']); cel.line(S, 1.6 * lw, c['line'], (.1, .6), seed=k)
            if wet and k % 2: cel.line((S - nn * ln * .022)[len(S) // 6:len(S) * 2 // 3], 1.5 * lw, c['light'], (.4, .5), seed=k + 9, alpha=.75 * wet)
    cel.unclip(); return cel


def water(cel, P, t, pal='rain', seed=0, r=5., n=4, floor=1200.):
    c = PAL[pal] if isinstance(pal, str) else pal; P = resample(np.asarray(P, float), 2.); out = []; rg = np.random.default_rng(seed); a = np.linspace(0, 2 * math.pi, 22)
    for k in range(n):
        per = rg.uniform(1.1, 2.0); ph = ((t + rg.uniform(0, per)) % per) / per; R = r * rg.uniform(.6, 1.15)
        if ph < .62:                                                                     # stick-slip run down the cloth
            u = ph / .62; u = u + .08 * math.sin(u * 12.6); i = int(np.clip(u, 0, 1) * (len(P) - 1)); x, y = P[i]; tail = P[max(0, i - 26):i + 1]
            if len(tail) > 3: cel.line(tail, R * .8, c['light'], (.9, .05), alpha=.55, seed=k)
            f = 0
        elif ph < .80: x, y = P[-1]; R *= .6 + (ph - .62) * 3.; y += R; f = 0
        else: dt = (ph - .80) * per; x, y = P[-1]; y += R + 1300 * dt * dt; f = 1
        if y > floor: continue
        q = np.c_[x + np.sin(a) * R * .8, y - np.cos(a) * R * (1.7 if f else 1.15)]; cel.fill(q, c['shade'], .55); cel.line(np.vstack([q, q[:1]]), 1.2, c['line'], (.1, .1), seed=k); cel.fill(np.c_[x - R * .25 + np.cos(a) * R * .22, y - R * .1 + np.sin(a) * R * .35], '#ffffff', .9); out.append((x, y, R, f))
    return out
