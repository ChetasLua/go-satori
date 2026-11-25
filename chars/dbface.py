"""Shared face builder for the lead's characters (Vegeta, Frieza): 3-D head, plane-shaded cel, authored features, 3-D hair blades.
    F = Face(props, cx, cy, s, yaw, pitch, roll, jaw=0).   F.P(xy) face coords -> px (never folds over the far side);  F.C(xy) spline
    F.planes(light3, th) -> list of px polygons (shadow planes of the face for a 3-D light);  blade(F, ...) -> hair blade"""
import numpy as np, cv2, math
from db.paint import rgb, f32, smooth
from db.ink import Cel, spl, resample
from db.kit import contours
from db.head3 import Head


class SHead(Head):
    jopen = 0.
    def P3(self, Q):
        Q = np.asarray(Q, float)
        if self.jopen:
            Q = Q.copy(); w = smooth(-.54, -.92, Q[..., 1]) * smooth(-.3, .35, Q[..., 2])
            Q[..., 1] -= self.jopen * w; Q[..., 2] -= self.jopen * .28 * w
        return Head.P3(self, Q)

    def limits(self):
        ys = np.linspace(-.99, .9, 48); V = self.R[2]; lo, hi = [], []
        for y in ys:
            a = np.interp(y, self.yy, self.A); x = np.linspace(-a * .985, a * .985, 121); e = 4e-3; z0 = self.z(x, y)
            nx = -(self.z(x + e, y) - self.z(x - e, y)) / (2 * e); ny = -(self.z(x, y + e) - self.z(x, y - e)) / (2 * e)
            c = (nx * V[0] + ny * V[1] + V[2]) / np.sqrt(nx * nx + ny * ny + 1); ok = c > .04
            j = int(np.argmax(c)); l = j; r = j
            while l > 0 and ok[l - 1]: l -= 1
            while r < len(x) - 1 and ok[r + 1]: r += 1
            lo.append(x[l] if l > 0 else -9.); hi.append(x[r] if r < len(x) - 1 else 9.)
        self._lim = (ys, np.array(lo), np.array(hi)); return self._lim

    def Pc(self, xy, dz=0., k=.10):
        xy = np.array(xy, float); ys, lo, hi = getattr(self, '_lim', None) or self.limits()
        l = np.interp(xy[..., 1], ys, lo); h = np.interp(xy[..., 1], ys, hi); x = xy[..., 0]
        x = np.where(x > h - k, h - k * np.exp(-np.clip((x - h + k) / k, 0, 20)), x); x = np.where(x < l + k, l + k * np.exp(-np.clip((l + k - x) / k, 0, 20)), x)
        xy[..., 0] = x; return self.P(xy, dz)


class Face:
    def __init__(self, props, cx, cy, s, yaw=0., pitch=0., roll=0., jaw=0., persp=.10):
        self.H = SHead(**props).pose(cx, cy, s, yaw, pitch, roll, persp); self.H.jopen = jaw; self.H.limits()
        self.s, self.yaw, self.pitch, self.roll = s, yaw, pitch, roll; self.jaw = jaw
        self.lw = float(np.clip((s / 260.) ** .62, .28, 2.4))
    @classmethod
    def of(cls, H, props, jaw=0.):
        return cls(props, H.cx, H.cy, H.s, H.yaw, H.pitch, H.roll, jaw, getattr(H, 'persp', .10))
    def P(self, xy, dz=0.): return self.H.Pc(np.asarray(xy, float), dz)
    def P3(self, q): return self.H.P3(np.asarray(q, float))
    def C(self, xy, n=10, closed=False, corners=(), dz=0.): return self.P(spl(np.asarray(xy, float), closed, corners, n), dz)
    def vis(self, x, y): return self.H.facing(float(np.clip(x, -.64, .64)), y)
    def contour(self): return self.H.face(.75)
    def planes(self, L, th=.12, relief=1., n=120, eps=.012, ymax=.8):
        """shadow polygons of the face for a light direction L given in SCREEN space (x right, y up, z to viewer)"""
        H = self.H; Lh = H.R.T @ (np.array(L, float) / np.linalg.norm(L))
        x = np.linspace(-.72, .72, n); y = np.linspace(-1., ymax, int(n * 1.3)); X, Y = np.meshgrid(x, y)
        A = np.interp(y, H.yy, H.A)[:, None]; u = np.clip(X / A, -.985, .985); Xs = u * A
        Z = H.z(Xs, Y) + relief * self.relief(Xs, Y)
        dy, dx = np.gradient(cv2.GaussianBlur(Z.astype(np.float64), (0, 0), n / 75.), y, x)
        nn = np.sqrt(dx * dx + dy * dy + 1); lam = (-dx * Lh[0] - dy * Lh[1] + Lh[2]) / nn
        lam = cv2.GaussianBlur(lam, (0, 0), n / 90.)
        m = np.pad(((th - lam) * 6 + .5).clip(0, 1) * (np.abs(X) < A * 1.02), 2).astype(f32); out = []
        cs, _ = cv2.findContours((m > .5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            if cv2.contourArea(c) < n * n * .0016: continue
            c = cv2.approxPolyDP(c.astype(np.float32), eps * n, True)[:, 0] - 2
            p = np.stack([np.interp(c[:, 0], np.arange(len(x)), x), np.interp(c[:, 1], np.arange(len(y)), y)], -1)
            p = resample(np.vstack([p, p[:1]]), .03); out.append(self.P(p))
        return out
    def relief(self, x, y):
        g = lambda x0, y0, sx, sy, k: k * np.exp(-((np.abs(x) - x0) / sx) ** 2 - ((y - y0) / sy) ** 2)
        return g(.40, .13, .30, .07, .05) - g(.30, -.02, .15, .07, .05) + g(.42, -.26, .16, .10, .045) - g(.34, -.52, .13, .16, .045) + g(0, -.25, .06, .22, .09) + g(0, -.40, .075, .07, .06) + g(.0, -.86, .16, .09, .035)


def bez(P0, P1, P2, n=14):
    t = np.linspace(0, 1, n)[:, None]; return (1 - t) ** 2 * np.array(P0, float) + 2 * t * (1 - t) * np.array(P1, float) + t * t * np.array(P2, float)


def blade(F, root, tip, bow=(0, 0, 0), r=.3, flat=1., belly=.35, n=16, pw=.8):
    """3-D hair blade -> dict(L, R edges px, spine px, depth, poly).  bow = offset of the middle control point"""
    root, tip = np.array(root, float), np.array(tip, float); mid = (root + tip) / 2 + np.array(bow, float)
    S = bez(root, mid, tip, n); t = np.linspace(0, 1, n); p = F.P3(S); H = F.H
    f = 1 / (1 - np.clip((S @ H.R.T)[:, 2] * H.persp, -.5, .5))
    rad = r * (1 - t) ** pw * (1 + belly * np.sin(np.pi * t)) * np.clip(.18 + t / .15, 0, 1) ** .8 * F.s * f * flat
    d = np.gradient(p, axis=0); ln = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6; d = d / ln; nrm = np.stack([-d[:, 1], d[:, 0]], -1)
    L, R = p + nrm * rad[:, None], p - nrm * rad[:, None]
    return dict(L=L, R=R, S=p, rad=rad, depth=float(F.H.depth((root * .5 + tip * .5))), poly=np.vstack([L, R[::-1]]))


def side_poly(b, lx, ly, k0=.1, t0=.0, t1=1.):
    """part of a blade on the side AWAY from screen light (lx right, ly up): from the spine offset k0 to the edge"""
    S, L, R = b['S'], b['L'], b['R']; mid = len(S) // 2; n = L[mid] - S[mid]; s = n[0] * lx - n[1] * ly
    E = R if s > 0 else L; i0, i1 = int(t0 * (len(S) - 1)), int(round(t1 * (len(S) - 1))) + 1
    inner = S + (E - S) * k0
    return np.vstack([inner[i0:i1], E[i0:i1][::-1]]), (L if s > 0 else R)


def scalp(F, c, th, col, y0=-.30, y1=.995, ny=26, nt=26):
    """fill the part of the skull behind the hairline.  th(y) -> azimuth of the hairline (0 front .. pi back) for each height"""
    H = F.H; p = H.k['p']; ys = np.linspace(y0, y1, ny); A, Ff, B = (np.interp(ys, H.yy, q) for q in (H.A, H.F, H.B))
    for sd in (-1, 1):
        T = np.stack([np.linspace(th(y), np.pi + .06, nt) for y in ys]); sx, cz = np.sin(T), np.cos(T); r = (np.abs(sx) ** p + np.abs(cz) ** p) ** (-1 / p)
        X = sd * A[:, None] * sx * r * 1.01; Z = np.where(cz >= 0, Ff[:, None], B[:, None]) * cz * r * 1.01; q = F.P3(np.stack([X, np.repeat(ys[:, None], nt, 1), Z], -1))
        a, b, d, e = q[:-1, :-1], q[:-1, 1:], q[1:, 1:], q[1:, :-1]; cr = (b - a)[..., 0] * (e - a)[..., 1] - (b - a)[..., 1] * (e - a)[..., 0]
        m = np.zeros((1, 1)); lo = q.reshape(-1, 2).min(0) - 4; hi = q.reshape(-1, 2).max(0) + 4; k = max(.4, min(1., 700 / max(hi - lo))); w, h = ((hi - lo) * k).astype(int) + 4; m = np.zeros((h, w), np.uint8)
        for i, j in zip(*np.where(cr * sd < 0)): cv2.fillConvexPoly(m, np.round((np.stack([a[i, j], b[i, j], d[i, j], e[i, j]]) - lo) * k * 16).astype(np.int32), 255, cv2.LINE_8, 4)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        for ct in contours(m / 255., min_len=12): c.fill(ct / k + lo, col)


def patch(F, c, edge, col, back=True, ny=30, nt=40, k_out=1.012, y1=1.0, lift=0., swell=0.):
    """skull patch above an edge curve: edge(theta) -> y (theta 0 front .. pi back).  Returns px contours"""
    H = F.H; p = H.k['p']; out = []; qs = []
    th = np.linspace(-np.pi, np.pi, nt * 2); e = np.array([edge(abs(t)) for t in th]); V = np.linspace(0, 1, ny)[:, None]; Y = e[None] + (y1 - e[None]) * V
    A, Ff, B = (np.interp(Y, H.yy, q) for q in (H.A, H.F, H.B)); sx, cz = np.sin(th)[None], np.cos(th)[None]; r = (np.abs(sx) ** p + np.abs(cz) ** p) ** (-1 / p)
    u = (Y - e[None]) / np.maximum(1 - e[None], .05); g = k_out + swell * np.sin(np.pi * np.clip(u, 0, 1) ** .8); q = F.P3(np.stack([A * sx * r * g, Y * k_out + lift * u ** 1.3, np.where(cz >= 0, Ff, B) * cz * r * g], -1))
    a, b, d, f = q[:-1, :-1], q[:-1, 1:], q[1:, 1:], q[1:, :-1]; cr = (b - a)[..., 0] * (f - a)[..., 1] - (b - a)[..., 1] * (f - a)[..., 0]
    lo = q.reshape(-1, 2).min(0) - 4; hi = q.reshape(-1, 2).max(0) + 4; k = max(.4, min(1., 700 / max(hi - lo))); w, h = ((hi - lo) * k).astype(int) + 4; m = np.zeros((h, w), np.uint8)
    for i, j in zip(*np.where(cr < 0)): cv2.fillConvexPoly(m, np.round((np.stack([a[i, j], b[i, j], d[i, j], f[i, j]]) - lo) * k * 16).astype(np.int32), 255, cv2.LINE_8, 4)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)); out = [ct / k + lo for ct in contours(m / 255., min_len=14)]
    for o in out: c.fill(o, col)
    return out


def skull_pt(F, th, y, k=1.0, e=None, lift=0., swell=0.):
    H = F.H; p = H.k['p']; th = np.asarray(th, float); y = np.asarray(y, float); A, Ff, B = (np.interp(y, H.yy, q) for q in (H.A, H.F, H.B)); sx, cz = np.sin(th), np.cos(th); r = (np.abs(sx) ** p + np.abs(cz) ** p) ** (-1 / p)
    if e is not None: u = np.clip((y - np.asarray(e)) / np.maximum(1 - np.asarray(e), .05), 0, 1); k2 = k + swell * np.sin(np.pi * u ** .8); return np.stack([A * sx * r * k2, y * k + lift * u ** 1.3, np.where(cz >= 0, Ff, B) * cz * r * k2], -1)
    return np.stack([A * sx * r * k, y * k, np.where(cz >= 0, Ff, B) * cz * r * k], -1)


def neck_poly(F, w0=.46, w1=.52, z0=-.12, z1=-.20, y0=-.52, y1=-3.4, n=14):
    """3-D neck column -> (polygon, left line, right line) px, any angle"""
    L, R = [], []; ph = np.linspace(0, 2 * np.pi, 36); ax = F.P3(np.array([(0, y0, z0), (0, y1, z1)])); d = ax[1] - ax[0]; d = d / (np.linalg.norm(d) + 1e-6); nrm = np.array([-d[1], d[0]])
    for t in np.linspace(0, 1, n):
        r = w0 + (w1 - w0) * t ** 1.6 - .03 * np.sin(np.pi * min(1, t * 1.3)); q = F.P3(np.stack([r * np.cos(ph), np.full_like(ph, y0 + (y1 - y0) * t), z0 + (z1 - z0) * t + r * .92 * np.sin(ph)], -1)); s = q @ nrm
        L.append(q[s.argmax()]); R.append(q[s.argmin()])
    L, R = np.array(L), np.array(R); return np.vstack([L, R[::-1]]), L, R
