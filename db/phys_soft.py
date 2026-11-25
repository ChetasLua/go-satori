"""phys soft: things that shred, float or bend: armour shattering as a travelling wave, cloth ribbons (chains in wind), ash that becomes petals of light, a ring through grass.  (PHYS lane)"""
from .phys_core import *
from .phys_rock import shatter as _solid_shatter, Mesh, draw_bodies
from scipy.spatial import Voronoi, cKDTree


# ------------------------------------------------------------- armour
def cells2d(poly, n=26, seed=0, focus=None, bias=.6):
    """Voronoi pieces of a polygon (screen px), denser near `focus` -> list of polygons that tile it"""
    P = np.asarray(poly, float); r = R(seed, 71); lo, hi = P.min(0), P.max(0); pts = []; m = P.astype(np.float32)
    while len(pts) < n:
        q = r.uniform(lo, hi)
        if focus is not None and r.random() < bias: q = np.asarray(focus) + r.normal(0, 1, 2) * (hi - lo) * .24
        if cv2.pointPolygonTest(m, (float(q[0]), float(q[1])), False) >= 0: pts.append(q)
    pts = np.array(pts); c = (lo + hi) / 2; R_ = np.hypot(*(hi - lo)) * 4; far = c + R_ * np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]]); V = Voronoi(np.vstack([pts, far])); out = []
    for i in range(n):
        reg = V.regions[V.point_region[i]]
        if -1 in reg or len(reg) < 3: continue
        q = hull2(V.vertices[reg]); out += clip_poly(P, q)
    return out


def clip_poly(subject, clipc):
    """polygon (may be concave) ∩ convex polygon -> [polygon]"""
    C_ = np.asarray(clipc, float); ar = np.sum(C_[:, 0] * np.roll(C_[:, 1], -1) - np.roll(C_[:, 0], -1) * C_[:, 1]); C_ = C_ if ar > 0 else C_[::-1]; S = [np.asarray(q, float) for q in subject]
    for a, b in zip(C_, np.roll(C_, -1, 0)):
        if not S: return []
        d = b - a; f = lambda p: d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0]); out = []; prev = S[-1]; fp = f(prev)
        for p in S:
            fc = f(p)
            if fc >= 0:
                if fp < 0: out.append(prev + (p - prev) * fp / (fp - fc))
                out.append(p)
            elif fp >= 0: out.append(prev + (p - prev) * fp / (fp - fc))
            prev, fp = p, fc
        S = out
    q = np.array(S)
    return [q] if len(q) >= 3 and cv2.contourArea(q.astype(np.float32)) > 14 else []


class ArmourBreak:
    """Armour bursts in a travelling wave.  ab = ArmourBreak([(polygon px, '#hex' or dict lit/mid/dark/ink, n_pieces), ...], hit=(x, y), speed=1400)
    ab.draw(cel, t)  t = seconds since contact: hair cracks race ahead of the front, white-hot at the front, then plates with thickness leave one after another, turn and fall.
    ab.gone(t) -> list of polygons still in place (draw your own intact armour with these, or let draw() do it with under=True)."""

    def __init__(self, parts, hit, speed=1400., blow=1100., grav=1500., push=(0., 0.), seed=0, thick=9., hold=.0):
        self.h = np.asarray(hit, float); self.sp, self.blow, self.g, self.push, self.seed, self.th, self.hold = speed, blow, grav, np.asarray(push, float), seed, thick, hold; self.C = []
        for k, p in enumerate(parts):
            poly, col = p[0], p[1]; n = p[2] if len(p) > 2 else 26; pal = col if isinstance(col, dict) else dict(lit=hexmix(col, '#ffffff', .35), mid=C(col), dark=hexmix(col, '#1a1630', .58), ink='#17131a')
            for i, q in enumerate(cells2d(poly, n, seed * 13 + k, self.h)): c = q.mean(0); self.C.append((q, c, pal, float(np.hypot(*(c - self.h))), k * 101 + i))

    def draw(self, cel, t, under=True, lw=2.2, light=(-.5, -.75), crack='#17131a', hot='#ffffff', glow=None):
        s = t - self.hold; fl = []
        for q, c, pal, d, sd in self.C:
            tau = s - d / self.sp
            if tau <= 0:
                if not under: continue
                cel.fill(q + (q - c) * .012, pal['mid'])
                if s > d / self.sp - .11 and s > 0:           # cracks ahead of the front
                    k = clamp((s - d / self.sp + .11) / .11); w = lw * (.4 + .9 * k)
                    for j, (a, b) in enumerate(zip(q, np.roll(q, -1, 0))):
                        if h1(sd, j, 1) < .85 * k + .1: wedge(cel, a, a + (b - a) * min(1., k * 1.4), w, w * .4, crack)
                    if k > .72:
                        for a, b in zip(q, np.roll(q, -1, 0)): wedge(cel, a, b, w * .8, w * .8, hot)
            else: fl.append((tau, q, c, pal, d, sd))
        fl.sort(key=lambda v: -v[0])
        L = np.array([light[0], light[1], .5]); L /= np.linalg.norm(L)
        for tau, q, c, pal, d, sd in fl:
            dr = unit(c - self.h + np.array([h1(sd, 2) - .5, h1(sd, 3) - .6]) * 60); v = dr * self.blow * (.35 + .9 * h1(sd, 4)) * math.exp(-d / 900.) + self.push + np.array([0, -220 * h1(sd, 5)]); dk = (1 - math.exp(-1.6 * tau)) / 1.6; p = c + v * dk + np.array([0, .5 * self.g * tau * tau]); ax = unit([h1(sd, 6) - .5, h1(sd, 7) - .5, (h1(sd, 8) - .5) * .7]); M = axis_m(ax, tau * (5 + 13 * h1(sd, 9))); zoom = 1 + tau * (h1(sd, 10) - .25) * 1.6
            T = np.c_[q - c, np.zeros(len(q))]; B = np.c_[(q - c) * .96, np.full(len(q), -self.th)]; T3, B3 = T @ M.T, B @ M.T; pr = lambda X: X[:, :2] * (zoom / (1 - X[:, 2:3] * .0011)) + p; t2, b2 = pr(T3), pr(B3); nz = M[2, 2]
            if p[1] > H + 300 or zoom < .1: continue
            Hl = hull2(np.vstack([t2, b2])); cel.fill(grow(Hl, lambda n: lw * (.5 + .5 * np.clip(n @ np.array([.5, .8]), 0, 1))), pal['ink'])
            for j in range(len(q)): k = (j + 1) % len(q); cel.fill(np.array([t2[j], t2[k], b2[k], b2[j]]), pal['dark'] if j % 2 else hexmix(pal['dark'], pal['mid'], .4))
            face, nn = (t2, M[:, 2]) if nz > 0 else (b2, -M[:, 2]); dd = float(nn @ L) ; cel.fill(face, (pal['lit'] if dd > .62 else pal['mid'] if dd > .18 else pal['dark']) if nz > 0 else (pal['dark'] if dd < .5 else hexmix(pal['dark'], pal['mid'], .5)))
            if nz > 0 and dd > .80: f = face; cc = f.mean(0); cel.fill(np.vstack([f[:2], [cc * .4 + f[1] * .6, cc * .5 + f[0] * .5]]), hot, .85)
        return cel

    def front(self, t):
        """radius (px) of the wave now"""
        return max(0., (t - self.hold) * self.sp)

    def gone(self, t): return [q for q, c, pal, d, sd in self.C if (t - self.hold) - d / self.sp <= 0]


# ------------------------------------------------------------- cloth ribbons
@njit(cache=True, fastmath=True)
def _chain(P, N, T, sub, seg, wx, wy, g, drag, rel, gust, sd, out):
    """verlet chains in a turbulent wind.  P (n, k, 2) rest; node 0 pinned until rel[i] (frames)"""
    n, k = P.shape[0], P.shape[1]; X = P.copy(); V = np.zeros_like(P); dt = 1. / (24. * sub)
    for i in range(n): out[i, 0] = P[i]
    for f in range(1, T):
        for s in range(sub):
            t = (f - 1) / 24. + s * dt; 
            for i in range(n):
                free = (f - 1) >= rel[i]
                for j in range(k):
                    if j == 0 and not free: continue
                    x, y = X[i, j]; n1 = vn3(x * .006 + sd + i * 5.3, y * .006, t * 2.6 + i * 1.7); n2 = vn3(x * .006 + 50. + i * 3.1, y * .006 + sd, t * 2.6 + i * 2.3); tx = X[i, min(j + 1, k - 1), 0] - X[i, max(j - 1, 0), 0]; ty = X[i, min(j + 1, k - 1), 1] - X[i, max(j - 1, 0), 1]; l = math.sqrt(tx * tx + ty * ty) + 1e-6; nx = -ty / l; ny = tx / l
                    ux = wx[f] * (1 + gust * n1) - V[i, j, 0]; uy = wy[f] + abs(wx[f]) * gust * n2 * .9 - V[i, j, 1]; un = ux * nx + uy * ny; V[i, j, 0] += dt * (drag * (un * nx + .22 * ux)); V[i, j, 1] += dt * (drag * (un * ny + .22 * uy) + g)
                for j in range(k):
                    if j == 0 and not free: V[i, 0, 0] = 0.; V[i, 0, 1] = 0.; continue
                    X[i, j, 0] += V[i, j, 0] * dt; X[i, j, 1] += V[i, j, 1] * dt
                for it in range(6):
                    for j in range(k - 1):
                        dx = X[i, j + 1, 0] - X[i, j, 0]; dy = X[i, j + 1, 1] - X[i, j, 1]; l = math.sqrt(dx * dx + dy * dy) + 1e-9; c = (l - seg[i]) / l; wa = 0. if (j == 0 and not free) else .5; wb = 1 - wa if wa == 0. else .5
                        X[i, j, 0] += dx * c * wa; X[i, j, 1] += dy * c * wa; X[i, j + 1, 0] -= dx * c * wb; X[i, j + 1, 1] -= dy * c * wb
                        V[i, j, 0] += dx * c * wa / dt * .9; V[i, j, 1] += dy * c * wa / dt * .9; V[i, j + 1, 0] -= dx * c * wb / dt * .9; V[i, j + 1, 1] -= dy * c * wb / dt * .9
                for j in range(1, k - 1):
                    V[i, j, 0] = V[i, j, 0] * .994 + .003 * (V[i, j - 1, 0] + V[i, j + 1, 0]); V[i, j, 1] = V[i, j, 1] * .994 + .003 * (V[i, j - 1, 1] + V[i, j + 1, 1])
        for i in range(n): out[i, f] = X[i]


class Ribbons:
    """Cloth shredding into ribbons.  rb = Ribbons(roots (n,2) px, dirs (n,2) or angle, length=260, width=34, seed=1); rb.bake(80, wind=lambda f: (900, -60), release=lambda i: 8 + i * 2)
    rb.draw(cel, f, pal=dict(lit, mid, dark, ink))    strips with torn ends, two-tone as they twist, pinned at the root until released, then carried off by the wind (px/s)."""

    def __init__(self, roots, dirs=(0, 1), length=260., width=34., seed=0, k=14):
        self.P0 = np.asarray(roots, float); n = len(self.P0); D = np.broadcast_to(np.asarray(dirs, float), (n, 2)).copy(); D /= np.linalg.norm(D, axis=1)[:, None]; r = R(seed, 91); self.L = length * r.uniform(.55, 1.25, n) if np.isscalar(length) else np.asarray(length, float); self.w = (width * r.uniform(.6, 1.35, n)); self.k, self.seed, self.n = k, seed, n; self.rest = self.P0[:, None] + D[:, None] * (np.linspace(0, 1, k)[None, :, None] * self.L[:, None, None])

    def bake(self, frames=80, wind=(700., 0.), release=None, g=700., drag=7., gust=.55, sub=10):
        T = frames + 1; w = np.array([wind(f) if callable(wind) else wind for f in range(T)], float); rel = np.array([1e9 if release is None else release(i) for i in range(self.n)], float); self.rel = rel; self.X = np.zeros((self.n, T, self.k, 2)); _chain(self.rest, self.n, T, sub, self.L / (self.k - 1), w[:, 0].copy(), w[:, 1].copy(), g, drag, rel, gust, self.seed * 3.7, self.X); return self

    def draw(self, cel, f, pal, lw=2.2, offset=(0, 0), alpha=1.):
        f = int(min(max(f, 0), self.X.shape[1] - 1)); N_ = self.k
        for i in range(self.n):
            Q = spl(self.X[i, f] + offset, n=3); m = len(Q); v = np.linspace(0, 1, m); T = np.gradient(Q, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; Nn = np.c_[-T[:, 1], T[:, 0]]; tw_ = np.cos(v * (2.2 + 3 * h1(self.seed, i, 1)) + f * .30 * (.6 + h1(self.seed, i, 2)) + i) ; hw = self.w[i] * .5 * (1 - .55 * v ** 2) * (.34 + .66 * np.abs(tw_)); jag = np.array([h1(self.seed, i, int(q * 9), 3) for q in v]); hw = hw * (1 - .22 * jag * (v > .12)) * (1. if f < self.rel[i] else np.minimum(1., .18 + v * 5.)); A, B = Q + Nn * hw[:, None], Q - Nn * hw[:, None]
            tipn = max(3, m // 7); e = Q[-1]; Z = [A[-tipn], A[-tipn // 2] * .5 + e * .5 + T[-1] * self.w[i] * .25, Q[-tipn // 2 - 1], e + T[-1] * self.w[i] * .30, Q[-tipn // 2] * .6 + B[-tipn // 2] * .4, B[-tipn]]; Pl = np.vstack([A[:-tipn], Z, B[:-tipn][::-1]]); c = Pl.mean(0); cel.fill(Pl + [.0, 0], pal['mid'], alpha)
            run = tw_ > 0; st = 0                             # back side where the strip turns over: dark; crest light
            for j in range(1, m):
                if run[j] != run[j - 1] or j == m - 1:
                    sl = slice(max(0, st - 0), j + 1)
                    if not run[st] and j - st > 1: cel.fill(np.vstack([A[sl], B[sl][::-1]]), pal['dark'], alpha)
                    elif j - st > 1: cel.fill(np.vstack([A[sl], (A[sl] * .62 + B[sl] * .38)[::-1]]), pal['lit'], alpha)
                    st = j
            for E in (A, B): cel.line(E[:-2], lw, pal['ink'], taper=(.1, .5), press=.4, seed=self.seed + i, tremor=.1, alpha=alpha)
        return cel


# ------------------------------------------------------------- ash -> petals of light
_ASH = np.array([[-.95, -.05], [-.55, -.48], [-.1, -.28], [.35, -.62], [.95, -.1], [.55, .22], [.22, .62], [-.3, .30], [-.62, .46]])
_PET = spl(np.array([[0, 1.0], [-.40, .60], [-.56, 0.], [-.42, -.58], [0, -.92], [.42, -.58], [.56, 0.], [.40, .60]]), True, corners=(0,), n=3)


def _blend_shape(k):
    a = np.linspace(0, 1, 40, endpoint=False); rs = lambda S: (lambda Q, d: np.c_[np.interp(a * d[-1], d, Q[:, 0]), np.interp(a * d[-1], d, Q[:, 1])])(np.vstack([S, S[:1]]), np.r_[0, np.cumsum(np.hypot(*np.diff(np.vstack([S, S[:1]]), axis=0).T))]); return rs(_ASH[[6, 7, 8, 0, 1, 2, 3, 4, 5]]) * (1 - k) + rs(_PET) * k


def flakes(f, n=90, morph=0., seed=0, wind=(-40., 60.), size=(3, 20), sway=46., spin=1.5, ash=('#8c8a90', '#55535c', '#2a2830'), petal=('#fffdf4', '#ffe6b0', '#ffb9a0'), glow='#ffd9a0', glow_amt=.8, box=None, near=5, step=2, w=W, h=H, centre=None, alpha=1.):
    """falling ash that becomes petals of light.  morph 0..1 (or a function of u 0..1): 0 torn ash, 1 luminous petals; every flake tumbles in 3-D on twos, keeps its identity.
    centre = (x, y, r) : flakes inside that radius turn first (a light that spreads).  -> (premultiplied RGBA layer, additive light RGB)"""
    t = tw(f, step) / 24.; cel = Cel(w, h); sf = Cel(w // 2, h // 2, ss=1); Lg = np.zeros((h // 4, w // 4), f32); x0, y0, x1, y1 = box or (-120, -120, w + 120, h + 120); bw, bh = x1 - x0, y1 - y0; it = []
    for i in range(n):
        z = h1(seed, i, 1); nr = i >= n - near; s = size[0] + (size[1] - size[0]) * z ** 2.; dep = .35 + .75 * z
        if nr: s = size[1] * (1.8 + 1.4 * h1(seed, i, 2)); dep = 1.8
        x = x0 + (h1(seed, i, 3) * bw + wind[0] * dep * t + sway * dep * math.sin(t * (.8 + h1(seed, i, 4)) + i * 2.1)) % bw; y = y0 + (h1(seed, i, 5) * bh + wind[1] * dep * t * (.75 + .5 * h1(seed, i, 6)) + sway * .35 * dep * math.sin(t * 1.9 + i)) % bh; u = h1(seed, i, 7); k = morph(u) if callable(morph) else morph
        if centre is not None: d = math.hypot(x - centre[0], y - centre[1]); k = clamp((centre[2] - d) / 160. + .5 * (u - .5) + k * 0)
        it.append((dep, i, x, y, s, clamp(float(k)), nr))
    it.sort(key=lambda q: q[0])
    for dep, i, x, y, s, k, nr in it:
        S = _blend_shape(smooth(0., 1., k)); w_ = spin * (.4 + h1(seed, i, 8)) * (1 - .35 * k); M = axis_m([1, .3, .2], t * w_ + i) @ axis_m([0, .4, 1], t * w_ * .7 + i * 2.2); bend = .30 * k + .12; P3 = np.c_[S, bend * (S[:, 0] ** 2 + .5 * S[:, 1] ** 2 - .4)] @ M.T; P = P3[:, :2] * (s / (1 - .2 * P3[:, 2:3])) + [x, y]; nz = abs(M[2, 2]); a_, b_ = [hexmix(ash[j], petal[j], k) for j in range(3)], None; col = a_[0] if nz > .62 else a_[1] if nz > .28 else a_[2]
        if nr: sf.fill(P / 2, col, .55 * alpha)
        else:
            cel.fill(P, col, alpha)
            if s > 7: m = len(P) // 2; cel.fill(np.vstack([P[m:], P[:1], [[x, y]]]), a_[1] if nz > .62 else a_[2], .9 * alpha); thin(cel, np.array([P[0], [x, y], P[m]]), max(.8, s * .06), a_[2] if k < .5 else hexmix(petal[1], '#ffffff', .4), .8 * alpha, .3, .6, 10)
        if k > .05:
            gx, gy = int(x / 4), int(y / 4); rr = int(max(2, s * .45 * (1 + nr)))
            if rr < gx < Lg.shape[1] - rr and rr < gy < Lg.shape[0] - rr: yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]; Lg[gy - rr:gy + rr + 1, gx - rr:gx + rr + 1] += (np.exp(-(xx ** 2 + yy ** 2) / (rr * rr * .22)) * k * (.5 + .5 * nz) * (.25 if nr else 1.)).astype(f32)
    lay = cel.layer(h, w)
    if near: q = cv2.GaussianBlur(sf.layer(h // 2, w // 2), (0, 0), 3.2); q = cv2.resize(q, (w, h), interpolation=cv2.INTER_LINEAR); lay = lay * (1 - q[..., 3:4]) + q
    return lay, cv2.resize(Lg, (w, h), interpolation=cv2.INTER_LINEAR)[..., None] * C(glow) * glow_amt


# ------------------------------------------------------------- a ring through grass
def grass_ring(t, centre=(0., 0.), speed=3.2, width=.35, life=2.2):
    """-> bend(P (n,2) world xy) -> (n,2) outward bend vector 0..1.2 for a small ring that runs through grass (h3).  Use it on your own grass, or with grass_field below"""
    c = np.asarray(centre, float); R_ = .10 + speed * t * (1 - .12 * t); amp = math.exp(-t * 1.5) if t > 0 else 0.
    def bend(P):
        d = np.asarray(P, float) - c; l = np.linalg.norm(d, axis=-1) + 1e-6; s = (l - R_) / width; e = np.where(s > 0, np.exp(-(s * 2.2) ** 2), np.exp(-(s * .8) ** 2) * np.cos(np.clip(-s, 0, 3) * 1.6)); return d / l[..., None] * (e * amp)[..., None]
    return bend


def grass_field(cel, cam, bend, rows=(-3., 12., 60), width=1.1, dens=70, seed=0, h=(.10, .24), pal=('#8fbf7a', '#5d9a68', '#35705c', '#e9f6c8'), wind=None, t=0., lw=0.):
    """simple drawn grass blades, far to near, bent by `bend`; the turned-over side flashes light on the ring"""
    y0, y1, nr = rows
    for j in range(nr - 1, -1, -1):
        v = j / (nr - 1); y = y0 + (y1 - y0) * v ** 1.8; hw = width * max(.2, float(cam.cam(np.array([0, y, 0.]))[2])) * cam.w * .5 / cam.f; n = int(dens * (1 + .8 * v)); X = np.linspace(-hw, hw, n) + np.array([h1(seed, j, i, 1) - .5 for i in range(n)]) * 2 * hw / n * 1.6; Yy = y + np.array([h1(seed, j, i, 2) - .5 for i in range(n)]) * (y1 - y0) / nr; B = bend(np.c_[X, Yy])
        for i in range(n):
            hh = h[0] + (h[1] - h[0]) * h1(seed, j, i, 3); wv = 0. if wind is None else wind * math.sin(t * 2.2 + X[i] * 1.3 + y * .8); bx, by = B[i]; ln = (h1(seed, j, i, 4) - .5) * .5 + wv; tip = np.array([X[i] + hh * (ln + bx * 1.5), Yy[i] + hh * by * 1.5, hh * (1 - .42 * min(1., math.hypot(bx, by)))]); mid = np.array([X[i] + hh * (ln + bx * 1.2) * .35, Yy[i] + hh * by * .4, hh * .60]); q, z = cam.project(np.array([[X[i], Yy[i], 0], mid, tip]))
            if z.min() < .3 or not -40 < q[0, 0] < cam.w + 40: continue
            w_ = max(.8, hh * .085 * cam.f / z[0]); fl = math.hypot(bx, by); c = pal[3] if fl > .45 else pal[0] if h1(seed, j, i, 5) > .55 else pal[1] if h1(seed, j, i, 6) > .35 else pal[2]; cel.fill(np.array([q[0] - [w_, 0], q[1] - [w_ * .8, 0], q[2], q[1] + [w_ * .8, 0], q[0] + [w_, 0]]), c)
    return cel
