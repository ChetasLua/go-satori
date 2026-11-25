"""phys rock: fractured convex meshes, real rigid bodies (ground contact, friction, rest), ground plates, pillars, trench.  (PHYS lane)"""
from .phys_core import *
from scipy.spatial import HalfspaceIntersection


# ------------------------------------------------------------- meshes
class Mesh:
    __slots__ = ('V', 'F', 'N', 'tag', 'A', 'b')

    def __init__(self, V, F, N, tag, A=None, b=None): self.V, self.F, self.N, self.tag, self.A, self.b = V, F, N, tag, A, b

    def scaled(self, s):
        s = np.broadcast_to(np.asarray(s, float), (3,))
        N = self.N / s; N /= np.linalg.norm(N, axis=1)[:, None]
        return Mesh(self.V * s, self.F, N, self.tag)

    @property
    def rad(self): return float(np.linalg.norm(self.V, axis=1).max())


def solid(A, b, inside, tag=None):
    """convex polyhedron {Ax <= b} -> Mesh recentred on its vertex mean; returns (mesh, centre) or None"""
    A = np.asarray(A, float); b = np.asarray(b, float)
    try: hs = HalfspaceIntersection(np.c_[A, -b], np.asarray(inside, float))
    except Exception: return None
    P = hs.intersections; P = P[np.isfinite(P).all(1)]
    if len(P) < 4: return None
    ext = np.ptp(P, axis=0).max() + 1e-9; key = np.round(P / ext * 4e3).astype(np.int64); _, ix = np.unique(key, axis=0, return_index=True); P = P[np.sort(ix)]
    F, N, T = [], [], []
    for i in range(len(A)):
        n = A[i] / (np.linalg.norm(A[i]) + 1e-12); on = np.where(np.abs(P @ A[i] - b[i]) < 2e-3 * ext)[0]
        if len(on) < 3: continue
        Q = P[on]; c = Q.mean(0); u = Q[0] - c; u /= np.linalg.norm(u) + 1e-12; v = np.cross(n, u); a = np.arctan2((Q - c) @ v, (Q - c) @ u); o = np.argsort(a)
        F.append(on[o]); N.append(n); T.append(0 if tag is None else tag[i])
    if len(F) < 4: return None
    c = P.mean(0); return Mesh(P - c, F, np.array(N), np.array(T), A, b - A @ c), c


_ck = {}


def chunk(seed, kind='block'):
    """unit broken rock: box split by tilted planes.  kind: block, shard, slab"""
    seed = int(seed) % 64; key = (seed, kind)
    if key in _ck: return _ck[key]
    r = R(seed, 41); dim = dict(block=(1., .78, .70), shard=(1.15, .36, .30), slab=(1., .78, .20))[kind] * r.uniform(.8, 1.2, 3)
    A, b, tg = [], [], []
    for ax in range(3):
        for s in (-1, 1):
            n = np.zeros(3); n[ax] = s; n = unit(n + r.normal(0, .20 if kind != 'slab' or ax < 2 else .05, 3)); A.append(n); b.append(dim[ax] * r.uniform(.8, 1.)); tg.append(int(r.random() < .55))
    for _ in range(int(r.integers(4, 8)) if kind != 'slab' else 4):
        n = r.normal(0, 1, 3); n[2] *= .35 if kind == 'slab' else 1; n = unit(n); sup = float(np.abs(n) @ dim); A.append(n); b.append(sup * r.uniform(.60, .84)); tg.append(int(r.random() < .7))
    q = solid(A, b, np.zeros(3), tg)
    m = q[0]; s = 1 / np.linalg.norm(m.V, axis=1).max(); _ck[key] = m.scaled(s); return _ck[key]


def box_planes(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float); A = np.vstack([np.eye(3), -np.eye(3)]); return A, np.r_[hi, -lo]


def hull_planes(P):
    """planes of the convex hull of points"""
    from scipy.spatial import ConvexHull
    q = ConvexHull(P).equations; k = np.unique(np.round(q, 5), axis=0); return k[:, :3], -k[:, 3]


def shatter(A, b, pts):
    """true Voronoi fracture of {Ax<=b}: cells tile the solid exactly -> [(mesh, centre)].  Faces on the old skin keep tag 0, break faces tag 1"""
    A = np.asarray(A, float); b = np.asarray(b, float); pts = np.asarray(pts, float); out = []
    for i, p in enumerate(pts):
        d = np.linalg.norm(pts - p, axis=1); nb = np.argsort(d)[1:18]; n = pts[nb] - p; l = np.linalg.norm(n, axis=1)[:, None]; n = n / l; bb = np.einsum('ij,ij->i', n, (pts[nb] + p) / 2)
        q = solid(np.vstack([A, n]), np.r_[b, bb], p, np.r_[np.zeros(len(A), int), np.ones(len(nb), int)])
        if q is not None: out.append(q)
    return out


# ------------------------------------------------------------- rigid bodies
@njit(cache=True, fastmath=True)
def _sim(V, nv, X0, Q0, V0, W0, I, t0, fl, e, mu, dr, G, sub, T, XO, QO):
    dt = 1. / (24. * sub)
    for i in range(X0.shape[0]):
        x, y, z = X0[i]; q0, q1, q2, q3 = Q0[i]; vx, vy, vz = V0[i]; wx, wy, wz = W0[i]; iI = 1. / I[i]
        XO[i, 0] = X0[i]; QO[i, 0] = Q0[i]; kd = math.exp(-dr[i] * dt)
        for k in range(1, T):
            for s in range(sub):
                if (k - 1) / 24. + s * dt < t0[i]: continue
                vz -= G[(k - 1) * sub + s] * dt; vx *= kd; vy *= kd; vz *= kd
                x += vx * dt; y += vy * dt; z += vz * dt
                a0 = -.5 * dt * (wx * q1 + wy * q2 + wz * q3); a1 = .5 * dt * (wx * q0 + wy * q3 - wz * q2); a2 = .5 * dt * (wy * q0 + wz * q1 - wx * q3); a3 = .5 * dt * (wz * q0 + wx * q2 - wy * q1)
                q0 += a0; q1 += a1; q2 += a2; q3 += a3; l = 1 / math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3); q0 *= l; q1 *= l; q2 *= l; q3 *= l
                r00 = 1 - 2 * (q2 * q2 + q3 * q3); r01 = 2 * (q1 * q2 - q0 * q3); r02 = 2 * (q1 * q3 + q0 * q2); r10 = 2 * (q1 * q2 + q0 * q3); r11 = 1 - 2 * (q1 * q1 + q3 * q3); r12 = 2 * (q2 * q3 - q0 * q1); r20 = 2 * (q1 * q3 - q0 * q2); r21 = 2 * (q2 * q3 + q0 * q1); r22 = 1 - 2 * (q1 * q1 + q2 * q2)
                pen = 0.; hit = 0
                for j in range(nv[i]):
                    a, b, c = V[i, j]; rz = r20 * a + r21 * b + r22 * c; d = z + rz - fl[i]
                    if d < 0:
                        if -d > pen: pen = -d
                        rx = r00 * a + r01 * b + r02 * c; ry = r10 * a + r11 * b + r12 * c
                        ux = vx + wy * rz - wz * ry; uy = vy + wz * rx - wx * rz; uz = vz + wx * ry - wy * rx; hit += 1
                        if uz < 0:
                            Kn = 1 + iI * (rx * rx + ry * ry); jn = -(1 + (e if uz < -.6 else 0.)) * uz / Kn; vz += jn; wx += iI * ry * jn; wy -= iI * rx * jn
                            ut = math.sqrt(ux * ux + uy * uy)
                            if ut > 1e-5:
                                Kt = 1 + iI * (rx * rx + ry * ry + rz * rz) * .8; jt = min(mu * jn, ut / Kt); fx = -ux / ut * jt; fy = -uy / ut * jt; vx += fx; vy += fy
                                wx += iI * (-rz * fy); wy += iI * (rz * fx); wz += iI * (rx * fy - ry * fx)
                if hit:
                    z += pen * .6; wx *= .992; wy *= .992; wz *= .985
                    if vx * vx + vy * vy + vz * vz < .05 and wx * wx + wy * wy + wz * wz < .25: vx *= .85; vy *= .85; wx *= .85; wy *= .85; wz *= .85
            XO[i, k, 0] = x; XO[i, k, 1] = y; XO[i, k, 2] = z; QO[i, k, 0] = q0; QO[i, k, 1] = q1; QO[i, k, 2] = q2; QO[i, k, 3] = q3


class Rubble:
    """Rigid rock bodies, metres.  rb = Rubble(); rb.add(mesh, x, v, w, t0) ... ; rb.bake(frames); rb.draw(cel, cam, f, pal)
    Convenience: rb.burst(origin, n, speed, up, spread, size) throws three size classes.  Real contact with the floor, friction, rest."""

    def __init__(self, seed=0): self.b = []; self.seed = seed; self.X = None

    def add(self, mesh, x, v=(0, 0, 0), w=(0, 0, 0), t0=0., q=(1, 0, 0, 0), floor=0., drag=None, seed=None):
        ext = mesh.rad; self.b.append((mesh, np.asarray(x, float), np.asarray(v, float), np.asarray(w, float), t0, q, floor, (.06 + .10 / (ext + .12)) if drag is None else drag, len(self.b) + self.seed * 131 if seed is None else seed)); return self

    def burst(self, origin, n=40, speed=(4, 14), up=.8, size=(.08, .9), dirn=None, spread=1.0, t0=0., seed=0, kinds=('block', 'shard', 'slab'), jit=(.3, .3, .1), spin=7., dt=.06, floor=0.):
        """n chunks in three size classes.  dirn = mean throw direction (x, y, z) or None (hemisphere); spread = half cone (rad-ish); up = extra lift fraction; small ones fly faster"""
        o = np.asarray(origin, float)
        for i in range(n):
            r = R(seed, i, 77); z = size3(r.random()); s = size[0] + (size[1] - size[0]) * z
            if dirn is None: a = r.uniform(0, TAU); d = unit([math.cos(a), math.sin(a), abs(r.normal(up, .5)) + .1])
            else: d = unit(unit(dirn) + r.normal(0, spread * .6, 3) + [0, 0, up * .5])
            v = d * r.uniform(*speed) * (1.35 - .75 * z); ax = unit(r.normal(0, 1, 3)); qa = unit(r.normal(0, 1, 4))
            self.add(chunk(int(r.integers(64)), kinds[int(r.random() ** 1.3 * len(kinds)) % len(kinds)]).scaled(s * .5), o + r.normal(0, 1, 3) * np.asarray(jit) + [0, 0, s * .4], v, ax * r.uniform(.3, 1.) * spin / (s + .35), t0 + r.random() * dt, qa, floor)
        return self

    def bake(self, frames=96, g=9.8, e=.28, mu=.55, sub=8):
        """simulate once (call in setup).  g = number or g(t seconds) -> m/s2 (may go negative: debris that stops obeying gravity)"""
        n = len(self.b); T = int(frames) + 1
        if not n: self.X = np.zeros((0, T, 3)); return self
        m = max(len(q[0].V) for q in self.b); V = np.zeros((n, m, 3)); nv = np.zeros(n, np.int64); I = np.zeros(n)
        for i, q in enumerate(self.b): k = len(q[0].V); V[i, :k] = q[0].V; nv[i] = k; I[i] = max(1e-6, .8 * float((q[0].V ** 2).sum(1).mean()) * .55)
        ts = (np.arange((T - 1) * sub) / (24. * sub)); G = np.broadcast_to(np.vectorize(g)(ts) if callable(g) else float(g), ts.shape).astype(float).copy()
        Q0 = np.array([q[5] for q in self.b], float); Q0 /= np.linalg.norm(Q0, axis=1)[:, None]
        self.X = np.zeros((n, T, 3)); self.Q = np.zeros((n, T, 4)); self.t0 = np.array([q[4] for q in self.b], float)
        _sim(V, nv, np.array([q[1] for q in self.b]), Q0, np.array([q[2] for q in self.b]), np.array([q[3] for q in self.b]), I, self.t0, np.array([q[6] for q in self.b], float), e, mu, np.array([q[7] for q in self.b], float), G, sub, T, self.X, self.Q); return self

    def pose(self, f):
        T = self.X.shape[1]; f = min(max(float(f), 0.), T - 1.001); k = int(f); u = f - k; X = self.X[:, k] * (1 - u) + self.X[:, k + 1] * u
        a, b = self.Q[:, k], self.Q[:, k + 1]; s = np.where((a * b).sum(1) < 0, -1., 1.)[:, None]; q = a * (1 - u) + b * s * u; q /= np.linalg.norm(q, axis=1)[:, None]; return X, quat_m(q), (self.X[:, k + 1] - self.X[:, k]) * 24.

    def items(self, f, hide=False):
        """[(mesh, centre, rot, seed, vel, moved)] for draw_bodies; hide = bodies not yet released are left out"""
        X, M, V = self.pose(f); return [(q[0], X[i], M[i], q[8], V[i], f / 24. >= q[4]) for i, q in enumerate(self.b) if not hide or f / 24. >= q[4]]

    def draw(self, cel, cam, f, pal='night', step=1, hide=True, **k):
        return draw_bodies(cel, cam, self.items(tw(f, step) if step > 1 else f, hide), pal, **k)


def _pal(p, tab=ROCK): return tab[p] if isinstance(p, str) else p


def draw_bodies(cel, cam, items, pal='night', light=(-.45, -.35, .80), rim_dir=None, lw=2.4, ink=True, shadow=None, shadow_dir=None, floor=0., detail=1., trails=.0, skin=None, min_px=1.2, fog=None, sort=True, sky=.16):
    """Cel-shade convex bodies far to near.  light = direction TO the key light (world).  rim_dir = direction to a second, hard rim light.
    shadow = '#hex' casts a hard ground shadow along shadow_dir (default -light).  trails = speed wedges (0..1).  skin = palette for tag-0 faces.
    fog = (rgb/hex, start m, density)."""
    pal = _pal(pal); skin = pal if skin is None else _pal(skin); L = unit(light); Rm = None if rim_dir is None else unit(rim_dir); eye = cam.eye; ob = []
    for it in items:
        m, x, M = it[0], it[1], it[2]; c = cam.cam(x)
        if c[2] < .15: continue
        s = cam.f / c[2]; r = m.rad * s; p = c[:2] / c[2] * cam.f + [cam.cx, cam.cy]
        if r < min_px * .5 or p[0] < -r - 60 or p[0] > cam.w + r + 60 or p[1] < -r - 200 or p[1] > cam.h + r + 60: continue
        ob.append((c[2] if len(it) < 7 else cam.cam(it[6])[2], it, s, r, p))
    if sort: ob.sort(key=lambda q: -q[0])
    if shadow is not None:
        sd = -L if shadow_dir is None else unit(shadow_dir); k = sd[:2] / min(sd[2], -.12)
        for z, it, s, r, p in ob:
            V = it[0].V @ it[2].T + it[1]; hgt = V[:, 2] - floor
            if hgt.min() > 14 or r < 3: continue
            G = np.c_[V[:, :2] - hgt[:, None] * k, np.full(len(V), floor)]; q, zz = cam.project(G)
            if (zz < .2).any(): continue
            cel.fill(hull2(q), shadow, .92 * max(.0, 1 - hgt.min() / 14))
    for z, it, s, r, p in ob:
        m, x, M, sd = it[:4]; V = m.V @ M.T + x; P, zz = cam.project(V)
        if (zz < .1).any(): continue
        fg = 0. if fog is None else 1 - math.exp(-max(0., z - fog[1]) * fog[2]); T = (lambda c: C(c)) if fg <= 0 else (lambda c: hexmix(c, fog[0], fg)); ik = T(pal['ink'])
        if trails and len(it) > 4:
            sp = np.linalg.norm(it[4]); d = P.mean(0) - cam.project(x - it[4] * .035 * trails)[0]; l = math.hypot(*d)
            if l > r * .9 and r < 60: n = np.array([-d[1], d[0]]) / l; q = P.mean(0); cel.fill(np.array([q + n * r * .55, q - d * 1.0, q - n * r * .55]), T(pal['rim']), .34)
        Hl = hull2(P); w = min(lw, .7 + r * .07)
        if r < 3.2: cel.fill(Hl, T(pal['dark'] if h1(sd, 1) < .6 else pal['mid'])); continue
        if ink: ld = np.array([L[0], -L[2]]); ld /= np.hypot(*ld) + 1e-6; cel.fill(grow(Hl, lambda n: w * (.30 + .85 * np.clip(-(n @ ld), 0, 1))), ik)
        cel.fill(Hl, T(pal['mid'])); vis = []; N = m.N @ M.T
        for j, ix in enumerate(m.F):
            q = V[ix]; n = N[j]
            if n @ (eye - q.mean(0)) <= 0: continue
            d = float(n @ L) + .10 * (h1(sd, j, 3) - .5) + sky * n[2]; pl = skin if m.tag[j] == 0 else pal
            col = pl['lit'] if d > .46 else pl['mid'] if d > .02 else pl['dark']
            if m.tag[j] == 1 and .02 < d <= .46: col = pl['fresh']
            if Rm is not None and float(n @ Rm) > .80 and d < .30: col = pl['bounce']
            vis.append((j, ix, n, d)); F = P[ix]; cel.fill(F, T(col))
            if r > 26 * detail ** -1 and detail > 0: _face_detail(cel, F, d, pl, T, w, sd * 7 + j, r)
        if r > 7:                                            # inner edges: hard, broken, tapered; rim on edges that face the light
            E = {}
            for j, ix, n, d in vis:
                for a, b in zip(ix, np.roll(ix, -1)): E.setdefault((min(a, b), max(a, b)), []).append((j, n, d))
            for (a, b), fs in E.items():
                if len(fs) < 2: 
                    if Rm is not None and r > 12:
                        n = fs[0][1]
                        if float(n @ Rm) > .45: A_, B_ = P[a], P[b]; nn = np.array([-(B_ - A_)[1], (B_ - A_)[0]]); nn /= np.hypot(*nn) + 1e-6; ci = P.mean(0); nn = nn if nn @ (ci - A_) > 0 else -nn; o = nn * w * .55; wedge(cel, A_ + o, B_ + o, w * 1.1, w * .3, T(pal.get('rim2', pal['rim'])))
                    continue
                dd = abs(fs[0][2] - fs[1][2]); ang = 1 - float(fs[0][1] @ fs[1][1]); u = h1(sd, a, b)
                if ang < .10 or (u < .25 and dd > .3): continue
                A_, B_ = P[a], P[b]
                if u > .6: A_, B_ = B_, A_
                lit = max(fs[0][2], fs[1][2]) > .40
                if lit and r > 14: wedge(cel, A_, A_ + (B_ - A_) * (.55 + .45 * h1(sd, a, b, 2)), w * .75, w * .1, T(pal['rim']), .9)
                else: wedge(cel, A_, A_ + (B_ - A_) * (.5 + .5 * h1(sd, a, b, 2)), w * .70, w * .12, ik, .95)
    return cel


def _face_detail(cel, F, d, pl, T, w, sd, r):
    """chips, strata ticks and a crack on one big face (fixed to the face: barycentric)"""
    c = F.mean(0); ar = cv2.contourArea(F.astype(np.float32))
    if ar < 700: return
    n = len(F); cel.clip(F); dn = pl['mid'] if d > .46 else pl['dark'] if d > .02 else pl['ink']; up = pl['rim'] if d > .46 else pl['lit'] if d > .02 else pl['mid']; k = int(1 + h1(sd, 1) * 2.4)
    for i in range(k):                                       # chipped corners: small polygons biting in from an edge
        e = int(h1(sd, i, 2) * n); a, b = F[e], F[(e + 1) % n]; u0 = h1(sd, i, 3) * .7; u1 = u0 + .10 + .22 * h1(sd, i, 4); dp = .10 + .20 * h1(sd, i, 5); q0, q1 = a + (b - a) * u0, a + (b - a) * u1; m = (q0 + q1) / 2; iv = c - m
        cel.fill(np.array([q0, q0 + iv * dp * .7 + (b - a) * .02, m + iv * dp * 1.25, q1 + iv * dp * .6, q1]), T(dn if h1(sd, i, 6) < .62 else up))
    if ar > 2600:                                            # strata ticks along the face's long axis, lifting pen
        a, b = F[0], F[n // 2]; ax = b - a; ax /= np.hypot(*ax) + 1e-6; nr = np.array([-ax[1], ax[0]]); sz = math.sqrt(ar)
        for i in range(int(2 + 3 * h1(sd, 8))):
            p = c + nr * sz * (h1(sd, i, 9) - .5) * .8 + ax * sz * (h1(sd, i, 10) - .5) * .6; l = sz * (.12 + .30 * h1(sd, i, 11)); wedge(cel, p, p + ax * l + nr * l * .12 * (h1(sd, i, 12) - .5), w * .62, .2, T(pl['ink']), .85)
        e = int(h1(sd, 13) * n); p0 = F[e] * .7 + F[(e + 1) % n] * .3; Z = zigzag(p0, p0 + (c - p0) * (.8 + .7 * h1(sd, 14)), sd, .17, 3); thin(cel, Z, w * 1.0, T(pl['ink']), .9, .02, .9)
    cel.unclip()


# ------------------------------------------------------------- ground plates and cracks
def _prism(P, th, seed=0, under=.72):
    """polygon (n,2) -> slab mesh hanging below z=0, undercut; returns mesh (centre at origin offset) and centre"""
    P = hull2(P); c = P.mean(0); ar = float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])); P = P if ar > 0 else P[::-1]; n = len(P)
    B = c + (P - c) * under; V = np.vstack([np.c_[P, np.zeros(n)], np.c_[B, np.full(n, -th)]]); F = [np.arange(n), np.arange(n, 2 * n)[::-1]]; N = [np.array([0, 0, 1.]), np.array([0, 0, -1.])]; tg = [0, 1]
    for i in range(n):
        j = (i + 1) % n; q = np.array([i, n + i, n + j, j]); nn = np.cross(V[q[1]] - V[q[0]], V[q[3]] - V[q[0]]); nn = unit(nn); nn = nn if nn[:2] @ (V[i, :2] + V[j, :2] - 2 * c) > 0 else -nn; F.append(q); N.append(nn); tg.append(1)
    m = np.array([c[0], c[1], -th * .45]); return Mesh(V - m, F, np.array(N), np.array(tg)), m


class Fracture:
    """Ground opens in plates with thickness; cracks race ahead.
    fr = Fracture((0, 0), rad=5, seed=1); fr.draw(cel, cam, t, 'night')      t = seconds since the hit
    tilt radians of the innermost plates (petals open outward), speed m/s of the fracture front, push = outward slide, hold=True plates stay tilted / False they drop back,
    lift = None or function(u 0..1 random, t) -> metres (plates that float up, f3)."""

    def __init__(self, centre=(0., 0.), rad=5., seed=0, n=13, rings=4, thick=.34, tilt=.85, speed=34., push=.35, kick=.35, hold=True, squash=1., cracks=1.0, r0=.14):
        self.c = np.asarray(centre, float); self.rad, self.seed, self.tilt, self.speed, self.push, self.kick, self.hold = rad, seed, tilt, speed, push, kick, hold; r = R(seed, 3); self.cells = []; self.th = thick
        th = (np.arange(n) + r.uniform(-.36, .36, n)) / n * TAU; rr = r0 + (1 - r0) * (np.linspace(0, 1, rings + 1) ** 1.25); V = np.zeros((rings + 1, n, 2))
        for k in range(rings + 1):
            for i in range(n): q = rad * rr[k] * r.uniform(.80, 1.24); a = th[i] + r.normal(0, .06); V[k, i] = (math.cos(a) * q, math.sin(a) * q * squash)
        V[0] *= .30
        for i in range(n):
            j = (i + 1) % n; k = 0
            while k < rings:
                st = 2 if (k + 2 <= rings and r.random() < .30) else 1; q = [V[k, i], V[k + st, i], V[k + st, j], V[k, j]]
                if r.random() < .42 and k > 0:
                    u = r.uniform(.35, .65); m0, m1 = q[0] * (1 - u) + q[3] * u, q[1] * (1 - u) + q[2] * u; self._cell([q[0], q[1], m1, m0], r); self._cell([m0, m1, q[2], q[3]], r)
                else: self._cell(q, r)
                k += st
        self.sp = []
        for i in range(int(n * 1.0)):
            a = th[i % n] + r.normal(0, .12); l = rad * (1.05 + 1.5 * r.random() ** 2) * cracks; p0 = np.array([math.cos(a), math.sin(a) * squash]) * rad * .55; p1 = np.array([math.cos(a), math.sin(a) * squash]) * l; Z = zigzag(p0, p1, seed * 7 + i, .13, 4); br = []
            for j in range(int(r.integers(1, 4))):
                k = int(r.integers(3, len(Z) - 2)); d = unit(Z[k] - Z[k - 1]); s = r.choice([-1, 1]); aa = s * r.uniform(.5, 1.1); d = np.array([d[0] * math.cos(aa) - d[1] * math.sin(aa), d[0] * math.sin(aa) + d[1] * math.cos(aa)]); br.append((k / len(Z), zigzag(Z[k], Z[k] + d * l * r.uniform(.10, .26), seed + i * 13 + j, .15, 3)))
            self.sp.append((a, l, Z, br, r.uniform(.6, 1.2)))

    def _cell(self, q, r):
        P = np.array(q); c = P.mean(0); P = c + (P - c) * r.uniform(.955, .985); d = float(np.hypot(*c)); size = math.sqrt(max(cv2.contourArea(P.astype(np.float32)), 1e-4))
        self.cells.append(dict(P=P, c=c, r=d, th=self.th * (.45 + .75 * min(1.6, size / (self.rad * .22))) * r.uniform(.8, 1.2), u=r.random(3), sd=len(self.cells) + self.seed * 97))

    def pose(self, cell, t):
        """-> (mesh, centre, rot, moved?)"""
        s = t - cell['r'] / self.speed; m, mc = cell.setdefault('m', _prism(cell['P'], cell['th'], cell['sd']))
        u = cell['u']; k = math.exp(-1.15 * (cell['r'] / self.rad) ** 1.3 * 1.5)
        if s <= 0: return m, mc, np.eye(3), False
        a = 1 - math.exp(-s * 11) * (math.cos(s * 19) + .35 * math.sin(s * 19))
        if not self.hold: a *= math.exp(-max(0, s - .25) * 4)
        e = cell['c'] / (cell['r'] + 1e-6); tang = np.array([e[1], -e[0], 0.]); th = -self.tilt * k * (.45 + .9 * u[0]) * a + .12 * (u[1] - .5) * a; M = axis_m(tang, th) @ axis_m([e[0], e[1], 0], (u[1] - .5) * .55 * k * a)
        piv = np.array([*(cell['c'] - e * cell['r'] * .16), -cell['th'] * .5]); x = piv + M @ (mc - piv) + np.array([*(e * self.push * k * (.4 + u[2]) * a), -.10 * self.rad * .25 * k * a + self.kick * k * max(0., math.sin(min(s * 7, math.pi))) * (.3 + u[0])])
        return m, x, M, True

    def draw(self, cel, cam, t, pal='night', light=(-.45, -.35, .80), lw=2.4, skin=None, rim_dir=None, shadow='#0d0b1c', glow=None, lift=None, cracks=True, detail=1., fog=None, crack_w=.17, floor=0.):
        pal = _pal(pal); o = np.array([self.c[0], self.c[1], floor]); g = lambda P, z=0.: cam.project(np.c_[P + self.c, np.full(len(P), z + floor)])[0]
        if cracks: self._cracks(cel, g, t, pal, lw, crack_w, glow)
        it = []; 
        for c in self.cells:
            m, x, M, mv = self.pose(c, t)
            if not mv: continue
            P = c['P']; P = c['c'] + (P - c['c']) * 1.06; z = -c['th'] * .7; Q = g(P); cel.fill(Q, pal['fresh']); cel.clip(Q); cel.fill(Q + [0, min(26., 8 + .16 * np.ptp(Q[:, 1]))], pal['pit']); cel.unclip(); s = t - c['r'] / self.speed
            if glow is not None: q = g(c['c'] + (P - c['c']) * .80, z); cel.fill(q, glow, .85 * math.exp(-c['r'] / self.rad * 1.6))
            if lift is not None: L_ = lift(c['u'][0], s); x = x + [0, 0, L_]; M = axis_m([c['u'][1] - .5, c['u'][2] - .5, .2], min(L_ * .22, 1.2)) @ M
            it.append((m, x + o, M, c['sd'], np.zeros(3)))
        draw_bodies(cel, cam, it, pal, light, rim_dir, lw, True, shadow, None, floor, detail, 0., skin, fog=fog); return cel

    def _cracks(self, cel, g, t, pal, lw, cw, glow):
        polys = []
        for a, l, Z, br, wd in self.sp:
            e = clamp((t * self.speed * 1.35 - self.rad * .35) / l)
            if e <= .02: continue
            polys.append(self._crack_poly(Z, e, cw * wd * self.rad * .24))
            for u, B in br:
                if e > u + .04: polys.append(self._crack_poly(B, clamp((e - u) * 3.5), cw * wd * self.rad * .11))
        dep = self.rad * .05
        for P in polys: draw_crack(cel, g, P, pal, dep, glow)

    def _crack_poly(self, Z, e, w):
        n = max(2, int(math.ceil((len(Z) - 1) * e)) + 1); Q = Z[:n]; v = np.linspace(0, 1, n); T = np.gradient(Q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]; h = (w * (1 - v) ** .75 * (.65 + .7 * np.abs(np.sin(v * 17 + w * 90))))[:, None] + w * .02; return np.vstack([Q + N * h, (Q - N * h * .8)[::-1]])


def draw_crack(cel, g, P, pal, depth=.2, glow=None):
    """crack polygon (world xy, two mirrored sides): black mouth, lit far wall, bright near lip"""
    top = g(P)
    if not np.isfinite(top).all() or np.abs(top).max() > 2e4: return
    m = len(top) // 2; u, d = top[:m], top[m:][::-1][:m]
    if u[:, 1].mean() > d[:, 1].mean(): u, d = d, u
    cel.fill(top, pal['pit']); k = min(.46, .16 + 3. / (np.abs(d - u).max() + 4)); w = u + (d - u) * k; cel.fill(np.vstack([u, w[::-1]]), pal['fresh'])
    if glow is not None: cel.fill(np.vstack([u + (d - u) * .40, (u + (d - u) * .78)[::-1]]), glow, .95)
    if len(d) > 2: thin(cel, d + [0, .9], 1.5, pal['rim'], .60, .15, .8, K=min(40, 2 * len(d)))


# ------------------------------------------------------------- pillars
class Pillar:
    """A rock pillar / mesa that bursts when a body goes through it.
    p = Pillar((4, 6), h=14, r=2.2, seed=3); p.hit(z=8, dirn=(1, .2, 0), t0=.4, speed=22); p.bake(90); p.draw(cel, cam, f, 'night')"""

    def __init__(self, xy=(0, 0), h=12., r=2., seed=0, taper=.72, n=9, lev=None, cap=1.12):
        rr = R(seed, 9); self.xy = np.asarray(xy, float); self.h, self.r, self.seed = h, r, seed; lev = lev or max(3, int(h / (r * 1.5))); zs = np.r_[0, np.sort((np.arange(1, lev) + rr.uniform(-.34, .34, lev - 1)) / lev), 1] * h; th = np.sort((np.arange(n) + rr.uniform(-.34, .34, n)) / n * TAU); jit = rr.uniform(.80, 1.14, n); self.seg = []; lean = np.zeros(2)
        for k in range(lev):
            u = (zs[k] + zs[k + 1]) / 2 / h; p = r * ((1 - u) * 1.12 + u * taper) * (cap if k == lev - 1 else 1.) * rr.uniform(.90, 1.10); tl = rr.uniform(.02, .13, n) * (1 if k < lev - 1 else rr.choice([-1, 1], n) * .7); q = p * jit * rr.uniform(.95, 1.05, n); N = np.c_[np.cos(th) * np.cos(tl), np.sin(th) * np.cos(tl), np.sin(tl)]; zm = (zs[k] + zs[k + 1]) / 2; lean = lean + rr.normal(0, r * .05, 2); tn = unit(np.r_[rr.normal(0, .07, 2), 1.]); bn = unit(np.r_[rr.normal(0, .05, 2), -1.]) if k else np.array([0, 0, -1.]); A = [*N, tn, bn]; c0 = np.r_[lean, zm]; B = [float(v @ c0) + q_ * math.cos(t_) for v, q_, t_ in zip(N, q, tl)] + [float(tn @ np.r_[lean, zs[k + 1] + r * .03]), float(bn @ np.r_[lean, zs[k]])]
            for _ in range(int(rr.integers(1, 4))): v = unit(np.r_[(lambda a: (math.cos(a), math.sin(a)))(rr.uniform(0, TAU)), rr.choice([-.9, 1.1]) * rr.uniform(.5, 1.2)]); A.append(v); B.append(float(v @ c0) + p * rr.uniform(.95, 1.2) + abs(v[2]) * (zs[k + 1] - zs[k]) * .30)       # broken corners
            s_ = solid(np.array(A), np.array(B), c0)
            if s_: self.seg.append((s_[0], s_[1], None, None, zs[k], zs[k + 1]))
        self.o = np.array([*self.xy, 0.]); self.rb = None; self.t0 = 1e9

    def hit(self, z, dirn=(1, 0, 0), t0=0., speed=20., cells=34, reach=1.0, up=.25, seed=0):
        d = unit(dirn); self.t0 = t0; rb = self.rb = Rubble(self.seed * 7 + 1); rr = R(self.seed, 21, seed); self.low = []; R_ = self.r * reach
        for j, (m, c, A, b, z0, z1) in enumerate(self.seg):
            if z1 < z - R_ * .95: self.low.append(j); continue
            if z0 > z + R_ * .95:                             # top: loses its support, keeps turning over the hit
                rb.add(m, c + self.o, d * speed * .05 + [0, 0, -.3], np.cross([0, 0, 1], d) * .5 + rr.normal(0, .15, 3), t0 + .05, seed=j + self.seed * 11, drag=.05); continue
            pts = rr.normal(0, 1, (cells * 8, 3)) * np.ptp(m.V, 0) * rr.choice([.12, .24, .40], (cells * 8, 1), p=[.3, .4, .3]) + np.array([0, 0, np.clip(z - c[2], -.4 * np.ptp(m.V, 0)[2], .4 * np.ptp(m.V, 0)[2])]) * .6; pts = pts[(pts @ m.N[:, :].T <= np.array([(m.V[f] @ n).mean() for f, n in zip(m.F, m.N)]) - .02 * self.r).all(1)][:cells]
            bb = np.array([(m.V[f] @ n).mean() for f, n in zip(m.F, m.N)])
            for k, (cm, cc) in enumerate(shatter(m.N, bb, pts)):
                p = cc + c; rel = p - [0, 0, z]; ax = rel - d * (rel @ d); dist = np.linalg.norm(ax) + 1e-3; fo = math.exp(-(dist / (R_ * .9)) ** 2); ex = (rel @ d) / self.r
                v = d * speed * fo * (.35 + .55 * (ex + 1) * .5 + .25 * rr.random()) + ax / dist * speed * .30 * (.4 + fo) * rr.uniform(.5, 1.2) + [0, 0, speed * up * rr.uniform(-.1, 1.) * (.3 + fo)]; v = v - d * speed * .22 * max(0., -ex) * rr.random()
                rb.add(cm, p + self.o, v, rr.normal(0, 1, 3) * speed * .30 / (cm.rad + .2), t0 + max(0., (ex + 1) * self.r / max(speed, 1) * .5), seed=k + j * 57 + self.seed * 11)
        return self

    def bake(self, frames=96, **k):
        if self.rb: self.rb.bake(frames, **k)
        return self

    def items(self, f):
        if self.rb is None or f / 24. < self.t0: return [(m, c + self.o, np.eye(3), j + self.seed * 11, np.zeros(3), True, self.o) for j, (m, c, *_) in enumerate(self.seg)]
        return [(self.seg[j][0], self.seg[j][1] + self.o, np.eye(3), j + self.seed * 11, np.zeros(3), True, self.o) for j in self.low] + self.rb.items(f)

    def draw(self, cel, cam, f, pal='night', **k): return draw_bodies(cel, cam, self.items(f), pal, **k)


# ------------------------------------------------------------- trench
class Trench:
    """A body ploughs a trench.  tr = Trench(path (k,2) metres, width=2.4, depth=1.1, seed=2); tr.draw(cel, cam, s, 'night')   s = metres ploughed so far.
    tr.rubble(speed_fn) -> Rubble thrown sideways as the front passes (front(t) metres); tr.front(s) -> world xy of the plough front."""

    def __init__(self, path, width=2.4, depth=1.1, bank=.9, seed=0, step=None):
        P = np.asarray(path, float); d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; self.L = d[-1]; self.P = P; self.d = d; self.w, self.dp, self.bk, self.seed = width * .5, depth, bank, seed; self.step = step or width * .42

    def at(self, s):
        s = np.asarray(s, float); p = np.stack([np.interp(s, self.d, self.P[:, 0]), np.interp(s, self.d, self.P[:, 1])], -1); e = .05; q = np.stack([np.interp(s + e, self.d, self.P[:, 0]), np.interp(s + e, self.d, self.P[:, 1])], -1) - np.stack([np.interp(s - e, self.d, self.P[:, 0]), np.interp(s - e, self.d, self.P[:, 1])], -1); q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-9); return p, np.stack([-q[..., 1], q[..., 0]], -1)

    def front(self, s): return self.at(min(s, self.L))[0]

    def section(self, s, sf):
        """8 cross points (x, y, z) of the profile at arc s when the front is at sf"""
        k = int(round(s / self.step)); sd = self.seed; w = self.w; p, n = self.at(s); tip = clamp((sf - s) / (w * 2.2)); e = math.sqrt(max(0., 1 - (1 - tip) ** 2)); st = clamp(s / (w * 1.5) + .12)
        j = lambda q, a: 1 + a * (h1(sd, k, q) - .5) * 2
        o = np.array([-(w * 1.62 + self.bk) * j(1, .13), -w * 1.16 * j(2, .10), -w * .90 * j(3, .07), -w * .46 * j(4, .12), w * .46 * j(5, .12), w * .90 * j(6, .07), w * 1.16 * j(7, .10), (w * 1.62 + self.bk) * j(8, .13)]) * (.12 + .88 * e)
        z = np.array([0, self.bk * j(9, .45), self.bk * .62 * j(10, .4), -self.dp * j(11, .12), -self.dp * j(12, .12), self.bk * .62 * j(13, .4), self.bk * j(14, .45), 0]) * e * st
        return np.c_[p + n * o[:, None], z]

    def draw(self, cel, cam, s, pal='night', light=(-.45, -.35, .80), lw=2.6, fog=None, near=1.2, cracks=True, streaks=True):
        pal = _pal(pal); L = unit(light); s = min(s, self.L)
        if s <= .05: return cel
        ks = np.arange(0, int(s / self.step) + 1) * self.step; ks = np.r_[ks, s] if s - ks[-1] > 1e-3 else ks; S = [self.section(q, s) for q in ks]; z = np.array([cam.cam(q.mean(0))[2] for q in S])
        T = lambda c, d: C(c) if fog is None else hexmix(c, fog[0], 1 - math.exp(-max(0., d - fog[1]) * fog[2])); sh = lambda n, j: (lambda d: pal['lit'] if d > .52 else pal['mid'] if d > .16 else pal['dark'])(float(n @ L) + .10 * n[2] + .16 * (h1(self.seed, j) - .5))
        order = np.argsort(-np.minimum(z[:-1], z[1:])); sdw = -L[:2] / max(L[2], .15)
        if cracks:
            g = lambda P, zz=0.: cam.project(np.c_[P, np.full(len(P), zz)])[0]
            for i in order:
                if z[i] < near or h1(self.seed, i, 30) > .60: continue
                for sd_, a in ((0, 0), (1, 7)):
                    if h1(self.seed, i, sd_, 31) > .55: continue
                    o = S[i][a, :2]; d = unit(o - S[i][3:5, :2].mean(0)); p, n = self.at(ks[i]); fw = np.array([n[1], -n[0]]); l = self.w * (1.2 + 2.6 * h1(self.seed, i, sd_, 32) ** 2); Z = zigzag(o - d * self.w * .2, o + (d + fw * .55) * l, self.seed + i * 3 + sd_, .14, 3); v = np.linspace(0, 1, len(Z)); tt = np.gradient(Z, axis=0); tt /= np.hypot(*tt.T)[:, None] + 1e-9; nn = np.c_[-tt[:, 1], tt[:, 0]] * (self.w * .11 * (1 - v) ** .8 + .004)[:, None]; draw_crack(cel, g, np.vstack([Z + nn, (Z - nn)[::-1]]), pal)
        for i in order:                                      # hard shadow of the berm on the plain
            if max(z[i], z[i + 1]) < near or min(z[i], z[i + 1]) < .25: continue
            for j, e in ((1, 0), (6, 7)):
                q = [np.r_[c[j, :2] + sdw * c[j, 2], 0] for c in (S[i], S[i + 1])]; Q = np.array([S[i][e], q[0], q[1], S[i + 1][e]])
                if (q[0][:2] - S[i][e, :2]) @ (S[i][e, :2] - S[i][j, :2]) > 0: cel.fill(cam.project(Q)[0], T(pal['dark'], z[i]), .92)
        for i in order:
            if max(z[i], z[i + 1]) < near or min(z[i], z[i + 1]) < .25: continue
            a, b = S[i], S[i + 1]; pa, pb = cam.project(a)[0], cam.project(b)[0]; zm = (z[i] + z[i + 1]) / 2; w_ = max(.6, min(lw, lw * cam.f / zm * self.w / 60))
            for j in (3, 0, 6, 1, 5, 2, 4):
                q3 = np.array([a[j], a[j + 1], b[j + 1], b[j]]); n = unit(np.cross(q3[1] - q3[0], q3[3] - q3[0])); n = n if n[2] > 0 else -n; q = np.array([pa[j], pa[j + 1], pb[j + 1], pb[j]]); col = pal['dark'] if j == 3 else sh(n, i * 9 + j)
                if float(n @ (cam.eye - q3.mean(0))) <= 0: continue
                cel.fill(q, T(col, zm)); cel.fill(q + (q - q.mean(0)) / (np.abs(q - q.mean(0)).max() + 1) * .7, T(col, zm))
                if j == 3: cel.fill(np.array([q[0], q[0] + (q[1] - q[0]) * (.30 + .1 * h1(self.seed, i, 40)), q[3] + (q[2] - q[3]) * (.30 + .1 * h1(self.seed, i + 1, 40)), q[3]]), T(pal['pit'], zm))
                if j in (2, 4): 
                    for v in range(2):                       # buttress wedges: the torn wall reads as vertical planes
                        if h1(self.seed, i, j, v, 6) < .62: u = h1(self.seed, i, j, v, 16); t0 = q[0] + (q[3] - q[0]) * u if j == 4 else q[1] + (q[2] - q[1]) * u; bt = q[1] + (q[2] - q[1]) * u if j == 4 else q[0] + (q[3] - q[0]) * u; ex = (q[3] - q[0]) * (.25 + .3 * h1(self.seed, i, j, v, 17)); cel.fill(np.array([t0, t0 + ex * .8, bt + ex * .25]), T(pal['dark'] if col != pal['dark'] else pal['pit'], zm))
                if j in (0, 6):
                    u = h1(self.seed, i, j, 18); e0, e1 = (q[0], q[1]) if j == 0 else (q[1], q[0]); cel.fill(np.array([e1 + (q[2 if j == 0 else 3] - e1) * u * .6, e1 + (q[2 if j == 0 else 3] - e1) * min(1, u * .6 + .4), e0 + (q[3 if j == 0 else 2] - e0) * .5]), T(pal['mid'] if col == pal['lit'] else pal['dark'] if col == pal['mid'] else pal['pit'], zm))
                if j in (1, 5) and h1(self.seed, i, j, 7) < .6: wedge(cel, q[0] if j == 1 else q[1], q[3] if j == 1 else q[2], w_ * 1.0, w_ * .3, T(pal['rim'], zm), .9)
            for j in (1, 6, 2, 5, 3, 4):
                if h1(self.seed, i, j, 8) < (.85 if j in (1, 6) else .6): wedge(cel, pa[j], pb[j], w_ * (1. if j in (1, 6) else .6), w_ * (.35 + .5 * h1(self.seed, i, j, 9)), T(pal['ink'], zm))
            if streaks:
                for v in range(3):
                    if h1(self.seed, i // 2, v, 10) < .7: u = .12 + .8 * h1(self.seed, i // 3, v, 11); wedge(cel, pa[3] * (1 - u) + pa[4] * u, pb[3] * (1 - u) + pb[4] * u, w_ * .7, w_ * .7, T(pal['pit'] if v else pal['mid'], zm), .85)
            it = []                                          # crust slabs stood up on both lips
            for sd_ in (0, 1):
                if h1(self.seed, i, sd_, 20) > .74 or zm > 60: continue
                c = S[i][1 if sd_ == 0 else 6]; p, n = self.at(ks[i]); out = -n if sd_ == 0 else n; sz = self.w * (.28 + .62 * h1(self.seed, i, sd_, 21) ** 1.6) * clamp((s - ks[i]) / (self.w * 1.2)); tang = np.array([n[1], -n[0], 0]); M = axis_m(tang, (1 if sd_ == 0 else -1) * -(.35 + .75 * h1(self.seed, i, sd_, 22))) @ axis_m([0, 0, 1], h1(self.seed, i, sd_, 23) * 6.28)
                if sz > .06: it.append((chunk(int(h1(self.seed, i, sd_, 24) * 64), 'slab').scaled(sz), np.r_[c[:2] + out * self.w * .10, c[2] + sz * .18], M, i * 5 + sd_, np.zeros(3)))
            if it: draw_bodies(cel, cam, it, pal, light, None, lw, fog=fog)
        return cel

    def rubble(self, front, frames=96, n=90, size=(.12, 1.1), speed=(3., 9.), seed=0, s0=0., g=9.8, back=.35):
        """front(t) -> metres ploughed.  Chunks leave both lips as the front passes, thrown sideways, forward and up; real landing"""
        ts = np.linspace(0, frames / 24., 600); ss = np.array([front(q) for q in ts]); rb = Rubble(seed + 5); r = R(seed, 88); smax = min(ss.max(), self.L)
        for i in range(n):
            s = s0 + (smax - s0) * r.random(); t0 = float(np.interp(s, ss, ts)); p, nn = self.at(s); fw = np.array([nn[1], -nn[0]]); sd = r.choice([-1, 1]); z = size3(r.random(), .16, .34); sz = size[0] + (size[1] - size[0]) * z; v0 = r.uniform(*speed) * (1.35 - .7 * z); v = np.array([*(nn * sd * r.uniform(.25, .9) + fw * r.uniform(-.1, back * 1.5)), r.uniform(.6, 1.3)]) * v0
            rb.add(chunk(int(r.integers(64)), ('block', 'slab', 'shard')[int(r.integers(3))]).scaled(sz * .5), np.array([*(p + nn * sd * self.w * r.uniform(.8, 1.2)), sz * .25]), v, r.normal(0, 1, 3) * 5 / (sz + .4), t0, unit(r.normal(0, 1, 4)))
        return rb.bake(frames, g)


def grit(cel, cam, t, origin, n=120, speed=(3, 16), up=.9, size=(.02, .07), col='#8d82c4', col2='#1a1932', seed=0, g=9.8, drag=.8, floor=0., dirn=None, spread=.8, streak=.02, t0=0.):
    """hundreds of tiny splinters in closed-form ballistics (with drag), each a sliver along its own motion"""
    o = np.asarray(origin, float); s = t - t0
    if s <= 0: return cel
    for i in range(n):
        r = h1(seed, i, 1); v0 = speed[0] + (speed[1] - speed[0]) * h1(seed, i, 2) ** 1.5; a = h1(seed, i, 3) * TAU; e = abs(math.sin(h1(seed, i, 4) * 1.4)) * up
        d = unit([math.cos(a), math.sin(a), e + .1]) if dirn is None else unit(unit(dirn) + spread * (np.array([h1(seed, i, 5), h1(seed, i, 6), h1(seed, i, 7)]) - .5) * 2); k = math.exp(-drag * s); p = o + d * v0 * (1 - k) / drag; p[2] -= g * s * s * .5 * (1 - .25 * min(1, drag * s)); v = d * v0 * k - np.array([0, 0, g * s])
        if p[2] < floor: continue
        q, z = cam.project(np.array([p, p - v * streak])); 
        if z.min() < .2: continue
        w = (size[0] + (size[1] - size[0]) * r) * cam.f / z[0]
        if w < .5 or not (-50 < q[0, 0] < cam.w + 50 and -50 < q[0, 1] < cam.h + 50): continue
        l = q[1] - q[0]; l = l / (np.hypot(*l) + 1e-6) * max(np.hypot(*l), w * 1.4); m = np.array([-l[1], l[0]]) / (np.hypot(*l) + 1e-6) * w * .5; cel.fill(np.array([q[0] - l * .15, q[0] + m, q[0] + l, q[0] - m]), col if h1(seed, i, 8) < .6 else col2)
    return cel
