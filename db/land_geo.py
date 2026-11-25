"""Rock built in metres: mesas, pillars, stacks, boulders, plates, ground, trench.  +X right +Y forward +Z up.

    M = Mesh(); mesa(M, (30, 200), R=40, H=150, seed=3); G = M.build()         Geo ready for land.render
    G.moved(R3, t, piv)   rigid move (returns a new Geo)        cat(G1, G2, ...)   join
"""
import numpy as np, math
from scipy.spatial import ConvexHull
from .paint import f32, rgb, fbm, smooth
from . import land_nb as nb


class Geo:
    K = ('T', 'FN', 'VN', 'VA', 'MAT', 'SD', 'EDG', 'COL')

    def __init__(s, **k): s.__dict__.update(k)

    @property
    def n(s): return len(s.T)

    def moved(s, R=None, t=(0, 0, 0), piv=(0, 0, 0)):
        R = np.eye(3) if R is None else np.asarray(R, float); piv = np.asarray(piv, float); t = np.asarray(t, float); g = Geo(**s.__dict__)
        g.T = ((s.T - piv) @ R.T + piv + t).astype(f32); g.FN = (s.FN @ R.T).astype(f32); g.VN = (s.VN @ R.T).astype(f32)
        return g

    def sel(s, m): return Geo(**{k: getattr(s, k)[m] for k in Geo.K})

    @property
    def center(s): return s.T.reshape(-1, 3).mean(0)


def cat(*gs):
    gs = [g for g in gs if g is not None and g.n]
    return gs[0] if len(gs) == 1 else Geo(**{k: np.concatenate([getattr(g, k) for g in gs]) for k in Geo.K})


class Mesh:
    def __init__(s): s.t, s.ao, s.mat, s.sd, s.col, s.ob = [], [], [], [], [], []; s.no = 0

    def add(s, T, ao=1., mat=0, seed=0, col=1., obj=None):
        T = np.asarray(T, float).reshape(-1, 3, 3); n = len(T)
        if n == 0: return
        if obj is None: s.no += 1; obj = s.no
        s.t.append(T); s.ao.append(np.broadcast_to(np.asarray(ao, float), (n, 3)) if np.ndim(ao) != 1 else np.broadcast_to(np.asarray(ao, float)[:, None], (n, 3)))
        s.mat.append(np.broadcast_to(mat, (n,))); s.sd.append(np.broadcast_to(seed, (n,))); s.ob.append(np.broadcast_to(obj, (n,)))
        c = np.asarray(col, float); s.col.append(np.broadcast_to(c, (n, 3)) if c.ndim <= 1 else c)

    def quad(s, a, b, c, d, **k): s.add(np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]), **k)

    def build(s, crease=.55):
        T = np.concatenate(s.t).astype(np.float64); e = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); l = np.linalg.norm(e, axis=1); k = l > 1e-7
        cat_ = lambda v: np.concatenate(v)[k]
        T = T[k]; FN = e[k] / l[k][:, None]; ob = cat_(s.ob).astype(np.int64); n = len(T)
        q = np.round(T.reshape(-1, 3) * 64).astype(np.int64); key = np.c_[np.repeat(ob, 3), q]
        _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel(); ng = inv.max() + 1
        order = np.argsort(inv, kind='stable'); start = np.r_[0, np.cumsum(np.bincount(inv, minlength=ng))]
        VN = np.zeros((n, 3, 3), f32); nb.vnormals(T.astype(f32), FN.astype(f32), inv.astype(np.int64), start.astype(np.int64), order.astype(np.int64), crease, VN)
        # signed dihedral of every edge (edge j opposite vertex j)
        F = inv.reshape(n, 3); E = np.stack([F[:, [1, 2]], F[:, [2, 0]], F[:, [0, 1]]], 1).reshape(-1, 2); Es = np.sort(E, 1); ek = Es[:, 0] * ng + Es[:, 1]
        o = np.argsort(ek, kind='stable'); sk = ek[o]; same = np.nonzero(sk[1:] == sk[:-1])[0]; EDG = np.zeros(n * 3, f32); C = T.mean(1)
        a, b = o[same], o[same + 1]; ta, tb = a // 3, b // 3; cs = (FN[ta] * FN[tb]).sum(1); sg = np.sign(-((C[tb] - C[ta]) * FN[ta]).sum(1)); v = sg * np.clip((.93 - cs) / .45, 0, 1); EDG[a] = v; EDG[b] = v
        return Geo(T=T.astype(f32), FN=FN.astype(f32), VN=VN, VA=cat_(s.ao).astype(f32), MAT=cat_(s.mat).astype(np.int32), SD=cat_(s.sd).astype(np.int32), EDG=EDG.reshape(n, 3), COL=cat_(s.col).astype(f32))


# ------------------------------------------------------------------------------------------------ rings -> body
def body(M, rings, ao, top=None, flip=False, **k):
    """rings (L, m, 3) bottom to top, CCW seen from above -> faceted skin; diagonals alternate by shape"""
    L, m = rings.shape[:2]; a = rings[:-1]; b = np.roll(rings, -1, 1)[:-1]; c = np.roll(rings, -1, 1)[1:]; d = rings[1:]
    A = ao[:-1]; B = np.roll(ao, -1, 1)[:-1]; Cc = np.roll(ao, -1, 1)[1:]; D = ao[1:]
    e1 = np.linalg.norm(a - c, axis=-1); e2 = np.linalg.norm(b - d, axis=-1); s = (e1 < e2)[..., None, None]
    T1 = np.where(s, np.stack([a, b, c], 2), np.stack([a, b, d], 2)); T2 = np.where(s, np.stack([a, c, d], 2), np.stack([b, c, d], 2))
    s2 = s[..., 0]; A1 = np.where(s2, np.stack([A, B, Cc], 2), np.stack([A, B, D], 2)); A2 = np.where(s2, np.stack([A, Cc, D], 2), np.stack([B, Cc, D], 2))
    M.no += 1; ob = M.no
    M.add(np.concatenate([T1.reshape(-1, 3, 3), T2.reshape(-1, 3, 3)]), ao=np.concatenate([A1.reshape(-1, 3), A2.reshape(-1, 3)]), obj=ob, **k)
    if top is not None:
        r0 = rings[-1]; c0 = r0.mean(0) + [0, 0, top]; q = np.roll(r0, -1, 0)
        M.add(np.stack([r0, q, np.broadcast_to(c0, r0.shape)], 1), ao=1., obj=ob, **k)
    return ob


def _foot(r, m, rough=.2, clefts=3, asp=1., rot=0.):
    """irregular angular footprint: unit radii with a few deep clefts and flat faces"""
    a = (np.arange(m) + r.uniform(-.32, .32, m)) / m * 2 * np.pi
    k = int(r.integers(4, 7)); ph = r.uniform(0, 6.28); poly = 1 / np.maximum(.45, np.cos(((a + ph) % (2 * np.pi / k)) - np.pi / k)) * math.cos(np.pi / k)       # k-gon: flat faces
    b = .16 + .84 * poly + r.normal(0, rough * .16, m)
    cl = np.zeros(m)
    for _ in range(clefts):
        j = int(r.integers(m)); d = r.uniform(.12, .34); b[j] -= d; cl[j] = d; b[(j + 1) % m] += d * .16
    c, s = math.cos(rot), math.sin(rot); x = np.cos(a) * b * asp; y = np.sin(a) * b
    return np.c_[x * c - y * s, x * s + y * c], cl


def mesa(M, c, R, H, seed=0, kind='mesa', sides=None, z0=-8., asp=1., rot=0., lean=(0, 0), mat=0, col=1., talus=1., caps=True, rocks=True, levels=1., clefts=None):
    """one table mountain / pillar / stack.  kind: mesa | pillar | stack | fin | butte"""
    r = np.random.default_rng(seed * 7919 + 13); m = sides or int(np.clip(10 + R * .28, 11, 34))
    P, cl = _foot(r, m, .22 if kind != 'pillar' else .13, clefts if clefts is not None else (4 if kind in ('mesa', 'fin') else 2), asp, rot)
    rows = []                                                                     # (t, scale, ao, jitter)
    if kind in ('mesa', 'butte', 'fin'):
        rows += [(0., 1.30 * (1 + .3 * talus), .75, .07), (.035, 1.22 * (1 + .2 * talus), .8, .06), (.09 + .04 * talus, 1.05, .62, .05)]
        n = int(r.integers(3, 6)); ts = np.sort(r.uniform(.2, .86, n)); s = 1.0
        for j, t in enumerate(ts):
            s -= r.uniform(.0, .03); rows.append((t - .02, s, .95, .035)); 
            if r.random() < .55: g = r.uniform(.03, .085); s -= g; rows.append((t - .004, s + g, 1., .03)); rows.append((t + .004, s, .5, .03))        # ledge
            else: g = r.uniform(.02, .055); rows.append((t - .006, s - g * .4, .42, .03)); s += g * .35; rows.append((t + .006, s, .95, .03))           # overhang
            rows.append((min(.9, t + r.uniform(.03, .07)), s - .005, .85, .035))
        rows += [(.895, s - .02, .6, .03), (.905, s + .045, .7, .02), (.955, s + .055, 1., .025), (.985, s + .04, 1., .02), (1., s - .01, 1., .02)]
    elif kind == 'stack':
        s = 1.; k = int(r.integers(3, 5)); tt = np.r_[0, np.sort(r.uniform(.15, .85, k - 1)), 1.]; sc = r.uniform(.62, 1.0, k); sc[0] = 1.05
        rows += [(0., 1.5, .7, .08)]
        for j in range(k):
            a, b = tt[j], tt[j + 1]; w = sc[j]; h = b - a
            rows += [(a + .02 * h + .004, w * .78, .35, .04), (a + .14 * h, w * .96, .75, .04), (a + .36 * h, w * 1.05, .95, .04), (a + .62 * h, w * 1.02, 1., .04), (a + .84 * h, w * .92, 1., .04), (a + .95 * h, w * .74, 1., .04)]
        rows.append((1., sc[-1] * .5, 1., .05))
    else:
        w = r.uniform(.66, .84); hd = r.uniform(1.0, 1.28); q = r.uniform(.5, .66)
        for t, s_, a_ in [(0, 1.75, .6), (.03, 1.45, .7), (.08, 1.18, .75), (.16, 1.04, .9), (.26, .95, .9), (.37, .88, .9), (q - .06, w + .03, .85), (q, w, .7), (q + .05, w + .01, .6), (q + .11, w + .10, .5), (.79, (w + hd) / 2, .6), (.835, hd, .75), (.88, hd * 1.03, 1.), (.93, hd * .97, 1.), (.965, hd * .86, 1.), (.99, hd * .66, 1.), (1., hd * .42, 1.)]: rows.append((t, s_, a_, .035))
        rows.sort(key=lambda v: v[0])
    L = len(rows); m_ = m; rings = np.zeros((L, m_, 3)); ao = np.zeros((L, m_)); slow = fbm(max(L, 8), max(m_, 8), (4, 2), (1, .6), seed + 5)[:L, :m_] * .030; brk = np.zeros((L, m_))
    for _ in range(int(3 * levels + L * .25)):                                     # blocks fallen out of the wall
        j0 = int(r.integers(2, max(3, L - 3))); i0 = int(r.integers(m_)); n_ = int(r.integers(1, 4)); hL = int(r.integers(1, 3)); d = r.uniform(.04, .12)
        for q in range(n_): brk[j0:j0 + hL, (i0 + q) % m_] = d
    for j, (t, s_, a_, jt) in enumerate(rows):
        f = s_ * (1 + slow[j] + r.normal(0, jt * .42, m_) - brk[j] * (1 if t < .97 else 0))
        pin = np.clip((t - .0) * 1.2, 0, 1); rr = P * f[:, None]
        if kind in ('mesa', 'butte', 'fin') and t < .1: rr = P / (np.linalg.norm(P, axis=1, keepdims=True) ** (.45)) * f[:, None]         # talus rounds the plan
        zz = z0 + (H - z0) * t + (r.normal(0, .006 * (H - z0), m_) if 0 < t < .985 else r.normal(0, .0035 * H, m_))
        rings[j] = np.c_[rr * R + np.asarray(c, float)[:2] + np.outer(np.full(m_, t), lean) * H, zz]
        ao[j] = np.clip(a_ * (1 - 1.5 * cl) * (1 - 1.6 * brk[j]) * (.55 + .45 * smooth(0, .16, t)), .12, 1)
    ob = body(M, rings, ao, top=None, mat=mat, seed=seed % 8000, col=col)
    # broken top: two inner rings with blocks of different height
    rt = rings[-1]; cc = rt.mean(0); in1 = cc + (rt - cc) * r.uniform(.60, .74, (m_, 1)); in1[:, 2] = H + r.normal(0, .006, m_) * H + .004 * H; in2 = cc + (rt - cc) * .28; in2[:, 2] = H + r.normal(.01, .008, m_) * H
    tr = np.stack([rt, in1, in2]); body(M, tr, np.ones((3, m_)), top=.2, mat=mat, seed=seed % 8000 + 1, col=col)
    if caps and kind in ('mesa', 'butte'):                                          # second, smaller table on the first
        if r.random() < .6: th = r.uniform(0, 6.28); mesa(M, (c[0] + math.cos(th) * R * .25 + lean[0] * H, c[1] + math.sin(th) * R * .25 + lean[1] * H), R * r.uniform(.34, .56), H + H * r.uniform(.06, .16), seed * 3 + 1, 'butte', z0=H - .03 * H, talus=.15, caps=False, rocks=False, mat=mat, col=col)
    if rocks:
        for _ in range(int(r.integers(5, 11))):
            th = r.uniform(0, 6.28); d = R * r.uniform(1.15, 1.65); s_ = R * r.uniform(.05, .18); boulder(M, (c[0] + math.cos(th) * d * asp, c[1] + math.sin(th) * d, z0 + (H - z0) * .012 + s_ * .2), s_, r, mat=mat, col=col)
        for _ in range(int(r.integers(2, 6))):
            th = r.uniform(0, 6.28); d = R * r.uniform(0, .7); s_ = R * r.uniform(.03, .10)
            if kind in ('mesa', 'fin'): boulder(M, (c[0] + lean[0] * H + math.cos(th) * d * asp * .8, c[1] + lean[1] * H + math.sin(th) * d * .8, H + s_ * .25), s_, r, mat=mat, col=col)
    return ob


def boulder(M, c, s, r, mat=0, col=1., sz=(1., .85, .7), n=None, ao=.9):
    """angular rock: hull of few points pushed on to random cutting planes"""
    n = n or int(r.integers(13, 21)); P = r.normal(0, 1, (n, 3)); P /= np.linalg.norm(P, axis=1, keepdims=True)
    for _ in range(4):                                                             # planar cuts
        d = r.normal(0, 1, 3); d /= np.linalg.norm(d); k = r.uniform(.5, .8); h = P @ d; P = P - np.where(h > k, h - k, 0)[:, None] * d
    P = P * np.asarray(sz) * s * r.uniform(.8, 1.2, 3); th = r.uniform(0, 6.28); ca, sa = math.cos(th), math.sin(th); P = P @ np.array([[ca, sa, 0], [-sa, ca, 0], [0, 0, 1]])
    H = ConvexHull(P); T = P[H.simplices]; nn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); out = (nn * T.mean(1)).sum(1) < 0; T[out] = T[out][:, [0, 2, 1]]
    z = T[..., 2]; a = np.clip(.45 + .55 * (z + s * .4) / (s * 1.0), .3, 1) * ao
    M.add(T + np.asarray(c, float), ao=a, mat=mat, seed=int(r.integers(8000)), col=col)


def slab(M, c, R, th, r, mat=0, col=1., sides=None, under=1., tilt=None, ao=1., top=None):
    """torn plate of ground: flat chipped top, faceted tapering underside.  c = top centre"""
    m = sides or int(r.integers(6, 10)); a = (np.arange(m) + r.uniform(-.3, .3, m)) / m * 2 * np.pi; b = R * r.uniform(.72, 1.1, m); X = np.c_[np.cos(a) * b, np.sin(a) * b]
    rows = [(1.0, 0., 1.), (1.05, -.16 * th, .95), (.98, -.55 * th, .8), (.74, -1.0 * th * under, .6), (.40, -1.5 * th * under, .5), (.10, -1.85 * th * under, .4)]
    rings = np.stack([np.c_[X * f * (1 + r.normal(0, .06, (m, 1)) * (j > 0)), np.full(m, z) + r.normal(0, .05 * th, m) * (j > 1)] for j, (f, z, _) in enumerate(rows)])[::-1]
    ao_ = np.stack([np.full(m, w) for *_, w in rows])[::-1] * ao
    A = np.eye(3) if tilt is None else np.asarray(tilt, float); rings = rings @ A.T + np.asarray(c, float)
    sd = int(r.integers(8000)); body(M, rings, ao_, top=None, mat=mat, seed=sd, col=col)
    rt = rings[-1]; cc = rt.mean(0); up = A @ [0, 0, 1.]; in1 = cc + (rt - cc) * .55 + up * r.normal(0, .02 * R, (m, 1)); body(M, np.stack([rt, in1]), np.ones((2, m)) * ao, top=0., mat=mat if top is None else top, seed=sd + 1, col=col)


def sheet(M, x, y, zf, mat=1, seed=0, jit=.32, col=1., ao=None, hole=None):
    """heightfield z = zf(X, Y) over the grid lines x, y (arrays), vertices jittered so that no grid shows"""
    r = np.random.default_rng(seed + 99); X, Y = np.meshgrid(np.asarray(x, float), np.asarray(y, float)); dx = np.gradient(X, axis=1); dy = np.gradient(Y, axis=0)
    X = X + r.uniform(-jit, jit, X.shape) * np.abs(dx); Y = Y + r.uniform(-jit, jit, Y.shape) * np.abs(dy); Z = zf(X, Y); V = np.stack([X, Y, Z], -1)
    A = np.ones_like(Z) if ao is None else ao(X, Y, Z)
    a, b, c, d = V[:-1, :-1], V[:-1, 1:], V[1:, 1:], V[1:, :-1]; a_, b_, c_, d_ = A[:-1, :-1], A[:-1, 1:], A[1:, 1:], A[1:, :-1]
    s = (r.random(a.shape[:2]) < .5)[..., None, None]; s2 = s[..., 0]
    T1 = np.where(s, np.stack([a, b, c], 2), np.stack([a, b, d], 2)); T2 = np.where(s, np.stack([a, c, d], 2), np.stack([b, c, d], 2)); A1 = np.where(s2, np.stack([a_, b_, c_], 2), np.stack([a_, b_, d_], 2)); A2 = np.where(s2, np.stack([a_, c_, d_], 2), np.stack([b_, c_, d_], 2))
    T = np.concatenate([T1.reshape(-1, 3, 3), T2.reshape(-1, 3, 3)]); Ao = np.concatenate([A1.reshape(-1, 3), A2.reshape(-1, 3)])
    if hole is not None: k = ~hole(T.mean(1)); T, Ao = T[k], Ao[k]
    M.add(T, ao=Ao, mat=mat, seed=seed % 8000, col=col)


def pebbles(M, n, box, zf, r, smin=.08, smax=.6, mat=0, col=1., p=2.5, keep=None):
    """scatter of individually cut stones, small ones common"""
    k = 0
    while k < n:
        x, y = r.uniform(box[0], box[1]), r.uniform(box[2], box[3])
        if keep is not None and not keep(x, y): continue
        s = smin + (smax - smin) * r.random() ** p; boulder(M, (x, y, float(zf(np.array(x), np.array(y))) + s * .18), s, r, mat=mat, col=col, n=int(r.integers(9, 15)), ao=.85); k += 1
