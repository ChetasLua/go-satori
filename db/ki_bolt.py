"""ki bolt: drawn electricity (hard elbows, thick-thin, forks, leader / strike / break-up) and arcs that jump along a silhouette."""
from .ki_core import *


def bolt_pts(p0, p1, seed, jag=.20, depth=5, bow=0.):
    """hard-elbowed path by uneven midpoint breaks; bow pushes the middle sideways (px)"""
    r = R(seed, 5); P = [np.asarray(p0, float), np.asarray(p1, float)]
    if bow: d = P[1] - P[0]; L = np.hypot(*d) + 1e-6; P = [P[0], (P[0] + P[1]) / 2 + np.array([-d[1], d[0]]) / L * bow, P[1]]
    for k in range(depth):
        Q = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            d = b - a; L = np.hypot(*d)
            if L > 16 and (k < 2 or L > 110 or r.random() < .78):
                u = r.uniform(.28, .72); n = np.array([-d[1], d[0]]) / (L + 1e-6); Q.append(a + d * u + n * L * jag * (.62, .85, 1.1, 1.2, 1.2, 1.2)[min(k, 5)] * r.choice([-1, 1]) * r.uniform(.35, 1.25))
            Q.append(b)
        P = Q
    return np.array(P)


def _strand(F, P, w, seed, edge, taper=(0., 1.), tex=1.):
    """kites along a bolt: each segment swells or thins, elbows stay sharp"""
    r = R(seed, 7); n = len(P); s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; s = s / (s[-1] + 1e-6); ww = w * (1 - taper[0] * (1 - s)) * (1 - taper[1] * s) ** .7 * r.uniform(.45, 1.25, n)
    for i in range(n - 1):
        a, b = P[i], P[i + 1]; d = b - a; L = np.hypot(*d) + 1e-6; o = d / L * min(L * .25, ww[i] * .9); m = a + d * r.uniform(.3, .7); big = max(ww[i], ww[i + 1]) * r.uniform(.8, 1.25); Q = np.array([a - o * .6, a, m, b, b + o * .6]); h = np.array([.2, ww[i] * .7, big * .75, ww[i + 1] * .7, .2])
        F.rib(Q, [h + edge, h + edge * .35, h * .48 - .15], 0.)


def bolt(F, p0, p1, w=7., seed=0, jag=.20, forks=3, edge=3., age=0, life=5, bow=0., depth=5, fork_len=.42, spread=.75):
    """one lightning stroke at `age` (0 leader, 1 strike, 2 second look, 3.. break-up).  F = Field(3)"""
    if age < 0 or age >= life: return F
    P = bolt_pts(p0, p1, seed, jag, depth, bow); n = len(P)
    if age == 0: m = max(3, int(n * .62)); _strand(F, P[:m], w * .30, seed, edge * .6, (0, 1.)); return F
    if age >= 2:                                           # same road, redrawn elbows
        r = R(seed, age, 3); amp = np.hypot(*(P[-1] - P[0])) / n * .16; P = P + r.normal(0, amp, P.shape) * np.r_[0, np.ones(n - 2), 0][:, None]
    k = {1: 1., 2: .62}.get(age, max(.16, .42 * (1 - (age - 3) / max(1, life - 3))))
    if age <= 2: _strand(F, P, w * k, seed + age, edge * (1. if age == 1 else .8), (0., .75))
    else:                                                  # broken into pieces that hang
        r = R(seed, age, 9); i = 0
        while i < n - 2:
            m = int(r.integers(2, 5))
            if r.random() > .30 + .12 * (age - 3): _strand(F, P[i:i + m + 1], w * k, seed + i, edge * .5, (.6, .9))
            i += m + int(r.integers(0, 3))
    if age in (1, 2, 3):
        r = R(seed, 21); L = np.hypot(*(np.asarray(p1, float) - p0))
        for j in range(forks):
            i = int(r.integers(1, max(2, n - 2))); d = P[min(n - 1, i + 1)] - P[i - 1]; th = math.atan2(d[1], d[0]) + r.choice([-1, 1]) * r.uniform(.4, 1.) * spread; l = L * fork_len * r.uniform(.35, 1.) * (1 - .5 * i / n); e = P[i] + np.array([math.cos(th), math.sin(th)]) * l; Q = bolt_pts(P[i], e, seed * 7 + j, jag * 1.15, max(2, depth - 1))
            if age == 3 and j % 2: continue
            _strand(F, Q, w * k * r.uniform(.30, .55), seed + j, edge * .6, (0., 1.))
            if l > 120 and age < 3: q = int(len(Q) * .5); th2 = th + r.choice([-1, 1]) * .7; _strand(F, bolt_pts(Q[q], Q[q] + np.array([math.cos(th2), math.sin(th2)]) * l * .4, seed * 11 + j, jag, 2), w * k * .2, seed + j + 3, edge * .5, (0., 1.))
    return F


def bolt_cel(F, kind='gold', ink=True):
    """three hard tones: dark saturated edge, colour, white core"""
    c = {'gold': ('#c25a06', '#ffd94a', '#fffdf0'), 'blue': ('#1746d6', '#7cc8ff', '#ffffff'), 'base': ('#4a8ae8', '#bfe0ff', '#ffffff'), 'violet': ('#5a12c8', '#b46cff', '#ffffff'), 'silver': ('#3f62c0', '#a9c8ff', '#ffffff'), 'red': ('#8e0618', '#ff3a3a', '#fff0e8')}[kind if kind != 'goldblue' else 'blue']
    return F.cel(c)


def storm(strikes, f, kind='gold', light=1.):
    """strikes = [(start frame, p0, p1, w, seed[, dict])] -> (layer, light) at frame f; lightning on ones"""
    F = Field(3); on = False
    for q in strikes:
        a = f - q[0]; kw = dict(q[5]) if len(q) > 5 else {}; life = kw.get('life', 6)
        if 0 <= a < life: bolt(F, q[1], q[2], q[3], q[4], age=a, **kw); on = True
    if not on: return None, None
    l = bolt_cel(F, kind); return l, bloom(l, .4, (4, 14, 46), (.5, .32, .24), pal(kind if kind != 'goldblue' else 'blue')['glow'], .8 * light)


def arcs(mask, f, kind='gold', n=7, seed=1, w=None, reach=.10, rate=1., hug=None, cross=.35, face=None):
    """electricity jumping along a silhouette -> (back, front, light).  each arc lives 5-6 frames: leader, strike, redraw, break-up, then another place"""
    from .ki_aura import Hug
    if mask.ndim == 3: mask = mask[..., 3]
    hug = hug or Hug(mask)
    if not hug.any: return None, None, None
    U = hug.U; w = w or max(1.6, U * .0075); B, F = Field(3), Field(3)
    for i in range(n):
        per = int(7 + 7 * h1(seed, i, 1)) / rate; x = f / per + h1(seed, i, 2); c = int(math.floor(x)); age = int((x - c) * per); sd = int(seed * 313 + i * 29 + c * 11) & 0xffffff; life = 5 + int(2 * h1(sd, 1))
        if age >= life: continue
        u = h1(sd, 2); du = (.05 + .11 * h1(sd, 3)) * (1 if h1(sd, 4) > .5 else -1); hops = 2 + int(2.2 * h1(sd, 5)); pts = [hug.root(u + du * j / hops) for j in range(hops + 1)]
        def out(p, k): return p + np.array([hug.samp(hug.nx, p[None])[0], hug.samp(hug.ny, p[None])[0]]) * U * k
        pts = [out(p, -.012 if j % 2 == 0 else reach * (.25 + .9 * h1(sd, j, 6))) for j, p in enumerate(pts)]; cr = h1(sd, 7) < cross
        if cr: q = hug.root(u + .5 * h1(sd, 8)); pts = [pts[0], (pts[0] + q) / 2 + R(sd).normal(0, U * .03, 2), q]
        G = F if cr or h1(sd, 9) < .4 else B
        for j, (a, b) in enumerate(zip(pts[:-1], pts[1:])): bolt(G, a, b, w * (.6 + .8 * h1(sd, 10)), sd + j, .20, 2 if j == 0 else 1, max(1.2, w * .42), age, life, U * .02 * (h1(sd, j, 11) - .5), 3, .6)
    fr = bolt_cel(F, kind); bk = bolt_cel(B, kind)
    if face is not None: x, y = grid(); d = np.sqrt((x - face[0]) ** 2 + (y - face[1]) ** 2); fr = fr * np.clip((d - face[2]) / 4, 0, 1)[..., None]
    return bk, fr, bloom(stack(bk, fr), .4, (3, 12, 36), (.45, .3, .2), pal(kind if kind != 'goldblue' else 'blue')['glow'], .7)
