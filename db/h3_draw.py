"""Draws the posed clay of db.hero3 as an inked anime cel: flats, one angular shadow, deep accents, gloss, rim planes, fold blades, muscle pen lines that start and stop."""
import numpy as np, cv2, math, zlib
from . import W, H
from .paint import rgb, f32
from .ink import Cel, spl, resample, over_roi, paste
from . import kit
from .h3_body import MATS, LINEW
from .h3_sdf import wrap

INK = '#17131a'
GLOSS = {'fv': (.90, 1.), 'bootw': (.965, .55), 'armw': (.955, .6), 'armg': (.95, .8), 'gold': (.95, .8), 'boot': (.97, .35), 'fw': (.975, .5), 'glove': (.985, .4)}
SKINS = {'skin', 'skinv', 'skink', 'green', 'pink', 'fw'}
OUTER = {'gi', 'lapr', 'lapl', 'bend0', 'bend1', 'capesh', 'belt', 'armour', 'armab', 'knot', 'ucol', 'bandL', 'bandR', 'glvL', 'glvR', 'slvL', 'slvR', 'scol', 'troL', 'trouL', 'trouR', 'trou', 'strap1', 'strap-1'}


# ------------------------------------------------------------------ vector helpers
def _smooth(P, k, closed):
    k = int(k) | 1
    if k < 3 or len(P) < k + 2: return P
    w = np.hanning(k + 2)[1:-1]; w /= w.sum(); h = k // 2
    Q = np.vstack([P[-h:], P, P[:h]]) if closed else np.vstack([2 * P[0] - P[h:0:-1], P, 2 * P[-1] - P[-2:-h - 2:-1]])
    return np.stack([np.convolve(Q[:, i], w, 'valid') for i in (0, 1)], -1)


def _dp(P, eps, closed=False):
    n = len(P)
    if n < 3: return np.arange(n)
    q = cv2.approxPolyDP(np.round(P * 16).astype(np.int32)[:, None], eps * 16, closed)[:, 0]
    R = np.round(P * 16).astype(np.int32); key = {}
    for i, p in enumerate(map(tuple, R)): key.setdefault(p, i)
    return np.array(sorted({key[p] for p in map(tuple, q) if p in key}))


def draw_contour(P, eps, closed=True, sharp=.75, n=5):
    """straights against curves: simplify, keep corners sharper than `sharp` rad, bend the rest"""
    if len(P) < 8: return P
    idx = _dp(P, eps, closed)
    if len(idx) < 3: return P[[0, len(P) // 2, -1]] if not closed else P
    V = P[idx]; m = len(V)
    a = np.roll(V, 1, 0) if closed else np.vstack([V[:1], V[:-1]]); b = np.roll(V, -1, 0) if closed else np.vstack([V[1:], V[-1:]])
    t1 = np.arctan2(*(V - a).T[::-1]); t2 = np.arctan2(*(b - V).T[::-1]); turn = np.abs((t2 - t1 + math.pi) % (2 * math.pi) - math.pi)
    c = [i for i in range(m) if turn[i] > sharp and (closed or 0 < i < m - 1)]
    if closed:
        if len(c) == 1: c.append((c[0] + m // 2) % m)
        return spl(V, True, corners=c, n=n)
    return spl(V, False, corners=c, n=n)


def clean_closed(P, eps, sharp=.5, n=6):
    """closed raw contour -> (clean closed polyline, dp index list, n)"""
    idx = _dp(P, eps, True)
    if len(idx) < 4: idx = np.unique(np.linspace(0, len(P) - 1, 5).astype(int)[:-1])
    V = P[idx]; d1 = np.roll(V, -1, 0) - V; L1 = np.hypot(*d1.T) + 1e-6; u1 = d1 / L1[:, None]; u0 = np.roll(u1, 1, 0); L0 = np.roll(L1, 1)
    turn = np.abs((np.arctan2(u1[:, 1], u1[:, 0]) - np.arctan2(u0[:, 1], u0[:, 0]) + math.pi) % (2 * math.pi) - math.pi)
    T = u0 * L1[:, None] + u1 * L0[:, None]; T /= np.hypot(*T.T)[:, None] + 1e-6
    c = (turn > sharp)[:, None]; to = np.where(c, u1, T); ti = np.where(c, u0, T)
    t = (np.arange(n) / n)[None, :, None]; h00 = 2 * t ** 3 - 3 * t ** 2 + 1; h10 = t ** 3 - 2 * t ** 2 + t; h01 = -2 * t ** 3 + 3 * t ** 2; h11 = t ** 3 - t ** 2
    d = L1[:, None, None] * .92; V1 = np.roll(V, -1, 0)
    return (h00 * V[:, None] + h10 * to[:, None] * d + h01 * V1[:, None] + h11 * np.roll(ti, -1, 0)[:, None] * d).reshape(-1, 2), idx, n


def clean_index(r, idx, n, N):
    k = np.searchsorted(idx, r, side='right') - 1
    if k < 0: a = idx[-1] - N; b = idx[0]; k = len(idx) - 1; return (k + (r - a) / max(b - a, 1)) * n
    a = idx[k]; b = idx[k + 1] if k + 1 < len(idx) else idx[0] + N
    return (k + (r - a) / max(b - a, 1)) * n


def mask_polys(m, g, box, eps=.6, up=2, min_area=10., smooth=5, sharp=.75, raw=False, sig=.6):
    ys, xs = np.nonzero(m)
    if len(ys) < 3: return []
    y0, y1, x0, x1 = max(0, ys.min() - 3), ys.max() + 4, max(0, xs.min() - 3), xs.max() + 4
    s = m[y0:y1, x0:x1].astype(f32); s = cv2.copyMakeBorder(s, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    if sig > 0: s = cv2.GaussianBlur(s, (0, 0), sig)
    s = cv2.resize(s, None, fx=up, fy=up, interpolation=cv2.INTER_LINEAR)
    cs, _ = cv2.findContours((s > .5).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE); out = []
    for c in cs:
        if len(c) < 8: continue
        c = c[:, 0].astype(float)
        if abs(cv2.contourArea(c.astype(f32))) < min_area * up * up: continue
        c = _smooth(c, smooth * up, True); c = ((c + .5) / up - 2 + [x0, y0]) / g + [box[0], box[1]]
        out.append(c if raw else draw_contour(c, eps, True, sharp))
    return out


def _hash(*a): return zlib.crc32(repr(a).encode()) & 0x7fffffff


def _mixc(a, b, t):
    a = rgb(a) if isinstance(a, str) else np.asarray(a, f32); b = rgb(b) if isinstance(b, str) else np.asarray(b, f32)
    return a * (1 - t) + b * t


# light presets: every colour of the figure is multiplied (mul lit, shmul shadow, litmul highlight); skins may be named outright so heads of the head lanes match
LIGHTS = dict(
    day=dict(), dawn=dict(mul='#fff0e6', shmul='#f2c8c8', litmul='#fff6ee'), lamp=dict(mul='#ffdfb0', shmul='#e09a78', litmul='#fff0cc', ink='#23120c'),
    night=dict(mul='#b2a8dc', shmul='#5c4f9e', litmul='#f0e2ff', add='#050310', ink='#0c0918', tint=('#6a5ab0', .34)), moon=dict(mul='#9486cc', shmul='#423478', litmul='#eeddff', add='#040210', ink='#090612', tint=('#5a48a8', .45)), dark=dict(mul='#5c5090', shmul='#2c2358', litmul='#b49ae6', add='#040210', ink='#07050f', tint=('#4a3a90', .55)),
    fire=dict(mul='#ffc890', shmul='#9a5a5e', litmul='#ffe9b8', ink='#1e0d10'), beam=dict(mul='#ffe2a8', shmul='#b87848', litmul='#fffbe0', ink='#2a140a'), gold=dict(mul='#ffeab0', shmul='#c48a52', litmul='#fffbe2', ink='#2a1608'),
    rain=dict(mul='#c8c8ca', shmul='#8d8f9a', litmul='#eaeef0', ink='#141317', tint=('#8c9098', .32)), storm=dict(mul='#b4bcc8', shmul='#6a7288', litmul='#e4ecf4', ink='#101218', tint=('#6c7890', .30)), space=dict(mul='#ccd2ea', shmul='#4c5488', litmul='#f4f8ff', ink='#080812', tint=('#5060a0', .22)), ki=dict(mul='#b8dcff', shmul='#4a6cb8', litmul='#f2fbff', ink='#0a1024'),
    grey=dict(mul='#ffffff', shmul='#ffffff', grey=True))


class Pal:
    """colour script: name in LIGHTS or dict(base='night', mats={...}, mul, shmul, litmul, add, ink)"""

    def __init__(self, d=None):
        if isinstance(d, str): d = dict(LIGHTS[d])
        d = dict(d or {}); b = dict(LIGHTS[d['base']]) if 'base' in d else {}; b.update(d); d = b
        self.d = d; self.mats = dict(MATS); self.mats.update(d.get('mats', {})); self.ink = d.get('ink', INK)
        v = lambda k, e: rgb(d[k]) if isinstance(d.get(k), str) else np.asarray(d.get(k, e), f32)
        self.mul = v('mul', (1, 1, 1)); self.shmul = v('shmul', (1, 1, 1)); self.add = v('add', (0, 0, 0)); self.litmul = v('litmul', self.mul); self.cache = {}

    def __call__(self, mat):
        if mat not in self.cache:
            c = [rgb(q) if isinstance(q, str) else np.asarray(q, f32) for q in self.mats.get(mat, MATS['grey'])]
            c = (c[0] * self.litmul + self.add, c[1] * self.mul + self.add, c[2] * self.shmul + self.add * .6, c[3] * self.shmul + self.add * .4)
            if 'tint' in self.d:                                              # the air of the scene: every colour leans toward it, shadows most
                tc, ta = rgb(self.d['tint'][0]), self.d['tint'][1]; tc = tc / max(float(tc @ np.array([.3, .59, .11], f32)), .05)
                c = tuple(x * (1 - a) + tc * float(x @ np.array([.3, .59, .11], f32)) * a for x, a in zip(c, (ta * .6, ta, min(1., ta * 1.35), min(1., ta * 1.5))))
            if self.d.get('grey'): c = tuple(np.full(3, float(x @ np.array([.3, .59, .11], f32)), f32) for x in c)
            self.cache[mat] = c
        return self.cache[mat]


LINEDIR = dict(pec=(lambda sk, s: -sk.Rc[:, 2] + sk.Rc[:, 0] * (.25 if s == 'R' else -.25) * 0, .10), trap=(lambda sk, s: -sk.Rc[:, 2], -.2), bic=(lambda sk, s: -sk.e1a[s] * 0 + np.cross(sk.e1a[s], sk.el[s] - sk.sh[s]) / (np.linalg.norm(sk.el[s] - sk.sh[s]) + 1e-9), -2.))


class Draw:
    linedir = LINEDIR
    def __init__(s, fig, cam, B, px, pal, light, key, weight, expr, head_fn, head, seed, fx, ink_only, rim, lines=1.0, shadow=True):
        s.fig, s.cam, s.B, s.px, s.pal, s.key, s.expr, s.head_fn, s.head, s.seed, s.fx = fig, cam, B, px, pal, key, expr, head_fn, head, seed, fx
        c2w = lambda v: cam.R.T @ (np.array([v[0], -v[1], -v[2]], float) / np.linalg.norm(v))
        rims = [] if rim is None else (rim if isinstance(rim, (list, tuple)) else [rim]); s.rims = [dict(r, D=c2w(r['dir'])) for r in rims if 'dir' in r]
        s.L = c2w(light); s.lc = np.asarray(light, float)
        s.wo = float(np.interp(px, [50, 120, 250, 450, 900, 1600, 2600], [1.0, 1.35, 1.9, 2.6, 3.5, 4.9, 6.2])) * weight
        s.wi = s.wo * .70; s.wf = s.wo * .52; s.ink = pal.ink; s.ink_only = ink_only; s.ml = lines; s.cast = shadow

    def run(s):
        cam, B, c = s.cam, s.B, s.B.c
        box = c.bbox(cam); x0, y0, x1, y1 = box
        if x1 - x0 < 4 or y1 - y0 < 4: return None
        big = max(x1 - x0, y1 - y0); s.g = g = min(1., float(np.interp(s.px, [600, 1300], [740., 860.])) / big) if s.px > 130 else 1.5 if s.px < 90 else 1.25
        s.box = box; s.bodybox = box
        s.buf = c.render(cam, box, g); s.Z, s.W, s.N, s.T, s.A, s.C = s.buf
        if (s.W >= 0).sum() < 6: return None
        s.hit = s.W >= 0; h, w = s.Z.shape; Wm = np.maximum(s.W, 0)
        names = sorted(set(c.lab)); s.lname = names; lid = {n: i for i, n in enumerate(names)}
        s.lab = np.where(s.hit, np.array([lid[l] for l in c.lab])[Wm], -1).astype(np.int16)
        mn = []; mi = {}
        def mid(n):
            if n not in mi: mi[n] = len(mn); mn.append(n)
            return mi[n]
        s.mat = np.where(s.hit, np.array([mid(m) for m in c.mat])[Wm], -1).astype(np.int16)
        mgn = sorted({m for m in c.mg if m}); s.mgn = mgn; gid = {n: i for i, n in enumerate(mgn)}
        s.mg = np.where(s.hit, np.array([gid.get(m, -1) if m else -1 for m in c.mg])[Wm], -1).astype(np.int16)
        done = {}
        for i, r in enumerate(c.rule):
            if not r: continue
            m = s.W == i
            if not m.any(): continue
            y_, x_ = np.nonzero(m); X = ((x_ + .5) / g + x0 - cam.cx) / cam.f; Y = ((y_ + .5) / g + y0 - cam.cy) / cam.f
            Q = cam.eye + (np.stack([X, Y, np.ones_like(X)], -1) @ cam.R) * s.Z[y_, x_][:, None]
            for fn, mt in r:
                ok = np.asarray(fn(s.T[y_, x_], s.A[y_, x_], Q), bool)
                if ok.any(): s.mat[y_[ok], x_[ok]] = mid(mt)
        for src, fn, dst, *mt in getattr(B, 'relabel', []):                            # parts handed to their neighbour (no seams at hips and shoulders, no body through cloth)
            if src not in lid: continue
            if dst not in lid: names.append(dst); lid[dst] = len(names) - 1
            y_, x_ = np.nonzero(s.lab == lid[src])
            if not len(y_): continue
            X = ((x_ + .5) / g + x0 - cam.cx) / cam.f; Y = ((y_ + .5) / g + y0 - cam.cy) / cam.f; Q = cam.eye + (np.stack([X, Y, np.ones_like(X)], -1) @ cam.R) * s.Z[y_, x_][:, None]; ok = fn(Q); s.lab[y_[ok], x_[ok]] = lid[dst]
            if mt: s.mat[y_[ok], x_[ok]] = mid(mt[0])
        s.mn = mn; s.hl = lid.get('head', -99)
        if not s.ink_only: s.light_pass()
        s.cel = Cel(x1 - x0, y1 - y0, x0, y0, ss=2)
        s.paint()
        if not s.ink_only: s.decals(); s.discs(); s.lines(); s.muscle_lines()
        img = s.cel.rgba(); s.anchors()
        if s.head and 'head' in lid: img = s.head_pass(img)
        return img, s.box

    def light_pass(s):
        g, N = s.g, s.N; sig = max(.7, s.px * g * .0048); m = s.hit.astype(f32); w = cv2.GaussianBlur(m, (0, 0), sig) + 1e-4
        n = cv2.GaussianBlur(N, (0, 0), sig) / w[..., None]; n = n * .8 + N * .2
        dz = np.abs(cv2.GaussianBlur(s.Z * m, (0, 0), sig) / w - s.Z); k = np.clip(1 - dz / (.045 * s.B.H / 1.76), 0, 1)[..., None]
        n = n * k + N * (1 - k); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6; s.n = n
        nd = n @ s.L.astype(f32)
        S = (nd < s.key)
        if s.cast:
            sh = s.B.c.shadow(s.cam, s.box, g, s.buf, s.L, 1.0 * s.B.H / 1.76, 2 if s.px * g > 200 else 1); S = S | (sh > 0)
        # folds between masses are dark on the side away from the light
        S = S | ((s.C < .60) & (nd < s.key + .30))
        S &= s.hit; ks = int(max(3, s.px * g * .011)) | 1
        s.S = (cv2.medianBlur(S.astype(np.uint8), min(ks, 9)) > 0) & s.hit
        h, w_ = s.Z.shape; X = ((np.arange(w_) + .5) / g + s.box[0] - s.cam.cx) / s.cam.f; Y = ((np.arange(h) + .5) / g + s.box[1] - s.cam.cy) / s.cam.f
        V = -(np.stack(np.broadcast_arrays(X[None, :], Y[:, None], np.ones((h, w_))), -1) @ s.cam.R).astype(f32); V /= np.linalg.norm(V, axis=-1, keepdims=True); s.V = V
        Hv = V + s.L.astype(f32); Hv /= np.linalg.norm(Hv, axis=-1, keepdims=True) + 1e-6
        s.sp_ = (n * Hv).sum(-1); s.fz = (n * V).sum(-1)
        s.Hl = (nd > .80) & (~s.S) & s.hit
        thin = np.isin(s.lab, [i for i, q in enumerate(s.lname) if q.startswith(('bend', 'lap', 'jks'))])
        s.Rm = [((n @ r['D'].astype(f32)) > r.get('thr', .42)) & s.hit & (s.C > .2) & (s.fz < r.get('face', .78)) & ~thin for r in s.rims]
        s.Dp = ((s.C < .40) | ((nd < -.62) & (s.C < .78))) & s.S & (s.lab != s.hl)

    # ------------------------------------------------ colour
    def polys(s, m, eps_k=1.0, min_area=8., **k):
        return mask_polys(m, s.g, s.box, eps=max(.45, s.px * .0011) * eps_k, min_area=min_area, smooth=max(3, int(3.2 * s.g + 1.5)), **k)

    def eps_of(s, nm): return max(.45, s.px * .0009) if nm.startswith(('hand', 'foot', 'neck', 'lap', 'knot', 'bend')) else max(.5, s.px * .0015)

    def paint(s):
        cel, pal, g = s.cel, s.pal, s.g; fig = s.hit.copy()
        if s.hl >= 0 and 'neck' in s.lname:
            s.mat[s.lab == s.hl] = s.mn.index(s.B.sp['skin']) if s.B.sp['skin'] in s.mn else 0
            if hasattr(s, 'S'): s.S = s.S | (s.lab == s.hl)
        s.figm = fig
        if s.ink_only: cel.fill(s.polys(fig), s.ink); return
        a = max(6., (s.px * g * .014) ** 2); pl = lambda m, e, sh, n=2, ar=1., sm=.007: [draw_contour(q, max(.8, s.px * e), True, sh, n=n) for q in mask_polys(m, g, s.box, raw=True, min_area=a * ar, smooth=max(3, int(s.px * g * sm)))]
        s.Sp = pl(s.S, .0032, .50); s.Dpp = pl(cv2.medianBlur(s.Dp.astype(np.uint8), 3), .0028, .5, 2, .5, .005); s.Hp = pl(cv2.medianBlur(s.Hl.astype(np.uint8), 5), .0030, .5, 2, 1.)
        s.Rp = [pl(cv2.medianBlur(m.astype(np.uint8), 3), .0024, .5, 2, .4, .005) for m in s.Rm]
        s.Gp = {}
        for nm in set(s.mn) & set(GLOSS):
            q = s.mn.index(nm); thr = GLOSS[nm][0]; m = (s.mat == q) & (s.sp_ > thr) & s.hit
            s.Gp[nm] = pl(cv2.medianBlur(m.astype(np.uint8), 3), .0022, .5, 3, .25, .005) if m.any() else []
            if nm == 'fv':                                                              # dark horizon inside the glass, lit lip under it
                e = (s.mat == q) & (s.fz < .42) & s.hit; s.Gp['fv2'] = pl(e, .0022, .5, 3, .3, .005)
        s.lpoly = {}
        ids = [i for i in range(len(s.lname)) if (s.lab == i).any()]; zm = {i: float(np.median(s.Z[s.lab == i])) for i in ids}; ids.sort(key=lambda i: -zm[i] - (100 if i == s.hl else 0))
        sm = max(3, int(2.8 * g + 1.5)); ks = np.ones((7, 7), np.uint8)
        for i in ids:
            nm = s.lname[i]; m = (s.lab == i)
            raw = [c for c in mask_polys(m, g, s.box, raw=True, up=2, min_area=3., smooth=sm, sig=.6) if len(c) >= 10]; cl = [clean_closed(c, s.eps_of(nm)) for c in raw]; s.lpoly[i] = (raw, cl)
            if not cl: continue
            polys = [c[0] for c in cl]
            near = np.isin(s.lab, [q for q in ids if zm[q] < zm[i] and q != i]); grow = (cv2.dilate(m.astype(np.uint8), ks) > 0) & (near | m)
            und = s.polys(grow, min_area=2.) if grow.sum() > m.sum() else []
            mats, cnt = np.unique(s.mat[m], return_counts=True); main = mats[np.argmax(cnt)]; nmn = s.mn[main]; base = pal(nmn)[1]
            def flat(col, ps): cel._path(ps); cel._src(col, 1.); cel.ctx.set_fill_rule(1); cel.ctx.fill()
            for ps in (und, polys):
                if ps: flat(base, ps)
            s.tones(polys + und, nmn)
            for q in mats:
                if q == main: continue
                mm = m & (s.mat == q)
                if mm.sum() < 4: continue
                Pm = s.polys((cv2.dilate(mm.astype(np.uint8), np.ones((2, 2), np.uint8)) & m.astype(np.uint8)) | (cv2.dilate(mm.astype(np.uint8), ks) & (grow & ~m).astype(np.uint8)), min_area=2., eps_k=.8)
                if not Pm: continue
                cel.ctx.save(); cel._path(polys + und); cel.ctx.set_fill_rule(0); cel.ctx.clip(); flat(pal(s.mn[q])[1], Pm); s.tones(Pm, s.mn[q]); cel.ctx.restore()
        s.matpoly = {}
        for i, (raw, cl) in s.lpoly.items():
            for q in np.unique(s.mat[s.lab == i]): s.matpoly.setdefault(s.mn[q], []).extend([c[0] for c in cl])

    def tones(s, clip, nm):
        cel = s.cel; lit, base, shd, deep = s.pal(nm); ctx = cel.ctx
        ctx.save(); cel._path(clip); ctx.set_fill_rule(0); ctx.clip()
        if s.Hp and s.px > 150 and nm not in ('inside', 'insideo', 'insideb'): cel.fill(s.Hp, _mixc(base, lit, .55 if nm in SKINS else .40), evenodd=True)
        if s.Sp: cel.fill(s.Sp, shd, evenodd=True)
        if s.Dpp and s.px > 110 and not nm.startswith('inside'): cel.fill(s.Dpp, _mixc(shd, deep, .62), evenodd=True)
        if nm in GLOSS and s.px > 90:
            if nm == 'fv' and s.Gp.get('fv2'): cel.fill(s.Gp['fv2'], _mixc(shd, deep, .3), evenodd=True)
            if s.Gp.get(nm): cel.fill(s.Gp[nm], _mixc(base, lit, GLOSS[nm][1] if nm != 'fv' else 1.) if nm != 'fv' else _mixc(lit, '#ffffff', .55), evenodd=True)
        for r, P in zip(s.rims, s.Rp):
            if P and not nm.startswith('inside'): cel.fill(P, _mixc(_mixc(base, lit, .5), rgb(r['col']) if isinstance(r['col'], str) else r['col'], r.get('amt', .85)), evenodd=True)
        ctx.restore()

    # ------------------------------------------------ surface decals
    def vis(s, Q, prim, tol=.035):
        p, z = s.cam.project(Q); g = s.g; h, w = s.Z.shape; u = np.floor((p[:, 0] - s.box[0]) * g).astype(int); v = np.floor((p[:, 1] - s.box[1]) * g).astype(int)
        ok = (u >= 0) & (v >= 0) & (u < w) & (v < h) & (z > .06); o = np.zeros(len(p), bool); i = np.nonzero(ok)[0]
        if len(i):
            zb = s.Z[v[i], u[i]]; lb = s.lab[v[i], u[i]]; okl = s._glab(s.B.c.I[prim][2]) if prim is not None else np.ones(len(s.lname), bool)
            o[i] = (np.abs(zb - z[i]) < tol * s.B.H / 1.76) & (lb >= 0) & okl[np.maximum(lb, 0)]
        return p, o

    def _glab(s, g):
        if not hasattr(s, '_gl'): s._gl = {}
        if g not in s._gl:
            c = s.B.c; names = {c.lab[i] for i in range(len(c.lab)) if c.I[i][2] == g}; s._gl[g] = np.array([n in names for n in s.lname])
        return s._gl[g]

    def ij(s, x, y):
        u = int((x - s.box[0]) * s.g); v = int((y - s.box[1]) * s.g); h, w = s.Z.shape
        return (u, v) if 0 <= u < w and 0 <= v < h else (None, None)

    def decals(s):
        B, cel, pal = s.B, s.cel, s.pal; lvl = 0 if s.px < 190 else 1 if s.px < 380 else 2
        for k, d in enumerate(B.deco):
            if d.get('pri', 0) > lvl - (s.px < 260) or d['kind'] != 'shard' or s.px < 120: continue
            t, p = d['t'], d['p']; cl = np.vstack([np.c_[t, p], [[t[0], p[0]]]]); Q = B.c.point(d['prim'], *resample2(cl, .04).T); xy, ok = s.vis(Q, d['prim'])
            if ok.mean() < .60: continue
            V = xy[ok]
            if len(V) < 4: continue
            mt = B.c.mat[d['prim']]; lit, base, shd, deep = pal(mt); cx_, cy_ = V.mean(0); u, v = s.ij(cx_, cy_); inS = s.S[v, u] if u is not None else False
            if inS: continue
            col = _mixc(base, shd, .95)
            V = draw_contour(V, max(.5, s.px * .0016), True, .6, n=3)
            cel.ctx.save(); cel._path(s.matpoly.get(mt, [V])); cel.ctx.set_fill_rule(0); cel.ctx.clip(); cel.fill(V, col); cel.ctx.restore()

    def discs(s):
        """kanji discs of the gi, dropped on the cloth"""
        B, sk = s.B, s.B.sk
        if not B.discs or s.px < 60: return
        try: from . import letters as LT
        except Exception: return
        for d in B.discs:
            if d.get('skip'): continue
            out = sk.Rc[:, 1] * d['side']; rt = sk.Rc[:, 0]; up = sk.Rc[:, 2]; o = B.C(d['x'] * 1., 0, d['z']); right = -rt * d['side']          # unit x to the viewer's right when facing it
            if (out @ (s.cam.eye - o)) / np.linalg.norm(s.cam.eye - o) < .08: continue
            r = d['r'] * B.m
            def T(U, o=o, out=out, right=right, up=up, r=r):
                U = np.atleast_2d(np.asarray(U, float)); P = o + right * (U[:, :1] * r) - up * (U[:, 1:2] * r) + out * .40 * B.m
                P3, hit = B.c.drop(P, -out, .8, labels=('gi',)); P3 = np.where(hit[:, None], P3, o + right * (U[:, :1] * r) - up * (U[:, 1:2] * r) + out * .12 * B.m)
                return s.cam.project(P3)[0]
            a = np.linspace(0, 2 * math.pi, 25)[:-1]; ring = np.c_[np.cos(a), np.sin(a)]
            P = o + right * (ring[:, :1] * r) - up * (ring[:, 1:2] * r) + out * .40 * B.m; P3, hit = B.c.drop(P, -out, .8, labels=('gi',)); xy, ok = s.vis(P3, B.prim['gi'], .05)
            if (ok & hit).mean() < .35: continue
            rp = float(np.hypot(*(xy.max(0) - xy.min(0)))) / 2.8
            lit, base, shd, deep = s.pal('cape'); u, v = s.ij(*xy.mean(0)); inS = bool(s.S[v, u]) if u is not None else False
            li = [i for i, n in enumerate(s.lname) if n == 'gi']; clip = [c[0] for i in li for c in s.lpoly.get(i, ([], []))[1]]
            if not clip: continue
            s.cel.ctx.save(); s.cel._path(clip); s.cel.ctx.set_fill_rule(0); s.cel.ctx.clip()
            if rp > 5:
                LT.disc(s.cel, d['ch'], T, white=base, ink=s.ink, w=max(.8, min(s.wf * 1.1, rp * .10)))
                s.cel.ctx.save(); s.cel._path([T(ring)]); s.cel.ctx.clip(); s.cel.fill(s.Sp, shd, .55, evenodd=True); s.cel.ctx.restore()
            else: s.cel.fill(xy, base if not inS else shd); s.cel.fill(xy * .45 + xy.mean(0) * .55, s.ink, .8)
            s.cel.ctx.restore()

    # ------------------------------------------------ ink
    def stroke(s, P, w, col=None, seed=0, taper=(.2, .2), split=True, press=.34, wfun=None):
        P = np.asarray(P, float)
        if len(P) < 4: return
        Q = resample(P, 1.6)
        if len(Q) < 4: return
        col = s.ink if col is None else col
        if split:
            T = np.diff(Q, axis=0); a = np.unwrap(np.arctan2(T[:, 1], T[:, 0])); k = 3; tn = np.abs(a[k:] - a[:-k]) if len(a) > k else np.zeros(0); cut = [0]; j = 0; th_ = .9 if s.px > 330 else 1.25
            while j < len(tn):
                if tn[j] > th_:
                    e = j
                    while e + 1 < len(tn) and tn[e + 1] > th_: e += 1
                    i = int(j + np.argmax(tn[j:e + 1])) + 2
                    if i - cut[-1] > 4: cut.append(i)
                    j = e + k
                j += 1
            cut.append(len(Q) - 1)
        else: cut = [0, len(Q) - 1]
        m_ = max(1, len(Q) - 1)
        for n_, (a_, b_) in enumerate(zip(cut[:-1], cut[1:])):
            a0, b0 = max(0, a_ - (2 if n_ else 0)), min(len(Q) - 1, b_ + (2 if n_ < len(cut) - 2 else 0)); g = Q[a0:b0 + 1]
            if len(g) < 4: continue
            L = len(g) * 1.6; c_ = .14 if s.px > 330 else .07; t0 = taper[0] if n_ == 0 else min(c_, 9 / L); t1 = taper[1] if n_ == len(cut) - 2 else min(c_, 9 / L)
            wf = None if wfun is None else (lambda t, a0=a0, b0=b0: wfun((a0 + t * (b0 - a0)) / m_))
            s.cel.line(g, w, col, taper=(t0, t1), press=press, seed=seed * 17 + n_, tremor=.2, tmin=.34 if s.px > 330 else .5, wfun=wf)

    def lines(s):
        g, Z, lab = s.g, s.Z, s.lab; H_, W_ = Z.shape; B = s.B; sc = B.H / 1.76; mats = s.mat; big = np.where(s.hit, Z, 1e6).astype(f32); up = 2; hands = [n.startswith('hand') for n in s.lname]
        sheet = np.array([n in ('capesh', 'lapl', 'lapr') or n.startswith(('bend', 'jks')) for n in s.lname] + [False])
        # 1. the silhouette of the whole figure: long gestures that do not know where one part ends and the next begins
        body = s.hit & (lab != s.hl); fine = np.array([.66 if n.startswith('hand') else .8 if n.startswith(('bend', 'knot', 'foot')) else 1. for n in s.lname] + [1.])
        for ci, c in enumerate(mask_polys(body, g, s.box, raw=True, up=2, min_area=4., smooth=max(3, int(2.8 * g + 1.5)), sig=.6)):
            n = len(c)
            if n < 12: continue
            T = np.roll(c, -2, 0) - np.roll(c, 2, 0); T /= np.hypot(*T.T)[:, None] + 1e-9; Nn = np.c_[T[:, 1], -T[:, 0]]
            def samp(o): q = (c + Nn * o / g - [s.box[0], s.box[1]]) * g; return np.clip(np.floor(q[:, 0]).astype(int), 0, W_ - 1), np.clip(np.floor(q[:, 1]).astype(int), 0, H_ - 1)
            u1, v1 = samp(1.6); u2, v2 = samp(-1.6)
            if body[v1, u1].mean() > body[v2, u2].mean(): (u1, v1), (u2, v2) = (u2, v2), (u1, v1); Nn = -Nn
            lo = lab[v1, u1]; li = lab[v2, u2]; cls = np.where((lo < 0) | ((lo == s.hl) & (big[v1, u1] - big[v2, u2] > .01 * sc)), 1, 0)
            wl = fine[li]; s.runs(c, cls, 0, 70 + ci, Nn, clean_closed(c, max(.5, s.px * .0011)), wl=wl)
        # 2. overlaps and seams, part by part
        for li, nm in enumerate(s.lname):
            if li == s.hl: continue
            m = lab == li
            if m.sum() < 5: continue
            raws, cls_ = s.lpoly.get(li, ([], []))
            for ci, c in enumerate(raws):
                n = len(c); T = np.roll(c, -2, 0) - np.roll(c, 2, 0); T /= np.hypot(*T.T)[:, None] + 1e-9; Nn = np.c_[T[:, 1], -T[:, 0]]
                def samp(off):
                    q = (c + Nn * off / g - [s.box[0], s.box[1]]) * g; return np.clip(np.floor(q[:, 0]).astype(int), 0, W_ - 1), np.clip(np.floor(q[:, 1]).astype(int), 0, H_ - 1)
                u1, v1 = samp(1.6); u2, v2 = samp(-1.6)
                if (lab[v1, u1] == li).mean() > (lab[v2, u2] == li).mean(): (u1, v1), (u2, v2) = (u2, v2), (u1, v1); Nn = -Nn
                lo = lab[v1, u1]; zi = big[v2, u2]; zo = big[v1, u1]; jump = zo - zi; ishand = hands[li]; oth_hand = np.array([hands[q] if q >= 0 else False for q in lo])
                layer = nm in OUTER
                thr = np.where(sheet[li] | sheet[lo], .0012, np.where(oth_hand & ishand, .004, np.where(layer, .0035, .011))) * sc; dm = mats[v1, u1] != mats[v2, u2]; nd = np.sum(s.N[v1, u1] * s.N[v2, u2], -1)
                cls = np.zeros(n, int); cls[(lo >= 0) & (lo != li) & (lo != s.hl) & (jump > thr)] = 2
                flat = (lo >= 0) & (lo != li) & (lo != s.hl) & (np.abs(jump) <= thr) & (lo > li); cls[flat & (dm | (nd < .72))] = 3; cls[flat & ishand & oth_hand & (nd < .93)] = 3
                s.runs(c, cls, li, ci, Nn, cls_[ci])
        for li, nm in enumerate(s.lname):
            if li == s.hl: continue
            ms = np.unique(mats[lab == li])
            if len(ms) < 2: continue
            for q in ms[1:]:
                m = (lab == li) & (mats == q)
                if m.sum() < 8: continue
                for ci, c in enumerate(mask_polys(m, g, s.box, raw=True, up=up, min_area=4., smooth=5)):
                    uu = (c - [s.box[0], s.box[1]]) * g; T = np.roll(c, -2, 0) - np.roll(c, 2, 0); T /= np.hypot(*T.T)[:, None] + 1e-9; Nn = np.c_[T[:, 1], -T[:, 0]]
                    def lb(off):
                        z = uu + Nn * off; u = np.clip(np.floor(z[:, 0]).astype(int), 0, W_ - 1); v = np.clip(np.floor(z[:, 1]).astype(int), 0, H_ - 1); return lab[v, u], mats[v, u]
                    l1, m1 = lb(1.6); l2, m2 = lb(-1.6); s.runs(c, np.where((l1 == li) & (l2 == li) & (m1 != m2), 3, 0), 900 + li * 10 + int(q), ci, Nn)
        lvl = 0 if s.px < 190 else 1 if s.px < 380 else 2
        for k, d in enumerate(B.deco):
            if d['kind'] != 'line' or d.get('pri', 0) > lvl or s.px < 110: continue
            c = np.c_[d['t'], d['p']]; C2 = spl(c, n=7) if not d.get('straight') else resample2(c, .06); Q = B.c.point(d['prim'], C2[:, 0], C2[:, 1], .001); xy, ok = s.vis(Q, d['prim'], .028)
            for a, b in _true_runs(ok, 3):
                seg = xy[a:b]
                if np.hypot(*np.diff(seg, axis=0).T).sum() < max(4., s.px * .012): continue
                s.cel.line(spl(seg, n=4), s.wf * d['w'] * 1.05, s.ink, taper=(.12, .55), press=.3, seed=_hash(s.seed, k) & 0xffff, tmin=.12)
        # authored pen lines given as 3-D polylines on the surface (anatomy of the trunk, folds of sheets)
        for k, d in enumerate(B.seams):
            if d.get('pri', 0) > lvl or s.px < d.get('minpx', 105) or (s.px < 200 and d.get('w', 1.) < .85): continue
            xy, ok = s.vis(d['Q'], None, d.get('tol', .022))
            li = d.get('labels')
            if li is not None:
                ids = [i for i, n in enumerate(s.lname) if n in li]
                for i, (x, y) in enumerate(xy):
                    if ok[i]: u, v = s.ij(x, y); ok[i] = u is not None and s.lab[v, u] in ids
            tp = d.get('taper', (.35, .55))
            for a, b in _true_runs(ok, 3):
                seg = xy[a:b]
                if np.hypot(*np.diff(seg, axis=0).T).sum() < max(3., s.px * .010): continue
                u, v = s.ij(*seg[len(seg) // 2]); mt = s.mn[s.mat[v, u]] if u is not None else 'skin'; lit, base, shd, deep = s.pal(mt); col = _mixc(deep, s.ink, .6 if mt in SKINS else .8)
                whole = b - a > .8 * len(xy); s.cel.line(spl(seg, n=3), s.wf * d.get('w', 1.) * 1.15 * s.ml, col, taper=tp if whole else (.3, .3), press=.25, seed=_hash(s.seed, 'seam', k) & 0xffff, tmin=.08)

    def muscle_lines(s):
        """pen lines in the valleys between muscles: drawn only where the valley is deep, thick where it is deepest"""
        if s.px < 95 or s.ml <= 0: return
        g, mg = s.g, s.mg; H_, W_ = s.Z.shape; sc = s.B.H / 1.76; Nc = s.N @ s.cam.R.T.astype(f32); lvl = 0 if s.px < 200 else 1 if s.px < 400 else 2
        w_of = []
        for n in s.mgn:
            k = n.rstrip('LR').rstrip('0123456789-'); w_of.append(LINEW.get(k, LINEW.get(n, 0.)))
        w_of = np.array(w_of + [0.]); w_of[w_of < (.78 if s.px < 230 else .55 if s.px < 400 else 0.)] = 0.; ids = [i for i, n in enumerate(s.mgn) if w_of[i] > 0 and (mg == i).any()]
        off = max(1.6, min(3.0, s.px * g * .006)); col_cache = {}
        for i in ids:
            m = (mg == i)
            if m.sum() < 12: continue
            for ci, c in enumerate(mask_polys(m, g, s.box, raw=True, up=2, min_area=8., smooth=max(4, int(s.px * g * .010)))):
                n = len(c)
                if n < 12: continue
                uu = (c - [s.box[0], s.box[1]]) * g; T = np.roll(c, -3, 0) - np.roll(c, 3, 0); T /= np.hypot(*T.T)[:, None] + 1e-9; Nn = np.c_[T[:, 1], -T[:, 0]]
                def at(o): z = uu + Nn * o; return np.clip(np.floor(z[:, 0]).astype(int), 0, W_ - 1), np.clip(np.floor(z[:, 1]).astype(int), 0, H_ - 1)
                u1, v1 = at(off); u2, v2 = at(-off)
                if (mg[v1, u1] == i).mean() > (mg[v2, u2] == i).mean(): (u1, v1), (u2, v2) = (u2, v2), (u1, v1); Nn = -Nn
                o = mg[v1, u1]; same = (s.lab[v1, u1] == s.lab[v2, u2]) & (o >= 0) & (o != i); jump = s.Z[v1, u1] - s.Z[v2, u2]
                kap = ((Nc[v2, u2, :2] - Nc[v1, u1, :2]) * Nn).sum(-1)
                ld = s.linedir.get(s.mgn[i].rstrip('LR').rstrip('0123456789-'))
                if ld is not None:
                    dv = ld[0](s.B.sk, s.mgn[i][-1]); d2 = np.array([s.cam.R[0] @ dv, s.cam.R[1] @ dv]); twin = np.array([s.mgn[q].rstrip('LR') == s.mgn[i].rstrip('LR') if q >= 0 else False for q in o]); kap = np.where(twin, kap, kap * np.clip((Nn @ d2 - ld[1]) / .25, 0, 1))
                wo = w_of[np.maximum(o, -1)]; first = (wo <= 0) | (i < o)
                st = np.where(same & first & (np.abs(jump) < .05 * sc), kap, 0.) ; st = np.where(same & (jump > .010 * sc), np.maximum(st, .95), st)
                st = _smooth(np.c_[st, st], 7, True)[:, 0]
                lo_, hi_ = .26, .46
                idx = _hyst(st, lo_, hi_)
                for a, b in idx:
                    P = c[np.arange(a, b) % n]; L = np.hypot(*np.diff(P, axis=0).T).sum()
                    if L < s.px * .016: continue
                    sv = np.clip((st[np.arange(a, b) % n] - lo_) / .55, 0, 1); ww = w_of[i]
                    mt = s.mn[s.mat[v2[a % n], u2[a % n]]]
                    if mt not in col_cache: lit, base, shd, deep = s.pal(mt); col_cache[mt] = _mixc(deep, s.ink, .55 if mt in SKINS else .75)
                    tt = np.linspace(0, 1, len(sv)); wf = lambda t, sv=sv, tt=tt: .42 + .75 * np.interp(t, tt, sv)
                    s.cel.line(draw_contour(P, max(.5, s.px * .0012), False, .9, n=4), s.wf * 1.15 * ww * s.ml, col_cache[mt], taper=(.22, .45) if (a + ci) % 2 else (.45, .22), press=.25, seed=_hash(s.seed, i, ci) & 0xffff, tmin=.08, wfun=wf)

    def runs(s, c, cls, li, ci, Nn, clean=None, wl=None):
        n = len(c); g = s.g; k = 5
        for v in (1, 2, 3):
            m = (cls == v).astype(np.uint8); mm = np.r_[m[-k:], m, m[:k]][None]; mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, np.ones((1, 2 * k + 1), np.uint8))[0, k:-k]; cls = np.where((cls == 0) & (mm > 0), v, cls)
        ch = np.nonzero(cls != np.roll(cls, 1))[0]
        if len(ch) == 0:
            if cls[0] == 0: return
            segs = [(0, n, cls[0], True)]
        else: segs = [(a, b if b > a else b + n, cls[a], False) for a, b in zip(ch, np.r_[ch[1:], ch[0] + n])]
        for a, b, v, closed in segs:
            if v == 0: continue
            idx = np.arange(a, b + 1) % n; P = c[idx]
            if clean is not None:
                Q, ix, nn_ = clean; i0 = clean_index(a, ix, nn_, n); M = len(Q); i1 = i0 + M if closed else clean_index(b % n, ix, nn_, n)
                if i1 <= i0: i1 += M
                ii = np.arange(math.ceil(i0), math.floor(i1) + 1) % M
                if len(ii) >= 3: P = Q[ii]
            L = np.hypot(*np.diff(P, axis=0).T).sum()
            if L < max(3.5, s.px * .010) and not closed: continue
            w = {1: s.wo, 2: s.wo * .80, 3: s.wf * 1.1}[int(v)]; nm = s.lname[li] if li < 900 and wl is None else ''
            if nm.startswith('hand'): w *= .66 if s.px > 250 else .8
            if nm.startswith(('lap', 'bend', 'knot', 'ucol')): w *= .78
            l2 = np.array([s.lc[0], -s.lc[1]]) / (np.hypot(s.lc[0], s.lc[1]) + 1e-6); lt = _smooth(np.c_[Nn[idx] @ l2, np.zeros(len(idx))], max(5, int(len(idx) * .12)), False)[:, 0]; tt = np.linspace(0, 1, len(lt))
            # thick-thin: heavy on the side that faces away from the light, thin toward it, a swell in the middle of every gesture
            if wl is not None: ww_ = _smooth(np.c_[wl[idx], wl[idx]], 9, False)[:, 0]
            else: ww_ = np.ones(len(idx))
            lt = lt + 0.; wf = lambda t, lt=lt, tt=tt, ww_=ww_: np.interp(t, tt, ww_) * (1 - .36 * np.interp(t, tt, lt)) * (.86 + .20 * np.sin(np.clip(t, 0, 1) * math.pi))
            tp = (.02, .02) if closed else ((.14, .14) if v == 1 else (.26, .34))
            s.stroke(P, w, s.line_col(c[idx], Nn[idx]), seed=_hash(s.seed, li, ci, int(v)) & 0xffff, taper=tp, wfun=wf)

    def line_col(s, c, Nn):
        """ink that leans toward the colour it bounds, and toward the rim light where that light falls"""
        q = (c - Nn * 2.2 / s.g - [s.box[0], s.box[1]]) * s.g; u = np.clip(q[:, 0].astype(int), 0, s.Z.shape[1] - 1); v = np.clip(q[:, 1].astype(int), 0, s.Z.shape[0] - 1); m = s.mat[v, u]; m = m[m >= 0]
        if not len(m): return s.ink
        mt = s.mn[int(np.bincount(m).argmax())]; col = _mixc(s.ink, s.pal(mt)[3], .30 if mt in SKINS else .18)
        for r in s.rims:
            d2 = np.array([s.cam.R[0] @ r['D'], s.cam.R[1] @ r['D']]); n = np.hypot(*d2)
            if n > .3: k = float(np.clip((Nn @ (d2 / n)).mean() - .25, 0, .6)) * r.get('line', .55); col = _mixc(col, _mixc(r['col'], s.ink, .45), k)
        return col

    # ------------------------------------------------ anchors + head
    def anchors(s):
        cam, sk = s.cam, s.B.sk; f = s.fig; p = lambda q: tuple(cam.project(np.asarray(q, float))[0]); hl = s.B.b['hand'] * sk.H
        f.info = dict(px=s.px, box=s.box, hands={q: p(sk.wr[q] + sk.Rhand[q][:, 1] * hl * .45) for q in 'LR'}, hands3={q: sk.wr[q] + sk.Rhand[q][:, 1] * hl * .45 for q in 'LR'}, wrists={q: p(sk.wr[q]) for q in 'LR'}, fists={q: p(sk.spot('fist' + q)) for q in 'LR'}, palms={q: p(sk.spot('palm' + q)) for q in 'LR'},
                      feet={q: p(sk.an[q]) for q in 'LR'}, soles={q: p(sk.spot('sole' + q)) for q in 'LR'}, elbows={q: p(sk.el[q]) for q in 'LR'}, knees={q: p(sk.kn[q]) for q in 'LR'}, shoulders={q: p(sk.sh[q]) for q in 'LR'}, pelvis=p(sk.pel), chest=p(sk.chest), stomach=p(sk.spot('stomach')), depth=float(cam.cam(sk.pel)[2]), skel=sk)
        from . import h3_head
        f.info['head'] = h3_head.pose_info(cam, sk)
        if hasattr(s.B, 'tailpts'): f.info['tail'] = cam.project(s.B.tailpts)[0]

    def head_pass(s, img):
        from . import h3_head
        hi = s.fig.info['head']; x0, y0, x1, y1 = s.box; m = int(hi['s'] * 2.9) + 8
        nx0, ny0, nx1, ny1 = min(x0, int(hi['x'] - m)), min(y0, int(hi['y'] - m)), max(x1, int(hi['x'] + m)), max(y1, int(hi['y'] + m)); nx0, ny0, nx1, ny1 = max(nx0, -260), max(ny0, -260), min(nx1, s.cam.w + 260), min(ny1, s.cam.h + 260)
        if nx1 - nx0 < 4 or ny1 - ny0 < 4: return img
        body = np.zeros((ny1 - ny0, nx1 - nx0, 4), f32); paste(body, img, x0 - nx0, y0 - ny0); roi = (nx0, ny0, nx1, ny1)
        back, hd = h3_head.render(s, hi, roi)
        if getattr(s.fig, 'head_warp', None) is not None: hd = s.fig.head_warp(hd, roi, hi)
        for r in s.rims:
            q = np.array(r['dir'], float); q = q[:2] / (np.hypot(*q[:2]) + 1e-6); d_ = max(1.0, hi['s'] * .06); hd = kit.rim(hd, -q[0] * d_, q[1] * d_, r['col'], r.get('amt', .85) * .9, .5)
        zref = hi['z']; near = s.hit & (s.lab != s.hl) & (s.Z < zref - .16 * hi['u'])
        for i, n in enumerate(s.lname):
            if n in ('neck', 'torso') and hi['R'][2, 2] > -.35: near &= ~((s.lab == i) & (s.Z > zref - 1.1 * hi['u']))
        fm = cv2.resize(near.astype(f32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR); tmp = np.zeros(body.shape[:2] + (1,), f32); paste(tmp, fm[..., None], x0 - nx0, y0 - ny0); F = np.clip((tmp[..., 0] - .5) * 6 + .5, 0, 1)
        r = max(1, int(round(s.wo * .6))); F = cv2.dilate(F, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        if back is not None:
            # back hair sits behind the body, except where the head itself is nearer than the body
            body_ = back * (1 - body[..., 3:4]) + body
        else: body_ = body
        out = body_ * (1 - hd[..., 3:4]) + hd; out = out * (1 - F[..., None] * body[..., 3:4]) + body * F[..., None]
        s.box = roi; s.fig.info['box'] = roi
        return out


def resample2(P, step):
    P = np.asarray(P, float); d = np.hypot(*np.diff(P, axis=0).T); s_ = np.r_[0, np.cumsum(d)]; n = max(len(P), int(s_[-1] / step) + 2); t = np.linspace(0, s_[-1], n)
    return np.c_[np.interp(t, s_, P[:, 0]), np.interp(t, s_, P[:, 1])]


def _true_runs(ok, n):
    out = []; a = None
    for i, v in enumerate(list(ok) + [False]):
        if v and a is None: a = i
        if not v and a is not None:
            if i - a >= n: out.append((a, i))
            a = None
    return out


def _hyst(st, lo, hi):
    """closed-curve runs where st > lo that contain a value > hi -> [(a, b)] (b may exceed n)"""
    n = len(st); on = st > lo
    if on.all(): return [(0, n)] if st.max() > hi else []
    if not on.any(): return []
    k = int(np.argmin(on)); q = np.roll(on, -k); out = []
    for a, b in _true_runs(q, 4):
        if np.roll(st, -k)[a:b].max() > hi: out.append((a + k, b + k))
    return out
