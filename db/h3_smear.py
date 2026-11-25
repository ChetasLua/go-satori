"""Smears for db.hero3: swept limbs with dry-brush tails and ink, after-image limbs, whole-figure speed tails."""
import numpy as np, math, cv2
from . import W, H
from .paint import rgb, f32, Canvas
from .ink import Cel, spl, resample
from . import kit
from .h3_pose import mix_pose


def _skel(fig, a, b, t):
    d = dict(b); d['pose'] = mix_pose(a['pose'], b['pose'], t)
    for k in ('x', 'y', 'z', 'heading', 'pitch', 'roll'): d[k] = a.get(k, 0.) * (1 - t) + b.get(k, 0.) * t
    d['pivot'] = b.get('pivot', 'ground'); d['look'] = None; d['reach'] = {q: dict(r if isinstance(r, dict) else dict(p=r), w=(r.get('w', 1.) if isinstance(r, dict) else 1.) * t) for q, r in (b.get('reach') or {}).items() if r is not None}
    return fig.skel_of(d)


def limb_points(sk, limb):
    """chain of (world point, radius) from root to tip"""
    s = limb[-1]; m = sk.H / 1.76 * sk.b['mass'] * sk.b['limb']; hl = sk.b['hand'] * sk.H
    if limb.startswith('arm'): return [(sk.sh[s], .060 * m), (sk.el[s], .046 * m), (sk.wr[s], .036 * m), (sk.wr[s] + sk.Rhand[s][:, 1] * hl * .10, .046 * m), (sk.wr[s] + sk.Rhand[s][:, 1] * hl * .60, .052 * m)]
    f = sk.Rfoot[s]; return [(sk.hip[s], .085 * m), (sk.kn[s], .070 * m), (sk.an[s], .050 * m), (sk.an[s] + f[:, 1] * .04, .055 * m), (sk.an[s] + f[:, 1] * .16 * m - f[:, 2] * .03, .050 * m)]


def limb_mats(fig, limb):
    from .h3_body import SPEC
    sp = dict(SPEC[fig.char]); sp.update(fig.spec); skin = sp['skin']
    if limb.startswith('arm'):
        suit = sp['top'] == 'suit' and not sp.get('shattered'); up = sp['cloth'] if suit else skin
        return dict(up=up, low=sp.get('band') or (sp['cloth'] if suit else skin), end=sp.get('glove') or skin)
    up = sp['cloth'] if sp['bottom'] in ('gi', 'suit') else skin; return dict(up=up, low=dict(goku='boot', vegeta='bootw', frieza='fw', piccolo='shoeb', kid='boot')[sp['shoe']], end=dict(goku='boot', vegeta='bootw', frieza='fw', piccolo='shoeb', kid='boot')[sp['shoe']])


def limb_smear(fig, cam, a, b, limb='armR', pal=None, n=24, seed=0, tail=.85, edge=True, lines=10):
    """premultiplied layer: the limb swept from state a to state b (a = where it was, b = now).  Draw UNDER the figure posed at b (or alone for a one-frame smear)."""
    from .h3_draw import Pal
    pal = pal if isinstance(pal, Pal) else Pal(pal); key = limb_mats(fig, limb); r = np.random.default_rng(seed + 17); ts = np.linspace(0, 1, n); ch = [limb_points(_skel(fig, a, b, t), limb) for t in ts]
    Pw = np.array([[q[0] for q in c] for c in ch]); R = np.array([[q[1] for q in c] for c in ch]); xy, z = cam.project(Pw); sc = cam.f / np.maximum(z, .05); L = np.zeros((cam.h, cam.w, 4), f32)
    if (z < .08).any(): return L
    travel = np.hypot(*np.diff(xy[:, -1], axis=0).T).sum()
    if travel < 6: return L
    cv = Canvas(cam.h, cam.w, color='#000000', layer=True); S = []
    for (i, j) in [(1, 2), (2, 3), (3, 4)]:
        w = np.hypot(*(xy[-1, j] - xy[-1, i])); m = max(2, int(w / 5.))
        for u in np.linspace(0, 1, m, endpoint=False): S.append((i, j, u + r.uniform(0, 1. / m)))
    tipw = float(R[-1, 4] * sc[-1, 4])
    for e in np.linspace(-1., 1.15, max(4, int(tipw / 3.0))): S.append((3, 4, 1 + e * .55))
    Pl, Wd, Col, Dr = [], [], [], []
    for q in r.permutation(len(S)):
        i, j, u = S[q]; rad = (i + u) / 4.; t0 = np.clip((1 - tail) + tail * (1 - rad) * .75 + abs(r.normal(0, .14)), 0, .82) if rad < .95 else np.clip((1 - tail) + abs(r.normal(0, .10)), 0, .6)
        tt = np.linspace(t0, 1., 14); line = np.stack([np.interp(tt, ts, xy[:, i, c] * (1 - u) + xy[:, j, c] * u) for c in (0, 1)], -1); mt = key['end' if i >= 2 else 'low' if u > .5 else 'up']; lit, base, shd, deep = pal(mt)
        Pl.append(line); Wd.append(max(1.2, r.uniform(2.2, 4.6) * max(.5, sc[-1, j] / 420.))); Col.append(base if r.random() < .6 else (shd if r.random() < .6 else lit)); Dr.append(r.uniform(.15, .55))
    cv.strokes(np.array(Pl)[:, ::-1], np.array(Wd), np.array(Col), op=np.ones(len(Pl)), dry=np.array(Dr), seed=seed, shape='taper', bristle=.10); L = cv.layer(); cel = Cel(cam.w, cam.h, 0, 0, ss=1)
    if edge:
        cel.line(spl(xy[int(n * .25):, 4] + 0, n=4), max(1.4, tipw * .10), pal.ink, taper=(.85, .05), seed=seed, press=.4); cel.line(spl(xy[int(n * .5):, 2], n=4), max(1.2, tipw * .07), pal.ink, taper=(.9, .05), seed=seed + 1, press=.4)
    for q in range(lines):
        u = r.uniform(.35, 1.12); i = min(3, int(u * 4)); v = u * 4 - i; tt = np.linspace(r.uniform(.0, .5), r.uniform(.75, 1.), 12); ln = np.stack([np.interp(tt, ts, xy[:, i, c] * (1 - v) + xy[:, i + 1, c] * v) for c in (0, 1)], -1)
        cel.line(ln, r.uniform(.8, 1.8) * max(.7, sc[-1, 4] / 500.), pal.ink if r.random() < .65 else '#f4efe8', taper=(.8, .2), seed=seed + q, press=.4)
    e = cel.rgba(); return L * (1 - e[..., 3:4]) + e


def after_limb(fig, cam, a, b, limb='armR', ts=(.35, .65), pal=None, seed=0, w=2.2):
    """ink-only after-image limbs at fractions ts between two states"""
    from .h3_draw import Pal
    pal = pal if isinstance(pal, Pal) else Pal(pal); cel = Cel(cam.w, cam.h, 0, 0, ss=2)
    for k, t in enumerate(ts):
        ch = limb_points(_skel(fig, a, b, t), limb); Q, z = cam.project(np.array([c[0] for c in ch])); r = np.array([c[1] for c in ch]) * cam.f / np.maximum(z, .06); T = np.gradient(Q, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; A, B = Q + N * r[:, None], Q - N * r[:, None]
        for S in (A, B): cel.line(spl(S, n=6), w * (.6 + .5 * t), pal.ink, taper=(.5, .2), seed=seed + k, press=.5)
        cel.line(spl(np.vstack([A[-1], Q[-1] + T[-1] * r[-1] * 1.1, B[-1]]), n=6), w * (.6 + .5 * t), pal.ink, taper=(.3, .3), seed=seed + 9 + k)
    return cel.rgba()


def tails(layer, v, n=60, seed=0, width=(2., 9.), length=(.35, 1.0), dark=.75, dry=.5, ink=True, where=None):
    """dry-brush speed tails off the trailing edge of a figure layer moving by v (px).  Returns a layer to put UNDER it"""
    h, w = layer.shape[:2]; a = layer[..., 3]; v = np.asarray(v, float); L = np.hypot(*v)
    if L < 3: return np.zeros_like(layer)
    d = v / L; k = max(6., L * .12); sh = kit.shifted(a, d[0] * k, d[1] * k); m = (a > .6) & (sh < .4)
    if where is not None: m &= where
    ys, xs = np.nonzero(m)
    if len(xs) == 0: return np.zeros_like(layer)
    r = np.random.default_rng(seed + 5); q = r.choice(len(xs), min(n, len(xs)), replace=False); cv = Canvas(h, w, color='#000000', layer=True); Pl, Wd, Cl = [], [], []; nrm = np.array([-d[1], d[0]])
    for i in q:
        p = np.array([xs[i], ys[i]], float) + d * 6; ln = L * r.uniform(*length); bow = r.normal(0, .05) * ln; u = np.linspace(0, 1, 9)[:, None]
        Pl.append(p - d * ln * u + nrm * bow * np.sin(u * math.pi)); Cl.append(layer[ys[i], xs[i], :3] / max(layer[ys[i], xs[i], 3], .2) * r.uniform(dark, 1.)); Wd.append(r.uniform(*width))
    cv.strokes(np.array(Pl), np.array(Wd), np.array(Cl), dry=dry, seed=seed, shape='taper', bristle=.1); out = cv.layer()
    if ink:
        cel = Cel(w, h, 0, 0, ss=1)
        for i in q[::4]: p = np.array([xs[i], ys[i]], float); ln = L * r.uniform(.4, 1.1); cel.line(np.array([p - d * ln, p - d * ln * .5, p]), r.uniform(.8, 1.9), '#17131a', taper=(.9, .1), seed=int(i) & 255)
        e = cel.rgba(); out = out * (1 - e[..., 3:4]) + e
    return out
