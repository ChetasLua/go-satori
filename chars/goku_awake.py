"""g3: the close-up of the film as separate layers, so that it can hold still and breathe.   L = layers(state) -> dict(back, body, face, front, G, X, hr)"""
import numpy as np, math, cv2
from db import W, H as HH, comp
from db.ink import Cel, spl, resample
from db.comp import blur, screen, radial, over
from db.kit import para, shifted
from db.paint import rgb, f32, smooth, ramp, fbm, grid
from . import goku, goku_hair as GH, goku_light as GL, goku_head as GF, goku_paint as GP, goku_heroes as GX
from .goku_head import GHead
from .goku_hair import Hair, mixc, surf, _b
A = np.array
D = math.pi / 180
SILV = dict(hi='#ffffff', light='#e3e9f4', base='#b3bccd', shade='#666e84', deep='#2d3040', line='#0e0e16', root='#3a3e4e', cap='#4c5164')
GOLDH = dict(hi='#fff8d0', light='#ffe06a', base='#f5bb2c', shade='#b8700e', deep='#5c2c08', line='#1e0d04', root='#6a3408', cap='#6a3408')
CAM = dict(cx=850, cy=528, s=398, yaw=22, pitch=-7, roll=6)


def head(**k): q = dict(CAM, **k); return GHead().pose(q['cx'], q['cy'], q['s'], q['yaw'] * D, q['pitch'] * D, q['roll'] * D)


def light(hairpal):
    L = dict(GL.light('ui')); L = dict(L, brow='#34343f', line='#120b0d', hair=dict(L['hair'], **hairpal)); L['hair']['dark'] = False; return L


def style(t):
    """hair shape: 0 standing gold flame .. 1 heavy silver"""
    return GH.get('ui') if t >= 1 else GH.get('ssj') if t <= 0 else GH.mix(GH.SSJ, GH.UI, t)


def tmap(hr, bl):
    """screen map of how far along its blade every hair pixel is (0 root .. 1 tip), painter's order"""
    c = Cel(W, HH, ss=1); n = 22
    for b in bl:
        m = len(b.t); k = np.linspace(0, m - 1, n + 1); S = lambda a: np.stack([np.interp(k, np.arange(m), a[:, j]) for j in (0, 1)], -1); a_, b_ = S(b.A), S(b.B)
        for i in range(n):
            v = (i + .5) / n * .96 + .02; c.fill(A([a_[i], a_[i + 1], b_[i + 1], b_[i]]), (v, v, v)); c.fill(A([a_[i], a_[min(n, i + 1)], b_[min(n, i + 1)], b_[i]]) , (v, v, v), .6)
    a = c.rgba(); return blur(a[..., 0], 3.) / np.maximum(blur(a[..., 3], 3.), 1e-3)          # runs a little past the edge, so the pen line belongs to its clump


def wash(gold, silver, T, p):
    """p 0..1: silver runs from the roots to the tips with a white-hot lip"""
    q = p * 1.25 - .12; n = wash.n if wash.n is not None else None
    if n is None: n = wash.n = fbm(HH, W, (90, 34, 12), (1, .5, .22), 77)
    d = q - T + .050 * n; m = smooth(-.035, .035, d)[..., None]; e = (np.exp(-(d / .05) ** 2))[..., None]
    o = gold * (1 - m) + silver * m; o[..., :3] = o[..., :3] + e * o[..., 3:4] * rgb('#eaf6ff') * .85 * (0 < p < 1); return o


wash.n = None
GRADF = [(0, '#ffffff'), (.30, '#f2f3fa'), (.52, '#a9aabd'), (.78, '#585a70'), (1, '#2c2d40')]
GRADB = [(0, '#ffffff'), (.35, '#b8bccc'), (.7, '#5c5f74'), (1, '#33354a')]


def hairlayers(G, st, pal, f, wind, seed, flat=False):
    L = light(pal); hr = Hair(G, st, L, f=f, wind=wind, seed=seed, painterly=True, detail=1.0 if not flat else .55, kids=2 if not flat else 1, flat=flat); cx, cy, s = G.cx, G.cy, G.s
    c = Cel(W, HH); hr.back(c); b = para(c.rgba(), (cx + 1.5 * s, cy - .5 * s), (cx - 1.8 * s, cy), GRADB); c = Cel(W, HH); hr.front(c); fr = para(c.rgba(), (cx + 1.3 * s, cy - .5 * s), (cx - 1.6 * s, cy + .1 * s), GRADF); return hr, b, fr


def layers(shape=1., silver=1., opened=1., expr='ui', f=0, wind=(-.10, .10), seed=3, cam=None, gx=-.014, byaw=-22, relax=0.):
    """shape 0..1 gold flame -> UI shape; silver 0..1 the wash; opened 0..1 lids; relax 0..1 brows at rest"""
    G0 = head(**(cam or {})); X = GF.getx(expr); X = dict(X, style='ui' if silver > .02 and expr == 'ui' else X['style'], gx=gx, gy=.004, hw=.158, my=-.655, mU=[-.010, .002, .004, .002, -.006], mL=[-.010, .002, .004, .002, -.006])
    if relax > 0: S = GF.getx('serene'); X['B'] = A(X['B'], float) * (1 - relax) + A(S['B'], float) * relax; X['bt'] = A(X['bt'], float) * (1 - relax) + A(S['bt'], float) * relax; X['furrow'] = X['furrow'] * (1 - relax); X['cheekline'] = X['cheekline'] * (1 - relax)
    if opened < 1: q = GF.lids(X, 1 - opened); X = dict(q, style=X['style'], glow=X['glow'])
    G = GHead.of(G0, X['jaw'], X['jx']); st = style(shape); hs = hg = None; s = G.s
    if silver > 0: hr, bs, fs = hairlayers(G, st, SILV, f, wind, seed)
    if silver < 1: hr_g, bg, fg = hairlayers(G, st, GOLDH, f, wind, seed, True); hr = hr_g if silver <= 0 else hr
    if 0 < silver < 1:
        Tb, Tf = tmap(hr, [b for b in hr.bl if b.dp < hr.split]), tmap(hr, [b for b in hr.bl if b.dp >= hr.split]); back, front = wash(bg, bs, Tb, silver), wash(fg, fs, Tf, silver)
    else: back, front = (bs, fs) if silver >= 1 else (bg, fg)
    L = light(SILV if silver > .3 else GOLDH); body = GX.ui_body(G, L, byaw, f, seed)
    if silver < .3: L = dict(L, brow='#5a3408')
    if expr == 'ui' and X['style'] == 'ui': face = GX.ui_face(G, X, L, hr, seed, opened)
    else: face = gold_face(G, X, L, hr, seed, opened)
    return dict(back=back, body=body, face=face, front=front, G=G, X=X, hr=hr, opened=opened)


def gold_face(G, X, L, hr, seed, opened):
    """the same painted face with his Super Saiyan eyes (or shut lids)"""
    orig = GX.ui_features
    def feat(F, op):
        F.c.clip(F.F)
        for q in (-1, 1): F.style = 'ssj'; F.eye(q)
        F.mouth(); F.c.unclip(); F.nose()
        for q in (-1, 1):
            F.brow(q); P = F.brow_poly(q); n = (len(P) - 1) // 2; r = np.random.default_rng(31 + q); up, dn = P[1:n + 1], P[n + 1:][::-1]; F.c.clip(F.H.Pc(P))
            for j in range(40): i = int(r.uniform(0, n * .9)); k = r.uniform(0, 1); b = up[i] * k + dn[min(i, len(dn) - 1)] * (1 - k); i2 = min(n - 1, i + int(n * r.uniform(.10, .22))); e = up[i2] * min(1, k + .25) + dn[min(i2, len(dn) - 1)] * (1 - min(1, k + .25)); F.c.line(F.H.Pc([b, (b + e) / 2 + A([0, .004]), e]), r.uniform(.9, 2.0), ['#e0a02c', '#8a5410', '#2a1404', '#ffd86a'][int(r.integers(4))] if q > 0 else ['#7a4a10', '#2a1404', '#a86c1c'][int(r.integers(3))], taper=(.2, .7), seed=j, alpha=.85)
            F.c.unclip()
        F.furrow(); F.lines()
    GX.ui_features = feat
    try: return GX.ui_face(G, X, L, hr, seed, opened)
    finally: GX.ui_features = orig


def rims(img, L, k=1., vio=0.):
    G = L['G']; cx, s = G.cx, G.s; al = np.maximum(np.maximum(L['back'][..., 3], L['front'][..., 3]), np.maximum(L['face'][..., 3], L['body'][..., 3])); x, y = grid(); sh_ = lambda a, dx, dy: cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (W, HH), borderMode=cv2.BORDER_REPLICATE)
    e = np.clip(al - sh_(al, -5, 3), 0, 1); e = cv2.GaussianBlur(e, (0, 0), .8) * al; img = img + (e * smooth(cx - .2 * s, cx + .9 * s, x))[..., None] * (rgb('#dfeaff') * .9 * k + rgb('#b46cff') * 1.2 * vio); e = np.clip(al - sh_(al, 4, 1), 0, 1); e = cv2.GaussianBlur(e, (0, 0), 1.2) * al
    if vio: e2 = np.clip(al - sh_(al, -14, 8), 0, 1) * al; img = img + (blur(e2, 4) * smooth(cx, cx + 1.1 * s, x))[..., None] * rgb('#a860ff') * .7 * vio
    return img + (e * smooth(cx - .1 * s, cx - 1.0 * s, x))[..., None] * rgb('#4468c8') * .50 * k, al


def eyes(img, L, glow=1.):
    G, X = L['G'], L['X']; s = G.s; x, y = grid(); gx = X['gx']
    for q in (-1, 1):
        p = G.Pc([(q * (X['xi'] + (X['xo'] - X['xi']) * .43) + gx, -.008)])[0]; k = glow * (1. if q < 0 else .75)
        img = screen(np.clip(img, 0, None), (radial(p[0], p[1], .50 * s, .34 * s, 2.6)[..., None] * rgb('#7fa4ff') * .55 + radial(p[0], p[1], .17 * s, .12 * s, 1.8)[..., None] * rgb('#e8f0ff') * .80) * k); st = np.exp(-np.abs(y - p[1]) / (.010 * s)) * np.exp(-np.abs(x - p[0]) / (.42 * s)); img = img + st[..., None] * rgb('#a9c6ff') * .35 * k
    return img


def strand(G, f, cut=None, push=0.):
    """the one loose hair at the temple: root in the hair, tip adrift over the dark.  -> (polyline px, root->tip)"""
    s = G.s; R0 = G.P3(A([(.74, .30, .52)]))[0]; t = np.linspace(0, 1, 44)[:, None]; a = math.sin(f * .105) + .4 * math.sin(f * .22 + 1); b = math.sin(f * .13 + 2.1)
    P0 = R0; P1 = R0 + A([.40, -.26 + .02 * a]) * s; P2 = R0 + A([.86, .06 + .050 * a + .02 * b - .16 * push]) * s; P3 = R0 + A([1.30 + .02 * b, -.06 + .105 * a + .03 * b - .46 * push]) * s
    return (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t * t * P2 + t ** 3 * P3


def paint_strand(c, P, w=17., seed=3):
    n = len(P); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]; t = np.linspace(0, 1, n); hw = w * (np.sin(np.minimum(t / .30, 1) * np.pi / 2) ** .8 * (1 - t ** 1.7) * .5 + .015); a, b = P + N * hw[:, None], P - N * hw[:, None]
    c.fill(np.vstack([a, b[::-1]]), '#c3cddf'); c.fill(np.vstack([P + N * (hw * .1)[:, None], b[::-1]]), '#4a5268'); i0, i1 = int(n * .18), int(n * .72); c.fill(np.vstack([a[i0:i1], (a + (P - a) * (.75 * np.sin(np.linspace(0, np.pi, n)))[:, None])[i0:i1][::-1]]), '#ffffff'); c.line(a, 2.2, '#12121a', taper=(.3, .2), seed=seed); c.line(b, 2.2, '#12121a', taper=(.3, .2), seed=seed + 1); c.line(P[int(n * .2):int(n * .85)] - N[int(n * .2):int(n * .85)] * (hw[int(n * .2):int(n * .85), None] * .35), 1.2, '#14141e', taper=(.5, .5), seed=seed + 2)
