"""Hands in plain white cloth gloves, drawn on the hand anatomy of db/hands.py.

    G = glove('claw', x, y, s, light='violet', wet=0., torn=0., ink=False, seed=3, t=0.)      s = px from wrist to middle knuckle
    G.layer -> premultiplied RGBA 1080x1920      G.pts[(finger, 'mcp'|'pip'|'dip'|'tip'|'pad')], G.pts['palm'|'back'|'wrist'] -> screen px
    G.drips -> [(x, y, r, falling)] of this frame (t seconds)      G.g -> the hands.Drawn geometry (masks, groups, strokes)
    glove(..., side='L', yaw=, pitch=, roll=, dist=, pose=, dlight=, sleeve='#hex'|None, lite=False)   any of these overrides the preset
Presets POSE: claw (at the lens) push (open palm) limp (hanging, dripping) shoulder (gripping edge) bun (cradling half a bun) bump (fist, side on)
relaxed.  Lights LIGHT: violet gold rain dawn day ink.  wet 0..1 = cloth clings, skin shows through, hard shines, water beads and drips.
torn 0..1 = holes with frayed threads.  ink=True = manga finish (blacks, hatching, worn white line).  Keep seed fixed while a pose holds."""
import numpy as np, cv2, math
from . import W, H as HH
from . import hands as HD
from .hands import FING, GID, _ring, _emit, _prim_of, surface_curve, _runs
from .paint import rgb, smooth, mix
from .ink import Cel, spl, resample
from .kit import rim, para, screentone, mask_of, shifted

E = HD.euler
POSE = dict(                 # pose, (yaw, pitch, roll), eye distance, pivot
    claw=(HD.pose((10, 46, 36, -14), (4, 44, 34, -2), (8, 50, 38, 9), (14, 56, 40, 22), thumb=(62, 34, 18, 14, 40), wrist=(-18, 0)), (186, 44, -6), 2.35, (0, .95, 0)),
    push=(HD.pose((-4, 6, 4, -9), (-6, 5, 4, -2), (-4, 8, 5, 6), (-2, 12, 7, 15), thumb=(62, 16, 34, 2, -6), wrist=(-34, 0)), (192, 14, 4), 3.6, (0, .70, 0)),
    limp=(HD.pose((24, 22, 12, -2), (30, 30, 16, 0), (36, 36, 18, 1), (42, 40, 20, 5), thumb=(30, 34, 16, 12, 16), wrist=(22, 4)), (52, 0, 176), 9., (0, 0, 0)),
    shoulder=(HD.pose((32, 34, 22, -5), (34, 36, 24, 0), (38, 38, 24, 4), (42, 40, 24, 9), thumb=(46, 36, 6, 12, 14), wrist=(-8, 0)), (-16, 50, 6), 6., (0, .75, 0)),
    bun=(HD.pose((40, 38, 24, -4), (42, 42, 24, 0), (46, 44, 26, 3), (50, 44, 26, 7), thumb=(50, 52, -14, 18, 18), wrist=(-14, 0), cup=.4), (150, -30, 64), 7., (0, .75, 0)),
    bump=('fist', (120, -20, -80), 5.5, (0, .8, 0)),
    relaxed=('relaxed', (12, 6, 0), 8., (0, .75, 0)))

def _p(white, shade, deep, line, skin, skin_s, shine, L, rimd, rimc, rima, sleeve, sleeve_s):
    return dict(white=white, shade=shade, deep=deep, line=line, skin=skin, skin_s=skin_s, shine=shine, L=L, rimd=rimd, rimc=rimc, rima=rima, sleeve=sleeve, sleeve_s=sleeve_s)

LIGHT = dict(
    day=_p('#f7f4ee', '#b4b3c9', '#7c7a9c', '#17131a', '#efc3a0', '#c08a74', '#ffffff', (-.5, .75, .55), (.8, .2, -.3), '#fff3da', .5, '#4a5568', '#2d3547'),
    violet=_p('#b9aede', '#5d4c96', '#2c2158', '#0d0918', '#a587ac', '#4c3a6a', '#f2e6ff', (-.35, .8, .35), (.85, .25, -.35), '#c784ff', .95, '#22254a', '#121430'),
    gold=_p('#fff0bc', '#d08a3e', '#7d4524', '#2c0f08', '#f7bb74', '#b0622e', '#fffbe6', (.75, .25, .6), (.9, .0, -.2), '#fff7c8', 1., '#5a4636', '#33261f'),
    rain=_p('#c3c6c8', '#7c8287', '#4b5157', '#121417', '#a79488', '#6f625c', '#eef2f4', (-.4, .85, .3), (.5, .6, -.5), '#dfe6ea', .7, '#2c3a4c', '#1a2431'),
    dawn=_p('#fff1e4', '#c9a3b4', '#8f6c93', '#23161f', '#f3bd9c', '#c28783', '#fffaf2', (.75, .45, .45), (-.8, .3, -.3), '#a9c9e8', .6, '#5a5f78', '#3c3f58'),
    ink=_p('#f6f4ee', '#f6f4ee', '#0b0a0c', '#0b0a0c', '#d8d5ce', '#0b0a0c', '#ffffff', (-.45, .7, .45), (.8, .2, -.3), '#ffffff', 0., '#0b0a0c', '#0b0a0c'))

BLD = HD.build(len=1.0, wid=1.10, boxy=.34, tip=.42, knuckle=.95, palm=1.03, wrist=1.06, meat=1.06)
HEM = (.40, .54)                                            # glove hem: distance below the wrist


def _leaf(P, w, side=1., peak=.4):
    """shadow blade that hugs a crease: thin tapered polygon on one side of the polyline"""
    P = resample(P, 3.); n = len(P)
    if n < 4: return None
    T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]] * side; t = np.linspace(0, 1, n)
    b = np.where(t < peak, t / peak, (1 - t) / (1 - peak)) ** .8
    return np.vstack([P, (P + N * (b * w)[:, None])[::-1]])


class Glove: pass


def _wrinkles(g, wet, seed):
    """cloth creases on the 3-D surface.  list of (polyline px, weight, blade width in hand units)"""
    sc = g.sc; out = []; r = np.random.default_rng(seed * 17 + 5); W_ = lambda i, q: q @ sc.R[i] + sc.O[i]
    for hi, h in enumerate(g.hands):
        p = h.pose; gid0 = hi * 16
        def ring(idx, y, ph, wf, blade, gids):
            n0 = len(g.creases); _emit(g, hi, W_(idx, _ring(sc, idx, y, ph)), 'cloth', wf, gids, .055)
            for c in g.creases[n0:]: out.append((c['P'], wf, blade))
            del g.creases[n0:]
        for f in FING + ('thumb',):
            th = f == 'thumb'; i0, i1, i2 = _prim_of(g, hi, f, 0), _prim_of(g, hi, f, 1), _prim_of(g, hi, f, 2); gids = [gid0 + GID[f], gid0 + 1]
            for idx, bend, yj in (((i0, p[f][4], 1.), (i2, p[f][3], .02)) if th else ((i0, p[f][1], 1.), (i1, p[f][2], 1.), (i0, p[f][0], .18))):
                if idx < 0: continue
                L = sc.L[idx]; k = .35 + min(1., abs(bend) / 70.)
                # palm side: folds squeezed by the bend, fanning to both flanks
                for j, (dy, a0, a1) in enumerate(() if (bend > 75 and not th) else ((-.05, -1.25, .9), (-.13, -.8, 1.5), (.04, -1.0, .7), (-.21, .1, 1.5))):
                    if j > 1 + 2 * k: break
                    ph = math.pi + np.linspace(a0, a1, 16) * (1 if (j + len(f)) % 2 else -1); bow = .035 * np.cos((ph - math.pi) * 1.4) * h.scale * (1 if dy > 0 else -1)
                    ring(idx, np.clip(L * (yj + dy * .9) + bow + r.normal(0, .006), -.02, L * 1.05), ph, .52 * (.7 + .5 * k), .030 * k, gids)
                # back: slack cloth over the straight joint, taut diagonal pulls over the bent one
                if bend < 45:
                    for dy, a0, a1 in ((-.045, -.75, .35), (.03, -.25, .8)):
                        ph = np.linspace(a0, a1, 12); ring(idx, np.clip(L * (yj + dy) + .016 * np.cos(ph * 2.4) * h.scale, 0, L * 1.05), ph, .40, .014, gids)
                else:
                    soft = .55 if bend > 75 else 1.
                    for sg in (-1, 1):
                        if bend > 75 and yj < .5: continue
                        ph = sg * np.linspace(.35, 1.45, 14); ring(idx, L * (yj - .02) - np.linspace(0, .26, 14) * L, ph, .55 * soft, .030 * soft, gids)
                        if bend > 75: continue
                        ph = sg * np.linspace(.7, 1.5, 10); ring(idx, L * (yj - .10) - np.linspace(0, .22, 10) * L, ph, .36, .018, gids)
            for idx in (i0, i1, i2):                                                        # side seam: short stitches
                if idx < 0: continue
                L = sc.L[idx]
                for y0 in np.arange(.12, .88, .145): ring(idx, np.linspace(y0, y0 + .075, 6) * L, np.full(6, (1.52 if h.side == 'L' else -1.52)), .30, 0., gids)
        S = lambda xy, side, wf, blade: out.extend((q, wf, blade) for q in surface_curve(g, hi, xy, side))
        for f in ('index', 'middle', 'ring'):                                                # three sewn points on the back
            b_, o = h.sk.fr[f + '_mc']; d = (o - b_)[:2]; S([o[:2] - d * .10, o[:2] - d * .34, o[:2] - d * .62], 1, .62, .020)
        S([(.34, .58), (.18, .50), (.04, .36), (-.02, .16)], -1, .70, .045); S([(.30, .66), (.12, .60), (-.08, .50), (-.24, .40)], -1, .58, .036)
        S([(-.36, .62), (-.18, .66), (.02, .74), (.16, .80)], -1, .52, .030); S([(.22, .40), (.10, .28), (.06, .10)], -1, .40, .028); S([(-.30, .34), (-.20, .20), (-.16, .04)], -1, .40, .026); S([(.02, .62), (-.06, .40), (-.04, .18)], -1, .32, .02)
        S([(.34, .62), (.22, .52), (.14, .34), (.12, .12)], 1, .55, .032); S([(-.36, .58), (-.28, .40), (-.18, .22), (-.14, .06)], 1, .48, .030); S([(.24, .70), (.12, .64), (.0, .62)], 1, .36, .02); S([(-.22, .72), (-.12, .58), (-.10, .44)], 1, .34, .02)
        for k in range(9):                                                                   # gathers at the wrist
            x = -.24 + k * .06 + r.normal(0, .010); y0 = r.uniform(-.16, -.04); y1 = y0 + r.uniform(.14, .30)
            for side in (-1, 1): S([(x, y0), (x * .9 + r.normal(0, .02), (y0 + y1) / 2), (x * .72, y1)], side, .38, .018)
        if wet:
            for k in range(int(10 * wet)):
                a = np.array([r.uniform(-.30, .30), r.uniform(.10, .80)]); d = np.array([r.normal(0, .10), r.uniform(.10, .22)])
                for side in (-1, 1): S([a, a + d * .5 + r.normal(0, .03, 2), a + d], side, .30, .016)
    return out


def _patches(g, n, seed, names=None, rmin=.06, rmax=.12):
    """irregular patches on the visible form: [(group, polygon px)], placed on finger backs, knuckles and the palm"""
    r = np.random.default_rng(seed * 101 + 9); out = []; gid = g.gb['gid'].copy(); h = g.hands[0]; m = gid % 16 == GID['palm']; j, i, P = HD._hit(g, m); W_ = h.sk.fr['wrist']
    low = (((P - h.o) @ h.A) - W_.o) @ W_.R; gid[j[low[:, 1] > HEM[0] - .22], i[low[:, 1] > HEM[0] - .22]] = 0        # never on the hem or the sleeve
    for _ in range(n * 6):
        if len(out) >= n: break
        j, i = np.nonzero(gid > 0); k = int(r.integers(len(j))); G = next((q for q in g.groups if q['id'] == gid[j[k], i[k]]), None)
        if G is None or G['name'] in ('cuff', 'cuffin', 'prop', 'steel') or (names and G['name'] not in names): continue
        if g.gb['N'][j[k], i[k], 2] < .35: continue
        c = g.b2s(np.array([[i[k], j[k]]], float))[0]; R = r.uniform(rmin, rmax) * g.cam.s * (g.cam.dist / max(.4, g.cam.dist - g.gb['z'][j[k], i[k]])); a = np.linspace(0, 2 * math.pi, 11, endpoint=False); ax = HD._axis2(g, G)
        q = np.c_[np.cos(a) * 1.7, np.sin(a)] * (R * (1 + .28 * r.normal(0, 1, 11)))[:, None]; co, si = math.cos(ax), math.sin(ax); out.append((G, c + np.c_[q[:, 0] * co - q[:, 1] * si, q[:, 0] * si + q[:, 1] * co]))
    return out


def glove(preset, x, y, s, light='day', wet=0., torn=0., ink=False, seed=3, t=0., side='R', yaw=None, pitch=None, roll=None, dist=None, pose=None, pivot=None, dlight=None, sleeve=None, lw=None, res=560, thr=.10, layer=True, rims=True, arm=2.4, sleeve_s=None):
    po, (a, b, c), dd, pv = POSE[preset]; po = pose or po; R = E(a if yaw is None else yaw, b if pitch is None else pitch, c if roll is None else roll)
    if side == 'L' and yaw is None: R = E(-a, b, -c)
    c = dict(LIGHT['ink' if ink else light]); c['L'] = dlight or c['L']
    if sleeve: c['sleeve'] = sleeve; c['sleeve_s'] = '#%02x%02x%02x' % tuple(int(v * 150) for v in rgb(sleeve)) if sleeve_s is None else sleeve_s
    h = HD.Hand(po, side, R, (0, 0, 0), BLD, pivot=pivot or pv, nails=None, marks=False, arm=arm); cam = HD.Cam(x, y, s, dist or dd)
    g = HD.render([h], cam, c['L'], res=res, clip=(-400, -400, W + 400, HH + 400), thr=.42 if ink else thr, marks=False, style=.28)
    G = Glove(); G.g, G.c, G.s, G.seed, G.wet, G.ink = g, c, s, seed, wet, ink; G.pts = g.pts[0]; G.lw = lw or float(np.clip((s / 260.) ** .55, .35, 2.2)); G.rims = rims
    G.wr = _wrinkles(g, wet, seed); G.deep = HD.depth_polys(g, c['L'], .16 if ink else -.28); G.holes = _patches(g, int(round(7 * torn)), seed) if torn else []; G.cling = _patches(g, int(round(12 * wet)), seed + 4, rmin=.06, rmax=.13) if wet else []
    G.shine = HD.big(HD.rim_polys(g, (c['L'][0] * .8, c['L'][1], .5), lo=.80, graze=.97, eps=.02), s, .0015) if wet else {}
    G.drips = _drips(G, t) if wet else []
    if layer: G.layer = _finish(G)
    return G


def split(G, names=('thumb',)):
    """(back, front) layers: front = the named fingers with their line, so a prop can sit between them"""
    m = mask_of([q for k in G.g.groups if k['name'] in names for q in k['polys']]); k = int(3 * G.lw) * 2 + 1
    m = cv2.GaussianBlur(cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), (0, 0), .7)[..., None]; return G.layer * (1 - m), G.layer * m


def _hemregion(g, lo, hi, grow=0.):
    """screen polygons of the visible forearm pixels between lo and hi below the wrist"""
    h = g.hands[0]; m = (g.gb['gid'] % 16 == GID['palm']); j, i, P = HD._hit(g, m); W_ = h.sk.fr['wrist']; w = (((P - h.o) @ h.A) - W_.o) @ W_.R
    ok = (w[:, 1] > lo) & (w[:, 1] < hi); M = np.zeros(m.shape, np.uint8); M[j[ok], i[ok]] = 1
    if M.sum() < 12: return []
    k = max(1, int(grow * g.u)); M = cv2.dilate(cv2.morphologyEx(M, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) if grow else M
    return [g.b2s(q) for q in HD._regions(M > 0)]


def _hem(cel, G, part):
    """sleeve below the hem and the rolled hem itself, only where the forearm is actually visible"""
    g, c, lw = G.g, G.c, G.lw; idx = next(i for i, (hh, pr) in enumerate(g.sc.meta) if hh == 0 and pr['where'] == 'body' and pr['L'] == g.hands[0].sk.arm)
    if part == 'sleeve':
        for q in _hemregion(g, HEM[1] - .03, 9., .01):
            for k in g.groups:
                if k['name'] != 'palm': continue
                HD._clipg(cel, k); cel.fill(q, c['sleeve']); sh = g.shadow.get(k['id'])
                if sh: cel.ctx.save(); cel._path(q); cel.ctx.clip(); cel.fill(sh, c['sleeve_s'], evenodd=True); cel.ctx.restore()
                cel.ctx.restore()
        return
    vis = _hemregion(g, HEM[0] - .02, HEM[1] + .02, .07)
    if not vis: return
    def near(y, k):
        ph = np.linspace(0, 2 * math.pi, 120, endpoint=False); q = _ring(g.sc, idx, np.full(120, y + .10), ph); q[:, [0, 2]] *= k; Pw = q @ g.sc.R[idx] + g.sc.O[idx]; m = Pw.mean(0); f = ((Pw - m) * (np.array([0, 0, g.cam.dist]) - Pw)).sum(1) > 0
        r = max(_runs(f, True), key=lambda v: v[1] - v[0], default=None)
        return None if r is None or r[1] - r[0] < 12 else g.cam.px(Pw)[np.arange(*r) % 120]
    A, M, B = near(HEM[0], 1.075), near(sum(HEM) / 2, 1.13), near(HEM[1], 1.10)
    if A is None or B is None or M is None: return
    band = np.vstack([A, M[-1:], B[::-1], M[:1]]); cel.ctx.save(); cel._path(vis); cel.ctx.clip(); cel.fill(band, c['white'])
    d = B.mean(0) - A.mean(0); d /= np.hypot(*d) + 1e-9; n_ = np.array([-d[1], d[0]]); L2 = np.array([c['L'][0], -c['L'][1]]); sg = -1 if n_ @ L2 > 0 else 1
    pr = band @ n_; w = pr.max() - pr.min(); cut = band.mean(0) + n_ * sg * w * .14
    cel.shade(np.array([cut - d * w, cut + d * w, cut + d * w + n_ * sg * w, cut - d * w + n_ * sg * w]), c['shade'] if not G.ink else '#0b0a0c', band)
    for q, ww in ((A, 3.4), (B, 3.4), (np.vstack([A[:1], M[:1], B[:1]]), 3.4), (np.vstack([A[-1:], M[-1:], B[-1:]]), 3.4)): cel.line(q, ww * lw, c['line'], (.04, .04), seed=G.seed)
    q = resample(M, max(4., .030 * G.s))
    for k in range(1, len(q) - 2, 2): cel.line(q[k:k + 2], 1.4 * lw, c['line'] if not G.ink else '#777777', (.1, .1), seed=k)
    n = len(A)
    for k in range(int(n * .08), int(n * .95), max(4, n // 14)):
        f = k / n; e = B[min(len(B) - 1, int(f * len(B)))]; cel.line(np.vstack([A[k], A[k] * .55 + e * .45]), 1.7 * lw, c['line'], (.1, .7), seed=k)
    cel.ctx.restore()


def _drips(G, t):
    """water hanging from the lowest fingertips and falling: deterministic in t"""
    out = []; tips = [(f, G.pts[f, 'tip']) for f in FING + ('thumb',)]
    for j, (f, p) in enumerate(tips):
        q = G.pts[f, 'dip' if f != 'thumb' else 'ip']
        if p[1] - q[1] < .45 * np.hypot(*(p - q)): continue                # water only hangs from a tip that points down
        per = 1.15 + .37 * ((j * 7919) % 5) / 5; ph = ((t + j * .43) % per) / per; r0 = .030 * G.s
        if ph < .72: g_ = ph / .72; out.append((p[0], p[1] + .03 * G.s + r0 * g_ * .8, r0 * (.35 + .65 * g_), 0))
        else: dt = (ph - .72) * per; out.append((p[0], p[1] + .05 * G.s + r0 + .5 * 2600 * dt * dt * (G.s / 420.), r0 * .92, 1))
    return out


def draw(cel, G):
    g, c, lw, s = G.g, G.c, G.lw, G.s; pal = dict(HD.PAL['day'], skin=c['white'], shade=c['shade'], line=c['line'], light=c['shine'], band=c['sleeve'])
    HD.fill_flat(cel, g, pal)
    if not G.ink:
        HD.fill_shadow(cel, g, pal)
        L2 = np.array([c['L'][0], -c['L'][1]])
        cel.ctx.save(); cel._path([q for k in g.groups for q in k['polys']]); cel.ctx.clip()
        for P, wf, blade in G.wr:                                       # every fold has a lit side and a shadow blade
            if blade <= 0: continue
            T = P[-1] - P[0]; n_ = np.array([-T[1], T[0]]); q = _leaf(P, blade * s * 1.15, -1. if n_ @ L2 > 0 else 1.)
            if q is not None: cel.fill(q, c['shade'], .9)
        cel.ctx.restore(); HD.fill_polys(cel, g, G.deep, c['deep'], .55)
        for k, q in G.cling:
            HD._clipg(cel, k); m = mix(rgb(c['white']), rgb(c['skin']), .55); cel.fill(spl(q, True, n=5), m, .75 * G.wet); sh = g.shadow.get(k['id'])
            if sh: cel.ctx.save(); cel._path(spl(q, True, n=5)); cel.ctx.clip(); cel.fill(sh, c['skin_s'], .6, evenodd=True); cel.ctx.restore()
            cel.ctx.restore()
        if G.shine: HD.fill_polys(cel, g, G.shine, c['shine'], .9)
    else:
        HD.hatch(cel, g, None, .034, 1.7 * lw, c['line'], G.seed, 1.1, length=.15, fade=.3); HD.hatch(cel, g, G.deep, .022, 2.4 * lw, c['line'], G.seed + 1, .75, length=.30)
        HD.fill_polys(cel, g, HD.big(G.deep, s, .008), c['line'], 1.)
        HD.hatch(cel, g, HD.big(G.deep, s, .008), .050, 1.3 * lw, '#f6f4ee', G.seed + 2, 1.0, length=.12, alpha=.9)
        for P, wf, blade in G.wr:
            q = _leaf(P, blade * s * .8, 1.)
            if q is not None and blade > .022: cel.fill(q, c['line'])
    _hem(cel, G, 'sleeve')
    for j, (k, q0) in enumerate(G.holes):                                  # torn: bare skin, curled edge, threads
        r = np.random.default_rng(G.seed * 7 + j); m = q0.mean(0); d = q0 - m; ln = np.hypot(*d.T).max(); ax = d[np.argmax(np.hypot(*d.T))] / ln; nx = np.array([-ax[1], ax[0]]); u = np.linspace(-1, 1, 13)
        up = m + ax * (u * ln * 1.25)[:, None] + nx * (ln * .42 * (1 - u ** 2) ** .8 * (1 + .25 * r.normal(0, 1, 13)))[:, None]; dn = m + ax * (u * ln * 1.25)[:, None] - nx * (ln * .30 * (1 - u ** 2) ** .8 * (1 + .25 * r.normal(0, 1, 13)))[:, None]
        q = np.vstack([spl(up, n=5), spl(dn[::-1], n=5)[1:]]); HD._clipg(cel, k); cel.fill(q, c['skin']); sh = g.shadow.get(k['id']); L2 = np.array([c['L'][0], -c['L'][1]])
        cel.ctx.save(); cel._path(q); cel.ctx.clip()
        if sh: cel.fill(sh, c['skin_s'], 1., evenodd=True)
        e_ = up if nx @ L2 > 0 else dn; cel.fill(np.vstack([spl(e_, n=5), (spl(e_, n=5) - L2 * ln * .16)[::-1]]), c['skin_s']); cel.ctx.restore()      # the cloth edge throws a shadow on the skin
        o_ = dn if nx @ L2 > 0 else up; lip = spl(o_, n=5); cel.fill(np.vstack([lip, (lip + (lip - m) * .16)[::-1]]), c['white']); cel.line(lip + (lip - m) * .16, 1.8 * lw, c['line'], (.3, .3), seed=j); cel.line(lip, 2.0 * lw, c['line'], (.2, .2), seed=j); cel.line(spl(e_, n=5), 3.4 * lw, c['line'], (.25, .25), press=.5, seed=j + 1)
        for _ in range(6):
            a = up[int(r.integers(2, 11))]; e = dn[int(r.integers(2, 11))] if r.random() < .12 else a + (m - a) * r.uniform(.4, 1.) + r.normal(0, .015 * s, 2); cel.line(spl([a, (a + e) / 2 + r.normal(0, .012 * s, 2), e], n=6), 1.5 * lw, c['white'], (.05, .5), seed=int(r.integers(1 << 20)))
        cel.ctx.restore()
    for j, (P, wf, blade) in enumerate(G.wr): cel.line(P, 2.9 * wf * lw * 1.15, c['line'], (.28, .5), press=.35, seed=G.seed * 3 + j, tremor=.2)
    if G.ink: HD.brush_lines(cel, g, 6.2 * lw, c['line'], G.seed)
    else: HD.ink_lines(cel, g, pal, 5. * lw, G.seed)
    _hem(cel, G, 'hem')
    for x, y, r, fall in G.drips:
        a = np.linspace(0, 2 * math.pi, 26); st = 1.9 if fall else 1.15; q = np.c_[x + np.sin(a) * r * (1 - .35 * (np.cos(a) > 0) * np.cos(a)), y - np.cos(a) * r * st]; cel.fill(q, c['shade']); cel.fill(np.c_[x + np.sin(a) * r * .75, y - np.cos(a) * r * st * .72 + r * .15], mix(rgb(c['shade']), rgb(c['shine']), .6)); cel.line(np.vstack([q, q[:1]]), 1.5 * lw, c['line'], (.1, .1), seed=2); cel.fill(np.c_[x - r * .32 + np.cos(a) * r * .2, y + r * .05 + np.sin(a) * r * .30], c['shine'])
    return cel


def _finish(G):
    cel = draw(Cel(W, HH), G); lay = cel.rgba(); c = G.c
    if G.ink: return lay
    d = np.array(c['rimd']); k = 3.6 * G.lw
    if c['rima'] and G.rims: lay = rim(lay, -d[0] * k, d[1] * k, c['rimc'], c['rima'], .7)
    L = c['L']; m = np.array([G.pts['palm'][0], G.pts['palm'][1]]); v = np.array([L[0], -L[1]]) * G.s * 1.3
    return para(lay, m + v, m - v, [(0, '#ffffff'), (.5, '#f4eff4'), (1, '#b7b0c8')], 'mul', .75)


def tone(G, pitch=7., dark=.30):
    """manga screentone layer (black dots) on the glove's cel shadow; composite over the ink layer"""
    c = Cel(W, HH)
    for k in G.g.groups:
        sh = G.g.shadow.get(k['id'])
        if k['polys'] and sh: HD._clipg(c, k); c.fill(sh, '#ffffff', evenodd=True); c.ctx.restore()
    a = screentone(c.rgba()[..., 3], pitch, dark=dark); return np.dstack([np.zeros((HH, W, 3), np.float32), a.astype(np.float32)])


def touch(A, B, overlap=3.):
    """two fists that meet: A points right, B points left (layers, or anything with alpha).  Returns (dxA, dxB) that make the silhouettes press together,
    each moved half way, with `overlap` px of contact on the row where they meet first."""
    a, b = A[..., 3] > .5, B[..., 3] > .5; row = np.nonzero(a.any(1) & b.any(1))[0]; xa = np.array([np.nonzero(a[y])[0].max() for y in row]); xb = np.array([np.nonzero(b[y])[0].min() for y in row]); gap = float((xb - xa).min()); d = gap - 1 + overlap
    return d / 2, -d / 2


def mirror(layer, cx):
    """flip a layer about the vertical line x = cx"""
    return cv2.warpAffine(layer, np.float32([[-1, 0, 2 * cx], [0, 1, 0]]), (W, HH), flags=cv2.INTER_LINEAR)


def move(layer, dx=0., dy=0., rot=0., k=1., piv=(0, 0)):
    c, s = math.cos(rot) * k, math.sin(rot) * k; x, y = piv
    return cv2.warpAffine(layer, np.float32([[c, -s, x - c * x + s * y + dx], [s, c, y - s * x - c * y + dy]]), (W, HH), flags=cv2.INTER_LINEAR)


def arm_planes(G, shade, line, lw=None, deep=None):
    """muscle planes on a bare forearm (layer to composite over G.layer): hard shadow polygons and two tapered lines on the 3-D surface"""
    g = G.g; lw = lw or getattr(G, 'lw', 1.); idx = next(i for i, (hh, pr) in enumerate(g.sc.meta) if hh == 0 and pr['where'] == 'body' and pr['L'] == g.hands[0].sk.arm); cel = Cel(W, HH)
    def curve(pts, n=10): q = spl(np.asarray(pts, float), n=n); P = _ring(g.sc, idx, q[:, 0], q[:, 1]); Pw = P @ g.sc.R[idx] + g.sc.O[idx]; c = Pw.mean(0); return g.cam.px(Pw), Pw
    reg = _hemregion(g, HEM[1] + .06, 99., .0)
    if not reg: return cel.rgba()
    cel.clip(reg)
    L = G.g.L; a0 = math.atan2((g.sc.R[idx] @ np.asarray(L))[0], (g.sc.R[idx] @ np.asarray(L))[2])
    q, _ = curve([(1.0, a0 + .55), (1.7, a0 + .25), (2.6, a0 + .40), (2.3, a0 + .62), (1.6, a0 + .55)]); cel.fill(q, shade, .8)
    for pts in ([(1.0, .2), (1.8, .45), (2.7, .30), (3.5, .42)], [(2.2, -.55), (3.0, -.35), (3.9, -.5)], [(3.3, .95), (4.0, .75), (4.8, .85)]):
        q, _ = curve([(y, a0 + p) for y, p in pts]); cel.line(q, 2.3 * lw, line, (.25, .65), press=.4, seed=int(pts[0][0] * 10))
    cel.unclip(); return cel.rgba()
