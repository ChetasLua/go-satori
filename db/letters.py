"""Letters of GO: EGO block letters (flat), the gi discs 悟 / 亀, the brush 吾 -> 悟, the seal, the hand-lettered credit.

USAGE (everything in screen px, layers premultiplied float32 RGBA, masks float32 0..1)
  EGO flat     m = ego_mask((x, y, h))                         worn mask of the word, cap height h, top-left (x, y)  [cached]
               img = ego_flat(img, (x, y, h), col='#f4f1ea')   composite it on an RGB picture (d4: behind the figure)
               P = ego_paths((x, y, h))                        {'E': [outer, hole...], 'G': ..., 'O': ...} clean polylines
               ego_width(h)                                    width of the word at cap height h
  EGO solid    from db.letters_3d import Monument; M = Monument(); M.setup(); cam = M.camera(); img, g = M.render(cam, f)      (d6: 30 m letters, moon, crack at 34, fall at 60)
               M.put(img, g, layer, world_xyz, cam) sets a figure layer behind whatever is nearer; M.ground_at(cam, u, v) = point of the waste under a pixel; screen right = heading -122 deg
  Gi discs     disc(cel, '悟', disc_at(cx, cy, r, rot=.1, sx=.8))   white disc, ink ring, ink character on an ink.Cel
               disc(cel, '亀', T)        T = any callable (n,2) unit coords [-1..1] (y down) -> (n,2) px: your own body surface
               disc_polys('悟')          list of unit polygons (strokes), to project or clip yourself
               disc_tex('悟', 256)       premultiplied RGBA square texture (for warps)
  Brush        from db.letters_brush import Satori; S = Satori(); S.setup(); img = S.frame(f)      (h5, local frame 0..59: 吾, one beat, heart, seal, credit)
  Manga page   from db import letters_manga as MG; img = MG.page_bg(f, cx, cy); ...figure...; MG.spark; MG.panel; MG.print_      (d4)
"""
import numpy as np, cv2, math
from . import W, H
from .paint import rgb, f32, fbm, smooth, grid
from .ink import Cel, spl, resample, ribbon, noise1

# ===================================================================== EGO: our own block letters
# 1000 em, y down.  E is a pair of jaws (beaked arms); G and O are chamfered slabs round one perfect circle.
CX_, R_ = 430., 214.                                                       # circle counter: centre x, radius (em)


def _circ(cx, cy, r, a0, a1, n=64):
    t = np.linspace(a0, a1, n); return np.c_[cx + np.cos(t) * r, cy + np.sin(t) * r]


def _build():
    E = [(0, 118), (118, 0), (716, 0), (694, 308), (596, 224), (276, 224), (276, 392), (566, 376), (544, 624), (276, 608), (276, 776), (608, 776), (706, 682), (730, 1000), (118, 1000), (0, 882)]
    g = 2 * math.pi
    # G: outer slab, mouth cut on the right, bar with a spur
    up = CX_ + math.sqrt(R_ ** 2 - (322 - 500) ** 2); yb = 500 + math.sqrt(R_ ** 2 - 32 ** 2)
    a0 = math.atan2(322 - 500, up - CX_); a1 = math.atan2(yb - 500, 32)
    arc = _circ(CX_, 500, R_, a0, a1 - g, 90)                                               # from upper lip, over the top, down the left, to the lower right
    G = [(0, 236), (236, 0), (672, 0), (838, 166), (838, 404), (716, 322)] + [tuple(p) for p in arc] + [(462, 492), (852, 492), (852, 800), (652, 1000), (236, 1000), (0, 764)]
    O = [(0, 236), (236, 0), (624, 0), (860, 236), (860, 764), (624, 1000), (236, 1000), (0, 764)]
    return {'E': dict(w=730., p=[np.array(E, float)]), 'G': dict(w=852., p=[np.array(G, float)]),
            'O': dict(w=860., p=[np.array(O, float), _circ(CX_, 500, R_, 0, -g, 97)[:-1]])}


EGO = _build()
GAP = 64.


def ego_width(h, letters='EGO', gap=GAP):
    return (sum(EGO[c]['w'] for c in letters) + gap * (len(letters) - 1)) * h / 1000.


def ego_paths(box, letters='EGO', gap=GAP, M=None):
    """{letter: [outer, holes...]} in px.  box = (x, y, cap height); M = optional 3x3 homography applied after"""
    x, y, h = box; k = h / 1000.; out = {}
    for c in letters:
        ps = [q * k + [x, y] for q in EGO[c]['p']]
        if M is not None:
            ps = [(lambda t: t[:, :2] / t[:, 2:3])(np.c_[resample(np.vstack([q, q[:1]]), 4.), np.ones(len(resample(np.vstack([q, q[:1]]), 4.)))] @ np.asarray(M).T) for q in ps]
        out[c] = ps; x += (EGO[c]['w'] + gap) * k
    return out


_tc = {}


def _wear(h, w, u, d, seed, amt):
    """erosion field (> 0 = paint gone).  u = px per 1/1000 em, d = distance inside the letter (px).  Old paint on stone:
    it lives in quiet areas and dies in colonies: along corners, in vertical weathering strips, round old blows."""
    r = np.random.default_rng(seed)
    S = lambda *s: tuple(max(1.3, q * u) for q in s)
    big = fbm(h, w, S(420, 190), (1, .55), seed); mid = fbm(h, w, S(64, 27, 11), (1, .75, .5), seed + 1); fine = fbm(h, w, S(8.5, 4.6, 2.4), (1, .9, .8), seed + 2)
    run = fbm(h, w, S(150, 44, 15), (1, .75, .5), seed + 3, aspect=.14); lay_ = fbm(h, w, S(150, 44, 15), (1, .75, .5), seed + 4, aspect=7.)
    edge = np.exp(-d / (34 * u)); deep = np.exp(-d / (9 * u))
    where = smooth(.15, 1.55, big + .55 * run + .85 * edge + .25 * lay_ + 1.1 * (amt - .6))                # the colonies
    f = where * 2.35 + .50 * mid + .46 * fine + .30 * run + .45 * deep - 2.05
    pit = fbm(h, w, S(17, 9), (1, .7), seed + 6); f = np.maximum(f, (pit - 2.75) * .9)                         # lone pits anywhere
    sc = np.zeros((h, w), f32)                                                                              # scratches: long, straight, a few families
    fam = [r.normal(-.20, .05), r.normal(1.05, .06), r.normal(-.75, .08)]
    for _ in range(int(64 * amt)):
        x, y = r.uniform(0, w), r.uniform(0, h); a = r.choice(fam) + r.normal(0, .05); L = r.gamma(1.7, 70) * u; m = 8; t = np.linspace(0, 1, m); c = r.normal(0, .04)
        P = np.c_[x + np.cos(a + c * t) * L * t, y + np.sin(a + c * t) * L * t] + r.normal(0, .4 * u, (m, 2)); hw = r.choice([.5, .8, 1.3, 2.4]) * max(1., u * 1.5) * np.sin(np.linspace(.12, math.pi - .05, m)) ** .7; N = np.array([-math.sin(a), math.cos(a)])
        cv2.fillPoly(sc, [np.round(np.vstack([P + N * hw[:, None], (P - N * hw[:, None])[::-1]]) * 16).astype(np.int32)], float(r.uniform(.6, 1.)), cv2.LINE_AA, 4)
    return np.maximum(f, np.where(sc > .05, (sc - .35) * 1.1 * smooth(-1.1, .5, fine), -9.))


def ego_mask(box, letters='EGO', wear=.62, seed=3, h=H, w=W, M=None, gap=GAP, chips=1.0, ss=2):
    """worn, chipped mask of the word (0..1).  wear 0 = clean, 1 = almost gone.  M = homography for a tilted word."""
    key = (tuple(np.round(box, 2)), letters, wear, seed, h, w, None if M is None else tuple(np.round(np.ravel(M), 5)), gap, chips)
    if key in _tc: return _tc[key].copy()
    P = ego_paths(box, letters, gap, M); hh, ww = h * ss, w * ss; u = box[2] / 1000. * ss
    m = np.zeros((hh, ww), np.uint8); r = np.random.default_rng(seed + 40)
    for c in letters:
        q = [np.round(p * ss * 16).astype(np.int32) for p in P[c]]; cv2.fillPoly(m, q, 255, cv2.LINE_AA, 4)
        if chips > 0:                                                           # broken corners and spalls: faceted bites, most of them at corners
            for oo in P[c]:
                o = resample(np.vstack([oo, oo[:1]]) * ss, 3.); T = np.gradient(o, axis=0); an = np.unwrap(np.arctan2(T[:, 1], T[:, 0])); kn = np.convolve(np.abs(np.gradient(an)), np.ones(9), 'same'); p = .14 + 7 * np.clip(kn, 0, 1) * (len(oo) < 40); p /= p.sum()
                for k in r.choice(len(o), int(26 * chips * (1 if len(oo) < 80 else .5)), p=p):
                    R = r.gamma(1.1, 6.0) * u * (2.6 if r.random() < .12 else 1) + 1.5 * u; n = int(r.integers(5, 10)); a = np.sort(r.uniform(0, 2 * math.pi, n)); rad = R * r.uniform(.6, 1.2, n); t = T[k] / (np.hypot(*T[k]) + 1e-6); st = r.uniform(1.0, 2.4)
                    pl = np.c_[np.cos(a) * rad * st, np.sin(a) * rad]; pl = np.c_[pl[:, 0] * t[0] - pl[:, 1] * t[1], pl[:, 0] * t[1] + pl[:, 1] * t[0]] + o[k]
                    cv2.fillPoly(m, [np.round(pl * 16).astype(np.int32)], 0, cv2.LINE_AA, 4)
    m = m.astype(f32) / 255.
    if wear > 0:
        d = cv2.distanceTransform((m > .5).astype(np.uint8), cv2.DIST_L2, 5)
        f = _wear(hh, ww, u, d, seed, wear); m = m * (1 - smooth(-.13, .13, f))
    out = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA); _tc[key] = out
    return out.copy()


def ego_flat(img, box, col='#f4f1ea', op=1.0, **k):
    """composite the worn word on an RGB picture"""
    a = ego_mask(box, h=img.shape[0], w=img.shape[1], **k)[..., None] * op; c = rgb(col) if isinstance(col, str) else np.asarray(col, f32)
    return img * (1 - a) + c * a


# ===================================================================== gi discs
# 1000 em inside a disc of radius 500 round (500, 500).  stroke = (keys (x, y, width), corner key indices, cut at start, cut at end)
DISC = {
    '悟': [([(226, 344, 34), (206, 440, 58), (176, 548, 70)], (), 0, -.35), ([(340, 326, 38), (366, 392, 62), (384, 452, 66)], (), 0, .28), ([(284, 160, 84), (282, 230, 74), (282, 780, 74), (282, 880, 78)], (), .5, -.1),
          ([(486, 214, 58), (540, 211, 54), (790, 199, 54), (828, 197, 64)], (), .45, -.32), ([(656, 206, 58), (634, 300, 58), (594, 470, 60)], (), 0, 0),
          ([(496, 340, 56), (550, 336, 52), (786, 324, 56), (752, 462, 64)], (2,), .45, 0),
          ([(452, 478, 62), (510, 474, 58), (820, 455, 60), (868, 452, 74)], (), .5, -.36),
          ([(514, 580, 66), (518, 640, 62), (526, 866, 62)], (), .3, 0), ([(496, 582, 54), (796, 568, 60), (774, 866, 66)], (1,), 0, 0), ([(524, 822, 54), (776, 812, 54)], (), 0, 0)],
    '亀': [([(458, 100, 70), (408, 190, 60), (296, 302, 26)], (), .25, 0), ([(420, 186, 52), (642, 172, 60), (610, 236, 62), (548, 302, 36)], (1,), 0, 0),
          ([(262, 326, 62), (264, 400, 62), (270, 694, 62)], (), .3, 0), ([(248, 330, 52), (738, 316, 60), (726, 694, 68)], (1,), 0, 0), ([(268, 500, 48), (728, 490, 48)], (), 0, 0), ([(270, 666, 50), (726, 658, 50)], (), 0, 0),
          ([(496, 318, 66), (496, 640, 64), (498, 766, 66), (518, 826, 70), (586, 850, 66), (680, 852, 62), (756, 846, 64), (800, 820, 64), (822, 772, 44), (832, 702, 14)], (), 0, 0)],
}


def _stroke_poly(K, corners=(), c0=0., c1=0.):
    """variable-width stroke outline with mitred corners and sheared ends"""
    K = np.asarray(K, float); cs = [0] + list(corners) + [len(K) - 1]; segs = []
    for a, b in zip(cs[:-1], cs[1:]):
        Q = K[a:b + 1]; m = len(Q); C = spl(Q[:, :2], n=8) if m > 2 else np.linspace(Q[0, :2], Q[-1, :2], 9); C = np.c_[C, np.interp(np.linspace(0, m - 1, len(C)), np.arange(m), Q[:, 2])]
        T = np.gradient(C[:, :2], axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]; segs.append((C[:, :2] + N * C[:, 2:] * .5, C[:, :2] - N * C[:, 2:] * .5, T, C[:, 2]))
    def meet(p, d, q, e):
        M = np.array([d, -e]).T
        if abs(np.linalg.det(M)) < 1e-6: return (p + q) / 2
        return p + d * np.linalg.solve(M, q - p)[0]
    A, B = [segs[0][0]], [segs[0][1]]
    for k in range(1, len(segs)):
        for L_, i in ((A, 0), (B, 1)):
            p, q = L_[-1], segs[k][i]; x = meet(p[-1], segs[k - 1][2][-1], q[0], segs[k][2][0]); L_[-1] = np.vstack([p[:-1], x]); L_.append(q[1:])
    A, B = np.vstack(A), np.vstack(B); t0, w0, t1, w1 = segs[0][2][0], segs[0][3][0], segs[-1][2][-1], segs[-1][3][-1]
    A = A.copy(); B = B.copy(); B[0] -= t0 * max(c0, 0) * w0; A[0] -= t0 * max(-c0, 0) * w0; A[-1] += t1 * max(c1, 0) * w1; B[-1] += t1 * max(-c1, 0) * w1
    out = []; i = 0                                                          # one polygon per straight run, so corners never cancel themselves
    for k, sg in enumerate(segs):
        n = len(sg[0]) - (1 if k else 0); a0 = max(0, i - 1); out.append(np.vstack([A[a0:i + n], B[a0:i + n][::-1]])); i += n
    return out


def disc_polys(ch):
    """stroke polygons of the character in unit disc coordinates (disc = radius 1, y down)"""
    return [(p - 500.) / 500. for s in DISC[ch] for p in _stroke_poly(*s)]


def disc_at(cx, cy, r, rot=0., sx=1., sy=1., skew=0.):
    """unit -> px map: scale (foreshorten with sx / sy), skew, rotate, move"""
    c, s = math.cos(rot), math.sin(rot)
    def T(P):
        P = np.asarray(P, float); x = (P[:, 0] + skew * P[:, 1]) * sx * r; y = P[:, 1] * sy * r
        return np.c_[x * c - y * s + cx, x * s + y * c + cy]
    return T


def _dense(P, step=.06):
    Q = np.vstack([P, P[:1]]); n = np.maximum(1, np.ceil(np.hypot(*np.diff(Q, axis=0).T) / step).astype(int)); out = []
    for i, k in enumerate(n): out.append(Q[i] + (Q[i + 1] - Q[i]) * (np.arange(k) / k)[:, None])
    return np.vstack(out)


def disc(cel, ch, T, white='#f7f3ea', ink='#17131a', w=2.2, ring=True, alpha=1.0, seed=0):
    """white disc + ink ring + character on an ink.Cel.  w = ring line width px (0 = none).  Returns the disc outline in px."""
    a = np.linspace(0, 2 * math.pi, 97)[:-1]; o = T(np.c_[np.cos(a), np.sin(a)])
    if white is not None: cel.fill(o, white, alpha)
    for P in disc_polys(ch): cel.fill(T(_dense(P)), ink, alpha)
    if ring and w: cel.line(spl(o, True, n=4), w, ink, taper=(0, 0), press=.18, seed=seed, tremor=.15, alpha=alpha, closed=True)
    return o


def disc_tex(ch, n=256, white='#f7f3ea', ink='#17131a', ring=.035):
    """square premultiplied RGBA texture of the disc (for cv2 warps)"""
    c = Cel(n, n); T = disc_at(n / 2, n / 2, n * .47); disc(c, ch, T, white, ink, w=ring * n)
    return c.rgba()
