"""ki hero stills (prototypes of my shots g8, d5 wide, d3, f3): python -m db.ki_hero beam|flash|threads|pillar [frame]"""
import sys, time
from .ki_core import *
from .ki_aura import aura, lit, Hug, pillar3, with_aura
from .ki_beam import *
from .ki_orb import *
from .ki_hit import *
from .ki_bolt import storm, arcs, bolt, bolt_cel
from .ki_scene import *
from . import comp, kit
from .paint import ramp
cv2.setNumThreads(3)
OUT = 'qa/ki/'


def sil(cx, foot, h, who='goku', pose='push', flip=1, col='#0d0b1e'):
    """stand-in back-view silhouettes in a firing stance (the body lane's figures replace them)"""
    s = h / 100.; X = lambda P: np.array([[cx + flip * x * s, foot - y * s] for x, y in P]); c = Cel()
    S = lambda P, k=(): c.fill(X(spl(P, True, corners=k, n=6)), col)
    S([(-30, 0), (-20, 0), (-17, 8), (-9, 26), (-1, 42), (4, 47), (-4, 52), (-12, 44), (-20, 28), (-27, 10)], (0, 1)); S([(10, 0), (24, 0), (25, 4), (21, 12), (18, 26), (12, 46), (0, 50), (2, 36), (8, 22), (9, 8)], (0, 1))
    S([(-8, 44), (9, 44), (12, 56), (15, 70), (12, 78), (4, 82), (-6, 80), (-12, 70), (-10, 56)]); S([(6, 78), (14, 78), (26, 72), (38, 69), (40, 64), (36, 62), (25, 64), (13, 66)]); S([(0, 74), (10, 68), (22, 62), (36, 60), (37, 65), (24, 69), (12, 74)])
    S([(1, 80), (9, 80), (13, 84), (13.5, 91), (10, 96), (3, 97), (-1, 92), (-1, 85)])
    H_ = [((-1, 96), (-14, 98), (-1, 86)), ((-1, 98), (-13, 108), (2, 92)), ((0, 98), (-4, 116), (6, 95)), ((3, 98), (6, 113), (10, 94)), ((6, 97), (16, 108), (12, 90)), ((-1, 90), (-9, 88), (0, 82))] if who == 'goku' else [((-1, 94), (-9, 108), (4, 90)), ((-1, 97), (-5, 117), (6, 92)), ((1, 98), (1, 122), (9, 92)), ((4, 98), (8, 118), (12, 90)), ((7, 96), (13, 110), (13, 88)), ((-1, 90), (-8, 100), (2, 86))]
    for a, t, b in H_: c.fill(X(kit.spike(a, t, b, .12, .04)), col)
    return c.layer()


def reflect_soft(img, hz, t, seed=1, amp=30.):
    """colour masses of everything above hz, mirrored and smeared: meant to be shown through hard wave shapes"""
    q = cv2.resize(np.clip(img, 0, 1.6), (W // 4, H // 4), interpolation=cv2.INTER_AREA); q = cv2.GaussianBlur(q, (0, 0), sigmaX=3.5, sigmaY=9.); q = cv2.resize(q, (W, H), interpolation=cv2.INTER_LINEAR); x, y = grid(); d = np.maximum(y - hz, 0); sx = x + nimg(H, W, 46, t, seed + 3, 2, 2, 3., .16) * amp * np.clip(d / 140., .1, 1.6); return cv2.remap(q, sx.astype(f32), np.clip(hz - d * .95, 0, hz - 1).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def glints(hz, t, seed=1, dens=0., k=1.):
    """hard horizontal wave shapes whose size grows with depth -> field, threshold at 0 (positive = light facet)"""
    x, y = grid(); d = np.maximum(y - hz, 1.); o = np.zeros((H, W), f32); wt = np.zeros((H, W), f32)
    for i, (sc, at) in enumerate(((7., 22.), (14., 70.), (28., 170.), (52., 330.))):
        g = np.exp(-(np.log(d / at)) ** 2 * 1.1); n = nimg(H, W, sc * k, t * 1.5, seed + i * 5, 2, 2, 3.4, .34, ox=-t * sc * 1.5); o += g * n; wt += g
    return o / np.maximum(wt, 1e-3) - dens


def reflect(img, hz, cam_y, t, seed=1, amp=26.):
    """broken mirror of everything above the horizon line hz -> RGB for y > hz"""
    x, y = grid(); d = np.maximum(y - hz, 0); nz = nimg(H, W, 46, t * 1.2, seed, 2, 2, 2.6, .28); sy = hz - d * .92 + nz * amp * np.clip(d / 160., .15, 1.5); sx = x + nimg(H, W, 60, t, seed + 3, 2, 4, 2., .3) * amp * .5 * np.clip(d / 160., .1, 1.5); return cv2.remap(img.astype(f32), sx.astype(f32), np.clip(sy, 0, hz - 1).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def g8(f=14, figures=True):
    t = f / 24.; x, y = grid(); hz = 742; MX, MY, MR = 1490, 322, 338; IX, IY = 1392, 392
    g = np.clip(y / hz, 0, 1); im = ramp(g, [(0, '#0a0c2a'), (.35, '#1d2158'), (.62, '#4a3f86'), (.82, '#b0708f'), (.94, '#f7b7a8'), (1, '#ffd9c2')]).astype(f32); n = nimg(H, W, 300, 0, 5, 3, 4, 2.4, .8); im = im * (1 + .10 * n[..., None]); st = np.random.default_rng(4)
    for i in range(240): sx, sy = st.uniform(0, W), st.uniform(0, 380); im[int(sy):int(sy) + 1 + (i % 9 == 0), int(sx):int(sx) + 1 + (i % 9 == 0)] += .5 * st.random() + .2
    # far cirrus torn outward from the impact
    F = Field(2); r = R(8)
    for i in range(170):
        a = TAU * (i + r.random()) / 170; d0 = r.uniform(400, 1250); l = r.uniform(120, 520) * (d0 / 700) ** .6; P0 = np.array([IX + math.cos(a) * d0, IY + math.sin(a) * d0 * .66]); v = lin(12); an = math.atan2(math.sin(a) * .66, math.cos(a)) + r.normal(0, .05); P = P0 + np.stack([np.cos(an + .10 * v) * l * v, np.sin(an + .10 * v) * l * v], -1) + np.array([math.cos(a), math.sin(a) * .66]) * (f * 2.2); w = (1.5 + 13 * bms(r.random(), .12, .3) ** 1.5) * (d0 / 800); F.rib(P, [w * np.sin(math.pi * v ** .6) ** .9 * q for q in (1, .5)], 5.)
    im = put(im, F.cel(('#54488e', '#9a84c0'), alpha=(.50, .55)) * np.clip((hz - 40 - y) / 160., 0, 1)[..., None])
    ml, mg = moon(MX, MY, MR); im = put(im, ml) + mg * .8
    # cracks on the disc: gold light through torn stone
    K1, rg = Field(3), R(31)
    from .ki_bolt import bolt_pts
    for i in range(15):
        a = (i + rg.random() * .7) / 15 * TAU; l = rg.uniform(130, 520); P = bolt_pts((IX, IY), (IX + math.cos(a) * l, IY + math.sin(a) * l * .95), i + 3, .16, 4); s_ = np.linspace(0, 1, len(P)); w = 11 * rg.uniform(.5, 1.2) * (1 - s_) ** 1.1 + .6; K1.rib(P, [w + 2.5 * (1 - s_), w * .45 - .3, w * .22 - .5], 0.)
        for k in range(3):
            j = int(rg.integers(1, max(2, len(P) - 2))); b_ = a + rg.choice([-1, 1]) * rg.uniform(.4, 1.1); l2 = l * rg.uniform(.12, .34); Q = bolt_pts(P[j], P[j] + np.array([math.cos(b_), math.sin(b_)]) * l2, i * 7 + k, .18, 3); w2 = w[j] * .5 * np.linspace(1, .05, len(Q)) + .4; K1.rib(Q, [w2 + 1.2, w2 * .35 - .3, w2 * .1 - .5], 0.)
    crl = K1.cel(('#3b2470', '#ffb52e', '#fffdf0')); crl *= ml[..., 3:4]; im = put(im, crl)
    # cloud banks on the horizon
    for k, (c0, seed, yy, hh, cols) in enumerate((((-60, 260, 520), 3, hz + 8, 190, ('#4a3c82', '#7a5f9f', '#c48aa0', '#ffd0b8')), ((820, 1120, 1960, 1700), 9, hz + 6, 130, ('#4a3c82', '#86669f', '#d494a0', '#ffd6bc')), ((100, 640, 1240, 1540), 14, hz + 14, 74, ('#5a4488', '#a0789f', '#eeb0a4', '#ffe6cf')))):
        F = Field(4)
        for j, c in enumerate(c0): puffs(F, heap(c, yy, 420 - 60 * k, hh * (.8 + .4 * h1(seed, j)), seed + j, 30), (.75, -.75), k=5.)
        L = F.cel(cols, lw=(0, 0, 0, 0)); im = put(im, L * np.clip((hz + 2 - y), 0, 1)[..., None])
    # beam
    ax = Axis2(spl([(290, 838), (560, 716), (930, 575), (1392, 392)], n=12), lambda v: (205 * (1 - v) ** 1.2 + 50) * (.42 + .58 * min(1., v / .10) ** .7), tilt=.34); hl, hg = helix(ax, f, speed=13., collars=6); il, ig = impact_front(IX, IY, 92, f, ('gold', 'blue'), asp=.94, rot=-.4, back=(math.pi * .86, .30), n=30)
    # sea
    dd = np.clip((y - hz) / (H - hz), 0, 1); wv = glints(hz, t, 3); sea = ramp(dd, [(0, '#a9788f'), (.06, '#6a5490'), (.25, '#2a2e70'), (1, '#0b1038')]).astype(f32); full = put(put(im, hl) + hg, il) + ig; ref = reflect_soft(full, hz, t); dk = np.clip((-wv - .10) * 16 + .5, 0, 1)[..., None]; sea = sea * (1 - dk * .30); m0 = np.clip((wv - .02) * 16 + .5, 0, 1)[..., None]; m1 = np.clip((wv - .34) * 16 + .5, 0, 1)[..., None]; sea = sea * (1 - m0 * .55) + ref * m0 * .52; sea = sea * (1 - m1 * .6) + (ref * .9 + .10) * m1 * .62
    im = np.where((y > hz)[..., None], sea, im)
    # the trench under the beam: walls of water thrown up both sides
    B = Field(4); D = Field(1); tr = Axis2(spl([(60, 1075), (520, 900), (940, 800), (1260, 752)], n=10), lambda v: 120 * (1 - v) ** 1.3 + 5, tilt=.2); D.rib(tr.pt(tr.msamp(0, tr.M, 14), .06), [tr.r(tr.msamp(0, tr.M, 14)) * .78], 4.)
    for sg in (1, -1):
        for row, (pit, sc) in enumerate(((.52, 1.), (.27, .55))):
            for c in range(int(tr.M / pit)):
                z = bms(h1(c, sg, row, 2), .25, .35) * sc; mm = (c + h1(c, sg, row)) * pit; rr = float(tr.r(np.array([mm]))[0]); q = f / 24. * (1.6 + h1(c, sg, 4)) + h1(c, sg, 5); u = q - math.floor(q); p = tr.pt(np.array([mm]), sg * (.80 + .12 * h1(c, 7)))[0]; h = rr * (.55 + 1.25 * z) * (1 - (2 * u - 1) ** 2 * .75); an = -math.pi / 2 + sg * (-.42) + (.30); B.rib(spine(p[0], p[1], h, an, 10, .06, .7, u * 6 + c, sg * -.35), kwidths(10, h, rr * (.13 + .18 * z), LV, .2, 1.1, 1.), rr * .05)
        mmm = tr.msamp(0, tr.M, 12); B.rib(tr.pt(mmm, sg * .86), [tr.r(mmm) * q for q in (.20, .15, .07, .0)], 6.)
    im = put(im, D.cel(('#070b2c',)) * .9); gl = streams(tr, t, 0, tr.M, 5, 26, 16, .6, (1, 3), (.02, .06)).cel(('#ffcf6a',)); im = put(im, gl * .9); im = put(im, B.cel(('#22439e', '#5aa6ee', '#c6ebff', '#fff6dc'), lw=(1.6, 0, 0, 0), ink=('#0a1446', None, None, None), seed=3))
    # energy
    im = put(im, hl) + hg; im = put(im, il) + ig; ringL = [ring(IX, IY, 262 + 150 * k + f * 3, 6 - 2 * k, f, 'base', .88, -.4, 5 + k, .62, .9, spurs=30, wob=.05) for k in range(2)]
    for L in ringL: im = put(im, L)
    # debris of the moon, rocks and drops
    c = Cel()
    for i in range(46):
        a = TAU * h1(i, 1); d = 130 + 560 * h1(i, 2) ** .8; p = np.array([IX + math.cos(a) * d, IY + math.sin(a) * d * .85]); s = 3 + 17 * bms(h1(i, 3), .16, .3); tail = np.array([math.cos(a), math.sin(a) * .85]) * s * (3 + 6 * h1(i, 4)); c.fill(np.array([p - tail, p + [s * .3, s * .3], p + [s * .35, -s * .3]]), '#e9dcff', .5); rock(c, chunk(p[0], p[1], s, i, a + 1, 1.3), i, ('#6a58a8', '#b9a6ea', '#fff6e0'), (-.8, .5), None, 1.2 if s > 7 else 0, '#2c1a5c')
    im = put(im, c.layer())
    # cliff, two figures, muzzle
    c = Cel(); cl = spl([(-20, 1100), (-20, 930), (120, 918), (250, 926), (330, 940), (368, 964), (352, 1000), (388, 1030), (360, 1100)], True, corners=range(9), n=3); c.fill(cl, '#0c0b20')
    for P, q in (([(-20, 940), (110, 922), (90, 960), (-20, 990)], '#2b2a58'), ([(120, 918), (250, 926), (236, 952), (130, 948)], '#5f6ab8'), ([(250, 926), (330, 940), (322, 968), (262, 960)], '#ffcf7a'), ([(330, 940), (368, 964), (352, 1000), (326, 972)], '#f08a2a'), ([(150, 960), (236, 964), (214, 1040), (164, 1012)], '#1d1c44'), ([(262, 972), (318, 984), (338, 1050), (282, 1030)], '#8a3a1c'), ([(356, 1010), (388, 1030), (374, 1070)], '#c8641e')): c.fill(np.array(P, float), q)
    for P in ([(120, 918), (136, 960), (128, 1010), (150, 1080)], [(250, 926), (262, 970), (250, 1010), (276, 1080)], [(330, 940), (322, 980), (345, 1030)]): c.line(P, 3., '#05040f', taper=(.05, .5), seed=3)
    im = put(im, c.layer())
    mu, mg2 = muzzle(306, 828, 70, f, 'gold', 3, 1.25, 0., 1., 'blue', .8, math.pi * 1.02, 34); im = put(im, mu) + mg2
    if figures:
        for who, cx, ft, h, k, sd in (('vegeta', 96, 934, 236, 'gold', 3), ('goku', 214, 944, 246, 'blue', 4)): L = sil(cx, ft, h, who); L = lit(L, k, 1.2, f, sd, None, 7., .6, 0., (.85, -.35), True, 0); im = put(im, L)
    
    c = Cel()
    for i in range(26):
        p = np.array([60 + 520 * h1(i, 11), 1040 - 330 * h1(i, 12)]); s = 5 + 26 * bms(h1(i, 13), .15, .3); rock(c, chunk(p[0], p[1], s, i + 70, h1(i, 14) * 6), i, ('#12102e', '#2c2c62', '#5a66b4'), (.75, -.65), ('#ffc46a' if i % 2 else '#8fd6ff', s * .75, -s * .65), 1.4 if s > 8 else 0)
    im = put(im, c.layer())
    return im, dict(bloom=.30, vig=.16, bloom_thr=.82)


def mesa(c, x0, x1, top, base, seed, cols, lx=-1, steps=5, edge=None):
    """flat-topped rock tower: strata ledges, a hard vertical terminator, lit face on one side (lx -1 left / +1 right)"""
    r = R(seed, 41); n = 9; L = [(x0, base)]; R2 = [(x1, base)]; w = x1 - x0; ins = lambda v: w * .20 * v ** 1.3
    for i in range(1, n + 1):
        v = i / n; y = base + (top - base) * v; L += [(x0 + ins(v) + r.normal(0, w * .018), y + w * .02 * r.random()), (x0 + ins(v) + w * .022 * r.random(), y - 2)]; R2 += [(x1 - ins(v) + r.normal(0, w * .018), y + w * .02 * r.random()), (x1 - ins(v) - w * .022 * r.random(), y - 2)]
    P = np.array(L + R2[::-1]); c.fill(P, cols[0]); c.clip(P); T = []
    k = 8; ys = np.linspace(base, top - 4, k); xm = (x0 + x1) / 2 + lx * w * .10 + r.normal(0, w * .05, k); far = x0 - w if lx < 0 else x1 + w; c.fill(np.vstack([np.stack([xm, ys], -1), [[far, top - 9], [far, base + 9]]]), cols[1])
    xm2 = (x0 + x1) / 2 + lx * w * .30 + r.normal(0, w * .03, k); c.fill(np.vstack([np.stack([xm2, ys], -1), [[far, top - 9], [far, base + 9]]]), cols[2])
    for i in range(steps): y = base + (top - base) * (i + .4 + .4 * r.random()) / steps; c.fill(np.array([[x0 - 9, y], [x1 + 9, y + r.normal(0, 6)], [x1 + 9, y + w * .012 + 2], [x0 - 9, y + w * .016 + 2]]), cols[0], .8)
    for i in range(int(steps * 1.2)): x = x0 + w * r.random(); y = base + (top - base) * r.random(); c.line([(x, y), (x + r.normal(0, w * .02), y + (base - top) * .12), (x + r.normal(0, w * .03), y + (base - top) * .26)], max(1., w * .010), cols[0], taper=(.1, .7), seed=i)
    if edge: c.line(np.array(L if lx < 0 else R2), max(1.6, w * .022), edge, taper=(.02, .02), press=.5, seed=seed, tremor=.2); c.line(np.array([(x0 + ins(1), top), ((x0 + x1) / 2, top - 2), (x1 - ins(1), top)]), max(1.4, w * .016), edge, taper=(.02, .4), seed=seed + 1)
    c.unclip(); return c


def frieza_sil(cx, foot, h, flip=1, col='#120516'):
    """stand-in: ragged small silhouette walking left, one hand parting the light"""
    s = h / 100.; X = lambda P: np.array([[cx + flip * x * s, foot - y * s] for x, y in P]); c = Cel(); S = lambda P, k=(): c.fill(X(spl(P, True, corners=k, n=6)), col)
    S([(-16, 0), (-6, 0), (-5, 10), (-3, 26), (1, 44), (-7, 47), (-9, 30), (-11, 12)], (0, 1)); S([(10, 2), (17, 4), (13, 14), (10, 28), (8, 46), (-1, 46), (2, 30), (6, 14)], (0, 1)); S([(-7, 44), (9, 44), (9, 56), (14, 68), (10, 77), (-2, 79), (-9, 74), (-7, 58)])
    S([(-5, 76), (-12, 74), (-24, 70), (-36, 69), (-37, 65.4), (-24, 65), (-10, 67)]); 
    for a in (-.5, -.2, .1, .4): S([(-36, 68.4 + a * 3), (-42, 69 + a * 12), (-42.5, 67.6 + a * 12), (-36, 66 + a * 3)])
    S([(10, 76), (15, 70), (18, 58), (17, 46), (14, 45), (13, 57), (9, 66)]); S([(-4, 78), (-6, 86), (-5, 94), (1, 98.5), (7, 95), (9, 87), (5, 79)]); c.line(X(spl([(8, 46), (22, 36), (32, 24), (30, 10), (40, 6), (50, 14)], n=10)), 6 * s, col, taper=(.0, .5), press=0, seed=1, tremor=0); return c.layer()


def d5(f=14):
    t = f / 24.; x, y = grid(); hz = 700; g = np.clip(y / hz, 0, 1); im = ramp(g, [(0, '#05040d'), (.5, '#140d2c'), (.82, '#33205c'), (1, '#5a3476')]).astype(f32); n = nimg(H, W, 330, 0, 5, 3, 4, 2.6, .7)
    F = Field(2); r = R(5)                                     # cloud lids dragged along the beam
    for i in range(70): y0 = 40 + 470 * r.random() ** 1.2; x0 = r.uniform(-200, W); l = r.uniform(160, 760); v = lin(12); w = (3 + 24 * bms(r.random(), .16, .3)); F.rib(np.stack([x0 + l * v + f * 3, y0 - (y0 - 430) * .16 * v + 5 * np.sin(v * 4 + i)], -1), [w * np.sin(math.pi * v ** .7) ** .9 * q for q in (1, .45)], 8.)
    im = put(im, F.cel(('#241844', '#3d2a66'), alpha=(.8, .7))); ml, mg = moon(1180, 150, 64); im = put(im, ml) + mg * .8
    BX = lambda q: Axis2(spl([(405, 556), (760, 528), (1250, 520), (2060, 560)], n=12), lambda v: (54 + 300 * v ** 1.45) * (.35 + .65 * min(1., v / .06) ** .7), tilt=.30, warp=q); ax = BX(None); lx = np.clip(np.exp(-np.abs(y - 540) / 330.), 0, 1)[..., None]
    im = im + C('#ff9a2e') * lx * .20 * np.clip((x - 200) / 900, 0, 1)[..., None]
    c = Cel()                                                  # far, middle mesas
    for i, (x0, w, top) in enumerate(((-60, 150, 520), (150, 90, 590), (560, 130, 560), (820, 100, 600), (1080, 170, 500), (1400, 120, 560), (1600, 210, 470), (1850, 160, 540))): mesa(c, x0, x0 + w, top, hz + 30, i + 3, ('#1a1233', '#33225c', '#4c3272'), 1 if x0 < 900 else -1, 4, '#8a5a8f')
    for x0, w, top, sd in ((1430, 300, 330, 21), (700, 190, 470, 23)): mesa(c, x0, x0 + w, top, hz + 60, sd, ('#140d28', '#3a2348', '#6a3a4c'), -1, 6, '#ffb454')
    im = put(im, c.layer())
    gd = np.clip((y - hz) / (H - hz), 0, 1); gr = ramp(gd, [(0, '#7a4a34'), (.12, '#4a2a38'), (.5, '#1d1330'), (1, '#0a0716')]).astype(f32); wv = glints(hz, 0, 9, 0, 1.5); m0 = np.clip((wv - .10) * 12 + .5, 0, 1)[..., None]; lit_ = np.clip(1.15 - np.abs(y - 800) / 260., 0, 1)[..., None] * np.clip((x - 260) / 700, .15, 1)[..., None]; gr = gr * (1 - .25 * np.clip((-wv - .15) * 12 + .5, 0, 1)[..., None]); gr = gr + m0 * lit_ * C('#ff9a30') * .42; im = np.where((y > hz)[..., None], gr, im)
    # trench ploughed under the beam: plates standing up
    tr = Axis2(spl([(420, 742), (900, 770), (1400, 850), (2100, 1010)], n=10), lambda v: 18 + 170 * v ** 1.3); c = Cel(); D = Field(1); mm_ = tr.msamp(0, tr.M, 12); D.rib(tr.pt(mm_, 0.), [tr.r(mm_) * .62], 4.); im = put(im, D.cel(('#0a0614',)) * .85); im = put(im, streams(tr, t, 0, tr.M, 5, 30, 18, .5, (1, 3), (.02, .06)).cel(('#ffb24a',)) * .9)
    for i in range(96):                                        # slabs torn up, leaning out
        sg = -1 if i % 2 else 1; m = (i + .9 * h1(i, sg)) / 96 * tr.M; rr = float(tr.r(np.array([m]))[0]); p = tr.pt(np.array([m]), sg * (.55 + .55 * h1(i, 2)))[0]; h = rr * (.22 + .80 * bms(h1(i, 3), .16, .34)); ln = -sg * .10 + .5 * (h1(i, 4) - .35); wd = h * (.35 + .5 * h1(i, 5)); up = np.array([math.sin(ln), -math.cos(ln)]); rt = np.array([math.cos(ln), math.sin(ln)]); P = np.array([p - rt * wd + [0, h * .1], p - rt * wd * .8 + up * h * (.5 + .3 * h1(i, 6)), p - rt * wd * .2 + up * h * (.8 + .4 * h1(i, 7)), p + rt * wd * .5 + up * h * (.7 + .4 * h1(i, 8)), p + rt * wd + up * h * .25, p + rt * wd * .9 + [0, h * .12]]); rock(c, P, i + 7, ('#160e26', '#4a2638', '#b06030'), (.15, -1), ('#ffd27a', .05 * h, -h * .8), 1.4 if h > 18 else 0, '#0a0612')
    im = put(im, c.layer())
    F = Field(4); pts = []                                     # dust rolling off both sides of the trench
    for i in range(26):
        u = (h1(i, 1) + t * .40) % 1.; m = u * tr.M; rr = float(tr.r(np.array([m]))[0]); sg = 1 if i % 2 else -1; p = tr.pt(np.array([m]), sg * (1.05 + .35 * h1(i, 2)))[0]; pts += heap(p[0], p[1] + rr * .15, rr * (1.2 + .8 * h1(i, 5)), rr * (.5 + .5 * h1(i, 4)) * (.6 + .5 * u), i, 12)
    puffs(F, pts, (.15, -.95), lv=((1., 0.), (.95, .20), (.84, .46), (.62, .80)), k=8.); im = put(im, F.cel(('#1f1530', '#54303f', '#a85c34', '#f6b25a'), lw=(1.4, 0, 0, 0), ink=('#120a1c', None, None, None), seed=3))
    # the beam, parted by one hand
    HX, HY = 1262, 640; ax0 = BX(None); mh = float(np.interp(0, [0, 1], [0, 1])); ms = np.linspace(0, ax0.M, 400); pp = ax0.pt(ms, 0.); mh = float(ms[np.argmin(np.abs(pp[:, 0] - HX))]); rh = float(ax0.r(np.array([mh]))[0]); n0 = -.16 * 0
    axp = BX(parting(mh - .15, .40, 1.5, 3.0, 9., 0.)); bl, bg = beam(axp, f, 'gold', seed=2, speed=22., collars=5, veil=.55, dens=1.7); im = put(im, bl) + bg
    wk = wake(BX(None), mh + .05, .24, f, 'gold', 4, .22, 4.6, 0., '#3a1204'); im = put(im, wk)
    cap = Field(4); p = ax0.pt(np.array([mh - .28]), 0.)[0]                        # bow of light pressed against the palm
    for i in range(11): a = math.pi + (i - 5) * .26 + .1 * (h1(i, f // 2) - .5); l = rh * (.22 + .34 * h1(i, f // 2, 2)); q = p + np.array([math.cos(a), math.sin(a)]) * rh * .05; cap.rib(spine(q[0], q[1], l, math.pi - (math.pi - a) * 1.55, 9, .05, .6, i, (1 if i > 5 else -1) * .9), twidths(9, l, rh * .045, LV, .2, 1.2), 6.)
    im = put(im, cap.cel(('#e8780c', '#ffd24a', '#fff4c0', '#ffffff'), lw=(1.6, 0, 0, 0), ink=('#7a2b04', None, None, None)))
    fs = frieza_sil(HX + rh * .10, HY + 4, 196); fl = lit(fs, 'gold', 1.4, f, 2, None, 6., .7, 0., (-1., -.1), True, 0); im = put(im, fl)
    # Vegeta and the muzzle
    c = Cel(); P = spl([(-20, 1100), (-20, 640), (90, 622), (210, 628), (330, 640), (362, 664), (340, 720), (372, 790), (350, 1100)], True, corners=range(9), n=3); c.fill(P, '#0c0819'); c.clip(P)
    for Q, q in (([(-20, 650), (80, 626), (70, 676), (-20, 700)], '#2b2048'), ([(80, 626), (210, 628), (200, 672), (90, 668)], '#6a3f52'), ([(210, 628), (330, 640), (316, 682), (206, 672)], '#f2a33c'), ([(330, 640), (362, 664), (340, 720), (312, 690)], '#ffd97a'), ([(206, 690), (312, 700), (330, 860), (236, 820)], '#7a3820'), ([(340, 726), (372, 790), (356, 960), (330, 800)], '#e0802a'), ([(40, 700), (180, 690), (190, 860), (70, 900)], '#1c1430')): c.fill(np.array(Q, float), q)
    for Q in ([(80, 630), (96, 700), (80, 800), (110, 960)], [(206, 630), (212, 700), (240, 840), (232, 1000)], [(316, 660), (326, 760), (320, 900)]): c.line(Q, 3.4, '#05030c', taper=(.05, .5), seed=3)
    c.unclip(); im = put(im, c.layer()); mu, mg2 = muzzle(392, 556, 50, f, 'gold', 3, 1.2, 0., 1., None, .8, math.pi, 30); L = lit(sil(262, 644, 210, 'vegeta'), 'gold', 1.3, f, 3, None, 7., .6, 0., (1., -.2), True, 0); im = put(im, L); im = put(im, mu) + mg2 * .8
    lt, lg = storm([(11, axp.pt(np.array([2.]), -.8)[0], axp.pt(np.array([7.]), -1.5)[0], 5., 4, dict(forks=3)), (12, axp.pt(np.array([6.]), .9)[0], axp.pt(np.array([9.5]), 1.5)[0], 6., 9, dict(forks=3))], f, 'gold')
    if lt is not None: im = put(im, lt) + lg
    c = Cel()
    for i in range(44):
        u = (h1(i, 31) + t * .3) % 1.; p = np.array([420 + 1560 * u, 640 + 420 * h1(i, 32) * (.3 + u) - 330 * (1 if h1(i, 33) > .6 else 0)]); s = (4 + 40 * bms(h1(i, 34), .14, .3)) * (.25 + u); rock(c, chunk(p[0], p[1], s, i + 70, h1(i, 35) * 6 + t * 2), i, ('#140c20', '#4a2638', '#a8562c'), (0, -1 if p[1] > 560 else 1), ('#ffd27a', 0, (-s if p[1] > 560 else s)), 1.4 if s > 9 else 0, '#0a0612')
    return put(im, c.layer()), dict(bloom=.30, vig=.18, bloom_thr=.82)


def f3(f=14, phi=None):
    t = f / 24.; x, y = grid(); phi = -1.9 + TAU * f / 40. if phi is None else phi; cam = Camera3D((7.2 * math.cos(phi), 7.2 * math.sin(phi), .62), (0, 0, 3.15), 900.); hz = float(cam.project(np.array([math.cos(phi + math.pi) * 1e5, math.sin(phi + math.pi) * 1e5, 0.])[None])[0][0, 1]); cx = 960.
    g = np.clip(y / hz, 0, 1); im = ramp(g, [(0, '#050b0c'), (.45, '#0f211e'), (.8, '#28453a'), (1, '#6d7a4a')]).astype(f32); d = np.abs(x - cx); im = im + (np.exp(-d / 330.) * (.30 + .25 * g))[..., None] * C('#f6a416') * .55
    # storm punched into a ring: cloud heaps on circles overhead, far ones first, edges lit gold from the hole
    for k, (rad, hgt, cols, n) in enumerate(((1000, 330, ('#14332c', '#26503f', '#4f6a42', '#a89a4a'), 26), (560, 262, ('#0d2421', '#1b3d34', '#4a5a34', '#d8a83e'), 22), (330, 214, ('#081716', '#122c28', '#3a4426', '#f2bc48'), 18))):
        pts = []
        for i in range(n):
            a = TAU * (i + .6 * h1(k, i, 1)) / n + f * .010 * (3 - k); X = np.array([math.cos(a) * rad * (1 + .14 * (h1(k, i, 2) - .5)), math.sin(a) * rad * (1 + .14 * (h1(k, i, 3) - .5)), hgt * (1 + .22 * (h1(k, i, 4) - .5))]); px, z = cam.project(X[None])
            if z[0] < 40: continue
            s = cam.f / z[0]; pts.append((z[0], px[0], rad * .42 * s, i))
        for side in (-1, 1):
            F = Field(4)
            for z, q, w, i in sorted(pts, key=lambda q: -q[0]):
                if (q[0] < cx) != (side < 0) or not (-w < q[0] < W + w and -w < q[1] < H): continue
                puffs(F, heap(q[0], q[1] + w * .20, w * 1.5, w * .62, i + k * 50, 14, .6), (-side * 1.0, .42), lv=((1., 0.), (.94, .22), (.80, .62), (.56, 1.15)), k=16.)
            im = put(im, F.cel(cols, (cols[0], cols[0], cols[1], cols[2]), np.clip((np.abs(x - cx) - 260 + 60 * nimg(H, W, 90, 0, k, 1, 4)) / 420., 0, 1), lw=(1.4, 0, 0, 0), ink=('#04090a', None, None, None), seed=k))
    # sea pushed into a dish: crest ring far off, wall of water, spray
    U, V, T = sea_uv(cam); rr = np.sqrt(U * U + V * V); sea_m = (T > 0) & (rr > 10.5); wv = sea_waves(rr * 1.0, np.arctan2(V, U) * 26, T, t * 2, .8); dd = np.clip((y - hz) / 90., 0, 1); sea = ramp(dd, [(0, '#5c6a44'), (.3, '#22423c'), (1, '#0b1c1e')]).astype(f32); sea = sea + (np.clip((wv - .30) * 14 + .5, 0, 1) * np.exp(-d / 420.))[..., None] * C('#ffcf5c') * .8 + np.clip((wv - .42) * 14 + .5, 0, 1)[..., None] * C('#8fa88a') * .25
    im = np.where(sea_m[..., None], sea, im); W_ = Field(4); S_ = Field(1)
    for q0 in range(3):                                        # wall of water: three overlapping swells with torn lips
        rad = 62 - 9 * q0; a = np.linspace(phi + math.pi - 1.0, phi + math.pi + 1.0, 130); hh = (5.6 - 1.1 * q0) * (.62 + .30 * noise(a * 7, q0 * 3., t * .8, 3) + .16 * noise(a * 23, q0, t * 2, 5)); X = np.stack([np.cos(a) * rad, np.sin(a) * rad, hh * .5], -1); px, z = cam.project(X); s = cam.f / z; W_.rib(px, [hh * .5 * s * q for q in (1.1, .86, .52, .05)], 3., [0, hh * s * .05, hh * s * .22, hh * s * .42])
        for j in range(90):
            i = int(h1(q0, j, 1) * 129); u = (t * (1.2 + h1(q0, j, 2)) + h1(q0, j, 3)) % 1.; p, zz_ = cam.project(np.array([[math.cos(a[i]) * rad, math.sin(a[i]) * rad, hh[i] * .95]])); l = (.6 + 2.4 * bms(h1(q0, j, 4), .2, .3)) * cam.f / zz_[0] * (1 - (2 * u - 1) ** 2 * .8); sg = 1 if p[0, 0] > cx else -1; W_.rib(spine(p[0, 0], p[0, 1] + l * .1, l, -math.pi / 2 + sg * .5, 8, .05, .6, j, sg * .8), kwidths(8, l, l * .13, LV, .2, 1.1), 2.)
    im = put(im, W_.cel(('#12302c', '#2f5c4c', '#8aa07a', '#fff0c0'), lw=(1.3, 0, 0, 0), ink=('#07100f', None, None, None), seed=3))
    # island of rock: ground fan with gold light, cracks
    Gm = (T > 0) & (rr <= 10.5 + .8 * nimg(H, W, 60, 0, 4, 2, 4)); th = np.arctan2(V, U); fac = noise(np.cos(th[::4, ::4]) * 5 + rr[::4, ::4] * .5, np.sin(th[::4, ::4]) * 5 + rr[::4, ::4] * .35, 0, 7); fac = cv2.resize(fac.astype(f32), (W, H), interpolation=cv2.INTER_LINEAR); l = np.clip(1.5 - rr / 5.2 + .25 * np.sign(fac) * np.clip(np.abs(fac) * 3, 0, 1), 0, 1.3); gr = ramp(np.clip(l, 0, 1), [(0, '#0d1413'), (.25, '#2a2a1c'), (.5, '#7a5420'), (.78, '#e8a436'), (1, '#ffe9a0')]); st = np.round(np.clip(l, 0, 1) * 5) / 5; gr2 = ramp(st, [(0, '#0b1110'), (.2, '#1d2018'), (.4, '#4a3818'), (.6, '#a8701e'), (.8, '#f2b23c'), (1, '#fff0b4')]); im = np.where(Gm[..., None], gr2, im).astype(f32)
    c = Cel(); rg = R(12)
    for i in range(17):
        a = TAU * (i + rg.random()) / 17; r0 = 1.6; P = []
        while r0 < 11: P.append((math.cos(a) * r0, math.sin(a) * r0, 0.)); r0 += rg.uniform(.5, 1.4); a += rg.normal(0, .10)
        px, z = cam.project(np.array(P)); ok = z > .5
        if ok.sum() > 2: q = px[ok]; w = min(7., np.median(cam.f / z[ok]) * .04); c.line(q, max(1.5, w), '#0a0d0c', taper=(.3, .3), seed=i, tremor=0); c.line(q + [0, w * .35], max(1., w * .4), '#ffd46a', taper=(.0, .8), seed=i, tremor=0)
    im = put(im, c.layer())
    # floating plates: sorted by depth, behind and in front of the pillar
    PL = []
    for i in range(64):
        a = TAU * h1(i, 1) + .25 * t; r_ = 2.4 + 5.6 * h1(i, 2) ** .8; z_ = .15 + 6.5 * h1(i, 3) ** 1.7 + .25 * math.sin(t * 3 + i) + .9 * t; X = np.array([math.cos(a) * r_, math.sin(a) * r_, z_]); px, z = cam.project(X[None]); PL.append((z[0], px[0], i, z_))
    PL.sort(key=lambda q: -q[0]); zc = cam.cam((0, 0, 1.))[2]
    def plates(front):
        c = Cel()
        for z, q, i, z_ in PL:
            if z < .9 or (z < zc) != front: continue
            s = cam.f / z * (.06 + .42 * bms(h1(i, 4), .16, .32)) * .55
            if s < 1.5 or not (-s < q[0] < W + s and -s < q[1] < H + s) or (front and abs(q[0] - cx) < 170 + s): continue
            lx = np.sign(cx - q[0]) or 1.; back = not front; rock(c, chunk(q[0], q[1], s, i + 40, (h1(i, 5) - .5) * 1.1 + t * (h1(i, 6) - .5), 1.9), i, ('#120f0c', '#3a2a16', '#a8721e') if front else ('#2a2010', '#7a5420', '#e8a436'), (lx, -.25), ('#fff0b4' if back else '#ffd46a', lx * s * .9, -s * .1), 1.8 if s > 10 else 0, '#050605')
        return c.layer()
    im = put(im, plates(False)); from .ki_qa import stand
    fp, fz = cam.project(np.array([[0, 0, 0.], [0, 0, 1.86]])); fh = fp[0, 1] - fp[1, 1]; fig, face = stand(fp[0, 0], fp[0, 1], fh); inn, fr, li = pillar3(cam, f, R0=1.35, height=10., n=84, hole=(fp[0, 0], (fp[0, 1] + fp[1, 1]) / 2 + 4, fh * .40, fh * .62))
    st_ = [(int(f) - 1, (250, -10), (520, hz + 10), 10., 4 + f // 5, dict(forks=5)), (int(f) - 2, (1660, -10), (1480, hz + 6), 9., 9 + f // 5, dict(forks=5)), (int(f) - 1, (760, 80), (330, 500), 6., 12 + f // 5, dict(forks=3))]; ll, lg = storm(st_, f, 'gold'); im = put(im, ll) + lg
    im = put(im, inn); im = put(im, lit(fig, 'gold', 1.1, f, 2, face, 4.5, .4, .10, None, True, 1.8)); im = im + li * (1 - .93 * fig[..., 3:4]); im = put(im, fr); b, fr2, al = arcs(fig, f, 'blue', 6, 3, face=face); im = put(im, fr2) + al; im = put(im, plates(True))
    for k in range(2): q = ring3(cam, (0, 0, .04), (0, 0, 1), 2.6 + 2.6 * ((t * 1.6 + k * .5) % 1.), .10, 'gold', 5 + k, .45, .85, .05, spurs=0); im = put(im, q) if q is not None else im
    return im, dict(bloom=.30, vig=.2, bloom_thr=.82)


def _curve(keys, u):
    """smooth 3-D path through keys [(frame, (x, y, z))] at frames u"""
    k = np.array([q[0] for q in keys], float); P = np.array([q[1] for q in keys], float); from scipy.interpolate import CubicSpline as PI; return np.stack([PI(k, P[:, i], bc_type='natural')(np.clip(u, k[0], k[-1])) for i in range(3)], -1)


_D3 = {}


def d3_plates():
    """sea from high above at night (moon road), coast with mesas, two decks of cloud"""
    if _D3: return _D3
    from .ki_trail import punch
    cam = Camera3D((0, -900, 1500), (40, 330, 0), 1500.); U, V, T = sea_uv(cam); x, y = grid(); wv = sea_waves(U, V, T, 0., 11.); big = noise(U[::4, ::4] / 620., V[::4, ::4] / 520., 0, 3); big = cv2.resize(big.astype(f32), (W, H), interpolation=cv2.INTER_LINEAR); road = np.exp(-(((U - 420) * .8 + (V - 900) * .25) / 330.) ** 2) * np.clip((V + 700) / 1500., 0, 1); g = np.clip(y / H, 0, 1); sea = ramp(g, [(0, '#2a2760'), (.4, '#171a4a'), (1, '#0a0c2c')]).astype(f32); sea = sea * (1 + .22 * np.round(big * 3)[..., None] / 3.); fr = np.exp(-(((U - 120) * .9 + (V - 500) * .30) / 250.) ** 2) * np.clip((V + 900) / 1300., .15, 1); wv2 = sea_waves(U, V, T, 0., 4.); a0 = np.clip((wv2 - (1.02 - .72 * fr)) * 26 + .5, 0, 1)[..., None]; a1 = np.clip((wv2 - (1.30 - .72 * fr)) * 26 + .5, 0, 1)[..., None]; sea = sea * (1 - np.clip((-wv - .18) * 26 + .5, 0, 1)[..., None] * .20); sea = sea * (1 - .55 * np.clip((wv2 - .40) * 26 + .5, 0, 1)[..., None]) + np.clip((wv2 - .40) * 26 + .5, 0, 1)[..., None] * C('#2f338a') * .55; sea = sea * (1 - a0) + C('#8a78e0') * a0; sea = sea * (1 - a1) + C('#f0e6ff') * a1
    # coast, lower left: cliffs, beach line, mesas seen from above with violet-lit tops
    cst = np.array([(-1400, -200), (-1010, -120), (-760, -260), (-560, -230), (-470, -420), (-250, -520), (-180, -700), (-1500, -900)], float); px = cam.project(np.c_[cst, np.zeros(len(cst))])[0]; C_ = spl(px, True, n=8); c = Cel(); c.fill(C_ + [0, 26], '#8a7fd0', .9); c.fill(C_ + [0, 18], '#12122e'); c.fill(C_, '#2a2550'); c.clip(C_); r = R(8)
    for i in range(40): p = px.mean(0) + r.normal(0, 1, 2) * [330, 190]; L = spl(p + np.array([(math.cos(q), math.sin(q) * .6) for q in np.linspace(0, TAU, 9)[:-1]]) * r.uniform(30, 120) * r.uniform(.7, 1.2, (8, 1)), True, n=4); c.fill(L, ('#352f66', '#221e44', '#40397a')[i % 3], .8)
    c.unclip()
    for i, (mx, my, rr, hh) in enumerate(((-620, -330, 60, 150), (-860, -260, 84, 210), (-420, -560, 70, 120), (-1060, -420, 70, 170), (-760, -520, 50, 90), (-300, -640, 48, 80))):
        a = np.linspace(0, TAU, 15)[:-1]; q = rr * (1 + .16 * r.normal(0, 1, 14)); B = cam.project(np.stack([mx + np.cos(a) * q, my + np.sin(a) * q, 0 * a], -1))[0]; Tp = cam.project(np.stack([mx + np.cos(a) * q * .82, my + np.sin(a) * q * .82, 0 * a + hh], -1))[0]; import scipy.spatial as ssp; h_ = ssp.ConvexHull(np.vstack([B, Tp])).vertices; A = np.vstack([B, Tp])[h_]; sh = B + [-hh * .5, hh * .34]; c.fill(np.vstack([B, sh])[ssp.ConvexHull(np.vstack([B, sh])).vertices], '#0c0b22', .8); c.fill(A, '#1a1738'); c.clip(A); c.fill(A + [rr * .5, -4], '#3a3470'); c.fill(A + [rr * .95, -4], '#6a58b0'); c.unclip(); c.fill(Tp, '#57499a'); c.fill(Tp * .82 + Tp.mean(0) * .18 + [rr * .08, -2], '#8f7ad0')
    sea = put(sea, c.layer())
    def deck(seed, n, hz_, w, cols, light):
        F = Field(4); rg = R(seed); items = []
        for i in range(n):
            X = np.array([rg.uniform(-1300, 1300), rg.uniform(-600, 1900), hz_]); p, z = cam.project(X[None]); items.append((z[0], p[0], w * cam.f / z[0] * (.35 + 1.1 * bms(rg.random(), .25, .35)), i))
        for z, p, s, i in sorted(items, key=lambda q: -q[0]): 
            if abs(p[0] - 960) < 1200 and -200 < p[1] < 1300: puffs(F, [(a0, b0, r0, s0) for a0, b0, r0, s0 in heap(p[0], p[1], s * 2.8, s * 1.1, seed + i, 22, .6)], light, lv=((1., 0.), (.95, .20), (.82, .55), (.52, 1.08)), k=9.)
        return F.cel(cols, lw=(1.2, 0, 0, 0), ink=('#0c0a26', None, None, None), seed=seed)
    _D3.update(cam=cam, sea=sea.astype(f32), lo=deck(5, 40, 260, 165, ('#1d1b4c', '#37317a', '#7466bd', '#c3b4f4'), (.6, -.8)), wv=wv2); return _D3


def d3(f=22):
    from .ki_trail import trail, trail_cel, head, clash_star, sea_dent, punch
    D = d3_plates(); cam = D['cam']; t = f / 24.; im = D['sea'].copy(); hit = [(5, (-520, 300, 330)), (10, (60, 720, 420)), (15, (470, 260, 300)), (20, (90, -40, 380))]
    def V3(k): return dict(hit)[k]
    G = [(-10, (-900, -300, 200)), (0, (-800, 60, 260)), (5, V3(5)), (7.5, (-260, 620, 460)), (10, V3(10)), (12.5, (380, 600, 280)), (15, V3(15)), (17.5, (420, -120, 460)), (20, V3(20)), (22.5, (-230, 60, 330)), (26, (-420, 420, 380))]; Bk = [(-10, (-1000, 500, 300)), (0, (-640, 620, 520)), (5, (-380, 420, 260)), (7.5, (-60, 260, 360)), (10, (190, 640, 300)), (12.5, (640, 420, 520)), (15, (560, 120, 380)), (17.5, (180, 160, 250)), (20, (-60, -120, 300)), (22.5, (60, 300, 520)), (26, (330, 520, 300))]; Vi = [(-10, (700, 900, 500)), (0, (-200, 60, 420)), (5, (-500, 290, 340)), (7.5, (-480, 640, 260)), (10, (70, 730, 430)), (12.5, (200, 300, 520)), (15, (480, 250, 310)), (17.5, (560, -60, 240)), (20, (100, -30, 390)), (22.5, (240, 160, 260)), (26, (60, 60, 420))]
    for i, (k, X) in enumerate(hit):
        L = sea_dent(cam, (X[0], X[1], 0), f - k, X[2] * .42, 3 + i * 7)
        if L is not None: im = put(im, L)
    cl = D['lo']
    for i, (k, X) in enumerate(hit):
        a = f - k
        if a > 0: p, z = cam.project(np.array([[X[0], X[1], 260.]])); r = 250 * cam.f / z[0] * (1 - math.exp(-a / 4.)); cl = punch(cl, p[0, 0], p[0, 1], r, .80, i + 3)
    # cloud shadow on the sea
    sh = np.roll(np.roll(cl[..., 3], 46, 0), -70, 1); im = im * (1 - .34 * sh[..., None]); im = put(im, cl); li = None
    P = [('violet', Vi, 12.), ('blue', Bk, 10.), ('gold', G, 11.)]; LY = []
    for kind, K, w in P:
        u = np.linspace(f - 13, f, 130); X = _curve(K, u); px, z = cam.project(X); F = trail(px, kind, w, 5, None, 1500. / z, True, t); lay = trail_cel(F, kind); LY.append((kind, lay)); old = _curve(K, np.linspace(f - 22, f - 11, 60)); q, zz = cam.project(old); G_ = Field(2); v = lin(60); bd = np.clip(1.6 * noise(v * 12, K[0][0], 0, 3) + .2, 0, 1); G_.rib(q, [2.2 * v * bd, 1. * v * bd - .2], 0.); im = put(im, G_.cel((pal(kind)['c'][1], pal(kind)['c'][2])) * .8)
        ref = cv2.GaussianBlur(cv2.resize(lay[..., :3], (W // 8, H // 8), interpolation=cv2.INTER_AREA), (0, 0), 9); im = im + cv2.resize(np.roll(ref, 9, 0), (W, H), interpolation=cv2.INTER_LINEAR) * (.35 + .9 * np.clip((D['wv'] - .1) * 6, 0, 1))[..., None] * (1 - .6 * cl[..., 3:4]) * .9
    for kind, lay in LY: im = put(im, lay) + bloom(lay, .45, (6, 22, 70), (.35, .3, .28), pal(kind)['glow'], .8)
    for kind, K, w in P: p, z = cam.project(_curve(K, np.array([float(f)]))); im = put(im, head(p[0, 0], p[0, 1], w * 1500 / z[0] * 1.25, kind, f))
    for i, (k, X) in enumerate(hit):
        p, z = cam.project(np.array([X])); L, g = clash_star(p[0, 0], p[0, 1], 46, f - k, ('gold' if i % 2 == 0 else 'blue', 'violet'), 5 + i, h1(i, 3) * 3)
        if L is not None: im = put(im, L) + g
        a = f - k
        if 0 < a < 16: q = ring3(cam, X, (h1(i, 4) - .5, h1(i, 5) - .5, 1.), 60 + 230 * (1 - math.exp(-a / 5.)), 9 * math.exp(-a / 7.), 'base', 5 + i, .35, .9 * math.exp(-a / 9.), spurs=14); im = put(im, q) if q is not None else im
    return im, dict(bloom=.28, vig=.2, bloom_thr=.8)


SC = {'beam': g8, 'flash': d5, 'pillar': f3, 'threads': d3}
if __name__ == '__main__':
    k = sys.argv[1]; fr = [int(v) for v in sys.argv[2:]] or [14]; ims = []
    for f in fr:
        t = time.time(); im, kw = SC[k](f); print(k, f, '%.2fs' % (time.time() - t)); im = comp.finish(im, f, **dict(dict(grain=.010), **kw)); ims.append(im)
    if len(ims) == 1: comp.save(OUT + f'{k}.jpg', comp.to8(ims[0]), 94)
    else:
        w = 640; ims = [cv2.resize(comp.to8(i), (w, 360), interpolation=cv2.INTER_AREA) for i in ims]; n = 2 if len(ims) <= 8 else 3; rows = [np.hstack((ims[i:i + n] + [np.zeros_like(ims[0])] * n)[:n]) for i in range(0, len(ims), n)]; comp.save(OUT + f'strip_{k}.jpg', np.vstack(rows), 90)
