"""GOHAN'S EYE, extreme close-up (c2): the moon of c1 reflected in it, the pupil closing.
    E = Eye();  img = E.draw(pupil=208, f=0, gaze=(0, 0), tear=1., moon=1., wide=0., tremble=0)     -> float RGB 1080x1920
    the pupil sits on the film's match circle: centre (960, 486), radius 208 at pupil=208 (c1's moon), 12 = a point
"""
from chars.fkit import *
from db.paint import ramp, grid
from db.comp import over, screen, blur, radial

CX, CY, RI = 960., 486., 356.
FIN = dict(bloom=.30, vig=.2, soft=.05, bloom_thr=.84)          # finish for c2: no diffusion, the pupil must stay black
SK = ('#d0aebe', '#8f6fa6', '#473876', '#1f1742')          # night skin: lit, shadow, deep, darkest
LINE = '#0c0718'


def lidcurve(P, n=14): return spl(A(P, float), n=n)


UP = [(96, 652), (190, 520), (330, 346), (520, 196), (740, 112), (960, 88), (1180, 110), (1400, 178), (1590, 284), (1740, 404), (1836, 506)]
LO = [(96, 652), (150, 694), (280, 770), (450, 846), (680, 902), (930, 922), (1180, 900), (1410, 836), (1600, 744), (1750, 628), (1836, 506)]


class Eye:
    def __init__(s, seed=3):
        s.seed = seed; s.gx, s.gy = grid(FH, W); s._polar(); s._plates()

    # ---------------------------------------------------------------- iris: fibres drawn by hand in polar space
    def _polar(s):
        TW_, TH_ = 2880, 420; r = np.random.default_rng(s.seed); cel = Cel(TW_ + 200, TH_, ss=1); L = lambda P, w, col, a=1., tp=(.3, .5): cel.line(A(P, float), w, col, taper=tp, press=.4, seed=int(r.integers(1 << 30)), alpha=a, tremor=.8)
        def fib(x, y0, y1, w, col, a, wob=10, tp=(.25, .5)):          # one fibre from the pupil (y 0) out to the limbus (y TH): a slow S, never lightning
            n = 5; y = np.linspace(y0, y1, n); xx = x + r.normal(0, wob) * np.sin(np.linspace(0, math.pi, n) * r.uniform(.6, 1.6) + r.uniform(0, 3)) + r.normal(0, wob * .7) * np.linspace(0, 1, n); P = spl(np.c_[xx + 100, y], n=8)
            for k in (-TW_, 0, TW_):
                Q = P + [k, 0]
                if Q[:, 0].max() > 0 and Q[:, 0].min() < TW_ + 200: L(Q, w, col, a, tp)
        cel.fill(A([(0, 0), (TW_ + 200, 0), (TW_ + 200, TH_), (0, TH_)]), '#150c1e')
        for j in range(420): fib(r.uniform(0, TW_), r.uniform(-30, 200), r.uniform(230, 440), r.uniform(7, 20), r.choice(['#24142c', '#321c3c', '#1c1026', '#38203a', '#2a1a44']), .75)
        for j in range(520): fib(r.uniform(0, TW_), r.uniform(60, 170), r.uniform(250, 410), r.uniform(2.0, 6.5), r.choice(['#4a2c5e', '#5c3a78', '#5a3640', '#6a4450', '#3c2654']), r.uniform(.4, .85))
        for j in range(34):                                        # crypts: dark lens-shaped gaps
            x = r.uniform(0, TW_); y = r.uniform(130, 235); w, h = r.uniform(10, 26), r.uniform(26, 70)
            for k in (-TW_, 0, TW_): cel.fill(spl(A([(x - w * .2, y - h), (x + w, y - h * .2), (x + w * .4, y + h), (x - w * .8, y + h * .3)]) + [100 + k, 0], True, n=7), '#080410', .80)
        for j in range(420): fib(r.uniform(0, TW_), r.uniform(-20, 14), r.uniform(90, 170), r.uniform(2.0, 6.0), r.choice(['#8a5e50', '#a0705c', '#744a58', '#b4826a', '#694650']), r.uniform(.35, .85), 6, (.15, .6))
        for j in range(260): fib(r.uniform(0, TW_), r.uniform(80, 200), r.uniform(240, 380), r.uniform(1.0, 3.0), r.choice(['#8468b8', '#a48cd4', '#7a5c9c', '#9c7a80']), r.uniform(.25, .65), 8, (.4, .5))
        # collarette: a ragged ring
        th = np.linspace(0, TW_ + 200, 260); yy = 126 + 12 * np.sin(th * TAU / TW_ * 5 + 1) + 8 * np.sin(th * TAU / TW_ * 13) + np.interp(th, np.linspace(0, TW_ + 200, 120), r.normal(0, 7, 120))
        L(np.c_[th, yy], 9, '#0e0718', .45, (0, 0)); L(np.c_[th, yy - 9], 3, '#b88e88', .22, (0, 0))
        for k in range(3):                                          # contraction furrows
            yy = 258 + 46 * k + np.interp(th, np.linspace(0, TW_ + 200, 60), r.normal(0, 9, 60)); L(np.c_[th, yy], 3.2, '#120a20', .35, (0, 0))
        t = cel.rgba()[:, 100:100 + TW_, :3]; s.tex = np.hstack([t, t[:, :8]]).astype(np.float32); s.TW_, s.TH_ = TW_, TH_

    def iris(s, cx, cy, rp):
        """RGB + alpha of the iris with pupil radius rp, centre (cx, cy)"""
        x, y = s.gx - cx, s.gy - cy; rr = np.sqrt(x * x + y * y); th = np.arctan2(y, x); ring = 1 + .0016 * np.sin(th * 7 + 1) + .0012 * np.sin(th * 19)
        edge = rp * (1 + .014 * np.sin(th * 23 + 2) + .010 * np.sin(th * 41)) if rp > 30 else rp * np.ones_like(th)
        u = (rr - edge) / np.maximum(RI * ring - edge, 1.); mx = ((th + math.pi) / TAU * s.TW_).astype(np.float32); my = (np.clip(u, 0, 1) * (s.TH_ - 1)).astype(np.float32)
        c = cv2.remap(s.tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # the light passes the cornea from upper left and lights the far side of the bowl (lower right)
        d = (x * .50 + y * .866) / RI; c = c * (.42 + 1.05 * smooth(-.30, .95, d)[..., None]); lo = smooth(.30, .95, d) * smooth(1.0, .72, u) * smooth(.05, .5, u)
        c = c + lo[..., None] * rgb('#a070e8') * .36
        c = c * (1 - .80 * smooth(.74, 1.0, u)[..., None]) * (1 - .45 * smooth(.14, .0, u)[..., None])      # limbal ring, pupil ruff
        pa = np.clip(edge - rr + 1.2, 0, 1)[..., None]; c = c * (1 - pa) + rgb('#030106') * pa
        a = np.clip((RI * ring - rr) / 5. + .5, 0, 1); return c.astype(f32), a.astype(f32)

    # ---------------------------------------------------------------- static paint
    def _plates(s):
        x, y = s.gx, s.gy; r = np.random.default_rng(s.seed + 1); n1 = fbm(FH, W, 5, seed=7, scale=260) if False else cv2.resize(cv2.GaussianBlur(r.standard_normal((FH // 8, W // 8)).astype(f32), (0, 0), 5), (W, FH), interpolation=cv2.INTER_CUBIC); n1 /= n1.std()
        pores = cv2.GaussianBlur(r.standard_normal((FH, W)).astype(f32), (0, 0), 1.3); pores /= pores.std()
        up, lo = lidcurve(UP), lidcurve(LO); s.up, s.lo = up, lo; s.open = np.vstack([up, lo[::-1]])
        # skin: cool moonlight from the front, darker toward nose and temple, hard shadow shapes
        g = smooth(1250, -100, y) * .25 + .55 + .10 * np.clip(n1, -2, 2) * .5; d = np.sqrt(((x - 980) / 1250) ** 2 + ((y - 500) / 820) ** 2); g = g * (1.12 - .55 * smooth(.45, 1.25, d))
        skin = ramp(np.clip(g, 0, 1), [(0, SK[3]), (.30, SK[2]), (.56, SK[1]), (.85, SK[0]), (1, '#dcc6ea')]); skin = skin * (1 + .020 * pores[..., None])
        cel = Cel(W, FH); H_ = lambda P, col, a=1., n=10: cel.fill(spl(A(P, float), True, n=n), col, a)
        # the nose plane and the temple plane: two long cuts with one corner each
        cel.fill(spl(A([(-60, 1140), (-60, 120), (40, 300), (86, 470), (40, 620), (64, 770), (190, 905), (250, 1010), (330, 1140)]), True, corners=(0, 1, 4, 8), n=10), SK[2], .85)
        cel.fill(spl(A([(1990, 220), (1930, 380), (1896, 520), (1820, 660), (1700, 800), (1560, 930), (1470, 1140), (1990, 1140)]), True, corners=(0, 2, 6, 7), n=10), SK[2], .80)
        # bag under the eye: strained, two folds
        cel.fill(spl(A([(300, 846), (480, 918), (700, 968), (940, 986), (1190, 966), (1420, 904), (1600, 812), (1480, 922), (1240, 1004), (960, 1034), (690, 1012), (460, 946)]), True, corners=(0, 6), n=9), SK[1], .75)
        for P, w, a in (([(250, 820), (460, 926), (700, 984), (940, 1004)], 5.5, .8), ([(1650, 776), (1480, 900), (1250, 978), (1060, 1002)], 5.0, .7), ([(150, 764), (260, 862), (420, 958), (560, 1010)], 3.6, .6), ([(1560, 930), (1420, 1004), (1260, 1052)], 3.2, .5), ([(60, 700), (110, 742), (160, 764)], 4., .8)): cel.line(spl(A(P, float), n=10), w, mixc(LINE, SK[3], .4), taper=(.5, .5), press=.35, seed=int(P[0][0]), alpha=a)
        hatch(cel, spl(A([(330, 884), (520, 962), (760, 1010), (900, 1022), (760, 1052), (520, 1016), (360, 940)]), True, n=6), .95, 15, 2.0, mixc(LINE, SK[3], .5), 5, length=70, alpha=.55)
        hatch(cel, spl(A([(1640, 820), (1500, 940), (1330, 1010), (1450, 1030), (1620, 950), (1720, 850)]), True, n=6), 2.2, 15, 2.0, mixc(LINE, SK[3], .5), 6, length=70, alpha=.5)
        # upper lid: lit lid plane under one deep crease
        fold_ = spl(A([(46, 560), (140, 380), (300, 190), (500, 46), (740, -34), (960, -58), (1200, -34), (1440, 44), (1640, 160), (1800, 300), (1900, 440)]), n=12); m_ = len(fold_); tt = np.linspace(0, 1, m_)
        cel.fill(np.vstack([up - [0, 0], fold_[::-1]]), mixc(SK[0], SK[1], .55), .9); cel.fill(np.vstack([fold_ - np.c_[0 * tt, 44 * np.sin(math.pi * tt) ** .6], fold_[::-1]]), SK[1], .55); cel.fill(np.vstack([fold_, (fold_ + np.c_[0 * tt, 30 * np.sin(math.pi * tt) ** .5])[::-1]]), SK[2], .80); cel.line(fold_, 8, LINE, taper=(.35, .35), press=.35, seed=4)
        sk = over(skin, cel.rgba()); s.skin = sk.astype(f32)
        # sclera
        om = mask_of(s.open); s.om = om; e = np.sqrt(((x - CX) / 940) ** 2 + ((y - 505) / 470) ** 2)
        w_ = ramp(np.clip(e, 0, 1.2), [(0, '#efe8fb'), (.45, '#e2d8f4'), (.75, '#b6a6dc'), (1.0, '#7c6bb4'), (1.2, '#54458e')]); w_ = w_ * (1 + .018 * np.clip(n1, -2, 2)[..., None])
        cel = Cel(W, FH); s._veins(cel, r); w_ = over(w_, cel.rgba()); s.white = w_.astype(f32)
        # lid shadow on the ball: one hard shape and a soft one under it
        n = len(up); t = np.linspace(0, 1, n); th = 84 * np.sin(math.pi * np.clip(t, .02, .98)) ** .5 + 8 * np.sin(t * 40) * np.sin(math.pi * t); s.lsh = mask_of(np.vstack([up - [0, 30], (up + np.c_[0 * t, th])[::-1]])); s.lsh2 = blur(mask_of(np.vstack([up - [0, 60], (up + np.c_[0 * t, th * 1.9])[::-1]])), 34)
        s.losh = blur(1 - mask_of(np.vstack([up - [0, 400], (lo + [0, -20])[::-1]])), 22) * om
        s._lids()

    def _veins(s, cel, r):
        def branch(p, ang, ln, w, depth):
            P = [A(p, float)]; a = ang
            for k in range(6): a += r.normal(0, .28); P.append(P[-1] + A([math.cos(a), math.sin(a)]) * ln / 6)
            cel.line(spl(A(P), n=6), w, r.choice(['#a64a86', '#8a3f90', '#b4587c']), taper=(.05, .8), press=.3, seed=int(r.integers(1 << 30)), alpha=.50)
            if depth:
                for k in range(2): q = P[int(r.integers(2, 6))]; branch(q, a + r.choice([-1, 1]) * r.uniform(.3, .8), ln * .55, w * .62, depth - 1)
        for p, a in (((170, 650), .05), ((200, 690), .22), ((260, 560), -.25), ((1790, 520), 3.05), ((1760, 590), 2.9), ((1700, 430), 3.35), ((330, 760), -.1), ((1600, 720), 3.3)): branch(p, a, r.uniform(150, 260), r.uniform(2.4, 3.6), 2)

    def _lids(s):
        """everything in front of the ball: lid rims, tear duct, lashes, lash line"""
        r = np.random.default_rng(s.seed + 9); up, lo = s.up, s.lo; cel = Cel(W, FH); n = len(up); t = np.linspace(0, 1, n)
        # lower lid: wet rim (its thickness)
        m = len(lo); tl = np.linspace(0, 1, m); hk = 30 * np.sin(math.pi * np.clip(tl, .03, .97)) ** .5; out = lo + np.c_[0 * tl, hk]
        cel.fill(np.vstack([lo - [0, 2], out[::-1]]), '#b27fae'); cel.fill(np.vstack([lo + np.c_[0 * tl, hk * .55], out[::-1]]), '#7d5490'); cel.line(out[int(.08 * m):int(.95 * m)], 5.0, mixc(LINE, '#3a1c4a', .35), taper=(.3, .25), press=.35, seed=3)
        # lower lashes: short, in little clumps
        for j in range(22):
            u = .15 + .79 * (j + r.uniform(-.22, .22)) / 21; i = int(u * (m - 1)); a0_ = math.pi / 2 - (u - .5) * 1.5
            for k_ in range(int(r.integers(1, 4))):
                p = out[i] + A([(k_ - 1) * 9 + r.normal(0, 3), -3]); a = a0_ + (k_ - 1) * .16 + r.normal(0, .06); ln = (30 + 62 * u) * r.uniform(.55, 1.12); bend = (u - .42) * .9
                Q = spl(A([p, p + A([math.cos(a), math.sin(a)]) * ln * .55, p + A([math.cos(a - bend * .5), math.sin(a - bend * .5)]) * ln]), n=8); cel.fill(wedge(Q, r.uniform(5, 8), 0., .8), LINE)
        # caruncle
        C = spl(A([(92, 654), (134, 622), (180, 624), (214, 652), (196, 694), (150, 700), (112, 680)]), True, corners=(0,), n=10); cel.fill(C, '#7a4580'); cel.clip(C); cel.fill(spl(A([(128, 644), (160, 630), (196, 646), (176, 666), (136, 664)]), True, n=8), '#a0658e'); cel.fill(spl(A([(150, 690), (190, 676), (230, 690), (200, 730), (140, 730)]), True, n=8), '#4c2a60'); cel.unclip()
        for x, y_, k in ((150, 640, 8), (182, 650, 5)): cel.fill(ell(x, y_, k * 1.6, k * .6, rot=-.35), '#ffffff', .85)
        cel.line(np.vstack([C, C[:1]]), 3.0, LINE, taper=(.2, .2), seed=8)
        # upper lid rim (underside) and the great lash line
        rim_ = up + np.c_[0 * t, -20 * np.sin(math.pi * t) ** .5]; cel.fill(np.vstack([up + [0, 3], rim_[::-1]]), '#6a447e')
        th = np.interp(t, [0, .06, .2, .42, .64, .82, .95, 1], [4, 16, 34, 50, 64, 78, 70, 30]); T = np.gradient(up, axis=0); T /= np.hypot(*T.T)[:, None]; N = np.c_[T[:, 1], -T[:, 0]]; top = up + N * (th + 14)[:, None]; bot = up + N * 10
        tip = A([(1926, 470), (1900, 500)]); cel.fill(np.vstack([top, tip, bot[::-1]]), LINE)
        K_ = 27
        for j in range(K_):                                        # clumps of two or three lashes, leaning out, longer to the temple
            u = .055 + .90 * (j / (K_ - 1)) ** .96 + r.normal(0, .004); i = int(np.clip(u, 0, 1) * (n - 1)); p = up[i] + N[i] * (th[i] * .50); ang = np.interp(u, [0, .2, .5, .75, 1], [-2.42, -2.10, -1.52, -.86, -.16]) + r.normal(0, .06); ln = np.interp(u, [0, .2, .5, .78, .92, 1], [90, 165, 225, 290, 260, 190]) * r.uniform(.80, 1.12); w = np.interp(u, [0, .3, .7, 1], [26, 44, 58, 46]) * r.uniform(.85, 1.1)
            for k_, (da, lk, wk) in enumerate(((0, 1., 1.), (.16, .80, .62), (-.15, .70, .50))[:int(r.integers(2, 4))]):
                c = ang + da; b1 = p + T[i] * da * 60; c1 = b1 + A([math.cos(c + .34), math.sin(c + .34)]) * ln * lk * .42; c2 = b1 + A([math.cos(c + .08), math.sin(c + .08)]) * ln * lk * .80; e = b1 + A([math.cos(c - .06), math.sin(c - .06)]) * ln * lk
                Q = spl(A([b1, c1, c2, e]), n=10); cel.fill(wedge(Q, w * wk, 0., .85), LINE)
                if k_ == 0 and r.random() < .55: cel.line(Q[int(len(Q) * .18):int(len(Q) * .62)] + [-w * .18, 0], 3.2, '#5a48a0', taper=(.5, .5), seed=j, alpha=.85)
        s.lidl = cel.rgba()
        # tear film: the meniscus along the lower lid
        cel = Cel(W, FH); men = lo + np.c_[0 * tl, -17 * np.sin(math.pi * np.clip(tl, .04, .96)) ** .4 - 5 * np.sin(tl * 26) ** 2 * np.sin(math.pi * tl)]; s.men = men
        cel.fill(np.vstack([lo + [0, 3], men[::-1]]), '#cfc0f4', .55); cel.line(men[int(.10 * m):int(.92 * m)], 4.2, '#ffffff', taper=(.25, .25), press=.5, seed=2, alpha=.95)
        for u0, u1, w in ((.24, .40, 11), (.46, .49, 8), (.60, .78, 13), (.82, .85, 7)): q = (lo * .45 + men * .55)[int(u0 * m):int(u1 * m)]; cel.line(q, w, '#ffffff', taper=(.4, .4), press=.4, seed=int(u0 * 100), alpha=.95)
        s.film = cel.rgba()
        # loose hair, out of focus, close to the lens
        cel = Cel(W, FH)
        for P, w in (([(1500, -80), (1600, 120), (1640, 330), (1612, 520)], 150), ([(1700, -80), (1800, 160), (1830, 420), (1780, 660)], 190), ([(250, -80), (190, 80), (160, 240), (190, 360)], 130), ([(60, -80), (20, 160), (10, 360), (60, 520)], 170)): l = Lock.d2(P, w, 'bang', m=30); cel.fill(l.poly, '#0a0714'); cel.fill(l.side(.05, .8, -.6, -.35), '#2c2150'); cel.fill(l.side(.1, .6, .2, .3), '#1a1436')
        h = cel.rgba(); s.hair = np.stack([blur(h[..., k], 9) for k in range(4)], -1)

    # ---------------------------------------------------------------- the world in the cornea
    def reflection(s, cx, cy, moon=1., f=0, mc=None, mr_px=None, clip=None, ghost=1., eyes=0.):
        """what he sees, bent by the cornea: moon, cloud bars, the tiny figure, the horizon and the road of light.  additive RGB"""
        x, y = (s.gx - cx) / RI, (s.gy - cy) / RI; mx, my, mr = (-.335, -.40, .262) if mc is None else ((mc[0] - cx) / RI, (mc[1] - cy) / RI, mr_px / RI); out = np.zeros((FH, W, 3), f32); r2 = np.sqrt(x * x + y * y); inside = smooth(1.0, .92, r2)
        hz = .20 + .20 * x * x; sky = smooth(hz + .004, hz - .004, y); out += (sky * smooth(-1., .2, y) * .016 * inside)[..., None] * rgb('#8a4cff')
        out += (np.exp(-np.abs(y - hz) / .010) * inside * .20)[..., None] * rgb('#b98cff')
        d = np.sqrt((x - mx) ** 2 + (y - my) ** 2); disc = smooth(mr + .006, mr - .006, d) * (1. if clip is None else ghost + (1 - ghost) * smooth(clip + 1.5, clip - 1.5, r2 * RI)); halo = np.exp(-np.clip(d - mr, 0, 9) / .075) * (1 - disc)
        bars = np.zeros_like(x)
        for j, (yy, h_, x0, x1) in enumerate(((.125, .017, -.30, .12), (.172, .010, -.10, .30), (.205, .012, -.22, .10))): bars = np.maximum(bars, smooth(h_, h_ * .55, np.abs(y - my - yy - .004 * np.sin((x - mx) * 14 + j))) * smooth(x0 - .05, x0 + .03, x - mx) * smooth(x1 + .05, x1 - .03, x - mx))
        out += (halo * .75 * moon * inside * (1. if clip is None else float(np.clip(mr_px / 110., .12, 1.))))[..., None] * rgb('#9a4cff')
        tone_ = rgb('#fdf9ff') * 1.14 * (1 - bars[..., None]) + rgb('#4a1c86') * bars[..., None]; out = out * (1 - disc[..., None]) + disc[..., None] * moon * tone_
        # the figure: head, shoulders, opened arms, one leg, tail
        if moon > 0:
            k = mr * RI; big = mc is not None; c = A([cx + mx * RI, cy + my * RI + (.06 if big else .05) * k]); fg = Cel(W, FH); sc = k * (.0178 if big else .0115); P = lambda L_: A(L_, float) * sc + c
            fg.fill(ell(*(c + A([0, -34]) * sc), 4.6 * sc, 6.0 * sc), '#000000'); fg.fill(spl(P([(-3, -28), (-9, -26), (-13, -22), (-20, -12), (-29, -5), (-38, -3), (-39, -.5), (-29, -1), (-18, -7), (-10, -15), (-7, -6), (-6, 4), (-5.5, 20), (-2, 44), (0, 50), (2, 44), (5.5, 20), (6, 4), (7, -6), (10, -15), (18, -7), (29, -1), (39, -.5), (38, -3), (29, -5), (20, -12), (13, -22), (9, -26), (3, -28)]), True, n=6), '#000000')
            fg.line(spl(P([(3, 6), (12, 14), (17, 28), (12, 40), (4, 46), (6, 54)]), n=6), 3.2 * sc, '#000000', taper=(.0, .7), press=0); k_ = fg.rgba()[..., 3:4] * disc[..., None]; out = out * (1 - k_) + rgb('#0c0418') * k_
            if eyes > 0 and k > 30:
                for q in (-1, 1): g_ = np.exp(-(((s.gx - c[0] - q * 2.1 * sc) / (1.9 * sc)) ** 2 + ((s.gy - c[1] + 33.2 * sc) / (1.0 * sc)) ** 2)); out = out + (g_ * eyes * disc)[..., None] * rgb('#ff3a44')
        # road of light on the sea
        rng = np.random.default_rng(11)
        for j in range(40):
            v = (j + .5) / 40; yy = .22 + .62 * v ** 1.1; w_ = (.05 + .20 * v) * rng.uniform(.4, 1.1); x0 = mx * (1 - .25 * v) + rng.normal(0, .05 * (.3 + v)); h_ = .0035 + .006 * v; tw = .6 + .4 * math.sin(f * .7 + j * 2.4)
            out += (np.exp(-((x - x0) / w_) ** 2 * 2.5) * np.exp(-((y - yy - .18 * x * x) / h_) ** 2) * inside * .55 * tw * (1 - .45 * v))[..., None] * rgb('#d9c4ff')
        return out, disc * moon

    def draw(s, pupil=208., f=0, gaze=(0., 0.), tear=1., moon=1., hair=1., tremble=0., mc=None, mr=None, ghost=1., lash=(0., 0.), eyes=0., clip=True):
        r = np.random.default_rng(1000 + f); j = r.normal(0, 1, 2) * tremble; cx, cy = CX + gaze[0] + j[0], CY + gaze[1] + j[1]
        ic, ia = s.iris(cx, cy, pupil); ball = s.white * (1 - ia[..., None]) + ic * ia[..., None]
        d = np.sqrt((s.gx - cx) ** 2 + (s.gy - cy) ** 2); ball = ball * (1 - .30 * (np.exp(-np.clip(d - RI, 0, 999) / 34.) * (1 - ia))[..., None] * rgb('#8a7aa8')[None, None] ** .2)          # limbus melts into the white
        ball = ball * (1 - .62 * s.lsh[..., None] * (1 - rgb('#6a55b0'))) * (1 - .38 * s.lsh[..., None] * (ia * .8)[..., None]); ball = ball * (1 - .42 * s.lsh2[..., None] * (1 - rgb('#7a66c0'))); ball = ball * (1 - .35 * s.losh[..., None] * (1 - rgb('#8a70c8')))
        R, da = s.reflection(cx, cy, moon, f, mc, mr, None if mc is None or not clip else pupil - 3, ghost, eyes); ball = ball * (1 - da[..., None]) + R
        # window of sky in the film, lower right, and the long soft gleam
        x, y = (s.gx - cx) / RI, (s.gy - cy) / RI; g = np.exp(-(((x - .44) * .8 + (y - .56) * .6) / .05) ** 2) * np.exp(-((-(x - .44) * .6 + (y - .56) * .8) / .20) ** 2); ball = ball + (np.clip(g, 0, 1) * .50)[..., None] * rgb('#cdb8ff')
        img = s.skin * (1 - s.om[..., None]) + ball * s.om[..., None]
        ll = s.lidl if lash == (0., 0.) else shifted(s.lidl, *lash); img = over(img, ll); fl = s.film * float(np.clip(tear, 0, 1)); img = img * (1 - fl[..., 3:4]) + fl[..., :3]
        if tear > .6: img = s.bead(img, tear)
        # sweat
        cel = Cel(W, FH)
        for px, py, k in ((196, 978, 33), (1716, 952, 24), (1812, 862, 13)): s.sweat(cel, px, py, k)
        img = over(img, cel.rgba())
        if hair > 0: img = over(img, (s.hair if lash == (0., 0.) else shifted(s.hair, -lash[0] * 2, lash[1])) * hair)
        # moonlight breathes out of the cornea
        g = radial(cx - .335 * RI, cy - .40 * RI, 240, power=3.2) if mc is None else radial(*mc, max(60., mr * 2.2), power=3.2); return img + (g * .07 * moon)[..., None] * rgb('#9a5cff')

    def sweat(s, cel, x, y, k):
        P = spl(A([(0, -1.5), (.62, -.35), (.80, .35), (0, 1.0), (-.80, .35), (-.62, -.35)]) * k + [x, y], True, corners=(0,), n=8); cel.fill(P + [4, 8], SK[3], .45); cel.fill(P, '#b9a0e0', .55); cel.clip(P); cel.fill(ell(x + .30 * k, y + .35 * k, k * .75, k * .6), '#e6d8ff', .55); cel.unclip(); cel.fill(ell(x - .28 * k, y - .10 * k, k * .17, k * .34, rot=.25), '#ffffff'); cel.line(np.vstack([P, P[:1]]), 2.4, mixc(LINE, SK[2], .3), taper=(.1, .1), seed=int(x))

    def bead(s, img, tear):
        """a tear gathering on the lower lid, ready to fall"""
        i = int(.70 * (len(s.lo) - 1)); x, y = s.lo[i]; q = s.lo[i + 6] - s.lo[i - 6]; rot = math.atan2(q[1], q[0]); k = 44 * tear; cel = Cel(W, FH); R = lambda L_: A(L_, float) @ A([[math.cos(rot), math.sin(rot)], [-math.sin(rot), math.cos(rot)]]) * k + [x, y + 4]
        P = spl(R([(-2.3, -.42), (-1.35, -.46), (-.9, -.15), (-.80, .42), (-.42, .98), (.12, 1.12), (.62, .86), (.86, .30), (.92, -.18), (1.4, -.44), (2.3, -.40), (1.0, -.62), (0, -.66), (-1.0, -.62)]), True, corners=(0, 10), n=8)
        cel.fill(P + [4, 12], SK[3], .42); cel.fill(P, '#b59ee2', .70); cel.clip(P); cel.fill(ell(*R([(.12, .62)])[0], k * .72, k * .52), '#efe6ff', .80); cel.fill(spl(R([(-.9, -.2), (0, -.02), (.95, -.2), (.7, .25), (0, .36), (-.7, .25)]), True, n=6), '#5c4396', .55); cel.unclip()
        cel.fill(ell(*R([(-.42, .20)])[0], k * .15, k * .30, rot=.3), '#ffffff'); cel.fill(ell(*R([(.46, .72)])[0], k * .09, k * .12), '#ffffff'); cel.line(P[int(len(P) * .10):int(len(P) * .72)], 2.8, mixc(LINE, '#4a3078', .35), taper=(.3, .3), seed=5); return over(img, cel.rgba())


if __name__ == '__main__':
    import sys
    E = Eye(); p = float(sys.argv[1]) if len(sys.argv) > 1 else 150
    img = E.draw(pupil=p); img = comp.finish(img, 0, **FIN); comp.save('qa/friends/eye.jpg', img, 93)
