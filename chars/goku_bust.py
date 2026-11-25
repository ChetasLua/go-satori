"""GOKU's neck, shoulders and gi.  Body points use head-front units (x right, y up, eye line 0, chin -1), turned about the neck by the body's own yaw/pitch/roll."""
import numpy as np, math
from db.ink import spl, resample, Cel
from db.kit import outline, hatch
from db import letters
from .goku_hair import mixc, _c
from .goku_head import lw, rad, lgrad
A = np.array


class Body:
    def __init__(s, H, byaw=None, bpitch=0., broll=0., raise_=0., breath=0., shift=(0, 0), scale=1.):
        s.H = H; s.s = H.s * scale; a = H.yaw * .45 if byaw is None else byaw; b = bpitch
        Ry = A([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]); Rx = A([[1, 0, 0], [0, math.cos(b), math.sin(b)], [0, -math.sin(b), math.cos(b)]]); c, n = math.cos(-broll), math.sin(-broll)
        s.R = A([[c, -n, 0], [n, c, 0], [0, 0, 1]]) @ Rx @ Ry; s.yaw = a; s.piv = A([0, -.80, -.16]); s.o = H.P3(s.piv[None])[0] + A(shift) * H.s; s.rz, s.br = raise_, breath

    def z(s, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float); d = np.clip((-1.5 - y) / 1.4, 0, 1)
        dep = .42 + .34 * d; z = dep * np.sqrt(np.clip(1 - (np.abs(x) / 2.55) ** 2.4, 0, 1)) - .16
        z = z + .20 * np.exp(-((y + 2.75) / .48) ** 2) * np.exp(-((np.abs(x) - .85) / .62) ** 2)          # pecs
        return z + .05 * s.br * d

    def P3(s, Q):
        Q = np.array(Q, float); Q[..., 1] = Q[..., 1] + s.rz * np.clip((np.abs(Q[..., 0]) - .5) / 1.2, 0, 1) * np.clip((-1.0 - Q[..., 1]) / .5, 0, 1)
        q = (Q - s.piv) @ s.R.T; f = 1 / (1 - np.clip(q[..., 2] * .06, -.4, .4)); return np.stack([s.o[0] + q[..., 0] * f * s.s, s.o[1] - q[..., 1] * f * s.s], -1)

    def P(s, xy, dz=0.):
        xy = np.array(xy, float); z = s.z(xy[..., 0], xy[..., 1]) + dz; return s.P3(np.concatenate([xy, z[..., None]], -1))


class Bust:
    def __init__(s, cel, G, L, byaw=None, bpitch=0., broll=0., raise_=0., breath=0., wear='gi', seed=0, lines=1., tendon=0., wet=0., dirt=0., shift=(0, 0), disc=True):
        s.c, s.G, s.L, s.seed, s.wear, s.tendon, s.wet, s.dirt, s.dsc = cel, G, L, int(seed), wear, tendon, wet, dirt, disc
        s.B = Body(G, byaw, bpitch, broll, raise_, breath, shift); s.u = G.s; s.w = lw(G.s) * lines; s.ink = L['line']; s.sg = -L['side']

    def skin(s, i): return s.L['skin'][i]
    def S(s, P, closed=False, corners=(), n=8, dz=0.): return spl(s.B.P(P, dz), closed, corners, n)

    def ln(s, P, w, col=None, taper=(.3, .4), seed=0, n=8, **k):
        P = np.asarray(P, float); P = s.B.P(P) if P.shape[-1] == 2 and np.abs(P).max() < 20 else P
        if 2 < len(P) <= 9: P = spl(P, n=n)
        s.c.line(P, max(.7, w * s.w), s.ink if col is None else col, taper=taper, seed=s.seed * 11 + seed, press=k.pop('press', .3), tremor=k.pop('tremor', .15) * min(1, s.u / 200), **k)

    def f(s, P, col, al=1., corners=None, sm=True):
        P = s.B.P(A(P, float)); s.c.fill(spl(P, True, corners=corners, n=6) if sm and corners is not None else P, col, al)

    # outline tables, half body (x>0): trapezius from under the ear, acromion, deltoid, arm
    TRAP = [(.60, -.62), (.60, -.95), (.64, -1.15), (.76, -1.30), (1.02, -1.46), (1.40, -1.60), (1.78, -1.72)]
    DELT = [(1.78, -1.72), (2.08, -1.78), (2.36, -1.98), (2.54, -2.34), (2.56, -2.74), (2.46, -3.10), (2.42, -3.50), (2.50, -3.95), (2.50, -4.6)]

    def neck(s):
        c, G, B = s.c, s.G, s.B; H = G; sg = s.sg
        # skin of neck + shoulders as one silhouette
        top = H.P3(A([(-.58, -.46, -.24), (.58, -.46, -.24)]))
        Lp = [top[0]] + list(B.P([(-x, y) for x, y in s.TRAP[1:]], -.10)); Rp = [top[1]] + list(B.P([(x, y) for x, y in s.TRAP[1:]], -.10))
        Ld = B.P([(-x, y) for x, y in s.DELT], -.1); Rd = B.P([(x, y) for x, y in s.DELT], -.1)
        l_ = spl(A(Lp), n=8); r_ = spl(A(Rp), n=8); ld, rd = spl(Ld, n=8), spl(Rd, n=8)
        s.sil = np.vstack([l_[::-1], r_, rd[1:], ld[::-1][:-1]]) if False else np.vstack([ld[::-1][:-1], l_[::-1], r_, rd[1:]])
        s.nl, s.nr, s.dl, s.dr = l_, r_, ld, rd
        c.fill(s.sil, s.skin(0))

    def neck_shade(s):
        c, G, B, L, sg = s.c, s.G, s.B, s.L, s.sg; am = L['amt']
        c.clip(s.sil); sh, dp = s.skin(1), s.skin(2)
        # side plane of the neck and shoulder away from the light
        if am > -.9:
            e = .10 + .30 * np.clip(am + .4, 0, 1.3)
            s.f([(sg * .80, -.4), (sg * (.60 - e), -.7), (sg * (.52 - e), -1.1), (sg * (.40 - e * .6), -1.62), (sg * (.60 - e * .3), -1.72), (sg * 1.1, -1.66), (sg * 1.6, -1.86), (sg * 2.1, -1.98), (sg * 2.02, -2.4), (sg * 2.16, -2.9), (sg * 2.06, -3.6), (sg * 3.2, -4.6), (sg * 3.2, -1.0)], sh)
        # cast shadow of the jaw on the neck: hard, with a point over the sternomastoid
        j = G.jaw(-1); k = G.jaw(1); dn = A([-math.sin(B.yaw * 0), 1.]); off = A([sg * .10, .30]) * s.u
        cast = np.vstack([j, k[::-1][1:]]); d = G.P3(A([(sg * .12, -1.36, .34), (sg * .30, -1.22, .26)])); mid = len(j)
        P = np.vstack([G.P3(A([(-.60, -.40, -.2)])), B.P([(-.60, -.98), (-.42, -1.18), (-.22, -1.12), (sg * .10, -1.36), (.20, -1.14), (.40, -1.20), (.60, -1.0)], .02), G.P3(A([(.60, -.40, -.2)]))]); c.fill(P, sh)
        c.fill(np.vstack([G.P3(A([(-.56, -.5, -.1)])), B.P([(-.50, -.92), (-.26, -1.04), (0, -1.10), (.26, -1.04), (.50, -.94)], .05), G.P3(A([(.56, -.5, -.1)]))]), dp, .75)
        # notch and clavicles
        s.f([(-.16, -1.74), (0, -1.66), (.16, -1.74), (.08, -1.92), (0, -1.98), (-.08, -1.92)], sh, 1, corners=(0, 2))
        for q in (-1, 1):
            s.f([(q * a, b) for a, b in [(.16, -1.80), (.60, -1.80), (1.10, -1.70), (1.60, -1.76), (1.15, -1.84), (.66, -1.94), (.24, -1.94)]], sh, .95, corners=(0, 3))
            s.f([(q * a, b) for a, b in [(.62, -1.30), (.86, -1.50), (1.30, -1.66), (.86, -1.66), (.60, -1.62), (.50, -1.45)]], sh, .8 if q == sg else .45, corners=(0, 2))
        if am > .3: s.f([(sg * a, b) for a, b in [(.9, -.5), (.62, -.8), (.58, -1.2), (.80, -1.44), (1.5, -1.66), (2.3, -1.9), (3, -1.2)]], dp, .55)
        c.unclip()

    def neck_lines(s, bare=False):
        G, B = s.G, s.B; t = s.tendon
        for k, e in ((0, s.nl), (1, s.nr)): s.c.line(e, s.w * .95, s.ink, taper=(.10, .12), press=.35, seed=s.seed + k, tremor=.15)
        if bare:
            for k, e in ((0, s.dl), (1, s.dr)): s.c.line(e, s.w, s.ink, taper=(.06, .10), press=.35, seed=s.seed + 5 + k, tremor=.15)
        if s.u < 40: return
        for q in (-1, 1):
            a = G.P3(A([(q * .56, -.62, -.08)]))[0]; P = np.vstack([a, B.P([(q * .47, -1.05), (q * .32, -1.45), (q * .13, -1.78)], .03)]); s.c.line(spl(P, n=10)[6:], s.w * (.60 + .2 * t), s.ink, taper=(.6, .25), seed=s.seed + 21 + q, alpha=.9)
            s.ln([(q * .14, -1.80), (q * .55, -1.82), (q * 1.05, -1.72), (q * 1.45, -1.74)], .55, seed=23 + q, taper=(.15, .6)); s.ln([(q * .34, -1.94), (q * .62, -1.93), (q * .95, -1.86)], .34, seed=25 + q, taper=(.5, .6))
            if t > .2:
                P = np.vstack([G.P3(A([(q * .24, -.90, .34)]))[0], B.P([(q * .22, -1.16), (q * .20, -1.48), (q * .10, -1.74)], .08)]); s.c.line(spl(P, n=10), s.w * .55 * min(1, t * 1.3), s.ink, taper=(.55, .4), seed=s.seed + 27 + q)
                P = np.vstack([G.P3(A([(q * .44, -.72, .10)]))[0], B.P([(q * .40, -1.10), (q * .50, -1.42)], .05)]); s.c.line(spl(P, n=10), s.w * .40, s.ink, taper=(.55, .6), seed=s.seed + 29 + q, alpha=t)
        s.ln([(-.05, -1.28), (.02, -1.36), (.07, -1.32)], .42, seed=31, taper=(.5, .5)); s.ln([(-.04, -1.84), (0, -1.90), (.05, -1.84)], .5, seed=33, taper=(.4, .4))

    def chest(s):
        """bare chest: pecs, deltoid split, serratus start"""
        c, sg, sh, dp = s.c, s.sg, s.skin(1), s.skin(2); am = s.L['amt']; c.clip(s.sil)
        for q in (-1, 1):
            d = q == sg
            s.f([(q * a, b) for a, b in [(.02, -2.3), (.07, -2.7), (.05, -3.1), (.30, -3.34), (.95, -3.46), (1.55, -3.22), (1.92, -2.82), (1.62, -3.34), (1.0, -3.66), (.30, -3.60), (.0, -3.5)]], sh, 1, corners=(0, 2, 6, 10))
            s.f([(q * a, b) for a, b in [(1.62, -1.94), (1.74, -2.3), (1.98, -2.78), (1.86, -2.84), (1.60, -2.50), (1.44, -2.10)]], sh, 1, corners=(0, 2))
            if d: s.f([(q * a, b) for a, b in [(.10, -2.06), (.50, -2.16), (.92, -2.12), (1.30, -2.36), (1.46, -2.72), (1.20, -2.62), (.92, -2.80), (1.02, -3.10), (.70, -3.40), (.30, -3.30), (.12, -2.7)]], sh, .9 if am > .2 else .35)
            s.f([(q * a, b) for a, b in [(2.0, -2.8), (2.12, -3.2), (2.04, -3.9), (2.2, -4.6), (2.6, -4.6), (2.34, -3.6), (2.3, -3.1)]], sh, .9, corners=(0,))
        c.unclip()
        for q in (-1, 1):
            s.ln([(0, -2.02), (.02, -2.5), (.0, -3.1), (.0, -3.5)], .55, seed=41, taper=(.5, .3)) if q > 0 else None; s.ln([(q * .20, -3.44), (q * .80, -3.56), (q * 1.45, -3.34), (q * 1.86, -2.90)], .8, seed=43 + q, taper=(.25, .3))
            s.ln([(q * 1.56, -1.92), (q * 1.70, -2.30), (q * 1.92, -2.78)], .60, seed=45 + q, taper=(.5, .25)); s.ln([(q * 1.98, -2.90), (q * 2.06, -3.3), (q * 2.02, -3.9)], .7, seed=47 + q, taper=(.2, .5))

    # ---- cloth
    def shirt(s):
        c, L, sg = s.c, s.L, s.sg; u0, u1, u2 = L['under']; wr = s.wear
        if wr == 'torn': return s.torn_shirt()
        # blue undershirt: round neck with a thick rib; sleeves over the deltoids
        neck = [(-.86, -1.36), (-.66, -1.62), (-.40, -1.88), (0, -2.02), (.40, -1.88), (.66, -1.62), (.86, -1.36)]
        s.f(neck + [(1.30, -1.52), (1.66, -1.64), (1.70, -2.6), (1.60, -4.6), (-1.60, -4.6), (-1.70, -2.6), (-1.66, -1.64), (-1.30, -1.52)], u0)
        s.f([(sg * .1, -2.04), (sg * .42, -1.92), (sg * .86, -1.40), (sg * 1.6, -1.6), (sg * 1.6, -4.6), (sg * .2, -4.6), (sg * .26, -3.0)], u1)
        rib = np.vstack([s.B.P(neck), s.B.P([(x * 1.11, y - .085) for x, y in neck][::-1])]); c.fill(rib, mixc(u0, u2, .25)); c.fill(np.vstack([s.B.P(neck[::sg][3:] if False else neck), s.B.P([(x * 1.04, y - .035) for x, y in neck][::-1])]), u2, .8)
        c.line(spl(s.B.P(neck), n=8), s.w * .85, s.ink, taper=(.1, .1), seed=s.seed + 51); c.line(spl(s.B.P([(x * 1.11, y - .085) for x, y in neck]), n=8), s.w * .45, s.ink, taper=(.2, .2), seed=s.seed + 52)
        # cast of the neck on the shirt
        for q in (-1, 1):
            torn = wr == 'torn' and q == 1
            D = [(q * 1.60, -1.62), (q * 1.80, -1.70), (q * 2.10, -1.76), (q * 2.40, -1.96), (q * 2.62, -2.34), (q * 2.66, -2.80), (q * 2.60, -3.14)]
            if torn:                                     # left sleeve torn off at the shoulder: a ragged edge
                E = [tuple(v) for v in spl(A([(1.60, -1.62), (1.80, -1.69), (2.06, -1.78), (2.04, -1.90), (2.12, -2.04), (1.98, -2.08), (1.90, -2.16), (1.98, -2.30), (2.00, -2.44), (1.86, -2.40), (1.80, -2.50), (1.76, -2.72), (1.70, -2.60), (1.60, -2.9)]), False, (2, 4, 6, 8, 10, 11, 12), 5)]; s.f(E, u0); s.f([(1.60, -1.9), (1.86, -2.0), (1.80, -2.66), (1.70, -2.60), (1.60, -2.9)], u1); outline(c, s.B.P(E), s.w * .7, s.ink, seed=s.seed + 53); continue
            sl = [(q * 2.56, -3.18), (q * 2.2, -3.30), (q * 1.8, -3.22), (q * 1.52, -3.0)]; P = np.vstack([spl(s.B.P(D, -.1), n=6), s.B.P(sl)]); c.fill(P, u0); c.clip(P)
            s.f([(q * a, b) for a, b in [(1.60, -1.9), (1.86, -2.3), (2.02, -2.9), (2.3, -3.26), (1.52, -3.05)]], u1)
            if q == sg: s.f([(q * a, b) for a, b in [(1.6, -1.8), (2.1, -2.0), (2.2, -2.5), (2.3, -3.3), (2.8, -3.2), (2.7, -2.0)]], u1)
            c.unclip(); c.line(spl(s.B.P(D, -.1), n=6), s.w, s.ink, taper=(.1, .1), press=.35, seed=s.seed + 55 + q); s.ln(sl, .8, seed=57 + q, taper=(.1, .2))
            for a, b_ in (((1.86, -1.92), (2.02, -2.5)), ((2.16, -1.96), (2.22, -2.36))): s.ln([(q * a[0], a[1]), (q * (a[0] + b_[0]) / 2 + q * .05, (a[1] + b_[1]) / 2), (q * b_[0], b_[1])], .45, seed=61 + int(a[0] * 9), taper=(.3, .6))

    def gi(s):
        c, L, sg, B = s.c, s.L, s.sg, s.B; g0, g1, g2 = L['gi']; torn = s.wear == 'torn'
        if torn: return s.torn_gi()
        for q in (-1, 1):
            tq = torn and q == 1
            # panel: shoulder strap -> V edge -> waist
            if tq: edge = [tuple(v) for v in spl(A([(.80, -1.40), (.96, -1.47), (1.10, -1.52), (1.04, -1.62), (1.00, -1.74), (1.10, -1.80), (1.26, -1.80), (1.20, -1.92), (1.06, -2.06), (1.14, -2.22), (1.22, -2.34), (1.42, -2.42), (1.36, -2.52), (1.30, -2.66), (1.44, -2.80), (1.62, -2.88), (1.56, -3.1), (1.50, -4.6)]), False, (2, 4, 6, 8, 11, 13, 15), 5)]
            else: edge = [(.80, -1.40), (1.30, -1.50), (1.74, -1.64), (1.80, -1.78), (1.64, -2.2), (1.62, -2.7), (1.74, -3.2), (1.72, -4.6)]
            vx = .05 if q < 0 else -.05
            P = [(q * x, y) for x, y in edge] + [(q * -.1, -4.6), (q * -.06, -3.16), (q * .30, -2.62), (q * .60, -1.96)]
            c.fill(B.P(P, .03), g0); c.clip(B.P(P, .03))
            # planes: side shadow, chest fold shadow, under-collar shadow
            if q == sg: s.f([(q * 1.30, -1.4), (q * 1.22, -2.1), (q * .92, -2.6), (q * 1.14, -3.1), (q * .82, -3.5), (q * .9, -4.6), (q * 2.4, -4.6), (q * 2.2, -1.4)], g1)
            else: s.f([(q * 1.70, -1.9), (q * 1.50, -2.4), (q * 1.46, -3.0), (q * 1.30, -3.6), (q * 1.36, -4.6), (q * 2.2, -4.6), (q * 2.0, -1.9)], g1)
            for k, F in enumerate(([(1.02, -1.52), (1.16, -1.86), (1.10, -2.24), (.98, -2.06), (.94, -1.72)], [(1.40, -1.60), (1.50, -1.92), (1.40, -2.3), (1.34, -1.98)], [(.74, -1.92), (.92, -2.26), (1.20, -2.48), (.86, -2.40)], [(1.62, -2.10), (1.30, -2.52), (.98, -2.70), (1.28, -2.66), (1.62, -2.44)], [(.62, -2.30), (.98, -2.62), (1.50, -2.86), (1.06, -2.80), (.70, -2.62)], [(.42, -2.80), (.78, -3.20), (1.36, -3.42), (.84, -3.40), (.46, -3.16)], [(.20, -3.3), (.50, -3.8), (1.1, -4.1), (.5, -4.1), (.12, -3.7)], [(1.50, -1.72), (1.42, -2.10), (1.20, -2.40), (1.36, -2.06)])): s.f([(q * a, b) for a, b in F], g1, 1, corners=(0, 2))
            s.f([(q * a, b) for a, b in [(.92, -1.46), (1.30, -1.56), (1.62, -1.66), (1.36, -1.74), (1.02, -1.64)]], g2, .9 if q != sg else .3, corners=(0, 2))
            c.unclip()
            outline(c, B.P([(q * x, y) for x, y in edge], .03), s.w * .85, s.ink, seed=s.seed + 71 + q, closed=False)
            for a, b_ in (((1.06, -1.54), (1.10, -2.16)), ((.80, -1.98), (1.14, -2.44)), ((1.60, -2.14), (1.06, -2.66)), ((1.46, -1.76), (1.24, -2.34)), ((1.70, -2.2), (1.30, -2.62)), ((.66, -2.46), (1.16, -2.76)), ((.50, -2.96), (1.0, -3.36))): s.ln([(q * a[0], a[1]), (q * ((a[0] + b_[0]) / 2 - .05), (a[1] + b_[1]) / 2 - .04), (q * b_[0], b_[1])], .46, seed=75 + int(a[0] * 10), taper=(.2, .6))
        # collar bands: thickness, his left over his right
        for q in ((-1, 1) if True else ()):
            if torn and q == 1: V = [(.80, -1.40), (.72, -1.72), (.52, -2.20), (.24, -2.72), (-.08, -3.18)]
            else: V = [(.80, -1.40), (.72, -1.72), (.52, -2.20), (.24, -2.72), (-.08, -3.18)]
            I = spl(B.P([(q * x, y) for x, y in V], .06), n=8); O = spl(B.P([(q * (x + .17 - .02 * i), y - .05) for i, (x, y) in enumerate(V)], .05), n=8); P = np.vstack([I, O[::-1]]); c.fill(P, g0 if q != sg else mixc(g0, g1, .55))
            M = spl(B.P([(q * (x + .045), y - .015) for x, y in V], .06), n=8); c.fill(np.vstack([I, M[::-1]]), g2 if q != sg else g0); c.fill(np.vstack([O, spl(B.P([(q * (x + .26), y - .10) for x, y in V], .03), n=8)[::-1]]), g1, .9)
            c.line(I, s.w * .95, s.ink, taper=(.08, .10), press=.3, seed=s.seed + 81 + q); c.line(O, s.w * .6, s.ink, taper=(.15, .2), press=.3, seed=s.seed + 83 + q)
        if s.dsc and not (torn and False):
            a = B.P([(.98, -2.72)], .05)[0]; ex = B.P([(1.30, -2.72)], .05)[0] - a; ey = B.P([(.98, -3.04)], .05)[0] - a
            T = lambda Q: a + np.asarray(Q)[:, :1] * ex * 1.0 + np.asarray(Q)[:, 1:2] * ey * 1.0
            letters.disc(c, '悟', T, white=mixc(L['disc'], L['skin'][1], .12), ink=s.ink, w=max(1., s.w * .45), seed=s.seed)

    # ---- after d6: left shoulder bare, the gi closed over the right (the costume of db/hero3 'torn' and 'ssj')
    RIP = [(.50, -2.30), (.60, -2.28), (.66, -2.42), (.80, -2.46), (.86, -2.64), (1.02, -2.66), (1.10, -2.84), (1.26, -2.86), (1.34, -3.06), (1.52, -3.14), (1.64, -3.36), (1.86, -3.46), (1.98, -3.66)]
    def rip(s): return [tuple(v) for v in spl(A(s.RIP), False, tuple(range(1, 12, 2)), 5)]

    def torn_shirt(s):
        c, L, sg, B = s.c, s.L, s.sg, s.B; u0, u1, u2 = L['under']
        D = [(-1.56, -1.60), (-1.80, -1.68), (-2.10, -1.74), (-2.40, -1.94), (-2.64, -2.32), (-2.70, -2.80), (-2.66, -3.16)]; sl = [(-2.62, -3.22), (-2.44, -3.30), (-2.30, -3.24), (-2.12, -3.36), (-1.90, -3.30), (-1.74, -3.34), (-1.52, -3.1)]; P = np.vstack([spl(B.P(D, -.1), n=6), B.P(sl)]); c.fill(P, u0); c.clip(P)
        s.f([(-1.5, -1.9), (-1.86, -2.2), (-2.02, -2.8), (-2.3, -3.4), (-1.4, -3.4)], u1)
        if sg < 0: s.f([(-1.9, -1.6), (-2.2, -2.0), (-2.28, -2.5), (-2.4, -3.5), (-3.0, -3.5), (-2.9, -1.8)], u1)
        else: s.f([(-2.1, -1.76), (-2.36, -1.96), (-2.5, -2.3), (-2.36, -2.3), (-2.16, -2.0)], u2, .6)
        c.unclip(); c.line(spl(B.P(D, -.1), n=6), s.w, s.ink, taper=(.1, .1), press=.35, seed=s.seed + 55); outline(c, B.P(sl), s.w * .7, s.ink, seed=s.seed + 57, closed=False); s.ln([(-2.24, -1.96), (-2.30, -2.2), (-2.28, -2.44)], .45, seed=61, taper=(.3, .6)); s.ln([(-1.98, -2.3), (-2.10, -2.7), (-2.06, -3.0)], .45, seed=62, taper=(.3, .6))
        T = [tuple(v) for v in spl(A([(-.52, -1.66), (-.40, -1.78), (-.28, -1.82), (-.18, -1.92), (-.02, -1.90), (.10, -1.96), (.20, -1.94), (.28, -2.06), (.38, -2.08), (.44, -2.22), (.56, -2.28), (.34, -2.86), (.06, -3.3), (-.4, -2.7)]), True, (0, 1, 3, 5, 7, 9, 10, 12), 4)]; s.f(T, u0); s.f([(.30, -2.06), (.56, -2.28), (.34, -2.86), (.06, -3.3), (-.06, -3.0), (.16, -2.6)], u1); s.f([(-.46, -1.72), (-.28, -1.86), (-.10, -1.94), (-.16, -2.02), (-.34, -1.96)], u2, .6); outline(c, B.P(T), s.w * .62, s.ink, seed=s.seed + 63)

    def torn_gi(s):
        c, L, sg, B = s.c, s.L, s.sg, s.B; g0, g1, g2 = L['gi']; R_ = s.rip()
        arm = [tuple(v) for v in spl(A([(-.80, -1.36), (-1.30, -1.47), (-1.76, -1.59), (-2.10, -1.70), (-2.16, -1.90), (-2.06, -2.10), (-1.98, -2.34), (-1.84, -2.42), (-1.80, -2.62), (-1.66, -2.74), (-1.70, -3.1), (-1.74, -3.5), (-1.72, -4.6)]), False, (3, 7, 9), 5)]
        col = [(-.80, -1.36), (-.70, -1.66), (-.46, -2.14), (-.18, -2.66), (.08, -3.10)]
        P = B.P(arm + [(2.0, -4.6)] + R_[::-1] + [(.08, -3.10)] + col[::-1], .03); c.fill(P, g0); c.clip(P)
        s.f([(-2.3, -2.0), (-1.90, -2.1), (-1.56, -2.5), (-1.34, -2.8), (-1.46, -3.3), (-1.16, -3.9), (-1.24, -4.6), (-2.4, -4.6)], g1)
        if sg < 0: s.f([(-1.44, -1.46), (-1.30, -2.0), (-1.02, -2.5), (-1.20, -3.0), (-.92, -3.5), (-1.0, -4.6), (-2.4, -4.6), (-2.4, -1.4)], g1)
        s.f([(q[0] + .02, q[1] - .02) for q in R_] + [(2.1, -4.0), (1.7, -3.62), (1.30, -3.26), (.92, -2.90), (.56, -2.60)], g1, .9)                              # the torn edge rolls and throws its shadow
        for F in ([(-1.06, -1.48), (-.92, -1.86), (-1.04, -2.26), (-1.18, -2.06), (-1.20, -1.70)], [(-1.62, -1.58), (-1.40, -1.92), (-1.16, -2.10), (-1.36, -2.10), (-1.62, -1.86)], [(-.72, -2.1), (-.90, -2.5), (-1.24, -2.74), (-.86, -2.70)], [(-.40, -2.8), (-.10, -3.36), (.40, -3.62), (-.10, -3.60), (-.44, -3.2)], [(.36, -2.96), (.80, -3.30), (1.30, -3.46), (.80, -3.52), (.34, -3.3)], [(.2, -3.8), (.8, -4.1), (1.5, -4.0), (.8, -4.3)], [(-.9, -3.5), (-.5, -4.0), (.0, -4.2), (-.6, -4.3)]): s.f(F, g1, 1, corners=(0, 2))
        s.f([(-.92, -1.43), (-1.32, -1.53), (-1.70, -1.62), (-1.40, -1.72), (-1.04, -1.62)], g2, .9 if sg > 0 else .35, corners=(0, 2)); s.f([(.30, -2.7), (.62, -2.72), (.98, -3.02), (1.34, -3.30), (.90, -3.22), (.46, -2.96)], g2, .8, corners=(0, 3))
        c.unclip(); outline(c, B.P(arm, .03), s.w * .85, s.ink, seed=s.seed + 71, closed=False); outline(c, B.P(R_, .03), s.w * .8, s.ink, seed=s.seed + 72, closed=False); c.line(B.P([(.50, -2.30), (.30, -2.74), (.08, -3.10)], .03), s.w * .8, s.ink, taper=(.1, .2), seed=s.seed + 73); c.line(B.P([(x - .03, y - .05) for x, y in R_], .03), s.w * .34, g2, taper=(.1, .1), seed=s.seed + 74, alpha=.8)
        for a, b_ in (((-1.06, -1.50), (-1.08, -2.16)), ((-1.56, -1.64), (-1.14, -2.16)), ((-.72, -2.16), (-1.18, -2.72)), ((-1.64, -2.76), (-1.04, -3.0)), ((.10, -3.14), (.64, -3.44)), ((-.16, -3.2), (.30, -3.66)), ((1.6, -3.5), (.9, -3.56))): s.ln([a, ((a[0] + b_[0]) / 2 - .04, (a[1] + b_[1]) / 2 - .05), b_], .46, seed=75 + int(abs(a[0]) * 10), taper=(.2, .6))
        I = spl(B.P(col, .06), n=8); O = spl(B.P([(x - .24 + .03 * i, y - .07) for i, (x, y) in enumerate(col)], .05), n=8); Q = np.vstack([I, O[::-1]]); c.fill(Q, g0 if sg > 0 else mixc(g0, g1, .45)); M = spl(B.P([(x - .06, y - .02) for x, y in col], .06), n=8); c.fill(np.vstack([I, M[::-1]]), g2 if sg > 0 else g0); c.line(I, s.w * .95, s.ink, taper=(.08, .10), press=.3, seed=s.seed + 81); c.line(O, s.w * .6, s.ink, taper=(.15, .2), press=.3, seed=s.seed + 83); s.threads(R_[::3], 22, 4)

    def threads(s, P, n=14, seed=0):
        r = np.random.default_rng(seed + 3); Q = s.B.P(P, .03)
        for j in range(n):
            i = int(r.integers(len(Q) - 1)); u = r.random(); p = Q[i] * (1 - u) + Q[i + 1] * u; a = r.uniform(.2, 2.6); ln = s.u * r.uniform(.03, .10); s.c.line(spl([p, p + A([math.cos(a), math.sin(a)]) * ln * .5 + r.normal(0, ln * .12, 2), p + A([math.cos(a), math.sin(a) + .5]) * ln], n=6), max(.7, s.w * .28), s.ink if r.random() < .5 else s.L['gi'][1], taper=(.05, .7), seed=j)

    def paint(s):
        s.neck(); s.neck_shade()
        if s.wear in ('gi', 'torn'): s.neck_lines(); s.shirt(); s.gi(); (s.arm_bare() if s.wear == 'torn' else None)
        else: s.chest(); s.neck_lines(True)

    def arm_bare(s):
        s.c.line(s.dr, s.w, s.ink, taper=(.06, .1), press=.35, seed=s.seed + 6); s.ln([(1.56, -1.94), (1.70, -2.30), (1.90, -2.80)], .58, seed=47, taper=(.5, .25)); s.ln([(1.94, -2.92), (2.02, -3.3), (1.98, -3.9)], .6, seed=48, taper=(.2, .5)); s.ln([(1.00, -2.58), (1.40, -2.74), (1.72, -2.78)], .42, seed=49, taper=(.5, .5))
