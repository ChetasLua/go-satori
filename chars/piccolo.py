"""PICCOLO.
    from chars.piccolo import Piccolo
    Piccolo.head(cel, H, info)     head_fn: info = dict(expr, light, f, wind, seed, turban=True, wet=0, droop=0..1, part='all'|'back'|'front')
    expr: neutral stern shut oneeye startled drenched gentle shield shout
"""
from chars.fkit import *
from chars import fkit as K
from db import hands as HD

SKIN = ('#8bbb5a', '#4e8550', '#2a563e')
PINK = ('#f0a08a', '#c06a68', '#8a4450'); PINKL = '#9c3a44'
GI = ('#5b3f9c', '#3a2572', '#241552'); GIH = '#7a5cc0'
OBI = ('#4f9cc2', '#31719e', '#1f4a74')
CAPE = ('#f6f4f4', '#b9b4cc', '#8079a4'); CAPEH = '#ffffff'
WRAP = ('#fbf8f2', '#c2bcd2', '#8c84aa'); DOME = ('#7a4fb4', '#4c2d86', '#301a60'); DOMEH = '#b48ae0'
BAND = ('#b8323c', '#7c1c2c', '#54101e'); SHOE = ('#a5682f', '#6b3f1c', '#452510')

D = dict(
    skin=SKIN, blush0=0., side=.0, nose='adult', nshadow=1.0, my=-.660, ears='namek', fang=1, mouthc=('#4a1830', '#b0507a', '#280a1c'),
    head=dict(a=[(-1.0, .155), (-.965, .245), (-.88, .375), (-.76, .480), (-.62, .555), (-.46, .600), (-.28, .635), (-.12, .660), (.10, .69), (.35, .715), (.6, .69), (.8, .56), (.93, .35), (1.0, .05)],
              p=2.3, nose=(.10, -.06, -.42, -.475, .175), ear=(.655, .06, -.40, -.06), cheek=.05, socket=.06),
    eye=dict(xi=.158, xo=.492, U=[-.040, .016, .044, .060, .044], Lo=[-.056, -.074, -.078, -.060, -.004], ir=(.052, .066), ic=.46, iy=-.006, iris='black', g2=0,
             lt=[(0, .006), (.18, .022), (.5, .030), (.82, .038), (1, .030)], down=.45, lo0=.40, low=.75, ls=.022, crease=0, corners=(0, 4), cl=.8, round=.4),
    brow=dict(kind='ridge', pts=[(.066, -.010), (.122, .050), (.240, .118), (.412, .182), (.545, .172), (.612, .096)], th=[(0, .006), (.12, .026), (.45, .034), (.75, .026), (1, .004)]),
    lines=[([(.030, .250), (.026, .190), (.038, .140)], .8), ([(.215, -.130), (.300, -.170), (.385, -.175)], .5), ([(.46, -.28), (.40, -.40), (.385, -.52)], .65), ([(.060, .36), (.20, .395), (.34, .375)], .5)])

M = lambda h=.115, u=0.: dict(hw=h, mU=[u, .002, .003, .002, u], mL=[u, .002, .003, .002, u])
E = dict(
    neutral=X(**M(.110, -.010), mw=1.1), stern=X(aB=-.0, dB=-.012, **M(.120, -.018), mw=1.2, furrow=.8),
    shut=X(closed=1, arc=0, dB=-.012, **M(.100, -.006)),
    oneeye=X(closed=1, arc=0, oneeye=1, dB=.014, **M(.095, .0), gy=-.012),
    startled=X(U=[.01, .090, .115, .105, .045], Lo=[-.07, -.100, -.108, -.095, -.04], ir=.72, dB=.065, hw=.075, my=-.675, mU=[-.01, .030, .036, .030, -.01], mL=[-.01, -.050, -.060, -.050, -.01], sweat=.4),
    drenched=X(U=[-.030, .006, .016, .010, -.004], Lo=[-.050, -.062, -.066, -.060, -.024], dB=-.004, gy=-.02, ir=.95, **M(.120, -.014), mw=1.2, under=.9),
    gentle=X(U=[-.026, .020, .036, .036, .014], aB=-.10, dB=.030, gy=-.014, hw=.125, mU=[.012, -.003, -.006, .0, .026], mL=[.012, -.004, -.008, -.001, .026], mw=1.05),
    shield=X(U=[-.030, .040, .070, .075, .040], ir=.80, dB=-.020, furrow=1.4, hw=.165, my=-.660, mU=[.010, .040, .034, .040, .016], mL=[-.020, -.062, -.070, -.062, -.016], clench=1, naso=1., sweat=.5, lip=1),
    shout=X(U=[-.020, .055, .085, .085, .045], ir=.75, dB=-.012, furrow=1.4, hw=.190, my=-.610, mU=[.016, .046, .036, .046, .020], mL=[-.040, -.230, -.285, -.230, -.036], teeth=.045, loteeth=.03, fang=1., tongue=.7, jaw=.11, naso=1., lip=0))


def ex(e):
    if isinstance(e, str): return E[e]
    if isinstance(e, (tuple, list)): return blend(ex(e[0]), ex(e[1]), e[2])
    return X(**e) if set(e) <= set(X0) else e


TW = [(.36, .76, .80, -.06), (.30, .84, .90, -.06), (.36, .93, 1.0, -.06), (.56, 1.0, 1.07, -.07), (.78, .985, 1.04, -.08), (.94, .88, .92, -.08), (1.00, .74, .78, -.08)]
TD = [(.92, .78, .82, -.08), (1.06, .76, .79, -.08), (1.22, .66, .68, -.09), (1.36, .48, .49, -.10), (1.44, .24, .25, -.10), (1.46, .02, .02, -.10)]


def turban(cel, H, L, seed=0, droop=0., wet=0.):
    w = lw(H.s); wr = cloth(L, WRAP if wet < .5 else tuple(mixc(c, '#b8c4d0', .35) for c in WRAP), '#ffffff'); dm = cloth(L, DOME, DOMEH); sg = -1 if L['ld'][0] > 0 else 1; a0 = -H.yaw
    sag = lambda th, v: 0. * th
    k = droop; prof_ = [(y - k * (.10 + .10 * (y - .3)), a * (1 + .06 * k * (y < .7)), b * (1 + .06 * k * (y < .7)), c) for y, a, b, c in TW]; pd = [(y - k * (.20 + .30 * (y - .92)), a, b, c) for y, a, b, c in TD]
    R = Rev(H, prof_, tilt=-.17); Dm = Rev(H, pd, tilt=-.06)
    S = Dm.sil(); outline(cel, S, w[0] * 1.6, dm[4], seed + 2, tremor=.12); cel.fill(S, dm[0]); cel.clip(S); v = np.linspace(0, 1, 9)
    cel.fill(np.vstack([Dm.patch(a0 + sg * (.35 + .12 * np.sin(v * 8)), v), Dm.patch(a0 + sg * 1.62, v[::-1])]), dm[1]); cel.fill(np.vstack([Dm.patch(a0 - sg * (.30 + .15 * v), .25 + v * .5), Dm.patch(a0 - sg * (.75 - .1 * v), (.25 + v * .5)[::-1])]), dm[3])
    for t in (a0 - .75, a0 + .05, a0 + .85):
        for q in Dm.curve(np.full(20, t), np.linspace(.05, .9, 20)): cel.line(q, w[2] * 1.1, dm[4], taper=(.05, .7), seed=seed + 5)
    cel.unclip()
    S = R.sil(every=True, span=1.75)
    for q in S: outline(cel, q, w[0] * 1.6, wr[4], seed + 1, tremor=.15)
    cel.fill(S, wr[0], evenodd=True); clip_eo(cel, S)
    v = np.linspace(0, 1, 13); cel.fill(np.vstack([R.patch(a0 + sg * (.48 + .13 * np.sin(v * 13)), v), R.patch(a0 + sg * 1.68, v[::-1])]), wr[1])
    # wraps: bands that cross at the front, each with a thin shadow under its edge
    for j, (t0, v0, t1, v1, bw) in enumerate(((-2.0, .12, 2.0, .60, .09), (-2.0, .34, 2.0, .86, .10), (2.0, .20, -2.0, .52, .08), (2.0, .55, -2.0, .92, .08), (-2.0, .66, .4, .97, .06))):
        u = np.linspace(0, 1, 60); th = t0 + (t1 - t0) * u; vv = v0 + (v1 - v0) * (u + .10 * np.sin(u * math.pi * 2 + j)) + k * .05 * np.sin(u * math.pi)
        for q, q2 in zip(R.curve(th, vv), R.curve(th, vv - bw * np.sin(u * math.pi) ** .5)):
            n = min(len(q), len(q2)); cel.fill(np.vstack([q[:n], q2[:n][::-1]]), wr[1], .85); cel.line(q, w[1], wr[4], taper=(.10, .12), press=.3, seed=seed + 11 + j)
    for j in range(9):
        t = a0 - 1.2 + j * .3; v0 = .15 + .09 * (j * 7 % 8)
        for q in R.curve(t + np.linspace(0, .22, 10), v0 + np.linspace(0, .10, 10)): cel.line(q, w[2] * 1.1, wr[4], taper=(.1, .75), seed=seed + 30 + j)
    cel.unclip()
    if k > .25:                                                    # soaked: one wrap has come loose and hangs over an eye and down the cheek, dripping
        q = -sg; e_ = .22 * k; P3 = lambda L_: H.P3(A([(q * x, y - dy * k, z) for x, y, z, dy in L_]))
        C = spl(P3([(.02, .60, 1.0, 0), (.30, .64, .98, 0), (.60, .56, .86, .02), (.72, .34, .76, .06), (.71, .02, .72, .16), (.66, -.26, .70, .26), (.67, -.50, .68, .36), (.56, -.46, .74, .38), (.47, -.56, .78, .36), (.40, -.42, .82, .34), (.36, -.10, .88, .20), (.33, .14, .93, .08), (.20, .30, .99, .02), (.04, .38, 1.02, 0)]), True, corners=(0, 6, 7, 8, 9, 13), n=7)
        outline(cel, C, w[0] * 1.4, wr[4], seed + 7); cel.fill(C, wr[0]); cel.clip(C)
        cel.fill(spl(P3([(.50, .60, .9, 0), (.56, .30, .85, .05), (.50, .05, .85, .15), (.58, -.2, .78, .25), (.52, -.7, .78, .3), (.9, -.7, .7, .3), (.9, .5, .7, 0)]), True, corners=range(7), n=5), wr[1])
        for L_, ww in (([(.22, .52, 1.0, 0), (.44, .42, .94, .03), (.56, .10, .84, .14), (.55, -.30, .78, .3)], 1.1), ([(.10, .46, 1.02, 0), (.32, .32, .96, .03), (.42, .02, .88, .15), (.42, -.28, .82, .3)], 1.0), ([(.48, .56, .92, 0), (.64, .36, .82, .05), (.66, .06, .75, .15)], .8)): cel.line(spl(P3(L_), n=8), w[1] * ww, wr[4], taper=(.15, .5), press=.3, seed=seed + 8)
        cel.unclip()
        for x, y, dy in ((.67, -.58, .36), (.47, -.66, .36)):
            p = P3([(x, y, .72, dy)])[0]; r_ = .045 * H.s; D_ = spl(A([(0, -1.8), (.6, -.3), (.7, .35), (0, 1.), (-.7, .35), (-.6, -.3)]) * r_ + p, True, corners=(0,), n=5); cel.fill(D_, '#d6ecf6', .9); cel.line(np.vstack([D_, D_[:1]]), max(.7, w[2]), mixc(wr[4], '#3a5a86', .5), taper=(.1, .1))


def antenna(cel, H, L, seed=0, sway=0.):
    w = lw(H.s); cols = cloth(L, SKIN)
    for q in (1 if H.yaw > 0 else -1, -1 if H.yaw > 0 else 1):
        z = float(H.z(.15, .60)); C = H.P3(A([(q * .150, .58, z - .04), (q * .19, .86, z + .10), (q * (.25 + sway), 1.02, z + .34), (q * (.31 + sway * 1.5), .98, z + .56), (q * (.34 + sway * 2), .84, z + .70)])); k = H.s
        P, a, b, c = limb(C, [.050 * k, .042 * k, .036 * k, .032 * k, .040 * k]); tip = ell(C[-1, 0], C[-1, 1], .048 * k)
        outline(cel, P, w[0] * 1.5, cols[4], seed + 1); outline(cel, tip, w[0] * 1.4, cols[4], seed + 2); cel.fill(P, cols[0]); cel.fill(tip, cols[0]); n = len(c); cel.fill(np.vstack([c[: n], b[::-1]]), cols[1], .9)
        for t in (.22, .30, .38): i = int(t * n); cel.line(np.vstack([a[i], b[i]]), w[2], cols[4], taper=(.2, .2))


class _Piccolo:
    D = D; skin = SKIN
    def make(s, cx, cy, sc, yaw=0., pitch=0., roll=0., persp=.10): return FHead(D).pose(cx, cy, sc, yaw, pitch, roll, persp)

    def neck(s, cel, H, L, seed=0, w0=.40):
        w = lw(H.s); c = cloth(L, SKIN); sg = -1 if L['ld'][0] > 0 else 1; a, b = [H.P3(A([(q * (w0 + .03), -.45, -.16), (q * w0, -.85, -.12), (q * (w0 + .0), -1.15, -.10), (q * (w0 + .10), -1.50, -.10), (q * (w0 + .45), -1.85, -.12)])) for q in (-1, 1)]
        P = np.vstack([spl(a, n=8), spl(b, n=8)[::-1]]); cel.fill(P, c[0]); cel.clip(P); J = np.vstack([H.jaw(-1), H.jaw(1)[::-1][1:]]) + A([sg * .06, .26]) * H.s; cel.fill(np.vstack([J, H.P3(A([(1.2, .3, 0), (-1.2, .3, 0)]))]), c[1]); e = spl(a if sg < 0 else b, n=8); cel.fill(np.vstack([e, (e + A([-sg * .16, 0]) * H.s)[::-1]]), c[1]); cel.unclip()
        for i, q in enumerate((a, b)): cel.line(spl(q, n=8), w[0] * .9, c[4], taper=(.25, .1), seed=seed + i)
        for q in (-1, 1): cel.line(spl(H.P3(A([(q * .33, -.80, .08), (q * .22, -1.12, .16), (q * .08, -1.48, .22)])), n=8), w[1], c[4], taper=(.4, .4), seed=seed + 4 + q)

    def head(s, cel, H, info=None, **k):
        i = dict(expr='neutral', light='day', f=0, wind=(0, 0), seed=1, turban=True, part='all', wet=0., droop=0.); i.update(info or {}); i.update(k)
        x = ex(i['expr']); L = light(i['light']); H = FHead.of(H, D, x['jaw']); w = lw(H.s)
        if i['part'] == 'back': return H
        F = Face(cel, H, D, x, L, i['seed'], i['wet']); sk = cloth(L, SKIN)
        F.ears_far()
        if not i['turban']:                                         # bald dome
            S = H.skull(); outline(cel, S, w[0] * 1.6, L['line'], i['seed'] + 9, tremor=.12); cel.fill(S, sk[0]); cel.clip(S); sg = F.sg
            sf = lambda th, y: H.P3(surf(H, th, y)[0][None])[0]
            cel.fill(np.vstack([[sf(sg * (.75 + .08 * math.sin(y * 9)), y) for y in np.linspace(-.1, .97, 9)], [A([H.cx + sg * 3 * H.s, H.cy - 2 * H.s]), A([H.cx + sg * 3 * H.s, H.cy + H.s])]]), sk[1])
            cel.fill(spl(A([sf(-sg * .95, .62), sf(-sg * .75, .78), sf(-sg * .42, .90), sf(-sg * .50, .80), sf(-sg * .78, .66)]), True, n=6), np.clip(sk[0] * 1.1 + .08, 0, 1)); cel.unclip()
        F.base(); F.features(); F.contour(); F.ears_near()
        if i['turban']: turban(cel, H, L, i['seed'], i['droop'], i['wet'])
        else: antenna(cel, H, L, i['seed'], .03 * math.sin(i['f'] * .5))
        return H

    # ------------------------------------------------------------------ cape, torso, arms
    def cape_back(s, cel, B, L, w, seed=0, spread=1., sway=0., length=4.6, torn=0., wet=0.):
        """the cape behind the body: inner side in shade"""
        cp = cloth(L, CAPE, CAPEH); k = spread; r = np.random.default_rng(seed)
        P = [(-.9, -.35), (-1.9 * k, .10), (-2.15 * k + sway * .3, 1.2), (-2.25 * k + sway * .7, 2.5), (-2.45 * k + sway, length * .86), (-2.30 * k + sway * 1.1, length), (-1.5 * k + sway, length * 1.02), (-.6 * k + sway * .9, length * .97), (.35 * k + sway * .9, length * 1.03), (1.3 * k + sway * .9, length * .98), (2.2 * k + sway * .9, length * 1.0), (2.45 * k + sway * .7, length * .84), (2.2 * k + sway * .5, 2.5), (2.1 * k, 1.2), (1.9 * k, .10), (.9, -.35)]
        if torn > 0:
            lo = [q for q in P if q[1] > length * .8]; hi = [q for q in P if q[1] <= length * .8]; n = 15; xs = np.linspace(-2.4 * k, 2.35 * k, n); rag = [(x + sway, length * (.72 + .30 * (j % 2) - .12 * r.random()) * (1 - .18 * torn * r.random())) for j, x in enumerate(xs)]; m = len([q for q in hi if q[0] < 0]); P = hi[:m] + rag + hi[m:]
            C = B(P, z=-.6); C = spl(C, True, corners=range(m, m + n), n=6)
        else: C = B.c(P, corners=(0, len(P) - 1), z=-.6)
        outline(cel, C, w[0] * 1.6, cp[4], seed, tremor=.15); cel.fill(C, mixc(cp[1], cp[2], .30)); cel.clip(C)
        for j, x in enumerate((-1.75, -1.1, -.35, .45, 1.15, 1.8)):
            q = spl(B([(x * .75 * k, .2), (x * .92 * k + sway * .3, 1.6), (x * k + sway * .7 + .08 * (-1) ** j, 3.2), (x * 1.05 * k + sway, length * 1.1)], z=-.6), n=8); T = wedge(q[::-1], .34 * B.u, .02 * B.u, .6); cel.fill(T + A([.10 * B.u * (1 if j % 2 else -1), 0]), cp[2]); cel.line(q, w[1], cp[4], taper=(.5, .1), seed=seed + j)
        cel.unclip(); return C

    def yoke(s, cel, B, L, w, seed=0, lift=0., wet=0., torn=0.):
        """shoulder pads and crossed collar"""
        cp = cloth(L, CAPE, CAPEH); sg = -1 if L['ld'][0] > 0 else 1
        for q in ((1, -1) if B.yaw < 0 else (-1, 1)):
            l = lift; P = [(q * .30, -.42), (q * .62, -.60), (q * 1.05, -.66 - l * .3), (q * 1.65, -.60 - l * .7), (q * 2.30, -.40 - l), (q * 2.86, -.06 - l), (q * 3.02, .16 - l), (q * 2.86, .30 - l * .9), (q * 2.35, .34 - l * .7), (q * 1.95, .42 - l * .4), (q * 1.62, .60), (q * 1.36, .92), (q * 1.05, 1.02), (q * .55, 1.06), (q * .02, 1.16), (q * .02, .4)]
            z = A([.1, .0, -.05, -.10, -.12, -.15, -.1, .05, .18, .28, .40, .50, .58, .66, .70, .5]); C = spl(B(np.c_[A(P), z]), True, corners=(0, 6, 11, 14, 15), n=8)
            if torn > 0:                                           # the blast has eaten the hem: long rags, one pad half gone
                rr = np.random.default_rng(int(seed) * 3 + (q > 0) + 11); Pt = P[:5 if q != sg else 4] + [(q * (2.55 if q != sg else 2.05), -.22 - l)]; xs = np.linspace(2.45 if q != sg else 2.0, .20, 13)
                for j, x_ in enumerate(xs): base_ = float(np.interp(x_, [.2, .6, 1.05, 1.4, 1.65, 2.0, 2.5], [1.14, 1.06, .98, .82, .56, .38, .26])); Pt.append((q * (x_ + .05 * (-1) ** j), base_ - (.02 if j % 2 else .40) * (.5 + rr.random()) * torn * (1 if x_ > .5 else .4) + (.16 * rr.random() if j % 2 else 0)))
                Pt += [(q * .02, 1.16), (q * .02, .4)]; zz = np.interp(np.abs(A(Pt)[:, 0]), [0, .6, 1.3, 2., 3.], [.66, .6, .45, .15, -.1]); n5 = 5 if q != sg else 4; C = spl(B(np.c_[A(Pt), zz]), True, corners=[0] + list(range(n5, len(Pt))), n=5)
            outline(cel, C, w[0] * 1.7, cp[4], seed + 5 + q, tremor=.15); cel.fill(C, cp[0]); cel.clip(C); V = lambda L_: spl(B([(q * x, y - l * max(0, x - .8) / 2.2, zz) for x, y, zz in L_]), n=8)
            if q == sg: cel.fill(V([(.9, -.8, 0), (1.5, -.30, .1), (1.2, .10, .3), (1.7, .05, .2), (2.3, -.05, 0), (3.1, .0, 0), (3.2, .6, 0), (1.6, .8, .3), (1.2, 1.3, .5), (.4, 1.3, .7), (.75, .70, .6), (.95, .25, .4)]), cp[1])
            cel.fill(V([(3.1, .2, 0), (2.5, .12, .1), (1.9, .18, .25), (1.45, .42, .4), (1.12, .74, .5), (1.0, 1.1, .6), (1.6, 1.0, .5), (2.2, .6, .2), (3.0, .5, 0)]), cp[1])
            cel.fill(V([(.6, .05, .4), (.85, .42, .5), (.95, .80, .55), (.75, 1.10, .6), (.62, .75, .6), (.55, .4, .5)]), cp[1])
            for j, (pts, ww) in enumerate((([(.70, -.20, .2), (1.15, -.06, .3), (1.55, .16, .3), (1.80, .44, .3)], 1.), ([(.62, -.02, .3), (1.0, .26, .45), (1.22, .60, .5), (1.28, .92, .5)], 1.), ([(.90, -.42, .1), (1.5, -.36, .1), (2.1, -.20, .05), (2.6, .06, .0)], .8), ([(.50, .25, .5), (.72, .60, .58), (.74, 1.0, .6)], .8), ([(2.5, .30, .1), (2.36, .12, .1), (2.10, .0, .1)], .7), ([(1.9, .44, .25), (1.72, .24, .3), (1.50, .10, .3)], .7))):
                cel.line(V(pts), w[1] * ww, cp[4], taper=(.1, .6), press=.3, seed=seed + 11 + j)
            cel.unclip()
        # crossed collar: a thick roll round the neck
        for q in (-sg, sg):
            P = [(q * .58, -.74, -.10), (q * .78, -.46, .10), (q * .66, -.02, .40), (q * .34, .34, .60), (-q * .16, .62, .70), (-q * .30, .46, .70), (-q * .10, .22, .66), (q * .18, -.10, .50), (q * .32, -.44, .25), (q * .40, -.70, .0)]
            C = spl(B(P), True, corners=(0, 4, 5, 9), n=8); outline(cel, C, w[0] * 1.5, cp[4], seed + 8 + q, tremor=.15); cel.fill(C, cp[0]); cel.clip(C); V = lambda L_: spl(B([(q * x, y, zz) for x, y, zz in L_]), n=8)
            cel.fill(V([(.46, -.80, 0), (.50, -.40, .2), (.30, .02, .5), (.0, .30, .65), (-.30, .60, .7), (-.4, .3, .7), (0, 0, .6), (.2, -.6, .2)]), cp[1])
            if q == sg: cel.fill(V([(.90, -.6, 0), (.74, -.10, .3), (.50, .18, .5), (.1, .62, .7), (.4, .6, .6), (1.0, 0, .2)]), cp[1])
            for j, pts in enumerate(([(.62, -.58, .0), (.58, -.22, .25), (.40, .10, .5), (.10, .34, .65)], [(.72, -.30, .15), (.60, .02, .4), (.36, .26, .6)])): cel.line(V(pts), w[1] * .9, cp[4], taper=(.1, .6), press=.3, seed=seed + 21 + j)
            cel.unclip()

    def torso(s, cel, B, L, w, seed=0, bottom=3.2, obi=True):
        g = cloth(L, GI, GIH); o = cloth(L, OBI); sg = -1 if L['ld'][0] > 0 else 1
        R_ = [(.6, -.1), (1.15, .25), (1.42, .70), (1.34, 1.25), (1.12, 1.85), (1.00, 2.35), (1.04, 2.75), (1.25, bottom)]; P = np.vstack([B([(-x, y) for x, y in R_], bow=(1.6, .62))[::-1][::-1], B(R_, bow=(1.6, .62))[::-1]]); C = spl(P, True, corners=(0, len(R_) - 1, len(R_), len(P) - 1), n=8)
        outline(cel, C, w[0] * 1.6, g[4], seed, tremor=.12); cel.fill(C, g[0]); cel.clip(C); V = lambda L_: B.c(L_, False, bow=(1.6, .62))
        cel.fill(B.c([(sg * .30, .3), (sg * .55, 1.0), (sg * .34, 1.5), (sg * .62, 1.9), (sg * .40, 2.4), (sg * .7, bottom + .2), (sg * 2, bottom + .2), (sg * 2, .2)], corners=(0, 1, 2, 3, 4, 5, 6), bow=(1.6, .62)), g[1])
        for q in (-1, 1): cel.fill(B.c([(q * 1.38, .9), (q * .95, 1.18), (q * .40, 1.30), (q * .30, 1.22), (q * .85, 1.06)], corners=(0, 2, 3), bow=(1.6, .62)), g[1])
        for j, pts in enumerate(([(-.9, .5), (-.5, 1.0), (-.12, 1.5), (.0, 2.0)], [(.9, .5), (.5, 1.1), (.16, 1.7), (.02, 2.2)], [(-1.2, 1.5), (-.8, 1.75), (-.5, 2.1)], [(1.2, 1.5), (.85, 1.8), (.55, 2.15)], [(-.9, 2.0), (-.6, 2.2), (-.4, 2.4)])): cel.line(V(pts), w[1], g[4], taper=(.1, .6), press=.3, seed=seed + j)
        if obi:
            O = np.vstack([V([(-1.3, 2.30), (-.6, 2.42), (0, 2.46), (.6, 2.42), (1.3, 2.30)]), V([(1.3, 2.78), (.6, 2.92), (0, 2.96), (-.6, 2.92), (-1.3, 2.78)])]); cel.fill(O, o[0]); cel.clip(O); cel.fill(B.c([(sg * .35, 2.2), (sg * .5, 2.6), (sg * .3, 3.1), (sg * 1.6, 3.1), (sg * 1.6, 2.2)], corners=(0, 1, 2, 3, 4), bow=(1.6, .62)), o[1]); cel.unclip(); outline(cel, O, w[1] * 1.4, g[4], seed + 9)
            for y in (2.56, 2.72): cel.line(V([(-1.1, y - .06), (0, y + .06), (1.1, y - .06)]), w[2], g[4], taper=(.2, .2), seed=seed + int(y * 10))
        cel.unclip()

    def lotus(s, cel, B, L, w, seed=0):
        """crossed legs under the belt: baggy trousers, folds pulled from knee and crotch"""
        g = cloth(L, GI, GIH); u = B.u; sg = -1 if L['ld'][0] > 0 else 1; V = lambda P: spl(B(A(P, float)), n=8)
        H_ = spl(B(A([(-1.1, 2.75), (1.1, 2.75), (1.3, 3.5), (0, 3.9), (-1.3, 3.5)])), True, n=8); outline(cel, H_, w[0] * 1.5, g[4], seed); cel.fill(H_, g[1])
        for q in (-sg, sg):
            T = capsule(B(A([(q * .55, 3.02), (q * 1.55, 3.02), (q * 2.55, 3.36)])), [.56 * u, .62 * u, .50 * u], 12); S = capsule(B(A([(q * 2.50, 3.50), (q * 1.45, 3.78), (q * .15, 3.96), (-q * .75, 3.92)])), [.46 * u, .46 * u, .38 * u, .30 * u], 12)
            for P in (S, T): outline(cel, P, w[0] * 1.6, g[4], seed + q, tremor=.12)
            for P, hi in ((S, 0), (T, 1)):
                cel.fill(P, g[0]); cel.clip(P); lo = P[:, 1].max(); cel.fill(np.vstack([V([(q * -1.2, 4.0), (q * .2, 3.66 - .28 * hi), (q * .9, 3.86 - .30 * hi), (q * 1.6, 3.52 - .30 * hi), (q * 2.2, 3.66 - .20 * hi), (q * 3.2, 3.36 - .1 * hi)]), B(A([(q * 3.4, 5.), (q * -1.4, 5.)]))]), g[1])
                if q == sg: cel.fill(V([(q * 2.0, 2.4), (q * 2.2, 3.0), (q * 2.05, 3.3), (q * 2.35, 3.6), (q * 3.4, 3.9), (q * 3.4, 2.4)]), g[1])
                cel.unclip()
            cel.fill(V([(q * .8, 2.62), (q * 1.4, 2.60), (q * 1.9, 2.78), (q * 1.4, 2.72)]), g[3])
            for pts, ww in (([(q * 2.72, 3.22), (q * 2.2, 3.20), (q * 1.75, 3.02)], 1.1), ([(q * 2.62, 3.50), (q * 2.1, 3.44), (q * 1.6, 3.48)], 1.0), ([(q * .5, 2.75), (q * .85, 3.05), (q * 1.35, 3.18)], 1.), ([(q * .35, 3.0), (q * .7, 3.30), (q * 1.1, 3.42)], .9), ([(q * 1.9, 3.80), (q * 1.4, 3.66), (q * .9, 3.72)], .9), ([(q * .6, 4.1), (q * .3, 3.86), (-q * .2, 3.80)], .8)): cel.line(V(pts), w[1] * ww, g[4], taper=(.08, .6), press=.3, seed=seed + int(pts[0][1] * 10))
            F_ = capsule(B(A([(-q * .55, 3.98), (-q * 1.25, 3.92), (-q * 1.65, 3.72)])), [.24 * u, .25 * u, .16 * u], 9); c = cloth(L, SHOE); poly(cel, F_, c, w[0] * 1.1, seed + 3, sh=[F_ + A([0, .16 * u])])

    def bare_arm(s, cel, L, sh, el, wr, u, w, sg, seed=0, over='f', band_=True, fist=None):
        """green arm with the pink muscle patches; returns the px unit for its hand"""
        sk = cloth(L, SKIN); pk = cloth(L, PINK); wd = [.40 * u, .37 * u, .285 * u, .315 * u, .185 * u]; U, Fo = arm(cel, sh, el, wr, wd, sk, w, sg, seed, over, muscle=1)
        def patch(Q, t0, t1, c0, c1):
            P, a, b, C = Q; n = len(C); i0, i1 = int(t0 * n), int(t1 * n); t = np.linspace(0, 1, i1 - i0)[:, None]; X_ = lambda c, i: C[i] + (a[i] - C[i]) * c
            e0 = np.array([X_(c0 + .10 * math.sin(v * 3.14), i0 + int(v * (i1 - i0 - 1))) for v in np.linspace(0, 1, 14)]); e1 = np.array([X_(c1, i0 + int(v * (i1 - i0 - 1))) for v in np.linspace(0, 1, 14)]); S = np.vstack([e0, e1[::-1]]); cel.clip(P); cel.fill(S, pk[0]); s_ = 1 if (a - b)[len(a) // 2, 0] * sg >= 0 else -1; cel.clip(S)
            cel.fill(np.vstack([C + (a - C) * (.30 * s_), C + (a - C) * 1.4 * s_][0:1] + [(C + (a - C) * 1.4 * s_)[::-1]]), pk[1]); cel.unclip(); cel.line(np.vstack([S, S[:1]]), w[1] * .9, tone(L, PINKL, 0), taper=(.02, .02), press=.2, seed=seed)
            for v in np.linspace(.16, .86, 6): i = i0 + int(v * (i1 - i0 - 1)); cel.line(np.vstack([X_(c0 + .14, i), X_((c0 + c1) / 2, i + int(.03 * n)), X_(c1 - .1, i)]), w[2] * 1.1, tone(L, PINKL, 0), taper=(.15, .3), seed=seed + int(v * 9))
            cel.unclip()
        patch(U, .22, .88, -.45, .70); patch(Fo, .12, .70, -.60, .55)
        if band_: d = A(el) - A(wr); band(cel, A(wr) - d / np.hypot(*d) * .02 * u, d, .205 * u, .44 * u, cloth(L, BAND), w, seed, sg=sg)
        return .46 * u

    def bust(s, cx, cy, sc, yaw=0., pitch=0., roll=0., expr='neutral', light='day', f=0, wind=(0, 0), seed=1, turban=True, wet=0., droop=0., pose='stand', byaw=None, torn=0., w=W, h=FH, rims=True, cape=True, lift=0.):
        L = globals()['light'](light); cel = Cel(w, h); H = s.make(cx, cy, sc, yaw, pitch, roll); wl = lw(sc); u = sc; o = H.P3(A([(0, -1.62, -.02)]))[0]; B = Body(o[0], o[1], u, yaw * .55 if byaw is None else byaw, 0., roll * .25); sg = -1 if L['ld'][0] > 0 else 1
        if cape: s.cape_back(cel, B, L, wl, seed, 1., wind[0] * 1.2, torn=torn)
        hands = []
        def put(q, el, wr, hp, R, front=True, hs=.46):
            sh = B([(q * 1.42, .52, .0)])[0]; el, wr = B([el])[0], B([wr])[0]; d = wr - el; hu = s.bare_arm(cel, L, sh, el, wr, u, wl, sg, seed + 3 + q)
            hand(cel, L, SKIN, hp, wr, hu, R, 'R' if q < 0 else 'L', HD.build(len=1.06, wid=1.02, boxy=.5, tip=.2, knuckle=1.1, wrist=1.0), seed=seed + q, style=dict(nails='claw', claw=.05, arm=.4), nail=('#f2efe0', '#c8c4b0', '#8a8674'))
        if pose == 'raise': put(-1, (-2.05, .25, .1), (-1.95, -1.15, .5), HD.POSES['open'], HD.euler(180, -8, 6))
        s.neck(cel, H, L, seed); s.torso(cel, B, L, wl, seed)
        if pose in ('stand', 'raise'):
            for q in ((1,) if pose == 'raise' else (-1, 1)): put(q, (q * 1.75, 1.75, .1), (q * 1.70, 3.1, .25), HD.POSES['fist'], HD.euler(q * 60, 0, 180))
        if pose == 'lotus': s.lotus(cel, B, L, wl, seed)
        if pose in ('cross', 'lotus'):
            for q in (-1, 1): put(q, (q * 1.75, 1.60, .5), (-q * .55, 1.72 - .22 * q, 1.0), HD.POSES['relaxed'], HD.euler(0, 0, q * 95))
        if cape: s.yoke(cel, B, L, wl, seed, lift, torn=torn)
        s.head(cel, H, dict(expr=expr, light=light, f=f, wind=wind, seed=seed, turban=turban, wet=wet, droop=droop)); s.last = (H, B)
        lay = cel.layer(h, w); return grade(lay, L, (cx - 3 * sc, cy - 1.8 * sc, cx + 3 * sc, cy + 5 * sc), max(.5, sc / 170), rims=rims)

    # ------------------------------------------------------------------ c4: he throws himself round the child
    def shield(s, cx, cy, sc, light='blast', f=0, seed=2, rip=0., sway=0., expr='shield', child='terror', yaw=-.22, pitch=-.22, roll=.10, turban=True, spread=1., w=W, h=FH, rims=True):
        """Piccolo seen from the front, his back to the blast, Gohan inside his arms, the cape wide as a wall.  (cx, cy) = his head; sc = px per head unit; rip 0..1 tears the cape to ribbons"""
        from chars.gohan import Gohan
        L = globals()['light'](light); wl = lw(sc); H = s.make(cx, cy, sc, yaw, pitch, roll); o = H.P3(A([(0, -1.30, -.10)]))[0]; u = sc; B = Body(o[0], o[1], u, -.05, 0., roll * .4); sg = -1 if L['ld'][0] > 0 else 1; cp = cloth(L, CAPE, CAPEH); cel = Cel(w, h); r = np.random.default_rng(seed); k = spread; g_ = lambda j: .10 * math.sin(f * .9 + j * 1.7)
        # the cape: two wings of cloth blown round them, each edge its own wave; ribbons when it rips
        for q in (-1, 1):
            m = 1.10 if q == -sg else .94; E_ = [(q * .45, -.55), (q * 1.5 * k * m, -1.25 + g_(1)), (q * 3.0 * k * m, -1.55 + g_(2)), (q * 4.5 * k * m, -1.10 + g_(3)), (q * 5.35 * k * m, -.10 + g_(4))]
            n = 9; edge = []
            for j in range(n):
                v = j / (n - 1); bx = q * (5.35 - 3.5 * v ** 1.15) * k * m; by = .10 + 4.5 * v ** .92; edge.append((bx + q * .16 * math.sin(j * 2.1 + f * .8), by + g_(j + 5)))
                if rip > 0 and j < n - 1: edge.append((bx - q * (.45 + 1.7 * rip * r.uniform(.5, 1.)) * k, by + .32 + .25 * r.random()))
            P = E_ + edge + [(q * .9, 4.8), (q * .4, 3.0), (q * .4, .5)]; nE = len(E_); cor = [0, nE - 1] + (list(range(nE, nE + len(edge))) if rip > 0 else []) + [len(P) - 3]
            C = spl(B(P, z=-.4), True, corners=cor, n=7); outline(cel, C, wl[0] * 1.7, cp[4], seed + q, tremor=.15); cel.fill(C, mixc(cp[0], cp[1], .35)); cel.clip(C)
            for j, (a0, l_) in enumerate(((-1.0, 5.4), (-.62, 5.6), (-.30, 5.6), (.02, 5.5), (.34, 5.3), (.66, 5.2), (1.0, 5.1))):
                d = A([q * math.cos(a0 * .9 + .12) * m * k, math.sin(a0 * .9 + .12)]); p0 = A([q * .95, -.10]); Q = [p0 + d * .6, p0 + d * l_ * .45 + A([0, .20 * (-1) ** j]), p0 + d * l_ * .98]; c = spl(B(Q, z=-.4), n=9); cel.fill(wedge(c[::-1], lambda t: u * (.75 * (1 - t) ** .9) * (1 if j % 2 else .62)) + A([0, .14 * u]), cp[1] if j % 2 else mixc(cp[1], cp[2], .55)); cel.line(c, wl[1] * 1.1, cp[4], taper=(.4, .15), press=.3, seed=seed + j)
            cel.fill(spl(B([(q * .2, -.6), (q * 1.4, -.4), (q * 1.9, .6), (q * 1.7, 2.0), (q * 2.1, 3.4), (q * 1.6, 5.2), (0, 5.2)], z=-.4), True, n=7), mixc(cp[1], cp[2], .6)); cel.unclip()
        if rip > .3:                                                # loose ribbons torn off and flying
            for j in range(int(16 * rip)):
                p = B([(r.uniform(-5.5, 5.5), r.uniform(-1.6, 3.5))])[0] - A([(30 + 14 * j % 50) * rip, 0]) * .0; a = math.pi + r.normal(0, .35); ln = u * r.uniform(.5, 1.3); c = spl(A([p, p + A([math.cos(a), math.sin(a)]) * ln * .5 + [0, u * .12 * (-1) ** j], p + A([math.cos(a), math.sin(a)]) * ln]), n=8); poly(cel, wedge(c, u * .16, u * .03), cp, wl[1], seed + j)
        G_ = lambda c: grade(c.layer(h, w), L, (cx - 5 * sc, cy - 2 * sc, cx + 5 * sc, cy + 5 * sc), max(.5, sc / 140), rims=rims); s.neck(cel, H, L, seed); s.torso(cel, B, L, wl, seed, 4.6); lay = G_(cel)
        # the child, pressed against his chest
        gx, gy = B([(.06, 1.86, .9)])[0]; gl = Gohan.bust(gx, gy, sc * .74, .22, -.06, -.16, child, light, f, (-.25, .05), seed + 1, False, 'fear', byaw=.1, bottom=3.0, w=w, h=h, rims=False); a = gl[..., 3]; sh = np.clip(shifted(a, 6, 10) * .0 + cv2.GaussianBlur(a, (0, 0), .12 * sc) * .55, 0, 1)[..., None] * (1 - a[..., None]); lay[..., :3] *= 1 - sh * .55; lay = lover(lay, gl)
        # his arms close over him
        cel = Cel(w, h)
        for q, el, wr, hp, R in ((1, (2.35, 2.35, .7), (-.35, 3.45, 1.5), 'rest', (-8, 12, 78)), (-1, (-2.40, 2.20, .7), (1.05, 3.02, 1.6), 'rest', (12, 14, -70))):
            sh_ = B([(q * 1.42, .50, .1)])[0]; e_, w_ = B([el])[0], B([wr])[0]; hu = s.bare_arm(cel, L, sh_, e_, w_, u, wl, sg, seed + 3 + q); hand(cel, L, SKIN, HD.POSES[hp], w_, hu * 1.1, HD.euler(*R), 'L' if q > 0 else 'R', HD.build(len=1.06, wid=1.02, boxy=.5, tip=.2, knuckle=1.1), seed=seed + q, style=dict(nails='claw', claw=.05, arm=.4), nail=('#f2efe0', '#c8c4b0', '#8a8674'))
        def put(lay, al, dy, sof, amt): a = al[..., 3]; sh = (np.clip(cv2.GaussianBlur(shifted(a, 0, dy * sc), (0, 0), sof * sc), 0, 1) * (1 - a))[..., None]; lay[..., :3] *= 1 - sh * amt; return lover(lay, al)
        lay = put(lay, G_(cel), .12, .08, .38); cel = Cel(w, h); s.yoke(cel, B, L, wl, seed, .30 + .06 * math.sin(f), torn=rip); s.head(cel, H, dict(expr=expr, light=light, f=f, seed=seed, turban=turban, part='all')); return put(lay, G_(cel), .10, .08, .34)


Piccolo = _Piccolo()
head_fn = Piccolo.head

if __name__ == '__main__':
    bg = np.ones((1080, 1920, 3), f32) * rgb('#8d78b8'); bg = comp.over(bg, Piccolo.shield(960, 330, 118, rip=.0)); a = cv2.resize(bg, (960, 540), interpolation=cv2.INTER_AREA)
    bg = np.ones((1080, 1920, 3), f32) * rgb('#e9d8ff'); bg = comp.over(bg, Piccolo.shield(960, 330, 118, rip=.9, turban=False, child='shut', seed=3)); b = cv2.resize(bg, (960, 540), interpolation=cv2.INTER_AREA)
    comp.save('qa/friends/_shield.jpg', np.hstack([a, b]))
