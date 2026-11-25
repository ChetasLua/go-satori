"""BULMA, KRILLIN, ROSHI: busts with a raised hand (g6), normal proportions.
    Bulma.bust(cx, cy, s, yaw, pitch, roll, expr, light, f, wind, seed, pose='raise')  -> premultiplied RGBA.   Krillin, Roshi the same.   .head(cel, H, info) head functions
"""
from chars.fkit import *
from db import hands as HD

# ================================================================= BULMA
B_SKIN = ('#ffdfc4', '#e2a08a', '#b2706a'); TEAL = dict(base=('#35b6b0', '#1f7f8a', '#165a6e'), sh=('#1c7a86', '#145a6a', '#0e3f52'), hi=('#8fe6da', '#5cbcb8', '#3a8a90'), sep=('#11505c', '#0c3a48', '#082a38'))
B_JACK = ('#c8dc3c', '#86a02a', '#55701c'); B_TOP = ('#b0305e', '#741c46', '#4a1030')
BD = dict(
    skin=B_SKIN, blush0=.25, blushc='#f07a78', side=-.40, nose='adult', nshadow=.6, my=-.665,
    head=dict(a=[(-1.0, .085), (-.965, .155), (-.88, .285), (-.76, .41), (-.60, .515), (-.42, .58), (-.22, .625), (-.05, .65), (.12, .675), (.35, .69), (.6, .66), (.8, .53), (.93, .33), (1.0, .05)], p=2.25, nose=(.07, -.12, -.40, -.445, .115), ear=(.655, .06, -.38, -.10), cheek=.05, socket=.03, low=.955),
    eye=dict(xi=.150, xo=.505, U=[.0, .100, .130, .108, .036], Lo=[-.082, -.118, -.128, -.112, -.046], ir=(.082, .118), ic=.46, iy=.006, iris=('#12343e', '#2a8fa0', '#86e0dc'), lt=[(0, .006), (.16, .022), (.45, .030), (.78, .040), (1, .030)], down=.30, lo0=.42, low=.8, ls=.036, cro=.036, corners=(0, 4), crease=1, tail=.030,
             lash=[(.86, .050, .032, .020), (.97, .058, .012, .020), (.72, .030, .042, .014)]),
    brow=dict(pts=[(.155, .230), (.265, .270), (.395, .275), (.520, .220)], th=[(0, .012), (.3, .022), (.7, .018), (1, .002)], col=('#1c6a74', '#155460', '#0e3f4c')))
BE = dict(neutral=X(hw=.095, mU=[.012, 0, -.004, 0, .012], mL=[.012, 0, -.004, 0, .012]), smile=X(hw=.135, my=-.65, mU=[.040, .002, -.010, .002, .042], mL=[.040, -.016, -.034, -.016, .042], blush=.3, dB=.02),
          hope=X(U=[.006, .112, .146, .120, .044], gy=.018, B=[(.140, .285), (.262, .290), (.395, .272), (.520, .205)], hw=.085, my=-.66, mU=[.0, .020, .022, .020, .0], mL=[.0, -.030, -.040, -.030, .0], tongue=.4, jaw=.02, blush=.2, tears=.5, glint=1.2),
          shout=X(B=[(.130, .190), (.258, .250), (.395, .290), (.525, .270)], hw=.17, my=-.60, mU=[.014, .030, .024, .030, .014], mL=[-.02, -.17, -.21, -.17, -.02], teeth=.04, tongue=.8, jaw=.09, furrow=.5))


def blocks(H):
    L = []; n = [0]
    def sp(th, y, d, ln, w, curl=(0, 0, 0), arc=(0, 0, 0), **k): n[0] += 1; L.append(Lock.sp(H, th, y, d, ln, w, curl, arc, id=n[0], **k))
    def fr(th, y, tip, w, **k): n[0] += 1; L.append(Lock.fr(H, th, y, tip, w, id=n[0], tag='fr', **k))
    # a bob to the jaw, ends turned in; a heavy side-swept fringe
    for q in (-1, 1):
        sp(q * 1.12, .62, (q * .20, -1, .18), 1.22, .40, (-q * .16, .02, .08), (q * .20, .10, .02), kind='bob', tag='side'); sp(q * 1.42, .58, (q * .30, -1, .02), 1.34, .46, (-q * .16, .04, .02), (q * .24, .10, 0), kind='bob'); sp(q * 1.75, .56, (q * .22, -1, -.10), 1.36, .48, (-q * .10, .04, 0), (q * .22, .10, -.06), kind='bob'); sp(q * 2.15, .55, (q * .14, -1, -.18), 1.34, .50, (-q * .06, .04, 0), (q * .14, .08, -.14), kind='bob'); sp(q * 2.65, .55, (q * .05, -1, -.2), 1.32, .50, (0, .04, .04), (q * .05, .08, -.2), kind='bob')
        sp(q * .98, .70, (q * .10, -1, .30), 1.10, .30, (-q * .10, 0, .05), (q * .12, .10, .05), kind='bob', tag='side'); sp(q * 1.30, .40, (q * .36, -1, .06), 1.02, .18, (q * .10, .10, 0), (q * .16, .04, 0), kind='flip')
    fr(.40, .92, (-.560, -.02), .46, asym=-.30, bulge=.30); fr(.46, .92, (-.360, .20), .30, asym=-.25, bulge=.30); fr(.50, .92, (.600, .02), .40, asym=.26, bulge=.24); fr(.36, .93, (-.150, .34), .22, asym=-.2, bulge=.30, dz=.09); fr(.50, .92, (.40, .30), .22, asym=.2, dz=.08, bulge=.24); fr(.30, .92, (-.630, .22), .26, asym=-.2, bulge=.30, dz=.03)
    return L


def wrapk(cel, P, cols, w, seed, sh=(), hi=(), lines=()): poly(cel, P, cols, w, seed, sh=sh, hi=hi, lines=lines)


class _Base:
    D = None; E = None; ND = .36; drop = 1.45; BLD = HD.MALE; hs = .40
    def make(s, cx, cy, sc, yaw=0., pitch=0., roll=0., persp=.10): return FHead(s.D).pose(cx, cy, sc, yaw, pitch, roll, persp)
    def ex(s, e): return s.E[e] if isinstance(e, str) else (blend(s.ex(e[0]), s.ex(e[1]), e[2]) if isinstance(e, (tuple, list)) else e)

    def neck(s, cel, H, L, seed=0):
        w = lw(H.s); c = cloth(L, s.D['skin']); sg = -1 if L['ld'][0] > 0 else 1; k = s.ND; a, b = [H.P3(A([(q * (k + .02), -.50, -.12), (q * k, -.90, -.10), (q * k, -1.15, -.10), (q * (k + .12), -1.50, -.10), (q * (k + .45), -1.75, -.12)])) for q in (-1, 1)]
        P = np.vstack([spl(a, n=8), spl(b, n=8)[::-1]]); cel.fill(P, c[0]); cel.clip(P); J = np.vstack([H.jaw(-1), H.jaw(1)[::-1][1:]]) + A([sg * .06, .24]) * H.s; cel.fill(np.vstack([J, H.P3(A([(1.2, .3, 0), (-1.2, .3, 0)]))]), c[1]); e = spl(a if sg < 0 else b, n=8); cel.fill(np.vstack([e, (e + A([-sg * .12, 0]) * H.s)[::-1]]), c[1]); cel.unclip()
        for i, q in enumerate((a, b)): cel.line(spl(q, n=8), w[0] * .85, c[4], taper=(.25, .1), seed=seed + i)
        for q in (-1, 1): cel.line(spl(H.P3(A([(q * k * .8, -.82, .08), (q * k * .5, -1.12, .16), (q * .06, -1.45, .22)])), n=8), w[2], c[4], taper=(.4, .5), seed=seed + 4 + q, alpha=.8)

    def rarm(s, cel, B, L, q, el, wr, w, seed, skin, wd=(.34, .30, .235, .25, .155), sleeve=None, pose='open', R=(180, -6, 8), sh=(1.15, .42), hs=None, band_=None):
        u = B.u; sg = -1 if L['ld'][0] > 0 else 1; sk = cloth(L, skin); p0 = B([(q * sh[0], sh[1], 0)])[0]; el, wr = B([el])[0], B([wr])[0]; d = el - wr; d /= np.hypot(*d); U, Fo = arm(cel, p0, el, wr, [v * u for v in wd], sk, w, sg, seed, 'f', muscle=s.muscle)
        hand(cel, L, skin, HD.POSES[pose] if isinstance(pose, str) else pose, wr - d * .02 * u, (hs or s.hs) * u, HD.euler(*R), 'R' if q < 0 else 'L', s.BLD, seed=seed, style=dict(arm=.35))
        if band_ is not None: band(cel, wr, d, (wd[4] + .02) * u, .40 * u, cloth(L, band_), w, seed, sg=sg)
        if sleeve is not None: s.sleeve(cel, B, L, p0, el, sleeve, w, seed, wd[0] * u)

    muscle = 0.; hp = 'open'
    def sleeve(s, cel, B, L, sh, el, tri, w, seed, r, ln=.62, flare=1.25):
        u = B.u; g = cloth(L, tri); sg = -1 if L['ld'][0] > 0 else 1; t = el - sh; D_ = np.hypot(*t); k = t / D_; n = A([-k[1], k[0]]); l = D_ * ln; X_ = lambda a_, b_: sh + k * a_ * l + n * b_ * r; s_ = 1 if n[0] * sg > 0 else -1; f = flare
        hem = [X_(1.0 - .12 * (1 - v * v), v * f) for v in np.linspace(-1, 1, 9)]; hb = [X_(1.0 + .13 * (1 - v * v), v * f) for v in np.linspace(-1, 1, 9)]; cel.fill(A(hem + hb[::-1]), g[2]); cel.line(A(hem), w[1], g[4], taper=(.05, .05), seed=seed)
        P = spl(A([X_(-.42, -.3), X_(-.28, -.86), X_(.1, -1.12), X_(.55, -1.12), X_(1.0, -f)]), n=7); Q = spl(A([X_(1.0, f), X_(.55, 1.12), X_(.1, 1.12), X_(-.28, .86), X_(-.42, .3)]), n=7); C = np.vstack([P, spl(A(hb), n=5)[1:-1], Q])
        cel.fill(C, g[0]); cel.clip(C); cel.fill(A([X_(-.5, s_ * .3), X_(-.05, s_ * .45), X_(.3, s_ * .12), X_(.55, s_ * .5), X_(1.2, s_ * .25), X_(1.3, s_ * 2), X_(-.5, s_ * 2)]), g[1]); cel.fill(A([X_(-.1, -s_ * .6), X_(.4, -s_ * .85), X_(.9, -s_ * .85), X_(.4, -s_ * .55)]), g[3])
        for x0, y0, x1, y1 in ((1.05, -.7, .45, -.4), (1.12, .2, .55, .05), (1.02, .85, .50, .6), (.0, -.95, .35, -.4)): cel.line(spl(A([X_(x0, y0), X_((x0 + x1) / 2, (y0 + y1) / 2 + .1), X_(x1, y1)]), n=6), w[1] * .8, g[4], taper=(.05, .7), seed=seed + 3)
        cel.unclip(); cel.line(C, w[0] * .95, g[4], taper=(.06, .06), press=.2, seed=seed + 9, tremor=.12)

    def bust(s, cx, cy, sc, yaw=0., pitch=0., roll=0., expr='neutral', light='day', f=0, wind=(0, 0), seed=1, pose='raise', w=W, h=FH, rims=True, byaw=None, arm_=None):
        L = globals()['light'](light) if isinstance(light, (str, dict)) else light; cel = Cel(w, h); H = s.make(cx, cy, sc, yaw, pitch, roll); wl = lw(sc); o = H.P3(A([(0, -s.drop, -.04)]))[0]; B = Body(o[0], o[1], sc, yaw * .5 if byaw is None else byaw, 0., roll * .3); inf = dict(expr=expr, light=light, f=f, wind=wind, seed=seed)
        s.head(cel, H, inf, part='back'); s.body(cel, H, B, L, wl, seed, pose, f, arm_, inf); s.head(cel, H, inf, part='front'); s.over(cel, H, B, L, wl, seed, pose, f, arm_); s.last = (H, B)
        lay = cel.layer(h, w); return grade(lay, L, (cx - 2 * sc, cy - 1.6 * sc, cx + 2 * sc, cy + 4 * sc), max(.5, sc / 160), rims=rims)

    def over(s, *a): pass


class _Bulma(_Base):
    D = BD; E = BE; ND = .27; drop = 1.42; BLD = HD.FEMALE; hs = .37
    def head(s, cel, H, info=None, **k):
        i = dict(expr='neutral', light='day', f=0, wind=(0, 0), seed=1, part='all'); i.update(info or {}); i.update(k); x = s.ex(i['expr']); L = light(i['light']); H = FHead.of(H, BD, x['jaw']); w = lw(H.s); LK = blocks(H); hc = dict(base=tone(L, TEAL['base'], 0), sh=tone(L, TEAL['sh'], 1), hi=tone(L, TEAL['hi'], 0), sep=tone(L, TEAL['sep'], 1), hip=.9, hiw=1.2); ln = mixc(L['line'], '#0c3a44', .5)
        for l in LK: l.build(H, wind_off(H, i['wind'], i['f'], (l.id * .618) % 1, {'fr': .5}.get(l.tag, .8)))
        ib = lambda l: l.tag != 'fr' and (l.vis < -.15 or l.depth < -.30)
        if i['part'] in ('all', 'back'): hair_group(cel, sorted([l for l in LK if ib(l)], key=lambda l: l.depth), dict(hc, base=hc['sh'], hi=None), w, i['seed'], L['ld'], caps=[scalp(H, lambda t: -.45, 1.10, front=False)], line=ln)
        if i['part'] == 'back': return H
        fr = sorted([l for l in LK if not ib(l)], key=lambda l: l.depth); F = Face(cel, H, BD, x, L, i['seed'], hairs=[l.poly for l in fr if l.tag == 'fr']); F.ears_far(); F.ears_near(); F.base(); F.features(); F.contour()
        # lips: a touch of colour
        hair_group(cel, fr, hc, w, i['seed'], L['ld'], caps=[scalp(H, lambda t: .58 - .28 * min(abs(t) / 1.05, 1) ** 2 - .6 * max(0., min((abs(t) - 1.3) / .8, 1.)), 1.10)], line=ln)
        for q in (-1, 1):                                           # earring
            if H.ear(q)['vis'] > .25 and q * H.yaw < .3: p = H.P3(A([(q * .70, -.46, -.05)]))[0]; cel.fill(ell(p[0], p[1], .060 * H.s), tone(L, '#f2c23a', 0)); cel.line(ell(p[0], p[1], .060 * H.s), w[1], L['line'], taper=(.05, .05))
        return H

    def body(s, cel, H, B, L, w, seed, pose, f, arm_, inf):
        j = cloth(L, B_JACK); t = cloth(L, B_TOP); sk = cloth(L, B_SKIN); sg = -1 if L['ld'][0] > 0 else 1; bw = (1.1, .42)
        P = spl(np.vstack([B([(-x, y) for x, y in ((.30, -.10), (.85, .16), (1.02, .45), (.92, 1.0), (.80, 1.6), (.74, 2.2), (.90, 3.4))], bow=bw), B([(.30, -.10), (.85, .16), (1.02, .45), (.92, 1.0), (.80, 1.6), (.74, 2.2), (.90, 3.4)], bow=bw)[::-1]]), True, corners=(0, 6, 7, 13), n=8); V = lambda Q, c=(): B.c(Q, True, corners=c, bow=bw)
        outline(cel, P, w[0] * 1.6, t[4], seed); cel.fill(P, t[0]); s.neck(cel, H, L, seed); cel.clip(np.vstack([P, B([(-.3, -.6), (.3, -.6)])]) if False else P); cel.fill(V([(-.46, -.3), (-.40, .20), (0, .52), (.40, .20), (.46, -.3)], (0, 4)), sk[0]); cel.fill(V([(-.40, .18), (0, .50), (.42, .18), (.36, .10), (0, .30), (-.36, .10)], (0, 2)), sk[1]); cel.line(B.c([(-.40, .2), (0, .52), (.40, .2)], False, bow=bw), w[1], t[4])
        cel.fill(V([(sg * .10, .5), (sg * .36, .9), (sg * .16, 1.3), (sg * .36, 2.0), (sg * .26, 3.5), (sg * 1.2, 3.5), (sg * 1.2, .3)], range(7)), t[1])
        for q in (-1, 1): cel.fill(V([(q * .12, 1.12), (q * .48, 1.22), (q * .80, 1.05), (q * .52, 1.12)], (0, 2)), t[1]); cel.line(B.c([(q * .14, 1.12), (q * .46, 1.24), (q * .74, 1.10)], False, bow=bw), w[2] * 1.2, t[4], taper=(.3, .5), seed=seed + q)
        # open jacket
        for q in (-1, 1):
            C = V([(q * .30, -.16), (q * .62, -.08), (q * 1.04, .16), (q * 1.14, .5), (q * 1.02, 1.1), (q * .92, 1.7), (q * .96, 2.5), (q * 1.1, 3.5), (q * .50, 3.5), (q * .46, 2.4), (q * .40, 1.6), (q * .52, .9), (q * .44, .35)], (0, 7, 8)); poly(cel, C, j, w[1] * 1.6, seed + q, sh=[V([(q * .30, -.2), (q * .56, .4), (q * .62, 1.0), (q * .50, 1.6), (q * .58, 2.4), (q * .56, 3.5), (q * .3, 3.5)], range(7))] + ([V([(q * .70, -.1), (q * .86, .5), (q * .72, 1.0), (q * .84, 1.7), (q * .80, 3.5), (q * 1.3, 3.5), (q * 1.3, 0)], range(7))] if q == sg else []), lines=[(B([(q * .95, .9), (q * .78, 1.2), (q * .72, 1.6)], bow=bw), .7), (B([(q * .60, .1), (q * .70, .5), (q * .66, .9)], bow=bw), .6)])
            K = V([(q * .24, -.30), (q * .50, -.28), (q * .72, .04), (q * .56, .42), (q * .40, .30), (q * .30, .0)], (0, 1, 2, 3)); poly(cel, K, j, w[1] * 1.4, seed + 3 + q, sh=[K + A([sg * .14, .06]) * B.u])
        cel.unclip()
        if pose == 'raise': s.rarm(cel, B, L, 1, (1.08, 1.45, .1), (.92, 2.5, .3), w, seed + 4, B_SKIN, wd=(.27, .24, .19, .19, .125), sleeve=B_JACK, sh=(.94, .44), pose='relaxed', R=(70, 0, 180))

    def over(s, cel, H, B, L, w, seed, pose, f, arm_):
        if pose == 'raise': el, wr = arm_ or ((-1.35, .28, .1), (-1.12, -.80, .4)); s.rarm(cel, B, L, -1, el, wr, w, seed + 5, B_SKIN, wd=(.27, .235, .185, .19, .120), sleeve=B_JACK, sh=(.94, .44), R=(178, -8, 4), pose=s.hp)


Bulma = _Bulma()

# ================================================================= KRILLIN
K_SKIN = ('#ffd6ac', '#d98f68', '#a05c4c'); K_GI = ('#f47b20', '#b84f1c', '#7a2f18'); K_BLUE = ('#2c4f92', '#1b3468', '#101f44')
KD = dict(
    skin=K_SKIN, blush0=.10, side=-.2, nose='none', my=-.60, topcut=.22,
    head=dict(a=[(-1.0, .15), (-.97, .235), (-.90, .36), (-.80, .48), (-.66, .59), (-.50, .665), (-.30, .715), (-.10, .74), (.12, .755), (.35, .765), (.6, .72), (.8, .59), (.93, .37), (1.0, .05)], p=2.2, nose=(.06, -.2, -.36, -.40, .02), ear=(.735, .05, -.42, -.08), cheek=.05, socket=.02, low=.86, relief=.7),
    eye=dict(xi=.130, xo=.440, U=[-.010, .098, .128, .112, .050], Lo=[-.094, -.128, -.136, -.118, -.050], ir=(.060, .088), ic=.45, iy=-.005, iris='black', lt=[(0, .008), (.15, .024), (.45, .032), (.8, .038), (1, .030)], down=.55, lo0=.4, low=.85, ls=.030, crease=0, corners=(0, 4), g2=0),
    brow=dict(pts=[(.115, .185), (.215, .245), (.345, .265), (.465, .225)], th=[(0, .030), (.15, .054), (.6, .046), (1, .004)], col='ink'))
KE = dict(neutral=X(hw=.11, mU=[.006, 0, -.004, 0, .006], mL=[.006, 0, -.004, 0, .006]), grin=X(hw=.19, my=-.56, mU=[.052, .012, .0, .012, .052], mL=[.040, -.090, -.12, -.090, .040], teeth=.05, tongue=.6, jaw=.03, dB=.03),
          firm=X(B=[(.100, .135), (.215, .205), (.350, .255), (.470, .260)], furrow=.8, hw=.12, mU=[-.012, .004, .008, .004, -.012], mL=[-.012, .004, .008, .004, -.012], mw=1.2),
          hope=X(B=[(.105, .235), (.215, .262), (.350, .258), (.465, .205)], gy=.012, hw=.10, mU=[.0, .016, .018, .016, .0], mL=[.0, -.03, -.038, -.03, .0], tongue=.4, jaw=.02, glint=1.2),
          shout=X(B=[(.100, .145), (.215, .215), (.350, .268), (.470, .272)], furrow=1., hw=.21, my=-.50, mU=[.018, .034, .026, .034, .018], mL=[-.03, -.22, -.27, -.22, -.03], teeth=.05, tongue=.8, jaw=.12))


class _Krillin(_Base):
    D = KD; E = KE; ND = .37; drop = 1.28; BLD = HD.MALE; hs = .42; muscle = .8
    def head(s, cel, H, info=None, **k):
        i = dict(expr='neutral', light='day', f=0, wind=(0, 0), seed=1, part='all'); i.update(info or {}); i.update(k); x = s.ex(i['expr']); L = light(i['light']); H = FHead.of(H, KD, x['jaw']); w = lw(H.s); sk = cloth(L, K_SKIN)
        if i['part'] == 'back': return H
        F = Face(cel, H, KD, x, L, i['seed']); F.ears_far(); F.ears_near(); sg = F.sg; sf = lambda th, y: H.P3(surf(H, th, y)[0][None])[0]; S = H.skull(); outline(cel, S, w[0] * 1.6, L['line'], i['seed'] + 9, tremor=.1); cel.fill(S, sk[0]); F.base(); cel.clip(S)
        e0 = [sf(sg * (.98 + .05 * math.sin(y * 8)), y) for y in np.linspace(.02, .97, 9)]; cel.fill(np.vstack([e0, [A([H.cx + sg * 3 * H.s, H.cy - 2 * H.s]), A([H.cx + sg * 3 * H.s, H.cy])]]), sk[1])
        g = [sf(-sg * .95, .55), sf(-sg * .72, .76), sf(-sg * .36, .90), sf(-sg * .46, .78), sf(-sg * .74, .58)]; cel.fill(spl(A(g), True, n=7), tone(L, '#fff6e6', 0)); cel.unclip(); F.features(); F.contour()
        for q in (-1, 1):                                           # six dots
            for yy in (.46, .60, .735):
                th = q * .155; p = sf(th, yy)
                if surf(H, th, yy)[1] @ H.R[2] > .05: cel.fill(ell(p[0], p[1], .040 * H.s * max(.45, surf(H, th, yy)[1] @ H.R[2]), .036 * H.s, n=20), mixc(sk[2], '#5a2c28', .6))
        return H

    def body(s, cel, H, B, L, w, seed, pose, f, arm_, inf):
        o = cloth(L, K_GI, '#ffa04a'); b = cloth(L, K_BLUE); sg = -1 if L['ld'][0] > 0 else 1; bw = (1.25, .46); R_ = ((.36, -.12), (.86, .04), (1.16, .30), (1.10, .8), (.94, 1.4), (.84, 1.95), (.92, 2.4), (1.06, 3.4)); P = spl(np.vstack([B([(-x, y) for x, y in R_], bow=bw), B(R_, bow=bw)[::-1]]), True, corners=(0, 7, 8, 15), n=8); V = lambda Q, c=(): B.c(Q, True, corners=c, bow=bw)
        if pose == 'raise': s.rarm(cel, B, L, 1, (1.36, 1.35, .1), (1.16, 2.3, .4), w, seed + 4, K_SKIN, wd=(.36, .32, .25, .27, .165), sleeve=K_BLUE, sh=(1.12, .42), pose='fist', R=(70, 0, 180), band_=K_BLUE)
        outline(cel, P, w[0] * 1.6, o[4], seed); cel.fill(P, o[0]); s.neck(cel, H, L, seed); cel.clip(P); cel.fill(V([(sg * .20, .1), (sg * .5, .7), (sg * .24, 1.1), (sg * .5, 1.7), (sg * .3, 2.2), (sg * .5, 3.5), (sg * 1.4, 3.5), (sg * 1.4, .1)], range(8)), o[1])
        for q in (-1, 1): cel.fill(V([(q * 1.1, .95), (q * .7, 1.16), (q * .22, 1.20), (q * .55, 1.02)], (0, 2)), o[1]); cel.line(B.c([(q * .62, .1), (q * .40, .62), (q * .10, 1.12)], False, bow=bw), w[1], o[4], taper=(.1, .4), seed=seed + q); cel.line(B.c([(q * .95, 1.2), (q * .66, 1.45), (q * .40, 1.8)], False, bow=bw), w[1], o[4], taper=(.1, .6), seed=seed + q + 3)
        C = V([(-.56, -.3), (-.50, .10), (-.26, .60), (0, .98), (.26, .60), (.50, .10), (.56, -.3)], (0, 3, 6)); cel.fill(C, b[0]); cel.clip(C); cel.fill(V([(sg * .04, 0), (sg * .16, .4), (sg * .06, 1.0), (sg * .8, .5), (sg * .8, -.3)], range(5)), b[1]); cel.fill(V([(-.5, -.3), (-.36, -.02), (0, .12), (.36, -.02), (.5, -.3)], (0, 4)), tone(L, K_SKIN, 1)); cel.unclip(); cel.line(B.c([(-.50, .0), (-.26, .55), (0, .98), (.26, .55), (.50, .0)], False, corners=(2,), bow=bw), w[1] * 1.3, o[4], taper=(.05, .05), seed=seed)
        O = np.vstack([B.c([(-1.1, 1.86), (0, 1.98), (1.1, 1.86)], False, bow=bw), B.c([(1.1, 2.18), (0, 2.32), (-1.1, 2.18)], False, bow=bw)]); poly(cel, O, b, w[1] * 1.4, seed + 9, sh=[V([(sg * .3, 1.8), (sg * .4, 2.1), (sg * .25, 2.4), (sg * 1.3, 2.4), (sg * 1.3, 1.8)], range(5))]); cel.unclip()
        p = B([(.52 * 1, .62)], bow=bw)[0]; r = .20 * B.u
        cel.fill(ell(p[0], p[1], r * max(.4, math.cos(B.yaw + .25)), r), tone(L, ('#ffffff', '#cfd0de', '#9a9ab4'), 0)); cel.line(ell(p[0], p[1], r * max(.4, math.cos(B.yaw + .25)), r), w[2] * 1.3, L['line'], taper=(.05, .05))
        try:                                                        # 亀 on the chest
            from db import letters as LT
            LT.disc(cel, '亀', LT.disc_at(float(p[0]), float(p[1]), float(r), rot=0., sx=max(.4, math.cos(B.yaw + .25))))
        except Exception: pass

    def over(s, cel, H, B, L, w, seed, pose, f, arm_):
        if pose == 'raise': el, wr = arm_ or ((-1.62, .02, .1), (-1.36, -1.10, .4)); s.rarm(cel, B, L, -1, el, wr, w, seed + 5, K_SKIN, wd=(.36, .32, .25, .27, .165), sleeve=K_BLUE, sh=(1.12, .42), R=(182, -8, 6), pose=s.hp, band_=K_BLUE)


Krillin = _Krillin()

# ================================================================= ROSHI
R_SKIN = ('#f6c99a', '#c98a62', '#8e5a48'); BEARD = ('#fbfaf6', '#c6c3d4', '#918cab'); SHIRT = ('#f2a33a', '#b86a28', '#7c431c'); GLASS = ('#b8262c', '#74151f', '#4a0c16')
RD = dict(
    skin=R_SKIN, blush0=.12, side=-.1, nose='adult', nshadow=1.2, my=-.70, topcut=.10,
    head=dict(a=[(-1.0, .14), (-.96, .24), (-.86, .38), (-.72, .49), (-.56, .565), (-.36, .62), (-.12, .665), (.12, .695), (.35, .715), (.6, .68), (.8, .55), (.93, .34), (1.0, .05)], p=2.2, nose=(.12, -.06, -.40, -.47, .175), ear=(.665, .08, -.42, -.10), cheek=.08, socket=.07, low=.95),
    eye=dict(xi=.16, xo=.48, U=[0, .04, .05, .04, 0], Lo=[-.04, -.06, -.06, -.06, -.04], ir=(.05, .06), iris='black'), brow=None, lines=[([(.20, .46), (.36, .50), (.50, .44)], .6), ([(.16, .56), (.34, .61), (.50, .55)], .5), ([(.50, -.34), (.46, -.46), (.47, -.58)], .7)])
RE = dict(neutral=X(closed=1, hw=.05, lip=0), hope=X(closed=1, hw=.085, my=-.70, mU=[.004, .016, .018, .016, .004], mL=[0, -.045, -.06, -.045, 0], tongue=.5, jaw=.03, lip=0), laugh=X(closed=1, hw=.17, my=-.66, mU=[.03, .03, .02, .03, .03], mL=[0, -.13, -.17, -.13, 0], teeth=.04, tongue=.8, jaw=.08, lip=0))


class _Roshi(_Base):
    D = RD; E = RE; ND = .34; drop = 1.36; BLD = HD.build(len=.98, wid=.98, boxy=.6, tip=.5, knuckle=1.2); hs = .42
    def head(s, cel, H, info=None, **k):
        i = dict(expr='neutral', light='day', f=0, wind=(0, 0), seed=1, part='all'); i.update(info or {}); i.update(k); x = s.ex(i['expr']); L = light(i['light']); H = FHead.of(H, RD, x['jaw']); w = lw(H.s); sk = cloth(L, R_SKIN); bd = cloth(L, BEARD, '#ffffff')
        if i['part'] == 'back': return H
        F = Face(cel, H, RD, x, L, i['seed']); F.ears_far(); F.ears_near(); sg = F.sg; sf = lambda th, y: H.P3(surf(H, th, y)[0][None])[0]; S = H.skull(); outline(cel, S, w[0] * 1.6, L['line'], i['seed'] + 9, tremor=.1); cel.fill(S, sk[0]); F.base(); cel.clip(S)
        cel.fill(np.vstack([[sf(sg * (.95 + .05 * math.sin(y * 8)), y) for y in np.linspace(.02, .97, 9)], [A([H.cx + sg * 3 * H.s, H.cy - 2 * H.s]), A([H.cx + sg * 3 * H.s, H.cy])]]), sk[1]); cel.fill(spl(A([sf(-sg * .90, .55), sf(-sg * .70, .78), sf(-sg * .34, .90), sf(-sg * .44, .78), sf(-sg * .70, .60)]), True, n=7), tone(L, '#fff6e6', 0)); cel.unclip()
        F.lines(); F.nose(); F.contour(); fl = math.sin(i['f'] * .4) * .03 + i['wind'][0] * .12
        # beard and moustache: white blades
        
        def B3(P):
            _, ys, lo_, hi_ = H._xlim(); Q = []
            for x_, y_, dz in P:
                a_, b_ = float(np.interp(max(y_, -.80), ys, lo_)) * .97, float(np.interp(max(y_, -.80), ys, hi_)) * .97; x2 = min(max(x_, a_), b_); Q.append((x2, y_, float(H.z(np.clip(x2, -.62, .62), max(y_, -.95))) + dz * (1. if abs(x_ - x2) < .01 else .3)))
            return H.P3(A(Q))
        C = spl(B3([(-.50, -.40, .02), (-.565, -.62, .02), (-.50, -.86, .06), (-.40 + fl * .4, -1.06, .10), (-.34 + fl, -1.22, .10), (-.22 + fl, -1.14, .14), (-.14 + fl, -1.36, .14), (.0 + fl, -1.24, .16), (.12 + fl, -1.40, .14), (.22 + fl, -1.16, .14), (.36 + fl, -1.24, .10), (.42 + fl * .4, -1.04, .10), (.50, -.86, .06), (.565, -.62, .02), (.50, -.40, .02), (.40, -.50, .05), (.26, -.60, .09), (0, -.64, .12), (-.26, -.60, .09), (-.40, -.50, .05)]), True, corners=(4, 6, 8, 10), n=7)
        outline(cel, C, w[0] * 1.5, bd[4], i['seed'] + 2, tremor=.15); cel.fill(C, bd[0]); cel.clip(C); cel.fill(spl(B3([(sg * .12, -.60, .1), (sg * .26, -.80, .1), (sg * .10, -.95, .14), (sg * .26, -1.10, .1), (sg * .10, -1.5, .1), (sg * .8, -1.4, 0), (sg * .8, -.3, 0)]), True, corners=range(7), n=5), bd[1])
        for q in (-.40, -.20, .02, .22, .42): cel.line(spl(B3([(q, -.72, .10), (q * .9 + fl * .5, -.92, .12), (q * .8 + fl, -1.12, .12)]), n=7), w[2] * 1.2, bd[4], taper=(.3, .5), seed=int(q * 50))
        cel.unclip(); F.mouth()
        for q in (-1, 1):
            M_ = spl(B3([(q * .015, -.535, .14), (q * .16, -.515, .13), (q * .34, -.560, .10), (q * .48, -.70, .08), (q * .535, -.88, .06), (q * .44, -.80, .10), (q * .34, -.705, .12), (q * .16, -.655, .14), (q * .02, -.640, .15)]), True, corners=(0, 4, 8), n=8); outline(cel, M_, w[0] * 1.3, bd[4], i['seed'] + 4 + q); cel.fill(M_, bd[0]); cel.clip(M_); cel.fill(spl(B3([(q * .04, -.62, .14), (q * .2, -.60, .13), (q * .38, -.66, .12), (q * .60, -.92, .08), (q * .40, -.80, .10), (q * .1, -.72, .12)]), True, n=6), bd[1]); cel.line(spl(B3([(q * .10, -.56, .14), (q * .28, -.585, .13), (q * .46, -.70, .10)]), n=7), w[2] * 1.2, bd[4], taper=(.3, .5), seed=5 + q); cel.unclip()
            # white brows
            if H.vis(q * .30, .20): E_ = spl(H.Pc(A([(q * .10, .150), (q * .24, .215), (q * .42, .225), (q * .60, .140), (q * .46, .165), (q * .28, .150), (q * .14, .110)])), True, corners=(0, 3), n=7); poly(cel, E_, bd, w[1] * 1.2, i['seed'] + 8 + q)
        # sunglasses
        gl = cloth(L, GLASS); far = 1 if H.yaw > 0 else -1
        for q in (far, -far):
            if not H.vis(q * .20, .0): continue
            t = np.linspace(0, TAU, 40)[:-1]; G_ = lambda rx, ry, dz=.17: H.P3(np.c_[q * .315 + np.cos(t) * rx, -.015 + np.sin(t) * ry, np.full(39, float(H.z(.30, 0)) + dz - .30 * 0) - np.abs(q * .315 + np.cos(t) * rx) ** 2 * .42]); o_ = G_(.250, .205); cel.fill(o_, gl[0]); outline(cel, o_, w[0] * .95, L['line'], 3 + q); i_ = G_(.195, .155); cel.fill(i_, tone(L, '#1c1418', 0)); cel.clip(i_); cel.fill(i_ + A([-.16, -.20]) * H.s * .9, tone(L, '#4a3038', 0)); cel.fill(A([(-.03, -.25), (.03, -.25), (-.10, .25), (-.16, .25)]) * H.s + np.mean(i_, 0), tone(L, '#f6dfc8', 0), .75); cel.unclip(); outline(cel, i_, w[1], L['line'], 5 + q)
            e = H.ear(q)
            if e['vis'] > .05: a_, b_ = H.P3(A([(q * .55, .02, float(H.z(.5, 0)) + .12), (q * .70, .03, .02)])); cel.line(np.vstack([a_, (a_ + b_) / 2, b_]), w[0] * 1.6, L['line'], taper=(0, 0), press=0); cel.line(np.vstack([a_, (a_ + b_) / 2, b_]), w[0] * .8, gl[0], taper=(0, 0), press=0)
        p = H.P3(A([(-.075, .03, float(H.z(0, 0)) + .18), (0, .06, float(H.z(0, 0)) + .20), (.075, .03, float(H.z(0, 0)) + .18)])); cel.line(spl(p, n=8), w[0] * 1.8, L['line'], taper=(0, 0), press=0); cel.line(spl(p, n=8), w[0] * .9, gl[0], taper=(0, 0), press=0)
        return H

    def body(s, cel, H, B, L, w, seed, pose, f, arm_, inf):
        o = cloth(L, SHIRT, '#ffcf6a'); sg = -1 if L['ld'][0] > 0 else 1; bw = (1.15, .46); R_ = ((.34, -.10), (.80, .10), (1.04, .40), (1.02, .9), (.98, 1.5), (1.02, 2.1), (1.12, 3.4)); P = spl(np.vstack([B([(-x, y) for x, y in R_], bow=bw), B(R_, bow=bw)[::-1]]), True, corners=(0, 6, 7, 13), n=8); V = lambda Q, c=(): B.c(Q, True, corners=c, bow=bw)
        outline(cel, P, w[0] * 1.6, o[4], seed); cel.fill(P, o[0]); s.neck(cel, H, L, seed); cel.clip(P); r = np.random.default_rng(5)
        for j in range(16):                                         # printed flowers
            x, y = r.uniform(-.95, .95), r.uniform(.25, 3.2); c = B([(x, y)], bow=bw)[0]; k = .105 * B.u
            for a_ in range(5): th = a_ * TAU / 5 + j; cel.fill(ell(c[0] + math.cos(th) * k, c[1] + math.sin(th) * k, k * .72), tone(L, ('#fff0a0', '#d8b05a', '#a8803c'), 0))
            cel.fill(ell(c[0], c[1], k * .5), tone(L, '#d8402c', 0))
        cel.ctx.save(); cel.ctx.set_operator(cairo.OPERATOR_MULTIPLY); cel.fill(V([(sg * .14, .1), (sg * .42, .7), (sg * .20, 1.2), (sg * .44, 1.9), (sg * .3, 3.5), (sg * 1.4, 3.5), (sg * 1.4, .1)], range(7)), mixc('#ffffff', o[1], .85)); cel.ctx.restore()
        cel.fill(V([(-.40, -.3), (-.34, .10), (0, .62), (.34, .10), (.40, -.3)], (0, 2, 4)), tone(L, R_SKIN, 1)); cel.line(B.c([(0, .62), (.04, 1.6), (.02, 3.4)], False, bow=bw), w[1], o[4], seed=3); cel.unclip()
        for q in (-1, 1): K = V([(q * .30, -.24), (q * .56, -.10), (q * .66, .26), (q * .40, .40), (q * .05, .60), (q * .20, .20)], (0, 1, 2, 3, 4)); poly(cel, K, o, w[1] * 1.5, seed + q, sh=[K + A([sg * .16, .04]) * B.u])
        if pose == 'raise': s.rarm(cel, B, L, 1, (1.42, 1.28, .1), (1.40, 1.86, .75), w, seed + 4, R_SKIN, wd=(.27, .23, .19, .18, .135), sleeve=SHIRT, sh=(1.0, .42), hs=.01); s.staff(cel, B, L, w, seed)

    def staff(s, cel, B, L, w, seed):
        c = cloth(L, ('#a8703c', '#6c4322', '#472a14')); P = B([(1.42, -1.4, .7), (1.40, 1.0, .7), (1.36, 3.6, .7)]); q, a, b, cc = limb(P, [.085 * B.u] * 3); poly(cel, q, c, w[0] * .9, seed + 30, sh=[np.vstack([cc, b[::-1]])]); k = B.u; t = B([(1.42, -1.55, .7)])[0]; C = spl(A([(-.10, .25), (-.26, .02), (-.22, -.28), (.04, -.40), (.28, -.26), (.30, .02), (.12, .25)]) * k + t, True, n=8); poly(cel, C, c, w[0], seed + 31, sh=[C + A([.16, .05]) * k]); u = np.linspace(0, 1, 40); an = u * 9.5; cel.line(np.c_[t[0] + np.cos(an) * .18 * k * (1 - u), t[1] - .06 * k + np.sin(an) * .18 * k * (1 - u)], w[1], c[4], taper=(.1, .3), seed=4)
        hand(cel, L, R_SKIN, HD.POSES['grip'], B([(1.40, 1.72, .9)])[0], .40 * k, HD.euler(150, 0, 95), 'L', s.BLD, seed=seed, style=dict(arm=.1), pivot=(0, .55, -.2))

    def over(s, cel, H, B, L, w, seed, pose, f, arm_):
        if pose == 'raise': el, wr = arm_ or ((-1.46, .10, .1), (-1.24, -.98, .4)); s.rarm(cel, B, L, -1, el, wr, w, seed + 5, R_SKIN, wd=(.27, .23, .19, .20, .135), sleeve=SHIRT, sh=(1.0, .42), R=(182, -8, 6), pose=s.hp)


Roshi = _Roshi()

if __name__ == '__main__':
    bg = np.ones((1080, 1920, 3), f32) * rgb('#c8d4d8')
    for k, (C, e, l, y) in enumerate([(Bulma, 'smile', 'day', -.35), (Krillin, 'firm', 'day', .25), (Roshi, 'neutral', 'day', -.3)]): bg = comp.over(bg, C.bust(330 + 640 * k, 430, 175, y, .03, 0, e, l, seed=k))
    comp.save('qa/friends/_fr.jpg', bg)
