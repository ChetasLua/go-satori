"""SON GOHAN, about five.
    from chars.gohan import Gohan
    lay = Gohan.bust(cx, cy, s, yaw, pitch, roll, expr='laugh', light='day', f=0, wind=(0,0), seed=1, hat=True, pose='stand')   -> premultiplied RGBA 1080x1920
    Gohan.head(cel, H, info)                head_fn for a body engine: H = any db.head3.Head, info = dict(expr, light, f, wind, seed, hat, part='all'|'back'|'front')
    expr: neutral smile laugh wonder terror determined shout gentle shut worry;  (a, b, t) blends;  dicts override
    cx, cy = centre of the head on the eye line, s = px from eye line to chin of the master head
"""
from chars.fkit import *
from chars import fkit as K
from db import hands as HD

SKIN = ('#ffd9b6', '#e09a78', '#b0655a')
GI = ('#6b3fa0', '#45267a', '#2c1758'); GIH = '#8d5fc2'
SASH = ('#d8342c', '#9c1c26', '#661020'); SASHH = '#f2604a'
SCARF = ('#fbf6ee', '#c3bcd4', '#8f88ac'); SCARFH = '#ffffff'
HAT = ('#d8352b', '#9a1d27', '#650f20'); HATH = '#f4664a'; TRIM = ('#f5c64a', '#c07f2c', '#8a521f')
SHOE = ('#8a5a34', '#5c371f', '#3c2214')
HAIRC = dict(base=('#24222c', '#1c1a24', '#16141c'), sh=('#0d0c12', '#0a0910', '#08070c'), hi=('#565e7c', '#464c68', '#383d56'), sep=('#4a5070', '#3c4260', '#30354e'))

D = dict(
    skin=SKIN, blush0=.20, blushc='#f2826c', side=-.35, nose='kid', nshadow=.75, my=-.665,
    head=dict(a=[(-1.0, .15), (-.97, .23), (-.90, .345), (-.80, .475), (-.68, .595), (-.54, .70), (-.36, .78), (-.12, .81), (.12, .81), (.35, .80), (.6, .745), (.8, .60), (.93, .38), (1.0, .05)],
              f=[(-1.0, .66), (-.92, .72), (-.8, .755), (-.64, .78), (-.46, .79), (-.28, .785), (-.06, .775), (.06, .785), (.2, .83), (.4, .83), (.6, .76), (.8, .58), (.93, .34), (1.0, .05)],
              b=[(-1.0, .05), (-.9, .10), (-.72, .22), (-.5, .48), (-.2, .78), (.1, .95), (.4, 1.0), (.65, .92), (.85, .64), (.95, .36), (1.0, .05)],
              zj=[(-1.0, .62), (-.99, .60), (-.96, .52), (-.91, .40), (-.85, .27), (-.76, .115), (-.66, .01), (-.56, -.04), (-.44, -.08), (-.2, -.10)],
              p=2.2, nose=(.07, -.16, -.385, -.435, .105), ear=(.80, .04, -.48, -.08), cheek=.075, socket=.02, low=.775, hi=1.02, relief=.40),
    eye=dict(xi=.142, xo=.492, U=[.020, .165, .212, .175, .050], Lo=[-.125, -.200, -.222, -.198, -.085], ir=(.100, .172), ic=.465, iy=.012, iris='black',
             lt=[(0, .010), (.12, .026), (.40, .038), (.75, .050), (1, .044)], down=.62, lo0=.34, low=.95, ls=.038, cro=.048, corners=(), crease=0, cl=1.15),
    brow=dict(pts=[(.150, .300), (.265, .352), (.390, .352), (.505, .285)], th=[(0, .016), (.22, .044), (.55, .040), (.85, .020), (1, .002)], col='ink'))

E = dict(
    neutral=X(hw=.075, mU=[.006, .0, -.004, .0, .006], mL=[.006, 0, -.004, 0, .006]),
    smile=X(hw=.150, my=-.640, mU=[.050, .004, -.012, .004, .050], mL=[.050, -.010, -.034, -.010, .050], blush=.35, dB=.03, lip=1, mw=1.1, cheek=1),
    laugh=X(closed=1, arc=1, hw=.265, my=-.545, mU=[.060, .030, .008, .030, .060], mL=[.030, -.235, -.320, -.235, .030], tongue=1.1, teeth=.052, jaw=.13, blush=.6, dB=.055, aB=.0, lip=0),
    wonder=X(U=[.030, .195, .250, .205, .062], ir=.98, gy=.010, glint=1.25, dB=.075, hw=.085, my=-.655, mU=[-.005, .040, .050, .040, -.005], mL=[-.005, -.055, -.072, -.055, -.005], tongue=.7, jaw=.05, blush=.30, lip=1),
    terror=X(U=[.045, .215, .265, .215, .070], Lo=[-.13, -.215, -.240, -.215, -.100], ir=.50, glint=.55, B=[(.120, .440), (.255, .445), (.400, .410), (.520, .300)], furrow=1., hw=.165, my=-.615,
             mU=[-.020, .030, .040, .030, -.020], mL=[-.050, -.165, -.190, -.165, -.050], teeth=0., tongue=.6, wob=.55, jaw=.09, sweat=1., tears=.45, under=1., blush=-.2, lip=0),
    determined=X(U=[-.010, .115, .165, .160, .060], Lo=[-.115, -.185, -.205, -.180, -.075], ir=.92, B=[(.112, .165), (.250, .250), (.395, .335), (.525, .365)], furrow=1., hw=.098, my=-.655,
                 mU=[-.012, .003, .008, .003, -.012], mL=[-.012, .003, .008, .003, -.012], mw=1.25, blush=.1),
    shout=X(U=[-.010, .120, .170, .165, .060], ir=.86, B=[(.112, .160), (.250, .250), (.395, .345), (.525, .375)], furrow=1., hw=.205, my=-.555, mU=[.020, .042, .030, .042, .020], mL=[-.030, -.260, -.325, -.260, -.030],
            teeth=.055, tongue=.9, jaw=.15, blush=.15, lip=0),
    gentle=X(U=[.0, .120, .155, .125, .030], gy=-.020, hw=.115, my=-.645, mU=[.024, -.002, -.010, -.002, .026], mL=[.024, -.002, -.012, -.002, .026], blush=.35, dB=.02),
    worry=X(B=[(.130, .410), (.262, .405), (.395, .370), (.515, .285)], furrow=.4, hw=.080, my=-.665, mU=[-.014, .006, .010, .006, -.014], mL=[-.014, .004, .008, .004, -.014], gy=.005),
    shut=X(closed=1, arc=0, hw=.085, mU=[.012, 0, -.004, 0, .012], mL=[.012, 0, -.004, 0, .012], blush=.25))


def ex(e):
    if isinstance(e, str): return E[e]
    if isinstance(e, (tuple, list)): return blend(ex(e[0]), ex(e[1]), e[2])
    return X(**e) if set(e) <= set(X0) else e


# ------------------------------------------------------------------ hair: a round bowl with pointed ends
def locks(H, hat=False):
    L = []; n = [0]
    def sp(th, y, d, ln, w, curl=(0, 0, 0), arc=(0, 0, 0), **k): n[0] += 1; L.append(Lock.sp(H, th, y, d, ln, w, curl, arc, id=n[0], **k))
    def fr(th, y, tip, w, **k): n[0] += 1; L.append(Lock.fr(H, th, y, tip, w, id=n[0], tag='fr', **k))
    # fringe: pointed clumps of three sizes, parted a little right of centre; the forehead and brows show between them
    fr(-.98, .66, (-.655, .020), .28, asym=-.12); fr(1.00, .66, (.655, .070), .28, asym=.12)
    fr(-.62, .78, (-.520, .215), .30, asym=-.22, bulge=.22); fr(.72, .78, (.555, .250), .28, asym=.22, bulge=.22)
    fr(-.22, .87, (-.235, .245), .30, asym=-.18, bulge=.24); fr(.36, .86, (.300, .330), .26, asym=.16, bulge=.24)
    fr(-.42, .84, (-.395, .385), .18, dz=.09, asym=-.1); fr(.06, .90, (.020, .345), .22, dz=.08, asym=-.1); fr(.16, .90, (.130, .430), .14, dz=.10); fr(.54, .83, (.430, .400), .16, dz=.10); fr(-.10, .90, (-.105, .440), .13, dz=.11)
    fr(-.82, .72, (-.600, .260), .14, dz=.10); fr(.86, .72, (.615, .280), .13, dz=.10)
    # sides: over the temples, in front of and behind the ears; they hang and flick out only at the tips
    sp(-1.10, .45, (-.10, -1, .16), .66, .30, (.05, 0, .04), kind='flip', tag='side'); sp(1.12, .45, (.10, -1, .16), .62, .30, (-.05, 0, .04), kind='flip', tag='side')
    sp(-1.34, .42, (-.20, -1, .04), .76, .34, (-.06, .05, 0), (-.06, 0, 0), kind='flip', tag='side'); sp(1.36, .42, (.20, -1, .04), .74, .34, (.06, .05, 0), (.06, 0, 0), kind='flip', tag='side')
    sp(-1.55, .40, (-.12, -1, -.06), .84, .40, (-.10, .08, 0), (-.09, 0, 0), kind='flip'); sp(1.56, .40, (.12, -1, -.06), .82, .40, (.10, .08, 0), (.09, 0, 0), kind='flip')
    sp(-1.50, .62, (-.34, -1, .0), .60, .38, (-.06, .04, 0), (-.10, .08, 0), kind='flip'); sp(1.51, .62, (.34, -1, .0), .58, .38, (.06, .04, 0), (.10, .08, 0), kind='flip')
    for th, y, d, ln, w, c in ((1.85, .34, (.10, -1, -.06), .84, .42, (.10, .08, 0)), (-1.85, .34, (-.10, -1, -.06), .86, .42, (-.10, .08, 0)), (2.25, .28, (.06, -1, -.06), .86, .44, (.06, .06, -.06)), (-2.25, .28, (-.06, -1, -.06), .88, .44, (-.06, .06, -.06)),
                               (2.65, .24, (.02, -1, -.06), .86, .44, (.04, .06, -.08)), (-2.65, .24, (-.02, -1, -.06), .86, .44, (-.04, .06, -.08)), (3.14, .24, (0, -1, -.06), .86, .44, (0, .06, -.10)),
                               (2.05, .30, (.16, -1, -.12), .80, .26, (.12, .12, -.04)), (-2.05, .30, (-.16, -1, -.12), .82, .26, (-.12, .12, -.04)), (2.45, .26, (.08, -1, -.16), .80, .26, (.06, .12, -.14)), (-2.45, .26, (-.08, -1, -.16), .82, .26, (-.06, .12, -.14)),
                               (1.80, .66, (.30, -1, -.16), .60, .38, (.08, .02, -.02)), (-1.80, .66, (-.30, -1, -.16), .62, .38, (-.08, .02, -.02)), (2.35, .64, (.20, -1, -.35), .62, .42, (.04, 0, -.06)), (-2.35, .64, (-.20, -1, -.35), .62, .42, (-.04, 0, -.06)), (3.14, .64, (0, -1, -.40), .62, .44, (0, 0, -.08))):
        sp(th, y, d, ln, w, c, (c[0] * .7, .06, c[2] * .7), kind='flip')
    if hat: L = [l for l in L if l.tag == 'fr' or l.p0[1] < .56]
    if not hat:                                                        # crown: round mass, short stubborn tufts
        sp(-.70, .86, (-.95, .42, .10), .40, .38, (-.10, -.18, 0), (0, .08, 0)); sp(.80, .86, (.95, .38, .08), .40, .38, (.10, -.18, 0), (0, .08, 0))
        sp(-.30, .95, (-.70, .80, .05), .34, .30, (-.14, -.10, 0), (0, .04, 0)); sp(.42, .95, (.62, .85, .04), .36, .30, (.16, -.10, 0), (0, .04, 0)); sp(.04, .98, (.22, 1, -.05), .30, .26, (.12, -.02, 0))
        sp(-1.15, .78, (-1, -.05, .04), .44, .38, (-.02, -.18, 0), (-.04, .10, 0)); sp(1.18, .78, (1, -.05, .04), .44, .38, (.02, -.18, 0), (.04, .10, 0))
        for th, y, d, ln, w, c in ((1.9, .86, (.8, .25, -.5), .42, .42, (.04, -.18, -.05)), (-1.9, .86, (-.8, .25, -.5), .42, .42, (-.04, -.18, -.05)), (2.7, .88, (.3, .3, -.9), .42, .44, (0, -.18, -.06)), (-2.7, .88, (-.3, .3, -.9), .42, .44, (0, -.18, -.06))): sp(th, y, d, ln, w, c, (0, .06, 0))
    return L


def hl(t): return .60 - .30 * min(abs(t) / 1.05, 1) ** 2 - .72 * max(0., min((abs(t) - 1.30) / .8, 1.))


def build_locks(H, f=0, wind=(0, 0), hat=False):
    L = locks(H, hat)
    for l in L: l.build(H, wind_off(H, wind, f, (l.id * .618) % 1, {'fr': .6, 'side': .75}.get(l.tag, 1.)))
    return L


def isback(l): return l.tag != 'fr' and (l.vis < -.20 or l.depth < -.36)


def hair_cols(L):
    return dict(base=tone(L, HAIRC['base'], 0), sh=tone(L, HAIRC['sh'], 1), hi=tone(L, HAIRC['hi'], 0), sep=tone(L, HAIRC['sep'], 1), hip=.75, hiw=1.0)


# ------------------------------------------------------------------ the hat
HP = [(.50, .80, .84, -.04), (.435, .87, .91, -.04), (.465, .95, .995, -.04), (.57, .975, 1.02, -.04), (.66, .945, .99, -.05), (.69, .905, .95, -.05)]
HC = [(.62, .905, .95, -.05), (.80, .89, .93, -.05), (1.00, .80, .84, -.06), (1.18, .63, .66, -.07), (1.31, .43, .45, -.08), (1.39, .21, .23, -.08), (1.41, .02, .02, -.08)]
BALL = (A([0, 1.50, .06]), .285)


def star(c, r, rot=0., sq=1.):
    t = np.arange(10) * math.pi / 5 - math.pi / 2 + rot; k = np.where(np.arange(10) % 2 == 0, r, r * .42); return np.c_[c[0] + np.cos(t) * k * sq, c[1] + np.sin(t) * k]


def ball(cel, c, r, L, seed=0, rot=0., glow=1., line=True):
    """the four-star ball, small (lane 'kid' owns the hero version, db/ball.py)"""
    x, y = c; E = ell(x, y, r, n=64); base, sh, dp = (tone(L, ('#ffab24', '#e8761a', '#b04a14'), i) for i in range(3)); k = glow
    cel.fill(E, mixc(base, '#ffab24', .45 * k)); cel.clip(E); ls = 1 if L['ld'][0] > 0 else -1
    rad(cel, None, (x + ls * r * .05, y - r * .02), r * 1.05, mixc(tone(L, '#ffd24a', 0), '#ffe27a', .5 * k), .85); rad(cel, None, (x - ls * r * .30, y + r * .42), r * .62, '#fff0a8', .55 * k + .15)
    cel.fill(np.vstack([ell(x, y, r * 1.01, a0=-.35 * math.pi if ls > 0 else .15 * math.pi, a1=.85 * math.pi if ls > 0 else 1.35 * math.pi, n=30)[::1], ell(x - ls * r * .22, y - r * .22, r * .98, a0=-.35 * math.pi if ls > 0 else .15 * math.pi, a1=.85 * math.pi if ls > 0 else 1.35 * math.pi, n=30)[::-1]]), mixc(dp, '#d2560e', .5 * k), .55)
    st = mixc(tone(L, '#d8201c', 0), '#e2261c', .4 * k)
    for dx, dy in ((-.30, -.22), (.26, -.28), (-.22, .30), (.32, .24)): cel.fill(star((x + dx * r * 1.0, y + dy * r * 1.0), r * .215, .15 * dx), st)
    cel.fill(spl(A([(-.62, -.30), (-.42, -.60), (-.10, -.72), (-.30, -.50), (-.50, -.26)]) * [ls * -1 * -1 if False else (1 if ls < 0 else -1), 1] * r + [x, y], True, n=6), '#fffbe6', .95); cel.fill(ell(x + ls * -.50 * r * -1 * (1 if ls > 0 else 1) * .0 + (-.62 if ls < 0 else .62) * r * .92, y + .02 * r, r * .07, r * .10), '#fffbe6', .9)
    cel.unclip()
    if line: cel.line(np.vstack([E, E[:1]]), max(1., lw(r * 3)[0] * .75), mixc(L['line'], '#5a2408', .5), taper=(.03, .03), press=.25, seed=seed)
    return cel


def ball_at(H, lift=0.):
    """px centre and radius of the ball on the hat"""
    c3, r3 = BALL; c = H.P3(c3 + [0, lift, 0]); return float(c[0]), float(c[1]), r3 * H.s


def hat_parts(cel, H, L, part='front', seed=0, glow=.6, lift=0., tip=0., ballon=True):
    """part: 'front' crown + brim + ball.  lift = rises off the head (units), tip = tilt"""
    w = lw(H.s); cr = cloth(L, HAT, HATH); tr = cloth(L, TRIM, '#fff0a8'); R = Rev(H, [(y + lift, a, b, c) for y, a, b, c in HC], tilt=-.13 + tip); B = Rev(H, [(y + lift, a, b, c) for y, a, b, c in HP], tilt=-.13 + tip)
    sg = -1 if L['ld'][0] > 0 else 1; a0 = -H.yaw
    S = R.sil(); outline(cel, S, w[0] * 1.7, cr[4], seed, tremor=.15); cel.fill(S, cr[0]); cel.clip(S)
    th = a0 + sg * np.linspace(.42, 1.7, 10); v = np.linspace(0, 1, 9)
    cel.fill(np.vstack([R.patch(a0 + sg * (.42 + .16 * np.sin(v * 7)), v), R.patch(a0 + sg * 1.62, v[::-1])]), cr[1])
    cel.fill(np.vstack([R.patch(a0 - sg * (.85 + .10 * np.sin(v * 6)), v * .7 + .06), R.patch(a0 - sg * 1.12, (v * .7 + .06)[::-1])]), cr[3])
    for t in (-2.2, -1.1, 0., 1.1, 2.2, 3.14):
        for q in R.curve(np.full(24, t), np.linspace(.0, .93, 24)): cel.line(q, w[1] * .8, cr[4], taper=(.1, .3), seed=seed + int(t * 10), press=.25)
    for t in np.linspace(-3, 3, 13):
        for q in R.curve(t + np.linspace(0, .05, 8), np.linspace(.0, .20, 8)): cel.line(q, w[2], cr[4], taper=(.05, .8), seed=seed + 40)
    cel.unclip()
    # rolled brim
    S = B.sil(every=True, span=1.72)
    for q in S: outline(cel, q, w[0] * 1.6, cr[4], seed + 1, tremor=.15)
    cel.fill(S, cr[0], evenodd=True); clip_eo(cel, S)
    t = np.linspace(-math.pi, math.pi, 120)
    for q in B.curve(t, np.full(120, .30)):
        pass
    for q0, q1 in zip(B.curve(t, np.full(120, .16)), B.curve(t, np.full(120, .50))):
        n = min(len(q0), len(q1)); cel.fill(np.vstack([q0[:n], q1[:n][::-1]]), cr[1])
    v = np.linspace(0, 1, 6); cel.fill(np.vstack([B.patch(a0 + sg * (.50 + .10 * np.sin(v * 9)), v), B.patch(a0 + sg * 1.62, v[::-1])]), cr[1])
    for q in B.curve(t, np.full(120, .84)): cel.line(q, max(1.6, .052 * H.s), tr[0], taper=(.02, .02), press=.1, seed=seed); cel.line(q + [0, .016 * H.s], max(.8, .016 * H.s), tr[1], taper=(.02, .02), press=.1, seed=seed)
    for q in B.curve(t, np.full(120, .42)): cel.line(q, w[2] * 1.1, cr[4], taper=(.1, .1), press=.2, seed=seed + 9, alpha=.75)
    for tt in np.linspace(-3., 3., 15):
        for q in B.curve(tt + np.linspace(0, .09, 8), np.linspace(.26, .58, 8)): cel.line(q, w[2], cr[4], taper=(.05, .8), seed=seed + 50)
    cel.unclip()
    # ball in its gold claw
    c3, r3 = BALL; c = H.P3(c3 + [0, lift, -.10 * 0]); r = r3 * H.s; base = H.P3(c3 + [0, lift - r3 * .75, 0])
    b0 = H.P3(c3 + [0, lift - r3 * .86, 0]); up = H.P3(c3 + [0, lift, 0]) - b0; up = up / (np.hypot(*up) + 1e-6); rt = A([-up[1], up[0]]); k = H.s
    if ballon: ball(cel, c, r, L, seed, glow=glow)
    a = np.linspace(0, math.pi, 13); E1 = lambda h, rx, ry: b0 + up * h * k + np.cos(a)[:, None] * rt * rx * k + -np.sin(a)[:, None] * up * ry * k
    G = np.vstack([E1(.055, .215, .050), E1(-.055, .235, .062)[::-1]]); cel.fill(G, tr[0]); cel.clip(G); cel.fill(np.vstack([E1(-.02, .23, .06), E1(-.10, .24, .07)[::-1]]), tr[1]); cel.fill(G + [-sg * .30 * k, .0], tr[3], .0); cel.unclip(); outline(cel, G, w[1] * 1.1, L['line'], seed + 5)
    return c, r


class _Gohan:
    D = D; skin = SKIN
    def make(s, cx, cy, sc, yaw=0., pitch=0., roll=0., persp=.10): return FHead(D).pose(cx, cy, sc, yaw, pitch, roll, persp)

    def head(s, cel, H, info=None, **k):
        """draw the head on a Cel.  part: 'back' (hair behind), 'front' (face, front hair, hat), 'all'"""
        i = dict(expr='neutral', light='day', f=0, wind=(0, 0), seed=1, hat=True, part='all', wet=0., glow=.6, hatlift=0.); i.update(info or {}); i.update(k)
        x = ex(i['expr']); L = light(i['light']); H = FHead.of(H, D, x['jaw']); w = lw(H.s); LK = build_locks(H, i['f'], i['wind'], i['hat'] and i['hatlift'] == 0); hc = hair_cols(L)
        if i['part'] in ('all', 'back'):
            bk = sorted([l for l in LK if isback(l)], key=lambda l: l.depth); hair_group(cel, bk, dict(hc, base=hc['sh'], hi=None), w, i['seed'], L['ld'], caps=[scalp(H, lambda t: -.50, 1.07, front=False)], line=L['line'])
        if i['part'] == 'back': return H
        fr = sorted([l for l in LK if not isback(l)], key=lambda l: l.depth)
        F = Face(cel, H, D, x, L, i['seed'], i['wet'], hairs=[l.poly for l in fr if l.tag in ('fr', 'side')])
        F.ears_far(); F.ears_near(); F.base(); F.features(); F.contour()
        hair_group(cel, fr, hc, w, i['seed'], L['ld'], caps=[scalp(H, hl, 1.07)], line=L['line'])
        # brows read over the fringe in anime: redraw them softly
        if i['hat']: hat_parts(cel, H, L, 'front', i['seed'], i['glow'], i['hatlift'], ballon=i.get('ballon', True))
        return H

    # ------------------------------------------------------------------ body
    KID = HD.build(len=.84, wid=1.10, boxy=.12, tip=.15, knuckle=.55, palm=1.04, wrist=.95, meat=1.20)

    def neck(s, cel, H, L, seed=0):
        w = lw(H.s); c = cloth(L, SKIN); sg = -1 if L['ld'][0] > 0 else 1; a, b = [H.P3(A([(q * .34, -.55, -.10), (q * .32, -.95, -.08), (q * .36, -1.25, -.08), (q * .60, -1.5, -.10)])) for q in (-1, 1)]
        P = np.vstack([spl(a, n=8), spl(b, n=8)[::-1]]); cel.fill(P, c[0]); cel.clip(P); J = np.vstack([H.jaw(-1), H.jaw(1)[::-1][1:]]) + A([sg * .05, .22]) * H.s; cel.fill(np.vstack([J, H.P3(A([(1.2, .3, 0), (-1.2, .3, 0)]))]), c[1]); cel.unclip()
        for i, q in enumerate((a, b)): cel.line(spl(q, n=8), w[0] * .8, c[4], taper=(.25, .1), seed=seed + i)

    def torso(s, cel, B, L, w, seed=0, bottom=2.7):
        g = cloth(L, GI, GIH); o = cloth(L, SASH, SASHH); sg = -1 if L['ld'][0] > 0 else 1; bw = (1.05, .42)
        R_ = [(.40, -.05), (.72, .10), (.90, .36), (.82, .72), (.78, 1.02), (.78, 1.32), (.98, 1.62), (1.18, bottom)]; P = np.vstack([B([(-x, y) for x, y in R_], bow=bw), B(R_, bow=bw)[::-1]]); C = spl(P, True, corners=(0, 7, 8, 15), n=8)
        outline(cel, C, w[0] * 1.6, g[4], seed, tremor=.12); cel.fill(C, g[0]); cel.clip(C); V = lambda L_: B.c(L_, False, bow=bw)
        cel.fill(B.c([(sg * .28, .2), (sg * .46, .6), (sg * .28, .9), (sg * .46, 1.2), (sg * .30, 1.6), (sg * .5, bottom + .1), (sg * 1.4, bottom + .1), (sg * 1.4, .1)], corners=range(8), bow=bw), g[1])
        cel.fill(V([(-.6, .2), (-.40, .55), (-.02, .80), (.40, .55), (.6, .2), (.36, .40), (0, .62), (-.36, .40)]), g[1])        # scarf shadow
        for j, pts in enumerate(([(-.70, .45), (-.48, .78), (-.20, 1.0), (-.10, 1.22)], [(.70, .45), (.50, .8), (.24, 1.04), (.10, 1.24)], [(-.76, .85), (-.55, 1.05), (-.36, 1.22)], [(.76, .85), (.55, 1.05), (.38, 1.22)], [(-.60, 1.62), (-.78, 2.0), (-.74, 2.5)], [(.55, 1.62), (.76, 2.0), (.74, 2.5)], [(-.25, 1.64), (-.36, 2.0), (-.30, 2.5)], [(.10, 1.7), (.14, 2.1), (.06, 2.6)])): cel.line(V(pts), w[1] * .9, g[4], taper=(.1, .6), press=.3, seed=seed + j)
        if bottom > 3.6:                                           # baggy trousers: the split of the legs and what hangs from the sash
            cel.fill(B.c([(-.10, 2.5), (.10, 2.5), (.20, 3.4), (.14, bottom + .2), (-.18, bottom + .2), (-.16, 3.4)], corners=(0, 1, 3, 4), bow=bw), g[2])
            for pts, ww in (([(.0, 1.9), (.03, 2.6), (.0, 3.4), (.02, bottom)], 1.3), ([(-.35, 1.66), (-.52, 2.1), (-.50, 2.7), (-.62, 3.4)], 1.), ([(.40, 1.66), (.56, 2.2), (.52, 2.8), (.66, 3.5)], 1.), ([(-.12, 2.4), (-.40, 2.75), (-.80, 2.85)], .9), ([(.14, 2.5), (.45, 2.9), (.86, 2.95)], .9), ([(-.20, 3.3), (-.48, 3.6), (-.85, 3.7)], .8), ([(.22, 3.4), (.55, 3.75), (.90, 3.8)], .8)): cel.line(V(pts), w[1] * ww, g[4], taper=(.1, .5), press=.3, seed=seed + int(pts[0][1] * 10))
        O = np.vstack([V([(-1.0, 1.20), (-.4, 1.28), (0, 1.30), (.4, 1.28), (1.0, 1.20)]), V([(1.1, 1.54), (.4, 1.63), (0, 1.65), (-.4, 1.63), (-1.1, 1.54)])]); cel.fill(O, o[0]); cel.clip(O); cel.fill(B.c([(sg * .22, 1.1), (sg * .34, 1.44), (sg * .20, 1.8), (sg * 1.3, 1.8), (sg * 1.3, 1.1)], corners=range(5), bow=bw), o[1]); cel.unclip(); outline(cel, O, w[1] * 1.4, g[4], seed + 9)
        for y in (1.38, 1.49): cel.line(V([(-.7, y - .04), (0, y + .04), (.7, y - .04)]), w[2], g[4], taper=(.2, .2), seed=seed + int(y * 10))
        cel.unclip()
        K_ = V([(-sg * .28, 1.66), (-sg * .10, 1.64), (-sg * .08, 1.88), (-sg * .28, 1.90)]); 
        for pts in ([(-sg * .22, 1.84), (-sg * .42, 2.2), (-sg * .46, 2.55), (-sg * .28, 2.50), (-sg * .14, 2.15), (-sg * .14, 1.86)], [(-sg * .14, 1.84), (-sg * .16, 2.2), (-sg * .10, 2.48), (sg * .06, 2.40), (-sg * .02, 2.1), (-sg * .08, 1.84)]):
            T = B.c([(x, y - .32) for x, y in pts], True, corners=(2, 3), bow=bw); poly(cel, T, o, w[1] * 1.3, seed + 4, sh=[T + A([sg * .10, 0]) * B.u * .8])
        T = B.c([(-sg * .30, 1.32), (-sg * .08, 1.30), (-sg * .05, 1.48), (-sg * .10, 1.60), (-sg * .28, 1.60), (-sg * .33, 1.46)], True, bow=bw); poly(cel, T, o, w[1] * 1.4, seed + 5, sh=[T + A([sg * .09, .05]) * B.u])

    def scarf(s, cel, B, L, w, seed=0, wind=(0, 0), f=0, part='front'):
        c = cloth(L, SCARF, SCARFH); sg = -1 if L['ld'][0] > 0 else 1; u = B.u
        if part == 'back':                                             # one end flies behind
            wx, wy = wind; d = -1 if wx < 0 else 1; m = min(1., math.hypot(wx, wy)); fl = math.sin(f * .55) * .10 * m; x0 = d * .45
            C = B(A([(x0, .02, -.3), (x0 + d * (.45 + .3 * m), .25 - .35 * m - fl, -.4), (x0 + d * (.6 + .9 * m), .85 - .9 * m + fl, -.5), (x0 + d * (.7 + 1.4 * m), 1.5 - 1.5 * m - fl * 1.4, -.5)])); P, a, b, cc = limb(C, [.20 * u, .24 * u, .27 * u, .30 * u]); poly(cel, P, c, w[0] * .95, seed + 60, sh=[np.vstack([cc, b[::-1]])], lines=[(cc[int(len(cc) * .1):int(len(cc) * .8)], .5)]); return
        P = [(-.62, -.30, .05), (-.84, -.10, .10), (-.86, .14, .20), (-.66, .40, .32), (-.28, .58, .44), (.10, .62, .48), (.46, .52, .42), (.78, .28, .26), (.88, .02, .14), (.80, -.18, .06), (.60, -.32, .05), (0, -.25, .2)]
        C = spl(B(P), True, n=8); outline(cel, C, w[0] * 1.6, c[4], seed, tremor=.15); cel.fill(C, c[0]); cel.clip(C); V = lambda L_: spl(B(L_), n=8)
        cel.fill(V([(-.9, -.4, .1), (-.5, .04, .4), (0, .16, .5), (.5, .04, .4), (.9, -.4, .1), (.9, -.6, 0), (-.9, -.6, 0)]), c[1])
        cel.fill(V([(sg * .95, -.1, .1), (sg * .62, .06, .3), (sg * .50, .30, .4), (sg * .16, .40, .5), (sg * .05, .66, .5), (sg * .6, .7, .4), (sg * 1.0, .3, .2)]), c[1])
        cel.fill(V([(-sg * .70, .12, .3), (-sg * .40, .34, .45), (sg * .05, .40, .5), (-sg * .02, .46, .5), (-sg * .42, .46, .45), (-sg * .74, .24, .3)]), c[1])
        for j, pts in enumerate(([(-.80, -.02, .2), (-.48, .22, .4), (-.04, .32, .5), (.30, .28, .46)], [(.82, -.06, .2), (.52, .12, .4), (.22, .20, .46)], [(-.78, .18, .25), (-.46, .42, .42), (-.10, .50, .48)], [(.72, .20, .3), (.48, .38, .42), (.22, .46, .46)], [(-.2, .10, .48), (.1, .22, .5), (.44, .20, .44)])): cel.line(V(pts), w[1] * .95, c[4], taper=(.12, .5), press=.3, seed=seed + j)
        cel.unclip()

    def put_arm(s, cel, B, L, q, el, wr, hp, R, w, seed=0, over='f', hs=.335, sleeve=True, hand_=True, elpx=None, wrpx=None, wd=(.24, .215, .18, .19, .145)):
        """q = -1 / +1 (screen left / right arm).  el, wr in body units (x, y, z)"""
        u = B.u; sg = -1 if L['ld'][0] > 0 else 1; sk = cloth(L, SKIN); g = cloth(L, GI, GIH); sh = B([(q * .80, .38, 0)])[0]; el = B([el])[0] if elpx is None else A(elpx, float); wr = B([wr])[0] if wrpx is None else A(wrpx, float); d = el - wr; d = d / np.hypot(*d)
        arm(cel, sh, el, wr, [v * u for v in wd], sk, w, sg, seed, over)
        if hand_: hand(cel, L, SKIN, hp, wr - d * .02 * u, hs * u, R, 'R' if q < 0 else 'L', s.KID, seed=seed, style=dict(arm=.35), marks=.6)
        band(cel, wr + d * .0 * u, d, .165 * u, .26 * u, cloth(L, SASH, SASHH), w, seed, sg=sg)
        if sleeve:                                                     # short gi sleeve: round shoulder, flared hem with its dark inside
            t = el - sh; k = t / (np.hypot(*t) + 1e-6); n = A([-k[1], k[0]]); ln = min(np.hypot(*t) * .58, .46 * u); X_ = lambda a_, b_: sh + k * a_ * ln + n * b_ * u
            hem = [X_(1.0 - .16 * (1 - v * v), v * .37) for v in np.linspace(-1, 1, 9)]; hb = [X_(1.0 + .17 * (1 - v * v), v * .37) for v in np.linspace(-1, 1, 9)]
            cel.fill(A(hem + hb[::-1]), g[2]); cel.line(A(hem), w[1], g[4], taper=(.05, .05), seed=seed)
            P = spl(A([X_(-.62, -.10), X_(-.50, -.26), X_(-.05, -.335), X_(.5, -.34), X_(1.0, -.37)]), n=7); Q = spl(A([X_(1.0, .37), X_(.5, .34), X_(-.05, .335), X_(-.50, .26), X_(-.62, .10)]), n=7); C = np.vstack([P, spl(A(hb), n=5)[1:-1], Q]); s_ = 1 if n[0] * sg > 0 else -1
            cel.fill(C, g[0]); cel.clip(C); cel.fill(A([X_(-.7, s_ * .10), X_(-.2, s_ * .14), X_(.15, s_ * .04), X_(.5, s_ * .20), X_(1.2, s_ * .08), X_(1.3, s_ * .6), X_(-.7, s_ * .6)]), g[1]); cel.fill(A([X_(-.3, -s_ * .20), X_(.3, -s_ * .26), X_(.85, -s_ * .24), X_(.3, -s_ * .18)]), g[3])
            for x0, y0, x1, y1 in ((1.05, -.22, .45, -.12), (1.12, .06, .55, .02), (1.02, .26, .50, .18), (-.1, -.30, .25, -.12)): cel.line(spl(A([X_(x0, y0), X_((x0 + x1) / 2, (y0 + y1) / 2 + .03), X_(x1, y1)]), n=6), w[2] * 1.3, g[4], taper=(.05, .7), seed=seed + 3)
            cel.unclip(); cel.line(C, w[0] * .95, g[4], taper=(.06, .06), press=.2, seed=seed + 9, tremor=.12)

    POSE = dict(stand=[(-1, (-1.06, 1.05, 0), (-1.02, 1.75, .25), 'fist', (-70, 0, 180), 0), (1, (1.06, 1.05, 0), (1.02, 1.75, .25), 'fist', (70, 0, 180), 0)],
                belly=[(-1, (-1.20, .92, .2), (-.40, 1.06, .75), 'rest', (-10, 10, -80), 1), (1, (1.20, .92, .2), (.42, 1.12, .75), 'rest', (10, 10, 80), 1)],
                raise1=[(1, (1.06, 1.05, 0), (1.00, 1.72, .25), 'fist', (70, 0, 180), 0), (-1, (-1.28, .02, .1), (-1.22, -.92, .3), 'open', (180, -6, 8), 1)],
                raise2=[(-1, (-1.32, .00, .1), (-1.26, -.95, .3), 'open', (180, -6, 8), 1), (1, (1.32, .00, .1), (1.26, -.95, .3), 'open', (180, -6, -8), 1)],
                fear=[(-1, (-.86, 1.12, .4), (-.30, .66, .85), 'fist', (-150, 20, -38), 1), (1, (.86, 1.12, .4), (.30, .70, .85), 'fist', (150, 20, 38), 1)],
                cheer=[(-1, (-1.20, .55, .1), (-1.05, -.35, .4), 'fist', (200, 10, 8), 1), (1, (1.20, .55, .1), (1.05, -.35, .4), 'fist', (160, 10, -8), 1)])

    def bust(s, cx, cy, sc, yaw=0., pitch=0., roll=0., expr='neutral', light='day', f=0, wind=(0, 0), seed=1, hat=True, pose='stand', byaw=None, wet=0., glow=.6, w=W, h=FH, rims=True, bottom=2.7, broll=None, over=None, ballon=True):
        L = globals()['light'](light); cel = Cel(w, h); H = s.make(cx, cy, sc, yaw, pitch, roll); wl = lw(sc); o = H.P3(A([(0, -1.30, -.05)]))[0]; B = Body(o[0], o[1], sc, yaw * .5 if byaw is None else byaw, 0., roll * .3 if broll is None else broll); inf = dict(expr=expr, light=light, f=f, wind=wind, seed=seed, hat=hat, wet=wet, glow=glow, ballon=ballon)
        s.head(cel, H, inf, part='back'); s.scarf(cel, B, L, wl, seed, wind, f, 'back'); arms = s.POSE[pose] if isinstance(pose, str) else pose
        for a_ in arms:
            if not a_[5]: s.put_arm(cel, B, L, a_[0], a_[1], a_[2], a_[3], HD.euler(*a_[4]), wl, seed + a_[0] + 2)
        s.neck(cel, H, L, seed); s.torso(cel, B, L, wl, seed, bottom); s.scarf(cel, B, L, wl, seed, wind, f)
        if pitch > -.3 and not (pose == 'raise'): pass
        for a_ in arms:
            if a_[5]: s.put_arm(cel, B, L, a_[0], a_[1], a_[2], a_[3], HD.euler(*a_[4]), wl, seed + a_[0] + 2)
        s.head(cel, H, inf, part='front')
        s.last = (H, B)
        if over: over(cel, H, B)
        lay = cel.layer(h, w); return grade(lay, L, (cx - 1.6 * sc, cy - 1.6 * sc, cx + 1.6 * sc, cy + 3.6 * sc), max(.5, sc / 160), rims=rims)


Gohan = _Gohan()
head_fn = Gohan.head

if __name__ == '__main__':
    bg = np.ones((1080, 1920, 3), f32) * rgb('#c8d4d8')
    for k, (e, y, p, l, hat, po, wd) in enumerate([('laugh', .30, .18, 'lake', 1, 'belly', (0, 0)), ('terror', -.25, 0, 'night', 0, 'fear', (-.6, .1)), ('determined', .3, .08, 'dawn', 0, 'raise2', (.7, .1))]):
        bg = comp.over(bg, Gohan.bust(330 + 640 * k, 430, 180, y, p, -.06 if e == 'laugh' else 0, e, l, hat=bool(hat), pose=po, seed=k, wind=wd))
    comp.save('qa/friends/_gohb.jpg', bg)
