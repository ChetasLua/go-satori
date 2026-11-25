"""B4  THE LAST BUN.  80 f.  0-39 straight down on the table, bowls on 8ths then 16ths; 40 the bun; 58 chopsticks meet; 66 the lamp; 70 it is gone.  76 a stomach.
(70-71: look at the right edge, behind the bowls.)"""
import numpy as np, cv2, math
from db import W, H, comp
from db.paint import f32, rgb, fbm, grid, ramp, smooth
from db.ink import Cel, spl
from chars.chibi_core import *
from chars import chibi_story as ST, chibi_stick as SK, chibi_feast as FE
from chars.chibi_world import plate

# frame a bowl lands, side (0 Goku, 1 Vegeta): 8ths, then 16ths
HITS = [(0, 0), (5, 1), (10, 0), (15, 1), (20, 0), (22, 1), (25, 0), (27, 1), (30, 0), (32, 1), (35, 0), (37, 1)]
TX, TY, TR = 960., 540., 418.
STK = [[(640, 336), (622, 742)], [(1280, 336), (1298, 742)]]


def shot_buns():
    q = [(0., 0.)]; 
    for r, n, o in ((66, 6, .26), (124, 6, .79)): q += [(r * math.cos(k * 2 * pi / n + o), r * math.sin(k * 2 * pi / n + o)) for k in range(n)]
    return q


BUNS = shot_buns()
TAKE = [8, 11, 9, 12, 7, 10, 2, 5, 3, 6, 4, 1]                # which bun each hit eats (index in BUNS), 0 stays


# ---------------------------------------------------------------- straight-down drawings
def top_plate():
    def fn(c):
        c.fill(A([(0, 0), (W, 0), (W, H), (0, H)]), '#6f3a22'); r = np.random.default_rng(3)
        for k, y in enumerate(range(-10, H, 92)):
            c.fill(A([(0, y), (W, y), (W, y + 88), (0, y + 88)]), ['#7d4327', '#743d24', '#82482a'][k % 3]); c.fill(A([(0, y + 88), (W, y + 88), (W, y + 92), (0, y + 92)]), '#472114')
            for q in range(3): x = r.uniform(0, W); c.fill(A([(x, y), (x + 4, y), (x + 4, y + 88), (x, y + 88)]), '#472114')
            for q in range(9): x = r.uniform(0, W); L = r.uniform(80, 300); yy = y + r.uniform(8, 80); c.line(spl([(x, yy), (x + L / 2, yy + r.normal(0, 4)), (x + L, yy)], n=6), 1.6, '#5f2f1c', taper=(.3, .3), seed=q)
        for x in (470, 1450): c.fill(ell(x + 8, 552, 190, 270, sq=3.2), '#3a1812', .6); c.fill(ell(x, 540, 186, 264, sq=3.2), '#b2413c'); c.fill(ell(x, 540, 150, 226, sq=3.2), '#c95348'); c.line(np.vstack([ell(x, 540, 150, 226, sq=3.2)] * 1), 3, '#e9a06a', taper=(0, 0), closed=True)
        c.fill(ell(TX + 16, TY + 26, TR + 10, TR + 10, n=120), '#2a100c', .62); c.fill(ell(TX, TY, TR, TR, n=140), '#8f5630'); c.fill(ell(TX - 2, TY - 4, TR - 9, TR - 9, n=140), '#d39150'); c.clip(ell(TX, TY, TR - 10, TR - 10, n=140))
        for k, y in enumerate(range(int(TY - TR), int(TY + TR), 104)): c.fill(A([(0, y), (W, y), (W, y + 3), (0, y + 3)]), '#94582e')
        for k in range(190): y = r.uniform(TY - TR, TY + TR); x = r.uniform(TX - TR, TX + TR); L = r.uniform(50, 320); c.line(spl([(x - L / 2, y), (x, y + r.normal(0, 3)), (x + L / 2, y + r.normal(0, 2))], n=6), r.uniform(1.1, 2.4), ['#b4743c', '#c07f44', '#a86a35'][k % 3], taper=(.3, .3), seed=k, alpha=.8)
        c.unclip(); c.line(np.c_[TX + (TR - 16) * np.cos(np.linspace(3.4, 5.6, 80)), TY + (TR - 16) * np.sin(np.linspace(3.4, 5.6, 80))], 5, '#ffe0a6', taper=(.4, .4))
    a = plate(W, H, fn, seed=21, wob=1.0)[..., :3]; x, y = grid(); g = np.exp(-(((x - TX) / 560) ** 2 + ((y - TY) / 500) ** 2)); return (a * (.44 + .86 * g[..., None]) + g[..., None] * rgb('#ffb45a') * .10).astype(f32)


def steamer(d):
    n = N().at(TX, TY)
    d.fl.fill(ell(TX + 10, TY + 16, 190, 190), '#5a2a16', .55); d.part(n(ell(0, 0, 184, 184, n=90)), '#c98a45', '#8f5a2c', w=1.1, shade=[n(ell(30, 40, 184, 184, n=60)) if False else n(spl([(190, -60), (190, 190), (-60, 190), (60, 150), (150, 60)], True, n=6))]); d.part(n(ell(0, 0, 166, 166, n=90)), '#e8c07c', '#b98548', w=.8)
    P = n(ell(0, 0, 164, 164, n=90))
    for k in range(-7, 8): y = k * 22; d.stroke(n([(-160, y), (160, y)]), .5, '#a9743a', taper=(0, 0), within=P)
    for k in range(-3, 4): x = k * 46; d.stroke(n([(x, -160), (x, 160)]), .35, '#b98548', taper=(0, 0), within=P)
    a = np.linspace(3.3, 5.4, 40); d.stroke(np.c_[TX + np.cos(a) * 175, TY + np.sin(a) * 175], .8, '#f7dca0', taper=(.3, .3))


def bun_top(d, x, y, r, seed=0, sx=1., sy=1., rot=0.):
    n = N().at(x, y, rot, r * sx, r * sy); d.fl.fill(n(ell(.14, .20, 1.0, 1.0)), '#8f5a2c', .5); P = n(ell(0, 0, 1, 1, n=50)); d.part(P, '#fffaf0', '#f0cfa2', w=.75, seed=seed, shade=[n(spl([(1.1, -.5), (.62, -.1), (.50, .5), (-.1, .78), (-.6, .8), (0, 1.3), (1.3, 1.2)], True, n=6))])
    for k in range(9): a = k * 2 * pi / 9 + seed * .37; d.stroke(n(spl([(.14 * math.cos(a), .14 * math.sin(a)), (.40 * math.cos(a + .30), .40 * math.sin(a + .30)), (.66 * math.cos(a + .42), .66 * math.sin(a + .42))], n=6)), .42, '#c08a56', taper=(.05, .6))
    d.part(n(ell(0, 0, .13, .13)), '#fff6e2', '#e6c296', w=.4); d.detail(n(ell(-.42, -.46, .22, .13, -.7)), '#ffffff', within=P, sh='#ffffff')


BC = [('#f6efe2', '#c9a78c', '#34508e'), ('#e9573f', '#9c2f24', '#f6c04a'), ('#f6efe2', '#c9a78c', '#c8342b'), ('#3f6fae', '#26437a', '#f6efe2')]


def bowl_top(d, x, y, r, pat=0, seed=0, full=False):
    col, sh, ac = BC[pat % 4]; n = N().at(x, y, 0, r); P = n(ell(0, 0, 1, 1, n=50)); d.part(P, col, sh, w=.85, seed=seed, shade=[n(spl([(1.1, -.4), (.8, .2), (.3, .75), (-.4, .95), (0, 1.4), (1.4, 1.2)], True, n=6))]); d.stroke(n(ell(0, 0, .90, .90, n=50)), .7, ac, taper=(0, 0))
    d.part(n(ell(0, 0, .80, .80, n=46)), mix(col, '#8a4a22', .16), mix(sh, '#3a1a12', .2), w=.5, shade=[n(spl([(-1, -.9), (.9, -.9), (.6, -.2), (-.1, -.35), (-.5, .2), (-.45, .8), (-1, .8)], True, n=6))]); d.part(n(ell(.04, .05, .40, .40)), mix(col, '#8a4a22', .06), sh, w=.35)
    for k in range(6): a = k * pi / 3 + seed; d.detail(n(star(.60 * math.cos(a), .60 * math.sin(a), .10, a, .5, 4)), ac, sh=mix(ac, '#2a0f14', .3))
    if full: d.part(n(spl(ell(0, 0, .70, .70, n=12) * 1., True, n=4)), '#fffaf0', '#e6c9a4', w=.5)


def arm_top(d, n, c, who, side, p, span, seed, hold=None):
    """one arm seen from above: shoulder at (-.35, side*1.2), a smear ring between dish and mouth.  p = turns"""
    col, sh = (c['skin'], c['skin2']) if who == 'goku' else (c['glove'], c['glove2']); sl, sl2 = (c['blue'], c['blue2']) if who == 'goku' else (c['suit'], c['suit2'])
    cx, cy, rx, ry = 1.95, side * .84, 1.35, .74; a1 = p * 2 * pi; u = np.linspace(0, 1, 40); an = a1 - span * (1 - u); P = np.c_[cx + rx * np.cos(an), cy + ry * np.sin(an) * side]
    if hold is not None: P = np.asarray(hold, float)
    h = P[-1]; sp = A([-.30, side * 1.28]); el = sp * .45 + h * .55 + A([-.2, side * .50])
    U = SK.limb_union(d, n, [sp, sp * .4 + el * .6, el, el * .4 + h * .6, h], (.46, .42, .30, .27, .20)); o = n.M[:2, :2] @ A([-.05, .16])
    if who == 'goku':
        d.part(U, col, sh, w=.9, seed=seed, shade=[U + o]); V = SK.limb_union(d, n, [sp, sp * .6 + el * .4, sp * .3 + el * .7], (.50, .47, .43)); d.part(V, sl, sl2, w=.9, seed=seed + 1, shade=[V + o])
    else: d.part(U, sl, sl2, w=.9, seed=seed, shade=[U + o])
    if hold is None:
        T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; Nn = np.c_[-T[:, 1], T[:, 0]]; w = .40 * u ** .7 + .012; Q = n(np.vstack([P + Nn * w[:, None], (P - Nn * w[:, None])[::-1]])); d.part(Q, col, sh, w=.6, seed=seed + 2, gaps=[(.42, .52)], shade=[n(np.vstack([P - Nn * w[:, None] * .2, (P - Nn * w[:, None])[::-1]]))])
        if who == 'goku': i = slice(27, 33); d.detail(n(np.vstack([P[i] + Nn[i] * w[i, None], (P[i] - Nn[i] * w[i, None])[::-1]])), sl, sh=sl2)
        for o_, q in ((-.16, .25), (.14, .40)): R_ = P + Nn * (w * o_ * 2.2)[:, None]; d.stroke(n(R_[int(40 * q):36]), .42, sh, taper=(.5, .2))
        for o_, q in ((.50, .45), (-.46, .6), (.66, .70)): R_ = np.c_[cx + (rx + o_) * np.cos(an), cy + (ry + o_ * .7) * np.sin(an) * side]; d.stroke(n(R_[int(40 * q):]), .5, taper=(.6, .1))
        for k in (22, 31): d.stroke(n(ell(P[k][0], P[k][1], .26, .24)), .42, sh, taper=(.1, .1))
        ang = math.atan2(T[-1][1], T[-1][0])
    else: e = P[-1] - P[0]; ang = math.atan2(e[1], e[0])
    hn = n.at(h[0], h[1], ang + pi / 2, .80); SK.hand_fist(d, hn, dict(c, skin=col, skin2=sh), seed + 5); return h


def head_top(d, n, who, c, t):
    """n faces +x.  hair from above, taken from the same 3-D clumps"""
    h = Hd(n.at(0, 0, -pi / 2), 0., pi / 2 - .12, (0, .04 * math.sin(t * 1.3)), t, .02); F = n(spl([(.30, -.80), (.78, -.55), (1.0, 0), (.78, .55), (.30, .80)], True, n=6)); d.part(F, c['skin'], c['skin2'], w=.8)
    d.part(n(spl([(.92, -.12), (1.12, 0), (.92, .12)], True, n=4)), c['skin'], c['skin2'], w=.6)
    for sg in (-1, 1): d.part(n(ell(.05, sg * 1.0, .20, .13)), c['skin'], c['skin2'], w=.7)
    L = sorted(HAIR[who], key=lambda q: h.rot(q['tip'])[0, 2]); O = n(ell(-.02, 0, 1.0, .98, n=50)); d.part(O, c['hair'], c['hair2'], w=.9)
    for i, q in enumerate(L): SK.slock(d, h, q, c, 10 + i, .75, deep=.12)
    if who == 'goku': pass
    d.stroke(n(spl([(-.1, .1), (.0, 0), (.14, -.06)], n=5)), .4, c['h1'])


HAIR = ST.HAIR


def person_top(d, who, x, y, S, flip, t, p, span, dy=0., hold=None):
    c = FE.C[who]; n = N().at(x, y + dy, 0., -S if flip else S, S)
    d.fl.fill(n(ell(-.7, .12, 1.0, 1.9)), '#2a100c', .5)
    if who == 'goku':
        B = n(spl([(-1.30, -1.0), (-.9, -1.62), (-.2, -1.70), (.30, -1.2), (.46, 0), (.30, 1.2), (-.2, 1.70), (-.9, 1.62), (-1.30, 1.0), (-1.42, 0)], True, n=8)); d.part(B, c['gi'], c['gi2'], w=1., shade=[n(spl([(-1.6, .2), (-.6, .7), (.5, .6), (.5, 2), (-1.6, 2)], True, n=5))]); d.part(n(ell(-.82, 0, .52, .52)), '#fffaf0', '#e2c09c', w=.6); m = n.at(-.82, 0, pi / 2, .50, .50)
        for L in ([(-.50, -.60), (-.50, .66)], [(-.76, -.15), (-.68, .10)], [(-.32, -.24), (-.24, -.02)], [(-.12, -.54), (.66, -.54)], [(.22, -.54), (.10, -.10)], [(-.05, -.30), (.52, -.30), (.52, -.08)], [(-.20, -.07), (.74, -.07)], [(-.02, .18), (-.02, .64)], [(-.02, .18), (.58, .18), (.58, .64)], [(-.02, .64), (.58, .64)]): d.stroke(m(A(L)), .6, taper=(.05, .15), tmin=.5)
        d.part(n(spl([(-.2, -.62), (.40, -.34), (.46, .34), (-.2, .62)], True, n=5)), c['blue'], c['blue2'], w=.7)
    else:
        B = n(spl([(-1.20, -1.0), (-.9, -1.58), (-.2, -1.66), (.30, -1.2), (.44, 0), (.30, 1.2), (-.2, 1.66), (-.9, 1.58), (-1.20, 1.0), (-1.32, 0)], True, n=8)); d.part(B, c['suit'], c['suit2'], w=1.); d.part(n(spl([(-1.14, -.86), (-.6, -1.06), (.1, -.96), (.36, 0), (.1, .96), (-.6, 1.06), (-1.14, .86)], True, n=6)), c['white'], c['white2'], w=.8, shade=[n(spl([(-1.6, .2), (-.6, .5), (.5, .4), (.5, 2), (-1.6, 2)], True, n=5))])
        for sg in (-1, 1): d.part(n(spl([(-.9, sg * .80), (-.3, sg * .98), (-.3, sg * 1.28), (-.9, sg * 1.16)], True, n=4)), c['gold'], c['gold2'], w=.6)
        d.part(n(ell(.0, 0, .56, .60)), c['suit'], c['suit2'], w=.7)
    if hold is None:
        for sd, o, sp in ((-1, 0., 20), (1, .5, 40)): arm_top(d, n, c, who, sd, p + o, span, sp)
    else:
        for sd, sp in ((-1, 20), (1, 40)): arm_top(d, n, c, who, sd, 0, 0, sp, hold=[A([1.0, sd * 1.2]), A(hold) * [1, sd]])
    head_top(d, n.at(.06 + .05 * math.sin(p * 4 * pi), 0.), who, c, t); return n


# ---------------------------------------------------------------- insets
PANEL = [A([(-20, -20), (318, -20), (268, H + 20), (-20, H + 20)]), A([(W + 20, -20), (W - 318, -20), (W - 268, H + 20), (W + 20, H + 20)])]


def inset(im, side, f, puff, flash, state='eat', k=None):
    who = ['goku', 'vegeta'][side]; x0 = 0 if side == 0 else W - 330; c = Cel(W, H, 0, 0, 2); c.fill(PANEL[side], '#ffffff'); m = c.rgba()[..., 3:4]
    g = Cel(W, H, 0, 0, 1); col = ['#e8622a', '#2f57a8'][side]; g.fill(A([(x0, 0), (x0 + 330, 0), (x0 + 330, H), (x0, H)]), col); cx = 150 if side == 0 else W - 150
    for q in range(26): a = q * 2 * pi / 26 + (f // 2) * .13; g.fill(A([(cx + math.cos(a) * 250, 560 + math.sin(a) * 330), (cx + math.cos(a - .05) * 900, 560 + math.sin(a - .05) * 900), (cx + math.cos(a + .05) * 900, 560 + math.sin(a + .05) * 900)]), mix(col, '#ffffff', .22))
    lay = g.rgba()[..., :3]; d = D(x0 - 160, 100, 650, 990, 2, lw=5.0, line=FE.INK, var=.3)
    kk = dict(state=state, puff=puff, t=f * 1.0, lag=(0, .05 * math.sin(f * 1.3)), yaw=.16, rice=1); kk.update(k or {})
    q = flash; n = N().at(cx, 560 + 10 * math.sin(f * 1.26), 0., (-1 if side else 1) * 112 * (1 + .06 * q), 112 * (1 - .06 * q))
    FE.body(d, n, who, kk); FE.head(d, n, who, kk); L = d.layer(); lay = comp.over(lay, L); lay = lay * (1 + .25 * q)
    im = im * (1 - m) + lay * m; e = Cel(W, H, 0, 0, 2); P = PANEL[side]; e.line(A([P[1], P[2]]), 22, '#fff6e2', taper=(0, 0), press=0.); e.line(A([P[1], P[2]]), 8, FE.INK, taper=(0, 0), press=0.); return comp.over(im, e.rgba())


def chibi_sh(d, x, y, r): d.fl.fill(ell(x, y, r, r), '#5a2a16', .5)


class Shot:
    id = 'b4'; n = 80
    finish = dict(bloom=.26, vig=.12, soft=.12, grain=.009, lift=.0)
    EVENTS = [(k, 'bowl', dict(size=.8, pan=-.45 if s == 0 else .45, note=('Goku' if s == 0 else 'Vegeta') + ' slams a bowl on the stack (woodblock)')) for k, s in HITS] + [
        (0, 'slurp_loop', dict(size=.6, dur=40, note='two arms whirling, faster and faster, shamisen')), (20, 'accelerate', dict(size=.5)),
        (40, 'hard_stop', dict(size=1.0, note='everything stops: the last bun, relic light, one choir breath')), (40, 'relic', dict(size=.6, dur=18, note='glow, steam')),
        (44, 'creep', dict(size=.3, dur=13, note='two chopsticks creep in from both sides, trembling')), (50, 'eyes_slide', dict(size=.3, note='both pairs of eyes slide to each other')), (54, 'eyes_narrow', dict(size=.2)),
        (58, 'spark_clash', dict(size=1.0, note='chopsticks meet over the bun; lightning between the eyes')), (58, 'crackle', dict(size=.6, dur=8)),
        (66, 'lamp_out', dict(size=.8, note='click, dark')), (67, 'lamp_flicker', dict(size=.4)), (68, 'lamp_out', dict(size=.6)), (68, 'cloth_rustle', dict(size=.25, pan=.3, note='somebody moves in the dark')),
        (70, 'lamp_on', dict(size=.8, note='light: the bun is gone')), (70, 'horror', dict(size=1.0, note='two chibi screams, pointing at each other')), (70, 'glove_exit', dict(size=.12, pan=1., note='a white glove slips out at the right edge, on his own side, almost nothing')),
        (76, 'stomach_growl', dict(size=1.0, pan=-.4, note='Goku\'s stomach: long, low; the bowls rattle'))]

    def setup(self):
        self.F = FE.Feast(); self.F.setup(); self.top = top_plate()
        self.dark = None

    # ---------------------------------------------------- A: straight down
    def down(self, f):
        im = self.top.copy(); d = D(lw=4.6, line=FE.INK, var=.28, light=(-.4, -.9)); rate = f / 5. if f < 20 else 4 + (f - 20) / 2.5; span = 2.5 if f < 10 else (3.1 if f < 20 else 4.4)
        steamer(d); eaten = {}
        for j, (k, s) in enumerate(HITS): eaten[TAKE[j]] = (k, s)
        for i, (bx, by) in enumerate(BUNS):
            if i in eaten and f >= eaten[i][0]: continue
            j = 0.; 
            if i in eaten and f == eaten[i][0] - 1: j = 1.
            bun_top(d, TX + bx, TY + by - 4 * j, 36 * (1 + .05 * j) if i else 40, i)
        # stacks: every bowl stands a little further from the lens axis
        ring = []
        for s in (0, 1):
            mine = [k for k, q in HITS if q == s]; cnt = [3, 2]
            for w in (0, 1):
                x, y = STK[s][w]; base = 4 if w == 0 else 3; ks = mine[w::2]; nn = base + sum(1 for k in ks if f >= k); dr = A([x - TX, y - TY]); dr = dr / np.hypot(*dr); d.fl.fill(ell(x + 10 - dr[0] * 6, y + 14, 54, 54), '#5a2a16', .5); sw = 2.2 * math.sin(f * .8 + w + s) * (f > 20)
                for b in range(nn):
                    q = 1.; last = None
                    if b >= base: last = f - ks[b - base]; q = {0: 1.16, 1: .97, 2: 1.02}.get(last, 1.)
                    px, py = A([x, y]) + dr * 8.0 * b + [sw * b * .12, 0]; bowl_top(d, px, py, 52 * (1 + .026 * b) * q, (b * 3 + w + s * 2) % 4, b)
                    if last in (0, 1): ring.append((px, py, 52 * (1 + .026 * b), last))
                nxt = [k for k in ks if k == f + 1]
                if nxt: b = nn; px, py = A([x, y]) + dr * 8.0 * b - dr * 30; bowl_top(d, px, py, 52 * (1 + .026 * b) * 1.42, (b * 3 + w + s * 2) % 4, b)
        hitL = min([f - k for k, s in HITS if s == 0 and f >= k] or [9]); hitR = min([f - k for k, s in HITS if s == 1 and f >= k] or [9])
        person_top(d, 'goku', 462, 540, 84, False, f, rate, span, 3 * math.sin(f * 1.26)); person_top(d, 'vegeta', 1458, 540, 84, True, f, rate + .25, span, -3 * math.sin(f * 1.26))
        # a bun on its way to a mouth
        for j, (k, s) in enumerate(HITS):
            if 0 <= f - k < 3: bx, by = BUNS[TAKE[j]]; u = [.35, .75, .97][f - k]; m = A([560 if s == 0 else 1360, 540]); p = (A([TX + bx, TY + by])) * (1 - u) + m * u; bun_top(d, p[0], p[1] - 30 * math.sin(pi * u), 36 * (1.16 - .3 * u), TAKE[j], 1.25 - .2 * u, .84, 0.)
        for x, y, r, q in ring:
            for a in np.linspace(0, 2 * pi, 9)[:-1]: e = A([math.cos(a), math.sin(a)]); d.part(tube([A([x, y]) + e * (r + 12 + 10 * q), A([x, y]) + e * (r + 34 + 16 * q)], (4.2 - 1.4 * q, 1.2)), '#fffdf4', '#fffdf4', w=.2)
        # flying rice
        r_ = np.random.default_rng(7)
        for j, (k, s) in enumerate(HITS):
            for g in range(9):
                a = r_.normal(0, .9); v = r_.uniform(9, 26); rot = r_.uniform(0, 3); T_ = int(r_.integers(8, 14)); t = f - k
                if t < 0: continue
                tt = min(t, T_); sg = 1 if s == 0 else -1; p = A([566 if s == 0 else 1354, 540]) + A([math.cos(a) * sg, math.sin(a)]) * v * tt * (1 - .025 * tt); z = math.sin(pi * tt / T_); rr = 5.2 * (1 + .9 * z); d.part(ell(p[0], p[1], rr * 1.25, rr * .72, rot + tt * .5, n=14), '#fffdf6', '#e6cfae', w=.34)
        L = d.rgba(); X, Y = grid(); g = np.exp(-(((X - TX) / 640) ** 2 + ((Y - TY) / 560) ** 2))[..., None]; L[..., :3] *= (.62 + .46 * g) * rgb('#fff0d2'); im = comp.over(im, L)
        im = comp.screen(im, (np.exp(-(((X - TX) / 300) ** 2 + ((Y - TY) / 280) ** 2)) * .16)[..., None] * rgb('#ffcf87'))
        im = self.zoom(im, 1.0 + .0016 * f, (0, 0) if f < 20 else (2.5 * math.sin(f * 2.4) * (f - 18) / 20, 1.8 * math.cos(f * 2.1) * (f - 18) / 20))
        eat = [sum(1 for k, s in HITS if s == q and f >= k) for q in (0, 1)]
        for s in (0, 1): hit = hitL if s == 0 else hitR; im = inset(im, s, f, .25 + .22 * eat[s] + (.25 if hit < 2 else 0), {0: 1., 1: .5}.get(hit, 0.))
        return im

    def zoom(self, im, z, o=(0, 0)):
        M = np.float32([[z, 0, W / 2 * (1 - z) + o[0]], [0, z, H / 2 * (1 - z) + o[1]]]); return cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # ---------------------------------------------------- B: the relic
    def relic(self, f):
        t = f - 40; im = self.room_blur(); X, Y = grid(); bx, by = 960., 800.; z = 1 + .0035 * t
        c = Cel(W, H, 0, 0, 1); c.grad(A([(0, 640), (W, 640), (W, H), (0, H)]), (0, 640), (0, H), [(0, '#7a4424', 1), (.3, '#c98343', 1), (1, '#6b3820', 1)]); im = comp.over(im, c.rgba()); g = np.exp(-(((X - bx) / 560) ** 2 + ((Y - by - 20) / 170) ** 2))[..., None] * (Y > 640)[..., None]; im = im * (.46 + .5 * np.exp(-(((X - bx) / 760) ** 2 + ((Y - 620) / 460) ** 2)))[..., None] + g * rgb('#ffcf87') * .42
        rr = np.sqrt(((X - bx) / 900) ** 2 + ((Y - 560) / 640) ** 2); h = Cel(W, H, 0, 0, 1); FE.halo(h, bx, 590, 1000, t * 1.6, 22, a=.9); u = Eo((t + 1) / 5.); hl = cv2.GaussianBlur(h.rgba()[..., :3], (0, 0), 7) * (np.clip(1 - rr, 0, 1) ** 1.0)[..., None]; im = comp.screen(im, (hl * .60 + (np.clip(1 - rr * 1.25, 0, 1) ** 2.0 * .52)[..., None] * rgb('#ffd27a')) * u)
        e = D(lw=6.4, line=FE.INK, var=.3, light=(0, -1)); e.fl.fill(ell(bx, by + 84, 320, 54), '#4a2012', .55); e.part(N().at(bx, by + 26)(spl([(-130, 0), (-96, 62), (96, 62), (130, 0)], True, n=4, corners=(0, 1, 2, 3))), '#3f6fae', '#26437a', w=.9, shade=[N().at(bx, by + 26)(A([(30, -5), (140, -5), (140, 70), (20, 70)]))]); n = N().at(bx, by); e.part(n(ell(0, 14, 296, 62, n=80)), '#f6efe2', '#c9a078', w=1., shade=[n(ell(0, 54, 296, 50, n=60))]); e.part(n(ell(0, 4, 244, 44, n=70)), '#fff8ea', '#e6c9a4', w=.5); e.stroke(n(ell(0, 14, 272, 53, n=70)), .6, '#3f6fae', taper=(0, 0))
        FE.bun(e, N().at(bx, by, 0., 1 + .006 * math.sin(t * .5), 1 - .006 * math.sin(t * .5)), 186); im = comp.over(im, e.rgba())
        s = Cel(W, H, 0, 0, 1); FE.steam(s, bx, by - 250, 120, f * 1.0, 4, a=.5); im = comp.over(im, cv2.GaussianBlur(s.rgba(), (0, 0), 2.2))
        q = Cel(W, H, 0, 0, 1)
        for i in range(11): a = i * 2 * pi / 11 + t * .03; R = 330 + 46 * math.sin(t * .35 + i * 1.7); q.fill(star(bx + math.cos(a) * R * 1.2, 610 + math.sin(a) * R * .66, 10 + 10 * abs(math.sin(t * .5 + i * 2)), 0., .2, 4), '#fffbe0')
        im = comp.over(im, q.rgba())
        # chopsticks: creep, stop, creep; they shake
        u = float(keys(f, [(42, 0.), (44, .0), (47, .34, 'e'), (50, .40, 'l'), (53, .66, 'e'), (55, .72, 'l'), (57, .965, 'e'), (58, 1.)])); d = D(lw=6.0, line=FE.INK, var=.3)
        for sg, col, sh in ((-1, '#c0392b', '#7d2018'), (1, '#2f4f9a', '#1d2f66')):
            sk = math.sin(f * 2.7 + sg) * 3.0 * (f // 1 % 2 * 2 - 1) * .6; tip = A([bx + sg * (14 + 900 * (1 - u)), 452 - 90 * (1 - u) + sk]); tail = tip + A([sg * 1060, 46 + 30 * (1 - u)]); ax = (tip - tail) / np.hypot(*(tip - tail)); nn = A([-ax[1], ax[0]])
            for o in (-1, 1): d.part(tube([tail + nn * o * 30, tail * .45 + tip * .55 + nn * o * 18, tip + nn * o * 5], (19, 15, 8), cap=.5), col, sh, w=.9, seed=3 + o, shade=[tube([tail + nn * o * 30 + [0, 12], tail * .45 + tip * .55 + nn * o * 18 + [0, 10], tip + nn * o * 5 + [0, 5]], (15, 11, 5))]); P = tip + nn * o * 5; d.part(tube([P - ax * 300 + nn * o * 7, P - ax * 262 + nn * o * 6], (16, 16)), '#f3c14b', '#b8812a', w=.5)
        im = comp.over(im, FE.amber(d.rgba()))
        # eyes
        for side in (0, 1): im = self.eyes(im, side, f)
        return im * (1 - .30 * np.clip((np.abs(X - 960) / 960) ** 2.6, 0, 1))[..., None]

    def room_blur(self):
        if not hasattr(self, '_rb'): self._rb = cv2.GaussianBlur(self.F.room, (0, 0), 14) * .62
        return self._rb.copy()

    def eyes(self, im, side, f):
        """a slit panel with two eyes: in at 42, look at the bun, slide to the rival at 50, narrow at 54"""
        who = ['goku', 'vegeta'][side]; u = Eo((f - 41 - side) / 4.); sg = -1 if side == 0 else 1; cx = 960 + sg * (500 + 900 * (1 - u)); cy = 168; wd, ht = 400, 104; sk = 34
        if u <= 0: return im
        P = A([(cx - wd + sk, cy - ht), (cx + wd + sk, cy - ht), (cx + wd - sk, cy + ht), (cx - wd - sk, cy + ht)]); c = FE.C[who]; m = Cel(W, H, 0, 0, 2); m.fill(P, '#ffffff'); k = m.rgba()[..., 3:4]; d = D(int(cx - wd - 60), int(cy - ht - 90), 2 * wd + 120, 2 * ht + 190, 2, lw=8.6, line=FE.INK, var=.3)
        n = N().at(cx + (30 if side == 0 else -30), cy + 26, 0., (-1 if side else 1) * 300, 300); d.part(P * 1., c['skin'], c['skin2'], w=0, shade=[n(spl([(-1.6, -.6), (-.86, -.4), (-.74, .0), (-.8, .5), (-1.7, .5)], True, n=5))]); h = Hd(n, .10, .02, front=FE.AF, side=FE.AS); F = n(ell(0, 0, 1.6, 1.2)); lx = float(keys(f, [(44, -.05), (50, -.02), (52, .62, 'o'), (80, .62)])); ly = float(keys(f, [(44, .42), (50, .42), (52, -.02, 'o'), (80, 0)])); cl = float(keys(f, [(53, .0), (55, .34, 'o'), (80, .36)])); sw = 1 if f % 7 == 3 and f < 50 else 0
        for q in (-1, 1): SK.eye(d, h, q * .36, .06, q, F, .24, .27, .10 + .34 * Eo((f - 52) / 3.), (lx, ly), iris=None, closed=cl, pup=.86, lw=1.); SK.brow(d, h, q * .38, -.30 + .10 * Eo((f - 52) / 3.), q, F, .28, .04 + .42 * Eo((f - 52) / 3.) + (.14 if who == 'vegeta' else 0), .12, c['hair'])
        if f >= 46: y = -.36 + .40 * E((f - 46) / 11.); o = n.at(-.84, y, 0, .34); d.part(o(spl(A([(0, -.30), (.11, -.06), (.145, .10), (.09, .22), (0, .26), (-.09, .22), (-.145, .10), (-.11, -.06)]) * 1.25, True, n=7, corners=(0,))), '#cfeafc', '#86bfe4', w=.6, ink='#2f5f8c')
        for j, (a, b_, e) in enumerate((((-1.3, -.40), (-1.1, -.10), (-.9, -.42)), ((-.9, -.46), (-.58, -.16), (-.34, -.46)), ((-.36, -.46), (-.05, -.08), (.22, -.46)), ((.3, -.46), (.60, -.18), (.84, -.46)))): d.part(n(blade2(a, b_, e, (0, 0), .08)), c['hair'], c['hair2'], w=.5) if who == 'goku' else None
        if who == 'vegeta': d.part(n(spl([(-1.3, -.6), (-.5, -.6), (-.2, -.5), (0, -.22), (.2, -.5), (.5, -.6), (1.3, -.6), (1.3, -.9), (-1.3, -.9)], True, n=5, corners=(3,))), c['hair'], c['hair2'], w=.6)
        L = np.zeros((H, W, 4), f32); paste(L, d.rgba(), d.x0, d.y0); L = FE.amber(L) if False else L; im = im * (1 - k) + (im * (1 - L[..., 3:4]) + L[..., :3]) * k; e = Cel(W, H, 0, 0, 2); Q = np.vstack([P, P[:1]]); [e.line(A([Q[i], Q[i + 1]]), 18, '#fff6e2', taper=(0, 0), press=0.) for i in range(4)]; [e.line(A([Q[i], Q[i + 1]]), 7, FE.INK, taper=(0, 0), press=0.) for i in range(4)]; return comp.over(im, e.rgba())

    # ---------------------------------------------------- C, D: standoff, lamp
    def standoff(self, f):
        t = f - 58; im = self.F.frame_standoff(float(58 + (t // 2) * 2), spark=1.)
        if t < 6:                                                            # the contact itself
            c = Cel(W, H, 0, 0, 2); r = np.random.default_rng(t); x, y = 960, 632; k = [1., .9, .7, .5, .35, .2][t]
            for q in range(13): a = q * 2 * pi / 13 + r.uniform(-.15, .15); L_ = r.uniform(120, 330) * k; c.fill(A([(x + math.cos(a + 1.57) * 7 * k, y + math.sin(a + 1.57) * 7 * k), (x + math.cos(a) * L_, y + math.sin(a) * L_), (x - math.cos(a + 1.57) * 7 * k, y - math.sin(a + 1.57) * 7 * k)]), '#fff6cf')
            c.fill(star(x, y, 92 * k, .2 * t, .34, 8), '#fffdf0'); L = c.rgba(); im = comp.screen(comp.over(im, L), cv2.GaussianBlur(L[..., :3], (0, 0), 26) * .8)
        if t == 0: im = im * .72 + rgb('#fff3cf') * .42
        o = [(0, 0), (-9, 6), (6, -5), (-4, 3), (2, -2), (-1, 1)][t] if t < 6 else (0, 0); return self.zoom(im, 1.02 + .0025 * t, o)

    def night(self, f, both=True):
        if self.dark is None: self.dark = cv2.GaussianBlur(self.F.frame_standoff(66., spark=0.), (0, 0), 1.2)
        L = self.dark.mean(-1, keepdims=True); k = 1. if both else (1 - .86 * smooth(900, 1180, grid()[0]))[..., None]; im = (L * .12 + self.dark * .03) * rgb('#7f95d8') * 1.1 * k + rgb('#070a18') * .6; return im

    def lampoff(self, f):
        if f == 67: return self.zoom(self.F.frame_standoff(66., spark=0.) * rgb('#ffd9a0') * .62, 1.04)
        im = self.night(f, f == 66); c = Cel(W, H, 0, 0, 2); G = [(0, 552)] if f in (68, 69) else [(0, 552), (1, 1368)]                        # in the second dark only one pair is left
        for s, x in G:
            for dx in (-58, 22) if s == 0 else (-30, 52):
                q = 1. if f != 69 else .55; c.fill(ell(x + dx, 448, 27, 31 * q), '#f4f1e4'); c.fill(ell(x + dx + (6 if s == 0 else -6), 450, 9, 13 * q), '#0a0a12')
        return self.zoom(comp.over(im, c.rgba()), 1.04)

    # ---------------------------------------------------- E: gone
    def gone(self, f):
        t = f - 70; g = t // 2; im = self.F.room.copy(); sw = .5 * math.sin(t * .62) * math.exp(-t / 9.); c = Cel(W, H, 0, 0, 1); FE.lamp(c, t, 1., sway=sw); im = comp.over(im, c.rgba()); im = FE.light(im, 1. if t > 0 else 1.25, sway=sw)
        d = D(lw=5.4, line=ST.LINE, var=.2); S = 158; shk = (0 if t >= 6 else [5, -5, 4, -4, 3][g % 5]); pop = {0: (1.10, .88), 1: (.94, 1.08), 2: (1.03, .98)}.get(t, (1., 1.)); H_ = []
        for who, x, fl in (('goku', 486, False), ('vegeta', 1434, True)):
            P = ST.walk(who, .25); calm = t >= 7; jaw = 1.15 + .10 * math.sin(t * 2.2 + (who == 'vegeta')) if not calm else .0; ex = dict(eyes='blank', mouth='scream', o=jaw, brow=-.40, bdy=-.16)
            if calm: ex = dict(eyes='dot', ek=1.2, look=(-.2, 1.0), mouth='o', brow=-.25, bdy=-.10) if who == 'goku' else dict(eyes='blank', mouth='flat', brow=-.1, bdy=-.1)
            P.update(hip=(0., -ST.PERS[who]['hip'] * .9), sq=(1 - pop[1]) if t < 3 else (.05 * math.sin(t * 3.) if t >= 6 and who == 'goku' else 0.), lean=-.06, fn=A([.16, -.17]), an=0., ff=A([-.20, -.17]), af=0., yaw=.42, hyaw=.42, roll=-.06 + .02 * (g % 2) * (t < 6), pitch=.0 if not calm else (.22 if who == 'goku' else 0.), crossed=False, armk=1.75, hs=1.45, handF=(1.62, -.02), hkF='point', handN=(-.70, 1.15), hkN='open', lag=(-.06, [-.35, .25, -.10, .05][t] if t < 4 else 0.), ex=ex)
            if calm and who == 'goku': P['handF'] = (.46, .70); P['hkF'] = 'open'; P['armk'] = 1.2
            if who == 'vegeta' and t < 3: P['noNear'] = True
            q = ST.fig(d, who, x + (shk if who == 'goku' else -shk), 846, S, P, fl); h = q['h']; H_.append(q)
            if not calm:                                                     # the blue of horror, strokes on the brow
                for k in range(6): xx = -.62 + k * .20; p = h.F([(xx, -.46 - .10 * (k % 2)), (xx, -.18 + .05 * (k % 3))]); d.stroke(p, .42, '#5b7fc0', taper=(.05, .5))
                for k, (x0, y0, a) in enumerate(((-1.15, -.5, -2.5), (-1.25, -.05, -2.9), (1.2, -.55, -.7), (1.30, -.1, -.2))):
                    j = (t + k * .7) % 4; o = h.n.at(x0 + math.cos(a) * j * .10, y0 + math.sin(a) * j * .10 + .03 * j * j, a + pi / 2, .42); d.part(o(spl(A([(0, -.30), (.11, -.06), (.145, .10), (.09, .22), (0, .26), (-.09, .22), (-.145, .10), (-.11, -.06)]), True, n=7, corners=(0,))), '#d3ecfb', '#8fc4e6', w=.5, ink='#2f5f8c')
        if t < 3:                                                            # nobody saw: his other arm, behind the bowls, reaches out of the frame on his own side
            w = D(1480, 480, 440, 320, 2, lw=4.6, line=FE.INK, var=.28); hx, hy = [(1862, 664), (1926, 676), (1770, 662)][t]; U = SK.limb_union(w, N(), [(1516, 636), (1600, 676), (hx - 120, hy + 12), (hx - 34, hy)], (34, 31, 27, 24)); w.part(U, ST.C['vegeta']['suit'], ST.C['vegeta']['suit2'], w=.95, shade=[U + A([0, 14])]); n = N().at(hx, hy, 1.50, 58); cc = dict(FE.C['vegeta'], skin='#f7f2f0', skin2='#b3a3b8')
            if t < 2: bz = n([(.10, -.42)])[0]; FE.bun(w, N().at(bz[0] + 10, bz[1] + 30, .2), 32)
            w.part(tube([(hx - 62, hy + 2), (hx - 22, hy)], (27, 24), cap=.2, cap0=.2), '#ffffff', '#bfb0c4', w=.85); FE.hand_grip(w, n, cc, 5); L = np.zeros((H, W, 4), f32); paste(L, w.rgba(), 1480, 480); im = comp.over(im, FE.amber(L))
        L = FE.amber(d.rgba()); im = comp.over(im, L); im = comp.over(im, self.F.table); e = D(lw=5.0, line=FE.INK, var=.30, light=(0, -1)); rat = (3.2 * math.sin(t * 5.) if t >= 6 else 0.)
        for x, y, r, n_, l, sd in ((262, 850, 76, 12, -.08, 3), (1658, 850, 76, 12, .08, 5), (420, 944, 82, 8, -.05, 11), (1502, 946, 82, 7, .05, 13)): e.fl.fill(ell(x, y + 2, r * .95, r * .26), '#5a2a16', .55); FE.stack(e, x + (rat if x < 900 else rat * .4), y, r, n_, l, sd, sway=.04 * math.sin(t * .5 + sd) + (.05 * math.sin(t * 4.1) if t >= 6 and x < 900 else 0))
        for x, y, s_, kd, ro in ((770, 944, 66, 'bone', -.2), (1150, 956, 70, 'fish', .06), (960, 1010, 60, 'bone', .3)): FE.leftovers(e, x, y, s_, kd, ro)
        for x, y, r, p_ in ((600, 964, 62, 1), (1330, 968, 62, 3)): e.fl.fill(ell(x, y + r * .82, r * .9, r * .2), '#5a2a16', .5); FE.bowl(e, N().at(x, y, .02), r, p_, 'rice', 50)
        bx, by = 960, 800; e.fl.fill(ell(bx, by + 44, 150, 30), '#5a2a16', .5); e.part(N().at(bx, by + 12, 0)(spl([(-60, 0), (-44, 34), (44, 34), (60, 0)], True, n=4, corners=(0, 1, 2, 3))), '#3f6fae', '#26437a', w=.8); e.part(N().at(bx, by, 0)(ell(0, 6, 138, 30)), '#f6efe2', '#c9a078', w=.95, shade=[ell(bx, by + 24, 138, 24)]); e.part(N().at(bx, by, 0)(ell(0, 2, 112, 20)), '#fff8ea', '#e6c9a4', w=.5); e.stroke(ell(bx, by + 7, 126, 25), .5, '#3f6fae', taper=(0, 0))
        for cx, cy, r in ((930, 798, 4), (975, 806, 3), (1002, 797, 3.4)): e.part(ell(cx, cy, r * 1.6, r), '#fffaf0', '#e6c9a4', w=.3)                          # crumbs
        # a missing thing: the dotted outline of a bun, winking
        if t % 4 < 2 and t < 6:
            a = np.linspace(pi, 2 * pi, 15); Q = np.c_[bx + np.cos(a) * 82, by - 6 + np.sin(a) * 104]
            for i in range(0, 14, 2): e.stroke(Q[i:i + 2], .8, '#fff3cf', taper=(.1, .1))
        j = [(0, 0), (6, -4), (-4, 3), (2, -2)][t] if t < 4 else (0, 0)
        for sg, col, sh, rr in ((-1, '#c0392b', '#7d2018', .06), (1, '#2f4f9a', '#1d2f66', -.05)): FE.chopsticks(e, (bx + sg * 330 + j[0], by + 70 + j[1]), (bx + sg * 120, by + 36 + rr * 60), 8.5, col, sh)
        if t >= 6:                                                           # 76: the growl, drawn
            for k in range(3):
                r = 46 + 40 * ((t - 6 + k * .9) % 3); al = np.linspace(.35, 2.4, 26); Q = np.c_[600 + np.cos(al) * r * 1.5, 716 + np.sin(al) * r * .8 + 4 * np.sin(al * 16 + t * 2)]; e.stroke(Q, .75, '#fff0c4', taper=(.2, .2))
        im = comp.over(im, FE.amber(e.rgba()))
        X, Y = grid(); im = im * (1 - .26 * np.clip((np.abs(X - 960) / 960) ** 2.4 + (np.clip(Y - 900, 0, 200) / 260) ** 2, 0, 1))[..., None]
        o = [(8, -7), (-6, 5), (4, -3), (-2, 2)][t] if t < 4 else (0, 0); return self.zoom(im, 1.0 + (.02 if t == 0 else 0), o)

    def frame(self, f):
        if f < 40: return self.down(f)
        if f < 58: return self.relic(f)
        if f < 66: return self.standoff(f)
        if f < 70: return self.lampoff(f)
        return self.gone(f)
