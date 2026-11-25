"""b4 / e3: anime-comedy adult Goku and Vegeta, the lamp-lit table, bowls, chopsticks, the last bun as a relic, sleeping Goku.
    F = Feast(); F.setup(); im = F.frame_standoff(t)          # b4 hero still
    gag(d, 'goku', x, y, S, flip, dict(state='eat'|'stop'|'glare', p=phase, puff=..., ...))      heads look at the middle of the table
    horror(d, 'goku', x, y, S, flip, t)          collapse into chibi horror, pointing
    sleep_goku(d, x, y, S, t)                    e3"""
import numpy as np, cv2, math
from db import W, H, comp
from db.paint import rgb, f32, ramp, fbm, grid, smooth
from .chibi_core import *
from . import chibi_story as ST, chibi_stick as SK
from .chibi_story import HAIR, HLINE
from .chibi_world import plate

INK = '#2a1413'
C = dict(
    goku=dict(skin='#ffcf9e', skin2='#d9814f', hair='#33201f', hair2='#1e1114', h1='#6a3c2c', h2='#b86a35', star='#ffcf3d', gi='#f98a2b', gi2='#b8511c', blue='#34508e', blue2='#222f5c'),
    vegeta=dict(skin='#ffcf9e', skin2='#d9814f', hair='#2c1c20', hair2='#190e14', h1='#5e3630', h2='#a8603a', star='#ffcf3d', suit='#3149a0', suit2='#1f2b66', white='#fff3da', white2='#d3a988', gold='#f6b53a', gold2='#b8741f', blue='#fff3da', blue2='#d3a988', glove='#ffffff', glove2='#bfb0c4'))
AF = A([(0, -.98), (.44, -.92), (.78, -.68), (.90, -.28), (.90, .10), (.86, .42), (.72, .72), (.44, .96), (.16, 1.06), (0, 1.07), (-.16, 1.06), (-.44, .96), (-.72, .72), (-.86, .42), (-.90, .10), (-.90, -.28), (-.78, -.68), (-.44, -.92)])
AS = A([(0, -.96), (.42, -.90), (.74, -.66), (.88, -.30), (.89, -.02), (.90, .24), (.86, .56), (.74, .88), (.52, 1.05), (.26, 1.02), (-.10, .86), (-.50, .62), (-.86, .28), (-.98, -.2), (-.87, -.6), (-.50, -.87)])


def head(d, n, who, k):
    """local: facing right, lamp in front and above"""
    c = C[who]; puff = k.get('puff', 0.); st = k.get('state', 'glare'); t = k.get('t', 0)
    h = Hd(n, k.get('yaw', .46), k.get('pitch', .06), k.get('lag', (0, 0)), t, 0., front=AF, side=AS, cheek=.48 * puff); h.cheeky = .54; h.cheekb = .42
    SK.shair_back(d, h, who, c, 10, w=.8)
    SK.sear(d, h, c, 1, .16, .9, seed=5) if abs(h.yaw) < .3 else None
    F = h.outline(); s = h.n.s; d.part(F, c['skin'], c['skin2'], w=.9, seed=1, shade=[h.n(spl([(-1.3, -.6), (-.62, -.40), (-.48, .0), (-.56, .40), (-.30, .72), (.10, .92), (.40, 1.0), (.5, 1.4), (-1.4, 1.4)], True, n=7))])
    for q in (-1, 1):
        o = h.loc([(q * .66, .52)])[0] + [q * .10 * puff, .02]; r = .19 + .05 * puff; d.detail(h.n(ell(o[0], o[1], r, r * .62)), mix(c['skin'], '#ff5a4a', .36 + .16 * puff), within=F, sh=mix(c['skin2'], '#ff5a4a', .3))
        if puff > .4: d.detail(h.n(ell(o[0] + .04, o[1] - .20, .09, .04, -.2)), '#fff0d4', within=F, sh='#f6c89a')
    SK.sear(d, h, c, -1, .16, 1., seed=6)
    ang = dict(eat=.10, stop=.0, glare=.42, horror=0.)[st]; lid = k.get('lid', dict(eat=0, stop=-.1, glare=.22, horror=0)[st]); kind = k.get('eyes', dict(eat='happy', stop='open', glare='open', horror='open')[st])
    for sg in (-1, 1):
        SK.eye(d, h, sg * .35, .13, sg, F, .215, .25, ang, k.get('look', (.22, .12)), iris=('#2a1a18' if who == 'goku' else '#241a24'), kind=kind, closed=lid, pup=k.get('pup', .92), lw=.95)
        SK.brow(d, h, sg * .35, -.16 + k.get('brow', dict(eat=-.06, stop=-.12, glare=.02, horror=-.14)[st]), sg, F, .25, dict(eat=-.02, stop=-.10, glare=.44, horror=-.3)[st] + (.16 if who == 'vegeta' else 0), .115, c['hair'])
    if puff > .5: o = h.loc([(-.50, .52)])[0]; d.stroke(h.n(spl([o + [-.30, .22], o + [-.10, .38], o + [.16, .36], o + [.32, .18]], n=7)), .5, '#a8502e', taper=(.35, .35), within=F)
    SK.nose(d, h, F, .02, .42, 1.2); m = k.get('mouth', dict(eat='chew', stop='mu', glare='mu', horror='o')[st])
    if m == 'mu':
        d.stroke(h.F(spl([(-.10, .70), (-.04, .665), (.02, .70), (.08, .665), (.14, .70)], n=6)), .85, taper=(.1, .1), tmin=.5, within=F)
        for sg in (-1, 1): o = h.loc([(sg * .34, .66)])[0]; d.stroke(h.n(spl([o + [0, -.16], o + [sg * .07, .0], o + [0, .17]], n=6)), .5, '#a8502e', taper=(.3, .3), within=F)
    elif m == 'chew':
        o = .5 + .5 * math.sin(t * 2.1); M = spl([(-.20, .62), (0, .60), (.22, .62), (.24, .70 + .12 * o), (.02, .74 + .18 * o), (-.20, .70 + .12 * o)], True, n=7); SK.smouth(d, h, F, M, ell(.02, .74 + .18 * o, .14, .09 * o + .02), None, w=.6)
        for x in (-.10, .0, .10): d.stroke(h.F(spl([(x, .66), (x + .03 * math.sin(t + x * 9), .90), (x + .02, 1.16)], n=6), .05), 1.0, '#f7dd9a', taper=(0, .1)); d.stroke(h.F(spl([(x + .03, .68), (x + .03 + .03 * math.sin(t + x * 9), .90), (x + .05, 1.16)], n=6), .05), .3, '#8a4a22')
    if k.get('rice', 1):
        for x, y in ((.40, .74), (-.30, .62)): d.part(h.F(ell(x, y, .045, .032, .5), .05), '#fffaf0', '#e8cfae', w=.34)
    SK.shair_front(d, h, who, c, F, 60, w=.8, drop=.12)
    if k.get('vein'): ST  # reserved
    return h, F


def chopsticks(d, a, b, w=7., col='#c0392b', sh='#7d2018', open_=.16):
    a, b = A(a, float), A(b, float); u = b - a; L = np.hypot(*u); u /= L; nn = A([-u[1], u[0]])
    for sg in (-1, 1): d.part(tube([a + nn * sg * w * 1.3 - u * L * .18, b + nn * sg * L * open_ * .22], (w * .62, w * .36)), col, sh, w=.5); d.part(tube([a + nn * sg * w * 1.3 - u * L * .18, a + nn * sg * w * 1.3 - u * L * .10], (w * .64, w * .62)), '#f3c14b', '#b8812a', w=.4)


def bowl(d, n, r=1., pat=0, food=None, seed=0):
    """rice bowl, rim ellipse at y = 0, foot at about y = .72 r"""
    cols = [('#f6efe2', '#c9a78c', '#34508e'), ('#e9573f', '#9c2f24', '#f6c04a'), ('#f6efe2', '#c9a78c', '#c8342b'), ('#3f6fae', '#26437a', '#f6efe2')][pat % 4]
    P = n(spl([(-r, 0), (-.94 * r, .28 * r), (-.66 * r, .60 * r), (-.36 * r, .68 * r), (-.34 * r, .80 * r), (.34 * r, .80 * r), (.36 * r, .68 * r), (.66 * r, .60 * r), (.94 * r, .28 * r), (r, 0)], True, n=6, corners=(0, 3, 4, 5, 6, 9)))
    d.part(P, cols[0], cols[1], w=.9, seed=seed, shade=[n(spl([(.16 * r, -.1 * r), (.40 * r, .3 * r), (.2 * r, .9 * r), (1.2 * r, .9 * r), (1.2 * r, -.1 * r)], True, n=4))])
    for i in range(6): x = (-.80 + .32 * i) * r; d.detail(n(spl([(x - .12 * r, .32 * r), (x, .18 * r), (x + .12 * r, .32 * r), (x, .26 * r)], True, n=4)), cols[2], within=P, sh=mix(cols[2], '#2a0f14', .35))
    d.stroke(n(spl([(-.86 * r, .42 * r), (0, .56 * r), (.86 * r, .42 * r)], n=8)), .5, cols[2], taper=(.02, .02), within=P)
    E_ = n(ell(0, 0, r, .20 * r)); d.part(E_, mix(cols[0], '#8a4a22', .30), mix(cols[1], '#3a1a12', .3), w=.8, seed=seed + 1)
    if food == 'rice':
        d.part(n(spl([(-.82 * r, .02 * r), (-.60 * r, -.30 * r), (-.2 * r, -.46 * r), (.3 * r, -.42 * r), (.66 * r, -.24 * r), (.82 * r, .02 * r), (0, .16 * r)], True, n=6)), '#fffaf0', '#e6c9a4', w=.7, seed=seed + 2)


def bowl_front(d, n, r=1., pat=0, seed=0):
    """a bowl in a nest: its top edge is the near rim, so the bowl standing in it passes behind"""
    cols = [('#f6efe2', '#c9a78c', '#34508e'), ('#e9573f', '#9c2f24', '#f6c04a'), ('#f6efe2', '#c9a78c', '#c8342b'), ('#3f6fae', '#26437a', '#f6efe2')][pat % 4]; a = np.linspace(pi, 0, 22); rim = np.c_[np.cos(a) * r, np.sin(a) * .20 * r]
    P = n(np.vstack([rim, spl([(r, 0), (.94 * r, .30 * r), (.68 * r, .62 * r), (.38 * r, .72 * r), (.36 * r, .82 * r), (-.36 * r, .82 * r), (-.38 * r, .72 * r), (-.68 * r, .62 * r), (-.94 * r, .30 * r), (-r, 0)], n=6, corners=(3, 4, 5, 6))[1:-1]]))
    d.part(P, cols[0], cols[1], w=.9, seed=seed, shade=[n(spl([(.16 * r, -.1 * r), (.40 * r, .3 * r), (.2 * r, .9 * r), (1.2 * r, .9 * r), (1.2 * r, -.1 * r)], True, n=4)), n(np.vstack([rim, (rim + [0, -.2 * r])[::-1]]))]); d.stroke(n(rim + [0, .07 * r]), .7, cols[2], taper=(.02, .02), within=P); d.stroke(n(rim * [1, 1] + [0, .13 * r]), .35, cols[2], taper=(.02, .02), within=P)


def stack(d, x, y, r, n_, lean=0., seed=0, sway=0.):
    """tower of empty bowls standing at (x, y): top first, every lower rim in front of the bowl it holds"""
    q = np.random.default_rng(seed); B = []
    for i in range(n_): u = i / max(1, n_); B.append((x + lean * r * (u ** 1.6) * n_ * .12 + sway * r * u * u + q.normal(0, .03) * r, y - .80 * r - i * .30 * r, q.normal(0, .03) + lean * .10 * u + sway * .06 * u, int(q.integers(0, 4))))
    for i in range(n_ - 1, -1, -1):
        bx, by, ro, pt = B[i]; nn = N().at(bx, by, ro)
        if i == n_ - 1: bowl(d, nn, r, pt, seed=seed + i)
        else: bowl_front(d, nn, r, pt, seed + i)


def leftovers(d, x, y, s, kind='bone', rot=0.):
    n = N().at(x, y, 0., s); d.fl.fill(n(ell(0, .16, 1.15, .30)), '#5a2a16', .5); d.part(n(ell(0, 0, 1.1, .30)), '#f6efe2', '#c9a078', w=.8, shade=[n(ell(.1, .18, 1.1, .24))]); d.part(n(ell(0, -.03, .84, .19)), '#fff8ea', '#e6c9a4', w=.4)
    m = n.at(0, -.16, rot)
    if kind == 'bone':
        P = [m(tube([(-.55, .02), (.55, -.02)], (.10, .10), cap=.5, cap0=.5))] + [m(ell(sx * .60, sy * .10, .13, .12)) for sx in (-1, 1) for sy in (-1, 1)]; U = d.union(P)[0]; d.part(U, '#fff6e0', '#dbb98c', w=.75, shade=[U + [0, .12 * s]])
    else:
        for i in range(6): d.stroke(m(spl([(-.42 + i * .16, -.24 + .02 * i), (-.36 + i * .16, 0), (-.42 + i * .16, .20 - .02 * i)], n=5)), .8, '#fff6e0', taper=(.2, .2)); 
        d.stroke(m([(-.7, 0), (.62, 0)]), 1.1, '#fff6e0', taper=(0, .1)); d.part(m(spl([(-1.0, .0), (-.80, -.22), (-.52, -.14), (-.52, .14), (-.80, .20)], True, n=5)), '#e9dcc4', '#b99c78', w=.7); d.detail(m(ell(-.80, -.03, .05, .06)), '#2a1413'); d.part(m(spl([(.52, 0), (.86, -.24), (.78, 0), (.86, .22)], True, n=4, corners=(1, 3))), '#e9dcc4', '#b99c78', w=.6)


def bun(d, n, r=1., bite=0., seed=0):
    """the last steamed bun: pleated dome on a paper, plump; node at its base centre"""
    d.part(n(spl([(-1.06 * r, .02 * r), (0, -.10 * r), (1.06 * r, .02 * r), (.8 * r, .16 * r), (0, .20 * r), (-.8 * r, .16 * r)], True, n=6)), '#fff6e2', '#e0bd92', w=.6, seed=seed)
    P = n(spl([(-.98 * r, -.02 * r), (-1.04 * r, -.36 * r), (-.80 * r, -.86 * r), (-.40 * r, -1.16 * r), (-.10 * r, -1.30 * r), (.06 * r, -1.36 * r), (.22 * r, -1.28 * r), (.46 * r, -1.14 * r), (.84 * r, -.82 * r), (1.04 * r, -.36 * r), (.98 * r, -.02 * r), (.5 * r, .10 * r), (-.5 * r, .10 * r)], True, n=8))
    d.part(P, '#fffaf0', '#f0cfa2', w=1., seed=seed + 1, shade=[n(spl([(-1.2 * r, -.3 * r), (-.72 * r, -.16 * r), (0, -.08 * r), (.72 * r, -.20 * r), (1.2 * r, -.5 * r), (1.2 * r, .3 * r), (-1.2 * r, .3 * r)], True, n=6))])
    for a, b, c_ in (((-.02, -1.24), (-.34, -1.00), (-.46, -.64)), ((.02, -1.22), (-.12, -.92), (-.16, -.58)), ((.08, -1.22), (.16, -.92), (.18, -.60)), ((.12, -1.24), (.40, -1.00), (.50, -.66))): d.stroke(n(spl(A([a, b, c_]) * r, n=7)), .62, '#c08a56', taper=(.06, .55))
    d.detail(n(spl([(-.66 * r, -.66 * r), (-.50 * r, -.86 * r), (-.30 * r, -.96 * r), (-.44 * r, -.72 * r), (-.56 * r, -.50 * r)], True, n=5)), '#ffffff', within=P, sh='#ffffff'); d.stroke(n(spl([(-.06 * r, -1.30 * r), (.04 * r, -1.40 * r), (.12 * r, -1.28 * r)], n=5)), .7, taper=(.1, .1))


def steam(c, x, y, s, t, n_=3, col='#fff3d6', a=.55):
    """curling ribbons of steam on a Cel (no ink)"""
    for i in range(n_):
        ph = t * .16 + i * 2.1; u = np.linspace(0, 1, 26); cx = x + (i - (n_ - 1) / 2) * s * .42 + s * .30 * np.sin(u * 5.2 - ph) * u ** .7; cy = y - u * s * (1.9 + .25 * i)
        w = s * .15 * np.sin(np.clip(u * .96 + .04, 0, 1) * pi) ** .8 * (1 + .3 * np.sin(u * 7 - ph)); T = np.gradient(np.c_[cx, cy], axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; Nn = np.c_[-T[:, 1], T[:, 0]]; P = np.c_[cx, cy]; c.fill(np.vstack([P + Nn * w[:, None], (P - Nn * w[:, None] * .55)[::-1]]), col, a)


def halo(c, x, y, r, t=0, n_=14, col='#ffe9a8', a=.5):
    """relic rays: hard wedges, alternate lengths"""
    for i in range(n_):
        an = i * 2 * pi / n_ + t * .012; L = r * (1.0 if i % 2 else .74); w = .085 if i % 2 else .06; e = lambda q, R: A([x + math.cos(q) * R, y + math.sin(q) * R * .92]); c.fill(A([e(an - w * .45, r * .30), e(an - w, L), e(an + w, L), e(an + w * .45, r * .30)]), col, a * (1 if i % 2 else .7))


def hand_grip(d, n, c, seed=0, lw=.8):
    """hand holding chopsticks: wrist at (0, .6), knuckles up"""
    d.part(n(spl([(-.28, .62), (-.38, .24), (-.36, -.10), (-.16, -.30), (.16, -.30), (.36, -.12), (.36, .22), (.26, .62)], True, n=7)), c['skin'], c['skin2'], w=lw, seed=seed, shade=[n(spl([(-.6, -.2), (-.16, -.1), (-.06, .3), (-.1, .8), (-.6, .8)], True, n=4))])
    for i, x in enumerate((-.22, -.06, .10, .25)): SK.tub(d, n, [(x, -.18), (x + .08, -.40), (x + .20, -.40 + .06 * i)], (.105, .10, .085), c['skin'], c['skin2'], lw * .8, seed + i + 1, shs=(-.06, .06))
    SK.tub(d, n, [(.22, .34), (.36, .02), (.30, -.26)], (.14, .115, .09), c['skin'], c['skin2'], lw * .85, seed + 7, shs=(-.08, .04), cap0=.9)


def body(d, n, who, k):
    c = C[who]; st = k.get('state', 'glare'); br = k.get('br', 0.); sh = lambda *P: [n(spl(P, True, n=6))]
    TP = [(-.42, .66), (-.90, .92), (-1.46, 1.06), (-1.84, 1.50), (-1.80, 2.1), (-1.48, 2.9), (-1.40, 3.9), (1.30, 3.9), (1.36, 2.9), (1.64, 2.1), (1.70, 1.50), (1.36, 1.06), (.86, .92), (.44, .66)]
    if who == 'goku':
        B = n(spl(TP, True, n=7)); d.part(B, c['gi'], c['gi2'], w=1., seed=40, shade=sh((-2.2, .8), (-.95, 1.2), (-.62, 1.9), (-.80, 2.6), (-.5, 4.2), (-2.2, 4.2)))
        d.detail(n(spl([(-.30, .6), (.50, .6), (.34, 1.5), (.22, 1.95)], True, n=3, corners=(0, 1, 2, 3))), c['blue'], within=B, sh=c['blue2'])
        nk = n(spl([(-.40, .60), (.46, .60), (.52, .96), (.30, 1.26), (.06, 1.34), (-.22, 1.18), (-.44, .92)], True, n=6)); d.part(nk, c['skin'], c['skin2'], w=.7, seed=41, shade=sh((-.6, .5), (.5, .5), (.6, 1.02), (.1, 1.14), (-.1, 1.5), (-.6, 1.5)))
        for L in ([(.18, 1.02), (.12, 1.18), (.08, 1.32)], [(-.30, 1.00), (-.14, 1.14), (.02, 1.20)], [(.46, 1.00), (.34, 1.12), (.20, 1.18)]): d.stroke(n(spl(L, n=5)), .5, within=nk)
        for L in ([(-.50, .74), (-.16, 1.4), (.34, 2.1), (.62, 2.9)], [(.54, .74), (.42, 1.5), (.24, 1.96)]): d.stroke(n(spl(L, n=8)), .95, within=B, taper=(.02, .1))
        for L in ([(-1.3, 2.0), (-.9, 2.2), (-.6, 2.5)], [(1.4, 1.9), (1.05, 2.15), (.86, 2.5)], [(-1.2, 1.5), (-.9, 1.58)], [(1.2, 2.6), (.9, 2.8)], [(-1.1, 2.7), (-.7, 2.85)]): d.stroke(n(spl(L, n=6) if len(L) > 2 else A(L)), .6, within=B, taper=(.05, .6))
        ds = n(ell(.98, 1.62, .25, .25)); d.part(ds, '#fffaf0', '#e2c09c', w=.7); m = n.at(.98, 1.62, 0., .25)
        for L in ([(-.50, -.60), (-.50, .66)], [(-.76, -.15), (-.68, .10)], [(-.32, -.24), (-.24, -.02)], [(-.12, -.54), (.66, -.54)], [(.22, -.54), (.10, -.10)], [(-.05, -.30), (.52, -.30), (.52, -.08)], [(-.20, -.07), (.74, -.07)], [(-.02, .18), (-.02, .64)], [(-.02, .18), (.58, .18), (.58, .64)], [(-.02, .64), (.58, .64)]): d.stroke(m(A(L)), .42, taper=(.05, .15), tmin=.5, within=ds)
    else:
        B = n(spl(TP, True, n=7)); d.part(B, c['suit'], c['suit2'], w=1., seed=40, shade=sh((-2.2, .8), (-1.1, 1.2), (-.9, 1.9), (-1.0, 2.6), (-.7, 4.2), (-2.2, 4.2)))
        nk = n(spl([(-.46, .62), (.52, .62), (.60, 1.0), (.1, 1.2), (-.52, 1.0)], True, n=6)); d.part(nk, c['suit'], c['suit2'], w=.8, seed=41, shade=sh((-.6, .5), (.3, .6), (.3, .96), (-.1, 1.4), (-.7, 1.4)))
        ar = n(spl([(-.80, 1.0), (-1.18, 1.2), (-1.34, 1.9), (-1.12, 2.7), (-1.0, 4.0), (1.06, 4.0), (1.16, 2.7), (1.32, 1.9), (1.16, 1.2), (.84, 1.0), (.62, 1.36), (.1, 1.52), (-.50, 1.36)], True, n=7)); d.part(ar, c['white'], c['white2'], w=.95, seed=42, shade=sh((-1.6, .9), (-.72, 1.4), (-.66, 2.2), (-.5, 4.2), (-1.8, 4.2)))
        for sg in (-1, 1): d.part(n(spl([(sg * .78 + .03, .98), (sg * 1.10 + .03, 1.14), (sg * 1.16 + .03, 1.46), (sg * .86 + .03, 1.36)], True, n=5)), c['gold'], c['gold2'], w=.7, seed=44 + sg)
        d.part(n(spl([(-.62, 2.36), (.84, 2.36), (.78, 4.0), (-.56, 4.0)], True, n=4, corners=(0, 1, 2, 3))), c['gold'], c['gold2'], w=.8, seed=46, shade=sh((-.9, 2.2), (-.14, 2.4), (-.1, 4.2), (-.9, 4.2)))
        for y in (2.62, 2.9, 3.2): d.stroke(n(spl([(-.60, y), (.1, y + .10), (.82, y)], n=6)), .6, '#8a5418', taper=(.05, .05))
        d.stroke(n(spl([(.12, 1.52), (.16, 1.9), (.12, 2.36)], n=5)), .8, '#a87e62'); d.stroke(n(spl([(-1.2, 1.9), (-.6, 2.24), (.12, 2.30), (.8, 2.24), (1.26, 1.9)], n=8)), .7, '#a87e62')
    return B


def sleeve_arm(d, n, who, sh_pt, el, hd, seed=0, hand='grip', hr=0., hs=.82, back=False):
    c = C[who]; sh_pt, el, hd = A(sh_pt), A(el), A(hd); u = (hd - el) / (np.hypot(*(hd - el)) + 1e-6)
    if who == 'goku':
        SK.tub(d, n, [sh_pt, sh_pt * .4 + el * .6, el, el * .5 + hd * .5, hd], (.42, .39, .31, .33, .22), c['skin'], c['skin2'], .95, seed, shs=(-.15, .16)); d.stroke(n(spl([el + [-.10, .12], el + [.06, .20], el + [.2, .16]], n=5)), .5, '#a8502e')
        SK.tub(d, n, [hd - u * .42, hd - u * .02], (.29, .28), c['blue'], c['blue2'], .9, seed + 2, cap=.15, cap0=.15, shs=(-.16, .06)); S_ = tube([sh_pt - (el - sh_pt) * .10, sh_pt * .6 + el * .4, sh_pt * .42 + el * .58], (.47, .50, .46), cap=.12, cap0=.9); d.part(n(S_), c['blue'], c['blue2'], w=1., seed=seed + 3, shade=[n(S_ + [-.30, .12])])
        for o in (-.2, .1): d.stroke(n(spl([sh_pt * .3 + el * .7 + [o, -.06], sh_pt * .45 + el * .55 + [o * .8, -.05]], n=4) if False else A([sh_pt * .3 + el * .7 + [o, -.04], sh_pt * .5 + el * .5 + [o * .7, -.10]])), .5, taper=(.05, .5))
    else:
        SK.tub(d, n, [sh_pt, sh_pt * .45 + el * .55, el, el * .45 + hd * .55, hd], (.47, .42, .33, .32, .22), c['suit'], c['suit2'], 1., seed, shs=(-.18, .18)); d.stroke(n(spl([el + [-.12, .05], el + [.02, .14], el + [.18, .10]], n=5)), .6); SK.tub(d, n, [hd - u * .55, hd - u * .02], (.33, .27), c['glove'], c['glove2'], .9, seed + 2, cap=.15, cap0=.15, shs=(-.16, .06))
    hn = n.at(hd[0] + u[0] * .30, hd[1] + u[1] * .30, math.atan2(u[0], -u[1]) + hr, hs)
    cc = dict(c, skin=c['glove'], skin2=c['glove2']) if who == 'vegeta' else c
    if hand == 'grip': hand_grip(d, hn, cc, seed + 10, .8)
    elif hand == 'fist': SK.hand_fist(d, hn, cc, seed + 10)
    elif hand == 'point': SK.hand_point(d, hn, cc, seed + 10)
    elif hand == 'hold': d.part(hn(spl([(-.28, .6), (-.40, .2), (-.44, -.10), (-.2, -.22), (.3, -.20), (.44, .0), (.36, .4), (.26, .6)], True, n=6)), cc['skin'], cc['skin2'], w=.85, seed=seed + 10, shade=[hn(spl([(-.6, .2), (-.1, .16), (.2, .3), (.5, .2), (.5, .8), (-.6, .8)], True, n=4))]); [d.stroke(hn(spl([(x, -.20), (x - .01, -.02), (x + .02, .12)], n=4)), .5, taper=(.05, .4)) for x in (-.20, .02, .24)]; SK.tub(d, hn, [(.34, .30), (.46, .02), (.34, -.22)], (.13, .12, .10), cc['skin'], cc['skin2'], .7, seed + 12, cap0=.8, shs=(-.06, .06))
    return hn


def smear_arm(d, n, who, p, seed=0, cx=1.15, cy=1.55, rx=.95, ry=.95, w=.30, span=2.6):
    """the eating arm as one drawn smear: an arc with a clear head (hand) and a lifting tail, two ghost knuckles, speed lines"""
    c = C[who]; col, sh = (c['skin'], c['skin2']) if who == 'goku' else (c['glove'], c['glove2']); a1 = p * 2 * pi; u = np.linspace(0, 1, 30); an = a1 - span * (1 - u); P = np.c_[cx + rx * np.cos(an), cy + ry * np.sin(an)]
    T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None]; Nn = np.c_[-T[:, 1], T[:, 0]]; wd = w * (u ** .9) * .95 + .015; d.part(n(np.vstack([P + Nn * wd[:, None], (P - Nn * wd[:, None] * .8)[::-1]])), col, sh, w=.75, seed=seed, gaps=[(.40, .52)], shade=[n(np.vstack([P - Nn * wd[:, None] * .1, (P - Nn * wd[:, None] * .8)[::-1]]))])
    for o, q in ((.42, .5), (-.30, .35)): Q = np.c_[cx + (rx + o) * np.cos(an), cy + (ry + o) * np.sin(an)]; d.stroke(n(Q[int(30 * q):]), .5, taper=(.5, .1))
    bl = np.clip((u - .72) / .28, 0, 1)
    if who == 'goku': Q = P[18:24]; d.detail(n(np.vstack([Q + Nn[18:24] * wd[18:24, None], (Q - Nn[18:24] * wd[18:24, None] * .8)[::-1]])), c['blue'], sh=c['blue2'])
    hn = n.at(P[-1][0], P[-1][1], a1 + pi, .62); hand_grip(d, hn, dict(c, skin=col, skin2=sh), seed + 5, .75); return P[-1]


def gag(d, who, x, y, S, flip=False, k=None):
    """comic adult at the table; (x, y) = head centre.  returns the node"""
    k = dict(k or {}); n = N().at(x, y, 0., -S if flip else S, S); st = k.get('state', 'glare'); ln = k.get('lean', .0); b = n.at(0, 3.4, 0.).at(0, -3.4 + k.get('dy', 0.), 0.); br = k.get('br', 0.)
    if st != 'eat': 
        pass
    body(d, b, who, k)
    if st != 'eat': sleeve_arm(d, b, who, (-1.42, 1.46), (-1.92, 2.5), (-1.3, 3.4), 140, 'fist', hs=.8)
    hn = n.at(.12 + k.get('hx', 0.), -.02 + k.get('hy', 0.) + br * .04, k.get('roll', .04)); head(d, hn, who, k)
    if st == 'eat':
        c = C[who]; p = k.get('p', 0.); q = .06 * math.sin(p * 2 * pi); col, sh = (c['blue'], c['blue2']) if who == 'goku' else (c['suit'], c['suit2'])
        U = SK.limb_union(d, b, [(1.30, 1.40), (1.62, 1.86), (1.50, 2.16)], (.50, .46, .40)); o = b.M[:2, :2] @ A([-.2, .12]); d.part(U, col, sh, w=1., seed=70, shade=[U + o])
        hp = smear_arm(d, b, who, p, 80, cx=.92, cy=1.46, rx=.78, ry=.86, w=.34, span=2.9); e = b([(hp[0] + .02, hp[1] - .1)])[0]; chopsticks(d, e, b([(hp[0] * .4 + .12, hp[1] * .4 + .42)])[0], 7., '#c0392b' if who == 'goku' else '#2f4f9a', '#7d2018' if who == 'goku' else '#1d2f66')
        bw = b.at(-.30, 1.30 + q, -.12 + q); bowl(d, bw, .62, 0 if who == 'goku' else 1, 'rice', 60)
        sleeve_arm(d, b, who, (-1.42, 1.46), (-1.30, 2.55), (-.46, 2.06 + q), 100, 'hold', hr=.25, hs=.74)
    return b


# ---------------------------------------------------------------- room
def room_plate():
    def fn(c):
        c.grad(A([(0, 0), (W, 0), (W, H), (0, H)]), (0, 0), (0, H), [(0, '#5a2a1c', 1), (.5, '#7d3c23', 1), (1, '#4a2219', 1)])
        for x in range(-20, W, 96):                                                   # boards
            c.fill(A([(x, 0), (x + 92, 0), (x + 92, 760), (x, 760)]), ['#6a321f', '#733822', '#642e1e'][(x // 96) % 3], .7); c.fill(A([(x + 92, 0), (x + 96, 0), (x + 96, 760), (x + 92, 760)]), '#3d1c15')
        # beams
        c.fill(A([(0, 92), (W, 92), (W, 134), (0, 134)]), '#3d1c15'); c.fill(A([(0, 92), (W, 92), (W, 100), (0, 100)]), '#8a4a2a')
        for x in (310, 1570): c.fill(A([(x, 0), (x + 40, 0), (x + 40, 800), (x, 800)]), '#43201a'); c.fill(A([(x, 0), (x + 8, 0), (x + 8, 800), (x, 800)]) if x < 900 else A([(x + 32, 0), (x + 40, 0), (x + 40, 800), (x + 32, 800)]), '#7a4026')
        # window, blue hour
        for x0 in (60, 1650):
            c.fill(A([(x0 - 12, 196), (x0 + 222, 196), (x0 + 222, 470), (x0 - 12, 470)]), '#3d1c15'); c.grad(A([(x0, 208), (x0 + 210, 208), (x0 + 210, 458), (x0, 458)]), (0, 208), (0, 458), [(0, '#27406f', 1), (.6, '#46598c', 1), (1, '#b97a6a', 1)])
            c.fill(spl([(x0, 458), (x0, 410), (x0 + 60, 384), (x0 + 120, 404), (x0 + 170, 378), (x0 + 210, 398), (x0 + 210, 458)], n=6), '#1f2d52'); c.fill(A([(x0 + 100, 208), (x0 + 110, 208), (x0 + 110, 458), (x0 + 100, 458)]), '#3d1c15'); c.fill(A([(x0, 326), (x0 + 210, 326), (x0 + 210, 335), (x0, 335)]), '#3d1c15')
            for sx, sy in ((30, 240), (70, 262), (150, 236), (186, 282), (140, 300)): c.fill(ell(x0 + sx, sy, 2.4, 2.4, n=10), '#dfe6ff')
        # shelf with jars, a scroll with one brush ring
        c.fill(A([(440, 318), (700, 318), (700, 332), (440, 332)]), '#8a4a2a'); c.fill(A([(440, 332), (700, 332), (700, 340), (440, 340)]), '#3d1c15')
        for x, w, hh, col in ((470, 44, 70, '#7c5a86'), (530, 36, 52, '#5f8a72'), (580, 50, 84, '#b5653a'), (646, 38, 60, '#9a3f36')): c.fill(spl([(x - w / 2, 318), (x - w * .56, 318 - hh * .5), (x - w * .36, 318 - hh * .86), (x + w * .36, 318 - hh * .86), (x + w * .56, 318 - hh * .5), (x + w / 2, 318)], n=6), col); c.fill(A([(x - w * .3, 318 - hh), (x + w * .3, 318 - hh), (x + w * .34, 318 - hh * .84), (x - w * .34, 318 - hh * .84)]), '#e0c9a0'); c.fill(spl([(x + w * .1, 318), (x + w * .3, 318 - hh * .5), (x + w * .2, 318 - hh * .86), (x + w * .36, 318 - hh * .86), (x + w * .56, 318 - hh * .5), (x + w / 2, 318)], n=5), '#3a1814', .35)
        c.fill(A([(1242, 176), (1408, 176), (1408, 420), (1242, 420)]), '#e9d3a2'); c.fill(A([(1230, 168), (1420, 168), (1420, 182), (1230, 182)]), '#3d1c15'); c.fill(A([(1230, 414), (1420, 414), (1420, 428), (1230, 428)]), '#3d1c15'); a = np.linspace(.4, 6.1, 60); r = 46 + 3 * np.sin(a * 2); c.line(np.c_[1325 + np.cos(a) * r, 296 + np.sin(a) * r], 13, '#2a1413', taper=(.02, .35), press=.4, seed=3)
        # floor
        c.fill(A([(0, 756), (W, 756), (W, H), (0, H)]), '#5f3220')
        for k in range(9): c.fill(A([(0, 772 + k * k * 4.2), (W, 772 + k * k * 4.2), (W, 775 + k * k * 4.2 + k * .4), (0, 775 + k * k * 4.2 + k * .4)]), '#43201a')
        for sx in (1, -1): c.fill(spl([(960 + sx * 900, 800), (960 + sx * 700, 830), (960 + sx * 740, 1000), (960 + sx * 1000, 1010)], True, n=6), '#8a2f34'); c.fill(spl([(960 + sx * 900, 820), (960 + sx * 740, 846), (960 + sx * 770, 970), (960 + sx * 980, 980)], True, n=6), '#b04a42'); c.fill(A([(0, 744), (W, 744), (W, 762), (0, 762)]), '#2a1210')
    a = plate(W, H, fn, seed=12, wob=1.2)[..., :3]
    x, y = grid(); g = np.exp(-(((x - 960) / 620) ** 2 + ((y - 420) / 420) ** 2)); a = a * (.62 + .78 * g[..., None]) + g[..., None] * rgb('#ff9a3c') * .13; return a.astype(f32)


TX, TY, TRX, TRY = 960., 842., 800., 214.


def table_plate():
    def fn(c):
        c.fill(ell(TX, TY + 46, TRX + 2, TRY + 6, n=120), '#2a1210'); c.fill(ell(TX, TY + 34, TRX, TRY, n=120), '#6b3820'); c.fill(ell(TX, TY, TRX, TRY, n=120), '#d9944f')
        c.clip(ell(TX, TY, TRX - 8, TRY - 4, n=120)); r = np.random.default_rng(8)
        for k in range(130):
            yy = TY + r.uniform(-1, 1) * TRY; x0 = TX + r.uniform(-1, 1) * TRX; L = r.uniform(80, 420); c.line(spl([(x0 - L / 2, yy + r.normal(0, 3)), (x0, yy + r.normal(0, 3)), (x0 + L / 2, yy + r.normal(0, 3))], n=8), r.uniform(1.2, 3), ['#b36e36', '#c98345', '#a8622f'][k % 3], taper=(.3, .3), seed=k, alpha=.8)
        for q in (-.62, -.22, .20, .60): c.fill(A([(0, TY + q * TRY), (W, TY + q * TRY), (W, TY + q * TRY + 3), (0, TY + q * TRY + 3)]), '#8a4a26')
        c.unclip(); c.line(np.c_[TX + (TRX - 14) * np.cos(np.linspace(pi * 1.02, pi * 1.98, 80)), TY + (TRY - 8) * np.sin(np.linspace(pi * 1.02, pi * 1.98, 80))], 4, '#ffe0a8', taper=(.3, .3))
    a = plate(W, H, fn, seed=14, wob=1.0); x, y = grid(); d2 = ((x - TX) / 620) ** 2 + ((y - TY + 40) / 190) ** 2; g = np.exp(-d2 * .9)[..., None]; a[..., :3] = a[..., :3] * (.50 + .72 * g) + a[..., 3:4] * g * rgb('#ffcf87') * .20; return a


def lamp(c, t=0., on=1., x=960, sway=0.):
    s = sway * 60; c.line(spl([(x, -10), (x + s * .4, 60), (x + s, 120)], n=8), 5, '#1d0d0c', taper=(0, 0)); P = A([(-34, 108), (34, 108), (150, 206), (-150, 206)]) + [x + s, 0]; c.fill(spl(P, True, n=3, corners=(0, 1, 2, 3)), '#a8322a'); c.fill(A([(6, 108), (34, 108), (150, 206), (60, 206)]) + [x + s, 0], '#7a1f20'); c.fill(A([(-28, 112), (-12, 112), (-70, 200), (-120, 200)]) + [x + s, 0], '#d8584a'); c.fill(ell(x + s, 206, 150, 17), '#5a1818'); c.fill(ell(x + s, 206, 140, 12), mix('#5a2a18', '#fff0c0', on)); c.fill(ell(x + s, 212, 34, 24), mix('#6a4a30', '#fffbe6', on)); c.fill(ell(x + s, 102, 14, 12), '#1d0d0c')


def light(im, on=1., x=960., sway=0.):
    """amber cone and pool"""
    X, Y = grid(); s = sway * 60; u = np.clip((Y - 206) / 640, 0, 1); hw = 140 + u * 620; cone = np.clip(1 - np.abs(X - x - s * (1 - u)) / hw, 0, 1) ** .8 * (Y > 200) * np.exp(-u * 1.1); g = np.exp(-(((X - x - s) / 120) ** 2 + ((Y - 214) / 90) ** 2))
    return comp.screen(im, (cone * .20 * on)[..., None] * rgb('#ffcf87') + (g * .75 * on)[..., None] * rgb('#fff0c0'))


class Feast:
    def setup(self): self.room = room_plate(); self.table = table_plate()

    def frame_standoff(self, t=0, spark=1., kg=None, kv=None):
        S = 128; im = self.room.copy(); c = Cel(W, H, 0, 0, 1); lamp(c, t); im = comp.over(im, c.rgba()); im = light(im)
        d = D(lw=5.2, line=INK, var=.30, soft=1.0, light=(0, -1))
        kg = dict(dict(state='glare', puff=1., t=t, br=math.sin(t * .4), lag=(0, .02 * math.sin(t))), **(kg or {})); kv = dict(dict(state='glare', puff=1., t=t, br=math.sin(t * .4 + 1), lag=(0, .02)), **(kv or {}))
        for x, n_, l, sd in ((118, 15, -.10, 3), (1806, 15, .10, 5)): pass
        bg_ = gag(d, 'goku', 452, 418, S, False, kg); bv = gag(d, 'vegeta', 1468, 426, S, True, kv)
        L = d.rgba(outer=('#ff9a3c', .0, 4)); L = amber(L); im = comp.over(im, L)
        im = comp.over(im, self.table)
        e = D(lw=5.0, line=INK, var=.30, soft=1.0, light=(0, -1))
        for x, y, r, n_, l, sd in ((262, 850, 76, 12, -.08, 3), (1658, 850, 76, 12, .08, 5), (420, 944, 82, 8, -.05, 11), (1502, 946, 82, 7, .05, 13)): e.fl.fill(ell(x, y + 2, r * .95, r * .26), '#5a2a16', .55); e.sc.fill(ell(x, y + 2, r * .95, r * .26), '#5a2a16', .55); stack(e, x, y, r, n_, l, sd, sway=.04 * math.sin(t * .5 + sd))
        for x, y, r, p_ in ((600, 964, 62, 1), (1330, 968, 62, 3)): e.fl.fill(ell(x, y + r * .82, r * .9, r * .2), '#5a2a16', .5); bowl(e, N().at(x, y, .02), r, p_, 'rice', 50)
        # the relic
        bx, by = 960, 800; X, Y = grid(); rr = np.sqrt(((X - bx) / 330) ** 2 + ((Y - by + 70) / 300) ** 2); c2 = Cel(W, H, 0, 0, 1); halo(c2, bx, by - 70, 340, t, 18, a=.9); hl = c2.rgba()[..., :3] * (np.clip(1 - rr, 0, 1) ** 1.1)[..., None]; im = comp.screen(im, hl * .62 + (np.clip(1 - rr * .9, 0, 1) ** 2.2 * .42)[..., None] * rgb('#ffd27a'))
        for x, y, s_, kd, ro in ((770, 944, 66, 'bone', -.2), (1150, 956, 70, 'fish', .06), (960, 1010, 60, 'bone', .3)): leftovers(e, x, y, s_, kd, ro)
        # plate on a little stand
        for q, col, sh in ((1.0, '#f6efe2', '#c9a078'),): e.fl.fill(ell(bx, by + 44, 150, 30), '#5a2a16', .5); e.part(N().at(bx, by + 12, 0)(spl([(-60, 0), (-44, 34), (44, 34), (60, 0)], True, n=4, corners=(0, 1, 2, 3))), '#3f6fae', '#26437a', w=.8); e.part(N().at(bx, by, 0)(ell(0, 6, 138, 30)), '#f6efe2', '#c9a078', w=.95, shade=[ell(bx, by + 24, 138, 24)]); e.part(N().at(bx, by, 0)(ell(0, 2, 112, 20)), '#fff8ea', '#e6c9a4', w=.5); e.stroke(ell(bx, by + 7, 126, 25), .5, '#3f6fae', taper=(0, 0))
        bun(e, N().at(bx, by, 0.), 80)
        # arms over the table, chopsticks meeting above the bun
        tx = A([bx, by - 168]); j = 2.0 * math.sin(t * 3.3) * spark
        for b, who, sg in ((bg_, 'goku', 1), (bv, 'vegeta', -1)):
            hn = sleeve_arm(e, b, who, (1.16, 1.48), (1.74, 2.52), (2.40, 2.20), 80 if sg > 0 else 120, 'grip', hr=-.25, hs=.78); h0 = hn([(.22, -.30)])[0]; chopsticks(e, h0, tx + [-sg * 5 + j, j * .5], 9.5, '#c0392b' if sg > 0 else '#2f4f9a', '#7d2018' if sg > 0 else '#1d2f66'); hand_top(e, hn, who)
        L2 = amber(e.rgba()); im = comp.over(im, L2)
        # sparks between the eyes
        if spark > 0: im = sparks(im, (566, 440), (1356, 448), t, spark)
        c3 = Cel(W, H, 0, 0, 1); steam(c3, bx, by - 118, 62, t, a=.50); sl = c3.rgba(); im = comp.over(im, cv2.GaussianBlur(sl, (0, 0), 1.6)); c3 = Cel(W, H, 0, 0, 1); r = np.random.default_rng(4)
        for i in range(7): a = i * 2 * pi / 7 + t * .03; q = 150 + 26 * math.sin(t * .5 + i); c3.fill(SKstar(bx + math.cos(a) * q * 1.15, by - 70 + math.sin(a) * q * .8, 9 + 5 * math.sin(t * .9 + i * 2)), '#fff8d8', .95)
        q = c3.rgba(); im = comp.over(im, q)
        X, Y = grid(); im = im * (1 - .26 * np.clip((np.abs(X - 960) / 960) ** 2.4 + (np.clip(Y - 900, 0, 200) / 260) ** 2, 0, 1))[..., None]; return im


def SKstar(x, y, r): return star(x, y, max(1., r), 0., .22, 4)


def hand_top(d, hn, who):
    c = C[who]; cc = dict(c, skin=c['glove'], skin2=c['glove2']) if who == 'vegeta' else c
    for i, x in enumerate((-.22, -.06, .10, .25)): SK.tub(d, hn, [(x, -.18), (x + .08, -.40), (x + .20, -.40 + .06 * i)], (.105, .10, .085), cc['skin'], cc['skin2'], .64, 300 + i, shs=(-.06, .06))


def amber(L, lamp=(960, 230)):
    """lamp-light over a cel layer: warm near the lamp, deep umber away from it"""
    X, Y = grid(*L.shape[:2]); d = np.sqrt(((X - lamp[0]) / 900) ** 2 + ((Y - lamp[1] - 380) / 640) ** 2); g = ramp(np.clip(d, 0, 1), [(0, '#fff2d8'), (.35, '#ffe0b4'), (.7, '#e8ad7c'), (1, '#9c6658')]); out = L.copy(); out[..., :3] *= g; return out


def sparks(im, a, b, t, amt=1., seed=0):
    """two jagged bolts meeting in the middle, hard core, small burst"""
    r = np.random.default_rng(int(t) // 2 * 7 + seed); c = Cel(W, H, 0, 0, 1); a, b = A(a, float), A(b, float); m = (a + b) / 2 + r.normal(0, 14, 2)
    for p, q, col in ((a, m, '#ffd36a'), (b, m, '#9fd0ff')):
        for w, cc, al in ((15, col, .55), (6, '#fff8e0', 1.)):
            rr = np.random.default_rng(int(t) // 2 * 11 + seed + (p[0] > 960)); n_ = 11; u = np.linspace(0, 1, n_)[:, None]; L = p + (q - p) * u; L[1:-1, 1] += rr.normal(0, 20, n_ - 2); L[1:-1, 0] += rr.normal(0, 12, n_ - 2); c.line(L, w, cc, taper=(.25, .05), press=.4, seed=3, alpha=al, tmin=.2)
        for k in range(3): i = 3 + k * 2; e = L[i] + rr.normal(0, 46, 2); c.line(A([L[i], (L[i] + e) / 2 + rr.normal(0, 9, 2), e]), 3.4, '#fff3cf', taper=(0, .6))
    c.fill(star(m[0], m[1], 64 * amt, r.uniform(0, 1), .26, 7), '#fff0b0', .85); c.fill(star(m[0], m[1], 34 * amt, r.uniform(0, 1), .34, 6), '#ffffff'); L = c.rgba() * amt; g = cv2.GaussianBlur(L[..., :3], (0, 0), 16); return comp.screen(comp.over(im, L), g * .5)


# ---------------------------------------------------------------- chibi horror
def horror(d, who, x, gy, S, flip=False, t=0, jaw=1.):
    """collapse: storybook body, blank eyes, jaw on the floor, arm out pointing"""
    c = ST.C[who]; P = ST.walk(who, .25); P.update(hip=(0., -ST.PERS[who]['hip'] * .86), sq=.10, lean=-.10 + .03 * math.sin(t * 2.6), fn=A([.16, -.17]), an=0., ff=A([-.20, -.17]), af=0., yaw=.42, roll=-.10 + .02 * math.sin(t * 2.6), pitch=-.05, crossed=False, handF=(1.06, .10), hkF='flat', handN=(-.52, .62), hkN='open', lag=(-.10, .10), ex=dict(eyes='blank', mouth='scream', o=jaw, brow=-.35, bdy=-.12)); return ST.fig(d, who, x, gy, S, P, flip)


# ---------------------------------------------------------------- e3: sleeping Goku
NIGHT = dict(goku=dict(skin='#f3d3bd', skin2='#b98f98', hair='#33283a', hair2='#231b2c', hair3='#54466a', gi='#d9854a', gi2='#9c5a4a', blue='#3a4f86', blue2='#2a3862'), line='#3a2a3a', quilt='#c0888c', quilt2='#8a5a78', quilt3='#f1ead8', pillow='#f4ecdc', pillow2='#b6a9bd')


def sleep_goku(d, x, y, S, t=0, foot=0., bubble=True, blanket=1., smile=0.):
    """lying on his back, head on a pillow at the left, seen from the side and above; (x, y) = head centre.  foot 0..1 = the uncovered foot is tucked in"""
    c = NIGHT['goku']; br = math.sin(t * .13); n = N().at(x, y, 0., S); ln, d.line = d.line, NIGHT['line']
    # futon and pillow
    d.part(n(spl([(-2.0, .62), (2.0, .50), (6.4, .78), (6.8, 1.40), (2.4, 1.70), (-1.9, 1.50), (-2.4, 1.0)], True, n=8)), '#e9dfd0', '#a99bb4', w=.9, seed=1, shade=[n(spl([(-2.4, 1.16), (2, 1.36), (6.8, 1.12), (6.9, 2), (-2.4, 2)], True, n=5))])
    PW_ = n(spl([(-2.16, .50), (-2.02, -.06), (-1.2, -.20), (-.2, -.10), (.9, -.12), (1.40, .14), (1.52, .80), (1.1, 1.22), (-.2, 1.16), (-1.50, 1.24), (-2.20, 1.00)], True, n=8)); d.part(PW_, NIGHT['pillow'], NIGHT['pillow2'], w=.95, seed=2, shade=[n(spl([(-2.3, .80), (-1.2, .92), (-.2, .74), (.8, .86), (1.6, .5), (1.7, 1.4), (-2.4, 1.4)], True, n=6))])
    for xx in (-1.98, -1.80, 1.16, 1.32): d.stroke(n(spl([(xx, -.1), (xx - .06, .5), (xx + .02, 1.2)], n=6)), 1.2, '#7f8fc0', taper=(.02, .02), within=PW_)
    for L in ([(-1.40, .34), (-1.16, .44), (-.98, .40)], [(1.06, .30), (.92, .44), (.96, .6)]): d.stroke(n(spl(L, n=6)), .5, taper=(.1, .5))
    # head: rolled on the pillow, face toward us
    h = Hd(n.at(.06, -.10 + br * .015, -.62), .30, -.04, (-.05, .10 + .02 * br)); ST.hair_back(d, h, 'goku', c, 10); ST.ear(d, h, c, 1, seed=4); F = h.outline(); d.part(F, c['skin'], c['skin2'], w=1., seed=3); d.shade(h.n(spl([(-1.3, -.2), (-.72, .0), (-.66, .4), (-.36, .70), (.1, .82), (.2, 1.3), (-1.3, 1.2)], True, n=6)), within=F); ST.ear(d, h, c, -1, seed=4)
    for sg in (-1, 1): d.detail(h.F(ell(sg * .60, .42, .18, .10)), mix(c['skin'], '#f08a86', .60), within=F)
    ST.eyes(d, h, 'sleep', F); ST.brows(d, h, F, -.16, -.08, .11, .9, c['hair']); d.stroke(h.F(spl([(-.03, .33), (.04, .36), (-.0, .40)], n=5)), .5)
    o = .55 + .30 * max(0., br); M = spl([(-.15, .50), (.02, .47), (.19, .50), (.18, .56 + .12 * o), (.02, .58 + .22 * o), (-.14, .56 + .12 * o)], True, n=7); Q = h.F(M)
    if smile < .5: d.part(Q, '#8a3540', '#6a2438', w=.7); d.detail(h.F(ell(.02, .58 + .20 * o, .10, .08)), '#e08a86', within=Q); dr = h.F([(-.14, .58)])[0]; d.part(N().at(dr[0], dr[1], 0., S)(spl([(-.02, 0), (.05, .10), (.07, .22), (.0, .28), (-.05, .20)], True, n=5)), '#cfe3f4', '#93a9cf', w=.4)
    else: ST.mouth(d, h, 'smile', F)
    ST.hair_front(d, h, 'goku', c, F, 20)
    # near arm flung up by the head
    SK.tub(d, n, [(1.10, .82), (.70, 1.40), (-.40, 1.34)], (.26, .24, .20), c['skin'], c['skin2'], .95, 40, shs=(.0, .14)); ST.hand(d, n.at(-.56, 1.32, pi / 2 - .1, 1.5), c, 'open', seed=44, fl=-1)
    # quilt: one big soft hill, breathing
    B = .10 * br
    top = [(.86, .70), (1.02, .06 - B), (1.7, -.46 - B * 1.6), (2.8, -.60 - B * 1.5), (3.7, -.44 - B), (4.5, -.20), (5.2, -.36), (5.7, -.50 - .10 * foot), (6.1, -.30), (6.3, .30), (6.5, 1.06)]; q = spl(top + [(5.6, 1.34), (4.2, 1.44), (2.8, 1.58), (1.5, 1.50), (.80, 1.26)], True, n=9)
    if foot < .6:                                                                           # the foot that sticks out
        fx = n.at(6.36 - foot * .5, .74, .10); d.part(fx(spl([(-.5, -.20), (.0, -.26), (.20, -.62), (.36, -.66), (.46, -.50), (.44, .0), (.30, .26), (-.5, .24)], True, n=7)), c['skin'], c['skin2'], w=.95, seed=50, shade=[fx(spl([(-.6, .06), (.2, .06), (.6, -.1), (.6, .5), (-.6, .5)], True, n=4))])
        for i, (tx, ty, r) in enumerate(((.26, -.70, .13), (.40, -.58, .10), (.46, -.44, .09), (.49, -.31, .08), (.49, -.18, .075))): d.part(fx(ell(tx, ty, r, r * .9)), c['skin'], c['skin2'], w=.7, seed=52 + i)
    P = d.part(n(q), NIGHT['quilt'], NIGHT['quilt2'], w=1.05, seed=60, shade=[n(spl([(.6, .9), (1.6, .80), (2.6, .52), (3.6, .62), (4.4, .40), (5.2, .62), (6.0, .40), (6.8, .5), (6.8, 2), (.6, 2)], True, n=6))])
    P = n(q); cf = spl([(.84, .72), (1.02, .06 - B), (1.36, -.26 - B * 1.4), (1.66, -.18 - B), (1.40, .30), (1.26, .86), (1.24, 1.44), (.82, 1.28)], True, n=7); d.part(n(cf), NIGHT['quilt3'], '#b9adc4', w=.9, seed=62, shade=[n(spl([(1.0, .6), (1.5, .4), (1.5, 1.6), (.8, 1.6)], True, n=4))])
    for cx, cy, r in ((2.1, .30, .17), (3.1, .02, .15), (2.9, .92, .18), (4.0, .46, .16), (5.0, .30, .15), (4.9, .98, .17), (5.8, .52, .14), (3.8, 1.08, .13), (2.0, 1.12, .14)): d.detail(n(star(cx, cy, r, cx, .5)), '#f3e3b0', within=P, sh='#b7a98e')
    for L in ([(1.5, .9), (2.3, .72), (3.0, .80)], [(3.4, -.2), (3.8, .10), (4.4, .06)], [(4.6, .7), (5.2, .72), (5.8, .9)], [(5.4, -.2), (5.5, .1), (5.4, .4)]): d.stroke(n(spl(L, n=7)), .6, NIGHT['quilt2'], taper=(.2, .5), within=P)
    if bubble:
        nz = h.F([(.06, .38)], .05)[0]; r = S * (.30 + .13 * br); ce = nz + A([.94, .30]) * r; d.part(ell(ce[0], ce[1], r, r * .97, n=50), '#cfe0f2', '#a9bce0', w=.5, ink='#6d82b4'); d.fl.ctx.save(); d.detail(ell(ce[0] - r * .34, ce[1] - r * .36, r * .22, r * .13, -.6), '#ffffff', sh='#ffffff'); d.detail(ell(ce[0] + r * .34, ce[1] + r * .40, r * .30, r * .08, -.7), '#e9f2fb', sh='#e0eaf8'); d.fl.ctx.restore()
    d.line = ln; return n
