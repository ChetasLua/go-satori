"""Storybook chibis: soft brown line, flat colour, tiny eyes.  Root node = ground under the hips, units = head half-width.
    d = D(...); put(d, 'goku', x, ground_y, S, walk('goku', p), yaw=.5)          p = cycle phase (1 cycle = 2 beats = 20 f, contacts at 0 and .5)"""
import numpy as np, math
from .chibi_core import *

LINE = '#4a2a1e'
SKIN, SKIN2, BLUSH = '#ffd9b4', '#f2ad86', '#fb8f78'
C = dict(
    goku=dict(skin=SKIN, skin2=SKIN2, hair='#3b2624', hair2='#2a1a1b', hair3='#5d3f38', gi='#f58a35', gi2='#d96a26', blue='#34589a', blue2='#26437a', shoe='#34589a', shoe2='#26437a', k=1.0),
    krillin=dict(skin='#ffd6ac', skin2='#f0a97e', gi='#f58a35', gi2='#d96a26', blue='#34589a', blue2='#26437a', shoe='#34589a', shoe2='#26437a', hair='#3b2624', hair2='#2a1a1b', k=.90),
    gohan=dict(skin=SKIN, skin2=SKIN2, hair='#3b2624', hair2='#2a1a1b', hair3='#5d3f38', gi='#7651a6', gi2='#573a85', puff='#7651a6', puff2='#573a85', red='#e0483c', red2='#b5322f', white='#fdf7ea', white2='#dfd3cd', shoe='#8a5638', shoe2='#663c2b', k=.80),
    bulma=dict(skin='#ffdfc2', skin2='#f5b594', hair='#4cc3bd', hair2='#2f9aa3', hair3='#8fe0d6', dress='#f58fa8', dress2='#dc6b89', puff='#fff6ee', puff2='#ecd2d4', red='#e94b4f', red2='#be3340', belt='#8a5a3c', belt2='#643d2b', shoe='#d2545a', shoe2='#a83c48', k=.97),
    vegeta=dict(skin=SKIN, skin2=SKIN2, hair='#352220', hair2='#261818', hair3='#57392f', suit='#3157ab', suit2='#233f84', white='#fcf6e8', white2='#ddd0c6', gold='#f4bd45', gold2='#d1922c', shoe='#fcf6e8', shoe2='#ddd0c6', k=.95),
    piccolo=dict(skin='#86c46c', skin2='#629f57', gi='#7550a6', gi2='#573a85', white='#fdf8ee', white2='#dcd4d8', red='#d94a4a', red2='#a9343d', blue='#5fb3d9', shoe='#9a5f3a', shoe2='#70422c', pink='#f0a58c', pink2='#d07d70', k=1.06),
    roshi=dict(skin='#fdd2a8', skin2='#eca67a', white='#fffaf0', white2='#dcd3d0', shirt='#f9b93a', shirt2='#dd9226', flower='#fff1c8', shorts='#8fb7a6', shorts2='#6b958a', shell='#7c8f57', shell2='#5c6f45', shellrim='#f3e3bd', wood='#b3783f', wood2='#87552f', shoe='#b3604a', shoe2='#87443a', k=.97),
)

# ---------------------------------------------------------------- hair (3-D blades; image-left = the character's right)
HAIR = dict(
    goku=lockset([
        ((-.66, -.66, .10), (-1.10, -1.98, -.22), (.16, -.96, -.05), (.20, .02, 0), .10, 'S'),
        ((-.94, -.30, .10), (-2.02, -1.10, -.30), (-.50, -.82, -.02), (.06, -.16, 0), .09, 'S'),
        ((-.97, .12, .0), (-1.84, -.10, -.35), (-.92, -.42, .0), (0, -.10, 0), .09, 'S'),
        ((-.84, .44, -.25), (-1.36, .46, -.5), (-.96, .08, -.2), (0, -.04, 0), .08, 's'),
        ((.50, -.82, -.10), (1.80, -.70, -.40), (.95, -.26, .0), (.0, -.16, 0), .09, 'S'),
        ((.92, -.34, -.05), (1.80, .02, -.42), (.97, .08, -.05), (0, -.10, 0), .09, 'S'),
        ((.95, .06, -.15), (1.36, .46, -.5), (.86, .42, -.2), (0, -.06, 0), .08, 's'),
        ((-.46, -.82, -.40), (-.50, -1.62, -1.30), (.36, -.84, -.40), (0, -.10, 0), .10, 'B'),
        ((-.55, -.50, -.65), (-.62, -.78, -1.98), (.35, -.56, -.75), (0, -.10, -.1), .10, 'B'),
        ((-.40, -.10, -.80), (.10, -.10, -1.90), (.62, -.20, -.65), (0, -.10, -.1), .10, 'B'),
        ((-.55, .20, -.70), (-.30, .46, -1.55), (.40, .22, -.75), (0, -.02, -.1), .10, 'B'),
        ((-.82, -.40, .46), (-.66, .12, .84), (-.46, -.60, .64), (-.06, .0, .06), .08, 'F'),
        ((-.56, -.58, .62), (-.20, .06, 1.02), (-.04, -.70, .70), (-.05, .0, .08), .08, 'F'),
        ((-.12, -.70, .70), (.26, -.14, 1.02), (.34, -.66, .66), (.06, 0, .08), .08, 'F'),
        ((.28, -.66, .66), (.66, -.10, .84), (.76, -.46, .52), (.06, 0, .05), .08, 'F')]),
    vegeta=lockset([
        ((-.46, -.80, .30), (-.30, -2.02, -.20), (.14, -.84, .35), (-.12, 0, 0), .13, 'S'),
        ((-.14, -.84, .35), (.36, -2.16, -.25), (.46, -.80, .30), (.02, 0, 0), .13, 'S'),
        ((-.84, -.50, .10), (-.86, -1.74, -.35), (-.16, -.86, .20), (-.12, 0, 0), .12, 'S'),
        ((.16, -.86, .20), (.92, -1.64, -.40), (.84, -.50, .10), (.10, 0, 0), .12, 'S'),
        ((-.99, -.10, -.05), (-1.26, -1.18, -.45), (-.66, -.70, .0), (-.10, .02, 0), .12, 'S'),
        ((.66, -.70, .0), (1.26, -1.06, -.50), (.99, -.10, -.05), (.10, .02, 0), .12, 'S'),
        ((-.5, -.6, -.45), (-.42, -1.66, -1.10), (.3, -.75, -.45), None, .12, 'B'),
        ((-.2, -.6, -.6), (.42, -1.40, -1.30), (.66, -.45, -.5), None, .12, 'B'),
        ((-.62, -.2, -.7), (-.60, -.92, -1.52), (.1, -.35, -.8), (0, 0, -.1), .12, 'B'),
        ((-.1, -.1, -.8), (.34, -.56, -1.62), (.66, -.1, -.65), (0, 0, -.1), .12, 'B'),
        ((-.6, .25, -.65), (-.10, .16, -1.35), (.5, .25, -.65), (0, .05, -.1), .12, 'B')]),
    gohan=lockset([
        ((-.98, -.12, .1), (-1.36, .34, -.15), (-.90, .30, -.1), (-.02, -.06, 0), .10, 'S'),
        ((-.95, .1, -.2), (-1.18, .66, -.55), (-.74, .46, -.4), (-.04, -.04, 0), .10, 'S'),
        ((.98, -.12, .1), (1.36, .34, -.15), (.90, .30, -.1), (.02, -.06, 0), .10, 'S'),
        ((.95, .1, -.2), (1.18, .66, -.55), (.74, .46, -.4), (.04, -.04, 0), .10, 'S'),
        ((-.6, .2, -.7), (-.45, .78, -1.12), (.0, .3, -.85), None, .10, 'B'), ((-.1, .3, -.85), (.35, .74, -1.12), (.6, .2, -.7), None, .10, 'B'),
        ((-.7, -.1, -.6), (-.86, .34, -1.26), (-.2, .2, -.85), None, .1, 'B'),
        ((-.80, -.22, .48), (-.70, .16, .80), (-.50, -.40, .66), (-.04, 0, .05), .08, 'F'), ((-.56, -.36, .70), (-.34, .10, .95), (-.18, -.44, .80), (-.04, 0, .05), .08, 'F'),
        ((-.24, -.42, .80), (.02, .04, 1.0), (.14, -.44, .80), (.03, 0, .05), .08, 'F'), ((.10, -.44, .80), (.38, .10, .95), (.50, -.38, .70), (.05, 0, .05), .08, 'F'), ((.46, -.38, .70), (.70, .14, .80), (.80, -.22, .48), (.04, 0, .04), .08, 'F')]),
    bulma=lockset([
        ((-1.0, -.2, .15), (-1.02, .86, .05), (-.80, .1, .0), (-.12, .0, 0), .10, 'S'), ((-.98, .0, -.25), (-.86, .98, -.45), (-.5, .2, -.6), (-.14, 0, -.1), .10, 'S'),
        ((.80, .1, .0), (.98, .80, .10), (1.0, -.2, .15), (.10, .0, 0), .08, 'S'),
        ((-.5, .1, -.7), (-.2, .92, -.95), (.4, .1, -.7), (0, 0, -.1), .12, 'B'),
        ((-.86, -.40, .42), (-.80, .20, .72), (-.48, -.62, .60), (-.08, 0, .06), .08, 'F'), ((-.60, -.58, .58), (-.46, .06, .92), (-.12, -.72, .68), (-.08, 0, .08), .08, 'F'),
        ((-.26, -.68, .66), (-.02, .02, 1.02), (.24, -.70, .68), (-.04, 0, .08), .08, 'F'), ((.10, -.70, .68), (.46, .04, .95), (.60, -.58, .58), (.03, 0, .08), .08, 'F'), ((.46, -.62, .60), (.82, .18, .70), (.88, -.38, .40), (.06, 0, .06), .08, 'F')]),
)
HLINE = dict(goku=[(-.98, .06), (-.90, -.28), (-.62, -.52), (-.2, -.62), (.2, -.62), (.62, -.52), (.90, -.28), (.98, .06)],
             vegeta=[(-.985, .10), (-.93, -.22), (-.70, -.56), (-.40, -.70), (-.20, -.62), (0, -.30), (.20, -.62), (.40, -.70), (.70, -.56), (.93, -.22), (.985, .10)],
             gohan=[(-.98, .06), (-.88, -.18), (-.5, -.34), (0, -.38), (.5, -.34), (.88, -.18), (.98, .06)],
             bulma=[(-.98, .10), (-.90, -.26), (-.6, -.54), (0, -.64), (.6, -.54), (.90, -.26), (.98, .10)])


def hair_back(d, h, who, c, seed=0, col=None, w=1.0):
    L = [q for q in HAIR[who] if q['tag'] != 'F' and lock_z(h, q) < -.12]; L.sort(key=lambda q: lock_z(h, q))
    if not L: return
    P = [lock_poly(h, q) for q in L] + [h.outline(.02)]
    for U in d.union(P)[:1]: d.part(U, col or c['hair'], c['hair2'], w=w, seed=seed)


def hair_front(d, h, who, c, face, seed=0, w=1.0, hi=True):
    L = [q for q in HAIR[who] if q['tag'] != 'F' and lock_z(h, q) >= -.12]; FL = [q for q in HAIR[who] if q['tag'] == 'F' and h.rot(q['tip'])[0, 2] > -.1]
    line = spl(HLINE[who], n=8) if who != 'vegeta' else spl(HLINE[who], n=6, corners=(5,))
    cp, vis = cap_poly(h, line); sk = h.outline(.015)
    polys = [lock_poly(h, q) for q in L + FL]; clips = [None] * len(polys)
    if cp is not None: polys.append(cp); clips.append(sk)
    Us = d.union(polys, clips)
    if not Us: return None
    U = Us[0]
    if face is not None:                                             # the bangs' shadow on the skin
        s = h.n.s; d.shade([np.asarray(q) + [.015 * s, .085 * s] for q in Us], within=face)
    d.part(Us, c['hair'], c['hair2'], w=w, seed=seed)
    if hi and 'hair3' in c:
        for i, q in enumerate(L + FL):
            F = lock2(h, q); n = 13; Aa, B = F[:n], F[n - 1:][::-1]
            u = .34 if i % 2 else .5; Cc = Aa * (1 - u) + B * u; a, b = (2, 9) if q['tag'] != 'F' else (3, 8)
            G = Cc[a:b]; d.stroke(h.n(G), .55 if q['tag'] != 'F' else .4, c['hair3'], taper=(.4, .55), seed=seed + i)
    return Us


# ---------------------------------------------------------------- face
def eyes(d, h, kind, face, look=(0, 0), k=1., blink=0., dy=0., sep=.37):
    for sg in (-1, 1):
        if h.z(sg * sep, .16) < .12: continue
        x, y = sg * sep + look[0] * .06, .17 + dy + look[1] * .05
        if blink > .5 or kind in ('happy', 'sleep'):
            s = -1 if kind == 'happy' and blink <= .5 else 1
            d.stroke(h.F(spl([(x - .10, y + .02 * s), (x, y + .06 * s), (x + .10, y + .02 * s)], n=8)) if s > 0 else h.F(spl([(x - .10, y + .03), (x, y - .06), (x + .10, y + .03)], n=8)), .85, taper=(.15, .15), tmin=.5, within=face)
        elif kind == 'blank':
            Q = h.F(ell(x, y - .02, .155, .20)); d.part(Q, '#ffffff', '#e6e0ee', w=.8); d.detail(h.F(ell(x + .03, y, .028, .034)), '#33201c', within=Q, sh='#33201c')
        elif kind == 'line': d.stroke(h.F([(x - .09, y), (x, y + .005), (x + .09, y)]), .9, within=face)
        else:
            rx, ry = .064 * k, .100 * k
            d.detail(h.F(ell(x, y, rx, ry)), '#33201c', within=face, sh='#33201c'); d.detail(h.F(ell(x + .016, y - .036, .022, .028)), '#ffffff', within=face, sh='#ffffff')


def mouth(d, h, kind, face, x=0., y=.43, k=1., o=1.):
    if kind in ('open', 'sing', 'laugh'):
        s = dict(open=(.20, .25), sing=(.11, .22), laugh=(.25, .31))[kind]; w_, hh = s[0] * k, s[1] * k * o
        M = spl([(x - w_, y - .02), (x - w_ * .5, y - .005), (x, y + .0), (x + w_ * .5, y - .005), (x + w_, y - .02), (x + w_ * .78, y + hh * .62), (x, y + hh), (x - w_ * .78, y + hh * .62)], True, n=7, corners=(0, 4))
        if kind == 'sing': M = ell(x, y + hh * .4, w_, hh * .62)
        Q = h.F(M); d.detail(Q, '#9d2f35', within=face, sh='#9d2f35'); d.detail(h.F(ell(x, y + hh * .80, w_ * .62, hh * .42)), '#f28d84', within=Q, sh='#f28d84'); d.contour(Q, .62)
    elif kind == 'scream':
        M = spl([(x - .20, y - .06), (x, y - .10), (x + .20, y - .06), (x + .24, y + .20 * o), (x + .18, y + .44 * o), (x, y + .50 * o), (x - .18, y + .44 * o), (x - .24, y + .20 * o)], True, n=7); Q = h.n(h.loc(M)); d.part(Q, '#7a1f2a', '#5a1420', w=.8); d.detail(h.n(h.loc(ell(x, y + .40 * o, .15, .14))), '#f28d84', within=Q, sh='#d86a6a')
    elif kind == 'smile': d.stroke(h.F(spl([(x - .13, y + .0), (x, y + .07), (x + .13, y + .0)], n=8)), .7, taper=(.15, .15), tmin=.5, within=face)
    elif kind == 'flat': d.stroke(h.F([(x - .09, y + .04), (x, y + .045), (x + .09, y + .035)]), .7, within=face)
    elif kind == 'frown': d.stroke(h.F(spl([(x - .12, y + .075), (x, y + .02), (x + .12, y + .075)], n=8)), .75, taper=(.15, .15), tmin=.5, within=face)
    elif kind == 'pout': d.stroke(h.F(spl([(x - .10, y + .05), (x - .04, y + .02), (x, y + .05), (x + .05, y + .02), (x + .10, y + .06)], n=6)), .7, within=face)
    elif kind == 'o': Q = h.F(ell(x, y + .06, .06 * k, .075 * k)); d.detail(Q, '#9d2f35', within=face, sh='#9d2f35'); d.contour(Q, .55)


def brows(d, h, face, ang=0., y=-.06, w=.11, wt=.9, col=None, sep=.37, dy=0.):
    for sg in (-1, 1):
        if h.z(sg * sep, y) < .1: continue
        s = math.sin(ang)
        L = [(sg * sep - sg * w, y + dy + s * w), (sg * sep, y + dy - .02), (sg * sep + sg * w, y + dy - s * w)]
        d.stroke(h.F(spl(L, n=8)), wt, col, taper=(.1, .3), tmin=.45, within=face)


def blush(d, h, face, c, y=.40, x=.62, a=.75):
    for sg in (-1, 1): d.detail(h.F(ell(sg * x, y, .15, .085)), mix(c['skin'], BLUSH, a), within=face, sh=mix(c['skin2'], BLUSH, a))


def ear(d, h, c, sg, y=.17, s=1., seed=0):
    R = h.rot([[sg * .96, y, -.02]])[0]
    P = h.n(ell(R[0] + .04 * sg, R[1], .135 * s, .185 * s, rot=.1 * sg)); d.part(P, c['skin'], c['skin2'], w=.9, seed=seed)
    q = h.n(spl([(R[0] + sg * .07, R[1] - .09 * s), (R[0] + sg * .0, R[1] - .05 * s), (R[0] + sg * .0, R[1] + .05 * s)], n=6)); d.stroke(q, .5, mix(LINE, c['skin2'], .35))


def face_base(d, h, c, seed=0, blusha=.75, shade=True):
    F = h.outline(); sg = 1 if h.yaw >= 0 else -1
    s = h.n.s; sh = [np.vstack([F + [0, .0], (F * 1 + [-.14 * s * sg, -.10 * s])[::-1]])] if False else []
    if shade:
        O = h.outline(raw=True); sh = [h.n(np.vstack([O, (O + [.05 * sg, -.10])[::-1]]))]; sh = []
    d.part(F, c['skin'], c['skin2'], shade=sh, w=1.0, seed=seed)
    d.shade(h.n(spl([(-1.1, .52), (-.6, .72), (0, .80), (.6, .72), (1.1, .50), (1.1, 1.1), (-1.1, 1.1)], True, n=6)), within=F)
    blush(d, h, F, c, a=blusha); return F


# ---------------------------------------------------------------- heads
def head_goku(d, h, c, ex, seed=0):
    hair_back(d, h, 'goku', c, seed)
    far, near = (1, -1) if h.yaw >= 0 else (-1, 1)
    if h.z(far * .9, .17) > -.3 or abs(h.yaw) < .2: ear(d, h, c, far, seed=seed)
    F = face_base(d, h, c, seed + 1); ear(d, h, c, near, seed=seed + 2)
    eyes(d, h, ex.get('eyes', 'dot'), F, ex.get('look', (0, 0)), blink=ex.get('blink', 0)); brows(d, h, F, ex.get('brow', .05), -.08, .12, 1.0, c['hair'])
    mouth(d, h, ex.get('mouth', 'open'), F, o=ex.get('o', 1.)); hair_front(d, h, 'goku', c, F, seed + 5)


def head_krillin(d, h, c, ex, seed=0):
    far, near = (1, -1) if h.yaw >= 0 else (-1, 1)
    if abs(h.yaw) < .4: ear(d, h, c, far, seed=seed)
    F = face_base(d, h, c, seed + 1); ear(d, h, c, near, seed=seed + 2)
    eyes(d, h, ex.get('eyes', 'dot'), F, ex.get('look', (0, 0)), blink=ex.get('blink', 0)); brows(d, h, F, ex.get('brow', -.02), -.10, .10, .9, '#4a2a1e')
    mouth(d, h, ex.get('mouth', 'open'), F, o=ex.get('o', 1.))
    for y in (-.66, -.50, -.34):
        for x in (-.11, .11): d.detail(h.F(ell(x, y, .035, .042)), '#4a2a1e', within=F, sh='#4a2a1e')
    d.detail(h.n(ell(-.42 if h.yaw >= 0 else .42, -.62, .20, .08, rot=-.5 if h.yaw >= 0 else .5)), '#fff3de', within=F, sh='#fff3de')


def ball4(d, n, r=1., n_star=4, spin=0., face=(0., 0.), w=1., seed=0):
    """dragon ball: node at its centre, r local radius.  spin rolls the stars about the view axis, face = (x, y) tilt of the star face"""
    P = n(ell(0, 0, r, r)); s = n.s * r
    d.part(P, '#f9a52e', '#e07d1c', shade=[n(spl([(-1.1 * r, .2 * r), (-.45 * r, .58 * r), (.2 * r, .52 * r), (.62 * r, .12 * r), (.78 * r, -.5 * r), (1.2 * r, -.4 * r), (1.2 * r, 1.2 * r), (-1.1 * r, 1.2 * r)], True, n=6))], w=w, seed=seed)
    d.detail(n(spl([(-.55 * r, .45 * r), (0, .74 * r), (.52 * r, .50 * r), (.2 * r, .60 * r), (-.2 * r, .60 * r)], True, n=6)), '#ffc65a', within=P, sh='#f09a30')
    lay = {1: [(0, 0)], 2: [(-.24, 0), (.24, 0)], 3: [(0, -.26), (-.26, .18), (.26, .18)], 4: [(-.25, -.25), (.25, -.25), (-.25, .25), (.25, .25)], 5: [(0, 0), (-.32, -.28), (.32, -.28), (-.32, .28), (.32, .28)],
           6: [(-.28, -.3), (.28, -.3), (-.32, 0.02), (.32, .02), (-.2, .36), (.2, .36)], 7: [(0, 0), (-.36, 0), (.36, 0), (-.2, -.34), (.2, -.34), (-.2, .34), (.2, .34)]}[n_star]
    cs, sn = math.cos(spin), math.sin(spin)
    for x, y in lay:
        X, Y = x * cs - y * sn + face[0], x * sn + y * cs + face[1]; q = X * X + Y * Y
        if q > .62: continue
        kx = math.sqrt(1 - q)
        d.detail(n(star(X * r, Y * r, .165 * r * (.75 + .25 * kx), rot=-pi / 2 + spin)), '#d8362c', within=P, sh='#b02a26')
    d.detail(n(ell(-.36 * r, -.42 * r, .22 * r, .13 * r, rot=-.6)), '#fff6d8', within=P, sh='#fff6d8'); d.detail(n(ell(-.60 * r, -.12 * r, .06 * r, .06 * r)), '#fff6d8', within=P, sh='#fff6d8')


def head_gohan(d, h, c, ex, seed=0):
    hair_back(d, h, 'gohan', c, seed); far, near = (1, -1) if h.yaw >= 0 else (-1, 1)
    F = face_base(d, h, c, seed + 1); ear(d, h, c, near, y=.22, s=.9, seed=seed + 2)
    eyes(d, h, ex.get('eyes', 'dot'), F, ex.get('look', (0, 0)), k=1.12, blink=ex.get('blink', 0), dy=.04); brows(d, h, F, ex.get('brow', -.1), -.02, .09, .75, c['hair'])
    mouth(d, h, ex.get('mouth', 'open'), F, y=.46, o=ex.get('o', 1.)); hair_front(d, h, 'gohan', c, F, seed + 5)
    # red hat: crown, turned-up brim, the four-star ball on top
    O = h.outline(.05, raw=True); ya = -.30
    cr = [(-1.06, ya + .06), (-1.05, -.52), (-.88, -.92), (-.48, -1.20), (0, -1.28), (.48, -1.20), (.88, -.92), (1.05, -.52), (1.06, ya + .06), (.5, ya + .12), (0, ya + .13), (-.5, ya + .12)]
    tilt = A([-.10 * h.sy, 0.]); P = h.n(spl(A(cr) + tilt * A([[1, 0]]) * 0, True, n=7))
    d.shade(h.n(spl([(-1.1, ya + .02), (0, ya + .26), (1.1, ya + .02), (1.1, -.9), (-1.1, -.9)], True, n=6)), within=F)
    d.part(P, c['red'], c['red2'], shade=[h.n(spl([(1.2, -1.3), (.62, -.95), (.40, -.55), (.45, -.2), (1.2, -.1)], True, n=6))], w=1., seed=seed + 11)
    for a_, m_, b_ in [((-.10, -1.24), (-.40, -.85), (-.46, -.34)), ((.12, -1.24), (.34, -.8), (.36, -.34))]: d.stroke(h.n(spl(A([a_, m_, b_]) + [.18 * h.sy, 0], n=8)), .5, mix(c['red2'], LINE, .5), taper=(.1, .4))
    br = spl([(-1.12, ya - .10), (-.6, ya + .0), (0, ya + .03), (.6, ya + .0), (1.12, ya - .10), (1.17, ya + .08), (1.06, ya + .18), (.55, ya + .27), (0, ya + .30), (-.55, ya + .27), (-1.06, ya + .18), (-1.17, ya + .08)], True, n=7)
    d.part(h.n(br), mix(c['red'], '#ff9a78', .25), c['red2'], w=1., seed=seed + 12)
    bn = h.n.at(.10 * h.sy, -1.40 + (ex.get('ball', 0.)), .0)
    d.part(bn(ell(0, .16, .24, .10)), c['red2'], w=.9, seed=seed + 14); ball4(d, bn, .32, 4, face=(.25 * h.sy, .04), w=1., seed=seed + 13)


def head_bulma(d, h, c, ex, seed=0):
    sw = ex.get('swing', (0., 0.))
    # ponytail (back of the crown, tied high), behind everything
    b0 = h.rot([[-.10, -.92, -.42]])[0, :2]; o_ = -max(-1., min(1., h.yaw / .35))
    t = A([b0, b0 + [o_ * .34, -.30], b0 + [o_ * .86 + sw[0] * .35, -.12 + sw[1] * .4], b0 + [o_ * 1.12 + sw[0] * .8, .48 + sw[1] * .9], b0 + [o_ * 1.02 + sw[0] * 1.2, 1.10 + sw[1] * 1.3]])
    d.part(h.n(tube(t, (.14, .26, .33, .26, .02))), c['hair'], c['hair2'], w=1., seed=seed + 30, shade=[h.n(tube(t + [o_ * -.16, .14], (.14, .24, .28, .2, .02)))])
    for o in (-.4, .30): d.stroke(h.n(spl(t[1:] + [o * .30, abs(o) * .1], n=8)), .55, c['hair3'] if o < 0 else c['hair2'], taper=(.3, .5))
    hair_back(d, h, 'bulma', c, seed); far, near = (1, -1) if h.yaw >= 0 else (-1, 1)
    F = face_base(d, h, c, seed + 1, .85)
    eyes(d, h, ex.get('eyes', 'dot'), F, ex.get('look', (0, 0)), k=1.08, blink=ex.get('blink', 0))
    for sg in (-1, 1):
        if h.z(sg * .37, .16) > .12 and ex.get('blink', 0) < .5: d.stroke(h.F([(sg * .43, .11), (sg * .50, .08)]), .6, within=F)
    brows(d, h, F, ex.get('brow', -.05), -.08, .09, .6, c['hair2']); mouth(d, h, ex.get('mouth', 'open'), F, k=.85, o=ex.get('o', 1.))
    hair_front(d, h, 'bulma', c, F, seed + 5)
    # ribbon
    r = h.n.at(b0[0] + .02, b0[1] - .06, -.25 + sw[0] * .2)
    for sg in (-1, 1):
        d.part(r(spl([(0, 0), (sg * .18, -.22), (sg * .46, -.28), (sg * .52, .0), (sg * .40, .16), (sg * .16, .10)], True, n=6, corners=(0,))), c['red'], c['red2'], shade=[r(spl([(0, 0), (sg * .2, .0), (sg * .5, .04), (sg * .4, .3), (0, .2)], True, n=5))], w=.95, seed=seed + 20)
        d.stroke(r(spl([(sg * .08, -.02), (sg * .22, -.08), (sg * .34, -.10)], n=6)), .45, c['red2'])
    d.part(r(ell(0, .0, .12, .12)), c['red'], c['red2'], w=.9, seed=seed + 22)


def head_vegeta(d, h, c, ex, seed=0):
    hair_back(d, h, 'vegeta', c, seed); far, near = (1, -1) if h.yaw >= 0 else (-1, 1)
    if abs(h.yaw) < .35: ear(d, h, c, far, seed=seed)
    F = face_base(d, h, c, seed + 1); ear(d, h, c, near, seed=seed + 2)
    eyes(d, h, ex.get('eyes', 'dot'), F, ex.get('look', (0, 0)), k=ex.get('ek', 1.), blink=ex.get('blink', 0), dy=.02); brows(d, h, F, ex.get('brow', .42), -.04, .15, 1.25, c['hair'], dy=ex.get('bdy', 0.))
    if ex.get('red', 0) > 0: blush(d, h, F, c, a=min(1.3, .75 + ex['red'] * .5))
    mouth(d, h, ex.get('mouth', 'frown'), F, o=ex.get('o', 1.)); hair_front(d, h, 'vegeta', c, F, seed + 5)


def head_piccolo(d, h, c, ex, seed=0):
    far, near = (1, -1) if h.yaw >= 0 else (-1, 1)
    def pear(sg, s0):
        R = h.rot([[sg * .95, .10, -.05]])[0]; e = h.n.at(R[0], R[1], 0.); k = 1. if sg == near else .8
        P = e(spl([(-sg * .06, -.18), (sg * .20, -.26), (sg * .62 * k, -.34 + ex.get('earw', 0)), (sg * .42 * k, .0), (sg * .18, .20), (-sg * .04, .18)], True, n=7, corners=(2,)))
        d.part(P, c['skin'], c['skin2'], shade=[e(spl([(0, .06), (sg * .3, -.04), (sg * .5, -.2), (sg * .5, .3), (0, .3)], True, n=5))], w=1., seed=s0); d.stroke(e(spl([(sg * .46 * k, -.22), (sg * .2, -.08), (sg * .06, .06)], n=6)), .5, mix(LINE, c['skin2'], .4))
    if abs(h.yaw) < 1.0: pear(far, seed)
    F = h.outline(); d.part(F, c['skin'], c['skin2'], w=1., seed=seed + 1); d.shade(h.n(spl([(-1.1, .52), (-.6, .72), (0, .80), (.6, .72), (1.1, .50), (1.1, 1.1), (-1.1, 1.1)], True, n=6)), within=F)
    blush(d, h, F, dict(skin=c['skin'], skin2=c['skin2']), a=.22); pear(near, seed + 2)
    eyes(d, h, ex.get('eyes', 'dot'), F, ex.get('look', (0, 0)), blink=ex.get('blink', 0), dy=.02); brows(d, h, F, ex.get('brow', .28), -.06, .15, 1.2, '#3d6b3a')
    mouth(d, h, ex.get('mouth', 'smile'), F, y=.46, o=ex.get('o', 1.))
    # turban: wrapped band, dome
    dm = spl([(-.86, -.62), (-.74, -1.02), (-.36, -1.30), (.06, -1.36), (.46, -1.26), (.78, -.98), (.86, -.60)], n=8)
    d.part(h.n(np.vstack([dm, [(0, -.4)]])), c['gi'], c['gi2'], shade=[h.n(spl([(.25, -1.5), (.45, -1.0), (.4, -.4), (1.2, -.4), (1.2, -1.5)], True, n=5))], w=1., seed=seed + 10)
    d.shade(h.n(spl([(-1.1, -.30), (0, -.08), (1.1, -.30), (1.1, -.9), (-1.1, -.9)], True, n=6)), within=F)
    tb = spl([(-1.10, -.40), (-1.08, -.72), (-.78, -.92), (-.3, -.98), (.3, -.98), (.78, -.92), (1.08, -.72), (1.10, -.40), (.95, -.24), (.45, -.16), (0, -.14), (-.45, -.16), (-.95, -.24)], True, n=7)
    d.part(h.n(tb), c['white'], c['white2'], shade=[h.n(spl([(-1.2, -.34), (-.5, -.26), (.2, -.26), (.75, -.44), (.82, -.9), (1.3, -.9), (1.3, 0), (-1.2, 0)], True, n=6))], w=1., seed=seed + 11)
    k = .25 * h.sy
    for L in ([(-1.06, -.60), (-.3 + k, -.52), (.4 + k, -.70), (.8, -.90)], [(-1.0, -.32), (-.2 + k, -.36), (.6 + k, -.50), (1.06, -.70)], [(-.7, -.90), (-.1 + k, -.74), (.3 + k, -.64), (1.0, -.36)]): d.stroke(h.n(spl(L, n=8)), .55, '#a89a9c', taper=(.12, .3))


def head_roshi(d, h, c, ex, seed=0):
    far, near = (1, -1) if h.yaw >= 0 else (-1, 1)
    if abs(h.yaw) < .4: ear(d, h, c, far, seed=seed)
    F = face_base(d, h, c, seed + 1); ear(d, h, c, near, seed=seed + 2, s=1.1)
    for y in (-.60, -.48): d.stroke(h.F(spl([(-.28, y), (0, y - .035), (.28, y)], n=8)), .45, mix(c['skin2'], LINE, .25), taper=(.3, .3), within=F)
    d.detail(h.n(ell(-.46 if h.yaw >= 0 else .46, -.60, .20, .08, rot=-.5 if h.yaw >= 0 else .5)), '#fff3de', within=F, sh='#fff3de')
    sw = ex.get('swing', (0., 0.))
    mouth(d, h, ex.get('mouth', 'open'), F, y=.46, k=.8, o=ex.get('o', 1.))
    # beard: one scalloped mass from ear to ear, moustache on top
    bc = h.loc([(0, .56)])[0]
    def Bp(P): P = np.atleast_2d(A(P, float)); lim = .99 if abs(h.yaw) < .1 else .99 - .40 * abs(h.sy); P = P.copy(); P[:, 0] = np.minimum(P[:, 0] * (1 if h.yaw >= 0 else -1), lim) * (1 if h.yaw >= 0 else -1); Q = h.loc(P, .04); w = np.clip((Q[:, 1] - .45) / .9, 0, 1.2)[:, None] ** 1.3; return h.n(Q + w * A(sw))
    be = spl([(-.98, .12), (-1.04, .46), (-.90, .78), (-.60, .92), (-.52, 1.10), (-.26, 1.08), (-.06, 1.36), (.10, 1.12), (.34, 1.14), (.52, .94), (.84, .80), (1.0, .46), (.96, .12), (.80, .34), (.48, .50), (.20, .60), (0, .64), (-.20, .60), (-.48, .50), (-.80, .34)], True, n=6, corners=(3, 5, 7, 9))
    bp = h.loc(be, .04); d.part(Bp(be), c['white'], c['white2'], shade=[Bp(spl([(.9, .2), (.5, .62), (.2, .86), (-.5, .92), (-.3, 1.5), (1.2, 1.4), (1.2, .2)], True, n=6))], w=1., seed=seed + 6)
    for a_, b_ in (((-.5, .62), (-.46, .84)), ((-.14, .72), (-.10, 1.0)), ((.28, .68), (.28, .92))): d.stroke(Bp(spl([a_, ((a_[0] + b_[0]) / 2 + .02, (a_[1] + b_[1]) / 2), b_], n=6)), .5, '#b7a9a6', taper=(.3, .5))
    mo = spl([(0, .40), (.22, .33), (.48, .40), (.62, .56), (.40, .55), (.18, .50), (0, .46), (-.18, .50), (-.40, .55), (-.62, .56), (-.48, .40), (-.22, .33)], True, n=6, corners=(3, 9)); d.part(h.F(mo, .06), c['white'], c['white2'], w=.85, seed=seed + 7)
    # nose and shades
    for sg in (-1, 1):
        if h.z(sg * .38, .1) > .05:
            G = h.F(ell(sg * .37, .13, .25, .20, sq=2.6), .05); d.part(G, '#4b2f36', '#33202a', hi=[(h.F(spl([(sg * .37 - .16, .10), (sg * .37 - .04, -.02), (sg * .37 + .02, -.02), (sg * .37 - .10, .16)], True, n=5), .05), '#8d6a78')], w=1.05, ink='#a53a35', seed=seed + 8)
    d.stroke(h.F(spl([(-.12, .10), (0, .07), (.12, .10)], n=6), .06), 1.1, '#a53a35', taper=(0, 0))
    R = h.rot([[near * .62, .10, h.dome([[near * .62, .1]])[0]]])[0]; R2 = h.rot([[near * .97, .10, -.02]])[0]; d.stroke(h.n(A([R[:2], (R[:2] + R2[:2]) / 2, R2[:2]])), 1.0, '#a53a35', taper=(0, 0), within=F)
    for sg in (-1, 1):
        if h.z(sg * .36, -.2) > .1: d.part(h.F(spl([(sg * .18, -.14), (sg * .34, -.24), (sg * .54, -.16), (sg * .46, -.08), (sg * .30, -.12)], True, n=5), .05), c['white'], c['white2'], w=.7, seed=seed + 9)


HEADS = dict(goku=head_goku, krillin=head_krillin, gohan=head_gohan, bulma=head_bulma, vegeta=head_vegeta, piccolo=head_piccolo, roshi=head_roshi)


# ---------------------------------------------------------------- hands, feet, limbs
def hand(d, n, c, kind='fist', col=None, sh=None, seed=0, fl=1, w=.9):
    """n at the wrist, +y toward the fingers; fl = thumb side"""
    col, sh = col or c['skin'], sh or c['skin2']; f = lambda P: n(np.asarray(P, float) * [fl, 1])
    if kind == 'fist':
        P = f(spl([(-.11, -.02), (-.17, .08), (-.17, .20), (-.10, .28), (.03, .295), (.13, .25), (.165, .14), (.12, .0)], True, n=7)); d.part(P, col, sh, w=w, seed=seed, shade=[f([(-.3, .22), (.3, .19), (.3, .4), (-.3, .4)])])
        for x in (-.06, .03): d.stroke(f(spl([(x, .215), (x - .005, .25), (x - .01, .285)], n=5)), .5, taper=(.3, .1))
        T = f(spl([(.09, .03), (.17, .06), (.19, .14), (.12, .19), (.03, .18), (.02, .12), (.08, .10)], True, n=6)); d.part(T, col, sh, w=w * .8, seed=seed + 1, gaps=[(.86, .06)])
    elif kind == 'open':
        P = f(spl([(-.10, -.02), (-.17, .10), (-.22, .22), (-.19, .31), (-.13, .26), (-.10, .36), (-.04, .40), (.0, .32), (.03, .39), (.09, .38), (.10, .28), (.13, .22), (.25, .20), (.27, .12), (.16, .08), (.11, .0)], True, n=5)); d.part(P, col, sh, w=w, seed=seed)
        for x0, x1 in ((-.11, -.125), (.01, .0)): d.stroke(f([(x1, .30 if x0 < 0 else .33), ((x0 + x1) / 2, .22), (x0 * .9, .18)]), .45, taper=(.1, .5))
    elif kind == 'flat':
        P = f(spl([(-.10, -.02), (-.15, .10), (-.15, .24), (-.08, .35), (.02, .36), (.10, .28), (.13, .18), (.20, .12), (.17, .04), (.10, .0)], True, n=6)); d.part(P, col, sh, w=w, seed=seed)
        for x in (-.07, .015): d.stroke(f([(x, .34), (x, .24)]), .45, taper=(.1, .5))
    elif kind == 'point':
        P = f(spl([(-.11, -.02), (-.16, .08), (-.15, .19), (-.06, .24), (.08, .24), (.15, .18), (.15, .06), (.11, .0)], True, n=7)); d.part(f(tube([(-.08, .14), (-.09, .30), (-.08, .46)], (.065, .06, .05), cap=.9)), col, sh, w=w, seed=seed + 2); d.part(P, col, sh, w=w, seed=seed, gaps=[(.36, .46)])
        for x in (.0, .07): d.stroke(f(spl([(x, .17), (x, .21), (x - .01, .235)], n=5)), .5, taper=(.3, .1))
        d.part(f(tube([(.14, .06), (.08, .14), (-.02, .15)], (.055, .05, .045), cap=.9)), col, sh, w=w * .75, seed=seed + 3)
    elif kind == 'grip':                                                  # knuckles round a vertical pole at x = 0
        P = f(spl([(-.12, -.04), (-.17, .06), (-.16, .20), (-.06, .26), (.08, .25), (.17, .18), (.17, .04), (.10, -.03)], True, n=6)); d.part(P, col, sh, w=w, seed=seed)
        for y in (.05, .115, .18): d.stroke(f(spl([(.16, y), (.09, y + .01), (.02, y - .005)], n=5)), .5, taper=(.1, .4))
    return


def shoe(d, n, c, kind='boot', seed=0, far=False):
    """n at the ankle, ground .17 below, toe +x"""
    col, sh = c['shoe'], c['shoe2']
    if kind == 'boot':
        P = spl([(-.15, -.10), (.0, -.115), (.13, -.09), (.15, .0), (.27, .03), (.38, .09), (.385, .165), (.15, .175), (-.17, .175), (-.19, .08)], True, n=7, corners=(0, 2)); Q = n(P)
        d.part(Q, col, sh, shade=[n([(-.3, .07), (.1, .10), (.5, .08), (.5, .3), (-.3, .3)])], w=1., seed=seed)
        d.stroke(n(spl([(-.185, .12), (.1, .128), (.385, .12)], n=6)), .6)
    elif kind == 'vboot':
        P = spl([(-.17, -.24), (.0, -.25), (.15, -.22), (.145, -.02), (.26, .02), (.38, .085), (.385, .165), (.15, .175), (-.17, .175), (-.185, .04)], True, n=7, corners=(0, 2)); Q = n(P)
        d.part(Q, c['white'], c['white2'], shade=[n([(-.3, .08), (.1, .10), (.5, .08), (.5, .3), (-.3, .3)])], w=1., seed=seed)
        d.part(n(spl([(.20, -.01), (.30, .02), (.385, .085), (.39, .165), (.24, .172), (.19, .08)], True, n=6)), c['gold'], c['gold2'], w=.8, seed=seed + 1); d.stroke(n(spl([(-.17, -.05), (.0, -.03), (.145, -.05)], n=6)), .5)
    elif kind == 'sandal':
        P = spl([(-.13, -.08), (.11, -.08), (.15, .02), (.28, .05), (.36, .10), (.36, .155), (-.17, .155), (-.17, .04)], True, n=7); d.part(n(P), c['skin'], c['skin2'], w=.95, seed=seed)
        d.part(n(spl([(-.20, .125), (.1, .13), (.40, .125), (.41, .19), (.1, .195), (-.21, .19)], True, n=6)), col, sh, w=.85, seed=seed + 1); d.stroke(n(spl([(.05, .13), (.14, .04), (.24, .06), (.30, .13)], n=6)), 1.2, col, taper=(0, 0))


def leg(d, hip, ank, c, col, sh, fa=0., kind='boot', l1=.36, l2=.36, baggy=1., seed=0, cuff=True, n=None, far=False, ktick=True):
    kn, _ = ik(hip, ank, l1, l2, -1.); s = n
    if kind == 'sandal' or kind == 'bare': w = (.20, .18, .13, .12)
    elif kind == 'vboot': w = (.25, .24, .19, .16)
    else: w = (.26 * baggy, .275 * baggy, .27 * baggy, .20 * baggy)
    foot = s.at(ank[0], ank[1], fa)
    if kind in ('boot',): pass
    up = A(ank) - (A(ank) - kn) * (.10 if kind == 'boot' else 0.)
    if kind == 'vboot':
        T = tube([hip, hip * .5 + kn * .5 + [.0, 0], kn, up], w); d.part(s(T), col, sh, w=1., seed=seed, shade=[s(T + [-.16, .02])] if not far else [s(T)]); shoe(d, foot, c, kind, seed + 1)
    else:
        shoe(d, foot, c, kind, seed + 1); T = tube([hip, hip * .5 + kn * .5 + [.03, 0], kn + [.02, 0], up], w, cap=.35)
        d.part(s(T), col, sh, w=1., seed=seed, shade=[s(T + [-.22, .05])] if not far else [s(T)])
        if ktick and kind == 'boot':
            e = A(ank) - kn; L = np.hypot(*e) + 1e-6; e /= L; nn = A([-e[1], e[0]])
            for o, ln in ((-.09, .13), (.05, .10)): d.stroke(s(spl([up + nn * o - e * .02, up + nn * (o * 1.3) - e * ln * .6, up + nn * (o * 1.2) - e * ln], n=5)), .5, taper=(.1, .5))
    return kn


def arm(d, n, sh_pt, hd_pt, c, kind='gi', hk='fist', seed=0, bend=1., l1=.30, l2=.32, fl=1, hrot=None, hs=1., far=False):
    """shoulder -> hand target in n's units"""
    el, hp = ik(sh_pt, hd_pt, l1, l2, bend); u = A(hp) - el; u = u / (np.hypot(*u) + 1e-6); rot = math.atan2(-u[0], u[1]) if hrot is None else hrot
    hn = n.at(hp[0], hp[1], rot, hs)
    P = [sh_pt, A(sh_pt) * .5 + el * .5, el, hp]
    if kind == 'gi':                                                    # skin forearm, wristband, short blue sleeve
        hand(d, hn, c, hk, seed=seed, fl=fl); T = tube([el - u * .06, el * .5 + hp * .5, hp], (.115, .12, .105)); d.part(n(T), c['skin'], c['skin2'], w=.95, seed=seed + 1)
        B = tube([el * .30 + hp * .70, el * .06 + hp * .94 + u * .03], (.135, .14)); d.part(n(B), c['blue'], c['blue2'], w=.9, seed=seed + 2, shade=[n(B)] if far else [])
        S = tube([sh_pt, A(sh_pt) * .4 + el * .6, el + (el - sh_pt) * .06], (.185, .19, .175), cap=.25, cap0=.8); d.part(n(S), c['blue'], c['blue2'], w=1., seed=seed + 3, shade=[n(S)] if far else [n(S + [-.14, .03])])
    elif kind == 'sleeve':                                              # long sleeve in c['arm'], optional cuff
        hand(d, hn, c, hk, c.get('hand'), c.get('hand2'), seed=seed, fl=fl); T = tube([sh_pt, A(sh_pt) * .45 + el * .55, el, el * .12 + hp * .88], (.17, .165, .15, .135), cap0=.8); d.part(n(T), c['arm'], c['arm2'], w=1., seed=seed + 1, shade=[n(T)] if far else [n(T + [-.12, .02])])
        if 'cuff' in c: B = tube([el * .22 + hp * .78, hp + u * .03], (.15, .165)); d.part(n(B), c['cuff'], c['cuff2'], w=.9, seed=seed + 2)
    elif kind == 'bare':
        hand(d, hn, c, hk, seed=seed, fl=fl); T = tube([sh_pt, A(sh_pt) * .45 + el * .55, el, hp], (.14, .14, .12, .105), cap0=.9); d.part(n(T), c['skin'], c['skin2'], w=.95, seed=seed + 1, shade=[n(T)] if far else [])
        if 'puff' in c: S = tube([sh_pt, A(sh_pt) * .45 + el * .55], (.19, .20), cap=.35, cap0=.8); d.part(n(S), c['puff'], c['puff2'], w=.95, seed=seed + 3, shade=[n(S)] if far else [n(S + [-.14, .04])])
    return el, hp, hn


TB = [(-.19, -.04), (-.38, .02), (-.46, .24), (-.45, .52), (-.47, .80), (-.22, .875), (.22, .875), (.47, .80), (.45, .52), (.46, .24), (.38, .02), (.19, -.04)]


def torso(d, t, c, col, sh, seed=0, k=0., B=None):
    P = t(spl(A(B if B is not None else TB) + [0, 0], True, n=7)); d.part(P, col, sh, shade=[t(spl([(-.6, .16), (-.28, .30), (-.22, .62), (-.30, .95), (-.7, .95)], True, n=6))] if k >= 0 else [t(spl([(.6, .16), (.28, .30), (.22, .62), (.30, .95), (.7, .95)], True, n=6))], w=1., seed=seed); return P


def kanji(d, t, x, y, r, dark='#33201c'):
    """white disc, tiny brush mark 悟 (a suggestion, hand drawn)"""
    P = t(ell(x, y, r, r)); d.part(P, '#fffaf0', '#e6d9d2', w=.6); q = lambda L, w=.42: d.stroke(t(A(L) * r + [x, y]), w, dark, taper=(.1, .2), tmin=.5, within=P)
    q([(-.50, -.55), (-.50, .62)]); q([(-.72, -.15), (-.66, .08)]); q([(-.34, -.22), (-.28, -.05)]); q([(-.12, -.50), (.62, -.50)]); q([(.22, -.5), (.12, -.08)]); q([(-.05, -.28), (.50, -.28), (.50, -.06)]); q([(-.18, -.06), (.70, -.06)]); q([(-.02, .18), (-.02, .60)]); q([(-.02, .18), (.55, .18), (.55, .60)]); q([(-.02, .60), (.55, .60)])


def belt(d, t, c, col, sh, k, seed=0, sw=0., y=.62):
    P = t(spl([(-.475, y), (-.2, y + .035), (.2, y + .035), (.475, y), (.485, y + .13), (.2, y + .17), (-.2, y + .17), (-.485, y + .13)], True, n=6, corners=(0, 3, 4, 7))); d.part(P, col, sh, w=.95, seed=seed)
    kx = k + .02
    for sg, a in ((-1, .28), (1, -.20)):
        e = A([kx + sg * .10 + sw * .20, y + .42 - abs(sg) * .02]); d.part(t(tube([(kx, y + .09), (kx + sg * .07, y + .24), e + [sg * .03, 0]], (.05, .065, .085))), col, sh, w=.85, seed=seed + 2 + sg)
    d.part(t(spl([(kx - .08, y + .02), (kx + .08, y + .02), (kx + .09, y + .15), (kx - .09, y + .15)], True, n=5)), col, sh, w=.85, seed=seed + 4)


def body_gi(d, t, c, k, sw, seed=0, mark='go'):
    B = torso(d, t, c, c['gi'], c['gi2'], seed, k)
    d.detail(t(spl([(k - .20, -.08), (k + .20, -.08), (k + .05, .26), (k - .0, .30)], True, n=5, corners=(0, 1, 2, 3))), c['blue'], within=B, sh=c['blue2'])
    for L in ([(k - .24, -.06), (k - .08, .18), (k + .10, .46), (k + .14, .62)], [(k + .22, -.06), (k + .12, .14), (k + .015, .33)]): d.stroke(t(spl(L, n=7)), .75, within=B, taper=(.05, .15), tmin=.4)
    d.stroke(t(spl([(k - .17, -.06), (k - .02, .18), (k + .03, .25)], n=6)), .5, c['blue2'], within=B)
    for L in ([(-.40, .42), (-.28, .50), (-.22, .60)], [(.40, .46), (.32, .54), (.28, .61)], [(-.30, .80), (-.22, .76)], [(.30, .80), (.24, .765)]): d.stroke(t(spl(L, n=5) if len(L) > 2 else L), .5, within=B, taper=(.1, .6))
    kanji(d, t, k + .235, .22, .10); belt(d, t, c, c['blue'], c['blue2'], k, seed + 5, sw)


def body_gohan(d, t, c, k, sw, seed=0):
    B = torso(d, t, c, c['gi'], c['gi2'], seed, k); d.stroke(t(spl([(k - .2, -.04), (k - .02, .24), (k + .12, .60)], n=6)), .7, within=B); belt(d, t, c, c['red'], c['red2'], k, seed + 5, sw)
    S = t(spl([(-.44, -.06), (-.2, .02), (.1, .03), (.44, -.06), (.48, .10), (.30, .24), (.0, .28), (-.30, .24), (-.48, .10)], True, n=6)); d.part(S, c['white'], c['white2'], shade=[t([(-.6, .13), (0, .20), (.6, .10), (.6, .4), (-.6, .4)])], w=.95, seed=seed + 8)
    for L in ([(-.36, .04), (-.1, .13), (.14, .11)], [(.40, .04), (.22, .16), (.0, .20)]): d.stroke(t(spl(L, n=6)), .45, '#b7a9a6', taper=(.1, .5))


def body_bulma(d, t, c, k, sw, seed=0):
    sk = [(-.18, -.03), (-.36, .03), (-.39, .24), (-.37, .50), (-.44, .72), (-.56 + sw * .08, 1.02), (-.2, 1.10), (.2, 1.10), (.58 + sw * .08, 1.00), (.44, .72), (.37, .50), (.39, .24), (.36, .03), (.18, -.03)]
    B = t(spl(sk, True, n=7, corners=(5, 8))); d.part(B, c['dress'], c['dress2'], shade=[t(spl([(-.7, .2), (-.24, .30), (-.22, .6), (-.34, 1.2), (-.8, 1.2)], True, n=6))], w=1., seed=seed)
    for x, x2 in ((-.22, -.30), (.02, .02), (.26, .33)): d.stroke(t(spl([(x, .72), (x2 * .9, .88), (x2, 1.05)], n=5)), .5, c['dress2'], taper=(.3, .2), within=B)
    for sg in (-1, 1): d.part(t(spl([(k, .02), (k + sg * .20, -.07), (k + sg * .27, .09), (k + sg * .10, .16)], True, n=5, corners=(0, 1, 2))), '#fff6ee', '#e9cfd2', w=.8, seed=seed + 3)
    d.part(t(ell(k, .08, .055, .05)), c['red'], c['red2'], w=.7)
    b = t(spl([(-.39, .50), (0, .55), (.39, .50), (.42, .62), (0, .68), (-.42, .62)], True, n=6, corners=(0, 2, 3, 5))); d.part(b, c['belt'], c['belt2'], w=.9, seed=seed + 5)
    d.part(t(spl([(-.30, .58), (-.08, .59), (-.07, .80), (-.16, .84), (-.30, .80)], True, n=5)), c['belt'], c['belt2'], w=.9, seed=seed + 6, shade=[t([(-.4, .72), (0, .72), (0, .9), (-.4, .9)])]); d.stroke(t(spl([(-.30, .66), (-.18, .70), (-.07, .66)], n=5)), .55); d.detail(t(ell(-.185, .71, .022, .022)), c['dress'])
    d.part(t(spl([(k + .05, .53), (k + .17, .53), (k + .17, .65), (k + .05, .65)], True, n=4)), '#f4c65a', '#d19a34', w=.7)


def body_vegeta(d, t, c, k, sw, seed=0):
    B = torso(d, t, c, c['suit'], c['suit2'], seed, k)
    ar = spl([(-.30, -.04), (-.44, .06), (-.47, .28), (-.43, .56), (-.36, .66), (0, .70), (.36, .66), (.43, .56), (.47, .28), (.44, .06), (.30, -.04), (.16, .06), (0, .09), (-.16, .06)], True, n=7)
    Q = t(ar); d.part(Q, c['white'], c['white2'], shade=[t(spl([(-.6, .1), (-.30, .26), (-.26, .5), (-.30, .8), (-.7, .8)], True, n=5))], w=1., seed=seed + 2)
    d.part(t(spl([(k - .26, .38), (k + .26, .38), (k + .24, .68), (k, .71), (k - .24, .68)], True, n=5, corners=(0, 1, 2, 4))), c['gold'], c['gold2'], shade=[t([(k - .4, .58), (k + .4, .56), (k + .4, .8), (k - .4, .8)])], w=.85, seed=seed + 4)
    for y in (.475, .575): d.stroke(t(spl([(k - .25, y), (k, y + .025), (k + .25, y)], n=5)), .55, '#a86c22', taper=(.1, .1))
    d.stroke(t(spl([(k, .10), (k + .01, .24), (k, .37)], n=5)), .6, '#a3918c'); d.stroke(t(spl([(-.40, .22), (k - .18, .36), (k, .365), (k + .18, .36), (.42, .22)], n=6)), .6, '#a3918c')
    for sg in (-1, 1): d.detail(t(spl([(sg * .30, -.03), (sg * .40, .04), (sg * .44, .13), (sg * .34, .08)], True, n=5)), c['gold'], within=Q, sh=c['gold2'])


def body_piccolo(d, t, c, k, sw, seed=0):
    B = torso(d, t, c, c['gi'], c['gi2'], seed, k, A(TB) * [.98, 1.22]); P2 = t(spl([(k - .18, -.06), (k + .18, -.06), (k + .04, .36), (k, .38)], True, n=4, corners=(0, 1, 2, 3))); d.detail(P2, c['skin'], within=B, sh=c['skin2'])
    for L in ([(k - .20, -.06), (k - .02, .34)], [(k + .20, -.06), (k + .03, .34)]): d.stroke(t(L), .7, within=B)
    b = t(spl([(-.47, .80), (0, .86), (.47, .80), (.49, .95), (0, 1.02), (-.49, .95)], True, n=6, corners=(0, 2, 3, 5))); d.part(b, c['blue'], mix(c['blue'], '#2a4a7a', .4), w=.95, seed=seed + 5)
    d.part(t(tube([(k + .05, .90), (k + .10, 1.10), (k + .13 + sw * .15, 1.34)], (.06, .075, .09))), c['blue'], mix(c['blue'], '#2a4a7a', .4), w=.85, seed=seed + 6)


def body_roshi(d, t, c, k, sw, seed=0):
    # shell on the back is drawn by fig(); open tropical shirt over a white vest
    B = torso(d, t, c, c['shirt'], c['shirt2'], seed, k); d.detail(t(spl([(k - .10, -.06), (k + .16, -.06), (k + .13, .9), (k - .05, .9)], True, n=4, corners=(0, 1, 2, 3))), c['white'], within=B, sh=c['white2'])
    for L in ([(k - .12, -.06), (k - .08, .4), (k - .06, .88)], [(k + .18, -.06), (k + .15, .4), (k + .14, .88)]): d.stroke(t(spl(L, n=5)), .7, within=B)
    for x, y, r in ((-.30, .22, .085), (-.24, .60, .075), (.36, .34, .07), (.33, .70, .08), (-.38, .46, .05)):
        for a in range(5): d.detail(t(ell(x + math.cos(a * 1.257) * r * .62, y + math.sin(a * 1.257) * r * .62, r * .48, r * .48)), c['flower'], within=B, sh=mix(c['flower'], c['shirt2'], .4))
        d.detail(t(ell(x, y, r * .28, r * .28)), '#e8743a', within=B)


BODY = dict(goku=body_gi, krillin=body_gi, gohan=body_gohan, bulma=body_bulma, vegeta=body_vegeta, piccolo=body_piccolo, roshi=body_roshi)

# ---------------------------------------------------------------- walks
PERS = dict(
    roshi=dict(A=.36, lift=.12, bob=.035, sq=.03, lean=.17, arm=.35, roll=.05, hbob=.03, hip=.74, yaw=.44, tl=.84),
    krillin=dict(A=.46, lift=.30, bob=.060, sq=.05, lean=-.06, arm=.85, roll=.035, hbob=.05, hip=.82, yaw=.46, tl=.84),
    goku=dict(A=.50, lift=.22, bob=.085, sq=.07, lean=.04, arm=.80, roll=.09, hbob=.06, hip=.86, yaw=.50, tl=.86),
    gohan=dict(A=.46, lift=.26, bob=.13, sq=.09, lean=.02, arm=.95, roll=.10, hbob=.07, hip=.78, yaw=.46, tl=.80),
    bulma=dict(A=.42, lift=.14, bob=.055, sq=.035, lean=-.04, arm=.60, roll=.05, hbob=.04, hip=.90, yaw=.48, tl=.86),
    vegeta=dict(A=.48, lift=.24, bob=.05, sq=.06, lean=-.10, arm=.5, roll=.02, hbob=.03, hip=.86, yaw=.48, tl=.92),
    piccolo=dict(A=.54, lift=.18, bob=.030, sq=.015, lean=.0, arm=.2, roll=.015, hbob=.015, hip=1.18, yaw=.50, tl=1.20, hs=.94))
AH, HEEL, BALL, DUTY = .17, -.15, .20, .56


def speed(who):
    """ground speed in head units per frame (so ground and feet agree)"""
    return 2 * PERS[who]['A'] / DUTY / 20.


def foot_at(p, Aa, lift):
    """ankle position relative to the ground point under the hips, foot angle (+ heel up)"""
    p = p % 1.; R = lambda a, v: A([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)])
    def st(u):
        gx = Aa * (1 - 2 * u); a = -.45 * (1 - E(u / .14)) if u < .14 else (0. if u < .60 else .85 * E((u - .60) / .40))
        pv = HEEL if a < 0 else BALL; v = A([pv, 0.]) + R(a, A([-pv, -AH])) if a != 0 else A([0., -AH]); return v + [gx, 0], a
    if p < DUTY: return st(p / DUTY)
    u = (p - DUTY) / (1 - DUTY); a0, q0 = st(1.)[0], st(1.)[1]; a1, q1 = st(0.)[0], st(0.)[1]; e = u * u * (3 - 2 * u)
    v = a0 * (1 - e) + a1 * e; v[1] -= lift * math.sin(pi * min(1, u ** .85)); ang = q0 * (1 - E(u * 1.3)) + q1 * E((u - .45) / .55); return v, ang


def walk(who, p, **kw):
    """pose of a marching character at phase p (near foot strikes at p = 0, far foot at .5)"""
    g = dict(PERS[who]); g.update(kw); p = p % 1. if who != 'x' else p; th = 4 * pi * (p - .055); c, s = math.cos(th), math.sin(th)
    fn, an = foot_at(p, g['A'], g['lift']); ff, af = foot_at(p + .5, g['A'], g['lift']); w = math.cos(2 * pi * p)
    P = dict(p=p, A=g['A'], hip=(0., -g['hip'] + g['bob'] * c), sq=g['sq'] * c, lean=g['lean'] + .02 * s, fn=fn, an=an, ff=ff, af=af, arm=g['arm'] * w, roll=g['roll'] * math.sin(2 * pi * (p - .06)) + g['lean'] * -.3,
             hdy=g['hbob'] * math.cos(th - .9), lag=(-.07 - .03 * w, .13 * math.sin(th - .5) * (g['bob'] / .08) ** .6), yaw=g['yaw'], pitch=.02 + .03 * math.cos(th - 1.2), sw=.5 * math.sin(th - 1.2) - .25, tw=w, ex={}, t=p * 20)
    if who == 'goku': P['ex'] = dict(mouth='open', o=.85 + .25 * math.cos(th - 1.0))
    elif who == 'krillin': P['ex'] = dict(mouth='open', o=.8 + .3 * c)
    elif who == 'gohan': P['ex'] = dict(mouth='laugh', eyes='dot', ball=-.03 * math.sin(th - .9))
    elif who == 'bulma': P['ex'] = dict(mouth='open', swing=(-.10 + .20 * math.sin(2 * pi * (p - .16)), -.14 * math.sin(th - .9)))
    elif who == 'vegeta': P['ex'] = dict(mouth='frown', look=(.3, 0.))
    elif who == 'piccolo': P['ex'] = dict(mouth='smile', eyes='dot')
    elif who == 'roshi': P['ex'] = dict(mouth='open', o=.8, swing=(-.06 + .10 * math.sin(2 * pi * (p - .2)), -.06 * math.sin(th - .9)))
    return P


# ---------------------------------------------------------------- whole figure
def fig(d, who, x, gy, S, P, flip=False, seed=None):
    """draw one marching figure; P from walk() (any field may be overridden)"""
    c = C[who]; seed = sum(map(ord, who)) if seed is None else seed
    S = S * c['k']; g = PERS[who]; n = N().at(x, gy, 0., S * (-1 if flip else 1), S)
    hx, hy = P['hip']; sq = P.get('sq', 0.); tl = g.get('tl', .86); k = .20 * math.sin(P.get('yaw', .5)) / .48
    hipn = n.at(hx, hy, P.get('lean', 0.), 1 + sq * .6, 1 - sq); t = hipn.at(0, -tl + .06)                    # torso node at the neck base
    local = lambda q: n.inv(t(A(q, float)))                                                                  # torso -> root units
    sw = P.get('sw', 0.); tw = P.get('tw', 0.)
    far_off, up = .13, -.045                                                                                 # far leg: a little ahead and up the picture
    hipN, hipF = A([hx - .12, hy - .02]), A([hx + .14, hy - .05]); aN, aF = A(P['fn']) + [-.02, 0.], A(P['ff']) + [.04, up]
    l = (g['hip'] - .10) / 2; pk = dict(l1=l, l2=l)
    shN, shF = local((-.36, .17)), local((.34, .15))
    swing = P.get('arm', 0.)
    def hp(sh, a, ln=.60): return sh + A([math.sin(a), math.cos(a) * 1.0]) * ln * .92
    aNh, aFh = hp(shN, -swing * .95 + .10), hp(shF, swing * .95 + .10); bN, bF = 1., 1.
    if 'handN' in P: aNh = local(P['handN'])
    if 'handF' in P: aFh = local(P['handF'])
    hd = hipn.at(.07 * math.sin(P.get('yaw', .5)) + P.get('hx', 0.), -tl + .06 - .76 + P.get('hdy', 0.), P.get('roll', 0.) - P.get('lean', 0.) * .6, g.get('hs', 1.))
    H_ = Hd(hd, P.get('hyaw', P.get('yaw', .5)), P.get('pitch', 0.), P.get('lag', (0, 0)), P.get('t', 0.), .03); ex = P.get('ex', {})
    armc = dict(gi='gi')
    # ------------- behind
    if who == 'piccolo':
        q = P.get('cape', None); s1, s2 = sw * .10 + tw * .05, sw * .16 + tw * .10
        cp = spl([(-.42, .0), (-.66, .30), (-.78 + s1, .85), (-.86 + s2, 1.50), (-.40 + s2, 1.62), (.0 + s2 * .6, 1.52), (.36 + s1, 1.60), (.60 + s1, 1.46), (.56, .80), (.50, .20), (.40, 0)], True, n=7, corners=(3, 5, 7)); d.part(t(cp), c['white'], c['white2'], shade=[t(cp)], w=1., seed=seed + 60)
    if who == 'roshi':
        sh_ = t.at(-.36, .40, .05); P_ = sh_(ell(0, 0, .50, .56, sq=2.2)); d.part(P_, c['shellrim'], mix(c['shellrim'], '#a9825a', .5), w=1., seed=seed + 60); Q_ = sh_(ell(-.03, 0, .40, .46, sq=2.2)); d.part(Q_, c['shell'], c['shell2'], shade=[sh_(ell(-.10, .30, .55, .40))], w=.85, seed=seed + 61)
        for L in ([(-.16, -.14), (.06, -.16), (.16, .02), (.04, .18), (-.18, .15), (-.26, .0), (-.16, -.14)], ): d.stroke(sh_(A(L)), .7, taper=(0, 0))
        for a_, b_ in (((-.16, -.14), (-.22, -.36)), ((.06, -.16), (.12, -.38)), ((.04, .18), (.10, .40)), ((-.18, .15), (-.24, .36)), ((-.26, 0), (-.42, .0))): d.stroke(sh_([a_, b_]), .6, taper=(0, 0))
    # ------------- far arm
    def arms(sh_p, hp_, far, fl):
        ak = P.get('armk', 1.); kw = dict(seed=seed + (10 if far else 20), far=far, fl=fl, l1=.30 * ak, l2=.32 * ak, hs=P.get('hs', 1.))
        if who in ('goku', 'krillin'): arm(d, n, sh_p, hp_, c, 'gi', P.get('hkF' if far else 'hkN', 'fist'), bend=1., **kw)
        elif who == 'gohan': cc = dict(c, blue=c['red'], blue2=c['red2']); arm(d, n, sh_p, hp_, dict(cc, blue=c['red'], blue2=c['red2']), 'bare', P.get('hkF' if far else 'hkN', 'fist'), bend=1., **kw)
        elif who == 'bulma': arm(d, n, sh_p, hp_, c, 'bare', P.get('hkF' if far else 'hkN', 'flat'), bend=1., **kw)
        elif who == 'vegeta': cc = dict(c, arm=c['suit'], arm2=c['suit2'], hand=c['white'], hand2=c['white2'], cuff=c['white'], cuff2=c['white2']); arm(d, n, sh_p, hp_, cc, 'sleeve', P.get('hkF' if far else 'hkN', 'fist'), bend=1., **kw)
        elif who == 'roshi': cc = dict(c, arm=c['shirt'], arm2=c['shirt2']); arm(d, n, sh_p, hp_, cc, 'sleeve', P.get('hkF' if far else 'hkN', 'fist'), bend=1., **dict(kw, l1=.50 if far else .30, l2=.52 if far else .32))
        elif who == 'piccolo': arm(d, n, sh_p, hp_, dict(c, blue=c['red'], blue2=c['red2']), 'bare', P.get('hkF' if far else 'hkN', 'fist'), bend=1., **kw)
    if who == 'vegeta' and P.get('crossed', True): pass
    elif who == 'piccolo' and not P.get('showarms'): pass
    elif who == 'roshi': pass
    else: arms(shF, aFh, True, 1)
    # ------------- staff (Roshi): far hand holds it in front
    if who == 'roshi':
        sp = P.get('staff', None); sx0 = 1.50
        if sp is None:
            q = (P['p'] + .5) % 1.; u = q / DUTY
            gA = P.get('A', g['A']); tip = A([sx0 + gA * (1 - 2 * u), up]) if u <= 1 else A([sx0 - gA + 2 * gA * E((q - DUTY) / (1 - DUTY)), up - .24 * math.sin(pi * (q - DUTY) / (1 - DUTY))])
        else: tip = A(sp)
        hh = local((1.18, .22)); hh = A([hh[0] + .10 + (tip[0] - sx0) * .30, hh[1]]); ax = hh - tip; ax /= np.hypot(*ax); top = tip + ax * 2.25; nn = A([-ax[1], ax[0]])
        arms(shF, hh - ax * .10, True, 1)
        stf = tube([tip, tip + ax * .75 + nn * .02, tip + ax * 1.5 - nn * .02, top], (.05, .06, .065, .085)); d.part(n(stf), c['wood'], c['wood2'], w=.95, seed=seed + 70, shade=[n(stf + [-.05, 0])])
        kb = n.at(top[0], top[1], math.atan2(ax[0], -ax[1])); d.part(kb(spl([(-.10, .10), (-.20, -.06), (-.16, -.26), (.02, -.34), (.18, -.24), (.20, -.04), (.10, .10)], True, n=6)), c['wood'], c['wood2'], w=1., seed=seed + 71, shade=[kb(ell(-.14, .02, .14, .3))])
        d.stroke(kb(spl([(.10, -.06), (.04, -.20), (-.08, -.18), (-.08, -.06), (.0, -.04), (.02, -.12)], n=6)), .6, c['wood2'] if False else mix(LINE, c['wood2'], .3)); hand(d, n.at(hh[0], hh[1] + .10, math.atan2(ax[0], -ax[1]) + pi, .95), c, 'grip', seed=seed + 72, fl=1)
    # ------------- legs
    lk = dict(goku=('gi', 'gi2', 'boot', 1.0), krillin=('gi', 'gi2', 'boot', .98), gohan=('gi', 'gi2', 'boot', .92), bulma=('skin', 'skin2', 'boot', .55), vegeta=('suit', 'suit2', 'vboot', 1.), piccolo=('gi', 'gi2', 'boot', 1.06), roshi=('skin', 'skin2', 'sandal', 1.))[who]
    for far, hip_, ank, fa in ((True, hipF, aF, P['af']), (False, hipN, aN, P['an'])):
        kw = dict(n=n, seed=seed + (30 if far else 40), far=far, **pk)
        if who == 'bulma':
            kn, _ = ik(hip_, ank, l, l, -1.); T = tube([hip_, kn, ank], (.16, .13, .10)); d.part(n(T), c['skin'], c['skin2'], w=.95, seed=kw['seed'], shade=[n(T)] if far else []); f_ = n.at(ank[0], ank[1], fa); shoe(d, f_, c, 'boot', kw['seed'] + 1)
            u_ = (A(ank) - kn); u_ /= np.hypot(*u_); Bt = tube([A(ank) - u_ * .20, A(ank) + u_ * .02], (.15, .14)); d.part(n(Bt), c['shoe'], c['shoe2'], w=.95, seed=kw['seed'] + 2, shade=[n(Bt)] if far else [])
        elif who == 'roshi':
            kn = leg(d, hip_, ank, c, c['skin'], c['skin2'], fa, 'sandal', **kw); T = tube([hip_, hip_ * .5 + kn * .5, kn], (.25, .25, .22)); d.part(n(T), c['shorts'], c['shorts2'], w=1., seed=kw['seed'] + 3, shade=[n(T)] if far else [])
        else: leg(d, hip_, ank, c, c[lk[0]], c[lk[1]], fa, lk[2], baggy=lk[3], **kw)
    BODY[who](d, t, c, k, sw, seed)
    # ------------- near arm / crossed arms / cape front
    if who == 'vegeta' and P.get('crossed', True):
        y0 = .36; cc = dict(c, arm=c['suit'], arm2=c['suit2'], hand=c['white'], hand2=c['white2'])
        FA = tube([(.40, .16), (.50, .40), (.26, .46), (-.10, .40)], (.16, .16, .145, .13)); d.part(t(FA), c['suit'], c['suit2'], w=1., seed=seed + 81, shade=[t(FA)])
        NA = tube([(-.36, .16), (-.50, .42), (-.24, .52), (.22, .44)], (.17, .17, .155, .14)); d.part(t(NA), c['suit'], c['suit2'], w=1., seed=seed + 80, shade=[t(NA + [0, .13])])
        hn = t.at(.20, .44, -pi / 2 - .25); hand(d, hn, c, 'fist', c['white'], c['white2'], seed=seed + 82, fl=-1); B_ = tube([(.10, .465), (.22, .44)], (.15, .16)); d.part(t(B_), c['white'], c['white2'], w=.9, seed=seed + 83)
    elif who == 'piccolo' and not P.get('showarms'):
        s1, s2 = sw * .10 + tw * .05, sw * .16 + tw * .10
        fr = spl([(-.40, .02), (-.64, .24), (-.76 + s1, .80), (-.84 + s2, 1.46), (-.52 + s2, 1.54), (-.22 + s2 * .7, 1.40), (-.10 + s1, .80), (-.02, .28), (.10, .10)], True, n=7, corners=(3, 5)); d.part(t(fr), c['white'], c['white2'], shade=[t(spl([(-.9, .5), (-.6, .7), (-.52 + s1, 1.1), (-.60 + s2, 1.6), (-1.0, 1.6)], True, n=5))], w=1., seed=seed + 90)
        for a_, b_, c_ in (((-.46, .34), (-.52 + s1, .80), (-.60 + s2, 1.36)), ((-.24, .36), (-.28 + s1, .80), (-.34 + s2, 1.30))): d.stroke(t(spl([a_, b_, c_], n=7)), .55, '#b3a7ab', taper=(.3, .3))
        f2 = spl([(.36, .02), (.54, .22), (.58, .80), (.62 + s1, 1.40), (.46 + s1, 1.46), (.38, .80), (.22, .22)], True, n=6, corners=(3, 4)); d.part(t(f2), c['white'], c['white2'], shade=[t(f2 + [.12, 0])], w=1., seed=seed + 91)
        sp = spl([(-.62, .22), (-.60, -.02), (-.36, -.14), (0, -.06), (.34, -.14), (.56, -.04), (.60, .18), (.34, .20), (.16, .30), (-.02, .32), (-.28, .28)], True, n=7); d.part(t(sp), c['white'], c['white2'], shade=[t(spl([(-.8, .12), (-.2, .20), (.2, .18), (.8, .06), (.8, .5), (-.8, .5)], True, n=5))], w=1., seed=seed + 92)
        for L in ([(-.44, .04), (-.2, .12), (.04, .12)], [(.44, .02), (.28, .12), (.10, .20)]): d.stroke(t(spl(L, n=6)), .5, '#b3a7ab', taper=(.1, .5))
    elif who == 'roshi': arms(shN, hp(shN, -swing * .9 + .15), False, 1)
    elif not P.get('armTop') and not P.get('noNear'): arms(shN, aNh, False, 1)
    if who == 'gohan':
        # scarf tail in the wind
        a0 = local((-.34, .14)); q = p_ = P['p']; fl_ = [a0, a0 + [-.22, -.02 + .05 * math.sin(6.28 * 2 * q)], a0 + [-.46, .06 * math.sin(6.28 * 2 * q - 1.2) - .02 + sw * .10], a0 + [-.70, .04 + .10 * math.sin(6.28 * 2 * q - 2.2) + sw * .16]]; d.part(n(tube(fl_, (.07, .12, .12, .10))), c['white'], c['white2'], w=.9, seed=seed + 95)
    HEADS[who](d, H_, c, ex, seed + 100)
    if P.get('armTop'): arms(shN, aNh, False, -1)
    H_.ext = ex
    return dict(n=n, t=t, h=H_, local=local)


def shadow(d, x, gy, S, w=.62, col='#b98f5a', a=.35, hy=0.):
    d.fl.fill(ell(x, gy + 2, w * S * (1 - hy), .10 * S), col, a); d.sc.fill(ell(x, gy + 2, w * S * (1 - hy), .10 * S), col, a)


def march(who, f, S, V, off=0., **kw):
    """walk pose at frame f (20 f per cycle) whose feet stick to ground that moves V px per frame under a figure of size S"""
    return walk(who, f / 20. + off, A=V * 20 * DUTY / (2 * S * C[who]['k']), **kw)
