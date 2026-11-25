"""Sticker chibis (b2): bold black line, strong two-tone, hard highlight shapes, stars, die-cut white border.  Each has a 10-frame acting.
    L, M = sticker('goku', f, cx, cy, S)  -> premultiplied RGBA frame layer with die-cut border, art alpha;   scene('goku', f) -> finished RGB frame (colour card + sticker)
    f = 0..9 acting; f may go past 9 (hold), blink=1 closes the eyes (the grid at b2 frame 70+)."""
import numpy as np, cv2, math
from db import W, H, comp
from db.paint import rgb, f32
from .chibi_core import *
from . import chibi_story as ST
from .chibi_story import HAIR, HLINE, ball4

INK = '#17101c'
BG = dict(goku='#e23b2b', vegeta='#14a3a0', gohan='#7cc21f', bulma='#7d1b6b', piccolo='#f2b23a', frieza='#c61f74', cell='#e0472f', buu='#1aa19a')
CARD = dict(goku='#ff7a4a', vegeta='#4fd0bd', gohan='#b5e34f', bulma='#b23e93', piccolo='#ffd67a', frieza='#f0589c', cell='#ff8a52', buu='#55cfc0')
NAMES = ['goku', 'vegeta', 'gohan', 'bulma', 'piccolo', 'frieza', 'cell', 'buu']
SK = dict(skin='#ffd9b4', skin2='#ef9f7e')
C = dict(
    goku=dict(SK, hair='#2c2146', hair2='#170f29', h1='#5a3f97', h2='#c8407e', star='#ffcf3d', gi='#ff8a1f', gi2='#d45a12', blue='#2b56a8', blue2='#1a3474'),
    vegeta=dict(SK, hair='#2a2148', hair2='#150e2a', h1='#4668b8', h2='#7fe0e6', star='#ffcf3d', suit='#23399a', suit2='#141f5c', white='#fdf7ea', white2='#c3bcd2', gold='#f7b52b', gold2='#c27612'),
    gohan=dict(SK, hair='#2c2146', hair2='#170f29', h1='#5a3f97', h2='#c8407e', star='#ffcf3d', gi='#7a3fa8', gi2='#4c2377', red='#ee3f38', red2='#a81f33', white='#fffaf0', white2='#cdc3d8'),
    bulma=dict(SK, hair='#1fc0b4', hair2='#0f7f92', h1='#86efda', h2='#e9fff6', star='#ffcf3d', red='#ff4b5c', red2='#b81f48', dress='#ff77a4', dress2='#c43f7a', white='#fffaf0', white2='#d8c6d8'),
    piccolo=dict(skin='#7fd05a', skin2='#3f9a4f', gi='#6a35a0', gi2='#3f1c6e', white='#fdf8ee', white2='#c2bcd4', apple='#e2322e', apple2='#9f1730', cape='#fdf8ee', star='#ffcf3d'),
    frieza=dict(skin='#fbf5fb', skin2='#c9b6d8', dome='#8f2fb0', dome2='#571a7e', dh='#d57cf0', eye='#e8203a', lip='#33203f', star='#ff9ad0'),
    cell=dict(skin='#f1e7f2', skin2='#bba8c9', green='#58b84a', green2='#2b7d45', spot='#1f5f3c', dark='#2d2440', dark2='#181124', orange='#ff9a2e', orange2='#cc5a1a', mark='#8a4fa8', eye='#d93a8a', star='#ffcf3d'),
    buu=dict(skin='#ff9fb6', skin2='#e0628c', hi='#ffd0da', cape='#5f2a98', cape2='#3a1669', gold='#ffc637', gold2='#c67d14', star='#fff0a8'))


# ---------------------------------------------------------------- parts
def slock(d, h, q, c, seed=0, w=.62, deep=.10, sh=.46, hi=True, loose=1., col=None):
    F = lock2(h, q, loose); n = 13; P = h.n(F); Aa, B = F[:n], F[n - 1:][::-1]; t = np.linspace(0, 1, n)[:, None]
    ma, mb = Aa.mean(0), B.mean(0); lit, dark = (Aa, B) if (ma - mb) @ A([-.6, -.8]) > 0 else (B, Aa)
    shp = []
    if sh > 0: u = sh * (1 - .25 * t) * np.clip((t - (.04 if deep else 0)) / .34, 0, 1) ** .8 if sh < .95 else np.ones_like(t) ; inner = dark * (1 - u) + lit * u; shp = [h.n(np.vstack([dark, inner[::-1]]))]
    per = np.hypot(*np.diff(P, axis=0).T).sum(); rt = np.hypot(*(P[0] - P[-1])); tot = per + rt
    d.part(P, col or c['hair'], c['hair2'], shade=shp, w=w, seed=seed, gaps=[(1 - rt / tot - deep, deep)] if deep else [(1 - rt / tot - .01, .01)])
    if hi:
        g = np.sin(np.clip((t - .10) / .66, 0, 1) * pi); c0 = lit * .84 + dark * .16; c1 = lit * (.84 - .30 * g) + dark * (.16 + .30 * g); d.detail(h.n(np.vstack([c0[1:10], c1[1:10][::-1]])), c['h1'], within=P, sh=c['h1'])
        g2 = np.sin(np.clip((t - .30) / .55, 0, 1) * pi); c0 = lit * .74 + dark * .26; c1 = lit * (.74 - .10 * g2) + dark * (.26 + .10 * g2); d.detail(h.n(np.vstack([c0[4:11], c1[4:11][::-1]])), c['h2'], within=P, sh=c['h2'])


def shair_back(d, h, who, c, seed=0, w=.62, hs=None):
    L = [q for q in (hs or HAIR[who]) if q['tag'] != 'F' and lock_z(h, q) < -.12]; L.sort(key=lambda q: lock_z(h, q))
    for i, q in enumerate(L): slock(d, h, q, c, seed + i, w, hi=False, sh=1.0)
    O = h.outline(.02); d.part(O, c['hair2'], c['hair2'], w=w, seed=seed + 20)


def shair_front(d, h, who, c, face, seed=0, w=.62, hs=None, line=None, drop=.15):
    Ls = hs or HAIR[who]; L = [q for q in Ls if q['tag'] != 'F' and lock_z(h, q) >= -.12]; L.sort(key=lambda q: lock_z(h, q)); FL = [q for q in Ls if q['tag'] == 'F']
    ln = spl(line or HLINE[who], n=8) if who != 'vegeta' else spl(line or HLINE[who], n=6, corners=(5,)); cp, vis = cap_poly(h, ln); sk = h.outline(.02); s = h.n.s
    polys = [lock_poly(h, q) for q in L + FL] + [cp]; clips = [None] * (len(polys) - 1) + [sk]
    if face is not None:
        for q, k in zip(polys, clips):
            off = [.03 * s, drop * s]
            if k is not None: d.mk.clip([face]); d.mk.clip([k + off]); d.mk.fill(q + off, '#ffffff'); d.mk.unclip(); d.mk.unclip()
            else: d.shade(q + off, within=face)
    Us = d.union([cp], [sk]); d.part(Us, c['hair'], c['hair2'], w=w, seed=seed)
    for i, q in enumerate(L): slock(d, h, q, c, seed + 3 + i, w)
    cu = Us[0]; d.part(cu * .0 + cu, c['hair'], c['hair2'], w=0) if not L else None
    # hide the roots: the cap again, pulled in, without line
    cap2 = d.union([cp], [h.outline(-.035)])
    if cap2: d.part(cap2[0], c['hair'], c['hair2'], w=0)
    if vis is not None and who == 'vegeta': d.stroke(vis, .6, taper=(.1, .1))
    for i, q in enumerate(FL):
        if h.rot(q['tip'])[0, 2] > -.1: slock(d, h, q, c, seed + 30 + i, w * .9, deep=.07)


def stars(d, h, c, L, seed=0):
    """printed stars: (x, y, r, rot) in head-local 2-D"""
    for x, y, r, ro in L: d.detail(h.n(star(x, y, r, ro, .48)), c['star'], sh=mix(c['star'], '#c06a20', .35))


def spike_dots(d, P, col, n=14, r=2.2):
    Q = resample(np.asarray(P, float), 3.); idx = np.linspace(0, len(Q) - 1, n).astype(int)
    for i in idx[::1]:
        e = Q[min(i + 1, len(Q) - 1)] - Q[max(i - 1, 0)]; e = e / (np.hypot(*e) + 1e-6); d.fl.line(A([Q[i] - e * r * 1.6, Q[i] + e * r * 1.6]), r, col, taper=(0, 0)); d.sc.line(A([Q[i] - e * r * 1.6, Q[i] + e * r * 1.6]), r, col, taper=(0, 0))


def sface(d, h, c, seed=0, w=.72, shade=.28, blush=True, by=.44, bx=.62, ink=None):
    F = h.outline(); sg = 1 if h.yaw >= 0 else -1; O = h.outline(raw=True); s = h.n.s
    sh = [h.n(spl([(1.2, -.5), (.84, -.2), (.90, .24), (.78, .52), (.42, .70), (-.1, .77), (-.6, .70), (-.95, .55), (-.9, 1.2), (1.2, 1.2)], True, n=7))]
    d.part(F, c['skin'], c['skin2'], shade=sh, w=w, seed=seed, ink=ink)
    if blush:
        for q in (-1, 1):
            d.detail(h.F(ell(q * bx, by, .17, .095)), mix(c['skin'], '#ff6a6a', .48), within=F)
            for k in (-1, 0, 1): d.stroke(h.F([(q * bx + k * .075 + .035, by - .05), (q * bx + k * .075 - .035, by + .05)]), .26, '#d9484f', within=F, taper=(.1, .1))
    return F


def sear(d, h, c, sg, y=.16, s=1., seed=0):
    R = h.rot([[sg * .95, y, -.02]])[0]; e = h.n.at(R[0], R[1], .1 * sg, s)
    P = e(spl([(-sg * .04, -.16), (sg * .12, -.24), (sg * .24, -.10), (sg * .20, .10), (sg * .08, .24), (-sg * .06, .16)], True, n=7)); d.part(P, c['skin'], c['skin2'], w=.62, seed=seed, shade=[e(spl([(sg * .0, -.02), (sg * .12, -.10), (sg * .16, .02), (sg * .06, .14), (sg * .3, .3), (-.2, .3)], True, n=5))])
    d.stroke(e(spl([(sg * .16, -.12), (sg * .05, -.10), (sg * .06, .0), (sg * .12, .06)], n=6)), .36)


def eye(d, h, x, y, sg, face, w=.20, hh=.26, ang=0., look=(0., 0.), iris=None, lid=0., pup=1., white='#ffffff', lash=0., closed=0., kind='open', lw=1.):
    """sticker eye.  ang > 0 angry.  closed 0..1 lowers the lid; kind open|happy|shut|spark"""
    if h.z(sg * abs(x), y) < .02: return
    if kind == 'happy': d.stroke(h.F(spl([(x - w, y + .06), (x - w * .5, y - .06), (x, y - .10), (x + w * .5, y - .06), (x + w, y + .06)], n=8)), 1.0 * lw, taper=(.12, .12), tmin=.4, within=face); return
    if kind == 'shut' or closed > .85:
        d.stroke(h.F(spl([(x - w, y + .02 - ang * .10 * -sg * -1), (x - w * .4, y + .09), (x + w * .4, y + .09), (x + w, y + .02)], n=8)), 1.0 * lw, taper=(.12, .12), tmin=.4, within=face)
        if lash:
            for k in (.55, .95): d.stroke(h.F([(x + sg * w * k, y + .07), (x + sg * w * (k + .30), y + .13)]), .5 * lw, within=face)
        return
    s_ = math.tan(ang) * w; top = lambda u: y - hh * .80 * (1 - closed * 1.1) + (-s_ * u * 1.0) + hh * .12 * u * u            # u = -1 inner .. 1 outer
    E_ = spl([(x - sg * w, top(-1)), (x - sg * w * .4, top(-.4) - .02), (x + sg * w * .45, top(.45) - .01), (x + sg * w, top(1) + .03), (x + sg * w * .96, y + hh * .35), (x + sg * w * .5, y + hh * .86), (x - sg * w * .35, y + hh * .90), (x - sg * w * .92, y + hh * .40)], True, n=7, corners=(0, 3))
    Q = h.F(E_); d.knock(Q); d.fl.fill(Q, white); d.sc.fill(Q, mix(white, '#7a6aa8', .35)); d.mk.fill(Q, '#000000')
    ix, iy = x + look[0] * w, y + .03 + look[1] * hh
    if iris:
        d.detail(h.F(ell(ix, iy, w * .58 * pup, hh * .70 * pup)), iris, within=Q, sh=iris); d.detail(h.F(ell(ix, iy - hh * .22, w * .58 * pup, hh * .42 * pup)), mix(iris, '#17101c', .45), within=Q); d.detail(h.F(ell(ix, iy, w * .26 * pup, hh * .34 * pup)), INK, within=Q, sh=INK)
    else: d.detail(h.F(ell(ix, iy, w * .46 * pup, hh * .66 * pup)), INK, within=Q, sh=INK)
    if kind == 'spark':
        d.detail(h.F(star(ix, iy - .02, w * .36, -pi / 2, .42, 4)), '#ffffff', within=Q, sh='#ffffff')
    else: d.detail(h.F(ell(ix - w * .16, iy - hh * .30, w * .17, hh * .19)), '#ffffff', within=Q, sh='#ffffff'); d.detail(h.F(ell(ix + w * .18, iy + hh * .24, w * .08, hh * .09)), '#ffffff', within=Q, sh='#ffffff')
    d.shade(h.F(spl([(x - sg * w * 1.1, top(-1) - .1), (x + sg * w * 1.1, top(1) - .1), (x + sg * w * 1.1, top(1) + .10), (x, top(0) + .10), (x - sg * w * 1.1, top(-1) + .10)], True, n=3)), within=Q)
    T = spl([(x - sg * w * 1.10, top(-1) + .015), (x - sg * w * .4, top(-.4) - .03), (x + sg * w * .45, top(.45) - .025), (x + sg * w * 1.04, top(1) + .01)] + ([(x + sg * w * (1.22 + lash * .3), top(1) - .05 * lash)] if lash else []), n=8)
    d.stroke(h.F(T), 1.25 * lw, taper=(.10, .16), tmin=.4, within=face); d.stroke(h.F(spl([(x + sg * w * .98, y + hh * .38), (x + sg * w * .5, y + hh * .90), (x - sg * w * .35, y + hh * .93)], n=7)), .48 * lw, taper=(.05, .5), within=face)
    d.stroke(h.F(spl([(x - sg * w * .98, top(-1)), (x - sg * w * .96, y + hh * .35)], n=3) if False else A([(x - sg * w * .99, top(-1)), (x - sg * w * .95, y + hh * .30)])), .42 * lw, taper=(0, .6), within=face); d.stroke(h.F(spl([(x + sg * w * 1.0, top(1)), (x + sg * w * 1.02, y + hh * .10), (x + sg * w * .94, y + hh * .42)], n=5)), .46 * lw, taper=(0, .2), within=face)


def brow(d, h, x, y, sg, face, w=.24, ang=0., t=.085, col=INK, arch=.03):
    if h.z(sg * abs(x), y) < .02: return
    s = math.sin(ang); u = np.linspace(-1, 1, 9); cx = x + sg * u * w; cy = y - s * u * w - arch * (1 - u * u); th = t * (1.0 - .42 * u) * np.where(u < -.9, .75, 1.)
    P = np.vstack([np.c_[cx, cy - th * .5], np.c_[cx, cy + th * .5][::-1]]); d.part(h.F(P, .02), col, mix(col, '#000000', .3), w=.28)


def smouth(d, h, face, M, tongue=None, teeth=None, w=.62, col='#8a1226', col2='#5c0a1c', ink=None, dz=0.):
    """M = outline (front coords); teeth = list of polygons (front coords) white"""
    Q = h.F(M, dz); d.part(Q, col, col2, w=0)
    if tongue is not None: d.detail(h.F(tongue, dz), '#ff7f86', within=Q, sh='#e85a6c')
    for t in (teeth or []): d.detail(h.F(t, dz), '#fffaf4', within=Q, sh='#e2d4e0')
    d.contour(Q, w, ink); return Q


def nose(d, h, face, x=0., y=.36, s=1.):
    d.stroke(h.F(spl([(x - .03 * s, y - .05 * s), (x + .05 * s, y + .01 * s), (x - .0 * s, y + .05 * s)], n=6)), .42, taper=(.3, .2), within=face)


def limb_union(d, n, P, w, flat=True):
    """bent limb without pinches: discs swept along the curve -> outline px"""
    P = np.asarray(P, float); C = spl(P, n=14); dd = np.r_[0, np.cumsum(np.hypot(*np.diff(C, axis=0).T))]; K_ = int(max(40, dd[-1] / (min(w) * .16 + .006))) if min(w) > .08 else 90; K_ = min(K_, 120); t = np.linspace(0, dd[-1], K_); C = np.c_[np.interp(t, dd, C[:, 0]), np.interp(t, dd, C[:, 1])]
    W_ = np.interp(np.linspace(0, 1, K_), np.linspace(0, 1, len(w)), w); polys = [n(ell(x, y, r, r, n=30)) for (x, y), r in zip(C, W_)]
    U = d.union(polys, sm=15); return U[0] if U else None


def tub(d, n, P, w, col, sh, lw=.66, seed=0, cap=.9, cap0=0., shs=(-.16, .1), gaps=()):
    P = np.asarray(P, float)
    if len(P) > 2:
        e = np.diff(P, axis=0); e /= np.hypot(*e.T)[:, None] + 1e-9; bend = np.max(np.abs(np.diff(np.unwrap(np.arctan2(e[:, 1], e[:, 0])))))
        if bend > .62:
            U = limb_union(d, n, P, w)
            if U is not None:
                o = n.M[:2, :2] @ A(shs if shs is not None else (0, 0)); d.part(U, col, sh, w=lw, seed=seed, shade=[U + o * 1.15] if shs is not None else []); return U
    T = tube(P, w, cap=cap, cap0=cap0); d.part(n(T), col, sh, w=lw, seed=seed, shade=[n(T + A(shs))] if shs is not None else [], gaps=gaps); return T


# ---------------------------------------------------------------- hands (local unit: palm half width .36, +y = wrist)
def hand_v(d, n, c, seed=0, spread=1.0, curl=0.):
    col, sh = c['skin'], c['skin2']
    d.part(n(spl([(-.22, .62), (-.34, .30), (-.36, .02), (-.28, -.18), (.26, -.18), (.36, .02), (.34, .30), (.24, .62)], True, n=7)), col, sh, w=.70, seed=seed, shade=[n(spl([(.1, .7), (.12, .36), (.3, .1), (.6, .1), (.6, .8)], True, n=5))])
    L = 1 - .55 * curl
    tub(d, n, [(.13, -.06), (.22 * spread, -.50 * L), (.34 * spread, -.98 * L)], (.155, .15, .135), col, sh, .70, seed + 2, shs=(.14, .04)); tub(d, n, [(-.15, -.06), (-.20 * spread, -.52 * L), (-.26 * spread, -1.02 * L)], (.16, .155, .14), col, sh, .70, seed + 1, shs=(.16, .04))
    d.part(n(spl([(-.24, -.02), (-.2, -.16), (.2, -.16), (.28, .0), (.2, .2), (-.2, .2)], True, n=6)), col, sh, w=0)
    tub(d, n, [(.27, .02), (.28, .16), (.18, .24)], (.13, .125, .10), col, sh, .62, seed + 4, cap=.8, cap0=.8, shs=(.0, .12)); tub(d, n, [(.08, -.03), (.09, .14), (.02, .22)], (.13, .13, .11), col, sh, .62, seed + 5, cap=.8, cap0=.8, shs=(.0, .12))
    tub(d, n, [(-.30, .40), (-.16, .20), (.10, .12)], (.16, .14, .115), col, sh, .68, seed + 6, shs=(.06, .10)); d.stroke(n(spl([(.04, .06), (.10, .06), (.13, .11)], n=5)), .36)


def hand_fist(d, n, c, seed=0, col=None, sh=None, fl=1):
    col, sh = col or c['skin'], sh or c['skin2']; f = lambda P: n(A(P, float) * [fl, 1])
    P = f(spl([(-.34, -.26), (-.12, -.34), (.14, -.34), (.34, -.25), (.40, .02), (.34, .30), (.0, .36), (-.34, .30), (-.40, .04)], True, n=7)); d.part(P, col, sh, w=.72, seed=seed, shade=[f(spl([(-.6, .10), (-.2, .16), (.3, .12), (.6, .0), (.6, .6), (-.6, .6)], True, n=5))])
    for x in (-.17, .01, .19): d.stroke(f(spl([(x, -.30), (x - .01, -.12), (x + .0, .04)], n=5)), .46, taper=(.05, .4))
    for x in (-.26, -.08, .10, .28): d.stroke(f(spl([(x - .05, -.10), (x, -.07), (x + .05, -.10)], n=4)), .32, taper=(.3, .3))
    T = tube([(-.36, .16), (-.12, .22), (.12, .16)], (.145, .13, .11), cap=.9, cap0=.5); d.part(f(T), col, sh, w=.62, seed=seed + 2, shade=[f(T + [.04, .10])]); d.stroke(f(spl([(.05, .10), (.11, .10), (.14, .15)], n=5)), .34)


def hand_cup(d, n, c, seed=0, fl=1, curl=0.):
    """palm under a cheek, fingers up along the face: wrist at +y"""
    col, sh = c['skin'], c['skin2']; f = n.at(0, 0, 0, fl, 1)
    for i, (a, l, x) in enumerate(((.26, .50, .24), (.14, .62, .11), (.03, .66, -.03), (-.09, .58, -.17))):
        tub(d, f, [(x, .06), (x + math.sin(a) * l * .55, -.02 - math.cos(a) * l * .5), (x + math.sin(a + curl) * l * 1.05, -.02 - math.cos(a + curl) * l)], (.115, .11, .085), col, sh, .58, seed + i, shs=(.08, .06))
    d.part(f(spl([(-.22, .66), (-.30, .30), (-.28, .04), (-.1, -.06), (.22, -.04), (.34, .10), (.30, .36), (.20, .66)], True, n=7)), col, sh, w=.66, seed=seed + 5, shade=[f(spl([(.0, .7), (.10, .3), (.4, .1), (.6, .8)], True, n=5))], gaps=[(.40, .62)])
    tub(d, f, [(-.20, .30), (-.40, .10), (-.46, -.14)], (.13, .115, .09), col, sh, .58, seed + 6, shs=(.06, .06))


def hand_point(d, n, c, seed=0, fl=1, col=None, sh=None, L=1.):
    col, sh = col or c['skin'], sh or c['skin2']; f = n.at(0, 0, 0, fl, 1)
    tub(d, f, [(-.10, -.10), (-.12, -.48 * L), (-.10, -.86 * L)], (.15, .14, .11), col, sh, .64, seed + 1, shs=(.14, .04))
    d.part(f(spl([(-.24, .60), (-.32, .24), (-.30, -.10), (-.12, -.22), (.18, -.22), (.34, -.08), (.36, .22), (.26, .60)], True, n=7)), col, sh, w=.70, seed=seed, shade=[f(spl([(.1, .7), (.14, .3), (.5, .1), (.6, .8)], True, n=5))], gaps=[(.36, .44)])
    for i, x in enumerate((.04, .18, .30)): tub(d, f, [(x, -.12), (x + .01, .06), (x - .05, .14)], (.10, .10, .08), col, sh, .52, seed + 2 + i, cap=.8, cap0=.8, shs=(.0, .10))
    tub(d, f, [(-.28, .34), (-.10, .18), (.08, .16)], (.13, .115, .09), col, sh, .58, seed + 6, shs=(.04, .08))


# ---------------------------------------------------------------- busts
def bust(d, n, P, col, sh, seed=0, shade=None, w=.80):
    Q = n(spl(P, True, n=7)); d.part(Q, col, sh, w=w, seed=seed, shade=shade or [n(spl([(.25, .4), (.4, 1.0), (.3, 2.2), (1.6, 2.2), (1.6, .4)], True, n=5))]); return Q


def b_goku(d, n, k):
    c = C['goku']; h = Hd(n, k.get('yaw', .22), k.get('pitch', 0.), k.get('lag', (0, 0)), k.get('t', 0), 0.); hb = n.at(*(A([0., 0.])), -k.get('roll', 0.) * .7)
    shair_back(d, h, 'goku', c, 10)
    B = bust(d, hb, [(-.34, .66), (-.86, .86), (-1.08, 1.22), (-1.02, 1.72), (-.4, 1.86), (.4, 1.86), (1.00, 1.72), (1.06, 1.22), (.86, .86), (.34, .66)], c['gi'], c['gi2'], 40)
    d.detail(hb(spl([(-.30, .6), (.34, .6), (.12, 1.30), (.06, 1.36)], True, n=4, corners=(0, 1, 2, 3))), c['blue'], within=B, sh=c['blue2']); d.part(hb(spl([(-.26, .62), (.30, .62), (.22, .92), (.02, 1.04), (-.18, .92)], True, n=6)), c['skin'], c['skin'], w=.5, shade=[]); d.shade(hb(spl([(-.4, .5), (.5, .5), (.5, .96), (-.1, .98), (-.4, .9)], True, n=5)), within=hb(spl([(-.26, .62), (.30, .62), (.22, .92), (.02, 1.04), (-.18, .92)], True, n=6)))
    for L in ([(-.38, .70), (-.14, 1.10), (.22, 1.62), (.26, 1.84)], [(.42, .70), (.26, 1.02), (.07, 1.36)]): d.stroke(hb(spl(L, n=7)), .72, within=B, taper=(.03, .12))
    for sg in (-1, 1): d.part(hb(spl([(sg * .78, .84), (sg * 1.08, 1.10), (sg * 1.16, 1.52), (sg * .96, 1.62), (sg * .74, 1.30)], True, n=6)), c['blue'], c['blue2'], w=.72, seed=44 + sg, shade=[hb(spl([(sg * .7, 1.2), (sg * 1.0, 1.3), (sg * 1.3, 1.2), (sg * 1.3, 1.8), (sg * .7, 1.8)], True, n=4))])
    for L in ([(-.72, 1.36), (-.50, 1.46), (-.36, 1.66)], [(.70, 1.40), (.56, 1.5), (.50, 1.7)]): d.stroke(hb(spl(L, n=5)), .42, within=B, taper=(.05, .6))
    kn = hb(ell(.56, 1.16, .20, .20)); d.part(kn, '#fffaf2', '#d8cbdc', w=.5); m = hb.at(.56, 1.16, 0., .20)
    for L in ([(-.50, -.60), (-.50, .66)], [(-.76, -.15), (-.68, .10)], [(-.32, -.24), (-.24, -.02)], [(-.12, -.54), (.66, -.54)], [(.22, -.54), (.10, -.10)], [(-.05, -.30), (.52, -.30), (.52, -.08)], [(-.20, -.07), (.74, -.07)], [(-.02, .18), (-.02, .64)], [(-.02, .18), (.58, .18), (.58, .64)], [(-.02, .64), (.58, .64)]): d.stroke(m(A(L)), .30, taper=(.05, .15), tmin=.5, within=kn)
    sear(d, h, c, 1, seed=5); F = sface(d, h, c, 1); sear(d, h, c, -1, seed=6)
    bl = k.get('blink', 0); ek = 'shut' if bl else k.get('eyes', 'open')
    for sg in (-1, 1): eye(d, h, sg * .36, .14, sg, F, .20, .28, .02, (-.10 * sg + .06, .04), kind=k.get('eyeR', ek) if sg > 0 and not bl else ek); brow(d, h, sg * .38, -.24 + k.get('brow', 0.), sg, F, .24, k.get('bang', -.06), .10, c['hair'])
    nose(d, h, F, .02, .38); o = k.get('o', 1.); x = .04
    M = spl([(x - .30, .50), (x - .14, .525), (x, .53), (x + .16, .525), (x + .32, .48), (x + .27, .50 + .20 * o), (x + .05, .52 + .27 * o), (x - .20, .50 + .19 * o)], True, n=8, corners=(0, 4))
    smouth(d, h, F, M, ell(x + .04, .50 + .27 * o, .20, .12 * o + .02), [spl([(x - .29, .49), (x + .31, .47), (x + .26, .56 + .02 * o), (x, .575 + .02 * o), (x - .24, .56 + .02 * o)], True, n=4)])
    shair_front(d, h, 'goku', c, F, 60); stars(d, h, c, [(-.42, -1.16, .12, -.2), (.28, -.78, .085, .3), (-1.18, -.86, .085, .1)]); (abs(k.get('lag', (0, 0))[1]) < .1) and spike_dots(d, h.n(spl([(-.10, -.98), (-.46, -1.30), (-.78, -1.62)], n=10)), '#8f76d6', 11, 2.0)
    hn = k.get('hand', (1.30, .86, .16, 1.12))
    if hn is not None:
        x_, y_, r_, s_ = hn; a = hb.inv(n([(x_, y_)]))[0] if False else A([x_, y_]); fa = [(.92, 1.46), (1.02 + (x_ - 1.3) * .5, 1.58 + (y_ - .86) * .3), (a[0] + math.sin(-r_) * .5 * s_ * -1, a[1] + .62 * s_)]
        tub(d, n, [fa[0], fa[1], A(a) + [-.0, .66 * s_]], (.22, .22, .20), c['skin'], c['skin2'], .72, 80, shs=(.12, .1)); tub(d, n, [A(a) + [0, .70 * s_], A(a) + [0, .40 * s_]], (.27 * s_ * .9, .28 * s_ * .9), c['blue'], c['blue2'], .72, 82, cap=.2, cap0=.2, shs=(.16, .0)); st = k.get('stretch', (1., 1.)); hand_v(d, n.at(a[0], a[1] - .3 * (st[1] - 1), r_, s_ * .74 * st[0], s_ * .74 * st[1]), c, 84, k.get('spread', 1.), k.get('curl', 0.))
    return h


def b_vegeta(d, n, k):
    c = C['vegeta']; h = Hd(n, k.get('yaw', -.2), k.get('pitch', 0.), k.get('lag', (0, 0)), k.get('t', 0), 0.); hb = n.at(0, 0, -k.get('roll', 0.) * .8)
    shair_back(d, h, 'vegeta', c, 10)
    B = bust(d, hb, [(-.30, .62), (-.80, .80), (-1.06, 1.14), (-1.04, 1.74), (-.4, 1.86), (.4, 1.86), (1.04, 1.74), (1.06, 1.14), (.80, .80), (.30, .62)], c['suit'], c['suit2'], 40)
    ar = hb(spl([(-.46, .66), (-.78, .84), (-.92, 1.20), (-.86, 1.9), (.86, 1.9), (.92, 1.20), (.78, .84), (.46, .66), (.30, .96), (0, 1.04), (-.30, .96)], True, n=6)); d.part(ar, c['white'], c['white2'], w=.72, seed=41, shade=[hb(spl([(.30, .5), (.50, 1.0), (.4, 2.0), (1.6, 2.0), (1.6, .5)], True, n=5))]); d.stroke(hb(spl([(0, 1.05), (.02, 1.3), (0, 1.5)], n=5)), .5)
    for sg in (-1, 1): d.part(hb(spl([(sg * .46, .66), (sg * .62, .74), (sg * .70, 1.04), (sg * .56, 1.02)], True, n=5)), c['gold'], c['gold2'], w=.5)
    far, near = (1, -1) if h.yaw >= 0 else (-1, 1); sear(d, h, c, far, seed=5); F = sface(d, h, c, 1, by=.46); sear(d, h, c, near, seed=6); puff = k.get('puff', 0.)
    bl = k.get('blink', 0); ek = 'shut' if bl else k.get('eyes', 'open')
    for sg in (-1, 1):
        eye(d, h, sg * .36, .16, sg, F, .20, .20, .34, k.get('look', (0., .04)), iris=None, kind=(k.get('eyeN', ek) if sg == near and not bl else ek)); brow(d, h, sg * .36, -.12 + k.get('brow', 0.), sg, F, .28, .42, .125, c['hair2'], .0)
    nose(d, h, F, .0, .40); m = k.get('mouth', 'hmph')
    if m == 'hmph': d.stroke(h.F(spl([(-.16, .60), (-.05, .55), (.06, .55), (.17, .62)], n=8)), .74, taper=(.1, .1), tmin=.4, within=F); d.stroke(h.F([(-.04, .67), (.05, .67)]), .34, within=F)
    elif m == 'puff':
        d.stroke(h.F(spl([(-.07, .575), (0, .56), (.07, .575)], n=6)), .74, taper=(.1, .1), tmin=.4, within=F)
        for sg in (-1, 1): d.stroke(h.F(spl([(sg * .42, .46), (sg * .50, .60), (sg * .44, .72)], n=6)), .40, taper=(.3, .3), within=F)
    shair_front(d, h, 'vegeta', c, F, 60, drop=.12); stars(d, h, c, [(-.40, -1.18, .11, .2), (.52, -1.04, .09, -.3)]); (abs(k.get('lag', (0, 0))[1]) < .1 and abs(k.get('lag', (0, 0))[0]) < .1) and spike_dots(d, h.n(spl([(.16, -.96), (.30, -1.4), (.36, -1.78)], n=10)), '#7f8fd6', 10, 2.0)
    if k.get('vein', 0): pass
    # crossed arms, white gloves
    y = 1.34 + k.get('arms', 0.); tub(d, hb, [(.92, 1.06), (1.08, 1.50 + (y - 1.34)), (.30, y + .16), (-.36, y + .06)], (.30, .28, .25, .22), c['suit'], c['suit2'], .76, 50, shs=(.02, .16)); tub(d, hb, [(-.92, 1.06), (-1.10, 1.54 + (y - 1.34)), (-.20, y + .26), (.50, y + .06)], (.30, .29, .26, .23), c['suit'], c['suit2'], .78, 52, shs=(.02, .18))
    tub(d, hb, [(.14, y + .14), (.50, y + .06)], (.25, .26), c['white'], c['white2'], .7, 54, cap=.2, shs=(.0, .16)); hand_fist(d, hb.at(.62, y + .02, -1.45 + .1, .62), c, 56, c['white'], c['white2']); tub(d, hb, [(-.52, y + .02), (-.74, y - .02), (-.82, y - .20)], (.16, .15, .12), c['white'], c['white2'], .62, 58, cap0=.3)
    return h


def hat(d, n, h, c, F, seed=0, k=None):
    ya = -.36
    d.shade(h.n(spl([(-1.2, ya + .1), (0, ya + .40), (1.2, ya + .1), (1.2, -1), (-1.2, -1)], True, n=6)), within=F)
    cr = [(-1.10, ya + .06), (-1.10, -.54), (-.92, -.98), (-.50, -1.28), (0, -1.37), (.50, -1.28), (.92, -.98), (1.10, -.54), (1.10, ya + .06), (.5, ya + .14), (0, ya + .16), (-.5, ya + .14)]
    P = h.n(spl(cr, True, n=7)); d.part(P, c['red'], c['red2'], w=.82, seed=seed, shade=[h.n(spl([(1.3, -1.4), (.66, -1.08), (.40, -.62), (.44, -.1), (1.3, -.1)], True, n=6))], hi=[(h.n(spl([(-.84, -.62), (-.66, -.98), (-.36, -1.16), (-.46, -.92), (-.62, -.58)], True, n=6, corners=(0, 2))), '#ff8f7a')])
    for a_, m_, b_ in [((-.10, -1.34), (-.42, -.9), (-.50, -.30)), ((.14, -1.34), (.36, -.86), (.38, -.30))]: d.stroke(h.n(spl(A([a_, m_, b_]) + [.2 * h.sy, 0], n=8)), .42, taper=(.08, .4))
    stars(d, h, c, [(-.02, -.80, .15, .1), (.66, -.72, .10, -.3)])
    d.part(h.n(spl([(-1.18, ya - .12), (-.6, ya + .0), (0, ya + .03), (.6, ya + .0), (1.18, ya - .12), (1.24, ya + .09), (1.12, ya + .21), (.56, ya + .31), (0, ya + .34), (-.56, ya + .31), (-1.12, ya + .21), (-1.24, ya + .09)], True, n=7)), mix(c['red'], '#ff9a6a', .25), c['red2'], w=.80, seed=seed + 1, shade=[h.n(spl([(-1.3, ya + .16), (0, ya + .25), (.7, ya + .12), (1.3, ya - .1), (1.3, ya + .5), (-1.3, ya + .5)], True, n=6))])
    bn = h.n.at(.10 * h.sy, -1.56 + (k or {}).get('ball', 0.)); d.part(h.n(ell(.10 * h.sy, -1.34, .28, .11)), c['red2'], w=.6); ball4(d, bn, .36, 4, face=(.22 * h.sy, .04), w=.66, seed=seed + 3)


def b_gohan(d, n, k):
    c = C['gohan']; h = Hd(n, k.get('yaw', .12), k.get('pitch', 0.), k.get('lag', (0, 0)), k.get('t', 0), 0.); hb = n.at(0, 0, -k.get('roll', 0.) * .7)
    shair_back(d, h, 'gohan', c, 10)
    B = bust(d, hb, [(-.30, .66), (-.72, .86), (-.90, 1.22), (-.86, 1.68), (-.4, 1.80), (.4, 1.80), (.86, 1.68), (.90, 1.22), (.72, .86), (.30, .66)], c['gi'], c['gi2'], 40); d.stroke(hb(spl([(-.2, .8), (.02, 1.2), (.22, 1.76)], n=6)), .6, within=B)
    sc = hb(spl([(-.80, .74), (-.4, .86), (.2, .88), (.80, .74), (.90, 1.0), (.5, 1.22), (0, 1.28), (-.5, 1.22), (-.90, 1.0)], True, n=6)); d.part(sc, c['white'], c['white2'], w=.72, seed=42, shade=[hb(spl([(-1, 1.04), (0, 1.14), (.5, .9), (1.1, .8), (1.1, 1.5), (-1, 1.5)], True, n=5))]); d.stroke(hb(spl([(-.6, .94), (-.1, 1.08), (.3, 1.06)], n=6)), .42, taper=(.1, .5)); d.stroke(hb(spl([(.62, .92), (.36, 1.10), (.0, 1.18)], n=6)), .42, taper=(.1, .5))
    F = sface(d, h, c, 1, by=.48); sear(d, h, c, -1, .22, .9, seed=6); sear(d, h, c, 1, .22, .9, seed=5)
    bl = k.get('blink', 0); ek = 'shut' if bl else k.get('eyes', 'open')
    for sg in (-1, 1): eye(d, h, sg * .38, .20, sg, F, .23, .32, -.02, (-.06 * sg, .04), kind=ek, iris='#5a3a24' if ek == 'spark' else None); brow(d, h, sg * .40, -.20 + k.get('brow', 0.), sg, F, .19, k.get('bang', .18), .085, c['hair'])
    nose(d, h, F, .01, .44, .8); o = k.get('o', 1.); x = .02
    M = spl([(x - .24, .555), (x, .575), (x + .24, .555), (x + .22, .60 + .13 * o), (x, .60 + .23 * o), (x - .22, .60 + .13 * o)], True, n=8, corners=(0, 2)); smouth(d, h, F, M, ell(x, .60 + .23 * o, .15, .11 * o + .02), [spl([(x - .2, .55), (x + .2, .55), (x + .17, .61), (x - .17, .61)], True, n=3)])
    shair_front(d, h, 'gohan', c, F, 60, drop=.12); hat(d, n, h, c, F, 90, k)
    for sg in (-1, 1):
        hx, hy = k.get('fists', (.86, 1.02)); sx = sg * hx; tub(d, hb, [(sg * .78, 1.34), (sg * 1.10, 1.50), (sx * 1.04, hy + .38)], (.22, .22, .20), c['gi'], c['gi2'], .70, 70 + sg, shs=(.1 * sg, .1)); tub(d, hb, [(sx * 1.04, hy + .40), (sx * 1.02, hy + .22)], (.22, .23), c['red'], c['red2'], .66, 74 + sg, cap=.2, cap0=.2, shs=(.12, 0)); hand_fist(d, hb.at(sx, hy, sg * -.12, .62), c, 76 + sg, fl=sg)
    return h


def b_bulma(d, n, k):
    c = C['bulma']; h = Hd(n, k.get('yaw', -.22), k.get('pitch', 0.), k.get('lag', (0, 0)), k.get('t', 0), 0.); hb = n.at(0, 0, -k.get('roll', 0.) * .7); sw = k.get('swing', (0., 0.))
    b0 = h.rot([[.62, -.70, -.30]])[0, :2]; t = A([b0, b0 + [.40, -.22], b0 + [.86 + sw[0] * .4, .16 + sw[1] * .4], b0 + [.92 + sw[0] * .9, .84 + sw[1] * .9], b0 + [.66 + sw[0] * 1.3, 1.50 + sw[1] * 1.2]])
    T = tube(t, (.16, .32, .42, .34, .02)); d.part(h.n(T), c['hair'], c['hair2'], w=.74, seed=30, shade=[h.n(tube(t + [.26, .06], (.14, .24, .34, .3, .02)))])
    PT = h.n(T)
    for o, cc, w_ in ((-.20, c['h1'], 1.0), (-.10, c['h2'], .5)): d.stroke(h.n(spl(t[1:4] + [o, .02], n=8)), w_, cc, taper=(.3, .5), within=PT)
    for o in (.14, .02): d.stroke(h.n(spl(np.vstack([t[1:4] + [o, .05], t[4:]]), n=8)), .4, taper=(.2, .4), within=PT)
    shair_back(d, h, 'bulma', c, 10)
    B = bust(d, hb, [(-.26, .62), (-.66, .84), (-.82, 1.20), (-.72, 1.74), (-.3, 1.84), (.3, 1.84), (.72, 1.74), (.82, 1.20), (.66, .84), (.26, .62)], c['dress'], c['dress2'], 40)
    for sg in (-1, 1): d.part(hb(spl([(0, .98), (sg * .26, .72), (sg * .50, .84), (sg * .42, 1.16), (sg * .18, 1.22)], True, n=5, corners=(0, 1, 2, 3, 4))), c['white'], c['white2'], w=.6, seed=43 + sg, shade=[hb(spl([(sg * .1, 1.1), (sg * .5, 1.0), (sg * .5, 1.4), (0, 1.4)], True, n=4))] if sg > 0 else [])
    d.part(hb(star(0, 1.06, .16, -pi / 2, .5)), '#ffcf3d', '#d98a18', w=.5)
    F = sface(d, h, c, 1, by=.46)
    bl = k.get('blink', 0); w_ = k.get('wink', 0.)
    for sg in (-1, 1):
        kd = 'shut' if bl else ('happy' if (sg > 0 and w_ > .6) else 'open'); eye(d, h, sg * .38, .16, sg, F, .21, .29, -.06, (-.06 * sg, .03), iris='#1f9fd0', lash=1., kind=kd, closed=w_ if (sg > 0 and kd == 'open') else 0.); brow(d, h, sg * .40, -.26 + (k.get('brow', 0.) if sg < 0 else 0.), sg, F, .18, -.10, .06, c['hair2'])
    if w_ > .6 and not bl:
        for q in (.62, .98): d.stroke(h.F([(.38 + .21 * q, .20), (.38 + .21 * (q + .36), .27)]), .5, within=F)
    nose(d, h, F, .0, .40, .8); o = k.get('o', 1.); x = k.get('mx', .03)
    M = spl([(x - .20, .545), (x - .05, .57), (x + .10, .565), (x + .24, .51), (x + .21, .57 + .14 * o), (x + .04, .58 + .21 * o), (x - .14, .57 + .13 * o)], True, n=8, corners=(0, 3)); smouth(d, h, F, M, ell(x + .04, .58 + .21 * o, .14, .09 * o + .02), [spl([(x - .19, .54), (x + .22, .51), (x + .18, .60), (x - .10, .615)], True, n=3)])
    shair_front(d, h, 'bulma', c, F, 60, drop=.10); stars(d, h, c, [(-.42, -.72, .11, .2), (.30, -.86, .085, -.2)])
    r = h.n.at(b0[0] - .02, b0[1] - .06, .35 + sw[0] * .2)
    for sg in (-1, 1): d.part(r(spl([(0, 0), (sg * .22, -.28), (sg * .56, -.34), (sg * .62, .02), (sg * .48, .22), (sg * .20, .12)], True, n=6, corners=(0,))), c['red'], c['red2'], w=.70, seed=20 + sg, shade=[r(spl([(0, .02), (sg * .3, .0), (sg * .62, .06), (sg * .5, .4), (0, .3)], True, n=5))], hi=[(r(spl([(sg * .20, -.18), (sg * .38, -.24), (sg * .36, -.12), (sg * .22, -.08)], True, n=5)), '#ff9a9a')])
    d.part(r(ell(0, .0, .15, .15)), c['red'], c['red2'], w=.64, seed=22, shade=[r(ell(.08, .08, .13, .13))])
    # finger to the cheek
    hx, hy, hr = k.get('hand', (.86, .98, -.45)); tub(d, hb, [(.66, 1.34), (.98, 1.66), (hx + .16, hy + .46)], (.18, .18, .165), c['skin'], c['skin2'], .70, 70, shs=(.1, .1)); hand_point(d, hb.at(hx, hy, hr, .60), c, 74, fl=1)
    return h


def apple(d, n, c, bite=0., seed=0):
    R_ = [(0, -.40), (.22, -.52), (.50, -.42), (.60, -.06), (.46, .36), (.16, .52), (0, .47), (-.16, .52)]; Lf = [(-.46, .36), (-.60, -.06), (-.50, -.42), (-.22, -.52)]
    if bite > 0:
        P = np.vstack([spl(R_, n=7), spl([(-.16, .52), (-.30, .46), (-.36, .38), (-.28, .30), (-.22, .24), (-.12, .16), (-.12, .04), (-.20, -.04), (-.26, -.10), (-.24, -.22), (-.32, -.32), (-.42, -.40), (-.36, -.50), (-.22, -.52), (0, -.40)], n=5, corners=(2, 4, 7, 9, 11))[1:-1]])
        fl = spl([(-.36, .38), (-.28, .30), (-.22, .24), (-.12, .16), (-.12, .04), (-.20, -.04), (-.26, -.10), (-.24, -.22), (-.32, -.32), (-.42, -.40), (-.22, -.36), (-.08, -.18), (.02, .0), (.0, .20), (-.14, .40)], True, n=5)
    else: P = spl(R_ + Lf, True, n=7)
    d.part(n(P), c['apple'], c['apple2'], w=.74, seed=seed, shade=[n(spl([(.1, .7), (.26, .34), (.30, -.1), (.44, -.5), (.9, -.4), (.9, .8)], True, n=6))], hi=[] if bite else [(n(spl([(-.42, -.22), (-.30, -.38), (-.16, -.36), (-.28, -.16), (-.34, .08)], True, n=6)), '#ff9a8a')])
    if bite > 0: d.detail(n(fl), '#fff2c8', within=n(P), sh='#e8cf94'); d.stroke(n(spl([(-.40, -.40), (-.22, -.36), (-.08, -.18), (.02, .0), (.0, .20), (-.14, .40), (-.34, .40)], n=6)), .42, '#7a1a24', within=n(P)); d.detail(n(ell(.0, .0, .03, .05, .4)), '#5a3320', within=n(P))
    d.stroke(n(spl([(0, -.38), (.02, -.54), (.12, -.66)], n=6)), 1.1, '#5a3320', taper=(0, 0)); d.part(n(spl([(.08, -.56), (.26, -.74), (.46, -.70), (.30, -.54)], True, n=6, corners=(0, 2))), '#5fc24a', '#2f8a40', w=.5)


def b_piccolo(d, n, k):
    c = C['piccolo']; h = Hd(n, k.get('yaw', .30), k.get('pitch', 0.), k.get('lag', (0, 0)), k.get('t', 0), 0.); hb = n.at(0, 0, -k.get('roll', 0.) * .7)
    B = bust(d, hb, [(-.40, .60), (-1.12, .80), (-1.34, 1.20), (-1.26, 1.74), (-.4, 1.88), (.4, 1.88), (1.26, 1.74), (1.34, 1.20), (1.12, .80), (.40, .60)], c['white'], c['white2'], 40)
    for L in ([(-.96, .96), (-1.04, 1.3), (-1.02, 1.7)], [(.96, .96), (1.04, 1.3), (1.02, 1.7)], [(-.66, 1.1), (-.66, 1.5), (-.60, 1.8)]): d.stroke(hb(spl(L, n=6)), .44, taper=(.2, .3))
    d.part(hb(spl([(-.36, .9), (.36, .9), (.30, 1.9), (-.30, 1.9)], True, n=3, corners=(0, 1, 2, 3))), c['gi'], c['gi2'], w=.62, seed=42)
    co = hb(spl([(-.74, .62), (-.3, .78), (.3, .78), (.74, .62), (.86, .90), (.4, 1.14), (0, 1.20), (-.4, 1.14), (-.86, .90)], True, n=6)); d.part(co, c['white'], c['white2'], w=.72, seed=44, shade=[hb(spl([(-1, .94), (0, 1.04), (.5, .8), (1.2, .7), (1.2, 1.5), (-1, 1.5)], True, n=5))])
    ew = k.get('ear', 0.); far, near = (1, -1) if h.yaw >= 0 else (-1, 1)
    def pear(sg, s0, kk):
        R = h.rot([[sg * .94, .10, -.05]])[0]; e = h.n.at(R[0], R[1], sg * ew); d.part(e(spl([(-sg * .08, -.22), (sg * .26, -.30), (sg * .92 * kk, -.40), (sg * .58 * kk, .02), (sg * .24, .26), (-sg * .06, .22)], True, n=7, corners=(2,))), c['skin'], c['skin2'], w=.72, seed=s0, shade=[e(spl([(-.2 * sg, .12), (sg * .4, -.02), (sg * .9, -.36), (sg * .8, .4), (-sg * .2, .4)], True, n=5))]); d.stroke(e(spl([(sg * .62 * kk, -.26), (sg * .28, -.10), (sg * .08, .06)], n=6)), .44, taper=(.2, .2))
    pear(far, 5, .8); ch = k.get('chew', 0.); h.cheek = .10 * ch
    F = h.outline(); d.part(F, c['skin'], c['skin2'], w=.76, seed=1, shade=[h.n(spl([(1.2, -.5), (.84, -.2), (.90, .24), (.78, .52), (.42, .70), (-.1, .77), (-.6, .70), (-.95, .55), (-.9, 1.2), (1.2, 1.2)], True, n=7))]); pear(near, 6, 1.)
    bl = k.get('blink', 0); ek = 'shut' if bl else k.get('eyes', 'open')
    for sg in (-1, 1): eye(d, h, sg * .36, .12, sg, F, .21, .20, .30, k.get('look', (.1, .05)), kind=ek, iris='#b3262e', white='#fff6e6'); brow(d, h, sg * .37, -.15 + k.get('brow', 0.), sg, F, .27, .36, .115, c['skin2'], .0)
    for x in (-.05, .07): d.stroke(h.F(spl([(x, -.12), (x + .02, -.04), (x, .02)], n=4)), .36, within=F)
    nose(d, h, F, .02, .34); o = k.get('o', .0); x = k.get('mx', .12)
    if o > .15:
        M = spl([(x - .34, .46), (x - .12, .44), (x + .12, .44), (x + .36, .46), (x + .40, .48 + .22 * o), (x + .12, .50 + .36 * o), (x - .14, .50 + .36 * o), (x - .40, .48 + .22 * o)], True, n=8); smouth(d, h, F, M, ell(x, .52 + .36 * o, .24, .16 * o), [spl([(x - .30, .45), (x - .22, .64), (x - .13, .45)], False, n=3, corners=(1,)), spl([(x + .14, .45), (x + .23, .64), (x + .31, .45)], False, n=3, corners=(1,)), spl([(x - .22, .50 + .34 * o), (x - .15, .50 + .34 * o - .12), (x - .08, .50 + .34 * o)], False, n=3, corners=(1,)), spl([(x + .10, .50 + .34 * o), (x + .17, .50 + .34 * o - .12), (x + .24, .50 + .34 * o)], False, n=3, corners=(1,))])
    else:
        a = .03 * math.sin(k.get('t', 0) * 3.1); d.stroke(h.F(spl([(x - .22, .56 + a), (x - .05, .60), (x + .12, .585), (x + .26, .53 - a)], n=8)), .74, taper=(.1, .1), tmin=.4, within=F); d.part(h.F(spl([(x + .10, .59), (x + .145, .68), (x + .19, .57)], True, n=2, corners=(0, 1, 2))), '#fffaf4', '#e2d4e0', w=.36)
        if ch > 0: d.stroke(h.F(spl([(x + .40, .40), (x + .48, .54), (x + .40, .68)], n=6)), .5, taper=(.3, .3), within=F)
    # turban
    dm = spl([(-.90, -.62), (-.78, -1.06), (-.38, -1.38), (.06, -1.45), (.48, -1.34), (.82, -1.02), (.90, -.60)], n=8); d.part(h.n(np.vstack([dm, [(0, -.4)]])), c['gi'], c['gi2'], w=.76, seed=10, shade=[h.n(spl([(.25, -1.6), (.48, -1.04), (.4, -.4), (1.3, -.4), (1.3, -1.6)], True, n=5))], hi=[(h.n(spl([(-.60, -.96), (-.40, -1.20), (-.14, -1.30), (-.26, -1.14), (-.44, -.94)], True, n=6, corners=(0, 2))), '#a56ad8')])
    stars(d, h, c, [(.12, -1.14, .10, .2)]); d.shade(h.n(spl([(-1.2, -.2), (0, .02), (1.2, -.2), (1.2, -.9), (-1.2, -.9)], True, n=6)), within=F)
    tb = spl([(-1.16, -.40), (-1.14, -.74), (-.82, -.96), (-.3, -1.03), (.3, -1.03), (.82, -.96), (1.14, -.74), (1.16, -.40), (1.0, -.22), (.46, -.13), (0, -.11), (-.46, -.13), (-1.0, -.22)], True, n=7); d.part(h.n(tb), c['white'], c['white2'], w=.80, seed=11, shade=[h.n(spl([(-1.3, -.34), (-.5, -.24), (.2, -.24), (.72, -.44), (.80, -1.1), (1.4, -1.1), (1.4, 0), (-1.3, 0)], True, n=6))])
    q = .22 * h.sy
    for L in ([(-1.10, -.60), (-.3 + q, -.52), (.4 + q, -.72), (.84, -.94)], [(-1.04, -.30), (-.2 + q, -.36), (.6 + q, -.50), (1.12, -.72)], [(-.72, -.94), (-.1 + q, -.76), (.3 + q, -.66), (1.06, -.36)]): d.stroke(h.n(spl(L, n=8)), .46, taper=(.1, .3))
    # hand with apple
    hx, hy, hr = k.get('hand', (.95, .86, 0.)); tub(d, hb, [(1.06, 1.40), (1.28, 1.66), (hx + .34, hy + .62)], (.26, .25, .22), c['skin'], c['skin2'], .74, 70, shs=(.12, .1)); tub(d, hb, [(hx + .38, hy + .74), (hx + .30, hy + .48)], (.25, .26), '#e23b3b', '#9a1c33', .70, 72, cap=.2, cap0=.2, shs=(.14, 0))
    an = hb.at(hx, hy, hr, .64); apple(d, an, c, k.get('bite', 0.), 74); hn = hb.at(hx + .16, hy + .20, hr, .60)
    d.part(hn(spl([(.10, -.34), (.40, -.40), (.58, -.10), (.60, .40), (.46, .74), (.10, .74), (.04, .30)], True, n=6)), c['skin'], c['skin2'], w=.68, seed=77, shade=[hn(spl([(.30, -.5), (.40, 0), (.2, .8), (.9, .8), (.9, -.5)], True, n=4))])
    for i, (x0, y0, x1, y1) in enumerate(((.42, -.22, -.20, -.22), (.46, .02, -.24, .04), (.42, .26, -.16, .30))): tub(d, hn, [(x0 + .06, y0 + .06), (x0 - .20, y0 - .02), (x1, y1)], (.135, .13, .10), c['skin'], c['skin2'], .60, 78 + i, shs=(.0, .10)); d.stroke(hn(spl([(x1 - .06, y1 - .04), (x1 - .11, y1 + .0), (x1 - .07, y1 + .06)], n=4)), .3)
    return h


# ---------------------------------------------------------------- Frieza, Cell, Buu
FZ_F = A([(0, -1.02), (.46, -.97), (.82, -.74), (1.0, -.34), (1.01, .06), (.93, .40), (.70, .70), (.32, .88), (0, .91), (-.32, .88), (-.70, .70), (-.93, .40), (-1.01, .06), (-1.0, -.34), (-.82, -.74), (-.46, -.97)])


def b_frieza(d, n, k):
    c = C['frieza']; h = Hd(n, k.get('yaw', .0), k.get('pitch', 0.), front=FZ_F); hb = n.at(0, 0, -k.get('roll', 0.) * .8); t = k.get('t', 0)
    # tail: curls up behind into half a heart
    tl = k.get('tail', 0.); T = A([(.40, 1.62), (1.00, 1.74), (1.46, 1.46), (1.50, .92), (1.44, .46), (1.62, .10), (1.96 + tl * .12, .10 + tl * .04), (2.06 + tl * .22, .42 + tl * .10), (1.86 + tl * .26, .62 + tl * .16)])
    U = limb_union(d, hb, T, (.30, .29, .27, .25, .22, .19, .15, .11, .05)); o = hb.M[:2, :2] @ A([.10, .10]); d.part(U, c['skin'], c['skin2'], w=.78, seed=30, shade=[U + o])
    B = bust(d, hb, [(-.26, .70), (-.70, .86), (-.98, 1.16), (-.86, 1.70), (-.3, 1.84), (.3, 1.84), (.86, 1.70), (.98, 1.16), (.70, .86), (.26, .70)], c['skin'], c['skin2'], 40)
    d.part(hb(spl([(-.36, 1.02), (0, .94), (.36, 1.02), (.40, 1.36), (0, 1.52), (-.40, 1.36)], True, n=6)), c['dome'], c['dome2'], w=.60, seed=42, hi=[(hb(spl([(-.26, 1.10), (-.06, 1.03), (.02, 1.08), (-.18, 1.22)], True, n=5)), c['dh'])])
    for sg in (-1, 1): d.part(hb(spl([(sg * .66, .86), (sg * 1.04, 1.02), (sg * 1.14, 1.34), (sg * .94, 1.44), (sg * .74, 1.16)], True, n=6)), c['dome'], c['dome2'], w=.70, seed=44 + sg, hi=[(hb(spl([(sg * .80, .98), (sg * .96, 1.06), (sg * .98, 1.18), (sg * .84, 1.10)], True, n=5)), c['dh'])])
    d.stroke(hb(spl([(-.5, 1.62), (0, 1.70), (.5, 1.62)], n=6)), .42, within=B)
    for sg in (-1, 1):
        R = h.rot([[sg * .97, .10, -.05]])[0]; e = h.n.at(R[0], R[1]); d.part(e(ell(sg * .10, 0, .20, .25)), c['skin'], c['skin2'], w=.68, seed=5 + sg, shade=[e(ell(sg * .16, .10, .18, .20))]); d.part(e(ell(sg * .13, .0, .10, .14)), '#b98fd0', '#8a5fa8', w=.44)
    F = h.outline(); d.part(F, c['skin'], c['skin2'], w=.78, seed=1, shade=[h.n(spl([(1.2, -.6), (.86, -.2), (.88, .22), (.72, .52), (.36, .72), (-.1, .79), (-.5, .72), (-.9, .5), (-.9, 1.2), (1.2, 1.2)], True, n=7))])
    dm = spl([(-.90, -.24), (-.86, -.60), (-.50, -.93), (0, -1.01), (.50, -.93), (.86, -.60), (.90, -.24), (.62, -.15), (.30, -.24), (0, -.21), (-.30, -.24), (-.62, -.15)], True, n=8); DQ = h.n(A(dm) * [1, 1] + [.10 * h.sy, 0]); d.part(DQ, c['dome'], c['dome2'], w=.66, seed=8, shade=[h.n(spl([(.2, -1.1), (.46, -.82), (.56, -.50), (.40, -.30), (.2, -.2), (1.2, -.1), (1.2, -1.1)], True, n=6))],
        hi=[(h.n(spl([(-.66, -.46), (-.58, -.66), (-.34, -.82), (-.10, -.87), (-.22, -.74), (-.44, -.58), (-.52, -.38)], True, n=6, corners=(0, 3))), c['dh']), (h.n(ell(-.40, -.36, .07, .05, -.5)), c['dh'])]); d.detail(h.n(spl([(.46, -.36), (.66, -.40), (.74, -.54), (.70, -.34), (.56, -.29)], True, n=5)), '#b44ad0', within=DQ, sh='#8a35b0'); stars(d, h, c, [(.10, -.60, .12, .1), (.46, -.66, .08, -.3)])
    bl = k.get('blink', 0); ek = 'shut' if bl else k.get('eyes', 'happy')
    for sg in (-1, 1):
        eye(d, h, sg * .38, .15, sg, F, .23, .25, k.get('ang', .22), (0., .02), iris=c['eye'], kind=ek, closed=k.get('lid', 0.), pup=k.get('pup', 1.), lw=.86, lash=.5); d.detail(h.F(ell(sg * .63, .46, .15, .085)), mix(c['skin'], '#ff6fb0', .55), within=F)
        for q_ in (-1, 0, 1): d.stroke(h.F([(sg * .63 + q_ * .07 + .03, .42), (sg * .63 + q_ * .07 - .03, .50)]), .26, '#d0458a', within=F, taper=(.1, .1))
        if ek != 'happy' and not bl: d.stroke(h.F(spl([(sg * .30, .40), (sg * .31, .52), (sg * .27, .62)], n=5)), .40, '#5a2a66', taper=(.1, .5), within=F)
    for sg in (-1, 1): d.stroke(h.F([(sg * .04, .38), (sg * .055, .42)]), .34, within=F)
    m = k.get('smile', 0.); L = spl([(-.22 - .06 * m, .54 - .05 * m), (-.10, .61), (0, .60), (.10, .61), (.22 + .06 * m, .54 - .05 * m)], n=8); L2 = spl([(-.22 - .06 * m, .54 - .05 * m), (-.10, .62 + .09 * (.4 + m)), (0, .64 + .11 * (.4 + m)), (.10, .62 + .09 * (.4 + m)), (.22 + .06 * m, .54 - .05 * m)], n=8)
    d.part(h.F(np.vstack([L, L2[::-1]])), '#7a1a3a', '#5a0f2a', w=.5); d.stroke(h.F(L), 1.15, c['lip'], taper=(.2, .2), within=F); d.stroke(h.F(L2[8:-8]), 1.0, c['lip'], taper=(.2, .2), within=F)
    for sg in (-1, 1):
        hx, hy = sg * k.get('hx', .80), k.get('hy', .66); tub(d, hb, [(sg * .92, 1.26), (sg * 1.16, 1.56), (hx * 1.08, hy + .50)], (.19, .18, .16), c['skin'], c['skin2'], .70, 70 + sg, shs=(.1 * sg, .08)); hand_cup(d, hb.at(hx, hy, sg * .16, .56), dict(skin=c['skin'], skin2=c['skin2']), 74 + sg, fl=sg)
        for i, (a, l, x) in enumerate(((.26, .50, .24), (.14, .62, .11), (.03, .66, -.03), (-.09, .58, -.17))):
            f_ = hb.at(hx, hy, sg * .16, .56).at(0, 0, 0, sg, 1); e_ = A([x + math.sin(a) * l * 1.0, -.02 - math.cos(a) * l * .99]); d.detail(f_(ell(e_[0], e_[1], .055, .075, a)), '#33203f', sh='#33203f')
    return h


CE_F = A([(0, -.94), (.44, -.89), (.78, -.66), (.94, -.28), (.95, .06), (.90, .38), (.70, .68), (.40, .86), (0, .93), (-.40, .86), (-.70, .68), (-.90, .38), (-.95, .06), (-.94, -.28), (-.78, -.66), (-.44, -.89)])


def b_cell(d, n, k):
    c = C['cell']; h = Hd(n, k.get('yaw', -.10), k.get('pitch', 0.), front=CE_F); hb = n.at(0, 0, -k.get('roll', 0.) * .7); sh_ = .10 * h.sy
    r = np.random.default_rng(3)
    # dark dome behind the crest
    d.part(h.n(spl([(-.72, -.4), (-.62, -1.10), (-.30, -1.42), (.0, -1.50), (.30, -1.42), (.62, -1.10), (.72, -.4)], True, n=7)), c['dark'], c['dark2'], w=.74, seed=2, hi=[(h.n(spl([(-.44, -1.02), (-.26, -1.28), (-.06, -1.36), (-.16, -1.18), (-.30, -.98)], True, n=6, corners=(0, 2))), '#6a56a8')]); cst = [(.06, -1.16, .10, .2), (.32, -.96, .07, -.2)]
    B = bust(d, hb, [(-.36, .66), (-.96, .84), (-1.22, 1.16), (-1.14, 1.72), (-.4, 1.86), (.4, 1.86), (1.14, 1.72), (1.22, 1.16), (.96, .84), (.36, .66)], c['dark'], c['dark2'], 40)
    ch = hb(spl([(-.50, .80), (0, .90), (.50, .80), (.66, 1.20), (.40, 1.90), (-.40, 1.90), (-.66, 1.20)], True, n=6)); d.part(ch, c['green'], c['green2'], w=.66, seed=42, shade=[hb(spl([(.2, .7), (.3, 1.2), (.2, 2), (1.2, 2), (1.2, .7)], True, n=4))]); d.stroke(hb(spl([(-.62, 1.22), (0, 1.34), (.62, 1.22)], n=6)), .5); d.stroke(hb([(0, .92), (0, 1.32)]), .5)
    for sg in (-1, 1):
        P = hb(spl([(sg * .56, .78), (sg * .98, .80), (sg * 1.30, 1.08), (sg * 1.26, 1.46), (sg * .92, 1.30), (sg * .70, 1.04)], True, n=6)); d.part(P, c['green'], c['green2'], w=.72, seed=44 + sg, shade=[hb(spl([(sg * .6, 1.1), (sg * 1.0, 1.1), (sg * 1.3, 1.0), (sg * 1.5, 1.6), (sg * .6, 1.6)], True, n=4))])
        for i in range(7): d.detail(hb(ell(sg * r.uniform(.78, 1.2), r.uniform(.90, 1.3), r.uniform(.03, .065), r.uniform(.025, .05), r.uniform(0, 3))), c['spot'], within=P, sh=mix(c['spot'], '#0a2a2a', .4))
        d.part(hb(spl([(sg * .66, .56), (sg * .96, .36), (sg * 1.02, .66), (sg * .84, .96), (sg * .56, .90)], True, n=5)), c['orange'], c['orange2'], w=.62, seed=46 + sg)
    F = h.outline(); d.part(F, c['skin'], c['skin2'], w=.76, seed=1, shade=[h.n(spl([(1.2, -.6), (.80, -.2), (.82, .24), (.68, .54), (.34, .74), (-.1, .82), (-.5, .74), (-.9, .5), (-.9, 1.2), (1.2, 1.2)], True, n=7))])
    for sg in (-1, 1):                                                                         # violet cheek stripe, orange jaw
        d.detail(h.F(spl([(sg * .46, .30), (sg * .66, .26), (sg * .72, .52), (sg * .62, .82), (sg * .52, .84), (sg * .54, .56)], True, n=6)), c['mark'], within=F, sh=mix(c['mark'], '#2a1050', .4)); d.detail(h.F(spl([(sg * .74, .10), (sg * .99, .0), (sg * .99, .60), (sg * .72, .70), (sg * .78, .40)], True, n=6)), c['orange'], within=F, sh=c['orange2']); d.stroke(h.F(spl([(sg * .74, .12), (sg * .79, .40), (sg * .72, .68)], n=6)), .42, within=F)
    sk_ = h.n(spl(A([(-.66, -.30), (-.60, -1.04), (-.2, -1.22), (.2, -1.22), (.60, -1.04), (.66, -.30)]) + [sh_, 0], True, n=6)); d.part(sk_, c['dark'], c['dark2'], w=0)
    stars(d, h, c, cst)
    # crest: helm with two blades
    for sg in (-1, 1):
        w_ = k.get('crest', 0.) * .06
        hn = spl([(sg * .10, -.40), (sg * .34, -.76), (sg * .56, -1.34), (sg * (.98 + w_), -2.02), (sg * 1.16, -1.36), (sg * 1.18, -.74), (sg * 1.06, -.22), (sg * .98, .16), (sg * .86, -.04), (sg * .62, -.18), (sg * .34, -.20), (sg * .12, -.10)], True, n=7, corners=(3, 7)); P = h.n(A(hn) + [sh_, 0]); d.part(P, c['green'], c['green2'], w=.80, seed=12 + sg, shade=[h.n(A([(sg * .70, -1.0), (sg * 1.02, -1.9), (sg * 1.4, -1.6), (sg * 1.4, .3), (sg * .92, .2), (sg * .86, -.3)]) + [sh_, 0])] if sg > 0 else [h.n(A([(-.10, -.30), (-.40, -.50), (-.62, -1.2), (-.96, -1.96), (-.40, -1.3), (-.1, -.6)]) + [sh_, 0])])
        d.detail(h.n(spl(A([(sg * .62, -1.16), (sg * .92, -1.74), (sg * .98, -1.30), (sg * .96, -.86), (sg * .80, -.80)]) + [sh_, 0], True, n=6, corners=(1,))), c['dark'], within=P, sh=c['dark2']); d.stroke(h.n(spl(A([(sg * .62, -1.16), (sg * .92, -1.74), (sg * .98, -1.30), (sg * .96, -.86), (sg * .80, -.80), (sg * .62, -1.16)]) + [sh_, 0], n=6, corners=(1,))), .46, within=P)
        for i in range(11): x, y = r.uniform(.30, 1.08), r.uniform(-.80, -.12); d.detail(h.n(ell(sg * x + sh_, y, r.uniform(.03, .075), r.uniform(.025, .05), r.uniform(0, 3))), c['spot'], within=P, sh=mix(c['spot'], '#0a2a2a', .4))
        d.detail(h.n(spl(A([(sg * .40, -.64), (sg * .56, -.98), (sg * .58, -.72), (sg * .50, -.54)]) + [sh_, 0], True, n=5)), '#9be06a', within=P, sh='#58b84a')
    d.part(h.n(spl(A([(-.16, -.42), (0, -.62), (.16, -.42), (.10, -.14), (0, .04), (-.10, -.14)]) + [sh_, 0], True, n=6, corners=(1, 4))), c['green'], c['green2'], w=.6, seed=16)
    d.shade(h.F(spl([(-.9, -.1), (-.34, -.06), (0, .14), (.34, -.06), (.9, -.1), (.9, .06), (.36, .06), (0, .26), (-.36, .06), (-.9, .06)], True, n=6)), within=F)
    bl = k.get('blink', 0); ek = 'shut' if bl else k.get('eyes', 'open')
    for sg in (-1, 1): eye(d, h, sg * .35, .20, sg, F, .19, .16, .34, k.get('look', (.2, .05)), iris=c['eye'], kind=(k.get('eyeL', ek) if sg < 0 and not bl else ek), closed=.18); brow(d, h, sg * .36, .02 + (k.get('brow', 0.) if sg > 0 else 0.), sg, F, .24, .36, .075, '#241834', .0)
    nose(d, h, F, .0, .44); x = 0.; m = k.get('smile', 1.)
    M = spl([(x - .20, .60), (x - .04, .63), (x + .14, .61), (x + .30, .53), (x + .25, .63 + .05 * m), (x + .08, .68 + .06 * m), (x - .08, .67 + .03 * m)], True, n=7, corners=(0, 3)); smouth(d, h, F, M, None, [spl([(x - .16, .60), (x + .28, .54), (x + .22, .66 + .04 * m), (x - .06, .66)], True, n=4)], w=.56)
    if k.get('ting', 0): g = k['ting']; o = h.F([(.17, .60)])[0]; s = h.n.s * .16 * g; d.fl.fill(star(o[0], o[1], s, 0., .16, 4), '#ffffff'); d.sc.fill(star(o[0], o[1], s, 0., .16, 4), '#ffffff')
    hx, hy, hr = k.get('hand', (-1.10, .06, .42)); sk = dict(skin=c['skin'], skin2=c['skin2']); tub(d, hb, [(-1.06, 1.22), (-1.46, 1.36), (hx - .12, hy + .56)], (.25, .24, .20), c['green'], c['green2'], .74, 70, shs=(.02, .14)); tub(d, hb, [(hx - .20, hy + .72), (hx - .11, hy + .44)], (.21, .23), c['dark'], c['dark2'], .66, 72, cap=.2, cap0=.2, shs=(.12, 0)); hn = hb.at(hx, hy, hr, .58); hand_point(d, hn, sk, 74, fl=-1)
    f_ = hn.at(0, 0, 0, -1, 1); d.detail(f_(ell(-.10, -.92, .07, .10)), '#241834', sh='#241834')
    return h


BU_F = A([(0, -.90), (.50, -.86), (.90, -.62), (1.12, -.18), (1.18, .22), (1.06, .60), (.72, .84), (.32, .92), (0, .93), (-.32, .92), (-.72, .84), (-1.06, .60), (-1.18, .22), (-1.12, -.18), (-.90, -.62), (-.50, -.86)])


def b_buu(d, n, k):
    c = C['buu']; g = k.get('grin', 0.); fr = BU_F.copy(); fr[:, 0] *= 1 + .10 * g * np.exp(-((fr[:, 1] - .45) / .35) ** 2); h = Hd(n, k.get('yaw', .0), k.get('pitch', 0.), front=fr); hb = n.at(0, 0, -k.get('roll', 0.) * .7); lg = k.get('lag', (0, 0))
    ant = A([(.02, -.78), (.04, -1.14), (.26 + lg[0] * .4, -1.50 + lg[1] * .3), (.74 + lg[0] * .8, -1.62 + lg[1] * .7), (1.14 + lg[0] * 1.2, -1.30 + lg[1] * 1.1), (1.20 + lg[0] * 1.6, -.94 + lg[1] * 1.5)]); T = tube(ant, (.30, .27, .25, .22, .18, .13), cap=.9)
    bc = hb(spl([(-.7, .7), (-1.30, .92), (-1.56, 1.4), (-1.44, 1.84), (1.44, 1.84), (1.56, 1.4), (1.30, .92), (.7, .7)], True, n=6)); d.part(bc, c['cape'], c['cape2'], w=.78, seed=38, shade=[hb(spl([(-1.6, 1.3), (-.6, 1.5), (.6, 1.5), (1.6, 1.3), (1.6, 2), (-1.6, 2)], True, n=4))])
    B = bust(d, hb, [(-.50, .66), (-1.00, .88), (-1.22, 1.24), (-1.16, 1.74), (-.4, 1.88), (.4, 1.88), (1.16, 1.74), (1.22, 1.24), (1.00, .88), (.50, .66)], c['skin'], c['skin2'], 40)
    for sg in (-1, 1): d.part(hb(spl([(sg * .46, .70), (sg * .98, .86), (sg * 1.22, 1.24), (sg * 1.16, 1.76), (sg * .62, 1.84), (sg * .46, 1.3)], True, n=6)), '#2b2140', '#171026', w=.70, seed=42 + sg); d.part(hb(tube([(sg * .46, .74), (sg * .46, 1.28), (sg * .62, 1.84)], (.06, .07, .07))), c['gold'], c['gold2'], w=.46)
    d.part(hb(spl([(-.66, .74), (0, .92), (.66, .74), (.74, .94), (0, 1.14), (-.74, .94)], True, n=6, corners=(0, 2, 3, 5))), c['cape'], c['cape2'], w=.66, seed=46); d.part(hb(ell(0, 1.0, .15, .15)), c['gold'], c['gold2'], w=.5)
    for i, (x, y) in enumerate(((-.92, .05), (-.97, .25), (.92, .05), (.97, .25))): pass
    d.part(h.n(T), c['skin'], c['skin2'], w=.80, seed=4, shade=[h.n(T + [.06, .14])]); d.stroke(h.n(spl(ant[1:5] + [-.10, -.10], n=8)), 1.0, c['hi'], taper=(.3, .4))
    F = h.outline(); d.part(F, c['skin'], c['skin2'], w=.80, seed=1, shade=[h.n(spl([(1.4, -.6), (.98, -.2), (1.02, .24), (.84, .58), (.42, .76), (-.1, .81), (-.6, .76), (-1.0, .56), (-1, 1.2), (1.4, 1.2)], True, n=7))])
    d.part(h.n(spl([(-.34, -.62), (-.28, -.96), (.0, -1.06), (.30, -.98), (.34, -.62), (.0, -.50)], True, n=6)), c['skin'], c['skin2'], w=0, shade=[h.n(spl([(.12, -1.1), (.16, -.7), (.5, -.6), (.5, -1.1)], True, n=4))]); d.detail(h.n(spl([(-.80, -.34), (-.66, -.60), (-.40, -.74), (-.50, -.54), (-.66, -.26)], True, n=6, corners=(0, 2))), c['hi'], within=F, sh=c['hi'])
    for sg in (-1, 1):
        for i in range(3): P = h.F(ell(sg * (.30 + .16 * i), -.66 + .04 * i + .035 * i * i, .055, .075, sg * .4)); d.part(P, '#8a1f4a', '#6a1238', w=.34)
        d.part(h.n(ell(sg * 1.10 * h.cy + .0, .02, .07, .10)), '#8a1f4a', '#6a1238', w=.4)
    stars(d, h, c, [(.02, -.40, .12, .1)])
    bl = k.get('blink', 0); op = k.get('open', 0.)
    for sg in (-1, 1):
        x = sg * .40; y = .04 - .05 * g
        if op > .3 and not bl: eye(d, h, x, y + .02, sg, F, .19, .15, .36, (0, 0), iris='#d8202e', white='#241524', closed=1 - op)
        else: d.stroke(h.F(spl([(x - .20, y + .05), (x - .08, y - .03), (x + .08, y - .03), (x + .20, y + .05)], n=8)), 1.15, taper=(.12, .12), tmin=.4, within=F)
        d.stroke(h.F(spl([(x - sg * .22, y - .24 - .04 * g), (x, y - .30 - .06 * g), (x + sg * .22, y - .20 - .02 * g)], n=7)), 1.0, '#a8325a', taper=(.2, .3), within=F)
        d.detail(h.F(ell(sg * .78, .34, .19, .115)), mix(c['skin'], '#ff4f6e', .5), within=F)
        for q in (-1, 0, 1): d.stroke(h.F([(sg * .78 + q * .08 + .035, .29), (sg * .78 + q * .08 - .035, .39)]), .28, '#d03a5e', within=F, taper=(.1, .1))
    wd = .16 + .50 * g; up = .02 + .20 * g; dp = .05 + .26 * g
    M = spl([(-wd, .46 - up), (-wd * .5, .47), (0, .48), (wd * .5, .47), (wd, .46 - up), (wd * .78, .48 + dp * .52), (0, .48 + dp), (-wd * .78, .48 + dp * .52)], True, n=8, corners=(0, 4))
    if g > .2:
        tt = [spl([(-wd * .96, .46 - up * .9), (0, .45), (wd * .96, .46 - up * .9), (wd * .80, .50 + dp * .16), (0, .50 + dp * .42), (-wd * .80, .50 + dp * .16)], True, n=6)]; Q = smouth(d, h, F, M, ell(0, .50 + dp * .98, wd * .5, dp * .30), tt, w=.66)
        for u in (-.62, -.36, -.12, .12, .36, .62): d.stroke(h.F([(wd * u, .455 - up * abs(u) * .8), (wd * u, .50 + dp * .34)]), .26, '#a890a8', within=Q, taper=(0, .2))
        for sg in (-1, 1): d.stroke(h.F(spl([(sg * (wd + .02), .40 - up), (sg * (wd + .10), .46 - up), (sg * (wd + .09), .55 - up)], n=6)), .56, taper=(.2, .4), within=F)
    else: d.stroke(h.F(spl([(-wd, .47), (0, .53), (wd, .47)], n=8)), .8, taper=(.12, .12), tmin=.4, within=F)
    for sg in (-1, 1): hx, hy = sg * k.get('hx', 1.16), k.get('hy', 1.34); tub(d, hb, [(sg * 1.10, 1.22), (sg * 1.42, 1.50), (hx * 1.04, hy + .22)], (.28, .27, .24), c['skin'], c['skin2'], .74, 70 + sg, shs=(.1 * sg, .12)); hand_fist(d, hb.at(hx * .98, hy, sg * -.2, .60), c, 74 + sg, '#ffd23f', '#d98f1a', fl=sg)
    return h


BUST = dict(goku=b_goku, vegeta=b_vegeta, gohan=b_gohan, bulma=b_bulma, piccolo=b_piccolo, frieza=b_frieza, cell=b_cell, buu=b_buu)


def puff(d, n, x, y, r, seed=0, col='#ffffff', sh='#cfe4ea', w=.5):
    q = np.random.default_rng(seed); L = [(0, 0, .72)] + [(math.cos(a) * .62, math.sin(a) * .46, q.uniform(.34, .50)) for a in np.arange(6) * 1.047 + q.uniform(0, 1)]
    P = [n(ell(x + a * r, y + b * r, c_ * r, c_ * r * .92, n=24)) for a, b, c_ in L]; U = d.union(P, sm=7)
    if U: d.part(U[0], col, sh, w=w, seed=seed, shade=[n(spl([(x - r * 1.2, y + r * .25), (x, y + r * .50), (x + r * .6, y + r * .1), (x + r * 1.3, y - r * .2), (x + r * 1.3, y + r * 1.3), (x - r * 1.2, y + r * 1.3)], True, n=5))])


# ---------------------------------------------------------------- acting (10 frames, on ones)
POP = {0: (1.03, .97, .02), 1: (1.18, .78, .09), 2: (.85, 1.19, -.13), 3: (1.08, .93, .02), 4: (.97, 1.03, -.02), 5: (1.01, .99, 0.)}


def act(who, f):
    """f 0..9 (later frames hold) -> (dx, dy, rot, sx, sy), params, marks"""
    g = min(f, 9); sx, sy, dy = POP.get(g, (1., 1., 0.)); dx = rot = 0.; k = dict(t=f); mk = []
    if who == 'goku':
        k.update(hand=[(1.06, 1.22, .4, .95), (.98, 1.42, .5, .88), (1.30, 1.00, -.15, 1.0), (1.46, .50, -.12, 1.38), (1.39, .64, .12, 1.26), (1.36, .68, .22, 1.22), (1.36, .68, .18, 1.22)][min(g, 6)], curl=[1., 1., .4, 0., 0, 0, 0][min(g, 6)], spread=[.4, .4, .8, 1.25, 1.1, 1., 1.][min(g, 6)], eyes='happy' if g < 3 else 'open', o=[.5, .3, .8, 1.25, 1.12, 1., 1.][min(g, 6)], lag=[(0, -.1), (0, -.3), (.0, .34), (.05, -.16), (0, .08), (0, -.03)][min(g, 5)], yaw=[.14, .08, .2, .28, .24][min(g, 4)], brow=[.02, .06, 0, -.06, -.03][min(g, 4)])
        rot = [.06, .12, -.04, -.10, -.07, -.05, -.05][min(g, 6)]; k['stretch'] = [(1, 1), (1, .9), (.84, 1.55), (1.08, .94)][g] if g < 4 else (1, 1)
        if g >= 7: k['eyeR'] = 'happy'
        if g == 2: mk.append(('lines', 1.22, 1.62, 1))
        if 3 <= g <= 4: mk.append(('burst', 1.42, -.30, .46 + .16 * (g - 3), g - 3))
        if g >= 7: mk.append(('star', 1.40 + .03 * (g - 7), -.46 - .05 * (g - 7), [.10, .22, .19][min(g, 9) - 7]))
    elif who == 'vegeta':
        k.update(yaw=[.06, .22, -.16, -.62, -.42, -.48, -.46, -.46, -.46, -.46][g], pitch=[.04, .18, -.04, -.24, -.15, -.18, -.16, -.16, -.16, -.16][g], eyes='open' if g < 2 else 'shut', mouth='puff' if g < 3 else 'hmph', look=(0., .05), brow=[.02, .06, 0, -.06, -.02, -.03, -.02, -.02, -.02, -.02][g], lag=[(0, -.1), (-.06, -.2), (.30, .2), (.16, .0), (-.08, 0), (.03, 0), (0, 0)][min(g, 6)], arms=[.02, .12, -.08, -.03, .02, 0][min(g, 5)])
        rot = [.02, .09, -.05, -.15, -.09, -.11, -.10, -.10, -.10, -.10][g]
        if g >= 8: k['eyeN'] = 'open'; k['look'] = (.58, .10)
        if g >= 3: mk.append(('huff', -.60, .52, g - 3))
    elif who == 'gohan':
        k.update(fists=[(.66, 1.30), (.58, 1.52), (.80, 1.06), (1.00, .54), (.91, .74), (.80, 1.00), (.72, 1.20), (1.00, .54), (.92, .72), (.90, .76)][g], eyes='spark' if g >= 3 else 'shut', o=[.4, .2, .7, 1.28, 1.05, .9, .6, 1.3, 1.1, 1.05][g], lag=[(0, -.1), (0, -.24), (0, .24), (0, -.14), (0, .06), (0, 0), (0, -.14), (0, .18), (0, -.06), (0, 0)][g], ball=[.02, .08, .06, -.24, -.10, .02, .06, -.20, -.08, .0][g], brow=[.02, .05, 0, -.06, -.02, 0, .05, -.06, -.02, 0][g])
        rot = [0, 0, 0, .04, .02, .0, -.02, -.06, -.04, -.03][g]
        if g == 6: sx, sy, dy = 1.10, .90, .07
        if g == 7: sx, sy, dy = .93, 1.10, -.10
        if g in (3, 4, 7, 8): mk += [('puffl', s * 1.32, .42, s, (g % 4 + 1) % 4) for s in (-1, 1)]
    elif who == 'bulma':
        k.update(wink=[0, 0, .5, 1, 1, 1, 1, 1, 1, 1][g], hand=[(.92, 1.10, -.35), (.94, 1.30, -.25), (.86, 1.0, -.45), (.82, .86, -.50), (.84, .92, -.46), (.85, .94, -.45)][min(g, 5)], o=[.9, .4, .5, 1.3, 1.1, 1.][min(g, 5)], swing=[(.25, 0.), (.5, -.1), (.1, .3), (-.40, -.12), (-.10, .05), (.18, .0), (.06, 0), (-.05, 0), (-.02, 0), (0, 0)][g], yaw=[-.24, -.30, -.24, -.14, -.17, -.18][min(g, 5)], lag=[(.04, -.1), (.1, -.22), (.05, .22), (-.08, -.06), (0, .02), (.03, 0), (0, 0)][min(g, 6)], brow=[0, .03, .02, -.07, -.05, -.04][min(g, 5)])
        rot = [.0, -.06, .0, .15, .10, .08, .085, .085, .085, .085][g]
        if g >= 3: mk.append(('winkstar', .80 + .22 * (g - 3) ** .8, -.10 - .24 * (g - 3) ** .9, .13 + .035 * (g - 3), g - 3))
    elif who == 'piccolo':
        k.update(o=[.9, 1.15, 1.30, .0, .0, 0, 0, 0, 0, 0][g], hand=[(.98, .96, -.1), (1.02, .98, -.1), (1.08, .86, -.06), (.50, .60, .10), (.66, .74, .06), (.92, .92, 0.), (1.04, 1.00, -.02), (1.06, 1.02, -.02), (1.06, 1.0, -.02), (1.06, 1.02, -.02)][g], bite=0. if g < 3 else 1., yaw=[.28, .26, .22, .46, .42, .36, .32, .30, .30, .30][g], eyes='open' if g < 3 else ('shut' if g in (3, 4) else 'open'), look=(.45, .25) if g < 5 else (.0, .05), chew=[0, 0, 0, .4, 1., .5, 1., .5, 1., .5][g], ear=[0, -.12, -.18, .22, .06, 0, 0, -.22, .10, 0][g], brow=[-.02, -.04, -.06, .05, .03, 0, 0, 0, 0, 0][g])
        dx = [0, -.08, -.15, .15, .07, .02, 0, 0, 0, 0][g]; rot = [0, -.07, -.14, .14, .07, .02, 0, 0, 0, 0][g]
        if g >= 3: mk.append(('crumbs', .66, .52, g - 3))
    elif who == 'frieza':
        k.update(eyes='happy' if g < 5 else 'open', lid=[0, 0, 0, 0, 0, .55, .12, .0, .0, .0][g], pup=[1, 1, 1, 1, 1, 1., .72, .62, .58, .58][g], ang=[.1, .1, .1, .1, .1, .14, .26, .30, .30, .30][g], smile=[0, .2, .4, .6, .5, .5, .8, 1., 1., 1.][g], tail=[.5, .7, -.5, .5, -.3, .1, -.2, .1, .0, .0][g], hy=[.70, .86, .60, .64, .66][min(g, 4)])
        rot = [.0, .05, -.03, .10, -.07, -.04, 0., 0., 0., 0.][g]
        if f > 9: k['tail'] = .10 * math.sin(f * .7)
        if 3 <= g <= 5: mk += [('heart', -1.30, -.50 - .10 * (g - 3), .14 + .02 * g), ('heart', 1.36, -.70 - .12 * (g - 3), .11 + .02 * g)]
    elif who == 'cell':
        k.update(hand=[(-1.38, .60, .30), (-1.52, .66, .20), (-1.40, .44, .38), (-1.16, .34, .50), (-1.36, .36, .42), (-1.16, .34, .50), (-1.17, .34, .50), (-1.19, .35, .50), (-1.18, .34, .50), (-1.18, .34, .50)][g], brow=[0, 0, 0, -.06, -.02, -.10, -.13, -.13, -.11, -.11][g], yaw=[-.06, -.16, -.12, -.06, -.12, -.07, -.10, -.10, -.10, -.10][g], crest=[0., 1., -1., .5, 0, 0, 0, 0, 0, 0][g], look=(.1, .05) if g < 5 else (-.05, .02), smile=[.4, .3, .5, .9, .7, 1., 1., 1., 1., 1.][g])
        rot = [0, -.05, -.02, .03, -.03, .01, -.03, -.03, -.03, -.03][g]
        if g in (3, 5, 6): mk.append(('tap', -1.00, -.14, g - 3 if g < 5 else g - 5, g >= 5))
        if 6 <= g <= 8: k['ting'] = [1., 1.4, .7][g - 6]
    elif who == 'buu':
        S_ = [(1.10, .90, .05), (1.42, .58, .25), (.74, 1.34, -.18), (1.20, .84, .06), (.93, 1.08, -.05), (1.03, .97, .02), (1.0, 1.0, 0.), (1.02, .98, 0), (1., 1., 0), (1., 1., 0)]; sx, sy, dy = S_[g]
        k.update(grin=[0, .0, .1, 1.28, 1.0, 1.12, 1.08, 1.1, 1.1, 1.1][g], lag=[(.2, .2), (0, .4), (0, -.1), (-.2, .3), (.15, -.15), (-.12, .1), (.06, -.04), (0, .04), (0, 0), (0, 0)][g], hx=[1.2, 1.3, 1.0, 1.2, 1.12, 1.17, 1.17, 1.17, 1.17, 1.17][g], hy=[1.4, 1.5, 1.3, 1.36, 1.28, 1.3, 1.3, 1.3, 1.3, 1.3][g])
        if 3 <= g and f <= 9: mk += [('steam', s * .48, -.66, s, g - 3) for s in (-1, 1)]
    if f > 9:                                                                              # grid: everyone breathes on the beat, Frieza does not move
        if who != 'frieza': u = math.sin((f - 10) / 10. * 2 * pi); sy = 1 + .012 * u; sx = 1 - .010 * u
    return (dx, dy, rot, sx, sy), k, mk


def marks(d, n, mk, who):
    """little drawn effects in front of the sticker (inked, flat)"""
    for m in mk:
        t = m[0]
        if t in ('star', 'winkstar'):
            x, y, r = m[1:4]; d.part(n(star(x, y, r * (1.35 if t == 'star' else 1.2), -pi / 2 + (m[4] * .5 if len(m) > 4 else 0), .46)), '#ffdf4a', '#f29a1f', w=.6)
            if t == 'winkstar':
                for j in (1, 2): q = max(0, m[4] - j * .9); xx, yy = .80 + .22 * q ** .8, -.10 - .24 * q ** .9; d.part(n(star(xx - .04 * j, yy + .06 * j, r * .5 / j, .3, .46)), '#fff3b0', '#f2c14f', w=.4) if m[4] >= j else None
        elif t == 'lines':
            x, y, i = m[1:]
            for j in (-1, 0, 1): d.part(n(tube([(x + j * .20, y + .05 * abs(j)), (x + .05 + j * .18, y - .55 + .1 * abs(j))], (.012, .05))), '#ffffff', '#ffffff', w=.3)
        elif t == 'burst':
            x, y, r, i = m[1:]
            for a in np.linspace(0, 2 * pi, 8, endpoint=False): e = A([math.cos(a + .2), math.sin(a + .2)]); d.part(n(tube([A([x, y]) + e * r, A([x, y]) + e * (r + .26 - .06 * i)], (.045 - .012 * i, .014))), '#ffffff', '#ffffff', w=.3)
        elif t == 'huff':
            x, y, i = m[1:]; i = min(i, 6); q = Eo((i + 1) / 6.)
            for j, (ax, ay, r) in enumerate(((-1.00, -.06, .24), (-.66, .10, .19), (-.34, .02, .13))): puff(d, n, x + ax * q, y + ay * q, r * (.5 + .6 * q), 3 + j)
            if i < 4:
                for a in (-.26, .22): d.stroke(n(A([(x - .20, y + a * .4), (x - .55 - .12 * i, y + a)])), .7, '#ffffff', taper=(.1, .4))
        elif t == 'puffl':
            x, y, s, i = m[1:]
            for a in (-.6, -.05, .5): e = A([s * math.cos(a), math.sin(a)]); d.part(n(tube([A([x, y]) + e * (.10 + .06 * i), A([x, y]) + e * (.36 + .05 * i)], (.035, .014))), '#ffffff', '#ffffff', w=.3)
        elif t == 'crumbs':
            x, y, i = m[1:]; r = np.random.default_rng(5)
            for j in range(9): vx, vy = r.uniform(-.5, 1.0) * .16, r.uniform(-1.1, -.2) * .20; px, py = x + .1 + vx * (i + 1), y + vy * (i + 1) + .045 * (i + 1) ** 2; s = r.uniform(.04, .08); d.part(n(ell(px, py, s, s * .7, r.uniform(0, 3), n=7)), ['#fff2c8', '#e2322e'][j % 2], '#d9b87a', w=.34) if py < 1.6 else None
            if i < 2:
                for a in np.linspace(-1.2, 1.2, 6): e = A([math.cos(a - .5), math.sin(a - .5)]); d.part(n(tube([A([x + .18, y]) + e * .44, A([x + .18, y]) + e * (.74 + .1 * i)], (.04, .012))), '#ffffff', '#ffffff', w=.3)
        elif t == 'heart':
            x, y, s = m[1:]; d.part(n(spl(A([(0, .9), (-.95, -.05), (-.62, -.85), (0, -.35), (.62, -.85), (.95, -.05)]) * s + [x, y], True, n=6, corners=(0, 3))), '#ffe3f2', '#f7a8d2', w=.55)
        elif t == 'tap':
            x, y, i, big = m[1:]
            for a in np.linspace(-2.6, -.8, 4) - .5: e = A([math.cos(a), math.sin(a)]); d.part(n(tube([A([x, y]) + e * (.16 + .08 * i), A([x, y]) + e * (.38 + .10 * i)], (.032, .012))), '#ffffff', '#ffffff', w=.3)
            if big: d.part(n(star(x - .28, y - .50, .20 + .05 * i, .0, .30, 4)), '#fff3a0', '#f2b93f', w=.5)
        elif t == 'steam':
            x, y, s, i = m[1:]; i = min(i, 6)
            for j, (ax, ay, r) in enumerate(((.16, -.14, .10), (.34, -.36, .16), (.60, -.62, .22))):
                q = Eo((i + 1.2 - j * .5) / 4.)
                if q > .05: puff(d, n, x + s * ax * q, y + ay * q, r * (.4 + .7 * q), 11 + j + (s > 0) * 5)
        elif t == 'smear':
            a, b, w, col = m[1:]; a, b = A(a), A(b); mid = (a + b) / 2 + [.10, .10]; d.fl.fill(n(tube([a, mid, b], (.04, w * .7, w))), col, .55); d.sc.fill(n(tube([a, mid, b], (.04, w * .7, w))), col, .55)


def diecut(L, S, border=.085, black=.018, shadow=True):
    """bold silhouette line, white die-cut paper border, small paper shadow"""
    a = L[..., 3]; px = S; m = (a > .5).astype(np.uint8); dt = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); mf = np.zeros_like(m); cv2.drawContours(mf, cs, -1, 1, -1)
    hole = (mf > 0) & (m == 0); nl, lb, st, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8)); big = [q for q in range(1, nl) if st[q, cv2.CC_STAT_AREA] > (px * .26) ** 2 * 3]
    for q in big: mf[lb == q] = 0
    m = mf; k = max(3, int(px * .13) | 1); m2 = cv2.morphologyEx(np.pad(m, k), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))[k:-k, k:-k]; R = px * border; dt2 = cv2.distanceTransform(1 - m2, cv2.DIST_L2, 5); w = cv2.GaussianBlur(np.clip(R - dt2 + .5, 0, 1).astype(f32), (0, 0), max(1., px * .012)); w = np.clip((w - .5) * max(2., px * .03) + .5, 0, 1)
    b = np.maximum(np.clip(px * black - dt + .5, 0, 1), a)
    out = np.zeros_like(L)
    if shadow: s = cv2.GaussianBlur(cv2.warpAffine(w, np.float32([[1, 0, int(px * .02)], [0, 1, int(px * .035)]]), (w.shape[1], w.shape[0])), (0, 0), px * .022) * .38; out[..., 3] = s; out[..., :3] = s[..., None] * rgb('#1c0f2a')
    e = np.clip(w - np.clip((cv2.GaussianBlur(w, (0, 0), 1.2) - .72) * 6, 0, 1), 0, 1)[..., None]
    wc = rgb('#fffdf8') * (1 - .16 * e); out = out * (1 - w[..., None]) + np.concatenate([np.broadcast_to(wc, w.shape + (3,)) * w[..., None], w[..., None]], -1)
    bb = b[..., None]; out = out * (1 - bb) + np.concatenate([rgb(INK) * bb, bb], -1); return out * (1 - L[..., 3:4]) + L


def sticker(who, f, cx, cy, S, blink=0, size=(W, H), ss=2, cut=True, with_marks=True):
    """die-cut sticker layer of that size; frame f of its acting"""
    (dx, dy, rot, sx, sy), k, mk = act(who, f); k['blink'] = blink; k['roll'] = rot
    d = D(0, 0, size[0], size[1], ss, lw=S * .062, line=INK, var=.22, soft=max(.6, S * .004)); base = cy + 1.70 * S                 # squash about the base, not the head
    n = N().at(cx + dx * S, base, 0., S * sx, S * sy).at(0, -1.70 + dy, rot); BUST[who](d, n, k); L = d.rgba(); out = diecut(L, S) if cut else L
    if with_marks and mk: e = D(0, 0, size[0], size[1], ss, lw=S * .055, line=INK, soft=1.); marks(e, N().at(cx, cy, 0., S), mk, who); m = e.rgba(); out = out * (1 - m[..., 3:4]) + m
    return out


def card(who, f=0, w=W, h=H):
    """flat colour, comic rays and halftone, pale tilted card"""
    c = Cel(w, h, 0, 0, 1); c.fill(A([(0, 0), (w, 0), (w, h), (0, h)]), BG[who]); cx, cy = w / 2, h * .52; dk = mix(BG[who], '#1a0a24', .14)
    for a in np.linspace(0, 2 * pi, 18, endpoint=False) + f * .010: e = lambda q: A([math.cos(q), math.sin(q)]) * w; c.fill(A([(cx, cy), A([cx, cy]) + e(a), A([cx, cy]) + e(a + .17)]), dk, .9)
    z = {0: .72, 1: 1.06, 2: .98}.get(f, 1.); s = h * .40 * z; r = -.06 + .02 * z; cr = N().at(cx, cy + h * .01, r); P = cr(ell(0, 0, s * 1.06, s * 1.02, n=60, sq=7.)); c.fill(P + [10, 14], mix(BG[who], '#1a0a24', .40)); c.fill(P, CARD[who])
    a = c.rgba()[..., :3]
    yy, xx = np.mgrid[:h, :w].astype(f32); p = 26.; u = (xx * .866 + yy * .5) / p; v = (-xx * .5 + yy * .866) / p; dd = np.sqrt((u - np.round(u)) ** 2 + (v - np.round(v)) ** 2); g = np.clip((yy / h - .30) * .62 + (np.abs(xx - cx) / w - .28) * .5, 0, .46); m = np.clip((g - dd) * p + .5, 0, 1)[..., None] * .42
    return a * (1 - m) + rgb(mix(BG[who], '#1a0a24', .34)) * m


def scene(who, f, S=248):
    im = card(who, f); return comp.over(im, sticker(who, f, W / 2 + dict(goku=-70, cell=50, frieza=-40, piccolo=-40, bulma=-30).get(who, 0), dict(goku=556, vegeta=600, gohan=596, bulma=530, piccolo=560, frieza=516, cell=610, buu=580).get(who, 560), S))


GRID = [('goku', 292, 300, -.07), ('vegeta', 732, 268, .05), ('gohan', 1186, 262, -.04), ('bulma', 1622, 306, .07), ('piccolo', 290, 800, .06), ('frieza', 734, 812, -.05), ('cell', 1192, 808, .05), ('buu', 1630, 806, -.06)]
_sheet = {}


def sheet_bg():
    if 'bg' not in _sheet:
        c = Cel(W, H, 0, 0, 1); c.fill(A([(0, 0), (W, 0), (W, H), (0, H)]), '#f4f2e8'); r = np.random.default_rng(2)
        for y in range(30, H, 60):
            for x in range(30 + (y // 60 % 2) * 30, W, 60): c.fill(ell(x, y, 3.2, 3.2, n=12), '#e2dfd2')
        a = c.rgba()[..., :3] * (1 + .012 * tooth(H, W))[..., None]; _sheet['bg'] = a
    return _sheet['bg'].copy()


def grid(f=10, blink=0, S=112, pop=None, only=None, ball=(960, 548, 1., 1., 0.), bg=None):
    """the sticker sheet: eight die-cut stickers round the four-star ball.  blink closes every eye but Frieza's; pop(i) -> 0..1 arrival of sticker i"""
    im = sheet_bg() if bg is None else bg
    for i, (w, x, y, ro) in enumerate(GRID):
        q = 1. if pop is None else pop(i)
        if q <= 0: continue
        z = 1 + 1.15 * (1 - min(1, q)) ** 1.6 - .06 * math.sin(pi * max(0., min(1., (q - 1) / .8))); sq = Cel(520, 560, int(x - 260), int(y - 280), 2); n = N().at(x, y + 6, ro, 182); P = n(ell(0, 0, 1, 1.02, n=60, sq=8.)); sq.fill(P + [5, 8], '#c9c4b4', .8); sq.fill(P, BG[w]); sq.fill(n(ell(0, 0, .90, .92, n=60, sq=8.)), mix(BG[w], CARD[w], .45)); a = sq.rgba()
        from db.ink import over_roi
        over_roi(im, a, int(x - 260), int(y - 280)); sz = (540, 600) if z < 1.15 else (1000, 1100); hw, hh = sz[0] // 2, sz[1] // 2; oy = dict(goku=.30, vegeta=.36, gohan=.30, bulma=.12, piccolo=.05, frieza=.0, cell=.36, buu=.14)[w]
        L = sticker(w, f, hw + (-20 if w == 'goku' else 0), hh - 10 + oy * S, S * z, blink=blink if w != 'frieza' else 0, size=sz, with_marks=False); M = cv2.getRotationMatrix2D((hw, hh), -math.degrees(ro) * (.6 + 2.2 * (z - 1)), 1.); L = cv2.warpAffine(L, M, sz, flags=cv2.INTER_LINEAR); over_roi(im, L, int(x - hw), int(y - hh))
    if ball is None: return im
    bx, by, bsx, bsy, spin = ball; oy_ = int(by - 260); d = D(700, oy_, 520, 520, 2, lw=S * .07, line=INK); bn = N().at(bx, by, .0, bsx, bsy); ball4(d, bn, 128, 4, spin=spin, w=1.); L = np.zeros((H, W, 4), f32); paste(L, diecut(d.rgba(), 128, shadow=True), 700, oy_); return comp.over(im, L)
