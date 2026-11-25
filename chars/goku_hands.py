"""Goku's hands on db.hands: Kamehameha cup, a father's hands over a child's, a fist closing in a puddle.  Blue wristbands included.

    h = kame(R=HD.euler(..), pos, open=0..1)             -> [left Hand, right Hand] (open 1 = cup wide round the light, 0 = heels and tips together)
    h = father(R, pos, close=0..1, child=True)           -> hands back to front: child's two, father's two (close 0 = open above, 1 = closed round them)
    h = puddle(t, R, pos)                                -> [Hand] t 0 = open palm down .. 1 = fist
    g, layer = paint(hands, cam, light='day'|..., w=5, seed=3, rim=...)   -> (geometry g, premultiplied RGBA)      cam = HD.Cam(cx, cy, s, dist)
    kame_centre(h) -> world point between the palms (put the pearl there: cam.px(point))
"""
import numpy as np, math
from db import hands as HD, W, H as HH
from db.hands import Rx, Ry, Rz, D, euler, Hand, Cam, POSES, pose
from db.ink import Cel, spl
from db.kit import rim as rim_, para
from db.paint import rgb
from . import goku_light as GL
A = np.array
BANDS = ((.12, .86),)          # blue wristband: distance below the wrist
CUPR = pose((14, 30, 22, -7, 0), (12, 32, 24, -1, 0), (14, 34, 24, 6, 0), (18, 36, 24, 15, 0), thumb=(58, 30, 16, 6, 20), wrist=(-16, 0), cup=.10)
CUPL = pose((16, 34, 22, -8, 0), (14, 34, 24, -1, 0), (16, 36, 24, 7, 0), (20, 38, 26, 16, 0), thumb=(56, 32, 14, 6, 18), wrist=(-16, 0), cup=.12)
OVER = pose((34, 44, 26, -2, 0), (38, 46, 28, 0, 0), (42, 48, 28, 1, 0), (46, 48, 28, 5, 0), thumb=(34, 46, -10, 10, 20), wrist=(-6, 0), cup=.30)
CHILD = HD.build(len=.86, wid=1.06, boxy=.22, tip=.15, knuckle=.6, palm=.95, wrist=.86, meat=1.15)
BIG = HD.build(len=1.0, wid=1.14, boxy=.60, tip=.62, knuckle=1.16, palm=1.10, wrist=1.18, meat=1.12)


def kame(R=None, pos=(0, 0, 0), open=1., bld=BIG, bands=True, tremble=0, seed=0, scale=1.):
    G = np.eye(3) if R is None else np.asarray(R, float); th = 10 + 30 * open; st = dict(bands=BANDS) if bands else {}
    pr, pl = CUPR, CUPL
    if tremble: pr, pl = HD.tremble(pr, seed, tremble), HD.tremble(pl, seed + 1, tremble)
    # fingers along +y, right hand above (palm down, -z), left below (palm up, +z); heels pressed, crossed a little
    RR = Rz(D(-9)) @ Rx(D(th)); RL = Rz(D(9)) @ Rx(D(-th)) @ Ry(D(180)); p = np.asarray(pos, float)
    return [Hand(pl, 'L', G @ RL, p + G @ A([.04, 0, -.05]) * scale, bld, scale, pivot=(0, .06, -.20), **st), Hand(pr, 'R', G @ RR, p + G @ A([-.04, 0, .05]) * scale, bld, scale, pivot=(0, .06, -.20), **st)]


def kame_centre(h): return (h[0].world(A([0, .95, -.55])) + h[1].world(A([0, .95, -.55]))) / 2


def father(R=None, pos=(0, 0, 0), close=1., child=True):
    """seen from in front and a little above: the child's cup in the middle, the father's hands close from both sides"""
    G = np.eye(3) if R is None else np.asarray(R, float); p = np.asarray(pos, float); out = []
    if child: out += kame(G @ Rx(D(-12)), p + G @ A([0, -.30, .10]), .85, CHILD, False, scale=.50)
    g = 1 - close; P = HD.blend('open', OVER, close)
    out.append(Hand(P, 'L', G @ Rz(D(24 + 16 * g)) @ Ry(D(112 - 24 * g)) @ Rx(D(10)), p + G @ A([.62 + .7 * g, -.30 - .1 * g, -.18 + .2 * g]), BIG, pivot=(0, .55, -.2), bands=BANDS))
    out.append(Hand(P, 'R', G @ Rz(D(-24 - 16 * g)) @ Ry(D(-112 + 24 * g)) @ Rx(D(10)), p + G @ A([-.62 - .7 * g, -.30 - .1 * g, -.18 + .2 * g]), BIG, pivot=(0, .55, -.2), bands=BANDS))
    return out


def puddle(t, R=None, pos=(0, 0, 0), side='R', tremor=0, seed=0):
    """a hand lying palm down in water that closes into a fist"""
    a = pose((12, 26, 14, -6, 0), (14, 28, 16, 0, 0), (18, 32, 18, 4, 0), (24, 36, 20, 10, 0), thumb=(50, 26, 20, 8, 12), wrist=(-18, 0)); q = HD.blend(a, dict(POSES['fist'], wrist=(-10, 0)), t, drag=.26, curl=6)
    if tremor: q = HD.tremble(q, seed, tremor)
    return [Hand(q, side, euler(0, 62, 18) if R is None else R, pos, BIG, bands=BANDS, arm=4.2)]


def palette(light='day'):
    L = GL.light(light) if isinstance(light, str) else light; s = L['skin']; u = L['under']
    return dict(skin=s[0], shade=s[1], line=L['line'], nail=mix_(s[0], s[3], .5), nail_shade=mix_(s[1], s[0], .4), light=s[3], cuff=u[0], cuff_shade=u[1], cuff_light=u[2], cuffin=u[1], band=mix_(u[0], u[1], .4))


def mix_(a, b, t):
    c = rgb(a) * (1 - t) + rgb(b) * t; return '#%02x%02x%02x' % tuple(int(round(float(v) * 255)) for v in c)


def paint(hands, cam, light='day', ldir=None, w=5., seed=3, res=560, clip=(0, 0, W, HH), deep=True, rims=True, width=W, height=HH, pal=None, thr=.14, hat=0.):
    L = GL.light(light) if isinstance(light, str) else light; P = dict(palette(L), **(pal or {})); ld = ldir or L['ldir']
    g = HD.render(hands, cam, light=ld, res=res, clip=clip, thr=thr); c = Cel(width, height); HD.fill_flat(c, g, dict(HD.PAL['day'], **P)); HD.fill_shadow(c, g, dict(HD.PAL['day'], **P))
    if deep: HD.fill_polys(c, g, HD.depth_polys(g, ld, -.34), L['skin'][2], .75)
    HD.fill_overlays(c, g, dict(HD.PAL['day'], **P), w)
    if hat: HD.hatch(c, g, HD.depth_polys(g, ld, .0), gap=.028, w=1.3, col=L['skin'][2], seed=seed, alpha=hat)
    HD.ink_lines(c, g, dict(HD.PAL['day'], **P), w, seed); a = c.rgba()
    if rims:
        for r in L.get('rim', []): m = max(.5, min(1.6, cam.s / 260.)); a = rim_(a, r['d'][0] * m, r['d'][1] * m, r['col'], r['amt'])
    return g, a
