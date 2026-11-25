"""e3 drawings: one soft picture-book style for the whole memory (indigo-brown line, flat colour, night / morning palettes).
room(pal) plate, table(pal) layer, vegeta_back(d, n, p), vegeta_profile(d, n, smile, lid), boot(d, n), goku_morning(d, ...), bowl_down(d, n, r)"""
import numpy as np, cv2, math
from db import W, H, comp
from db.paint import rgb, f32, ramp, fbm, grid, smooth
from db.ink import Cel, spl
from db import kit
from .chibi_core import *
from . import chibi_story as ST, chibi_stick as SK, chibi_feast as FE
from .chibi_world import plate

LINE = '#3a2a3a'
PAL = dict(
    night=dict(wall='#5d5788', wall2='#4a4573', beam='#352f55', shoji='#8a8fc6', frame='#3f3762', floor='#6d6690', floor2='#56507a', mat='#4a4470', sky='#8c9fe0', sky2='#5f70b8', scroll='#b9aed0', ring='#352f55', tab='#8b6468', tab2='#5c4358', tabl='#c9b6cf', lamp='#ffe1a0', moon='#f6f4ff'),
    morn=dict(wall='#f9e2cc', wall2='#f3cfb2', beam='#a8705a', shoji='#fff4e2', frame='#bd8468', floor='#f4d6ba', floor2='#e8bf9e', mat='#cf9c80', sky='#fff6ea', sky2='#ffd9b4', scroll='#fdf2e2', ring='#6f4a3a', tab='#c98a5c', tab2='#9a6344', tabl='#ffe9bf', lamp='#f3e3c0', moon='#fffdf0'))


def room(which):
    p = PAL[which]
    def fn(c):
        c.grad(A([(0, 0), (W, 0), (W, 470), (0, 470)]), (0, 0), (0, 470), [(0, p['wall2'], 1), (1, p['wall'], 1)]); c.fill(A([(0, 0), (W, 0), (W, 46), (0, 46)]), p['beam']); c.fill(A([(0, 440), (W, 440), (W, 468), (0, 468)]), p['beam'])
        c.grad(A([(0, 466), (W, 466), (W, H), (0, H)]), (0, 466), (0, H), [(0, p['floor2'], 1), (.35, p['floor'], 1), (1, p['floor2'], 1)])
        for k, y in enumerate((466, 560, 720, 960)): c.fill(A([(0, y), (W, y), (W, y + 5 + k * 2), (0, y + 5 + k * 2)]), p['mat'])
        for x0, x1 in ((330, 180), (960, 960), (1590, 1740)): c.fill(A([(x0, 466), (x0 + 6, 466), (x1 + 12, H), (x1, H)]), p['mat'])
        # shoji
        for x in (300, 520, 740): c.fill(A([(x, 70), (x + 212, 70), (x + 212, 440), (x, 440)]), p['shoji']); [c.fill(A([(x + i * 70, 70), (x + i * 70 + 7, 70), (x + i * 70 + 7, 440), (x + i * 70, 440)]), p['frame']) for i in range(4)]; [c.fill(A([(x, y), (x + 218, y), (x + 218, y + 7), (x, y + 7)]), p['frame']) for y in (70, 162, 254, 346, 434)]
        # scroll with the brush ring
        c.fill(A([(1120, 92), (1262, 92), (1262, 380), (1120, 380)]), p['scroll']); c.fill(A([(1108, 84), (1274, 84), (1274, 96), (1108, 96)]), p['beam']); c.fill(A([(1108, 376), (1274, 376), (1274, 390), (1108, 390)]), p['beam']); a = np.linspace(.4, 6.1, 60); r = 40 + 3 * np.sin(a * 2); c.line(np.c_[1191 + np.cos(a) * r, 226 + np.sin(a) * r], 11, p['ring'], taper=(.02, .35), press=.4, seed=3)
        # open door, sky
        c.fill(A([(1500, 46), (1860, 46), (1860, 468), (1500, 468)]), p['beam']); c.grad(A([(1522, 60), (1838, 60), (1838, 468), (1522, 468)]), (0, 60), (0, 468), [(0, p['sky2'], 1), (1, p['sky'], 1)]); c.fill(spl([(1522, 468), (1522, 402), (1600, 376), (1680, 396), (1760, 366), (1838, 392), (1838, 468)], n=6), mix(p['sky2'], p['beam'], .45))
        # patch of light on the floor
        c.fill(A([(1522, 470), (1838, 470), (1500, 860), (960, 860)]), p['sky'], .24)
        # standing lantern on the left
        c.fill(ell(150, 560, 92, 20), p['beam'], .6); c.fill(A([(92, 300), (208, 300), (214, 540), (86, 540)]), p['lamp']); [c.fill(A([(x, 296), (x + 8, 296), (x + 8, 552), (x, 552)]), p['beam']) for x in (86, 146, 206)]; [c.fill(A([(82, y), (220, y), (220, y + 8), (82, y + 8)]), p['beam']) for y in (294, 416, 540)]
    a = plate(W, H, fn, seed=31, wob=1.6, grain=.04, pig=.045)[..., :3]; X, Y = grid()
    if which == 'night':
        g = np.exp(-(((X - 150) / 620) ** 2 + ((Y - 430) / 460) ** 2)); a = a * (.82 + .12 * smooth(900, 1800, X))[..., None]; a = a * (1 + g[..., None] * (rgb('#ffc98a') * 1.5 - .5)) + g[..., None] * rgb('#ff9a4a') * .16 + (np.exp(-(((X - 150) / 130) ** 2 + ((Y - 420) / 170) ** 2)) * .34)[..., None] * rgb('#fff0c0'); a = a + (np.exp(-(((X - 1680) / 330) ** 2 + ((Y - 300) / 330) ** 2)) * .12)[..., None] * rgb('#b9c8ff')
    else:
        g = np.exp(-(((X - 1680) / 800) ** 2 + ((Y - 320) / 620) ** 2)); a = a * (.80 + .14 * g[..., None]) + g[..., None] * rgb('#fff2c8') * .10
    return np.clip(a, 0, 1.2).astype(f32)


TBL = (1010., 1046., 640., 226.)                  # foreground low table: centre, radii
PLT = (1010., 886.)                               # plate centre


def table(which):
    p = PAL[which]; cx, cy, rx, ry = TBL
    def fn(c):
        c.fill(ell(cx, cy + 40, rx + 4, ry + 6, n=120), mix(p['tab2'], '#1a1228', .5)); c.fill(ell(cx, cy + 26, rx, ry, n=120), p['tab2']); c.fill(ell(cx, cy, rx, ry, n=120), p['tab']); c.clip(ell(cx, cy, rx - 8, ry - 4, n=120)); r = np.random.default_rng(8)
        for k in range(90): y = cy + r.uniform(-1, 1) * ry; x = cx + r.uniform(-1, 1) * rx; L = r.uniform(80, 380); c.line(spl([(x - L / 2, y), (x, y + r.normal(0, 2)), (x + L / 2, y)], n=6), r.uniform(1.2, 2.6), p['tab2'], taper=(.3, .3), seed=k, alpha=.5)
        c.unclip(); t = np.linspace(pi * 1.03, pi * 1.97, 80); c.line(np.c_[cx + (rx - 12) * np.cos(t), cy + (ry - 6) * np.sin(t)], 5, p['tabl'], taper=(.3, .3), alpha=.8)
    a = plate(W, H, fn, seed=33, wob=1.3, grain=.05, pig=.07); X, Y = grid()
    if which == 'night': g = np.exp(-(((X - 700) / 620) ** 2 + ((Y - 930) / 260) ** 2))[..., None]; a[..., :3] = a[..., :3] * (.70 + .34 * g * rgb('#ffd9a8'))
    return a


def light_layer(L, which):
    """lantern from the left, moon from the right, over a cel layer"""
    X, Y = grid(*L.shape[:2]); o = L.copy()
    if which == 'night': g = np.clip((X - 150) / 1500, 0, 1); o[..., :3] *= ramp(g, [(0, '#ffe0b0'), (.28, '#f4d0b4'), (.55, '#c3bfe0'), (1, '#b4bff0')])
    else: g = np.clip((X - 150) / 1600, 0, 1); o[..., :3] *= ramp(g, [(0, '#f6dcc0'), (.6, '#fff2da'), (1, '#fff8e8')])
    return o


def dish(d, x, y, r, which='night'):
    c = dict(night=('#f2e9df', '#a99ab8', '#5c6ea8'), morn=('#fffaf0', '#e3c5a2', '#4f78b8'), grey=('#d9d9d6', '#8f9294', '#6f7478'))[which]; n = N().at(x, y); d.fl.fill(n(ell(6, 30, r * 1.06, r * .25)), '#2a1c2e', .42); d.part(n(ell(0, 14, r, r * .262, n=80)), c[0], c[1], w=1., shade=[n(ell(0, 34, r, r * .22, n=60))]); d.part(n(ell(0, 4, r * .78, r * .17, n=70)), mix(c[0], '#ffffff', .25), c[1], w=.5); d.stroke(n(ell(0, 13, r * .90, r * .222, n=70)), .55, c[2], taper=(0, 0))


def bun_soft(d, n, r=1., which='night'):
    """our bun in the memory's colours; node at its base centre"""
    c = dict(night=('#f6e8da', '#b3a0bb', '#fffaf0', '#8c6a72'), morn=('#fffaf0', '#f0cfa6', '#ffffff', '#b07a52'))[which]
    P = n(spl(A([(-.98, -.02), (-1.04, -.36), (-.80, -.86), (-.40, -1.14), (-.10, -1.26), (.06, -1.32), (.22, -1.24), (.46, -1.12), (.84, -.82), (1.04, -.36), (.98, -.02), (.5, .12), (-.5, .12)]) * r, True, n=8)); d.part(P, c[0], c[1], w=.9, seed=1, shade=[n(spl(A([(-1.2, -.2), (-.60, -.10), (0, -.12), (.46, -.40), (.70, -.80), (1.3, -.8), (1.2, .3), (-1.2, .3)]) * r, True, n=6))])
    for a, b, e in (((-.02, -1.20), (-.34, -.98), (-.46, -.62)), ((.02, -1.18), (-.12, -.90), (-.16, -.56)), ((.08, -1.18), (.16, -.90), (.18, -.58)), ((.12, -1.20), (.40, -.98), (.50, -.64))): d.stroke(n(spl(A([a, b, e]) * r, n=7)), .55, c[3], taper=(.06, .55))
    d.detail(n(spl(A([(-.66, -.64), (-.50, -.84), (-.30, -.94), (-.44, -.70), (-.56, -.48)]) * r, True, n=5)), c[2], within=P, sh=c[2]); d.stroke(n(spl(A([(-.06, -1.26), (.04, -1.36), (.12, -1.24)]) * r, n=5)), .6, taper=(.1, .1))


def bowl_down(d, n, r=1., which='night'):
    """a rice bowl upside down; node at the centre of its rim (the rim rests on the plate)"""
    c = dict(night=('#e9e0da', '#9a8fb4', '#5c6ea8'), morn=('#fffaf0', '#dcc0a4', '#4f78b8'))[which]
    P = n(np.vstack([spl(A([(-1., 0), (-.95, -.28), (-.70, -.62), (-.40, -.74), (-.38, -.88), (.38, -.88), (.40, -.74), (.70, -.62), (.95, -.28), (1., 0)]) * r, n=6, corners=(3, 4, 5, 6)), np.c_[np.cos(np.linspace(0, pi, 22)) * r, np.sin(np.linspace(0, pi, 22)) * .20 * r][1:-1]]))
    d.part(P, c[0], c[1], w=1., seed=4, shade=[n(spl(A([(.16, -1.), (.36, -.6), (.30, .3), (1.3, .3), (1.3, -1)]) * r, True, n=5)), n(spl(A([(-1.1, .02), (0, .12), (1.1, .02), (1.1, .4), (-1.1, .4)]) * r, True, n=5))])
    for i in range(5): x = -.62 + .31 * i; d.detail(n(spl(A([(x - .12, -.30), (x, -.46), (x + .12, -.30), (x, -.36)]) * r, True, n=4)), c[2], within=P, sh=mix(c[2], '#2a1c3a', .4))
    t = np.linspace(pi, 0, 24); d.stroke(n(np.c_[np.cos(t) * .965 * r, np.sin(t) * .16 * r - .07 * r]), .55, c[2], taper=(.02, .02), within=P); d.stroke(n(spl(A([(-.36, -.80), (0, -.76), (.36, -.80)]) * r, n=6)), .5)


def boot(d, n, which='night'):
    """Vegeta's boot and shin, toe to the left; node at the heel-ground point, units ~ foot length 1.6"""
    c = dict(night=('#efe8e4', '#9d93ba', '#e9b36a', '#a5764e', '#3a48a0', '#232c68'))[which] if which == 'night' else ('#fff8ea', '#d8c0a8', '#f4bd45', '#c98a2c', '#3157ab', '#233f84')
    L = n(spl([(-.12, -1.3), (-.24, -2.4), (-.20, -3.6), (-.42, -5.0), (-.46, -8.2), (1.0, -8.2), (.92, -5.0), (.62, -3.6), (.68, -2.4), (.50, -1.3)], True, n=6, corners=(4, 5))); d.part(L, c[4], c[5], w=1., seed=3, shade=[n(spl([(.18, -1.2), (.26, -3), (.30, -5), (.3, -8.6), (1.4, -8.6), (1.4, -1.2)], True, n=5))]); d.stroke(n(spl([(-.10, -1.9), (.08, -2.2), (.16, -2.8)], n=6)), .6, taper=(.1, .5)); d.stroke(n(spl([(-.22, -3.7), (.10, -3.9), (.34, -3.8)], n=6)), .55, taper=(.1, .5)); d.stroke(n(spl([(.60, -3.5), (.36, -3.66), (.20, -3.6)], n=6)), .5, taper=(.1, .5))
    B = n(spl([(.50, -1.72), (.56, -1.0), (.50, -.48), (.58, -.22), (.56, -.02), (.0, .02), (-.5, .02), (-1.06, .0), (-1.14, -.18), (-.98, -.42), (-.60, -.56), (-.20, -.70), (-.14, -1.1), (-.22, -1.66), (.12, -1.78)], True, n=7)); d.part(B, c[0], c[1], w=1.05, seed=4, shade=[n(spl([(.20, -1.9), (.24, -1.0), (.16, -.5), (-.2, -.30), (-1.2, -.14), (-1.2, .2), (1., .2), (1., -1.9)], True, n=6))])
    d.part(n(spl([(-.62, -.55), (-.98, -.42), (-1.14, -.18), (-1.06, .0), (-.66, .02), (-.56, -.26)], True, n=6)), c[2], c[3], w=.8, seed=5, shade=[n(spl([(-1.2, -.12), (-.5, -.14), (-.5, .2), (-1.2, .2)], True, n=3))]); d.stroke(n(spl([(-.16, -1.0), (.10, -.92), (.52, -1.0)], n=6)), .5); d.stroke(n(spl([(-.12, -.74), (.10, -.62), (.30, -.66)], n=6)), .45, taper=(.1, .5)); d.stroke(n(spl([(.56, -.14), (.0, -.10), (-.56, -.10)], n=6)), .5)


# ---------------------------------------------------------------- Vegeta, from behind, walking away
def vegeta_back(d, n, p=0., turn=0.):
    """n: ground point, 1 unit = head half width; about 6.3 tall.  p = walk phase (2 steps per 1), turn -1..1 turns the head to its profile side"""
    s, s2, w, w2, g, g2 = '#4a5aa8', '#2c3674', '#e9e2ea', '#8c84b4', '#d9a866', '#8f6a5c'; a = 2 * pi * p; bob = .05 * math.cos(2 * a); sw = .06 * math.sin(a)
    def leg(sg, ph):
        up = max(0., math.sin(ph)); kx = sg * (.40 + .03 * up); ax = sg * (.36 - .04 * up); lift = .46 * up; hip = A([sg * .36 + sw * .4, -2.45 + bob]); kn = A([kx, -1.40 - .10 * up]); an = A([ax, -.52 - lift]); T = tube([hip, hip * .5 + kn * .5 + [sg * .06, 0], kn, an], (.54, .52, .42, .34)); d.part(n(T), s, s2, w=.95, seed=20 + sg, shade=[n(T + [.18, 0])])
        B = spl([(ax - .36, -1.16 - lift * .9), (ax - .31, -.62 - lift), (ax - .34, -.22 - lift * .8), (ax - .30, .0 - lift * .55), (ax + .30, .0 - lift * .55), (ax + .34, -.22 - lift * .8), (ax + .31, -.62 - lift), (ax + .36, -1.16 - lift * .9)], True, n=6); d.part(n(B), w, w2, w=.95, seed=22 + sg, shade=[n(B + [.18, .02])])
        if up > .25: d.part(n(ell(ax, -.05 - lift * .5, .22, .10 + .10 * up)), '#7d6a80', '#4a3f5c', w=.7)
        d.stroke(n(spl([(ax - .29, -.86 - lift), (ax, -.80 - lift), (ax + .29, -.86 - lift)], n=5)), .5)
    leg(-1, a); leg(1, a + pi)
    t = n.at(sw, bob, .02 * math.sin(a))
    for sg in (-1, 1):
        q = .16 * math.sin(a + (0 if sg > 0 else pi)); T = tube([(sg * 1.02, -4.02), (sg * 1.34, -3.5), (sg * 1.38, -2.85 + q * .2), (sg * 1.30, -2.35 + q * .3)], (.36, .33, .27, .22), cap0=.8); d.part(t(T), s, s2, w=1., seed=30 + sg, shade=[t(T + [.14, .04])]); h = t.at(sg * 1.27, -2.30 + q * .3, sg * .10); d.part(h(spl([(-.24, -.36), (-.25, .06), (-.20, .40), (.0, .52), (.20, .40), (.25, .06), (.24, -.36)], True, n=6)), w, w2, w=.95, seed=34 + sg, shade=[h(spl([(.06, -.5), (.10, .6), (.5, .6), (.5, -.5)], True, n=3))]); d.stroke(h(spl([(-.24, -.14), (0, -.08), (.24, -.14)], n=5)), .5); d.stroke(h(A([(-.04, .20), (-.03, .46)])), .45); d.stroke(h(A([(.08, .20), (.08, .44)])), .45)
    B = t(spl([(-.34, -4.34), (-.86, -4.20), (-1.14, -3.94), (-1.02, -3.3), (-.72, -2.7), (-.70, -2.28), (.70, -2.28), (.72, -2.7), (1.02, -3.3), (1.14, -3.94), (.86, -4.20), (.34, -4.34)], True, n=7)); d.part(B, s, s2, w=1., seed=40, shade=[t(spl([(.2, -4.5), (.4, -3.4), (.2, -2.2), (1.4, -2.2), (1.4, -4.5)], True, n=4))])
    P = t(spl([(-.60, -4.22), (-.94, -4.04), (-.96, -3.4), (-.74, -2.82), (-.70, -2.52), (.70, -2.52), (.74, -2.82), (.96, -3.4), (.94, -4.04), (.60, -4.22), (.50, -3.96), (-.50, -3.96)], True, n=6)); d.part(P, w, w2, w=1., seed=42, shade=[t(spl([(.16, -4.4), (.34, -3.5), (.14, -2.4), (1.4, -2.4), (1.4, -4.4)], True, n=4))])
    d.part(t(spl([(-.52, -3.12), (.52, -3.12), (.46, -2.50), (-.46, -2.50)], True, n=3, corners=(0, 1, 2, 3))), g, g2, w=.8, seed=43, shade=[t(A([(.10, -3.2), (.6, -3.2), (.6, -2.4), (.06, -2.4)]))]); d.stroke(t(spl([(-.50, -2.92), (0, -2.88), (.50, -2.92)], n=5)), .5, '#7a5a50'); d.stroke(t(spl([(-.48, -2.70), (0, -2.66), (.48, -2.70)], n=5)), .5, '#7a5a50'); d.stroke(t(spl([(0, -3.94), (.02, -3.5), (0, -3.14)], n=5)), .6, '#7f779c')
    for sg in (-1, 1): d.part(t(spl([(sg * .52, -4.26), (sg * .92, -4.08), (sg * .90, -3.78), (sg * .56, -3.92)], True, n=4)), g, g2, w=.7, seed=44 + sg)
    nk = t(spl([(-.36, -4.72), (.36, -4.72), (.40, -4.30), (0, -4.20), (-.40, -4.30)], True, n=5)); d.part(nk, '#c9a9b4', '#6c5f8c', w=.9, seed=46); d.part(t(spl([(-.44, -4.40), (0, -4.28), (.44, -4.40), (.46, -4.20), (0, -4.06), (-.46, -4.20)], True, n=5)), s, s2, w=.9, seed=47)
    hd = t.at(0, -5.30, .0, 1.06); h = Hd(hd, pi - .02 - turn, .0, (.0, .03 * math.sin(a * 2)), 0., 0.); c = dict(hair='#2c2542', hair2='#191430', hair3='#5d5f9c')
    for sg in (-1, 1):
        R = h.rot([[sg * .96, .17, -.02]])[0]; d.part(h.n(ell(R[0] + .05 * np.sign(R[0] + 1e-6), R[1], .12, .17)), '#c9a9b4', '#6c5f8c', w=.8)
    L = sorted([q for q in ST.HAIR['vegeta']], key=lambda q: lock_z(h, q)); polys = [lock_poly(h, q) for q in L] + [h.outline(.02)]; U = d.union(polys)[0]; d.part(U, c['hair'], c['hair2'], w=1., seed=50, shade=[U + A([.22, .02]) * h.n.s])
    for i, q in enumerate(L[-7:]): F = lock2(h, q); Aa, B_ = F[:13], F[12:][::-1]; d.stroke(h.n((Aa * .55 + B_ * .45)[2:10]), .55, c['hair3'], taper=(.4, .55), seed=i)
    return t


PROF = A([(0, -.96), (.42, -.89), (.70, -.68), (.82, -.40), (.865, -.14), (.83, .02), (.885, .16), (.985, .255), (.945, .33), (.865, .37), (.885, .445), (.845, .52), (.865, .60), (.835, .68), (.865, .78), (.79, .92), (.58, .985), (.28, .93), (-.04, .80), (-.22, .66), (-.48, .60), (-.78, .34), (-.96, -.05), (-.94, -.45), (-.70, -.78), (-.36, -.93)])


def vegeta_profile(d, n, smile=0., lid=0., t=0.):
    """close profile, facing right, the moon in front of him, in the picture-book manner.  n = head centre"""
    c = dict(skin='#ece2ec', skin2='#8f86b6', hair='#302848', hair2='#1d1736', h1='#5a609c', h2='#9aa7e0')
    h = Hd(n, 1.40, 0., (-.02, .02 * math.sin(t * .4)), t, .015, front=PROF, side=PROF); q = lambda P: n(A(P, float))
    # neck, collar, armour
    d.part(n(spl([(-.56, .40), (.30, .84), (.36, 1.24), (.66, 1.9), (-1.1, 1.9), (-.78, 1.1)], True, n=6)), c['skin'], c['skin2'], w=1., seed=2, shade=[n(spl([(-.9, .3), (.4, .76), (.42, 1.04), (.0, 1.16), (-.12, 2.0), (-1.2, 2.0)], True, n=5))])
    d.part(n(spl([(-.92, 1.08), (-.3, 1.34), (.38, 1.32), (.64, 1.50), (.86, 2.1), (1.2, 3.2), (-2.6, 3.2), (-2.2, 1.9), (-1.3, 1.46)], True, n=7)), '#4a5aa8', '#2c3674', w=1.05, seed=3, shade=[n(spl([(-2.6, 1.2), (-.6, 1.5), (-.2, 2.2), (-.1, 3.3), (-2.6, 3.3)], True, n=5))]); d.stroke(n(spl([(-.82, 1.30), (-.2, 1.56), (.54, 1.54)], n=7)), .6)
    d.part(n(spl([(-.74, 1.86), (-.1, 2.02), (.58, 1.94), (.90, 2.4), (1.05, 3.3), (-1.8, 3.3), (-1.5, 2.3)], True, n=6)), '#efe9f2', '#8c84b4', w=1., seed=4, shade=[n(spl([(-1.9, 1.8), (-.3, 2.1), (.1, 2.6), (.1, 3.4), (-1.9, 3.4)], True, n=5))]); d.part(n(spl([(-1.44, 1.66), (-.78, 1.84), (-.84, 2.40), (-1.54, 2.18)], True, n=5)), '#d9a866', '#8f6a5c', w=.85, seed=5)
    SK.shair_back(d, h, 'vegeta', c, 10, w=.8)
    F = h.outline(); d.part(F, c['skin'], c['skin2'], w=1., seed=6, shade=[h.n(spl([(-1.2, -1.2), (.06, -1.2), (.16, -.30), (.24, .12), (.18, .46), (.40, .78), (.36, 1.2), (-1.2, 1.2)], True, n=7))])
    d.detail(q(ell(.50, .40, .17, .10)), mix(c['skin'], '#f09a9a', .22 + .20 * smile), within=F, sh=mix(c['skin2'], '#c986a4', .30))
    # eye: an ink oval under a heavy lid that comes down a little when he smiles
    op = max(.05, 1 - lid); Q = q(ell(.665, .075, .060, .115)); d.detail(Q, '#241c34', within=q(A([(.50, .20 - .235 * op), (.84, .18 - .22 * op), (.84, .30), (.50, .30)])), sh='#241c34'); 
    if op > .35: d.detail(q(ell(.682, .075 - .02 * op, .020, .028)), '#ffffff', within=Q, sh='#ffffff')
    y0 = .19 - .235 * op; d.stroke(q(spl([(.765, y0 - .015), (.66, y0 - .035), (.56, y0 - .01), (.46, y0 + .055)], n=8)), 1.15, taper=(.15, .3), tmin=.4)
    d.part(q(spl([(.82, -.175), (.70, -.215), (.54, -.235), (.36, -.215), (.52, -.165), (.70, -.135), (.81, -.115)], True, n=5, corners=(0, 3))), c['hair'], c['hair2'], w=.35)
    d.stroke(q(spl([(.90, .275), (.905, .32), (.865, .335)], n=5)), .45)
    u = smile; d.stroke(q(spl([(.835, .535), (.74, .548 - .010 * u), (.64, .540 - .045 * u), (.575, .520 - .095 * u)], n=8)), .85, taper=(.10, .20), tmin=.35)
    if u > .25: d.stroke(q(spl([(.555, .37 - .02 * u), (.525, .43), (.545, .49 - .02 * u)], n=6)), .42 * u, '#7a6f9e', taper=(.3, .4))
    d.stroke(q(spl([(.22, .88), (.05, .78), (-.10, .60)], n=6)), .42, '#6a6194', taper=(.3, .4), within=F)
    L = [q_ for q_ in ST.HAIR['vegeta'] if lock_z(h, q_) >= -.12]; L.sort(key=lambda q_: lock_z(h, q_))
    line = [(.76, -.60), (.60, -.46), (.40, -.36), (.30, -.14), (.20, .04), (.10, .20), (.02, .02), (-.12, -.10), (-.26, .02), (-.28, .28), (-.24, .46), (-.44, .56)]
    cap = n(np.vstack([spl(line, n=6, corners=(3, 5)), spl([(-.80, .36), (-.99, -.1), (-.93, -.5), (-.6, -.9), (0, -1.0), (.5, -.86)], n=6)])); d.part(cap, c['hair'], c['hair2'], w=.95, seed=12)
    for i, q_ in enumerate(L): SK.slock(d, h, q_, c, 20 + i, .8, deep=.16)
    d.part(n(spl([(.62, -.60), (.38, -.46), (.24, -.22), (.06, -.10), (-.16, -.20), (-.38, .02), (-.50, .36), (-.74, .28), (-.92, -.1), (-.84, -.5), (-.5, -.83), (.1, -.93), (.48, -.80)], True, n=6)), c['hair'], c['hair2'], w=0, shade=[n(spl([(-1.2, -1.2), (-.3, -1.), (-.2, -.3), (-.5, .6), (-1.2, .6)], True, n=5))])
    for a_, b_, e_ in (((.46, -.52), (.32, -.76), (.20, -.98)), ((.16, -.26), (-.08, -.56), (-.32, -.82)), ((-.24, -.12), (-.46, -.32), (-.70, -.50)), ((.32, -.40), (.10, -.68), (-.06, -.94))): d.stroke(n(spl([a_, b_, e_], n=8)), .6, c['h1'], taper=(.4, .5))
    e = n.at(-.10, .16, -.06, 1.05); d.part(e(spl([(-.10, -.22), (.06, -.24), (.14, -.10), (.08, .12), (-.02, .25), (-.12, .12)], True, n=7)), mix(c['skin2'], c['skin'], .45), c['skin2'], w=.9, seed=7, shade=[e(spl([(-.2, -.3), (-.02, -.1), (-.04, .3), (-.3, .3)], True, n=4))]); d.stroke(e(spl([(.06, -.15), (-.04, -.11), (-.02, .02), (.02, .08)], n=6)), .5)
    return h


# ---------------------------------------------------------------- Goku, morning
def goku_morning(d, n, k):
    """sitting up behind the low table.  n = head centre, unit = head half width.  k: state lift|see|joy, t, lift 0..1, o"""
    c = dict(ST.C['goku']); st = k['state']; t = k['t']; by = k.get('by', 0.); b = n.at(0, .02, 0.)
    # quilt round the waist
    q = spl([(-2.1, 2.5), (-1.7, 1.9), (-.9, 1.62), (0, 1.7), (.9, 1.60), (1.7, 1.86), (2.2, 2.5), (2.2, 3.4), (-2.2, 3.4)], True, n=7); P = d.part(n(q), '#e2a79e', '#b9787a', w=1., seed=2, shade=[n(spl([(-2.3, 2.5), (-.6, 2.2), (.6, 2.4), (2.4, 2.1), (2.4, 3.6), (-2.3, 3.6)], True, n=5))])
    for x, y, r in ((-1.4, 2.2, .16), (-.4, 2.06, .13), (.8, 2.1, .15), (1.6, 2.4, .13), (.1, 2.6, .14)): d.detail(n(star(x, y, r, x, .5)), '#ffe9a8', within=n(q), sh='#d9c79a')
    T = spl([(-.28, .78), (-.74, .92), (-.92, 1.3), (-.84, 1.9), (.84, 1.9), (.92, 1.3), (.74, .92), (.28, .78)], True, n=7); d.part(n(T), '#4f6fb8', '#34508e', w=1., seed=4, shade=[n(spl([(-1.1, 1.0), (-.5, 1.2), (-.4, 2.1), (-1.1, 2.1)], True, n=4))]); d.part(n(spl([(-.26, .72), (.26, .72), (.18, 1.02), (0, 1.12), (-.18, 1.02)], True, n=5)), c['skin'], c['skin2'], w=.7)
    for L in ([(-.5, 1.5), (-.36, 1.62), (-.3, 1.8)], [(.5, 1.46), (.38, 1.6), (.34, 1.8)]): d.stroke(n(spl(L, n=5)), .5, taper=(.1, .5))
    h = Hd(n.at(0, by, k.get('roll', 0.)), k.get('yaw', .28), k.get('pitch', .10), k.get('lag', (0, 0)), t, .02); ex = dict(look=(.6, .9), brow=-.10, mouth='o', o=1.)
    if st == 'see': ex = dict(look=(.5, .8), brow=-.30, mouth='open', o=.8, eyes='dot')
    if st == 'joy': ex = dict(eyes='happy', mouth='laugh', o=1.15 + .12 * math.sin(t * .9), brow=-.22)
    def arm_(sg, hp, hk):
        sp = A([sg * .78, 1.12]); el, hp = ik(sp, hp, .78, .78, -sg * 1.); SK.tub(d, n, [el * .9 + sp * .1, el, hp], (.22, .21, .17), c['skin'], c['skin2'], .95, 30 + sg, shs=(.04, .10)); S_ = tube([sp, sp * .5 + el * .5, el], (.30, .29, .27), cap=.2, cap0=.8); d.part(n(S_), '#4f6fb8', '#34508e', w=1., seed=32 + sg, shade=[n(S_ + [-.16, .06])]); u = A(hp) - el; u /= np.hypot(*u) + 1e-6; return n.at(hp[0] + u[0] * .02, hp[1] + u[1] * .02, math.atan2(-u[0], u[1]), 1.35)
    if st in ('lift', 'see'):
        hb = k['hand']; hn2 = arm_(-1, A([-.52, 1.84]), 'open'); ST.hand(d, hn2, c, 'open', seed=44, fl=1); ST.head_goku(d, h, c, ex, 50); hn = arm_(1, hb, 'flat'); return h, hn
    # joy: the bun in both hands beside the cheek
    ST.head_goku(d, h, c, ex, 50); bp = A([.30, 1.30 + by * .5]); 
    bun_soft(d, n.at(bp[0], bp[1] + .28, .06 * math.sin(t * .9)), .52, 'morn')
    for sg in (-1, 1): hn = arm_(sg, bp + [sg * .46 - .02, .10], 'flat'); ST.hand(d, n.at(bp[0] + sg * .46, bp[1] + .22, pi * (1 - .12 * sg), 1.45), c, 'open', seed=60 + sg, fl=-sg)
    return h, None
