"""Storybook noon at Kame island (b1): wide plates, painted flat with crayon tooth and a wobbling edge."""
import numpy as np, cv2, math
from db import W, H, comp
from db.paint import rgb, f32, ramp, fbm, warp_by, smooth, grid
from db.ink import Cel, spl
from .chibi_core import *

HOR = 536; GY = 872; SY = 290        # sea plate starts at y = SY                       # horizon, walkers' ground line


def plate(w, h, fn, wob=1.7, grain=.035, pig=.05, seed=1, ss=2):
    """fn(cel) draws flats -> textured premultiplied RGBA float32 (h, w, 4)"""
    c = Cel(w, h, 0, 0, ss); fn(c); a = c.rgba()
    if wob: a = warp_by(a, wob, 26, seed, border=cv2.BORDER_REPLICATE)
    g = tooth(h, w, seed * 37 % 900, seed * 91 % 900); q = fbm(h, w, (90, 34), None, seed + 3).astype(f32); q = (q - q.mean()) / (q.std() + 1e-6)
    a[..., :3] *= (1 + g * grain + q * pig * .5)[..., None]; a[..., :3] = np.minimum(a[..., :3], a[..., 3:4] * 1.02); return a


def sky():
    y = np.linspace(0, 1, H, dtype=f32)[:, None] * np.ones((1, W), f32)
    s = ramp(y, [(0, '#8ccfee'), (.28, '#a9dcf3'), (.46, '#c6e6ff'), (.545, '#eef6ea'), (1, '#f4f1dc')])
    n = fbm(H, W, (160, 60, 20), None, 7, aspect=4.).astype(f32); n = (n - n.mean()) / n.std(); s = s * (1 + .010 * n)[..., None]; return s * (1 + .012 * tooth(H, W))[..., None]


def puffs(c, circ, base, col='#fffdf6', s1='#e3e9f6', s2='#cfdcf0', k=1.):
    """cumulus from circles [(x, y, r)] sitting on a flat base line y = base"""
    xs = [q[0] for q in circ]; lo, hi = min(q[0] - q[2] for q in circ) - 4, max(q[0] + q[2] for q in circ) + 4; top = min(q[1] - q[2] for q in circ) - 4
    c.clip(A([(lo, top), (hi, top), (hi, base - 8), (hi - 40, base), (lo + 40, base), (lo, base - 8)]))
    for x, y, r in circ:                                                         # every puff: lit face, cool crescent low on the right
        c.fill(ell(x, y, r, r * .94, n=46), s1); c.fill(ell(x - r * .08, y - r * .11, r * .93, r * .86, n=46), col); c.fill(ell(x - r * .28, y - r * .34, r * .40, r * .28, n=30), '#ffffff', .85)
    tops = min(q[1] - q[2] for q in circ); hh = base - tops
    for x, y, r in circ:
        if y > base - hh * .34: c.fill(ell(x + r * .10, y + r * .44, r * .92, r * .62, n=46), s1, .9); c.fill(ell(x + r * .16, y + r * .80, r * .86, r * .52, n=46), s2, .9)
    c.unclip()


def cloudbank(seed, cx, base, w, h, n=16):
    r = np.random.default_rng(seed); out = []
    for i in range(n):
        u = r.uniform(-1, 1); e = (1 - abs(u) ** 1.6); rr = h * (.16 + .30 * e * r.uniform(.6, 1.1)); out.append((cx + u * w * .5, base - e * h * r.uniform(.15, .78), rr))
    for i in range(int(n * .7)):
        u = -1 + 2 * (i + .5) / (n * .7); out.append((cx + u * w * .5, base - h * .05, h * r.uniform(.15, .24)))
    out.sort(key=lambda q: q[1]); return out


def clouds_plate(w=2300):
    def fn(c):
        for s, cx, b, ww, hh, n in ((3, 250, 596, 760, 320, 20), (8, 1980, 590, 760, 380, 22), (5, 1120, 580, 640, 130, 16)): puffs(c, cloudbank(s, cx, b, ww, hh, n), b)
        for x, y, r in ((620, 150, 34), (1500, 90, 26), (1180, 250, 22), (330, 60, 22), (1960, 150, 30)):
            puffs(c, [(x - r * 1.3, y, r * .7), (x, y - r * .4, r), (x + r * 1.2, y - r * .05, r * .8), (x + r * 2.2, y + r * .1, r * .5), (x - r * 2.2, y + r * .15, r * .45)], y + r * .55)
    return plate(w, 620, fn, seed=2)


def palm(c, x, y, s, lean=.15, seed=0, tr=('#a97a4c', '#86583a'), lf=('#6fb05a', '#4d9150', '#8cc56a')):
    r = np.random.default_rng(seed); top = A([x + lean * s, y - s]); P = [(x, y), (x + lean * s * .25, y - s * .5), top]
    c.fill(tube(P, (s * .055, s * .042, s * .034)), tr[0]); c.fill(tube(A(P) + [s * .02, 0], (s * .026, s * .02, s * .016)), tr[1], .8)
    for k in range(7): q = k / 7.; p = spl(A(P), n=20)[int(q * 39)]; c.line(spl([p + [-s * .045, 0], p + [0, s * .012], p + [s * .045, 0]], n=4), 2.0, tr[1], taper=(.1, .1))
    A_ = [-2.9, -2.35, -1.9, -1.35, -.85, -.3, .12, 3.0]
    for i, a in enumerate(A_):
        a += r.normal(0, .08); L = s * r.uniform(.42, .56); e = A([math.cos(a), math.sin(a)]); tip = top + e * L + [0, L * .42]; m = top + e * L * .55 + [0, -L * .05]
        F = np.vstack([bez(top, top + e * L * .3 + [0, -L * .20], m + [0, -L * .16], tip, 12), bez(tip, m + [0, L * .18], top + e * L * .2 + [0, L * .10], top, 12)]); c.fill(F, lf[i % 2])
        mid = bez(top, top + e * L * .3 + [0, -L * .05], m, tip, 12); c.line(mid, 2.0, lf[1] if i % 2 == 0 else '#3f7d48')
        for j in range(3, 11, 2): c.fill(A([mid[j], mid[j] + [s * .012, s * .10], mid[j] + [s * .030, s * .02]]), lf[2] if i % 2 else lf[1])
    for o in ((-.03, .02), (.03, .03), (.0, .05)): c.fill(ell(top[0] + o[0] * s, top[1] + o[1] * s, s * .030), '#8a5a38')


def island(c, x, y, s):
    """Kame island: x centre, y water line, s = half width"""
    L = lambda P: A(P, float) * s + [x, y]
    c.fill(L(spl([(-1.15, .02), (-.8, -.10), (-.3, -.17), (.3, -.17), (.8, -.10), (1.15, .02), (.6, .07), (0, .085), (-.6, .07)], True, n=8)), '#f6e6b8'); c.fill(L(spl([(-1.05, .02), (-.4, .035), (.4, .035), (1.05, .02), (.5, .07), (-.5, .07)], True, n=6)), '#e3cb92')
    c.fill(L(spl([(-.92, -.07), (-.6, -.20), (-.1, -.26), (.5, -.22), (.90, -.08), (.4, -.10), (-.3, -.09)], True, n=8)), '#b9d27c')
    for px, ps, l, sd in ((-.62, .86, -.14, 3), (-.30, .62, .10, 4)): palm(c, x + px * s, y - .14 * s, ps * s, l, sd)
    # the house: pink walls, red roof with upturned eaves, porch, three painted signs
    H_ = lambda P: A(P, float) * s * .82 + [x + .24 * s, y - .16 * s]
    c.fill(H_([(-.46, .02), (-.46, -.40), (.02, -.44), (.02, .02)]), '#f9c3c0'); c.fill(H_([(.02, .02), (.02, -.44), (.50, -.38), (.50, .0)]), '#e79fa5')
    c.fill(H_([(-.46, -.40), (-.22, -.84), (.02, -.44)]), '#f9c3c0')
    c.fill(H_(spl([(-.22, -.96), (.30, -.92), (.64, -.36), (.04, -.42)], True, n=3, corners=(0, 1, 2, 3))), '#d9584d'); c.fill(H_(spl([(-.22, -.96), (-.58, -.36), (-.50, -.32), (-.21, -.80), (.06, -.38), (.12, -.44)], True, n=3, corners=(0, 1, 2, 3, 4, 5))), '#b8413e')
    for k in range(4): c.line(H_([(-.10 + k * .03, -.88 + k * .12), (.36 + k * .065, -.86 + k * .13)]), 1.6, '#b8413e', taper=(.1, .1))
    c.fill(H_([(.30, -.90), (.30, -1.06), (.40, -1.06), (.40, -.80)]), '#e79fa5'); c.fill(H_([(-.38, -.04), (-.38, -.28), (-.22, -.29), (-.22, -.04)]), '#6fa7c6'); c.fill(H_([(-.16, .02), (-.16, -.30), (-.02, -.31), (-.02, .02)]), '#8a5a3c'); c.fill(H_([(.14, -.10), (.14, -.29), (.34, -.27), (.34, -.10)]), '#6fa7c6')
    c.fill(H_([(.14, -.20), (.34, -.185), (.34, -.175), (.14, -.19)]), '#fff6ea'); c.fill(H_([(-.31, -.04), (-.295, -.04), (-.295, -.29), (-.31, -.29)]), '#fff6ea')
    for k, (dx, sh) in enumerate(((.0, 0), (.09, 1), (.18, 0))):                         # KAME: painted marks, not letters
        o = H_([(-.36 + dx * 1.25, -.52 + .0 * sh)])[0]; c.line(A([o + [0, -4], o + [0, 4]]), 2.2, '#c24a45'); c.line(A([o, o + [5, -4]]), 2.0, '#c24a45'); c.line(A([o, o + [5, 4]]), 2.0, '#c24a45')
    palm(c, x + .92 * s, y - .10 * s, .80 * s, .18, 9)
    for bx, by, br, col in ((-.80, -.08, .10, '#6fa860'), (-.46, -.12, .13, '#86ba66'), (.70, -.10, .12, '#6fa860'), (.02, -.09, .08, '#86ba66')):
        for q in (-.7, 0, .7): c.fill(ell(x + (bx + q * br) * s, y + by * s, br * s, br * s * .75), col)
    for rx, rr in ((-1.0, .07), (.98, .09), (1.10, .05)): c.fill(ell(x + rx * s, y - .0 * s, rr * s, rr * s * .7), '#a9a39b'); c.fill(ell(x + (rx + .015) * s, y + .015 * s, rr * s * .8, rr * s * .45), '#8b867f')


def sea_plate(w=2500, ix=1660, isz=236):
    """y starts at SY (transparent above the horizon except the island and hills)"""
    h = H - SY; o = HOR - SY
    def fn(c):
        c.grad(A([(0, o), (w, o), (w, h), (0, h)]), (0, o), (0, h), [(0, '#9ad8ee', 1), (.06, '#79c9ea', 1), (.18, '#5db8e6', 1), (.40, '#47a7dd', 1), (.70, '#3d9bd4', 1), (1, '#368dc6', 1)])
        for x, hh, ww, col in ((300, 34, 420, '#b4d6e4'), (760, 22, 300, '#c2dee8'), (2150, 30, 420, '#b4d6e4')): c.fill(spl([(x - ww / 2, o + 1), (x - ww * .2, o - hh * .6), (x, o - hh), (x + ww * .22, o - hh * .55), (x + ww / 2, o + 1)], n=8), col)
        r = np.random.default_rng(5)
        for i in range(46):                                                          # long soft swells, then clustered dashes
            u = r.random() ** 1.3; y = o + 10 + u * 520; L = r.uniform(260, 620) * (.4 + u); x = r.uniform(-100, w); th = 3 + 16 * u
            c.fill(spl([(x - L / 2, y), (x - L * .2, y - th * .5), (x + L * .25, y - th * .3), (x + L / 2, y), (x + L * .1, y + th * .5)], True, n=8), ['#8fd4ee', '#3187c3', '#7accea'][i % 3], .30)
        for k in range(170):
            u = r.random() ** 1.3; y0 = o + 8 + u * 520; x0 = r.uniform(0, w)
            for q in range(int(r.integers(2, 6))):
                y = y0 + r.normal(0, 3 + 9 * u); L = (12 + 80 * u) * r.uniform(.5, 1.4); x = x0 + r.normal(0, 30 + 110 * u); th = 1.5 + 5.5 * u
                c.fill(spl([(x - L / 2, y), (x - L * .1, y - th * .5), (x + L / 2, y + th * .1), (x + L * .05, y + th * .55)], True, n=5), ['#c8edf4', '#9ad8ee', '#2f86c0', '#effaf6'][int(r.choice(4, p=[.36, .26, .22, .16]))], .78)
        wl = o + 66; c.fill(spl([(ix - isz * 1.5, wl + 10), (ix - isz * .6, wl - 14), (ix + isz * .6, wl - 14), (ix + isz * 1.5, wl + 10), (ix + isz * .5, wl + 46), (ix - isz * .5, wl + 46)], True, n=8), '#8fe0e0', .8)
        island(c, ix, wl, isz)
        for i in range(11): q = (i - 5) / 5.; c.fill(spl([(ix + q * isz * 1.1 - 34, wl + 14 + (i % 2) * 5), (ix + q * isz * 1.1, wl + 10 + (i % 2) * 5), (ix + q * isz * 1.1 + 34, wl + 14 + (i % 2) * 5), (ix + q * isz * 1.1, wl + 19 + (i % 2) * 5)], True, n=5), '#f2fbf6', .95)
        for i in range(70): q = r.normal(0, .5); y = wl + 22 + r.random() * 60; c.fill(spl([(ix + q * isz - 18, y), (ix + q * isz, y - 2), (ix + q * isz + 18, y), (ix + q * isz, y + 2.5)], True, n=4), ['#b7cf9a', '#f0b3ae', '#cfe6c0', '#d98a80'][i % 4], .5)
        sx, sy = 560, o + 26; c.fill(A([(sx, sy), (sx, sy - 46), (sx + 28, sy - 8)]), '#fff9ea'); c.fill(A([(sx - 6, sy - 2), (sx - 6, sy - 32), (sx - 22, sy - 6)]), '#f4d9c8'); c.fill(spl([(sx - 28, sy + 2), (sx + 34, sy + 2), (sx + 24, sy + 11), (sx - 18, sy + 11)], True, n=4, corners=(0, 1, 2, 3)), '#b3604a')
    return plate(w, h, fn, seed=4, wob=1.3)


def jetty_plate(w=2760, beach=2080):
    """planks (walking surface y 826..918), fascia beam, posts in the water; sand and grass from x = beach on"""
    y0, y1, yb = 822, 920, 948; h = H - 700; oy = 700; WL = 1012
    def fn(c):
        r = np.random.default_rng(11); Y = lambda y: y - oy
        # posts first
        for x in range(60, beach + 80, 290):
            c.fill(A([(x - 19, Y(yb - 6)), (x + 19, Y(yb - 6)), (x + 17, Y(WL)), (x - 17, Y(WL))]), '#a9794f'); c.fill(A([(x + 3, Y(yb - 6)), (x + 19, Y(yb - 6)), (x + 17, Y(WL)), (x + 3, Y(WL))]), '#86583c'); c.fill(A([(x - 18, Y(WL - 22)), (x + 18, Y(WL - 22)), (x + 17, Y(WL)), (x - 17, Y(WL))]), '#4f7a66', .6)
            for k in range(2): c.fill(spl([(x - 34 - k * 18, Y(WL + 1 + k * 8)), (x, Y(WL - 3 + k * 8)), (x + 34 + k * 18, Y(WL + 1 + k * 8)), (x, Y(WL + 5 + k * 8))], True, n=6), '#d7f1f2', .85 - .3 * k)
        c.fill(A([(0, Y(y0 + 1)), (beach + 70, Y(y0 + 1)), (beach + 70, Y(y1)), (0, Y(y1))]), '#6f4a34')
        x = -60.
        while x < beach + 60:
            pw = r.uniform(62, 84); t = .16; sk = lambda q: q - (q - w / 2) * .0
            col = mix('#e0b27a', '#c9935c', r.random() * .8); P = A([(x + 10, Y(y0)), (x + pw + 8, Y(y0)), (x + pw - 2, Y(y1)), (x, Y(y1))]); c.fill(P, '#8d6140'); c.fill(A([(x + 12, Y(y0 + 1)), (x + pw + 5, Y(y0 + 1)), (x + pw - 6, Y(y1)), (x + 2, Y(y1))]), col)
            for k in range(3): xx = x + r.uniform(10, pw - 12); c.line(spl([(xx + 6, Y(y0 + r.uniform(4, 26))), (xx + r.normal(2, 2), Y((y0 + y1) / 2)), (xx - 1, Y(y1 - r.uniform(4, 26)))], n=6), 1.7, mix(col, '#7a4b2c', .45), taper=(.3, .3), seed=int(xx))
            if r.random() < .35: c.line(spl(ell(x + pw * .5, Y(y0 + r.uniform(25, 70)), 7, 4), True, n=2), 1.6, mix(col, '#7a4b2c', .5))
            for yy in (y0 + 9, y1 - 10): c.fill(ell(x + pw * .30, Y(yy), 2.8, 2.1), '#7a5138'); c.fill(ell(x + pw * .72, Y(yy), 2.8, 2.1), '#7a5138')
            x += pw + 1.5
        c.fill(A([(0, Y(y1)), (beach + 60, Y(y1)), (beach + 60, Y(yb)), (0, Y(yb))]), '#a9764a'); c.fill(A([(0, Y(y1)), (beach + 60, Y(y1)), (beach + 60, Y(y1 + 7)), (0, Y(y1 + 7))]), '#d9a56c'); c.fill(A([(0, Y(yb - 8)), (beach + 60, Y(yb - 8)), (beach + 60, Y(yb)), (0, Y(yb))]), '#7f5436')
        for k in range(130): xx = r.uniform(0, beach); L = r.uniform(30, 160); c.line([(xx, Y(r.uniform(y1 + 10, yb - 10))), (xx + L, Y(r.uniform(y1 + 10, yb - 10)))] if False else spl([(xx, Y(y1 + 14 + (k % 3) * 5)), (xx + L / 2, Y(y1 + 15 + (k % 3) * 5)), (xx + L, Y(y1 + 14 + (k % 3) * 5))], n=4), 1.5, '#8a5c3a', taper=(.3, .3))
        # beach: wet edge, foam line, dry sand, grass
        edge = spl([(beach - 250, Y(1110)), (beach - 190, Y(1046)), (beach - 120, Y(1000)), (beach - 60, Y(950)), (beach + 0, Y(886)), (beach + 70, Y(842)), (beach + 260, Y(812)), (w + 40, Y(800))], n=10)
        def off(dx, dy): return np.vstack([edge + [dx, dy], [(w + 40, Y(1110))]])
        c.clip(A([(0, Y(yb - 1)), (w + 60, Y(yb - 1)), (w + 60, Y(1120)), (0, Y(1120))])); c.fill(off(-74, 6), '#9be3df', .75); c.fill(off(-40, 4), '#f4fbf4'); c.fill(off(-26, 5), '#bfe6dc'); c.unclip(); c.fill(off(-12, 3), '#d9bd85'); c.fill(off(14, 2), '#f6e6b8')
        c.clip(A([(0, Y(yb - 1)), (w + 60, Y(yb - 1)), (w + 60, Y(1120)), (0, Y(1120))]))
        for k in range(26): q = edge[int((k + .5) / 26 * (len(edge) - 60))] + [-36 + r.normal(0, 5), r.normal(0, 3)]; c.fill(ell(q[0], q[1], r.uniform(5, 13), r.uniform(3, 6)), '#f7fdf8', .95)
        c.unclip()
        c.fill(spl([(beach + 190, Y(826)), (beach + 330, Y(790)), (w + 40, Y(770)), (w + 40, Y(838)), (beach + 420, Y(842))], True, n=8), '#d7df8a'); c.fill(spl([(beach + 40, Y(1110)), (beach + 210, Y(1034)), (beach + 420, Y(1004)), (w + 40, Y(980)), (w + 40, Y(1110))], n=8), '#ecd59c'); c.fill(spl([(beach + 130, Y(1110)), (beach + 360, Y(1036)), (w + 40, Y(1000)), (w + 40, Y(1110))], n=8), '#d7df8a')
        for k in range(120):
            gx, gy = (r.uniform(beach + 230, w), r.uniform(806, 836)) if k % 2 else (r.uniform(beach + 330, w), r.uniform(1030, 1078))
            for j in (-1, 0, 1): c.fill(A([(gx + j * 6 - 3, Y(gy)), (gx + j * 11, Y(gy - 20 - (k % 3) * 5)), (gx + j * 6 + 3, Y(gy))]), ['#b5c86a', '#9dbb5c', '#c9d67c'][k % 3])
        for k in range(8):
            fx, fy = beach + 380 + k * 47, Y(1042 + (k * 37) % 30)
            for j in range(5): c.fill(ell(fx + math.cos(j * 1.257) * 6, fy + math.sin(j * 1.257) * 6, 4.5), '#fffdf4')
            c.fill(ell(fx, fy, 3.6), '#f4c04a')
        for k in range(90): c.fill(ell(r.uniform(beach + 60, w), Y(r.uniform(866, 1010)), r.uniform(1, 2.8), r.uniform(.8, 1.7)), ['#d3b377', '#fff5d6', '#c9a15f'][k % 3])
        for sx, sy, rr in ((beach + 126, 952, 17), (beach + 560, 954, 14)): c.fill(star(sx, Y(sy), rr, .4, .5), '#e8845e'); c.fill(star(sx, Y(sy), rr * .5, .4, .5), '#f3a884')
    return plate(w, h, fn, seed=6, wob=1.2), oy


def fg_plate(w=3400):
    """near the lens, soft: reeds in the corners; moves faster than the jetty"""
    def fn(c):
        r = np.random.default_rng(21)
        for cx, n in ((250, 26), (3050, 26)):
            for k in range(n):
                x = cx + r.normal(0, 110); hh = r.uniform(60, 210); b = r.normal(0, 46); c.fill(np.vstack([bez((x - 9, 300), (x - 7, 300 - hh * .5), (x + b * .5, 300 - hh * .8), (x + b, 300 - hh), 8), bez((x + b, 300 - hh), (x + b * .5 + 9, 300 - hh * .7), (x + 11, 300 - hh * .4), (x + 9, 300), 8)]), ['#7fae58', '#a9c86a', '#5f9550'][k % 3])
    a = plate(w, 300, fn, seed=8, wob=0); return cv2.GaussianBlur(a, (0, 0), 3.2), H - 300


def gull(d, x, y, s, t, seed=0):
    """small storybook gull seen from the side, wings beat"""
    n = N().at(x, y, -.06, s); w = math.sin(t * .50 + seed)
    for sg, col, sh in ((1, '#e3e8f0', '#c3cbdc'), (-1, '#fffdf6', '#d5dcea')):
        u = w * (1. if sg < 0 else .92); tip = A([-.50 - .25 * abs(u) + sg * .22, -1.15 * u - .10]); P = spl([(.18, -.05), (.02, -.50 * u - .12), tip * .5 + [.02, -.16 * abs(u)], tip, tip * .55 + [-.16, .16 * (1 if u > 0 else -1) * .6], (-.30, -.02)], True, n=6, corners=(3,)); d.part(n(P), col, sh, w=.38, seed=seed + sg); d.detail(n(spl([tip, tip * .86 + [.07, .03], tip * .80 + [-.05, .07]], True, n=3, corners=(0, 1, 2))), '#6b6470')
    d.part(n(spl([(-.78, -.02), (-.2, -.16), (.3, -.20), (.56, -.10), (.50, .04), (.2, .13), (-.3, .10)], True, n=6, corners=(0,))), '#fffdf6', '#d5dcea', w=.40, seed=seed, shade=[n([(-.8, .03), (.6, .0), (.5, .3), (-.6, .3)])]); d.part(n([(.54, -.10), (.78, -.03), (.53, .0)]), '#f5b53c', w=.3); d.detail(n(ell(.40, -.09, .035, .04)), '#33201c')


def crab(d, x, gy, s, t):
    """walks sideways, claws up on the beat"""
    n = N().at(x, gy - .42 * s - abs(math.sin(t * .314)) * .08 * s, .06 * math.sin(t * .314), s); u = math.cos(t * .628)
    for sg in (-1, 1):
        for k in range(3): a = .25 * math.sin(t * .628 + k * 1.6 + sg); d.stroke(n(spl([(sg * .30, .08), (sg * (.52 + k * .07), -.02 + k * .10 + a * .2), (sg * (.56 + k * .05), .40)], n=5)), 1.1, '#a53a35', taper=(0, .3))
        c_ = n.at(sg * .62, -.34 - .10 * u, sg * (.25 + .2 * u)); d.stroke(n(spl([(sg * .34, -.02), (sg * .56, -.06), (sg * .62, -.26)], n=5)), 1.6, '#c94a3f', taper=(0, 0)); d.part(c_(spl([(-.02, .16), (-.20, -.02), (-.17, -.26), (-.03, -.34), (-.02, -.14), (.06, -.34), (.20, -.22), (.18, .04)], True, n=5, corners=(3, 4, 5))), '#e8604e', '#c2463c', w=.8)
    d.part(n(ell(0, .04, .46, .32, sq=2.3)), '#e8604e', '#c2463c', w=.9, shade=[n([(-.6, .14), (.6, .14), (.6, .5), (-.6, .5)])])
    for sg in (-1, 1): d.stroke(n([(sg * .14, -.2), (sg * .16, -.40)]), 1.0, '#a53a35', taper=(0, 0)); d.detail(n(ell(sg * .16, -.44, .07, .085)), '#33201c'); d.detail(n(ell(sg * .16 + .02, -.47, .024, .028)), '#ffffff')
    d.stroke(n(spl([(-.10, .10), (0, .16), (.10, .10)], n=5)), .7)


_rn = {}
def reflect(base, layer, y_line, y_to=H, amt=.45, squash=.62, f=0, tint='#3a93c9'):
    """jetty and walkers mirrored in the water below y_line, broken by ripples"""
    h = y_to - y_line; yv = np.arange(h); k = 1 + yv / 24.
    if h not in _rn: q = fbm(h + 80, W + 400, (26, 10), None, 31, aspect=7.).astype(f32); _rn[h] = (q - q.mean()) / q.std()
    o = int(f * 1.5) % 80; nz = _rn[h][o:o + h, int(f * 3) % 400:int(f * 3) % 400 + W]
    ys = y_line - (yv[:, None] / squash) + nz * 5.; X = np.arange(W, dtype=np.float32)[None] + nz * 9. * k[:, None] + (6 * np.sin(yv * .30 + f * .3))[:, None]
    q = cv2.remap(layer, X.astype(np.float32), np.clip(ys, 0, H - 1).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT); q = cv2.GaussianBlur(q, (0, 0), 1.6)
    br = np.clip(.62 + .55 * nz, 0, 1)[..., None]; fade = np.linspace(1, .25, h, dtype=f32)[:, None, None] * amt * br
    c = q[..., :3] * .72 + rgb(tint) * q[..., 3:] * .28; base[y_line:y_to] = base[y_line:y_to] * (1 - q[..., 3:] * fade) + c * fade; return base
