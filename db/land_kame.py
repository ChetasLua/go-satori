"""Kame house at blue hour (b5): window, sill, wind-chime, in real perspective.  All planes are painted textures seen through one lens.

    D = Dusk(); img = D.frame(u, lit=1., chime=(a, b), laugh=0, glow=1., t=f)    u 0 wide .. 1 the ball fills the match circle (960, 486) r 208
    D.ball(u) -> (cx, cy, r) px of the ball on the sill;  lit = window light 0..1;  glow = inner light of the ball 0..1
"""
import numpy as np, cv2, math
from . import W, H
from .paint import rgb, f32, fbm, smooth, mix, grid, Canvas, arcs, jit_color, ramp, warp_by
from .cam import Camera3D
from .comp import blur, over, radial, screen
from .ink import Cel, spl, resample
from .land_core import C, paint_over
from . import land_sky as SK

F = 2600.; RB = .06; BALL = np.array([0., -.16, RB]); D0, D1 = 2.6, RB * F / 208.
WX0, WX1, WZ1 = -.97, .27, .80                          # window opening (x, z) at the wall plane y = 0; its sill is z = 0
CORNER = .44                                             # right corner of the house
M = 1500                                                 # wall texture px per metre


def cam_at(u, dx=0., dz=0.):
    lg = math.exp(math.log(D0) + (math.log(D1) - math.log(D0)) * u); k = (lg - D1) / (D0 - D1)
    px = 960 + 96 * k; pz = 486 + 252 * k; ex = -(px - 960) / F * lg; ez = RB + (pz - 540) / F * lg
    eye = np.array([ex + dx, BALL[1] - lg, ez + dz]); return Camera3D(eye, eye + [0, 1, 0], F)


def _strokes(cv, P, wd, col, op=.5, dry=.3, seed=0, bristle=.06): cv.strokes(P, wd, np.clip(col, 0, 1.2), op=op, dry=dry, seed=seed, bristle=bristle)


class Dusk:
    def __init__(s): s.ok = False

    # ------------------------------------------------------------------------------------ textures
    def setup(s):
        if s.ok: return s
        r = np.random.default_rng(6); s.tex = {}
        # wall: clapboards, sun-bleached rose paint gone violet in the dusk, warm spill round the window
        x0, x1, z0, z1 = -1.5, CORNER, -.62, 1.25; w, h = int((x1 - x0) * M), int((z1 - z0) * M); s.wall = (x0, x1, z0, z1); y, x = np.mgrid[0:h, 0:w].astype(f32); X = x0 + x / M; Z = z1 - y / M
        bh = .118; q = (Z + .03 * 0) / bh; kb = np.floor(q); fr = q - kb; tone = np.random.default_rng(2).normal(0, .05, 64)[(kb.astype(int) % 64)]
        base = ramp(np.clip(.5 + tone * 2 + (fr - .5) * .34 + .06 * fbm(h, w, (300, 90, 24), (1, .5, .3), 3, aspect=3.), 0, 1), [(0, '#141d40'), (.35, '#22305f'), (.7, '#35437a'), (1, '#4d5a90')])
        wob = .0012 * fbm(1, w, (90, 30), (1, .5), 5)[0]
        gap = np.clip(1 - np.abs((fr + wob / bh) * bh - .0) / .0030, 0, 1) + np.clip(1 - np.abs((1 - fr + wob / bh) * bh) / .0012, 0, 1); lip = np.clip(1 - np.abs(fr * bh - .0065) / .0022, 0, 1)
        base = base * (1 - .66 * np.clip(gap, 0, 1)[..., None]) + C('#7f88c0') * (.30 * lip * (fbm(h, w, (200, 40), (1, .6), 9, aspect=8.) > -.5))[..., None]
        sh = smooth(.16, .02, 1.25 - Z) * 0; eav = smooth(.62, .95, Z); base = base * (1 - .5 * eav[..., None])
        dw = np.hypot(np.maximum(np.maximum(WX0 - X, X - WX1), 0), np.maximum(np.maximum(-.02 - Z, Z - WZ1), 0)); sp = np.exp(-dw / .13); s.spillw = None
        warm = C('#ff9a4a') * (sp * (.75 + .35 * (fr < .25)))[..., None]
        cv = Canvas(h, w, img=np.clip(base, 0, 1)); n = 5200; px = r.uniform(0, w, n); pb = r.integers(0, int(h / M / bh) + 1, n); py = (pb + r.uniform(.1, .9, n)) * bh * M; ix, iy = px.astype(int).clip(0, w - 1), py.astype(int).clip(0, h - 1); _strokes(cv, arcs(px, py, r.normal(0, .02, n), r.uniform(60, 420, n), r.normal(0, .03, n), 5), r.uniform(2, 13, n), jit_color(cv.img[iy, ix], r, .012, .03, .07), op=r.uniform(.15, .5, n), dry=r.uniform(.2, .9, n), seed=3)
        n = 70; px = r.uniform(0, w, n); py = (r.integers(0, int(h / M / bh), n) + r.uniform(.02, .10, n)) * bh * M                # paint flaked off along the board edges
        _strokes(cv, arcs(px, py, r.normal(0, .02, n), r.uniform(30, 150, n), r.normal(0, .04, n), 5), r.uniform(2, 7, n), np.stack([C(c) for c in np.array(['#141a38', '#10152f', '#5d679c'])[r.integers(0, 3, n)]]), op=.75, dry=.6, seed=8)
        for k in range(int(h / M / bh) + 1):                                                                             # nail heads
            for xx in np.arange(x0 + .1, x1, .42): cv2.circle(cv.img, (int((xx - x0 + r.normal(0, .01)) * M), int((k + .5) * bh * M)), 4, tuple(float(v) for v in C('#26244a')), -1, cv2.LINE_AA)
        s.tex['wall'] = cv.img.copy(); s.tex['wallw'] = warm.astype(f32)
        hole = ((X > WX0) & (X < WX1) & (Z > -.02) & (Z < WZ1)); s.tex['walla'] = (~hole).astype(f32)
        # casing round the window (own plane, 2 cm proud): mitred boards
        c = Cel(w, h, 0, 0, 1); T = lambda a, b: ((a - x0) * M, (z1 - b) * M); cw = .075
        for (p, col) in [([T(WX0 - cw, WZ1 + cw), T(WX1 + cw, WZ1 + cw), T(WX1, WZ1), T(WX0, WZ1)], '#323a6c'), ([T(WX0 - cw, WZ1 + cw), T(WX0, WZ1), T(WX0, 0), T(WX0 - cw, 0)], '#262e5c'), ([T(WX1 + cw, WZ1 + cw), T(WX1, WZ1), T(WX1, 0), T(WX1 + cw, 0)], '#3a4378')]: c.fill(np.array(p), col)
        a = c.rgba(); cv = Canvas(h, w, img=a[..., :3]); m = a[..., 3] > .5; n = 2600; from_ = np.argwhere(m[::4, ::4]) * 4; k = r.integers(0, len(from_), n); py, px = from_[k, 0].astype(f32), from_[k, 1].astype(f32); vert = (np.abs((px / M + x0) - WX0 + cw / 2) < cw) | (np.abs((px / M + x0) - WX1 - cw / 2) < cw); _strokes(cv, arcs(px, py, np.where(vert, np.pi / 2, 0.) + r.normal(0, .02, n), r.uniform(60, 300, n), r.normal(0, .03, n), 5), r.uniform(2, 9, n), jit_color(cv.img[py.astype(int), px.astype(int)], r, .012, .03, .09), op=r.uniform(.2, .6, n), dry=r.uniform(.3, .9, n), seed=4)
        ca = Cel(w, h, 0, 0, 1)
        for p in ([T(WX0, WZ1), T(WX1, WZ1)], [T(WX0, WZ1), T(WX0, 0)], [T(WX1, WZ1), T(WX1, 0)]): ca.line(np.array(p), 7., '#ffb766', taper=(.02, .02), tremor=.8, seed=3); ca.line(np.array(p), 2.6, '#fff0c4', taper=(.1, .1), tremor=.8, seed=4)
        for p in ([T(WX0 - cw, WZ1 + cw), T(WX0, WZ1)], [T(WX1 + cw, WZ1 + cw), T(WX1, WZ1)], [T(WX1 + cw, WZ1 + cw), T(WX1 + cw, 0)], [T(WX0 - cw, WZ1 + cw), T(WX1 + cw, WZ1 + cw)]): ca.line(np.array(p), 4., '#1d1a38', taper=(.02, .02), tremor=.6)
        e = ca.rgba(); s.tex['case'] = np.dstack([(cv.img * a[..., 3:4]) * (1 - e[..., 3:4]) + e[..., :3], a[..., 3] + e[..., 3] * (1 - a[..., 3])]).astype(f32)
        # sill: top and front, thick old plank
        Ms = 2400; sw, sd = 1.435, .34; s.sill = (-1.06, sd); tw, th = int(sw * Ms), int(sd * Ms); s.Ms = Ms
        def plank(w_, h_, seed, stops, rows=1):
            g = fbm(h_, w_, (46, 17, 7), (1, .7, .45), seed, aspect=26.); kn = np.zeros((h_, w_), f32); yy, xx = np.mgrid[0:h_, 0:w_].astype(f32); q = np.random.default_rng(seed)
            for _ in range(5): a_, b_ = q.uniform(0, w_), q.uniform(0, h_); d = np.sqrt(((xx - a_) / 90) ** 2 + ((yy - b_) / 16) ** 2); kn += np.exp(-d * d) * np.sin(d * 9)
            return ramp(np.clip(.5 + .17 * g + .25 * kn, 0, 1), stops)
        top = plank(tw, th, 5, [(0, '#161c3c'), (.35, '#28305c'), (.65, '#3e4678'), (1, '#626a9c')]) if False else plank(tw, th, 5, [(0, '#161c3c'), (.35, '#28305c'), (.65, '#3e4678'), (1, '#626a9c')])
        cv = Canvas(th, tw, img=top); n = 2600; px = r.uniform(0, tw, n); py = r.uniform(0, th, n); _strokes(cv, arcs(px, py, r.normal(0, .012, n), r.uniform(60, 620, n), r.normal(0, .03, n), 6), r.uniform(1, 7, n), jit_color(cv.img[py.astype(int), px.astype(int)], r, .014, .03, .10), op=r.uniform(.2, .65, n), dry=r.uniform(.3, .95, n), seed=5)
        for k in range(7): yy_ = r.uniform(.05, .95) * th; xx0 = r.uniform(0, tw * .7); P = np.c_[np.linspace(xx0, xx0 + r.uniform(300, 1100), 14), yy_ + np.cumsum(r.normal(0, 2.2, 14))]; cv.strokes(P[None], 2.4, C('#15102c'), shape='taper', seed=k); cv.strokes(P[None] + [[[0, 3.2]]], 1.4, C('#8d7cae'), op=.6, shape='taper', seed=k + 9)
        s.tex['top'] = cv.img
        fw, fh = tw, int(.062 * Ms); fr_ = plank(fw, fh, 9, [(0, '#10142e'), (.4, '#1c2348'), (.7, '#293260'), (1, '#3f4a7c')]); cv = Canvas(fh, fw, img=fr_); n = 1100; px = r.uniform(0, fw, n); py = r.uniform(0, fh, n); _strokes(cv, arcs(px, py, r.normal(0, .015, n), r.uniform(60, 500, n), 0., 4), r.uniform(1, 6, n), jit_color(cv.img[py.astype(int), px.astype(int)], r, .012, .03, .10), op=r.uniform(.2, .6, n), dry=.6, seed=6); cv.img[:7] = cv.img[:7] * .3 + C('#8f95c4') * .7 * (.6 + .4 * np.clip(fbm(7, fw, (120, 30), (1, .5), 4), -1, 1))[..., None]; cv.img[-int(fh * .16):] *= .62; s.tex['front'] = cv.img
        # eave beam
        bw, bh_ = int(2.6 * 900), int(.10 * 900); b_ = plank(bw, bh_, 12, [(0, '#0f0e22'), (.5, '#1c1a36'), (1, '#2f2c52')]); b_[-5:] = C('#5b5488'); s.tex['beam'] = b_
        s.sky = s.dusk_sky(); s.ok = True; return s

    def dusk_sky(s, h=1240, w=2200):
        hz = 800; img = SK.gradient(h, w, hz, [(-1, '#0e1a3a'), (0, '#ffc070'), (.025, '#f5a45c'), (.06, '#c9808a'), (.13, '#7c6aa4'), (.26, '#43569a'), (.5, '#27407a'), (.8, '#192b5c'), (1.2, '#101d44')], .03, w * .75, 5); src = (w * .82, hz)
        img = SK.torn(img, hz * .50, hz * 1.0, '#3a3f7a', '#8a6c9c', '#ffb070', src, 7, .42, 8., .9, .85, .3); img = SK.torn(img, hz * .05, hz * .6, '#1e2f62', '#3c4f8e', '#7c80b4', src, 17, .30, 5., 1.2, .6, .4); img = SK.brush(img, 3, .9, (36, 15))
        x, y = grid(h, w); sea = y > hz; t = np.clip((y - hz) / (h - hz), 0, 1); sc = ramp(t, [(0, '#8a7aa0'), (.05, '#4a5a94'), (.3, '#26407a'), (1, '#142650')]); n = np.empty((h, w), f32); fa, fb_ = fbm(h, w, (7, 3.4), (1, .6), 4, aspect=11.), fbm(h, w, (22, 9), (1, .6), 5, aspect=9.); n = fa * (1 - t) + fb_ * t; sc = sc + (C('#6f7fb8') - sc) * (smooth(.5, .9, n) * .6)[..., None]; sc = sc + (C('#0f1f48') - sc) * (smooth(.8, 1.2, -n) * .6)[..., None]
        g = np.exp(-((x - w * .80) / (40 + 300 * t)) ** 2); gl = smooth(.40, .62, n * .6 + g * 1.1 - .55) * g; sc = sc + (C('#ffb872') - sc) * (gl * .85)[..., None]; img = np.where(sea[..., None], sc, img)
        c = Cel(w, h, 0, 0, 1)
        c.fill(spl(np.array([(w * .60, hz + 2), (w * .64, hz - 14), (w * .67, hz - 30), (w * .70, hz - 24), (w * .73, hz - 40), (w * .77, hz - 20), (w * .80, hz - 8), (w * .83, hz + 2)]), n=8), '#4a4a84'); c.line(np.array([(w * .60, hz + 1), (w * .83, hz + 1)]), 3, '#ffcf98', taper=(.3, .3))
        for (a, b, R) in [(w * .90, 160, 3.2), (w * .71, 96, 2.2), (w * .80, 250, 1.8), (w * .95, 330, 1.6), (w * .66, 250, 1.5)]: c.fill(np.array([(a, b - R * 2.4), (a + R * .5, b - R * .5), (a + R * 2.4, b), (a + R * .5, b + R * .5), (a, b + R * 2.4), (a - R * .5, b + R * .5), (a - R * 2.4, b), (a - R * .5, b - R * .5)]), '#f2f0ff')
        o = over(img.astype(f32), c.rgba()); s.hz = hz; return o

    # ------------------------------------------------------------------------------------ live parts
    def curtain(s, laugh, lit):
        """the lit window as seen from outside: thin curtain, lamp, the family in silhouette.  texture of the opening, 700 px per metre"""
        k = 700; w, h = int((WX1 - WX0) * k), int(WZ1 * k); y, x = np.mgrid[0:h, 0:w].astype(f32); X = WX0 + x / k; Z = WZ1 - y / k
        d = np.hypot(X + .50, (Z - .60) * 1.15); img = ramp(np.clip(d / 1.05, 0, 1), [(0, '#fff2c4'), (.14, '#ffe19a'), (.40, '#ffc472'), (.72, '#f09a4c'), (1, '#c2662f')])
        fold = np.sin(X * 33 + 2.2 * np.sin(X * 9) + .5 * Z) * .5 + .3 * np.sin(X * 81 + 1.4 * Z); img = img * (1 + .055 * fold[..., None]) ; hem = (fold > .42) & (Z < .30); img = img * (1 - .05 * hem[..., None])
        c = Cel(w, h, 0, 0, 1); T = lambda P: np.c_[(np.asarray(P, float)[:, 0] - WX0) * k, (WZ1 - np.asarray(P, float)[:, 1]) * k]
        lam = T(spl(np.array([(-.64, .64), (-.60, .74), (-.48, .74), (-.44, .64)]), n=6)); c.fill(lam, '#fffbe6'); c.line(lam[[0, -1]], 3, '#c88a44')
        img = over(img, c.rgba())
        for al, col, so in (laugh or []):                                                # back to front: far ones paler and softer
            m = cv2.GaussianBlur(al.astype(f32) / 255. if al.dtype == np.uint8 else al, (0, 0), so); img = img + (img * C(col) - img) * m[..., None]
        # glazing bars
        c = Cel(w, h, 0, 0, 1)
        for p in ([(-.42, 0), (-.42, WZ1)], [(WX0, .675), (WX1, .675)]): P = T(p); c.line(P, 30, '#2a1c2c', taper=(.001, .001), tremor=.5, press=0.); c.line(P + (np.array([-11, 0]) if p[0][0] == p[1][0] else np.array([0, -11])), 4, '#ffcf8a', taper=(.01, .01)); c.line(P + (np.array([10, 0]) if p[0][0] == p[1][0] else np.array([0, 10])), 5, '#160e1c', taper=(.01, .01))
        img = over(img, c.rgba()); L = .10 + .90 * lit; lu = img.mean(-1, keepdims=True); dim = lu * np.array([.22, .17, .44], f32) * .85; return np.clip(img * L * lit ** .5 + dim * (1 - lit), 0, 1.5).astype(f32)

    def chime(s, c, cam, ang, strip, lit, ring=0.):
        """glass wind-bell on its string: ang = swing of the bell, strip = swing of the paper (radians)"""
        top = np.array([.385, -.50, .80]); px = cam.project(top)[0]; k = F / (cam.cam(top)[2]); L = .49 * k
        def R(P, a, o): ca, sa = math.cos(a), math.sin(a); P = np.asarray(P, float); return np.c_[o[0] + P[:, 0] * ca + P[:, 1] * sa * -1, o[1] + P[:, 0] * sa + P[:, 1] * ca]
        bell = px + np.array([-math.sin(ang) * L, math.cos(ang) * L]); c.line(np.array([px, (px + bell) / 2 + [1.5, 0], bell]), .0042 * k, '#d8c4d8', taper=(.02, .02))
        b = .052 * k; prof = spl(np.array([(-.16, 0), (-.55, .10), (-.92, .48), (-1.0, .98), (-.80, 1.42), (-.66, 1.55), (.66, 1.55), (.80, 1.42), (1.0, .98), (.92, .48), (.55, .10), (.16, 0)]), n=8) * b; P = R(prof, ang, bell); c.fill(P, '#7fa8d8', .34); c.fill(R(spl(np.array([(-1., .95), (-.8, 1.42), (-.66, 1.55), (.66, 1.55), (.8, 1.42), (1., .95), (0, 1.22)]), n=6) * b, ang, bell), '#3d5aa8', .45)
        for (u0, v0, sc, fl) in [(-.30, .72, 1., 1), (.36, .98, .8, -1)]:                                     # two goldfish painted on the glass
            f = np.array([(0, 0), (.14, -.09), (.30, -.02), (.46, -.14), (.44, .02), (.48, .16), (.30, .06), (.14, .10)]) * sc * [fl, 1] + [u0, v0]; c.fill(R(spl(f, closed=True, corners=(3, 5), n=5) * b, ang, bell), '#e8483c', .95)
        c.fill(R(spl(np.array([(-.80, .42), (-.62, .20), (-.52, .24), (-.72, .60), (-.78, .86), (-.86, .80)]), closed=True, n=5) * b, ang, bell), '#f4f8ff', .85); c.fill(R(np.array([(.62, .44), (.74, .62), (.70, .70), (.58, .52)]) * b, ang, bell), '#ffd9a0', .9 * lit + .1)
        c.line(P, .0030 * k, '#cfe0ff', taper=(.05, .05)); c.line(R(np.array([(-.66, 1.55), (0, 1.62), (.66, 1.55)]) * b, ang, bell), .0034 * k, '#eaf2ff', taper=(.1, .1)); c.fill(R(np.array([(-.18, -.10), (.18, -.10), (.16, .06), (-.16, .06)]) * b, ang, bell), '#c9a25a')
        a2 = ang * .6 + strip * .4; m0 = R(np.array([(0, .2)]) * b, ang, bell)[0]; clap = m0 + np.array([-math.sin(a2), math.cos(a2)]) * 1.05 * b; end = clap + np.array([-math.sin(strip), math.cos(strip)]) * 1.0 * b; c.line(np.array([m0, clap, end]), .0030 * k, '#b79ab0', taper=(.02, .02)); c.fill(np.array([clap + [math.cos(t) * b * .15, math.sin(t) * b * .15] for t in np.linspace(0, 6.28, 14)]), '#e7d9b0')
        st = np.array([(-.30, 0), (.30, 0), (.33, 2.5), (-.27, 2.5)]) * b; bend = np.array([[0, 0], [0, 0], [strip * -.9 * b, 0], [strip * -.9 * b, 0]]); Q = R(st, strip, end) + bend; c.fill(Q, '#e9dcd4'); c.fill(np.array([Q[0], (Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2, Q[3]]), '#b9acc8'); c.line(np.vstack([Q, Q[:1]]), .0022 * k, '#4b4470', taper=(.02, .02)); mid = (Q[0] + Q[1]) / 2; mid2 = (Q[2] + Q[3]) / 2
        for q in (.2, .36, .52, .68): a_ = mid + (mid2 - mid) * q; c.line(np.array([a_ + [-.08 * b, -.05 * b], a_ + [.09 * b, .04 * b]]), .003 * k, '#c8323a', taper=(.2, .3))
        if ring > 0:                                                                                           # the note itself: two thin arcs
            for q in (1.35, 1.75): th = np.linspace(-.6, .6, 10); cc = bell + [0, b]; c.line(np.c_[cc[0] + np.cos(th) * b * q, cc[1] + np.sin(th) * b * q], .0026 * k, '#eef4ff', taper=(.4, .4), alpha=ring); c.line(np.c_[cc[0] - np.cos(th) * b * q, cc[1] + np.sin(th) * b * q], .0026 * k, '#eef4ff', taper=(.4, .4), alpha=ring)

    def ball(s, u, dx=0., dz=0.):
        cam = cam_at(u, dx, dz); p, z = cam.project(BALL); return float(p[0]), float(p[1]), RB * F / float(z)

    # ------------------------------------------------------------------------------------ picture
    def frame(s, u=0., lit=1., chime=(0., 0.), laugh=None, glow=1., t=0, ring=0., dx=0., dz=0., ballfn=True, stars=None, dim=0., wind=0., rim=0.):
        s.setup(); cam = cam_at(u, dx, dz); e1, e2 = np.array([1., 0, 0]), np.array([0, 0, -1.])
        # sky, far away: only a breath of movement
        z = 1.02 + .10 * u; h, w = s.sky.shape[:2]; cx, cy = w * .50 + 120 + cam.eye[0] * 60, h * .53 - 60 * u; img = cv2.warpAffine(s.sky, np.float32([[1 / z, 0, cx - W / 2 / z], [0, 1 / z, cy - H / 2 / z]]), (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        if dim > 0: img = img * (1 - dim * (1 - np.array([.10, .09, .20], f32)))
        # window light
        cu = s.curtain(laugh, lit); img = over(img, cam.warp_plane(cu, (WX0, .05, WZ1), e1, e2, WX1 - WX0, WZ1))
        # reveals (the thickness of the wall, lit)
        c = Cel(); pj = lambda P: cam.project(np.asarray(P, float))[0]
        rc = mix(C('#2a2350'), C('#f0a050'), .9 * lit); rc2 = mix(C('#1d1840'), C('#b8662e'), .9 * lit)
        c.fill(pj([(WX0, 0, 0), (WX0, .05, 0), (WX0, .05, WZ1), (WX0, 0, WZ1)]), rc); c.fill(pj([(WX1, 0, 0), (WX1, .05, 0), (WX1, .05, WZ1), (WX1, 0, WZ1)]), rc2); c.fill(pj([(WX0, 0, WZ1), (WX0, .05, WZ1), (WX1, .05, WZ1), (WX1, 0, WZ1)]), rc2); img = over(img, c.rgba())
        x0, x1, z0, z1 = s.wall; amb = 1 - .62 * dim; wt = s.tex['wall'] * (.86 + .14 * lit) * amb + s.tex['wallw'] * lit * (.35 + .65 * s.tex['wall']); wall = np.dstack([wt * s.tex['walla'][..., None], s.tex['walla']]).astype(f32); img = over(img, cam.warp_plane(wall, (x0, 0, z1), e1, e2, x1 - x0, z1 - z0))
        c = Cel(); q = pj([(CORNER, 0, z0), (CORNER, 0, z1)]); q2 = pj([(CORNER, .4, z0), (CORNER, .4, z1)]); c.fill(np.vstack([q + [[-10, 0], [-10, 0]], q[::-1]]), '#6f6aa6'); c.line(q, 3, '#a9a2d4', taper=(.1, .1)); img = over(img, c.rgba())
        cs = s.tex['case'].copy(); cs[..., :3] *= (.34 + .66 * lit) * amb; img = over(img, cam.warp_plane(cs, (x0, -.022, z1), e1, e2, x1 - x0, z1 - z0))
        # shadow of the sill on the wall
        c = Cel(); c.fill(pj([(-1.06, 0, -.062), (.375, 0, -.062), (.335, 0, -.20), (-1.06, 0, -.24)]), '#0c0a22', .62); img = over(img, cv2.GaussianBlur(c.rgba(), (0, 0), 2.2))
        # sill
        sx, sd = s.sill; sw = s.tex['top'].shape[1] / s.Ms; th, tw = s.tex['top'].shape[:2]; tx = sx + np.arange(tw, dtype=f32) / s.Ms; ty = np.arange(th, dtype=f32) / s.Ms                      # v = distance out from the wall
        win = np.exp(-ty / .13)[:, None] * smooth(WX0 - .05, WX0 + .12, tx)[None] * (1 - smooth(WX1 - .12, WX1 + .05, tx))[None]; bars = 1 - .55 * np.exp(-((tx[None] + .42 + (0.) * ty[:, None]) / (.012 + ty[:, None] * .08)) ** 2); top = s.tex['top'] * (.80 + .2 * lit) * amb; top = top + C('#ffb05a') * (win * bars * 1.05 * lit)[..., None] * (.35 + .9 * s.tex['top']); d = np.hypot(tx[None] - BALL[0], (ty[:, None] - .02 + BALL[1]) * 1.0)
        gl = np.exp(-(d / .085) ** 2) * 1.5 + np.exp(-d / .16) * .55; top = top + C('#ff9a2e') * (gl * glow)[..., None] * (.30 + .9 * s.tex['top']); con = np.exp(-(np.hypot(tx[None] - BALL[0] - .004, ty[:, None] + BALL[1] - .012) / .030) ** 2); top = top * (1 - .85 * con * (1 - .45 * glow))[..., None]
        if lit > .05: shd = np.exp(-(np.abs(tx[None] - BALL[0]) / (.046 + .10 * np.clip(ty[:, None] + BALL[1], 0, 1))) ** 4) * (ty[:, None] > -BALL[1] + .01) * np.exp(-np.clip(ty[:, None] + BALL[1], 0, 1) / .10); top = top * (1 - .6 * shd * lit * (1 - .55 * glow))[..., None]
        tl = np.dstack([top, np.ones((th, tw), f32)]).astype(f32); img = over(img, cam.warp_plane(tl, (sx, 0, 0), e1, np.array([0, -1., 0]), sw, th / s.Ms))
        fr = s.tex['front'] * amb; img = over(img, cam.warp_plane(np.dstack([fr, np.ones(fr.shape[:2], f32)]).astype(f32), (sx, -sd, 0), e1, e2, sw, fr.shape[0] / s.Ms))
        c = Cel(); s.props(c, cam, pj, lit, wind, amb); q = pj([(sx + sw, -sd, 0), (sx + sw, -sd, -.062)]); c.line(q, 3., C('#0f1028') * amb, taper=(.1, .1)); img = over(img, c.rgba())
        # eave beam and the bell
        bm = s.tex['beam'] * amb; img = over(img, cam.warp_plane(np.dstack([bm, np.ones(bm.shape[:2], f32)]).astype(f32), (-1.6, -.5, .86), e1, e2, bm.shape[1] / 900., bm.shape[0] / 900.))
        c = Cel(); s.chime(c, cam, chime[0], chime[1], lit, ring); img = over(img, c.rgba())
        # light in the air
        x, y = grid(); wp = pj([(-.42, .05, .55)])[0]; k = F / cam.cam(np.array([0, .05, 0]))[2]; g = np.exp(-np.hypot((x - wp[0]) / (.75 * k), (y - wp[1]) / (.55 * k)) * 1.6)[..., None]; img = img + C('#ff9a48') * g * .20 * lit
        bx, by, br = s.ball(u, dx, dz); g = np.exp(-np.hypot(x - bx, y - by) / (br * 2.2))[..., None]; img = img + C('#ff8a1e') * g * .38 * glow
        if ballfn:
            from . import ball as BL
            img = BL.draw(np.clip(img, 0, 1.3).astype(f32), bx, by, br, stars=(glow,) * 4 if stars is None else stars, spec=float(np.clip((lit - .30) / .40, 0, 1) if lit > .30 else .34 * np.clip((.30 - lit) / .20, 0, 1)), sun=tuple(mix(C('#9a7cf0'), C('#fff0c0'), float(max(glow, lit)))), glass=.30 * float(np.clip(.25 + .75 * max(glow, lit * .6), 0, 1)), body=1.15 * glow, t=t, light=(-.35, .35, -.6) if lit > .3 else (.5, .5, .4), shade=tuple(np.clip(.25 + .75 * max(glow, lit * .55), 0, 1) * np.array([1., .93, .95])) )
        if rim > 0:                                                             # the first violet light finds the edge of the glass
            x, y = grid(); q = np.hypot(x - bx, y - by); a_ = np.arctan2(y - by, x - bx); e_ = np.clip(1 - np.abs(q - br + 1.6) / 2.4, 0, 1) * (.35 + .65 * smooth(-.2, .9, np.cos(a_ + 1.1))); img = img + C('#a06aff') * (e_ * rim * 1.1 + np.exp(-np.abs(q - br) / 16.) * .20 * rim)[..., None]
        return img.astype(f32)

    def props(s, c, cam, pj, lit, wind=0., amb=1.):
        """little things left on the sill: a pot with a succulent, a conch, a folded red cap string"""
        k = F / cam.cam(np.array([0, -.15, 0]))[2]; o = pj([(-.80, -.14, 0)])[0]; w = lambda P: np.asarray(P, float) * k * [1, -1] + o
        lc = mix(C('#5a4a78') * amb, C('#c98058'), .6 * lit); dk = mix(C('#2a2348') * amb, C('#5a3242'), .6 * lit)
        c.fill(w([(-.062, .085), (.062, .085), (.048, 0), (-.048, 0)]), lc); c.fill(w([(.004, .085), (.062, .085), (.048, 0), (.0, 0)]), dk); c.fill(w([(-.070, .105), (.070, .105), (.066, .082), (-.066, .082)]), mix(lc, C('#ffd9a0'), .25)); c.fill(w([(.01, .105), (.070, .105), (.066, .082), (.006, .082)]), dk * 1.1); c.line(w([(-.062, .085), (-.048, 0), (.048, 0), (.062, .085)]), .003 * k, '#120e26')
        r = np.random.default_rng(4); lf = sorted([(a, r.uniform(.10, .19)) for a in np.linspace(-1.25, 1.25, 13)], key=lambda v: -abs(v[0]) + r.normal(0, .3))
        for a, L in lf:
            a += r.normal(0, .08) + wind * (.4 + L * 3.); b = np.array([a * .022, .10]); tip = b + np.array([math.sin(a), math.cos(a)]) * L + [0, -.02 * abs(a)]; m = (b + tip) / 2 + np.array([math.cos(a), -math.sin(a)]) * .012; n = np.array([math.cos(a), -math.sin(a)]) * .017; P = np.vstack([spl(np.array([b - n * .6, m - n, tip]), n=6), spl(np.array([tip, m + n, b + n * .6]), n=6)]); c.fill(w(P), mix(C('#24406a') * amb, C('#5f7a52'), .5 * lit)); c.fill(w(np.vstack([spl(np.array([b, m, tip]), n=6), spl(np.array([tip, m + n, b + n * .6]), n=6)])), mix(C('#182a52') * amb, C('#2c3c3c'), .5 * lit)); c.line(w(P[:12]), .0028 * k, '#0e1230', taper=(.1, .3)); c.line(w(spl(np.array([b - n * .5, m - n * .9, tip]), n=6)), .0024 * k, mix(C('#8a9ac8') * amb, C('#ffd98a'), lit), taper=(.3, .4))
