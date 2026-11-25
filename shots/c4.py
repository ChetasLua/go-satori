"""c4 LIGHT BEFORE SOUND.  Level horizon at 792 (c1's sea).  0-16 a hair line is drawn along the far ridge; 16-30 nothing; 30 the ridge comes up off the line, unzips, turns to light;
40 the pressure front races over the sea; 52 Piccolo throws himself round Gohan; 60 it arrives: the first sound.  White, two silhouettes, the cape goes to ribbons; 72 white thins to violet smoke."""
import numpy as np, cv2, math, os
from db import W, H, comp, kit, moon as M
from db.paint import rgb, f32, smooth
from db.ink import Cel
from db.phys import *
from db.phys_kit import ring_flash

HY, CX = 792, 905
keys = comp.lerp_keys
HUG = (932., 628., 162.)                                            # closer camera from 55: the child's head centre, px per Piccolo head unit
ZOOM = (905., 640., 1.55)                                            # what the closer camera sees of the wide picture
GOH = (-.30, .25, .30)                                              # Gohan's feet, figure-camera metres
KEYS = {55: dict(close=0., tuck=0., spread=.55, swing=1., flinch=.3, dx=-70.), 56: dict(close=.45, tuck=.40, spread=.82, swing=.55, flinch=.6, dx=-22.), 57: dict(close=.85, tuck=.82, spread=1.04, swing=.15, flinch=.9, dx=0.), 58: dict(close=1., tuck=1.05, spread=1.07, swing=0., flinch=1., dy=13.), 59: dict(close=1., tuck=1., spread=1.0, swing=-.06, flinch=1., dy=5.)}
_hc = {}


def hug(f, rip=0., turban=True, press=0., jit=(0., 0.), seed=2, **k):
    """the wrap drawing (cached per process: drawings are held on twos)"""
    from shots.c4_fig import hug as H_
    kw = dict(KEYS[min(f, 59)]) if f < 60 else dict(close=1., tuck=1., flinch=1.); kw.update(k); dx, dy = kw.pop('dx', 0.), kw.pop('dy', 0.); key = (f, round(rip, 3), turban, round(press, 3), seed, tuple(sorted(kw.items())), dx, dy, tuple(np.round(jit, 2)))
    if key not in _hc:
        if len(_hc) > 3: _hc.clear()
        _hc[key] = H_(HUG[0] + dx + jit[0], HUG[1] + dy + jit[1], HUG[2], rip=rip, turban=turban, press=press, f=f, seed=seed, **kw).astype(f32)
    return _hc[key]


def over4(a, b): return a * (1 - b[..., 3:4]) + b


def tail(L, dx, dy, n=26):
    """speed smear that only drags behind the drawing"""
    o = np.zeros_like(L)
    for j in range(1, n + 1): t = j / n; o += cv2.warpAffine(L, np.float32([[1, 0, dx * t], [0, 1, dy * t]]), (W, H)) * (1 - t)
    return o / (n * .5)


def backlit(L, lit, d):
    """light from behind eats a hard edge into the cel; moonlit (lit=0) only a thin violet one on the moon's side"""
    a = L[..., 3]; n = 5 if lit else 3; er = cv2.erode(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))); k = np.clip(a - er, 0, 1) if lit else np.clip(a - cv2.warpAffine(a, np.float32([[1, 0, -d[0] * 3.5], [0, 1, -d[1] * 3.5]]), (W, H)), 0, 1); k = cv2.GaussianBlur(k, (0, 0), .6)[..., None] * (.95 if lit else .75); o = L.copy(); o[..., :3] = o[..., :3] * (1 - k) + C('#fbf7ff' if lit else '#7a4fe0') * k * a[..., None]; return o


def back_hair(c, x, y, s, roll, seed, wind=0.):
    """a small boy's mop from behind: crown whorl, tufts all round, ragged nape"""
    R2 = np.array([[math.cos(roll), -math.sin(roll)], [math.sin(roll), math.cos(roll)]]); P = lambda q: np.array([x, y - .02 * s]) + (np.asarray(q, float) @ R2.T) * s; ink, base, pl, hi = '#05030a', '#0d0a18', '#1c1634', '#3a2f66'
    T = [(-.16, 1.16), (-.58, 1.10), (-.98, .84), (-1.20, .40), (-1.26, -.14), (-1.16, -.70), (-.80, -1.16), (-.26, -1.40), (.26, -1.36), (.78, -1.18), (1.16, -.68), (1.28, -.12), (1.20, .42), (.96, .86), (.54, 1.12), (.16, 1.18)]
    th = np.linspace(0, TAU, 40); base_ = P(np.c_[np.cos(th) * .90, np.sin(th) * .98 - .06]); locks = []
    for i, t in enumerate(T):
        a = math.atan2(t[1], t[0]); w = .26 + .06 * h1(seed, i); r0 = .80; A, B_ = (math.cos(a - w) * r0, math.sin(a - w) * (r0 + .06)), (math.cos(a + w) * r0, math.sin(a + w) * (r0 + .06)); tip = (t[0] * (1 + .10 * (h1(seed, i, 2) - .5)) + wind, t[1] * (1 + .08 * (h1(seed, i, 3) - .5)) - abs(wind) * .3); locks.append((kit.spike(P([A])[0], P([tip])[0], P([B_])[0], .14, .10 * (1 if i % 2 else -1)), a))
    for L, _ in locks: c.fill(L + (L - L.mean(0)) * .035, ink)
    c.fill(P(np.c_[np.cos(th) * .925, np.sin(th) * 1.005 - .06]), ink); c.fill(base_, base)
    for L, a in locks: c.fill(L, base)
    for i in range(7):                                       # strands fall from the crown to the nape, never a pinwheel
        u = (i / 6. - .5) * 2 + .10 * (h1(seed, i, 7) - .5); q = spl(P(np.array([(u * .22, -.62 + .10 * abs(u)), (u * .58 + .05, -.18), (u * .80, .38 + .40 * (1 - abs(u)) * h1(seed, i, 9) + .12)])), n=8); thin(c, q, s * (.04 + .10 * h1(seed, i, 8) ** 2), pl if i % 3 else hi, 1., .35, .7, 18)
    for i, (x0, y0) in enumerate(((-.72, -.80), (.10, -1.05), (.62, -.86), (-.95, -.10), (.98, -.04))): thin(c, P(np.array([(x0 * .80, y0 * .84), (x0 * 1.08, y0 * 1.06)])), s * .07, hi, 1., .3, .7, 10)


def tail(L, dx, dy, n=26):
    """speed smear that only drags behind the drawing"""
    o = np.zeros_like(L)
    for j in range(1, n + 1): t = j / n; o += cv2.warpAffine(L, np.float32([[1, 0, dx * t], [0, 1, dy * t]]), (W, H)) * (1 - t)
    return o / (n * .5)


def backlit(L, lit, d):
    """light from behind eats a hard edge into the cel; moonlit (lit=0) only a thin violet one on the moon's side"""
    a = L[..., 3]; n = 5 if lit else 3; er = cv2.erode(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))); k = np.clip(a - er, 0, 1) if lit else np.clip(a - cv2.warpAffine(a, np.float32([[1, 0, -d[0] * 3.5], [0, 1, -d[1] * 3.5]]), (W, H)), 0, 1); k = cv2.GaussianBlur(k, (0, 0), .6)[..., None] * (.95 if lit else .75); o = L.copy(); o[..., :3] = o[..., :3] * (1 - k) + C('#fbf7ff' if lit else '#7a4fe0') * k * a[..., None]; return o


def back_hair(c, x, y, s, roll, seed, wind=0.):
    """a small boy's mop from behind: crown whorl, tufts all round, ragged nape"""
    R2 = np.array([[math.cos(roll), -math.sin(roll)], [math.sin(roll), math.cos(roll)]]); P = lambda q: np.array([x, y - .02 * s]) + (np.asarray(q, float) @ R2.T) * s; ink, base, pl, hi = '#05030a', '#0d0a18', '#1c1634', '#3a2f66'
    T = [(-.16, 1.16), (-.58, 1.10), (-.98, .84), (-1.20, .40), (-1.26, -.14), (-1.16, -.70), (-.80, -1.16), (-.26, -1.40), (.26, -1.36), (.78, -1.18), (1.16, -.68), (1.28, -.12), (1.20, .42), (.96, .86), (.54, 1.12), (.16, 1.18)]
    th = np.linspace(0, TAU, 40); base_ = P(np.c_[np.cos(th) * .90, np.sin(th) * .98 - .06]); locks = []
    for i, t in enumerate(T):
        a = math.atan2(t[1], t[0]); w = .26 + .06 * h1(seed, i); r0 = .80; A, B_ = (math.cos(a - w) * r0, math.sin(a - w) * (r0 + .06)), (math.cos(a + w) * r0, math.sin(a + w) * (r0 + .06)); tip = (t[0] * (1 + .10 * (h1(seed, i, 2) - .5)) + wind, t[1] * (1 + .08 * (h1(seed, i, 3) - .5)) - abs(wind) * .3); locks.append((kit.spike(P([A])[0], P([tip])[0], P([B_])[0], .14, .10 * (1 if i % 2 else -1)), a))
    for L, _ in locks: c.fill(L + (L - L.mean(0)) * .035, ink)
    c.fill(P(np.c_[np.cos(th) * .925, np.sin(th) * 1.005 - .06]), ink); c.fill(base_, base)
    for L, a in locks: c.fill(L, base)
    for i in range(8):                                       # a few planes over the crown, two at the nape; never a pinwheel
        a = -math.pi + math.pi * (i + .5 + .7 * (h1(seed, i, 7) - .5)) / 6 if i < 6 else (.9 if i == 6 else 2.2); r0, r1 = .30 + .25 * h1(seed, i, 9), .86 + .12 * h1(seed, i, 10); p0 = P([(math.cos(a - .18) * r0, -.06 + math.sin(a - .18) * r0)])[0]; p2 = P([(math.cos(a + .10) * r1, -.06 + math.sin(a + .10) * r1)])[0]; thin(c, np.array([p0, (p0 + p2) / 2 + [s * .04, -s * .03], p2]), s * (.05 + .15 * h1(seed, i, 8) ** 2), pl if i % 3 else hi, 1., .5, .6, 16)


def tail(L, dx, dy, n=26):
    """speed smear that only drags behind the drawing"""
    o = np.zeros_like(L)
    for j in range(1, n + 1): t = j / n; o += cv2.warpAffine(L, np.float32([[1, 0, dx * t], [0, 1, dy * t]]), (W, H)) * (1 - t)
    return o / (n * .5)


def backlit(L, lit, d):
    """light from behind eats a hard edge into the cel; moonlit (lit=0) only a thin violet one on the moon's side"""
    a = L[..., 3]; n = 5 if lit else 3; er = cv2.erode(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))); k = np.clip(a - er, 0, 1) if lit else np.clip(a - cv2.warpAffine(a, np.float32([[1, 0, -d[0] * 3.5], [0, 1, -d[1] * 3.5]]), (W, H)), 0, 1); k = cv2.GaussianBlur(k, (0, 0), .6)[..., None] * (.95 if lit else .75); o = L.copy(); o[..., :3] = o[..., :3] * (1 - k) + C('#fbf7ff' if lit else '#7a4fe0') * k * a[..., None]; return o


def back_hair(c, x, y, s, roll, seed, wind=0.):
    """a small boy's mop from behind: crown whorl, tufts all round, ragged nape"""
    R2 = np.array([[math.cos(roll), -math.sin(roll)], [math.sin(roll), math.cos(roll)]]); P = lambda q: np.array([x, y - .02 * s]) + (np.asarray(q, float) @ R2.T) * s; ink, base, pl, hi = '#05030a', '#0d0a18', '#1c1634', '#3a2f66'
    T = [(-.16, 1.16), (-.58, 1.10), (-.98, .84), (-1.20, .40), (-1.26, -.14), (-1.16, -.70), (-.80, -1.16), (-.26, -1.40), (.26, -1.36), (.78, -1.18), (1.16, -.68), (1.28, -.12), (1.20, .42), (.96, .86), (.54, 1.12), (.16, 1.18)]
    th = np.linspace(0, TAU, 40); base_ = P(np.c_[np.cos(th) * .90, np.sin(th) * .98 - .06]); locks = []
    for i, t in enumerate(T):
        a = math.atan2(t[1], t[0]); w = .26 + .06 * h1(seed, i); r0 = .80; A, B_ = (math.cos(a - w) * r0, math.sin(a - w) * (r0 + .06)), (math.cos(a + w) * r0, math.sin(a + w) * (r0 + .06)); tip = (t[0] * (1 + .10 * (h1(seed, i, 2) - .5)) + wind, t[1] * (1 + .08 * (h1(seed, i, 3) - .5)) - abs(wind) * .3); locks.append((kit.spike(P([A])[0], P([tip])[0], P([B_])[0], .14, .10 * (1 if i % 2 else -1)), a))
    for L, _ in locks: c.fill(L + (L - L.mean(0)) * .035, ink)
    c.fill(P(np.c_[np.cos(th) * .925, np.sin(th) * 1.005 - .06]), ink); c.fill(base_, base)
    for L, a in locks: c.fill(L, base)
    for i in range(9):                                       # a few planes fall from the whorl
        a = i / 9 * TAU + .8 * h1(seed, i, 7); p0 = P([(math.cos(a) * .16, -.34 + math.sin(a) * .16)])[0]; p1 = P([(math.cos(a + .25) * .62, -.14 + math.sin(a + .25) * .74)])[0]; p2 = P([(math.cos(a + .34) * .92, -.06 + math.sin(a + .34) * 1.0)])[0]; thin(c, spl(np.array([p0, p1, p2]), n=8), s * (.07 + .16 * h1(seed, i, 8) ** 2), pl if i % 4 else hi, 1., .5, .6, 16); thin(c, spl(np.array([p0 + [2, 1], p1 + [3, 1], p2]), n=8), 1.5, ink, 1., .3, .5, 14)


class Shot:
    id = 'c4'; n = 80
    finish = dict(bloom=.28, vig=.16, grain=.011, soft=.12)
    EVENTS = [(0, 'silence', dict(frames=60, note='no sound at all from the blast: light arrives before sound; only a held breath of sea air until 30')),
              (0, 'line_draw', dict(frames=16, silent=True, note='hair line drawn left to right')), (30, 'light', dict(silent=True, note='the ridge lifts; everything goes white-violet, still silent')),
              (37, 'condensation', dict(silent=True)), (40, 'front', dict(frames=20, silent=True, note='pressure ring races over the sea, 1.3 km in 20 frames')),
              (46, 'scarf', dict(size=.2, note='cloth of the child starts to move')), (52, 'throw', dict(size=.5, note='Piccolo lunges in from the left: cloth, boots on rock')), (55, 'cut_close', dict(note='hard cut to a closer lens, same axis; cape swings round (55), arms close (56-57)')), (57, 'grab', dict(size=.6, note='hands land on cloth and hair, head tucks; cape snaps open on 58')),
              (60, 'blast', dict(size=1.6, note='THE FIRST SOUND: whole orchestra, front arrives, impact frame')), (61, 'white', dict(frames=11, note='roar, high wind, ears ringing')),
              (62, 'rip', dict(size=.6)), (63, 'turban', dict(size=.3, note='turban torn off')), (64, 'rip', dict(size=.7)), (67, 'rip', dict(size=.6)), (70, 'rip', dict(size=.5)), (72, 'smoke', dict(frames=8, note='roar falls away to rolling smoke'))]

    # ------------------------------------------------------------ setup
    def setup(self):
        self.cam = stage(hy=HY, gx=960, gy=1080, ppm=33, focal=1150); self.sea = Sea(waves(.30, 14, 8, (.12, -1), 8, 1.7, .5)); self.c = (float(0.), 1500.)
        z = ZOOM[2]; self.zcam = Camera3D(self.cam.eye, self.cam.eye + np.array([0, 10., 0]), 1150 * z, cx=960 + (960 - ZOOM[0]) * z, cy=540 + (HY - ZOOM[1]) * z); self.fcam = stage(hy=HY, gx=960, gy=1330, ppm=420, focal=1150)
        self.rg = Ridge(70, 1856, HY + 3, [(190, 66, 170), (360, 112, 220), (560, 92, 190), (760, 150, 230), (CX + 10, 226, 300), (1090, 158, 220), (1270, 112, 230), (1450, 164, 240), (1640, 88, 200), (1780, 50, 130)], 2, 240, col=('#33285e', '#150f2b', '#090613'), moon=(.6, -.5))
        y = np.arange(H, dtype=f32)[:, None]; sky = np.broadcast_to(ramp(y / HY, [(0, '#05040a'), (.55, '#08061a'), (.88, '#140d2a'), (1, '#241641')]), (H, W, 3)).astype(f32).copy()
        self.mo = (1600, 372, 84); m, d = M.disc(*self.mo); g = M.halo_rgb(M.halo(np.maximum(d, 0), self.mo[2] * 1.25) * .85); self.halo = np.exp(-np.maximum(d, 0) / 190.).astype(f32); sky = sky + g * (np.arange(H)[:, None, None] < HY); self.sky = (sky * (1 - m[..., None]) + rgb('#fbf7ff') * m[..., None]).astype(f32); self.md = m
        M.sparks(self.sky, 0, 3, avoid=(1600, 372, 150), born=1., amt=.55); self.sky[HY:] = rgb('#0a0716'); self.sky = self.sky.astype(np.float16)
        self.ledge()
        self.figs()
        ox, oy, u = HUG[0] - 3, HUG[1] + .12 * HUG[2], HUG[2]; roots = []
        for q in (-1, 1):
            for j in range(11): v = (j + .4) / 11 * .66; roots.append((ox + q * (5.2 - 3.4 * v ** 1.15) * u * .93, oy + (.15 + 4.9 * v ** .92) * u * .95, q))
        dr = np.array([[q * .95, .30] for _, _, q in roots]); P = np.array([r[:2] for r in roots]); L = [i for i, r in enumerate(roots) if r[2] < 0]; R_ = [i for i, r in enumerate(roots) if r[2] > 0]
        self.ribs = [Ribbons(P[ix], dr[ix], 360, 52, 5 + k).bake(34, wind=lambda f, w=w: (w[0] * (1 + .25 * math.sin(f * 1.1)), w[1] * (1 + .5 * math.sin(f * .7 + 1))), release=lambda i: 2 + (i * 7) % 11 * 1.25, drag=9., gust=.75) for k, (ix, w) in enumerate(((L, (-2500., -620.)), (R_, (2500., -760.))))]
        self.ribs += [Ribbons(P[ix] * [.96, 1.] + [0, 30], dr[ix] * [1, .4], 260, 30, 15 + k).bake(34, wind=lambda f, w=w: (w[0], w[1]), release=lambda i: 7 + (i * 5) % 11 * 1.1, drag=10., gust=.9) for k, (ix, w) in enumerate(((L[::2], (-2900., -300.)), (R_[::2], (2900., -500.))))]
        hx, hy_ = HUG[0] - 8, HUG[1] - 2.15 * u; self.tur = Ribbons(np.array([[hx, hy_], [hx + 16, hy_ + 12], [hx - 14, hy_ + 22]]), (.9, -.3), 520, 46, 9).bake(34, wind=lambda f: (2100., -900.), release=lambda i: 2 + i * 1.5, drag=9., gust=.8)

    def ledge(self):
        """the rock they stand on: dark angular shelf across the foot of the frame, with and without the light of the blast"""
        r = R(4, 4); X = [-40]; Y = [1004.]
        while X[-1] < W + 60: X.append(X[-1] + r.uniform(60, 190)); Y.append(r.uniform(966, 1012) + 20 * math.sin(X[-1] / 260.) + (22 if X[-1] > 1250 else 0) + (X[-1] > 1500) * 26)
        P = np.c_[X, Y]; self.led = []
        for lit in (0, 1):
            c = Cel(); B_ = np.vstack([P, [[W + 80, H + 20], [-60, H + 20]]]); c.fill(B_, '#0a0714' if not lit else '#1a1034')
            c.clip(B_)
            for i in range(len(P) - 1):                              # front planes: facets hang from the lip
                a, b = P[i], P[i + 1]; d = 26 + 60 * h1(i, 3); m = (a + b) / 2; c.fill(np.array([a, b, b + [-18, d * .8], m + [0, d * 1.5], a + [14, d]]), ('#120c22' if i % 2 else '#0d0919') if not lit else ('#2c1b58' if i % 2 else '#22144a')); wedge(c, m + [0, d * 1.5], m + [(h1(i, 5) - .5) * 60, d * 1.5 + 70], 3, .5, '#05030a')
            c.unclip(); rm = '#3a2a66' if not lit else '#fbf7ff'
            for a, b in zip(P[:-1], P[1:]): wedge(c, a + [0, 1.5], b + [0, 1.5], 2.2 + 3.4 * lit, 1. + 1.6 * lit, rm)
            for i in range(26): x = r.uniform(0, W); yy = np.interp(x, P[:, 0], P[:, 1]); s = r.uniform(3, 12); c.fill(np.array([[x - s, yy + 2], [x - s * .5, yy - s * .8], [x + s * .4, yy - s], [x + s, yy + 2]]), '#0a0714' if not lit else '#1a1034'); wedge(c, (x - s * .5, yy - s * .8), (x + s * .4, yy - s), 1.6 + 1.2 * lit, 1, rm)
            self.led.append(c.layer().astype(np.float16))
        self.ly = P

    def figs(self):
        """lunge keys and the child's drawings: still, then 44 46 48 50 52 54 on twos (shoulders go up, he rocks back, arms come up)"""
        from db.hero3 import Fig, POSES, mix_pose
        fn = 'cache/c4_figs_v8.npz'
        if os.path.exists(fn): z = np.load(fn); self.G = [z['g%d' % i] for i in range(8)]; self.Pl = [z['p%d' % i] for i in range(3)]; return
        G = []; rim0 = dict(dir=(.75, .5, -.6), col='#8a5cf0', thr=.55, amt=.8); rim1 = dict(dir=(.0, .35, -.95), col='#e6d2ff', thr=.34, amt=1.); S_, B_, X_ = POSES['stand'], POSES['lean_back'], POSES['block_cross']; up = dict(S_, cL=(5, 0), cR=(5, 0)); seqs = [(S_, 0, 0), (S_, 0, 1), (mix_pose(up, B_, .14), 2, 1), (mix_pose(up, B_, .42), 4, 1), (mix_pose(S_, B_, .66), 6, 1), (mix_pose(mix_pose(S_, B_, .66), X_, .42), 8, 1), (mix_pose(mix_pose(S_, B_, .5), X_, .80), 9, 1), (mix_pose(mix_pose(S_, B_, .36), X_, .94), 10, 1)]
        for pose, wd, lit in seqs:
            F = Fig('gohan', 'nohat').pose(pose).place(*GOH, heading=-.12).blow(0, -wd, wd * .2); L = F.draw(self.fcam, 0, light=(.5, .6, -.6) if not lit else (0, .4, -.9), pal='moon', rim=rim1 if lit else rim0, head_fn='own'); h = F.info['head']; c = Cel(); back_hair(c, h['x'], h['y'], h['s'] * 1.02, -h['roll'] * .8, 3, wd * .012); L = over4(L, c.layer()); G.append(backlit(L, lit, (.5, -.4)).astype(np.float16))
        P = []
        for x, hd, k in ((-2.62, -1.18, 0.), (-1.86, -1.24, .0), (-1.06, -1.40, .45)):
            pz = mix_pose(POSES['lunge'], dict(POSES['lunge'], hL=(.62, .66, .20), hR=(.70, .30, .10), gL='open', gR='open'), k if k else .25); F = Fig('piccolo', 'clean').pose(pz).place(x, .55, .34, heading=hd).blow(-10, -2, 1.5); P.append(backlit(F.draw(self.fcam, 0, light=(.1, .5, -.8), pal='moon', rim=rim1), 1, (.6, -.5)).astype(np.float16))
        os.makedirs('cache', exist_ok=True); np.savez(fn, **{'g%d' % i: g for i, g in enumerate(G)}, **{'p%d' % i: g for i, g in enumerate(P)}); self.G, self.Pl = G, P

    # ------------------------------------------------------------ clocks
    def light(self, f): return 0. if f < 30 else min(1., ((f - 29) / 7.) ** 1.4)
    def dist(self, f): return float(np.exp(keys(f, [(39, math.log(1400.)), (44, math.log(590.)), (48, math.log(250.)), (52, math.log(104.)), (55, math.log(50.)), (58, math.log(20.)), (60, math.log(7.))], lambda t: t))) if f >= 39 else 1e9

    def frame(self, f):
        if f == 60: return self.impact()
        if f > 60: return self.white(f)
        return self.world(f)

    # ------------------------------------------------------------ the world
    def world(self, f, fig=True):
        cam = self.cam; b = self.light(f); wt = (f * .55 if f < 30 else 16.5 + (f - 30) * .9) / 24.; img = self.sky.astype(f32).copy(); sk = (np.arange(H)[:, None, None] < HY)
        d = comp.radial(CX, HY, 1500, 1000, 1.7)
        if b > 0: img = img + ((d[..., None] * C('#4a18a8') * .85 + comp.radial(CX, HY, 700, 470, 1.6)[..., None] * C('#b47cff') * .85) * b) * sk
        hr = 0. if f < 36 else 1900 * (1 - math.exp(-(f - 36) / 13.)); cm, li, cz = cloud_deck(cam, f / 24. * .5, 1300, 1050, -.02, 5, (0, 1500, 1500, b), None if f < 36 else (0, 1500, hr)); col = np.empty((H, W, 3), f32); col[:] = C('#090714'); far = np.clip(cz / 9000, 0, 1)[..., None]; col = col * (1 - far) + C('#1c1236') * far; mh = self.halo[..., None] * np.clip(1.2 - cm[..., None], .25, 1); col = col + C('#5a2ab0') * mh * .55
        for th, cc in ((.06, '#241058'), (.17, '#4a1ca6'), (.36, '#8a4cf0'), (.66, '#d6b8ff')): k = np.clip((li - th) / .015 + .5, 0, 1)[..., None]; col = col * (1 - k) + C(cc) * k
        img = img * (1 - cm[..., None]) + col * cm[..., None]
        cel = Cel(); t = f / 24.; dr = 0. if f < 33 else 430. * (1 - math.exp(-(f - 33) / 10.)) + (f - 33) * 7.
        if f >= 33: dome(cel, CX, HY + 2, dr, t, 'violet', 3, .92, .045, .70)
        if f < 30: self.rg.still(cel)
        else: self.rg.breakup(cel, (f - 30) / 24., 'violet', centre=CX, lift=20.)
        yl = HY - 1
        if f < 34:                                            # the hair line, drawn by an unseen hand, then left glowing
            e = clamp((f + 1) / 16.); e = e * e * (3 - 2 * e); x0 = 64.; x1 = x0 + 1796. * e; w = 2.0 if f < 16 else 1.5 if f < 30 else 3. + (f - 29) * 2.6; cel.fill(np.array([[x0, yl], [x0 + 40, yl - w * .5], [max(x0 + 41, x1 - 40), yl - w * .5], [x1, yl], [max(x0 + 41, x1 - 40), yl + w * .5], [x0 + 40, yl + w * .5]]), '#b987ff'); cel.fill(np.array([[x0 + 10, yl], [max(x0 + 41, x1 - 30), yl - w * .24], [x1, yl], [max(x0 + 41, x1 - 30), yl + w * .24]]), '#ffffff')
            if f < 16:
                for k in range(4): a = k * math.pi / 2 + .3; wedge(cel, (x1, yl), (x1 + math.cos(a) * 24, yl + math.sin(a) * 24 * (.5 if k % 2 else 1.4)), 3.4, 0, '#ffffff')
        if f >= 37: q = (f - 37) / 24.; condensation(cel, CX, HY - dr * .50, dr * (1.02 + 1.7 * q), dr * (.15 + .30 * q), t, 16 + 34 * q, 'violet', 4, .25 + .45 * q)
        img = over(img, cel.layer())
        if f < 34:                                            # its light on the air and on the foot of the ridge
            e = clamp((f + 1) / 16.); e = e * e * (3 - 2 * e); x = np.arange(W); hx = smooth(40, 120, x) * (1 - smooth(64 + 1796 * e - 40, 64 + 1796 * e + 30, x)); img = img + (np.exp(-np.abs(np.arange(H) - yl) / (4. + 4 * (f > 29) * (f - 29)))[:, None] * hx[None])[..., None] * C('#7a2cf0') * (.40 if f < 16 else .24 if f < 30 else .6)
        D = self.dist(f); ring = None; pal = dict(WATER['night'], deep='#040308', mid='#0d0820', lit='#2a1758', foam='#fbf7ff', white='#fbf7ff', shade='#8a4cf0')
        if f >= 39 and D > 8: rr = math.hypot(self.c[0], self.c[1] - cam.eye[1]) - D; ring = (self.c[0], self.c[1], rr, 3.2, max(6., D * .05 + 4), .1, 80., D * .16 + 3)
        sea, m, buf = self.sea.draw(cam, wt, pal, light=(-.045, 1, .05 + .12 * b), ring=ring, road=b, road_w=.16 + .85 * b, road_col='#fbf7ff', fog=('#1c1036' if b == 0 else hexmix('#1c1036', '#b48cff', b * .8), 90, .004), lit_bias=.025 * b, sky=.35 + .6 * b, light2=dict(dir=(.556, 1., .27), road=.9 * (1 - .75 * b), width=.15, skin='#5a2ab8', col='#e9ddff'), streak=None if ring is None else (1., D * .20 + 3, 620.))
        if b > 0: sea = sea + C('#5a22c0') * np.exp(-np.maximum(0, np.arange(H) - HY) / 200.)[:, None, None].astype(f32) * .42 * b
        if f < 34: sea = sea + (np.exp(-np.maximum(0, np.arange(H) - HY) / 5.)[:, None] * hx[None])[..., None] * C('#8a4cf0') * .5
        img = img * (1 - m[..., None]) + sea * m[..., None]
        if f >= 55: img = comp.cam2d(img.astype(f32), *ZOOM, interp=cv2.INTER_CUBIC); cam = self.zcam
        if ring is not None:
            h = float(keys(f, [(39, 13.), (46, 20.), (54, 27.), (60, 34.)])); cel = Cel(); wall(cel, cam, self.c, ring[2], h, t * 1.6, 'violet', 5, .34); img = over(img, cel.layer()); img = over(img, wall_lobes(cam, self.c, ring[2] + h * .06, h, t, 7, 170, ss=1.0, k=.15 if D > 60 else .11))
            if D < 90: cel = Cel(); spray_sheet(cel, cam, t, int(170 * (1 - D / 90)) + 24, 3); img = over(img, cel.layer())
        if f >= 56: k = (f - 55) / 5.; img = img + (1 - np.clip(img, 0, 1)) * comp.radial(CX, 560, 1500 * (.4 + k), 1100 * (.4 + k), 1.2)[..., None] * k * .62
        if f < 55: img = over(img, self.led[0].astype(f32) * (1 - b) + self.led[1].astype(f32) * b)
        if fig: img = self.people(img, f, b)
        return img.astype(f32)

    def people(self, img, f, b):
        g = self.G
        if f < 55:
            if f < 44: G = g[0].astype(f32) * (1 - b) + g[1].astype(f32) * b if b > 0 else g[0].astype(f32)
            else: G = g[2 + min(5, (f - 44) // 2)].astype(f32)
            if 52 <= f: img = self.lunge(img, f)                # he passes on the far side of the child
            img = over(img, G); img = over(img, self.led[1].astype(f32) * b) if b > 0 else img
        else:
            P = hug(f)
            if f == 55: img = over(img, tail(P, -260, 14) * .8)
            img = over(img, P)
            if f >= 57: k = (f - 56) / 4.; a = P[..., 3]; hl = np.clip(cv2.GaussianBlur(1 - a, (0, 0), 16) * 1.5 - .15, 0, 1) * a; img = img + (1 - np.clip(img, 0, 1)) * hl[..., None] * k * .75
        return img

    def lunge(self, img, f):
        P = self.Pl[f - 52].astype(f32); img = over(img, tail(P, -(150 + 30 * (f - 52)), 8) * .8); img = over(img, P); c = Cel(); r = R(f, 9)
        for i in range(16): y = r.uniform(380, 1000); x = r.uniform(-100, 500 + 250 * (f - 52)); wedge(c, (x - r.uniform(300, 700), y + 10), (x, y), .4, r.uniform(2, 6), '#e6d2ff', .75)
        return over(img, c.layer())

    # ------------------------------------------------------------ the front arrives
    def impact(self):
        """one frame: two tones and a violet; the blast is drawn as black-violet claws on white, the two of them cut out in white"""
        ink, paper, vio = '#12052e', '#fbf7ff', '#7a2cf0'; c = Cel(); c.fill(np.array([[0, 0], [W, 0], [W, H], [0, H]]), paper); cx, cy = CX, 640
        kit.focus_lines(c, cx, cy, 330, 1500, 210, 13., ink, 11, .7, .62); kit.focus_lines(c, cx, cy, 250, 1500, 60, 22., vio, 5, .7, .6)
        th = np.linspace(0, TAU, 300); rr = 300 * (1 + .10 * np.abs(noise(np.cos(th) * 6, np.sin(th) * 6, .3)) + .22 * (np.abs(np.sin(th * 9 + 1)) ** 6)); c.fill(np.c_[cx + np.cos(th) * rr, cy + np.sin(th) * rr * .62], paper)
        img = over(np.zeros((H, W, 3), f32), c.layer()); P = hug(58, rip=.12); a = P[..., 3]; col = P[..., :3] / np.maximum(a, 1e-3)[..., None]; L = lum(col); body = np.where((L > .40)[..., None], C(paper), C(ink)); body = np.where(((L > .20) & (L <= .40))[..., None], C(vio), body); o = cv2.dilate(a, np.ones((9, 9), np.uint8)); img = img * (1 - o[..., None]) + C(ink) * o[..., None]; img = img * (1 - a[..., None]) + body * a[..., None]
        return img.astype(f32)

    def white(self, f):
        """61-71 white, two silhouettes hold, the cape goes; 72-79 the white tears open into violet smoke"""
        g = f // 2 * 2; k = f - 60; rip = float(keys(g, [(60, .12), (64, .5), (70, .88), (72, .95), (80, 1.)])); press = float(keys(g, [(60, .0), (64, .45), (68, .75), (74, 1.1), (80, 1.2)])); tr = float(keys(g, [(60, 5.), (66, 3.4), (74, 2.0), (80, 1.2)])); rr = R(g, 17); gust = math.sin(g * .95); P = hug(60 + (g - 60), rip=rip, turban=f < 63, press=press, jit=rr.normal(0, tr, 2) + [0, 5. * press], seed=2 + (g // 2) % 3, spread=1.02 + .05 * gust - .006 * k, swing=.10 * math.sin(g * 1.3), lift=.05 * gust); a = P[..., 3]
        q = clamp((f - 70) / 10.); sm = self.smoke(f, q) if f >= 70 else np.broadcast_to(C('#fbf7ff'), (H, W, 3)).astype(f32).copy()
        c = Cel(); r = R(f, 31); n = 0 if f >= 74 else int(150 * (1 - q) ** 2)
        for i in range(n):                                    # the wind itself: pale streaks rushing out of the light
            th = r.uniform(0, TAU); r0 = r.uniform(260, 1000); l = r.uniform(120, 520); d = np.array([math.cos(th), math.sin(th) * .8]); wedge(c, np.array([CX, 600]) + d * r0, np.array([CX, 600]) + d * (r0 + l), .6, r.uniform(2, 12), '#e4d0ff' if r.random() < .6 else '#cfaaff', .9)
        img = over(sm, c.layer())
        # the two of them: light eats the edges, the core stays
        core = cv2.GaussianBlur(a, (0, 0), 9); pul = .10 * math.sin(g * 2.1) + (.10 if g in (64, 68) else 0.); eat = np.clip((core - (.62 + pul * (1 - q) - .34 * q)) / .30, 0, 1); col = P[..., :3] / np.maximum(a, 1e-3)[..., None]; L = lum(col)[..., None]; sil = C('#3a1488') * (1 - q) + C('#140a28') * q; cl_ = np.clip((col[..., 2:3] - col[..., 0:1] * 1.02) / .03, 0, 1) * np.clip((L - .30) / .05, 0, 1); cl_ = cv2.GaussianBlur(cl_, (0, 0), 1.0)[..., None]; ln = np.clip((.16 - L) / .08, 0, 1); flat = sil * (1 - ln * .45); flat = flat * (1 - cl_ * (.85 - .6 * q)) + C('#9a6cf0') * cl_ * (.85 - .6 * q) * (1 - .35 * ln); flat = flat * (1 - .55 * q) + (col * np.array([.62, .50, .86], f32)) * .55 * q; lx = math.cos(g * .55); side = np.clip(a - cv2.warpAffine(a, np.float32([[1, 0, 11 * lx], [0, 1, 5]]), (W, H)), 0, 1)[..., None] * (.35 + .5 * q); flat = flat * (1 - side) + C('#e6d2ff' if q < .5 else '#b48cff') * side; shade = flat * eat[..., None] + (C('#b48cff') * (1 - q) + flat * q) * (1 - eat[..., None]); img = img * (1 - a[..., None]) + shade * a[..., None]
        c = Cel(); p1 = dict(lit='#b48cff', mid='#7a45d8', dark='#3a1488', ink='#2a0a5e'); fr = k
        if f >= 61:
            for rb in self.ribs: rb.draw(c, fr, p1, lw=1.8)
        if f >= 63: self.tur.draw(c, fr - 2, dict(lit='#e6d2ff', mid='#b48cff', dark='#6a2fd0', ink='#2a0a5e'), lw=1.8)
        rl = c.layer(); img = over(img, rl * (1 - .25 * q))
        if f >= 71: img = over(img, self.near_smoke(f, q))
        if f <= 62: img = img + (1 - np.clip(img, 0, 1)) * (.55 if f == 61 else .25)
        return img.astype(f32)

    def smoke(self, f, q):
        """the white is a wall of lit vapour; it rolls outward, its light dies from white through lilac to violet, it thins and the night shows through"""
        t = f / 24.; y = np.arange(H, dtype=f32)[:, None]; e = smooth(.05, .85, q); bg = np.broadcast_to(ramp(y / H, [(0, '#150c2e'), (.6, '#2a1758'), (1, '#1a0f38')]), (H, W, 3)).astype(f32); bg = bg + comp.radial(CX, 640, 1100, 700, 1.6)[..., None] * C('#7a3af0') * .55; bg = bg * e ** 2 + ramp(np.full((H, 1), float(e)), [(0, '#fbf7ff'), (.35, '#cfb0ff'), (.65, '#6a3fc8'), (1, '#1c1138')]).reshape(H, 1, 3) * (1 - e ** 2); mx = lambda a, b, p=1.: hexmix(a, b, float(e) ** p); L = []; Fg = []
        for i in range(150):
            z = size3(h1(i, 2), .25, .42); a = h1(i, 1) * TAU; r0 = .4 + 5.2 * math.sqrt(h1(i, 3)); o = 1 + (.55 + .5 * h1(i, 6)) * q + .06 * t; X = math.cos(a) * r0 * 1.25 * o; Z = 1.25 + math.sin(a) * r0 * .62 * o; er = .30 * q * h1(i, 5); L.append(lobe(X, 2.2 + 4.5 * h1(i, 4), Z, (.34 + .85 * z) * (1 + .12 * q), i + 5, (1 if X > 0 else -1) * t * 1.1 + i, math.pi / 2, er, 0., -1e9))
        out = over(bg, lobes_layer(self.fcam, L, dict(lit=mx('#ffffff', '#b08cf4'), mid=mx('#fbf6ff', '#6c43c4'), dark=mx('#efe2ff', '#36207c'), ink=mx('#e2cfff', '#22114e'), rim='#fff'), ss=1., light=(-.2, -.8, .5), t_lit=.52, t_dark=.12, ink=.8 * (1 - q), ink_in=1 - q, K1=13, K2=5) if q < .35 else cv2.GaussianBlur(lobes_layer(self.fcam, L, dict(lit=mx('#ffffff', '#8f66e0'), mid=mx('#fbf6ff', '#5c36b0'), dark=mx('#efe2ff', '#331d78'), ink=mx('#e2cfff', '#22114e'), rim='#fff'), ss=1., light=(-.2, -.8, .5), t_lit=.52, t_dark=.12, ink=0., ink_in=.5 * (1 - q), K1=13, K2=5), (0, 0), .6 + 5.5 * (q - .35)) * (1 - .35 * (q - .35) / .65)); return out

    def near_smoke(self, f, q):
        t = f / 24.; L = []; e = float(smooth(.05, .85, q)); mx = lambda a, b: hexmix(a, b, e)
        for i in range(26): sd = -1 if i % 2 else 1; L.append(lobe(sd * (1.5 + 1.6 * h1(i, 1) + .5 * q), -.75 + .5 * h1(i, 3), -.10 + .34 * h1(i, 4) + .10 * t, .22 + .34 * size3(h1(i, 2), .3, .4), i + 71, sd * t * 1.3 + i, math.pi / 2, .15 * q + .2 * h1(i, 5) * q, 0., -1e9))
        return cv2.GaussianBlur(lobes_layer(self.fcam, L, dict(lit=mx('#ffffff', '#8a62dc'), mid=mx('#f6eeff', '#4c2c98'), dark=mx('#e6d6ff', '#261456'), ink=mx('#d6bcff', '#180a38'), rim='#fff'), ss=1., light=(-.2, -.8, .5), ink=0., ink_in=.5), (0, 0), 3.2) * smooth(.1, .4, q)
