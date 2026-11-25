"""Storm ceiling that can be punched into a ring (f3, g1) and the climb through it into sunlight (g2).
    R = Ring(); im, z = R.frame(c, f, k=230., gold=1.)     # c any Camera3D under the ceiling (z up, ceiling at 620 m); k = ring radius in metres (0 = closed),
                                                            # gold = strength of a lamp at R.lamp (an aura / a pillar of light); flash 0..1 lightning inside the cloud
    c = ring_cam(az, f=...)                                 # orbit camera looking up past a body standing at (0,0)
    C = Climb(); im, z, info = C.frame(f)                   # g2, local frames 0..79: storm, sunlit white, blue, indigo, black, the curve of the planet"""
import numpy as np, cv2, math
from . import W, H
from .paint import f32, rgb, smooth, ramp, fbm, grid, mix, _vn
from .cam import Camera3D
from .comp import ease, blur
from . import kit
from .sky_vol import Vol, tones, over_g, prem, floor_field
from .sky_orbit import stars
from .sky_fly import burst_white

ZC = 620.
STORM = dict(a=('#8fb0a6', '#4f706a', '#2b4745', '#193032', '#0b1719', '#274442'), b=('#b8c6d4', '#8294aa', '#4e5e76', '#313e54', '#1a2232', '#2a3448'), c=('#a9c4bb', '#6f938a', '#385854', '#1e3838', '#0d1b1d', '#2c4a47'))
GREY = dict(a=('#c4ced8', '#7d8c9f', '#4c5a6d', '#323d50', '#1c2534', '#3f4b5c'), c=('#d6dee6', '#8c9aad', '#57667b', '#394558', '#202a3a', '#46536a'))


def ring_cam(az=0., dist=14., eye_z=.5, look_z=9., focal=820., cx=960, cy=540, roll=0.):
    e = np.array([math.sin(az) * -dist, -math.cos(az) * dist, eye_z]); return Camera3D(e, (0, 0, look_z), focal, cx=cx, cy=cy, roll=roll)


class Ring:
    def __init__(self, seed=0, span=3000):
        V = Vol(); hh = lambda r: [(0, 0, r)]
        V.deck((-11000, 11000), (-11000, 11000), ZC + 120, 560, seed=4 + seed, rel=.22, levels=1, kids=(8,), sink=.62, under=True, jitter=.55, hole=hh(3700))
        V.deck((-4300, 4300), (-4300, 4300), ZC + 40, 230, seed=5 + seed, rel=.30, levels=1, kids=(9,), sink=.50, under=True, jitter=.55, hole=hh(1500))
        V.deck((-1800, 1800), (-1800, 1800), ZC, 112, seed=11 + seed, rel=.40, levels=3, kids=(8, 6, 4), sink=.42, under=True, jitter=.55, detail_d=1700, eye=(0, 0))
        for i, zz in enumerate((-10., 75., 165., 255.)): V.deck((-460, 460), (-460, 460), ZC + zz, 52, seed=30 + i + seed, pal=1, rel=.45, levels=2, kids=(8, 5), sink=.35, under=True, jitter=.6, thr=.8)
        V.build(sun=(.25, .4, .88), res=512); self.V = V; self.r0 = np.hypot(V.C[:, 0], V.C[:, 1]).astype(f32); self.th = np.arctan2(V.C[:, 1], V.C[:, 0]).astype(f32); self.n = np.array([_vn(math.cos(t) * 2.2 + 9, math.sin(t) * 2.2 + 9, 3) for t in self.th], f32) - .5; self.PS = {k: np.stack([tones(*q['a']), tones(*q['c'])]) for k, q in (('green', STORM), ('grey', GREY))}; self.lamp = np.array([0, 0, 330.]); self.x, self.y = grid()

    def move(self, k, swirl=0., lift=0.):
        """area-keeping push: everything that filled the hole is piled into its rim, lips curl up"""
        def m(C):
            ke = k * (1 + .06 * self.n); r = np.sqrt(self.r0 ** 2 + ke ** 2 * np.exp(-(self.r0 / (ke * 3.2 + 1.)) ** 2)); t = self.th + swirl * np.exp(-((r - ke) / (ke * 1.1 + 40)) ** 2); z = C[:, 2] + (k * .16) * np.exp(-((r - ke) / (k * .30 + 12)) ** 2) * np.sin(self.n * 18 + self.th * 3) * .8 + lift * np.exp(-(r / (k * 1.5 + 60)) ** 2)
            z = z - k * .10 * np.exp(-((r - ke * 1.35) / (k * .35 + 15)) ** 2); r = np.where(self.r0 < ke * 3.5, np.maximum(r, ke + self.V.R * .86), r); return np.stack([np.cos(t) * r, np.sin(t) * r, z], -1)
        return m if k > 0 else None

    def sky(self, c, f, k, top=('#02040c', '#0a1430', '#1b2f58'), hzc='#5f7c78', st=.8, base='#142426'):
        d = c.rays(); e = d[..., 2] / np.linalg.norm(d, axis=-1); im = ramp(np.clip(e, 0, 1), [(0, top[2]), (.35, top[1]), (1, top[0])]).astype(f32)
        if st > 0: im = stars(im, c, st, belt=.12)
        t = np.maximum(np.where(e > .004, (ZC + 150 - c.eye[2]) / np.maximum(d[..., 2], 1e-4), 0), 0); X, Y = c.eye[0] + d[..., 0] * t, c.eye[1] + d[..., 1] * t; r = np.hypot(X, Y); back = np.where(e > .004, smooth(k * 1.06, k * 1.30 + 4, r), 1.) if ZC + 150 > c.eye[2] else np.zeros_like(e)
        dist = t * np.linalg.norm(d, axis=-1); col = rgb(base) + (rgb(hzc) - rgb(base)) * (1 - np.exp(-(dist / 5200) ** 1.4))[..., None]; lo = ramp(np.clip(e, -.3, .1), [(-.3, '#0e1718'), (-.02, mix(rgb(hzc), rgb('#22363a'), .35)), (.0, hzc), (.06, mix(rgb(hzc), rgb(base), .25))]); col = col + (lo - col) * smooth(.05, .01, e)[..., None]
        return (im + (col - im) * back[..., None]).astype(f32)

    def frame(self, c, f=0, k=0., gold=0., flash=0., swirl=None, goldc='#ffe06a', goldc2='#f6a416', skin='green', hzc='#46625f', lamp=None, above=0., top=('#060a20', '#101e4a', '#26427c'), st=1.1, near=(10., 60.), **kw):
        """above 0..1: light pouring down through the hole onto its lip (sunlit world over the storm)"""
        sw = .10 * f / 24. + .12 * min(1., k / 200.) if swirl is None else swirl; im = self.sky(c, f, k, top=top, hzc=hzc, st=st, base='#142426' if skin == 'green' else '#1b2230')
        lamp = self.lamp if lamp is None else np.asarray(lamp, float); L = []
        if gold > 0: L = [dict(pos=(lamp[0], lamp[1], ZC - 90), col=goldc2, I=gold * .85, r=max(150., k * 1.30)), dict(pos=(lamp[0], lamp[1], ZC + 120), col=goldc, I=gold * 1.15, r=max(120., k * 1.22))]
        if above > 0: L.append(dict(pos=(60., 90., ZC + 420), col='#f2f7ff', I=above * 1.3, r=max(260., k * 1.5 + 260)))
        if flash > 0: L.append(dict(pos=(700, 900, ZC - 60), col='#e9f0ff', I=flash * 1.2, r=420.))
        q = dict(fog=dict(d=3300., p=1.35, col=hzc), light=(.15, .93, -.28), shadow=False, tl=.66, th=.34, eh=.22, el=.06, bounce=.0, skyl=0., deep=.0, acc=.3, accs=.35, noise=.16, mixn=(.14, .28, .75), melt=5, tol=1.8, wob=.15, move=self.move(k, sw), lamps=L, near=near); q.update(kw)
        pal = self.PS[skin].copy(); (hp,), (hz_,) = c.project(np.array([[0, 0, ZC + 200.]]))
        if k > 0 and hz_ > 0:                                             # the lip catches the light that comes through the hole
            q.update(sil=min(1.15, gold * 1.1 + above * 1.05), silw=.42, silr=float(np.clip(c.f * k * 1.5 / hz_ / c.w, .06, 1.2)), sunscreen=(float(hp[0]), float(hp[1])))
            if gold > 0: pal[:, 0] = rgb('#fff3b8')
            elif above > 0: pal[:, 0] = rgb('#f4f8ff')
        g = self.V.render(c, pal, **q); z = np.where(g['a'] > .5, g['z'], 1e9).astype(f32); im = over_g(im, g); gl = max(gold, above * .8)
        if gl > 0 and k > 0:                                              # thin edges let the light through: glow dies away from the hole's edge into the cloud
            d = c.rays(); m = ((1 - g['a']) * (d[..., 2] > .05)).astype(f32); q4 = cv2.resize(m, (c.w // 4, c.h // 4), interpolation=cv2.INTER_AREA); G = sum(w * cv2.GaussianBlur(q4, (0, 0), s) for w, s in ((.30, 2.2), (.34, 7.), (.30, 20.), (.22, 46.))); G = cv2.resize(G, (c.w, c.h), interpolation=cv2.INTER_LINEAR) * (1 - m)
            col = rgb('#ffd35a' if gold > 0 else '#dbe7f6'); im = im + (col - im) * np.clip(G * gl * ((.62 if gold > 0 else 1.0) + .9 * (g['v'] - .2)), 0, .92)[..., None]
            if gold > 0: im = im + rgb(goldc) * (cv2.resize(cv2.GaussianBlur(q4, (0, 0), 34), (c.w, c.h)) * .20 * gold * (1 - .85 * m))[..., None]
        if k > 60: im = self.wisps(im, c, f, k, sw, z, rgb('#ffe9a0' if gold > .05 else '#e6edf6' if above > 0 else '#8fb0a6'), max(gold, above, .4))
        return im, z

    def wisps(self, im, c, f, k, sw, z, col, amt):
        """torn vapour dragged round the lip"""
        if not hasattr(self, 'ws'):
            from .sky_dawn import _wisp
            self.ws = [_wisp(s + 40, 520, 120) for s in range(4)]; r = np.random.default_rng(8); self.wl = [(r.uniform(0, 6.283), r.uniform(.96, 1.45), r.uniform(-70, 150), r.uniform(.28, .62), int(r.integers(0, 4)), r.uniform(.6, 1.6), r.uniform(.25, .8)) for _ in range(46)]
        for (a, q, dz, L, j, sp, op) in self.wl:
            a = a + sw * sp * 1.6; R = k * q; P0 = np.array([math.cos(a) * R, math.sin(a) * R, ZC + dz]); T = np.array([-math.sin(a), math.cos(a), .10]) * L * k * .5; (p, p1), (zz, _) = c.project(np.stack([P0, P0 + T])); d = p1 - p; n = np.hypot(*d)
            if zz < 40 or n < 12 or not (-n < p[0] < c.w + n and -n < p[1] < c.h + n): continue
            t = self.ws[j]; sx = n / (t.shape[1] / 2.); ang = math.atan2(d[1], d[0]); ca, sa = math.cos(ang) * sx, math.sin(ang) * sx; sy = min(1., sx * .8); b = int(n * 1.15 + 60); x0, x1, y0, y1 = int(max(0, p[0] - b)), int(min(c.w, p[0] + b)), int(max(0, p[1] - b)), int(min(c.h, p[1] + b))
            if x1 - x0 < 8 or y1 - y0 < 8: continue
            M = np.float32([[ca, -math.sin(ang) * sy, 0], [sa, math.cos(ang) * sy, 0]]); M[:, 2] = np.array([p[0] - x0, p[1] - y0]) - M[:, :2] @ np.array([t.shape[1] / 2, t.shape[0] / 2]); m = cv2.warpAffine(t, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR) * smooth(-120, 60, z[y0:y1, x0:x1] - zz) * op * amt * .75; im[y0:y1, x0:x1] += (col - im[y0:y1, x0:x1]) * m[..., None]
        return im


class Climb:
    """altitudes in metres: storm 620..930, tops at 960, towers to 2300"""

    def __init__(self):
        self.ring = Ring(seed=2, span=2200); V = Vol(); r = np.random.default_rng(12)
        V.deck((-2600, 2600), (-2600, 2600), 905., 105, seed=8, pal=0, rel=.34, levels=2, kids=(8, 5), sink=.5, thr=None, hole=[(0, 0, 165)], detail_d=1400, eye=(0, 0))
        V.deck((-9000, 9000), (-9000, 9000), 880., 420, seed=9, pal=0, rel=.30, levels=1, kids=(7,), sink=.55, hole=[(0, 0, 2450)])
        V.deck((-26000, 26000), (-26000, 26000), 820., 1150, seed=10, pal=0, rel=.26, levels=1, kids=(6,), sink=.6, hole=[(0, 0, 8700)])
        for i in range(30):
            a = i * 2.399 + r.normal(0, .2); d = 330 + 2300 * (i / 30.) ** 1.1; w = r.uniform(320, 640) * (.75 + d / 2600.); hh = w * r.uniform(.42, .95) * (1.2 if i < 8 else 1.); V.cumulus(math.cos(a) * d, math.sin(a) * d, 880., w, hh, seed=300 + i, pal=1, peaks=5, levels=2, foot=.08, size=1.2, kids=(14, 8))
        V.build(sun=(.78, .50, .34), res=4096); self.V = V; self.P = np.stack([tones('#fff4d6', '#ffd9a0', '#d9a3a0', '#8f86b8', '#5d629e', '#b48fa8'), tones('#fffaea', '#ffe6bc', '#e3b7ac', '#9a94c4', '#6871ac', '#c4a0b4')])
        U = Vol()                                                         # thin high scraps at 3000
        for i in range(16): a = r.uniform(0, 6.28); d = r.uniform(120, 1500); U.puff(math.cos(a) * d, math.sin(a) * d, 2900 + r.uniform(-120, 160), r.uniform(26, 80), seed=i, pal=0, n=8, stretch=(2.4, 1.2, .3), kids=(8, 5))
        U.build(sun=(.78, .5, .2), res=1024); self.U = U; self.PU = np.stack([tones('#ffffff', '#fff0da', '#eee0ea', '#cfc8e6', '#b4b4de', '#dcd6ee')]); self.x, self.y = grid()

    def sky(self, c, alt, st=0.):
        """altitude ramp: storm grey -> sunlit white -> blue -> indigo -> black, always with a bright air skin at the horizon"""
        d = c.rays(); e = d[..., 2] / np.linalg.norm(d, axis=-1); K = [(900, ('#ffe2b4', '#f3c7ae', '#8fb4dc', '#4f86cc')), (1800, ('#ffe9c4', '#e8c9c4', '#6aa0dc', '#2f63b8')), (3200, ('#fff0d2', '#bcd0ea', '#3a78cc', '#173a8c')), (6000, ('#fdf3e0', '#8fbcee', '#1f4ba8', '#0c1c58')), (14000, ('#f2f6ff', '#5e9cf0', '#122c7a', '#070d30'))]
        a = [q[0] for q in K]; i = int(np.clip(np.searchsorted(a, alt) - 1, 0, len(K) - 2)); t = float(np.clip((alt - a[i]) / (a[i + 1] - a[i]), 0, 1)); C = [rgb(p) + (rgb(q) - rgb(p)) * t for p, q in zip(K[i][1], K[i + 1][1])]
        im = ramp(np.clip(e, -.05, 1), [(-.05, C[0]), (.0, C[0]), (.05, C[1]), (.28, C[2]), (.9, C[3])]).astype(f32); return stars(im, c, st, belt=.10) if st > 0 else im

    def mover(self, c, holes=(), punch=()):
        """holes = [(x, y, k)]: round shafts blown straight down through the deck;  punch = [(world point, radius m)]: a clean hole along our line of sight through whatever cloud stands there"""
        holes = [q for q in holes if q[2] > 1]; punch = [q for q in punch if q[1] > 1]
        if not holes and not punch: return None
        R = self.V.R
        def m(C):
            C = C.astype(np.float64).copy()
            for (x, y, k, *al) in holes:
                dx, dy = C[:, 0] - x, C[:, 1] - y; r0 = np.hypot(dx, dy) + 1e-6; low = (C[:, 2] < 1150) | bool(al and al[0]); r = np.where(low, np.sqrt(r0 ** 2 + k ** 2 * np.exp(-(r0 / (k * 2.6)) ** 2)), r0); r = np.where(low & (r0 < k * 2.6), np.maximum(r, k + R * .8), r); lip = np.exp(-((r - k * 1.12) / (k * .30)) ** 2) * low
                C[:, 0] = x + dx / r0 * r; C[:, 1] = y + dy / r0 * r; C[:, 2] += lip * k * .16
            for (P, rad, *asp) in punch:
                asp = asp[0] if asp else 1.; q = (C - np.asarray(c.eye)) @ c.R.T; w = (np.asarray(P) - np.asarray(c.eye)) @ c.R.T; vx, vy = w[0] / w[2], w[1] / w[2]; z = np.maximum(q[:, 2], 1.); ox, oy = q[:, 0] / z - vx, (q[:, 1] / z - vy) / asp; r0 = np.hypot(ox, oy) + 1e-9; ke = rad / w[2]; on = (q[:, 2] > w[2] * .15) & (q[:, 2] < w[2] * 14) & (r0 < ke * 4 + R / z)
                r = np.sqrt(r0 ** 2 + ke ** 2 * np.exp(-(r0 / (ke * 2.2)) ** 2)); r = np.maximum(r, ke + R / z * 1.02 / asp); r = np.where(on, r, r0); lip = np.exp(-((r - ke * 1.1) / (ke * .35)) ** 2) * on; q2 = np.stack([(vx + ox / r0 * r) * z, (vy + oy / r0 * r * asp) * z, z * (1 - .03 * lip)], -1); C = np.where(on[:, None], np.asarray(c.eye) + q2 @ c.R, C)
            return C
        return m

    def world(self, c, f, st=0., holes=(), punch=(), sky_alt=None, near=(8., 40.)):
        alt = float(c.eye[2]); im = self.sky(c, 900 + (alt - 900) * 2.1 if sky_alt is None else sky_alt, st); hz = self.sky_h(alt); mv = self.mover(c, holes, punch); F = floor_field(self.V, c, 900., l=3000., ky=1.0, sun=(.8, .5), seed=9); d = F[..., 0]; m = d > 0
        if m.any():
            col = rgb('#cf9ea6') + (rgb('#ffdda8') - rgb('#cf9ea6')) * (smooth(.02, .16, F[..., 2]) * .55)[..., None]; col = col + (rgb('#8f83b4') - col) * (smooth(.0, .25, -F[..., 2]) * .4)[..., None]; col = col + (hz - col) * (1 - np.exp(-(d / (9000. + alt * 2.2)) ** 1.5))[..., None]; dk = smooth(150, 190, np.hypot(F[..., 4], F[..., 5]) + 20 * F[..., 1])
            for (hx, hy, hk, *al) in holes: dk = np.minimum(dk, smooth(hk * .97, hk * 1.06 + 4, np.hypot(F[..., 4] - hx, F[..., 5] - hy) + hk * .06 * F[..., 1]))
            pit = np.broadcast_to(rgb('#2a3346'), col.shape).copy()                # what lies under a hole: the dark storm, a little lighter toward the lit rim
            vd = np.asarray(c.R[2][:2]); vd = vd / (np.linalg.norm(vd) + 1e-6)
            for (hx, hy, hk, *al) in holes:
                if hk > 1:                                                         # we look obliquely into a shaft: its far wall is lit at the lip and falls into the dark storm
                    ux, uy = (F[..., 4] - hx) / hk, (F[..., 5] - hy) / hk; q = np.hypot(ux, uy); v = ux * vd[0] + uy * vd[1]; w = smooth(-.25, 1., v - .55 * (1 - np.sqrt(np.maximum(1 - (ux * vd[1] - uy * vd[0]) ** 2, 0))) + .10 * F[..., 1]); p_ = rgb('#0f1624') + (rgb('#4a5878') - rgb('#0f1624')) * (w ** 1.6)[..., None]; p_ = p_ + (rgb('#b79ab8') - p_) * smooth(.78, 1., w)[..., None] * .8; pit = np.where((q < 1.08)[..., None], p_, pit)
            for (Pw, rad, *asp) in punch:                                          # a shaft along our line of sight opens the floor too: far wall lit at the lip, then dark
                asp = asp[0] if asp else 1.; (hp,), (hz_,) = c.project(np.asarray(Pw)[None])
                if rad > 1 and hz_ > 0:
                    R = c.f * rad / hz_; ux, uy = (self.x - hp[0]) / R, (self.y - hp[1]) / (R * asp); th = np.arctan2(uy, ux); q = np.hypot(ux, uy) * (1 + .030 * np.abs(np.sin(th * 6.5 + 1)) + .022 * np.abs(np.sin(th * 14 + .3))); dk = np.where(d > hz_ * .3, np.minimum(dk, smooth(.99, 1.015, q)), dk)
                    e = np.sqrt(np.maximum(1 - np.minimum(ux * ux, 1), 0)); w = np.clip((-uy + e) / (e + 1e-3) * .5, 0, 1); w = w + .05 * np.sin(ux * 21 + 2 * np.sin(ux * 7)) * e; p_ = rgb('#101726') + (rgb('#46557a') - rgb('#101726')) * smooth(.42, .80, w)[..., None]; p_ = p_ + (rgb('#9a86b4') - p_) * smooth(.76, .82, w)[..., None] * .85; p_ = p_ + (rgb('#e6bfae') - p_) * smooth(.91, .95, w)[..., None] * .8; pit = np.where((q < 1.02)[..., None], p_, pit)
            im = np.where(m[..., None], pit + (col - pit) * dk[..., None], im).astype(f32)
        (sp,), (sz,) = c.project((np.asarray(c.eye) + self.V.L * 1e5)[None]); ss = tuple(sp) if sz > 0 else (c.cx - (sp[0] - c.cx) * 50, c.cy - (sp[1] - c.cy) * 50); kw = dict(tl=.24, th=-.06, bounce=.45, skyl=.18, acc=.45, accs=.22, noise=.12, mixn=(.2, .3, .65), near=near, sil=.75, silw=.26, sunscreen=ss)
        g = self.V.render(c, self.P, fog=dict(d=5200. + alt * 1.2, p=1.35, col=hz), minpx=.9, move=mv, **kw); z = np.where(g['a'] > .5, g['z'], np.where(m, d, 1e9)).astype(f32); im = over_g(im, g)
        if alt > 1500: h = self.U.render(c, self.PU, fog=dict(d=9000.), **{**kw, 'near': (6., 60.)}); im = over_g(im, h, .85); z = np.where(h['a'] > .5, np.minimum(z, h['z']), z)
        return im, z

    # ------------------------------------------------------------ g2 staging (local frames 0..79)
    def fighters(self, f):
        """altitude of the two lights (m) while they are in the air world"""
        return float(np.interp(f, [0, 10, 20, 30, 40], [230, 560, 1010, 2300, 6200]))

    def cam(self, f):
        h = self.fighters(f)
        if f < 20: e = np.array([6 * math.sin(f * .6), -26., h - 85 - 2.5 * f]); t = np.array([0, 4., h + 40]); foc = 900. - 12 * f; ro = math.radians(3 * math.sin(f * .5))
        else:
            u = (f - 20) / 20.; a = math.radians(205 + 62 * ease(u)); r = 320 + 60 * u; up = np.interp(u, [0, .35, .7, 1], [-60, 20, 160, 330]); e = np.array([math.cos(a) * r, math.sin(a) * r, h + up]); t = np.array([0, 0, h + np.interp(u, [0, .35, .7, 1], [70, 10, 40, 150])]); foc = 860.; ro = math.radians(-7 + 10 * u)
        return Camera3D(e, t, foc, cx=960, cy=540, roll=ro)

    def frame(self, f, c=None, blur=True):
        """0..39 -> rgb, z, (screen xy of the lights, px per metre)"""
        c = self.cam(f) if c is None else c; h = self.fighters(f); (p,), (pz,) = c.project(np.array([[0, 0, h]])); info = (tuple(p), c.f / max(pz, 1.))
        if f < 20:
            k = 205. + 2. * f; q = min(1., max(0., (f - 4) / 13.)); tp = [rgb(a) + (rgb(b) - rgb(a)) * q for a, b in zip(('#26406e', '#5a7cb0', '#8fa6c4'), ('#7fb2e8', '#cfe2f4', '#fff4dc'))]
            im, z = self.ring.frame(c, f, k, skin='grey', hzc='#6a7688', above=.25 + .75 * q, top=tp, st=0., lamps=None) if False else self.ring.frame(c, f, k, skin='grey', hzc='#6a7688', above=.25 + .75 * q, top=tp, st=0.)
            if blur: im = kit.zoomblur(im, p[0], p[1], .012 + .030 * f / 20., 7)
            if f >= 15: im = burst_white(im, f, (.12, .30, .55, .85, 1.)[f - 15], '#fffdf4', '#ffe9c4', (p[0], p[1]))
        else:
            im, z = self.world(c, f, st=max(0., (f - 33) / 8.))
            if f < 24: im = burst_white(im, f, (.72, .42, .22, .08)[f - 20], '#fffdf4', '#ffe9c4', (p[0], p[1]))
        return im, z, info

    def sky_h(self, alt): return ramp(np.float32([alt]), [(900, '#f6cfae'), (2000, '#f0d6cc'), (4000, '#cfdcee'), (9000, '#a9c6f0')])[0]
