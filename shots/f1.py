"""f1 THE RAIN STOPS.  Close on the puddle and the stone ball.  0-1 rain; 2 every drop stops dead in mid-air and is a lens: the whole grey world hangs upside down inside.  Nothing moves.
14 one by one they go up, rings run inward and close, crowns un-happen; 24 a pebble lifts and turns; 32 the water is pressed into a bowl and a dark face is in it; 38 everything trembles."""
import numpy as np, cv2, math, os
from db import W, H, comp, kit
from db.paint import rgb, f32, smooth
from db.ink import Cel, spl
from db.cam import Camera3D
from db.phys import *
keys = comp.lerp_keys

EYE, TGT, FOC = (-.18, -.95, 1.05), (-.13, .25, 0.), 1650.
BALL = (-.47, .20, .030, .098)                                       # x y z(centre) r, metres: a stone ball standing in the water
BOWL = (.20, .16, .455)                                               # x y radius
FACE_Y = 1.52
# frozen rings: x, y, [(radius, amp)], frame the last one closes
RINGS = [(.36, .02, ((.115, 1.), (.072, .7)), 22), (-.12, -.12, ((.165, .8), (.11, .7), (.06, .5)), 27), (.70, .26, ((.09, .9),), 19), (.05, .42, ((.14, .9), (.085, .8)), 30), (-.78, -.02, ((.10, .9), (.05, .6)), 24), (-.22, .40, ((.06, 1.),), 17), (.52, .44, ((.12, .8), (.07, .6)), 33), (.28, -.30, ((.08, 1.),), 20), (-.55, -.16, ((.07, .9),), 29)]
CROWNS = [(.36, .02, .62, .030, .085, 15), (-.12, -.12, .50, .024, .060, 18), (-.78, -.02, .55, .026, .105, 21), (.05, .42, .55, .026, .050, 20)]       # x y v0 r0 frozen-at s, frame it starts to run backward
HERO = [(1268, 356, 148, 22.0, .02, 0.), (470, 566, 92, 17.5, .02, 0.), (1640, 640, 78, 15.5, .02, 0.), (842, 196, 62, 19., .02, 0.), (1010, 742, 44, 14.2, .03, 0.), (1500, 152, 40, 24.5, .02, 0.), (214, 262, 52, 21., .02, 0.), (1352, 900, 33, 16.5, .02, 0.),
        (690, 420, 28, 23., .02, 0.), (1790, 330, 30, 18.2, .02, 0.), (330, 842, 25, 20.2, .02, 0.), (1404, 700, 22, 25., .02, 0.), (596, 108, 20, 16., .02, 0.), (1066, 120, 22, 27., .02, 0.), (80, 560, 24, 26., .02, 0.), (1710, 930, 28, 22.5, .02, 0.), (880, 560, 17, 28., .02, 0.)]
NEAR = [(96, 930, 150, 19., 16.), (1860, 96, 120, 23.5, 13.)]        # out of focus, close to the lens


class Shot:
    id = 'f1'; n = 40
    finish = dict(bloom=.22, vig=.18, grain=.011, soft=.10)
    EVENTS = [(0, 'rain', dict(frames=2, note='full rain, then')), (2, 'rain_stops', dict(size=1., note='every sound is cut dead on this frame; only one low tone')), (14, 'drop_up', dict(size=.2)), (16, 'drop_up', dict(size=.25)), (17, 'ring_closes', dict(size=.2)), (18, 'drop_up', dict(size=.2)), (19, 'ring_closes', dict(size=.2)),
              (20, 'ring_closes', dict(size=.25)), (22, 'drop_up', dict(size=.5, note='the big drop')), (24, 'pebble', dict(size=.5, note='a pebble lifts and turns')), (27, 'ring_closes', dict(size=.2)), (28, 'pebble', dict(size=.3)), (31, 'pebble', dict(size=.3)), (32, 'bowl', dict(size=.9, note='the water is pressed into a bowl: pressure, sub')), (38, 'tremble', dict(frames=2, size=1.))]

    def setup(self):
        self.cam = Camera3D(EYE, TGT, FOC); fn = 'cache/f1_plate_v2.npz'
        if os.path.exists(fn): z = np.load(fn); self.pl, self.wm, self.face = z['pl'].astype(f32), z['wm'].astype(f32), z['face'].astype(f32)
        else:
            from db import land; from chars import goku
            o = land.puddle(0., self.cam); self.pl, self.wm = o.img.astype(f32), o.water.astype(f32)
            HH = goku.GHead().pose(960, 470, 150, -.06, .30, .03); self.face = goku.bust(HH, 'griefhid', 'rain', hair='base', wet=1., f=3)[..., :4].astype(f32)
            os.makedirs('cache', exist_ok=True); np.savez(fn, pl=self.pl.astype(np.float16), wm=self.wm.astype(np.float16), face=self.face.astype(np.float16))
        self.wm = cv2.GaussianBlur(self.wm, (0, 0), 1.0); r = self.cam.rays().astype(np.float64); self.d = r; ys, xs = np.where(self.wm > .01); self.roi = (max(0, xs.min() - 8), max(0, ys.min() - 60), min(W, xs.max() + 8), min(H, ys.max() + 8))
        self.pl = np.clip(self.pl * 1.12 + .012, 0, 1.2); L = lum(self.pl)[..., None]; q = 9.; Lq = (np.floor(L * q) + smooth(.35, .65, (L * q) % 1)) / q; self.pl = (self.pl * np.clip((Lq + .03) / (L + .03), .7, 1.35)).astype(f32)       # the rock drops into the film's cel tones
        self.rn = Rain(560, (3.2, 3.4, 2.2), 6, (.35, 0), speed=6., near=.42, shutter=.05); from db import ball as B; self.B = B
        bx, by, bz, br = BALL; q, z = self.cam.project(np.array([[bx, by, bz]])); self.bp = (float(q[0, 0]), float(q[0, 1]), br * FOC / float(z[0]))
        self.peb = [(chunk(11, 'block').scaled(.070), np.array([.37, -.235, .030]), 24, 5, 1.0), (chunk(23, 'shard').scaled(.030), np.array([.60, -.05, .012]), 28, 9, 1.2), (chunk(5, 'block').scaled(.024), np.array([-.93, .36, .035]), 31, 3, 1.1), (chunk(41, 'block').scaled(.020), np.array([.16, -.34, .010]), 30, 13, 1.3)]

    # ------------------------------------------------------------ water: a reflecting surface with real normals
    def rings(self, f):
        out = []
        for j, (x, y, rs, end) in enumerate(RINGS):
            go = max(14, end - 9 - j % 3); u = clamp((f - go) / (end - go)); k = 1 - u * u * (3 - 2 * u) * .5 - u * .5
            for r, a in rs:
                R_ = r * k
                if u < 1 and R_ > .002: out.append((x, y, R_, .0075 + .004 * r / .1, .0019 * a * (1 + .9 * u)))
        for j, (x, y, rs, end) in enumerate(RINGS):           # the point where it closed jumps up once
            s = f - end
            if 0 <= s < 3: out.append((x, y, .0, .016, .006 * (1 - s / 3.)))
        return out

    def water(self, f, rings=None):
        x0, y0, x1, y1 = self.roi; d = self.d[y0:y1, x0:x1]; e = np.array(EYE); rg = self.rings(f) if rings is None else rings; A = self.bowl(f); bx, by, br = BOWL; Z = np.zeros(d.shape[:2])
        def field(X, Y):
            h = np.zeros_like(X); gx = np.zeros_like(X); gy = np.zeros_like(X); bx_ = np.zeros_like(X); by_ = np.zeros_like(X)
            for (cx, cy, R_, w, a) in rg:
                dx, dy = X - cx, Y - cy; r = np.sqrt(dx * dx + dy * dy) + 1e-6; s = (r - R_) / w; g = np.exp(-s * s); c = np.cos(s * 1.9); h += a * g * c; k = a * g * (-2 * s * c - 1.9 * np.sin(s * 1.9)) / w / r; gx += k * dx; gy += k * dy
            if A > 0:
                dx, dy = X - bx, (Y - by) * 1.35; r = np.sqrt(dx * dx + dy * dy) / br + 1e-6; g = np.exp(-r ** 4); h -= .080 * A * g; k = .080 * A * g * 4 * r ** 3 / br / (r * br); bx_ += k * dx; by_ += k * dy * 1.35
                s = (r - 1.16) / .17; g = np.exp(-s * s); h += .064 * A * g; k = .064 * A * g * (-2 * s / .17) / br / (r * br); bx_ += k * dx; by_ += k * dy * 1.35
            if f >= 37:                                       # trembling: standing ripples
                
                for j in range(5): an = j * math.pi / 5 + .3; kf = 95. + 9 * j; ph = (X * math.cos(an) + Y * math.sin(an)) * kf + j * 1.9 + f * 2.4; a = .00010 * (f - 36); h += a * np.sin(ph); gx += a * kf * np.cos(ph) * math.cos(an); gy += a * kf * np.cos(ph) * math.sin(an)
            return h, gx, gy, bx_, by_
        for _ in range(3 if A > 0 else 1): t = (Z - e[2]) / d[..., 2]; X, Y = e[0] + d[..., 0] * t, e[1] + d[..., 1] * t; Z, gx, gy, bgx, bgy = field(X, Y)
        fx_, fy_ = gx + .07 * bgx, gy + .07 * bgy; gx, gy = gx + bgx, gy + bgy
        nl = 1 / np.sqrt(fx_ * fx_ + fy_ * fy_ + 1); nx, ny, nz = -fx_ * nl, -fy_ * nl, nl; dl = 1 / np.sqrt((d * d).sum(-1)); vx, vy, vz = d[..., 0] * dl, d[..., 1] * dl, d[..., 2] * dl; dn = vx * nx + vy * ny + vz * nz; qrx, qry, qrz = vx - 2 * dn * nx, vy - 2 * dn * ny, vz - 2 * dn * nz
        nl = 1 / np.sqrt(gx * gx + gy * gy + 1); nx, ny, nz = -gx * nl, -gy * nl, nl; dl = 1 / np.sqrt((d * d).sum(-1)); vx, vy, vz = d[..., 0] * dl, d[..., 1] * dl, d[..., 2] * dl; dn = vx * nx + vy * ny + vz * nz; rx, ry, rz = vx - 2 * dn * nx, vy - 2 * dn * ny, vz - 2 * dn * nz
        az = np.arctan2(rx, np.maximum(ry, 1e-3)); el = np.clip(rz, 0, 1); sk = ramp(np.clip(el, 0, 1), [(0, '#7a7e7e'), (.35, '#a3a7a6'), (.62, '#7f8384'), (1, '#54585b')]); cl = noise(az * 2.6, el * 4.2, 1.3, 5) + .5 * noise(az * 6.2, el * 9., 2.1, 6); k = np.clip((cl - .12) / .05, 0, 1)[..., None]; sk = sk * (1 - k * .16); k = np.clip((-cl - .30) / .05, 0, 1)[..., None]; sk = sk * (1 + k * .14)
        # the figure on the far side, a standing picture at y = FACE_Y
        s = (FACE_Y - Y) / np.maximum(qry, 1e-3); U = X + qrx * s; V = Z + qrz * s + .20 * A; ppm = 430.; mx = (960 + (U - .40) * ppm).astype(f32); my = (560 - (V - 1.26) * ppm).astype(f32); fc = cv2.remap(self.face, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT); st = float(keys(f, [(0, .42), (14, .46), (30, .66), (36, 1.), (40, 1.)])); fa = fc[..., 3:4] * st; dark = fc[..., :3] * .74 + fc[..., 3:4] * C('#0d1014') * .10; env = sk * (1 - fa) + dark * st
        # the ball seen in the water
        o = np.array(BALL[:3]); qx, qy, qz = X - o[0], Y - o[1], Z - o[2]; b = qx * rx + qy * ry + qz * rz; c_ = qx * qx + qy * qy + qz * qz - BALL[3] ** 2; D = b * b - c_; hit = (D > 0) & (-b > 0); ts = -b - np.sqrt(np.maximum(D, 0)); hx, hz = (qx + rx * ts) / BALL[3], (qz + rz * ts) / BALL[3]; sh = np.where((hx * .5 + hz * .8)[..., None] > .15, C('#4c5052'), C('#25282a')); env = np.where(hit[..., None], sh, env)
        F = (.30 + .70 * (1 - np.clip(-dn, 0, 1)) ** 2.2)[..., None]; bed = self.pl[y0:y1, x0:x1] * .55 + C('#202426') * .45; col = bed * (1 - F) + env * F
        # hard light on every slope that faces the pale rift (right, far)
        l = unit((.55, .75, .55)); spec = rx * l[0] + ry * l[1] + rz * l[2]; slope = np.sqrt(gx * gx + gy * gy); k = np.clip((spec - .955) / .012 + .5, 0, 1) * np.clip((slope - .05) / .010, 0, 1); k = cv2.GaussianBlur(k.astype(f32), (0, 0), .7); col = col * (1 - k[..., None]) + C('#f3f6f6') * k[..., None]; k = np.clip((.40 - spec) / .06, 0, 1) * np.clip((slope - .12) / .05, 0, 1); col = col * (1 - k[..., None] * .62) + C('#15181a') * k[..., None] * .62
        out = self.pl.copy(); m = self.wm[y0:y1, x0:x1][..., None]
        if A > 0: dx, dy = X - bx, (Y - by) * 1.35; m = np.maximum(m, np.clip(1.45 - np.sqrt(dx * dx + dy * dy) / br, 0, .12)[..., None] / .12 * min(1., A * 3))
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - m) + col.astype(f32) * m; return out

    def bowl(self, f):
        t = (f - 32) / 24.
        return 0. if t <= 0 else 1.04 * (1 - math.exp(-t * 11) * (math.cos(t * 17) + .4 * math.sin(t * 17)))

    def scene(self, f):
        """puddle, ball, crowns, pebbles (everything a drop can see)"""
        img = self.water(f); cam = self.cam; jit = np.zeros(2) if f < 38 else R(f, 5).normal(0, 1.6, 2); bx, by, br = self.bp; bx += jit[0]; by += jit[1] - (2.2 * (f - 37) if f >= 38 else 0)
        wl = by + br * .30; bl = self.B.draw(img, bx, by, br, stars=(0,) * 4, stone=1., wet=1., line=1.1, spin=(.35, -.12, .22), light=(.45, .75, .35), shade='#c9cdcf'); yy = np.arange(H)[:, None]; xx = np.arange(W)[None]; cut = wl + 5 * np.sqrt(np.clip(1 - ((xx - bx) / (br * .98)) ** 2, 0, 1)) * 2.6; m = np.clip(cut - yy + .5, 0, 1)[..., None] * (np.hypot(xx - bx, yy - by) < br * 1.25)[..., None]; img = img * (1 - m) + bl * m
        c = Cel(); a = np.linspace(.05, math.pi - .05, 40); rx = br * .965; P = np.c_[bx + np.cos(a) * rx, wl + np.sin(a) * 13]; thin(c, P, 3.4, '#e9eeee', .95, .25, .25, 30); thin(c, P + [0, 4.5], 2.4, '#15181a', .8, .25, .25, 30)
        # crowns: frozen, then run backward and close
        for j, (x, y, v0, r0, s0, go) in enumerate(CROWNS):
            s = s0 if f < go else s0 * (1 - clamp((f - go) / 6.)) if f >= 2 else s0 * (.5 + .25 * f)
            if s > .004: splash(c, cam, s, (x, y), v0 * 2.1, r0, 11, .0065, .05, .42, 'grey', 3 + j, .45, 0., 0, ink=1.3)
        img = over(img, c.layer())
        it = []; c = Cel()
        for k, (m_, p, f0, sd, g) in enumerate(self.peb):
            s = max(0., (f - f0) / 24.); z = .5 * g * .42 * s * s + .10 * s * (s > 0) + (.0016 * math.sin(f * 2.4 + k) if f >= 22 and f < f0 and f0 - f < 3 else 0); M = axis_m([.3, .5, 1], s * (1.4 + .5 * k)) @ axis_m([1, .2, 0], s * 1.1 + k * 1.7); it.append((m_, p + np.array([0, 0, z]), M, sd, np.zeros(3)))
            if s > .02:                                       # water that clings runs off it upward
                for j in range(4): q = p + np.array([(h1(k, j) - .5) * .05, (h1(k, j, 2) - .5) * .05, max(.0, z - .01 - .03 * h1(k, j, 3) + .22 * s * s * (.5 + h1(k, j, 4)))]); drop(c, cam, q, .0035 + .003 * h1(k, j, 5), WATER['grey'])
        draw_bodies(c, cam, it, 'grey', light=(.5, .5, .8), shadow='#303537', lw=2.4, rim_dir=(.7, .6, .1)); return over(img, c.layer())

    def frame(self, f):
        img = self.scene(f); cam = self.cam; tr = f >= 38
        if f >= 32:                                           # the pressure: a rim of thrown beads, held
            A = self.bowl(f); c = Cel(); bx, by, br = BOWL
            for i in range(46): a = h1(i, 41) * TAU; s = (f - 32) / 24.; v = .35 + .5 * h1(i, 42); rr = br * 1.12 + .25 * v * s; z = .05 + v * s * (1 - .8 * s) ; drop(c, cam, (bx + math.cos(a) * rr, by + math.sin(a) * rr / 1.35, z), .0022 + .0045 * h1(i, 43) ** 2, WATER['grey'], (math.cos(a) * .4, math.sin(a) * .4, 1. * v))
            az = np.linspace(0, TAU, 220); S_ = az * 100; mk = lambda rr, z: cam.project(np.c_[bx + np.cos(az) * br * rr, by + np.sin(az) * br * rr / 1.35, np.full(220, z)])[0]; B0 = mk(1.17, .052 * A); hh = .052 * A * (1.2 if f < 35 else 1.)
            for pt, zz, am, col, off in ((17., hh, (.0, 1.), '#15181a', 1.6), (17., hh, (.0, 1.), '#dfe5e6', 0), (9., hh * .55, (.0, .7), '#8d9496', 0), (5.5, hh * .30, (.0, .6), '#f6f8f8', 0)):
                e = claw_edge(0, TAU * 100, pt, 7, .15, am); q = claw_band(S_, B0, mk(1.17 + zz * 1.1, .052 * A + zz), e[0], e[1], .0); c.fill(q + [0, off], col)
            img = over(img, c.layer())
        bg = img.copy(); gold = None if f < 38 else '#ffe9a8'
        if f < 2:
            img = over(img, self.rn.draw(cam, f / 24., 1., 'grey', warp=_above, near_soft=8.)); c = Cel(W // 2, H // 2)
            for x, y, r, *_ in HERO + NEAR: yy = y - (2 - f) * 760 + (300 if f == 1 else 0); wd = max(1.2, r * .10); c.fill(np.array([[x - 18, yy - 620 - r * 4], [x + wd, yy], [x, yy + wd], [x - wd, yy]]) / 2, '#e6ebec', .75)
            return over(img, cv2.resize(cv2.GaussianBlur(c.layer(H // 2, W // 2), (0, 0), 1.1), (W, H))).astype(f32)
        tau, v = stop_clock(f, 2, 14, 20, rise=.26, accel=1.3, ease=0); img = over(img, self.rn.draw(cam, tau, v, 'grey', warp=_above, bg=bg, bead_r=.0085, hero=17, near_soft=6.))
        lay = np.zeros((H, W, 4), f32)
        for i, (x, y, r, f0, wb, bl) in enumerate(sorted(HERO, key=lambda q: q[2])):
            st, ang, dx, dy, sc = self.hero(f, f0, i); jx, jy = R(f, i, 7).normal(0, 1.8, 2) if tr else (0, 0); Y = y + dy * (.6 + r / 130.)
            if Y < -r * 2.5: continue
            lens_drop(lay, bg, x + dx + jx, Y + jy, r * sc, st, ang, i, wob=wb + (.05 if f == 2 else .03 if f == 3 else 0) + (.03 if tr else 0), warm=gold)
        img = over(img, lay); nr = np.zeros((H, W, 4), f32)
        for i, (x, y, r, f0, bl) in enumerate(NEAR):
            st, ang, dx, dy, sc = self.hero(f, f0, 40 + i); Y = y + dy * 2.1
            if -r * 3 < Y: lens_drop(nr, bg, x, Y, r * sc, st, ang, i, blur_=bl, cel_=0, warm=gold)
        img = over(img, nr)
        if tr: s = R(f, 99).normal(0, 2.6, 2); img = cv2.warpAffine(img, np.float32([[1, 0, s[0]], [0, 1, s[1]]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return img.astype(f32)

    def hero(self, f, f0, i):
        """frozen drop: one squash on the stop, then nothing; then it sags, lets go upward and stretches.  -> stretch, tail angle, dx, dy px, scale"""
        if f <= 3: return (.93 if f == 2 else 1.07), -math.pi / 2, 0., (6. if f == 2 else -2.), 1.
        s = f - f0
        if s < -2: return 1.045, -math.pi / 2, 0., 0., 1.
        if s < 0: return 1.0 - .03 * (s + 2), -math.pi / 2, 0., 3.0 * (s + 2), 1.
        v = 1.9 * s; return 1 + min(.55, .018 * v + .02 * s), math.pi / 2, 0., 4 - .95 * s * s - 2. * s, 1 + .006 * s


def _above(P, u):
    """no drops under the water or in the rock"""
    Q = P.copy(); bad = (P[:, 2] < .03); Q[bad] = np.array([0, -30., 0]); return Q, None
