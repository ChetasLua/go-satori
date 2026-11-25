"""c2 THE PUPIL.  0-23 the eye: the moon of c1 lies where it was (960,486 r 208); the iris jumps under it by 4; the pupil closes on it, 208 -> 12 by 20, in two heartbeats (10, 20).
24-33 the hat lifts off in the wind; the ball falls past his eye, in focus, slowly.  34 stone.  It rolls at the lens and goes dark on the match circle (39)."""
import numpy as np, cv2, math
from db import W, H as FH, comp, kit, land, ball as BALL
from db.paint import rgb, f32, smooth, grid
from db.ink import Cel, spl
from db.cam import Camera3D
from chars import gohan as G, gohan_eye as GE, fkit as K
from chars.gohan import Gohan
from chars.fkit import lw, light, grade

A = np.array; keys = comp.lerp_keys; over, screen, radial, blur = comp.over, comp.screen, comp.radial, comp.blur
CX, CY, R0 = 960., 486., 208.
eo = lambda t: 1 - (1 - t) ** 3


def pupil(f):
    return float(keys(f, [(0, 208.), (4, 208.), (9, 190.), (10, 186.), (11, 118.), (12, 98.), (13, 112.), (14, 108.), (18, 92.), (19, 58.), (20, 12.), (21, 15.), (22, 12.), (40, 12.)], lambda t: t))


class Shot:
    id = 'c2'; n = 40
    finish = dict(bloom=.28, vig=.20, soft=.05, bloom_thr=.84, grain=.011)
    EVENTS = [(0, 'match', dict(note='moon disc of c1 in an eye, same place; all sound out, a held breath')), (2, 'saccade', dict(size=.15, note='a wet tick, the eye finds it by 4')), (10, 'heartbeat', dict(size=.9)), (10, 'pupil_close', dict(size=.4)), (20, 'heartbeat', dict(size=1.)), (20, 'pupil_point', dict(size=.6, note='208 -> 12')),
              (24, 'wind', dict(size=.5, note='one gust, far away')), (25, 'hat_lift', dict(size=.3, note='felt, soft')), (26, 'ball_free', dict(size=.15, note='a tiny metal click; then time slows')), (34, 'ball_stone', dict(size=.8, note='glass on rock, the only loud sound')), (36, 'ball_stone', dict(size=.3)), (36, 'ball_roll', dict(frames=3)), (39, 'ball_dark', dict(size=.3, note='the hum of the stars stops'))]

    def setup(self):
        self.E = GE.Eye(); self._p = {}

    # ------------------------------------------------------------ the eye
    def eye(self, f):
        p = pupil(f); gz = keys(f, [(0, (-66., 38.)), (1, (-66., 38.)), (2, (-30., 18.)), (3, (10., -6.)), (4, (0., 0.)), (40, (0., 0.))], lambda t: t)
        mr = float(keys(f, [(0, 208.), (1, 208.), (2, 196.), (4, 152.), (9, 146.), (10, 146.)], eo)); mr = min(mr, .80 * p); gh = float(keys(f, [(0, 1.), (2, 1.), (4, .55), (7, .10), (40, .07)]))
        b = {10: (0., 1.), 11: (0., -1.), 12: (0., 1.), 13: (0., 0.), 20: (0., 1.), 21: (0., -1.), 22: (0., 1.)}.get(f, (0., 0.))
        tr = 0. if f < 20 else .8
        img = self.E.draw(p, f, tuple(gz), tear=float(keys(f, [(0, .55), (12, .62), (23, 1.)])), moon=1., tremble=tr, mc=(CX, CY), mr=mr, ghost=gh, lash=b, eyes=float(keys(f, [(0, 1.), (4, .8), (40, .8)])), clip=f >= 2)
        hb = float(keys(f, [(0, 0), (9.9, 0), (10, 1.), (14, 0), (19.9, 0), (20, 1.2), (24, .2)])); x, y = grid(FH, W); d = np.sqrt(((x - CX) / 1150.) ** 2 + ((y - CY) / 700.) ** 2); img = img * (1 - (.20 * hb * smooth(.35, 1.1, d))[..., None])
        z = 1 + .006 * hb + .010 * f / 23.; return comp.cam2d(img.astype(f32), CX + (960 - CX) / z * 0 + 0, 540, 1.0) if abs(z - 1) < 1e-4 else self.zoom(img, z)

    def zoom(self, img, z): return cv2.warpAffine(img.astype(f32), np.float32([[z, 0, CX * (1 - z)], [0, z, CY * (1 - z)]]), (W, FH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)          # the circle never leaves its place

    # ------------------------------------------------------------ the hat and the ball
    def night(self):
        if 'n' not in self._p:
            z = float(land.ground_z(0, -20)); cam = Camera3D((0, -26, z + 1.2), (1.5, 10, z + 2.2), 1500.); im = land.ground(cam, 0., 'night', moon=None).img.astype(f32); y = grid(FH, W)[1]; im = im * A([.52, .46, .62], f32); self._p['n'] = blur(im, 8.).astype(f32)
        return self._p['n']

    def face(self, f):
        g = (f // 2) * 2 if f < 34 else 32; u = (f - 24) / 9.; x, y = grid(FH, W); L = light('night', rim=[((-6, 4), '#c68cff', 1.)]); push = float(keys(f, [(24, 0.), (34, 1.)], lambda t: t)); cx, cy, sc = 800. - 24 * push, 596. + 12 * push, 306. + 26 * push; wind = (-.95 - .25 * math.sin(f * .9), .30); dn = float(keys(f, [(24, 0.), (27, 0.), (33, 1.)])); head = (.30 + .05 * dn, .05 - .10 * dn, -.03)
        e = dict(G.E['terror'], gx=.02 * dn, gy=.02 - .075 * dn, ir=.46)
        lay = Gohan.bust(cx, cy, sc, *head, e, 'night', f, wind, 9, False, 'fear', byaw=.16, bottom=3.2); out = float(keys(f, [(24, 0.), (27, .4), (30, 3.2), (33, 5.)]))
        bg = self.night(); img = bg.copy(); lay = np.stack([blur(lay[..., q], out) for q in range(4)], -1) if out > .3 else lay; img = over(img, lay)
        # hat: shiver, lift, gone to the left
        k = keys(f, [(24, (0., 0., .0)), (25, (-6., -30., -.06)), (26, (-34., -96., -.20)), (27, (-120., -190., -.46)), (28, (-280., -300., -.82)), (29, (-500., -400., -1.15)), (30, (-760., -470., -1.5)), (32, (-1300., -560., -2.1))], lambda t: t)
        if f < 31:
            cel = Cel(W, FH); H2 = Gohan.make(cx + k[0], cy + k[1], sc, head[0] - k[2] * .5, head[1] + abs(k[2]) * .30, head[2] + k[2]); G.hat_parts(cel, H2, L, 'front', 3, .3, 0., 0., ballon=False); hl = grade(cel.layer(), L, (cx - 400, cy - 600, cx + 400, cy + 300), 1.3); sp = np.hypot(k[0], k[1])
            if sp > 100: hl = kit.smear(hl, -60 * min(1, sp / 400), -34 * min(1, sp / 400), 9)
            img = over(img, hl)
        # the ball: first on the hat, then alone in the air, slow
        H1 = Gohan.make(cx, cy, sc, *head); bx, by, br = G.ball_at(H1); t = max(0, f - 25) / 8.
        if f <= 25: kx = keys(f, [(24, (0., 0.)), (25, (-6., -30.))]); px, py, pr = bx + kx[0], by + kx[1], br
        else: px = bx - 14 + 290 * t; py = by - 62 - 70 * t + 840 * t * t * .92 + 250 * t; pr = br * (1 + .72 * t ** 1.4)
        st = float(keys(f, [(24, .9), (28, .75), (33, .45)])); img = BALL.draw(np.clip(img, 0, 4).astype(f32), px, py, pr, stars=(st,) * 4, t=f, glass=.8, spin=(.5 * t * 2, -1.1 * t * 2, .6 * t), light=(.55, .7, .2), body=.75, shade='#b9a6da')
        return comp.cam2d(img.astype(f32), 960 + 1.5 * math.sin(f), 540, 1.02)

    # ------------------------------------------------------------ stone
    def ground(self):
        if 'g' not in self._p:
            z = float(land.ground_z(0, 0)); cam = Camera3D((0, -1.6, z + .30), (.1, 4, z + .12), 1500., cy=430); o = land.ground(cam, 0., 'night', moon=None); im = o.img.astype(f32); x, y = grid(FH, W); far = smooth(470, 400, y); im = im * (1 - far[..., None]) + blur(im, 9.) * far[..., None]; im = im * A([.40, .36, .50], f32) * (1.15 - .55 * np.clip(np.abs(x - 960) / 1100., 0, 1) ** 1.6)[..., None]; self._p['g'] = im.astype(f32)
        return self._p['g']

    def stone(self, f):
        img = self.ground().copy(); x, y = grid(FH, W)
        P = {34: (1236., 566., 62.), 35: (1200., 520., 70.), 36: (1152., 566., 92.), 37: (1086., 548., 124.), 38: (1004., 516., 176.), 39: (960., 486., 208.)}[f]; px, py, pr = P; roll = {34: 0., 35: .5, 36: 1.15, 37: 2.1, 38: 3.0, 39: 3.35}[f]; air = 34. if f == 35 else 0.
        # contact shadow and the red light of the stars on the rock
        gy = py + pr + air; s = np.exp(-(((x - px + .25 * pr) / (pr * 1.15)) ** 2 + ((y - gy + .04 * pr) / (pr * .24)) ** 2)); img = img * (1 - (s * (.72 - .012 * air))[..., None]); st = {34: 1.5, 35: .9, 36: 1.0, 37: .75, 38: .42, 39: .0}[f]
        rg = np.exp(-(((x - px) / (pr * 2.1)) ** 2 + ((y - gy - .3 * pr) / (pr * .8)) ** 2)); img = img + (rg * .20 * st * (y > gy - pr * .3))[..., None] * rgb('#ff5a3a')
        img = BALL.draw(img.astype(f32), px, py, pr, stars=(st,) * 4, t=f, glass=float(keys(f, [(34, .8), (38, .4), (39, .12)])), spin=(-.35 * roll, 1.25 * roll, .2 * roll), light=(.35, .75, .1), body=float(keys(f, [(34, .75), (37, .55), (39, .10)])), shade='#9a88c4' if f < 39 else '#6f60a0')
        if f in (34, 36):                                          # glass on rock: a white tick, chips, one ring of dust
            cel = Cel(); r = np.random.default_rng(f); k = 1. if f == 34 else .5; c = A([px, gy])
            for j in range(int(16 * k) + 4): a = math.pi + math.pi * r.random(); l = pr * r.uniform(1.0, 2.6) * k; d = A([math.cos(a), math.sin(a) * .6]); n = A([-d[1], d[0]]) * pr * .05; cel.fill(A([c + d * pr * .5 + n, c + d * l, c + d * pr * .5 - n]), '#f2e8ff')
            for j in range(9): a = math.pi + math.pi * r.random(); q = c + A([math.cos(a) * pr * r.uniform(1., 2.2), math.sin(a) * pr * r.uniform(.4, 1.5)]) * k; cel.fill(q + r.normal(0, 1, (4, 2)) * pr * .07, '#2a2348')
            img = over(img, cel.layer())
            if f == 34: img = screen(img, np.clip(radial(px, gy, 300, 120, 2.2) * .5, 0, 1)[..., None] * rgb('#d8c4ff'))
        if f in (35, 37, 38):
            cel = Cel(); r = np.random.default_rng(f + 9)
            for j in range(7): q = A([1210 + r.uniform(-120, 80), 650 + r.uniform(-24, 6)]); cel.fill(K.ell(q[0], q[1], r.uniform(8, 22) * (1 + (f - 34) * .3), r.uniform(5, 11)), '#584a86', .38)
            img = over(img, cel.layer())
        sh = kit.shake(f, 5., 3, 2., 34); return comp.cam2d(img.astype(f32), 960 + (sh[0] if f < 38 else 0), 540 + (sh[1] if f < 38 else 0), 1.)

    def frame(self, f):
        if f < 24: return self.eye(f).astype(f32)
        if f < 34: return self.face(f).astype(f32)
        return self.stone(f).astype(f32)
