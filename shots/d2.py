"""d2 THE SLINGSHOT.  One travelling camera, no cut: 0 crouch, 4 launch (ground fans, dust columns), 8-18 behind Goku's shoulder between the mesas, 18 punch blocked, 22, 25 two more, 28 Vegeta's kick from the blind side on the tail,
33-38 the tail takes Goku's ankle, swings him once round and lets go, pillars at 40, 50, 60, 64 Vegeta has his wrist, 66-74 one full turn, 74 thrown, 78 impact.
Local frame of the fight: x = along the corridor (screen right in the side view), y = away from the camera, z up, metres, origin PF."""
import numpy as np, cv2, math, os
from db import W, H as HH, comp, kit, land, ki
from db.cam import Camera3D
from db.ink import Cel, spl
from db.paint import rgb, f32, fbm, smooth
from db.hero3 import *
from db.phys_kit import PillarBurst, Crater, lobes_layer, plume, dust_ring, over, grit, ring_flash, ROCK
from db.cam import voronoi_crevice
FAST = bool(os.environ.get('BD_FAST'))
TOP = land.TOP; PF = np.array([2., 244., 56.5]); FW = np.array([-64., 140., 0.]); FW /= np.linalg.norm(FW); NR = np.array([FW[1], -FW[0], 0.]); NL = -NR; UP = np.array([0, 0, 1.])
MOON = np.array([.46, .86, .235]); MOON /= np.linalg.norm(MOON); HF = math.atan2(-FW[0], FW[1])                        # heading that looks along FW
LW = lambda p: PF + FW * p[0] + NL * p[1] + UP * p[2]                                                                   # local -> world
S = lambda u: float(np.clip(u, 0, 1)) ** 2 * (3 - 2 * float(np.clip(u, 0, 1)))
PX = (-10., -43., -73.); PT = (40, 50, 60); VX0 = np.array([-84., .0, -.95])                                            # pillars, and where Vegeta waits (his ground point)
TAB = np.array([0., -22., TOP]); _pts = np.vstack([TAB + [0, 0, .9], TAB + [1, 12, 9], [5, 40, 34], [14, 96, 52], [26, 160, 57], PF - FW * 14, PF + FW * -.8]); _d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(_pts, axis=0), axis=1))]
from db.h3_body import _cr3
_LEN = float(_d[-1])


def along(s):
    """flight path from the table to the fight (pelvis), by arc length"""
    return _cr3(_pts, _d, np.array([np.clip(s, 0, _LEN)]))[0]


def sflt(f): return _LEN * float(np.clip((f - 3.2) / 13.6, 0, 1)) ** 1.62


# ------------------------------------------------------------------ who is where (local, pelvis)
def gpos(f):
    if f < 18: return None
    if f <= 33: return track(f, [(18, (-1.02, 0, 0), 'hold'), (20, (-1.02, 0, 0), 'io'), (22, (-.92, -.04, 0), 'hold'), (24, (-.92, -.04, 0), 'io'), (25, (-.90, .02, .04), 'hold'), (28, (-.95, .02, .06), 'io'), (31, (-1.15, 0, .04), 'out'), (33, (-1.35, 0, .0), 'hold')])
    if f <= 38:
        th = math.radians(180 + 300 * swing1(f)); r = 2.05 + .25 * S((f - 33) / 3.); return np.array([.42 + r * math.cos(th), r * math.sin(th), .10 + .25 * math.sin(th * .5)])
    ks = [(38, (-.73, 2.0, .22)), (40, (PX[0], .0, .10)), (50, (PX[1], .0, .0)), (60, (PX[2], .0, .05)), (64, (-85.1, .25, .45))]
    if f < 64:
        for (a, p), (b, q) in zip(ks[:-1], ks[1:]):
            if f <= b:
                u = (f - a) / float(b - a); u = 1 - (1 - u) ** 1.18 if a >= 40 else u; p, q = np.array(p), np.array(q); v = p * (1 - u) + q * u; v[2] += (.55 if a >= 40 else .0) * math.sin(math.pi * u) * (-1 if a == 50 else 1) * .6; return v
    if f <= 74: return None
    return track(f, [(74, (-83.2, -2.05, .35), 'lin'), (75, (-60., -1.4, .35), 'lin'), (76, (-26., -.6, .3), 'lin'), (77, (-2.9, -.1, .20), 'lin'), (78, (-.92, 0, .10), 'out'), (80, (-.72, 0, .10), 'hold')])


def swing1(f): return float(np.clip((f - 33) / 5., 0, 1)) ** 1.5
def swing2(f): return float(np.clip((f - 64.5) / 9.5, 0, 1)) ** 1.55          # 0..1 over the hammer turn


def theta2(f): return math.radians(180 + 450 * swing2(f))


class Shot:
    id = 'd2'; n = 80
    finish = dict(bloom=.26, vig=.18)
    EVENTS = [(0, 'crouch', dict(size=.3)), (4, 'launch', dict(size=1.2, note='two launches, ground fans open in plates, dust columns')), (8, 'flight', dict(to=17, size=.8, note='air rips past, mesas whoosh by at 9 12 14 16')), (18, 'block', dict(size=1.1, note='punch caught on a forearm, pressure ring')), (22, 'block', dict(size=.8)), (25, 'block', dict(size=.8)),
              (28, 'block', dict(size=1.1, note='heel on the tail, whip-crack, ring')), (33, 'tail_grab', dict(size=.5)), (34, 'whoosh', dict(to=38, size=.9, note='swung once round')), (38, 'whip', dict(size=1., note='tail lets go: crack')), (40, 'rock_break', dict(size=1.2)), (50, 'rock_break', dict(size=1.2)), (60, 'rock_break', dict(size=1.2)),
              (64, 'catch', dict(size=.8, note='glove closes on wrist, boots grind')), (66, 'whoosh', dict(to=73, size=1., note='one full hammer turn, pitch rises')), (74, 'throw', dict(size=1.2)), (75, 'whoosh', dict(to=77, size=1.2)), (78, 'impact', dict(size=1.6, note='fist in the face, impact frame 78 79'))]

    def setup(self):
        cv2.setNumThreads(2); G = self.G = Fig('goku', 'clean', seed=3); V = self.V = Fig('vegeta', 'clean', seed=4); Z = self.Z = Fig('frieza', seed=5)
        fl = P('fly_punch', body=(64, 6, 0), hR=(.36, -.30, -.82), hL=(.30, -.26, -.86), eR=(.5, -1, .3), cR=(1, -3), cst=(-8, 0, 0), wst=(-4, 0, 0), nck=(-26, 0, 0), hed=(-30, 0, 0))
        wd = P('punch_wind', body=(30, 0, 0), air=1, fL=(.08, .10, .10), fR=(.08, -.24, .10), tL=(6, -34, -10), tR=(6, -46, -12)); air = dict(air=1, fL=(.10, .20, .12), fR=(.07, -.32, .16), tL=(6, -30, -8), tR=(6, -46, -12), kL=(.3, 1, .3), kR=(.1, 1, -.4))
        ph = P('punch_hit', **air); ph2 = mirror(ph); kk = P('air_kick'); sw = P('swung', body=(0, 0, 0), hR=(.30, .05, .95), hL=(.42, -.05, .90), gR='splay', gL='splay', cst=(-14, 4, 6), wst=(-8, 4, 4), pel=(-6, 0, 6), nck=(-6, 8, 0), hed=(-8, 8, 0), fL=(.05, -.02, .03), fR=(.16, .10, .20))
        arch = P('hit_up', air=1, hL=(.60, .30, .10), hR=(.66, .34, .24), fL=(.10, .26, .18), fR=(.08, .14, .10)); hb = P('hang_back', body=(-22, 0, 0))
        grabd = P('thrown', body=(0, 0, 0), hR=(.10, .10, .99), hL=(.40, .30, .60), gR='fist', gL='splay', air=1, fL=(.07, .0, .05), fR=(.06, .10, .14), cst=(-10, 0, 0), wst=(-6, 0, 0), nck=(-8, 0, 0), hed=(-10, 0, 0))
        self.gk = [(0, P('launch'), 'hold'), (3, P('launch', root=(0, .02, -.245)), 'in'), (5, P('rise', body=(34, 0, 0)), 'io'), (9, fl, 'io'), (14, fl, 'io'), (16, wd, 'in'), (18, ph, 'hold'), (20, ph, 'in'), (22, ph2, 'hold'), (23, ph2, 'in'), (25, kk, 'hold'), (27, kk, 'io'), (31, P('hover_guard', body=(-10, 0, 0)), 'io'), (33, arch, 'in'), (34, sw, 'hold'), (38, sw, 'out'), (40, arch, 'hold'), (42, arch, 'io'), (46, hb, 'io'), (50, arch, 'hold'), (52, arch, 'io'), (56, P(hb, body=(-40, 20, 0)), 'io'), (60, arch, 'hold'), (62, arch, 'io'), (64, grabd, 'hold'), (74, grabd, 'out'), (76, P('fly_both', body=(0, 0, 0)), 'io'), (77, P('fly_punch', body=(0, 0, 0)), 'hold'), (80, P('fly_punch', body=(0, 0, 0)), 'hold')]
        G.anim(self.gstate); V.anim(self.vstate); Z.anim(self.zstate); Z.tail = self.tail
        self.vk = [(0, P('launch', root=(0, .03, -.235)), 'hold'), (3, P('launch', root=(0, .02, -.25)), 'in'), (5, P('rise', body=(34, 0, 0)), 'io'), (9, P('fly', body=(62, -8, 0)), 'io'), (13, P('fly_bank', body=(58, -34, 0)), 'io'), (24, P('axe_up', air=1, body=(10, 0, 0)), 'in'), (28, P('kick', air=1, body=(14, 0, 0), fL=(.06, -.06, .24), kL=(.3, 1, .4), tL=(8, -30, -8)), 'hold'), (31, P('kick', air=1, body=(14, 0, 0), fL=(.06, -.06, .24)), 'out'), (34, 'tumble', 'io'), (38, 'hang_back', 'hold')]
        self.zk = [(0, 'arms_crossed', 'hold'), (15, 'arms_crossed', 'io'), (18, P('block_high', **air, root=(0, -.02, 0), hR=(-.20, .46, .30), hL=(.30, .06, -.86)), 'hold'), (20, P('block_high', **air, root=(0, -.02, 0), hL=(-.12, .48, .24), hR=(.24, .20, -.70)), 'io'), (22, P('block_high', **air, root=(0, -.02, 0), hL=(-.12, .50, .22), hR=(.24, .20, -.70)), 'hold'), (23, P('hover'), 'io'), (25, P('block_cross', **air, root=(0, -.03, 0)), 'hold'), (28, P('hover', hL=(.30, .20, -.80)), 'hold'), (31, 'hover', 'io'), (33, P('hover', pel=(0, 0, 40), cst=(0, 0, 20)), 'io'), (38, P('hover', pel=(0, 0, -30), cst=(0, 0, -14), hL=(.6, .1, -.5), hR=(.6, .1, -.5)), 'io'), (42, 'hover', 'hold'), (72, P('conduct', **{}), 'hold'), (77, P('hover', nck=(0, 0, 0), hed=(-6, 0, 0), hL=(.5, .2, -.6), hR=(.5, .2, -.6)), 'hold'), (78, P('hit_head', air=1, nck=(-4, 10, 0), hed=(-6, 10, 0), pel=(-8, 6, 10), wst=(-4, 4, 6), cst=(-6, 6, 8), fL=(.09, .10, .10), fR=(.10, -.10, .14)), 'hold')]
        NR_ = {**ROCK['night'], **dict(lit='#5b5090', fresh='#7d70bc')}; k = dict(pal=NR_, shadow='#171030', light=(float(MOON[0]), float(MOON[1]), .5), rim=tuple(-FW * .8 + [0, 0, .2]), dl=(.55, -.60, .50))
        self.PB = []
        for i, x in enumerate(PX): p = LW((x, 0, 0)); self.PB.append(PillarBurst((p[0], p[1]), h=40., r=1.02 + .10 * i, z=float(gz(i)), dirn=tuple(-FW + [0, 0, .04]), speed=34., seed=3 + 4 * i, floor=36., cells=30, dust=.6, **k))
        p = LW(VX0); self.ST = PillarBurst((p[0], p[1]), h=float(p[2] - 20.), r=3.4, z=40., dirn=(1, 0, 0), speed=1., seed=21, floor=20., cells=4, **k)
        z = lambda x, y: float(land.ground_z(x, y)); self.gA, self.gB = np.array([-.66, -22., z(-.66, -22.)]), np.array([.66, -21.7, z(.66, -21.7)]); self.fanA, self.fanB = land.Fan((-.66, -22.), R=2.6, seed=2, z=self.gA[2]), land.Fan((.66, -21.7), R=2.4, seed=7, z=self.gB[2]); self._bg = {}

    # ------------------------------------------------------------------ states
    def gstate(self, f):
        p = seq(f if f >= 33 or f in (17, 18, 19, 21, 22, 24, 25) else on(f, 2), self.gk); d = dict(pose=p, heading=HF, pitch=0., roll=0., pivot='pelvis')
        if f < 3.3: d.update(pos=self.gA, pivot='ground', heading=0.); return d
        if f < 18:
            q = along(sflt(f)); v = along(sflt(f) + 3) - q; off = NR * (-.75 * S((f - 3) / 4.)) if f < 18 else 0; h = math.atan2(-v[0], v[1]); k = S((f - 3.2) / 3.); k = S((f - 3.2) / 1.3); d.update(pos=q * k + (self.gA + [0, 0, .75]) * (1 - k) + off * k * (1 - S((f - 13) / 5.)), heading=h); return d
        L = gpos(f)
        if f >= 64 and L is None:
            th = theta2(f); r = np.array([math.cos(th), math.sin(th) * .98, .16 * math.sin(th) + .02]); r /= np.linalg.norm(r); c = VX0 + [0, 0, 1.28]; w = FW * r[0] + NL * r[1] + UP * r[2]; d.update(pos=LW(c + r * 1.95), heading=math.atan2(w[0], -w[1]), pitch=math.pi / 2 - math.asin(r[2]) * -1 - 0.); d['heading'] = math.atan2(w[0], -w[1]); d['pitch'] = math.pi / 2; return d
        if 33 < f <= 38:
            th = math.radians(180 + 300 * swing1(f)); r = np.array([math.cos(th), math.sin(th), 0.]); w = FW * r[0] + NL * r[1]; d.update(pos=LW(L), heading=math.atan2(-w[0], w[1]), pitch=math.pi / 2); return d
        if f >= 75: d.update(pos=LW(L), heading=HF, pitch=math.pi / 2 * (1 - .25 * S((f - 77) / 1.)))
        else: d.update(pos=LW(L))
        if 18 <= f < 33: d['look'] = LW((.60, -.10, .66))
        if 40 <= f < 64: d['roll'] = .25 * math.sin((f - 40) * .55); d['pitch'] = -.25 - .25 * math.sin((f - 40) * .32)
        return d

    def vstate(self, f):
        p = seq(on(f, 2) if f < 26 else f, self.vk); d = dict(pose=p, heading=HF, pitch=0., roll=0., pivot='pelvis', reach={})
        if f < 3.3: d.update(pos=self.gB, pivot='ground', heading=0.); return d
        if f < 24:
            q = along(sflt(f)); v = along(sflt(f) + 3) - q; h = math.atan2(-v[0], v[1]); k = S((f - 3.2) / 1.3); lat = .9 + .6 * S((f - 4) / 5.) - 28 * float(np.clip((f - 11) / 7., 0, 1)) ** 2; uv = 1.2 * S((f - 8) / 4.) + 14 * float(np.clip((f - 11.5) / 6.5, 0, 1)) ** 2; d.update(pos=q * k + (self.gB + [0, 0, .75]) * (1 - k) + NR * lat * k + UP * uv, heading=h); return d
        if f <= 40:
            L = track(f, [(24, (-5.2, 3.6, 5.0), 'in'), (28, (.02, .62, 1.10), 'hold'), (31, (.02, .62, 1.10), 'out'), (34, (-2.6, 1.8, 3.6), 'lin'), (40, (-14., 6., 10.), 'lin')]); d.update(pos=LW(L), heading=HF + .5)
            if 26 <= f <= 31: d['reach'] = dict(fR=dict(p=self.tailtop(), part='ankle', w=S((f - 25) / 3.)))
            if f > 31: d['roll'] = -(f - 31) * .8; d['pitch'] = -(f - 31) * .5
            return d
        # the catch and the hammer turn: he always faces the man he holds, leans away from him, boots grinding round
        th = theta2(max(f, 60)); w = FW * math.cos(th) + NL * math.sin(th); h = math.atan2(-w[0], w[1]); k = S((f - 62) / 2.5); lean = 16 * S((f - 65) / 4.) if f < 74 else 0
        q = P(lead='root', root=(0, -.06 * k - .04 * S((f - 65) / 4), -.10 - .04 * k), pel=(10 - lean, 0, 0), wst=(4 - lean * .3, 0, 0), cst=(0 - lean * .3, 0, 0), nck=(-4, 0, 0), hed=(-8, 0, 0), hR=(-.06, .96, .16 + .10 * k), hL=(-.30, .84, .10 + .10 * k), eR=(.8, 0, -.8), eL=(.8, 0, -.8), gR='hold' if f >= 64 else 'open', gL='claw', fL=(.12, .16, .045), fR=(.14, -.14, .045), kL=(.4, 1, 0), kR=(.6, 1, 0), tL=(16, 0, 0), tR=(40, 0, 0))
        if f >= 74: q = mix_pose(q, P('punch_follow', lead='root', hR=(-.2, .9, .2), hL=(.4, -.3, .2), gR='open'), S((f - 73) / 2.))
        d.update(pose=q, pos=LW(VX0), pivot='ground', heading=h if f < 74 else HF + (1 - S((f - 73) / 2.)) * -.6); return d

    def zstate(self, f):
        L = track(f, [(0, (.55, 0, 0), 'hold'), (18, (.55, 0, 0), 'out'), (20, (.62, 0, 0), 'io'), (28, (.55, 0, 0), 'out'), (30, (.52, 0, -.05), 'io'), (34, (.42, 0, 0), 'io'), (78, (.42, 0, 0), 'out'), (80, (.75, 0, .04), 'hold')]); h = HF + math.pi
        if 33 <= f <= 38: h = HF + math.pi + math.radians(300) * swing1(f) * .5
        elif 38 < f < 46: h = HF + math.pi + math.radians(300) * .5 * (1 - S((f - 38) / 8.))
        return dict(pose=seq(f, self.zk), pos=LW(L), heading=h, pivot='pelvis', look=LW((-1.0, -1.5, .55)) if f >= 76 else LW(gpos(f)) if 18 <= f < 33 else None, reach={})

    def tailtop(self): return LW((.52, .30, 1.02))

    def tail(self, sk, f):
        b = sk.pel + sk.Rp @ np.array([0, -.07, 0]); n = 11
        if 25 <= f <= 32:                                               # rises behind him into a shield; the heel lands on it at 28
            k = S((f - 24) / 3.); t = self.tailtop(); sg = LW((.52, .30, 1.02 - .06 * math.exp(-(f - 28) / 1.5) * (f >= 28))); c = [b, b + FW * .40 + UP * .10 * k, LW((1.15, .25, .55 * k)), LW((1.0, .32, .55 + .50 * k)), sg * k + (1 - k) * LW((1.3, .3, .2)), LW((.05, .25, 1.12 * k + .1)), LW((-.25, .16, 1.02 * k))]
        elif 32 < f <= 38:                                              # takes the ankle and swings
            a = self.G.skel().an['L'] if f > 33 else LW((-1.0, 0, -.9)); m = (a + b) / 2; c = [b, b * .7 + a * .3 - UP * .16 + (NL if f < 35 else -NL) * .10, m - UP * .05, b * .25 + a * .75 + UP * .06, a, a + UP * .10 + FW * .06, a - UP * .06 - FW * .05]
        else: return tail_curve(sk, f, 'sway', 1.3)
        c = np.array(c); t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]; return _cr3(c, t, np.linspace(0, t[-1], n + 1)[1:])

    # ------------------------------------------------------------------ camera
    def shake(self, f):
        a = np.zeros(2)
        for t0, amp in ((4, 16), (18, 18), (22, 9), (25, 9), (28, 20), (38, 8), (40, 30), (50, 30), (60, 30), (64, 12), (74, 14), (78, 34)):
            if f >= t0: k = f - t0; a += np.array([math.sin(k * 2.9 + t0), math.cos(k * 2.3 + t0 * 1.7)]) * amp * math.exp(-k / 1.9)
        return a

    def camera(self, f):
        s = self.shake(f)
        if f < 18:
            q = along(sflt(f)); v = along(sflt(f) + 4) - along(max(0, sflt(f) - 4)); v = v / np.linalg.norm(v); rt = np.cross(v, UP); rt /= np.linalg.norm(rt); k = S((f - 3.7) / 2.4)
            lag = float(track(f, [(0, 0., 'lin'), (4, 2.6, 'in'), (6.5, 5.2, 'io'), (10, 2.4, 'io'), (18, 2.0, 'lin')])); sc = max(0., sflt(f) - lag); p = along(sc) - v * float(np.interp(f, [0, 4, 11], [0, 0, 0])); e = p + rt * float(track(f, [(0, 0, 'lin'), (4, .2, 'io'), (10, .60, 'io'), (18, .80, 'lin')])) + UP * float(track(f, [(0, 0, 'lin'), (4, .3, 'io'), (10, .52, 'io'), (18, .46, 'lin')]))
            t = along(min(_LEN, sc + 40.)) + FW * max(0, sc + 40. - _LEN) + rt * .4
            e0, t0 = np.array([.04, -24.9, TOP + .42]), np.array([.06, -18., TOP + 1.9]) if f < 4 else np.array([.06, -18., TOP + 1.9 + 2.6 * (f - 3.4)]); e = e0 * (1 - k) + e * k if f < 8 else e; t = t0 * (1 - k) + t * k if f < 8 else t; foc = float(track(f, [(0, 1250., 'io'), (6, 1000., 'io'), (12, 1180., 'io'), (18, 1400., 'hold')])); roll = float(track(f, [(0, .02, 'io'), (8, -.10, 'io'), (13, .14, 'io'), (18, -.03, 'hold')]))
        else:
            C, d, ph, up, foc, roll = self.orbit(f); a = math.radians(ph); t = LW(C); e = t - (FW * math.cos(a) + NL * math.sin(a)) * d + UP * up
            if f < 20: pass
        return Camera3D(e, t, foc, W, HH, cx=W / 2 + s[0], cy=HH / 2 + s[1], roll=roll)

    def orbit(self, f):
        ph = float(track(f, [(18, 14., 'in'), (21, 40., 'lin'), (25, 80., 'out'), (30, 97., 'io'), (33, 100., 'io'), (38, 92., 'io'), (48, 80., 'io'), (58, 96., 'io'), (64, 92., 'io'), (74, 80., 'io'), (78, 94., 'hold')]))
        d = float(track(f, [(18, 2.9, 'io'), (25, 4.3, 'io'), (28, 4.7, 'io'), (33, 5.4, 'io'), (38, 6.4, 'io'), (40, 8.4, 'io'), (44, 6.4, 'io'), (50, 8.2, 'io'), (54, 6.2, 'io'), (60, 8.0, 'io'), (66, 6.6, 'io'), (73, 5.6, 'io'), (77, 4.0, 'io'), (78, 2.7, 'hold')])); up = float(track(f, [(18, .45, 'io'), (25, .10, 'io'), (30, .75, 'io'), (36, 1.5, 'io'), (42, -.5, 'io'), (50, 1.1, 'io'), (60, -.3, 'io'), (66, 2.3, 'io'), (73, 1.7, 'io'), (78, -.15, 'hold')]))
        roll = float(track(f, [(18, -.03, 'io'), (25, .05, 'io'), (33, -.04, 'io'), (38, -.14, 'io'), (46, .10, 'io'), (56, -.08, 'io'), (64, .03, 'io'), (73, -.12, 'io'), (78, .07, 'hold')])); foc = float(track(f, [(18, 1400., 'io'), (25, 1780., 'io'), (30, 1760., 'io'), (36, 1640., 'io'), (41, 1780., 'io'), (58, 1820., 'io'), (64, 1780., 'io'), (73, 1640., 'io'), (78, 1760., 'hold')]))
        if f <= 38: C = track(f, [(18, (-.30, 0, .18), 'io'), (25, (-.12, 0, .25), 'io'), (29, (-.10, .1, .45), 'io'), (34, (.25, 0, .25), 'io'), (38, (.0, .3, .2), 'lin')])
        elif f <= 62:
            g = gpos(f); k = S((f - 38) / 2.); C = np.array([.0, .3, .2]) * (1 - k) + (g + [-1.15 + .35 * math.sin((f - 38) * .25), 0, .12]) * k
        elif f <= 74: k = S((f - 61) / 3.); C = (gpos(62) + [-.5, 0, .25]) * (1 - k) + (VX0 + [.25, 0, 1.25]) * k
        else: C = track(f, [(74, VX0 + [.25, 0, 1.25], 'in'), (75, np.array(gpos(75)) + [.5, .3, .15], 'lin'), (76, np.array(gpos(76)) + [.9, .2, .1], 'lin'), (77, (-.9, 0, .35), 'lin'), (78, (.02, 0, .55), 'out'), (80, (.10, 0, .55), 'hold')])
        return C, d, ph, up, foc, roll

    # ------------------------------------------------------------------ picture
    def lights(self, cam):
        l = cam.R @ MOON; light = np.array([l[0], -l[1] + .15, -l[2]]); rims = [dict(dir=(float(np.clip(l[0], -.9, .9)) * .9 + .25, .50, -.50), col='#f2e2ff', thr=.46, amt=1., face=.76), dict(dir=(-.9, -.22, -.25), col='#ff9a54', thr=.66, amt=.5, face=.58, line=.2)]
        if light[2] < -.2: light = np.array([light[0] * .8 + .1, light[1] + .2, -.10])
        return tuple(light), rims

    def back(self, cam, f):
        if FAST:
            y = np.mgrid[0:HH, 0:W][0].astype(f32)[..., None] / HH; img = (rgb('#1a1238') * (1 - y) + rgb('#5a3f9a') * y).astype(f32); c = Cel(W, HH, ss=1)
            for s_ in np.arange(0, 420, 22.):
                q = land.path(s_)
                for sg in (-1, 1):
                    b = q + NR * sg * 38; seg = cam.clip_seg(b * [1, 1, 0] + [0, 0, 0], b * [1, 1, 0] + [0, 0, 110])
                    if seg is not None: c.line(seg, max(2., min(90., 14 * cam.f / max(2., cam.cam(b)[2]))), '#2c2056', taper=(0, 0))
            return comp.over(img, c.rgba()), None
        dyn = None; fires = [(-26., 60., 60., 22., '#ff8a3c', .8), (10., 200., 40., 26., '#ff8a3c', .7)]
        if f < 10:
            u = float(np.clip((f - 3.5) / 3.2, 0, 1.15)); dyn = land.cat(self.fanA.geo(u), self.fanB.geo(u)) if u > 0 else None; o = land.ground(cam, f / 24., 'night', dyn=dyn, fires=fires)
        else: o = land.coast(cam, f / 24., 'night', fires=fires)
        return o.img, o

    def air(self, img, o, cam, f):
        """lighter, cooler air between the far rock and the men: distance first, then a soft well of light where they are"""
        if o is not None and hasattr(o, 'z'):
            k = np.clip((o.z.astype(f32) - 26.) / 150., 0, 1); k = cv2.GaussianBlur(k, (0, 0), 3)[..., None] * .27; img = img * (1 - k) + rgb('#6f86d6') * k
        p, z = cam.project(self.G.skel().pel if not 64 <= f < 75 else LW(VX0 + [0, 0, 1.2])); p = np.clip(p, [-200, -200], [W + 200, HH + 200]) if z > .2 else np.array([W / 2, HH / 2])
        g = comp.radial(float(p[0]), float(p[1]), 1150., 720., 2.0)[..., None] * .13; return (img * np.array([.90, .98, 1.05], f32)) * (1 - g) + rgb('#8ea6f0') * g

    def blurv(self, f):
        """apparent motion of the far world between this frame and the next, px at the picture centre"""
        a, b = self.camera(f), self.camera(min(79, f + 1)); P_ = a.eye + a.R[2] * 45.; p, z = b.project(P_); v = p - [a.cx, a.cy] if z > 1 else np.zeros(2); return v

    def frame(self, f):
        cam = self.camera(f); img, o = self.back(cam, f); img = img.astype(f32).copy()
        if f >= 18 and not FAST:
            v = self.blurv(f) * .55; n = float(np.hypot(*v))
            if n > 10: img = kit.smear(img, float(v[0]), float(v[1]), n=int(min(45, 11 + n // 6)))
        G, V, Z = self.G, self.V, self.Z; G.at(f); V.at(f); Z.at(f); img = self.air(img, o, cam, f); light, rims = self.lights(cam); pal = dict(base='night', mul='#b8acde', shmul='#574a98'); kw = dict(light=light, rim=rims, pal=pal, key=.12)
        if FAST: kw.update(lod=0, shadow=False, rim=None)
        show = [G] + ([V] if f < 16 or 24 <= f <= 39 or f >= 58 and f <= 75 else []) + ([Z] if 12 <= f <= 42 or f >= 76 else [])
        # ---- exact contacts
        if f in (18, 19): hit(G, 'R', Z, Z.skel().spot('forearmR') - FW * .05, part='fist', body=.6)
        if f in (22,): hit(G, 'L', Z, Z.skel().spot('forearmL') - FW * .05, part='fist', body=.6)
        if f == 25: p = Z.skel().spot('forearmL') - FW * .08; G._reach = dict(fR=dict(p=p, part='ankle'))
        if 33 <= f <= 33: pass
        if 64 <= f < 74: self.lock(G, V)
        if f >= 78: hit(G, 'R', Z, 'cheekR', push=.03 if f == 78 else .07, body=.9)
        expr = {G: dict(expr='determined' if f < 33 else 'grit' if f < 64 else 'shout' if f < 78 else 'rage'), V: dict(expr='fire'), Z: dict(expr='polite' if f < 25 else 'cruel' if f < 76 else 'surprise' if f < 78 else 'disbelief')}
        if f == 4 or f == 5: expr[G] = dict(expr='shout')
        # ---- world behind
        for i, B in enumerate(self.PB):
            t = (f - PT[i]) / 24.
            if not (PT[i] - 12 <= f <= PT[i] + 13): continue
            if f != PT[i]: c = Cel(); B.rocks(c, cam, t if f >= PT[i] else -1.); img = over(img, c.rgba())
        if 56 <= f <= 75: c = Cel(); self.ST.rocks(c, cam, -1.); img = over(img, c.rgba()); img = self.rocktop(img, cam)
        img = self.launch_fx(img, cam, f, back=True)
        L = compose([(F, dict(kw, expr=expr[F])) for F in show], cam, f); under = np.zeros_like(L)
        sm = [(17, G, 'armR'), (18, G, 'armR'), (21, G, 'armL'), (22, G, 'armL'), (24, G, 'legR'), (25, G, 'legR'), (27, V, 'legR'), (28, V, 'legR')]
        for t, F, limb in sm:
            if f == t and F in show: under = comp_over(under, F.smear(cam, f - 1.4, f, limb, pal=pal, seed=f))
        v = self.figv(cam, f, G)
        if (40 < f < 64 or 6 <= f < 16 or f in (35, 36, 37, 38)) and np.hypot(*v) > 30: under = comp_over(under, tails(L, v / np.hypot(*v) * min(230., np.hypot(*v)), n=46, seed=f))
        if 66 <= f <= 74: under = comp_over(under, self.hammer_arc(cam, f, pal))
        if 75 <= f <= 77: img = self.pan_lines(img, f)
        inside = [i for i in range(3) if f == PT[i]]
        if inside:                                                        # in the rock: the pillar's near face is over him for this one drawing
            B = self.PB[inside[0]]; c = Cel(); B.rocks(c, cam, .06); img = over(img, c.rgba()); img = over(img, under); img = over(img, L); c = Cel(); grit(c, cam, .05, B.o - B.d * .3, 70, (8., 26.), .9, (.03, .10), seed=f, dirn=B.d, floor=B.fl); img = over(img, c.rgba())
        elif f == 77: img = over(img, Z.last()); img = self.streak(img, cam, f, G.last())
        else: img = over(img, under); img = over(img, L)
        # ---- world in front
        for i, B in enumerate(self.PB):
            t = (f - PT[i]) / 24.
            if f >= PT[i] and t < .42: c = Cel(); grit(c, cam, t + .02, B.o + B.d * B.r, 90, (12., 34.), .6, (.02, .06), seed=i, dirn=B.d, floor=B.fl); (ring_flash(c, cam, B.o + B.d * B.r * 1.02, B.d, B.r, t) if t < .30 else 0); img = over(img, c.rgba()); img = B.dust(img, cam, B.lobes(t + .01))
        img = self.launch_fx(img, cam, f, back=False); img = self.rings(img, cam, f, L)
        if 10 <= f <= 17: img = self.wind_lines(img, cam, f)
        if f == 78:
            p = self.Z.info['head']; img = ki.impact(img, 1, float(p['x']), float(p['y']), subject=L, accent='#b46bff', dirn=0.)
        elif f == 79: img = img * .25 + rgb('#f4eaff') * .75; img = over(img, L); c = Cel(); kit.focus_lines(c, float(self.Z.info['head']['x']), float(self.Z.info['head']['y']), 420, 1500, n=70, w=10., col='#1a0c38', seed=3, ry=.7); img = over(img, c.rgba() * .9)
        return img

    # ------------------------------------------------------------------ pieces
    def lock(self, G, V):
        """Goku hangs from the glove: pelvis moved so that his right wrist lies in Vegeta's right hand"""
        d = V.skel().spot('cupR') - G.skel().spot('wristR'); G.pos = np.asarray(G.pos, float) + d

    def figv(self, cam, f, F):
        a, b = F.skel(f - 1).pel, F.skel(f).pel; pa, za = self.camera(f - 1).project(a) if f > 0 else cam.project(a); pb, zb = cam.project(b); pa2, z2 = cam.project(a); return (pb - pa2) if z2 > .3 else np.zeros(2)

    def hammer_arc(self, cam, f, pal):
        """the swung body as a drawn smear: strokes along the circle it has just travelled"""
        c = VX0 + [0, 0, 1.28]; r_ = np.random.default_rng(f); cel = Cel(); th1 = theta2(f); th0 = theta2(f - 1.25); pl = Pal(pal)
        if th1 - th0 < .15: return cel.rgba()
        cols = [pl('gi')[1], pl('gi')[2], pl('gi')[0], pl('boot')[1], pl('skin')[1], rgb(pal['ink'] if 'ink' in pal else '#0c0918')]
        for i in range(44):
            q = r_.uniform(1.0, 3.0); a = np.linspace(th0 + (th1 - th0) * r_.uniform(-.25, .5), th1 - .05, 16); P_ = np.stack([np.cos(a) * q, np.sin(a) * q, np.sin(a) * .16 * q + r_.uniform(-.14, .14)], -1) + c; xy, z = cam.project(PF + P_[:, :1] * FW + P_[:, 1:2] * NL + P_[:, 2:3] * UP)
            if (z < .5).any(): continue
            k = 0 if q < 2.1 else 4; col = cols[k] if r_.random() < .55 else cols[int(r_.integers(0, 3))] if q < 2.2 else cols[4]
            cel.line(xy, r_.uniform(3, 11) * cam.f / 1400. / max(.6, z.mean() / 6.), col if r_.random() > .12 else cols[5], taper=(.85, .05), alpha=.9, seed=i)
        return cel.rgba()

    def streak(self, img, cam, f, L):
        """77: one drawing of pure speed, the whole body pulled into a horizontal blade"""
        p = np.array(self.G.info['pelvis']); q = stretch(L, p, (1, 0), 2.6); T = tails(q, (420, 0), n=120, seed=3, width=(3, 16), length=(.5, 1.6)); return over(over(img, T), q)

    def rocktop(self, img, cam):
        """the flat head of the stack he stands on: plates, cracks, a worn lip, painted in true perspective"""
        c0 = LW(VX0); r = np.random.default_rng(4); t = np.linspace(0, 2 * math.pi, 15)[:-1]; R = r.uniform(3.0, 3.9, 14); P3 = np.c_[c0[0] + R * np.cos(t), c0[1] + R * np.sin(t), np.full(14, c0[2])]; q, z = cam.project(P3)
        if (z < .3).any(): return img
        poly = cv2.convexHull(q.astype(np.float32))[:, 0]; x0, y0 = np.maximum(0, np.floor(poly.min(0)).astype(int)); x1, y1 = np.minimum([W, HH], np.ceil(poly.max(0)).astype(int))
        if x1 - x0 < 4 or y1 - y0 < 4: return img
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(f32); d = np.stack([(xx - cam.cx) / cam.f, (yy - cam.cy) / cam.f, np.ones_like(xx)], -1) @ cam.R.astype(f32); tt = (c0[2] - cam.eye[2]) / np.where(np.abs(d[..., 2]) < 1e-4, -1e-4, d[..., 2]); X = cam.eye[0] + d[..., 0] * tt - c0[0]; Y = cam.eye[1] + d[..., 1] * tt - c0[1]
        e, idd = voronoi_crevice(X.astype(f32), Y.astype(f32), .85, 3); e2, id2 = voronoi_crevice(X.astype(f32) + 3, Y.astype(f32), .26, 5); col = rgb('#4b4180') * (.82 + .34 * idd[..., None]) * (.94 + .12 * id2[..., None]); col = np.where((id2 > .82)[..., None], rgb('#6a5fa6'), col); ds = np.abs(d[..., 2]) * cam.f / np.maximum(tt, .1); lw = np.clip(1.6 / np.maximum(ds, 1), .004, .06)
        col = np.where((e < lw * 1.4)[..., None], rgb('#150f30'), col); col = np.where(((e2 < lw * .6) & (idd > .4))[..., None], col * .72, col); m = np.zeros((y1 - y0, x1 - x0), np.uint8); cv2.fillConvexPoly(m, np.round((poly - [x0, y0]) * 16).astype(np.int32), 255, cv2.LINE_AA, 4); a = (m / 255.)[..., None].astype(f32); img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + col.astype(f32) * a
        c = Cel(); n = len(poly); pc = poly.mean(0)
        for i in range(n):
            a_, b_ = poly[i], poly[(i + 1) % n]; up = (a_[1] + b_[1]) / 2 < pc[1]; c.line(np.array([a_, (a_ + b_) / 2 + r.normal(0, 2, 2), b_]), 3.2 if not up else 2.2, '#d8c6ff' if up else '#0e0a22', taper=(.1, .1), seed=i)
        return over(img, c.rgba())

    def wind_lines(self, img, cam, f):
        c = Cel(); kit.focus_lines(c, cam.cx + 40, cam.cy - 30, 430, 1500, n=52, w=4.5, col='#e6d6ff', seed=on(f, 2), ry=.62); kit.focus_lines(c, cam.cx + 40, cam.cy - 30, 380, 1500, n=40, w=6., col='#0d0820', seed=on(f, 2) + 50, ry=.62); return over(img, c.rgba() * .5)

    def pan_lines(self, img, f):
        c = Cel(); r = np.random.default_rng(f)
        for i in range(90): y = r.uniform(0, HH); x = r.uniform(-200, W); l = r.uniform(300, 1500); c.line(np.array([[x, y], [x + l * .5, y + r.normal(0, 3)], [x + l, y]]), r.uniform(1.5, 9), '#f3e6ff' if r.random() < .45 else '#140c2a', taper=(.5, .5), alpha=.6, seed=i)
        return over(img, c.rgba())

    def rings(self, img, cam, f, L):
        if not hasattr(self, 'cps'): z = self.Z.skel; self.cps = [z(18).spot('forearmR') - FW * .05, z(22).spot('forearmL') - FW * .05, z(25).spot('forearmL') - FW * .08]
        ev = [(18, self.cps[0], FW, 1.0), (22, self.cps[1], FW, .7), (25, self.cps[2], FW, .7), (28, self.tailtop(), UP * .8 + FW * .4, 1.1), (38, LW((.2, 1.9, .2)), FW, .8)]
        for t0, p, ax, s in ev:
            k = f - t0
            if 0 <= k < 5:
                rad = s * (.14 + 1.8 * (1 - (1 - k / 5.) ** 2.0)); r = ki.ring3(cam, p, ax, rad, s * (.05 + .05 * min(1., k)) * (1 - k / 5.) ** 1.2 + .008, 'base' if t0 != 28 else 'violet', seed=t0, depth=.06)
                if r is not None: img = ki.put(img, r)
                if k < 2:
                    q = cam.project(p)[0]; c = Cel(); a = np.linspace(0, 2 * math.pi, 23)[:-1]; rr = np.where(np.arange(22) % 2, .05, np.random.default_rng(t0).uniform(.16, .42, 22)) * cam.scale(p) * s * (1. if k == 0 else .55); c.fill(np.c_[q[0] + rr * np.cos(a + .2 * k), q[1] + rr * np.sin(a + .2 * k)], '#fffdf6'); img = over(img, c.rgba())
        if 64 <= f <= 73:                                               # boots grinding round on the rock
            c = Cel(); r_ = np.random.default_rng(f); c0 = VX0 + [0, 0, .01]; th = theta2(f); pl = [(.34, 1.), (.40, 2.6), (.30, 4.1)]                        # scratches the boots have cut, and the grit they throw
            for j, (q, ph) in enumerate(pl):
                a = np.linspace(th + ph - min(2.6, (f - 63) * .5), th + ph, 18); P_ = c0 + np.stack([np.cos(a) * q * 1.25, np.sin(a) * q * 1.25, 0 * a], -1); xy, z = cam.project(PF + P_[:, :1] * FW + P_[:, 1:2] * NL + P_[:, 2:3] * UP)
                if (z > .3).all(): c.line(xy, 4.5 - j, '#1a1236', taper=(.7, .05), seed=j); c.line(xy + [0, -3], 1.8, '#cbbcf4', taper=(.8, .05), seed=j + 9)
            grit(c, cam, .06 + .02 * (f % 3), LW(VX0 + [0, 0, .04]), 46, (2., 7.), .5, (.012, .035), seed=f // 2, floor=float(LW(VX0)[2])); img = over(img, c.rgba())
        return img

    def launch_fx(self, img, cam, f, back):
        if not 4 <= f <= 11: return img
        t = (f - 3.6) / 24.; L_ = []
        for p, sd in ((self.gA, 1), (self.gB, 2)): L_ += plume(t, (p[0], p[1], p[2] + .1), (0, .25, 1.), 16., 34, (.25, .95), .40, .18, sd, 1.6, 3.0, .5) + dust_ring(t, tuple(p[:2]), .8, 13., .55, 34, 3, sd + 9, .9, 3.2, float(p[2]), rise=.5)
        yc = -22.2; sel = [q for q in L_ if (q[1] > yc) == back]
        return over(img, lobes_layer(cam, sel, 'night', light=(.55, -.65, .45), ss=1.25)) if sel else img


def gz(i): return LW((PX[i], 0, gpos(PT[i])[2]))[2]
def comp_over(a, b): return a * (1 - b[..., 3:4]) + b
