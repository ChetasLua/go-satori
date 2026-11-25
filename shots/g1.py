"""g1 THE BLOW.  0 a blade of gold arrives, 1 the fist is in the stomach, 4-15 dead stillness (one ripple runs out through skin and cloth, one drop falls), 16 everything the blow was worth: the rain is blown out to the horizon in a cone,
Frieza goes up, 23 Goku after him through his own rings; a five-hit climb on 30 40 50 60 70, the camera going round and round them, the storm ring above."""
import numpy as np, cv2, math, os
from db import W, H as HH, comp, kit, land, ki, sky
from db.cam import Camera3D
from db.ink import Cel, spl
from db.paint import rgb, f32, smooth
from db.hero3 import *
from db.phys_kit import RainBlast, Crater, lobes_layer, plume, dust_ring, over, grit
from db.phys_rain import lens_drop
FAST = bool(os.environ.get('BD_FAST'))
TOP = land.TOP; R_, L_ = -math.pi / 2, math.pi / 2; HITS = (30, 40, 50, 60, 70); UP = np.array([0, 0, 1.])
S = lambda u: float(np.clip(u, 0, 1)) ** 2 * (3 - 2 * float(np.clip(u, 0, 1)))
G0, Z0 = np.array([-.62, -22.]), np.array([.42, -22.])


def zf(f):
    """height of the climb above the rock (Frieza's pelvis): every hit stops the world for three drawings, then flings it on"""
    return float(track(f, [(16, 1.02, 'in'), (19, 5.2, 'lin'), (23, 20., 'lin'), (27, 39., 'out'), (30, 47., 'hold'), (32, 47.6, 'in'), (36, 72., 'out'), (40, 92., 'hold'), (42, 92.8, 'in'), (46, 126., 'out'), (50, 148., 'hold'), (52, 149., 'in'), (56, 186., 'out'), (60, 210., 'hold'), (62, 211., 'in'), (66, 250., 'out'), (70, 276., 'hold'), (73, 277., 'in'), (80, 420., 'lin')]))


def azim(f): return 180. + 360. * (S((f - 20) / 26.) * .5 + .0134 * max(0, f - 33) + .02 * S((f - 30) / 50.))          # degrees


class Shot:
    id = 'g1'; n = 80
    finish = dict(bloom=.26, vig=.18, halation='#ffe6b0')
    EVENTS = [(0, 'whoosh', dict(size=.8, note='he is simply there')), (1, 'impact', dict(size=.2, note='fist in the stomach: NO sound, everything cuts')), (4, 'silence', dict(to=15)), (12, 'drop', dict(size=.15, note='one drop leaves the chin')), (16, 'impact', dict(size=2.0, note='the blow arrives: rain blown away to the horizon')), (17, 'whoosh', dict(to=22, size=1.2, note='Frieza goes up')), (23, 'launch', dict(size=1.2, note='Goku after him')), (26, 'ring', dict(size=.7)), (28, 'ring', dict(size=.7)),
              (30, 'hit', dict(size=1., note='uppercut')), (40, 'hit', dict(size=1., note='knee')), (50, 'hit', dict(size=1.1, note='spinning heel')), (60, 'hit', dict(size=1.2, note='through the guard')), (70, 'hit', dict(size=1.6, note='both fists, he is sent into the storm ring')), (74, 'whoosh', dict(to=79, size=1.))]

    def setup(self):
        cv2.setNumThreads(2); self.z0 = float(land.ground_z(*G0)); self.z1 = float(land.ground_z(*Z0)); G = self.G = Fig('goku', 'ssj', seed=3); Z = self.Z = Fig('frieza', seed=5); self.Rg = sky.Ring(); self.Rg.wisps = lambda im, *a: im
        air = dict(air=1, fL=(.10, .18, .14), fR=(.07, -.30, .18), kL=(.3, 1, .3), kR=(.1, 1, -.4), tL=(6, -30, -8), tR=(6, -46, -12))
        self.blow = mirror(P('gut_punch', root=(-.03, .13, -.20), nck=(-10, 4, -16), hed=(-10, 4, -14))); up = mirror(P('uppercut', **dict(air, root=(0, 0, 0), fL=(.06, .10, .16), fR=(.10, -.22, .10)))); kn = P('knee', **dict(root=(0, 0, 0), air=1, fL=(.05, -.10, .08), tL=(6, -40, -10), tR=(6, -20, 0))); ks = P('kick_spin', **dict(root=(0, 0, 0), air=1, fL=(.05, .0, .10), tL=(8, -40, -10), tR=(0, -36, -8)))
        pu = P('punch_hit', **dict(air, root=(0, 0, 0))); pw = P('punch_wind', **dict(air, root=(0, 0, 0))); dbl = P('uppercut', **dict(air, root=(0, 0, 0), pel=(-10, 0, 0), wst=(-10, 0, 0), cst=(-14, 0, 0), nck=(-6, 0, 0), hed=(-16, 0, 0), hR=(-.10, .50, .78), hL=(-.10, .50, .78), eL=(.8, .6, -1), gL='fist', wL=(40, -10, 0), fL=(.07, .06, .10), fR=(.07, -.18, .10)))
        ch = P('rise', body=(0, 0, 0)); cr = P('launch', root=(0, 0, 0), air=1, body=(-20, 0, 0), fL=(.09, .10, .20), fR=(.09, -.06, .18))
        self.gk = [(0, self.blow, 'hold'), (19, self.blow, 'out'), (21, P('launch'), 'hold'), (22, P('launch', root=(0, .02, -.25)), 'in'), (24, ch, 'hold'), (27, ch, 'io'), (28, P(up, hL=(.30, -.30, -.60), root=(0, 0, -.04), pel=(14, 0, 20), wst=(8, 0, 10), cst=(8, 0, 14)), 'in'), (30, up, 'hold'), (32, up, 'io'), (34, cr, 'in'), (36, ch, 'hold'), (38, P(kn, fR=(.06, -.20, .16), pel=(10, 0, -10)), 'in'), (40, kn, 'hold'), (42, kn, 'io'), (44, cr, 'in'), (46, ch, 'hold'), (48, P('kick_wind', root=(0, 0, 0), air=1, pel=(0, -10, 100)), 'in'), (50, ks, 'hold'), (52, ks, 'io'), (54, cr, 'in'), (56, ch, 'hold'), (58, pw, 'in'), (60, pu, 'hold'), (62, pu, 'io'), (64, cr, 'in'), (66, ch, 'hold'), (68, P(dbl, hR=(-.10, .40, -.60), hL=(-.10, .40, -.60), pel=(20, 0, 0), wst=(14, 0, 0), cst=(14, 0, 0)), 'in'), (70, dbl, 'hold'), (73, P(dbl, pel=(6, 0, 0), wst=(4, 0, 0), cst=(0, 0, 0), nck=(-4, 0, 0), hed=(-8, 0, 0)), 'io'), (76, P(dbl, pel=(20, 0, 0), wst=(12, 0, 0), cst=(8, 0, 0), nck=(-10, 0, 0), hed=(-14, 0, 0)), 'hold')]
        hg = P('hit_gut', wst=(38, 0, 0), cst=(44, 2, 2), hL=(.30, .86, -.30), hR=(.36, .84, -.18)); hu = P('hit_up', **air)
        self.zk = [(0, P('hit_gut', wst=(24, 0, 0), cst=(26, 0, 0)), 'out'), (4, hg, 'hold'), (16, hg, 'out'), (18, P('hit_gut', body=(-30, 0, 0)), 'io'), (24, P('fall', body=(-50, 0, 0)), 'io'), (28, P('hang_back', body=(-26, 0, 0)), 'hold'), (30, hu, 'out'), (32, P(hu, cst=(-22, 4, 6), hed=(-40, 4, 0)), 'io'), (37, P('tumble', body=(-70, 20, 0)), 'io'), (39, P('hang_back'), 'hold'), (40, hg, 'hold'), (42, hg, 'io'), (47, P('fall', body=(-70, 30, 0)), 'io'), (49, P('hang_back', body=(-16, 0, 0)), 'hold'), (50, 'hit_spin', 'out'), (52, P('hit_spin', body=(30, 70, 0)), 'io'), (57, P('hover_guard', body=(-8, 0, 0)), 'io'), (58, P('block_cross', **air), 'hold'), (60, P('block_cross', **dict(air, hL=(.72, .30, -.10), hR=(.70, .26, .10), cst=(-8, 0, 0), nck=(-4, 8, 0), hed=(-6, 8, 0))), 'out'), (62, P('hit_head', **air), 'io'), (67, P('tumble', body=(-60, -20, 0)), 'io'), (69, P('hang_back', body=(-20, 0, 0)), 'hold'), (70, P(hu, nck=(-30, 0, 0), hed=(-40, 0, 0)), 'out'), (74, P('fall', body=(-74, 0, 0)), 'hold')]
        pu = mirror(pu); pw = mirror(pw); self.gk = [(a_, (pw if (a_ == 58) else pu if a_ in (60, 62) else b_), c_) for a_, b_, c_ in self.gk]; G.anim(self.gstate); Z.anim(self.zstate); Z.tail = self.tail; self.rb = RainBlast(np.array([.10, -22., self.z0 + 1.05]), np.array([1., .30, .16]), half=.50, c=58., pal='gold' if False else 'grey', n=1300, wind=(1.6, 0.)); self.rb2 = None; self._p = {}

    # ------------------------------------------------------------------ states
    def right(self, f): a = math.radians(azim(f)); return np.array([math.cos(a), math.sin(a), 0.])

    def center(self, f):
        if f < 16: return np.array([Z0[0], Z0[1], self.z1 + 1.0])
        k = S((f - 16) / 12.); x = Z0[0] + .9 * k; return np.array([x, Z0[1] + .0 * k, self.z1 + zf(f)])

    def zstate(self, f):
        c = self.center(f)
        if f < 16:
            p = np.array([Z0[0] - .02, Z0[1], self.z1 + .10]) + np.array([float(track(f, [(0, -.10, 'out'), (4, .0, 'hold')])), 0, float(track(f, [(0, -.10, 'out'), (4, .0, 'hold')]))]); return dict(pose=seq(f, self.zk), pos=p, heading=L_, reach={})
        h = min(HITS, key=lambda q: (abs(f - q) if f <= q + 4 else 99)); t = h if f <= h + 4 and f >= h - 6 else f
        rt = self.right(t if t > 26 else 30); hd = math.atan2(rt[0], -rt[1])                    # faces the man who comes from screen left
        return dict(pose=self.loose(seq(f, self.zk), f), pos=c, heading=hd + (.55 if 58 <= f <= 63 else 0.), pivot='pelvis', roll=.10 * math.sin(f * .7) * (f > 18) * (0 if any(q <= f <= q + 2 for q in HITS) else 1), reach={}, **(dict(look=c + UP * .30 - np.cross(UP, rt) * 2. + rt * .5, lookw=.9 * S((f - 57.) / 2.5) * S((66.5 - f) / 3.)) if 57 < f < 66.5 else {}))

    def loose(self, p, f):
        """after every blow the limbs swing on by themselves and calm down"""
        hs = [h for h in (16,) + HITS if h <= f]
        if not hs or f < 18: return p
        k = f - hs[-1]; a = math.exp(-k / 6.5) * (k > 2 or hs[-1] == 16); w = lambda ph, s=1.15: math.sin(k * s + ph) * a
        return add_pose(p, hL=(.16 * w(0), .30 * w(.6), .26 * w(1.4)), hR=(.16 * w(2.2), .30 * w(3.1), .26 * w(.2)), fL=(.02 * w(1.), .10 * w(2.5), .09 * w(.9)), fR=(.02 * w(2.), .10 * w(.4), .09 * w(3.3)), nck=(10 * w(1.8), 8 * w(.5), 10 * w(2.8)))

    def tail(self, sk, f):
        """a whip: thrown by the blow, a wave runs out to the tip and dies away"""
        b = sk.pel + sk.Rp @ np.array([0, -.07, 0]); L = 1.30 * sk.H; n = 12
        if f < 17: return tail_curve(sk, min(f, 4), 'sway', 1.)
        hs = [h for h in (16,) + HITS if h <= f]; h = hs[-1]; k = f - h; rt = self.right(f if f > 26 else 30); fw = np.cross(UP, rt); amp = .55 + 1.25 * math.exp(-k / 5.); sg = (1 if HITS.index(h) % 2 else -1) if h in HITS else 1
        th0 = {30: .9, 40: -.8, 50: -1.7, 60: 1.0, 70: .7}.get(h, .4) * math.exp(-k / 7.) * sg; p = b.copy(); out = []
        for i in range(n):
            u = (i + 1.) / n; a = th0 * (1 - u) + amp * (.25 + .95 * u) * math.sin(5.2 * u - 1.05 * k + .7 * sg); d = -UP * math.cos(a) + rt * math.sin(a); d = d + fw * .45 * math.sin(3.4 * u - .8 * k + 1.1); d = d / np.linalg.norm(d); p = p + d * L / n; out.append(p)
        return np.array(out)

    def gstate(self, f):
        if f < 23: return dict(pose=seq(f, self.gk), pos=np.array([G0[0] + float(track(f, [(0, -.05, 'out'), (2, 0., 'hold')])), G0[1], self.z0]), heading=R_, reach={}, look=np.array([Z0[0] + .05, Z0[1] - .80, self.z1 + 1.42]) if f < 17 else np.array([Z0[0] + .6, Z0[1] - .85, self.z1 + 1.2 + zf(f) * .6]))
        c = self.center(f); h = [q for q in HITS if q >= f - 3]; h = h[0] if h else 70; rt = self.right(h); hd = math.atan2(-rt[0], rt[1])
        off = {30: np.array([-.62, 0, -.22]), 40: np.array([-.50, 0, -.22]), 50: np.array([-.80, 0, .02]), 60: np.array([-.92, 0, -.10]), 70: np.array([-.56, 0, -.34])}[h]; o = rt * off[0] + UP * off[2]; lk = dict(look=c + UP * (.42 + 14. * S((f - 72.5) / 2.)) - np.cross(UP, rt) * (.55 if h == 60 else .35), lookw=float(track(f, [(23, 1., 'hold'), (70, 1., 'io'), (73, 0., 'hold')])))
        if f < 30: lag = float(track(f, [(23, 19.2, 'lin'), (26, 9.0, 'out'), (28, 3.6, 'in'), (29, 1.5, 'in'), (30, 0., 'hold')])); sd = .7
        elif f <= h + 3 and f >= h: lag, sd = 0., 0.
        else:
            prev = h - 10; lag = float(track(f, [(prev + 3, 0., 'out'), (prev + 5, 3.9, 'io'), (prev + 7, 3.6, 'in'), (h - 1, 1.2, 'in'), (h, 0., 'hold')])); sd = .45
        if f > 73: lag = float(track(f, [(73, 0., 'in'), (76, 14., 'in'), (80, 44., 'lin')])); p = c + o - UP * lag; return dict(pose=seq(f, self.gk), pos=p, heading=hd, pivot='pelvis', reach={}, **lk)
        p = c + o - UP * lag - rt * sd * min(1., lag / 3.); return dict(pose=seq(f, self.gk), pos=p, heading=hd, pivot='pelvis', reach={}, roll=(.5 * math.sin(f * 1.1) if lag > 1. and 34 < f else 0.), **lk)

    # ------------------------------------------------------------------ camera
    def shake(self, f):
        a = np.zeros(2)
        for t0, amp in ((16, 42), (23, 16), (30, 22), (40, 22), (50, 24), (60, 26), (70, 40)):
            if f >= t0: k = f - t0; a += np.array([math.sin(k * 2.7 + t0), math.cos(k * 2.2 + t0)]) * amp * math.exp(-k / 2.2)
        return a

    def lagof(self, f): return float(max(0., self.center(f)[2] - self.gstate(f)['pos'][2])) if f >= 23 else 0.

    def camera(self, f):
        s = self.shake(f); e0, t0 = np.array([-.10, -22. - 3.60, self.z0 + .62]), np.array([-.04, -22., self.z0 + 1.00])
        if f < 17: return Camera3D(e0, t0, 2180., W, HH, cx=W / 2 + s[0], cy=HH / 2 + s[1], roll=.015)
        c = self.center(f); look0 = t0 * (1 - S((f - 17) / 3.)) + np.array([c[0] * .8, c[1], min(c[2] - .5, self.z0 + 40.)]) * S((f - 17) / 3.)           # on the ground, head thrown back after him
        a = math.radians(azim(f)); ek = [(22, -62, 'io'), (27, -60, 'io'), (30, -14, 'io'), (32, -6, 'io'), (36, -58, 'io'), (40, 10, 'io'), (42, 12, 'io'), (46, -62, 'io'), (50, -16, 'io'), (52, -12, 'io'), (56, -60, 'io'), (59, -6, 'io'), (63, -4, 'io'), (66, -62, 'io'), (70, -26, 'io'), (72, -30, 'io'), (76, -58, 'io'), (80, -68, 'hold')]
        el = math.radians(float(track(f, ek))); d = float(track(f, [(22, 4.4, 'io'), (28, 4.6, 'io'), (31, 3.8, 'io'), (36, 5.0, 'io'), (40, 3.9, 'io'), (46, 5.0, 'io'), (50, 4.0, 'io'), (56, 5.0, 'io'), (59, 2.3, 'io'), (63, 2.15, 'io'), (66, 5.0, 'io'), (70, 3.9, 'io'), (74, 4.4, 'io'), (80, 6.0, 'hold')]))
        if f < 30: g = self.gstate(max(f, 23))['pos']; mid = g + UP * min(2.2, .55 * max(0., c[2] - g[2]))
        elif f < 73: kk = S((f - 56.) / 2.) * S((66. - f) / 2.) * (1 - .8 * max(0., 1 - abs(f - 59.))); mid = c - UP * .52 * (1 - .86 * kk) * self.lagof(f) + (UP * .52 - self.right(60) * .30) * S((f - 56.5) / 2.5) * S((65.5 - f) / 2.5)
        else: mid = self.gstate(f)['pos'] + UP * float(track(f, [(73, .30, 'io'), (80, 3.2, 'hold')]))
        fw = np.array([-math.sin(a), math.cos(a), 0.]); e = mid - fw * d * math.cos(el) + UP * d * math.sin(el); roll = .20 * math.sin((f - 20) * .17); k = S((f - 22.6) / 2.6); ee = e0 * (1 - k) + e * k; tt = look0 * (1 - k) + mid * k; foc = 2180. + (1180. - 2180.) * S((f - 17) / 4.) + 260. * S((f - 28) / 4.)
        return Camera3D(ee, tt, foc, W, HH, cx=W / 2 + s[0], cy=HH / 2 + s[1], roll=.015 * (1 - k) + roll * k)

    # ------------------------------------------------------------------ picture
    def back(self, f, cam):
        if FAST:
            y = np.mgrid[0:HH, 0:W][0].astype(f32)[..., None] / HH; img = (rgb('#101c1a') * (1 - y) + rgb('#39514a') * y).astype(f32); c = Cel(W, HH, ss=1); t = np.linspace(0, 2 * math.pi, 60)
            for r, z in ((420, 600), (700, 640)):
                q, zz = cam.project(np.c_[r * np.cos(t), -22 + r * np.sin(t), np.full(60, z)]); [c.line(q[i:i + 2], 14, '#c9b060', taper=(0, 0)) for i in range(59) if zz[i] > 5 and zz[i + 1] > 5]
            q, zz = cam.project(np.c_[4000 * np.cos(t), 4000 * np.sin(t), np.zeros(60)]); [c.line(q[i:i + 2], 5, '#8fae9a', taper=(0, 0)) for i in range(59) if zz[i] > 5 and zz[i + 1] > 5]
            return comp.over(img, c.rgba())
        g = 0 if f < 16 else f
        if g in self._p: return self._p[g]
        cam = self.camera(0) if f < 16 else cam; fn = 'cache/g1_p1_%02d.npy' % g
        if os.path.exists(fn): d = np.load(fn).astype(f32)
        else:
            sk, _ = self.Rg.frame(cam, g, k=60. + 420. * S((g - 16) / 10.), gold=.25 + .75 * S((g - 16) / 6.)); p = self.gstate(g)['pos']; lamp = (float(p[0]), float(p[1]), float(p[2]) + .9, 14. if g < 24 else 40., '#ffd56a', 1.2)
            o = land.ground(cam, g / 24., 'storm', sky=sk, fires=[lamp], rings=[(0., -22., 60. + 26. * (g - 16), 60., 3.)] if g >= 16 else ()) if g < 24 else land.coast(cam, g / 24., 'storm', sky=sk, rings=[(0., -22., 60. + 26. * (g - 16), 60., 3.)]); d = o.img; np.save(fn, d.astype(np.float16)) if g < 16 else None
        self._p.clear(); self._p[g] = d; return d

    def frame(self, f):
        cam = self.camera(f); img = self.back(f, cam).astype(f32).copy()
        if f < 17:
            s = self.shake(f); img = cv2.warpAffine(img, np.float32([[1, 0, s[0]], [0, 1, s[1]]]), (W, HH), borderMode=cv2.BORDER_REPLICATE)
        fr = f if f < 4 else 4 if f < 16 else f                                        # the drawing that holds
        G, Z = self.G, self.Z; G.at(fr); Z.at(fr); near = [h for h in HITS if h <= fr <= h + 2]
        if 1 <= fr < 17: hit(G, 'L', Z, 'stomach', push=.02 if fr == 1 else .085, body=.5)
        elif near:
            h = near[0]
            if h == 30: hit(G, 'L', Z, 'chin', push=.02, body=.8)
            elif h == 40: G._reach = dict(fR=dict(p=Z.spot('stomach') - UP * .32 + self.right(40) * .02, part='ankle', pole=UP + self.right(40) * .6))
            elif h == 50: p = Z.skel(); G._reach = dict(fR=dict(p=p.sh['R'] * .5 + p.sh['L'] * .5 - self.right(50) * .12, part='ankle', R=None))
            elif h == 60: hit(G, 'L', Z, 'cheekL' if False else 'cheekR', push=.05, body=.9)
            else: hit(G, 'R', Z, 'stomach', push=.04, body=.6); G._reach['L'] = dict(p=Z.spot('stomach') + np.array([0, 0, .07]), part='fist', body=.2)
        warm = dict(base='gold', mul='#ffe6ae', shmul='#b87a48'); cold = dict(base='storm', mul='#bcc7c2', shmul='#4d5c60')
        kg = dict(light=(.15, .80, .45), pal=warm, rim=[dict(dir=(.85, .35, -.35), col='#fff6cc', thr=.54, amt=.95, face=.68)], key=.06)
        a = G.skel().chest - Z.skel().chest; l = cam.R @ (a / (np.linalg.norm(a) + 1e-9)); kz = dict(light=(float(l[0]) * .8, .45 - float(l[1]) * .6, .40), pal=cold, rim=[dict(dir=(float(l[0]) * .9, float(-l[1]) * .9 + .1, -.25), col='#ffd76a', thr=.40, amt=1., face=.74), dict(dir=(-float(l[0]), .4, -.5), col='#b9d8d0', thr=.62, amt=.5)], key=.14)
        if FAST: kg.update(lod=0, shadow=False, rim=None); kz.update(lod=0, shadow=False, rim=None)
        eg = dict(expr='scream_b' if f < 17 else 'rage' if f < 24 else 'scream_b' if f in (30, 31, 40, 41, 50, 51, 60, 61, 70, 71, 72) else 'grit', hair='ssj'); ez = dict(expr='disbelief' if f < 16 else 'fear' if f < 56 else 'rage' if f < 60 else 'disbelief')
        L = compose([(Z, dict(kz, expr=ez)), (G, dict(kg, expr=eg))], cam, fr); gl = G.last(); hd = G.info['head']; face = (hd['x'], hd['y'], hd['s'] * (2.0 if f < 24 else 1.7))
        under = np.zeros_like(L)
        if f == 0: under = G.smear(cam, dict(G.state(0), x=G0[0] - 2.6), G.state(1), 'armL', pal=warm, seed=3); T = tails(gl, (420, -20), n=120, seed=2, width=(3, 14), length=(.4, 1.3)); under = under * (1 - T[..., 3:4]) + T
        for h in HITS:
            limb = 'armL' if h in (30, 60) else 'armR' if h == 70 else 'legR'
            if f in (h - 1, h): sb = G.state(f); q = G.smear(cam, dict(G.state(f - 1.5), x=sb['x'], y=sb['y'], z=sb['z']), sb, limb, pal=warm, seed=h + f); under = under * (1 - q[..., 3:4]) + q
        pw = .85 if f < 16 else .55 if f < 24 else 1.0; af = fr if f < 16 else f
        if FAST: img = over(img, under); img = over(img, L)
        else:
            up = (0., -1.) if f < 23 else self.flow(cam, f); A = ki.aura(gl, af, 'gold', pw * (.55 if f >= 23 else .8), seed=2, height=.42, up=up, face=face, ground=f < 23)
            img = A.under(img) if A.back is not None else img; img = over(img, under); img = over(img, L); img = A.over(img) if A.back is not None else img
        img = self.rings(img, cam, f)
        if f < 16: img = self.still(img, f, L, cam)
        elif f == 16: img = ki.impact(img, 3, float(G.info['fists']['L'][0]), float(G.info['fists']['L'][1]), subject=L, accent='#ffb01e')
        elif f < 26:
            t = (f - 16) / 24.; img = self.rb.draw(img, cam, t, 4 / 24. + t); img = self.burst(img, cam, f)
        if f == 70: img = ki.impact(img, 2, float(G.info['fists']['R'][0]), float(G.info['fists']['R'][1]), subject=L, accent='#ffb01e', dirn=-1.2)
        if f >= 24 and f != 70: img = self.rush(img, cam, f)
        return img

    # ------------------------------------------------------------------ pieces
    def flow(self, cam, f):
        a = cam.project(self.center(f))[0]; b = cam.project(self.center(f) + UP * .5)[0]; d = a - b; n = np.hypot(*d); return tuple(-(-d / n)) if n > 1 else (0., 1.)

    def still(self, img, f, L, cam):
        """nothing moves: rain hangs; from 5 one ripple leaves the fist through skin and cloth; a drop falls from the chin"""
        if f < 4: return self.rb.draw(img, cam, -1., f / 24.)
        p = np.array(self.G.info['fists']['L'])
        if 5 <= f <= 13:
            rad = 34. + (f - 5) * 74.; w = ki.refract_ring(img, float(p[0]), float(p[1]), rad, 46., 15. * (1 - (f - 5) / 10.)); m = np.clip(L[..., 3:4], 0, 1); img = img * (1 - m) + w * m
            c = Cel(); t = np.linspace(0, 2 * math.pi, 60); ring = np.c_[p[0] + rad * np.cos(t), p[1] + rad * np.sin(t)]; c.line(ring, 3.2 * (1 - (f - 5) / 10.), '#fff4d0', closed=True, taper=(0, 0)); q = c.rgba() * m; img = over(img, q)
        rl = self.rb.rn.draw(cam, 4 / 24., .05, 'grey', bg=img, bead_r=.012); img = over(img, rl)
        ch = np.array(self.Z.info['head']['x']), np.array(self.Z.info['head']['y']); hs = self.Z.info['head']['s']; x, y = float(ch[0]) - hs * .72, float(ch[1]) + hs * .55
        g = max(0., f - 10.); lay = np.zeros((HH, W, 4), f32); lay = lens_drop(lay, img, x, y + .5 * 5.2 * g * g, 13., st=1.0 + .05 * g, ang=math.pi / 2, seed=3, warm='#ffd76a'); return over(img, lay)

    def burst(self, img, cam, f):
        c = Cel(); t = (f - 16) / 24.; a = np.array([.15, -22., self.z0 + 1.05]); grit(c, cam, t, a, 160, (8., 46.), .4, (.008, .03), seed=5, dirn=(1, .25, .15), floor=self.z0); img = over(img, c.layer())
        if f >= 23: pass
        return img

    def rings(self, img, cam, f):
        ev = [(16, np.array([.16, -22., self.z0 + 1.05]), np.array([1., .2, .1]), .75), (23, np.array([G0[0], G0[1], self.z0 + .3]), UP, 1.6), (25, None, UP, 1.0), (27, None, UP, 1.0)] + [(h, self.center(h) - UP * .1, UP if h != 60 else self.right(60), 1.25) for h in HITS] + [(h + 5, self.center(h) - UP * 2.5, UP, 1.1) for h in HITS[:-1]] + [(74, self.center(73) + UP * 1., UP, 1.4), (76, self.center(73) + UP * 7, UP, 1.4)]
        for t0, p, ax, s in ev:
            k = f - t0
            if p is None: p = self.gstate(t0)['pos']
            if 0 <= k < 7 and f != 16:
                rad = s * (.28 + 2.4 * (1 - (1 - k / 7.) ** 2.0)); r = ki.ring3(cam, p, ax, rad, s * .10 * (1 - k / 7.) ** 1.2 + .012, 'gold', seed=t0, depth=.05)
                if r is not None: img = ki.put(img, r)
            if t0 in HITS and 0 <= k < 2 and t0 != 70:
                q = np.array(self.G.info['fists']['L'] if t0 in (30, 60) else self.G.info['knees']['R'] if t0 == 40 else self.G.info['feet']['R']); c = Cel(); a = np.linspace(0, 2 * math.pi, 23)[:-1]; rr = np.where(np.arange(22) % 2, 30, np.random.default_rng(t0).uniform(120, 300, 22)) * (1. if k == 0 else .5); c.fill(np.c_[q[0] + rr * np.cos(a + .2 * k), q[1] + rr * np.sin(a + .2 * k)], '#fffdf0'); m = np.clip(self.G.mask('handL', 'farmL') * (t0 in (30, 60)), 0, 1)[..., None]; img2 = over(img, c.rgba()); img = img2 * (1 - m) + img * m
        return img

    def rush(self, img, cam, f):
        """the air itself goes down past us: world streaks with true parallax"""
        v = (zf(f + .5) - zf(f - .5)); c = Cel(); r = np.random.default_rng(3); C = self.center(f); n = 120; ln = min(9., .30 * abs(v) + .05)
        if abs(v) < .6: return img
        for i in range(n):
            x, y = r.uniform(-7, 7, 2); z = (r.uniform(0, 16) - zf(f)) % 16 - 8; a = C + [x, y, z]; s = cam.clip_seg(a, a + UP * ln)
            if s is None or cam.cam(a)[2] < 1.2: continue
            c.line(s, float(np.clip(5.5 / cam.cam(a)[2] * 2.2, .8, 5.)), '#fff2c4' if i % 3 else '#0c1412', taper=(.45, .45), alpha=.42, seed=i)
        return over(img, c.rgba())
