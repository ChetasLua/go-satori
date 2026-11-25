"""d1 TWO BOOTS.  0 an orange trouser and a blue boot slam down; 10 a white boot beside it; 20 a weighted wristband craters the rock between them; 26-35 tilt up: two backs, the 悟, the armour, the moon small above them."""
import numpy as np, cv2, math, os
from db import W, H as HH, comp, kit, land
from db.cam import Camera3D
from db.ink import Cel, spl
from db.paint import rgb, f32, fbm, smooth
from db.hero3 import *
from db.phys_kit import BootPlant, Crater, lobes_layer, plume, dust_ring, over, grit, tw
from db import moon as MN
TOP = land.TOP
GX, GY, VX, VY = -.50, -21.95, .56, -21.78                       # where the two men stand
BX, BY = .04, -21.92                                             # where the band lands
EYE = np.array([.03, -23.24, TOP + .20])
KEY, RIM = (.62, .55, -.10), [dict(dir=(.46, .62, -.62), col='#f2e2ff', thr=.46, amt=1., face=.74), dict(dir=(-.92, -.10, -.22), col='#ff9a54', thr=.64, amt=.55, face=.60, line=.2)]
MOON_DIR = np.array([.105, .60, .655]); MOON_DIR /= np.linalg.norm(MOON_DIR)

S_ = lambda u: float(u) * float(u) * (3 - 2 * float(u))


def tilt(f):
    """camera pitch in degrees: 0 at the boots, up to the backs; anticipates 1 degree down, overshoots, settles"""
    return float(track(f, [(0, 8.4, 'lin'), (23, 8.4, 'io'), (25, 6.8, 'in'), (34, 34.0, 'out'), (37, 32.6, 'io'), (40, 32.3, 'hold')]))


def shake_(f):
    a = np.zeros(2)
    for t0, amp in ((0, 15), (10, 13), (20, 20)):
        if f >= t0: k = f - t0; a += np.array([math.sin(k * 2.7 + t0) * .5, math.cos(k * 3.3 + t0)]) * amp * math.exp(-k / 2.1)
    return a


def camera(f):
    p = math.radians(tilt(f)); s = shake_(f); dl = track(f, [(0, (0, .10, 0), 'out'), (23, (0, 0, 0), 'io'), (36, (-.03, -.60, .06), 'io'), (40, (-.03, -.63, .06), 'hold')])
    e = EYE + dl; return Camera3D(e, e + np.array([.012, math.cos(p), math.sin(p)]), 1330., W, HH, cx=W / 2 + s[0], cy=HH / 2 + s[1], roll=-.018 + .0015 * s[0])


def band(cel, cam, c, ax, r, h, pal, seed=0):
    """Goku's weighted wristband: a thick blue ring of cloth (hollow), lit by the moon"""
    ax = np.asarray(ax, float) / np.linalg.norm(ax); e1 = np.cross(ax, [.3, .9, .2]); e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1); t = np.linspace(0, 2 * math.pi, 49)
    ring = lambda k, q: cam.project(c + ax * k * h + (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2) * r * q)[0]
    to = cam.eye - c; s = 1. if ax @ to > 0 else -1.; A, B, I = ring(.5 * s, 1.), ring(-.5 * s, 1.), ring(.5 * s, .74); hull = cv2.convexHull(np.vstack([A, B]).astype(np.float32))[:, 0].astype(float)
    lit, base, sh, dp = (pal('band')[i] for i in range(4)); w = max(1.4, .006 * cam.scale(c))
    kit.outline(cel, hull, w * 2.2, pal.ink, seed=seed); cel.fill(hull, base); cel.clip(hull); m = hull.mean(0); d = np.array([.6, -.5]); cel.fill(hull - d * np.ptp(hull, 0).max() * .30, sh); cel.unclip()
    cel.fill(A, lit); cel.fill(I, dp); cel.line(A, w * .9, pal.ink, closed=True, taper=(0, 0), seed=seed + 1); cel.line(I, w * .8, pal.ink, closed=True, taper=(0, 0), seed=seed + 2)
    q = ring(.2 * s, 1.01); v = ((q - q.mean(0)) @ (B.mean(0) - A.mean(0))) > 0
    if v.sum() > 4: cel.line(q[v], w * .7, pal.ink, taper=(.2, .2), seed=seed + 3)


class Shot:
    id = 'd1'; n = 40
    finish = dict(bloom=.28, vig=.20)
    EVENTS = [(0, 'impact', dict(size=.9, note='blue boot plants, rock cracks', x=.36)), (2, 'debris', dict(size=.4, note='pebbles hop and settle')), (10, 'impact', dict(size=.8, note='white boot plants beside it', x=.62)), (12, 'debris', dict(size=.4, note='pebbles')),
              (12, 'whoosh', dict(size=.25, note='band falls, 8 frames')), (20, 'impact', dict(size=1.3, note='weighted wristband craters the rock, very low thud', x=.5)), (22, 'debris', dict(size=.8, note='stones jump and land until 36')), (26, 'whoosh', dict(size=.6, note='camera tilts up')), (30, 'wind', dict(size=.7, note='wind takes cloth and hair')), (30, 'hit', dict(size=.5, note='two backs, the mark'))]

    def setup(self):
        cv2.setNumThreads(2)
        z = lambda x, y: float(land.ground_z(x, y)); self.zg, self.zv, self.zb = z(GX + .2, GY), z(VX - .2, VY), z(BX, BY)
        self.cache = {}
        k = dict(pal='night', light=(.55, .35, .75), rim=(-.8, .3, .15), dl=(.55, -.70, .40), top='#6a5aa6')
        self.P1 = BootPlant((GX + .20, GY - .30), size=.25, seed=1, floor=self.zg, **k); self.P2 = BootPlant((VX - .20, VY - .12), size=.25, seed=4, floor=self.zv, **k); self.K = Crater((BX, BY), rad=.40, power=.55, seed=2, floor=self.zb, n=60, **k)
        G = self.G = Fig('goku', 'clean', seed=1); V = self.V = Fig('vegeta', 'clean', seed=2)
        base = P('fists_down', fL=(.12, .16, .045), fR=(.115, -.17, .045), kL=(.35, 1, 0), kR=(.45, 1, 0), tL=(20, 0, 0), tR=(34, 0, 0), hL=(.34, .05, -.88), hR=(.36, .02, -.88), cst=(-4, 0, 0), nck=(2, 0, 0), hed=(-4, 0, 0))
        dn = lambda p, a: add_pose(p, root=(0, 0, -.055 * a), pel=(6 * a, 0, 0), wst=(3 * a, 0, 0), cst=(3 * a, 0, 0))
        G.anim(lambda f: dict(pose=dn(base, float(track(f, [(-3, .0, 'in'), (0, .5, 'out'), (2, 1., 'io'), (9, .12, 'io'), (20, .10, 'out'), (22, .30, 'io'), (28, 0., 'io'), (40, 0., 'io')]))), pos=(GX, GY, self.zg), heading=-.02, reach=dict(fR=dict(p=self.footG(f), part='sole')), wind=self.wind(f)))
        vb = P('fists_down', fL=(.10, -.10, .045), fR=(.13, .10, .045), kL=(.4, 1, 0), tL=(26, 0, 0), tR=(30, 0, 0), hL=(.40, .02, -.84), hR=(.40, .04, -.84), eL=(.9, -1, 0), eR=(.9, -1, 0), cL=(2, -1), cR=(2, -1), cst=(-6, 0, 0), nck=(0, 0, 0), hed=(-6, 0, 0))
        V.anim(lambda f: dict(pose=dn(vb, float(track(f, [(6, .0, 'in'), (10, .5, 'out'), (12, 1., 'io'), (19, .1, 'io'), (20, .1, 'out'), (22, .25, 'io'), (28, 0., 'io'), (40, 0., 'io')]))), pos=self.rootV(f), heading=.03, reach=dict(fL=dict(p=self.footV(f), part='sole')), wind=self.wind(f)))
        self.n1 = fbm(HH // 2, W // 2, (230, 110), (1, .45), seed=11); self.n2 = fbm(HH // 2, W // 2, (170, 80), (1, .4), seed=12)

    # ---- motion
    def footG(self, f):
        u = float(np.clip((f + 4) / 4., 0, 1)) ** 1.6; a = np.array([GX + .34, GY - .95, self.zg + .62]); b = np.array([GX + .20, GY - .30, self.zg]); return a * (1 - u) + b * u

    def rootV(self, f):
        u = S_(np.clip((f - 3) / 8., 0, 1)); return np.array([VX + .95 * (1 - u), VY - .10 * (1 - u), self.zv])

    def footV(self, f):
        u = float(np.clip((f - 5) / 5., 0, 1)) ** 1.7; a = np.array([VX + .74, VY - .22, self.zv + .24]); b = np.array([VX - .20, VY - .12, self.zv]); p = a * (1 - u) + b * u; p[2] += .12 * math.sin(math.pi * u) * (u < 1); return p

    def wind(self, f):
        k = float(track(f, [(0, .25, 'io'), (24, .25, 'io'), (30, .95, 'out'), (34, 1.25, 'io'), (40, .9, 'io')])); return (-.95 * k, .25 * k, .16 * k)

    def band_at(self, f):
        """free fall from his hand: released at 11, on the rock at 20"""
        if f < 11: return None
        t0, t1 = 11., 20.; h = .5 * 9.81 * ((t1 - t0) / 24.) ** 2 * 1.35
        if f < t1: u = (f - t0) / (t1 - t0); return np.array([BX - .10 * (1 - u), BY + .10 * (1 - u), self.zb + .045 + h * (1 - u * u)]), (math.sin(u * 1.1) * .5, .25, 1.)
        q = min(1., (f - t1) / 5.); return np.array([BX, BY, self.zb - .035 + .012 * math.sin(q * 9) * math.exp(-q * 4)]), (.16, .10, 1.)

    # ---- plates
    def plate(self, f):
        g = f if f >= 24 else 0; cam = camera(g) if f >= 24 else Camera3D(EYE, EYE + np.array([.012, math.cos(math.radians(8.4)), math.sin(math.radians(8.4))]), 1330., W, HH, roll=-.018)
        key = g
        if key in self.cache: return self.cache[key]
        fn = 'cache/d1_p4_%02d.npy' % key
        if os.path.exists(fn): d = np.load(fn)
        else:
            c0 = cam if f < 24 else Camera3D(cam.eye, cam.eye + cam.R[2], 1330., W, HH, roll=0.); 
            if f >= 24: c0 = cam
            o = land.ground(c0, t=g / 24., time='night', moon=None, fires=[(-26., -2., TOP + 1., 12., '#ff8a3c', .8)]); d = o.img.astype(np.float16); sky = (o.z > 1e5)
            img = d.astype(f32); self.disc(img, c0, sky); d = img.astype(np.float16); os.makedirs('cache', exist_ok=True); np.save(fn, d)
        if len(self.cache) > 3: self.cache.clear()
        self.cache[key] = d; return d

    def disc(self, img, cam, sky):
        p, z = cam.project(cam.eye + MOON_DIR * 1000.)
        if z < 0: return
        x, y = np.meshgrid(np.arange(W), np.arange(HH)); d = (np.hypot(x - p[0], y - p[1])).astype(f32); r = 58.; h = MN.halo(d - r, r * 2.2)[..., None] if False else np.maximum(0, MN.halo(np.maximum(d - r, 0), r * 2.0))[..., None]
        img += MN.halo_rgb(h[..., 0]) * .75 * (sky[..., None] * .75 + .25); m = np.clip(r - d + .5, 0, 1)[..., None] * sky[..., None]; img[:] = img * (1 - m) + rgb('#fbf7ff') * m

    def smoke(self, img, f, amt=1.):
        """what c4 left behind: torn smoke that slides off to the left and opens; two hard tones"""
        a = float(track(f, [(0, 1., 'out'), (9, .62, 'io'), (24, .34, 'io'), (36, .0, 'io')]))
        a *= amt
        if a < .02: return img
        y = np.linspace(0, 1, HH // 2)[:, None]; out = img
        for n_, sp, col, thr, k in ((self.n1, 5.0, '#2c2350', .52, .95), (self.n2, 11.0, '#4a3a86', .62, .80)):
            m = np.roll(n_, int(-f * sp) % (W // 2), 1); m = (m - m.min()) / np.ptp(m); lv = thr + (1 - a) * .55 + .16 * np.abs(y - .45) - .10 * (1 - y) * a; q = smooth(lv, lv + .03, m) * (.55 + .45 * smooth(lv + .07, lv + .10, m)); q = cv2.resize((q * k * min(1., a * 1.4)).astype(f32), (W, HH))[..., None]; out = out * (1 - q) + rgb(col) * q
        return out

    def frame(self, f):
        a_ = float(track(f, [(24, .20, 'io'), (34, .10, 'hold')])); gl = comp.radial(980., 520., 1200., 620., 2.0)[..., None] * a_
        g = f if f in (0, 1, 2, 10, 11, 20, 21) or 24 <= f <= 36 else on(f, 2)             # hits and the camera move on ones, the men on twos
        cam = camera(f); img = self.plate(f).astype(f32) * np.array([.90, .98, 1.04], f32); img = img * (1 - gl) + rgb('#8ea6f0') * gl
        if f < 24:
            s = shake_(f); M = np.float32([[1, 0, s[0]], [0, 1, s[1]]]); img = cv2.warpAffine(img, M, (W, HH), borderMode=cv2.BORDER_REPLICATE)
        t1, t2, t3 = f / 24. + .02, (f - 10) / 24. + .02, (f - 20) / 24. + .02
        tx = lambda t: t if t > .02 - 1e-6 else -1.
        low = tilt(f) < 24                                                                 # effects on the ground leave the frame with the tilt
        if low:
            img = self.K.under(img, cam, tx(t3)); img = self.pback(img, cam, self.P2, tx(t2)); img = self.pback(img, cam, self.P1, tx(t1)); img = self.K.back(img, cam, tx(t3))
        self.G.at(g); self.V.at(g); pal = dict(base='night', mul='#b8acde', shmul='#5a4c9a')
        kw = dict(light=KEY, pal=pal, rim=RIM, key=.14)
        it = [(self.G, dict(kw, expr=dict(expr='serious')))] + ([(self.V, dict(kw, expr=dict(expr='pride')))] if f >= 4 else []); L = compose(it, cam, g)
        img = self.smoke(img, f)
        if f in (0, 1): img = over(img, self.G.smear(cam, -3, 0, 'legR', pal=pal, seed=3, tail=.95))
        if 7 <= f <= 10: img = over(img, self.V.smear(cam, f - 2, f, 'legL', pal=pal, seed=4))
        img = self.contact_shadow(img, cam, f); img = over(img, L)
        b = self.band_at(f)
        if low:
            if b is not None and f >= 12:
                c = Cel(); pl = Pal(pal)
                if 15 <= f < 20: p0, p1 = cam.project(b[0] + [0, 0, .42])[0], cam.project(b[0])[0]; [c.line(np.array([p0 + [x, 0], p1 + [x, -10]]), w, '#8fa4e6', taper=(.95, .05), alpha=.55, seed=i) for i, (x, w) in enumerate(((-26, 5), (4, 9), (28, 4)))]
                band(c, cam, b[0], b[1], .058, .075, pl); img = over(img, c.rgba())
            img = self.pfront(img, cam, self.P1, tx(t1)); img = self.pfront(img, cam, self.P2, tx(t2)); img = self.K.front(img, cam, tx(t3))
            for k_, sk_, t0 in ((self.G, 'R', 0), (self.V, 'L', 10)):
                if f in (t0, t0 + 1): img = self.star(img, cam, k_.spot('sole' + sk_), f - t0, t0)
        img = self.smoke(img, f, .34)
        if f >= 27: img = self.gust(img, f)
        return img

    def pd(self, P, t): return dust_ring(tw(t * 24 + .5, 2) / 24., P.xy, .14, 1.9, .075, 11, 2, P.seed + 5, .8, 3.8, P.fl, rise=.05)
    def pback(self, img, cam, P, t): return P.dust(img, cam, [q for q in self.pd(P, t) if q[1] > P.xy[1]] + plume(tw(t * 24 + .5, 2) / 24., (P.xy[0], P.xy[1] + .35, P.fl + .02), (0, .5, .6), 2.2, 7, (.04, .15), .5, .1, P.seed + 2, 1.2, 2.6, .2)) if t > 0 else img

    def pfront(self, img, cam, P, t):
        if t < 0: return img
        c = Cel(); P.rb.draw(c, cam, t * 24, P.pal, step=2, lw=1.8, **P.kw); grit(c, cam, t, (*P.xy, P.fl), 60, (1., 3.4), .9, (.004, .014), seed=P.seed, floor=P.fl); img = over(img, c.layer()); return P.dust(img, cam, [q for q in self.pd(P, t) if q[1] <= P.xy[1]], 1.5) if t > 0 else img

    def star(self, img, cam, p, k, seed):
        q = cam.project(p)[0]; m = cam.scale(p); c = Cel(); a = np.linspace(math.pi, 2 * math.pi, 15); r = np.random.default_rng(seed + k); rr = np.where(np.arange(15) % 2, .05, r.uniform(.16, .40, 15)) * m * (1. if k == 0 else .7); P_ = np.c_[q[0] + rr * np.cos(a) * 1.25, q[1] + rr * np.sin(a) * .55]; c.fill(P_, '#fff8ff'); return over(img, c.rgba())

    def contact_shadow(self, img, cam, f):
        c = Cel(); t = np.linspace(0, 2 * math.pi, 25)[:-1]
        for p, z, r in ((self.G.spot('soleR'), self.zg, .15), (self.G.spot('soleL'), self.zg, .15), (self.V.spot('soleL'), self.zv, .14), (self.V.spot('soleR'), self.zv, .14))[:4 if f >= 4 else 2]:
            h = p[2] - z; k = max(.3, 1 - h * 1.6); P3 = np.c_[p[0] - .10 * (1 + h) + r * k * 1.7 * np.cos(t), p[1] - .04 + r * k * 1.5 * np.sin(t), np.full(24, z + .004)]; q, zc = cam.project(P3)
            if (zc > .2).all() and h < .5: c.fill(q, '#171030', .62 * k)
        return over(img, c.rgba())

    def gust(self, img, f):
        """dust lines in the wind that takes the cloth"""
        c = Cel(); r = np.random.default_rng(7); k = float(track(f, [(27, 0, 'out'), (31, 1, 'io'), (40, .7, 'io')]))
        for i in range(26):
            y = r.uniform(60, 1060); x = (r.uniform(0, W + 900) - (f - 27) * r.uniform(90, 150)) % (W + 900) - 450; ln = r.uniform(120, 420) * k; s = r.uniform(-16, 16)
            c.line(np.array([[x, y], [x - ln * .5, y + s], [x - ln, y + s * .4 - 10]]), r.uniform(1., 2.6), '#d8c8ff', taper=(.5, .5), alpha=.28 * k, seed=i)
        return over(img, c.rgba())
