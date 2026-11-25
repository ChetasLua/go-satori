"""Cloud worlds to fly through.  Real 3-D billows lit by a real sun; 0.4-0.9 s a frame.
    D = Dive(); im, z = D.frame(f)                 # a3's POV fall, local f 0..15 (40..55 of the shot) through three decks, then D.burst(f) 0..23 over the sea
    c = D.cam(f); im, z, front = D.frame(f, split=40.)     # front = premultiplied RGBA of cloud nearer than split metres (goes over your character)
    W = cumulus_world('day'); im, z = W.frame(cam, f)      # sunlit cumulus over the sea, any Camera3D between 200 and 2600 m up (a4, d3)
    im = burst_white(im, f, k)                             # wall of light: torn vapour + white, k 0..1"""
import numpy as np, cv2, math
from . import W, H
from .paint import f32, rgb, smooth, ramp, fbm, grid, mix
from .cam import Camera3D
from .comp import ease
from . import kit
from .sky_vol import Vol, tones, over_g, prem, vn3
from .sky_sea import sea, SEA, mixpal

ISL = [(420, 1500, 190, 3), (-760, 2300, 300, 8), (1500, 3900, 520, 14), (-2200, 5200, 700, 21), (90, 820, 60, 30), (-300, 7800, 1000, 33), (2700, 7600, 800, 41), (-3800, 9800, 1200, 55), (3900, 2600, 360, 60)]
PAL = {'gold': ('#fff3cf', '#ffd896', '#e2a094', '#a87c9e', '#7a6aa2', '#dc9a90'), 'cream': ('#fffaf0', '#ffeccb', '#e6c4c0', '#a9a0c6', '#8189bd', '#dcc6cc'),
       'white': ('#ffffff', '#fff6e4', '#cad8ec', '#86abd8', '#6190c8', '#b7d8ea'), 'storm': ('#aab4c6', '#78869c', '#4f5b72', '#333d52', '#1f2636', '#2a3142'), 'night': ('#b8a8e8', '#7c70b4', '#3f3d70', '#1e1e44', '#0d0d26', '#241f4c')}


def P(*names): return np.stack([tones(*PAL[n]) for n in names])


_V = {}
def vapour(seed=0, n=1):
    """torn radial vapour texture (H, W) 0..1 for flying through a deck"""
    k = ('v', seed)
    if k not in _V:
        x, y = grid(H // 2, W // 2); cx, cy = W / 4, H / 4; th = np.arctan2(y - cy, x - cx); r = np.hypot(x - cx, y - cy)
        a = fbm(256, 2048, (60, 22, 9), (1, .6, .4), 50 + seed, aspect=.22); u = ((th / (2 * math.pi) + .5) * 2047).astype(f32); v = np.clip(np.log(r / 30. + 1) * 70, 0, 255).astype(f32); t = cv2.remap(a, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP); _V[k] = cv2.resize(t, (W, H), interpolation=cv2.INTER_LINEAR)
    return _V[k]


def burst_white(im, f, k, col='#fffaf0', col2='#ffe6bd', c=(960, 486)):
    """wall of light while the camera is inside a deck: k 0..1; hard torn tongues streaming past the lens, never a flat fade"""
    if k <= 0: return im
    x, y = grid(); t = vapour(f % 3); r = np.hypot(x - c[0], (y - c[1]) * 1.2) / 1100.; q = k * 1.9 - .55; m = smooth(-.10, .10, t * .22 + q - .75 * (1 - np.clip(r, 0, 1.2)) * (1 - k)); m2 = smooth(-.06, .06, t * .20 + q - .45 - .6 * (1 - np.clip(r, 0, 1.2)) * (1 - k))
    im = im + (rgb(col2) - im) * (m * .92)[..., None]; return im + (rgb(col) * 1.04 - im) * (m2 * .96)[..., None]


class Dive:
    """altitudes (m): decks at 1020, 640, 300, sea 0.  Sun low ahead-right at the top, clear day below."""
    DECK = (1020., 640., 300.)

    def __init__(self):
        z1, z2, z3 = self.DECK; r = np.random.default_rng(4); self.V = []
        spec = [(z1, 0, dict(size=74, seed=3, rel=.50, thr=.86), (-1000, 1000, -420, 3000), [(-330, 420, 300, 240), (360, 640, 380, 300), (-120, 1100, 420, 300), (700, 1500, 600, 420), (-800, 1700, 600, 420)], (.80, .34, .26)),
                (z2, 1, dict(size=56, seed=16, rel=.52, thr=.64), (-760, 760, -420, 1900), [(-230, 260, 220, 230), (270, 380, 260, 230), (-420, 760, 420, 300), (120, 900, 300, 240), (560, 1100, 460, 320)], (.74, .40, .36)),
                (z3, 2, dict(size=42, seed=23, rel=.55, thr=.55), (-640, 640, -420, 1700), [(190, 200, 170, 150), (-210, 330, 210, 170), (330, 620, 300, 200), (-420, 900, 380, 240), (80, 720, 200, 150)], (.80, .30, .30))]
        for (z, pal, dk, (x0, x1, y0, y1), tw, sun) in spec:
            V = Vol(); V.deck((x0, x1), (y0, y1), z, dk['size'], seed=dk['seed'], pal=pal, rel=dk['rel'], levels=2, kids=(8, 5), sink=.42, thr=dk['thr'], detail_d=800, eye=(0, 0), foot=40.)
            for i, (x, y, w, h) in enumerate(tw): V.cumulus(x, y, z - 30, w, h, seed=60 + i + 9 * pal, pal=pal, levels=2, peaks=4, foot=.12)
            for i in range(9): V.puff(r.uniform(-200, 200), r.uniform(-60, 420), z + r.uniform(-110, 60), r.uniform(12, 30), seed=i + 20 * pal, pal=pal, n=7, kids=(8, 5))
            V.build(sun=sun, res=3072, cast=0); self.V.append(V)
        self.x, self.y = grid(); self.P = P('gold', 'cream', 'white')

    def alt(self, f):
        """fall: a touch of slow at the lip, then full speed"""
        return float(np.interp(f, [0, 2, 5, 8, 11, 14, 16, 20], [1210, 1110, 900, 700, 500, 330, 215, 120]))

    def cam(self, f, roll=None):
        z = self.alt(f); y = 6. * f + .4 * f * f * .25; pit = math.radians(np.interp(f, [0, 2, 5, 12, 16, 20], [38, 56, 64, 62, 46, 20])); ro = math.radians(np.interp(f, [0, 4, 9, 14, 18], [-3, 4, -5, 3, 0])) if roll is None else roll
        d = np.array([.05 * math.sin(f * .5), math.cos(pit), -math.sin(pit)]); e = np.array([8 * math.sin(f * .35), y - 120, z]); return Camera3D(e, e + d, 1080., cx=960, cy=486, roll=ro)

    def whiteness(self, f):
        z = self.alt(f); return .74 * max(math.exp(-((z - d - o) / s) ** 2) for d, o, s in zip(self.DECK, (22, 10, 4), (40., 38., 32.)))

    def skyc(self, f, c):
        z = self.alt(f); k = float(np.clip((z - 250) / 800., 0, 1)); d = c.rays(); e = d[..., 2] / np.linalg.norm(d, axis=-1)
        top = mix(rgb('#57a6e0'), rgb('#f9a850'), k); hz = mix(rgb('#dff2f8'), rgb('#fff0c8'), k); return (hz + (top - hz) * np.clip(e * 2.2, 0, 1)[..., None]).astype(f32), hz

    def frame(self, f, c=None, split=None, blur=True, white=True):
        c = self.cam(f) if c is None else c; z0 = float(c.eye[2]); k = float(np.clip((z0 - 250) / 800., 0, 1)); sk, hz = self.skyc(f, c); pal = mixpal('day', 'gold', k * .85); pal[13] = mix(rgb('#d8eef4'), rgb('#fbe4ba'), k)
        im, ds = sea(c, f, pal, sun=(.30, .62, .55), sky=sk, islands=ISL, shadows=(420., .40, 0, 0), haze=3000 + 9000 * (1 - k), swell=160.)
        fog = dict(d=2300., p=1.25, col=mix(rgb('#cfe6f2'), rgb('#ffe2b0'), k)); kw = dict(fog=fog, sil=0., tl=.32, th=.04, bounce=.5, skyl=.12, acc=.4, accs=.16, kd=.5, deep=.5, noise=.12, mixn=(.24, .3, .6), near=(8., 34.), minpx=.8)
        z = np.where(ds > 0, ds, 1e9).astype(f32); ft = None
        for V, zd in zip(self.V[::-1], self.DECK[::-1]):
            if z0 < zd - 130 and c.R[2][2] < -.25: continue          # already above and behind us
            g = V.render(c, self.P, sky=None, **kw); z = np.where(g['a'] > .5, np.minimum(z, g['z']), z)
            if split is None: im = over_g(im, g)
            else:
                b = dict(g); b['a'] = g['a'] * (g['z'] > split); im = over_g(im, b); q = dict(g); q['a'] = g['a'] * (g['z'] <= split); q = prem(q); ft = q if ft is None else q + ft * (1 - q[..., 3:4])
        w = max(self.whiteness(f), float(np.interp(f, [3, 5, 14, 16], [0, .20, .20, .12]))) if white else 0.
        if blur: v = abs(self.alt(f + .5) - self.alt(f - .5)); im = kit.zoomblur(im, 960, 486 + 60, min(.030, v * .00038), 7)
        if w > .05: im = burst_white(im, f, w, *(('#fffdf2', '#ffe3b0') if k > .6 else ('#ffffff', '#f3e6e0') if k > .25 else ('#ffffff', '#dceaf4')))
        return (im, z) if split is None else (im, z, ft)

    def burst(self, f):
        """after 56: pull up over the bright sea: f 0..23 -> eye from 215 m to 6 m, horizon swings into frame"""
        t = min(1., f / 22.); e = ease(t); z = 215 * (1 - e) ** 1.6 + 5.; pit = math.radians(46 * (1 - ease(min(1, t * 1.25))) + 4); y = 60 + 24 * f; c = Camera3D((0, y, z), (0, y + math.cos(pit), z - math.sin(pit)), 1080., cx=960, cy=486, roll=math.radians(-6 * math.sin(math.pi * t)))
        return self.low(f, c), c

    def low(self, f, c):
        sk = day_sky(c); im, d = sea(c, f, 'day', sun=(.12, .9, .42), sky=sk, islands=[(a, b + 900, r, s) for a, b, r, s in ISL], haze=9000, swell=float(np.clip(c.eye[2] * 1.1, 12, 160)))
        return im


def day_sky(c, pal=('#e8f6f6', '#a6dcf0', '#4fa6e2', '#1f6fc0'), clouds=None):
    """clear noon sky for any camera"""
    d = c.rays(); e = d[..., 2] / np.linalg.norm(d, axis=-1); return ramp(np.clip(e, 0, 1), [(0, pal[0]), (.10, pal[1]), (.38, pal[2]), (1, pal[3])])


class CumulusWorld:
    def __init__(self, kind='day', seed=0, n=26, span=3600, hole=None):
        V = Vol(); r = np.random.default_rng(90 + seed)
        for i in range(n):
            y = r.uniform(350, span); x = r.uniform(-1, 1) * (300 + y * .75); w = r.uniform(260, 620) * (.8 + y / span * .7)
            if abs(x) < 190 + .10 * y: x += math.copysign(260 + .12 * y, x)
            V.cumulus(x, y, 420 + r.uniform(-30, 40), w, w * r.uniform(.45, .95), seed=200 + seed + i, peaks=4, levels=2, foot=.14)
        V.build(sun=(-.55, .25, .70), res=4096); self.V = V; self.kind = kind; self.P = P({'day': 'white', 'gold': 'gold', 'dawn': 'cream', 'night': 'night', 'storm': 'storm'}[kind])

    def frame(self, c, f=0, split=None, near=(8., 40.)):
        sk = day_sky(c); im, ds = sea(c, f, 'day', sun=(-.3, .8, .6), sky=sk, islands=ISL, shadows=(520., .38, .02 * f, 0), haze=14000); g = self.V.render(c, self.P, fog=dict(d=7000., p=1.3, col='#cfe8f4'), tl=.18, th=-.10, bounce=.5, acc=.4, accs=.12, noise=.12, mixn=(.2, .3, .65), near=near)
        z = np.where(ds > 0, ds, 1e9).astype(f32); z = np.where(g['a'] > .5, np.minimum(z, g['z']), z)
        if split is None: return over_g(im, g), z
        b = dict(g); b['a'] = g['a'] * (g['z'] > split); im = over_g(im, b); q = dict(g); q['a'] = g['a'] * (g['z'] <= split); return im, z, prem(q)


def cumulus_world(kind='day', **k): return CumulusWorld(kind, **k)
