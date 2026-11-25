"""e1 STANDING IN THE RAIN (80).  Hook: shots/e1_fig.py  fig(f) -> layer; anchors(f) -> dict(hand=(x, y, s, side), rags=[(polyline, length)], drip=[(polyline)], feet=(x, y)).
Very slow push.  Everything else is the place: cliff, sea, sky, rain, one lily opening (frames 20-70)."""
import numpy as np, cv2, math, os
from db import W, H, storm, glove as GL, rag as RG, props as PR
from db.paint import rgb, f32, smooth, fbm, grid, ramp, mix
from db.ink import Cel, spl
from db.comp import over, blur, cam2d
from db.kit import outline, hatch

HZ = 742


class Shot:
    id = 'e1'; n = 80
    finish = dict(bloom=.16, vig=.22, soft=.12, sat=.96)
    EVENTS = [(0, 'rain_bed', dict(to=80, size=.5, note='steady rain, wide and quiet')), (0, 'wind_bed', dict(to=80, size=.35)), (0, 'sea_far', dict(to=80, size=.25))] + \
             [(q, 'drip', dict(size=.2, note='single drop from the glove fingertip')) for q in (9, 26, 41, 57, 73)] + [(16, 'lily_open', dict(to=72, size=.15, note='petals open, almost nothing')), (46, 'cloth_gust', dict(size=.4)), (70, 'drop_petal', dict(size=.15))]

    def setup(self):
        self.mod = storm.hook('e1'); P = 'cache/e1_land1_%s.npy'
        if not all(os.path.exists(P % k) for k in ('back', 'cliff', 'puddle', 'pts')):
            from db import land
            d = land.cliff_e1(); os.makedirs('cache', exist_ok=True)
            for k in ('back', 'cliff', 'puddle'): np.save(P % k, np.asarray(d[k], np.float16))
            np.save(P % 'pts', np.asarray(d['pts'], np.float32))
        self.back, self.cliff, self.puddle = (np.load(P % k) for k in ('back', 'cliff', 'puddle')); self.pts = [tuple(q) for q in np.load(P % 'pts')]

    def frame(self, f):
        z = 1 + .042 * (f / 79.)
        K = lambda lay, k: cv2.warpAffine(lay, np.float32([[1 + (z - 1) * k, 0, -(z - 1) * k * 1075], [0, 1 + (z - 1) * k, -(z - 1) * k * 640]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        img = K(self.back.astype(f32), .25); img = over(img, storm.rain(f, 80, 1, .085, 46., (110, 240), 1.0, '#c9ced0', .34)); img = over(img, K(self.cliff.astype(f32), 1.0)); c = Cel(W, H); storm.splashes(c, f, self.pts, 5); img = over(img, K(c.rgba(), 1.0))
        lay = storm.call(self.mod, 'fig', f); an = storm.call(self.mod, 'anchors', f) or {}
        if an.get('feet'): x, y = an['feet']; a = np.linspace(0, 6.28, 30); c = Cel(W, H); c.fill(np.c_[x + 40 + np.cos(a) * 120, y + 4 + np.sin(a) * 10], '#1c1f22', .55); img = over(img, K(c.rgba(), 1.0))
        if lay is not None: img = over(img, K(lay, 1.0))
        if an.get('rags') or an.get('hand') or an.get('drip'):
            c = Cel(W, H); self.rg = getattr(self, 'rg', None) or [RG.Rag(e, L, seed=11 + i, tear=.7, wet=.9).run(lambda t: (1.5 + .9 * math.sin(t * 1.7) + 1.1 * math.exp(-((t - 2.0) / .35) ** 2), .0), frames=80) for i, (e, L) in enumerate(an.get('rags', []))]
            for q in self.rg: q.draw(c, f, 'rain', 1.5)
            for i, P in enumerate(an.get('drip', [])): RG.water(c, P, f / 24., 'rain', 7 + i, 4.)
            img = over(img, K(c.rgba(), 1.0))
            if an.get('hand'): x, y, s, sd = an['hand']; G = GL.glove('limp', x, y, s, light='rain', wet=1., torn=.5, t=f / 24., side=sd); img = over(img, K(G.layer, 1.0))
        img = over(img, storm.rain(f, 34, 2, .10, 84., (260, 560), 1.8, '#e3e8ea', .50)); c = Cel(W, H); u = float(np.clip((f - 16) / 56., 0, 1)); w = .35 + .25 * math.sin(f * .11) + .35 * math.exp(-((f - 48) / 7.) ** 2); k = PR.lily(c, 420, 1104, 430, u, f / 24., w, 'rain', 2)
        if f > 30: PR.lily_drops(c, k, 430, f / 24.)
        img = over(img, K(c.rgba(), 1.7)); near = storm.rain(f, 7, 3, .12, 150., (560, 980), 4.2, '#f0f3f4', .42); return over(img, blur(near, 1.6)).astype(f32)
