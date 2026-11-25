"""g5 THE MOON FALLS.  0-9 close: a tiny figure, arms up, under a curve that fills the sky.  10-29 one slow pull-out: the scale arrives.
30-39 insert: a torn white glove on wet rock; the violet of the falling moon grows on it; the fingers close (34-38).
No sound: weight = slowness, the sea going flat, the cloud collar pressed down and out, the town's lights going out, birds leaving.
fig_g5(f, x, y, h) is a plain placeholder: (x, y) = its crown on screen, h = its height in px."""
import numpy as np, cv2, math
from db import W, H as HH, comp, kit, moon as M
from db.moon_fall import Fall
from db.paint import rgb, f32, smooth, grid, ramp, fbm, mix
keys = comp.lerp_keys
FX, FY = 1010., 705.                                            # where the figure hangs in world px (just under the lower limb)


def fig_g5(f, x, y, h):
    """PLACEHOLDER: one plain bar with two raised bars, float mask"""
    m = np.zeros((HH, W), f32); s = h / 8.; p = lambda L: np.round((np.array(L, float) * s + (x, y)) * 16).astype(np.int32)
    cv2.circle(m, tuple(p([(0, .5)])[0].tolist()), max(8, int(.5 * s * 16)), 1., -1, cv2.LINE_AA, 4); cv2.fillPoly(m, [p([(-.75, 1.1), (.75, 1.1), (.28, 8), (-.28, 8)])], 1., cv2.LINE_AA, 4)
    for d in (-1, 1): cv2.fillPoly(m, [p([(d * .75, 1.1), (d * .45, 1.5), (d * 1.35, -1.6), (d * 1.6, -1.45)])], 1., cv2.LINE_AA, 4)
    return m


class Shot:
    id = 'g5'; n = 40
    finish = dict(bloom=.22, vig=.18, grain=.010, soft=.12)
    EVENTS = [(0, 'silence', dict(note='no sound effects in space or here: music only')), (0, 'heart', dict(size=.3, note='inside the body, one slow beat')), (10, 'pull_out', dict(frames=20, note='music widens; a sub note that only rises in size')), (20, 'heart', dict(size=.4)), (30, 'insert', dict(note='cut to the glove: wet rock, one drip')), (34, 'glove_close', dict(frames=5, size=.35, note='wet cloth creaks'))]
    fig = staticmethod(fig_g5)

    def setup(self): self.F = Fall(zmax=2.6); self.rock = self._rock()

    # ------------------------------------------------------------ the insert
    def _rock(self):
        """wet slab seen close from low: cracked plates in hard planes, a film of water at the front -> (tone 0..1, wet mask, crack mask, height)"""
        x, y = grid(); r = np.random.default_rng(12); P = np.stack([r.uniform(-200, W + 200, 46), 330 + 900 * r.random(46) ** .8], 1); k = .30 + .85 * np.clip((y - 330) / 750, 0, 1.2); n = fbm(HH, W, (90, 34, 12), (1, .5, .25), 4, aspect=2.2); u, v = x + 22 * n, (y + 11 * n) / .42
        a = np.full((HH, W), 1e9, f32); b = a.copy(); ia = np.zeros((HH, W), np.int32)
        for i, (px, py) in enumerate(P): d = (u - px) ** 2 + (v - py / .42) ** 2; c = d < a; b = np.where(c, a, np.minimum(b, d)); ia = np.where(c, i, ia); a = np.where(c, d, a)
        e = np.sqrt(b) - np.sqrt(a); cr = smooth(26, 11, e); lip = smooth(64, 46, e) * (1 - cr); t = r.uniform(.16, .62, len(P))[ia] + .13 * (fbm(HH, W, (28, 10, 4), (1, .6, .35), 9, aspect=2.6) > .45) - .10 * (fbm(HH, W, (46, 15), (1, .5), 19, aspect=3.) > .8); wet = smooth(-.1, .12, n * .5 + .9 * np.clip((y - 760) / 260, -1, 1) - .18 + .25 * fbm(HH, W, (210, 80), (1, .5), 23, aspect=2.8))
        return np.clip(t + .22 * lip, 0, 1).astype(f32), wet.astype(f32), cr.astype(f32), n.astype(f32)

    def _wrist0(self):
        from db import glove as GV, hands as HD
        p = HD.pose((24, 22, 12, -2), (30, 30, 16, 0), (36, 36, 18, 1), (42, 40, 20, 5), thumb=(30, 34, 16, 12, 16), wrist=(-6, 4)); return np.asarray(GV.glove('relaxed', 1000., 690., 300., light='violet', pose=p, yaw=20., pitch=62., roll=-72., dist=5.5, pivot=(0, .75, 0)).pts['wrist'], float)[:2]

    def insert(self, f):
        from db import glove as GV, hands as HD, storm
        from db.ink import Cel
        g = f - 30; x, y = grid(); t, wet, cr, n = self.rock; k = float(smooth(-2, 9, g))                                      # k: the light of the falling moon growing
        c = ramp(np.clip(t, 0, 1), [(0, '#0d0b18'), (.30, '#1a1730'), (.302, '#262244'), (.52, '#37325e'), (.522, '#4a4478'), (1, '#6a62a0')]); c = c * (.55 + .30 * k) + (np.clip(t - .30, 0, 1) * k)[..., None] * rgb('#7a3cf0') * .75
        c = c * (1 - .80 * cr[..., None]) + rgb('#06050c') * (.80 * cr)[..., None]; sky = smooth(365, 330, y + 10 * n); wet = wet * (1 - sky); cyb = -1010 + 78 * g / 9. + 30 * k; d = np.hypot(x - 1130, y - cyb) - 1120; lb = smooth(16, -16, d); hl = M.halo_rgb(np.where(d > -20, M.halo(d, 260), 0)) * (.35 + .65 * k); sk = ramp(y / 365, [(0, '#120a28'), (1, '#2a1560')]) * (.6 + .5 * k) + hl; sk = sk * (1 - lb[..., None]) + mix(rgb('#8a6ad8'), rgb('#efe2ff'), smooth(10, -120, d)[..., None]) * (.55 + .45 * k) * lb[..., None]; img = c * (1 - sky[..., None]) + sk * sky[..., None]
        ref = smooth(.30, .62, np.sin(y * .085 + n * 2.4 + g * .22) * .5 + .5) * np.exp(-((x - 1010 - 40 * n) / (190 + .30 * np.clip(y - 380, 0, 700))) ** 2); img = img * (1 - .50 * wet[..., None]) + rgb('#07060f') * (.50 * wet)[..., None] + (wet * ref * (.22 + .78 * k))[..., None] * rgb('#d9b8ff') * .85      # the moon's road comes into the puddle
        u = float(keys(g, [(0, 0.), (4, 0.), (5, .22), (6, .55), (7, .86), (8, 1.04), (9, 1.)], lambda q: q)); limp = HD.pose((24, 22, 12, -2), (30, 30, 16, 0), (36, 36, 18, 1), (42, 40, 20, 5), thumb=(30, 34, 16, 12, 16), wrist=(-6, 4)); p = limp if u <= 0 else HD.blend(limp, 'relaxed', min(1., u / .35), drag=.2) if u < .35 else HD.blend('relaxed', 'fist', min(1., (u - .35) / .65), drag=.28, order=(3, 2, 1, 0))
        key = (round(u, 2), round(k, 1)); G = self._gl.get(key) if hasattr(self, '_gl') else None
        if G is None: self._gl = {}; G = GV.glove('relaxed', 1000., 690., 300., light='violet', wet=.75, torn=.6, seed=5, t=g / 24., pose=p, yaw=20., pitch=62., roll=-72., dist=5.5, pivot=(0, .75, 0), sleeve='#a58fc4'); self._gl[key] = G
        # the wrist is dead weight (it does not move); the limb of the falling ball, out of focus, is the lamp
        L = G.layer; wp = np.asarray(G.pts['wrist'], float)[:2]; self._wr = getattr(self, '_wr', None) if getattr(self, '_wr', None) is not None else self._wrist0(); L = cv2.warpAffine(L, np.float32([[1, 0, self._wr[0] - wp[0]], [0, 1, self._wr[1] - wp[1]]]), (W, HH)); a = L[..., 3]; sh = np.maximum(cv2.GaussianBlur(kit.shifted(a, -16, 30), (0, 0), 7), .9 * cv2.GaussianBlur(kit.shifted(a, -3, 11), (0, 0), 3)) * (1 - sky); img = img * (1 - .78 * sh[..., None]) + rgb('#05040b') * (.70 * sh)[..., None]      # contact shadow: it LIES on the rock
        fl = cv2.warpAffine(L, np.float32([[1, 0, 0], [0, -.42, 690 * 1.42 + 236]]), (W, HH)); fl = cv2.GaussianBlur(fl, (0, 0), sigmaX=5, sigmaY=2); img = img * (1 - fl[..., 3:4] * (wet * .42)[..., None]) + fl[..., :3] * (wet * .34)[..., None]
        Lg = L.copy(); Lg[..., :3] *= .52 + .30 * k; rimv = (a * (kit.shifted(a, -7, 9) < .5) * k)[..., None]; Lg[..., :3] = Lg[..., :3] * (1 - rimv * .85) + rgb('#e9d2ff') * rimv * .85; img = comp.over(img.astype(f32), Lg)
        cl = Cel(); storm.rain(f, n=26, seed=7, slope=.06, speed=60., L=(60, 190), w=1.1, col='#c9b8f0', a=.22, cel=cl); img = comp.over(img.astype(f32), cl.layer())
        img = img + (np.exp(-((x - 1040) / 760) ** 2 - ((y - 150) / 420) ** 2) * k)[..., None] * rgb('#8a2cff') * .30; return comp.cam2d(img.astype(f32), 960, 545, 1.02 + .0022 * g)


    def cam(self, f):
        """close 0-9 (zoom 2.5 easing a hair), then one continuous pull-out with a long soft landing"""
        z = float(keys(f, [(0, 2.50), (9, 2.44)])) if f < 10 else float(np.exp(np.interp(1 - (1 - min(1., (f - 9) / 21.)) ** 2.4, [0, 1], [math.log(2.44), 0.])))
        k = (z - 1) / 1.5; return z, 960 + (FX - 960) * min(1, k * 1.3), 540 + (FY - 85 - 540) * min(1, k * 1.25)

    def frame(self, f):
        if f >= 30: return self.insert(f)
        v = self.cam(f); u = .46 + .155 * (f / 29.) ** 1.15; press = float(smooth(4, 29, f)) * .95; wt = float(np.interp(f, [0, 10, 29], [0, .42, .74]))      # the water's own clock slows as it is pressed flat
        img = self.F.frame(u, f * 1.0, v, press, lights=1 - .85 * smooth(16, 28, f), water_t=wt)
        x, y = self.F.at(FX, self.F.cy(u) + self.F.r + 16, v); h = 34 * v[0]; m = self.fig(f, x, y, h); z = v[0]
        if m.ndim == 3: return comp.over(img, m).astype(f32)
        dt = cv2.distanceTransform((m > .5).astype(np.uint8), cv2.DIST_L2, 3); Y = np.arange(HH)[:, None]; top = smooth(y + h * .7, y, Y); b = np.broadcast_to(rgb('#0a0616'), img.shape).copy(); b = np.where(((dt < 1.0 * z) & (top > .15))[..., None], rgb('#f0dcff') * top[..., None] + rgb('#7a2cf0') * (1 - top)[..., None], b)      # lit from above only: the ball is the lamp
        g = cv2.GaussianBlur(m, (0, 0), 5 * z); img = img * (1 - .34 * g[..., None])                                  # a little darkness round it so it is found at once
        return (img * (1 - m[..., None]) + b * m[..., None]).astype(f32)


from chars import figs_lead as _FL                                  # the real figure
Shot.fig = staticmethod(_FL.g5_fig)
