"""Dawn over a sea of cloud (a2, h4).  Real 3-D, about 1 s a frame, so any camera move is allowed.
    D = Dawn('a2')                         # 'a2' orange | 'h4' clear pink dawn (same clouds, same sun)
    c = D.cam(zoom=2., dx=0, dz=0)         # zoom 1 = wide (sun radius 104 at (960,486)); zoom 2 = match circle, radius 208.  dx right, dz up, dy forward (m)
    im, z = D.frame(f, c)                  # finished background rgb (1080,1920,3), z = distance per pixel (1e9 sky)
    im, z, front = D.frame(f, c, split=700)    # front = premultiplied RGBA of everything nearer than 700 m: put it over your character
    (x, y), s = D.at(c, (40, 620, -95))    # world point (x right, y away, z up; cloud floor at z=-290) -> screen px, px per metre
Nimbus and boy belong near (120, 520, -105)."""
import numpy as np, cv2, math
from . import W, H
from .paint import f32, rgb, smooth, ramp, fbm, grid, mix, Canvas, arcs
from .cam import Camera3D
from .sky_vol import Vol, tones, floor_field, over_g, prem, soften

HZ, SX, SY, SR, FL, ZF = 610, 960, 486, 104, 1500., -290.
# crest lines of the sea: (distance, billow radius, palette, open share); towers: (x, y, width, height, seed, pal)
ROWS = [(690, 44, 0, .50), (960, 52, 0, .55), (1400, 60, 0, .52), (1820, 64, 0, .38), (2400, 72, 1, .42), (3150, 78, 1, .36), (4200, 86, 1, .34), (5600, 94, 1, .30), (7600, 100, 2, .28), (10200, 104, 2, .22), (13500, 110, 2, .15)]
BANKS = [(-1000, 1500, 620, 210, 21, 0), (-1900, 2500, 1100, 330, 13, 0), (1480, 1950, 760, 230, 45, 0), (2500, 3900, 1500, 420, 17, 1), (-2700, 4600, 1500, 380, 38, 1),
         (4600, 6400, 2300, 560, 72, 1), (-5200, 7400, 2300, 560, 73, 2), (4600, 10500, 3000, 520, 95, 2), (-4200, 11500, 3800, 560, 96, 2), (9400, 12500, 3600, 700, 97, 2), (-10200, 13000, 4200, 760, 98, 2)]
SKIN = {'a2': dict(sky=[(0, '#ffe2a6'), (.10, '#fdcf86'), (.30, '#fbb765'), (.6, '#f99c48'), (1.02, '#ee8436'), (1.5, '#e07434')], sun='#fffdf2', halo='#fff3d0', fog='#ffd99c', haze='#ffe6b4', mist='#e9b598', cirrus=('#ffdca4', '#f9ad5c'),
                   lit=['#f2b183', '#f8c28a', '#fbd08a', '#ffe3ae', '#fff0c8'], half=['#cf949a', '#dfa495', '#f0b68e', '#fbd3a0', '#ffeabf'], shd=['#7f6f9f', '#a07fa0', '#c99298', '#e8b496', '#f7d7a8'],
                   pal=[('#ffe2a8', '#f7b880', '#c58e98', '#8f79a2', '#6c6398', '#d99a8c'), ('#fff1c9', '#fbd08a', '#d79d8e', '#b9879e', '#957ca2', '#f0b48c'), ('#fff6dc', '#ffe0a6', '#f4bb8e', '#e2a898', '#cc979c', '#ffce98')]),
        'h4': dict(sky=[(0, '#fff1dc'), (.12, '#ffe0c6'), (.35, '#f7c2b4'), (.62, '#d9bccb'), (1.02, '#a9c9e8'), (1.5, '#8fb8e0')], sun='#fffdf4', halo='#fff4dc', fog='#ffdcc4', haze='#ffeeda', mist='#e6bcc2', cirrus=('#fff2e4', '#eac0c6'),
                   lit=['#f7c4b4', '#fbd0b8', '#ffdcc0', '#ffe9d2', '#fff4e4'], half=['#c9a5c2', '#dcb0bf', '#f2c2b8', '#fbd8c6', '#ffead8'], shd=['#7d86bc', '#9a92c0', '#c4a3bd', '#e6c0bd', '#f6dccc'],
                   pal=[('#fff3e2', '#ffd9c2', '#c4a0be', '#8589c0', '#6674b4', '#e0b4bc'), ('#fff6e8', '#ffe0cc', '#e0b4c0', '#b79cc4', '#938fc2', '#f4c8c0'), ('#fffaf0', '#ffecda', '#f8cfc4', '#e2b8c4', '#c6abc6', '#ffdcd0')])}


def _wisp(seed, w=640, h=150):
    """torn streak of vapour, straight alpha mask"""
    r = np.random.default_rng(seed); cv = Canvas(h, w, color=np.ones(3, f32), layer=True); n = 46
    x = r.normal(w * .5, w * .17, n); y = h * .5 + r.normal(0, h * .10, n) + 18 * np.sin(x / w * 4 + seed); P = arcs(x, y, r.normal(-.04, .08, n), r.uniform(90, 330, n) * (1 - .6 * np.abs(x / w - .5)), r.normal(0, .35, n), K=8)
    cv.strokes(P, r.uniform(1.2, 9, n), np.ones(3, f32), op=r.uniform(.10, .42, n), dry=r.uniform(.2, .8, n), seed=seed, shape='round'); e = np.minimum(1, np.minimum(np.arange(w), np.arange(w)[::-1]) / 60.); return cv2.GaussianBlur(cv.a, (0, 0), 1.2) * e[None].astype(f32)


class Dawn:
    def __init__(self, skin='a2', floor_drift=0.):
        self.k = k = SKIN[skin]; V = Vol()
        for i, (y, R, pal, gap) in enumerate(ROWS):
            hw = y * .78 + 300; V.ridge(y, (-hw, hw), R * 1.45, ZF, seed=40 + i, pal=pal, gap=gap, rise=lambda x, y=y, R=R: R * 1.1 * smooth(.16, .62, np.abs(x) / y), levels=2 if y < 5000 else 1, kids=(7, 4), lam=7., depth=.9)
        for (x, y, w, h, sd, pal) in BANKS: V.bank(x, y, ZF - 45, w, h, seed=sd, pal=pal, k=1.9, kids=(11, 5))
        V.build(sun=(0., 1., (HZ - SY) / FL), res=4096); self.V = V; self.P = np.stack([tones(*p) for p in k['pal']]); self.x, self.y = grid()
        self.wisps = [_wisp(s) for s in range(5)]; r = np.random.default_rng(6)
        self.W = [(r.uniform(-1, 1) * y * .9, y, ZF + r.uniform(8, 60) * (.5 + y / 4000), r.uniform(.16, .30) * y ** .9 * 1.3, int(r.integers(0, 5)), r.uniform(-.4, 1.) * 9) for y in np.geomspace(560, 5200, 26)]
        self.bar = fbm(1400, 2400, (130, 52, 21, 9), (1, .62, .4, .25), 31, aspect=5.); self.n1 = fbm(H // 2, W // 2, (150, 52, 18), (1, .5, .25), 12, aspect=5.); self.paper = 1 + .012 * fbm(H, W, (2.6, 1.3), (1, .8), 9)[..., None] + .012 * fbm(H, W, (240, 70, 24), (1, .5, .3), 10)[..., None]

    def cam(self, zoom=1., dx=0., dy=0., dz=0., tilt=0., roll=0.):
        """lens shift keeps the sun where it is; tilt = px the picture slides down"""
        c = Camera3D((dx, dy, dz), (dx + math.sin(roll) * 0, dy + 1000, dz), FL * zoom, cx=SX, cy=SY + (HZ - SY) * zoom + tilt, roll=roll); c.zoom = zoom; c.sun = (SX, SY + tilt); return c

    def at(self, c, P):
        p, z = c.project(np.asarray(P, float)[None]); return (float(p[0, 0]), float(p[0, 1])), float(c.f / max(z[0], 1.))

    def sky(self, f, c):
        k = self.k; x, y = self.x, self.y; hz = c.cy; z = c.zoom; t = (hz - y) / (HZ * z); sx, sy = c.sun; d = np.hypot(x - sx, y - sy) / z
        n = cv2.resize(self.n1, (W, H), interpolation=cv2.INTER_LINEAR)
        s = ramp(t + .010 * n * np.clip(t * 6, 0, 1), k['sky'])
        s = s + (rgb(k['halo']) - s) * (np.exp(-d / 250) * .62)[..., None]
        for R, a, w in ((150, .20, 3.), (232, .15, 5.), (345, .12, 8.), (500, .09, 12.)):      # gouache steps of the glow, hand-cut
            s = s + (rgb(k['halo']) - s) * (a * smooth(R + w, R - w, d + n * R * .035))[..., None]
        tt = np.maximum(t, .10); v = 1 / tt; u = (x - sx) / z / 1500. * v * .62; T = self.bar; f_ = lambda du, dv: cv2.remap(T, ((u * 420 + 1200 + f * .35 + du) % 2399).astype(f32), np.clip(v * 120 + dv - 90, 0, 1398).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        a0 = f_(0, 0); b0 = f_(2, 9.); m = smooth(.42, .62, a0) * smooth(.24, .40, t) * (1 - .6 * smooth(1.0, 1.4, t)); lit = smooth(.0, .16, b0 - a0); cc = rgb(k['cirrus'][1]) + (rgb(k['cirrus'][0]) - rgb(k['cirrus'][1])) * lit[..., None]       # bars of high cirrus, lit along their lower edge
        s = s + (cc - s) * (m * (.30 + .34 * lit))[..., None]
        s = s + (rgb(k['sun']) * 1.12 - s) * smooth(SR + .6, SR - .6, d)[..., None]
        return s.astype(f32)

    def floor(self, f, c):
        k = self.k; F = floor_field(self.V, c, ZF, l=460., ky=1.6, sun=(0., 1.), seed=3, off=(f * .9, 0.), soft=.012); d = F[..., 0]; q = np.clip(np.log(np.maximum(d, 1.) / 700.) / math.log(16000 / 700.), 0, 1); st = (0, .25, .55, .8, 1.)
        L, M, S = [ramp(q, list(zip(st, k[n]))) for n in ('lit', 'half', 'shd')]; v, g, s = F[..., 1], F[..., 2], F[..., 3]
        road = np.exp(-((F[..., 4] / (90 + d * .085)) ** 2)) * smooth(600, 2600, d)          # the road of light under the sun
        col = M + (L - M) * np.clip(smooth(.04, .14, g) * .38 + .5 * road + .2 * smooth(.02, .30, g), 0, 1)[..., None]
        col = col + (M * .55 + S * .45 - col) * (smooth(.03, .28, -g) * .35 * (1 - road))[..., None]
        sh = 1 - smooth(.32, .62, s + v * .10); S2 = S + (M - S) * np.clip(smooth(.03, .20, g) * .34 + smooth(.0, .5, v) * .16, 0, 1)[..., None] - (smooth(.06, .34, -g) * .05)[..., None]
        col = col + (S2 - col) * (sh * .88)[..., None]
        col = col + (rgb(k['haze']) - col) * (1 - np.exp(-(d / 15000) ** 1.5))[..., None]
        return col, d

    def frame(self, f, c, split=None, wisps=True):
        k = self.k; sky = self.sky(f, c); fl, d = self.floor(f, c); m = (d > 0)[..., None]; im = np.where(m, fl, sky); hb = np.exp(-np.abs(self.y - c.cy) / (26 * c.zoom)); im = im + (rgb(k['haze']) - im) * (hb * .7 * (np.hypot(self.x - c.sun[0], self.y - c.sun[1]) > SR * c.zoom + 1))[..., None]
        kw = dict(fog=dict(d=14500, p=1.4, col=k['fog']), light=(0, .62, .78), sil=1., silw=.26, sunscreen=c.sun, tl=.60, th=.22, skyl=.25, bounce=.4, mist=dict(z=ZF - 30, h=80, col=k['mist'], amt=.55), kd=.25, acc=.35, accs=.06, noise=.12, mixn=(.16, .28, .70), melt=6, wob=.13, el=.035, deep=.45)
        g = self.V.render(c, self.P, sky=None, **kw); z = np.where(d > 0, d, 1e9).astype(f32); z = np.where(g['a'] > .5, np.minimum(g['z'], z), z); front = None
        if split is None: im = over_g(im, g)
        else:
            b = dict(g); b['a'] = g['a'] * (g['z'] > split); im = over_g(im, b); ft = dict(g); ft['a'] = g['a'] * (g['z'] <= split); front = prem(ft)
        if wisps: im = self.vapour(im, f, c, z)
        im = im * self.paper
        return (im, z) if split is None else (im, z, front)

    def vapour(self, im, f, c, z):
        for (x, y, zz, w, j, vx) in self.W:
            (px, py), s = self.at(c, (x + vx * f * .12, y, zz)); ww = w * s
            if ww < 40 or px + ww < 0 or px - ww > W or py > H + 80 or y - c.eye[1] < 60: continue
            t = self.wisps[j]; sc = 2 * ww / t.shape[1]; M = np.float32([[sc, 0, px - ww], [0, sc * .8, py - t.shape[0] * sc * .4]]); x0, x1, y0, y1 = int(max(0, px - ww)), int(min(W, px + ww)), int(max(0, py - t.shape[0] * sc * .5)), int(min(H, py + t.shape[0] * sc * .5))
            if x1 - x0 < 4 or y1 - y0 < 4: continue
            M[0, 2] -= x0; M[1, 2] -= y0; a = cv2.warpAffine(t, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR) * smooth(0, 60, z[y0:y1, x0:x1] - (y - c.eye[1])) * .62
            q = float(np.clip(math.log(y / 700.) / math.log(16000 / 700.), 0, 1)); col = ramp(np.float32([q]), list(zip((0, .25, .55, .8, 1.), self.k['lit'])))[0]; im[y0:y1, x0:x1] += (col - im[y0:y1, x0:x1]) * a[..., None]
        return im
