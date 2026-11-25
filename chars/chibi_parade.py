"""b1 staging: the parade on the jetty (setup + frame).  World scrolls V px/frame under the walkers, feet agree with it."""
import numpy as np, cv2, math
from db import W, H, comp
from db.paint import f32, rgb
from .chibi_core import *
from . import chibi_story as ST, chibi_world as WD, chibi_dragon as DR
FINISH = dict(bloom=.16, vig=.08, ca=.4, grain=.009, soft=.08, lift=.0, sat=1.10)
V, S = 8., 108
ORDER = [('piccolo', 150, 4), ('vegeta', 424, -3), ('bulma', 676, 5), ('gohan', 908, 0), ('goku', 1152, 5), ('krillin', 1412, -3), ('roshi', 1640, 4)]
BALLS = [(292, 36), (548, 32), (796, 38), (1034, 34), (1288, 38), (1532, 32), (1800, 36)]


class Parade:
    def setup(self):
        self.sky = WD.sky(); self.sky = comp.over(self.sky, self._sun()); self.cl = WD.clouds_plate().astype(np.float16); self.sea = WD.sea_plate().astype(np.float16); j, self.oy = WD.jetty_plate(); self.jt = j.astype(np.float16); g, self.fy = WD.fg_plate(); self.fg = g.astype(np.float16)

    def _sun(self):
        def fn(c): c.fill(ell(742, 214, 74, 74), '#fff3bf', .55); c.fill(ell(742, 214, 60, 60), '#fff0a8')
        return WD.plate(W, H, fn, seed=3)

    @staticmethod
    def put(im, pl, ox, y):
        ox = float(ox); i = int(math.floor(ox)); fr = ox - i; q = pl[:, i:i + W + 1].astype(f32)
        q = q[:, :W] * (1 - fr) + q[:, 1:W + 1] * fr if q.shape[1] > W else q[:, :W]
        h = min(q.shape[0], H - y); r = im[y:y + h, :q.shape[1]]; a = q[:h, :, 3:4]; r[:] = r * (1 - a) + (q[:h] if im.shape[2] == 4 else q[:h, :, :3])

    def walkers(self, d, f, veg=None):
        gy = WD.GY
        for w, x, dy in ORDER: ST.shadow(d, x, gy + dy, S, .55, '#6b452c', .42)
        for i, (x, yy) in enumerate(BALLS): DR.bounce_ball(d, x, gy + yy - 4, 27, f, (i % 2) * 5, i + 1, hop=1.2, V=V, seed=i)
        for w, x, dy in ORDER:
            if w == 'vegeta' and veg is not None: veg(d, x, gy + dy); continue
            P = ST.march(w, f, S, V, off=.25 if w == 'vegeta' else 0.)
            if w == 'gohan': P['handN'] = (-.90, -.20 + .05 * math.sin(f * .63)); P['hkN'] = 'fist'; P['armTop'] = True
            q = ST.fig(d, w, x, gy + dy, S, P)
            if w == 'gohan': self.kite = q['t']([(-.93, -.36)])[0]

    def frame(self, f, veg=None):
        im = self.sky.copy(); self.put(im, self.cl, 60 + f * .4, 0); self.put(im, self.sea, 250 + f * 1.2, WD.SY)
        d0 = D(lw=4.6, line=ST.LINE); P = DR.path(f, 1490, 262, amp=80)
        DR.shenron(d0, P, 90, f)
        for i, (x, y, s_) in enumerate(((1046, 74, 46), (1190, 112, 36), (84, 118, 40))): WD.gull(d0, x + f * 1.6, y + 6 * math.sin(f * .2 + i), s_, f, i * 2)
        lay = np.zeros((H, W, 4), f32); self.put(lay, self.jt, 196 + f * V, self.oy)
        d = D(lw=4.8, line=ST.LINE); self.walkers(d, f, veg)
        i = int(np.argmin(np.abs(P[:, 0] - (800 + 6 * math.sin(f * .2))))); T, Nn = DR.frames(P); w = DR.width(i / len(P), 90); B = Nn[i] * (1 if Nn[i][1] < 0 else -1)
        a, b = P[i] + B * w * 1.02, P[i] - B * w * 1.06; d0.part(tube([a, (a + b) / 2 + T[i] * 10, b], (8, 9, 8), cap=.4, cap0=.4), '#e0483c', '#b5322f', w=.8)                       # his collar: a red sash with a bow
        for sg in (-1, 1): d0.part(spl([b, b + T[i] * sg * 26 + [0, -10], b + T[i] * sg * 34 + [0, 10], b + T[i] * sg * 12 + [0, 12]], True, n=5, corners=(0,)), '#e0483c', '#b5322f', w=.8)
        k = self.kite; m = (b + k) / 2 + [-20 + 8 * math.sin(f * .31), 52]; d0.stroke(spl([k, m, b + [0, 4]], n=16), .55, '#c8392f', taper=(.02, .02), press=.05, tmin=.8)
        if f * V > 60: WD.crab(d, 2280 - f * V, 985, 30, f)
        fl = d.layer(grain=.03); lay = lay * (1 - fl[..., 3:4]) + fl
        im = comp.over(im, d0.layer(grain=.025)); im = WD.reflect(im, lay, 1012, f=f, amt=.6, squash=.5); im = comp.over(im, lay); self.put(im, self.fg, 60 + f * V * 1.5, self.fy)
        return im
