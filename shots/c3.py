"""c3 ONE FINGER.  0-19: a hand rises, a dot of light the size of a pearl is born on the fingertip; light and colour bend toward it.
20-39 reverse: four people frozen on the edge of the mesa, the colour drained; a drop of sweat falls and stops in mid-air at 36.
hand_c3(f), tip_c3(f) and figs_c3(f) are plain placeholders."""
import numpy as np, cv2, math
from db import W, H as HH, comp, kit, moon as M, moon_fx as X
from db.paint import rgb, f32, fbm, smooth, mix, ramp, grid
keys = comp.lerp_keys
KEEP = .35                                                      # how much violet the drain must leave


def drain_(img, *a, **k): return img * KEEP + X.drain(img, *a, **k) * (1 - KEEP)


PAD = 340
DOT = 10                                                        # the frame the pearl is born


def tip_c3(f):
    g = (int(f) // 2) * 2
    y = float(keys(g, [(0, 1130.), (2, 1060.), (8, 520.), (10, 474.), (12, 486.), (20, 486.)], lambda t: 1 - (1 - t) ** 2)); return 960., y


def hand_c3(f):
    """PLACEHOLDER: one upright bar with a block under it, float mask"""
    x, y = tip_c3(f); m = np.zeros((HH, W), f32); p = lambda L: np.round((np.array(L) + (x, y)) * 16).astype(np.int32)
    cv2.fillPoly(m, [p([(-24, 30), (-14, 12), (14, 12), (24, 30), (36, 330), (-36, 330)])], 1., cv2.LINE_AA, 4); cv2.fillPoly(m, [p([(-120, 320), (110, 300), (150, 520), (96, 1000), (-100, 1000), (-146, 520)])], 1., cv2.LINE_AA, 4); return m


def figs_c3(f):
    """PLACEHOLDER: four plain standing pins on the ledge, float mask (feet on y = LEDGE)"""
    m = np.zeros((HH, W), f32)
    for x, s in ((560, 46), (820, 50), (1130, 58), (1330, 27)):
        p = lambda L: np.round((np.array(L, float) * s + (x, LEDGE + 6)) * 16).astype(np.int32)
        cv2.circle(m, tuple(p([(0, -6.5)])[0].tolist()), int(.5 * s * 16), 1., -1, cv2.LINE_AA, 4); cv2.fillPoly(m, [p([(-.8, -5.8), (.8, -5.8), (.6, 0), (-.6, 0)])], 1., cv2.LINE_AA, 4)
    return m


LEDGE = 812


def child_c3(f):
    """PLACEHOLDER for the close-up of the child (looking up-right at the light): a plain egg bust -> mask or premultiplied RGBA"""
    m = np.zeros((HH, W), f32); k = (f - 25) / 15.; cv2.ellipse(m, (int(760 * 16), int((470 - 6 * k) * 16)), (int(255 * 16), int(300 * 16)), 0, 0, 360, 1., -1, cv2.LINE_AA, 4); cv2.fillPoly(m, [np.round(np.array([(640, 690), (880, 690), (900, 800), (1180, 900), (1260, 1090), (260, 1090), (340, 900), (620, 800)]) * 16.).astype(np.int32)], 1., cv2.LINE_AA, 4); return m


def chin_c3(f): return 936., 700. - 6 * (f - 25) / 15.             # where the drop hangs from (screen px): the owner of child_c3 moves this


CHILD = (1330., 700.)                                             # where the smallest of the four stands in the wide plate


class Shot:
    id = 'c3'; n = 40
    finish = dict(bloom=.30, vig=.22, grain=.010, soft=.12)
    EVENTS = [(0, 'hand_rise', dict(frames=10, note='cloth-less, silent')), (DOT, 'pearl', dict(size=.4, note='pure sine, almost inaudible, then every other sound bends down toward it')), (20, 'cut', dict(note='sound drained: only blood in the ears')),
              (22, 'push_in', dict(frames=6, note='breath sucked in')), (28, 'drop_falls', dict(size=.2, note='leaves the chin')), (36, 'drop_stops', dict(size=.6, note='all sound stops with it'))]
    hand = staticmethod(hand_c3); figs = staticmethod(figs_c3); tip = staticmethod(tip_c3); child = staticmethod(child_c3); chin = staticmethod(chin_c3)

    def setup(self):
        S = M.Sky(960, 486, 208, 3); self.S = S; a = S.frame(40, 1.); M.sea(a, 792, t=1.9); M.sparks(a, 56, 3, avoid=(960, 486, 262), amt=.9)
        self.plate = a; self.backP = cv2.filter2D(comp.cam2d(a, 960, 486 + 54 / 2.25, 2.25, w=W + 2 * PAD, h=HH + 2 * PAD), -1, comp.disc_kernel(9.)); self.back = self.backP[PAD:PAD + HH, PAD:PAD + W]      # the c1 plate; from frame 8 a long lens: the disc fills the frame behind the hand
        self.rev = self._reverse(); b = comp.cam2d(self.rev, CHILD[0] - 120, CHILD[1] - 60, 3.0); self.close = cv2.filter2D(b, -1, comp.disc_kernel(17.)); del b

    _x, _y = grid()

    def zoom(self, a, z): return comp.cam2d(a, 960, 486 + 54 / z, z)        # the match circle keeps its centre (960, 486)

    def _reverse(self):
        """heroes' side, lit from the right by the pearl: black rock, pale dead haze over the sea, designed angular cliffs"""
        from db.ink import Cel
        x, y = grid(); img = ramp(y / HH, [(0, '#0a0618'), (.28, '#1d0f45'), (.52, '#4b2c93'), (.66, '#8f74cf'), (.735, '#cdbdf0'), (.75, '#b7a4e4'), (.752, '#5a3fa0'), (1, '#1a0d3a')])
        n = fbm(HH, W, (46, 16, 6), (1, .6, .3), 12, aspect=9.); st = smooth(.5, .62, n + 1.1 * np.exp(-((y - 560) / 150) ** 2) - .5); img = img * (1 - .30 * st[..., None]) + rgb('#c9b6f2') * (.30 * st * smooth(300, 700, y))[..., None]; st2 = smooth(.35, .5, fbm(HH, W, (60, 20, 7), (1, .6, .3), 21, aspect=8.) - .25); img = img * (1 - .34 * (st2 * smooth(620, 200, y))[..., None])
        r = np.random.default_rng(4); c = Cel()
        def mesa(x0, x1, top, base, lit, mid, sh, seed):
            q = np.random.default_rng(seed); m = max(4, int((x1 - x0) / 62)); xs = np.linspace(x0, x1, m + 1) + np.r_[0, q.normal(0, 13, m - 1), 0]; tp = top + np.r_[26, q.uniform(-9, 11, m - 1), 20]; H_ = base - top
            spread = H_ * .34; talus = base - H_ * .34; out = [(x0 - spread, base), (x0 - spread * .34, talus), (xs[0], tp[0])] + [p for k in range(m) for p in ((xs[k] + 3, tp[k]), (xs[k + 1] - 5, tp[k] + q.uniform(-2, 4)))] + [(xs[-1], tp[-1]), (x1 + spread * .4, talus), (x1 + spread, base)]
            c.fill(np.array(out), mid); c.clip(np.array(out))
            for k in range(m):                                          # buttresses: shadow left plane, light right plane
                xa, xb = xs[k], xs[k + 1]; xm = xa + (xb - xa) * q.uniform(.35, .6); t_ = talus + q.uniform(-14, 16); c.fill(np.array([(xa, tp[k] - 4), (xm, tp[k] - 4), (xm - 7, t_), (xa - 8, t_ + 12)]), sh); xr = xb - (xb - xa) * q.uniform(.10, .3); c.fill(np.array([(xr, tp[k] + 6), (xb, tp[k] + 4), (xb + 5, t_ - q.uniform(0, H_ * .2)), (xr + 5, t_ - q.uniform(.1, .5) * H_ * .5)]), lit)
                c.fill(np.array([(xa, tp[k] + H_ * .20), (xb, tp[k] + H_ * .17), (xb, tp[k] + H_ * .19), (xa, tp[k] + H_ * .225)]), sh)
                g = xa + q.uniform(0, 1) * (xb - xa); c.fill(np.array([(g, t_), (g + 16, t_ + 8), (g + 46, base), (g - 30, base)]), sh)
            c.fill(np.array([p for k in range(m) for p in ((xs[k], tp[k] - 6), (xs[k + 1], tp[k] - 6))] + [(xs[k], tp[min(k, m - 1)] + 5 + q.uniform(0, 4)) for k in range(m, -1, -1)]), lit); c.unclip()
        mesa(-160, 250, 566, 800, '#8f74cf', '#6a4fb0', '#553b9a', 2); mesa(1480, 1990, 590, 800, '#9a82d6', '#7258b8', '#5b41a2', 5); mesa(1010, 1190, 668, 800, '#a08add', '#7c63c0', '#674dae', 9)
        mesa(-260, 470, 640, 838, '#5f44a6', '#3a2478', '#26145a', 12); mesa(1640, 2150, 520, 838, '#6d50b6', '#422a86', '#2a1762', 15)
        L = c.layer(); img = comp.over(img, L)
        led = LEDGE + 9 * np.interp(np.arange(W), np.arange(0, W + 60, 60), r.normal(0, 1, W // 60 + 1)[:len(np.arange(0, W + 60, 60))] if len(np.arange(0, W + 60, 60)) <= W // 60 + 1 else r.normal(0, 1, len(np.arange(0, W + 60, 60)))); m = smooth(-1, 1.2, y - led[None]); n = fbm(HH, W, (30, 11, 4), (1, .6, .35), 31, aspect=2.6); gap = y - led[None]
        g = np.broadcast_to(rgb('#0c0620'), img.shape).copy(); g = np.where(((n > .30) & (gap < 120))[..., None], rgb('#22104e'), g); g = np.where(((n > .95) & (gap < 60))[..., None], rgb('#4a2c94'), g); g = g * (1 - .65 * smooth(30, 270, gap))[..., None]; g = np.where(((gap < 5 + 3 * n) & (n > -.9))[..., None], rgb('#c9b2f6'), g)
        self.led = led; return (img * (1 - m[..., None]) + g * m[..., None]).astype(f32)

    def frame(self, f):
        return self.a(f) if f < 20 else self.b(f)

    def a(self, f):
        x, y = self.tip(f); k = float(keys(f, [(0, 0.), (DOT - 1, 0.), (DOT, .25), (DOT + 1, 1.15), (DOT + 3, .9), (20, 1.)])); q = float(keys(f, [(0, 0.), (DOT, 0.), (DOT + 2, .35), (20, 1.)]))
        if f < 8: e = comp.ease(f / 8.); z = float(np.exp(math.log(2.25) * e)); src = self.zoom(self.plate, z); src = src if e < .04 else cv2.filter2D(src, -1, comp.disc_kernel(.6 + 8.4 * e))          # opens ON the match circle, r 208, sharp; racks focus as it pushes
        else: src = self.back
        img = src * mix(rgb('#ffffff'), rgb('#8d78cf'), q) * (1 - .22 * q)
        py = y - 14
        if k > 0: img = X.bend(self.backP * mix(rgb('#ffffff'), rgb('#8d78cf'), q) * (1 - .22 * q), x + PAD, py + PAD, 520, .62 * k, .30 * k)[PAD:PAD + HH, PAD:PAD + W].copy(); img = drain_(img, .42 * smooth(DOT, 20, f), x, py, 700, .20, '#a79fc4'); img = X.threads(img, x, py, 760, f, .85 * min(k, 1), 3, 54)
        m = self.hand(f); rgba = m.ndim == 3
        if rgba: img = comp.over(img, m)
        else:
            dt = cv2.distanceTransform((m > .5).astype(np.uint8), cv2.DIST_L2, 5); X_, Y_ = grid(); d = np.hypot(X_ - x, Y_ - py); b = np.broadcast_to(rgb('#0c0816'), img.shape) + (np.exp(-d / 150.) * k * .9)[..., None] * rgb('#6a28e0'); b = np.where((dt < 4.5 - 2.2 * smooth(0, 500, d))[..., None], rgb('#8a2cff') * .7 + rgb('#f0dcff') * (np.exp(-d / 260.) * k)[..., None] * .8, b); img = img * (1 - m[..., None]) + b * m[..., None]
        if k > 0: img = X.pearl(img, x, py, 12.5 * min(1.25, k), f, min(1., k))
        return img.astype(f32)

    def b(self, f):
        """20-24 the four; 25-27 the lens is pulled into the child; 28-39 close: the drop leaves his chin, falls, and stops at 36"""
        g = f - 20
        if f < 28:
            img = self.rev.copy(); m = self.figs(f)
            if m.ndim == 3: img = comp.over(img, m)
            else:
                dt = cv2.distanceTransform((m > .5).astype(np.uint8), cv2.DIST_L2, 5); rim = m * (kit.shifted(m, -5, 1) < .5); b = np.broadcast_to(rgb('#05030a'), img.shape) + (np.exp(-dt / 8.) * .3)[..., None] * rgb('#3d1f78'); b = b * (1 - rim[..., None]) + rgb('#e6d6ff') * rim[..., None]; img = img * (1 - m[..., None]) + b * m[..., None]
            e = float(keys(f, [(20, 0.), (24, .05), (25, .22), (26, .52), (27, .86), (28, 1.)], lambda t: t)); s = float(np.exp(math.log(1.04) * (1 - e) + math.log(3.4) * e)); cx, cy = 960 + (CHILD[0] - 960) * min(1, e * 1.15), 540 + (CHILD[1] - 540) * min(1, e * 1.15); img = comp.cam2d(img, cx, cy, s)
            if f >= 25: img = kit.zoomblur(img, 960, 540, .07 + .03 * (f - 25), 19)
            img = drain_(img, .25 + .50 * smooth(0, 10, g), 1672, 120, 620, .25, '#c4bfd2'); return self._pearl_soft(img, 1 - smooth(25, 28, f)).astype(f32)
        z = 1.0 + .0035 * (f - 28); img = comp.cam2d(self.close, 960, 540, z); y = self._y; x = self._x
        img = img + (np.exp(-((x - 1900) / 760) ** 2 - ((y - 60) / 560) ** 2))[..., None] * rgb('#8a2cff') * .55; img = drain_(img, .72, 1760, 90, 900, .30, '#c4bfd2')                              # the only colour left is the pearl's, from the right
        m = self.child(f)
        if m.ndim == 3: img = comp.over(img, m)
        else:
            dt = cv2.distanceTransform((m > .5).astype(np.uint8), cv2.DIST_L2, 5); rim = m * (kit.shifted(m, -13, 6) < .5); in_ = m * (kit.shifted(m, 8, 2) < .5); b = np.broadcast_to(rgb('#181222'), img.shape) + (np.exp(-dt / 60.) * .22)[..., None] * rgb('#5a4a90'); b = b * (1 - in_[..., None] * .6) + rgb('#3a3358') * (in_[..., None] * .6); b = b * (1 - rim[..., None]) + rgb('#eadcff') * rim[..., None]; img = img * (1 - m[..., None]) + b * m[..., None]
        cx, cy = self.chin(f); u = min(max(f, 28), 36) - 28; dy = 2.9 * u * u; sw = 1.0 + .022 * u if f < 36 else 1.04 - .12 * math.exp(-(f - 36) / 1.1) * math.cos((f - 36) * 2.6)
        if f < 30: img = self._neck(img, cx, cy + 16, cy + 16 + dy)
        return X.drop(img, cx, cy + 20 + dy, 36, sw * 1.14).astype(f32)

    def _neck(self, img, x, y0, y1):
        """the thread of water between chin and drop, breaking"""
        if y1 - y0 < 2: return img
        c = np.zeros((HH, W), f32); P = np.array([(x - 7, y0 - 6), (x - 2.2, (y0 + y1) / 2), (x - 8, y1), (x + 8, y1), (x + 2.2, (y0 + y1) / 2), (x + 7, y0 - 6)]); cv2.fillPoly(c, [np.round(P * 16).astype(np.int32)], 1., cv2.LINE_AA, 4); return img * (1 - .55 * c[..., None]) + rgb('#d9ccff') * (.45 * c)[..., None]

    def _pearl_soft(self, img, k):
        if k <= 0: return img
        o = np.zeros((HH, W, 4), f32); cv2.circle(o, (1672, 138), 76, (1, 1, 1, 1), -1, cv2.LINE_AA); bk = cv2.filter2D(o[..., 0], -1, comp.disc_kernel(20)) * k; hal = cv2.resize(cv2.GaussianBlur(cv2.resize(o[..., 0], (W // 4, HH // 4)), (0, 0), 26), (W, HH)) * k
        img = img + hal[..., None] * rgb('#8a2cff') * 1.1; return img * (1 - bk[..., None] * .92) + rgb('#f3e6ff') * bk[..., None] * .95


from chars import figs_lead as _FL                                  # the real hand and figures
Shot.hand = staticmethod(_FL.c3_hand(tip_c3)); Shot.figs = staticmethod(_FL.c3_figs(LEDGE))
from chars import gohan_terror as _T
Shot.child = staticmethod(_T.c3_child); Shot.chin = staticmethod(_T.c3_chin)
