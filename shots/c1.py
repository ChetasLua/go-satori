"""c1 THE MOON.  Match circle (960,486) r 208 held for the whole shot.  The clouds part; a small figure comes down in front of the disc;
the sea slows and stops when it arrives; sparks hang; at 60 two eyes.  After 56 nothing moves but the eyes.
fig_c1(f) and eyes_c1(f) are plain placeholders: replace them (or assign Shot.fig / Shot.eyes)."""
import numpy as np, cv2, math
from db import W, H as HH, comp, kit, moon as M
from db.paint import rgb, f32, smooth

CX, CY, R = M.C0
HZ = 792
keys = comp.lerp_keys


def path_c1(f):
    """(x, y, px per head, open 0..1): pelvis position of the descending figure on twos.  It falls slowly, brakes, and hangs."""
    g = (int(f) // 2) * 2
    y = float(keys(g, [(0, -190.), (12, -190.), (40, 452.), (48, 520.), (54, 512.), (80, 512.)], lambda t: 1 - (1 - t) ** 2.2))
    return CX, y, 42., float(keys(g, [(0, 0.), (26, 0.), (44, .55), (54, 1.), (57, .97), (80, .97)]))


def fig_c1(f):
    """PLACEHOLDER: a plain pin-figure mask, float (1080,1920)"""
    x, y, s, o = path_c1(f); m = np.zeros((HH, W), f32); p = lambda L: np.round((np.array(L) * s + (x, y)) * 16).astype(np.int32)
    cv2.circle(m, tuple(p([(0, -3.45)])[0].tolist()), int(.48 * s * 16), 1., -1, cv2.LINE_AA, 4); cv2.fillPoly(m, [p([(-.72, -2.75), (.72, -2.75), (.42, -.4), (.16, 4.1), (-.16, 4.1), (-.42, -.4)])], 1., cv2.LINE_AA, 4)
    for d in (-1, 1):
        a = math.radians(78 - 62 * o); e = np.array([d * (.66 + 2.6 * math.cos(a)), -2.55 + 2.6 * math.sin(a)]); cv2.fillPoly(m, [p([(d * .55, -2.72), (d * .62, -2.2), e + (0, .16), e - (0, .16)])], 1., cv2.LINE_AA, 4)
    return m


def eyes_c1(f):
    """PLACEHOLDER: [(x, y, opening 0..1)] on screen"""
    x, y, s, _ = path_c1(f); o = float(keys(f, [(0, 0.), (59, 0.), (60, .35), (61, 1.), (80, 1.)]))
    return [(x + d * .17 * s, y - 3.42 * s, o) for d in (-1, 1)]


def back_lit(img, m, L=1.):
    """put a figure (mask, or premultiplied RGBA) in front of the disc: near-black body, light wrapping its edge, hot where the disc is behind"""
    if m.ndim == 3: return comp.over(img, m)
    dt = cv2.distanceTransform((m > .5).astype(np.uint8), cv2.DIST_L2, 5); lit = cv2.GaussianBlur(img, (0, 0), 9).mean(-1)
    b = np.broadcast_to(rgb('#0a0712'), img.shape) + (np.exp(-dt / 9.) * np.clip(lit, 0, 1) * .35)[..., None] * rgb('#8a2cff')
    b = np.where((dt < 1.9 + 1.3 * np.clip(lit, 0, 1))[..., None], rgb('#8a2cff') * np.clip(.25 + lit, 0, 1)[..., None] + rgb('#f0dcff') * (np.clip(lit - .5, 0, 1) * .55)[..., None], b)
    return img * (1 - m[..., None]) + b * m[..., None]


class Shot:
    id = 'c1'; n = 80
    finish = dict(bloom=.26, vig=.20, grain=.010, soft=.10)
    EVENTS = [(0, 'match', dict(note='circle matches b5; sea bed and a low hollow tone, no music')), (0, 'cloud_part', dict(frames=44, note='almost silent, a long breath of air')),
              (14, 'descend', dict(frames=40, note='no sound of its own')), (30, 'spark', dict(size=.2)), (36, 'spark', dict(size=.25)), (42, 'spark', dict(size=.3)),
              (48, 'sea_stops', dict(note='water sound dies over 12 frames; by 56 total silence')), (54, 'hang', dict(note='arms fully open, everything still')), (60, 'eyes', dict(size=1., note='two red eyes open; single dry hit'))]
    fig = staticmethod(fig_c1); eyes = staticmethod(eyes_c1)

    def setup(self):
        self.sky = M.Sky(CX, CY, R, 3)
        self.ts = np.cumsum([float(keys(f, [(0, 1.), (30, 1.), (48, .25), (56, 0.), (80, 0.)])) for f in range(90)]) / 24.      # water clock: slows to a stop

    def frame(self, f):
        part = float(keys(f, [(0, .02), (6, .04), (34, .86), (50, 1.), (80, 1.)])); td = float(self.ts[f]) * 24
        img = self.sky.frame(td, part)
        up = smooth(.0, .8, part); M.sea(img, HZ, CX, CY, R, t=float(self.ts[f]) * 1.15, amt=.62 + .38 * up, col='#%02x%02x%02x' % tuple(int(v) for v in np.array([150, 96, 236]) * (1 - up) + np.array([251, 247, 255]) * up))
        M.sparks(img, min(f, 56), 3, avoid=(CX, CY, 262), born=float(np.clip((f - 26) / 30., 0, 1)), amt=.95)
        m = self.fig(f); img = back_lit(img, m)
        a = m if m.ndim == 2 else m[..., 3]                                # the figure breaks the road: a thin dark column in the glitter
        ys = np.arange(HH)[:, None]; k = np.clip((ys - HZ) / (HH - HZ), 0, 1); col = a[HZ - 330:HZ].max(0) if a[HZ - 330:HZ].max() > 0 else np.zeros(W, f32); col = cv2.GaussianBlur(col[None], (0, 0), 9)[0]
        img *= 1 - (.26 * col[None] * np.where(ys >= HZ, 1 - .75 * k, 0))[..., None]
        for x, y, o in self.eyes(f):
            if o <= 0: continue
            g = np.zeros((HH, W), f32); cv2.ellipse(g, (int(x * 16), int(y * 16)), (int(5.2 * 16), int(max(.5, 2.3 * o) * 16)), 0, 0, 360, 1., -1, cv2.LINE_AA, 4); q = cv2.GaussianBlur(g, (0, 0), 8) * 2.2 + cv2.GaussianBlur(g, (0, 0), 2)
            img = img + (q * o)[..., None] * rgb('#ff1f2c') * .9 + g[..., None] * rgb('#ff5a50') * 1.1
        return img.astype(f32)


from chars import figs_lead as _FL                                  # the real figure
Shot.fig = staticmethod(_FL.c1_fig(path_c1)); Shot.eyes = staticmethod(_FL.c1_eyes(path_c1, Shot.__dict__['fig'].__func__))
