"""g9 WHITE.  0-24 rage, 25-49 disbelief, 50-59 empty (and small), all inside the light; 58 the disc (match circle 960,486 r 208) breaks into rings,
petals of light; 70 everything is white; white held to 79.
head_g9(f, expr, x, y, s) is a plain placeholder: s = px of the head's height.  Mask or premultiplied RGBA.  It faces LEFT, into the light."""
import numpy as np, cv2, math
from db import W, H as HH, comp, kit, moon as M, moon_fx as X
from db.paint import rgb, f32, smooth, mix, grid, ramp, lum
keys = comp.lerp_keys
CX, CY, R = M.C0


HOLD = [(0, X._lut(X.LIGHT)), (8, X._lut(X.LIGHT_GOLD)), (16, X._lut(X.LIGHT_BLUE)), (24, X._lut(X.LIGHT))]      # each light is held 8 frames and cross-fades over 5 (photosensitive safety: never faster)


VIA = X._lut([(0, '#3a12a0'), (.20, '#6a2ce0'), (.202, '#8f5cf6'), (.34, '#b48cff'), (.342, '#d2b8ff'), (.50, '#ecdfff'), (.502, '#f7f0ff'), (.66, '#fdfaff'), (1, '#ffffff')])


def light_g9(f):
    """colour table of the streams at frame f"""
    i = max(k for k, (a, _) in enumerate(HOLD) if a <= f); u = f - HOLD[i][0]
    if i == 0 or u >= 5: return HOLD[i][1]
    e = float(comp.ease((u + 1) / 6.)); return HOLD[i - 1][1] * (1 - e) ** 2 + VIA * 2 * e * (1 - e) + HOLD[i][1] * e * e          # gold and blue meet through violet, not through grey


def level_g9(f): return float(keys(f, [(0, .640), (8, .655), (16, .630), (24, .655), (34, .82), (49, .945)]))      # mean luma the light is governed to: a slow curve, a few per cent of swing


def expr(f): return 'rage' if f < 25 else 'disbelief' if f < 50 else 'empty'


def place(f):
    """(x, y, s): rage fills the frame; disbelief eases back; empty hangs small in the centre of the match circle"""
    if f < 25: return 1080., 560., float(keys(f, [(0, 640.), (24, 690.)], lambda t: t))
    if f < 50: return float(keys(f, [(25, 1060.), (49, 1010.)])), float(keys(f, [(25, 552.), (49, 530.)])), float(keys(f, [(25, 520.), (30, 486.), (49, 396.)]))
    return CX, CY + 4., float(keys(f, [(50, 92.), (60, 84.), (70, 70.)], lambda t: t))


def head_g9(f, e, x, y, s):
    """PLACEHOLDER: a plain egg on a plain neck"""
    m = np.zeros((HH, W), f32); p = lambda L: np.round((np.array(L, float) * s + (x, y)) * 16).astype(np.int32)
    cv2.ellipse(m, tuple(p([(0, -.1)])[0].tolist()), (int(.36 * s * 16), int(.48 * s * 16)), -6, 0, 360, 1., -1, cv2.LINE_AA, 4); cv2.fillPoly(m, [p([(-.14, .25), (.2, .25), (.24, .55), (.8, .78), (.9, 2.2), (-.8, 2.2), (-.7, .78), (-.16, .55)])], 1., cv2.LINE_AA, 4); return m


class Shot:
    id = 'g9'; n = 80
    finish = dict(bloom=.24, vig=.08, grain=.009, soft=.10)
    EVENTS = [(0, 'inside_light', dict(note='a held roar, all air, no impact')), (25, 'expr', dict(note='the roar loses its low end')), (50, 'expr', dict(note='everything drops to one thin high note: he is small')), (58, 'break', dict(size=1., note='one glass bell; the circle opens')),
              (60, 'rings', dict(size=.8)), (62, 'petals', dict(size=.5)), (70, 'silence', dict(frames=10, note='white, total silence; runs on into h1'))]
    head = staticmethod(head_g9)

    def setup(self): pass

    def frame(self, f):
        e = expr(f); x, y, s = place(f); g = f if f < 25 else 25 + (f - 25)
        if f >= 70: return np.broadcast_to(rgb(M.DEAD) * 1.06, (HH, W, 3)).astype(f32).copy()
        # 0-49: the light pours in from the left and parts round the head; the streams run to a point far off on the right
        S_ = [(0, .17), (24, .13), (36, .07), (50, .02)]
        if f < 50:
            shk = kit.shake(f, 5.5 * (1 - smooth(16, 30, f)), 7); x2, y2 = x + shk[0], y + shk[1]; ph = float(sum(float(keys(q, S_)) for q in range(f))); hush = float(keys(f, [(0, 0.), (24, .04), (49, .72)]))
            lay = self.head(f, e, x2, y2, s); rgba = lay.ndim == 3; m = lay[..., 3] if rgba else lay
            st = light_g9(f); lv = .13
            img = X.beam_inside(m, palm=(x2 - .40 * s, y2 - .12 * s), t=ph * 1.5, vp=(2650., 500.), stops=st, body='#1c0f3a', glow='#6a3ae0', tatter=0., wrap=1.15, reach=80 + .13 * s, figure=lay if rgba else None, heart=0., hush=hush, seed=11, lift=lv, freq=4.2, length=.42, wake=1.)
            w_ = (1 - m)[::6, ::6]; mu = float((lum(np.clip(img[::6, ::6], 0, 1)) * w_).sum() / (w_.sum() + 1e-6)); tg = level_g9(f)                   # governor: the mean follows a slow curve whatever the palette does
            bg = (1 - (1 - np.clip(img, 0, 1)) ** (math.log(max(1e-3, 1 - tg)) / math.log(max(1e-3, min(.995, 1 - mu))))) if mu < tg else img * (tg / mu); img = img * m[..., None] + bg * (1 - m)[..., None]
            on = smooth(26, 36, f)
            if on > 0:                                                  # the disc behind him, at last seen; it is what is breaking: cracks of light in IT
                r2 = float(keys(f, [(26, 470.), (49, 430.)])); cy = 486.; B = M.ball(960., cy, r2, crack=float(keys(f, [(26, .02), (38, .40), (49, .66)])), seed=4, tilt=.3, dark=.62); A = (B[..., 3] * on * (1 - m) * .92)[..., None]
                img = img * (1 - A) + B[..., :3] * on * .92 * (1 - m)[..., None]; X_, Y_ = grid(); d = np.hypot(X_ - 960., Y_ - cy); rim = np.exp(-((d - r2) / 3.2) ** 2) * on * (1 - m); img = img * (1 - .8 * rim[..., None]) + rgb('#9a5cff') * (.8 * rim)[..., None]
            return img.astype(f32)
        # 50-69: c1 again, emptied: the white disc, the violet dark, and something very small in front of it
        X_, Y_ = grid(); d = np.hypot(X_ - CX, Y_ - CY); k = max(0., (f - 57) / 12.); bg = ramp(np.clip((d - R) / 980, 0, 1), [(0, '#c9a6ff'), (.06, '#9a66f0'), (.22, '#6234bf'), (.6, '#371c80'), (1, '#1d0d4c')]); m0 = smooth(R + .7, R - .7, d); calm = smooth(50, 58, f)
        img = bg * (1 + .10 * calm) * (1 - m0[..., None]) + rgb(M.DEAD) * m0[..., None]
        q = 1 - comp.ease((f - 49) / 6.) if f < 55 else 0.; img = img * (1 - .80 * q) + rgb(M.DEAD) * (.80 * q)           # the glare of the cut dies away over 6 frames (no flash)
        e_ = float(np.clip(k * 1.6, 0, 1))
        if k > 0: img = img * (1 - (m0 * smooth(.0, .5, k))[..., None]) + (rgb('#f2e6ff') * smooth(0, .5, k)) * m0[..., None]
        img = X.burst(img.astype(f32), CX, CY, R * 1.35, min(k, .98), f * 1.0, petals=120, rings=4) if k > 0 else img
        lay = self.head(f, e, x, y, s); rgba = lay.ndim == 3; m = lay[..., 3] if rgba else lay; ff = 1.
        if rgba: img = comp.over(img, lay, ff)
        else:
            dt = cv2.distanceTransform((m > .5).astype(np.uint8), cv2.DIST_L2, 3); b = np.where((dt < 2.2)[..., None], rgb('#b98aff'), rgb('#1c0f3a')); img = img * (1 - (m * ff)[..., None]) + b * (m * ff)[..., None]
        return X.burnout(img, float(np.clip((f - 60) / 9., 0, 1)) ** 1.25, CX, CY, 30).astype(f32)


from chars import figs_lead as _FL                                  # the real figure
Shot.head = staticmethod(_FL.g9_head)
