"""b5 DUSK.  Kame house from outside, blue hour.  Behind the curtain: Goku and Vegeta argue over nothing, Gohan bounces, Krillin and Roshi laugh, Piccolo stands.
24: the wind-chime stops dead, every head turns; 25-31 the four stars go out one by one; the ball, dark, comes to the match circle (960, 486) r 208 at 37 and stays."""
import numpy as np, cv2, math, os
from scipy.interpolate import PchipInterpolator
from db import W, H, comp
from db.paint import f32, smooth
from db.cam import Camera3D
from db.ink import Cel, spl
from db.land_kame import Dusk, WX0, WX1, WZ1
from db.hero3 import Fig, P, POSES, mix_pose

K = 700; TW, TH = int((WX1 - WX0) * K), int(WZ1 * K); VER = 6
U = PchipInterpolator([0, 12, 24, 31, 37, 60], [0, .13, .34, .74, 1, 1])
RINGS = (2, 9, 16)


def _cam(): return Camera3D(((WX0 + WX1) / 2, -9., WZ1 / 2), ((WX0 + WX1) / 2, 0, WZ1 / 2), K * 9., TW, TH)


def _pose(base, **k): return P(base, **k)


def acting(g):
    """who does what at frame g (even): list of (key, Fig, pose, x, heading, z offset, colour, softness)"""
    b = [0, 1, .4, -.2][(g // 2) % 4]; q = g % 8; still = g >= 24
    L = []
    # Piccolo: a post with folded arms; he feels it first (20)
    pk = smooth(19, 23, g); L.append(('piccolo', 'clean', _pose('arms_crossed', nck=(6 - 12 * pk, 0, 14 * pk), hed=(6 - 16 * pk, 0, 34 * pk)), -.80, math.pi + .25, -.84, .60, '#d08a5c', 2.4))
    # Gohan: jumps for joy on the beat, hat and ball
    up = 0. if still else max(0., math.sin(math.pi * (g % 10) / 10.)) * .06
    if still: gp = _pose('stand', nck=(-8, 0, 20), hed=(-14, 0, 40 if g > 24 else 16))
    else: gp = mix_pose(POSES['stand'], POSES['hands_up'], .35 + .65 * (up / .06))
    L.append(('gohan', 'clean', gp, -.535, math.pi - .35, -.60 + up, .70, '#8a4226', 1.4))
    # Vegeta, facing right
    if g < 8: th = [0, 1, .55, .2][(g // 2) % 4]; vp = _pose('stand', wst=(5 + 6 * th, 0, 0), cst=(4, 0, 0), nck=(9, 0, 0), hed=(-6, 0, 0), hR=(.10, .56 + .22 * th, .10 + .10 * th), eR=(.8, -.3, -.6), gR='point', wR=(-40, 0, 0), hL=(.30, -.02, -.90), gL='fist'); vh = -math.pi / 2
    elif g < 16: th = [1, .3, .9, .2][((g - 8) // 2) % 4]; vp = _pose('stand', wst=(10, 0, 0), cst=(6, 0, 0), nck=(12, 0, 0), hed=(-10 - 4 * th, 0, 0), hR=(.12, .70 + .12 * th, -.12 + .16 * th), eR=(.8, -.3, -.6), gR='point', wR=(-40, 0, 0), hL=(.12, .58 + .10 * th, -.30), eL=(.8, -.2, -.6), gL='fist', cL=(2 + 2 * th, 0), cR=(2 + 2 * th, 0)); vh = -math.pi / 2
    elif not still: vp = _pose('arms_crossed', cst=(-8, 0, 0), nck=(-2, 0, 0), hed=(-16, 4, 0)); vh = math.pi / 2 - .15
    elif g < 26: vp = _pose('arms_crossed', nck=(-2, 0, 0), hed=(-12, 0, 10)); vh = math.pi - .1
    else: vp = _pose('arms_crossed', nck=(-6, 0, 0), hed=(-16, 0, 0)); vh = -math.pi / 2 - .25
    L.append(('vegeta', 'clean', vp, -.265 if not still else -.205, vh, -.80, .68, '#8a4226', 1.1))
    gx = .150
    # Goku, facing left: shrugs, then laughs with his whole back
    if g < 8: th = [.2, .6, 1, .6][(g // 2) % 4]; gk = _pose('stand', wst=(4, 0, 0), nck=(8, 0, 0), hed=(-4 + 6 * th, 0, 0), cL=(3 + 3 * th, 0), cR=(3 + 3 * th, 0), hL=(.62, .42, -.22 + .10 * th), hR=(.62, .42, -.22 + .10 * th), eL=(.4, -.6, -.8), eR=(.4, -.6, -.8), gL='open', gR='open', wL=(80, 0, 0), wR=(80, 0, 0)); gh = math.pi / 2; dz = 0
    elif g < 16: th = [1, .35, 1, .2][((g - 8) // 2) % 4]; gk = _pose('stand', wst=(-5, 0, 0), cst=(-7 - 4 * th, 0, 0), nck=(-11, 0, 0), hed=(-18 - 6 * th, 0, 0), cL=(4 * th, 0), cR=(4 * th, 0), hL=(.22, .30, -.60), hR=(-.10, .52, -.46), eL=(.8, -.5, -.3), eR=(.9, .1, -.5), gL='open', gR='open'); gh = math.pi / 2; dz = .010 * th; gx = .030
    elif not still: th = [1, .35, 1, .2][((g - 16) // 2) % 4]; gk = _pose('stand', wst=(8, 0, 0), cst=(4 - 5 * th, 0, 0), nck=(2, 0, 0), hed=(-14 - 8 * th, 0, 0), cL=(3 * th, 0), cR=(3 * th, 0), hR=(.10, .86, -.02), eR=(.6, -.4, -.8), gR='open', wR=(-30, 0, 0), hL=(.24, .20, -.70), gL='open'); gh = math.pi / 2; dz = .008 * th
    elif g < 26: gk = _pose('stand', nck=(0, 0, 0), hed=(-10, 0, 16)); gh = math.pi - .15; dz = 0
    else: gk = _pose('stand', nck=(-6, 0, 0), hed=(-16, 0, 0)); gh = -math.pi / 2 - .30; dz = 0
    L.append(('goku', 'clean', gk, gx, gh, -.80 + dz, .66, '#8a4226', 1.1))
    return L


def friends(g):
    """Krillin and old Roshi, drawn by hand: round heads thrown back, shoulders going up and down.  -> list of (alpha, col, soft)"""
    still = g >= 24; j = [0, 1, .3, .8][(g // 2) % 4] * (not still); out = []
    def T(Q, o=(-.80, 0.), sc=.92): Q = (np.asarray(Q, float) - o) * [sc * .9, sc] + o; return np.c_[(Q[:, 0] - WX0) * K, (WZ1 - Q[:, 1]) * K]
    def rot(Q, o, a): c_, s_ = math.cos(a), math.sin(a); return (np.asarray(Q, float) - o) @ np.array([[c_, s_], [-s_, c_]]) + o
    c = Cel(TW, TH, 0, 0, 1); o = np.array([-.665, .17 + .012 * j]); a = .10 + (.22 * j if not still else -.30)
    c.fill(T(spl(np.array([(-.78, -.2), (-.785, .04), (-.755, .115 + .02 * j), (-.70, .145 + .02 * j), (-.63, .145 + .02 * j), (-.575, .115 + .02 * j), (-.545, .04), (-.55, -.2)]), n=6)), '#000000')
    c.fill(T(rot(spl(np.array([(-.70, .13), (-.71, .17), (-.755, .19), (-.762, .23), (-.745, .28), (-.70, .312), (-.645, .312), (-.60, .285), (-.582, .24), (-.588, .215), (-.572, .20), (-.588, .185), (-.612, .165), (-.63, .13)]), n=6), o, a)), '#000000'); out.append((c.rgba()[..., 3], '#b4683e', 2.))
    c = Cel(TW, TH, 0, 0, 1); o = np.array([-.93, .17 + .01 * j]); a = -.05 + (.14 * (1 - j) if not still else .12)
    c.fill(T(spl(np.array([(-1.06, -.2), (-1.05, .06), (-1.0, .115 + .015 * j), (-.94, .135 + .015 * j), (-.87, .115 + .015 * j), (-.83, .06), (-.82, -.2)]), n=6)), '#000000')
    c.fill(T(rot(spl(np.array([(-.955, .12), (-.965, .16), (-1.0, .19), (-1.005, .24), (-.97, .30), (-.915, .31), (-.87, .28), (-.857, .235), (-.83, .228), (-.83, .205), (-.855, .20), (-.85, .18), (-.80, .17), (-.79, .11), (-.83, .04), (-.87, .06), (-.89, .12)]), n=6), o, a)), '#000000'); out.append((c.rgba()[..., 3], '#b4683e', 2.))
    return out


class Shot:
    id = 'b5'; n = 40
    finish = dict(bloom=.34, vig=.20, soft=.14)
    EVENTS = [(0, 'laughter_bed', dict(to=23, size=.7, note='a family laughing behind a window, warm, muffled')), (0, 'wind_bed', dict(to=23, size=.25)), (0, 'sea_far', dict(to=39, size=.25)),
              (2, 'chime', dict(size=.8)), (6, 'argue', dict(size=.2, note='two men arguing about nothing, muffled')), (9, 'chime', dict(size=.6)), (16, 'chime', dict(size=.75)),
              (24, 'chime_stop', dict(note='dead stop: chime, wind and laughter cut on this frame, only the sea')), (25, 'star_out', dict(k=3)), (27, 'star_out', dict(k=2)), (29, 'star_out', dict(k=1)), (31, 'star_out', dict(k=0, note='last bell, reversed')),
              (32, 'glow_dead', dict(note='near silence')), (36, 'violet', dict(size=.2, note='one low note begins; leads into c1')), (37, 'match', dict(note='circle (960,486) r 208'))]

    def setup(self):
        self.D = Dusk().setup(); p = f'cache/b5_sil_v{VER}.npz'; self.keys = list(range(0, 30, 2))
        if os.path.exists(p): z = np.load(p); self.sil = {int(k[1:]): z[k] for k in z.files}
        else:
            cam = _cam(); self.sil = {}
            for g in self.keys:
                st = []
                for (ch, var, pose, x, h, z0, sc, col, so) in acting(g):
                    F = Fig(ch, var, scale=sc).pose(pose).place(x, 0, z0, heading=h); L = F.draw(cam, g, shadow=False, lines=0.); st.append((L[..., 3] * 255).astype(np.uint8))
                self.sil[g] = np.stack(st)
            os.makedirs('cache', exist_ok=True); np.savez_compressed(p, **{'k%d' % g: v for g, v in self.sil.items()})

    def layers(self, f):
        g = min((int(f) // 2) * 2, 28); act = acting(g); S = self.sil[g]; fr = friends(g)
        mx = lambda *q: np.maximum.reduce([np.asarray(v, f32) / (255. if v.dtype == np.uint8 else 1.) for v in q])
        out = [(mx(S[0]), '#d79a6a', 2.4), (mx(*[v[0] for v in fr]), '#b06a40', 1.7), (mx(S[1], S[2], S[3]), '#8a4226', 1.1)]
        return out

    def swing(self, f):
        """chime: the glass bell on its string and the paper below it; both at rest, instantly, from 24"""
        if f >= 24: return 0., 0.
        t = f / 24.; gust = .55 + .45 * math.sin(t * 5.1 + .4); a = .085 * gust * math.sin(t * 8.6 + .3) + .03; s = .36 * gust * math.sin(t * 8.6 - .55) + .14 * math.sin(t * 13.1) + .12
        return a, s

    def frame(self, f):
        u = float(U(f)); lit = float(comp.lerp_keys(f, [(0, 1.), (25, 1.), (30, .55), (34, .20), (38, .05), (40, .05)], lambda t: t)); dim = float(smooth(25, 37, f))
        st = [float(1. if f < k else max(0., 1 - (f - k + 1) / 2.)) for k in (31, 29, 27, 25)]; glow = float(np.mean(st)) ** .8
        if 24 <= f < 32: glow *= [1, .86, .97, .80, .93, .72, .85, .6][f - 24]
        ring = max([max(0., 1 - (f - k) / 5.) for k in RINGS if f >= k] + [0.]); ring = ring if f < 24 else 0.
        wind = 0. if f >= 24 else .05 * math.sin(f * .42) + .03 * math.sin(f * .9)
        return self.D.frame(u, lit, self.swing(f), self.layers(f), glow, f, ring, stars=tuple(st), dim=dim, wind=wind, rim=float(smooth(33, 38, f)))
