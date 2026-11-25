"""SKY: painted skies and cloud worlds a camera really moves through.  All pure functions of the frame, 0.3-1.1 s a frame, fork-safe, no disk plates (worlds build in < 1 s).
from db import sky

DAWN SEA  (a2, h4; also any hero shot above the clouds)
    D = sky.Dawn('a2' | 'h4');  c = D.cam(zoom=2..1, dx=0, dy=0, dz=0, tilt=0, roll=0)      zoom 2 -> sun at (960,486) r 208 (match circle); zoom 1 -> wide, r 104
    im, z = D.frame(f, c);  im, z, front = D.frame(f, c, split=700);  (x, y), px_per_m = D.at(c, (120, 520, -105))       <- the Nimbus lives about here
DIVE  (a3 40..79)
    V = sky.Dive();  im, z = V.frame(f)  f 0..15 POV fall through three decks (gold, cream, white);  im, c = V.burst(f)  f 0..23 pull-up over the bright sea;  V.cam(f) to place things
    im, z, front = V.frame(f, split=40.)                front = cloud nearer than 40 m, put it over your character (chase camera: pass any Camera3D as c)
CUMULUS OVER THE SEA  (a4, b, d3 from above)
    w = sky.CumulusWorld('day'|'gold'|'night'|'storm'|'dawn', seed);  im, z = w.frame(cam, f)          cam between 150 and 2600 m, looking along +Y
    H = sky.Hole();  im = H.frame(f, k)                  a4: looking up at one cumulus, k 0..1 = clean round hole punched through it, sun in the hole at (960,486); H.c = its camera
SEA  any height 2 m..30 km
    im, d = sky.sea(cam, f, 'day'|'gold'|'storm'|'night'|'dawn', sun=(x,y,z), sky=rgb, islands=sky.ISL, shadows=(size, cover, 0, 0), swell=220 high / 14 skimming)
STORM CEILING + RING  (f3, g1; ceiling at 620 m)
    R = sky.Ring();  im, z = R.frame(cam, f, k=430., gold=1., flash=0., skin='green'|'grey')      k = hole radius m (0 closed, grows on the punch), gold = aura light on its lip;  cam = sky.ring_cam(az, dist, eye_z, look_z, focal) or any Camera3D
CLIMB  (g2)
    C = sky.Climb();  im, z, ((x, y), px_per_m) = C.frame(f)   f 0..39 storm hole -> wall of light -> sunlit tops -> deep blue;  then sky.climb_space(f) f 40..79;  C.world(cam, f, st) any camera above the storm
ALTITUDE RAMP    im = sky.altitude(cam, metres, stars)          storm grey -> white -> blue -> indigo -> black, air skin on the horizon
ORBIT (km)  (g2-g5, g8)
    c = sky.orbit_cam(alt=420, pitch=-18, yaw=20, roll=-8, focal=1100, pos=(0,-9));  im = sky.earth(c, f, sky.sun_dir(elev, az));  im = sky.aurora(im, c, f, amt);  sky.P(lon, lat, alt) -> c.project(...)
    night: elev -14..-20; dawn line on the coast: elev -6..-2;  im = sky.stars(im, cam, amt, mask=...) anywhere
SPACE BACKGROUND, camera circling  (g4; g3)
    im, L = sky.space(az, f=f, pitch=-7, roll=0, focal=900, lock=False);  F.draw(cam, f, light=L['light'], rim=[L['rim'], L['earth']], pal='space');  im = sky.wash(im, amt=.35)       moon's violet light from screen right, Earth limb below
PINK DAWN, FIRST COLOUR  (g6, h1-h3)
    im = sky.pink_dawn(hz=620, seed=0, f=0);  im = sky.pink_zenith(f)  (straight up / its reflection in the lake, h1);  im = sky.first_colour(k, hz=640, f=0)  k 0 night .. 1 first dawn, colour spreads from the horizon
WALL OF LIGHT    im = sky.burst_white(im, f, k, col, col2, c=(x,y))        torn radial vapour, k 0..1
OWN WORLDS  db/sky_vol.py: Vol().cumulus / bank / ridge / deck / puff / build / render"""
import numpy as np, cv2, math
from . import W, H
from .paint import f32, rgb, smooth, ramp, fbm, grid, mix, Canvas, arcs
from .cam import Camera3D
from .comp import ease
from .sky_vol import Vol, tones, over_g, prem, soften, floor_field
from .sky_sea import sea, SEA, mixpal
from .sky_dawn import Dawn
from .sky_fly import Dive, CumulusWorld, cumulus_world, burst_white, day_sky, ISL, PAL
from .sky_storm import Ring, ring_cam, Climb, ZC
from .sky_orbit import orbit_cam, earth, aurora, stars, sun_dir, P, RE
from .sky_space import space, wash


_C = {}
def _climb():
    if 'c' not in _C: _C['c'] = Climb()
    return _C['c']


def altitude(c, alt, st=0.):
    """the sky seen from alt metres; st = stars 0..1"""
    return Climb.sky(None, c, alt, st)


def climb_space(f):
    """g2 from 40: rgb, camera (km).  The planet curves away under two lights that climb from 40 to 900 km."""
    u = (f - 40) / 39.; e = ease(u); alt = 110 + 700 * e ** 1.4; c = orbit_cam(alt=alt, pitch=-9 - 15 * e, yaw=34 - 30 * e, roll=-14 + 9 * e, focal=860 - 60 * e, pos=(-1.0, -4. - 5. * e)); return aurora(earth(c, f, sun_dir(-4.5, 78)), c, f, .55 + .6 * e), c


class Hole:
    """a4: straight up at one sunlit cumulus; a clean round hole with the sun in it"""

    def __init__(self):
        V = Vol(); V.cumulus(0, 0, 360, 760, 260, seed=77, peaks=5, depth=640, levels=2, n=90, size=.75, foot=0., skirt=.9); r = np.random.default_rng(5)
        for i in range(16): a = r.uniform(0, 6.28); d = r.uniform(420, 1100); V.cumulus(math.cos(a) * d, math.sin(a) * d, 400 + r.uniform(0, 160), r.uniform(160, 420), r.uniform(80, 200), seed=90 + i, foot=0., levels=2)
        V.build(sun=(.05, -.1, 1.), res=2048); self.V = V; self.x, self.y = grid(); self.c = Camera3D((0, -150, 2.), (0, 12, 420.), 900., cy=560)

    def frame(self, f=0, k=0., c=None, R=208., sun=(960, 486), sun_r=118):
        """k: 0 whole cloud, .02-1 the hole blows open to radius R px on screen with a rolled lip"""
        c = self.c if c is None else c; d = c.rays(); e = d[..., 2] / np.linalg.norm(d, axis=-1); im = ramp(np.clip(e, 0, 1), [(0, '#e8f6f6'), (.3, '#8fd0f2'), (.75, '#3f9ae0'), (1, '#2b7fd0')]).astype(f32); sx, sy = sun; D = np.hypot(self.x - sx, self.y - sy); im = im + (rgb('#fff6d8') - im) * (np.exp(-D / 260) * .85)[..., None]; im = im + (rgb('#fffdf2') * 1.15 - im) * smooth(sun_r + 1, sun_r - 1, D)[..., None]
        vx, vy = (sx - c.cx) / c.f, (sy - c.cy) / c.f; q = (self.V.C - np.asarray(c.eye)) @ c.R.T; ox, oy = q[:, 0] / q[:, 2] - vx, q[:, 1] / q[:, 2] - vy; r0 = np.hypot(ox, oy); th = np.arctan2(oy, ox); rr = R / c.f; ke = rr * ease(min(1., k * 1.1)) * (1 + .012 * np.sin(th * 5 + 1)) if k > 0 else None
        def mv(C):                                                        # the hole is cut along the line of sight, so it is a clean circle on screen
            r = np.sqrt(r0 ** 2 + ke ** 2 * 1.12 * np.exp(-(r0 / (ke * 3.2 + 1e-4)) ** 2)) + self.V.R / q[:, 2] * np.exp(-(r0 / (ke * 1.2 + 1e-4)) ** 2) * .9; lip = np.exp(-((r - ke) / (rr * .30)) ** 2); z = q[:, 2] * (1 - .035 * lip * k); return np.asarray(c.eye) + np.stack([(vx + np.cos(th) * r) * z, (vy + np.sin(th) * r) * z, z], -1) @ c.R
        g = self.V.render(c, tones('#ffffff', '#fdf3df', '#c6d8ee', '#8fb2de', '#6f97cf', '#b9d4ec'), fog=dict(d=4000., col='#d9eef8'), light=(.35, -.75, .30), shadow=False, tl=1.6, th=.06, eh=.34, skyl=.0, bounce=0., sil=1.0, silw=.42, silr=.50, sunscreen=(sx, sy), mixn=(.2, .3, .6), accs=.2, noise=.1, move=mv if k > 0 else None)
        if k > 0: g['a'] = g['a'] * smooth(R * ease(min(1., k * 1.1)) * .955, R * ease(min(1., k * 1.1)) * .975, D + 5 * np.sin(np.arctan2(self.y - sy, self.x - sx) * 13) * np.sin(np.arctan2(self.y - sy, self.x - sx) * 5 + 1))
        z = g['a']; im = over_g(im, g); m = cv2.resize((1 - z).astype(f32), (W // 4, H // 4), interpolation=cv2.INTER_AREA) * cv2.resize(np.exp(-(D / 420.) ** 2), (W // 4, H // 4)); G = cv2.resize(.4 * cv2.GaussianBlur(m, (0, 0), 3) + .35 * cv2.GaussianBlur(m, (0, 0), 9) + .3 * cv2.GaussianBlur(m, (0, 0), 26), (W, H)) * z; im = im + (rgb('#fffbe8') * 1.05 - im) * np.clip(G * 1.1, 0, .95)[..., None]
        return im


# ------------------------------------------------------------------ painted gradients with designed cloud bars (g6, h1-h3)
def _bars(im, hz, seed, f, cols, amt=1., n=1., y0=.04, y1=.55, drift=.15):
    """long flat bars of cloud in perspective: hard lit under-edge, lost top"""
    cols = [rgb(q) if isinstance(q, str) else q for q in cols]; h, w = im.shape[:2]; x, y = grid(h, w); t = (hz - y) / max(hz, 1); v = 1 / np.maximum(t, .035); u = (x - w / 2) / w * v * 1.2 + f * drift * .01 * v
    key = ('b', seed)
    if key not in _C: _C[key] = fbm(700, 2400, (150, 60, 24, 10), (1, .62, .36, .2), seed, aspect=4.5)
    T = _C[key]; s = lambda du, dv: cv2.remap(T, ((u * 150 + 1200 + du) % 2399).astype(f32), np.clip(v * 52 + dv, 0, 698).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    a = s(0, 0); b = s(2, 5.); m = smooth(.30 / n, .62 / n, a) * smooth(y0, y0 + .05, t) * (1 - smooth(y1 * .7, y1, t)); lit = smooth(.02, .16, b - a); core = smooth(.5, 1.5, a)
    col = cols[1] + (cols[2] - cols[1]) * core[..., None]; col = col + (cols[0] - col) * lit[..., None]; return im + (col - im) * (m * amt)[..., None]


def pink_dawn(hz=620, seed=0, f=0, w=W, h=H, cx=.62):
    """H: clear pink dawn, few quiet clouds. hz = horizon y"""
    x, y = grid(h, w); t = (hz - y) / max(hz, 1); n = fbm(h // 2, w // 2, (200, 80), (1, .5), seed + 3); n = cv2.resize(n, (w, h)); im = ramp(t + .012 * n + .05 * ((x / w - cx)) ** 2, [(-1, '#f2c4b0'), (0, '#fff0d8'), (.07, '#ffe1c6'), (.2, '#ffd0bc'), (.38, '#f7bdb4'), (.58, '#dcbccb'), (.8, '#b8c3e0'), (1.05, '#9fc3e6')]).astype(f32)
    g = np.exp(-np.hypot((x - w * cx) / (w * .30), (y - hz) / (h * .26)) * 1.3); im = im + (rgb('#fff6e2') - im) * (g * .8)[..., None]; im = _bars(im, hz, 21 + seed, f, ('#fff2e0', '#f2b8ae', '#d9a6b6'), .85, 1.); im = _bars(im, hz, 35 + seed, f, ('#ffe9dc', '#dcb4c4', '#c0a4c4'), .55, .9, .34, 1.0, .08); return np.clip(im * (1 + .010 * cv2.resize(fbm(h // 2, w // 2, (3, 1.5), (1, .7), 9), (w, h))[..., None]), 0, 1.2)


def pink_zenith(f=0, seed=0, w=W, h=H):
    """the sky straight overhead (h1: also the picture that lies in the lake): pale blue to pink, two combed feathers of cirrus"""
    x, y = grid(h, w); q = (x / w * .55 + y / h * .75); n = cv2.resize(fbm(h // 2, w // 2, (220, 90, 34), (1, .5, .25), seed + 8), (w, h)); im = ramp(q + .03 * n, [(0, '#a9c9e8'), (.35, '#c3cbe6'), (.6, '#e9c6d0'), (.85, '#f9c8bc'), (1.3, '#ffdcc4')]).astype(f32)
    if ('z', seed) not in _C: a = fbm(h, w, (170, 70, 28, 11), (1, .62, .42, .28), seed + 14, aspect=6.); M = cv2.getRotationMatrix2D((w / 2, h / 2), 24, 1.5); _C['z', seed] = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    a = np.roll(_C['z', seed], (int(f * .12), int(-f * .5)), (0, 1)); m = smooth(.35, 1.15, a); e = smooth(.02, .16, a - np.roll(a, (5, -4), (0, 1))); col = rgb('#f6ccc8') + (rgb('#fff3e6') - rgb('#f6ccc8')) * e[..., None]; return im + (col - im) * (m * .72)[..., None]


def first_colour(k, hz=640, f=0, seed=0, w=W, h=H, cx=.5, st=1.):
    """g6: k 0 = deep night, 1 = the first dawn colour has spread up from the horizon (rose, then gold); stars drown"""
    x, y = grid(h, w); t = (hz - y) / max(hz, 1); r = np.hypot((x / w - cx) * 1.5, np.maximum(t, 0) * 1.05); n = cv2.resize(fbm(h // 2, w // 2, (180, 70), (1, .5), seed + 4), (w, h))
    im = ramp(t + .01 * n, [(-1, '#070b1c'), (0, '#26325e'), (.12, '#1c2a58'), (.4, '#111c44'), (1.1, '#080e28')]).astype(f32)
    if st > 0 and k < .95: im = stars(im, Camera3D((0, 0, 0), (0, 1, .18), 1100.), st * (1 - k) ** 1.2, mask=np.where(t > 0, 1., 0.).astype(f32))
    D = ramp(t + .012 * n, [(-1, '#5a4a78'), (0, '#ffd9a0'), (.05, '#ffb98a'), (.13, '#f0929a'), (.26, '#b57ab4'), (.45, '#6a6cb4'), (.75, '#344a94'), (1.1, '#1c2c68')]).astype(f32); m = smooth(0, .62, k * 1.75 - r * .95 + .05 * n)[..., None]
    im = im + (D - im) * m; im = _bars(im, hz, 51 + seed, f, tuple(rgb(a) + (rgb(b) - rgb(a)) * k for a, b in zip(('#3a4a80', '#1c2750', '#141c3e'), ('#ffd0a0', '#c8849c', '#6f5f9a'))), .9, 1.); return im
