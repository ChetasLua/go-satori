"""g4 (and any shot up there): the background of space, camera circling two fighters, Earth limb below, the violet moon's light from screen right.
    im, L = space(az, f=f, pitch=-7, roll=0, focal=900)          # az degrees = how far the camera has circled (the whole background pans true); about .5 s
        L['moon'] = (x, y, r) px (may lie off screen);  L['light'] -> hero3 light=;  L['rim'] -> hero3 rim= (violet, from the moon);  L['earth'] -> second rim: blue, from the limb below
    im = wash(im, side=(1, -.15), amt=.35)                       # the moon's violet veil laid over background AND figures so that they sit in one air
    lock=True keeps the moon at screen right whatever az (a rig light); lock=False: it is a thing in the world and passes by
Cheap version for fast cuts: aur=0, stars=.6, detail lower focal.  Static plate: cache one frame and comp.cam2d it."""
import numpy as np, cv2, math
from . import W, H
from .paint import f32, rgb, smooth, grid
from .sky_orbit import orbit_cam, earth, aurora, sun_dir, stars, P, RE
from . import moon as M

VIO = ['#0a0a2c', '#171542', '#2b2468', '#5a44b4', '#cdb4ff', '#ffdb9a', '#ff9d45']          # the night side under a violet moon
_g = {}


def _dir(pos, az, el):
    e = P(pos[0], pos[1], 0); up = e + [0, 0, RE]; up /= np.linalg.norm(up); n = np.array([0, 1., 0]) - up * up[1]; n /= np.linalg.norm(n); ea = np.cross(n, up); a, b = math.radians(az), math.radians(el); return (n * math.cos(a) + ea * math.sin(a)) * math.cos(b) + up * math.sin(b)


def space(az=0., f=0, pitch=-7., roll=0., focal=900., alt=900., pos=(0., -9.), moon=(52., 9., 4.6), sun=(-15., 252.), aur=.75, st=1., lock=False, cx=W / 2, cy=H / 2, lamps=1.):
    if 'x' not in _g: _g['x'], _g['y'] = grid()
    c = orbit_cam(alt=alt, pitch=pitch, yaw=az, roll=roll, focal=focal, pos=pos, cx=cx, cy=cy); im = earth(c, f, sun_dir(*sun), night=VIO, lamps=lamps, aircol=('#4a3cff', '#cdbcff', '#ff9a6a', '#8f7af0'), starry=st)
    if aur > 0: im = aurora(im, c, f, aur)
    m = _dir(pos, (az if lock else 0.) + moon[0], moon[1]); q = c.R @ m; L = dict(cam=c, light=(float(q[0]), float(-q[1]), float(-q[2])))
    L['rim'] = dict(dir=(float(q[0]), float(-q[1]), float(min(-q[2], -.15))), col='#c784ff', thr=.42, amt=.95); L['earth'] = dict(dir=(-.25, -.9, -.25), col='#7fb4ff', thr=.5, amt=.7); L['moon'] = None
    if q[2] > .03:
        x, y = q[0] / q[2] * focal + cx, q[1] / q[2] * focal + cy; r = focal * math.tan(math.radians(moon[2])) / q[2] ** .5; L['moon'] = (float(x), float(y), float(r))
        if -2200 < x < W + 2200 and -1600 < y < H + 1600:
            d = np.hypot(_g['x'] - x, _g['y'] - y) - r; im = im + M.halo_rgb(M.halo(np.maximum(d, 0), r * 1.4)).astype(f32) * .85; a = smooth(.8, -.8, d)[..., None]; im = im + (rgb(M.DEAD) * 1.12 - im) * a
    return np.maximum(im, 0).astype(f32), L


def wash(im, side=(1., -.15), amt=.35, col='#7a35e0'):
    """violet veil falling in from one side of the frame, hard-edged in two gouache steps"""
    if 'x' not in _g: _g['x'], _g['y'] = grid()
    n = math.hypot(*side); u = ((_g['x'] - W / 2) * side[0] / n + (_g['y'] - H / 2) * side[1] / n) / (W * .55); k = smooth(-.2, 1.1, u) * .75 + smooth(.60, .66, u + .02 * np.sin(_g['y'] * .012)) * .12 + smooth(.86, .91, u + .02 * np.sin(_g['y'] * .017 + 1)) * .13; return im + rgb(col) * (k * amt)[..., None] * (1 - np.clip(im, 0, 1) * .6)
