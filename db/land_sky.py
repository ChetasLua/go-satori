"""Skies that belong to the places of land.py (the sky lane's pictures can replace any of them: pass sky=...)."""
import numpy as np, cv2, math
from .paint import rgb, f32, fbm, smooth, mix, grid, Canvas, arcs, jit_color, ramp, warp_by, lum
from .comp import blur, radial
from . import clouds
from .land_core import C, hx, paint_over, horizon


def gradient(h, w, hz, stops, bow=0., cx=None, seed=0, wob=.010):
    """ramp measured from the horizon (0) up to the frame top (1) and down to the bottom (-1)"""
    x, y = grid(h, w); cx = w / 2 if cx is None else cx; t = (hz - y) / max(abs(hz), h * .4) + bow * ((x - cx) / w) ** 2 + wob * fbm(h, w, (420, 150), (1, .5), seed + 1)
    return ramp(np.clip(t, -1, 1.4), stops)


def brush(img, seed=0, amt=.5, sizes=(46, 20), val=(.016, .014), tilt=0.):
    h, w = img.shape[:2]; return paint_over(img, seed, sizes, (amt * .6, amt * .5), val, .004, .5, (.006, .010), fbm(h, w, (500, 180), (1, .5), seed + 5) * .14 + tilt, 5.5, bristle=.03)


def torn(img, y0, y1, dark, lit, rim, src, seed=0, cover=.5, aspect=3.2, size=1., amt=1., edge=.18):
    """night stratus: torn flat shapes with a hard lit edge on the side of the light at src=(x, y)"""
    h, w = img.shape[:2]; x, y = grid(h, w); env = smooth(y0 - (y1 - y0) * .2, y0 + (y1 - y0) * .25, y) * (1 - smooth(y1 - (y1 - y0) * .3, y1 + (y1 - y0) * .08, y))
    n = fbm(h, w, (170 * size, 74 * size, 31 * size, 13 * size, 5.5 * size), (1, .62, .42, .28, .16), seed, aspect=aspect); n = warp_by(n, 5 * size, 90 * size, seed + 3); th = .95 - 1.7 * cover
    f = (n + (env - 1) * 2.2).astype(f32); m = smooth(th, th + edge, f)
    dx, dy = src[0] - x, src[1] - y; d = np.sqrt(dx * dx + dy * dy) + 1; gx = cv2.Sobel(cv2.GaussianBlur(f, (0, 0), 2.5), cv2.CV_32F, 1, 0) / 8; gy = cv2.Sobel(cv2.GaussianBlur(f, (0, 0), 2.5), cv2.CV_32F, 0, 1) / 8
    facing = -(gx * dx + gy * dy) / d; near = np.exp(-d / (w * .30)); body = smooth(th, th + 1.5, f)
    c = mix(C(lit), C(dark), np.clip(body * 1.25 - near * .35, 0, 1)[..., None]); fl = smooth(.004, .030, facing) * (1 - smooth(.35, .9, f - th)); c = c + (C(rim) - c) * (fl * (.22 + .9 * near)).clip(0, 1)[..., None]
    return img + (c - img) * (m * amt)[..., None]


def moon_disc(img, cx, cy, r, col='#fbf7ff', halo='#8a2cff', halo_amt=.9, rim='#c784ff'):
    h, w = img.shape[:2]; x, y = grid(h, w); d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2); q = np.clip(d - r, 0, None)
    img = img + C(halo) * (halo_amt * (.60 * np.exp(-q / (r * .42)) + .32 * np.exp(-q / (r * 1.6)) + .16 * np.exp(-q / (r * 5.))))[..., None]
    img = img + (C(rim) - img) * (np.clip(1 - np.abs(d - r) / (r * .07 + 1.5), 0, 1) * .9)[..., None]
    return img + (C(col) * 1.06 - img) * np.clip(r - d + .5, 0, 1)[..., None]


def moon_halo(img, cx, cy, r, halo='#8a2cff', amt=.9):
    h, w = img.shape[:2]; x, y = grid(h, w); q = np.clip(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r, 0, None)
    return img + C(halo) * (amt * (.60 * np.exp(-q / (r * .42)) + .30 * np.exp(-q / (r * 1.6)) + .14 * np.exp(-q / (r * 5.))))[..., None]


def disc(cam, img, d, ang=.0545, col='#fbf7ff', rim='#c784ff'):
    """the moon as a true circle on this camera's picture, in world direction d (angular radius ang)"""
    p, z = cam.project(cam.eye + np.asarray(d, float) * 1e5)
    if z <= 0: return img
    h, w = img.shape[:2]; x, y = grid(h, w); r = ang * cam.f; q = np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2)
    img = img + (C(rim) - img) * (np.clip(1 - np.abs(q - r) / (r * .07 + 1.5), 0, 1) * .9)[..., None]
    return img + (C(col) * 1.06 - img) * np.clip(r - q + .5, 0, 1)[..., None]


def night(h, w, hz, moon=None, seed=0, stops=None, cov=.46, cl=('#0c0a1c', '#2b2256', '#b48cff'), disc=True):
    """violet moon night.  moon = (x, y, r) px or None"""
    st = stops or [(-1, '#140f2e'), (0, '#43307f'), (.05, '#35266b'), (.14, '#241b4c'), (.34, '#141030'), (.7, '#0b091d'), (1.2, '#070512')]
    img = gradient(h, w, hz, st, .10, None if moon is None else moon[0], seed)
    src = (w * .7, -h * .2) if moon is None else moon[:2]
    x, y = grid(h, w); d = np.sqrt((x - src[0]) ** 2 + (y - src[1]) ** 2); img = img + C('#3a1f86') * (.8 * np.exp(-d / (min(w, 2400) * .15)))[..., None]
    img = torn(img, -h * .1, hz * .72, cl[0], cl[1], cl[2], src, seed + 11, cov, 5.0, 1.15 * w / 1920)
    img = torn(img, hz * .50, hz * 1.0, '#1a1440', '#443084', '#a583ee', src, seed + 23, cov - .05, 9., .55 * w / 1920, .9)
    r = np.random.default_rng(seed + 4); n = int(90 * w / 1920)
    for a, b, v in zip(r.uniform(0, w, n), r.uniform(0, hz * .75, n), r.random(n) ** 3):
        i, j = int(b), int(a)
        if 1 < i < h - 2 and 1 < j < w - 2 and img[i, j].mean() < .16: img[i, j] += (.15 + .55 * v) * C('#dccfff'); img[i, j + 1] += .12 * v
    img = brush(np.maximum(img, 0), seed + 5, .45)
    if moon is not None and disc: img = moon_disc(img, *moon)
    elif moon is not None: img = moon_halo(img, *moon)
    return img


SKIES = dict(
    storm=dict(st=[(-1, '#0a1112'), (0, '#7d8a66'), (.05, '#56685a'), (.14, '#2f403c'), (.34, '#16211f'), (.7, '#0b1214'), (1.2, '#05090a')], cl=('#080e10', '#1f2d2c', '#7a7a52'), src=(.5, -.1)),
    rain=dict(st=[(-1, '#6d706f'), (0, '#a9aba8'), (.10, '#9c9f9d'), (.35, '#808382'), (.7, '#636666'), (1.2, '#4b4d4e')], cl=('#55585a', '#7d8080', '#b4b6b3'), src=(.75, .55)),
    first=dict(st=[(-1, '#2c2a4e'), (0, '#f0b39a'), (.05, '#d79a9c'), (.14, '#8f7a9e'), (.32, '#4a4a82'), (.65, '#23264e'), (1.2, '#10122a')], cl=('#25264e', '#6a5c88', '#ffb894'), src=(.95, .5)),
    gold=dict(st=[(-1, '#1a1014'), (0, '#7a4a2a'), (.08, '#4e3030'), (.3, '#251c26'), (.7, '#120e16'), (1.2, '#08060c')], cl=('#140f16', '#3c2a2c', '#ffc050'), src=(.5, .6)))


def sky(name, h, w, hz, seed=0):
    d = SKIES[name]; img = gradient(h, w, hz, d['st'], .06, w * d['src'][0], seed); src = (w * d['src'][0], h * d['src'][1]); c = d['cl']
    img = torn(img, -h * .2, hz * .80, c[0], c[1], c[2], src, seed + 11, .60, 3.2, 1.5 * w / 1920, .9, .5); img = torn(img, hz * .35, hz * 1.02, mix(C(c[0]), C(c[1]), .5), mix(C(c[1]), C(c[2]), .3), c[2], src, seed + 23, .48, 7., .7 * w / 1920, .8, .35)
    return np.maximum(brush(img, seed + 5, .6), 0)


# ------------------------------------------------------------------------------------------------ panorama: the sky stays put when the camera turns or banks
AZ, E0, E1 = math.radians(84), math.radians(-14), math.radians(64)
PW, PH = 3840, 1800
_pano = {}


def pano(name, moon=None, seed=3, ver=4):
    """equirect picture of a sky, azimuth -84..84 deg about +Y, elevation 64..-14 deg.  moon = world direction drawn into it"""
    import os
    k = (name, None if moon is None else tuple(np.round(moon, 3)), seed)
    if k in _pano: return _pano[k]
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cache')); p = os.path.join(root, 'land_sky_%s_%d_%s_v%d.npy' % (name, seed, 'm' if moon is not None else 'n', ver))
    if os.path.exists(p): _pano[k] = np.load(p).astype(f32); return _pano[k]
    hz = PH * E1 / (E1 - E0); mo = None
    if moon is not None: a, e = math.atan2(moon[0], moon[1]), math.atan2(moon[2], math.hypot(moon[0], moon[1])); mo = (float((a + AZ) / (2 * AZ) * PW), float((E1 - e) / (E1 - E0) * PH), .0545 * PW / (2 * AZ))
    img = night(PH, PW, hz, mo, seed, disc=False) if name == 'night' else sky(name, PH, PW, hz, seed)
    os.makedirs(root, exist_ok=True); np.save(p, img.astype(np.float16)); _pano[k] = img.astype(f32); return _pano[k]


def view(cam, img):
    """the panorama seen through a camera"""
    d = cam.rays(); az = np.arctan2(d[..., 0], d[..., 1]); el = np.arctan2(d[..., 2], np.hypot(d[..., 0], d[..., 1])); h, w = img.shape[:2]
    return cv2.remap(img, ((az + AZ) / (2 * AZ) * w - .5).astype(f32), ((E1 - el) / (E1 - E0) * h - .5).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def mirror(cam, img, z=0.):
    """the panorama as a flat mirror shows it to this camera"""
    d = cam.rays(); az = np.arctan2(d[..., 0], d[..., 1]); el = np.abs(np.arctan2(d[..., 2], np.hypot(d[..., 0], d[..., 1]))); h, w = img.shape[:2]
    return cv2.remap(img, ((az + AZ) / (2 * AZ) * w - .5).astype(f32), ((E1 - el) / (E1 - E0) * h - .5).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
