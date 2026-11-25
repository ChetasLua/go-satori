"""Painterly close-ups: the cel's hard shadow drawn as a mask, then toned by distance from its edge (warm terminator, cool depth), airbrush, hatching, light."""
import numpy as np, math, cv2, cairo
from db import W, H as HH
from db.ink import Cel, spl, resample
from db.comp import blur
from db.paint import rgb, f32, smooth, ramp, fbm, lum, grid
from db.kit import hatch
from . import goku_hair as GH, goku_head as GF
from .goku_hair import mixc
A = np.array


def mask(polys, w=W, h=HH, within=None, ss=2):
    c = Cel(w, h, ss=ss)
    if within is not None: c.clip(within)
    for p in polys:
        if isinstance(p, tuple): c.fill(p[0], '#ffffff', p[1])
        else: c.fill(p, '#ffffff')
    if within is not None: c.unclip()
    return c.rgba()[..., 3]


def tone(sil, M, lit, shd, sig, M2=None, deep=None, edge=None, edge_amt=.6, noise=.0, seed=0):
    """sil = silhouette alpha, M = hard shadow mask 0..1; lit / shd = ramps over blurred mask; returns RGB (not premultiplied)"""
    B = blur(M, sig) / np.maximum(blur(sil, sig), .02); B = np.clip(B, 0, 1)
    if noise: B = np.clip(B + fbm(M.shape[0], M.shape[1], (60, 22, 9), (1, .6, .4), seed) * noise, 0, 1)
    a, b = ramp(B, lit), ramp(B, shd); o = a * (1 - M[..., None]) + b * M[..., None]
    if edge is not None:                                # warm light bleeding along the terminator
        e = blur(M, sig * .10); e = np.clip(4 * e * (1 - e), 0, 1) ** 1.5 * (.35 + .65 * M); o = o + (rgb(edge) - o) * (e * edge_amt)[..., None]
    if M2 is not None: o = o + (rgb(deep) - o) * np.clip(M2, 0, 1)[..., None] * .85
    return o


def prem(rgb_, al): return np.concatenate([rgb_ * al[..., None], al[..., None]], -1).astype(f32)


def over_(a, b):
    """premultiplied b over premultiplied a"""
    return b + a * (1 - b[..., 3:4])


def cheek_hatch(F, pts, col, w=1.2, seed=0, alpha=.8):
    """tufts of pen strokes: pts = (x0,y0,x1,y1,n,len,dx,dy) leave the segment in direction (dx,dy), longest in the middle, slightly fanned (front coordinates)"""
    r = np.random.default_rng(seed); H = F.H
    for x0, y0, x1, y1, n, l, dx, dy in pts:
        us = np.sort(np.clip(np.linspace(0, 1, n) + r.normal(0, .5 / n, n), 0, 1)); ph = r.uniform(0, 6)
        for j, u in enumerate(us):
            env = (.25 + .75 * math.sin(math.pi * min(1, max(0, u))) ** .8) * (.75 + .25 * math.sin(u * 9 + ph)); q = l * env * r.uniform(.75, 1.15); a = (u - .5) * .35; c, s_ = math.cos(a), math.sin(a); ddx, ddy = dx * c - dy * s_, dx * s_ + dy * c
            x, y = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
            if not H.vis(x * 1.03, y) or not H.vis((x + ddx * q) * 1.02, y + ddy * q): continue
            bw = r.normal(0, .004); F.c.line(spl(H.Pc([(x - ddx * q * .12, y - ddy * q * .12), (x + ddx * q * .45 + bw, y + ddy * q * .45), (x + ddx * q, y + ddy * q)]), n=8), w * r.uniform(.7, 1.3), col, taper=(.02, .85), press=.2, seed=int(r.integers(1 << 30)), alpha=alpha * r.uniform(.6, 1.), tremor=.06, tmin=.03)


def dabs(cel, P, n, w, col, seed=0, al=.2, ang=0., L=30):
    pass
