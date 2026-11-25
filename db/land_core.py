"""Renderer of the painted places: geometry + water + fog -> picture, with depth and masks.  See land.py for the sets."""
import numpy as np, cv2, math, os
import numba
from . import W, H
from .paint import rgb, f32, fbm, lum, smooth, mix, grid, Canvas, arcs, jit_color, ramp
from .cam import Camera3D
from .comp import blur, over, radial
from . import land_nb as nb
from .land_geo import Geo, Mesh, cat

try: numba.set_num_threads(min(4, numba.get_num_threads()))
except Exception: pass
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FAR = 1e6


def C(c): return rgb(c) if isinstance(c, str) else np.asarray(c, f32)


def hx(*c): return np.stack([C(v) for v in c]).astype(f32)


# ------------------------------------------------------------------------------------------------ looks
MATS = 6            # 0 cliff rock  1 ground rock  2 boulder / loose  3 earth, sand  4 turf ground  5 dark accent rock
PAR0 = dict(key=.62, amb=.10, sky=.12, side=.12, chisel=1., stretch=2.1, crack=1., strata=1., brush=1., edge=.5, sharp=.24, hue=.10, cell=1., fcell=.5, wet=0., turf=0., canopy=0., crown=6.)
KEYS = list(PAR0)


class Look:
    """one lighting of the place.  ramp = 6 colours dark -> light (per material, or one for all)"""

    def __init__(s, ramp, light, key=1., bounce=(-.6, -.3, .2), fog=None, water=None, turf=None, par=None, mats=None, wetcol=None, glint=None):
        s.light = np.asarray(light, float) / np.linalg.norm(light); s.bounce = np.asarray(bounce, float) / np.linalg.norm(bounce)
        s.glint = s.light if glint is None else np.asarray(glint, float) / np.linalg.norm(glint)
        s.RAMP = np.stack([hx(*(ramp[m] if isinstance(ramp, dict) and m in ramp else ramp[0] if isinstance(ramp, dict) else ramp)) for m in range(MATS)]).astype(np.float64)
        s.par = np.zeros((MATS, len(KEYS)))
        for m in range(MATS):
            d = dict(PAR0); d.update(par or {}); d.update((mats or {}).get(m, {})); s.par[m] = [d[k] for k in KEYS]
        s.fog = np.zeros(8); s.fogc = np.zeros((4, 3))
        if fog:
            s.fog[:7] = [fog.get('dens', .001), fog.get('start', 0.), fog.get('hf', 0.), fog.get('floor', .35), fog.get('base', 0.), fog.get('pw', 3.), fog.get('max', .93)]
            s.fogc[0] = C(fog['col']); s.fogc[1] = C(fog.get('col2', fog['col'])); d = s.light.copy(); d[2] = 0; s.fogc[2] = d / (np.linalg.norm(d) + 1e-9)
        s.fogc[3] = C(wetcol) if wetcol is not None else s.fogc[0]
        s.turf = hx(*(turf or ('#20331f', '#36532a', '#5b7d35', '#8fa84a', '#cfd67a'))).astype(np.float64)
        w = dict(z=0., on=1, scale=7., slope=.1, refl=.8, glint=1., foam=.8, stroke=1., far=0., distort=26., sigma=.12, clear=0.); w.update(water or {}); s.water = w
        s.wat = np.array([w['z'], w['on'], w['scale'], w['slope'], w['refl'], w['glint'], w['foam'], w['stroke'], 0., w['far'], w['distort'], w['sigma'], 0., w['clear']], float)
        s.WC = np.stack([C(c) for c in w.get('cols', ('#0a0b1c', '#2a2456', '#07071a', '#3c3578', '#e9dcff', '#a596dc', '#f4edff'))]).astype(np.float64)


# ------------------------------------------------------------------------------------------------ render
def _project(T, cam, ss, near=.12):
    c = (T.astype(np.float64) - cam.eye) @ np.asarray(cam.R).T; zc = c[..., 2]; ins = (zc > near).sum(1); ok = ins == 3; cr = (ins > 0) & (ins < 3)
    idx = np.nonzero(ok)[0]; pts = [c[ok]]; ids = [idx]; bar = []
    if cr.any():
        cp, ci = [], []
        for i in np.nonzero(cr)[0]:
            V = c[i]; poly = []
            for a in range(3):
                b = (a + 1) % 3
                if V[a, 2] > near: poly.append(V[a])
                if (V[a, 2] > near) != (V[b, 2] > near): t = (near - V[a, 2]) / (V[b, 2] - V[a, 2]); poly.append(V[a] + (V[b] - V[a]) * t)
            for a in range(1, len(poly) - 1): cp.append((poly[0], poly[a], poly[a + 1])); ci.append(i)
        if cp: pts.append(np.array(cp)); ids.append(np.array(ci))
    c = np.concatenate(pts); idx = np.concatenate(ids); clip = np.r_[np.zeros(len(pts[0]), bool), np.ones(len(c) - len(pts[0]), bool)]
    iz = 1 / np.maximum(c[..., 2], near); sx = (c[..., 0] * iz * cam.f + cam.cx) * ss; sy = (c[..., 1] * iz * cam.f + cam.cy) * ss
    vis = (sx.max(1) > 0) & (sx.min(1) < cam.w * ss) & (sy.max(1) > 0) & (sy.min(1) < cam.h * ss)
    return sx[vis], sy[vis], iz[vis], idx[vis], clip[vis]


def shadow_map(G, light, res=3072, box=None, only=None):
    """hard shadow depth of a static Geo seen from the light.  box = (x0, x1, y0, y1, z0, z1) region that matters"""
    sd = np.asarray(light, float); up = np.array([0, 0, 1.]) if abs(sd[2]) < .95 else np.array([0, 1., 0]); u = np.cross(up, sd); u /= np.linalg.norm(u); v = np.cross(sd, u)
    Q = G.T.reshape(-1, 3).astype(float)
    bb = Q if box is None else np.array([(x, y, z) for x in box[:2] for y in box[2:4] for z in box[4:6]], float)
    uu, vv = bb @ u, bb @ v; span = max(uu.max() - uu.min(), vv.max() - vv.min()) * 1.02; cu, cv_ = (uu.max() + uu.min()) / 2, (vv.max() + vv.min()) / 2; k = res / span
    M = np.zeros((3, 4)); M[0, :3] = u * k; M[0, 3] = res / 2 - cu * k; M[1, :3] = v * k; M[1, 3] = res / 2 - cv_ * k; M[2, :3] = sd; M[2, 3] = -(Q @ sd).min() + 10
    T = G.T.astype(float); X = T @ M[0, :3] + M[0, 3]; Y = T @ M[1, :3] + M[1, 3]; Z = T @ M[2, :3] + M[2, 3]; vis = (X.max(1) > 0) & (X.min(1) < res) & (Y.max(1) > 0) & (Y.min(1) < res)
    ID = np.full((res, res), -1, np.int32); ZB = np.zeros((res, res), f32); b = np.zeros((res, res), f32)
    nb.raster(X[vis], Y[vis], Z[vis], res, res, ID, ZB, b, b.copy(), 16)
    return ZB, M, span / res * 1.8


def foam_map(G, box, res=1024, z=0.):
    """metres to the nearest rock at water level, over box=(x0,x1,y0,y1)"""
    x0, x1, y0, y1 = box; k = res / max(x1 - x0, y1 - y0); w, h = int((x1 - x0) * k), int((y1 - y0) * k); m = np.zeros((h, w), np.uint8)
    T = G.T; s = (T[..., 2].min(1) < z) & (T[..., 2].max(1) > z)
    lo = T[T[..., 2].max(1) > z]
    for t in T[s]:
        p = [];
        for a in range(3):
            b = (a + 1) % 3; za, zb = t[a, 2] - z, t[b, 2] - z
            if za * zb < 0: q = za / (za - zb); p.append(t[a] + (t[b] - t[a]) * q)
        if len(p) == 2: cv2.line(m, tuple(int((v - o) * k * 16) for v, o in zip(p[0][:2], (x0, y0))), tuple(int((v - o) * k * 16) for v, o in zip(p[1][:2], (x0, y0))), 255, 2, cv2.LINE_AA, 4)
    d = cv2.distanceTransform((m < 60).astype(np.uint8), cv2.DIST_L2, 5) / k
    return d.astype(f32), np.array([x0, y0, k], float)


def horizon(cam):
    """screen row of the horizon (upright camera)"""
    d = np.array(cam.R[2], float); d[2] = 0; d /= np.linalg.norm(d) + 1e-9; return cam.cy + cam.f * (cam.R[1] @ d) / max(1e-6, cam.R[2] @ d)


_NOFM = (np.full((4, 4), 999., f32), np.array([0, 0, 0.], float))


class Out: pass


def render(cam, G, look, t=0., sky=None, shadow=None, foam=None, fires=(), rings=(), ss=2, reflect=True, hz=None, dyn=None, sky_reflect=None, water=True, key=1.):
    """cam Camera3D, G static Geo, dyn extra Geo.  sky = (h,w,3) picture behind everything (also mirrored in the water) or None.
    rings rows (x, y, R, width, amp) on the water.  fires rows (x, y, z, radius, '#col', amt).
    -> Out: img premult-over-sky (h,w,3), a (geometry+water coverage), z metres, water mask, rock mask, val tone 0..1"""
    g = G if dyn is None else cat(G, dyn); w, h = cam.w, cam.h; o = Out()
    eye = np.array(cam.eye, float); R = np.array(cam.R, float)
    NOS = (np.zeros((1, 1), f32), np.zeros((3, 4)), 0.)
    (SM, SMm, bias), (SM2, SMm2, bias2) = (NOS, NOS) if shadow is None else (shadow, NOS) if not isinstance(shadow, list) else shadow          # a list = [fine map, coarse map]
    fr = np.array([[x, y, z, rad, *(C(c) * a)] for (x, y, z, rad, c, a) in fires], float).reshape(-1, 7); rg = np.array(rings, float).reshape(-1, 5)
    fm, fmm = foam if foam is not None else _NOFM
    wat = look.wat.copy(); wat[8] = t
    if foam is None: wat[6] = 0.
    if not water: wat[1] = 0.
    par = look.par.copy(); par[:, 0] *= key
    def go(gg, cm, ss_, REF, wat_, ff=1.):
        sx, sy, iz, idx, clip = _project(gg.T, cm, ss_); ww, hh = cm.w * ss_, cm.h * ss_
        ID = np.full((hh, ww), -1, np.int32); ZB = np.zeros((hh, ww), f32); B1 = np.zeros((hh, ww), f32); B2 = np.zeros((hh, ww), f32)
        if len(idx): nb.raster(sx, sy, iz, ww, hh, ID, ZB, B1, B2, 16)
        out = np.zeros((cm.h, cm.w, 3), f32); cov = np.zeros((cm.h, cm.w), f32); zo = np.zeros((cm.h, cm.w), f32); ido = np.zeros((cm.h, cm.w), np.int32); vo = np.zeros((cm.h, cm.w), f32)
        E = gg.EDG[idx].copy(); E[clip] = 0
        nb.shade(ID, ZB, B1, B2, cm.w, cm.h, ss_, np.array(cm.eye, float), np.array(cm.R, float), cm.f, cm.cx, cm.cy, sx.astype(f32), sy.astype(f32), gg.FN[idx], gg.VN[idx], gg.VA[idx], gg.MAT[idx], gg.SD[idx], E, gg.COL[idx],
                 look.light, SM, SMm, SM.shape[0] if SM.shape[0] > 1 else 0, bias, SM2, SMm2, SM2.shape[0] if SM2.shape[0] > 1 else 0, bias2, par, look.RAMP, look.bounce, look.turf, fr, look.fog, look.fogc, wat_, look.WC, REF, fm, fmm, rg, look.glint, out, cov, zo, ido, vo)
        return out, cov, zo, ido, vo
    REF = np.zeros((1, 1, 3), f32)
    sk = sky if sky is not None else None
    if wat[1] > 0 and look.water['refl'] > 0:
        hy = float(horizon(cam)) if hz is None else hz
        if sky_reflect is not None: RS = sky_reflect
        elif sk is not None:
            yy = np.clip(2 * hy - np.arange(h), 0, h - 1).astype(int); RS = cv2.blur(sk[yy], (9, 3))
        else: RS = np.broadcast_to(look.fogc[0].astype(f32), (h, w, 3)).copy()
        REF = RS.astype(f32)
        if reflect:
            cm = Camera3D(cam.eye, cam.eye + R[2], cam.f, w, h, cam.cx, cam.cy); cm.R = R; m = Geo(**g.__dict__); F = np.array([1, 1, -1], f32); up = g.T[..., 2].max(1) > wat[0] + .02
            m = g.sel(up); T = m.T.copy(); T[..., 2] = np.maximum(T[..., 2], wat[0] - .01); T[..., 2] = 2 * wat[0] - T[..., 2]; m.T = T; m.FN = m.FN * F; m.VN = m.VN * F
            w0 = wat.copy(); w0[1] = 0.; LK = look.light; look.light = LK * [1, 1, -1]; bo = look.bounce; look.bounce = bo * [1, 1, -1]
            for q_ in (0, 1):
                Q_ = (SMm if q_ == 0 else SMm2).copy(); Q_[:, 2] *= -1; Q_[:, 3] = Q_[:, 3] - Q_[:, 2] * 2 * wat[0]
                if q_ == 0: SMm = Q_
                else: SMm2 = Q_
            ri, rc, *_ = go(m, cm, 1, REF, w0); look.light = LK; look.bounce = bo
            for Q_ in (SMm, SMm2): Q_[:, 3] = Q_[:, 3] + Q_[:, 2] * 2 * wat[0]; Q_[:, 2] *= -1
            REF = (ri + REF * (1 - rc[..., None])).astype(f32)
            REF = cv2.GaussianBlur(REF, (0, 0), sigmaX=.8, sigmaY=2.2)
    out, cov, zo, ido, vo = go(g, cam, ss, REF, wat)
    o.rgb = out; o.a = cov; o.z = zo; o.id = ido; o.val = vo; o.cam = cam; o.water = ((ido == -2) & (cov > .3)).astype(f32); o.rock = (ido >= 0).astype(f32)
    o.img = out + (1 - cov[..., None]) * sky if sky is not None else out
    return o


# ------------------------------------------------------------------------------------------------ painterly finish for plates
def paint_over(img, seed=1, sizes=(22, 10, 5), op=(.5, .5, .45), val=(.03, .025, .02), hue=.008, dry=.3, thr=(.016, .03, .05), ang=None, aspect=4.2, amt=1., density=1., bristle=.05, mask=None):
    """real gouache strokes over a guide: big only where the picture is quiet, small nearer detail (plates, not per frame)"""
    img = np.asarray(img, f32); g = np.clip(img, 0, 1); ext = img - g; h, w = g.shape[:2]
    L = lum(g); gx = cv2.Sobel(L, cv2.CV_32F, 1, 0) / 8; gy = cv2.Sobel(L, cv2.CV_32F, 0, 1) / 8; mag = np.abs(gx) + np.abs(gy)
    if ang is None:
        s = lambda a: cv2.GaussianBlur(a, (0, 0), 7); jxx, jyy, jxy = s(gx * gx), s(gy * gy), s(gx * gy); ang = .5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    cv = Canvas(h, w, img=g); r = np.random.default_rng(seed + 100)
    for li, hw in enumerate(sizes):
        en = cv2.boxFilter(mag, -1, (int(hw * 2) | 1, int(hw * 2) | 1)); step = hw * 1.55 / math.sqrt(density); yy, xx = np.mgrid[step * .5:h:step, step * .5:w:step].astype(f32)
        x = (xx + r.uniform(-.6, .6, xx.shape) * step).ravel().clip(0, w - 1); y = (yy + r.uniform(-.6, .6, xx.shape) * step).ravel().clip(0, h - 1); ix, iy = x.astype(int), y.astype(int)
        k = en[iy, ix] < thr[li] * r.uniform(.5, 1.5, x.size)
        if mask is not None: k &= mask[iy, ix] > r.uniform(.05, .95, x.size)
        ix, iy, x, y = ix[k], iy[k], x[k], y[k]
        if x.size == 0: continue
        p = r.permutation(x.size); ix, iy, x, y = ix[p], iy[p], x[p], y[p]; ref = cv2.GaussianBlur(g, (0, 0), max(.6, hw * .15)); base = g[iy, ix] * .7 + ref[iy, ix] * .3
        col = jit_color(base, r, hue, .03, val[li]); lm = lum(base); col = base + (col - base) * np.clip(.35 + lm * 1.6, .35, 1)[:, None]
        P = arcs(x, y, ang[iy, ix] + r.normal(0, .14, x.size), hw * aspect * r.uniform(.55, 1.3, x.size), r.normal(0, .35, x.size))
        cv.strokes(P, hw * r.uniform(.55, 1.15, x.size), col, op=op[li] * r.uniform(.6, 1.1, x.size) * amt, dry=dry * r.uniform(0, 2.2, x.size), seed=seed + li * 31, bristle=bristle)
    return cv.img + ext


def occl(z, amt=.3, scales=(6, 22, 70), k=9.):
    """painted contact shade and depth separation from a depth map -> multiplier"""
    m = (z < FAR * .5).astype(f32); lz = np.where(m > 0, np.log(np.maximum(z, .05)), 0).astype(f32); ao = np.zeros_like(lz); hh, ww = z.shape
    for s in scales:
        q = max(1, int(s // 4)); a = cv2.resize(lz * m, (ww // q, hh // q), interpolation=cv2.INTER_AREA); b = cv2.resize(m, (ww // q, hh // q), interpolation=cv2.INTER_AREA)
        a = cv2.GaussianBlur(a, (0, 0), s / q) / (cv2.GaussianBlur(b, (0, 0), s / q) + 1e-3); a = cv2.resize(a, (ww, hh), interpolation=cv2.INTER_LINEAR); ao += np.clip((lz - a) * k / (1 + s * .05), 0, 1) * m
    return 1 - amt * np.clip(ao / len(scales) * 1.4, 0, 1)


def parallax(img, z, dx=0., dy=0., zoom=1., ref=40., cx=None, cy=None):
    """small 2.5-D camera move over a plate: nearer things move more.  dx, dy in px for something at `ref` metres"""
    h, w = z.shape; cx = w / 2 if cx is None else cx; cy = h / 2 if cy is None else cy; x, y = grid(h, w); zz = cv2.dilate(-np.minimum(z, 1e5), np.ones((3, 3))) * -1; k = ref / np.maximum(zz, ref * .12)
    s = 1 + (zoom - 1) * (.25 + .75 * np.minimum(k, 2.5)); return cv2.remap(img, (cx + (x - cx) / s + dx * k).astype(f32), (cy + (y - cy) / s + dy * k).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
