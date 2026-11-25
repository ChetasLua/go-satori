"""Sea in true perspective from any height (2 m .. 30 km): four gouache values, foam-free flat swell shapes, sun glitter from the real mirror geometry,
cloud shadows, islands with lagoon, reef foam, beach, jungle and rock lit by the sun.
    im, d = sea(cam, f, SEA['day'], sun=(.3,.8,.5), sky=sky_rgb_behind)         # d = distance (0 = sky); pass sky=(1080,1920,3) to fill above the horizon
    isl = [(x, y, radius, seed), ...];  sea(..., islands=isl, shadows=(size_m, cover 0..1, dx, dy))"""
import numpy as np, math
from numba import njit, prange
from .paint import f32, rgb, _vn
from .sky_vol import _ss, _h3

# deep, mid, light (swell facets), far (near the horizon), glint, shallow, lagoon, foam, sand, grass, jungle, jungle shade, rock, haze
SEA = {'day': ['#0f5fa8', '#1c7fc2', '#3fa3d8', '#8fd0ea', '#fff6dc', '#25a9c8', '#79dccf', '#f6fbf4', '#f7e3b4', '#8cbf5a', '#4f9150', '#2f6a55', '#a08f80', '#d6eef4'],
       'gold': ['#17609a', '#2580b4', '#58a9c6', '#f3c98e', '#fff0c0', '#2aa6b8', '#86d8c4', '#fff6e0', '#fbdfa6', '#a9b65a', '#6a8e50', '#3f6a5c', '#b08a78', '#fbe2b8'],
       'storm': ['#141c2a', '#1e2a3a', '#33445a', '#4a5a6c', '#b8c4d6', '#1e3844', '#2c4c54', '#aebccb', '#4c4a4c', '#26352f', '#1c2a28', '#131d1f', '#2c2c34', '#4d5a6a'],
       'night': ['#05060f', '#0a0c1c', '#161634', '#2a2454', '#f3ecff', '#0c1428', '#141f38', '#8f86c0', '#26233c', '#101424', '#0b0e1a', '#070912', '#191829', '#1c1838'],
       'dawn': ['#3a4f8c', '#5a6ea8', '#a58fb4', '#f3bfae', '#fff0d6', '#5f88b0', '#9ab8c4', '#ffeee0', '#e8c4ac', '#7a8a70', '#55706c', '#3d5566', '#8a7a88', '#ffdcc8']}


@njit(parallel=True, cache=True, fastmath=True)
def _sea(out, D, eye, Rm, f, cx, cy, P, sun, t, rough, haze_d, isl, shs, shc, shx, shy, glit, l0, curv):
    h, w = D.shape
    for y_ in prange(h):
        y = np.int64(y_)
        for x in range(w):
            dx = (x + .5 - cx) / f; dy = (y + .5 - cy) / f
            rx = Rm[0, 0] * dx + Rm[1, 0] * dy + Rm[2, 0]; ry = Rm[0, 1] * dx + Rm[1, 1] * dy + Rm[2, 1]; rz = Rm[0, 2] * dx + Rm[1, 2] * dy + Rm[2, 2]
            if rz >= -1e-5: continue
            rl = math.sqrt(rx * rx + ry * ry + rz * rz); tt = -eye[2] / rz; X = eye[0] + rx * tt; Y = eye[1] + ry * tt; dist = tt * rl
            vx = -rx / rl; vy = -ry / rl; vz = -rz / rl
            lat = tt / f; foot = lat / (vz + .02)
            # swell: ridged, wind from the left, drifting
            n = 0.0; gx = 0.0; am = 1.0; wl = l0; sm = 0.0
            for o in range(6):
                at = (wl / foot - 1.4) / 5.0
                if at <= 0: break
                if at > 1: at = 1.0
                a = (X * .92 + Y * .38) / wl + o * 11.3 + t * (.028 + .012 * o) * math.sqrt(220. / wl) * .5 + 900.; b = (Y * .92 - X * .38) / (wl * 1.9) + o * 5.7 + 900.
                wa = _vn(a * .4 + 7, b * .4 + 2, 60 + o) * 1.7
                q = _vn(a + wa, b - wa * .5, 31 + o); q = 1 - abs(2 * q - 1); q2 = _vn(a + wa + .12, b - wa * .5, 31 + o); q2 = 1 - abs(2 * q2 - 1)
                n += am * at * (q - .5); gx += am * at * (q - q2); sm += am; am *= .62; wl *= .42
            n = n / 1.6; F = (1 - vz); F = F * F * F * (1 + F) * .5
            k = F + n * .42 * rough * (.50 + .50 * vz)
            c0 = P[0, 0]; c1 = P[0, 1]; c2 = P[0, 2]
            e = _ss(.00, .07, k); c0 += (P[1, 0] - c0) * e; c1 += (P[1, 1] - c1) * e; c2 += (P[1, 2] - c2) * e
            e = _ss(.15, .20, k + gx * .10 * rough) * .85; c0 += (P[2, 0] - c0) * e; c1 += (P[2, 1] - c1) * e; c2 += (P[2, 2] - c2) * e
            e = _ss(.30, .85, F + n * .10); c0 += (P[3, 0] - c0) * e; c1 += (P[3, 1] - c1) * e; c2 += (P[3, 2] - c2) * e
            land = 0.0
            for i in range(isl.shape[0]):
                r = isl[i, 2]; ex = (X - isl[i, 0]) / r; ey = (Y - isl[i, 1]) / r; d2 = ex * ex + ey * ey
                if d2 > 9: continue
                sd = int(isl[i, 3]); nn = (_vn(ex * 1.7 + 40, ey * 1.7 + 40, sd) - .5) * .9 + (_vn(ex * 4.3 + 40, ey * 4.3 + 40, sd + 1) - .5) * .42 + (_vn(ex * 11. + 40, ey * 11. + 40, sd + 2) - .5) * .16
                d = math.sqrt(d2) + nn * .62
                e = _ss(2.0, 1.45, d) * .85; c0 += (P[5, 0] - c0) * e; c1 += (P[5, 1] - c1) * e; c2 += (P[5, 2] - c2) * e
                e = _ss(1.42, 1.34, d + (n * .06)) * .92; c0 += (P[6, 0] - c0) * e; c1 += (P[6, 1] - c1) * e; c2 += (P[6, 2] - c2) * e
                wv = .02 * math.sin(t * .35 + nn * 20); e = _ss(1.44 + wv, 1.42 + wv, d) * _ss(1.36 + wv, 1.39 + wv, d) * .9; c0 += (P[7, 0] - c0) * e; c1 += (P[7, 1] - c1) * e; c2 += (P[7, 2] - c2) * e
                e = _ss(1.10 + wv, 1.08 + wv, d); c0 += (P[7, 0] - c0) * e; c1 += (P[7, 1] - c1) * e; c2 += (P[7, 2] - c2) * e
                e = _ss(1.065, 1.055, d); c0 += (P[8, 0] - c0) * e; c1 += (P[8, 1] - c1) * e; c2 += (P[8, 2] - c2) * e
                if d < 1.0:
                    hh = _vn(ex * 5 + 80, ey * 5 + 80, sd + 5) * .6 + _vn(ex * 13 + 80, ey * 13 + 80, sd + 6) * .4; h2 = _vn(ex * 5 + 80 + sun[0] * .25, ey * 5 + 80 + sun[1] * .25, sd + 5) * .6 + _vn(ex * 13 + 80 + sun[0] * .4, ey * 13 + 80 + sun[1] * .4, sd + 6) * .4
                    e = _ss(.97, .94, d); c0 += (P[9, 0] - c0) * e; c1 += (P[9, 1] - c1) * e; c2 += (P[9, 2] - c2) * e
                    e = _ss(.90, .84, d + (hh - .5) * .5); c0 += (P[10, 0] - c0) * e; c1 += (P[10, 1] - c1) * e; c2 += (P[10, 2] - c2) * e
                    e = _ss(.02, .05, h2 - hh) * _ss(.95, .85, d); c0 += (P[11, 0] - c0) * e; c1 += (P[11, 1] - c1) * e; c2 += (P[11, 2] - c2) * e
                    e = _ss(.40, .34, d + (hh - .5) * .3) * _ss(.05, -.02, h2 - hh); c0 += (P[12, 0] - c0) * e; c1 += (P[12, 1] - c1) * e; c2 += (P[12, 2] - c2) * e
                    land = 1.0
                elif d < 1.065: land = 1.0
            if shs > 0:
                a = X / shs + shx + 300; b = Y / shs + shy + 300; wa = _vn(a * 1.9, b * 1.9, 71); q = _vn(a + wa * .5, b + wa * .5, 70) * .62 + _vn(a * 2.6 + 3, b * 2.6, 72) * .26 + _vn(a * 6.1, b * 6.1, 73) * .12
                e = _ss(1 - shc + .02, 1 - shc - .02 + .06, q) * .42; c0 *= 1 - e * 1.15; c1 *= 1 - e * .95; c2 *= 1 - e * .70; glk = 1 - e * 2.2
            else: glk = 1.0
            if glit > 0 and land == 0:
                hx = vx + sun[0]; hy = vy + sun[1]; hz = vz + sun[2]
                if hz > 1e-3:
                    sx = hx / hz; sy = hy / hz; s2 = sx * sx + sy * sy; sg = .016 + .040 * rough
                    pr = math.exp(-s2 / (2 * sg)) * glit * glk
                    if pr > .004:
                        cz = math.log2(lat * 5.5 + 1e-6); ci = math.floor(cz); cs = 2.0 ** ci; asp = 1.0 / (vz + .10)
                        if asp > 7: asp = 7.0
                        for q in range(3):
                            cc = cs * (1.0, 2.0, 1.45)[q]; wb = _vn(X / (cc * 9) + 40, Y / (cc * 9 * asp) + 40, 81 + q) * 5.; u = X / (cc * 2.6) + wb + .5 * math.floor(Y / (cc * .9 * asp)); v = Y / (cc * .9 * asp) + wb * .6
                            iu = math.floor(u); iv = math.floor(v); h0 = _h3(int(iu), int(iv), 3, 9 + q); tk = int(math.floor(t * .5 + h0 * 7)); rnd = _h3(int(iu), int(iv), tk, 5 + q)
                            wt = (1 - (cz - ci)) if q == 0 else ((cz - ci) if q == 1 else .55)
                            if rnd < (pr * .95 * wt + max(0., pr - .80) * .5) * (.45 + 1.1 * max(0., n + .35)):
                                fu = (u - iu - .5 - (h0 - .5) * .5) * 2; fv = (v - iv - .5 - (_h3(int(iu), int(iv), 2, 8) - .5) * .7) * 2; rr = .22 + .62 * _h3(int(iu), int(iv), 1, 8) ** 1.5
                                if fu * fu + fv * fv * 2.6 < rr * rr: c0 = P[4, 0]; c1 = P[4, 1]; c2 = P[4, 2]
                        e = _ss(.45, .95, pr) * .22; c0 += (P[4, 0] - c0) * e; c1 += (P[4, 1] - c1) * e; c2 += (P[4, 2] - c2) * e
                        e = pr * pr * .25; c0 += (P[4, 0] - c0) * e; c1 += (P[4, 1] - c1) * e; c2 += (P[4, 2] - c2) * e
            e = 1 - math.exp(-(dist / haze_d) ** 1.5); out[y, x, 0] = c0 + (P[13, 0] - c0) * e; out[y, x, 1] = c1 + (P[13, 1] - c1) * e; out[y, x, 2] = c2 + (P[13, 2] - c2) * e
            D[y, x] = dist


def sea(cam, f=0, pal='day', sun=(.2, .9, .4), sky=None, rough=1., haze=26000., islands=(), shadows=(0., 0., 0., 0.), glitter=1., swell=220.):
    """-> rgb, distance (0 where there is sky).  swell = longest wave in metres: 220 from high up, 14 for a camera skimming the water"""
    P = np.stack([rgb(c) for c in (SEA[pal] if isinstance(pal, str) else pal)]).astype(f32) if not isinstance(pal, np.ndarray) else pal.astype(f32)
    out = np.zeros((cam.h, cam.w, 3), f32) if sky is None else sky.astype(f32).copy(); D = np.zeros((cam.h, cam.w), f32); s = np.asarray(sun, f32); s = s / np.linalg.norm(s)
    I = np.asarray(islands, f32).reshape(-1, 4)
    _sea(out, D, np.asarray(cam.eye, f32), cam.R.astype(f32), f32(cam.f), f32(cam.cx), f32(cam.cy), P, s.astype(f32), f32(f), f32(rough), f32(haze), I, f32(shadows[0]), f32(shadows[1]), f32(shadows[2]), f32(shadows[3]), f32(glitter), f32(swell), f32(0))
    return out, D


def mixpal(a, b, t):
    A, B = [np.stack([rgb(c) for c in (SEA[q] if isinstance(q, str) else q)]) if not isinstance(q, np.ndarray) else q for q in (a, b)]; return (A + (B - A) * t).astype(f32)
