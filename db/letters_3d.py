"""EGO as monuments: solid letters, 30 m tall, standing in the waste.  The E cracks, leans, falls; GO remains.

    M = Monument(); M.setup()                      # static world, physics of the fall (cached in cache/)
    cam = M.camera(push=0.)                        # db.cam.Camera3D (world: X along the word, Y away, Z up, metres)
    img, g = M.render(cam, t)                      # t = frames (d6 local).  RGB float + g-buffer dict(Z distance, P world, ID)
    M.put(img, g, layer, (x, y, z))                # premultiplied layer standing at a world point: hidden by what is nearer
    Timing: M.T_CRACK (34), M.T_FALL (60).  World layout: E x 0-21.9, G 25.4-51.0, O 54.5-80.3, fronts at y = 0, backs at y = 8.
"""
import numpy as np, cv2, math, os
from numba import njit, prange
from . import W, H, comp
from .paint import rgb, f32, fbm, smooth, grid, mix, lum, _h2
from .cam import Camera3D
from .letters import EGO, GAP

K = .03                        # metres per em unit: letters are 30 m tall
DEP = 8.0                      # thickness
GAP3 = 118.
X0 = {'E': 0., 'G': (730 + GAP3) * K, 'O': (730 + GAP3 + 852 + GAP3) * K}
TX, TM = 2.5, 150.             # sdf texture: em per texel, margin


# ------------------------------------------------------------ noise, letter distance
@njit(cache=True, fastmath=True)
def _h3(ix, iy, iz, s):
    n = (ix * 374761393 + iy * 668265263 + iz * 2147483647 + s * 1274126177) & 0xffffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0xffffffff
    return ((n ^ (n >> 16)) & 0xffff) / 65535.0


@njit(cache=True, fastmath=True)
def vn3(x, y, z, s):
    ix = int(math.floor(x)); iy = int(math.floor(y)); iz = int(math.floor(z))
    fx = x - ix; fy = y - iy; fz = z - iz
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz)
    def_ = 0.0
    a = _h3(ix, iy, iz, s); b = _h3(ix + 1, iy, iz, s); c = _h3(ix, iy + 1, iz, s); d = _h3(ix + 1, iy + 1, iz, s)
    e = _h3(ix, iy, iz + 1, s); f = _h3(ix + 1, iy, iz + 1, s); g = _h3(ix, iy + 1, iz + 1, s); h = _h3(ix + 1, iy + 1, iz + 1, s)
    p = (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy
    q = (e + (f - e) * fx) * (1 - fy) + (g + (h - g) * fx) * fy
    return p + (q - p) * fz + def_


@njit(cache=True, fastmath=True)
def _tex(T, x, y):
    h, w = T.shape
    if x < 0: x = 0.
    if y < 0: y = 0.
    if x > w - 1.001: x = w - 1.001
    if y > h - 1.001: y = h - 1.001
    ix = int(x); iy = int(y); fx = x - ix; fy = y - iy
    return (T[iy, ix] * (1 - fx) + T[iy, ix + 1] * fx) * (1 - fy) + (T[iy + 1, ix] * (1 - fx) + T[iy + 1, ix + 1] * fx) * fy


@njit(cache=True, fastmath=True)
def facet(x, y, z, sc, s):
    """broken-stone value: every jittered cell is its own tilted flat plane, about -.5 .. .5"""
    x /= sc; y /= sc; z /= sc
    ix = int(math.floor(x - .5)); iy = int(math.floor(y - .5)); iz = int(math.floor(z - .5)); bd = 1e9; v = 0.
    for c in range(8):
        jx = ix + (c & 1); jy = iy + ((c >> 1) & 1); jz = iz + (c >> 2)
        a = _h3(jx, jy, jz, s); b = _h3(jx, jy, jz, s + 1); e = _h3(jx, jy, jz, s + 2)
        px = x - jx - .5 - (a - .5) * .7; py = y - jy - .5 - (b - .5) * .7; pz = z - jz - .5 - (e - .5) * .7
        d = px * px + py * py + pz * pz
        if d < bd: bd = d; f = _h3(jx, jy, jz, s + 3); v = f - .5 + (px * (a - .5) + py * (e - .5) + pz * (b - .5)) * 1.1
    return v


@njit(cache=True, fastmath=True)
def sd_let(T, x, y, z, seed):
    """letter local metres: x along, y depth (0 front .. DEP), z up.  signed distance with chamfered, chipped edges"""
    d2 = _tex(T, (x / 0.03 + 150.) / 2.5, (1000. - z / 0.03 + 150.) / 2.5)
    dz = abs(y - 4.0) - 4.0
    d = max(d2, dz)
    if d > 2.4: return d - .3
    e_ = d2 + dz
    if e_ > -5. and d < .9:                                                  # along edges: chamfer, then faceted breaks in colonies
        n1 = vn3(x * .15 + 3.1, y * .15, z * .15, seed)
        ch = .16 + 2.6 * max(0., n1 - .50) * (1. + 1.9 * facet(x, y, z, 2.6, seed + 20)) + .55 * max(0., facet(x, y, z, 1.05, seed + 24) - .12)
        d = max(d, (e_ + ch) * .7071)
    if d < .6:
        s1 = vn3(x * .10, y * .10, z * .075, seed + 2)                        # spalled faces: plates of surface gone
        if s1 > .60:
            st = (s1 - .60) * 6. + .6 * facet(x, y, z, 1.9, seed + 28)
            if st > 0: d += .30 * min(1., st * 1.6)
        # courses of the stone: thin grooves every 4.5 m, staggered joints
        zc = z * .222; f = abs(zc - math.floor(zc) - .5) * 4.5; gr = 2.25 - f
        row = int(math.floor(zc)); xc = (x + y + (row & 1) * 3.1 + 1.7 * (_h3(row, 3, 1, seed) - .5)) * .161; gx = 3.1 - abs(xc - math.floor(xc) - .5) * 6.2
        m = min(gr, gx)
        if m < .06: d += .06
    return d


@njit(cache=True, fastmath=True)
def sd_cell(SD, i, x, y, z, seed):
    """how far outside fracture cell i (voronoi in the letter plane, boundaries warped so they break like stone)"""
    px = x + 1.5 * (vn3(x * .30, y * .22, z * .30, seed) - .5) + 1.5 * facet(x, y, z, 2.3, seed + 5) + .5 * facet(x, y, z, .8, seed + 9) + .16 * (y - 4.0)
    pz = z + 1.5 * (vn3(x * .30 + 7., y * .22, z * .30 + 5., seed + 1) - .5) + 1.5 * facet(x, y, z, 2.3, seed + 13) + .5 * facet(x, y, z, .8, seed + 17) - .10 * (y - 4.0)
    best = -1e9
    ax = SD[i, 0]; az = SD[i, 1]
    for j in range(SD.shape[0]):
        if j == i: continue
        bx = SD[j, 0] - ax; bz = SD[j, 1] - az; L = math.sqrt(bx * bx + bz * bz)
        q = ((px - ax - bx * .5) * bx + (pz - az - bz * .5) * bz) / L
        if q > best: best = q
    return best


@njit(cache=True, fastmath=True)
def sd_obj(T, SD, piece, x, y, z, seed):
    d = sd_let(T, x, y, z, seed)
    if piece >= 0 and d < 3.:
        c = sd_cell(SD, piece, x, y, z, 77)
        if c > d: d = c
    return d


@njit(cache=True, fastmath=True, parallel=True)
def trace(T, SD, piece, seed, lo, hi, R, tr, piv, ox, O0, OA, OB, D0, DA, DB, x0, x1, y0, y1, Z, N, PL, ID, oid, maxst, far):
    """sphere trace one rigid piece.  ray(px, py) = O0 + px OA + py OB  +  t * normalise(D0 + px DA + py DB)"""
    for py in prange(y0, y1):
        for px in range(x0, x1):
            o0 = O0[0] + px * OA[0] + py * OB[0]; o1 = O0[1] + px * OA[1] + py * OB[1]; o2 = O0[2] + px * OA[2] + py * OB[2]
            d0 = D0[0] + px * DA[0] + py * DB[0]; d1 = D0[1] + px * DA[1] + py * DB[1]; d2 = D0[2] + px * DA[2] + py * DB[2]
            L = 1.0 / math.sqrt(d0 * d0 + d1 * d1 + d2 * d2); d0 *= L; d1 *= L; d2 *= L
            # into the piece
            q0 = o0 - tr[0] - piv[0]; q1 = o1 - tr[1] - piv[1]; q2 = o2 - tr[2] - piv[2]
            a0 = R[0, 0] * q0 + R[1, 0] * q1 + R[2, 0] * q2 + piv[0] - ox; a1 = R[0, 1] * q0 + R[1, 1] * q1 + R[2, 1] * q2 + piv[1]; a2 = R[0, 2] * q0 + R[1, 2] * q1 + R[2, 2] * q2 + piv[2]
            b0 = R[0, 0] * d0 + R[1, 0] * d1 + R[2, 0] * d2; b1 = R[0, 1] * d0 + R[1, 1] * d1 + R[2, 1] * d2; b2 = R[0, 2] * d0 + R[1, 2] * d1 + R[2, 2] * d2
            tn = 0.0; tf = min(far, Z[py, px])
            for k in range(3):
                a = a0 if k == 0 else (a1 if k == 1 else a2); b = b0 if k == 0 else (b1 if k == 1 else b2)
                if abs(b) < 1e-9:
                    if a < lo[k] or a > hi[k]: tn = 1e9
                else:
                    u = (lo[k] - a) / b; v = (hi[k] - a) / b
                    if u > v: u, v = v, u
                    if u > tn: tn = u
                    if v < tf: tf = v
            if tn >= tf: continue
            t = tn; hit = False
            for s in range(maxst):
                d = sd_obj(T, SD, piece, a0 + b0 * t, a1 + b1 * t, a2 + b2 * t, seed)
                if d < .012 + t * .00006: hit = True; break
                t += d * .78 if d < 1.2 else d * .92
                if t > tf: break
            if not hit: continue
            x = a0 + b0 * t; y = a1 + b1 * t; z = a2 + b2 * t; e = .07
            g1 = sd_obj(T, SD, piece, x + e, y - e, z - e, seed); g2 = sd_obj(T, SD, piece, x - e, y - e, z + e, seed); g3 = sd_obj(T, SD, piece, x - e, y + e, z - e, seed); g4 = sd_obj(T, SD, piece, x + e, y + e, z + e, seed)
            n0 = g1 - g2 - g3 + g4; n1 = -g1 - g2 + g3 + g4; n2 = -g1 + g2 - g3 + g4; m = 1.0 / math.sqrt(n0 * n0 + n1 * n1 + n2 * n2 + 1e-12)
            n0 *= m; n1 *= m; n2 *= m
            Z[py, px] = t; ID[py, px] = oid
            PL[py, px, 0] = x; PL[py, px, 1] = y; PL[py, px, 2] = z
            N[py, px, 0] = R[0, 0] * n0 + R[0, 1] * n1 + R[0, 2] * n2; N[py, px, 1] = R[1, 0] * n0 + R[1, 1] * n1 + R[1, 2] * n2; N[py, px, 2] = R[2, 0] * n0 + R[2, 1] * n1 + R[2, 2] * n2


@njit(cache=True, fastmath=True)
def hulls(PN, PD, CNT, C, RAD, boxes, O0, OA, OB, D0, DA, DB, Z, N, ID, FID, oid0):
    """convex rocks: planes n.x <= d about the centre C"""
    for i in range(C.shape[0]):
        x0, x1, y0, y1 = boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3]
        for py in range(y0, y1):
            for px in range(x0, x1):
                o0 = O0[0] + px * OA[0] + py * OB[0] - C[i, 0]; o1 = O0[1] + px * OA[1] + py * OB[1] - C[i, 1]; o2 = O0[2] + px * OA[2] + py * OB[2] - C[i, 2]
                d0 = D0[0] + px * DA[0] + py * DB[0]; d1 = D0[1] + px * DA[1] + py * DB[1]; d2 = D0[2] + px * DA[2] + py * DB[2]
                L = 1.0 / math.sqrt(d0 * d0 + d1 * d1 + d2 * d2); d0 *= L; d1 *= L; d2 *= L
                tn = 0.0; tf = Z[py, px]; kn = -1; ok = True
                for k in range(CNT[i]):
                    a = PN[i, k, 0]; b = PN[i, k, 1]; c = PN[i, k, 2]
                    den = a * d0 + b * d1 + c * d2; num = PD[i, k] - (a * o0 + b * o1 + c * o2)
                    if abs(den) < 1e-9:
                        if num < 0: ok = False; break
                    else:
                        t = num / den
                        if den < 0:
                            if t > tn: tn = t; kn = k
                        elif t < tf: tf = t
                    if tn >= tf: ok = False; break
                if not ok or kn < 0: continue
                Z[py, px] = tn; ID[py, px] = oid0; FID[py, px] = i * 64 + kn
                N[py, px, 0] = PN[i, kn, 0]; N[py, px, 1] = PN[i, kn, 1]; N[py, px, 2] = PN[i, kn, 2]


@njit(cache=True, fastmath=True)
def _hgt(Hm, x, y):
    return _tex(Hm, (x + 256.) * 2., (y + 256.) * 2.)


@njit(cache=True, fastmath=True, parallel=True)
def ground(Hm, O0, OA, OB, D0, DA, DB, y0, Z, N, ID, oid, far):
    h, w = Z.shape
    for py in prange(y0, h):
        for px in range(w):
            o0 = O0[0] + px * OA[0] + py * OB[0]; o1 = O0[1] + px * OA[1] + py * OB[1]; o2 = O0[2] + px * OA[2] + py * OB[2]
            d0 = D0[0] + px * DA[0] + py * DB[0]; d1 = D0[1] + px * DA[1] + py * DB[1]; d2 = D0[2] + px * DA[2] + py * DB[2]
            L = 1.0 / math.sqrt(d0 * d0 + d1 * d1 + d2 * d2); d0 *= L; d1 *= L; d2 *= L
            lim = min(far, Z[py, px]); t = .5; hit = False; pt = t
            for s in range(170):
                if t > lim: break
                x = o0 + d0 * t; y = o1 + d1 * t; z = o2 + d2 * t
                dh = z - _hgt(Hm, x, y)
                if dh < 0: hit = True; break
                pt = t; t += max(.05 + t * .006, dh * .55)
                if d2 > 0 and z > 14.: break
            if not hit: continue
            a = pt; b = t
            for s in range(9):
                m = .5 * (a + b)
                if o2 + d2 * m - _hgt(Hm, o0 + d0 * m, o1 + d1 * m) < 0: b = m
                else: a = m
            t = b
            if t >= Z[py, px]: continue
            x = o0 + d0 * t; y = o1 + d1 * t; e = .35 + t * .004
            gx = (_hgt(Hm, x + e, y) - _hgt(Hm, x - e, y)) / (2 * e); gy = (_hgt(Hm, x, y + e) - _hgt(Hm, x, y - e)) / (2 * e); m = 1 / math.sqrt(1 + gx * gx + gy * gy)
            Z[py, px] = t; ID[py, px] = oid; N[py, px, 0] = -gx * m; N[py, px, 1] = -gy * m; N[py, px, 2] = m


@njit(cache=True, fastmath=True, parallel=True)
def shadow_look(P, SMz, o, ax, ay, dr, s, bias, out, M):
    h, w = out.shape; n = SMz.shape[0]
    for y in prange(h):
        for x in range(w):
            if not M[y, x]: continue
            q0 = P[y, x, 0] - o[0]; q1 = P[y, x, 1] - o[1]; q2 = P[y, x, 2] - o[2]
            u = (q0 * ax[0] + q1 * ax[1] + q2 * ax[2]) * s; v = (q0 * ay[0] + q1 * ay[1] + q2 * ay[2]) * s; t = q0 * dr[0] + q1 * dr[1] + q2 * dr[2] - bias
            iu = int(math.floor(u - .5)); iv = int(math.floor(v - .5))
            if iu < 0 or iv < 0 or iu >= n - 1 or iv >= n - 1: continue
            fu = u - .5 - iu; fv = v - .5 - iv
            a = 1. if SMz[iv, iu] < t else 0.; b = 1. if SMz[iv, iu + 1] < t else 0.; c = 1. if SMz[iv + 1, iu] < t else 0.; d = 1. if SMz[iv + 1, iu + 1] < t else 0.
            out[y, x] = (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


@njit(cache=True, fastmath=True, parallel=True)
def shafts(O0, OA, OB, D0, DA, DB, Z, q, SMz, o, ax, ay, dr, s, n, far, out, gh):
    """light scattered in dusty air along each view ray (low resolution), using the shadow map"""
    h, w = out.shape; m = SMz.shape[0]
    for y in prange(h):
        for x in range(w):
            px = (x + .5) * q; py = (y + .5) * q
            d0 = D0[0] + px * DA[0] + py * DB[0]; d1 = D0[1] + px * DA[1] + py * DB[1]; d2 = D0[2] + px * DA[2] + py * DB[2]
            L = 1.0 / math.sqrt(d0 * d0 + d1 * d1 + d2 * d2); d0 *= L; d1 *= L; d2 *= L
            tm = Z[min(int(py), Z.shape[0] - 1), min(int(px), Z.shape[1] - 1)]
            if tm > far: tm = far
            acc = 0.; jit = _h2(x, y, 5)
            for k in range(n):
                t = tm * (k + jit) / n
                X = O0[0] + d0 * t; Y = O0[1] + d1 * t; Zz = O0[2] + d2 * t
                q0 = X - o[0]; q1 = Y - o[1]; q2 = Zz - o[2]
                u = (q0 * ax[0] + q1 * ax[1] + q2 * ax[2]) * s; v = (q0 * ay[0] + q1 * ay[1] + q2 * ay[2]) * s; r = q0 * dr[0] + q1 * dr[1] + q2 * dr[2]
                iu = int(u); iv = int(v); lit = 1.
                if iu >= 0 and iv >= 0 and iu < m and iv < m:
                    if SMz[iv, iu] < r: lit = 0.
                acc += lit * math.exp(-max(0., Zz) / gh) * math.exp(-t / 260.)
            out[y, x] = acc * tm / n


# ------------------------------------------------------------ dust: cauliflower puffs as z-buffered balls
@njit(cache=True, fastmath=True)
def balls(B, ex, ey, ez, R, f, cx, cy, Z, N, AG, ID):
    """B (n, 6): x y z radius age id"""
    h, w = Z.shape
    for k in range(B.shape[0]):
        q0 = B[k, 0] - ex; q1 = B[k, 1] - ey; q2 = B[k, 2] - ez
        a = R[0, 0] * q0 + R[0, 1] * q1 + R[0, 2] * q2; b = R[1, 0] * q0 + R[1, 1] * q1 + R[1, 2] * q2; c = R[2, 0] * q0 + R[2, 1] * q1 + R[2, 2] * q2
        if c < 1.: continue
        s = f / c; u = a * s + cx; v = b * s + cy; r = B[k, 3] * s
        if r < .6: continue
        x0 = max(0, int(u - r)); x1 = min(w, int(u + r) + 2); y0 = max(0, int(v - r)); y1 = min(h, int(v + r) + 2)
        for y in range(y0, y1):
            for x in range(x0, x1):
                d0 = (x - u) / r; d1 = (y - v) / r; d2 = d0 * d0 + d1 * d1
                if d2 >= 1: continue
                n = math.sqrt(1 - d2); t = c - n * B[k, 3]
                if t < Z[y, x]: Z[y, x] = t; N[y, x, 0] = d0; N[y, x, 1] = d1; N[y, x, 2] = n; AG[y, x] = B[k, 4]; ID[y, x] = int(B[k, 5])


@njit(cache=True, fastmath=True)
def cells(PL, M, SD, I, J, B):
    """fracture cell, nearest neighbour cell and distance to their common boundary, per E pixel"""
    h, w = M.shape
    for y in range(h):
        for x in range(w):
            if not M[y, x]: continue
            X, Y, Zz = PL[y, x, 0], PL[y, x, 1], PL[y, x, 2]; s = 77
            px = X + 1.5 * (vn3(X * .30, Y * .22, Zz * .30, s) - .5) + 1.5 * facet(X, Y, Zz, 2.3, s + 5) + .5 * facet(X, Y, Zz, .8, s + 9) + .16 * (Y - 4.0)
            pz = Zz + 1.5 * (vn3(X * .30 + 7., Y * .22, Zz * .30 + 5., s + 1) - .5) + 1.5 * facet(X, Y, Zz, 2.3, s + 13) + .5 * facet(X, Y, Zz, .8, s + 17) - .10 * (Y - 4.0)
            bi = -1; bd = 1e9
            for i in range(SD.shape[0]):
                q = (px - SD[i, 0]) ** 2 + (pz - SD[i, 1]) ** 2
                if q < bd: bd = q; bi = i
            bj = -1; bb = 1e9
            for j in range(SD.shape[0]):
                if j == bi: continue
                bx = SD[j, 0] - SD[bi, 0]; bz = SD[j, 1] - SD[bi, 1]; L = math.sqrt(bx * bx + bz * bz)
                q = -((px - SD[bi, 0] - bx * .5) * bx + (pz - SD[bi, 1] - bz * .5) * bz) / L
                if q < bb: bb = q; bj = j
            I[y, x] = bi; J[y, x] = bj; B[y, x] = bb


# ------------------------------------------------------------ the fall: rigid lumps of particles (shape matching over clusters that split in time)
@njit(cache=True, fastmath=True)
def _sim(X, pc, np_, te, cl, free, grav, push, pwin, dt, nsub, nfr, Hm, rad, Rs, Ts, IM):
    """te[k], cl[k, p]: from time te[k] on piece p belongs to cluster cl[k, p]; free[p]: time it stops being held by the ground"""
    n = X.shape[0]; V = np.zeros((n, 3)); X0 = X.copy(); C0 = np.zeros((np_, 3)); cnt = np.zeros(np_); XP = X.copy()
    for i in range(n):
        for k in range(3): C0[pc[i], k] += X[i, k]
        cnt[pc[i]] += 1
    for p in range(np_):
        for k in range(3): C0[p, k] /= max(1., cnt[p])
    T = 0.0; h = dt / nsub; A = np.zeros((np_, 3, 3)); C = np.zeros((np_, 3)); G = np.zeros((np_, 3)); m_ = np.zeros(np_); fr_ = np.zeros(np_); RR = np.zeros((np_, 3, 3)); lo = np.zeros((np_, 3)); hi = np.zeros((np_, 3))
    for fr in range(nfr):
        imp = np.zeros(np_)
        for s in range(nsub):
            T += h; ek = 0
            for k in range(te.shape[0]):
                if T >= te[k]: ek = k
            for i in range(n):
                p = pc[i]
                for k in range(3): XP[i, k] = X[i, k]
                if free[p] > T: continue
                V[i, 2] -= grav * h
                if pwin[0] < T < pwin[1]:
                    u = (X0[i, 2] - 8.) / 22.
                    V[i, 0] += push[0] * u * h; V[i, 1] += push[1] * u * h
                for k in range(3): V[i, k] *= .9994; X[i, k] += V[i, k] * h
            for p in range(np_):
                for k in range(3): lo[p, k] = 1e9; hi[p, k] = -1e9
            for i in range(n):
                p = pc[i]
                for k in range(3):
                    if X[i, k] < lo[p, k]: lo[p, k] = X[i, k]
                    if X[i, k] > hi[p, k]: hi[p, k] = X[i, k]
            for i in range(n):
                p = pc[i]
                for j in range(i + 1, n):
                    q_ = pc[j]
                    if cl[ek, q_] == cl[ek, p]: continue
                    dx = X[j, 0] - X[i, 0]
                    if dx > 1.5 or dx < -1.5: continue
                    dy = X[j, 1] - X[i, 1]; dz = X[j, 2] - X[i, 2]; L2 = dx * dx + dy * dy + dz * dz
                    if L2 > 2.2: continue
                    rx = X0[j, 0] - X0[i, 0]; ry = X0[j, 1] - X0[i, 1]; rz = X0[j, 2] - X0[i, 2]; r0 = math.sqrt(rx * rx + ry * ry + rz * rz); m = min(2 * rad, r0 * .985)
                    if L2 < m * m:
                        L = math.sqrt(L2) + 1e-9; wi = 0. if free[p] > T else 1.; wj = 0. if free[q_] > T else 1.
                        if wi + wj == 0: continue
                        vn = ((X[j, 0] - XP[j, 0] - X[i, 0] + XP[i, 0]) * dx + (X[j, 1] - XP[j, 1] - X[i, 1] + XP[i, 1]) * dy + (X[j, 2] - XP[j, 2] - X[i, 2] + XP[i, 2]) * dz) / L
                        q = min((m - L) * .4, max(0., -vn) * .5 + 1.2 * h) / L / (wi + wj)          # stop the approach, then ease apart: no energy from nothing
                        if wi > 0:
                            X[i, 0] -= dx * q; X[i, 1] -= dy * q; X[i, 2] -= dz * q
                        if wj > 0:
                            X[j, 0] += dx * q; X[j, 1] += dy * q; X[j, 2] += dz * q
                        # friction between faces
                        for k in range(3):
                            rel = ((X[j, k] - XP[j, k]) - (X[i, k] - XP[i, k])) * .10
                            if wi > 0: X[i, k] += rel * (wi / (wi + wj))
                            if wj > 0: X[j, k] -= rel * (wj / (wi + wj))
            for i in range(n):
                g = _hgt(Hm, X[i, 0], X[i, 1]) + rad * .5
                if X[i, 2] < g and free[pc[i]] <= T:
                    vz = (XP[i, 2] - X[i, 2]) / h
                    if vz > 1.5: imp[pc[i]] += vz
                    X[i, 2] = min(g, max(X[i, 2], XP[i, 2] + 1.0 * h))
                    for k in range(2): X[i, k] += (XP[i, k] - X[i, k]) * .75
            for p in range(np_):
                m_[p] = 0.; fr_[p] = 0.
                for a in range(3):
                    C[p, a] = 0.; G[p, a] = 0.
                    for b in range(3): A[p, a, b] = 0.
            for i in range(n):
                c = cl[ek, pc[i]]; m_[c] += 1
                if free[pc[i]] > T: fr_[c] = 1.
                for k in range(3): C[c, k] += X[i, k]; G[c, k] += X0[i, k]
            for c in range(np_):
                for k in range(3): C[c, k] /= max(1., m_[c]); G[c, k] /= max(1., m_[c])
            for i in range(n):
                c = cl[ek, pc[i]]
                for a in range(3):
                    for b in range(3): A[c, a, b] += (X[i, a] - C[c, a]) * (X0[i, b] - G[c, b])
            for c in range(np_):
                for a in range(3):
                    for b in range(3): RR[c, a, b] = 1. if a == b else 0.
                if m_[c] < 4 or fr_[c] > 0: continue
                U, S_, Vt = np.linalg.svd(A[c]); Rm = U @ Vt
                if np.linalg.det(Rm) < 0: U[:, 2] = -U[:, 2]; Rm = U @ Vt
                RR[c] = Rm
            for i in range(n):
                c = cl[ek, pc[i]]
                if fr_[c] > 0:
                    if free[pc[i]] > T:
                        for a in range(3): X[i, a] = X0[i, a]
                    continue
                for a in range(3):
                    g = C[c, a]
                    for b in range(3): g += RR[c, a, b] * (X0[i, b] - G[c, b])
                    X[i, a] = g
            for i in range(n):
                for k in range(3): V[i, k] = (X[i, k] - XP[i, k]) / h
                sp = math.sqrt(V[i, 0] ** 2 + V[i, 1] ** 2 + V[i, 2] ** 2)
                if sp > 34.:
                    for k in range(3): V[i, k] *= 34. / sp
        for p in range(np_):
            c = cl[ek, p]
            for a in range(3):
                q = C[c, a]
                for b in range(3):
                    q += RR[c, a, b] * (C0[p, b] - G[c, b]); Rs[fr, p, a, b] = RR[c, a, b]
                Ts[fr, p, a] = q - C0[p, a]
            if free[p] > T:
                for a in range(3): Ts[fr, p, a] = 0.
            IM[fr, p] = imp[p] / max(1., cnt[p])
    return C0


def fxaa(img, span=8., red=1 / 8., mn=1 / 128.):
    """edge antialiasing of a finished 1-sample picture (no softening elsewhere)"""
    h, w = img.shape[:2]; x, y = grid(h, w); Lm = np.sqrt(np.clip(lum(img), 0, 1.5)).astype(f32); sh = lambda a, dx, dy: cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    nw, ne, sw, se = sh(Lm, 1, 1), sh(Lm, -1, 1), sh(Lm, 1, -1), sh(Lm, -1, -1); lo = np.minimum.reduce([Lm, nw, ne, sw, se]); hi = np.maximum.reduce([Lm, nw, ne, sw, se])
    dx = -((nw + ne) - (sw + se)); dy = (nw + sw) - (ne + se); r = np.maximum((nw + ne + sw + se) * .25 * red, mn); k = 1 / (np.minimum(np.abs(dx), np.abs(dy)) + r); dx = np.clip(dx * k, -span, span); dy = np.clip(dy * k, -span, span)
    S = lambda t: cv2.remap(img.astype(f32), (x + dx * t).astype(f32), (y + dy * t).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    A = .5 * (S(1 / 3. - .5) + S(2 / 3. - .5)); B = A * .5 + .25 * (S(-.5) + S(.5)); lb = np.sqrt(np.clip(lum(B), 0, 1.5)); o = np.where(((lb < lo) | (lb > hi))[..., None], A, B); e = smooth(.035, .08, hi - lo)[..., None]; return img * (1 - e) + o * e


# ------------------------------------------------------------ helpers
def poly_sdf(P, x, y):
    """exact signed distance to a closed polygon (negative inside)"""
    d = np.full(x.shape, 1e18); ins = np.zeros(x.shape, bool); n = len(P)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]; e = b - a; L = e @ e
        if L < 1e-9: continue
        wx, wy = x - a[0], y - a[1]; t = np.clip((wx * e[0] + wy * e[1]) / L, 0, 1); d = np.minimum(d, (wx - e[0] * t) ** 2 + (wy - e[1] * t) ** 2)
        c = (a[1] > y) != (b[1] > y); ins ^= c & (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1] + 1e-12) + a[0])
    return np.where(ins, -1., 1.) * np.sqrt(d)


def letter_sdf(ch, chips=True, seed=2):
    """distance texture in metres (TX em per texel), with a few corners broken off"""
    w, h = int(round((860 + 2 * TM) / TX)), int(round((1000 + 2 * TM) / TX)); x, y = np.meshgrid(np.arange(w) * TX - TM, np.arange(h) * TX - TM); P = EGO[ch]['p']; r = np.random.default_rng(seed + ord(ch))
    d = poly_sdf(P[0], x, y)
    for q in P[1:]: d = np.maximum(d, -poly_sdf(q, x, y))
    if chips:
        for k in range(8):
            c = P[0][int(r.integers(len(P[0])))] + r.normal(0, 10, 2)
            if c[1] > 930: continue
            R = r.uniform(24, 66); n = int(r.integers(5, 8)); a = np.sort(np.linspace(0, 2 * math.pi, n, endpoint=False) + r.normal(0, .3, n)); rad = R * r.uniform(.7, 1.2, n); d = np.maximum(d, -poly_sdf(np.c_[c[0] + np.cos(a) * rad, c[1] + np.sin(a) * rad], x, y))
    return (d * K).astype(f32)


def rock_planes(r, R, flat=.7, n=15, sharp=.34, base=True):
    """random convex rock about the origin: (normals, offsets).  Directions spread evenly, so it is always a closed block"""
    k = np.arange(n) + .5; ph = np.arccos(1 - 1.72 * k / n); th = math.pi * (1 + 5 ** .5) * k + r.uniform(0, 6); v = np.c_[np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)] + r.normal(0, .20, (n, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    d = R * r.uniform(1 - sharp, 1.0, n); st = r.uniform(.75, 1.45); s = np.array([st, 1 / st, flat]); nn = v / s; L = np.linalg.norm(nn, axis=1); nn = nn / L[:, None]; dd = d / L
    if base: nn = np.vstack([nn, [0, 0, -1]]); dd = np.r_[dd, R * .6]
    ang = r.uniform(0, 2 * math.pi); tl = r.normal(0, .22); c, q = math.cos(ang), math.sin(ang); Rz = np.array([[c, q, 0], [-q, c, 0], [0, 0, 1]]); c, q = math.cos(tl), math.sin(tl); Rx = np.array([[1, 0, 0], [0, c, q], [0, -q, c]]); return nn @ Rx @ Rz, dd


def mesa_planes(r, R, h, n=9, batter=.10):
    a = np.sort(r.uniform(0, 2 * math.pi, n)) ; nn = np.c_[np.cos(a), np.sin(a), np.full(n, batter) * r.uniform(.4, 1.6, n)]; L = np.linalg.norm(nn, axis=1); nn /= L[:, None]; dd = R * r.uniform(.72, 1.05, n) / L
    nn = np.vstack([nn, [r.normal(0, .03), r.normal(0, .03), 1.]]); dd = np.r_[dd, h]; nn[-1] /= np.linalg.norm(nn[-1]); return np.vstack([nn, [0, 0, -1]]), np.r_[dd, 30.]


# E breaks into twelve: seeds in letter metres (x, z)
SEEDS = np.array([(5.4, 1.8), (11.6, 2.4), (18.4, 4.2), (2.4, 8.4), (6.6, 12.4), (2.0, 15.6), (13.6, 14.8), (6.4, 19.2), (2.2, 22.6), (5.0, 28.0), (12.4, 26.6), (18.8, 25.8)], float)


class Monument:
    T_CRACK, T_FALL = 34., 60.
    parts = 'EGO'                                                               # letters standing (set to 'GO' once the E is gone)
    L = np.array([.205, .91, .235]); L = L / np.linalg.norm(L)                   # toward the moon (behind, a little left)
    VER = 'v6'
    PAL = dict(top='#06050f', sky='#120e2b', mid='#221a4a', low='#3b2f74', haze='#57469c', line='#07060f', moon='#fbf7ff', fire='#ff8a3c')

    # ------------------------------------------------ world
    def setup(s, fall=True):
        s.T = {c: letter_sdf(c) for c in 'EGO'}; s.SD = SEEDS.copy(); r = np.random.default_rng(11)
        # ground
        n = 1024; hm = .55 * fbm(n, n, (150, 60, 24), (1, .5, .28), 5) + .10 * fbm(n, n, (9, 4), (1, .6), 6); x, y = grid(n, n); wx, wy = x / 2. - 256, y / 2. - 256
        near = np.exp(-(np.maximum(0, np.abs(wx - 38) - 42) ** 2 + np.maximum(0, np.abs(wy - 4) - 6) ** 2) / 90.); hm = hm * (1 - .75 * near) + .75 * near                                       # bedded feet
        s.Hm = hm.astype(f32)
        hz = lambda px, py: float(s.Hm[int(np.clip((py + 256) * 2, 0, n - 1)), int(np.clip((px + 256) * 2, 0, n - 1))])
        # rocks
        PN, PD, C = [], [], []
        def add(p, nd): PN.append(nd[0]); PD.append(nd[1]); C.append(p)
        for _ in range(150):                                                    # rubble round the feet
            c = r.choice(['E', 'G', 'O']); wd = EGO[c]['w'] * K; px = X0[c] + r.uniform(-3, wd + 3); py = r.choice([r.normal(-2.2, 1.6), r.normal(9.5, 1.5)], p=[.8, .2]); R = r.gamma(1.4, .42) + .18; add((px, py, hz(px, py) + R * .10), rock_planes(r, R, r.uniform(.5, .9)))
        for _ in range(230):                                                    # the waste
            a = r.uniform(-.40, 1.5); q = r.uniform(9, 120) ** 1.0; px, py = -26 + math.sin(a) * q, -92 + math.cos(a) * q
            if -6 < px < 82 and -5 < py < 13: continue
            R = r.gamma(1.1, .33) * (1 + q / 120.) + .10; add((px, py, hz(px, py) + R * .08), rock_planes(r, R, r.uniform(.45, .95)))
        for px, py, R, fl in ((-20.5, -80.2, 1.15, .62), (-13.6, -76.5, .55, .7), (-9.5, -82.0, .8, .6), (6.0, -70.0, 1.3, .7), (9.4, -67.6, .62, .8), (31., -62., 1.7, .6), (24.4, -71., .8, .6), (44., -52., 2.2, .62), (48.2, -50., 1.0, .7), (-14., -40., 1.9, .7), (-11., -38., .9, .8), (-34, -50, 2.6, .7), (66., -22., 2.4, .7)): add((px, py, hz(px, py) + R * .05), rock_planes(r, R, fl, 17))
        r2 = np.random.default_rng(77)
        for _ in range(170):
            a = r2.uniform(-2.2, 2.6); q = 4.5 + 60 * r2.random() ** 1.3; px, py = -6 + math.sin(a) * q, -58 + math.cos(a) * q; d0 = math.hypot(px + 24, py + 86); aa = math.atan2(px + 24, py + 86)
            if (-.1 < aa < 1.1 and d0 < 60 and q > 9) or (-6 < px < 82 and -6 < py < 13): continue                 # keep the stage of the wide shot clean
            R = r2.gamma(1.2, .16) * (1 + q / 40.) + .05; add((px, py, hz(px, py) + R * .08), rock_planes(r2, R, r2.uniform(.45, .9)))
        s.n_small = len(C)
        s.n_stat = len(C); s._rocks(PN, PD, C)
        # the fall
        pa = f'cache/mon_fall_{s.VER}.npz'; os.makedirs('cache', exist_ok=True)
        if fall:
            if os.path.exists(pa): z = np.load(pa); s.Rs, s.Ts, s.C0, s.IM = z['R'], z['T'], z['C'], z['I']
            else: s._fall(); q = pa + f'.{os.getpid()}.npz'; np.savez(q, R=s.Rs, T=s.Ts, C=s.C0, I=s.IM); os.replace(q, pa)
        # shadow map frame (orthographic, seen from the moon)
        l = s.L; s.dr = -l; s.ax = np.cross(l, [0, 0, 1.]); s.ax /= np.linalg.norm(s.ax); s.ay = np.cross(s.ax, l); s.n_sm = 1800; s.s_sm = s.n_sm / 150.; ctr = np.array([34., -30., 6.]); s.o_sm = ctr - s.ax * 75. - s.ay * 75. + l * 220.
        s._sm = None; s._gb = {}

    def _rocks(s, PN, PD, C):
        m = max(len(a) for a in PN); n = len(C); s.PN = np.zeros((n, m, 3)); s.PD = np.zeros((n, m)); s.CN = np.array([len(a) for a in PN]); s.C = np.array(C, float)
        for i in range(n): s.PN[i, :len(PN[i])] = PN[i]; s.PD[i, :len(PD[i])] = PD[i]
        s.RAD = np.array([np.max(np.abs(PD[i])) * 2.1 for i in range(n)])

    # ------------------------------------------------ physics
    def _fall(s, nfr=150, dt=1 / 24.):
        g = 1.42; xs, ys, zs = np.meshgrid(np.arange(.6, 21.9, g), np.arange(.8, 7.6, g), np.arange(.7, 30., g), indexing='ij'); P = np.c_[xs.ravel(), ys.ravel(), zs.ravel()]
        d = np.array([sd_let(s.T['E'], *p, 1) for p in P]); P = P[d < -.3]; pc = np.array([np.argmin([sd_cell(s.SD, i, *p, 77) for i in range(len(s.SD))]) for p in P]); np_ = len(s.SD)
        up = [3, 4, 5, 6, 7, 8, 9, 10, 11]; base = [0, 1, 2]
        ev = [(0., [base, up]), (1.05, [base, [6], [3, 4, 5, 7, 8, 9, 10, 11]]), (1.35, [base, [6], [10, 11], [3, 4, 5, 7, 8, 9]]), (1.75, [[0], [1, 2], [6], [10], [11], [3, 4, 5], [7, 8, 9]]), (2.15, [[0], [1], [2], [6], [10], [11], [3], [4, 5], [7], [8, 9]]), (2.5, [[i] for i in range(12)])]
        te = np.array([e[0] for e in ev]); cl = np.zeros((len(ev), np_), np.int64)
        for k, (_, G) in enumerate(ev):
            for q in G: cl[k, q] = q[0]
        free = np.zeros(np_); free[base] = 1.75
        Rs = np.zeros((nfr, np_, 3, 3)); Ts = np.zeros((nfr, np_, 3)); IM = np.zeros((nfr, np_))
        s.C0 = _sim(P.copy(), pc, np_, te, cl, free, 9.81, np.array([-5.0, -2.6]), np.array([.1, 1.3]), dt, 12, nfr, s.Hm, .70, Rs, Ts, IM); s.Rs, s.Ts, s.IM = Rs, Ts, IM; s.npart = len(P)

    def clock(s, t):
        """film frame -> seconds of the fall (slow groan, then it goes)"""
        a, b = s.T_CRACK + 10, s.T_FALL
        return float(np.interp(t, [a, b, b + 8, b + 17, b + 60], [0., 1.05, 2.3, 4.1, 6.2]))

    def pose(s, t):
        q = np.clip(s.clock(t) * 24., 0, len(s.Rs) - 1.001); i = int(q); u = q - i; R = s.Rs[i] * (1 - u) + s.Rs[i + 1] * u; T = s.Ts[i] * (1 - u) + s.Ts[i + 1] * u; out = []
        for p in range(len(R)): U, _, Vt = np.linalg.svd(R[p]); out.append(U @ Vt)
        return np.array(out), T

    # ------------------------------------------------ camera
    def camera(s, push=0., eye=None, target=None, focal=2080., roll=0.):
        e = np.array([-24., -86., 3.5]) if eye is None else np.asarray(eye, float); tg = np.array([25.5, -6., 10.6]) if target is None else np.asarray(target, float)
        d = tg - e; return Camera3D(e + d * push, tg + d * push, focal, roll=roll)

    @staticmethod
    def _rays(cam):
        R = cam.R; D0 = R[2] - R[0] * cam.cx / cam.f - R[1] * cam.cy / cam.f; z = np.zeros(3)
        return (cam.eye.astype(float), z, z.copy(), D0 + .5 * (R[0] + R[1]) / cam.f, R[0] / cam.f, R[1] / cam.f)

    def _box(s, cam_rays, lo, hi, R, tr, piv, ox, w, h, project):
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float) + [ox, 0, 0]; c = (c - piv) @ R.T + piv + tr; q = project(c)
        x0, y0 = np.floor(q.min(0) - 2).astype(int); x1, y1 = np.ceil(q.max(0) + 2).astype(int); return max(0, x0), min(w, x1), max(0, y0), min(h, y1)

    def _draw(s, rays, project, w, h, t, parts, Z, N, PL, ID, FID=None, far=2000., small_only=False, ground_=True):
        I3 = np.eye(3); z3 = np.zeros(3); SD = s.SD.astype(float)
        for c, oid in (('G', 2), ('O', 3)):
            if c in parts:
                lo, hi = np.array([-.3, -.3, -1.]), np.array([EGO[c]['w'] * K + .3, DEP + .3, 30.3]); b = s._box(rays, lo, hi, I3, z3, z3, X0[c], w, h, project); trace(s.T[c], SD, -1, ord(c), lo, hi, I3, z3, z3, X0[c], *rays, *b, Z, N, PL, ID, oid, 90, far)
        if 'E' in parts:
            if t is None or s.clock(t) <= 0: lo, hi = np.array([-.3, -.3, -1.]), np.array([22.3, DEP + .3, 30.3]); b = s._box(rays, lo, hi, I3, z3, z3, 0., w, h, project); trace(s.T['E'], SD, -1, 1, lo, hi, I3, z3, z3, 0., *rays, *b, Z, N, PL, ID, 1, 90, far)
            else:
                R, T = s.pose(t)
                for p in range(len(SD)):
                    if not hasattr(s, '_pb'): s._pb = s._piece_boxes()
                    lo, hi = s._pb[p]; b = s._box(rays, lo, hi, R[p], T[p], s.C0[p], 0., w, h, project)
                    if b[1] > b[0] and b[3] > b[2]: trace(s.T['E'], SD, p, 1, lo, hi, R[p], T[p], s.C0[p], 0., *rays, *b, Z, N, PL, ID, 10 + p, 110, far)

    def _piece_boxes(s):
        x, y, z = np.meshgrid(np.arange(-.2, 22.3, .5), np.arange(0, 8.1, 1.), np.arange(0, 30.3, .5), indexing='ij'); P = np.c_[x.ravel(), y.ravel(), z.ravel()]; out = []
        ins = np.array([sd_let(s.T['E'], *p, 1) < .4 for p in P]); P = P[ins]
        for i in range(len(s.SD)):
            c = np.array([sd_cell(s.SD, i, *p, 77) < .5 for p in P]); q = P[c]; out.append((q.min(0) - .7, q.max(0) + .7))
        return out

    def _hull(s, rays, project, scale, w, h, Z, N, ID, FID, sel, oid):
        if len(sel) == 0: return
        q = project(s.C[sel]); rad = s.RAD[sel] * scale(s.C[sel]); b = np.c_[q[:, 0] - rad, q[:, 0] + rad, q[:, 1] - rad, q[:, 1] + rad]; b = np.clip(np.nan_to_num(b, nan=0, posinf=1e5, neginf=-1e5), -10, 1e5); b[:, :2] = np.clip(b[:, :2], 0, w); b[:, 2:] = np.clip(b[:, 2:], 0, h)
        hulls(s.PN[sel], s.PD[sel], s.CN[sel], s.C[sel], s.RAD[sel], b.astype(np.int64), *rays, Z, N, ID, FID, oid)

    def shadow_map(s, t):
        n = s.n_sm; px = 1. / s.s_sm; rays = (s.o_sm + .5 * px * (s.ax + s.ay), s.ax * px, s.ay * px, s.dr.copy(), np.zeros(3), np.zeros(3)); proj = lambda P: np.c_[(np.asarray(P) - s.o_sm) @ s.ax * s.s_sm, (np.asarray(P) - s.o_sm) @ s.ay * s.s_sm]
        mk = lambda: (np.full((n, n), 1e6, f32), np.zeros((n, n, 3), f32), np.zeros((n, n, 3), f32), np.zeros((n, n), np.int32), np.zeros((n, n), np.int32))
        if s._sm is None:
            Z, N, PL, ID, F = mk(); s._draw(rays, proj, n, n, None, 'GO', Z, N, PL, ID, far=1e5); s._hull(rays, proj, lambda P: np.full(len(P), s.s_sm), n, n, Z, N, ID, F, np.arange(s.n_small), 5); s._sm = Z
        Z = s._sm.copy()
        if 'E' in s.parts: _, N, PL, ID, F = mk(); s._draw(rays, proj, n, n, t, 'E', Z, N, PL, ID, far=1e5)
        return Z

    def gbuf(s, cam, t, static=None):
        rays = s._rays(cam); h, w = H, W; proj = lambda P: cam.project(P)[0]; Z = np.full((h, w), 1e6, f32); N = np.zeros((h, w, 3), f32); PL = np.zeros((h, w, 3), f32); ID = np.zeros((h, w), np.int32); F = np.zeros((h, w), np.int32)
        s._draw(rays, proj, w, h, t, s.parts, Z, N, PL, ID)
        front = cam.cam(s.C)[:, 2] > 2.; sel = np.nonzero(front)[0]; s._hull(rays, proj, lambda P: cam.scale(P), w, h, Z, N, ID, F, sel[sel < s.n_small], 5); s._hull(rays, proj, lambda P: cam.scale(P), w, h, Z, N, ID, F, sel[sel >= s.n_small], 6)
        hy = int(max(0, proj(np.array([[cam.eye[0] + cam.R[2, 0] * 1e5, cam.eye[1] + cam.R[2, 1] * 1e5, cam.eye[2]]]))[0, 1] - 14)); ground(s.Hm, *rays, hy, Z, N, ID, 4, 1500.)
        x, y = grid(h, w); D = (rays[3] + x[..., None] * rays[4] + y[..., None] * rays[5]).astype(f32); D /= np.linalg.norm(D, axis=-1, keepdims=True); P = cam.eye.astype(f32) + D * np.minimum(Z, 1e5)[..., None]
        return dict(Z=Z, N=N, PL=PL, ID=ID, F=F, P=P, D=D, hy=hy + 14)

    # ------------------------------------------------ painting
    PAIRS = {(0, 3): (34.0, 8.0, 6.0), (0, 4): (34.6, 8.2, 7.2), (1, 4): (34.2, 8.4, 6.6), (3, 4): (39.5, 5.0, 8.4), (3, 5): (42.5, 3.6, 12.), (4, 5): (41.0, 5.2, 13.), (4, 6): (44.0, 9.4, 12.6), (5, 7): (46.0, 5.0, 16.6), (4, 7): (45.0, 7.4, 15.6), (6, 7): (46.0, 9.0, 16.),
             (5, 8): (49.0, 3.6, 19.4), (7, 8): (48.0, 6.6, 20.6), (7, 9): (51.0, 8., 23.), (8, 9): (52.0, 5.6, 25.), (7, 10): (50.0, 8.4, 23.4), (9, 10): (50.5, 9.2, 24.6), (10, 11): (54.0, 16.4, 24.), (0, 1): (47., 8.6, 5.4), (1, 2): (55., 15.6, 5.)}

    def cracks(s, g, t):
        """painted cracks on the E at frame t: (dark 0..1, wide 0..1)"""
        M = (g['ID'] == 1) | ((g['ID'] >= 10) & (g['ID'] < 30)); h, w = M.shape; I = np.zeros((h, w), np.int32); J = np.zeros((h, w), np.int32); B = np.full((h, w), 9., f32)
        if t < s.T_CRACK or not M.any(): return np.zeros((h, w), f32)
        cells(g['PL'], M, s.SD, I, J, B); a, b = np.minimum(I, J), np.maximum(I, J); out = np.zeros((h, w), f32); prog = np.zeros((h, w), f32)
        for (i, j), (t0, x0, z0) in s.PAIRS.items():
            if t < t0: continue
            m = M & (a == i) & (b == j)
            if not m.any(): continue
            d = np.hypot(g['PL'][..., 0] - x0, g['PL'][..., 2] - z0) + .10 * np.abs(g['PL'][..., 1]); u = (t + .9 - t0) * 2.6 - d; main = (i, j) in ((0, 3), (0, 4), (1, 4)); wd = np.clip(u / 7., 0, 1) * (.07 + .16 * min(1., (t - t0) / 14.)) * (1.9 if main else 1.2); out = np.maximum(out, m * np.clip((wd - B) / .05 + .5, 0, 1) * (u > 0)); prog = np.where(m, np.clip(u / 10., 0, 1), prog)
        from .cam import voronoi_crevice
        X, Zz = g['PL'][..., 0] + .25 * g['PL'][..., 1], g['PL'][..., 2]; cv1, id1 = voronoi_crevice(X + .5 * np.sin(Zz * 1.3), Zz + .5 * np.sin(X * 1.1), (1.5, 1.9), 9, .95); hair = smooth(.055, .02, cv1) * (id1 > .42) * (B < .35 + 1.5 * prog * id1) * (prog > .05) * M        # hairlines that run off the big breaks
        return np.maximum(out, hair * .75).astype(f32)

    def _tex(s):
        """worn paint per letter, 1 px per em"""
        if not hasattr(s, '_paint'):
            from .letters import ego_mask
            s._paint = {c: ego_mask((0., 0., 1000.), letters=c, h=1000, w=880, wear=.66, seed=5 + k, chips=0., ss=1) for k, c in enumerate('EGO')}
        return s._paint

    def backdrop(s, cam, seed=4):
        """night sky, moon, torn cloud, glow: full-frame RGB for this camera"""
        x, y = grid(); P = s.PAL; mp, mz = cam.project(cam.eye + s.L * 1e4); mp = np.asarray(mp, float).ravel(); cz = cam.cam(cam.eye + s.L * 1e4); mp = mp if cz[2] > 1. else np.array([W / 2 + math.copysign(5200., cz[0]), -900.]); hy = float(cam.project(cam.eye + np.array([cam.R[2, 0], cam.R[2, 1], 0.]) * 1e4)[0].ravel()[1])
        from .paint import ramp
        v = np.clip((hy - y) / 900., -.1, 1.1) + .10 * smooth(500, 1900, x) + .04 * smooth(700, 0, y) * 0; sky = ramp(v, [(0., P['haze']), (.07, '#493a8a'), (.20, P['low']), (.42, P['mid']), (.72, P['sky']), (1., P['top'])])
        d = np.hypot(x - mp[0], y - mp[1]); sky = sky + (np.exp(-d / 420.) * .15 + np.exp(-d / 150.) * .26)[..., None] * rgb('#8f72ee')
        if not hasattr(s, '_cl'): s._cl = {}
        if seed not in s._cl:                                                   # two decks of torn bars, drawn long in the wind
            n = fbm(H, W, (170, 74, 30, 12, 5), (1, .70, .46, .30, .18), seed, aspect=6.0); q = fbm(H, W, (120, 50, 20, 9, 4), (1, .70, .5, .34, .2), seed + 7, aspect=4.2); s._cl[seed] = (n, q)
        n, q = s._cl[seed]; hgt = smooth(hy + 6, hy - 140, y)
        def deck(c, th, dark, mid, lit, hot, k):
            m = smooth(th, th + .10, c); e = smooth(th + .02, th + .10, c) * smooth(th + .62, th + .10, c)                                          # thin edge band
            up_ = np.roll(m, 5, 0) ; side = np.clip(m - cv2.warpAffine(m, np.float32([[1, 0, -5 if mp[0] < 960 else 5], [0, 1, 4]]), (W, H), borderMode=cv2.BORDER_REPLICATE), 0, 1); near = np.exp(-d / 460.); wide = np.exp(-d / 1050.)
            col = mix(rgb(dark), rgb(mid), smooth(th + .1, th + 1.5, c)[..., None] * .8); col = mix(col, rgb(lit), (e * (.16 * wide + .62 * near))[..., None]); col = mix(col, rgb(hot), (side * np.clip(near * 1.35 - .12, 0, 1))[..., None]); return m * hgt * k, col
        for c, th, pl, k in ((n + .30 * np.sin(y / 88. + 1.), .55, ('#15112f', '#0c0a1e', '#4a3b8e', '#b39bf5'), .85), (q - .2 + .35 * np.sin(y / 61.), .95, ('#0c0a1d', '#080714', '#5c4ba8', '#efe6ff'), .95)):
            m, col = deck(c, th, *pl, k)
            m = m * (1 - .55 * smooth(130, 40, d)); sky = sky * (1 - m[..., None]) + col * m[..., None]
        sky = sky + (np.exp(-np.maximum(0, hy - y) / 150.) * (.08 + .12 * np.exp(-((x - mp[0] - 460) / 700.) ** 2)))[..., None] * rgb('#8c78dc')
        fx = np.exp(-((x - 30) / 430.) ** 2) * np.exp(-np.maximum(0, hy - y) / 90.) + .55 * np.exp(-((x - 1830) / 200.) ** 2) * np.exp(-np.maximum(0, hy - y) / 60.); sky = sky + fx[..., None] * rgb('#ff6426') * .20
        return sky.astype(f32), mp, hy

    MESAS = ((0, -11., 5.2, 4.6), (0, 4.4, 6.0, 3.6), (0, 17.5, 7.5, 5.2), (0, 31., 5.0, 3.4), (0, 43., 6.5, 5.0), (0, 56., 6., 4.2),
             (1, -3.5, 2.6, 7.6), (1, 12.4, 1.5, 5.6), (1, 23.8, 2.2, 9.0), (1, 37.4, 1.3, 6.2), (1, 48.6, 2.4, 7.2), (1, 62., 3., 8.),
             (2, 7.6, 1.0, 6.4), (2, 8.9, 2.4, 3.0), (2, 35.2, 2.0, 3.4), (2, 52.6, 1.4, 9.8), (2, 54.9, 3.0, 4.2),
             (0, 74., 8., 4.2), (1, 82., 2.4, 7.6), (0, 93., 9., 3.4), (1, 99., 3.2, 5.6), (2, 88.4, 1.3, 5.2), (2, 105., 1.8, 7.4), (0, 112., 8., 4.6), (1, 120., 3., 6.), (0, -22., 8., 4.4), (1, -29., 2.4, 7.0), (0, -38., 8., 3.6), (1, -48., 3.2, 5.6), (2, -34.5, 1.5, 5.0), (2, -43., 1.2, 8.2), (0, -58., 9., 4.2))             # (layer, azimuth, width, height) degrees: buttes and stone needles of the coast

    def mesas(s, img, cam, mp, hy):
        """far rocks, drawn: ledges, talus, hard shadow planes, strata, an edge of moonlight"""
        from .ink import Cel
        for lay, (base, shade, edge, haze) in enumerate((('#3d3278', '#33296a', '#8470cc', .0), ('#2b225a', '#221a4b', '#8f7bd8', 0.), ('#1c163f', '#140f30', '#a48fea', 0.))):
            c = Cel(W, H)
            for k, (L, az, wd, ht) in enumerate(s.MESAS):
                if L != lay: continue
                r = np.random.default_rng(900 + k); a = math.radians(az); q, zc = cam.project(cam.eye + np.array([math.sin(a), math.cos(a), 0.]) * 1e4); zc = float(np.ravel(zc)[0])
                if zc <= 0: continue
                x0 = float(np.ravel(q)[0]); pw = cam.f * math.radians(wd); ph = cam.f * math.radians(ht); lit_left = mp[0] < x0
                def side(sg):                                                  # base to top
                    P = []; u = .0; xx = pw * .5 * r.uniform(1.25, 1.6)
                    P.append((xx, 0.)); xx = pw * .5 * r.uniform(.95, 1.1); u = r.uniform(.16, .30); P.append((xx, u))
                    while u < .94:
                        du = r.uniform(.05, .20); u = min(1., u + du); xx -= pw * r.uniform(.0, .022); P.append((xx + r.normal(0, .010) * pw, u))
                        if r.random() < .62 and u < .95: s_ = pw * r.uniform(.015, .07) * (1 if r.random() < .75 else -1.2); xx -= s_; P.append((xx, u + r.uniform(0, .012)))
                    return [(x0 + sg * p[0], hy + 4 - p[1] * ph) for p in P]
                l, rr = side(-1), side(1); top = []; xa, xb = l[-1][0], rr[-1][0]; n = int(r.integers(3, 7))
                for i in range(1, n): top.append((xa + (xb - xa) * i / n + r.normal(0, pw * .02), hy + 4 - ph * r.uniform(.96, 1.06)))
                poly = np.array(l + top + rr[::-1]); c.fill(poly, base); c.clip(poly)
                # shadow plane: angular, on the side away from the moon
                ls = l if lit_left else rr; sg = 1 if lit_left else -1; sp = []
                for (px, py) in ls: sp.append((px + sg * pw * (r.uniform(.22, .42) if len(sp) % 2 else r.uniform(.30, .50)), py + r.normal(0, ph * .012)))
                far = [(x0 + sg * pw * 2, hy + 10)]; c.fill(np.array(sp + [(x0 + sg * pw * 2, hy - ph * 1.3)] + far), shade)
                for _ in range(int(5 + pw / 26)):                               # clefts
                    px = x0 + r.uniform(-.5, .5) * pw; py = hy - ph * r.uniform(.25, .98); c.line(np.array([(px, py), (px + r.normal(0, 3), py + ph * r.uniform(.06, .2)), (px + r.normal(0, 5), py + ph * r.uniform(.2, .4))]), r.uniform(1.2, 4.) * (.6 + .3 * lay), shade, taper=(.1, .8), seed=int(r.integers(1 << 30)))
                for _ in range(int(4 + ph / 40)):                               # strata
                    py = hy - ph * r.uniform(.18, .95); xs = np.linspace(x0 - pw * .6, x0 + pw * .6, 12); a0 = r.integers(0, 6); c.line(np.c_[xs, py + r.normal(0, 1.2, 12)][a0:a0 + r.integers(4, 7)], r.uniform(.8, 2.0), shade, taper=(.3, .5), seed=int(r.integers(1 << 30)), alpha=.8)
                E_ = np.array(ls); m = E_ + np.c_[sg * np.abs(r.normal(2.2 + pw * .012, 1.2, len(E_))), np.zeros(len(E_))]; c.fill(np.vstack([E_[1:], m[1:][::-1]]), edge, .85)
                tp = np.array([ls[-1]] + (top if lit_left else top[::-1])); c.line(tp, 2.2, edge, taper=(.05, .6), seed=k, alpha=.9); c.unclip()
            q = c.rgba(); x, y = grid(); fade = smooth(hy + 4, hy - 120, y)[..., None]; img = img * (1 - q[..., 3:4]) + (q[..., :3] * (.55 + .45 * fade) + img * q[..., 3:4] * (.45 * (1 - fade)))
        return img

    def _brush(s, seed=8):
        """one sheet of gouache: for every pixel, where the brush mark that covers it took its paint.  (sx, sy, bristle)"""
        key = '_br%d' % seed
        if not hasattr(s, key):
            from .paint import Canvas, arcs
            r = np.random.default_rng(seed); out = []
            for hw, st, asp, ang in ((7.5, 13., 5.0, .0), (3.6, 6.2, 5.0, .0)):
                gy, gx = np.mgrid[0:H:st, 0:W:st * 1.5].astype(f32); px = (gx + r.uniform(-.7, .7, gx.shape) * st * 1.5).ravel(); py = (gy + r.uniform(-.7, .7, gx.shape) * st).ravel(); o = r.permutation(px.size); px, py = px[o], py[o]; n = px.size
                fl = fbm(H, W, (240, 90), (1, .5), seed + 3); an = fl[np.clip(py, 0, H - 1).astype(int), np.clip(px, 0, W - 1).astype(int)] * .22 + r.normal(0, .16, n) + ang
                cv_ = Canvas(H, W, img=np.dstack([*[g_ / q for g_, q in zip(grid(), (W, H))], np.full((H, W), .5, f32)])); col = np.c_[px / W, py / H, np.clip(.5 + r.normal(0, .18, n), 0, 1)].astype(f32)
                cv_.strokes(arcs(px, py, an, hw * asp * r.uniform(.6, 1.5, n), r.normal(0, .35, n)), hw * r.uniform(.55, 1.2, n), col, op=1., dry=.0, seed=seed, bristle=.0); out.append((np.clip(cv_.img[..., 0] * W, 0, W - 1).astype(f32), np.clip(cv_.img[..., 1] * H, 0, H - 1).astype(f32), cv_.img[..., 2] - .5))
            setattr(s, key, out)
        return getattr(s, key)

    def gouache(s, img, lab, amt=1.0, layer=0):
        """lay the picture as brush marks that never cross a label edge"""
        sx, sy, b = s._brush()[layer]; q = cv2.remap(img, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101); same = lab[np.clip(sy, 0, H - 1).astype(int), np.clip(sx, 0, W - 1).astype(int)] == lab
        k = (cv2.GaussianBlur(same.astype(f32), (0, 0), .7) * same * amt)[..., None]; q = q * (1 + .16 * b[..., None]); return img * (1 - k) + q * k

    def render(s, cam, t=0., dust=True, lines=True, g=None, seed=4, flames=True, puff=None, glow=0., glow_e=0.):
        P = s.PAL; g = g if g is not None else s.gbuf(cam, t); ID, N, Z, Pw, PL = g['ID'], g['N'], g['Z'], g['P'], g['PL']; h, w = ID.shape; x, y = grid()
        sky, mp, hy = s.backdrop(cam, seed); rays = s._rays(cam)
        sm = s.shadow_map(t); sh = np.zeros((h, w), f32); solid = ID > 0; shadow_look(Pw, sm, s.o_sm, s.ax, s.ay, s.dr, s.s_sm, .30, sh, solid); sh = cv2.GaussianBlur(sh, (0, 0), .8)
        dif = N @ s.L.astype(f32); E = (ID == 1) | ((ID >= 10) & (ID < 30)); G_, O_ = ID == 2, ID == 3; let = E | G_ | O_; rk = ID == 5; mesa = ID == 6; gr = ID == 4
        key = smooth(.02, .06, dif) * (1 - sh) * solid
        fire = np.zeros((h, w), f32)
        for X, Y, R in s.FIRES:
            vx, vy, vz = X - Pw[..., 0], Y - Pw[..., 1], R * 1.1 + .6 - Pw[..., 2]; q2 = vx * vx + vy * vy + vz * vz; k_ = np.clip((N[..., 0] * vx + N[..., 1] * vy + N[..., 2] * vz) / np.sqrt(q2 + 1e-3), 0, 1); fire += k_ * R * R * 1.7 / (q2 * .16 + 2.0)
        fire = np.clip(fire * (Z < 260) * (.94 + .06 * math.sin(t * 2.1)), 0, 1).astype(f32)
        up = N[..., 2] * .5 + .5; dist = np.where(solid, Z, 1200.); zz_ = np.maximum(0, Pw[..., 2])
        if not hasattr(s, '_nt'): s._nt = fbm(1024, 1024, (64, 24, 9, 4), (1, .7, .5, .35), 31)
        fold = lambda q: (lambda m: np.where(m > 1023, 2047 - m, m))(np.floor(q).astype(np.int64) & 2047); tex = lambda u, v, k: s._nt[fold(v * k + 511), fold(u * k + 300)]
        u_ = np.where(np.abs(N[..., 0]) > np.abs(N[..., 1]), PL[..., 1], PL[..., 0]); v_ = PL[..., 2]
        n2 = np.where(let, tex(u_, v_, 22.), tex(Pw[..., 0], Pw[..., 1] + 1.7 * Pw[..., 2], np.where(Z < 40, 26., 9.)))
        # fill: sky from above, light thrown back by the lit waste from below and in front, dying with height
        fill = .40 * up + .34 * np.clip(-N[..., 1], 0, 1) * (.28 + .72 * np.exp(-zz_ / 15.)) + .20 * np.clip(-N[..., 0], 0, 1) + .05 * n2
        step = smooth(.17, .20, fill) * .34 + smooth(.32, .35, fill) * .33 + smooth(.50, .54, fill) * .33
        out = sky.copy(); lab = np.where(solid, ID, 0).astype(np.int32); lab = np.where(rk | mesa, 1000 + g['F'], lab)
        # ---- letters
        col = mix(rgb('#0d0b1e'), rgb('#2b2353'), step[..., None]); pal = s._tex(); paint = np.zeros((h, w), f32)
        for c, m in (('E', E), ('G', G_), ('O', O_)):
            if not m.any(): continue
            pm = pal[c]; q_ = cv2.remap(pm, np.clip(PL[..., 0] / K, 0, pm.shape[1] - 1.01).astype(f32), np.clip(1000. - PL[..., 2] / K, 0, 998.9).astype(f32), cv2.INTER_LINEAR); paint = np.where(m, q_, paint)
        pf = smooth(.62, .34, PL[..., 1]); rim_ = smooth(-.55, -.86, np.where(ID >= 10, -1., N[..., 1])); paint = paint * pf * rim_
        grad = np.clip(.42 + .58 * np.exp(-zz_ / 18.), 0, 1); pc = mix(rgb('#1d183c'), rgb('#a69bd6'), (grad * (.25 + .75 * step))[..., None]); col = mix(col, pc, (paint * .90)[..., None])
        raw = np.zeros((h, w), f32)
        if (ID >= 10).any():                                                # fresh breaks: paler, drier stone
            I = np.zeros((h, w), np.int32); J = I.copy(); B = np.full((h, w), 9., f32); cells(PL, (ID >= 10) & (ID < 30), s.SD, I, J, B); raw = smooth(.16, .05, B) * (ID >= 10) * smooth(.15, .5, PL[..., 1]); col = mix(col, mix(rgb('#1a1430'), rgb('#54467f'), step[..., None]), raw[..., None])
        streak = tex(u_ * 3.2, v_ * .28, 14.); col = col * (1 - .16 * smooth(.2, 1.2, streak))[..., None] * (1 + .10 * np.clip(tex(u_, v_, 7.), -1.5, 1.5))[..., None]
        ck = s.cracks(g, t) if t >= s.T_CRACK else np.zeros((h, w), f32); col = mix(col, rgb('#040308'), ck[..., None] * .97); cl = np.clip(np.roll(ck, (2, 2), (0, 1)) - ck, 0, 1); col = col + cl[..., None] * rgb('#4c3f80') * .6
        if glow > 0 or glow_e > 0:                                          # dust lit by the moon throws its light back on G and O: the paint wakes
            gm = ((G_ | O_) * glow + E * glow_e)[..., None]; w_ = mix(rgb('#3d3370'), rgb('#faf6ff') * 1.06, (np.clip(paint * 1.15, 0, 1) * (.50 + .50 * step) * (.80 + .20 * np.clip(1 - zz_ / 40., 0, 1)))[..., None]); w_ = w_ * (1 - .12 * smooth(.2, 1.2, tex(u_ * 3.2, v_ * .28, 14.)))[..., None]; col = col * (1 - gm * .94) + w_ * gm * .94
        kc = mix(rgb('#8f78dd'), rgb('#f2eaff'), smooth(.25, .75, dif)[..., None]); col = mix(col, kc, (key * (.92 - .16 * smooth(.4, 1.2, n2)))[..., None])
        fr = smooth(.12, .18, fire + .02 * n2) * .5 + smooth(.32, .38, fire + .02 * n2) * .5; warm = rgb('#ff7d33') * (.35 + .65 * col / (col.max(-1, keepdims=True) + 1e-3)); col = col + (fr * .20 * (1 - key))[..., None] * warm
        out = np.where(let[..., None], col, out)
        # ---- rocks and mesas
        rs = .62 * up + .22 * np.clip(-N[..., 0], 0, 1) + .10 * np.clip(-N[..., 1], 0, 1) + .035 * n2; st3 = smooth(.36, .385, rs) * .40 + smooth(.56, .585, rs) * .60; far_ = smooth(10, 80, Z)
        c2 = mix(rgb('#05040b'), mix(rgb('#1f1940'), rgb('#3f3474'), far_[..., None]), st3[..., None]); c2 = mix(c2, mix(rgb('#6a59b6'), rgb('#d6caff'), smooth(.3, .85, dif)[..., None]), (key * (.95 - .35 * smooth(.2, 1.2, n2)))[..., None]); f2 = smooth(.22, .27, fire + .03 * n2); c2 = c2 + (f2 * .30 * (1 - key))[..., None] * rgb('#e0622a')
        fe = (np.roll(g['F'], (1, 0), (0, 1)) != g['F']) | (np.roll(g['F'], (0, 1), (0, 1)) != g['F']); fe = fe & rk & (up > .55); fe = cv2.GaussianBlur(fe.astype(f32), (0, 0), .8); c2 = c2 + (fe * np.clip(40. / np.maximum(Z, 10), .25, 1.3) * smooth(-.6, .6, n2))[..., None] * rgb('#7d6cc4') * .42
        out = np.where(rk[..., None], c2, out)
        st_ = tex(Pw[..., 0] * .03 + Pw[..., 1] * .03, Pw[..., 2] * 1.6, 6.); c3 = mix(rgb('#150f33'), rgb('#3a2f70'), (step * .7 + .16 * smooth(.2, .5, st_))[..., None]); out = np.where(mesa[..., None], c3, out)
        # ---- ground: hard dry plates; light lies on it only where the letters let it through
        if gr.any():
            yy, xx = np.nonzero(gr); gx_, gy_ = Pw[yy, xx, 0], Pw[yy, xx, 1]; zz = Z[yy, xx]
            from .cam import voronoi_crevice
            wxx, wyy = gx_ + 1.6 * tex(gx_, gy_, 8.) , gy_ + 1.6 * tex(gx_ + 40, gy_ + 9, 8.); cv1, id1 = voronoi_crevice(wxx, wyy, (7.5, 5.0), 3, .95); cv2_, id2 = voronoi_crevice(wxx * 1.0, wyy, 1.5, 4, .95)
            wl = np.clip(zz / 1700., .004, .2); ck1 = smooth(wl * 1.5 + .022, wl * .5, cv1) * np.exp(-zz / 120.) * smooth(-.5, .6, tex(gx_, gy_, 5.)); ck2 = smooth(wl * 1.4 + .035, wl * .7, cv2_) * np.exp(-zz / 38.) * smooth(.2, .9, tex(gx_ + 11, gy_, 7.))
            mott = tex(gx_ * .45, gy_ * .16, 10.); fine = tex(gx_, gy_, 34.); near = smooth(60, 12, zz)
            lit_ = key[yy, xx] * smooth(.05, .17, dif[yy, xx] + .035 * mott); dk = mix(rgb('#080614'), rgb('#251d49'), (smooth(-.7, 1.2, mott + .9 * (id1 - .5)) * (.35 + .65 * smooth(14, 80, zz)) + .06 * fine)[:, None]); lt = mix(rgb('#3f3478'), rgb('#927ed4'), np.clip(smooth(-1.0, 1.3, mott * .6 + 1.5 * (id1 - .5) + .5 * (id2 - .5) * np.exp(-zz / 40.)) * .8 + smooth(20, 85, zz) * .30 - .28 * near + .05 * fine, 0, 1)[:, None]); crest = smooth(.235, .30, dif[yy, xx] + .02 * fine) * lit_; lt = mix(lt, rgb('#c9bbf2'), (crest * .55)[:, None])
            lip = np.roll(ck1, 1) ; c4 = mix(dk, lt, lit_[:, None]); c4 = c4 + (smooth(.1, .5, ck1) * smooth(.75, .25, ck1) * (1 - lit_) * .10)[:, None] * rgb('#7d6cc4'); c4 = c4 * (1 - (np.maximum(ck1 * .80, ck2 * .55))[:, None]); c4 = c4 * (1 + .07 * np.clip(fine, -2, 2) * np.exp(-zz / 50.))[:, None]
            c4 = c4 + (np.clip(fire[yy, xx] * 1.6, 0, 1) * .30 * (1 - .5 * lit_))[:, None] * rgb('#ff6f2c'); out[yy, xx] = c4
        sky_m = ~solid; o0 = s.gouache(s.gouache(sky, lab * 0, .45, 0), lab * 0, .85, 1); o0 = s.gouache(s.mesas(o0, cam, mp, hy), lab * 0, .22, 1); o0 = np.where((y > hy - 1)[..., None], mix(rgb('#2d245a'), rgb('#4a3b8a'), np.clip(1 - np.abs(x - mp[0] - 200) / 900., 0, 1)[..., None]), o0); out = np.where(sky_m[..., None], o0, out); o1 = s.gouache(out, np.where(gr, 50 + (key > .5) * 1, lab), .48, 1); out = np.where((gr | rk)[..., None], o1, out)
        d_ = np.hypot(x - mp[0], y - mp[1]); R = 46.; disc = smooth(R + .8, R - .8, d_) * sky_m; bar = smooth(.95, 1.15, s._cl[seed][1] - .2 + .35 * np.sin(y / 61.)) * .8 * smooth(-8, 26, y - mp[1]); out = out * (1 - (disc * (1 - bar))[..., None]) + rgb(P['moon']) * 1.3 * (disc * (1 - bar))[..., None]
        # ---- air: dust, thickest along the ground, lit from the moon's side, with the letters' shadows standing in it
        hz = np.exp(-zz_ / 22.); fog = (1 - np.exp(-np.maximum(0, dist - 40) / 520.)) * (.22 + .78 * np.where(solid, hz, 0.)) * solid
        side = smooth(1700, 0, np.abs(x - mp[0]))[..., None]; hc = mix(mix(rgb('#241c4e'), rgb('#42357f'), side), mix(rgb('#3a2e72'), rgb('#6f5cbc'), side), smooth(60, 600, dist)[..., None]); out = out * (1 - fog[..., None] * .88) + hc * fog[..., None] * .88
        if glow > 0: out = out * (1 - .34 * glow * (1 - np.where(G_ | O_, 1., np.where(E, glow_e, 0.))))[..., None]
        q = 4; sf = np.zeros((h // q, w // q), f32); shafts(*rays, Z, q, sm, s.o_sm, s.ax, s.ay, s.dr, s.s_sm, 48, 260., sf, 13.); sf = cv2.GaussianBlur(sf, (0, 0), 1.6); sf = cv2.resize(sf, (w, h), interpolation=cv2.INTER_LINEAR); ph = np.clip(g['D'] @ s.L.astype(f32), 0, 1) ** 8
        out = out + (sf / 60. * (.05 + .26 * ph + .20 * np.clip(g['D'] @ s.L.astype(f32), 0, 1) ** 3))[..., None] * rgb('#8468e6')
        # ---- ink: only accents.  letters get a broken pen line, near rocks a breath of one
        if lines:
            sd = lambda a, dy, dx: np.roll(a, (dy, dx), (0, 1)); zj = .03 * Z + .35; e1 = np.zeros((h, w), bool); e3 = np.zeros((h, w), bool)
            for dy, dx in ((0, 1), (1, 0)):
                o = sd(ID, dy, dx); dz = np.abs(Z - sd(Z, dy, dx)); e1 |= (let | ((o >= 1) & (o <= 3)) | (o >= 10)) & ((o != ID) | (dz > zj * 1.6)) & ~(rk | gr | np.isin(o, (4, 5))); nd = (N * sd(N, dy, dx)).sum(-1); e3 |= let & (o == ID) & (nd < .74) & (dz < zj)
            e1[:, 0] = e1[0] = False; nz = cv2.resize(s._nt[:270, :480], (w, h))
            def ink(m, w0): d = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5); return np.clip(w0 * (1 + .55 * nz) - d + .45, 0, 1)
            a = np.maximum(ink(e1, 1.25), ink(e3, .62) * .62) * np.exp(-dist / 700.); out = out * (1 - a[..., None] * .85) + rgb(P['line']) * a[..., None] * .85
        # rim: where a letter's edge faces the moon across open sky
        ux, uy = mp[0] - x, mp[1] - y; L_ = np.hypot(ux, uy) + 1; ux, uy = ux / L_, uy / L_; m = let.astype(f32); off = cv2.remap(((ID == 0) | mesa).astype(f32), (x + ux * 2.6).astype(f32), (y + uy * 2.6).astype(f32), cv2.INTER_LINEAR); rim = cv2.GaussianBlur(m * off, (0, 0), .6) * np.exp(-np.hypot(x - mp[0], y - mp[1]) / 1000.) * smooth(-.9, .9, nz + .8); out = out + rim[..., None] * rgb('#ece2ff') * 1.7
        out = fxaa(out.astype(f32)); g['sky'], g['moon'], g['hy'], g['sm'], g['key'] = sky, mp, hy, sm, key
        if flames: out = s.flames(out, g, cam, t)
        if dust: out = s.dust(out, g, cam, t)
        return out.astype(f32), g

    # ------------------------------------------------ small fires among the rubble, smoke
    FIRES = ((-12.4, -9., 1.5), (-15.5, -2., .9), (-7.2, -18., .7), (86., -4., 1.6), (-30., 10., 2.2), (30., 22., 1.6))

    def flames(s, img, g, cam, t):
        from .ink import Cel, spl
        c = Cel(W, H); k = int(t // 3); gl = np.zeros((H // 4, W // 4), f32); gx, gy = grid(H // 4, W // 4); zv = g['Z'] * (g['D'] @ cam.R[2].astype(f32)); occ = np.ones((H, W), f32)
        for i, (X, Y, R) in enumerate(s.FIRES):
            z0 = float(_hgt(s.Hm, X, Y)); q, zc = cam.project(np.array([X, Y, z0])); zc = float(np.ravel(zc)[0]); q = np.ravel(q); sc = cam.f / zc * R
            if zc < 2 or not (-200 < q[0] < W + 200): continue
            r = np.random.default_rng(k * 31 + i)
            for col, kk in (('#a8301c', 1.0), ('#ff7a2c', .70), ('#ffe0a0', .36)):
                for j in range(9):
                    rr = np.random.default_rng(k * 31 + i * 7 + j); u = (j - 4) / 4.; x0 = q[0] + u * sc * .62 + rr.normal(0, .05) * sc; hh = sc * (1.5 - 1.0 * abs(u) ** 1.3) * rr.uniform(.55, 1.15) * kk; ww = sc * .105 * (.7 + .6 * kk); sw = rr.normal(.18, .12) * hh
                    P_ = spl(np.array([(x0 - ww, q[1] + sc * .05), (x0 - ww * 1.15 - sw * .12, q[1] - hh * .30), (x0 - ww * .25 + sw * .30, q[1] - hh * .66), (x0 + sw * 1.1, q[1] - hh), (x0 + ww * .55 + sw * .35, q[1] - hh * .60), (x0 + ww * 1.15 - sw * .05, q[1] - hh * .26), (x0 + ww, q[1] + sc * .05)]), True, corners=(0, 3, 6), n=7); c.fill(P_, col)
            gl += np.exp(-np.hypot((gx * 4 - q[0]) / (sc * 4.2), (gy * 4 - q[1] + sc * .6) / (sc * 2.6)) ** 1.2) * (.9 + .1 * r.normal()) * (.5 + .5 * min(1, sc / 50.))
        L = c.rgba(); v = (zv > 70.).astype(f32)[..., None] * 0 + 1.; img = img + cv2.resize(gl, (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * rgb('#ff6a22') * .30
        hide = np.zeros((H, W), f32); a = L[..., 3:4]; near = ((g['ID'] == 5) & (zv < 72.)).astype(f32)[..., None]; a = a * (1 - near); return img * (1 - a) + L[..., :3] * (1 - near) * 1.15

    # ------------------------------------------------ dust and crumbs
    def puffs(s, t, seed=0):
        """lumps of dust alive at frame t: (n, 6) x y z radius age id.  Born where pieces strike and along the foot"""
        if not hasattr(s, '_em'):
            r = np.random.default_rng(91 + seed); em = []; inv = lambda c: float(np.interp(c, [0., 1.05, 2.3, 4.1, 6.2], [s.T_CRACK + 10, s.T_FALL, s.T_FALL + 8, s.T_FALL + 17, s.T_FALL + 60]))
            for fr in range(0, len(s.IM)):
                for p in range(s.IM.shape[1]):
                    v = s.IM[fr, p]
                    if v > .30 and r.random() < .8:
                        c = s.C0[p] + s.Ts[fr, p]; tt = inv(fr / 24.)
                        for _ in range(int(min(3, 1 + v * .5))): a = r.uniform(0, 2 * math.pi); sp = r.gamma(2., .9); em.append((tt + r.uniform(-.3, .8), c[0] + r.normal(0, 2.6), c[1] + r.normal(0, 2.6), r.uniform(.3, 1.2), math.cos(a) * sp, math.sin(a) * sp - .6, r.uniform(.2, 1.2), r.uniform(1.5, 3.2), r.uniform(22, 42)))
            for k in range(64): u = r.uniform(-4, 24); fr_ = r.random() < .72; em.append((s.T_FALL - 7 + r.gamma(2.2, 3.4), u, (-1.6 if fr_ else 9.4) + r.normal(0, .7), .5, (u - 10) * .07 + r.normal(0, .4), (-1 if fr_ else 1) * r.uniform(.3, 1.4), r.uniform(.1, .5), r.uniform(1.3, 2.8), r.uniform(20, 40)))
            for k in range(40): u = r.uniform(-12, 20); fr_ = r.random() < .8; em.append((s.T_FALL + 6 + r.uniform(0, 15), u, (-2.2 if fr_ else 9.) + r.normal(0, 1.6), r.uniform(.4, 2.), (u - 8) * .10 + r.normal(0, .5), (-1 if fr_ else 1) * r.uniform(.4, 2.2), r.uniform(.2, 1.0), r.uniform(1.8, 3.6), r.uniform(26, 46)))
            s._em = em
        out = []
        for k, (t0, X, Y, Zz, vx, vy, vz, R, life) in enumerate(s._em):
            a = t - t0
            if a < 0 or a > life: continue
            e = (1 - math.exp(-.16 * a)) / .16; u = a / life; rad = R * (.30 + 1.7 * (1 - math.exp(-.12 * a))); out.append((X + vx * e * .6, Y + vy * e * .6, max(rad * .35, Zz + vz * e * .5 + .012 * a), rad, u, k))
        return np.array(out, float).reshape(-1, 6)

    def dust(s, img, g, cam, t):
        """crumbs, then the dust of the fall as drawn lobes (phys lane's kit), in three depth slices"""
        img = s.trickle(img, g, cam, t); img = s.crumbs(img, g, cam, t); B = s.puffs(t)
        if len(B) == 0: return img
        from .phys_dust import lobes_layer, age_erode
        k = np.arange(len(B)); rows = np.array([(b[0], b[1], b[2], b[3] * 1.18, int(b[5] * 37) % 9973, .5 * b[4], .6 + .4 * math.sin(b[5]), age_erode(b[4], .34), 0., .2) for b in B]); mp = np.array(g['moon']); l = (mp[0] - 700., mp[1] - 700.); n = math.hypot(*l); l = (l[0] / n * .80, l[1] / n * .80 - .12, .22)
        pal = dict(lit='#a08de0', mid='#6453a8', dark='#352a6a', ink='#1a1440', rim='#efe7ff')
        depth = cam.cam(B[:, :3])[:, 2]; edges = [1e9, 100., 93., 0.]
        for hi, lo in zip(edges[:-1], edges[1:]):
            m = (depth <= hi) & (depth > lo)
            if m.any():
                L = lobes_layer(cam, rows[m], pal, light=l, ss=1.25, ink=.6, ink_in=.5, t_lit=.46, t_dark=.06, group=.50, rim=(l[0], l[1]), rim_amt=.5, micro=22., rough=.16); img = s.put(img, g, L, B[m, :3].mean(0), cam, soft=1.0)
        return img

    def trickle(s, img, g, cam, t):
        """fine dust pouring out of the cracks in thin curtains, down the face"""
        if t < s.T_CRACK or t > s.T_FALL + 6: return img
        k_all = float(np.clip((s.T_FALL + 6 - t) / 8., 0, 1))
        from .ink import Cel
        c = Cel(W, H)
        for k, ((i, j), (t0, x0, z0)) in enumerate(s.PAIRS.items()):
            r = np.random.default_rng(300 + k)
            for q in range(5):
                a = t - t0 - r.uniform(.5, 6)
                if a <= 0 or a > 22: continue
                x, z = x0 + r.normal(0, 2.2), z0 + r.normal(0, 1.6); L = min(1.25 * a + .05 * a * a, 11.); top = max(0., a - 12) * .9; n = 9; zz = np.linspace(z - top, max(.4, z - L), n)
                if zz[0] - zz[-1] < .5: continue
                P = cam.project(np.c_[x + np.cumsum(r.normal(.03, .08, n)), np.full(n, -.35), zz])[0]; c.line(P, r.uniform(5, 15), '#a596de', taper=(.15, .85), press=.5, seed=k * 9 + q, alpha=.20 * k_all * min(1, (22 - a) / 8.)); c.line(P, r.uniform(1.5, 4), '#d9cdfa', taper=(.1, .9), seed=k * 9 + q + 5, alpha=.20 * k_all * min(1, (22 - a) / 8.))
        L = c.rgba(); E = ((g['ID'] == 1) | ((g['ID'] >= 10) & (g['ID'] < 30)) | (g['ID'] == 0)).astype(f32); L = cv2.GaussianBlur(L, (0, 0), 1.6) * cv2.GaussianBlur(E, (0, 0), 1.)[..., None]; return img * (1 - L[..., 3:4]) + L[..., :3]

    def crumbs(s, img, g, cam, t):
        if not hasattr(s, '_cr'):
            r = np.random.default_rng(17); L = []
            for (i, j), (t0, x0, z0) in s.PAIRS.items():
                for _ in range(18): L.append((t0 + r.uniform(0, 16), x0 + r.normal(0, 2.), -.15 - .2 * r.random(), z0 + r.normal(0, 1.8), r.normal(0, 1.1), -r.gamma(1.5, .9), r.normal(.2, 1.0), min(.30, r.gamma(1.2, .075) + .04)))
            for _ in range(160): L.append((s.T_FALL + r.gamma(2.2, 4.), r.uniform(-4, 20), r.uniform(-3, 8), r.uniform(2, 24), r.normal(-1.5, 3.), -r.gamma(1.5, 1.6), r.normal(2., 3.), r.gamma(1.3, .16) + .06))
            s._cr = L; s._crp = [rock_planes(np.random.default_rng(400 + k), 1., .8, 9, .4, False) for k in range(24)]
        PN, PD, C = [], [], []; a_ = 1.0
        for k, (t0, x, y, z, vx, vy, vz, R) in enumerate(s._cr):
            a = (t - t0) / 24. * 1.9
            if a < 0: continue
            zz = z + vz * a - 4.9 * a * a; g0 = float(_hgt(s.Hm, x + vx * a, y + vy * a)) + R * .4
            if zz < g0:
                ah = (vz + math.sqrt(max(0, vz * vz + 19.6 * (z - .75)))) / 9.8; a = min(a, ah + .18); zz = g0
            nn, dd = s._crp[k % 24]; th = a * (3 + k % 5); c, q = math.cos(th), math.sin(th); M = np.array([[c, 0, q], [0, 1, 0], [-q, 0, c]]); PN.append(nn @ M); PD.append(dd * R); C.append((x + vx * a, y + vy * a, zz))
        if not PN: return img
        o = Monument.__new__(Monument); o.PAL = s.PAL; s._rocks.__func__(o, PN, PD, C); rays = s._rays(cam); Z = g['Z'].copy(); N = np.zeros_like(g['N']); ID = np.zeros_like(g['ID']); F = np.zeros_like(g['ID']); o._hull(rays, lambda P: cam.project(P)[0], lambda P: cam.scale(P), W, H, Z, N, ID, F, np.arange(len(C)), 7)
        m = ID == 7; d = N @ s.L.astype(f32); c = mix(rgb('#1a1636'), rgb('#4d4288'), smooth(.1, .8, N[..., 2:3] * .5 + .4)); c = mix(c, rgb('#d9cbff'), smooth(.05, .12, d)[..., None]); e = (cv2.dilate(m.astype(np.uint8), np.ones((3, 3))) > 0) & ~m
        a = cv2.GaussianBlur(m.astype(f32), (0, 0), .6)[..., None]; ea = cv2.GaussianBlur(e.astype(f32), (0, 0), .6)[..., None] * .7; img = img * (1 - ea) + rgb(s.PAL['line']) * ea; return img * (1 - a) + c * a

    def stones(s, cam, L, light=None, rimc='#e6dcff', pal=('#07060f', '#231c48', '#4b3f86'), ink=1.3, warm=None, edge=1.):
        """loose convex stones as one premultiplied layer.  L = [(normals (k,3), offsets (k,), centre)], world space.  returns (layer, depth)"""
        o = Monument.__new__(Monument); o.PAL = s.PAL; s._rocks.__func__(o, [q[0] for q in L], [q[1] for q in L], [q[2] for q in L]); rays = s._rays(cam); Z = np.full((H, W), 1e6, f32); N = np.zeros((H, W, 3), f32); ID = np.zeros((H, W), np.int32); F = np.zeros((H, W), np.int32)
        o._hull(rays, lambda P: cam.project(P)[0], lambda P: cam.scale(P), W, H, Z, N, ID, F, np.arange(len(L)), 7); m = ID == 7; x, y = grid()
        if not hasattr(s, '_nt'): s._nt = fbm(1024, 1024, (64, 24, 9, 4), (1, .7, .5, .35), 31)
        D = (rays[3] + x[..., None] * rays[4] + y[..., None] * rays[5]).astype(f32); D /= np.linalg.norm(D, axis=-1, keepdims=True); Pw = cam.eye.astype(f32) + D * np.minimum(Z, 1e5)[..., None]; fold = lambda q: (lambda k: np.where(k > 1023, 2047 - k, k))(np.floor(q).astype(np.int64) & 2047)
        n2 = s._nt[fold((Pw[..., 1] + 1.7 * Pw[..., 2]) * 130), fold(Pw[..., 0] * 130)]; l = s.L if light is None else np.asarray(light, float) / np.linalg.norm(light); up = N[..., 2] * .5 + .5
        rs = .60 * up + .24 * np.clip(-N[..., 0], 0, 1) + .12 * np.clip(-N[..., 1], 0, 1) + .05 * n2; st = smooth(.34, .365, rs) * .45 + smooth(.56, .585, rs) * .55; c = mix(mix(rgb(pal[0]), rgb(pal[1]), np.clip(st * 2, 0, 1)[..., None]), rgb(pal[2]), np.clip(st * 2 - 1, 0, 1)[..., None])
        d = N @ l.astype(f32); k = smooth(.04, .10, d + .02 * n2); c = mix(c, mix(rgb('#7d6cc8'), rgb(rimc), smooth(.3, .8, d)[..., None]), (k * .92)[..., None])
        fe = (np.roll(F, (1, 0), (0, 1)) != F) | (np.roll(F, (0, 1), (0, 1)) != F); dd = cv2.distanceTransform((~(fe & m)).astype(np.uint8), cv2.DIST_L2, 5); c = c + (np.clip(1.4 - dd, 0, 1) * (up > .5) * .22 * edge)[..., None] * rgb('#9c8be0'); c = c * (1 - (np.clip(1.5 - dd, 0, 1) * (up <= .5) * .6 * edge)[..., None])
        for k_ in range(3): pass
        a = cv2.GaussianBlur(m.astype(f32), (0, 0), .7); ed = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(ink + .6) + 1,) * 2)).astype(f32); ed = cv2.GaussianBlur(ed, (0, 0), .7); al = np.maximum(a, ed); col = rgb(s.PAL['line']) * (1 - a[..., None]) + cv2.GaussianBlur(np.where(m[..., None], c, 0), (0, 0), .35) ** 1. * a[..., None] if ink > 0 else c
        c = np.where(m[..., None], c, rgb(s.PAL['line'])); c = fxaa(c.astype(f32)); return np.dstack([c * al[..., None], al]).astype(f32), Z

    def put(s, img, g, layer, p, cam, soft=.0):
        """composite a premultiplied full-frame layer standing at world point p: hidden by anything nearer"""
        zc = float(np.ravel(cam.cam(np.asarray(p, float)))[2]); zs = g['Z'] * (g['D'] @ cam.R[2].astype(f32)); v = (zs > zc - .4).astype(f32) if soft <= 0 else smooth(-soft, soft * .4, zs - zc).astype(f32); v = cv2.GaussianBlur(v, (0, 0), .6)[..., None]; a = layer[..., 3:4] * v; return img * (1 - a) + layer[..., :3] * v

    def ground_at(s, cam, u, v):
        """world point of the waste under screen pixel (u, v)"""
        O0, _, _, D0, DA, DB = s._rays(cam); d = D0 + u * DA + v * DB; t = (cam.eye[2] - .4) / max(1e-6, -d[2]); p = cam.eye + d * t
        for _ in range(6): t = (cam.eye[2] - float(_hgt(s.Hm, p[0], p[1]))) / max(1e-6, -d[2]); p = cam.eye + d * t
        return np.array([p[0], p[1], float(_hgt(s.Hm, p[0], p[1]))])
