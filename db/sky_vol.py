"""Cloud worlds a camera can fly through: every cloud is a heap of wobbling billows (spheres) in metres, splatted per frame,
lit by one sun with a real shadow map, then laid in flat gouache tones: crisp sunlit crowns, quiet shadow masses, lost dark edges.
    V = Vol(); V.cumulus(0, 900, 200, 420, 380, seed=3); V.deck((-3000,3000), (0,8000), 0., 90, seed=1); V.build(sun=(.4,.6,.7))
    g = V.render(cam, pal=[tones('#fffdf4','#ffe9c8','#c9b6cc','#8d8cba','#6a6ba4','#e7c9c0')], sky=sky_rgb, fog=dict(d=4200))
    im = over_g(sky_rgb, g)       # g = dict(rgb, a, z, v)
+X right, +Y forward, +Z up (db.cam.Camera3D).  palette row: highlight, lit, half, shadow, deep, bounce."""
import numpy as np, cv2, math
from numba import njit, prange
from .paint import f32, rgb, smooth, fbm, lum, grid, _vn, _h2

NB = 18


@njit(cache=True, fastmath=True)
def _h3(x, y, z, s):
    n = (x * 374761393 + y * 668265263 + z * 2147483647 + s * 1274126177) & 0xffffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0xffffffff
    return ((n ^ (n >> 16)) & 0xffff) / 65535.0


@njit(cache=True, fastmath=True)
def vn3(x, y, z, s):
    ix = int(math.floor(x)); iy = int(math.floor(y)); iz = int(math.floor(z))
    fx = x - ix; fy = y - iy; fz = z - iz
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz)
    r = 0.0
    for k in range(2):
        a = _h3(ix, iy, iz + k, s); b = _h3(ix + 1, iy, iz + k, s); c = _h3(ix, iy + 1, iz + k, s); d = _h3(ix + 1, iy + 1, iz + k, s)
        v = (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy
        r += v * (fz if k else 1 - fz)
    return r


@njit(parallel=True, cache=True, fastmath=True)
def _splat(Z, ID, NX, NY, X, Y, PR, ZC, RR, S, SQ, wob, nb, zmin):
    h, w = Z.shape; n = X.shape[0]; bh = (h + nb - 1) // nb
    for b in prange(nb):
        y0b = b * bh; y1b = min(h, y0b + bh)
        for i in range(n):
            pr = PR[i]; py = pr * SQ[i]; cy = Y[i]; cx = X[i]
            if cy + py * 1.16 < y0b or cy - py * 1.16 >= y1b: continue
            if cx + pr * 1.16 < 0 or cx - pr * 1.16 >= w: continue
            zc = ZC[i]; r = RR[i]; zf = zc - r; s = S[i]
            ya = int(cy - py * 1.16)
            if ya < y0b: ya = y0b
            yb = int(cy + py * 1.16) + 1
            if yb > y1b - 1: yb = y1b - 1
            for y in range(ya, yb + 1):
                dy = (y + .5 - cy) / py
                rem = 1.33 - dy * dy
                if rem <= 0: continue
                hx = math.sqrt(rem) * pr
                xa = int(cx - hx)
                if xa < 0: xa = 0
                xb = int(cx + hx) + 1
                if xb > w - 1: xb = w - 1
                for x in range(xa, xb + 1):
                    if zf >= Z[y, x]: continue
                    dx = (x + .5 - cx) / pr
                    d2 = dx * dx + dy * dy
                    if d2 >= 1.33: continue
                    k = 1.0
                    if d2 > .08 and pr > 2.:
                        d = math.sqrt(d2); c = dx / d; sn = dy / d
                        # billowed outline: rounded lobes, sharp creases between them
                        nn = _vn(c * 1.8 + 10, sn * 1.8 + 10, s) * .62 + abs(2 * _vn(c * 3.4 + 30, sn * 3.4 + 30, s + 1) - 1) * .42
                        if pr > 22: nn += (abs(2 * _vn(c * 7.5 + 50, sn * 7.5 + 50, s + 2) - 1) - .4) * .17
                        if pr > 80: nn += (abs(2 * _vn(c * 16. + 90, sn * 16. + 90, s + 3) - 1) - .4) * .05
                        u = (d2 - .08) * 2.4
                        if u > 1: u = 1.0
                        k = 1.0 / (1 + wob * (nn - .48) * 2 * u)
                    q2 = d2 * k * k
                    if q2 >= 1: continue
                    z = zc - r * math.sqrt(1 - q2)
                    if z < zmin: continue
                    if z < Z[y, x]:
                        Z[y, x] = z; ID[y, x] = i; NX[y, x] = dx * k; NY[y, x] = dy * k


@njit(cache=True, fastmath=True)
def _ss(a, b, x):
    t = (x - a) / (b - a)
    if t < 0: t = 0.0
    if t > 1: t = 1.0
    return t * t * (3 - 2 * t)


@njit(cache=True, fastmath=True)
def _brush(px, py, pz, z, f, size, an, sd):
    """bristle noise of constant screen size, glued to the world: -.5 .. .5"""
    c = z / f * size
    if c < 1e-4: return 0.0
    l = math.log2(c); i = math.floor(l); t = l - i; c0 = 2.0 ** i
    a = vn3(px / (c0 * an), py / (c0 * an), pz / c0, sd); c1 = c0 * 2; b = vn3(px / (c1 * an) + 5.5, py / (c1 * an), pz / c1, sd + 3)
    return a + (b - a) * t - .5


@njit(parallel=True, cache=True, fastmath=True)
def _gbuf(B, Z, ID, NX, NY, eye, Rm, f, cx, cy, R, P, FN, PR_, SM, bu, bv, bw, su, sv, sk, w0, w1, w2, namp, ncell, shs, brush):
    """B[...,0:3] mixed form normal, 3 sun visibility, 4 noise"""
    h, w = Z.shape; sh, sw = SM.shape
    for y in prange(h):
        for x in range(w):
            i = ID[y, x]
            if i < 0: continue
            z = Z[y, x]; dx = (x + .5 - cx) / f; dy = (y + .5 - cy) / f
            px = eye[0] + (Rm[0, 0] * dx + Rm[1, 0] * dy + Rm[2, 0]) * z
            py = eye[1] + (Rm[0, 1] * dx + Rm[1, 1] * dy + Rm[2, 1]) * z
            pz = eye[2] + (Rm[0, 2] * dx + Rm[1, 2] * dy + Rm[2, 2]) * z
            nx = NX[y, x]; ny = NY[y, x]; nn = nx * nx + ny * ny
            nz = math.sqrt(1 - nn) if nn < 1 else 0.0
            wx = Rm[0, 0] * nx + Rm[1, 0] * ny - Rm[2, 0] * nz
            wy = Rm[0, 1] * nx + Rm[1, 1] * ny - Rm[2, 1] * nz
            wz = Rm[0, 2] * nx + Rm[1, 2] * ny - Rm[2, 2] * nz
            gx = px - P[i, 0]; gy = py - P[i, 1]; gz = pz - P[i, 2]; gl = math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-6
            ax = wx * w0 + gx / gl * w1 + FN[i, 0] * w2; ay = wy * w0 + gy / gl * w1 + FN[i, 1] * w2; az = wz * w0 + gz / gl * w1 + FN[i, 2] * w2
            al = math.sqrt(ax * ax + ay * ay + az * az) + 1e-6
            r = R[i]; c0 = ncell * PR_[i]
            n1 = vn3(px / c0, py / c0, pz / c0, 7) - .5; n2 = vn3(px / c0 * 2.7 + 9, py / c0 * 2.7, pz / c0 * 2.7, 11) - .5
            jx = px + wx * r * .12 + n1 * c0 * .25; jy = py + wy * r * .12 + n2 * c0 * .25; jz = pz + wz * r * .12
            u = (jx * bu[0] + jy * bu[1] + jz * bu[2] - su) * sk; v = (jx * bv[0] + jy * bv[1] + jz * bv[2] - sv) * sk; dw = jx * bw[0] + jy * bw[1] + jz * bw[2]
            s = 1.0
            if u > 2 and v > 2 and u < sw - 3 and v < sh - 3:
                m0 = SM[int(v), int(u)]; gap = dw - m0; rad = 0.8 + (gap * shs * sk if gap > 0 else 0.0)
                if rad > 9: rad = 9.0
                acc = 0.0
                for k in range(7):
                    ox = (0.0, 1.0, -1.0, 0.5, -0.5, .5, -.5)[k] * rad; oy = (0.0, 0., 0., .87, -.87, -.87, .87)[k] * rad
                    uu = min(max(int(u + ox), 0), sw - 1); vv = min(max(int(v + oy), 0), sh - 1)
                    acc += 1.0 - _ss(0.0, r * .3 + 3.0 / sk, dw - SM[vv, uu] - r * .10 - 2.0 / sk)
                s = acc / 7.0
            B[y, x, 0] = ax / al; B[y, x, 1] = ay / al; B[y, x, 2] = az / al; B[y, x, 3] = s; B[y, x, 4] = n1 * .65 + n2 * .35 + brush * _brush(px, py, pz, z, f, 5.0, 5.0, 23)


@njit(parallel=True, cache=True, fastmath=True)
def _atrous(src, dst, Z, ID, R, PR_, step, vert, tolk):
    """one separable edge-keeping smoothing pass: billows melt together, occlusion edges stay cut"""
    h, w, nc = src.shape
    for y_ in prange(h):
        y = np.int64(y_)
        for x in range(w):
            i = ID[y, x]
            if i < 0: continue
            z0 = Z[y, x]; tol = max(R[i], .55 * PR_[i]) * tolk
            ws = .375
            for c in range(nc): dst[y, x, c] = src[y, x, c] * .375
            for k in range(-2, 3):
                if k == 0: continue
                yy = y + k * step if vert else y; xx = x if vert else x + k * step
                if yy < 0 or xx < 0 or yy >= h or xx >= w: continue
                if ID[yy, xx] < 0: continue
                d = abs(Z[yy, xx] - z0) / tol
                if d >= 1: continue
                g = (.25 if abs(k) == 1 else .0625) * (1 - d * d)
                ws += g
                for c in range(nc): dst[y, x, c] += src[yy, xx, c] * g
            for c in range(nc): dst[y, x, c] /= ws


@njit(parallel=True, cache=True, fastmath=True)
def _tone(out, A, VV, B, F, RI, Z, ID, NX, NY, eye, Rm, f, cx, cy, PAL, BASE, FOOT, PALS, L, sunx, suny, sunin, fog_d, fog_p, fogc, usesky, sky, mist_z, mist_h, mist_c, mist_a, q, near0, near1, PLp, PLc, PLi, PLr):
    h, w = Z.shape
    tl, el, th, eh, namp, sil, silw, bnc, hgrad, skyl, kd, acc, accs, silr, deep = q
    for y in prange(h):
        for x in range(w):
            i = ID[y, x]
            if i < 0: continue
            z = Z[y, x]; dx = (x + .5 - cx) / f; dy = (y + .5 - cy) / f
            pz = eye[2] + (Rm[0, 2] * dx + Rm[1, 2] * dy + Rm[2, 2]) * z
            ax = B[y, x, 0]; ay = B[y, x, 1]; az = B[y, x, 2]; s = B[y, x, 3]; nm = B[y, x, 4]
            al = math.sqrt(ax * ax + ay * ay + az * az) + 1e-6; az_ = az / al
            xs = (ax * L[0] + ay * L[1] + az * L[2]) / al; bx = F[y, x, 0]; by = F[y, x, 1]; bz = F[y, x, 2]; bl = math.sqrt(bx * bx + by * by + bz * bz) + 1e-6
            dxf = (bx * L[0] + by * L[1] + bz * L[2]) / bl - xs; sf = F[y, x, 3]
            xx = xs + kd * dxf + namp * nm
            sn = s + (sf - s) * .35 + nm * namp * .5
            lit = _ss(tl - el, tl + el, xx) * _ss(.46 - el, .46 + el * 2, sn)
            hlf = _ss(th - eh, th + eh, xx - (1 - s) * .42 + skyl * az_)
            hi = _ss(tl + .30, tl + .30 + el * 2.5, xx) * lit
            dp = _ss(.10, -.80, az_ + nm * .4) * (1 - lit) * (1 - .6 * hlf) * deep
            p = PAL[i]
            r0 = PALS[p, 3, 0]; g0 = PALS[p, 3, 1]; b0 = PALS[p, 3, 2]
            r0 += (PALS[p, 4, 0] - r0) * dp; g0 += (PALS[p, 4, 1] - g0) * dp; b0 += (PALS[p, 4, 2] - b0) * dp
            r0 += (PALS[p, 2, 0] - r0) * hlf; g0 += (PALS[p, 2, 1] - g0) * hlf; b0 += (PALS[p, 2, 2] - b0) * hlf
            if bnc > 0:
                bk = _ss(.0, -.8, az_) * bnc * (1 - lit)
                r0 += (PALS[p, 5, 0] - r0) * bk; g0 += (PALS[p, 5, 1] - g0) * bk; b0 += (PALS[p, 5, 2] - b0) * bk
            r0 += (PALS[p, 1, 0] - r0) * lit; g0 += (PALS[p, 1, 1] - g0) * lit; b0 += (PALS[p, 1, 2] - b0) * lit
            r0 += (PALS[p, 0, 0] - r0) * hi; g0 += (PALS[p, 0, 1] - g0) * hi; b0 += (PALS[p, 0, 2] - b0) * hi
            if acc > 0:                                       # small forms: quiet half-tone crescents inside the light, faint lifts inside the shade
                k1 = _ss(.10, .34, -dxf + nm * .10 + (s - sf) * .5) * lit * acc * (1 - .35 * hi); k2 = _ss(.14, .46, dxf + nm * .10) * (1 - lit) * accs; k3 = _ss(.12, .5, -dxf) * (1 - lit) * accs * .7
                r0 += (PALS[p, 2, 0] - r0) * k1; g0 += (PALS[p, 2, 1] - g0) * k1; b0 += (PALS[p, 2, 2] - b0) * k1
                r0 += (PALS[p, 2, 0] - r0) * k2; g0 += (PALS[p, 2, 1] - g0) * k2; b0 += (PALS[p, 2, 2] - b0) * k2
                r0 += (PALS[p, 4, 0] - r0) * k3; g0 += (PALS[p, 4, 1] - g0) * k3; b0 += (PALS[p, 4, 2] - b0) * k3
            vis = lit + hlf * .4
            for k in range(PLi.shape[0]):                     # near lamps: auras, beams, lightning
                px = eye[0] + (Rm[0, 0] * dx + Rm[1, 0] * dy + Rm[2, 0]) * z; py = eye[1] + (Rm[0, 1] * dx + Rm[1, 1] * dy + Rm[2, 1]) * z
                qx = PLp[k, 0] - px; qy = PLp[k, 1] - py; qz = PLp[k, 2] - pz; qd = math.sqrt(qx * qx + qy * qy + qz * qz) + 1e-3
                nd = (ax * qx + ay * qy + az * qz) / (al * qd) + .35 * (bx * qx + by * qy + bz * qz) / (bl * qd); fa = PLi[k] / (1 + (qd / PLr[k]) ** 4)
                e = (_ss(.10, .10 + el * 3, nd + nm * .2) * .62 + _ss(.55, .55 + el * 3, nd + nm * .2) * .38) * fa
                if e > 1: e = 1.0
                r0 += (PLc[k, 0] - r0) * e; g0 += (PLc[k, 1] - g0) * e; b0 += (PLc[k, 2] - b0) * e; vis += e
            if sil > 0:
                nx = NX[y, x]; ny = NY[y, x]
                fx_ = sunx - x; fy_ = suny - y; fl = math.sqrt(fx_ * fx_ + fy_ * fy_) + 1.0
                dd = (nx * fx_ + ny * fy_) / fl * sunin; near = 1.0 / (1.0 + fl / (w * .42))
                sw_ = silw * min(1., 95. / (RI[i] * f / z)); e = _ss(1 - sw_, 1 - sw_ * .62, dd + nm * sw_ * .6) * sil * (.4 + .6 * s)
                if silr > 0: e *= 1.25 * math.exp(-(fl / (w * silr)) ** 2)
                else: e *= .35 + 1.1 * near
                if e > 1: e = 1.0
                r0 += (PALS[p, 0, 0] - r0) * e; g0 += (PALS[p, 0, 1] - g0) * e; b0 += (PALS[p, 0, 2] - b0) * e; vis += e
            vv2 = 1 + hgrad * nm
            r0 *= vv2; g0 *= vv2; b0 *= vv2
            a = 1.0
            if FOOT[i] > 0: a = _ss(BASE[i], BASE[i] + FOOT[i], pz + nm * FOOT[i] * .7)
            if near1 > 0: a *= _ss(near0, near1, z + nm * (near1 - near0) * .5)
            dist = z * math.sqrt(1 + dx * dx + dy * dy); k = 1 - math.exp(-(dist / fog_d) ** fog_p)
            if mist_a > 0:
                mk = (1 - _ss(mist_z, mist_z + mist_h, pz + nm * mist_h * .5)) * mist_a
                r0 += (mist_c[0] - r0) * mk; g0 += (mist_c[1] - g0) * mk; b0 += (mist_c[2] - b0) * mk; vis *= 1 - mk
            if usesky: r1 = sky[y, x, 0]; g1 = sky[y, x, 1]; b1 = sky[y, x, 2]
            else: r1 = fogc[0]; g1 = fogc[1]; b1 = fogc[2]
            out[y, x, 0] = r0 + (r1 - r0) * k; out[y, x, 1] = g0 + (g1 - g0) * k; out[y, x, 2] = b0 + (b1 - b0) * k
            A[y, x] = a; VV[y, x] = vis * (1 - .6 * k)


def tones(*c):
    return np.stack([rgb(v) if isinstance(v, str) else np.asarray(v, f32) for v in c]).astype(f32)


def mix_pal(a, b, t): return a + (b - a) * t


class Vol:
    def __init__(self):
        self.c, self.r, self.p, self.fn, self.pr, self.pal, self.base, self.foot, self.lv = [], [], [], [], [], [], [], [], []
        self.smap = None

    # ------------------------------------------------------------------ building
    def add(self, C, R, P, FN, PR, pal=0, base=-1e9, foot=0., lv=0):
        n = len(R)
        if n == 0: return self
        self.c.append(np.asarray(C, f32)); self.r.append(np.asarray(R, f32))
        for L, v in ((self.p, P), (self.fn, FN)): L.append(np.broadcast_to(np.asarray(v, f32), (n, 3)).copy())
        for L, v in ((self.pr, PR), (self.base, base), (self.foot, foot)): L.append(np.broadcast_to(np.asarray(v, f32), (n,)).copy())
        for L, v in ((self.pal, pal), (self.lv, lv)): L.append(np.broadcast_to(np.asarray(v, np.int32), (n,)).copy())
        return self

    def crust(self, C, R, FN, seed, pal, base, foot, levels=2, kids=(12, 9, 7), sc=(.30, .56), up=.35, keep=None, cover=None):
        """children billows riding every parent, pruned where buried"""
        r = np.random.default_rng(seed * 331 + 7); C0, R0, F0 = np.asarray(C, float), np.asarray(R, float), np.asarray(FN, float)
        big = (C0, R0) if cover is None else cover
        for lv in range(levels):
            n = kids[min(lv, len(kids) - 1)]; k = np.repeat(np.arange(len(R0)), n); v = r.normal(0, 1, (len(k), 3)) + F0[k] * (.75 + .4 * lv); v[:, 2] += up * .6; v /= np.linalg.norm(v, axis=1)[:, None]
            s = r.uniform(*sc, len(k)) * (1 - .06 * lv); c = C0[k] + v * (R0[k] * r.uniform(.70, .95, len(k)))[:, None]; rr = R0[k] * s; fn = F0[k] * .62 + v * .38; fn /= np.linalg.norm(fn, axis=1)[:, None]
            vis = np.ones(len(k), bool)
            for a in range(0, len(big[1]), 256):
                D = ((c[:, None] - big[0][None, a:a + 256]) ** 2).sum(-1); vis &= (D > np.maximum(big[1][None, a:a + 256] - rr[:, None] * .80, 0) ** 2).all(1)
            c, rr, fn, k = c[vis], rr[vis], fn[vis], k[vis]
            if keep is not None: m = keep(c, rr, lv); c, rr, fn, k = c[m], rr[m], fn[m], k[m]
            if not len(rr): break
            self.add(c, rr, C0[k], fn, R0[k], pal, base, foot, lv + 1); C0, R0, F0 = c, rr, fn
        return self

    def cumulus(self, x, y, z, w, h, seed=0, pal=0, depth=None, peaks=3, lean=0., levels=2, kids=(17, 8, 5), foot=.16, n=46, size=1., skirt=.5):
        """a standing cumulus: a mountain of billows with `peaks` summits, base at altitude z, w wide, h tall"""
        r = np.random.default_rng(seed * 7907 + 11); d = w * .62 if depth is None else depth; pk = []; a0 = r.uniform(0, 6.28)
        for k in range(peaks):
            a = a0 + k * 2.4 + r.normal(0, .4); q = r.uniform(0, .25) if k == 0 else r.uniform(.45, .95); am = 1. if k == 0 else r.uniform(.34, .80)
            pk.append((math.cos(a) * q * w * .36, math.sin(a) * q * d * .30, am, r.uniform(.20, .27) if k == 0 else r.uniform(.12, .20)))
        def H(px, py):
            e = np.zeros_like(px)
            for (ax, ay, am, sg) in pk: q = np.sqrt(((px - ax) / (w * sg)) ** 2 + ((py - ay) / (d * sg * 1.5)) ** 2); e = np.maximum(e, am * (.62 * np.exp(-q ** 2.6) + .38 * np.exp(-(q * .62) ** 2)))
            q = np.clip((px / (w * .5)) ** 2 + (py / (d * .5)) ** 2, 0, 1); return np.maximum(e * np.clip((1.02 - q) * 4, 0, 1), skirt * .42 * np.sqrt(np.clip(1 - q, 0, 1))) * h
        a = r.uniform(0, 6.283, n); q = np.sqrt(r.random(n)) * .97; px, py = np.cos(a) * q * w * .5, np.sin(a) * q * d * .5
        hz = H(px, py); e = w * .03; gx = (H(px + e, py) - H(px - e, py)) / (2 * e); gy = (H(px, py + e) - H(px, py - e)) / (2 * e)
        fn = np.c_[-gx, -gy, np.full(n, .75)]; fn /= np.linalg.norm(fn, axis=1)[:, None]
        rr = w * size * (.075 + .085 * r.random(n) ** 1.5) * (1.15 - .40 * hz / h); zc = np.maximum(hz - rr * .75, rr * .55)
        C = [np.c_[px, py, zc]]; Rr = [rr]; F = [fn]
        m2 = int(n * 2.2); a = r.uniform(0, 6.283, m2); q = np.sqrt(r.random(m2)) * .95; qx, qy = np.cos(a) * q * w * .5, np.sin(a) * q * d * .5; hq = H(qx, qy)      # fill the flanks under the summits
        gx = (H(qx + e, qy) - H(qx - e, qy)) / (2 * e); gy = (H(qx, qy + e) - H(qx, qy - e)) / (2 * e); f2 = np.c_[-gx, -gy, np.full(m2, .45)]; f2 /= np.linalg.norm(f2, axis=1)[:, None]
        r2 = w * size * (.085 + .085 * r.random(m2)); z2 = hq * r.uniform(.12, .86, m2); m = hq - z2 > r2 * .5; C.append(np.c_[qx, qy, np.maximum(z2, r2 * .4)][m]); Rr.append(r2[m]); F.append(f2[m])
        C, Rr, F = np.concatenate(C), np.concatenate(Rr), np.concatenate(F); F /= np.linalg.norm(F, axis=1)[:, None]
        C[:, 0] += lean * C[:, 2]; C = C + [x, y, z]; P = np.array([x, y, z + h * .25]); b, ft = z - .01 * h, foot * h
        self.add(C, Rr, P, F, w * .30, pal, b, ft, 0)
        return self.crust(C, Rr, F, seed + 1, pal, b, ft, levels, kids)

    def bank(self, x, y, z, w, h, seed=0, pal=0, depth=.34, peaks=5, levels=2, kids=(15, 7), foot=.18, skirt=.6, n=None, k=1.):
        """a long low bank of cloud lying on a floor: billow size follows its height"""
        size = min(1.15, 2.1 * h / w) * k; n = int(np.clip(26 / size ** 1.35, 34, 170)) if n is None else n
        return self.cumulus(x, y, z, w, h * .82, seed, pal, w * depth, peaks, 0., levels, kids, foot, n, size, skirt)

    def ridge(self, y, xr, R, z, seed=0, pal=0, amp=.6, lam=9., gap=.25, rise=None, depth=1.2, levels=2, kids=(9, 6), foot=1.1, sag=.15):
        """one crest line of a cloud sea running left-right at distance y: scalloped crown, foot lost in the mist of the floor.  gap = share left open, rise(x) extra height"""
        r = np.random.default_rng(seed * 4099 + 1); n = int((xr[1] - xr[0]) / (R * .62)); x = np.linspace(*xr, n) + r.normal(0, .25 * R, n)
        nz = lambda l, s: np.array([_vn(a / l + 70.3, y * .01 + 3.7, seed + s) for a in x]); u = nz(R * lam, 5) * .62 + nz(R * lam * .31, 6) * .38; g = nz(R * lam * 1.7, 9)
        keep = g > np.quantile(g, gap) if gap > 0 else np.ones(n, bool); e = smooth(np.quantile(g, gap), np.quantile(g, min(1., gap + .18)), g) if gap > 0 else np.ones(n)
        rr = R * (.55 + .75 * u) * r.uniform(.75, 1.2, n) * (.45 + .55 * e); zc = z + R * amp * (u - .45) * 2 - rr * sag + (rise(x) if rise is not None else 0.); yy = y + r.normal(0, R * depth * .5, n)
        C = np.c_[x, yy, zc][keep]; rr = rr[keep]; dz = np.gradient(zc, x)[keep]; F = np.c_[-dz * .5, np.zeros(len(rr)), np.ones(len(rr))]; F /= np.linalg.norm(F, axis=1)[:, None]
        b = z - R * .35; self.add(C, rr, C - [0, 0, 1] * (rr * 1.2)[:, None], F, rr, pal, b, foot * R, 0)
        return self.crust(C, rr, F, seed + 2, pal, b, foot * R, levels, kids, (.36, .66), .5, keep=lambda c, q, lv: c[:, 2] > z - R * .15)

    def deck(self, xr, yr, z, size, seed=0, pal=0, jitter=.45, rel=.55, levels=2, kids=(7, 5), thr=None, sink=.45, foot=0., hole=None, detail_d=None, eye=None, grow=0., den=1., under=False):
        """a floor of cloud: billows on a plane at altitude z; rel = relief (swell of the floor), thr = cover 0..1 (None = closed),
        hole = [(x, y, r)] clear discs; grow = size added per metre of distance from eye (far billows bigger)"""
        r = np.random.default_rng(seed * 6007 + 3); y = yr[0]; rows = []
        while y < yr[1]:
            d = abs(y - (eye[1] if eye is not None else y)); k = 1 + grow * d; s = size * 1.02 / den * k; n = int((xr[1] - xr[0]) / s) + 1
            rows.append(np.c_[xr[0] + (np.arange(n) + r.uniform(-jitter, jitter, n)) * s, y + r.uniform(-jitter, jitter, n) * s, np.full(n, size * k)]); y += s
        gx, gy, sz = np.concatenate(rows).T
        sw = lambda X, Y: np.array([(_vn(a / (size * 9) + 50.3, b / (size * 9) + 50.7, seed + 5) - .5) * 1.3 + (_vn(a / (size * 3.4) + 50.3, b / (size * 3.4) + 50.7, seed + 6) - .5) * .8 for a, b in zip(X, Y)])
        f = sw(gx, gy); e = size * .5; FN = np.c_[-(sw(gx + e, gy) - f) / e, -(sw(gx, gy + e) - f) / e, np.zeros(len(gx))] * size * 2.6 * rel; FN[:, 2] = 1.; FN /= np.linalg.norm(FN, axis=1)[:, None]
        keep = np.ones(len(gx), bool)
        if thr is not None: keep &= f > np.quantile(f, 1 - thr)
        if hole is not None:
            for (hx, hy, hr) in hole: keep &= np.hypot(gx - hx, gy - hy) > hr * (1 + .25 * f)
        gx, gy, sz, f, FN = gx[keep], gy[keep], sz[keep], f[keep], FN[keep]
        R = sz * (.60 + .55 * r.random(len(gx)) ** 1.6) * (1 + .8 * np.clip(f, -.4, .8)); sg = -1. if under else 1.; zc = z + sg * (f * size * 2.6 * rel - R * sink); FN[:, 2] *= sg
        C = np.c_[gx, gy, zc]; P = C - [0, 0, sg] * (R * 1.4)[:, None]; b = -1e9 if under else z - size * 1.4; foot = 0. if under else foot; self.add(C, R, P, FN, R, pal, b, foot, 0)
        C0, R0, F0 = C, R, FN
        for lv in range(levels):
            if detail_d is not None and eye is not None: m = np.hypot(C0[:, 0] - eye[0], C0[:, 1] - eye[1]) < detail_d / (1 + lv); C0, R0, F0 = C0[m], R0[m], F0[m]
            if len(R0) == 0: break
            n = kids[min(lv, len(kids) - 1)]; k = np.repeat(np.arange(len(R0)), n); v = r.normal(0, 1, (len(k), 3)); v[:, 2] = (np.abs(v[:, 2]) * .8 + .12) * sg; v /= np.linalg.norm(v, axis=1)[:, None]
            s = r.uniform(.28, .56, len(k)); c = C0[k] + v * (R0[k] * r.uniform(.72, .97, len(k)))[:, None]; rr = R0[k] * s; fn = F0[k] * .55 + v * .45; fn /= np.linalg.norm(fn, axis=1)[:, None]
            self.add(c, rr, C0[k], fn, R0[k], pal, b, foot, lv + 1); C0, R0, F0 = c, rr, fn
        return self

    def puff(self, x, y, z, s, seed=0, pal=0, n=10, foot=0., stretch=(1.6, 1., .5), levels=2, kids=(9, 6)):
        """small torn scrap of cloud"""
        r = np.random.default_rng(seed * 31 + 5); v = r.normal(0, .42, (n, 3)) * np.array(stretch) * s; R = s * r.uniform(.28, .62, n) * np.exp(-np.abs(v[:, 0]) / (s * 2.2)); C = v + [x, y, z]
        fn = v / s + [0, 0, .6]; fn /= np.linalg.norm(fn, axis=1)[:, None]; b = z - s * .5; self.add(C, R, (x, y, z - s * .3), fn, s * .6, pal, b, foot, 0); return self.crust(C, R, fn, seed + 3, pal, b, foot, levels, kids)

    def build(self, sun=(.4, .5, .75), res=2048, box=None, cast=0):
        """concatenate and make the shadow map (billows of level <= cast throw shadows).  box = (xmin,xmax,ymin,ymax,zmin,zmax) region that casts"""
        self.C, self.R, self.P, self.FN, self.PR = [np.concatenate(v).astype(f32) for v in (self.c, self.r, self.p, self.fn, self.pr)]
        self.PAL, self.LV = np.concatenate(self.pal), np.concatenate(self.lv); self.BASE, self.FOOT = [np.concatenate(v).astype(f32) for v in (self.base, self.foot)]
        self.S = (np.arange(len(self.R)) * 613 + 17).astype(np.int64) & 0xffffff
        self.c = self.r = self.p = self.fn = self.pr = self.pal = self.lv = self.base = self.foot = None; return self.sun(sun, res, box, cast)

    def sun(self, sun, res=2048, box=None, cast=0, C=None):
        L = np.asarray(sun, float); L = L / np.linalg.norm(L); up = np.array([0, 0, 1.]) if abs(L[2]) < .95 else np.array([0, 1., 0])
        bu = np.cross(up, L); bu /= np.linalg.norm(bu); bv = np.cross(L, bu); bw = -L; C = self.C if C is None else C
        m = self.LV <= cast
        if box is not None: m &= (C[:, 0] > box[0]) & (C[:, 0] < box[1]) & (C[:, 1] > box[2]) & (C[:, 1] < box[3]) & (C[:, 2] > box[4]) & (C[:, 2] < box[5])
        u, v, w = C[m] @ bu, C[m] @ bv, C[m] @ bw; R = self.R[m]; u0, u1, v0, v1 = (u - R).min(), (u + R).max(), (v - R).min(), (v + R).max(); sk = (res - 8) / max(u1 - u0, v1 - v0); su, sv = u0 - 4 / sk, v0 - 4 / sk
        M = np.full((int((v1 - v0) * sk) + 9, int((u1 - u0) * sk) + 9), 1e9, f32); o = np.argsort(w - R)
        _smap2(M, ((u - su) * sk)[o].astype(f32), ((v - sv) * sk)[o].astype(f32), w[o].astype(f32), R[o], f32(sk))
        self.L, self.bu, self.bv, self.bw, self.su, self.sv, self.sk, self.smap = L.astype(f32), bu.astype(f32), bv.astype(f32), bw.astype(f32), f32(su), f32(sv), f32(sk), M
        return self

    # ------------------------------------------------------------------ drawing
    def render(self, cam, pal, sky=None, fog=None, light=None, zr=(1., 1e9), wob=.11, sil=0., silw=.45, tl=.22, el=.05, th=-.10, eh=.22, mixn=(.30, .35, .55), noise=.14, ncell=.45, sh_soft=.015, bounce=.5, skyl=.12,
               mist=None, near=(0., 0.), lost=.9, grain=.03, sunscreen=None, minpx=.7, move=None, melt=5, tol=1.2, fine=1, kd=.35, acc=.55, lamps=None, accs=.2, brush=.5, shadow=True, silr=0., deep=1.):
        """-> dict(rgb, a, z, v).  pal (n,6,3).  fog=dict(d=metres, p=1.2, col=None->sky).  mist=dict(z,h,col,amt) pools below z+h.
        zr: depth range of this layer; near=(z0,z1) dissolves billows that reach the lens; sil: silver lining toward sunscreen=(x,y) px.
        lamps=[dict(pos=(x,y,z), col, I, r)] local coloured lights.  move(C)->C displaces the billows (ring punches, wind).  melt = how far billows melt together (0..4)."""
        C = self.C if move is None else move(self.C).astype(f32); Pm = self.P if move is None else self.P + (C - self.C)
        Rm = cam.R.astype(f32); eye = np.asarray(cam.eye, f32); q = (C - eye) @ Rm.T; z = q[:, 2]; R = self.R
        m = (z + R > max(zr[0], .5)) & (z - R < zr[1]) & (z > R * .25); pr = cam.f * R / np.maximum(z, 1e-3); m &= pr > minpx
        x = q[:, 0] / np.maximum(z, 1e-3) * cam.f + cam.cx; y = q[:, 1] / np.maximum(z, 1e-3) * cam.f + cam.cy
        m &= (x + pr * 1.2 > 0) & (x - pr * 1.2 < cam.w) & (y + pr * 1.2 > 0) & (y - pr * 1.2 < cam.h)
        ix = np.nonzero(m)[0]; ix = ix[np.argsort((z - R)[ix])]
        h, w = cam.h, cam.w; Z = np.full((h, w), 1e9, f32); ID = np.full((h, w), -1, np.int32); NX = np.zeros((h, w), f32); NY = np.zeros((h, w), f32)
        out = np.zeros((h, w, 3), f32); A = np.zeros((h, w), f32); V = np.zeros((h, w), f32)
        if len(ix) == 0: return dict(rgb=out, a=A, z=Z, v=V)
        _splat(Z, ID, NX, NY, x[ix].astype(f32), y[ix].astype(f32), pr[ix].astype(f32), z[ix].astype(f32), R[ix], self.S[ix], np.ones(len(ix), f32), f32(wob), NB, f32(zr[0]))
        if zr[1] < 1e8: ID[Z > zr[1]] = -1
        fg = dict(d=1e7, p=1.2, col=None); fg.update(fog or {}); usesky = fg['col'] is None and sky is not None
        fc = np.zeros(3, f32) if fg['col'] is None else (rgb(fg['col']) if isinstance(fg['col'], str) else np.asarray(fg['col'], f32))
        ms = dict(z=0., h=1., col='#ffffff', amt=0.); ms.update(mist or {}); mc = rgb(ms['col']) if isinstance(ms['col'], str) else np.asarray(ms['col'], f32)
        L = self.L if light is None else (np.asarray(light, f32) / np.linalg.norm(light)).astype(f32)
        sc, sin_ = ((w / 2, -h), 1.) if sunscreen is None else (sunscreen, 1.)
        pal = np.asarray(pal, f32)
        if pal.ndim == 2: pal = pal[None]
        pl = lamps or []; PLp = np.array([q['pos'] for q in pl], f32).reshape(-1, 3); PLc = np.array([rgb(q['col']) if isinstance(q['col'], str) else q['col'] for q in pl], f32).reshape(-1, 3); PLi = np.array([q.get('I', 1.) for q in pl], f32); PLr = np.array([q.get('r', 300.) for q in pl], f32)
        cf, ccx, ccy, Ri = f32(cam.f), f32(cam.cx), f32(cam.cy), R[ix]
        B = np.zeros((h, w, 5), f32)
        _gbuf(B, Z, ID, NX, NY, eye, Rm, cf, ccx, ccy, Ri, Pm[ix], self.FN[ix], self.PR[ix], self.smap if shadow else np.full((8, 8), 1e9, f32), self.bu, self.bv, self.bw, self.su, self.sv, self.sk, f32(mixn[0]), f32(mixn[1]), f32(mixn[2]), f32(noise), f32(ncell), f32(sh_soft), f32(brush))
        if melt > 0:
            T = np.zeros_like(B); Pr = self.PR[ix]; F = B
            for k in range(melt):
                _atrous(B, T, Z, ID, Ri, Pr, 1 << k, False, f32(tol)); _atrous(T, B, Z, ID, Ri, Pr, 1 << k, True, f32(tol))
                if k == fine: F = B.copy()
        else: F = B
        qq = np.array([tl, el, th, eh, noise, sil, silw, bounce, grain, skyl, kd, acc, accs, silr, deep], f32)
        _tone(out, A, V, B, F, Ri, Z, ID, NX, NY, eye, Rm, cf, ccx, ccy, self.PAL[ix], self.BASE[ix], self.FOOT[ix], pal, L, f32(sc[0]), f32(sc[1]), f32(sin_), f32(fg['d']), f32(fg['p']), fc, usesky, sky if usesky else np.zeros((1, 1, 3), f32),
              f32(ms['z']), f32(ms['h']), mc, f32(ms['amt']), qq, f32(near[0]), f32(near[1]), PLp, PLc, PLi, PLr)
        g = dict(rgb=out, a=A, z=Z, v=V)
        return soften(g, lost) if lost > 0 else g


@njit(cache=True, fastmath=True)
def _smap2(M, U, Vv, Wd, R, sk):
    h, w = M.shape
    for i in range(U.shape[0]):
        r = R[i] * sk; cx = U[i]; cy = Vv[i]
        if r < .6: continue
        x0 = max(0, int(cx - r)); x1 = min(w - 1, int(cx + r) + 1); y0 = max(0, int(cy - r)); y1 = min(h - 1, int(cy + r) + 1)
        wf = Wd[i] - R[i]
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if wf >= M[y, x]: continue
                dx = (x + .5 - cx) / r; dy = (y + .5 - cy) / r; d2 = dx * dx + dy * dy
                if d2 >= 1: continue
                z = Wd[i] - R[i] * math.sqrt(1 - d2)
                if z < M[y, x]: M[y, x] = z


def soften(g, amt=.9, s=3.2):
    """painter's edges: where both sides of an edge are dark the edge is lost, lit crowns stay cut"""
    a = g['a']; P = np.dstack([g['rgb'] * a[..., None], a]); B = cv2.GaussianBlur(P, (0, 0), s); vmax = cv2.dilate(g['v'], np.ones((5, 5), np.uint8)); k = (1 - smooth(.10, .62, cv2.GaussianBlur(vmax, (0, 0), 1.4))) * amt
    aa = cv2.GaussianBlur(a, (0, 0), 1.0); k = k * smooth(0, .5, aa)
    O = P + (B - P) * k[..., None]; A = O[..., 3]; g['rgb'] = O[..., :3] / np.maximum(A, 1e-4)[..., None]; g['a'] = A; return g


def prem(g): return np.dstack([g['rgb'] * g['a'][..., None], g['a']])


def over_g(base, g, op=1.):
    a = (g['a'] * op)[..., None]; return base * (1 - a) + g['rgb'] * a


@njit(parallel=True, cache=True, fastmath=True)
def _floor(out, eye, Rm, f, cx, cy, z0, l, ky, sx, sy, SM, bu, bv, bw, su, sv, sk, seed, ox, oy, shs):
    """flat cloud floor at altitude z0: out[...,0] distance (0 = no hit), 1 swell value, 2 slope toward the sun, 3 sun visibility, 4/5 world x,y"""
    h, w = out.shape[:2]; sh, sw = SM.shape
    for y_ in prange(h):
        y = np.int64(y_)
        for x in range(w):
            dx = (x + .5 - cx) / f; dy = (y + .5 - cy) / f
            rx = Rm[0, 0] * dx + Rm[1, 0] * dy + Rm[2, 0]; ry = Rm[0, 1] * dx + Rm[1, 1] * dy + Rm[2, 1]; rz = Rm[0, 2] * dx + Rm[1, 2] * dy + Rm[2, 2]
            if rz * (z0 - eye[2]) <= 0 or abs(rz) < 1e-5: continue
            t = (z0 - eye[2]) / rz; X = eye[0] + rx * t; Y = eye[1] + ry * t
            foot = t / f / (abs(rz) / math.sqrt(rx * rx + ry * ry + rz * rz) + .02)        # metres per px along depth
            v = 0.0; g = 0.0; am = 1.0; wl = l
            for o in range(5):
                at = (wl / foot - 1.6) / 6.0
                if at <= 0: break
                if at > 1: at = 1.0
                a = (X + ox) / wl + o * 17.3 + 100.; b = (Y + oy) / (wl * ky) + o * 9.1 + 100.
                wa = _vn(a * .33 + 3, b * .33 + 8, seed + 40 + o) * 2.2
                n = _vn(a + wa, b - wa * .7, seed + o); n2 = _vn(a + wa + sx * .22, b - wa * .7 + sy * .22, seed + o)
                n = 1 - abs(2 * n - 1); n2 = 1 - abs(2 * n2 - 1); n = n * n * (3 - 2 * n); n2 = n2 * n2 * (3 - 2 * n2)
                v += am * at * (n - .5); g += am * at * (n - n2); am *= .55; wl *= .46
            jx = X; jy = Y; jz = z0
            u = (jx * bu[0] + jy * bu[1] + jz * bu[2] - su) * sk; vv = (jx * bv[0] + jy * bv[1] + jz * bv[2] - sv) * sk; dw = jx * bw[0] + jy * bw[1] + jz * bw[2]
            s = 1.0
            if u > 2 and vv > 2 and u < sw - 3 and vv < sh - 3:
                m0 = SM[int(vv), int(u)]; gap = dw - m0; rad = 0.7 + (gap * shs * sk if gap > 0 else 0.0)
                if rad > 9: rad = 9.0
                acc = 0.0
                for k in range(7):
                    qx = (0.0, 1.0, -1.0, 0.5, -0.5, .5, -.5)[k] * rad; qy = (0.0, 0., 0., .87, -.87, -.87, .87)[k] * rad
                    uu = min(max(int(u + qx + v * .8), 0), sw - 1); ww = min(max(int(vv + qy), 0), sh - 1)
                    acc += 0.0 if dw - SM[ww, uu] > 3.0 / sk else 1.0
                s = acc / 7.0
            out[y, x, 0] = t; out[y, x, 1] = v; out[y, x, 2] = g; out[y, x, 3] = s; out[y, x, 4] = X; out[y, x, 5] = Y


def floor_field(V, cam, z0, l=400., ky=1.5, sun=(0., 1.), seed=3, off=(0., 0.), soft=.01):
    out = np.zeros((cam.h, cam.w, 6), f32)
    _floor(out, np.asarray(cam.eye, f32), cam.R.astype(f32), f32(cam.f), f32(cam.cx), f32(cam.cy), f32(z0), f32(l), f32(ky), f32(sun[0]), f32(sun[1]), V.smap, V.bu, V.bv, V.bw, V.su, V.sv, V.sk, seed, f32(off[0]), f32(off[1]), f32(soft))
    return out
