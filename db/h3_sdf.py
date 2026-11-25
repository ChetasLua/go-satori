"""Construction clay for db.hero3: posed muscles, bones, cloth shells and plates as a blended distance field, cast through a Camera3D into
label / depth / normal / (t, phi) / cavity buffers.  Nothing here is shown: h3_draw re-draws the buffers as ink and cel polygons.
prim types: 0 loft (axis + radius table R[t, phi]), 1 ellipsoid, 2 ring, 3 egg (skewed, boxy ellipsoid: a muscle), 4 tapered capsule."""
import numpy as np, math
from numba import njit

NT, NP = 24, 28
BIG = 1e9
TWO_PI = 2 * math.pi
NF = 24                              # floats per prim: 0-2 origin, 3-5 e1, 6-8 axis, 9-11 e2, 12-14 par, 15 k, 16-18 bound centre, 19 bound radius, 20 max r, 21-23 extra


@njit(cache=True, fastmath=True, inline='always')
def _tab(TB, k, t, ph):
    u = t * (NT - 1)
    i = int(u)
    if i < 0: i = 0
    if i > NT - 2: i = NT - 2
    fu = u - i
    v = ph * (NP / TWO_PI)
    if v < 0: v += NP
    j = int(v)
    fv = v - j
    if j >= NP: j -= NP
    j2 = j + 1
    if j2 >= NP: j2 = 0
    a = TB[k, i, j] * (1 - fv) + TB[k, i, j2] * fv
    b = TB[k, i + 1, j] * (1 - fv) + TB[k, i + 1, j2] * fv
    return a * (1 - fu) + b * fu


@njit(cache=True, fastmath=True)
def _prim(F, ty, tk, TB, px, py, pz):
    dx = px - F[0]; dy = py - F[1]; dz = pz - F[2]
    y = dx * F[6] + dy * F[7] + dz * F[8]
    if ty == 5: return 1e6           # a cloth sheet: drawn by raster(), never marched
    if ty == 4:                      # tapered capsule: L, r0, r1
        L = F[12]
        t = y / L
        if t < 0: t = 0.
        elif t > 1: t = 1.
        qx = dx - F[6] * t * L; qy = dy - F[7] * t * L; qz = dz - F[8] * t * L
        return math.sqrt(qx * qx + qy * qy + qz * qz) - (F[13] + (F[14] - F[13]) * t)
    x = dx * F[3] + dy * F[4] + dz * F[5]
    z = dx * F[9] + dy * F[10] + dz * F[11]
    if ty == 3:                      # egg: radii 12 (e1), 13 (axis), 14 (e2); 21 skew along the axis, 22 boxiness, 23 bow along e2
        u = y / F[13]
        uc = u
        if uc > 1: uc = 1.
        elif uc < -1: uc = -1.
        z = z + F[23] * (uc * uc)
        g = 1 + F[21] * uc
        rx = F[12] * g; rz = F[14] * g
        a = x / rx; b = u; c = z / rz
        a2 = a * a; b2 = b * b; c2 = c * c
        k0 = math.sqrt(a2 + b2 + c2)
        bx = F[22]
        if bx > 0:
            k0 = (1 - bx) * k0 + bx * math.sqrt(math.sqrt(a2 * a2 + b2 * b2 + c2 * c2))
        a /= rx; b /= F[13]; c /= rz
        k1 = math.sqrt(a * a + b * b + c * c) + 1e-9
        return k0 * (k0 - 1) / k1
    if ty == 0:                      # loft
        L = F[12]
        rho = math.sqrt(x * x + z * z)
        ph = math.atan2(z, x)
        if y < 0:
            over = -y; t = 0.; cap = F[13]
        elif y > L:
            over = y - L; t = 1.; cap = F[14]
        else:
            over = 0.; t = y / L; cap = 0.
        R = _tab(TB, tk, t, ph)
        if over > 0:
            if cap > 0:
                yy = over * cap
                return math.sqrt(rho * rho + yy * yy) - R
            if rho < R: return over
            return math.sqrt((rho - R) ** 2 + over * over)
        return rho - R
    if ty == 1:                      # ball: radii F[12], F[13], F[14] along e1, ax, e2
        a = x / F[12]; b = y / F[13]; c = z / F[14]
        k0 = math.sqrt(a * a + b * b + c * c)
        a /= F[12]; b /= F[13]; c /= F[14]
        k1 = math.sqrt(a * a + b * b + c * c) + 1e-9
        return k0 * (k0 - 1) / k1
    rho = math.sqrt(x * x + z * z)
    ph = math.atan2(z, x)
    C = _tab(TB, tk, 0., ph); r = _tab(TB, tk, 8. / (NT - 1), ph); o = _tab(TB, tk, 16. / (NT - 1), ph); q = _tab(TB, tk, 1., ph)
    return math.sqrt((rho - C) ** 2 + ((y - o) * q) ** 2) - r


@njit(cache=True, fastmath=True)
def _map(px, py, pz, act, n, PF, PI, TB, GF, ng, gd, gi):
    """signed distance + winning prim.  act = active prim ids (n of them).  PI: type, table, group"""
    for g in range(ng): gd[g] = BIG
    for q in range(n):
        i = act[q]
        bx = px - PF[i, 16]; by = py - PF[i, 17]; bz = pz - PF[i, 18]
        r2 = bx * bx + by * by + bz * bz
        g = PI[i, 2]
        a = gd[g]
        rb = PF[i, 19]
        lim = rb + .12
        if r2 > lim * lim:
            db = math.sqrt(r2) - rb
            if db > a: continue
            d = db
            if db < .25: d = _prim(PF[i], PI[i, 0], PI[i, 1], TB, px, py, pz)
        else:
            d = _prim(PF[i], PI[i, 0], PI[i, 1], TB, px, py, pz)
        k = PF[i, 15]
        if a >= BIG:
            gd[g] = d; gi[g] = i
        else:
            if d < a: gi[g] = i
            e = abs(a - d)
            if e < k:
                h = (k - e) / k
                gd[g] = min(a, d) - h * h * k * .25
            else:
                gd[g] = min(a, d)
    D = BIG; w = -1
    for g in range(ng):
        d = gd[g]
        if d >= BIG: continue
        if GF[g, 9] > 0:                                       # torn along a ragged plane: the side the normal points to is gone
            u = px * GF[g, 10] + py * GF[g, 11] + pz * GF[g, 12]
            f1 = u * GF[g, 13]; f1 = abs(f1 - math.floor(f1) - .5) * 2
            f2 = u * GF[g, 13] * 2.37 + .31; f2 = abs(f2 - math.floor(f2) - .5) * 2
            c = px * GF[g, 5] + py * GF[g, 6] + pz * GF[g, 7] - GF[g, 8] + GF[g, 9] * (f1 + .6 * f2 - .8)
            if c > d: d = c
        if w < 0:
            D = d; w = gi[g]; continue
        k = GF[g, 0]
        if k > 0:
            jx = px - GF[g, 1]; jy = py - GF[g, 2]; jz = pz - GF[g, 3]
            u = math.sqrt(jx * jx + jy * jy + jz * jz) / GF[g, 4]
            k = k * (1 - u) if u < 1 else 0.
        if d < D: w = gi[g]
        e = abs(D - d)
        if k > 1e-4 and e < k:
            h = (k - e) / k
            D = min(D, d) - h * h * k * .25
        else:
            D = min(D, d)
    return D, w


@njit(cache=True, fastmath=True)
def tile_lists(PF, sk, np_, eye, R, f, cx, cy, ox, oy, g, w, h, ts, maxp):
    tw = (w + ts - 1) // ts; th = (h + ts - 1) // ts
    cnt = np.zeros((th, tw), np.int32); lst = np.zeros((th, tw, maxp), np.int16)
    z0 = np.full((th, tw), 1e9); z1 = np.full((th, tw), -1e9)
    for i in range(np_):
        if PF[i, 19] <= 0 or sk[i] == 5: continue
        a = PF[i, 16] - eye[0]; b = PF[i, 17] - eye[1]; c = PF[i, 18] - eye[2]; r = PF[i, 19] + .015
        X = R[0, 0] * a + R[0, 1] * b + R[0, 2] * c; Y = R[1, 0] * a + R[1, 1] * b + R[1, 2] * c; Z = R[2, 0] * a + R[2, 1] * b + R[2, 2] * c
        if Z + r < .06: continue
        if Z - r < .06:
            x0 = 0; x1 = tw - 1; y0 = 0; y1 = th - 1
        else:
            q = math.sqrt(X * X + Y * Y + Z * Z)
            rr = f * r / (Z - r) * q / Z
            u = (f * X / Z + cx - ox) * g; v = (f * Y / Z + cy - oy) * g; rr *= g
            x0 = int(math.floor((u - rr) / ts)); x1 = int(math.floor((u + rr) / ts)); y0 = int(math.floor((v - rr) / ts)); y1 = int(math.floor((v + rr) / ts))
            if x0 < 0: x0 = 0
            if y0 < 0: y0 = 0
            if x1 > tw - 1: x1 = tw - 1
            if y1 > th - 1: y1 = th - 1
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                n = cnt[ty, tx]
                if n < maxp:
                    lst[ty, tx, n] = i; cnt[ty, tx] = n + 1
                if Z - r < z0[ty, tx]: z0[ty, tx] = Z - r
                if Z + r > z1[ty, tx]: z1[ty, tx] = Z + r
    return cnt, lst, z0, z1


@njit(cache=True, fastmath=True)
def march(PF, PI, TB, GF, eye, R, f, cx, cy, ox, oy, g, w, h, ts, cnt, lst, z0, z1):
    """-> Z (0 = empty), prim id, normal, t, phi, cavity.  Cone-marched in 4x4 blocks first, then per pixel."""
    ng = GF.shape[0]
    Z = np.zeros((h, w), np.float32); W = np.full((h, w), -1, np.int16); N = np.zeros((h, w, 3), np.float32)
    T = np.zeros((h, w), np.float32); A = np.zeros((h, w), np.float32); C = np.ones((h, w), np.float32)
    gd = np.zeros(ng); gi = np.zeros(ng, np.int64)
    B = 4; fg = f * g
    for bj in range(0, h, B):
        ty = min(bj + 1, h - 1) // ts
        for bi in range(0, w, B):
            tx = min(bi + 1, w - 1) // ts
            n = cnt[ty, tx]
            if n == 0: continue
            act = lst[ty, tx]
            X = ((bi + B * .5) / g + ox - cx) / f; Y = ((bj + B * .5) / g + oy - cy) / f
            d0 = X * R[0, 0] + Y * R[1, 0] + R[2, 0]; d1 = X * R[0, 1] + Y * R[1, 1] + R[2, 1]; d2 = X * R[0, 2] + Y * R[1, 2] + R[2, 2]
            ln = math.sqrt(X * X + Y * Y + 1)
            t = max(z0[ty, tx], .05); t1 = z1[ty, tx]; cone = B * .72 / fg * ln; found = False
            for s in range(60):
                d, wi = _map(eye[0] + d0 * t, eye[1] + d1 * t, eye[2] + d2 * t, act, n, PF, PI, TB, GF, ng, gd, gi)
                if d < cone * t * 1.6:
                    found = True; break
                t += (d - cone * t) * .72 / ln
                if t > t1: break
            if not found: continue
            ts0 = max(t - cone * t * 1.5, .05)
            for j in range(bj, min(h, bj + B)):
                for i in range(bi, min(w, bi + B)):
                    X = ((i + .5) / g + ox - cx) / f; Y = ((j + .5) / g + oy - cy) / f
                    d0 = X * R[0, 0] + Y * R[1, 0] + R[2, 0]; d1 = X * R[0, 1] + Y * R[1, 1] + R[2, 1]; d2 = X * R[0, 2] + Y * R[1, 2] + R[2, 2]
                    t = ts0; hit = False; wi = -1; px = py = pz = 0.; eps = .35 / fg
                    for s in range(70):
                        px = eye[0] + d0 * t; py = eye[1] + d1 * t; pz = eye[2] + d2 * t
                        d, wi = _map(px, py, pz, act, n, PF, PI, TB, GF, ng, gd, gi)
                        if d < eps * t:
                            hit = True; break
                        t += max(d * .78 / ln, eps * t * .6)
                        if t > t1: break
                    if not hit or wi < 0: continue
                    Z[j, i] = t; W[j, i] = wi
                    e = max(.003, 1.2 * t / fg)
                    a, _ = _map(px + e, py - e, pz - e, act, n, PF, PI, TB, GF, ng, gd, gi)
                    b, _ = _map(px - e, py - e, pz + e, act, n, PF, PI, TB, GF, ng, gd, gi)
                    c, _ = _map(px - e, py + e, pz - e, act, n, PF, PI, TB, GF, ng, gd, gi)
                    d, _ = _map(px + e, py + e, pz + e, act, n, PF, PI, TB, GF, ng, gd, gi)
                    nx = a - b - c + d; ny = -a - b + c + d; nz = -a + b - c + d
                    q = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
                    nx /= q; ny /= q; nz /= q
                    N[j, i, 0] = nx; N[j, i, 1] = ny; N[j, i, 2] = nz
                    dx = px - PF[wi, 0]; dy = py - PF[wi, 1]; dz = pz - PF[wi, 2]
                    x = dx * PF[wi, 3] + dy * PF[wi, 4] + dz * PF[wi, 5]; y = dx * PF[wi, 6] + dy * PF[wi, 7] + dz * PF[wi, 8]; z = dx * PF[wi, 9] + dy * PF[wi, 10] + dz * PF[wi, 11]
                    T[j, i] = y / PF[wi, 12] if (PI[wi, 0] == 0 or PI[wi, 0] == 4) else y
                    A[j, i] = math.atan2(z, x)
                    if (i & 1) == 0 and (j & 1) == 0:
                        o = 0.
                        for hh in (.022, .065):
                            dd, _ = _map(px + nx * hh, py + ny * hh, pz + nz * hh, act, n, PF, PI, TB, GF, ng, gd, gi)
                            o += max(0., hh - dd) / hh
                        o = 1 - o / 2
                        C[j, i] = o
                        if i + 1 < w: C[j, i + 1] = o
                        if j + 1 < h:
                            C[j + 1, i] = o
                            if i + 1 < w: C[j + 1, i + 1] = o
    return Z, W, N, T, A, C


@njit(cache=True, fastmath=True)
def raster(V, NV, UV, T3, ida, idb, f, cx, cy, ox, oy, g, Z, W, N, T, A, C, R):
    """z-buffered triangles of a cloth sheet into the clay buffers.  V: camera-space verts, NV: world normals, UV: (v, u) per vertex, ida / idb: prim ids of the outer / inner face"""
    h, w = Z.shape
    for k in range(T3.shape[0]):
        a, b, c = T3[k, 0], T3[k, 1], T3[k, 2]
        if V[a, 2] < .06 or V[b, 2] < .06 or V[c, 2] < .06: continue
        xa = (V[a, 0] / V[a, 2] * f + cx - ox) * g - .5; ya = (V[a, 1] / V[a, 2] * f + cy - oy) * g - .5
        xb = (V[b, 0] / V[b, 2] * f + cx - ox) * g - .5; yb = (V[b, 1] / V[b, 2] * f + cy - oy) * g - .5
        xc = (V[c, 0] / V[c, 2] * f + cx - ox) * g - .5; yc = (V[c, 1] / V[c, 2] * f + cy - oy) * g - .5
        den = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
        if abs(den) < 1e-9: continue
        x0 = int(max(0., math.floor(min(xa, min(xb, xc))))); x1 = int(min(w - 1., math.ceil(max(xa, max(xb, xc))))); y0 = int(max(0., math.floor(min(ya, min(yb, yc))))); y1 = int(min(h - 1., math.ceil(max(ya, max(yb, yc)))))
        for j in range(y0, y1 + 1):
            for i in range(x0, x1 + 1):
                l0 = ((yb - yc) * (i - xc) + (xc - xb) * (j - yc)) / den; l1 = ((yc - ya) * (i - xc) + (xa - xc) * (j - yc)) / den; l2 = 1 - l0 - l1
                if l0 < -.02 or l1 < -.02 or l2 < -.02: continue
                iz = l0 / V[a, 2] + l1 / V[b, 2] + l2 / V[c, 2]
                z = 1 / iz
                if Z[j, i] > 0 and Z[j, i] <= z: continue
                nx = l0 * NV[a, 0] + l1 * NV[b, 0] + l2 * NV[c, 0]; ny = l0 * NV[a, 1] + l1 * NV[b, 1] + l2 * NV[c, 1]; nz = l0 * NV[a, 2] + l1 * NV[b, 2] + l2 * NV[c, 2]
                q = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9; nx /= q; ny /= q; nz /= q
                X = ((i + .5) / g + ox - cx) / f; Y = ((j + .5) / g + oy - cy) / f
                d0 = X * R[0, 0] + Y * R[1, 0] + R[2, 0]; d1 = X * R[0, 1] + Y * R[1, 1] + R[2, 1]; d2 = X * R[0, 2] + Y * R[1, 2] + R[2, 2]
                s = nx * d0 + ny * d1 + nz * d2
                if s > 0: nx = -nx; ny = -ny; nz = -nz
                Z[j, i] = z; W[j, i] = idb if s > 0 else ida; N[j, i, 0] = nx; N[j, i, 1] = ny; N[j, i, 2] = nz
                T[j, i] = l0 * UV[a, 0] + l1 * UV[b, 0] + l2 * UV[c, 0]; A[j, i] = l0 * UV[a, 1] + l1 * UV[b, 1] + l2 * UV[c, 1]; C[j, i] = .95


@njit(cache=True, fastmath=True)
def cast(Z, W, N, PF, PI, TB, GF, eye, R, f, cx, cy, ox, oy, g, lx, ly, lz, maxd, skip, step):
    """hard shadow mask toward the light (world dir).  skip[i]=1 prims that do not cast"""
    h, w = Z.shape; ng = GF.shape[0]; np_ = PF.shape[0]
    S = np.zeros((h, w), np.uint8); act = np.zeros(np_, np.int16)
    gd = np.zeros(ng); gi = np.zeros(ng, np.int64)
    for j in range(0, h, step):
        for i in range(0, w, step):
            if W[j, i] < 0: continue
            nl = N[j, i, 0] * lx + N[j, i, 1] * ly + N[j, i, 2] * lz
            if nl < .02: continue
            X = ((i + .5) / g + ox - cx) / f; Y = ((j + .5) / g + oy - cy) / f; t = Z[j, i]
            px = eye[0] + (X * R[0, 0] + Y * R[1, 0] + R[2, 0]) * t + N[j, i, 0] * .014
            py = eye[1] + (X * R[0, 1] + Y * R[1, 1] + R[2, 1]) * t + N[j, i, 1] * .014
            pz = eye[2] + (X * R[0, 2] + Y * R[1, 2] + R[2, 2]) * t + N[j, i, 2] * .014
            n = 0
            for q in range(np_):
                if skip[q]: continue
                a = PF[q, 16] - px; b = PF[q, 17] - py; c = PF[q, 18] - pz
                u = a * lx + b * ly + c * lz; r = PF[q, 19]
                if u < -r or u > maxd + r: continue
                if a * a + b * b + c * c - u * u > r * r: continue
                act[n] = q; n += 1
            if n == 0: continue
            s = .02
            for m in range(36):
                d, _ = _map(px + lx * s, py + ly * s, pz + lz * s, act, n, PF, PI, TB, GF, ng, gd, gi)
                if d < .0025:
                    for jj in range(j, min(h, j + step)):
                        for ii in range(i, min(w, i + step)): S[jj, ii] = 1
                    break
                s += max(d * .8, .008)
                if s > maxd: break
    return S


@njit(cache=True, fastmath=True)
def drop(O, DD, maxt, PF, PI, TB, GF, use):
    """cast rays O[q] + t DD[q] onto the prims with use[i]=1 -> t (or -1) and prim"""
    n = len(O); out = np.full(n, -1.); wo = np.full(n, -1, np.int64); ng = GF.shape[0]; gd = np.zeros(ng); gi = np.zeros(ng, np.int64)
    act = np.zeros(len(PF), np.int16)
    for q in range(n):
        t = 0.; m = 0
        d = DD[q]
        for i in range(len(PF)):                              # only the prims this ray can meet
            if not use[i]: continue
            a = PF[i, 16] - O[q, 0]; b = PF[i, 17] - O[q, 1]; c = PF[i, 18] - O[q, 2]; u = a * d[0] + b * d[1] + c * d[2]; r = PF[i, 19] + .03
            if a * a + b * b + c * c - u * u <= r * r: act[m] = i; m += 1
        if m == 0: continue
        for s in range(90):
            D, w = _map(O[q, 0] + d[0] * t, O[q, 1] + d[1] * t, O[q, 2] + d[2] * t, act, m, PF, PI, TB, GF, ng, gd, gi)
            if D < .0006:
                out[q] = t; wo[q] = w; break
            t += max(D * .8, .0005)
            if t > maxt: break
    return out, wo


class Clay:
    """collects primitives for one posed figure"""

    def __init__(self):
        self.F = []; self.I = []; self.TB = []; self.lab = []; self.mat = []; self.name = []; self.groups = {}; self.gf = []; self.rule = []; self.cast = []; self.mg = []; self.sheets = []

    def group(self, name, joint=None, R=.2, k=0.):
        if name not in self.groups:
            self.groups[name] = len(self.gf)
            j = (0, 0, 0) if joint is None else joint
            self.gf.append([k, j[0], j[1], j[2], R] + [0.] * 9)
        return self.groups[name]

    def carve(self, group, o, n, t, amp=.006, freq=16.):
        """tear a group along the plane through o with normal n (removed side); t = direction along the tear"""
        n = np.asarray(n, float) / np.linalg.norm(n); t = np.asarray(t, float) / np.linalg.norm(t); g = self.gf[self.group(group)]; g[5:14] = [*n, float(n @ np.asarray(o, float)), amp, *t, freq]

    def _add(self, ty, o, e1, ax, e2, par, k, tab, bc, br, group, label, mat, name, rule, cast, mg):
        row = np.zeros(NF); row[0:3] = o; row[3:6] = e1; row[6:9] = ax; row[9:12] = e2; row[12:12 + len(par)] = par; row[15] = k; row[16:19] = bc; row[19] = br
        self.F.append(row); self.I.append([ty, tab, self.group(group)]); self.lab.append(label); self.mat.append(mat); self.name.append(name or label); self.rule.append(rule); self.cast.append(cast); self.mg.append(mg)
        return len(self.F) - 1

    def loft(self, a, b, e1, R, group, label, mat, cap=(1., 1.), k=.02, side=1., name=None, rule=None, cast=True, mg=None):
        """R: table (NT, NP) metres; e1 = phi 0 direction; e2 = side * (axis x e1) (phi 90)"""
        a = np.asarray(a, float); b = np.asarray(b, float); d = b - a; L = np.linalg.norm(d) + 1e-9; ax = d / L
        e1 = np.asarray(e1, float); e1 = e1 - ax * (e1 @ ax); n = np.linalg.norm(e1)
        if n < 1e-6: e1 = np.cross(ax, [.3, .5, .8]); n = np.linalg.norm(e1)
        e1 = e1 / n; e2 = np.cross(ax, e1) * side
        R = np.asarray(R, np.float64); assert R.shape == (NT, NP), R.shape
        self.TB.append(R)
        rm = float(R.max()); ext = rm / max(min([c for c in cap if c > 0] or [1.]), .3)
        i = self._add(0, a, e1, ax, e2, (L, cap[0], cap[1]), k, len(self.TB) - 1, (a + b) / 2, math.hypot(L / 2 + (ext if max(cap) > 0 else 0) * .75, rm), group, label, mat, name, rule, cast, mg)
        self.F[i][20] = rm; return i

    def ball(self, c, r, group, label, mat, ax=(0, 0, 1), e1=(1, 0, 0), k=.03, name=None, rule=None, cast=True, side=1., mg=None):
        ax = np.asarray(ax, float); ax = ax / np.linalg.norm(ax); e1 = np.asarray(e1, float); e1 = e1 - ax * (e1 @ ax); e1 /= np.linalg.norm(e1) + 1e-9
        r = np.broadcast_to(np.asarray(r, float), (3,))
        return self._add(1, np.asarray(c, float), e1, ax, np.cross(ax, e1) * side, tuple(r), k, 0, c, float(r.max()), group, label, mat, name, rule, cast, mg)

    def egg(self, c, ax, e1, r, group, label, mat, k=.015, skew=0., box=.0, bow=0., name=None, rule=None, cast=True, mg=None):
        """muscle: radii r = (across = e1, along = ax, thick = e2 = ax x e1), skew > 0 fattens the far end, box 0..1, bow bends it"""
        ax = np.asarray(ax, float); ax = ax / np.linalg.norm(ax); e1 = np.asarray(e1, float); e1 = e1 - ax * (e1 @ ax); n = np.linalg.norm(e1)
        if n < 1e-6: e1 = np.cross(ax, [.3, .5, .8]); n = np.linalg.norm(e1)
        e1 = e1 / n; e2 = np.cross(ax, e1); r = np.asarray(r, float)
        br = float(max(r[1], r[0] * (1 + abs(skew)), r[2] * (1 + abs(skew)))) * (1 + .22 * box) + abs(bow)
        i = self._add(3, np.asarray(c, float), e1, ax, e2, tuple(r), k, 0, c, br, group, label, mat, name, rule, cast, mg)
        self.F[i][21] = skew; self.F[i][22] = box; self.F[i][23] = bow; return i

    def cap(self, a, b, r0, r1, group, label, mat, k=.01, name=None, rule=None, cast=True, mg=None, e1=None):
        """tapered capsule"""
        a = np.asarray(a, float); b = np.asarray(b, float); d = b - a; L = np.linalg.norm(d) + 1e-9; ax = d / L
        e1 = np.cross(ax, [.31, .52, .79]) if e1 is None else np.asarray(e1, float); e1 = e1 - ax * (e1 @ ax); e1 /= np.linalg.norm(e1) + 1e-9
        return self._add(4, a, e1, ax, np.cross(ax, e1), (L, r0, r1), k, 0, (a + b) / 2, L / 2 + max(r0, r1), group, label, mat, name, rule, cast, mg)

    def ring(self, c, ax, e1, C, r, off, squash, group, label, mat, k=.02, name=None, rule=None, cast=True, mg=None):
        ax = np.asarray(ax, float); ax = ax / np.linalg.norm(ax); e1 = np.asarray(e1, float); e1 = e1 - ax * (e1 @ ax); e1 /= np.linalg.norm(e1) + 1e-9
        tb = np.zeros((NT, NP)); bc = lambda v: np.broadcast_to(np.asarray(v, float), (NP,))
        tb[0:5] = bc(C); tb[5:12] = bc(r); tb[12:20] = bc(off); tb[20:] = bc(squash)
        self.TB.append(tb)
        br = float(np.max(bc(C) + bc(r)) + np.max(np.abs(bc(off))))
        return self._add(2, np.asarray(c, float), e1, ax, np.cross(ax, e1), (1., 1., 1.), k, len(self.TB) - 1, c, br, group, label, mat, name, rule, cast, mg)

    def sheet(self, Q, label, mat, back=None, hole=None, rule=None, lift=0.):
        """thin cloth: Q = (nv, nu, 3) world points.  hole = bool (nv-1, nu-1) of missing cells.  (t, p) of a rule = (v, u) in 0..1"""
        Q = np.asarray(Q, float); nv, nu = Q.shape[:2]; c = Q.reshape(-1, 3).mean(0); r = float(np.linalg.norm(Q.reshape(-1, 3) - c, axis=1).max())
        ida = self._add(5, c, (1, 0, 0), (0, 1, 0), (0, 0, 1), (1., 1., 1.), 0., 0, c, r, 'sheet', label, mat, label, rule, False, None)
        idb = self._add(5, c, (1, 0, 0), (0, 1, 0), (0, 0, 1), (1., 1., 1.), 0., 0, c, r, 'sheet', label, back or mat, label, None, False, None)
        du = np.gradient(Q, axis=1); dv = np.gradient(Q, axis=0); n = np.cross(du, dv); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
        i = np.arange(nv * nu).reshape(nv, nu); a, b, c_, d = i[:-1, :-1], i[:-1, 1:], i[1:, 1:], i[1:, :-1]; ok = np.ones(a.shape, bool) if hole is None else ~np.asarray(hole, bool)
        tri = np.concatenate([np.stack([a[ok], b[ok], c_[ok]], -1), np.stack([a[ok], c_[ok], d[ok]], -1)]).astype(np.int64)
        uv = np.stack(np.meshgrid(np.linspace(0, 1, nv), np.linspace(0, 1, nu), indexing='ij'), -1).reshape(-1, 2)
        self.sheets.append((Q.reshape(-1, 3), n.reshape(-1, 3), uv, tri, ida, idb)); return ida

    def pack(self):
        if not self.TB: self.TB.append(np.zeros((NT, NP)))
        # prims of one group must be evaluated in the order they were added; sort nothing
        self.PF = np.array(self.F); self.PI = np.array(self.I, np.int64); self.T3 = np.array(self.TB); self.GF = np.array(self.gf, float)
        return self

    def bbox(self, cam, pad=6):
        c = cam.cam(self.PF[:, 16:19]); r = self.PF[:, 19]
        x0 = y0 = 1e9; x1 = y1 = -1e9
        for (X, Y, Z), q in zip(c, r):
            if Z + q < .06: continue
            zz = max(Z - q * .8, .12)
            for s in (-1, 1):
                u = (X + s * q) / zz * cam.f + cam.cx if (X + s * q) * s > 0 else (X + s * q) / (Z + q * .8) * cam.f + cam.cx
                v = (Y + s * q) / zz * cam.f + cam.cy if (Y + s * q) * s > 0 else (Y + s * q) / (Z + q * .8) * cam.f + cam.cy
                x0 = min(x0, u); x1 = max(x1, u); y0 = min(y0, v); y1 = max(y1, v)
        m = 140
        return (int(max(-m, x0 - pad)), int(max(-m, y0 - pad)), int(min(cam.w + m, x1 + pad)), int(min(cam.h + m, y1 + pad)))

    def render(self, cam, box, g, ts=10, maxp=72):
        x0, y0, x1, y1 = box; w = max(4, int(math.ceil((x1 - x0) * g))); h = max(4, int(math.ceil((y1 - y0) * g)))
        eye = cam.eye.astype(float); R = cam.R.astype(float)
        cnt, lst, z0, z1 = tile_lists(self.PF, np.ascontiguousarray(self.PI[:, 0]), len(self.PF), eye, R, cam.f, cam.cx, cam.cy, float(x0), float(y0), g, w, h, ts, maxp)
        out = march(self.PF, self.PI, self.T3, self.GF, eye, R, cam.f, cam.cx, cam.cy, float(x0), float(y0), g, w, h, ts, cnt, lst, z0, z1)
        for Q, n, uv, tri, ida, idb in self.sheets: raster(np.ascontiguousarray(cam.cam(Q)), n, uv, tri, ida, idb, cam.f, cam.cx, cam.cy, float(x0), float(y0), g, *out, R)
        return out

    def shadow(self, cam, box, g, buf, L, maxd=1.2, step=2):
        Z, W, N = buf[:3]; L = np.asarray(L, float); L = L / np.linalg.norm(L)
        skip = np.array([0 if c else 1 for c in self.cast], np.uint8)
        return cast(Z, W, N, self.PF, self.PI, self.T3, self.GF, cam.eye.astype(float), cam.R.astype(float), cam.f, cam.cx, cam.cy, float(box[0]), float(box[1]), g, L[0], L[1], L[2], maxd, skip, step)

    def drop(self, O, d, maxt=.6, labels=None, names=None):
        """points O dropped along direction d onto the surface of the chosen labels -> (world points, hit mask)"""
        use = np.array([0 if self.I[j][0] == 5 else 1 if ((labels is None or l in labels) and (names is None or n in names)) else 0 for j, (l, n) in enumerate(zip(self.lab, self.name))], np.uint8)
        O = np.atleast_2d(np.asarray(O, float)); d = np.asarray(d, float); d = np.broadcast_to(d, O.shape); d = np.ascontiguousarray(d / np.linalg.norm(d, axis=-1, keepdims=True))
        t, w = drop(np.ascontiguousarray(O), d, maxt, self.PF, self.PI, self.T3, self.GF, use)
        return O + d * np.maximum(t, 0)[:, None], t >= 0

    def point(self, i, t, phi, lift=0.):
        """world point(s) on loft i at (t, phi)"""
        F = self.PF[i] if hasattr(self, 'PF') else self.F[i]; tb = self.TB[self.I[i][1]]
        t = np.asarray(t, float); phi = np.asarray(phi, float)
        u = np.clip(t, 0, 1) * (NT - 1); a = np.clip(u.astype(int), 0, NT - 2); fu = u - a
        v = (phi % TWO_PI) * NP / TWO_PI; b = v.astype(int) % NP; fv = v - np.floor(v); b2 = (b + 1) % NP
        r = (tb[a, b] * (1 - fv) + tb[a, b2] * fv) * (1 - fu) + (tb[a + 1, b] * (1 - fv) + tb[a + 1, b2] * fv) * fu + lift
        return F[0:3] + np.multiply.outer(t * F[12], F[6:9]) + np.multiply.outer(r * np.cos(phi), F[3:6]) + np.multiply.outer(r * np.sin(phi), F[9:12])


# ------------------------------------------------------------ radius tables
TT = np.linspace(0, 1, NT)[:, None] * np.ones((1, NP))
PH = np.ones((NT, 1)) * (np.arange(NP) * TWO_PI / NP)[None]


def prof(t, stations):
    """smooth interpolation of (t, value) stations at array t"""
    s = np.asarray(stations, float); x, y = s[:, 0], s[:, 1]
    m = np.gradient(y, x) if len(x) > 2 else np.full(len(x), (y[-1] - y[0]) / (x[-1] - x[0]))
    i = np.clip(np.searchsorted(x, t, side='right') - 1, 0, len(x) - 2)
    h = x[i + 1] - x[i]; u = np.clip((t - x[i]) / h, 0, 1)
    return (2 * u ** 3 - 3 * u ** 2 + 1) * y[i] + (u ** 3 - 2 * u ** 2 + u) * h * m[i] + (-2 * u ** 3 + 3 * u ** 2) * y[i + 1] + (u ** 3 - u ** 2) * h * m[i + 1]


def table(st, n=2.0, scale=1.0):
    """st = list of (t, front, out, back, in) radii -> R table (superellipse quadrants, exponent n)"""
    s = np.asarray(st, float); t = TT[:, 0]
    f, o, b, i = (prof(t, s[:, [0, k]]) for k in (1, 2, 3, 4))
    c, q = np.cos(PH), np.sin(PH)
    A = np.where(c >= 0, f[:, None], b[:, None]); B = np.where(q >= 0, o[:, None], i[:, None])
    return scale * ((np.abs(c) / A) ** n + (np.abs(q) / B) ** n) ** (-1. / n)


def wrap(p):
    return (p + math.pi) % TWO_PI - math.pi


def bump(x, w):
    return np.clip(1 - np.abs(x) / w, 0, 1)


def ridge(T, P, t0, p0, skew, w, ln, amp):
    """one angular fold ridge: centred at (t0, p0), running along phi with slope skew (dt per rad), half width w (in t), half length ln (rad)"""
    dp = wrap(P - p0); d = T - t0 - skew * dp
    return amp * bump(d, w) * bump(dp, ln) ** .7
