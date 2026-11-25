"""The planet from above the air (km).  Painted sphere: hard coast shapes, clouds with lit edges, coast lights, dawn line, air glow, aurora curtains, stars.
    c = orbit_cam(alt=420, pitch=-14, yaw=0, roll=0, focal=1100, pos=(0,0))     # camera above the story coast (lon, lat offsets in degrees in pos), looks along +Y (north) when yaw=0
    im = earth(c, f, sun=sun_dir(elev=-9, az=80))                    # full frame with stars; sun elevation in degrees at the story coast, azimuth from north, east = 90
    im = aurora(im, c, f, amt=1.)                                    # curtains north of the coast
    xy, z = c.project(P(lon, lat, alt_km)[None])                     # put a light / a beam / a moon anywhere
    im = stars(im, c, amt=1., mask=sky_mask)"""
import numpy as np, cv2, math
from numba import njit, prange
from . import W, H
from .paint import f32, rgb, smooth, ramp, fbm, grid, _vn
from .cam import Camera3D
from .sky_vol import vn3, _ss, _h3

RE = 6371.
NIGHT = ['#081033', '#111a40', '#22336e', '#4a66b0', '#a3bbec', '#ffdb9a', '#ff9d45']       # ocean, land, cloud body, cloud lit, cloud crest, lamps, lamp halo
DAY = ['#1b5f9c', '#7c8a58', '#b9a77a', '#9fc0dc', '#ffffff', '#2f86b8', '#4f7a4c']        # ocean, land, dry land, cloud shade, cloud lit, shallow, forest


def P(lon, lat, alt=0.):
    """story coast = (0, 0); x east, y north, z up at the coast; planet centre at (0,0,-RE)"""
    a, b = math.radians(lon), math.radians(lat); r = RE + alt; return np.array([r * math.sin(a) * math.cos(b), r * math.sin(b), r * math.cos(a) * math.cos(b) - RE])


def sun_dir(elev=-9., az=80.):
    e, a = math.radians(elev), math.radians(az); return np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)])


def orbit_cam(alt=420., pitch=-14., yaw=0., roll=0., focal=1100., pos=(0., 0.), cx=W / 2, cy=H / 2):
    e = P(pos[0], pos[1], alt); up = e + [0, 0, RE]; up /= np.linalg.norm(up); n = np.array([0, 1., 0]) - up * up[1]; n /= np.linalg.norm(n); ea = np.cross(n, up); y, p = math.radians(yaw), math.radians(pitch)
    d = (n * math.cos(y) + ea * math.sin(y)) * math.cos(p) + up * math.sin(p); c = Camera3D(e, e + d, focal, cx=cx, cy=cy); rt = np.cross(d, up); rt /= np.linalg.norm(rt); u2 = np.cross(rt, d); r = math.radians(roll); ca, sa = math.cos(r), math.sin(r)
    rt, u2 = rt * ca + u2 * sa, u2 * ca - rt * sa; c.R = np.stack([rt, -u2, d]); return c


@njit(cache=True, fastmath=True)
def _fb(x, y, z, l, n, s):
    v = 0.0; a = 1.0; t = 0.0
    for o in range(n):
        v += a * (vn3(x / l + 7.3 * o, y / l + 3.1 * o, z / l + 5.7 * o, s + o) - .5); t += a; a *= .56; l *= .47
    return v / t * 2


@njit(parallel=True, cache=True, fastmath=True)
def _earth(out, M, eye, Rm, f, cx, cy, S, N, D, t, lamps, cy_, air, airc, detail):
    h, w = M.shape
    for y_ in prange(h):
        y = np.int64(y_)
        for x in range(w):
            dx = (x + .5 - cx) / f; dy = (y + .5 - cy) / f; rx = Rm[0, 0] * dx + Rm[1, 0] * dy + Rm[2, 0]; ry = Rm[0, 1] * dx + Rm[1, 1] * dy + Rm[2, 1]; rz = Rm[0, 2] * dx + Rm[1, 2] * dy + Rm[2, 2]
            l = math.sqrt(rx * rx + ry * ry + rz * rz); rx /= l; ry /= l; rz /= l
            ox = eye[0]; oy = eye[1]; oz = eye[2] + RE; b = ox * rx + oy * ry + oz * rz; q = ox * ox + oy * oy + oz * oz
            c2 = q - b * b                                                        # squared closest approach
            disc = RE * RE - c2; alt0 = math.sqrt(q) - RE
            # sun on the limb
            if disc > 0 and b < 0:
                tt = -b - math.sqrt(disc); X = ox + rx * tt; Y = oy + ry * tt; Z = oz + rz * tt; nx = X / RE; ny = Y / RE; nz = Z / RE; mu = -(nx * rx + ny * ry + nz * rz); sd = nx * S[0] + ny * S[1] + nz * S[2]
                px = tt / f; no = 4 if px > 2.5 else (6 if px > .6 else (7 if px > .25 else 9))
                # land: story coast runs through (0,0); sea to the south and east
                big = _fb(X, Y, Z, 2300., 3, 11) * 1.0 + _fb(X, Y, Z, 520., no, 17) * .62; bias = .46 * (Y * .9 - X * .35) / 900.
                if bias > .5: bias = .5
                if bias < -.5: bias = -.5
                ld = big * .62 + bias * math.exp(-(X * X + Y * Y) / (2600. * 2600.)) + .05
                land = _ss(-.008, .008, ld); shelf = _ss(-.10, -.0, ld)
                # clouds: warped, with two cyclones
                ux = X; uy = Y
                for k in range(2):
                    ex = X - (-1150., 1500.)[k]; ey = Y - (-900., 1500.)[k]; rr = math.sqrt(ex * ex + ey * ey); an = (1.7, -2.1)[k] * math.exp(-(rr / (560., 760.)[k]) ** 1.3); ca = math.cos(an); sa = math.sin(an); qx = ex * ca - ey * sa; qy = ex * sa + ey * ca; ux += (qx - ex); uy += (qy - ey)
                wx = _fb(ux, uy, Z, 900., 2, 41) * 330.; wy = _fb(ux, uy, Z, 900., 2, 47) * 330.
                cl = _fb(ux + wx - t * .6, (uy + wy) * 1.35, Z, 760., no, 31); c_s = _fb(ux + wx - t * .6 - S[0] * 40., (uy + wy) * 1.35 - S[1] * 40., Z, 760., no - 1, 31)
                cy2 = cl * .75 + _fb(ux + wx * .4, uy * 1.1, Z, 150., max(2, no - 2), 53) * .38
                ca_ = _ss(.15, .185, cy2); cb = _ss(.35, .41, cy2); ce = _ss(.02, .075, cl - c_s)
                day = _ss(-.05, .16, sd); tw = math.exp(-((sd - .015) / .060) ** 2)
                # day paint
                r0 = D[0, 0]; g0 = D[0, 1]; b0 = D[0, 2]
                r0 += (D[5, 0] - r0) * shelf * .75; g0 += (D[5, 1] - g0) * shelf * .75; b0 += (D[5, 2] - b0) * shelf * .75
                dry = _ss(.0, .12, _fb(X, Y, Z, 700., 3, 61)); lr = D[1, 0] + (D[2, 0] - D[1, 0]) * dry; lg = D[1, 1] + (D[2, 1] - D[1, 1]) * dry; lb = D[1, 2] + (D[2, 2] - D[1, 2]) * dry; fo = _ss(.05, .10, _fb(X, Y, Z, 260., 4, 67)); lr += (D[6, 0] - lr) * fo; lg += (D[6, 1] - lg) * fo; lb += (D[6, 2] - lb) * fo
                r0 += (lr - r0) * land; g0 += (lg - g0) * land; b0 += (lb - b0) * land
                sh = _ss(.14, .24, cy2 - .06 + (cl - c_s) * .0) * 0.0
                kc = .55 + .45 * ce; cr = D[3, 0] + (D[4, 0] - D[3, 0]) * max(cb, ce) ; cg = D[3, 1] + (D[4, 1] - D[3, 1]) * max(cb, ce); cbv = D[3, 2] + (D[4, 2] - D[3, 2]) * max(cb, ce)
                r0 += (cr - r0) * ca_; g0 += (cg - g0) * ca_; b0 += (cbv - b0) * ca_
                # night paint
                r1 = N[0, 0] + (N[1, 0] - N[0, 0]) * land; g1 = N[0, 1] + (N[1, 1] - N[0, 1]) * land; b1 = N[0, 2] + (N[1, 2] - N[0, 2]) * land
                sk = land * 0.0
                # lamps: towns along the coast and the valleys
                if lamps > 0 and ld > -.004:
                    coast = math.exp(-max(0., ld) / .035); ridge = 1 - abs(2 * _fb(X, Y, Z, 340., 4, 71)); ridge = _ss(.80, .97, ridge) * math.exp(-max(0., ld) / .22); town = max(coast * _ss(.0, .35, _fb(X, Y, Z, 210., 3, 73) + .10), ridge * .8)
                    for k in range(3):
                        cs = (7.0, 14.0, 30.0)[k]
                        if cs < px * 1.1: continue
                        ia = math.floor(X / cs); ib = math.floor(Y / cs); ic = math.floor(Z / cs) if k < 0 else 0.0
                        hh = _h3(int(ia), int(ib), 0, 91 + k)
                        if hh < town * (.55, .34, .22)[k]:
                            fa = X / cs - ia - .5 - (_h3(int(ia), int(ib), 1, 92) - .5) * .5; fb_ = Y / cs - ib - .5 - (_h3(int(ia), int(ib), 2, 93) - .5) * .5; d2 = math.sqrt(fa * fa + fb_ * fb_) * cs; sz = (1.1, 2.0, 3.6)[k] * (.5 + hh / max(town, .05) * .8)
                            e = _ss(sz + px * .8, sz * .4, d2) * lamps * (1 - day) * (.35 + .65 * (1 - ca_ * .8)); r1 += (N[5, 0] * 1.15 - r1) * e; g1 += (N[5, 1] * 1.1 - g1) * e; b1 += (N[5, 2] - b1) * e
                    gl = town * .30 * lamps * (1 - day); r1 += N[6, 0] * gl; g1 += N[6, 1] * gl; b1 += N[6, 2] * gl * .8
                ce2 = ce; nc = N[2, 0] + (N[3, 0] - N[2, 0]) * max(cb * .7, ce); ng = N[2, 1] + (N[3, 1] - N[2, 1]) * max(cb * .7, ce); nb_ = N[2, 2] + (N[3, 2] - N[2, 2]) * max(cb * .7, ce); ck = ca_ * .88
                r1 += (nc - r1) * ck; g1 += (ng - g1) * ck; b1 += (nb_ - b1) * ck; e = ce * ca_ * (.25 + .75 * cb) * .75; r1 += (N[4, 0] - r1) * e; g1 += (N[4, 1] - g1) * e; b1 += (N[4, 2] - b1) * e
                r2 = r1 + (r0 - r1) * day; g2 = g1 + (g0 - g1) * day; b2 = b1 + (b0 - b1) * day
                # dawn line: rose on cloud crests, amber haze
                e = tw * (.07 + .10 * land + .78 * ca_ * (.35 + .65 * max(ce, cb))); r2 += (1.0 - r2) * e * .95; g2 += (.66 - g2) * e * .82; b2 += (.56 - b2) * e * .70
                # air between
                g = (1 - mu); g = g * g * g * g; e = g * (.20 + .75 * _ss(-.22, .2, sd)) * air; aw = _ss(-.10, .10, sd); ar = airc[0] + (airc[6] - airc[0]) * tw * .8; ag = airc[1] + (airc[7] - airc[1]) * tw * .8; ab = airc[2] + (airc[8] - airc[2]) * tw * .8
                r2 += (ar * (.30 + .9 * aw + tw) - r2) * min(e, .92); g2 += (ag * (.30 + .9 * aw + tw) - g2) * min(e, .92); b2 += (ab * (.36 + .9 * aw + tw * .8) - b2) * min(e, .92)
                out[y, x, 0] = r2; out[y, x, 1] = g2; out[y, x, 2] = b2; M[y, x] = 1.0
            else:
                hgt = math.sqrt(max(c2, 0.)) - RE if b < 0 else alt0
                if hgt < 0: hgt = 0.0
                tb = -b if b < 0 else 0.0; X = ox + rx * tb; Y = oy + ry * tb; Z = oz + rz * tb; rr = math.sqrt(X * X + Y * Y + Z * Z); sd = (X * S[0] + Y * S[1] + Z * S[2]) / rr
                aw = _ss(-.22, .16, sd); tw = math.exp(-((sd + .03) / .13) ** 2)
                e1 = math.exp(-hgt / 13.) * (.35 + 1.1 * aw + .7 * tw); e2 = math.exp(-hgt / 46.) * (.14 + .75 * aw + .25 * tw); e3 = math.exp(-((hgt - 92.) / 7.) ** 2) * (1 - aw) * .34 + math.exp(-((hgt - 90.) / 24.) ** 2) * (1 - aw) * .10
                r2 = e2 * airc[0] + e1 * (airc[3] + (airc[6] - airc[3]) * min(1., tw * 1.2)) + e3 * airc[9]; g2 = e2 * airc[1] + e1 * (airc[4] + (airc[7] - airc[4]) * min(1., tw * 1.2)) + e3 * airc[10]; b2 = e2 * airc[2] + e1 * (airc[5] + (airc[8] - airc[5]) * min(1., tw * 1.2)) + e3 * airc[11]
                out[y, x, 0] += r2; out[y, x, 1] += g2; out[y, x, 2] += b2; M[y, x] = 0.0


def earth(c, f=0, sun=None, lamps=1., air=1., space='#02030a', starry=1., night=NIGHT, day=DAY, aircol=('#2f7dff', '#bfe6ff', '#ff9a4a', '#8fe08a'), ss=1):
    """-> (1080,1920,3).  aircol = blue veil, thin white-blue skin, dawn skin, night air glow"""
    S = sun_dir() if sun is None else np.asarray(sun, float); S = S / np.linalg.norm(S); out = np.zeros((c.h, c.w, 3), f32) + rgb(space); M = np.zeros((c.h, c.w), f32)
    if starry > 0: out = stars(out, c, starry)
    A = np.concatenate([rgb(q) for q in aircol]).astype(f32)
    _earth(out, M, np.asarray(c.eye, np.float64), c.R.astype(np.float64), float(c.f), float(c.cx), float(c.cy), S.astype(np.float64), np.stack([rgb(q) for q in night]).astype(f32), np.stack([rgb(q) for q in day]).astype(f32), float(f), float(lamps), 0., float(air), A, 1.)
    return out


_ST = {}
def _catalog(n, seed):
    if (n, seed) not in _ST:
        r = np.random.default_rng(seed); v = r.normal(0, 1, (n, 3)); ax = np.array([.45, .30, .84]); ax /= np.linalg.norm(ax); k = r.random(n) < .42; v[k] -= np.outer(v[k] @ ax, ax) * .86; v /= np.linalg.norm(v, axis=1)[:, None]          # 42 % crowd the galactic belt
        m = 6.6 + np.log10(np.maximum(r.random(n), 1e-5)) / .50; m = np.maximum(m, -1.2); tc = r.random(n); col = ramp(tc.astype(f32), [(0, '#9fbcff'), (.35, '#d5e2ff'), (.65, '#fffaf2'), (.86, '#ffe2b8'), (1, '#ffb488')]); _ST[n, seed] = (v, m, col, ax)
    return _ST[n, seed]


@njit(cache=True, fastmath=True)
def _splat_stars(im, X, Y, A, S, C):
    h, w = im.shape[:2]
    for i in range(X.shape[0]):
        s = S[i]; r = int(s * 3.2) + 1; x0 = int(X[i]); y0 = int(Y[i])
        for y in range(max(0, y0 - r), min(h - 1, y0 + r) + 1):
            for x in range(max(0, x0 - r), min(w - 1, x0 + r) + 1):
                d = ((x + .5 - X[i]) ** 2 + (y + .5 - Y[i]) ** 2) / (s * s); e = A[i] * math.exp(-d * .9)
                if e > .002:
                    for k in range(3): im[y, x, k] += C[i, k] * e


def stars(im, c, amt=1., n=9000, seed=5, belt=.07, mask=None):
    """stars at infinity with believable magnitudes (each magnitude 2.5x fainter and 3x more numerous), plus the faint galactic belt"""
    v, m, col, ax = _catalog(n, seed); q = v @ c.R.T; ok = q[:, 2] > .05; x = q[:, 0] / np.maximum(q[:, 2], .05) * c.f + c.cx; y = q[:, 1] / np.maximum(q[:, 2], .05) * c.f + c.cy; ok &= (x > -8) & (x < c.w + 8) & (y > -8) & (y < c.h + 8)
    fx = 10 ** (-.4 * (m[ok] - 3.6)) ; A = np.clip(fx, .02, 1.25) ** .62 * amt * 1.05; S = .62 + .55 * np.clip(np.log10(np.maximum(fx, .1)) + .6, 0, 2.2); L = np.zeros_like(im) if mask is not None else im
    _splat_stars(L, x[ok].astype(f32), y[ok].astype(f32), A.astype(f32), S.astype(f32), col[ok].astype(f32))
    if belt > 0:
        d = c.rays(c.h // 4, c.w // 4); gy, gx = np.mgrid[0:c.h // 4, 0:c.w // 4].astype(f32); d = np.stack([(gx * 4 + 2 - c.cx) / c.f, (gy * 4 + 2 - c.cy) / c.f, np.ones_like(gx)], -1) @ c.R.astype(f32); d /= np.linalg.norm(d, axis=-1)[..., None]; b = d @ ax.astype(f32)
        u = np.arctan2(d @ np.cross(ax, [0, 0, 1]).astype(f32), d[..., 2]); t = fbm(512, 1024, (90, 34, 12), (1, .6, .4), 77, aspect=2.2); nn = cv2.remap(t, ((u / 6.2832 + .5) * 1023).astype(f32), ((b + 1) * 255).astype(f32), cv2.INTER_LINEAR); g = np.exp(-(b / .13) ** 2) * smooth(.0, 1.6, nn) * (1 - .75 * np.exp(-((b - .02) / .035) ** 2) * smooth(-.2, .8, nn)); g = cv2.resize(g, (c.w, c.h), interpolation=cv2.INTER_LINEAR); L += (g * belt * amt)[..., None] * rgb('#8fa4f0')
    if mask is not None: im = im + L * mask[..., None]
    return im


@njit(cache=True, fastmath=True)
def _sp(im, X, Y, I, C):
    h, w = im.shape[:2]
    for i in range(X.shape[0]):
        x = X[i] - .5; y = Y[i] - .5; x0 = int(math.floor(x)); y0 = int(math.floor(y))
        if x0 < 0 or y0 < 0 or x0 >= w - 1 or y0 >= h - 1: continue
        fx = x - x0; fy = y - y0
        for (a, b, k) in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            e = I[i] * k
            for q in range(3): im[y0 + b, x0 + a, q] += C[i, q] * e


CURT = [dict(lat=7.2, lon=(-78, 78), amp=1.7, seed=1, I=1., h=(98, 520)), dict(lat=10.0, lon=(-74, 82), amp=1.3, seed=2, I=.7, h=(98, 420)), dict(lat=13.4, lon=(-84, 84), amp=1.8, seed=3, I=.5, h=(98, 360))]


def aurora(im, c, f=0, amt=1., curtains=CURT, cols=('#e8fff0', '#57f2a0', '#2fb7c8', '#8a4dff', '#ff5fae'), mask=None):
    """curtains of light hanging over the north: bright cut lower hem, combed rays, violet tips; slow drift.  earth occludes by itself"""
    hb = (c.h // 2, c.w // 2); B = np.zeros(hb + (3,), f32)
    for q in curtains:
        nu, nv = 4600, 56; u = np.linspace(0, 1, nu); lo = q['lon'][0] + (q['lon'][1] - q['lon'][0]) * u; s = q['seed']; nz = lambda l, o, sp: np.array([_vn(a / l + 100.5 + f * sp, s * 7.7 + o, 5) for a in lo]) - .5
        la = q['lat'] + q['amp'] * (np.sin(lo * .22 + s + f * .010) + 1.6 * nz(5., 1, .006) + .9 * nz(1.6, 2, .012)); fold = .95 * np.sin(lo * 1.25 + 2 * s + f * .03) * (nz(7., 3, .004) + .5); lo2 = lo + fold
        ray = np.clip(.50 + 1.3 * nz(.16, 4, .06) + 1.2 * nz(.75, 5, .03), 0, 1.6); env = np.clip(.15 + 1.7 * (nz(6., 6, .01) + .25), 0, 1) * np.clip(np.minimum(u, 1 - u) / .22, 0, 1) ** 1.6; top = q['h'][0] + (q['h'][1] - q['h'][0]) * (.35 + .65 * np.clip(ray, 0, 1.3)) * (.6 + .8 * (nz(3., 7, .015) + .5))
        a, b = np.radians(lo2), np.radians(la); D = np.stack([np.sin(a) * np.cos(b), np.sin(b), np.cos(a) * np.cos(b)], -1); v = np.linspace(0, 1, nv) ** 1.25; alt = q['h'][0] + (top - q['h'][0])[:, None] * v[None]; Pw = D[:, None] * (RE + alt)[..., None]; Pw[..., 2] -= RE
        p, z = c.project(Pw); prof = (np.exp(-v * 2.6) + .9 * np.exp(-(v / .05) ** 2)) * (1 - v) ** .5; I = (env * (.25 + .75 * np.clip(ray, 0, 1)))[:, None] * prof[None]
        # visible? ray to the point must clear the planet
        e = np.asarray(c.eye) + [0, 0, RE]; d = Pw + [0, 0, RE] - e; L = np.linalg.norm(d, axis=-1); d /= L[..., None]; bb = d @ e; c2 = e @ e - bb ** 2; hid = (bb < 0) & (c2 < (RE + 15) ** 2) & (-bb - np.sqrt(np.maximum(RE ** 2 - c2, 0)) < L) & (c2 < RE ** 2); ok = (z > 10) & ~hid
        du = np.hypot(*np.moveaxis(np.gradient(p, axis=0), -1, 0)) ; dv = np.hypot(*np.moveaxis(np.gradient(p, axis=1), -1, 0)); w = np.clip(du * dv / 4., 0, 30.) * np.clip(3. / np.maximum(du, .15), .6, 6)
        cc = ramp(np.broadcast_to(v[None], I.shape).astype(f32), [(0, cols[0]), (.05, cols[1]), (.34, cols[2] if s % 2 == 0 else cols[1]), (.62, cols[3]), (1, cols[4])]); I2 = (I * w * q['I'] * ok * .46).astype(f32)
        m = ok & (I2 > 1e-4); _sp(B, (p[..., 0] / 2)[m].astype(f32), (p[..., 1] / 2)[m].astype(f32), I2[m], cc[m])
    B = cv2.GaussianBlur(B, (0, 0), sigmaX=.9, sigmaY=1.5); G = cv2.GaussianBlur(B, (0, 0), 14); B = np.minimum(B, 1.3); A = cv2.resize(B + G * .35, (c.w, c.h), interpolation=cv2.INTER_LINEAR) * amt
    if mask is not None: A *= mask[..., None]
    return 1 - (1 - np.minimum(im, 1.2)) * (1 - np.clip(A, 0, .97))
