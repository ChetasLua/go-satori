"""THE FOUR-STAR BALL: a real glass sphere, ray traced and then painted (hard tones, coloured pen line, no CGI smear).

    img = ball.draw(img, cx, cy, r, stars=(1, 1, 1, 1), t=f)            # img = the picture BEHIND the ball (H, W, 3); returns the picture with the ball
    img = ball.draw(img, 960, 486, 1500, ball.ignite(f, (10, 20, 30, 40)), glass=.05, env=None, pulses=ball.pulses(f, (10, 20, 30, 40)))   # macro
    img = ball.draw(img, cx, cy, r, stars=(0,)*4, stone=1., wet=1.)      # dead grey stone in the rain
    img = ball.draw(img, cx, cy, r, stars=ball.relight(f, (10, 20, 30, 40)), stone=1., relight=ball.relight(f, (10, 20, 30, 40)))   # h2: stone clears from each star

cx, cy, r   screen px of the sphere (60 .. 3000 px across); only its box is touched, plus the outer glow
stars       4 levels: 0 dead red-brown chip in the glass, 1 burning, > 1 = the flash of ignition (white-hot)
glass       how much of the world comes through (1 clear amber, 0 dark amber as at the start of a1); body = the ball's own inner fire 0..1.5
env         picture seen through the glass when it is not `img` (same camera).  focal px of that camera: the world is at infinity, so it hangs upside down, small
spin        (yaw, pitch, roll) radians of the stars in the glass: real parallax, because they float at four depths
light       direction TO the sun (x right, y up, z to us).  z < 0 = sun behind the ball: the caustic gathers on OUR side.  sun=colour
pulses      [(star 0..3, radius 0..3, strength)]: a shell of light running through the glass from a star (the travelling glint)
stone 0..1, wet 0..1, drip = frame number for running water; relight = 4 levels 0..1.3: how far the glass has cleared round each star
near        premultiplied RGBA frame (H, W, 4) of something close behind the ball (an eye): magnified, upright, at near_d radii behind the glass (.2 = 2x, .5 = 3x, .66 = 4.5x);  star_op < 1 = the stars let what is behind them show
line        pen weight multiplier (0 = no contour);  shade = multiply colour of the surroundings (night, rain)
"""
import numpy as np, cv2, math
from numba import njit
from .paint import rgb, f32, fbm, smooth, _vn
from .comp import blur

N_GLASS = 1.48
# star centres in the glass (x right, y up, z to us), radius, own tilt.  a kite, not a grid; four depths.
STAR = np.array([(-.17, .235, .06), (.235, .16, -.16), (-.235, -.165, -.12), (.17, -.235, .10)])
STAR_R = np.array([.178, .170, .172, .180])
STAR_ROT = np.array([.10, -.16, .22, -.05])


def _rot(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    return np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1.]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])


# ------------------------------------------------------------- 1-D caustics (rotationally symmetric, so a table is exact)
def _sun_caustic(m=256):
    """parallel light through a sphere: irradiance on the inner far surface against the angle from the exit axis"""
    i = np.linspace(0, math.pi / 2 - 1e-3, 60000); r = np.arcsin(np.sin(i) / N_GLASS)
    phi = np.abs(2 * r - i); F = .04 + .96 * (1 - np.cos(i)) ** 5
    w = np.cos(i) * np.sin(i) * (1 - F)
    h, e = np.histogram(phi, m, (0, math.pi), weights=w); c = (e[1:] + e[:-1]) / 2
    d = h / (np.sin(c) + .012); k = np.array([.1, .2, .4, .2, .1]); d = np.convolve(d, k, 'same')
    return (d / (np.percentile(d[d > 0], 70) + 1e-9)).astype(f32)


def _star_caustic(a, m=256):
    """point light at distance a from the centre: light mirrored ONCE off the inner wall, landing density against angle from the star's axis"""
    th = np.linspace(1e-3, math.pi - 1e-3, 60000); dx, dy = np.cos(th), np.sin(th)         # star at (a, 0), emission angle th, meridian plane
    b = a * dx; L = -b + np.sqrt(np.maximum(b * b + 1 - a * a, 0)); px, py = a + L * dx, L * dy
    dn = dx * px + dy * py; rx, ry = dx - 2 * dn * px, dy - 2 * dn * py; L2 = -2 * (rx * px + ry * py); qx, qy = px + L2 * rx, py + L2 * ry
    ang = np.abs(np.arctan2(qy, qx)); sn = np.sqrt(np.maximum(0, 1 - dn * dn)); F = np.where(sn * N_GLASS > 1, 1., .06 + .5 * np.clip(sn * N_GLASS, 0, 1) ** 8)
    h, e = np.histogram(ang, m, (0, math.pi), weights=np.sin(th) * F); c = (e[1:] + e[:-1]) / 2
    d = h / (np.sin(c) + .02); d = np.convolve(d, [.1, .2, .4, .2, .1], 'same')
    return (d / (np.percentile(d, 70) + 1e-9)).astype(f32)


_SUN = _sun_caustic()
_STC = np.stack([_star_caustic(float(np.linalg.norm(s))) for s in STAR])
_r = np.random.default_rng(4)
_B = _r.normal(0, 1, (30, 3)); _B /= np.linalg.norm(_B, axis=1)[:, None]; _B *= (.35 + .6 * _r.random((30, 1)) ** .6)
BUB = np.c_[_B, .004 + .012 * _r.random(30) ** 3].astype(np.float64)


@njit(cache=True, fastmath=True)
def _sdstar(x, y, r):
    """distance to a 5-point star (tip radius r) in units of r, and the bevel facet -1 / +1"""
    k = math.atan2(x, y)                                   # 0 = up
    s = 2 * math.pi / 5
    q = math.floor(k / s + .5); a = k - q * s
    rho = math.sqrt(x * x + y * y) / r
    u = rho * math.sin(abs(a)); v = rho * math.cos(a)        # wedge: tip at (0,1), valley at angle pi/5, radius .42
    vx = .42 * 0.5877853; vy = .42 * 0.8090170
    ex = vx; ey = vy - 1.; L = math.sqrt(ex * ex + ey * ey)
    d = (u * ey - (v - 1.) * ex) / L                        # signed distance to the edge line, + outside
    return -d, (1. if a > 0 else -1.), q * s + (a * 0.), rho


@njit(cache=True, fastmath=True)
def _tab(T, x):
    n = T.shape[0]; u = x * (n / math.pi) - .5
    if u < 0: return T[0]
    if u > n - 1.001: return T[n - 1]
    i = int(u); f = u - i
    return T[i] * (1 - f) + T[i + 1] * f


@njit(cache=True, fastmath=True)
def _samp(img, x, y, out):
    """bilinear, mirrored at the borders"""
    h = img.shape[0]; w = img.shape[1]
    px = 2 * (w - 1); py = 2 * (h - 1)
    x = x % px; y = y % py
    if x > w - 1: x = px - x
    if y > h - 1: y = py - y
    i = int(x); j = int(y)
    if i > w - 2: i = w - 2
    if j > h - 2: j = h - 2
    if i < 0: i = 0
    if j < 0: j = 0
    fx = x - i; fy = y - j
    for c in range(3):
        a = img[j, i, c] * (1 - fx) + img[j, i + 1, c] * fx
        b = img[j + 1, i, c] * (1 - fx) + img[j + 1, i + 1, c] * fx
        out[c] = a * (1 - fy) + b * fy


@njit(cache=True, fastmath=True)
def _st(x, a, b):
    t = (x - a) / (b - a)
    if t < 0: return 0.
    if t > 1: return 1.
    return t * t * (3 - 2 * t)


@njit(cache=True, fastmath=True)
def _trace(x0, y0, w, h, cx, cy, r, env, small, sk, ox, oy, focal, W, H, S, SN, SU, SR, lev, glass, body, lx, ly, lz, suncol, SUN, STC, SD, pul, npul, BUB, tone, out, A, STM, flare, near, nB, hasnear, sop):
    """glass pass.  out[h,w,3] colour, A coverage, STM star mask (for the bloom)"""
    n = 1.48; eta = 1. / n
    D = 5.2                                              # eye distance in radii: gentle perspective
    e = np.zeros(3); e2 = np.zeros(3); g = np.zeros(3); col = np.zeros(3)
    aa0 = 1.0 / r
    for j in range(h):
        for i in range(w):
            u = (x0 + i + .5 - cx) / r; v = -(y0 + j + .5 - cy) / r
            rho2 = u * u + v * v; rho = math.sqrt(rho2)
            cov = (1. - rho) * r + .5
            if cov <= 0: continue
            if cov > 1: cov = 1.
            A[j, i] = cov
            if rho > .9995: u *= .9995 / rho; v *= .9995 / rho; rho2 = u * u + v * v; rho = .9995
            pz = math.sqrt(1 - rho2)
            # eye ray
            ex = u / D; ey = v / D; ez = -1.0; q = math.sqrt(ex * ex + ey * ey + 1); ex /= q; ey /= q; ez /= q
            ci = -(ex * u + ey * v + ez * pz)
            if ci < .012: ci = .012
            F = .04 + .96 * (1 - ci) ** 5
            ct = math.sqrt(1 - eta * eta * (1 - ci * ci)); k = eta * ci - ct
            dx = eta * ex + k * u; dy = eta * ey + k * v; dz = eta * ez + k * pz
            L = -2 * (dx * u + dy * v + dz * pz)
            qx = u + L * dx; qy = v + L * dy; qz = pz + L * dz
            # out of the far side
            c2 = dx * qx + dy * qy + dz * qz
            s2 = n * n * (1 - c2 * c2)
            if s2 > 1: s2 = 1.
            c3 = math.sqrt(1 - s2); k2 = n * c2 - c3
            tx = n * dx - k2 * qx; ty = n * dy - k2 * qy; tz = n * dz - k2 * qz
            F2 = .04 + .96 * (1 - c3) ** 5
            if tz > -.06: tz = -.06
            sx = W * .5 + focal * tx / -tz + (cx - W * .5 + r * qx) * .0 + (cx - W * .5) * 1.0; sy = H * .5 - focal * ty / -tz + (cy - H * .5) * 1.0
            # the rim compresses the world: read it from the blurred copy
            mm = _st(rho, .80, .97)
            if glass > .04: _samp(env, sx - ox, sy - oy, e)
            else: e[0] = 0.; e[1] = 0.; e[2] = 0.
            if mm > 0 and glass > .04:
                _samp(small, (sx - ox) / sk, (sy - oy) / sk, e2)
                for c in range(3): e[c] = e[c] * (1 - mm) + e2[c] * mm
            if hasnear:                                          # something held close behind the glass: magnified, upright
                nx = cx + r * (qx + nB * tx / -tz); ny = cy - r * (qy + nB * ty / -tz)
                if 0 <= nx < W - 1.001 and 0 <= ny < H - 1.001:
                    i_ = int(nx); j_ = int(ny); fx = nx - i_; fy = ny - j_
                    w0 = (1 - fx) * (1 - fy); w1 = fx * (1 - fy); w2 = (1 - fx) * fy; w3 = fx * fy
                    na = near[j_, i_, 3] * w0 + near[j_, i_ + 1, 3] * w1 + near[j_ + 1, i_, 3] * w2 + near[j_ + 1, i_ + 1, 3] * w3
                    for c in range(3): e[c] = e[c] * (1 - na) + near[j_, i_, c] * w0 + near[j_, i_ + 1, c] * w1 + near[j_ + 1, i_, c] * w2 + near[j_ + 1, i_ + 1, c] * w3
            # amber: Beer-Lambert over the chord
            T0 = math.exp(-.02 * L); T1 = math.exp(-.34 * L); T2 = math.exp(-1.15 * L)
            col[0] = e[0] * T0 * glass; col[1] = e[1] * T1 * glass; col[2] = e[2] * T2 * glass
            # body: the ball's own amber, deeper toward the shaded side; three flat tones with short painted steps
            wob = (_vn(u * 2.3 + 9, v * 2.3 + 4, 11) - .5) * .10
            sd = u * lx + v * ly                              # toward the light on screen
            b = rho2 * .95 + .28 * sd * (1 if lz > 0 else -.4) + wob      # dark rim, a little heavier on the light side (light gathers opposite)
            m1 = _st(b, .40 - tone, .40 + tone); m2 = _st(b, .74 - tone, .74 + tone)
            bm = body * (1 - .80 * glass)
            for c in range(3):
                a0 = 1.0 if c == 0 else (.62 if c == 1 else .13)
                a1 = .95 if c == 0 else (.43 if c == 1 else .06)
                a2 = .66 if c == 0 else (.21 if c == 1 else .04)
                bb = a0 * (1 - m1) + (a1 * (1 - m2) + a2 * m2) * m1
                col[c] += bb * bm
                # dark amber rim over the refraction, so it reads as glass and not as a hole
                kk = (.42 * m2 + .16 * m1) * min(1., glass + .25)
                col[c] = col[c] * (1 - kk) + (a2 * (.35 + .65 * body)) * kk
            # sun caustic on the far wall (and on the near wall when the sun is behind)
            cf = -(qx * lx + qy * ly + qz * lz); cn = -(u * lx + v * ly + pz * lz)
            if cf > 1: cf = 1.
            if cf < -1: cf = -1.
            if cn > 1: cn = 1.
            if cn < -1: cn = -1.
            I1 = _tab(SUN, math.acos(cf)); I2 = _tab(SUN, math.acos(cn)) * .5
            I = I1 + I2
            kc = (_st(I, .55 - tone, .55 + tone) * .20 + _st(I, 1.05 - tone, 1.05 + tone) * .34 + _st(I, 2.0 - tone, 2.0 + tone) * .55) * sunk(suncol)
            col[0] += kc * suncol[0]; col[1] += kc * suncol[1] * .80; col[2] += kc * suncol[2] * .42
            # stars: light in the volume, caustics, the shapes themselves
            for s in range(4):
                lv = lev[s]
                if lv <= .003: continue
                wx = S[s, 0] - u; wy = S[s, 1] - v; wz = S[s, 2] - pz
                tt = wx * dx + wy * dy + wz * dz
                tt = min(max(tt, 0.), L)
                ax = wx - dx * tt; ay = wy - dy * tt; az = wz - dz * tt
                d2 = (ax * ax + ay * ay + az * az) / (SR[s] * SR[s])
                gl = 1. / (1 + d2 * .55); gl = gl * gl * lv
                col[0] += gl * .62; col[1] += gl * .17 * (1 + .6 * max(0., lv - 1)); col[2] += gl * .035 * (1 + 3 * max(0., lv - 1))
                # hard inner halo, like a painted light
                if gl > .16:
                    thr = math.atan2(ay, ax); ray = _vn(math.cos(thr) * 3.2 + 10 + s * 3, math.sin(thr) * 3.2 + 10, 41 + s) * .6 + _vn(math.cos(thr) * 9 + 30, math.sin(thr) * 9 + 30 + s, 47) * .4
                    gh = gl * (.74 + .52 * ray)
                    hl = (_st(gh, .30, .44) * .5 + _st(gh, .56, .70) * .5) * min(lv, 1.6)
                    col[0] += hl * .17; col[1] += hl * .115; col[2] += hl * .02
                dn = math.sqrt(S[s, 0] ** 2 + S[s, 1] ** 2 + S[s, 2] ** 2) + 1e-9
                c_ = (qx * S[s, 0] + qy * S[s, 1] + qz * S[s, 2]) / dn; c_ = min(1., max(-1., c_)); Ia = _tab(STC[s], math.acos(c_))
                c_ = (u * S[s, 0] + v * S[s, 1] + pz * S[s, 2]) / dn; c_ = min(1., max(-1., c_)); Ib = _tab(STC[s], math.acos(c_))
                wv = .55 + .9 * _vn(qx * 5 + s * 7, qy * 5 + qz * 3, 23) if (Ia > 1.45 or Ib > 2.5) else 0.
                kc = (_st(Ia, 1.5, 1.5 + tone * 6) * .16 + _st(Ia, 2.6, 2.6 + tone * 8) * .30 + _st(Ib, 2.6, 2.6 + tone * 8) * .12) * lv * wv
                col[0] += kc * 1.0; col[1] += kc * .40; col[2] += kc * .10
            if r > 300 and flare > .01:
                tt = L * .5; ax = u + dx * tt; ay = v + dy * tt; az = pz + dz * tt
                wa = _vn(ax * 2.1 + 5, ay * 2.1 + 9, 71) * 2.2; wb = _vn(ax * 2.1 + 25, ay * 2.1 + 2, 72) * 2.2
                rg = 1 - abs(2 * _vn(ax * 3.4 + wa * 2.0, ay * 3.4 + az * 2 + wb * 2.0, 73) - 1); rg2 = 1 - abs(2 * _vn(ax * 7.5 + wb * 2.4, ay * 7.5 - wa * 2.4, 74) - 1)
                st_ = _st(rg, .86, .99) * .7 + _st(rg2, .90, 1.0) * .45
                lit = .035 * body + .02 * glass; sk_ = 1 - .70 * min(1., glass)
                for s in range(4):
                    qd = (ax - S[s, 0]) ** 2 + (ay - S[s, 1]) ** 2 + (az - S[s, 2]) ** 2; lit += lev[s] * .030 * sk_ / (.06 + qd)
                lit *= flare; col[0] += st_ * lit; col[1] += st_ * lit * .55; col[2] += st_ * lit * .14
            # pulses: a shell of light running through the glass
            for p in range(npul):
                s = int(pul[p, 0]); R = pul[p, 1]; kp = pul[p, 2]
                acc = 0.
                wd = .016 + .020 * R
                wx = S[s, 0] - u; wy = S[s, 1] - v; wz = S[s, 2] - pz
                tt = wx * dx + wy * dy + wz * dz
                ax = wx - dx * tt; ay = wy - dy * tt; az = wz - dz * tt
                bb = math.sqrt(ax * ax + ay * ay + az * az)
                if bb < R:
                    hh = math.sqrt(R * R - bb * bb); gg = wd * 2.2 / math.sqrt(max(1 - (bb / R) ** 2, .012))
                    if 0 < tt - hh < L: acc += gg
                    if 0 < tt + hh < L: acc += gg
                else:
                    dd = (bb - R) / (wd * .55)
                    if 0 < tt < L: acc += 2 * wd * 2.2 / .11 * math.exp(-dd * dd)
                ax = qx - S[s, 0]; ay = qy - S[s, 1]; az = qz - S[s, 2]; dd = (math.sqrt(ax * ax + ay * ay + az * az) - R) / (wd * .40)
                acc += math.exp(-dd * dd) * 1.5
                ax = u - S[s, 0]; ay = v - S[s, 1]; az = pz - S[s, 2]; dd = (math.sqrt(ax * ax + ay * ay + az * az) - R) / (wd * .30)
                acc += math.exp(-dd * dd) * 1.0
                acc *= kp * (.45 + 1.1 * _vn(u * 3 + s * 5 + R, v * 3, 31))
                col[0] += acc * 1.0; col[1] += acc * .62; col[2] += acc * .22
            # bubbles and dust in the glass (only resolvable when the ball is large)
            if r > 160:
                for s in range(BUB.shape[0]):
                    wx = BUB[s, 0] - u; wy = BUB[s, 1] - v; wz = BUB[s, 2] - pz
                    tt = wx * dx + wy * dy + wz * dz
                    if tt < 0 or tt > L: continue
                    ax = wx - dx * tt; ay = wy - dy * tt; az = wz - dz * tt
                    rb = BUB[s, 3]; d2 = ax * ax + ay * ay + az * az
                    if d2 > rb * rb * 1.2: continue
                    dd = math.sqrt(d2) / rb; a_ = aa0 / rb
                    ring_ = _st(dd, .62, .62 + a_ + .12) * (1 - _st(dd, 1 - a_, 1 + a_ * .2))
                    sp = 1 - _st(math.sqrt((ax + .35 * rb) ** 2 + (ay + .45 * rb) ** 2) / rb, .22, .22 + a_ + .1)
                    lit = .16 * glass + .10 * body
                    for s3 in range(4):
                        qd = (BUB[s, 0] - S[s3, 0]) ** 2 + (BUB[s, 1] - S[s3, 1]) ** 2 + (BUB[s, 2] - S[s3, 2]) ** 2
                        lit += lev[s3] * .10 / (.08 + qd)
                    for p in range(npul):
                        s3 = int(pul[p, 0]); qd = math.sqrt((BUB[s, 0] - S[s3, 0]) ** 2 + (BUB[s, 1] - S[s3, 1]) ** 2 + (BUB[s, 2] - S[s3, 2]) ** 2)
                        qd = (qd - pul[p, 1]) / .10; lit += 2.4 * pul[p, 2] * math.exp(-qd * qd)
                    kk = ring_ * .55 * min(lit, 1.1); col[0] += kk; col[1] += kk * .55; col[2] += kk * .18
                    kk = sp * min(lit, 1.6); col[0] += kk; col[1] += kk * .8; col[2] += kk * .5
            # the four stars, nearest hit on top; then their ghost mirrored off the far wall
            for ps in range(2):
                if ps == 0: rx = u; ry = v; rz = pz; vx = dx; vy = dy; vz = dz; LL = L; am = sop
                else:
                    kk = 2 * (dx * qx + dy * qy + dz * qz); vx = dx - kk * qx; vy = dy - kk * qy; vz = dz - kk * qz; rx = qx; ry = qy; rz = qz; LL = L; am = min(.34, F2 * 1.2 + .015) * sop
                # order by distance
                o0 = 0; best = np.zeros(4); cnt = 0
                for s in range(4):
                    dn = vx * SN[s, 0] + vy * SN[s, 1] + vz * SN[s, 2]
                    best[s] = 1e9
                    if abs(dn) < .05: continue
                    tt = ((S[s, 0] - rx) * SN[s, 0] + (S[s, 1] - ry) * SN[s, 1] + (S[s, 2] - rz) * SN[s, 2]) / dn
                    if tt <= .01 or tt >= LL: continue
                    best[s] = tt
                for rep in range(4):
                    s = -1; bt = -1.
                    for q_ in range(4):
                        if best[q_] < 1e8 and best[q_] > bt: bt = best[q_]; s = q_
                    if s < 0: break
                    best[s] = 1e9
                    hx = rx + vx * bt - S[s, 0]; hy = ry + vy * bt - S[s, 1]; hz = rz + vz * bt - S[s, 2]
                    # star frame
                    ux, uy, uz = SU[s, 0], SU[s, 1], SU[s, 2]
                    wx = uy * SN[s, 2] - uz * SN[s, 1]; wy = uz * SN[s, 0] - ux * SN[s, 2]; wz = ux * SN[s, 1] - uy * SN[s, 0]
                    a_ = hx * wx + hy * wy + hz * wz; b_ = hx * ux + hy * uy + hz * uz
                    if a_ * a_ + b_ * b_ > 1.9 * SR[s] * SR[s]: continue
                    d, side, ang, rr = _sdstar(a_, b_, SR[s])
                    if d > .34: continue
                    fw = aa0 / SR[s] * 1.0
                    lv = lev[s]; o = min(lv, 1.); fl = max(0., lv - 1.)
                    # thickness: the same star a little deeper in the glass, seen past the edge of the first
                    sv = vx * SN[s, 0] + vy * SN[s, 1] + vz * SN[s, 2]
                    a2 = a_ - (vx * wx + vy * wy + vz * wz) / sv * .055; b2 = b_ - (vx * ux + vy * uy + vz * uz) / sv * .055
                    d2s, sd2, an2, r2s = _sdstar(a2, b2, SR[s])
                    dm_ = d if d < d2s else d2s
                    seat = (1 - _st(dm_, .050 - fw, .050 + fw)) * am
                    if seat > 0:                                          # dark seat and side wall
                        kk = seat * (.92 - .55 * min(1., fl))
                        wl = .16 + .10 * _st(ux * 0 + math.sin(an2 + sd2 * .55) * lx + math.cos(an2 + sd2 * .55) * ly, -.1, .1)
                        col[0] = col[0] * (1 - kk) + (wl * (.30 + .30 * body + 1.5 * o)) * kk; col[1] = col[1] * (1 - kk) + .018 * (1 + 2 * o) * kk; col[2] = col[2] * (1 - kk) + .022 * kk
                    inside = 1 - _st(d, -fw, fw)
                    if inside > 0:
                        ca = ang + side * .55; fx = math.sin(ca); fy = math.cos(ca)
                        li = fx * lx + fy * ly
                        hi = _st(li, -.10, .10)
                        h1 = _st(-d, .075, .075 + fw * 1.5); h2 = _st(-d, .20, .20 + fw * 1.5 + .02)      # nested bevels: rim, face, hot heart
                        wm = .5 + .5 * _vn(a_ / SR[s] * 5 + 3, b_ / SR[s] * 5 + s * 9, 61)
                        c0 = (.23 + .17 * hi) * (1 - o) + (.62 + .20 * hi + .18 * h1) * o
                        c1 = (.035 + .025 * hi) * (1 - o) + (.018 + .050 * hi + .085 * h1 * wm + .33 * h2) * o
                        c2_ = (.035 + .012 * hi) * (1 - o) + (.045 + .020 * hi + .015 * h1 + .07 * h2) * o
                        rimk = (1 - h1) * hi                                                              # the lit edge of the bevel
                        c0 += rimk * (.10 + .22 * o); c1 += rimk * (.03 + .16 * o); c2_ += rimk * (.03 + .09 * o)
                        c0 += fl * 1.0; c1 += fl * (.40 + .55 * h1 + .45 * h2); c2_ += fl * (.12 + .42 * h1 + .6 * h2)
                        kk = inside * am
                        col[0] = col[0] * (1 - kk) + c0 * kk; col[1] = col[1] * (1 - kk) + c1 * kk; col[2] = col[2] * (1 - kk) + c2_ * kk
                        if ps == 0 and kk * lv > STM[j, i]: STM[j, i] = kk * lv
            # fresnel: the world mirrored on the near face, in one hard band
            kk = 2 * (ex * u + ey * v + ez * pz); fx = ex - kk * u; fy = ey - kk * v; fz = ez - kk * pz
            tz = fz if fz > .05 else .05
            _samp(small, (W * .5 + focal * .5 * fx / (abs(tz) + .5) - ox) / sk, (H * .5 - focal * .5 * fy / (abs(tz) + .5) - oy) / sk, g)
            fr = min(1., F * 1.1) * _st(rho, .905 - tone, .905 + tone * 2) * .80
            for c in range(3): out[j, i, c] = col[c] * (1 - fr) + g[c] * fr


@njit(cache=True)
def sunk(c):
    return 1.0


@njit(cache=True, fastmath=True)
def _highlights(x0, y0, w, h, cx, cy, r, lx, ly, lz, sc, out, amt):
    """hard painted speculars: the sun as a curved window high on the light side, a small second glint, a low bounce crescent opposite"""
    q = math.sqrt(lx * lx + ly * ly) + 1e-6; ax = lx / q; ay = ly / q
    aa = 1.2 / r
    for j in range(h):
        for i in range(w):
            u = (x0 + i + .5 - cx) / r; v = -(y0 + j + .5 - cy) / r
            rho = math.sqrt(u * u + v * v)
            if rho > 1: continue
            p = u * ax + v * ay; t = -u * ay + v * ax              # p toward the light, t across
            # window: between two arcs of circles, cut by two radial edges
            th = math.atan2(t, p)
            tw = th + .03; thk = .082 * max(0., 1 - (tw / .50) ** 2) ** .62 * (1. + .35 * tw) + .006 * (_vn(th * 9 + 4, 2., 7) - .5)
            mid = .765 - .02 * tw
            w1 = (1 - _st(abs(rho - mid), thk, thk + aa)) * (1. if abs(tw) < .5 else 0.)
            cut = _st(abs(th - .30), .017, .017 + aa * 1.4)         # window bar
            w2 = 1 - _st(math.sqrt((th + .66) ** 2 * .60 + (rho - .785) ** 2 * 1.3), .040, .040 + aa)
            # bounce crescent: between the rim and a shifted circle, on the far side
            o1 = math.sqrt((p + .0) ** 2 + t * t); o2 = math.sqrt((p - .115) ** 2 + t * t)
            b1 = _st(o2, .985, .985 + aa * 1.5) * (1 - _st(o1, .955, .955 + aa * 1.5)) * _st(-p, .25, .55)
            k = min(1., w1 * cut + w2) * amt
            for c in range(3): out[j, i, c] = out[j, i, c] * (1 - k * .93) + (1.0 if c < 1 else .985 if c < 2 else .90) * k * .93
            k = b1 * .55 * amt
            for c in range(3): out[j, i, c] = out[j, i, c] * (1 - k) + sc[c] * k


# ---------------------------------------------------------------------- stone
_STN = {}


def _stone_tex():
    """equirect-free stone: noise volumes baked once as front/back hemisphere heights on a 512 grid (u, v)"""
    if 'h' not in _STN:
        n = 640
        big = fbm(n, n, (150, 60, 22, 8), (1, .6, .32, .18), 71); fine = fbm(n, n, (7, 3.2, 1.7), (1, .8, .5), 72); pits = fbm(n, n, (16, 7), (1, .6), 73); vein = np.abs(fbm(n, n, (120, 44), (1, .5), 74)); dr = fbm(n, n, (90, 30, 9), (1, .6, .3), 75)
        _STN.update(h=big.astype(f32), fine=fine.astype(f32), pits=pits.astype(f32), vein=vein.astype(f32), dr=dr.astype(f32), n=n)
    return _STN


def _stone(u, v, rho, S, SU, SN, SR, lx, ly, lz, wet, drip, shade, r):
    T = _stone_tex(); n = T['n']
    z = np.sqrt(np.clip(1 - u * u - v * v, 0, 1)); k = 1 / (1 + .55 * z)                  # spread the rim, so texture follows the sphere
    mu = ((u * k * .62 + .5) * (n - 1)).astype(f32); mv = ((.5 - v * k * .62) * (n - 1)).astype(f32)
    g = lambda a: cv2.remap(a, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    big, fine, pit, vein = g(T['h']), g(T['fine']), g(T['pits']), g(T['vein'])
    q = np.array([lx, ly, max(lz, .25)]); q /= np.linalg.norm(q)
    nl = u * q[0] + v * q[1] + z * q[2] + .055 * big + .022 * fine
    c = [rgb(h) for h in ('#1d1e21', '#34363a', '#54575a', '#83868a')]
    c = [x * (1 - .34 * wet) + np.array([0, .004, .012], f32) * wet for x in c]
    s = .035
    o = c[0] + (c[1] - c[0]) * smooth(.02 - s, .02 + s, nl)[..., None]; o = o + (c[2] - o) * smooth(.42 - s, .42 + s, nl)[..., None]; o = o + (c[3] - o) * (smooth(.80 - s, .80 + s, nl) * (1 - .5 * wet))[..., None]
    o = o * (1 + .05 * np.clip(fine, -2, 2))[..., None] * (1 - .35 * smooth(1.25, 1.5, pit))[..., None]
    o = o * (1 - .45 * (1 - smooth(.0, .05, vein)) * smooth(-.4, .6, big))[..., None]               # two hair cracks
    # stars cut into the face
    for i in range(4):
        hx, hy, hz = u - S[i, 0] * 1.35, v - S[i, 1] * 1.35, 0
        wx = np.cross(SU[i], SN[i]); a_ = hx * wx[0] + hy * wx[1]; b_ = hx * SU[i, 0] + hy * SU[i, 1]
        rr = SR[i] * 1.30; th = np.arctan2(a_, b_); st = 2 * math.pi / 5; a = th - np.floor(th / st + .5) * st; rad = np.hypot(a_, b_) / rr
        uu = rad * np.sin(np.abs(a)); vv = rad * np.cos(a); vx, vy = .42 * .5877853, .42 * .809017; ex, ey = vx, vy - 1.; Ln = math.hypot(ex, ey); d = -(uu * ey - (vv - 1) * ex) / Ln
        fw = max(1.0 / (r * rr), .004); ins = 1 - smooth(-fw, fw, d)
        th0 = np.floor(th / st + .5) * st + np.sign(a) * .55; li = np.sin(th0) * lx + np.cos(th0) * ly            # cut walls: lit where they face the light from INSIDE
        o = o + (c[0] * .8 - o) * (ins * .80)[..., None]
        o = o + (c[2] * .9 - o) * (ins * smooth(.25, .35, -li) * smooth(-.16, -.02, d) * .8)[..., None]
        lip = smooth(.0 - fw, .0 + fw, d) * (1 - smooth(.05 - fw, .05 + fw, d)); o = o + (c[3] - o) * (lip * smooth(.0, .3, -li) * .6)[..., None]
    if wet > 0:                                                          # a skin of water: hard sky reflection, crawling rivulets, beads
        hh = np.clip(u * lx + v * ly, -1, 1); sky = rgb('#aeb9c6')
        wob = .05 * big + .02 * np.sin(v * 17 + u * 6)
        rimw = smooth(.80, .83, rho + wob) * smooth(-.25, .15, hh); o = o + (sky - o) * (rimw * .42 * wet)[..., None]
        win = smooth(.60, .62, hh + wob * 1.2) * (1 - smooth(.83, .85, rho)); o = o + (np.array([.93, .96, 1.], f32) - o) * (win * .78 * wet)[..., None]
        x = u * 6.5 + .9 * np.sin(v * 3.1 + 1) + .35 * np.sin(v * 9 + u * 3); fr = np.abs(x - np.round(x)); ln = np.round(x)
        ph = (np.sin(ln * 12.9898) * 43758.5453) % 1; on = (ph > .42)
        head = 1 - 2 * ((drip * (.012 + .016 * ph) + ph * 7.3) % 1.35); body_ = smooth(head - .02, head + .01, v) * (1 - smooth(head + .30, head + .9, v))
        wdt = .13 + .10 * np.exp(-((v - head) / .06) ** 2); rv = (1 - smooth(wdt - .05, wdt, fr)) * on * body_ * smooth(1., .9, rho)
        o = o * (1 - .30 * rv * wet)[..., None]; hi = (1 - smooth(.02, .06, np.abs(x - np.round(x) + .05))) * on * body_ * smooth(1., .9, rho); o = o + (sky * 1.1 - o) * (hi * .75 * wet)[..., None]
    o = o * (1 - .55 * smooth(.86, 1.0, rho))[..., None]
    return (o * np.asarray(shade, f32)).astype(f32)


# ---------------------------------------------------------------------- public
def star_frame(spin=(0., 0., 0.)):
    R = _rot(*spin); S = STAR @ R.T
    SN = np.zeros((4, 3)); SU = np.zeros((4, 3))
    for i in range(4):
        n0 = np.array([STAR[i, 0] * .55, STAR[i, 1] * .55, 1.0]); n0 /= np.linalg.norm(n0); up = np.array([math.sin(STAR_ROT[i]), math.cos(STAR_ROT[i]), 0.]); up = up - n0 * (up @ n0); up /= np.linalg.norm(up)
        SN[i] = R @ n0; SU[i] = R @ up
    return S, SN, SU


def star_px(cx, cy, r, spin=(0., 0., 0.), k=1.30):
    """where the stars appear on screen (refraction enlarges them about 1.3x)"""
    S, _, _ = star_frame(spin); return np.c_[cx + S[:, 0] * r * k, cy - S[:, 1] * r * k]


def draw(img, cx, cy, r, stars=(1., 1., 1., 1.), t=0., glass=.75, body=.75, env=None, focal=1500., spin=(0., 0., 0.), light=(-.45, .70, .55), sun='#fff0c0', pulses=(), stone=0., wet=0., drip=0.,
         relight=None, line=1., shade=(1., 1., 1.), glow=1., spec=1., ink='#4a1606', tone=None, near=None, near_d=.66, stria=1., star_op=1.):
    base = np.asarray(img, f32); H, W = base.shape[:2]; out_img = base.copy()
    env = base if env is None else np.asarray(env, f32)
    pad = 3; x0, y0 = int(max(0, math.floor(cx - r - pad))), int(max(0, math.floor(cy - r - pad))); x1, y1 = int(min(W, math.ceil(cx + r + pad))), int(min(H, math.ceil(cy + r + pad)))
    if x1 - x0 < 2 or y1 - y0 < 2: return out_img
    w, h = x1 - x0, y1 - y0
    L = np.asarray(light, float); L = L / np.linalg.norm(L); sc = rgb(sun) if isinstance(sun, str) else np.asarray(sun, f32)
    S, SN, SU = star_frame(spin); lev = np.asarray(stars, float)
    tone = (max(.012, 1.2 / r) if tone is None else tone)
    shade = np.asarray(shade if not isinstance(shade, str) else rgb(shade), f32)
    relight = None if relight is None else np.asarray(relight, float)
    need_glass = stone < 1 or relight is not None
    A = np.zeros((h, w), f32); out = np.zeros((h, w, 3), f32); STM = np.zeros((h, w), f32)
    if need_glass:
        sk = 8; small = cv2.resize(env, (max(2, W // sk), max(2, H // sk)), interpolation=cv2.INTER_AREA); small = cv2.GaussianBlur(small, (0, 0), 1.2)
        e2 = np.ascontiguousarray(env); pl = np.array([p for p in pulses], float).reshape(-1, 3) if len(pulses) else np.zeros((1, 3))
        _trace(x0, y0, w, h, float(cx), float(cy), float(r), e2, small, float(sk), 0., 0., float(focal), float(W), float(H), S, SN, SU, STAR_R, lev, float(glass), float(body), L[0], L[1], L[2], sc.astype(np.float64), _SUN, _STC, 0., pl, len(pulses), BUB, float(tone), out, A, STM, float(stria), (np.zeros((2, 2, 4), f32) if near is None else np.ascontiguousarray(near, f32)), float(near_d), near is not None, float(star_op))
        out *= shade
        if spec > 0: _highlights(x0, y0, w, h, float(cx), float(cy), float(r), L[0], L[1], L[2], (sc * rgb('#ffcf7a')).astype(np.float64), out, float(spec))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(f32); u = (xx + .5 - cx) / r; v = -(yy + .5 - cy) / r; rho = np.sqrt(u * u + v * v)
    A = np.clip((1 - rho) * r + .5, 0, 1)
    if stone > 0:
        sv = _stone(u, v, rho, S, SU, SN, STAR_R, L[0], L[1], L[2], wet, drip, shade, r)
        if relight is not None:                                           # the stone burns back to glass from every star outward, with a hard ember edge
            nz = cv2.resize(_stone_tex()['dr'], (w, h), interpolation=cv2.INTER_LINEAR) * .085
            m = np.zeros_like(u); em = np.zeros_like(u)
            for i in range(4):
                if relight[i] <= 0: continue
                d = np.hypot(u - S[i, 0] * 1.30, v - S[i, 1] * 1.30) + nz; R = min(relight[i], 1.) * .40 + max(0., relight[i] - 1.) * 6.5
                k = 1 - smooth(R - .012, R + .012, d); m = np.maximum(m, k); em = np.maximum(em, np.exp(-((d - R) / .045) ** 2) * (relight[i] < 1.28) + .5 * np.exp(-np.maximum(d - R, 0) / .16) * (d > R) * (relight[i] < 1.28))
            em = em * np.where(m > .5, smooth(1, .4, m), 1); kk = np.clip(m, 0, 1)[..., None]
            o = sv * (1 - kk) + out * kk; o = o + (rgb('#ffb54a') - o) * np.clip(em * .85 * (1 - .75 * kk[..., 0]), 0, 1)[..., None]; out = o.astype(f32)
        else: out = (sv * stone + out * (1 - stone)) if stone < 1 else sv
    # pen: contour swells on the shadow side, thins into the light
    if line > 0:
        lw = float(np.interp(r, [20, 60, 210, 600, 1600], [1.0, 1.7, 4.2, 6.5, 9.])) * line; th = np.arctan2(v, u); sh = -(np.cos(th) * L[0] + np.sin(th) * L[1]) / (math.hypot(L[0], L[1]) + 1e-6)
        wl = lw * (.62 + .38 * sh + .10 * np.sin(th * 3 + 1) + .06 * np.sin(th * 7)) / r; aa = .7 / r
        k = smooth(1 - wl - aa, 1 - wl + aa, rho); ic = rgb('#17181b') if (stone > .5 and relight is None) else rgb(ink) * np.clip(shade, .25, 1)
        out = out * (1 - k[..., None]) + ic * k[..., None]
    a = A[..., None]; out_img[y0:y1, x0:x1] = out_img[y0:y1, x0:x1] * (1 - a) + out * a
    # light on the world: only where there is light
    if glow > 0:
        e = float(np.clip(lev, 0, 2).sum()) * .25
        if e > .01 and (stone < 1 or relight is not None):
            k = e if relight is None else e * .9
            R = r * 2.6; X0, Y0, X1, Y1 = int(max(0, cx - R)), int(max(0, cy - R)), int(min(W, cx + R)), int(min(H, cy + R))
            if X1 > X0 and Y1 > Y0:
                ys, xs = np.mgrid[Y0:Y1:4, X0:X1:4].astype(f32); d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / r
                gq = np.where(d > 1, np.clip(1 - (d - 1) / 1.6, 0, 1) ** 2.6, 0) ; gq = cv2.resize(gq, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_LINEAR)
                out_img[Y0:Y1, X0:X1] += gq[..., None] * np.array([.50, .20, .05], f32) * .42 * k * glow
        b = STM * np.clip(STM, 0, 2)
        if b.max() > .05:
            s = max(2., r * .05); q = blur(b, s) * .40 + blur(b, s * 3.2) * .34; q = q * A
            out_img[y0:y1, x0:x1] += q[..., None] * np.array([1., .34, .10], f32) * glow
    return out_img


# ---------------------------------------------------------------------- timing
def ignite(f, at=(10, 20, 30, 40), hold=1.0):
    """4 star levels at frame f: nothing, a white-hot flash on the bell, then a steady burn that breathes"""
    o = []
    for i, a in enumerate(at):
        d = f - a
        if d < -2: o.append(0.)
        elif d < 0: o.append(.10 * (d + 3))
        else: o.append(hold * (1 + .05 * math.sin(f * .31 + i * 1.7)) + 1.35 * math.exp(-d / 2.6))
    return tuple(o)


def pulses(f, at=(10, 20, 30, 40), speed=.11, life=20):
    """the glint that leaves each star on its bell and runs through the glass"""
    return [(i, .10 + speed * (f - a) * (1 + .015 * (f - a)), 1.1 * math.exp(-(f - a) / (life * .45))) for i, a in enumerate(at) if 0 <= f - a < life]


def relight(f, at=(10, 20, 30, 40), dur=9):
    """h2: how far the glass has cleared round each star (0..1 its own quarter; after the last bell 1..1.3 = the whole ball)"""
    e = lambda x: 1 - (1 - min(1., max(0., x))) ** 2.2
    return tuple(e((f - a) / dur) + .3 * e((f - at[-1] - 4) / 14.) * (f >= at[-1]) if f >= a else 0. for a in at)
