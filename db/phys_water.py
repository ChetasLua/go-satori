"""phys water: a true height-field sea (waves, pressure ring that flattens it, dish, light road), ballistic jets and crowns, column, rooster tail and V wake, rings, fist squeeze, water pulled to an orb.  (PHYS lane)"""
from .phys_core import *
G_ = 9.8


# ------------------------------------------------------------- sea
@njit(cache=True, fastmath=True)
def _height(x, y, t, W_, lod, ring, dish):
    """height and gradient"""
    z = 0.; gx = 0.; gy = 0.; wx = 1.9 * vn3(x * .045, y * .045, t * .12); wy = 1.9 * vn3(x * .045 + 41., y * .045, t * .12)
    amp = 1.; d = 0.; hz = 0.; hx = 0.; hy = 0.
    if ring[3] > 0.:
        dx = x - ring[0]; dy = y - ring[1]; d = math.sqrt(dx * dx + dy * dy) + 1e-6; s = d - ring[2] * (1 + .012 * vn3(dx / d * 9., dy / d * 9., 3.) + .006 * vn3(dx / d * 31., dy / d * 31., 5.)); w = ring[4]
        if s < 0: amp = ring[5] + (1 - ring[5]) * min(1., max(0., (-s - ring[6]) / (ring[6] * 2 + 1e-6)))
        elif ring[7] > 0.: amp = 1. - .90 * math.exp(-s / ring[7])
        else: amp = 1. + .5 * math.exp(-s / (w * 2.))
        e = s / (w * (.40 if s > 0 else 1.5)); q = ring[3] * math.exp(-e * e); hz += q; k = -2 * e / (w * (.40 if s > 0 else 1.5)) * q; hx += k * dx / d; hy += k * dy / d
    if dish[3] != 0.:
        dx = x - dish[0]; dy = y - dish[1]; d = math.sqrt(dx * dx + dy * dy) + 1e-6; r = dish[2]; e = d / r; q = -dish[3] * math.exp(-e * e * e * e); k = -q * 4 * e * e * e / r; k = -4 * e * e * e / r * q; e2 = (d - r * 1.12) / (r * .16); q2 = dish[4] * math.exp(-e2 * e2); k2 = -2 * e2 / (r * .16) * q2; hz += q + q2; hx += (k + k2) * dx / d; hy += (k + k2) * dy / d
        if d < r * 1.3: amp *= .25 + .75 * min(1., max(0., (d - r * .9) / (r * .4)))
    for i in range(W_.shape[0]):
        kx, ky, k, a, om, ph, sh = W_[i]; lam = 6.2832 / k; f = min(1., max(0., lam / (lod * 2.4) - .6))
        if f <= 0: continue
        th = kx * (x + wx * lam * .12) + ky * (y + wy * lam * .12) - om * t + ph; s = math.sin(th); e = math.exp(sh * (s - 1)); A = a * f * amp * (.65 + .5 * vn3(x * .02 + i * 7., y * .02, t * .06)); z += A * (e - .35); g = A * e * sh * math.cos(th); gx += g * kx; gy += g * ky
    return z + hz, gx + hx, gy + hy, amp


@njit(cache=True, fastmath=True)
def _cast(h, w, f, cx, hy, ex, ey, ez, fx, fy, t, W_, ring, dish, zn, zf, D, Z, GX, GY, AM):
    rx = fy; ry = -fx
    for i in range(w):
        u = (i + .5 - cx) / f; dx = fx + rx * u; dy = fy + ry * u; y0 = h - 1; z = zn; first = True; pz = 0.; pg = 0.; pq = 0.; ps = h * 1.; pd = zn; pa = 1.
        while z < zf and y0 >= 0:
            st = max(.02, min(1.3 * z * z / (f * max(ez, .3)), .022 * z)); hgt, gx, gy, am = _height(ex + dx * z, ey + dy * z, t, W_, max(st, z / f * 1.2), ring, dish); sy = hy + f * (ez - hgt) / z
            if sy < y0 + .5:
                y1 = max(0, int(math.ceil(sy - .5)))
                for y in range(y0, y1 - 1, -1):
                    q = 0. if ps - sy < 1e-6 else (ps - (y + .5)) / (ps - sy); q = 1. if first else min(1., max(0., q)); D[y, i] = pd + (z - pd) * q; Z[y, i] = pz + (hgt - pz) * q; GX[y, i] = pg + (gx - pg) * q; GY[y, i] = pq + (gy - pq) * q; AM[y, i] = pa + (am - pa) * q
                y0 = y1 - 1; first = False
            ps = sy; pz = hgt; pg = gx; pq = gy; pd = z; pa = am; z += st


def waves(amp=.35, lam=9., n=7, wind=(0.25, -1.), seed=0, sharp=1.6, spread=.8):
    """a sea state: n wave trains.  amp metres of the longest, lam its wavelength, wind = travel direction"""
    r = R(seed, 3); a0 = math.atan2(wind[1], wind[0]); W_ = []
    for i in range(n): l = lam * .62 ** i * r.uniform(.85, 1.15); k = TAU / l; a = a0 + r.normal(0, spread * (.4 + .12 * i)); W_.append((k * math.cos(a), k * math.sin(a), k, amp * .60 ** i * r.uniform(.8, 1.2), math.sqrt(G_ * k), r.uniform(0, TAU), sharp))
    return np.array(W_)


def _pw(p): return WATER[p] if isinstance(p, str) else p


class Sea:
    """Height-field sea for a LEVEL camera (stage()).  sea = Sea(waves(.35, 9)); rgb, mask, buf = sea.draw(cam, t, 'night', light=(0, 1, .25))
    ring = (x, y, radius, height, width, inside wave factor, calm length) pressure front; dish = (x, y, radius, depth, rim height).
    light = direction TO the glitter light; road = strength of its hard glints; fog = (hex, start m, density) toward the horizon.
    buf: dict of D (depth m), Z, X, Y, S (ring-distance) per pixel."""

    def __init__(self, W_=None): self.W_ = waves() if W_ is None else W_

    def cast(self, cam, t, ring=None, dish=None, near=.6, far=6000., h=H, w=W):
        D = np.zeros((h, w), f32); Z = np.zeros((h, w), f32); GX = np.zeros((h, w), f32); GY = np.zeros((h, w), f32); AM = np.ones((h, w), f32); fw = cam.R[2]; nrm = math.hypot(fw[0], fw[1])
        _cast(h, w, cam.f, cam.cx, cam.cy, *cam.eye, fw[0] / nrm, fw[1] / nrm, float(t), self.W_, np.zeros(8) if ring is None else np.array([*ring] + [0.] * (8 - len(ring)), float), np.zeros(5) if dish is None else np.asarray(dish, float), near, far, D, Z, GX, GY, AM); return D, Z, GX, GY, AM

    def draw(self, cam, t, pal='night', light=(0, 1., .25), ring=None, dish=None, road=1., road_w=.10, road_col=None, fog=None, foam=.0, h=H, w=W, sky=.5, lit_bias=0., far=6000., near=.6, sss=None, crest=None, light2=None, streak=None):
        pal = _pw(pal); D, Z, GX, GY, AM = self.cast(cam, t, ring, dish, near, far, h, w); m = D > 0; y, x = np.mgrid[0:h, 0:w].astype(f32); fw = cam.R[2]; nrm = math.hypot(fw[0], fw[1]); fx, fy = fw[0] / nrm, fw[1] / nrm; rx, ry = fy, -fx; u = (x + .5 - cam.cx) / cam.f; vx = fx + rx * u; vy = fy + ry * u; vz = -(y + .5 - cam.cy) / cam.f; vl = np.sqrt(vx * vx + vy * vy + vz * vz); vx, vy, vz = vx / vl, vy / vl, vz / vl
        nl = np.sqrt(GX * GX + GY * GY + 1); nx, ny, nz = -GX / nl, -GY / nl, 1 / nl; L = unit(light); hx, hy_, hz = L[0] - vx, L[1] - vy, L[2] - vz; hl = np.sqrt(hx * hx + hy_ * hy_ + hz * hz); spec = (nx * hx + ny * hy_ + nz * hz) / hl
        away = nx * fx + ny * fy                              # + = face tilts away from us: shows sky; - = we look into the water
        s1 = away + lit_bias; aa = lambda v, th: (lambda g: np.clip((v - th) / g + .5, 0, 1))(np.clip(np.abs(cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=1)) + np.abs(cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=1)), .004, .4) * .6)
        col = np.empty((h, w, 3), f32); col[:] = C(pal['mid']); k = aa(-s1.astype(f32), .055)[..., None]; col = col * (1 - k) + C(pal['deep']) * k; k = aa(s1.astype(f32), .11)[..., None] * sky * 1.0; col = col * (1 - k) + C(pal['lit']) * k
        if sss is not None: k = aa((Z - sss[1]).astype(f32), 0.)[..., None] * .8; col = col * (1 - k) + C(sss[0]) * k
        if fog is not None: k = (1 - np.exp(-np.maximum(0, D - fog[1]) * fog[2]))[..., None]; col = col * (1 - k) + C(fog[0]) * k
        if road > 0:
            sp = spec.astype(f32); gr = np.abs(cv2.Sobel(sp, cv2.CV_32F, 1, 0, ksize=1)) + np.abs(cv2.Sobel(sp, cv2.CV_32F, 0, 1, ksize=1)); rc = C(road_col or pal['foam']); br = noise_img(h, w, 9., t * 1.3, 8, 2, 2, 7.) * .5     # far water: the road is cut into strips, never a soft fan
            for th, cc, al in ((1 - road_w * .24, pal['lit'], .85), (1 - road_w * .10, rc, 1.)):
                v = sp + br * road_w * .05 * np.clip(D / 60, .15, 1.6); g = np.clip((v - th) / np.clip(gr, .0006, .03) / .8 + .5, 0, 1) * road * al; col = col + (C(cc) * (1.1 if al == 1. else 1.) - col) * g[..., None]
        if light2 is not None:                                 # a second source (moon): dict(dir, road, width, skin, col)
            L2 = unit(light2['dir']); gx, gy, gz = L2[0] - vx, L2[1] - vy, L2[2] - vz; sp2 = ((nx * gx + ny * gy + nz * gz) / np.sqrt(gx * gx + gy * gy + gz * gz)).astype(f32); gr = np.abs(cv2.Sobel(sp2, cv2.CV_32F, 1, 0, ksize=1)) + np.abs(cv2.Sobel(sp2, cv2.CV_32F, 0, 1, ksize=1)); br = noise_img(h, w, 9., t * 1.3 + 3, 9, 2, 2, 7.) * .5
            for th, cc, al in ((1 - light2['width'] * .24, light2['skin'], .85), (1 - light2['width'] * .10, light2['col'], 1.)): g = np.clip((sp2 + br * light2['width'] * .05 * np.clip(D / 60, .15, 1.6) - th) / np.clip(gr, .0006, .03) / .8 + .5, 0, 1) * light2['road'] * al; col = col + (C(cc) - col) * g[..., None]
        if ring is not None and ring[3] > 0 and streak is not None:   # water beaten flat and white ahead of the front: streaks run out along the rays
            Xr = cam.eye[0] + (fx + rx * u) * D - ring[0]; Yr = cam.eye[1] + (fy + ry * u) * D - ring[1]; dr = np.sqrt(Xr * Xr + Yr * Yr) + 1e-6; Sg = dr - ring[2]; an = np.arctan2(Yr, Xr); k = np.exp(-np.maximum(Sg, 0) / max(streak[1], 1e-3)) * (Sg > -2); ns = noise(an * streak[2], np.log(np.maximum(D, 1.)) * 1.2 - t * 5., 2.).astype(f32) * .6 + noise(an * streak[2] * 3.1, np.log(np.maximum(D, 1.)) * 2. - t * 8., 5.).astype(f32) * .4
            for q, cc in ((.10, pal['lit']), (.34, pal['shade']), (.62, pal['foam']), (.86, pal['white'])): col = col + (C(cc) - col) * (np.clip((k + .34 * ns * (k > .02) - q) / .03 + .5, 0, 1) * m)[..., None]
        if dish is not None and dish[3] > 0:                   # the pushed bowl: streaks run out from the centre, the rim is torn white
            Xd = cam.eye[0] + (fx + rx * u) * D - dish[0]; Yd = cam.eye[1] + (fy + ry * u) * D - dish[1]; rd = np.sqrt(Xd * Xd + Yd * Yd) / dish[2]; an = np.arctan2(Yd, Xd); ns = noise(np.cos(an)[::2, ::2] * 16, np.sin(an)[::2, ::2] * 16, rd[::2, ::2] * 2.2 - t * 3.).astype(f32); ns = cv2.resize(ns, (w, h), interpolation=cv2.INTER_LINEAR); k = np.clip((ns - .22 - .5 * np.clip(.35 - rd, 0, 1)) / .05, 0, 1) * (rd < 1.04) * m; col = col * (1 - k[..., None] * .85) + C(pal['lit']) * k[..., None] * .85; k = np.clip((ns - .46) / .05, 0, 1) * (rd < 1.02) * (rd > .3) * m; col = col * (1 - k[..., None]) + C(pal['foam']) * k[..., None]; k = np.clip((.07 + .10 * ns - np.abs(rd - 1.12)) / .012, 0, 1) * m; col = col * (1 - k[..., None]) + C(pal['white']) * k[..., None]
        if crest is not None: k = np.clip((Z - crest) / .06 + .5, 0, 1) * (np.clip(GX * 0 + 1, 0, 1)); n = noise_img(h, w, 22, t * .8, 5, 2, 2, 3.); k = k * np.clip(n * 6 + .5, 0, 1); col = col * (1 - k[..., None]) + C(pal['foam']) * k[..., None]
        X = cam.eye[0] + (fx + rx * u) * D; Y = cam.eye[1] + (fy + ry * u) * D; buf = dict(D=D, Z=Z, X=X, Y=Y, GX=GX, GY=GY, AM=AM, away=s1, spec=spec)
        if ring is not None and ring[3] > 0: buf['S'] = np.hypot(X - ring[0], Y - ring[1]) - ring[2]
        return col.astype(f32), m.astype(f32), buf


# ------------------------------------------------------------- jets
def _ribbon3(cam, P, wr):
    """world polyline + world radii -> screen centre, normals, half widths (px)"""
    q, z = cam.project(P); hw = np.asarray(wr) * cam.f / np.maximum(z, .1); T = np.gradient(q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-6; return q, np.stack([-T[:, 1], T[:, 0]], -1), hw, z


def jet_path(x0, v0, t, em, g=G_, drag=.35, K=18, decay=.70, wob=0., seed=0, floor=0.):
    """stream from a nozzle that slows down while it lasts.  -> points base..tip (only airborne part), age s, emission e"""
    e1 = min(t, em); e = np.linspace(e1, 0., K); s = t - e; k = (1 - decay * (e / em) ** .8)[:, None]; dk = ((1 - np.exp(-drag * s)) / drag)[:, None]; P = np.asarray(x0, float) + np.asarray(v0, float) * k * dk; P[:, 2] -= .5 * g * s * s * (1 - .2 * np.minimum(1, drag * s))
    if wob: P[:, 0] += wob * noise(e * 9, seed * 1.7, 0.) * s; P[:, 2] += wob * .6 * noise(e * 9, seed * 1.7 + 40, 0.) * s
    ok = P[:, 2] >= floor - 1e-3
    if not ok.any(): return None
    return P[ok], s[ok], e[ok]


def draw_jet(cel, cam, x0, v0, t, em, rad, pal, g=G_, drag=.35, decay=.70, seed=0, side=1., ink=True, break_age=.30, wob=.25, floor=0., beads=True):
    """one ballistic jet: tapered ribbon, shadow side, tip that breaks into beads"""
    J = jet_path(x0, v0, t, em, g, drag, 20, decay, wob, seed, floor)
    if J is None or len(J[0]) < 4: return
    P, s, e = J; cut = s < break_age * (.8 + .5 * h1(seed, 1)); n = int(cut.sum()); done = t > em
    r = rad * (.35 + .65 * np.clip(e / em, 0, 1)) * (1 + .25 * np.sin(s * 40 + seed)) * np.clip(1.25 - s * .55, .3, 1)
    if n >= 4:
        q, nn, hw, z = _ribbon3(cam, P[:n], r[:n]); v = np.linspace(0, 1, n); hw = hw * np.minimum(1, (1 - v) * 3.2 + .06) ** .7 * (np.minimum(1, v * 6 + .3) if done else 1)
        A, B = q + nn * hw[:, None], q - nn * hw[:, None]
        if side < 0: A, B = B, A
        if ink: cel.fill(np.vstack([A + nn * .3 * side, (B - nn * side * np.minimum(1.4, .35 * hw + .3)[:, None])[::-1]]), pal['ink'], .75)
        cel.fill(np.vstack([A, B[::-1]]), pal['foam']); cel.fill(np.vstack([q + (B - q) * .12, B[::-1]]), pal['shade']); cel.fill(np.vstack([A, (A + (q - A) * .55)[::-1]]), pal['white'])
    if beads:
        for i in range(max(0, n - 1), len(P)):
            if h1(seed, i, 3) < .55: continue
            p = P[i] + (np.array([h1(seed, i, 4), 0, h1(seed, i, 5)]) - .5) * rad * (1.2 + 3 * s[i]); drop(cel, cam, p, rad * (.14 + .46 * h1(seed, i, 6) ** 2.2), pal, v=(np.asarray(v0) * math.exp(-drag * s[i]) * .8 - np.array([0, 0, g * s[i]])))


def drop(cel, cam, p, r, pal, v=None, stretch=.022):
    """one flying drop: sliver along its velocity, hard light and shade; tiny ones are plain white slivers"""
    q, z = cam.project(np.asarray(p, float)[None]); z = float(z[0])
    if z < .2: return
    q = q[0]; R_ = r * cam.f / z
    if R_ < .4 or not (-40 < q[0] < cam.w + 40 and -40 < q[1] < cam.h + 40): return
    if v is None: d = np.array([0, 1.]); l = 1.
    else: q2 = cam.project((np.asarray(p, float) - np.asarray(v, float) * stretch)[None])[0][0]; d = q2 - q; l = math.hypot(*d); d = d / (l + 1e-6); l = max(1.0, min(3.4, .9 + l / (R_ * 1.6 + .6)))
    n = np.array([-d[1], d[0]])
    if R_ < 2.6: n = n * R_ * .85; cel.fill(np.array([q - d * R_, q - d * R_ * .4 + n * .9, q + d * R_ * .4 + n * .6, q + d * R_ * (.6 + l), q + d * R_ * .4 - n * .6, q - d * R_ * .4 - n * .9]), pal['white']); return
    a = np.linspace(0, TAU, 16, endpoint=False); ca, sa = np.cos(a), np.sin(a); pts = lambda k, ox, oy: q + (np.outer(ca * R_ * k * np.where(sa > 0, 1 - .55 * sa ** 2 * min(1, l - 1), 1) + R_ * ox, n) + np.outer(np.where(sa > 0, sa * l, sa) * R_ * k + R_ * oy, d)); cel.fill(pts(1., 0, 0), pal['shade']); cel.fill(pts(.84, .10, -.06), pal['foam']); cel.fill(pts(.40, .26, -.22), pal['white'])
    if R_ > 5: cel.line(np.vstack([pts(1., 0, 0), pts(1., 0, 0)[:1]]), max(.8, R_ * .12), pal['ink'], taper=(.3, .3), press=.4, alpha=.7, tremor=0.)


def ripple(cel, cam, xy, t, speed=1.1, n=3, gap=.16, w=.05, life=1.6, pal='night', z=0., seed=0, amp=1., start=0., light=(-.6, .8)):
    """rings on water: light and dark crescents with real thickness, spreading, thinning"""
    pal = _pw(pal); ld = np.asarray(light, float)
    for k in range(n):
        s = t - k * gap
        if s <= 0.02 or s > life: continue
        r = start + speed * s; q = s / life; a = np.linspace(0, TAU, 64); c_ = np.cos(a); s_ = np.sin(a); f = c_ * ld[0] + s_ * ld[1]; wob = 1 + .02 * np.sin(a * 5 + seed + k * 2); ww = w * amp * (1 - q) ** 1.2 * (1 - .22 * k)
        for sg, col, al in ((1, pal['foam'], 1.), (-1, pal['ink'], .8)):
            th = ww * np.clip(sg * f, 0, 1) ** .7 * .9; i = th > ww * .04; idx = np.where(i)[0]
            if len(idx) < 4: continue
            roll = np.where(np.diff(idx) > 1)[0]; idx = np.r_[idx[roll[0] + 1:], idx[:roll[0] + 1]] if len(roll) and not (roll[0] == 0) else idx
            R1, R0 = r * wob[idx] + th[idx], r * wob[idx] - th[idx] * .5; A = cam.project(np.c_[xy[0] + c_[idx] * R1, xy[1] + s_[idx] * R1, np.full(len(idx), z)])[0]; B = cam.project(np.c_[xy[0] + c_[idx] * R0, xy[1] + s_[idx] * R0, np.full(len(idx), z)])[0]; cel.fill(np.vstack([A, B[::-1]]), col, al * (1 - q) ** .8)
    return cel


def splash(cel, cam, t, xy, v0=4., r0=.25, n=14, rad=.07, em=.16, lean=.55, pal='day', seed=0, sheet=.45, jetv=.8, rings=3, z=0., g=G_, dirn=None, ring_speed=1.6, side=1., ink=1.6):
    """crown splash as one drawn wall: inner far side in shade, outer near side in light, curved fingers on the rim, every point of it a true ballistic path; fingers let go of beads,
    rings run out, a central jet rises after.  v0 m/s sets the scale, lean = outward / upward, n fingers, sheet = how far up the wall is closed, dirn = (x, y) pushes it to one side"""
    if t <= 0: return cel
    pal = _pw(pal); xy = np.asarray(xy, float); push = np.zeros(2) if dirn is None else np.asarray(dirn, float); e1 = min(t, em); T_ = 2 * v0 / g; open_ = clamp(1.25 - t / (T_ * .95)); life = t / T_
    def P(az, k, hgt):                                       # point of the wall: azimuth, 0 base .. 1 rim
        az, k, hgt = np.broadcast_arrays(np.asarray(az, float), np.asarray(k, float), np.asarray(hgt, float)); e = e1 * (1 - k); s = t - e; d = np.stack([np.cos(az), np.sin(az)], -1); pv = 1 + .40 * (d @ push); v = v0 * (1 - .62 * (e / em) ** .8) * (.82 + .26 * hgt) * pv; R_ = r0 * (1 + .55 * e / em) + lean * v * s + .10 * v0 * np.minimum(s, .2); zz = z + v * s - .5 * g * s * s; X = xy + d * R_[..., None] + push * (v0 * .42 * s)[..., None]; return np.concatenate([X, np.maximum(zz, z)[..., None]], -1), zz > z - .004
    hk = lambda az: np.array([h1(seed, int(v), 7) for v in np.floor(np.asarray(az) / TAU * n)])
    if rings: ripple(cel, cam, xy + push * v0 * .2 * t, t, ring_speed, rings, .14, r0 * .26, 1.4, pal, z, seed, 1., r0)
    pj = lambda X: cam.project(X)[0]
    def half(a0, a1, cols, sdx):
        az = np.linspace(a0, a1, 90); base, _ = P(az, 0., 1.); base_s = pj(base)
        for lo, am, pt, c, al, off in cols:
            q = claw_edge(a0, a1, TAU / n * pt, seed * 5 + sdx, .10, am, 9); ok = (q[0] >= a0) & (q[0] <= a1); A, Q = q[0][ok], q[1][ok]
            if len(A) < 6: continue
            k = np.clip(lo + (1 - lo) * Q * (1 - .62 * clamp((life - .40) / .45)), 0, 1); W3, up = P(A, k, .55 + .9 * Q * hk(A + .01)); poly = np.vstack([base_s[:1], pj(W3), base_s[::-1]]); cel.fill(poly + [0, off], c, al)
    inner = [(sheet * open_, (.45, 1.), 1., pal['ink'], .6, -ink), (sheet * open_, (.45, 1.), 1., pal['shade'], 1., 0), (.10, (.1, .75), .62, pal['lit'], 1., 0)]; outer = [(sheet * open_, (.45, 1.), 1., pal['ink'], .75, ink), (sheet * open_, (.45, 1.), 1., pal['foam'], 1., 0), (sheet * .25 * open_, (.2, .80), .5, pal['shade'], 1., 0), (.02, (.08, .42), .36, pal['white'], 1., 0)]
    if life < 1.05: half(.0, math.pi, inner, 1)
    if jetv > 0 and t > em * 1.2: draw_jet(cel, cam, np.r_[xy + push * v0 * .3 * t, z], np.array([push[0] * .3, 0, v0 * 1.35 * jetv]), t - em * 1.2, em * 2.4, rad * 2.6, pal, g, .2, .85, seed + 99, side, break_age=.6, wob=.04, floor=z)
    if life < 1.05: half(math.pi, TAU, outer, 2)
    for i in range(int(n * 2.4)):                            # beads: thrown off the fingers
        az = h1(seed, i, 21) * TAU; u = h1(seed, i, 22); d = np.array([math.cos(az), math.sin(az)]); vv = v0 * (.85 + .62 * h1(seed, i, 23)); v = np.r_[d * lean * vv * (.7 + .8 * u) + push * v0 * .5, vv]; s = t - .03 * h1(seed, i, 24); p = np.r_[xy + d * r0, z] + v * s - np.array([0, 0, .5 * g * s * s])
        if s > .05 and p[2] > z: drop(cel, cam, p, rad * (.10 + .48 * h1(seed, i, 25) ** 3), pal, v - np.array([0, 0, g * s]))
    return cel


def column(cel, cam, xy, z0, z1, r, t, pal='lake', seed=0, flow=6., collar=True, dome=.5, wob=.03, flare=.6, ink=2.2, bottom=None):
    """a clean column of water from z0 to z1 (metres): flat tones, flow lines that run up (flow m/s, negative = falling), rounded head, flared foot.
    bottom = None (stands on water) or 'round' (a slug hanging in the air)"""
    pal = _pw(pal); H_ = z1 - z0
    if H_ <= .02: return cel
    hd0 = min(r * dome, H_ * .45); z = np.r_[np.linspace(z0, z1 - hd0, 24)[:-1], z1 - hd0 + hd0 * np.sin(np.linspace(0, math.pi / 2, 16))]; K = len(z); v = (z - z0) / H_; px = cam.scale(np.array([xy[0], xy[1], (z0 + z1) / 2])); hd = min(r * dome, H_ * .45); prof = np.where(z > z1 - hd, np.sqrt(np.clip(1 - ((z - (z1 - hd)) / hd) ** 2, 0, 1)), 1.)
    if bottom == 'round': hb = min(r * .8, H_ * .4); prof = prof * np.where(z < z0 + hb, np.sqrt(np.clip(1 - ((z0 + hb - z) / hb) ** 2, 0, 1)), 1.)
    else: prof = prof * (1 + flare * np.exp(-(z - z0) / (r * .7)))
    sway = wob * r * (noise(z / r * .5 - t * flow / r * .5, seed * 3.1) + .5 * noise(z / r * 1.3 - t * flow / r * 1.3, seed * 3.1 + 9)); wl = r * prof * (1 + .04 * noise(z / r * 1.1 - t * flow / r * 1.1, 5., seed)); wr = r * prof * (1 + .04 * noise(z / r * 1.1 - t * flow / r * 1.1, 25., seed)); c = np.c_[xy[0] + sway, np.full(K, xy[1]), z]; q, zz = cam.project(c); s = cam.f / np.maximum(zz, .1); L_ = q - np.c_[wl * s, np.zeros(K)]; R_ = q + np.c_[wr * s, np.zeros(K)]; Q = np.vstack([L_, R_[::-1]])
    if ink: cel.fill(np.vstack([L_ - [ink * .55, 0], (R_ + [ink, 0])[::-1]]), pal['ink'])
    cel.fill(Q, pal['lit']); cel.clip(Q); w = (wl + wr) * .5 * s
    tip = lambda a, b, col, al=1.: cel.fill(np.vstack([q + np.c_[w * a, np.zeros(K)], (q + np.c_[w * b, np.zeros(K)])[::-1]]), col, al)
    tip(.34 + .10 * noise(v * 7 - t * flow * .3, 2.), 1.2, pal['mid']); tip(.74 + .08 * noise(v * 9 - t * flow * .4, 4.), 1.2, pal['deep']); tip(-.90, -.66 + .08 * noise(v * 8 - t * flow * .3, 6.), pal['foam']); tip(-.60, -.53, pal['foam'])
    n = int(max(6, H_ / r * 4))
    for i in range(n):                                       # flow lines run with the water
        u = h1(seed, i, 1); per = H_ * 1.3; zc = z0 - .15 * H_ + ((h1(seed, i, 2) * per + t * flow * (.8 + .4 * h1(seed, i, 3))) % per); l = r * (.5 + 1.6 * h1(seed, i, 4)); a = -.85 + 1.7 * u; zs = np.linspace(zc, zc + l, 8); cc = np.c_[np.interp(zs, z, c[:, 0]) + a * r * np.interp(zs, z, prof), np.full(8, xy[1]), zs]; thin(cel, cam.project(cc)[0], max(1.2, r * px * .07 * (.4 + h1(seed, i, 5))), pal['white'] if u < .62 else pal['deep'], .9, .3, .5, 10)
    cel.unclip()
    tp = cam.project(np.array([[xy[0] - r * .34 + sway[-1], xy[1], z1 - hd * .52]]))[0][0]; a = np.linspace(0, TAU, 20); cel.fill(np.c_[tp[0] + np.cos(a) * r * px * .26, tp[1] + np.sin(a) * r * px * .13], pal['white'])
    if collar and bottom is None:
        for i in range(9): th = (i + .5) / 9 * TAU; k = (t * 1.7 + h1(seed, i, 7)) % 1.; draw_jet(cel, cam, np.array([xy[0] + math.cos(th) * r * 1.45, xy[1] + math.sin(th) * r * 1.45, z0]), np.array([math.cos(th) * 1.4, math.sin(th) * 1.4, 2.6]) * math.sqrt(r) * (.7 + .5 * h1(seed, i, 8)), .05 + k * .55 * math.sqrt(r), .22, r * .13, pal, seed=seed + i, floor=z0)
    return cel


def column_fall(cel, cam, t, xy, z_top, z_hit, r, length, pal='lake', seed=0, v0=0., ground=0.):
    """a slug of water of `length` metres hangs with its foot at z_top, drops, and breaks on something at z_hit (a head) into an umbrella that falls to `ground`"""
    z = z_top - v0 * t - .5 * G_ * t * t; v = v0 + G_ * t; pal = _pw(pal)
    if z > z_hit: return column(cel, cam, xy, z, z + length, r, t, pal, seed, -2., False, bottom='round')
    th = (-v0 + math.sqrt(v0 * v0 + 2 * G_ * (z_top - z_hit))) / G_; s = t - th; v = v0 + G_ * th; left = length - (z_hit - z)
    n = 26; em = length / max(v, .1); on = min(s, em); tail = max(0., s - em)
    vo, vu = v * .36, v * .10; th_ = np.linspace(0, TAU, n, endpoint=False)
    def bell(sd):                                            # steady umbrella: every meridian is a true parabola
        ss_ = np.linspace(tail, min(s, .9), 12); K = []
        for i in range(n): k = .80 + .30 * h1(seed, i, sd, 2); zz = z_hit + vu * k * ss_ - .5 * G_ * ss_ ** 2; R_ = r * .75 + vo * k * ss_; K.append(np.c_[xy[0] + np.cos(th_[i]) * R_, xy[1] + np.sin(th_[i]) * R_, np.maximum(zz, ground)])
        return np.array(K)
    K = bell(0)
    if left > 0: column(cel, cam, xy, z_hit - r * .1, z_hit + left, r, t, pal, seed, -v * .6, False, dome=.35, flare=.45, ink=0)
    for far in (True, False):
        for i in range(n):
            j = (i + 1) % n; my = math.sin((th_[i] + th_[i] + TAU / n) / 2)
            if (my > 0) != far: continue
            A, B_ = K[i], K[j]; m = max(3, int(12 * (.62 + .38 * h1(seed, i, 4)))); q = cam.project(np.vstack([A[:m], B_[:m][::-1]]))[0]; lit = .5 - .5 * math.cos(th_[i] - 2.4); cel.fill(q, pal['shade'] if far else (pal['foam'] if lit > .40 else pal['lit']), .95)
        for i in range(n):
            if (math.sin(th_[i]) > 0) != far: continue
            q, nn, hw, zz = _ribbon3(cam, K[i], np.full(12, r * (.06 + .10 * h1(seed, i, 5)))); v_ = np.linspace(0, 1, 12); hw = hw * (.5 + .9 * v_) * np.minimum(1, (1 - v_) * 5 + .25); cel.fill(np.vstack([q + nn * hw[:, None], (q - nn * hw[:, None])[::-1]]), pal['white'] if not far else pal['foam']); cel.fill(np.vstack([q - nn * hw[:, None], (q - nn * hw[:, None] * .3)[::-1]]), pal['shade']); e = K[i][-1]
            if e[2] > ground + .02:
                for j in range(2): drop(cel, cam, e + np.array([math.cos(th_[i]), math.sin(th_[i]), -.4]) * r * (.2 + .5 * h1(seed, i, j, 6)), r * (.04 + .09 * h1(seed, i, j, 7) ** 2), pal, np.r_[np.cos(th_[i]) * vo, np.sin(th_[i]) * vo, -G_ * s * .6])
    return cel


# ------------------------------------------------------------- skimming
def wake(cel, cam, t, path, pal='day', life=1.5, side=2.4, up=3.2, back=.5, rad=.05, plume=1.0, seed=0, dt=.022, z=0., v_wake=True, v_ang=.34, w0=.10, speed_hint=12.):
    """a hand / body skims water.  path(e seconds) -> (x, y) contact point at time e.  Two curved spray walls in true arcs, a rooster tail behind, beads, and the V wake on the surface."""
    pal = _pw(pal); ks = np.arange(math.floor((t - life) / dt), math.floor(t / dt) + 1); E = np.unique(np.r_[ks * dt, t]); E = E[(E >= t - life) & (E >= 0)]
    if len(E) < 4: return cel
    P = np.array([path(e) for e in E]); P2 = np.array([path(e + .01) for e in E]); T = P2 - P; T /= np.linalg.norm(T, axis=1)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]; S = t - E; hid = np.round(E / dt).astype(int)
    if v_wake:
        dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[::-1], axis=0), axis=1))][::-1]
        for sd in (-1, 1):
            off = (w0 + dist * math.tan(v_ang) * .62); wd = (w0 * .55 + dist * .05) * (1 - S / life) ** .7; jag = np.array([.5 + h1(seed, k, sd + 1, 1) for k in hid]); A = P + N * (sd * (off + wd * jag))[:, None]; B = P + N * (sd * (off - wd * .4))[:, None]; D = P + N * (sd * (off - wd * 1.6))[:, None]; pr = lambda X: cam.project(np.c_[X, np.full(len(X), z)])[0]
            cel.fill(np.vstack([pr(B), pr(D)[::-1]]), pal['deep'], .55); cel.fill(np.vstack([pr(A), pr(B)[::-1]]), pal['foam']); cel.fill(np.vstack([pr(A), pr(A * .6 + B * .4)[::-1]]), pal['white'])
        for k, (p, n_, s) in enumerate(zip(P, N, S)):        # churned foam down the middle
            if h1(seed, hid[k], 7) > .55 or s > life * .9: continue
            rr = w0 * (.6 + 2.2 * s / life) * (.5 + h1(seed, hid[k], 8)); c = np.r_[p + n_ * (h1(seed, hid[k], 9) - .5) * w0 * 4 * (.3 + s), z]; a = np.linspace(0, TAU, 14); W3 = np.c_[c[0] + np.cos(a) * rr * (1.2 + .3 * np.sin(a * 3 + k)), c[1] + np.sin(a) * rr * (1 + .3 * np.sin(a * 2 + k)), np.full(14, z)]; cel.fill(cam.project(W3)[0], pal['foam'], .85 * (1 - s / life))
    def sheet(sd, vs, vu, vb, col, shade, amp, sk):
        V = N * (sd * vs); env = .72 + .28 * noise(E / dt / 9., sk * 3.1, sd * 2.); Zz = z + vu * env * S - .5 * G_ * S * S; ok = Zz > z - .01; n = int(ok[::-1].argmin()) if not ok.all() else len(ok); n = max(n, 5); sl = slice(len(S) - n, None); X = P + V * S[:, None] - T * (vb * S)[:, None]; Tp = cam.project(np.c_[X, np.maximum(Zz, z)])[0][sl]; Bp = cam.project(np.c_[P + V * (S * .22)[:, None], np.full(len(P), z)])[0][sl]; s_ = E[sl]
        for (pt, lo, am, c, al, sdx) in ((dt * 4.2, .30, (.30, 1.), pal['ink'], .55, 1), (dt * 4.2, .30, (.30, 1.), col, 1., 1), (dt * 2.8, .12, (.1, .66), shade, 1., 2), (dt * 2.1, .02, (.05, .34), pal['white'], 1., 3)):
            q = claw_edge(s_[0], s_[-1], pt, seed * 7 + sd + sk * 5 + sdx, -.95, am, 9); poly = claw_band(s_, Bp, Tp, q[0], q[1], lo)
            if poly is not None: cel.fill(poly + ([0, 1.4] if c is pal['ink'] else [0, 0]), c, al)
        for k in range(len(S) - n, len(S)):                  # beads leave the claw tips in their own arcs
            for j in range(2):
                if h1(seed, hid[k], sd + 2, sk, j, 11) < .52: continue
                q = h1(seed, hid[k], sd + 2, sk, j, 12); st = .75 + .6 * q; zz = z + vu * st * S[k] - .5 * G_ * S[k] ** 2
                if zz > z: xy = P[k] + V[k] * S[k] * (.7 + .7 * h1(seed, hid[k], j, sk, 13)) - T[k] * vb * S[k] * (.6 + .8 * h1(seed, hid[k], j, sk, 14)); drop(cel, cam, np.r_[xy, zz], rad * (.18 + .8 * h1(seed, hid[k], j, sk, 15) ** 3), pal, np.r_[V[k] - T[k] * vb, vu * st - G_ * S[k]])
    far = -1 if (cam.cam(np.r_[P[-1] + N[-1], z])[2] < cam.cam(np.r_[P[-1] - N[-1], z])[2]) else 1
    sheet(far, side, up, back, pal['foam'], pal['shade'], 1., 1)
    if plume > 0:
        sheet(far, side * .18, up * 1.55 * plume, -back * 3., pal['white'], pal['foam'], 1., 3); sheet(-far, side * .18, up * 1.5 * plume, -back * 3., pal['foam'], pal['shade'], 1., 4)
    sheet(-far, side, up, back, pal['white'], pal['shade'], 1., 2)
    return cel


def squeeze(cel, t, gaps, pal='grey', em=.16, v0=900., seed=0, rad=7., g=2600.):
    """a fist closes in a puddle: water is forced out between the fingers as thin torn fans.  gaps = [(x, y, angle radians on screen)] px; v0, g in px/s.  Screen-space, true arcs."""
    if t <= 0: return cel
    pal = _pw(pal); cam = flat()
    for i, (x, y, a) in enumerate(gaps):
        J = []
        for j in range(5):
            aa = a + (j - 2) * .20 + .10 * (h1(seed, i, j) - .5); v = v0 * (.55 + .60 * h1(seed, i, j, 1)) * (1 - .10 * abs(j - 2)); J.append((np.array([math.cos(aa) * v, 0, -math.sin(aa) * v]), em * (.8 + .5 * h1(seed, i, j, 2)), rad * (.5 + .8 * h1(seed, i, j, 3))))
        for j in range(4):                                   # the web between two streams
            A = jet_path(np.array([x, 0., -y]), J[j][0], t, J[j][1], g, .8, 14, .75, 0, 0, -1e9); B_ = jet_path(np.array([x, 0., -y]), J[j + 1][0], t, J[j + 1][1], g, .8, 14, .75, 0, 0, -1e9); m = max(3, int(14 * clamp(.80 - 2.6 * max(0., t - em)))); PA, PB = A[0][:m], B_[0][:m]; c = (PA[-1] + PB[-1]) * .5; c = c + (np.array([x, 0., -y]) - c) * .22; cel.fill(cam.project(np.vstack([PA, c[None], PB[::-1]]))[0], pal['shade'] if j % 2 else pal['lit'], .88) if t < em * 2.4 else None
        for j in range(5): draw_jet(cel, cam, np.array([x, 0., -y]), J[j][0], t, J[j][1], J[j][2], pal, g, .8, .75, seed * 7 + i * 11 + j, 1 if J[j][0][0] > 0 else -1, break_age=.22, wob=40., floor=-1e9)
        for j in range(12): aa = a + (h1(seed, i, j, 11) - .5) * 1.5; v = v0 * (.5 + .9 * h1(seed, i, j, 12)); p = np.array([x + math.cos(aa) * v * t, 0, -y - math.sin(aa) * v * t - .5 * g * t * t]); drop(cel, cam, p, rad * (.12 + .5 * h1(seed, i, j, 13) ** 3), pal, np.array([math.cos(aa) * v, 0, -math.sin(aa) * v - g * t]), .018)
    return cel


def pull(cel, cam, t, orb, srcs, pal='blue', seed=0, life=1.1, turns=.6, rad=.07, z=0., every=.35, start=0., strength=1.):
    """water climbs toward an orb: peaks lift off the surface, strands spiral in, thin, break into beads that fall into the light.  srcs = [(x, y)] on the water; orb = (x, y, z)"""
    pal = _pw(pal); orb = np.asarray(orb, float)
    for i, (x, y) in enumerate(srcs):
        ph = h1(seed, i, 1) * every; n = int(life / every) + 2; k0 = math.floor((t - start - ph) / every); pk = min(1., max(0., t - start) * 1.5) * rad * 7 * (.8 + .2 * math.sin(t * 9 + i)) * strength; p0 = np.array([x, y, z + pk * 1.6]); a = np.linspace(-1, 1, 25); prof = np.maximum(0, (1 - np.abs(a)) ** 3.2); tilt = np.clip((orb - [x, y, z])[0], -2, 2) * .09 * prof; q, zz = cam.project(np.c_[x + a * rad * 6.5 + tilt * 2, np.full(25, y), z + prof * pk * 1.7]); sd_ = 1 if orb[0] > x else -1; cel.fill(q, pal['lit']); h_ = q[12:] if sd_ > 0 else q[:13][::-1]; cel.fill(np.vstack([h_, (h_ * [.5, 1] + [q[12, 0] * .5, 0])[::-1]]), pal['foam']); thin(cel, q, 2.2, pal['foam'], .9, .4, .4)
        for k in range(k0, k0 - n, -1):
            s = t - start - ph - k * every
            if k < 0 or not 0 < s < life * 1.25: continue
            hd = s / life; tl = hd - .42; u = np.linspace(max(0, tl), min(1., hd), 20)
            if u[-1] - u[0] < .03: continue
            e = u * u * (.35 + .65 * u); sgn = 1 if h1(seed, i, 2) > .5 else -1; d = orb - p0; per = np.cross(d, [0, 1., .2]); per /= np.linalg.norm(per) + 1e-9; per2 = np.cross(unit(d), per); th = sgn * turns * TAU * u + h1(seed, i, k, 3) * 1.5; R_ = np.linalg.norm(d) * .16 * np.sin(math.pi * np.clip(e, 0, 1)) * (.6 + .8 * h1(seed, i, k, 4)); P = p0 + d * e[:, None] + per * (np.cos(th) * R_)[:, None] + per2 * (np.sin(th) * R_)[:, None]; rr = rad * (.45 + .7 * h1(seed, i, k, 5)) * (1 - .55 * e) * np.sin(np.linspace(.2, math.pi - .03, 20)) ** .55 * 1.5; qq, nn, hw, zz = _ribbon3(cam, P, rr); sd = np.sign(nn @ (cam.project(orb[None])[0][0] - qq.mean(0))).mean() or 1.; A, B = qq + nn * hw[:, None] * sd, qq - nn * hw[:, None] * sd
            cel.fill(np.vstack([A, (B - nn * sd * 1.1)[::-1]]), pal['ink'], .7); cel.fill(np.vstack([A, B[::-1]]), pal['lit']); cel.fill(np.vstack([A, (A * .45 + B * .55)[::-1]]), pal['foam']); cel.fill(np.vstack([A, (A * .80 + B * .20)[::-1]]), pal['white'])
            for j in range(3): uu = min(1., hd + .04 + .05 * j); e_ = uu * uu * (.35 + .65 * uu); pb = p0 + d * e_ + per * (h1(seed, i, k, j, 6) - .5) * np.linalg.norm(d) * .08 * (1 - e_); drop(cel, cam, pb, rad * .35 * (1 - .5 * e_), pal, d / life, .03) if uu < .98 else None
    return cel


# ------------------------------------------------------------- a falling bell of water, broken
def water_bell(img, cx, top, T, dur, w=144., vx=1520., vy=560., g=8600., pal='lake', seed=0, life=.62, thin_=1., beads=110, see=.62, edge=1.):
    """A column breaks on something at (cx, top) px and opens into a bell: one torn, see-through sheet (it bends the picture behind it), thick streaks, holes that open as it thins,
    white rolled edges, fingers and beads.  T = seconds since the hit, dur = seconds the column keeps feeding it.  Every point is a true parabola (same law as strand u: x = cx + u (w + vx s), y = top - vy (1.1 - .55|u|) s + g s^2 / 2).
    img RGB -> RGB.  Nothing to draw before T > 0 or after dur + life."""
    pal = _pw(pal); h, W_ = img.shape[:2]; e0 = max(0., T - dur)
    if T <= .004 or e0 > life: return img
    s1 = min(T, life); ymax = int(min(h, top + .5 * g * s1 * s1 + 40)); ymin = int(max(0, top - .5 * (vy * 1.1) ** 2 / g - 40))
    if ymax - ymin < 8: return img
    y, x = np.mgrid[ymin:ymax, 0:W_].astype(np.float64); u = np.sign(x - cx) * .4; s = np.zeros_like(x); k = 1.
    for _ in range(6): b = vy * (1.1 - .55 * np.minimum(np.abs(u), 1.1)) * k; s = (b + np.sqrt(np.maximum(b * b + 2 * g * (y - top), 0))) / g; u = (x - cx) / (w + vx * k * s); k = .88 + .16 * np.interp(np.clip(u, -1.2, 1.2).ravel(), np.linspace(-1.2, 1.2, 160), noise(np.linspace(-1.2, 1.2, 160) * 4.2, seed * 1.7)).reshape(u.shape)
    e = T - s; ok = (np.abs(u) < 1.06) & (e > -.03) & (e < dur + .03) & (b * b + 2 * g * (y - top) >= 0)
    n1 = (noise(u * 19, e * 7., seed) + .55 * noise(u * 47, e * 15., seed + 3)) * .8 - .22 * u; n2 = noise(u * 9., e * 21., seed + 7) * .75 + .45 * noise(u * 21., e * 44., seed + 9) + .18 * noise(u * 52., e * 90., seed + 11); thr = np.interp(s, [0, .10, .22, .40, .62], [-2.2, -1.35, -.72, -.12, .38]) * 1.0 + .25 * (thin_ - 1)
    fr = .010 + .060 * np.clip(noise(u * 13., seed + 4.) + .5 * noise(u * 31., seed + 5.), 0, 1.4) ** 1.2 * np.clip(s / .25, .25, 1.2); tl = .012 + .050 * np.clip(noise(u * 11., seed + 6.) + .5 * noise(u * 27., seed + 2.), 0, 1.4); q = np.minimum(np.minimum(n2 - thr, (e - fr) * 14.), np.minimum((dur - e - tl) * 14. + (0 if T > dur else 5.), (1.0 - np.abs(u) - .05 * (1 + noise(e * 16., seed + 8.))) * 9.)); q = np.where(ok, q, -3.).astype(f32)
    gq = np.clip(np.abs(cv2.Sobel(q, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(q, cv2.CV_32F, 0, 1, ksize=3)), .016, 8.) / 8.; d = q / gq                                   # signed distance to the torn edge, px
    d = cv2.GaussianBlur(np.clip(d, -40, 40), (0, 0), .8); a = np.clip(d + .5, 0, 1)[..., None]; n = cv2.GaussianBlur(n1.astype(f32), (0, 0), 1.); gx, gy = cv2.Sobel(n, cv2.CV_32F, 1, 0, ksize=3) / 8, cv2.Sobel(n, cv2.CV_32F, 0, 1, ksize=3) / 8; mx, my = (x + np.clip(gx * 420, -34, 34)).astype(f32), (y + 10 + np.clip(gy * 420, -20, 20)).astype(f32); bg = cv2.remap(img.astype(f32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    body = bg * see + C(pal['lit']) * (1 - see) + .04; st = lambda v, t: np.clip((v - t) / np.clip(np.abs(gx) + np.abs(gy), .004, .06) / 1.4 + .5, 0, 1)[..., None]; th = np.clip(1.05 - s / .75, .25, 1)[..., None].astype(f32)
    body = body + (C(pal['shade']) - body) * st(-n, .30) * .55 * th; body = body + (C(pal['mid']) - body) * st(-n, .66) * .75 * th; body = body + (C(pal['foam']) - body) * st(n, .26 + .10 * (1 - th[..., 0])) * .92; body = body + (C(pal['white']) - body) * st(n, .62)
    rw = (2.4 + 3.4 * np.clip(noise(u * 30., e * 30., seed + 13), 0, 1)) * edge; rim_ = np.clip(rw - d + .5, 0, 1)[..., None]; body = body + (C(pal['white']) - body) * rim_; ins = np.clip(rw + 2.2 - d + .5, 0, 1)[..., None] - rim_; body = body + (C(pal['shade']) - body) * np.clip(ins, 0, 1) * .8       # rolled lip, then its own thin shade
    out = img.copy(); R_ = out[ymin:ymax]; sh = np.clip(d + 3.2, 0, 1)[..., None] * (1 - a) * .55 * ((np.abs(u) < .93) & ok)[..., None]; R_[:] = R_ * (1 - sh) + C(pal['ink']) * sh; R_[:] = R_ * (1 - a) + body * a
    cel = Cel(W_, h); r = R(seed, 61)
    for i in range(beads):                                    # beads run ahead of the front and hang in the holes
        uu = r.uniform(-1, 1) * .98; ahead = r.random() < .62; ss_ = T * r.uniform(.96, 1.22) + .01 if ahead else r.uniform(max(e0, .16), max(min(T, life), .17)); kk = r.uniform(.9, 1.12); zr = r.random() ** 2.6
        if ss_ > life * 1.15 or ss_ < .03: continue
        X = cx + uu * (w + vx * kk * ss_) + r.normal(0, 5); Y = top - vy * kk * (1.1 - .55 * abs(uu)) * ss_ + .5 * g * ss_ * ss_; v = np.array([uu * vx * kk, -vy * kk + g * ss_]); l = math.hypot(*v) + 1e-6; dv = v / l; rr = (2.0 + 9.5 * zr); st_ = min(3., 1 + l / 2600.); nv = np.array([-dv[1], dv[0]]); th_ = np.linspace(0, TAU, 16, endpoint=False); pt = lambda kx, off: np.array([X, Y]) + np.outer(np.sin(th_) * rr * kx, nv) - np.outer(np.where(np.cos(th_) > 0, np.cos(th_) * st_, np.cos(th_)) * rr * kx, dv) + off
        if not (-30 < X < W_ + 30 and -30 < Y < h + 30): continue
        if rr > 3.4: cel.fill(pt(1.16, [1., 1.5]), pal['ink'], .55); cel.fill(pt(1., 0), pal['foam']); cel.fill(pt(.62, [rr * .22, rr * .25]), pal['shade'], .8); cel.fill(pt(.40, [-rr * .22, -rr * .25]), pal['white'])
        else: cel.fill(pt(1., 0), pal['white'])
    L = cel.layer(h, W_); return (out * (1 - L[..., 3:4]) + L[..., :3]).astype(f32)


def spray_crown(cel, x, y, t, size=1., pal='lake', seed=0, n=22, tilt=.34):
    """screen-space crown of torn water thrown up from (x, y) px (use instead of a star of needles): curved fingers, shade side, beads; t seconds, size 1 = about 260 px high"""
    cam = stage(hy=y - 520 * tilt, gx=x, gy=y, ppm=150 * size, focal=1500); return splash(cel, cam, t, (0, 0), 5.6, .55, n, .11, .14, 1.05, pal, seed, .34, 0., 0)
