"""phys rain: world-space rain for any camera (normal, straight down, stopping, rising, blown away in a cone), lens drops that refract the real picture, hits on rock, run-off, drips.  (PHYS lane)"""
from .phys_core import *

RAINPAL = dict(grey=dict(col='#d7dbdd', near='#f0f3f4', dark='#2c3034', bead='#e8ecee'), gold=dict(col='#ffe6a0', near='#fff7c8', dark='#2a2012', bead='#fff3c4'), violet=dict(col='#c9b2ff', near='#fbf7ff', dark='#150a30', bead='#e6d6ff'), night=dict(col='#a7b0d8', near='#e6ebff', dark='#0e1026', bead='#dfe6ff'))


class Rain:
    """Drops live in a box of air round the camera (metres) and fall with wind; the same drops can slow, hang, rise or be blown away.
    rn = Rain(n=520, seed=1, wind=(1.2, 0)); layer = rn.draw(cam, tau, v=1)     tau = rain clock in seconds, v = d tau / dt now (stroke length follows |v|)
    tau / v may be functions of u (0..1 per-drop random array) -> arrays, so drops can stop or rise one by one.
    warp(P (n,3), u) -> (P2, V2 m/s or None) moves drops (cone, aura).  mask = (h,w) 0..1 where rain is seen."""

    def __init__(self, n=520, box=(16., 26., 12.), seed=0, wind=(1.0, 0.), speed=9., near=1.1, shutter=.052, size=(.8, 1.35), far_bias=.75):
        r = R(seed, 61); self.n = n; self.box = np.asarray(box, float); self.U = r.random((n, 3)); self.U[:, 1] = self.U[:, 1] ** far_bias; self.u = r.random(n); self.k = r.random((n, 3)); self.sp = speed * r.uniform(.88, 1.12, n); self.w = np.asarray(wind, float); self.near, self.sh, self.size, self.seed = near, shutter, size, seed

    def world(self, cam, tau):
        """positions now (n,3), velocity (n,3), wrap count"""
        B = self.box; t = tau(self.u) if callable(tau) else np.full(self.n, float(tau)); fw, rt, up = cam.R[2], cam.R[0], -cam.R[1]; V = np.c_[np.full(self.n, self.w[0]), np.full(self.n, self.w[1]), -np.ones(self.n)] * np.c_[self.sp / 9., self.sp / 9., self.sp]
        P = (self.U - .5) * np.array([B[0], B[1], B[2]]) + np.array([0, 0, 0]); c = cam.eye + fw * (B[1] * .5 + self.near); lo = c - B / 2; P = P + c; P = P + V * t[:, None]; cyc = np.floor((P - lo) / B); P = lo + np.mod(P - lo, B); return P, V, cyc

    def draw(self, cam, tau, v=1., pal='grey', alpha=1., warp=None, mask=None, dof=True, len_k=1., width=.0022, bg=None, bead_r=.0052, near_soft=7., hero=26., h=H, w=W, flare_dir=(-.4, -.8)):
        pal = RAINPAL[pal] if isinstance(pal, str) else pal; P, V, cyc = self.world(cam, tau); vv = v(self.u) if callable(v) else np.full(self.n, float(v)); Vn = V * vv[:, None]
        if warp is not None:
            P, V2 = warp(P, self.u)
            if V2 is not None: Vn = np.where(np.isnan(V2), Vn, V2)
        q, z = cam.project(P); ok = (z > .25); tail = P - Vn * self.sh * len_k * (.45 + 1.5 * self.k[:, 0:1] ** 2.2); q2, z2 = cam.project(tail); ok &= z2 > .12; pxm = cam.f / np.maximum(z, .1); cel = Cel(w, h); far = np.zeros((h, w), np.uint8); soft = []; heroes = []
        order = np.argsort(-z)
        for i in order:
            if not ok[i]: continue
            x, y = q[i]; d = q2[i] - q[i]; l = math.hypot(*d); wd = width * pxm[i] * (.7 + .6 * self.k[i, 1]); 
            if not (-300 < x < w + 300 and -300 - l < y < h + 300 + l): continue
            fade = clamp(1.35 - z[i] / (self.box[1] * 1.05), .12, 1.); near = z[i] < 2.2
            rb = bead_r * pxm[i] * (.55 + .9 * self.k[i, 2] ** 1.6)
            if l > rb * 2.2 and l > 3:                         # a stroke
                if wd < 1.15: cv2.line(far, (int(q2[i, 0] * 16), int(q2[i, 1] * 16)), (int(x * 16), int(y * 16)), int(255 * alpha * fade * (.30 + .35 * self.k[i, 0]) * min(1., wd / .8)), 1, cv2.LINE_AA, 4); continue
                n = np.array([-d[1], d[0]]) / l * wd * .5; a = q[i]; pts = np.array([q2[i], a + d * .30 + n, a + n * .7, a - d / l * wd * .8, a - n * .7, a + d * .30 - n])
                if near and dof: soft.append((pts, .34 + .3 * self.k[i, 0])); continue
                cel.fill(pts, pal['col'], alpha * fade * (.42 + .4 * self.k[i, 0]))
                if wd > 2.2: cel.fill(np.array([a + d * .72, a + d * .2 + n * .5, a - d / l * wd * .4, a + d * .12]), pal['near'], alpha * .9)
            else:
                st = clamp(1 + l / (rb * 2 + 1e-6), 1, 2.6); ang = math.atan2(d[1], d[0]) if l > .8 else -math.pi / 2
                if rb >= hero and bg is not None: heroes.append((i, x, y, rb, st, ang))
                else: bead(cel, x, y, rb, pal, st, ang, alpha * max(.45, fade))
        lay = cel.layer(h, w); a = far.astype(f32)[..., None] / 255.; A = np.dstack([a * C(pal['col']), a]); lay = A * (1 - lay[..., 3:4]) + lay
        if soft:
            c2 = Cel(w // 2, h // 2, ss=1)
            for p, al in soft: c2.fill(p / 2, pal['near'], al * alpha)
            s = cv2.GaussianBlur(c2.layer(h // 2, w // 2), (0, 0), near_soft * .5); s = cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR); lay = lay * (1 - s[..., 3:4]) + s
        for i, x, y, rb, st, ang in heroes: lay = lens_drop(lay, bg, x, y, rb, st, ang, self.seed * 31 + i)
        if mask is not None: lay = lay * mask[..., None]
        return lay.astype(f32)


def bead(cel, x, y, r, pal, st=1., ang=-math.pi / 2, al=1.):
    """a small hanging drop: dark crown, light belly, one hard glint"""
    if r < .5: return
    if r < 1.7: a = np.linspace(0, TAU, 8, endpoint=False); cel.fill(np.c_[x + np.cos(a) * r, y + np.sin(a) * r * 1.15], pal['bead'], .75 * al); return
    a = np.linspace(0, TAU, 22, endpoint=False); ca, sa = np.cos(a), np.sin(a); d = np.array([math.cos(ang), math.sin(ang)]); n = np.array([-d[1], d[0]]); E = lambda k, ox=0., oy=0.: np.array([x, y]) + np.outer(sa * r * k + oy * r, n) + np.outer(np.where(ca > 0, ca * st * (1 + .12 * ca), ca) * r * k, d) + np.array([ox * r, 0]); O = E(1.)
    cel.fill(O, pal['dark'], .72 * al); cel.clip(O); cel.fill(E(1.05) + [0, r * .62], pal['col'], .85 * al); cel.fill(E(.86) + [0, r * .20], pal['dark'], .38 * al); cel.unclip()
    cel.fill(np.c_[x - r * .34 + ca * r * .24, y - r * .36 + sa * r * .17], '#ffffff', al)
    if r > 5: th = np.linspace(.35, 2.2, 10); cel.fill(np.vstack([np.c_[x + np.cos(th) * r * .80, y + np.sin(th) * r * .80], np.c_[x + np.cos(th) * r * .68, y + np.sin(th) * r * .64][::-1]]), '#ffffff', .9 * al); cel.line(np.vstack([O, O[:1]]), max(.8, r * .05), pal['dark'], taper=(.3, .3), press=.5, tremor=0, alpha=.8 * al)


def lens_drop(layer, bg, x, y, r, st=1., ang=-math.pi / 2, seed=0, ink='#15181c', sky='#f4f7f8', key=(-.42, -.50), wob=.02, tint=None, warm=None, line=1., view=.86, cel_=5, blur_=0., glint=1.):
    """a big drop as a lens: the WHOLE picture behind it hangs upside down inside (wide angle, squeezed toward the rim), drawn in flat tones; thick-thin ink, dark crown, bright foot, hard window glint.
    layer: premultiplied RGBA to draw into; bg: RGB picture behind (keep it sharp).  view = how much of the picture it holds, cel_ = tone steps (0 off), blur_ = out-of-focus px, warm = colour of the foot light"""
    h, w = layer.shape[:2]; pad = int(r * max(st, 1.15) * 1.22 + 6 + blur_ * 3); x0, y0, x1, y1 = max(0, int(x - pad)), max(0, int(y - pad)), min(w, int(x + pad) + 1), min(h, int(y + pad) + 1)
    if x1 - x0 < 4 or y1 - y0 < 4: return layer
    ss = 2 if r < 130 else 1; yy, xx = np.mgrid[0:(y1 - y0) * ss, 0:(x1 - x0) * ss].astype(f32); xx = (xx + .5) / ss + x0; yy = (yy + .5) / ss + y0; d = np.array([math.cos(ang), math.sin(ang)]); n = np.array([-d[1], d[0]]); u = ((xx - x) * d[0] + (yy - y) * d[1]) / r; v = ((xx - x) * n[0] + (yy - y) * n[1]) / r; u = np.where(u > 0, u / st, u * 1.03); v = v * (1 + .16 * (st - 1) * np.clip(u, 0, 1)); th = np.arctan2(v, u); rr = 1 + wob * (np.sin(th * 2 + seed) * .7 + np.sin(th * 3 - seed * 1.7) * .5 + np.sin(th * 5 + seed * .6) * .18); rad = np.sqrt(u * u + v * v) / rr; px = 1. / (r * ss); aa = lambda q, k=1.: np.clip(q / (px * k * 1.1) + .5, 0, 1)
    ux = (xx - x) / r / rr; uy = (yy - y) / r / rr; s = np.clip(rad, 0, 1); g = .60 + .55 * s ** 2.2; sx = x * .55 + w * .5 * .45 - ux * g * w * .5 * view; sy = y * .40 + h * .5 * .60 - uy * g * h * .62 * view; src = np.clip(bg, 0, 1.2).astype(f32); k_ = max(1, int(w * view / (2 * r)) // 2)
    if k_ > 1: src = cv2.blur(src, (k_, k_))
    col = cv2.remap(src, np.clip(sx, 1, w - 2).astype(f32), np.clip(sy, 1, h - 2).astype(f32), cv2.INTER_LINEAR); col = np.clip((col - .40) * 1.30 + .46, 0, 1.05)
    if cel_: L = lum(col)[..., None]; Lq = (np.floor(L * cel_) + smooth(.42, .58, (L * cel_) % 1)) / cel_ + .5 / cel_ * .0; col = col * np.clip((Lq + .06) / (L + .06), .55, 1.6)
    if tint is not None: col = col * (.75 + .5 * C(tint))
    def put_(m, c, o=1.):
        nonlocal col; m = m[..., None] * o; col = col * (1 - m) + (C(c) if isinstance(c, str) else c) * m
    top = np.clip(-uy + .10 * ux, -1, 1); put_(aa(rad - (.97 - .20 * np.clip(top, 0, 1) ** 1.5)), ink, .86)                          # dark crown, crescent
    foot = np.clip(uy * .95 + ux * .30, -1, 1); wdt = .155 * np.clip((foot - .30) / .7, 0, 1) ** .8; put_(aa(.955 - rad) * aa(rad - (.955 - wdt)) * (wdt > .004), sky if warm is None else warm, .95)
    put_(aa(.80 - rad) * aa(rad - (.80 - .035 * np.clip((foot - .55) / .45, 0, 1))) * (foot > .56), sky if warm is None else warm, .55)
    th2 = np.arctan2(uy, ux); kth = math.atan2(key[1], key[0]); da = np.angle(np.exp(1j * (th2 - kth))); r0 = math.hypot(*key); win = aa(.085 - np.abs(rad - r0 - .06), 1.) * aa((.50 - .9 * np.abs(rad - r0 - .06)) - np.abs(da), 2.); bar = aa(np.abs(da + .12) - .035, 2.); put_(win * bar * glint, '#ffffff'); put_(aa(.042 - np.hypot(ux - .40 * math.cos(kth + 2.2), uy - .40 * math.sin(kth + 2.2)), 1.) * glint, '#ffffff', .8)
    wl = min(.16, (1.5 * line / r) * 1.0) * (1.0 + 1.1 * np.clip(uy * .7 + ux * .6, -.55, 1)); put_(aa(rad - (1 - wl)), ink); a = aa(1 - rad); o = np.dstack([col * a[..., None], a[..., None]]).astype(f32)
    if blur_ > 0: o = cv2.filter2D(o, -1, disc_(blur_ * ss))
    if ss > 1: o = cv2.resize(o, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    L_ = layer[y0:y1, x0:x1]; layer[y0:y1, x0:x1] = L_ * (1 - o[..., 3:4]) + o; return layer


def disc_(r):
    n = int(math.ceil(r)) * 2 + 1; y, x = np.mgrid[:n, :n] - n // 2; k = np.clip(r + .5 - np.sqrt(x * x + y * y), 0, 1); return (k / k.sum()).astype(f32)


# ------------------------------------------------------------- clocks for rain that stops and rises
def stop_clock(f, f_stop=0, f_rise=20, spread=14, f0=None, rise=.35, accel=1.2, ease=2):
    """per-drop rain clock for f1: full speed, hard stop over `ease` frames at f_stop, then every drop starts up at its own frame f_rise + spread * u.
    -> (tau(u), v(u)) for Rain.draw"""
    def tau(u):
        t = np.full(u.shape, min(f, f_stop) / 24.); s = max(0., f - f_stop) / 24.; e = ease / 24.; t = t + (e * .5 * (1 - (1 - min(1., s / e)) ** 2) if e > 0 else 0.); r = np.maximum(0., (f - (f_rise + spread * u)) / 24.); return t - rise * .5 * accel * r * r * (1 / (.25 + r * 0.0 + .75)) * 1.
    def v(u):
        s = (f - f_stop) / max(ease, 1e-3); v0 = np.full(u.shape, 1. if s <= 0 else max(0., 1 - s)); r = np.maximum(0., (f - (f_rise + spread * u)) / 24.); return np.where(r > 0, -np.minimum(rise * accel * r, 1.2), v0)
    return tau, v


def cone_blast(cel, cam, apex, dirn, half, t, c=46., col='#f0f3f4', shade='#85898c', seed=0, rings=4, life=.9, streaks=46, reach=40.):
    """what the air does when the rain is blown away: shock rings with thickness running down the cone, radial streaks, all in perspective"""
    A = np.asarray(apex, float); D = unit(dirn); e1 = unit(np.cross(D, [0, 0, 1.])); e2 = np.cross(e1, D); a = np.linspace(0, TAU, 72); ca, sa = np.cos(a), np.sin(a)
    for k in range(rings):
        d = c * (t - .09 * k); q = t - .09 * k
        if d <= .6 or d > reach: continue
        R_ = d * math.tan(half) * (1 - .10 * k); w = R_ * (.15 - .02 * k) * (1 - d / reach) ** .5; wob = 1 + .012 * np.sin(a * 5 + seed + k) + .008 * np.sin(a * 11 - k); m = np.clip(.5 + .9 * noise(ca * 1.6 + k * 7, sa * 1.6, seed * 1.1) - .25 * q / life, 0, 1) ** 1.3; pr = lambda rr, dd: cam.project(A + D * np.broadcast_to(dd, a.shape)[:, None] + e1 * (ca * rr)[:, None] + e2 * (sa * rr)[:, None]); (o, zo), (i, zi) = pr(R_ * wob, d), pr(R_ * wob - w * m, d - w * 1.8 * m)
        if min(zo.min(), zi.min()) < .4: continue
        al = 1 - d / reach; cel.fill([o, i], shade, .35 * al, evenodd=True); mm = o + (i - o) * .55; cel.fill([o, mm], col, .90 * al, evenodd=True)
    for j in range(streaks):
        th = h1(seed, j, 5) * TAU; r0 = math.sqrt(h1(seed, j, 6)); hd = c * t * (.55 + .45 * h1(seed, j, 7)); tl = max(.5, hd - c * .10 * (1 + h1(seed, j, 8)))
        if hd > reach or hd < .8: continue
        dv = D + (e1 * math.cos(th) + e2 * math.sin(th)) * math.tan(half) * r0; q, z = cam.project(np.array([A + dv * hd, A + dv * tl]))
        if z.min() > .4: wedge(cel, q[0], q[1], max(.8, .03 * cam.f / z[0]), .2, col, .7 * (1 - hd / reach))
    return cel


def cone_warp(apex, dirn, half=.5, t=0., c=60., gain=26., soft=.25, spread=2.2):
    """rain blown away: a shock front leaves `apex` along dirn inside a cone at c m/s; every drop it passes is thrown outward and leaves the cone clean.  -> warp for Rain.draw"""
    A = np.asarray(apex, float); D = unit(dirn)
    def warp(P, u):
        r = P - A; a = r @ D; rad = r - a[:, None] * D; rl = np.linalg.norm(rad, axis=1) + 1e-6; dist = np.linalg.norm(r, axis=1); ang = np.arctan2(rl, np.maximum(a, 1e-3)); inside = (a > 0) & (ang < half * (1 + soft * (u - .5))); s = np.maximum(0., t - dist / c); on = inside & (s > 0); o = unit_rows(r * .8 + rad * spread * (1 + 0 * rl[:, None])); k = gain * s * np.exp(-s * .6) * (.7 + .7 * u); Q = P + np.where(on[:, None], o * (k * (1 + 3 * s))[:, None], 0.); V = np.where(on[:, None], o * (gain * (1 + 3 * np.minimum(s, .4)) * 2.0)[:, None], np.nan); return Q, V
    return warp


def unit_rows(a): return a / (np.linalg.norm(a, axis=1)[:, None] + 1e-9)


# ------------------------------------------------------------- rain meets things
def hit(cel, x, y, s, k, seed=0, pal='grey', asp=.36, al=1.):
    """one raindrop bursting on a surface at screen (x, y): k frames 0..6, s px.  asp = foreshortening of the surface (1 = facing us)"""
    if k < 0 or k > 6: return cel
    pal = RAINPAL[pal] if isinstance(pal, str) else pal; e = 1 - (1 - k / 6.5) ** 2; rx = s * (.18 + .95 * e); w = s * .13 * (1 - k / 7.)
    for g in range(2):
        a0 = h1(seed, g) * TAU + g * 2.9; a = np.linspace(a0, a0 + 1.5 + h1(seed, g, 2), 12); hw = w * np.sin(np.linspace(0, np.pi, 12)); cel.fill(np.vstack([np.c_[x + np.cos(a) * (rx + hw), y + np.sin(a) * (rx + hw) * asp], np.c_[x + np.cos(a) * (rx - hw), y + np.sin(a) * (rx - hw) * asp][::-1]]), pal['col'], .85 * al)
    n = 5 + int(h1(seed, 3) * 4)
    for i in range(n):                                       # fingers: needles that leave, arc, and become beads
        a = (i + h1(seed, i, 4)) / n * TAU; sp = s * (.22 + .30 * h1(seed, i, 5)); up = s * (.34 + .34 * h1(seed, i, 6)) * (1 - asp * .6); t = k + .6; px = x + math.cos(a) * sp * t; py = y + math.sin(a) * sp * t * asp - up * t + .075 * s * t * t * (1 - asp * .5) * 1.4
        if k > 0 and py > y + math.sin(a) * sp * t * asp + s * .06: continue
        vx = math.cos(a) * sp; vy = math.sin(a) * sp * asp - up + .15 * s * t * (1 - asp * .5) * 1.4; l = 1.5 if k < 3 else .55; wd = s * .085 * (1.15 - k / 8.); d = np.array([vx, vy]); ln = math.hypot(*d) + 1e-6; m = np.array([-d[1], d[0]]) / ln * wd * .5; p = np.array([px, py]); cel.fill(np.array([p - d * l, p + m, p + d / ln * wd, p - m]), pal['near'], .95 * al)
    if k == 0: a = np.linspace(0, TAU, 10); cel.fill(np.c_[x + np.cos(a) * s * .2, y + np.sin(a) * s * .2 * asp], '#ffffff', al)
    return cel


def hits(cel, f, place, n=60, seed=0, size=(6, 22), rate=1., pal='grey', asp=.36, step=1, stop=None, al=1.):
    """field of rain bursts.  place(u, w) -> (x, y, scale 0..1) or None; stop = frame after which nothing new lands"""
    for i in range(n):
        per = (16 + 14 * h1(seed, i, 1)) / rate; q = f / per + h1(seed, i, 2); cyc = math.floor(q); born = (cyc - h1(seed, i, 2)) * per; k = int((f - born) // step * step)
        if stop is not None and born > stop: continue
        p = place(h1(seed, i, cyc, 3), h1(seed, i, cyc, 4))
        if p is not None: hit(cel, p[0], p[1], size[0] + (size[1] - size[0]) * p[2], k, seed * 3 + i + cyc * 7, pal, asp, al)
    return cel


def runoff(cel, f, paths, seed=0, pal='grey', n=3, speed=46., w=2.6, trail=90., al=1.):
    """rivulets on skin, rock or cloth: beads that stick and slip down the given screen polylines, each pulling a thin bright tail"""
    pal = RAINPAL[pal] if isinstance(pal, str) else pal
    for j, P in enumerate(paths):
        P = resample(np.asarray(P, float), 2.); d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; L = d[-1]
        for i in range(n):
            per = L / speed * 24 * (1.1 + .6 * h1(seed, j, i, 1)) + 14; t = (f / per + h1(seed, j, i, 2)); cyc = math.floor(t); u = t - cyc; k = 5 + int(3 * h1(seed, j, i, 3)); st = math.floor(u * k) / k; fr = (u * k) % 1; uu = st + (max(0., fr - .55) / .45) ** 2 / k; s = uu * L * 1.02
            if s > L: continue
            c = np.array([np.interp(s, d, P[:, 0]), np.interp(s, d, P[:, 1])]); s0 = max(0., s - trail * (.5 + h1(seed, j, i, cyc, 4))); m = np.linspace(s0, s, 14); T = np.c_[np.interp(m, d, P[:, 0]), np.interp(m, d, P[:, 1])]; thin(cel, T, w * .62, pal['col'], .75 * al, .9, .02, 16); bead(cel, c[0], c[1], w * 1.05, pal, 1.35, -math.pi / 2, al)
    return cel


def drip(cel, f, x, y, period=26, seed=0, r=5., pal='grey', g=1500., ground=1200., phase=0., al=1.):
    """water running off a fingertip: the drop swells, necks, lets go, falls with gravity (px), -> (cel, landed_y or None)"""
    pal = RAINPAL[pal] if isinstance(pal, str) else pal; t = (f + phase * period) % period; grow_ = min(1., t / (period * .62)); rr = r * (.25 + .75 * grow_ ** .7); hang = y + rr * (.6 + 1.1 * grow_ ** 3); tl = t - period * .62
    if tl < 0:
        neck = r * .55 * (1 - grow_ ** 4 * .75); a = np.linspace(.0, math.pi, 12); B = np.c_[x + np.cos(a) * rr, hang + np.sin(a) * rr * 1.1]; Q = np.vstack([[x + neck, y - 1], [x + rr * .92, hang - rr * .3], B, [x - rr * .92, hang - rr * .3], [x - neck, y - 1]]); cel.fill(spl(Q, True, n=3), pal['dark'], .6 * al); bead(cel, x, hang, rr * .94, pal, 1.15, -math.pi / 2, al); return cel
    s = tl / 24.; yy = hang + .5 * g * s * s; v = g * s
    if yy < ground: l = min(60., v * .018); bead(cel, x, yy, r * .85, pal, 1 + l / (r * 3), -math.pi / 2, al); thin(cel, np.array([[x, yy - l], [x, yy]]), r * .5, pal['col'], .6 * al, .9, .05, 8)
    bead(cel, x, y + r * .25, r * .35, pal, 1, -math.pi / 2, al); return cel
