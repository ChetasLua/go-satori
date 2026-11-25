"""ki hit: pressure rings, shock cones, dust, teleport strokes, impact frames, ink-wash after-images, the water-ring in space."""
from .ki_core import *
from .comp import blur
from .kit import impact as _tone


def ellipse_pts(cx, cy, rx, ry, rot=0., n=96, a0=0., a1=TAU, wob=0., seed=0, z=0.):
    a = np.linspace(a0, a1, n); k = 1 + wob * (noise(np.cos(a) * 1.8, np.sin(a) * 1.8, z, seed) + .4 * noise(np.cos(a) * 6, np.sin(a) * 6, z, seed + 1)) if wob else 1.; x, y = np.cos(a) * rx * k, np.sin(a) * ry * k; c, s = math.cos(rot), math.sin(rot); return np.stack([cx + x * c - y * s, cy + x * s + y * c], -1), a


def circle3(cam, P0, axis, rad, n=96, wob=0., seed=0, z=0.):
    """projected 3-D circle -> (px (n,2), depth (n,), angle)"""
    ax = np.asarray(axis, float); ax = ax / np.linalg.norm(ax); u = np.cross(ax, [0, 0, 1.]) if abs(ax[2]) < .9 else np.cross(ax, [0, 1., 0]); u /= np.linalg.norm(u); w = np.cross(ax, u); a = np.linspace(0, TAU, n); k = rad * (1 + wob * noise(np.cos(a) * 1.8, np.sin(a) * 1.8, z, seed)); X = np.asarray(P0, float) + (u[None] * np.cos(a)[:, None] + w[None] * np.sin(a)[:, None]) * np.broadcast_to(k, a.shape)[:, None]; p, d = cam.project(X); return p, d, a


def ring_strokes(Pout, Pin, kind='base', seed=0, gaps=.30, lw=1.4, cols=None, spurs=10, alpha=1.):
    """a pressure ring with real thickness between two closed polylines (outer lip, inner lip): pale crest, shadowed inside, torn gaps, dashes shed outward -> layer"""
    n = len(Pout); a = np.linspace(0, TAU, n); cut = noise(np.cos(a) * 1.7, np.sin(a) * 1.7, 2., seed) * 1.1 + .55 - gaps; th = np.clip(cut * 2.2, 0, 1); mid = (Pout + Pin) / 2; hw = np.hypot(*(Pout - Pin).T) / 2 * th - .3 * (th < .05); P_ = pal(kind); cc = cols or (P_['c'][0], P_['c'][1], P_['c'][3]); F = Field(3); nn = (Pout - mid) / (np.hypot(*(Pout - mid).T)[:, None] + 1e-6)
    F.rib(mid, [hw, hw * .74, hw * .36 - .2], 2.); F.D[1] = F.D[1]
    for j in range(spurs):
        i = int(h1(seed, j, 4) * (n - 1)); l = hw[i] * (1.5 + 5 * h1(seed, j, 5))
        if hw[i] < .8: continue
        p = Pout[i]; q = p + nn[i] * l; F.rib(np.linspace(p, q, 5), [np.array([.3, hw[i] * .30, hw[i] * .2, hw[i] * .08, -.2]) * z for z in (1, .6, .3)], 0.)
    o = F.cel(cc, lw=(lw, 0, 0), ink=(P_['ink'], None, None), seed=seed); return o * alpha


def ring(cx, cy, rad, thick, f=0, kind='base', asp=1., rot=0., seed=0, gaps=.3, alpha=1., wob=.03, **k):
    """flat pressure ring seen as an ellipse"""
    A, _ = ellipse_pts(cx, cy, rad + thick, (rad + thick) * asp, rot, 120, wob=wob, seed=seed); B, _ = ellipse_pts(cx, cy, rad - thick * .4, (rad - thick * .4) * asp, rot, 120, wob=wob, seed=seed); return ring_strokes(A, B, kind, seed, gaps, alpha=alpha, **k)


def ring3(cam, P0, axis, rad, thick, kind='base', seed=0, gaps=.3, alpha=1., wob=.03, depth=0., **k):
    """pressure ring in the world.  depth > 0 gives it a lip along the axis"""
    ax = np.asarray(axis, float) / np.linalg.norm(axis); A, da, _ = circle3(cam, np.asarray(P0, float) + ax * depth, ax, rad + thick, 120, wob, seed); B, db, _ = circle3(cam, P0, ax, rad - thick * .4, 120, wob, seed)
    if min(da.min(), db.min()) < .2: return None
    return ring_strokes(A, B, kind, seed, gaps, alpha=alpha, **k)


def refract_ring(img, cx, cy, rad, width, amp, asp=1., rot=0.):
    """the air inside a ring bends what is behind it"""
    x, y = grid(); c, s = math.cos(rot), math.sin(rot); u, v = (x - cx) * c + (y - cy) * s, (-(x - cx) * s + (y - cy) * c) / max(asp, .05); d = np.sqrt(u * u + v * v) + 1e-3; e = (d - rad) / width; k = (np.exp(-e * e) * np.sin(e * 2.4) * amp).astype(f32); du, dv = u / d * k, v / d * k * asp; return displace(img, du * c - dv * s, du * s + dv * c)


def cone(tip, dirn, length, half, f, kind='base', seed=1, rings=4, alpha=.9, streak=1., bend=.62):
    """shock cone (vapour) from a tip, opening against the travel direction dirn: translucent shell that fades to a torn skirt, pen edges, streaks, ribs -> layer"""
    t = f / 24.; P_ = pal(kind); d = np.array([math.cos(dirn), math.sin(dirn)]); nrm = np.array([-d[1], d[0]]); tip = np.asarray(tip, float); hf = math.tan(half); F, G, Sh = Field(2), Field(2), Field(2); pf = lambda v: np.maximum(v, 0) ** bend
    v = lin(40); e = np.vstack([tip - d * (length * v)[:, None] + nrm * (length * hf * pf(v) * 1.02)[:, None], (tip - d * (length * v)[:, None] - nrm * (length * hf * pf(v) * 1.02)[:, None])[::-1]]); cel = Cel(ss=1); cel.fill(e, '#ffffff', 1.); m = cel.rgba()[..., 3]; x, y = grid(); u = np.clip(((tip[0] - x) * d[0] + (tip[1] - y) * d[1]) / length, 0, 1); lat = np.abs((x - tip[0]) * nrm[0] + (y - tip[1]) * nrm[1]) / (length * hf * np.maximum(u, .02) ** bend); nz = nimg(H, W, 34, t * 3, seed, 2, 4, 1, 1, ox=d[0] * 700 * t, oy=d[1] * 700 * t); ed = np.clip((lat - .80 + .10 * nz) * 9, 0, 1); ms = m * np.clip((1.02 - u + .22 * nz) * 5, 0, 1); base = ms * (.10 + .34 * ed)
    for sg in (1, -1):
        P = tip - d[None] * (length * v)[:, None] + nrm[None] * (sg * length * hf * pf(v))[:, None]; hw = length * .011 * np.sin(math.pi * np.clip(v * .9 + .06, 0, 1)) ** .7 * np.clip(.8 + .9 * noise(v * 7 - t * 16, sg, 0, seed), 0, 2) - .2; F.rib(P, [hw, hw * .5 - .2], 1.)
        for j in range(9):
            q = (j + h1(seed, j, sg)) / 9; q = .30 + .70 * ((q + t * 1.8) % 1.); p = tip - d * length * q + nrm * sg * length * hf * pf(np.array(q)); l = length * (.08 + .22 * h1(seed, j, 3)) * q; an = math.atan2(*(-d + nrm * sg * .22)[::-1]); F.rib(spine(p[0], p[1], l, an, 9, .05, .6, j, sg * .35), lwidths(9, l, l * .05, LV[:2], .3), 2.)
    for i in range(int(40 * streak)):
        a = (h1(seed, i, 7) * 2 - 1); x0 = (h1(seed, i, 8) + t * 2.2) % 1.; l = .12 + .25 * h1(seed, i, 9); q = np.clip(x0 + l * lin(10), 0.02, 1.1); P = tip - d[None] * (length * q)[:, None] + nrm[None] * (a * length * hf * pf(q) * .97)[:, None]; w = length * .0055 * (.3 + h1(seed, i, 10)) * np.sin(math.pi * lin(10)); G.rib(P, [w, w * .45 - .2], 0.)
    for i in range(rings):
        q = ((i + .5) / rings + t * 1.2) % 1.; c = tip - d * length * q; r = length * hf * float(pf(np.array(q))); A, th = ellipse_pts(c[0], c[1], r * .16, r, dirn, 90, wob=.02, seed=seed + i); w = (length * .004 + r * .020) * np.clip(.35 + 1.0 * noise(np.cos(th) * 1.6, np.sin(th) * 1.6, i, seed), 0, 1.4) * (1 - q * .4) * np.where(np.cos(th) < 0, .5, 1.) - .25; Sh.rib(A, [w, w * .4 - .2], 0.)
    c = P_['c']; return stack(np.dstack([base[..., None] * C(c[1]), base]), G.cel((c[1], c[3])) * .8, Sh.cel((c[1], c[3])), F.cel((c[1], c[3]))) * alpha


# ------------------------------------------------------------------ teleport
def vanish(layer, k, ang=0., seed=1, stretch=2.2, kind='base', bands=15):
    """a figure leaves: k 0 = sliced and stretched, 1 = long slivers, 2 = hair dashes, 3 = two last specks; >3 nothing.  ang = direction of leaving (0 right, pi/2 down)"""
    if k > 3 or k < 0: return None
    a = layer[..., 3]; ys, xs = np.nonzero(a > .3)
    if len(xs) < 20: return None
    P_ = pal(kind); c, s = math.cos(ang), math.sin(ang); u = xs * c + ys * s; v = -xs * s + ys * c; v0, v1 = v.min(), v.max(); out = np.zeros_like(layer); n = bands; F = Field(3); x, y = grid(); V = -x * s + y * c
    for i in range(n):
        w0 = v0 + (v1 - v0) * (i + .5 * (h1(seed, i, 1) - .5)) / n; w1 = v0 + (v1 - v0) * (i + 1 + .5 * (h1(seed, i + 1, 1) - .5)) / n; m = (v >= w0) & (v < w1)
        if not m.any(): continue
        ua, ub = u[m].min(), u[m].max(); z = bms(h1(seed, i, 2), .25, .35); ln = (ub - ua); mid = (w0 + w1) / 2; h = (w1 - w0) * .5
        if k == 0:
            sh = (h1(seed, i, 3) - .5) * ln * .9; S = 1 + .5 * z; q = ((V >= w0) & (V < w1)).astype(f32)[..., None]; uc = (ua + ub) / 2; M = np.float32([[c, -s], [s, c]]) @ np.diag([1 / S, 1.]).astype(f32) @ np.float32([[c, s], [-s, c]]); ctr = np.array([uc * c - mid * s, uc * s + mid * c]); off = ctr - M @ (ctr + np.array([c, s]) * sh); out = np.maximum(out, cv2.warpAffine(layer, np.float32(np.c_[M, off]), (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP) * q)
        if k >= 1 or i % 3 == 0:
            if k == 2 and h1(seed, i, 4) > .62: continue
            if k == 3 and h1(seed, i, 4) > .22: continue
            L = ln * stretch * (.5 + z) * (1.15 if k == 1 else .9 if k == 2 else .4); uc = (ua + ub) / 2 + (h1(seed, i, 5) - .5) * ln; th = h * (.40 if k <= 1 else .13 if k == 2 else .08) * (.5 + z); w_ = np.linspace(uc - L / 2, uc + L / 2, 16); P = np.stack([w_ * c - mid * s, w_ * s + mid * c], -1); e = lwidths(16, L, th, ((1., 1.), (.55, .8), (.2, .5)), .5, .6, .6); F.rib(P, e, 0.)
    st = F.cel((P_['c'][0], P_['c'][2], P_['c'][3])); return st if k else stack(out, st * .9)


# ------------------------------------------------------------------ impact frames (graphic, each different)
def _wedges(c, cx, cy, n, r0, r1, seed, w, col, asp=.75, clump=.6):
    r = R(seed, 6); ct = r.uniform(0, TAU, max(3, n // 6))
    for i in range(n):
        a = r.choice(ct) + r.normal(0, .16) if r.random() < clump else r.uniform(0, TAU); z = bms(r.random(), .12, .3); q = r0 * r.uniform(.75, 1.9) * (1.15 - .3 * z); hw = w * z * r.uniform(.6, 1.5) + .0015; c.fill(np.array([[cx + math.cos(a) * q, cy + math.sin(a) * q * asp], [cx + math.cos(a - hw) * r1, cy + math.sin(a - hw) * r1], [cx + math.cos(a + hw) * r1, cy + math.sin(a + hw) * r1]]), col)


def _shard(c, P0, ang, L, w, col, r):
    d = np.array([math.cos(ang), math.sin(ang)]); n = np.array([-d[1], d[0]]); k = r.uniform(.15, .5); c.fill(np.array([P0, P0 + d * L * k + n * w * r.uniform(.3, 1), P0 + d * L, P0 + d * L * k - n * w * r.uniform(.3, 1)]), col)


INK_, PAPER_ = '#07060c', '#fbf7ff'


def impact(base, n, cx=960, cy=540, subject=None, seed=0, accent='#ffb01e', dirn=0.):
    """impact frame number n (0..7), a drawing in two or three flat tones built on the picture: 0 paper, black focus wedges | 1 black, white kite star | 2 split field, inverted halves |
    3 concentric torn rings | 4 slabs of speed, accent | 5 cracked glass | 6 halftone burst | 7 negative with one accent bar.   subject = premult layer kept as a graphic silhouette"""
    r = R(seed, n, 17); im = np.empty((H, W, 3), f32); c = Cel(ss=2); k, p = C(INK_), C(PAPER_); a = subject[..., 3:4] if subject is not None else None; col = np.where(a > .02, subject[..., :3] / np.maximum(a, .02), 0).astype(f32) if subject is not None else base
    def sub(mode, lo=.22, hi=.6, acc=None):
        o = _tone(col, mode, lo, hi, INK_, PAPER_, acc or accent)
        if a is None: return o
        return o
    def place(bg, mode, edge=None, w=4):
        if a is None: return bg
        if edge is not None: e = cv2.dilate(a[..., 0], np.ones((3, 3), np.uint8), iterations=w)[..., None]; bg = bg * (1 - e) + C(edge) * e
        return bg * (1 - a) + sub(mode) * a
    if n == 0: im[:] = p; _wedges(c, cx, cy, 120, 300, 2300, seed + 1, .030, INK_); im = put(im, c.layer()); return place(im, 'bw', PAPER_, 5)
    if n == 1:
        im[:] = k; _wedges(c, cx, cy, 40, 420, 2300, seed + 2, .012, PAPER_); ax = dirn
        for i in range(34):
            an = r.uniform(0, TAU); z = bms(r.random(), .22, .34); L = (180 + 900 * z) * (1 + .9 * abs(math.cos(an - ax)) ** 3); _shard(c, np.array([cx, cy]), an, L, L * .045 * r.uniform(.5, 1.2), PAPER_, r)
        c.fill(spl([(cx + math.cos(q) * 90 * (1 + .25 * (i % 2)), cy + math.sin(q) * 90 * (1 + .25 * (i % 2))) for i, q in enumerate(np.linspace(0, TAU, 14)[:-1])], True, n=4), PAPER_); im = put(im, c.layer()); return place(im, 'inv', INK_, 4)
    if n == 2:
        x, y = grid(); s = ((x - cx) * -math.sin(dirn + .35) + (y - cy) * math.cos(dirn + .35)) + 26 * np.sign(np.sin(x * .021 + y * .013)) * (np.abs(np.sin(x * .0063 + 1)) ** 3); m = (s > 0).astype(f32)[..., None]; base2 = p * m + k * (1 - m); L = (_tone(base, 'bw', .2, .5)[..., :1] > .5).astype(f32); im = base2 * .0 + np.where(m > .5, p * L + k * (1 - L), k * L + p * (1 - L))
        for i in range(26): q = r.uniform(-900, 900); d = np.array([math.cos(dirn + .35), math.sin(dirn + .35)]); _shard(c, np.array([cx, cy]) + d * q + r.normal(0, 14, 2), dirn + .35 + r.normal(0, .06) + (math.pi if r.random() < .5 else 0), r.uniform(200, 900), r.uniform(3, 18), accent if i % 4 == 0 else (INK_ if i % 2 else PAPER_), r)
        im = put(im, c.layer())
        if a is None: return im
        S1, S2 = _tone(col, 'bw'), _tone(col, 'inv'); S = np.where(m > .5, S1, S2); return im * (1 - a) + S * a
    if n == 3:
        im[:] = k; L = put(im, None)
        for i in range(9):
            rr = 130 * 1.42 ** i; Po, _ = ellipse_pts(cx, cy, rr * 1.16, rr * 1.16 * .72, .12, 140, wob=.10, seed=seed + i); Pi, _ = ellipse_pts(cx, cy, rr * .99, rr * .99 * .72, .12, 140, wob=.07, seed=seed + i + 40); L = put(L, ring_strokes(Po, Pi, 'base', seed + i, .10 + .03 * i, 0, (PAPER_, PAPER_, PAPER_ if i % 3 else accent), 26))
        return place(L, 'inv', INK_, 6)
    if n == 4:
        im[:] = C(accent); d = np.array([math.cos(dirn), math.sin(dirn)]); m = np.array([-d[1], d[0]])
        for i in range(70): q = np.array([960, 540]) + m * r.uniform(-1000, 1000) + d * r.uniform(-900, 700); L = r.uniform(500, 2400); w = 60 * bms(r.random(), .12, .3) ** 1.6 + 2; c.fill(np.array([q - d * L / 2, q + m * w * .5, q + d * L / 2, q - m * w * .5]), INK_ if i % 3 else PAPER_)
        im = put(im, c.layer()); return place(im, 'bw', INK_, 5)
    if n == 5:
        im = _tone(base, 'inv', .2, .55); Q = [];
        for i in range(17):
            an = (i + r.random()) / 17 * TAU; P = [np.array([cx, cy], float)]; st = r.uniform(60, 130)
            while 0 < P[-1][0] < W and 0 < P[-1][1] < H and len(P) < 16: an += r.normal(0, .22); P.append(P[-1] + np.array([math.cos(an), math.sin(an)]) * st); st *= r.uniform(.9, 1.5)
            Q.append(np.array(P)); c.line(P, 9. * r.uniform(.5, 1.3), accent if i % 5 == 0 else INK_, taper=(.0, .6), press=.4, seed=i, tremor=.0)
        for i in range(len(Q)):
            A, B = Q[i], Q[(i + 1) % len(Q)]
            for j in range(1, min(len(A), len(B)) - 1):
                if r.random() < .5: c.line([A[j], (A[j] + B[j]) / 2 + r.normal(0, 12, 2), B[j]], 3.6, INK_, taper=(.1, .1), seed=j)
            if r.random() < .35 and len(A) > 4 and len(B) > 4: j = int(r.integers(1, 3)); c.fill(np.array([A[j], A[j + 1], B[j + 1], B[j]]), accent if r.random() < .3 else INK_, 1.)
        im = put(im, c.layer()); return place(im, 'bw', INK_, 4)
    if n == 6:
        from .kit import screentone
        x, y = grid(); d = np.sqrt((x - cx) ** 2 + ((y - cy) * 1.25) ** 2); im[:] = p; D = np.clip((d - 160) / 900, 0, .78); T = screentone(np.ones((H, W), f32), 13., .62, np.maximum(D, .001)); im = im * (1 - T[..., None]) + k * T[..., None]; _wedges(c, cx, cy, 60, 420, 2300, seed + 4, .05, INK_); _wedges(c, cx, cy, 30, 520, 2300, seed + 5, .02, PAPER_); im = put(im, c.layer()); return place(im, 'bw', PAPER_, 7)
    im = _tone(base, 'inv', .18, .5); d = np.array([math.cos(dirn), math.sin(dirn)]); m = np.array([-d[1], d[0]]); q = np.array([cx, cy], float); c.fill(np.array([q - d * 2200, q + m * 70, q + d * 2200, q - m * 70]), accent); c.fill(np.array([q - d * 2200, q + m * 22 + m * 9, q + d * 2200, q + m * 22 - m * 12]), PAPER_)
    for i in range(30): _shard(c, q + r.normal(0, 30, 2), r.uniform(0, TAU), r.uniform(80, 520), r.uniform(3, 16), INK_ if i % 2 else PAPER_, r)
    im = put(im, c.layer()); return place(im, 'inv', accent, 3)


# ------------------------------------------------------------------ ink-wash after-image
def inkwash(layer, age, seed=1, ink='light', drift=(0., 0.), life=16., strength=1.):
    """what stays of a figure for a moment: brushed wash with a pooled edge, granulation, dry holes, carried by a drift, thinning with age (frames).
    ink 'light' for dark space, 'dark' for paper, or a hex -> premult layer"""
    u = age / float(life)
    if u >= 1 or age < 0: return None
    col = {'light': '#cfe0ff', 'dark': '#17131a'}.get(ink, ink); a = layer[..., 3]; L = layer[..., :3].mean(-1) / np.maximum(a, .02); x, y = grid(); dv = np.array(drift, float); dn = np.hypot(*dv); th = math.atan2(dv[1], dv[0]) if dn > 1 else -math.pi / 2
    n1 = nimg(H, W, 90, age * .04, seed, 3, 4); n2 = nimg(H, W, 44, age * .06, seed + 5, 2, 4); sp = 5 + 46 * u ** 1.4; dx, dy = (n1 * sp + dv[0] * u).astype(f32), (n2 * sp * .8 + dv[1] * u).astype(f32); m = cv2.remap(a, x - dx, y - dy, cv2.INTER_LINEAR); l = cv2.remap(L, x - dx, y - dy, cv2.INTER_LINEAR)
    c_, s_ = math.cos(th), math.sin(th); U_, V_ = x * c_ + y * s_, -x * s_ + y * c_; q = 4; uu, vv = U_[::q, ::q], V_[::q, ::q]; hair = noise(uu / 260., vv / 5.2, 0, seed + 11) * .6 + noise(uu / 90., vv / 11., 0, seed + 12) * .5; hair = cv2.resize(hair.astype(f32), (W, H), interpolation=cv2.INTER_LINEAR); g = nimg(H, W, 5, 0, seed + 9, 2, 2)
    pool = np.clip((m - blur(m, 3.5 + 5 * u)) * 2.6, 0, 1); rim_ = np.clip((blur(m, 9) - blur(m, 26)) * .9, 0, 1) * m; tone = (l if ink == 'light' else 1 - l); wash = blur(m, 1.2 + 4 * u) * (.16 + .62 * np.round(tone * 3) / 3.); grain = 1 + .26 * g + .22 * n2; hole = np.clip((hair + .95 - 1.75 * u + .25 * g) * 4., 0, 1); d = np.clip((wash * grain + pool * .75 + rim_ * .25), 0, 1) * hole * (1 - u) ** .6 * strength; return np.dstack([d[..., None] * C(col), d]).astype(f32)


# ------------------------------------------------------------------ water ring in space
def water_ring(img, cx, cy, f, t0=0, speed=34., asp=.42, rot=0., seed=1, amp=26., kind='silver', rings=4, life=26):
    """a palm is laid on nothing and space ripples like water: refraction + lit crests + dark troughs + a few round drops -> RGB"""
    a = f - t0
    if a < 0 or a > life + 10: return img
    o = img; lays = []; F = Field(3)
    for i in range(rings):
        q = a - i * 3.2
        if q <= 0: continue
        rad = 20 + speed * 22 * (1 - math.exp(-q / 14.)) * (1 - .10 * i); k = math.exp(-q / (life * .42)) * (1 - .16 * i)
        if k < .05: continue
        o = refract_ring(o, cx, cy, rad, 34 + rad * .10, amp * 1.5 * k, asp, rot); P, th = ellipse_pts(cx, cy, rad, rad * asp, rot, 140, wob=.012, seed=seed + i, z=a * .05); n = np.stack([np.cos(th), np.sin(th)], -1); tk = (5. + rad * .022) * k * np.clip(.65 + .75 * noise(np.cos(th) * 2.4, np.sin(th) * 2.4, i + a * .03, seed), 0, 1.5) * (.55 + .45 * np.sin(th + 2.2)); F.rib(P, [tk + .7 * k, tk * .62, tk * .25 - .2], 0.)
        P2, _ = ellipse_pts(cx, cy, rad * .95, rad * .95 * asp, rot, 140, wob=.012, seed=seed + i, z=a * .05); D = Field(1); D.rib(P2, [tk * .8 * np.clip(noise(np.cos(th) * 3, np.sin(th) * 3, i, seed + 3) + .3, 0, 1) - .2], 0.); lays.append(D.cel(('#05081c',)) * .55); Z = Field(1); P3, _ = ellipse_pts(cx, cy, rad * .90, rad * .90 * asp, rot, 140, wob=.02, seed=seed + i, z=a * .05); Z.rib(P3, [(8 + rad * .05) * k * np.clip(.4 + .8 * noise(np.cos(th) * 1.6, np.sin(th) * 1.6, i, seed + 6), 0, 1.3) - .3], 0.); lays.append(Z.cel((pal(kind)['c'][1],)) * .26)
    for j in range(16):
        s0 = h1(seed, j, 1); th = TAU * h1(seed, j, 2); v = 5 + 16 * h1(seed, j, 3); q = a - 1; r = v * q * math.exp(-q / 40.); z = 7 * s0 ** 2 * math.exp(-q / 18.); c, s = math.cos(rot), math.sin(rot); x, y = math.cos(th) * r, math.sin(th) * r * asp - (40 + 60 * s0) * (1 - math.exp(-q / 10.)) * .5; p = np.array([cx + x * c - y * s, cy + x * s + y * c])
        if q > 0 and z > .6: F.rib(np.array([p - [z * .3, 0], p, p + [z * .3, 0]]), [np.full(3, z), np.full(3, z * .7), np.full(3, z * .25)], 0., [0, -z * .1, -z * .3])
    P_ = pal(kind); lays.append(F.cel((P_['c'][0], P_['c'][2], '#ffffff'))); return put(o, stack(*lays))
