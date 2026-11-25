"""phys dust: dust, smoke, steam, explosion cloud as drawn cauliflower lobes: a z-buffer of rolling noise spheres, cut into hard cel tones with a thick-thin line.  (PHYS lane)"""
from .phys_core import *
from .phys_core import _hash3

# a lobe row: 0 X, 1 Y, 2 Z, 3 R (metres), 4 seed, 5 roll, 6 axis angle on screen, 7 erode 0..1, 8 heat 0..1, 9 floor z (metres, -1e9 none)


@njit(cache=True, fastmath=True)
def _grow(A, K1, K2, rmin2, out, meta):
    """lobe rows (px) -> sphere rows [x, y, z, r, seed] + [lobe index, erode, heat, floor, lx, ly, lz, lr]; children ride on the parent's skin and turn with its roll"""
    n = 0; cap = out.shape[0]; ga = 2.399963
    for i in range(A.shape[0]):
        cx, cy, cz, r, sd, roll, ax, er, ht, fy = A[i]; s0 = int(sd) * 131 + 7; ca = math.cos(ax); sa = math.sin(ax); cr = math.cos(roll); sr = math.sin(roll); k1_ = 1 - cr
        core = r * (1 - .80 * max(0., er - .12) / .88); meta0 = (i, er, ht, fy)
        if core > 1. and n < cap: out[n, 0] = cx; out[n, 1] = cy; out[n, 2] = cz; out[n, 3] = core * .80; out[n, 4] = sd; meta[n, 0] = i; meta[n, 1] = er; meta[n, 2] = ht; meta[n, 3] = fy; n += 1
        kk = K1 if r > 9 else (5 if r > 5 else 0); o1 = (_hash3(s0, 1, 2) * .5 + .5) * 6.283
        for k in range(kk):
            h = _hash3(s0, k, 3) * .5 + .5; h2_ = _hash3(s0, k, 5) * .5 + .5; h3_ = _hash3(s0, k, 9) * .5 + .5
            zz = 1 - 2 * (k + .5) / kk + .5 * (h - .5) / kk; q = math.sqrt(max(0., 1 - zz * zz)); th = ga * k + o1 + .6 * (h2_ - .5); px = q * math.cos(th); py = q * math.sin(th); pz = zz
            dt = px * ca + py * sa; x_ = px * cr + sa * pz * sr + ca * dt * k1_; y_ = py * cr - ca * pz * sr + sa * dt * k1_; z_ = pz * cr + (ca * py - sa * px) * sr
            if z_ < -.75: continue
            rb = (.22 + .26 * h3_ * h3_ + .10 * h); sh = 1 - max(0., er * 1.9 - .75 * h2_)
            if sh <= .06 or (er > .25 and rb * sh < .17): continue
            dist = .80 + .16 * h + .55 * er * h2_; rb *= r * min(1., sh); bx = cx + x_ * r * dist; by = cy + y_ * r * dist; bz = cz + z_ * r * dist
            if n < cap: out[n, 0] = bx; out[n, 1] = by; out[n, 2] = bz; out[n, 3] = rb; out[n, 4] = s0 + k; meta[n, 0] = i; meta[n, 1] = er; meta[n, 2] = ht; meta[n, 3] = fy; n += 1
            if rb > rmin2:
                nk = min(K2 + int(rb / 16.), K2 + 8)
                for j in range(nk):
                    u = _hash3(s0 + k * 17, j, 11) * .5 + .5; u2 = _hash3(s0 + k * 17, j, 13) * .5 + .5; u3 = _hash3(s0 + k * 17, j, 15) * .5 + .5; a2 = ga * j + u * 1.2 + roll * .7; t2 = .35 + 1.15 * u2; w = math.sin(t2); qx = w * math.cos(a2); qy = w * math.sin(a2); qz = math.cos(t2)
                    sx = x_ * .9 + qx * .8; sy = y_ * .9 + qy * .8; sz = z_ * .5 + qz * .55; l = 1 / (math.sqrt(sx * sx + sy * sy + sz * sz) + 1e-6); rs = rb * (.18 + .30 * u3 * u3 + .06 * u)
                    if n < cap: out[n, 0] = bx + sx * l * rb * .86; out[n, 1] = by + sy * l * rb * .86; out[n, 2] = bz + sz * l * rb * .86; out[n, 3] = rs; out[n, 4] = s0 + k * 31 + j; meta[n, 0] = i; meta[n, 1] = er; meta[n, 2] = ht; meta[n, 3] = fy; n += 1
    return n


@njit(cache=True, fastmath=True)
def _splat(h, w, A, S, M, n, rough, zb, idb, hb, N):
    for i in range(n):
        cx, cy, cz, r, sd = S[i]; li = int(M[i, 0]); fy = M[i, 3]; ht = M[i, 2]; ro = rough + .42 * min(1., M[i, 1]); er = min(1., M[i, 1]); st = 1.5 if fy < 1e7 else .45; kx = 1 / (1 + st * er); ky = 1 + .37 * st * er; R_ = r * (1 + ro) * (1 + st * er) + 1
        x0 = max(0, int(cx - R_)); x1 = min(w - 1, int(cx + R_) + 1); y0 = max(0, int(cy - R_)); y1 = min(h - 1, int(cy + r * 1.7) + 1); y0 = max(0, int(cy - r * 1.7))
        if x0 > x1 or y0 > y1: continue
        o = (sd * 1.37) % 97.; lx = A[li, 0]; ly = A[li, 1]; lz = A[li, 2]; lr = A[li, 3] * 1.1; ir = 1 / r
        for y in range(y0, y1 + 1):
            if y > fy + lr * .16: break
            v = (y - cy) * ir * ky; nearf = y > fy - lr * .16
            for x in range(x0, x1 + 1):
                u = (x - cx) * ir * kx; d2 = u * u + v * v
                if d2 > 2.1: continue
                if nearf and y > fy + lr * .15 * vn3(x / (lr * .42) + o * .01, li * 3.7, .5): continue
                q = 1 + ro * vn3(u * 1.7 + o, v * 1.7 - o, o * .3); g = q * q - d2
                if g <= 0: continue
                hh = math.sqrt(g); z = cz + r * hh
                if z > zb[y, x]:
                    zb[y, x] = z; idb[y, x] = li; hb[y, x] = ht; l = 1 / math.sqrt(d2 + hh * hh); ax = u * l; ay = v * l; az = hh * l
                    bx = (x - lx) / lr; by = (y - ly) / lr; bz = max(.02, (z - lz) / lr); l = 1 / math.sqrt(bx * bx + by * by + bz * bz); N[y, x, 0] = ax * .42 + bx * l * .58; N[y, x, 1] = ay * .42 + by * l * .58; N[y, x, 2] = az * .42 + bz * l * .58


def lobes_layer(cam, L, pal='night', light=(-.50, -.72, .48), ss=1.5, ink=1.0, t_lit=.60, t_dark=.20, group=.36, ao=.70, rim=None, rim_amt=1., heat=None, h=H, w=W, fog=None, min_r=1.5, ink_in=1.0, K1=15, K2=6, micro=20., rough=.10):
    """lobe rows (world) -> premultiplied RGBA layer (h, w, 4).  light = SCREEN light vector (x right, y down, z to us).
    pal: DUST key or dict lit/mid/dark/ink/rim.  ink = outer line px (0 none), ink_in = inner line strength 0..1.  heat = None or emission ramp [(0,'#outer'),(.5,..),(1,'#core')] for rows with heat > 0.
    rim = screen direction of a hard rim light.  group = how much the whole mass is lit as one form.  micro = bump radius px above which it grows its own small bumps."""
    out = np.zeros((h, w, 4), f32); L = np.asarray(L, float)
    if L.size == 0: return out
    pal = DUST[pal] if isinstance(pal, str) else pal; c = cam.cam(L[:, :3]); ok = c[:, 2] > .3; L, c = L[ok], c[ok]
    if not len(L): return out
    s = cam.f / c[:, 2]; sx = c[:, 0] * s + cam.cx; sy = c[:, 1] * s + cam.cy; r = L[:, 3] * s; keep = (r > min_r) & (sx > -r * 1.5) & (sx < w + r * 1.5) & (sy > -r * 1.5) & (sy < h + r * 1.5); L, c, s, sx, sy, r = L[keep], c[keep], s[keep], sx[keep], sy[keep], r[keep]
    if not len(L): return out
    zr = np.median(c[:, 2]); sz = -(c[:, 2] - zr) * (cam.f / zr); fl = L[:, 9]; fp = np.c_[L[:, 0], L[:, 1], fl]; fy = np.where(fl > -1e8, (cam.cam(fp)[:, 1] * s + cam.cy), 1e9)
    x0, y0 = max(0, int((sx - r * 1.6).min()) - 8), max(0, int((sy - r * 1.6).min()) - 8); x1, y1 = min(w, int((sx + r * 1.6).max()) + 9), min(h, int((sy + r * 1.6).max()) + 9); hh, ww = int((y1 - y0) * ss), int((x1 - x0) * ss)
    if hh < 4 or ww < 4: return out
    A = np.stack([(sx - x0) * ss, (sy - y0) * ss, sz * ss, r * ss, L[:, 4], L[:, 5], L[:, 6], np.clip(L[:, 7], 0, 1.2), L[:, 8], (fy - y0) * ss], 1)
    SP = np.empty((len(A) * (1 + K1 * (1 + K2 + 8)), 5)); MT = np.empty((len(SP), 4)); n = _grow(A, K1, K2, micro * ss, SP, MT)
    zb = np.full((hh, ww), -1e9, np.float64); idb = np.full((hh, ww), -1, np.int32); hb = np.zeros((hh, ww), f32); N = np.zeros((hh, ww, 3), f32); _splat(hh, ww, A, SP, MT, n, rough, zb, idb, hb, N)
    m = idb >= 0
    if not m.any(): return out
    mf = m.astype(f32); l = unit(light).astype(f32); N = cv2.GaussianBlur(N, (0, 0), .7 * ss); N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-4; lam = N @ l; zf = np.where(m, zb, 0).astype(f32); rm = float(np.percentile(A[:, 3], 60)); zmin = float(zb[m].min()); sg = max(6., rm * .9); q = 4; sm = lambda a: cv2.resize(cv2.GaussianBlur(cv2.resize(a, (max(2, ww // q), max(2, hh // q)), interpolation=cv2.INTER_AREA), (0, 0), sg / q), (ww, hh), interpolation=cv2.INTER_LINEAR)
    wz = sm(mf) + 1e-3; zs = sm(np.where(m, zf - zmin, 0)) / wz; Zb = sm((zf - zmin + rm) * mf); gx = cv2.Sobel(Zb, cv2.CV_32F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(Zb, cv2.CV_32F, 0, 1, ksize=3) / 8; gx = cv2.GaussianBlur(gx, (0, 0), 2.6 * ss); gy = cv2.GaussianBlur(gy, (0, 0), 2.6 * ss); nb = np.stack([-gx, -gy, np.full_like(gx, .6)], -1); nb /= np.linalg.norm(nb, axis=-1, keepdims=True); lamB = nb @ l
    occ = np.clip((zs - (zf - zmin)) / (rm * .9 + 1e-3), 0, 1.1); S = lam * (1 - group) + lamB * group - ao * occ + .12
    fw = np.abs(cv2.Sobel(S, cv2.CV_32F, 1, 0, ksize=1)) + np.abs(cv2.Sobel(S, cv2.CV_32F, 0, 1, ksize=1)); fw = np.clip(fw * .7, .010, .08)
    tl = np.clip((S - t_lit) / fw + .5, 0, 1)[..., None]; td = np.clip((t_dark - S) / fw + .5, 0, 1)[..., None]; col = C(pal['mid']) * (1 - tl) + C(pal['lit']) * tl; col = col * (1 - td) + C(pal['dark']) * td
    if heat is not None and (hb > 0).any():
        hv = hb * .86 + occ * .62 * (hb > 0) - .40 * np.clip(lam, -.2, 1) * (hb > 0) + .10; hv = np.clip(hv, 0, 1); fq = np.abs(cv2.Sobel(hv, cv2.CV_32F, 1, 0, ksize=1)) + np.abs(cv2.Sobel(hv, cv2.CV_32F, 0, 1, ksize=1)); fq = np.clip(fq, .010, .07); hc = np.empty_like(col); hc[:] = C(heat[0][1])
        for t0, cc in heat[1:]: k = np.clip((hv - t0) / fq + .5, 0, 1)[..., None]; hc = hc * (1 - k) + C(cc) * k
        k = np.clip((hv - heat[0][0]) / fq + .5, 0, 1)[..., None] * (hb > 0)[..., None]; col = col * (1 - k) + hc * k
    if rim is not None:
        d = unit([rim[0], rim[1], 0])[:2]; px = max(1., 2.4 * ss); sh = cv2.warpAffine(mf, np.float32([[1, 0, -d[0] * px], [0, 1, -d[1] * px]]), (ww, hh)); k = np.clip(mf - sh, 0, 1); k = cv2.GaussianBlur(k, (0, 0), .5 * ss)[..., None] * rim_amt; col = col * (1 - k) + C(pal['rim']) * k
    if ink > 0 and ink_in > 0:
        kz = 2 * max(1, int(round(.8 * ss))) + 1; zd = cv2.dilate(zf, np.ones((kz, kz), np.uint8)); jump = (zd - zf) * m; th = max(3., rm * .20); k = smooth(th, th * 1.4, jump) * (.5 + .5 * smooth(th, rm * .7, jump)); k = cv2.GaussianBlur(k.astype(f32), (0, 0), .42 * ss) * (hb < .5); kl = np.clip(k * 1.6, 0, 1) * (S >= t_lit + .06); k = np.clip(k * 1.25 * ink_in, 0, 1) * (S < t_lit + .06); col = col * (1 - k[..., None]) + C(pal['ink']) * k[..., None]; kl = kl[..., None] * .9; col = col * (1 - kl) + C(pal['mid']) * kl
    a = cv2.GaussianBlur(mf, (0, 0), .45 * ss); dil = cv2.dilate(col * m[..., None], np.ones((3, 3), np.uint8), iterations=2); col = np.where(m[..., None], col, dil)
    if ink > 0:                                              # outer line: heavier on the shadow side, opens on the lit side
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(round(ink * 1.1 * ss)) + 1,) * 2); big = cv2.dilate(mf, k1); big = cv2.warpAffine(big, np.float32([[1, 0, -l[0] * ink * .8 * ss], [0, 1, -l[1] * ink * .8 * ss]]), (ww, hh)); big = cv2.GaussianBlur(np.maximum(big, mf), (0, 0), .45 * ss); fill = cv2.GaussianBlur(mf, (0, 0), .4 * ss); rgbp = C(pal['ink']) * (1 - fill[..., None]) + col * fill[..., None]; a = big
    else: rgbp = col
    o = np.dstack([rgbp * a[..., None], a]).astype(f32)
    if fog is not None: k = 1 - math.exp(-max(0., zr - fog[1]) * fog[2]); o[..., :3] = o[..., :3] * (1 - k) + C(fog[0]) * o[..., 3:4] * k
    out[y0:y1, x0:x1] = cv2.resize(o, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA); return out


def lobe(X, Y, Z, r, seed, roll=0., ax=0., er=0., heat=0., floor=0.): return (X, Y, Z, r, seed % 9973, roll, ax, er, heat, floor)
def age_erode(q, a=.42): return clamp((q - a) / (1 - a)) ** 1.15 * 1.05


# ------------------------------------------------------------- generators (world metres, t seconds) -> list of lobe rows
def dust_ring(t, centre=(0, 0), r0=.6, v0=9., minor=.8, n=22, m=3, seed=0, life=1.6, drag=2.2, floor=0., arc=(0, TAU), lag=.35, rise=.35, squash=1.):
    """ground vortex ring running outward (landing, shock): wheels roll, tops go outward; tail puffs lag behind and thin out"""
    if t <= 0 or t > life * 1.15: return []
    out = []; q = t / life; gr = (1 - math.exp(-drag * t)) / drag
    for i in range(n):
        az = arc[0] + (arc[1] - arc[0]) * (i + .7 * h1(seed, i, 1)) / n; sc = .55 + .75 * size3(h1(seed, i, 2), .3, .4); a = minor * sc * (.30 + .70 * (1 - math.exp(-5 * t))) * (1 + .35 * q); Rm = r0 + v0 * gr * (.80 + .35 * h1(seed, i, 3)); om = (v0 * math.exp(-drag * t) / max(a, .05)) * .55
        for j in range(m + 1):
            tail = j == m; ph = TAU * (j + h1(seed, i, j, 4)) / m - 2.4 * (1 - math.exp(-1.3 * t)) * (1.2 + v0 * .05); RR = Rm - (a * (1.2 + 1.5 * h1(seed, i, 9)) + lag * Rm * h1(seed, i, 10) if tail else 0); ca, sa = math.cos(az), math.sin(az); X = centre[0] + ca * (RR + a * .62 * math.cos(ph)); Y = centre[1] + sa * squash * (RR + a * .62 * math.cos(ph)); Z = a * (.62 + .55 * math.sin(ph)) + (rise * t * (1 + tail)) ; r = a * (.50 + .34 * h1(seed, i, j, 5)) * (.8 if tail else 1.) * (1 - .25 * max(0., q - .5))
            out.append(lobe(X, Y, Z + floor, r, seed * 131 + i * 17 + j, -ph * .8 * (1 if ca > 0 else -1), math.pi / 2 + .5 * sa, age_erode(q * (.85 + .4 * h1(seed, i, j, 6)) + (.15 if tail else 0)), 0., floor - .05))
    return out


def plume(t, origin=(0, 0, 0), dirn=(0, 0, 1), v0=9., n=26, r=(.4, 1.5), spread=.45, emit=.35, seed=0, life=2.0, drag=2.0, rise=.6, floor=-1e9, grow=2.2, swirl=1.6, jit=.35):
    """thrown dust / smoke: puffs leave the origin over `emit` seconds along a cone, slow down, swell, roll over outward, then tear and thin out"""
    out = []; d = unit(dirn); o = np.asarray(origin, float); e1 = unit(np.cross(d, [.31, .95, .12])); e2 = np.cross(d, e1)
    for i in range(n):
        u = h1(seed, i, 1); tb = emit * (i / max(1, n - 1)) ** 1.2; s = t - tb; lf = life * (.65 + .6 * h1(seed, i, 2))
        if s <= 0 or s > lf: continue
        q = s / lf; th = h1(seed, i, 3) * TAU; z = size3(h1(seed, i, 4), .26, .40); sp = spread * math.sqrt(h1(seed, i, 5)); v = (d * math.cos(sp) + (e1 * math.cos(th) + e2 * math.sin(th)) * math.sin(sp)) * v0 * (.45 + .75 * h1(seed, i, 6)) * (1.15 - .35 * tb / (emit + 1e-6)); g = (1 - math.exp(-drag * s)) / drag; p = o + v * g + np.array([0, 0, rise * s]) + (e1 * math.cos(th * 3) + e2 * math.sin(th * 3)) * jit * r[1] * h1(seed, i, 7)
        rr = (r[0] + (r[1] - r[0]) * z) * (.28 + .72 * (1 - math.exp(-grow * s))) * (1 - .22 * max(0., q - .55)); sg = 1 if (v[0] > 0) else -1
        out.append(lobe(p[0], p[1], p[2], rr, seed * 131 + i, sg * swirl * (1 - math.exp(-1.4 * s)) * 1.6 + i, math.pi / 2 + (h1(seed, i, 8) - .5), age_erode(q), 0., floor))
        if z > .7:
            for j in range(3): b = TAU * h1(seed, i, j, 9) + sg * swirl * (1 - math.exp(-1.4 * s)); out.append(lobe(p[0] + math.cos(b) * rr * .78, p[1] - rr * .25 * h1(seed, i, j, 12), p[2] + math.sin(b) * rr * .62 + rr * .12, rr * (.30 + .28 * h1(seed, i, j, 10)), seed * 131 + i * 7 + j + 3, sg * swirl * s + j, math.pi / 2, age_erode(q * 1.12), 0., floor))
    return out


def dust_trail(t, at, front, s_max, step=1.2, seed=0, r=(.3, .95), side=2.2, up=1.6, life=2.2, drag=1.8, w=1.2, floor=0., every=1):
    """dust thrown out of both lips as a front passes.  at(s) -> (xy, left normal), front(t) -> metres"""
    out = []; ts = np.linspace(0, max(t, 1e-3), 160); ss = np.array([front(q) for q in ts])
    for k in range(int(min(s_max, ss[-1]) / step) + 1):
        s0 = k * step
        if s0 > ss[-1]: break
        t0 = float(np.interp(s0, ss, ts)); s = t - t0; p, nn = at(s0)
        for sd in (-1, 1):
            for j in range(2):
                lf = life * (.6 + .7 * h1(seed, k, sd + 1, j, 1)); 
                if s <= 0 or s > lf: continue
                q = s / lf; z = size3(h1(seed, k, sd + 1, j, 2), .25, .45); g = (1 - math.exp(-drag * s)) / drag; fw = np.array([nn[1], -nn[0]]); x = p + nn * sd * (w + side * g * (.5 + 1.1 * h1(seed, k, sd + 1, j, 3))) + fw * g * 2.5 * (h1(seed, k, sd + 1, j, 4) - .2); zz = up * g * (.3 + 1.1 * h1(seed, k, sd + 1, j, 5)) + .25 * s; rr = (r[0] + (r[1] - r[0]) * z) * (.25 + .75 * (1 - math.exp(-3.2 * s)))
                out.append(lobe(x[0], x[1], zz + rr * .35, rr, seed * 71 + k * 5 + j + (sd + 1) * 3, sd * 1.7 * (1 - math.exp(-1.2 * s)) + k, math.pi / 2, age_erode(q), 0., floor - .1))
    return out


def wisps(cel, cam, t, centre, n=14, r=(2., 6.), seed=0, col='#8d82c4', col2='#564e86', life=2.5, drift=(1.2, 0, .5), lw=(3, 14)):
    """what is left when dust thins out: long curling hooks, two tone, drawn as tapered strokes"""
    for i in range(n):
        q = t / life / (.7 + .6 * h1(seed, i, 1))
        if not 0 < q < 1: continue
        a = h1(seed, i, 2) * TAU; R_ = r[0] + (r[1] - r[0]) * h1(seed, i, 3); p = np.array([centre[0] + math.cos(a) * R_, centre[1] + math.sin(a) * R_ * .6, centre[2] + R_ * .5 * h1(seed, i, 4)]) + np.asarray(drift) * t; l = R_ * (.35 + .5 * h1(seed, i, 5)) * (1 + q); v = np.linspace(0, 1, 18); sg = 1 if h1(seed, i, 6) > .5 else -1; th = h1(seed, i, 7) * .8 + sg * (1.0 + 3.2 * q) * v ** 1.7; st = l / 17; x = np.cumsum(np.cos(th)) * st; z = np.cumsum(np.sin(th)) * st * .6; P = cam.project(np.c_[p[0] + x, np.full(18, p[1]), p[2] + z])[0]; wd = (lw[0] + (lw[1] - lw[0]) * h1(seed, i, 8)) * math.sin(math.pi * q) ** .6; thin(cel, P, wd, col2, .8, .35, .6); thin(cel, P + [-wd * .12, -wd * .22], wd * .5, col, .85, .4, .7)
    return cel
