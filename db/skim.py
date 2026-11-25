"""A bright sea taken at speed, for the kid's skim (a3): flat painted facets in perspective, a V wake cut by a hand, spray on true parabolas, a broken reflection.
    S = Skim(hz=300, yc=806, V=34)            # hz = horizon line, yc = screen y of the contact line, V = px per frame the water runs LEFT at yc
    im = S.sea(f, sky)                        # sky = (H, W, 3) picture above the horizon; returns the whole picture with water
    im = S.reflect(im, layer, yw, f)          # premultiplied RGBA rider layer mirrored about y = yw, torn by the waves
    im = S.wake(im, x, y, f)                  # V wake of a hand at (x, y): put UNDER the hand
    im = S.spray(im, x, y, f, front=False/True)"""
import numpy as np, cv2, math
from numba import njit
from . import W, H
from .paint import rgb, f32, smooth, ramp, fbm, _vn, grid
from .ink import Cel, spl
from .comp import over, blur

pi = math.pi


@njit(cache=True, fastmath=True)
def _field(hz, K, X0, seed, w, h, bob):
    """wave height and slope toward +x (the sun side) in perspective, octaves dropped before they alias"""
    A = np.zeros((h, w), np.float32); G = np.zeros((h, w), np.float32)
    for j in range(h):
        d = j + .5 - hz
        if d < 1: continue
        Z = K / d; pxu = d / 500. * 500. / 1.0
        for i in range(w):
            X = (i - w * .5) * Z / 500. + X0
            v = 0.; g = 0.; a = 1.; fx = 1.15; fz = 2.9
            for o in range(5):
                lam = d * d / (K * fz)                        # px height of one wave row
                at = (lam - 1.2) / 2.6
                if at <= 0: break
                if at > 1: at = 1.
                s = seed + o
                wx = X * fx + bob * (o + 1) * .6; wz = Z * fz + o * 7.7
                q = _vn(wx * .35 + 5, wz * .5 + 9, s + 50) * 2.2
                n = _vn(wx + q, wz - q * .5, s); n2 = _vn(wx + q + .22, wz - q * .5, s)
                n = 1 - abs(2 * n - 1); n2 = 1 - abs(2 * n2 - 1)
                v += a * at * (n - .5); g += a * at * (n2 - n); a *= .62; fx *= 1.9; fz *= 1.85
            A[j, i] = v; G[j, i] = g
    return A, G


DAY = dict(far=('#d6f2ee', '#9adfe6', '#5cc1dd', '#36a3cf'), mid=('#8fe3e6', '#3fb9dc', '#1c93c6', '#1276ae'), near=('#49c8dd', '#1596cb', '#0c74b0', '#0a5a96'))


class Skim:
    def __init__(s, hz=300, yc=806, V=34., K=640., pal=None, seed=3):
        s.hz, s.yc, s.V, s.K, s.pal, s.seed = hz, yc, V, K, pal or DAY, seed; s.dX = V * (K / (yc - hz)) / 500.; s.x, s.y = grid()

    def sea(s, f, sky, sun=(1500, 120)):
        A, G = _field(float(s.hz), s.K, f * s.dX, s.seed, W, H, f * .02); s.A, s.G = A, G; t = np.clip((s.y - s.hz) / (H - s.hz), 0, 1); q = [(0., 'far'), (.30, 'mid'), (1., 'near')]
        c = [ramp(t ** .75, [(p, s.pal[n][i]) for p, n in q]) for i in range(4)]; e = .030
        o = c[3] + (c[2] - c[3]) * smooth(-.30 - e, -.30 + e, A)[..., None]; o = o + (c[1] - o) * smooth(.06 - e, .06 + e, A + G * .4)[..., None]; o = o + (c[0] - o) * (smooth(.20 - e, .20 + e, G + A * .22))[..., None]
        # glitter road under the sun: hard chips, denser toward its azimuth
        road = np.exp(-((s.x - sun[0]) / (120 + 420 * t)) ** 2) * smooth(.0, .05, t); chip = smooth(.30, .34, G + .30 * A - .30 * (1 - road)); o = o + (np.array([1.12, 1.08, .94], f32) - o) * (chip * np.clip(road * 1.3, 0, 1))[..., None]
        hzl = np.exp(-np.clip(s.y - s.hz, 0, None) / 10.)[..., None]; o = o + (rgb('#fff3d8') - o) * hzl * .55
        m = smooth(s.hz - .7, s.hz + .7, s.y)[..., None]; return (sky * (1 - m) + o * m).astype(f32)

    def reflect(s, im, L, yw, f, amt=.55, tint='#1c86b8', squash=.78):
        """mirror about yw (screen), compressed, shifted by the wave slope, cut by the bright facets"""
        yy = yw - (s.y - yw) / squash; dx = s.G * (34 + .22 * np.clip(s.y - yw, 0, 400)) + s.A * 18; R = cv2.remap(L, (s.x + dx).astype(f32), (yy + s.A * 44).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        a = R[..., 3] * smooth(yw - 2, yw + 8, s.y) * (1 - smooth(yw + 90, yw + 340, s.y)) * amt * (1 - .85 * smooth(.02, .14, s.G + .35 * s.A)); col = R[..., :3] / np.maximum(R[..., 3:4], 1e-3); col = col * .70 + rgb(tint) * .30; return im * (1 - a[..., None]) + col * a[..., None]

    def _persp(s, x, y, X, Z):
        """offsets on the water from the contact point: X px along the run (negative = behind), Z px toward us, at the scale of y"""
        yy = y + Z * (1 + Z / 900.); k = (yy - s.hz) / (y - s.hz); return np.stack([W / 2 + (x - W / 2 + X) * k, yy], -1)

    def wake(s, im, x, y, f, scale=1., amt=1., age=999.):
        """two crests that peel off the hand at the same angle forever, foam between them"""
        um = min(1., age * s.V * 2.3 / (1150. * scale) + .02); u = np.linspace(0, um, 60); V = s.V; cel = Cel(W, H, ss=2); cB = Cel(W, H, ss=2)
        ph = f * V
        for sd, spread in ((-1, 74.), (1, 190.)):
            X = -u * 1150 * scale; Z = sd * spread * scale * u ** .86; wob = 5 * np.sin((X + ph) * .021 + sd) * u + 3 * np.sin((X + ph) * .057 + 2 * sd) * u; C = s._persp(x, y, X, Z + wob * .6); k = np.interp(u, [0, .06, .3, 1], [.3, 1., .85, .25]) * np.clip((um - u) / .10 + (um >= 1) * 1., 0, 1); hgt = (26 if sd > 0 else 14) * scale * k * (1 + .22 * np.sin((X + ph) * .035 + sd * 2)); Tn = np.gradient(C, axis=0); Tn /= np.hypot(*Tn.T)[:, None] + 1e-6
            up = np.c_[0 * u, -hgt]; base = C + np.c_[0 * u, hgt * .30]; out = C + np.c_[0 * u, hgt * .30] + np.c_[-Tn[:, 1], Tn[:, 0]] * (hgt * .9 * (1 if sd < 0 else 1) * -sd)[:, None] * .0
            cB.fill(np.vstack([base + [0, 2], (C + np.c_[0 * u, hgt * (1.5 if sd > 0 else .8)])[::-1]]), '#0a5f9a', .55 * amt)         # trough in shade
            cel.fill(np.vstack([C + up, base[::-1]]), '#62cfe2' if sd > 0 else '#3fb0d4', .92 * amt)                                          # lit face, light comes through it
            mid = C + up * .52 + np.c_[0 * u, 2.5 * np.sin((X + ph) * .09)]; cel.fill(np.vstack([C + up, mid[::-1]]), '#c9f3f2', .95 * amt)
            lip = C + up + np.c_[0 * u, np.minimum(0, -hgt * .26 * np.abs(np.sin((X + ph) * .047 + sd)) ** 3)]; cel.fill(np.vstack([lip, (C + up * .80)[::-1]]), '#ffffff', amt)
        # foam: torn lace between the arms, stretched by speed
        im = over(im, cB.layer())
        n = fbm_shift(f * V * .5); yy = s.y; u2 = np.clip((x - s.x) / (1150. * scale), 0, 1.2); z = (yy - y); zn = np.where(z < 0, -z / (74. * scale * np.maximum(u2, .02) ** .86 + 6), z / (190. * scale * np.maximum(u2, .02) ** .86 + 6)); inside = (s.x < x + 4) * smooth(um + .02, um - .06, u2 + (um >= 1) * -1.) * smooth(1.0, .86, zn) * smooth(1.12, .4, u2); th = .05 + .95 * u2 + .35 * (1 - np.abs(zn)); fo = smooth(th, th + .10, n) * inside; fo2 = smooth(th + .55, th + .62, n) * inside
        im = im + (rgb('#bfeef2') - im) * (fo * .55 * amt)[..., None]; im = im + (rgb('#ffffff') - im) * (fo2 * .9 * amt)[..., None]
        return over(im, cel.layer())

    def spray(s, im, x, y, f, front=True, scale=1., n=13, seed=5, amt=1., g=1.75, sheet=True, age=999., kick=1.):
        """a clear curling sheet off the palm, then streaklines of a moving source (true parabolas) that break into drops; the drops run along them and fall.  age = frames since the hand touched, kick > 1 = the first hit"""
        cel = Cel(W, H, ss=2); r = np.random.default_rng(seed + (50 if front else 0)); V = s.V; sg = 1. if front else -.6
        if sheet:
            for q, (ln, hh, sp, col, col2) in enumerate(((360, 50, 36, '#74cfe2', '#c4f1f4'), (230, 34, 44, '#a5e6ee', '#f4ffff')) if front else ((440, 150, -26, '#62c2da', '#aee6ee'), (300, 96, -10, '#86d6e5', '#d8f6f6'))):
                l0 = ln * scale; ln = min(l0, age * V * 1.9 + 40); hh = hh * scale * kick * min(1., ln / l0) ** .8; u = np.linspace(0, 1, 64); X = x + 6 - u * ln; arch = 4 * u * (1 - u) * (1 - .30 * u); ph = f * V / 130.
                sc = .80 + .20 * np.abs(np.sin(u * ln / 130. * pi + ph * pi + q * 1.3)) ** .8 + .07 * np.sin(u * ln / 47. * pi + ph * 2.7 * pi); top = y + sp * u - hh * arch * sc - 8 * scale * (1 - u); bot = y + sp * u * 1.5 + 6 * np.sin(u * 14 + f * .8) * u; bot = top + (bot - top) * np.clip((1 - u) / .5, 0, 1) ** .7
                top = np.convolve(np.r_[top[:1], top[:1], top, top[-1:], top[-1:]], [.1, .2, .4, .2, .1], 'valid')
                cel.fill(np.vstack([np.c_[X, top], np.c_[X, bot][::-1]]), col, .84 * amt)
                th = np.clip(.20 + .16 * np.sin(u * ln / 90. * pi + ph * pi + 1), .06, .5) * np.clip(1.25 - u, .2, 1); t2 = top + (bot - top) * th; cel.fill(np.vstack([np.c_[X, top], np.c_[X, t2][::-1]]), '#ffffff', .96 * amt)
                for j, (v0, v1) in enumerate(((.48, .60), (.74, .80))):      # lines of flow inside the sheet
                    m = slice(4 + 3 * j, 52 - 5 * j); a0 = (top + (bot - top) * (v0 + .06 * np.sin(u * 7 - f * .6 + j)))[m]; a1 = (top + (bot - top) * (v1 + .05 * np.sin(u * 9 - f * .6 + j)))[m]; w_ = np.sin(np.pi * np.linspace(0, 1, len(a0))) ** .6; a1 = a0 + (a1 - a0) * w_; cel.fill(np.vstack([np.c_[X[m], a0], np.c_[X[m], a1][::-1]]), col2, .75 * amt)
        for k in range(n):
            vy = (8 + (19 if front else 27) * r.random() ** .9) * scale ** .5 * kick ** .6; vx = r.uniform(4, 20); side = r.uniform(.05, 1.) * sg; vy *= 1 + .10 * math.sin(f * .9 + k * 2.1); T = 2 * vy / g; br = r.uniform(.14, .36) * T; o = np.array([x - r.uniform(0, 150) * scale, y + r.uniform(-4, 14) * scale * sg])
            def pos(a): a = np.asarray(a, float); return np.stack([o[0] + (vx - V) * a, o[1] - vy * a + .5 * g * a * a + side * 4.6 * scale * a * (1 + a * .012)], -1)
            if not front and k % 5 == 0 and min(br, age) > 2.5:
                a = np.linspace(0, min(br, age), 18); C = pos(a); T_ = np.gradient(C, axis=0); T_ /= np.hypot(*T_.T)[:, None] + 1e-6; N = np.c_[-T_[:, 1], T_[:, 0]]; uu = a / max(min(br, age), 1e-3); wd = r.uniform(1.6, 4.) * scale * np.interp(uu, [0, .25, .8, 1], [.5, 1., .5, .0]) + .8 * np.sin(uu * 9 + f); wd = np.maximum(wd, .3); cel.fill(np.vstack([C + N * wd[:, None], (C - N * wd[:, None])[::-1]]), '#ffffff', .95 * amt)
                if wd.max() > 3: cel.fill(np.vstack([C + N * (wd * .75)[:, None], (C + N * (wd * .05)[:, None])[::-1]]), '#a9e4ee', .8 * amt)
            m = int(r.integers(6, 11)); gap = (T * .98 - br) / m; frac = (f + r.uniform(0, gap)) % gap; jx, jy, rs = r.uniform(-9, 9, m), r.uniform(-10, 10, m), r.random(m) ** 2.2
            for j in range(m):
                a0 = br + j * gap + frac
                if a0 > T or a0 > age: continue
                p0 = pos(a0) + np.array([jx[j], jy[j]]) * scale * (a0 / T); d = np.array([vx - V, -vy + g * a0 + side * 4.6 * scale]); sp = np.hypot(*d); d = d / sp; R = (1.3 + 6.2 * rs[j]) * scale * (1 - .35 * a0 / T); ln = R * (1.1 + min(sp, 40) * .020); nn = np.array([-d[1], d[0]]); tt = np.linspace(0, 2 * pi, 17)[:-1]; c_ = np.cos(tt)
                P_ = p0 + np.outer(c_ * np.where(c_ > 0, R, ln), d) + np.outer(np.sin(tt) * R * np.where(c_ > 0, 1., .80), nn); cel.fill(P_, '#ffffff', amt)
                if R > 3.0: lo = p0 + np.array([0, R * .34]); cel.fill(np.c_[lo[0] + c_ * R * .66, lo[1] + np.sin(tt) * R * .36], '#a6def0', .9 * amt)
        return over(im, cel.layer())

    def crown(s, im, x, y, f, age, scale=1., seed=3):
        """the first touch: a ring of water opens and throws a short crown, ballistic, gone in a dozen frames"""
        if not 0 <= age < 16: return im
        cel = Cel(W, H, ss=2); r = np.random.default_rng(seed); g = 1.9; V = s.V * .35; a = age
        R = (18 + 26 * a ** .72) * scale; cx = x - V * a; th = np.linspace(0, 2 * pi, 70); k = np.clip(1 - a / 14., 0, 1); w_ = 5 * k + .6
        for q, col, al in ((1.0, '#ffffff', .95), (.72, '#b6ecf1', .6)): Rq = R * q; O = np.c_[cx + np.cos(th) * Rq * 1.25, y + np.sin(th) * Rq * .30]; I = np.c_[cx + np.cos(th) * (Rq - w_) * 1.25, y + np.sin(th) * (Rq - w_) * .30]; cel.fill([O, I[::-1]], col, al * k, evenodd=True)
        for j in range(16):
            an = 2 * pi * j / 16 + r.uniform(-.15, .15); v = r.uniform(5, 9) * scale; vy = r.uniform(9, 19) * scale ** .5; px = cx + math.cos(an) * v * a; py = y + math.sin(an) * v * a * .30 - vy * a + .5 * g * a * a
            if py > y + math.sin(an) * v * a * .30 + 4: continue
            b = max(0, a - 2.2); qx = cx + math.cos(an) * v * b; qy = y + math.sin(an) * v * b * .30 - vy * b + .5 * g * b * b; R1 = r.uniform(2.5, 6) * scale * (1 - a / 22.); d = np.array([px - qx, py - qy]); d /= np.hypot(*d) + 1e-6; nn = np.array([-d[1], d[0]]); tt = np.linspace(-pi / 2, pi / 2, 9); cap = np.array([px, py]) + np.outer(np.cos(tt) * R1, d) + np.outer(np.sin(tt) * R1, nn); cel.fill(np.vstack([cap, [[qx, qy]]]), '#ffffff', .97)
        return over(im, cel.layer())

    def near(s, im, x, y, f, n=7, seed=21):
        """big drops between us and him, out of focus, same parabolas"""
        cel = Cel(W, H, ss=1); r = np.random.default_rng(seed); g = 1.75
        for k in range(n):
            vy, vx = r.uniform(20, 33), r.uniform(8, 20); T = 2 * vy / g * 1.25; a = (f * 1. + r.uniform(0, T)) % T; p = np.array([x + (vx - s.V) * a, y + 20 - vy * a + .5 * g * a * a + 9.5 * a]); R = r.uniform(6, 15); d = np.array([vx - s.V, -vy + g * a]); sp = np.hypot(*d); d /= sp; nn = np.array([-d[1], d[0]]); tt = np.linspace(0, 2 * pi, 22)[:-1]; c_ = np.cos(tt)
            cel.fill(p + np.outer(c_ * R * (1.25 + sp * .012), d) + np.outer(np.sin(tt) * R, nn), '#ffffff', .85); cel.fill(p + np.outer(c_ * R * .7, d) + np.outer(np.sin(tt) * R * .6 + R * .22, nn), '#8fd6e8', .8)
        return over(im, blur(cel.layer(), 4.2))

    def bow(s, im, x, y, f, w=64., hh=26., amt=1.):
        """the little wall of water a palm pushes: covers what is under the surface"""
        cel = Cel(W, H, ss=2); u = np.linspace(-1, 1, 40); b = 1 + .10 * np.sin(u * 7 + f * 1.1); top = y - hh * np.clip(1 - np.abs(u) ** 2.2, 0, 1) * b * (1 + .35 * u) + 2 * np.sin(u * 11 - f * .9); X = x + u * w * np.where(u < 0, 1.9, 1.); bot = y + 20 * np.sqrt(np.clip(1 - u * u, 0, 1))
        cel.fill(np.vstack([np.c_[X, top], np.c_[X, bot][::-1]]), '#4fc0da', .92 * amt); cel.fill(np.vstack([np.c_[X, top], np.c_[X, top + (bot - top) * (.30 + .12 * np.sin(u * 9 + f))][::-1]]), '#ffffff', .96 * amt); cel.fill(np.vstack([np.c_[X, top + (bot - top) * .45][8:34], np.c_[X, top + (bot - top) * (.62 + .1 * np.sin(u * 6 - f))][8:34][::-1]]), '#bff0f2', .9 * amt)
        return over(im, cel.layer())

    def graze(s, im, x0, x1, y, f, n=26, seed=9, amt=1.):
        """where the cloud's air pushes the water: short white cats' paws, streaming back"""
        cel = Cel(W, H, ss=2); r = np.random.default_rng(seed)
        for k in range(n):
            L = x1 - x0; p = ((r.random() * L - f * s.V * .55) % L); X = x0 + p; q = p / L; yy = y + r.normal(0, 14) + 10 * math.sin(q * 5); ln = r.uniform(26, 90) * (.5 + .6 * math.sin(pi * q)); th = r.uniform(1.2, 4.2) * math.sin(pi * q); t = np.linspace(0, 1, 12); C = np.c_[X - ln * t, yy + 3 * np.sin(t * 5 + k)]; wd = th * np.sin(pi * t) ** .7; cel.fill(np.vstack([C - np.c_[0 * t, wd], (C + np.c_[0 * t, wd])[::-1]]), '#ffffff' if k % 3 else '#b9ecf0', .85 * amt)
        return over(im, cel.layer())

    def streak(s, im, f, k=.55):
        """shutter on the near water only"""
        a = cv2.blur(im, (int(s.V * k * .6) | 1, 1)); b = cv2.blur(im, (int(s.V * k * 1.6) | 1, 1)); t = np.clip((s.y - s.hz) / (s.yc - s.hz), 0, 1.6)[..., None]; w1 = smooth(.15, .6, t); w2 = smooth(.6, 1.3, t); o = im + (a - im) * w1; return o + (b - o) * w2


_F = {}


def fbm_shift(dx):
    if 'n' not in _F: a = fbm(H, W + 1024, (60, 22, 9), (1, .8, .5), 61, 3.0); _F['n'] = np.hstack([a, a[:, ::-1]]).astype(f32)
    n = _F['n']; w = n.shape[1]; x = (np.arange(W) + int(dx)) % w; return n[:, x]
