"""g5 THE MOON FALLS: the white ball, bigger than mountains, over the whole sea.  F = Fall(); F.frame(u, t, view=(zoom, fx, fy), press=0..1)
u 0 (high) .. 1 (its limb near the horizon).  view: zoom about the world point (fx, fy) (world = the 1920x1080 picture at zoom 1), crisp at any zoom <= 2.2.
press: the weight arrives: the sea goes flat, the cloud collar is pushed down and out.  No characters.  F.at(x, y, view) -> screen px of a world point."""
import numpy as np, cv2, math
from . import W, H as HH, comp
from .paint import rgb, f32, fbm, smooth, mix, grid, ramp
from . import moon as M

HZ = 824.


class Fall:
    def __init__(self, cx=1010., r=780., seed=5, zmax=2.0):
        self.cx, self.r = cx, r; rr = np.random.default_rng(seed); self.col = []
        for k in range(26):                                                 # a collar of torn banks round the ball, seen a little from below
            a = k / 26 * 2 * math.pi + rr.normal(0, .07); front = math.sin(a) > 0; X = math.cos(a) * r * rr.uniform(1.02, 1.22); Y = math.sin(a) * r * .17 + rr.normal(0, 14); wd, ht = r * rr.uniform(.34, .62), r * rr.uniform(.06, .12) * (1.25 if front else .9); ss = 1.6
            M_ = np.zeros((int((ht * 3.2 + 90) * ss), int((wd * 1.5 + 260) * ss)), f32); base = (ht * 2.2 + 40); M._cloud(M_, (130 + wd * .25) * ss, int(base * ss), wd * ss, ht * ss, rr, 1.3)
            n = fbm(*M_.shape, (34, 13, 6), (1, .5, .25), int(rr.integers(1 << 20)), aspect=2.2); s = cv2.GaussianBlur(M_, (0, 0), 3.4) + .09 * n; al = smooth(.46, .56, s); b8 = cv2.morphologyEx((al > .5).astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))); al = np.maximum(al, smooth(.36, .64, cv2.GaussianBlur(b8.astype(f32), (0, 0), 2.6)))
            dt = cv2.GaussianBlur(np.maximum(0, cv2.distanceTransform((al > .5).astype(np.uint8), cv2.DIST_L2, 5) - .5 + al), (0, 0), 1.2); T = np.exp(-np.maximum(dt, 0) / (7.6 * np.clip(1 + .7 * n, .2, 2.4) + .6)); self.col.append((front, X - wd * .5 - 130, Y - base, al.astype(f32), T.astype(f32), math.copysign(1, math.cos(a)), math.sin(a), ss))
        self.col.sort(key=lambda q: q[6]); self.x, self.y = grid()
        storm = fbm(HH, W, (120, 46, 17, 7), (1, .6, .35, .2), seed + 1, aspect=3.2); self.storm = cv2.GaussianBlur(np.where(storm > .7, 1., np.where(storm > .0, .55, .18)).astype(f32), (0, 0), 1.1)
        self.fog = fbm(HH, W, (40, 14, 5), (1, .6, .35), 3, aspect=6.).astype(f32)
        q = np.random.default_rng(seed + 8); xs = np.arange(-400, W + 401); lnd = np.zeros(len(xs))
        for (a, b, h0) in ((-380, 60, 30), (-80, 250, 24), (120, 380, 15), (330, 520, 9), (1500, 1720, 13), (1640, 2000, 27), (1380, 1520, 6), (1900, 2300, 22)):
            X0 = a
            while X0 < b: bw = q.uniform(26, 74); hh = h0 * q.uniform(.5, 1.15); e = q.uniform(5, 11); lnd = np.maximum(lnd, hh * np.clip(np.minimum((xs - X0) / e, (X0 + bw - xs) / e), 0, 1) ** .7); X0 += bw * q.uniform(.7, 1.2)
            lnd = np.maximum(lnd, 4.5 * np.clip(np.minimum((xs - a + 60) / 60, (b + 90 - xs) / 90), 0, 1))
        self.lx, self.land = xs, lnd; self.town = [(q.uniform(30, 470), q.uniform(1, 8), q.uniform(.5, 1)) for _ in range(46)] + [(q.uniform(1660, 1890), q.uniform(1, 9), q.uniform(.4, 1)) for _ in range(22)]
        self.birds = [dict(x=q.uniform(330, 1640), y=q.uniform(676, 802), s=float(2.0 + 5.2 * q.random() ** 1.8), ph=q.uniform(0, 6.28), w=q.uniform(.45, 1.25), glide=q.random() < .35, bank=q.normal(0, .34), bend=q.uniform(.25, .75), up=q.uniform(.2, 1.1), tail=q.uniform(.25, .6), v=q.uniform(.55, 1.5), fan=q.uniform(.75, 1.25)) for _ in range(22)]
        self.R0 = r * zmax; S = int(self.R0) + 4; b = M.ball(S, 2 * S - 2 - S, self.R0, spin=0., seed=4, h=S * 2, w=S * 2, tilt=.45, grain=1.25); self.S = S; y = np.arange(S * 2, dtype=f32)[:, None]; sh = smooth(-.10, .55, (y - (S - 2)) / self.R0)[..., None]      # baked once; its underside is a step darker
        b[..., :3] *= 1 - .16 * sh; self.ballc = S - 2; self.tex = b.astype(np.float16)

    def cy(self, u): return -430 + 520 * u
    def at(self, x, y, view=(1., 960., 540.)): z, fx, fy = view; return (x - fx) * z + 960, (y - fy) * z + 540

    def _cloud(self, img, c, cy, L, press, view):
        fr, X, Y, al, T, sd, sa, ss = c; z = view[0]; x0, y0 = self.at(self.cx + X + sd * press * 78, cy + self.r * (.62 + .05 * press) + Y * (1 - .10 * press) + (26 * press if fr else 8 * press), view); k = z / ss; ky = k * (1 - .20 * press); h, w = al.shape; ww, hh_ = int(w * k) + 2, int(h * ky) + 2; ix, iy = int(math.floor(x0)), int(math.floor(y0 + h * (k - ky) * .6))
        xa, xb, ya, yb = max(0, ix), min(W, ix + ww), max(0, iy), min(HH, iy + hh_)
        if xa >= xb or ya >= yb: return
        Mx = np.float32([[k, 0, x0 - ix - (xa - ix)], [0, ky, -(ya - iy)]]); fl = cv2.INTER_LINEAR; a = cv2.warpAffine(al, Mx, (xb - xa, yb - ya), flags=fl); t = cv2.warpAffine(T, Mx, (xb - xa, yb - ya), flags=fl); l = L[ya:yb, xa:xb]; k1 = np.clip(l * 1.1, 0, 1)[..., None]
        body = mix(rgb('#0b0717'), rgb('#231148'), np.clip(l * .6, 0, 1)[..., None]); mid = mix(body, rgb('#5a22b8'), k1); hot = mix(mid, rgb('#ecd8ff'), k1); t3, t2, t1 = smooth(.53, .67, t)[..., None], smooth(.29, .39, t)[..., None], smooth(.07, .13, t)[..., None]; c_ = body + (body * .3 + rgb('#1a0a3a') * k1 * .5) * t1; c_ = c_ * (1 - t2) + mid * t2; c_ = c_ * (1 - t3) + hot * t3; A = a[..., None]; img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - A) + c_ * A

    def frame(self, u=.6, t=0., view=(1., 960., 540.), press=0., lights=1., water_t=None):
        x, y, z = self.x, self.y, view[0]; cw = self.cy(u); cx, cy = self.at(self.cx, cw, view); r = self.r * z; hz = self.at(0, HZ, view)[1]; d = np.hypot(x - cx, y - cy) - r
        L = np.where(d > 0, M.halo(d, 420 * z) * 1.1 + .24 * np.exp(-np.maximum(d, 0) / (620 * z)), 1.).astype(f32); yy = (y - 540) / z + view[2]
        st = self.storm if z == 1 else cv2.warpAffine(self.storm, np.float32([[1 + (z - 1) * .6, 0, (1 - z) * .6 * view[1]], [0, 1 + (z - 1) * .6, (1 - z) * .6 * view[2]]]), (W, HH), borderMode=cv2.BORDER_REFLECT_101)
        img = ramp(yy / HH, [(0, '#0c0719'), (.5, '#150b2c'), (.76, '#2d1858')]) * (.45 + .75 * st)[..., None] + (np.clip(L - .05, 0, 1) * (.25 + .75 * st) * (d > 0))[..., None] * rgb('#5a22c8') * .75
        img = (img + M.halo_rgb(np.where(d > -3, M.halo(d, 380 * z), 0) * .55)).astype(f32)
        for c in self.col:
            if not c[0]: self._cloud(img, c, cw, L, press, view)
        k = r / self.R0; B = cv2.warpAffine(self.tex.astype(f32), np.float32([[k, 0, cx - self.S * k], [0, k, cy - self.ballc * k]]), (W, HH), flags=cv2.INTER_LINEAR if k > .7 else cv2.INTER_AREA); img = comp.over(img, B)
        for c in self.col:
            if c[0]: self._cloud(img, c, cw, L, press, view)
        hzg = np.exp(-np.maximum(hz - y, 0) / (90 * z)) * (y < hz) * np.exp(-((x - cx) / (900 * z)) ** 2); img = img + hzg[..., None] * rgb('#8a5cf0') * .34                # air between us and it
        if hz < HH - 4: M.sea(img, hz, cx, cy, r, t=(t / 24. if water_t is None else water_t), seed=5, calm=1.35 - 1.12 * press, f=1500 * z, cam=2.6, amt=.80, deep=('#2a1358', '#100828', '#05030c')); self._dish(img, press, t, hz, cx, z)
        lx = (np.arange(W) - 960) / z + view[1]; ld = np.interp(lx, self.lx, self.land) * z; m = smooth(-.6, .8, y - (hz - ld[None])) * (y < hz + 2 + 2 * np.sin(x * .05)); lc = np.where(((hz - y) > ld[None] * .70)[..., None], rgb('#b79af0'), rgb('#1a0c3a')); img = img * (1 - m[..., None]) + lc * m[..., None]
        if lights > 0:
            acc = np.zeros((HH, W), f32)
            for i, (a, b, s) in enumerate(self.town):
                X_, Y_ = self.at(a, HZ - b, view); cv2.circle(acc, (int(X_ * 16), int(Y_ * 16)), int(1.05 * z * 16), float(s * np.clip(lights * 1.4 - .4 * ((i * .37) % 1), 0, 1)), -1, cv2.LINE_AA, 4)
            img = img + acc[..., None] * rgb('#ffcf87') * 1.1 + cv2.GaussianBlur(acc, (0, 0), 4)[..., None] * rgb('#ff9a3c') * .8
        for b in self.birds:                                                  # gulls leaving: each its own size, bank, beat, bend of wing; some glide
            dr = math.copysign(1, b['x'] - self.cx); X_ = b['x'] + dr * (1.5 * t + .05 * t * t) * b['v']; Y_ = b['y'] + 9 * math.sin(b['ph'] + t * .05) - b['up'] * t * .4; X_, Y_ = self.at(X_, Y_, view); S = b['s'] * z; fl = (.25 + .12 * math.sin(t * .2 + b['ph'])) if b['glide'] else math.sin(t * b['w'] + b['ph']) * (1.0 if math.sin(t * b['w'] + b['ph']) > 0 else .6); P = []
            ca, sa = math.cos(b['bank'] + .12 * dr), math.sin(b['bank'] + .12 * dr); rot = lambda u, v: (X_ + u * ca - v * sa, Y_ + u * sa + v * ca)
            for sd in (-1, 1):
                k = b['fan'] if sd > 0 else 1. / b['fan']; el = (sd * .95 * S * k, -(.55 * fl + .18) * S); tp = (sd * 2.05 * S * k, -(1.0 * fl - b['bend'] * .5) * S); th = max(.35, .13 * S); cv2.fillPoly(img, [np.round(np.array([rot(0, -th), rot(el[0], el[1] - th * .9), rot(*tp), rot(el[0] * .96, el[1] + th * .9), rot(0, th * 1.2)]) * 16).astype(np.int32)], (.03, .02, .07), cv2.LINE_AA, 4)
            cv2.fillPoly(img, [np.round(np.array([rot(dr * .62 * S, -.10 * S), rot(dr * .1 * S, -.20 * S), rot(-dr * (.45 + b['tail']) * S, -.04 * S), rot(-dr * (.40 + b['tail']) * S, .12 * S), rot(dr * .1 * S, .18 * S)]) * 16).astype(np.int32)], (.03, .02, .07), cv2.LINE_AA, 4)
        return img.astype(f32)

    def _dish(self, img, u, t, hz, px, z):
        """pressure on the water: rings run out from under the ball and die; a seam of spray stands at the horizon"""
        x, y = self.x, self.y; k = np.clip((y - hz) / ((HH - HZ) * z), 0, 1.6); Z = 1. / (k + .045); X = (x - px) / (1500 * z) * Z * 5.2; D = np.hypot(X, (Z - 5.4) * 1.35); n = self.fog
        for q in range(4):
            v = (t / 46. + q / 4.); R = .55 + (v % 1.) * 3.1; th = .055 + .05 * R; e = D - R + .05 * n; w = smooth(th, th * .6, np.abs(e)) * (1 - smooth(2.2, 3.5, R)) * smooth(.0, .25, v % 1.) * (y > hz + 3) * u; f = smooth(-.2, .7, n + .4); img[:] = img * (1 - (.55 * w)[..., None]) + mix(rgb('#1d0c44'), rgb('#e4d2ff'), (f * smooth(th * .5, -th * .2, e))[..., None]) * (.55 * w)[..., None]
        fog = np.exp(-((y - hz - 1) / (10 * z)) ** 2) * np.exp(-((x - px) / (520 * z)) ** 2) * (.55 + .45 * np.tanh(n)) * u; img[:] = img + fog[..., None] * rgb('#dcc6ff') * .55
