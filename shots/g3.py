"""g3 AWAKENING (80).  0 the thousand beams, 4-9 he closes his eyes, 10 20 30 40 four bells in the dark (the boy on the cloud), 44 the eyes open silver, the gold runs out of the hair root to tip by 50, then the picture HOLDS."""
import numpy as np, cv2, math, os
from db import W, H, ki, comp, kit
from db.paint import rgb, f32, smooth, fbm, grid, ramp
from db.ink import Cel, spl
from db.comp import over, blur, radial, screen
from chars import goku_awake as AW, goku_heroes as GX
A = np.array
C0 = (960, 486, 208); BELL = (10, 20, 30, 40); VER = 2


class Shot:
    id = 'g3'; n = 80
    finish = dict(bloom=.30, vig=.22, halation='#cfdcff')
    EVENTS = [(0, 'silence', dict(note='no sound effects in space: breath and heart only', to=79)), (2, 'heart', dict(size=.4, note='inside the body')), (4, 'breath_in', dict(size=.3, note='slow, he closes his eyes')), (9, 'eyes_close', dict(size=.1, note='the world goes out'))] + \
             [(q, 'bell', dict(star=k + 1, size=.8, note='star bell %d: the boy on the cloud for an instant' % (k + 1))) for k, q in enumerate(BELL)] + [(26, 'heart', dict(size=.2)), (36, 'heart', dict(size=.15)), (44, 'eyes_open', dict(size=.6, note='silver; glass breath, not a hit')), (44, 'hair_drain', dict(to=50, size=.3, note='gold runs out, root to tip: a glass glissando')),
              (52, 'breath_out', dict(size=.35)), (54, 'static', dict(to=79, size=.08, note='blue static, felt more than heard')), (54, 'breath', dict(to=76, size=.25, note='one breath')), (63, 'eye_shift', dict(size=.05)), (66, 'beam_pass', dict(to=68, size=.12, note='a beam parts one hair: almost nothing')), (70, 'breath_in', dict(size=.25))]

    def setup(self):
        self.x, self.y = grid(); self.bells(); p = 'cache/g3_hold_v%d.npz' % (VER + 1)
        if not os.path.exists(p):
            L = AW.layers(1., 1., 1., f=52); fx = {'face%d' % i: AW.GX.ui_face(L['G'], dict(L['X'], gx=g), AW.light(AW.SILV), L['hr'], 3, 1.).astype(np.float16) for i, g in enumerate((-.004, .008))}; os.makedirs('cache', exist_ok=True); np.savez(p, **{k: L[k].astype(np.float16) for k in ('back', 'body', 'face', 'front')}, **fx)
        self.hold_p = p; self.hold = None; self.HG = None; r = np.random.default_rng(5); st = np.zeros((H, W), f32)
        for _ in range(260): st[int(r.integers(0, H)), int(r.integers(0, W))] = r.random() ** 3
        st = cv2.GaussianBlur(st, (0, 0), .9); v = np.clip((self.x / W) * .9 + (1 - self.y / H) * .2, 0, 1); bg = ramp(v, [(0, '#04050c'), (.4, '#0a0d20'), (.75, '#171a3c'), (1, '#2a2460')]); d = np.sqrt((self.x - 420) ** 2 + (self.y - 3150) ** 2) - 2240
        bg = bg + np.exp(-np.abs(np.minimum(d, 0)) / 46)[..., None] * rgb('#2f6cc8') * .55 * (d < 0)[..., None] + np.exp(-np.maximum(d, 0) / 70)[..., None] * rgb('#3f86ff') * .40 * (d >= 0)[..., None]; cl = np.clip(fbm(H, W, (120, 40, 14), (1, .5, .3), 9, aspect=4.) * .5 + .2, 0, 1) * (d < -30); bg = bg + (cl * np.clip((-d - 30) / 160, 0, 1))[..., None] * rgb('#4a6ab0') * .30
        self.bg0 = (bg + st[..., None] * 3.).astype(np.float16); self.bm = [(r.normal(176, 4.5), r.uniform(-250, 1250), r.uniform(360, 1200), r.uniform(0, 1), r.uniform(3, 20), r.uniform(.6, 1.4)) for _ in range(46)]

    # ------------------------------------------------------------------ the four instants
    def bells(self):
        p = 'cache/g3_bells_v%d.npy' % VER
        if not os.path.exists(p): self.make_bells(p)
        self.B = np.load(p, mmap_mode='r')

    def make_bells(self, path):
        import importlib
        from db import ball
        from chars.kid import P, mixp
        S = importlib.import_module('shots.a2').Shot(); S.setup(); out = np.zeros((4, 4, 480, 480, 3), np.float16); R = 240; cx, cy = C0[0], C0[1]
        crop = lambda im: im[cy - R:cy + R, cx - R:cx + R]
        for j in range(4):
            im = S.frame(60 + 2 * j); M = cv2.getAffineTransform(np.float32([[840 - 330, 655 - 330], [840 + 330, 655 - 330], [840 - 330, 655 + 330]]), np.float32([[0, 0], [480, 0], [0, 480]])); out[0, j] = cv2.warpAffine(im, M, (480, 480), flags=cv2.INTER_AREA)
        sky, z, front = S.D.frame(60, S.D.cam(zoom=1.35, dx=110, dy=0, dz=0), split=600.); sky = sky.astype(f32)
        for k, (S_, by, yaw, pit, ex, sx, sy) in enumerate(((46, 590, .55, .0, 'calm', 60, -60), (104, 760, .95, -.10, 'smile', 160, -40), (150, 905, -.12, .02, 'blink', -90, -120)), 1):
            for j in range(4):
                f = 60 + 2 * j; im = np.roll(np.roll(blur(sky, 3 + 3 * k), sx, 1), sy, 0); bx = cx + (-10, -70, 4)[k - 1]; bo = 1.5 * math.sin(f * .16); nr = dict(yaw=.10 if k < 3 else 1.2, pitch=.20, roll=.03)
                if k < 3:
                    b, fr = S.N.draw(bx, by + bo, S_, f, pal='dawn', light=(.42, .46, -.30), speed=.12, tone=(86, 60, 22), **nr); im = over(im, b)
                pose = P('sit_hold', armR=(-.12, -.34, .64), armL=(-.12, -.34, .64), hed=(0, .02, 0)); L = S.K.draw(bx, by + bo, S_, pose, yaw + (.4 if k < 3 else 0), pit, 0, expr=ex if not (k == 3 and j >= 3) else 'smile', pal='dawn', light=(.50, .45, -.35 if k < 3 else .1), f=f, rim=dict(dx=-3.0, dy=2.0, col='#fff0bd', amt=1., soft=.5), wind=(-1, -.1, .25)); im = over(im, L)
                hx, hy = S.K.info['handR']
                if k < 3: im = ball.draw(im, hx + .1 * S_, hy - .25 * S_, .46 * S_, light=(.3, .3, -.8), glass=.9, body=.9, line=.8)
                im = im + radial(cx + sx, cy + sy, 420, power=1.6)[..., None] * rgb('#ffe9b0') * .25; out[k, j] = crop(im)
        np.save(path, out)

    def dark(self, f):
        """behind the lids: warm black that breathes, the beams only as slow violet stains"""
        br = .5 + .5 * math.sin((f - 10) * 2 * math.pi / 34.); img = np.zeros((H, W, 3), f32) + rgb('#050208'); img = img + radial(960, 520, 1250, 820, 1.4)[..., None] * rgb('#2a0c14') * (.38 + .22 * br)
        for k in range(5): x0 = 2100 - ((f * (3.2 + k * .8) + 431 * k) % 2400); y0 = 120 + 210 * k + 26 * math.sin(f * .05 + k); img = img + radial(x0, y0, 620, 90 + 24 * k, 1.8)[..., None] * rgb('#3a1466') * (.30 + .12 * math.sin(f * .2 + k * 2))
        return img

    def bell(self, img, f, k):
        a = f - BELL[k]; cx, cy, R = C0; op = (.62, 1., 1., 1., .96, .84, .62, .38, .18, .06)[a]; r = R * (.93, 1.0, 1.0, 1.0, 1., 1., .985, .96, .93, .90)[a]; j = min(3, a // 2); pic = self.B[k, j].astype(f32); zo = 1.0 + .010 * a; M = cv2.getRotationMatrix2D((240, 240), 0, zo); pic = cv2.warpAffine(pic, M, (480, 480), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sf = 3.5 + 2.2 * a; d = np.hypot(self.x - cx, self.y - cy); d = d * (1 + .010 * np.sin(np.arctan2(self.y - cy, self.x - cx) * 5 + k * 2)); m = smooth(r + sf * .5, r - sf, d) * op; full = np.zeros_like(img); full[cy - 240:cy + 240, cx - 240:cx + 240] = blur(pic, .4 + .5 * a) if a > 4 else pic
        img = img * (1 - m[..., None]) + full * m[..., None]; img = img + (np.exp(-np.maximum(d - r, 0) / 120.) * (d > r - 2))[..., None] * rgb('#ffb45a') * .30 * op
        ring = r + 14 + a * a * 1.2 + a * 13; w = max(1.2, 5. - .45 * a); img = img + np.exp(-((d - ring) / w) ** 2)[..., None] * rgb('#fff0c8') * (.95 * math.exp(-a / 3.)); ring2 = r + 6 + a * 34; img = img + np.exp(-((d - ring2) / (w * 1.6)) ** 2)[..., None] * rgb('#ffc37a') * (.45 * math.exp(-a / 2.2))
        if a == 0: img = img + np.exp(-((d - r) / 10.) ** 2)[..., None] * rgb('#fff6dc') * 1.2
        return img

    # ------------------------------------------------------------------ the place
    def beams(self, f, sp, al=1., front=False, t0=0.):
        """the volley, out of focus; sp = px per frame"""
        c = Cel(W, H, ss=1)
        for i, (ang, y0, L, u, ww, v) in enumerate(self.bm):            # p = the bright head, it leads to the LEFT; the tail trails behind it on the right
            if (i % 6 == 0) != front: continue
            a = math.radians(ang + 180); dv = A([math.cos(a), math.sin(a)]); nr = A([-dv[1], dv[0]]); tot = 3400; x0 = 2300 - ((u * tot + (f - t0) * sp * v) % tot); p = A([x0, y0 - (x0 - 1700) * dv[1]]); w2 = ww * (2.4 if front else 1.)
            c.fill(spl([p, p + dv * L * .10 + nr * w2, p + dv * L * .5 + nr * w2 * .5, p + dv * L, p + dv * L * .5 - nr * w2 * .5, p + dv * L * .10 - nr * w2], True, (0, 3), 6), '#ffffff', .35 + .6 * u)
        m = c.rgba()[..., 3]; b = 16 if front else 9
        return (blur(m, b)[..., None] * rgb('#9a4cff') * .85 + blur(m, b * .35)[..., None] * rgb('#dcb8ff') * .28) * al

    def static(self, img, f, hr, amt=1.):
        g = f // 2; r = np.random.default_rng(40 + g); c = Cel(W, H)
        for j in range(int(round(8 * amt))):
            b = hr.bl[int(r.integers(len(hr.bl)))]
            if b.k[0] not in 'LRTFC': continue
            i0 = int(r.integers(8, 20)); i1 = min(len(b.C) - 1, i0 + int(r.integers(6, 14))); p, q = b.C[i0] + b.N[i0] * b.hw[i0] * r.uniform(-.9, .9), b.C[i1] + b.N[i1] * b.hw[i1] * r.uniform(-1, 1); GX.bolts(c, p, q, int(r.integers(1 << 30)), 2.0 if f % 2 == 0 else 1.2, '#e6f5ff', .20, 5, 2)
        a = c.rgba(); k = 1. if f % 2 == 0 else .55; img = img + (GX.glow_of(a, 9, '#3f8cff', 1.3) + GX.glow_of(a, 2.5, '#7cc0ff', .9)) * k; return over(img, a * k)

    def compose(self, f, L, bg, glow=1., vio=0., warp=None):
        img = bg; al_ = L['front'][..., 3]
        for k in ('back', 'body', 'face', 'front'): img = over(img, L[k])
        img, al = AW.rims(img, L, 1., vio); return (AW.eyes(img, L, glow) if glow > 0 else img), al

    def frame(self, f):
        bg0 = self.bg0.astype(f32)
        if f < 10:
            u = f; op = 1. if f < 4 else (1., .85, .55, .28, .08, 0.)[f - 4]; d = (f // 2) * 2
            L = AW.layers(0., 0., op, expr='determined', f=d, wind=(-.16, .30), gx=.040 if f < 4 else .030 - .006 * (f - 4), relax=float(comp.ease(np.clip((f - 3) / 6., 0, 1))))
            img, al = self.compose(f, L, bg0 * .9 + self.beams(f, 450, 1.), 0., .55); img = img + self.beams(f, 720, .5, True); k = float(np.clip((f - 4) / 6., 0, 1)); return (img * (1 - .72 * k * k)).astype(f32)
        if f < 44:
            img = self.dark(f)
            for k in range(4):
                if 0 <= f - BELL[k] < 10: img = self.bell(img, f, k)
            if f == 43: img = img + (np.exp(-np.abs(self.y - 500) / 5.) * np.exp(-((self.x - 900) / 420.) ** 2))[..., None] * rgb('#a9c6ff') * .8
            return img.astype(f32)
        if f < 54:
            a = f - 44; d = f; op = (.34, .72, 1.06)[a] if a < 3 else 1.; sv = float(np.clip(a / 6., 0, 1)); sh = float(comp.ease(np.clip((d - 44) / 8., 0, 1))); sh = sh + .06 * math.sin(math.pi * np.clip((d - 49) / 5., 0, 1)); sv = max(.03, sv) if f < 50 else 1.
            L = AW.layers(min(sh, 1.0) if f < 53 else 1., sv, min(1., op), f=52, wind=(-.10, .10 + .28 * (1 - sh)))
            img, al = self.compose(f, L, bg0 + self.beams(f, 450 * .12, 1., t0=-40 - 60. * (1 - math.exp(-a / 3.))), (.9, 1.5, 1.75, 1.45, 1.25, 1.15, 1.08, 1.04, 1., 1.)[a]); img = self.static(img, f, L['hr'], .4 + .6 * min(1, a / 5.))
            if a < 4: w = (.9, .5, .25, .1)[a]; img = img + (radial(860, 540, 1300, 760, 1.6))[..., None] * rgb('#cfe0ff') * .35 * w
            if f >= 49:                                            # one hair stays loose
                c = Cel(W, H); P = AW.strand(L['G'], f); m = max(6, int(len(P) * (f - 48) / 6.)); AW.paint_strand(c, np.stack([np.interp(np.linspace(0, m - 1, 44), np.arange(m), P[:m, j]) for j in (0, 1)], -1), 17. * (f - 47) / 7.); img = over(img, c.rgba())
            return self.push(img, f).astype(f32)
        # ---- the held close-up
        if self.hold is None:
            z = np.load(self.hold_p); self.hold = {k: z[k] for k in ('back', 'body', 'face', 'front', 'face0', 'face1')}; L0 = AW.head(); X = AW.GF.getx('ui'); X = dict(X, gx=-.014, gy=.004); G = AW.GHead.of(L0, 0, 0); self.hold.update(G=G, X=X); self.hr = AW.Hair(G, 'ui', AW.light(AW.SILV), f=52, wind=(-.10, .10), seed=3, painterly=True, kids=2)
        L = {k: (v.astype(f32) if isinstance(v, np.ndarray) and not (k in ('face0', 'face1') and f < 63) else v) for k, v in self.hold.items()}; G = L['G']
        if f >= 63: L['face'] = L['face0'] if f == 63 else L['face1']; L['X'] = dict(L['X'], gx=-.004 if f == 63 else .008)          # the eyes go to the beam before it is there
        br = math.sin(math.pi * float(np.clip((f - 54) / 22., 0, 1))) ** 1.4          # one breath
        L['body'] = cv2.warpAffine(L['body'], np.float32([[1 + .0055 * br, 0, -.0055 * br * 900], [0, 1, -6.0 * br]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)      # breath: shoulders only
        hit = f - 66; vio = (.0, .9, .6, .3, .12)[hit + 1] if -1 <= hit < 4 and hit >= 0 else 0.
        vio = {0: 1., 1: .75, 2: .4, 3: .18, 4: .06}.get(hit, 0.); bg = bg0 + self.beams(f, 54, 1., t0=-40 - 60. * (1 - math.exp(-(f - 44) / 3.))); zc = AW.strand(G, 66)[29]; P0, P1 = zc + A([640., -20.]), zc + A([-1800., 58.])
        if 0 <= hit < 5: bm = self.needle(P0, P1, hit); bg = over(bg + GX.glow_of(bm, 10, '#8a2cff', 1.1), bm)
        img, al = self.compose(f, L, bg, 1.0 + .05 * math.sin(f * .31) + .25 * vio, vio); img = self.static(img, f, self.hr, .55 + .25 * (hit in (0, 1, 2)))
        img = self.hair(img, f, G, hit, P0, P1); return self.push(img, f).astype(f32)

    def needle(self, P0, P1, hit):
        """a hair-thin beam"""
        c = Cel(W, H); d = P1 - P0; h_ = (.55, 1.05, 1.3, 1.6, 2.)[hit]; t = (0., 0., .12, .45, .8)[hit]; a, b = P0 + d * t, P0 + d * min(1., h_); n = A([-d[1], d[0]]) / np.hypot(*d); w = (5., 4.2, 3., 2., 1.1)[hit]
        for col, k in (('#8a2cff', 1.9), ('#dcb0ff', 1.0), ('#ffffff', .45)): c.fill(np.vstack([a, a + (b - a) * .04 + n * w * k, b, a + (b - a) * .04 - n * w * k]), col)
        return c.rgba()

    def hair(self, img, f, G, hit, P0, P1):
        """the loose strand drifts; at 66 the beam goes through it: the end floats off with a hot tip"""
        g = f; push = 0. if hit < 0 else .30 * (1 - math.exp(-hit / 1.6)) * math.exp(-hit / 7.); P = AW.strand(G, g, push=push); c = Cel(W, H)
        u = (P0[0] - P[:, 0]) / (P0[0] - P1[0]); yb = P0[1] + (P1[1] - P0[1]) * u; i = 29
        if hit < 0: AW.paint_strand(c, P)
        else:
            Q = P[:i + 1]; AW.paint_strand(c, np.stack([np.interp(np.linspace(0, i, 44), np.arange(i + 1), Q[:, j]) for j in (0, 1)], -1)); Z = AW.strand(G, 66)[i:]; cen = Z.mean(0); th = .10 + .030 * hit; ca, sa = math.cos(th * (hit > 0)), math.sin(th * (hit > 0)); W_ = (Z - cen) @ A([[ca, sa], [-sa, ca]]) + cen + A([-1.2 * hit, 2.6 * hit - .03 * hit * hit]); AW.paint_strand(c, np.stack([np.interp(np.linspace(0, len(W_) - 1, 44), np.arange(len(W_)), W_[:, j]) for j in (0, 1)], -1), 10.)
            hot = math.exp(-hit / 4.); 
            for e in (Q[-1], W_[0]): img = img + radial(e[0], e[1], 44, power=2.2)[..., None] * rgb('#c784ff') * .9 * hot + radial(e[0], e[1], 9, power=1.5)[..., None] * rgb('#ffffff') * 1.1 * hot
            r = np.random.default_rng(6)
            for j in range(7): a = r.uniform(-.9, .9) + math.pi; q = Q[-1] + A([math.cos(a), math.sin(a)]) * (6 + 5.0 * hit * r.uniform(.4, 1.2)); c.fill(np.c_[q[0] + np.cos(np.linspace(0, 6.28, 9)) * 2.2 * hot, q[1] + np.sin(np.linspace(0, 6.28, 9)) * 2.2 * hot], '#efdcff', .9 * hot)
        return over(img, c.rgba())

    def push(self, img, f):
        z = 1 + .016 * float(np.clip((f - 44) / 36., 0, 1)); return cv2.warpAffine(img, np.float32([[z, 0, (1 - z) * 840], [0, z, (1 - z) * 560]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
