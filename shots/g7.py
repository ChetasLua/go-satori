"""g7 KA ME HA ME (80).  Low angle.  KA 0 hands forward and back to the hip; ME 20 the cup: blue pearl, water climbs to it; 30 his face; 34 a torn white glove lands on his shoulder and Vegeta's bruised profile leans in: he is standing; HA 40 side by side, gold and blue; ME 60 the rock under them sinks.
Each syllable hits on ones with a small push.  Black hair, first dawn behind, the white disc coming down."""
import numpy as np, cv2, math, os
from db import W, H, comp, kit, land, ki, moon, hero3, glove as GV, hands as HD, phys_rock as PR, phys_water as PW
from db.hero3 import Fig, P, mix_pose
from db.cam import Camera3D
from db.paint import rgb, f32, smooth, fbm, grid, ramp
from db.ink import Cel, spl
from db.comp import over, blur, radial, ease
from chars import goku, goku_hands as GK, goku_light as GL
A = np.array
VER = 2; SYL = (0, 20, 40, 60); Z0 = 34.06
GB_ = dict(heads=6.35, neckz=.834, shz=.804)          # low lens: the head sits down between the shoulders
DAWN = dict(base='dawn', mul=(.80, .70, .84), shmul=(.46, .38, .62), litmul=(1.0, .86, .84), add='#080514', ink='#140c1c')
ROCKP = dict(lit='#b98288', mid='#775675', dark='#3a2b4e', rim='#f0c2b6', ink='#170e20', fresh='#86607c', pit='#150d1c', bounce='#556aa4', rim2='#96b4e6')
RIMW = dict(dir=(.75, .60, -.25), col='#ffe6e0', thr=.46, amt=.8)


class Shot:
    id = 'g7'; n = 80
    finish = dict(bloom=.28, vig=.18)
    EVENTS = [(0, 'syllable', dict(text='か', size=.7)), (0, 'stomp', dict(size=.5)), (3, 'cloth', dict(size=.4, note='arms draw back to the hip')), (12, 'ki_pearl', dict(size=.2)), (20, 'syllable', dict(text='め', size=.75)), (20, 'ki_ignite', dict(size=.6, note='blue pearl')), (21, 'water_pull', dict(to=79, size=.45, note='threads of sea climb to the light')), (22, 'ki_hum', dict(to=79, size=.5, note='rises to the end')),
              (30, 'cut', dict(note='his face, blue from below')), (32, 'silence', dict(to=34, note='two frames of nothing')), (34, 'glove_grip', dict(size=.8, note='a torn glove lands on his shoulder')), (36, 'breath', dict(size=.35, note='Vegeta: one short breath through the nose, close')), (38, 'breath', dict(size=.15, note='Goku lets his own go')), (40, 'syllable', dict(text='は', size=.9)), (40, 'aura_ignite', dict(size=.8, note='gold beside blue')), (41, 'rock_crack', dict(size=.6)), (46, 'ki_charge', dict(to=79, size=.6, note='second voice, gold')), (60, 'syllable', dict(text='め', size=1.)), (60, 'rock_sink', dict(size=.9, note='the ground goes down under them')), (61, 'debris_up', dict(to=79, size=.4)), (72, 'ki_peak', dict(to=79, size=.9))]

    CAMS = dict(a=((1.55, -25.15, 34.28), (.16, -22.0, 35.10), 1640.), b=((3.00, -25.40, 34.26), (.10, -21.95, 35.12), 1880.), c=((2.45, -24.95, 34.20), (.08, -21.95, 35.16), 1700.))
    MOON = dict(a=(1470, 130, 400), b=(1290, 60, 560), c=(1290, -20, 760))

    def cam(self, k): e, t, f = self.CAMS[k]; return Camera3D(e, t, focal=f)

    def setup(self):
        self.x, self.y = grid(); self.bg = {}; os.makedirs('cache', exist_ok=True)
        for k in 'abc':
            p = 'cache/g7_%s_v%d.npy' % (k, VER)
            if not os.path.exists(p):
                cam = self.cam(k); o = land.ground(cam, t=.5, time='first', moon=None); img = o.img.astype(f32).copy(); mx, my, mr = self.MOON[k]; B = moon.ball(mx, my, mr, spin=.3, t=.2); sky = img.copy(); moon.ball_halo(sky, mx, my, mr, .42); B2 = B.copy(); B2[..., :3] = B2[..., :3] * rgb('#f2dfea') * .92; sky = over(sky, B2); hz = smooth(H * .75, 0, self.y)[..., None] * .16; sky = sky * (1 - hz) + rgb('#7c5a8e') * hz
                m = (o.a.astype(f32) if o.a.ndim == 2 else o.a[..., 0].astype(f32))[..., None]; img = img * m + sky * (1 - m); np.save(p, np.dstack([img, m]).astype(np.float16))
            self.bg[k] = np.load(p, mmap_mode='r')
        self.deb = np.random.default_rng(21).random((90, 7))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def hipcup(F, hd, w, out=.30):
        """both hands cupped beside the near hip, clear of the body so that the drawing reads"""
        if w <= 0: return
        s = F.skel(); r_, f_ = A([math.cos(hd), math.sin(hd), 0]), A([-math.sin(hd), math.cos(hd), 0]); c = s.pel + r_ * out + f_ * -.03 + A([0, 0, .05]); ax = r_ * .55 + f_ * .75 - A([0, 0, .25])
        F._reach = dict(R=dict(p=c + A([0, 0, .085]) - f_ * .03, part='palm', palm=A([0, 0, -1.]) + f_ * .35, fingers=ax, g='cup', body=.05, w=w), L=dict(p=c - A([0, 0, .085]) + f_ * .02, part='palm', palm=A([0, 0, 1.]) + f_ * .35, fingers=ax, g='cup', body=.10, w=w))

    def push(self, img, f, c=(960, 540), base=1.0):
        """every syllable pushes the frame: up on the frame, decays in 8"""
        k = [f - q for q in SYL if f >= q][-1]; z = base + .050 * math.exp(-k / 2.6) + .0012 * k; sx, sy = kit.shake(f, 8., 5, 2.2, f - k); return cv2.warpAffine(img, np.float32([[z, 0, (1 - z) * c[0] + sx], [0, z, (1 - z) * c[1] + sy]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    def stones(self, f, f0, cam, n, front=False, amt=1., col=('#4a3560', '#241a3a', '#c79aa8')):
        """pebbles that lift off the ground and hang, each one on its own clock"""
        c = Cel(W, H); t = f - f0
        for i in range(n):
            u = self.deb[i]; 
            if (u[6] > .82) != front: continue
            s = t - 10 * u[0]
            if s < 0: continue
            ang, rad = 6.283 * u[1], .7 + 3.6 * u[2]; x, y = .1 + math.cos(ang) * rad, -22 + math.sin(ang) * rad * .8
            if front: x, y = cam.eye[0] - .9 + 1.6 * (u[1] - .5) - .5, cam.eye[1] + 1.4 + u[2]
            z = Z0 + (1 - math.exp(-s / 9.)) * (.35 + 1.7 * u[3]) * amt + .02 * math.sin(f * .25 + i); pz, zc = cam.project(A([[x, y, z]])); px, r = pz[0], (.020 + .055 * u[4] ** 2.2) * cam.f / max(zc[0], .4)
            if not (-60 < px[0] < W + 60 and -60 < px[1] < H + 60): continue
            m = 5 + int(3 * u[5]); th = np.linspace(0, 6.283, m, endpoint=False) + u[5] * 6 + f * .05 * (u[0] - .5); q = np.c_[np.cos(th), np.sin(th) * (.6 + .3 * u[3])] * r * (1 + .25 * np.sin(th * 2 + i))[:, None] + px; c.fill(q, col[1]); cc = q.mean(0); c.fill(A([cc + (v - cc) * (1 if v[1] < cc[1] else .25) for v in q]), col[0]); c.fill(A([cc + (v - cc) * (.92 if v[0] > cc[0] and v[1] < cc[1] else .2) for v in q]), col[2], .8); c.line(np.vstack([q, q[:1]]), max(.8, min(2.6, r * .16)), '#120a1a', taper=(0, 0), seed=i)
        a = c.rgba(); return np.dstack([blur(a[..., j], 5.) for j in range(4)]) if front else a

    def shadow(self, img, F, cam, z, w=.55, al=.5):
        c = Cel(W, H, ss=1); sk = F.skel()
        for s in 'LR': p, zz = cam.project(A([[*sk.an[s][:2], z]])); k = cam.f / zz[0]; a = np.linspace(0, 6.28, 30); c.fill(np.c_[p[0, 0] + np.cos(a) * w * k * .55, p[0, 1] + 5 + np.sin(a) * w * k * .07], '#170c22', al)
        a = c.rgba(); return over(img, np.dstack([blur(a[..., j], 4) for j in range(4)]))

    def ring(self, img, cam, f, f0, kind='base', r0=.5, v=.55, th=.22, xy=(.05, -22.1)):
        k = f - f0
        if k < 0 or k > 9: return img
        L = ki.ring3(cam, (xy[0], xy[1], Z0 + .03), (0, 0, 1), r0 + v * k * (1 - .03 * k), th * math.exp(-k / 4.), kind, seed=f0, gaps=.35, alpha=.45 * (1 - k / 10.)); L = L[0] if isinstance(L, tuple) else L; return img if L is None else over(img, L)

    def gap(self, before, after, a):
        """between the legs and under the arms the fire is thin: the set shows through instead of one flat gold block"""
        q = cv2.resize(np.float32(a > .4), (W // 4, H // 4), interpolation=cv2.INTER_AREA); c = cv2.morphologyEx(q, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (75, 75)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        m = cv2.GaussianBlur(c, (0, 0), 5); m = cv2.resize(np.clip((m - .35) / .55, 0, 1), (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * .74; return after * (1 - m) + before * m

    def burn(self, img, lay, f, kind, p, seed, face, h):
        """aura that stays behind the drawing: full flame, only a thin light on the body"""
        AU = ki.aura(lay, f, kind, p, seed, height=h * .82, width=.70, face=face, veil=.16, front=.10); img0 = img; img = AU.under(img) if AU.back is not None else img; a = lay[..., 3]; img = img if kind == 'blue' else self.gap(img0, img, a); e = cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))); e = cv2.GaussianBlur(e, (0, 0), 1.2)[..., None] * .88; img = img * (1 - e) + e * rgb('#0a1c5c' if kind == 'blue' else '#5c2204')          # a dark rim keeps the drawing off its own fire
        img = ki.put(img, ki.lit(lay, kind, p * .30, f, seed, face, tint=.02, core=.05)); return AU.over(img) if AU.back is not None else img

    SLAB = ((-70, 330, 150, -.16, 0), (235, 250, 96, .10, 1.2), (470, 300, 70, -.06, .6), (760, 330, 58, .05, 1.8), (1080, 300, 66, -.05, 1.0), (1330, 260, 92, -.14, .3), (1560, 300, 128, .12, 1.4), (1800, 320, 190, .18, .5))
    def slabs(self, a, gp):
        """the broken rim of the hole they sink into: tilted plates come up past their feet, blue from the orb on one face, dawn on the other"""
        c = Cel(W, H); r = np.random.default_rng(12)
        for i, (x, w, h, tl, d) in enumerate(self.SLAB):
            t = a - d; u = 0. if t < 0 else 1 - math.exp(-t / 2.4) * math.cos(t * .75); y = 1112 - h * (u * 1.0 + .035 * max(0., a)) ; n = 7; xs = np.linspace(0, w, n); jag = r.normal(0, 7, n); jag[0] = jag[-1] = 0; ct, st = math.cos(tl), math.sin(tl); R_ = lambda P: np.c_[x + P[:, 0] * ct - P[:, 1] * st, y + P[:, 0] * st + P[:, 1] * ct]
            top = np.c_[xs, jag * .5]; back = np.c_[xs[::-1] * .86 + w * .07, -h * .20 - 8 + jag[::-1] * .4]; T = R_(np.vstack([top, back])); F_ = np.vstack([R_(top), [[x + w + 40, 1200], [x - 40, 1200]]]); lt = 1 if x + w / 2 < gp[0] else -1
            c.fill(F_, ROCKP['dark']); c.clip(F_); 
            for k in range(5): q = r.uniform(0, w); c.fill(R_(A([(q, 0), (q + r.uniform(20, 70), 0), (q + r.uniform(10, 50), h * .7), (q - 16, h * .9)])), ROCKP['mid'] if k % 2 else ROCKP['pit'], .7)
            ex = w * (.80 if lt > 0 else .0); c.fill(R_(A([(ex, 0), (ex + w * .22, 0), (ex + w * (.16 if lt > 0 else .10), h * 1.4), (ex + w * .05, h * 1.4)])), ROCKP['bounce'], .75); c.unclip()
            c.fill(T, ROCKP['lit'] if i % 3 else ROCKP['fresh']); c.clip(T); q = r.uniform(.2, .6) * w; c.fill(R_(A([(q, 6), (q + w * .35, 6), (q + w * .22, -h), (q - 10, -h)])), ROCKP['mid'], .8); c.unclip(); c.line(R_(top), 2.6, ROCKP['rim'], taper=(.1, .1), press=.5, seed=i); kit.outline(c, np.vstack([T]), 3.4, ROCKP['ink'], seed=i); c.line(np.vstack([R_(top)[:1], [[x - 34, 1190]]]), 3.6, ROCKP['ink'], taper=(.05, .3), seed=i + 20); c.line(np.vstack([R_(top)[-1:], [[x + w + 34, 1190]]]), 3.6, ROCKP['ink'], taper=(.05, .3), seed=i + 30)
            for k in range(3): q = r.uniform(.1, .9) * w; c.line(R_(A([(q, 2), (q + r.normal(0, 10), h * .35), (q + r.normal(0, 20), h * .8)])), 2.0, ROCKP['ink'], taper=(.1, .7), seed=k + i)
        return c.rgba()

    def streams(self, f, f0, o, n):
        """threads of sea water pulled in from the sides of the frame: tapered, spiralling, beading at the head"""
        c = Cel(W, H); t = f - f0
        for i in range(n):
            u = self.deb[i]; per = 15 + 7 * u[0]; ph = (t + per * u[1]) / per
            if t < 0 or ph < 0: continue
            q = ph - math.floor(ph); sd = -1 if u[2] < .5 else 1; p0 = A([960 + sd * (1040 + 200 * u[3]), 820 + 230 * u[4]]); v = np.linspace(max(0, q * 1.25 - .34), min(1., q * 1.25), 22)
            if v[-1] - v[0] < .04: continue
            e = v * v * (3 - 2 * v); d = A([o[0], o[1]]) - p0; L = np.hypot(*d); nr = A([-d[1], d[0]]) / L; P = p0 + d * e[:, None] + nr * (L * .16 * np.sin(np.pi * v) * np.sin(2.2 * np.pi * v + 6.28 * u[5]))[:, None] - A([0, 1.]) * (L * .10 * np.sin(np.pi * v))[:, None]
            w = (3 + 9 * u[6]) * np.sin(np.linspace(0, np.pi, 22) ** .8 * 1.0 * np.pi ** .2) .clip(0) * (1 - .55 * v); w = np.maximum(w, .4); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]
            for col, k, al in (('#12306e', 1.35, .85), ('#4c9cf0', 1., .95), ('#bfe8ff', .5, 1.)): c.fill(np.vstack([P + N * (w * k)[:, None], (P - N * (w * k)[:, None])[::-1]]), col, al)
            c.line(P[6:18] + N[6:18] * (w[6:18, None] * .3), 1.4, '#ffffff', taper=(.4, .4), seed=i)
            for j in range(4): vv = min(1., v[-1] + .02 + .03 * j); b = P[-1] + (A([o[0], o[1]]) - P[-1]) * (.06 + .07 * j) + N[-1] * (6 * math.sin(i + j * 2.)); r_ = (4.5 - .8 * j) * (.6 + .6 * u[6]); a = np.linspace(0, 6.28, 13); c.fill(np.c_[b[0] + np.cos(a) * r_, b[1] + np.sin(a) * r_], '#2a62c0'); c.fill(np.c_[b[0] + np.cos(a) * r_ * .7, b[1] + np.sin(a) * r_ * .7], '#8ecbff'); c.fill(np.c_[b[0] - r_ * .25 + np.cos(a) * r_ * .25, b[1] - r_ * .25 + np.sin(a) * r_ * .25], '#ffffff')
        return c.rgba()

    # ------------------------------------------------------------------ 0..19  KA
    def ka(self, f):
        cam = self.cam('a'); b = self.bg['a']; img = b[..., :3].astype(f32).copy(); hd = -math.pi / 2 - .42
        d = f if f < 12 else (f // 2) * 2; t = float(ease(np.clip((d - 2) / 8., 0, 1))); ov = math.exp(-max(0, d - 10) / 2.4) * math.sin(max(0, d - 10) * .9) if d > 10 else 0.; sq = math.exp(-d / 2.0)
        fire = P('kame_fire', root=(0, .07, -.145 - .03 * sq)); wind = P('kame_wind', root=(.015, -.03, -.135 - .02 * ov)); F = Fig('goku', 'bare', body=GB_).pose(mix_pose(fire, wind, min(1., t + .05 * ov))).place(0, -22, Z0, heading=hd); F.look((6., -24., 36.4)); F.blow(-.15 - .3 * sq, 0, .10); self.hipcup(F, hd, float(ease(np.clip((d - 4) / 6., 0, 1))))
        img = self.shadow(img, F, cam, Z0); c = Cel(W, H); fr = getattr(self, '_fr', None) or PR.Fracture((.05, -22.05), rad=1.7, seed=4, tilt=.10, kick=.02, push=.05, r0=.2); self._fr = fr; fr._cracks(c, lambda Q, z=0.: cam.project(np.c_[Q + fr.c, np.full(len(Q), z + Z0)])[0], (f + .4) / 24., ROCKP, 2.2, .10, None); img = over(img, c.rgba())
        img = self.ring(img, cam, f, 0); img = over(img, self.stones(f, 0, cam, 50, amt=.35))
        L = F.draw(cam, d, light=(.65, .6, .35), pal=DAWN, rim=RIMW, expr=dict(expr='determined', dmg=.7, hair='base', wind=(-.10, .06))); hp = F.info['palms']; px = (A(hp['L'], float).reshape(-1)[:2] + A(hp['R'], float).reshape(-1)[:2]) / 2
        if f >= 11: r = 3.5 + 9 * float(np.clip((f - 11) / 8., 0, 1)) + 1.5 * math.sin(f * 1.7); L = ki.light_on(L, px[0], px[1], r * 1.5, 'blue', 4., .45)
        img = over(img, L)
        if f >= 11: lay, li = ki.orb(px[0], px[1], r, f, 'blue'); img = ki.put(img + li * .9, lay)
        img = over(img, self.stones(f, 2, cam, 90, True, .5)); return self.push(img, f, (960, 620))

    # ------------------------------------------------------------------ 20..29  ME, the cup
    def me(self, f):
        a = f - 20; img = self.plate(f); g = float(ease(np.clip(a / 9., 0, 1))); trem = .6 + 1.2 * g; r = 10 + 8.2 * a + .7 * math.sin(f * 1.9)
        hs = GK.kame(HD.Rz(math.radians(-72)) @ HD.Ry(math.radians(64)) @ HD.Rx(math.radians(8)), open=.70 + .30 * g, tremble=trem, seed=f); cam = HD.Cam(690, 600, 330, 5.6); cen = cam.px(GK.kame_centre(hs)); cx, cy = float(cen[0]) + 26, float(cen[1])
        L = dict(GL.light('dawn', side=1)); L.update(skin=('#d9a892', '#84586e', '#42284e', '#ffe2d4'), ldir=(.6, .7, .3), line='#170c1c', under=('#2b3f7a', '#172250', '#6f8ccc'), rim=[dict(d=(-6, 4), col='#ffe0d8', amt=.9)])
        img = img + radial(cx, cy, 1000, 760, 1.8)[..., None] * rgb('#1c5cff') * (.30 + .35 * g); ch = ki.charge(cx, cy, r, f, 'blue', amount=.5 + .6 * g, reach=10., water=True); img = ki.put(img, ch)
        for k, h in enumerate(sorted(hs, key=lambda h: float(h.world(A([0, .55, 0]))[2]))):
            _, lay = GK.paint([h], cam, L, w=5.4, seed=5 + k, res=620, hat=.5); lay = ki.light_on(lay, cx, cy, r * 1.15, 'blue', 4.6, 1.); img = over(img, lay)
            if k == 0: o, li = ki.orb(cx, cy, r, f, 'blue', power=.7 + .5 * g); img = ki.put(img + li, o)
        img = img + radial(cx, cy, r * 2.0 + 30, power=2.0)[..., None] * rgb('#9fd6ff') * .55
        if a < 5: lay = ki.ring(cx, cy, 60 + 120 * a, 26 * math.exp(-a / 3.), f, 'blue', asp=.9, gaps=.2, alpha=.9 - .16 * a); img = over(img, lay)
        return self.push(img, f, (cx, cy))

    def plate(self, f):
        if not hasattr(self, '_pl'):
            b = self.bg['a'][..., :3].astype(f32); z = 1.9; img = cv2.warpAffine(b, np.float32([[z, 0, -500], [0, z, -700]]), (W, H), borderMode=cv2.BORDER_REFLECT_101); img = blur(img, 16); c = Cel(W, H)          # his belt and trousers, close, out of focus on the left
            c.fill(spl([(-60, -60), (560, -60), (600, 200), (520, 420), (470, 560), (-60, 620)], True, (0, 1, 5), 8), '#b57d72'); c.fill(spl([(-60, -60), (250, -60), (300, 260), (240, 560), (-60, 600)], True, (0, 1, 4), 8), '#70475e'); c.fill(spl([(-60, 560), (480, 520), (540, 650), (-60, 730)], True, (0, 1, 2, 3), 8), '#2b3f7a'); c.fill(spl([(-60, 560), (260, 545), (300, 690), (-60, 730)], True, (0, 1, 2, 3), 8), '#1a2650')
            c.fill(spl([(-60, 700), (545, 645), (660, 860), (700, 1140), (-60, 1140)], True, (0, 1, 3, 4), 8), '#c26a3c'); c.fill(spl([(-60, 700), (300, 670), (330, 860), (260, 1140), (-60, 1140)], True, (0, 1, 3, 4), 8), '#7a3a36')
            for q in ((330, 700, 420, 1140), (440, 690, 560, 1100), (200, 720, 180, 1100)): c.line(spl([(q[0], q[1]), ((q[0] + q[2]) / 2 - 20, (q[1] + q[3]) / 2), (q[2], q[3])], n=10), 9, '#3a1a24', taper=(.2, .4), seed=q[0])
            a = c.rgba(); a = np.dstack([blur(a[..., j], 9) for j in range(4)]); self._pl = over(img * .78, a)
        return self._pl.copy()

    # ------------------------------------------------------------------ 30..39  he is not alone: two frames of nothing, the glove (34), the face
    def prince(self, a):
        """Vegeta, bruised, leaning in over the shoulder, lit from below by the pearl"""
        from chars import vegeta as VG
        off = {0: -760., 1: -300., 2: -84., 3: 10., 4: 3., 5: 0.}[a]; cx, cy, s = 604 + off, 424 - off * .05, 262.; L = dict(VG.LIGHTS['dawn'], sk='#d8a493', sh='#7c5270', dp='#42284e', hi='#ffe2d6', ln='#170c1c', hr='#191521', hs='#0b0911', hl='#6f5c92', rim=('#9adaff', -7, -4), key=(.75, -.10, .5), eye='#f6eaf0')
        v = VG.Vegeta(cx, cy, s, math.radians(64 - 4 * min(1, max(0, a - 2) / 2.)), -.04, .13 + off * .0004, expr='smile', light=L, f=0, scratch=1., hair_drop=-.2); c = Cel(W, H); q = lambda P_: A([(cx + x * s, cy + y * s) for x, y in P_])
        # torn suit: trapezius, shoulder, chest, a bare patch where it is gone
        c.fill(spl(q([(-2.6, 3.0), (-2.35, 1.95), (-1.75, 1.45), (-1.05, 1.18), (-.62, 1.06), (-.46, 1.30), (-.18, 1.42), (.10, 1.36), (.28, 1.24), (.52, 1.55), (.70, 2.1), (.72, 3.0)]), True, (0, 4, 8, 11), 8), '#2b3c80'); c.fill(spl(q([(-2.6, 3.0), (-2.35, 1.95), (-1.75, 1.45), (-1.30, 1.40), (-1.1, 1.9), (-.7, 2.3), (-.8, 3.0)]), True, (0, 3, 6), 8), '#18204e'); c.fill(spl(q([(-.16, 1.43), (.06, 1.39), (.22, 1.30), (.36, 1.52), (.20, 1.62), (.28, 1.86), (.06, 1.72), (-.06, 1.94), (-.14, 1.60)]), True, (0, 2, 3, 4, 5, 6, 7, 8), 5), '#c08a84'); c.line(spl(q([(-2.35, 1.95), (-1.75, 1.45), (-1.05, 1.18), (-.62, 1.06)]), n=10), 5.5, '#170c1c', taper=(.1, .2), press=.4, seed=2); c.line(spl(q([(.28, 1.20), (.52, 1.55), (.70, 2.1), (.72, 3.0)]), n=10), 5., '#170c1c', taper=(.1, .2), seed=3); c.line(spl(q([(.36, 1.34), (.56, 1.66), (.66, 2.2)]), n=10), 4., '#8fd0ff', taper=(.3, .4), seed=4, alpha=.8)
        c.line(spl(q([(-.62, 1.06), (-.46, 1.30), (-.18, 1.42), (.10, 1.36), (.28, 1.24)]), n=10), 4.6, '#170c1c', taper=(.1, .1), press=.4, seed=6); c.line(spl(q([(-.58, 1.16), (-.42, 1.38), (-.18, 1.49)]), n=8), 2.6, '#5d74c4', taper=(.2, .5), seed=7); su = c.rgba(); hd = v.layer(); lay = su + hd * (1 - su[..., 3:4]); c = Cel(W, H); F = v.F; pt = lambda x, y: F.P3(A([(x, y, float(F.H.z(x, y)))]))[0]
        b_ = pt(-.44, -.30); c.fill(spl(A([b_ + A([math.cos(t), math.sin(t) * .62]) * 30 * (1 + .18 * math.sin(t * 3)) for t in np.linspace(0, 6.28, 14)[:-1]]), True, (), 5), '#5a2e66', .42); c.fill(spl(A([b_ + A([4, 3]) + A([math.cos(t), math.sin(t) * .6]) * 17 * (1 + .2 * math.sin(t * 4)) for t in np.linspace(0, 6.28, 12)[:-1]]), True, (), 5), '#8a3a52', .42)
        for P_, w in (([(-.30, -.46), (-.36, -.55), (-.39, -.64)], 3.2), ([(-.16, .30), (-.24, .25), (-.31, .17)], 3.4), ([(-.50, -.02), (-.56, -.08), (-.60, -.17)], 2.6)): c.line(spl(A([pt(*z) for z in P_]), n=8), w, '#8f2f3c', taper=(.35, .5), press=.4, seed=int(w * 9))
        ov = c.rgba() * hd[..., 3:4]; lay = ov + lay * (1 - ov[..., 3:4]); return ki.light_on(lay, 1090, 960, 230, 'blue', 3.6, .62)

    def hand(self, f):
        a = f - 34; b = self.bg['b'][..., :3].astype(f32); z = 1.5; img = blur(cv2.warpAffine(b, np.float32([[z, 0, -700], [0, z, -330]]), (W, H), borderMode=cv2.BORDER_REFLECT_101), 12); dip = 0. if a < 0 else (12., 24., 17., 12., 13., 12.5)[a]; close = 0. if a < 0 else (.45, .9, 1., 1., 1., 1.)[a]; ff = min(f, 32)
        img = img + radial(1400, 1180, 1150, 800, 1.6)[..., None] * rgb('#2f7dff') * (.16 + .05 * (f - 30) / 9.)
        turn = float(ease(np.clip((a + .5) / 4., 0, 1))); Hd = goku.GHead().pose(1300 + 4 * turn, 322 + dip * .5, 292, (50 - 12 * turn) * math.pi / 180, (3 - 3 * turn) * math.pi / 180, (-2 - 2 * turn) * math.pi / 180)
        X = goku.getx(('determined', 'shock', .28) if a in (0, 1) else 'determined' if a < 0 else ('serious', 'smile', (.35, .7, .9, 1.)[a - 2]), gx=.018 if a < 0 else (.012, -.014, -.034, -.040, -.040, -.040)[a], iris=.084, crease=.6, cheekline=0.)
        if not hasattr(self, '_L'): L = goku.light_('dawn', side=1); L.update(skin=('#e6b49c', '#93617a', '#4a2e58', '#ffe6da'), amt=.15, line='#170c1c', rim=[dict(d=(-7, 4), col='#ffe6de', amt=1.)], hair=dict(L['hair'], light='#4a3a5a', hi='#c79ab0')); self._L = L
        if a >= 0: img = over(img, self.prince(a))
        lay = goku.bust(Hd, X, dict(self._L), 'base', f=ff, wind=(-.22, .62), dmg=.6, wear='bare', byaw=34 * math.pi / 180, shift=(.04, -.10), seed=3); lay = cv2.warpAffine(lay, np.float32([[1, 0, 0], [0, 1, dip * .3]]), (W, H)); img = over(img, ki.light_on(lay, 1500, 980, 150, 'blue', 5., .55))
        if a >= 0:
            gx, gy = 826 - (1 - close) * 170, 806 + dip - (1 - close) * 120; po = HD.blend(dict(HD.POSES['open'], wrist=(-8, 0)), GV.POSE['shoulder'][0], close); G = GV.glove('shoulder', gx, gy, 250, light='dawn', torn=1., seed=3, side='R', pose=po, yaw=-14, pitch=30, roll=-104 + 10 * (1 - close), dist=6., sleeve='#d3a08e', sleeve_s='#7e4f62', pivot=(0, .80, 0), arm=5.2)
            c = Cel(W, H); sh = blur(G.layer[..., 3], 9); img = img * (1 - .35 * np.roll(np.roll(sh, 16, 0), 10, 1)[..., None] * lay[..., 3:4]); img = over(img, ki.light_on(G.layer, 1500, 1000, 150, 'blue', 5.5, .40))
            if a in (0, 1):
                for k in range(8): th = -2.6 + .34 * k; p = A([gx + 40, gy - 30]); c.line(A([p + A([math.cos(th), math.sin(th)]) * (170 + 34 * a), p + A([math.cos(th), math.sin(th)]) * (230 + 64 * a)]), 5 - 1.6 * a, '#fff0ec', taper=(.1, .6), seed=k, alpha=.8)
                img = over(img, c.rgba())
        k = max(0, a); z = 1.02 + (.03 * math.exp(-k / 2.) if a >= 0 else 0) + .0015 * (f - 30); return cv2.warpAffine(img, np.float32([[z, 0, (1 - z) * 880], [0, z, (1 - z) * 700]]), (W, H), borderMode=cv2.BORDER_REFLECT_101)

    # ------------------------------------------------------------------ 40..79  side by side
    def two(self, f):
        k = 'b' if f < 60 else 'c'; s0 = 40 if f < 60 else 60; a = f - s0; cam = self.cam(k); img = self.bg[k][..., :3].astype(f32).copy(); hd = -math.pi / 2 - .30; d = f if a < 4 else (f // 2) * 2; e = math.exp(-a / 2.2)
        sink = (.00 + .07 * (1 - math.exp(-a / 3.)) if f < 60 else .07 + .24 * (1 - math.exp(-a / 3.2)) + .03 * math.sin(a * 1.4) * math.exp(-a / 4.)); z = Z0 - sink
        # rock: plates break and go down
        fr = self.__dict__.setdefault('_f' + k, PR.Fracture((.0, -21.9), rad=2.1 if k == 'b' else 2.35, seed=7 if k == 'b' else 9, tilt=.16 if k == 'b' else .42, kick=.04, push=.05, r0=.52, thick=.24, n=15, rings=3)); c = Cel(W, H); fr.draw(c, cam, (a + .5) / 24. if k == 'b' else (a + .5) / 24. + .10, ROCKP, floor=Z0 - .04, detail=.9, glow='#4aa8ff' if a > 1 else None); img = over(img, c.rgba())
        img = self.ring(img, cam, f, s0, 'blue' if k == 'b' else 'base', .6, .75, .32)
        gp3, vp3 = ((-.60, -22.54), (.80, -21.44)) if k == 'b' else ((-.56, -22.48), (.80, -21.46)); trem = np.random.default_rng(f).normal(0, .004, 3) * (1 + 1.5 * (k == 'c'))
        V = Fig('vegeta', 'shattered').pose(P('ff_wide', root=(0, -.02 - .02 * e, -.10 - .05 * e), cst=(-16 + 8 * e, 0, 0)) if k == 'b' else mix_pose(P('ff_wide'), P('ff_fire', root=(0, .05, -.16 - .04 * e)), float(ease(np.clip(a / 5., 0, 1))))).place(vp3[0] + trem[0], vp3[1], z + trem[2], heading=hd); V.look((8., -22.6, 36.6)); V.blow(0, 0, 0)
        G = Fig('goku', 'bare', body=GB_).pose(P('kame_wind', root=(.015, -.03, -.135 - .05 * e), cst=(8 + 8 * e, 8, -24))).place(gp3[0] + trem[1], gp3[1], z + trem[2], heading=hd); G.look((7., -24., 36.5)); G.blow(-.10, 0, .10); self.hipcup(G, hd, 1., .32)
        img = self.shadow(img, V, cam, z); img = self.shadow(img, G, cam, z); img = over(img, self.stones(f, 38, cam, 70, amt=.8 + .6 * (k == 'c')))
        LV = V.draw(cam, d, light=(.65, .6, .35), pal=DAWN, rim=[RIMW, dict(dir=(-.8, .3, -.3), col='#ffd76a', thr=.5, amt=.8)], expr=dict(expr='fire' if k == 'b' else 'shout', scratch=1, hair_drop=-.50, wind=-.10 + .04 * math.sin(f * 1.3))); LG = G.draw(cam, d, light=(.65, .6, .35), pal=DAWN, rim=[RIMW, dict(dir=(-.8, .2, -.3), col='#7cc8ff', thr=.55, amt=.6)], expr=dict(expr='grit' if k == 'b' else ('grit', 'rage', .6), dmg=.7, hair='base', wind=(-.05, .22 + .10 * math.sin(f * 1.1) + .25 * e)))
        pw = float(ease(np.clip((a + 2) / 8., 0, 1))); xy = lambda v: A(v, float).reshape(-1)[:2]; vp = [xy(V.info['palms'][s]) for s in 'LR']; gp = (xy(G.info['palms']['L']) + xy(G.info['palms']['R'])) / 2; hv = V.info['head']; hg = G.info['head']; hd_ = lambda h: (float(h['x']), float(h['y'])) if isinstance(h, dict) else (float(h[0]), float(h[1])); fv = (*hd_(hv), 80.); fg = (*hd_(hg), 95.)
        # water from the sea climbs to the blue
        img = over(img, self.streams(f, 37, gp, 9 if k == 'b' else 16))
        img = self.burn(img, LV, f, 'gold', .30 + .25 * pw + .25 * (k == 'c'), 2, fv, .34 + .20 * (k == 'c'))
        rv = 9 + 8 * pw if k == 'b' else 22 + 30 * pw
        if k == 'b' or a < 5:
            rv = rv if k == 'b' else 17.
            for p in vp: o, li = ki.orb(p[0], p[1], rv + 2 * math.sin(f * 1.8), f, 'gold', seed=4); img = ki.put(img + li * .8, o)
        else: p = (vp[0] + vp[1]) / 2; img = ki.put(img, ki.charge(p[0], p[1], rv, f, 'gold', amount=.5, reach=7., water=False)); o, li = ki.orb(p[0], p[1], rv + 2 * math.sin(f * 1.8), f, 'gold', seed=4); img = ki.put(img + li * .8, o)
        rg = 20 + 10 * pw + 2 * math.sin(f * 2.1) if k == 'b' else 44 + 30 * pw + 3 * math.sin(f * 2.1); img = ki.put(img, ki.charge(gp[0], gp[1], rg, f, 'blue', amount=.7, reach=8., water=True)); LG = ki.light_on(LG, gp[0], gp[1], rg * .95, 'blue', 3.0, .38); img = self.burn(img, LG, f, 'blue', .30 + .25 * pw + .25 * (k == 'c'), 5, fg, .34 + .20 * (k == 'c'))
        o, li = ki.orb(gp[0], gp[1], rg, f, 'blue', seed=6, power=1.); img = ki.put(img + li, o)
        if a > 2:
            L, li2 = ki.storm([(s0 + 3, (gp[0], gp[1]), (gp[0] - 260, gp[1] + 170), 5, 40, dict(forks=3)), (s0 + 8, (gp[0], gp[1]), (gp[0] + 240, gp[1] + 200), 5, 44, dict(forks=3)), (s0 + 13, (gp[0], gp[1]), (gp[0] - 120, gp[1] - 260), 5, 48, dict(forks=3)), (s0 + 16, tuple(vp[0]), (vp[0][0] + 260, vp[0][1] + 160), 5, 52, dict(forks=2))], f, 'blue')
            if L is not None: img = over(img + li2 * .6, L)
        if k == 'c': img = over(img, self.slabs(a, gp))
        img = over(img, self.stones(f, 41, cam, 90, True, .9))
        if k == 'c' and f >= 72: img = img + (radial(gp[0], gp[1], 900, power=1.6)[..., None] * rgb('#8fd0ff') + radial(vp[0][0], vp[0][1], 800, power=1.6)[..., None] * rgb('#ffd76a')) * .30 * ((f - 71) / 8.) ** 2
        if a == 0: img = img + radial(960, 600, 1300, 900, 1.2)[..., None] * rgb('#d6ecff') * .35
        return self.push(img, f, (960, 640))

    def frame(self, f): return (self.ka(f) if f < 20 else self.me(f) if f < 30 else self.hand(f) if f < 40 else self.two(f)).astype(f32)
