"""f3 THE PILLAR (40).  Half an orbit (180 degrees) round a body that does not move: aura as a furnace, plates of rock hanging, the sea pushed into a dish,
the storm punched into a ring over his head, lightning.  Opens out of f2's pupil: the match circle (960, 486, r 208)."""
from db.ki_stage import *
from db import land, sky, ki, kit
from db import phys_rock as PR
from db.phys_core import axis_m
TOP = land.TOP; C0 = np.array([0., -22.]); G = lambda x, y: float(land.ground_z(x, y))
BEATS = (0, 10, 20, 30)


class Shot:
    id = 'f3'; n = 40
    finish = dict(bloom=.22, vig=.20, bloom_thr=.86, halation='#ffe6b0', soft=.07)
    EVENTS = [(0, 'aura_ignite', dict(size=1.2, note='furnace: the pillar is already at full; match circle opens')), (0, 'pillar', dict(to=39, size=1.3)), (4, 'storm_punch', dict(size=1., note='the ceiling opens into a ring, air sucked up')), (2, 'sea_dish', dict(size=.8, note='the sea is pushed away all round')), (6, 'plates_lift', dict(size=.7))] + [(b, 'aura_pulse', dict(size=.8)) for b in BEATS] + [(a, 'thunder', dict(size=s)) for a, s in ((2, .8), (9, .6), (12, 1.), (20, .7), (23, .9), (29, .7), (33, 1.))]

    def cam(self, f):
        u = f / 39.; az = -.30 + math.pi * u; e = es(u); d = lerp(3.8, 4.35, e); ez = lerp(.42, .95, e); lz = lerp(1.25, 3.18, es(max(0, u * 1.15 - .10))); z0 = self.z0; n = np.array([math.sin(f * 2.2) * .02 + math.sin(f * 5.1) * .012, math.cos(f * 1.9) * .02, math.sin(f * 3.3) * .015]); return Camera3D(np.array([C0[0] - math.sin(az) * d, C0[1] - math.cos(az) * d, z0 + ez]) + n, (C0[0], C0[1], z0 + lz), lerp(800., 610., e), roll=.02 * math.sin(u * 3))

    def setup(self):
        cv2.setNumThreads(1); self.z0 = G(*C0); self.R = sky.Ring(); self.R.wisps = lambda im, *a: im; self.F = Fig('goku', 'ssj', seed=4); self.F.lane_min = 9.; self.F.pose('power').place(C0[0], C0[1], self.z0, heading=math.pi)
        r = np.random.default_rng(5); path = np.array([self.cam(q).eye for q in range(40)]); self.pl = []
        while len(self.pl) < 78:
            th, q = r.uniform(0, TAU), 2.6 + 22 * r.random() ** 1.6; p = np.array([C0[0] + math.cos(th) * q, C0[1] + math.sin(th) * q]); sz = (.10 + .95 * bms(r.random(), .12, .30) ** 1.6) * (.6 + q / 22.); h = .4 + (1.2 + .32 * q) * r.random() ** 1.3 * 2.2
            if np.hypot(*(path[:, :2] - p).T).min() < 1.2 + sz * 1.6 and h + self.z0 < path[:, 2].max() + 2: continue
            self.pl.append((p, sz, h, r.random() * .35, r.normal(0, 1, 3), r.uniform(0, TAU), r.normal(0, .5), int(r.integers(1, 900)), ('slab', 'slab', 'block', 'shard')[int(r.integers(0, 4))]))
        self.ms = {q[7]: PR.chunk(q[7], q[8]) for q in self.pl}
        self.st = []
        for i, (a, s) in enumerate(((2, .8), (3, .6), (9, .6), (12, 1.), (13, .6), (20, .7), (23, .9), (24, .5), (29, .7), (33, 1.), (34, .6))):
            c = self.cam(a + 2); fw = math.atan2(c.R[2][1], c.R[2][0]); th = fw + (.22 + .42 * h1(i, 1)) * (1 if i % 2 else -1); q = 260 + 260 * h1(i, 2); self.st.append((a, np.array([c.eye[0] + math.cos(th) * q + 60 * (h1(i, 3) - .5), c.eye[1] + math.sin(th) * q, 470.]), np.array([c.eye[0] + math.cos(th) * q, c.eye[1] + math.sin(th) * q, 0.]), 9. * s, 31 + i * 7, q))

    def plates(self, cam, t, f, front, zc):
        cel = Cel(); sec = {}
        for p, sz, h, dl, ax, ph, w, sd, kd in self.pl:
            u = np.clip((t - dl - .02) / .55, 0, 1); e = 1 - (1 - u) ** 2.2 + .12 * math.sin(math.pi * min(1, u * 1.1)) * (u < 1); X = np.array([*p, self.z0 - .2 * sz + h * e + .10 * math.sin(t * 2.6 + ph) * e + t * .35 * e])
            z = cam.cam(X)[2]
            if z < .8 or (z < zc) != front: continue
            d = C0 - p; th = math.atan2(d[1], d[0]); k = int((th + math.pi) / TAU * 8) % 8; sec.setdefault(k, []).append((self.ms[sd].scaled(sz * .62), X, axis_m(ax / np.linalg.norm(ax), ph * .4 * e + w * t * 1.6), sd))
        for k, it in sec.items(): th = (k + .5) / 8 * TAU - math.pi; PR.draw_bodies(cel, cam, it, 'gold', light=(math.cos(th) * .9, math.sin(th) * .9, .25), rim_dir=(math.cos(th), math.sin(th), -.1), lw=2.2)
        return cel.layer()

    def sea_wall(self, img, cam, t, f, vis, pulse):
        """the sea seen from inside its dish: three walls of water painted per pixel in wall metres (U along, V up): ribbed tones that climb, lace of old foam on the face, a torn foam crest with a lit and a shadow tone, spray flung off it"""
        az = math.atan2(cam.R[2][1], cam.R[2][0]); tq = math.floor(t * 12) / 12.; out = np.zeros((H, W, 4), f32); xs = np.arange(0, W, 2, dtype=np.float64); up = lambda a, hh: cv2.resize(a.astype(f32), (W, hh), interpolation=cv2.INTER_CUBIC)
        def st(g, k=1.):                                           # field -> antialiased mask
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3, scale=.125); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3, scale=.125); return np.clip(g / (np.sqrt(gx * gx + gy * gy) * k + 1e-5) + .5, 0, 1)
        for q in range(3):
            rad = 176 + 104 * eo(t / 1.3, 1.5) - 30 * q; a = np.linspace(az - 1.25, az + 1.25, 420); hh = (60 + 78 * eo(t / .9, 1.6) - 30 * q) * (.80 + .17 * noise(a * 4.5, q * 3., tq * .7, 3) + .08 * noise(a * 12, q, tq * 1.1, 5)); X = lambda z_, dr: np.stack([C0[0] + np.cos(a) * (rad + dr), C0[1] + np.sin(a) * (rad + dr), z_ + 0 * a], -1)
            tp, z = cam.project(X(hh, hh * .25)); bs, _ = cam.project(X(-4. + 0 * hh, 0.)); ok = z > 5
            if ok.sum() < 8: continue
            tp, bs, hk, ak, sc = tp[ok], bs[ok], hh[ok], a[ok], cam.f / z[ok]; o = np.argsort(tp[:, 0]); tp, hk, ak, sc = tp[o], hk[o], ak[o], sc[o]; o = np.argsort(bs[:, 0]); bs = bs[o]
            yc = np.interp(xs, tp[:, 0], tp[:, 1]); yb = np.interp(xs, bs[:, 0], bs[:, 1]); hm = np.interp(xs, tp[:, 0], hk); A = np.interp(xs, tp[:, 0], ak); hp = np.maximum(yb - yc, 8.); y0 = int(max(0, (yc - hp * .42).min())) // 2 * 2; y1 = int(min(H, yb.max() + 2)) // 2 * 2
            if y1 - y0 < 20: continue
            Y = np.arange(y0, y1, 2, dtype=np.float64)[:, None]; v = (yb - Y) / hp; sg = np.where(xs > 960, 1., -1.) * np.clip(np.abs(xs - 960) / 500., 0, 1); V = v * hm; U = A * rad + sg * .22 * V * np.clip(v, 0, 1.3); fl = tq * 13.; N = lambda su, sv, sd, fw=1.: noise(U / su, (V - fl * fw) / sv, tq * .5, sd + 40 * q)
            rib = N(3.4, 30., 1); rb2 = N(1.5, 15., 2, 1.3); big = N(17., 16., 3, .4); m1 = N(8., 7., 4, .7); m2 = N(3.1, 2.9, 5, .8); m3 = N(1.35, 1.3, 6, .9); lg = y1 - y0; T = up(v + .13 * rib + .06 * rb2 + .10 * big, lg); fo = up(v + .115 * m1 + .085 * m2 + .045 * m3 - .05 * big, lg); fs = up(.55 * m1 + .75 * m2 + .4 * m3 + .25 * rib, lg); vv = up(np.broadcast_to(v, m1.shape), lg); lace = up(np.abs(.75 * N(6.2, 5.2, 8, .55) + .35 * N(2.6, 2.4, 9, .55)), lg); top = up(1.0 + .13 * np.maximum(0, noise(U / 8., q, tq * 1.6, 11 + q)) + .075 * m2 + .06 * m3, lg); fk = up(.6 * noise(U / 1.6, (V - fl * .9) / 1.7, tq, 21 + q) + .4 * noise(U / .8, (V - fl * .9) / .9, tq, 22 + q), lg)
            col = [C(h) for h in (('#173a40', '#235357', '#3d776e', '#78a88d'), ('#0f2b30', '#1a4347', '#30655f', '#5f927d'), ('#0a1f23', '#123338', '#25524e', '#4b7c6c'))[q]]; hot = np.clip(1.1 - np.abs(np.arange(W, dtype=f32) - 960) / 640., 0, 1)[None, :, None]; body = st(top - vv); K = np.broadcast_to(col[0], (lg, W, 3)).copy()
            for th, c_ in ((.30, col[1]), (.55, col[2]), (.83, col[3])): w = st(T - th)[..., None]; K = K * (1 - w) + c_ * w
            w = (st(.055 + .05 * np.clip(vv, 0, 1) - lace) * np.clip((vv - .22) / .5, 0, 1) * .75)[..., None]; K = K * (1 - w) + (col[3] * .55 + C('#e6e2bd') * .45) * w
            foam = st(fo - .925); ink = np.clip(cv2.dilate(foam, np.ones((3, 3), np.uint8)) - foam, 0, 1)[..., None] * .55; K = K * (1 - ink) + C('#0a2022') * ink
            FC = np.broadcast_to(C('#efe6bd'), K.shape).copy()
            for th, c_ in ((.10, '#bfc9a0'), (.46, '#7f9f8a')): w = st(fs - th)[..., None]; FC = FC * (1 - w) + C(c_) * w
            K = K * (1 - foam[..., None]) + FC * foam[..., None]; gl = (np.clip((vv - .25) / .8, 0, 1) ** 1.5)[..., None] * hot * (.34 + .12 * pulse); K = K * (1 - gl * .55) + C('#ffd27a') * gl * .75
            mi = np.clip(1 - vv / (.16 + .05 * float(q == 2)), 0, 1)[..., None] ** 1.4 * .55; K = K * (1 - mi) + C('#6f8a72') * mi
            fleck = st(fk - (.16 + np.clip((vv - top) / .34, 0, 1) ** .7 * .62)) * (vv > top - .02) * (vv < top + .36); al = np.maximum(body, fleck); ed = np.clip(cv2.dilate(body, np.ones((3, 3), np.uint8)) - body, 0, 1) * (1 - fleck) * .8; K = np.where((fleck > body)[..., None], C('#e9e3c0'), K); lay = np.dstack([K * al[..., None] + C('#081416') * (ed * (1 - al))[..., None], np.maximum(al, ed)]).astype(f32); lay = lay * np.clip((y1 - np.arange(y0, y1)) / 3., 0, 1)[:, None, None].astype(f32)
            ms = cv2.resize(cv2.GaussianBlur(cv2.resize(body * foam, (W // 8, max(2, lg // 8)), interpolation=cv2.INTER_AREA), (0, 0), 3.2), (W, lg)) * .22; sub = out[y0:y1]; sub[:] = ki.put4(sub, np.dstack([ms[..., None] * C('#cfd6ae'), ms])) if hasattr(ki, 'put4') else np.dstack([sub[..., :3] * (1 - ms[..., None]) + ms[..., None] * C('#cfd6ae'), sub[..., 3] * (1 - ms) + ms]); sub[:] = lay + sub * (1 - lay[..., 3:4])
        B = Cel(ss=1)
        for i in range(90):                                        # drops flung off the near lip
            j = int(h1(5, i, 31) * 419); u = (t * (1.0 + h1(5, i, 32)) + h1(5, i, 33)) % 1.; aa = az - 1.1 + 2.2 * j / 419.; r_ = 176 + 104 * eo(t / 1.3, 1.5) - 34 - 20 * h1(5, i, 35); z_ = 66 * eo((t + .1) / .9, 1.6) * (.75 + .25 * h1(5, i, 36)) + 34 * u - 46 * u * u; p_, zz = cam.project(np.array([[C0[0] + math.cos(aa) * r_, C0[1] + math.sin(aa) * r_, z_]]))
            if zz[0] < 5: continue
            r = cam.f / zz[0] * (.25 + 1.0 * h1(5, i, 34) ** 2) * (1 - u); x0, yy = p_[0]; B.fill(np.array([[x0 - r * .6, yy], [x0, yy - r * 1.7], [x0 + r * .6, yy], [x0, yy + r]]), '#f3ebc2', .9)
        out = stack(out, B.layer()); return ki.put(img, out * cv2.GaussianBlur(vis.astype(f32), (0, 0), .8)[..., None])

    def frame(self, f):
        t = f / 24.; cam = self.cam(f); pulse = math.exp(-(f % 10) / 2.6); k = 620. * eo(min(1., (f + 2) / 15.), 2.0) + 40 * math.sin(min(1, f / 15.) * math.pi); fl = sum(max(0., 1 - (f - s[0] - 1) / 2.) for s in self.st if 0 <= f - s[0] - 1 < 2)
        sk, _ = self.R.frame(cam, f, k=max(60., k), gold=.95 + .25 * pulse, flash=min(1., fl)); o = land.ground(cam, t, 'storm', sky=sk, fires=[(C0[0], C0[1], self.z0 + 2.2, 13 + 2 * pulse, '#ffc95a', .86 + .22 * pulse)], rings=[(C0[0], C0[1], 110 + 190 * eo(t / 1.2, 1.6), 46, 5.5)]); img = o.img
        Z = np.where(o.a > .5, o.z, 1e6) if hasattr(o, 'a') else o.z; rk = o.rock
        # the sea pushed away: a wall of water running out on all sides
        img = self.sea_wall(img, cam, t, f, (Z > 130) & (rk < .5), pulse)
        S = []
        for a, A_, B_, w, sd, dist in self.st:
            if 0 <= f - a < 6:
                p, z = cam.project(np.stack([A_, B_]))
                if z.min() > 5: S.append((a, p[0], p[1], w, sd, dict(forks=5)))
        l, g = ki.storm(S, f, 'gold') if S else (None, None)
        if l is not None: v = cv2.GaussianBlur((Z > 200).astype(f32), (0, 0), .8)[..., None]; img = ki.put(img, l * v) + g * v + .05 * min(1., fl)
        zc = cam.cam((C0[0], C0[1], self.z0 + 1.))[2]; img = ki.put(img, self.plates(cam, t, f, False, zc))
        F = self.F; F.blow(0, 0, .95); Lg = safe(F, cam, f // 2 * 2, pal=dict(base='gold', litmul='#ffe9b4', mul='#f2c48a', shmul='#8c5636'), light=(.15, .75, .5), weight=1.5, rim=[dict(dir=(.92, .15, -.35), col='#fff3c0', thr=.52, amt=.9), dict(dir=(-.92, .15, -.35), col='#ffe27a', thr=.54, amt=.85)], expr=dict(expr='rage')); bx = F.info['box']; hp, hz = cam.project(np.array([[C0[0], C0[1], self.z0], [C0[0], C0[1], self.z0 + 1.78]])); fh = hp[0, 1] - hp[1, 1]; cy = (hp[0, 1] + hp[1, 1]) / 2
        inn, fr, li = ki.pillar3(cam, f, P0=(C0[0], C0[1], self.z0 + .02), R0=1.12, height=34., n=128, tl=2.5, power=1. + .12 * pulse, hole=(hp[0, 0], cy + fh * .02, fh * .46, fh * .66)); A = ki.aura(Lg, f, 'gold', .85 + .25 * pulse, 4, height=.42, face=face(F), front=.25, light=.2, veil=0., dens=.8, keep=[(*F.info['fists'][q], fh * .05) for q in 'LR'] if isinstance(F.info.get('fists'), dict) else ())
        img = ki.put(img, inn); sm_ = cv2.GaussianBlur(cv2.resize(Lg[..., 3], (W // 8, H // 8), interpolation=cv2.INTER_AREA), (0, 0), max(1.5, fh * .040 / 8.)); hm = np.clip(cv2.resize(sm_, (W, H), interpolation=cv2.INTER_LINEAR) * 1.25, 0, 1)[..., None]; img = img * (1 - hm * .46) + C('#e2881a') * hm * .46; img = A.under(img); img = ki.put(img, A.front); img = ki.put(img, Lg); img = img + li * (1 - Lg[..., 3:4]) * (1 - .6 * hm); img = ki.put(img, fr)
        _, af, ag = ki.arcs(Lg, f, 'blue', 6, 3, face=face(F)); img = ki.put(img, af) + ag
        for j in range(2):
            u = (t * 1.5 + j * .5) % 1.; q = ki.ring3(cam, (C0[0], C0[1], self.z0 + .05), (0, 0, 1), 2.2 + 8. * u, .055 * (1 - u) + .012, 'gold', 5 + j + int(t * 1.5 + j * .5) * 3, .35, .85 * (1 - u), .05, spurs=0)
            if q is not None: img = ki.put(img, q)
        img = ki.put(img, self.plates(cam, t, f, True, zc))
        D = Field(2)                                               # grit and drops going up, streaks on ones
        for i in range(90):
            q = t * (.7 + .9 * h1(i, 1)) + h1(i, 2); u = q - math.floor(q); th = TAU * h1(i, 3, math.floor(q)); rr = 1.6 + 7 * h1(i, 4); X = np.array([[C0[0] + math.cos(th) * rr, C0[1] + math.sin(th) * rr, self.z0 + 8 * u ** 1.2 * (.4 + h1(i, 5))]]); X = np.vstack([X, X - [0, 0, .12 + .5 * h1(i, 6)]]); p, z = cam.project(X)
            if z.min() < .6: continue
            w = cam.f / z[0] * (.006 + .016 * h1(i, 7) ** 2) * min(1., u * 5, (1 - u) * 4); D.rib(np.array([p[0], (p[0] + p[1]) / 2, p[1]]), [np.array([.2, w, .2]), np.array([-.4, w * .5, -.4])], 0.)
        img = ki.put(img, D.cel(('#ffcf5c', '#fffbe0'))); img = shaken(img, *(kick(f, [(0, 1.2), (10, .7), (20, .7), (30, .7)], 8., 3., 5) + kit.shake(f, 1.6, 9)))
        if f < 4:                                                  # out of the pupil
            x, y = grid(); d = np.hypot(x - 960, y - 486); r = (208, 268, 470, 900)[f]; th = np.arctan2(y - 486, x - 960); tb = np.linspace(-math.pi, math.pi, 721); nz = np.abs(noise(tb * 14, f * 3., 0, 5)) + .6 * np.abs(noise(tb * 37, f * 5., 0, 6)); nz[-1] = nz[0]; rr = r * (1 + (.0, .22, .34, .40)[f] * np.interp(th.ravel(), tb, nz).reshape(th.shape)); m = np.clip((d - rr) / 1.5 + .5, 0, 1)[..., None]; c = Cel()
            for i in range(150): a = TAU * (i + h1(f, i)) / 150; q = r * (1.0 + .9 * h1(f, i, 2)); w = .006 + .022 * h1(f, i, 3) ** 2; c.fill(np.array([[960 + math.cos(a) * q, 486 + math.sin(a) * q], [960 + math.cos(a - w) * 2300, 486 + math.sin(a - w) * 2300], [960 + math.cos(a + w) * 2300, 486 + math.sin(a + w) * 2300]]), ('#ffc83c', '#f08a1e', '#ffffff')[i % 3])
            out = ki.put(np.broadcast_to(C('#ffeaa0'), img.shape).astype(f32), c.layer()); img = img * (1 - m) + out * m
            if f == 0: img = ki.put(img, ki.ring(960, 486, 208, 9, 0, 'gold', 1., 0, 7, 0., 1., spurs=0))
        return img.astype(f32)
