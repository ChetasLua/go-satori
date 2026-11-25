"""E3  THE LAST ONE.  80 f.  Memory.  0-39 one held picture: he sleeps; a white glove sets the bun down (8), hesitates (10-19), covers it with a bowl (28), a boot draws the quilt over a foot (34-38).
40 Vegeta walks to the moonlit door, 48-59 his profile: a small smile nobody sees.  60 morning: joy.  74 the empty plate; 76 one drop, and the colour goes."""
import numpy as np, cv2, math
from db import W, H, comp
from db.paint import f32, rgb, grid, smooth, lum
from db.ink import Cel, spl
from db import kit, props
from db import hands as HD, glove as GL
from chars.chibi_core import *
from chars import chibi_story as ST, chibi_stick as SK, chibi_feast as FE, chibi_memory as M
from chars.chibi_world import plate

GL.LIGHT['memo'] = GL._p('#fbf1e2', '#a79ac0', '#6f6496', '#3a2a3a', '#efc3a0', '#c08a74', '#fffaf0', (-.6, .55, .5), (.8, .35, -.3), '#b9c6f0', .45, '#3a48a0', '#232c68')
HOLD = HD.pose((34, 32, 16, -8), (34, 34, 16, -1), (36, 36, 18, 6), (42, 38, 20, 14), thumb=(58, 44, 10, 10, 10), wrist=(-6, 0))
OPEN = HD.pose((20, 22, 12, -9), (18, 22, 12, -1), (22, 24, 14, 7), (28, 28, 16, 16), thumb=(60, 32, 8, 6, 6), wrist=(-4, 0))
HALF = HD.blend(HOLD, OPEN, .55)
BOWLG = HD.pose((30, 30, 14, -12), (28, 30, 14, -3), (30, 32, 16, 7), (36, 34, 18, 17), thumb=(60, 40, 10, 10, 8), wrist=(-10, 0))
GX, GY, GS = 1200., 560., 138.                                              # reference place of the drawn gloves
SX, SY, SS = 560., 452., 150.                                               # the sleeper
BR = 78.


class Shot:
    id = 'e3'; n = 80
    finish = dict(bloom=.20, vig=.10, soft=.18, grain=.006, ca=.3, lift=.0)
    EVENTS = [(0, 'projector', dict(size=.3, dur=74, note='super-8 whirr, very soft; stops dead at 76')), (0, 'breath_sleep', dict(size=.3, dur=39, pan=-.4, note='Goku asleep, bubble swells and shrinks, period 48 f')),
              (8, 'bun_down', dict(size=.35, pan=.05, note='bun touches the plate, almost no sound')), (10, 'hesitate', dict(size=.1, dur=10, note='nothing. the hand stays')), (14, 'finger', dict(size=.08, note='one finger draws back')),
              (28, 'bowl_down', dict(size=.4, pan=.05, note='upturned bowl settles over it, ceramic, tender')), (32, 'boot', dict(size=.2, pan=.5)), (34, 'quilt', dict(size=.3, pan=.45, note='cloth drawn over a foot')),
              (40, 'step', dict(size=.3, pan=.1, note='bare floor, one step')), (47, 'step', dict(size=.25, pan=.1)), (48, 'smile', dict(size=.2, dur=12, note='his profile, the moon; the music box finds its last note here')),
              (60, 'morning', dict(size=.5, note='light, sparrows')), (61, 'bowl_lift', dict(size=.4)), (64, 'see', dict(size=.3)), (67, 'joy', dict(size=.7, note='he laughs, the bun against his cheek')),
              (74, 'cut_plate', dict(size=.2, note='all sound thins')), (76, 'drop', dict(size=.6, note='one raindrop on the empty plate; projector stops; colour drains')), (78, 'rain_far', dict(size=.3))]

    def setup(self):
        self.rn, self.rm = M.room('night'), M.room('morn'); self.tn, self.tm = M.table('night'), M.table('morn'); self._g = {}; self._s = {}

    # ------------------------------------------------ pieces
    def glove(self, key):
        if key not in self._g:
            p, y, pi_, r = dict(hold=(HOLD, 58, -14, 128), half=(HALF, 58, -13, 128), half2=(HD.blend(HOLD, OPEN, .75), 58, -12, 129), open=(OPEN, 60, -10, 128), bowl=(BOWLG, 56, -20, 131))[key]
            G = GL.glove('relaxed', GX, GY, GS, light='memo', torn=0., side='R', pose=p, yaw=y, pitch=pi_, roll=r, dist=9.); pt = lambda f, j: A(G.pts[(f, j)], float)
            al = G.layer[..., 3] > .5; ys = np.where(al.sum(1) > 24)[0]; y0 = ys[0] + 84; w0 = np.where(al[y0])[0]; y1 = y0 + 90; w1 = np.where(al[y1])[0]; c0, c1 = A([w0.mean(), y0]), A([w1.mean(), y1]); u = (c0 - c1) / np.hypot(*(c0 - c1)); cs = abs(u[1]); h0, h1 = len(w0) / 2 * cs, len(w1) / 2 * cs; far = c0 + u * 1500
            dd = D(0, -400, W + 700, H + 400, 2, lw=3.6, line=M.LINE, soft=1.2, var=.2); gr = max(.02, (h0 - h1) / 90.); Pt = np.vstack([[c0 - u * 30 + A([-u[1], u[0]]) * h0 * .99, c0 + A([-u[1], u[0]]) * h0], [far + A([-u[1], u[0]]) * (h0 + min(gr, .05) * 900), far - A([-u[1], u[0]]) * (h0 + min(gr, .05) * 900)], [c0 - A([-u[1], u[0]]) * h0, c0 - u * 30 - A([-u[1], u[0]]) * h0 * .99]]); nn = A([-u[1], u[0]]); dd.part(Pt, '#3a48a0', '#232c68', w=1., shade=[np.vstack([[c0 + nn * h0 * .05, far + nn * h0 * .10, far + nn * 400, c0 + nn * 400]])] if (nn[0] > 0) else [np.vstack([[c0 - nn * h0 * .05, far - nn * h0 * .10, far - nn * 400, c0 - nn * 400]])]); ext = dd.rgba()
            full = np.zeros((H + 400, W + 700, 4), f32); full[400:, :W] = G.layer; full = ext * (1 - full[..., 3:4]) + full; self._g[key] = (full.astype(np.float16), (pt('index', 'pad') + pt('middle', 'pad')) / 2, pt('thumb', 'pad'), A(G.pts['palm'], float))
        return self._g[key]

    def put(self, im, key, at, anchor):
        """glove layer moved so that its anchor point (in reference space) sits at `at`"""
        L = self.glove(key)[0].astype(f32); dx, dy = A(at, float) - anchor; L = cv2.warpAffine(L, np.float32([[1, 0, dx], [0, 1, dy - 400]]), (W, H), flags=cv2.INTER_LINEAR)
        L[:, :, :] = M.light_layer(L, 'night'); sh = cv2.GaussianBlur(L[..., 3], (0, 0), 9); sh = cv2.warpAffine(sh, np.float32([[1, 0, 26], [0, 1, 34]]), (W, H)); 
        return im, L, sh

    def sleeper(self, f, foot=0.):
        g = f // 3 * 3; key = (g, round(foot, 2))
        if key not in self._s:
            if len(self._s) > 4: self._s.clear()
            d = D(lw=4.6, line=M.LINE, soft=1.3, var=.24); FE.sleep_goku(d, SX, SY, SS, g, foot=0., smile=0.); 
            # the quilt's hem, drawn over the foot as `foot` goes 0 -> 1
            n = N().at(SX, SY, 0., SS); u = foot; br = math.sin(g * .13)
            hem = spl([(5.7, -.46), (6.04 + .30 * u, -.34 - .10 * u), (6.36 + .50 * u, -.02 - .18 * u), (6.52 + .52 * u, .48), (6.56 + .46 * u, 1.10), (6.0, 1.30), (5.5, 1.1), (5.46, .2)], True, n=7); P = n(hem)
            d.part(P, FE.NIGHT['quilt'], FE.NIGHT['quilt2'], w=1., seed=70, gaps=[(.60, .995)], shade=[n(spl([(5.3, .62), (6.0, .40), (6.7, .5), (7.4, .4), (7.4, 1.6), (5.3, 1.6)], True, n=5))])
            for L in ([(6.0, .0), (6.2 + .2 * u, .24), (6.34 + .3 * u, .56)], [(5.9, .7), (6.2 + .1 * u, .86), (6.44 + .3 * u, .98)]): d.stroke(n(spl(L, n=6)), .55, FE.NIGHT['quilt2'], taper=(.2, .5), within=P)
            self._s[key] = M.light_layer(d.rgba(grain=.04), 'night').astype(np.float16)
        return self._s[key].astype(f32)

    # ------------------------------------------------ A: the night
    def night(self, f):
        im = self.rn.copy(); g = f // 2 * 2; foot = float(keys(g, [(32, 0.), (34, .22, 'e'), (36, .72, 'e'), (38, 1.0, 'o'), (80, 1.)])); X, Y = grid()
        sh = np.exp(-(((X - 1050) / 620) ** 2 + ((Y - 715) / 52) ** 2)) * .34; im = im * (1 - sh[..., None]) + rgb('#1d1836') * sh[..., None]
        im = comp.over(im, self.sleeper(g, foot))
        # the boot: a toe that knows how to be quiet
        if True:
            x = float(keys(g, [(0, 1812.), (30, 1812.), (32, 1776., 'e'), (34, 1700., 'e'), (36, 1626., 'e'), (38, 1636., 'o'), (40, 1640.)])); y = float(keys(g, [(0, 676.), (30, 676.), (32, 660., 'e'), (34, 634., 'e'), (36, 622., 'e'), (38, 660., 'e'), (40, 668.)])); r = float(keys(g, [(30, 0.), (32, .05), (34, .20, 'e'), (36, .10, 'e'), (38, 0., 'e')])); d = D(lw=4.8, line=M.LINE, soft=1.3, var=.24); d.fl.fill(ell(x - 30, 676, 120, 18), '#1d1836', .35); M.boot(d, N().at(x, y, r, 98)); im = comp.over(im, M.light_layer(d.rgba(grain=.04), 'night'))
        im = comp.over(im, self.tn); px, py = M.PLT
        d = D(lw=5.2, line=M.LINE, soft=1.4, var=.28); M.dish(d, px, py, 196); base = A([px, py + 4])
        # where bun, bowl and hand are
        if g < 10: bb = base + [float(keys(g, [(0, 92.), (8, 0., 'e')])), float(keys(g, [(0, -330.), (4, -150., 'o'), (8, 0., 'o')]))]; key = 'hold'
        else: bb = base
        G = self.glove('hold'); mid = (G[1] + G[2]) / 2; r = np.hypot(*(G[2] - G[1])) / 2 * 1.02; off = A([-4, -r * .40])                    # bun centre relative to pads
        rise = None
        if g < 10: hk, hp = 'hold', bb
        elif g < 14: hk, hp = 'half', bb + [2, -8]
        elif g < 20: hk, hp = 'half2', bb + [6, -22]
        elif g < 24: hk, hp = 'open', bb + [float(keys(g, [(20, 30.), (23, 420., 'i')])), float(keys(g, [(20, -70.), (23, -520., 'i')]))]
        elif g < 30: hk = 'bowl'; hp = base + [float(keys(g, [(24, 300.), (28, 0., 'e')])), float(keys(g, [(24, -420.), (26, -110., 'o'), (28, 0., 'o'), (40, 0.)]))]
        else: hk = 'open'; hp = base + [float(keys(g, [(30, 6.), (34, 60., 'e'), (40, 520., 'i')])), float(keys(g, [(30, -40.), (34, -150., 'e'), (40, -700., 'i')]))]
        bowl = hk == 'bowl' or g >= 30; bp = hp if hk == 'bowl' else base
        e = D(lw=5.2, line=M.LINE, soft=1.4, var=.28)
        if not (bowl and np.hypot(*(bp - base)) < 30): M.bun_soft(e, N().at(bb[0], bb[1]), BR)
        if bowl: M.bowl_down(e, N().at(bp[0], bp[1] + 12, .0 if np.hypot(*(bp - base)) < 20 else .05), 128)
        dish = M.light_layer(d.rgba(grain=.03), 'night'); food = M.light_layer(e.rgba(grain=.03), 'night'); im = comp.over(im, dish)
        if g < 40:
            if hk == 'bowl': an = self.glove('bowl')[3]; at = hp + [26, -132]
            else:
                K_ = self.glove(hk); an = (K_[1] + K_[2]) / 2; at = hp + [0, -BR * .62] if hk == 'hold' else hp + [0, -BR * .62]
            im, L, s = self.put(im, hk, at, an); s = s * smooth(760, 850, Y) * .34; im = im * (1 - s[..., None]) + rgb('#1d1430') * s[..., None]; im = comp.over(im, food)
            if g < 28: im = self.steam(im, bb, f, .55)
            im = comp.over(im, L)
        else: im = comp.over(im, food)
        X0 = np.clip((X - 960) / 960, -1, 1); return im * (1 - .16 * X0[..., None] ** 2)

    def steam(self, im, b, f, amt):
        if amt <= 0: return im
        c = Cel(W, H, 0, 0, 1); FE.steam(c, b[0], b[1] - 96, 58, f * 1.0, 3, '#e9e2f4', amt * .8); return comp.over(im, cv2.GaussianBlur(c.rgba(), (0, 0), 2.4))

    # ------------------------------------------------ B: the door, the profile
    def door(self, f):
        t = f - 40; g = t // 2 * 2; X, Y = grid(); mx, my = 960., 372.
        if 'door' not in self._g:
            def fn(c):
                c.fill(A([(0, 0), (W, 0), (W, H), (0, H)]), '#2b2650'); c.fill(A([(0, 0), (W, 0), (W, 820), (0, 820)]), '#332d5c')
                for x in range(-40, W, 130): c.fill(A([(x, 0), (x + 3, 0), (x + 3, 820), (x, 820)]), '#262047')
                c.grad(A([(610, 60), (1310, 60), (1310, 826), (610, 826)]), (0, 60), (0, 826), [(0, '#4f5fae', 1), (.5, '#7f90dc', 1), (1, '#b9c4f4', 1)])
                c.fill(spl([(610, 826), (610, 650), (720, 606), (830, 640), (960, 590), (1090, 636), (1200, 600), (1310, 640), (1310, 826)], n=8), '#6573b8'); c.fill(spl([(610, 826), (610, 730), (760, 700), (940, 736), (1120, 700), (1310, 740), (1310, 826)], n=8), '#4c5796')
                r = np.random.default_rng(2)
                for k in range(26): c.fill(ell(r.uniform(630, 1290), r.uniform(80, 520), r.uniform(1.2, 2.6), r.uniform(1.2, 2.6), n=10), '#eef2ff')
                for q in ([(574, 30), (616, 30), (616, 840), (574, 840)], [(1304, 30), (1346, 30), (1346, 840), (1304, 840)], [(560, 22), (1360, 22), (1360, 70), (560, 70)]): c.fill(A(q), '#1d1838')
                for x in (40, 1390): c.fill(A([(x, 110), (x + 480, 110), (x + 480, 800), (x, 800)]), '#423b70'); [c.fill(A([(x + i * 118, 110), (x + i * 118 + 8, 110), (x + i * 118 + 8, 800), (x + i * 118, 800)]), '#262047') for i in range(5)]; [c.fill(A([(x, yy), (x + 480, yy), (x + 480, yy + 8), (x, yy + 8)]), '#262047') for yy in (110, 250, 390, 530, 670, 792)]
                c.fill(A([(0, 820), (W, 820), (W, H), (0, H)]), '#2a2450'); c.fill(A([(616, 824), (1304, 824), (1640, H), (280, H)]), '#6d78ba', .55); c.fill(A([(590, 820), (1330, 820), (1336, 838), (584, 838)]), '#1d1838')
            a = plate(W, H, fn, seed=41, wob=1.8, grain=.05, pig=.08)[..., :3]; rr = np.sqrt((X - mx) ** 2 + (Y - my) ** 2); m = smooth(150, 147, rr)[..., None]; a = a * (1 - m) + (rgb('#fbf8ff') * (1 - .05 * np.exp(-(((X - mx + 40) / 60) ** 2 + ((Y - my + 10) / 46) ** 2)) - .05 * np.exp(-(((X - mx - 46) / 46) ** 2 + ((Y - my - 50) / 40) ** 2)))[..., None]) * m; a = comp.screen(a, (np.clip(1 - rr / 620, 0, 1) ** 2.2 * .42)[..., None] * rgb('#c9d4ff')); a = a + (np.exp(-(((X - 60) / 520) ** 2 + ((Y - 900) / 420) ** 2)) * .16)[..., None] * rgb('#ff9a4a'); self._g['door'] = a.astype(np.float16)
        im = self._g['door'].astype(f32); p = .06 + g / 16.; S = 82 - .8 * g; gy = 924 - 3.0 * g; d = D(lw=4.6, line='#1c1636', soft=1.4, var=.28); M.vegeta_back(d, N().at(960, gy, 0., S), p); L = d.rgba(grain=.04); L[..., :3] *= rgb('#8f8fc8') * .62
        for dx in (7, -7): L = kit.rim(L, dx, 4, '#dfe6ff', .85, .8)
        a = L[..., 3]; sh = cv2.warpAffine(a, np.float32([[1.05, 0, -48], [0, -.62, gy * 1.62]]), (W, H), flags=cv2.INTER_LINEAR); sh = cv2.GaussianBlur(sh, (0, 0), 5) * (Y > gy - 8) * .72; im = im * (1 - sh[..., None]) + rgb('#171230') * sh[..., None]
        im = comp.over(im, L); z = 1.0 + .0022 * t; return cv2.warpAffine(im, np.float32([[z, 0, 960 * (1 - z)], [0, z, 500 * (1 - z)]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    def profile(self, f):
        t = f - 48; X, Y = grid()
        if 'sky' not in self._g:
            c = Cel(W, H, 0, 0, 1); c.grad(A([(0, 0), (W, 0), (W, H), (0, H)]), (0, 0), (0, H), [(0, '#3d4a94', 1), (.55, '#7484cf', 1), (1, '#a9b4ec', 1)]); r = np.random.default_rng(6)
            for k in range(34): c.fill(ell(r.uniform(0, W), r.uniform(0, 700), r.uniform(2, 5), r.uniform(2, 5), n=12), '#eef2ff', .8)
            c.fill(spl([(-20, H), (-20, 930), (300, 860), (620, 920), (980, 850), (1400, 910), (1960, 850), (1960, H)], n=8), '#5763a8'); a = plate(W, H, lambda q: None, seed=3)[..., :3] * 0 + cv2.GaussianBlur(c.rgba()[..., :3], (0, 0), 7)
            rr = np.sqrt((X - 1395) ** 2 + (Y - 430) ** 2); m = smooth(246, 232, rr)[..., None]; a = a * (1 - m) + rgb('#fbf8ff') * m; a = comp.screen(a, (np.clip(1 - rr / 980, 0, 1) ** 2 * .46)[..., None] * rgb('#c9d4ff')); b = Cel(W, H, 0, 0, 1); b.fill(A([(-10, -10), (196, -10), (176, H + 10), (-10, H + 10)]), '#1d1838'); q = cv2.GaussianBlur(b.rgba(), (0, 0), 10); a = comp.over(a, q); a = a * (1 + .015 * tooth(H, W))[..., None]; self._g['sky'] = a.astype(np.float16)
        im = self._g['sky'].astype(f32); g = t // 2 * 2; sm = float(keys(t, [(1, 0.), (6, 1., 'e'), (12, 1.)])); lid = float(keys(t, [(0, .30), (3, .34), (7, .62, 'e'), (12, .64)])); d = D(lw=6.2, line=M.LINE, soft=1.6, var=.30, light=(.8, -.5)); n = N().at(770 + 1.1 * t, 566 + 1.6 * math.sin(t * .26), .03 + .012 * Eo(t / 8.), 250); M.vegeta_profile(d, n, sm, lid, g); L = d.rgba(grain=.04)
        a = L[..., 3]; k = smooth(1100, 420, X + (Y - 500) * .15); L[..., :3] *= (1 - .42 * k)[..., None] * (rgb('#ffffff') * (1 - .5 * k)[..., None] + rgb('#a9a3e0') * (.5 * k)[..., None]); L = kit.rim(L, -8, 4, '#f4f7ff', .95, 1.2); warm = kit.rim(L, 7, -3, '#ffb877', .30, 1.6, where=(X < 640).astype(f32)); return comp.over(im, warm)

    # ------------------------------------------------ C: morning
    def morning(self, f):
        t = f - 60; g = t // 2 * 2; im = self.rm.copy(); X, Y = grid(); d = D(lw=4.8, line='#5a3a30', soft=1.3, var=.24); n = N().at(880, 452, 0., 150); px, py = M.PLT; base = A([px, py + 4]); e = D(lw=5.2, line='#5a3a30', soft=1.4, var=.28)
        sh = np.exp(-(((X - 900) / 520) ** 2 + ((Y - 800) / 60) ** 2)) * .22; im = im * (1 - sh[..., None]) + rgb('#8a5a3a') * sh[..., None]
        if t < 7:
            up = float(keys(g, [(0, 0.), (2, .30, 'o'), (4, .85, 'o'), (6, 1., 'o')])); bw = base + [40 + 190 * up, 12 - 300 * up]; hand = n.inv(bw + [-26, -104 + 10 * up]); k = dict(state='lift' if t < 4 else 'see', t=t, hand=hand, yaw=.30, pitch=.16, roll=.04 - .05 * up, lag=(0, .05), by=.02 * up); h, hn = M.goku_morning(d, n, k); im = comp.over(im, M.light_layer(d.rgba(grain=.04), 'morn')); im = comp.over(im, self.tm); M.dish(e, px, py, 196, 'morn'); M.bun_soft(e, N().at(*base), BR, 'morn') if up > .12 else None; M.bowl_down(e, N().at(bw[0], bw[1], .34 * up), 128, 'morn'); ST.hand(e, N().at(bw[0] - 10 - 30 * up, bw[1] - 112 + 6 * up, .34 * up + .2, 1.5 * 112), ST.C['goku'], 'open', seed=5); im = comp.over(im, M.light_layer(e.rgba(grain=.03), 'morn'))
            if up > .1: c = Cel(W, H, 0, 0, 1); FE.steam(c, base[0], base[1] - 90, 92, f * 1.6, 5, '#ffffff', .62 * min(1, up * 1.4)); im = comp.over(im, cv2.GaussianBlur(c.rgba(), (0, 0), 2.4))
        else:
            j = t - 7; by = [-.16, -.24, -.10, -.02, -.10, -.14, -.06][min(j, 6)]; k = dict(state='joy', t=t, by=by, yaw=.18, pitch=-.04, roll=[-.02, -.10, -.08, .02, .08, .06, 0.][min(j, 6)], lag=(0, [.3, .2, -.2, -.1, .14, .08, -.08][min(j, 6)])); M.goku_morning(d, n, k); im = comp.over(im, M.light_layer(d.rgba(grain=.04), 'morn')); im = comp.over(im, self.tm); M.dish(e, px, py, 196, 'morn'); M.bowl_down(e, N().at(px + 330, py + 70, .9), 118, 'morn'); im = comp.over(im, M.light_layer(e.rgba(grain=.03), 'morn')); q = n([(.30, 1.0)])[0]; c = Cel(W, H, 0, 0, 1); FE.steam(c, q[0] + 150, q[1] + 40, 46, f * 1.6, 2, '#ffffff', .4); im = comp.over(im, cv2.GaussianBlur(c.rgba(), (0, 0), 2.4)); im = self.joy(im, n, j)
        g_ = (np.clip(1 - np.abs((X - 1500) + (Y - 300) * .9) / 420, 0, 1) ** 1.5 * np.clip(1 - (Y - 100) / 1000, 0, 1) * .16)[..., None]; return comp.screen(im, g_ * rgb('#fff0c0'))

    def spark(self, im, n, j):
        e = D(lw=4.8, line='#5a3a30'); 
        for a in (-2.2, -1.57, -.95): v = A([math.cos(a), math.sin(a)]); c0 = A([.05, -.2]); e.part(n(tube([c0 + v * (1.62 + .08 * j), c0 + v * (1.92 + .08 * j)], (.05, .015))), '#fffbe6', '#fffbe6', w=.4)
        return comp.over(im, e.rgba())

    def joy(self, im, n, j):
        e = D(lw=4.8, line='#5a3a30')
        for i, (x, y, r) in enumerate(((-1.75, -.7, .15), (1.9, -.2, .13), (1.5, -1.25, .17), (-1.5, .5, .11), (1.85, .75, .12))): k = .75 + .25 * math.sin(j * 1.1 + i * 2); e.part(n(star(x, y - .03 * j, r * k, .3 * i + .1 * j, .5)), '#ffe27a', '#f0b53f', w=.5)
        return comp.over(im, e.rgba())

    # ------------------------------------------------ D: the plate
    def empty(self, f):
        t = f - 74; X, Y = grid(); px, py = M.PLT; z = 1.92 + .02 * t; cx, cy = 960., 640.
        T = lambda q: cv2.warpAffine(q, np.float32([[z, 0, cx - z * px], [0, z, cy - z * py]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        if 'rmb' not in self._g: self._g['rmb'] = cv2.GaussianBlur(self.rm, (0, 0), 5)
        im = T(self._g['rmb']) * .92; im = comp.over(im, T(self.tm)); e = D(lw=7.0, line='#4a3a40', soft=1.6, var=.28); r = 196 * z; M.dish(e, cx, cy, r, 'morn')
        for dx, dy, q in ((-96, 4, 7), (52, 26, 5), (120, -8, 6)): e.part(ell(cx + dx, cy + dy, q * 1.6, q), '#fffaf0', '#e6c9a4', w=.28)
        im = comp.over(im, M.light_layer(e.rgba(grain=.03), 'morn')); u = t - 2; x, y = cx + 24, cy + 10
        d = D(lw=4.4, line='#4a4f58', soft=.9)
        if t == 1: d.part(tube([(x + 10, y - 620), (x + 2, y - 250)], (2, 11), cap=1.), '#eef3f6', '#b9c4cc', w=.5)
        if u >= 0:                                                        # crown, then rings that do not stop
            if u < 3:
                for a in np.linspace(pi + .25, 2 * pi - .25, 7): R = [60, 96, 112][u]; hh = [70, 84, 40][u] * (.8 + .2 * math.sin(a * 5)); q = A([x + math.cos(a) * R, y - 8 + math.sin(a) * R * .24]); d.part(tube([A([x + math.cos(a) * R * .55, y + math.sin(a) * R * .14]), q + [0, -hh * .5], q + [math.cos(a) * 14, -hh]], (11, 6, 3), cap=1.), '#f1f5f8', '#b4bfc8', w=.5)
            if u == 1: [d.part(ell(x + dx, y - hh, 6, 8), '#f1f5f8', '#b4bfc8', w=.4) for dx, hh in ((-60, 150), (14, 186), (76, 140))]
            for k in range(2):
                R = 44 + 84 * (u + 1 - k * 1.2); a = np.linspace(0, 2 * pi, 80)
                if 0 < R < 360: d.stroke(np.c_[x + np.cos(a) * R, y + 8 + np.sin(a) * R * .25], .85 - .25 * k, '#6f7a84', taper=(0, 0))
        im = comp.over(im, d.rgba())
        if u >= 0:                                                        # the colour goes out of it from where the drop fell
            R = [190, 620, 1300, 2600][min(u, 3)]; rr = np.sqrt((X - x) ** 2 + ((Y - y) * 1.5) ** 2); m = smooth(R, R * .62, rr)[..., None]; L = lum(im)[..., None]; gr = np.clip(L * .80 - .04, 0, 1) * rgb('#b9bfc4') * 1.06; im = im * (1 - m) + gr * m
        return im

    def frame(self, f):
        if f < 40: im = self.night(f)
        elif f < 48: im = self.door(f)
        elif f < 60: im = self.profile(f)
        elif f < 74: im = self.morning(f)
        else: im = self.empty(f)
        k = [0, 0, 1.6, .8, .4][f - 58] if 58 <= f < 63 and f >= 60 else 0                                  # morning leaks in
        if f in (60, 61, 62): im = im + rgb('#ffd9a0') * [.40, .20, .08][f - 60]
        if f < 2: im = im * [.45, .8][f]
        amt = float(keys(f, [(0, .78), (74, .78), (79, .0, 'e')])); l = lum(np.clip(im, 0, 2))[..., None]; im = l + (im - l) * (1.22 if f < 74 else 1.1); return props.super8(im.astype(f32), f // 2 * 2 if f < 74 else f, amt, seed=3, warm='#ffe2b8', weave=.8, dust=.7)
