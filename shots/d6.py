"""d6 GO.  80 frames, one violet night among the letters.
0-14  Goku on one knee under a slab; far off a finger rises and a violet dot is born.
15-23 Vegeta: the legs go first.  24-33 his face finds out.  34 the E cracks (one graphic frame).  35-43 he runs at us out of the letters, palm first.
44-45 the palm is on the chest: nothing.  46-47 Goku is gone, bottom left.  48-49 行け.  50-55 white, held: his outline, a line of light through it, a point of light in the chest.
56-59 EGO, white in the moon, a man standing before it.  60 the E breaks (no sound from here).  68-79 GO, clean and white, the man standing, wind."""
import numpy as np, cv2, math, os
from db import W, H as HH, comp, kit
from db.paint import rgb, f32, fbm, smooth, mix, grid
from db.ink import Cel, spl, over_roi
from db.cam import Camera3D
from db.letters_3d import Monument, _hgt, rock_planes
from db import letters as LT, letters_manga as MG
from db.hero3 import *
RAD = math.radians; keys = comp.lerp_keys
VER, FVER = 'v1', 'f7'
VIO, VIOL, DEAD = '#8a2cff', '#c784ff', '#fbf7ff'
G0 = np.array([-6., -58.]); V0 = np.array([7., -38.])
NEU = dict(eo=.52, look=(.0, -.25), brow=(.01, .0), mouth='flat', jaw=0.)
MOVE = dict(eo=1.55, look=(.25, 0), brow=(.085, .06), mouth='frown', jaw=.05, tiny=.48)
I_ = dict(eo=1.25, look=(-.2, -.05), brow=(-.02, .065), mouth='shout', jaw=.02); RUN = dict(eo=1.4, look=(0, 0), brow=(.02, .07), mouth='shout', jaw=.075, tiny=.7); KE = dict(eo=1.15, look=(-.1, 0), brow=(-.05, .07), mouth='shout', jaw=.13); AFTER = dict(eo=1.25, look=(0, 0), brow=(.05, .03), mouth='frown', jaw=.03, tiny=.7); STAND_ = dict(eo=.5, look=(0, -.1), brow=(.02, -.005), mouth='hint', jaw=0.)
LEGS = ('root', 'pel', 'fL', 'fR', 'kL', 'kR', 'tL', 'tR', 'sL', 'sR')


def look(eye, az, pitch, foc):
    a, p = RAD(az), RAD(pitch); d = np.array([math.sin(a) * math.cos(p), math.cos(a) * math.cos(p), math.sin(p)]); return Camera3D(eye, np.asarray(eye, float) + d * 10, foc)


def legs_first(a, b, tl, tu):
    """pose a -> b: the legs at tl, everything above the belt at tu"""
    g = lambda p: POSES[p] if isinstance(p, str) else p; L = mix_pose(g(a), g(b), tl); U = mix_pose(g(a), g(b), tu); q = dict(U)
    for k in LEGS:
        if k in L: q[k] = L[k]
    return q


def tiny(F, cam, g, p, k=4, **kw):
    """a figure too small for its own line weights: drawn k times larger, then reduced"""
    q, _ = cam.project(np.asarray(p, float)); u, v = [float(t) for t in np.ravel(q)[:2]]; c = Camera3D(cam.eye, cam.eye + cam.R[2], cam.f * k, cx=W / 2 - k * (u - cam.cx), cy=HH / 2 - k * (v - cam.cy)); c.R = cam.R
    L = F.draw(c, g, **kw); S = cv2.resize(L, (W // k, HH // k), interpolation=cv2.INTER_AREA); out = np.zeros((HH, W, 4), f32); x0, y0 = int(round(u - W / k / 2)), int(round(v - HH / k / 2)); from db.ink import paste; paste(out, S, x0, y0); return out


class Shot:
    id = 'd6'; n = 80
    finish = dict(bloom=.24, vig=.16, soft=.09, grain=.010, halation='#d8c8ff')
    EVENTS = [(0, 'cut', dict(note='night, stone, breath')), (0, 'slab_strain', dict(dur=14, size=.6, note='stone grinds, dust runs')), (6, 'slab_lift', dict(size=.4)), (10, 'slab_sink', dict(size=.5)), (6, 'red_eyes', dict(size=.3, pan=.5, note='two red eyes light up far off')), (7, 'finger_light', dict(size=.7, pan=.5, note='a violet dot is born and swells, high thin tone')), (11, 'aim_flare', dict(size=.5, pan=.3, note='the flare points at Goku')),
              (15, 'time_slow', dict(dur=19, note='everything stretches, one low heart')), (18, 'foot_dig', dict(size=.5, pan=-.3, note='the legs go before he knows')), (22, 'launch', dict(size=.6)), (27, 'eyes', dict(size=.3, note='he finds out')), (34, 'crack', dict(size=1.3, note='the E cracks: one graphic frame')),
              (36, 'step', dict(size=.5)), (38, 'step', dict(size=.7)), (40, 'step', dict(size=.9)), (42, 'step', dict(size=1.1)), (43, 'rush', dict(size=.8)), (44, 'palm', dict(size=.9, note='flat palm on cloth, then NOTHING for two frames')), (44, 'silence', dict(frames=2)), (46, 'shove', dict(size=1.0, pan=-.4, note='Goku leaves bottom left, the slab bursts')),
              (46, 'voice', dict(text='い')), (48, 'voice', dict(text='け')), (50, 'beam_line', dict(size=1.3, frames=6, note='hair-thin violet line right to left, straight through him; white inverted frame HELD 50-55, the line and the point of light stay')), (53, 'ring_out', dict(size=.3, note='one high tone dying to nothing by 56')),
              (56, 'cut', dict(note='EGO, white in the moon, a small man standing before it: almost nothing')), (60, 'silence', dict(frames=20, note='the E breaks and falls: no sound at all until the cut')), (60, 'e_breaks', dict(size=0., note='picture only; gone and dust clear by 68')), (68, 'go_holds', dict(size=0., note='clean white GO held to 79, wind only as picture'))]

    # ------------------------------------------------------------ plates
    def cams(self):
        z = lambda p: float(_hgt(self.M.Hm, p[0], p[1])); g, v = z(G0), z(V0); self.zg, self.zv = g, v
        return dict(A=look((G0[0] - 3.1, G0[1] - 1.0, g + .58), 92., 4.2, 1500.), B=look((V0[0] - .6, V0[1] - 4.3, v + .42), 5., 10., 1450.), D=look((G0[0], G0[1], g + .92), 15.5, 10., 1300.), E=look((G0[0] + 1.46, G0[1] - 4.05, g + .78), -14., 3.6, 1650.), H=self.M.camera())

    def plate(self, name, cam, t, hide=(), depth=False, **kw):
        p = f'cache/d6_{name}_{VER}.npz'
        if not os.path.exists(p):
            C0 = self.M.RAD.copy()
            if len(hide):                                                       # rocks that would stand in the way of this camera
                for x, y, r in hide: self.M.RAD[(np.hypot(self.M.C[:, 0] - x, self.M.C[:, 1] - y) < r)] = 0.
            im, g = self.M.render(cam, t, **kw); self.M.RAD = C0; zv = g['Z'] * (g['D'] @ cam.R[2].astype(f32)); mp = np.array(list(g['moon']) + [g['hy']], float)
            os.makedirs('cache', exist_ok=True); q = p + f'.{os.getpid()}.npz'; np.savez(q, im=np.clip(im, 0, 8).astype(np.float16), z=np.minimum(zv[::2, ::2], 6e4).astype(f32) if depth else np.zeros(1, f32), m=mp, let=np.packbits(np.isin(g['ID'], (1, 2, 3)) | ((g['ID'] >= 10) & (g['ID'] < 30)))); os.replace(q, p)
        return p

    def get(self, name):
        if name not in self._p:
            q = np.load(f'cache/d6_{name}_{VER}.npz'); self._p[name] = (q['im'].astype(f32), q['m'])
            while len(self._p) > 4: self._p.pop(next(iter(self._p)))
        return self._p[name][0]

    def glow(self, f): return float(comp.ease(np.clip((f - 63) / 11., 0, 1)))

    def setup(self):
        self.M = Monument(); self.M.setup(); self.C = self.cams(); C = self.C; self._p = {}; self._l = {}
        run = [(*(V0 * (1 - u) + G0 * u + np.array([0, 0])), 3.2) for u in np.linspace(0, 1, 12)] + [(-2. - 4 * u, -36. - 22 * u, 3.0) for u in np.linspace(0, 1, 12)]
        self.plate('A', C['A'], 0., hide=[(G0[0], G0[1], 2.6), (-3.2, -59.4, 2.2)], flames=True)
        self.plate('B', C['B'], 15., hide=[(V0[0] - .6 + .1 * k, V0[1] - 4.3 + k * .9, 1.9) for k in range(6)] + [(V0[0] + k, V0[1], 1.6) for k in range(5)])
        for t in (36, 39, 42): self.plate(f'D{t}', C['D'], float(t), hide=run)
        for t in (45,): self.plate(f'E{t}', C['E'], float(t), hide=[(G0[0], G0[1], 3.0), (G0[0] - 2, G0[1] - 2, 2.5), (G0[0] + 1.5, G0[1] - 2.5, 2.4)])
        self.M.clock = lambda t: float(np.interp(t, [55, 57, 59, 60, 61, 62, 63, 64, 65, 66], [.25, .55, .80, 1.10, 1.75, 2.45, 3.20, 4.0, 4.9, 6.1]))                 # the E goes in six frames
        for t in (56, 58, 60, 61, 62, 63, 64, 65): self.plate(f'W{t}', C['H'], float(t), dust=False, glow=1., glow_e={56: 1., 58: 1., 60: 1., 61: .75, 62: .5, 63: .32, 64: .2, 65: .1}[t])
        self.M.parts = 'GO'; self.plate('W66', C['H'], 66., dust=False, glow=1.); self.M.parts = 'EGO'
        self.x, self.y = grid(); r = np.random.default_rng(6); self.motes = r.uniform(0, 1, (90, 6)); self.ego = LT.ego_mask((610, 40, 1000), letters='E', wear=.46, seed=3, chips=1.1)

    # ------------------------------------------------------------ figures
    def light(self, cam, fill=(-.2, .35, .55), k=.65):
        l = cam.R @ self.M.L; m = np.array([l[0], -l[1], -l[2]]); v = m * k + np.array(fill) * (1 - k); return tuple(v / np.linalg.norm(v)), tuple(m)

    def fig(self, key, make):
        """a drawn layer (or (layer, extra...)): kept in memory, and on disk as 8-bit so every frame stays quick"""
        if key in self._l: return self._l[key]
        p = 'cache/d6f_%s_%s.npz' % (FVER, '_'.join(str(k) for k in key)) if isinstance(key, tuple) else 'cache/d6f_%s_%s.npz' % (FVER, key)
        if os.path.exists(p):
            q = np.load(p); x0, y0, x1, y1 = [int(v) for v in q['b']]; L = np.zeros((HH, W, 4), f32); L[y0:y1, x0:x1] = q['a'].astype(f32) / 255.; v = (L, q['e']) if 'e' in q.files and len(q['e']) else L
        else:
            v = make(); L = v[0] if isinstance(v, tuple) else v; e = np.asarray(v[1], float).ravel() if isinstance(v, tuple) else np.zeros(0); ys, xs = np.nonzero(L[..., 3] > .002); bx = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else (0, 0, 2, 2)
            q = p + f'.{os.getpid()}.npz'; np.savez_compressed(q, a=np.clip(L[bx[1]:bx[3], bx[0]:bx[2]] * 255 + .5, 0, 255).astype(np.uint8), b=np.array(bx), e=e); os.replace(q, p); v = (L, e) if len(e) else L
        self._l[key] = v
        while len(self._l) > 5: self._l.pop(next(iter(self._l)))
        return v

    def lay(self, im, L, op=1.):
        a = L[..., 3:4] * op; return im * (1 - a) + L[..., :3] * op

    def slab(self, dz=0., roll=0., at=None, parts=None):
        r = np.random.default_rng(12); n = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (.7, .7, .1), (-.7, .6, .1), (.6, -.75, -.1), (-.7, -.7, .05), (.75, 0, .65), (-.6, .1, .75)]; d = [.24, .24, 1.15, 1.30, .95, .95, 1.22, 1.25, 1.20, 1.26, 1.00, 1.05]
        n = np.array(n, float); n /= np.linalg.norm(n, axis=1)[:, None]; a = RAD(-33) + roll; c, s = math.cos(a), math.sin(a); Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]); b = RAD(90); cz, sz = math.cos(b), math.sin(b); Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]); t = RAD(7); ct, st = math.cos(t), math.sin(t); Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        return (n @ (Rx @ Rz @ Ry).T, np.array(d), np.array([G0[0] + .10, G0[1] + .52, self.zg + 1.08 + dz]) if at is None else at)

    def slabA(self, dz):
        """(normals, offsets, centre, long axis, normal, side) of the stone lying along his back"""
        F = self.atlas(dz); sk = F.skel(); d = unit(sk.chest - sk.pel); h = RAD(-(92 + 84)); rt = np.array([math.cos(h), math.sin(h), 0.]); n = unit(np.cross(rt, d)); n = n if n[2] > 0 else -n; w = np.cross(n, d); d = unit(d + np.array([0, 0, -.10])); n = unit(n - d * (n @ d)); w = np.cross(n, d)
        V = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (.82, .45, .2), (-.8, .55, .15), (.7, -.65, -.1), (-.75, -.62, .1), (.8, .1, .6), (-.72, .1, .65), (.1, .8, .62), (.0, -.85, .55), (.55, .35, -.75)]; D = [.23, .20, 1.30, 1.55, .85, .85, 1.22, 1.40, 1.20, 1.38, 1.10, 1.22, .80, .82, .92]
        V = np.array(V, float); V /= np.linalg.norm(V, axis=1)[:, None]; N = V @ np.array([d, w, n]); c = sk.chest + n * (.155 + .20 + dz * .25) - d * .22; return N, np.array(D), c, d, n, w

    def slabset(self, dz):
        """the slab with its broken edge and the crumbs that came down with it"""
        N, D, c, d, n, w = self.slabA(dz); L = [(N, D, c)]
        for k in range(22):
            r = np.random.default_rng(800 + k); e = k % 3; p = c + (d * r.uniform(-1.4, 1.2) + w * (.82 * r.choice([-1, 1])) if e else d * (1.25 if r.random() < .5 else -1.45) + w * r.uniform(-.8, .8)) + n * r.uniform(-.16, .24); nn, dd = rock_planes(r, r.uniform(.06, .20), .7, 11, .4, False); L.append((nn, dd, p))
        for k in range(12): r = np.random.default_rng(850 + k); nn, dd = rock_planes(r, r.uniform(.05, .20), .7); L.append((nn, dd, np.array([G0[0] + r.normal(.4, .7), G0[1] + r.normal(.5, .9), self.zg + .03])))
        return L

    def atlas(self, dz):
        """Goku bent under the stone: one knee, one fist in the ground, the other hand on the knee"""
        h = RAD(-(92 + 84)); fw = np.array([-math.sin(h), math.cos(h), 0]); rt = np.array([fw[1], -fw[0], 0]); up = dz / .08
        F = Fig('goku', 'torn').pose(P('kneel', root=(0, -.02, -.270 + .02 * up), pel=(34 - 5 * up, 0, 0), wst=(16 - 3 * up, 0, 0), cst=(14 - 2 * up, 0, 0), nck=(-20, 0, 0), hed=(-18, 0, 0), gR='fist', gL='claw', eR=(.3, -1, .2), eL=(.8, -.6, 0))).place(G0[0], G0[1], self.zg, h); sk = F.skel()
        base = np.array([G0[0], G0[1], self.zg]); F.reach(R=dict(p=base + fw * .44 + rt * .20 + [0, 0, .075], part='fist', w=1., body=.2), L=dict(p=base + fw * .30 - rt * .12 + [0, 0, .52], part='palm', palm=(0, 0, -1), fingers=tuple(fw), g='claw', w=1., body=.2)); F.look(base + fw * 8 + np.array([1.2, 0, 1.0])); return F

    # ------------------------------------------------------------ A 0-14
    def A(self, f):
        M, cam = self.M, self.C['A']; g = on(f, 2); im = self.get('A').copy(); lit = float(smooth(7, 14, f)); dz = float(keys(g, [(0, 0.), (4, .012), (7, .075), (9, .080), (12, .02), (14, .0)])) + (.006 if (g // 2) % 2 and 5 < g < 12 else 0.)
        # Frieza, far: he raises one finger, his eyes light, the dot swells and its flare points at Goku
        def fz():
            p = np.array([G0[0] + 13.4, G0[1] - 4.35, self.zg + .34]); h = -RAD(92 + 180 + 64); F = Fig('frieza').pose(mix_pose(POSES['stand'], POSES['finger_aim'], float(comp.ease((g - 1) / 5.)))).place(*p, h); F.tail = lambda sk, ff: tail_curve(sk, 10, kind='low', amp=.8); l, m = self.light(cam, k=.8); LKP = np.array([G0[0] - 1.2, G0[1] + .6, self.zg + 1.2]); F.look(LKP)
            L = tiny(F, cam, g, p + [0, 0, .8], 3, light=l, pal='moon', rim=[dict(dir=m, col='#efe6ff', thr=.36, amt=.95)]); sk = F.skel(); fw = np.array([-math.sin(h), math.cos(h), 0]); rt = np.array([fw[1], -fw[0], 0]); q = lambda X: np.ravel(cam.project(np.asarray(X, float))[0])[:2]
            tips = [sk.spot('finger' + s_) for s_ in 'LR']; tp = max(tips, key=lambda X: (X - sk.pel) @ fw) + fw * .06; lk = unit(LKP - sk.head); lt = unit(np.cross(lk, np.array([0, 0, 1.]))); hd = sk.head + lk * .10 + np.array([0, 0, .012]); return L, np.concatenate([q(tp), q(hd - lt * .042), q(hd + lt * .042)])
        L, e = self.fig(('fz', g), fz); tip, eyes = e[:2], (e[2:4], e[4:6]); im = self.lay(im, L); x, y = self.x, self.y
        ek = float(smooth(5, 8, f))
        if ek > 0:
            for ex, ey in eyes: d = np.hypot(x - ex, y - ey); im = im + ((np.exp(-d / 9.) * .95 + np.exp(-np.abs(y - ey) / 1.6) * np.exp(-np.abs(x - ex) / 26.) * .8) * ek)[..., None] * rgb('#ff1a2e') + (np.exp(-(d / 2.6) ** 2) * 1.3 * ek)[..., None] * rgb('#ffb0a8')
        if f >= 6:
            k = float(np.interp(f, [5, 7, 10, 14], [0., .35, .80, 1.])) * (1 + .06 * math.sin(f * 2.3)); R0 = 7 + 15 * k; d = np.hypot(x - tip[0], y - tip[1])
            im = im + (np.exp(-d / (46 + 120 * k)) * .62 * k)[..., None] * rgb(VIO) + (smooth(R0 + 5, R0 - 3, d) * (.6 + .9 * k))[..., None] * rgb(VIOL) + (smooth(R0 * .62 + 4, R0 * .62 - 3, d) * 1.3)[..., None] * rgb('#fbf4ff')
            tg = np.array([560., 470.]); v = tg - tip; Lg = float(np.hypot(*v)); v = v / Lg; u_ = (x - tip[0]) * v[0] + (y - tip[1]) * v[1]; w_ = -(x - tip[0]) * v[1] + (y - tip[1]) * v[0]; ln = Lg * (.12 + .80 * float(smooth(7, 13, f)))         # the flare: one long blade of light toward him, short ones elsewhere
            bl = np.clip(1 - u_ / ln, 0, 1) * (u_ > -R0); im = im + (np.exp(-np.abs(w_) / (1.2 + 5.5 * bl)) * bl ** .8 * .95 * k)[..., None] * rgb(VIOL) + (np.exp(-np.abs(w_) / (.7 + 1.6 * bl)) * bl * .9 * k)[..., None] * rgb('#fff6ff')
            for an, Ls in ((math.pi / 2 + .12, 120.), (-math.pi / 2 + .12, 120.), (0.06, 210.)):
                c_, s__ = math.cos(an), math.sin(an); uu = (x - tip[0]) * c_ + (y - tip[1]) * s__; ww = -(x - tip[0]) * s__ + (y - tip[1]) * c_; b_ = np.clip(1 - uu / (Ls * k + 1), 0, 1) * (uu > 0); im = im + (np.exp(-np.abs(ww) / (.8 + 2.4 * b_)) * b_ ** 1.4 * .8 * k)[..., None] * rgb(VIOL)
            gy = smooth(tip[1] + 30, HH, y) * np.exp(-np.abs(x - tip[0]) / 420.); im = im + (gy * .10 * k)[..., None] * rgb(VIO)
        # Goku and the slab
        def gk():
            l, m = self.light(cam, fill=(.45, .20, .45), k=.50); F = self.atlas(dz); return F.draw(cam, g, light=l, pal='moon', rim=[dict(dir=m, col='#dccffb', thr=.46, amt=.8), dict(dir=(.95, .1, .1), col=VIOL, thr=.56 - .08 * lit, amt=.15 + .6 * lit)], expr=dict(expr='grit' if g < 12 else 'rage', dmg=.6, light='moon'))
        key = int(round(dz * 1000)); S = self.fig(('slab', key), lambda: M.stones(cam, self.slabset(dz), ink=2.6, pal=('#05040b', '#1a1438', '#3a2f6c'))[0]); Fr = self.fig(('slabf', key), lambda: M.stones(cam, self.slabset(dz)[:1], ink=2.6, pal=('#05040b', '#1a1438', '#3a2f6c'))[0]) if False else None; G = self.fig(('gk', g), gk)
        im = self.lay(im, S); im = self.lay(im, G)
        im = self.trickle(im, cam, f, dz); z = 1.02 + .0026 * f; return comp.cam2d(im, 960 + 10 + f * .8, 540 + (1.2 * math.sin(f * 1.9) if 5 < f < 12 else 0), z)

    def trickle(self, im, cam, f, dz):
        c = Cel(W, HH); N, D, ctr, d, n, w = self.slabA(dz)
        for k in range(18):
            r = np.random.default_rng(60 + k); u = r.uniform(-1.2, 1.15); p = ctr + d * u + w * .84 * r.choice([-1, 1]) - n * .16; a = (f * .9 + r.uniform(0, 16)) % 16; amp = 1. if 4 < f < 13 else .45; z0 = p[2] - .10 * a; z1 = max(self.zg, z0 - .07 - .012 * a)
            if z0 < self.zg + .05: continue
            P_ = cam.project(np.array([[p[0] + .01 * a, p[1], z0], [p[0] + .02 * a, p[1], (z0 + z1) / 2], [p[0] + .03 * a, p[1], z1]]))[0]; c.line(P_, r.uniform(3, 8), '#a695e2', taper=(.5, .8), seed=k, alpha=.22 * amp)
        for k in range(30):
            r = np.random.default_rng(160 + k); p = ctr + d * r.uniform(-1.3, 1.2) + w * .85 * r.choice([-1, 1]) - n * .18; a = (f + r.uniform(0, 20)) % 20; z = p[2] - .011 * a * a
            if z < self.zg: continue
            q = cam.project(np.array([p[0] + r.normal(0, .02) * a, p[1] + r.normal(0, .01) * a, z]))[0]; R = r.uniform(1.2, 4.5); th = np.sort(r.uniform(0, 6.28, 6)); c.fill(np.c_[q[0] + np.cos(th) * R, q[1] + np.sin(th) * R], '#2c2456'); c.line(np.array([q + [-R * .6, -R * .6], q + [R * .5, -R * .7]]), 1.2, '#b5a3ee', seed=k)
        L = c.rgba(); L = cv2.GaussianBlur(L, (0, 0), .7); return self.lay(im, L)

    def air(self, im, f, n=60, dx=0., slow=.25, col='#b7a4f0', seed=0):
        """things hanging in the slowed air: dust and chips, some out of focus"""
        c1, c2 = Cel(W, HH), Cel(W // 2, HH // 2, ss=1)
        for k in range(n):
            m = self.motes[k]; z = .4 + 2.6 * m[2] ** 2; x = (m[0] * 2300 - 190 - dx * z * .8 + math.sin(f * slow * .2 + m[3] * 6) * 6) % 2300 - 190; y = (m[1] * 1200 - 60 - f * slow * 3. * z) % 1200 - 60; R = (1. + 2.4 * m[4] ** 2) * z; th = np.sort(m[5] * 6.28 + np.array([0, 1.1, 2.2, 3.4, 4.6, 5.4])) + f * slow * .12 * (m[3] - .5); P = np.c_[x + np.cos(th) * R * 1.2, y + np.sin(th) * R]
            if z > 1.9: c2.fill(P / 2, col, .5)
            else: c1.fill(P, col if m[3] > .35 else '#0d0a1c', .8)
        im = self.lay(im, c1.rgba()); q = cv2.resize(cv2.GaussianBlur(c2.rgba(), (0, 0), 3.2), (W, HH)); return self.lay(im, q, .7)

    # ------------------------------------------------------------ B 15-23 the legs, 24-33 the face
    def B1(self, f):
        cam = self.C['B']; g = on(f, 2) if f < 21 else f; u = f - 15; im = self.get('B').copy(); tl = float(keys(g, [(15, 0.), (17, .18), (18, .75), (20, 1.0), (23, 1.0)], lambda t: t)); tu = float(keys(g, [(15, 0.), (19, .05), (21, .45), (23, .95)], lambda t: t)); run = float(keys(g, [(15, 0.), (18, .04), (20, .42), (22, 1.35), (23, 2.1), (24, 3.1)]))
        def vg():
            l, m = self.light(cam, fill=(-.35, .35, .6), k=.45); d = np.array([math.sin(RAD(5 + 80)), math.cos(RAD(5 + 80))]); p = V0 + d * (run - .35); F = Fig('vegeta', 'shattered').pose(legs_first('stagger', 'run_push', tl, tu)).place(p[0], p[1], self.zv, RAD(-(5 + 98))); F.blow(-.5 * run, 0, .15)
            if tu < .3: F.look(F.skel().head + np.array([.5, -.5, -.8]))
            return F.draw(cam, g, light=l, pal='night', rim=[dict(dir=m, col='#e4d8ff', thr=.42, amt=.95), dict(dir=(-.95, -.05, .05), col='#ff9a5c', thr=.50, amt=.55)], expr=dict(expr=NEU, scratch=1, wind=min(1., run * .6), light='night'))
        L = self.fig(('v1', g), vg)
        if f >= 18: im = self.dig(im, cam, f)
        if f >= 22: L2 = kit.smear(L, 90 * (f - 21), -6, 9); im = self.lay(im, L2, .75); im = self.lay(im, L, .9)
        else: im = self.lay(im, L)
        im = self.air(im, f, 70); z = 1.03 + .004 * u; return comp.cam2d(im, 960 + 4 * u, 548, z)

    def dig(self, im, cam, f):
        """the foot digs: stones and dust thrown back, slow"""
        a = f - 17.; c = Cel(W, HH); base = np.array([V0[0] - .55, V0[1] - .1, self.zv])
        for k in range(34):
            r = np.random.default_rng(500 + k); v = np.array([-r.gamma(2., .42), r.normal(0, .3), r.gamma(2., .42)]); t = a / 24. * 1.3; p = base + v * t * 1.4 + np.array([0, 0, -2.2 * t * t]); q, z = cam.project(p); q = np.ravel(q); R = r.gamma(1.3, .016) * cam.f / float(np.ravel(z)[0]) + 1.; th = np.sort(r.uniform(0, 6.28, 6)) + a * .1; P = np.c_[q[0] + np.cos(th) * R * 1.25, q[1] + np.sin(th) * R]; c.fill(P, '#1b1638'); c.fill((P - q) * [.8, .5] + q + [-R * .1, -R * .35], '#8e7bd6')
        im = self.lay(im, c.rgba())
        from db.phys_dust import lobes_layer
        rows = []
        for k in range(12): r = np.random.default_rng(700 + k); t = a / 24.; s = r.uniform(.2, 1.1); rows.append((base[0] - s * (1 - math.exp(-3 * t)) * 1.3, base[1] + r.normal(0, .18), base[2] + .08 + r.uniform(0, .35) * t * 2, r.uniform(.07, .17) * (1 + 3 * t), k * 37, t, 0, min(.9, t * 1.1), 0, self.zv))
        l, m = self.light(cam); return self.lay(im, lobes_layer(cam, np.array(rows), dict(lit='#9b87dc', mid='#54458f', dark='#271e52', ink='#0d0a1c', rim='#efe7ff'), light=(m[0], -m[1], .3), ss=1.25, ink=.8, rim=(m[0], -m[1]), rim_amt=.5), .92)

    def B2(self, f):
        g = on(f, 2) if f < 27 else on(f - 1, 2) + 1; u = f - 24; wake = f >= 27; P0 = self.get('B'); sx = 140 * u; k = 1.65
        bg = comp.cam2d(P0, 700 + sx * .9, 430, k); bg = kit.smear(cv2.resize(bg, (W // 2, HH // 2), interpolation=cv2.INTER_AREA), 46, 0, 11); bg = cv2.resize(cv2.GaussianBlur(bg, (0, 0), 2.2), (W, HH), interpolation=cv2.INTER_LINEAR) * .92
        def vg():
            F = Fig('vegeta', 'shattered').pose(sprint(.06 + .05 * (g - 24))).place(0, 0, 0, RAD(-112.)); sk = F.skel(); cam = Camera3D(tuple(sk.head + np.array([.42, -1.12, -.01])), tuple(sk.head + np.array([.15, 0, -.045])), 2150.); F.blow(-1.0, 0, .1); F.look(sk.head + np.array([2.4, -1.75, -.06 if wake else -.60]))
            return F.draw(cam, g, light=(-.25, .30, .75), pal='night', rim=[dict(dir=(.70, .45, -.45), col='#b7a6ee', thr=.68, amt=.5), dict(dir=(-.95, .05, 0), col='#ff9a5c', thr=.60, amt=.5)], expr=dict(expr=MOVE if wake else NEU, scratch=.5, wind=.95 + .1 * math.sin(g), light='night', f=g))
        L = self.fig(('v2', g, wake), vg); bob = 7 * math.sin(f * 1.25); pop = math.exp(-(f - 27) / 1.6) if wake else 0.
        im = bg
        ln = Cel(W, HH, ss=1); kit.speed_lines(ln, 0., 34, 3.2, (300, 1500), '#d9cbff', 40 + f, alpha=.30); kit.speed_lines(ln, 0., 22, 5., (300, 1300), '#0b0918', 90 + f, alpha=.5); im = self.lay(im, cv2.GaussianBlur(ln.rgba(), (0, 0), .8)); im = self.lay(im, L); im = self.air(im, f, 46, dx=-150 * u, slow=.2)
        if wake and f < 29: im = im * (1 + .22 * pop)
        z = 1.04 + .006 * u + .035 * pop; return comp.cam2d(im, 960 + 4 * u, 530 + bob, z, .004 * u)

    # ------------------------------------------------------------ 34
    def crack(self):
        """one frame: black, the letter in white, broken through"""
        r = np.random.default_rng(8); B = []; mains = []
        def bolt(p, ang, L, w, d=0):
            P = [np.array(p, float)]; n = max(4, int(L / 52))
            for _ in range(n): a2 = ang + r.normal(0, .50); P.append(P[-1] + np.array([math.cos(a2), math.sin(a2)]) * L / n)
            P = np.array(P); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; hw = w * .5 * np.linspace(1, .04, len(P)) ** .75 * r.uniform(.6, 1.45, len(P)); B.append((P, N, hw))
            if d < 2:
                for i in range(2, len(P) - 1, 2 + d): bolt(P[i], ang + r.choice([-1, 1]) * r.uniform(.5, 1.1), L * r.uniform(.18, .42), w * .42, d + 1)
        [(mains.append(len(B)), bolt(*q)) for q in (((1000, 690), math.pi * 1.04, 560, 66), ((985, 690), -1.32, 640, 46), ((1000, 700), .30, 520, 34))]; a = self.ego; y, x = self.y, self.x; b2 = B[mains[1]][0]; xb = np.interp(y[:, 0], b2[::-1, 1], b2[::-1, 0])
        main = B[0][0]; yc = np.interp(x[0], main[::-1, 0], main[::-1, 1]); up = (y < yc[None]) & (x < xb[:, None]) & (x < 1080); sh = cv2.warpAffine(a * up, np.float32([[1, 0, -22], [0, 1, 10]]), (W, HH)); lo = a * (~up); a = np.maximum(sh, lo)
        im = np.ones((HH, W, 3), f32) * rgb('#040308'); im = im * (1 - a[..., None]) + rgb(DEAD) * a[..., None]; c1, c2 = Cel(W, HH), Cel(W, HH)
        for P, N, hw in B: c1.fill(np.vstack([P + N * (hw + 7)[:, None], (P - N * (hw + 7)[:, None])[::-1]]), VIO); c2.fill(np.vstack([P + N * hw[:, None], (P - N * hw[:, None])[::-1]]), '#040308')
        for k in range(60): t = r.uniform(0, 6.28); d = r.gamma(2., 130); R = r.gamma(1.2, 6) + 2; q = np.array([1000 + math.cos(t) * d * 1.3, 690 + math.sin(t) * d * .8]); th = np.sort(r.uniform(0, 6.28, 5)); c2.fill(np.c_[q[0] + np.cos(th) * R * 1.4, q[1] + np.sin(th) * R], DEAD if r.random() < .6 else VIO)
        for c in (c1, c2): q = c.rgba(); im = im * (1 - q[..., 3:4]) + q[..., :3]
        return im

    # ------------------------------------------------------------ D 35-43
    def D(self, f):
        cam = self.C['D']; im = self.get('D%d' % (36 if f < 39 else 39 if f < 42 else 42)).copy(); u = (f - 35) / 8.; ds = [15.5, 12.6, 10.0, 7.9, 6.1, 4.6, 3.4, 2.35, 1.42][f - 35]; th = RAD(float(keys(f, [(35, 7.), (41, 11.), (43, 15.)]))); a = np.array([math.sin(th), math.cos(th)]); p = G0 + a * ds; jump = 0.10 * abs(math.sin(f * 1.5))
        def vg():
            l, m = self.light(cam, fill=(.25, .25, .7), k=.40); F = Fig('vegeta', 'shattered'); ps = sprint(.12 + .21 * (f - 35)) if f < 41 else mix_pose(sprint(.12 + .21 * 6), POSES['shove'], {41: .45, 42: .85, 43: 1.}[f]); F.pose(ps).place(p[0], p[1], self.zg + (jump if f < 42 else .02), RAD(-(math.degrees(math.atan2(a[0], a[1])) + 180)))
            if f >= 41: F.reach(R=dict(p=cam.eye + np.array([a[0], a[1], 0]) * {41: 2.75, 42: 1.62, 43: .62}[f] + np.array([.10, -.03, {41: .0, 42: .10, 43: .02}[f]]), part='palm', palm=(-a[0], -a[1], 0), fingers=(0, 0, 1), g='splay', w=1.)); F.look(cam.eye)
            F.blow(0, 1.6, .2); return F.draw(cam, f, light=l, pal='night', rim=[dict(dir=m, col='#efe4ff', thr=.42, amt=.95), dict(dir=(.95, .05, .05), col=VIOL, thr=.50, amt=.6)], expr=dict(expr=dict(MOVE, tiny=.6, look=(0, 0)) if f < 42 else RUN, scratch=1, wind=1., light='night', f=f))
        L = self.fig(('v3', f), vg)
        im = self.wake(im, cam, f, a, ds); im = self.lay(im, L)
        if f >= 39:
            k = (f - 38) / 5.; fl = MG.flash(960, 560, 700 - 60 * k, .66, 260, 60 + f, w=3 + 7 * k); im = im * (1 - fl[..., None] * .85 * k) + rgb('#0a0716') * fl[..., None] * .85 * k; w_ = MG.flash(960, 560, 760, .66, 110, 80 + f, w=2 + 4 * k); im = im + w_[..., None] * rgb('#cdb6ff') * .35 * k
        s = kit.shake(f, 2 + 7 * u, 5); z = 1.14 - .10 * u; im = comp.cam2d(im, 985 - 20 * u + s[0], 520 + 8 * u + s[1], z)
        if f >= 42: im = kit.zoomblur(cv2.resize(im, (W // 2, HH // 2), interpolation=cv2.INTER_AREA), 480, 280, .05, 5) if False else im
        return im

    def wake(self, im, cam, f, a, ds):
        from db.phys_dust import lobes_layer
        rows = []; nrm = np.array([-a[1], a[0]])
        for k in range(40):
            d = ds + .5 + k * .55; age = (k * .55) / 3.2 / 24. * 6; r = np.random.default_rng(900 + int(d * 2.2) * 7 + k % 2); p = G0 + a * d + nrm * r.normal(0, .22 + .5 * age)
            if d > 23.5: continue
            rows.append((p[0], p[1], self.zg + .12 + .35 * age, (.16 + .5 * age) * r.uniform(.7, 1.2), int(d * 100) % 9000, age * 2, 0, min(.95, age * 1.2), 0, self.zg))
        l, m = self.light(cam)
        if rows: im = self.lay(im, lobes_layer(cam, np.array(rows), dict(lit='#8f7bd4', mid='#4d3f88', dark='#241c4c', ink='#0d0a1c', rim='#efe7ff'), light=(m[0], -m[1], .3), ss=1.0, ink=.7, rim=(m[0], -m[1]), rim_amt=.5, K2=3), .85)
        return im

    # ------------------------------------------------------------ E 44-49 the shove
    VD = np.array([math.sin(RAD(58)), math.cos(RAD(58))])                       # from Goku to Vegeta

    def E(self, f):
        C0 = self.C['E']; zz, ax, ay = 1.34, 870., 610.; cam = Camera3D(C0.eye, C0.eye + C0.R[2], C0.f * zz, cx=960 + (960 - ax) * zz, cy=540 + (540 - ay) * zz); cam.R = C0.R; im = comp.cam2d(self.get('E45'), ax, ay, zz); g = 44 if f < 46 else f; k = max(0, f - 45); gp = G0 + np.array([-.62, -.62]) * [0, .55, 1.55, 3.0][min(k, 3)]
        def two():
            l, m = self.light(cam, fill=(-.15, .30, .6), k=.45); Vp = G0 + self.VD * 1.27; Vh = RAD(-(58 + 180)); Gh = RAD(-84.)
            Gk = Fig('goku', 'torn').pose('kneel_up' if k == 0 else mix_pose(POSES['kneel_up'], POSES['thrown'], min(1., .6 + .3 * k))).place(gp[0], gp[1], self.zg + [0, -.04, -.14, 0][min(k, 3)], Gh, pitch=[0, -.18, -.5, -.6][min(k, 3)]); V = Fig('vegeta', 'shattered').pose('shove').place(Vp[0], Vp[1], self.zg, Vh)
            if k == 0: touch(V, 'R', Gk, 'chest'); self.pc = None
            else: V.reach(R=dict(p=G0 + self.VD * .30 if False else np.array([*(G0 + self.VD * .06), self.zg + 1.02]), part='palm', palm=(-self.VD[0], -self.VD[1], 0), fingers=(0, 0, 1), g='splay', w=1.))
            V.look(np.array([*gp, self.zg + 1.1])); Gk.look(np.array([Vp[0] + .55, Vp[1] - .75, self.zg + 1.40])); V.blow(-.9, -.5, .2); Gk.blow(.6 * k, .6 * k, 0)
            rim = [dict(dir=m, col='#efe4ff', thr=.42, amt=.95), dict(dir=(.95, .05, .10), col=VIOL, thr=.50, amt=.7)]
            return compose([(Gk, dict(light=l, pal='night', rim=rim, expr=dict(expr='shock', dmg=.6, light='moon'))), (V, dict(light=l, pal='night', rim=rim, expr=dict(expr=I_ if k < 2 else KE, scratch=1, wind=1., light='night')))], cam, g)
        L = self.fig(('e', g), two); S = self.slabs(cam, f)
        im = self.lay(im, S)
        if 0 < k < 3: B = kit.smear(L, -70 * k, 46 * k, 9); im = self.lay(im, B, .55)
        im = self.lay(im, L)
        if k == 1:
            q = cam.project(np.array([*G0, self.zg + .95]))[0]; c = Cel(W, HH, ss=1); kit.focus_lines(c, q[0], q[1], 330, 1500, 120, 5., '#f3ebff', 5, ry=.9); im = self.lay(im, c.rgba(), .55); im = im * 1.12
        s = kit.shake(f, 13., 3, 2.2, 46) if f >= 46 else (0, 0); return comp.cam2d(im, 960 + s[0], 540 + s[1], 1.02 + .02 * (f >= 46))

    def slabs(self, cam, f):
        """the slab behind him: whole, then bursting as he goes through it"""
        c0 = np.array([G0[0] - .78, G0[1] - .20, self.zg + .78])
        if f < 46: return self.fig('slabE', lambda: self.M.stones(cam, [self._slabE(c0)], ink=2.4, pal=('#05040b', '#161030', '#322860'))[0])
        a = (f - 45) / 24. * 2.2; L = []
        for k in range(16):
            r = np.random.default_rng(40 + k); nn, dd = rock_planes(r, r.uniform(.12, .42), .55, 10, .4, False); v = np.array([-r.gamma(2., 1.5), -r.gamma(1.8, 1.2), r.uniform(-.5, 3.)]); th = a * r.uniform(-4, 4); c, s = math.cos(th), math.sin(th); Rm = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]); L.append((nn @ Rm, dd, c0 + r.normal(0, .4, 3) * [.7, .7, .8] + v * a + np.array([0, 0, -4.9 * a * a * .5])))
        return self.fig(('burst', f), lambda: self.M.stones(cam, L, ink=2.0, pal=('#05040b', '#161030', '#3a2f70'))[0])

    def _slabE(self, c):
        n, d, _ = self.slab(); a = RAD(62); ca, sa = math.cos(a), math.sin(a); R_ = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]]); return (n @ R_.T, d * np.array([1, 1, .95, .95, .9, .9] + [.95] * 6), c)

    def E2(self, f):
        """行け: POV of the one who is thrown away"""
        g = f; k = f - 48; bg = comp.cam2d(self.get('D42'), 900, 470, 1.25); bg = kit.zoomblur(cv2.resize(cv2.GaussianBlur(bg, (0, 0), 3), (W // 2, HH // 2), interpolation=cv2.INTER_AREA), 520, 260, .10, 9); bg = cv2.resize(bg, (W, HH), interpolation=cv2.INTER_LINEAR) * .85
        def vg():
            F = Fig('vegeta', 'shattered').pose('shove').place(0, 0, 0, RAD(-168)); sk = F.skel(); off = np.array([.30, -1.62 - .22 * k, .02]); cam = Camera3D(tuple(sk.head + off), tuple(sk.head + np.array([.16, 0, -.07])), 1900.); F.look(sk.head + off * 2); F.blow(.2, 1.4, .2)
            F.reach(R=dict(p=sk.head + np.array([.34, -.68, -.13]), part='palm', palm=(0, -1, .12), fingers=(.1, .1, 1), g='splay', w=1.)); return F.draw(cam, g, light=(.15, .30, .75), pal='night', rim=[dict(dir=(-.5, .55, -.5), col='#efe4ff', thr=.45, amt=.9), dict(dir=(.95, .05, .05), col=VIOL, thr=.44, amt=.95)], expr=dict(expr=KE, scratch=1, wind=1., light='night', f=f))
        im = self.lay(bg, self.fig(('ke', f), vg)); fl = MG.flash(1010, 470, 720, .7, 220, 30 + f, w=6.); im = im * (1 - fl[..., None] * .7) + rgb('#0a0716') * fl[..., None] * .7; im = im + (smooth(1200, 1920, self.x) * (.16 + .22 * k))[..., None] * rgb(VIO); return comp.cam2d(im, 960, 540, 1.07 - .03 * k)

    # ------------------------------------------------------------ 50-55 held: white, his outline, a line of light through it
    def S(self, f):
        def mk():
            cam = Camera3D((0, -6, 1.0), (0, 0, 1.0), 4400., cx=1015, cy=650); F = Fig('vegeta', 'shattered').pose(P('shove', root=(-.02, .08, -.08), pel=(8, -4, 14), wst=(-10, 0, 6), cst=(-14, 0, 8), nck=(6, 0, -8), hed=(2, 0, -8), hL=(.62, -.62, -.38), hR=(-.02, 1.0, .08), fL=(.11, .26, .045), fR=(.07, -.30, .08))).place(0, 0, 0, RAD(80)); F.blow(.1, 0, .1); sk = F.skel()
            L = F.draw(cam, 50, light=(-.3, .4, .6), pal='night', expr=dict(expr=dict(KE, mouth='part', jaw=0.), wind=0., light='night')); q = np.ravel(cam.project(sk.chest + np.array([0, 0, .10]))[0])[:2]; return L, np.array(q)
        L, c = self.fig(('sil',), mk); x, y = self.x, self.y; a = smooth(.35, .75, cv2.morphologyEx(L[..., 3], cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))); cx, cy = float(c[0]), float(c[1]); u = f - 50; tl = .020; yl = cy + (x - cx) * -tl; d = np.abs(y - yl); r = np.hypot(x - cx, y - cy)
        bg = np.ones((HH, W, 3), f32) * np.array([1.02, 1.0, 1.04], f32); bg = bg * (1 - (.10 * smooth(380, 1250, r))[..., None] * np.array([1., 1.25, .35], f32)); fl = self.fig(('silfl',), lambda: np.dstack([MG.flash(cx, cy, 620, .8, 210, 4, w=4.)] * 4)); bg = bg * (1 - fl[..., 3:4] * .16 * np.array([.9, 1.2, .35], f32))
        w = {0: 15., 1: 8.5, 2: 5.6, 3: 4.8, 4: 4.2, 5: 3.4}[u]                                         # on white the light is a violet line with a burning core
        bg = bg * (1 - (np.exp(-d / (w * 4.5)) * .50)[..., None] * (1 - rgb(VIOL))); k = np.clip(w * 1.5 - d + .5, 0, 1)[..., None]; bg = bg * (1 - k) + rgb(VIO) * k; k = np.clip(w * .42 - d + .5, 0, 1)[..., None]; bg = bg * (1 - k) + np.array([1.25, 1.2, 1.3], f32) * k
        im = bg * (1 - a[..., None]) + rgb('#07040c') * a[..., None]
        # through him: white-hot, the dark of the body burns violet along it
        gl = np.exp(-d / (9 + w * 2.2)) * .85 + np.exp(-r / 70.) * 1.1 + np.exp(-r / 210.) * .28; im = im + (a * gl)[..., None] * rgb('#7a1dff') * .85; k = (np.clip(w * .95 - d + .5, 0, 1) * a)[..., None]; im = im * (1 - k) + np.array([1.2, 1.1, 1.3], f32) * k
        R0 = {0: 46., 1: 36., 2: 30.}.get(u, 27.) + 1.2 * math.sin(u * 2.1); ring = np.exp(-((r - R0 * 1.55) / 5.5) ** 2); im = im * (1 - ring[..., None] * .95) + rgb(VIO) * ring[..., None] * .95; pt = smooth(R0 + 2.5, R0 - 2.5, r)[..., None]; im = im * (1 - pt) + np.array([1.35, 1.3, 1.4], f32) * pt
        for an, Ls, ww in ((math.pi / 2, 230., 7.), (-math.pi / 2, 150., 7.), (math.pi * .25, 110., 5.), (math.pi * .75, 110., 5.), (-math.pi * .25, 110., 5.), (-math.pi * .75, 110., 5.)):
            c_, s_ = math.cos(an), math.sin(an); uu = (x - cx) * c_ + (y - cy) * s_; vv = np.abs(-(x - cx) * s_ + (y - cy) * c_); b_ = np.clip(1 - uu / Ls, 0, 1) * (uu > 0); k = np.clip(ww * b_ ** 1.3 - vv + .5, 0, 1) * (b_ > 0); im = im * (1 - k[..., None]) + (np.array([1.3, 1.25, 1.35], f32) * a[..., None] + rgb(VIO) * (1 - a[..., None])) * k[..., None]
        # where it leaves him, splinters of light and cloth fly on along the line
        cl = Cel(W, HH); ex = cx - 70
        for k in range(54):
            rr = np.random.default_rng(90 + k); t = (2.2 + u) * rr.uniform(.5, 1.3); px = ex - 46 * t * rr.uniform(.5, 2.2); py = cy + (px - cx) * -tl + rr.normal(0, 4 + 9 * t); Ls = rr.uniform(6, 34); hh = rr.uniform(1.2, 5.); an = math.pi + rr.normal(0, .25); dd = np.array([math.cos(an), math.sin(an)]); nn = np.array([-dd[1], dd[0]]); q = np.array([px, py]); cl.fill(np.array([q - dd * Ls * .4, q + nn * hh, q + dd * Ls, q - nn * hh]), VIO if rr.random() < .6 else '#07040c')
        q = cl.rgba(); q = q * (1 - a[..., None] * .0); im = im * (1 - q[..., 3:4]) + q[..., :3]
        s = kit.shake(f, 7., 3, 1.0, 50); return comp.cam2d(im.astype(f32), 960 + s[0], 540 + s[1], 1.0 + .006 * u)

    # ------------------------------------------------------------ W 56-79 EGO -> GO
    def mound(self, cam):
        """the rise he stands on: (layer, top point)"""
        if not hasattr(self, '_mt'): p = self.M.ground_at(cam, 1150, 735 + 3.5 * cam.f / 36.); self._mt = np.array([p[0], p[1], p[2] + 2.35])
        t = self._mt
        def mk():
            L = []; r = np.random.default_rng(21); base = np.array([t[0], t[1], t[2] - 2.35])
            def block(c, R, h, n=9, bt=(.22, .42), st=1.):
                th = np.linspace(0, 2 * math.pi, n, endpoint=False) + r.normal(0, .16, n); N = np.c_[np.cos(th), np.sin(th) / st, r.uniform(*bt, n)]; ln = np.linalg.norm(N, axis=1); D = R * r.uniform(.88, 1.05, n) / ln; N /= ln[:, None]; top = np.array([r.normal(0, .02), r.normal(0, .02), 1.]); top /= np.linalg.norm(top); L.append((np.vstack([N, top, [0, 0, -1]]), np.r_[D, h, .12], c))
            block(base, 1.75, 2.35); block(base + [-1.75, .3, 0], 1.45, 1.35, st=1.2); block(base + [1.7, .2, 0], 1.35, 1.0, st=1.2); block(base + [.5, -1.3, 0], 1.2, .70)
            for k in range(16): R = r.gamma(1.6, .16) + .12; nn, dd = rock_planes(r, R, r.uniform(.5, .8)); L.append((nn, dd, base + np.array([r.normal(0, 2.3), r.normal(-.6, 1.1), R * .15])))
            return self.M.stones(cam, L, ink=1.2, pal=('#07060f', '#1b1538', '#332961'), edge=.3)[0]
        return self.fig(('mound',), mk), t

    def shards(self, cam, f):
        """what the E throws as it goes"""
        a = (f - 59.4) / 24. * 2.6; L = []
        for k in range(120):
            r = np.random.default_rng(6100 + k); R = min(1.5, r.gamma(1.4, .26) + .12); nn, dd = rock_planes(r, R, .8, 10, .4, False); p0 = np.array([r.uniform(0, 20), r.uniform(-.5, 4), r.uniform(3, 29)]); v = np.array([-r.gamma(2., 3.2) + 3.0, -r.gamma(1.6, 1.6), r.uniform(-3, 7)]); p = p0 + v * a + np.array([0, 0, -4.9 * a * a]); th = a * r.uniform(-5, 5); c, s = math.cos(th), math.sin(th); Rm = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            if p[2] > -.5 and (f > 60 or k % 4 == 0): L.append((nn @ Rm, dd, p))
        return self.M.stones(cam, L, ink=1.0, pal=('#15102e', '#4a3f86', '#b5a8de'))[0] if L else np.zeros((HH, W, 4), f32)

    def fall_dust(self, cam, f):
        from db.phys_dust import lobes_layer
        a = f - 59.5; rows = []; er = float(np.clip((a - 3.4) / 4.4, 0, 1.05)) ** 1.1
        for k in range(130):
            r = np.random.default_rng(7300 + k); t0 = r.uniform(0, 2.6); b = a - t0
            if b <= 0: continue
            g = 1 - math.exp(-.55 * b); X = r.uniform(-4, 25) - (1.2 + 2.0 * r.random()) * b + (r.uniform(-4, 4)) * g; Y = r.choice([r.uniform(-5, -1), r.uniform(1, 9)], p=[.72, .28]) - .5 * b; R = r.uniform(1.5, 3.6) * (.35 + .9 * g); Z = r.uniform(.2, 11.) * g * r.random() + R * .4 + .10 * b; rows.append((X, Y, Z, R, 300 + k * 29, .10 * b, .5, min(1.1, er * r.uniform(.75, 1.35)), 0., .1))
        if not rows or er >= 1.04: return np.zeros((HH, W, 4), f32)
        mp = self._mp; l = np.array([mp[0] - 600., mp[1] - 620.]); l /= np.hypot(*l); L = lobes_layer(cam, np.array(rows), dict(lit='#bcaef2', mid='#8575c6', dark='#4b3e8a', ink='#1f1848', rim='#f6f0ff'), light=(l[0] * .8, l[1] * .8 - .1, .22), ss=1.15, ink=.6, ink_in=.45, t_lit=.42, t_dark=.04, group=.52, rim=(l[0], l[1]), rim_amt=.5, micro=22., rough=.16); return L * float(np.clip((8.6 - a) / 2.2, 0, 1))

    def rubble(self, cam):
        def mk():
            L = []
            for k in range(70): r = np.random.default_rng(8200 + k); R = min(1.5, r.gamma(1.5, .34) + .15); nn, dd = rock_planes(r, R, r.uniform(.5, .85)); X, Y = r.uniform(-5, 23), r.normal(1.5, 2.6); L.append((nn, dd, np.array([X, Y, float(_hgt(self.M.Hm, X, Y)) + R * .12])))
            Q = self.M.stones(cam, L, ink=0., pal=('#1a1440', '#352c68', '#6a5cae'))[0]; return np.dstack([Q[..., :3] * .62 + rgb('#5a4a9a') * .38 * Q[..., 3:4], Q[..., 3]])
        return self.fig(('rubble',), mk)

    def Wd(self, f):
        cam = self.C['H']; t = 56 if f < 58 else 58 if f < 60 else f if f < 66 else 66; im = self.get(f'W{t}').copy(); self._mp = self._p[f'W{t}'][1]; g = on(f, 3); md, top = self.mound(cam)
        if f >= 63: im = self.lay(im, self.rubble(cam), float(smooth(62.5, 65, f)))
        if 60 <= f < 67: im = self.lay(im, self.fig(('shd', f), lambda: self.shards(cam, f)))
        if 60 <= f < 68: im = self.lay(im, self.fig(('fd', f), lambda: self.fall_dust(cam, f)))
        im = self.lay(im, md)
        def vg():
            l, m = self.light(cam, fill=(-.1, .35, .6), k=.6); F = Fig('vegeta', 'shattered').pose('contra').place(*top, RAD(-122 - 10)); F.blow(-1.2 - .4 * math.sin(g), .2, .1); L = tiny(F, cam, g, top + [0, 0, .85], 4, light=l, pal=dict(base='night', mul=.55), rim=[dict(dir=m, col='#f6f0ff', thr=.40, amt=.9)], expr=dict(expr=STAND_, wind=.8, light='night')); return kit.rim(L, 1.5, 1.0, '#f4ecff', .9, soft=.4)
        im = self.lay(im, self.fig(('wv', g), vg))
        c = Cel(W, HH, ss=1)                                                    # wind along the ground
        for k in range(40):
            r = np.random.default_rng(1200 + k); v = r.uniform(9, 26); px = (r.uniform(0, 2600) + f * v) % 2600 - 340; py = 790 + 280 * r.random() ** 1.4; ln = r.uniform(60, 300) * (py - 640) / 260.; y0 = py + 3 * math.sin(f * .3 + k); P_ = np.array([(px, y0), (px + ln * .4, y0 - 3), (px + ln, y0 + r.normal(0, 4))]); c.line(spl(P_, n=7), r.uniform(.8, 3.0) * (py - 620) / 260., '#cabdf2', taper=(.5, .7), seed=k, alpha=.20 * r.uniform(.4, 1))
        im = self.lay(im, cv2.GaussianBlur(c.rgba(), (0, 0), 1.0)); im = self.flecks(im, f); e = comp.ease(np.clip((f - 66) / 13., 0, 1)); s = kit.shake(f, 11., 6, 2.6, 60) if f >= 60 else (0, 0)
        return comp.cam2d(im, 960 + 66 * e + s[0], 540 + 2 * e + s[1], 1.012 + .075 * e)

    def flecks(self, im, f):
        c = Cel(W, HH)
        for k in range(60):
            m = self.motes[k]; z = .5 + 1.8 * m[2] ** 2; x = (m[0] * 2300 + f * (7 + 16 * m[4]) * z) % 2300 - 190; y = 300 + m[1] * 760 + 18 * math.sin(f * .21 + m[3] * 9) * z + f * 1.2; R = (.7 + 1.5 * m[5] ** 2) * z; c.line(np.array([(x - R * 2.4, y), (x, y - R * .4), (x + R * 2.4, y)]), R, '#e9e0ff' if m[3] > .4 else '#0d0a1c', taper=(.4, .4), seed=k, alpha=.7)
        return self.lay(im, c.rgba(), .8)

    def frame(self, f):
        if f < 15: im = self.A(f)
        elif f < 24: im = self.B1(f)
        elif f < 34: im = self.B2(f)
        elif f == 34: im = self.crack()
        elif f < 44: im = self.D(f)
        elif f < 48: im = self.E(f)
        elif f < 50: im = self.E2(f)
        elif f < 56: im = self.S(f)
        else: im = self.Wd(f)
        return np.maximum(im, 0).astype(f32)
