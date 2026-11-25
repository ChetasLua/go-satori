"""d5 HE WALKS THROUGH IT.
0-29 the golden beam from the side and above over the mesa coast.  30-49 inside it: the walk, a foot planted on 30 and on 40.  50-59 face to face.
60 the flick: macro; 62 contact; 63-66 nothing at all; 67 impact frame; 68-70 the armour-burst wave; 71-79 the trench, at the horizon by 76.
Hooks (plain placeholders; return a float mask or a premultiplied RGBA layer):
  walker_d5(f) -> (mask, palm(x, y), [(foot x, foot y, weight)]);  duo_d5(f) -> [left head, right head];  flick_d5(f) -> [head, hand] (contact at TOUCH);  torso_d5(f) -> mask"""
import numpy as np, cv2, math
from db import W, H as HH, comp, kit, land, moon as M, moon_fx as X, ki_beam as KB, ki_hit as KH, ki_core as KC
from db import phys_rock as PR, phys_dust as PD, phys_core as PC
from db.cam import Camera3D
from db.ink import Cel
from db.paint import rgb, f32, fbm, smooth, mix, grid, ramp
keys = comp.lerp_keys
TOUCH = (905., 470.)
PIL = {p['id']: p['c'] for p in land.PILLARS(400)}
P_ = lambda L, x, y, s: np.round((np.array(L, float) * s + (x, y)) * 16).astype(np.int32)


def step_d5(f):
    """(feet y, px per head, phase 0..1 of the stride, which foot leads): the plant lands on 30 and 40, the body keeps travelling between them"""
    g = min(max(f, 26), 49); k = (g - 30) / 10.; i = int(math.floor(k)); u = k - i; return 905 + 62 * (k + .12 * math.sin(u * math.pi)), 62 * 1.27 ** k, u, i % 2


def walker_d5(f):
    """PLACEHOLDER walker: pins for limbs, true contact: the planted sole stays, the other swings"""
    fy, s, u, lead = step_d5((f // 2) * 2 if (f % 10) > 2 else f); x = 975.; m = np.zeros((HH, W), f32); bob = -.10 * math.sin(math.pi * u); y = fy
    T = lambda L: cv2.fillPoly(m, [P_(L, x, y + bob * s, s)], 1., cv2.LINE_AA, 4)
    cv2.ellipse(m, tuple(P_([(.02, -7.25)], x, y + bob * s, s)[0].tolist()), (int(.42 * s * 16), int(.54 * s * 16)), 0, 0, 360, 1., -1, cv2.LINE_AA, 4)
    T([(-.2, -6.8), (.2, -6.8), (.95, -6.35), (.62, -4.8), (.60, -3.5), (-.60, -3.5), (-.62, -4.8), (-.95, -6.35)]); feet = []
    for sd in (-1, 1):
        planted = (sd > 0) == (lead == 0); z = -.30 * (u - .15) if planted else -.95 * math.sin(math.pi * min(1, u * 1.02)) * (u < .96); fx = sd * .34; T([(sd * .05 - .28, -3.6), (sd * .05 + .30, -3.6), (fx + .25, -1.8 + z * .6), (fx + .18, z * (not planted)), (fx - .20, z * (not planted)), (fx - .25, -1.8 + z * .6)]); feet.append((x + fx * s, y + (0 if planted else z * s) + (bob * s if not planted else 0), 1. if planted else float(smooth(.80, 1., u))))
    T([(.70, -6.35), (1., -6.2), (1.35, -5.35), (1.7, -6.35), (1.98, -6.22), (1.55, -4.92), (1.2, -4.9), (.72, -5.8)]); T([(-.72, -6.35), (-1., -6.2), (-1.22, -4.9), (-1.12, -3.7), (-.88, -3.7), (-.9, -4.9), (-.66, -5.8)]); return m, (x + 1.84 * s, y + (bob - 6.35) * s), feet


def _egg(cx, cy, s, body=1):
    m = np.zeros((HH, W), f32); cv2.ellipse(m, (int(cx * 16), int(cy * 16)), (int(.36 * s * 16), int(.48 * s * 16)), 0, 0, 360, 1., -1, cv2.LINE_AA, 4); cv2.fillPoly(m, [P_([(-.15, .3), (.15, .3), (.2, .6), (.9, .85), (1.1, 2.4), (-1.1, 2.4), (-.9, .85), (-.2, .6)], cx, cy, s)], 1., cv2.LINE_AA, 4); return m


def duo_d5(f): k = (f - 50) / 10.; return [_egg(590 + 10 * k, 500, 520), _egg(1340 - 22 * k, 470, 520)]


def flick_d5(f):
    k = 0 if f < 62 else 1; hx = np.zeros((HH, W), f32); cv2.ellipse(hx, (int(470 * 16), int(560 * 16)), (int(430 * 16), int(610 * 16)), 0, 0, 360, 1., -1, cv2.LINE_AA, 4); h = np.zeros((HH, W), f32); dx = 0 if k else 60 - 6 * (f - 60)
    cv2.fillPoly(h, [P_([(0, -24), (70, -34), (420, -70), (430, 60), (90, 34), (8, 22)], TOUCH[0] + dx, TOUCH[1], 1.)], 1., cv2.LINE_AA, 4); cv2.fillPoly(h, [P_([(400, -190), (1100, -260), (1100, 420), (420, 300)], TOUCH[0] + dx, TOUCH[1], 1.)], 1., cv2.LINE_AA, 4); return [hx, h]


def torso_d5(f): return _egg(860, 190, 560)


def lit(img, m, body, rims):
    """placeholder shading: flat body + hard rims [(dx, dy, colour, amount)]; an RGBA layer is laid over untouched"""
    if m.ndim == 3: return comp.over(img, m)
    b = np.broadcast_to(rgb(body), img.shape).copy()
    for dx, dy, c, a in rims: r = (m * (kit.shifted(m, dx, dy) < .5) * a)[..., None]; b = b * (1 - r) + rgb(c) * r
    return img * (1 - m[..., None]) + b * m[..., None]


WIDE = [0, 1, 2, 3] + list(range(4, 30, 2))                        # drawings of the coast: ones while it is lit, then twos
FALL = [(1, 34., 5), (3, 42., 8), (5, 50., 11), (6, 66., 14), (9, 64., 17), (17, 42., 20), (28, 44., 23)]      # (pillar, height it is cut at, frame): the light cuts them one after another, left to right


class _Axis(KB.Axis3):
    """KB.Axis3 without the 2-D np.cross that this numpy refuses"""
    def __init__(self, cam, A, B, R0, R1=None, warp=None, near=.6):
        self.cam = cam; A, B = np.asarray(A, float), np.asarray(B, float); self.warp = warp; d = B - A; self.Lw = float(np.linalg.norm(d)); self.d = d / self.Lw; self.A = A; self.R0 = R0; self.R1 = R0 if R1 is None else R1
        self.M = self.Lw / R0; q = np.unique(np.clip(np.concatenate([np.linspace(0, 1, 400) ** 2.2, 1 - np.linspace(0, 1, 400) ** 2.2, np.linspace(0, 1, 200)]), 0, 1)); X_ = A + d * q[:, None]; px, z = cam.project(X_); ok = z > near
        self.ok = ok.any(); q = q[ok]; px = px[ok]; z = z[ok]; self.mq = q * self.M; self.m0 = float(self.mq[0]); self.M1 = float(self.mq[-1]); self.sq = np.r_[0, np.cumsum(np.hypot(*np.diff(px, axis=0).T))]; self.rq = (self.R0 + (self.R1 - self.R0) * q) * cam.f / z
        v = cam.eye - (A + d * float(np.clip((cam.eye - A) @ self.d / self.Lw, .02, .98))); v = v - self.d * (v @ self.d); v = v / (np.linalg.norm(v) + 1e-9); self.h = v; self.n = np.cross(self.d, v); mid = X_[len(X_) // 2]
        a, _ = cam.project(np.stack([mid, mid + self.n * R0])); t = cam.project(A + d * np.array([[.5], [.52]]))[0]; T = t[1] - t[0]; u = a[1] - a[0]
        if T[0] * u[1] - T[1] * u[0] > 0: self.n = -self.n


class Shot:
    id = 'd5'; n = 80
    finish = dict(bloom=.30, vig=.14, grain=.010, soft=.14)
    EVENTS = [(0, 'ignite', dict(size=.8, note='a white point on the left')), (1, 'beam', dict(frames=29, size=1.4, note='held brass chord; the widest energy so far')), (3, 'boom', dict(size=1., note='the sea takes the sound')), (30, 'step', dict(size=.8, note='inside: the chord is cut by the foot')), (40, 'step', dict(size=1.)),
              (50, 'beam_end', dict(size=.5, note='gold dies to falling sparks; two people breathe')), (60, 'finger_coil', dict(size=.2)), (62, 'flick', dict(size=.15, note='a tiny tick, dry')), (63, 'silence', dict(frames=4, note='nothing at all')), (67, 'impact', dict(size=1.6)), (68, 'armour_burst', dict(frames=3, size=1.2)),
              (71, 'trench', dict(frames=6, size=1.3, note='ploughed to the horizon, sound runs away from us')), (76, 'far_impact', dict(size=.5, note='arrives late, small and deep'))]
    walker = staticmethod(walker_d5); duo = staticmethod(duo_d5); flick = staticmethod(flick_d5); torso = staticmethod(torso_d5)

    # ------------------------------------------------------------ plates
    def setup(self):
        p = 'cache/d5_wide_v7.npz'
        import os
        A, B = np.array([-70., -210., 112.]), np.array([40., 6400., 118.]); self.cam = Camera3D((480, -120, 230), (-30, 520, 20), 1200); self.ax = _Axis(self.cam, A, B, 74., 80.)
        if os.path.exists(p): q = np.load(p); self.night = q['n'].astype(f32); self.G8, self.F8, self.W8 = q['g'], q['f'], q['w']
        else:
            axp, z = self.cam.project(A + (B - A) * np.linspace(0, 1, 900)[:, None]); ok = z > 30; zi = np.interp(np.arange(W), axp[ok, 0], z[ok]); self.night = land.coast(self.cam, 0., 'night', moon=None, ss=2).img.astype(f32); G8, F8, W8 = [], [], []
            for g in WIDE:
                br = [(i, (g - t0) / 24.) for i, zc, t0 in FALL if g >= t0]; dyn = [land.break_pillar(i, zc, (.75, .66), (g - t0) / 24. * 2.3, seed=4, power=1.15) for i, zc, t0 in FALL if g >= t0]; fr_ = [(-60, float(y), 112., 380., '#ffb02e', 1.5 * min(1., (g + 1) / 4.)) for y in np.linspace(-80, 1500, 9)]
                rg = [(-60. + 8 * math.sin(k), 20. + 130 * k, 10 + 2.6 * max(0., g - 3 - .5 * k), 9., .9 * math.exp(-max(0., g - 3 - .5 * k) / 14.)) for k in range(10) if g > 3 + .5 * k]
                o = land.coast(self.cam, g / 24., 'night', moon=None, ss=2, fires=fr_, rings=rg, dyn=land.cat(*dyn) if dyn else None, hide=[i for i, _ in br]); G8.append(np.clip(o.img / 1.5 * 255 + .5, 0, 255).astype(np.uint8)); F8.append((255 * cv2.GaussianBlur(((o.rock > .5) & (o.z < zi[None] - 20)).astype(f32), (0, 0), .7)).astype(np.uint8)); W8.append((o.water * 255).astype(np.uint8))
            self.G8, self.F8, self.W8 = np.stack(G8), np.stack(F8), np.stack(W8); os.makedirs('cache', exist_ok=True); np.savez(p, n=self.night.astype(np.float16), g=self.G8, f=self.F8, w=self.W8)
        self._lock = {}
        p = 'cache/d5_waste_v2.npz'; self.wcam = Camera3D((0, -16, land.TOP + 9.5), (0, 60, land.TOP - 1), 1150); self.tcam = Camera3D((0, -16, 9.5), (0, 60, -1), 1150)
        if os.path.exists(p): self.waste = np.load(p)['a'].astype(f32)
        else: self.waste = land.waste(self.wcam, time='night', ss=2).img.astype(f32); np.savez(p, a=self.waste.astype(np.float16))
        ys = np.linspace(2, 1150, 60); self.tr = PR.Trench(np.c_[2.2 * np.sin(ys * .011), ys], width=7.5, depth=3., bank=2.0, seed=3, step=4.6); self.fr = lambda t: float(1. / np.interp(min(1., max(0., t * 24 / 5.)), [0, 1], [1 / 18.5, 1 / 1158.]) - 18. + 2)           # 5 frames, even on SCREEN: near ground first
        self.rb = self.tr.rubble(self.fr, frames=14, n=70, size=(.25, 1.9), speed=(5., 17.), seed=4); self.sh = PR.Rubble(9)
        for k in range(3): self.sh.burst((0, 0, 2.6 - 1.3 * k), n=30, speed=(6, 22), up=.25, size=(.08, .62), dirn=(0, -1, .1), spread=1.5, t0=k / 24., seed=k * 7, kinds=('shard', 'slab'), jit=(.8, .1, .5), spin=10.)
        self.sh.bake(8, g=9.8); x, y = grid(); self.x, self.y = x, y; self.n1 = fbm(HH, W, (60, 20, 7), (1, .6, .35), 14, aspect=2.2).astype(f32); self._c = {}

    # ------------------------------------------------------------ 0-29 wide
    def wide(self, f):
        g = f if f < 4 else (f // 2) * 2; k = float(keys(f, [(0, .0), (1, .25), (2, .75), (4, 1.), (30, 1.)], lambda t: t)); head = self.ax.M * float(keys(f, [(0, 0.), (1, .10), (2, .38), (3, .72), (4, 1.), (30, 1.)], lambda t: t)); w = float(keys(f, [(0, .35), (3, .80), (5, 1.18), (9, .96), (18, 1.10), (30, 1.42)]))
        key = (g, round(head, 2))
        if key not in self._c: self._c.clear(); self._c[key] = KB.beam(self.ax, g, 'gold', head=head if f < 4 else None, seed=5, w=w, speed=30., licks=1., collars=4, power=1.15, step=1, dens=1.2)
        lay, li = self._c[key]; hx = float(self.cam.project(self.ax.A + self.ax.d * head * 74.)[0][0]) if f < 4 else 3000.; sw = smooth(hx + 500, hx - 100, self.x) * k if f < 4 else np.float32(1.); fl = 1 + .05 * math.sin(f * 2.1) + (.10 * math.exp(-(f - 3) / 3.) if f >= 3 else 0.)
        j = max(i for i, q in enumerate(WIDE) if q <= f); gold = self.G8[j].astype(f32) * (1.5 / 255.); self.front = self.F8[j].astype(f32) / 255.; self.wat = self.W8[j].astype(f32) / 255.
        img = self.night * .80 + (gold * 1.15 - self.night * .80) * (sw[..., None] if np.ndim(sw) else sw) * min(1.15, fl)
        if lay is not None:
            q = cv2.resize(lay[..., :3], (W // 4, HH // 4), interpolation=cv2.INTER_AREA); r = cv2.GaussianBlur(q, (0, 0), sigmaX=4, sigmaY=26); r = cv2.warpAffine(r, np.float32([[1, 0, 0], [0, 1, 46]]), (W // 4, HH // 4)); r = cv2.resize(r, (W, HH)); gl = np.where(np.sin(self.y * .30 + self.n1 * 2.4 + g * .7) + self.n1 * .9 > .15, 1., .34); img = img + r * (self.wat * gl * .85 * (1 - self.front))[..., None]      # the sea under it burns in broken bands
            img = img + li * .75; a = lay * (1 - self.front[..., None]); img = KC.put(img, a); rim = self.front * (1 - kit.shifted(self.front, 4, 4)) * smooth(.02, .3, cv2.GaussianBlur(lay[..., 3], (0, 0), 40)); img = img * (1 - rim[..., None]) + rgb('#ffe06a') * rim[..., None] * 1.1
        img = self._steam(img, g)
        e_ = self.cam.project(self.ax.A + self.ax.d * self.ax.Lw)[0]
        if f >= 4: img = kit.flare(img, float(e_[0]), float(e_[1]), 300, '#ffd54a', .55 + .10 * math.sin(f * 1.7), streak=.5)
        if f < 6:
            p = self.cam.project(self.ax.A + self.ax.d * 250.)[0]; img = kit.flare(img, float(p[0]), float(p[1]), 900, '#ffe9a0', 1.2 * math.exp(-f / 2.), streak=.8, ang=-.3)
        s = kit.shake(f, 9., 4, 7., 3) + kit.shake(f, 1.6, 9); img = img + (np.exp(-((self.y - 300 + .20 * (self.x - 960)) / 300) ** 2) * (sw if np.ndim(sw) else 1.) * .20 * min(1., k) * (.7 + .3 * self.n1))[..., None] * rgb('#f6a416'); e = f / 30.; z = 1.03 + .13 * e * e * (3 - 2 * e); return comp.cam2d(img.astype(f32), 960 + s[0] + 70 * e, 540 + s[1] - 34 * e, z)

    def _steam(self, img, g):
        """the sea under the beam boils: a trench of steam opens along it; each cut pillar throws dust and a flash"""
        t = g / 24.; L = []
        if g >= 3:
            at = lambda s: (np.array([-18. - 10 * math.sin(s * .012), -40. + s]), np.array([1., 0.])); fr = lambda q: float(np.interp(q, [.10, .24, .6], [0, 900, 1500])); L += PD.dust_trail(t, at, fr, 1300, step=58., seed=7, r=(13., 46.), side=70., up=92., life=2.2, drag=1.3, w=34.)
        for i, zc, t0 in FALL:
            if g >= t0 - 1: c = PIL[i]; L += PD.plume((g - t0 + 1) / 24., (c[0], c[1], zc), (.6, .4, .5), 46., n=12, r=(4., 13.), spread=.8, emit=.2, seed=i, life=1.5)
        if L:
            lay = PD.lobes_layer(self.cam, L, dict(PC.DUST['steam'], lit='#fffaf0', mid='#ffdf9c', dark='#c98a6a', ink='#9a6450', rim='#ffffff'), light=(-.30, -.86, .30), ss=1., ink=.35, fog=(rgb('#8a5a6a'), 420., .0011)); img = comp.over(img.astype(f32), lay * (1 - .92 * self.front[..., None]), .90)
        for i, zc, t0 in FALL:
            if 0 <= g - t0 <= 3: c = PIL[i]; p = self.cam.project(np.array([c[0], c[1], zc]))[0]; img = kit.flare(img, float(p[0]), float(p[1]), 170 - 34 * (g - t0), '#ffe9a0', 1.1 * math.exp(-(g - t0) / 1.2))
        return img

    # ------------------------------------------------------------ 30-49 inside
    def _foot(self, f):
        """lock: within each step the planted sole keeps its screen place; the body travels over it.  -> (dx, dy) to add to the whole figure"""
        if f in self._lock: return self._lock[f]
        off = np.zeros(2)
        for s0 in (30, 40):
            mid = self.walker(s0 + 4)[2]; k = int(np.argmax([v[2] for v in mid])); ref = np.array(self.walker(s0)[2][k][:2]) + off
            for q in range(s0, s0 + 10): now = np.array(self.walker(q)[2][k][:2]); self._lock[q] = tuple(ref - now)
            off = np.array(self._lock[s0 + 9])
        return self._lock[f]

    def inside(self, f):
        m, palm, feet = self.walker(f); ox, oy = self._foot(f); m = cv2.warpAffine(m, np.float32([[1, 0, ox], [0, 1, oy]]), (W, HH)) if abs(ox) + abs(oy) > .05 else m; palm = (palm[0] + ox, palm[1] + oy); feet = [(a_ + ox, b_ + oy, c_) for a_, b_, c_ in feet]; lay = m if m.ndim == 3 else None; a = lay[..., 3] if lay is not None else m; t = 3 + (f - 30) * .42; hit = math.exp(-((f % 10)) / 2.2)
        img = X.beam_inside(a, palm=palm, t=t, vp=(palm[0] * .18 + 975 * .82, 470.), floor=600, feet=feet, anchor=(0, int(feet[0][1])), tatter=.9, figure=lay, seed=4)
        s = kit.shake(f, 7. * hit, 3); return comp.cam2d(img, 960 + s[0], 540 + s[1] + 5 * hit, 1.02)

    # ------------------------------------------------------------ 50-59 two-shot
    def two(self, f):
        g = f - 50; x, y = self.x, self.y; L, R_ = self.duo(f); n = self.n1
        bg = ramp(np.clip(x / W + .06 * n, 0, 1), [(0, '#6a3a0c'), (.18, '#4a2818'), (.42, '#231630'), (.62, '#1a1038'), (1, '#2c1260')]); band = smooth(.35, .55, n + .9 * np.exp(-((y - 700) / 190) ** 2) - .3); bg = bg * (1 + .55 * band[..., None] * smooth(1300, 200, x)[..., None]); d = np.hypot(x - 962, y - 250); dm = smooth(60.7, 59.3, d)
        img = bg + M.halo_rgb(np.where(d > 60, M.halo(d - 60, 60), 0) * .8); img = img * (1 - dm[..., None]) + rgb(M.DEAD) * dm[..., None]; fl = math.exp(-g / 3.); img = img + (np.exp(-((x + 80) / 620) ** 2) * fl)[..., None] * rgb('#ffd54a') * .9
        img = cv2.GaussianBlur(img.astype(f32), (0, 0), 2.)
        img = lit(img, L, '#140c10', [(9, 2, '#e08a14', .85), (4, 1, '#ffe9a0', 1.), (-5, 2, '#6a2cf0', .8)]); img = lit(img, R_, '#0d0820', [(-9, 2, '#5a1fd0', .8), (-4, 1, '#d9b8ff', 1.), (5, 2, '#c8761c', .5)])
        acc = np.zeros((HH, W), f32); r = np.random.default_rng(3)                                                    # what is left of the beam: gold falling between them
        for i in range(150):
            px, py, v, sz = r.uniform(0, W), r.uniform(-200, HH), r.uniform(1.5, 6), r.uniform(.8, 3.4); X0 = px + 5 * math.sin(g * .5 + i) + (14 * fl); Y0 = py + v * g; cv2.circle(acc, (int(X0 * 16), int(Y0 * 16)), int(sz * 16), float((.35 + .65 * r.random()) * (.35 + .65 * fl)), -1, cv2.LINE_AA, 4)
        img = img + acc[..., None] * rgb('#ffd98a') + cv2.GaussianBlur(acc, (0, 0), 6)[..., None] * rgb('#f6a416') * .9; return comp.cam2d(img.astype(f32), 960, 520, 1.0 + .022 * g / 10.)

    # ------------------------------------------------------------ 60-70 flick, nothing, burst
    def macro(self, f):
        x, y = self.x, self.y; hd, hn = self.flick(min(f, 62)); bg = ramp(np.clip((x + y * .25) / 2100, 0, 1), [(0, '#1c1010'), (.4, '#0d0818'), (1, '#2a1160')]); r = np.random.default_rng(8); acc = np.zeros((HH, W), f32)
        for _ in range(26): cv2.circle(acc, (int(r.uniform(0, W) * 16), int(r.uniform(0, HH) * 16)), int(r.uniform(10, 34) * 16), float(r.uniform(.12, .4)), -1, cv2.LINE_AA, 4)
        img = bg + cv2.GaussianBlur(acc, (0, 0), 2)[..., None] * rgb('#c8761c') * .7; img = lit(img, hd, '#1b1016', [(16, 5, '#8a4a14', .8), (-14, -6, '#8a2cff', .9), (-5, -2, '#f0dcff', 1.)]); img = lit(img, hn, '#0e0922', [(-2, 9, '#5a1fd0', .9), (3, 4, '#d9b8ff', 1.)])
        if f in (61, 62):                                               # two drawn smears: the swing as torn tapered blades on an arc from the cocked place (right) to TOUCH
            c = Cel(); r = np.random.default_rng(61 + f); e0, e1 = (0., .72) if f == 61 else (.42, 1.); C0, C1, C2 = np.array([1300., 250.]), np.array([1180., 520.]), np.array(TOUCH)
            for i in range(13):
                o = r.normal(0, 1); a0, a1 = e0 + r.uniform(0, .16), e1 - r.uniform(0, .10) * (f == 61); u = np.linspace(a0, a1, 14)[:, None]; P = (1 - u) ** 2 * C0 + 2 * u * (1 - u) * C1 + u * u * C2; N = np.array([-.55, .83]); P = P + N * (o * 46 * (1 - u ** 2)) ; w_ = (74 if i < 3 else 30) * r.uniform(.6, 1.15); c.line(P, w_, ['#2a1254', '#5a1fd0', '#0e0922', '#d9b8ff'][i % 4], taper=(.75, .18), press=.4, seed=i + f, tremor=.4)
            img = comp.over(img.astype(f32), c.layer())
        if f == 62:                                                     # the contact: one frame, a thread of light and a hair ring
            a = np.zeros((HH, W), f32); M._star(a, TOUCH[0], TOUCH[1], 54, 1.); d = np.hypot(x - TOUCH[0], y - TOUCH[1]); a = np.maximum(a, .7 * np.exp(-((d - 74) / 1.6) ** 2)); img = img + a[..., None] * rgb('#f0dcff') + cv2.GaussianBlur(a, (0, 0), 9)[..., None] * rgb('#8a2cff')
        return img.astype(f32)

    def burst(self, f):
        k = f - 67; b = self.macro(62)
        if k == 0: hd, hn = self.flick(62); a = np.clip((hd if hd.ndim == 2 else hd[..., 3]) + (hn if hn.ndim == 2 else hn[..., 3]), 0, 1); return KH.impact(b, 5, TOUCH[0], TOUCH[1], seed=4, accent='#8a2cff').astype(f32)
        x, y = self.x, self.y; m = self.torso(f); fy = 120 + (k - 1) * 360.; cx = 860.; img = ramp(np.clip(np.hypot(x - cx, (y - fy)) / 1300, 0, 1), [(0, '#fff6d8'), (.10, '#f3d9ff'), (.16, '#a05cff'), (.40, '#3a1290'), (1, '#0a0618')]); an = np.arctan2(y - fy, x - cx); ray = smooth(.5, .62, np.sin(an * 31 + self.n1 * 1.1) * .5 + .5 * np.sin(an * 13 + 2)); img = img * (1 - .34 * ray[..., None])
        img = lit(img, m, '#120a1e', [(8, 10, '#5a1fd0', .9), (-7, -9, '#f3d9ff', 1.)]); a = m if m.ndim == 2 else m[..., 3]
        e = (y - fy) + 46 * np.sin((x - cx) * .011) + 24 * self.n1; w = np.exp(-(e / 26) ** 2) * a; tail = smooth(0, -260, e) * a; img = img * (1 - (.55 * tail)[..., None]) + rgb('#c784ff') * (.18 * tail)[..., None]; img = img * (1 - w[..., None]) + rgb('#fffbe8') * 1.15 * w[..., None]                 # the wave runs down the body
        ck = smooth(.07, .03, np.abs(np.sin(self.n1 * 6.5 + x * .013))) * smooth(170, 10, e) * smooth(-12, 10, e) * a; img = img * (1 - ck[..., None]) + rgb('#fff1c8') * ck[..., None]                                                                    # cracks run ahead of it
        c = Cel(); cm = Camera3D((0, -9., 1.6), (0, 0, 1.5), 1500, cx=cx, cy=430); pal = dict(PR._pal('night')); pal.update(lit='#fff6e6', mid='#cbb4e6', dark='#5a3f9a', rim='#ffffff', ink='#160a2c'); PR.draw_bodies(c, cm, self.sh.items(k + 1.6 * k, True), pal, (-.3, -.6, .75), None, 3.0); return comp.over(img.astype(f32), c.layer())

    # ------------------------------------------------------------ 71-79 trench
    def trench(self, f):
        t = (f - 71 + 1) / 24.; s = self.fr(t); img = self.waste.copy(); c = Cel(); fog = ('#37296f', 60., .0032); self.tr.draw(c, self.tcam, s, 'night', (.74, -.2, .62), 2.6, fog=fog, cracks=True)
        img = comp.over(img, c.layer()); L = PD.dust_trail(t + .03, self.tr.at, self.fr, 620, step=11., seed=4, r=(1.6, 5.2), side=7., up=6.5, life=2.4, w=4.2); lay = PD.lobes_layer(self.tcam, L, 'night', light=(.6, -.55, .5), ss=1., ink=.8, fog=(rgb('#37296f'), 60., .0032)) if L else None
        if lay is not None: img = comp.over(img, lay)
        c = Cel(); self.rb.draw(c, self.tcam, t * 24, 'night'); img = comp.over(img, c.layer()); p, z = self.tcam.project(np.array([*self.tr.front(min(s, 1140)), 1.5]))
        if f <= 77: r = min(30., max(5., 1150 / max(float(z), 4) * 1.1)); al = 1. if f < 76 else (1.5 if f == 76 else .6); img = kit.flare(img, float(p[0]), float(p[1]), r * 7 + (170 if f == 76 else 0), '#c784ff', .9 * al, streak=.35 if f >= 75 else 0.); d = np.hypot(self.x - p[0], self.y - p[1]); img = img + (np.exp(-(d / (r * .8)) ** 2) * al)[..., None] * rgb(M.DEAD)
        s_ = kit.shake(f, 13., 6, 3.2, 71); return comp.cam2d(img.astype(f32), 960 + s_[0], 540 + s_[1], 1.025)

    def frame(self, f):
        return (self.wide if f < 30 else self.inside if f < 50 else self.two if f < 60 else self.macro if f < 67 else self.burst if f < 71 else self.trench)(f).astype(f32)


from chars import figs_lead as _FL                                  # the real figures
Shot.walker = staticmethod(_FL.d5_walker(step_d5)); Shot.duo = staticmethod(_FL.d5_duo); Shot.flick = staticmethod(_FL.d5_flick(TOUCH)); Shot.torso = staticmethod(_FL.d5_torso)
