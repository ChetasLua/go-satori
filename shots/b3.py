"""b3 SMALL HAND, BIG HAND.  0-29 close: the pearl is born on 12.  30 wide on the lake: pfff, three skips (36, 39, 42), plop 44, nothing... 47 a perfect column.
50 Piccolo: one eye (50), it looks up (52), the shadow arrives (53), he closes it again (55), SPLASH 56, the turban droops (62), dead-pan.  68 belly laugh."""
import numpy as np, cv2, math
from db import W, H as FH, comp, kit, land, phys_water as PW
from db.paint import rgb, f32, smooth, grid
from db.ink import Cel, spl
from db.cam import Camera3D
from db.land_sets import lake_z
from db.hero3 import Fig, POSES, compose
from chars import gohan_lake as GL, gohan as G, fkit as K, friends_water as FW
from chars.gohan import Gohan
from chars.piccolo import Piccolo
from chars import piccolo as PI

A = np.array; keys = comp.lerp_keys; over, screen, radial, blur = comp.over, comp.screen, comp.radial, comp.blur
RIM = dict(dir=(-.7, .6, -.3), col='#fff3b8', thr=.45, amt=.85); LT = (-.6, .6, .45)
# ---- the wide: metres (x right, y away, z up, water z = 0)
WCAM = dict(eye=(1.1, -11.6, 3.1), tgt=(2.1, 60, .9), f=1270.)                 # locked off: the joke needs a still frame
GP = (-1.75, -4.35); PP = (5.25, 4.6, 1.55); CP = (5.65, 7.6); CR = .74
SK = [(33, (-.45, -4.0, .92)), (36, (1.45, 3.0, 0.)), (39, (3.0, 4.9, 0.)), (42, (4.45, 6.5, 0.)), (44, (CP[0] - .1, CP[1] - .2, 0.))]
HOP = [1.1, 1.15, .75, .35]


def wcam(f=0): return Camera3D(A(WCAM['eye']), A(WCAM['tgt']), WCAM['f'])


def pellet(f):
    """position of the tiny wave at frame f (float), None when it is not flying"""
    if f < 33 or f > 44.4: return None
    for j in range(len(SK) - 1):
        a, b = SK[j], SK[j + 1]
        if f <= b[0]: u = (f - a[0]) / (b[0] - a[0]); p = A(a[1]) * (1 - u) + A(b[1]) * u; p[2] += HOP[j] * 4 * u * (1 - u); return p
    return None


def col_h(f):
    """height of the column of water (m)"""
    return float(keys(f, [(45, 0.), (46, .22), (47, .55), (48, 4.6), (49, 7.6), (50, 9.2), (52, 10.2), (60, 10.6)], lambda t: t ** .8))


class Shot:
    id = 'b3'; n = 80
    finish = dict(bloom=.30, vig=.14, soft=.12, grain=.011)
    EVENTS = [(0, 'lake_amb', dict(note='warm lake, reeds, wind; child breath held')), (4, 'spark_fail', dict(size=.2, note='two tiny fizzles, 4 and 6')), (6, 'spark_fail', dict(size=.15)), (4, 'hands_close', dict(frames=7, note='soft skin on skin, cloth')), (12, 'pearl_born', dict(size=.8, note='glass bell pop on the beat, hum follows')), (18, 'hands_seat', dict(size=.3)), (20, 'child_joy', dict(note='small breath of wonder')),
              (30, 'cut_wide', dict(note='hum cut')), (33, 'pfff', dict(size=.3, note='comic, tiny: a cork and air')), (36, 'skip', dict(size=.35, pan=.10)), (39, 'skip', dict(size=.3, pan=.25)), (42, 'skip', dict(size=.22, pan=.38)), (44, 'plop', dict(size=.2, pan=.45)), (45, 'silence', dict(frames=2, note='one bird')), (46, 'water_swell', dict(size=.3, pan=.5)), (47, 'column_rise', dict(size=.9, frames=3, pan=.5)),
              (50, 'eye_open', dict(size=.3, note='dry tick, like a lid')), (52, 'eye_up', dict(size=.15)), (53, 'shadow', dict(note='light goes, water sound from above')), (55, 'eye_shut', dict(size=.2, note='resigned')), (56, 'splash', dict(size=1., frames=6, note='a whole column on one head')), (62, 'turban_droop', dict(size=.35, note='wet cloth slap')), (63, 'drip', dict(size=.3)), (65, 'drip', dict(size=.35)), (67, 'drip', dict(size=.3)),
              (68, 'laugh', dict(frames=12, note='belly laugh, bounces on 68 71 74 77')), (70, 'ball_glint', dict(size=.5))]

    def setup(self):
        self._w = {}; self._bg = {}; self._ov = None

    def near(self):
        """what stands in front of the water in the wide: bank, reeds, grass, tree (soft mask, once)"""
        if self._ov is None:
            a = land.lake(1.5, cam=wcam(), overlays=False); b = land.lake(1.5, cam=wcam()); m = smooth(.015, .05, np.abs(b.img - a.img).max(-1)); y = grid(FH, W)[1]; m = np.maximum(m * (y > 600), ((a.rock > 0) & (a.z < 14)).astype(f32)); self._ov = cv2.GaussianBlur(m.astype(f32), (0, 0), .7)[..., None]
        return self._ov

    # ------------------------------------------------------------ wide
    def wide(self, f):
        g = (f // 2) * 2; t = f / 24.; cam = wcam(f); rings = []
        for fs, p in SK[1:]:
            if f >= fs: rings.append((p[0], p[1], .25 + (f - fs) / 24. * 2.1, .22, .05 * math.exp(-(f - fs) / 14.)))
        if f >= 46: rings.append((CP[0], CP[1], .7 + (f - 46) / 24. * 3.4, .35, .10))
        nr = self.near(); o = land.lake(t * .9, cam=cam, rings=rings); img = o.img
        # column behind Piccolo, its long reflection first
        h = col_h(f); pal = 'lake'
        if h > .05:                                                # a swell, then a torn column: water that climbs, sheds and crowns
            b0 = cam.project(A([[CP[0], CP[1], 0]]))[0][0]; sc = cam.scale(A([CP[0], CP[1], h * .5])); yt = cam.project(A([[CP[0], CP[1], h]]))[0][0][1]; y0 = float(b0[1])
            if h > 1.2: lay = FW.col(float(b0[0]), float(yt), y0, CR * sc, (f - 44) / 24., seed=3, flow=1., head=1.25, lean=.012 * (f - 48))
            else: cel = Cel(); self.bulge(cel, cam, h); lay = cel.layer()
            img = self.reflect(img, lay, y0, f, .26); img = over(img, lay * (1 - nr))
            if f >= 47: cel = Cel(); PW.splash(cel, cam, (f - 46.4) / 24., A(CP), v0=5.2, r0=CR * 1.15, n=16, rad=.10, pal='lake', seed=31, sheet=.30, jetv=0, rings=3); img = over(img, cel.layer() * (1 - nr))
        # Piccolo in mid-air, his reflection broken by the ripples
        bob = .05 * math.sin(g * .22); q = dict(POSES['lotus'])
        for k in ('hL', 'hR', 'eL', 'eR', 'gL', 'gR', 'wL', 'wR'): q[k] = POSES['arms_crossed'].get(k, q.get(k))
        key = ('p', g)
        P = Fig('piccolo', 'clean').pose(q).place(PP[0], PP[1], PP[2] + bob, heading=math.pi - .30); P.blow(-.35, 0, 0); lp = P.draw(cam, g, light=LT, pal='day', rim=RIM, expr=dict(expr='shut')); y0 = float(cam.project(A([[PP[0], PP[1], 0]]))[0][0][1]); img = self.reflect(img, lp, y0, f, .42); img = over(img, lp * (1 - nr))
        # the shore: father on one knee behind the boy
        gz = float(lake_z(*GP)); rec = float(keys(g, [(30, 0.), (32, -.05), (33, .07), (36, .03), (40, 0.), (50, 0.)])); lean = float(keys(g, [(30, 0.), (44, 0.), (47, 1.), (50, 1.)]))
        GO = Fig('goku').pose('kneel_up').place(GP[0] - .03 - rec * .5, GP[1], gz, heading=-math.pi / 2 - .30); KI = Fig('gohan', 'clean').pose('kame_fire').place(GP[0] + .58 - rec, GP[1] - .10, float(lake_z(GP[0] + .6, GP[1])) + .0, heading=-math.pi / 2 - .30); KI.blow(-.5, 0, .1); w_ = A([GP[0] + 1.06 - rec, GP[1] + .02, gz + .66])
        GO.reach(R=dict(p=tuple(w_ + [0, -.14, .02]), part='wrist', w=1, body=.25), L=dict(p=tuple(w_ + [.02, .16, .02]), part='wrist', w=1, body=.25)); GO.look((8., 6., 1.2)); KI.look((8., 6., 1.0 + lean * 4))
        img = over(img, compose([(GO, dict(light=LT, pal='day', rim=RIM, expr=dict(expr='smile'))), (KI, dict(light=LT, pal='day', rim=RIM, expr=dict(expr='shout' if f < 37 else 'wonder')))], cam, g))
        img = self.wave(img, cam, f); return img

    def reflect(self, img, lay, y0, f, amt):
        """upside-down picture of a layer in the water: ripple-cut, greener, fading with depth"""
        x, y = grid(FH, W); m = y > y0; sy = 2 * y0 - y; dx = 3.2 * np.sin(y * .21 + f * .55) + 5. * np.sin(y * .067 - f * .4); r = cv2.remap(lay.astype(f32), (x + dx).astype(f32), sy.astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT); k = (m * np.clip(1 - (y - y0) / 420., 0, 1) * (np.sin(y * .33 + np.sin(x * .02 + f * .2) * 2) > -.45))[..., None] * amt * (1 - self.near())
        return img * (1 - r[..., 3:4] * k) + r[..., :3] * A([.78, .95, .86], f32) * k

    def bulge(self, cel, cam, h):
        a = np.linspace(0, math.pi, 30); P = np.c_[CP[0] + np.cos(a) * CR * 1.5, np.full(30, CP[1]), np.sin(a) * h]; q = cam.project(P)[0]; cel.fill(q, '#b9d98a'); cel.fill(q[:14], '#f4fbe0'); cel.line(q, 2.2, '#163c38', taper=(.2, .2)); PW.ripple(cel, cam, A(CP), .25, 1.4, 2, pal='lake')

    def wave(self, img, cam, f):
        cel = Cel(); t = lambda fs: (f - fs) / 24.
        for j, (fs, p) in enumerate(SK[1:]):
            if f > fs - .01: PW.splash(cel, cam, t(fs) + .03, A(p[:2]), v0=(3.4, 3.0, 2.5, 1.5)[j], r0=.20, n=9, rad=.06, pal='lake', seed=20 + j, sheet=.35, jetv=.9 if j == 3 else .4, rings=2, dirn=(.5, .45))
        if 33 <= f <= 39:                                          # pfff: a round puff from the little hands
            s = cam.scale(A(SK[0][1])); c = cam.project(A([SK[0][1]]))[0][0]; u = (f - 33) / 6.
            for j in range(6): a = j * 1.05 + .4; r = s * (.05 + .16 * u) * (1.15 - .12 * j); q = c + A([math.cos(a) * s * (.06 + .26 * u) + s * .10 * u, math.sin(a) * s * (.05 + .16 * u) - s * .10 * u]); cel.fill(K.ell(q[0], q[1], r * (1 - u * .55), r * .86 * (1 - u * .55)), '#eef8ff' if j % 2 else '#bfe0f4', 1 - u * .7)
        pos = pellet(f)
        if pos is not None and f < 44:                             # the pellet: a comet the size of a thumb, trail along its own arc
            tr = [pellet(f - q * .28) for q in range(9)]; tr = [p for p in tr if p is not None]
            if len(tr) > 2:
                P = cam.project(A(tr))[0]; s = cam.scale(pos); w_ = np.linspace(.24, .0, len(P)) * s
                for k, c, a in ((1.7, '#4fb0f0', .55), (1.15, '#8fd8ff', .8), (.6, '#ffffff', .9)): T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; cel.fill(np.vstack([P + N * (w_ * k)[:, None] * .5, (P - N * (w_ * k)[:, None] * .5)[::-1]]), c, a)
            c = cam.project(A([pos]))[0][0]; s = cam.scale(pos); s = max(s, 70.); cel.fill(K.ell(c[0], c[1], s * .19), '#3fa4ee'); cel.fill(K.ell(c[0], c[1], s * .15), '#b8ecff'); cel.fill(K.ell(c[0] - 1, c[1] - 1, s * .10), '#ffffff')
            gl = cam.project(A([[pos[0], pos[1], 0]]))[0][0]; cel.fill(K.ell(gl[0], gl[1], s * .10, s * .022), '#e8f8ff', .5)
        lay = cel.layer(); far = float(np.clip((f - 34.5) / 1.5, 0, 1)); img = over(img, lay * (1 - self.near() * far))
        if pos is not None and f < 44: c = cam.project(A([pos]))[0][0]; img = screen(img, np.clip(radial(c[0], c[1], 90, power=2.4) * .5, 0, 1)[..., None] * rgb('#6cc4ff'))
        return img

    # ------------------------------------------------------------ Piccolo
    PC = (1000., 318., 122.)

    def med_bg(self, f):
        k = min(f, 67) // 3
        if k not in self._bg:
            if len(self._bg) > 4: self._bg.clear()
            cam = Camera3D((3.9, -3.4, 2.45), (5.35, 30, 2.9), 2100.); o = land.lake(f / 24. * .9, cam=cam, overlays=False); b = blur(o.img.astype(f32), 3.2); self._bg[k] = (b, cam)
        return self._bg[k]

    def med(self, f):
        g = (f // 2) * 2; bg, cam = self.med_bg(f); img = bg.copy(); cx, cy, sc = self.PC; x, y = grid(FH, W); hit = f - 56
        dk = float(keys(f, [(50, 0.), (52, 0.), (54, .55), (56, .75), (57, .2), (60, 0.), (80, 0.)]))
        # the column stands behind him, then is gone upward
        if f < 56:
            img = over(img, FW.col(1120., -420., 1130., 236., (f - 44) / 24., seed=3, flow=1., skirt=.0, ragged=.07, drops=.5, blur_=1.8))
        e = 'shut'
        if f >= 50: e = dict(PI.E['oneeye'], oneeye=1, gy=-.004, gx=.012)
        if f == 50: e = dict(e, U=[-.034, -.006, .006, .012, .012])
        if 52 <= f < 55: e = dict(e, gy=.026, gx=.010, dB=.030)
        if f >= 55: e = dict(PI.E['shut'], dB=-.020, aB=.02, hw=.100, mU=[-.020, .004, .010, .004, -.020], mL=[-.020, .004, .010, .004, -.020])
        wet = dr = 0.; pit = -.05; ky = 0.; lift = 0.
        if hit >= 0: pit = float(keys(f, [(56, -.05), (57, -.34), (60, -.30), (63, -.12), (80, -.10)])); ky = float(keys(f, [(56, 0.), (57, 30.), (59, 22.), (62, 8.), (66, 2.), (80, 2.)])); wet = 1. if f >= 58 else .6; lift = float(keys(f, [(56, 0.), (57, -.34), (62, -.26), (80, -.22)]))
        if f >= 57: e = dict(PI.E['shut'], dB=-.03, furrow=.8, hw=.105, mU=[-.022, .004, .010, .004, -.022], mL=[-.022, .004, .010, .004, -.022])
        if f >= 61: dr = float(keys(f, [(60, 0.), (61, .30), (62, .85), (63, 1.12), (65, .95), (67, 1.), (80, 1.)]))
        if f >= 63: e = dict(PI.E['drenched'])
        if f >= 65: e = dict(e, gx=.02, gy=-.004)
        lay = Piccolo.bust(cx, cy + ky + (3 * math.sin(g * .25) if f < 56 else 0), sc, -.24, pit, -.02, e, 'lake', g, (-.25, 0) if f < 56 else (-.05, 0), 4, True, wet, dr, 'lotus', byaw=-.16, lift=lift)
        if dk > 0: lay[..., :3] *= 1 - dk * A([.46, .34, .30], f32); img = img * (1 - dk * .30 * (np.exp(-((x - cx - 60) / 420.) ** 2))[..., None])
        img = over(img, lay)
        if hit >= -1: img = self.slam(img, f, cx, cy)
        if f >= 60: img = self.drips(img, f, cx, cy)
        sh = kit.shake(f, 9., 5, 3., 56) if hit >= 0 else (0, 0); z = float(keys(f, [(50, 1.0), (55, 1.035), (56, 1.05), (58, 1.0), (68, 1.0)])); return comp.cam2d(img.astype(f32), 960 + sh[0], 540 + sh[1], z)

    def slam(self, img, f, cx, cy):
        """the column lands on the crown of the turban: ragged foot, a star of spray, sheets that arch, tear and turn to drops, gone"""
        hit = f - 56; top = cy - 1.34 * self.PC[2]; hx = cx + 4; w = 226.
        if hit == -1: return over(img, FW.col(hx, -500., 120., w, (f - 44) / 24., seed=5, flow=-1., foot=True, skirt=0., drops=.6))
        s = (hit + .55) / 24. * 2.8; end = .50
        if s > 1.35: return img
        img = over(img, FW.bell(hx, top + 14, w, s, end, seed=2))
        if s < end: img = over(img, FW.col(hx, -500., top + 60, w * (1. if hit < 3 else .86), (f - 44) / 24., seed=5, flow=-1., foot=True, head=.5, skirt=0., drops=0.))
        if hit <= 1: c_ = Cel(); PW.spray_crown(c_, hx, top + 40, .055 + .05 * hit, 1.15, 'lake', seed=4); img = over(over(img, c_.layer()), FW.star(hx, top + 18, (hit + 1) / 2., 4, spikes=False))
        return kit.impact(img, 'bw', .30, .60, ink='#123c38', paper='#f7fbe6') if hit == 0 else img

    def drips(self, img, f, cx, cy):
        cel = Cel(); sc = self.PC[2]
        src = [(cx - 1.25 * sc, cy + 1.6 * sc, 61), (cx + .86 * sc, cy + .42 * sc, 62), (cx - .56 * sc, cy + .30 * sc, 63), (cx + 2.5 * sc, cy + 1.9 * sc, 60), (cx - 2.7 * sc, cy + 1.9 * sc, 62), (cx + .10 * sc, cy + .98 * sc, 65), (cx - .62 * sc, cy + .32 * sc, 67), (cx + 1.7 * sc, cy + 2.5 * sc, 64), (cx - 1.9 * sc, cy + 2.45 * sc, 65), (cx + .88 * sc, cy + .44 * sc, 66)]
        for x, y, f0 in src:
            t = (f - f0) / 24.
            if t < -.09 or t > .42: continue
            if t < 0: k = 1 + t / .09; self.drop(cel, x, y, 6 + 5 * k, .2); continue
            self.drop(cel, x, y + .5 * 2600 * t * t, 10, min(1.4, t * 6))
        rr = np.random.default_rng(2)
        for j in range(9):                                         # beads that run down cape and arms
            x0, y0 = cx + rr.uniform(-2.6, 2.6) * sc, cy + rr.uniform(1.35, 2.2) * sc; v = rr.uniform(30, 80); yy = y0 + v * (f - 60) / 24. * 2; cel.line(A([(x0, yy - 26), (x0 + 2, yy - 12), (x0, yy)]), 4.5, '#ffffff', taper=(.8, .1), press=0, alpha=.75)
        return over(img, cel.layer())

    def drop(self, cel, x, y, r, st=0.): P = spl(A([(0, -1.6 - st), (.6, -.3), (.72, .35), (0, 1.), (-.72, .35), (-.6, -.3)]) * r + [x, y], True, corners=(0,), n=6); cel.fill(P + [1.5, 2], '#163c38', .6); cel.fill(P, '#e2f6ec'); cel.fill(K.ell(x - r * .25, y + r * .1, r * .18, r * .34), '#ffffff')

    # ------------------------------------------------------------ the laugh
    def laugh(self, f):
        if 'l' not in self._bg: cam = Camera3D((-.4, -11., 1.9), (3., 60, 2.2), 1900.); self._bg['l'] = blur(land.lake(3.2, cam=cam, overlays=False).img.astype(f32), 4.5)
        u = (f - 68) % 3; n_ = (f - 68) // 3; b = (1., -.62, -.10)[u] * (1 - .07 * n_); L = 'lake'; cx, cy, sc = 900. + 10 * b * (-1) ** n_, 524. + 27 * b, 214.; head = (-.24 + .04 * (-1) ** n_, .19 + .13 * b, -.10 + .075 * b * (-1) ** n_); e = dict(G.E['laugh'], hw=.262 + .028 * b, jaw=.13 + .055 * b, mL=[.030, -.235 - .05 * b, -.320 - .07 * b, -.235 - .05 * b, .030], tears=0.); bl = [(-1, (-1.22, .92 + .03 * b, .2), (-.40, 1.02 + .05 * b, .75), 'rest', (-10, 10, -80), 1), (1, (1.20, .92 + .03 * b, .2), (.42, 1.08 + .05 * b, .75), 'rest', (10, 10, 80), 1)]
        img = self._bg['l'].copy(); img = img * .96 + rgb('#fff0b0') * .06
        lay = Gohan.bust(cx, cy, sc, *head, e, L, (f // 2) * 2, (-.22, .06), 6, True, bl, byaw=-.10, bottom=3.2, ballon=False, broll=.05 * b); sq = 1 + .030 * b; M = np.float32([[1 / sq, 0, cx * (1 - 1 / sq)], [0, sq, 980 * (1 - sq)]]); lay = cv2.warpAffine(lay, M, (W, FH), flags=cv2.INTER_LINEAR); img = over(img, lay)
        H = Gohan.make(cx, cy, sc, *head); bx, by, br = G.ball_at(H); bx, by = bx / sq + cx * (1 - 1 / sq), by * sq + 980 * (1 - sq)
        from db import ball as B_
        x, y = grid(FH, W); up = A([math.sin(head[2]), -math.cos(head[2])]); im2 = B_.draw(img.astype(f32), bx, by, br, t=f, light=(-.65, .6, .5), body=1.15, env=None); kp = np.clip((x - bx) * up[0] + (y - by) * up[1] + br * .80 + .5, 0, 1)[..., None].astype(f32); img = img * (1 - kp) + im2 * kp
        gl = float(keys(f, [(68, 0.), (69, 0.), (70, 1.), (72, .75), (76, .25), (80, .4)]))
        if gl > 0: img = self.glint(img, bx - br * .34, by - br * .38, 105 * gl, f)
        cel = Cel()                                                # laughter tears fly off in arcs
        for j, (f0, dx) in enumerate(((69, -1), (71, 1), (74, -1), (76, 1))):
            t = (f - f0) / 24.
            if 0 <= t < .3: px, py = cx + dx * sc * .62 + dx * 330 * t, cy - .02 * sc - 260 * t + 1300 * t * t; self.drop(cel, px, py, 10, 0.)
        img = over(img, cel.layer()); z = float(keys(f, [(68, 1.04), (80, 1.0)])); return comp.cam2d(img.astype(f32), 960, 540 + 5 * b, z)

    def glint(self, img, x, y, r, f):
        cel = Cel(); a0 = .10 * (f - 70) * .25
        for j in range(4): a = a0 + j * math.pi / 2; l = r * (1. if j % 2 else .62); n = A([-math.sin(a), math.cos(a)]) * r * .045; c = A([math.cos(a), math.sin(a)]); cel.fill(A([n, c * l, -n]) + [x, y], '#fffdf0')
        cel.fill(K.ell(x, y, r * .10), '#ffffff'); return over(screen(img, np.clip(radial(x, y, r * 2.2, power=2.4) * .5, 0, 1)[..., None] * rgb('#ffe9a0')), cel.layer())

    def frame(self, f):
        if f < 30: return GL.off_close(f).astype(f32)
        if f < 50: return self.wide(f).astype(f32)
        if f < 68: return self.med(f).astype(f32)
        return self.laugh(f).astype(f32)
