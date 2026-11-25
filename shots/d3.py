"""d3 THREE THREADS (80).  0-35 from high above: three lights weave over the night sea; a clash on every 8th note (5, 10, .. 35) dents the water and punches a ring in the cloud.
36-39 we dive in.  40, 50, 60 inside: three rounds, hits on ones.  70 two fists in two white palms.  Dead hold.  He is smiling."""
import os
from scipy.interpolate import CubicSpline
from db.ki_stage import *
from db import ki, kit, hero3
from db.ki_trail import trail, trail_cel, head, clash_star, sea_dent, punch
KEYS_V = [(-8, (820, 1300, 700)), (-2, (60, 640, 560)), (5, (-520, 250, 360)), (7.5, (-380, 700, 300)), (10, (110, 800, 420)), (12.5, (520, 780, 500)), (15, (580, 330, 320)), (17.5, (410, -80, 260)), (20, (60, 30, 420)), (22.5, (-380, 150, 520)), (25, (-440, 620, 380)), (27.5, (-60, 760, 300)), (30, (350, 130, 460)), (32.5, (260, 560, 600)), (35, (0, 330, 520)), (40, (0, 330, 640))]
KEYS_G = [(-8, (-1300, -500, 250)), (-2, (-960, -20, 330)), (5, (-540, 236, 350)), (7.5, (-640, -90, 460)), (10, (-180, 20, 280)), (12.5, (280, 60, 380)), (15, (560, 316, 328)), (17.5, (760, 130, 460)), (20, (40, 14, 412)), (22.5, (-80, -330, 380)), (25, (-460, 606, 372)), (27.5, (-700, 330, 480)), (30, (-190, 20, 300)), (32.5, (-260, 240, 420)), (35, (-14, 320, 514)), (40, (-60, 300, 640))]
KEYS_B = [(-8, (-950, 1600, 400)), (-2, (-760, 980, 480)), (5, (-420, 520, 560)), (7.5, (-260, 900, 460)), (10, (128, 786, 428)), (12.5, (380, 1060, 300)), (15, (780, 660, 420)), (17.5, (330, 360, 520)), (20, (80, 46, 426)), (22.5, (430, -90, 300)), (25, (700, 320, 420)), (27.5, (620, -120, 520)), (30, (366, 144, 452)), (32.5, (180, -140, 420)), (35, (16, 340, 524)), (40, (60, 360, 640))]
HITS = [(5, 'white'), (10, 'blue'), (15, 'white'), (20, 'both'), (25, 'white'), (30, 'blue'), (35, 'both')]
SP = {k: CubicSpline([q[0] for q in K], np.array([q[1] for q in K], float), bc_type='natural') for k, K in (('violet', KEYS_V), ('white', KEYS_G), ('blue', KEYS_B))}
NL = dict(base='night', mul='#a99fd6', litmul='#f4ecff', shmul='#4c418c')
RV = lambda a=.9: dict(dir=(.7, .5, -.5), col='#d9b4ff', thr=.46, amt=a)


from db.h3_body import Body
Body.HAND.setdefault('catch', ([(30, 52, 34), (32, 54, 34), (34, 56, 36), (38, 58, 38)], 7, ((.31, .28, -.14), (.36, .43, -.27), (.28, .54, -.38))))          # a palm that has closed over somebody else's knuckles
LEG = dict(trail=dict(fL=(.06, -.20, .13), fR=(.07, -.02, .25), kL=(.15, 1, -.2), kR=(.2, 1, .2), tL=(6, -46, -12), tR=(6, -50, -12)), dash=dict(fL=(.05, -.06, .05), fR=(.06, .12, .26), kL=(.1, 1, 0), kR=(.2, 1, .3), tL=(6, -52, -14), tR=(6, -44, -12)), tuck=dict(fL=(.09, .18, .30), fR=(.09, -.14, .20), kL=(.3, 1, .5), kR=(.3, 1, -.1), tL=(6, -26, -6), tR=(6, -50, -12)),
           hang=dict(fL=(.085, .06, .10), fR=(.10, -.10, .17), kL=(.2, 1, 0), kR=(.3, 1, 0), tL=(10, -36, -8), tR=(12, -50, -10)), kickL=dict(fL=(.06, -.10, .24), kL=(.3, 1, .2), tL=(6, -50, -12)))


def AIR(name, leg='trail', tilt=0., side=0., **kw):
    """upper body of a ground pose, legs of a body that hangs or flies: toes pointed, one knee drawn up, whole body pitched"""
    return P(name, **{**dict(air=1, root=(0, 0, 0), body=(tilt, side, 0)), **LEG[leg], **kw})


def top_cam(f): u = f / 36.; return Camera3D((-60 + 150 * u, -930 + 90 * u, 1520 - 110 * u), (30 + 30 * u, 320 + 10 * u, 0), 1500., roll=.05 - .10 * u)


class Shot:
    id = 'd3'; n = 80
    finish = dict(bloom=.28, vig=.18, bloom_thr=.8)
    EVENTS = [(0, 'threads', dict(to=36, note='three lights: gold-white, blue-white, violet; whooshes pan with them'))] + [(k, 'clash', dict(size=.55 + .1 * i, note=w)) for i, (k, w) in enumerate(HITS)] + [(k + 1, 'sea_dent', dict(size=.5)) for k, w in HITS] + [(36, 'dive', dict(to=39, size=.9)), (42, 'punch_miss', dict(size=.5)), (44, 'kick_block', dict(size=1.)), (47, 'vanish', dict(size=.6)), (48, 'kick_whiff', dict(size=.5)), (52, 'kick_block', dict(size=1.1)), (56, 'punch_block', dict(size=.8)), (59, 'backfist_block', dict(size=1.))] + [(k, 'block', dict(size=.8)) for k in (60, 63, 66)] + [(62, 'vanish', dict(size=.4)), (65, 'vanish', dict(size=.4)), (68, 'vanish', dict(size=.4)), (70, 'double_catch', dict(size=1.3, note='both fists stop dead in two palms')), (71, 'silence', dict(to=79, note='nothing at all; he is smiling'))]

    # ------------------------------------------------------------------ plates
    def setup(self):
        cv2.setNumThreads(1); p = 'cache/d3_v5.npz'; self.bc = Camera3D((30, -900, 1500), (40, 325, 0), 1500. * .80)
        if os.path.exists(p): q = np.load(p); self.sea, self.gl, self.lo, self.hi, self.sky, self.skb = [q[k].astype(f32) for k in ('sea', 'gl', 'lo', 'hi', 'sky', 'skb')]
        else: self.paint(); np.savez(p, **{k: getattr(self, k).astype(np.float16) for k in ('sea', 'gl', 'lo', 'hi', 'sky', 'skb')})
        self.G = Fig('goku', 'clean', seed=1); self.V = Fig('vegeta', 'clean', seed=2); self.Z = Fig('frieza', seed=3)

    def paint(self):
        cam = self.bc; U, V, T = ki.sea_uv(cam); x, y = grid(); wv = ki.sea_waves(U, V, T, 0., 11.); w2 = ki.sea_waves(U, V, T, 0., 4.); big = noise(U[::4, ::4] / 620., V[::4, ::4] / 520., 0, 3); big = cv2.resize(big.astype(f32), (W, H), interpolation=cv2.INTER_LINEAR); from db.paint import ramp
        sea = ramp(y / H, [(0, '#302c6e'), (.4, '#191c50'), (1, '#0a0c2e')]).astype(f32); sea = sea * (1 + .20 * np.round(big * 3)[..., None] / 3.); fr = np.exp(-(((U - 160) * .9 + (V - 500) * .30) / 260.) ** 2) * np.clip((V + 900) / 1300., .12, 1); dk = np.clip((-wv - .18) * 26 + .5, 0, 1)[..., None]; a = np.clip((w2 - .40) * 26 + .5, 0, 1)[..., None]; a0 = np.clip((w2 - (1.02 - .72 * fr)) * 26 + .5, 0, 1)[..., None]; a1 = np.clip((w2 - (1.30 - .72 * fr)) * 26 + .5, 0, 1)[..., None]
        sea = sea * (1 - dk * .20); sea = sea * (1 - .55 * a) + a * C('#2f338a') * .55; sea = sea * (1 - a0) + C('#8a78e0') * a0; sea = sea * (1 - a1) + C('#f0e6ff') * a1; c = Cel(); r = R(8)
        for i, (mx, my, rr, hh) in enumerate(((-760, -330, 54, 120), (-930, -210, 80, 190), (-600, -480, 40, 80), (900, 1150, 70, 150), (1040, 1010, 40, 70))):
            a_ = np.linspace(0, TAU, 17)[:-1]; q = rr * (1 + .16 * r.normal(0, 1, 16)); B = cam.project(np.stack([mx + np.cos(a_) * q, my + np.sin(a_) * q, 0 * a_], -1))[0]; Tp = cam.project(np.stack([mx + np.cos(a_) * q * .78, my + np.sin(a_) * q * .78, 0 * a_ + hh], -1))[0]; import scipy.spatial as ssp; hull = lambda Q: Q[ssp.ConvexHull(Q).vertices]
            c.fill(hull(np.vstack([B, B + [-hh * .55, hh * .30]])), '#080a24', .7); c.fill(spl(B + (B - B.mean(0)) * .12, True, n=4), '#b9aef0', .85); c.fill(B, '#141233'); A = hull(np.vstack([B, Tp])); c.fill(A, '#1a1738'); c.clip(A); c.fill(A + [rr * .45, -3], '#3a3470'); c.fill(A + [rr * .80, -3], '#6a58b0')
            for j in range(9): x0 = A[:, 0].min() + np.ptp(A[:, 0]) * r.random(); y0 = A[:, 1].min() + np.ptp(A[:, 1]) * r.random(); c.line([(x0, y0), (x0 + r.normal(0, 2), y0 + 12), (x0 + r.normal(0, 4), y0 + 30)], 1.3, '#0d0b22', taper=(.1, .6), seed=j)
            c.unclip(); c.fill(Tp, '#57499a'); c.fill(Tp * .84 + Tp.mean(0) * .16 + [rr * .08, -2], '#9a86dc');
        self.sea = put(sea, c.layer()).astype(f32); self.gl = w2.astype(f32)
        def deck(seed, n, z_, w, box_, cols, gap=0.):
            rg = R(seed); it = []
            for i in range(n):
                X = np.array([rg.uniform(*box_[0]), rg.uniform(*box_[1]), z_])
                if gap and abs((X[0] - 100) + .25 * (X[1] - 300)) < gap: continue
                p, z = cam.project(X[None]); it.append((z[0], p[0], w * cam.f / z[0] * (.30 + 1.1 * bms(rg.random(), .25, .35)), i))
            pts = []
            for z, p, s, i in sorted(it, key=lambda q: -q[0]): pts += ki.heap(p[0], p[1], s * 2.7, s * 1.15, seed + i, 20, .6)
            return ki.cumulus(pts, (.6, -.8), cols, seed=seed, k=9.)
        self.lo = deck(5, 44, 260, 170, ((-1500, 1500), (-700, 2100)), ('#1c1a4a', '#37317a', '#6a5cb4', '#b9a8ee', '#efe6ff')); self.hi = deck(23, 13, 640, 120, ((-900, 900), (-420, 900)), ('#16143c', '#2c2768', '#5548a0', '#a391e0', '#e6dcff'), 420)
        # two night skies for the rounds inside: cloud towers with a moon, cloud walls
        g = y / H; sk = ramp(g, [(0, '#0a0820'), (.45, '#1f1850'), (.8, '#43307e'), (1, '#6a4a9e')]).astype(f32); m, mg = ki.moon(1420, 260, 120); a = put(sk, m) + mg * .9; far = []; mid = []
        for i in range(9): far += ki.heap(80 + i * 230 + 60 * h1(i, 1), 900 + 40 * h1(i, 2), 360, 260 + 260 * h1(i, 3), i + 60, 26, .4)
        a = put(a, ki.cumulus(far, (.5, -.8), ('#2c2562', '#40357e', '#6250a6', '#9a84d8', '#d6c6fa'), seed=4, k=10.)); 
        for i in range(6): mid += ki.heap(-60 + i * 400 + 90 * h1(i, 4), 1120, 560, 300 + 260 * h1(i, 5), i + 90, 30, .4)
        a = put(a, ki.cumulus(mid, (.5, -.8), ('#15123c', '#2a2362', '#4a3c90', '#7f6cc6', '#c9b8f6'), seed=5, k=12.)); b = ramp(g, [(0, '#1a1346'), (.5, '#3a2a78'), (1, '#140f36')]).astype(f32); w1 = []
        for i in range(16): w1 += ki.heap(W * h1(i, 6) * 1.1 - 80, 220 + 900 * h1(i, 7), 520, 240 + 220 * h1(i, 8), i + 130, 24, .5)
        b = put(b, ki.cumulus(w1, (-.5, -.8), ('#221a54', '#352a74', '#54449c', '#8a74cc', '#c8b4f4'), seed=6, k=12.)); self.sky, self.skb = a.astype(f32), b.astype(f32)

    def warp(self, pl, cam, z):
        Hb = self.bc.H_plane((0, 0, z), (1., 0, 0), (0, 1., 0), 1, 1, 1, 1); Hc = cam.H_plane((0, 0, z), (1., 0, 0), (0, 1., 0), 1, 1, 1, 1); return cv2.warpPerspective(pl, Hc @ np.linalg.inv(Hb), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # ------------------------------------------------------------------ from above
    def top(self, f):
        t = f / 24.; cam = top_cam(f); im = self.warp(self.sea, cam, 0.).copy(); gl = self.warp(self.gl, cam, 0.)
        for i, (k, w) in enumerate(HITS):
            X = SP['violet'](k); L = sea_dent(cam, (X[0], X[1], 0), f - k, 130 + .10 * X[2] + 30 * (w == 'both'), 3 + i * 7, life=34, cols=('#3f3fa8', '#b9aef6', '#ffffff'))
            if L is not None: im = put(im, L)
        out = im; H_ = [(k, SP['violet'](k)) for k, w in HITS]
        def deck(pl, z, out, rad, sh):
            cl = self.warp(pl, cam, z)
            for i, (k, X) in enumerate(H_):
                a = f - k
                if a > 0: p, zz = cam.project(np.array([[X[0], X[1], z]])); r = rad * cam.f / zz[0] * (1 - math.exp(-a / 4.2)) * (.75 + .25 * h1(i, 5)); cl = punch(cl, p[0, 0], p[0, 1], r, .80, i + 3, light=('#fbf4ff', '#cbbaf6'))
            if sh: s_ = cv2.warpAffine(cl[..., 3], np.float32([[1, 0, -70], [0, 1, 46]]), (W, H)); out = out * (1 - .36 * s_[..., None])
            return put(out, cl), cl
        out, cl = deck(self.lo, 260., out, 245., True); LY = []; W_ = dict(violet=13., white=12., blue=11.)
        for kind in ('violet', 'blue', 'white'):
            u = np.linspace(f - 12.5, f, 130); X = SP[kind](u); px, z = cam.project(X); F = trail(px, kind, W_[kind], 5, None, 1500. / z, True, t); lay = trail_cel(F, kind); LY.append((kind, lay, px, z)); old = SP[kind](np.linspace(f - 26, f - 11, 80)); q, zz = cam.project(old); G_ = Field(2); v = lin(80); bd = np.clip(1.6 * noise(np.linspace(f - 26, f - 11, 80) * 1.1, len(kind) * 1., 0, 3) + .25, 0, 1); G_.rib(q, [2.6 * v * bd, 1.2 * v * bd - .2], 0.); out = put(out, G_.cel((pal(kind)['c'][1], pal(kind)['c'][2])) * .8)
            ref = cv2.GaussianBlur(cv2.resize(lay[..., :3], (W // 8, H // 8), interpolation=cv2.INTER_AREA), (0, 0), 9); out = out + cv2.resize(np.roll(ref, 9, 0), (W, H), interpolation=cv2.INTER_LINEAR) * (.32 + .9 * np.clip((gl - .1) * 6, 0, 1))[..., None] * (1 - .55 * cl[..., 3:4]) * .9
        for kind, lay, px, z in LY: out = put(out, lay) + bloom(lay, .45, (6, 22, 70), (.35, .3, .28), pal(kind)['glow'], .8)
        for kind, lay, px, z in LY: d = px[-1] - px[-4]; h_ = head(px[-1, 0], px[-1, 1], W_[kind] * 1500 / z[-1] * .95, kind, f, 3, math.atan2(d[1], d[0])); out = put(out, h_) + bloom(h_, .5, (5, 18, 50), (.3, .3, .25), pal(kind)['glow'], .7)
        out, _ = deck(self.hi, 640., out, 150., False)
        for i, (k, w) in enumerate(HITS):
            X = SP['violet'](k); p, z = cam.project(np.array([X])); L, g = clash_star(p[0, 0], p[0, 1], 52 + 16 * (w == 'both'), f - k, ('white' if w != 'blue' else 'blue', 'violet'), 5 + i, h1(i, 3) * 3)
            if L is not None: out = put(out, L) + g
            a = f - k
            if 0 < a < 18:
                q = ki.ring3(cam, X, (h1(i, 4) - .5, h1(i, 5) - .5, 1.), 60 + 260 * (1 - math.exp(-a / 5.)), 9 * math.exp(-a / 7.), 'base', 5 + i, .35, .9 * math.exp(-a / 9.), spurs=14)
                if q is not None: out = put(out, q)
        return shaken(out, *kick(f, [(k, .5 if w != 'both' else .9) for k, w in HITS], 7., 1.8, 2)) if f > 4 else out

    def dive(self, f):
        a = f - 36; base = self.top(f); cam = top_cam(f); p, z = cam.project(np.array([SP['violet'](36.)])); cx, cy = p[0]; amt = (.18, .45, 1.0, 2.0)[a]; s = 1 + (.10, .5, 1.5, 4.)[a]; im = cv2.warpAffine(base.astype(f32), np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]]), (W, H), borderMode=cv2.BORDER_REFLECT_101); im = kit.zoomblur(im, cx, cy, amt, 26); c = Cel()
        for i in range(120): an = TAU * h1(i, 1); q0 = 60 + 500 * h1(i, 2) ** 1.4; q1 = q0 + 200 + 900 * h1(i, 3); w = .004 + .012 * h1(i, 4); c.fill(np.array([[cx + math.cos(an) * q0, cy + math.sin(an) * q0], [cx + math.cos(an - w) * q1, cy + math.sin(an - w) * q1], [cx + math.cos(an) * q1 * 1.06, cy + math.sin(an) * q1 * 1.06], [cx + math.cos(an + w) * q1, cy + math.sin(an + w) * q1]]), ('#fff3d0', '#bfe0ff', '#c784ff', '#1a1440')[i % 4], (.4, .7, .95, 1.)[a])
        im = put(im, c.layer()); k = (0, .0, .25, .85)[a]; return im * (1 - k) + C('#fbf4ff') * k

    # ------------------------------------------------------------------ inside
    def bg(self, f, which, dx, dy, zoom, rot, blur_, off=(0, 0), dim=1.):
        pl = self.sky if which == 0 else self.skb; x, y = off[0] + dx * f, off[1] + dy * f; M = np.float32([[zoom * math.cos(rot), -zoom * math.sin(rot), 0], [zoom * math.sin(rot), zoom * math.cos(rot), 0]]); M[:, 2] = np.array([W / 2 - x, H / 2 - y]) - M[:, :2] @ np.array([W / 2, H / 2]); im = cv2.warpAffine(pl, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        if blur_[0] or blur_[1]: im = kit.smear(im, blur_[0], blur_[1], 9)
        return im * dim

    def lines(self, f, ang, n, col='#e9ddff', al=.8, seed=1, w=5.):
        c = Cel(ss=1); kit.speed_lines(c, ang, n, w, (300, 1500), col, seed * 100 + f, alpha=al); return c.layer()

    def st(self, F, pose, pos, heading, pitch=0., roll=0., look=None, reach=None, wind=(0, 0, 0)): return F.set(dict(pose=POSES[pose] if isinstance(pose, str) else pose, x=pos[0], y=pos[1], z=pos[2], heading=heading, pitch=pitch, roll=roll, pivot='pelvis', look=look, reach=reach, wind=wind, hand={}))

    def duo(self, cam, f, A, Bz, kA, kZ, vA=None, vZ=None):
        """two or three figures with correct occlusion; speed tails under each"""
        return compose([(q[0], q[1]) for q in A], cam, f)

    def flash(self, img, p, s, f, a, kinds=('white', 'violet'), seed=1, rot=0., ring=True):
        L, g = clash_star(p[0], p[1], s, (0, 2, 5, 7, 8)[a] if a < 5 else 99, kinds, seed, rot)
        if L is not None: img = put(img, L) + g
        if ring and 0 <= a < 6: img = put(img, ki.ring(p[0], p[1], s * (1.2 + 2.6 * (1 - math.exp(-(a + .6) / 2.))), s * .20 * math.exp(-a / 2.2), f, 'base', .45, rot, seed, .30, .95 * math.exp(-a / 4.), spurs=18))
        return img

    def tl(self, img, F, cam, f, kw, v, n=70):
        L = safe(F, cam, f, **kw); return put(img, tails(L, v, n, f)), L

    def A(self, f):
        """Goku against Frieza in the air: 42 a right that meets nothing, 44-45 a shin that meets a forearm (seen), 46 the flash, 47 a counter that meets nobody"""
        k = f - 40; cam = Camera3D((.0 + .012 * k, -5.0, .10), (.03, 0, .12), 1960., roll=-.06 + .004 * k); h = -math.pi / 2; hz = math.pi / 2; G, Z = self.G, self.Z
        gp = [(-1.50, 0, .42), (-1.08, 0, .26), (-.66, -.02, .12), (-.60, 0, .16), (-.60, -.03, .16), (-.57, -.03, .16), (-.60, -.03, .17), (-.66, 0, .18), (-.66, 0, .18), (-.66, 0, .18)][k]; zp = [(.74, .02, .0), (.74, .02, .02), (.82, .06, .05), (.84, .06, .03), (.80, .04, .02), (.83, .04, .01), (.96, .04, .0), (.84, .0, .06), (.56, -.02, .10), (.48, -.02, .10)][k]
        gps = [AIR('punch_wind', 'trail', 34), AIR('punch_wind', 'dash', 46), AIR('punch_hit', 'dash', 44), AIR('kick_wind', 'tuck', 4), AIR('kick', 'kickL', -14), AIR('kick', 'kickL', -16), AIR('kick', 'kickL', -12), AIR('guard', 'tuck', 0), AIR('guard', 'tuck', 0), AIR('guard', 'tuck', 0)][k]; zps = [POSES['hover_guard'], POSES['hover_guard'], AIR('slip', 'hang', -6), AIR('slip', 'hang', -4), AIR('block_high', 'hang', 4), AIR('block_high', 'hang', 2), AIR('block_high', 'hang', -8), AIR('kick_wind', 'hang', 6), AIR('kick_side', 'hang', -10, fR=POSES['kick_side']['fR'], kR=POSES['kick_side'].get('kR', (.6, .4, .5))), AIR('kick_side', 'hang', -10, fR=POSES['kick_side']['fR'], kR=POSES['kick_side'].get('kR', (.6, .4, .5)))][k]
        img = self.bg(k, 0, 46, 4, 1.25, -.05 + .004 * k, (70, 6), (120, -40)); img = put(img, self.lines(f, .06, 46, '#d9ccff', .55, 2, 4.)); self.st(Z, zps, zp, hz); zk = Z.skel(); r = None
        if k in (4, 5, 6): r = dict(fR=dict(p=zk.spot('forearmR') + np.array([.06, -.03, .03]), part='ankle'))
        if k == 2: r = dict(R=dict(p=zk.spot('face') + np.array([.10, -.18, .10]), part='fist'))
        self.st(G, gps, gp, h, reach=r); kw = dict(pal=NL, light=(.45, .7, .5), rim=[RV(), dict(dir=(-.8, .3, -.4), col='#fff0c8', thr=.5, amt=.6)]); ex = dict(expr='grit' if k < 7 else 'shock'); zx = dict(expr='polite' if k < 7 else 'cruel'); v = {0: (120, 60), 1: (150, 60), 2: (150, 50), 4: (70, 10)}.get(k)
        if v: img, _ = self.tl(img, G, cam, f, dict(kw, expr=ex), v)
        if k in (2, 4): a = G.state(0) | dict(pose=gps if k == 9 else AIR('punch_wind', 'dash', 46) if k == 2 else AIR('kick_wind', 'tuck', 4), reach=None); img = put(img, limb_smear(G, cam, a, G.state(0), 'armR' if k == 2 else 'legR', pal=NL, seed=f))
        if k == 8: a = Z.state(0) | dict(pose=AIR('kick_wind', 'hang', 6)); img = put(img, limb_smear(Z, cam, a, Z.state(0), 'legR', pal=NL, seed=f))
        it = [(Z, dict(kw, expr=zx))] + ([(G, dict(kw, expr=ex))] if k < 7 else []); px = lambda X: cam.project(np.asarray(X, float)[None])[0][0]; pc = px(zk.spot('forearmR')); L = compose(it, cam, f)
        if k == 6: img = self.flash(img, pc, 92, f, 0, ('white', 'violet'), 3, .5)          # the light stands BEHIND the two bodies
        if k >= 7: self.st(G, gps, gp, h); LG = safe(G, cam, f, **dict(kw, expr=dict(expr='grit'))); q = ki.vanish(LG, k - 7, 0., 4, 2.2, 'white', 18); L = stack(q, L) if k == 7 else (stack(L, q) if q is not None else L)
        img = put(img, L)
        if k in (4, 5): c = Cel(); [c.fill(np.array([pc + [math.cos(t_) * 30, math.sin(t_) * 30], pc + [math.cos(t_ + .05) * (70 + 60 * h1(k, i)), math.sin(t_ + .05) * (70 + 60 * h1(k, i))], pc + [math.cos(t_ + .12) * 32, math.sin(t_ + .12) * 32]]), '#fffaf0') for i, t_ in enumerate(np.linspace(-2.6, -.4, 7))]; img = put(img, c.layer())          # contact ticks only: the shin on the forearm is the picture
        if k in (6, 7, 8): img = put(img, ki.ring(pc[0], pc[1], 100 + 190 * (k - 5), 20 / (k - 5), f, 'base', .45, .5, 3, .35, .9 / (k - 5), spurs=16))
        if k in (2, 3): q = px(zk.spot('face') + np.array([.14, -.1, .12])); img = put(img, ki.ring(q[0], q[1], 60 + 90 * (k - 2), 14 - 6 * (k - 2), f, 'base', 1.6, .0, 4, .4, .8, spurs=8))
        d = kick(f, [(44, .6), (46, 1.2), (48, .5)], 12., 1.5, 4); return kit.zoomblur(img, 1150, 420, .09, 26) * 1.15 if k == 0 else shaken(img, *d)

    def B(self, f):
        """Vegeta out of the sky: 52-53 heel on crossed arms (seen), 54 flash; 56-57 a straight left stopped by a forearm (seen), 58 flash; 59 the backfist"""
        k = f - 50; cam = Camera3D((.12, -4.7, -.12), (.0, 0, .16), 1800., roll=.07 - .004 * k); h = -math.pi / 2; hz = math.pi / 2; V, Z = self.V, self.Z
        vp = [(-1.45, 0, 1.15), (-1.02, 0, .80), (-.50, -.04, .46), (-.49, -.04, .44), (-.78, 0, .52), (-1.02, 0, .26), (-.62, -.03, .12), (-.60, -.03, .12), (-.66, -.02, .12), (-.92, 0, .14)][k]; vpz = [POSES['air_kick'], POSES['air_kick'], P(POSES['air_kick'], fR=(.10, .66, .24), body=(-24, -14, 0)), P(POSES['air_kick'], fR=(.10, .66, .24), body=(-24, -14, 0)), POSES['tuck'], AIR('punch_wind', 'trail', 30), AIR('punch_hit', 'dash', 40), AIR('punch_hit', 'dash', 40), AIR('punch_hit', 'dash', 34), AIR('block_high', 'trail', -14)][k]; vpt = [.5, .5, .34, .34, 0, 0, 0, 0, 0, 0][k]
        zp = [(.70, .02, .02), (.70, .02, .02), (.64, .02, -.02), (.67, .02, -.09), (.74, .02, -.22), (.72, .02, -.08), (.72, .04, .02), (.74, .04, .02), (.80, .05, .0), (.62, .0, .04)][k]; zps = [POSES['hover_guard'], AIR('block_cross', 'hang', -8), AIR('block_cross', 'hang', -14), AIR('block_cross', 'hang', -18), AIR('block_cross', 'hang', -22), AIR('guard', 'hang', 4), AIR('block_high', 'hang', 2), AIR('block_high', 'hang', 0), AIR('block_high', 'hang', -8), AIR('backfist', 'hang', 10)][k]
        img = self.bg(k, 1, 30, -26, 1.2, .06, (46, 40) if k < 4 else (60, 0), (0, 60)); img = put(img, self.lines(f, .75 if k < 4 else .05, 44, '#cfe0ff', .55, 4, 4.)); self.st(Z, zps, zp, hz); zk = Z.skel(); r = None; cx = (zk.spot('wristL') + zk.spot('wristR')) / 2
        if k in (2, 3): r = dict(fR=dict(p=cx + np.array([-.03, -.03, .05]), part='ankle'))
        if k in (6, 7, 8): r = dict(R=dict(p=zk.spot('forearmR') + np.array([-.02, -.02, 0]), part='fist', body=.5))
        self.st(V, vpz, vp, h, vpt, reach=r)
        if k == 9: self.st(Z, zps, zp, hz, reach=dict(R=dict(p=V.skel().spot('forearmL') + np.array([.02, 0, 0]), part='back', body=.5)))
        kw = dict(pal=NL, light=(-.4, .7, .5), rim=[RV(), dict(dir=(-.8, .4, -.4), col='#bfe0ff', thr=.5, amt=.7)]); vx = dict(expr='shout' if k < 9 else 'pain'); zx = dict(expr='polite' if k < 9 else 'contempt'); v = {0: (140, 110), 1: (160, 130), 2: (150, 110), 5: (60, 10), 6: (130, 0)}.get(k)
        if v: img, _ = self.tl(img, V, cam, f, dict(kw, expr=vx), v)
        if k == 2: a = V.state(0) | dict(pose=POSES['air_kick'], x=-1.0, z=.78, reach=None); img = put(img, limb_smear(V, cam, a, V.state(0), 'legR', pal=NL, seed=f))
        if k == 6: a = V.state(0) | dict(pose=AIR('punch_wind', 'trail', 30), x=-.95, z=.22, reach=None); img = put(img, limb_smear(V, cam, a, V.state(0), 'armR', pal=NL, seed=f))
        if k == 9: a = Z.state(0) | dict(pose=AIR('guard', 'hang', 4), reach=None); img = put(img, limb_smear(Z, cam, a, Z.state(0), 'armR', pal=NL, seed=f))
        px = lambda X: cam.project(np.asarray(X, float)[None])[0][0]; pc = px(cx) if k < 5 else px(zk.spot('forearmR')) if k < 9 else px(V.skel().spot('forearmL')); L = compose([(Z, dict(kw, expr=zx)), (V, dict(kw, expr=vx))], cam, f)
        if k in (4, 8): img = self.flash(img, pc, 96, f, 0, ('blue', 'violet'), 4 + k, .8)
        img = put(img, L)
        if k in (2, 3, 6, 7): c = Cel(); [c.fill(np.array([pc + [math.cos(t_) * 28, math.sin(t_) * 28], pc + [math.cos(t_ + .05) * (66 + 60 * h1(k, i)), math.sin(t_ + .05) * (66 + 60 * h1(k, i))], pc + [math.cos(t_ + .12) * 30, math.sin(t_ + .12) * 30]]), '#f4faff') for i, t_ in enumerate(np.linspace(-3.0, -.2, 8))]; img = put(img, c.layer())
        for t0, rot in ((4, .8), (8, -.1)):
            e = k - t0
            if 0 <= e < 4: img = put(img, ki.ring(pc[0] if e == 0 else px(cx if t0 == 4 else zk.spot('forearmR'))[0], pc[1] if e == 0 else px(cx if t0 == 4 else zk.spot('forearmR'))[1], 110 + 200 * e, 20 / (e + 1), f, 'base', .45, rot, 4 + t0, .35, .9 / (e + 1), spurs=16))
        return shaken(img, *kick(f, [(52, .6), (54, 1.3), (56, .4), (58, 1.0), (59, .9)], 12., 1.5, 6)) if k else kit.zoomblur(img, 1000, 480, .09, 26) * 1.15

    def Cc(self, f):
        """both at once: contacts at 60-61, 63-64, 66-67 (each seen for two frames), light at 62, 65, 68; he barely turns.  69 both wind up"""
        k = f - 60; e = min(3, k // 3); sub = k - 3 * e; cam = Camera3D((.10, -5.6, .12), (-.22, 0, .18), 1800., roll=.035 * math.sin(e * 2.1 + .4)); G, V, Z = self.G, self.V, self.Z; hz = math.pi / 2; r8 = -math.pi / 2
        SET = [dict(g=((-.82, 0, .46), AIR('jab', 'dash', 40)), v=((-1.35, .2, -.42), AIR('guard', 'trail', 24)), z=AIR('block_high', 'hang', 2), gr=('R', 'forearmR'), vr=None, b=0, l=.16), dict(g=((-1.30, .25, -.40), AIR('punch_wind', 'trail', 30)), v=((-.80, -.10, .92), P(POSES['air_kick'], fR=(.10, .66, .22), body=(-24, -14, 0))), z=AIR('block_high', 'hang', -8), gr=None, vr=('fR', 'forearmR'), b=1, l=.62),
               dict(g=((-.86, .16, .62), AIR('elbow', 'tuck', 20)), v=((-.84, -.16, -.34), AIR('punch_hit', 'trail', 24)), z=AIR('block_cross', 'hang', -4), gr=('R', 'forearmR'), vr=('R', 'forearmL'), b=0, l=-.3), dict(g=((-1.30, .2, .66), AIR('punch_wind', 'tuck', 26)), v=((-1.24, -.2, -.30), AIR('punch_wind', 'trail', 16)), z=POSES['hover'], gr=None, vr=None, b=1, l=.1)][e]
        push = .05 * sub + (.10 if sub == 2 else 0); zp = (.52 + push + .03 * e, 0, .0); img = self.bg(k, SET['b'], (-40, 40)[e % 2], 10, 1.35, .10 * math.sin(e * 1.7), (60, 22), (200 * e, 0)); img = put(img, self.lines(f, SET['l'], 56, '#e2d6ff', .6, 6, 5.)); kw = dict(pal=NL, light=(.2, .8, .5), rim=[RV()])
        self.st(Z, SET['z'], zp, hz); zk = Z.skel(); gr = {SET['gr'][0]: dict(p=zk.spot(SET['gr'][1]) + np.array([-.02, -.02, 0]), part='fist', body=.6)} if SET['gr'] and sub < 3 else None; vr = ({SET['vr'][0]: dict(p=zk.spot(SET['vr'][1]) + np.array([-.02, -.02, .03]), part='ankle' if SET['vr'][0] == 'fR' else 'fist', body=.6)}) if SET['vr'] else None
        self.st(G, SET['g'][1], SET['g'][0], r8, .0, reach=gr); self.st(V, SET['v'][1], SET['v'][0], r8, .40 if e == 1 else 0., reach=vr); sg, sv = dict(G.state(0)), dict(V.state(0)); px = lambda X: cam.project(np.asarray(X, float)[None])[0][0]
        if e > 0 and sub < 2:                                      # where they were a moment ago: slivers
            P0 = [((-.82, 0, .46), AIR('jab', 'dash', 40), (-1.35, .2, -.42), AIR('guard', 'trail', 24)), ((-1.30, .25, -.40), AIR('punch_wind', 'trail', 30), (-.80, -.10, .92), POSES['air_kick']), ((-.86, .16, .62), AIR('elbow', 'tuck', 20), (-.84, -.16, -.34), AIR('punch_hit', 'trail', 24))][e - 1]
            for F_, po, q, kd, sd, an in ((G, P0[1], P0[0], 'white', 4 + e, .3), (V, P0[3], P0[2], 'blue', 9 + e, -.2)):
                self.st(F_, po, q, r8); l = ki.vanish(safe(F_, cam, f, pal=NL, lod=1, shadow=False), 1 + sub, an, sd, 1.5, kd, 14); img = put(img, l) if l is not None else img
        G.set(sg); V.set(sv); kg = dict(kw, expr=dict(expr='rage' if e < 3 else 'grit')); kv = dict(kw, expr=dict(expr='shout'))
        if sub == 0 or e == 3: img, _ = self.tl(img, G, cam, f, kg, (110, -30), 50); img, _ = self.tl(img, V, cam, f, kv, (110, 30), 50)
        CP = [px(zk.spot(q[1])) for q in (SET['gr'], SET['vr']) if q]; L = compose([(Z, dict(kw, expr=dict(expr='polite'))), (G, kg), (V, kv)], cam, f)
        if sub == 2 and e < 3:
            for i, p_ in enumerate(CP): img = self.flash(img, p_, 84, f, 0, (('white', 'violet') if i == 0 and e != 1 else ('blue', 'violet')), 20 + e + i, .4 * e)
        img = put(img, L)
        if sub < 2 and e < 3:
            for p_ in CP: c = Cel(); [c.fill(np.array([p_ + [math.cos(t_) * 24, math.sin(t_) * 24], p_ + [math.cos(t_ + .06) * (54 + 50 * h1(k, i)), math.sin(t_ + .06) * (54 + 50 * h1(k, i))], p_ + [math.cos(t_ + .14) * 26, math.sin(t_ + .14) * 26]]), '#fffaf0') for i, t_ in enumerate(np.linspace(-3.0, .4, 8))]; img = put(img, c.layer())
        if sub == 2 and e < 3:
            for p_ in CP: img = put(img, ki.ring(p_[0], p_[1], 150, 15, f, 'base', .45, .4 * e, 20 + e, .35, .8, spurs=16))
        if e == 3:
            for i, P_ in enumerate((SET['g'][0], SET['v'][0])): q = px(np.array(P_) + [-.10, 0, .30]); img = put(img, ki.charge(q[0], q[1], 22, f, ('white', 'blue')[i], 3 + i, .7, 7., False, False))
        return shaken(img, *kick(f, [(60, .5), (62, 1.), (63, .5), (65, 1.1), (66, .5), (68, 1.2)], 11., 1.3, 8)) if k else kit.zoomblur(img, 900, 500, .06, 26) * 1.10

    # ------------------------------------------------------------------ 70
    def dset(self, k=0):
        """two fists shut in two palms: every arm locked straight, bodies placed from the hands back"""
        G, V, Z = self.G, self.V, self.Z; hz = math.pi / 2 + .62; e_ = es(min(1., max(0., (k - 2) / 7.))); cam = Camera3D((-.10 + .26 * e_, -4.4, .35), (-.10 + .26 * e_, 0, .33), 3150. + 470. * e_); zp = (.60, .06, -.08); u = lambda v: np.asarray(v, float) / np.linalg.norm(v); dR, dL = u((-.95, -.16, .26)), u((-.90, -.34, -.24)); alen = lambda sk, F: (sk.b['arm'] + sk.b['fore']) * sk.H * sk.b['arml']; hl = lambda sk: sk.b['hand'] * sk.H
        def zset(pg, pv): self.st(Z, 'hover', zp, hz, look=(-1.0, -3.0, .30), reach=dict(R=dict(p=pg, part='palm', palm=dR, fingers=(0, .05, 1.), g='catch', body=0., stretch=1.02), L=dict(p=pv, part='palm', palm=dL, fingers=(0, -.10, 1.), g='catch', body=0., stretch=1.02))); Z.hand = dict(L=1.42, R=1.42)
        self.st(Z, 'hover', zp, hz); sk = Z.skel(); pg = sk.sh['R'] + dR * alen(sk, Z) * .985; pv = sk.sh['L'] + dL * alen(sk, Z) * .985; zset(pg, pv); sk = Z.skel(); pg = sk.sh['R'] + dR * (alen(sk, Z) * .97 + .02); pv = sk.sh['L'] + dL * (alen(sk, Z) * .97 + .02); zset(pg, pv)
        for F, p_, d, pos, pit, sd in ((G, pg, -dR, np.array([-1.2, .1, .6]), .16, 'R'), (V, pv, -dL, np.array([-1.1, -.3, -.4]), -.26, 'R')):
            hd = -math.atan2(d[0], d[1]); fist = p_ + d * .030
            for it in range(5):
                self.st(F, P(POSES['fly_punch'], body=(62, 0, 0), hR=(.02, .20, .97)), tuple(pos), hd, pit, reach={sd: dict(p=fist, part='fist', body=0., stretch=1.0, pole=(0, -.3, -1.))}); k_ = F.skel(); want = k_.sh[sd] + d * (alen(k_, F) * .992 + .60 * hl(k_)); pos = pos + (fist - want)
        return cam, pg, pv

    def D(self, f):
        k = f - 70; cam, pg, pv = self.dset(k); G, V, Z = self.G, self.V, self.Z; kw = dict(pal=NL, light=(-.30, .75, .5), rim=[RV()]); q = min(f, 72); it = [(Z, dict(kw, expr=dict(expr='polite'), light=(-.55, .55, .55), lines=1.1)), (G, dict(kw, expr=dict(expr='rage'), rim=[RV(), dict(dir=(-.8, .4, -.4), col='#fff0c8', thr=.5, amt=.7)])), (V, dict(kw, expr=dict(expr='shout'), light=(.42, .78, .40), rim=[RV(), dict(dir=(-.8, .4, -.4), col='#bfe0ff', thr=.5, amt=.7)]))]
        for F in (G, V): F.blow(-1.2 if k < 2 else -.25, 0, 0)
        L = compose(it, cam, q); base = self.bg(0, 0, 0, 0, 1.12, -.02, (0, 0), (150, -60), .92); img = comp.bokeh(np.dstack([base, np.ones((H, W), f32)]).astype(f32), 4.5)[..., :3] if k >= 3 else base; px = lambda X: cam.project(np.asarray(X, float)[None])[0][0]; a, b = px(pg), px(pv); c = (a + b) / 2
        # the catching fingers come round in front of the knuckles: lay Frieza's hands over the fists
        HZ = safe(Z, cam, q, **it[0][1]); mk = Z.mask('handR', 'handL'); fr = HZ * mk[..., None]
        if k == 0: return ki.impact(put(img, L), 1, float(c[0]), float(c[1]), L, 7, '#c784ff', dirn=.1)
        if k < 3: img = kit.smear(img, (60, 24)[k - 1], 0, 9); img = put(img, self.lines(f, .0, 40, '#d9ccff', .5, 8, 4.))
        for j, (p, kn) in enumerate(((a, 'white'), (b, 'blue'))):
            e = k - 1 + .4; R_ = 80 + 760 * (1 - math.exp(-e / 2.0)); al = math.exp(-e / 2.2)
            if al > .06: img = ki.refract_ring(img, p[0], p[1], R_, 60, 26 * al, .42, math.pi / 2 - .1) if j == 0 else img; img = put(img, ki.ring(p[0], p[1], R_, 26 * al + 1, f, 'base', .40, math.pi / 2 - .12 + .25 * j, 4 + j, .25, al, spurs=20))
        if k < 4:
            for p, kn in ((a, 'white'), (b, 'blue')): img = self.flash(img, p, 100, f, k, (kn, 'violet'), 5, .3, False)
        img = put(img, L); img = put(img, fr); return shaken(img, *kick(f, [(70, 1.6)], 14., 1.2, 5)) if k < 5 else img

    def frame(self, f):
        if f < 36: o = self.top(f)
        elif f < 40: o = self.dive(f)
        elif f < 50: o = self.A(f)
        elif f < 60: o = self.B(f)
        elif f < 70: o = self.Cc(f)
        else: o = self.D(f)
        return np.ascontiguousarray(o).astype(f32)
