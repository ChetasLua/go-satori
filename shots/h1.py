"""h1 HUNGRY.  Straight down on a crater lake at dawn: a small round islet in a mirror of pink sky, two men on their backs, head beside head like a yin-yang, boots in the water.
Nothing for two beats.  24 a stomach growls (Goku), 36 a second, bigger one answers (Vegeta), 44 laughter.  Birds cross the reflected sky; one petal of ash comes down."""
import numpy as np, cv2, math
from db import W, H, comp, land
from db.paint import rgb, f32, smooth, fbm, grid, mix
from db.comp import over, blur
from db.ink import Cel, spl
from db.hero3 import Fig, P, POSES, compose, mix_pose

ROT = math.radians(-17)                                  # the axis they lie on
GAP = (.83, .775, .213)                                  # distance of the pelvises from the middle, and the two lines they lie on
LIGHT = (.78, .42, .46)
G1, G2, LAUGH = 24, 36, 44


JOLT = (.25, 1., .15, .55, .08, .25, 0.)                  # a startled hand, on twos
ARM = [(52, (.62, -.12, .62)), (54, (.76, -.04, .42)), (56, (.80, .18, .08)), (58, (.56, .42, -.28)), (60, (.10, .50, -.50)), (62, (-.18, .42, -.56)), (64, (-.18, .40, -.56))]


def rot(x, y): c, s = math.cos(ROT), math.sin(ROT); return x * c - y * s, x * s + y * c


def over_pm(a, b): return b + a * (1 - b[..., 3:4])


def two(f): return (int(f) // 2) * 2


def breath(f, ph): return math.sin(2 * math.pi * (f + ph) / 46.)


def goku_pose(f):
    g = two(f); b = breath(g, 0) if g < G1 else breath(g, 0) * .6
    p = dict(cst=(-2.2 * b, 0, 0), cL=(.5 * b, 0), cR=(.5 * b, 0), hL=(.86, -.13, -.30), hR=(.62, -.12, .62), eL=(.3, -1, .2), eR=(.9, -.6, -.2), fL=(.20, .06, .18), kL=(1., .7, 0), fR=(.10, -.075, .035), hed=(0, 0, 0), nck=(2, 0, 0))
    e = 'serene'
    if g >= G1:
        u = g - G1; tw = 9. * math.exp(-u / 4.) * math.cos(u * 1.7); p['wst'] = (tw * .8, 0, 0); p['cst'] = (-2.2 * b - tw * .7, 0, 0); j = JOLT[min(u // 2, 6)]
        p['hL'] = (.86 - .10 * j, -.13 + .34 * j, -.30 + .12 * j); p['hR'] = (.62, -.12 + .26 * j, .62 - .05 * j); p['gL'] = 'splay' if j > .3 else 'open'; p['fL'] = (.20, .06 + .07 * j, .18); p['cL'] = p['cR'] = (4. * j, 0)
        if u >= 2: e = 'calm'
    if g >= G2 + 4: e = 'smile'; p['hed'] = (0, 0, -8 * smooth(G2 + 4, G2 + 8, g))
    if g >= LAUGH:
        u = g - LAUGH; e = 'laugh'; th = [0, 1, .45, .9, .2][(u // 2) % 5]; k = smooth(0, 4, u)
        p['cst'] = (10 * th * k, 0, 3 * math.sin(u * .8) * k); p['wst'] = (5 * th * k, 0, 0); p['nck'] = (2 - 12 * k + 8 * th * k, 0, 0); p['hed'] = (-8 * k - 7 * th * k, 0, 16 * math.sin(u * .34) * k); p['cL'] = p['cR'] = (6. * th * k, 0)
        # the left arm beats the grass on 52, 60, 68; the knee knocks
        q = [(LAUGH, 0.), (LAUGH + 4, .9), (52, 0.), (56, .85), (60, 0.), (64, .75), (68, 0.), (80, 0.)]; a = float(comp.lerp_keys(g, q, lambda t: t * t)); p['hL'] = (.86 - .10 * a, -.13 + .20 * a, -.30 + .74 * a); p['fL'] = (.20 + .05 * th * k, .06 + .07 * th * k, .18 - .02 * th * k); p['fR'] = (.10 + .02 * th * k, -.075 + .05 * th * k, .035)
        p['hR'] = tuple(comp.lerp_keys(g, ARM, lambda t: t)); w = float(smooth(54, 62, g)); p['eR'] = tuple(np.array((.9, -.6, -.2)) * (1 - w) + np.array((1., .2, -.4)) * w); p['gR'] = 'open' if g < 60 else 'relax'
    return P('lie', **p), e


V_LAUGH = dict(eo=0., look=(0, 0), brow=(.05, .0), mouth='shout', jaw=.15)


def vegeta_pose(f):
    g = two(f); b = breath(g, 17)
    p = dict(cst=(-2.0 * b, 0, 0), cL=(.4 * b, 0), cR=(.4 * b, 0), hL=(.80, -.13, -.46), hR=(-.16, .36, -.60), eL=(.3, -1, .2), eR=(1., .1, -.4), gR='relax', fL=(.09, -.07, .03), fR=(.13, -.07, .05), hed=(0, 0, 0), nck=(4, 0, 0))
    e = 'peace'
    if G1 + 4 <= g < G2: e = 'scorn'; p['hed'] = (0, 0, -6)                        # one eye on Goku
    if g >= G2:
        u = g - G2; tw = 11. * math.exp(-u / 4.5) * math.cos(u * 1.6); p['wst'] = (tw * .8, 0, 0); p['cst'] = (-tw * .7, 0, 0); j = JOLT[min(u // 2, 6)] * 1.15; p['hR'] = (-.16 + .10 * j, .36 + .34 * j, -.60 + .10 * j); p['gR'] = 'splay' if j > .3 else 'relax'; p['cL'] = p['cR'] = (4. * j, 0); p['fR'] = (.13, -.07 + .06 * j, .05); p['hL'] = (.80, -.13 + .16 * j, -.46); e = 'shock' if u < 6 else 'away'
        if u >= 6: p['hed'] = (0, 0, 34 * smooth(6, 10, u)); p['nck'] = (4, 0, 14 * smooth(6, 10, u))
    if g >= 54:
        u = g - 54; th = [0, 1, .4, .8, .15][(u // 2) % 5]; k = smooth(0, 4, u); e = V_LAUGH if u >= 2 else 'smile'
        p['hed'] = (-8 * k - 6 * th * k, 0, 34 * (1 - k) - 10 * math.sin(u * .4) * k); p['nck'] = (4 - 10 * k + 7 * th * k, 0, 14 * (1 - k)); p['cst'] = (9 * th * k, 0, -3 * math.sin(u * .8) * k); p['wst'] = (4 * th * k, 0, 0); p['cL'] = p['cR'] = (5.6 * th * k, 0); p['fL'] = (.09 + .03 * th * k, -.07 + .05 * th * k, .03); p['fR'] = (.13, -.07 + .03 * (1 - th) * k, .05)
        fall = float(comp.lerp_keys(g, [(56, 0.), (62, .55), (66, 1.), (80, 1.)], lambda t: t * t)); up = math.sin(math.pi * min(1, fall)) * .5
        p['hR'] = tuple(np.array((-.16, .36, -.60)) * (1 - fall) + np.array((.84, -.13, -.20)) * fall + np.array((0, up, 0))); p['eR'] = tuple(np.array((1., .1, -.4)) * (1 - fall) + np.array((.3, -1, .2)) * fall); p['wR'] = (60 * fall, 0, 0); p['gR'] = 'relax' if fall < .4 else 'open'
    return P('lie', **p), e


def figs(g):
    A = Fig('goku', 'waist'); pa, ea = goku_pose(g); x, y = rot(-GAP[0], GAP[2]); A.pose(pa).place(x, y, .265, heading=math.pi / 2 + ROT, pivot='pelvis')
    B = Fig('vegeta', 'shattered'); pb, eb = vegeta_pose(g); x, y = rot(GAP[1], -GAP[2]); B.pose(pb).place(x, y, .255, heading=-math.pi / 2 + ROT, pivot='pelvis'); return A, B, ea, eb


class Shot:
    id = 'h1'; n = 80
    finish = dict(bloom=.20, vig=.10, soft=.18, halation='#ffe6d2')
    EVENTS = [(0, 'lake_bed', dict(to=80, size=.2, note='small water, first light, almost nothing')), (4, 'breath', dict(size=.15)), (24, 'growl', dict(size=.6, who='goku', note='long, honest stomach growl, silence round it')), (26, 'ring', dict(size=.15, note='water answers')),
              (36, 'growl', dict(size=1., who='vegeta', note='deeper, longer, ends in a question mark')), (38, 'ring', dict(size=.25)), (44, 'laugh', dict(to=80, who='goku', size=.8)), (46, 'birds', dict(to=74, size=.3, note='wings high above, right to left')),
              (52, 'slap_grass', dict(size=.35)), (54, 'laugh', dict(to=80, who='vegeta', size=.6, note='first a snort')), (60, 'slap_grass', dict(size=.35)), (66, 'arm_falls', dict(size=.3)), (68, 'slap_grass', dict(size=.3)), (70, 'petal', dict(size=.1, note='lands on the water'))]

    def setup(self):
        self.cam = land.top_cam(z=4.75); self.L = land.islet_layers(self.cam); self.x, self.y = grid(); self.fc = {}
        U = self.L['up']; self.U = U.astype(f32); self.n1 = fbm(H, W, (90, 34), (1, .5), 3, aspect=1.6); self.n2 = fbm(H, W, (70, 26), (1, .5), 9, aspect=1.6); self.px = self.cam.f / self.cam.eye[2]
        # drop shadow of the land on the water, foot of the bank
        a = self.L['land'][..., 3]; self.shw = np.clip(cv2.GaussianBlur(cv2.warpAffine(a, np.float32([[1, 0, -15], [0, 1, 8]]), (W, H)), (0, 0), 3.) - a, 0, 1)
        self.petal = [(40, 1250., -30., 2.7), (46, 1430., 60., 2.2), (52, 1385., 150., 1.8), (58, 1540., 240., 1.45), (64, 1500., 330., 1.15), (70, 1590., 402., .95), (80, 1598., 406., .95)]
        self.ring_src = []

    def rings(self, f):
        """(x px, y px, radius px, width px, amp)"""
        R = []; c = (W / 2, H / 2)
        R += [(*c, 1e9, 0., 0.)][:0]
        A, B = self.feet
        for (q, amp, per, ph) in [(A[0], .30, 26, 0), (A[1], .24, 30, 11), (B[0], .26, 28, 6), (B[1], .24, 32, 17)]:
            for k in range(3):
                t = (f + ph) % per + k * per
                R.append((q[0], q[1], 10 + 2.5 * t, 3.2 + t * .05, amp * math.exp(-t / 22.) * (1. if f < LAUGH else 1.5)))
        if f >= 70: t = f - 70; [R.append((1590., 402., 4 + 3.1 * max(0, t - k * 3), 2.6, .5 * math.exp(-t / 7.) * (t > k * 3))) for k in range(2)]
        return R

    def water(self, f):
        t = f / 24.; x, y = self.x, self.y; hh, ww = self.U.shape[:2]; dx = 2.2 * np.sin(t * 1.1) * self.n1 + 2.0 * np.cos(t * .8) * self.n2; dy = 1.7 * np.cos(t * .9 + 1) * self.n1 + 1.5 * np.sin(t * 1.2) * self.n2; li = np.zeros((H, W), f32)
        for (cx, cy, R, w, a) in self.rings(f):
            if a < .02: continue
            m = int(R + w * 4 + 2); x0, x1, y0, y1 = max(0, int(cx - m)), min(W, int(cx + m)), max(0, int(cy - m)), min(H, int(cy + m))
            if x1 <= x0 or y1 <= y0: continue
            X, Y = x[y0:y1, x0:x1] - cx, y[y0:y1, x0:x1] - cy; d = np.sqrt(X * X + Y * Y) + 1e-3; q = (d - R) / w; e = np.exp(-q * q) - .55 * np.exp(-(q + 1.9) ** 2); s = a * e; g = -2 * q * np.exp(-q * q) * a; dx[y0:y1, x0:x1] += g * X / d * 7; dy[y0:y1, x0:x1] += g * Y / d * 7; li[y0:y1, x0:x1] += s
        # rings that leave the shore keep the shape of the shore, broken like a drawn line
        sh = self.L['s']; brk = smooth(-.55, .15, self.n1 + .35 * np.sin(5 * np.arctan2(y - 540, x - 960)))
        for (f0, n, amp, v) in ((G1 + 1, 2, .55, .040), (G2 + 1, 3, .80, .044), (52, 1, .34, .036), (60, 1, .34, .036), (68, 1, .30, .036)):
            for k in range(n):
                tt = f - f0 - 5 * k
                if tt <= 0: continue
                am = amp * math.exp(-tt / 10.) * (.72 ** k)
                if am < .03: continue
                R_ = .97 + v * tt - .00022 * tt * tt; w = .0095 + .0006 * tt; q = (sh - R_) / w; e = np.exp(-q * q) - .5 * np.exp(-(q + 1.8) ** 2); li += am * e * (.25 + .75 * brk); gq = am * -2 * q * np.exp(-q * q); d = np.hypot(x - 960, y - 540) + 1; dx += gq * (x - 960) / d * 5; dy += gq * (y - 540) / d * 5
        img = cv2.remap(self.U, (x + (ww - W) / 2 - 5.5 * t + dx).astype(f32), (y + (hh - H) / 2 + 1.2 * t + dy).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        b = self.birds(f)
        if b is not None: b = cv2.remap(b, (x + dx * 1.6).astype(f32), (y + dy * 1.6).astype(f32), cv2.INTER_LINEAR); img = over(img, b)
        k = self.L['tint']; img = img + (rgb('#575a80') - img) * k[..., None]; img = img * (1 - .34 * self.shw[..., None])
        up = np.clip(li, 0, 1); img = img + (rgb('#fff3e8') - img) * (smooth(.10, .22, up) * np.clip(.25 + up, 0, .85))[..., None]; dn = np.clip(-li, 0, 1); img = img + (rgb('#8b7fa6') - img) * (smooth(.06, .16, dn) * .40)[..., None]
        return img

    def birds(self, f):
        """five birds crossing the sky, seen upside down in the lake"""
        if not 45 <= f <= 80: return None
        c = Cel(); g = two(f); t = (f - 46) / 24.
        for i, (ox, oy, s, ph) in enumerate([(0, 0, 1., 0), (64, 40, .92, 2), (70, -46, .95, 3), (140, 82, .85, 1), (150, -96, .88, 4)]):
            cx, cy = 2010 + ox - 900 * t, 250 + oy + 160 * t + 8 * math.sin(t * 2 + i); fl = [0., .6, 1., .5, -.2, -.5][((g // 2) + ph) % 6]; S = 25 * s; a = math.atan2(160, -900); ca, sa = math.cos(a), math.sin(a)
            def T(Q): Q = np.asarray(Q, float) * S; return np.c_[cx + Q[:, 0] * ca - Q[:, 1] * sa, cy + Q[:, 0] * sa + Q[:, 1] * ca]
            sp = 1.5 - .55 * abs(fl); sw = .30 + .35 * fl
            wing = lambda d: spl(np.array([(.30, d * .06), (.22 - sw * .4, d * sp * .5), (-.25 - sw * .7, d * sp), (-.18 - sw * .4, d * sp * .55), (-.30, d * .10)]), closed=True, corners=(2,), n=6)
            col = '#6f5f90'; c.fill(T(wing(1)), col, .80); c.fill(T(wing(-1)), col, .80); c.fill(T(spl(np.array([(.75, 0), (.40, .10), (-.30, .11), (-.95, .18), (-.85, 0), (-.95, -.18), (-.30, -.11), (.40, -.10)]), closed=True, n=5)), col, .86)
        return c.rgba()

    def actors(self, f):
        g = two(f)
        if g not in self.fc:
            A, B, ea, eb = figs(g); k = dict(light=LIGHT, pal='dawn', rim=dict(dir=(.85, .45, -.30), col='#fff0d8', thr=.5, amt=.8)); ex = lambda e, **q: dict(expr=e, scratch=1, **q)
            LA = A.draw(self.cam, g, **dict(k, expr=ex(ea, dmg=.6))); mb = A.mask('band'); lu = LA[..., :3].mean(-1) / np.maximum(LA[..., 3], .02); t_ = (mb * smooth(.42, .60, lu))[..., None]; LA = np.dstack([LA[..., :3] * (1 - t_) + rgb('#4169b8') * LA[..., 3:4] * t_, LA[..., 3]])
            LB = B.draw(self.cam, g, **dict(k, expr=ex(eb))); hd = [self.cam.project(F.skel().head)[0] for F in (A, B)]; L = np.zeros((H, W, 4), f32)
            for Lx, (hx_, hy_) in ((LA, hd[0]), (LB, hd[1])):                         # dawn on the edge of the hair, all round on the side of the other head
                al = Lx[..., 3]; near = smooth(120, 84, np.hypot(self.x - hx_, self.y - hy_)); e = np.clip(al - cv2.warpAffine(al, np.float32([[1, 0, -3.0], [0, 1, 1.9]]), (W, H)), 0, 1) + .75 * np.clip(al - cv2.erode(al, np.ones((4, 4))), 0, 1); dk = smooth(.34, .14, Lx[..., :3].mean(-1)); e = np.clip(cv2.GaussianBlur(e, (0, 0), .6), 0, 1) * near * (.25 + .75 * dk) * al; Lx = Lx.copy(); Lx[..., :3] = Lx[..., :3] + (rgb('#ffe2b4') * al[..., None] - Lx[..., :3]) * (e * .85)[..., None]; L = Lx + L * (1 - Lx[..., 3:4])
            an = [self.cam.project(F.skel().an[s])[0] for F in (A, B) for s in 'LR']; st = [np.asarray(F.info['stomach'], float).reshape(-1)[:2] if F.info.get('stomach') is not None else self.cam.project(F.skel().waist)[0] for F in (A, B)]
            if len(self.fc) > 3: self.fc.pop(next(iter(self.fc)))
            self.fc[g] = (L, (an[:2], an[2:]), st, hd, self.overlap(L[..., 3], hd))
        return self.fc[g]

    def overlap(self, al, heads):
        """whole blades that stand outside the figure and lean over its edge"""
        P_, wd, col = self.L['blades']; dt = cv2.distanceTransform((al > .5).astype(np.uint8), cv2.DIST_L2, 3); g = lambda q, A: A[np.clip(q[:, 1].astype(int), 0, H - 1), np.clip(q[:, 0].astype(int), 0, W - 1)]
        k = (g(P_[:, 0], al) < .2) & (g(P_[:, -1], dt) > 1.) & (g(P_[:, -1], dt) < 15) & (np.modf(P_[:, 0, 0] * 7.13)[0] < .62)
        for hx_, hy_ in heads: k &= np.hypot(P_[:, -1, 0] - hx_, P_[:, -1, 1] - hy_) > 62
        c = land.Canvas(H, W, layer=True); c.strokes(P_[k], wd[k] * 1.05, col[k] * 1.02, shape='taper', seed=3, bristle=0.); return c.layer().astype(f32)

    def growl(self, L, p, u, amp, ink):
        """the stomach: it jumps, ripples, and dies away (on ones), with the ripple drawn on the muscle"""
        if not 0 <= u < 13: return L
        A_ = (.20, .62, -.34, .52, -.26, .42, -.20, .30, -.14, .20, -.08, .10, .0)[int(u)] * amp; R = 110; x0, y0 = int(p[0]) - R, int(p[1]) - R; x, y = np.meshgrid(np.arange(2 * R, dtype=f32), np.arange(2 * R, dtype=f32)); X, Y = x - R, y - R; d = np.sqrt(X * X + Y * Y); w = A_ * np.exp(-(d / 40.) ** 2); wv = .10 * amp * np.sin(d / 7. - u * 1.5) * np.exp(-(d / 52.) ** 2) * math.exp(-u / 6.); k = 1 - w - wv
        pc = cv2.remap(L, (R + X * k + x0).astype(f32), (R + Y * k + y0).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT); out = L.copy(); out[y0:y0 + 2 * R, x0:x0 + 2 * R] = pc
        c = Cel(); fade = math.exp(-u / 7.)
        for side in (0, math.pi):
            for j in range(3):
                r_ = 22 + 13 * j + (u % 2) * 5 + u * 1.2; th = -ROT + side + np.linspace(-.62 + .10 * j, .62 - .10 * j, 11); Q = np.c_[p[0] + np.cos(th) * r_, p[1] + np.sin(th) * r_ * .92]; c.line(Q, 3.4 - .6 * j, ink, taper=(.35, .35), press=.3, seed=int(u) * 7 + j, alpha=.85 * fade)
        mk = c.rgba() * out[..., 3:4]; return over_pm(out, mk)

    def frame(self, f):
        L, self.feet, st, self.heads, self.fr = self.actors(f); L = self.growl(L, st[0], f - G1, .95, '#7a4326'); L = self.growl(L, st[1], f - G2, 1.0, '#0c1a4c')
        img = self.water(f); land_ = self.L['land']; al = L[..., 3]; la = land_[..., 3]
        img = over(img, land_)
        # grass ring that runs out from a growl
        for f0, p in ((G1, st[0]), (G2, st[1])):
            t = f - f0
            if 0 < t < 9: q = np.abs(np.hypot(self.x - p[0], self.y - p[1]) - 30 - 52 * t) / 16.; img = img + (rgb('#f6ecb4') - img) * (np.clip(1 - q, 0, 1) * la * .55 * (1 - t / 9.))[..., None]
        # shadow on the grass and on the water: thrown down-left, darkest at the contact
        sh = np.zeros((H, W), f32)
        for k in range(1, 6): sh = np.maximum(sh, cv2.warpAffine(al, np.float32([[1, 0, -5.2 * k], [0, 1, 2.9 * k]]), (W, H)))
        con = cv2.GaussianBlur(al, (0, 0), 7); sh = cv2.GaussianBlur(sh, (0, 0), 1.0); img = img * (1 - (sh * .46 * la)[..., None] * (1 - rgb('#3a5a6a')) * 1.25); img = img * (1 - (con * .30 * la)[..., None]); img = img * (1 - (sh * .30 * (1 - la))[..., None] * (1 - rgb('#6a6a94')))
        img = over(img, L)
        # boots in the water: the lake lies over them
        wet = al * (1 - la) * np.clip(cv2.GaussianBlur(1 - la, (0, 0), 6) * 1.3 - .25, 0, 1); ref = rgb('#e9c3c4'); img = img + (ref * .5 + img * .55 - img) * (wet * .42)[..., None]
        wl = np.clip(al - cv2.erode(al, np.ones((3, 3))), 0, 1) * (1 - la); img = img + (rgb('#fff4ea') - img) * (cv2.GaussianBlur(wl, (0, 0), .7) * .55)[..., None]
        # blades over their edges
        img = over(img, self.fr)
        return self.ash(img, f).astype(f32)

    def ash(self, img, f):
        if f < 40: return img
        K = self.petal; P_ = np.array([(a, b) for _, a, b, _ in K]); ts = [k[0] for k in K]; from scipy.interpolate import PchipInterpolator as PI
        x, y, s = float(PI(ts, P_[:, 0])(f)), float(PI(ts, P_[:, 1])(f)), float(PI(ts, [k[3] for k in K])(f)); a = f * .30 if f < 70 else 70 * .30 + .02 * (f - 70); tip = math.cos(f * .42) if f < 70 else .92
        Q = spl(np.array([(-1, 0), (-.5, -.42), (.3, -.40), (1., -.06), (.45, .34), (-.4, .36)]), closed=True, corners=(0, 3), n=6) * [11 * s, 11 * s * (.35 + .65 * abs(tip))]; ca, sa = math.cos(a), math.sin(a); R = np.c_[x + Q[:, 0] * ca - Q[:, 1] * sa, y + Q[:, 0] * sa + Q[:, 1] * ca]; c = Cel()
        if f < 70: c.fill(R + [-(70 - f) * 2.2, (70 - f) * 1.2], '#5a5f80', .22 * (1 - (70 - f) / 34.))
        c.fill(R, '#b9b2bb' if tip > 0 else '#8f8896'); c.fill((R - [x, y]) * .55 + [x - 1, y - 1], '#ded6d6' if tip > 0 else '#a59da9'); c.line(R, 1.3, '#4b4458', taper=(.1, .1)); return over(img, c.rgba())
