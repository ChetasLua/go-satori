"""e4 I NEVER NOTICED (80).  0 he tips, Goku catches him and goes down on his knees with the weight (contact 14, knees 20); 30 the stone ball in the puddle; 45 hidden eyes and one drop that is not rain; 65 a hand closes in the puddle, the water is squeezed out on a heartbeat (70)."""
import numpy as np, cv2, math
from db import W, H, comp, kit, land, storm, ball, phys_water as PW
from db import land_sets as LS, hero3, hands as HD
from db.hero3 import Fig, POSES, P, mix_pose, compose, carry
from db.cam import Camera3D
from db.paint import rgb, f32, smooth, fbm, grid, ramp
from db.ink import Cel, spl
from db.comp import over, blur, radial, ease
from db.land_grow import rain_rings
from chars import goku_heroes as GX, goku_hands as GK, goku_light as GL
A = np.array
CATCH, KNEES, CL = 14, 20, LS.CLIFF


def over_p(a, b): return b + a * (1 - b[..., 3:4])


class Shot:
    id = 'e4'; n = 80
    finish = dict(bloom=.18, vig=.24, soft=.12, sat=.92)
    EVENTS = [(0, 'rain_bed', dict(to=79, size=.5)), (1, 'steps_wet', dict(to=12, size=.5, note='four running steps in water, 1 4 7 10')), (2, 'cloth', dict(size=.2, note='he tips')), (CATCH, 'catch', dict(size=.7, note='a body in his arms, breath knocked out')), (16, 'breath_out', dict(size=.4)), (KNEES, 'knees_splash', dict(size=.9, note='both knees in the puddle, the weight')), (24, 'hand_splash', dict(size=.3, note='the glove falls in the water')),
              (30, 'cut', dict(note='stone ball in the puddle')), (30, 'drip_stone', dict(to=44, size=.3, note='rain on stone, hollow')), (33, 'drip', dict(size=.25)), (38, 'drip', dict(size=.3)), (42, 'drip', dict(size=.25)), (45, 'cut', dict(note='close: rain goes far away')), (47, 'teeth', dict(size=.2)), (52, 'breath_in', dict(size=.35, note='it catches twice')), (58, 'tear', dict(to=64, size=.2, note='one warm drop down the cheek')), (64, 'tear_fall', dict(size=.2)),
              (65, 'cut', dict(note='hand in the puddle')), (65, 'heart', dict(size=.3)), (70, 'heart', dict(size=.9, note='the fist closes on this beat')), (70, 'squeeze_water', dict(size=.7, note='water forced out between the fingers')), (74, 'fist_tremble', dict(to=79, size=.3)), (78, 'rain_stop_hint', dict(size=.1, note='the rain thins: f1'))]

    def setup(self):
        import os
        self.x, self.y = grid(); P_ = 'cache/e4_land2_%s.npy'
        if not all(os.path.exists(P_ % k) for k in ('back', 'cliff', 'puddle', 'pts')):
            d = LS.cliff_e1(cam=self.cam1(), turn=.36)                            # e1's own promontory, seen by this camera
            for k in ('back', 'cliff', 'puddle'): np.save(P_ % k, np.asarray(d[k], np.float16))
            np.save(P_ % 'pts', A(d['pts']))
        self.back, self.cliff, self.pud = (np.load(P_ % k, mmap_mode='r') for k in ('back', 'cliff', 'puddle')); self.pts = [tuple(q) for q in np.load(P_ % 'pts')]

    # ------------------------------------------------------------------ 0..29 the catch, on e1's cliff
    def cam1(self): return Camera3D((.10, -5.0, CL + .84), (-.12, 0, CL + .60), focal=1900.)

    def figs(self, f):
        d = (f // 2) * 2 if f < 12 or f > 25 else f; E = lambda a, b: float(ease(np.clip((d - a) / (b - a), 0, 1))); HG0, HG1 = -math.pi / 2, -2.72; HV0, HV1 = -.95, -math.pi / 2 + .12
        G, V = Fig('goku', 'torn'), Fig('vegeta', 'shattered')
        # --- Vegeta: an inverted pendulum, stiff, then a weight that folds
        th = math.radians(8.0) * math.cosh(min(d, CATCH) / 7.6); st = P('lookup', wst=(-4 * min(1, d / 10.), 0, 0), hed=(-14, 4 * min(1, d / 10), 0), gL='limp', gR='limp'); k = E(CATCH, 21); lol = math.exp(-max(0, d - 22) / 2.6) * math.cos(max(0, d - 22) * .95) if d >= 22 else 0.
        sag = P(lead='root', body=(-64 - 2 * lol, 0, 0), pel=(4, 0, 0), wst=(14 - 5 * lol, 2, 0), cst=(8, 4, 4), nck=(-34 - 9 * lol, 6, 4), hed=(-32 - 12 * lol, 8, 6), hL=(-.1, .5, -.5), hR=(.5, -.6, -.4), eL=(.6, -.4, -1), eR=(.8, -.3, -1), gL='limp', gR='limp', fL=(.085, .0, .225), fR=(.075, .04 + .03 * lol, .20), kL=(.25, 1, .6), kR=(.2, 1, .5), tL=(14, -34, -4), tR=(14, -34, -4), air=1)
        # --- Goku: runs in low, takes him under the arms, the weight puts him on his knees (20), then back on his heels (24)
        down, sit = float(np.clip((d - CATCH) / (KNEES - CATCH), 0, 1)) ** 1.5, E(KNEES, 24.5); dip = math.exp(-(d - KNEES) / 2.2) * math.cos((d - KNEES) * 1.0) if d >= KNEES else 0.; bow = E(19, 27)
        knees = P('knees', root=(0, -.03, -.262 - .022 * max(0, dip)), pel=(6 + 8 * dip, 0, 0), wst=(12 + 9 * dip, 0, 4), cst=(10 + 8 * dip, 2, 4), nck=(12 + 8 * dip, 0, 0), hed=(8 + 8 * max(0, dip), -2, 0), gL='open', gR='open')
        seiza = P('knees', root=(0, -.04, -.345 - .012 * dip), pel=(8, 0, 0), wst=(13 + 5 * dip, 0, 2), cst=(9 + 4 * dip, 0, 2), nck=(8 + 6 * bow, 0, 0), hed=(4 + 10 * bow + 4 * max(0, dip), -4, 0), fL=(.095, -.07, .045), fR=(.085, -.08, .045), kL=(.30, 1, -.2), kR=(.25, 1, -.2), gL='open', gR='open')
        arms = P('lunge', lead='root', root=(0, .06, -.10), pel=(22, 0, 0), wst=(6, 0, 0), cst=(0, 0, 0), nck=(-6, 0, 0), hed=(0, 0, 0), hL=(-.05, .78, -.30), hR=(.02, .78, -.22), eL=(1, -.2, -.6), eR=(1, -.2, -.6), gL='open', gR='open')
        run = hero3.sprint(.30 + min(d, 10) * .142); gp = mix_pose(run, 'lunge', float(np.clip((d - 7) / 3., 0, 1)) * .6) if d < 10 else mix_pose(mix_pose(run, arms, float(np.clip((d - 9) / 4., 0, 1))), mix_pose(knees, seiza, sit), down)
        ux = float(np.clip(d / CATCH, 0, 1)); gx = -3.1 + (3.1 - .92) * (1 - (1 - ux) ** 1.7); gy = .10; u2 = E(CATCH, 22); gx, gy = gx + (-.45 + .92) * u2, gy + .10 * u2; hd = HG0 + (HG1 - HG0) * E(12, 22)
        G.pose(gp).place(gx, gy, CL, heading=hd); s_ = G.skel(); lap = (s_.hip['L'] + s_.hip['R'] + s_.kn['L'] + s_.kn['R']) / 4 + A([.02, -.02, .17])
        back = A([-1., -.12]); back /= np.hypot(*back); Lp = .86; pf = A([.06 + back[0] * Lp * math.sin(th), .02 + back[1] * Lp * math.sin(th), CL + Lp * math.cos(th)])
        held = A([gx + .30, gy - .22, CL + .66 - .10 * down]); u = E(CATCH - 1, 19); pos = pf * (1 - u) + (held * (1 - sit) + lap * sit) * u + A([0, 0, -.030 * max(0, lol) * sit - .02 * max(0, dip)])
        V.pose(mix_pose(st, sag, k)).place(*pos, heading=HV0 + (HV1 - HV0) * E(13, 20), pitch=-th * (1 - k), pivot='pelvis'); v = V.skel(); wr, wl = E(17, 24), E(16, 22)
        if wr > 0: V._reach = dict(R=dict(p=A([v.sh['R'][0] + .10, v.sh['R'][1] - .10, CL + .035]), part='wrist', w=wr), L=dict(p=v.pel + A([-.02, -.06, .13]), part='palm', palm=A([0, 0, -1.]), w=wl)); v = V.skel()
        if d >= CATCH - 2:
            w = float(np.clip((d - CATCH + 2) / 3., 0, 1)); m = E(15, 21); nk = (v.neck * (.4 + .6 * m) + v.chest * (.6 - .6 * m)) + A([-.02, -.10 - .03 * m, -.07]); kn = (v.kn['L'] + v.kn['R']) / 2 + A([-.05, -.12, -.08]); hip = v.pel + A([-.05, -.12, -.06])
            G._reach = dict(R=dict(p=nk, part='palm', palm=A([0, 0, 1.]), fingers=A([0, -1., .2]), g='open', body=.06, w=w), L=dict(p=hip * (1 - m) + kn * m, part='palm', palm=A([0, 0, 1.]), fingers=A([0, -1., .2]), g='open', body=.05, w=w))
        if d >= 14:                                                # his face goes down to him
            h_ = G.skel(); wl_ = .72 * E(14, 18) + .10 * bow; fw = A([-math.sin(hd), math.cos(hd), 0.]); dv = v.head + A([.04, -.03, .02]) - h_.head; dv = dv / np.linalg.norm(dv); t_ = fw * (1 - wl_) + dv * wl_; G.look(h_.head + t_ / np.linalg.norm(t_))
        G.blow(-.8 * (d < 14), 0, .8 * max(0, dip) if d >= KNEES else (-.9 if CATCH < d < KNEES else 0.)); V.blow(0, 0, 0); return G, V, d, v

    def part1(self, f):
        cam = self.cam1(); G, V, d, v = self.figs(f); k = dict(light=(-.25, .9, .2), pal='rain', rim=dict(dir=(.35, .85, -.30), col='#e4eaec', thr=.46, amt=.75)); ex = 'shock' if d < 17 else ('shock', 'griefhid', float(np.clip((d - 17) / 5., 0, 1)))
        L = compose([(G, dict(k, expr=dict(expr=ex, wet=.95, dmg=.9))), (V, dict(k, expr=dict(expr='peace', wet=1., scratch=1, hair_drop=.55, wind=.05)))], cam, d)
        z = 1 + .40 * float(ease(np.clip((f - 2) / 27., 0, 1))); sx, sy = kit.shake(f, 8., 7, 2.4, KNEES) if f >= KNEES else (0, 0); px, py = 800., 720.
        K = lambda lay, q: cv2.warpAffine(lay, np.float32([[1 + (z - 1) * q, 0, -(z - 1) * q * px + sx * q], [0, 1 + (z - 1) * q, -(z - 1) * q * py + sy * q]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        img = K(self.back.astype(f32), .22); img = over(img, storm.rain(f, 80, 1, .10, 46., (110, 240), 1.0, '#c9ced0', .34)); cl = self.cliff.astype(f32); pm = self.pud.astype(f32)
        pj = lambda q: cam.project(A([q], float))[0][0]; gs = G.skel(); low = [(gs.kn[s] if d >= 17 else gs.an[s]) for s in 'LR'] + [gs.an[s] for s in 'LR'] + ([v.an[s] for s in 'LR'] if d < 17 else [v.wr['R']])
        c = Cel(W, H, ss=1)
        for q in low:
            if q[2] < CL + .22: p_ = pj((q[0], q[1], CL)); a = np.linspace(0, 6.28, 26); c.fill(np.c_[p_[0] + np.cos(a) * 92, p_[1] + 4 + np.sin(a) * 13], '#0c0e10', .42)
        if d >= 18: p_ = pj((-.35, -.02, CL)); a = np.linspace(0, 6.28, 40); c.fill(np.c_[p_[0] + np.cos(a) * 260, p_[1] + 2 + np.sin(a) * 24], '#0c0e10', .38 * min(1, (d - 17) / 4.))
        sh = c.rgba(); sh = np.dstack([blur(sh[..., j], 5) for j in range(4)]); sh *= cl[..., 3:4]; cl = sh + cl * (1 - sh[..., 3:4])
        base = max(pj((q[0], q[1], CL))[1] for q in low); rf = cv2.warpAffine(L, np.float32([[1, 0, 0], [0, -1, 2 * base - 4]]), (W, H)); rf = cv2.remap(rf, (self.x + 4 * np.sin(self.y * .16 + f * .9) + 2.4 * np.sin(self.y * .41 - f * 1.3)).astype(f32), self.y.astype(f32), cv2.INTER_LINEAR); m = (rf[..., 3] * pm)[..., None]; cl[..., :3] = cl[..., :3] * (1 - m * .70) + rf[..., :3] * pm[..., None] * .55
        c = Cel(W, H); r = np.random.default_rng(31)                           # rain on the puddle and on the rock
        for i in range(34):
            ph = (f + 14 * r.random()) % 14; cyc = int((f + 14 * r.random()) // 14); rx, ry = r.uniform(-.9, 1.3), r.uniform(-.75, .25); p_ = pj((rx, ry, CL)); ix, iy = int(np.clip(p_[0], 0, W - 1)), int(np.clip(p_[1], 0, H - 1))
            if pm[iy, ix] > .4: PW.ripple(c, cam, (rx, ry), ph / 24. + .03, .62, 2, .10, .016, .6, 'grey', CL, i)
        rp = c.rgba() * pm[..., None]; cl = rp + cl * (1 - rp[..., 3:4]); c = Cel(W, H); storm.splashes(c, f, self.pts, 5); spl_ = c.rgba(); img = over(img, K(over_p(cl, spl_), 1.)); lay = L
        c = Cel(W, H)                                                          # water answers: running steps, knees, the hand
        for q, (x_, y_) in ((1, (-2.7, .05)), (4, (-2.1, .14)), (7, (-1.6, .04)), (10, (-1.2, .14))):
            if 0 < f - q < 12: PW.splash(c, cam, (f - q) / 24., (x_, y_), v0=1.1, r0=.08, n=8, rad=.013, pal='grey', seed=q, rings=0, dirn=(.5, 0), z=CL)
        if f > KNEES:
            for j, s in enumerate('LR'): PW.splash(c, cam, (f - KNEES) / 24., (float(gs.kn[s][0]), float(gs.kn[s][1])), v0=1.55, r0=.13, n=11, rad=.015, em=.10, pal='grey', seed=30 + j, rings=2 if j else 0, lean=.95, sheet=.30, z=CL)
        if f > 24: PW.splash(c, cam, (f - 24) / 24., (float(v.wr['R'][0]), float(v.wr['R'][1])), v0=1.0, r0=.06, n=8, rad=.012, pal='grey', seed=44, rings=2, z=CL)
        img = over(img, K(lay, 1.)); img = over(img, K(c.rgba(), 1.))
        img = over(img, storm.rain(f, 40, 2, .12, 88., (280, 600), 1.9, '#e6ebee', .46)); near = storm.rain(f, 8, 3, .13, 150., (560, 980), 4.2, '#f0f3f4', .40); return over(img, blur(near, 1.6))

    # ------------------------------------------------------------------ 30..44 the stone
    def part2(self, f):
        u = (f - 30) / 14.; e = float(ease(u)); cam = Camera3D((-.10 + .03 * e, -1.02 + .05 * e, .30), (-.02, .02, .055), focal=2050., cx=960, cy=560); rr = rain_rings(f, (-.8, .8, -.6, 1.2), 34, rmax=.13) + [(0., .03, .10 + .016 * ((f * .9 + 4 * k) % 16), .010, .5 * (1 - ((f * .9 + 4 * k) % 16) / 16.)) for k in range(3)]; o = land.puddle(cam=cam, t=f / 24., rings=rr); img = o.img.astype(f32).copy(); x, y = self.x, self.y
        cx, cy, r = 960., 486., 208. * (1 + .012 * e); wl = cy + r * .52                # water line
        bg = img.copy(); full = ball.draw(img.copy(), cx, cy, r, stars=(0., 0., 0., 0.), stone=1., wet=1., drip=f, shade=(.86, .88, .92), light=(-.3, .85, .4))
        d = np.hypot(x - cx, y - cy); a = smooth(r + 1.2, r - 1.2, d); rx = math.sqrt(max(0., r * r - (wl - cy) ** 2)); ell = ((x - cx) / rx) ** 2 + ((y - wl) / (rx * .17)) ** 2; wav = 2.0 * np.sin(x * .045 + f * .7) + 1.2 * np.sin(x * .11 - f * 1.1)
        above = smooth(1.5, -1.5, y - (wl + np.sqrt(np.clip(1 - ((x - cx) / rx) ** 2, 0, 1)) * rx * .17 + wav))[..., None]
        # drowned half: dark, soft, bent by the rings; mirrored half on the water
        drown = cv2.remap(full, (x + 5 * np.sin(y * .10 + f * .8)).astype(f32), (y - 4).astype(f32), cv2.INTER_LINEAR) * rgb('#4a5156') + rgb('#1a1d20') * .35; m_dr = (a * .62)[..., None] * (1 - above); img = img * (1 - m_dr) + drown * m_dr
        fl = cv2.warpAffine(full, np.float32([[1, 0, 0], [0, -1, 2 * wl + 60]]), (W, H), borderMode=cv2.BORDER_REPLICATE); am = cv2.warpAffine(a, np.float32([[1, 0, 0], [0, -1, 2 * wl + 60]]), (W, H)); wob = 9 * np.sin(y * .085 + f * .8) * np.clip((y - wl) / 160, 0, 1.4) + 4 * np.sin(y * .23 - f * 1.4); fl = cv2.remap(fl, (x + wob).astype(f32), y.astype(f32), cv2.INTER_LINEAR); am = cv2.remap(am, (x + wob).astype(f32), y.astype(f32), cv2.INTER_LINEAR); m2 = (am * smooth(wl + 8, wl + 40, y) * (1 - a * .4) * .50)[..., None]; img = img * (1 - m2) + (fl * .62) * m2
        img = img * (1 - a[..., None] * above) + full * a[..., None] * above
        c = Cel(W, H); t = np.linspace(-.02, math.pi + .02, 90); q = np.c_[cx + np.cos(t) * rx * 1.005, wl + np.sin(t) * rx * .17 + 2.0 * np.sin((cx + np.cos(t) * rx) * .045 + f * .7)]; c.line(q, 3.0, '#e6ecee', taper=(.2, .2), press=.6, seed=3, alpha=.85); c.line(q + [0, 6], 2.2, '#14171a', taper=(.2, .2), press=.5, seed=4, alpha=.8)
        for k in range(3):                                          # rings that leave the stone
            s_ = ((f * 1.0 + 5.3 * k) % 16) / 16.; R = rx * (1.05 + .75 * s_); tt = np.linspace(0, 6.283, 120); qq = np.c_[cx + np.cos(tt) * R, wl + np.sin(tt) * R * .17]; c.line(qq[8:56], 2.6 * (1 - s_) + .6, '#dfe6e8', taper=(.3, .3), press=.7, seed=k, alpha=.8 * (1 - s_)); c.line(qq[66:112] if s_ > .2 else qq[70:108], 2.0 * (1 - s_) + .4, '#15181b', taper=(.3, .3), press=.6, seed=k + 9, alpha=.6 * (1 - s_))
        r_ = np.random.default_rng(5)
        for k in range(26):                                         # rain strikes the stone: ticks and running beads
            t0 = int(r_.integers(0, 12)); ph = (f - t0) % 12; th = r_.uniform(-2.7, -.45); rr_ = r * math.sqrt(r_.uniform(.3, 1.)); px, py = cx + math.cos(th) * rr_, cy + math.sin(th) * rr_ * (.98)
            if ph < 3 and py < wl - 10:
                for j in range(4): an = -math.pi / 2 + (j - 1.5) * .6 + r_.normal(0, .15); l0, l1 = 4 + 7 * ph, 11 + 13 * ph; c.line(A([(px + math.cos(an) * l0, py + math.sin(an) * l0 * .8), (px + math.cos(an) * l1, py + math.sin(an) * l1 * .8 + ph * 2)]), 2.2 - .5 * ph, '#f2f6f7', taper=(.1, .6), seed=j, alpha=.85 - .2 * ph)
        img = over(img, c.rgba()); img = over(img, storm.rain(f, 40, 5, .10, 96., (260, 560), 1.8, '#e0e6e8', .42)); near = storm.rain(f, 6, 6, .12, 170., (620, 1000), 5., '#f0f3f4', .40); img = over(img, blur(near, 2.0))
        g = smooth(300, 900, np.hypot((x - cx), (y - cy) * 1.2)); b = blur(img, 5); return img + (b - img) * (g * .8)[..., None]

    # ------------------------------------------------------------------ 45..64 the face
    def part3(self, f):
        a = f - 45; d = (f // 2) * 2; r = np.random.default_rng(70 + d); tr = r.normal(0, 1, 3); hit = math.exp(-((f - 52.) / 2.2) ** 2) + .8 * math.exp(-((f - 57.) / 2.) ** 2); tear = float(np.clip((f - 49) / 14., 0, 1)); tear = .12 + .80 * tear ** 1.4 if f >= 48 else 0.
        img = GX.grief(cx=905 + .8 * tr[0], cy=424 + 5 * hit + .8 * tr[1], s=322 + a * .8, pitch=-19 - 1.6 * hit, roll=11 + .25 * tr[2], f=f, tear=tear, fg=True, wind=(.10, -.04), mouth=float(.30 + .25 * hit + .12 * abs(tr[0])), fd=d)
        return img

    # ------------------------------------------------------------------ 65..79 the fist
    def part4(self, f):
        WL = 640.; a = f - 65; x, y = self.x, self.y; cam = Camera3D((0, -1.05, .62), (0, .06, .0), focal=1650., cx=960, cy=560); beat = math.exp(-max(0, f - 70) / 4.) if f >= 70 else 0.
        ring = [(0., -.08, .16 + .060 * (f - 70), .016, .9 * math.exp(-(f - 70) / 7.)), (0., -.08, .10 + .050 * (f - 71), .012, .7 * math.exp(-(f - 71) / 6.))] if f >= 72 else []
        o = land.puddle(cam=cam, t=f / 24., rings=rain_rings(f, (-.7, .7, -.5, .9), 26, rmax=.12) + ring); img = o.img.astype(f32) * .92
        t = 0. if f < 67 else float(np.clip((f - 67) / 4., 0, 1)); t = t * t * (3 - 2 * t) if f < 71 else 1.; tremor = (1.2 if f >= 72 else .5 if f < 67 else 0.); sq = math.exp(-max(0, f - 70) / 1.6) * (f >= 70)
        L = dict(GL.light('rain', side=1)); L.update(skin=('#b3a69d', '#74696a', '#443f45', '#ece7e2'), ldir=(.72, .62, .10), under=('#33425c', '#1c2436', '#67788f'), rim=[dict(d=(-4, 6), col='#eef3f6', amt=.9)])
        hs = GK.puddle(t, R=HD.euler(-10 + 4 * t, 46 + 18 * t, 190 - 4 * t), tremor=tremor, seed=f // 2); cam2 = HD.Cam(965 + (f % 2) * (1.6 if f >= 72 else 0), 446 + 12 * t + 8 * sq, 400, 4.8); g, lay = GK.paint(hs, cam2, L, w=5.2, seed=5, res=760, hat=.8, thr=.05)
        # in the water: a water line across the wrist side and the finger tips; below it drowned and bent
        al = lay[..., 3]; wl = WL - 34 * t * 0 + 8 * np.sin(x * .012 + f * .25) + 3 * np.sin(x * .05 - f * .5) + .05 * (x - 960); under = smooth(-3, 3, y - wl)[..., None]
        rf = cv2.remap(lay, (x + 7 * np.sin(y * .09 + f * .9)).astype(f32), (y + 3 * np.sin(x * .05 + f)).astype(f32), cv2.INTER_LINEAR); rf[..., :3] *= rgb('#5a656c'); sh = blur(al, 22); img = img * (1 - .38 * sh[..., None])
        mirror = cv2.warpAffine(lay, np.float32([[1, 0, 0], [0, -.72, WL * 1.72 + 26]]), (W, H)); mirror = cv2.remap(mirror, (x + 10 * np.sin(y * .07 + f * .8)).astype(f32), y.astype(f32), cv2.INTER_LINEAR); img = img * (1 - mirror[..., 3:4] * .40) + mirror[..., :3] * .30
        img = over(img, rf * under * .72); img = over(img, lay * (1 - under))
        c = Cel(W, H); edge = np.abs(y - wl) < 2.5; xs = np.where((al > .5) & edge)[1]
        if len(xs):
            x0, x1 = xs.min(), xs.max(); q = np.linspace(x0 - 26, x1 + 26, 70); yy = WL + 8 * np.sin(q * .012 + f * .25) + 3 * np.sin(q * .05 - f * .5) + .05 * (q - 960); m = np.sin(np.linspace(0, math.pi, 70)); c.line(np.c_[q, yy + 1], 3.4, '#eef3f5', taper=(.3, .3), press=.8, seed=f // 3, alpha=.9); c.line(np.c_[q, yy + 8 + 3 * m], 2.4, '#14171a', taper=(.3, .3), press=.7, seed=f // 3 + 1, alpha=.8)
            for k in range(3): s_ = ((f + 4.7 * k) % 14) / 14.; w_ = (x1 - x0) * (.56 + .35 * s_) + 60 * s_; tt = np.linspace(0, 6.283, 100); qq = np.c_[(x0 + x1) / 2 + np.cos(tt) * w_, WL + 8 + .05 * ((x0 + x1) / 2 - 960) + np.sin(tt) * w_ * .26]; c.line(qq[4:48], 2.8 * (1 - s_) + .5, '#dfe7ea', taper=(.3, .3), press=.8, seed=k, alpha=.75 * (1 - s_))
        if f >= 70:                                              # squeezed out between the fingers
            P_ = g.pts[0] if isinstance(g.pts, (list, tuple)) else list(g.pts.values())[0]; gaps = []
            fs = ['index', 'middle', 'ring', 'little']
            for u, v in zip(fs[:-1], fs[1:]): p = (A(P_[(u, 'pip')], float) + A(P_[(v, 'pip')], float)) / 2; gaps.append((float(p[0]), min(float(p[1]), WL + 6), -math.pi / 2 + (p[0] - 965) / 190.))
            p = A(P_[('thumb', 'tip')], float); gaps.append((float(p[0]), min(float(p[1]), WL), -math.pi * .5 - .9 * np.sign(965 - p[0]))); p = A(P_[('little', 'pip')], float); gaps.append((float(p[0]) + 26 * np.sign(p[0] - 965), WL, -math.pi / 2 + 1.05 * np.sign(p[0] - 965)))
            PW.squeeze(c, (f - 70) / 24. + .02, gaps, pal='grey', em=.15, v0=980., seed=3, rad=9.)
        img = over(img, c.rgba()); thin = 1. if f < 74 else max(.25, 1 - (f - 74) / 7.); img = over(img, storm.rain(f, int(44 * thin), 7, .10, 100., (240, 520), 1.8, '#e0e6e8', .42)); near = storm.rain(f, int(6 * thin), 8, .12, 170., (620, 1000), 5., '#f0f3f4', .38); img = over(img, blur(near, 2.0))
        if f >= 70: sx, sy = kit.shake(f, 5., 9, 2., 70); img = cv2.warpAffine(img, np.float32([[1.008, 0, sx - 8], [0, 1.008, sy - 4]]), (W, H), borderMode=cv2.BORDER_REFLECT_101)
        z = 1 + .020 * float(ease(a / 14.)) + .012 * beat; img = cv2.warpAffine(img, np.float32([[z, 0, (1 - z) * 980], [0, z, (1 - z) * 560]]), (W, H), borderMode=cv2.BORDER_REFLECT_101); return img

    def frame(self, f):
        img = self.part1(f) if f < 30 else self.part2(f) if f < 45 else self.part3(f) if f < 65 else self.part4(f); return img.astype(f32)
