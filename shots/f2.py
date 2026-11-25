"""f2 THE BREAK (80).  The face builds, holds, breaks: five flashes (0, 20, 40, 50, 60), hair rises a stage at each, rain still going up, tears to steam, pupils gone, then teal; scream 60..;
last frames crash into the eye: its iris is the film's match circle (960, 486, r 208)."""
import numpy as np, cv2, math
from db import W, H, ki, comp, kit
from db.paint import rgb, f32, smooth, fbm, grid, ramp
from db.ink import Cel, spl
from db.comp import over, blur, radial
from chars import goku_break as GB, goku_heroes as GX
A = np.array
FL = (0, 20, 40, 50, 60); RISE = (0., .16, .36, .60, .82, 1.0); FLOOR = (.06, .16, .30, .40, .95); HF = (.0, .05, .30, .55, 1.); ACC = ('#ffd23a', '#ffb01e', '#fff0a0', '#ff8a1e', '#ffe24a'); GR = (1, 3, 5, 6, 7)


def stage(f): return sum(f >= q for q in FL) - 1


class Shot:
    id = 'f2'; n = 80
    finish = dict(bloom=.24, vig=.20, halation='#ffe2a0')
    EVENTS = [(q, 'flash', dict(size=s, note='gold burst %d, hair lifts' % (k + 1))) for k, (q, s) in enumerate(zip(FL, (.5, .6, .75, .85, 1.)))] + [(q, 'impact_frame', dict(frames=2)) for q in FL] + \
             [(2, 'rain_up', dict(to=53, size=.2, note='drops tick upward')), (4, 'teeth', dict(size=.3, note='jaw grinds')), (12, 'breath_in', dict(size=.3, note='through the teeth')), (26, 'growl', dict(to=39, size=.5)), (24, 'tear', dict(size=.2)), (41, 'steam', dict(to=79, size=.4, note='tears hiss away')),
              (42, 'ground_rumble', dict(to=79, size=.6)), (43, 'vein', dict(size=.2)), (52, 'breath_in', dict(size=.7, note='gasp, jaw opens')), (54, 'silence', dict(to=60, note='everything holds, light goes out of the frame')), (60, 'scream', dict(to=79, size=1., note='singer')), (60, 'aura_ignite', dict(size=1.)), (62, 'lightning', dict(size=.7)), (68, 'lightning', dict(size=.8)), (74, 'zoom', dict(to=79, size=.7, note='crash into the eye'))]

    def setup(self):
        n = fbm(H, W, (300, 120, 50, 20), (1, .6, .35, .18), 31); x, y = grid(); v = np.clip(.5 + .22 * n - .35 * (y / H - .4), 0, 1); self.grey = (ramp(v, [(0, '#1c1e21'), (.4, '#3a3d40'), (.75, '#62666a'), (1, '#9a9d9e')])).astype(np.float16); self.storm = ramp(v, [(0, '#050a09'), (.45, '#0c1a16'), (.75, '#1f3327'), (1, '#4a5a34')]).astype(np.float16); self.x, self.y = x, y
        r = np.random.default_rng(8); self.bd = np.c_[r.uniform(0, W, 120), r.uniform(0, H + 200, 120), r.uniform(1.2, 5.5, 120) ** 1.0, r.random(120)]

    # ------------------------------------------------------------------ state of the drawing
    def state(self, f):
        k = stage(f); a = f - FL[k]; d = (f // 2) * 2 if f < 60 else f; e = d - FL[k]                                    # drawings on twos, scream on ones
        sp = 1 + (.95 * math.exp(-e / 2.6) * math.cos(e * .72) if e < 12 else 0.); r0, r1 = RISE[k], RISE[k + 1]; rise = r0 + (r1 - r0) * 1. + (.20 if k < 4 else .10) * (sp - 1) * 1.1
        fl = FLOOR[k]; gold = fl + (1 - fl) * math.exp(-max(0, a - 1) / (4.2 if k < 4 else 9.)); hg = HF[k] + (1 - HF[k]) * math.exp(-max(0, a - 2) / (2.6 if k < 3 else 4.))
        hold = f >= 54 and f < 60
        if hold: gold *= .55 - .06 * (f - 54); hg = .22
        p = dict(f=d, rise=rise, gold=float(np.clip(gold, 0, 1)), hgold=float(np.clip(hg, 0, 1)), wet=float((.95, .55, .25, .1, 0)[k] * (1 if e > 3 else .6)), eyes='base', aura=gold, vein=float((0, .3, 1., 1., 1.)[k]))
        tr = lambda amp, sd: np.random.default_rng(sd * 131 + (f // 1)).normal(0, amp, 2)
        ez = lambda u: float(comp.ease(np.clip(u, 0, 1)))
        if k == 0:
            u = ez((a - 8) / 10.); p.update(expr='griefhid' if d < 12 else ('griefhid', 'grit', .55), hide=1. if d < 12 else .45, pitch=-19 + 3 * u, yaw=-2, roll=5 - 1 * u, s=272 + a * .3, cx=960, cy=474, raise_=.10 + .04 * u, byaw=6, shift=(0, -.32), wind=(.05, -.04 + .30 * math.exp(-e / 3.)), over=dict(clench=1., mU=[-.044, .010, .022, .012, -.038], mL=[-.062, -.060, -.054, -.054, -.056]) if d < 12 else {})
            p['cx'] += float(tr(.9, 1)[0]) * (a > 10); p['cy'] += float(tr(.9, 2)[0]) * (a > 10)
        elif k == 1:
            u = ez(a / 10.); p.update(expr=('grit', 'rage', ez((d - 22) / 12.)), hide=0., pitch=-15 + 9 * u, yaw=-4, roll=3 - 4 * u, s=286 + 8 * u + a * .3, cx=964, cy=486, raise_=.13, byaw=8, shift=(0, -.26 * (1 - .5 * u)), wind=(0, .10 + .5 * math.exp(-e / 4.)), tears=float(np.clip((a - 4) / 10., 0, 1)), over=dict(iris=.058 - .012 * u, well=.9, tendon=.4 + .3 * u))
            q = 1.2 + 1.2 * a / 20; p['cx'] += float(tr(q, 3)[0]); p['cy'] += float(tr(q, 4)[0]); p['roll'] += float(tr(.25, 5)[0])
        elif k == 2:
            u = ez(a / 8.); p.update(expr='rage', hide=0., pitch=-4, yaw=-7, roll=-2, s=304 + a * .6, cx=968, cy=494, raise_=.17, byaw=10, shift=(0, -.10), wind=(0, .22 + .6 * math.exp(-e / 4.)), tears=float(np.clip(1 - a / 8., .25, 1)), eyes='base', over=dict(blank=1., tendon=.9, vein=1.))
            p['cx'] += float(tr(2.6, 6)[0]); p['cy'] += float(tr(2.6, 7)[0]); p['roll'] += float(tr(.4, 8)[0])
        elif k == 3:
            u = ez(a / 3.); h_ = ez((f - 53) / 2.); p.update(expr='blank', hide=0., pitch=-5 - 5 * h_, yaw=-9, roll=-4 - 1 * h_, s=318 + 3 * u, cx=972, cy=502 + 8 * h_, raise_=.19 + .10 * h_, byaw=11, shift=(0, -.06), wind=(0, .30 + .5 * math.exp(-e / 3.) - .25 * h_), tears=.25 * (1 - u), over=dict(tears=0.))
            p['over'] = dict(p['over'], jaw=.10 * u * (1 - .55 * h_)); p['jaw'] = .10 * u * (1 - .55 * h_)
            if not hold: p['cx'] += float(tr(3.2, 9)[0]); p['cy'] += float(tr(3.2, 10)[0])
            p['dark'] = .30 * h_
        else:
            u = float(np.clip((a - 0) / 5., 0, 1)); op = 1 - (1 - u) ** 2; r = np.random.default_rng(900 + f); j = r.normal(0, 1, 4) * (1. if f < 75 else .4); p.update(expr=('scream_a', 'scream', min(1., op * 1.0)), eyes='ssj', hide=0., pitch=2 + 6 * math.exp(-a / 4.) + .5 * j[0], yaw=-12 + .4 * j[1], roll=-8 + .7 * j[2], s=312 + 16 * op + a * 1.0, cx=984 + 3 * j[0], cy=516 + 3 * j[1], raise_=.16 + .02 * j[3], byaw=14, shift=(0, 0), wind=(0, .5 + .08 * math.sin(f * 2.1)), tears=0., over=dict(jx=.008 * j[2]))
            # the mouth lives: two surges with a catch of breath between, jaw trembling on ones, tongue and lips never the same twice
            K = {62: (.20, .262, 0.), 63: (.262, .285, .2), 64: (.318, .302, .5), 65: (.336, .312, .9), 66: (.306, .300, .3), 67: (.322, .296, 1.2), 68: (.272, .276, 1.8), 69: (.232, .258, 2.2), 70: (.296, .290, .6), 71: (.352, .318, .1), 72: (.326, .306, 1.), 73: (.344, .314, 1.6), 74: (.318, .304, .4)}; jw, hw, ph = K.get(min(f, 74), K[62]); s0 = (hw - .258) / .06
            p['over'].update(hw=hw + .003 * j[0], my=-.612 + .012 * s0 + .003 * j[1], mrot=.016 * math.sin(f * 2.3), asym=.5 * math.sin(f * 1.7), tongue=.72 + .5 * abs(math.sin(f * 1.15)), tph=ph, gum=.45 + .55 * s0, tU=.070 + .020 * s0, tL=.046 + .020 * abs(math.sin(f * 1.9)), mU=[-.024 - .020 * s0, .042 + .022 * s0, .054 + .022 * s0, .044 + .022 * s0, -.020 - .016 * s0], mL=[-.150 - .105 * s0, -.270 - .160 * s0, -.300 - .165 * s0, -.270 - .155 * s0, -.145 - .100 * s0])
            p['jaw'] = jw + .010 * j[3]; p['rise'] = 1.0 + .10 * math.exp(-a / 3.) * math.cos(a * .9) + .012 * math.sin(f * 2.7)
        return p, k, a, hold

    def zoomed(self, f, p):
        """74..79: the camera falls into the eye"""
        if f < 74: return p
        u = (f - 73) / 6.; u = u * u * (3 - 2 * u) if u < 1 else 1.; s0 = p['s']; s = s0 * (2735. / s0) ** u; G = GB.GHead().pose(p['cx'], p['cy'], s0, p['yaw'] * GB.D, p['pitch'] * GB.D, p['roll'] * GB.D); X = GB.GF.getx(p['expr']); e0 = G.Pc([((X['xi'] + (X['xo'] - X['xi']) * .43), -.012 + X['gy'] - .010 * u)])[0]
        X2 = dict(p['over'], B=[(b[0], b[1] + (.075 - .03 * i) * u) for i, b in enumerate(X['B'])], U=[v + d * u for v, d in zip(X['U'], (.0, .024, .040, .020, .0))], gy=X['gy'] - .010 * u); off = (e0 - A([p['cx'], p['cy']])) / s0; tgt = e0 * (1 - u ** .6) + A([960., 486.]) * u ** .6; q = dict(p, over=X2, s=s, cx=float(tgt[0] - off[0] * s), cy=float(tgt[1] - off[1] * s)); return q

    def background(self, f, p, k, a, hold):
        g = p['gold']; img = (self.grey.astype(f32) * (1 - .0) * (1 - min(1, k * .22)) + self.storm.astype(f32) * min(1, k * .22)) if k < 5 else self.storm.astype(f32)
        d = np.sqrt(((self.x - p['cx']) / 1150) ** 2 + ((self.y - p['cy'] - 120) / 760) ** 2); img = img * (1 - .25 * g) + (np.clip(1 - d, 0, 1) ** 1.3)[..., None] * rgb('#f09a1e') * .60 * g
        if hold: img = img * (.55 - .03 * (f - 54))
        if g > .25:
            c = Cel(W, H, ss=1); r = np.random.default_rng(60 + f // 2); cx, cy = p['cx'], p['cy'] + 100
            for _ in range(int(90 * g)): t = r.uniform(0, 6.283); q0 = r.uniform(420, 760); q1 = q0 + r.uniform(200, 900); ww = r.uniform(1, 9) * r.random(); o = A([math.cos(t), math.sin(t) * .8]); nr = A([-o[1], o[0]]); c.fill(np.vstack([A([cx, cy]) + o * q0, A([cx, cy]) + o * (q0 + (q1 - q0) * .7) + nr * ww, A([cx, cy]) + o * q1, A([cx, cy]) + o * (q0 + (q1 - q0) * .7) - nr * ww]), '#ffd76a', r.uniform(.15, .6) * g)
            lay = c.rgba(); img = over(img, lay) + GX.glow_of(lay, 6, '#ff9a2a', .25)
        return img.astype(f32)

    def beads(self, f, g, k, a, front):
        """rain that goes up; a little faster after every flash, still in the hold"""
        t = f if f < 54 else 54 + .02 * (f - 54); tot = 0.
        for j, q in enumerate(FL[:4]): tot += max(0., t - q) * (2.2 + 1.6 * j)
        c = Cel(W, H, ss=2)
        for i, (x, y, r, u) in enumerate(self.bd):
            if (i % 3 == 0) != front: continue
            rr = r * (2.4 if front else 1.); yy = (y - tot * (.5 + u) * (1.8 if front else 1.)) % (H + 200) - 100; xx = x + 6 * math.sin(f * .12 + i); st = 1 + .5 * u * (a < 6) + .25
            q = np.linspace(0, 6.283, 17)[:-1]; P = np.c_[xx + np.cos(q) * rr, yy + np.sin(q) * rr * st * 1.25]; c.fill(P, GB.lp('#3a4046', '#a2560e', g), .85); c.fill(np.c_[xx + np.cos(q) * rr * .78, yy + rr * .10 + np.sin(q) * rr * st * .95], GB.lp('#9aa4ac', '#ffc85a', g), .9); c.fill(np.c_[xx - rr * .25 + np.cos(q) * rr * .28, yy - rr * st * .45 + np.sin(q) * rr * .34], '#ffffff', .95)
        lay = c.rgba(); return np.dstack([blur(lay[..., j], 2.2) for j in range(4)]) if front else lay

    @staticmethod
    def bolts(bg, L, li):
        if li is not None: bg = bg + li * .7
        return bg if L is None else over(bg, L)

    def frame(self, f):
        p, k, a, hold = self.state(f); p = self.zoomed(f, p); zoom = p['s'] > 900; bg = self.background(f, p, k, a, hold)
        if not zoom and f < 60: bg = over(bg, self.beads(f, p['gold'], k, a, False))
        if k == 4 and p['s'] < 700: bg = GX.rage_aura(bg, p['cx'], p['cy'] + 150, f, .75 + .5 * min(1, a / 6.))
        elif a >= 2 and a < 10 and k >= 1 and not hold: bg = GX.rage_aura(bg, p['cx'], p['cy'] + 370 + 15. * a * a, f, max(.25, .95 - .09 * a))          # the flame sinks back under the frame
        if k in (1, 2, 3) and 2 <= a < 10 and not hold:
            L, li = ki.storm([(FL[k] + 2, (200 + 300 * k, -30), (420 + 260 * k, 520), 6, 40 + k, dict(forks=4)), (FL[k] + 4, (1700 - 120 * k, -30), (1480, 480), 5, 60 + k, dict(forks=3))], f, 'gold'); bg = self.bolts(bg, L, li)
        if k == 4 and not zoom:
            L, li = ki.storm([(62, (300, -30), (420, 560), 8, 41, dict(forks=5)), (66, (1650, -30), (1500, 500), 7, 43, dict(forks=4)), (69, (700, -30), (560, 420), 6, 47, dict(forks=4)), (72, (1250, -30), (1420, 620), 8, 49, dict(forks=5))], f, 'gold'); bg = self.bolts(bg, L, li)
        if zoom: bg = ramp(np.clip(self.y / H + .12 * np.abs(self.x - 960) / 960, 0, 1), [(0, '#8a2c06'), (.5, '#d8660e'), (1, '#ffb52c')]).astype(f32)
        img, lay, G, X, hr = GB.picture(p, bg)
        if not zoom:
            c = Cel(W, H, ss=1); r = np.random.default_rng(90 + f // 2); ns = (0, 0, 5, 3, 8)[k]
            for q in (-1, 1):                                    # tears leave as steam
                for j in range(ns):
                    z = G.Pc([(q * r.uniform(.22, .50), -.14 - .05 * r.random())])[0]; gr = .55 + .45 * min(1, a / 6.); tip = z + A([q * r.uniform(20, 120), -r.uniform(120, 330)]) * gr; m = (z + tip) / 2 + A([q * r.normal(20, 24), 0]); P = spl([z, z * .6 + m * .4 + A([r.normal(0, 14), 0]), m, tip * .7 + m * .3 + A([r.normal(0, 22), 0]), tip], n=10); c.line(P, r.uniform(3, 11), '#fffaf0', taper=(.10, .8), press=.6, seed=int(r.integers(1 << 30)), alpha=r.uniform(.10, .34))
            q_ = c.rgba(); img = over(img, np.dstack([blur(q_[..., j], 2.2) for j in range(4)])) if ns else img
            if f < 60: img = over(img, self.beads(f, p['gold'], k, a, True))
            if p['gold'] > .4 and not hold:
                c = Cel(W, H); r = np.random.default_rng(500)
                for j in range(int(80 * p['gold'])):              # sparks rise, pure function of f
                    x0, v, ph = r.uniform(0, W), r.uniform(8, 30), r.uniform(0, H + 100); yy = (ph - v * f) % (H + 100) - 50; d = A([r.normal(0, 3), -v * 1.1]); z = A([x0 + 14 * math.sin(f * .3 + j), yy]); c.line(A([z, z + d * .5 + A([r.normal(0, 2), 0]), z + d]), r.uniform(1.2, 4.2), '#fff6c8', taper=(.3, .5), seed=j)
                q_ = c.rgba(); img = over(img + GX.glow_of(q_, 5, '#ffb030', .8), q_)
        if zoom or f >= 74:
            u = (f - 73) / 6.; img = kit.zoomblur(img, 960, 486, .10 * math.sin(math.pi * min(1, u)) + .0, 19) if f < 79 else img
        # flashes: two inverted drawings, then the burn dies
        if a == 0: img = ki.impact(img, GR[k], p['cx'], p['cy'], subject=lay, seed=k, accent=ACC[k], dirn=.4 * k)
        elif a == 1:
            neg = 1 - np.clip(img, 0, 1); neg = neg * rgb('#ffe9b0') if k % 2 else neg[..., ::-1] * rgb('#fff2d0'); e = np.clip(cv2.dilate(lay[..., 3], np.ones((3, 3), np.uint8), iterations=5) - lay[..., 3], 0, 1)[..., None]; img = neg * (1 - e) + e * rgb(ACC[k])
        elif a < 7 and not zoom:
            w = math.exp(-(a - 2) / (1.5 if k < 4 else 2.2)) * (.5, .55, .65, .7, .9)[k]; img = img + w * (rgb('#ffd45a') * .22 + (radial(p['cx'], p['cy'] + 60, 900, 700, 1.4)[..., None] * rgb('#ffe9a8') * .40))
        if k == 4 and not zoom: sx, sy = kit.shake(f, 13., 5, 9., 60); img = cv2.warpAffine(img, np.float32([[1.02, 0, sx - 19], [0, 1.02, sy - 11]]), (W, H), borderMode=cv2.BORDER_REFLECT_101)
        elif a < 8 and a >= 2 and not hold: sx, sy = kit.shake(f, 7. + 2 * k, 3 + k, 2.6, FL[k]); img = cv2.warpAffine(img, np.float32([[1, 0, sx], [0, 1, sy]]), (W, H), borderMode=cv2.BORDER_REFLECT_101)
        return img.astype(f32)
