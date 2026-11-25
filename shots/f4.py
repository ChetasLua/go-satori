"""f4 ONE STEP (80).  Long lens on wet rock.  He walks through a wall of violet beams, one step a beat (0, 10, .. 70); beams slide round him or burst on the flame.
Frieza's white foot, out of focus in front of us, takes one step back (50-58).  76 he is not there any more; 77 the rain comes back."""
import os
from db.ki_stage import *
from db import land, ki, kit
from db import phys_dust as PD
TOP = land.TOP; HD = math.pi + .105; EYE = np.array([1.05, -36.3, TOP + .17]); FOC, CY = 2900., 818.
G = lambda x, y: TOP - .01                                              # at this grazing angle the rock is one plane: no steps under his feet


def stomp(f):
    """his walk, one step a beat: heel on the beat, the whole weight drops into it (2-3 frames), settles, the other knee drives up and through; shoulders and arms counter-swing.  R lands on even beats"""
    k = math.floor(f / 10.); x = f / 10. - k; s = x ** 1.28; c = math.cos(math.pi * s); sn = math.sin(math.pi * s); drop = math.exp(-((s - .16) / .15) ** 2); lift = math.sin(math.pi * min(1., max(0., (s - .06) / .94))) ** 1.2
    fw = .25 - .51 * s; sf = -.26 + .51 * (s * s * (3 - 2 * s)); zsw = .05 + .20 * lift * (1 - .25 * s); heel = max(0., (s - .66) / .34)
    p = P(root=(.030 * sn, 0, -.034 - .062 * drop + .046 * sn ** 2), pel=(6 + 5 * drop, 6.5 * sn * (1 if s > .05 else 0), -17 * c), wst=(2, -2.5 * sn, 8 * c), cst=(-3 + 8 * drop, -5 * sn, 24 * c), nck=(7 - 3 * drop, 2 * sn, -9 * c), hed=(1 - 4 * drop, 0, -6 * c), cL=(3 + 3 * drop, 2), cR=(3 + 3 * drop, 2),
          hR=(.36, .10 - .44 * c, -.80 - .05 * c * c), hL=(.36, .10 + .44 * c, -.80 - .05 * c * c), eL=(.9, -.8, 0), eR=(.9, -.8, 0), gL='fist', gR='fist', wL=(40, -8, 0), wR=(40, -8, 0),
          fR=(.062, fw, .045 + .014 * heel + .030 * max(0, 1 - s / .10)), fL=(.070, sf, zsw if s < .97 else .055), kR=(.16, 1, 0), kL=(.26, 1, .25), tR=(10, 22 * max(0, 1 - s / .12) - 34 * heel, 30 * heel), tL=(10, -52 * (1 - s) + 18 * max(0, (s - .7) / .3), 40 * max(0, 1 - s * 3)))
    from db.hero3 import mirror as M_
    return p if k % 2 == 0 else M_(p)


class Shot:
    id = 'f4'; n = 80
    finish = dict(bloom=.26, vig=.20, bloom_thr=.82, halation='#ffe2b0')
    EVENTS = [(k, 'giant_step', dict(size=1., note='plant %d: rock cracks, dust ring, pebbles' % (k // 10 + 1))) for k in range(0, 71, 10)] + [(0, 'beam_wall', dict(to=76, note='a wall of thin violet beams, the pitch rises with him')), (50, 'frieza_step_back', dict(size=.35, note='one soft bare foot, 50 lift 57 down')), (76, 'vanish', dict(size=.8)), (76, 'beam_wall_end', dict(note='the beams hit nothing')), (77, 'rain', dict(to=79, size=.5, note='the rain comes back')), (77, 'silence', dict(to=79))] + [(int(k), 'beam_splash', dict(size=.25)) for k in range(4, 76, 4)]

    def setup(self):
        cv2.setNumThreads(1); self.cam = cam = Camera3D(EYE, EYE + [0, 20., 0], FOC, cy=CY); p = 'cache/f4_v6.npz'; self.hz = CY
        if os.path.exists(p): q = np.load(p); self.lit, self.dark = q['a'].astype(f32), q['b'].astype(f32)
        else:
            lit = land.ground(cam, .5, 'storm', fires=[(.2, -21.3, TOP + 1., 11, '#ffd56a', 1.35)]).img; dark = land.ground(cam, .5, 'storm').img; lit, dark = [comp.bokeh(np.dstack([q, np.ones((H, W), f32)]), 3.4)[..., :3] for q in (lit, dark)]; np.savez(p, a=lit.astype(np.float16), b=dark.astype(np.float16)); self.lit, self.dark = lit, dark
        m = (np.clip((290 - np.arange(W)) / 60., 0, 1)[None] * np.clip((np.arange(H) - 836) / 24., 0, 1)[:, None]).astype(f32)[..., None]     # a dark seam of the rock crosses the corner: paint it over with its neighbour
        for q in (self.lit, self.dark): q[:, :300] = q[:, :300] * (1 - m[:, :300]) + q[:, 301:601][:, ::-1] * m[:, :300]
        self.wet = wet(self.hz + 4, 5); self.F = Fig('goku', 'ssj', seed=2); self.root = rooted(stomp, lambda f: 'R' if (f // 10) % 2 == 0 else 'L', 80, self.F, (-.30, -23.3), HD)
        self.feet = []
        for k in range(0, 80, 10): sk = self.F.pose(stomp(k)).place(*self.root[k], G(*self.root[k]), heading=HD).skel(); self.feet.append(sk.an['R' if (k // 10) % 2 == 0 else 'L'].copy())
        self.Z0 = self.foot(0); self.Z1 = self.foot(79); self.beams = self.make()

    # ------------------------------------------------------------------ Frieza's legs
    def foot(self, f):
        """his two shins, huge, out of focus, black against the beams with a violet edge: cut at the knee by the top of the frame; R steps back 50-56, inside the frame"""
        S_, D_, gy, x0 = 2200., 3.0, 998., 1268.; cam = Camera3D((0, 0, (gy - CY) / S_), (0, 20., (gy - CY) / S_), S_ * D_, cy=CY); u = np.clip((f - 50) / 6., 0, 1); e = es(u); lift = math.sin(math.pi * u) ** .8 * .060; set_ = math.exp(-((f - 57) / 1.4) ** 2) * (f >= 56)
        Z = Fig('frieza', spec=dict(tail=False), seed=3); b = dict(POSES['contra']); b['fL'] = (.042, .01, .045); b['fR'] = (.042, .06 - .142 * e, .045 + lift); b['root'] = np.array(b['root'], float) + [.004 * e, -.05 * e, -.010 * math.sin(math.pi * u) - .008 * e - .006 * set_]; b['hL'] = (.30, .06, -.9); b['tR'] = (14, -26 * math.sin(math.pi * u), 8 * math.sin(math.pi * u)); b['tL'] = (20, 0, 0)
        Z.pose(b).place(0, D_, 0, heading=.62); d = (x0 - float(cam.project(Z.skel().an['L'][None])[0][0, 0])) / S_; Z.place(d, D_, 0, heading=.62); K = dict(base='storm', mul='#0c0a12', litmul='#100d18', shmul='#07060b', add='#040308', ink='#020203', tint=('#4a3a70', .25))
        L = Z.draw(cam, 0, light=(-.80, .45, .40), key=-.12, pal=K, rim=[dict(dir=(.93, .10, -.28), col='#b46bff', thr=.50, amt=1.), dict(dir=(-.90, .10, -.42), col='#8a46e6', thr=.80, amt=.85)], lines=.4, head=False); sk = Z.skel(); sh = np.zeros((H, W), f32); y, x = np.mgrid[0:H, 0:W].astype(f32)
        for q in 'LR': p = cam.project(sk.an[q][None])[0][0]; up_ = max(0., float(sk.an[q][2]) - .072); sh = np.maximum(sh, np.exp(-((x - p[0] + 66) / (215 + 900 * up_)) ** 2 - ((y - gy - 6) / 17.) ** 2) * (.82 - 4.5 * up_))
        sh = np.clip(sh, 0, 1); lay = np.dstack([L[..., :3] + C('#040308') * (sh * (1 - L[..., 3]))[..., None], L[..., 3] + sh * (1 - L[..., 3])]).astype(f32); return comp.blur(comp.bokeh(lay, 10., 0), 4.)

    # ------------------------------------------------------------------ the wall of beams
    def centre(self, f): g = min(f, 75); p, z = self.cam.project(np.array([[*self.root[g], G(*self.root[g]) + .92]])); return p[0], self.cam.f / z[0]

    def make(self):
        r = np.random.default_rng(12); out = []
        for i in range(520):
            t0 = -8 + 84 * (i + r.random()) / 520; (c, s) = self.centre(int(max(0, t0 + 3))); p0 = np.array([2060., r.uniform(20, 900)]); aim = c + r.normal(0, 1, 2) * [60, 210]; d = aim - p0; p1 = p0 + d * r.uniform(1.75, 2.2); kind = 'hit' if r.random() < .34 else 'bend'; w = 2.2 + 10 * r.random() ** 3; rx, ry = .42 * s, 1.12 * s
            if kind == 'hit':
                v = np.linspace(0, 1, 160); Q = p0 + (p1 - p0) * v[:, None]; e = ((Q[:, 0] - c[0]) / rx) ** 2 + ((Q[:, 1] - c[1]) / ry) ** 2; j = np.nonzero(e < 1.)[0]
                if len(j): out.append((t0, p0, Q[j[0]], w * 1.25, 4., 'hit', r.random() < .6)); continue
            out.append((t0, p0, p1, w, 9., 'bend', r.random() < .5))
        return out

    def frame(self, f):
        t = f / 24.; cam = self.cam; gone = f >= 76; c, s = self.centre(f); steps = [(k, 1. if k else .7) for k in range(0, 71, 10)]
        fl = 1 + .05 * math.sin(f * 2.1) + .04 * math.sin(f * 3.7); L = fl if not gone else (.42, .10, .05, .03)[f - 76]; base = self.dark * (1 - min(1, L)) + self.lit * min(1, L); base = base * (1 + .06 * (fl - 1) * 5) if not gone else base
        hit_now = sum(1 for b in self.beams if b[5] == 'hit' and 0 <= f - (b[0] + b[4]) < 2); vl = 0 if gone and f > 76 else min(1., hit_now / 3.); x, y = grid(); base = base + (np.exp(-((x - c[0] - 200) / 620) ** 2 - ((y - 796) / 150) ** 2) * (.10 + .16 * vl) * (not gone))[..., None] * C('#8a2cff')
        # far bursts where beams that missed find the rock behind him
        img = base; F_ = Field(4); n_ = 0
        for i, b in enumerate(self.beams):
            if b[5] != 'bend': continue
            a = f - (b[0] + 8.5)
            if 0 <= a < 6 and 40 < b[2][0] < 900 and i % 6 == 0: q = np.array([b[2][0] * .7 + 160, min(752., b[2][1] * .25 + 576)]); ki.petals(F_, q[0], q[1], 6 + 5 * a, 34 + 16 * a, 11, i, t, .7, 0, None, .4, .22 * (1 - a / 6.), 3., 1., boil=False); n_ += 1
        if n_: lay = F_.cel(ki.PAL['violet']['c']); img = ki.put(img, comp.blur(lay, 2.2)) + ki.bloom(lay, .3, (6, 22, 60), (.4, .3, .3), '#8a2cff', .8)
        rows = []
        for k, (p, (a, sz)) in enumerate(zip(self.feet, steps)):
            u = (f - a) / 24.
            if 0 <= u < .40 and not (gone and a < 60): rows += PD.dust_ring(max(u, .02), (p[0], p[1]), r0=.12, v0=3.6, minor=.085, n=15, m=3, seed=k * 7, life=.46, drag=3.0, floor=p[2] - .15, rise=.14)
        if gone: u = (f - 75) / 24.; q = self.root[75]; rows += PD.dust_ring(u, (q[0], q[1]), r0=.2, v0=5.2, minor=.13, n=13, m=2, seed=91, life=.5, drag=2.8, floor=G(*q) - .05, rise=.2)
        zc = cam.cam(np.array([*self.root[min(f, 75)], TOP]))[2]; rows = [np.array(q, float) for q in rows]; back = [q for q in rows if cam.cam(q[:3])[2] > zc]; fr = [q for q in rows if cam.cam(q[:3])[2] <= zc]; dk = 'gold' if not gone else 'storm'
        if back: img = comp.over(img, PD.lobes_layer(cam, back, dk, light=(.1, -.85, .4), ss=1.25))
        if not gone:
            pos = self.root[f]; self.F.pose(stomp(f)).place(*pos, G(*pos), heading=HD).blow(0, 0, .22); Fg = self.F
            Lg = safe(Fg, cam, f, pal='gold', light=(-.35, .75, .45), rim=[dict(dir=(.95, .12, -.2), col='#c784ff', thr=.46 - .10 * vl, amt=.55 + .45 * vl)], expr=dict(expr='determined')); fc = face(Fg); A = ki.aura(Lg, f, 'gold', 1.0 + .10 * math.exp(-(f % 10) / 2.), 4, height=.42, face=fc, wob=.34, front=.45); fy = max(q[1] for q in Fg.info['feet'].values()) if isinstance(Fg.info.get('feet'), dict) else c[1] + .95 * s
            # reflection on the wet rock
            rf = mirror(ki.stack(A.back, Lg), fy + 4, .50, 10., t, 7); k_ = self.wet * np.clip(1 - (y - fy) / 330., 0, 1) * .66; img = img * (1 - (rf[..., 3] * k_)[..., None]) + rf[..., :3] * k_[..., None]
        bendf = ki.deflect(c[0] + .04 * s, c[1] - .05 * s, .46 * s, 1.10 * s) if not gone else None; B0 = [b[:5] for b in self.beams if not b[6]]; B1 = [b[:5] for b in self.beams if b[6]]; stop = 76
        if f < stop + 2:
            sp = lambda Q: [q for q in Q if q[0] < 73 and (not gone or q[0] + q[4] > f - 1)]
            l, g = ki.volley(sp(B0), f, 'violet', 1., 2, bendf, .9, persp=(1.5, .34)); img = ki.put(img, l) + g
        if not gone:
            img = A.under(img); img = ki.put(img, Lg); img = A.over(img, Lg[..., 3]); l, g = ki.volley(sp(B1), f, 'violet', 1., 3, bendf, .9, persp=(1.5, .34)); g = g * (1 - .55 * Lg[..., 3:4]); img = ki.put(img, l) + g
            H_ = [(math.floor(b[0] + b[4]), b[2][0], b[2][1], 15 + 5 * b[3] * .5, math.atan2(*(b[1] - b[2])[::-1]) + .0) for b in self.beams if b[5] == 'hit']; l, g = ki.splashes(H_, f, 'violet', 5)
            if l is not None: img = ki.put(img, l) + g * (1 - .5 * Lg[..., 3:4])
        else:
            k = f - 75
            if f <= 77: v = ki.vanish(self.last(), {76: 1, 77: 3}[f], 0., 6, 2.4, 'gold', 20); img = ki.put(img, v) if v is not None else img
            q = self.root[75]; p, z = cam.project(np.array([[*q, G(*q) + .03]])); S_ = cam.f / z[0]; rr = S_ * (.5 + 1.6 * (1 - math.exp(-k / 2.))); img = ki.put(img, ki.ring(p[0, 0], p[0, 1], rr, 9 * math.exp(-k / 2.5) + 1, f, 'gold', .07, 0, 4, .25, max(0, 1 - k / 5.), spurs=16)); F2 = Field(4)
            if f < 79:
                for i in range(int(26 / k)): x0 = c[0] + (h1(i, 1) - .5) * .9 * s; y0 = c[1] + .9 * s - 2.1 * s * h1(i, 2) - 60 * k; l_ = s * (.10 + .34 * h1(i, 3)) / k; F2.rib(spine(x0, y0, l_, -math.pi / 2 + .3 * (h1(i, 4) - .5), 10, .08, .8, i), lwidths(10, l_, l_ * .09, LV, .35), 2.)
                img = ki.put(img, F2.cel(ki.PAL['gold']['c'], lw=(1.4, 0, 0, 0), ink=(ki.PAL['gold']['ink'], None, None, None)))
        Cw = Field(4); Cb = Field(4)
        for k, (p, (a, sz)) in enumerate(zip(self.feet, steps)):
            e = f - a
            if not 0 <= e < 9 or gone: continue
            q, z = cam.project(np.array([[p[0], p[1], p[2] - .14]])); S_ = cam.f / z[0]; g_ = 1 - math.exp(-(e + .6) / 1.6); fall = max(0., 1 - e / 9.)
            for j in range(22):
                th = TAU * (j + .7 * h1(k, j, 41)) / 22; z_ = bms(h1(k, j, 42), .22, .34); l_ = S_ * (.05 + .30 * z_) * g_ * fall ** .4; x0 = q[0, 0] + math.cos(th) * S_ * (.16 + .42 * g_); y0 = q[0, 1] + math.sin(th) * S_ * .012; an = -math.pi / 2 + .62 * math.cos(th); D_ = Cw if math.sin(th) > 0 else Cb; D_.rib(spine(x0, y0 - S_ * .05 * e * fall * .3, l_, an, 9, .05, .6, j, .5 * math.cos(th)), lwidths(9, l_, l_ * .085 * fall + .4, LV, .3), 2.)
            img = ki.put(img, ki.ring(q[0, 0], q[0, 1], S_ * (.25 + 1.5 * g_), 5. * fall + .5, f, 'gold', .045, 0, 30 + k, .35, .85 * fall, spurs=0))
        W_C = ('#4a3a1c', '#c08a34', '#ffe08a', '#fffbe6'); img = ki.put(img, Cw.cel(W_C, lw=(1.1, 0, 0, 0), ink=('#241604', None, None, None)))
        # pebbles: jump on every step, some never come down
        cel = Cel(); fl_ = 1 - (gone * min(1., (f - 75) / 4.))
        for k, (p, (a, sz)) in enumerate(zip(self.feet, steps)):
            u = (f - a) / 24.
            if not 0 <= u < .8: continue
            for j in range(11):
                th = TAU * h1(k, j, 1); v = .5 + 1.6 * h1(k, j, 2); X = p + np.array([math.cos(th) * v * u, math.sin(th) * v * u, (2.2 + 2.6 * h1(k, j, 3)) * u - 4.9 * u * u])
                if X[2] < p[2] - .14: continue
                q, z = cam.project(X[None]); r = cam.f / z[0] * (.012 + .045 * h1(k, j, 4) ** 2); ki.rock(cel, ki.chunk(q[0, 0], q[0, 1], r, k * 20 + j, j + u * 9), j, ('#181208', '#5a3c14', '#d89a34'), (-.2, -1), ('#fff0b4', -r * .3, -r * .8), 1.2 if r > 7 else 0, '#0a0703')
            for j in range(7):                                    # slabs round the foot tip up and drop back
                e = f - a
                if e > 9: break
                th = TAU * (j + .6 * h1(k, j, 61)) / 7; rr = .22 + .30 * h1(k, j, 62); X = p + np.array([math.cos(th) * rr, math.sin(th) * rr, -.14]); q, z = cam.project(X[None]); S_ = cam.f / z[0]; w_ = S_ * (.028 + .075 * h1(k, j, 63)); q[0, 1] += S_ * .030 * math.sin(th); tl = (1 - math.exp(-e / .8)) * math.exp(-e / 4.5); sg = 1 if math.cos(th) > 0 else -1; h_ = w_ * (.25 + 1.5 * tl); P_ = np.array([[q[0, 0] - w_, q[0, 1] + 2 - w_ * .30 * (sg < 0)], [q[0, 0] - w_ * .8 + sg * h_ * .25, q[0, 1] - h_ * (.9 if sg > 0 else .35)], [q[0, 0] + w_ * .7 + sg * h_ * .3, q[0, 1] - h_ * (.35 if sg > 0 else .95)], [q[0, 0] + w_, q[0, 1] + 2 - w_ * .30 * (sg > 0)]]); ki.rock(cel, P_, k * 20 + j, ('#181208', '#5a3c14', '#d89a34'), (-.2, -1), ('#fff0b4', -w_ * .2, -w_ * .7), 1.3, '#0a0703')
        for j in range(30):
            q = t * (.22 + .25 * h1(j, 5)) + h1(j, 6); u = q - math.floor(q); th = TAU * h1(j, 7, math.floor(q)); rr = (.35 + 1.2 * h1(j, 8)) * s; px_ = c[0] + math.cos(th) * rr; py_ = c[1] + .95 * s - 2.0 * s * u ** 1.2 if not gone else c[1] + .95 * s - 2. * s * u + (f - 75) ** 2 * 9; r = s * (.010 + .035 * h1(j, 9) ** 2) * min(1., u * 6)
            if abs(math.cos(th)) > .35 and py_ < 876 and r > 1.2: ki.rock(cel, ki.chunk(px_, py_, r, j + 300, j + t * 2), j, ('#181208', '#4a3212', '#c08a30') if not gone else ('#0c0e10', '#20262a', '#39444a'), (np.sign(c[0] - px_), -.3), ('#ffe9a0', np.sign(c[0] - px_) * r * .8, 0) if not gone else None, 1.2 if r > 6 else 0, '#0a0703')
        img = ki.put(img, cel.layer())
        if fr: img = comp.over(img, PD.lobes_layer(cam, fr, dk, light=(.1, -.85, .4), ss=1.25) * (.9 if not gone else 1.))
        d = kick(f, steps + [(76, .8)], 13., 2.0, 3); img = shaken(img, *d)
        Z = self.Z0 if f < 50 else self.Z1 if f >= 57 else self.foot(f); dm = 1. if not gone else (.8, .6, .52, .5)[f - 76]; Z = Z * np.array([dm, dm, dm * 1.04, 1.], f32); img = ki.put(img, shaken(Z, d[0] * 1.6, d[1] * 1.6, 1.0)); 
        if f >= 77: img = ki.put(img, rain(f, (0, 90, 220, 320)[f - 76], 4, .12)); img = ki.put(img, rain(f, (0, 12, 30, 46)[f - 76], 9, .12, (160, 420), (2., 4.4), '#aeb8c0', .35, 150.))
        return img.astype(f32)

    def last(self):
        pos = self.root[75]; self.F.pose(stomp(75)).place(*pos, G(*pos), heading=HD); return safe(self.F, self.cam, 75, pal='gold', light=(-.35, .75, .45))
