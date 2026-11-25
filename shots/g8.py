"""g8 HAA (80).  0-29 behind and beside the two: ignition (the biggest impact frame), they plough back, hold, push; gold and blue twist into one.
30-49 from orbit: one thin bright line from the coast to the falling moon.  50-59 a small pair of hands; 60-69 everyone's.  70-79 the moon cracks from the centre, white seams."""
import os
from db.ki_stage import *
from db.ki_hero import reflect_soft, glints
from db import land, sky, ki, kit, moon
from db import phys_dust as PD
from db.paint import ramp
TOP = land.TOP; G = lambda x, y: float(land.ground_z(x, y))
FW = np.array([-.30, -.954, 0.]); FW /= np.linalg.norm(FW); RT = np.array([FW[1], -FW[0], 0.]); DH = FW * .62 + RT * .78; DH /= np.linalg.norm(DH); HEAD = math.atan2(-DH[0], DH[1])
P0 = np.array([2., -132., 0.]); MX, MY, MR, IX, IY, HZ = 1490, 300, 330, 1380, 372, 612
DAWN = dict(base='dawn', mul=(.78, .68, .84), shmul=(.42, .34, .60), litmul=(1.0, .88, .86), add='#080514', ink='#140c1c')
RB = lambda c, a=1.: dict(dir=(.92, .20, -.10), col=c, thr=.34, amt=a)


def slide(f):
    """metres pushed back along the ground (positive = back)"""
    if f < 2: return 0.
    if f < 20: return .95 * (1 - math.exp(-(f - 2) / 4.2))
    return .95 * (1 - math.exp(-18 / 4.2)) - .42 * es((f - 20) / 9.)


class Shot:
    id = 'g8'; n = 80
    finish = dict(bloom=.28, vig=.16, bloom_thr=.84)
    EVENTS = [(0, 'impact_frame', dict(frames=2, size=1.6, note='HAA: the biggest of the film')), (0, 'beam_fire', dict(size=1.6, note='gold and blue')), (2, 'slide_back', dict(to=16, size=1., note='four feet plough rock')), (7, 'beam_hits_moon', dict(size=1.4)), (8, 'boom', dict(size=1.2)), (2, 'beam_hold', dict(to=29, size=1.2, note='roar, turning')), (20, 'push', dict(to=29, size=1., note='they lean in: the roar climbs')), (30, 'orbit', dict(to=49, note='no effects: music only, one thin line')), (50, 'roar_again', dict(to=59, size=.8)), (51, 'small_hands', dict(size=.35, note='a child puts his palms on their backs')), (54, 'big_hands', dict(size=.5, note='Piccolo')), (58, 'push_together', dict(size=.8)), (60, 'everyone', dict(to=69, size=.8, note='voices join')), (70, 'moon_crack', dict(size=1.4, note='one enormous crack from the centre')), (72, 'cracks', dict(size=.8)), (75, 'cracks', dict(size=1.)), (78, 'cracks', dict(size=1.2))]

    def setup(self):
        cv2.setNumThreads(1); z0 = self.z0 = G(*P0[:2]); C = np.array([P0[0], P0[1], z0]); self.C = C; eye = C - FW * 5.2 + RT * 1.62 + [0, 0, 1.32]; self.cam = cam = Camera3D(eye, eye + FW * 10 + [0, 0, .58], 1250.); self.hz = float(cam.project((eye + FW * 1e6)[None] * [1, 1, 0] + [0, 0, eye[2]])[0][0, 1]); p = 'cache/g8_v4.npz'
        if os.path.exists(p): q = np.load(p); self.sky0, self.gl, self.ra, self.rb = [q[k].astype(f32) for k in ('s', 'g', 'a', 'b')]
        else:
            self.sky0 = self.paint_sky(); self.gl = glints(self.hz, 0., 3); L = []
            m = C + FW * 1.4 + RT * 1.2 + [0, 0, 1.4]
            for fires in ((), [(*m, 8., '#ffd98a', 2.0), (*(m + RT * 1.6), 7., '#7fc8ff', 1.3)]): o = land.ground(cam, .5, 'first', moon=None, fires=fires); a = o.a if o.a.ndim == 2 else o.a[..., 0]; al = np.clip(o.rock * (o.z < 90), 0, 1).astype(f32); al = cv2.GaussianBlur(al, (0, 0), .6); L.append(np.dstack([o.img * al[..., None], al]).astype(f32))
            self.ra, self.rb = L; np.savez(p, **{k: v.astype(np.float16) for k, v in (('s', self.sky0), ('g', self.gl), ('a', L[0]), ('b', L[1]))})
        self.K = Fig('goku', 'bare', seed=1); self.V = Fig('vegeta', 'shattered', seed=2); self.O = Fig('gohan', 'nohat', seed=3); self.Q = Fig('piccolo', 'torn', seed=4)

    def paint_sky(self):
        x, y = grid(); hz = self.hz; g = np.clip(y / hz, 0, 1.2); im = ramp(g, [(0, '#0a0c2a'), (.35, '#1d2158'), (.62, '#4a3f86'), (.84, '#b0708f'), (.95, '#f7b7a8'), (1, '#ffd9c2')]).astype(f32); n = nimg(H, W, 300, 0, 5, 3, 4, 2.4, .8); im = im * (1 + .10 * n[..., None]); st = np.random.default_rng(4)
        for i in range(240): sx, sy = st.uniform(0, W), st.uniform(0, 340); im[int(sy):int(sy) + 1 + (i % 9 == 0), int(sx):int(sx) + 1 + (i % 9 == 0)] += .5 * st.random() + .2
        for k, (c0, sd, yy, hh, cols) in enumerate((((60, 320, 560), 3, hz + 8, 170, ('#3f3478', '#5f4a90', '#8f6a9f', '#d79aa0', '#ffd8bc')), ((840, 1140, 1960, 1700), 9, hz + 6, 120, ('#4a3c82', '#6f569a', '#a37c9f', '#e6a8a0', '#ffe2c8')), ((100, 640, 1240, 1540), 14, hz + 12, 66, ('#5a4488', '#85649c', '#c08ea0', '#f4bcaa', '#fff0da')))):
            pts = []
            for j, c in enumerate(c0): pts += ki.heap(c, yy, 430 - 60 * k, hh * (.8 + .4 * h1(sd, j)), sd + j, 30)
            L = ki.cumulus(pts, (.75, -.75), cols, ink='#2a2050', seed=sd, k=6., lw=.8); im = put(im, L * np.clip((hz + 2 - y), 0, 1)[..., None])
        return im

    # ------------------------------------------------------------------ 0-29
    def heroes(self, f, cam, root=0., lean=0.):
        K, V = self.K, self.V; K.reach(); V.reach(); s = slide(f); off = -DH * s; hold = es((f - 3) / 6.) * (1 - es((f - 20) / 8.)); push = es((f - 20) / 8.); pk = mix_pose(POSES['kame_fire'], POSES['kame_hold'], hold); pv = mix_pose(POSES['ff_fire'], P('ff_fire', root=(0, .0, -.185), pel=(20, 0, 0), wst=(10, 0, 0)), hold); tr = .006 * math.sin(f * 2.7)
        if push > 0: pk = P(pk, pel=(22 * push + 18 * (1 - push), 0, 0)); pv = P(pv, pel=(22 * push + 16 * (1 - push), 0, 0))
        for F, p_, d in ((K, pk, -RT * .62 + FW * .30), (V, pv, RT * .42 - FW * .12)):
            q = self.C + d + off; F.pose(p_).place(q[0], q[1], G(q[0], q[1]) + tr, heading=HEAD, pitch=-.03); F.blow(*(-DH * (2.4 if f < 8 else 1.6)))
            sk = F.skel(); sh = (sk.sh['L'] + sk.sh['R']) / 2; a = DH * .965 + np.array([0, 0, .30]); a /= np.linalg.norm(a); tg = sh + a * .62 * sk.H / 1.76; F.reach(R=dict(p=tg + [0, 0, .035], part='wrist', w=1., body=.2), L=dict(p=tg - [0, 0, .035], part='wrist', w=1., body=.2))
        kw = dict(pal=DAWN, light=(.65, .45, .45)); LK = safe(K, cam, f, **dict(kw, rim=[RB('#a9e4ff'), dict(dir=(.5, .7, .2), col='#f2fbff', thr=.62, amt=.8)], expr=dict(expr='scream'))); LV_ = safe(V, cam, f, **dict(kw, rim=[RB('#ffe08a'), dict(dir=(.5, .7, .2), col='#fff3c8', thr=.62, amt=.8)], expr=dict(expr='fire' if f >= 6 else 'shout'))); h_ = lambda F: np.mean([np.array(F.info['hands'][q]) for q in 'LR'], 0); return put(LK, LV_) if False else stack(LK, LV_), LK, LV_, h_(K), h_(V)

    def s1(self, f, graphic=True):
        t = f / 24.; cam = self.cam; x, y = grid(); L, LK, LV_, hk, hv = self.heroes(f, cam); M0 = hv * .62 + hk * .38 + [46, -6]; kick_ = 1.; on = es((f - 1) / 3.); imp = max(0., (f - 7)); rec = min(1., imp / 14.); mx, my = MX + 16 * eo(rec) - 22 * (1 - es(f / 9.)) * 0, MY - 9 * eo(rec); k = 1. + .30 * math.exp(-max(0, f - 2) / 2.2) - .13 * es((f - 10) / 6.) * (1 - es((f - 20) / 5.)) + .16 * es((f - 20) / 8.)
        ax = ki.Axis2(spl([M0, M0 + (np.array([IX, IY]) - M0) * .30 + [0, 26], M0 + (np.array([IX, IY]) - M0) * .64 + [0, 14], (IX, IY)], n=14), lambda v: (172 * (1 - v) ** 1.25 + 46) * (.50 + .50 * min(1., v / .12) ** .7) * k, tilt=.34)
        im = self.sky0.copy(); F = Field(2); r = R(8)              # cirrus torn away from the impact
        for i in range(150):
            a = TAU * (i + r.random()) / 150; d0 = r.uniform(400, 1250) ; l = r.uniform(120, 520) * (d0 / 700) ** .6; w = (1.5 + 13 * bms(r.random(), .12, .3) ** 1.5) * (d0 / 800); d1 = d0 + 170 * eo(imp / 20.); P0_ = np.array([IX + math.cos(a) * d1, IY + math.sin(a) * d1 * .66]); v = lin(12); an = math.atan2(math.sin(a) * .66, math.cos(a)) + r.normal(0, .05); Q = P0_ + np.stack([np.cos(an + .10 * v) * l * v, np.sin(an + .10 * v) * l * v], -1); F.rib(Q, [w * (.35 + .65 * min(1, imp / 5. + .3)) * np.sin(math.pi * v ** .6) ** .9 * q for q in (1, .5)], 5.)
        sm = np.clip((self.hz - y), 0, 1)[..., None]; im = put(im, F.cel(('#54488e', '#9a84c0'), alpha=(.50, .55)) * np.clip((self.hz - 40 - y) / 160., 0, 1)[..., None]); B = moon.ball(mx, my, MR, spin=.3, t=.2 + t * .3); B[..., :3] *= C('#fbeaf2'); hl = moon.ball_halo(np.zeros((H, W, 3), f32), mx, my, MR, .55); im = im + hl * sm; im = put(im, B * sm)
        Z = []
        if f >= 2: hl_, hg = ki.ignite(ax, f, 0, 'goldblue', frames=7, style='helix', seed=3, muzzle_r=46. * k, back=math.pi, collars=5, dens=1.) ; Z.append((hl_, hg))
        if imp > 0 or f == 7: il, ig = ki.impact_front(IX, IY, (40 + 62 * eo((imp + 1) / 7.)) * (.92 + .10 * k), f, ('gold', 'blue'), asp=.94, rot=-.4, back=(math.pi * .86, .30), n=28); Z.append((il, ig))
        full = im
        for l, g in Z: full = put(full, l) + g * .8
        # the sea
        dd = np.clip((y - self.hz) / (H - self.hz), 0, 1); wv = self.gl; sea = ramp(dd, [(0, '#a9788f'), (.06, '#6a5490'), (.25, '#2a2e70'), (1, '#0b1038')]).astype(f32); ref = reflect_soft(full, self.hz, t); dk = np.clip((-wv - .10) * 16 + .5, 0, 1)[..., None]; sea = sea * (1 - dk * .30); m0 = np.clip((wv - .02) * 16 + .5, 0, 1)[..., None]; m1 = np.clip((wv - .34) * 16 + .5, 0, 1)[..., None]; sea = sea * (1 - m0 * .55) + ref * m0 * .55; sea = sea * (1 - m1 * .6) + (ref * .95 + .10) * m1 * .64; im = np.where((y > self.hz)[..., None], sea, im).astype(f32)
        if f >= 4:                                                  # the sea opens under the beam
            tr = ki.Axis2(spl([(M0[0] + 120, 1010), (900, 820), (1140, 704), (1330, self.hz + 10)], n=10), lambda v: 110 * (1 - v) ** 1.35 + 5, tilt=.2); hdm = tr.M * min(1., (f - 3) / 6.); Bw = Field(4); D = Field(1); mm = tr.msamp(0, hdm, 14); D.rib(tr.pt(mm, .06), [tr.r(mm) * .78], 4.)
            for sg in (1, -1):
                for row, (pit, sc) in enumerate(((.52, 1.), (.27, .55))):
                    for c in range(int(hdm / pit)):
                        z = bms(h1(c, sg, row, 2), .25, .35) * sc; mq = (c + h1(c, sg, row)) * pit; rr = float(tr.r(np.array([mq]))[0]); q = t * (1.6 + h1(c, sg, 4)) + h1(c, sg, 5); u = q - math.floor(q); p = tr.pt(np.array([mq]), sg * (.80 + .12 * h1(c, 7)))[0]; h = rr * (.55 + 1.25 * z) * (1 - (2 * u - 1) ** 2 * .75); Bw.rib(spine(p[0], p[1], h, -math.pi / 2 - sg * .42 + .30, 10, .06, .7, u * 6 + c, sg * -.35), kwidths(10, h, rr * (.13 + .18 * z), LV, .2, 1.1, 1.), rr * .05)
                Bw.rib(tr.pt(mm, sg * .86), [tr.r(mm) * q for q in (.20, .15, .07, .0)], 6.)
            im = put(im, D.cel(('#070b2c',)) * .9); im = put(im, ki.streams(tr, t, 0, hdm, 5, 24, 16, .6, (1, 3), (.02, .06)).cel(('#ffcf6a',)) * .9); im = put(im, Bw.cel(('#22439e', '#5aa6ee', '#c6ebff', '#fff6dc'), lw=(1.5, 0, 0, 0), ink=('#0a1446', None, None, None), seed=3))
        for l, g in Z: im = put(im, l) + g
        if imp > 0:
            for k2 in range(2):
                q = imp - 4 * k2
                if 0 < q < 18: im = put(im, ki.ring(IX, IY, 120 + 460 * (1 - math.exp(-q / 7.)), 12 * math.exp(-q / 6.) + 1, f, 'base', .9, -.4, 5 + k2, .45, math.exp(-q / 9.), spurs=26, wob=.04))
            c = Cel()
            for i in range(46):
                a = TAU * h1(i, 1); v = 60 + 760 * h1(i, 2); d = 90 + v * (1 - math.exp(-imp / 14.)); p = np.array([IX + math.cos(a) * d, IY + math.sin(a) * d * .85 + imp * imp * .25]); s = (3 + 17 * bms(h1(i, 3), .16, .3)) * min(1., imp / 4.); tail = np.array([math.cos(a), math.sin(a) * .85]) * s * (5 * math.exp(-imp / 8.) + 1.2); c.fill(np.array([p - tail, p + [s * .3, s * .3], p + [s * .35, -s * .3]]), '#e9dcff', .5); ki.rock(c, ki.chunk(p[0], p[1], s, i, a + 1 + imp * .08 * (h1(i, 5) - .5) * 4, 1.3), i, ('#6a58a8', '#b9a6ea', '#fff6e0'), (-.8, .5), None, 1.2 if s > 7 else 0, '#2c1a5c')
            im = put(im, c.layer())
        fl = (1 + .08 * math.sin(f * 2.3)) * on; rk = self.ra * (1 - min(1, fl)) + self.rb * min(1., fl); lu = np.clip(rk[..., :3].mean(-1) * 3.4, 0, 1); fa = np.exp(-np.hypot((x - M0[0] - 40) / 620., (y - 800) / 190.)); q_ = np.round(np.clip(fa * (.35 + .95 * lu) * fl, 0, 1) * 5) / 5. * rk[..., 3]; rg_ = np.exp(-((x - M0[0] - 520) / 560.) ** 2 - ((y - 800) / 150.) ** 2) * np.round(lu * 4) / 4. * rk[..., 3] * on
        rk = rk.copy(); rk[..., :3] += q_[..., None] * C('#ffb45a') * .95 + (rg_[..., None] * C('#7fb0ff') * .55); im = put(im, rk)
        # furrows, dust, plates where the four feet dig in
        c = Cel(); s = slide(min(f, 19)); rows = []
        for F_, d in ((self.K, -RT * .62 + FW * .30), (self.V, RT * .42 - FW * .12)):
            sk = F_.skel()
            for q in 'LR':
                a = sk.an[q].copy(); a0 = a + DH * slide(f) ; a1 = a + DH * (slide(f) - s) if f < 20 else a0 - DH * s; a0[2] = G(a0[0], a0[1]) + .01; a1[2] = G(a1[0], a1[1]) + .01
                if s > .06:
                    n = np.cross(DH, [0, 0, 1.]); w = .085; v = lin(14)[:, None]; Pm = a0 + (a1 - a0) * v; jag = (.75 + .5 * np.array([h1(i, ord(q), 3) for i in range(14)]))[:, None]; A_ = cam.project(Pm + n * w * jag)[0]; B_ = cam.project(Pm - n * w * jag)[0]; c.fill(np.vstack([A_, B_[::-1]]), '#150c22'); c.line(A_, 3.2, '#ffcf8a', taper=(.1, .1), seed=3); c.line(B_, 2.2, '#3a2a52', taper=(.1, .1), seed=4)
                    for j in range(9): X = a1 - DH * (.05 + .22 * h1(j, ord(q), 1)) + n * (h1(j, ord(q), 2) - .5) * .42 + [0, 0, .05 * h1(j, 5)]; p, z = cam.project(X[None]); r_ = cam.f / z[0] * (.02 + .075 * h1(j, ord(q), 6) ** 2) * min(1., s * 2); ki.rock(c, ki.chunk(p[0, 0], p[0, 1] - r_ * .4, r_, j + ord(q), h1(j, 7) * 6), j, ('#1a1230', '#4a3a66', '#c79aa8'), (.6, -.8), ('#ffdca0', r_ * .7, -r_ * .5), 1.3 if r_ > 6 else 0, '#0d0818')
                if 2 <= f < 20:
                    for e in range(3): u = (f - 2 - e * 2) / 24.; rows += [w for w in PD.plume(max(u, .02), tuple(a1), tuple(-DH * .9 + np.array([0, 0, .55])), v0=3.4 * math.exp(-e * .4), n=4, r=(.04, .15), spread=.5, emit=.3, seed=e * 7 + ord(q), life=.9, drag=2.6, rise=.3, floor=a1[2] - .02)] if u > 0 else []
        im = put(im, c.layer()); c = Cel()
        for i in range(34):                                         # stones that the light lifts
            th = TAU * h1(i, 21); rr = .8 + 3.4 * h1(i, 22); X = self.C + np.array([math.cos(th) * rr, math.sin(th) * rr, (1 - math.exp(-f / 9.)) * (.2 + 2.0 * h1(i, 23)) + .02 * math.sin(f * .3 + i)]) - DH * .06 * f * h1(i, 24); p, z = cam.project(X[None])
            if z[0] < 1.0: continue
            r_ = cam.f / z[0] * (.018 + .085 * h1(i, 25) ** 2.4); ki.rock(c, ki.chunk(p[0, 0], p[0, 1], r_, i + 70, h1(i, 26) * 6 + f * .05 * (h1(i, 27) - .5)), i, ('#1a1230', '#4a3a66', '#b48aa0'), (.8, -.5), ('#ffd79a' if i % 2 else '#a9dcff', r_ * .8, -r_ * .2), 1.3 if r_ > 7 else 0, '#0d0818')
        im = put(im, c.layer())
        pw = (1. + .6 * math.exp(-f / 4.)) * (1 + .12 * es((f - 20) / 8.)); AV = ki.aura(LV_, f, 'gold', pw, 3, height=.46, up=(-.10, -1.), front=.0, light=.45, veil=.25, keep=[(*hv, 34.)]); AK = ki.aura(LK, f, 'blue', pw, 9, height=.50, up=(-.46, -.89), front=.0, light=.45, veil=.25, keep=[(*hk, 34.)]); im = AV.under(im); im = AK.under(im)
        if f >= 2: m_, g_ = ki.muzzle(M0[0] + 6, M0[1], 52 * k, f, 'gold', 5, 1.15, 0, 1., 'blue', .7, math.pi, 34); im = put(im, m_) + g_ * .9 * (1 - L[..., 3:4])
        sh = np.zeros((H, W), f32)
        for F_ in (self.K, self.V):
            p, z = cam.project(np.array([F_.skel().an[q] * [1, 1, 0] + [0, 0, G(*F_.skel().an[q][:2])] for q in 'LR'])); 
            for q in p: cv2.ellipse(sh, (int(q[0] - 40), int(q[1] + 6)), (95, 15), 4, 0, 360, 1., -1, cv2.LINE_AA)
        im = im * (1 - .45 * cv2.GaussianBlur(sh, (0, 0), 3)[..., None])
        if rows: im = comp.over(im, PD.lobes_layer(cam, rows, 'dawn', light=(.75, -.6, .3), ss=1.25) * .9)
        im = put(im, L); im = im + (AV.light + AK.light) * (1 - .9 * L[..., 3:4])
        if f >= 2:                                                  # the two muzzles, one each: gold in Vegeta's hands, blue in Goku's
            for h_, kd, kd2, sd in ((hk, 'blue', None, 7), (hv, 'gold', None, 4)):
                m, g = ki.muzzle(h_[0] + 16, h_[1] - 4, 25 * k, f, kd, sd, 1.1, 0, 1., kd2, .6, math.pi, 22, spikes=True); im = put(im, m) + g * (1 - .6 * L[..., 3:4])
        d = kick(f, [(0, 1.4), (7, 1.), (20, .4)], 13., 4., 5) + kit.shake(f, 2.6, 3); im = shaken(im, *d)
        if graphic and f < 2: return self.burst(im, L, M0, f)
        return im.astype(f32)

    def burst(self, im, L, M0, f):
        """the biggest impact frame: 0 black, the two beams as paper-cut gold and blue; 1 the negative"""
        c = Cel(ss=2); out = np.empty((H, W, 3), f32); bk, pw = ('#07060c', '#fffaf0') if f == 0 else ('#fffaf0', '#07060c'); out[:] = C(bk); d = np.array([IX, IY]) - M0; th = math.atan2(d[1], d[0]); dv = np.array([math.cos(th), math.sin(th)]); n = np.array([-dv[1], dv[0]]); r = R(f + 3, 5)
        for i in range(120): a = r.uniform(0, TAU); z = bms(r.random(), .12, .3); q = 260 * r.uniform(.8, 2.); w = .022 * z * r.uniform(.6, 1.6) + .0015; c.fill(np.array([M0 + [math.cos(a) * q, math.sin(a) * q], M0 + [math.cos(a - w) * 2600, math.sin(a - w) * 2600], M0 + [math.cos(a + w) * 2600, math.sin(a + w) * 2600]]), pw)
        for j, (col, o, w0, w1) in enumerate((('#ffc21e', -1, 60, 300), ('#2f8cff', 1, 60, 300))):
            v = lin(40); wob = 26 * np.sin(v * 9 + j * math.pi) * v; Pc = M0 + dv * (2300 * v)[:, None] + n * (o * (w0 * .6 + (w1 - w0) * .38 * v) + wob)[:, None]; hw = (w0 + (w1 - w0) * v) * .5 * (1 + .16 * noise(v * 14, j * 3., 0, f)); Pl, Pr = Pc + n * hw[:, None], Pc - n * hw[:, None]; c.fill(np.vstack([Pl, Pr[::-1]]), col)
            for i in range(26): q = int(r.integers(2, 36)); s = r.choice([-1, 1]); b = (Pl if s > 0 else Pr)[q]; l = r.uniform(60, 260); c.fill(np.array([b - dv * 20, b + n * s * l * .34 + dv * l, b + dv * 26]), col)
            for i in range(16): q0 = r.uniform(.02, .8); o2 = r.uniform(-.8, .8); s0 = M0 + dv * 2300 * q0 + n * (o * (w0 * .6 + (w1 - w0) * .38 * q0) + o2 * hw.mean()); c.fill(np.array([s0, s0 + dv * r.uniform(120, 500) + n * r.uniform(2, 9), s0 + dv * r.uniform(40, 200) - n * 5]), pw if i % 2 else bk)
        v = lin(30); c.fill(np.vstack([M0 + dv * (2300 * v)[:, None] + n * (10 + 46 * v + 8 * np.sin(v * 30))[:, None], (M0 + dv * (2300 * v)[:, None] - n * (10 + 46 * v + 8 * np.sin(v * 26 + 1))[:, None])[::-1]]), '#fffaf0')
        for i in range(30): a = r.uniform(0, TAU); l = 150 + 640 * bms(r.random(), .2, .3); dd = np.array([math.cos(a), math.sin(a)]); nn = np.array([-dd[1], dd[0]]) * l * .05; c.fill(np.array([M0, M0 + dd * l * .3 + nn, M0 + dd * l, M0 + dd * l * .3 - nn]), '#fffaf0' if f == 0 or i % 3 else '#ffc21e')
        out = put(out, c.layer()); a = L[..., 3:4]; col = np.where(a > .02, L[..., :3] / np.maximum(a, .02), 0).astype(f32); tone = kit.impact(col, 'inv' if f == 0 else 'bw', .22, .5, '#07060c', '#fffaf0'); e = cv2.dilate(a[..., 0], np.ones((3, 3), np.uint8), iterations=4)[..., None]; out = out * (1 - e) + C(pw) * e; return (out * (1 - a) + tone * a).astype(f32)

    # ------------------------------------------------------------------ 30-49
    def s2(self, f):
        k = (f - 30) / 19.; c = sky.orbit_cam(alt=430 + 10 * k, pitch=-17.2, yaw=14 + 1.6 * k, roll=-8 + .6 * k, focal=1100, pos=(0, -9)); im = sky.earth(c, f, sky.sun_dir(-5 + .5 * k, 80)); p, z = c.project(sky.P(0, 0, 0)[None]); A_ = p[0]; B_ = np.array([1290 + 10 * k, 318 - 12 * eo(k)]); R_ = 148; x, y = grid(); moon.ball_halo(im, B_[0], B_[1], R_, .50); b = moon.ball(B_[0], B_[1], R_, spin=.3, t=f / 24., crack=0.); dk = np.clip(((x - B_[0]) * .75 + (y - B_[1]) * -.35) / R_ * .5 + .62, .30, 1.06)[..., None]; b[..., :3] *= dk; im = put(im, b)
        d = B_ - A_; Lg = np.hypot(*d); u_ = d / Lg; n = np.array([-u_[1], u_[0]]); I = B_ - u_ * R_ * .30 + n * 8; v = np.linspace(0, 1, 160); d2 = I - A_; per = 1.0 - .45 * v; Pm = A_ + d2 * v[:, None] + n * (14 * np.sin(math.pi * v))[:, None]
        im = im + (np.exp(-np.hypot(x - A_[0], (y - A_[1]) * 1.7) / 120.) * .55)[..., None] * C('#ffd89a') + (np.exp(-np.hypot(x - A_[0], (y - A_[1]) * 1.7) / 34.) * .9)[..., None] * C('#cfeaff')
        F = Field(3); ph = f * 1.1
        for col_, s in ((0, 1), (1, -1)):
            T_ = Field(2); q = 6.0 * np.sin(v * 70 - ph + (0 if s > 0 else math.pi)) * (1 - .5 * v); T_.rib(Pm + n * q[:, None], [1.9 * per * (np.cos(v * 70 - ph + (0 if s > 0 else math.pi)) > -.6) + .5, 1.0 * per * (np.cos(v * 70 - ph) > 0) - .2], 0.); im = put(im, T_.cel((('#ffb01e', '#ffe58a'), ('#3f9dff', '#bfeaff'))[col_]))
        F.rib(Pm, [4.0 * per, 2.9 * per, 2.0 * per], 0.); l = F.cel(('#8fd0ff', '#fff2c0', '#ffffff')); im = put(im, l) + bloom(l, .3, (4, 16, 60), (.55, .4, .35), '#e8f2ff', 1.1)
        for j in range(5):
            q = ((f * .06 + j / 5.) % 1.); p_ = A_ + d2 * q + n * 14 * math.sin(math.pi * q); im = put(im, ki.ring(p_[0], p_[1], (12 + 8 * h1(j)) * (1.1 - .5 * q), 1.6, f, 'base', .34, math.atan2(u_[1], u_[0]) + math.pi / 2, j, .3, .8, spurs=0))
        il, ig = ki.impact_front(I[0], I[1], 15, f, ('gold', 'blue'), asp=.8, rot=.5, back=(math.atan2(-u_[1], -u_[0]), .4), n=18); im = put(im, il) + ig
        for j in range(3):
            q = (f - 30) * 1.0 - j * 7 + 8
            if q > 0: r_ = 22 + 105 * (1 - math.exp(-q / 12.)); m = np.clip(R_ - np.hypot(x - B_[0], y - B_[1]) + .5, 0, 1)[..., None]; im = put(im, ki.ring(I[0], I[1], r_, 2.4 * math.exp(-q / 16.) + .4, f, 'violet', .80, .5, 9 + j, .3, .8 * math.exp(-q / 18.), spurs=0) * m)
        st = ki.rays(A_[0], A_[1], 10, 90 + 8 * math.sin(f * 2.1), 4, 3, .03, .7, .1, '#ffffff', None, 0.); im = put(im, st); return im.astype(f32)

    # ------------------------------------------------------------------ 50-69
    def glare(self, f, cx, cy, s=1., ang=-.42):
        """what the beam looks like from straight behind it: a torn white well of light, rings of gold and blue running away from us, sky full of wedges"""
        x, y = grid(); d = np.hypot(x - cx, (y - cy) * 1.05); g = np.clip(d / (1150 * s), 0, 1); im = ramp(g, [(0, '#ffffff'), (.10, '#fff3cc'), (.22, '#ffc766'), (.36, '#8a6ab0'), (.6, '#2c2c6e'), (1, '#0c0e30')]).astype(f32); t = f / 24.; L = []
        for (col, sd, n, w, r1) in (('#ffd25a', 3, 70, .020, 1400), ('#6fb8ff', 5, 60, .016, 1300), ('#fffbe8', 7, 60, .010, 1200), ('#1a1c50', 9, 40, .016, 1600)):
            c = Cel(ss=1)
            for i in range(n):
                q = t * (1.2 + 1.2 * h1(sd, i, 1)) + h1(sd, i, 2); u = q - math.floor(q); a = TAU * h1(sd, i, 3, math.floor(q)); r0 = (90 + 900 * u ** 1.6) * s; l = (80 + 420 * h1(sd, i, 4)) * (.3 + u) * s; hw = w * r0 * (.3 + h1(sd, i, 5)); e = np.array([math.cos(a), math.sin(a)]); m = np.array([-e[1], e[0]]); p = np.array([cx, cy]) + e * r0; c.fill(np.array([p, p + e * l * .35 + m * hw, p + e * l, p + e * l * .35 - m * hw]), col, .85 if sd < 9 else .5)
            L.append(c.layer())
        im = put(im, stack(*L)); F = Field(4, None); ki.petals(F, cx, cy, 70 * s, 360 * s, 36, 4, t, 1., 0, None, .5, .12, 6., 1., 2.2); ki.disc(F, cx, cy, 120 * s, 3, t, 1., 0, (1.15, 1., .86, .70), .10, 8.); B = Field(4, None); ki.petals(B, cx, cy, 90 * s, 470 * s, 34, 9, t, 1., 0, None, .6, .11, 6., 1., 2.); im = put(im, B.cel(PAL['blue']['c'], lw=(2, 0, 0, 0), ink=(PAL['blue']['ink'], None, None, None))); im = put(im, F.cel(PAL['gold']['c'], lw=(2, 0, 0, 0), ink=(PAL['gold']['ink'], None, None, None))); return im

    def backs(self, f, cam, wide=False):
        K, V, O, Q = self.K, self.V, self.O, self.Q; tr = .004 * math.sin(f * 2.9); BK = dict(base='dawn', mul=(.50, .42, .66), shmul=(.24, .18, .42), litmul=(.80, .66, .78), add='#0a0618', ink='#0c0716'); rim = lambda c, s: [dict(dir=(s * .9, .25, -.35), col=c, thr=.30, amt=1.), dict(dir=(-s * .75, .5, -.4), col='#fff3d0', thr=.46, amt=.9)]
        w_ = .30 if wide else 0.; K.reach(); V.reach(); O.reach(); Q.reach(); K.pose('kame_hold').place(-(.46 + w_), .05 + tr, 0, heading=0.); V.pose('ff_fire').place(.46 + w_, .10 - tr, 0, heading=0.); 
        for F_ in (K, V): F_.blow(0, -2.2, .4)
        u = es((f - 50) / 3.2); O.pose(P(root=(0, .02, -.02), pel=(12, 0, 0), wst=(4, 0, 0), nck=(-10, 0, 0), hed=(-14, 0, 0), hL=(.26, .66, .34), hR=(.26, .66, .34), gL='splay', gR='splay', fL=(.10, .10, .045), fR=(.10, -.16, .05))).place(0, -.72 + .14 * u, 0, heading=0.).blow(0, -2., .3); rk = K.skel(); rv = V.skel(); a, b = rv.spot('back') + np.array([-.13, -.02, -.10]), rk.spot('back') + np.array([.13, -.02, -.10]); w = u
        O.reach(R=dict(p=a, part='palm', palm=(0, 1, 0), fingers=(0, .2, 1), w=w, g='splay'), L=dict(p=b, part='palm', palm=(0, 1, 0), fingers=(0, .2, 1), w=w, g='splay')); it = [(K, dict(pal=BK, light=(.2, .6, -.5), rim=rim('#a9e4ff', 1))), (V, dict(pal=BK, light=(-.2, .6, -.5), rim=rim('#ffe08a', -1))), (O, dict(pal=BK, light=(0, .6, -.5), rim=[dict(dir=(.8, .4, -.4), col='#a9e4ff', thr=.40, amt=1.), dict(dir=(-.8, .4, -.4), col='#ffe08a', thr=.40, amt=1.)]))]
        if wide:
            Q.pose(P('palm', hL=(.16, .70, -.10), hR=(.16, .70, -.10), g='splay')).place(.80, -1.45, 0, heading=.12).blow(0, -2.4, .4); so = O.skel(); Q.reach(L=dict(p=so.sh['R'] + np.array([-.04, -.09, .02]), part='palm', palm=(0, 1, 0), fingers=(0, .2, 1), g='splay'), R=dict(p=V.skel().spot('back') + np.array([.04, -.02, .0]), part='palm', palm=(0, 1, 0), fingers=(0, .2, 1), g='splay')); it.append((Q, dict(pal=BK, light=(0, .6, -.5), rim=[dict(dir=(.85, .35, -.4), col='#a9e4ff', thr=.36, amt=1.), dict(dir=(-.85, .35, -.4), col='#ffe08a', thr=.36, amt=1.)])))
        L = compose(it, cam, f); return L, a, b

    def press(self, L, p, r, ink, seed, amt, cloth):
        """what a palm does to a back: cloth pulls into it in short hard folds, skin dimples; a shadow crescent under the heel of the hand -> layer clipped to the bodies"""
        c = Cel(); n = 13 if cloth else 8; a0 = h1(seed, 1) * .4
        t = np.linspace(.15, math.pi - .15, 26); o, i_ = np.stack([np.cos(t) * r * 1.16, np.sin(t) * r * 1.30], -1), np.stack([np.cos(t) * r * 1.02, np.sin(t) * r * .92], -1); c.fill(p + np.vstack([o, i_[::-1]]), ink, .42 * min(1., amt))
        for i in range(n):
            a = TAU * (i + .7 * h1(seed, i, 2)) / n + a0; l = r * (.35 + .75 * h1(seed, i, 3)) * amt * (1. if cloth else .6); e = np.array([math.cos(a), math.sin(a)]); m = np.array([-e[1], e[0]]); b = (h1(seed, i, 4) - .5) * l * .5; s0 = p + e * r * (.92 + .12 * h1(seed, i, 5)); Q = spl([s0, s0 + e * l * .5 + m * b, s0 + e * l + m * b * 1.4], n=8); c.line(Q, (4.2 if cloth else 3.0) * (.6 + .6 * h1(seed, i, 6)), ink, taper=(.04, .75), press=.3, seed=seed + i, tremor=.1)
            if cloth and i % 2: c.fill(np.vstack([Q + m * 2, (Q + m * (3 + 5 * np.sin(np.linspace(0, math.pi, len(Q))))[:, None])[::-1]]), '#5a78c8', .55)
        l = c.layer(); return l * L[..., 3:4]

    def palms(self, u, at, s):
        """a child's palms, then Piccolo's: the hands are the friends lane's drawing; the forearms come in from the lower corners, 15 degrees more than in its own arrangement, and each hand arrives along its own arm"""
        from chars import friends_hands as FHD, piccolo as PI, gohan as GH
        from db import hands as HD
        LT = FHD.light('gold'); out = np.zeros((H, W, 4), f32); push = float(FHD.smooth(.84, .94, u)); land_ = lambda t0, d: float(np.clip((u - t0) / d, 0, 1)); ax = lambda a: np.array([math.sin(math.radians(a)), math.cos(math.radians(a))])
        def lay(l, c): nonlocal out; l = FHD.grade(l, LT, (c[0] - 400, c[1] - 400, c[0] + 400, c[1] + 400), 1.); out = stack(out, l)
        t = land_(.28, .08)
        if t > 0:
            for i, (c, sd, dx) in enumerate(((at[0], 'L', -1), (at[1], 'R', 1))): a = dx * 25.; q = np.array(c, float) + [dx * s * 1.62, s * .30] + ax(a) * ((1 - t) * 190 - 8 * push); lay(FHD._flat(LT, q[0], q[1], s * 1.95, sd, PI.SKIN, HD.build(len=1.06, wid=1.02, boxy=.5, tip=.2, knuckle=1.1), a + dx * 9 * (1 - t), t, PI.BAND, 10 + i, True), q)
        t = land_(.0, .08)
        if t > 0:
            for i, (c, sd, dx) in enumerate(((at[0], 'L', 1), (at[1], 'R', -1))): a = -dx * 22.; q = np.array(c, float) + ax(a) * ((1 - t) * 130 - 6 * push); lay(FHD._flat(LT, q[0], q[1], s, sd, GH.SKIN, GH.Gohan.KID, a - dx * 9 * (1 - t), t, GH.SASH, i, spread=1.15), q)
        return out

    def sparks_in(self, img, f, t0, targets, n, seed, box):
        """small lights that run from behind us into the backs"""
        F = Field(2)
        for i in range(n):
            q = (f - t0) / 9. * (.7 + .6 * h1(seed, i, 1)) + h1(seed, i, 2); u = q - math.floor(q); 
            if q < h1(seed, i, 2) + .02: continue
            p0 = np.array([box[0] + (box[2] - box[0]) * h1(seed, i, 3), box[1] + (box[3] - box[1]) * h1(seed, i, 4)]); p1 = np.array(targets[i % len(targets)]); c = (p0 + p1) / 2 + [0, -160 * h1(seed, i, 5)]; v = np.clip(u - .10 * lin(9), 0, 1)[:, None]; Q = p0 * (1 - v) ** 2 + c * 2 * v * (1 - v) + p1 * v * v; w = (2.2 + 5 * h1(seed, i, 6) ** 2) * (1.25 - .6 * u); F.rib(Q, [w * (1 - lin(9)) + .2, w * .55 * (1 - lin(9)) - .1], 0.)
        l = F.cel(('#ffd98a', '#ffffff')); return put(img, l) + bloom(l, .4, (4, 14, 40), (.4, .3, .25), '#ffe2a6', .8)

    def s3(self, f):
        t = f / 24.
        if f < 60:
            from chars import friends_hands as FHD                 # two backs, close: a child's palms, Piccolo's, then everyone's (lane friends)
            k = (f - 50) / 9.; K, V = self.K, self.V; K.reach(); V.reach(); cam = Camera3D((.0, -2.45 + .10 * k, 1.36), (0, 0, 1.22), 2350.); tr = .003 * math.sin(f * 2.9); c = cam.project(np.array([[0, 1.2, 1.66]]))[0][0]; img = self.glare(f, c[0], c[1], 1.25); BK = dict(base='dawn', mul=(.78, .64, .78), shmul=(.40, .30, .56), litmul=(1.02, .88, .88), add='#080514', ink='#0c0716'); hit_ = lambda t0, a_: a_ * (1 - math.exp(-max(0, f - t0 + 1) / 1.2)) * (f >= t0) * (1 + .35 * math.exp(-max(0, f - t0) / 1.6)); fwd = hit_(51, .022) + hit_(54, .034) + hit_(58, .050)
            PS = lambda n: P(n, root=(0, .02, -.06), pel=(7 + 70 * fwd, 0, 0), wst=(3, 0, 0), cst=(-3, 0, 0), nck=(-4 + 60 * fwd, 0, 0), hed=(-4, 0, 0), hL=(.06, .92, .10), hR=(.06, .92, .10))
            K.pose(PS('kame_fire')).place(-.355, .05 + tr + fwd, 0, heading=0.).blow(0, -1.6, .3); V.pose(PS('ff_fire')).place(.355, .08 - tr + fwd, 0, heading=0.).blow(0, -1.6, .3); L = compose([(K, dict(pal=BK, light=(-.55, .60, .50), lines=1.3, weight=1.1, rim=[dict(dir=(.9, .25, -.45), col='#a9e4ff', thr=.60, amt=1.), dict(dir=(-.7, .6, -.45), col='#fff3d0', thr=.64, amt=.9)])), (V, dict(pal=BK, light=(.55, .60, .50), lines=1.3, weight=1.1, rim=[dict(dir=(-.9, .25, -.45), col='#ffe08a', thr=.60, amt=1.), dict(dir=(.7, .6, -.45), col='#fff3d0', thr=.64, amt=.9)]))], cam, f); img = put(img, L)
            pa = cam.project(K.skel().spot('back')[None] + [-.01, 0, -.07])[0][0]; pb = cam.project(V.skel().spot('back')[None] + [.01, 0, -.07])[0][0]; pa, pb = pa + [0, 120], pb + [0, 102]; q = f // 2 * 2; u = (q - 50) / 8.6; key = (q,); sz = 86.
            for j, (p_, dx, ink) in enumerate(((pa, -1, '#3a1410'), (pb, 1, '#070b24'))):
                if f >= 51: img = put(img, self.press(L, p_ + [0, 6], sz * .80, ink, 5 + j, min(1., (f - 50) / 2.) * (1. + .4 * (f >= 58)), bool(j)))
                if f >= 54: img = put(img, self.press(L, p_ + np.array([dx * sz * 1.62, sz * .30]), sz * 1.45, ink, 11 + j, min(1., (f - 53) / 2.) * (1. + .4 * (f >= 58)), bool(j)))
            if getattr(self, '_ck', None) != key: self._ck = key; self._cl = self.palms(min(1., u), [pa, pb], sz)
            # a soft shadow of the hands on the two backs, then the hands
            ha = self._cl[..., 3]; sm_ = cv2.resize(ha, (W // 4, H // 4), interpolation=cv2.INTER_AREA); sh = lambda dx, dy, sg: cv2.resize(cv2.GaussianBlur(cv2.warpAffine(sm_, np.float32([[1, 0, dx / 4.], [0, 1, dy / 4.]]), (W // 4, H // 4)), (0, 0), sg), (W, H)); cs = np.clip(.85 * sh(10, 24, 3.6) + .50 * sh(18, 48, 10.), 0, .84) * L[..., 3] * (1 - ha); img = img * (1 - cs[..., None]) + C('#140a22') * cs[..., None] * .55
            img = put(img, self._cl)
            for t0, kd in ((51, 'gold'), (54, 'gold'), (58, 'blue')):      # each arrival is a ring of light through the cloth
                e = f - t0
                if 0 <= e < 6:
                    for p in (pb, pa): img = put(img, ki.ring(p[0], p[1], 90 + 190 * (1 - math.exp(-e / 2.)), 9 * math.exp(-e / 2.5) + 1, f, kd, 1., 0, t0, .35, max(0., 1 - e / 6.), spurs=10))
            img = self.sparks_in(img, f, 51, [pb, pa], 34, 4, (120, 980, 1800, 1150)); return shaken(img, *(kit.shake(f, 2.6, 7) + kick(f, [(51, .35), (54, .55), (58, .9)], 10., 2.2, 3)))
        from chars import friends_hands as FHD
        k = (f - 60) / 9.; cam = Camera3D((0, -7.6 - .5 * eo(k), 1.45), (0, 0, 1.28), 1500.); c = cam.project(np.array([[0, 1.4, 1.5]]))[0][0]; img = self.glare(f, c[0], c[1], .8); g = np.clip((grid()[1] - 742) / 60, 0, 1)[..., None]; gr = ramp(np.clip((grid()[1] - 742) / 338, 0, 1), [(0, '#b8805a'), (.12, '#5a3a62'), (.5, '#2a1f48'), (1, '#110c26')]).astype(f32); gl = np.clip((glints(742., 0., 5, 0, 1.2) - .15) * 12 + .5, 0, 1)[..., None] * np.clip(1 - np.abs(grid()[0] - c[0]) / 700, 0, 1)[..., None]; gr = gr + gl * C('#ffcf8a') * .5 * np.clip(1.2 - (grid()[1][..., None] - 742) / 300, 0, 1); img = img * (1 - g) + gr * g; L, a, b = self.backs(f, cam, True); img = put(img, L); tg = [cam.project(self.Q.skel().spot('back')[None])[0][0], cam.project(a[None])[0][0], cam.project(b[None])[0][0]]
        for row, (y0, h, n, dep, sd) in enumerate(((1010, 330, 11, .25, 3), (1130, 470, 8, .10, 9), (1300, 690, 6, .0, 15))):
            cel = Cel(); ht = []
            for i in range(n):
                x0 = 960 + (i - (n - 1) / 2) * (2150 / n) + 60 * (h1(sd, i, 1) - .5); gap = 250 - 30 * row
                if row < 2 and abs(x0 - 960) < gap: x0 += math.copysign(gap - abs(x0 - 960), x0 - 960 + .01)
                up = eo((f - 60 - 1.2 * h1(sd, i, 2) - row) / 4.5); hp = FHD.person(cel, x0 + 3 * math.sin(f * .5 + i), y0 + 14 * math.sin(f * .35 + i * 1.7), h * (.92 + .16 * h1(sd, i, 3)), sd * 31 + i, dep, '#ffd9a0', .35 + .65 * up, None, 2.2 + row * .6)
                ht += [np.array(q) for q in (hp or [])]
            l = cel.layer(); dm = (.62, .46, .34)[row]; l[..., :3] *= dm; l = kit.rim(l, 0, 5 + 2 * row, '#ffd79a', .8, .8); l = comp.blur(l, (0., 1.2, 5.)[row]); img = put(img, l); u = es((f - 62 - row) / 4.)
            for j, q in enumerate(ht[:16]):
                if u > .05: img = FHD.mote(img, float(q[0]), float(q[1]) - 8, (2.6 + 1.6 * row) * u, f, j + row * 20, .9) if j % (1 + (row == 0)) == 0 else img
        img = self.sparks_in(img, f, 61, tg, 60, 8, (60, 640, 1860, 1120)); return shaken(img, *kit.shake(f, 2.4, 9))

    # ------------------------------------------------------------------ 70-79
    def s4(self, f):
        e = f - 70; x, y = grid(); CX, CY, R_ = 960., 486., 452. + 3. * e; cr = (.10, .20, .28, .38, .46, .56, .66, .78, .88, .97)[e]; g = np.clip(y / H, 0, 1); im = ramp(g, [(0, '#0a0c2a'), (.55, '#35307a'), (.85, '#a86c92'), (1, '#f2b0a4')]).astype(f32); F = Field(2); r = R(18)
        for i in range(150):
            a = TAU * (i + r.random()) / 150; d0 = r.uniform(480, 1150) + 10 * e; l = r.uniform(90, 420); P_ = np.array([CX + math.cos(a) * d0, CY + math.sin(a) * d0 * .8]); v = lin(10); w = 1.5 + 13 * bms(r.random(), .12, .3) ** 1.5; F.rib(P_ + np.stack([math.cos(a) * l * v, math.sin(a) * l * v * .8], -1), [w * np.sin(math.pi * v ** .6) ** .9 * q for q in (1, .5)], 4.)
        im = put(im, F.cel(('#5a4c94', '#a48cc8'), alpha=(.5, .55))); sx, sy = kit.shake(f, 5. + 1.2 * e, 4); CX += sx; CY += sy; moon.ball_halo(im, CX, CY, R_, .7); im = put(im, moon.ball(CX, CY, R_, spin=.3, t=3.4, crack=cr)); t = f / 24.
        d = np.hypot(x - CX, y - CY); wl = np.clip(1 - d / (R_ * (.16 + .55 * cr)), 0, 1) ** 1.4; im = im + (wl * (.35 + .9 * cr))[..., None] * C('#fff6e0')
        S_ = Cel(ss=1)                                              # light standing out of the seams
        for i in range(44):
            a = TAU * (i + h1(i, 1)) / 44; r0 = R_ * cr * (.2 + .8 * h1(i, 2)); l = R_ * (.15 + .95 * cr * h1(i, 3)) * (.5 + .5 * math.sin(f * 1.3 + i) ** 2); w = .010 + .020 * h1(i, 4); ee = np.array([math.cos(a), math.sin(a)]); m = np.array([-ee[1], ee[0]]); p = np.array([CX, CY]) + ee * r0; S_.fill(np.array([p, p + ee * l * .3 + m * w * R_, p + ee * l, p + ee * l * .3 - m * w * R_]), '#fffdf2', .55 + .35 * cr)
        im = put(im, S_.layer())
        ax = ki.Axis2(spl([(-260, 1290), (200, 1040), (620, 720), (CX - 8, CY + 8)], n=12), lambda v: 250 * (1 - v) ** 1.1 + 58, tilt=.34); hl, hg = ki.helix(ax, f, speed=14., collars=3, dens=.9); im = put(im, hl) + hg * .8; il, ig = ki.impact_front(CX, CY, 66 + 2 * e, f, ('gold', 'blue'), asp=.96, rot=-.5, back=(math.pi * .78, .35), n=22, reach=3.2); im = put(im, il) + ig
        c = Cel()
        for i in range(38):
            a = TAU * h1(i, 11); v = 40 + 520 * h1(i, 12); dd = R_ * (.12 + .85 * cr * h1(i, 13)) + v * max(0, e - 1) / 24. * 2.2; p = np.array([CX + math.cos(a) * dd, CY + math.sin(a) * dd]); s = (4 + 26 * bms(h1(i, 14), .16, .3)) * min(1., max(0., e - 1 + h1(i, 15)) / 3.)
            if s > 2: ki.rock(c, ki.chunk(p[0], p[1], s, i + 9, a + e * .12 * (h1(i, 16) - .5) * 3, 1.5), i, ('#7a5fc0', '#cbb6f4', '#fffaf0'), (-math.cos(a), -math.sin(a)), ('#ffffff', -math.cos(a) * s * .7, -math.sin(a) * s * .7), 1.4 if s > 8 else 0, '#2c1a5c')
        im = put(im, c.layer()); z = 1 + .006 * e; im = cv2.warpAffine(im.astype(f32), np.float32([[z, 0, 960 * (1 - z)], [0, z, 486 * (1 - z)]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        if e == 0: return self.crackframe()
        if e >= 8: im = im + (np.clip(1 - d / (R_ * (1.15 if e == 8 else 1.9)), 0, 1) ** .8 * (.55 if e == 8 else 1.1))[..., None] * C('#fffaf0')
        return im.astype(f32)

    def crackframe(self):
        """70: one drawing. paper sky, black disc, white cracks from the centre, the two beams as paper cuts"""
        out = np.empty((H, W, 3), f32); out[:] = C('#fffaf0'); c = Cel(ss=2); r = R(70, 2); o = np.array([960., 486.]); R_ = 452.
        for i in range(110): a = TAU * (i + r.random()) / 110; q = R_ * r.uniform(1.04, 1.7); w = .020 * bms(r.random(), .14, .3) * r.uniform(.5, 1.5) + .0012; c.fill(np.array([o + [math.cos(a) * q, math.sin(a) * q], o + [math.cos(a - w) * 2600, math.sin(a - w) * 2600], o + [math.cos(a + w) * 2600, math.sin(a + w) * 2600]]), '#07060c')
        t = np.linspace(0, TAU, 220); c.fill(np.stack([o[0] + np.cos(t) * R_, o[1] + np.sin(t) * R_], -1), '#07060c'); out = put(out, c.layer()); K1, K2 = Field(1), Field(1)
        for i in range(17):
            a = TAU * (i + .7 * r.random()) / 17; l = R_ * r.uniform(.55, 1.22); P_ = ki.bolt_pts(o, o + np.array([math.cos(a), math.sin(a)]) * l, i + 3, .14, 4); s_ = np.linspace(0, 1, len(P_)); w = 24 * r.uniform(.4, 1.2) * (1 - s_) ** .9 + 1.; (K2 if i % 5 == 2 else K1).rib(P_, [w], 0.)
            for k in range(3): j = int(r.integers(1, max(2, len(P_) - 2))); b = a + r.choice([-1, 1]) * r.uniform(.4, 1.0); Q = ki.bolt_pts(P_[j], P_[j] + np.array([math.cos(b), math.sin(b)]) * l * r.uniform(.14, .36), i * 7 + k, .16, 3); K1.rib(Q, [w[j] * .5 * np.linspace(1, .06, len(Q)) + .4], 0.)
        out = put(out, K1.cel(('#fffaf0',))); out = put(out, K2.cel(('#ffc21e',))); c = Cel(ss=2); d = o - np.array([-200., 1270.]); L_ = np.hypot(*d); d = d / L_; n = np.array([-d[1], d[0]]); v = lin(40)
        for col, off in (('#ffc21e', 1), ('#2f8cff', -1)):
            Pc = o - d * (L_ * v)[:, None] + n * (off * (16 + 120 * v) + 14 * np.sin(v * 11 + off))[:, None]; hw = (12 + 105 * v) * (1 + .18 * noise(v * 13, off * 3., 0, 4)); A_, B_ = Pc + n * hw[:, None], Pc - n * hw[:, None]; c.fill(np.vstack([A_, B_[::-1]]), col)
            for i in range(18): q = int(r.integers(3, 38)); s_ = r.choice([-1, 1]); b = (A_ if s_ > 0 else B_)[q]; l = r.uniform(40, 190) * (.3 + q / 40.); c.fill(np.array([b - d * 16, b + n * s_ * l * .34 - d * l, b + d * 20]), col)
        c.fill(np.vstack([o - d * (L_ * v)[:, None] + n * (6 + 40 * v)[:, None], (o - d * (L_ * v)[:, None] - n * (6 + 36 * v)[:, None])[::-1]]), '#fffaf0')
        for i in range(34): a = r.uniform(0, TAU); l = 90 + 380 * bms(r.random(), .2, .3); dd = np.array([math.cos(a), math.sin(a)]); nn = np.array([-dd[1], dd[0]]) * l * .06; c.fill(np.array([o, o + dd * l * .3 + nn, o + dd * l, o + dd * l * .3 - nn]), '#fffaf0')
        return put(out, c.layer()).astype(f32)

    def frame(self, f): return np.ascontiguousarray(self.s1(f) if f < 30 else self.s2(f) if f < 50 else self.s3(f) if f < 70 else self.s4(f)).astype(f32)
