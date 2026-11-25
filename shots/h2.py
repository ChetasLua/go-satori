"""h2 FOUR BELLS.  Crater lake, clear dawn, everything still grey.  Gohan lifts the stone ball from the shallows; star by star (10 20 30 40) it comes back and colour spreads out of it over the frame;
cut 50: his face in its light; 60 Piccolo's big hand sets the hat on his head."""
import os, math, numpy as np, cv2
from db import W, H, comp, kit, ball, land
from db.comp import over, blur, ease, bokeh
from db.paint import rgb, smooth, grid, f32, fbm, lum, ramp
from db.ink import Cel, spl
from db.skim import Skim
from db import hands as HD
from chars import kid_hands as KH, gohan as G
from chars.gohan import Gohan
from chars.friends_hands import hold_ball
from chars.fkit import light as flight, lw as flw

BELL = (10, 20, 30, 40)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); KEY = 'h2_v1'
RH = 208.
def kf(f, K, e=ease):
    if f <= K[0][0]: return np.asarray(K[0][1], float)
    for (a, va), (b, vb) in zip(K[:-1], K[1:]):
        if f < b: t = float(e((f - a) / float(b - a))); return np.asarray(va, float) * (1 - t) + np.asarray(vb, float) * t
    return np.asarray(K[-1][1], float)
def two(f): return f - f % 2
out = lambda t: 1 - (1 - t) ** 3
PINK = dict(far=('#fff0dc', '#f6c9bd', '#d9aebf', '#b59cc2'), mid=('#fbdccd', '#dcb0c2', '#a99ac6', '#8186ba'), near=('#f0c8c6', '#b9a0c6', '#8288bd', '#5f6ca6'))


class Shot:
    id = 'h2'; n = 80
    finish = dict(bloom=.28, vig=.14)
    EVENTS = [(0, 'water', dict(size=.35, note='stone ball lifted out of the shallows; runs and drips until 30')), (4, 'drip', dict(size=.2)), (10, 'bell', dict(size=.8, star=1, note='first star wakes; stone cracks to glass, colour ring 1')), (13, 'drip', dict(size=.15)), (20, 'bell', dict(size=.85, star=2)), (24, 'drip', dict(size=.12)), (30, 'bell', dict(size=.9, star=3)), (40, 'bell', dict(size=1., star=4, note='all four: colour runs out to the edges of the world 40-49')), (50, 'cut', dict(note='his face, warm light from below')), (52, 'breath', dict(size=.2, note='small gasp / sniff')), (58, 'cloth', dict(size=.2, note='big hand comes down with the hat')), (60, 'hat', dict(size=.5, note='hat lands on his head: soft pat')), (63, 'laugh', dict(size=.5, note='Gohan laughs 63-74')), (70, 'pat', dict(size=.3, note='one more pat'))]

    def setup(s):
        cv2.setNumThreads(1); s.xx, s.yy = grid(); p = os.path.join(ROOT, 'cache', KEY + '_eye.npy')
        if not os.path.exists(p): o = land.crater_eye(0.); os.makedirs(os.path.dirname(p), exist_ok=True); np.save(p, o.img.astype(np.float16))
        eye = np.load(p).astype(f32); s.eye = eye
        s.far = bokeh(comp.cam2d(eye, 1010, 624.8, 1.35).astype(f32), 9, 0)               # close: mountains and mist, out of focus
        s.SK = Skim(560, 900, 1.6, K=300., pal=PINK, seed=8); s.SB = Skim(640, 980, 1.6, K=430., pal=PINK, seed=8); s.D = np.hypot(s.xx - 960, s.yy - 486); s.th = np.arctan2(s.yy - 486, s.xx - 960); s.nz = cv2.resize(fbm(H // 2, W // 2, (90, 34, 12), (1, .6, .4), 21), (W, H))
        s.hd = hold_ball(960, 486, RH, 'dawn')[:H]
        q = os.path.join(ROOT, 'cache', KEY + '_chest.npy')
        if not os.path.exists(q):
            L = Gohan.bust(960, -96, 356, 0., -.10, 0., 'gentle', 'dawn', hat=False, pose=[], seed=2); np.save(q, L.astype(np.float16))
        s.chest = bokeh(np.load(q).astype(f32), 5, 0)

    # -------------------------------------------------------------------------------- 0-49
    def colour(s, im, f):
        """grey world; each bell sends colour further out of the ball, with a live edge"""
        R = 0.; e = 0.
        for a, r0, r1, d in ((10, 0, 300, 9), (20, 300, 520, 9), (30, 520, 800, 9), (40, 800, 2300, 10)):
            if f >= a: t = min(1., (f - a) / d); R = r0 + (r1 - r0) * float(out(t)); e = math.exp(-(f - a) / 5.)
        if R >= 2200: return im
        g = lum(im)[..., None]; gr = (g * .80 + .055) * np.array([.985, 1.0, 1.03], f32)
        if R <= 0: return gr
        d = s.D * (1 + .035 * s.nz + .010 * np.sin(s.th * 9 + s.nz * 2)); m = smooth(R + 2.5, R - 2.5, d)[..., None]; o = gr * (1 - m) + im * m
        edge = np.exp(-((d - R) / (5 + 10 * e)) ** 2) * (.30 + .70 * e); halo = np.exp(-np.maximum(R - d, 0) / 60.) * (d < R) * .16 * (e + .3); o = o + (rgb('#ffd79a') * 1.06 - o) * np.clip(edge * .85, 0, 1)[..., None]; return o + (rgb('#ffe2b0') - o) * halo[..., None]

    def close(s, f):
        t = f / 49.; lift = float(kf(f, [(0, 360), (8, -14), (11, 0)], lambda u: 1 - (1 - u) ** 2.2)); cx = 960 + 2.5 * math.sin(f * .13); cy = 486 + lift + 2.2 * math.sin(f * .17)
        far = comp.cam2d(s.far, W / 2 + 10 * t, H / 2 - 6 * t, 1.02 + .02 * t); c_ = cv2.warpAffine(s.chest, np.float32([[1.02, 0, -19], [0, 1.02, -12 + (lift - 360) * .08 + 3 * math.sin(f * .13)]]), (W, H))
        SK = s.SK; SK.hz = 520.; im = SK.sea(f, far, sun=(1260, 100)); w2 = 812. + 4 * np.sin(s.xx * .022 + f * .16) + 2.5 * np.sin(s.xx * .06 - f * .23); im = SK.reflect(im, c_, 818., f, amt=.42, tint='#8d86bd', squash=.55); cm = smooth(w2 + 3, w2 - 3, s.yy)[..., None]; im = over(im, c_ * cm); ln_ = np.exp(-((s.yy - w2) / 2.6) ** 2) * c_[..., 3]; im = im + (rgb('#fff0ea') - im) * (ln_ * .7)[..., None]; back = im
        st = ball.ignite(f, BELL); rl = ball.relight(f, BELL); dry = float(np.clip((f - 12) / 30., 0, 1))
        hl = KH.warp(s.hd, cx, cy, 1.); ar = KH.arms(cx, cy, 1., RH, 130, at=(.82, .60), roll=4, skin=('#d79a84', '#f7cba4', '#a86c70'), band=('#b5303a', '#7c1c2c', '#e2605a'), line='#2a1626', bend=(1.1, 3.4), w=3.6)
        lay = np.zeros((H, W, 4), f32); lay = lay * (1 - ar[..., 3:4]) + ar; b4 = np.ones((H, W, 4), f32)
        # reflection of hands and ball in the shallows
        wl = 902.; disc = np.zeros((H, W, 4), f32); m = smooth(RH + 1, RH - 1, np.hypot(s.xx - cx, s.yy - cy)); amb = float(np.clip(sum(min(q, 1) for q in st) / 4., 0, 1)); disc[..., :3] = (rgb('#5f6266') * (1 - amb) + rgb('#ffa63a') * amb) * m[..., None]; disc[..., 3] = m; rf = disc * (1 - hl[..., 3:4]) + hl; rf = rf * (1 - ar[..., 3:4] * 0) ; rf = rf + ar * (1 - rf[..., 3:4])
        im = SK.reflect(im, rf, wl, f, amt=.55, tint='#b49ac4', squash=.62); im = s.rings(im, f, cx, wl)
        a = ar * (1 - smooth(wl - 4, wl + 10, s.yy + 5 * np.sin(s.xx * .03 + f * .2)))[..., None]; im = over(im, a)
        env = back; pre = im; im = ball.draw(im, cx, cy, RH, st, stone=1. if f < 60 else 0., wet=1 - .85 * dry, drip=f * 1.6, relight=rl, env=env, light=(-.30, .80, .40), sun='#ffe9cc', glass=.85, body=.72, shade=(.96, .95, 1.0), pulses=[(i, r, k * .8) for (i, r, k) in ball.pulses(f, BELL, speed=.12, life=14)], glow=.9, spec=.9)
        im = over(im, hl)
        if lift > 2:                                                          # still half in the water
            wv = wl + 4 * np.sin(s.xx * .03 + f * .4) + 2 * np.sin(s.xx * .071 - f * .5); um = smooth(wv - 3, wv + 3, s.yy)[..., None]; im = im * (1 - um) + (pre * .72 + im * .28) * um; ed = np.exp(-((s.yy - wv) / 3.2) ** 2) * (np.abs(s.xx - cx) < 400) * np.clip(1.2 - np.abs(s.xx - cx) / 360., 0, 1); im = im + (1.0 - im) * (ed * .6)[..., None]
        im = s.drips(im, f, cx, cy, wl)
        if amb > 0: d = np.hypot(s.xx - cx, s.yy - cy) / RH; im = im + rgb('#ff9a4a') * (.10 * amb) * (np.exp(-np.maximum(d - 1, 0) * 1.6) * (d > 1))[..., None]
        for i, b in enumerate(BELL):
            if 0 <= f - b <= 2: p = ball.star_px(cx, cy, RH)[i]; im = kit.flare(im, p[0], p[1], [300, 240, 170][f - b], '#ffb25a', [.6, .35, .14][f - b], streak=.25, ang=.2 + i)
        return s.colour(im, f)

    def rings(s, im, f, cx, wl):
        cel = Cel(W, H, ss=2)
        for b, x, k in ((0, 0, 1.9), (3, -150, .9), (5, 120, .8), (8, -40, .9), (13, 60, .7), (18, -90, .6), (24, 30, .7), (31, -30, .5)):
            a = f - b
            if not 0 <= a < 26: continue
            for j in range(2):
                R = (22 + 15 * (a - j * 5)) * k
                if R < 10: continue
                al = max(0., 1 - a / 26.) * (.8 - .3 * j); th = np.linspace(0, 2 * math.pi, 70); w = 2.6 * al + .4; O = np.c_[cx + x + np.cos(th) * R, wl + 6 + np.sin(th) * R * .22]; I = np.c_[cx + x + np.cos(th) * (R - w), wl + 6 + np.sin(th) * (R - w) * .22 - .5]; cel.fill([O, I[::-1]], '#fff6ee', al, evenodd=True); cel.fill([O + np.array([0, 3.]), I[::-1] + np.array([0, 3.])], '#7f7fb0', al * .5, evenodd=True)
        return over(im, cel.layer())

    def drips(s, im, f, cx, cy, wl):
        cel = Cel(W, H, ss=2); r = np.random.default_rng(4)
        for k in range(30):
            b = r.uniform(-3, 9) if k < 18 else r.uniform(9, 34); x0 = cx + r.normal(0, 130 if k < 18 else 60); y0 = 486 + RH * .92 if k >= 18 else cy + RH * r.uniform(.4, .95); a = f - b; R = r.uniform(2.2, 6.0)
            if a < 0: continue
            y = y0 + .5 * 1.5 * a * a + (6 * a if k < 18 else 0)
            if y > wl + 2: continue
            v = 1.5 * a; tt = np.linspace(0, 2 * math.pi, 17)[:-1]; P = np.c_[x0 + np.cos(tt) * R, y + np.sin(tt) * R * np.where(np.sin(tt) < 0, 1 + v * .12, 1.)]; cel.fill(P, '#f3eef6', .95); cel.fill(np.c_[x0 + np.cos(tt) * R * .5 + R * .15, y + R * .25 + np.sin(tt) * R * .5], '#a9a4cc', .8)
        return over(im, cel.layer())

    # -------------------------------------------------------------------------------- 50-79
    def bust(s, f):
        g = two(f); t = (f - 50) / 29.; zb = 1.10 + .03 * t; bg = comp.cam2d(s.eye, 930 + 14 * t, 520 - 8 * t, zb); bg = bokeh(bg.astype(f32), 4, 0); SB = s.SB; SB.hz = 540 + (610 - 520 + 8 * t) * zb; bg = SB.sea(f, bg, sun=(960 + (1300 - 930 - 14 * t) * zb, 100))
        land_ = float(kf(f, [(57, 0), (60, 1.), (62, -.25), (65, 0)])) if f >= 57 else 0.; dip = 16 * math.exp(-max(0, f - 60) / 3.) * (f >= 60) + 9 * math.exp(-max(0, f - 70) / 2.5) * (f >= 70)
        sc = 196.; hx, hy = 960 + 3 * math.sin(f * .12), 424 + dip + 2 * math.sin(f * .2); pit = float(kf(g, [(50, -.16), (56, -.14), (60, -.20), (64, .10), (70, .04), (74, .14)])); ro = float(kf(g, [(50, -.03), (60, -.05), (65, .06), (79, .02)])); yw = float(kf(g, [(50, .0), (60, .04), (66, .16), (79, .12)]))
        e = 'wonder' if g < 58 else ('wonder', 'shut', float(np.clip((g - 58) / 2., 0, 1))) if g < 62 else ('shut', 'laugh', float(np.clip((g - 62) / 3., 0, 1))) if g < 74 else 'laugh'
        hat_on = f >= 60; lift = 0. if hat_on else float(kf(f, [(54, 2.1), (58, .75), (60, .0)], lambda u: u ** 1.3)); L_ = flight('dawn')
        bx, by, br = hx + 2, hy + 1.96 * sc + 3 * math.sin(f * .15), 104.
        def ov(cel, H_, B):
            w = flw(sc)
            for q in (-1, 1): Gohan.put_arm(cel, B, L_, q, (q * 1.22, 1.02, .10), (q * .80, 1.62, .55), 'fist', HD.euler(0, 0, 0), w, 5 + q, hand_=False)
            if not hat_on and f >= 54: G.hat_parts(cel, H_, L_, 'front', 1, .6, lift, .10 * min(1, lift), ballon=False)
        L = Gohan.bust(hx, hy, sc, yw, pit, ro, e, 'dawn', f=f, wind=(.25 + .1 * math.sin(f * .2), .05), hat=hat_on, ballon=False, pose=[], over=ov, seed=2)
        # amber light from the ball on chin, scarf, chest
        am = np.exp(-((np.hypot((s.xx - bx) / 1.15, s.yy - (by - 40)) / 330.) ** 2)) * (s.yy < by + 130); L[..., :3] = L[..., :3] * (1 + am[..., None] * np.array([.42, .16, -.10], f32)) + (rgb('#ff9a3a') * .10 * am[..., None]) * L[..., 3:4]
        wl = 962.; wv = wl + 3.5 * np.sin(s.xx * .021 + f * .15) + 2. * np.sin(s.xx * .057 - f * .21); bg = SB.reflect(bg, L, wl + 2, f, amt=.46, tint='#8d86bd', squash=.6); im = kit.aura(bg, L, 10, '#ffe2c8', .16); im = over(im, L * smooth(wv + 3, wv - 3, s.yy)[..., None]); ed = np.exp(-((s.yy - wv) / 2.6) ** 2) * L[..., 3]; im = im + (rgb('#fff0ea') - im) * (ed * .75)[..., None]; im = s.still(im, f, hx, wl)
        k = br / RH
        im = ball.draw(im, bx, by, br, tuple(1 + .05 * math.sin(f * .3 + i) for i in range(4)), env=bg, light=(.45, .6, .4), glass=.70, body=.85, glow=1.1, shade=(1.0, .97, .98)); hl = KH.warp(s.hd, bx, by, k); im = over(im, hl)
        if f >= 53: im = over(im, s.pic(f, hx, hy, sc, lift, hat_on))
        return im

    def still(s, im, f, cx, wl):
        """slow rings round him in the shallows"""
        cel = Cel(W, H, ss=2)
        for b in (44, 54, 64, 72):
            a = f - b
            if not 0 <= a < 30: continue
            R = 170 + 11. * a; al = max(0., 1 - a / 30.) * .7; th = np.linspace(0, 2 * math.pi, 90); w = 2.4 * al + .5; O = np.c_[cx + np.cos(th) * R, wl + 8 + np.sin(th) * R * .13]; I = np.c_[cx + np.cos(th) * (R - w), wl + 8 + np.sin(th) * (R - w) * .13 - .4]; cel.fill([O, I[::-1]], '#fff6ee', al, evenodd=True); cel.fill([O + np.array([0, 3.]), I[::-1] + np.array([0, 3.])], '#8f86bb', al * .5, evenodd=True)
        return over(im, cel.layer())

    def pic(s, f, hx, hy, sc, lift, on):
        """Piccolo's big green hand: brings the hat down, pats, withdraws"""
        up = 0. if not on else float(kf(f, [(60, 0), (63, 0), (66, .55), (69, .62), (70, .05), (72, .05), (78, 2.8)])); y = hy - (1.22 + lift + up * .5) * sc - 10; x = hx + 26 + up * 18
        cel = Cel(W, H, ss=2); po = HD.pose((34, 30, 18, 8), (36, 34, 20, 1), (38, 36, 20, -5), (40, 38, 22, -12), thumb=(40, 30, 10, 12, 10), wrist=(18, 0), cup=.3) if up < .8 else HD.POSES['relaxed']
        h = HD.Hand(po, 'L', R=HD.euler(12, 34, 162), pos=(0, 0, 0), bld=HD.build(len=1.06, wid=1.04, boxy=.5, tip=.25, knuckle=1.1, wrist=1.0), pivot=(0, .75, 0), nails='claw'); g = HD.render([h], HD.Cam(x, y, 160, 6.), light=(.8, .5, .35), res=420, clip=(-80, -260, W + 80, H)); pal = dict(HD.PAL['day'], skin='#8fbc66', shade='#4f7f5e', line='#1a1226', nail='#d9e6c0', nail_shade='#9ab08a', nail_dark='#5a7050'); HD.ink(cel, g, pal, w=4.4, seed=2)
        d = np.array([math.sin(math.radians(162 - 180)) * -1, -1.]); d /= np.hypot(*d); wr = np.array([x, y]) + np.array([.14, -.80]) * 160
        n = np.array([1., .12]); t0, t1 = wr + np.array([.0, .06]) * 160, wr + np.array([.14, -.52]) * 160; Q = spl(np.array([t0 - n * 58, t0 + n * 58, t1 + n * 64, t1 - n * 64]), True, corners=(0, 1, 2, 3), n=6)
        arm = spl(np.array([t1, t1 + np.array([18, -150.]), t1 + np.array([60, -420.]), t1 + np.array([120, -760.])]), n=8); T = np.gradient(arm, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; wd = np.interp(np.linspace(0, 1, len(arm)), [0, .3, 1], [54, 80, 110]); A, B = arm + N * wd[:, None], arm - N * wd[:, None]; c2 = Cel(W, H, ss=2); c2.fill(np.vstack([A, B[::-1]]), '#8fbc66'); lf = A if A[6, 0] < B[6, 0] else B; c2.fill(np.vstack([lf, (arm * .55 + lf * .45)[::-1]]), '#4f7f5e'); c2.line(A, 4.2, '#1a1226', taper=(.02, .02), seed=3); c2.line(B, 4.2, '#1a1226', taper=(.02, .02), seed=4)
        M = arm * .62 + (A if lf is B else B) * .38; pk = spl(np.vstack([M[4:26:7] + N[4:26:7] * 10, (M[4:26:7] - N[4:26:7] * 26)[::-1]]), True, n=6); c2.fill(pk, '#e39a88'); kit.outline(c2, pk, 2.6, '#9c3a44', 4)
        c2.fill(Q, '#b8323c'); c2.clip(Q); c2.fill(Q - np.array([54., 0]), '#7c1c2c'); c2.unclip(); kit.outline(c2, Q, 4.2, '#1a1226', 5)
        L = c2.rgba(); h_ = cel.rgba(); o = h_ * (1 - L[..., 3:4]) + L; return kit.rim(o, -5, 3, '#ffcfa0', .85, .6)

    def frame(s, f): return (s.close(f) if f < 50 else s.bust(f)).astype(f32)
