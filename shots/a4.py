"""a4 THE HOLE IN THE CLOUD.  Straight up from below: he climbs past us, whooping, fist first; 18 the fist meets the cloud; a clean round hole with a thick lip, curls of vapour on its rim, the sun in it on the match circle; bloom to white 30-39."""
import math, numpy as np, cv2
from db import W, H, comp, kit, sky
from db.comp import over, blur, ease
from db.paint import rgb, smooth, grid, f32
from db.ink import Cel, spl
from db.nimbus import Nimbus, E as NE
from db.cam import Camera3D
from chars.kid import Kid, P, mixp, E
from chars.kid_head import mixx, EXPR

CX, CY, R0 = 960., 486., 208.
RZ = np.array([[0, -1., 0], [1, 0, 0], [0, 0, 1]])
F_ = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]
def kf(f, K, e=ease):
    if f <= K[0][0]: return np.asarray(K[0][1], float)
    for (a, va), (b, vb) in zip(K[:-1], K[1:]):
        if f < b: t = float(e((f - a) / float(b - a))); return np.asarray(va, float) * (1 - t) + np.asarray(vb, float) * t
    return np.asarray(K[-1][1], float)


class Shot:
    id = 'a4'; n = 40
    finish = dict(bloom=.30, vig=.10)
    EVENTS = [(0, 'whoosh', dict(size=.8, note='climbs past the lens, close, left to centre')), (1, 'voice', dict(size=.8, note='kid whoop 1-14, rising, receding')), (18, 'punch', dict(size=.8, note='fist meets the cloud: soft, huge, like a drum full of air')), (19, 'ring', dict(size=.6, note='ring runs through the cloud, hole opens 18-22 with a small overshoot')),
              (22, 'curl', dict(size=.3, note='vapour curls roll off the rim 22-34')), (24, 'light', dict(size=.5, note='sun pours through the hole')), (30, 'bloom', dict(size=1., note='bloom to white 30-39, choir of air'))]

    def setup(s):
        cv2.setNumThreads(1); s.H = sky.Hole(); s.N = Nimbus(3); s.K = Kid(); s.xx, s.yy = grid(); s.D = np.hypot(s.xx - CX, s.yy - CY); s.th = np.arctan2(s.yy - CY, s.xx - CX)

    def frame(s, f):
        t = f - 18
        R = R0 * float(np.interp(f, [18, 19, 20, 21, 22, 24, 27], [.30, .62, .90, 1.06, 1.08, 1.02, 1.])); k = 0. if f < 18 else float(np.interp(f, [18, 19, 20, 21], [.75, .9, 1., 1.]))
        im = s.H.frame(f, k, c=s.climb(f), R=R if f >= 18 else R0)
        if f < 17: im = kit.zoomblur(im.astype(f32), CX, CY, .018 + .050 * (1 - f / 18.), 13)
        if f < 18: im = s.wisps(im, f, False)
        if 18 <= f < 36:                                                      # the blow runs outward through the soft mass: a real refraction ring, then a fainter one
            q = np.zeros_like(s.D)
            for (dl, am) in ((0, 1.), (4, .45)):
                a = t - dl + .6
                if a <= 0: continue
                Rr = R * .9 + 74 * a ** .92; w = 30 + 5.5 * a; q += am * np.exp(-((s.D - Rr) / w) ** 2) * math.exp(-a / 7.)
            dx = np.cos(s.th) * q; dy = np.sin(s.th) * q; im = cv2.remap(im.astype(f32), (s.xx - dx * 34).astype(f32), (s.yy - dy * 34).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT); lit = np.clip(-(np.sin(s.th)) * .5 + .5, 0, 1); im = im + (rgb('#ffffff') - im) * np.clip(q * .42 * lit, 0, .6)[..., None] - (rgb('#5a78b0') * np.clip(q * .20 * (1 - lit), 0, .5)[..., None]) * .5
        if f >= 19: im = s.lip(im, f, R)
        if f >= 20: im = s.shafts(im, f, R)
        if f < 25: im = s.boy(im, f)
        if f < 18: im = s.wisps(im, f, True)
        if f >= 18: im = s.debris(im, f, R)
        z = 1 + .05 * float(ease(min(1, f / 18.))) + (.010 * math.exp(-max(0, t) / 4.) if t >= 0 else 0.); sh = kit.shake(f, 6., 5, decay=4., t0=18) if t >= 0 else (0, 0)
        M = np.float32([[z, 0, CX * (1 - z) + sh[0]], [0, z, CY * (1 - z) + sh[1]]]) if f < 18 else np.float32([[1, 0, sh[0]], [0, 1, sh[1]]]); im = cv2.warpAffine(im.astype(f32), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if f >= 28:                                                          # the hole fills with white and then it is everything, edge first soft then hard
            u = float(np.clip((f - 28) / 11., 0, 1)); r1 = R0 * (.40 + .62 * float(ease(min(1, u * 2.2)))) + 1500 * max(0, u - .30) ** 2.1; edge = 14 + 420 * u * u; w = smooth(r1 + edge, r1 - 6, s.D + 2.5 * np.sin(s.th * 5 + f * .2) * (1 - u)); im = im + (rgb('#fffef8') * 1.06 - im) * np.clip(w, 0, 1)[..., None]
            g = np.exp(-s.D / (260 + 900 * u)) * (.25 + .6 * u); im = im + (rgb('#fff8e2') - im) * np.clip(g, 0, 1)[..., None]
            if f >= 37: im = im + (1.04 - im) * [.55, .85, 1.][f - 37]
        return im.astype(f32)

    def climb(s, f):
        """the camera climbs with him: the cloud rushes at us, the sun stays where it is"""
        if f >= 18: return None
        d = np.array([0, 162., 418.]); d /= np.linalg.norm(d); e = np.array([0, -150., 2.]) - d * 1500. * (1 - f / 18.) ** 1.45 + np.array([26 * math.sin(f * .55), 0, 0]) * (1 - f / 18.); return Camera3D(e, e + d, 900., cy=560, roll=.10 * math.sin(f * .42) * (1 - f / 18.))

    def wisps(s, im, f, near):
        """vapour torn past the lens: born at the vanishing point, longer and faster as it comes"""
        cel = Cel(W, H, ss=1); r = np.random.default_rng(31 + near); n = 9 if near else 20
        for j in range(n):
            an = r.uniform(0, 2 * math.pi); sp = r.uniform(.085, .13) * (1.35 if near else 1.); q = (f * sp + r.random()) % 1.; r0 = 60 + 1500 * q ** 2.3; ln = (30 + 620 * q ** 1.6) * (1.5 if near else 1.); wd = (2 + 26 * q) * (2.6 if near else 1.); a = np.array([math.cos(an), math.sin(an)]); nn = np.array([-a[1], a[0]]); u = np.linspace(0, 1, 16); C = np.array([CX, CY]) + a * (r0 + ln * u)[:, None] + nn * (wd * .35 * np.sin(u * 5 + j))[:, None]; w = wd * np.sin(np.pi * u ** 1.5) ** .8; cel.fill(np.vstack([C + nn * w[:, None], (C - nn * w[:, None])[::-1]]), '#ffffff' if j % 3 else '#dbe9f8', (.50 if near else .42) * min(1., q * 3))
        return over(im, blur(cel.layer(), 11. if near else 3.2))

    # -------------------------------------------------------------------------------------- the climb
    def boy(s, im, f):
        S = float(np.interp(f, F_, [196, 168, 140, 112, 86, 64, 46, 32, 22, 14, 9.5, 7, 4.5])); px = float(np.interp(f, F_, [742, 800, 842, 874, 900, 922, 938, 948, 955, 959, 960, 960, 960])); py = float(np.interp(f, F_, [1180, 1030, 930, 850, 780, 715, 655, 600, 550, 508, 494, 490, 488]))
        D = float(np.clip(1500. / S, 5.2, 80)); ph = float(np.interp(f, [0, 6, 12, 18, 25], [.10, .34, .62, .86, 1.0])); yaw = .50 + .25 * math.sin(f * .28); roll = .16 * math.sin(f * .36 + .6) + .10
        Rm = E((yaw, ph, roll)); g = f if f < 4 else f - f % 2
        pose = P('punch', armR=(.34, .98, .40), stR=1.42, armL=(.42, -.40, .34), legL=(.44, -1.22, .58), legR=(.30, -2.0, -.30), cst=(-.10 + .06 * math.sin(f * .5), .16, -.06), hed=(.14, -.06, 0)); pose['tail'] = dict(dir=(.15, -1.0, -.45), curl=.7, side=.8, amp=.65, length=2.7, speed=2 * math.pi / 9., lag=.6)
        if f <= 1: pose = mixp(pose, P(pose, armR=(.40, .60, .50), stR=1.1), [1., .4][f])
        ex = dict(EXPR['whoop']); ex['mo'] = 1.2 + .12 * math.sin(f * 1.1); ex['look'] = (.1, .5)
        # the Nimbus under his feet, nose up, its tail poured down toward us
        W0 = Rm @ np.array([-1.25, -3.75, 0.]); kz = D / max(D - W0[2], .8); ncx, ncy, nS = px + W0[0] * kz * S, py - W0[1] * kz * S, S * kz
        nb, nf = s.N.draw(ncx, ncy, nS, f, R=Rm @ RZ, D=max(3.2, D - W0[2]), pal='day', light=(.1, .80, .10), speed=1.5, stretch=1.62 + .26 * math.sin(f * 1.25), squash=.92, tail=1.75, bend=(.85 * math.sin(f * .62 + .5), .45 * math.cos(f * .47)), tone=(84, 50, 14), breath=1.4)
        ax = np.array([px - ncx, py - ncy]); ax = ax / (np.hypot(*ax) + 1e-6); ln = nS * (.55 + .45 * min(1., f / 8.)); sm = kit.smear(nb, ax[0] * ln, ax[1] * ln, 15); nb = nb * .30 + sm * .70      # speed pulls the lobes into one plume
        pl = Cel(W, H, ss=2); nn_ = np.array([-ax[1], ax[0]])
        for j in range(6): o_ = (-1 + .4 * j) * .62 * nS; t0 = .9 + .5 * ((f * .9 + j * 1.7) % 2.2); P0 = np.array([ncx, ncy]) + nn_ * o_ - ax * t0 * nS * .9; pl.line(spl(np.array([P0, P0 - ax * nS * 1.1 + nn_ * .05 * nS * math.sin(f + j), P0 - ax * nS * 2.3]), n=8), max(1., nS * .045), '#fff6d2' if j % 2 else '#d9923e', taper=(.4, .6), seed=j, alpha=.85)
        pl_ = pl.layer(); nb = nb * (1 - pl_[..., 3:4] * np.clip(nb[..., 3:4] * 1.4, 0, 1)) + pl_ * np.clip(nb[..., 3:4] * 1.4, 0, 1)
        L = s.K.draw(px, py, S, pose, R=Rm, expr=ex, pal='sky', light=(.10, .90, .15), f=f, wind=(.10 * math.sin(f * .7), 1., .55 if f < 12 else .4), D=D, rim=dict(dx=0., dy=max(1., min(5., S * .035)), col='#fffdf0', amt=.95, soft=.5))
        im = over(im, s.N.vapour(ncx, ncy, nS, f + 30, R=Rm @ RZ, vel=(-2.5, 20), n=12, life=22, pal='white', D=max(3.2, D), tail=2.2, op=.9, size=1.1))
        if f >= 18:                                                          # through the hole: only inside it
            m = smooth(R0 * 1.0, R0 * .93, s.D)[..., None]; L = L * m; nb = nb * m; L[..., :3] *= .62; nb[..., :3] *= .9
        im = over(im, L); im = over(im, nb)
        if f < 5: d = [26, 22, 16, 10, 5][f]; im = kit.smear(im, d * .5, -d, 9)
        if 1 <= f <= 13:
            c = Cel(W, H, ss=1); kit.focus_lines(c, CX, CY, 300 + 10 * f, 1400, 26, 5., '#ffffff', seed=f // 2, ry=.8, a0=.25, a1=math.pi - .25); im = over(im, c.layer() * .45)
        return im

    # -------------------------------------------------------------------------------------- the rim
    def lip(s, im, f, R):
        """thickness: the wall of the hole, lit on the inside by the sun, and curls of vapour that roll outward off the edge"""
        t = f - 18; th, D = s.th, s.D
        e = R * (1 + .0035 * np.sin(th * 5 + 1) + .003 * np.sin(th * 11)); inn = e * (.872 + .0035 * np.sin(th * 9 + 2)) ; off = np.hypot(s.xx - CX, s.yy - (CY - 7)); wall = smooth(inn - 2, inn + 2, off) * smooth(e + 2, e - 2, D); up = .5 - .5 * np.sin(th)
        col = rgb('#fff6da')[None, None] * np.clip(up, 0, 1)[..., None] + rgb('#d3e1f7')[None, None] * (1 - np.clip(up, 0, 1))[..., None]; n = np.sin(th * 23 + np.sin(th * 5) * 3) * .5 + .5; im = im * (1 - wall[..., None]) + col * wall[..., None]
        core = smooth(inn - 3, inn - 1, off) * smooth(inn + 3, inn + 1, off); im = im + (rgb('#ffffff') * 1.1 - im) * (core * .9)[..., None]
        shd = np.exp(-((D - e - 16) / 22) ** 2) * (D > e) * (.5 + .5 * np.sin(th)); im = im + (rgb('#7f9fd2') - im) * (shd * .32)[..., None]
        cel = Cel(W, H, ss=2); r = np.random.default_rng(6); m = 17
        for i in range(m):
            a0 = 2 * math.pi * (i + r.uniform(-.25, .25)) / m; g = np.clip((t - .8 - r.uniform(0, 2.6)) / 13., 0, 1)
            if g <= .02 or r.random() < .30: continue
            env = math.sin(math.pi * min(1., g * .78 + .10)) ** .7; d = 1 if i % 2 else -1; rad = (20 + 62 * r.random() ** 1.6) * (.45 + .9 * g ** .6); ang = a0 + d * .05 * g; p = np.array([CX, CY]) + np.array([math.cos(ang), math.sin(ang)]) * (R * 1.0 + rad * .55 + 26 * g); u = np.linspace(0, 1, 44); th_ = ang + math.pi + d * (2 * math.pi * (1.05 - .25 * g) * (1 - u) + .9 * g); rr = rad * (.12 + .88 * u ** .8)
            C = p + np.c_[np.cos(th_) * rr, np.sin(th_) * rr]; foot = np.array([CX, CY]) + np.array([math.cos(ang - d * .16), math.sin(ang - d * .16)]) * (R + 1); C = C + (foot - C[-1]) * (np.clip(u - .62, 0, 1) / .38)[:, None] ** 1.6; wd = rad * .30 * env * np.sin(np.pi * np.clip(u * .9 + .08, 0, 1)) ** .8 * (1.25 - .55 * u); T = np.gradient(C, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; A, B = C + N * wd[:, None], C - N * wd[:, None]
            cel.fill(np.vstack([A, B[::-1]]), '#dbe7f8', .92); lit = A if (N[20] @ np.array([-math.cos(ang), -math.sin(ang)])) > 0 else B; cel.fill(np.vstack([lit, (C * .85 + lit * .15)[::-1]]), '#ffffff')
        return over(im, cel.layer())

    def shafts(s, im, f, R):
        """sunlight falls through the hole in hard-edged beams"""
        a = float(np.clip((f - 20) / 8., 0, 1)); n = 46; r = np.random.default_rng(3); tb = np.repeat(r.random(n) > .42, 24).astype(f32); tb = cv2.GaussianBlur(tb[None], (0, 0), 2.2)[0]; tb[-60:] = tb[0] * np.linspace(0, 1, 60) + tb[-60] * np.linspace(1, 0, 60); u = ((s.th / (2 * math.pi) + .5 + .0012 * f) * len(tb)) % len(tb); b = np.interp(u.ravel(), np.arange(len(tb) + 1), np.r_[tb, tb[:1]]).reshape(H, W); reach = 360 + 900 * a; m = b * smooth(R * .98, R * 1.06, s.D) * np.clip(1 - (s.D - R) / reach, 0, 1) ** 1.2
        return im + (rgb('#fffbe6') - im) * (m * .34 * a)[..., None]

    def debris(s, im, f, R):
        """lumps of cloud thrown out of the hole toward us: they swell (they come closer), lag, thin"""
        cel = Cel(W, H, ss=2); r = np.random.default_rng(15)
        for k in range(20):
            b = r.uniform(0, 2.2); a = f - 18 - b; life = r.uniform(9, 16); ang = r.uniform(0, 2 * math.pi); v = r.uniform(34, 92); r0 = r.uniform(5, 20)
            if not 0 < a < life: continue
            d = R * .8 + v * 9 * (1 - math.exp(-a / 9.)); x, y = CX + math.cos(ang) * d, CY + math.sin(ang) * d + .16 * a * a; q = r0 * (1 + .16 * a) * max(.05, 1 - a / life) ** .6; th = np.linspace(0, 2 * math.pi, 30)[:-1]; w = 1 + .07 * np.sin(th * 5 + k + a * .15) + .05 * np.sin(th * 7 - k * 2); Pq = np.c_[x + np.cos(th) * q * w, y + np.sin(th) * q * w * .9]; cel.fill(Pq, '#c3d5f0', .96); cel.fill(Pq * .78 + np.array([x - math.cos(ang) * q * .2, y - math.sin(ang) * q * .2]) * .22, '#f4f8ff', .98); cel.fill(Pq * .42 + np.array([x - math.cos(ang) * q * .45, y - math.sin(ang) * q * .45]) * .58, '#ffffff')
        return over(im, cel.layer())
