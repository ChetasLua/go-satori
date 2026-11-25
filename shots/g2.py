"""g2 UP.  0-9 out of the storm ring, seen from below.  10-19 sunlit towers, side: the uppercut lands on 10, its shock bores a round hole through the violet cloud behind.  20-29 from above: the axe kick
lands on 20, the deck far below opens.  30-39 blue to indigo, two helical vapour trails.  40-79 space, silent: the curve of the Earth rolls over, aurora, two lights meet on 48, 58, 68, each meeting leaves a ring of frozen vapour; 76-79 still."""
import math, numpy as np, cv2
from db import W, H, comp, kit, sky, ki
from db.sky_storm import Climb, ZC
from db.sky_fly import burst_white
from db import sky_orbit as O
from db.paint import rgb, f32, smooth, mix, grid, ramp, Canvas, arcs, fbm
from db.comp import ease, over, blur
from db.cam import Camera3D
from db.hero3 import Fig, hit, seq, compose, POSES, mix_pose
from db.ink import Cel

MEET = (48, 58, 68)
AZ = {'A': 0., 'B': math.radians(270), 'C': 0., 'D': math.radians(225)}


def rz(v, a): c, s = math.cos(a), math.sin(a); return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]])
def twos(f): return f - f % 2
def lk(f, K): return float(np.interp(f, [k for k, _ in K], [v for _, v in K]))


class Stage:
    """local frame: x right, y away, z up around the fight's middle at altitude h, turned by az"""
    def __init__(s, h, az): s.o = np.array([0, 0, h], float); s.az = az
    def P(s, v): return s.o + rz(v, s.az)
    def cam(s, e, t, foc, roll=0., cx=960, cy=540): return Camera3D(s.P(e), s.P(t), foc, cx=cx, cy=cy, roll=roll)
    def put(s, F, pose, v, head, pitch=0., roll=0.): F.reach(); F.look(None); return F.pose(pose).place(*s.P(v), head + s.az, pitch, roll, 'pelvis')


def alt(f):
    """altitude of the fight in metres"""
    return lk(f, [(0, 360), (9, 872), (10, 1000), (19, 1120), (20, 1780), (29, 2120), (30, 4200), (39, 7600)])


class Shot:
    id = 'g2'; n = 80
    finish = dict(bloom=.34, vig=.14)
    EVENTS = [(0, 'whoosh', dict(size=1., note='out of the storm ring from below: wet roar of cloud walls rushing past, 0-9, rising in pitch')), (0, 'aura', dict(size=.8, kind='gold', dur=40, note='gold furnace climbing with the lens, 0-39, thinner with altitude')), (2, 'thunder', dict(size=.25, note='storm left behind, muffled, below us; no flash on screen')),
              (8, 'burst', dict(size=.7, note='wall of light: they break out of the top of the storm')), (10, 'impact', dict(size=.9, note='uppercut in the stomach, bar line, first frame of the cut, contact held 10-11')), (12, 'impact_cut', dict(size=.6, note='flash and ring')), (12, 'cloud_hole', dict(size=.8, note='the shock bores a round hole through the cloud behind: soft huge pop and a long breath')), (15, 'sonic', dict(size=.5, note='vapour cone snaps round Goku')), (12, 'whoosh', dict(size=.6, dur=8, note='towers falling past, sunlit')),
              (20, 'impact', dict(size=1., note='axe kick lands on crossed forearms, contact held clean 20-21')), (22, 'impact_cut', dict(size=.9, note='impact drawing, one frame')), (23, 'cloud_hole', dict(size=1., note='far below the deck opens in a ring')), (24, 'aura', dict(size=.9, kind='gold', note='flare, he goes past the lens on 28-29')), (25, 'sonic', dict(size=.6)), (28, 'whoosh', dict(size=.9, note='past the lens')),
              (30, 'wind', dict(size=.3, dur=10, note='thin air: everything loses its low end, 30-39')), (33, 'sonic', dict(size=.35, note='last vapour cone, far')), (36, 'sonic', dict(size=.2)), (39, 'cut_silence', dict(note='no outside sound from 40 until g5')),
              (40, 'inside', dict(kind='breath', size=.5, note='his own breath, close')), (48, 'inside', dict(kind='heart', size=.4, note='first meeting: no sound but one heartbeat')), (58, 'inside', dict(kind='heart', size=.45)), (64, 'inside', dict(kind='breath', size=.4)), (68, 'inside', dict(kind='heart', size=.5)), (76, 'inside', dict(kind='breath', size=.25, note='held breath into g3'))]

    def setup(s):
        cv2.setNumThreads(2); s.C = Climb(); s.G = Fig('goku', 'ssj', seed=3); s.Z = Fig('frieza', seed=5); s.x, s.y = grid(); s.frost = [cv2.GaussianBlur(fbm(H // 2, W // 2, (40, 15, 6), (1, .6, .4), 60 + k), (0, 0), .6) for k in range(2)]
        # B: the point of the violet cloud the shock goes through (found once by looking)
        S, c = s.camB(12); z = s.C.world(c, 12)[1]; q = (1130, 400); d = float(np.median(z[q[1] - 60:q[1] + 60, q[0] - 60:q[0] + 60])); d = d if d < 5000 else 420.
        r = c.R.T @ np.array([(q[0] - c.cx) / c.f, (q[1] - c.cy) / c.f, 1.]); s.PB = np.asarray(c.eye) + r * d; s.dB = d
        S, c = s.camC(22); r = c.R.T @ np.array([(560 - c.cx) / c.f, (700 - c.cy) / c.f, 1.]); t = (905 - c.eye[2]) / r[2]; s.PC = np.asarray(c.eye) + r * t

    # ------------------------------------------------------------------ cameras
    def camA(s, f):
        S = Stage(alt(f), AZ['A']); u = f / 9.; return S, S.cam((1.05 - .3 * u, -1.75 - .15 * u, -2.05 - .35 * u), (.30, .12, 1.02 + .35 * u), 1010 - 60 * u, math.radians(7 * math.sin(f * .55) - 4))

    def camB(s, f):
        S = Stage(alt(f), AZ['B']); u = (f - 10) / 9.; k = math.exp(-(f - 10) / 2.) if f >= 10 else 0.; return S, S.cam((-.35 - .5 * u, -3.5 - .7 * u, -.55 - .1 * u), (.12 + .10 * u, 0, -.02 + .36 * u), 1120., math.radians(-5 + 9 * u + 1.5 * k * (1 if f % 2 else -1)))

    def camC(s, f):
        S = Stage(alt(f), AZ['C']); u = (f - 20) / 9.; return S, S.cam((.80 - .4 * u, -2.75, 2.05 + .25 * u), (.10, .10, .30 + .30 * u), 1160 - 110 * u, math.radians(6 - 13 * u))

    def camD(s, f):
        S = Stage(alt(f), AZ['D']); u = (f - 30) / 9.; return S, S.cam((-6 + 8 * u, -50, -22 + 11 * u), (0, 0, 0 + 4 * u), 1000., math.radians(-9 + 12 * u))

    # ------------------------------------------------------------------ helpers
    def face(s, F):
        h = F.info.get('head') if F.info else None
        return None if not h else (h['x'], h['y'], max(18., h['s'] * .9))

    def fig(s, im, F, c, f, kind, power, up=(0, 1), **kw):
        """figure with a thin flight flame: a close coat that nearly vanishes round face and fists, sharp tongues streaming off the trailing side; light on the edges only"""
        L = F.draw(c, f, **kw); h = s.face(F); keep = ([h] if h else []) + [(float(v[0]), float(v[1]), 46. * F.info['px'] / 520.) for v in F.info.get('fists', {}).values()]
        A = ki.aura(L, f, kind, power, face=h, up=up, ground=False, height=.72, width=.62, veil=.10, front=.18, lat=.75, keep=keep, sharp=1.25)
        if A.back is None: return over(im, L), L
        return A.over(ki.put(A.under(im), ki.lit(L, kind, min(1., power) * .55, f, face=h, core=.22, tint=.0))), L

    def fz(s, im, c, f, arcs=5, glow=.30, **kw):
        """Frieza: violet glow, hard rims, a few arcs (no flame coat)"""
        L = s.Z.draw(c, f, **kw); q = cv2.resize(L[..., 3], (W // 4, H // 4), interpolation=cv2.INTER_AREA); g = cv2.resize(.6 * cv2.GaussianBlur(q, (0, 0), 7) + .5 * cv2.GaussianBlur(q, (0, 0), 20), (W, H), interpolation=cv2.INTER_LINEAR) * (1 - L[..., 3]); im = im + (rgb('#a45cff') - im * .55) * (g * glow)[..., None]
        if arcs:
            b, fr, li = ki.arcs(L, f, 'violet', n=arcs, face=s.face(s.Z)); im = ki.put(over(ki.put(im, b), L), fr); im = im + li * .7 if li is not None else im
        else: im = over(im, L)
        return im, L

    def vp(s, c, v=(0, 0, 1)):
        (p,), (z,) = c.project((np.asarray(c.eye) + np.asarray(v, float) * 1e5)[None]); return (float(np.clip(p[0], -3000, 5000)), float(np.clip(p[1], -5000, 5000))) if z > 0 else (c.cx, -2500.)

    def streaks(s, im, f, p, n=46, seed=0, col='#ffffff', k=1., tip=True):
        """vapour streaks rushing along the flight direction (drawn, tapered, away from p = vanishing point)"""
        r = np.random.default_rng(seed * 100 + f); cel = Cel(W, H, ss=1)
        for _ in range(n):
            a, b = r.uniform(0, W), r.uniform(0, H); d = np.array([a - p[0], b - p[1]]); L = np.hypot(*d) + 1; d /= L; l = r.uniform(60, 330) * k; q = abs(a - W / 2) / (W / 2)
            if r.random() > .25 + .75 * q: continue
            cel.line(np.array([[a, b], [a, b] + d * l * .5, [a, b] + d * l]), r.uniform(1.2, 4.2) * (.5 + q), col, taper=(.5, .5), seed=int(r.integers(1 << 20)), tremor=0, alpha=float(r.uniform(.25, .75)))
        return over(im, cel.layer())

    # ------------------------------------------------------------------ 0-9
    def A(s, f):
        # 0-6 is still g1's night: navy through the hole, the lip gold from his own light; 7-9 altitude brings the blue
        S, c = s.camA(f); g = twos(f); k = 200. + 3. * f; q = float(smooth(5.5, 9.6, f)); tp = [rgb(a) + (rgb(b) - rgb(a)) * q for a, b in zip(('#060a20', '#101e4a', '#26427c'), ('#2a5fb4', '#5f9ce2', '#cfe4f8'))]
        h = alt(f); im, z = s.C.ring.frame(c, f, k, gold=.84 - .34 * q, skin='green', above=.75 * q, top=tp, st=1.1 * (1 - q)); p = s.vp(c); im = kit.zoomblur(im, p[0], p[1], .012 + .012 * min(1., f / 7.), 17)
        if f >= 7: im = burst_white(im, f, (.14, .30, .44)[f - 7], '#fffaf0', '#ffe2a4', p)
        im = s.streaks(im, f, p, 60, 1, '#fff2c4' if f < 7 else '#fff8e8', 1.2)
        dz = lk(g, [(0, 2.35), (4, 2.0), (6, 1.65), (8, 1.25)]); zp = seq(g, [(0, 'hit_up', 'out'), (4, 'hang_back', 'out'), (8, 'hover_guard', 'hold')]); gp = seq(g, [(0, 'rise', 'hold'), (4, 'rise', 'in'), (6, 'punch_wind', 'out'), (8, 'uppercut', 'hold')])
        S.put(s.Z, zp, (.62 + .42 * f / 9., .20, dz + .18 * f / 9.), math.pi * .72, -.15, .10); S.put(s.G, gp, (0, 0, 0), -.45, -.12 if g < 6 else -.30, -.05); s.G.look(S.P((.4, .8, dz)))
        n = f < 8; zr_ = [dict(dir=(-.55, -.75, -.25), col='#ffd35a', thr=.46, amt=.9), dict(dir=(.1, .9, -.3), col='#8fa8e8' if n else '#eef4ff', thr=.5, amt=.8)]
        im, _ = s.fz(im, c, f, rim=zr_, pal=FZ_NIGHT if n else 'dawn', light=(-.45, -.35, .55) if n else (.15, .9, .30))
        im, L = s.fig(im, s.G, c, f, 'gold', ki.envelope(f, on=-10, flares=[(0, .5), (6, .4)]), rim=dict(dir=(.15, .9, -.35), col='#ffe28a' if n else '#ffe9b0', thr=.66, amt=.85), expr=dict(expr='shout'), pal=GI_NIGHT if n else 'dawn', light=(.15, .9, .30))
        if f >= 8: im = over(im, s.G.smear(c, dict(s.G.state(f), pose=POSES['punch_wind']), dict(s.G.state(f), pose=POSES['uppercut']), 'armR', seed=f))
        sx, sy = kit.shake(f, 7, 3); return comp.cam2d(im, W / 2 + sx, H / 2 + sy, 1.012)

    # ------------------------------------------------------------------ 10-19
    def B(s, f):
        S, c = s.camB(f); g = twos(f); a = f - 10; hole = 0. if a < 2 else s.dB * .235 * (1 - math.exp(-(a - 1.4) / 2.0)); im, z = s.C.world(c, f, punch=[(s.PB, hole)]); p = s.vp(c)
        if hole > 0:                                                              # light pours through, vapour ring rolls out of it
            (hp,), (hz_,) = c.project(s.PB[None]); R = c.f * hole / hz_; d = np.hypot(s.x - hp[0], s.y - hp[1]); im = im + rgb('#fff1cf') * (np.exp(-((d - R) / (R * .5)) ** 2) * .16 + np.exp(-(d / (R * 1.1)) ** 2) * .10)[..., None]
            L = ki.ring(hp[0], hp[1], R * (1.10 + .10 * a), 16 + 3 * a, f, 'base', asp=.96, seed=4, gaps=.45, alpha=max(0., .85 - .09 * a)); im = ki.put(im, L)
        im = kit.smear(im, 0, 8 + 1.5 * a, 15); im = s.streaks(im, f, (p[0], -4000), 34, 2, '#fff6e4', 1.1)
        # poses: contact held two drawings, then he is thrown up and away and Goku goes after him
        if a < 2: zp, gp, zv, gv, zr, gr = 'hit_gut', 'uppercut', (.36, 0, .36), (-.48, 0, -.36), .12, -.04
        else:
            t = (g - 12) / 7.; zp = seq(g, [(12, 'hit_gut', 'out'), (14, 'thrown', 'out'), (18, 'hang_back', 'hold')]); gp = seq(g, [(12, 'uppercut', 'out'), (14, 'fly', 'out'), (16, 'fly_punch', 'hold')]); zv = (.42 + .34 * t, .18 * t, .55 + .72 * t); gv = (-.50 + .12 * t, 0, -.42 - .05 * t); zr = .25 + .55 * t; gr = -.10 - .75 * min(1, t * 2)
        S.put(s.Z, zp, zv, math.pi / 2, zr, 0.); S.put(s.G, gp, gv, -math.pi / 2, gr, 0.); s.G.look(S.P((zv[0], zv[1], zv[2] + .3)))
        if a < 2: hit(s.G, 'R', s.Z, 'stomach', push=.05)
        sun = dict(pal='dawn', light=(-.5, .55, .45)); zk = dict(rim=[dict(dir=(.8, .4, -.3), col='#fff0cc', thr=.45, amt=.8), dict(dir=(-.8, -.4, -.2), col='#ffd35a', thr=.55, amt=.8)], **sun); gk = dict(rim=dict(dir=(.7, .5, -.35), col='#fff3c4', thr=.42, amt=.9), expr=dict(expr='shout'), **sun)
        im, LZ = s.fz(im, c, f, **zk)
        if 2 <= a < 6: im = ki.put(im, s.tails(LZ, (-.45, 1.), f))
        if 5 <= a: im = over(im, self_cone(s.G, f, c))
        im, LG = s.fig(im, s.G, c, f, 'gold', ki.envelope(f, on=-10, flares=[(10, .7), (14, .5)]), up=(-.30, 1), **gk)
        if a == 0: im = over(im, s.G.smear(c, dict(s.G.state(f), pose=POSES['punch_wind']), s.G.state(f), 'armR', seed=f))
        if a < 7:
            x0, y0 = [float(v) for v in c.project(S.P((.16 + .02 * a, 0, .52 - .05 * a))[None])[0][0]]; q = a - 2
            if q >= 0: im = ki.put(im, ki.ring(x0, y0, 60 + 170 * (q + .4) ** .75, 40 - 6 * q, f, 'base', asp=.46, rot=-.9, seed=9, gaps=.35, alpha=.95 - .15 * q)); im = im + rgb('#fff3c8') * (np.exp(-(np.hypot(s.x - x0, s.y - y0) / (70 + 60 * q)) ** 2) * (1., .55, .28, .14, .06)[q] * .8)[..., None]
            else: im = im + rgb('#fff3c8') * (np.exp(-(np.hypot(s.x - x0, s.y - y0) / 26) ** 2) * .5)[..., None]
        if a < 6: dx, dy = kit.shake(f, (20, 6, 22, 12, 6, 3)[a], 7); im = comp.cam2d(im, W / 2 + dx, H / 2 + dy, 1.03)
        return im

    def tails(s, L, d, f, n=9):
        """dry-brush speed tails on the far side of a thrown figure"""
        a = L[..., 3]; ys, xs = np.nonzero(a > .6)
        if len(xs) < 50: return None
        r = np.random.default_rng(f * 7 + 1); d = np.array(d) / np.hypot(*d); cv = Canvas(H, W, layer=True); k = r.integers(0, len(xs), n * 3); P = np.stack([np.stack([xs[k] + d[0] * u * r.uniform(60, 190, len(k)), ys[k] + d[1] * u * r.uniform(60, 190, len(k))], -1) for u in np.linspace(0, 1, 5)], 1); col = L[ys[k], xs[k], :3] / np.maximum(L[ys[k], xs[k], 3:4], .1)
        cv.strokes(P, r.uniform(2, 7, len(k)), col, op=.50, dry=.85, seed=f, shape='taper'); Q = cv.layer(); Q *= (1 - a[..., None]); return Q

    # ------------------------------------------------------------------ 20-29
    def Cc(s, f):
        S, c = s.camC(f); g = twos(f); a = f - 20; k = 0. if a < 3 else 520 * (1 - math.exp(-(a - 2.2) / 2.0)); im, z = s.C.world(c, f, punch=[(s.PC, k, .60)], holes=[(0, 0, 168.)], sky_alt=3200 + 160 * a); p = s.vp(c, (0, 0, -1))
        if a >= 3:
            (hp,), (hz_,) = c.project(s.PC[None]); R = c.f * k / hz_; im = ki.put(im, ki.ring(hp[0], hp[1], R * (1.15 + .09 * a), 20 + 2 * a, f, 'base', asp=.60, rot=0., seed=6, gaps=.4, alpha=max(0., .9 - .08 * a)))
        im = kit.zoomblur(im, p[0], p[1], .006, 9); im = s.streaks(im, f, p, 40, 3, '#fff6e4', 1.1)
        # 20-21 the heel lies on the crossed forearms and nothing else happens; 22 the drawing; then they part
        sink = (0., .035, .06, .06)[min(a, 3)]
        if a < 4: zp, gp, zv, gv, zr = P_axe(), 'block_cross', (.30, .02, .98 - sink), (0, 0, -.42 - sink), .30
        else:
            t = (g - 22) / 7.; zp = seq(g, [(22, P_axe(), 'out'), (24, 'air_kick', 'out'), (26, 'hang_back', 'out'), (28, 'hover_guard', 'hold')]); gp = seq(g, [(22, 'block_cross', 'out'), (24, 'power_in', 'out'), (26, 'rise', 'hold')]); zv = (.36 + .42 * t, .22 + .55 * t, 1.02 + .16 * t); gv = {24: (-.04, -.02, -.52), 26: (-.14, -.25, -.22), 28: (-.10, -1.30, .72)}[g]; zr = .3 - .5 * t
        S.put(s.G, gp, gv, -math.pi * .78, -.45 if a < 4 else -.30, 0.); s.G.look(S.P((zv[0], zv[1] - .3, zv[2] + .4)) if a < 4 else S.P((gv[0] + .45, gv[1] - 2.4, gv[2] + .95))); S.put(s.Z, zp, zv, math.pi * .25, zr, 0.)
        sk = s.G.skel(); P0 = (sk.wr['L'] + sk.wr['R']) / 2 + rz(np.array([.15, -.03, .045]), S.az)
        if a < 4: s.Z.reach(fR=dict(p=P0 + np.array([0, 0, .03])))
        sun = dict(pal='dawn', light=(.45, .55, .5)); zk = dict(rim=[dict(dir=(.8, .4, -.3), col='#fff0cc', thr=.45, amt=.8), dict(dir=(-.4, -.8, -.2), col='#ffd35a', thr=.6, amt=.7)], expr=dict(expr='rage'), **sun); gk = dict(rim=dict(dir=(.7, .5, -.35), col='#fff3c4', thr=.5, amt=.85), expr=dict(expr='grit' if a < 4 else 'shout'), **sun)
        (x0, y0), = c.project(P0[None])[0]
        if 2 <= a < 8: q = a - 2; im = im + rgb('#fff3e8') * (np.exp(-(np.hypot(s.x - x0, s.y - y0) / (110 + 60 * q)) ** 2) * (.9, .6, .35, .2, .1, .05)[q])[..., None]          # light behind the figures, never over the block
        if a >= 5: im = over(im, self_cone(s.G, f, c))
        if 3 <= a < 8: q = a - 3; im = ki.put(im, ki.ring(x0, y0, 90 + 170 * q ** .75, 30 - 4 * q, f, 'violet' if a % 2 else 'gold', asp=.62, rot=.1, seed=12, gaps=.25, alpha=.9 - .14 * q))
        im, LG = s.fig(im, s.G, c, f, 'gold', (.55, .55, 1.2, 1.0)[a] if a < 4 else ki.envelope(f, on=-10, flares=[(24, .9)]), up=(0, 1), **gk)
        im, LZ = s.fz(im, c, f, arcs=5 if a > 2 else 0, **zk)
        if a < 2: im = im + rgb('#ffffff') * (np.exp(-(np.hypot(s.x - x0, s.y - y0) / 15) ** 2) * .55)[..., None]
        if a == 2: im = ki.impact(im, 2, int(x0), int(y0), subject=ki.stack(LG, LZ), accent='#ffb01e')
        if a == 0: im = comp.cam2d(im, W / 2 + (x0 - W / 2) * .02, H / 2 + (y0 - H / 2) * .02, 1.02)
        if 2 < a < 8: dx, dy = kit.shake(f, (0, 0, 0, 26, 17, 10, 6, 3)[a], 9); im = comp.cam2d(im, W / 2 + dx, H / 2 + dy, 1.03)
        return im

    # ------------------------------------------------------------------ 30-39
    def helix(s, q, sd):
        """where light sd is at (possibly past or fractional) frame q of cut D, local metres about the fight"""
        a = q * .80 + (0 if sd > 0 else math.pi) + .4; r = 8.2 + 2.4 * math.sin(q * .5); return np.array([math.cos(a) * r, math.sin(a) * r * .8, 1.6 * sd])

    def D(s, f):
        S, c = s.camD(f); a = f - 30; u = a / 9.; im, z = s.C.world(c, f, st=.75 * smooth(.35, 1., u), holes=[(0, 0, 168.)], sky_alt=4600 + 17000 * u ** 1.1)
        L = Canvas(H, W, layer=True); lights = []
        for sd, kd in ((-1, 'violet'), (1, 'gold')):                               # vapour left hanging: the helix of the last 26 frames, falling away below
            t = np.linspace(0, 26, 90); X = np.array([s.helix(f - q, sd) + [0, 0, -q * 2.6 - .05 * q * q] for q in t]); Pw = np.array([S.P(v) for v in X]); p, zc = c.project(Pw); wd = np.clip(c.f * (.22 + .10 * t) / zc, 1.2, 60); n1 = np.interp(t, np.linspace(0, 26, 40), np.random.default_rng(sd + 5).normal(0, 1, 40))
            nrm = np.c_[np.gradient(p[:, 1]), -np.gradient(p[:, 0])]; nrm /= np.hypot(*nrm.T)[:, None] + 1e-6; p2 = p + nrm * (n1 * wd * .25)[:, None]
            for i in range(0, len(t) - 10, 4): L.strokes(p2[None, i:i + 11], wd[i + 5] * .60, rgb('#fbfcff') if sd > 0 else rgb('#e6defa'), op=float(.58 * (1 - t[i] / 27.)), dry=float(.04 + .5 * t[i] / 26), seed=i + 3 * sd, shape='round')
            lights.append((sd, kd, p[:14], zc[0], p[0]))
        im = over(im, L.layer())
        for sd, kd, tr, zc, p in sorted(lights, key=lambda q: -q[3]):
            F = ki.trail(tr[::-1], kd, 7.5 * 74 / zc, t=f / 24.); im = ki.put(im, ki.trail_cel(F, kd)); im = ki.put(im, ki.head(p[0], p[1], 8.5 * 74 / zc, kd, f, 3 + sd)); im = im + rgb('#ffe08a' if sd > 0 else '#b48aff') * (np.exp(-(np.hypot(s.x - p[0], s.y - p[1]) / 90) ** 2) * .35)[..., None]
            if (f in (33, 34, 35) and sd > 0) or (f in (36, 37, 38) and sd < 0): k = f - (33 if sd > 0 else 36); im = ki.put(im, ki.cone((p[0], p[1] - 14), -math.pi / 2, 120 + 60 * k, .62 + .10 * k, f, 'base', alpha=.50 - .15 * k))
        return im

    # ------------------------------------------------------------------ 40-79
    def lights(s, f):
        """screen offsets (before roll) of the two lights about the centre; they touch at 48, 58, 68"""
        K = [(40, 330), (48, 0), (53, -300), (58, 0), (63, 340), (68, 0), (72, -190), (76, -262), (90, -268)]; i = max(0, min(len(K) - 2, int(np.searchsorted([k for k, _ in K], f, 'right')) - 1)); (a, va), (b, vb) = K[i], K[i + 1]; t = float(np.clip((f - a) / (b - a), 0, 1)); t = 1 - (1 - t) ** 2.2 if va == 0 else (t ** 2.2 if vb == 0 else ease(t)); r = va + (vb - va) * t
        an = .35 + .115 * min(f, 74) - .0 + .25 * math.sin(f * .21); e = np.array([math.cos(an) * r, math.sin(an) * r * .62])
        if f > 68:                                                                 # fly out of the last meeting to where they will hang: gold left, violet right
            ro = s.roll(f); ca, sa = math.cos(ro), math.sin(ro); es = np.array([-262., 78.]); u = min(1., (f - 68) / 8.); e = np.array([es @ [ca, sa], es @ [-sa, ca]]) * (1 - (1 - u) ** 2.4)
        return np.stack([e, -e])

    def roll(s, f): t = np.clip((f - 40) / 36., 0, 1); t = t + .0; e = t * t * (3 - 2 * t) * .72 + t * .28; return math.radians(-13 + 158 * e)

    def camE(s, f):
        t = float(np.clip((f - 40) / 36., 0, 1)); e = ease(t); return O.orbit_cam(alt=120 + 760 * t ** 1.3, pitch=-10. + 1.5 * e + 9. * smooth(.62, 1., t), yaw=30 - 30 * e, roll=0, focal=900 - 60 * e, pos=(-1., -4. - 5. * e), cx=960, cy=486), s.roll(f)

    def E(s, f):
        c, ro = s.camE(f); ca, sa = math.cos(ro), math.sin(ro); r0, d0 = c.R[0].copy(), c.R[1].copy(); c.R = np.stack([r0 * ca + d0 * sa, -r0 * sa + d0 * ca, c.R[2]]); im = O.earth(c, f, O.sun_dir(-4.5, 78)); im = O.aurora(im, c, f, .75 + .35 * smooth(40, 60, f))
        down = np.array([-sa, ca]); rt = np.array([ca, sa]); cen = np.array([960., 486.]); toS = lambda v: cen + rt * v[0] + down * v[1]; rs = lambda q: s.roll(q)
        # rings of frozen vapour, left behind and falling slowly away toward the planet
        for j, m in enumerate(MEET):
            a = f - m
            if a < 0: continue
            q0 = np.array([0., 5.2 * a + .10 * a * a]); P = toS(q0); sc = 1 / (1 + .020 * a); R = (30 + 215 * (1 - math.exp(-a / 4.2)) + 7.2 * a) * sc; rot = (.5, -.7, 1.1)[j] + ro; asp = (.34, .46, .28)[j] + .004 * a; al = float(np.clip(1.05 - .020 * a, 0, 1)) * (1 if a > 0 else .6)
            im = s.frozen(im, f, P, R, asp, rot, a, al, j)
        # trails: the recent path carried down by the climb
        pos = s.lights(f)
        for k, kd in ((1, 'violet'), (0, 'gold')):
            n = int(min(13, max(4, f - 40 + 4))); fade = 1 - smooth(72, 77, f); q = np.array([s.lights(f - i * .5)[k] + np.array([0, 3.4 * i * .5 + .55 * (i * .5) ** 2]) for i in range(n * 2)]); X = np.array([toS(v) for v in q])
            if fade > .02 and f - 40 > 1: im = ki.put(im, ki.trail_cel(ki.trail(X[::-1], kd, 9., t=f / 24.), kd), float(fade))
            p = X[0]; hr = 8.5 - 3.2 * smooth(70, 77, f); im = ki.put(im, ki.head(p[0], p[1], hr, kd, twos(f) if f < 76 else 76, 5 + k)); im = im + rgb('#ffdc8a' if kd == 'gold' else '#b184ff') * (np.exp(-(np.hypot(s.x - p[0], s.y - p[1]) / 64) ** 2) * (.42 - .14 * smooth(70, 77, f)))[..., None]
        for j, m in enumerate(MEET):
            a = f - m
            if 0 <= a <= 9:
                L, li = ki.clash_star(960, 486, 62 - 4 * j, a, ('gold', 'violet'), seed=3 + j, rot=ro * .5 + j); im = ki.put(im, L)
                if li is not None: im = im + li * .8
                if a < 3: im = im + rgb('#f4f1ff') * (np.exp(-np.hypot(s.x - 960, s.y - 486) / (260 - 60 * a)) * (.65, .38, .16)[a])[..., None]
        if f >= 48:                                                              # meetings shake nothing: only the picture breathes
            pass
        return im

    def frozen(s, im, f, P, R, asp, rot, a, al, j):
        """a ring of ice: pale crest with torn gaps, then a thousand slow crystals that flash in the sun"""
        thick = (5 + 10 * math.exp(-a / 3.) + .35 * a); L = ki.ring(P[0], P[1], R, thick, f, 'base', asp=asp, rot=rot, seed=20 + j, gaps=.30 + .012 * a, alpha=al * math.exp(-a / 5.), cols=('#4c66b8', '#a9c4ff', '#f4f8ff'), lw=0., spurs=0); im = ki.put(im, L)
        r = np.random.default_rng(300 + j); n = 900; th = r.uniform(0, 2 * math.pi, n); dr = r.normal(0, 1, n) * (.012 + .0040 * a) * R + r.exponential(1., n) * (.015 + .006 * a) * R * r.choice([-1., 1.], n, p=[.62, .38]); e1 = np.array([math.cos(rot), math.sin(rot)]); e2 = np.array([-e1[1], e1[0]]); q = (R + dr)[:, None]; X = P + (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2 * asp) * q; ph = r.uniform(0, 6.28, n); sp = r.uniform(.25, .9, n); tw = np.maximum(0, np.sin(f * sp + ph)) ** 6; A = (.30 + 1.25 * tw) * al * r.uniform(.25, 1., n) ** 1.4 * (1 if a > 0 else .4); sz = np.where(tw > .4, 1.15, .80) * r.uniform(.75, 1.7, n); col = np.where((np.cos(th) * 1.0)[:, None] > .2, rgb('#fff2d8'), rgb('#cfe0ff')).astype(f32)
        B, _ = ki.ellipse_pts(P[0], P[1], R, R * asp, rot, 160, wob=.03, seed=20 + j); m = np.zeros((H // 2, W // 2), f32); cv2.polylines(m, [np.round(B / 2 * 8).astype(np.int32)], True, 1., max(1, int(thick * .5)), cv2.LINE_AA, 3); g = cv2.GaussianBlur(m, (0, 0), 2.5 + .22 * a) * (1 + .5 * s.frost[j % 2]); im = im + rgb('#7fa6ff') * (cv2.resize(np.maximum(g, 0), (W, H), interpolation=cv2.INTER_LINEAR) * al * (.16 + .5 * math.exp(-a / 7.)))[..., None]
        im = np.ascontiguousarray(im).astype(f32); O._splat_stars(im, X[:, 0].astype(f32), X[:, 1].astype(f32), A.astype(f32), sz.astype(f32), col); return im

    def frame(s, f):
        im = s.A(f) if f < 10 else s.B(f) if f < 20 else s.Cc(f) if f < 30 else s.D(f) if f < 40 else s.E(f)
        return np.maximum(np.ascontiguousarray(im), 0).astype(f32)


GI_NIGHT = dict(base='gold', mul='#ffd9ae', shmul='#9c5c3e', litmul='#ffeecb')      # g1's night: he is his own lamp, the orange stays orange
FZ_NIGHT = dict(base='storm', mul='#b4bcd6', shmul='#464f78', litmul='#e6ecff', tint=('#4a5690', .22))


_AXE = None
def P_axe():
    """the kick landing: heel down on the guard"""
    global _AXE
    if _AXE is None: _AXE = dict(POSES['axe_up'], fR=(.03, .40, .30), kR=(.25, .7, .6), tR=(-20, -6, 0), fL=(.16, -.36, .42), kL=(.55, 1, .25), tL=(20, -40, -10), pel=(-6, -8, 16), wst=(14, 0, 0), cst=(16, 4, -6))
    return _AXE


def self_cone(F, f, c):
    """vapour cone standing on the head of a figure that goes up (from the skeleton, so it never depends on an earlier draw)"""
    hd = F.skel().head; p, z = c.project(np.asarray(hd, float)[None]); r = float(c.scale(np.asarray(hd, float))) * .13; L = ki.cone((p[0, 0] + 6, p[0, 1] - r * 1.5), -math.pi / 2, 300 + 22 * (f % 10), .56, f, 'base', alpha=.26); return L if L is not None else np.zeros((H, W, 4), f32)
