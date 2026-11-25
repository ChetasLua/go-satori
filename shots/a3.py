"""a3 THE DIVE.  0 he looks over the edge and grins; 20 the Nimbus tips and dives at the lens; 40 his eyes: three decks, three walls of light; 56 out over a bright sea, pull up, skim, his hand cuts the water at 64."""
import math, numpy as np, cv2
from db import W, H, comp, kit, clouds, sky
from db.comp import over, blur, ease, bokeh
from db.paint import rgb, smooth, fbm, ramp, grid, f32
from db.ink import Cel, spl
from db.nimbus import Nimbus
from db.skim import Skim
from chars.kid import Kid, P, mixp, Chain, unit
from chars.kid_head import EXPR, mixx

T = clouds.tones
def kf(f, K, e=ease):
    """eased keys of scalars or arrays"""
    if f <= K[0][0]: return np.asarray(K[0][1], float)
    for (a, va), (b, vb) in zip(K[:-1], K[1:]):
        if f < b: t = float(e((f - a) / float(b - a))); return np.asarray(va, float) * (1 - t) + np.asarray(vb, float) * t
    return np.asarray(K[-1][1], float)
lin = lambda t: t
def over_l(a, b): return a * (1 - b[..., 3:4]) + b
def two(f): return f - f % 2


class Shot:
    id = 'a3'; n = 80
    finish = dict(bloom=.28, vig=.14)
    EVENTS = [(0, 'ambience', dict(note='close on his face: wind in hair and cloth, warm')), (4, 'blink', dict(size=.1)), (8, 'grin', dict(size=.3, note='corners go up; on the beat 10 it is a full grin')), (10, 'accent', dict(size=.4, note='grin lands')), (14, 'cloth', dict(size=.25, note='he leans, gi and belt flap')), (17, 'drop', dict(size=.4, note='weight goes over the edge')),
              (20, 'cut', dict(note='from below: edge of the cloud')), (23, 'puff', dict(size=.5, note='Nimbus tips over the lip, a lump of vapour breaks off')), (26, 'dive', dict(size=.8, note='falling whistle / jet of air rises 26-39, comes straight at us')), (30, 'voice', dict(size=.6, note='kid whoop begins, doppler rising')), (39, 'pass', dict(size=1., note='passes the lens: gold fills the frame, air slap')),
              (40, 'cut', dict(note='his eyes; wind roar')), (41, 'deck', dict(size=.7, note='wall of light 1, gold')), (46, 'deck', dict(size=.85, note='wall of light 2, cream')), (52, 'deck', dict(size=1., note='wall of light 3, white')),
              (56, 'burst', dict(size=1., note='out of the cloud, sudden open bright air, gulls')), (61, 'pullup', dict(size=.7, note='hard pull-up, Nimbus squashes on the air over the water')), (63, 'squash', dict(size=.5)), (64, 'splash', dict(size=.9, note='hand cuts the water; crown; then a steady hiss of wake to the end')), (66, 'voice', dict(size=.6, note='belly laugh 66-80')), (70, 'spray', dict(size=.4)), (75, 'spray', dict(size=.35, note='near drops pass the lens'))]

    def setup(s):
        cv2.setNumThreads(1); s.N = Nimbus(3); s.K = Kid(); s.xx, s.yy = grid(); s.SK = Skim(296, 800, 34.)
        s.bgA = s.plateA(); s.sky1, s.lip, s.far = s.plateB(); s.sky3 = s.plateD(); s.DV = sky.Dive()

    # ----------------------------------------------------------------- plates
    def plateA(s):
        """the sky opposite the sun, lit full face: what he is in front of"""
        y = s.yy / H; im = ramp(y, [(0, '#6f8fd6'), (.28, '#a9a8da'), (.52, '#e9b9c2'), (.72, '#ffd2ac'), (1, '#f7b7a0')]); S = clouds.Sky(H, W, im, light=(.55, -.45, .62), seed=3)
        for (x, b, w, h, sd, t) in ((300, 800, 1000, 300, 4, 0), (1500, 820, 1100, 380, 9, 0), (900, 880, 1500, 220, 12, 1), (100, 1010, 1100, 300, 21, 2), (1750, 1040, 1200, 360, 23, 2), (960, 1120, 1700, 260, 31, 2)):
            S.bank(x, b, w, h, (T('#ffe9cc', '#fbcbb0', '#c9a3c0', '#a08cc0'), T('#fff0cf', '#ffd0a6', '#cf9fb0', '#9d86b4'), T('#fff4d2', '#ffcf94', '#cc8f98', '#8f77a6'))[t], '#e8b9bc', seed=sd, detail=.9, levels=2)
        return bokeh(S.img.astype(f32), 9, 0)

    def plateB(s):
        """under the lip of the cloud sea, looking up: sky with the sun beyond the edge; the overhang; far deck below"""
        x, y = s.xx, s.yy; d = np.hypot(x - 1010, (y - 190) * 1.1); im = ramp(np.clip(d / 1250, 0, 1), [(0, '#fff6dc'), (.14, '#ffe2a0'), (.34, '#fbc07a'), (.6, '#f0a08a'), (.85, '#b98cae'), (1, '#8d80b4')])
        U = clouds.Sky(H + 500, W + 400, '#000000', light=(.05, .80, .30), seed=6, layered=True); Tu = T('#e0a094', '#a9789c', '#7f68a2', '#5f5592')
        for (cx, b, w, h, sd) in ((500, 1010, 1200, 300, 51), (1750, 990, 1300, 340, 52), (1160, 1040, 1500, 250, 53), (80, 960, 800, 360, 54), (2260, 960, 800, 380, 55), (1130, 1080, 900, 180, 56)): U.bank(cx, b, w, h, Tu, '#8a6f9c', seed=sd, levels=3, detail=1., foot=.05, pool=0)
        und = U.layer()[::-1].copy()
        S = clouds.Sky(H + 500, W + 400, '#000000', light=(.05, -.85, -.25), seed=5, layered=True); Tl = T('#fff2c4', '#f7b98a', '#b98098', '#74629a')
        for (cx, b, w, h, sd) in ((420, 640, 1300, 300, 3), (1800, 660, 1400, 340, 6), (1150, 590, 1000, 190, 12), (100, 760, 900, 300, 15), (2250, 780, 900, 340, 18), (1160, 700, 2400, 200, 22)): S.bank(cx, b, w, h, Tl, '#8a6f9c', seed=sd, levels=3, detail=1., foot=.55, pool=0)
        top = S.layer(); und = cv2.warpAffine(und, np.float32([[1, 0, 0], [0, 1, 150]]), (W + 400, H + 500)); lip = (und * (1 - top[..., 3:4]) + top).astype(np.float16)
        F = clouds.Sky(H, W, im, light=(.1, -.8, .2), seed=8)
        for (cx, b, w, h, sd) in ((250, 1080, 900, 300, 41), (1650, 1100, 1000, 360, 43), (960, 1160, 1300, 240, 47)): F.bank(cx, b, w, h, T('#ffe9c0', '#f6bd9a', '#bd8aa4', '#8577aa'), '#b58fb0', seed=sd, levels=2)
        return im.astype(f32), lip, F.img.astype(f32)

    def plateD(s):
        hz = 296; im = ramp(s.yy / hz, [(0, '#3f93de'), (.45, '#7cc0ee'), (.80, '#cfeaf2'), (1., '#fff2d2')]); S = clouds.Sky(H, W, im, light=(.62, -.60, .30), seed=2); Tt = T('#fffdf4', '#fff0dc', '#c9d6ea', '#9db6da')
        for (x, b, w, h, sd) in ((250, 300, 900, 250, 4), (1480, 300, 760, 190, 9), (900, 300, 620, 90, 12), (1860, 300, 460, 120, 15)): S.bank(x, b, w, h, Tt, '#dcecf2', seed=sd, detail=1., levels=3)
        return S.img.astype(f32)

    # ----------------------------------------------------------------- 0-19 the grin
    def grin(s, f):
        g = two(f); t = f / 20.; img = comp.cam2d(s.bgA, W / 2 - 26 + 60 * t, H / 2 + 10 - 24 * t, 1.06 + .03 * t)
        fall = float(kf(f, [(13, 0), (16, -.12), (20, 1.)])); lean = float(kf(g, [(0, .25), (6, .30), (9, .0), (10, -.06), (12, 0.), (14, -.10), (19, .85)]))
        look = float(kf(g, [(0, 1.), (6, 1.), (9, .0)])); gr = float(np.clip((g - 6) / 4., 0, 1))
        ex = mixx('curious', 'grin', gr); ex['look'] = (.35 * look + .12, -.55 * look); ex['open'] = .12 if f in (4, 5) else (.6 if f in (3, 6) else ex['open']); ex['tilt'] = ex['tilt'] + .10 * float(kf(g, [(8, 0), (10, 1.2), (13, .8)])); ex['mw'] = ex['mw'] * (1 + .10 * float(kf(g, [(9, 0), (10, 1), (13, .2)])))
        pose = P('sit', wst=(0, -.14 - .22 * lean, .03), cst=(.04, -.08 - .20 * lean, .05 * lean), hed=(-.06 + .10 * (1 - look), .10 - .30 * look + .30 * max(0, lean), .07 + .05 * gr - .06 * lean), armL=(.26, -.74, .40), armR=(.20, -.74, .46), elbL=(1, .1, -.6))
        w = .62 + .16 * math.sin(f * .42) + .10 * math.sin(f * .93 + 1); cx, cy, S = 900 + 60 * lean + 70 * fall, 1190 + 34 * lean + 170 * fall ** 2 + 5 * math.sin(f * .35), 204 * (1 + .05 * fall)
        n, fr = s.N.draw(cx, cy + 10, S, f, yaw=.25, pitch=.16, roll=.02 + .10 * fall, pal='dawn', light=(.55, .45, .55), speed=.4, lost=0, D=8.)
        img = over(img, s.N.vapour(cx, cy - 2.2 * S, S * .62, f + 40, vel=(-26, -3), n=7, life=22, pal='dawn', op=.75, size=.8))
        img = over(img, n)
        L = s.K.draw(cx, cy, S, pose, .52, -.06, .02 + .05 * lean, expr=ex, pal='dawn', light=(.55, .42, .62), f=f, wind=(-1, -.16, w), tail=dict(dir=(.6, .1, -1), curl=1.1, side=.6, amp=.3), rim=dict(dx=-3.2, dy=2.2, col='#fff0c8', amt=.75, soft=.6), D=8.)
        img = kit.aura(img, L, 14, '#ffcf8a', .16); img = over(img, L); img = over(img, fr)
        if 10 <= f <= 13:                                                     # one small star on a tooth, on the beat
            q = [1., .85, .5, .25][f - 10]; gx, gy = s.K.H.P(np.array([(.26, -.585)]))[0]; cel = Cel(W, H, ss=2); th = .25 + .12 * (f - 10)
            for k in range(4): a_ = th + k * math.pi / 2; d = np.array([math.cos(a_), math.sin(a_)]); nn = np.array([-d[1], d[0]]); cel.fill(np.array([gx, gy]) + np.array([nn * 5 * q, d * (62 if k % 2 == 0 else 40) * q, -nn * 5 * q]), '#ffffff')
            st = cel.layer(); img = img + blur(st[..., 3], 7)[..., None] * rgb('#fff2cc') * .55 * q; img = over(img, st)
        if f >= 17: img = kit.smear(img, 0, 22 * (f - 16), 9)
        return img

    # ----------------------------------------------------------------- 20-39 over the edge, at the lens
    def dive(s, f):
        t = f - 20; g = f if f >= 26 else two(f)
        up = float(kf(f, [(20, 0), (25, 6), (39, 1050)], lambda u: u ** 1.9)); z = 1 + .0007 * up
        sh = kit.shake(f, np.interp(f, [26, 39], [0, 7]), 11); img = comp.cam2d(s.far, W / 2 + sh[0], H / 2 - .12 * up + 40 + sh[1], 1.08)
        lp = s.lip.astype(f32); zl = 1 + .00045 * up; M = np.float32([[zl, 0, -200 * zl + (1 - zl) * 0 + sh[0] * 1.6 - (zl - 1) * 960], [0, zl, -300 - up * 1.0 + sh[1] * 1.6]]); lay = cv2.warpAffine(lp, M, (W, H), flags=cv2.INTER_LINEAR)
        # size by true perspective: he falls at the camera
        S = float(np.interp(f, [20, 23, 25, 27, 29, 31, 33, 35, 36, 37, 38, 39], [25, 25.5, 27, 31, 39, 52, 74, 118, 165, 245, 390, 700])); D = float(np.clip(1500. / S, 2.7, 40))
        px = float(np.interp(f, [20, 23, 26, 30, 34, 37, 39], [1034, 1026, 1012, 992, 972, 962, 960])); py = float(np.interp(f, [20, 22, 24, 26, 28, 30, 32, 34], [196, 190, 208, 262, 330, 392, 440, 470]))
        tip = float(kf(g, [(20, 0), (22, -.10), (27, 1.)])); cork = float(kf(f, [(26, 0), (31, .55), (35, -.30), (39, .10)]))
        img = over(img, lay)
        nim = dict(yaw=-math.pi / 2 + .67 * (1 - tip) + .40 * tip, pitch=.12 + .16 * tip, roll=.10 * tip + cork * .8, D=D, pal='dawn', light=(.15, .75, -.10 + .35 * tip), speed=.2 + 1.1 * tip, stretch=1 + .34 * tip, squash=1 - .12 * tip, tail=1 + 2.3 * tip, bend=(1.0 * tip, 0), tone=(82, 54, 18))
        # cloud crumbs of the broken lip
        cel = Cel(W, H, ss=1); r = np.random.default_rng(4)
        kp = -1.50 * tip - .06 * (1 - tip); ky = .55 * (1 - tip) + .06
        a = mixp('sit', 'lean', min(1, tip * 1.8)) if tip < .55 else mixp('lean', 'dive', min(1., (tip - .55) / .45)); pose = dict(a); pose['tail'] = dict(dir=(.2, .3, -1), curl=.9, side=.6 + .3 * math.sin(f * .7), amp=.5)
        if f >= 26: k = float(np.clip((f - 33) / 4., 0, 1)); pose = mixp(pose, P('dive', armR=(.20, .92, .62), armL=(.62, .30, .62), curlR=1., curlL=.8), k)
        if f >= 33:                                                           # aim the face, not the hips, at the lens
            s.K.build(px, py, S, pose, ky, kp, .02 + cork, None, D); hx, hy, hs = s.K.info['head']; w_ = float(np.interp(f, [33, 36], [0, 1])); tx, ty = 960, float(np.interp(f, [33, 36, 38], [400, 500, 560])); px += (tx - hx) * w_; py += (ty - hy) * w_
        n, fr = s.N.draw(px, py + .18 * S * tip, S, f, front_z=.6, **nim)
        if f >= 23: img = over(img, s.crumbs(f, lay is not None))
        img = over(img, n)
        ex = mixx('grin', 'whoop', float(np.clip((g - 26) / 4., 0, 1))); ex['look'] = (0., -.1)
        L = s.K.draw(px, py - .10 * S * (1 - tip), S, pose, ky, kp, .02 + cork, expr=ex, pal='dawn', light=(.25, .70, .35), f=f, wind=(.05 * math.sin(f), -1, .55 * tip + .15), D=D, rim=dict(dx=0., dy=max(1., min(6., S * .02)), col='#fff3d0', amt=.95, soft=.5))
        img = over(img, L)
        if tip < .6: img = over(img, fr)
        if f >= 27:                                                           # air
            k = float(np.clip((f - 27) / 8., 0, 1)); cel = Cel(W, H, ss=1); kit.focus_lines(cel, px, py - 40, 260 + 420 * k, 1500, int(26 + 34 * k), 6 + 8 * k, '#fff3dc', seed=f // 2, ry=.85); img = over(img, cel.layer() * (.30 + .35 * k))
        if f >= 34: img = kit.zoomblur(img, 960, 500, .020 + .016 * (f - 34), 17)
        if f >= 38: img = sky.burst_white(img, f, .56 if f == 38 else .66, '#fff3cc', '#ffd27a', (960, 520))
        return img

    def crumbs(s, f, _):
        """small lumps of cloud knocked off the lip: they lag, spin and thin"""
        cel = Cel(W, H, ss=2); r = np.random.default_rng(19)
        for k in range(15):
            a = f - 23 - r.uniform(0, 3)
            if not 0 < a < 17: continue
            x = 1010 + r.normal(0, 70) + r.normal(0, 5) * a; y = 240 + 2.0 * a + .62 * a * a * r.uniform(.4, 1.2); R = (6 + 15 * r.random()) * (1 + .14 * a) * (1 - a / 18.) ** .5; th = np.linspace(0, 2 * math.pi, 24)[:-1]; q = R * (1 + .18 * np.sin(th * 3 + k + a * .2) + .10 * np.sin(th * 5 - k)); Pq = np.c_[x + np.cos(th) * q, y + np.sin(th) * q * .8]; cel.fill(Pq, '#f0b184', .95); cel.fill(Pq * .72 + np.array([x, y - R * .25]) * .28, '#fff0c4', .95)
        return cel.layer()

    # ----------------------------------------------------------------- 40-55 his eyes
    def pov(s, f):
        t = f - 40; im, z = s.DV.frame(t); sh = kit.shake(f, 5, 3)
        S = 360; cx, cy = 960 + sh[0] * 2 + 34 * math.sin(f * .45), 700 + sh[1] * 2 + 16 * math.sin(f * .62) - 46 * math.exp(-t / 2.5); n, fr = s.N.draw(cx, cy, S, f, yaw=math.pi / 2 + .05 * math.sin(f * .4), pitch=.30, roll=.06 * math.sin(f * .45), D=9., xmin=.55, pal='dawn' if t < 7 else 'day', light=(.35, .65, .3), speed=1.4, stretch=1.1, tail=1., lost=0, curls=True, tone=(80, 48, 14))
        im = over(im, n)
        cel = Cel(W, H, ss=2); r = np.random.default_rng(2)                  # the tips of his own fringe whip at the top of the frame
        for k, (x0, w, ln, d) in enumerate(((330, 70, 190, 1), (640, 92, 250, 1), (1010, 64, 150, -1), (1380, 88, 230, -1), (1640, 60, 150, -1))):
            sw = 46 * math.sin(f * 1.05 + k * 1.7) + 24 * math.sin(f * 2.1 + k); L = ln * (.62 + .22 * math.sin(f * .9 + k * 2.2)); tip = np.array([x0 + d * 110 + sw, -40 + L]); B = kit.spike((x0 - w, -60), tip, (x0 + w, -60), .10, -.12 * d + .08 * math.sin(f * .8 + k)); cel.fill(B, '#221822'); cel.fill(np.vstack([B[:len(B) // 2], (B[:len(B) // 2] * .7 + tip * .3)[::-1]]), '#120c14')
        im = over(im, blur(cel.layer(), 5.5))
        return im

    # ----------------------------------------------------------------- 56-79 the sea
    def skim(s, f):
        t = f - 56; hz = float(kf(f, [(56, -420), (58, -250), (63, 322), (66, 286), (70, 296)])); K = float(kf(f, [(56, 4200), (64, 640)], lambda u: 1 - (1 - u) ** 2.2)); age = f - 64
        SK = s.SK; SK.hz = hz; SK.K = K; SK.yc = 800.; SK.dX = SK.V * (K / (800 - hz)) / 500.
        dy = hz - 296; sk = cv2.warpAffine(s.sky3, np.float32([[1, 0, 0], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REPLICATE) if abs(dy) > .5 else s.sky3
        im = SK.sea(f, sk); im = SK.streak(im, f, .55 + .6 * max(0, 1 - t / 8.))
        bx, by = kf(f, [(56, (640, -260)), (58, (560, 60)), (60, (560, 330)), (62, (640, 545)), (63, (690, 592)), (64, (728, 596)), (66, (772, 560)), (68, (784, 538)), (71, (780, 552)), (75, (790, 545)), (80, (800, 549))], lin if f < 62 else ease)
        by += 3.2 * math.sin(f * .52) * (f > 66); S = float(kf(f, [(56, 58), (60, 76), (63, 90), (66, 88)])); roll = float(kf(f, [(56, 1.05), (59, .95), (62, .25), (64, -.16), (67, .13), (71, .06), (80, .09)])); st = float(kf(f, [(56, 1.55), (60, 1.5), (63, .96), (66, 1.2), (70, 1.14)])); sq = float(kf(f, [(56, .9), (60, .95), (63, .46), (66, .76), (69, .63), (72, .67)])); tl = float(kf(f, [(56, 3.2), (62, 3.0), (66, 2.2), (72, 1.7)]))
        g = two(f) if f > 66 else f; p1 = P('skim', armR=(.50, -.26, .95), elbR=(.6, .5, -.4), palmR=(-.5, .2, 1), armL=(.10, .70, .46), elbL=(1, .2, .2), hed=(-.50, .52, -.05), wst=(.06, .24, .16), cst=(.10, .24, .20)); p0 = P('lie', hed=(-.2, .80, 0), armR=(.34, .30, .70), armL=(.30, .40, .66)); k = float(kf(f, [(59, 0), (62, .45), (63, 1.06), (65, 1.)])); pose = mixp(p0, p1, k)
        br = math.sin(f * .85) * (f >= 66); pose['cst'] = tuple(np.array(pose['cst']) + [0, .035 * br, 0]); pose['hed'] = tuple(np.array(pose['hed']) + [0, .05 * br + float(kf(f, [(63, 0), (66, -.16), (70, .0)])), .03 * math.sin(f * .4)]); wv = float(np.clip((f - 64) / 4., 0, 1)); w0 = 2 * math.pi * (f - 64) / 10.; pose['armL'] = tuple(np.array(pose['armL']) + np.array([.10 + .34 * math.sin(w0), .10 - .12 * abs(math.cos(w0)), .10 * math.cos(w0)]) * wv); pose['elbL'] = (1 + .3 * wv * math.sin(w0 - .9), .2 - .5 * wv, .2 + .5 * wv * math.cos(w0 - .9)); pose['palmL'] = tuple(np.array([-1, 0, .2]) * (1 - wv) + np.array([.55 * math.sin(w0 - 1.3), -.1, 1.]) * wv); pose['curlL'] = .6 + (.10 + .16 * (.5 + .5 * math.sin(w0 - 1.9)) - .6) * wv; pose['sprL'] = .15 + .30 * wv
        pose['tail'] = dict(dir=(-.15, -1.1, -.45), curl=1.25, side=.7, amp=.55, length=2.9, speed=2 * math.pi / 12., lag=.55)
        ex = dict(mixx('whoop', 'laugh', float(np.clip((f - 62) / 3., 0, 1)))); ex['look'] = tuple(kf(f, [(62, (.0, -.5)), (67, (-.10, -.6)), (70, (.10, -.05))])); ex['mo'] = ex['mo'] * (1 + .12 * br)
        if f in (74, 75): ex['joy'] = 1.
        nk = dict(yaw=.14, pitch=.30, roll=roll); n, fr = s.N.draw(bx, by, S, f, pal='day', light=(.5, .7, .4), speed=.9, stretch=st, tail=tl, squash=sq, front_z=1.35, bend=(float(kf(f, [(56, 0), (62, 1.0), (66, .35), (72, 0)])), 0), **nk)
        L = s.K.draw(bx + 26 * S / 88, by + 92 * S / 88 * (sq / .66) ** .35, S, pose, 1.04, -1.20, .24 + (roll - .08) * .85, expr=ex, pal='sky', light=(.5, .7, .4), f=f, wind=(-1, -.10 - .5 * max(0, roll), float(kf(f, [(56, .9), (64, .75), (70, .55)])) + .06 * math.sin(f * .7)), D=5.8, rim=dict(dx=-2.2, dy=2.2, col='#fff6dc', amt=.8, soft=.5))
        hx, hy = s.K.info['handR']; hxy = (np.array([907.9, 746.3]) if False else np.array([hx, hy])); cx, cy = hxy[0] + 18, hxy[1] + 12
        if age >= 0:                                                        # once it is in, the waterline is where the hand is
            wl = float(kf(f, [(64, 0), (66, 1)])); cy = cy * 1.; SK.yc = cy; wet = smooth(cy + 2, cy + 12, s.yy) * smooth(hx - 44, hx - 10, s.xx); rider = over_l(n, L) * (1 - wet[..., None])
        else: rider = over_l(n, L); SK.yc = max(cy + 40, hz + 60)
        am = float(np.clip((f - 60) / 4., 0, 1)); yw = cy + 2 + max(0, -age) * 26
        rows = np.nonzero((n[..., 3] > .5).any(1))[0]; yb = float(rows[-1]) - 8 if len(rows) else yw; air = max(0., yw - cy - 2)
        im = SK.reflect(im, L, yw + 2, f, amt=.52 * am, squash=.74); im = SK.reflect(im, n, min(yb + air, yw + 6), f, amt=.62 * am, squash=.70)      # each thing mirrored at its own waterline, so it stays attached
        if age >= 0:
            kick = 1 + 1.0 * math.exp(-age / 3.6); sc = 1.28 + .42 * math.exp(-age / 4.)
            im = SK.graze(im, bx - 400, bx + 230, cy - 26, f, amt=min(1, age / 4.)); im = SK.wake(im, cx, cy, f, age=age); im = SK.crown(im, cx, cy, f, age, 1.9); im = SK.spray(im, cx, cy - 6, f, front=False, n=26, age=age + .5, kick=kick, scale=sc)
        im = over(im, s.N.vapour(bx, by, S, f, vel=(-18, -2 - 14 * max(0, roll)), n=10, life=26, pal='day', speed=.9, tail=1.5, size=1.0, yaw=.16, pitch=.30, roll=roll)); im = over(im, rider)
        if age >= 0: im = SK.bow(im, cx + 10, cy + 10, f, hh=26 * min(1., .4 + age / 4.)); im = SK.spray(im, cx - 10, cy + 12, f, front=True, n=16, age=age + .5, kick=kick, scale=sc); im = SK.near(im, cx - 30, cy, f) if age > 6 else im
        v = math.hypot(*(kf(f + .5, [(56, (640, -260)), (58, (560, 60)), (60, (560, 330)), (62, (640, 545))], lin) - kf(f - .5, [(56, (640, -260)), (58, (560, 60)), (60, (560, 330)), (62, (640, 545))], lin))) if f < 62 else 0
        if v > 20: im = kit.smear(im, 0, v * .22, 7)
        w = float(np.interp(f, [56, 57, 58, 59, 60], [.66, .57, .44, .22, 0]))
        if w > 0: im = sky.burst_white(im, f, w, '#ffffff', '#dceaf4', (960, 300))
        return im

    def frame(s, f): return (s.grin(f) if f < 20 else s.dive(f) if f < 40 else s.pov(f) if f < 56 else s.skim(f)).astype(f32)
