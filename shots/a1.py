"""a1 THE BALL.  Black.  Inside dark amber glass four stars ignite on the four bells (10 20 30 40); 42 light floods in and the camera pulls back: a glass ball in a child's two hands, his magnified eye behind it (blink 62);
66 he lifts it; 78 the sun stands exactly behind it, on the match circle."""
import math, numpy as np, cv2
from db import W, H, comp, kit, ball, sky
from db import hands as HD
from db.comp import over, blur, ease, bokeh
from db.paint import rgb, smooth, grid, f32, fbm
from db.ink import Cel, spl
from db.sky_dawn import Dawn
from shots.a2 import cam_at
from chars.kid import Kid, P
from chars import kid_hands as KH
from chars.friends_hands import hold_ball
from chars.kid_head import KHead, draw_head, EXPR, PAL

BELL = (10, 20, 30, 40)
def kf(f, K, e=ease):
    if f <= K[0][0]: return np.asarray(K[0][1], float)
    for (a, va), (b, vb) in zip(K[:-1], K[1:]):
        if f < b: t = float(e((f - a) / float(b - a))); return np.asarray(va, float) * (1 - t) + np.asarray(vb, float) * t
    return np.asarray(K[-1][1], float)
out = lambda t: 1 - (1 - t) ** 3
S_ = ball.STAR[:, :2] * 1.3
R1 = 404.                                                    # ball radius while he looks through it
KIDH = HD.build(len=.84, wid=1.16, boxy=.22, tip=.15, knuckle=.78, palm=1.06, wrist=1.0, meat=1.22)


class Shot:
    id = 'a1'; n = 80
    finish = dict(bloom=.34, vig=.22)
    EVENTS = [(0, 'silence', dict(note='black; a breath of room tone inside glass')), (10, 'bell', dict(size=.8, star=1, note='first star ignites: bell 1, glint runs through the glass to 28')), (20, 'bell', dict(size=.85, star=2, note='bell 2')), (30, 'bell', dict(size=.9, star=3, note='bell 3')), (40, 'bell', dict(size=1., star=4, note='bell 4')),
              (42, 'light', dict(size=.8, note='light floods the glass, camera pulls back 42-54, air opens')), (52, 'touch', dict(size=.15, note='small fingers on glass')), (56, 'hum', dict(size=.1, note='his eye behind the glass, curious')), (62, 'blink', dict(size=.25, note='the huge eye blinks: soft wet tick')), (66, 'cloth', dict(size=.2, note='he lifts the ball to the sky, 66-76')), (78, 'align', dict(size=.8, note='sun exactly behind the ball: glass sings, cut to the sun'))]

    def setup(s):
        cv2.setNumThreads(1); s.xx, s.yy = grid(); s.D = Dawn('a2'); s.black = np.zeros((H, W, 3), f32) + rgb('#020103'); s.hd = s.hands(); s.K = Kid()

    # ------------------------------------------------------------------------------------ plates
    def hands(s): return kit.tint(hold_ball(960, 486, R1, 'gold')[:H], '#e4a074', .78)                  # lane friends: small hands cupped round the glass, occlusion solved

    def face(s, f, dy):
        """the boy behind the ball, backlit, amber on his face; -> layer"""
        op = {59: .85, 60: .62, 61: .22, 62: 0., 63: .16, 64: .62, 65: .92}.get(f, 1.06); lk = tuple(kf(f, [(50, (0, 0)), (53, (.16, .06)), (55, (.16, .06)), (57, (-.16, -.08)), (59, (-.16, -.08)), (61, (0, 0)), (66, (0, .05)), (69, (0, .9))])); ex = dict(EXPR['wonder']); ex.update(open=op, look=lk, pupil=1.12, brow=.16 + .06 * math.sin(f * .2))
        sc = 266.; pal = dict(PAL['dawn'], skin='#cf7e50', skin_sh='#a05548', skin_dk='#7d3d42', white='#fff6e2', white_sh='#e0b08c', iris='#140a0c', iris2="#2a100c", hair='#2a1620', hair_hi='#7a3c34', line='#2a0f14', gi='#d9692a', gi_sh='#9a3e2c', blue='#2c3564', blue_sh='#1b2148')
        ex["blush"] = 0.; ex["lid"] = -.02; pose = P('stand', armL=(-.25, -.45, -.55), armR=(-.25, -.45, -.55), elbL=(.6, -.2, -1), elbR=(.6, -.2, -1), hed=(0, .02, .015 * math.sin(f * .12))); S = sc / 1.16; tx, ty = 960 - .375 * sc * 1.07 + 3 * math.sin(f * .21), 486 - .125 * sc + dy + 2 * math.sin(f * .17)
        for _ in range(2): s.K.build(960, 1300, S, pose, 0, 0, 0, None, 12.); hx, hy, hs = s.K.info['head']; S *= sc / hs
        s.K.build(960, 1300, S, pose, 0, 0, 0, None, 12.); hx, hy, hs = s.K.info['head']
        L = s.K.draw(960 + tx - hx, 1300 + ty - hy, S, pose, 0, 0, 0, expr=ex, pal=pal, light=(.12, -.25, .9), f=f, wind=(-.3, -.2, .18 + .05 * math.sin(f * .3)), D=12., tail=None, knot=False)
        return kit.rim(L, 0, 3.2, '#fff0c0', .9, .6)

    # ------------------------------------------------------------------------------------ inside
    def macro(s, f):
        r = float(kf(f, [(0, 2150), (10, 2000), (20, 1840), (30, 1690), (42, 1540)], lambda t: t)); foc = kf(f, [(0, S_[0] * .80), (9, S_[0] * .62), (14, S_[0] * .58), (20, S_[1] * .58), (24, S_[1] * .55), (30, S_[2] * .58), (34, S_[2] * .54), (40, S_[3] * .50), (42, S_[3] * .46)])
        cx, cy = 960 - foc[0] * r, 486 + foc[1] * r; st = ball.ignite(f, BELL); lv = sum(min(q, 1.) for q in st); spin = (-.34 + .0085 * f, .16 - .004 * f, -.05 + .0012 * f)
        fl = float(np.clip((f - 40) / 4., 0, 1))
        im = ball.draw(s.black, cx, cy, r, st, glass=.0, body=.012 + .030 * lv, pulses=ball.pulses(f, BELL, speed=.105, life=22), spec=0., light=(-.5, .7, .4), sun='#000000', env=s.black, spin=spin, stria=.95, glow=1., line=0.)
        dark = np.interp(f, [0, 8, 10], [0., .75, 1.]); im = im * dark
        for i, a in enumerate(BELL):                                          # the ignition itself: one hard flash shaped by the star, two frames
            if 0 <= f - a <= 2:
                p = ball.star_px(cx, cy, r, spin)[i]; im = kit.flare(im, p[0], p[1], [520, 420, 300][f - a], '#ff9a4a', [.55, .34, .14][f - a], streak=.16, ang=.35 + i * .8)
        return im

    # ------------------------------------------------------------------------------------ outside
    def frame(s, f):
        if f < 42: return s.macro(f).astype(f32)
        tilt = float(kf(f, [(42, -430), (66, -430), (75, 16), (78, 0)])); lift = float(kf(f, [(66, 0), (74, 1.), (80, 1.1)], lambda t: t * t * (3 - 2 * t))); dy = 1180 * lift
        r = float(kf(f, [(42, 1540), (54, 418), (66, R1), (76, 208), (80, 208)], lambda t: 1 - (1 - t) ** 2.6)) if f < 66 else float(kf(f, [(66, R1), (67, 408), (76, 208)]))
        p0 = np.array([960., 486.]) + (S_[3] * np.array([-1, 1]) * .46 * 1540) * (1 - float(out(min(1., (f - 42) / 10.)))); cx = p0[0] + 5 * math.sin(f * .19) * (f < 72); cy = p0[1] + 4 * math.sin(f * .23) * (f < 72) + 26 * math.sin(math.pi * np.clip((f - 64) / 10., 0, 1))
        q0 = cam_at(0); bg, _ = s.D.frame(0, s.D.cam(zoom=q0['zoom'], dx=q0['dx'], dy=q0['dy'], dz=q0['dz'], tilt=tilt)); bg = bg.astype(f32)
        flood = float(np.interp(f, [42, 44, 48, 52], [.06, .45, .95, 1.0])); st = tuple(1. + .04 * math.sin(f * .3 + i) + (1.2 * math.exp(-(f - 40) / 2.6) if i == 3 else 0.) for i in range(4))
        spin = tuple(kf(f, [(42, (.017, -.008, 0.)), (66, (.16, .05, .05)), (79, (.0, .0, .0))]))
        fl = s.face(f, dy + (cy - 486)); im = bg
        if lift < .98:
            q = bokeh(np.dstack([fl[..., :3], fl[..., 3]]), 6.5, 0) if f < 74 else fl; im = over(im, q)
        blow = float(np.interp(f, [74, 76, 77, 78, 79], [0, .10, .30, .68, .94]))
        near = fl if lift < .75 else None
        if r < 1150: im = over(im, KH.arms(cx, cy + 260 * (1 - float(np.clip((1150 - r) / 260., 0, 1))) ** 2, r / R1, R1, at=(.82, .60), roll=4, bend=(.40, 4.6), skin=('#b3624e', '#e6a06a', '#8a4242')))
        im = ball.draw(im, cx, cy, r, st, glass=1.04 * flood, body=.14 + .42 * flood + .6 * blow, env=bg, focal=3000., near=near, near_d=.42, star_op=float(np.interp(f, [44, 49, 65, 69], [1., .16, .16, 1.])), light=(.02, .10 + .5 * (1 - lift), -1.), sun='#fff2cc', spin=spin, spec=.75 * flood, stria=.35, pulses=ball.pulses(f, BELL, speed=.105, life=22) if f < 60 else (), glow=.55)
        k = r / R1                                                            # the hands ride on the ball
        if r < 1150:
            g = float(np.clip((1150 - r) / 260., 0, 1)); breathe = 1 + .004 * math.sin(f * .4); M = np.float32([[k * breathe, 0, cx - 960 * k * breathe], [0, k * breathe, cy - 486 * k * breathe + 260 * (1 - g) ** 2]]); hl = cv2.warpAffine(s.hd, M, (W, H), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR); im = over(im, hl)
            # light through thin skin where fingers lie on the glass
            d = np.hypot(s.xx - cx, s.yy - cy) / r; w = hl[..., 3] * smooth(1.10, .80, d) * (1 - smooth(.80, .55, d) * .5); im = im + (rgb('#ff6a3a') * .20) * (cv2.GaussianBlur(w, (0, 0), 9) * hl[..., 3])[..., None]
        if f >= 66:                                                           # the sun climbs in behind the glass
            a = float(np.interp(f, [70, 75, 78, 79], [0, .25, .75, 1.])); d = np.hypot(s.xx - 960, s.yy - 486); ring = np.exp(-np.abs(d - r) / (14 + 40 * a)) * a; im = im + rgb('#fff0c0') * (ring * .55)[..., None]
            if f == 78: im = kit.flare(im, 960, 486, 900, '#fff0c8', .30, streak=.5, ang=0.)
        if blow > 0: d = np.hypot(s.xx - 960, s.yy - 486); m = smooth(r + 1.5, r - 1.5, d) * blow; im = im + (rgb('#fffdf2') * 1.10 - im) * m[..., None]
        if f < 46: im = im * np.interp(f, [42, 44, 46], [.82, 1., 1.])
        return im.astype(f32)
