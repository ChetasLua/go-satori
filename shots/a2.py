"""a2 ALONE ON A CLOUD.  The ball was the sun: the sun never leaves the match circle's centre; one slow crane back finds one small boy on one small cloud.  Hold."""
import math, numpy as np, cv2
from db import W, H, ball, kit, comp
from db.comp import over, blur, ease
from db.paint import rgb, smooth, fbm, grid, f32
from db.sky_dawn import Dawn
from db.nimbus import Nimbus
from db.ink import Cel, spl
from chars.kid import Kid, P, mixp

BOY = (74.4, 520., -80.4); M = 7.64                           # world place of the boy (sky_dawn metres) and metres per kid unit


def cam_at(f):
    """zoom 2 (match circle) -> 1, the camera sinks back and up; never stops completely"""
    t = np.clip((f - 2) / 46., 0, 1); e = t * t * t * (t * (6 * t - 15) + 10); c = min(1., max(0., (f - 40) / 40.)); e2 = e * (1 - .012) + .012 * c
    return dict(zoom=2 - e2, dx=150 - 40 * (1 - e) , dy=150 * (1 - e) - 10 * c, dz=52 * (1 - e) + 3. * c)


def lift(f):
    """0 ball in his lap .. 1 against the sun; small dip before, overshoot, slow way back"""
    K = [(0, 0.), (45, 0.), (48, -.09), (59, 1.06), (62, 1.), (67, 1.0), (79, .06), (80, .0)]
    for (a, va), (b, vb) in zip(K[:-1], K[1:]):
        if f < b: return va + (vb - va) * float(ease((f - a) / (b - a)))
    return 0.


class Shot:
    id = 'a2'; n = 80
    finish = dict(bloom=.32, vig=.16)
    EVENTS = [(0, 'ambience', dict(note='huge open dawn: high thin wind, almost nothing. Sound perspective opens with the crane, 2-48')), (0, 'note', dict(note='match cut from the ball to the sun (960,486) r208')),
              (10, 'tail', dict(size=.1, note='tail sways on every beat: 10 20 30 ... soft cloth-like brush, barely there from 30')), (27, 'enter', dict(size=.2, note='boy on the cloud comes into frame from below; soft puff of the Nimbus')),
              (48, 'cloth', dict(size=.15, note='he draws the ball up: cloth rustle')), (59, 'glint', dict(size=.35, note='ball reaches the sun: one tiny glass ting as light passes through it')), (67, 'cloth', dict(size=.1, note='arm comes slowly down until 79')),
              (34, 'birds', dict(size=.15, note='far birds, right, three small calls, 34 47 63'))]

    def setup(s):
        cv2.setNumThreads(1); s.D = Dawn('a2'); s.N = Nimbus(3); s.K = Kid(); s.xx, s.yy = grid(); s.nz = cv2.resize(fbm(H // 2, W // 2, (60, 22), (1, .5), 5, 3.2), (W, H))

    def birds(s, im, f, c):
        cel = Cel(W, H, ss=3); r = np.random.default_rng(8)
        for k in range(7):
            (x, y), sc = s.D.at(c, (1150 + 170 * k + f * 3.4 + 40 * math.sin(k * 2.), 5200 + 260 * r.random(), 120 + 60 * math.sin(k * 1.7) + 5 * math.sin(f * .11 + k))); S = 15. * c.zoom * (.8 + .4 * r.random()) * .62; w = math.sin(f * .55 + k * 1.9) * (1 if math.sin(f * .07 + k) > -.4 else .15); L = np.array([(-1, -.10 - .55 * w), (-.5, -.28 - .20 * w), (0, 0), (.5, -.28 - .20 * w), (1, -.10 - .55 * w)]) * S
            cel.line(spl(np.array([x, y]) + L, n=6), max(.9, S * .13), '#8d5a78', taper=(.6, .6), press=.1, seed=k, tremor=0, wfun=lambda t: .4 + np.sin(np.pi * t), alpha=.8)
        return over(im, cel.layer())

    def frame(s, f):
        q = cam_at(f); c = s.D.cam(**q); im, z, front = s.D.frame(f, c, split=max(300., 820 - q['dy'])); xx, yy = s.xx, s.yy
        im = s.birds(im, f, c)
        bob = math.sin(f * 2 * math.pi / 40.) * 1.6; (bx, by), ms = s.D.at(c, (BOY[0], BOY[1], BOY[2] + bob * .45)); S = ms * M
        if by - 8 * S > H + 10: return over(im, front) if front is not None else im
        k = S / 22.
        # the cloud's shadow, long toward us, and a veil of lit air behind the boy
        m = np.zeros((H, W), f32)                                            # the cloud's shadow on the deck: a cloud's shape, soft, long toward us
        for (ox, oy, rx, ry) in ((-70, 150, 120, 26), (-128, 172, 104, 30), (-26, 166, 92, 24), (-170, 200, 128, 30), (-98, 206, 120, 34), (-214, 236, 100, 26), (-150, 244, 120, 28)): m = np.maximum(m, np.exp(-(((xx - bx - ox * k) / (rx * k)) ** 2 + ((yy - by - oy * k) / (ry * k)) ** 2)))
        m = smooth(.26, .80, m + .16 * s.nz * m) * .30; m = cv2.GaussianBlur(m.astype(f32), (0, 0), max(2., 5 * k)); im = im * (1 - m[..., None]) + rgb('#5d5690') * m[..., None]
        if front is not None: im = over(im, front)
        hz = np.exp(-(((xx - bx - 10 * k) / (210 * k)) ** 2 + ((yy - by + 40 * k) / (120 * k)) ** 2)) * .34; im = im + (rgb('#efbfa6') - im) * hz[..., None]
        R = dict(yaw=.10, pitch=.20, roll=.03 + .012 * math.sin(f * 2 * math.pi / 40. + 1))
        im = over(im, s.N.vapour(bx, by, S, f, vel=(-1.5 * k, -.10), n=13, life=110, pal='dawn', yaw=.10, pitch=.20, speed=.15, op=.80, size=1.25))
        b, fr = s.N.draw(bx, by, S, f, pal='dawn', light=(.42, .46, -.30), speed=.12, tone=(86, 60, 22), **R)
        im = im + (rgb('#ffd27a') * .10) * blur(b[..., 3], 16)[..., None]; im = over(im, b)
        g = f - f % 2; L_ = lift(f); up = P('sit_ball', armR=(.10, .80, .58), armL=(.20, -.60, .46), hed=(-.10, .40, 0)); lo = P('sit_hold', armR=(-.12, -.34, .64), armL=(-.12, -.34, .64), hed=(0, -.02 + .10 * float(ease((f - 38) / 8.)), 0)); pose = mixp(lo, up, float(np.clip(L_, -.1, 1.1)))
        tl = dict(dir=(.45, .0, -1), curl=1.05, side=.45, amp=.26, speed=2 * math.pi / 20., phase=-.5)
        L = s.K.draw(bx, by, S, pose, 1.62, -.10, 0, expr='calm', pal='dawn', light=(.50, .45, -.62), f=g, rim=dict(dx=-2.0 * min(k, 1.6), dy=1.4 * min(k, 1.6), col='#fff0bd', amt=1., soft=.4), wind=(-1, -.1, .30 + .10 * math.sin(f * .16)), tail=tl)
        im = kit.aura(im, L, 5, '#ffcf80', .22); im = over(im, L); hx, hy = s.K.info['handR']
        e = float(np.clip(L_, 0, 1)); br = 9.5 * k
        gl = math.exp(-((f - 60) / 3.2) ** 2) if f > 48 else 0.
        ball_at = lambda im: ball.draw(im, hx + 2 * k, hy - 9 * k * e - 3 * k * (1 - e), br, light=(.3, .3, -.8), glass=.9, body=.9, line=.8)
        if e > .25: im = over(im, fr); im = ball_at(im)
        else: im = ball_at(im); im = over(im, fr)
        if gl > .02: im = kit.flare(im, hx + 2 * k, hy - 9 * k * e, 34 * k, '#ffe2a0', .26 * gl, streak=.5, ang=0.)
        return im
