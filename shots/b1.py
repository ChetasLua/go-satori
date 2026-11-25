"""B1  PARADE.  80 f.  Everyone on the beat but one.  Glances, an upside-down dragon, a sweat drop, a hop: at 60 he is in step."""
import numpy as np, cv2, math
from db import W, H, comp
from db.paint import f32, rgb
from db.ink import spl
from chars.chibi_core import D, N, E, Eo, back, keys, A, ell, tube, star, pi
from chars import chibi_story as ST, chibi_world as WD, chibi_dragon as DR, chibi_stick as SK
from chars.chibi_parade import V, S, ORDER, FINISH

BALLS = [(288, 38), (548, 32), (800, 40), (1040, 34), (1290, 40), (1536, 32), (1842, 38)]
WHO = [w for w, _, _ in ORDER]
CAM = [(0, (1.0, 960, 540)), (24, (1.0, 960, 540)), (43, (1.40, 640, 640), 'e'), (58, (1.46, 610, 648), 'l'), (60, (1.44, 612, 646), 'l'), (63, (1.45, 612, 646), 'l'), (77, (1.0, 960, 540), 'e'), (80, (1.0, 960, 540))]
# who turns to look at him, when
TURN = dict(bulma=(18, -.62, 63), gohan=(22, -.66, 64), goku=(27, -.72, 65), krillin=(32, -.72, 66), roshi=(37, -.60, 67))


def veg(f):
    """his pose and face"""
    if f <= 55: P = ST.march('vegeta', f, S, V, off=-.25, bob=.105, sq=.085, lean=-.13)
    elif f >= 60: P = ST.march('vegeta', f, S, V, off=0., bob=.085, sq=.07, lean=-.06); j = f - 60; k = [.11, .05, -.04, .02, 0][j] if j < 5 else 0.; P['sq'] += k; P['hip'] = (0., P['hip'][1] + k * .5)
    else:
        a, b = ST.march('vegeta', 55, S, V, off=-.25), ST.march('vegeta', 60, S, V, off=0.); u = (f - 55) / 5.; P = dict(b); h = math.sin(pi * u)
        P['fn'] = a['fn'] * (1 - u) + b['fn'] * u + [.10 * h, -.46 * h]; P['ff'] = a['ff'] * (1 - u) + b['ff'] * u + [-.10 * h, -.52 * h]; P['an'] = .5 * h; P['af'] = .7 * h; P['hip'] = (0., a['hip'][1] * (1 - u) + b['hip'][1] * u - .60 * h); P['sq'] = -.12 * h; P['lean'] = -.04; P['lag'] = (-.05, .22 * (1 - 2 * u)); P['sw'] = 0
    ex = dict(mouth='frown', look=(.3, 0.), brow=.46)
    if f < 48: ex.update(blink=1, mouth='frown'); P['pitch'] = -.14; P['hyaw'] = .40
    else:
        hy = keys(f, [(47, .40), (50, -.72, 'o'), (52, -.72), (55, .80, 'o'), (56, .80), (58, .02, 'o'), (59, .02), (61, .46, 'o')]); P['hyaw'] = float(hy); P['pitch'] = float(keys(f, [(47, -.14), (50, .02), (56, .0), (58, -.06), (60, .06), (64, .03)]))
        if f < 56: ex.update(eyes='dot', ek=1.15, brow=.10, bdy=-.07, mouth='flat', look=(-.9 if f < 52 else .9, -.1), red=.2)
        elif f < 60: ex.update(eyes='dot', ek=1.3, brow=-.25, bdy=-.12, mouth='pout', look=(0, -.3), red=1.)
        else: ex.update(eyes='dot', brow=.44, bdy=.03, mouth='frown', look=(.6, .2), red=1. if f < 72 else .6, blink=1 if f in (66, 67) else 0)
    P['ex'] = ex; return P


def other(w, f):
    bob = {}
    P = ST.march(w, f, S, V, **bob); ex = dict(P['ex'])
    if w == 'gohan': P['handN'] = (-.90, -.20 + .05 * math.sin(f * .63)); P['hkN'] = 'fist'; P['armTop'] = True
    if w == 'piccolo':
        if 28 <= f < 64: ex.update(look=(1.0, .25), brow=.02 if f < 60 else .28, mouth='flat'); P['pitch'] = .10
        if f >= 64: ex.update(mouth='smile', blink=1 if 64 <= f < 70 else 0)
    if w in TURN:
        a, y, b = TURN[w]; y0 = P['yaw']
        if a <= f: P['hyaw'] = float(keys(f, [(a, y0), (a + 3, y, 'o'), (b, y), (b + 4, y0, 'b')]))
        if a <= f < a + 2: ex['blink'] = 1
        if a + 2 <= f < b: ex.update(mouth=dict(bulma='o', gohan='o', goku='smile', krillin='o', roshi='open')[w], look=(-.8, .1)); P['roll'] = P['roll'] + (-.06 if w != 'roshi' else -.03)
        if b <= f: ex.update(eyes='happy' if f < b + 12 else ex.get('eyes', 'dot'), mouth='laugh' if w != 'roshi' else 'open')
        if f >= 70 and w in ('goku', 'krillin'): u = Eo((f - 69) / 3.); P['handN'] = (-.30 - .10 * u, .70 - 1.05 * u - .10 * math.sin((f - 70) * .63)); P['hkN'] = 'fist'; P['armTop'] = True
    P['ex'] = ex; return P


def dragon_path(f):
    P = DR.path(f, 1490, 262, amp=80); h = float(keys(f, [(29, 0.), (37, 1., 'e'), (46, 2., 'o'), (60, 2.), (66, 1., 'e'), (75, 0., 'o')]))
    if h <= 0: return P, 0.
    q = P[[0, 10, 22, 36, 52]]; w = 10 * math.sin(f * .33); M = A([(1330, 372), (1342, 284), (1292, 196), (1176, 152), (1040, 176)]) + [0, w]; B = A([(1026, 284), (1126, 272), (1208, 208), (1198, 124), (1088, 92)]) + [w * .5, w * .6]
    c = q * (1 - h) + M * h if h <= 1 else M * (2 - h) + B * (h - 1)
    T = np.vstack([c, P[92::9]]); C = spl(T, n=12); dd = np.r_[0, np.cumsum(np.hypot(*np.diff(C, axis=0).T))]; t = np.linspace(0, dd[-1], 240); return np.c_[np.interp(t, dd, C[:, 0]), np.interp(t, dd, C[:, 1])], min(1., h)


class Shot:
    id = 'b1'; n = 80
    finish = dict(FINISH)
    EVENTS = [(0, 'white_in', dict(dur=5, note='arrives out of the white cloud hole'))] + \
             [(k, 'step', dict(size=.7, pan=0., note='six walkers step together (Roshi, Krillin, Goku, Gohan, Bulma, Piccolo); wooden jetty, sand from 60')) for k in range(0, 80, 10)] + \
             [(k, 'step_off', dict(size=.9, pan=-.4, note='Vegeta alone, half a beat late: hard boot stomp on planks (clarinet)')) for k in range(5, 56, 10)] + \
             [(k, 'ball_bounce', dict(size=.5, pan=.1, note='seven glass balls land together on the beat')) for k in range(0, 80, 10)] + \
             [(k, 'staff_tap', dict(size=.5, pan=.7, note='Roshi\'s staff on the plank')) for k in (10, 30, 50, 70)] + \
             [(a, 'head_turn', dict(size=.2, pan=p, note=w + ' glances back at Vegeta')) for w, a, p in (('Bulma', 18, -.2), ('Gohan', 22, 0.), ('Goku', 27, .2), ('Krillin', 32, .4), ('Roshi', 37, .6))] + \
             [(30, 'dragon_coil', dict(size=.5, dur=16, pan=.4, note='Shenron bends his neck down, upside down, to look')), (46, 'dragon_peek', dict(size=.4, pan=.1, note='stops a hand from Vegeta, blinks')),
              (40, 'sweat_drop', dict(size=.4, pan=-.35, note='drop swells, slides to 56')), (48, 'eye_open', dict(size=.3, pan=-.35)), (50, 'look', dict(size=.3, pan=-.5, note='looks left: Piccolo stares')), (55, 'look', dict(size=.3, pan=-.2, note='looks right: everyone stares')),
              (58, 'gulp', dict(size=.5, pan=-.35)), (56, 'hop', dict(size=.5, dur=4, pan=-.35, note='change-step hop')), (60, 'land_in_step', dict(size=1.0, pan=-.35, note='he lands exactly with everybody: the gag pays off, music in unison')), (64, 'relief', dict(size=.6, note='everyone breathes out, laughs, turns forward')), (70, 'cheer', dict(size=.5, pan=.3, note='Goku and Krillin fists up'))]

    def setup(self):
        self.sky = comp.over(WD.sky(), WD.plate(W, H, lambda c: (c.fill(ell(742, 214, 74, 74), '#fff3bf', .55), c.fill(ell(742, 214, 60, 60), '#fff0a8')), seed=3))
        self.cl = WD.clouds_plate().astype(np.float16); self.sea = WD.sea_plate().astype(np.float16); j, self.oy = WD.jetty_plate(); self.jt = j.astype(np.float16); g, self.fy = WD.fg_plate(); self.fg = g.astype(np.float16)

    # --------------------------------------------- camera
    def cam(self, f):
        z, x, y = keys(f, CAM)
        if 60 <= f < 66: y += [5, -4, 2.5, -1.5, .8, 0][f - 60]
        return float(z), float(x), float(y)

    def put(self, im, pl, ox, oy, c, k):
        """plate with parallax k under the camera"""
        z = 1 + (c[0] - 1) * k; cx = 960 + (c[1] - 960) * k; cy = 540 + (c[2] - 540) * k
        x0, x1 = int(max(0, math.floor(cx - 960 / z + ox) - 2)), int(min(pl.shape[1], math.ceil(cx + 960 / z + ox) + 3)); q = pl[:, x0:x1].astype(f32)
        M = np.float32([[1 / z, 0, cx - 960 / z + ox - x0], [0, 1 / z, cy - 540 / z - oy]])
        l = cv2.warpAffine(q, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT); a = l[..., 3:4]
        return im * (1 - a) + (l if im.shape[2] == 4 else l[..., :3])

    def frame(self, f):
        c = self.cam(f); z, cx, cy = c; X = lambda p: (A(p, float) - [cx, cy]) * z + [960, 540]
        kz = 1 + (z - 1) * .16; kx, ky = 960 + (cx - 960) * .16, 540 + (cy - 540) * .16; im = cv2.warpAffine(self.sky, np.float32([[1 / kz, 0, kx - 960 / kz], [0, 1 / kz, ky - 540 / kz]]), (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101)
        im = self.put(im, self.cl, 60 + f * .4, 0, c, .24); im = self.put(im, self.sea, 250 + f * 1.2, WD.SY, c, .42)
        # Shenron, a little nearer than the island
        kd = .72; zd = 1 + (z - 1) * kd; xd, yd = 960 + (cx - 960) * kd, 540 + (cy - 540) * kd; Xd = lambda p: (A(p, float) - [xd, yd]) * zd + [960, 540]
        P, hk = dragon_path(f); d0 = D(lw=4.6 * zd, line=ST.LINE); PX = Xd(P); ha = math.atan2(*((PX[0] - PX[7])[::-1]))
        if hk > 0 and ha < -1.9: ha += 2 * pi
        DR.shenron(d0, PX, 90 * zd, f, head_ang=ha * (1 + .82 * hk) if hk < 1 else ha * 1.82, mouth=1. if not 46 <= f < 58 else .35, look=-1.2 * hk, hs=1.22 - .35 * hk)
        for i, (x, y, s_) in enumerate(((1046, 74, 46), (1190, 112, 36), (84, 118, 40))): q = Xd([(x + f * 1.6, y + 6 * math.sin(f * .2 + i))])[0]; WD.gull(d0, q[0], q[1], s_ * zd, f, i * 2)
        lay = self.put(np.zeros((H, W, 4), f32), self.jt, 196 + f * V, self.oy, c, 1.)
        d = D(lw=4.8 * z, line=ST.LINE); gy = WD.GY; Sz = S * z; POS = {}
        for w, x, dy in ORDER: p = X([(x, gy + dy)])[0]; ST.shadow(d, p[0], p[1], Sz, .55 if not (w == 'vegeta' and 56 <= f < 60) else .42, '#6b452c', .42)
        for i, (x, yy) in enumerate(BALLS): p = X([(x, gy + yy - 4)])[0]; DR.bounce_ball(d, p[0], p[1], 27 * z, f, 0, i + 1, hop=1.0 + .25 * ((i * 3) % 4) / 3., V=V * z, seed=i)
        # his boot dust, off the beat
        for k in range(5, 56, 10):
            j = f - k
            if 0 <= j < 7:
                r = .16 * Eo((j + 1) / 4.) * (1 - max(0, j - 3) / 4.5); nn = N().at(*X([(424 + 44 - j * V, gy - 6)])[0], 0., Sz)
                for q, (ax, ay) in enumerate(((.18, -.06), (.42, -.12), (-.16, -.04))): SK.puff(d, nn, ax * (1 + j * .22), ay - .02 * j, r * (1 - .18 * q), q, '#f6e2bd', '#dcb98a', .5)
        for w, x, dy in ORDER:
            p = X([(x, gy + dy)])[0]; q = ST.fig(d, w, p[0], p[1], Sz, veg(f) if w == 'vegeta' else other(w, f)); POS[w] = q
            if w == 'vegeta': self.veg_fx(d, q, f, X, gy + dy, Sz)
            if w == 'gohan': kite = q['t']([(-.93, -.36)])[0]
        # landing in step: white ticks round his boots
        if 60 <= f < 64:
            o = X([(424, gy - 2)])[0]
            for a in (-2.75, -2.3, -.85, -.4): e = A([math.cos(a), math.sin(a)]); r0, r1 = (52 + 16 * (f - 60)) * z, (88 + 18 * (f - 60)) * z; d.part(tube([o + e * r0, o + e * r1], (3.6 * z, 1.2 * z)), '#fffdf4', '#fffdf4', w=.28)
            for sg in (-1, 1): SK.puff(d, N().at(o[0] + sg * 50 * z * (1 + .3 * (f - 60)), o[1] - 8 * z, 0., Sz), 0, 0, .20 * (1 - (f - 60) * .18), 5 + sg, '#f6e2bd', '#dcb98a', .5)
        # relief
        for w, a0, sg in (('bulma', 64, 1), ('krillin', 67, 1), ('gohan', 65, 1)):
            j = f - a0
            if 0 <= j < 8: hn = POS[w]['h'].n; u = Eo((j + 1) / 6.); SK.puff(d, hn, 1.12 + .40 * u, .46 - .16 * u, (.10 + .10 * u) * (1 - max(0, j - 4) / 5.), 7, '#ffffff', '#dfeaf2', .42)
        # kite string and sash
        i = 92 + int(np.argmin(np.abs(P[92:, 0] - (716 - 130 * hk + 6 * math.sin(f * .2))))); T, Nn = DR.frames(P); wd = DR.width(i / len(P), 90); B = Nn[i] * (1 if Nn[i][1] < 0 else -1); a, b = P[i] + B * wd * 1.02, P[i] - B * wd * 1.06
        d0.part(tube([Xd(a[None])[0], Xd(((a + b) / 2 + T[i] * 10)[None])[0], Xd(b[None])[0]], (8 * zd, 9 * zd, 8 * zd), cap=.4, cap0=.4), '#e0483c', '#b5322f', w=.8); bx = Xd(b[None])[0]
        for sg in (-1, 1): d0.part(spl([bx, bx + (T[i] * sg * 26 + [0, -10]) * zd, bx + (T[i] * sg * 34 + [0, 10]) * zd, bx + (T[i] * sg * 12 + [0, 12]) * zd], True, n=5, corners=(0,)), '#e0483c', '#b5322f', w=.8)
        m = (bx + kite) / 2 + A([-20 + 8 * math.sin(f * .31), 52]) * zd; d0.stroke(spl([kite, m, bx + [0, 4]], n=16), .55, '#c8392f', taper=(.02, .02), press=.05, tmin=.8)
        if f * V > 40: p = X([(2260 - f * V, 988)])[0]; WD.crab(d, p[0], p[1], 32 * z, f)
        fl = d.layer(grain=.03); lay = lay * (1 - fl[..., 3:4]) + fl
        im = comp.over(im, d0.layer(grain=.025)); yl = int(X([(0, 1012)])[0][1])
        if yl < H - 8: im = WD.reflect(im, lay, yl, f=f, amt=.6, squash=.5)
        im = comp.over(im, lay); im = self.put(im, self.fg, 60 + f * V * 1.5, self.fy, (1 + (z - 1) * 1.0, cx, cy), 1.35)
        if f < 5: k = [1., .86, .58, .30, .10][f]; im = im * (1 - k) + k * rgb('#fffaf0')
        return im

    def veg_fx(self, d, q, f, X, gy, Sz):
        h = q['h']; n = h.n
        if 22 <= f < 57:                                                        # the little red cross pulses on HIS beat
            k = 1 + .28 * max(0., math.cos((f - 5) * pi / 5.)); o = n([(-.36 if h.yaw > 0 else .36, -.48)])[0]; u = Sz * .13 * k
            for i in range(4):
                a = i * pi / 2 + .5; c_, s_ = math.cos(a), math.sin(a); L = A([(.26, 1.05), (.28, .32), (1.05, .26)]); L = np.c_[L[:, 0] * c_ - L[:, 1] * s_, L[:, 0] * s_ + L[:, 1] * c_] * u + o; d.stroke(spl(L, n=7), 1.1, '#d8322e', taper=(.1, .1), tmin=.5)
        if 40 <= f < 64:                                                        # sweat drop: swells, slides, is shaken off by the landing
            j = f - 40; s = (back(j / 4.) if j < 4 else 1.) * .95; y = -.62 + .62 * E((f - 44) / 14.); x = -.98 - .10 * math.sin(max(0., (f - 44) / 14.) * 1.6)
            if f >= 60: x -= .30 * (f - 59); y += -.10 * (f - 59) + .055 * (f - 59) ** 2
            o = n.at(x, y, -.10, s); P = spl(A([(0, -.30), (.11, -.06), (.145, .10), (.09, .22), (0, .26), (-.09, .22), (-.145, .10), (-.11, -.06)]) * 1.25, True, n=7, corners=(0,)); d.part(o(P), '#bfe6fb', '#7fbfe8', w=.75, ink='#2f6f9c', shade=[o(spl([(.0, -.3), (.08, .0), (.02, .2), (.3, .4), (.3, -.3)], True, n=4))]); d.detail(o(ell(-.05, .08, .04, .075)), '#ffffff', sh='#ffffff')
