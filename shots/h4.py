"""h4 EVERYONE ON A CLOUD.  0-13 a2's composition exactly: same sun, same sea, clear morning; 14-44 one smooth push through the real decks to the hopelessly overloaded Nimbus; 45-55 everybody laughs on the beat,
Krillin topples off the rim and Vegeta pulls him back without looking; 56-59 tilt up: the sun lands on the match circle for h5's paper sun."""
import math, numpy as np, cv2
from db import W, H, kit, comp
from db.comp import over, blur, ease
from db.paint import rgb, smooth, fbm, grid, f32
from db.sky_dawn import Dawn
from db.nimbus import Nimbus, E as NE
from db.ink import Cel, spl
from chars import chibi_story as ST
from chars.chibi_core import D
from chars.chibi_dragon import shenron
from shots.a2 import BOY, M, cam_at

BIG, CH = 1.86, .84                                                    # the overloaded cloud against a2's, chibi against cloud
# who, x, z, stacked height, size, yaw : painter's order
CREW = [('piccolo', -2.0, -1.0, 1.95, 1.02, .30), ('goku', .36, -.9, 1.95, 1.06, .04), ('roshi', 1.98, -1.05, 1.95, 1.0, -.05), ('bulma', -1.74, .35, .35, 1.0, .20), ('gohan', -.28, .70, .34, 1.06, .10), ('krillin', 1.36, .85, .16, .95, .06), ('vegeta', 2.72, .40, .02, 1.0, .78)]
SAG = [(-1.6, -.6, .9), (1.2, -.6, .9), (-.4, -.6, .8), (-1.0, .4, .8), (.4, .3, 1.1), (1.5, .6, .8), (2.6, .3, .9)]
PALC = dict(light='#fff3d8', base='#ffd892', shade='#f3ad7c', deep='#c08a98', line='#a35a44', rim='#fffaf0', curl='#c98350', air='#ffd9c2')
PALS = dict(light='#fff3e2', base='#ffdcc6', shade='#d8aebc', deep='#9c93c4', line='#9a7aa0', rim='#fffaf0', curl='#b88aa4', air='#ffd9c2')
D0, D1 = 530., 186.                                                    # camera to the boy's place (m): a2's end, close


def hop(f, ph=0.): return abs(math.sin(math.pi * (f - ph) / 10.))
def beat(f, ph=0.): return math.sin(2 * math.pi * (f - ph) / 10.)
def kf(f, K, e=ease):
    if f <= K[0][0]: return float(K[0][1])
    for (a, va), (b, vb) in zip(K[:-1], K[1:]):
        if f < b: return float(va + (vb - va) * float(e((f - a) / float(b - a))))
    return float(K[-1][1])


def cam(f):
    """-> Dawn.cam keywords.  push: the distance shrinks at a constant rate of scale; the cloud glides from a2's place to the middle; tilt lifts the sun to the top, then 56-59 brings it down on the match circle"""
    if f < 14: return dict(cam_at(79))
    t = min(1., (f - 13.5) / 31.); e = t * t * t * (t * (6 * t - 15) + 10); e2 = e + .012 * max(0, f - 45) / 10. * (f > 44)
    d = D0 * (D1 / D0) ** e2; z = 1.0 + .10 * e; tl = -232. * e; x = 742 + 206 * e; y = 842 - 70 * e; q = cam_at(79); z0 = q['zoom']; z = z0 + (1.10 - z0) * e; cy = 486 + 124 * z + tl
    c = dict(zoom=z, dx=BOY[0] - (x - 960) * d / (1500. * z), dy=BOY[1] - d, dz=BOY[2] + (y - cy) * d / (1500. * z), tilt=tl)
    if f >= 56: u = [.14, .52, .88, 1.][f - 56]; c['zoom'] = z + (2. - z) * u; c['tilt'] = tl * (1 - u)
    return c


class Shot:
    id = 'h4'; n = 60
    finish = dict(bloom=.28, vig=.13)
    EVENTS = [(0, 'ambience', dict(note='same open air as a2, morning. SUN: frames 0-13 (960,486) r 104; 45-55 (960,254) r 114; frame 59 exactly (960,486) r 208 for h5')), (0, 'laugh', dict(size=.35, note='seven people laughing far away on every beat 0 10 20 30; closer and closer with the push 14-44')), (14, 'push', dict(size=.3, note='air opens, 14-44')), (25, 'squash', dict(size=.3, note='overloaded Nimbus squashes on off-beats 25 35 45 55, wheeze of vapour')), (28, 'dragon', dict(size=.35, note='a small Shenron peeps out of the cloud behind, tiny rumble; settles by 40')), (40, 'laugh', dict(size=.9, note='everybody, close, beats 40 and 50')),
              (39, 'wobble', dict(size=.3, note='Krillin rocks, arms windmill 39-43')), (40, 'sag', dict(size=.8, note='the big laugh drives the overloaded Nimbus down 40-43: one deep sag, long soft groan; springs back 45')), (44, 'yelp', dict(size=.6, note='Krillin topples off the rim, yelp 44-47')), (48, 'grab', dict(size=.6, note='white glove snaps shut on his collar, dead stop; Vegeta does not look')), (50, 'pull', dict(size=.4, note='hauled back 50-53')), (53, 'plop', dict(size=.35, note='sits down again, puff')), (54, 'huff', dict(size=.2, note='Vegeta: hn.')), (56, 'whip', dict(size=.5, note='tilt up 56-59 into the sun'))]

    def setup(s): cv2.setNumThreads(1); s.D = Dawn('h4'); s.N = Nimbus(3); s.N2 = Nimbus(11, size=.62); s.xx, s.yy = grid(); s.nz = cv2.resize(fbm(H // 2, W // 2, (60, 22), (1, .5), 5, 3.2), (W, H)); s.seat = [s.N.top(min(max(x * .92, -2.2), 2.1), zz, sag=SAG, f=0, stretch=1.16, squash=.92) for (_, x, zz, _, _, _) in CREW]

    def birds(s, im, f, c):
        cel = Cel(W, H, ss=3); r = np.random.default_rng(8)
        for k in range(9):
            (x, y), sc = s.D.at(c, (1150 + 150 * k + f * 3.4 + 40 * math.sin(k * 2.), 5200 + 260 * r.random(), 120 + 60 * math.sin(k * 1.7) + 5 * math.sin(f * .11 + k))); S = 9. * c.zoom * (.8 + .4 * r.random()); w = math.sin(f * .55 + k * 1.9) * (1 if math.sin(f * .07 + k) > -.4 else .15); L = np.array([(-1, -.10 - .55 * w), (-.5, -.28 - .20 * w), (0, 0), (.5, -.28 - .20 * w), (1, -.10 - .55 * w)]) * S
            cel.line(spl(np.array([x, y]) + L, n=6), max(.9, S * .13), '#7c6c96', taper=(.6, .6), press=.1, seed=k, tremor=0, wfun=lambda t: .4 + np.sin(np.pi * t), alpha=.8)
        return over(im, cel.layer())

    # ------------------------------------------------------------------------------------ acting
    def pose(s, who, f, i):
        """sitting, legs over the rim, laughing on the beat; everyone a little differently.  -> pose, offset (cloud units)"""
        ph = (i * 1.3) % 3 - 1; g = f - f % 2; b, h = beat(g, ph), hop(g, ph); P = dict(p=0., A=0., hip=(0., -.30 - .06 * h), sq=-.06 * b, lean=.05 * b, fn=(.40, -.06), an=-.4, ff=(.52, -.10), af=-.4, arm=0., roll=.08 * b, hdy=-.04 * h, lag=(-.02, .10 * beat(g, ph + 2.2)), yaw=CREW[i][5], pitch=-.10 * h, sw=.2 * b, tw=0., ex={}, t=g); off = (0., 0.)
        big = g >= 40; ey = 'happy' if big and (g // 5) % 4 < 3 else 'dot'
        if who == 'goku': P.update(handN=(-.82, -.60 - .18 * h), handF=(.84, -.64 - .18 * beat(g, ph + 3)), hkN='open', hkF='open', ex=dict(mouth='laugh' if big else 'open', o=.9 + .3 * h, eyes=ey))
        elif who == 'gohan': P.update(handN=(-.62, -.50 - .20 * h), handF=(.60, -.50 - .20 * h), ex=dict(mouth='laugh', eyes=ey, ball=-.04 * b))
        elif who == 'bulma': P.update(handN=(-.30, .34), handF=(.72, -.55 - .14 * h), ex=dict(mouth='laugh' if big else 'open', eyes=ey, swing=(.06 * b, -.10 * h)))
        elif who == 'roshi': P.update(handN=(-.36, .42), handF=(.46, .36), ex=dict(mouth='laugh' if big else 'open', o=.8 + .3 * h, swing=(.04 * b, -.05 * h)), staff=(1.35 + .04 * b, -.30 - .10 * h))
        elif who == 'piccolo': P.update(ex=dict(mouth='smile', eyes='line' if 44 <= g <= 47 else 'dot')); P['roll'] *= .4; P['hyaw'] = .30 + (.35 if 44 <= g < 53 else 0.)
        elif who == 'krillin':
            P.update(handN=(-.62, -.52 - .18 * h), handF=(.60, -.52 - .18 * h), hkN='open', hkF='open', ex=dict(mouth='laugh' if big else 'open', o=.9 + .3 * h, eyes=ey))
            if f >= 39:                                                     # rock back, windmill, topple to the left over the rim, caught, hauled up
                ln = kf(f, [(39, 0), (41, .16), (43, -.10), (44, -.22), (46, -.62), (47, -.95), (48, -1.06), (49, -.94), (50, -.98), (52, .16), (54, -.04), (56, 0)], lambda u: u if 44 <= f <= 48 else float(ease(u))); dy = kf(f, [(43, 0), (45, .10), (47, .52), (48, .62), (49, .56), (50, .58), (52, -.10), (53, .04), (55, 0)]); dx = kf(f, [(43, 0), (47, -.30), (48, -.34), (50, -.33), (52, .04), (54, 0)]); off = (dx, dy); P['lean'] = ln; P['roll'] = ln * .12 + (.10 * math.sin(f * 2.1) if 44 <= f < 50 else 0); kick = float(np.clip(-ln, 0, 1)); P['fn'] = (.40 + .30 * kick, -.06 - .62 * kick); P['ff'] = (.52 + .30 * kick, -.10 - .78 * kick)
                if f < 49: w = f * 2.4; P['handN'] = (-.70 + .16 * math.cos(w), -.42 + .34 * math.sin(w)); P['handF'] = (.70 + .16 * math.cos(w + 2), -.42 + .34 * math.sin(w + 2))
                if 43 <= f < 50: P['ex'] = dict(mouth='scream', o=1., eyes='blank')
                elif 50 <= f < 53: P['ex'] = dict(mouth='o', eyes='blank')
                elif f >= 53: P['ex'] = dict(mouth='open', o=.9, eyes='happy'); P['handN'] = (-.30, -.30); P['handF'] = (.62, .30)
        elif who == 'vegeta':
            on = f >= 38; k = h if on else 0.; P.update(roll=-.06 + (.05 * b if on else 0.), lean=.02, ex=dict(mouth='frown' if f < 54 else 'smile', eyes='line' if 44 <= f <= 55 else 'dot', look=(.35, 0.))); P['hip'] = (0., -.30 - .03 * k); P['fn'] = (.44, -.08 - .10 * (hop(f, 5.) if on else 0.))
            if 28 <= f < 36: P['hyaw'] = .10; P['ex']['look'] = (-.5, 0.)      # one sour look back at them, then front again
        return P, off

    def dip(s, f): return kf(f, [(38, 0), (40, -.10), (43, 1.), (45, -.34), (47, .16), (48, .34), (50, .12), (52, -.06), (54, .50), (56, 0)])

    # ------------------------------------------------------------------------------------ frame
    def frame(s, f):
        q = cam(f); c = s.D.cam(**q); d_ = BOY[1] - q['dy']; im, z, front = s.D.frame(f, c, split=(820. if f < 14 else max(60., d_ + 60.))); xx, yy = s.xx, s.yy; im = s.birds(im, f, c)
        sq = beat(f, 2.5); (bx, by), ms = s.D.at(c, BOY); S = ms * M * BIG; k = S / 22.; kk = S / 41.; bob = (-2.6 * hop(f) + 1.4) * kk; by = by + 14 * kk + bob + ([0, 60, 260, 520][f - 56] if f >= 56 else 0) + s.dip(f) * .40 * S; roll = .035 + .020 * math.sin(2 * math.pi * f / 20. + 1) + kf(f, [(43, 0), (48, -.075), (51, .03), (55, 0)])
        u = np.clip((yy - (by + 96 * k)) / (250. * k), -.3, 1.4); a_ = .34 * float(np.clip(1.5 - k / 3.2, 0, 1))
        if a_ > .01: cxs = bx - 62 * k - 190 * k * u + 20 * k * s.nz; hw = (52 + 72 * np.clip(u, 0, 2)) * k * (1 + .30 * s.nz); m = smooth(1.05, .55, np.abs(xx - cxs) / np.maximum(hw, 8)) * smooth(0., .16, u) * (1 - .75 * smooth(.25, 1.0, u)) * smooth(-1.0, .1, s.nz + .4); im = im * (1 - a_ * m[..., None]) + rgb('#6470b2') * a_ * m[..., None]
        if front is not None: im = over(im, front)
        hz = np.exp(-(((xx - bx - 10) / (130 * kk * 1.9)) ** 2 + ((yy - by + 40 * kk) / (78 * kk * 1.9)) ** 2)) * .36; im = im + (rgb('#ffe6d8') - im) * hz[..., None]
        if by - 7 * S > H + 60: return s.whip(im, f)
        im = s.dragon(im, f, bx, by, S)
        sag = [(x, zz, w * (1 + .20 * hop(f, i * 1.3 + 5))) for i, (x, zz, w) in enumerate(SAG)]; dp = s.dip(f); st = 1.16 + .05 * sq + .10 * dp; sc = .92 - .06 * sq - .20 * dp; sag = [(x, zz, w * (1 + 2.4 * max(0., dp) * math.exp(-(x / 1.9) ** 2))) for x, zz, w in sag]; R = dict(yaw=.06, pitch=.24, roll=roll)
        im = over(im, s.N.vapour(bx, by, S, f, vel=(-1.6 * kk, .25 * kk), n=16, life=90, pal=PALC, yaw=.06, pitch=.24, speed=.15, op=.85, size=1.0)); im = s.lump(im, f, bx, by, S)
        b, fr = s.N.draw(bx, by, S, f, pal=PALC, light=(.42, .46, -.30), speed=.18, tone=(84, 56, 20), sag=sag, stretch=st, squash=sc, breath=1.8, front_z=1.05, **R); im = im + (rgb('#ffe0a0') * .10) * blur(b[..., 3], 16)[..., None]; im = over(im, b)
        Sc = S * CH; x0, y0 = int(max(0, bx - 5. * S)), int(max(0, by - 6. * S)); x1, y1 = int(min(W, bx + 5.2 * S)), int(min(H, by + 2.6 * S)); Rm = NE((R['yaw'], R['pitch'], R['roll']))
        mk = lambda: D(x0, y0, max(8, x1 - x0), max(8, y1 - y0), ss=3 if S < 60 else 2, lw=max(1.8, 4.5 * Sc / 108.), line='#5a3326')
        d, d2, tops = mk(), None, {}; out2 = 44 <= f <= 52
        if out2: d2 = mk()
        for i, (who, x, zz, up, sz, yaw) in enumerate(CREW):
            y = (s.seat[i] + 1.1) * sc / .92 - 1.1 - .34 * max(0., dp) * math.exp(-(x / 1.6) ** 2); v = Rm @ np.array([x, y, zz]); kz = 9. / (9. - v[2]); P, off = s.pose(who, f, i); P['lean'] = P.get('lean', 0.) - (.16 * dp * math.tanh(x) if who != 'krillin' or f < 44 else 0.); P['sq'] = P.get('sq', 0.) + .12 * dp; px, py = bx + v[0] * kz * S + off[0] * S, by - v[1] * kz * S + .10 * S - up * .74 * S + off[1] * S
            if who == 'vegeta' and 47 <= f <= 53 and 'krillin' in tops:          # the arm goes out sideways; the head does not turn
                t0 = ST.fig(D(0, 0, 8, 8, ss=1), who, px, py, Sc * sz, P)['t']; C = tops['krillin']['t'](np.array([[.22, .02]]))[0]; loc = np.asarray(t0.inv(np.asarray(C)[None]) if np.ndim(C) == 1 else t0.inv(C)).reshape(-1)[:2]; sh = np.array([-.36, .17]); g = kf(f, [(47, .55), (48, 1.04), (49, 1.)]); hp = sh + (loc - sh) * g; P.update(crossed=False, handN=tuple(hp), handF=(.22, .46), hkN='grip' if f >= 48 else 'open', armk=max(1., float(np.hypot(*(hp - sh))) / .56), armTop=True)
            tops[who] = ST.fig(d2 if out2 and who in ('krillin', 'vegeta') else d, who, px, py, Sc * sz, P)
        rd = min(4., 1.3 + S * .020); fin = lambda q: kit.tint(kit.rim(q.layer(grain=.02), -rd, rd * .75, '#fff6d8', .75, .4), '#ffe9dc', .35); L = fin(d); im = kit.aura(im, L, 6 * min(kk, 2), '#ffe0b0', .18); im = over(im, L); im = over(im, fr)
        if out2: im = over(im, fin(d2))                                       # he is over the rim now, and so is the arm
        if f >= 44: im = s.sweat(im, f, tops)
        return s.whip(im, f)

    def whip(s, im, f):
        if f >= 56: im = kit.smear(im.astype(f32), 0, [34, 110, 80, 0][f - 56], 41)
        return im.astype(f32)

    def dragon(s, im, f, bx, by, S):
        """a small Shenron looks out of a bank behind them, delighted"""
        r = kf(f, [(27, 0), (34, .86), (37, 1.08), (40, 1.)]) if f >= 27 else 0.
        if r <= .01: return im
        r -= .05 * hop(f, 3.); hx, hy = bx - 4.25 * S, by + (.75 - 2.05 * r) * S; sw = .10 * math.sin(f * .30) * S; P = np.array([(hx + sw, hy), (hx - .42 * S + sw * .7, hy - .02 * S), (hx - .74 * S + sw * .4, hy + .42 * S), (hx - .72 * S, hy + 1.0 * S), (hx - .60 * S, hy + 1.6 * S), (hx - .52 * S, hy + 2.0 * S)]); P = spl(P, n=26); t = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; u = np.linspace(0, t[-1], 200); P = np.c_[np.interp(u, t, P[:, 0]), np.interp(u, t, P[:, 1])]
        x0, y0 = int(max(0, bx - 6.4 * S)), int(max(0, hy - 2.2 * S)); w, h = int(min(W, bx) - x0), int(min(H, by + 3 * S) - y0)
        if w < 20 or h < 20: return im
        d = D(x0, y0, w, h, ss=2, lw=max(1.6, 3.9 * S * CH / 108.), line='#4a3a3a'); shenron(d, P, S * .46, f, legs=(), hs=1.22); L = kit.tint(d.layer(grain=.02), '#ffe0dc', .40); b, fr = s.N2.draw(bx - 4.75 * S, by + .42 * S, S * 1.25, f, yaw=.2, pitch=.18, pal=PALS, light=(.42, .46, -.30), speed=.05, tone=(82, 52, 18), tail=.6)
        im = over(im, L); return over(im, b)

    def sweat(s, im, f, tops):
        """Krillin's alarm: three drops and two motion ticks, drawn"""
        if 'krillin' not in tops or not 44 <= f < 53: return im
        cel = Cel(W, H, ss=2); n = tops['krillin']['h'].n; u = float(np.hypot(*(n(np.array([[1., 0]]))[0] - n(np.array([[0., 0]]))[0]))) ; a = f - 44
        for j, (x, y) in enumerate(((.95, -.85), (1.25, -.45), (.55, -1.25))):
            p = n(np.array([[x + .05 * a, y - .10 * a + .02 * a * a]]))[0]; r = u * (.10 - .006 * a); th = np.linspace(0, 2 * math.pi, 16)[:-1]; Q = np.c_[p[0] + np.cos(th) * r * .8, p[1] + np.sin(th) * r * np.where(np.sin(th) < 0, 1.7, 1.)]; cel.fill(Q, '#bfe3f4'); kit.outline(cel, Q, max(1.2, u * .03), '#5a7fa0', j)
        return over(im, cel.layer())

    def lump(s, im, f, bx, by, S):
        """lumps squeezed off the tail: they fall, squash on nothing, thin"""
        cel = Cel(W, H, ss=2)
        for b, x0 in ((20, -3.4), (34, -2.8), (48, 1.4)):
            a = f - b
            if not 0 <= a < 22: continue
            x, y = bx + x0 * S - .03 * S * a, by + 1.5 * S + .0013 * S * a * a; R = .30 * S * (1 - a / 23.) ** .6; th = np.linspace(0, 2 * math.pi, 26)[:-1]; w = 1 + .08 * np.sin(th * 5 + a * .2) + .05 * np.sin(th * 7); Pq = np.c_[x + np.cos(th) * R * w * 1.15, y + np.sin(th) * R * w * .85]; cel.fill(Pq, PALC['shade'], .95); cel.fill(Pq * .74 + np.array([x + R * .12, y - R * .25]) * .26, PALC['light'], .95)
        return over(im, cel.layer())
