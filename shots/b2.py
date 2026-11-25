"""B2  EIGHT STICKERS.  80 f.  0 Goku, 10 Vegeta, 20 Gohan, 30 Bulma, 40 Piccolo, 50 Frieza, 60 Cell, 65 Buu squashes Cell, 70 the sheet.  76 everyone blinks.  Almost everyone."""
import numpy as np, cv2, math
from db import W, H, comp
from db.paint import f32, rgb
from db.ink import Cel, spl
from chars.chibi_core import N, E, Eo, back, A, ell, star, tube, mix, pi, keys
from chars import chibi_stick as SK
from chars.chibi_stick import BG, CARD, sticker, grid

SLOT = [('goku', 0), ('vegeta', 10), ('gohan', 20), ('bulma', 30), ('piccolo', 40), ('frieza', 50), ('cell', 60), ('buu', 65)]
DIR = dict(goku=(0, -1), vegeta=(1, 0), gohan=(0, 1), bulma=(-1, 0), piccolo=(.8, -.6), frieza=(-1, 0), cell=(0, 1), buu=(1, 0))        # side it comes in from
CXY = dict(goku=(W / 2 - 70, 556), vegeta=(W / 2, 604), gohan=(W / 2, 596), bulma=(W / 2 - 30, 530), piccolo=(W / 2 - 40, 560), frieza=(W / 2 - 40, 516), cell=(W / 2 + 50, 612), buu=(W / 2, 580))
TILT = dict(goku=-.06, vegeta=.05, gohan=-.03, bulma=.07, piccolo=-.05, frieza=0., cell=.04, buu=-.05)
DK = lambda c, t=.15: mix(c, '#1a0a24', t)
LT = lambda c, t=.16: mix(c, '#ffffff', t)
S0 = 252


def polar(cx, cy, r, a): return A([cx + math.cos(a) * r, cy + math.sin(a) * r])


def field(c, who, g, t):
    """the moving graphic of each colour, full frame; g = frame in slot (for growth), t = running time"""
    b = BG[who]; R = lambda P, col, al=1.: c.fill(P, col, al); cx, cy = W / 2, H * .52; full = A([(0, 0), (W, 0), (W, H), (0, H)])
    if who == 'goku':                                                       # burst: rays shoot out and turn
        R(full, b); rot = .50 * Eo((g + 1) / 5.) + .014 * t; L = 1300 * Eo((g + 1) / 3.)
        for k in range(18): a = k * 2 * pi / 18 + rot; R(A([(cx, cy), polar(cx, cy, L, a), polar(cx, cy, L, a + .175)]), DK(b))
        for k in range(18): a = k * 2 * pi / 18 + rot + .20; q = (g * 170 + k * 37) % 900 + 380; R(A([polar(cx, cy, q, a), polar(cx, cy, q + 150, a - .008), polar(cx, cy, q + 150, a + .008)]), LT(b, .5), .9) if g < 8 else None
    elif who == 'vegeta':                                                   # blades: diagonal stripes slash through, always moving right to left
        R(full, b); o = -g * 46 + 220 * (1 - Eo(g / 4.))
        for k in range(-4, 18): x = k * 170 + o % 170 - 170; R(A([(x + 320, 0), (x + 400, 0), (x + 400 - 420, H), (x + 320 - 420, H)]), DK(b, .17)); R(A([(x + 410, 0), (x + 420, 0), (x + 0, H), (x - 10, H)]), LT(b, .28))
    elif who == 'gohan':                                                    # dots that jump with his fists
        R(full, b); p = [0, 0, .2, 1., .6, .3, 0., 1., .6, .3][min(g, 9)]
        for j in range(-1, 9):
            for i in range(-1, 15):
                x, y = i * 146 + (j % 2) * 73, j * 138; d = math.hypot((x - cx) / 900, (y - cy) / 560); r = (20 + 30 * max(0., 1 - abs(d - (.25 + .16 * (g % 4))) * 2.2) + 14 * p); R(ell(x, y - 16 * p, r, r, n=20), DK(b, .14))
    elif who == 'bulma':                                                    # stars inside stars, opening
        R(full, b)
        for k in range(7, -1, -1): r = ((k * 220 + g * 46 + 60 * Eo(g / 3.)) % 1760) + 60; R(star(cx + 10, cy + 20, r, -pi / 2 + .10 * math.sin(k) + .03 * g, .52), DK(b, .16) if k % 2 else LT(b, .05))
    elif who == 'piccolo':                                                  # waves of bites: scallops march across
        R(full, b)
        for j in range(6):
            y = j * 220 - 60; o = ((j % 2) * 2 - 1) * g * 14
            for i in range(-2, 16): x = i * 150 + (o % 150) + (j % 2) * 75; R(ell(x, y + 60, 82, 82, n=24), DK(b, .12))
            R(A([(0, y + 60), (W, y + 60), (W, y + 150), (0, y + 150)]), DK(b, .12)); [R(ell(i * 150 + (o % 150) + (j % 2) * 75 + 75, y + 152, 66, 40, n=20), b) for i in range(-2, 16)]
        if 3 <= g <= 5:
            for k in range(12): a = k * 2 * pi / 12 + .2; r0 = 300 + 200 * (g - 3); R(A([polar(cx + 180, cy + 60, r0, a), polar(cx + 180, cy + 60, r0 + 230, a - .03), polar(cx + 180, cy + 60, r0 + 230, a + .03)]), '#fff4cf')
    elif who == 'frieza':                                                   # a target that closes on us, then stops dead when his eyes open
        g2 = min(g, 5) + .0; R(full, b)
        for k in range(10, -1, -1): r = k * 150 + 150 - (g2 * 42) % 150; R(ell(cx, cy - 20, r, r, n=90), DK(b, .18) if (k + int(g2 * 42 // 150)) % 2 else LT(b, .03))
        if g >= 6: R(full, '#2a0a2e', .10 + .05 * min(3, g - 6))
    elif who == 'cell':                                                     # honeycomb turning over, cell by cell
        R(full, b); s = 104
        for j in range(-1, 8):
            for i in range(-1, 13):
                x, y = i * s * 1.732 + (j % 2) * s * .866, j * s * 1.5; d = math.hypot(x - 500, y - 420) / 1200.; u = min(1., max(0., (g + 1) / 3. - d)); ph = math.sin(g * .9 - d * 6); k = (.60 + .30 * max(0, ph)) * u; R(A([polar(x, y, s * k, q * pi / 3 + pi / 6) for q in range(6)]), DK(b, .15) if (i + j) % 3 else LT(b, .12))
    elif who == 'buu':                                                      # gum: bubbles rise and swell
        R(full, b); r_ = np.random.default_rng(9)
        for k in range(30): x = r_.uniform(0, W); y0 = r_.uniform(0, H * 1.4); r = r_.uniform(26, 120); y = (y0 - g * 34 * (60 / r + .4)) % (H * 1.4) - 120; w = 1 + .10 * math.sin(g * 1.1 + k); R(ell(x, y, r * w, r / w, n=28), DK(b, .13) if k % 3 else LT(b, .10)); R(ell(x - r * .3, y - r * .35, r * .22, r * .14, -.6, n=14), LT(b, .4), .7)


def card(c, who, g):
    z = {0: .55, 1: .86, 2: 1.10, 3: 1.04, 4: .985}.get(g, 1.); s = H * .405 * z; cr = N().at(W / 2, H * .53, TILT[who] + .05 * (1 - Eo(g / 4.))); P = cr(ell(0, 0, s * 1.06, s * 1.02, n=70, sq=7.)); c.fill(P + [10, 15], DK(BG[who], .42)); c.fill(P, CARD[who]); c.fill(cr(ell(0, 0, s * .985, s * .945, n=70, sq=7.)), LT(CARD[who], .08))


def wipe(who, g):
    """mask polygons of the incoming colour over the old one at g = 0, 1 (None = full)"""
    cx, cy = W / 2, H * .52; u = [.50, .86][g] if g < 2 else 1.
    if g >= 2: return None
    if who == 'vegeta': x = W * (1.15 - 1.35 * u); return [A([(x + 260, 0), (W + 600, 0), (W + 600, H), (x - 180, H)])] + [A([(x - 60 - k * 130 + 260, 0), (x - 10 - k * 130 + 260 + 30, 0), (x - 10 - k * 130 - 180 + 30, H), (x - 60 - k * 130 - 180, H)]) for k in range(3)]
    if who == 'gohan': return [ell(i * 146 + (j % 2) * 73, j * 138, 118 * u * (1.15 - .5 * math.hypot((i * 146 - cx) / 1200, (j * 138 - cy) / 800)), None, n=24) for j in range(-1, 9) for i in range(-1, 15)]
    if who == 'bulma': return [star(cx, cy, 1500 * u ** 1.6 + 160, -pi / 2 + .3 * (1 - u), .55)]
    if who == 'piccolo': x = W * (1.05 - 1.15 * u); return [A([(x, 0), (W + 10, 0), (W + 10, H), (x, H)])] + [ell(x + (k % 2) * 26, k * 168 + 40, 118, 118, n=30) for k in range(8)]
    if who == 'frieza': r = 1200 * (1 - u) + 60; th = np.linspace(0, 2 * pi, 90); return [np.vstack([np.c_[cx + np.cos(th) * 1800, cy + np.sin(th) * 1800], np.c_[cx + np.cos(th[::-1]) * r, cy + np.sin(th[::-1]) * r]])]
    if who == 'cell': s = 104; return [A([polar(i * s * 1.732 + (j % 2) * s * .866, j * s * 1.5, s * min(1.02, max(0., 1.5 * u - math.hypot(i * s * 1.732 - cx, j * s * 1.5 - H) / 1500)), q * pi / 3 + pi / 6) for q in range(6)]) for j in range(-1, 8) for i in range(-1, 13)]
    return None


_HT = {}


def halftone(a, who, w0=0, w1=W):
    if 'g' not in _HT:
        y, x = np.mgrid[:H, :W].astype(f32); p = 26.; u, v = (x * .866 + y * .5) / p, (-x * .5 + y * .866) / p; d = np.sqrt((u - np.round(u)) ** 2 + (v - np.round(v)) ** 2); g = np.clip((y / H - .34) * .60 + (np.abs(x - W / 2) / W - .28) * .5, 0, .46); _HT['g'] = (np.clip((g - d) * p + .5, 0, 1) * .40)[..., None]
    m = _HT['g']; return a * (1 - m) + rgb(DK(BG[who], .36)) * m


def warp(L, dx, dy, rot=0., sx=1., sy=1., cx=W / 2, cy=H / 2):
    c, s = math.cos(rot), math.sin(rot); M = np.float32([[c * sx, -s * sy, cx + dx - c * sx * cx + s * sy * cy], [s * sx, c * sy, cy + dy - s * sx * cx - c * sy * cy]]); return cv2.warpAffine(L, M, (W, H), flags=cv2.INTER_LINEAR)


class Shot:
    id = 'b2'; n = 80
    finish = dict(bloom=.10, vig=.06, ca=.3, grain=.007, soft=.04, lift=0., sat=1.04)
    EVENTS = [(k, 'sticker_slap', dict(size=.8, pan=p, note=f'{w} sticker slaps in, knocks the last one off the frame')) for (w, k), p in zip(SLOT, (0, .5, 0, -.5, .4, -.5, 0, .7))] + [
        (3, 'toy_pose', dict(who='goku', size=1., note='V sign at the lens: bright toy trumpet')), (7, 'ping', dict(size=.5, note='wink, star between the fingers')),
        (13, 'toy_pose', dict(who='vegeta', size=1., note='head snaps away: HMPH, steam from the nose')), (18, 'tick', dict(size=.3, note='one eye peeks back')),
        (23, 'toy_pose', dict(who='gohan', size=1., note='two fists up, squeak')), (27, 'toy_pose', dict(who='gohan', size=.8, note='fists up again')),
        (33, 'toy_pose', dict(who='bulma', size=1., note='wink: glockenspiel, a star flies off to the right')),
        (43, 'toy_pose', dict(who='piccolo', size=1., note='CHOMP, apple crunch')), (46, 'munch', dict(size=.5)), (48, 'munch', dict(size=.5)), (47, 'ear_flick', dict(size=.2)),
        (53, 'toy_pose', dict(who='frieza', size=.9, note='cheeks in hands, hearts, music-box sweetness')), (56, 'wrong_note', dict(size=1., note='his eyes open; everything stops; one cold wrong note held to 60')),
        (63, 'toy_pose', dict(who='cell', size=.8, note='taps his temple: woodblock tick')), (65, 'squash', dict(size=1., pan=-.5, note='Buu slams in from the right and flattens Cell against the left edge: rubber')),
        (68, 'toy_pose', dict(who='buu', size=1., note='enormous grin, kazoo; steam from both sides')),
        (70, 'sheet_slaps', dict(size=1., dur=4, note='eight stickers rain onto the paper, one every half frame')), (73, 'ball_bounce', dict(size=1., note='four-star ball lands in the centre')), (77, 'ball_bounce', dict(size=.5)),
        (76, 'blink_all', dict(size=.7, note='seven blinks together, tiny click')), (78, 'cold', dict(size=.4, note='Frieza has not blinked; the sheet dims 2 %'))]

    def setup(self): self._c = {}

    def stk(self, who, g, marks=True, S=S0):
        return sticker(who, g, CXY[who][0], CXY[who][1], S, with_marks=marks)

    def ground(self, who, g, prev=None, t=0):
        c = Cel(W, H, 0, 0, 1); field(c, who, g, t); w = wipe(who, g) if prev else None
        a = halftone(c.rgba()[..., :3], who); cd = Cel(W, H, 0, 0, 1); card(cd, who, g); a = comp.over(a, cd.rgba())
        if w is not None:
            p = Cel(W, H, 0, 0, 1); field(p, prev, 9, t); card(p, prev, 9); b = halftone(p.rgba()[..., :3], prev); m = Cel(W, H, 0, 0, 2); [m.fill(q, '#ffffff') for q in w]; k = m.rgba()[..., 3:4]; e = np.clip(cv2.dilate(k, np.ones((9, 9), np.uint8))[..., None] - k, 0, 1); a = b * (1 - k) + a * k; a = a * (1 - e) + rgb('#fffaf0') * e
        return a

    def frame(self, f):
        if f >= 70: return self.sheet(f)
        i = max(q for q in range(8) if SLOT[q][1] <= f); who, k0 = SLOT[i]; g = f - k0
        if who == 'buu': return self.squash(f, g)
        prev = SLOT[i - 1][0] if i else None; im = self.ground(who, g, prev, f); dx, dy = DIR[who]
        if prev and g < 3:                                                   # the old one flies, spinning, squashed by the blow
            t = [.11, .48, 1.05][g]; L = self.stk(prev, 9, False); r = (.10 if dx >= 0 else -.10) * [1, 3.2, 6][g] * (1 if dy <= 0 else -1); q = [.84, 1.06, 1.][g]; L = warp(L, -dx * W * t * .9, -dy * H * t * 1.15, r, q if dx else 1, q if dy else 1); im = comp.over(im, L)
        L = self.stk(who, g)
        if g == 0: L = warp(L, dx * 330, dy * 250, -.08 * (dx or .6), 1.16 if dx else .94, 1.16 if dy else .94); L = self.trail(L, dx, dy)
        elif g == 1: L = warp(L, -dx * 22, -dy * 18)
        if i and g == 0: im = self.hit(im, W / 2 + dx * 60, H / 2 + dy * 60, f)
        return comp.over(im, L)

    def trail(self, L, dx, dy):
        """drawn speed: three hard echoes of the paper edge behind it"""
        a = L[..., 3]; out = np.zeros_like(L)
        for k, al in ((3, .30), (2, .45), (1, .65)): m = cv2.warpAffine(a, np.float32([[1, 0, dx * 70 * k], [0, 1, dy * 60 * k]]), (W, H))[..., None] * al; out = out * (1 - m) + np.concatenate([rgb('#fffdf8') * m, m], -1)
        return out * (1 - L[..., 3:4]) + L

    def hit(self, im, x, y, f):
        c = Cel(W, H, 0, 0, 2); r = np.random.default_rng(f)
        for k in range(11): a = k * 2 * pi / 11 + r.uniform(-.1, .1); r0, r1 = r.uniform(330, 400), r.uniform(560, 760); c.fill(A([polar(x, y, r0, a - .02), polar(x, y, r1, a), polar(x, y, r0, a + .02)]), '#fffdf4')
        return comp.over(im, c.rgba())

    def squash(self, f, g):
        """65-69: Cell was so clever"""
        e = [1430, 1010, 905, 985, 960][g]; a = self.ground('cell', 5 + g, None, f); c = Cel(W, H, 0, 0, 1); field(c, 'buu', g, f); b = halftone(c.rgba()[..., :3], 'buu'); y = np.linspace(-20, H + 20, 40); x = e + [10, 34, -30, 16, -8][g] * np.sin(y / 120 + g * 1.4); m = Cel(W, H, 0, 0, 2); m.fill(np.vstack([np.c_[x, y], [(W + 20, H + 20), (W + 20, -20)]]), '#ffffff'); k = m.rgba()[..., 3:4]; l = Cel(W, H, 0, 0, 2); l.line(np.c_[x, y], 16, '#fffdf8', taper=(0, 0), press=.05); im = comp.over(a * (1 - k) + b * k, l.rgba())
        cc = Cel(W, H, 0, 0, 1); s = H * .40 * [.6, .95, 1.06, .98, 1.][g]; P = N().at([1640, 1440, 1410, 1445, 1440][g], H * .53, -.05)(ell(0, 0, s * .98, s * 1.0, n=70, sq=7.)); cc.fill(P + [10, 15], DK(BG['buu'], .42)); cc.fill(P, CARD['buu']); lay = cc.rgba(); lay = lay * k; im = comp.over(im, lay)
        q = [(.90, 1.02, -120), (.44, 1.20, -600), (.40, 1.24, -640), (.60, 1.10, -560), (.55, 1.14, -575)][g]; L = warp(self.stk('cell', 5 + g if g < 1 else 6, g == 0), q[2], 0, -.03 * g, q[0], q[1], cx=CXY['cell'][0], cy=CXY['cell'][1] + 1.7 * S0); im = comp.over(im, L)
        B = self.stk('buu', g); dx = [760, 420, 400, 470, 455][g]; B = warp(B, dx, 0, [.10, -.04, .02, 0, 0][g], [1.12, .96, 1., 1, 1][g], 1.); B = self.trail(B, 1, 0) if g == 0 else B; im = comp.over(im, B)
        return self.hit(im, e, 560, f) if g in (0, 1) else im

    def sheet(self, f):
        t = f - 70; pop = lambda i: (t + 1 - ORDER.index(i) * .30) / 2.2
        y = float(keys(f, [(69, -260), (70, -150), (73, 548, 'i'), (75, 452, 'o'), (77, 548, 'i'), (78, 524, 'o'), (79, 540, 'i')])); sq = {73: (1.16, .84), 74: (.96, 1.05), 77: (1.07, .93)}.get(f, (1., 1.)); by = y + (128 * (1 - sq[1]) if sq[1] < 1 else 0)
        im = grid(10 + t, 1 if f in (76, 77) else 0, pop=pop if t < 6 else None, ball=(960, by, sq[0], sq[1], .0 if f >= 73 else (f - 73) * .16))
        if t == 0: im = im * .82 + .18
        if f >= 76:                                                          # the paper dims, only two eyes stay in focus
            u = min(1., (f - 75) / 4.); d = cv2.GaussianBlur(im, (0, 0), .4 + 1.7 * u) * (1 - .02 * u - .01 * max(0, f - 78)); yy, xx = np.mgrid[:H, :W].astype(f32); m = np.clip(1.6 - np.sqrt(((xx - 736) / 112) ** 2 + ((yy - 829) / 46) ** 2), 0, 1)[..., None]; m = m * m * (3 - 2 * m); im = d * (1 - m) + im * m
        return im


ORDER = [0, 7, 2, 5, 1, 6, 3, 4]
