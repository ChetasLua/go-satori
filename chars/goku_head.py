"""GOKU's face on db.head3: construction, expressions, cel painting.  Author in FRONT coordinates (x right, y up, eye line 0, chin -1)."""
import numpy as np, math, cv2, cairo
from db.head3 import Head
from db.ink import spl, resample, Cel
from db.kit import outline, svg
from db.paint import rgb, f32, smooth
from .goku_hair import mixc, _c
from .goku_light import IRIS
A = np.array

TAB = dict(a=[(-1.0, .19), (-.985, .260), (-.95, .335), (-.88, .425), (-.78, .512), (-.66, .592), (-.54, .650), (-.42, .692), (-.28, .722), (-.1, .742), (.1, .750), (.35, .745), (.6, .70), (.8, .56), (.93, .34), (1.0, .05)],
           f=[(-1.0, .66), (-.93, .735), (-.82, .735), (-.66, .755), (-.48, .765), (-.28, .755), (-.06, .74), (.08, .755), (.2, .80), (.4, .79), (.6, .72), (.8, .54), (.93, .31), (1.0, .05)],
           b=[(-1.0, .05), (-.9, .10), (-.72, .24), (-.5, .52), (-.2, .78), (.1, .90), (.4, .92), (.65, .84), (.85, .58), (.95, .32), (1.0, .05)])


class GHead(Head):
    """Head with Goku's skull, a jaw that opens and a projection that never folds over the far cheek"""
    jopen = 0.; jx = 0.

    def __init__(self, jaw=0., **kw):
        Head.__init__(self, a=TAB['a'], f=TAB['f'], b=TAB['b'], p=2.35, nose=(.10, -.06, -.385, -.445, .150), cheek=.050, socket=.060, **kw); self.jopen = jaw; self._xs = None

    @classmethod
    def of(cls, H, jaw=0., jx=0.):
        q = cls(); q.pose(H.cx, H.cy, H.s, H.yaw, H.pitch, H.roll, H.persp); q.R = H.R; q.jopen, q.jx = jaw, jx; return q

    def _mesh(self, y0, y1, half, ny=90, nf=90):
        if not half: return Head._mesh(self, y0, y1, half, ny, nf)
        y = np.linspace(y0, y1, ny)
        zj = np.interp(y, [-1.0, -.99, -.965, -.92, -.86, -.79, -.73, -.67, -.61, -.52, -.42, -.30, -.2], [.64, .62, .565, .46, .32, .15, .02, -.07, -.11, -.13, -.135, -.14, -.14])
        F_, B, Aa = (np.interp(y, self.yy, t) for t in (self.F, self.B, self.A)); p = self.k['p']
        c = np.clip(zj / np.maximum(np.where(zj >= 0, F_, B), 1e-3), -1, .98)
        t = np.where(c >= 0, np.arctan2((1 - np.abs(c) ** p) ** (1 / p), np.abs(c)), np.pi / 2 + np.arcsin(np.clip(-c, 0, 1)))
        PH = np.linspace(-1, 1, nf)[None] * np.minimum(math.pi * .5 * half, t)[:, None]
        sx, cz = np.sin(PH), np.cos(PH); r = (np.abs(sx) ** p + np.abs(cz) ** p) ** (-1 / p)
        X = Aa[:, None] * sx * r; Z = np.where(cz >= 0, F_[:, None], B[:, None]) * cz * r; Y = np.repeat(y[:, None], nf, 1); fr = np.clip(cz, 0, 1)
        Z = Z + fr * (self._mid(X, Y) + self.k['cheek'] * np.exp(-((Y + .22) / .13) ** 2) * np.exp(-((np.abs(X) - .44) / .2) ** 2) * 1.4 - self.k['socket'] * np.exp(-((Y - .02) / .10) ** 2) * np.exp(-((np.abs(X) - .40) / .24) ** 2) * 1.3)
        return np.stack([X, Y, Z], -1)

    def _mid(self, x, y):
        g = lambda y0, sy, sx, k: k * np.exp(-((y - y0) / sy) ** 2) * np.exp(-(x / sx) ** 2)
        return g(-.590, .055, .17, .024) + g(-.712, .048, .15, .020) - g(-.650, .022, .16, .008) - g(-.800, .050, .2, .020) + g(-.925, .075, .24, .036) + g(.17, .07, .42, .030) - g(.055, .06, .16, .035)

    def P3(self, Q):
        Q = np.array(Q, float); Q[..., 1] = np.where(Q[..., 1] < -.12, -.12 + (Q[..., 1] + .12) * .925, Q[..., 1])          # the lower face a touch shorter than the master head
        if self.jopen or self.jx:
            w = smooth(-.52, -.86, Q[..., 1]) * smooth(-.25, .40, Q[..., 2])
            Q[..., 0] = Q[..., 0] + self.jx * w; Q[..., 1] = Q[..., 1] - self.jopen * w; Q[..., 2] = Q[..., 2] - self.jopen * .25 * w
        return Head.P3(self, Q)

    def face(self, top=.75):
        F = Head.face(self, top); n = max(3, int(.030 * self.s / 3.0)) * 2 + 1; k = np.hanning(n + 2)[1:-1]; k /= k.sum(); m = n // 2
        G = np.stack([np.convolve(np.r_[F[-m:, i], F[:, i], F[:m, i]], k, 'valid') for i in (0, 1)], -1)
        return G if abs(self.yaw) <= .75 else F

    def _xlim(self):
        if self._xs is not None and self._xs[0] == (self.yaw, self.pitch, self.roll): return self._xs
        ys = np.linspace(-.99, .9, 40); lo, hi = [], []; V = self.R[2]
        for y in ys:
            a = np.interp(y, self.yy, self.A); x = np.linspace(-a * .985, a * .985, 121); e = 4e-3; z0 = self.z(x, y)
            nx = -(self.z(x + e, y) - self.z(x - e, y)) / (2 * e); ny = -(self.z(x, y + e) - self.z(x, y - e)) / (2 * e); c = (nx * V[0] + ny * V[1] + V[2]) / np.sqrt(nx * nx + ny * ny + 1); ok = np.where(c > .03)[0]
            if len(ok) == 0: lo.append(0); hi.append(0)
            else:
                j = int(np.argmax(c)); l = j
                while l > 0 and c[l - 1] > .03: l -= 1
                r = j
                while r < len(x) - 1 and c[r + 1] > .03: r += 1
                lo.append(x[l] if l > 0 else -9.); hi.append(x[r] if r < len(x) - 1 else 9.)
        self._xs = ((self.yaw, self.pitch, self.roll), ys, A(lo), A(hi)); return self._xs

    def Pc(self, xy, dz=0., squeeze=True):
        xy = np.array(xy, float); _, ys, lo, hi = self._xlim(); l = np.interp(xy[..., 1], ys, lo); h = np.interp(xy[..., 1], ys, hi); x = xy[..., 0]
        if squeeze:
            k = .10
            lm = np.maximum(h * .955 - k, .02); d = np.maximum(x - k, 0); x = np.where(x > k, k + lm * np.tanh((d / lm) ** 1.25) ** .8, x)
            lm = np.maximum(-l * .955 - k, .02); d = np.maximum(-x - k, 0); x = np.where(x < -k, -k - lm * np.tanh((d / lm) ** 1.25) ** .8, x)
        else: x = np.clip(x, l, h)
        return self.P(np.stack([x, xy[..., 1]], -1), dz)

    def vis(self, x, y):
        _, ys, lo, hi = self._xlim(); return np.interp(y, ys, lo) < x < np.interp(y, ys, hi)

    def rest_y(self, y):
        g = np.linspace(-1.02, .5, 220); e = np.where(g < -.12, -.12 + (g + .12) * .925, g); e = e - self.jopen * smooth(-.52, -.86, e); return np.interp(np.where(np.asarray(y) < -.12, -.12 + (np.asarray(y) + .12) * .925, y), e, g)

    def Pm(self, xy, dz=0.):
        xy = np.array(xy, float); xy[..., 1] = self.rest_y(xy[..., 1]); dz = dz - 2.0 * xy[..., 0] ** 2 * float(smooth(.55, 1.25, abs(self.yaw))); return self.Pc(xy, dz)


def lw(s): return float(np.interp(math.log(max(s, 8)), [math.log(v) for v in (14, 40, 110, 300, 420, 700, 1200)], [1.0, 1.7, 3.0, 4.8, 5.6, 7.0, 9.]))


# ------------------------------------------------------------------ expressions
T5 = A([0, .22, .50, .80, 1.])
_E0 = dict(
    U=[-.030, .054, .096, .106, .062], Lo=[-.064, -.110, -.122, -.104, -.052], xi=.136, xo=.515, out=.055,          # lids inner->outer; out = height of the outer wall
    iris=.078, irish=1.30, pupil=.0, gx=0., gy=0., lidw=1.0, crease=1.0, glint=1., style='base',
    B=[(.100, .136), (.250, .186), (.435, .244), (.572, .198), (.655, .118)], bt=[.040, .050, .054, .032, .003], furrow=0.,
    mx=0., my=-.665, hw=.170, mU=[0, .004, .006, .004, 0], mL=[0, .004, .006, .004, 0], tU=0., tL=0., tongue=0., jaw=0., lip=1., clench=0., asym=0., mrot=0., gum=0., jx=0.,
    naso=0., blush=0., sweat=0., tears=0., well=0., vein=0., dark=0., under=0., closed=0., wrink=0., flare=0., cheek=0., black=0., tendon=0., cheekline=.0, brow_shadow=0., blank=0., glow=0., squeeze=0., tph=0.)


def _e(**k): d = dict(_E0); d.update(k); return d


_ANG = dict(U=[-.050, .008, .052, .096, .074], Lo=[-.070, -.104, -.112, -.092, -.040], B=[(.072, .010), (.215, .070), (.400, .158), (.540, .232), (.640, .186)], bt=[.044, .054, .058, .034, .003], furrow=.8, crease=0., out=.060, iris=.064)
EXPR = dict(
    calm=_e(),
    smile=_e(U=[-.030, .046, .080, .086, .044], Lo=[-.060, -.088, -.094, -.082, -.044], B=[(.112, .152), (.255, .206), (.430, .256), (.560, .206), (.640, .128)], hw=.205, my=-.650, mU=[.044, .004, -.008, .004, .050], mL=[.044, .004, -.008, .004, .050], blush=.25, cheek=.4, asym=.3),
    grin=_e(U=[-.030, .050, .084, .088, .044], Lo=[-.060, -.078, -.080, -.072, -.044], B=[(.112, .164), (.255, .226), (.430, .276), (.560, .222), (.640, .140)], hw=.285, my=-.600, mU=[.060, .022, .008, .022, .066], mL=[.046, -.170, -.222, -.176, .052], tU=.082, tongue=1., jaw=.075, blush=.3, cheek=.8, naso=.2),
    laugh=_e(U=[-.074, -.004, .032, .016, -.050], Lo=[-.078, -.012, .022, .006, -.054], closed=1., crease=0., B=[(.118, .190), (.260, .252), (.430, .288), (.560, .226), (.640, .150)], hw=.300, my=-.590, mU=[.072, .034, .018, .034, .078], mL=[.030, -.250, -.316, -.256, .034], tU=.088, tongue=1.1, jaw=.15, blush=.35, cheek=1., naso=.3),
    serious=_e(**dict(_ANG, B=[(.084, .052), (.225, .100), (.405, .176), (.545, .232), (.640, .182)], U=[-.042, .022, .064, .096, .070], furrow=.45, hw=.150, my=-.668, mU=[-.004, .002, .004, .002, -.006], mL=[-.004, .002, .004, .002, -.006])),
    determined=_e(**dict(_ANG, hw=.165, mU=[-.010, .003, .006, .002, -.006], mL=[-.010, .003, .006, .002, -.006], cheekline=.7)),
    smirk=_e(**dict(_ANG, hw=.175, mU=[.000, -.006, -.004, .010, .040], mL=[.000, -.006, -.004, .010, .040], furrow=.6, asym=.2, cheekline=.4)),
    shock=_e(U=[-.020, .082, .124, .124, .062], Lo=[-.066, -.126, -.140, -.124, -.062], iris=.046, irish=1.2, B=[(.108, .236), (.255, .280), (.430, .306), (.560, .248), (.640, .178)], hw=.130, my=-.690, mU=[-.02, .050, .062, .050, -.02], mL=[-.04, -.085, -.100, -.085, -.04], tU=.030, jaw=.07, sweat=1., under=.5, out=.07),
    fear=_e(U=[-.018, .074, .108, .104, .048], Lo=[-.064, -.118, -.130, -.114, -.056], iris=.040, B=[(.098, .258), (.245, .262), (.425, .252), (.560, .186), (.645, .110)], hw=.150, my=-.680, mU=[-.026, .006, .012, .006, -.022], mL=[-.030, -.044, -.048, -.040, -.026], clench=1., sweat=1., furrow=.5, under=.7),
    grief=_e(U=[-.022, .026, .046, .034, -.014], Lo=[-.058, -.086, -.092, -.082, -.048], iris=.064, gy=-.012, B=[(.090, .226), (.240, .210), (.420, .178), (.555, .120), (.640, .060)], bt=[.034, .040, .042, .024, .003], furrow=-.9, hw=.165, my=-.678, mU=[-.040, .006, .020, .010, -.034], mL=[-.052, -.052, -.048, -.046, -.046], clench=1., well=.8, tears=.6, naso=.4, under=.6),
    griefhid=_e(U=[-.048, -.070, -.080, -.072, -.040], Lo=[-.052, -.074, -.084, -.076, -.044], closed=1., crease=0., B=[(.086, .210), (.240, .190), (.420, .156), (.555, .100), (.640, .050)], furrow=-1., hw=.190, my=-.672, mU=[-.046, .016, .030, .018, -.040], mL=[-.074, -.074, -.066, -.066, -.066], clench=1., naso=.6, jaw=.02),
    shut=_e(U=[-.044, -.066, -.076, -.066, -.036], Lo=[-.048, -.070, -.080, -.070, -.040], closed=1., crease=.6),
    serene=_e(U=[-.044, -.066, -.076, -.066, -.036], Lo=[-.048, -.070, -.080, -.070, -.040], closed=1., crease=.6, B=[(.100, .100), (.245, .150), (.425, .206), (.558, .176), (.640, .110)], hw=.140, mU=[.004, .002, .002, .002, .006], mL=[.004, .002, .002, .002, .006]),
    ui=_e(**dict(_ANG, U=[-.054, .006, .050, .092, .076], Lo=[-.078, -.114, -.122, -.100, -.036], xi=.126, xo=.528, B=[(.072, .020), (.215, .074), (.402, .156), (.545, .234), (.655, .194)], style='ui', iris=.066, irish=1.15, hw=.150, my=-.668, mU=[-.006, .002, .003, .002, -.004], mL=[-.006, .002, .003, .002, -.004], furrow=.9, cheekline=.6, brow_shadow=.6, glow=1.)),
    rage=_e(**dict(_ANG, B=[(.060, -.018), (.205, .050), (.390, .150), (.535, .240), (.645, .200)], U=[-.064, -.004, .048, .098, .082], iris=.050, hw=.235, my=-.665, mU=[-.030, .052, .062, .056, -.014], mL=[-.072, -.098, -.104, -.098, -.056], clench=1., naso=.8, wrink=.8, flare=.8, vein=.6, furrow=1., cheekline=1., jaw=.02, tendon=.6)),
    grit=_e(**dict(_ANG, B=[(.060, -.018), (.205, .050), (.390, .150), (.535, .240), (.645, .200)], U=[-.064, -.004, .048, .098, .082], hw=.215, my=-.668, mU=[-.030, .038, .046, .040, -.020], mL=[-.052, -.070, -.074, -.070, -.044], clench=1., naso=.6, wrink=.5, flare=.5, furrow=1., cheekline=.8)),
    blank=_e(**dict(_ANG, B=[(.056, -.026), (.200, .032), (.388, .126), (.534, .220), (.648, .190)], U=[-.068, -.010, .042, .092, .082], Lo=[-.078, -.106, -.112, -.094, -.040], blank=1., hw=.200, my=-.690, mU=[-.02, .060, .072, .062, -.02], mL=[-.06, -.130, -.150, -.130, -.06], tU=.050, tL=.030, jaw=.10, wrink=.6, vein=1., furrow=1., tears=1., naso=.6, cheekline=.8, flare=.6)),
)


def _scr(k, j=0.):
    m = lambda a, b: a + (b - a) * k
    return _e(**dict(_ANG, B=[(.052, m(-.030, -.046)), (.196, .022 + .004 * j), (.385, .118), (.532, m(.214, .224)), (.650, .194)], bt=[.050, .058, .060, .036, .003], U=[-.072, m(-.018, -.010), m(.034, .046), m(.084, .096), .082], Lo=[-.080, -.100, m(-.104, -.108), -.092, -.040], xi=.128, xo=.520,
                     iris=m(.066, .060), irish=1.20, gy=.004, hw=m(.215, .300) + .004 * j, my=m(-.640, -.600), mrot=.012 * j, jaw=m(.14, .30) + .006 * j, jx=.006 * j,
                     mU=[m(-.020, -.040), m(.040, .062), m(.052, .074), m(.042, .064), m(-.016, -.034)], mL=[m(-.110, -.250), m(-.210, -.420), m(-.238, -.452), m(-.212, -.416), m(-.104, -.242)],
                     tU=m(.060, .086), tL=m(.040, .062), gum=m(.2, .8), tongue=1., lip=0., naso=1., wrink=1., vein=1., furrow=1., flare=1., cheekline=1., tendon=1., black=.3 * k, under=.8))


EXPR['scream_a'] = _scr(.2); EXPR['scream_b'] = _scr(.65); EXPR['scream'] = _scr(1.); EXPR['scream_d'] = _scr(1., 1.)
EXPR['roar'] = EXPR['scream']


def getx(e, **over):
    if isinstance(e, str): d = dict(EXPR[e])
    elif isinstance(e, (tuple, list)) and len(e) == 3: d = dict(blend(*e))
    else: d = _e(**e) if not set(_E0) <= set(e) else dict(e)
    d.update(over); return d


def blend(a, b, t):
    a, b = getx(a), getx(b); o = {}
    for k in _E0:
        if isinstance(a[k], str): o[k] = a[k] if t < .5 else b[k]
        else: o[k] = (A(a[k], float) * (1 - t) + A(b[k], float) * t) if np.ndim(a[k]) else a[k] * (1 - t) + b[k] * t
    return o


def lids(x, t):
    X = getx(x); S = EXPR['shut']; o = dict(X)
    for k, a in (('U', 1.), ('Lo', 1.), ('B', .22)): o[k] = A(X[k], float) * (1 - a * t) + A(S[k], float) * a * t
    return o


def prof(t, pts): p = A(pts, float); return np.interp(t, p[:, 0], p[:, 1])


def crv(xs, ys, n=48, straight=0.):
    m = max(4, n // (len(xs) - 1)); S = spl(np.c_[xs, ys], n=m)
    if straight:
        u = np.arange(len(S)) / m; P = np.c_[np.interp(u, np.arange(len(xs)), xs), np.interp(u, np.arange(len(xs)), ys)]; S = S * (1 - straight) + P * straight
        k = np.hanning(7); k /= k.sum(); q = np.vstack([S[:1]] * 3 + [S] + [S[-1:]] * 3); q = np.stack([np.convolve(q[:, i], k, 'valid') for i in (0, 1)], -1); q[0], q[-1] = S[0], S[-1]; S = q
    return S


def rad(cel, clip, c, r, col, a=1., ry=None, rot=0., op=None, st=((0, 1.), (.35, .78), (.7, .28), (1, 0))):
    x = cel.ctx; x.save()
    if clip is not None: cel._path(clip); x.clip()
    if op is not None: x.set_operator(op)
    x.translate(*c); x.rotate(rot); x.scale(1, (ry or r) / r); g = cairo.RadialGradient(0, 0, 0, 0, 0, r); q = _c(col)
    for t, v in st: g.add_color_stop_rgba(t, float(q[0]), float(q[1]), float(q[2]), a * v)
    x.set_source(g); x.arc(0, 0, r, 0, 2 * math.pi); x.fill(); x.restore()


def lgrad(cel, clip, p0, p1, stops):
    x = cel.ctx; x.save()
    if clip is not None: cel._path(clip); x.clip()
    g = cairo.LinearGradient(*[float(v) for v in (*p0, *p1)])
    for t, col, a in stops: q = _c(col); g.add_color_stop_rgba(t, float(q[0]), float(q[1]), float(q[2]), a)
    x.set_source(g); x.paint(); x.restore()


# ------------------------------------------------------------------ the face
class Face:
    def __init__(s, cel, H, X, L, seed=0, hair=None, dmg=0., wet=0., lines=1., painterly=False, style=None):
        s.c, s.H, s.X, s.L, s.seed, s.hair, s.dmg, s.wet, s.painterly = cel, H, X, L, int(seed), hair, dmg, wet, painterly
        s.u = H.s; s.w = lw(H.s) * lines; s.ink = L['line']; s.F = H.face(); s.sg = -L['side']; s.style = style or X['style']

    def skin(s, i): return s.L['skin'][i]

    def ln(s, P, w, col=None, taper=(.3, .4), seed=0, **k):
        P = np.asarray(P, float)
        if 2 < len(P) <= 8: P = spl(P, n=8)
        s.c.line(P, max(.7, w * s.w), s.ink if col is None else col, taper=taper, seed=s.seed * 7 + seed, press=k.pop('press', .30), tremor=k.pop('tremor', .15) * min(1, s.u / 200), **k)

    def f(s, P, col, al=1., corners=None, smooth_=False, m=False):
        P = A(P, float)
        if smooth_: P = spl(P, True, corners=corners or (), n=6)
        s.c.fill((s.H.Pm if m else s.H.Pc)(P), col, al)

    # --- nose in 3-D
    def N(s, pts):
        w, yt, ytip, yb, d = s.H.k['nose']
        def g(y): return np.interp(y, [yb - .05, yb, ytip - .03, ytip, ytip + .03, (yt + ytip) / 2, yt, .10], [0, .30 * d, .82 * d, d, .93 * d, .55 * d, .18 * d, -.01])
        return s.H.P3(A([(x, y, float(s.H.z(x, y)) + g(y) * k) for x, y, k in pts]))

    # --- shadow
    def side_line(s, sg, amt):
        y = A([.80, .56, .36, .20, .08, -.02, -.12, -.20, -.28, -.42, -.54, -.64, -.74, -.84, -1.0])
        x0 = A([.60, .60, .63, .60, .53, .56, .58, .585, .53, .42, .39, .355, .30, .255, .20])          # side plane with the cheekbone break
        x1 = A([.03, .05, .07, .09, .04, .025, .020, .020, .022, .024, .015, .020, .030, .020, .035])     # half the face
        xm = A([.80, .79, .77, .76, .74, .74, .74, .73, .70, .640, .580, .520, .450, .380, .24])          # only the jaw edge
        t = float(np.clip(amt, -1, 1.1)); x = x0 * (1 - t) + x1 * t if t >= 0 else x0 * (1 + t) + xm * (-t); return np.c_[sg * x, y]

    def Pn(s, xy):
        """front coordinates lifted on to the ridge of the nose (a shadow line that runs along it)"""
        xy = A(xy, float); w, yt, ytip, yb, d = s.H.k['nose']; g = np.interp(xy[:, 1], [yb - .05, yb, ytip - .03, ytip, ytip + .03, (yt + ytip) / 2, yt, .10], [0, .30 * d, .82 * d, d, .93 * d, .55 * d, .18 * d, -.01])
        _, ys, lo, hi = s.H._xlim(); x = np.clip(xy[:, 0], np.interp(xy[:, 1], ys, lo), np.interp(xy[:, 1], ys, hi)); return s.H.P(np.c_[x, xy[:, 1]], g * np.clip(1 - np.abs(x) / .12, 0, 1))

    def shadow_polys(s):
        """list of front polygons in screen px: the hard shadow of this face"""
        H, X, L, sg = s.H, s.X, s.L, s.sg; amt = L['amt']; out = []
        sl = s.side_line(sg, amt); Q = s.Pn(resample(sl, .03)) if amt > .4 else H.Pc(sl, squeeze=False); d = A([sg * 3 * s.u * math.cos(H.roll), sg * 3 * s.u * math.sin(H.roll)]); out.append(np.vstack([Q, (Q + d)[::-1]]))
        F = lambda P: out.append(H.Pc(A(P, float)))
        # brow / socket wedge both sides
        B = A(X['B'], float); by = lambda x: np.interp(x, B[:, 0], B[:, 1]); U = A(X['U'], float)
        k = L.get('socket', .55)
        for q in (-1, 1):
            if not H.vis(q * .25, .02): continue
            sh = q == sg
            if k > .3 or sh: F([(q * a, b) for a, b in [(.070, by(.08) + .02), (.15, by(.15)), (.30, by(.30)), (.44, by(.44)), (.40, max(U[3] + .025, by(.40) - .055)), (.30, max(U[2] + .022, by(.30) - .070)), (.21, U[1] + .02), (.155, U[0] + .02), (.135, -.07), (.105, -.11), (.092, -.02)]])
        # nose: side plane + under plane
        kn = L.get('nose', .6); ytop = -.34 + .34 * kn
        if abs(H.yaw) < 1.2 and sg * H.yaw < .5 and amt < .6: out.append(s.N([(sg * .012, ytop, 1), (sg * .016, -.25, 1), (sg * .018, -.372, 1), (sg * .004, -.412, .9), (sg * .034, -.444, .52), (sg * .086, -.450, .2), (sg * (.080 + .02 * kn), -.415, .2), (sg * (.050 + .02 * kn), -.34, .42), (sg * (.034 + .014 * kn), -.22, .6), (sg * .030, (ytop - .22) / 2, .7)]))
        out.append(s.N([(-.072, -.436, .34), (-.02, -.424, .95), (.02, -.424, .95), (.072, -.436, .34), (.052, -.458, .25), (sg * .04, -.500, .04), (-.052, -.458, .25)]))
        # under lip, chin
        yb = X['my'] + min(X['mL']) - .050; a = A([(-.075 + sg * .014, yb + .012), (0, yb + .018), (.075 + sg * .014, yb + .012), (.048 + sg * .02, yb - .020), (sg * .016, yb - .030), (-.048 + sg * .02, yb - .020)]); a[:, 0] += X['mx']; out.append(H.Pm(a))
        return out

    def base(s):
        c, H, L = s.c, s.H, s.L
        if H.pitch > .05: c.fill(H.under(), s.skin(2) if L['amt'] > .5 else s.skin(1))
        c.fill(s.F, s.skin(0)); c.clip(s.F)
        for q in (-1, 1):
            if H.vis(.36 * q, -.22): rad(c, s.F, H.Pc([(.40 * q, -.27)])[0], .24 * s.u, L['blush'], .10 + .26 * s.X['blush'], ry=.15 * s.u, rot=H.roll)
        rad(c, s.F, s.N([(0, -.38, 1)])[0], .13 * s.u, L['blush'], .16)
        lgrad(c, s.F, H.P([(0, -.55)])[0], H.P([(0, -1.02)])[0], [(0, s.skin(1), 0.), (1, s.skin(1), .22)])
        for p in s.shadow_polys(): c.fill(p, s.skin(1))
        s.hair_cast(); s.deep(); c.unclip()

    def hair_cast(s):
        if s.hair is None: return
        H = s.H; k = s.L.get('cast', .075); ax = A([math.cos(H.roll), math.sin(H.roll)]); dn = A([-math.sin(H.roll), math.cos(H.roll)]); off = (ax * s.sg * .05 + dn * k) * H.s
        s.hair.cast(s.c, off, s.skin(1))

    def deep(s):
        """second, small shadow: under brow at the nose root, beside the nose, under the cheekbone, jaw edge"""
        H, sg, X = s.H, s.sg, s.X; c2 = s.skin(2); amt = s.L['amt']
        if s.u < 70: return
        if amt > -.2 and H.vis(sg * .3, -.2):
            s.f([(sg * a, b) for a, b in [(.80, -.34), (.64, -.50), (.50, -.70), (.34, -.90), (.24, -1.02), (.80, -1.02)]], c2, .50)
        for q in (-1, 1):
            if H.vis(q * .12, .02): s.f([(q * a, b) for a, b in [(.090, .05), (.135, .028), (.150, -.03), (.128, -.075), (.104, -.05)]], c2, .7 if q == sg else .45)

    # --- ears
    def ears(s, which=(-1, 1)):
        c, H, L = s.c, s.H, s.L
        for q in which:
            if H.ear(q)['vis'] <= -.05: continue
            ex = .720; z = -.10; sd = (q == s.sg) and L['amt'] > -.8
            def Q(L_, n=8): return spl(H.P3(A([(q * (ex + u * .70), v, z + .02 - u * 1.30) for u, v in L_])), n=n)
            o = Q([(-.02, .035), (.010, .100), (.055, .150), (.112, .150), (.158, .095), (.172, .010), (.156, -.085), (.124, -.175), (.098, -.245), (.080, -.315), (.048, -.365), (.014, -.360), (-.012, -.31), (-.02, -.24)]); cl = np.vstack([o, o[:1]])
            c.fill(cl, s.skin(0)); c.clip(cl)
            if sd: c.fill(Q([(-.05, .18), (.11, .17), (.130, .08), (.115, -.02), (.125, -.11), (.09, -.22), (.05, -.28), (.06, -.40), (-.05, -.40)]), s.skin(1))
            c.fill(Q([(.018, .035), (.055, .095), (.104, .098), (.126, .040), (.118, -.04), (.094, -.030), (.084, .030), (.052, .040), (.044, -.020), (.056, -.105), (.072, -.175), (.045, -.225), (.018, -.195), (.030, -.13), (.012, -.06)]), s.skin(2) if sd else s.skin(1)); c.unclip()
            s.c.line(o, s.w * .85, s.ink, taper=(.10, .12), press=.25, seed=s.seed + 3 + q, tremor=.15)
            if s.u > 45:
                s.ln(Q([(.018, .015), (.034, .070), (.072, .105), (.112, .088), (.130, .020), (.118, -.06), (.098, -.13)]), .42, seed=11 + q, taper=(.3, .5))
                s.ln(Q([(.050, .035), (.068, -.010), (.054, -.080), (.068, -.145), (.056, -.205), (.026, -.216)]), .42, seed=13 + q, taper=(.5, .4))

    # --- eyes
    def eye(s, q):
        c, H, L, X = s.c, s.H, s.L, s.X
        if not H.vis(.24 * q, -.02): return
        far = q * H.yaw > .25; Uv, Lv = A(X['U'], float), A(X['Lo'], float); xs = X['xi'] + (X['xo'] - X['xi']) * T5
        up = crv(xs, Uv, straight=.55); lo = crv(xs - A([0, .012, .010, -.004, -.014]), Lv, straight=.5); n = len(up); t = np.linspace(0, 1, n)
        Mx = lambda P: np.c_[q * np.asarray(P)[:, 0], np.asarray(P)[:, 1]]; Pu, Pl = H.Pc(Mx(up)), H.Pc(Mx(lo)); poly = np.vstack([Pu, Pl[::-1]])
        gap = float(np.max(up[:, 1] - lo[:, 1])); closed = gap < .034; arc = closed and Uv[2] > Uv[0] + .04; wh, wh2 = L['white']; sh = (q == s.sg) and L['amt'] > .5
        if sh and not X['glow']: wh = mixc(wh, wh2, .55)
        if not closed:
            c.fill(poly, wh); c.clip([poly])
            ix = X['xi'] + (X['xo'] - X['xi']) * .43 + q * X['gx']; iy = -.012 + X['gy']; rx = X['iris']; ry = rx * X['irish']; a = np.linspace(0, 2 * np.pi, 49)[:-1]
            E = lambda r1, r2, dx=0, dy=0: H.Pc(np.c_[q * ix + dx + np.cos(a) * r1, iy + dy + np.sin(a) * r2])
            if X['blank'] < .5:
                d0, d1, d2 = IRIS[s.style]
                if s.style == 'base':
                    c.fill(E(rx, ry), L['line'] if False else d0)
                    if s.u > 150: c.fill(E(rx * .80, ry * .50, 0, -ry * .40), d1); c.fill(E(rx * .50, ry * .24, 0, -ry * .62), d2, .8)
                elif s.style == 'ssj':
                    c.fill(E(rx * 1.04, ry * 1.04), mixc(d0, s.ink, .55)); c.fill(E(rx * .90, ry * .92), d1); c.fill(E(rx * .74, ry * .46, 0, -ry * .46), d2, .9)
                    c.clip([E(rx * .90, ry * .92)]); c.fill(E(rx * 1.1, ry * .62, 0, ry * .60), d0, .85); c.unclip(); c.fill(E(rx * .40, ry * .42), '#04201f')
                else:
                    c.fill(E(rx * 1.04, ry * 1.04), d0); c.fill(E(rx * .92, ry * .93), d1)
                    if s.u > 200:
                        r = np.random.default_rng(s.seed + 91 + q)
                        for j in range(26):
                            th = j / 26 * 6.283 + r.normal(0, .06); c.line(H.Pc([(q * ix + np.cos(th) * rx * .46, iy + np.sin(th) * ry * .46), (q * ix + np.cos(th) * rx * .86, iy + np.sin(th) * ry * .86)]), max(.7, .0045 * s.u), d0 if j % 3 else '#ffffff', taper=(.1, .6), seed=j, alpha=.6)
                    c.fill(E(rx * .60, ry * .60), d2, .9); c.fill(E(rx * .42, ry * .42), '#8d97ab'); c.fill(E(rx * .30, ry * .30), '#3c4252')
                th = prof(t, [(0, .010), (.3, .022), (.6, .030), (.9, .026), (1, .012)]) * (.6 + .4 * min(1.2, gap / .18))
                c.fill(H.Pc(Mx(np.vstack([up, np.c_[up[:, 0], up[:, 1] - th * (1.5 if X['furrow'] > .7 else 1.)][::-1]]))), wh2, .85 if not X['glow'] else .45)
                g = X['glint']
                if g > 0:
                    ls = L['side']; gc = '#ffffff'; gx0, gy0 = q * ix + ls * rx * .30, iy + ry * .34; k = rx / .072 * g * (.85 if s.style != 'ui' else .6)
                    c.fill(H.Pc(spl([(gx0 - .022 * k, gy0 - .006 * k), (gx0 - .004 * k, gy0 + .024 * k), (gx0 + .022 * k, gy0 + .010 * k), (gx0 + .010 * k, gy0 - .020 * k)], True, n=5)), gc)
                    if s.style != 'ui': c.fill(E(rx * .13, ry * .09, -ls * rx * .40, -ry * .48), gc, .75)
            else:
                th = prof(t, [(0, .010), (.5, .034), (1, .016)]); c.fill(H.Pc(Mx(np.vstack([up, np.c_[up[:, 0], up[:, 1] - th][::-1]]))), wh2, .7)
            if X['well'] > .02:
                hgt = .026 * X['well'] + .009 * X['well'] * np.sin(np.pi * t * 3) ** 2; top = np.c_[lo[:, 0], lo[:, 1] + hgt * np.sin(np.pi * np.clip(t, .03, .97)) ** .5]
                c.fill(H.Pc(Mx(np.vstack([lo, top[::-1]]))), mixc(wh, '#bfe0f5', .55), .5); c.line(H.Pc(Mx(top[6:-6])), max(.9, .008 * s.u), '#ffffff', taper=(.4, .4), seed=5, alpha=.9)
            c.unclip()
        # upper lid: thick, with the outer wall of the Dragon Ball eye
        ex, ey = up[-1]
        if arc: th = prof(t, [(0, .004), (.12, .014), (.45, .032), (.75, .034), (.93, .020), (1, .006)]) * X['lidw']
        elif closed: th = prof(t, [(0, .003), (.15, .010), (.5, .022), (.8, .026), (1, .010)]) * X['lidw']
        elif X['furrow'] > .7: th = prof(t, [(0, .006), (.12, .016), (.36, .022), (.60, .026), (.80, .030), (.95, .030), (1, .024)]) * X['lidw']
        else: th = prof(t, [(0, .006), (.12, .018), (.36, .026), (.62, .034), (.82, .042), (.95, .042), (1, .034)]) * X['lidw']
        Bk = np.c_[up[:, 0], up[:, 1] + th]
        if closed or far and abs(H.yaw) > .6: lid = np.vstack([Mx(Bk), Mx(up)[::-1]])
        else:
            o = X['out']; yl = lo[-1, 1]; lid = np.vstack([Mx(Bk), Mx(A([(ex + .022, ey + th[-1] - .004), (ex + .030, ey - o * .10), (ex + .022, ey - o * .75), (ex + .004, max(yl + .012, ey - o * 1.35)), (ex + .004, ey - o * .6), (ex - .006, ey - .006)])), Mx(up)[::-1]])
        c.fill(H.Pc(lid), s.ink)
        if not closed:
            m = len(lo); cl_ = mixc(s.ink, s.skin(2), .12)
            s.ln(H.Pc([(q * (up[0, 0] - .002), up[0, 1] + .004), (q * (up[0, 0] - .012), (up[0, 1] + lo[0, 1]) / 2), (q * (lo[2, 0] + .004), lo[2, 1] + .006)]), .50, seed=43 + q, taper=(.2, .5))
            s.ln(H.Pc(Mx(lo[int(m * .16):int(m * .62)])), .50, seed=45 + q, taper=(.6, .5), col=cl_)
            if not far: s.ln(H.Pc(Mx(lo[int(m * .66):])), .66, seed=46 + q, taper=(.4, .15), col=cl_)
        if X['crease'] > 0 and not arc:
            off = .032; cr = np.c_[up[:, 0] + .004, up[:, 1] + th + off * prof(t, [(0, .4), (.25, .7), (.6, 1.), (1, 1.1)])]; s.ln(H.Pc(Mx(cr[int(.26 * n):int(.86 * n)])), .42, seed=47 + q, taper=(.5, .35), col=mixc(s.ink, s.skin(2), .30), alpha=float(X['crease']))
        if X['under'] > 0:
            P = np.c_[lo[:, 0] - .01, lo[:, 1] - .044 - .012 * np.sin(np.pi * t)]; s.ln(H.Pc(Mx(P[int(.05 * n):int(.50 * n)])), .38, seed=48 + q, taper=(.3, .6), col=mixc(s.ink, s.skin(2), .35), alpha=float(X['under']))

    def brow_poly(s, q):
        B, th = A(s.X['B'], float), A(s.X['bt'], float); mid = crv(B[:, 0], B[:, 1], 60, straight=.62); n = len(mid); t = np.linspace(0, 1, n)
        # parameter along the five knots
        u = np.linspace(0, 4, n); w = np.interp(u, np.arange(5), th); T = np.gradient(mid, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
        up = mid + N * w[:, None]; dn = mid - N * (w * .82)[:, None]; i = A([(mid[0, 0] - .018, mid[0, 1] - .012)])
        return np.vstack([i, up, dn[::-1]]) * A([q, 1.])

    def brow(s, q):
        c, H, L, X = s.c, s.H, s.L, s.X; B = A(X['B'], float)
        if not H.vis(B[1, 0] * q, B[1, 1] - .04): return
        P = s.brow_poly(q); col = L['brow']; c.fill(H.Pc(P), col)
        if s.style in ('ssj', 'ui') and s.u > 60: outline(c, H.Pc(P), s.w * .55, s.L['hair']['line'], seed=s.seed + 63 + q)
        if s.u > 150:
            r = np.random.default_rng(s.seed + 61 + q); n = len(P) // 2
            for j in range(7):
                i = int(1 + (j + .3 * r.random()) / 7 * (n * .55)); b = P[i]; e = b + A([q * .030, .014]) * r.uniform(.5, 1.1) if i < 4 else b + A([q * .034, .010]) * r.uniform(.4, 1.)
                s.ln(H.Pc([b - A([q * .012, .004]), (b + e) / 2, e]), .40, col=col, seed=63 + j, taper=(.1, .8))

    def furrow(s):
        f_ = s.X['furrow']; H = s.H
        if abs(f_) < .05: return
        y0 = s.X['B'][0][1]; a = min(1, abs(f_) * 1.15)
        for q in (-1, 1):
            if not H.vis(.06 * q, .15): continue
            if f_ > 0:
                s.ln(H.Pc([(q * .050, y0 + .120), (q * .036, y0 + .070), (q * .040, y0 + .020), (q * .062, y0 - .020)]), .62, seed=71 + q, taper=(.55, .2), alpha=a)
                if f_ > .6: s.ln(H.Pc([(q * .006, y0 - .020), (q * .030, y0 - .044), (q * .062, y0 - .050)]), .44, seed=73 + q, taper=(.5, .5), alpha=a); s.ln(H.Pc([(q * .115, y0 + .130), (q * .085, y0 + .095), (q * .078, y0 + .060)]), .36, seed=74 + q, taper=(.6, .4), alpha=a)
            else:
                s.ln(H.Pc([(q * .070, y0 + .050), (q * .050, y0 + .010), (q * .062, y0 - .050)]), .56, seed=75 + q, taper=(.5, .4), alpha=a); s.ln(H.Pc([(q * .170, y0 + .070), (q * .100, y0 + .066), (q * .030, y0 + .084)]), .42, seed=77 + q, taper=(.6, .5), alpha=a)

    def nose(s):
        H, X = s.H, s.X; yw = H.yaw; d = 1 if yw >= 0 else -1; ay = abs(yw); ns = X['flare']; sg = s.sg
        if ay < .16:
            s.ln(s.N([(sg * .030, -.300, .62), (sg * .040, -.350, .50), (sg * .064, -.398, .22), (sg * .060, -.430, .22), (sg * .034, -.440, .55)]), .70, seed=80, taper=(.65, .3))
            s.ln(s.N([(-.030, -.432, .85), (-.004, -.440, .82), (.026, -.434, .85)]), .66, seed=81, taper=(.5, .5))
            for q in (-1, 1): s.c.fill(spl(s.N([(q * .026, -.440, .55), (q * .042, -.432, .48), (q * .062, -.438, .34), (q * .044, -.448, .38)]), True, corners=(0, 2), n=5), mixc(s.skin(2), s.ink, .75))
        elif ay < .95:
            pr = spl(s.H.P3(H.nose_profile()[1:7]), n=8); m = len(pr); s.ln(pr[int(m * .42):int(m * .90)], .95, seed=83, taper=(.6, .22))
            s.c.fill(spl(s.N([(-d * .010, -.438, .60), (-d * .036, -.428, .52), (-d * .064, -.436, .32), (-d * .040, -.450, .38)]), True, corners=(0, 2), n=5), mixc(s.skin(2), s.ink, .78))
            s.ln(s.N([(-d * .066, -.388, .22), (-d * .088, -.412, .12), (-d * .080, -.438, .18)]), .42, seed=84, taper=(.6, .3))
        else:
            pr = spl(s.H.P3(H.nose_profile()), n=8); m = len(pr); s.ln(pr[int(m * .04):int(m * .93)], 1.0, seed=83, taper=(.4, .3)); s.ln(s.N([(-d * .062, -.395, .22), (-d * .088, -.415, .14), (-d * .078, -.440, .20)]), .46, seed=84, taper=(.6, .3))
        if ns > .3:
            for q in (-1, 1):
                if q == d and ay > .30: continue
                s.ln(s.N([(q * .078, -.370, .18), (q * (.104 + .020 * ns), -.400, .10), (q * (.100 + .020 * ns), -.432, .14), (q * .074, -.444, .30)]), .52, seed=85 + q, taper=(.6, .25))
        if X['wrink'] > .05:
            for q in (-1, 1):
                for j, (y, l) in enumerate(((-.040, .085), (-.095, .100), (-.150, .080))):
                    if j == 2 and X['wrink'] < .8: continue
                    if H.vis(q * .08, y): s.ln(H.Pc([(q * .022, y + .026), (q * .054, y - .006), (q * (.052 + l), y + .016)]), .50 - .06 * j, seed=87 + j + q, taper=(.45, .6), alpha=float(min(1, X['wrink'] * 1.2)))

    def mouth(s):
        c, H, L, X = s.c, s.H, s.L, s.X; t5 = A([-1, -.5, 0, .5, 1.]); mx, my, hw = X['mx'], X['my'], X['hw']; Uv, Lv = A(X['mU'], float).copy(), A(X['mL'], float).copy()
        if X['asym']: Uv[4] += .020 * X['asym']; Lv[4] += .020 * X['asym']
        rot = X['mrot']; cr, sr = math.cos(rot), math.sin(rot); piv = my + (Uv[2] + Lv[2]) / 2
        def Rt(P): P = np.array(P, float); x, y = P[..., 0] - mx, P[..., 1] - piv; return np.stack([mx + x * cr - y * sr, piv + x * sr + y * cr], -1)
        cv = lambda v: np.c_[mx + hw * t5, my + v]
        st = .55 if X['clench'] > .5 or X['tL'] > .03 else .25
        U, Lw = Rt(crv(cv(Uv)[:, 0], cv(Uv)[:, 1], 64, st)), Rt(crv(cv(Lv)[:, 0], cv(Lv)[:, 1], 64, st)); n = len(U); t = np.linspace(0, 1, n); op = float(np.max(U[:, 1] - Lw[:, 1])); P = H.Pm
        if op > .012:
            side = lambda a, b, sg_: np.vstack([a, [((a[0] + b[0]) / 2) + A([sg_ * .012, 0])], b])
            pl = np.vstack([P(U), P([((U[-1] + Lw[-1]) / 2) + A([.010, 0])]), P(Lw)[::-1], P([((U[0] + Lw[0]) / 2) - A([.010, 0])])]); m0, m1, m2 = L['mouth']; deep = mixc(m0, m2, .55); tw0, tw1 = L['teeth']
            if X['clench'] > .5:
                c.fill(pl, tw0); c.clip([pl])
                # shadow of the upper lip, row line, few verticals near the corners
                sh = U.copy(); sh[:, 1] -= op * .20; c.fill(np.vstack([P(U), P(sh)[::-1]]), tw1)
                for sg_ in (-1, 1):
                    e = 0 if sg_ < 0 else n - 1; k_ = U[e, 0]; c.fill(P([(k_, U[e, 1] + .03), (k_ - sg_ * .05, my + .04), (k_ - sg_ * .030, my - .06), (k_, Lw[e, 1] - .03)]), mixc(tw1, deep, .55))
                mid = U * .52 + Lw * .48; mid[:, 1] += .004 * np.sin(t * 22); s.ln(P(mid), .55, seed=91, taper=(.15, .15), col=mixc(m2, s.ink, .5))
                if s.u > 110:
                    for j, u in enumerate((-.78, -.56, -.30, .02, .34, .60, .80)):
                        i = int((u + 1) / 2 * (n - 1)); a_, b_ = U[i], Lw[i]; s.ln(P([a_ * .92 + b_ * .08, a_ * .5 + b_ * .5]), .26, seed=92 + j, taper=(.2, .5), col=mixc(m0, s.ink, .3)); s.ln(P([a_ * .46 + b_ * .54, a_ * .14 + b_ * .86]), .22, seed=99 + j, taper=(.2, .6), col=mixc(m0, s.ink, .3), alpha=.8)
                c.unclip()
            else:
                c.fill(pl, deep); c.clip([pl])
                if X['tongue'] > 0 and op > .06:
                    a, b = int(n * .12), int(n * .88) + 1; tt = np.linspace(-1, 1, b - a); base = Lw[a:b]; h = op * (.40 * X['tongue']) * (1 - np.abs(tt) ** 2.6) ** .6 * (1 + .16 * np.cos(tt * 4.2 + .5 + X['tph'])); hump = np.c_[base[:, 0], base[:, 1] + h + .02 * op]
                    c.fill(np.vstack([P(base - A([0, .03])), P(hump)[::-1]]), m1); c.fill(np.vstack([P(base - A([0, .03])), P(base + np.c_[0 * h, h * .45])[::-1]]), mixc(m1, m0, .5)); i = len(tt) // 2; s.ln(P([hump[i] - A([0, .02 * op]), hump[i] - A([.004, h[i] * .5]), hump[i] - A([.0, h[i] * .85])]), .34, seed=96, col=mixc(m0, m1, .3), taper=(.2, .6))
                arch = lambda e, k: np.clip(1 - np.abs((2 * t - 1) / e) ** k, 0, 1) ** .5
                if X['tL'] > .004:
                    tb = Lw.copy(); tb[:, 1] += X['tL'] * arch(.84, 4.); i0, i1 = int(n * .09), int(n * .91) + 1; c.fill(np.vstack([P(Lw[i0:i1]), P(tb[i0:i1])[::-1]]), mixc(tw0, tw1, .55))
                    if s.u > 110:
                        s.ln(P(tb[i0:i1]), .30, seed=95, taper=(.3, .3), col=mixc(m0, s.ink, .45))
                        for j, u in enumerate((-.62, -.40, -.18, .05, .28, .50, .68)): i = int((u + 1) / 2 * (n - 1)); s.ln(P([tb[i], Lw[i] * .6 + tb[i] * .4]), .20, seed=195 + j, taper=(.1, .7), col=mixc(m0, s.ink, .3), alpha=.7)
                if X['tU'] > .004:
                    tb = U.copy(); tb[:, 1] -= X['tU'] * arch(.96, 5.) * (1 + .05 * np.sin(t * 40)); i0, i1 = int(n * .03), int(n * .97) + 1
                    c.fill(np.vstack([P(U[i0:i1]), P(tb[i0:i1])[::-1]]), tw0); sh_ = U.copy(); sh_[:, 1] -= X['tU'] * .24 * arch(.9, 4.); c.fill(np.vstack([P(U[i0:i1]), P(sh_[i0:i1])[::-1]]), tw1)
                    if X['gum'] > .1: gm = U.copy(); gm[:, 1] -= X['tU'] * .34 * X['gum'] * arch(.86, 3.); c.fill(np.vstack([P(U[i0:i1]), P(gm[i0:i1])[::-1]]), mixc(m1, m0, .45))
                    if X['tU'] > .03 and s.u > 110:
                        s.ln(P(tb[int(n * .06):int(n * .94)]), .36, seed=91, taper=(.3, .3), col=mixc(m0, s.ink, .45))
                        for j, u in enumerate((-.74, -.52, -.28, .0, .27, .50, .72)): i = int((u + 1) / 2 * (n - 1)); s.ln(P([U[i] * .3 + tb[i] * .7, tb[i]]), .22, seed=205 + j, taper=(.6, .1), col=mixc(m0, s.ink, .3), alpha=.75)
                        if X['gum'] > .3:
                            for sg_ in (-1, 1):                      # canines
                                i = int((sg_ * .60 + 1) / 2 * (n - 1)); b_ = tb[i]; c.fill(P([b_ + A([-.020, .006]), b_ + A([.0, -.020]), b_ + A([.020, .006])]), tw0)
                if X['gum'] > .5 and s.u > 150:
                    for q, x0, x1 in ((0, -.60, -.54), (1, .66, .58)):
                        i, j = int((x0 + 1) / 2 * (n - 1)), int((x1 + 1) / 2 * (n - 1)); a_, b_ = U[i] - A([0, X['tU']]), Lw[j] + A([0, X['tL']]); c.line(P(spl([a_, (a_ * .6 + b_ * .4) + A([.010, 0]), b_], n=10)), max(.8, s.w * .18), '#ffffff', taper=(.1, .1), seed=q, alpha=.55)
                c.unclip()
            s.ln(P(U), 1.05, seed=101, taper=(.10, .10), wfun=lambda q: 1.15 - .45 * np.sin(np.pi * q)); i0 = .05 if op > .10 or X['clench'] > .5 else .20; s.ln(P(Lw[int(n * i0):int(n * (1 - i0))]), .86 if op > .10 else .60, seed=102, taper=(.35, .35))
            for q, i in ((-1, 0), (1, n - 1)):
                if H.vis(U[i, 0], float(H.rest_y(U[i, 1]))): s.ln(P([(U[i, 0] - q * .008, U[i, 1] + .004), (U[i, 0] + q * .012, (U[i, 1] + Lw[i, 1]) / 2), (Lw[i, 0] - q * .004, Lw[i, 1] - .002)]), .85, seed=103 + q, taper=(.2, .4)); s.ln(P([(U[i, 0] + q * .006, U[i, 1] + .008), (U[i, 0] + q * .024, U[i, 1] - .004), (U[i, 0] + q * .030, U[i, 1] - .03)]), .42, seed=107 + q, taper=(.2, .7))
        else:
            mid = (U + Lw) / 2; s.ln(P(mid), .95, seed=101, taper=(.16, .16), wfun=lambda q: .72 + .50 * np.abs(q - .5) * 2)
            for q, i in ((-1, 0), (1, n - 1)):
                if H.vis(mid[i, 0], mid[i, 1]): s.ln(P([(mid[i, 0] - q * .026, mid[i, 1] - .002), (mid[i, 0], mid[i, 1]), (mid[i, 0] + q * .012, mid[i, 1] + (mid[i, 1] - mid[n // 2, 1]) * .35 - .006)]), .85, seed=103 + q, taper=(.5, .4))
        yb = piv + (min(Lv) + my - piv) - .050
        if X['lip'] > 0: s.ln(P([(mx - .056, yb + .006), (mx, yb - .004), (mx + .052, yb + .006)]), .50, seed=104, taper=(.5, .5), alpha=.9)
        if X['naso'] > .03:
            for q in (-1, 1):
                y_ = U[0, 1] if q < 0 else U[-1, 1]; xe = abs((U[0, 0] if q < 0 else U[-1, 0])) + .056; pts = [(q * .108, -.405), (q * (.11 + (xe - .11) * .45), -.480), (q * xe, y_ + .040), (q * (xe + .014), y_ - .06 - op * .30), (q * (xe - .02), y_ - .12 - op * .55)]; pr = H.Pm(spl(pts, n=10)); m = len(pr); e = int(m * (.45 + .5 * X['naso']))
                if H.vis(pts[2][0], -.6): s.ln(pr[int(m * .08):e], .60, seed=105 + q, taper=(.45, .55), alpha=min(1, .5 + X['naso'] * .6), col=mixc(s.ink, s.skin(2), .10))

    def lines(s):
        """cheekbone break, smile folds, veins, tendon of the jaw"""
        H, X = s.H, s.X
        if X['cheekline'] > .05:
            for q in (-1, 1):
                if H.vis(q * .52, -.40): s.ln(H.Pc([(q * .585, -.300), (q * .520, -.370), (q * .470, -.470)]), .48, seed=121 + q, taper=(.5, .5), alpha=min(1, X['cheekline']), col=mixc(s.ink, s.skin(2), .15)); s.ln(H.Pc([(q * .560, -.400), (q * .525, -.450), (q * .505, -.500)]), .34, seed=123 + q, taper=(.5, .6), alpha=X['cheekline'] * .8, col=mixc(s.ink, s.skin(2), .2))
        if X['cheek'] > .05:
            for q in (-1, 1):
                if H.vis(q * .40, -.2): s.ln(H.Pc([(q * .250, -.175), (q * .340, -.200), (q * .430, -.178)]), .36, seed=125 + q, taper=(.5, .5), alpha=min(1, X['cheek'] * .7))
        if X['vein'] > .3:
            for q in (-1, 1):
                if H.vis(q * .46, .36):
                    s.ln(H.Pc([(q * .40, .52), (q * .455, .450), (q * .430, .395), (q * .49, .340), (q * .475, .30)]), .46, seed=131 + q, taper=(.3, .5)); s.ln(H.Pc([(q * .455, .450), (q * .515, .440), (q * .55, .46)]), .36, seed=133 + q, taper=(.1, .7)); s.ln(H.Pc([(q * .430, .395), (q * .385, .375), (q * .36, .34)]), .34, seed=135 + q, taper=(.1, .7))
            if X['vein'] > .8 and H.vis(-s.sg * .62, .05): q = -s.sg; s.ln(H.Pc([(q * .655, .20), (q * .64, .12), (q * .67, .06), (q * .65, -.02)]), .40, seed=137, taper=(.3, .5))

    def contour(s):
        H, F = s.H, s.F; near = -1 if H.yaw > 0 else 1; ax = A([math.sin(H.roll), -math.cos(H.roll)]); cen = H.P3([(0, 0, 0)])[0]; v = (F - cen) @ ax / s.u; keep = v < .50 * math.cos(H.pitch) + .35 * math.sin(H.pitch)
        if abs(H.yaw) > .10:
            lob = H.P3([(near * .70, -.34, -.10)])[0]; rt = A([math.cos(H.roll), math.sin(H.roll)]); e = (F - lob) @ rt * near / s.u; keep &= ~((v > -.36) & (e > -.14))
        n = len(F); i0 = int(np.argmin(keep)) if not keep.all() else 0; K = np.r_[keep[i0:], keep[:i0]]; G = np.r_[F[i0:], F[:i0]]; runs = []; j = 0
        while j < n:
            if K[j]:
                e = j
                while e + 1 < n and K[e + 1]: e += 1
                if e - j > 6: runs.append(G[j:e + 1])
                j = e + 1
            else: j += 1
        ld = A([s.L['side'] * .6, -.8])
        for r, g in enumerate(runs):
            k = np.hanning(9); k /= k.sum(); pad = np.vstack([g[:1]] * 4 + [g] + [g[-1:]] * 4); g = np.stack([np.convolve(pad[:, i], k, 'valid') for i in (0, 1)], -1); g = resample(g, 1.6); T = np.gradient(g, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[T[:, 1], -T[:, 0]]
            if ((g + N * 3 - cen) ** 2).sum() < ((g - cen) ** 2).sum(): N = -N
            hv = .78 + .52 * np.clip(-(N @ ld), -.75, 1); kk = np.hanning(61); kk /= kk.sum(); hv = np.convolve(np.r_[np.full(30, hv[0]), hv, np.full(30, hv[-1])], kk, 'valid') if len(hv) > 4 else hv
            s.c.line(g, s.w, s.ink, taper=(.12, .12), press=.42, seed=s.seed + 121 + r, tremor=.2, wfun=lambda q, hv=hv: np.interp(q, np.linspace(0, 1, len(hv)), hv))

    # --- water, tears, dirt
    def stream(s, pts, w, seed=0, al=.62, col='#a9cdf0'):
        c, H, L = s.c, s.H, s.L; P = spl(A(pts, float), n=10); n = len(P); t = np.linspace(0, 1, n); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
        wd = w * (.55 + .30 * np.sin(t * 9 + seed) ** 2) * np.clip(t * 6, .25, 1); wd[-6:] *= A([1.2, 1.45, 1.55, 1.35, .95, .35]); Q = H.Pc(np.vstack([P + N * wd[:, None], (P - N * wd[:, None])[::-1]]))
        c.fill(Q, mixc(L['white'][0], col, .30), al); c.line(H.Pc(P + N * (wd * .95)[:, None]), max(.8, s.w * .20), mixc(s.skin(2), s.ink, .35), taper=(.5, .1), seed=seed, alpha=.8); c.line(H.Pc((P - N * (wd * .35)[:, None])[int(n * .10):int(n * .55)]), max(.8, s.w * .22), '#ffffff', taper=(.3, .6), seed=seed + 1, alpha=.95)

    def bead(s, x, y, r, ang=0.):
        H = s.H; c = s.c; u = r; q = np.linspace(0, 2 * np.pi, 33)[:-1]; ang = ang - H.roll; ca, sa = math.cos(ang), math.sin(ang); cen = H.Pc([(x, y)])[0]
        def D(k, ox=0., oy=0., sy=1.):
            X_, Y_ = u * k * 1.02 * np.sin(q) * np.abs(np.sin(q / 2)) ** 1.1 + ox * u, u * k * 1.45 * sy * np.cos(q) + oy * u; return np.c_[cen[0] + (X_ * ca - Y_ * sa) * H.s, cen[1] - (X_ * sa + Y_ * ca) * H.s]
        c.fill(D(1.05, s.sg * .35, -.30), mixc(s.skin(2), s.ink, .25), .45); c.fill(D(1.), mixc(s.skin(1), s.skin(2), .35), .85); c.fill(D(.80, s.sg * .06, -.34, .70), mixc(s.L['white'][0], s.skin(0), .30), .90); Q = D(1.); c.line(Q[6:27], max(.8, s.w * .30), s.ink, taper=(.35, .35), seed=s.seed + int(abs(x) * 99), alpha=.85)
        e = cen + A([-s.sg * .30, -.28]) * u * H.s; rr = u * H.s * .20; c.fill(np.c_[e[0] + np.cos(q) * rr * .7, e[1] + np.sin(q) * rr], '#ffffff', .95)

    def tearlines(s):
        X, H = s.X, s.H; k = X['tears']
        if k <= .02: return
        for q in (-1, 1):
            if not H.vis(q * .30, -.2): continue
            y0 = float(np.mean(A(X['Lo'], float)[1:4])); a = [(q * .230, y0 + .004), (q * .215, y0 - .12), (q * .210, -.36), (q * .255, -.50), (q * .310, -.64), (q * .290, -.80), (q * .215, -.93)]; b = [(q * .440, y0 + .012), (q * .470, y0 - .10), (q * .480, -.34), (q * .455, -.52), (q * .400, -.70)]
            s.stream(a[:max(3, int(round(len(a) * min(1, k + .15))))], .0165, 3 + q, .52)
            if k > .45: s.stream(b[:max(3, int(round(len(b) * k)))], .011, 5 + q, .48)

    def rainskin(s, ymax=9.):
        H, c = s.H, s.c; r = np.random.default_rng(s.seed * 5 + 17); ls = s.L['side']; g = s.skin(3)
        for P, al in (([(ls * .012, -.06), (ls * .030, -.20), (ls * .026, -.35), (ls * .006, -.24)], .9), ([(ls * .02, -.895), (ls * .12, -.885), (ls * .20, -.860), (ls * .10, -.905)], .7), ([(ls * .30, .150), (ls * .42, .156), (ls * .50, .12), (ls * .40, .135)], .6), ([(ls * .40, -.16), (ls * .52, -.19), (ls * .58, -.26), (ls * .50, -.215)], .8)):
            if H.vis(*P[1]) and P[0][1] < ymax: c.fill(H.Pc(spl(P, True, corners=(0, 2), n=6)), g, al * s.wet)
        for j, (x, y, l) in enumerate(((-.04, .42, .11), (.22, .46, .07), (-.30, .40, .05), (.50, .30, .10), (-.56, .16, .12), (.07, .30, 0), (-.18, .33, 0), (.52, -.36, .10), (-.56, -.30, .14), (-.50, -.62, 0), (.46, -.56, 0), (.60, -.20, 0), (-.60, -.06, 0), (.04, -.885, 0), (-.16, -.905, .06), (.24, -.84, 0), (.10, -.22, 0), (-.08, -.30, 0), (-.37, -.50, 0), (.36, -.36, 0), (.015, -.985, .03))):
            if j / 21. > s.wet or y > ymax or not H.vis(x * 1.05, y): continue
            if l:
                P = spl(H.Pc([(x, y), (x + r.normal(0, .006), y - l * .33), (x + r.normal(0, .005), y - l * .66), (x + r.normal(0, .008), y - l)]), n=8); c.line(P, max(.9, .011 * s.u), mixc(s.L['white'][0], s.skin(0), .25), taper=(.6, .08), seed=j, alpha=.8); c.line(P + A([.008 * s.u, 0]), max(.7, .005 * s.u), s.skin(2), taper=(.5, .2), seed=j + 1, alpha=.6)
                e = H.Pc([(x, y - l)])[0]; k = .011 * s.u; a = np.linspace(0, 2 * np.pi, 11)[:-1]; c.fill(np.c_[e[0] + np.cos(a) * k * .7, e[1] + np.sin(a) * k], mixc(s.L['white'][0], s.skin(0), .2), .9)
            else:
                k = r.uniform(.007, .012); a = np.linspace(0, 2 * np.pi, 11)[:-1]; c.fill(H.Pc(np.c_[x + np.cos(a) * k * .8, y + np.sin(a) * k * 1.2]), mixc(s.L['white'][0], s.skin(0), .35), .85); c.fill(H.Pc(np.c_[x + k * .3 + np.cos(a) * k * .5, y - k * .55 + np.sin(a) * k * .45]), s.skin(2), .55)

    def damage(s):
        H, c, d = s.H, s.c, s.dmg; r = np.random.default_rng(404); dirt = mixc(s.skin(1), '#3a2f2c', .45)
        for k, (x, y, w, h, a) in enumerate(((.44, -.47, .12, .06, .6), (-.50, -.30, .10, .06, -.5), (.22, .40, .12, .04, .15), (-.12, -.87, .09, .035, -.1), (.56, .12, .045, .08, .2))):
            if k / 5 > d or not H.vis(x, y): continue
            ca, sa = math.cos(a), math.sin(a); R2 = lambda q: np.c_[x + q[:, 0] * ca - q[:, 1] * sa, y + q[:, 0] * sa + q[:, 1] * ca]
            for part in range(3):
                m = 9; q = np.linspace(0, 2 * np.pi, m, endpoint=False) + r.uniform(0, 1); rr = r.uniform(.45, 1.15, m); ox, oy, sc = r.normal(0, w * .55), r.normal(0, h * .55), r.uniform(.28, .62); c.fill(H.Pc(R2(np.c_[ox + np.cos(q) * w * sc * rr, oy + np.sin(q) * h * sc * rr])), dirt, .34)
            for j in range(2):
                oy = r.normal(0, h * .4); l = w * r.uniform(.6, 1.2); ox = r.normal(0, w * .3); c.line(H.Pc(spl(R2(A([[ox - l, oy], [ox, oy + h * r.normal(0, .25)], [ox + l, oy + h * r.normal(0, .3)]])), n=8)), max(.8, .008 * s.u), dirt, taper=(.5, .6), seed=j + k * 9, alpha=.6)
        # the cheek scratch of d6 (viewer's left cheek)
        P = [(-.30, -.225), (-.39, -.262), (-.49, -.318)]
        if H.vis(-.42, -.28): c.line(H.Pc(spl(P, n=10)), max(1, .016 * s.u), mixc('#a8303a', s.skin(2), .30), taper=(.35, .5), seed=9, press=.4); c.line(H.Pc(spl(A(P) + A([.02, -.034]), n=10)), max(.8, .008 * s.u), mixc('#a8303a', s.skin(2), .45), taper=(.5, .6), seed=10, press=.3)

    def features(s):
        far = abs(s.H.yaw) < .75
        if far: s.c.clip(s.F)
        for q in (-1, 1): s.eye(q)
        s.mouth()
        if s.dmg > .02: s.damage()
        if s.wet > .05: s.rainskin()
        s.tearlines()
        if far: s.c.unclip()
        s.nose()
        for q in (-1, 1): s.brow(q)
        s.furrow(); s.lines()
        X, H = s.X, s.H
        if X['sweat'] > .05: d = 1 if H.yaw >= 0 else -1; s.bead(-d * .52, .10, .030 * X['sweat']); s.bead(d * .30 if H.vis(d * .30, .35) else -d * .1, .34, .020 * X['sweat'])
