"""FRIENDS kit: the drawing hand shared by Gohan, Piccolo, Bulma, Krillin and Roshi (chars/gohan.py, piccolo.py, friends.py).
    H = FHead(design).pose(cx, cy, s, yaw, pitch, roll)        FHead.of(any db.head3.Head, design) adopts another head's pose
    Face(cel, H, design, expr, light, seed)  .ears_far() .base() .features() .ears_near() .contour()
    Rev(H, profile) surface of revolution round the head (hat, turban, scarf): sil() curve() patch()
    Lock / hair_group = 3-D hair blades;  LIGHT[name] = light set-ups;  tone(L, triple, i) = a local colour in that light
Everything is authored in head front coordinates (x right, y up, eye line 0, chin -1, skull top +1) and projected.
"""
import numpy as np, cv2, math, cairo
from db import W, H as FH
from db.head3 import Head, DEF
from db.ink import Cel, spl, resample, ribbon, noise1
from db.kit import outline, shape, spike, contours, rim, para, tint, mask_of, hatch, aura, shifted
from db.paint import rgb, f32, smooth, mix, fbm
from db import comp

A = np.array
INK = '#17131a'
TAU = 2 * math.pi


def c_(c): return rgb(c) if isinstance(c, str) else np.asarray(c, f32)
def mixc(a, b, t): return c_(a) * (1 - t) + c_(b) * t


def lw(s):
    """outer contour, inner line, fine line for a head of s px per unit"""
    o = float(np.interp(math.log(max(s, 6)), [math.log(v) for v in (12, 30, 60, 110, 200, 330, 700)], [1.0, 1.4, 2.1, 3.0, 4.1, 5.3, 7.0]))
    return o, max(.8, o * .56), max(.7, o * .36)


# ------------------------------------------------------------------ light set-ups
# mul = multipliers for (lit, shadow, deep) authored in daylight; ld = direction the key comes FROM on screen (x right, y up); rim list, para stops
def _l(**k): return k


LIGHT = dict(
    day=_l(mul=((1, 1, 1), (1, 1, 1), (1, 1, 1)), ld=(-.62, .70), line=INK, rim=[], white=('#ffffff', '#cdd3e6'), para=[('#fff8ea', 0), ('#ffffff', .5), ('#d9d2ea', 1)], pamt=.6, side=-.35),
    lake=_l(mul=((1.0, .975, .88), (.93, .97, .88), (.86, .92, .88)), ld=(-.70, .60), line='#1e1512', rim=[((4, 3), '#fff2b4', .85)], white=('#fffdf0', '#c7d0b8'), para=[('#fff6cf', 0), ('#ffffff', .45), ('#b9c7b0', 1)], pamt=.85, side=-.15),
    pearl=_l(mul=((.86, .80, .74), (.72, .72, .78), (.60, .64, .78)), ld=(.1, -.9), line='#1a1420', rim=[((6, 4), '#ffeeb0', .9)], white=('#eaf4ff', '#8fa8cc'), para=[('#e8d9b8', 0), ('#d9d3cf', .5), ('#c4e4ff', 1)], pamt=.9, side=-1.),
    night=_l(mul=((.60, .55, .82), (.44, .40, .74), (.36, .31, .64)), ld=(.55, .75), line='#0b0816', rim=[((-6, 4), '#c68cff', 1.)], white=('#d8cdf6', '#6d5fa8'), para=[('#e6d6ff', 0), ('#c9bce6', .4), ('#5c4f94', 1)], pamt=.9, side=.0),
    blast=_l(mul=((.80, .72, .90), (.50, .44, .74), (.36, .31, .60)), ld=(.95, .15), line='#0d0818', rim=[((-9, 1), '#fbeaff', 1.)], white=('#f6e8ff', '#8a78bc'), para=[('#ffffff', 0), ('#d7c0f2', .35), ('#4a3a7c', 1)], pamt=.95, side=.35),
    rain=_l(mul=((.66, .68, .72), (.56, .58, .68), (.48, .50, .62)), ld=(-.4, .85), line='#131318', rim=[((4, 5), '#e2eaf2', .7)], white=('#dfe3e8', '#8f959f'), para=[('#f2f5f8', 0), ('#d6d9de', .5), ('#7e828c', 1)], pamt=.9, side=-.1),
    dawn=_l(mul=((.94, .83, .86), (.70, .64, .92), (.56, .52, .84)), ld=(.80, .42), line='#1a1226', rim=[((-6, 3), '#ffcfa0', 1.)], white=('#fdf0ea', '#9f9cc6'), para=[('#ffe0c8', 0), ('#f0e2ee', .4), ('#7c84b8', 1)], pamt=.95, side=.05),
    gold=_l(mul=((1.0, .95, .84), (.94, .84, .80), (.86, .74, .76)), ld=(.80, .50), line='#24131a', rim=[((-6, 3), '#fff0c0', 1.)], white=('#fff8ea', '#d6b8b0'), para=[('#fff0cc', 0), ('#ffffff', .5), ('#c3a9d0', 1)], pamt=.8, side=-.15),
)


def light(l, **over):
    L = dict(LIGHT[l] if isinstance(l, str) else l); L.update(over); return L


def flip(L):
    """mirror a light set-up left-right"""
    L = dict(L); L['ld'] = (-L['ld'][0], L['ld'][1]); L['rim'] = [((-d[0], d[1]), c, a) for d, c, a in L['rim']]; return L


def tone(L, tri, i=0):
    """colour of a (lit, shadow, deep) triple in light L"""
    if isinstance(tri, str): tri = (tri, tri, tri)
    i = min(i, len(tri) - 1); return np.clip(c_(tri[i]) * A(L['mul'][min(i, 2)], f32), 0, 1)


# ------------------------------------------------------------------ head
class FHead(Head):
    """Head + child proportions + a jaw that opens + features that never fold over the far silhouette"""
    jopen = 0.; low = 1.; hi = 1.; warp = None; chin = 0.

    def __init__(s, D=None, **kw):
        k = dict((D or {}).get('head', {})); k.update(kw); s.low = k.pop('low', 1.); s.hi = k.pop('hi', 1.); s.zj = k.pop('zj', None); s.relief = k.pop('relief', 1.); s._xs = None
        Head.__init__(s, **k)

    @classmethod
    def of(cls, H, D, jaw=0.):
        """adopt the pose of any Head"""
        q = cls(D).pose(H.cx, H.cy, H.s, H.yaw, H.pitch, H.roll, H.persp); q.jopen = jaw; q.warp = getattr(H, 'warp', None); return q

    def pose(s, *a, **k): s._xs = None; return Head.pose(s, *a, **k)

    def P3(s, Q):
        Q = np.array(Q, float); y = Q[..., 1]
        if s.jopen:
            w = smooth(-.52, -.92, y) * smooth(-.25, .40, Q[..., 2]); Q[..., 1] = y - s.jopen * w; Q[..., 2] = Q[..., 2] - s.jopen * .22 * w; y = Q[..., 1]
        if s.low != 1. or s.hi != 1.: Q[..., 1] = np.where(y < 0, y * s.low, y * s.hi)
        P = Head.P3(s, Q); return P if s.warp is None else s.warp(P)

    def _mid(s, x, y): return Head._mid(s, x, y) * s.relief

    def _mesh(s, y0, y1, half, ny=90, nf=90):
        """as Head._mesh with the jaw cut given by the design (zj = depth of the jaw edge at each height)"""
        if not half or s.zj is None: return Head._mesh(s, y0, y1, half, ny, nf)
        y = np.linspace(y0, y1, ny); Zt = A(s.zj, float); zj = np.interp(y, Zt[:, 0], Zt[:, 1]); a, f, b = (np.interp(y, s.yy, t) for t in (s.A, s.F, s.B)); p = s.k['p']
        c = np.clip(zj / np.maximum(np.where(zj >= 0, f, b), 1e-3), -1, .98); t = np.where(c >= 0, np.arctan2((1 - np.abs(c) ** p) ** (1 / p), np.abs(c)), np.pi / 2 + np.arcsin(np.clip(-c, 0, 1)))
        PHI = np.linspace(-1, 1, nf)[None] * np.minimum(math.pi * .5 * half, t)[:, None]; sx, cz = np.sin(PHI), np.cos(PHI); r = (np.abs(sx) ** p + np.abs(cz) ** p) ** (-1 / p)
        Xx = a[:, None] * sx * r; Z = np.where(cz >= 0, f[:, None], b[:, None]) * cz * r; Y = np.repeat(y[:, None], nf, 1); fr = np.clip(cz, 0, 1)
        Z = Z + fr * (s._mid(Xx, Y) + s.k['cheek'] * np.exp(-((Y + .22) / .13) ** 2) * np.exp(-((np.abs(Xx) - .44) / .2) ** 2) * 1.4 - s.k['socket'] * np.exp(-((Y - .02) / .10) ** 2) * np.exp(-((np.abs(Xx) - .40) / .24) ** 2) * 1.3)
        return np.stack([Xx, Y, Z], -1)

    def face(s, top=.75):
        F = Head.face(s, top)
        n = max(2, int((.05 if abs(s.yaw) < .75 else .022) * s.s / 3.0)) * 2 + 1; k = np.hanning(n + 2)[1:-1]; k /= k.sum(); m = n // 2
        return np.stack([np.convolve(np.r_[F[-m:, i], F[:, i], F[:m, i]], k, 'valid') for i in (0, 1)], -1)

    def _xlim(s):
        key = (s.yaw, s.pitch, s.roll)
        if s._xs is not None and s._xs[0] == key: return s._xs
        ys = np.linspace(-.99, .9, 40); lo, hi = [], []; V = s.R[2]
        for y in ys:
            a = np.interp(y, s.yy, s.A); x = np.linspace(-a * .985, a * .985, 121); e = 4e-3
            z0 = s.z(x, y); nx = -(s.z(x + e, y) - s.z(x - e, y)) / (2 * e); ny = -(s.z(x, y + e) - s.z(x, y - e)) / (2 * e)
            c = (nx * V[0] + ny * V[1] + V[2]) / np.sqrt(nx * nx + ny * ny + 1); ok = np.where(c > .03)[0]
            if len(ok) == 0: lo.append(0); hi.append(0); continue
            j = int(np.argmax(c)); l = j
            while l > 0 and c[l - 1] > .03: l -= 1
            r = j
            while r < len(x) - 1 and c[r + 1] > .03: r += 1
            lo.append(x[l] if l > 0 else -9.); hi.append(x[r] if r < len(x) - 1 else 9.)
        s._xs = (key, ys, A(lo), A(hi)); return s._xs

    def Pc(s, xy, dz=0., squeeze=True):
        xy = np.array(xy, float); _, ys, lo, hi = s._xlim(); l = np.interp(xy[..., 1], ys, lo); h = np.interp(xy[..., 1], ys, hi); x = xy[..., 0]
        if squeeze:
            k = .10
            lm = np.maximum(h * .955 - k, .02); d = np.maximum(x - k, 0); x = np.where(x > k, k + lm * np.tanh((d / lm) ** 1.25) ** .8, x)
            lm = np.maximum(-l * .955 - k, .02); d = np.maximum(-x - k, 0); x = np.where(x < -k, -k - lm * np.tanh((d / lm) ** 1.25) ** .8, x)
        else: x = np.clip(x, l, h)
        return s.P(np.stack([x, xy[..., 1]], -1), dz)

    def under(s):
        """underside of the chin: a shallow crescent that runs into the neck, never a second chin"""
        a, b = s.jaw(-1), s.jaw(1); k = len(a) // 3; back = s.P3(A([(.40, -.62, .00), (.22, -.80, .16), (0, -.84, .22), (-.22, -.80, .16), (-.40, -.62, .00)])); return np.vstack([a[k:], b[::-1][1:len(b) - k], back])

    def vis(s, x, y):
        _, ys, lo, hi = s._xlim(); return np.interp(y, ys, lo) < x < np.interp(y, ys, hi)

    def rest_y(s, y):
        g = np.linspace(-1.3, .5, 260); e = g - s.jopen * smooth(-.52, -.92, g); return np.interp(y, e, g)

    def Pm(s, xy, dz=0.):
        """mouth points in final front coordinates (jaw already open)"""
        xy = np.array(xy, float); xy[..., 1] = s.rest_y(xy[..., 1]); return s.Pc(xy, dz - 2.2 * xy[..., 0] ** 2 * float(smooth(.55, 1.25, abs(s.yaw))))

    def nose3(s, y):
        w, yt, ytip, yb, d = s.k['nose']
        return np.interp(y, [yb - .05, yb, ytip - .03, ytip, ytip + .03, (yt + ytip) / 2, yt, .10], [0, .30 * d, .82 * d, d, .93 * d, .55 * d, .18 * d, -.01])

    def N(s, pts): return s.P3(A([(x, y, float(s.z(x, y)) + float(s.nose3(y)) * k) for x, y, k in pts]))


# ------------------------------------------------------------------ small drawing helpers
T5 = A([0, .22, .50, .78, 1.])


def crv(xs, ys, n=48):
    return spl(np.c_[xs, ys], n=max(4, n // (len(xs) - 1)))


def prof(t, pts): p = A(pts, float); return np.interp(t, p[:, 0], p[:, 1])


def rad(cel, clip, c, r, col, a=1., ry=None, rot=0.):
    """airbrush dab"""
    x = cel.ctx; x.save()
    if clip is not None: cel._path(clip); x.clip()
    x.translate(float(c[0]), float(c[1])); x.rotate(rot); x.scale(1, (ry or r) / r); g = cairo.RadialGradient(0, 0, 0, 0, 0, r); q = c_(col)
    for t, v in ((0, 1.), (.35, .78), (.7, .28), (1, 0)): g.add_color_stop_rgba(t, float(q[0]), float(q[1]), float(q[2]), a * v)
    x.set_source(g); x.arc(0, 0, r, 0, TAU); x.fill(); x.restore()


def lgrad(cel, clip, p0, p1, stops):
    x = cel.ctx; x.save()
    if clip is not None: cel._path(clip); x.clip()
    g = cairo.LinearGradient(*[float(v) for v in (*p0, *p1)])
    for t, col, a in stops: q = c_(col); g.add_color_stop_rgba(t, float(q[0]), float(q[1]), float(q[2]), a)
    x.set_source(g); x.paint(); x.restore()


def clip_eo(cel, polys):
    """even-odd clip (shapes with holes); close with cel.unclip()"""
    cel.ctx.save(); cel._path(polys); cel.ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD); cel.ctx.clip(); cel.ctx.set_fill_rule(cairo.FILL_RULE_WINDING)


def ell(cx, cy, rx, ry=None, n=40, rot=0., a0=0., a1=TAU):
    t = np.linspace(a0, a1, n + 1)[:-1] if a1 - a0 > 6.28 else np.linspace(a0, a1, n); ry = rx if ry is None else ry; c, s = math.cos(rot), math.sin(rot)
    x, y = np.cos(t) * rx, np.sin(t) * ry; return np.c_[cx + x * c - y * s, cy + x * s + y * c]


def wedge(P, th0, th1=None, sharp=.75):
    """tapered polygon along a polyline (thickness th0 px at the start, th1 at the end or a profile function)"""
    P = np.asarray(P, float); n = len(P); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]; t = np.linspace(0, 1, n)
    w = th0(t) if callable(th0) else th0 + ((0. if th1 is None else th1) - th0) * t ** sharp
    return np.vstack([P + N * w[:, None] / 2, (P - N * w[:, None] / 2)[::-1]])


# ------------------------------------------------------------------ expressions: lids are heights at T5 inner -> outer
X0 = dict(U=None, Lo=None, UL=None, UR=None, LoL=None, LoR=None, ir=1., gx=0., gy=0., glint=1., B=None, BL=None, BR=None, dB=0., aB=0., furrow=0., closed=0., arc=0.,
          mx=0., my=None, hw=.12, mU=[0, .002, .004, .002, 0], mL=[0, .002, .004, .002, 0], teeth=0., loteeth=0., tongue=0., jaw=0., wob=0., lip=1., mw=1., fang=0.,
          blush=0., sweat=0., tears=0., dark=0., naso=0., cheek=0., squint=0., white=0., under=0., oneeye=None, flare=0., clench=0.)


def X(**k): d = dict(X0); d.update(k); return d


def blend(a, b, t):
    o = {}
    for k in X0:
        u, v = a.get(k), b.get(k)
        if u is None or v is None: o[k] = v if t > .5 else u; continue
        if isinstance(u, str) or isinstance(v, str): o[k] = v if t > .5 else u; continue
        o[k] = (A(u, float) * (1 - t) + A(v, float) * t) if np.ndim(u) else u * (1 - t) + v * t
    return o


# ------------------------------------------------------------------ face
class Face:
    def __init__(s, cel, H, D, x, L, seed=0, wet=0., hairs=(), cast=None):
        s.c, s.H, s.D, s.L, s.seed, s.wet = cel, H, D, L, int(seed), wet
        e = dict(X0); e.update(D.get('rest', {})); e.update({k: v for k, v in x.items() if v is not None}); s.X = e
        if e['my'] is None: e['my'] = D.get('my', -.66)
        s.u = H.s; s.wo, s.wi, s.wf = lw(H.s); s.ink = L['line']; s.F = H.face(); s.hairs = hairs; s.cast = cast
        s.sg = -1 if L['ld'][0] > 0 else 1                           # shadow side (+1 = viewer's right)

    def sk(s, i): return tone(s.L, s.D['skin'], i)

    def ln(s, P, w=1., col=None, taper=(.3, .4), seed=0, **k):
        P = np.asarray(P, float)
        if 2 < len(P) <= 7: P = spl(P, n=8)
        s.c.line(P, max(.7, w * s.wi), s.ink if col is None else col, taper=taper, seed=s.seed * 7 + seed, press=k.pop('press', .25), tremor=k.pop('tremor', .15) * min(1, s.u / 200), **k)

    # ---- flats and cel shadows
    def base(s):
        c, H, L, D, X_ = s.c, s.H, s.L, s.D, s.X
        if H.pitch > .05: c.fill(H.under(), s.sk(1))
        c.fill(s.F, s.sk(0)); c.clip(s.F); sh = s.sk(1)
        lgrad(c, s.F, H.P([(0, -.40)])[0], H.P([(0, -1.02)])[0], [(0, sh, 0.), (1, sh, .28)])
        b = X_['blush'] + D.get('blush0', .10)
        for q in (-1, 1):
            if H.vis(.40 * q, -.25) and b > 0: rad(c, s.F, H.Pc([(.46 * q, -.30)])[0], .24 * s.u, tone(L, D.get('blushc', '#f08a78'), 0), min(.85, .6 * b), ry=.15 * s.u, rot=H.roll)
        amt = L['side'] + D.get('side', 0.)
        y = A([.80, .60, .36, .20, .08, -.06, -.16, -.28, -.42, -.58, -.70, -.84, -1.0])
        mid = A([.66, .65, .645, .66, .60, .575, .62, .575, .50, .44, .37, .27, .10]); far = A([.86, .84, .80, .80, .77, .745, .765, .72, .66, .575, .48, .34, .14]); cen = A([.05, .07, .10, .13, .06, .03, .05, .04, .03, .02, .02, .0, .0])
        x = mid * (1 - amt) + cen * amt if amt >= 0 else mid * (1 + amt) + far * (-amt)
        if amt > -.99 and not D.get('noside'):
            Q = H.Pc(np.c_[s.sg * x, y], squeeze=False); c.fill(np.vstack([Q, (Q + [s.sg * 4 * s.u, 0])[::-1]]), sh)
        if s.cast is not None:
            for p in s.cast: c.fill(p, sh)
        if s.hairs:
            off = A([s.sg * .05, .10]) * s.u
            for p in s.hairs: c.fill(p + off, sh)
        s.planes(); c.unclip()

    def planes(s):
        """under-brow, nose, under-lip and chin planes"""
        c, H, D, X_, sg = s.c, s.H, s.D, s.X, s.sg; sh = s.sk(1)
        if D.get('nose') and D['nose'] != 'none':
            w, yt, ytip, yb, d = H.k['nose']; k = D.get('nshadow', 1.)
            if k > 0 and abs(H.yaw) < 1.2:
                P = [(sg * .004, ytip + .06, 1), (sg * .012, ytip - .01, .95), (-sg * .012, yb + .012, .7), (sg * .010, yb - .020 * k, .25), (sg * .062 * k, yb - .010, .2), (sg * .050 * k, yb + .028, .3)]
                c.fill(H.N(P), sh)
        yb = X_['my'] + float(np.min(X_['mL'])) - .055
        if D.get('lipsh', 1): c.fill(H.Pm([(X_['mx'] - .05, yb + .010), (X_['mx'], yb + .016), (X_['mx'] + .05, yb + .010), (X_['mx'] + .03, yb - .016), (X_['mx'] - .03, yb - .016)]), sh, .9)
        if X_['dark'] > 0:
            for q in (-1, 1):
                if H.vis(q * .3, .05): c.fill(H.Pc([(q * .09, .22), (q * .30, .28), (q * .60, .18), (q * .66, -.02), (q * .50, -.16), (q * .24, -.18), (q * .11, -.10)]), s.sk(2), X_['dark'])

    # ---- ears
    def ears_far(s): ay = abs(s.H.yaw); s.ears([q for q in (-1, 1) if ay < .12 or q == (1 if s.H.yaw > 0 else -1)])
    def ears_near(s):
        if abs(s.H.yaw) >= .12: s.ears([-1 if s.H.yaw > 0 else 1])

    def ears(s, which=(-1, 1)):
        if s.D.get('ears') == 'namek': return s.namek_ears(which)
        c, H, L = s.c, s.H, s.L
        for q in which:
            if H.ear(q)['vis'] <= 0: continue
            ex, t, b, z = H.k['ear']; sd = (q == s.sg); k = (t - b) / .5
            def Q(L_, n=8): return spl(H.P3(A([(q * (ex - .02 + u * .62 * k), b + (v + .335) * k, z + .02 - u * 1.25) for u, v in L_])), n=n)
            o = Q([(-.02, .040), (.012, .105), (.062, .150), (.118, .142), (.160, .085), (.168, .0), (.152, -.085), (.122, -.160), (.098, -.225), (.082, -.290), (.052, -.335), (.018, -.335), (-.012, -.295), (-.02, -.24)])
            cl = np.vstack([o, o[:1]]); c.fill(cl, s.sk(0)); c.clip(cl)
            if sd: c.fill(Q([(-.05, .20), (.10, .16), (.118, .06), (.105, -.02), (.118, -.10), (.09, -.20), (.05, -.26), (.06, -.40), (-.05, -.40)]), s.sk(1))
            c.fill(Q([(.020, .040), (.055, .095), (.104, .090), (.122, .030), (.112, -.04), (.088, -.030), (.080, .022), (.052, .030), (.044, -.020), (.056, -.095), (.070, -.160), (.045, -.205), (.018, -.180), (.030, -.12), (.012, -.06)]), s.sk(2) if sd else s.sk(1))
            c.unclip(); c.line(o, s.wo * .82, s.ink, taper=(.10, .12), press=.22, seed=s.seed + 3 + q, tremor=.12)
            s.ln(Q([(.022, .015), (.034, .068), (.070, .100), (.110, .085), (.126, .020), (.114, -.055), (.094, -.115)]), .62, seed=11 + q, taper=(.3, .5))
            s.ln(Q([(.050, .030), (.066, -.010), (.052, -.070), (.066, -.130), (.056, -.185), (.028, -.196)]), .62, seed=13 + q, taper=(.5, .4))

    def namek_ears(s, which):
        """long pointed ears swept out and back"""
        c, H, L = s.c, s.H, s.L; ex, t, b, z = H.k['ear']
        for q in which:
            vis = float((H.R @ A([q * .9, 0, .10]))[2])
            if vis < -.55: continue
            def Q(L_, n=8, cn=()): return spl(H.P3(A([(q * (ex - .04 + u), v, z - w) for u, v, w in L_])), n=n, corners=cn)
            o = Q([(-.03, -.04, -.02), (.06, .10, .02), (.20, .17, .10), (.40, .27, .22), (.60, .40, .36), (.47, .13, .26), (.33, -.06, .18), (.20, -.22, .10), (.08, -.33, .04), (-.02, -.36, .0)], cn=(4,))
            cl = np.vstack([o, o[:1]]); c.fill(cl, s.sk(0)); c.clip(cl); sd = (q == s.sg)
            c.fill(Q([(.02, -.06, .0), (.14, .09, .06), (.30, .17, .16), (.52, .33, .32), (.36, .06, .20), (.24, -.10, .12), (.14, -.20, .07), (.10, -.12, .05), (.08, -.02, .03)], cn=(3,)), s.sk(1 if not sd else 2))
            if sd: c.fill(Q([(.10, -.40, .04), (.20, -.23, .10), (.36, -.02, .20), (.60, .40, .36), (.50, .24, .30), (.30, -.02, .16), (.14, -.18, .06), (.05, -.2, .02)], cn=(3,)), s.sk(1))
            c.unclip(); c.line(o, s.wo * .9, s.ink, taper=(.10, .10), press=.22, seed=s.seed + 3 + q, tremor=.12)
            s.ln(Q([(.03, -.02, .0), (.13, .085, .06), (.28, .165, .15), (.47, .29, .28)]), .75, seed=11 + q, taper=(.3, .6))
            s.ln(Q([(.10, -.04, .04), (.16, -.08, .07), (.15, -.16, .07), (.08, -.22, .04)]), .62, seed=13 + q, taper=(.4, .5))

    # ---- eyes
    def lids(s, q):
        E = s.D['eye']; X_ = s.X; sfx = 'R' if q > 0 else 'L'
        g = lambda k: A(X_[k + sfx] if X_.get(k + sfx) is not None else (X_[k] if X_[k] is not None else E[k]), float)
        return E['xi'] + (E['xo'] - E['xi']) * T5, g('U'), g('Lo')

    def eye(s, q):
        c, H, L, D, X_ = s.c, s.H, s.L, s.D, s.X; E = D['eye']
        if not H.vis(q * (E['xi'] + .10), -.02): return
        xs, U, Lo = s.lids(q); far = q * H.yaw > .25; one = X_['oneeye']
        closed = X_['closed'] > .5 and not (one is not None and one == q)
        Mx = lambda P: np.c_[q * np.asarray(P)[:, 0], np.asarray(P)[:, 1]]; ink = s.ink
        if closed:                                             # arc = laughing ^ ^, else a calm downward line
            xm = xs.copy(); k = X_['arc']
            y0 = (U + Lo) / 2; yb = y0 * 0 + float(np.mean(y0)) - .01
            h = A([-.06, .03, .065, .035, -.045]) if k > .5 else A([.02, -.028, -.045, -.034, .012])
            P = crv(xm, yb + h * E.get('cl', 1.) + A([0, 0, 0, 0, 0]), 48); t = np.linspace(0, 1, len(P)); th = prof(t, [(0, .004), (.15, .018), (.5, .030), (.82, .030), (1, .004)]) * E.get('lw', 1.)
            B = P + np.c_[0 * th, th]; c.fill(H.Pc(Mx(np.vstack([B, P[::-1]]))), ink)
            if k <= .5 and E.get('lash'): s.lashes(q, P, th * 0, down=True)
            if k <= .5: s.ln(H.Pc(Mx(np.c_[P[:, 0], P[:, 1] + .06][8:-8])), .5, seed=33 + q, taper=(.5, .5), col=s.sk(2))
            if k > .5:
                for j, (dx, dy) in enumerate(((.030, -.016), (.030, .022))): s.ln(H.Pc(Mx(A([(xs[4] + .012, P[-1, 1] + .010), (xs[4] + .012 + dx, P[-1, 1] + .01 + dy)]))), .6, seed=35 + j, taper=(.1, .7))
            return
        up = crv(xs + A([0, -.004, 0, .004, 0]), U, 48); lo = crv(xs + A([.012, 0, 0, 0, -.012]) * E.get('round', 1.), Lo, 48); n = len(up); t = np.linspace(0, 1, n)
        op = spl(np.vstack([np.c_[xs, U], np.c_[xs + A([.012, 0, 0, 0, -.012]), Lo][::-1]]), True, corners=E.get('corners', (0, 4)), n=7)
        OP = H.Pc(Mx(op)); wh, wh2 = L['white']; c.fill(OP, wh); c.clip(OP)
        ix = E['xi'] + (E['xo'] - E['xi']) * E.get('ic', .46) + X_['gx'] * q; iy = E.get('iy', 0.) + X_['gy']; rx, ry = E['ir'][0] * X_['ir'], E['ir'][1] * X_['ir']
        a = np.linspace(0, TAU, 49)[:-1]; El = lambda r1, r2, dx=0., dy=0.: H.Pc(np.c_[q * ix + dx + np.cos(a) * r1, iy + dy + np.sin(a) * r2])
        ls = 1 if L['ld'][0] > 0 else -1
        ic = E['iris']
        if ic == 'black':
            c.fill(El(rx, ry), '#0b090e'); c.fill(El(rx * .66, ry * .34, 0, -ry * .46), mixc('#0b090e', '#4a4258', .55))
        elif ic == 'white': pass
        else:
            c.fill(El(rx, ry), ic[0]); c.fill(El(rx * .84, ry * .88), tone(L, ic[1], 0)); c.clip(El(rx * .84, ry * .88))
            c.fill(H.Pc(np.c_[q * ix + A([-1.2, -.5, .2, 1.2, 1.2, -1.2]) * rx, iy + A([.18, .28, .16, .26, 1.4, 1.4]) * ry]), ic[0]); c.fill(El(rx * .66, ry * .40, 0, -ry * .58), tone(L, ic[2], 0)); c.unclip(); c.fill(El(rx * .34, ry * .40), '#0b090e')
        th = E.get('ls', .040)
        c.fill(H.Pc(Mx(np.vstack([up + [0, .05], np.c_[up[:, 0], up[:, 1] - th * (.5 + .6 * np.sin(math.pi * np.clip(t * .9 + .05, 0, 1)))][::-1]]))), wh2, .85)
        if ic == 'black': c.fill(El(rx, ry), '#0b090e', .0)
        g = X_['glint']
        if g > 0:
            c.fill(El(rx * .34 * g, ry * .25 * g, ls * rx * .26, ry * .38), '#ffffff')
            if g > .6 and E.get('g2', 1): c.fill(El(rx * .17 * g, ry * .12 * g, -ls * rx * .34, -ry * .44), '#ffffff', .85)
        if X_['tears'] > 0:
            hg = .030 * X_['tears'] * (1 + .35 * np.sin(t * 9 + q)); c.fill(H.Pc(Mx(np.vstack([lo - [0, .02], np.c_[lo[:, 0], lo[:, 1] + hg][::-1]]))), mixc(wh, '#a8d4ff', .5), .75); c.line(H.Pc(Mx(np.c_[lo[:, 0], lo[:, 1] + hg][4:-4])), max(.8, s.wf), '#ffffff', taper=(.4, .4), alpha=.9, seed=5)
        c.unclip()
        # upper lid: one thick tapered shape that turns down the outer side
        lt = E.get('lt', [(0, .006), (.14, .020), (.42, .030), (.75, .040), (1, .034)]); thk = prof(t, lt) * E.get('lw', 1.)
        T = np.gradient(up, axis=0); T /= np.hypot(*T.T)[:, None]; Nn = np.c_[-T[:, 1], T[:, 0]]; B = up + Nn * thk[:, None]
        dn = E.get('down', .55); ox, oy = up[-1]; ly = Lo[4] + (U[4] - Lo[4]) * (1 - dn); w = thk[-1]
        sd = A([(ox + w * .95, oy + w * .25), (ox + w * .80, oy - (oy - ly) * .45), (ox - .010, ly)]); inn = A([(ox - .004, oy - (oy - ly) * .5), (ox - .004, oy)])
        tail = E.get('tail', .0); tp = A([(ox + w * .6 + tail, oy + w * .9 + tail * .30)]) if tail else np.zeros((0, 2))
        lid = np.vstack([B, tp, spl(sd, n=6), inn, up[::-1]]) if not far else np.vstack([B, A([(ox + w * .5, oy)]), up[::-1]])
        c.fill(H.Pc(Mx(lid)), ink)
        if E.get('lash') and not far: s.lashes(q, up, thk)
        m = len(lo); s.ln(H.Pc(Mx(lo[int(m * E.get('lo0', .30)):int(m * .92)])), E.get('low', .85), seed=45 + q, taper=(.5, .25))
        s.ln(H.Pc(Mx(A([up[0] + [-.004, .004], up[0] + [-.010, -.030], (up[0] * .55 + lo[0] * .45)]))), .55, seed=43 + q, taper=(.1, .7))
        if E.get('crease', 1) and np.max(U - Lo) > .05:
            cr = up + Nn * (thk + E.get('cro', .050))[:, None]; s.ln(H.Pc(Mx(cr[int(.30 * n):int(.86 * n)])), .55, seed=47 + q, taper=(.5, .4), col=mixc(ink, s.sk(2), .35))
        if X_['under'] > 0:
            P = np.c_[lo[:, 0], lo[:, 1] - .055 - .016 * np.sin(math.pi * t)]; s.ln(H.Pc(Mx(P[int(.1 * n):int(.62 * n)])), .55, seed=48 + q, taper=(.4, .6), col=mixc(ink, s.sk(2), .4), alpha=float(min(1, X_['under'])))

    def lashes(s, q, P, thk, down=False):
        n = len(P)
        for j, (u, dx, dy, w) in enumerate(s.D['eye']['lash']):
            i = int(u * (n - 1)); p = P[i] + [0, thk[i] * .5]; dy = -abs(dy) * .9 if down else dy
            Q = spl([p, p + [dx * .45, dy * .25], p + [dx, dy]], n=6); s.c.fill(s.H.Pc(np.c_[q * wedge(Q, w)[:, 0], wedge(Q, w)[:, 1]]), s.ink)

    def brow(s, q):
        c, H, D, X_ = s.c, s.H, s.D, s.X; BD = D.get('brow')
        if BD is None: return
        if BD.get('kind') == 'ridge': return s.ridge(q, BD)
        sfx = 'R' if q > 0 else 'L'; B = A(X_['B' + sfx] if X_.get('B' + sfx) is not None else (X_['B'] if X_['B'] is not None else BD['pts']), float).copy()
        B[:, 1] += X_['dB'] + X_['aB'] * (B[:, 0] - .33)
        if not H.vis(q * B[1, 0], B[1, 1] - .05): return
        mid = crv(B[:, 0], B[:, 1], 48); t = np.linspace(0, 1, len(mid)); T = np.gradient(mid, axis=0); T /= np.hypot(*T.T)[:, None]; N = np.c_[-T[:, 1], T[:, 0]]; th = prof(t, BD['th'])
        P = np.vstack([mid + N * th[:, None] * .5, (mid - N * th[:, None] * .5)[::-1]]); c.fill(H.Pc(np.c_[q * P[:, 0], P[:, 1]]), tone(s.L, BD['col'], 0) if BD['col'] != 'ink' else s.ink)

    def ridge(s, q, BD):
        """a brow without hair: a heavy ridge line over a cast shadow that joins the upper lid"""
        c, H, X_ = s.c, s.H, s.X; B = A(BD['pts'], float).copy(); B[:, 1] += X_['dB'] + X_['aB'] * (B[:, 0] - .33)
        if not H.vis(q * B[2, 0], B[2, 1] - .06): return
        xs, U, Lo = s.lids(q); up = crv(xs, U, 30); mid = crv(B[:, 0], B[:, 1], 60); t = np.linspace(0, 1, len(mid))
        c.clip(s.F); c.fill(H.Pc(np.c_[q * np.r_[mid[:, 0], up[::-1, 0]], np.r_[mid[:, 1], up[::-1, 1] + .012]]), s.sk(1), .95)
        c.fill(H.Pc(np.c_[q * np.r_[mid[:, 0], mid[::-1, 0]], np.r_[mid[:, 1] - .004, mid[::-1, 1] - .046 * np.sin(math.pi * t[::-1]) ** .6]]), s.sk(2), .75); c.unclip()
        T = np.gradient(mid, axis=0); T /= np.hypot(*T.T)[:, None]; N = np.c_[-T[:, 1], T[:, 0]]; th = prof(t, BD['th'])[:, None]; P = np.vstack([mid + N * th * .5, (mid - N * th * .5)[::-1]]); c.fill(H.Pc(np.c_[q * P[:, 0], P[:, 1]]), s.ink)
        s.ln(H.Pc([(q * (B[2, 0] - .10), B[2, 1] + .065), (q * B[2, 0], B[2, 1] + .085), (q * (B[3, 0] - .02), B[3, 1] + .062)]), .55, seed=61 + q, taper=(.5, .5), col=mixc(s.ink, s.sk(2), .3))

    def lines(s):
        for j, (P, w) in enumerate(s.D.get('lines', ())):
            for q in (-1, 1):
                if all(s.H.vis(q * a, b) for a, b in P): s.ln(s.H.Pc([(q * a, b) for a, b in P]), w, seed=120 + j, taper=(.5, .5), col=mixc(s.ink, s.sk(2), .2))

    def furrow(s):
        f = s.X['furrow']; H = s.H
        if f <= 0: return
        for q in (-1, 1):
            if not H.vis(.06 * q, .15): continue
            s.ln(H.Pc([(q * .052, .235), (q * .040, .165), (q * .052, .095)]), .8, seed=71 + q, taper=(.5, .3), alpha=min(1., f))
            if f > 1: s.ln(H.Pc([(q * .10, .25), (q * .085, .20), (q * .09, .15)]), .6, seed=73 + q, taper=(.5, .4))

    def nose(s):
        H, D = s.H, s.D; k = D.get('nose', 'kid'); yw = H.yaw; ay = abs(yw); d = 1 if yw >= 0 else -1; w, yt, ytip, yb, dd = H.k['nose']
        if k == 'none': return
        if ay > .95:
            return
        if k == 'kid':
            if ay < .14: s.ln(H.N([(-.020, yb + .016, .8), (s.sg * .006, yb + .004, .85), (.022, yb + .014, .8)]), .8, seed=81, taper=(.5, .4))
            else:
                pr = spl(H.P3(H.nose_profile()[3:7]), n=8); m = len(pr); s.ln(pr[int(m * .10):int(m * .95)], 1.0, seed=83, taper=(.5, .25))
            return
        if ay < .14:
            s.ln(H.N([(s.sg * .010, ytip + .10, 1), (s.sg * .020, ytip + .03, 1), (s.sg * .036, ytip - .012, .9), (s.sg * .014, yb + .004, .75), (-s.sg * .016, yb + .008, .75)]), 1.0, seed=81, taper=(.65, .3))
            for q in (-1, 1): s.ln(H.N([(q * .050, yb + .014, .4), (q * .070, yb + .020, .25), (q * .076, yb + .040, .2)]), .6, seed=82 + q, taper=(.3, .6))
        else:
            pr = spl(H.P3(H.nose_profile()[1:7]), n=8); m = len(pr); s.ln(pr[int(m * .25):int(m * .95)], 1.15, seed=83, taper=(.6, .2))
            s.ln(H.N([(-d * .046, yb + .010, .45), (-d * .074, yb + .020, .25), (-d * .080, yb + .046, .18)]), .7, seed=84, taper=(.3, .5))

    def mouth(s):
        c, H, L, D, X_ = s.c, s.H, s.L, s.D, s.X; t5 = A([-1, -.5, 0, .5, 1.]); mx, my, hw = X_['mx'], X_['my'], X_['hw']
        Uv, Lv = A(X_['mU'], float), A(X_['mL'], float); U = spl(np.c_[mx + hw * t5, my + Uv], n=12); Lw = spl(np.c_[mx + hw * t5, my + Lv], n=12); n = len(U); t = np.linspace(0, 1, n)
        if X_['wob'] > 0: U = U + np.c_[0 * t, np.sin(t * 21) * .010 * X_['wob']]; Lw = Lw + np.c_[0 * t, np.sin(t * 19 + 1) * .012 * X_['wob']]
        PU, PL = H.Pm(U), H.Pm(Lw); gap = float(np.max(Uv - Lv))
        if gap > .014:
            op = np.vstack([PU, PL[::-1]]); mc = [tone(L, v, 1) for v in D.get('mouthc', ('#5a1620', '#d4636a', '#2c080e'))]; c.fill(op, mc[0]); c.clip(op)
            c.fill(H.Pm(np.vstack([U + [0, .02], (U - np.c_[0 * t, gap * .34 * np.sin(math.pi * t)])[::-1]])), mc[2])
            if X_['tongue'] > 0:
                yl = my + float(Lv[2]); k = X_['tongue']; c.fill(H.Pm(spl([(mx - hw * .62, yl - .03), (mx - hw * .50, yl + gap * .22 * k), (mx - hw * .12, yl + gap * .36 * k), (mx + hw * .06, yl + gap * .31 * k), (mx + hw * .48, yl + gap * .24 * k), (mx + hw * .62, yl - .03)], True, n=7)), mc[1])
                s.ln(H.Pm([(mx, yl + gap * .30 * k), (mx + .004, yl + gap * .18 * k), (mx, yl + gap * .08)]), .5, seed=96, col=mc[0], taper=(.2, .6))
            if X_['teeth'] > 0:
                tk = X_['teeth']; c.fill(H.Pm(np.vstack([U + [0, .02], (U - np.c_[0 * t, tk * np.sin(math.pi * np.clip(t, .08, .92)) ** .35])[::-1]])), tone(L, ('#ffffff', '#e2dfe8', '#b8b4c8'), 0));
                s.ln(H.Pm((U - np.c_[0 * t, tk * np.sin(math.pi * np.clip(t, .08, .92)) ** .35])[4:-4]), .45, seed=97, col=mixc(s.ink, '#a08088', .6))
            if X_['loteeth'] > 0:
                tk = X_['loteeth']; c.fill(H.Pm(np.vstack([Lw - [0, .02], (Lw + np.c_[0 * t, tk * np.sin(math.pi * np.clip(t, .08, .92)) ** .4])[::-1]])), tone(L, ('#f4f1f4', '#e2dfe8', '#b8b4c8'), 0)); s.ln(H.Pm((Lw + np.c_[0 * t, tk * np.sin(math.pi * np.clip(t, .08, .92)) ** .4])[4:-4]), .45, seed=98, col=mixc(s.ink, '#a08088', .6))
            if X_['clench'] > 0:
                ym = (U + Lw) / 2; c.fill(op, tone(L, ('#ffffff', '#e2dfe8', '#b8b4c8'), 0)); s.ln(H.Pm(ym[3:-3]), .6, seed=99, taper=(.1, .1))
                for x in np.linspace(-.75, .75, 7): i = int((x + 1) / 2 * (n - 1)); s.ln(H.Pm([U[i], ym[i], Lw[i]]), .4, seed=101, col=mixc(s.ink, '#a08088', .5), taper=(.1, .1))
            if X_['fang'] > 0:
                for q in (-1, 1): i = int((.5 + q * .30) * (n - 1)); b = U[i]; c.fill(H.Pm([b + [-.018, .012], b + [.018, .012], b + [0, -.055 * X_['fang']]]), '#ffffff')
            c.unclip(); s.ln(PU, 1.15 * X_['mw'], seed=91, taper=(.12, .12)); s.ln(PL, 1.0 * X_['mw'], seed=92, taper=(.16, .16))
        else:
            s.ln(PU, 1.2 * X_['mw'], seed=91, taper=(.25, .25))
            if X_['fang'] > 0 and D.get('fang'): i = int(.70 * (n - 1)); b = U[i]; c.fill(H.Pm([b + [-.014, 0], b + [.014, 0], b + [.002, -.05]]), '#ffffff'); s.ln(H.Pm([b + [-.014, 0], b + [.002, -.05], b + [.014, 0]]), .5, seed=93, taper=(.1, .1))
        if X_['lip'] > 0:
            yb = my + float(np.min(Lv)) - .050; w = min(.07, hw * .55); s.ln(H.Pm([(mx - w, yb + .004), (mx, yb - .004), (mx + w, yb + .004)]), .6, seed=94, taper=(.5, .5), col=mixc(s.ink, s.sk(2), .4))
        if X_['naso'] > 0:
            for q in (-1, 1): s.ln(H.Pm([(q * (hw + .020) + mx, my + Uv[0 if q < 0 else 4] + .055), (q * (hw + .048) + mx, my + Uv[0 if q < 0 else 4] + .005), (q * (hw + .040) + mx, my + Uv[0 if q < 0 else 4] - .055)]), .7, seed=95 + q, taper=(.5, .5), alpha=min(1., X_['naso']))

    def extras(s):
        c, H, X_ = s.c, s.H, s.X
        if X_['blush'] > .45 and s.u > 60:                        # two or three little pen strokes on the cheek
            for q in (-1, 1):
                if not H.vis(q * .46, -.28): continue
                for j in range(3): x = .40 + .055 * j; s.ln(H.Pc([(q * (x + .02), -.255), (q * (x - .012), -.335)]), .55, seed=111 + j, col=mixc(tone(s.L, '#d2564a', 0), s.sk(1), .25), taper=(.3, .5))
        if X_['sweat'] > 0:
            for j, (x, y, k) in enumerate(((s.sg * .58, .12, 1.), (-s.sg * .62, -.18, .7), (s.sg * .30, .46, .6))[:int(1 + 2 * X_['sweat'])]):
                if H.vis(x, y): s.drop(H.Pc([(x, y)])[0], .050 * s.u * k)
        if X_['tears'] > .5:
            for q in (-1, 1):
                E = s.D['eye']; x = E['xi'] + (E['xo'] - E['xi']) * .70
                if H.vis(q * x, -.2):
                    P = H.Pc(spl([(q * x, -.14), (q * (x + .02), -.28), (q * (x - .01), -.46), (q * (x - .05), -.62)], n=8)); c.line(P, max(1.2, .034 * s.u), mixc('#ffffff', '#a8d4ff', .35), taper=(.2, .3), alpha=.85, seed=7); c.line(P + [.006 * s.u, 0], max(.8, .010 * s.u), '#ffffff', taper=(.3, .3), alpha=.9, seed=8)
        if s.wet > 0: s.wetness()

    def drop(s, p, r):
        P = spl(A([(0, -1.7), (.55, -.4), (.72, .3), (0, 1.), (-.72, .3), (-.55, -.4)]) * r + p, True, corners=(0,), n=6); c = s.c
        c.fill(P, mixc('#ffffff', tone(s.L, '#8ec8f2', 0), .55), .85); c.fill(ell(p[0] - r * .25, p[1] + r * .1, r * .16, r * .30), '#ffffff'); c.line(np.vstack([P, P[:1]]), max(.7, s.wf * .9), mixc(s.ink, '#3a5a86', .5), taper=(.1, .1), seed=3)

    def wetness(s):
        r = np.random.default_rng(s.seed + 5); H = s.H
        for j in range(int(14 * s.wet)):
            x, y = r.uniform(-.6, .6), r.uniform(-.8, .3)
            if H.vis(x, y) and abs(x) > .05: s.drop(H.Pc([(x, y)])[0], r.uniform(.012, .030) * s.u)
        for j in range(int(5 * s.wet)):
            x, y = r.uniform(-.55, .55), r.uniform(-.2, .2); l = r.uniform(.2, .5)
            if H.vis(x, y - l): s.c.line(H.Pc(spl([(x, y), (x + r.normal(0, .02), y - l * .5), (x + r.normal(0, .03), y - l)], n=6)), max(1., .016 * s.u), '#eaf4ff', taper=(.4, .25), alpha=.7, seed=j)

    def features(s):
        far = 1 if s.H.yaw > 0 else -1
        for q in (far, -far):
            if q * s.H.yaw > .25: s.c.clip(s.F)
            s.eye(q)
            if q * s.H.yaw > .25: s.c.unclip()
        s.lines(); s.nose(); s.mouth(); s.furrow()
        for q in (-1, 1): s.brow(q)
        s.extras()

    def contour(s):
        """jaw and cheek line: heavy at the chin, light on the cheek, lifted at the temple"""
        H, c = s.H, s.c
        if abs(H.yaw) > .75 and s.D.get('nose') != 'none': c.fill(spl(H.P3(H.nose_profile()), True, n=6), s.sk(0)); d_ = 1 if H.yaw > 0 else -1; w_, yt, yp, yb, dd = H.k['nose']; s.ln(H.N([(-d_ * .030, yb + .014, .5), (-d_ * .056, yb + .024, .3), (-d_ * .058, yb + .050, .2)]), .6, seed=84, taper=(.3, .5))
        P = resample(np.vstack([s.F, s.F[:1]]), 2.0); p0 = H.P3(A([(0, -.99, .62)]))[0]; i = int(np.argmin(np.hypot(*(P - p0).T))); P = np.vstack([P[i:-1], P[:i + 1]]); n = len(P); half = n // 2; ov = max(3, int(.05 * s.u / 2))
        upv = A([math.sin(H.roll), -math.cos(H.roll)]); cut = float((H.P([(0, s.D.get('topcut', .30))])[0]) @ upv)
        for k, seg in enumerate((np.vstack([P[-ov:-1], P[:half]]), np.vstack([P[half:], P[1:ov]])[::-1])):
            above = np.nonzero(seg @ upv > cut)[0]
            if len(above) and above[0] > 12: seg = seg[:above[0]]
            Ln = np.r_[0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
            wf = lambda t, Ln=Ln, seg=seg: (lambda q: .95 + .22 * np.exp(-((q - .05) / .16) ** 2) - .28 * np.exp(-((q - .38) / .16) ** 2) + .16 * np.exp(-((q - .70) / .12) ** 2) - .30 * np.exp(-((q - 1.05) / .22) ** 2))(np.interp(t, np.linspace(0, 1, len(seg)), Ln) / s.u)
            c.line(seg, s.wo, s.ink, taper=(0, .04), press=.16, seed=s.seed + k, tremor=.12, wfun=wf)


# ------------------------------------------------------------------ surfaces of revolution round the head
class Rev:
    """prof rows (y, rx, rz, cz); tilt raises (+) the back; off = centre offset; bump(th, v) -> radial offset"""
    def __init__(s, H, prof_, tilt=0., n=40, nf=97, smooth_=True, bump=None, cx=0.):
        s.H, s.tilt, s.bump, s.cx = H, tilt, bump, cx; P = A(prof_, float); s.P = spl(P[:, :2], n=10) if False else P; k = np.linspace(0, 1, len(P)); s.k = k
        s.n, s.nf = n, nf

    def _row(s, v):
        v = np.clip(np.asarray(v, float), 0, 1); m = len(s.P) - 1; i = np.minimum((v * m).astype(int), m - 1); f = v * m - i
        E = np.vstack([2 * s.P[0] - s.P[1], s.P, 2 * s.P[-1] - s.P[-2]]); a, b, c, d = E[i], E[i + 1], E[i + 2], E[i + 3]; f = f[..., None]
        return .5 * (2 * b + (c - a) * f + (2 * a - 5 * b + 4 * c - d) * f * f + (3 * b - a - 3 * c + d) * f ** 3)

    def Q(s, th, v):
        th, v = np.broadcast_arrays(np.asarray(th, float), np.asarray(v, float)); r = s._row(v); b = 0. if s.bump is None else s.bump(th, v)
        return np.stack([s.cx + np.sin(th) * (r[..., 1] + b), r[..., 0] - np.cos(th) * s.tilt, r[..., 3] + np.cos(th) * (r[..., 2] + b)], -1)

    def nrm(s, th, v):
        e = 1e-2; a = s.Q(th + e, v) - s.Q(th - e, v); b = s.Q(th, np.minimum(v + e, 1)) - s.Q(th, np.maximum(v - e, 0)); n = np.cross(a, b); return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)

    def face(s, th, v): return (s.nrm(np.asarray(th, float), np.asarray(v, float)) @ s.H.R.T)[..., 2]
    def pt(s, th, v): return s.H.P3(s.Q(th, v))

    def mesh(s, v0=0., v1=1.): return s.Q(np.linspace(-math.pi, math.pi, s.nf)[None], np.linspace(v0, v1, s.n)[:, None])
    def sil(s, v0=0., v1=1., thr=-.06, every=False, span=None):
        """contour of the camera-facing part (the inside of an open rim never shows)"""
        H = s.H; th = np.linspace(-math.pi, math.pi, s.nf); v = np.linspace(v0, v1, s.n); TH, V = np.meshgrid(th, v); P = H.P3(s.Q(TH, V)); f = s.face(TH, V) > thr; ok = f[:-1, :-1] | f[1:, :-1] | f[:-1, 1:] | f[1:, 1:]
        if span: ok &= (np.abs((TH + H.yaw + math.pi) % TAU - math.pi) < span)[:-1, :-1]
        k = max(.5, min(1.5, 1000. / (H.s * 3.2))); lo = P.reshape(-1, 2).min(0) - 8; hi = P.reshape(-1, 2).max(0) + 8; w, h = np.ceil((hi - lo) * k).astype(int) + 2; m = np.zeros((h, w), np.uint8); q = (P - lo) * k
        quad = np.stack([q[:-1, :-1], q[:-1, 1:], q[1:, 1:], q[1:, :-1]], 2)[ok]
        for c in np.round(quad * 16).astype(np.int32): cv2.fillConvexPoly(m, c, 255, cv2.LINE_8, 4)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)); cs = contours(m / 255.); cs = [resample(c / k + lo, 2.0) for c in cs if len(c) > 40]; return cs if every else max(cs, key=len)

    def curve(s, th, v, thr=.02):
        """visible pieces of a curve on the surface (list of px polylines)"""
        th, v = np.broadcast_arrays(np.asarray(th, float), np.asarray(v, float)); f = s.face(th, v) > thr; P = s.pt(th, v); out = []; i = 0
        while i < len(f):
            if not f[i]: i += 1; continue
            j = i
            while j + 1 < len(f) and f[j + 1]: j += 1
            if j - i >= 2: out.append(P[i:j + 1])
            i = j + 1
        return out

    def patch(s, th, v):
        """polygon on the visible side (points that turn away are pushed to the silhouette)"""
        th, v = [np.array(q, float) for q in np.broadcast_arrays(th, v)]; a0 = -s.H.yaw
        th = np.clip(th, a0 - math.pi / 2 - .05, a0 + math.pi / 2 + .05); return s.pt(th, v)


# ------------------------------------------------------------------ 3-D hair blades
def surf(H, theta, y, lift=0.):
    a = np.interp(y, H.yy, H.A); b = np.interp(y, H.yy, H.F if math.cos(theta) >= 0 else H.B); p = H.k['p']
    sx, cz = math.sin(theta), math.cos(theta); r = (abs(sx) ** p + abs(cz) ** p) ** (-1 / p); q = A([a * sx * r, y, b * cz * r]); n = A([sx / max(a, .1), (y - .1) * .95, cz / max(b, .1)]); n /= np.linalg.norm(n); return q + n * lift, n


def scalp(H, hl, infl=1.05, ny=22, nf=73, top=1.0, front=True, rise=0.):
    th = np.linspace(-math.pi, math.pi, nf); M = np.zeros((ny, nf, 3)); N = np.zeros((ny, nf, 3))
    for j, t in enumerate(th):
        for i, y in enumerate(np.linspace(hl(t), top, ny)): q, n = surf(H, t, y); M[i, j] = q * [infl, 1, infl] + [0, (infl - 1 + rise) * max(y, 0), 0]; N[i, j] = n
    if not front: return H._sil([M])
    vis = (N @ H.R.T)[..., 2] > -.12; vis = vis[:-1, :-1] & vis[1:, :-1] & vis[:-1, 1:] & vis[1:, 1:]
    Q = H.P3(M); k = max(.5, min(1., 900. / (H.s * 3.2))); lo = Q.reshape(-1, 2).min(0) - 8; hi = Q.reshape(-1, 2).max(0) + 8; w, h = np.ceil((hi - lo) * k).astype(int) + 2; m = np.zeros((h, w), np.uint8); q = (Q - lo) * k
    quad = np.stack([q[:-1, :-1], q[:-1, 1:], q[1:, 1:], q[1:, :-1]], 2)[vis]
    for c in np.round(quad * 16).astype(np.int32): cv2.fillConvexPoly(m, c, 255, cv2.LINE_8, 4)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)); cs = contours(m / 255.); return resample(max(cs, key=len) / k + lo, 3.0) if cs else Q[0, :3]


class Lock:
    """hair blade: 3-D centre line (quadratic) + width profile -> screen polygon"""
    def __init__(s, H, theta, y, p1, p2, w, kind='spike', asym=0., flex=1., lift=.02, tag='', id=0):
        s.p0, s.n = surf(H, theta, y, lift); s.p1, s.p2 = np.asarray(p1, float), np.asarray(p2, float); s.w, s.kind, s.asym, s.flex, s.tag, s.id = w, kind, asym, flex, tag, id

    @staticmethod
    def sp(H, theta, y, d, L, w, curl=(0, 0, 0), arc=(0, 0, 0), **k):
        p0, n = surf(H, theta, y, k.get('lift', .02)); d = np.asarray(d, float); d = d / (np.linalg.norm(d) + 1e-9); return Lock(H, theta, y, p0 + d * L * .5 + np.asarray(arc, float), p0 + d * L + np.asarray(curl, float), w, **k)

    @staticmethod
    def fr(H, theta, y, tip, w, dz=.06, bulge=.16, **k):
        p0, n = surf(H, theta, y, k.get('lift', .02)); tx, ty = tip[0], tip[1]; am = float(np.interp(ty, H.yy, H.A)) * .97
        p2 = A([tx, ty, float(H.z(np.clip(tx, -am, am), ty)) + dz]) if len(tip) < 3 else A(tip, float)
        p1 = (p0 + p2) / 2 + A([n[0] * bulge, n[1] * bulge * .6, max(n[2], .55) * bulge * 1.6]); p1[1] += .06; return Lock(H, theta, y, p1, p2, w, kind=k.pop('kind', 'bang'), **k)

    def build(s, H, off=(0, 0, 0), m=24):
        t = np.linspace(0, 1, m)[:, None]; C = (1 - t) ** 2 * s.p0 + 2 * (1 - t) * t * s.p1 + t * t * s.p2 + t ** 2 * np.asarray(off, float) * s.flex
        Q = H.P3(C); s.depth = float(H.depth(C[m // 3])); s.vis = float((H.R @ s.n)[2]); return s._poly(Q, s.w * H.s)

    def _poly(s, Q, wpx):
        m = len(Q); u = np.linspace(0, 1, m); T = np.gradient(Q, axis=0); ln = np.hypot(*T.T); tot = ln.sum(); T = T / (ln[:, None] + 1e-6)
        if tot < wpx * .6: g = Q[-1] - Q[0]; g = g / (np.hypot(*g) + 1e-6); T = T * .3 + g * .7; T /= np.hypot(*T.T)[:, None] + 1e-6
        N = np.c_[-T[:, 1], T[:, 0]]; w = wpx * .5
        pr = {'spike': (1 - u) ** 1.10 * (1 + .30 * np.sin(math.pi * np.clip(u * 1.3, 0, 1))), 'bang': (.70 + .30 * np.sin(math.pi * np.clip(u * .9, 0, 1))) * np.clip(1 - u ** 1.9, 0, 1),
              'flame': np.interp(u, [0, .12, .35, .62, .84, 1], [.40, .78, 1.0, .74, .30, 0]), 'bob': np.interp(u, [0, .10, .24, .45, .72, .90, 1], [.10, .52, .86, 1.0, .96, .58, 0]), 'flip': np.interp(u, [0, .12, .34, .54, .72, .87, 1], [.40, .70, 1.0, .94, .62, .27, 0])}[s.kind]
        b = s.asym * np.sin(math.pi * u) * w; s.Q = Q; s.Aa = Q + N * (w * pr + b)[:, None]; s.Bb = Q - N * (w * pr - b)[:, None]; s.poly = np.vstack([s.Aa, s.Bb[::-1]]); return s

    @staticmethod
    def d2(pts, wpx, kind='flame', asym=0., id=0, tag='', m=24, rel=.2, depth=0.):
        l = Lock.__new__(Lock); l.kind, l.asym, l.id, l.tag, l.w, l.depth, l.vis = kind, asym, id, tag, rel, depth, 1.
        q = spl(np.asarray(pts, float), n=8) if len(pts) > 2 else np.asarray(pts, float); d = np.r_[0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]; v = np.linspace(0, d[-1], m); return l._poly(np.c_[np.interp(v, d, q[:, 0]), np.interp(v, d, q[:, 1])], wpx)

    def side(s, t0, t1, a, b):
        n = len(s.Q); i0, i1 = int(t0 * (n - 1)), int(t1 * (n - 1)) + 1; mid = (s.Aa + s.Bb) / 2; h = (s.Bb - s.Aa) / 2; k = np.sin(np.linspace(0, math.pi, i1 - i0))[:, None]; c = (a + b) / 2
        return np.vstack([mid[i0:i1] + h[i0:i1] * (c + (a - c) * k), (mid[i0:i1] + h[i0:i1] * (c + (b - c) * k))[::-1]])


def wind_off(H, wind, f, ph, amp=1.0):
    """3-D tip offset for a screen wind (wx, wy up) in head units + flutter"""
    wx, wy = wind[0] * .42, wind[1] * .42; v = math.hypot(wx, wy) + 1e-6; a = f * TAU / 24.; fl = .25 * v * (.65 * math.sin(a * 1.9 + ph * 6.28) + .35 * math.sin(a * 3.7 + ph * 17.1))
    return H.R.T @ (A([wx + fl * (-wy / v), wy + fl * (wx / v), 0.]) * amp)


def hair_group(cel, locks, cols, w, seed=0, ld=(-.6, .7), caps=(), line=INK, inner=.30):
    """locks far -> near.  cols = dict(base, sh, hi, sep=line between clumps, hip).  One heavy outer contour, one shadow and one light shape per lock"""
    wo, wi, wf = w
    for c in caps: outline(cel, c, wo * 1.7, line, seed + 3, tremor=.2)
    for i, l in enumerate(locks): outline(cel, l.poly, wo * 1.7 * min(1., .30 + float(np.hypot(*(l.Aa - l.Bb).T).max()) / (wo * 5.)), line, seed + 11 * i + l.id, tremor=.2)
    for c in caps: cel.fill(c, cols['base'])
    for l in locks: cel.fill(l.poly, cols['base'])
    kx, ky = ld[0], -ld[1]
    for i, l in enumerate(locks):
        r = np.random.default_rng(l.id * 131 + seed + 7); cel.fill(l.poly, cols['base']); m = len(l.Q) // 2; nv = l.Aa[m] - l.Bb[m]; lit = 1 if nv @ A([kx, ky]) > 0 else -1
        a = r.uniform(.05, .35); t1 = r.uniform(.84, .97)
        cel.fill(l.side(.0, t1, a, 1.05) if lit > 0 else l.side(.0, t1, -1.05, -a), cols['sh'])
        if cols.get('hi') is not None and l.w > .08 and r.random() < cols.get('hip', .7):
            c = -r.uniform(.45, .72) * lit; t0 = r.uniform(.10, .28); hw = r.uniform(.07, .15) * cols.get('hiw', 1.); cel.fill(l.side(t0, t0 + r.uniform(.30, .50), c - hw, c + hw), cols['hi'])
        n = len(l.Q); sep = cols.get('sep', line)
        for q, P in enumerate((l.Aa, l.Bb)):
            i0 = int(n * (inner + r.uniform(-.14, .12))); seg = P[max(1, i0):]
            if len(seg) > 3: cel.line(seg, wi * r.uniform(.8, 1.1), sep, taper=(.5, .10), press=.3, seed=l.id * 7 + q + seed)
        if l.w > .16 and cols.get('in', 1):
            for q in range(int(r.integers(1, 3))):
                c = r.uniform(-.5, .5); t0 = r.uniform(.05, .3); t1 = t0 + r.uniform(.3, .5); i0, i1 = int(t0 * n), min(n, int(t1 * n)); P = (l.Aa[i0:i1] * (1 - c) + l.Bb[i0:i1] * (1 + c)) / 2
                if len(P) > 3: cel.line(P, wf * r.uniform(.8, 1.2), sep, taper=(.45, .5), press=.3, seed=l.id * 13 + q + seed)
    return cel


# ------------------------------------------------------------------ cloth and limbs (authored in px)
def fold(cel, P, th, cols, w, seed=0, hi=True):
    """one fold: pen line, dark sliver on one side, lit sliver on the other.  cols = (base, shadow, deep, hi, line)"""
    P = spl(np.asarray(P, float), n=8) if len(P) < 8 else np.asarray(P, float); n = len(P); t = np.linspace(0, 1, n); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]
    k = (np.clip(t / .15, 0, 1) ** .9 * (1 - t) ** .8 * th)[:, None]; cel.fill(np.vstack([P, (P + N * k)[::-1]]), cols[1])
    if hi: cel.fill(np.vstack([P, (P - N * k * .55)[::-1]]), cols[3])
    cel.line(P, w, cols[4], taper=(.06, .6), press=.3, seed=seed)


def limb(P, w):
    """tapered tube along a path: w = half widths at evenly spaced stations (px).  returns (polygon, A side, B side, centre)"""
    C = resample(spl(np.asarray(P, float), n=14) if len(P) > 2 else np.asarray(P, float), 3.0); n = len(C); T = np.gradient(C, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
    k = spl(np.c_[np.linspace(0, 1, len(w)), w], n=8); r = np.interp(np.linspace(0, 1, n), k[:, 0], k[:, 1])[:, None]; a, b = C + N * r, C - N * r; return np.vstack([a, b[::-1]]), a, b, C


def capsule(P, w, n=9):
    """limb with round ends"""
    Q, a, b, C = limb(P, w); t0, t1 = C[0] - C[1], C[-1] - C[-2]; f = lambda c, t, r, up: np.array([c + r * (math.cos(v) * A([-up * t[1], up * t[0]]) / np.hypot(*t) + math.sin(v) * t / np.hypot(*t)) for v in np.linspace(0, math.pi, n)[1:-1]])
    return np.vstack([a, f(C[-1], t1, w[-1], 1), b[::-1], f(C[0], t0, w[0], 1)])


def poly(cel, P, cols, w, seed=0, sh=None, hi=None, lines=(), closed=True, gaps=None, line=None, **k):
    """flat + clipped shadow / light polygons + ink.  cols = (base, shadow, deep, hi, line)"""
    cel.fill(P, cols[0]); cel.clip(P)
    for q in (sh or ()): cel.fill(q, cols[1])
    for q in (hi or ()): cel.fill(q, cols[3])
    cel.unclip(); outline(cel, P, w, cols[4] if line is None else line, seed, tremor=.15, gaps=gaps, **k)
    for j, (q, ww) in enumerate(lines): cel.line(spl(np.asarray(q, float), n=8) if len(q) < 8 else q, w * ww, cols[4], taper=(.25, .5), press=.3, seed=seed + 31 + j)


def cloth(L, tri, hi=None, line=None):
    """(base, shadow, deep, hi, line) of a local triple in light L"""
    b, s, d = (tone(L, tri, i) for i in range(3)); return (b, s, d, np.clip(b * 1.0 + (1 - b) * .22, 0, 1) if hi is None else tone(L, hi, 0), L['line'] if line is None else line)


def grade(layer, L, box, rimw=1., amt=None, rims=True):
    """airbrush gradient along the light + rim pass over a finished character layer.  box = (x0, y0, x1, y1)"""
    x0, y0, x1, y1 = box; kx, ky = L['ld'][0], -L['ld'][1]; n = math.hypot(kx, ky); kx, ky = kx / n, ky / n; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; r = math.hypot(x1 - x0, y1 - y0) * .45
    layer = para(layer, (cx + kx * r, cy + ky * r), (cx - kx * r, cy - ky * r), [(t, c) for c, t in L['para']], 'mul', L['pamt'] if amt is None else amt)
    if rims:
        for (dx, dy), c, a in L['rim']: layer = rim(layer, dx * rimw, dy * rimw, c, a, .5)
    return layer


def paste_rgba(dst, src, ox, oy):
    """premultiplied src over premultiplied dst at an offset (in place)"""
    h, w = dst.shape[:2]; sh, sw = src.shape[:2]; x0, y0 = max(0, ox), max(0, oy); x1, y1 = min(w, ox + sw), min(h, oy + sh)
    if x1 <= x0 or y1 <= y0: return dst
    s = src[y0 - oy:y1 - oy, x0 - ox:x1 - ox]; d = dst[y0:y1, x0:x1]; dst[y0:y1, x0:x1] = s + d * (1 - s[..., 3:4]); return dst


def lover(a, b):
    """premultiplied layer b over layer a"""
    return b + a * (1 - b[..., 3:4])


# ------------------------------------------------------------------ arms and hands
def hpal(L, skin, nail=None):
    """palette for db.hands in this light"""
    b, sh, d = (tone(L, skin, i) for i in range(3)); n = tone(L, nail or (mixc(skin[0], '#ffffff', .35), mixc(skin[1], '#ffffff', .2), skin[2]), 0)
    return dict(skin=b, shade=sh, line=L['line'], nail=n, nail_shade=mixc(n, sh, .6), nail_dark=tone(L, '#2a1c20', 0), light=np.clip(b * 1.05 + .06, 0, 1), band=L['line'])


def ld3(L, z=.45): return (L['ld'][0], L['ld'][1], z)


def hand(cel, L, skin, pose, at, s, R, side='R', bld=None, w=None, seed=0, dist=7., res=None, pivot=(0, 0, 0), style=None, nail=None, pal=None, clip=None, ldir=None, **k):
    """one inked hand: wrist (pivot) at px `at`, s = px per hand unit (wrist to knuckle)"""
    from db import hands as HD
    cam = HD.Cam(float(at[0]), float(at[1]), s, dist); st = dict(arm=.5); st.update(style or {}); r = res or int(np.clip(s * 2.6, 130, 640))
    h = HD.Hand(pose, side, R, (0, 0, 0), bld or HD.MALE, pivot=pivot, **st); g = HD.render([h], cam, ldir or ld3(L), res=r, clip=clip or (-40, -40, W + 40, FH + 40)); HD.ink(cel, g, pal or hpal(L, skin, nail), w=w or max(1.0, lw(s * .9)[0] * .95), seed=seed, **k); return g


def arm(cel, sh, el, wr, wd, cols, w, sg=1, seed=0, over='f', muscle=0., lines=True):
    """bare arm: shoulder, elbow, wrist px; wd = half widths (shoulder, mid upper, elbow, mid fore, wrist); sg = shadow side on screen (+1 right); over 'f' forearm drawn over upper"""
    sh, el, wr = [np.asarray(v, float) for v in (sh, el, wr)]; m1 = (sh + el) / 2; m2 = el + (wr - el) * .38; n1 = A([-(el - sh)[1], (el - sh)[0]]) / (np.hypot(*(el - sh)) + 1e-6); n2 = A([-(wr - el)[1], (wr - el)[0]]) / (np.hypot(*(wr - el)) + 1e-6)
    U = limb([sh, sh + (el - sh) * .30, m1 + (el - sh) * .06, el], [wd[0] * (1 + .10 * muscle), wd[0] * (.92 - .06 * muscle), wd[1] * (1 + .12 * muscle), wd[2] * (1 - .04 * muscle)]); Fo = limb([el, m2, (m2 + wr) / 2, wr], [wd[2] * 1.02, wd[3] * (1 + .08 * muscle), (wd[3] + wd[4]) / 2 * 1.02, wd[4]]); parts = [U, Fo] if over == 'f' else [Fo, U]; J = ell(el[0], el[1], wd[2] * 1.02)
    for q in parts: outline(cel, q[0], w[0] * 1.7, cols[4], seed, tremor=.12)
    outline(cel, J, w[0] * 1.6, cols[4], seed + 1, tremor=.1)
    for q in parts: cel.fill(q[0], cols[0])
    cel.fill(J, cols[0])
    for q, wq, pts in ((U, wd[1], ((0, .10), (.30, .46), (.50, .30), (.60, .52), (.80, .30), (1, .34))), (Fo, wd[3], ((0, .30), (.28, .52), (.55, .30), (.80, .34), (1, .26)))):
        P, a, b, C = q; s_ = 1 if (a - b)[len(a) // 2, 0] * sg >= 0 else -1; E = a if s_ > 0 else b; n = len(C); t = np.linspace(0, 1, n); k = np.interp(t, [x for x, _ in pts], [y for _, y in pts])[:, None]
        cel.clip(P); cel.fill(np.vstack([C + (E - C) * k, (C + (E - C) * 1.3)[::-1]]), cols[1]); cel.unclip()
    if lines:
        cel.line(spl(A([el + n1 * wd[2] * .55 - (el - sh) * .02, el + n1 * wd[2] * .05, el - n1 * wd[2] * .30 + (wr - el) * .03]), n=6), w[2] * 1.2, cols[4], taper=(.2, .7), seed=seed + 3)
        if muscle: cel.line(spl(A([sh + (el - sh) * .30 + n1 * wd[1] * .72, sh + (el - sh) * .52 + n1 * wd[1] * .45, sh + (el - sh) * .74 + n1 * wd[1] * .55]), n=6), w[1], cols[4], taper=(.4, .5), seed=seed + 4); cel.line(spl(A([sh + (el - sh) * .06 - n1 * wd[0] * .5, sh + (el - sh) * .30 - n1 * wd[0] * .12, sh + (el - sh) * .42 + n1 * wd[0] * .2]), n=6), w[1], cols[4], taper=(.4, .6), seed=seed + 5)
    return U, Fo


def band(cel, wr, d, half, length, cols, w, seed=0, bow=.18, sg=1, t0=0.):
    """wristband: wr = its hand-side centre, d = unit direction toward the elbow"""
    wr = np.asarray(wr, float); d = np.asarray(d, float); d = d / np.hypot(*d); n = A([-d[1], d[0]]); t = np.linspace(-1, 1, 11)
    E = lambda u, h: wr + d * (u * length) + n * (t * h)[:, None] + d * ((1 - t ** 2) * -bow * half)[:, None]
    P = np.vstack([E(0, half), E(1, half * 1.12)[::-1]]); cel.fill(P, cols[0]); cel.clip(P); s_ = 1 if n[0] * sg > 0 else -1; q = np.vstack([E(0, half)[5 + 2 * s_:] if s_ > 0 else E(0, half)[:4], (E(1, half * 1.12)[6:] if s_ > 0 else E(1, half * 1.12)[:5])[::-1]]); cel.fill(q, cols[1]); cel.line(E(.28, half * 1.02), w[2], cols[4], taper=(.1, .1)); cel.unclip()
    outline(cel, P, w[0] * .9, cols[4], seed, tremor=.1); return P


class Body:
    """2.5-D torso frame: origin = pit of the neck, x right, y DOWN, z forward; unit u px"""
    def __init__(s, ox, oy, u, yaw=0., pitch=0., roll=0., persp=.05):
        s.o = A([ox, oy], float); s.u = u; s.yaw, s.pitch, s.roll, s.persp = yaw, pitch, roll, persp
        a, b, c = yaw, pitch, roll; Ry = A([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]); Rx = A([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]]); Rz = A([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]]); s.R = Rz @ Rx @ Ry

    def __call__(s, P, z=0., bow=None):
        """points (x, y) or (x, y, z) -> px.  bow = (half width, depth): puts the points on a rounded front"""
        P = np.asarray(P, float)
        if P.ndim == 1: P = P[None]
        if P.shape[-1] == 2:
            zz = np.broadcast_to(np.asarray(z, float), P.shape[:-1]).copy()
            if bow is not None: zz = zz + bow[1] * np.sqrt(np.clip(1 - (np.abs(P[:, 0]) / bow[0]) ** 2.2, 0, 1))
            P = np.c_[P, zz]
        q = P @ s.R.T; f = 1 / (1 - np.clip(q[:, 2] * s.persp, -.5, .5)); return np.c_[q[:, 0] * f * s.u, q[:, 1] * f * s.u] + s.o

    def c(s, P, closed=True, corners=(), n=8, **k): return spl(s(P, **k), closed, corners=corners, n=n)
