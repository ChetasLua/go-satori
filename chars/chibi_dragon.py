"""Shenron as one living ribbon (storybook), and the seven balls.
    shenron(d, P, S, t)   P = centre line head -> tail (n,2) px, S = head half height px;  path(f, ...) makes an undulating one."""
import numpy as np, math
from .chibi_core import *
from .chibi_story import LINE, ball4
G = dict(skin='#7dbb72', skin2='#5b9a63', deep='#3f8560', deep2='#2f6a53', belly='#f6deb0', belly2='#dfb985', horn='#a8744a', horn2='#7f5236', line=LINE, whisk='#9ccb7a')


def path(f, x0=1560., y0=250., L=2350., amp=92., wl=760., wob=1.0, rise=.02, period=40., n=240, loop=None):
    """undulating body: a wave travels from head to tail; head at (x0, y0), tail trails to the left.  arc-length parametrised"""
    s = np.linspace(0, L, n); ph = 2 * pi * (s / wl - f / period)
    env = np.interp(s / L, [0, .10, .35, 1.0], [.45, .8, 1.0, 1.12]); x = x0 - s * .80 + 34 * wob * np.sin(2 * ph + .6) * env; y = y0 + rise * s + amp * env * np.sin(ph) + 16 * np.sin(2 * pi * f / period * .5 + s / 300.)
    P = np.c_[x, y]; d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; t = np.linspace(0, d[-1], n); return np.c_[np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])]


def frames(P):
    T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; return T, np.c_[-T[:, 1], T[:, 0]]


def width(s, S):
    """half width along the body (s 0 head .. 1 tail tip)"""
    return S * np.interp(s, [0, .05, .16, .45, .75, .92, 1.0], [.50, .52, .62, .60, .42, .20, .015])


def limb_claw(d, o, ang, S, k, seed, wave=0., far=False):
    n = N().at(o[0], o[1], ang, S); col, sh = G['skin'], G['skin2']
    a = tube([(0, 0), (.10, .30), (.02 + wave * .2, .58), (.16 + wave * .4, .84)], (.20, .19, .15, .12), cap0=.4); d.part(n(a), col, sh, w=1., seed=seed, shade=[n(a)] if far else [n(a + [-.14, .0])])
    h = n.at(.16 + wave * .4, .84, -.2 - wave * .5)
    for i, t_ in enumerate((-.75, -.1, .55)):
        q = h.at(0, .02, t_); d.part(q(tube([(0, 0), (.0, .16), (0, .26)], (.10, .085, .07), cap=.8)), col, sh, w=.9, seed=seed + 1 + i); d.part(q(spl([(-.055, .26), (.0, .42), (.055, .26)], True, n=4, corners=(1,))), '#fff4da', '#e2cba8', w=.7, seed=seed + 4 + i)
    d.part(h(ell(0, .04, .12, .11)), col, sh, w=0)


def shenron(d, P, S, t=0., legs=(.17, .50), head_ang=None, mouth=1., seed=7, belly=1., chunks=26, look=0., fins=True, hs=1.22):
    """draw tail -> head so later coils cover earlier ones"""
    T, Nn = frames(P); n = len(P); s = np.linspace(0, 1, n); w = width(s, S)
    an = np.unwrap(np.arctan2(T[:, 1], T[:, 0])); dsq = np.hypot(*np.gradient(P, axis=0).T); kv = np.abs(np.convolve(np.pad(np.gradient(an) / (dsq + 1e-6), 4, 'edge'), np.hanning(9) / 4., 'valid')); w = np.minimum(w, .92 / (kv + 1e-6)); w = np.convolve(np.pad(w, 4, 'edge'), np.hanning(9) / np.hanning(9).sum(), 'valid')
    up = np.where(Nn[:, 1] <= 0, 1., -1.)
    # the back is the side that is uppermost in a side view of a left-trailing body: keep it continuous along the body
    sgn = 1. if np.median(Nn[:, 1]) < 0 else -1.; B = Nn * sgn
    bl = belly * (.38 + .34 * np.sin(2 * pi * s * 2.3 + t * .03))                    # where the belly plate starts across the width: the body rolls a little
    top, bot = P + B * w[:, None], P - B * w[:, None]; bel = P - B * (w * np.clip(bl, .12, .80))[:, None]
    dl = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; fs = np.arange(S * .5, dl[-1] - S * .6, S * .40); fi = np.interp(fs, dl, np.arange(n))
    lg = [(q, i) for i, q in enumerate(legs)]
    edges = np.linspace(0, n - 1, chunks + 1).astype(int)
    for c_ in range(chunks - 1, -1, -1):
        a, b = max(0, edges[c_] - 1), min(n - 1, edges[c_ + 1] + 2)
        # dorsal fins under this chunk's back edge
        if fins:
            for j, q in enumerate(fi):
                i = int(q)
                if not (edges[c_] <= i < edges[c_ + 1]): continue
                wi = w[i]; h_ = (.30 + .10 * math.sin(j * 1.7)) * S * (.35 + .65 * min(1, wi / (S * .55))); i0, i1 = max(0, int(q - S * .28 / (dl[-1] / n))), min(n - 1, int(q + S * .16 / (dl[-1] / n)))
                A0, A1 = P[i0] + B[i0] * w[i0] * .86, P[i1] + B[i1] * w[i1] * .86; tipp = P[i] + B[i] * (wi + h_) + T[i] * h_ * .65
                F = spl([A0, A0 * .45 + tipp * .55 + B[i] * h_ * .12 - T[i] * h_ * .05, tipp, A1 * .6 + tipp * .4 + T[i] * h_ * .02, A1], True, n=6, corners=(0, 2, 4)); d.part(F, G['deep'], G['deep2'], w=.85, seed=seed + j)
                d.stroke(spl([A0 * .5 + A1 * .5, (A0 + A1) * .25 + tipp * .5, tipp * .85 + A0 * .15], n=5), .4, G['deep2'])
        for q, k in lg:
            i = int(q * n)
            if edges[c_] <= i < edges[c_ + 1]: limb_claw(d, P[i] - B[i] * w[i] * .55 + T[i] * S * .18, math.atan2(-B[i][0] * -1, B[i][1] * -1) * 0 + math.atan2(T[i][1], T[i][0]) * .0 + .55 + .25 * math.sin(t * .31 + k), S * .72, k, seed + 60 + 10 * k, wave=.35 * math.sin(t * .31 + k * 1.4), far=True)
        Q = np.vstack([top[a:b + 1], bot[a:b + 1][::-1]]); d.knock(Q); d.fl.fill(Q, G['skin']); d.sc.fill(Q, G['skin2']); d.mk.fill(Q, '#000000')
        bq = np.vstack([bel[a:b + 1], bot[a:b + 1][::-1]]); d.fl.fill(bq, G['belly']); d.sc.fill(bq, G['belly2'])
        # shade: under the back's turn, and along the bottom edge
        d.mk.shade(np.vstack([(bel - B * 0 + B * (w * .20)[:, None])[a:b + 1], bel[a:b + 1][::-1]]), '#ffffff', within=Q)
        d.mk.shade(np.vstack([(bot + B * (w * .20)[:, None])[a:b + 1], bot[a:b + 1][::-1]]), '#ffffff', within=Q)
        # scales and belly plates
        for j, q in enumerate(fi):
            i = int(q)
            if not (edges[c_] <= i < edges[c_ + 1]): continue
            m0, m1 = bel[i], bot[i]; mid = (m0 + m1) / 2 + T[i] * S * .07; d.stroke(spl([m0, mid, m1], n=6), .5, mix(LINE, G['belly2'], .45), taper=(.1, .1))
            for u, o in ((.70, .0), (.35, .5), (.02, 0.), (-.25, .5)):
                i2 = min(n - 2, int(q + o * S * .40 / (dl[-1] / n)));
                if -u > bl[i2] - .16: continue
                c0 = P[i2] + B[i2] * w[i2] * u; r = S * .085 * min(1, w[i2] / (S * .5)); d.stroke(spl([c0 - B[i2] * r, c0 + T[i2] * -r * .9, c0 + B[i2] * r], n=5), .5, G['skin2'] if (j + int(u * 10)) % 3 else G['deep'], taper=(.2, .2), tmin=.4)
        d.stroke(bel[a:b + 1], .55, mix(LINE, G['skin2'], .35), taper=(0, 0)); d.stroke(top[a:b + 1], .95, taper=(0, 0)); d.stroke(bot[a:b + 1], 1.1, taper=(0, 0))
    # tail tuft
    e = P[-1]; ea = math.atan2(T[-1][1], T[-1][0]); tn = N().at(e[0], e[1], ea, S * .8)
    d.part(tn(spl([(-.22, 0), (-.05, -.22), (.30, -.34), (.24, -.12), (.62, .0), (.26, .12), (.30, .36), (-.05, .22)], True, n=5, corners=(2, 4, 6))), G['deep'], G['deep2'], w=.95, seed=seed + 90)
    ha = math.atan2(*((P[0] - P[7])[::-1])) if head_ang is None else head_ang
    return head(d, N().at(P[0][0], P[0][1], ha * .55, S * hs), t, mouth, seed + 100, look)


def head(d, h, t=0., mouth=1., seed=0, look=0.):
    """facing right; origin at the neck"""
    o = h.at(.52, -.10, -.02)
    # mane lobes behind the jaw
    for i, (a, l, w_) in enumerate(((2.05, .95, .30), (2.55, 1.10, .33), (3.10, 1.08, .34), (3.65, .95, .32), (4.15, .80, .28))):
        a += .05 * math.sin(t * .3 + i); e = A([math.cos(a), math.sin(a)]); nrm = A([-e[1], e[0]]); c0 = A([-.22, .02]) + e * .25
        d.part(o(spl([c0 - nrm * w_, c0 + e * l * .55 - nrm * w_ * 1.1, c0 + e * l, c0 + e * l * .62 + nrm * w_ * .9, c0 + nrm * w_], True, n=6, corners=(2,))), G['deep'], G['deep2'], w=1., seed=seed + i); d.stroke(o(spl([c0 + e * .2, c0 + e * l * .75], n=5) if False else A([c0 + e * .2, c0 + e * l * .78])), .45, G['deep2'])
    # far antler, far whisker
    def antler(x, y, k, s0):
        an = o.at(x, y, .12 * k)
        U_ = d.union([an(tube([(0, .12), (-.06, -.30), (-.30, -.66), (-.36, -1.02)], (.12, .115, .10, .07), cap=.9, cap0=.5)), an(tube([(-.16, -.46), (.06, -.66), (.12, -.88)], (.085, .085, .06), cap=.9)), an(tube([(-.26, -.60), (-.52, -.66), (-.66, -.84)], (.08, .08, .055), cap=.9))])[0]
        d.part(U_, G['horn'], G['horn2'], w=1., seed=s0, shade=[an(tube([(.08, .12), (.02, -.30), (-.22, -.66), (-.28, -1.02)], (.06, .06, .05, .04)))]); d.stroke(an(spl([(-.07, .0), (-.10, -.24), (-.22, -.46)], n=5)), .45, G['horn2'])
    antler(.30, -.66, 1., seed + 20)
    def whisk(x, y, k, s0):
        q = [A([x, y])]; L = 2.1
        for i in range(1, 9):
            u = i / 8.; q.append(A([x - L * u * .95 * k + .25 * math.sin(u * 3.1), y + (.10 + .30 * u) * math.sin(u * 5.6 - t * .31) + .10 * u + .16 * math.sin(u * 3.1)]))
        W_ = tube(q, (.07, .07, .065, .055, .045, .03), cap=.8, cap0=1.); d.part(o(W_), G['whisk'], G['skin2'], w=.85, seed=s0)
    whisk(1.12, .18, .8, seed + 26)
    sk = spl([(-.62, .18), (-.74, -.22), (-.60, -.62), (-.22, -.86), (.18, -.84), (.46, -.66), (.62, -.50), (.92, -.46), (1.24, -.38), (1.46, -.16), (1.46, .08), (1.28, .22), (1.18, .46), (.86, .62), (.36, .70), (-.16, .64), (-.48, .44)], True, n=8)
    Q = o(sk); d.part(Q, G['skin'], G['skin2'], w=1.05, seed=seed + 30)
    jw = spl([(1.30, .22), (.90, .30), (.44, .40), (.10, .58), (-.2, .76), (.36, .9), (1.2, .7), (1.5, .3)], True, n=6); d.detail(o(jw), G['belly'], within=Q, sh=G['belly2'])
    d.shade(o(spl([(-.8, .2), (-.3, .34), (.1, .60), (.5, .66), (.9, .56), (1.2, .40), (1.3, .9), (-.8, .9)], True, n=6)), within=Q)
    d.shade(o(spl([(.40, -.66), (.56, -.40), (.50, -.10), (.62, -.2), (.74, -.44)], True, n=5)), within=Q)
    # mouth
    m = .10 + .22 * mouth; Mq = o(spl([(.46, .33), (.72, .30), (1.02, .25), (1.33, .18), (1.22, .20 + m), (.90, .30 + m), (.62, .34 + m * .8)], True, n=7, corners=(0, 3))); d.part(Mq, '#9d2f35', '#7d222c', w=.75, seed=seed + 31); d.detail(o(ell(.95, .30 + m * .95, .26, .12)), '#f28d84', within=Mq, sh='#e27a74')
    for x in (.66, 1.18): d.part(o(spl([(x - .05, .285 - (x - .66) * .11), (x, .40 - (x - .66) * .11), (x + .05, .275 - (x - .66) * .11)], True, n=3, corners=(0, 1, 2))), '#fffaf0', '#e8dccb', w=.5)
    d.stroke(o(spl([(.46, .33), (.36, .28), (.30, .18)], n=5)), .8)
    d.stroke(o(spl([(1.18, -.20), (1.28, -.16), (1.26, -.06), (1.17, -.08)], n=6)), .75); d.stroke(o(spl([(.98, -.40), (1.10, -.30), (1.26, -.30)], n=6)), .5, G['deep'])
    # brow crest, eyes, blush
    d.part(o(spl([(.20, -.60), (.44, -.72), (.72, -.52), (.64, -.40), (.44, -.50)], True, n=6, corners=(0,))), G['deep'], G['deep2'], w=.9, seed=seed + 33)
    d.part(o(spl([(-.34, -.60), (-.10, -.74), (.20, -.58), (.14, -.42), (-.08, -.52)], True, n=6, corners=(0,))), G['deep'], G['deep2'], w=.9, seed=seed + 34)
    for x, y, k in ((.02, -.22, 1.), (.56, -.24, .82)):
        d.detail(o(ell(x + look * .03, y, .075 * k, .115)), '#33201c', within=Q, sh='#33201c'); d.detail(o(ell(x + .02 + look * .03, y - .045, .026, .034)), '#ffffff', within=Q, sh='#ffffff')
    d.detail(o(ell(-.04, .08, .17, .09)), mix(G['skin'], '#f79f88', .45), within=Q, sh=mix(G['skin2'], '#f79f88', .4))
    for x, y in ((-.42, -.10), (-.34, .16), (-.50, -.36), (-.16, -.62)): d.stroke(o(spl([(x - .07, y - .03), (x, y + .05), (x + .08, y - .02)], n=5)), .5, G['deep'])
    antler(-.06, -.72, -1., seed + 40)
    # ear fan
    d.part(o(spl([(-.44, -.10), (-.72, -.40), (-1.02, -.36), (-.90, -.14), (-1.08, .06), (-.82, .14), (-.82, .34), (-.52, .22)], True, n=5, corners=(2, 4, 6))), mix(G['skin'], G['deep'], .35), G['deep2'], w=1., seed=seed + 44); d.stroke(o(A([(-.50, .0), (-.86, -.24)])), .5, G['deep2']); d.stroke(o(A([(-.52, .10), (-.86, .08)])), .5, G['deep2'])
    whisk(1.00, .24, 1., seed + 46)
    return o


def bounce_ball(d, x, gy, r, t, phase=0., star_n=4, hop=1.0, V=8., seed=0, shadow=True):
    """a ball that rolls and lands on every beat (10 f): squash on contact, true arc between"""
    u = ((t + phase) % 10.) / 10.; y = 4 * u * (1 - u) * hop * r * 2.1; q = max(0., 1 - min(u, 1 - u) / .10) if hop > .05 else 0.; sq = .18 * q * min(1, hop)
    if shadow: d.fl.fill(ell(x, gy + r * .06, r * (1.0 - .25 * min(1, y / (r * 3))), r * .20), '#8a5f3a', .30)
    n = N().at(x, gy - r * (1 - sq) - y, 0., 1 + sq * .7, 1 - sq); ball4(d, n, r, star_n, spin=(t * V) / r * .55 + seed, w=.95, seed=seed)
