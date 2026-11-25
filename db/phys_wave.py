"""phys wave: the far explosion.  Light first (a ridge that lifts, unzips from the top and dissolves; a dome of light), then the condensation ring, then the pressure front that races over the sea.  (PHYS lane)"""
from .phys_core import *
from .phys_dust import lobes_layer, lobe
from .phys_soft import cells2d
from .phys_water import drop, _pw

VIOLET = dict(core='#fbf7ff', hot='#e6d2ff', mid='#b070ff', edge='#6a1fd0', deep='#2a0a5e', ink='#12052e')
GOLDL = dict(core='#fffbe6', hot='#ffe99a', mid='#ffc23a', edge='#d9761a', deep='#5a2c0e', ink='#2a1206')
BLUEL = dict(core='#f4fdff', hot='#bfeaff', mid='#62c8ff', edge='#2a6fe0', deep='#0e2a70', ink='#061030')
LIGHTS = dict(violet=VIOLET, gold=GOLDL, blue=BLUEL)
def _pl(p): return LIGHTS[p] if isinstance(p, str) else p


# ------------------------------------------------------------- mountains
def ridge_line(x0, x1, base, peaks, seed=0, rough=.16, step=9.):
    """angular mountain skyline.  peaks = [(x, height px, half width px), ...] -> polyline left to right"""
    xs = np.arange(x0, x1 + step, step); y = np.zeros_like(xs)
    for i, (px, ph, pw) in enumerate(peaks): d = np.abs(xs - px) / pw; k = np.where(xs < px, 1 + .25 * (h1(seed, i, 1) - .5), 1 + .25 * (h1(seed, i, 2) - .5)); y = np.maximum(y, ph * np.clip(1 - d * k, 0, 1) ** (1.05 + .3 * h1(seed, i, 3)))
    n = noise(xs / 46., seed * 3.3) * .6 + np.abs(noise(xs / 21., seed * 3.3 + 9)) * .5 + noise(xs / 9.5, seed * 5.1) * .28; y = y * (1 + rough * n) + 3 * n; y = np.maximum(y, 0); keep = np.ones(len(xs), bool)
    for i in range(1, len(xs) - 1): keep[i] = h1(seed, i, 7) < .72 or y[i] > max(y[i - 1], y[i + 1])
    return np.c_[xs, base - y][keep]


class Ridge:
    """A far ridge that can be cut, lift, unzip from the top and turn into light.
    rg = Ridge(250, 1700, 560, [(700, 130, 260), ...], seed=2); rg.draw(cel, f)  with f_lift (frame the lift starts) ; pal from LIGHTS for its light."""

    def __init__(self, x0, x1, base, peaks, seed=0, n=150, col=('#2a2148', '#140f28', '#0a0716'), moon=(-.6, -.5), cut=.0):
        self.L = ridge_line(x0, x1, base, peaks, seed); self.base, self.seed, self.col, self.moon = base, seed, col, moon; self.poly = np.vstack([self.L, [x1, base + 4], [x0, base + 4]]); self.hmax = base - self.L[:, 1].min(); self.cx = float((x0 + x1) / 2); self.w = x1 - x0; _sq = np.array([1., .42]); self.C = [(q / _sq, q.mean(0) / _sq) for q in cells2d(self.poly * _sq, n, seed + 3)]; self.peaks = peaks

    def still(self, cel, lit=None, amt=1.):
        """unbroken ridge with facets.  lit = colour of a light behind / on it"""
        a, b, c = self.col; cel.fill(self.poly, b); L = self.L; cel.clip(self.poly)
        for i in range(1, len(L) - 1):                       # spur facets fall from every local summit
            if L[i, 1] < L[i - 1, 1] and L[i, 1] <= L[i + 1, 1]:
                h = self.base - L[i, 1]
                if h < 8: continue
                s = -1 if self.moon[0] < 0 else 1; w = h * (.5 + .7 * h1(self.seed, i, 11)); p = L[i]; k = h * (.55 + .4 * h1(self.seed, i, 12)); ex = h * .12 * (h1(self.seed, i, 14) - .5); P = np.array([p, p + [s * w * .30 + ex, k * .42], p + [s * w * .22, k * .58], p + [s * w * .62, k], p + [s * w * .12 + ex, k * .86], p + [ex * .5, k * .5]]); cel.fill(P, a if lit is None else hexmix(a, lit, amt)); Q = np.array([p, p + [-s * w * .06, k * .5], p + [-s * w * .34, k * 1.0], p + [-s * w * .55, k * 1.05], p + [-s * w * .3, k * .55]]); cel.fill(Q, c)
        cel.unclip(); return cel

    def breakup(self, cel, t, pal='violet', accel=420., spread=.34, top=.34, under=True, centre=None, outward=.35, lift=16.):
        """t seconds since the lift: the whole range comes up off the cut, a seam of light opens under it; then it unzips from the summits down, shards rise, shrink and burn white"""
        pal = _pl(pal); cx = self.cx if centre is None else centre; e = clamp(t * 4.5); up = lift * (1 - (1 - e) ** 2.2) + 26 * max(0., t - .1) ** 1.4
        if under: cel.fill(self.poly + [0, -up * .9], hexmix(pal['mid'], pal['core'], clamp(t * 5))); cel.fill(np.array([[self.L[0, 0] - 10, self.base + 3], [self.L[0, 0] + 60, self.base - up - 4], [self.L[-1, 0] - 60, self.base - up - 4], [self.L[-1, 0] + 10, self.base + 3]]), pal['core'])
        cel.ctx.save(); cel.ctx.translate(0, -up); self.still(cel); cel.ctx.restore(); fly = []
        for i, (q, c) in enumerate(self.C):
            loc = clamp((self.base - c[1]) / max(8., float(np.interp(c[0], self.L[:, 0], self.base - self.L[:, 1])))); t0 = .06 + (1 - loc) * top + abs(c[0] - cx) / self.w * spread + .05 * h1(self.seed, i, 1); s = t - t0
            if s <= 0:                                       # still holds; hairlines open just before it goes
                k = clamp((s + .07) / .07)
                if k > 0:
                    for a_, b_ in zip(q, np.roll(q, -1, 0)): wedge(cel, a_ + [0, -up], b_ + [0, -up], 2.2 * k, 1.2 * k, pal['hot'])
                continue
            cel.fill(c + (q - c) * 1.05 + [0, -up], hexmix(pal['mid'], pal['core'], clamp(s * 7))); g = 1 - s / (.34 + .36 * h1(self.seed, i, 2)) * .9
            if g > .05: fly.append((i, q, c, s, g))
        for i, q, c, s, g in fly:
            a = (h1(self.seed, i, 3) - .5) * 3 * s; R2 = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]); d = np.array([(c[0] - cx) * outward * s * (1 + s), -up - .5 * accel * s * s * (.5 + h1(self.seed, i, 4)) - 70 * s]); st = 1 + s * 2.4; P = (q - c) * g @ R2.T * np.array([1 / math.sqrt(st), st ** .85]) + c + d; k = clamp(s * 5.); cel.fill(P, hexmix(self.col[1], pal['edge'], min(1., k * 1.5)) if k < .5 else hexmix(pal['mid'], pal['core'], (k - .5) * 2))
        return cel


def cloud_deck(cam, t=0., alt=900., size=620., cover=.10, seed=3, light=None, hole=None, h=H, w=W, drift=(30., -10.)):
    """night cloud ceiling in true perspective (level camera).  -> (cover mask 0..1, light 0..1 on the clouds, distance m) per pixel, hard edged.
    light = (x, y, reach m, amount): a far light below that paints the bellies and the near rims;  hole = (x, y, r): a round hole pushed into the deck with a thick lip"""
    q = 2; y, x = np.mgrid[0:h // q, 0:w // q].astype(np.float64) * q + q * .5; up = np.maximum(cam.cy - y, .5); Z = cam.f * (alt - cam.eye[2]) / up; X = (x - cam.cx) / cam.f * Z; Y = Z + cam.eye[1]; X2, Y2 = X + drift[0] * t, Y + drift[1] * t; lip = 0.
    if hole is not None: dx, dy = X - hole[0], Y - hole[1]; d = np.hypot(dx, dy) + 1e-6; r = hole[2]; k = np.where(d > r, r * np.exp(-(d - r) / (r * .55 + 80)), d * .95); X2 = X2 - dx / d * k; Y2 = Y2 - dy / d * k; lip = .42 * np.exp(-((d - r * 1.10) / (r * .10 + 30)) ** 2) - 1.6 * np.clip((r - d) / 60., 0, 1)
    n = np.zeros_like(X); a = 1.; s = size
    for o in range(4): vis = np.clip(s / (Z / cam.f * 4.) - .6, 0, 1); n += a * vis * noise(X2 / s, Y2 / s * 1.0, o * 7.3 + t * .03, seed + o); a *= .60; s *= .45
    n = n + lip; up_ = lambda v: cv2.resize(v.astype(f32), (w, h), interpolation=cv2.INTER_CUBIC); N_ = up_(np.clip(n, -3, 3)); gw = np.clip((np.abs(cv2.Sobel(N_, cv2.CV_32F, 1, 0, ksize=1)) + np.abs(cv2.Sobel(N_, cv2.CV_32F, 0, 1, ksize=1))) * .6, .008, .10); m = np.clip((N_ - cover) / gw + .5, 0, 1) * (np.arange(h)[:, None] < cam.cy - 2); li = np.zeros_like(N_)
    if light is not None:
        dx, dy = X - light[0], Y - light[1]; d = np.hypot(dx, dy) + 1e-6; fall = up_(np.exp(-d / light[2]) * light[3]); bel = np.clip((N_ - cover) * 1.15, 0, 1); li = fall * (1.15 - 1.05 * bel) + .45 * fall * np.clip(1 - (N_ - cover) / .20, 0, 1)
    return m.astype(f32), li.astype(f32), up_(Z)


# ------------------------------------------------------------- dome of light
def dome(cel, cx, cy, r, t, pal='violet', seed=0, squash=.92, boil=.05, core=.72, spikes=True, rays=0):
    """half dome of light standing on the horizon line cy: hard layers, a boiling hand-drawn edge that climbs, white inside"""
    pal = _pl(pal)
    if r < 2: return cel
    th = np.linspace(math.pi, TAU, 260)
    def edge(k, amp, sd, lick=0.):
        n = noise(np.cos(th) * 2.3 + sd, np.sin(th) * 2.3 + t * 1.2, t * .9, seed) * .5 + (1 - np.abs(noise(np.cos(th) * 5.2 + sd, np.sin(th) * 5.2 + t * 1.8, t * .8, seed)) * 2.2) * .55; rr = r * k * (1 + amp * n); 
        if lick: u = (th * 11 / math.pi + sd) % 1; rr = rr * (1 + lick * u ** 2.2 * (.5 + .5 * np.sin(th * 5 + sd)))
        return np.c_[cx + np.cos(th) * rr, cy + np.sin(th) * rr * squash]
    for k, amp, c, sd, lk in ((1.00, boil, pal['deep'], 1., 0), (.965, boil * .9, pal['edge'], 2., 0), (.90, boil * .8, pal['mid'], 3., 0), (.825, boil * .8, pal['hot'], 4., 0), (core, boil, pal['core'], 5., 0)):
        E = edge(k, amp, sd, lk); cel.fill(np.vstack([E, [[cx + r * k * 1.02, cy], [cx - r * k * 1.02, cy]]]), c)
    for i in range(int(16 * min(1., r / 160))):              # darker scallops turning over inside the shell
        a = math.pi + math.pi * (i + h1(seed, i, 1)) / 16; q = .70 + .22 * ((h1(seed, i, 2) + t * .55) % 1.); p = np.array([cx + math.cos(a) * r * q, cy + math.sin(a) * r * q * squash]); w = r * (.04 + .06 * h1(seed, i, 3)); v = np.linspace(0, 1, 12); th2 = a + math.pi / 2 + 2.2 * (v - .5); thin(cel, np.c_[p[0] + np.cos(a + (v - .5) * .30) * 0 + w * np.cos(th2) * 2.0, p[1] + w * np.sin(th2) * 2.0], max(1., r * .012), pal['mid'], .9, .4, .4, 14)
    if rays:
        for i in range(rays): a = math.pi * (1 + (i + h1(seed, i, 7)) / rays); l0, l1 = r * 1.05, r * (1.5 + 1.8 * h1(seed, i, 8)); d = np.array([math.cos(a), math.sin(a) * squash]); wedge(cel, np.array([cx, cy]) + d * l0, np.array([cx, cy]) + d * l1, r * .035 * h1(seed, i, 9), 0, pal['hot'], .85)
    return cel


def fireball(cam, t, centre, r, seed=0, n=60, pal='violet', life=3.0, ss=1.25, rise=8., **k):
    """a close explosion as boiling lobes: hot seams, dark smoke skin, hard cel tones.  -> RGBA layer"""
    P = _pl(pal); L = []; q = t / life
    for i in range(n):
        z = 1 - 2 * (i + .5) / n; th = i * 2.399963 + h1(seed, i, 1); s = math.sqrt(1 - z * z); g = 1 - math.exp(-4.5 * t * (.8 + .5 * h1(seed, i, 2))); rr = r * g * (.62 + .30 * h1(seed, i, 3)); L.append(lobe(centre[0] + math.cos(th) * s * rr, centre[1] + math.sin(th) * s * rr * .8, centre[2] + z * rr + rise * t * .12 * r / 8, r * (.20 + .24 * h1(seed, i, 4)) * (.4 + .6 * g), seed * 91 + i, t * 1.6 * (1 if math.cos(th) > 0 else -1) + i, math.pi / 2, clamp(q * 1.3 - .35) * 1.05, clamp(1.05 - q * 1.5 + .2 * h1(seed, i, 5)), -1e9))
    return lobes_layer(cam, L, dict(lit=P['edge'], mid=P['deep'], dark=P['ink'], ink=P['ink'], rim=P['hot']), ss=ss, heat=[(.12, P['edge']), (.34, P['mid']), (.58, P['hot']), (.80, P['core'])], **k)


def condensation(cel, cx, cy, rx, ry, t, w=26., pal='violet', seed=0, gaps=.25, back=True, front=True, alpha=1.):
    """condensation ring: one thin torus in perspective, hair-thin far, fat near, its outer edge scalloped, wearing through in a few places as gaps grows"""
    pal = _pl(pal); a = np.linspace(0, TAU, 420); ca, sa = np.cos(a), np.sin(a); per = .16 + .84 * ((sa + 1) * .5) ** 1.4; sc = 1 + .55 * np.abs(noise(ca * 7, sa * 7, t * 1.1, seed)) + .25 * np.abs(noise(ca * 19, sa * 19, t * 1.6, seed)); hole = smooth(gaps - .30, gaps + .05, noise(ca * 2.2 + 9, sa * 2.2, t * .5, seed) * .5 + .5); ww = w * per * hole; X = np.c_[cx + ca * rx, cy + sa * ry]; N = np.c_[ca * ry, sa * rx]; N /= np.hypot(*N.T)[:, None] + 1e-6
    for half, on in ((sa < 0, back), (sa >= 0, front)):
        if not on: continue
        idx = np.where(half)[0]; runs = np.split(idx, np.where((np.diff(idx) > 1))[0] + 1)
        for run in runs:
            if len(run) < 6: continue
            O = X[run] + N[run] * (ww[run] * .55 * sc[run])[:, None]; I = X[run] - N[run] * (ww[run] * .40)[:, None]; M = X[run] + N[run] * (ww[run] * .10 * sc[run])[:, None]; cel.fill(np.vstack([O + [0, w * .10], (I + [0, w * .16])[::-1]]), pal['edge'], alpha * .85); cel.fill(np.vstack([O, I[::-1]]), pal['hot'], alpha); cel.fill(np.vstack([O, M[::-1]]), pal['core'], alpha)
    return cel


# ------------------------------------------------------------- pressure front over the sea
def wall_arc(cam, centre, R_, amax=1.25, n=220, near=1.2, p=1.35):
    """azimuths and base points of the part of a ring in front of the camera"""
    d = cam.eye[:2] - np.asarray(centre); a0 = math.atan2(d[1], d[0]); u = np.linspace(-1, 1, n); a = a0 + amax * np.sign(u) * np.abs(u) ** p; P = np.c_[centre[0] + np.cos(a) * R_, centre[1] + np.sin(a) * R_, np.zeros(n)]; z = cam.cam(P)[:, 2]; return a, P, z > near


def wall(cel, cam, centre, R_, h, t, pal='violet', seed=0, lean=.35, sheet=True, pitch=.55):
    """the pressure front as a standing, leaning sheet of vapour: deep crown, lilac body, white root, every edge a row of curved claws that streams; streaks climb it; a bright lip rolls over the top.  h metres"""
    pal = _pl(pal); a, P, ok = wall_arc(cam, centre, R_, n=420, p=float(np.interp(abs(np.hypot(*(cam.eye[:2] - np.asarray(centre))) - R_), [8, 30, 140], [4.2, 3.0, 1.35])))
    if ok.sum() < 8: return cel
    a, P = a[ok], P[ok]; S = a * 100.; o = np.c_[np.cos(a), np.sin(a), np.zeros(len(a))]; n = len(a); k = cam.f * h / np.maximum(cam.cam(P)[:, 2], .5); bx = cam.project(P)[0]; on = (bx[:, 0] > -200) & (bx[:, 0] < cam.w + 200); on = on if on.sum() > 4 else np.ones(n, bool); S0 = S[on]; pk = float(np.median(k[on])); dens = np.hypot(*np.diff(bx[on], axis=0).T).sum() / max(1e-6, abs(S0[-1] - S0[0])); Hh = h * (.86 + .20 * noise(S * .5, seed, t * .8) + .10 * noise(S * 1.7, seed + 3, t * 1.3)); pr = lambda z, f: cam.project(P + o * (np.broadcast_to(f, (n,)) * h)[:, None] + np.c_[np.zeros((n, 2)), z])[0]; base = pr(np.zeros(n), .12)
    def band(zk, fw, pt, sdx, col, al=1., ln=.5, sh=0., am=(.30, 1.), tall=2.3):
        top = pr(Hh * zk, fw)[on]; hp = max(6., float(np.median(np.hypot(*(top - base[on]).T)))); px_ = float(np.clip(hp * .30, 7., pt)); lo = max(.30, 1 - tall * px_ / hp); pp = px_ / max(dens, 1e-6); e = claw_edge(S0[0] - sh, S0[-1] - sh, pp, seed * 11 + sdx, ln, am, 9); q = claw_band(S0, base[on], top, e[0] + sh, e[1], lo)
        if q is not None: cel.fill(q, col, al)
    tiers = [(1.0, lean, 120., pal['edge'], .45), (.95, lean * .9, 96., pal['edge'], .75), (.88, lean * .78, 104., pal['mid'], .55), (.80, lean * .66, 84., pal['mid'], 1.), (.71, lean * .55, 92., pal['edge'], .60), (.64, lean * .46, 74., pal['hot'], 1.), (.56, lean * .38, 80., pal['mid'], .85), (.48, lean * .31, 64., pal['hot'], 1.), (.40, lean * .24, 56., pal['core'], 1.), (.31, lean * .18, 60., pal['hot'], 1.), (.24, lean * .12, 46., pal['core'], 1.)]
    for j, (zk, fw, pt, c_, al) in enumerate(tiers if sheet else tiers[3:]): band(zk, fw, pt, j + 1, c_, al, .75 - .03 * j, t * (.4 + .16 * j))
    for i in range(0, n - 1):                                # streaks climb the sheet
        if h1(seed, i, 21) > .22 or k[i] < 30 or k[i] > 620: continue
        v0 = .25 + .4 * h1(seed, i, 25); v = np.linspace(v0, min(.95, v0 + .25 + .3 * h1(seed, i, 22)), 9); X = P[i] + o[i] * (lean * h * v ** 1.5)[:, None] + np.c_[np.zeros((9, 2)), Hh[i] * v]; thin(cel, cam.project(X)[0], max(1., min(9., k[i] * .010) * (.4 + h1(seed, i, 23))), pal['core'] if h1(seed, i, 24) < .55 else pal['edge'], .85, .3, .6, 12)
    return cel


def wall_lobes(cam, centre, R_, h, t, seed=0, n=150, pal=None, roll=3.2, ss=1., layers=3, k=.24, **kw):
    """rolling spray at the foot of the front: a continuous bank of lobes whose tops run ahead of their feet.  -> RGBA layer"""
    a, P, ok = wall_arc(cam, centre, R_, n=n); L = []; pal = pal or dict(lit='#fbf7ff', mid='#d6b8ff', dark='#9457ec', ink='#5a22b8', rim='#ffffff'); gap = np.r_[np.hypot(*np.diff(P[:, :2], axis=0).T), 0]
    for i in range(n - 1):
        if not ok[i]: continue
        o = np.array([math.cos(a[i]), math.sin(a[i])]); tg = np.array([-o[1], o[0]])
        for j in range(layers):
            r = max(gap[i] * .62, h * k * (.50 + .75 * size3(h1(seed, i, j, 1), .3, .4))) * (1 - .18 * j); ph = h1(seed, i, j, 2) * TAU - t * roll; x = P[i, :2] + o * (math.cos(ph) * r * .45 + r * (.35 - .85 * j) + (h1(seed, i, j, 3) - .5) * r) + tg * (h1(seed, i, j, 4) - .5) * gap[i]; L.append(lobe(x[0], x[1], r * (.30 + .30 * math.sin(ph) + .75 * j), r, seed * 77 + i * 3 + j, t * roll + i, 0., .12 * h1(seed, i, j, 5), 0., -.2))
    return lobes_layer(cam, L, pal, ss=ss, **{**dict(light=(-.15, -.85, .42), t_lit=.64, t_dark=.16, ink=.8), **kw})


def spray_sheet(cel, cam, t, n=90, seed=0, pal='night', vp=None, power=1., col=None):
    # streaks start far and rush past the lens
    """spray torn off wave tops, streaming at the lens in true perspective (last frames before the front arrives)"""
    for i in range(n):
        u = (h1(seed, i, 1) + t * (1.1 + .8 * h1(seed, i, 2)) * power) % 1.; z = 60 * (1 - u) ** 2 + 3.2; x = (h1(seed, i, 3) - .5) * 50; h = h1(seed, i, 4) ** 2 * 3.5; a = cam.eye + np.array([x, z, h - cam.eye[2] * .6]); b = a + np.array([0, 4.5 * (.5 + u), 0]); q, zz = cam.project(np.array([a, b]))
        if zz.min() < .4: continue
        wedge(cel, q[0], q[1], min(9., max(.8, .03 * cam.f / zz[0])), .2, col or '#fbf7ff', .85)
    return cel
