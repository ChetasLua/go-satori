"""ki aura: flame that hugs any figure mask, streams along one flow, tears off in tongues.  See db/ki.py for usage."""
from .ki_core import *


class Hug:
    """distance field and flow of a figure mask (1/4 res): roots on the outline, streamlines that hug the body then converge along `up`"""
    def __init__(self, mask, up=(0., -1.), pull=.55, close=.16):
        q = 4; self.q = q; m = cv2.resize(mask.astype(f32), (W // q, H // q), interpolation=cv2.INTER_AREA); self.m = m; b = (m > .5).astype(np.uint8); yy = np.nonzero(b.any(1))[0]; U0 = (yy[-1] - yy[0] + 1) if len(yy) else 20
        kr = max(3, int(U0 * close)); pad = kr + 8; mp = cv2.copyMakeBorder(b, pad, pad, pad, pad, cv2.BORDER_REPLICATE); k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kr + 1, 2 * kr + 1))
        mp = cv2.morphologyEx(mp, cv2.MORPH_CLOSE, k); self.d = (cv2.distanceTransform(1 - mp, cv2.DIST_L2, 5)[pad:-pad, pad:-pad] * q).astype(f32); mp = mp[pad:-pad, pad:-pad]
        g = cv2.GaussianBlur(self.d, (0, 0), 3.0); gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3); n = np.sqrt(gx * gx + gy * gy) + 1e-6; self.nx, self.ny = gx / n, gy / n
        ys, xs = np.nonzero(mp); self.any = len(xs) > 12
        if not self.any: return
        self.up = np.array(up, float) / np.hypot(*up); self.ax = np.array([-self.up[1], self.up[0]]); self.pull = pull
        X = np.stack([xs, ys], -1) * float(q) + q / 2; self.c = X.mean(0); s = (X - self.c) @ self.ax; a = (X - self.c) @ self.up; self.half = float(np.percentile(np.abs(s), 96)) + 1; self.top = float(a.max()); self.bot = float(a.min()); self.U = max(60., self.top - self.bot)
        cs, _ = cv2.findContours(mp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); P = []
        for c in cs:
            c = c[:, 0].astype(float)
            if len(c) < 20: continue
            kk = np.hanning(7); kk /= kk.sum(); cc = np.vstack([c[-6:], c, c[:6]]); P.append(np.stack([np.convolve(cc[:, i], kk, 'same') for i in (0, 1)], -1)[6:-6] * q + q / 2)
        self.cont = P; L = [float(np.hypot(*np.diff(p, axis=0).T).sum()) for p in P]; self.len = np.array(L); self.tot = float(sum(L))

    def samp(self, A, P):
        return cv2.remap(A, (P[:, 0] / self.q - .5).astype(f32)[None], (P[:, 1] / self.q - .5).astype(f32)[None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]

    def root(self, u):
        """point at fraction u of the whole outline"""
        s = (u % 1.) * self.tot
        for p, l in zip(self.cont, self.len):
            if s <= l: return p[min(len(p) - 1, int(s / l * len(p)))]
            s -= l
        return self.cont[-1][-1]

    def dirs(self, P, wob=0., t=0., R_=None, lat=1.):
        d = self.samp(self.d, P); n = np.stack([self.samp(self.nx, P), self.samp(self.ny, P)], -1); R_ = R_ or self.U * .10; e = np.exp(-d / R_)[:, None]; s = (P - self.c) @ self.ax; a = (P - self.c) @ self.up
        hi = np.clip((a - self.top * .35) / (self.U * .55), 0, 1.4)[:, None]
        v = self.up[None] * (.62 + .5 * (1 - e)) + n * e * .80 * lat - self.ax[None] * ((s / self.half)[:, None] * self.pull * hi)
        if wob: w = wob * noise(P[:, 0] / (self.U * .34), P[:, 1] / (self.U * .34) + t * 2.2, t * .6, 3); v = v + self.ax[None] * w[:, None]
        return v / (np.hypot(v[:, 0], v[:, 1])[:, None] + 1e-6)

    def streams(self, P0, L, K=15, wob=.3, t=0., lat=1.):
        """integrate K points of streamline of length L[i] from each root -> (n, K, 2)"""
        P = np.asarray(P0, float).copy(); out = [P.copy()]; st = np.asarray(L, float) / (K - 1)
        for k in range(K - 1):
            v = self.dirs(P, wob, t, lat=lat); Pm = P + v * st[:, None] * .5; P = P + self.dirs(Pm, wob, t, lat=lat) * st[:, None]; out.append(P.copy())
        return np.stack(out, 1)


_AUR2 = {'gold': ('#b8410a', '#ee8412', '#ffc23a', '#ffeeb0'), 'blue': ('#1330a8', '#2370f0', '#62c8ff', '#d8f4ff'), 'base': ('#5e98e0', '#93c4fb', '#c8e6ff', '#f4fbff'),
         'violet': ('#36067c', '#6a1ce0', '#a45cff', '#efdcff'), 'silver': ('#44587e', '#8097ba', '#bccbe2', '#f2f7ff'), 'red': ('#5e060a', '#b8181a', '#f0622e', '#ffd8b4')}


def _parcels(hug, n, t, seed, hmax, wfac, hz=(1.5, 2.6), under=-.45, hbias=None, lat=1., K=15, wob=.3, inset=0., lead=.55):
    """living tongues: (spine, length, belly, seed, tear 0..1) ; attached while growing, torn and rising at the end of life"""
    roots, Ls, info = [], [], []
    for i in range(n):
        f_ = hz[0] + (hz[1] - hz[0]) * h1(seed, i, 1); x = t * f_ + h1(seed, i, 2); c = math.floor(x); tau = x - c; sd = int(seed * 131 + i * 17 + c * 7) & 0xffffff
        p = hug.root((i + .9 * (h1(sd, 3) - .5)) / n); nn = np.array([hug.samp(hug.nx, p[None])[0], hug.samp(hug.ny, p[None])[0]]); du = float(nn @ hug.up)
        if du < under: continue
        z = bms(h1(sd, 4), .24, .36); a = ((p - hug.c) @ hug.up - hug.bot) / hug.U                   # 0 feet .. 1 top
        g = (lead + (1.2 - lead) * a) * (.75 + .45 * max(0, du)) * (hbias(p, a, du) if hbias else 1.); L = hmax * z * g
        if L < 14: continue
        grow = min(1., tau / .52); grow = 1 - (1 - grow) ** 2.2; full = L * (1.0 if tau < .52 else 1 + .55 * (tau - .52) / .48)
        roots.append(p - nn * inset); Ls.append(full * 1.02); info.append((L, full, grow, tau, z, sd))
    if not roots: return []
    S = hug.streams(np.array(roots), np.array(Ls), K, wob, t, lat); out = []
    for P, (L, full, grow, tau, z, sd) in zip(S, info):
        pth = Path(P, max(5., full / 30.)); b0 = wfac * L * (.80 + .45 * h1(sd, 5)) * (1.15 - .3 * z); hd = pth.L * (grow if tau < .52 else 1.); sway = L * .075 * (.5 + z)
        ph = TAU * (.95 * t * 1.6 + h1(sd, 6)); s0 = 0.; b = b0; tear = 0.
        if tau > .58: tear = (tau - .58) / .42; s0 = hd * (.06 + .80 * tear ** 1.2); b = b0 * (1 - tear) ** .8 * .9
        v = lin(K); s = s0 + (hd - s0) * v; Q = pth.at(s, sway * v ** 1.35 * np.sin(TAU * .95 * v - ph)); ln = hd - s0
        if ln < 10 or b < 1: continue
        out.append((Q, ln, b, sd, tear))
    return out


class Aura:
    """back: layer behind the figure; front: wisps over it; light: additive; hug: flow; lit(layer): figure with lit edges"""
    def __init__(self, **k): self.__dict__.update(k)

    def under(self, img): return put(img, self.back)

    def over(self, img, protect=None):
        img = put(img, self.front) if self.front is not None else img
        return img + self.light * (1 - (self.mask if protect is None else protect) * .88)[..., None] if self.light is not None else img


def envelope(f, on, off=None, up=4, flares=(), die=10):
    """power over time: rises with overshoot from frame `on`, 1 in the loop, flares=[(frame, amount)], dies from `off`"""
    if f < on: return 0.
    x = (f - on + 1) / float(up); p = min(1., x) ** .7 + (.55 * math.exp(-max(0., x - 1) * 1.4) if x >= 1 else .55 * x * x)
    for a, g in flares:
        if f >= a: p += g * min(1., (f - a + 1) / 2.) * math.exp(-max(0, f - a - 1) / 4.)
    if off is not None and f >= off: p *= max(0., 1 - (f - off) / float(die)) ** 1.3
    return p


def aura(mask, f, kind='gold', power=1., seed=1, height=.46, dens=1., up=(0., -1.), step=2, width=1., veil=.42, fill=1., ink=True, front=.5, face=None, lat=1., wob=.30, light=1., sparks=1., bottom=1., hug=None, ground=True, keep=(), fringe=1., sharp=1.):
    """flame round a figure mask (float H x W or a premult layer).  kind gold|blue|base|violet|red|silver|goldblue.
    height = tongue length / figure height; up = flow direction (opposite of travel when flying); face=(x,y,r) kept clear; power from envelope().
    keep=[(x,y,r), ...] circles (face, fists) round which the flame thins; fringe = amount of small sharp tips on the edge; sharp = tip sharpness"""
    if mask.ndim == 3: mask = mask[..., 3]
    if kind == 'silver': return aura_ui(mask, f, power, seed, face=face, up=up, hug=hug)
    if kind == 'goldblue':
        hug = hug or Hug(mask, up); B = aura(mask, f, 'blue', power, seed + 50, height * 1.20, dens * .8, up, step, width * 1.25, veil * .6, fill, ink, 0., face, lat * 1.3, wob, light * .7, sparks, bottom, hug, ground, keep, fringe, sharp); A = aura(mask, f, 'gold', power, seed, height * .92, dens, up, step, width * .80, 0., fill, ink, 0., face, lat * .9, wob, light, sparks, bottom, hug, ground, keep, fringe, sharp)
        if A.back is None: return A
        t = (f // step) * step / 24.; F = Field(4); U = hug.U                        # blue leaves that cross the gold mass, so the two weave
        for Q, L, b, sd, tear in _parcels(hug, 26, t * .9, seed + 70, U * height * .75 * min(1.2, power), .085, (1.3, 2.1), -.2, wob=wob, inset=-U * .02, K=16): F.rib(Q, lwidths(len(Q), L, b, LV, .40, .7, 1.2), U * .008)
        o = F.cel(PAL['blue']['c'], _AUR2['blue'], None, None, (max(1., U * .003), 0, 0, 0), (PAL['blue']['ink'], None, None, None), seed); A.back = stack(B.back, A.back, o); A.light = A.light + B.light * (1 - .8 * A.back[..., 3:4]); A.kind = kind
        if front: A.front = _front(hug, mask, t, seed + 21, U * height * .62, PAL['blue'], _AUR2['blue'], 40, U * .02, front, face, wob)
        return A
    ld = .55 if ground else .12; lat = lat if ground else lat * .45                                    # flying: short and close at the leading end, streaming long behind
    p = power; hug = hug or Hug(mask, up); none = Aura(back=None, front=None, light=None, hug=hug, mask=mask, kind=kind, power=p, f=f, seed=seed, face=face)
    if p <= .02 or not hug.any: return none
    P_ = pal(kind); C2 = _AUR2.get(kind, P_['c']); t = (f // step) * step / 24.; U = hug.U; hm = U * height * (.35 + .65 * p); n = int(np.clip(hug.tot / (U * .036) * dens, 18, 130))
    k = U * .008
    dist = cv2.resize(hug.d, (W, H), interpolation=cv2.INTER_LINEAR); sc = 620 * t; th = U * .040 * width * (.25 + .75 * min(p, 1.5))
    vx, vy = hug.up; nz = nimg(H, W, U * .085, t * 1.1, seed, 2, 4, 1. if abs(vy) >= abs(vx) else 3.2, 3.2 if abs(vy) >= abs(vx) else 1., ox=-vx * sc, oy=-vy * sc); s_ = ((grid()[0] - hug.c[0]) * hug.up[0] + (grid()[1] - hug.c[1]) * hug.up[1])
    lo = np.clip((s_ - hug.bot + U * .02) / (U * .16), 0, 1); band = th * (1 + .60 * nz) * lo - (1 - lo) * U * .03
    cut = (s_ - hug.bot + U * .016) * (1. if ground else 99.) + (0 if ground else 1e4)
    g = np.clip(dist / (hm * .80), 0, 1); er = 0.
    for c_ in list(keep) + ([face] if face is not None else []): dk = (grid()[0] - c_[0]) ** 2 + (grid()[1] - c_[1]) ** 2; er = er + U * .060 * np.exp(-dk / (c_[2] * 1.35) ** 2)          # the flame thins round faces and fists
    # far veil: big translucent blades
    V = Field(2); V.sdf([np.minimum(band * .8 - dist, cut), None], k)
    for Q, L, b, sd, tear in _parcels(hug, int(n * .55), t * .8, seed + 9, hm * 1.40, .115 * width, (1.1, 1.9), -.4, lat=lat * 1.30, wob=wob, lead=ld): V.rib(Q, twidths(len(Q), L, b, LV[:2], .22, 1.5 * sharp) if tear == 0 else lwidths(len(Q), L, b * .8, LV[:2], .35), k)
    if ground: V.D = np.minimum(V.D, (cut + U * .015 + U * .03 * nz)[None])
    back = V.cel((P_['c'][0], P_['c'][1]), (C2[0], C2[0]), g, (veil * .75, veil), seed=seed)
    # main flame: a thin hugging mass, long blades, and an edge made of small sharp tips; the palest tone lives only inside the blades
    M = Field(4); M.sdf([np.minimum(band - dist, cut), np.minimum(band * .62 - dist, cut - U * .006), np.minimum(band * .16 - dist, cut - U * .012), None], k)
    par = _parcels(hug, n, t, seed, hm, .112 * width, (1.5, 2.7), -.45, lat=lat * 1.2, wob=wob, inset=-U * .004, lead=ld) + _parcels(hug, 10, t * .85, seed + 31, hm * 1.35, .128 * width, (1.3, 1.9), -.1, hbias=lambda p, a, du: .55 + .6 * a, lat=lat * 1.2, wob=wob, lead=ld)
    for Q, L, b, sd, tear in par:
        if tear > .12: M.rib(Q, lwidths(len(Q), L, b * .75, LV, .34, .7, 1.3), k * .5)
        else: lick(M, Q, b, L, k, LV, sd, 3, off=.30 * (1 if h1(sd, 8) > .5 else -1), need=1.5 * sharp)
    if fringe:
        nf = int(np.clip(hug.tot / (U * .022) * fringe, 24, 200))
        for Q, L, b, sd, tear in _parcels(hug, nf, t * 1.2, seed + 61, U * (.10 + .10 * p), .125, (2.0, 3.4), -.40, lat=lat * 1.1, wob=wob * .6, K=9, inset=-U * .004, lead=ld): M.rib(Q, twidths(9, L, b, LV, .16, 1.6 * sharp) if tear < .12 else lwidths(9, L, b * .7, LV, .34), k * .5)
    if bottom and ground: _skirt(M, hug, t, seed + 3, hm * .50 * bottom, .14 * width, k)
    if ground: M.D = np.minimum(M.D, (cut + U * .022 + U * .032 * nz - .10 * U * np.clip(np.abs((grid()[0] - hug.c[0]) * hug.ax[0] + (grid()[1] - hug.c[1]) * hug.ax[1]) / hug.half - .9, 0, 1) ** 2)[None])
    if np.ndim(er): M.D = M.D - (er * np.clip(dist / (U * .004), 0, 1))[None] * np.array([.8, .9, 1.1, 1.3], f32)[:, None, None]
    lw = (max(1.1, U * .0028), 0, max(.8, U * .0017), 0) if ink else None
    main = M.cel(P_['c'], C2, g, (1., 1., fill, fill), lw, (P_['ink'], None, P_['c'][0], None), seed + int(t * 12), .014)
    I = Field(3)                                             # furnace behind the body: rising blades inside the mass
    for i in range(int(22 * dens)):
        f_ = 1.2 + 1.1 * h1(seed, i, 81); x = t * f_ + h1(seed, i, 82); c = math.floor(x); tau = x - c; sd = int(seed * 57 + i * 19 + c * 5) & 0xffffff; z = bms(h1(sd, 1), .3, .4); L = U * (.22 + .55 * z); q = hug.c + hug.ax * hug.half * (h1(sd, 2) * 2 - 1) * .95 + hug.up * (hug.bot + U * (h1(sd, 3) * .75 - .08)); q = q + hug.up * U * .30 * tau
        e = math.sin(math.pi * tau) ** .7; Q = spine(q[0], q[1], L * (.5 + .5 * e), math.atan2(hug.up[1], hug.up[0]) + .12 * (h1(sd, 4) - .5) - .25 * float((q - hug.c) @ hug.ax) / hug.half * .4, 12, .09, .8, -TAU * t * 1.3 + 6 * h1(sd, 5)); I.rib(Q, lwidths(12, L, U * (.016 + .034 * z) * e, LV[:3], .32, .7, 1.2), U * .008)
    inn = I.cel((P_['c'][1], C2[2] if kind != 'gold' else '#ffd046', P_['c'][3])); inn *= M.mask(2)[..., None]
    # flow slivers: dark ones split the pale tone, pale ones ride the amber
    S = Field(1); S2 = Field(1); N_ = 60; rr = [hug.root((i + h1(seed, i, 71)) / N_) for i in range(N_)]; LL = [U * (.55 + .5 * h1(seed, i, 72)) for i in range(N_)]; ST = hug.streams(np.array(rr), np.array(LL), 14, wob, t, lat)
    for i in range(N_):
        x = t * (1.5 + h1(seed, i, 73)) + h1(seed, i, 74); u = x - math.floor(x); pth = Path(ST[i], 8.); ln = U * (.06 + .20 * h1(seed, i, math.floor(x)) ** 1.5); a = -ln + (pth.L + ln) * u * .8; Q = pth.sub(max(0, a), min(pth.L, a + ln), lambda v: U * .018 * (h1(seed, i, 75) * 2 - .4) + U * .010 * np.sin(v * 5 + i), 10); bb = U * (.0032 + .0115 * h1(seed, i, 76) ** 2)
        (S if i % 3 else S2).rib(Q, lwidths(10, ln, bb, LV[:1], .3), 2.)
    m2 = M.mask(1); s1 = S.cel((P_['c'][0],)); s1 *= m2[..., None] * .9; s2 = S2.cel((P_['c'][3],)); s2 *= (M.mask(0) * (1 - M.mask(2)))[..., None]
    back = stack(back, main, inn, s1, s2)
    if sparks: back = stack(back, _sparks(hug, t, seed, hm, P_, int(26 * sparks * min(p, 1.6))))
    fr = None
    if front > 0: fr = _front(hug, mask, t, seed + 21, hm * .62, P_, C2, n, k, front, face, wob)
    li = None
    if light:
        li = bloom(main, .5, (6, 26, 90), (.22, .22, .26), P_['glow'], .62 * light * min(1.4, p)); li = li * (1 - .55 * main[..., 3:4])
    none.back, none.front, none.light = back, fr, li; return none


def _skirt(F, hug, t, seed, h, wf, k):
    """short tongues splaying along the ground at the feet"""
    for i in range(16):
        x = t * (1.6 + 1.2 * h1(seed, i, 1)) + h1(seed, i, 2); c = math.floor(x); tau = x - c; sd = int(seed * 77 + i * 13 + c * 5) & 0xffffff; sg = 1 if i % 2 else -1; u = h1(sd, 3); z = bms(h1(sd, 4))
        b0 = hug.c + hug.up * (hug.bot + hug.U * .012) + hug.ax * sg * hug.half * (.12 + .62 * u); L = h * z * (.5 + .9 * u) * 1.3; gr = 1 - (1 - min(1., tau / .5)) ** 2; end = (1 - max(0., tau - .55) / .45)
        a = math.atan2(*(hug.up * (.35 + .3 * (1 - u)) + hug.ax * sg * (.6 + .6 * u))[::-1]); P = spine(b0[0], b0[1], L * gr, a, 9, .06, .8, t * 9 + i, sg * (-.75 if True else 0))
        if L * gr > 8 and end > .05: F.rib(P, twidths(9, L * gr, L * wf * end, LV, .2, 1.2), k)


def _sparks(hug, t, seed, hm, P_, n):
    F = Field(2)
    for i in range(n):
        x = t * (1.0 + 1.2 * h1(seed, i, 31)) + h1(seed, i, 32); c = math.floor(x); tau = x - c; sd = int(seed * 53 + i * 11 + c * 3) & 0xffffff; p0 = hug.c + hug.ax * hug.half * (h1(sd, 1) * 2 - 1) * 1.7 + hug.up * (hug.bot + hug.U * h1(sd, 2) * .9)
        p = p0 + hug.up * hm * 1.6 * tau ** .8 + hug.ax * hm * .08 * math.sin(tau * 7 + i); ln = hug.U * (.012 + .035 * h1(sd, 3)) * (1 - tau * .5); w = hug.U * .0034 * (.5 + h1(sd, 4)) * min(1., (1 - tau) * 2.2, tau * 8)
        P = np.array([p, p - hug.up * ln * .5 + hug.ax * ln * .05, p - hug.up * ln]); F.rib(P, [np.array([.2, w, .2]), np.array([-.5, w * .55, -.5])], 1.)
    return F.cel((P_['c'][1], P_['c'][3]))


def _front(hug, mask, t, seed, h, P_, C2, n, k, amt, face, wob):
    """sparse tongues that start inside the silhouette's edge and lick over it (never over the face)"""
    F = Field(4); inset = hug.U * .018
    for Q, L, b, sd, tear in _parcels(hug, max(8, int(n * .42 * amt * 2)), t, seed, h, .115, (1.6, 2.8), -.9, hbias=lambda p, a, du: 1.1 if a < .55 else (.55 if a < .8 else .0), wob=wob, inset=inset, K=16):
        if L > h * .16: F.rib(Q, lwidths(len(Q), L, b, LV, .38, .7, 1.15), k * .4)
    o = F.cel(P_['c'], C2, None, (.82, .9, .95, 1.), (max(1., hug.U * .0022), 0, 0, 0), (P_['ink'], None, None, None), seed)
    if face is not None: x, y = grid(); d = np.sqrt((x - face[0]) ** 2 + (y - face[1]) ** 2); o *= np.clip((d - face[2]) / 6, 0, 1)[..., None]
    return o


# ------------------------------------------------------------------ ultra instinct: heat-haze and slow threads
def aura_ui(mask, f, power=1., seed=1, face=None, up=(0., -1.), hug=None, threads=46):
    """silver: slow veils, long thin threads, a broken thread of light along the outline, motes.  A.haze(img) bends the background first"""
    hug = hug or Hug(mask, up, .35, .03); A = Aura(back=None, front=None, light=None, hug=hug, mask=mask, kind='silver', power=power, f=f, seed=seed, face=face)
    if power <= .02 or not hug.any: return A
    U = hug.U; t = f / 24.; P_ = PAL['silver']; dist = cv2.resize(hug.d, (W, H), interpolation=cv2.INTER_LINEAR); sc = 150 * t; nz = nimg(H, W, U * .17, t * .35, seed, 2, 4, 1., 2.6, ox=-hug.up[0] * sc, oy=-hug.up[1] * sc)
    s_ = ((grid()[0] - hug.c[0]) * hug.up[0] + (grid()[1] - hug.c[1]) * hug.up[1]); lo = np.clip((s_ - hug.bot + U * .02) / (U * .22), 0, 1)
    soft = np.exp(-dist / (U * (.085 + .03 * nz))) * lo * .34 * power; haze = np.dstack([soft[..., None] * C('#8fb0ee'), soft])
    V = Field(2)                                              # broad slow blades of bent air
    for Q, L, b, sd, tear in _parcels(hug, 20, t * .36, seed + 2, U * .78 * power, .060, (1., 1.5), -.35, wob=.5, K=22, inset=-U * .015): V.rib(Q, lwidths(len(Q), L, b, LV[:2], .42, .8, 1.1), U * .012)
    veil = stack(haze, V.cel(('#7f9fe0', '#d8e6fb'), None, None, (.26, .22), (max(1., U * .0022), max(1., U * .0016)), ('#eaf1ff', '#ffffff'), seed, .006, line_a=.8))
    def thr(n, sd0, over):
        F = Field(3); rr = []; LL = []; inf = []
        for i in range(n):
            x = t * (.42 + .25 * h1(sd0, i, 1)) + h1(sd0, i, 2); c = math.floor(x); tau = x - c; sd = int(sd0 * 91 + i * 7 + c * 3) & 0xffffff
            u = (i + h1(sd, 3)) / n; p = hug.root(u); nn = np.array([hug.samp(hug.nx, p[None])[0], hug.samp(hug.ny, p[None])[0]])
            if nn @ hug.up < -.5: continue
            rr.append(p + nn * U * (-.02 if over else (.005 + .05 * h1(sd, 4)))); LL.append(U * (.75 + .6 * h1(sd, 5))); inf.append((tau, sd))
        if not rr: return None
        S = hug.streams(np.array(rr), np.array(LL), 22, .55, t * .5, .7)
        for P, (tau, sd) in zip(S, inf):
            pth = Path(P, 9.); a = pth.L * max(0., tau * 1.35 - .42); e = pth.L * min(1., tau * 1.5 + .04); ln = e - a
            if ln < 12: continue
            Q = pth.sub(a, e, lambda v: U * .022 * np.sin(v * (5 + 3 * h1(sd, 6)) + t * 2.1 + 6 * h1(sd, 7)) * v, 22); w = U * (.0022 + .0062 * h1(sd, 8) ** 2) * min(1., (1 - tau) * 2.5)
            F.rib(Q, lwidths(22, ln, w, ((1., 1.), (.62, .9), (.30, .7)), .42), 1.5)
        return F.cel(('#6f8fd6', '#cfdff8', '#ffffff'), None, None, (.85, 1, 1))
    back = stack(veil, thr(threads, seed + 5, False)); M = Field(2)
    for i in range(34):                                          # motes
        x = t * (.30 + .3 * h1(seed, i, 31)) + h1(seed, i, 32); c = math.floor(x); tau = x - c; sd = int(seed * 53 + i * 11 + c * 3) & 0xffffff; p = hug.c + hug.ax * hug.half * (h1(sd, 1) * 2 - 1) * 2.0 + hug.up * (hug.bot + hug.U * (h1(sd, 2) * 1.0 + .35 * tau)) + hug.ax * U * .02 * math.sin(tau * 5 + i); r = U * .0045 * (.4 + h1(sd, 4)) * math.sin(math.pi * tau); M.rib(np.array([p - hug.up * r * 1.8, p, p + hug.up * r * 1.8]), [np.array([.1, r * .8, .1]), np.array([-.4, r * .5, -.4])], 1.)
    back = stack(back, M.cel(('#8fb4ff', '#ffffff')))
    fr = thr(9, seed + 8, True)
    if fr is not None and face is not None: x, y = grid(); d = np.sqrt((x - face[0]) ** 2 + (y - face[1]) ** 2); fr = fr * np.clip((d - face[2]) / 14, 0, 1)[..., None]
    li = bloom(back, .55, (4, 16, 60), (.3, .22, .2), '#b8d0ff', .55 * power); A.back, A.front, A.light = back, fr, li
    amp = np.exp(-dist / (U * .16)) * U * .011 * power; hx = nimg(H, W, U * .05, t * 1.1, seed + 4, 2, 4, 1., 1.8, oy=sc * 2.2)
    A.haze = lambda img: displace(img, amp * hx, amp * .55 * np.roll(hx, 97, 1)); return A


# ------------------------------------------------------------------ light back onto the figure
def lit(layer, kind='gold', power=1., f=0, seed=1, face=None, w=None, core=.5, tint=.18, dirs=None, inner=True, line=1.6):
    """figure layer with the aura's light on its edges: a hard angular band that breaks over the planes, a thin white inner band, a warm grade.
    dirs=(dx,dy) light mostly from that side (unit vector pointing to the light); face=(x,y,r) gets a thin rim only"""
    if power <= .02: return layer
    P_ = pal(kind if kind != 'goldblue' else 'gold'); a = layer[..., 3]; q = 2; m = cv2.resize(a, (W // q, H // q), interpolation=cv2.INTER_AREA); b = (m > .5).astype(np.uint8)
    if b.sum() < 30: return layer
    pad = cv2.copyMakeBorder(b, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0); di = cv2.distanceTransform(pad, cv2.DIST_L2, 5)[2:-2, 2:-2] * q; g = cv2.GaussianBlur(di, (0, 0), 2.5); gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3); n = np.sqrt(gx * gx + gy * gy) + 1e-6; nx, ny = -gx / n, -gy / n
    ys = np.nonzero(b.any(1))[0]; U = (ys[-1] - ys[0] + 1) * q; w = w or max(2.5, U * .012); nz = nimg(H // q, W // q, U * .10 / q, 0., seed + 3, 2, 2); nz = np.round(nz * 2.5) / 2.5 * .7 + nz * .3
    fac = np.ones_like(nx) * .75 if dirs is None else np.clip(nx * dirs[0] + ny * dirs[1] + .15, 0, 1); fac = np.clip(fac + .30 * ny * (dirs is None), .25, 1.2)
    if face is not None: x, y = grid(H // q, W // q); fac = fac * (1 - .62 * np.clip(1.15 - np.sqrt((x * q - face[0]) ** 2 + (y * q - face[1]) ** 2) / face[2], 0, 1))
    wr = w * min(power, 1.5) ** .5 * fac * np.clip(.85 + .75 * nz, .25, 1.8); R1 = np.clip((wr - di) / q + .5, 0, 1); R2 = np.clip((wr * .42 - di) / q + .5, 0, 1) * core * 2
    up = lambda z: cv2.resize(z.astype(f32), (W, H), interpolation=cv2.INTER_LINEAR); R1, R2 = up(R1)[..., None] * a[..., None], np.clip(up(R2), 0, 1)[..., None] * a[..., None]; o = layer.copy(); c = o[..., :3]; keep = 1.
    if face is not None: x, y = grid(); keep = np.clip((np.sqrt((x - face[0]) ** 2 + (y - face[1]) ** 2) / face[2] - .80) * 3.5, 0, 1)[..., None]; R1 = R1 * (.45 + .55 * keep); R2 = R2 * keep        # a face keeps its drawing: its planes, eyes and lines are never washed
    if tint: lt = C(P_['lit']); c = c * (1 + (lt * 1.12 - 1) * tint * (.30 + .70 * keep))                                                            # multiplied grade: shadows stay shadows
    c = c * (1 - R1 * .86 * min(1., power + .2)) + C(P_['lit']) * R1 * .86 * min(1., power + .2)
    if inner: c = c * (1 - R2 * .9) + C(P_['lit2']) * R2 * .9
    if line:
        e = 1 - cv2.erode(a, np.ones((3, 3), np.uint8), iterations=max(1, int(round(line)))); e = np.clip(cv2.GaussianBlur(e, (0, 0), .7) * 1.15, 0, 1)[..., None] * a[..., None]; c = c * (1 - e * .92) + C(P_['ink']) * e * .92
    o[..., :3] = c; return o


def with_aura(img, layer, f, kind='gold', power=1., seed=1, face=None, shade=0., **k):
    """whole stack in one call: back flame, lit figure, front wisps, light"""
    A = aura(layer, f, kind, power, seed, face=face, **k)
    if A.back is None: return put(img, layer)
    if hasattr(A, 'haze'): img = A.haze(img)
    img = A.under(img); L = lit(layer, kind, power, f, seed, face); return A.over(put(img, L))


# ------------------------------------------------------------------ the furnace: a pillar of flame in the world (orbit it with a Camera3D)
def pillar3(cam, f, P0=(0, 0, 0), R0=1.5, height=11., kind='gold', seed=1, power=1., step=2, n=64, twist=.9, flare=1.9, pinch=.62, speed=6.5, hole=None, dens=1., tl=3.6, skirt=1.):
    """flame column round a figure at P0: parcels climb a flared, twisting tube, the base splays over the ground.
    -> (back layer, front layer, light).  hole=(x, y, rx, ry) keeps the figure readable in the front layer"""
    t = (f // step) * step / 24.; P_ = pal(kind); C2 = _AUR2.get(kind, P_['c']); P0 = np.asarray(P0, float); zc = cam.cam(P0 + [0, 0, 1.])[2]; pw = .4 + .6 * power
    def rad(z): q = z / height; return R0 * pw * (pinch + (flare - pinch) * np.exp(-q * 9.) + .55 * np.clip(q - .15, 0, 2) ** .8)
    def pts(th0, z, tw=1.): r = rad(z); a = th0 + twist * tw * z / R0 * .35; return np.stack([P0[0] + np.cos(a) * r, P0[1] + np.sin(a) * r, P0[2] + z], -1)
    B, F, V = Field(4), Field(4), Field(2); SB, SF = Field(1), Field(1)
    def ribbon3(Fd, X, wm, prof, k, kids=0, sd=0):
        px, z = cam.project(X)
        if z.min() < .25: return
        sc = cam.f / z; hw = [q * sc * wm for q in prof]; L = float(np.hypot(*np.diff(px, axis=0).T).sum())
        if L > 8 and px[:, 1].min() < H + 50 and px[:, 1].max() > -120:
            Fd.rib(px, [np.where(h > 0, h, -1.) for h in hw], k)
            if kids and hw[0].max() > 9:                      # the long edges split into tips
                T_ = np.gradient(px, axis=0); nk = len(px)
                for j in range(kids):
                    q = .14 + .66 * (j + h1(sd, j, 81)) / kids; i = min(nk - 2, int(q * (nk - 1))); sg = 1 if (j + sd) % 2 else -1; th = math.atan2(T_[i, 1], T_[i, 0]); w0 = max(hw[0][i], 2.); m_ = np.array([-math.sin(th), math.cos(th)]) * sg; l = L * (.14 + .20 * h1(sd, j, 82)) * (1.05 - .5 * q); p_ = px[i] + m_ * w0 * .62
                    if l > 14: Fd.rib(spine(p_[0], p_[1], l, th + sg * (.30 + .22 * h1(sd, j, 83)), 9, .06, .7, h1(sd, j, 84) * 6, -sg * (.42 + .3 * h1(sd, j, 85))), kwidths(9, l, w0 * (.30 + .22 * h1(sd, j, 86)), LV), k * .6)
    items = []
    for row, (m, sc, sp) in enumerate(((int(n * .55 * dens), 1., 1.), (int(n * dens), .55, 1.3))):
        for i in range(m):
            T = (1.15 + .7 * h1(seed, row, i, 1)) / sp; x = t / T + h1(seed, row, i, 2); c = math.floor(x); tau = x - c; sd = int(seed * 97 + row * 31 + i * 7 + c * 3) & 0xffffff; z_ = bms(h1(sd, 1), .25, .35) * sc; th = TAU * (i + .8 * h1(sd, 2)) / m; ln = tl * R0 * (.45 + 1.15 * z_) * pw; z0 = -.10 * R0 + (height * 1.05) * tau ** 1.15 * (.08 + .92 * h1(sd, 3) ** 1.6); items.append((th, z0, ln, z_, sd, tau))
    for th, z0, ln, z_, sd, tau in items:
        v = lin(16); v0 = float(np.clip(-z0 / ln, 0, .92)); v = v0 + (1 - v0) * v; z = z0 + ln * v; X = pts(th, z, .7 + .6 * h1(sd, 4)); v = (v - v0) / (1 - v0); sw = .16 * R0 * z_ * np.sin(v * 6 - t * 11 + 6 * h1(sd, 5)) * v; a = th + 1.57; lean = R0 * (.15 + .95 * z_) * v ** 1.6; X = X + np.stack([np.cos(th) * (sw + lean), np.sin(th) * (sw + lean), 0 * v], -1); e = math.sin(math.pi * min(1., tau * 1.02)) ** .5; w = R0 * (.09 + .24 * z_) * e * pw; front = cam.cam(X[6])[2] < zc; pr = [np.where(u > 0, u, -.02) for u in lwidths(16, 1., 1., LV, .30, .55, 1.15)]; ribbon3(F if front else B, X, w, pr, 3., 5, sd)
    for i in range(int(38 * dens * min(1., skirt) + .5)):                           # ground skirt
        x = t * (1.5 + h1(seed, i, 41)) + h1(seed, i, 42); c = math.floor(x); tau = x - c; sd = int(seed * 71 + i * 5 + c) & 0xffffff; th = TAU * (i + .7 * h1(sd, 1)) / 38; z_ = bms(h1(sd, 2)); g = 1 - (1 - min(1., tau / .5)) ** 2; v = lin(14); ln = R0 * (.5 + 1.0 * z_) * g * pw * skirt; rr = R0 * pw * flare * .58 + ln * (1 - (1 - v) ** 1.6); zz = R0 * (.5 + 1.1 * z_) * v ** 2.2 * g * skirt; X = P0 + np.stack([np.cos(th + .25 * v) * rr, np.sin(th + .25 * v) * rr, zz + .02], -1); end = 1 - max(0., tau - .55) / .45; ribbon3(F if cam.cam(X[4])[2] < zc else B, X, R0 * (.06 + .11 * z_) * end, lwidths(14, 1., 1., LV, .30, .8, 1.1), 2.)
    for i in range(int(70 * dens)):                           # slivers
        x = t * (1.2 + 1.2 * h1(seed, i, 51)) + h1(seed, i, 52); c = math.floor(x); tau = x - c; sd = int(seed * 59 + i * 5 + c) & 0xffffff; th = TAU * h1(sd, 1); ln = height * (.05 + .14 * h1(sd, 2)); z = np.maximum(.0, height * 1.1 * tau - ln * .3) + ln * lin(10); X = pts(th, z); X[:, :2] = P0[:2] + (X[:, :2] - P0[:2]) * (.55 + .5 * h1(sd, 3)); ribbon3(SF if cam.cam(X[4])[2] < zc else SB, X, R0 * (.012 + .03 * h1(sd, 4) ** 2), lwidths(10, 1., 1., LV[:1], .3), 1.)
    # solid mass behind the figure
    zz = np.linspace(0, height * 1.15, 30); ctr, zd = cam.project(P0 + np.stack([0 * zz, 0 * zz, zz], -1)); hw = rad(zz) * cam.f / zd * (1 + .20 * noise(zz / R0 * .9 - t * speed / R0, 3., t, seed)) * np.minimum(1., pinch / flare + .25 + zz / R0 * .6); ok = zd > .3; ctr = ctr + np.stack([.14 * hw * noise(zz / R0 * .6 - t * speed * .8 / R0, 9., t, seed), 0 * hw], -1); M = Field(4); M.rib(ctr[ok], [hw[ok] * q * (1 + .12 * noise(zz[ok] / R0 * 1.6 - t * speed * 1.3 / R0, 5. + j, t, seed)) for j, q in enumerate((.74, .62, .44, .22))], 10.); M.D = np.minimum(M.D, (ctr[0, 1] + hw[0] * .10 - grid()[1])[None]); inner = Field(1); inner2 = Field(1)
    for i in range(int(40 * dens)):
        x = t * (1.0 + .9 * h1(seed, i, 61)) + h1(seed, i, 62); c = math.floor(x); tau = x - c; ln = .10 + .22 * h1(seed, i, c, 63); q0 = tau * 1.2 - ln; qq = np.clip(q0 + ln * lin(12), 0, 1); o = (h1(seed, i, c, 64) * 2 - 1) * .62; Q = np.stack([np.interp(qq, zz / zz[-1], ctr[:, 0]) + o * np.interp(qq, zz / zz[-1], hw) * (.75 + .12 * np.sin(qq * 14 + i)), np.interp(qq, zz / zz[-1], ctr[:, 1])], -1); w_ = np.interp(qq, zz / zz[-1], hw) * (.015 + .07 * h1(seed, i, 65) ** 2) * np.sin(math.pi * lin(12)) ** .8; (inner if i % 2 else inner2).rib(Q, [w_ - .1], 2.); g = np.clip((ctr[0, 1] - grid()[1]) / max(1., ctr[0, 1] - ctr[-1, 1]), 0, 1) ** .9
    lw = lambda k: (k, 0, k * .55, 0); ink = (P_['ink'], None, P_['c'][0], None); bk = B.cel((P_['ink'], C2[0], C2[1], C2[2]), None, None, None, lw(1.8), (P_['deep'], None, P_['ink'], None), seed); ms = M.cel(P_['c'], C2, g, None, lw(2.0), ink, seed + 1); s1 = SB.cel((P_['c'][0],)); s1 *= M.mask(0)[..., None]; i1 = inner.cel((P_['c'][1],)); i1 *= M.mask(1)[..., None]; i2 = inner2.cel((P_['c'][3],)); i2 *= M.mask(0)[..., None]; inn = stack(bk, ms, i1, i2, s1)
    fr = stack(F.cel(P_['c'], C2, g, None, lw(2.4), ink, seed + 2), SF.cel((P_['c'][3],)))
    if hole is not None:
        x, y = grid(); d = np.sqrt(((x - hole[0]) / hole[2]) ** 2 + ((y - hole[1]) / hole[3]) ** 2); fr = fr * np.clip((d - 1 + .10 * nimg(H, W, 26, t * 3, seed, 1, 4)) * 14, 0, 1)[..., None]
    li = bloom(stack(inn, fr), .5, (8, 34, 120), (.20, .22, .34), P_['glow'], .7 * power); return inn, fr, li
