"""ki orb: a sphere charging (pearl -> orb) between cupped hands, what it drags in, and its light on what is near."""
from .ki_core import *
from .ki_beam import disc, petals, rays
from .ki_bolt import bolt, bolt_cel


def _cres(F, x, y, r, a0, span, w, spiral=0., n=20, asp=1., rot=0., lv=(1.,)):
    v = lin(n); a = a0 + span * v; rr = r * (1 - spiral * v); hw = w * np.sin(math.pi * v ** .8) ** .9; P = np.stack([np.cos(a) * rr, np.sin(a) * rr * asp], -1)
    if rot: c, s = math.cos(rot), math.sin(rot); P = np.stack([P[:, 0] * c - P[:, 1] * s, P[:, 0] * s + P[:, 1] * c], -1)
    F.rib(P + [x, y], [hw * q - (0 if q == 1 else .1) for q in lv], 0.); return F


def orb(x, y, r, f, kind='blue', seed=1, power=1., corona=1., bolts=.6, step=1, pull=0., sparkle=1.):
    """energy sphere of radius r px -> (layer, light): dark edge, rolling bands, white heart off-centre, corona of short licks, hard highlight.
    r < 12 draws a pearl with a four-point glint"""
    t = (f // step) * step / 24.; P_ = pal(kind); c = P_['c']; box = (x - r * 3.4 - 30, y - r * 3.4 - 30, x + r * 3.4 + 30, y + r * 3.4 + 30); L = []
    if sparkle: L.append(rays(x, y, r * .8, r * (1.9 + .25 * math.sin(t * 22)) + 8, 4, seed, .07, 1., math.pi / 4 * 0 + .08 * math.sin(t * 5), '#ffffff', None, 0.) * .95 if r < 26 else rays(x, y, r, r * 2.3, 14, seed + f // 2, .020, 1., 0, c[3], c[2]))
    F = Field(4, box)
    if corona and r > 7: petals(F, x, y, r * .86, r * (1.75 + .25 * power), int(16 + r * .05), seed, t, 1., 0., None, .9, .20, r * .10, power, 2.2)
    disc(F, x, y, r, seed, t, 1., 0., (1.08, .96, .70, .40), .035 + .02 * power, r * .10); L.append(F.cel(c, lw=(max(1., r * .030), 0, max(.8, r * .012), 0), ink=(P_['ink'], None, c[0], None), seed=seed))
    if r > 14:
        m = F.mask(1); A, B, D = Field(1, box), Field(1, box), Field(1, box)
        for i in range(7): _cres(A, x, y, r * (.90 - .07 * (i % 4)), t * (3.4 + .5 * i) * (1 if i % 2 else -1) + i * 1.1, 1.2 + .7 * h1(seed, i), r * (.030 + .05 * h1(seed, i, 2)), .25, asp=.96)
        for i in range(6): _cres(B, x, y, r * (.80 - .08 * (i % 3)), -t * (4.2 + .4 * i) + i * 1.3, 1.0 + .6 * h1(seed, i, 3), r * (.020 + .04 * h1(seed, i, 4)), .4)
        for q, col in ((A, c[0]), (B, c[3])): o = q.cel((col,)); o *= m[..., None]; L.append(o)
        _cres(D, x, y, r * .86, 3.55 + .05 * math.sin(t * 7), 1.15, r * .055, .02); _cres(D, x, y, r * .84, 5.0, .22, r * .04); L.append(D.cel(('#ffffff',)))
    if bolts and r > 16:
        G = Field(3, box)
        for i in range(int(5 * bolts + .99)):
            per = 7; a_ = (f + i * 3) % per; cc = (f + i * 3) // per; a0 = TAU * h1(seed, i, cc, 5); a1 = a0 + (h1(seed, i, cc, 6) - .5) * 1.6; p0 = (x + math.cos(a0) * r * .9, y + math.sin(a0) * r * .9); p1 = (x + math.cos(a1) * r * (1.5 + .8 * h1(seed, cc, i)), y + math.sin(a1) * r * (1.5 + .8 * h1(seed, cc, i))); bolt(G, p0, p1, max(1.4, r * .045), seed + i * 7 + cc, .3, 2, max(1., r * .02), a_, 5, 0., 3)
        L.append(bolt_cel(G, kind))
    lay = stack(*L); g = np.zeros((H // 8, W // 8), f32); cv2.circle(g, (int(x / 8), int(y / 8)), max(1, int(r / 8)), 1., -1, cv2.LINE_AA); gl = .55 * cv2.GaussianBlur(g, (0, 0), max(1., r * .12)) + .55 * cv2.GaussianBlur(g, (0, 0), max(2., r * .34)); li = cv2.resize(gl, (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * C(P_['glow']) * .75 * power; return lay, li * (1 - .85 * lay[..., 3:4])


def charge(x, y, r, f, kind='blue', seed=1, amount=1., reach=8., water=True, dust=True, step=1, asp=1.):
    """what a charging sphere drags in: long tapering lines of light, drops of water, specks, closing rings -> layer (put UNDER the orb)"""
    t = (f // step) * step / 24.; P_ = pal(kind); c = P_['c']; F = Field(3); Wt = Field(3); D = Field(1)
    def path(i, sd, ph, ln, sw):
        a = TAU * h1(sd, 1); r0 = r * reach * (.55 + .6 * h1(sd, 2)); u = np.clip(ph - ln * lin(14), 0, 1); rr = r * .7 + (r0 - r * .7) * (1 - u ** 2.1); aa = a + sw * (1 - rr / r0) * 1.1; return np.stack([x + np.cos(aa) * rr, y + np.sin(aa) * rr * asp], -1)
    for i in range(int(72 * amount)):
        sp = .9 + 1.0 * h1(seed, i, 1); q = t * sp + h1(seed, i, 2); ph = q - math.floor(q); sd = int(seed * 41 + i * 7 + math.floor(q) * 3) & 0xffffff; P = path(i, sd, ph ** 1.25, .16 + .16 * h1(sd, 3), .9 + .6 * h1(sd, 4)); w = r * .052 * (.3 + 1.3 * bms(h1(sd, 5))) * min(1., ph * 5, (1 - ph) * 8) + .3; e = lin(14); F.rib(P, [w * (1 - e) ** .7 + .5 * (e < .95), w * .6 * (1 - e) ** .8, w * .3 * (1 - e) - .1], 0.)
    if water:
        for i in range(int(44 * amount)):
            sp = .55 + .6 * h1(seed, i, 11); q = t * sp + h1(seed, i, 12); ph = q - math.floor(q); sd = int(seed * 43 + i * 11 + math.floor(q) * 5) & 0xffffff; P = path(i, sd, ph ** 1.7, .03 + .05 * ph, .5); b = r * (.04 + .13 * h1(sd, 5) ** 2) * min(1., ph * 6, (1 - ph) * 6); e = lin(14); hw = b * np.sin(math.pi * np.clip(.12 + e * .88, 0, 1)) ** .7; Wt.rib(P, [hw + .8, hw * .72, hw * .26 - .2], 0., [0, -b * .10, -b * .25])
    if dust:
        for i in range(int(46 * amount)):
            sp = .7 + .8 * h1(seed, i, 21); q = t * sp + h1(seed, i, 22); ph = q - math.floor(q); sd = int(seed * 47 + i * 13 + math.floor(q) * 5) & 0xffffff; P = path(i, sd, ph ** 1.5, .012, .7); w = max(.4, r * .012 * h1(sd, 5)); D.rib(P[:3], [np.array([w, w, .2])], 0.)
    R_ = Field(2)
    for i in range(3):                                        # closing rings
        q = t * 1.3 + i / 3.; ph = q - math.floor(q); rr = r * (1.2 + 3.6 * (1 - ph) ** 1.6); a = np.linspace(0, TAU, 80); wb = 1 + .04 * noise(np.cos(a) * 2, np.sin(a) * 2, math.floor(q), seed + i); hw = r * .05 * ph * np.clip(.5 + 1.0 * noise(np.cos(a) * 1.8, np.sin(a) * 1.8, math.floor(q) + 7., seed + i), 0, 1.4) - .25; R_.rib(np.stack([x + np.cos(a) * rr * wb, y + np.sin(a) * rr * wb * asp], -1), [hw, hw * .5 - .2], 0.)
    return stack(R_.cel((c[1], c[3])), D.cel((c[2],)), Wt.cel((c[0], c[2], '#ffffff')), F.cel((c[0], c[2], '#ffffff')))


def light_on(layer, x, y, r, kind='blue', reach=5., amt=1., steps=(.75, .42, .16), seed=1):
    """hard-stepped light of a sphere at (x, y) on a premult layer: three planes of tone that fall off, plus a rim on the sides facing it"""
    P_ = pal(kind); a = layer[..., 3]; q = 2; gx, gy = grid(H // q, W // q); dx, dy = x - gx * q, y - gy * q; d = np.sqrt(dx * dx + dy * dy) + 1e-3; m = cv2.GaussianBlur(cv2.resize(a, (W // q, H // q), interpolation=cv2.INTER_AREA), (0, 0), max(2., r * .10)); nx, ny = -cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3), -cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3); L = cv2.GaussianBlur(cv2.resize(layer[..., :3].mean(-1), (W // q, H // q), interpolation=cv2.INTER_AREA), (0, 0), 2.); n = nimg(H // q, W // q, r * .5 / q, 0, seed, 2, 2)
    fall = np.clip(1 - (d / (r * reach)) * (1 + .18 * n), 0, 1) ** 1.2 * (.75 + .5 * L); face = np.clip((nx * dx + ny * dy) / d * 2.6, 0, 1) * np.clip(1 - d / (r * reach * 1.3), 0, 1); o = layer.copy(); c = o[..., :3]
    for k, (th, col, al) in enumerate(zip(sorted(steps), (P_['c'][0], P_['c'][1], P_['c'][2]), (.22, .30, .42))):
        z = np.clip((fall - th) * 30 + .5, 0, 1); z = cv2.resize(z.astype(f32), (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * a[..., None]; c = c + (1 - c / np.maximum(a[..., None], 1e-3)) * z * C(col) * al * amt
    rm = np.clip((face - .22) * 14 + .5, 0, 1); rm = cv2.resize(rm.astype(f32), (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * a[..., None]; c = c * (1 - rm * .85 * min(1, amt)) + C(P_['lit2']) * rm * .85 * min(1, amt); o[..., :3] = c; return o
