"""Drawn water that breaks (b3).  Both functions work in the water's own coordinates: noise is stuck to the parcels, so sheets tear, hole and turn to drops as they fly, and nothing boils.
    col(cx, y_top, y_bot, hw, t, ...)      a standing / rising column: torn flanks, ragged head, skirt, inner flow
    bell(hx, hy, w, s, end, ...)           a column breaking on a head: sheets with broken edges, ligaments, drops (s = seconds of water time since the hit)
returns premultiplied RGBA (1080, 1920, 4)"""
import numpy as np, cv2, math
from db import W, H as FH
from db.paint import rgb, f32, fbm, smooth, grid

PAL = dict(deep='#2c7a6c', mid='#63b394', lit='#b4e0c4', pale='#dff5e2', white='#ffffff', ink='#154840')
_T = {}


def tex(seed, scales, weights):
    k = (seed, scales)
    if k not in _T:
        n = fbm(1024, 1024, scales, weights, seed); e = 160; m = np.clip((np.arange(1024) - (1024 - e)) / e, 0, 1).astype(f32)      # make it tile
        n = n * (1 - m[None]) + np.roll(n, 1024, 1) * 0 + np.roll(n, e, 1) * m[None] if False else n
        a = np.hstack([n, n[:, ::-1]]); _T[k] = np.vstack([a, a[::-1]]).astype(f32)                                                   # mirror tile: seamless
    return _T[k]


def smp(T, u, v): return cv2.remap(T, np.mod(u, 2047.).astype(f32), np.mod(v, 2047.).astype(f32), cv2.INTER_LINEAR)


def ink_layer(m, tone, foam_px=6., line_px=2.4, pal=PAL, white=None, foam_k=1.):
    """hard-edged cel from a soft field m (> 0 inside) and a tone field 0..1: body tones, white torn rim, pen edge"""
    b = (m > 0).astype(np.uint8); b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)); a = cv2.GaussianBlur(b.astype(f32), (0, 0), .8); a = smooth(.35, .65, a)
    dt = cv2.distanceTransform(b, cv2.DIST_L2, 5); c = np.where((tone > .62)[..., None], rgb(pal['pale']), np.where((tone > .36)[..., None], rgb(pal['lit']), np.where((tone > .14)[..., None], rgb(pal['mid']), rgb(pal['deep']))))
    c = cv2.GaussianBlur(c.astype(f32), (0, 0), .7)
    if white is not None: w_ = smooth(.0, .12, white)[..., None]; c = c * (1 - w_) + rgb(pal['white']) * w_
    f = (smooth(foam_px, foam_px * .72, dt) * foam_k)[..., None]; c = c * (1 - f) + rgb(pal['white']) * f
    l = smooth(line_px + .5, line_px - .6, dt)[..., None]; c = c * (1 - l) + rgb(pal['ink']) * l
    return np.dstack([c * a[..., None], a]).astype(f32)


def beads(lay, pts, pal=PAL):
    """round drops with a pen edge and a light"""
    from db.ink import Cel
    cel = Cel(W, FH); t = np.linspace(0, 2 * math.pi, 22)[:-1]
    for x, y, r, vx, vy in pts:
        if r < .8 or not (-40 < x < W + 40 and -40 < y < FH + 40): continue
        sp = math.hypot(vx, vy) + 1e-6; st = 1 + min(1.1, sp / 1500.); a = math.atan2(vy, vx); ca, sa = math.cos(a), math.sin(a); P = np.c_[np.cos(t) * r * st, np.sin(t) * r]; P = np.c_[P[:, 0] * ca - P[:, 1] * sa + x, P[:, 0] * sa + P[:, 1] * ca + y]
        Po = (P - [x, y]) * (1 + 1.5 / r) + [x, y]; cel.fill(Po, pal['ink']); cel.fill(P, pal['pale']); cel.fill((P - [x, y]) * .62 + [x - r * .16, y - r * .18], pal['white'])
    b = cel.layer(); return b + lay * (1 - b[..., 3:4])


def col(cx, y_top, y_bot, hw, t, seed=0, flow=1., head=1., skirt=1., ragged=.10, blur_=0., foot=False, lean=0., drops=1.):
    """hw = half width px; flow +1 water climbs, -1 falls; foot = the BOTTOM end is the free ragged end (a slug that arrives from above)"""
    x, y = grid(FH, W); T1, T2, T3 = tex(3, (60, 24, 10), (1, .55, .3)), tex(7, (26, 10), (1, .5)), tex(11, (90, 34, 13), (1, .6, .35)); v = (y_bot - y) / hw; a = (x - cx - lean * v * hw) / hw; adv = flow * t * 5.2
    side = np.where(a < 0, 0., 900.); e1 = smp(T1, side + 40, (v - adv) * 62. + 300); e2 = smp(T2, side + 210, (v - adv * 1.25) * 96. + 80)
    wd = 1 + ragged * (.75 * e1 + .40 * e2) + (.62 * np.exp(-np.maximum(v, 0) / .34) + .20 * np.exp(-np.maximum(v, 0) / 1.1)) * skirt * (not foot)
    vt = (y_bot - y_top) / hw; fg = smp(T2, a * 170 + 500, np.full_like(a, 37. + t * 150)) * .55 + smp(T1, a * 120 + 900, np.full_like(a, 400 + t * 90)) * .45; sky = v - vt
    m = wd - np.abs(a)
    if foot: ft = .30 * fg + .16 * smp(T2, a * 320 + 100, np.full_like(a, 200.)); m = np.minimum(m, (v - ft * head - .12 * (1 - a * a)) * 1.)
    else:
        crown = .34 * np.abs(fg) ** .8 * np.sign(fg) + .20 * (1 - np.clip(a * a, 0, 1)); m = np.minimum(m, (crown * head - sky) * .9)
        sp = smp(T2, a * 260 + 40, (v - adv * 1.6) * 190.); sp2 = smp(T1, a * 200 + 700, (v - adv * 1.3) * 110.); near = np.exp(-np.maximum(sky, 0) / .55) * (sky > -.1) * np.exp(-np.maximum(np.abs(a) - 1.05, 0) / .35); m = np.maximum(m, (sp * .5 + sp2 * .5 - (1.35 - .62 * near * head)) * .5 * (near > .12))
        # the flanks shed ligaments
        fl = np.exp(-np.maximum(np.abs(a) - 1, 0) / .16) * (v > .1) * (v < vt); m = np.maximum(m, (sp2 * .6 + sp * .4 - (1.65 - .62 * fl)) * .5 * (fl > .25))
    st1 = smp(T3, a * 210 + 60, (v - adv) * 30.); st2 = smp(T1, a * 380 + 700, (v - adv * 1.15) * 46.); tone = .62 - .55 * np.clip(a + .18 * st1, -1, 1.2) ** 1 + .10 * st2; tone = np.clip(tone - .10, 0, 1); wh = np.where(a < .35, st2 - 1.05 + .3 * (a < -.45), st2 - 1.55) * .5
    lay = ink_layer(m, tone, max(1.6, hw * .050), max(1.1, min(2.8, hw * .022)), white=wh)
    if drops > 0:
        r = np.random.default_rng(seed + 5); pts = []
        for j in range(int(34 * drops)):
            per = .55 + .5 * r.random(); ph = r.random(); k = ((t / per) + ph) % 1.; th = r.uniform(-1, 1); tt = k * per; at_head = (j % 3 != 0) and not foot
            if at_head: vx, vy = th * hw * 2.2, -hw * (1.6 + 1.6 * r.random()) * (flow > 0); x0, y0 = cx + th * hw * .9 + lean * vt * hw, y_top
            else: sg = 1 if j % 2 else -1; vx, vy = sg * hw * (.8 + 1.3 * r.random()), -hw * (1.4 + 1.5 * r.random()); x0, y0 = cx + sg * hw * 1.35, y_bot
            g = hw * 7.; pts.append((x0 + vx * tt, y0 + vy * tt + .5 * g * tt * tt, hw * (.025 + .10 * r.random() ** 2.4), vx * 2, (vy + g * tt) * 2))
        lay = beads(lay, [q for q in pts if q[1] < y_bot + hw * .2])
    if blur_ > 0: lay = np.stack([cv2.GaussianBlur(lay[..., k], (0, 0), blur_) for k in range(4)], -1)
    return lay


def bell(hx, hy, w, s, end, seed=0, V=1500., vy=-980., g=12500., drops=1.):
    """hx, hy = crown of the head; w = half width of the column; s = water seconds since the hit, end = when the column has run out.  Sheets leave the crown, arch over and tear"""
    x, y = grid(FH, W); T1, T2, T3 = tex(3, (60, 24, 10), (1, .55, .3)), tex(7, (26, 10), (1, .5)), tex(11, (90, 34, 13), (1, .6, .35))
    out = np.zeros((FH, W, 4), f32); tail = max(0., s - end) * 1.25
    for far in (True, False):                                       # the far half of the bell is darker, smaller, seen through the tears of the near one
        Vq, vq, gq, w0 = (V * .80, vy * 1.10, g * .88, w * .55) if far else (V, vy, g, w * .60); dy = y - hy - (0 if far else 10); D_ = vq * vq + 2 * gq * dy; ta = (-vq + np.sqrt(np.maximum(D_, 0))) / gq; tb = (-vq - np.sqrt(np.maximum(D_, 0))) / gq
        for br in ((tb, ta) if dy.min() < 0 else (ta,)):            # rising branch, then falling branch
            tau = br; ok = (D_ > 0) & (tau >= 0); u = (x - hx) / (w0 + Vq * np.maximum(tau, 0)); em = s - tau                       # em = when this parcel left the head
            sd = 600. if far else 0.
            fing = smp(T2, u * 210 + 300 + sd, np.full_like(u, 60.)) * .5 + smp(T1, u * 130 + 80 + sd, np.full_like(u, 310.)) * .5; f2 = smp(T2, u * 620 + 120 + sd, tau * 420.); front = s * (.84 + .20 * np.clip(fing, -1.2, 1.6) + .075 * f2); age = np.clip(tau / .55, 0, 1.4)
            n1 = smp(T1, u * 240 + 700 + sd, em * 520 + 90); n2 = smp(T2, u * 420 + 60 + sd, em * 1150 + 400); n3 = smp(T3, u * 150 + sd, em * 260 + 900); rib = smp(T2, u * 640 + 900 + sd, em * 150 + 33)
            hole = .62 * n1 + .42 * n2 + .22 * rib; thr = -2.2 + 2.35 * age ** .72 + .55 * np.clip((np.abs(u) - .62) / .38, 0, 1); m = (hole - thr) * .5; m = np.minimum(m, (front - tau) * 9.); m = np.minimum(m, (tau - tail * (.95 + .07 * fing)) * 12.) if tail <= 0 else np.minimum(m, ((tau - tail * (.95 + .06 * fing)) / .115 - .55 + .95 * n2 + .45 * n1) * .5); m = np.minimum(m, (1.0 + .06 * n1 - np.abs(u)) * 4.); m = np.where(ok, m, -1.)
            tone = .64 + .22 * n3 + .16 * rib - .26 * np.clip(u, -1, 1) - (.36 if far else 0.) + .10 * (1 - age); wh = rib * .8 + .35 * n2 - 1.12 - (.5 if far else 0.)
            if br is tb: tone = tone + .18
            l = ink_layer(m, np.clip(tone, 0, 1), 5.2 if not far else 3.2, 2.6 if not far else 2.0, white=wh); out = l + out * (1 - l[..., 3:4])
    r = np.random.default_rng(seed + 3); pts = []
    for j in range(int(210 * drops)):                               # drops: born where sheets tear, each on its own parabola
        u = r.uniform(-1.05, 1.05); e0 = r.uniform(0, end); k = r.uniform(.86, 1.22); tb_ = .10 + .26 * r.random(); tt = s - e0
        if tt < tb_ or tt > .95: continue
        vx = V * u * k; px, py = hx + u * w * .6 + vx * tt, hy + vy * k * tt + .5 * g * tt * tt; jx = (r.normal(0, 190), r.normal(0, 150)); px, py = px + jx[0] * (tt - tb_), py + jx[1] * (tt - tb_); pts.append((px, py, 2. + 11 * r.random() ** 2.6, vx, vy + g * tt))
    return beads(out, pts)


def star(hx, hy, k, seed=0, spikes=True):
    """the first two frames of the hit: a crown of torn spray thrown up and out (k 0..1)"""
    from db.ink import Cel, spl
    r = np.random.default_rng(seed + 8); cel = Cel(W, FH)
    for j in range(30):
        a = math.pi * (1.0 + (j + r.uniform(-.4, .4)) / 29.); l = r.uniform(90, 330) * (.60 + .55 * k) * (.65 + .35 * abs(math.sin(a * 3.7 + seed))); d = np.array([math.cos(a), math.sin(a) * .86]); n = np.array([-d[1], d[0]]); sg = 1 if d[0] > 0 else -1; b = np.array([hx, hy]) + d * np.array([120, 46]); w0 = r.uniform(12, 30); bow = np.array([sg * .10, .16]) * l
        q = [b + n * w0, b + d * l * .30 + n * w0 * .75 + n * r.normal(0, 6), b + d * l * .55 + n * w0 * r.uniform(.15, .5) + bow * .3, b + d * l * .72 + n * w0 * r.uniform(.2, .5) + bow * .5, b + d * l + bow, b + d * l * .70 - n * w0 * .3 + bow * .5, b + d * l * .42 - n * w0 * r.uniform(.3, .9), b - n * w0]; P = spl(np.array(q), True, corners=(0, 4, 7), n=6)
        if spikes: cel.fill(P + [2, 3], PAL['ink'], .95); cel.fill(P, PAL['white'] if j % 3 else PAL['pale']); cel.fill((P - b) * [.55, .55] + b + d * 8, PAL['lit'] if j % 2 else PAL['pale'], .8)
        for i in range(3): e = b + d * l * r.uniform(.95, 1.55) + bow * r.uniform(.8, 1.6) + r.normal(0, 14, 2); rd = r.uniform(2, 10); t = np.linspace(0, 2 * math.pi, 18)[:-1]; cel.fill(np.c_[e[0] + np.cos(t) * (rd + 1.5), e[1] + np.sin(t) * (rd + 1.5)], PAL['ink']); cel.fill(np.c_[e[0] + np.cos(t) * rd, e[1] + np.sin(t) * rd], PAL['white'])
    return cel.layer()


if __name__ == '__main__':
    from db import comp
    bg = np.ones((FH, W, 3), f32) * rgb('#5a8f78'); ims = []
    for s in (.064, .18, .30, .41, .53, .65, .76): ims.append(comp.over(bg, bell(1000, 160, 232, s, .50)))
    c = comp.over(bg, col(500, 160, 980, 190, .31)); c = comp.over(c, col(1300, 620, 900, 46, .2)); c = comp.over(c, col(1600, 300, 900, 46, .35)); ims.append(c)
    ims.append(comp.over(bg, col(960, -300, 330, 230, .4, flow=-1., foot=True)))
    ims = [cv2.resize(i, (640, 360), interpolation=cv2.INTER_AREA) for i in ims]; comp.save('qa/friends/_water.jpg', np.vstack([np.hstack(ims[k:k + 3]) for k in (0, 3, 6)]))
