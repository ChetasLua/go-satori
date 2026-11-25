"""g6: hands go up.  Four pictures at first dawn colour, one per beat.
    frame(k, u=1., f=0)      k 0 Gohan, 1 Piccolo, 2 Bulma Krillin Roshi on the beach, 3 the whole town;  u 0..1 how far hands and lights have risen -> float RGB
"""
from chars.fkit import *
from chars.gohan import Gohan
from chars import gohan as G
from chars.piccolo import Piccolo
from chars.friends import Bulma, Krillin, Roshi
from db.paint import ramp, grid, fbm
from db.comp import over, screen, blur, radial
from db import hands as HD

_c = {}


def dawn_bg(seed=1, hz=800, sun=1500, sea=True, rise=.6, moon=None):
    """last of the night above, the first dawn colour low on the right; hard-edged clouds lit from below"""
    k = (seed, hz, sun, sea, rise, moon)
    if k in _c: return _c[k]
    x, y = grid(FH, W); t = np.clip(y / hz, 0, 1); side = np.clip((x - sun) / 1500., -1, 1); q_ = np.clip(t + .10 * (1 - np.abs(side)) - .06, 0, 1); n_ = ramp(q_, [(0, '#0b0e26'), (.35, '#1a1d46'), (.62, '#3a3568'), (.82, '#6a4c7e'), (.93, '#a86a80'), (1, '#d8968a')]); d_ = ramp(q_, [(0, '#232a5e'), (.28, '#454a8a'), (.52, '#8a6aa4'), (.72, '#d58e9a'), (.87, '#ffb694'), (1, '#ffe6b8')]); img = n_ * (1 - rise) + d_ * rise
    img = img + (np.exp(-((x - sun) / (380. + 260 * rise)) ** 2) * np.exp(-((y - hz) / (130. + 110 * rise)) ** 2))[..., None] * rgb('#ffcf90') * (.18 + .42 * rise)
    if rise > .75: ds = np.hypot(x - sun, y - hz + 6); img = img + (smooth(126, 121, ds) * (y < hz))[..., None] * rgb('#fff6da') * .9 + (np.exp(-np.clip(ds - 121, 0, 999) / 70.) * .55)[..., None] * rgb('#ffd79a')
    r = np.random.default_rng(seed); vio = 0. * x; dm = 0. * x + 9e3
    if moon is not None:                                            # the falling moon: its lower limb hangs over everything, its violet drowns the top of the sky
        from db import moon as MN
        mx, my, mr = moon; dm = np.hypot(x - mx, y - my) - mr; g_ = np.where(dm > 0, MN.halo(dm, 380.), 0.); vio = np.exp(-np.maximum(dm, 0) / 330.).astype(f32); img = img * (1 - .42 * vio[..., None] * (dm > 0)[..., None]) + rgb('#2a1066') * (.30 * vio * (dm > 0))[..., None]; img = img + MN.halo_rgb(g_ * (.62 - .16 * rise)); b_ = MN.ball(mx, my, mr, spin=.12 * seed, seed=4, tilt=.45, grain=1.2); air = .10 + .10 * rise; b_[..., :3] = b_[..., :3] * (1 - air) + rgb('#a878e0') * air * b_[..., 3:4]; low = smooth(my + mr - 330, my + mr, y)[..., None]; b_[..., :3] = b_[..., :3] * (1 - .16 * low * rgb('#6a8a6a'))
    cel = Cel(W, FH, ss=1)
    for j in range(150):
        yy = r.random() ** 1.8 * hz * .62; cel.fill(ell(r.uniform(0, W), yy, r.uniform(.6, 1.7)), '#e8ecff', r.uniform(.25, .9) * (1 - yy / hz))
    img = over(img, cel.rgba())
    if moon is not None: img = over(img, b_)
    n = fbm(FH, W, (300, 120, 46, 19), seed=seed + 3, aspect=3.2) if False else cv2.resize(fbm(FH, W, (150, 60, 24, 10), seed=seed + 3), (W, FH))[:, :]; n = cv2.resize(n[:, :W // 3], (W, FH), interpolation=cv2.INTER_CUBIC); d = n + 1.1 * np.exp(-((t - .42) / .22) ** 2) - .35 + .4 * np.exp(-((t - .80) / .10) ** 2) - 1.4 * np.exp(-((t - .02) / .10) ** 2)
    m = smooth(.42, .56, d); dd = np.roll(d, -26, 0); lit = smooth(.10, .55, d - dd) * m; core = smooth(.95, 1.3, d)
    c = ramp(t, [(0, '#1d2148'), (.4, '#35356b'), (.7, '#67548a'), (.9, '#b07a96'), (1, '#e0a498')]); c = c * (1 - .55 * rise) + ramp(t, [(0, '#5a5aa0'), (.4, '#a878a8'), (.7, '#e89a9c'), (1, '#ffd0a8')]) * .55 * rise; c2 = ramp(t, [(0, '#4a4a86'), (.4, '#8a6a9e'), (.65, '#e0928e'), (.9, '#ffc79a'), (1, '#ffe0b8')])
    if moon is not None:                                            # clouds before the disc are black-violet; their tops and bellies near it burn violet
        vc = np.maximum(vio, (dm < 0) * 1.)[..., None]; c = c * (1 - .72 * vc) + rgb('#1c0f46') * .72 * vc; c2 = c2 * (1 - .85 * vc) + rgb('#b48cff') * .85 * vc; m = np.where(dm < 0, smooth(.52, .60, d) * .96, m); top = smooth(.10, .50, d - np.roll(d, 24, 0)) * m * (np.maximum(vio, (dm < 0) * 1.)) ** .8
    img = img * (1 - m[..., None] * .92) + c * m[..., None] * .92; img = img * (1 - core[..., None] * .35) + ramp(t, [(0, '#12153a'), (1, '#4a3f74')]) * core[..., None] * .35; img = img * (1 - lit[..., None] * .85) + c2 * lit[..., None] * .85
    if moon is not None: img = img * (1 - top[..., None] * .80) + rgb('#e4ccff') * top[..., None] * .80
    if sea:
        s = (y > hz); v = np.clip((y - hz) / (FH - hz), 0, 1); w_ = ramp(v, [(0, '#f0a99a'), (.06, '#b98aa8'), (.25, '#5c5c94'), (.6, '#333a72'), (1, '#1d2250')]); gl = cv2.resize(r.standard_normal((200, 90)).astype(f32), (W, FH), interpolation=cv2.INTER_CUBIC); g2 = smooth(.55, 1.3, gl) * (np.exp(-((x - sun) / (240 + 500 * v)) ** 2) * .95 + .10); w_ = w_ + g2[..., None] * rgb('#ffcf9c') * .8; img = np.where(s[..., None], w_, img); img = img + (np.exp(-np.abs(y - hz) / 3.) * .30)[..., None] * rgb('#ffe2bc')
    _c[k] = img.astype(f32); return _c[k]


def mote(img, x, y, r, f=0, seed=0, g=1., col='#ffe9b0'):
    """one small light: a drawn four-point spark, a core, a soft halo"""
    img = screen(img, np.clip(radial(x, y, r * 9, power=2.6) * .55 * g, 0, 1)[..., None] * rgb(col))
    cel = Cel(W, FH); k = r * (1 + .12 * math.sin(f * .9 + seed)); a0 = .12 * math.sin(f * .35 + seed)
    for j in range(4): a = a0 + j * TAU / 4; l = k * (3.4 if j % 2 else 2.4); n = A([-math.sin(a), math.cos(a)]) * k * .26; c = A([math.cos(a), math.sin(a)]); cel.fill(A([n, c * l, -n]) + [x, y], '#fff8e2', .95)
    cel.fill(ell(x, y, k * 1.05), col); cel.fill(ell(x, y, k * .78), '#ffffff'); return over(img, cel.rgba())


def motes(img, n, box, f=0, seed=1, r=(1.5, 4.5), u=1.):
    """a field of rising lights (cheap): box = x0, y0, x1, y1"""
    rr = np.random.default_rng(seed); cel = Cel(W, FH); h = np.zeros((FH // 4, W // 4), f32); x0, y0, x1, y1 = box
    for j in range(n):
        q = rr.random(5); y = y1 - ((q[1] * (y1 - y0) + f * (1.5 + 3 * q[2])) % (y1 - y0)); x = x0 + q[0] * (x1 - x0) + math.sin(f * .12 + j) * 8; k = (r[0] + (r[1] - r[0]) * q[3] ** 2) * u
        if k < .6: continue
        cel.line(A([(x, y + k * 7), (x, y + k * 3), (x, y)]), k * .9, '#ffe9b8', taper=(.9, .1), press=0, alpha=.5); cel.fill(ell(x, y, k, n=16), '#fff9e6'); cv2.circle(h, (int(x / 4), int(y / 4)), max(1, int(k / 2)), 1., -1)
    g = cv2.resize(cv2.GaussianBlur(h, (0, 0), 3) + .6 * cv2.GaussianBlur(h, (0, 0), 8), (W, FH)); img = screen(img, np.clip(g * 1.3, 0, 1)[..., None] * rgb('#ffdca0')); return over(img, cel.rgba())


def hand_light(lay, x, y, r=420, amt=.55, col='#ffe2a6'):
    """the small light on the figure under it"""
    X_, Y_ = grid(FH, W); d = np.sqrt((X_ - x) ** 2 + (Y_ - y) ** 2); k = np.exp(-(d / r) ** 1.5) * amt; out = lay.copy(); out[..., :3] += (lay[..., 3] * k)[..., None] * rgb(col) * (.12 + lay[..., :3]); return out


# ------------------------------------------------------------------ 3: the town
SHIRTS = ['#b4506a', '#4f7fb0', '#c99a4a', '#6d9a6a', '#8a6ab0', '#d8d0c4', '#a85a3c', '#3f6a8a', '#c07a94', '#5a5a8a']
HAIRS = ['#2c2434', '#4a3030', '#6a4a30', '#8a7a8a', '#3a5a6a', '#7a3a3a']


def person(cel, x, y, h, seed, depth, rimc, arm_=1., kind=None, w=2., light_=None):
    """one of the crowd, from behind or three-quarter behind, looking up: feet at (x, y), height h px.  depth 0 near .. 1 far (air)"""
    r = np.random.default_rng(seed); k = h; ln_ = r.normal(0, .035); fc = r.choice([-1, 1]); P = lambda L_: A([(x + (px + ln_ * -py) * k, y + py * k) for px, py in L_], float); kind = kind or r.choice(['man', 'woman', 'child', 'old', 'man', 'woman', 'girl', 'bear', 'cat', 'farmer']); hd = {'child': .115, 'girl': .115}.get(kind, .082); wd = {'man': .145, 'woman': .115, 'child': .125, 'girl': .12, 'old': .125, 'bear': .18, 'cat': .12, 'farmer': .15}[kind]; hy = -.895 if hd < .1 else -.86; nk = hy + hd * .9
    air = rgb('#8a6f9e') * depth; dusk = lambda c, m: np.clip(c_(c) * A([.46, .40, .66], f32) * m * (1 - depth) + air * .92, 0, 1); lit = lambda c: np.clip(c_(c) * A([1.0, .78, .70], f32) * (.92 - .30 * depth) + air * .45, 0, 1)
    sh_, hr, sk = str(r.choice(SHIRTS)), str(r.choice(HAIRS)), ('#f2c4a0' if kind not in ('bear', 'cat') else '#8a5a3c'); tr = str(r.choice(['#3a4468', '#5a4a4a', '#6a6a74', '#4a3a5a'])); line = np.clip(rgb('#120f26') * (1 - depth) + air * .75, 0, 1); out = []
    def blob(Q, col, side=.40, ww=w, gaps=None):
        Q = np.asarray(Q, float); cel.fill(Q, dusk(col, 1.)); x1, x0 = Q[:, 0].max(), Q[:, 0].min(); cel.clip(Q); yy = np.linspace(Q[:, 1].min() - 4, Q[:, 1].max() + 4, 7); e = x1 - (x1 - x0) * side * (.75 + .45 * np.abs(np.sin(np.arange(7) * 2.4 + seed)))
        cel.fill(np.vstack([np.c_[e, yy], [(x1 + 9, yy[-1]), (x1 + 9, yy[0])]]), lit(col)); cel.unclip(); outline(cel, Q, ww, line, seed, tremor=.1)
    # legs or skirt
    if kind in ('woman', 'girl', 'old'): blob(spl(P([(-wd * .9, -.46), (wd * .9, -.46), (wd * 1.35, -.03), (wd * .4, .0), (-wd * .5, .0), (-wd * 1.35, -.03)]), True, corners=(0, 1, 2, 5), n=5), tr if r.random() < .5 else sh_)
    else: blob(spl(P([(-wd * .92, -.46), (wd * .92, -.46), (wd * .98, -.20), (wd * .80, 0), (wd * .18, 0), (wd * .04, -.30), (-wd * .14, 0), (-wd * .82, 0), (-wd * .98, -.20)]), True, corners=(0, 1, 3, 4, 5, 6, 7), n=4), tr)
    arms = (-1, 1) if r.random() < .45 else (r.choice([-1, 1]),); far = -fc
    def arm2(q):
        up = .10 + .34 * arm_ * r.uniform(.88, 1.06); a_, e_, h_ = (q * wd * .80, nk + .105), (q * (wd + .070), nk + .105 - up * .38), (q * (wd + .020 + .03 * r.random()), nk + .105 - up); c = P([h_])[0]; d_ = c - P([e_])[0]; d_ /= np.hypot(*d_); a0 = math.atan2(d_[1], d_[0]); n_ = A([-d_[1], d_[0]])
        base = c + d_ * .030 * k
        for j in range(4): an = a0 + (j - 1.5) * .20; f0 = base + n_ * (j - 1.5) * .0135 * k; F_ = capsule(A([f0 - d_ * .008 * k, f0 + A([math.cos(an), math.sin(an)]) * (.062 - .007 * abs(j - 1.4)) * k]), [.0085 * k, .0068 * k], 5); cel.fill(F_, dusk(sk, 1.1)); cel.line(np.vstack([F_, F_[:1]]), max(.6, w * .42), line, taper=(.05, .05), press=.1)
        an = a0 + (-q) * 1.0; F_ = capsule(A([c, c + A([math.cos(an), math.sin(an)]) * .052 * k]), [.011 * k, .0075 * k], 5); cel.fill(F_, dusk(sk, 1.1)); cel.line(np.vstack([F_, F_[:1]]), max(.6, w * .42), line, taper=(.05, .05), press=.1)
        blob(capsule(P([e_, ((e_[0] + h_[0]) / 2 + q * .012, (e_[1] + h_[1]) / 2), h_]), [.033 * k, .030 * k, .019 * k]), sk, .42, w * .8); pm = spl(A([c - n_ * .017 * k - d_ * .006 * k, c - n_ * .027 * k + d_ * .020 * k, base - n_ * .026 * k + d_ * .012 * k, base + d_ * .018 * k, base + n_ * .026 * k + d_ * .012 * k, c + n_ * .027 * k + d_ * .020 * k, c + n_ * .017 * k - d_ * .006 * k]), True, n=4); cel.fill(pm, dusk(sk, 1.1)); cel.clip(pm); cel.fill(pm + [.026 * k, 0], lit(sk)); cel.unclip(); cel.line(pm, max(.6, w * .6), line, taper=(.1, .1), press=.2)
        S_ = capsule(P([a_, ((a_[0] + e_[0]) / 2 + q * .006, (a_[1] + e_[1]) / 2 + .012), e_]), [.060 * k, .056 * k, .046 * k]); blob(S_, sh_, .45, w * .9); return c - d_ * 0 - [0, .10 * k]
    for q in arms:
        if q == far: out.append(arm2(q))
    if len(arms) == 1 and arms[0] == far: pass
    for q in (-1, 1):
        if q not in arms: blob(capsule(P([(q * wd * .84, nk + .11), (q * (wd + .035), nk + .24), (q * (wd + .02), nk + .40)]), [.056 * k, .046 * k, .030 * k]), sh_, .42, w * .8)
    cel.fill(P([(-.034, hy), (.034, hy), (.042, nk + .05), (-.042, nk + .05)]), dusk(sk, .9))
    blob(spl(P([(-.045, nk + .02), (.045, nk + .02), (wd * .80, nk + .055), (wd * 1.02, nk + .13), (wd * .92, -.60), (wd * .95, -.44), (-wd * .95, -.44), (-wd * .92, -.60), (-wd * 1.02, nk + .13), (-wd * .80, nk + .055)]), True, corners=(0, 1, 5, 6), n=5), sh_)
    for x0, y0, x1, y1 in ((-.3, .18, -.1, .30), (.35, .20, .15, .34), (-.6, .30, -.35, .40)): cel.line(spl(P([(x0 * wd, nk + y0), ((x0 + x1) / 2 * wd, nk + (y0 + y1) / 2 + .01), (x1 * wd, nk + y1)]), n=5), max(.6, w * .55), line, taper=(.1, .6))
    for q in arms:
        if q != far: out.append(arm2(q))
    # head turned up, three-quarter from behind: a slice of cheek and ear shows on one side
    c = P([(0, hy)])[0]; R_ = hd * k; H_ = lambda L_, cn=(): spl(A([(fc * px, py) for px, py in L_]) * R_ + c, True, corners=cn, n=5); blob(H_([(-.80, .1), (-.72, -.62), (0, -1.02), (.70, -.70), (.90, -.12), (.78, .22), (.86, .40), (.60, .86), (.10, 1.0), (-.50, .80)]), sk, .40, w)
    s = int(r.integers(5))
    if kind in ('bear', 'cat'):
        for q in (-1, 1): blob(ell(*(c + A([q * .66, -.82]) * R_), R_ * .34, n=18) if kind == 'bear' else A([c + A([q * .86, -.30]) * R_, c + A([q * .80, -1.45]) * R_, c + A([q * .12, -.86]) * R_]), sk, .45, w * .8)
    elif kind == 'farmer': blob(H_([(-1.75, -.30), (-.85, -.62), (-.62, -1.25), (.50, -1.30), (.82, -.70), (1.75, -.46), (.4, -.12), (-.6, -.08)], (0, 5)), '#d8b060', .42, w)
    else:
        base = [(-.92, .30), (-.94, -.55), (-.30, -1.14), (.45, -1.08), (.84, -.66), (.55, -.40), (.34, .05), (.40, .48), (.10, .86), (-.52, .84)]
        if s == 0: base[8:10] = [(.15, 1.65), (-.9, 1.6)]
        if s == 1: base = [(-.90, .55), (-1.25, .10), (-1.0, -.25), (-1.35, -.75), (-.80, -.80), (-.62, -1.50), (-.10, -1.08), (.35, -1.48), (.50, -.95), (1.0, -.95), (.72, -.45), (.40, .0), (.42, .5), (.1, .95), (-.4, .75)]
        blob(H_(base, range(len(base)) if s == 1 else (4, 7)), hr if kind != 'old' else '#c8c0cc', .34, w)
        if s == 2 or kind == 'old': blob(ell(*(c + A([-fc * .35, -1.10]) * R_), R_ * .42, n=18), hr if kind != 'old' else '#c8c0cc', .4, w * .8)
        if s == 3: blob(H_([(-.6, -.6), (-1.35, -.65), (-1.75, .3), (-1.45, 1.6), (-1.15, .6), (-.75, .1)], (3,)), hr, .35, w * .8); cel.fill(ell(*(c + A([-fc * .82, -.48]) * R_), R_ * .20), lit('#d8404a'))
    return out


def town(u=1., f=0):
    img = dawn_bg(9, 720, 1380, False).copy(); x, y = grid(FH, W); cel = Cel(W, FH); r = np.random.default_rng(4)
    # far mountains and the sea of roofs: capsule domes, chimneys, a tower
    cel.fill(spl(A([(-20, 740), (120, 640), (260, 700), (420, 580), (560, 660), (760, 560), (900, 650), (1100, 600), (1300, 690), (1500, 630), (1700, 700), (1940, 650), (1940, 1100), (-20, 1100)]), True, corners=(0, 2, 4, 6, 8, 10, 11, 12, 13), n=6), '#6b5a94')
    cel.line(spl(A([(420, 580), (560, 660), (760, 560), (900, 650), (1100, 600), (1300, 690)]), corners=(1, 2, 3, 4), n=6), 2.2, '#f0b4a4', taper=(.1, .1), alpha=.7)
    def dome(cx, by, R_, col, lit, win='#ffd98a'):
        t = np.linspace(math.pi, TAU, 30); D_ = np.vstack([np.c_[cx + np.cos(t) * R_, by + np.sin(t) * R_ * .92], [(cx + R_, by + R_ * .5), (cx - R_, by + R_ * .5)]]); cel.fill(D_, col); cel.clip(D_); cel.fill(D_ + [R_ * .62, R_ * .04] if False else ell(cx + R_ * .95, by - R_ * .1, R_ * .62, R_ * 1.1), lit, .8); cel.fill(A([(cx - R_, by + R_ * .06), (cx + R_, by + R_ * .06), (cx + R_, by + R_ * .12), (cx - R_, by + R_ * .12)]), mixc(col, '#000000', .25), .6); cel.unclip()
        for q in (-.45, .0): cel.fill(ell(cx + q * R_, by - R_ * .28, R_ * .13, n=20), win); cel.line(ell(cx + q * R_, by - R_ * .28, R_ * .13, n=24), 1.4, '#2a2450', taper=(0, 0))
        cel.line(np.vstack([np.c_[cx + np.cos(t) * R_, by + np.sin(t) * R_ * .92]]), 2.2, '#241f48', taper=(.1, .1), seed=int(cx))
    for cx, by, R_, a in ((140, 800, 120, 0), (400, 770, 90, 1), (590, 800, 130, 2), (860, 772, 84, 0), (1080, 800, 120, 1), (1330, 770, 96, 2), (1560, 800, 140, 0), (1820, 780, 120, 1)): dome(cx, by, R_, ('#4f4684', '#5a4a80', '#453f7c')[a], ('#d99a9a', '#e8a890', '#c98ca0')[a])
    T_ = A([(1180, 800), (1192, 470), (1180, 430), (1200, 380), (1220, 430), (1208, 470), (1220, 800)]); cel.fill(T_, '#4a4280'); cel.line(A([(1218, 440), (1208, 470), (1220, 790)]), 3, '#f0b49c', taper=(.1, .3)); cel.fill(ell(1200, 372, 7), '#fff0c0')
    # a long-necked dinosaur looks over the roofs
    N_ = spl(A([(250, 860), (262, 700), (238, 560), (250, 452), (300, 404), (372, 402), (412, 430), (398, 452), (340, 452), (308, 478), (304, 570), (332, 700), (360, 860)]), True, n=8); cel.fill(N_, '#3f3a74'); cel.line(N_[int(len(N_) * .44):int(len(N_) * .92)], 4, '#e8a498', taper=(.2, .2), seed=2); cel.fill(ell(352, 424, 5), '#fff2c8')
    for x0 in (250, 246, 252, 260): pass
    for j, (px, py) in enumerate(((1500, 250), (1580, 220), (1650, 262), (700, 300), (640, 280))): cel.line(spl(A([(px - 26, py - 8 - 4 * math.sin(f * .6 + j)), (px - 10, py - 12), (px, py), (px + 10, py - 12), (px + 26, py - 8 - 4 * math.sin(f * .6 + j))]), corners=(2,), n=5), 3, '#2a2656', taper=(.6, .6), press=0)
    img = over(img, cel.rgba()); haze = np.exp(-((y - 800) / 120.) ** 2); img = img * (1 - .32 * haze[..., None]) + rgb('#f2b09a') * .32 * haze[..., None]
    lights = []
    for row, (by, h0, n, dp, rc, ww) in enumerate(((905, 125, 40, .62, '#f0b6a4', .9), (968, 185, 26, .40, '#f6b89c', 1.2), (1060, 272, 16, .18, '#ffc59c', 1.7))):
        cel = Cel(W, FH); xs = (np.arange(n) + r.uniform(-.38, .38, n)) / (n - 1) * (W + 160) - 80
        for j, x0 in enumerate(xs): lights += [(q, h0) for q in person(cel, x0, by + r.normal(0, h0 * .05), h0 * r.uniform(.78, 1.12), int(r.integers(1 << 30)), dp, rc, u, w=ww)]
        lay = cel.rgba(); img = over(img, lay)
        if row < 2: k = np.exp(-((y - by) / (h0 * .6)) ** 2) * (.20 - .06 * row); img = img * (1 - k[..., None]) + rgb('#c98aa0') * k[..., None]
    # foreground: big backs, a child on her father's shoulders, a dog
    cel = Cel(W, FH)
    for x0, h, sd, kd in ((190, 620, 5, 'farmer'), (1030, 560, 21, 'woman'), (1740, 660, 8, 'bear'), (1370, 480, 33, 'old'), (640, 560, 12, 'man')): lights += [(q, h) for q in person(cel, x0, 1330, h, sd, 0., '#ffb98c', u, kd, w=3.6)]
    lights += [(q, 300) for q in person(cel, 648, 1010, 300, 77, 0., '#ffc298', u, 'girl', w=3.2)]
    D_ = spl(A([(1420, 1100), (1432, 1020), (1424, 968), (1436, 930), (1452, 952), (1470, 950), (1488, 926), (1500, 960), (1520, 984), (1512, 1004), (1490, 1010), (1500, 1060), (1512, 1100)]), True, corners=(3, 6, 8), n=7); cel.fill(D_, '#4a3444'); cel.line(D_[int(len(D_) * .48):int(len(D_) * .95)], 3.6, '#ffb98c', taper=(.3, .3)); cel.line(spl(A([(1426, 1060), (1390, 1030), (1384, 990)]), n=6), 14, '#4a3444', taper=(0, .5), press=0)
    img = over(img, cel.rgba())
    if u > .15:
        for q, h in lights:
            pass
        cel = Cel(W, FH); hm = np.zeros((FH // 4, W // 4), f32)
        for j, (q, h) in enumerate(lights):
            k = max(1.6, h * .018) * min(1, u * 1.2); yy = q[1] - (u - .15) * h * .20 - (f * .6 * (1 + (j % 5) * .2)) % (h * .25); cel.fill(ell(q[0], yy, k, n=16), '#fffaf0'); 
            for a in range(4): b = a * TAU / 4; c = A([math.cos(b), math.sin(b)]); n_ = A([-c[1], c[0]]) * k * .22; cel.fill(A([n_, c * k * (3. if a % 2 else 2.), -n_]) + [q[0], yy], '#fff2cc', .9)
            cv2.circle(hm, (int(q[0] / 4), int(yy / 4)), max(1, int(k / 2.2)), 1., -1)
        g = cv2.resize(cv2.GaussianBlur(hm, (0, 0), 3.5) + .8 * cv2.GaussianBlur(hm, (0, 0), 10), (W, FH)); img = screen(img, np.clip(g * 1.5, 0, 1)[..., None] * rgb('#ffd796')); img = over(img, cel.rgba())
        img = motes(img, int(170 * u), (0, 60, W, 820), f, 3, (1.2, 3.4), u)
    return img


def palm_px(last, wr, k): H, B = last; p = B([wr])[0]; return float(p[0]), float(p[1]) - k


def frame(k, u=1., f=0):
    L = 'dawn'
    if k == 0:
        img = dawn_bg(3, 830, 1620).copy(); img = motes(img, 46, (0, 120, W, 860), f, 5, (1.4, 4.0), u)
        wr = (-1.30, -.62 - .42 * u, .45); arms = [(1, (1.06, 1.05, 0), (1.00, 1.72, .25), 'fist', (70, 0, 180), 0), (-1, (-1.36, .20 - .16 * u, .1), wr, 'open', (182, -4, 6), 1)]
        X_ = dict(G.E['determined'], tears=.55, gy=.022, gx=-.012, hw=.085, dB=.02, glint=1.2)
        lay = Gohan.bust(1120, 500, 236, -.40, .12, -.05, X_, L, f, (.75, .18), 4, False, arms, byaw=-.12, bottom=3.2); px, py = palm_px(Gohan.last, wr, 236 * .86); lay = hand_light(lay, px, py); img = over(img, lay); return mote(img, px, py, 17 * u, f, 1) if u > .3 else img
    if k == 1:
        img = dawn_bg(6, 900, 1560).copy(); img = motes(img, 40, (0, 80, W, 900), f, 8, (1.4, 4.2), u)
        lay = Piccolo.bust(930, 432, 178, .34, .04, .03, 'gentle', L, f, (.45, .1), 5, False, pose='raise', torn=1., byaw=.06); H, B = Piccolo.last; p = B([(-1.95, -1.15, .5)])[0]; px, py = p[0] + 10, p[1] - 178 * 1.14; lay = hand_light(lay, px, py); img = over(img, lay); return mote(img, px, py, 18 * u, f, 2)
    if k == 2:
        img = dawn_bg(12, 690, 1500).copy(); x, y = grid(FH, W); cel = Cel(W, FH)
        # beach, Kame House, palms
        cel.fill(spl(A([(-20, 800), (300, 776), (700, 800), (1100, 786), (1500, 830), (1940, 812), (1940, 1100), (-20, 1100)]), True, corners=(0, 5, 6, 7), n=8), '#6b5880'); cel.fill(spl(A([(-20, 806), (300, 784), (700, 806), (1100, 792), (1500, 836), (1940, 818), (1940, 850), (1500, 868), (1000, 822), (500, 840), (-20, 836)]), True, corners=(0, 5, 6, 10), n=8), '#d99c98')
        cel.fill(A([(1420, 770), (1420, 690), (1650, 690), (1650, 775)]), '#8a6a94'); cel.fill(A([(1580, 690), (1650, 690), (1650, 775), (1580, 772)]), '#e6a89a'); cel.fill(A([(1395, 700), (1470, 610), (1610, 606), (1680, 700)]), '#7c3552'); cel.fill(A([(1580, 607), (1610, 606), (1680, 700), (1610, 700)]), '#d0686a'); cel.line(A([(1395, 700), (1470, 610), (1610, 606), (1680, 700), (1395, 700)]), 2.4, '#241f48', taper=(.02, .02)); cel.fill(A([(1500, 710), (1550, 710), (1550, 772), (1500, 772)]), '#ffd98a'); cel.fill(ell(1612, 728, 14), '#ffe2a0')
        for x0, y0, h, ln in ((1760, 800, 330, .25), (1340, 786, 250, -.2), (110, 800, 420, .3)):
            c = spl(A([(x0, y0), (x0 + ln * 40, y0 - h * .5), (x0 + ln * 160, y0 - h)]), n=8); q, a, b, cc = limb(c, [13, 9, 7]); cel.fill(q, '#3a3262'); cel.line(a if a[:, 0].mean() > b[:, 0].mean() else b, 2.4, '#f0b49c', taper=(.1, .3)); t_ = c[-1]
            for j in range(8): an = j * TAU / 8 + .3 + .04 * math.sin(f * .4 + j); e = t_ + A([math.cos(an), math.sin(an) * .55]) * h * .42; m = (t_ + e) / 2 - [0, h * .12]; cel.fill(wedge(spl(A([t_, m, e]), n=10), lambda t: 22 * np.sin(math.pi * np.clip(t * .9 + .08, 0, 1)) * h / 330), '#2e3560' if j % 2 else '#3d4470')
        img = over(img, cel.rgba()); img = motes(img, 44, (0, 80, W, 800), f, 11, (1.3, 3.8), u); ps = []
        for C_, cx, cy, sc, yw, rl, e, arm_ in ((Roshi, 1330, 462, 134, -.34, .03, 'hope', ((-1.46, .30 - .20 * u, .1), (-1.24, -.35 - .63 * u, .4))), (Bulma, 470, 470, 138, .34, -.02, 'hope', ((-1.35, .48 - .20 * u, .1), (-1.12, -.20 - .62 * u, .4))), (Krillin, 905, 560, 132, .05, 0., 'hope', ((-1.62, .22 - .20 * u, .1), (-1.36, -.50 - .62 * u, .4)))):
            lay = C_.bust(cx, cy, sc, yw, .10, rl, e, L, f, (.35, .1), int(cx) % 17, arm_=arm_); H, B = C_.last; p = B([arm_[1]])[0]; q = (p[0] + 4, p[1] - sc * 1.06); lay = hand_light(lay, *q, r=300, amt=.45); img = over(img, lay); ps.append(q)
        for j, q in enumerate(ps): img = mote(img, q[0], q[1], 12 * u, f, j)
        return img
    return town(u, f)


# ================================================================== the shot g6
RIM_ = [dict(dir=(.9, .10, -.3), col='#ffcf9c', thr=.46, amt=.95), dict(dir=(-.35, .90, -.15), col='#a070ff', thr=.40, amt=.85)]
MOON = ((1380., -690., 900.), (1120., -700., 900.), (1010., -715., 930.), (840., -760., 1010.))          # (cx, cy, r) per cut: the lower limb at y 210, 200, 215, 250


def DAWN2(): return light('dawn', rim=[((-6, 3), '#ffcfa0', 1.), ((4, 6), '#a878ff', .9)])


def moon_light(lay, amt=1.):
    """violet from above: a band along every top edge and a wash that dies down the figure"""
    a = lay[..., 3]; out = rim(lay, 1, 6, '#b48cff', .75 * amt, .6); ys = np.nonzero(a.max(1) > .3)[0]
    if len(ys) == 0: return out
    y0, y1 = ys[0], max(ys[-1], ys[0] + 200); y = np.arange(FH)[:, None]; k = np.clip(1 - (y - y0) / (y1 - y0), 0, 1) ** 1.5 * .20 * amt; out[..., :3] = out[..., :3] + (a * k)[..., None] * rgb('#7a3cff') * (1.15 - np.clip(out[..., :3], 0, 1)); return out


def cheek(lay, hd, D_, side, clip_=None, col='#9a5cff', a=.55, skin_=True):
    """violet plane on the cheek and temple that look away from the sun.  hd = hero3 head info (x y s yaw pitch roll) or an FHead; side -1 / +1"""
    H = hd if isinstance(hd, FHead) else FHead(D_).pose(hd['x'], hd['y'], hd['s'], hd['yaw'], hd['pitch'], hd['roll']); cel = Cel(W, FH); P = H.Pc(spl(A([(side * .40, .10), (side * .62, .16), (side * .78, .02), (side * .80, -.26), (side * .66, -.52), (side * .52, -.62), (side * .56, -.36), (side * .44, -.20)]), True, corners=(0, 5, 7), n=6), squeeze=False); cel.fill(P, col); q = cel.rgba()[..., 3]
    if clip_ is not None: q = q * clip_
    if col is not None and D_ is not None and skin_: c_ = lay[..., :3] / np.maximum(lay[..., 3:4], 1e-3); sk = ((c_[..., 0] > c_[..., 1] * 1.04) & (c_.mean(-1) > .30)).astype(f32); q = q * cv2.GaussianBlur(cv2.morphologyEx(sk, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), (0, 0), 1.2)          # skin only: never over the hair or the eyes
    out = lay.copy(); k = (q * a * lay[..., 3])[..., None]; out[..., :3] = out[..., :3] * (1 - k) + (np.clip(out[..., :3], 0, 1) * .55 + rgb(col) * .55 * lay[..., 3:4]) * k; return out


def swing(lf, hit=5):
    """progress of a rising arm inside a ten-frame cut: hold, dip (anticipation), whip, overshoot, settle"""
    return float(comp.lerp_keys(lf, [(0, .0), (1, -.04), (2, -.12), (3, -.14), (4, .42), (hit, 1.07), (hit + 1, 1.02), (hit + 2, .985), (hit + 3, 1.), (12, 1.005)], lambda t: t))


def bez(a, b, c, t): a, b, c = A(a, float), A(b, float), A(c, float); return a * (1 - t) ** 2 + 2 * b * t * (1 - t) + c * t * t


def raise_pose(p, side='L', rest=(.02, .42, -.24), via=(.62, .52, .30), top=(.42, .24, .93), **k):
    """hero3 pose: one arm from the chest, out and up"""
    from db.hero3 import POSES
    q = dict(POSES['stand']); t = float(np.clip(p, -.2, 1.15)); o = 'R' if side == 'L' else 'L'; q['h' + side] = tuple(bez(rest, via, top, t)); U_ = POSES['hands_up']
    q['e' + side] = tuple(A((.8, -.6, -.5)) * (1 - np.clip(t, 0, 1)) + A(U_['e' + side]) * np.clip(t, 0, 1)); q['w' + side] = tuple(A(U_['w' + side]) * np.clip(t, 0, 1)); q['g' + side] = 'fist' if t < .30 else 'relax' if t < .75 else 'open'; q['c' + side] = (3 * t, 0)
    q.update(k); return q


def leave(img, x, y, lf, r, hit=5, seed=0, u=1.):
    """the light leaves the palm: a kindling on `hit`, then it climbs and draws a thin line behind it"""
    if lf < hit: return img
    d = lf - hit; yy = y - r * 1.4 - 8 * d - 9.5 * d * d * u; xx = x + 5 * math.sin(d * .8 + seed); k = r * (1.35 if d == 0 else 1.1 if d == 1 else 1.)
    if d > 0:
        cel = Cel(W, FH); P = A([(x, y - r), ((x + xx) / 2 + 3, (y - r + yy) / 2), (xx, yy)]); cel.line(P, r * .34, '#ffe9b8', taper=(.95, .1), press=0, alpha=.75); img = over(img, cel.rgba())
    return mote(img, xx, yy, k, lf, seed, 1.25 if d == 0 else 1.)


def shade_layer(lay, rise, x0=960, warm=1.):
    """before sunrise figures are cooler and darker; the gradient turns warm cut by cut"""
    lay = para(lay, (x0 + 900, 150), (x0 - 800, 1050), [(0, '#fff0de'), (.45, '#f0dde8'), (1, '#8088bc')], 'mul', .9); k = .62 + .42 * rise; lay[..., :3] *= A([k, k * (.96 + .04 * rise), 1 - .28 * (1 - rise)], f32); return lay


def g6_gohan(lf, f):
    from db.hero3 import Fig
    from db.cam import Camera3D
    rise = .16; g = lf if 2 <= lf <= 6 else (lf // 2) * 2; p = swing(g); tilt = float(comp.lerp_keys(lf, [(0, 0.), (2, -.015), (6, .05), (10, .065)])); cam = Camera3D((.30, -1.42, .95), (.05, 0, 1.010 + tilt), 1860.); dip = float(comp.lerp_keys(g, [(0, .0), (3, -.030), (5, .022), (7, .010), (10, .012)]))
    F = Fig('gohan', 'nohat').pose(raise_pose(p, 'L', cst=(-2 - 4 * max(0, p), 0, -6 - 6 * p), nck=(3, 0, 6), hed=(9 - 4 * p, 0, 8 + 4 * p), hR=(.30, .16, -.86), gR='fist', eR=(.8, -.6, 0))).place(-.13, 0, dip, heading=math.pi + .62); F.blow(-1.1, .15, .2)
    e = dict(G.E['determined'], hw=.085, glint=1.25, ir=.94) if lf < 4 else dict(G.E['shout'], hw=.15, jaw=.07, mL=[-.02, -.16, -.20, -.16, -.02], glint=1.25)
    img = dawn_bg(3, 960, 1640, True, rise, MOON[0]).copy(); img = motes(img, 34, (0, 120, W, 940), f, 5, (1.2, 3.4), 1.); lay = F.draw(cam, f, light=(.7, .45, .35), pal='dawn', rim=RIM_, expr=dict(expr=e)); hd = F.info['head']; L = light('dawn'); u = hd['s']; wl = lw(u); o_ = A([hd['x'], hd['y']]) + A([.10, 1.22]) * u; B = Body(o_[0], o_[1], u, .40, 0., .02); ck = Cel(W, FH); Gohan.scarf(ck, B, L, wl, 4, (-.95, .16), f, 'back'); bk = grade(ck.layer(), L, (o_[0] - 600, o_[1] - 200, o_[0] + 300, o_[1] + 400), 1.)
    ck = Cel(W, FH); Gohan.scarf(ck, B, L, wl, 4, (-.95, .16), f, 'front'); fr = grade(ck.layer(), L, (o_[0] - 300, o_[1] - 200, o_[0] + 300, o_[1] + 300), 1.); hm = F.mask('head'); up_ = fr * (1 - np.clip(cv2.GaussianBlur((hm > .3).astype(f32), (0, 0), 1.), 0, 1))[..., None]
    lay = lover(lover(bk, lay), up_)
    lay = cheek(lay, hd, G.D, -1, np.clip(F.mask('head'), 0, 1))
    px, py = [float(v) for v in F.info['palms']['L']]
    lay = moon_light(shade_layer(lay, rise, 900))
    if lf >= 5: lay = hand_light(lay, px, py, 380, .60)
    img = over(img, lay); return leave(img, px, py - 10, lf, 19, seed=1)


def rag_cape(cel, B, L, w, u, f, seed=6):
    """half a cape: a torn sheet flying to +x from the far shoulder, a wave running out to its ragged edge, two loose strips.  Returns the mask of its holes."""
    from chars import piccolo as PI
    r = np.random.default_rng(seed)
    def Q(x, y): k = max(0., x - .9); ph = f * .62 - x * 1.25 + y * .25; return (.55 + (x - .55) * 1.42 + .05 * k * math.cos(ph), y * 1.04 + .125 * k * math.sin(ph) + .03 * k * k)
    P = lambda M: B([Q(*m) for m in M]); n = 13; t_ = np.linspace(0, 1, n)
    top = [(.55, .34), (1.2, .30), (1.9, .46), (2.7, .50), (3.5, .78), (4.35, .74)]; tr = [(4.35 - 1.15 * t - (r.uniform(.30, .95) if i % 2 else -r.uniform(.0, .38)) * (0 < i < n - 1), .74 + 2.75 * t + r.uniform(-.07, .07)) for i, t in enumerate(t_)]; bot = [(3.2, 3.49), (2.5, 3.16), (1.9, 3.50), (1.3, 3.30), (.6, 3.9), (.2, 3.9), (.2, 2.0)]
    for j, (x0, y0, ln, th) in enumerate(((3.75, 1.15, 1.75, .17), (3.05, 3.05, 1.35, .20))): C = spl(P([(x0 + ln * q, y0 + (.10 - .30 * j) * q + .16 * math.sin(f * .62 - 2.6 * q + j * 2)) for q in (0, .33, .66, 1)]), n=9); poly(cel, wedge(C, lambda t, th=th: u * th * (1 - t) ** .6 * (1 + .22 * np.sin(t * 11))), cloth(L, PI.CAPE, PI.CAPEH), w[1] * 1.2, 50 + j, sh=[wedge(C + [0, .10 * u], u * th * .6, 0.)])
    S = np.vstack([spl(P(top), n=8), P(tr), spl(P(bot), n=6)]); folds = []
    for j, (a, b_, th) in enumerate((((.8, .85), (4.0, 1.30), .30), ((.7, 1.5), (3.9, 2.05), .24), ((.6, 2.3), (3.5, 2.80), .30))): C = spl(P([(a[0] + (b_[0] - a[0]) * q, a[1] + (b_[1] - a[1]) * q + .10 * math.sin(3.2 * q + j)) for q in np.linspace(0, 1, 5)]), n=8); folds.append(wedge(C, lambda t, th=th: u * th * (.15 + .85 * t) * (1 - .25 * t)))
    poly(cel, S, cloth(L, PI.CAPE, PI.CAPEH), w[0] * 1.15, 44, sh=folds + [P([(.2, .3), (1.5, .3), (1.25, 1.6), (1.4, 3.4), (.2, 4.)])]); hc = Cel(W, FH); nc = []
    for x0, y0, k in ((3.10, 1.62, .26), (2.45, 2.52, .19)): H_ = P([(x0 + k * 1.5 * math.cos(q) * r.uniform(.5, 1.25), y0 + k * math.sin(q) * r.uniform(.55, 1.2)) for q in np.linspace(0, 2 * math.pi, 9)[:-1]]); hc.fill(H_, '#ffffff'); cel.line(np.vstack([H_, H_[:1]]), w[1] * 1.5, L['line'], taper=(.1, .1))
    for j, (x0, y0) in enumerate(((2.2, .52), (3.0, .62), (1.6, 3.42))): cel.line(P([(x0, y0), (x0 + .16, y0 + (.34 if j < 2 else -.36)), (x0 + .10, y0 + (.62 if j < 2 else -.7))]), w[1] * 1.3, L['line'], taper=(.05, .9))
    return hc.rgba()[..., 3]


def g6_piccolo(lf, f):
    """bare-headed, awake, the cape in rags: drawn by hand round the posed body"""
    from db.hero3 import Fig
    from db.cam import Camera3D
    from chars import piccolo as PI
    rise = .42; g = lf if 2 <= lf <= 6 else (lf // 2) * 2; p = swing(g); cam = Camera3D((-.72, -1.62, 1.50), (.10, 0, 1.885 + .006 * lf), 1330.); dip = float(comp.lerp_keys(g, [(0, 0.), (3, -.04), (5, .02), (10, .01)])); q_ = float(np.clip(p, 0, 1))
    F = Fig('piccolo', 'nocape').pose(raise_pose(p, 'R', rest=(.04, .36, -.36), via=(.70, .34, .20), top=(.34, .16, .95), cst=(-4 - 3 * p, 0, 8 + 6 * p), nck=(2, 0, -4), hed=(6 - 7 * p, 0, -12), hL=(.36, .02, -.88), gL='fist', eL=(.8, -.8, 0), fL=(.11, .10, .045), fR=(.10, -.10, .05))).place(.10, 0, dip, heading=math.pi - .42); F.blow(.9, .5, .25)
    AW = dict(PI.E['stern'], U=[-.036, .024, .058, .078, .058], Lo=[-.058, -.078, -.084, -.068, -.010], gx=-.012, gy=.020, glint=1.15, furrow=1., hw=.120, mU=[-.014, .003, .006, .003, -.014], mL=[-.014, .003, .006, .003, -.014]); RS = dict(AW, U=[-.030, .034, .070, .086, .062], dB=.012, gy=.026, hw=.150, my=-.650, mU=[.004, .022, .020, .022, .006], mL=[-.010, -.040, -.046, -.040, -.008], clench=1, naso=.7)
    img = dawn_bg(6, 1010, 420, True, rise, MOON[1]).copy(); img = motes(img, 40, (0, 80, W, 980), f, 8, (1.2, 3.8), 1.); lay = F.draw(cam, f, light=(-.7, .40, .35), pal='dawn', rim=[dict(dir=(-.9, .10, -.3), col='#ffcf9c', thr=.46, amt=.95), dict(dir=(.35, .90, -.15), col='#a070ff', thr=.56, amt=.85)], expr=dict(expr=AW if lf < 4 else RS)); px, py = [float(v) for v in F.info['palms']['R']]
    hd = F.info['head']; u = hd['s']; L = flip(light('dawn')); wl = lw(u); ax = A([math.sin(hd['roll']), -math.cos(hd['roll'])]); o_ = A([hd['x'], hd['y']]) - ax * 1.66 * u + A([-.04, 0]) * u; B = Body(o_[0], o_[1], u, -.30, 0., .10 * q_); sway = 1.25 + .16 * math.sin(f * .8)
    cb = Cel(W, FH); Piccolo.cape_back(cb, B, L, wl, 6, 1.02, sway, 4.3, 1.); lb = grade(cb.layer(), L, (o_[0] - 5 * u, o_[1] - u, o_[0] + 5 * u, o_[1] + 5 * u), 1.1)
    cy_ = Cel(W, FH); Piccolo.yoke(cy_, B, L, wl, 6 + int(f // 2) % 2 * 0, .10 + .05 * math.sin(f * .9), torn=1.); ly = grade(cy_.layer(), L, (o_[0] - 3 * u, o_[1] - u, o_[0] + 3 * u, o_[1] + 1.4 * u), 1.1)
    front = np.clip(cv2.GaussianBlur(np.maximum(F.mask('head'), (F.mask('uarmR') + F.mask('farmR') + F.mask('bandR') + F.mask('handR')) * (0. if q_ < .25 else 1.)), (0, 0), .8), 0, 1)[..., None]
    ck = Cel(W, FH); hole = rag_cape(ck, B, L, wl, u, (f // 2) * 2)  # what is left of the cape, blown out behind him
    lr = grade(ck.layer(), L, (o_[0], o_[1], o_[0] + 5 * u, o_[1] + 5 * u), 1.) * (1 - hole[..., None]); lay = lover(lover(lover(lb, lr), lay), ly * (1 - front)); lay = cheek(lay, hd, PI.D, 1, np.clip(F.mask('head'), 0, 1), a=.50, skin_=False); lay = lay[:, ::-1]; lay = shade_layer(lay, rise, W - 900); lay = moon_light(lay[:, ::-1].copy())
    if lf >= 5: lay = hand_light(lay, px, py, 360, .6)
    img = over(img, lay); return leave(img, px, py - 12, lf, 20, seed=2)


def turtle(cel, x, y, k, L, lf):
    """the old sea turtle lifts a flipper too"""
    sh = cloth(L, ('#6a8a5a', '#46654a', '#2c4436')); sk = cloth(L, ('#c9a46a', '#946c48', '#664432')); p = max(0, swing(lf, 6)); w = lw(k * .9); P = lambda M: spl(A(M, float) * k + [x, y], True, n=8)
    F_ = capsule(A([(.55, .05), (1.0, -.10 - .35 * p), (1.28, -.05 - .78 * p)]) * k + [x, y], [.16 * k, .18 * k, .08 * k], 9); poly(cel, F_, sk, w[0], 3, sh=[F_ + A([0, .08]) * k])
    S = P([(-1.2, .30), (-1.1, -.20), (-.6, -.62), (0, -.76), (.6, -.60), (.98, -.18), (1.02, .28), (0, .40)]); poly(cel, S, sh, w[0] * 1.2, 4, sh=[P([(-1.3, .5), (-1.0, .10), (-.3, .02), (.4, .12), (1.2, .0), (1.2, .6)])], lines=[(A([(-.9, -.25), (-.3, -.10), (.4, -.12), (.9, -.2)]) * k + [x, y], .7), (A([(-.35, -.66), (-.40, -.35), (-.30, -.10)]) * k + [x, y], .6), (A([(.30, -.68), (.36, -.40), (.40, -.12)]) * k + [x, y], .6)])
    H_ = P([(.85, .10), (1.05, -.28), (1.40, -.46), (1.66, -.30), (1.68, -.02), (1.40, .14), (1.05, .24)]); poly(cel, H_, sk, w[0], 5, sh=[P([(.85, .3), (1.1, -.02), (1.4, .02), (1.7, -.02), (1.6, .4)])]); c = A([1.42, -.20]) * k + [x, y]; cel.fill(ell(c[0], c[1], .055 * k, .075 * k), '#120e14'); cel.fill(ell(c[0] + .02 * k, c[1] - .03 * k, .02 * k), '#ffffff'); cel.line(A([(1.50, -.02), (1.60, -.05), (1.68, -.10)]) * k + [x, y], w[1], L['line'], taper=(.3, .3))


def g6_beach(lf, f):
    rise = .68; LD = DAWN2(); img = dawn_bg(12, 700, 1560, True, rise, MOON[2]).copy(); img = beach(img, f); img = motes(img, 44, (0, 80, W, 820), f, 11, (1.2, 3.6), 1.); ps = []; L = 'dawn'
    def one(C_, cx, cy, sc, yw, rl, e, d, rest, via, top, bob, mir=False):
        if mir:
            lay, pm = one(C_, W - cx, cy, sc, yw, rl, e, d, rest, via, top, bob, None); return lay[:, ::-1].copy(), A([W - 1 - pm[0], pm[1]])
        Lq = flip(LD) if mir is None else LD
        g = lf if 2 + d <= lf <= 6 + d else (lf // 2) * 2; p = swing(max(0, g - d)); q_ = float(np.clip(p, -.15, 1.08)); el = bez((-1.36, 1.42, .1), (-1.95, .95, .2), (top[0] * 1.16 - .14, top[1] + 1.14, .1), float(np.clip(q_, 0, 1))); qt = q_ if q_ < 0 else 1 - (1 - min(q_, 1.)) ** 3 + max(0., q_ - 1); th = math.radians(-4 + (96 + 6 * (top[0] < -1.2)) * qt); ln_ = 1.17 - .10 * math.sin(math.pi * np.clip(q_, 0, 1)); wr = el + A([math.cos(th) * ln_, -math.sin(th) * ln_, .3])          # the forearm keeps its length and turns at the elbow
        C_.hp = 'fist' if p < .3 else 'relaxed' if p < .7 else 'open'; lay = C_.bust(cx, cy - bob * 14 * max(0, p) + (10 if 2 + d <= lf <= 3 + d else 0), sc, yw, .02 + .13 * max(0, p), rl, e[0] if lf < 4 + d else e[1], Lq, f, (.30, .10), int(cx) % 17, arm_=(tuple(el), tuple(wr)), rims=True); H, B = C_.last; q = B([wr])[0]; n = q - B([el])[0]; n /= np.hypot(*n); pm = q + n * C_.hs * sc * 1.15; lay = cheek(lay, H, C_.D, (1 if mir is None else -1), None, a=.42); return moon_light(shade_layer(lay, rise, cx) if mir is not None else shade_layer(lay[:, ::-1], rise, W - cx)[:, ::-1].copy()), pm
    for C_, a in ((Roshi, (1515, 452, 112, -.42, .03, ('neutral', 'hope'), 2, (-.55, 1.15, .6), (-2.1, .55, .5), (-1.35, -1.30, .5), 1., True)), (Bulma, (1120, 452, 138, .30, -.05, ('neutral', 'hope'), 1, (-.35, .95, .7), (-2.0, .45, .6), (-1.22, -1.05, .5), 1.)), (Krillin, (520, 520, 214, .46, .04, ('firm', 'shout'), 0, (-.40, 1.15, .9), (-2.5, .55, .8), (-1.12, -1.42, .8), 1.2))):
        lay, pm = one(C_, *a)
        if lf >= 5 + a[6]: lay = hand_light(lay, *pm, r=a[2] * 2.2, amt=.5)
        img = over(img, lay); ps.append((pm, a[2], 5 + a[6]))
    cel = Cel(W, FH); turtle(cel, 1650, 1010, 120, light('dawn'), lf); lay = shade_layer(grade(cel.layer(), light('dawn'), (1480, 880, 1900, 1080), .8), rise, 1650); img = over(img, lay)
    for j, (pm, sc, h) in enumerate(ps): img = leave(img, pm[0], pm[1], lf, 6 + sc * .055, h, j, .8)
    return img


def beach(img, f):
    cel = Cel(W, FH)
    cel.fill(spl(A([(-20, 800), (300, 776), (700, 800), (1100, 786), (1500, 830), (1940, 812), (1940, 1100), (-20, 1100)]), True, corners=(0, 5, 6, 7), n=8), '#7a6088'); cel.fill(spl(A([(-20, 806), (300, 784), (700, 806), (1100, 792), (1500, 836), (1940, 818), (1940, 856), (1500, 872), (1000, 826), (500, 846), (-20, 842)]), True, corners=(0, 5, 6, 10), n=8), '#e8ac9c')
    for j in range(7): yy = 900 + j * 28; cel.line(spl(A([(60 + j * 190, yy), (260 + j * 170, yy - 10), (520 + j * 150, yy + 4)]), n=8), 3 + j * .6, '#94749a', taper=(.4, .5), seed=j, alpha=.7)
    cel.fill(A([(1250, 770), (1250, 682), (1440, 682), (1440, 775)]), '#94709c'); cel.fill(A([(1370, 682), (1440, 682), (1440, 775), (1370, 772)]), '#f0b4a0'); cel.fill(A([(1226, 692), (1290, 606), (1410, 602), (1470, 692)]), '#84385a'); cel.fill(A([(1384, 603), (1410, 602), (1470, 692), (1406, 692)]), '#dc7470'); cel.line(A([(1226, 692), (1290, 606), (1410, 602), (1470, 692), (1226, 692)]), 2.4, '#241f48', taper=(.02, .02)); cel.fill(A([(1310, 704), (1350, 704), (1350, 772), (1310, 772)]), '#ffd98a'); cel.fill(ell(1406, 724, 12), '#ffe8b0')
    for x0, y0, h, ln in ((1800, 806, 350, .25), (1200, 790, 230, -.25), (150, 804, 440, .3)):
        c = spl(A([(x0, y0), (x0 + ln * 40, y0 - h * .5), (x0 + ln * 160, y0 - h)]), n=8); q, a, b, cc = limb(c, [13, 9, 7]); cel.fill(q, '#40386a'); cel.line(a if a[:, 0].mean() > b[:, 0].mean() else b, 2.6, '#ffc0a0', taper=(.1, .3)); t_ = c[-1]
        for j in range(9): an = j * TAU / 9 + .3 + .05 * math.sin(f * .4 + j); e = t_ + A([math.cos(an) - .18, math.sin(an) * .55]) * h * .42; m = (t_ + e) / 2 - [0, h * .12]; cel.fill(wedge(spl(A([t_, m, e]), n=10), lambda t: 22 * np.sin(math.pi * np.clip(t * .9 + .08, 0, 1)) * h / 330), '#2e3560' if j % 2 else '#46507e')
    return over(img, cel.rgba())


def g6_town(lf, f): return town2(lf, f)


def bird(cel, x, y, k, fl, col='#f4f0ea', line='#1a1430'):
    """small bird: perched (fl None) or flying (wing phase)"""
    B_ = spl(A([(-1.0, .10), (-.5, -.35), (.2, -.45), (.62, -.62), (.95, -.45), (.80, -.10), (.45, .30), (-.2, .38), (-1.5, .50)]) * k + [x, y], True, corners=(0, 8), n=6); cel.fill(B_, col); cel.line(np.vstack([B_, B_[:1]]), max(1.2, k * .07), line, taper=(.05, .05)); cel.fill(A([(.92, -.50), (1.28, -.40), (.92, -.32)]) * k + [x, y], '#f2a53a'); cel.fill(ell(x + .70 * k, y - .42 * k, .07 * k), line)
    if fl is None: W_ = spl(A([(-.2, -.2), (.35, -.1), (.1, .2), (-.9, .30)]) * k + [x, y], True, corners=(3,), n=5); cel.fill(W_, '#b7b0c4')
    else:
        s = math.sin(fl)
        for q, c in ((-1, '#c9c2d4'), (1, col)): W_ = spl(A([(.25, -.35), (.0 - .2 * q, -.9 - 1.2 * s), (-.6 - .3 * q, -1.2 - 1.6 * s), (-.55, -.5 - .5 * s), (-.35, -.2)]) * k + [x, y], True, corners=(2,), n=6); cel.fill(W_, c); cel.line(W_, max(1.2, k * .07), line, taper=(.1, .1))


def town2(lf, f):
    """the whole town: a wave of hands from the back to us; the sun's rim comes up"""
    rise = .95; img = dawn_bg(9, 720, 1380, False, rise, MOON[3]).copy(); x, y = grid(FH, W); cel = Cel(W, FH); r = np.random.default_rng(4)
    cel.fill(spl(A([(-20, 740), (120, 640), (260, 700), (420, 580), (560, 660), (760, 560), (900, 650), (1100, 600), (1300, 690), (1500, 630), (1700, 700), (1940, 650), (1940, 1100), (-20, 1100)]), True, corners=(0, 2, 4, 6, 8, 10, 11, 12, 13), n=6), '#8a6ea4'); cel.line(spl(A([(420, 580), (560, 660), (760, 560), (900, 650), (1100, 600), (1300, 690)]), corners=(1, 2, 3, 4), n=6), 2.6, '#ffc8a8', taper=(.1, .1), alpha=.8)
    def dome(cx, by, R_, col, lit, win='#ffe2a0'):
        t = np.linspace(math.pi, TAU, 30); D_ = np.vstack([np.c_[cx + np.cos(t) * R_, by + np.sin(t) * R_ * .92], [(cx + R_, by + R_ * .5), (cx - R_, by + R_ * .5)]]); cel.fill(D_, col); cel.clip(D_); cel.fill(ell(cx + R_ * .95, by - R_ * .1, R_ * .62, R_ * 1.1), lit, .9); cel.fill(A([(cx - R_, by + R_ * .06), (cx + R_, by + R_ * .06), (cx + R_, by + R_ * .12), (cx - R_, by + R_ * .12)]), mixc(col, '#000000', .25), .6); cel.unclip()
        for q in (-.45, .0): cel.fill(ell(cx + q * R_, by - R_ * .28, R_ * .13, n=20), win); cel.line(ell(cx + q * R_, by - R_ * .28, R_ * .13, n=24), 1.4, '#2a2450', taper=(0, 0))
        cel.line(np.c_[cx + np.cos(t) * R_, by + np.sin(t) * R_ * .92], 2.2, '#241f48', taper=(.1, .1), seed=int(cx))
    for cx, by, R_, a in ((90, 800, 130, 0), (860, 772, 84, 0), (1080, 800, 120, 1), (1560, 800, 140, 0), (1830, 780, 120, 1)): dome(cx, by, R_, ('#6a5694', '#785c90', '#5c5090')[a], ('#f0b0a0', '#ffc09c', '#e0a0a8')[a])
    T_ = A([(1180, 800), (1192, 470), (1180, 430), (1200, 380), (1220, 430), (1208, 470), (1220, 800)]); cel.fill(T_, '#5c4c88'); cel.line(A([(1218, 440), (1208, 470), (1220, 790)]), 3.4, '#ffc8a4', taper=(.1, .3)); cel.fill(ell(1200, 372, 8), '#fff0c0')
    # the long-necked dinosaur: turns its head up to the lights
    nd = float(comp.lerp_keys(lf, [(0, 0.), (3, -8.), (6, 34.), (10, 38.)])); hx = float(comp.lerp_keys(lf, [(0, 0.), (6, -10.), (10, -12.)]))
    N_ = spl(A([(392, 900), (404, 720), (376, 560), (392 + hx * .3, 440 - nd * .7), (446 + hx, 380 - nd), (520 + hx, 372 - nd * 1.25), (568 + hx, 398 - nd * 1.5), (552 + hx, 424 - nd * 1.3), (490 + hx, 428 - nd), (456 + hx * .6, 462 - nd * .6), (452, 570), (486, 720), (540, 900)]), True, n=8); cel.fill(N_, '#4e7a78'); cel.clip(N_); cel.fill(N_ + [52, 6], '#8fb08c'); cel.fill(N_ + [86, 10], '#f4c69a'); cel.unclip(); outline(cel, N_, 3.2, '#1c2440', 3); cel.fill(ell(518 + hx, 392 - nd * 1.2, 6, 8), '#10142a'); cel.fill(ell(520 + hx, 389 - nd * 1.2, 2.2), '#ffffff'); cel.line(A([(540 + hx, 412 - nd * 1.4), (556 + hx, 408 - nd * 1.5), (566 + hx, 402 - nd * 1.5)]), 2.4, '#1c2440', taper=(.2, .2))
    for j in range(5): q = N_[int(len(N_) * (.20 + .035 * j))]; cel.fill(A([q + [-2, 14], q + [-20, 0], q + [-2, -12]]), '#3c5f66')
    # a river on the left with a fishing boat
    cel.fill(spl(A([(-20, 832), (180, 826), (330, 850), (300, 900), (80, 930), (-20, 930)]), True, corners=(0, 5), n=7), '#c79ab0'); cel.fill(spl(A([(40, 850), (200, 846), (260, 862), (120, 868)]), True, n=6), '#ffd9b4')
    B_ = spl(A([(60, 872), (110, 898), (230, 900), (292, 866), (262, 876), (96, 882)]), True, corners=(0, 3), n=7); cel.fill(B_, '#5a3a4c'); cel.line(B_, 2.2, '#1c1636', taper=(.1, .1))
    img = over(img, cel.rgba()); haze = np.exp(-((y - 800) / 120.) ** 2); img = img * (1 - .30 * haze[..., None]) + rgb('#ffc8a4') * .30 * haze[..., None]
    lights = []
    def lit(img, lights):
        cel = Cel(W, FH); hm = np.zeros((FH // 4, W // 4), f32)
        for j, (q, h, d) in enumerate(lights):
            t = lf + 1 - (5 + min(d, 2))
            if t < 0: continue
            k = max(1.8, h * .019) * (1.3 if t == 0 else 1.); yy = q[1] - (h * .028 * t + h * .010 * t * t); cel.fill(ell(q[0], yy, k, n=16), '#fffaf0')
            if t > 0: cel.line(A([(q[0], yy + h * .05 * t), (q[0], yy + h * .02 * t), (q[0], yy)]), k * .7, '#ffe9b8', taper=(.9, .1), press=0, alpha=.6)
            for a in range(4): b = a * TAU / 4; c = A([math.cos(b), math.sin(b)]); n_ = A([-c[1], c[0]]) * k * .22; cel.fill(A([n_, c * k * (3. if a % 2 else 2.), -n_]) + [q[0], yy], '#fff2cc', .9)
            cv2.circle(hm, (int(q[0] / 4), int(yy / 4)), max(1, int(k / 2.2)), 1., -1)
        g = cv2.resize(cv2.GaussianBlur(hm, (0, 0), 3.5) + .8 * cv2.GaussianBlur(hm, (0, 0), 10), (W, FH)); return over(screen(img, np.clip(g * 1.5, 0, 1)[..., None] * rgb('#ffd796')), cel.rgba())
    W_ = lambda d: max(0., swing(max(0, lf + 1 - min(d, 2)))) if lf + 1 >= d else 0.
    cel = Cel(W, FH)                                                # two fishermen in hats, one with an oar
    for x0, sd in ((128, 61), (206, 64)): lights += [(q, 120, 1) for q in person(cel, x0, 896, 118, sd, .30, '#ffc8a4', W_(0), 'farmer', w=1.4)]
    cel.line(A([(252, 800), (282, 860), (312, 918)]), 4, '#4a3040', taper=(.05, .05)); img = over(img, cel.rgba())
    for row, (by, h0, n, dp, rc, ww, d0) in enumerate(((905, 125, 40, .58, '#f0b6a4', .9, 0), (968, 185, 25, .36, '#f6b89c', 1.2, 1), (1060, 272, 15, .16, '#ffc59c', 1.7, 1))):
        cel = Cel(W, FH); xs = (np.arange(n) + r.uniform(-.38, .38, n)) / (n - 1) * (W + 160) - 80
        for j, x0 in enumerate(xs):
            jy, jh, sd = r.normal(0, h0 * .05), r.uniform(.78, 1.12), int(r.integers(1 << 30))
            if row == 0 and x0 < 340: continue
            d = d0 + int(sd % 3); lights += [(q, h0, d) for q in person(cel, x0, by + jy, h0 * jh, sd, dp, rc, W_(d), w=ww)]
        img = over(img, cel.rgba())
        if row < 2: k = np.exp(-((y - by) / (h0 * .6)) ** 2) * (.18 - .06 * row); img = img * (1 - k[..., None]) + rgb('#f0b4a4') * k[..., None]
        img = lit(img, lights); lights = []
    # foreground: backs close to us, each somebody
    cel = Cel(W, FH)
    for x0, h, sd, kd, d in ((1060, 570, 21, 'woman', 2), (1400, 490, 33, 'old', 3), (640, 570, 12, 'man', 2)): lights += [(q, h, d) for q in person(cel, x0, 1330, h, sd, 0., '#ffb98c', W_(d), kd, w=3.6)]
    ch = person(cel, 648, 1016, 306, 77, 0., '#ffc298', W_(2), 'girl', w=3.2); lights += [(q, 306, 2) for q in ch[1:]]
    cel.line(A([(1452, 1180), (1476, 1010), (1494, 900)]), 13, '#5a3a3c', taper=(.02, .02), press=0)                # granny's stick
    lights += [(q, 660, 1) for q in person(cel, 190, 1350, 640, 5, 0., '#ffb98c', W_(1), 'farmer', w=4.)]; lights += [(q, 690, 3) for q in person(cel, 1760, 1350, 690, 8, 0., '#ffb98c', W_(3), 'bear', w=4.)]
    D_ = spl(A([(1520, 1100), (1532, 1020), (1524, 968), (1536, 930), (1552, 952), (1570, 950), (1588, 926), (1600, 960), (1620, 984), (1612, 1004), (1590, 1010), (1600, 1060), (1612, 1100)]), True, corners=(3, 6, 8), n=7); cel.fill(D_, '#6a4650'); cel.clip(D_); cel.fill(D_ + [38, 0], '#e0a080'); cel.unclip(); outline(cel, D_, 3., '#1a1430', 2); cel.line(spl(A([(1526, 1060), (1490, 1030 - 10 * math.sin(f * 1.5)), (1484, 985 - 12 * math.sin(f * 1.5))]), n=6), 14, '#6a4650', taper=(0, .5), press=0)
    if ch:                                                          # the bird that sat on the child's hand goes first
        q = ch[0]
        if lf < 4: bird(cel, q[0] + 4, q[1] + 8, 24, None)
        else: d = lf - 4; bird(cel, q[0] + 26 * d, q[1] - 22 * d - 4 * d * d, 26, 1.2 + d * 1.9)
    img = over(img, cel.rgba()); img = lit(img, lights)
    return motes(img, int(60 + 11 * lf), (0, 40, W, 820), f, 3, (1.2, 3.4), 1.)


# ================================================================== for other lanes
def _flat(L, x, y, s, side, skin, bld=None, ang=0., press=1., band_=None, seed=0, claw=False, arm_=4.4, spread=1., tilt=-16.):
    """one open hand seen from its back, pressed flat on something in front of it: fingers point to `ang` degrees (0 = up, + = left), forearm leaves the other way -> premultiplied layer"""
    from db import hands as HD
    k = spread; p = HD.pose((-4 * press + 10 * (1 - press), 8 + 10 * (1 - press), 6, -8 * k), (-6 * press + 12 * (1 - press), 7 + 10 * (1 - press), 5, -1), (-5 * press + 12 * (1 - press), 8 + 12 * (1 - press), 6, 6 * k), (-2 * press + 14 * (1 - press), 12 + 12 * (1 - press), 8, 15 * k), thumb=(58, 8, 46, 2, -4), wrist=(-20 * press, 0)); st = dict(arm=arm_)
    if band_ is not None: st['bands'] = ((.26, .95),)
    if claw: st['nails'] = 'claw'; st['claw'] = .05
    h = HD.Hand(p, side, HD.euler(6 if side == 'R' else -6, tilt, ang), (0, 0, 0), bld or HD.MALE, pivot=(0, .60, 0), **st); g = HD.render([h], HD.Cam(float(x), float(y), s, 7), light=ld3(L, .6), res=int(np.clip(s * 2.6, 140, 640)), clip=(-160, -160, W + 160, FH + 160)); pal = dict(HD.PAL['day'], **hpal(L, skin, ('#f2efe0', '#c8c4b0', '#8a8674') if claw else None))
    if band_ is not None: pal['band'] = tone(L, band_, 0)
    cel = Cel(W, FH); HD.ink(cel, g, pal, w=max(1.4, lw(s * .85)[0]), seed=seed, marks=.7); return cel.layer()


def chain(u, f=0, at=((660., 560.), (1260., 540.)), s=118., light_='gold', many=12):
    """g8 50-69, seen from behind the two fighters: put it over their two backs (at = a point between the shoulder blades of each, px).
    u 0..1:  0 the boy's two little palms land, one on each back;  .28 Piccolo's big hands land beside them (small hand, big hand, as in b3);  .52.. `many` hands from every edge, each on its own frame;
    u > .86 all of them sink and push together.  s = px of one child hand unit (wrist to knuckle).  returns premultiplied RGBA (1080, 1920, 4)"""
    from chars import piccolo as PI, friends as FR
    from db import hands as HD
    L = light(light_); out = np.zeros((FH, W, 4), f32); push = float(smooth(.84, .94, u)); j_ = 1.4 * float(u > .5); r = np.random.default_rng(6); land = lambda t0, d=.06: float(np.clip((u - t0) / d, 0, 1)); fin = lambda l, c: grade(l, L, (c[0] - 400, c[1] - 400, c[0] + 400, c[1] + 400), 1.)
    def put(l): nonlocal out; a = l[..., 3]; sh = np.clip(blur(shifted(a, 5, 12), 9) * .5, 0, 1) * (1 - a); out[..., :3] *= 1 - sh[..., None] * .8; out[..., 3] = out[..., 3] + sh * .55 * (1 - out[..., 3]); out = lover(out, l)
    folk = [(FR.B_SKIN, HD.FEMALE, .86, None), (FR.K_SKIN, HD.MALE, 1.04, FR.K_BLUE), (FR.R_SKIN, HD.MALE, .98, None), (('#b8845c', '#7c5238', '#4e3226'), HD.HEAVY, 1.08, None), (('#f2c4a0', '#c48a72', '#8a5a50'), HD.SLIM, .94, None), (('#8a5a3c', '#5c3a28', '#3a2418'), HD.HEAVY, 1.16, None), (('#ffdfc4', '#e2a08a', '#b2706a'), Gohan.KID, .70, None), (('#d8a078', '#a06a50', '#6e4438'), HD.MALE, 1.0, None)]
    mid = A([(at[0][0] + at[1][0]) / 2, (at[0][1] + at[1][1]) / 2])
    for j in range(many):                                          # from the edges toward the two backs; the furthest first
        t = land(.52 + .025 * j)
        if t <= 0: continue
        sk, bl, k, bd = folk[j % len(folk)]; c = A(at[j % 2]); th = (j // 2 + .5) / (many / 2) * TAU + (.4 if j % 2 else -.1) + r.normal(0, .10); d = A([math.cos(th), math.sin(th)]); q = c + d * A([s * 3.05, s * 2.45]) * (1 + .10 * r.normal()); q[0] = min(max(q[0], 60), W - 60); dr = q - c; ang = math.degrees(math.atan2(-dr[0] * -1, dr[1])) if False else math.degrees(math.atan2(dr[0], dr[1])); q = q + dr / np.hypot(*dr) * ((1 - t) * 150 - 7 * push) + r.normal(0, 1, 2) * j_
        put(fin(_flat(L, q[0], q[1], s * 1.55 * k, 'R' if j % 3 else 'L', sk, bl, ang, t, bd, 30 + j), q))
    t = land(.28, .08)
    if t > 0:
        for i, (c, sd, dx) in enumerate(((at[0], 'L', -1), (at[1], 'R', 1))): q = A(c) + [dx * s * 1.62, s * .30 + (1 - t) * 190 - 8 * push]; put(fin(_flat(L, q[0], q[1], s * 1.95, sd, PI.SKIN, HD.build(len=1.06, wid=1.02, boxy=.5, tip=.2, knuckle=1.1), dx * 10., t, PI.BAND, 10 + i, True), q))
    t = land(.0, .08)
    if t > 0:
        for i, (c, sd, dx) in enumerate(((at[0], 'L', 1), (at[1], 'R', -1))): q = A(c) + [0, (1 - t) * 130 - 6 * push]; put(fin(_flat(L, q[0], q[1], s, sd, G.SKIN, Gohan.KID, -dx * 7., t, G.SASH, i, spread=1.15), q))
    return out


def hold_ball(cx, cy, r, light_='gold', f=0, tremble=0., grip=1., roll=0., seed=1, view='front'):
    """Gohan's two hands holding the ball (h2).  Draw the ball at (cx, cy, r) FIRST, then put this premultiplied RGBA over it: whatever lies behind the glass is already cut away.
    grip 1 closed round it .. 0 just opening; roll degrees turns the whole picture; tremble degrees of nervous fingers (held on twos)"""
    from db import hands as HD
    L = light(light_); R0 = .70; s = r / R0; m = lambda a, b: a + (b - a) * grip; hs = []; Z = HD.Rz(math.radians(roll))
    for q, sd in ((-1, 'R'), (1, 'L')):
        p = HD.pose((m(6, 30), m(10, 34), m(8, 22), -4), (m(8, 32), m(12, 36), m(8, 24), 0), (m(12, 34), m(16, 36), m(10, 24), 4), (m(18, 38), m(20, 38), m(12, 24), 10), thumb=(58, 14, 26, 6, 10), cup=.3)
        if tremble: p = HD.tremble(p, 31 + f // 2 + (q > 0), tremble)
        z = HD.unit((q * .62, -.70, m(-.20, -.34))); y = A([-q * .10, .40, .91]); y = HD.unit(y - z * float(y @ z)); x = np.cross(y, z); Rm = Z @ np.stack([x, y, z], -1); hs.append(HD.Hand(p, sd, Rm @ np.diag([1., 1., 1.]), tuple(Z @ (z * (R0 * .98 + m(.16, .0)) + A([0, .0, .10]))), Gohan.KID, scale=.95, pivot=(0, .58, -.16), arm=1.5))
    ball_ = HD.rod((0, 0, -.001), (0, 0, .001), R0, R0, 0., R0, 'prop', 0., 0); g = HD.render(hs, HD.Cam(float(cx), float(cy), s, 7.), light=ld3(L), res=int(np.clip(s * 2.4, 200, 640)), clip=(-160, -160, W + 160, FH + 160), props=[ball_]); cel = Cel(W, FH); pal = dict(HD.PAL['day'], **hpal(L, G.SKIN))
    g.strokes = [q for q in g.strokes if not _on_ball(q['P'], cx, cy, r)]
    for i in (0, 1):
        q = HD.only(g, i); q.groups = [G_ for G_ in q.groups if G_['name'] != 'prop']; q.core = []; HD.ink(cel, q, pal, w=lw(s * .8)[0], seed=seed + i, marks=.55)
    lay = grade(cel.layer(), L, (cx - 2 * r, cy - 2 * r, cx + 2 * r, cy + 2.5 * r), max(.6, s / 200), rims=False); x, y = grid(FH, W); d = np.hypot(x - cx, y - cy); k_ = np.exp(-np.clip(d - r, 0, 9999) / (r * .55)) * .34
    lay[..., :3] += (lay[..., 3] * k_)[..., None] * rgb('#ffb45a') * (.25 + lay[..., :3]); return lay


def _on_ball(P, cx, cy, r):
    d = np.hypot(*(np.asarray(P) - [cx, cy]).T); return len(d) > 8 and np.abs(d - r).max() < .06 * r


if __name__ == '__main__':
    ims = []
    for k in range(4):
        im = comp.finish(frame(k), 0, bloom=.34, vig=.16); ims.append(cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA)); comp.save('qa/friends/_g6_%d.jpg' % k, im)
    comp.save('qa/friends/hands.jpg', np.vstack([np.hstack(ims[:2]), np.hstack(ims[2:])]), 93)
