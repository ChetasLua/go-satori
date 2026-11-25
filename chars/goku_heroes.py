"""Hero close-ups of Goku: ui(...) for g3, rage(...) for f2, grief(...) for e4.  Each returns a 1080x1920 float RGB frame (may exceed 1 where light burns)."""
import numpy as np, math, cv2, cairo
from db import W, H as HH, comp
from db.ink import Cel, spl, resample
from db.comp import blur, screen, radial, over
from db.kit import rim as rim_, para, shifted, outline, hatch
from db.paint import rgb, f32, smooth, ramp, fbm, lum, grid
from . import goku, goku_hair as GH, goku_light as GL, goku_head as GF, goku_paint as GP
from .goku_head import GHead, getx, Face, lw, rad
from .goku_hair import mixc, Hair
from .goku_bust import Bust, Body
A = np.array
D = math.pi / 180


def bolts(cel, p, q, seed, w=2.2, col='#bfe4ff', amp=.16, n=5, forks=2):
    """one electric arc with forks"""
    r = np.random.default_rng(seed); P = A([p, q], float)
    for _ in range(n):
        m = (P[:-1] + P[1:]) / 2; d = P[1:] - P[:-1]; m = m + np.c_[-d[:, 1], d[:, 0]] * r.normal(0, amp, len(d))[:, None]; o = np.empty((len(P) + len(m), 2)); o[0::2] = P; o[1::2] = m; P = o
    cel.line(P, w, col, taper=(.10, .35), press=.5, seed=seed, tremor=0, tmin=.15)
    for _ in range(forks):
        i = int(r.integers(len(P) // 5, len(P) * 4 // 5)); d = (P[-1] - P[0]) * r.uniform(.15, .35); a = r.choice([-1, 1]) * r.uniform(.4, 1.0); d = A([d[0] * math.cos(a) - d[1] * math.sin(a), d[0] * math.sin(a) + d[1] * math.cos(a)])
        bolts(cel, P[i], P[i] + d, int(r.integers(1 << 30)), w * .6, col, amp, 3, 0)


def glow_of(layer, r, col, amt): return blur(layer[..., 3], r)[..., None] * rgb(col) * amt


# -------------------------------------------------------------------------------------------------- g3
UI_LIT = [(0, '#fff0e0'), (.14, '#fbd9bd'), (.34, '#e9b28c'), (.6, '#c9835c'), (1, '#a05a3e')]
UI_SHD = [(0, '#9a4e36'), (.48, '#84402c'), (.60, '#5c2c20'), (.76, '#3a1c18'), (.92, '#221114'), (1, '#170c10')]


def ui(cx=850, cy=528, s=398, yaw=22, pitch=-7, roll=6, byaw=-22, f=0, seed=3, opened=1., glow=1., wind=(-.10, .10), static=1., bg=None, gx=-.014, hair=None):
    h, w = HH, W; H0 = GHead().pose(cx, cy, s, yaw * D, pitch * D, roll * D)
    G, X, L = goku.setup(H0, 'ui', 'ui', 'ui'); X = dict(X, gx=gx, gy=.004, hw=.158, my=-.655, mU=[-.010, .002, .004, .002, -.006], mL=[-.010, .002, .004, .002, -.006])
    if opened < 1: X = GF.lids(X, 1 - opened); X['style'] = 'ui'; X['glow'] = 1.
    L = dict(L, brow='#34343f', line='#120b0d', hair=dict(L['hair'], hi='#ffffff', light='#e3e9f4', base='#b3bccd', shade='#666e84', deep='#2d3040', line='#0e0e16', root='#4a4f62', cap='#4c5164'))
    hr = hair or Hair(G, 'ui', L, f=f, wind=wind, seed=seed, painterly=True, detail=1.0, kids=2)
    if bg is None: bg = ui_bg(f)
    img = bg.copy()
    # ---- hair behind
    c = Cel(w, h); hr.back(c); back = c.rgba(); back = para(back, (cx + 1.5 * s, cy - .5 * s), (cx - 1.8 * s, cy), [(0, '#ffffff'), (.35, '#b8bccc'), (.7, '#5c5f74'), (1, '#33354a')]); img = over(img, back)
    # ---- body
    body = ui_body(G, L, byaw, f, seed); img = over(img, body)
    # ---- face
    fl = ui_face(G, X, L, hr, seed, opened); img = over(img, fl)
    # ---- front hair
    c = Cel(w, h); hr.front(c); fr = c.rgba(); fr = para(fr, (cx + 1.3 * s, cy - .5 * s), (cx - 1.6 * s, cy + .1 * s), [(0, '#ffffff'), (.30, '#f2f3fa'), (.52, '#a9aabd'), (.78, '#585a70'), (1, '#2c2d40')])
    img = over(img, fr)
    # rim of the whole figure
    al = np.maximum(np.maximum(back[..., 3], fr[..., 3]), np.maximum(fl[..., 3], body[..., 3]))
    sh_ = lambda a, dx, dy: cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (W, HH), borderMode=cv2.BORDER_REPLICATE)
    e = np.clip(al - sh_(al, -5, 3), 0, 1); e = cv2.GaussianBlur(e, (0, 0), .8) * al; x, y = grid(); img = img + (e * smooth(cx - .2 * s, cx + .9 * s, x))[..., None] * rgb('#dfeaff') * .9
    e = np.clip(al - sh_(al, 4, 1), 0, 1); e = cv2.GaussianBlur(e, (0, 0), 1.2) * al; img = img + (e * smooth(cx - .1 * s, cx - 1.0 * s, x))[..., None] * rgb('#4468c8') * .50
    # ---- eyes
    if opened > .2 and glow > 0:
        for q in (-1, 1):
            p = G.Pc([(q * (X['xi'] + (X['xo'] - X['xi']) * .43) + gx, -.008)])[0]; k = glow * opened * (1. if q < 0 else .75)
            img = screen(np.clip(img, 0, None), (radial(p[0], p[1], .50 * s, .34 * s, 2.6)[..., None] * rgb('#7fa4ff') * .55 + radial(p[0], p[1], .17 * s, .12 * s, 1.8)[..., None] * rgb('#e8f0ff') * .80) * k) if True else img
            st = np.exp(-np.abs(y - p[1]) / (.010 * s)) * np.exp(-np.abs(x - p[0]) / (.42 * s)); img = img + st[..., None] * rgb('#a9c6ff') * .35 * k
    # ---- static
    if static > 0:
        r = np.random.default_rng(40 + int(f // 2)); c = Cel(w, h)
        for j in range(int(9 * static)):
            b = hr.bl[int(r.integers(len(hr.bl)))]
            if b.k[0] not in 'LRTFC': continue
            i0 = int(r.integers(8, 20)); i1 = min(len(b.C) - 1, i0 + int(r.integers(6, 14))); p, q = b.C[i0] + b.N[i0] * b.hw[i0] * r.uniform(-.9, .9), b.C[i1] + b.N[i1] * b.hw[i1] * r.uniform(-1, 1)
            bolts(c, p, q, int(r.integers(1 << 30)), 2.0, '#e6f5ff', .20, 5, 2)
        a = c.rgba(); img = img + glow_of(a, 9, '#3f8cff', 1.3) + glow_of(a, 2.5, '#7cc0ff', .9); img = over(img, a)
    return img


def ui_bg(f=0):
    x, y = grid(); v = np.clip((x / W) * .9 + (1 - y / HH) * .2, 0, 1); bg = ramp(v, [(0, '#04050c'), (.4, '#0a0d20'), (.75, '#171a3c'), (1, '#2a2460')])
    # limb of the earth, low left
    d = np.sqrt((x - 420) ** 2 + (y - 3150) ** 2) - 2240; bg = bg + np.exp(-np.abs(np.minimum(d, 0)) / 46)[..., None] * rgb('#2f6cc8') * .55 * (d < 0)[..., None] + np.exp(-np.maximum(d, 0) / 70)[..., None] * rgb('#3f86ff') * .40 * (d >= 0)[..., None]
    r = np.random.default_rng(5); st = np.zeros((HH, W), f32)
    for _ in range(240): st[int(r.integers(0, HH)), int(r.integers(0, W))] = r.random() ** 3
    bg = bg + cv2.GaussianBlur(st, (0, 0), .9)[..., None] * 3.0
    # violet beams arriving out of focus
    c = Cel(W, HH, ss=1)
    for k in range(26):
        a = math.radians(176 + r.normal(0, 5)); y0 = r.uniform(-200, 1200); L = r.uniform(300, 1100); x0 = r.uniform(1150, 2300); ww = r.uniform(3, 20); dv = A([math.cos(a), math.sin(a)]); nr = A([-dv[1], dv[0]]); p = A([x0, y0]); c.fill(np.vstack([p, p + dv * L * .12 + nr * ww, p + dv * L, p + dv * L * .12 - nr * ww]), '#b06cff', r.uniform(.25, .8))
    a = c.rgba(); m = blur(a[..., 3], 10); bg = bg + m[..., None] * rgb('#9a4cff') * .75 + blur(a[..., 3], 3)[..., None] * rgb('#d9b0ff') * .22
    return bg.astype(f32)


def ui_body(G, L, byaw, f, seed):
    c = Cel(W, HH); B = Bust(c, G, L, byaw=byaw * D, bpitch=.02, wear='bare', seed=seed, tendon=.5); B.neck(); sil = c.rgba()[..., 3]; s = G.s; bd = B.B; sg = -1
    P = lambda L_, cn=None: spl(bd.P(A(L_, float)), True, cn, 8) if cn is not None else bd.P(A(L_, float))
    j, k = G.jaw(-1), G.jaw(1)
    M = [P([(-.02, -.8), (-.06, -1.30), (-.12, -1.72), (-.02, -1.90), (-.30, -1.92), (-.9, -1.80), (-1.5, -1.92), (-2.2, -2.2), (-2.2, -3.0), (-3.2, -3.4), (-3.2, -.8)]),
         np.vstack([G.P3(A([(-.62, -.40, -.2)])), bd.P([(-.6, -1.2), (-.16, -1.46), (.06, -1.52), (.16, -1.34), (.30, -1.30), (.44, -1.14), (.62, -1.02)], .02), G.P3(A([(.64, -.40, -.2)]))]),
         P([(.17, -1.30), (.23, -1.50), (.15, -1.76), (.04, -1.86), (.10, -1.60)], (0, 3)), P([(.40, -1.30), (.56, -1.44), (.66, -1.62), (.95, -1.68), (1.30, -1.70), (.95, -1.78), (.58, -1.78), (.30, -1.80), (.34, -1.58)], (0, 4, 7)),
         P([(.20, -1.94), (.60, -1.96), (1.0, -1.88), (1.46, -1.86), (1.10, -2.04), (.66, -2.16), (.22, -2.16)], (0, 3, 6)), P([(1.58, -1.94), (1.70, -2.2), (1.94, -2.7), (1.80, -2.75), (1.52, -2.4), (1.42, -2.1)], (0, 3)), P([(.06, -2.0), (.16, -2.4), (.10, -3.2), (-.2, -3.2), (-.2, -2.0)])]
    Mm = GP.mask(M, within=B.sil); M2 = GP.mask([np.vstack([j, k[::-1][1:], bd.P([(.40, -1.0), (.10, -1.16), (-.3, -1.10)], .03)]), P([(-.6, -1.0), (-.30, -1.4), (-.14, -1.8), (-.5, -1.86), (-1.2, -1.75), (-1.6, -1.5), (-.9, -1.1)])], within=B.sil)
    o = GP.tone(sil, Mm, UI_LIT, UI_SHD, .24 * s, blur(M2, 10) * .8, '#12090c', '#b4563a', .55, .03, 8)
    x, y = grid(); d = smooth(G.cy + 1.1 * s, G.cy + 2.4 * s, y)[..., None]; o = o * (1 - .55 * d)
    lay = GP.prem(o, sil)
    c = Cel(W, HH); B.c = c; B.ink = '#120b0d'; r = np.random.default_rng(12)
    for k_, e_ in ((0, B.nl), (1, B.nr)): c.line(e_, 5.2, '#140b0c', taper=(.08, .12), press=.4, seed=k_)
    for P_, w_, al_ in (([(.47, -1.02), (.36, -1.32), (.20, -1.62), (.12, -1.80)], 4.2, 1.), ([(.14, -1.82), (.50, -1.83), (1.0, -1.74), (1.45, -1.75)], 3.6, 1.), ([(.34, -1.95), (.64, -1.94), (.96, -1.87)], 2., .8), ([(-.02, -1.84), (.03, -1.92), (.08, -1.86)], 3., 1.), ([(.02, -1.30), (.08, -1.38), (.13, -1.33)], 2.2, .8), ([(.62, -1.30), (.72, -1.46), (.92, -1.58)], 2.4, .9), ([(1.56, -1.93), (1.70, -2.28), (1.92, -2.76)], 3.4, 1.)): c.line(spl(bd.P(P_), n=10), w_, '#1c0e0c', taper=(.45, .3), press=.35, seed=int(w_ * 10), alpha=al_)
    # cross-hatching at the turn of neck and clavicle
    c.clip(B.sil)
    for (x0, y0, x1, y1, n, l, dx, dy, a) in ((.14, -1.36, .05, -1.82, 22, .14, .8, .25, .55), (.42, -1.36, .34, -1.74, 18, .12, .7, -.5, .5), (.40, -1.80, 1.30, -1.74, 34, .09, .25, -1., .45), (.30, -1.20, .58, -1.06, 16, .12, .25, -1, .6), (-.02, -1.3, -.1, -1.8, 20, .16, -.8, -.2, .5), (.30, -2.12, 1.2, -2.0, 26, .10, .2, -1, .4)):
        for j_ in range(n):
            u = (j_ + r.uniform(-.3, .3)) / (n - 1); xx, yy = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u; q = l * r.uniform(.4, 1.2); c.line(spl(bd.P([(xx, yy), (xx + dx * q * .5, yy + dy * q * .5), (xx + dx * q, yy + dy * q)]), n=6), r.uniform(.9, 1.9), '#2a130f', taper=(.05, .8), seed=j_, alpha=a, tremor=.1)
    for k_ in range(60):
        xx, yy = r.uniform(.2, 2.0), r.uniform(-2.6, -1.5); q = r.uniform(.03, .10); c.line(bd.P([(xx, yy), (xx + q * .8, yy + q * .25)]), r.uniform(.7, 1.4), '#fff0e2' if r.random() < .5 else '#8a4a34', taper=(.4, .6), seed=k_, alpha=.22)
    c.unclip(); return GP.over_(lay, c.rgba())


def ui_face(G, X, L, hr, seed, opened=1.):
    s = G.s; H = G; c = Cel(W, HH); F = Face(c, G, X, L, seed, hr, 0., 0., 1.05, True, style='ui')
    F.ears([1]); ear_far = c.rgba(); sil = GP.mask([F.F]); sg = -1
    f_ = lambda P, cn=None: H.Pc(spl(A(P, float), True, cn, 6) if cn is not None else A(P, float))
    off = A([-.07, .085]) * s; casts = []
    for b in hr.bl:
        if b.k[0] in 'FC' and b.dp > hr.split: casts.append(b.P + off)
    casts += [p + off * .7 for p in hr.cap_front]
    sl = resample(F.side_line(-1, .90), .03); Q = F.Pn(sl); side = np.vstack([Q, (Q + A([-3 * s, 0]))[::-1]])
    B = A(X['B'], float); by = lambda x: np.interp(x, B[:, 0], B[:, 1])
    soc = [f_([(.055, .06), (.12, by(.12) + .02), (.28, by(.28)), (.46, by(.46)), (.60, by(.60) - .02), (.57, .04), (.50, .035), (.40, .05), (.28, .01), (.18, -.045), (.145, -.09), (.115, -.13), (.085, -.06)])]
    cheek = [f_([(.64, -.24), (.555, -.30), (.47, -.39), (.375, -.52), (.335, -.64), (.35, -.72), (.40, -.62), (.46, -.52), (.55, -.42), (.63, -.36)], (0, 5)), f_([(.30, -.92), (.46, -.75), (.56, -.60), (.64, -.46), (.8, -.4), (.8, -1.1), (.0, -1.1), (.10, -.985), (.22, -.97)])]
    nose = [F.N([(.015, -.425, .95), (.05, -.430, .55), (.086, -.440, .15), (.075, -.462, .10), (.05, -.500, .0), (-.05, -.50, 0), (-.06, -.44, .3)]), F.N([(.060, -.33, .30), (.088, -.39, .12), (.084, -.435, .16), (.066, -.40, .3)]), f_([(.04, .03), (.085, -.02), (.11, -.10), (.075, -.14), (.055, -.06)])]
    lips = [H.Pm(A([(-.15, -.652), (0, -.640), (.12, -.650), (.135, -.658), (.04, -.672), (-.1, -.672)])), H.Pm(A([(-.085, -.735), (.0, -.722), (.085, -.735), (.055, -.772), (-.02, -.785), (-.06, -.77)]))]
    temple = [f_([(.52, .60), (.54, .40), (.575, .28), (.63, .22), (.8, .2), (.8, .7)])]
    M = GP.mask([side] + casts + soc + cheek + nose + lips + temple, within=F.F)
    # accents of light inside the dark half: cheekbone, wing of the nose, lower lid
    acc = [(f_([(-.22, -.160), (-.36, -.185), (-.52, -.165), (-.60, -.20), (-.50, -.235), (-.36, -.235)], (0, 3)), .42), (F.N([(-.020, -.10, .9), (-.028, -.25, .9), (-.034, -.36, .9), (-.050, -.36, .55), (-.046, -.24, .5)]), .35), (H.Pm(A([(-.16, -.70), (-.05, -.695), (-.02, -.715), (-.12, -.72)])), .35), (f_([(-.16, -.90), (-.02, -.93), (-.03, -.965), (-.20, -.945)]), .35)]
    Ac = blur(GP.mask(acc, within=F.F), 3.0); M = np.clip(M - Ac * .55 * (M > .5), 0, 1)
    deep = [f_([(-.07, .05), (-.12, by(.12) + .01), (-.30, by(.30)), (-.50, by(.50) - .01), (-.58, .05), (-.52, .02), (-.40, .055), (-.28, .015), (-.18, -.035), (-.15, -.10), (-.11, -.13), (-.085, -.05)]), f_([(-.80, -.06), (-.66, -.18), (-.54, -.30), (-.45, -.44), (-.48, -.50), (-.585, -.38), (-.66, -.32), (-.80, -.26)]),
            f_([(-.80, -.34), (-.62, -.54), (-.46, -.76), (-.30, -.93), (-.2, -1.05), (-.80, -1.05)]), f_([(.075, .05), (.12, by(.12)), (.26, by(.26) - .01), (.38, by(.38) - .03), (.28, .02), (.18, -.035), (.14, -.085), (.105, -.11), (.085, -.02)]), F.N([(-.07, -.436, .3), (.0, -.430, .9), (.065, -.438, .3), (.04, -.458, .2), (-.04, -.458, .2)])]
    M2 = blur(GP.mask(deep, within=F.F), 2.2) * .75
    o = GP.tone(sil, M, UI_LIT, UI_SHD, .155 * s, M2, '#150a0c', '#c0603f', .60, .03, 4)
    # cool bounce from the left edge, warm bounce under the jaw
    x, y = grid(); e = np.clip(sil - shifted(blur(sil, 22), 46, 6), 0, 1) * smooth(G.cx, G.cx - .6 * s, x); o = o + (blur(e, 8) * .30)[..., None] * rgb('#3f62b8')
    # grain of the skin
    o = o * (1 + .020 * fbm(HH, W, (3, 1.6), (1, .7), 21)[..., None]) * (1 + .016 * fbm(HH, W, (150, 60), (1, .6), 22)[..., None])
    lay = GP.prem(np.clip(o, 0, 2), sil); lay = GP.over_(ear_far, lay)
    c = Cel(W, HH); F.c = c; F.ears([-1]); ea = c.rgba(); ea[..., :3] *= rgb('#6a5350') * (.55 + .45 * smooth(G.cx - .95 * s, G.cx - .70 * s, x))[..., None]; lay = GP.over_(lay, ea)
    # ---- pen
    c = Cel(W, HH); F.c = c
    c.clip(F.F); dk = '#1c0c0a'
    hk = '#6d3222'
    GP.cheek_hatch(F, [(.60, -.28, .40, -.52, 20, .13, -.55, .55), (.40, -.52, .34, -.72, 14, .09, -.8, .25), (.20, -.145, .50, -.085, 24, .085, .45, -.9), (.53, .44, .58, .24, 14, .12, -.85, -.35)], hk, 1.5, 7, .8)
    GP.cheek_hatch(F, [(-.20, -.15, -.52, -.12, 24, .10, -.35, -.95), (-.42, -.36, -.30, -.60, 18, .12, .5, -.75)], '#1a0c0a', 1.5, 8, .8)
    GP.cheek_hatch(F, [(.30, -.30, .44, -.24, 9, .05, .8, .5), (.12, -.90, .24, -.84, 8, .05, .7, .5)], '#b2644a', 1.1, 11, .6); GP.cheek_hatch(F, [(.38, -.17, .56, -.19, 12, .05, .9, -.2), (.10, -.44, .12, -.40, 4, .03, .6, .8)], '#fff6ea', 1.2, 12, .7)
    for P_, w_ in (([(.30, -.25), (.38, -.29), (.47, -.33)], 2.4), ([(.36, -.235), (.42, -.262)], 1.5)): c.line(spl(H.Pc(P_), n=8), w_, '#7c2a22', taper=(.4, .5), seed=3)   # the cheek scratch, healed thin
    c.unclip(); F.contour(); ui_features(F, opened)
    return GP.over_(lay, c.rgba())


def ui_features(F, opened):
    """eyes, brows, nose and mouth at full size"""
    H, X, c, s = F.H, F.X, F.c, F.u
    c.clip(F.F)
    for q in (-1, 1): (ui_eye(F, q) if opened > .12 else F.eye(q))
    F.mouth(); c.unclip(); F.nose()
    for q in (-1, 1):
        F.brow(q); P = F.brow_poly(q); n = (len(P) - 1) // 2; r = np.random.default_rng(31 + q); up, dn = P[1:n + 1], P[n + 1:][::-1]; c.clip(H.Pc(P))
        for j in range(46):
            i = int(r.uniform(0, n * .9)); k = r.uniform(0, 1); b = up[i] * k + dn[min(i, len(dn) - 1)] * (1 - k); i2 = min(n - 1, i + int(n * r.uniform(.10, .22))); e = up[i2] * min(1, k + .25) + dn[min(i2, len(dn) - 1)] * (1 - min(1, k + .25))
            c.line(H.Pc([b, (b + e) / 2 + A([0, .004]), e]), r.uniform(.9, 2.0), ['#8f96aa', '#5a5e70', '#14141c', '#c9d2e2'][int(r.integers(4))] if q > 0 else ['#3c3e4c', '#16161e', '#5a5f74'][int(r.integers(3))], taper=(.2, .7), seed=j, alpha=.85)
        c.unclip(); outline(c, H.Pc(P), 2.2, '#0e0a0e', seed=60 + q)
    F.furrow(); F.lines()


def ui_eye(F, q):
    """the opened eye: dark water round a ring of light"""
    c, H, L, X, s = F.c, F.H, F.L, F.X, F.u; T5 = GF.T5; xs = X['xi'] + (X['xo'] - X['xi']) * T5; up = GF.crv(xs, A(X['U'], float), straight=.55); lo = GF.crv(xs - A([0, .012, .010, -.004, -.014]), A(X['Lo'], float), straight=.5); n = len(up); t = np.linspace(0, 1, n); Mx = lambda P: np.c_[q * np.asarray(P)[:, 0], np.asarray(P)[:, 1]]
    Pu, Pl = H.Pc(Mx(up)), H.Pc(Mx(lo)); poly = np.vstack([Pu, Pl[::-1]]); dark = q < 0
    top, bot = Pu.mean(0), Pl.mean(0); c.grad(poly, top, bot, [(0, '#1f2030' if dark else '#4a4c5e', 1), (.35, '#4d5068' if dark else '#a6a8bc', 1), (1, '#9a9fb8' if dark else '#f2f4fa', 1)])
    c.clip([poly]); ix = X['xi'] + (X['xo'] - X['xi']) * .43 + q * 0 + X['gx'] * q * q if False else q * (X['xi'] + (X['xo'] - X['xi']) * .43) + X['gx']; iy = -.014 + X['gy']; rx = X['iris'] * 1.10; ry = rx * X['irish']; a = np.linspace(0, 2 * np.pi, 61)[:-1]
    E = lambda r1, r2, dx=0, dy=0: H.Pc(np.c_[ix + dx + np.cos(a) * r1, iy + dy + np.sin(a) * r2])
    c.fill(E(rx * 1.10, ry * 1.08), '#151824'); c.fill(E(rx * .99, ry * .98), '#59627a'); c.fill(E(rx * .92, ry * .92), '#93a0ba'); r = np.random.default_rng(77 + q)
    for j in range(64):
        th = j / 64 * 6.283 + r.normal(0, .03); r0, r1 = r.uniform(.40, .55), r.uniform(.74, .95); c.line(H.Pc([(ix + math.cos(th) * rx * r0, iy + math.sin(th) * ry * r0), (ix + math.cos(th + .04) * rx * r1, iy + math.sin(th + .04) * ry * r1)]), r.uniform(.9, 2.2), '#ffffff' if j % 3 else '#3c4660', taper=(.1, .5), seed=j, alpha=.85)
    rad(c, None, H.Pc([(ix, iy)])[0], rx * .95 * s, '#ffffff', .95); c.fill(E(rx * .40, ry * .42), '#eef3ff'); c.fill(E(rx * .22, ry * .24), '#aab6d0'); c.fill(E(rx * .15, ry * .17), '#5d6880')
    th = GF.prof(t, [(0, .012), (.3, .030), (.6, .040), (.9, .034), (1, .016)]); c.fill(H.Pc(Mx(np.vstack([up, np.c_[up[:, 0], up[:, 1] - th][::-1]]))), '#0f1018', .80); c.fill(H.Pc(Mx(np.vstack([up, np.c_[up[:, 0], up[:, 1] - th * 1.8][::-1]]))), '#1a1c2c', .35)
    gx0, gy0 = ix + rx * .34, iy + ry * .30; c.fill(H.Pc(spl([(gx0 - .020, gy0 - .006), (gx0 - .003, gy0 + .020), (gx0 + .020, gy0 + .008), (gx0 + .008, gy0 - .018)], True, n=5)), '#ffffff'); c.fill(E(rx * .10, ry * .07, -rx * .42, -ry * .50), '#ffffff', .9)
    c.unclip()
    # wet lower rim + lashes of light
    c.line(H.Pc(Mx(lo[int(n * .2):int(n * .8)] + A([0, .006]))), 1.8, '#dfe8ff', taper=(.4, .4), alpha=.8)
    ex, ey = up[-1]; th = GF.prof(t, [(0, .008), (.12, .022), (.36, .030), (.62, .038), (.82, .046), (.95, .046), (1, .036)]); Bk = np.c_[up[:, 0], up[:, 1] + th]; o = X['out']; yl = lo[-1, 1]
    lid = np.vstack([Mx(Bk), Mx(A([(ex + .024, ey + th[-1] - .004), (ex + .034, ey - o * .10), (ex + .026, ey - o * .75), (ex + .006, max(yl + .012, ey - o * 1.35)), (ex + .004, ey - o * .6), (ex - .006, ey - .006)])), Mx(up)[::-1]]); c.fill(H.Pc(lid), '#0c080a')
    m = len(lo); F.ln(H.Pc([(q * (up[0, 0] - .002), up[0, 1] + .004), (q * (up[0, 0] - .014), (up[0, 1] + lo[0, 1]) / 2), (q * (lo[2, 0] + .004), lo[2, 1] + .006)]), .55, seed=43 + q, taper=(.2, .5)); F.ln(H.Pc(Mx(lo[int(m * .12):int(m * .62)])), .60, seed=45 + q, taper=(.6, .5)); F.ln(H.Pc(Mx(lo[int(m * .64):])), .80, seed=46 + q, taper=(.4, .15))
    for k, (d, a0, a1, w_) in enumerate(((.034, .05, .55, .42), (.058, .12, .46, .30), (.085, .20, .40, .24))): P = np.c_[lo[:, 0] - .01, lo[:, 1] - d - .012 * np.sin(np.pi * t)]; F.ln(H.Pc(Mx(P[int(a0 * n):int(a1 * n)])), w_, seed=48 + q + k, taper=(.3, .6), col='#2a120e', alpha=.8)


# -------------------------------------------------------------------------------------------------- f2
GOLD_LIT = [(0, '#fff0c4'), (.10, '#ffe0a2'), (.32, '#f8c67c'), (.62, '#e6a258'), (1, '#c97c3e')]
GOLD_SHD = [(0, '#d9822f'), (.48, '#bd6226'), (.60, '#94431a'), (.78, '#642912'), (.92, '#41190c'), (1, '#2a0f08')]


def flame(c, cx, cy, rx, ry, n, L, col, seed, f=0, up=.75, a0=-205, a1=25, edge=None, ew=3.):
    """hard-edged aura: a ring of S-curved tongues leaning up"""
    r = np.random.default_rng(seed); polys = []; a = np.radians(np.linspace(a0, a1, n) + r.normal(0, 3.5, n))
    core = np.c_[cx + np.cos(np.linspace(0, 6.283, 80)) * rx * 1.02, cy + np.sin(np.linspace(0, 6.283, 80)) * ry * 1.02]
    for k, t in enumerate(a):
        b = A([cx + math.cos(t) * rx * .86, cy + math.sin(t) * ry * .86]); o = A([math.cos(t), math.sin(t)]); d = o * (1 - up) + A([0, -1.]) * up; d /= np.hypot(*d); q = L * r.uniform(.45, 1.15) * (1 + .12 * math.sin(f * 1.7 + k * 2.1)); nr = A([-d[1], d[0]]); w = L * r.uniform(.16, .30); sw = q * r.normal(0, .14) + q * .06 * math.sin(f * 1.1 + k)
        tip = b + d * q + nr * sw; c1 = b + o * q * .30; u = np.linspace(0, 1, 22)[:, None]; C = (1 - u) ** 3 * b + 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u * u * (tip - d * q * .34 - nr * sw * .9) + u ** 3 * tip; T = np.gradient(C, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]; wd = (w * (1 - u[:, 0] ** 1.5) * (.75 + .35 * np.sin(u[:, 0] * 3.0)))[:, None]; polys.append(np.vstack([C + N * wd, (C - N * wd)[::-1]]))
    if edge is not None:
        for p in polys: outline(c, p, ew * 2, edge, seed=seed)
        outline(c, core, ew * 2, edge, seed=seed)
    for p in polys: c.fill(p, col)
    c.fill(core, col); return polys


def rage_bg(f=0, cx=960, cy=640, power=1.):
    x, y = grid(); n = fbm(HH, W, (260, 110, 46, 18), (1, .6, .35, .18), 31 + f // 40); v = np.clip(.5 + .22 * n - .35 * (y / HH - .4), 0, 1)
    bg = ramp(v, [(0, '#050a09'), (.45, '#0c1a16'), (.75, '#1f3327'), (1, '#4a5a34')]).astype(f32); d = np.sqrt(((x - cx) / 1150) ** 2 + ((y - cy) / 760) ** 2); bg = bg + (np.clip(1 - d, 0, 1) ** 1.3)[..., None] * rgb('#f09a1e') * .55 * power
    c = Cel(W, HH); r = np.random.default_rng(60 + int(f) // 2)
    for k in range(90):                                          # streaks driven outward by the burst
        t = r.uniform(0, 6.283); q0 = r.uniform(420, 760); q1 = q0 + r.uniform(200, 900); ww = r.uniform(1, 9) * r.random(); o = A([math.cos(t), math.sin(t) * .8]); nr = A([-o[1], o[0]]); c.fill(np.vstack([A([cx, cy]) + o * q0, A([cx, cy]) + o * (q0 + (q1 - q0) * .7) + nr * ww, A([cx, cy]) + o * q1, A([cx, cy]) + o * (q0 + (q1 - q0) * .7) - nr * ww]), '#ffd76a', r.uniform(.15, .6))
    a = c.rgba(); bg = over(bg, a) + glow_of(a, 6, '#ff9a2a', .25)
    c = Cel(W, HH)
    for k in range(4): bolts(c, (r.uniform(0, W), -20), (r.uniform(0, W), r.uniform(300, 700)), 700 + k + int(f) // 3, 3.0, '#f4ffe6', .16, 6, 3)
    a = c.rgba(); bg = bg + glow_of(a, 14, '#8ad86a', .8) + glow_of(a, 3, '#d8ffc0', .5); bg = over(bg, a); return bg


def rage(cx=985, cy=520, s=306, yaw=-12, pitch=1, roll=-8, byaw=14, f=0, seed=5, expr='scream', hair='ssj', power=1., wind=(.0, .25), eyes='ssj', tears=.8, steam=1., bg=None, light=None, aura=1.):
    """gold break.  hair 'base'|'s1'..'s5'|'ssj'; power 0..1 = how gold the light is"""
    H0 = GHead().pose(cx, cy, s, yaw * D, pitch * D, roll * D); G, X, L = goku.setup(H0, expr, 'gold', hair, 'gold', eyes); X = dict(X, tears=0.)
    L = dict(L, line='#2a0f06', brow='#b36c14', hair=dict(L['hair'], hi='#fffbd8', light='#ffe670', base='#ffc928', shade='#e08a14', deep='#a2500c', line='#4a1e06', root='#b8640e', cap='#b05a0e'), side=1, amt=-.15, ldir=(.35, .55, .75), hdir=(.40, .35, .85), socket=.6)
    hr = Hair(G, hair, L, f=f, wind=wind, seed=seed, painterly=True, kids=1, detail=.42, flat=True); x, y = grid()
    img = (rage_bg(f, cx, cy + 120, power) if bg is None else bg).copy()
    if aura > 0: img = rage_aura(img, cx, cy + 150, f, aura)
    c = Cel(W, HH); hr.back(c); back = c.rgba(); back = para(back, (cx, cy - 2.4 * s), (cx, cy + .4 * s), [(0, '#fff4d8'), (.5, '#e8b45c'), (1, '#9c5618')]); img = over(img, back)
    body = rage_body(G, L, X, byaw, f, seed); img = over(img, body)
    fl = rage_face(G, X, L, hr, seed, f); img = over(img, fl)
    c = Cel(W, HH); hr.front(c); fr = c.rgba(); fr = para(fr, (cx, cy - 2.2 * s), (cx, cy), [(0, '#ffffff'), (.6, '#fff4dc'), (1, '#f2cc8c')]); img = over(img, fr)
    al = np.maximum(np.maximum(back[..., 3], fr[..., 3]), np.maximum(fl[..., 3], body[..., 3])); sh_ = lambda a, dx, dy: cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (W, HH), borderMode=cv2.BORDER_REPLICATE)
    for dx, dy, k in ((-5, 3, 1.), (5, 3, .9)): e = cv2.GaussianBlur(np.clip(al - sh_(al, dx, dy), 0, 1), (0, 0), .8) * al; img = img + e[..., None] * rgb('#fff2b8') * .85 * k
    # light of the aura eats a little of the edges, never the face
    e = np.clip(blur(1 - al, 26) * 1.6 - .25, 0, 1) * al; img = img + e[..., None] * rgb('#ffb23a') * .28
    if eyes == 'ssj' and X['blank'] < .5:
        for q in (-1, 1): p = G.Pc([(q * (X['xi'] + (X['xo'] - X['xi']) * .43), -.01)])[0]; img = img + radial(p[0], p[1], .20 * s, .14 * s, 2.)[..., None] * rgb('#3fffe0') * .30
    # tears leave as steam; sparks; bits of light
    c = Cel(W, HH); r = np.random.default_rng(90 + int(f) // 2)
    for q in (-1, 1):
        for j in range(int(7 * steam)):
            p = G.Pc([(q * r.uniform(.22, .50), -.14 - .05 * r.random())])[0]; tip = p + A([q * r.uniform(20, 120), -r.uniform(120, 360)]); m = (p + tip) / 2 + A([q * r.normal(20, 24), 0]); P = spl([p, p * .6 + m * .4 + A([r.normal(0, 14), 0]), m, tip * .7 + m * .3 + A([r.normal(0, 22), 0]), tip], n=10); c.line(P, r.uniform(2, 9), '#fffaf0', taper=(.10, .8), press=.6, seed=int(r.integers(1 << 30)), alpha=r.uniform(.10, .35))
    a = c.rgba(); img = over(img, np.dstack([blur(a[..., k], 2.2) for k in range(4)]))
    c = Cel(W, HH)
    for j in range(70):
        t = r.uniform(0, 6.283); q = r.uniform(250, 1000); p = A([cx + math.cos(t) * q, cy + 200 + math.sin(t) * q * .8]); d = A([r.normal(0, 4), -r.uniform(10, 46)]); c.line(A([p, p + d * .5 + A([r.normal(0, 3), 0]), p + d]), r.uniform(1.2, 4.5), '#fff6c8', taper=(.3, .5), seed=j)
    a = c.rgba(); img = over(img + glow_of(a, 5, '#ffb030', .8), a)
    c = Cel(W, HH)
    for j in range(3): b = hr.bl[int(r.integers(len(hr.bl)))]; bolts(c, b.C[6], b.C[-1] + r.normal(0, 60, 2), int(r.integers(1 << 30)), 2.2, '#fffbe0', .18, 5, 2)
    a = c.rgba(); img = over(img + glow_of(a, 8, '#ffe06a', .9), a)
    return img


def rage_aura(img, cx, cy, f=0, k=1.):
    """layers of hard flame, brighter toward the feet, each with a hot drawn edge"""
    x, y = grid()
    for j, (dx, dy, rx, ry, n, L, cols, ed, ew, up) in enumerate(((-10, 340, 840, 760, 46, 660, ('#6e1e04', '#b4440a', '#d8660e'), '#ff9a34', 3.2, .62), (14, 350, 690, 700, 42, 540, ('#a23a06', '#e07412', '#f59a1c'), '#ffc65a', 2.8, .68), (-20, 370, 560, 620, 36, 420, ('#d4660a', '#f59a18', '#ffc23c'), '#ffe692', 2.6, .74), (10, 400, 420, 520, 28, 320, ('#ee8c12', '#ffc02c', '#ffe27a'), '#fffbe0', 2.4, .80))):
        c = Cel(W, HH); flame(c, cx + dx, cy + dy, rx, ry, n, L * (.65 + .35 * k), '#ffffff', 11 + j * 3, f, up, -196, 16); m = c.rgba()[..., 3]
        e = np.clip(m - cv2.erode(m, np.ones((3, 3), np.uint8), iterations=int(ew)), 0, 1); e = cv2.GaussianBlur(e, (0, 0), .7); wv = np.clip(.5 + .5 * fbm(HH, W, (110, 50), (1, .5), 70 + j), .05, 1.2)
        v = np.clip((y - (cy - 560)) / 1150 + .10 * np.abs(x - cx) / 900, 0, 1); col = ramp(v, [(0, cols[0]), (.55, cols[1]), (1, cols[2])])
        # the old drawn heat: long wedges cut into the flame, following it up
        c = Cel(W, HH); r = np.random.default_rng(200 + j)
        for q in range(34):
            t = r.uniform(-3.3, .15); p = A([cx + dx + math.cos(t) * rx * r.uniform(.80, 1.0), cy + dy + math.sin(t) * ry * r.uniform(.80, 1.0)]); d = A([math.cos(t) * (1 - up), math.sin(t) * (1 - up) - up]); d /= np.hypot(*d); ln = L * r.uniform(.35, .85); nr = A([-d[1], d[0]]); w = r.uniform(3, 11); sw = r.normal(0, .05) * ln
            c.fill(spl([p, p + d * ln * .45 + nr * (w + sw), p + d * ln + nr * sw * 2, p + d * ln * .45 - nr * (w - sw)], True, (0, 2), 8), cols[0] if r.random() < .6 else ed, .55)
        in_ = c.rgba() * m[..., None]; lay = np.concatenate([col * m[..., None], m[..., None]], -1) * .94; lay = GP.over_(lay, in_); img = over(img, lay) + (e * wv)[..., None] * rgb(ed) * .85 + blur(e, 8)[..., None] * rgb(ed) * .28
    return img


def rage_body(G, L, X, byaw, f, seed):
    c = Cel(W, HH); B = Bust(c, G, L, byaw=byaw * D, bpitch=-.04, raise_=.16, wear='torn', seed=seed, tendon=1., dirt=1.); B.neck(); sil = c.rgba()[..., 3]; s = G.s; bd = B.B
    P = lambda L_, cn=None: spl(bd.P(A(L_, float)), True, cn, 8) if cn is not None else bd.P(A(L_, float)); j, k = G.jaw(-1), G.jaw(1)
    M = [np.vstack([G.P3(A([(-.62, -.40, -.2)])), bd.P([(-.62, -1.0), (-.44, -1.22), (-.30, -1.10), (-.10, -1.34), (.06, -1.20), (.22, -1.36), (.42, -1.12), (.62, -1.0)], .02), G.P3(A([(.64, -.40, -.2)]))])]
    for q in (-1, 1): M += [P([(q * .46, -1.0), (q * .34, -1.4), (q * .16, -1.76), (q * .26, -1.78), (q * .46, -1.5), (q * .56, -1.2)], (0, 2)), P([(q * .20, -1.2), (q * .24, -1.5), (q * .12, -1.74), (q * .14, -1.4)], (0, 2)), P([(q * .62, -1.28), (q * .9, -1.5), (q * 1.2, -1.62), (q * .8, -1.7), (q * .56, -1.6)], (0, 2)), P([(q * .18, -1.88), (q * .6, -1.9), (q * 1.0, -1.84), (q * .6, -2.06), (q * .2, -2.1)], (0, 2))]
    M += [P([(2.0, -2.2), (2.2, -2.6), (2.0, -3.2), (2.3, -4.), (1.6, -4.), (1.5, -2.2)]), P([(-.1, -1.86), (0, -1.78), (.1, -1.86), (0, -2.04)])]
    Mm = GP.mask(M, within=B.sil); M2 = GP.mask([np.vstack([j, k[::-1][1:], bd.P([(.40, -1.0), (.0, -1.06), (-.40, -1.0)], .03)])], within=B.sil)
    o = GP.tone(sil, Mm, GOLD_LIT, GOLD_SHD, .16 * s, blur(M2, 8) * .7, '#4a1c0a', '#ff7a2a', .5, .02, 8); lay = GP.prem(o, sil)
    c = Cel(W, HH); B.c = c
    for k_, e_ in ((0, B.nl), (1, B.nr)): c.line(e_, 5., '#2a0f06', taper=(.08, .12), press=.4, seed=k_)
    B.neck_lines(True)
    for q in (-1, 1):                                            # tendons and a vein
        for P_, w_ in (([(.50, -.98), (.40, -1.30), (.22, -1.62), (.12, -1.80)], 4.6), ([(.30, -1.02), (.26, -1.34), (.16, -1.66)], 3.0), ([(.40, -1.10), (.44, -1.25), (.36, -1.36), (.40, -1.5)], 2.0)): c.line(spl(bd.P([(q * a, b) for a, b in P_]), n=10), w_, '#4a1c08', taper=(.4, .35), press=.35, seed=int(w_ * 9))
    B.shirt(); B.gi(); B.arm_bare(); a = c.rgba(); x, y = grid(); a = para(a, (G.cx, G.cy + .8 * s), (G.cx, G.cy + 2.4 * s), [(0, '#ffffff'), (.5, '#f0cfa0'), (1, '#a86a3a')], 'mul', .8)
    return GP.over_(lay, a)


def rage_face(G, X, L, hr, seed, f):
    s = G.s; H = G; c = Cel(W, HH); F = Face(c, G, X, L, seed, hr, .01, 0., 1.22, True); F.ears([-1, 1]); ear = c.rgba(); sil = GP.mask([F.F])
    f_ = lambda P, cn=None: H.Pc(spl(A(P, float), True, cn, 6) if cn is not None else A(P, float)); B = A(X['B'], float); by = lambda x: np.interp(x, B[:, 0], B[:, 1]); hw, my = X['hw'], X['my']
    off = A([-.03, .060]) * s; casts = [b.P + off for b in hr.bl if b.k[0] == 'F' and b.dp > hr.split and (b.P[:, 1].max() > H.P([(0, .5)])[0][1])] + [p + off * .9 for p in hr.cap_front]
    M, M2 = list(casts), []
    for q in (-1, 1):
        sh = q < 0
        M.append(f_([(q * a, b) for a, b in [(.045, by(.06) + .03), (.14, by(.14)), (.30, by(.30)), (.46, by(.46)), (.60, by(.6) - .03), (.565, .09), (.46, .10), (.36, .065), (.26, .02), (.18, -.02), (.145, -.085), (.12, -.13), (.085, -.06)]]))
        M.append(f_([(q * a, b) for a, b in [(.70, -.10), (.60, -.22), (.50, -.30), (.42, -.42), (.40, -.60), (.44, -.80), (.50, -.62), (.56, -.50), (.64, -.42), (.76, -.36)]], (0, 5)))
        M.append(H.Pm(A([(q * a, b) for a, b in [(.105, -.40), (.18, -.44), (.28, -.50), (hw + .05, my - .02), (hw + .03, my - .20), (hw + .075, my - .02), (.36, -.50), (.24, -.42)]])))
        M.append(f_([(q * a, b) for a, b in [(.50, .62), (.545, .44), (.60, .28), (.655, .16), (.625, .04), (.66, -.06), (.625, -.16), (.585, -.24), (.50, -.30), (.42, -.42), (.385, -.58), (.40, -.74), (.46, -.62), (.52, -.50), (.60, -.44), (.60, -.62), (.48, -.80), (.28, -1.0), (.9, -1.1), (.9, .6)]]))
        M.append(F.N([(q * .022, -.10, .8), (q * .030, -.26, .8), (q * .048, -.36, .6), (q * .090, -.40, .15), (q * .078, -.32, .2), (q * .052, -.20, .5)]) if sh else F.N([(q * .05, -.34, .5), (q * .09, -.40, .15), (q * .08, -.33, .2)]))
        M2.append(f_([(q * a, b) for a, b in [(.07, .02), (.12, by(.12) - .0), (.20, by(.2) - .01), (.19, .0), (.15, -.06), (.115, -.10), (.09, -.04)]]))
        M2.append(f_([(q * a, b) for a, b in [(.72, -.20), (.60, -.30), (.52, -.42), (.50, -.56), (.58, -.46), (.70, -.38)]]))
    M += [F.N([(-.082, -.440, .2), (-.04, -.426, .8), (.04, -.426, .8), (.082, -.440, .2), (.06, -.47, .1), (0, -.50, .0), (-.06, -.47, .1)]), f_([(-.05, .16), (0, .20), (.05, .16), (.028, .02), (.02, -.06), (-.02, -.06), (-.028, .02)]), H.Pm(A([(-.15, my - .50), (0, my - .485), (.15, my - .50), (.09, my - .55), (0, my - .56), (-.09, my - .55)]))]
    M.append(f_([(-.30, -.99), (-.44, -.80), (-.50, -.66), (-.8, -.6), (-.8, -1.2), (0, -1.2), (-.12, -1.0)]))
    Mm = GP.mask(M, within=F.F); D2 = blur(GP.mask(M2, within=F.F), 2.) * .75
    M2b = blur(GP.mask([f_([(q * a, b) for a, b in [(.75, .3), (.70, .1), (.72, -.1), (.66, -.26), (.60, -.42), (.56, -.62), (.44, -.82), (.30, -.98), (.9, -1.1), (.9, .3)]]) for q in (-1, 1)], within=F.F), 5) * .55; o = GP.tone(sil, Mm, GOLD_LIT, GOLD_SHD, .12 * s, np.maximum(D2, M2b), '#5e240e', '#ff6a1c', .60, .02, 4); x, y = grid(); o = o + (blur(np.clip(sil - shifted(sil, -40, 10), 0, 1), 16) * .30 + blur(np.clip(sil - shifted(sil, 40, 10), 0, 1), 16) * .22)[..., None] * rgb('#ffcf5a'); o = o * (1 + .014 * fbm(HH, W, (3, 1.6), (1, .7), 21)[..., None])
    lay = GP.over_(ear, GP.prem(np.clip(o, 0, 2), sil)); c = Cel(W, HH); F.c = c; c.clip(F.F); hk = '#a24a18'
    for q in (-1, 1):
        GP.cheek_hatch(F, [(q * .62, -.22, q * .42, -.44, 20, .12, q * -.5, .5), (q * .20, -.15, q * .48, -.12, 22, .09, q * .5, -.8), (q * .14, -.10, q * .10, -.02, 8, .06, q * .7, -.6), (q * .50, .40, q * .60, .24, 10, .09, q * -.8, -.4)], hk, 1.6, 7 + q, .85)
        GP.cheek_hatch(F, [(q * .08, .28, q * .22, .36, 10, .08, q * .3, .9)], hk, 1.3, 17 + q, .7)
    for q in (-1, 1): GP.cheek_hatch(F, [(q * .30, -.24, q * .44, -.20, 10, .05, q * .8, .5)], '#fffdf0', 1.4, 27 + q, .7)
    c.unclip(); F.contour(); F.features()
    for q in (-1, 1):                                            # the lid pressed by the brow: one heavy black line under it
        P = F.brow_poly(q); n = (len(P) - 1) // 2; dn = P[n + 1:][::-1]; F.ln(H.Pc(dn[int(n * .04):int(n * .86)] - A([0, .004])), 1.15, seed=150 + q, taper=(.10, .30)); outline(c, H.Pc(P), 2.6, '#4a1e06', seed=60 + q)
        if H.vis(q * .40, -.28): F.ln(H.Pc([(q * .30, -.222), (q * .40, -.262), (q * .50, -.318)]), .36, seed=160 + q, col='#8a2a1c', taper=(.4, .6)) if q < 0 else None
    # what is left of the tears
    for q in (-1, 1): F.stream([(q * .30, -.125), (q * .315, -.20), (q * .345, -.27), (q * .39, -.32)], .013, 3 + q, .55, '#fff6e0')
    return GP.over_(lay, c.rgba())


# -------------------------------------------------------------------------------------------------- e4
RAIN_LIT = [(0, '#e4ddd6'), (.14, '#cfc4bb'), (.36, '#b3a59d'), (.64, '#988a86'), (1, '#837878')]
RAIN_SHD = [(0, '#7f7476'), (.48, '#6d6468'), (.62, '#534d55'), (.80, '#3a373f'), (1, '#232229')]
STUCK = tuple(GH._b(th, y, tip, d, w, out=.12, k1=.36, k2=.40, pw=1.5, belly=.02, thick=.30, bias=.66, cut=.12, r0=.5, inset=-.03) for th, y, tip, d, w in (
    (-34, .60, (-.50, -.10, .80), (.10, -1, -.05), .20), (-20, .62, (-.27, -.18, .89), (-.06, -1, -.05), .17), (-12, .62, (-.13, .04, .95), (.10, -1, 0), .11), (-3, .64, (-.02, -.22, .93), (-.10, -1, -.05), .16),
    (6, .64, (.11, .02, .93), (-.12, -1, 0), .10), (14, .62, (.23, -.16, .91), (.06, -1, -.05), .17), (24, .60, (.40, -.02, .83), (-.04, -1, 0), .13), (36, .56, (.54, -.20, .72), (-.06, -1, -.05), .18)))


def grief_bg(f=0):
    x, y = grid(); n = fbm(HH, W, (340, 150, 60, 22), (1, .55, .3, .15), 12); bg = ramp(np.clip(.52 - .34 * (y / HH) + .10 * n + .12 * (x / W - .5), 0, 1), [(0, '#2e3034'), (.35, '#55585b'), (.65, '#7f8283'), (1, '#b2b5b4')]).astype(f32)
    # far cliff edge and sea, out of focus
    c = Cel(W, HH, ss=1); r = np.random.default_rng(4); P = [(-50, 760)]
    for k in range(26): P.append((-50 + k * 46, 740 + 60 * math.sin(k * .7) + r.normal(0, 24) + k * 6))
    P += [(1150, 1180), (-50, 1180)]; c.fill(A(P, float), '#2b2c30'); a = c.rgba(); a = np.dstack([blur(a[..., q], 16) for q in range(4)]); return over(bg, a).astype(f32)


def rain_layer(f, n, L, w, al, seed, ang=.10, col='#e8eef2', speed=90):
    c = Cel(W, HH); r = np.random.default_rng(seed)
    for k in range(n):
        x0, y0, v = r.uniform(-200, W + 200), r.uniform(0, HH + L), r.uniform(.8, 1.2); yy = (y0 + f * speed * v) % (HH + 2 * L) - L; ln = L * r.uniform(.5, 1.2); p = A([x0 - ang * yy, yy]); d = A([-ang, 1.]) * ln; c.line(A([p, p + d * .5, p + d]), w * r.uniform(.6, 1.3), col, taper=(.7, .25), press=.1, seed=k, alpha=al * r.uniform(.4, 1.), tremor=0)
    return c.rgba()


def grief(cx=905, cy=430, s=330, yaw=24, pitch=-19, roll=11, byaw=12, f=0, seed=6, tear=.78, expr='griefhid', hidden=1., bg=None, rain=1., wind=(.10, -.04), fg=True, mouth=None, fd=None):
    """eyes hidden by wet hair; tear 0..1 = how far the one warm drop has run down the cheek"""
    H0 = GHead().pose(cx, cy, s, yaw * D, pitch * D, roll * D); G, X, L = goku.setup(H0, expr, 'rain', 'base'); X = dict(X, hw=.200, my=-.668, mU=[-.050, .018, .034, .020, -.046], mL=[-.084, -.084, -.072, -.074, -.074], jaw=.025, furrow=-1.)
    if mouth is not None: m_ = mouth; X = dict(X, hw=.176, my=-.664, mU=[-.046 - .010 * m_, .008 + .006 * m_, .022 + .006 * m_, .010 + .004 * m_, -.040 - .010 * m_], mL=[-.052 - .012 * m_, -.016 - .050 * m_, -.004 - .060 * m_, -.014 - .052 * m_, -.048 - .012 * m_], jaw=.006 + .03 * m_, clench=1.)
    L = dict(L, side=1, amt=.30, ldir=(.45, .85, .20), socket=.9, line='#0e0d12', hair=dict(L['hair'], root='#0b0c10', base='#15171d', light='#2f343f', hi='#7d8996', shade='#0d0e13', deep='#07080a', line='#030304', hl_thr=2.)); L['under'] = ('#33425c', '#1c2436', '#67788f'); L['gi'] = ('#8f5f46', '#573a31', '#b98866')
    fd = f if fd is None else fd; hr = Hair(G, 'base', L, f=fd, wind=wind, wet=1., seed=seed, painterly=False, extra=STUCK, kids=1); x, y = grid(); img = (grief_bg(f) if bg is None else bg).copy()
    if rain: img = over(img, rain_layer(f, 150, 300, 1.8, .42, 1, .13, speed=70))
    c = Cel(W, HH); hr.back(c); back = wet_hair(c, hr, False).rgba(); back = para(back, (cx, cy - 2 * s), (cx, cy + s), [(0, '#ffffff'), (1, '#8e929c')]); img = over(img, back)
    body = grief_body(G, L, X, byaw, f, seed); img = over(img, body)
    fl = grief_face(G, X, L, hr, seed, tear, hidden); img = over(img, fl)
    c = Cel(W, HH); hr.front(c); wet_hair(c, hr, True); fr = c.rgba(); img = over(img, fr)
    al = np.maximum(np.maximum(back[..., 3], fr[..., 3]), np.maximum(fl[..., 3], body[..., 3])); sh_ = lambda a, dx, dy: cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (W, HH), borderMode=cv2.BORDER_REPLICATE)
    e = cv2.GaussianBlur(np.clip(al - sh_(al, -4, 6), 0, 1), (0, 0), .9) * al; brk = np.clip(.5 + .7 * fbm(HH, W, (40, 16), (1, .5), 9), 0, 1.2); img = img + (e * brk)[..., None] * rgb('#e9f0f5') * .55
    # splashes where the rain strikes hair and shoulders
    c = Cel(W, HH); r = np.random.default_rng(300 + int(f) // 2); ed = np.argwhere((al > .5) & (sh_(al, 0, 6) < .5))
    for k in range(int(80 * rain)):
        yy, xx = ed[int(r.integers(len(ed)))]
        for q in range(int(r.integers(2, 4))): a = -math.pi / 2 + r.normal(0, .8); ln = r.uniform(3, 9); c.line(A([(xx, yy), (xx + math.cos(a) * ln * .6, yy + math.sin(a) * ln * .6), (xx + math.cos(a) * ln, yy + math.sin(a) * ln + 3)]), r.uniform(1, 1.8), '#eef4f8', taper=(.1, .7), seed=q, alpha=r.uniform(.2, .7))
    img = over(img, c.rgba())
    if fg:                                                       # what he holds, out of focus: the prince, eyes shut, smiling a little
        img = over(img, _prince())
    if rain: a = rain_layer(f, 26, 520, 4.2, .34, 2, .13, speed=150); img = over(img, np.dstack([blur(a[..., q], 1.6) for q in range(4)]))
    return img


_PR = {}
def _prince():
    if not _PR:
        from .vegeta import Vegeta
        a = Vegeta(1545, 1012, 236, -.42, .10, 1.30, expr='peace', light='rain', f=0, wet=1.).layer(); e = np.clip(a[..., 3] - shifted(a[..., 3], -3, 7), 0, 1); a = a.copy(); a[..., :3] = a[..., :3] * .62 + (e * a[..., 3])[..., None] * rgb('#aeb9c6') * .40; _PR['a'] = np.dstack([blur(a[..., q], 5.5) for q in range(4)]).astype(f32)
    return _PR['a']


def wet_hair(c, hr, front):
    """beads, drips and long wet lights on the clumps"""
    r = np.random.default_rng(17)
    for b in hr.bl:
        if (b.dp >= hr.split) != front or b.hw.max() < 8: continue
        n = len(b.t); c.clip(b.P)
        for j in range(1 + int(b.hw.max() / 16)):
            u = r.uniform(-.85, .85); i0 = int(r.uniform(2, n * .45)); i1 = int(min(n - 1, i0 + r.uniform(.2, .5) * n)); ii = np.arange(i0, i1)
            if len(ii) > 4: c.line(b.C[ii] + b.N[ii] * (b.hw[ii] * u)[:, None], r.uniform(.9, 2.6), '#aebac6' if u * hr.L['side'] > -.2 else '#48505c', taper=(.4, .6), press=.4, seed=j, alpha=r.uniform(.35, .85))
        c.unclip()
        if r.random() < .8 and b.k[0] in 'FXLRCS':                  # a drop hangs at the tip
            p = b.C[-1]; k = r.uniform(3.5, 6.5); q = np.linspace(0, 6.283, 21)[:-1]; P = np.c_[p[0] + k * .8 * np.sin(q) * np.abs(np.sin(q / 2)), p[1] + k * .9 - k * 1.3 * np.cos(q)]; c.fill(P, '#59636e', .9); c.fill(np.c_[p[0] - k * .15 + np.cos(q) * k * .26, p[1] + k * 1.2 + np.sin(q) * k * .42], '#f6fafc', .95)
    return c


def grief_body(G, L, X, byaw, f, seed):
    c = Cel(W, HH); B = Bust(c, G, L, byaw=byaw * D, bpitch=-.10, broll=.06, raise_=.16, wear='torn', seed=seed, tendon=.3, wet=1., disc=True, shift=(.02, -.50)); B.neck(); B.neck_shade(); B.neck_lines(); B.shirt(); B.gi(); B.arm_bare(); sil = c.rgba()[..., 3]
    # wet cloth: dark soak, hard lights along folds, water running
    c.clip(B.B.P([(-2.2, -1.4), (2.2, -1.4), (2.2, -4.6), (-2.2, -4.6)])); r = np.random.default_rng(6)
    for k in range(90):
        xx, yy = r.uniform(-1.7, 1.7), r.uniform(-3.6, -1.5); q = r.uniform(.15, .5); c.line(spl(B.B.P([(xx, yy), (xx + r.normal(0, .02), yy - q * .5), (xx + r.normal(0, .04), yy - q)]), n=8), r.uniform(1.2, 3.2), '#d9c8b4' if r.random() < .5 else '#3a2a28', taper=(.4, .5), seed=k, alpha=r.uniform(.2, .5))
    c.unclip(); a = c.rgba(); s = G.s
    # the bowed head keeps the rain off the chest: its shadow, hard
    j, k = G.jaw(-1), G.jaw(1); sh = GP.mask([np.vstack([j, k[::-1][1:], B.B.P([(.62, -1.2), (.70, -1.9), (.44, -2.5), (.12, -2.9), (-.30, -2.62), (-.62, -2.1), (-.70, -1.3)], .1)])]) * a[..., 3]; a[..., :3] *= 1 - .42 * blur(sh, 1.2)[..., None]
    return para(a, (G.cx, G.cy + .5 * s), (G.cx - .2 * s, G.cy + 2.1 * s), [(0, '#dfe2e8'), (.45, '#9a9ca8'), (1, '#3c3e4a')], 'mul', 1.)


def grief_face(G, X, L, hr, seed, tear, hidden):
    s = G.s; H = G; c = Cel(W, HH); F = Face(c, G, X, L, seed, hr, 0., 0., 1.08, True); d = 1 if G.yaw > 0 else -1; F.ears([d]); ear = c.rgba(); ear[..., :3] *= .62; sil = GP.mask([F.F]); sg = F.sg
    f_ = lambda P, cn=None: H.Pc(spl(A(P, float), True, cn, 6) if cn is not None else A(P, float)); hw, my = X['hw'], X['my']
    off = A([sg * .05, .11]) * s; casts = [b.P + off for b in hr.bl if b.k[0] in 'FCX' and b.dp > hr.split] + [p + off * .9 for p in hr.cap_front]
    M = list(casts)
    sl = F.side_line(sg, .22); Q = H.Pc(sl, squeeze=False); M.append(np.vstack([Q, (Q + A([sg * 3 * s, 0]))[::-1]]))
    veil = [(-.9, .9), (-.9, -.16), (-.66, -.20), (-.56, -.10), (-.48, -.19), (-.38, -.13), (-.27, -.22), (-.17, -.15), (-.10, -.19), (-.04, -.30), (.02, -.24), (.07, -.20), (.15, -.13), (.24, -.21), (.34, -.12), (.44, -.20), (.55, -.10), (.66, -.20), (.9, -.16), (.9, .9)]
    if hidden > 0: M.append((f_([(a, b + (1 - hidden) * .5) for a, b in veil]), 1.))
    for q in (-1, 1): M.append(H.Pm(A([(q * a, b) for a, b in [(.105, -.40), (.18, -.45), (.27, -.52), (hw + .055, my + .0), (hw + .04, my - .12), (hw + .085, my + .02), (.34, -.50), (.22, -.42)]])))
    M += [F.N([(-.082, -.440, .2), (-.03, -.426, .9), (.03, -.426, .9), (.082, -.440, .2), (.05, -.47, .1), (sg * .04, -.505, .0), (-.05, -.47, .1)]), F.N([(sg * .012, -.05, 1), (sg * .02, -.28, .9), (sg * .03, -.38, .7), (sg * .088, -.42, .15), (sg * .07, -.30, .25), (sg * .05, -.1, .4)]), H.Pm(A([(-.085, my - .115), (0, my - .10), (.085, my - .115), (.06, my - .155), (0, my - .165), (-.06, my - .155)])), f_([(-.30, -.97), (-.44, -.82), (-.60, -.62), (-.9, -.6), (-.9, -1.2), (.5, -1.2), (.30, -.96), (0, -.985)])]
    Mm = GP.mask(M, within=F.F); deep = [f_([(a, b + .16) for a, b in veil])] if hidden > 0 else []; D2 = blur(GP.mask(deep, within=F.F), 10) * .85
    o = GP.tone(sil, Mm, RAIN_LIT, RAIN_SHD, .13 * s, D2, '#1c1c22', '#8a6f6a', .40, .02, 4); x, y = grid(); e = np.clip(sil - shifted(blur(sil, 14), -30, 8), 0, 1) * (sil > .5); o = o + (blur(e, 6) * .22)[..., None] * rgb('#dfe9f2'); o = o * (1 + .016 * fbm(HH, W, (3, 1.6), (1, .7), 21)[..., None])
    lay = GP.over_(ear, GP.prem(np.clip(o, 0, 2), sil)); c = Cel(W, HH); F.c = c; F.ears([-d]); ea = c.rgba(); ea[..., :3] *= rgb('#a2a0a8'); lay = GP.over_(lay, ea); c = Cel(W, HH); F.c = c; c.clip(F.F); hk = '#3c383f'
    GP.cheek_hatch(F, [(sg * .62, -.26, sg * .40, -.56, 22, .11, sg * -.4, -.6), (-.50, -.16, -.24, -.20, 16, .07, .1, -1.), (.18, -.15, .50, -.16, 20, .07, -.1, -1.), (sg * .36, -.60, sg * .30, -.86, 14, .09, sg * -.85, -.1)], hk, 1.4, 7, .75)
    c.unclip(); F.contour(); fa = abs(H.yaw) < .75; c.clip(F.F)
    if hidden < .8:
        for q in (-1, 1): F.eye(q)
    F.mouth(); F.wet = 1.; F.rainskin(-.14 if hidden > .5 else 9.); c.unclip(); F.nose(); F.lines()
    if hidden < .8:
        for q in (-1, 1): F.brow(q)
    # clenched jaw muscle and the crease of the chin
    for q in (-1, 1):
        if H.vis(q * .54, -.52): F.ln(H.Pc([(q * .60, -.40), (q * .555, -.50), (q * .52, -.60)]), .46, seed=170 + q, taper=(.5, .5), alpha=.9)
    F.ln(H.Pm([(-.10, my - .150), (0, my - .172), (.09, my - .152)]), .42, seed=175, taper=(.5, .5)); F.ln(H.Pc([(-.05, -.90), (.0, -.915), (.05, -.90)]), .34, seed=176, taper=(.5, .5))
    # the one drop that is not rain: warm
    if tear > 0:
        tx = d * -.30; pts = A([(tx, -.15), (tx + .005, -.25), (tx + .02, -.36), (tx + .00, -.48), (tx - .02, -.60), (tx + .02, -.74), (tx + .10, -.88)]); P = spl(pts, n=14); m = max(8, int(len(P) * tear)); F.stream(P[:m:6], .020, 3, .60, '#ffe6c8')
        p = P[m - 1]; F.bead(p[0], p[1], .036); q = np.linspace(0, 6.283, 25)[:-1]; e = H.Pc([(p[0], p[1])])[0]; rad(c, None, (e[0] + .004 * s, e[1] + .010 * s), .036 * s, '#ffcf94', .85); c.fill(np.c_[e[0] + .010 * s + np.cos(q) * .007 * s, e[1] - .012 * s + np.sin(q) * .011 * s], '#ffffff')
    return GP.over_(lay, c.rgba())
