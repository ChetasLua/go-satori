"""f2: one face that goes from the rain to the gold.  picture(p) -> (RGB, figure layer) for a dict p of state values (see DEF)."""
import numpy as np, math, cv2
from db import W, H as HH, ki
from db.ink import Cel, spl, resample
from db.comp import blur, over, radial
from db.kit import rim as rim_, para, shifted, outline
from db.paint import rgb, f32, smooth, ramp, fbm, grid
from . import goku, goku_hair as GH, goku_light as GL, goku_head as GF, goku_paint as GP, goku_heroes as GX
from .goku_head import GHead, Face, rad
from .goku_hair import mixc, Hair
from .goku_bust import Bust
A = np.array
D = math.pi / 180
RSHD = [(0, '#7f7476'), (.48, '#6d6468'), (.60, '#534d55'), (.78, '#3a373f'), (.92, '#2a2930'), (1, '#1e1d23')]


def hx(c): c = np.clip(np.asarray(c, float), 0, 1) * 255 + .5; return '#%02x%02x%02x' % tuple(int(v) for v in c[:3])
def lp(a, b, t):
    if isinstance(a, str): return hx(rgb(a) * (1 - t) + rgb(b) * t)
    if isinstance(a, dict): return {k: (lp(a[k], b[k], t) if k in b else a[k]) for k in a}
    if isinstance(a, (tuple, list)): return type(a)(lp(x, y, t) for x, y in zip(a, b))
    if isinstance(a, bool) or a is None: return a
    return a * (1 - t) + b * t


RAINL = None; GOLDL = None
def lights(gold, hgold):
    global RAINL, GOLDL
    if RAINL is None:
        r = dict(GL.light('rain', side=1)); r.update(amt=.10, ldir=(.35, .55, .75), socket=.8, line='#0e0d12', under=('#33425c', '#1c2436', '#67788f'), gi=('#8f5f46', '#573a31', '#b98866'))
        g = dict(GL.light('gold')); g.update(line='#2a0f06', amt=-.15, ldir=(.35, .55, .75), socket=.6)
        RAINL = r; GOLDL = g
    L = {}
    for k in GOLDL: L[k] = lp(RAINL[k], GOLDL[k], gold) if k in RAINL and k in ('skin', 'white', 'under', 'gi', 'line', 'mouth', 'teeth', 'disc', 'blush', 'amt', 'socket') else GOLDL[k]
    L['side'] = 1; L['rim'] = []; L['para'] = []; L['hdir'] = (.40, .35, .85)
    K = dict(root='#0b0c10', base='#15171d', light='#2f343f', hi='#8793a0', shade='#0d0e13', deep='#07080a', line='#030304', cap='#0b0c10')
    Gd = dict(root='#b8640e', base='#ffc928', light='#ffe670', hi='#fffbd8', shade='#e08a14', deep='#a2500c', line='#4a1e06', cap='#b05a0e')
    Am = dict(root='#2a1006', base='#5a2a0a', light='#8a4a10', hi='#d08a2a', shade='#3c1a06', deep='#241004', line='#0c0402', cap='#2a1006'); l3 = lambda a, m, b, t: lp(a, m, t / .4) if t < .4 else lp(m, b, (t - .4) / .6)
    L['hair'] = dict(l3(K, Am, Gd, hgold), dark=hgold < .4, hl_thr=.2); L['brow'] = l3('#101015', '#4a2208', '#b36c14', hgold); return L


DEF = dict(cx=985, cy=520, s=306, yaw=-12, pitch=1, roll=-8, byaw=14, rise=1., gold=1., hgold=1., expr='scream', eyes='ssj', wet=0., hide=0., raise_=.16, f=0, seed=5, wind=(0, .25), tears=0., steam=1., vein=1., aura=1., lw=1.22, bpitch=-.04, jaw=None, macro=0., dark=0., hair=True, over={})


def picture(p, bg=None, extras=True):
    p = dict(DEF, **p); H0 = GHead().pose(p['cx'], p['cy'], p['s'], p['yaw'] * D, p['pitch'] * D, p['roll'] * D); X = GF.getx(p['expr']); X = dict(X, **dict(p['over'], tears=0.)); X['style'] = p['eyes']; X['vein'] = X['vein'] * p['vein']
    if p['jaw'] is not None: X['jaw'] = p['jaw']
    L = lights(p['gold'], p['hgold']); G = GHead.of(H0, X['jaw'], X['jx']); s = G.s; f = p['f']; x, y = grid(); cx, cy = p['cx'], p['cy']
    st = GH.mix(GH.BASE, GH.SSJ, p['rise']) if p['rise'] > 0 else GH.BASE; ex = [dict(q) for q in GX.STUCK] if p['wet'] > .45 and p['rise'] < .3 else ()
    zoom = s > 900
    hr = Hair(G, st, L, f=f, wind=p['wind'], wet=p['wet'], seed=p['seed'], painterly=p['hgold'] > .4, kids=1, detail=.42, flat=True, extra=ex) if p['hair'] and not zoom else None
    lit = [(a[0] * (1 - p['gold']) + b[0] * p['gold'], lp(a[1], b[1], p['gold'])) for a, b in zip(GX.RAIN_LIT, GX.GOLD_LIT)]; shd = [(a[0] * (1 - p['gold']) + b[0] * p['gold'], lp(a[1], b[1], p['gold'])) for a, b in zip(RSHD, GX.GOLD_SHD)]
    lay = np.zeros((HH, W, 4), f32); g = p['gold']
    if hr is not None:
        c = Cel(W, HH); hr.back(c); (GX.wet_hair(c, hr, False) if p['wet'] > .3 else None); b = c.rgba(); lay = para(b, (cx, cy - 2.4 * s), (cx, cy + .4 * s), [(0, lp('#ffffff', '#fff4d8', g)), (.5, lp('#c9cbd2', '#e8b45c', g)), (1, lp('#8e929c', '#9c5618', g))])
    if not zoom: lay = GP.over_(lay, body(G, L, X, p, lit, shd))
    lay = GP.over_(lay, face(G, X, L, hr, p, lit, shd))
    if hr is not None:
        c = Cel(W, HH); hr.front(c); (GX.wet_hair(c, hr, True) if p['wet'] > .3 else None); b = c.rgba(); b = para(b, (cx, cy - 2.2 * s), (cx, cy), [(0, '#ffffff'), (.6, lp('#e6e8ec', '#fff4dc', g)), (1, lp('#b9bcc4', '#f2cc8c', g))]); lay = GP.over_(lay, b)
    al = lay[..., 3]; sh_ = lambda a, dx, dy: cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (W, HH), borderMode=cv2.BORDER_REPLICATE)
    for dx, dy, k in ((-5, 3, 1.), (5, 3, .9)):
        e = cv2.GaussianBlur(np.clip(al - sh_(al, dx, dy), 0, 1), (0, 0), .8) * al; lay[..., :3] += e[..., None] * rgb('#fff2b8') * .85 * k * g + e[..., None] * rgb('#dfe6ea') * .30 * k * (1 - g)
    e = np.clip(blur(1 - al, 26) * 1.6 - .25, 0, 1) * al; lay[..., :3] += e[..., None] * rgb('#ffb23a') * .28 * g * p['aura']
    if p['dark']: lay[..., :3] *= 1 - p['dark']
    if p['eyes'] == 'ssj' and X['blank'] < .5 and not zoom:
        for q in (-1, 1): z = G.Pc([(q * (X['xi'] + (X['xo'] - X['xi']) * .43), -.01)])[0]; lay[..., :3] += (radial(z[0], z[1], .20 * s, .14 * s, 2.) * al)[..., None] * rgb('#3fffe0') * .30
    if X['blank'] > .5 and not zoom:
        for q in (-1, 1): z = G.Pc([(q * .31, -.01)])[0]; lay[..., :3] += (radial(z[0], z[1], .26 * s, .16 * s, 2.) * al)[..., None] * rgb('#fff6cc') * .30 * max(g, .35)
    img = bg.copy() if bg is not None else np.zeros((HH, W, 3), f32); img = over(img, lay); return img, lay, G, X, hr


def body(G, L, X, p, lit, shd):
    c = Cel(W, HH); s = G.s; g = p['gold']; B = Bust(c, G, L, byaw=p['byaw'] * D, bpitch=p['bpitch'], raise_=p['raise_'], wear='torn', seed=p['seed'], tendon=X['tendon'], dirt=1., shift=p.get('shift', (0, 0))); B.neck(); sil = c.rgba()[..., 3]; bd = B.B; t = X['tendon']
    P = lambda L_, cn=None: spl(bd.P(A(L_, float)), True, cn, 8) if cn is not None else bd.P(A(L_, float)); j, k = G.jaw(-1), G.jaw(1)
    M = [np.vstack([G.P3(A([(-.62, -.40, -.2)])), bd.P([(-.62, -1.0), (-.44, -1.22), (-.30, -1.10), (-.10, -1.34), (.06, -1.20), (.22, -1.36), (.42, -1.12), (.62, -1.0)], .02), G.P3(A([(.64, -.40, -.2)]))])]
    for q in (-1, 1): M += [P([(q * .46, -1.0), (q * .34, -1.4), (q * .16, -1.76), (q * .26, -1.78), (q * .46, -1.5), (q * .56, -1.2)], (0, 2)), (P([(q * .20, -1.2), (q * .24, -1.5), (q * .12, -1.74), (q * .14, -1.4)], (0, 2)), float(t)), P([(q * .62, -1.28), (q * .9, -1.5), (q * 1.2, -1.62), (q * .8, -1.7), (q * .56, -1.6)], (0, 2)), P([(q * .18, -1.88), (q * .6, -1.9), (q * 1.0, -1.84), (q * .6, -2.06), (q * .2, -2.1)], (0, 2))]
    M += [P([(1.58, -1.96), (1.70, -2.24), (1.94, -2.74), (1.80, -2.78), (1.52, -2.42), (1.42, -2.12)], (0, 3)), P([(1.20, -2.56), (1.66, -2.74), (1.94, -2.82), (2.04, -3.3), (1.98, -4.2), (1.6, -4.2), (1.7, -3.2), (1.4, -2.9)], (0, 2)), P([(2.16, -2.2), (2.30, -2.7), (2.20, -3.3), (2.4, -4.2), (2.9, -4.2), (2.7, -2.2)]), P([(-.13, -1.80), (0, -1.745), (.13, -1.80), (.06, -1.90), (0, -1.96), (-.06, -1.90)])]
    Mm = GP.mask(M, within=B.sil); M2 = GP.mask([np.vstack([j, k[::-1][1:], bd.P([(.40, -1.0), (.0, -1.06), (-.40, -1.0)], .03)])], within=B.sil)
    o = GP.tone(sil, Mm, lit, shd, .16 * s, blur(M2, 8) * .7, lp('#1c1c22', '#4a1c0a', g), lp('#8a6f6a', '#ff7a2a', g), .5, .02, 8); lay = GP.prem(o, sil); c = Cel(W, HH); B.c = c
    for k_, e_ in ((0, B.nl), (1, B.nr)): c.line(e_, 5., L['line'], taper=(.08, .12), press=.4, seed=k_)
    B.neck_lines(True); lc = lp('#2c2a30', '#4a1c08', g)
    for q in (-1, 1):
        for P_, w_ in (([(.50, -.98), (.40, -1.30), (.22, -1.62), (.12, -1.80)], 4.6), ([(.30, -1.02), (.26, -1.34), (.16, -1.66)], 3.0), ([(.40, -1.10), (.44, -1.25), (.36, -1.36), (.40, -1.5)], 2.0)): c.line(spl(bd.P([(q * a, b) for a, b in P_]), n=10), w_, lc, taper=(.4, .35), press=.35, seed=int(w_ * 9), alpha=float(t) * .9 + .1)
    B.shirt(); B.gi(); B.arm_bare(); a = c.rgba(); a = para(a, (G.cx, G.cy + .8 * s), (G.cx, G.cy + 2.4 * s), [(0, lp('#dfe2e8', '#ffffff', g)), (.5, lp('#9a9ca8', '#f0cfa0', g)), (1, lp('#3c3e4a', '#a86a3a', g))], 'mul', .9); return GP.over_(lay, a)


VEIL = [(-.9, .9), (-.9, -.16), (-.66, -.20), (-.56, -.10), (-.48, -.19), (-.38, -.13), (-.27, -.22), (-.17, -.15), (-.10, -.19), (-.04, -.30), (.02, -.24), (.07, -.20), (.15, -.13), (.24, -.21), (.34, -.12), (.44, -.20), (.55, -.10), (.66, -.20), (.9, -.16), (.9, .9)]


def face(G, X, L, hr, p, lit, shd):
    s = G.s; H = G; g = p['gold']; zoom = s > 900; c = Cel(W, HH); F = Face(c, G, X, L, p['seed'], hr, .01, 0., p['lw'], True, style=X['style'])
    if not zoom: F.ears([-1, 1])
    ear = c.rgba(); sil = GP.mask([F.F], ss=1 if zoom else 2)
    f_ = lambda P, cn=None: H.Pc(spl(A(P, float), True, cn, 6) if cn is not None else A(P, float)); B = A(X['B'], float); by = lambda x: np.interp(x, B[:, 0], B[:, 1]); hw, my = X['hw'], X['my']; op = float(np.clip(X['jaw'] / .2, 0, 1)); ang = float(np.clip(X['furrow'], 0, 1))
    casts = []
    if hr is not None: off = A([-.03, .060 + .04 * p['wet']]) * s; casts = [b.P + off for b in hr.bl if b.k[0] in ('FX' if p['rise'] > .25 else 'FXC') and b.dp > hr.split and (b.P[:, 1].max() > H.P([(0, .5)])[0][1])] + [q + off * .9 for q in hr.cap_front]
    M, M2 = list(casts), []
    for q in (-1, 1):
        sh = q < 0
        if ang > .2: M.append(f_([(q * a, b) for a, b in [(.045, by(.06) + .03), (.14, by(.14)), (.30, by(.30)), (.46, by(.46)), (.60, by(.6) - .03), (.565, .09), (.46, .10), (.36, .065), (.26, .02), (.18, -.02), (.145, -.085), (.12, -.13), (.085, -.06)]]))
        else: M.append(f_([(q * a, b) for a, b in [(.06, by(.1)), (.16, by(.16)), (.30, by(.30)), (.44, by(.44)), (.38, by(.4) - .06), (.26, by(.26) - .08), (.18, by(.18) - .10), (.13, -.06), (.09, -.08)]]))
        M.append(f_([(q * a, b) for a, b in [(.70, -.10), (.60, -.22), (.50, -.30), (.42, -.42), (.40, -.60), (.44, -.80), (.50, -.62), (.56, -.50), (.64, -.42), (.76, -.36)]], (0, 5)))
        M.append(H.Pm(A([(q * a, b) for a, b in [(.105, -.40), (.18, -.44), (.28, -.50), (hw + .05, my - .02), (hw + .03, my - .20 * max(op, .4)), (hw + .075, my - .02), (.36, -.50), (.24, -.42)]])))
        M.append(f_([(q * a, b) for a, b in [(.50, .62), (.545, .44), (.60, .28), (.655, .16), (.625, .04), (.66, -.06), (.625, -.16), (.585, -.24), (.50, -.30), (.42, -.42), (.385, -.58), (.40, -.74), (.46, -.62), (.52, -.50), (.60, -.44), (.60, -.62), (.48, -.80), (.28, -1.0), (.9, -1.1), (.9, .6)]]))
        M.append(F.N([(q * .022, -.10, .8), (q * .030, -.26, .8), (q * .048, -.36, .6), (q * .090, -.40, .15), (q * .078, -.32, .2), (q * .052, -.20, .5)]) if sh else F.N([(q * .05, -.34, .5), (q * .09, -.40, .15), (q * .08, -.33, .2)]))
        M2.append(f_([(q * a, b) for a, b in [(.07, .02), (.12, by(.12) - .0), (.20, by(.2) - .01), (.19, .0), (.15, -.06), (.115, -.10), (.09, -.04)]])); M2.append(f_([(q * a, b) for a, b in [(.72, -.20), (.60, -.30), (.52, -.42), (.50, -.56), (.58, -.46), (.70, -.38)]]))
    M += [F.N([(-.082, -.440, .2), (-.04, -.426, .8), (.04, -.426, .8), (.082, -.440, .2), (.06, -.47, .1), (0, -.50, .0), (-.06, -.47, .1)]), H.Pm(A([(-.15, my + min(X['mL']) - .055), (0, my + min(X['mL']) - .04), (.15, my + min(X['mL']) - .055), (.09, my + min(X['mL']) - .10), (0, my + min(X['mL']) - .115), (-.09, my + min(X['mL']) - .10)]))]
    if ang > .2: M.append(f_([(-.05, .16), (0, .20), (.05, .16), (.028, .02), (.02, -.06), (-.02, -.06), (-.028, .02)]))
    if p['hide'] > 0: M.append((f_([(a, b + (1 - p['hide']) * .45) for a, b in VEIL]), 1.))
    M.append(f_([(-.30, -.99), (-.44, -.80), (-.50, -.66), (-.8, -.6), (-.8, -1.2), (0, -1.2), (-.12, -1.0)]))
    Mm = GP.mask(M, within=F.F, ss=1 if zoom else 2); D2 = blur(GP.mask(M2, within=F.F, ss=1), 2.) * .75; M2b = blur(GP.mask([f_([(q * a, b) for a, b in [(.75, .3), (.70, .1), (.72, -.1), (.66, -.26), (.60, -.42), (.56, -.62), (.44, -.82), (.30, -.98), (.9, -1.1), (.9, .3)]]) for q in (-1, 1)], within=F.F, ss=1), 5) * .55; D2 = np.maximum(D2, M2b)
    if p['hide'] > 0: D2 = np.maximum(D2, blur(GP.mask([f_([(a, b + .16 + (1 - p['hide']) * .45) for a, b in VEIL])], within=F.F, ss=1), 10) * .85 * p['hide'])
    sg = .12 * s if not zoom else 110.
    o = GP.tone(sil, Mm, lit, shd, sg, D2, lp('#1c1c22', '#5e240e', g), lp('#8a6f6a', '#ff6a1c', g), (.4 + .2 * g) * (0 if zoom else 1), .02, 4); o = o + (0 if zoom else (blur(np.clip(sil - shifted(sil, -40, 10), 0, 1), 16) * .30 + blur(np.clip(sil - shifted(sil, 40, 10), 0, 1), 16) * .22)[..., None] * lp_rgb('#b9c4cc', '#ffcf5a', g) * (.5 + .5 * g)); o = o * (1 + .014 * fbm(HH, W, (3, 1.6), (1, .7), 21)[..., None])
    lay = GP.over_(ear * (1 if g > .5 else A([.7, .7, .7, 1.], f32)), GP.prem(np.clip(o, 0, 2), sil)); c = Cel(W, HH); F.c = c; hk = lp('#3c383f', '#a24a18', g)
    if not zoom:
        c.clip(F.F)
        for q in (-1, 1):
            GP.cheek_hatch(F, [(q * .62, -.22, q * .42, -.44, 20, .12, q * -.5, .5), (q * .20, -.15, q * .48, -.12, 22, .09, q * .5, -.8), (q * .14, -.10, q * .10, -.02, 8, .06, q * .7, -.6), (q * .50, .40, q * .60, .24, 10, .09, q * -.8, -.4)], hk, 1.6, 7 + q, .85); GP.cheek_hatch(F, [(q * .08, .28, q * .22, .36, 10, .08, q * .3, .9)], hk, 1.3, 17 + q, .7)
        c.unclip(); F.contour()
    if p['hide'] > .9: features_hidden(F)
    elif zoom: zoom_features(F, p)
    else: F.features()
    if not zoom and p['hide'] < .9:
        for q in (-1, 1):
            P = F.brow_poly(q); n = (len(P) - 1) // 2; dn = P[n + 1:][::-1]
            if ang > .5: F.ln(H.Pc(dn[int(n * .04):int(n * .86)] - A([0, .004])), 1.15, seed=150 + q, taper=(.10, .30))
            outline(c, H.Pc(P), 2.6, lp('#08080a', '#4a1e06', p['hgold']), seed=60 + q)
            if q < 0: F.ln(H.Pc([(q * .30, -.222), (q * .40, -.262), (q * .50, -.318)]), .36, seed=160 + q, col='#8a2a1c', taper=(.4, .6))
            t = p['tears']
            if t > .05:
                F.stream(A([(q * .30, -.125), (q * .31, -.20), (q * .325, -.30), (q * .33, -.42), (q * .35, -.54), (q * .33, -.68)])[:max(3, int(round(6 * t)))], .014, 3 + q, .55, lp('#a9cdf0', '#fff6e0', g))
                if t > .5: F.stream(A([(q * .46, -.10), (q * .49, -.19), (q * .50, -.32), (q * .47, -.48)])[:max(3, int(round(4 * t)))], .010, 5 + q, .5, lp('#a9cdf0', '#fff6e0', g))
    return GP.over_(lay, c.rgba())


def lp_rgb(a, b, t): return rgb(a) * (1 - t) + rgb(b) * t


def features_hidden(F):
    c = F.c; c.clip(F.F); F.mouth(); F.wet = 1.; F.rainskin(-.14); c.unclip(); F.nose(); F.lines()
    for q in (-1, 1):
        if F.H.vis(q * .54, -.52): F.ln(F.H.Pc([(q * .60, -.40), (q * .555, -.50), (q * .52, -.60)]), .46, seed=170 + q, taper=(.5, .5), alpha=.9)


def zoom_features(F, p):
    """only what is near the eye, at macro size"""
    H, X, c, s = F.H, F.X, F.c, F.u; w0 = F.w; F.w = min(F.w, 16.); macro_eye(F, 1, p); L = F.L
    # brow: a wedge that comes out of the furrow, swells and runs out; every edge is hair
    B = A(X['B'], float) + A([0, .012]); mid = GF.crv(B[:, 0], B[:, 1], 90, straight=.45); n = len(mid); u = np.linspace(0, 4, n); w = np.interp(u, [0, .25, .8, 1.6, 2.5, 3.3, 4], [.004, .020, .040, .050, .046, .026, .0]); T = np.gradient(mid, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]; up, dn = mid + N * w[:, None], mid - N * (w * .82)[:, None]; P = np.vstack([up, dn[::-1]]); a_, b_ = H.Pc([mid[n // 2] + N[n // 2] * .05])[0], H.Pc([mid[n // 2] - N[n // 2] * .05])[0]; c.grad(H.Pc(P), a_, b_, [(0, '#ffd668', 1), (.45, '#d08a1c', 1), (1, '#6a2e08', 1)]); r = np.random.default_rng(31)
    for k in range(340):
        i = int(r.uniform(0, n * .97)); v = r.uniform(-.95, 1.0); e_ = v > .55 or v < -.6; q = mid[i] + N[i] * (w[i] * (v if v > 0 else v * .82)); ln = r.uniform(.030, .075) * (1 - .4 * i / n); lift = .30 * (v > 0) + r.normal(0, .08) - .12 * (v < -.5); d = T[i] * math.cos(lift) + N[i] * math.sin(lift); tip = q + d * ln; c.line(H.Pc([q - d * ln * .25, q + d * ln * .4 + N[i] * .002, tip]), r.uniform(2., 6.5) * (.8 if e_ else 1.), ['#fff0b0', '#ffd24a', '#b4640c', '#7a3606', '#4a1c04'][int(r.integers(0, 3) if v > .1 else r.integers(2, 5))], taper=(.25, .75), press=.3, seed=k, alpha=.92, tmin=.03)
    F.furrow(); F.w = w0


def macro_eye(F, q, p):
    """the teal iris as a burning ring; the gold flame stands in it"""
    c, H, L, X, s = F.c, F.H, F.L, F.X, F.u; T5 = GF.T5; xs = X['xi'] + (X['xo'] - X['xi']) * T5; up = GF.crv(xs, A(X['U'], float) + A([0, .006, .022, .012, 0]), straight=.35); lo = GF.crv(xs - A([0, .012, .010, -.004, -.014]), A(X['Lo'], float) - A([0, .004, .006, .004, 0]), straight=.4); n = len(up); t = np.linspace(0, 1, n); Mx = lambda P: np.c_[q * np.asarray(P)[:, 0], np.asarray(P)[:, 1]]
    up = spl(A([(.116, -.082), (.150, -.030), (.205, .014), (.285, .056), (.370, .078), (.455, .066), (.515, .030), (.556, -.010)]), n=14); lo = spl(A([(.116, -.082), (.150, -.104), (.220, -.126), (.320, -.134), (.420, -.118), (.505, -.068), (.556, -.010)]), n=16); lo = np.stack([np.interp(np.linspace(0, 1, len(up)), np.linspace(0, 1, len(lo)), lo[:, k_]) for k_ in (0, 1)], -1); n = len(up); t = np.linspace(0, 1, n)
    Pu, Pl = H.Pc(Mx(up)), H.Pc(Mx(lo)); poly = np.vstack([Pu, Pl[::-1]]); c.grad(poly, Pu.mean(0), Pl.mean(0), [(0, '#c98f4a', 1), (.22, '#f4dcae', 1), (1, '#fffaf0', 1)]); c.clip([poly]); r = np.random.default_rng(3)
    ix, iy = q * (X['xi'] + (X['xo'] - X['xi']) * .43) + X['gx'], -.012 + X['gy']; rx = X['iris'] * 1.04; a = np.linspace(0, 2 * np.pi, 90)[:-1]
    cen = H.Pc([(ix, iy)])[0]; e1 = H.Pc([(ix + rx, iy)])[0] - cen; R = float(np.hypot(*e1)); E = lambda k, dx=0, dy=0: np.c_[cen[0] + dx * R + np.cos(a) * R * k, cen[1] + dy * R + np.sin(a) * R * k]
    for j in range(26):                                          # little veins
        th = r.choice([0, math.pi]) + r.normal(0, .35); p0 = cen + A([math.cos(th), math.sin(th)]) * R * r.uniform(1.6, 2.3); P = [p0]
        for k in range(4): P.append(P[-1] + (cen - P[-1]) * .12 + r.normal(0, R * .05, 2))
        c.line(spl(P, n=6), r.uniform(1.2, 3.2), '#d9542a', taper=(.1, .8), seed=j, alpha=.5)
    c.fill(E(1.06), '#032c2a'); c.fill(E(1.0), '#0a6b64'); rad(c, None, cen, R * 1.0, '#19c8b2', .95)
    for j in range(120):
        th = j / 120 * 6.283 + r.normal(0, .02); r0, r1 = r.uniform(.36, .50), r.uniform(.70, .97); c.line(A([cen + A([math.cos(th), math.sin(th)]) * R * r0, cen + A([math.cos(th + .05), math.sin(th + .05)]) * R * r1]), R * r.uniform(.008, .024), ['#8dfff0', '#0a6b64', '#d8fff6', '#1fd9c2'][j % 4], taper=(.1, .5), seed=j, alpha=.85)
    c.fill(E(.40), '#02100f')
    # the flame reflected, standing in the pupil, thin over the iris
    for cl, al0 in ((E(.98), .30), (E(.40), 1.)):
        c.clip([cl])
        for k, (col, sc, al) in enumerate((('#c25a08', 1.0, .9), ('#ffb31e', .74, .95), ('#fff3c0', .44, 1.))):
            for j in range(9): u = (j - 4) / 4.; h_ = R * sc * (1.15 - .75 * u * u) * (1 + .10 * math.sin(j * 2.1 + p['f'] * 1.3)); bx = cen[0] + u * R * .46 * sc; by_ = cen[1] + R * .60; c.fill(spl([(bx - R * .15 * sc, by_), (bx - R * .06 * sc + R * .04, by_ - h_ * .5), (bx + R * .05 * math.sin(j * 1.7), by_ - h_), (bx + R * .09 * sc, by_ - h_ * .45), (bx + R * .15 * sc, by_)], True, (2,), 8), col, al * al0)
        c.unclip()
    c.clip([E(.98)])
    c.unclip(); th_ = GF.prof(t, [(0, .006), (.5, .018), (1, .008)]); c.fill(H.Pc(Mx(np.vstack([up, np.c_[up[:, 0], up[:, 1] - th_][::-1]]))), '#3a1406', .72)
    c.fill(spl([cen + A([.30, -.46]) * R, cen + A([.50, -.62]) * R, cen + A([.62, -.40]) * R, cen + A([.42, -.28]) * R], True, n=8), '#ffffff'); c.fill(E(.07, -.42, .46), '#ffffff', .8); c.unclip()
    # inner corner
    c.fill(H.Pc(Mx(spl(A([(.116, -.082), (.140, -.066), (.156, -.080), (.146, -.098), (.128, -.094)]), True, (0,), 6))), '#e8a07a'); c.fill(H.Pc(Mx(spl(A([(.122, -.083), (.138, -.074), (.146, -.084), (.136, -.091)]), True, (0,), 6))), '#fff0dc', .8)
    # upper lid: thin at the inner corner, swelling, running out past the outer corner; lashes grow from it
    th2 = GF.prof(t, [(0, .003), (.15, .011), (.45, .020), (.80, .027), (.94, .022), (1, .010)]); tip = A([(.588, .004), (.618, .022)]); Bk = np.c_[up[:, 0], up[:, 1] + th2]; c.fill(H.Pc(spl(Mx(np.vstack([up[::6], up[-1:], tip[::-1][:1], Bk[::-1][::6], Bk[:1]])), True, (0, len(up[::6]), len(up[::6]) + 1), 6)), '#1a0803')
    r = np.random.default_rng(12)
    for j in range(17):
        i = int(n * (.22 + .046 * j)); i = min(i, n - 1); b = up[i] + A([0, th2[i] * .6]); ln = (.012 + .026 * (j / 16.)) * r.uniform(.8, 1.15); ang = 1.15 - 1.0 * (j / 16.) + r.normal(0, .06); d = A([math.cos(ang), math.sin(ang)]); cv = A([d[1], -d[0]]) * ln * .22; c.line(H.Pc(Mx(A([b - d * .004, b + d * ln * .5 + cv, b + d * ln]))), 13. - .25 * j, '#1a0803', taper=(.02, .92), press=.2, seed=j, tmin=.03)
    c.line(H.Pc(Mx(lo[int(n * .10):int(n * .97)])), 8.5, '#3a1406', taper=(.5, .25), press=.4, seed=4); c.line(H.Pc(Mx(lo[int(n * .16):int(n * .82)] + A([0, .0045]))), 4., '#fff6d8', taper=(.4, .4), alpha=.9)
    c.line(H.Pc(Mx(up[int(n * .16):int(n * .92)] + A([.004, .052]))), 5.5, '#8a3c12', taper=(.5, .4), seed=8); c.line(H.Pc(Mx(lo[int(n * .06):int(n * .5)] - A([.006, .040]))), 4.5, '#a24a18', taper=(.3, .6), seed=9, alpha=.8)
