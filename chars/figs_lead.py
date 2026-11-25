"""Figures the lead places into staged shots (Frieza in c1, c3, g5, g9, d5; Vegeta in d4, e1, e2, h3)."""
import numpy as np, math, cv2
from db import W, H as HH, comp, kit
from db.paint import rgb, f32, smooth
from db.cam import Camera3D
from db.hero3 import *
from db import hero3, hands as HD
from db.ink import Cel
from functools import lru_cache
keys = comp.lerp_keys
VIO = '#c784ff'


def c1_fig(path):
    def fig(f):
        x, y, s, o = path(f); g = (int(f) // 2) * 2; D = 12.; H_ = s * 8.15; foc = H_ * D / 1.60
        cam = Camera3D((0, -D, 0), (0, 0, 0), focal=foc, cx=x, cy=486); z = (486 - y) / foc * D
        F = Fig('frieza').pose(mix_pose(POSES['fists_down'], POSES['conduct'], float(np.clip(o, 0, 1)))).fly((0, 0, z)); F.heading = math.pi
        F.tail = lambda sk, ff: tail_curve(sk, g * .6, kind='sway', amp=.55 + .6 * o)
        F.blow(0, 0, .6 * (1 - o))
        L = F.draw(cam, g, light=(-.15, .35, -.75), pal='moon', rim=[dict(dir=(.95, .25, -.35), col=VIO, thr=.40, amt=.95), dict(dir=(-.95, .25, -.35), col='#f0dcff', thr=.46, amt=.9)], expr=dict(expr='closed' if f < 60 else 'serene'), key=.5, lines=.9)
        fig.info = F.info; L = L.copy(); L[..., :3] *= np.array([.46, .42, .60], f32); return L
    return fig


def c1_eyes(path, fig):
    def eyes(f):
        x, y, s, _ = path(f); o = float(keys(f, [(0, 0.), (59, 0.), (60, .35), (61, 1.), (80, 1.)])); inf = getattr(fig, 'info', None)
        if inf is None or 'head' not in inf: return [(x + d * .17 * s, y - 3.42 * s, o) for d in (-1, 1)]
        q = inf['head']; hx, hy = (q['x'], q['y']) if isinstance(q, dict) else np.asarray(q, float).reshape(-1)[:2]; return [(hx + d * .20 * s, hy + .02 * s, o) for d in (-1, 1)]
    return eyes


FZ = dict(skin='#8d86ad', shade='#322b4e', line='#08060e', nail='#1a0f2c', nail_shade='#0d0818', nail_dark='#0a0612')
def c3_hand(tip):
    def hand(f):
        g = (int(f) // 2) * 2; x, y = tip(f); c = Cel(W, HH)
        po = dict(HD.POSES['point']); cam = HD.Cam(0, 0, 300, 6.5); h = HD.Hand(po, 'R', R=HD.euler(168, -6, 4), pos=(0, 0, 0), bld=HD.SLIM, pivot=(0, 0, 0), nails='claw', marks=False)
        gg = HD.render([h], HD.Cam(960, 1160, 300, 6.5), light=(-.55, .45, .70), res=640, clip=(0, -400, W, HH + 900))
        tx, ty = gg.pts[list(gg.pts)[0]][('index', 'tip')] if isinstance(gg.pts, dict) else (960, 500)
        # place: move the camera so that the fingertip sits on the pearl
        gg = HD.render([h], HD.Cam(960 + (x - tx), 1160 + (y + 6 - ty), 300, 6.5), light=(-.55, .45, .70), res=640, clip=(0, -200, W, HH + 400))
        HD.ink(c, gg, dict(HD.PAL['night'], **FZ), w=5., seed=3); L = c.layer()
        L = kit.rim(L, 6, 5, VIO, .9, soft=.8); L = kit.rim(L, -5, 4, '#8a2cff', .7, soft=.8); L = L.copy()
        Y, X = np.mgrid[0:HH, 0:W].astype(f32); d = np.hypot(X - x, Y - y); L[..., :3] *= (.40 + .75 * np.exp(-d / 330.))[..., None]; L[..., :3] += (L[..., 3] * np.exp(-d / 120.) * .35)[..., None] * rgb('#b78cff'); return L
    return hand


def c3_figs(LEDGE):
    cache = {}
    def figs(f):
        if 'm' in cache: return cache['m']
        cam = Camera3D((0, -9, 1.15), (0, 0, 1.15), focal=2000., cx=960, cy=LEDGE - 258); LY = []
        for ch, var, x, pose, hd, e in (('piccolo', 'clean', -1.70, 'stand', -2.05, 'shock'), ('goku', 'clean', -.60, 'guard', -2.25, 'shock'), ('vegeta', 'clean', .70, 'stance_vegeta', -1.95, 'shock'), ('gohan', 'nohat', 1.62, 'stand', -2.2, 'shock')):
            F = Fig(ch, var).pose(pose).place(x, 0, 0, heading=hd).look((5., -4., 4.6))
            LY.append((F, dict(light=(.7, .35, -.45), pal='moon', rim=[dict(dir=(.9, .35, -.25), col='#e6d6ff', thr=.40, amt=.95)], key=.55, expr=dict(expr=e))))
        out = np.zeros((HH, W, 4), f32)
        for F, k in LY: L = F.draw(cam, 0, **k); out = L + out * (1 - L[..., 3:4])
        out = out.copy(); out[..., :3] *= np.array([.42, .40, .52], f32); cache['m'] = out; return out
    return figs


_E1 = {}
def e1(f):
    """Vegeta on his feet, dead still, only cloth and hair move (on threes)"""
    g = (int(f) // 3) * 3
    cam = Camera3D((0, -6, .55), (0, 0, .62), focal=2420., cx=1075, cy=640)
    F = Fig('vegeta', 'shattered').pose('lookup').place(0, 0, 0, heading=-.95); F.blow(-1.6, .2, .1)
    L = F.draw(cam, g, light=(-.25, .95, .10), pal='rain', rim=[dict(dir=(.35, .85, -.30), col='#e4eaec', thr=.46, amt=.8)], expr=dict(expr='peace', wet=1., wind=.10 + .06 * math.sin(g * .5), scratch=1, hair_drop=.18))
    _E1['info'] = F.info; return L
@lru_cache(2)
def _e1a():
    return _e1_anchors()
def e1_anchors(f): return _e1a()
def _e1_anchors():
    f = 0
    i = _E1.get('info')
    if not i: e1(f); i = _E1['info']
    q = lambda k, d: np.asarray(i[k][d], float).reshape(-1)[:2]
    sh = {d: q('shoulders', d) for d in 'LR'}; el = {d: q('elbows', d) for d in 'LR'}; kn = {d: q('knees', d) for d in 'LR'}; ha = {d: q('hands', d) for d in 'LR'}; so = {d: q('soles', d) for d in 'LR'}; pv = np.asarray([float(v) for v in i['pelvis']])
    rags = []
    drip = [np.array([el[d] + (3, -6), (el[d] + ha[d]) / 2 + (4, 0), ha[d] + (0, 12), ha[d] + (2, 34)]) for d in 'LR']
    return dict(feet=tuple((so['L'] + so['R']) / 2), rags=rags, drip=drip)


def e2(f):
    """straight down at his face (the head is thrown back, so from above we look at it frontally)"""
    cam = Camera3D((0, -.10, 3.55 - .004 * f), (0, -.02, 1.4), focal=9400., cx=960, cy=790, roll=.12)
    F = Fig('vegeta', 'shattered').pose('lookup').place(0, 0, 0, heading=math.pi)
    L = F.draw(cam, 0, light=(-.2, .5, .85), pal='rain', rim=[dict(dir=(.5, .5, -.4), col='#e4eaec', thr=.5, amt=.6)], expr=dict(expr='peace', wet=1., scratch=1, hair_drop=.35)); e2.info = F.info; return L
def e2_anchors(f):
    if not hasattr(e2, 'info'): e2(0)
    q = e2.info['head']; x, y = (q['x'], q['y']) if isinstance(q, dict) else np.asarray(q, float).reshape(-1)[:2]; r = np.random.default_rng(4)
    Hh = q.get('H') if isinstance(q, dict) else None; tear = Hh.P(np.array([(.50, -.05), (.555, -.12), (.60, -.20), (.63, -.32), (.645, -.46)])) if Hh is not None else None
    return dict(hits=np.c_[x + r.normal(0, 330, 190), y + 30 + r.normal(0, 330, 190)], tear=tear)


_D4 = {}
def d4(f):
    """manga bust: greys, one burning eye, arm driving at the lens (the big glove of the shot covers the hand)"""
    g = (int(f) // 4) * 4
    if g not in _D4:
        cam = Camera3D((.10, -2.6, 1.36), (-.02, 0, 1.36), focal=3050. + 9 * g, cx=720, cy=600, roll=-.05)
        F = Fig('vegeta', 'clean').pose('beam_one').place(0, 0, 0, heading=math.pi + .42)
        L = F.draw(cam, g, light=(-.65, .55, .35), pal='grey', rim=[dict(dir=(.9, .3, -.3), col='#ffffff', thr=.42, amt=.95)], expr=dict(expr='ego', wind=.5), lines=1.5)
        try:
            hm = np.clip(sum(np.asarray(F.mask(q), f32) for q in ('handR', 'handL')), 0, 1); hm = cv2.dilate(hm, np.ones((9, 9), np.uint8)); hm[:, :900] = 0; L = L * (1 - hm[..., None])
        except Exception: pass
        Y = L[..., :3].mean(-1, keepdims=True) / np.maximum(L[..., 3:4], 1e-4); Y = np.clip((Y - .12) * 1.35, 0, 1); _D4[g] = np.concatenate([np.repeat(Y, 3, 2) * L[..., 3:4], L[..., 3:4]], 2).astype(f32); _D4['info'] = F.info
    return _D4[g]


def h3(f):
    """dawn: Vegeta takes his half looking the other way (30-49)"""
    if not 30 <= f < 50: return None
    g = (f // 3) * 3; cam = Camera3D((0, -4, 1.40), (0, 0, 1.40), focal=5400., cx=1700, cy=690)
    F = Fig('vegeta', 'shattered').pose('contra').place(0, 0, 0, heading=-(math.pi * .5 + .45))
    L = F.draw(cam, g, light=(-.6, .5, .4), pal='dawn', rim=[dict(dir=(-.9, .3, -.3), col='#fff1da', thr=.45, amt=.9)], expr=dict(expr='away' if f < 42 else 'smile', scratch=1)); return L


# ------------------------------------------------------------------ d5, g5, g9
from chars import frieza as _Z, vegeta as _V
from functools import lru_cache


def d5_walker(step):
    @lru_cache(40)
    def W_(g):
        fy, s, u, lead = step(g); D = 9.; foc = 7.75 * s * D / 1.58; cam = Camera3D((0, -D, .9), (0, 0, .9), focal=foc, cx=975., cy=fy - .1 * foc)
        p = dict(walk((lead + u) * .5)); q = POSES['part']
        for k in ('hR', 'eR', 'gR', 'wR'):
            if k in q: p[k] = q[k]
        F = Fig('frieza').pose(p).place(0, 0, 0, heading=math.pi); F.tail = lambda sk, ff: tail_curve(sk, g, kind='wrap', amp=.6)
        L = F.draw(cam, g, light=(0, .2, -.9), pal='beam', rim=[dict(dir=(.9, .2, -.4), col='#fff3b8', thr=.38, amt=1.), dict(dir=(-.9, .2, -.4), col='#ffd35a', thr=.38, amt=1.)], key=.7, expr=dict(expr='contempt')).copy()
        a = L[..., 3:4]; lum = L[..., :3].max(-1, keepdims=True) / np.maximum(a, 1e-4); keep = np.clip((lum - .80) * 5, 0, 1); L[..., :3] = L[..., :3] * keep + a * (1 - keep) * (rgb('#2a1206') + .10 * lum * rgb('#a0601c'))
        i = F.info; palm = i['palms']['R'] if isinstance(i['palms'], dict) else i['palms'][1]; so = i['soles']; so = list(so.values()) if isinstance(so, dict) else list(so); so = [np.asarray(v, float).reshape(-1, 2).mean(0) for v in so]; ym = max(v[1] for v in so)
        return L, (float(np.asarray(palm).reshape(-1)[0]), float(np.asarray(palm).reshape(-1)[1])), [(float(v[0]), float(v[1]), float(np.clip(1 - (ym - v[1]) / (.5 * s), 0, 1))) for v in so]
    return lambda f: W_((f // 2) * 2 if (f % 10) > 2 else f)


@lru_cache(8)
def _bust(ch, var, pose, hd, cx, cy, foc, pal, e, rim, light, g=0, z=1.40, key=.1):
    cam = Camera3D((0, -4, z), (0, 0, z), focal=foc, cx=cx, cy=cy); F = Fig(ch, var).pose(pose).place(0, 0, 0, heading=hd); L = F.draw(cam, g, light=light, pal=pal, rim=[dict(dir=r[0], col=r[1], thr=r[2], amt=r[3]) for r in rim], expr=dict(e), key=key); return L
def d5_duo(f):
    k = (f - 50) / 10.
    A = _bust('vegeta', 'clean', 'stance_vegeta', -(math.pi * .5 + .25), 430., 640., 7600., 'moon', (('expr', 'fire'), ('wind', .3)), (((-.9, .3, -.3), '#ffcf6a', .46, .9),), (.75, .45, .45), 0, 1.36)
    B = _bust('frieza', 'clean', 'stand', math.pi * .5 + .25, 1430., 520., 7200., 'moon', (('expr', 'polite'),), (((.9, .3, -.3), '#c784ff', .42, .95), ((-.9, .2, -.4), '#ffcf6a', .50, .7)), (-.5, .5, .2), 0, 1.36)
    M = lambda L, dx: cv2.warpAffine(L, np.float32([[1, 0, dx], [0, 1, 0]]), (W, HH)); return [M(A, 10 * k), M(B, -22 * k)]


def d5_flick(T):
    @lru_cache(4)
    def H_(k, dx):
        s = 820.; yw = math.radians(48); v = _V.Vegeta(0, 0, s, yaw=yw, expr='shock', light='night', tiny=.5, seed=3); z = float(v.F.H.z(.10, .30)); q = v.F.P3(np.array([(.10, .30, z)]))[0]
        v = _V.Vegeta(T[0] - q[0], T[1] - q[1], s, yaw=yw, expr='shock' if k else 'fire', light='night', scratch=0); hd = v.layer(neck=False)
        po = dict(HD.POSES['point']) if k else dict(HD.POSES['pinch']); cam = lambda x, y: HD.Cam(x, y, 620, 5.); R = HD.euler(150, 12, 84); h = HD.Hand(po, 'R', R=R, pos=(0, 0, 0), bld=HD.SLIM, pivot=(0, 0, 0), nails='claw', marks=False)
        gg = HD.render([h], cam(1500, 620), light=(-.5, .75, .5), res=620, clip=(-600, -600, W + 900, HH + 900)); p = gg.pts[list(gg.pts)[0]][('index', 'tip')]
        gg = HD.render([h], cam(1500 + T[0] + dx - p[0], 620 + T[1] - p[1]), light=(-.5, .75, .5), res=620, clip=(-200, -200, W + 300, HH + 300)); c = Cel(W, HH); HD.ink(c, gg, dict(HD.PAL['night'], **FZ), w=6.5, seed=3); hn = kit.rim(c.layer(), 6, 6, VIO, .9, soft=.8); return hd, hn
    return lambda f: list(H_(0 if f < 62 else 1, 0 if f >= 62 else int(800 - 60 * (f - 60))))
def d5_torso(f): return _bust('vegeta', 'clean', 'hit_head', math.pi, 860., 480., 5200., 'night', (('expr', 'pain'),), (((.6, .6, -.4), '#f0dcff', .40, .95), ((-.9, .2, -.3), VIO, .44, .9)), (.3, .9, .3), 0, 1.30)


@lru_cache(4)
def _g5(g):
    D = 12.; h = 300.; foc = h * D / 1.85; cam = Camera3D((0, -D, .9), (0, 0, .9), focal=foc, cx=960, cy=540); F = Fig('frieza').pose('moon_up').place(0, 0, 0, heading=math.pi * .5 + .6); F.tail = lambda sk, ff: tail_curve(sk, g, kind='sway', amp=.9)
    L = F.draw(cam, g, light=(0, .95, -.1), pal='moon', rim=[dict(dir=(0, .95, -.2), col='#f0dcff', thr=.35, amt=1.)], key=.4).copy(); L[..., :3] *= np.array([.62, .58, .72], f32); return L
def g5_fig(f, x, y, h):
    k = h / 300.; L = _g5((f // 4) % 3 * 4); s = max(k, .02); M = np.float32([[s, 0, x - 960 * s], [0, s, y + h * .5 - 540 * s]]); n = int(max(1, 1 / s / 2)); Q = cv2.blur(L, (n, n)) if n > 1 else L; return cv2.warpAffine(Q, M, (W, HH), flags=cv2.INTER_AREA)


@lru_cache(90)
def g9_head(f, e, x, y, s):
    g = (f // 2) * 2
    if s < 150:
        D = 12.; foc = 8 * s * D / 1.58 * .42; cam = Camera3D((0, -D, .9), (0, 0, .9), focal=foc, cx=x, cy=y + .25 * s); F = Fig('frieza').pose('knees').place(0, 0, .25, heading=math.pi * .5 + .9); F.tail = lambda sk, ff: tail_curve(sk, 20, kind='wrap', amp=.5)
        L = F.draw(cam, 0, light=(0, .2, -.9), pal='moon', rim=[dict(dir=(.9, .2, -.3), col='#ffffff', thr=.36, amt=1.), dict(dir=(-.9, .2, -.3), col='#ffffff', thr=.36, amt=1.)], key=.6).copy(); L[..., :3] *= np.array([.50, .44, .66], f32); return L
    foc = s * .47 * 2 / .222 * 4; D = 4.; cam = Camera3D((0, -D, 1.40), (0, 0, 1.40), focal=foc, cx=x, cy=y)
    jit = (2.2 * math.sin(g * 2.3), 1.8 * math.cos(g * 1.7)) if e == 'rage' else (0, 0)
    F = Fig('frieza').pose('fists_down' if e == 'rage' else 'stagger' if e == 'disbelief' else 'stand').place(0, 0, 0, heading=math.pi * .5 + .85); F.look((-4, -2.5, 1.5))
    kw = dict(light=(-.95, .15, .25), pal='space', expr=dict(expr={'rage': 'rage', 'disbelief': 'disbelief', 'empty': 'empty'}[e], jaw=(.10 + .09 * ((g // 2) % 3) / 2) if e == 'rage' else (.07 if e == 'disbelief' else 0.)), key=.30)
    F.draw(cam, 0, **kw); q = F.info['head']; cam = Camera3D((0, -D, 1.40), (0, 0, 1.40), focal=foc, cx=x + jit[0] + (x - q['x']), cy=y + jit[1] + (y - q['y']))
    return F.draw(cam, 0, rim=[dict(dir=(-.95, .1, .1), col='#ffffff', thr=.30, amt=1.)], **kw)


def _limb(c, P, Wd, sk, sh, ln, lw=5., side=1):
    """a drawn limb along a centre line with widths: flat, one hard shadow plane, pen line"""
    from db.ink import spl, resample
    Q = resample(spl(np.asarray(P, float), n=14), 4.); t = np.linspace(0, 1, len(Q)); w = np.interp(t, np.linspace(0, 1, len(Wd)), Wd); w = np.convolve(np.pad(w, 12, mode='edge'), np.hanning(25) / np.hanning(25).sum(), 'valid')
    d = np.gradient(Q, axis=0); d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-6; n = np.stack([-d[:, 1], d[:, 0]], -1); A, B = Q + n * w[:, None], Q - n * w[:, None]; poly = np.vstack([A, B[::-1]]); c.fill(poly, sk)
    E = A if side > 0 else B; k = .18 + .16 * np.sin(t * 9.); c.shade(np.vstack([E, (Q + (E - Q) * k[:, None])[::-1]]), sh, poly); c.line(A, lw, ln, taper=(.15, .1), seed=3); c.line(B, lw, ln, taper=(.15, .1), seed=5)


def _h3_arm(L, f):
    g = f - 30; sm = lambda a, b: float(smooth(a, b, g)); tin, back = sm(3, 12), sm(14, 20); Wp = np.array([1470. + 560 * (1 - tin) + 330 * back, 790. + 60 * (1 - tin) + 30 * back]); S = np.array([1608., 838.]); a, b = 250., 235.; d = Wp - S; D = min(np.linalg.norm(d), a + b - 4); u = d / (np.linalg.norm(d) + 1e-6)
    h = (a * a - b * b + D * D) / (2 * D); k = math.sqrt(max(a * a - h * h, 0)); nrm = np.array([-u[1], u[0]]); E1, E2 = S + u * h + nrm * k, S + u * h - nrm * k; El = E1 if E1[1] > E2[1] else E2; c = Cel(W, HH)
    sk, sh, ln = '#f6c79f', '#c98c78', '#23161f'; sd = 1 if E1[1] > E2[1] else -1; t = np.linspace(0, 2 * np.pi, 40); ring = El + 70 * np.stack([np.cos(t), np.sin(t)], -1)
    _limb(c, [S, (S * 2 + El) / 3, (S + El * 2) / 3, El], [88, 100, 88, 72], sk, sh, ln, 5., sd); c.fill(ring, sk); c.line(ring[2:17], 5., ln, taper=(.2, .2)); _limb(c, [El, (El * 2 + Wp) / 3, (El + Wp * 2) / 3, Wp], [70, 84, 70, 52], sk, sh, ln, 5., sd)
    A = c.layer(); return A + L * (1 - A[..., 3:4])


_C3 = {}
def c3_child(f):
    """Gohan close, looking up right at the light (held drawing, trembles on twos)"""
    from chars import gohan as GH
    g = (f // 2) % 3
    if g not in _C3:
        G = GH.Gohan; k = (0, 1.2, -1.0)[g]
        L = G.bust(800 + k, 470 - .5 * k, 330, yaw=.38, pitch=.16, roll=-.05, expr='terror', light=_c3_light, f=0, wind=(.3, 0), hat=False)
        H_ = G.last[0]; q = H_.P([(.02, -.985)])[0]; L = L.copy(); grey = L[..., :3].mean(-1, keepdims=True); L[..., :3] = L[..., :3] * .55 + grey * .45 * np.array([.92, .88, 1.08], f32); _C3[g] = (L.astype(f32), (float(q[0]), float(q[1])))
    return _C3[g][0]
def c3_chin(f): c3_child(f); return _C3[(f // 2) % 3][1]
_c3_light = 'night'
