"""b3 close-up: a father's hands round a child's hands, a pearl of blue light between them.
    Nest(cx, cy, s).layer(t)      t 0 open .. 1 closed round the child's hands -> (premultiplied RGBA, dict(pearl=(x, y)))
    still(t=1.) -> finished composition (float RGB) used for qa/friends/lake.jpg
"""
from chars.fkit import *
from chars.gohan import Gohan, ball
from chars import gohan as G
from db import hands as HD
from db.paint import ramp, grid
from db.comp import over, screen, blur, radial, bokeh, glow

P_ = HD.pose
DAD = HD.build(len=1.02, wid=1.06, boxy=.52, tip=.5, knuckle=1.12, palm=1.04, wrist=1.1, meat=1.08)
GSKIN = ('#f6c79a', '#c98560', '#8c4f42')
ORANGE = ('#f47b20', '#b84f1c', '#7a2f18'); ORANGEH = '#ffa04a'; BLUE = ('#2c4f92', '#1b3468', '#101f44'); BLUEH = '#4a74c0'
OPEN_D = P_((8, 12, 8, -6), (10, 14, 8, 0), (14, 18, 10, 5), (18, 24, 12, 12), thumb=(60, 8, 36, 0, -4), cup=.2)
CUP_R = P_((30, 30, 16, -4), (35, 34, 18, 0), (38, 40, 20, 4), (44, 44, 22, 9), thumb=(62, 8, 40, 0, -6), cup=.35)
CUP_L = P_((27, 32, 18, -5), (33, 36, 18, 0), (40, 38, 20, 3), (43, 47, 22, 8), thumb=(64, 6, 42, 2, -4), cup=.35)
KR = P_((14, 24, 12, -8), (17, 27, 14, -1), (22, 31, 16, 5), (27, 36, 18, 13), thumb=(56, 22, 14, 6, 8), cup=.32)
KL = P_((17, 22, 14, -7), (16, 30, 14, 1), (24, 30, 16, 5), (25, 38, 20, 14), thumb=(52, 26, 10, 10, 12), cup=.32)


class Nest:
    def __init__(s, cx=960., cy=900., sc=236., res=640, G=None): s.cx, s.cy, s.sc, s.res, s.G = cx, cy, sc, res, np.eye(3) if G is None else G

    def hands(s, t=1., jig=0., kp=0., wrists=None):
        """t = how far the father's hands have come round.  wrists = {'R': px, 'L': px} of the body's forearms: the open hands sit exactly on them"""
        e = float(np.clip(t, 0, 1)); e = e * e * (3 - 2 * e); hs = []
        for q, sd, cup, da in ((-1, 'R', CUP_R, (2., 1.5, 0.)), (1, 'L', CUP_L, (-3., -2., 3.))):
            po = HD.blend(OPEN_D, cup, float(np.clip(t, 0, 1)), drag=.25, over=.06); m = lambda a_, b_: a_ + (b_ - a_) * e; p1 = A([q * .11, .02 * q, 0.])
            p0 = p1 + [q * .62, -.34, -.30] if wrists is None else A([(wrists[sd][0] - s.cx) / s.sc, -(wrists[sd][1] - s.cy) / s.sc, -.10])
            p1 = s.G @ p1; hs.append(HD.Hand(po, sd, s.G @ HD.euler(180 + da[0] - q * m(34, 0), m(74, 45) + da[1], -q * (m(12, 75) + da[2])), tuple(p0 + (p1 - p0) * e), DAD, pivot=(0, 0, 0), arm=.40))
        for q, sd, po, da in ((-1, 'R', KR, (3., 2., 1.)), (1, 'L', KL, (-2., -3., -2.))):
            p = HD.tremble(po, 7 + int(jig), 1.6 * kp) if kp > 0 else po; hs.append(HD.Hand(p, sd, s.G @ HD.euler(180 + da[0], 27 + da[1], -q * (41 + da[2])), tuple(s.G @ A([q * .080, .47 + .008 * q, .66])), Gohan.KID, scale=.62, pivot=(0, 0, 0), arm=.36))
        return hs

    def draw(s, cel, L, t=1., jig=0., kp=0., wrists=None, ldir=(0., .55, .80), w=4.6, seed=1):
        hs = s.hs = s.hands(t, jig, kp, wrists); g = HD.render(hs, HD.Cam(s.cx, s.cy, s.sc, 6.6), light=ldir, res=s.res, clip=(-200, -100, W + 200, FH + 120), cross=False); pd, pk = hpal(L, GSKIN), hpal(L, G.SKIN); pd = dict(HD.PAL['day'], **pd); pk = dict(HD.PAL['day'], **pk); pd['band'] = tone(L, BLUE, 0); pk['band'] = tone(L, G.SASH, 0)
        wp = {i: np.mean([g.pts[i]['palm'], g.pts[i]['wrist']], 0) if 'wrist' in g.pts[i] else g.pts[i]['palm'] for i in (2, 3)}; g.bands = [b for b in g.bands if b['hand'] < 2 or _near(b, g.pts[b['hand']], s.sc)]
        for i in range(4): HD.ink(cel, HD.only(g, i), pd if i < 2 else pk, w if i < 2 else w * .78, seed + i, marks=1. if i < 2 else .55)
        s.g = g; a = [np.mean([g.pts[i][('middle', 'tip')] for i in (2, 3)], 0), np.mean([g.pts[i]['palm'] for i in (2, 3)], 0)]; s.pearl = a[1] * .42 + a[0] * .58; return g


def _near(b, pts, sc):
    P = [np.asarray(v, float) for v in b.values() if isinstance(v, np.ndarray) and v.ndim == 2] + [np.asarray(q, float) for v in b.values() if isinstance(v, list) for q in v if np.ndim(q) == 2]
    if not P: return True
    c = np.mean(np.vstack(P), 0); return c[1] > pts['palm'][1] + .12 * sc


def _ori(y, zb):
    """hand orientation from its finger direction and the direction its back faces"""
    y = HD.unit(A(y, float)); z = A(zb, float); z = HD.unit(z - y * float(z @ y)); return np.stack([np.cross(y, z), y, z], -1)


CUP2 = P_((13, 22, 12, -7), (15, 25, 13, -1), (18, 28, 15, 5), (22, 32, 16, 12), thumb=(46, 24, 12, 8, 10), cup=.25)          # fingers curved round a ball of air
CUP3 = P_((20, 26, 14, -4), (23, 29, 15, 0), (26, 32, 16, 4), (30, 36, 18, 9), thumb=(44, 26, 10, 8, 12), cup=.3)


OPEN2 = P_((10, 14, 8, -3), (12, 16, 9, 0), (15, 19, 10, 2), (18, 23, 12, 5), thumb=(50, 10, 28, 0, -4), cup=.2)                 # fingers side by side: nothing splays
KA = P_((16, 24, 13, -3), (18, 27, 14, -.5), (21, 30, 15, 2), (25, 34, 16, 4.5), thumb=(46, 24, 12, 8, 10), cup=.28)
KB = P_((30, 36, 20, -2), (32, 39, 21, 0), (35, 42, 22, 1.5), (39, 46, 23, 3.5), thumb=(42, 28, 14, 10, 12), cup=.36)
CUP4 = P_((32, 34, 18, -2), (35, 37, 19, 0), (38, 40, 20, 1.5), (42, 44, 22, 3.5), thumb=(44, 26, 12, 8, 12), cup=.36)


class Cup:
    """b3 close, second version: two small palms facing each other with the pearl between them, the father's two palms outside: (( o ))
    G = turn of the whole group; (cx, cy) = the pearl, sc = px per adult hand unit"""
    KS = .64

    def __init__(s, cx, cy, sc, G=None, res=640): s.cx, s.cy, s.sc, s.G, s.res = cx, cy, sc, np.eye(3) if G is None else G, res

    def hands(s, t, jig=0, kp=0.):
        e = float(np.clip(t, 0, 1.1)); hs = []; G = s.G
        for q, sd, dy, tw in ((-1, 'R', .02, .05), (1, 'L', -.03, -.03)):                                                           # father: from outside and below, open, then round the boy's
            m = lambda a_, b_: a_ + (b_ - a_) * e; po = HD.blend(OPEN2, CUP4, float(np.clip(t, 0, 1)), drag=.2, over=.05); y = (q * m(.10, -.02) + tw, m(.55, .80), m(.72, .52)); z = (q * m(.80, .97), m(-.50, -.10), m(-.15, .20)); hs.append(HD.Hand(po, sd, G @ _ori(y, z), tuple(G @ A([q * m(1.58, 1.06 if q < 0 else .92), m(-.62, -.12) + dy, m(-.10, -.50 if q < 0 else -.16)])), DAD, pivot=(0, .56, -.18), arm=.20))
        for q, sd, dy, tw in ((-1, 'R', .0, .03), (1, 'L', .02, -.05)):
            p = HD.blend(KA, KB, float(np.clip(t, 0, 1)), drag=.2, over=.05); p = HD.tremble(p, 7 + int(jig) + (q > 0), 1.4 * kp) if kp > 0 else p; hs.append(HD.Hand(p, sd, G @ _ori((q * .10 + tw, .90, .40), (q * .95, -.14, .28)), tuple(G @ A([q * (.46 - .06 * e), -.05 + dy, .06])), Gohan.KID, scale=s.KS, pivot=(0, .58, -.16), arm=.12))
        return hs

    def draw(s, cel, L, t=1., jig=0., kp=0., ldir=(0., .35, .90), w=4.4, seed=1):
        hs = s.hs = s.hands(t, jig, kp); g = s.g = HD.render(hs, HD.Cam(s.cx, s.cy, s.sc, 6.6), light=ldir, res=s.res, clip=(-200, -100, W + 200, FH + 120)); pd, pk = dict(HD.PAL['day'], **hpal(L, GSKIN)), dict(HD.PAL['day'], **hpal(L, G.SKIN))
        for i in range(4): HD.ink(cel, HD.only(g, i), pd if i < 2 else pk, w if i < 2 else w * .78, seed + i, marks=1. if i < 2 else .55)
        s.wr = [HD.Cam(s.cx, s.cy, s.sc, 6.6).px(h.world((0, -.02, 0))) for h in hs]; return g


# ------------------------------------------------------------------ the lake behind, out of focus
def lake_bg(seed=4):
    x, y = grid(FH, W); r = np.random.default_rng(seed); n = cv2.resize(cv2.GaussianBlur(r.standard_normal((FH // 12, W // 12)).astype(f32), (0, 0), 4), (W, FH), interpolation=cv2.INTER_CUBIC); n /= n.std()
    img = ramp(np.clip(y / FH + .03 * n, 0, 1), [(0, '#fff3bf'), (.14, '#f3dd8c'), (.30, '#c9cf72'), (.36, '#7fa45c'), (.44, '#6a9a62'), (.50, '#bcc98a'), (.62, '#7fae9a'), (.82, '#4d8a84'), (1, '#2f6468')])
    cel = Cel(W, FH, ss=1)
    for j in range(46):                                            # far trees: soft masses, light on their left
        bx, by, br = r.uniform(-60, W + 60), r.uniform(250, 420), r.uniform(60, 150); k = r.random(); cel.fill(ell(bx, by, br, br * .72), mixc('#4f7f4c', '#86a552', k), .8); cel.fill(ell(bx - br * .3, by - br * .25, br * .55, br * .36), mixc('#a9bf62', '#e8dc86', k), .55)
    img = over(img, np.stack([blur(cel.rgba()[..., k], 16) for k in range(4)], -1))
    cel = Cel(W, FH, ss=1)
    for j in range(240):                                           # water: long horizontal dabs of sky and reed green
        v = r.random(); yy = 520 + 600 * v ** 1.2; ln = r.uniform(60, 260) * (.5 + v); xx = r.uniform(-100, W + 100); cel.fill(ell(xx, yy, ln, 3 + 12 * v), r.choice(['#e9e0a0', '#9cc4a0', '#5d9a8e', '#3d7a78', '#f7ecb8', '#74a878']), r.uniform(.25, .7))
    img = over(img, np.stack([blur(cel.rgba()[..., k], 7) for k in range(4)], -1))
    b = np.zeros((FH, W, 4), f32); cel = Cel(W, FH, ss=1)
    for j in range(150):                                           # sun glints: bokeh discs
        v = r.random(); cel.fill(ell(r.uniform(0, W), 470 + 420 * v, 2.2 + 3 * v), '#fff6c8', r.uniform(.5, 1.))
    for j in range(70): cel.fill(ell(r.uniform(0, W), r.uniform(0, 420), r.uniform(3, 6)), r.choice(['#fff4b8', '#e9f0a0', '#ffe28c']), r.uniform(.4, .9))
    b = bokeh(cel.rgba(), 15, 0); img = img + b[..., :3] * 2.6
    return np.clip(img, 0, 1.4).astype(f32)


def father(t=1., f=0, breathe=0.):
    """hero3 Goku behind the child, arms coming round; returns (whole layer, arms-only mask)"""
    from db.hero3 import Fig
    from db.cam import Camera3D
    cam = Camera3D((.0, -1.75, 1.62), (.0, 0, 1.240), 3050.); F = Fig('goku').pose('stand', gR='fist', gL='fist').place(0, 0, .004 * breathe, heading=math.pi + .045); e = float(np.clip(t, 0, 1)); e = e * e * (3 - 2 * e); o = (1 - e) * .13; z = 1.105 - .02 * (1 - e)
    F.reach(R=dict(p=(-.034 - o, -.42, z), part='wrist', w=1, body=.10, pole=(-1, .1, -.25)), L=dict(p=(.034 + o, -.42, z - .008), part='wrist', w=1, body=.10, pole=(1, .1, -.25)))
    L = F.draw(cam, f, light=(-.62, .55, .45), pal=dict(base='gold'), rim=dict(dir=(-.9, .5, -.3), col='#fff3b8', thr=.45, amt=.9), head=False)
    u = F.mask('uarm'); nl, lb, st, ce = cv2.connectedComponentsWithStats((u > .5).astype(np.uint8))      # the undershirt covers the armpit: recolour the bare patch inside the sleeve
    for k in range(1, nl):
        if st[k, 4] > 400 and ce[k][1] < 400:
            q = cv2.GaussianBlur((lb == k).astype(f32), (0, 0), 1.0)[..., None]; lu = (L[..., :3] @ A([.3, .59, .11], f32))[..., None]; c = np.where(lu > .62, rgb('#375a82'), np.where(lu > .40, rgb('#28455f'), rgb('#132033'))) * smooth(.06, .22, lu); L[..., :3] = L[..., :3] * (1 - q) + c * L[..., 3:4] * q
    hk = cv2.GaussianBlur(F.mask('hand'), (0, 0), .8)[..., None]; lu = (L[..., :3] @ A([.3, .59, .11], f32))[..., None]; L[..., :3] = L[..., :3] * (1 - hk) + np.where(lu > .5, rgb('#a8400f'), rgb('#6e2410')) * smooth(.06, .22, lu) * L[..., 3:4] * hk
    m = np.clip(F.mask('uarm') + F.mask('farm') + F.mask('band') + F.mask('hand'), 0, 1); return L, m, F


E_LOOK = dict(G.E['wonder'], gy=-.078, gx=.012, U=[.0, .118, .156, .132, .036], dB=.060, glint=1.15)
_bg = None


def still(t=1., f=0, glow_=1., jig=0., expr=None, kp=0., hat=True, head=(.16, -.20, .07)):
    global _bg
    if _bg is None: _bg = lake_bg()
    img = _bg.copy(); L = light('pearl', mul=((.98, .92, .88), (.82, .78, .86), (.66, .66, .82)), rim=[((4, 3), '#fff0b4', .85)]); expr = expr or dict(E_LOOK)
    gl, am, F = father(t, f); gl = para(gl, (520, 40), (1400, 1080), [(0, '#fff3d2'), (.5, '#f6dccb'), (1, '#8f9cc4')], 'mul', .8)
    bl = np.stack([cv2.GaussianBlur(gl[..., k], (0, 0), 1.8) for k in range(4)], -1); img = over(img, bl)
    cx, cy, sc = 966., 300., 158.
    # the child stands inside the arms: soft occlusion on the father's chest first
    sh = radial(cx - 20, cy + 190, 420, 430, 1.4); img = img * (1 - .38 * sh[..., None] * (1 - rgb('#7a3a3a')) * gl[..., 3:4])
    lay = Gohan.bust(cx, cy, sc, head[0], head[1], head[2], expr, L, f, (0., 0.), 3, hat, [(-1, (-1.02, 1.25, .1), (-.45, 2.2, .8), 'fist', (0, 0, 0), 0), (1, (1.02, 1.25, .1), (.45, 2.2, .8), 'fist', (0, 0, 0), 0)], rims=True, glow=.9, bottom=5.4, byaw=.05, ballon=False)
    N_ = Nest(960, 936, 236); cel = Cel(W, FH); N_.draw(cel, L, t, jig, kp, F.info['wrists']); hl = cel.layer(); px, py = N_.pearl; py -= 34
    x, y = grid(FH, W); d = np.sqrt((x - px) ** 2 + (y - py) ** 2)
    up = smooth(-40, 420, y - cy + 150) * np.exp(-(d / 460.) ** 2); lay[..., :3] = lay[..., :3] + (lay[..., 3] * up * .30 * glow_)[..., None] * rgb('#7fd0ff') * (.34 + lay[..., :3]) * 1.15; img = over(img, lay)
    arms = gl * am[..., None]; img = over(img, arms)                  # the father's arms close in front of him
    k_ = np.exp(-(d / 330.) ** 1.6) * .20 * glow_; img = img + (k_ * am)[..., None] * rgb('#69c6ff') * (.2 + img)
    sh = np.clip(blur(shifted(hl[..., 3], 0, 34), 22) * .42, 0, 1) * (1 - hl[..., 3]); img = img * (1 - sh[..., None] * (1 - rgb('#5a2c34')))
    # hands: the pearl lights the palms in two hard zones, backs fall into warm shade
    hl = para(hl, (px, py), (px, FH + 140), [(0, '#eef4ff'), (.30, '#ffeada'), (.70, '#dcae94'), (1, '#8d6a78')], 'mul', .95)
    th_ = np.arctan2(y - py, x - px)
    for r_, a_, c_2 in ((300, .55, '#bfe2ff'), (165, .60, '#d9f2ff')): z_ = (smooth(r_ + 8, r_ - 8, d + 9 * np.sin(th_ * 7) + 5 * np.sin(th_ * 17)) * a_ * .42 * glow_)[..., None]; hl[..., :3] = hl[..., :3] * (1 - z_) + np.minimum(hl[..., :3] * 1.0 + .10, 1.) * rgb(c_2) * hl[..., 3:4] * z_ * 1.12
    hl = rim(hl, 5, 5, '#fff0b8', .8, .6, where=smooth(330, 480, d).astype(f32)); img = over(img, hl); img = pearl(img, px, py, 30 * (.55 + .45 * t), f, glow_)
    if hat:
        from db import ball as B_
        H = Gohan.make(cx, cy, sc, *head); bx, by, br = G.ball_at(H); up_ = A([math.sin(head[2]), -math.cos(head[2])]); im2 = B_.draw(img.astype(f32), bx, by, br, t=f, light=(-.65, .6, .5), body=1.1, env=None)
        keep = np.clip(((x - bx) * up_[0] + (y - by) * up_[1] + br * .80) * 1. + .5, 0, 1)[..., None].astype(f32); img = img * (1 - keep) + im2 * keep
    return img


def father_off(t, f=0, hd=.42, breathe=0., wrists=None, k=1., at=(960., 540.)):
    """off-axis father: turned to screen right; hands reach round the child"""
    from db.hero3 import Fig
    from db.cam import Camera3D
    cam = Camera3D((.0, -1.75, 1.60), (.0, 0, 1.235), 2960. * k, cx=(960 - at[0]) * k + 960, cy=(540 - at[1]) * k + 540); h = math.pi + hd; Fw = A([-math.sin(h), math.cos(h)]); Rt = A([Fw[1], -Fw[0]]); p = A([-.105, .05]); F = Fig('goku').pose('stand', gR='fist', gL='fist').place(p[0], p[1], .004 * breathe, heading=h); e = float(np.clip(t, 0, 1)); o = .118 + (1 - e) * .125; z = 1.085 - .050 * (1 - e)
    r_, l_ = np.r_[p + Fw * .43 + Rt * o, z], np.r_[p + Fw * .43 - Rt * o, z - .006]; pr, pl = Rt, -Rt
    if wrists is not None:                                         # the forearms end exactly where the drawn hands begin
        un = lambda px, yy: (lambda d: cam.eye + d * ((yy - cam.eye[1]) / d[1]))(cam.R.T @ A([(px[0] - cam.cx) / cam.f, (px[1] - cam.cy) / cam.f, 1.])); r_, l_ = un(wrists[0], p[1] + Fw[1] * .43 - .03), un(wrists[1], p[1] + Fw[1] * .43 + .06)
    F.hand = dict(L=.06, R=.06)
    F.reach(R=dict(p=tuple(r_), part='wrist', w=1, body=.10, pole=(pr[0] - Fw[0] * .1, pr[1] - Fw[1] * .1, -.25)), L=dict(p=tuple(l_), part='wrist', w=1, body=.10, pole=(pl[0] - Fw[0] * .1, pl[1] - Fw[1] * .1, -.25)))
    L = F.draw(cam, f, light=(-.62, .55, .45), pal=dict(base='gold'), rim=dict(dir=(-.9, .5, -.3), col='#fff3b8', thr=.45, amt=.9), head=False)
    hk = cv2.GaussianBlur(F.mask('hand'), (0, 0), .8)[..., None]; lu = (L[..., :3] @ A([.3, .59, .11], f32))[..., None]; L[..., :3] = L[..., :3] * (1 - hk) + np.where(lu > .5, rgb('#a8400f'), rgb('#6e2410')) * smooth(.06, .22, lu) * L[..., 3:4] * hk
    return L, np.clip(F.mask('uarm') + F.mask('farm') + F.mask('band') + F.mask('hand'), 0, 1), F


E_TRY = dict(G.E['determined'], gy=-.085, gx=.02, U=[-.006, .080, .116, .104, .030], B=[(.130, .205), (.255, .262), (.395, .318), (.520, .318)], hw=.052, my=-.660, mU=[-.004, .010, .014, .010, -.004], mL=[-.004, -.012, -.016, -.012, -.004], blush=.35, furrow=.6)
E_WON = dict(E_LOOK, gy=-.070, U=[.012, .160, .212, .176, .050], dB=.085)
E_JOY = dict(E_LOOK, hw=.150, my=-.610, mU=[.040, .012, -.002, .012, .044], mL=[.030, -.085, -.112, -.085, .034], tongue=.8, teeth=.03, jaw=.03, blush=.5, dB=.065)
_mem = {}


PUSH = [(0, 1.12), (2, 1.135), (5, 1.22), (8, 1.34), (11, 1.42), (15, 1.45), (30, 1.52)]           # the lens goes in on the hands, 1.35x by 11
def push(f): return float(comp.lerp_keys(f, PUSH, lambda t: t)), A([float(comp.lerp_keys(f, [(0, 966.), (3, 968.), (11, 990.), (30, 996.)])), float(comp.lerp_keys(f, [(0, 566.), (3, 570.), (11, 648.), (30, 658.)]))])


def off_layers(q):
    """heavy drawings of the close shot, one per pair of frames (q even), each drawn at its own size: father, child, hands"""
    if q in _mem: return _mem[q]
    if len(_mem) > 3: _mem.clear()
    Z, C0 = push(q); k = Z / 1.045; M = lambda p: (A(p, float) - C0) * k + [960., 540.]
    K = comp.lerp_keys; t = float(K(q, [(0, .46), (2, .43), (4, .50), (6, .72), (8, .94), (10, 1.05), (12, .98), (14, 1.02), (18, 1.), (30, 1.)], lambda t: t)); L = light('pearl', mul=((.98, .92, .88), (.82, .78, .86), (.66, .66, .82)), rim=[((4, 3), '#fff0b4', .85)])
    N_ = Cup(*M((1036, 776)), 202 * k, HD.euler(20, 0, -3), res=720); hw_ = N_.hands(t); gl, am, F = father_off(t, q, breathe=math.sin(q * .3), wrists=[HD.Cam(N_.cx, N_.cy, N_.sc, 6.6).px(h.world((0, -.12, 0))) for h in hw_[:2]], k=k, at=C0); gl = para(gl, tuple(M((420, 40))), tuple(M((1300, 1080))), [(0, '#fff3d2'), (.5, '#f6dccb'), (1, '#8f9cc4')], 'mul', .8)
    e = E_TRY if q < 12 else E_WON if q < 18 else (E_WON, E_JOY, float(np.clip((q - 16) / 6., 0, 1)));  e = G.ex(e) if isinstance(e, tuple) else e
    if q in (6,): e = dict(e, closed=1, arc=0)
    bob = float(K(q, [(0, 0.), (10, 3.), (12, -9.), (15, 3.), (20, 0.), (30, 2.)])); head = (.40, float(K(q, [(0, -.30), (11, -.33), (13, -.20), (20, -.22), (30, -.18)])), float(K(q, [(0, .04), (12, .10), (22, .13), (30, .11)]))); cx, cy = M((880., 366. + bob)); sc = 152. * k
    g = .40 + .22 * math.sin(q * .42) + .10 * math.sin(q * .9 + 1); ch = Gohan.bust(cx, cy, sc, *head, e, L, q, (-g, .10 + .04 * math.sin(q * .6)), 3, True, [], rims=True, glow=.9, bottom=5.4, byaw=.34, ballon=False); Bd = Gohan.last[1]
    cel = Cel(W, FH); N_.draw(cel, L, t, q // 2, .9 if q < 12 else .2, w=4.4 * k); px, py = np.mean([N_.g.pts[2]['palm'], N_.g.pts[3]['palm']], 0) + A([2, -38]) * k; H = Gohan.make(cx, cy, sc, *head)
    ac = Cel(W, FH); wl = lw(sc)                                       # his arms: elbows back, wrists exactly on the drawn hands
    for i_, qq, el in ((3, 1, (1.16, 1.42, .35)), (2, -1, (-.62, 1.72, .30))): Gohan.put_arm(ac, Bd, L, qq, el, el, None, np.eye(3), wl, 5 + qq, hand_=False, wrpx=N_.wr[i_], wd=(.25, .22, .19, .20, .17))
    al = grade(ac.layer(), L, (cx - 1.6 * sc, cy - 1.6 * sc, cx + 1.6 * sc, cy + 3.6 * sc), .95); ch = lover(ch, al)
    _mem[q] = dict(gl=gl, am=am, ch=ch, hl=cel.layer(), p=(float(px), float(py)), ball=G.ball_at(H), roll=head[2], t=t, k=k, C0=C0, c=(float(cx), float(cy - bob * k)), Z=Z); return _mem[q]


def off_close(f):
    """b3 0-29: off-axis, the lens pushing in on the hands.  a spark that fails (4-8), big hands close round (cupped by 10), the pearl is born on 12"""
    global _bg
    if _bg is None: _bg = lake_bg()
    K = comp.lerp_keys; D_ = off_layers((f // 2) * 2); k = D_['k']; px, py = D_['p']; x, y = grid(FH, W); d = np.sqrt((x - px) ** 2 + (y - py) ** 2) / k; dx = float(K(f, [(0, 26.), (30, -22.)])); sb = 1.06 * (1 + .40 * (k - 1.07)); bg = cv2.warpAffine(_bg, np.float32([[sb, 0, 960 * (1 - sb) + dx], [0, sb, 540 - sb * 547]]), (W, FH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    gg = float(K(f, [(0, 0.), (11.9, 0.), (12, 1.5), (15, .85), (18, 1.05), (30, 1.)])) * (1 + .06 * math.sin(f * 1.1)); weak = float(K(f, [(0, 0), (3, 0), (4, .5), (5, .2), (6, .42), (8, 0.), (30, 0)])) if f < 12 else 0.
    g_ = max(gg, weak); gl, am, ch, hl = D_['gl'], D_['am'], D_['ch'], D_['hl']; img = over(bg, np.stack([cv2.GaussianBlur(gl[..., j], (0, 0), 1.9) for j in range(4)], -1))
    cx, cy = D_['c']; sh = radial(cx - 30 * k, cy + 190 * k, 400 * k, 420 * k, 1.4); img = img * (1 - .38 * sh[..., None] * (1 - rgb('#7a3a3a')) * gl[..., 3:4])
    up = smooth(-60, 380, (y - cy) / k + 120) * np.exp(-(d / 460.) ** 2); c2 = ch.copy(); c2[..., :3] += (ch[..., 3] * up * .36 * g_)[..., None] * rgb('#7fd0ff') * (.34 + ch[..., :3]); img = over(img, c2)
    img = over(img, gl * am[..., None]); k_ = np.exp(-(d / 330.) ** 1.6) * .20 * g_; img = img + (k_ * am)[..., None] * rgb('#69c6ff') * (.2 + img)
    sh = np.clip(blur(shifted(hl[..., 3], 0, int(34 * k)), 22 * k) * .42, 0, 1) * (1 - hl[..., 3]); img = img * (1 - sh[..., None] * (1 - rgb('#5a2c34')))
    h2 = para(hl, (px, py), (px, FH + 140), [(0, '#eef4ff'), (.30, '#ffeada'), (.70, '#dcae94'), (1, '#8d6a78')], 'mul', .45 + .5 * min(1, g_)); th_ = np.arctan2(y - py, x - px)
    for r_, a_, c_2 in ((300, .55, '#bfe2ff'), (165, .60, '#d9f2ff')): z_ = (smooth(r_ + 8, r_ - 8, d + 9 * np.sin(th_ * 7) + 5 * np.sin(th_ * 17)) * a_ * .42 * g_)[..., None]; h2[..., :3] = h2[..., :3] * (1 - z_) + np.minimum(h2[..., :3] + .10, 1.) * rgb(c_2) * h2[..., 3:4] * z_ * 1.12
    img = over(img, rim(h2, 5, 5, '#fff0b8', .8, .6, where=smooth(330, 480, d).astype(f32)))
    if weak > 0: img = spark(img, px, py + 6 * k, (9 + 9 * weak) * k, f)
    if f >= 12: r = float(K(f, [(12, 14.), (13, 43.), (15, 25.), (17, 33.), (20, 30.), (30, 32.)])) * (1 + .05 * math.sin(f * 1.1)) * k; img = pearl(img, px, py, r, f, min(1.5, gg))
    if f in (12, 13): img = flash_ring(img, px, py, (90 + 150 * (f - 12)) * k, 1 - .45 * (f - 12))
    bx, by, br = D_['ball']
    if by > -br * 1.1:
        from db import ball as B_
        up_ = A([math.sin(D_['roll']), -math.cos(D_['roll'])]); im2 = B_.draw(img.astype(f32), bx, by, br, t=f, light=(-.65, .6, .5), body=1.1, env=None); keep = np.clip((x - bx) * up_[0] + (y - by) * up_[1] + br * .80 + .5, 0, 1)[..., None].astype(f32); img = img * (1 - keep) + im2 * keep
    Z, C = push(f); sx, sy = 1.6 * math.sin(f * .31), 1.2 * math.sin(f * .23 + 1); c = (C - D_['C0']) * k + [960., 540.]; return comp.cam2d(img.astype(f32), c[0] + sx, c[1] + sy, Z / k, -.034)


def spark(img, x, y, r, f):
    """the first try: a few blue crumbs that go out"""
    cel = Cel(W, FH); rr = np.random.default_rng(f // 2 + 3)
    for j in range(7):
        a = rr.uniform(0, TAU); l = r * rr.uniform(.6, 1.8); c = A([math.cos(a), math.sin(a)]); n = A([-c[1], c[0]]) * r * .12; cel.fill(A([c * r * .3 + n, c * l, c * r * .3 - n]) + [x, y], '#a8e2ff', .9)
    cel.fill(ell(x, y, r * .42), '#d9f4ff'); cel.fill(ell(x, y, r * .22), '#ffffff'); return over(screen(img, np.clip(radial(x, y, r * 6, power=2.2) * .30, 0, 1)[..., None] * rgb('#4fb6ff')), cel.rgba())


def flash_ring(img, x, y, r, a):
    cel = Cel(W, FH); t = np.linspace(0, TAU, 90); w_ = 13 * a * (1 + .3 * np.sin(t * 5)); cel.fill([np.c_[x + np.cos(t) * (r + w_), y + np.sin(t) * (r + w_) * .86], np.c_[x + np.cos(t) * (r - w_), y + np.sin(t) * (r - w_) * .86]], '#d6f2ff', .55 * a, evenodd=True); return over(img, cel.rgba())


def cast(img, lay, dx, dy): return img


def pearl(img, x, y, r, f=0, g=1.):
    """the pearl: white core, hard blue shell, a few drawn rays and motes that hang round it"""
    img = screen(img, np.clip(radial(x, y, r * 13, power=2.8) * .34 * g, 0, 1)[..., None] * rgb('#4fb6ff')); img = screen(img, np.clip(radial(x, y, r * 5, power=1.8) * .55 * g, 0, 1)[..., None] * rgb('#8fdcff'))
    cel = Cel(W, FH); rr = np.random.default_rng(5); ph = f * .22
    for j in range(16):
        a = j * TAU / 16 + ph * (.3 if j % 2 else -.2) + rr.normal(0, .06); l0, l1 = r * 1.28, r * rr.uniform(1.7, 3.1) * (1 + .12 * math.sin(f * .9 + j)); n = A([-math.sin(a), math.cos(a)]) * r * .07; c = A([math.cos(a), math.sin(a)]); cel.fill(A([c * l0 + n, c * l1, c * l0 - n]) + [x, y], '#d8f4ff', .85)
    for j in range(22):
        a = rr.uniform(0, TAU) + ph * rr.uniform(-.5, .5); q = r * rr.uniform(2., 7.); k = rr.uniform(1.6, 4.2); cel.fill(ell(x + math.cos(a) * q, y + math.sin(a) * q * .8 - ((f * rr.uniform(.4, 1.2)) % 30), k), '#e6f8ff', rr.uniform(.5, 1.))
    cel.fill(ell(x, y, r * 1.10), '#58bcff', .95); cel.fill(ell(x, y, r * .96), '#b8ecff'); cel.fill(ell(x - r * .06, y - r * .06, r * .74), '#ffffff'); t = np.linspace(.5, 2.4, 14); cel.fill(np.vstack([np.c_[x + np.cos(t) * r * .92, y + np.sin(t) * r * .92], np.c_[x + np.cos(t) * r * .76 - r * .05, y + np.sin(t) * r * .74 - r * .06][::-1]]), '#7cd0ff', .8)
    return over(img, cel.rgba()) + np.clip(radial(x, y, r * 2.2, power=1.4), 0, 1)[..., None] * rgb('#8fd8ff') * .5 * g


if __name__ == '__main__':
    import sys
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 1.
    img = still(t); img = comp.finish(img, 0, bloom=.32, vig=.14); comp.save('qa/friends/lake.jpg', img, 93)
