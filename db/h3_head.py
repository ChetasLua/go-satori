"""Heads for db.hero3: hands the posed head to the head lanes (chars/goku, vegeta, frieza, gohan, piccolo) and draws a small clean head of its own for far figures."""
import numpy as np, math, cv2, inspect, importlib
from .paint import rgb, f32
from .ink import Cel, spl, resample
from . import kit
from .head3 import Head
from .h3_draw import _mixc, draw_contour

PI = math.pi
A_ = [(-1.0, .14), (-.965, .25), (-.89, .37), (-.76, .50), (-.58, .59), (-.36, .645), (-.12, .675), (.12, .70), (.35, .71), (.6, .685), (.8, .55), (.93, .34), (1.0, .05)]
KID = [(-1.0, .15), (-.97, .23), (-.90, .345), (-.80, .475), (-.68, .595), (-.54, .70), (-.36, .78), (-.12, .81), (.12, .81), (.35, .80), (.6, .745), (.8, .60), (.93, .38), (1.0, .05)]
PROP = dict(goku=dict(a=A_), vegeta=dict(a=A_, nose=(.10, -.04, -.43, -.49, .18)), piccolo=dict(a=A_), gohan=dict(a=KID, nose=(.07, -.2, -.385, -.425, .085)), kid=dict(a=KID, nose=(.07, -.2, -.385, -.425, .085)),
            frieza=dict(a=[(-1.0, .10), (-.965, .20), (-.89, .31), (-.74, .44), (-.54, .545), (-.30, .605), (-.05, .645), (.2, .69), (.42, .73), (.62, .735), (.8, .63), (.93, .40), (1.0, .06)], nose=(.06, -.14, -.40, -.45, .115)))
HAIR = dict(black=('#4a5680', '#1d1b27', '#0c0b12', '#08070c'), gold=('#fff6b0', '#ffd95a', '#e08f22', '#7a3d0c'), silver=('#ffffff', '#dfe5f0', '#93a3c2', '#39425e'))
_heads = {}


def head_for(char):
    if char not in _heads: _heads[char] = Head(**PROP.get(char, {}))
    return _heads[char]


def pose_info(cam, sk):
    """screen pose of the head: x, y (eye-line centre), s (px per unit), z, u, R (head -> view), yaw, pitch, roll"""
    c = sk.head; p, z = cam.project(c); t = cam.eye - c; D = np.linalg.norm(t); t = t / D
    up = -cam.R[1]; r = np.cross(up, t); r /= np.linalg.norm(r) + 1e-9; u = np.cross(t, r); M = np.stack([-sk.Rh[:, 0], sk.Rh[:, 2], sk.Rh[:, 1]], 1); R = np.stack([r, u, t]) @ M
    pitch = math.asin(float(np.clip(-R[2, 1], -1, 1))); yaw = math.atan2(-R[2, 0], R[2, 2]); cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    Q = R @ np.array([[cy, 0, sy], [-sp * sy, cp, sp * cy], [-cp * sy, -sp, cp * cy]]).T; roll = -math.atan2(Q[1, 0], Q[0, 0]); s = cam.f / max(float(z), .05) * sk.u
    return dict(x=float(p[0]), y=float(p[1]), s=float(s), z=float(z), u=sk.u, R=R, yaw=yaw, pitch=pitch, roll=roll, persp=float(min(.42, sk.u / D)), view=np.stack([r, u, t]), H=None, world=c)


def posed(char, hi):
    H = head_for(char).pose(hi['x'], hi['y'], hi['s'], hi['yaw'], hi['pitch'], hi['roll'], hi['persp']); H.R = hi['R']; return H


# ------------------------------------------------------------------ lanes
LMAP = dict(
    goku=dict(day='day', dawn='dawn', lamp='lamp', night='moon', moon='moon', dark='moon', fire='lamp', beam='gold', gold='gold', rain='rain', storm='rain', space='ui', ki='ui', ui='ui', grey='rain'),
    vegeta=dict(day='day', dawn='dawn', lamp='lamp', night='night', moon='night', dark='night', fire='lamp', beam='beam', gold='beam', rain='rain', storm='rain', space='night', ki='night', grey='rain'),
    frieza=dict(day='gold', dawn='gold', night='moon', moon='moon', dark='back', back='back', fire='gold', beam='beam', gold='gold', rain='moon', storm='moon', space='space', ki='space', white='white', grey='space'),
    fkit=dict(day='day', dawn='dawn', lamp='gold', night='night', moon='night', dark='night', fire='gold', beam='gold', gold='gold', rain='rain', storm='rain', space='night', ki='night', grey='day'))


def _hex(c):
    c = np.clip(np.asarray(c, float), 0, 1) * 255 + .5; return '#%02x%02x%02x' % tuple(int(v) for v in c[:3])


def match_light(char, pal, lc, skin, name):
    """light set-up of the head lane rebuilt from this figure's own palette and light, so that head and body are one drawing"""
    sk = [_hex(c) for c in pal(skin)]; mul = lambda h, m: _hex(rgb(h) * m + pal.add * .5); l = np.asarray(lc, float); l = l / np.linalg.norm(l); az = math.degrees(math.atan2(abs(l[0]), l[2])); ink = pal.ink
    try:
        if char == 'goku':
            from chars import goku_light as GL
            base = LMAP['goku'].get(name, 'day'); L = dict(GL.LIGHT[base if base != 'ui' else 'day']); L.update(skin=(sk[1], sk[2], sk[3], sk[0]), ldir=tuple(float(v) for v in l), side=1 if l[0] > 0 else -1, amt=float(np.interp(az, [0, 20, 50, 80, 110, 150], [-1., -.8, -.35, .15, .6, 1.2])), rim=[], para=[], line=ink, white=(mul('#fdfaf4', pal.mul), mul('#c9c2cc', pal.shmul)), brow=mul('#17151d', pal.mul))
            L.pop('hdir', None); L['hair'] = {k: (mul(v, pal.mul if k in ('base', 'light', 'hi') else pal.shmul) if isinstance(v, str) and v.startswith('#') else v) for k, v in GL.BLACK.items()}; return L
        if char == 'vegeta':
            from chars import vegeta as V
            L = dict(V.LIGHTS[LMAP['vegeta'].get(name, 'day')]); L.update(sk=sk[1], sh=sk[2], dp=sk[3], hi=sk[0], ln=ink, rim=None, key=tuple(float(v) for v in l), eye=mul('#fbf7f1', pal.mul), hr=mul('#1f1e2a', pal.mul), hs=mul('#0c0b12', pal.shmul), hl=mul('#4a5c86', pal.litmul)); return L
        if char == 'frieza':
            fv = [_hex(c) for c in pal('fv')]; return dict(sk=sk[1], sh=sk[2], dp=sk[3], hi=sk[0], ln=ink, dm=fv[1], ds=fv[2], dh=_hex(_mixc(pal('fv')[0], '#ffffff', .6)), d2=fv[0], eye=mul('#f8f4ff', pal.litmul), iris='#e0202e', lip=mul('#120c1a', pal.mul), rim=None, key=tuple(float(v) for v in l))
        from chars import fkit as FK
        L = dict(FK.LIGHT[LMAP['fkit'].get(name, 'day')]); L.update(mul=(tuple(pal.mul), tuple(pal.shmul), tuple(pal.shmul)), ld=(float(l[0]), float(l[1])), line=ink, rim=[], white=(mul('#ffffff', pal.mul), mul('#cdd3e6', pal.shmul)), side=float(np.interp(az, [0, 30, 60, 90, 130], [-1., -.6, -.35, 0., .8]))); return L
    except Exception: return None


def _try(names, *a):
    for n in names:
        if n in a[0]: return n
    return next(iter(a[0]))


def lane(char):
    """head_fn of the head lane for this character, or None"""
    try:
        if char == 'goku':
            m = importlib.import_module('chars.goku'); fn = lambda cel, H, i: m.head_fn(cel, H, i); fn.parts = True; return fn
        if char == 'gohan':
            m = importlib.import_module('chars.gohan'); fn = lambda cel, H, i: m.Gohan.head(cel, H, i); fn.parts = True; return fn
        if char == 'piccolo':
            m = importlib.import_module('chars.piccolo'); fn = lambda cel, H, i: m.Piccolo.head(cel, H, i); fn.parts = False; return fn
        if char == 'vegeta':
            m = importlib.import_module('chars.vegeta')
            def fn(cel, H, i): m.Vegeta.of(H, **{k: i[k] for k in ('expr', 'light', 'f', 'wet', 'wind', 'scratch', 'hair_drop') if k in i}).draw(cel, neck=False)
            fn.parts = False; return fn
        if char == 'frieza':
            m = importlib.import_module('chars.frieza')
            def fn(cel, H, i): m.Frieza.of(H, **{k: i[k] for k in ('expr', 'light', 'f', 'glow') if k in i}).draw(cel, neck=False)
            fn.parts = False; return fn
    except Exception as e:
        return None
    return None


DEF_EXPR = dict(goku='serious', vegeta='pride', frieza='serene', piccolo='stern', gohan='neutral', kid='smile')


def render(s, hi, roi):
    """-> (back layer or None, head layer), premultiplied RGBA of the roi"""
    nx0, ny0, nx1, ny1 = roi; char = s.fig.char; H = posed(char, hi); hi['H'] = H; sp = s.B.sp; e = dict(s.expr or {}); lag = hi['view'] @ np.asarray(s.fx.get('hair', np.zeros(3)), float)
    hairk = sp.get('hair', char); style = e.get('hair', dict(goku='base', ui='ui', ssj='ssj', vssj='ssj').get(hairk, 'base'))
    lightname = e.get('light', s.fig.light_name); fam = char if char in ('goku', 'vegeta', 'frieza') else 'fkit'
    info = dict(hi, f=s.fx['f'], roi=roi, fig=s.fig, pal=s.pal, wo=s.wo, lightv=s.lc, seed=s.seed, hair=style, wind=(float(lag[0]), float(-lag[1])) if char in ('goku', 'gohan', 'piccolo') else float(min(1.2, np.hypot(lag[0], lag[1]))), expr=e.get('expr', DEF_EXPR.get(char, 'calm')),
                light=LMAP[fam].get(lightname, 'day' if fam != 'frieza' else 'moon'), turban=sp.get('turban', False), hat=sp.get('hat', False), wet=e.get('wet', 0.), dmg=e.get('dmg', 0.), glow=e.get('glow', 0.), scratch=e.get('scratch', 0.))
    info.update({k: v for k, v in e.items() if k not in info or k in ('hair', 'light', 'expr', 'wind', 'f', 'seed')})
    if 'light' not in e:
        ml = match_light(char, s.pal, s.lc, sp['skin'], lightname)
        if ml is not None: info['light'] = ml
    fn = s.head_fn
    if fn == 'own': fn = None
    elif fn == 'lane' or (fn is None and hi['s'] >= s.fig.lane_min): fn = lane(char)
    elif not callable(fn): fn = None
    s.fig.head_info = info
    def cel_of(): return Cel(nx1 - nx0, ny1 - ny0, nx0, ny0, ss=2 if hi['s'] > 14 else 3)
    if fn is not None:
        try:
            np_ = sum(q.default is inspect._empty and q.kind in (0, 1) for q in inspect.signature(fn).parameters.values())
            if np_ >= 3:
                back = None
                if getattr(fn, 'parts', False):
                    c = cel_of(); fn(c, H, dict(info, part='back')); back = c.rgba(); c = cel_of(); fn(c, H, dict(info, part='front'))
                else: c = cel_of(); r = fn(c, H, info)
                return back, c.rgba()
            L = fn(H, info)
            if L is not None: return None, L if L.shape[:2] == (ny1 - ny0, nx1 - nx0) else _crop(L, roi)
        except Exception as ex:
            if s.head_fn is not None and s.head_fn != 'lane': raise
            s.fig.head_error = repr(ex)
    if hi['s'] < 3.2: return speck(s, hi, style, cel_of)
    return own(s, hi, H, roi, style, info, cel_of)


def speck(s, hi, style, cel_of):
    """a head of a few pixels: two dabs"""
    c = cel_of(); a = np.linspace(0, 2 * PI, 15)[:-1]; r = max(.8, hi['s']); x, y = hi['x'], hi['y']; u = hi['R'][:2, 1] * [1, -1]; ch = s.fig.char; sk = s.pal(s.B.sp['skin'])
    c.fill(np.c_[x + r * .75 * np.cos(a), y + r * np.sin(a)], sk[1]); h = s.pal('fv')[1] if ch == 'frieza' else sk[1] if ch == 'piccolo' else rgb(HAIR['gold' if style == 'ssj' else 'silver' if style == 'ui' else 'black'][1]) * s.pal.mul
    if ch != 'piccolo': q = np.array([x, y]) + u * r * (.75 if ch != 'frieza' else .55); k = 1.05 if ch != 'frieza' else .7; c.fill(np.c_[q[0] + r * k * np.cos(a), q[1] + r * k * .95 * np.sin(a)], h)
    return None, c.rgba()


def _crop(L, roi):
    x0, y0, x1, y1 = roi; out = np.zeros((y1 - y0, x1 - x0, 4), f32); h, w = L.shape[:2]; a0, b0, a1, b1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if a1 > a0 and b1 > b0: out[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = L[b0:b1, a0:a1]
    return out


# ------------------------------------------------------------------ own small head
def surf(H, th, y):
    a = np.interp(y, H.yy, H.A); f = np.interp(y, H.yy, H.F); b = np.interp(y, H.yy, H.B); p = H.k['p']; sx, cz = math.sin(th), math.cos(th); r = (abs(sx) ** p + abs(cz) ** p) ** (-1 / p); d = f if cz >= 0 else b
    n = np.array([sx / max(a, .18), (y - .10) * .95, cz / max(d, .18)]); n /= np.linalg.norm(n); return np.array([a * sx * r, y, d * cz * r]), n


def blades(H, char, style, hairk):
    """[(control points 4x3, half width, kind)] kind f fringe, o other"""
    out = []
    if char in ('goku', 'kid', 'gohan'):
        try:
            from chars import goku_hair as GH
            D = GH.get(style if char == 'goku' else 'base')
        except Exception: D = {}
        if char == 'gohan':
            D = {}
            for i, (th, y, tip) in enumerate([(-30, .80, (-.62, .42, .78)), (-8, .82, (-.18, .36, .98)), (14, .82, (.24, .38, .96)), (36, .80, (.64, .40, .74)), (-70, .55, (-1.02, .30, -.05)), (70, .55, (1.02, .30, -.05)), (-60, .85, (-.95, 1.05, -.1)), (60, .85, (.95, 1.0, -.1)), (0, .97, (-.1, 1.42, -.1)), (-110, .6, (-1.05, .55, -.7)), (110, .6, (1.05, .55, -.7)), (150, .6, (.5, .6, -1.25)), (-150, .6, (-.5, .6, -1.25)), (180, .9, (0, 1.2, -.95)), (180, .1, (0, -.25, -1.15)), (140, .1, (.7, -.25, -.95)), (-140, .1, (-.7, -.25, -.95)), (-95, .15, (-1.0, -.15, -.25)), (95, .15, (1.0, -.15, -.25))]):
                r, n = surf(H, math.radians(th), y); tip = np.array(tip, float); L = np.linalg.norm(tip - r); d = tip - r; d /= np.linalg.norm(d); out.append((np.stack([r, r + (n * .5 + d * .5) * .38 * L, tip - d * .36 * L, tip]), .22, 'f' if i < 4 else 'o'))
            return out
        for k, q in D.items():
            r, n = surf(H, math.radians(q['th']), q['y']); tip = np.asarray(q['tip'], float)
            if char == 'kid': tip = tip * [1.05, .98, 1.05]
            L = np.linalg.norm(tip - r); d = tip - r; d = d / (np.linalg.norm(d) + 1e-9); g = n * q['out'] + d * (1 - q['out']); g /= np.linalg.norm(g) + 1e-9
            out.append((np.stack([r - n * .05, r + g * q['k1'] * L, tip - q['d'] * q['k2'] * L, tip]), q['w'] * .46, 'f' if k[0] == 'F' else 'o'))
    elif char == 'vegeta':
        try:
            from chars import vegeta as V
            T = [(b, 'o') for b in V.TOP + V.BACK] + [(b, 'f') for b in V.FRONT]
        except Exception: T = []
        for (r, t, b, rad), kd in T:
            r, t, b = np.array(r, float), np.array(t, float), np.array(b, float)
            if style == 'ssj': t = t * [1, 1.06, .9]
            m = (r + t) / 2 + b; out.append((np.stack([r, r + (m - r) * .67, t + (m - t) * .67, t]), rad, kd))
    return out


def own(s, hi, H, roi, style, info, cel_of):
    char = s.fig.char; px = hi['s']; R = hi['R']; pal = s.pal; ink = s.ink; lc = s.lc; fz = R[2, 2]; cel = cel_of(); bk = cel_of(); sp = s.B.sp
    lw = min(s.wo, max(.8, px * .085)); lo, li, lf = lw * .9, lw * .62, lw * .5; skin = pal(sp['skin']); hk = 'gold' if style == 'ssj' else 'silver' if style == 'ui' else 'black'
    hc = [rgb(c) for c in HAIR[hk]]; hl, hb, hs, hn = hc[0] * pal.litmul + pal.add, hc[1] * pal.mul + pal.add, hc[2] * pal.shmul + pal.add * .6, hc[3]
    ld = np.array([lc[0], -lc[1]]); ld = ld / (np.hypot(*ld) + 1e-6); lag = hi['R'].T @ (hi['view'] @ np.asarray(s.fx.get('hair', np.zeros(3)), float))
    BL = []
    t = np.linspace(0, 1, 13)[:, None]; pf = np.interp(t[:, 0], [0, .08, .35, .7, 1], [.55, .92, 1., .55, .0])
    for i, (C4, hw, kd) in enumerate(blades(H, char, style, sp.get('hair'))):
        L = np.linalg.norm(C4[3] - C4[0]); C4 = C4.copy(); C4[3] += lag * L * .42; C4[2] += lag * L * .20
        Q = (1 - t) ** 3 * C4[0] + 3 * (1 - t) ** 2 * t * C4[1] + 3 * (1 - t) * t * t * C4[2] + t ** 3 * C4[3]; p = H.P3(Q); T = np.gradient(p, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; w = (hw * px * pf)[:, None]
        dz = H.depth(Q); n0 = R[2] @ (C4[0] / (np.linalg.norm(C4[0]) + 1e-9)); BL.append(dict(P=np.vstack([p + N * w, (p - N * w)[::-1]]), A=p + N * w, B=p - N * w, S=p, dp=float(dz.mean()), n0=float(n0), kd=kd, i=i, N=N))
    back = sorted([b for b in BL if b['n0'] < -.12 and b['kd'] != 'f' and b['dp'] < .0], key=lambda b: b['dp']); ids = {b['i'] for b in back}; front = sorted([b for b in BL if b['i'] not in ids and not (b['kd'] == 'f' and fz < -.1)], key=lambda b: b['dp'])
    def put(c, T, full=True):
        for b in T: kit.outline(c, b['P'], lo, ink, seed=b['i'])
        for b in T:
            c.fill(b['P'], hb)
            if px > 6:
                dk = b['A'] if (b['A'][6] - b['B'][6]) @ ld < 0 else b['B']; lt = b['B'] if dk is b['A'] else b['A']; S = b['S']
                c.fill(np.vstack([dk, (S + (dk - S) * .05)[::-1]]), hs)
                if px > 14: a, z = 3, 10; c.fill(np.vstack([(S + (lt - S) * .82)[a:z], (S + (lt - S) * .45)[a:z][::-1]]), hl, .9 if hk != 'black' else .75)
            if px > 9:
                for E_ in (b['A'], b['B']): c.line(E_[3:], lf, hn if hk == 'black' else _mixc(hn, ink, .3), taper=(.6, .05), seed=b['i'], tmin=.1)
    tgt = bk if fz > -.2 else cel
    put(tgt, back)
    o_ = H.P3(np.array([(0., 0., 0.), (0., -1., 0.), (0., 1., 0.)])); dn = (o_[1] - o_[0]) / (np.hypot(*(o_[1] - o_[0])) + 1e-6)
    sk_ = H._sil([H._mesh(-.52, 1 if char in ('frieza', 'piccolo') else .80, 0)])
    fr = char == 'frieza'
    if fr or char == 'piccolo':
        kit.outline(cel, sk_, lo, ink, seed=3)
    cel.fill(sk_, skin[1] if fz > -.1 or fr or char == 'piccolo' else hb if char in ('goku', 'vegeta', 'kid', 'gohan') else skin[1])
    for e in (-1, 1):
        E = H.ear(e)
        if E['vis'] > .05 and not fr:
            ear = E['outer']
            if char == 'piccolo': ear = H.P3(np.array([(e * .68, .02, -.10), (e * 1.05, .42, -.42), (e * .74, -.28, -.16), (e * .66, -.36, -.06)]))
            kit.shape(cel, ear, skin[1] if (e * ld[0]) > 0 else skin[2], lo * .75, ink, seed=7 + e)
        if fr and E['vis'] > -.1: kit.shape(cel, H.P3(np.array([(e * .64, .10, -.02), (e * .80, .06, -.06), (e * .82, -.20, -.10), (e * .66, -.30, -.06)])), skin[1] if (e * ld[0]) > 0 else skin[2], lo * .75, ink, seed=7 + e)
    if fz > -.30:
        if hi['pitch'] > -.02: cel.fill(H.under(), skin[2])
        F = H.face(); cel.fill(F, skin[1]); cel.ctx.save(); cel._path([F, sk_]); cel.ctx.set_fill_rule(0); cel.ctx.clip()
        for q_ in face_shadow(H, lc, .10): cel.fill(q_, skin[2])
        if lc[1] > .1 and char not in ('frieza', 'piccolo'): cel.fill(H.P([(-.66, .80), (-.62, .40), (-.44, .50), (-.28, .34), (-.12, .48), (.04, .36), (.22, .50), (.36, .36), (.48, .48), (.64, .38), (.66, .8)]), skin[2])
        cel.ctx.restore()
        poly = sk_.astype(np.float32)[:, None]; inside = np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), True) for x, y in F]) > .045 * px; vis = ~inside; r_ = int(np.argmin(vis)) if (~vis).any() else 0; Fr = np.roll(F, -r_, 0); vr = np.roll(vis, -r_)
        for a_, b_ in _runs(vr, 5):
            seg = Fr[a_:b_]
            if len(seg) > 4: cel.line(spl(seg[::max(1, len(seg) // 40)], n=4), lo * .95, ink, taper=(.14, .14), press=.25, seed=11)
        for e in (-1, 1):
            if H.ear(e)['vis'] > .10 and abs(hi['yaw']) > .18 and e * hi['yaw'] < 0: cel.line(spl(H.jaw(e)[2:], n=5), li * .9, ink, taper=(.65, .10), seed=15 + e)
        if abs(hi['yaw']) > .75: cel.line(H.P3(H.nose_profile()[1:8]), lo * .75, ink, taper=(.45, .15), seed=5)
        marks(cel, H, char, px, li, lf, ink, skin, hi, pal, info, hk, ld)
    if fr or char == 'piccolo': bald(cel, H, char, px, lo, li, lf, ink, skin, pal, info, ld, sk_, fz)
    put(cel, front)
    if char == 'gohan' and sp.get('hat'): hat(cel, H, px, lo, ink, pal, ld)
    return bk.rgba() if back and fz > -.2 else None, cel.rgba()


def face_shadow(H, lv, thr=.06):
    y = np.linspace(-1, .78, 40); u = np.linspace(-1, 1, 34); a = np.interp(y, H.yy, H.A)[:, None]; X = u[None] * a * .995; Y = np.repeat(y[:, None], 34, 1); Z = H.z(X, Y); e = .012
    zx = (H.z(np.clip(X + e, -a, a), Y) - H.z(np.clip(X - e, -a, a), Y)) / (2 * e); zy = (H.z(X, Y + e) - H.z(X, Y - e)) / (2 * e); n = np.stack([-zx, -zy, np.ones_like(Z)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True); l = np.asarray(lv, float); l = H.R.T @ (l / np.linalg.norm(l))
    m = (n @ l) < thr; P = H.P3(np.stack([X, Y, Z], -1)); q = np.stack([P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]], 2).reshape(-1, 4, 2)[(m[:-1, :-1] & m[1:, 1:]).ravel()]
    if len(q) < 4: return []
    k = max(.8, min(4., 120. / max(H.s, 6))); lo = q.reshape(-1, 2).min(0) - 4; hi = q.reshape(-1, 2).max(0) + 4; w, h = np.ceil((hi - lo) * k).astype(int) + 2; M = np.zeros((h, w), np.uint8)
    for c in np.round((q - lo) * k * 16).astype(np.int32): cv2.fillConvexPoly(M, c, 255, cv2.LINE_8, 4)
    M = cv2.morphologyEx(cv2.morphologyEx(M, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); out = []
    for c in kit.contours(M / 255., min_len=16):
        v = cv2.approxPolyDP(c.astype(np.float32)[:, None], 120. * .035, True)[:, 0] / k + lo
        if len(v) >= 3: out.append(v)
    return out


MOOD = dict(serious='firm', pride='firm', serene='calm', stern='firm', neutral='calm', calm='calm', smile='smile', grin='grin', laugh='shout', determined='rage', shout='shout', rage='rage', roar='shout', scream='shout', scream_a='shout', scream_b='shout', scream_d='shout', grit='rage', pain='rage', fire='rage', cruel='grin', contempt='calm', fear='shout', disbelief='shout', shock='shout', scorn='grin', peace='shut', shut='shut', closed='shut', ego='firm', move='shout', surprise='shout', empty='sad', grief='sad')


def marks(cel, H, char, s, li, lf, ink, skin, hi, pal, info, hk, ld):
    x = info.get('expr'); mood = MOOD.get(x if isinstance(x, str) else 'serious', 'firm'); am = dict(firm=-.035, calm=-.005, smile=.01, grin=-.05, sad=.045, rage=-.075, shout=-.06, shut=0.).get(mood, -.02); op = dict(shout=1.15, rage=.95, grin=.85, calm=.9, shut=.0).get(mood, 1.)
    iris = dict(gold='#2aa08a', silver='#b4c0d8').get(hk, '#1a1418') if char != 'frieza' else '#d81e2c'
    kid = char in ('gohan', 'kid'); k = 1.6 if kid else 1.
    for e in (-1, 1):
        if H.facing(.325 * e, 0.) < .10: continue
        X = lambda L: H.P(spl(np.array([(e * x, y) for x, y in L]), True, n=4), .006)
        if s < 7: cel.line(H.P([(e * .16, -.02), (e * .5, .06)]), max(.9, li * 1.2), ink, taper=(.2, .2)); continue
        if mood == 'shut': cel.line(H.P(spl(np.array([(e * .18, -.02), (e * .33, -.05), (e * .50, .0)]), n=5), .006), max(1., li * 1.3), ink, taper=(.2, .2)); continue
        W = X([(.155, -.02 * k), (.22, .035 * op * k), (.335, .062 * op * k), (.43, .05 * op * k), (.50, .01 * k), (.41, -.048 * k), (.28, -.052 * k)]); cel.fill(W, rgb('#f7f3ee') * pal.mul + pal.add); cel.clip(W)
        cx, a = .315, np.linspace(0, 2 * PI, 14)[:-1]; cel.fill(H.P(np.c_[e * cx + .062 * np.cos(a) * (1.1 if kid else 1), .002 + .070 * k * np.sin(a)], .006), rgb(iris))
        if s > 18 and not kid and hk != 'black': cel.fill(H.P(np.c_[e * cx + .024 * np.cos(a), .002 + .028 * np.sin(a)], .006), ink)
        if s > 20: cel.fill(H.P(np.c_[e * cx - .02 + .014 * np.cos(a), .030 * k + .016 * np.sin(a)], .006), '#ffffff')
        cel.unclip(); th = .026
        cel.fill(X([(.135, -.024 * k), (.215, .038 * op * k + th * .4), (.335, .070 * op * k + th), (.44, .058 * op * k + th), (.545, .014 * k + th * .5), (.43, .045 * op * k), (.335, .055 * op * k), (.225, .026 * op * k)]), ink)
        if char not in ('frieza', 'piccolo'):
            bc = ink if hk == 'black' else rgb(HAIR[hk][2]); y0 = .05 * (k - 1)
            cel.fill(X([(.09, .085 + am + y0), (.11, .125 + am + y0), (.22, .165 + am * .7 + y0), (.38, .220 + am * .1 + y0), (.56, .150 - am * .3 + y0), (.38, .172 + am * .1 + y0), (.23, .120 + am * .7 + y0)]), bc)
        else: cel.line(H.P(spl(np.array([(e * .07, .04 + am), (e * .22, .12 + am * .6), (e * .42, .19), (e * .58, .14)]), n=5), .008), max(1., li * 1.5), skin[3] if char == 'frieza' else ink, taper=(.05, .5))
    np_ = H.nose_profile(); sg = 1 if ld[0] < 0 else -1
    if s > 10 and abs(hi['yaw']) < 1.0: cel.fill(H.P3(np.array([np_[4] + [0, .03, 0], np_[5] + [sg * .02, -.01, -.01], np_[5] + [sg * .07, -.012, -.07], np_[4] + [sg * .05, .02, -.05]])), skin[2])
    my = -.665
    if H.facing(0, my) > .02 and s > 7:
        lip = ink if char != 'frieza' else rgb('#120c1a')
        if mood in ('shout', 'grin', 'rage'):
            M = dict(grin=[(-.22, my + .05), (-.1, my + .01), (0, my + .005), (.1, my + .01), (.22, my + .06), (.13, my - .045), (0, my - .055), (-.13, my - .04)], shout=[(-.16, my + .05), (0, my + .085), (.16, my + .05), (.19, my - .10), (.1, my - .18), (-.1, my - .18), (-.19, my - .10)], rage=[(-.2, my - .01), (-.14, my + .045), (.14, my + .045), (.2, my - .01), (.18, my - .085), (-.18, my - .085)])[mood]
            P = H.P(spl(np.array(M), True, n=4)); cel.fill(P, rgb('#4a1420') * pal.mul if mood == 'shout' else rgb('#f2eae0') * pal.mul + pal.add)
            if mood == 'shout' and s > 12: cel.clip(P); cel.fill(H.P([(-.2, my + .12), (.2, my + .12), (.17, my + .035), (-.17, my + .035)]), rgb('#f2eae0') * pal.mul); cel.unclip()
            kit.outline(cel, P, lf * 1.3, lip, seed=3)
        else: cel.line(H.P(spl(np.array([(-.14, my - .012 + (.03 if mood == 'smile' else 0)), (0, my + (.0 if mood != 'smile' else -.02)), (.15, my - .012 + (.04 if mood == 'smile' else 0))]), n=6)), max(.9, li * (1.1 if char != 'frieza' else 1.7)), lip, taper=(.3, .3))


def bald(cel, H, char, s, lo, li, lf, ink, skin, pal, info, ld, sk_, fz):
    th = np.linspace(-PI, PI, 61)
    if char == 'frieza':
        edge = np.interp(np.abs(th), [0, .14, .34, .58, .85, 1.15, 1.45, 1.9, 2.5, 3.15], [.335, .37, .47, .555, .58, .53, .44, .34, .28, .25]); quads = []
        for a, b, e0, e1 in zip(th[:-1], th[1:], edge[:-1], edge[1:]):
            ys = np.linspace(0, 1, 9)
            for y0, y1 in zip(ys[:-1], ys[1:]):
                Q = [surf(H, a, e0 + (1 - e0) * y0), surf(H, b, e1 + (1 - e1) * y0), surf(H, b, e1 + (1 - e1) * y1), surf(H, a, e0 + (1 - e0) * y1)]
                if sum((H.R @ q[1])[2] for q in Q) / 4 > -.05: quads.append(H.P3(np.array([q[0] for q in Q])))
        if not quads: return
        k = max(1., min(4., 100. / max(s, 6))); P = np.array(quads); lo_ = P.reshape(-1, 2).min(0) - 3; hi_ = P.reshape(-1, 2).max(0) + 3; w, h = np.ceil((hi_ - lo_) * k).astype(int) + 2; m = np.zeros((h, w), np.uint8)
        for c in np.round((P - lo_) * k * 16).astype(np.int32): cv2.fillConvexPoly(m, c, 255, cv2.LINE_8, 4)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)); fv = pal('fv')
        for c in kit.contours(m / 255., min_len=16):
            D = draw_contour(c / k + lo_, max(.4, s * .02), True, .6, n=4); cel.fill(D, fv[1]); cel.clip(D); ctr = D.mean(0); r = s * .5
            cel.fill(D - ld * s * .0 + (D - ctr) * 0, fv[1]); cel.fill(D * .70 + (ctr - ld * r * .5) * .30 - ld * r * .25, fv[2])
            q = ctr + ld * r * .55; a = np.linspace(0, 2 * PI, 14)[:-1]; cel.fill(np.c_[q[0] + r * .26 * np.cos(a), q[1] + r * .20 * np.sin(a)], _mixc(fv[0], '#ffffff', .7)); cel.unclip(); kit.outline(cel, D, li, ink, seed=9)
    else:
        sp_t = info.get('turban')
        if sp_t:
            D = H._sil([H._mesh(.22, 1, 0) * [1.10, 1.06, 1.10]]); kit.shape(cel, D, pal('cape')[1], lo, ink, seed=5); cel.clip(D); cel.fill(D - ld * s * .35, pal('cape')[2], .0)
            for y in (.35, .52, .70): cel.line(H.P3(np.array([(x * 1.08, y + .10 * x, .9 * math.sqrt(max(0, 1 - (x / .72) ** 2))) for x in np.linspace(-.70, .70, 9)])), lf, ink, taper=(.2, .2))
            cel.unclip()
        else:
            for e in (-1, 1):
                r0 = np.array([e * .16, .74, .66]); cel.line(H.P3(np.array([r0, r0 + [e * .06, .30, .16], r0 + [e * .14, .34, .42], r0 + [e * .20, .16, .55]])), max(1.2, s * .05), skin[1], taper=(.1, .6)); cel.line(H.P3(np.array([r0, r0 + [e * .06, .30, .16], r0 + [e * .14, .34, .42], r0 + [e * .20, .16, .55]])) + [0, s * .02], lf, ink, taper=(.1, .4))


def hat(cel, H, s, lo, ink, pal, ld):
    D = H._sil([H._mesh(.50, 1, 0) * [1.12, 1.12, 1.12]]); r = pal('hat'); kit.shape(cel, D, r[1], lo, ink, seed=5); cel.clip(D); cel.fill(D - ld * s * .45, r[2], .0); cel.unclip()
    p = H.P3(np.array([[0, 1.22, .0]]))[0]; a = np.linspace(0, 2 * PI, 20)[:-1]; B = np.c_[p[0] + s * .26 * np.cos(a), p[1] + s * .26 * np.sin(a)]; kit.shape(cel, B, rgb('#f5a02c') * pal.mul, lo * .8, ink, seed=6)


def _runs(ok, n):
    out = []; a = None
    for i, v in enumerate(list(ok) + [False]):
        if v and a is None: a = i
        if not v and a is not None:
            if i - a >= n: out.append((a, i))
            a = None
    return out
