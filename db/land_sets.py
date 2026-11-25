"""The quiet places: cliff in the rain, puddle ground, Mount Paozu lake, crater lake.  Reached through land.py."""
import numpy as np, cv2, math, os
from .land_core import *
from .land_geo import *
from . import land_sky as SK, land_grow as GW
from .land import look, LOOKS, Look, mesa, _cr, coast_sky
from .ink import Cel, spl, resample
from .paint import warp_by

_s = {}


def hill(M, c, R, H, seed=0, mat=3, z0=-3., asp=1., rot=0., steep=.55, sides=None, n=11, cliff=0.):
    """rounded karst hill under forest"""
    r = np.random.default_rng(abs(int(seed)) * 101 + 7); m = sides or int(np.clip(16 + R * 1.1, 24, 84)); n = max(n, int(np.clip(H / 4.5, 8, 26))); a = (np.arange(m) + r.uniform(-.2, .2, m)) / m * 2 * np.pi; b = 1 + .12 * np.sin(a * 2 + r.uniform(0, 6)) + .09 * np.sin(a * 3 + r.uniform(0, 6)) + r.normal(0, .04, m); X = np.c_[np.cos(a) * b * asp, np.sin(a) * b]; cr, sr = math.cos(rot), math.sin(rot); X = X @ np.array([[cr, sr], [-sr, cr]])
    t = np.linspace(0, 1, n); s = (1 - t ** (1.4 + 2.0 * steep)) ** (.62); off = r.normal(0, .12, 2) * R
    rings = np.stack([np.c_[X * R * max(si, .02) * (1 + r.normal(0, np.clip(1.5 / R, .02, .12), (m, 1))) + np.asarray(c, float)[:2] + off * ti ** 1.5, z0 + (H - z0) * ti + r.normal(0, min(1.6, .05 * H), m) * (0 < ti < 1)] for ti, si in zip(t, s)])
    ao = np.clip(.45 + .75 * t, 0, 1)[:, None] * np.ones(m); body(M, rings, ao, top=.01 * H, mat=mat, seed=int(r.integers(8000)))


# ================================================================================================ E1: the cliff edge in the rain
CLIFF = 64.             # height of the cliff top; Vegeta stands at (0, 0, CLIFF)


def cliff_look(name='cliff'):
    if name not in LOOKS:
        LOOKS[name] = Look(ramp={0: ('#141618', '#232628', '#383b3c', '#575b5b', '#858886', '#c9cbc8'), 5: ('#34383a', '#4a4e50', '#626667', '#7c8080', '#989b9a', '#c0c2c0')}, light=(.55, .28, .79), bounce=(-.85, -.4, .2), fog=dict(col='#8f9291', col2='#a3a6a4', dens=.0030, start=10, hf=.0, floor=1., max=.93),
            par=dict(key=.34, amb=.09, sky=.13, side=.10, hue=.03, crack=1.), mats={0: dict(wet=.42, cell=.085, fcell=.06, chisel=1.5, edge=.8), 5: dict(cell=.8, chisel=.7)}, water=dict(scale=16., refl=.5, glint=0., foam=.9, stroke=1.5, distort=20, cols=('#3e4446', '#6b6f70', '#2c3133', '#b4b7b5', '#ffffff', '#c6c9c7', '#d7dad8')), wetcol='#b9bcba')
    return LOOKS[name]


def cliff_cam(push=0., f=1500.): return Camera3D((3.1 - .30 * push, -4.2 + .55 * push, CLIFF + .42), (-.15, 1.2, CLIFF + 1.36), f)


def _cliff():
    if 'cliff' in _s: return _s['cliff']
    r = np.random.default_rng(3); M = Mesh()
    foot = FOOT
    P = resample(np.vstack([foot, foot[:1]]), .42)[:-1]; m = len(P); P = P + r.normal(0, .06, P.shape); d = P - np.array([-3.5, .6]); d /= np.linalg.norm(d, axis=1, keepdims=True)
    lv = [(-26, -1.0), (-17, -.7), (-12, -.9), (-9, -.35), (-7.2, -.62), (-5.6, -.2), (-4.6, -.5), (-3.6, -.12), (-2.9, -.34), (-2.2, -.02), (-1.75, -.22), (-1.3, .06), (-.98, -.12), (-.66, .07), (-.44, -.05), (-.24, .05), (-.09, .05), (0., -.02)]
    n1 = fbm(len(lv) * 4, m, (9, 4), (1, .6), 5)[::4]; brk = np.zeros((len(lv), m)); 
    for _ in range(26): j0 = int(r.integers(5, len(lv) - 1)); i0 = int(r.integers(m)); brk[j0:j0 + int(r.integers(1, 4)), np.arange(i0, i0 + int(r.integers(1, 4))) % m] -= r.uniform(.05, .22)
    rings = np.stack([np.c_[P + d * (o + brk[j][:, None] + n1[j][:, None] * .16 * (1 if j < len(lv) - 2 else .3)), np.full(m, CLIFF + z) + r.normal(0, .05 if z < -.3 else .02, m)] for j, (z, o) in enumerate(lv)])
    ao = np.stack([np.clip((.45 + .55 * smooth(-12, -.2, np.full(m, z))) * (.5 if (o < -.15 and j < len(lv) - 1) else 1.), .2, 1) for j, (z, o) in enumerate(lv)]); body(M, rings, ao, top=None, mat=0, seed=41)
    rt = rings[-1]; cc = np.array([-4., .5, CLIFF]); k = [np.c_[cc[:2] + (rt[:, :2] - cc[:2]) * s, CLIFF + r.normal(0, .022, m) + .03 * (1 - s)] for s in (.93, .8, .62, .40, .18)]; body(M, np.stack([rt] + k), np.ones((6, m)), top=0., mat=0, seed=43)
    pebbles(M, 46, (-7, 1.1, -1.6, 1.8), lambda x, y: np.full_like(x, CLIFF), r, .03, .22, p=3., keep=lambda x, y: math.hypot(x, y) > .55 and cv2.pointPolygonTest((foot * .92).astype(np.float32), (float(x), float(y)), False) > 0)
    Gp = M.build(); M = Mesh()
    for (x, y, R, H, kd) in [(-110, 210, 30, 50, 'mesa'), (-190, 260, 44, 74, 'mesa'), (-74, 330, 15, 58, 'pillar'), (96, 420, 20, 46, 'mesa'), (150, 520, 13, 62, 'stack'), (-250, 520, 60, 96, 'mesa'), (40, 700, 24, 40, 'butte'), (230, 900, 60, 60, 'fin'), (-140, 800, 40, 70, 'mesa')]: mesa(M, (x, y), R, H, int(abs(x) * 7 + y), kd, mat=5)
    Gh = M.build(); _s['cliffp'], _s['cliffh'] = Gp, Gh; _s['cliff'] = cat(Gp, Gh); return _s['cliff']


def cliff(t=0., cam=None, sky=None, dyn=None, ss=2, time='cliff'):
    """e1: the wet promontory, sea far below, headlands in rain fog.  default camera = cliff_cam()"""
    cam = cam or cliff_cam(); G = _cliff(); L = cliff_look()
    if sky is None: sky = rain_sky(cam)
    if 'cliff_sh' not in _s: _s['cliff_sh'] = shadow_map(G, L.light, 2048, (-16, 3, -5, 8, CLIFF - 8, CLIFF + 2)); _s['cliff_fm'] = foam_map(G, (-420, 420, -60, 800))
    return render(cam, G, L, t, sky, _s['cliff_sh'], _s['cliff_fm'], (), (), ss, True, dyn=dyn)


def _puddles(o, cam, sky, amt=1., seed=5, scale=1., foot=None):
    """standing water in the hollows of the cliff top: mirrors the sky, dark wet rim"""
    d = cam.rays(); X = cam.eye[0] + d[..., 0] * o.z; Y = cam.eye[1] + d[..., 1] * o.z; Z = cam.eye[2] + d[..., 2] * o.z; h, w = cam.h, cam.w
    top = ((o.rock > 0) & (Z > CLIFF - .028) & (Z < CLIFF + .05)).astype(f32); top = cv2.GaussianBlur(cv2.erode(cv2.morphologyEx(top, cv2.MORPH_CLOSE, np.ones((5, 5))), np.ones((13, 25))), (0, 0), 1.5); N = fbm(400, 400, (46, 20, 8), (1, .55, .25), seed); n = cv2.remap(N, np.mod(X * 42 * scale + 900, 399).astype(f32), np.mod(Y * 42 * scale + 900, 399).astype(f32), cv2.INTER_LINEAR)
    if foot is not None:                                                             # no water on the bevel: at least a hand's width from the edge
        pm = np.zeros((1400, 1700), np.uint8); cv2.fillPoly(pm, [((np.asarray(foot) - [-12., -8.]) * 100).astype(np.int32)], 255); di = cv2.distanceTransform(pm, cv2.DIST_L2, 5) / 100.; top = top * smooth(.16, .30, cv2.remap(di, ((X + 12.) * 100).astype(f32), ((Y + 8.) * 100).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
    m = smooth(1.02, 1.10, n) * top; rim = smooth(.80, 1.02, n) * top * (1 - m); hy = horizon(cam); yy = np.clip(2 * hy - np.arange(h), 0, h - 1).astype(int); ref = cv2.blur(sky[yy], (15, 5)) * .96
    img = o.img * (1 - .38 * rim[..., None]); img = img + (ref - img) * (m * .86 * amt)[..., None]; o.puddle = m; e = np.clip(m - np.roll(m, 2, 0), 0, 1); return img + (C('#e4e6e3') - img) * (e * .5)[..., None]


def cliff_e1(cam=None, turn=.36, seed=3):
    """plates for e1 in the figure's own camera (Vegeta on (0, 0, CLIFF)): dict(back=(h,w,3) sky + far sea + headlands, cliff=premult RGBA of the promontory, puddle=(h,w) mask of standing water (mirror the figure in it), pts=[(x, y, scale)] where drops splash, hz=horizon row, cam=).
    lay them with your parallax K(back, .25), K(cliff, 1.); feet land at (1075, 890)"""
    key_ = ('e1', turn, None if cam is None else (tuple(np.round(cam.eye, 3)), round(cam.f, 1), tuple(np.round(np.ravel(cam.R), 4)), cam.cx, cam.cy))
    if key_ in _s: return _s[key_]
    from .cam import rot_z
    cam = cam or Camera3D((0, -6, CLIFF + .55), (0, 0, CLIFF + .62), 2420., cx=1075, cy=640); _cliff(); Gp = _s['cliffp'].moved(rot_z(turn), (0, 0, 0), (0, 0, CLIFF)); L = cliff_look(); hy = float(horizon(cam)); fs = 1250.; sky = rain_sky(Camera3D(cam.eye, cam.eye + [-.10, 1, 0.], fs, cx=1075, cy=hy)); sky = sky * np.array([.90, .905, .91], f32) - .02
    M = Mesh()
    for (x, y, R, H, kd, k) in [(-118, 330, 30, 40, 'mesa', 3), (-160, 400, 46, 66, 'mesa', 5), (-64, 420, 11, 46, 'pillar', 9), (175, 760, 18, 50, 'stack', 4), (230, 980, 50, 44, 'fin', 2), (60, 1200, 30, 30, 'butte', 6)]: mesa(M, (x, y), R, H, k * 37 + 5, kd, mat=5)
    Gh = M.build(); fm = foam_map(Gh, (-420, 420, 300, 1400)); b = render(cam, Gh, L, 0., sky, None, fm, (), (), 2, True); back = b.img; h, w = cam.h, cam.w; x, y = grid(h, w); tq = np.clip((y - hy) / (h - hy), 0, 1); back = np.where((y > hy)[..., None] & (b.a[..., None] > .5), back * (1 - .46 * tq ** .55)[..., None], back)
    sh = shadow_map(Gp, L.light, 2048, (-9, 3, -5, 5, CLIFF - 4, CLIFF + 1)); o = render(cam, Gp, L, 0., np.zeros((h, w, 3), f32), sh, None, (), (), 2, False, water=False); lay = np.dstack([o.rgb, o.a]).astype(f32)
    o.img = np.where(o.a[..., None] > .02, o.rgb / np.maximum(o.a, .02)[..., None], 0.); o.img = _puddles(o, cam, sky, foot=FOOT @ rot_z(turn)[:2, :2].T); lay[..., :3] = o.img * o.a[..., None]
    veil = np.exp(-np.abs(y - hy) / 16.) * .40; back = back + (C('#a4a7a6') - back) * veil[..., None]
    r = np.random.default_rng(seed); pts = []; tz = o.z
    ys, xs = np.nonzero((o.a > .95) & (np.abs(cam.eye[2] + (cam.rays()[..., 2] * o.z) - CLIFF) < .07))
    for k in r.choice(len(xs), 70): pts.append((float(xs[k]), float(ys[k]), float(np.clip(5.2 / o.z[ys[k], xs[k]], .35, 1.5))))
    _s[key_] = dict(back=back.astype(f32), cliff=lay, puddle=o.puddle, pts=pts, hz=hy, cam=cam, sky=sky); return _s[key_]


FOOT = np.array([(-16, -3.2), (-11, -2.7), (-7.5, -2.5), (-5.2, -1.9), (-3.4, -2.0), (-1.9, -1.45), (-.8, -1.5), (.35, -1.05), (.95, -.95), (1.35, -.35), (1.18, .25), (1.32, .62), (.62, 1.05), (-.3, 1.2), (-1.6, 1.75), (-4, 2.3), (-9, 3.4), (-16, 6.)])      # outline of the promontory, metres, Vegeta at (0, 0)


def _cliff_ground(turn):
    if ('cg', turn) in _s: return _s[('cg', turn)]
    from .cam import rot_z
    Rz = rot_z(turn)[:2, :2]; foot = FOOT @ Rz.T; P = resample(np.vstack([foot, foot[:1]]), .05)[:-1]; from scipy.spatial import cKDTree
    tree = cKDTree(P); inside = lambda q: np.array([cv2.pointPolygonTest(foot.astype(np.float32), (float(a), float(b)), False) > 0 for a, b in q])
    c = .11; xs = np.arange(-10.2, 2.6, c); ys = np.arange(-6.8, 3.6, c); r = np.random.default_rng(12); X, Y = np.meshgrid(xs, ys); X = X + r.uniform(-.3, .3, X.shape) * c; Y = Y + r.uniform(-.3, .3, Y.shape) * c; q = np.c_[X.ravel(), Y.ravel()]; ins = inside(q); d, k = tree.query(q); snap = ~ins & (d < c * 1.6); q[snap] = P[k[snap]]; ok = (ins | snap).reshape(X.shape); X, Y = q[:, 0].reshape(X.shape), q[:, 1].reshape(Y.shape); di = np.where(ins, d, 0.).reshape(X.shape)
    e_ = smooth(.03, .42, di); Z = puddle_z(X, Y) * e_ + .062 * (1 - e_) - .022 * (1 - smooth(0., .10, di)); V = np.stack([X, Y, Z], -1); a, b, cc, dd = V[:-1, :-1], V[:-1, 1:], V[1:, 1:], V[1:, :-1]; o4 = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, 1:] & ok[1:, :-1]
    M = Mesh(); s_ = (r.random(a.shape[:2]) < .5)[..., None, None]; T1 = np.where(s_, np.stack([a, b, cc], 2), np.stack([a, b, dd], 2))[o4]; T2 = np.where(s_, np.stack([a, cc, dd], 2), np.stack([b, cc, dd], 2))[o4]; M.add(np.concatenate([T1, T2]), ao=1., mat=1, seed=3)
    pebbles(M, 120, (-8, 1.2, -3.6, 1.6), puddle_z, r, .02, .20, p=3.4, keep=lambda x, y: float(puddle_z(x, y)) > -.02 and (math.hypot(x, y) > 1.0) and not (-3. < x < .4 and abs(y) < .45) and cv2.pointPolygonTest((foot * .95).astype(np.float32), (float(x), float(y)), False) > 0)
    for (x, y, sz) in [(-3.9, 1.15, .42), (-4.6, .75, .26), (-3.3, 1.30, .16), (-2.6, -1.95, .22), (-6.2, 1.2, .7)]: boulder(M, (x, y, float(puddle_z(x, y)) + sz * .25), sz, r, n=22)
    top = M.build(); _cliff(); W_ = _s['cliffp'].moved(rot_z(turn), (0, 0, -CLIFF + .045), (0, 0, CLIFF)); W_ = Geo(**W_.__dict__); cz = W_.T[..., 2].mean(1); k = ~((W_.T[..., 2].max(1) > -.02) & (W_.FN[:, 2] > .55)) & (cz < .045); W_ = W_.sel(k)
    g = cat(top, W_); m = np.zeros((1400, 1700), np.uint8); cv2.fillPoly(m, [((foot - [-12., -8.]) * 100).astype(np.int32)], 255); m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))); _s[('cg', turn)] = (g, m, shadow_map(g, cliff_look().light, 2048, (-7, 3, -5, 4, -2., 1.))); return _s[('cg', turn)]


def cliff_back(turn=.36, w=3000, h=1700, f=1560.):
    """far sea, headlands and rain sky of e1, seen level from the cliff top (wide plate for any camera standing there)"""
    k = ('cb', w, h)
    if k in _s: return _s[k]
    cam = Camera3D((.1, -3.6, CLIFF + .7), (.1, 100, CLIFF + .7), f, w, h); L = cliff_look(); hy = float(horizon(cam)); sky = rain_sky(Camera3D(cam.eye, cam.eye + [-.10, 1, 0.], f * .62, w, h)); sky = sky * np.array([.90, .905, .91], f32) - .02; M = Mesh()
    for (x, y, R, H, kd, kk) in [(-118, 330, 30, 40, 'mesa', 3), (-160, 400, 46, 66, 'mesa', 5), (-64, 420, 11, 46, 'pillar', 9), (175, 760, 18, 50, 'stack', 4), (230, 980, 50, 44, 'fin', 2), (60, 1200, 30, 30, 'butte', 6), (-300, 620, 60, 80, 'mesa', 7)]: mesa(M, (x, y), R, H, kk * 37 + 5, kd, mat=5)
    Gh = M.build(); b = render(cam, Gh, L, 0., sky, None, foam_map(Gh, (-420, 420, 250, 1400)), (), (), 2, True); x, y = grid(h, w); tq = np.clip((y - hy) / 460., 0, 1); back = np.where((y > hy)[..., None] & (b.a[..., None] > .5), b.img * (1 - .46 * tq ** .55)[..., None], b.img); veil = np.exp(-np.abs(y - hy) / 16.) * .40; back = back + (C('#a4a7a6') - back) * veil[..., None]; _s[k] = (back.astype(f32), cam); return _s[k]


def cliff_ground(cam, t=0., rings=(), dyn=None, turn=.36, ss=2):
    """e4, f1: the place of e1 seen close: wet cliff top with the big puddle at the origin (same ground height as land.puddle_z, sea far below, horizon under rain).  same use and same Out as land.puddle"""
    g, m, sh = _cliff_ground(turn); back, bc = cliff_back(turn); h, w = cam.h, cam.w
    Hm = cam.K @ np.array(cam.R) @ np.array(bc.R).T @ np.linalg.inv(bc.K); bg = cv2.warpPerspective(back, Hm, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if 'rainpano' not in _s: rain_sky(cam)
    L = puddle_look(); o = render(cam, g, L, t, np.zeros((h, w, 3), f32), sh, None, (), rings, ss, True, dyn=dyn, sky_reflect=SK.mirror(cam, _s['rainpano']))
    d = cam.rays(); X = cam.eye[0] + d[..., 0] * np.minimum(o.z, 1e4); Y = cam.eye[1] + d[..., 1] * np.minimum(o.z, 1e4); inn = cv2.remap(m, ((X + 12.) * 100).astype(f32), ((Y + 8.) * 100).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(f32) / 255.; wat = o.id == -2; keep = np.where(wat, (inn > .5).astype(f32), 1.); keep = cv2.GaussianBlur(keep, (0, 0), .7) * 1.; keep = np.where(o.id >= 0, 1., keep)
    B_ = render(cam, g, L, t, np.zeros((h, w, 3), f32), sh, None, (), (), ss, False, dyn=dyn, water=False); under = B_.rgb + bg * (1 - B_.a[..., None])                      # the face of the cliff, and the sea below it
    k_ = (np.clip(o.a, 0, 1) * keep)[..., None]; o.img = under * (1 - k_) + o.rgb * keep[..., None]; o.a = np.maximum(k_[..., 0], B_.a); o.water = (wat & (inn > .5)).astype(f32); o.z = np.where((o.id >= 0) | (o.water > 0), o.z, B_.z); o.back = bg; return o


def rain_sky(cam, seed=6):
    """heavy rain cloud: three soft-torn decks, one pale rift low on the right"""
    k = 'rainpano'
    if k not in _s:
        w, h = 3200, 1500; hz = h * SK.E1 / (SK.E1 - SK.E0); x, y = grid(h, w); el = np.clip((hz - y) / hz, -.1, 1.2); cx = w * .60
        # cloud as big washes: one value field, cut into a few gouache tones whose edges are lost above and found below
        n1 = warp_by(fbm(h, w, (330, 150, 64, 27, 11), (1, .62, .40, .24, .13), seed, aspect=1.7), 22, 160, seed + 3); n2 = fbm(h, w, (210, 90, 38, 15), (1, .6, .4, .22), seed + 8, aspect=3.4)
        rift = np.exp(-(((x - cx) / (w * .16)) ** 2 + ((el - .10) / .16) ** 2)); v = .80 - .62 * smooth(.02, .75, el) + .34 * rift + .085 * n1 * smooth(.02, .35, el) + .060 * n2 * (1 - smooth(.25, .7, el)); und = np.roll(n1, 26, 0) - n1; v = v + .030 * np.clip(und, -1.5, 1.5) * smooth(.05, .4, el)
        T = [('#3a3d3f', .0), ('#4b4e50', .17), ('#606365', .32), ('#767a7a', .46), ('#8e9190', .60), ('#a8aaa8', .74), ('#c2c4c0', .88), ('#dadbd6', 1.04)]; img = np.empty((h, w, 3), f32); img[:] = C(T[0][0]); soft_ = .030 + .045 * np.clip(.5 + .5 * n2, 0, 1)
        for (c_, th) in T[1:]: img = img + (C(c_) - img) * smooth(th - soft_, th + soft_, v)[..., None]
        sea = y > hz; img = np.where(sea[..., None], C('#8b8e8d'), img); hzb = np.exp(-np.abs(el) / .012) * .5; img = img + (C('#b9bbb8') - img) * hzb[..., None]; img = SK.brush(img, seed + 3, .6, (30, 13), (.012, .012)); _s[k] = img.astype(f32)
    return SK.view(cam, _s[k])


# ================================================================================================ E4, F1: puddles on rock
def puddle_look():
    if 'puddle' not in LOOKS:
        LOOKS['puddle'] = Look(ramp={1: ('#16181a', '#282b2d', '#3d4142', '#5a5e5e', '#828684', '#c2c5c2'), 0: ('#141618', '#232628', '#3a3d3e', '#5a5e5e', '#888b89', '#c9cbc8')}, light=(.50, .62, .60), bounce=(-.85, -.4, .2), fog=dict(col='#8f9291', col2='#a3a6a4', dens=.0055, start=9, hf=0., floor=1., max=.95),
            par=dict(key=.30, amb=.08, sky=.10, side=.10, hue=.03), mats={1: dict(wet=.50, fcell=.055, cell=.06, crack=1.1, chisel=2.6), 0: dict(wet=.4, cell=.05, fcell=.04), 5: dict(cell=.9, wet=0.)}, water=dict(scale=.9, refl=1., glint=0., foam=0., distort=3.2, clear=.62, stroke=.35, cols=('#2c3032', '#8b8e8d', '#33383a', '#b2b5b3', '#ffffff', '#c6c9c7', '#f3f5f4')), wetcol='#b9bcba')
    return LOOKS['puddle']


def puddle_cam(f=1350.): return Camera3D((-.5, -3.5, 1.15), (.1, .2, .05), f)


def puddle_z(x, y):
    """ground height; below 0 there is water.  the big puddle lies at the origin, about 2.2 x 1.3 m"""
    x, y = np.asarray(x, float), np.asarray(y, float); n = .060 * np.sin(x * 1.7 + 1.3 * np.sin(y * 1.1)) * np.sin(y * 2.1 + .8 * np.sin(x * .9)) + .030 * np.sin(x * 4.3 + y * 2.2) + .016 * np.sin(x * 9.1 - y * 7.3); q = np.sqrt((x / 1.25) ** 2 + ((y + .12 * np.sin(x * 2.2)) / .74) ** 2)
    return .052 + n * np.clip(q - .4, 0, 1) - .16 * np.clip(1.15 - q, 0, 1) + .05 * np.clip(np.hypot(x, y) - 3, 0, 9) * .5


def puddle(t=0., cam=None, rings=(), dyn=None, sky=None, ss=2, time='puddle'):
    cam = cam or puddle_cam()
    if 'pud' not in _s:
        r = np.random.default_rng(9); M = Mesh(); e = np.r_[-np.geomspace(60, 4.2, 20), np.arange(-4, 4.05, .14), np.geomspace(4.2, 60, 20)]; y = np.r_[-np.geomspace(8, 4.2, 5), np.arange(-4, 5, .14), np.geomspace(5.1, 90, 30)]; sheet(M, e, y, puddle_z, mat=1, seed=3)
        pebbles(M, 150, (-5, 5, -3.2, 8), puddle_z, r, .02, .26, p=3.2, keep=lambda a, b: float(puddle_z(a, b)) > -.02)
        for (x, yy, s) in [(-3.2, 5.5, .7), (2.9, 4.1, .5), (4.6, 7.2, 1.1), (-6., 9., 1.4), (1.2, 9.8, .9)]: boulder(M, (x, yy, float(puddle_z(x, yy)) + s * .25), s, r, n=22)
        for k in range(9): xx, yy = r.uniform(-260, 260), r.uniform(150, 420); mesa(M, (xx, yy), r.uniform(12, 34), r.uniform(40, 120), k + 300, r.choice(['mesa', 'fin', 'butte', 'pillar', 'stack']), z0=-.5, mat=5)
        _s['pud'] = M.build(); _s['pud_sh'] = shadow_map(_s['pud'], puddle_look().light, 2048, (-7, 7, -4, 11, -.3, 2))
    sky = rain_sky(cam) if sky is None else sky; k = Camera3D(cam.eye, cam.eye + cam.R[2], cam.f, cam.w, cam.h, cam.cx, cam.cy); k.R = cam.R; return render(cam, _s['pud'], puddle_look(), t, sky, _s['pud_sh'], None, (), rings, ss, True, dyn=dyn, sky_reflect=SK.mirror(cam, _s['rainpano']) if 'rainpano' in _s else None)


# ================================================================================================ B3: the lake at Mount Paozu
def lake_cam(dx=0., f=1180.): return Camera3D((dx, -11.5, 2.9), (dx * .6 + 2., 60, 6.2), f)


def lake_z(X, Y):
    ys = -3.2 + 1.5 * np.sin(X * .17 + .6) + .7 * np.sin(X * .43) - .055 * X + 5.2 * smooth(-7, -16, X); d = ys - Y; return np.where(d > 0, .06 + .10 * d + .045 * np.maximum(d - 2, 0) + .05 * np.sin(X * .9) * np.sin(Y * .7 + 1), .06 + .35 * d) - 0. * X


def lake_sky(cam, seed=12):
    from . import clouds
    if 'lakepano' not in _s:
        w, h = 3200, 1500; hz = h * SK.E1 / (SK.E1 - SK.E0); img = SK.gradient(h, w, hz, [(-1, '#bfe0d0'), (0, '#fdf0c4'), (.07, '#f2eec2'), (.2, '#cfe6c8'), (.42, '#a2d6d2'), (.75, '#7cc0d4'), (1.2, '#5aa6cc')], .05, w * .35, seed); x, y = grid(h, w)
        img = SK.brush(img, 3, .8); S = clouds.Sky(h, w, img, light=(-.62, -.55, .4), seed=5); T = clouds.tones('#fff8e0', '#f4e3bf', '#c9d2c4', '#9fbcc0')
        for fx, bs, fw, fh, sd, pk in [(.60, .02, .26, .34, 3, .45), (.40, -.02, .20, .22, 4, .6), (.80, .05, .24, .26, 5, .35), (.22, .04, .22, .20, 6, .5)]: S.bank(x=w * fx, base=hz * (1 - bs), width=w * fw, height=hz * fh, T=T, mist='#e9ecc8', seed=sd, profile=clouds.peak(pk, .25), lit=.30, half=.32, soft=.5)
        _s['lakepano'] = S.paint(seed=4)
    return _s['lakepano']


def _lake():
    if 'lake' in _s: return _s['lake']
    r = np.random.default_rng(21); M = Mesh(); x = np.r_[-np.geomspace(40, 12.5, 14), np.arange(-12, 14, .42), np.geomspace(14.5, 40, 14)]; y = np.arange(-16, 6, .34); sheet(M, x, y, lake_z, mat=4, seed=8)
    for (xx, yy, s) in [(-6.2, -.4, .75), (-5.2, .5, .42), (-7.4, .7, .95), (4.6, -2.0, .38), (5.4, -1.5, .6), (6.3, -2.4, .34), (9.6, -2.6, .7), (1.2, -2.2, .22), (-3.6, -.9, .26), (2.1, -1.4, .3)]: boulder(M, (xx, yy, .06 + s * .12), s, r, mat=2, n=24, sz=(1.2, .9, .6))
    # far shore, wooded capes, karst
    sheet(M, np.linspace(-260, 260, 60), np.linspace(118, 240, 14), lambda X, Y: -1.2 + 5.5 * smooth(120 + 9 * np.sin(X * .05) + 5 * np.sin(X * .13), 150, Y) + .9 * np.sin(X * .2), mat=4, seed=5)
    for (xx, yy, R, H, st) in [(-62, 128, 26, 20, .3), (-34, 140, 16, 13, .2), (70, 120, 24, 17, .3), (42, 150, 18, 14, .2), (4, 158, 22, 12, .2), (108, 132, 30, 24, .4), (-104, 126, 34, 30, .4)]: hill(M, (xx, yy), R, H, int(xx + 300), 3, steep=st)
    for (xx, yy, R, H, st) in [(-58, 230, 34, 92, 1.0), (-8, 300, 30, 128, 1.2), (40, 270, 26, 78, 1.0), (84, 330, 38, 146, 1.1), (-112, 300, 44, 116, .9), (132, 260, 36, 88, .9), (-176, 380, 60, 150, .9), (12, 460, 70, 110, .5), (190, 420, 60, 170, 1.0), (-70, 520, 60, 190, 1.0), (60, 620, 70, 230, 1.0), (-220, 640, 110, 220, .7), (240, 700, 120, 240, .7), (-20, 900, 160, 260, .6), (150, 1000, 160, 300, .7), (-160, 1000, 180, 280, .6)]: hill(M, (xx, yy), R, H, int(xx * 3 + yy), 3, steep=st, n=14)
    _s['lake'] = M.build(); return _s['lake']


def _forest_fringe(o, r): return o


def lake(t=0., cam=None, dx=0., rings=(), dyn=None, overlays=True, ss=2, seed=2):
    """b3: green-gold afternoon.  Out.img; Out.layers = {'tree': near foliage RGBA, 'shade': dapple}; overlays=True draws grass, reeds, tree"""
    cam = cam or lake_cam(dx); G = _lake(); L = look('lake')
    if 'lake_sh' not in _s: _s['lake_sh'] = shadow_map(G, L.light, 3072, (-180, 180, -16, 420, 0, 120))
    P = lake_sky(cam); o = render(cam, G, L, t, SK.view(cam, P), _s['lake_sh'], None, (), rings, ss, True, dyn=dyn, sky_reflect=cv2.blur(SK.mirror(cam, P), (7, 3)))
    r = np.random.default_rng(seed); img = o.img
    if overlays:
        # light pouring over the bank: dapple under the near tree
        h, w = cam.h, cam.w; d = cam.rays(); X = cam.eye[0] + d[..., 0] * o.z; Y = cam.eye[1] + d[..., 1] * o.z; bank = (o.rock > 0) & (Y < 8)
        n = fbm(512, 512, (34, 15, 7), (1, .7, .4), 5); nn = cv2.remap(n, np.mod(X * 34 + 2000, 511).astype(f32), np.mod(Y * 34 + 2000, 511).astype(f32), cv2.INTER_LINEAR); cover = (1 - smooth(-5.5, 1.5, X + .35 * (Y + 8))); sh = smooth(-.10, .16, .55 - nn) * cover * bank; img = img * (1 - .46 * sh[..., None] * (1 - np.array([.10, .0, .12], f32)))
        N = 70000; px = r.uniform(-14, 14, N); py = -13.4 + 14.5 * r.random(N) ** 1.15; pz = lake_z(px, py); k = pz > .10; px, py, pz = px[k], py[k], pz[k]; cl = fbm(256, 256, (30, 12, 5), (1, .6, .4), 8)[np.clip((py + 16) * 11, 0, 255).astype(int), np.clip((px + 14) * 9, 0, 255).astype(int)]; k = r.random(len(px)) < .45 + .5 * smooth(-.6, .9, cl); px, py, pz, cl = px[k], py[k], pz[k], cl[k]
        lit = np.clip(.62 + .22 * cl - .55 * (1 - smooth(-5.5, 1.5, px + .35 * (py + 8))) * smooth(-.1, .5, -cl + .4), 0, 1); hg = np.clip(.16 + .10 * cl + r.normal(0, .05, len(px)), .05, .6) * np.where(pz < .35, 1.4, 1.); wnd = .16 + .10 * np.sin(t * 2.2 + px * .7)
        img = GW.blades(img, cam, o.z, np.c_[px, py, pz], hg, ('#1f4632', '#6a9a43', '#d9e27c'), r, wind=wnd, width=.011, light=lit)
        n2 = 900; qx = np.r_[r.normal(-6.6, 1.1, n2 // 2), r.normal(6.0, 1.6, n2 // 3), r.normal(1.4, .5, n2 // 6)]; ys = -3.2 + 1.5 * np.sin(qx * .17 + .6) + .7 * np.sin(qx * .43) - .055 * qx + 5.2 * smooth(-7, -16, qx); qy = ys + r.normal(.15, .35, len(qx)); img = GW.blades(img, cam, o.z, np.c_[qx, qy, np.full(len(qx), .02)], r.uniform(.5, 1.25, len(qx)), ('#1d4030', '#6b9444', '#ddde84'), r, wind=.10 + .06 * math.sin(t * 2.), lean=.12, width=.018, eps=.4, bend=.25)
        # flowers on the bank
        fl = r.random(len(px)) < .012; s, zc = cam.project(np.c_[px, py, pz + hg * .9][fl]); cv = Canvas(h, w, img=img)
        for (a, b), q in zip(s, cam.f / zc):
            if 0 < a < w and 0 < b < h and q > 40: m = 5; th = np.arange(m) / m * 6.28 + r.uniform(0, 6); rr = q * .020; cv.strokes(arcs(a + np.cos(th) * rr, b + np.sin(th) * rr * .6, th, rr * 1.7, 0., 3), rr * .55, C('#fffbe6') if r.random() < .6 else C('#ffd25a'), shape='leaf', seed=3); cv.strokes(arcs([a], [b], [0.], [rr * .8], 0., 3), rr * .4, C('#f2b23a'), shape='round')
        img = cv.img
    o.img = img; o.layers = dict(tree=lake_tree(seed)) if overlays else {}
    if overlays: o.img = over(img, o.layers['tree'])
    return o


def lake_tree(seed=2, h=H, w=W):
    """near foliage hanging into the top left of the frame (its own parallax plane), premult RGBA"""
    k = ('ltree', seed, h, w)
    if k in _s: return _s[k]
    r = np.random.default_rng(seed + 50); cv = Canvas(h, w, layer=True); T = ('#0f2a28', '#22503a', '#4f8538', '#b9cc56'); bark = ('#2a2620', '#5a4a34', '#b99a5c')
    def limb(p, w0, w1, n=16): P = spl(np.array(p, float), n=n); P = P[np.linspace(0, len(P) - 1, 14).astype(int)]; [cv.strokes(P[None] + [[[ox, oy]]], wd, C(c), shape='taper', seed=s_, dry=dr) for ox, oy, wd, c, s_, dr in ((0, 0, w0, bark[0], 1, 0.), (-w0 * .22, -w0 * .28, w0 * .5, bark[1], 2, .25), (-w0 * .34, -w0 * .5, w0 * .18, bark[2], 3, .45))]
    limb([(-40, 560), (120, 360), (330, 220), (640, 128), (900, 120)], 40, 4); limb([(220, 270), (360, 300), (520, 330), (640, 400)], 20, 3); limb([(-40, 240), (140, 130), (380, 60), (560, -10)], 30, 4); limb([(420, 180), (560, 60), (720, 0)], 15, 2)
    sp = [((-60, 40), (960, 60), 150, 34), ((0, 170), (820, 150), 84, 22), ((-40, 330), (90, 460), 84, 8), ((330, 270), (660, 400), 60, 12), ((820, 70), (1050, 150), 54, 10), ((500, -30), (940, 10), 90, 10)]; cl = []
    for (a, b, R, n) in sp:
        for j in range(n): t = (j + r.uniform(-.4, .4)) / n; cl.append((a[0] + (b[0] - a[0]) * t + r.normal(0, R * .35), a[1] + (b[1] - a[1]) * t + r.normal(0, R * .34), R * r.uniform(.42, .92)))
    cl.sort(key=lambda c: c[1] + r.normal(0, 40)); lim = len(cl) // 2
    for j, (x, y, R) in enumerate(cl):
        GW.leaves(cv, x, y, R, int(18 + R * 1.5), T, r, (-.40, -.62), (8, 18), .74, .5, lit=float(np.clip(.25 + 1.1 * r.random() ** 2 + (x > 700) * .5, .1, 1.5)))
        if j == lim: limb([(120, 360), (330, 220), (640, 128), (900, 120)], 24, 3); limb([(320, 285), (520, 330), (640, 400)], 14, 2)
    for (x, y) in [(1040, 190), (690, 430), (600, 440), (950, 190), (480, 360), (30, 540), (760, 210), (300, 260)]:                # hanging sprigs
        n = 16; a = r.normal(1.3, .45, n); P = arcs(x + r.normal(0, 22, n), y + r.normal(0, 18, n), a, r.uniform(16, 30, n), r.normal(0, .4, n), 4); cv.strokes(P, r.uniform(3.5, 7, n), np.stack([C(T[int(q)]) for q in r.choice([1, 1, 2, 2, 3], n)]), shape='leaf', seed=9)
    _s[k] = cv.layer(); return _s[k]


# ================================================================================================ H: the crater lake
def dawn_up(h=H, w=W, seed=14):
    """the sky straight overhead at dawn, as the still lake shows it: rose, gold, lilac, blue with combed cloud"""
    x, y = grid(h, w); t = (x * .55 + (h - y) * .85) / (w * .55 + h * .85); img = ramp(t + .03 * fbm(h, w, (400, 160), (1, .5), seed), [(0, '#8fa6d2'), (.22, '#b3b4d8'), (.42, '#e2b8c4'), (.60, '#f7bfae'), (.78, '#ffd2b4'), (1, '#ffe9cc')])
    img = SK.torn(img, -h, h * 2.2, '#c08fae', '#f2b2a8', '#fff0d8', (w * 1.1, -h * .2), seed, .40, 3.2, 1.5, .75, .5); img = SK.torn(img, -h, h * 2.2, '#dba3ac', '#ffd0b8', '#fff6e2', (w * 1.1, -h * .2), seed + 40, .36, 5., .7, .7, .35); return SK.brush(img, seed, .8, (40, 18))


def top_look():
    if 'dawn_top' not in LOOKS:
        L = look('dawn'); LOOKS['dawn_top'] = Look(ramp={4: ('#1f3a44', '#2c4e50', '#416a56', '#5c8459', '#87a268', '#c9c587'), 0: ('#3a3850', '#5a5470', '#86778c', '#b79a9c', '#e6bda8', '#ffe4cc'), 2: ('#3a3850', '#5a5470', '#86778c', '#b79a9c', '#e6bda8', '#ffe4cc')}, light=(.78, .42, .46), bounce=(-.8, -.4, .3), fog=None, par=dict(key=.60, amb=.14, sky=.14, side=.10), mats={4: dict(chisel=.3, crack=0., fcell=.10, hue=.12), 2: dict(cell=.05, fcell=.04), 0: dict(cell=.06, fcell=.05)},
            water=dict(scale=1.1, refl=1., glint=0., foam=0., distort=7., clear=.90, stroke=.35, cols=('#6d7fa8', '#8a8fb8', '#8f86b0', '#ffe2cf', '#fff4e0', '#fff0e4', '#fff8ee')))
    return LOOKS['dawn_top']


def islet_z(X, Y):
    """round grass islet for two men lying head to head (radius about 1.4 m, their boots reach into the water)"""
    X, Y = np.asarray(X, float), np.asarray(Y, float); q = _islet_q(X, Y); return np.where(q < 1, .05 + .10 * np.clip(1 - q, 0, 1) ** .6, .05 - .42 * (q - 1)) + .006 * np.sin(X * 5 + Y * 3)


def _islet_q(X, Y):
    """1 on the shore, 0 in the middle, grows outward"""
    th = np.arctan2(Y, X); q = np.sqrt((X / 1.36) ** 2 + (Y / 1.20) ** 2) / (1 + .045 * np.sin(3 * th + 1) + .040 * np.sin(5 * th + 2) + .026 * np.sin(11 * th) + .020 * np.sin(19 * th + 1.3) + .012 * np.sin(31 * th)); return q


def top_cam(z=4.75, x=0., y=0., roll=0.): return Camera3D((x, y, z), (x, y + 1e-4, 0), 1500., roll=roll)


def islet_layers(cam, seed=4, grass=1.):
    """h1 as layers, so that only the water is redrawn per frame: land (premult RGBA: islet, rocks, grass, dew), front (RGBA blades that may overlap a lying figure), tint (h,w) how much of the bank shows under water"""
    crater_top(0., cam, grass=0); G = _s['top']; o = render(cam, G, top_look(), 0., np.zeros((cam.h, cam.w, 3), f32), _s['top_sh'], None, (), (), 2, False, water=False); h, w = cam.h, cam.w; d_ = cam.rays(); tt = -cam.eye[2] / d_[..., 2]; X_ = cam.eye[0] + d_[..., 0] * tt; Y_ = cam.eye[1] + d_[..., 1] * tt; zl = islet_z(X_, Y_)
    Zp = cam.eye[2] + d_[..., 2] * np.minimum(o.z, 1e4); dry = np.clip(cv2.GaussianBlur((Zp > -.002).astype(f32), (0, 0), .8) * 1.0, 0, 1) * (o.a > 0); al = (o.a * dry).astype(f32); cv = Canvas(h, w, layer=True); cv.img[:] = o.rgb / np.maximum(o.a, .02)[..., None]; cv.a[:] = al
    r = np.random.default_rng(seed); N = int(23000 * grass); x = r.uniform(-1.7, 1.7, N); y = r.uniform(-1.5, 1.5, N); z = islet_z(x, y); k = z > .056; x, y, z = x[k], y[k], z[k]; cl = fbm(220, 220, (40, 18, 7), (1, .6, .4), 3)[np.clip((y + 1.9) * 56, 0, 219).astype(int), np.clip((x + 1.9) * 56, 0, 219).astype(int)]; edge = np.arctan2(y, x)
    L = np.clip(.098 + .028 * cl + r.normal(0, .019, len(x)), .055, .18) * (1 + .8 * (1 - smooth(.055, .085, z))); a = r.normal(.2, .7, len(x)) + 1.25 * cl; a = np.where(z < .082, edge + r.normal(0, .5, len(x)), a); lit = np.clip(.50 + .17 * cl + .18 * np.cos(a - .5) + r.normal(0, .09, len(x)), .10, 1)
    s, zc = cam.project(np.c_[x, y, z]); ppm = cam.f / zc; P = arcs(s[:, 0] + np.cos(a) * L * ppm * .5, s[:, 1] - np.sin(a) * L * ppm * .5, -a, L * ppm, r.normal(0, .7, len(x)), 5); d, m, li = C('#2b4e50'), C('#6a9062'), C('#e8d99c'); col = np.where((lit < .5)[:, None], mix(d, m, np.clip(lit * 2, 0, 1)[:, None]), mix(m, li, np.clip(lit * 2 - 1, 0, 1)[:, None])) * (1 + r.normal(0, .04, (len(x), 1))); od = np.argsort(z + r.normal(0, .02, len(z))); wd = np.maximum(.6, ppm * .0085 * r.uniform(.6, 1.3, len(x)))
    cv.strokes(P[od], wd[od], np.clip(col[od], 0, 1.2), shape='taper', op=r.uniform(.8, 1, len(x)), seed=7, bristle=0.); dew = r.random(len(x)) < .016
    for (a_, b_), q in zip(P[dew][:, -1], r.uniform(.3, 1, dew.sum())): cv2.circle(cv.img, (int(a_ * 8), int(b_ * 8)), int(8 * (.7 + 1.1 * q)), (1.12, 1.06, .98), -1, cv2.LINE_AA, 3); cv2.circle(cv.a, (int(a_ * 8), int(b_ * 8)), int(8 * (.7 + 1.1 * q)), 1., -1, cv2.LINE_AA, 3)
    fr = r.random(len(x)) < .30; lay = Canvas(h, w, layer=True); lay.strokes(P[fr], wd[fr] * 1.05, np.clip(col[fr] * 1.03, 0, 1.2), shape='taper', seed=8, bristle=0.)
    tint = np.exp(-np.clip(-zl, 0, 5) * 15.) * .70; tint[zl > 0] = .70; return dict(land=cv.layer().astype(f32), front=lay.layer().astype(f32), blades=(P.astype(f32), wd.astype(f32), np.clip(col, 0, 1.2).astype(f32)), s=_islet_q(X_, Y_).astype(f32), tint=tint.astype(f32), X=X_.astype(f32), Y=Y_.astype(f32), up=_s['up'])


def crater_top(t=0., cam=None, rings=(), dyn=None, ss=2, grass=1., drift=(5., -1.), sky=None, seed=4):
    """h1: straight down.  islet at the origin in a mirror of dawn sky.  drift = px per second the reflected clouds travel.  o.layers['front'] = blades that overlap whoever lies there (draw after figures)"""
    cam = cam or top_cam(); 
    if 'top' not in _s:
        r = np.random.default_rng(33); M = Mesh(); sheet(M, np.arange(-2.4, 2.41, .068), np.arange(-2.1, 2.11, .068), islet_z, mat=4, seed=2, hole=lambda c: c[:, 2] < -.35)
        for (x, y, s) in [(-.86, 1.08, .11), (-1.06, 1.00, .06), (1.00, -.98, .13), (1.22, -.86, .07), (.48, 1.24, .05), (-.55, -1.24, .09), (-.36, -1.31, .045), (2.05, 1.16, .20), (2.30, 1.00, .10), (-2.15, -.96, .16), (-2.40, -1.16, .23), (-1.95, 1.36, .09), (2.55, -1.28, .13), (-2.70, .85, .11)]: boulder(M, (x, y, .02 + s * .1), s, r, mat=2, n=26, sz=(1.2, .9, .7))
        g_ = M.build(); zc_ = g_.T[..., 2].mean(1); low_ = (g_.MAT == 4) & (zc_ < .058); g_.MAT[low_] = 2; g_.COL[low_] *= np.array([.46, .46, .52], f32); _s['top'] = g_; _s['top_sh'] = shadow_map(_s['top'], top_look().light, 1536, (-6, 6, -3.4, 3.4, -.2, .8)); _s['up'] = dawn_up(1400, 2500)
    U = _s['up'] if sky is None else sky; hh, ww = U.shape[:2]; M2 = np.float32([[1, 0, -(ww - W) / 2 + drift[0] * t], [0, 1, -(hh - H) / 2 + drift[1] * t]]); RS = cv2.warpAffine(U, M2, (cam.w, cam.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    o = render(cam, _s['top'], top_look(), t, RS, _s['top_sh'], None, (), rings, ss, True, dyn=dyn, sky_reflect=RS); img = o.img; r = np.random.default_rng(seed); o.layers = {}
    d_ = cam.rays(); tt = -cam.eye[2] / d_[..., 2]; X_ = cam.eye[0] + d_[..., 0] * tt; Y_ = cam.eye[1] + d_[..., 1] * tt; dep = np.clip(-islet_z(X_, Y_), 0, 5); kw = np.exp(-dep * 9.) * o.water * .72; img = img + (C('#4f5870') - img) * kw[..., None]        # the bank goes on under water
    rim_ = np.abs(dep - .10 - .012 * math.sin(t * 2.)); img = img + (C('#fff0e2') - img) * (np.clip(1 - rim_ / .006, 0, 1) * o.water * .26 * (fbm(cam.h, cam.w, (60, 20), (1, .5), 3) > .1))[..., None]
    if grass:
        N = int(26000 * grass); x = r.uniform(-3.2, 3.2, N); y = r.uniform(-1.8, 1.8, N); z = islet_z(x, y); k = z > .058; x, y, z = x[k], y[k], z[k]; cl = fbm(200, 320, (40, 18, 7), (1, .6, .4), 3)[np.clip((y + 1.8) * 55, 0, 199).astype(int), np.clip((x + 3.2) * 50, 0, 319).astype(int)]
        L = np.clip(.115 + .03 * cl + r.normal(0, .02, len(x)), .075, .2) * (1 + .9 * (1 - smooth(.06, .10, z))); a = r.normal(.2, .8, len(x)) + 1.3 * cl + .10 * np.sin(t * 2 + x * 2.2); edge = np.arctan2(y * 1.9, x); a = np.where(z < .088, edge + r.normal(0, .5, len(x)), a); lit = np.clip(.50 + .17 * cl + .16 * np.cos(a - .5) + r.normal(0, .09, len(x)), .12, 1)
        s, zc = cam.project(np.c_[x, y, z]); ppm = cam.f / zc; P = arcs(s[:, 0] + np.cos(a) * L * ppm * .5, s[:, 1] - np.sin(a) * L * ppm * .5, -a, L * ppm, r.normal(0, .7, len(x)), 5); d = C('#2f5350'); m = C('#6f9463'); li = C('#ecdc9c'); col = np.where((lit < .5)[:, None], mix(d, m, np.clip(lit * 2, 0, 1)[:, None]), mix(m, li, np.clip(lit * 2 - 1, 0, 1)[:, None])) * (1 + r.normal(0, .04, (len(x), 1))); od = np.argsort(z + r.normal(0, .02, len(z)))
        cv = Canvas(cam.h, cam.w, img=img); cv.strokes(P[od], np.maximum(.6, ppm[od] * .010 * r.uniform(.6, 1.3, len(x))), np.clip(col[od], 0, 1.2), shape='taper', op=r.uniform(.8, 1, len(x)), seed=7, bristle=0.); img = cv.img
        dew = r.random(len(x)) < .035; tip = P[dew][:, -1]; br = r.uniform(.4, 1, dew.sum())
        for (a_, b_), q in zip(tip, br): cv2.circle(img, (int(a_ * 8), int(b_ * 8)), int(8 * (.7 + .9 * q)), (1.15, 1.08, .98), -1, cv2.LINE_AA, 3)
        fr = r.random(len(x)) < .05; lay = Canvas(cam.h, cam.w, layer=True); lay.strokes(P[fr], np.maximum(.7, ppm[fr] * .011), np.clip(col[fr] * 1.04, 0, 1.2), shape='taper', seed=8, bristle=0.); o.layers['front'] = lay.layer()
    o.img = img; return o


def eye_cam(dx=0., f=1250., z=.62): return Camera3D((dx, -4.6, z + .25), (dx + .6, 20, 1.5 + z), f)


def eye_z(X, Y):
    ys = 2.2 + .9 * np.sin(X * .33 + .5) + .35 * np.sin(X * .9); d = ys - Y; return np.where(d > 0, .05 + .055 * d + .012 * np.sin(X * 2.1) * np.sin(Y * 1.7), .05 + .22 * d)


def dawn_pano(seed=8):
    from . import clouds
    if 'dawnpano' not in _s:
        w, h = 3200, 1500; hz = h * SK.E1 / (SK.E1 - SK.E0); cx = w * .60; img = SK.gradient(h, w, hz, [(-1, '#e6c0bc'), (0, '#ffe9c6'), (.05, '#ffdcbc'), (.13, '#f9c4b2'), (.27, '#e0b4c2'), (.46, '#b7b2d6'), (.75, '#94acd8'), (1.2, '#7c9ad0')], .05, cx, seed); x, y = grid(h, w); g = np.exp(-np.sqrt(((x - cx) / (w * .15)) ** 2 + ((y - hz * .80) / (hz * .34)) ** 2)); img = img + (C('#ffe3b4') - img) * (.55 * g)[..., None]
        img = SK.torn(img, hz * .15, hz * .80, '#a393c0', '#eab2b4', '#ffeccb', (cx, hz * .9), seed + 2, .36, 6., 1.1, .7, .4); img = SK.torn(img, hz * .62, hz * 1.0, '#c79db6', '#ffcdb4', '#fff6dc', (cx, hz * .9), seed + 5, .42, 11., .55, .75, .3); img = SK.brush(img, 3, .8)
        S = clouds.Sky(h, w, img, light=(.45, -.30, .35), seed=5); T = clouds.tones('#fff2d8', '#ffd2b8', '#d3a5bd', '#a793c4')
        for fx, bs, fw, fh, sd in [(.30, .0, .30, .20, 3), (.78, -.01, .28, .16, 4), (.50, .0, .18, .10, 6)]: S.bank(x=w * fx, base=hz * (1 - bs), width=w * fw, height=hz * fh, T=T, mist='#ffdcc4', seed=sd, profile=clouds.peak(.5, .3), lit=.22, half=.34, soft=.7)
        _s['dawnpano'] = S.paint(seed=4)
    return _s['dawnpano']


def crater_eye(t=0., cam=None, rings=(), dyn=None, ss=2, grass=1., seed=6, sun=True, gust=None):
    """h2, h3: eye level on the shore, wet grass, crater wall in mist.  Out.layers['front'] = the nearest blades.  gust = (x, y, R, width) ring that bends the grass"""
    cam = cam or eye_cam(); L = look('dawn')
    if 'eye' not in _s:
        r = np.random.default_rng(44); M = Mesh(); x = np.r_[-np.geomspace(60, 8.5, 12), np.arange(-8, 9, .28), np.geomspace(9.2, 60, 12)]; sheet(M, x, np.arange(-6, 5, .26), eye_z, mat=4, seed=2)
        for (xx, yy, s) in [(-3.4, 2.6, .30), (-3.9, 3.0, .5), (-2.8, 3.1, .18), (4.2, 1.9, .36), (4.9, 2.3, .6), (3.6, 2.5, .16), (1.0, 3.6, .14), (-6.2, 2.9, .8), (7.6, 2.6, .9), (-.8, 4.1, .2)]: boulder(M, (xx, yy, .03 + s * .1), s, r, mat=2, n=26, sz=(1.25, .9, .65))
        n1 = fbm(64, 512, (26, 12, 5), (1, .6, .35), 12)
        def rim(X, Y):
            dx, dy = X, Y - 170.; q = np.hypot(dx, dy); th = np.arctan2(dx, dy); n = np.interp((th + np.pi) / (2 * np.pi) * 511, np.arange(512), n1[20]); H_ = 66 + 16 * n + 16 * np.sin(th * 3 + .8); return -2 + H_ * smooth(150, 215 + 10 * n, q) * (1 - .6 * smooth(240, 420, q)) + 5 * np.sin(th * 30 + n * 3) * smooth(155, 200, q)
        a = np.linspace(-np.pi * .98, np.pi * .98, 220); rr = np.r_[np.linspace(140, 250, 26), np.linspace(262, 430, 8)]; A, R_ = np.meshgrid(a, rr); j = np.random.default_rng(3); A = A + j.normal(0, .006, A.shape); R_ = R_ + j.normal(0, 1.3, A.shape); X = np.sin(A) * R_; Y = np.cos(A) * R_ + 170; Z = rim(X, Y) + j.normal(0, 1.1, A.shape); V = np.stack([X, Y, Z], -1); p, q, u, v = V[:-1, :-1], V[:-1, 1:], V[1:, 1:], V[1:, :-1]
        T = np.concatenate([np.stack([p, u, q], 2).reshape(-1, 3, 3), np.stack([p, v, u], 2).reshape(-1, 3, 3)]); T = T[T[..., 1].min(1) > 95]; M.add(T, ao=1., mat=0, seed=77)
        for k in range(26): th = r.uniform(-1.5, 1.5); d = r.uniform(140, 158); hill(M, (math.sin(th) * d, 170 + math.cos(th) * d), r.uniform(6, 14), r.uniform(4, 11), k + 60, 3, steep=.2, n=7, sides=14)
        _s['eye'] = M.build(); _s['eye_sh'] = shadow_map(_s['eye'], L.light, 2048, (-240, 240, -6, 400, 0, 90))
    P = dawn_pano(); o = render(cam, _s['eye'], L, t, SK.view(cam, P), _s['eye_sh'], None, (), rings, ss, True, dyn=dyn, sky_reflect=cv2.blur(SK.mirror(cam, P), (9, 3))); img = o.img; h, w = cam.h, cam.w; hy = horizon(cam); x, y = grid(h, w)
    # mist lying on the far water
    m = np.exp(-((y - hy - 4) / 34.) ** 2) * (.55 + .25 * fbm(h, w, (300, 90), (1, .5), 5, aspect=6.)); m = np.clip(m, 0, 1) * (o.z > 30); img = img + (C('#ffe9d6') - img) * (.62 * m)[..., None]
    if sun:                                                                         # the sun stands on the rim; its light pours over it
        sd = np.array([math.sin(math.radians(16.8)), math.cos(math.radians(16.8)), math.tan(math.radians(14.2))]); sp, sz = cam.project(cam.eye + sd * 1e5)
        if sz > 0:
            q = np.sqrt((x - sp[0]) ** 2 + (y - sp[1]) ** 2); skym = (o.a < .5); rr = .036 * cam.f; img = img + (C('#fffdf0') * 1.15 - img) * (np.clip(rr - q + .5, 0, 1) * skym)[..., None]; img = img + (C('#ffe6b8') * (np.exp(-q / 230.) * .15 + np.exp(-q / 60.) * .20)[..., None]) * (1 - np.clip(img, 0, 1) * .6)
    r = np.random.default_rng(seed); o.layers = {}; o.base = img.copy()
    if grass:
        N = int(90000 * grass); px = r.uniform(-9, 9, N); py = -4.2 + 8 * r.random(N) ** 1.25; pz = eye_z(px, py); k = pz > .065; px, py, pz = px[k], py[k], pz[k]; cl = fbm(256, 256, (30, 12, 5), (1, .6, .4), 8)[np.clip((py + 5) * 25, 0, 255).astype(int), np.clip((px + 9) * 14, 0, 255).astype(int)]; hg = np.clip(.13 + .06 * cl + r.normal(0, .035, len(px)), .04, .36); lit = np.clip(.50 + .25 * cl + r.normal(0, .1, len(px)), 0, 1); wind = .12 + .08 * np.sin(t * 2.1 + px * 1.1)
        if gust is not None: dd = np.hypot(px - gust[0], py - gust[1]) - gust[2]; g_ = np.exp(-(dd / gust[3]) ** 2); wind = wind + g_ * 1.2 * np.sign(px - gust[0]); lit = np.clip(lit + .35 * g_, 0, 1)
        front = py < -3.35; img = GW.blades(img, cam, o.z, np.c_[px, py, pz][~front], hg[~front], ('#2a4a52', '#7f9c6c', '#f6e2a4'), r, wind=wind[~front], width=.0055, light=lit[~front], tip=.5)
        tips = np.c_[px, py, pz + hg * .82]; s, zc = cam.project(tips); dw = (r.random(len(px)) < .016) & (zc > .5) & ~front
        for (a, b), q in zip(s[dw], cam.f / zc[dw]):
            if 0 < a < w and 0 < b < h: cv2.circle(img, (int(a * 8), int(b * 8)), int(8 * min(3.0, .45 + q * .004 * r.random())), (1.2, 1.12, 1.0), -1, cv2.LINE_AA, 3)
        lay = Canvas(h, w, layer=True); z9 = np.full((h, w), 1e6, f32); cv = lay; al = np.zeros((h, w), f32); fi = np.zeros((h, w, 3), f32); GW.blades(fi, cam, z9, np.c_[px, py, pz][front], hg[front] * 1.2, ('#16303c', '#426a5a', '#c9c48c'), r, wind=wind[front], width=.0055, light=lit[front] * .8, tip=.4, alpha=al, eps=9.); o.layers['front'] = np.dstack([fi * 1., al]).astype(f32)
    o.img = img; return o
