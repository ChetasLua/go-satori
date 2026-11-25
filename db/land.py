"""LAND: painted places with real cameras.  (rock, water and forest are painted from real geometry: ~1 s a frame at 1080p, first call ~10 s to compile)
Every place returns Out: .img (1080,1920,3) float, .z metres (sky 1e6), .a coverage (0 = sky), .rock .water masks, .cam -> cam.project(XYZ) puts characters in.
Figures behind rock: mask their layer with (o.z > distance).  Keep static pictures in cache/ (np.float16).

from db import land; from db.cam import Camera3D           metres, +X right, +Y forward, +Z up; sea z = 0; heroes' table top z = land.TOP = 34
COAST   cam = land.chase_cam(s, side=0, up=0, f=1150)      flying camera on the corridor between the mesas, s = metres 0..1600, banks by itself
        o = land.coast(cam, t=f/24, time='night', moon='auto'|None, rings=[(x,y,R,width,amp)], fires=[(x,y,z,radius,'#col',amt)], dyn=Geo, hide=[ids])
        time: night | storm (F) | gold | first (G6-G8) | rain          land.path(s) point on the corridor      o.img of the hero view: Camera3D((-58,-118,66),(46,400,70),1300)
        land.PILLARS() -> dict(id, c, R, H, kind, s)      hit one: coast(cam, hide=[id], dyn=land.break_pillar(id, z, (dx, dy), u_seconds))
        three pillars in a row (d2): land.coast(land.trio_cam(f), f/24, dyn=land.trio_geo(f)); body at land.trio_point(f): hits at 40, 50, 60, flies left to right
        land.flames(cam, [(x,y,z,h)], f, o.z) -> RGBA      small hard-edged fires on twos
GROUND  land.ground(cam, t, time, dyn=, crater=[(x,y,R,depth)], fires=)    heroes' table close up (flat round x=0 y=-22), coast behind; land.ground_z(x, y)
        land.Fan((x,y), R=6).geo(u) plates opening u 0..1      land.Slabs((x,y), R=30, n=18).geo(u, t) floating ground      land.rubble((x,y), R, n, u=)
        land.waste(cam, t, time, trench=((x0,y0),(x1,y1), half_w, u, depth))     flat land to the horizon with a ploughed trench (d5, d6)
RIDGE   land.ridge(u_line, u_lift, t, rings=) -> Out (.crest px line, .glow)       c4: hair line 0..1, then lift 0..1: slabs rise off the light
CLIFF   land.cliff_e1() plates in e1's own camera;  land.cliff(t) (Vegeta stands at (0, 0, land.CLIFF); camera land.cliff_cam(push))    land.lily(cel, x, y, h, open, sway, seed)    land.rain(f, n, freeze=, up=, near=) RGBA    land.splashes(f, cam, z, pts)
        land.cliff_ground(cam, t, rings=) the same cliff top close up with the big puddle at the origin (e4, f1), same use as land.puddle
PUDDLE  land.puddle(t, cam, rings=land.rain_rings(f, (-1.4, 1.4, -.9, .9)))   land.puddle_z(x, y)      (e4, f1)
LAKE    land.lake(t, dx=) (b3)  o.layers['tree'] near foliage; land.lake_z      CRATER  land.crater_top(t, rings=) (h1) o.layers['front'] blades over figures;  land.crater_eye(t, gust=(x,y,R,w)) (h2, h3), land.eye_z
DUSK    from db.land_kame import Dusk; Dusk().frame(u, lit, chime, laugh, glow, t)    b5: Kame house window, sill, bell; u 0 wide .. 1 ball fills the match circle
"""
import numpy as np, cv2, math
from .land_core import *
from .land_geo import *
from . import land_sky as SK, land_nb as nb
from .cam import rot_x, rot_y, rot_z

MOON = (.46, .86, .235)            # where the moon is drawn; its light is cheated round to the side for clearer planes
LOOKS = {}


def look(name):
    if name in LOOKS: return LOOKS[name]
    if name == 'night':
        L = Look(ramp={0: ('#080715', '#16142e', '#2b2757', '#544794', '#8f78d2', '#d9c8ff'), 1: ('#0b0a1a', '#1a1733', '#322c5c', '#5c4f9c', '#9a84d6', '#e2d4ff')},
                 light=(.84, .10, .54), bounce=(-.92, -.30, .22), glint=MOON, fog=dict(col='#37296f', col2='#6443b4', dens=.00125, start=60, hf=.0032, floor=.30, pw=5., max=.88),
                 par=dict(key=.74, amb=.09, sky=.11, side=.15), mats={1: dict(key=.44, sky=.09, amb=.07, chisel=1.5, fcell=.42)}, water=dict(scale=9., refl=.9, glint=1., foam=.8, distort=30, sigma=.17, cols=('#080817', '#241d4c', '#06060f', '#3a3072', '#f1e6ff', '#7d6fbc', '#e6d9ff')), wetcol='#5a48a8')
    elif name == 'storm':                                                          # F: green-black storm, gold from the heroes (give fires) and from above
        L = Look(ramp={0: ('#05090a', '#0e1718', '#1c2b29', '#37463a', '#6d6a44', '#e0bd62')}, light=(.35, -.30, .89), bounce=(-.8, -.4, .3), fog=dict(col='#1b2a2a', col2='#3a4a3c', dens=.0011, start=30, hf=.003, floor=.35, max=.9),
                 par=dict(key=.34, amb=.12, sky=.16, side=.14), water=dict(scale=10., refl=.85, glint=.3, foam=.9, distort=34, sigma=.2, cols=('#060d0f', '#1a2b2c', '#040809', '#2f4640', '#ffe9a0', '#8d9a86', '#f6e7ae')), wetcol='#4a5a4c')
    elif name == 'rain':
        L = Look(ramp={0: ('#17191b', '#2a2d2f', '#424647', '#646868', '#8f9290', '#c6c7c3')}, light=(.30, .62, .72), bounce=(-.85, -.4, .2), fog=dict(col='#8d908f', col2='#a6a8a6', dens=.0030, start=6, hf=.0, floor=1., max=.94),
                 par=dict(key=.40, amb=.14, sky=.22, side=.12, hue=.04, crack=.9), mats={1: dict(wet=.75), 0: dict(wet=.45)}, water=dict(scale=8., refl=.75, glint=0., foam=.55, distort=24, cols=('#34393b', '#858988', '#2a2f31', '#9ea2a0', '#ffffff', '#b9bcba', '#d7dad8')), wetcol='#b4b7b5')
    elif name == 'first':                                                          # G6-G8: last of the night, first dawn colour low on the right
        L = Look(ramp={0: ('#0c0c1c', '#1c1c38', '#37345f', '#6f5a86', '#c58a8e', '#ffd0a8')}, light=(.88, .30, .30), bounce=(-.85, -.4, .25), glint=(.6, .8, .05), fog=dict(col='#4a4a80', col2='#d79a98', dens=.0010, start=40, hf=.003, floor=.35, pw=3., max=.9),
                 par=dict(key=.66, amb=.10, sky=.14, side=.14), water=dict(scale=9., refl=.95, glint=.8, foam=.8, distort=30, sigma=.2, cols=('#12142c', '#4f4d80', '#0c0d20', '#8a74a0', '#ffe2c4', '#b9a3c4', '#ffe9d8')), wetcol='#8a7aa0')
    elif name == 'gold':
        L = look('night'); L = Look(ramp={0: ('#0c0912', '#1f1724', '#3d2c36', '#7a4f3a', '#d08a36', '#ffe06a')}, light=(-.2, -.75, .62), bounce=(.9, -.2, .2), fog=dict(col='#2a2030', col2='#5a3a2a', dens=.0010, start=40, floor=.4, max=.9), water=dict(scale=9., refl=.9, glint=.6, foam=.8, distort=30, cols=('#0a0a14', '#2c2230', '#06060c', '#6a4a34', '#ffe9a0', '#b58a5a', '#ffedb0')))
    elif name == 'lake':
        L = Look(ramp={0: ('#1c2d2a', '#33493a', '#5f7350', '#9b9a66', '#d8c286', '#fff0bf'), 3: ('#16302c', '#23503c', '#3d7a3f', '#78a542', '#b9cd52', '#f1ea86'), 4: ('#1d3a2c', '#2f5a38', '#4f7f3d', '#7fa945', '#bccf58', '#f3ee94'), 2: ('#22302f', '#3d4c48', '#6b7467', '#a59f80', '#d9c99b', '#fff0cc')},
                 light=(-.62, .34, .62), bounce=(.8, -.5, .25), fog=dict(col='#a9cfc6', col2='#ffe9b0', dens=.0024, start=26, hf=.0, floor=1., pw=2.5, max=.82), par=dict(key=.62, amb=.12, sky=.14, side=.10, hue=.12, turf=.0),
                 mats={3: dict(canopy=1., crown=5.2, chisel=.5, crack=0, strata=0, key=.60, hue=.16), 4: dict(canopy=0., chisel=.4, crack=0., fcell=.3, hue=.14), 0: dict(turf=1., cell=.5), 2: dict(cell=.12, fcell=.1)},
                 turf=('#1d3a2c', '#37653a', '#629142', '#9cbf4c', '#e0e477'), water=dict(scale=2.6, refl=1., glint=.55, foam=0., distort=15, sigma=.16, clear=.16, cols=('#1d5148', '#79b0a0', '#163f3a', '#b9d99a', '#fff6cf', '#e8f2d8', '#fdfbe0')))
    elif name == 'dawn':
        L = Look(ramp={0: ('#2c2b45', '#4b4568', '#7a6a8c', '#b58f9f', '#eab8a8', '#ffe4cc'), 3: ('#26324a', '#3a4f5c', '#5d7a6c', '#9a9f80', '#d9c096', '#ffe8c0'), 4: ('#22384a', '#38575a', '#557660', '#7c9468', '#b5b27c', '#f2e2b0')},
                 light=(.70, .58, .34), bounce=(-.8, -.4, .3), glint=(.289, .957, .12), fog=dict(col='#c9bcd8', col2='#ffd6c0', dens=.0042, start=24, hf=.012, floor=.42, pw=2.2, max=.86), par=dict(key=.62, amb=.12, sky=.16, side=.12),
                 mats={4: dict(canopy=0., chisel=.4, crack=0., fcell=.3, hue=.12), 0: dict(turf=.9), 2: dict(cell=.12, fcell=.1), 3: dict(canopy=1., crown=4., chisel=.4, crack=0., strata=0.)}, turf=('#24384a', '#3c5a5a', '#5f8063', '#8fa374', '#cfc48e'),
                 water=dict(scale=3.4, refl=1., glint=.9, foam=0., distort=12, sigma=.12, clear=.2, stroke=.55, cols=('#6d7fa8', '#e0b6b8', '#5f6f9c', '#ffd2bd', '#fff4e0', '#fff0e4', '#fff8ee')))
    LOOKS[name] = L; return L


# ================================================================================================ the mesa coast
PATH = np.array([(0, -40, 62), (8, 60, 60), (34, 170, 58), (-30, 310, 54), (-58, 440, 50), (6, 580, 56), (74, 720, 60), (52, 880, 58), (-36, 1030, 56), (-50, 1190, 58), (20, 1360, 60), (60, 1540, 62)], float)
TOP = 34.                          # height of the heroes' table above the sea


def _cr(P, n=24):
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]]); out = []
    for i in range(1, len(Q) - 2):
        u = np.linspace(0, 1, n, endpoint=False)[:, None]; a, b, c, d = Q[i - 1:i + 3]; out.append(.5 * (2 * b + (c - a) * u + (2 * a - 5 * b + 4 * c - d) * u ** 2 + (3 * b - a - 3 * c + d) * u ** 3))
    return np.vstack(out + [P[-1:]])


_PP = _cr(PATH); _PS = np.r_[0, np.cumsum(np.linalg.norm(np.diff(_PP, axis=0), axis=1))]


def path(s):
    """point on the flight corridor at arc length s (metres, 0..~1640)"""
    s = np.asarray(s, float); return np.stack([np.interp(s, _PS, _PP[:, k]) for k in range(3)], -1)


def _pdist(x, y): return np.hypot(_PP[:, 0] - x, _PP[:, 1] - y).min()


def chase_cam(s, side=0., up=0., ahead=70., f=1150., roll=None, aim=(0, 0), w=W, h=H):
    """flying camera in the corridor; banks into the bends by itself (roll=None)"""
    p, a, q = path(s), path(s + ahead), path(s + 6); t = (q - p) / np.linalg.norm(q - p); n = np.array([t[1], -t[0], 0.])
    b = path(s + 30) - path(s - 10); c = path(s + 70) - path(s + 30); k = math.atan2(b[0] * c[1] - b[1] * c[0], b[0] * c[0] + b[1] * c[1])
    return Camera3D(p + n * side + [0, 0, up], a + n * (side * .35 + aim[0]) + [0, 0, up * .35 + aim[1]], f, w, h, roll=(-.5 * k if roll is None else roll))


_co = {}


def _coast():
    if 'G' in _co: return _co
    r = np.random.default_rng(5); spec = []
    def add(x, y, R, H, kind, **k): spec.append(dict(c=(x, y), R=R, H=H, kind=kind, seed=len(spec) * 37 + 5011, **k))
    add(-8, -70, 92, TOP, 'mesa', sides=44, talus=.3, caps=False, asp=1.15, clefts=0)
    for s in np.arange(40, 1640, 46):
        p = path(s); q = path(s + 2); t = (q - p)[:2]; t /= np.linalg.norm(t); n = np.array([-t[1], t[0]])
        for sg in (-1, 1):
            if r.random() < .10: continue
            kind = r.choice(['mesa', 'mesa', 'pillar', 'stack', 'butte']); R = r.uniform(15, 34) if kind in ('mesa', 'butte') else r.uniform(9, 17); d = 30 + R + r.uniform(0, 26)
            x, y = p[:2] + n * sg * d + t * r.uniform(-14, 14)
            if _pdist(x, y) < R * 1.25 + 20 or y < 60: continue
            if any(np.hypot(x - v['c'][0], y - v['c'][1]) < (R + v['R']) * .95 for v in spec): continue
            add(x, y, R, r.uniform(74, 168), kind, rot=r.uniform(0, 3), asp=r.uniform(.8, 1.35), lean=tuple(r.normal(0, .015, 2)))
    k = 0
    while k < 84:
        y = 90 + 2400 * r.random() ** .8; x = r.normal(0, 260 + .42 * y); kind = r.choice(['mesa', 'mesa', 'butte', 'pillar', 'stack', 'fin']); R = r.uniform(18, 62) if kind in ('mesa', 'butte', 'fin') else r.uniform(10, 24)
        if _pdist(x, y) < R + 70: k += .02; continue
        if any(np.hypot(x - v['c'][0], y - v['c'][1]) < (R + v['R']) * 1.02 for v in spec): k += .02; continue
        add(x, y, R, r.uniform(60, 190) * (1.15 if kind in ('pillar', 'stack') else 1), kind, rot=r.uniform(0, 3), asp=r.uniform(.75, 1.5) if kind != 'fin' else r.uniform(1.8, 2.6), lean=tuple(r.normal(0, .012, 2))); k += 1
    M = Mesh(); own = []; count = lambda: sum(len(t) for t in M.t)
    for v in spec: a = count(); mesa(M, **v); own.append((a, count()))
    # loose rock and plates on the heroes' table
    pebbles(M, 170, (-95, 80, -150, 14), lambda x, y: np.full_like(np.asarray(x, float), TOP), r, .25, 2.4, p=4., keep=lambda x, y: np.hypot((x + 8) / 1.15, y + 70) < 84)
    G = M.build(); k = np.nonzero(np.linalg.norm(np.cross(np.concatenate(M.t)[:, 1] - np.concatenate(M.t)[:, 0], np.concatenate(M.t)[:, 2] - np.concatenate(M.t)[:, 0]), axis=1) > 1e-7)[0]
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.cross(np.concatenate(M.t)[:, 1] - np.concatenate(M.t)[:, 0], np.concatenate(M.t)[:, 2] - np.concatenate(M.t)[:, 0]), axis=1) > 1e-7)]
    a0, b0 = int(cum[own[0][0]]), int(cum[own[0][1]]); tp = (G.FN[a0:b0, 2] > .7) & (G.T[a0:b0, :, 2].min(1) > TOP - 2); G.MAT[a0:b0][tp] = 1
    _co.update(spec=spec, G=G, own=[(int(cum[a]), int(cum[b])) for a, b in own], sh={}, fm=foam_map(G, (-600, 600, -200, 1400)))
    return _co


def PILLARS(dmax=110):
    """mesas and pillars standing beside the corridor: dict(id, c, R, H, kind, s = where on the path)"""
    out = []
    for i, v in enumerate(_coast()['spec']):
        d = np.hypot(_PP[:, 0] - v['c'][0], _PP[:, 1] - v['c'][1])
        if i and d.min() < dmax: out.append(dict(id=i, c=v['c'], R=v['R'], H=v['H'], kind=v['kind'], s=float(_PS[d.argmin()]), dist=float(d.min())))
    return sorted(out, key=lambda v: v['s'])


def _shadow(name, box=(-500, 500, -160, 1100, 0, 200), res=3072):
    c = _coast()
    if name not in c['sh']: c['sh'][name] = shadow_map(c['G'], look(name).light, res, box)
    return c['sh'][name]


def coast_sky(cam, time='night', moon='auto', seed=3):
    """default sky of the coast as this camera sees it, and its mirror image for the water.  moon 'auto' = drawn at direction MOON, None = no disc"""
    P = SK.pano(time, MOON if (time == 'night' and moon is not None) else None, seed); a, b = SK.view(cam, P), SK.mirror(cam, P)
    if time == 'night' and moon is not None: a = SK.disc(cam, a, MOON); b = SK.disc(cam, b, (MOON[0], MOON[1], -MOON[2]))
    return a, b


def coast(cam, t=0., time='night', moon='auto', rings=(), fires=(), dyn=None, hide=(), sky=None, ss=2, reflect=True, key=1.):
    c = _coast(); G = c['G']
    if len(hide):
        k = np.ones(G.n, bool)
        for i in hide: a, b = c['own'][i]; k[a:b] = False
        G = G.sel(k)
    sr = None
    if sky is None: sky, sr = coast_sky(cam, time, moon)
    return render(cam, G, look(time), t, sky, _shadow(time), c['fm'], fires, rings, ss, reflect, dyn=dyn, key=key, sky_reflect=sr)


# ================================================================================================ pieces that move
def break_pillar(i, z, d, u, seed=0, power=1.):
    """what pillar i looks like u seconds after something went through it at height z in direction d=(x, y): use with hide=[i].
    stump with a torn lip, the head tipping over, a burst of individually cut blocks under gravity"""
    c = _coast(); a, b = c['own'][i]; g = c['G']; m = np.zeros(g.n, bool); m[a:b] = True; return _break(g.sel(m), c['spec'][i], z, d, u, seed + i * 71, power)


def _break(P, v, z, d, u, seed=0, power=1.):
    cz = P.T[..., 2].mean(1); r = np.random.default_rng(seed); hb = min(.09 * v['H'], 12.)
    d = np.r_[np.asarray(d, float)[:2], 0.]; d /= np.linalg.norm(d); u = max(0., float(u)); x = np.array([*v['c'], 0.]) + np.r_[np.asarray(v.get('lean', (0, 0))) * z, 0]
    jag = r.normal(0, .25, len(cz)) * hb
    stump = P.sel(cz < z - hb + jag); head = P.sel(cz > z + hb + jag); ax = np.cross([0, 0, 1.], d); w = .55 * power * u * u + .25 * u; w = min(w, 1.9); K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]); R = np.eye(3) + math.sin(w) * K + (1 - math.cos(w)) * K @ K
    head = head.moved(R, d * (3.5 * u + 5 * u * u) * power + [0, 0, -min(9.81 * .5 * u * u * 1.2, z * .95)], x + [0, 0, z + hb]) if head.n else None
    M = Mesh(); R0 = v['R']
    for _ in range(22):                                                             # torn lips of stump and head
        th = r.uniform(0, 6.28); q = R0 * r.uniform(0, .95); boulder(M, x + [math.cos(th) * q, math.sin(th) * q, z - hb * r.uniform(.7, 1.4)], R0 * r.uniform(.16, .36), r)
    lip = M.build(); M = Mesh()
    for _ in range(18): th = r.uniform(0, 6.28); q = R0 * r.uniform(0, .9); boulder(M, x + [math.cos(th) * q, math.sin(th) * q, z + hb * r.uniform(.8, 1.5)], R0 * r.uniform(.16, .34), r)
    lip2 = M.build(); lip2 = lip2.moved(R, d * (3.5 * u + 5 * u * u) * power + [0, 0, -min(9.81 * .5 * u * u * 1.2, z * .95)], x + [0, 0, z + hb]) if head is not None else None
    out = [stump, head, lip, lip2]
    for k in range(34):                                                             # the burst
        th = r.uniform(0, 6.28); q = R0 * r.uniform(.1, .95); s = R0 * r.uniform(.07, .30) * (1.2 if k < 8 else .8); p0 = x + [math.cos(th) * q, math.sin(th) * q, z + r.uniform(-hb, hb)]; vel = (d * r.uniform(10, 46) * (1.1 - s / R0) + np.array([math.cos(th), math.sin(th), 0]) * r.uniform(4, 18) + [0, 0, r.uniform(-2, 20)]) * power
        M = Mesh(); boulder(M, (0, 0, 0), s, r); ax_ = r.normal(0, 1, 3); ax_ /= np.linalg.norm(ax_); w = r.uniform(-3.5, 3.5) * u; K = np.array([[0, -ax_[2], ax_[1]], [ax_[2], 0, -ax_[0]], [-ax_[1], ax_[0], 0]])
        p = p0 + vel * u + [0, 0, -4.905 * u * u]
        if p[2] < -s: continue
        out.append(M.build().moved(np.eye(3) + math.sin(w) * K + (1 - math.cos(w)) * K @ K, p))
    return cat(*out)


_trio = {}
TRIO_HITS = (40, 50, 60)


def _trio_build():
    if _trio: return _trio
    spec = _coast()['spec']; best = None; r = np.random.default_rng(2)
    for _ in range(4000):                                                           # the freest straight lane of 270 m near the corridor, running roughly across the view to the moon
        a = r.uniform(-.55, .55); x0, y0 = r.uniform(-420, -60), r.uniform(180, 900); d = np.array([math.cos(a), math.sin(a)]); cl = min(min(abs((v['c'][0] - x0) * -d[1] + (v['c'][1] - y0) * d[0]) - v['R'] * 1.3 if -60 < (v['c'][0] - x0) * d[0] + (v['c'][1] - y0) * d[1] < 330 else 999 for v in spec), 60)
        if best is None or cl > best[0]: best = (cl, x0, y0, d)
    cl, x0, y0, d = best; o = np.array([x0, y0]); V = 7.5; sp = []; gs = []
    for k, (kind, R, H_, z) in enumerate([('pillar', 11., 118., 66.), ('stack', 12., 104., 63.), ('butte', 15., 112., 60.)]):
        c = o + d * (45 + 75. * k); v = dict(c=tuple(c), R=R, H=H_, kind=kind, seed=881 + 37 * k, rot=.4 * k, lean=(0, 0)); M = Mesh(); mesa(M, **v); sp.append((v, z)); gs.append(M.build())
    _trio.update(o=o, d=d, n=np.array([d[1], -d[0]]), sp=sp, gs=gs, V=V, clear=cl); return _trio


def trio_point(f):
    """(d2) where the thrown body is at frame f: through the hearts of three pillars at 40, 50, 60, on a falling line.  world xyz"""
    q = _trio_build(); s = 45 + q['V'] * (f - 40); z = 66 - .30 * (f - 40) - .0 * f; p = q['o'] + q['d'] * s; return np.array([p[0], p[1], z])


def trio_cam(f, side=58., back=10., up=5., focal=1150., lead=16.):
    """camera running beside him, on the side from which he flies left to right, a little behind and catching up"""
    q = _trio_build(); p = trio_point(f); g = p.copy(); g[:2] += q['n'] * side - q['d'] * back; g[2] += up; t = p.copy(); t[:2] += q['d'] * lead; t[2] -= 3; return Camera3D(g, t, focal, roll=-.05)


def trio_geo(f, power=1.15):
    """the three pillars at frame f (pass as dyn= to land.coast): whole before the hit, then stump, tipping head and flying blocks"""
    q = _trio_build(); out = []
    for k, ((v, z), g) in enumerate(zip(q['sp'], q['gs'])):
        u = (f - TRIO_HITS[k]) / 24.; out.append(g if u < 0 else _break(g, v, z, q['d'], u + .02, 31 * k + 5, power))
    return cat(*out)


class Fan:
    """ground that opens in plates round a launch.  F = Fan((x, y), R=6); F.geo(u) u 0 shut .. 1 wide open (overshoot welcome)"""

    def __init__(s, c, R=6., z=TOP, n=9, rings=2, seed=0, mat=1):
        r = np.random.default_rng(seed + 5); s.pl = []; a0 = r.uniform(0, 6.28)
        for j in range(rings):
            m = n + j * 4; r0, r1 = (R * (.22 + .62 * j), R * (.84 + .60 * j)); cut = a0 + j * .3 + (np.arange(m + 1) / m) * 2 * np.pi + np.r_[r.uniform(-.10, .10, m), 0]; cut[-1] = cut[0] + 2 * np.pi
            for k in range(m):
                A, B = cut[k] + .012, cut[k + 1] - .012; o = r.uniform(.86, 1.15); th = R * r.uniform(.08, .14); ang = np.linspace(A, B, 4); ro = r1 * o * (1 + r.normal(0, .05, 4)); ri = r0 * (1 + r.normal(0, .12, 4)).clip(.5, 1.5)
                P = np.r_[np.c_[np.cos(ang) * ro, np.sin(ang) * ro], np.c_[np.cos(ang[::-1]) * ri, np.sin(ang[::-1]) * ri]]; M = Mesh(); m8 = len(P)
                rows = [(.80, -th * 1.7), (.97, -th), (1.02, -th * .25), (1., 0.)]; cen = P.mean(0); rg = np.stack([np.c_[cen + (P - cen) * f * (1 + r.normal(0, .02, (m8, 1)) * (f < 1)), np.full(m8, z_) + r.normal(0, .05 * th, m8) * (f < 1)] for f, z_ in rows]); sd = int(r.integers(8000))
                body(M, rg, np.stack([np.full(m8, v) for v in (.45, .6, .85, 1.)]), top=0., mat=mat, seed=sd); g = M.build(); mid = (A + B) / 2
                s.pl.append((g, np.array([math.cos(mid), math.sin(mid), 0.]), r1 * o, r.uniform(.75, 1.25) * (1.0 - .34 * j), r.uniform(0, .18) + .16 * j, r.normal(0, .16)))
        s.c = np.array([c[0], c[1], z])

    def geo(s, u, lift=.9):
        out = []
        for g, d, ro, am, dl, tw in s.pl:
            q = np.clip((u - dl) / max(1e-3, 1 - dl), 0, 2); ax = np.cross([0, 0, 1.], d); w = lift * am * q; K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]); R = (np.eye(3) + math.sin(w) * K + (1 - math.cos(w)) * K @ K) @ rot_z(tw * q)
            out.append(g.moved(R, s.c + d * ro * .18 * q + [0, 0, ro * .03 * q], d * ro))
        return cat(*out)


class Slabs:
    """torn ground hanging in the air.  S = Slabs((x, y), R=30, n=18); S.geo(u, t): u 0 in the ground .. 1 risen; t seconds, slow turn"""

    def __init__(s, c, R=30., z=TOP, n=18, h=(3, 26), size=(.8, 4.2), seed=0, keep=3.5, mat=1):
        r = np.random.default_rng(seed + 31); s.it = []
        while len(s.it) < n:
            th, q = r.uniform(0, 6.28), R * math.sqrt(r.random()); sz = size[0] + (size[1] - size[0]) * r.random() ** 2
            if q < keep + sz: continue
            M = Mesh(); slab(M, (0, 0, 0), sz, sz * r.uniform(.28, .5), r, mat=mat)
            for _ in range(int(r.integers(2, 6))): boulder(M, np.r_[r.normal(0, sz * .5, 2), -sz * r.uniform(.5, 1.2)], sz * r.uniform(.05, .14), r)
            s.it.append((M.build(), np.array([c[0] + math.cos(th) * q, c[1] + math.sin(th) * q, z]), r.uniform(*h) * (1 - .4 * sz / size[1]), r.normal(0, .25, 3), r.uniform(0, .45), r.uniform(0, 6.28), r.uniform(.2, .6)))

    def geo(s, u, t=0.):
        out = []
        for g, p, h, w, dl, ph, bob in s.it:
            q = np.clip((u - dl) / (1 - dl), 0, 1.6); e = q * q * (3 - 2 * min(q, 1)) if q < 1 else q; R = rot_z(w[2] * t * e) @ rot_x(w[0] * e + .06 * math.sin(t * 1.3 + ph)) @ rot_y(w[1] * e); out.append(g.moved(R, p + [0, 0, h * e + bob * math.sin(t * 1.7 + ph) * e]))
        return cat(*out)


def rubble(c, R, n=60, z=TOP, seed=0, smin=.1, smax=1.6, u=None, up=8., mat=0):
    """stones lying round c; with u (seconds) they are thrown outward and fall again"""
    r = np.random.default_rng(seed + 3); out = []
    for k in range(n):
        th, q = r.uniform(0, 6.28), R * math.sqrt(r.random()); sz = smin + (smax - smin) * r.random() ** 3; M = Mesh(); boulder(M, (0, 0, 0), sz, r, mat=mat); p = np.array([c[0] + math.cos(th) * q, c[1] + math.sin(th) * q, z + sz * .2])
        if u is None: out.append(M.build().moved(None, p)); continue
        v = np.array([math.cos(th), math.sin(th), 0]) * r.uniform(2, 9) + [0, 0, up * r.uniform(.4, 1.2)]; T = 2 * v[2] / 9.81; tt = min(u, T); w = r.normal(0, 3, 3) * tt; out.append(M.build().moved(rot_z(w[2]) @ rot_x(w[0]), p + v * tt + [0, 0, -4.905 * tt * tt]))
    return cat(*out)


def flames(cam, pts, f, z=None, cols=('#b8321c', '#ff7a2a', '#ffd06a')):
    """small burning places: pts rows (x, y, z, height m).  hard-edged, on twos, hidden behind nearer rock when the depth map z is given.  premult RGBA"""
    from .ink import Cel, spl
    from .comp import over as ov
    out = np.zeros((cam.h, cam.w, 4), f32); t = int(f) // 2
    for i, (x, y, zz, hgt) in sorted(enumerate(pts), key=lambda v: -cam.cam(np.array(v[1][:3]))[2]):
        s, zc = cam.project(np.array([x, y, zz], float))
        if zc < 1: continue
        k = cam.f / zc * hgt; c = Cel(cam.w, cam.h, 0, 0, 1); r = np.random.default_rng(i * 97 + 5)
        for lay, (w_, h_, col) in enumerate([(.62, 1., cols[0]), (.46, .74, cols[1]), (.26, .46, cols[2])]):
            top = [(-.5 * w_, 0.)]
            for j in range(5):
                q = (j + .5) / 5; ph = r.uniform(0, 6.28); a = h_ * (.45 + .55 * math.sin(math.pi * q)) * (.75 + .25 * math.sin(t * 1.9 + ph)); dx = .05 * math.sin(t * 1.3 + ph * 2); top += [((q - .5) * w_ + dx, -a), ((q - .5 + .1) * w_, -a * .48)]
            P = spl(np.array(top + [(.5 * w_, 0.)]), n=4, corners=tuple(range(1, 11, 2))) * k + s; c.fill(np.vstack([P, [s + [.2 * w_ * k, .06 * k]], [s + [-.2 * w_ * k, .06 * k]]]), col)
        l = c.rgba()
        if z is not None: l = l * (z > zc * .985)[..., None]
        out = out * (1 - l[..., 3:4]) + l
    return out.astype(f32)


# ================================================================================================ ground seen close, and the waste
_gr = {}


def _ground(kind='table', crater=(), trench=None, seed=7):
    k = (kind, tuple(crater), None if trench is None else str(trench))
    if k in _gr: return _gr[k]
    r = np.random.default_rng(seed); n1 = fbm(512, 512, (60, 22, 8), (1, .5, .25), seed)
    def relief(X, Y, s=1.):
        u, v = np.mod(X * 2.1, 511), np.mod(Y * 2.1, 511); return .11 * s * cv2.remap(n1, u.astype(f32).reshape(1, -1), v.astype(f32).reshape(1, -1), cv2.INTER_LINEAR).reshape(np.shape(X))
    def carve(X, Y):
        z = 0.
        for (cx, cy, R, dep) in crater: q = np.hypot(X - cx, Y - cy) / R; z = z - dep * np.clip(1 - q * q, 0, 1) ** .7 + .22 * dep * np.exp(-((q - 1.12) / .16) ** 2)
        if trench is not None:
            (ax, ay), (bx, by), w, u, dep = trench; d = np.array([bx - ax, by - ay]); L = np.hypot(*d); d = d / L; s_ = (X - ax) * d[0] + (Y - ay) * d[1]; e = (X - ax) * -d[1] + (Y - ay) * d[0]; e = e + .16 * w * np.sin(s_ * .21) + .1 * w * np.sin(s_ * .63 + 1); on = smooth(-w, w, s_) * (1 - smooth(L * u - w, L * u + w * .2, s_)); q = np.abs(e) / w
            z = z + on * (-dep * np.clip(1 - q * q, 0, 1) ** .6 + .42 * dep * np.exp(-((q - 1.22) / .22) ** 2) * (1 + .4 * np.sin(s_ * .9 + np.sign(e) * 2)))
        return z
    M = Mesh()
    if kind == 'table':
        x = np.arange(-60, 48, 1.15); y = np.arange(-120, 13, 1.15)
        def zf(X, Y): q = np.hypot((X + 8) / 1.15, Y + 70) / 80.; return TOP + .06 + relief(X, Y) * (1 - smooth(.9, 1., q)) + carve(X, Y) - .9 * smooth(.92, 1.02, q)
        sheet(M, x, y, zf, mat=1, seed=seed, hole=lambda c: np.hypot((c[:, 0] + 8) / 1.15, c[:, 1] + 70) > 82.); pebbles(M, 260, (-40, 34, -70, 8), zf, r, .05, .55, p=3.5, keep=lambda a, b: np.hypot(a, b + 22) > 2.2)
    else:
        e = np.r_[-np.geomspace(1500, 26, 30), np.linspace(-24, 24, 46), np.geomspace(26, 1500, 30)]; y = np.r_[np.linspace(-40, 60, 70), np.geomspace(62, 2600, 80)]
        zf = lambda X, Y: TOP + relief(X, Y, 1.6) * (1 + np.clip(Y, 0, 900) / 90.) + carve(X, Y); sheet(M, e, y, zf, mat=1, seed=seed); pebbles(M, 240, (-60, 60, -30, 120), zf, r, .12, 1.9, p=4., keep=lambda a, b: np.hypot(a, b) > 3)
        for j in range(46):
            yy = 240 + 1900 * r.random(); xx = r.uniform(60, 420 + .55 * yy) * r.choice([-1, 1]); kd = r.choice(['mesa', 'butte', 'pillar', 'stack', 'fin']); R = r.uniform(14, 60) if kd in ('mesa', 'butte', 'fin') else r.uniform(9, 22); mesa(M, (xx, yy), R, TOP + r.uniform(30, 150), j * 31 + 77, kd, z0=TOP - 6, rot=r.uniform(0, 3), asp=r.uniform(.8, 1.4))
    G = M.build(); sh = {}; _gr.clear(); _gr[k] = (G, zf, sh); return _gr[k]


def ground(cam, t=0., time='night', dyn=None, crater=(), fires=(), rings=(), hide=(), moon='auto', sky=None, ss=2, key=1.):
    """the heroes' table seen close (top at z = TOP = 34, flat place round x=0 y=-22), coast and sea behind.  crater = [(x, y, R, depth)]"""
    g, zf, _ = _ground('table', tuple(crater)); c = _coast(); G = c['G'].sel(np.arange(c['G'].n) >= 0)
    a, b = c['own'][0]; m = np.ones(G.n, bool); T = G.T.mean(1); m[a:b] = ~((T[a:b, 2] > TOP - .5) & (np.hypot((T[a:b, 0] + 8) / 1.15, T[a:b, 1] + 70) < 74)); loose = (T[:, 2] > TOP - .3) & (T[:, 2] < TOP + 3) & (np.hypot((T[:, 0] + 8) / 1.15, T[:, 1] + 70) < 84); m &= ~loose
    for i in hide: m[c['own'][i][0]:c['own'][i][1]] = False
    sr = None
    if sky is None: sky, sr = coast_sky(cam, time, moon)
    if 'f' + time not in c['sh']: c['sh']['f' + time] = shadow_map(cat(c['G'].sel(m), g) if not len(hide) else c['G'], look(time).light, 2560, (-46, 34, -92, 8, TOP - .5, TOP + 2.5))
    return render(cam, cat(G.sel(m), g), look(time), t, sky, [c['sh']['f' + time], _shadow(time)], c['fm'], fires, rings, ss, True, dyn=dyn, key=key, sky_reflect=sr)


def waste(cam, t=0., time='night', dyn=None, trench=None, crater=(), fires=(), moon='auto', sky=None, ss=2, key=1.):
    """the flat waste behind the coast (z = TOP, +Y to the horizon, mesas on both sides).  trench = ((x0, y0), (x1, y1), half width, u 0..1, depth)"""
    g, zf, sh = _ground('waste', tuple(crater), trench)
    if time not in sh: sh[time] = shadow_map(g, look(time).light, 2560, (-420, 420, -60, 900, TOP - 8, 200))
    if sky is None: sky, _ = coast_sky(cam, time, moon)
    return render(cam, g, look(time), t, sky, sh[time], None, fires, (), ss, False, dyn=dyn, key=key, water=False)


def ground_z(x, y, kind='table', crater=(), trench=None):
    """height of the ground (metres) so that feet can stand on it"""
    return _ground(kind, tuple(crater), trench)[1](np.asarray(x, float), np.asarray(y, float))


# ================================================================================================ C4: the far ridge
_rd = {}


def _ridge():
    if _rd: return _rd
    n = fbm(400, 1600, (190, 90, 40), (1, .5, .25), 31); ridg = 1 - np.abs(fbm(400, 1600, (170, 74, 34), (1, .45, .22), 35)) * .62
    M = Mesh(); x = np.linspace(-5200, 5200, 260); y = np.r_[np.linspace(3500, 5400, 30)]
    prof = np.interp(x, [-5200, -3600, -2300, -1500, -700, 0, 600, 1500, 2400, 3600, 5200], [190, 400, 300, 600, 470, 860, 540, 720, 360, 470, 200])
    def zf(X, Y):
        u = ((X + 5200) / 10400 * 1599); v = (Y - 3500) / 1900 * 399; s = lambda a: cv2.remap(a.astype(f32), u.astype(f32), v.astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        q = (Y - 3500) / 1900.; e = np.sin(np.pi * np.clip(q * .92 + .02, 0, 1)) ** .8; return -30 + e * np.interp(X, x, prof) * np.clip(.25 + .75 * s(ridg), 0, 1.2) + 34 * s(n) * e
    sheet(M, x, y, zf, mat=5, seed=4, jit=.4); G = M.build(); C_ = G.T.mean(1); _rd.update(G=G, C=C_); return _rd


def ridge_cam(f=1500., h=9.): return Camera3D((0, 0, h), (0, 1000, h), f, cy=640)


def ridge(u_line=0., u_lift=0., t=0., cam=None, rings=(), time='night', ss=2, light='#f3d9ff', core='#ffffff', zip_x=0.):
    """(c4) wide level sea, far mountain ridge.  u_line 0..1: a hair of violet drawn along the crest from left to right; u_lift 0..: the ridge unzips
    from the crest at zip_x outward, lifts in blocks and turns into light.  -> Out with .crest (k,2) px of the crest line, .glow mask"""
    d = _ridge(); cam = cam or ridge_cam(); G = d['G']; L = look(time)
    if 'look' not in d: LK = Look(ramp={0: ('#080715', '#14122a', '#262250', '#4a3f88', '#7a68be', '#b9a4ee')}, light=(.80, -.05, .6), bounce=(-.9, -.4, .2), glint=MOON, fog=dict(col='#3a2b78', col2='#5d40ac', dens=.00012, start=200, floor=.55, hf=.0012, max=.8), par=dict(key=.62, amb=.10, sky=.10, side=.12, cell=22., fcell=20., crack=.25, strata=.15, chisel=.8, brush=1.), water=dict(scale=9., refl=.9, glint=1., foam=.0, distort=30, sigma=.15, cols=('#080817', '#2a2258', '#06060f', '#3a3072', '#f1e6ff', '#7d6fbc', '#f6efff'))); d['look'] = LK
    sky, sr = coast_sky(cam, time, 'auto' if time == 'night' else None)
    base = render(cam, G, d['look'], t, sky, None, None, (), rings, ss, True, sky_reflect=sr); o = base
    crest = []; m = (base.id >= 0); xs = np.arange(0, cam.w, 3)
    for xx in xs: ys = np.nonzero(m[:, xx])[0]; crest.append((xx, ys[0] if len(ys) else horizon(cam)))
    o.crest = np.array(crest, float)
    if u_lift > 0:
        # the ridge is a picture: it is cut along drawn cracks into slabs, and every slab is lifted off the light underneath
        w, h = cam.w, cam.h; hz = float(horizon(cam)); x, y = grid(h, w); zx = cam.project(np.array([zip_x, 4400, 200.]))[0][0]; mk = (base.rock * (y < hz + 1)).astype(f32)
        key = ('pc', w, h, round(hz), round(cam.f))
        if d.get('pk') != key:
            r = np.random.default_rng(9); ys, xs = np.nonzero(mk[::2, ::2] > .5); n = 120; k = r.choice(len(xs), n); sx, sy = xs[k] * 2. + r.uniform(0, 2, n), ys[k] * 2. + r.uniform(0, 2, n); wx = x + 16 * fbm(h, w, (90, 30, 9), (1, .6, .4), 3); wy = y + 14 * fbm(h, w, (90, 30, 9), (1, .6, .4), 4); best = np.full((h, w), 1e9, f32); lab = np.zeros((h, w), np.int32)
            for j in range(n): q = ((wx - sx[j]) / 1.0) ** 2 + ((wy - sy[j]) / .62) ** 2; m_ = q < best; best[m_] = q[m_]; lab[m_] = j
            lab[mk < .02] = -1; pc = []
            for j in range(n):
                m_ = lab == j
                if m_.sum() < 60: continue
                yy, xx = np.nonzero(m_); a0, b0, a1, b1 = max(0, xx.min() - 3), max(0, yy.min() - 3), min(w, xx.max() + 4), min(h, yy.max() + 4); al = cv2.GaussianBlur((m_[b0:b1, a0:a1] * mk[b0:b1, a0:a1]).astype(f32), (0, 0), .6); pc.append((a0, b0, al, xx.mean(), yy.mean(), r.random(5)))
            d['pk'] = key; d['pc'] = sorted(pc, key=lambda v: v[4])
        rgb_ = base.rgb / np.maximum(base.a, .05)[..., None]; reach = np.clip(u_lift * 2.1 - np.abs(x - zx) / (w * .62), 0, 1.6); up_ = u_lift
        n1 = fbm(h, w, (60, 24, 9), (1, .6, .35), 12 + int(t * 12) % 5, aspect=.5); top = np.interp(np.arange(w), base.crest[:, 0], base.crest[:, 1])[None]; hgt = np.clip((hz - y) / np.maximum(hz - top, 1), 0, 9)              # 1 on the crest
        fl = 1.02 + (.14 * n1 + 2.2 * up_ ** 2 * np.exp(-((x - zx) / (w * (.10 + .22 * up_))) ** 2)) * np.clip(reach, 0, 1); hotm = np.clip((fl - hgt) * 26, 0, 1) * np.clip(reach * 3, 0, 1) * (y < hz + 1.5); rimv = np.clip((fl + .16 - hgt) * 20, 0, 1) * np.clip(reach * 3, 0, 1) * (y < hz + 1.5)
        img = sky + (C('#8a2cff') * 1.0 - sky) * (rimv * .85)[..., None]; img = img + (C(light) * 1.05 - img) * (np.clip((fl + .05 - hgt) * 22, 0, 1) * np.clip(reach * 3, 0, 1) * (y < hz + 1.5))[..., None]; img = img + (C(core) * 1.3 - img) * hotm[..., None]; img = img + C('#8a2cff') * (blur(hotm, 46) * .55 + blur(hotm, 150) * .35)[..., None]
        for (a0, b0, al, mx_, my_, q) in d['pc']:
            dl = abs(mx_ - zx) / (w * .62) * .95 + .30 * (1 - min(1., (hz - my_) / 190.)) + .10 * q[0]; v = min(max(0., up_ * 2.1 - dl), 2.2); sc = max(.08, 1 - .30 * v ** 1.4); rot = (q[1] - .5) * 60 * v; dy = -v ** 1.7 * (70 + 150 * q[2]); dx = v ** 1.3 * ((mx_ - zx) * .16 + (q[3] - .5) * 50); ph, pw = al.shape; cxp, cyp = mx_ - a0, my_ - b0; pad = int(8 + .35 * max(pw, ph) * (v > 0)); M_ = cv2.getRotationMatrix2D((cxp, cyp), rot, sc); M_[:, 2] += pad; col = rgb_[b0:b0 + ph, a0:a0 + pw]; hot_ = min(1., max(0., v - .25) * .85); col = col + (C(light) * 1.1 - col) * hot_ ** 1.5
            src = np.dstack([col * al[..., None], al]).astype(f32); W_, H_ = pw + 2 * pad, ph + 2 * pad
            if v > .02: e = np.clip(al - cv2.warpAffine(al, np.float32([[1, 0, 0], [0, 1, -3.2]]), (pw, ph)), 0, 1); src[..., :3] = src[..., :3] * (1 - e[..., None]) + C(core) * 1.2 * e[..., None] * al[..., None]
            ox, oy = int(math.floor(a0 - pad + dx)), int(math.floor(b0 - pad + dy)); M_[0, 2] += a0 - pad + dx - ox; M_[1, 2] += b0 - pad + dy - oy; pc = cv2.warpAffine(src, M_, (W_, H_), flags=cv2.INTER_LINEAR); from .ink import over_roi; over_roi(img, pc, ox, oy)
        sea = ((y >= hz - .5))[..., None]; hy = int(round(hz)); mir = np.zeros_like(img); n_ = min(h - hy, hy); mir[hy:hy + n_] = img[hy - n_:hy][::-1]; jx = (6 + 30 * np.clip((y - hz) / 300, 0, 1)) * fbm(h, w, (14, 5), (1, .6), 7 + int(t * 12), aspect=14.); mir = cv2.remap(mir, (x + jx).astype(f32), (y + .3 * jx).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101); k_ = np.exp(-np.clip(y - hz, 0, None) / 240.)[..., None] * np.clip(lum(mir) * 1.1 - .12, 0, .9)[..., None]
        o.img = np.where(sea, base.img + (mir - base.img) * k_, img); o.glow = hotm
    if u_line > 0 and u_lift <= .05:
        n = int(len(o.crest) * min(u_line, 1)); P = o.crest[:max(2, n)]; a = np.zeros((cam.h, cam.w), f32); cv2.polylines(a, [(P * 16).astype(np.int32)], False, 1., 1, cv2.LINE_AA, 4); hd = P[-1]; x, y = grid(cam.h, cam.w); spark = np.exp(-(((x - hd[0]) ** 2 + (y - hd[1]) ** 2) / 160.)) * (u_line < 1)
        o.img = o.img + (C('#e8c8ff') * 1.2 - o.img) * a[..., None] + C('#8a2cff') * (blur(a, 3) * 1.2 + blur(a, 14) * .8)[..., None] + C('#f4e6ff') * spark[..., None]
    return o


from .land_sets import (cliff, cliff_e1, cliff_ground, cliff_cam, CLIFF, rain_sky, puddle, puddle_cam, puddle_z, lake, lake_cam, lake_z, lake_tree, crater_top, islet_layers, top_cam, islet_z, crater_eye, eye_cam, eye_z, dawn_pano, dawn_up, hill)
from .land_grow import rain, rain_rings, splashes, lily, blades, leaves, tree
