"""PHYS KIT: ready hits for figure shots.  Build once in setup(); every .method(img, cam, t) returns the new RGB frame; t = seconds since the contact (negative = nothing yet).  World metres, ground z = floor.
    P = BootPlant((x, y), size=1.0, seed=1, pal='night')          d1 boots: cracks, hopping pebbles, a low rolling skirt of dust
    K = Crater((x, y), rad=2.2, power=1.0, seed=2, pal='night')   dropped wristband, landing, launch: plates with thickness, rigid rubble that lands, dust ring.  lift=True launch fan (plates stand up)
        img = K.under(img, cam, t)  (cracks, pit, plates)   img = K.back(img, cam, t)  (dust behind the figure)   ... figure ...   img = K.front(img, cam, t)  (rubble, near dust)
    B = PillarBurst((x, y), h=14, r=1.8, z=7, dirn=(1, 0, .1), speed=24, seed=3, pal='night')     d2: a body goes through it at height z.   img = B.draw(img, cam, t)
    C = RainBlast(apex, dirn, half=.55, c=46)                     g1: img = C.draw(img, cam, t, tau)   rain + cone + rings; the cone stays clean
    T = Plough(path_xy, width=2.4, depth=1.1, v0=180, k=1.9)      d5 / d2: img = T.draw(img, cam, t)
top = '#hex' of your ground, so the lifted plates match it.  light = direction TO the key (world), rim = direction to a second hard light (fire, aura) or None, dl = SCREEN light for dust (x right, y down, z to us)"""
from .phys import *


def _pal_(p): return ROCK[p] if isinstance(p, str) else p


class _Base:
    def _l(s, pal, light, rim, dl, shadow): s.pal, s.light, s.rim, s.dl = pal, light, rim, dl; s.shadow = shadow or ROCK[pal]['pit'] if isinstance(pal, str) else shadow; s.kw = dict(light=light, rim_dir=rim, shadow=s.shadow)
    def dust(s, img, cam, L, ss=1.25): return over(img, lobes_layer(cam, L, s.pal, light=s.dl, ss=ss, rim=None if s.rim is None else (s.rim[0], -s.rim[2] - .4))) if L else img


class BootPlant(_Base):
    def __init__(s, xy=(0, 0), size=1.0, seed=0, pal='night', light=(.5, -.3, .8), rim=(-.8, -.2, .1), dl=(.5, -.75, .45), floor=0., shadow=None, top=None):
        s._l(pal, light, rim, dl, shadow); s.skin = None if top is None else {**_pal_(pal), 'lit': top, 'fresh': top}; s.xy, s.z, s.sz, s.seed, s.fl = np.asarray(xy, float), size, size, seed, floor
        s.fr = Fracture(xy, .62 * size, seed, 9, 2, .07 * size, .26, 26, .05, .05, cracks=1.5); s.rb = Rubble(seed); s.rb.burst((*xy, floor + .02), 22, (.6 * size, 2.4 * size), 1.0, (.02 * size, .10 * size), seed=seed + 3, jit=(.25 * size, .16 * size, .0), floor=floor); s.rb.bake(60)
    def under(s, img, cam, t):
        if t < 0: return img
        c = Cel(); s.fr.draw(c, cam, tw(t * 24 + .5, 2) / 24., s.pal, light=s.light, rim_dir=s.rim, shadow=s.shadow, lw=1.8, floor=s.fl, skin=s.skin); return over(img, c.layer())
    def _d(s, t): return dust_ring(tw(t * 24 + .5, 2) / 24., s.xy, .16 * s.sz, 2.6 * s.sz, .20 * s.sz, 18, 2, s.seed + 5, .75, 3.6, s.fl, rise=.12 * s.sz)
    def back(s, img, cam, t): return s.dust(img, cam, [q for q in s._d(t) if q[1] > s.xy[1]]) if t > 0 else img
    def front(s, img, cam, t):
        if t < 0: return img
        c = Cel(); s.rb.draw(c, cam, t * 24, s.pal, step=2, lw=1.8, **s.kw); grit(c, cam, t, (*s.xy, s.fl), 40, (1. * s.sz, 4. * s.sz), .9, (.006, .018), seed=s.seed, floor=s.fl); img = over(img, c.layer()); return s.dust(img, cam, [q for q in s._d(t) if q[1] <= s.xy[1]])
    def draw(s, img, cam, t): return s.front(s.back(s.under(img, cam, t), cam, t), cam, t)


class Crater(_Base):
    def __init__(s, xy=(0, 0), rad=2.2, power=1.0, seed=0, pal='night', light=(.5, -.3, .8), rim=(-.8, -.2, .1), dl=(.5, -.75, .45), floor=0., lift=False, n=70, shadow=None, top=None):
        s._l(pal, light, rim, dl, shadow); s.skin = None if top is None else {**_pal_(pal), 'lit': top, 'fresh': top}; s.xy, s.rad, s.pw, s.seed, s.fl = np.asarray(xy, float), rad, power, seed, floor; s.fr = Fracture(xy, rad, seed, 13, 4, .10 * rad, 1.05 if lift else .62, 14 * rad ** .5 + 10, .09 * rad, .10 * rad, cracks=1.2); v = math.sqrt(rad) * power
        s.rb = Rubble(seed); s.rb.burst((*xy, floor + .05), n, (2.2 * v, 7.5 * v), .95, (.03 * rad, .30 * rad), seed=seed + 3, jit=(.25 * rad, .25 * rad, .02), floor=floor); s.rb.burst((*xy, floor + .05), n // 2, (2 * v, 9 * v), .7, (.015 * rad, .08 * rad), seed=seed + 8, floor=floor); s.rb.bake(96)
    def under(s, img, cam, t):
        if t < 0: return img
        c = Cel(); s.fr.draw(c, cam, tw(t * 24 + .5, 2) / 24., s.pal, light=s.light, rim_dir=s.rim, shadow=s.shadow, floor=s.fl, skin=s.skin); return over(img, c.layer())
    def _d(s, t): return dust_ring(tw(t * 24 + .5, 2) / 24., s.xy, s.rad * .70, s.rad * 3.2 * s.pw, s.rad * .20, 40, 3, s.seed + 5, 1.1, 2.8, s.fl, rise=.1 * s.rad) + plume(tw(t * 24 + .5, 2) / 24., (*s.xy, s.fl + .1), (0, 0, 1), 3.6 * math.sqrt(s.rad) * s.pw, 22, (.06 * s.rad, .26 * s.rad), .55, .12, s.seed + 9, 1.0, 2.6, .3, s.fl)
    def back(s, img, cam, t): return s.dust(img, cam, [q for q in s._d(t) if q[1] > s.xy[1] + .1 * s.rad]) if t > 0 else img
    def front(s, img, cam, t):
        if t < 0: return img
        c = Cel(); s.rb.draw(c, cam, t * 24, s.pal, step=2, trails=.7, **s.kw); grit(c, cam, t, (*s.xy, s.fl), 110, (2.5 * math.sqrt(s.rad), 9 * math.sqrt(s.rad)), 1., (.006 * s.rad, .02 * s.rad), seed=s.seed, floor=s.fl); img = over(img, c.layer()); return s.dust(img, cam, [q for q in s._d(t) if q[1] <= s.xy[1] + .1 * s.rad])
    def draw(s, img, cam, t): return s.front(s.back(s.under(img, cam, t), cam, t), cam, t)


class PillarBurst(_Base):
    """the body's centre line is (x, y, z) at t = 0, flying along dirn"""
    def __init__(s, xy=(0, 0), h=14., r=1.8, z=7., dirn=(1, 0, .1), speed=24., seed=0, pal='night', light=(.5, -.3, .8), rim=(-.8, -.2, .1), dl=(.5, -.75, .45), floor=0., shadow=None, dust=1.0, cells=34, fog=None):
        s._l(pal, light, rim, dl, shadow); s.o = np.array([*xy, z]); s.d = unit(dirn); s.r, s.sp, s.seed, s.fl, s.du, s.fog = r, speed, seed, floor, dust, fog; s.p = Pillar(xy, h, r, seed); s.p.o = np.array([*xy, floor]); s.p.hit(z - floor, dirn, 0., speed, cells); s.p.bake(96)
    def rocks(s, cel, cam, t, **k): return draw_bodies(cel, cam, s.p.items(-1 if t < 0 else tw(t * 24 + .5, 2)), s.pal, trails=.5, fog=s.fog, **{**s.kw, **k})
    def lobes(s, t): t = tw(t * 24 + .5, 2) / 24.; e = s.o + s.d * s.r; i = s.o - s.d * s.r; return plume(t, e, s.d + [0, 0, .15], s.sp * .70, int(46 * s.du), (.16 * s.r, .70 * s.r), .75, .22, s.seed + 1, 1.6, 2.6, .5) + plume(t, i, -s.d * .6 + [0, 0, .3], s.sp * .28, int(16 * s.du), (.14 * s.r, .42 * s.r), .7, .2, s.seed + 2, 1.3, 2.6, .4) + plume(t - .30, (s.o[0], s.o[1], s.fl + .2), (0, 0, 1), 4., 18, (.2 * s.r, .6 * s.r), 1.2, .4, s.seed + 3, 1.4, 2.4, .3, s.fl) if t > 0 else []
    def draw(s, img, cam, t):
        c = Cel(); s.rocks(c, cam, t); 
        if t > 0:
            grit(c, cam, t, s.o + s.d * s.r, 110, (s.sp * .3, s.sp), .6, (.02, .06), seed=s.seed, dirn=s.d, floor=s.fl)
            if t < .30: ring_flash(c, cam, s.o + s.d * s.r * 1.02, s.d, s.r, t)
        return s.dust(over(img, c.layer()), cam, s.lobes(t)) if t > 0 else over(img, c.layer())


def ring_flash(cel, cam, P, axis, r, t, col='#fbf7ff', life=.30, grow=3.2):
    """the hard white pressure ring where a body leaves a wall: a thin torus in perspective that opens and breaks"""
    q = t / life; a = np.linspace(0, TAU, 80); d = unit(axis); e1 = unit(np.cross(d, [0, .02, 1.])); e2 = np.cross(e1, d); R_ = r * (.6 + grow * (1 - (1 - q) ** 2)); w = r * .55 * (1 - q) ** 1.4; m = np.clip(.55 + .8 * noise(np.cos(a) * 1.7, np.sin(a) * 1.7, r) - .3 * q, 0, 1); f = lambda rr, o: cam.project(np.asarray(P) + d * o + e1 * (np.cos(a) * rr)[:, None] + e2 * (np.sin(a) * rr)[:, None]); (o, zo), (i, zi) = f(R_ * np.ones(80), 0.), f(R_ - w * m, 0.)
    if min(zo.min(), zi.min()) > .4: cel.fill([o, i], col, .95, evenodd=True)
    return cel


class RainBlast:
    """the rain is blown away in a cone.  draw(img, cam, t, tau): t seconds since the blow (before it: plain rain), tau = rain clock (f/24)"""
    def __init__(s, apex, dirn, half=.55, c=46., n=1500, box=(22, 34, 14), seed=5, wind=(1., 0.), pal='grey', col='#f0f3f4', reach=44.): s.a, s.d, s.h, s.c, s.pal, s.col, s.reach, s.seed = apex, dirn, half, c, pal, col, reach, seed; s.rn = Rain(n, box, seed, wind, near=.9)
    def draw(s, img, cam, t, tau, mask=None, cel_only=False):
        img = over(img, s.rn.draw(cam, tau, 1., s.pal, warp=None if t < 0 else cone_warp(s.a, s.d, s.h, t, s.c, 24.), mask=mask))
        if t >= 0: c = Cel(); cone_blast(c, cam, s.a, s.d, s.h, t + .02, s.c, s.col, seed=s.seed, reach=s.reach); img = over(img, c.layer())
        return img


class Plough(_Base):
    """a thrown body ploughs a trench: front(t) = v0 * (1 - exp(-k t)) / k metres"""
    def __init__(s, path, width=2.4, depth=1.1, bank=.75, v0=180., k=1.9, seed=2, pal='night', light=(.5, .25, .8), rim=(-.8, -.2, .1), dl=(.5, -.75, .45), fog=None, n=150, shadow=None): s._l(pal, light, rim, dl, shadow); s.tr = Trench(path, width, depth, bank, seed); s.front = lambda t: v0 * (1 - math.exp(-k * t)) / k if t > 0 else 0.; s.rb = s.tr.rubble(s.front, 96, n, (.10, 1.0), (3, 8.5), seed + 2); s.fog, s.seed, s.w = fog, seed, width * .5
    def draw(s, img, cam, t, split=9.):
        if t <= 0: return img
        t2 = tw(t * 24 + .5, 2) / 24.; sf = s.front(t); c = Cel(); s.tr.draw(c, cam, sf, s.pal, light=s.light, fog=s.fog); img = over(img, c.layer()); D = dust_trail(t2, s.tr.at, s.front, s.tr.L, max(1.1, s.w), s.seed, (.2, .75), 3., 2.2, 1.5, 2., s.w * 1.2); fx = s.tr.front(sf); D2 = plume(.25, (fx[0], fx[1], .3), (0, .6, .6), 16, 30, (.6, 2.2), .9, .2, int(t * 12), 1.2, 2.) if sf < s.tr.L * .96 and s.front(t + .05) - sf > .4 else []
        img = s.dust(img, cam, [q for q in D if q[1] > split] + D2); c = Cel(); s.rb.draw(c, cam, t * 24, s.pal, step=2, trails=.8, fog=s.fog, **s.kw); return s.dust(over(img, c.layer()), cam, [q for q in D if q[1] <= split])
