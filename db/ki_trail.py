"""ki trail: light trails of fighters, clash stars, dents in the sea, holes punched in cloud."""
from .ki_core import *
from .ki_hit import ring3, circle3, ring_strokes, ellipse_pts
from .ki_beam import rays, petals


def trail(px, kind='gold', w=9., seed=1, age=None, scale=None, beads=True, t=0.):
    """one comet trail from a screen polyline px (oldest -> newest): needle tail, hot head, torn outer flame, side sparks -> Field(4)
    scale = per-point size factor (perspective)"""
    P = np.asarray(px, float); n = len(P); F = Field(4, seg_box(P, w * 6 + 20)); v = lin(n); sc = np.ones(n) if scale is None else np.asarray(scale); hw = w * sc * v ** 1.6
    if n < 4: return F
    b = np.clip(1.25 * noise(v * 9 - t * 7, seed * 2.1) + .75 + 1.6 * v, 0, 1) if beads else 1.
    F.rib(P, [hw * 1.0 * (.4 + .6 * b) + .5 * (v > .05), hw * .74 * b, hw * .50 * b - .1, hw * .28 * b - .25], 2.); N = normals(P); T = np.gradient(P, axis=0)
    for i in range(int(n * .34)):
        j = int(n * (.15 + .8 * h1(seed, i, 1))); sg = 1 if i % 2 else -1; l = w * sc[j] * (2.5 + 9 * h1(seed, i, 2)) * v[j] ** 1.5; d = T[j] / (np.hypot(*T[j]) + 1e-6)
        if l < 4: continue
        a = math.atan2(*(-d + N[j] * sg * (.07 + .16 * h1(seed, i, 3)))[::-1]); p = P[j] + N[j] * sg * hw[j] * .7; F.rib(spine(p[0], p[1], l, a, 8, .05, .6, i + t * 8, sg * .2), lwidths(8, l, l * .035 + .4, LV, .25), 1.)
    return F


def trail_cel(F, kind='gold', full=True): P_ = pal(kind); return F.cel(P_['c'], lw=(1.2, 0, 0, 0), ink=(P_['ink'], None, None, None), full=full)


def head(x, y, r, kind='gold', f=0, seed=1, ang=0.):
    """the fighter as a comet: bow shock, hot drop, swept torn tongues, a flare along the flight, crackle.  ang = direction of flight"""
    P_ = pal(kind); d = np.array([math.cos(ang), math.sin(ang)]); n = np.array([-d[1], d[0]]); p = np.array([x, y], float); F = Field(4, (x - r * 9, y - r * 9, x + r * 9, y + r * 9)); G = Field(3, (x - r * 9, y - r * 9, x + r * 9, y + r * 9))
    for sg in (1, -1):                                        # bow wave, then two swept wings
        v = lin(16); Q = p + d * (r * 1.55 - 5.2 * r * v[:, None] ** 1.5) + n * (sg * r * 2.3 * v[:, None] ** .62); G.rib(Q, lwidths(16, r * 6, r * .17, ((1., 1.), (.55, .85), (.2, .6)), .18), 0.); F.rib(p + d * (r * .9 - 4.4 * r * v[:, None]) + n * (sg * r * (.55 + .75 * v[:, None] ** .7)), lwidths(16, r * 5, r * .30, LV, .28), r * .2)
    for i in range(7): z = bms(h1(seed, i, f), .3, .35); a = ang + math.pi + (h1(seed, i, f, 2) - .5) * .7; l = r * (1.8 + 4.8 * z); q = p - d * r * .2 + n * (h1(seed, i, f, 3) - .5) * r * 1.2; F.rib(spine(q[0], q[1], l, a, 10, .07, .8, i + f, 0), kwidths(10, l, r * (.16 + .20 * z), LV), r * .2)
    F.rib(np.array([p + d * r * 1.15, p + d * r * .7, p, p - d * r * 1.3, p - d * r * 3.0]), [np.array([.1, .62, 1., .62, .1]) * r * q for q in (1.15, .95, .72, .46)], r * .25)
    L = [rays(x, y, r, r * 4.4, 2, seed, .05, 1., ang + math.pi / 2, '#ffffff', None, 0.), G.cel((P_['c'][0], P_['c'][2], '#ffffff')), F.cel(P_['c'], lw=(max(1., r * .07), 0, 0, 0), ink=(P_['ink'], None, None, None), seed=f)]; c = Cel()
    for i in range(6): a = ang + math.pi + (h1(seed, i, f, 8) - .5) * 1.6; l = r * (2.5 + 6 * h1(seed, i, f, 9)); s0 = r * (.06 + .10 * h1(seed, i, 10)); q = p + np.array([math.cos(a), math.sin(a)]) * l; e = np.array([math.cos(a), math.sin(a)]) * r * (.4 + h1(seed, i, f, 11)); c.fill(np.array([q - e, q + n * s0, q + e, q - n * s0]), P_['c'][3])
    L.append(c.layer()); return stack(*L)


def clash_star(x, y, r, age, kinds=('gold', 'violet'), seed=1, rot=0., asp=1.):
    """two trails meet: white kite star (age 0), ring and petals in both colours (1-3), hanging sparks (4-8) -> (layer, light) """
    if age < 0 or age > 9: return None, None
    k = (.75, 1., .92, .8, .62, .45, .32, .2, .12, .06)[int(age)]; L = []
    if age < 4: L.append(rays(x, y, r * .5 * k, r * 3.2 * k, 12, seed + int(age), .05, asp, rot, '#ffffff', pal(kinds[0])['c'][2]))
    for j, q in enumerate(kinds):
        P_ = pal(q); F = Field(4, (x - r * 4, y - r * 4, x + r * 4, y + r * 4)); petals(F, x, y, r * (.5 + .22 * age), r * (1.6 + .35 * age), 12, seed + j * 3, age * .04, asp, rot, None, .5, .16 * k, 3., 1., boil=False); L.append(F.cel(P_['c'], lw=(1., 0, 0, 0), ink=(P_['ink'], None, None, None)) * min(1., k * 1.6))
    if age < 3: F = Field(2); F.rib(np.array([[x - .5, y], [x, y], [x + .5, y]]), [np.full(3, r * .62 * k), np.full(3, r * .5 * k)], 1.); L.append(F.cel((pal(kinds[0])['c'][2], '#ffffff')))
    lay = stack(*L); return lay, bloom(lay, .4, (6, 24, 80), (.4, .35, .3), '#f0e2ff', 1.0 * k)


def sea_dent(cam, X, age, size, seed=1, kind='blue', light='#dff0ff', life=26, cols=None):
    """what a clash does to the sea under it: a dish with a hard lit crest, a second swell, foam dashes, a crown of spray -> layer"""
    if age < 0 or age > life: return None
    u = age / float(life); P_ = pal(kind); L = []; F = Field(3); D = Field(1); X = np.asarray(X, float)
    for k in range(3):
        q = age - k * 3.4
        if q <= 0: continue
        rad = size * (.30 + 1.9 * (1 - math.exp(-q / 8.5))) * (1 - .12 * k); k_ = math.exp(-q / (life * .36)); p, z, a = circle3(cam, X, (0, 0, 1), rad, 130, .035, seed + k); sc = cam.f / z; w = size * .085 * sc * k_ * np.clip(.62 + .9 * noise(np.cos(a) * 2, np.sin(a) * 2, 1., seed + k), 0, 1.6)
        F.rib(p, [w + .4, w * .60 - .2, w * .25 - .4], 1.); pi, zi, _ = circle3(cam, X, (0, 0, 1), rad * .90, 130, .04, seed + k + 9); D.rib(pi, [w * 1.5 * np.clip(.7 + noise(np.cos(a) * 3, np.sin(a) * 3, 2., seed), 0, 1.5)], 2.)
    for j in range(60):                                       # foam dashes running out
        a = TAU * h1(seed, j, 1); q0 = size * (.4 + 1.4 * h1(seed, j, 2)); q1 = q0 * (1 + 1.2 * (1 - math.exp(-age / 9.))); ln = size * (.06 + .22 * h1(seed, j, 3)) * (1 - u); r_ = q1 + ln * lin(6); Q, z = cam.project(X + np.stack([np.cos(a) * r_, np.sin(a) * r_, 0 * r_], -1)); w = size * .012 * cam.f / z * (1 - u) * (.4 + h1(seed, j, 4)); F.rib(Q, [w * np.sin(math.pi * lin(6)) + .2, w * .5 * np.sin(math.pi * lin(6)), w * 0 - 1.], 0.)
    S = Field(4)
    if age < 14:                                              # crown of water, thrown up and out
        for j in range(26):
            a = TAU * (j + h1(seed, j, 11)) / 26; g = min(1., (age + 1) / 4.); fall = max(0., 1 - age / 14.); hh = size * (.25 + .9 * bms(h1(seed, j, 12), .25, .3)) * g * fall ** .5; r0 = size * (.22 + .08 * age ** .7); v = lin(9); Q, z = cam.project(X + np.stack([np.cos(a) * (r0 + hh * .55 * v), np.sin(a) * (r0 + hh * .55 * v), hh * v * (1.4 - .4 * v)], -1)); S.rib(Q, [q * size * .05 * cam.f / z * fall for q in lwidths(9, 1., 1., LV, .3)], 2.)
    c = P_['c']; return stack(D.cel(('#060a26',)) * (.75 * (1 - u)), F.cel(cols or (c[0], c[2], '#ffffff')) * min(1., 1.4 * (1 - u)), S.cel(('#3a3f9e', '#8f8ce6', '#dcd6ff', '#ffffff')))


def punch(cloud, cx, cy, r, asp=.7, seed=1, rag=.12, lip=.16, light=('#f4ecff', '#b9a4e6')):
    """hole in a premult cloud layer: matter is pushed outward, the lip rolls over and catches the light -> layer"""
    if r < 2: return cloud
    x, y = grid(); dx, dy = x - cx, (y - cy) / asp; d = np.sqrt(dx * dx + dy * dy) + 1e-3; a = np.arctan2(dy, dx); tb = np.linspace(-np.pi, np.pi, 361); nz = noise(np.cos(tb) * 3.2, np.sin(tb) * 3.2, 0, seed) + .5 * noise(np.cos(tb) * 9, np.sin(tb) * 9, 0, seed + 1); nz[-1] = nz[0]; R_ = (r * (1 + rag * np.interp(a.ravel(), tb, nz).reshape(a.shape))).astype(f32); push = np.where(d > R_, np.sqrt(np.maximum(d * d - R_ * R_, 0)), 0).astype(f32); s = push / d; o = cv2.remap(cloud, (cx + dx * s).astype(f32), (cy + dy * s * asp).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT); m = np.clip((d - R_) / 1.2 + .5, 0, 1); o = o * m[..., None]; al = o[..., 3:4]
    w = r * lip * (1 + .6 * np.interp(a.ravel(), tb, np.roll(nz, 40)).reshape(a.shape)); side = np.clip(.62 - .5 * np.sin(a) , .15, 1.2)
    b1 = np.clip((w * side - (d - R_)) / 1.2 + .5, 0, 1)[..., None] * al; b2 = np.clip((w * .42 * side - (d - R_)) / 1.2 + .5, 0, 1)[..., None] * al; o = o.copy(); o[..., :3] = o[..., :3] * (1 - b1) + C(light[1]) * b1; o[..., :3] = o[..., :3] * (1 - b2) + C(light[0]) * b2; return o
