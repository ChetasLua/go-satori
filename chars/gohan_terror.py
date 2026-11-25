"""Gohan's terror for c3 (and anywhere): a slow push-in, one bead of sweat that slides down the cheek, hangs from the chin and lets go.
    lay, info = terror(u, f=0, cx=960, cy=470, s=(300, 372), light='night', yaw=.30)     u 0..1 = time of the push
    info: chin (px), drop = None or (x, y, r, stretch) once it has left (draw it yourself, e.g. db.moon_fx.drop(img, x, y, r, stretch), so it can stop in mid-air), bead 0..1, H, B
    the bead: gathers at the temple 0-.30, slides .30-.62, hangs .62-.74, falls from .74.  Hold u to freeze everything."""
from chars.fkit import *
from chars.gohan import Gohan
from chars import gohan as G

PATH = [(-.66, .20), (-.70, -.05), (-.72, -.30), (-.64, -.56), (-.48, -.78), (-.28, -.93), (-.08, -1.005)]          # near temple -> cheek -> jaw -> chin (front coordinates)
FALL = .74


def bead(cel, q, r, st, L, sc):
    """one bead of sweat: dark line, pale body, hot spot, so it also reads over the pale scarf"""
    T = spl(A([(0, -1.5 * st), (.55, -.45), (.74, .30), (0, 1.), (-.74, .30), (-.55, -.45)]) * r + q, True, corners=(0,), n=7)
    cel.fill(T + [.10 * r, .28 * r], L['line'], .40); cel.fill(T, tone(L, '#b4d2f4', 0), .95); cel.clip(T); cel.fill(ell(q[0] + r * .2, q[1] + r * .35, r * .6, r * .5), mixc(tone(L, '#eaf4ff', 0), '#ffffff', .4), .85); cel.unclip(); cel.fill(ell(q[0] - r * .22, q[1] - r * .05, r * .18, r * .34), '#ffffff'); cel.line(np.vstack([T, T[:1]]), max(1.4, .010 * sc), mixc(L['line'], '#2a2c5a', .25), taper=(.08, .08))


def drop_layer(info, light='night', w=W, h=FH):
    """the fallen bead as a premultiplied layer (use it when the bead crosses the scarf; db.moon_fx.drop for glass against the sky)"""
    cel = Cel(w, h)
    if info['drop']: x, y, r, st = info['drop']; bead(cel, A([x, y]), r, st, globals()['light'](light), info['s'])
    return cel.layer(h, w)


def terror(u, f=0, cx=960., cy=470., s=(300., 372.), light='night', yaw=.30, pitch=.02, roll=-.02, wind=(-.18, .03), hat=False, seed=9, fall_px=2600., bead_=True, h=FH):
    k = u * u * (3 - 2 * u); sc = s[0] + (s[1] - s[0]) * k; shiver = .0; e = dict(G.E['terror'], sweat=0., tears=.30, ir=.50 - .10 * k, under=1., hw=.150, mU=[-.020, .024, .032, .024, -.020], mL=[-.045, -.140 - .03 * k, -.165 - .03 * k, -.140 - .03 * k, -.045]); L = globals()['light'](light) if isinstance(light, str) else light
    lay = Gohan.bust(cx, cy, sc, yaw, pitch, roll, e, light, (f // 2) * 2, wind, seed, hat, 'fear', byaw=yaw * .5, bottom=3.2, h=h); H, B = Gohan.last; sd = -1 if yaw >= 0 else 1
    P = H.Pc(A([(x * (-sd), y) for x, y in PATH]), squeeze=False); C = spl(P, n=12); d = np.r_[0, np.cumsum(np.hypot(*np.diff(C, axis=0).T))]; chin = H.P3(A([(0, -1.005, .62)]))[0] + [0, .012 * sc]; info = dict(chin=tuple(chin), drop=None, bead=float(np.clip(u / .30, 0, 1)), H=H, B=B, s=sc); r0 = .040 * sc
    if u < FALL and bead_:
        v = float(np.clip((u - .30) / .32, 0, 1)) ** 1.35; i = v * d[-1]; q = A([np.interp(i, d, C[:, j]) for j in (0, 1)]); g = float(np.clip(u / .30, .15, 1)); cel = Cel(lay.shape[1], lay.shape[0])
        if v > .02: t = C[(d < i) & (d > i - .9 * sc * v)]; cel.line(t, r0 * .55, mixc(tone(L, '#cfe4ff', 0), '#ffffff', .35), taper=(.9, .1), press=0, alpha=.55) if len(t) > 3 else None
        hang = float(np.clip((u - .62) / (FALL - .62), 0, 1))
        if hang > 0: q = q * (1 - hang) + chin * hang + [0, r0 * (.5 + 1.1 * hang)]
        st = 1 + .9 * hang; bead(cel, q, r0 * (.45 + .55 * g), st, L, sc); lay = lover(lay, cel.layer(*lay.shape[:2]))
    elif bead_: t = u - FALL; info['drop'] = (float(chin[0]), float(chin[1] + r0 * 1.8 + fall_px * t * t), float(r0 * 1.15), float(1. + min(.5, 2.2 * t)))
    return lay, info


_C = {}


def c3_child(f, s0=330., s1=366., at=(800., 470.)):
    """ready for shots/c3.py (Shot.child, frames 28-39): the same face on a slow push-in, held drawing, trembling on twos, the wet track of the bead on his cheek (c3 draws the drop)"""
    g = (f // 2) % 3
    if g not in _C:
        k = (0, 1.2, -1.0)[g]; lay, i = terror(.735, 0, at[0] + k, at[1] - .5 * k, (s1, s1), yaw=.38, pitch=.14, roll=-.05, wind=(.3, 0.), bead_=False, h=FH + 160); H = i['H']; C = spl(H.Pc(A(PATH[2:]), squeeze=False), n=12); cel = Cel(W, lay.shape[0]); cel.line(C, .013 * s1, '#dfe6ff', taper=(.95, .25), press=0, alpha=.55); lay = lover(lay, cel.layer(*lay.shape[:2]) * lay[..., 3:4])
        grey = lay[..., :3].mean(-1, keepdims=True); lay[..., :3] = lay[..., :3] * .55 + grey * .45 * A([.92, .88, 1.08]); _C[g] = (lay.astype(f32), i['chin'])
    k = float(np.clip((f - 28) / 11., 0, 1)); z = (s0 + (s1 - s0) * (k * (2 - k))) / s1; M = np.float32([[z, 0, at[0] * (1 - z)], [0, z, at[1] * (1 - z)]]); cx, cy = _C[g][1]; c3_child.chin = (cx * z + at[0] * (1 - z), cy * z + at[1] * (1 - z)); return np.clip(cv2.warpAffine(_C[g][0], M, (W, FH), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE), 0, 4)


def c3_chin(f): c3_child(f); return c3_child.chin


if __name__ == '__main__':
    ims = []
    for u in (.2, .5, .72, .90):
        bg = np.ones((FH, W, 3), f32) * rgb('#1a1030'); lay, i = terror(u); bg = comp.over(bg, lay)
        bg = comp.over(bg, drop_layer(i))
        ims.append(cv2.resize(comp.finish(bg, 0, bloom=.28, vig=.2, soft=.05), (960, 540), interpolation=cv2.INTER_AREA))
    comp.save('qa/friends/terror.jpg', np.vstack([np.hstack(ims[:2]), np.hstack(ims[2:])]))
