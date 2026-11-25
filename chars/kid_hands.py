"""Two small hands holding a ball, seen from in front of the child (backs of the hands to us, fingers wrapped on the glass).  Uses db/hands.py.
    L = hold(R, pal, band=('#22366e', '#141f48', '#4a5fa6'), bld=KIDH, s=252)      -> RGBA for a ball of radius R px centred at (960, 486); warp it with the ball (warp(L, cx, cy, k))
    pal = overrides of hands.PAL['day'];  rim = colour of the ball's light on the inner edges; light_up = 0..1"""
import numpy as np, cv2, math
from db import W, H, kit
from db import hands as HD
from db.ink import Cel, spl
from db.paint import rgb, f32

KIDH = HD.build(len=.84, wid=1.16, boxy=.22, tip=.15, knuckle=.78, palm=1.06, wrist=1.0, meat=1.22)
CUP = HD.pose((24, 34, 22, -5), (30, 38, 24, 0), (36, 40, 24, 5), (44, 44, 26, 11), thumb=(46, 18, 20, 10, 10), cup=.30)


def band(g, q, cols, line, w=4.6):
    cel = Cel(W, H, ss=2)
    try:
        R_ = sorted([r for r in g.wrist[0] if -1.15 < r['y'] < -.38], key=lambda r: r['y'])
        if len(R_) >= 2:
            hull = cv2.convexHull(np.vstack([np.asarray(R_[0]['P'], float), np.asarray(R_[-1]['P'], float)]).astype(np.float32))[:, 0].astype(float); c = hull.mean(0); hull = c + (hull - c) * 1.12; P = spl(hull[::max(1, len(hull) // 14)], True, n=5); cel.fill(P, cols[0]); cel.clip(P); cel.fill(P + np.array([q * 46., 24.]), cols[1]); cel.fill(P + np.array([-q * 120., -60.]), cols[2], .55); cel.unclip(); kit.outline(cel, P, w, line, 5)
    except Exception: pass
    return cel.rgba()


def hold(R, pal, cols=('#22366e', '#141f48', '#4a5fa6'), bld=KIDH, s=252, pose=CUP, at=(.985, .80), eul=(14, 8, 46), rim='#ffc36a', light=(.55, .70, .05), w=4.6, res=560):
    o = np.zeros((H, W, 4), f32)
    for q, sd in ((-1, 'R'), (1, 'L')):
        h = HD.Hand(pose, sd, R=HD.euler(-q * eul[0], eul[1], q * eul[2]), pos=(0, 0, 0), bld=bld, pivot=(0, .75, 0), nails='short'); cam = HD.Cam(960 + q * R * at[0], 486 + R * at[1], s, 7.); g = HD.render([h], cam, light=(-q * light[0], light[1], light[2]), res=res, clip=(-60, -60, W + 60, H + 60))
        pl = dict(HD.PAL['day'], **pal); cel = Cel(W, H, ss=2); HD.ink(cel, g, pl, w=w, seed=3)
        if rim:
            try: HD.fill_polys(cel, g, HD.rim_polys(g, (-q * .72, .62, -.10), lo=.30, graze=.75), rim); HD.ink_lines(cel, g, pl, w, 3)
            except Exception: pass
        L = cel.rgba()
        if cols: b = band(g, q, cols, pl['line'], w); L = L * (1 - b[..., 3:4]) + b
        o = o * (1 - L[..., 3:4]) + L
    return o


def warp(L, cx, cy, k, dy=0.):
    return cv2.warpAffine(L, np.float32([[k, 0, cx - 960 * k], [0, k, cy - 486 * k + dy]]), (W, H), flags=cv2.INTER_AREA if k < .99 else cv2.INTER_LINEAR)


def cut(L, cx, cy, k, R, s=252, at=(.985, .80), roll=46):
    """remove the caster's own short forearms from a warped hand layer"""
    yy, xx = np.mgrid[:H, :W].astype(f32); m = np.ones((H, W), f32); sc = s * k
    for q in (-1, 1):
        palm = np.array([cx + q * R * at[0] * k, cy + R * at[1] * k]); d = np.array([-q * math.sin(math.radians(roll)), -math.cos(math.radians(roll))]); t = ((xx - palm[0]) * d[0] + (yy - palm[1]) * d[1]) / sc; side = (xx - cx) * q > 0; m = np.where(side, m * np.clip((t + 1.02) * sc + .5, 0, 1), m)
    return L * m[..., None]


def arms(cx, cy, k, R, s=252, at=(.985, .80), roll=46, skin=('#b3624e', '#e6a06a', '#8a4242'), band=('#22366e', '#151f4a', '#4a5fa6'), line='#2c1014', bend=(1.25, 4.4), w=4.6, sleeve=None):
    """forearm and wristband under each hand, drawn at final size so they leave the frame.  sleeve = (colour, shade) for a cuff higher up"""
    from db.ink import spl as _s
    cel = Cel(W, H, ss=2); sc = s * k; w = max(1.6, w * min(1, k + .2))
    for q in (-1, 1):
        palm = np.array([cx + q * R * at[0] * k, cy + R * at[1] * k]); d = np.array([-q * math.sin(math.radians(roll)), -math.cos(math.radians(roll))]); n = np.array([-d[1], d[0]]); wr = palm - d * .86 * sc; el = wr + np.array([q * bend[0], bend[1]]) * sc; m = wr - d * .55 * sc
        C = spl(np.array([wr + d * .05 * sc, m, m * .55 + el * .45 + np.array([q * .10, 0]) * sc, el]), n=10); T = np.gradient(C, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; wd = sc * np.interp(np.linspace(0, 1, len(C)), [0, .2, .5, 1], [.27, .29, .37, .43]); A, B = C + N * wd[:, None], C - N * wd[:, None]; cel.fill(np.vstack([A, B[::-1]]), skin[0]); inn = A if (N[8] @ np.array([-q, 0])) > 0 else B; ou = B if inn is A else A
        cel.fill(np.vstack([inn, (C * .40 + inn * .60)[::-1]]), skin[1]); cel.fill(np.vstack([ou, (C * .30 + ou * .70)[::-1]]), skin[2]); cel.line(A, w, line, taper=(.02, .02), seed=7); cel.line(B, w, line, taper=(.02, .02), seed=8)
        t0, t1 = wr + d * .02 * sc, wr - d * .50 * sc; hb = .335 * sc; Q = spl(np.array([t0 + n * hb * .93, t0 + d * .05 * sc, t0 - n * hb * .93, t1 - n * hb, t1 - d * .06 * sc, t1 + n * hb]), True, corners=(0, 2, 3, 5), n=6); cel.fill(Q, band[0]); cel.clip(Q); cel.fill(Q + np.array([q * .30, .0]) * sc, band[1]); cel.fill(Q + np.array([-q * .52, -.10]) * sc, band[2], .8); cel.unclip(); kit.outline(cel, Q, w, line, 9)
    return cel.rgba()
