"""KID GOKU, any angle, any pose: a 3-D build re-drawn as an inked cel (chars/kid_head.py = face and hair).

    K = Kid()
    L = K.draw(cx, cy, S, pose='sit', yaw=.5, pitch=0, roll=0, expr='smile', pal='dawn', light=(.6, .6, -.2), f=f)     # premultiplied (H, W, 4)
    K.info -> dict(head=(x, y, s), handL, handR, footL, footR, pelvis, chest, neck: screen px, box)

cx, cy = screen px of the PELVIS, S = px per unit (head half width = 1; standing height about 7 units, sitting about 4.6).  D = camera distance in units (9 calm, 4 strong, 2.6 huge).
yaw > 0 turns the whole boy to the viewer's right, pitch > 0 tips him back / looks up, roll > 0 clockwise.   R = own 3x3 instead.   x: + is the viewer's right when he faces us = HIS left.
pose: name in POSES, or a dict (P('sit', armR=...)).   mixp(a, b, t) in-between.  expr: name in kid_head.EXPR or a dict.
wind = (wx, wy, amount) screen direction, blows hair, belt ends, sleeves.  tail = list of 3-D points (root frame) or None; tail_curve(f, ...) / Chain give follow-through.
hold = layer_fn(cel_xy) not needed: draw your ball at K.info['handR'] between K.draw(..., parts='back') and parts='front' (see Kid.split).
parts: 'all' | 'body' (no head) | 'head'.   ink only silhouettes: pal='black'."""
import numpy as np, cv2, math
from db import W, H
from db.cam import Camera3D
from db.fig_sdf import Clay, table, NT, NP, wrap
from db.ink import Cel, spl, resample, paste, noise1
from db.kit import outline, rim as kit_rim
from db.paint import rgb, f32
from db import letters as LT
from . import kid_head as KH
from .kid_head import KHead, PAL, EXPR, mixx
from .kid_ink import mask_polys, clean_closed, clean_index, draw_contour, _hash, _mixc, _true_runs, _smooth

pi = math.pi
BLACK = {k: '#17131a' for k in PAL['day']}
PAL['black'] = BLACK


def E(a):
    """(yaw, pitch, roll) radians -> matrix in figure axes (x right, y up, z to us)"""
    y, p, r = a; Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]]); Rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]]); c, n = math.cos(-r), math.sin(-r); return np.array([[c, -n, 0], [n, c, 0], [0, 0, 1]]) @ Rx @ Ry


def unit(v): v = np.asarray(v, float); return v / (np.linalg.norm(v) + 1e-12)
def w3(p): p = np.asarray(p, float); return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


def ik(a, c, l1, l2, pole):
    d = c - a; L = np.linalg.norm(d); m = (l1 + l2) * .995
    if L > m: c = a + d / L * m; d = c - a; L = m
    u = d / (L + 1e-9); x = (L * L + l1 * l1 - l2 * l2) / (2 * L + 1e-9); h = math.sqrt(max(l1 * l1 - x * x, 1e-4)); n = np.asarray(pole, float); n = n - u * (n @ u); n = unit(n) if np.linalg.norm(n) > 1e-4 else unit(np.cross(u, [.3, .5, .7])); return a + u * x + n * h, c


# -------------------------------------------------------------------------------------------- poses
# pel wst cst hed: (yaw, pitch, roll) of each link relative to the one below; pitch < 0 leans forward
# arm: wrist from the shoulder in CHEST axes (out, up, fwd) in arm lengths; elb: elbow pole (out, up, fwd); palm = (palm normal out, up, fwd); curl 0 open .. 1 fist
# leg: ankle from the pelvis in ROOT axes (out, up, fwd) units; kne: knee pole; toe: toe direction (out, up, fwd); sole normal is found from shin
BASE = dict(bag=1., stL=1., stR=1., pel=(0, 0, 0), wst=(0, 0, 0), cst=(0, 0, 0), hed=(0, 0, 0), clL=(0, 0, 0), clR=(0, 0, 0),
            armL=(.22, -.90, .10), armR=(.22, -.90, .10), elbL=(.4, -.1, -1), elbR=(.4, -.1, -1), palmL=(-1, 0, .2), palmR=(-1, 0, .2), curlL=.35, curlR=.35, sprL=.15, sprR=.15,
            legL=(.36, -2.14, 0), legR=(.36, -2.14, 0), kneL=(.15, 0, 1), kneR=(.15, 0, 1), toeL=(.25, 0, 1), toeR=(.25, 0, 1), root=(0, 0, 0), tail=dict(dir=(.25, -.15, -1), curl=.9, side=.3))


def P(base=None, **k):
    d = dict(BASE if base is None else (POSES[base] if isinstance(base, str) else base)); d.update(k); return d


POSES = {}
POSES['stand'] = P()
POSES['sit'] = P(pel=(0, -.06, 0), wst=(0, -.10, 0), cst=(0, -.02, 0), hed=(0, .10, 0), armL=(.10, -.62, .56), armR=(.10, -.62, .56), elbL=(1, -.2, -.3), elbR=(1, -.2, -.3), palmL=(-.2, -1, .2), palmR=(-.2, -1, .2), curlL=.5, curlR=.5,
                 legL=(-.16, -.46, 1.08), legR=(-.06, -.60, .92), kneL=(1, .16, .10), kneR=(1, .16, .10), toeL=(-1, -.10, .30), toeR=(-1, -.10, .22), bag=.80, tail=dict(dir=(.45, .05, -1), curl=1.0, side=.5))
POSES['sit_ball'] = P('sit', armR=(.16, .62, .62), elbR=(1, -.6, .1), palmR=(0, .2, -1), curlR=.45, sprR=.3, cst=(-.10, .06, .06), hed=(-.10, .26, 0), armL=(.16, -.66, .46))
POSES['sit_hold'] = P('sit', armL=(-.20, -.18, .72), armR=(-.20, -.18, .72), elbL=(1, -.8, 0), elbR=(1, -.8, 0), palmL=(-1, .35, 0), palmR=(-1, .35, 0), curlL=.45, curlR=.45)
POSES['lean'] = P('sit', pel=(0, -.30, 0), wst=(.10, -.30, .05), cst=(.10, -.22, .08), hed=(.1, .10, 0), armL=(.34, -.70, .52), armR=(.02, -.72, .50), curlL=.1, curlR=.4, elbL=(1, .2, -.8), palmL=(0, -1, 0))
POSES['dive'] = P(pel=(0, .10, 0), wst=(0, .16, 0), cst=(0, .14, 0), hed=(0, .95, 0), armL=(.50, .46, .60), armR=(.42, .58, .58), elbL=(1, .1, -.2), elbR=(1, .1, -.2), palmL=(0, -.3, 1), palmR=(0, -.3, 1), curlL=.55, curlR=.55,
                  legL=(.46, -2.10, -.35), legR=(.62, -1.70, -.85), kneL=(.7, .1, .9), kneR=(.8, .1, .9), toeL=(.2, -1, -.3), toeR=(.3, -.8, -.6), tail=dict(dir=(.1, -.6, -1), curl=.5, side=.5))
POSES['lie'] = P('dive', pel=(0, .02, 0), wst=(0, .22, 0), cst=(0, .22, 0), hed=(0, .72, 0), armL=(.30, .10, .62), armR=(.30, .05, .64), elbL=(1, -.4, .2), elbR=(1, -.4, .2), palmL=(0, 0, 1), palmR=(0, 0, 1), curlL=.35, curlR=.35, legL=(.40, -2.30, -.12), legR=(.46, -1.55, -1.05), kneL=(.4, 0, 1), kneR=(.5, 0, 1))
POSES['skim'] = P('lie', wst=(.10, .20, .20), cst=(.16, .18, .26), hed=(-.30, .62, -.10), armR=(.66, -.30, .70), elbR=(.5, .6, -.3), palmR=(-.2, -.2, 1), curlR=.12, sprR=.4, armL=(.20, .22, .60), curlL=.6, legL=(.50, -2.18, -.30), legR=(.52, -1.60, -1.0))
POSES['punch'] = P(pel=(0, .06, 0), wst=(-.10, .12, -.06), cst=(-.16, .14, -.08), hed=(.10, .42, 0), armR=(.10, .98, .12), elbR=(1, 0, .3), palmR=(0, 0, 1), curlR=1., armL=(.42, -.42, .30), elbL=(.8, -.4, -1), palmL=(-1, 0, 0), curlL=1.,
                   legL=(.42, -1.36, .62), legR=(.34, -2.34, -.34), kneL=(.5, .2, 1), kneR=(.2, 0, 1), toeL=(.2, -.9, .5), toeR=(.1, -1, -.1), tail=dict(dir=(.1, -1, -.5), curl=.4, side=.4))
POSES['whoop'] = P(pel=(0, .08, 0), wst=(0, .12, 0), cst=(0, .16, 0), hed=(0, .30, .06), armL=(.50, .80, .10), armR=(.46, .84, .06), elbL=(1, -.3, .3), elbR=(1, -.3, .3), palmL=(0, 0, 1), palmR=(0, 0, 1), curlL=1., curlR=1., legL=(.52, -1.72, .50), legR=(.48, -2.0, -.50), kneL=(.6, .2, 1), kneR=(.5, 0, 1), toeL=(.2, -.8, .5), toeR=(.1, -1, -.2), tail=dict(dir=(.3, .2, -1), curl=1.4, side=.6))


def mixp(a, b, t):
    a = POSES[a] if isinstance(a, str) else a; b = POSES[b] if isinstance(b, str) else b; o = {}
    for k, v in a.items():
        q = b.get(k, v)
        if isinstance(v, dict): o[k] = {j: (tuple(np.array(v.get(j, q.get(j)), float) * (1 - t) + np.array(q.get(j, v.get(j)), float) * t) if np.ndim(v.get(j, q.get(j))) else v.get(j, q.get(j)) * (1 - t) + q.get(j, v.get(j)) * t) for j in {**v, **q}}
        elif np.ndim(v): o[k] = tuple(np.array(v, float) * (1 - t) + np.array(q, float) * t)
        else: o[k] = v * (1 - t) + q * t
    return o


LU, LF = .82, .78           # arm
TH, SN = 1.02, .98         # leg


class Skel:
    def __init__(s, p, R, S0=np.zeros(3)):
        s.p = p; s.Rr = R; s.pel = np.asarray(S0, float) + R @ np.asarray(p['root'], float); s.Rp = R @ E(p['pel']); s.Rw = s.Rp @ E(p['wst']); s.Rc = s.Rw @ E(p['cst']); s.Rh = s.Rc @ E(p['hed'])
        s.waist = s.pel + s.Rp @ [0, .40, 0]; s.chest = s.waist + s.Rw @ [0, .52, .02]; s.neck = s.chest + s.Rc @ [0, .70, -.02]; s.head = s.neck + s.Rc @ [0, .18, 0] + s.Rh @ [0, .80, .14]
        s.sh, s.el, s.wr, s.hip, s.kn, s.an, s.Rhand, s.toe, s.shin = {}, {}, {}, {}, {}, {}, {}, {}, {}
        for q, m in (('L', 1.), ('R', -1.)):
            M = np.array([m, 1, 1]); s.sh[q] = s.chest + s.Rc @ (np.array([.60, .46, -.02]) * M + np.asarray(p['cl' + q], float) * M)
            st = p.get('st' + q, 1.); tgt = s.sh[q] + s.Rc @ (np.asarray(p['arm' + q], float) * M) * (LU + LF) * st; s.el[q], s.wr[q] = ik(s.sh[q], tgt, LU * st, LF * st, s.Rc @ (np.asarray(p['elb' + q], float) * M))
            y = unit(s.wr[q] - s.el[q]); z = s.Rc @ (np.asarray(p['palm' + q], float) * M); z = z - y * (z @ y); z = unit(z) if np.linalg.norm(z) > 1e-3 else unit(np.cross(y, [.2, .9, .3])); s.Rhand[q] = np.stack([np.cross(y, z), y, z], -1)
            s.hip[q] = s.pel + s.Rp @ (np.array([.32, -.18, 0]) * M); tgt = s.pel + R @ (np.asarray(p['leg' + q], float) * M); s.kn[q], s.an[q] = ik(s.hip[q], tgt, TH, SN, R @ (np.asarray(p['kne' + q], float) * M)); s.toe[q] = unit(R @ (np.asarray(p['toe' + q], float) * M))


def _clay(sk, fine=True):
    c = Clay(); G = lambda n, *a: c.group(n, *a); z = lambda M, i=2: w3(M[:, i])
    s = sk; W_ = w3
    G('body'); ids = {}
    ids['pelvis'] = c.loft(W_(s.pel - s.Rp[:, 1] * .42), W_(s.pel + s.Rp[:, 1] * .50), z(s.Rp), table([(0, .30, .34, .32, .34), (.25, .42, .50, .44, .50), (.6, .44, .53, .44, .53), (1, .40, .48, .40, .48)], 2.2), 'body', 'pelvis', 'gi', cap=(1., 1.), k=.10)
    ids['chest'] = c.loft(W_(s.waist - s.Rw[:, 1] * .05), W_(s.neck + s.Rc[:, 1] * .02), z(s.Rc), table([(0, .41, .48, .40, .48), (.30, .46, .55, .45, .55), (.60, .47, .62, .47, .62), (.80, .42, .60, .45, .60), (.93, .32, .42, .36, .42), (1, .26, .32, .30, .32)], 2.3), 'body', 'torso', 'gi', cap=(1., .5), k=.12)
    for q, m in (('L', 1.), ('R', -1.)): c.ball(W_(s.sh[q] - s.Rc[:, 0] * m * .10 + s.Rc[:, 1] * .0), (.30, .28, .30), 'body', 'torso', 'gi', k=.12)
    cphi = np.arange(NP) * 2 * pi / NP; C = 1 / np.sqrt((np.cos(cphi) / .425) ** 2 + (np.sin(cphi) / .505) ** 2)
    c.ring(W_(s.pel + s.Rp[:, 1] * .44), z(s.Rp, 1), z(s.Rp), C, .085, 0., .52, G and 'belt', 'belt', 'blue', k=.0)
    G('neck'); c.loft(W_(s.neck - s.Rc[:, 1] * .20), W_(s.neck + (s.head - s.neck) * .42), z(s.Rc), table([(0, .34, .36, .34, .36), (.4, .27, .27, .28, .27), (1, .25, .25, .27, .25)]), 'neck', 'neck', 'skin', cap=(0., 1.))
    for q, m in (('L', 1.), ('R', -1.)):
        g = 'arm' + q; G(g); ax = unit(s.el[q] - s.sh[q]); fr = s.Rc[:, 2]
        c.loft(W_(s.sh[q]), W_(s.el[q]), W_(fr), table([(0, .20, .20, .20, .20), (.5, .20, .185, .19, .18), (1, .165, .165, .165, .165)]), g, 'arm' + q, 'skin', k=.08)
        ids['fore' + q] = c.loft(W_(s.el[q]), W_(s.wr[q]), W_(fr), table([(0, .165, .165, .165, .165), (.30, .19, .185, .185, .185), (1, .125, .135, .125, .135)]), g, 'arm' + q, 'skin', k=.08)
        u = unit(s.wr[q] - s.el[q]); c.loft(W_(s.wr[q] - u * .31), W_(s.wr[q] - u * .03), W_(fr), table([(0, .175, .18, .175, .18), (.12, .198, .20, .198, .20), (.5, .185, .19, .185, .19), (.9, .192, .197, .192, .197), (1, .17, .175, .17, .175)]), 'band' + q if G('band' + q) >= 0 else 0, 'band' + q, 'blue', cap=(0., 0.))
        sl = 'sleeve' + q; G(sl); ids[sl] = c.loft(W_(s.sh[q] - ax * .16), W_(s.sh[q] + ax * .46), W_(fr), table([(0, .27, .27, .27, .27), (.4, .285, .285, .28, .285), (.8, .29, .30, .28, .28), (1, .275, .295, .265, .27)]), sl, sl, 'blue', cap=(1., 0.))
        hp = 'hand' + q; G(hp); Rq = s.Rhand[q]; o = s.wr[q] + Rq[:, 1] * .02; tf = lambda v: W_(o + Rq @ (np.asarray(v, float) * [-m, 1, 1])); cu = s.p['curl' + q]; sp = s.p['spr' + q]
        c.ball(tf((0, .20, .0)), (.175, .20, .105), hp, hp, 'skin', ax=W_(Rq[:, 1]), e1=W_(Rq[:, 0]), k=.06)
        for j, (x, ln) in enumerate(((.135, .92), (.047, 1.), (-.045, .95), (-.130, .80))):
            a0 = (x * .95, .34, .005); fa = (-x * sp * 2.2); a1 = 1.45 * cu; a2 = 2.75 * cu; d1 = np.array([math.sin(-fa), math.cos(a1), math.sin(a1)]); d1 /= np.linalg.norm(d1); d2 = np.array([math.sin(-fa) * .6, math.cos(a2), math.sin(a2)]); d2 /= np.linalg.norm(d2); p1 = np.array(a0) + d1 * .17 * ln; p2 = p1 + d2 * .15 * ln; r = .064
            tb = table([(0, r, r, r, r), (1, r * .95, r * .95, r * .95, r * .95)]); hj = hp + str(j); c.loft(tf(a0), tf(p1), W_(Rq[:, 2]), tb, hp, hj, 'skin', k=.03); c.loft(tf(p1), tf(p2), W_(Rq[:, 2]), tb * .93, hp, hj, 'skin', k=.03)
        t0 = np.array((.12, .10, .03)); t1 = t0 + unit((.8 - .5 * cu, .75, .35 + .5 * cu)) * .17; t2 = t1 + unit((.25 - .9 * cu, .9 - .2 * cu, .3 + .4 * cu)) * .14; tb = table([(0, .075, .075, .075, .075), (1, .066, .066, .066, .066)]); c.loft(tf(t0), tf(t1), W_(Rq[:, 2]), tb, hp, hp + 't', 'skin', k=.05); c.loft(tf(t1), tf(t2), W_(Rq[:, 2]), tb * .9, hp, hp + 't', 'skin', k=.03)
        # leg
        g = 'leg' + q; G(g, W_(s.kn[q]), .5, .0); kf = s.kn[q] - (s.hip[q] + s.an[q]) / 2; kf = unit(kf) if np.linalg.norm(kf) > .05 else s.Rr[:, 2]
        ids['thigh' + q] = c.loft(W_(s.hip[q]), W_(s.kn[q]), W_(kf), table([(0, .40, .40, .40, .34), (.4, .41, .42, .41, .37), (.8, .385, .40, .40, .37), (1, .35, .37, .37, .35)], 2.1), g, 'thigh' + q, 'gi', k=.10)
        u = unit(s.an[q] - s.kn[q]); b = s.an[q] - u * .16; bg = s.p.get('bag', 1.)
        ids['shin' + q] = c.loft(W_(s.kn[q]), W_(b), W_(kf), table([(0, .34, .35, .36, .35), (.30, .36 * bg, .39 * bg, .42 * bg, .39 * bg), (.62, .36 * bg, .40 * bg, .42 * bg, .39 * bg), (.86, .33 * bg, .36 * bg, .37 * bg, .36 * bg), (.97, .24, .25, .26, .25), (1, .20, .21, .22, .21)], 2.1), g, 'thigh' + q, 'gi', cap=(1., 0.), k=.10)
        G('foot' + q); fq = 'foot' + q; td = s.toe[q]; up = -u - td * (-u @ td); up = unit(up) if np.linalg.norm(up) > .1 else unit(np.cross(td, [1, 0, 0.1])); sx = np.cross(up, td) * m
        c.loft(W_(s.an[q] - u * .40), W_(s.an[q] + u * .02), W_(td), table([(0, .15, .15, .15, .15), (1, .14, .145, .15, .145)]), fq, fq, 'skin', k=.08)
        fo = s.an[q]; ft = lambda v: W_(fo + sx * v[0] + up * v[1] + td * v[2])
        c.loft(ft((0, -.13, -.17)), ft((.02, -.17, .40)), W_(up), table([(0, .13, .13, .10, .13), (.35, .16, .155, .10, .14), (.75, .11, .20, .085, .18), (1, .085, .20, .08, .17)]), fq, fq, 'skin', k=.10)
        for j in range(5): x = -.145 + j * .075; c.ball(ft((x + .01, -.195, .46 - abs(j - .6) * .035)), (.052, .075, .05) if j else (.072, .10, .062), fq, fq, 'skin', ax=W_(td), e1=W_(sx), k=.03)
    return c.pack(), ids


def lw(px): return float(np.interp(px, [60, 140, 260, 450, 900, 1600], [1.0, 1.5, 2.1, 2.9, 4.2, 5.6]))


class Kid:
    MAT = dict(gi=('gi_hi', 'gi', 'gi_sh', 'gi_dk'), blue=('blue_hi', 'blue', 'blue_sh', 'blue_dk'), skin=('skin', 'skin', 'skin_sh', 'skin_dk'))

    def __init__(s): s.info = {}; s.H = KHead()

    def build(s, cx, cy, S, pose='stand', yaw=0., pitch=0., roll=0., R=None, D=9., persp=None):
        s.pose = POSES[pose] if isinstance(pose, str) else pose; s.R = E((yaw, pitch, roll)) if R is None else np.asarray(R, float); s.S = S; s.D = D
        s.sk = Skel(s.pose, s.R); s.cam = Camera3D((0, -D, 0), (0, 0, 0), S * D, W, H, cx=cx, cy=cy); s.px = S * 5.2
        pr = lambda p: s.cam.project(w3(np.asarray(p, float))); k = s.sk
        hq, hz = pr(k.head); s.hk = D / max(float(hz), .4); hp = {q: pr(k.wr[q] + k.Rhand[q][:, 1] * .28)[0] for q in 'LR'}
        s.info = dict(head=(float(hq[0]), float(hq[1]), S * s.hk), headz=float(hz), handL=hp['L'], handR=hp['R'], wristL=pr(k.wr['L'])[0], wristR=pr(k.wr['R'])[0], footL=pr(k.an['L'])[0], footR=pr(k.an['R'])[0], pelvis=pr(k.pel)[0], chest=pr(k.chest)[0], neck=pr(k.neck)[0], kneeL=pr(k.kn['L'])[0], kneeR=pr(k.kn['R'])[0],
                      hz={q: float(pr(k.wr[q])[1]) for q in 'LR'}, tail=pr(k.pel + k.Rp @ [0, -.08, -.46])[0])
        return s

    def draw(s, cx, cy, S, pose='stand', yaw=0., pitch=0., roll=0., expr='smile', pal='dawn', light=(.6, .6, .5), wind=(0, 0, 0), f=0, tail='auto', parts='all', R=None, D=9., key=.06, seed=0, weight=1., rim=None, head=None, knot=True, under=None, between=None):
        """under(cel): hook drawn first.  between(layer)->layer: called after body, before the near arms go back over the head"""
        s.build(cx, cy, S, pose, yaw, pitch, roll, R, D); pal = PAL[pal] if isinstance(pal, str) else pal; s.pal = pal; s.light = np.asarray(light, float) / np.linalg.norm(light); s.wind = wind; s.f = f; s.seed = seed
        c, s.ids = _clay(s.sk); s.c = c; s.wo = lw(s.px) * weight; s.wf = s.wo * .58
        x0, y0, x1, y1 = c.bbox(s.cam, 12); hx, hy, hs = s.info['head']; m = hs * 3.5 + 10
        tl = s._tail(tail); pts = [p for p in (tl[0] if tl is not None else [])] if tl is not None else []
        if len(pts): P_ = np.array(pts); r = tl[1].max() + 6; x0, y0, x1, y1 = min(x0, P_[:, 0].min() - r), min(y0, P_[:, 1].min() - r), max(x1, P_[:, 0].max() + r), max(y1, P_[:, 1].max() + r)
        x0, y0, x1, y1 = int(max(-160, min(x0, hx - m))), int(max(-160, min(y0, hy - m))), int(min(W + 160, max(x1, hx + m))), int(min(H + 160, max(y1, hy + m)))
        if x1 - x0 < 4 or y1 - y0 < 4: return np.zeros((H, W, 4), f32)
        s.box = (x0, y0, x1, y1); s.info['box'] = s.box
        s.buffers()
        mk = lambda: Cel(x1 - x0, y1 - y0, x0, y0, ss=2)
        img = np.zeros((y1 - y0, x1 - x0, 4), f32)
        def put(L): nonlocal img; img = img * (1 - L[..., 3:4]) + L
        if parts in ('all', 'body'):
            behind = tl is not None and tl[2] < 0
            cel = mk()
            if under is not None: under(cel)
            if tl is not None and behind: s.draw_tail(cel, tl)
            hs0 = None
            if parts == 'all' and head is not False:
                s.H.pose(hx, hy, hs, R=s.sk.Rh); nz = float(s.cam.project(w3(s.sk.neck))[1]); s.zcut = (s.info['headz'] - nz) - .10; KH.draw_head(cel, s.H, expr, pal, tuple(s.light), wind, f, neck=False, parts='k', zcut=s.zcut)
            s.paint(cel); s.deco(cel); s.lines(cel)
            if knot: s.knot(cel)
            if tl is not None and not behind: s.draw_tail(cel, tl)
            body = cel.rgba(); put(body)
            if between is not None: img = between(img, s)
        if parts in ('all', 'head') and head is not False:
            hc = mk(); s.H.pose(hx, hy, hs, R=s.sk.Rh); ml = s.light
            lc = (ml[0], ml[1], ml[2]); KH.draw_head(hc, s.H, expr, pal, lc, wind, f, neck=False, zcut=getattr(s, 'zcut', None) if parts == 'all' else None, neckpx=s.info['neck'], w=lw(s.px) * 1.02 * weight if s.px < 900 else None)
            hd = hc.rgba()
            put(hd)
            if parts == 'all':                                             # arms and hands nearer than the face go back on top
                near = np.isin(s.lab, [i for i, n in enumerate(s.lname) if n.startswith(('hand', 'arm', 'band', 'sleeve', 'foot', 'thigh'))]) & (s.Z < s.info['headz'] - .75) & s.hit
                if near.any():
                    F = cv2.resize(near.astype(f32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR); F = np.clip((F - .5) * 6 + .5, 0, 1); r = max(1, int(round(s.wo * .6))); F = cv2.dilate(F, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))[..., None]
                    put(body * F)
        if rim: img = kit_rim(img, rim.get('dx', 2), rim.get('dy', 2), rim.get('col', pal['rim']), rim.get('amt', .9), rim.get('soft', .5))
        out = np.zeros((H, W, 4), f32); paste(out, img, x0, y0); return out

    # ------------------------------------------------ buffers
    def buffers(s):
        x0, y0, x1, y1 = s.box; big = max(x1 - x0, y1 - y0); s.g = g = min(1., 640. / big) if s.px > 260 else (1.3 if s.px > 130 else 2.0)
        Z, Wp, N, T, A, C = s.c.render(s.cam, s.box, g); s.Z, s.W, s.N, s.C = Z, Wp, N, C; s.hit = Wp >= 0
        names = sorted(set(s.c.lab)); s.lname = names; lid = {n: i for i, n in enumerate(names)}; pl = np.array([lid[l] for l in s.c.lab]); s.lab = np.where(s.hit, pl[np.maximum(Wp, 0)], -1).astype(np.int16); s.mat = {}
        for l, m in zip(s.c.lab, s.c.mat): s.mat[lid[l]] = m
        c2w = lambda v: np.array([v[0], -v[2], v[1]], float) / np.linalg.norm(v); s.L = c2w(s.light)
        sg = max(.8, s.px * g * .008); m = s.hit.astype(f32); w = cv2.GaussianBlur(m, (0, 0), sg) + 1e-4; n = cv2.GaussianBlur(N, (0, 0), sg) / w[..., None]; dz = np.abs(cv2.GaussianBlur(Z * m, (0, 0), sg) / w - Z); k = np.clip(1 - dz / .25, 0, 1)[..., None]; n = (n * .8 + N * .2) * k + N * (1 - k); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
        s.n = n; nd = n @ s.L.astype(f32); sh = s.c.shadow(s.cam, s.box, g, (Z, Wp, N), s.L, 2.2, 2) if s.hit.any() else np.zeros_like(Z)
        # the head throws a shadow on the chest and shoulders
        hx, hy, hs = s.info['head']; o = s.sk.head; yy, xx = np.nonzero(s.hit); S = ((nd < .06) | (sh > 0)) & s.hit
        if len(yy):
            X = ((xx + .5) / g + x0 - s.cam.cx) / s.cam.f; Y = ((yy + .5) / g + y0 - s.cam.cy) / s.cam.f; zc = Z[yy, xx]; Q = np.stack([X * zc, zc - s.D, -Y * zc], -1); d = w3(o) - Q; t = d @ s.L; dist = np.linalg.norm(d - t[:, None] * s.L, axis=1); S[yy, xx] |= (t > .2) & (dist < .98) & (nd[yy, xx] > -.2)
        ks = int(max(3, s.px * g * .014)) | 1; s.Sh = (cv2.medianBlur(S.astype(np.uint8), min(ks, 9)) > 0) & s.hit; s.Dp = (C < .30) & s.hit

    def paint(s, cel):
        g, pal = s.g, s.pal; a = max(5., (s.px * g * .018) ** 2)
        Sp = [draw_contour(q, max(.7, s.px * .0040), True, .55, n=3) for q in mask_polys(s.Sh, g, s.box, raw=True, min_area=a, smooth=max(3, int(s.px * g * .010)))]
        Dp = [draw_contour(q, max(.6, s.px * .0034), True, .5, n=3) for q in mask_polys(cv2.medianBlur(s.Dp.astype(np.uint8), 3), g, s.box, raw=True, min_area=a * .7, smooth=5)]
        s.Sp = Sp; s.lpoly = {}; ids = [i for i in range(len(s.lname)) if (s.lab == i).any()]; zm = {i: float(np.median(s.Z[s.lab == i])) for i in ids}; ids.sort(key=lambda i: -zm[i]); sm = max(3, int(3.2 * g + 1.5)); ks = np.ones((7, 7), np.uint8); s.zm = zm
        for i in ids:
            nm = s.lname[i]; m = s.lab == i; raw = [c for c in mask_polys(m, g, s.box, raw=True, up=2, min_area=3., smooth=sm, sig=.6) if len(c) >= 10]; e = max(.45, s.px * (.0012 if nm[:4] in ('hand', 'foot', 'neck') else .0020)); cl = [clean_closed(c, e) for c in raw]; s.lpoly[i] = (raw, cl)
            if not cl: continue
            polys = [c[0] for c in cl]; near = np.isin(s.lab, [q for q in ids if zm[q] < zm[i]]); grow = (cv2.dilate(m.astype(np.uint8), ks) > 0) & (near | m); und = mask_polys(grow, g, s.box, eps=max(.45, s.px * .0015), min_area=2., smooth=sm) if grow.sum() > m.sum() else []
            hi, base, shd, deep = [pal[k] for k in s.MAT[s.mat[i]]]
            for ps in (und, polys):
                if ps: cel._path(ps); cel._src(base, 1.); cel.ctx.set_fill_rule(1); cel.ctx.fill()
            cel.ctx.save(); cel._path(polys + und); cel.ctx.set_fill_rule(0); cel.ctx.clip()
            if Sp: cel.fill(Sp, shd, evenodd=True)
            if Dp and s.px > 120: cel.fill(Dp, _mixc(shd, deep, .55), evenodd=True)
            cel.ctx.restore()

    # ------------------------------------------------ surface drawing
    def vis(s, Q, labels, tol=.10):
        p, z = s.cam.project(Q); g = s.g; h, w = s.Z.shape; u = np.floor((p[:, 0] - s.box[0]) * g).astype(int); v = np.floor((p[:, 1] - s.box[1]) * g).astype(int); ok = (u >= 0) & (v >= 0) & (u < w) & (v < h); o = np.zeros(len(p), bool); i = np.nonzero(ok)[0]; L = [k for k, n in enumerate(s.lname) if n in labels]
        if len(i): o[i] = (np.abs(s.Z[v[i], u[i]] - z[i]) < tol) & np.isin(s.lab[v[i], u[i]], L)
        return p, o

    def clip_to(s, cel, labels):
        polys = [c[0] for i, (raw, cl) in s.lpoly.items() if s.lname[i] in labels for c in cl]
        cel.ctx.save(); cel._path(polys if polys else [np.zeros((3, 2))]); cel.ctx.set_fill_rule(0); cel.ctx.clip()

    def deco(s, cel):
        pal, c, k = s.pal, s.c, s.sk; ch = s.ids['chest']; view = np.array([0, 0, 1.]); fz = float(k.Rc[2, 2]); L_ = float(np.linalg.norm(k.neck - k.waist)) + .07
        pt = lambda i, L, lift=.0: c.point(i, np.array([a for a, _ in L]), np.array([b for _, b in L]), lift); pr = lambda Q: s.cam.project(Q)[0]
        def region(L, col, colsh, lab=('torso',), closed=True, n=6, corners=()):
            Pq = pr(pt(ch, [tuple(v) for v in spl(np.array(L, float), closed, corners=corners, n=n)]))
            s.clip_to(cel, lab); cel.fill(Pq, col)
            if s.Sp: cel.clip(Pq); cel.fill(s.Sp, colsh, evenodd=True); cel.unclip()
            cel.ctx.restore(); return Pq
        def strk(i, L, w, col, lab, n=8, taper=(.12, .45), press=.25):
            V = spl(np.array(L, float), n=n); p, o = s.vis(pt(i, [tuple(v) for v in V], .004), lab)
            for a, b in _true_runs(o, 3): cel.line(p[a:b], w, col, taper=taper, press=press, seed=_hash(s.seed, i, L[0]) & 0xffff, tmin=.15)
        ln = pal['line']; wf = s.wf
        if fz > -.15:
            V = [(1.06, -.95), (.94, -.62), (.80, -.34), (.60, .04), (.80, .40), (.94, .66), (1.06, .95)]
            region(V, pal['blue'], pal['blue_sh'], corners=(0, 3, 6)); region([(1.08, -.62), (.97, -.36), (.915, 0.), (.97, .36), (1.08, .62)], pal['skin'], pal['skin_sh'], corners=(0, 4))
            if s.px > 90:
                strk(ch, [(1.02, -.86), (.94, -.62), (.80, -.34), (.60, .04), (.40, .20), (.14, .30)], wf * 1.25, ln, ('torso',), taper=(.1, .2)); strk(ch, [(1.02, .86), (.94, .66), (.80, .40), (.62, .07)], wf * 1.25, ln, ('torso',), taper=(.1, .15)); strk(ch, [(1.03, -.52), (.97, -.34), (.915, 0), (.97, .34), (1.03, .52)], wf * .9, ln, ('torso',))
                if s.px > 200:
                    strk(ch, [(.99, -.98), (.90, -.74), (.76, -.46), (.56, -.10)], wf * .6, pal['gi_dk'], ('torso',)); strk(ch, [(.99, .98), (.90, .78), (.76, .52), (.60, .22)], wf * .6, pal['gi_dk'], ('torso',))
        rr = .165
        if s.px > 70:
            Td = lambda tc, pc, r, rad: (lambda U: pr(c.point(ch, tc - U[:, 1] * r / L_, pc + U[:, 0] * r / rad, .004)))
            def face(phi): n = np.cos(phi) * k.Rc[:, 2] + np.sin(phi) * k.Rc[:, 0]; return n[2]
            if face(.72) > .12:
                s.clip_to(cel, ('torso',)); o = LT.disc(cel, '亀', Td(.46, .74, rr, .50), white=pal['white'], ink=pal['line'], w=max(.7, wf * .9), seed=3)
                if s.Sp: cel.clip(o); cel.fill(s.Sp, pal['white_sh'], .55, evenodd=True); cel.unclip()
                cel.ctx.restore()
            if face(pi) > .12:
                s.clip_to(cel, ('torso',)); o = LT.disc(cel, '亀', Td(.56, pi, .30, -.50 if False else .50), white=pal['white'], ink=pal['line'], w=max(.8, wf), seed=4)
                if s.Sp: cel.clip(o); cel.fill(s.Sp, pal['white_sh'], .55, evenodd=True); cel.unclip()
                cel.ctx.restore()
        if s.px > 120:                                                      # folds, from their tension points
            gd = pal['gi_dk'] if s.px > 300 else pal['line']
            for q, m in (('L', 1.), ('R', -1.)):
                th, sn, lab = s.ids['thigh' + q], s.ids['shin' + q], ('thigh' + q,)
                for L, w in (([(.10, .5), (.26, .2), (.40, .1)], 1.), ([(.06, -.6), (.22, -.4), (.36, -.25)], .9), ([(.80, 1.5), (.90, .95), (.95, .4)], 1.), ([(.80, -1.5), (.90, -.95), (.96, -.4)], .9)): strk(th, L, wf * w, ln, lab)
                for L, w in (([(.05, 2.4), (.14, 1.7), (.18, 1.0)], .9), ([(.06, -2.4), (.15, -1.6), (.2, -.9)], .9), ([(.96, -.7), (.84, -.55), (.70, -.6)], 1.), ([(.97, .5), (.86, .62), (.74, .55)], .9), ([(.95, 1.8), (.82, 1.7), (.72, 1.85)], .8), ([(.95, -2.0), (.84, -2.1), (.74, -1.95)], .8), ([(.955, -2.6), (.95, -1.2), (.955, 0), (.95, 1.2), (.955, 2.6)], .75)): strk(sn, L, wf * w, ln, lab)
                sl = s.ids['sleeve' + q]
                for L in ([(.35, .4), (.6, .3), (.8, .55)], [(.30, 1.6), (.6, 1.5), (.85, 1.8)], [(.3, -1.0), (.65, -.9), (.9, -1.2)]): strk(sl, L, wf * .8, pal['blue_dk'] if s.px > 300 else ln, ('sleeve' + q,))
                if s.px > 260: strk(sl, [(.90, -2.9), (.90, -1.4), (.90, 0), (.90, 1.4), (.90, 2.9)], wf * .5, pal['blue_dk'], ('sleeve' + q,), taper=(.1, .1))
            for L, w in (([(.02, -.9), (.16, -.6), (.30, -.5)], 1.), ([(.02, .7), (.14, .55), (.24, .6)], .9), ([(.02, 1.6), (.16, 1.45), (.28, 1.55)], .9), ([(.02, 2.5), (.15, 2.35), (.26, 2.45)], .8), ([(.02, -2.3), (.16, -2.2), (.30, -2.35)], .8), ([(.80, 1.25), (.66, 1.15), (.52, 1.2)], .9), ([(.80, -1.25), (.66, -1.15), (.52, -1.2)], .9)): strk(ch, L, wf * w, ln, ('torso',))
            for L, w in (([(.48, -.3), (.36, -.2), (.26, -.28)], .9), ([(.48, .6), (.34, .5), (.22, .6)], .9), ([(.48, 2.6), (.34, 2.4), (.22, 2.5)], .8)): strk(s.ids['pelvis'], L, wf * w, ln, ('pelvis',))

    def stroke(s, cel, Pq, w, col, seed=0, taper=(.2, .2), press=.30):
        Q = resample(np.asarray(Pq, float), 1.5)
        if len(Q) < 4: return
        T = np.diff(Q, axis=0); a = np.unwrap(np.arctan2(T[:, 1], T[:, 0])); k = 3; tn = np.abs(a[k:] - a[:-k]) if len(a) > k else np.zeros(0); cut = [0]; j = 0; th = .95 if s.px > 300 else 1.3
        while j < len(tn):
            if tn[j] > th:
                e = j
                while e + 1 < len(tn) and tn[e + 1] > th: e += 1
                i = int(j + np.argmax(tn[j:e + 1])) + 2
                if i - cut[-1] > 4: cut.append(i)
                j = e + k
            j += 1
        cut.append(len(Q) - 1)
        for n_, (a_, b_) in enumerate(zip(cut[:-1], cut[1:])):
            g = Q[a_:b_ + 1]
            if len(g) < 4: continue
            Ln = len(g) * 1.5; c_ = .16 if s.px > 300 else .07; cel.line(g, w, col, taper=(taper[0] if n_ == 0 else min(c_, 9 / Ln), taper[1] if n_ == len(cut) - 2 else min(c_, 9 / Ln)), press=press, seed=seed * 17 + n_, tremor=.2, tmin=.28 if s.px > 300 else .55)

    def lines(s, cel):
        g, Z, lab = s.g, s.Z, s.lab; h, w = Z.shape; big = np.where(s.hit, Z, 1e6).astype(f32); ink = s.pal['line']; lc = s.light
        jobs = [(li, nm, {li}) + tuple(s.lpoly.get(li, ([], []))) for li, nm in enumerate(s.lname)]
        for q in 'LR':
            U = {i for i, n in enumerate(s.lname) if n.startswith('hand' + q)}; m = np.isin(lab, list(U))
            if m.sum() > 8: raw = [c for c in mask_polys(m, g, s.box, raw=True, up=2, min_area=3., smooth=max(3, int(3.2 * g + 1.5)), sig=.6) if len(c) >= 10]; jobs.append((500 + ord(q), 'hand' + q, U, raw, [clean_closed(c, max(.45, s.px * .0012)) for c in raw]))
        for li, nm, own, raws, cls_ in jobs:
            merged = li >= 500; fing = nm.startswith('hand') and not merged
            for ci, c in enumerate(raws):
                n = len(c); T = np.roll(c, -2, 0) - np.roll(c, 2, 0); T /= np.hypot(*T.T)[:, None] + 1e-9; Nn = np.c_[T[:, 1], -T[:, 0]]
                def samp(off): q = (c + Nn * off / g - [s.box[0], s.box[1]]) * g; return np.clip(np.floor(q[:, 0]).astype(int), 0, w - 1), np.clip(np.floor(q[:, 1]).astype(int), 0, h - 1)
                u1, v1 = samp(1.6); u2, v2 = samp(-1.6)
                if np.isin(lab[v1, u1], list(own)).mean() > np.isin(lab[v2, u2], list(own)).mean(): (u1, v1), (u2, v2) = (u2, v2), (u1, v1); Nn = -Nn
                lo = lab[v1, u1]; jump = big[v1, u1] - big[v2, u2]; ishand = nm.startswith('hand'); oth = np.array([q >= 0 and s.lname[q].startswith('hand') and s.lname[q][4] == nm[4:5] for q in lo]); thr = np.where(ishand & oth, .016, .035); cls = np.zeros(n, int); oo = ~np.isin(lo, list(own)); cls[lo < 0] = 1; occ = (lo >= 0) & oo & (jump > thr); cls[occ] = 2; flat = (lo >= 0) & oo & (np.abs(jump) <= thr) & (lo > (min(own))); cls[flat] = 3
                if fing: cls[~oth] = 0
                tp = [s.lname[q][:5] if q >= 0 else '' for q in lo]
                for j in range(n):                                          # no seam where thigh meets pelvis, or foot meets its own ankle
                    if cls[j] == 3 and ({nm[:5], tp[j]} <= {'pelvi', 'thigh', 'torso'} or (ishand and tp[j][:4] == 'hand' and (s.S < 60 or {nm[-1], tp[j][-1]} & {'L', 'R', 't'}))): cls[j] = 0
                s.runs(cel, c, cls, li, ci, Nn, cls_[ci], ink)

    def runs(s, cel, c, cls, li, ci, Nn, clean, ink):
        n = len(c); k = 5
        for v in (1, 2, 3):
            m = (cls == v).astype(np.uint8); mm = np.r_[m[-k:], m, m[:k]][None]; mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, np.ones((1, 2 * k + 1), np.uint8))[0, k:-k]; cls = np.where((cls == 0) & (mm > 0), v, cls)
        ch = np.nonzero(cls != np.roll(cls, 1))[0]
        if len(ch) == 0:
            if cls[0] == 0: return
            segs = [(0, n, cls[0], True)]
        else: segs = [(a, b if b > a else b + n, cls[a], False) for a, b in zip(ch, np.r_[ch[1:], ch[0] + n])]
        for a, b, v, closed in segs:
            if v == 0: continue
            idx = np.arange(a, b + (1 if closed else 1)) % n; Pq = c[idx]; Q, ix, nn_ = clean; i0 = clean_index(a, ix, nn_, n); M = len(Q); i1 = i0 + M if closed else clean_index(b % n, ix, nn_, n)
            if i1 <= i0: i1 += M
            ii = np.arange(math.ceil(i0), math.floor(i1) + 1) % M
            if len(ii) >= 3: Pq = Q[ii]
            L = np.hypot(*np.diff(Pq, axis=0).T).sum()
            if L < max(3., s.px * .010) and not closed: continue
            w = {1: s.wo, 2: s.wo * .84, 3: s.wf * 1.1}[int(v)]; nm = s.lname[li] if li < 500 else 'hand'
            if nm.startswith('hand'): w *= .85 if li >= 500 else .56
            nn = Nn[idx].mean(0); lit = nn @ np.array([s.light[0], -s.light[1]]) / (np.hypot(s.light[0], s.light[1]) + 1e-6) if abs(nn).sum() > .3 else 0; w *= 1 - .22 * lit
            s.stroke(cel, Pq, w, ink, seed=_hash(s.seed, li, ci, int(v)) & 0xffff, taper=(.02, .02) if closed else ((.12, .12) if v == 1 else (.2, .28)))

    # ------------------------------------------------ belt knot, tail
    def knot(s, cel):
        k = s.sk
        if s.px < 70: return
        o = k.pel + k.Rp @ [.02, .40, .50]; n = k.Rp[:, 2]
        if n[2] < -.05: return
        px = s.S * s.hk; pr = lambda Q: s.cam.project(w3(np.asarray(Q)))[0]; p0, ok = s.vis(w3(o)[None], ('belt', 'pelvis', 'torso'), .45)
        if not ok[0]: return
        wv = np.array([s.wind[0], -s.wind[1], 0.]) * s.wind[2]; grav = np.array([0, -1., 0.]); pal = s.pal; up = k.Rp[:, 1]; rt = k.Rp[:, 0]
        for q, (sx, ln) in enumerate(((-1, .62), (1, .72))):
            Pq = [o]; d = unit(rt * sx * .55 - up * .75 + n * .25)
            for j in range(6):
                d = unit(d + grav * .16 + wv * .42 + n * .02 + rt * .10 * math.sin(s.f * .55 + j * .9 + q * 2) * s.wind[2] * 1.4); Pq.append(Pq[-1] + d * ln / 6)
            C = spl(pr(Pq), n=5); wd = s.S * .098 * np.linspace(.72, 1.15, len(C)); T = np.gradient(C, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; A, B = C + N * wd[:, None], C - N * wd[:, None]; Pl = np.vstack([A, B[::-1]]); cel.fill(Pl, pal['blue']); sd = 1 if (N[len(N) // 2] @ np.array([s.light[0], -s.light[1]])) < 0 else -1
            cel.fill(np.vstack([C + N * (wd * .12 * sd)[:, None], (A if sd > 0 else B)[::-1]]), pal['blue_sh']); outline(cel, Pl, s.wf * 1.25, pal['line'], 60 + q)
        for sx in (-1, 1):
            Lp = spl(pr([o, o + rt * sx * .17 + up * .10 + n * .05, o + rt * sx * .24 + up * .0 + n * .04, o + rt * sx * .15 - up * .07 + n * .05]), True, n=5); cel.fill(Lp, pal['blue']); outline(cel, Lp, s.wf * 1.1, pal['line'], 63 + sx)
        a = np.linspace(0, 2 * pi, 13)[:-1]; K = pr(np.array([o + rt * .085 * math.cos(t) + up * .10 * math.sin(t) + n * .07 for t in a])); cel.fill(spl(K, True, n=3), pal['blue']); cel.fill(spl(K, True, n=3)[18:34], pal['blue_sh']); outline(cel, spl(K, True, n=3), s.wf * 1.1, pal['line'], 66)

    def tail_root(s): k = s.sk; return k.pel + k.Rp @ np.array([0, -.10, -.42])

    def _tail(s, tail):
        """-> (screen spine, radii, depth sign relative to the pelvis)"""
        if tail is None or tail is False: return None
        k = s.sk
        if isinstance(tail, str): Q = tail_curve(s.f, **s.pose['tail'])
        elif isinstance(tail, dict): Q = tail_curve(s.f, **{**s.pose['tail'], **tail})
        else: Q = np.asarray(tail, float)
        Q = KH._spl3(Q, 40); Pw = s.tail_root() + Q @ s.sk.Rp.T; p, z = s.cam.project(w3(Pw)); t = np.linspace(0, 1, len(Q)); r = s.S * s.D / np.maximum(z, .4) * .125 * (.92 + .22 * np.sin(np.pi * t) - .10 * t) * np.sqrt(np.clip((1.03 - t) / .10, 0, 1)); dz = float(np.mean(z[8:]) - s.D)
        return p, np.maximum(r, .4), (-1 if (z[6] > s.D + .05) else 1), z

    def draw_tail(s, cel, tl):
        p, r = tl[0], tl[1]; pal = s.pal; T = np.gradient(p, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; n = len(p); key = np.array([s.light[0], -s.light[1]]); sd = 1 if (N[n // 2] @ key) > 0 else -1
        t = np.linspace(0, 1, n); tf = np.linspace(0, n - 1, n * 5); I = lambda a: np.c_[np.interp(tf, np.arange(n), a[:, 0]), np.interp(tf, np.arange(n), a[:, 1])]; Pf, Nf = I(p), I(N); rf = np.interp(tf, np.arange(n), r); u = tf / (n - 1)
        big = s.S > 26
        def saw(ph):                                                       # fur: irregular tufts, never a saw blade
            if not big: return 0 * u
            r_ = np.random.default_rng(int(ph * 100) + 9); cuts = np.sort(r_.uniform(.06, .97, 9)); o = np.zeros_like(u)
            for c_ in cuts: o = np.maximum(o, np.clip(1 - np.abs(u - c_) / .035, 0, 1) ** 1.4 * (u <= c_) * r_.uniform(.5, 1.1))
            return o
        A = Pf + Nf * (rf * (1 + .20 * saw(0.)))[:, None]; B = Pf - Nf * (rf * (1 + .20 * saw(.45)))[:, None]; tip = Pf[-1] + T[-1] * r[n - 4] * .5; Pl = np.vstack([A, [tip], B[::-1]]); cel.fill(Pl, pal['fur'])
        cel.clip(Pl); D_, Lt = (B, A) if sd > 0 else (A, B); core = Pf - Nf * (rf * sd * (.10 + (.30 * saw(.2))))[:, None]; cel.fill(np.vstack([core, D_[::-1] + (D_ - Pf)[::-1] * .5]), pal['fur_sh'])
        if s.S > 18: h = Pf + Nf * (rf * sd * .58)[:, None]; i0, i1 = int(len(h) * .15), int(len(h) * .7); wd = rf[i0:i1] * .12 * np.sin(np.pi * np.linspace(0, 1, i1 - i0)); cel.fill(np.vstack([h[i0:i1] + Nf[i0:i1] * wd[:, None], (h[i0:i1] - Nf[i0:i1] * wd[:, None])[::-1]]), pal['fur_hi'])
        cel.unclip()
        s.stroke(cel, A, s.wo * .9, pal['line'], seed=71, taper=(.05, .05)); s.stroke(cel, B, s.wo * .9, pal['line'], seed=72, taper=(.05, .05)); cel.line(np.vstack([A[-6:], [tip], B[-6:][::-1]]), s.wo * .9, pal['line'], taper=(.2, .2), seed=73)
        if big:
            for j in range(5): q = int(len(Pf) * (.22 + .14 * j)); d = T[min(n - 1, q // 5)]; c = Pf[q] - Nf[q] * rf[q] * sd * .25; cel.line(np.array([c, c + d * rf[q] * .5 - Nf[q] * rf[q] * .12, c + d * rf[q] * 1.0 - Nf[q] * rf[q] * .05]), s.wf * .7, pal['line'], taper=(.15, .6), seed=80 + j)


def tail_curve(f=0, dir=(.3, .3, -1.), curl=.9, side=.2, amp=.20, speed=2 * pi / 20., phase=0., length=2.35, n=11, lag=.30, up=None):
    """resting tail with a travelling wave (one sway per 2 beats).  Every link repeats its parent a little later: follow-through.  Root-frame 3-D points"""
    d = unit(dir); Q = [np.zeros(3)]; L = length / (n - 1); ax1 = unit(np.cross(d, [0, 1, 0.])) if abs(d[1]) < .95 else np.array([1., 0, 0]); ax2 = np.array([0, 1., 0])
    for j in range(n - 1):
        u = j / (n - 1.); bend = curl * (.40 * math.sin(math.pi * min(1., u / .62)) if u < .62 else -.62 * math.sin(math.pi * (u - .62) / .55)); sway = amp * (.20 + 1.2 * u) * math.sin(f * speed + phase - lag * j) * .62 + side * (.5 - 1.25 * u) * .34
        d = rot(d, ax1, bend); d = rot(d, ax2, sway); Q.append(Q[-1] + d * L)
    return np.array(Q)


def rot(v, ax, a): ax = unit(ax); return v * math.cos(a) + np.cross(ax, v) * math.sin(a) + ax * (ax @ v) * (1 - math.cos(a))


class Chain:
    """real follow-through: a hanging chain dragged by its root.  C = Chain(root_fn, dir_fn, n, length, stiff, drag, grav, wind); C.at(f) -> (n, 3) points RELATIVE to the root, pure function of f (simulated from frame f0)"""
    def __init__(s, root, dirv, n=11, length=2.35, stiff=.34, drag=.16, grav=(0, -.010, 0), wind=None, f0=-24, sub=2): s.root, s.dirv, s.n, s.Ln, s.st, s.dr, s.g, s.w, s.f0, s.sub = root, dirv, n, length, stiff, drag, np.asarray(grav, float), wind, f0, sub; s.c = {}

    def at(s, f):
        f = int(f)
        if f in s.c: return s.c[f]
        L = s.Ln / (s.n - 1); r = np.asarray(s.root(s.f0), float); d = np.asarray(s.dirv(s.f0), float); Q = r + np.cumsum(np.vstack([[0, 0, 0], d]), 0) * 1. if len(d) == s.n - 1 else r + unit(d) * (np.arange(s.n) * L)[:, None]; V = np.zeros_like(Q)
        for k in range(s.f0 * s.sub + 1, f * s.sub + 1):
            t = k / s.sub; r = np.asarray(s.root(t), float); D_ = np.asarray(s.dirv(t), float); rest = np.vstack([[0, 0, 0], np.cumsum(D_, 0)]) if D_.ndim == 2 else unit(D_) * (np.arange(s.n) * L)[:, None]
            V = V * (1 - s.dr) + s.g / s.sub + (np.asarray(s.w(t), float) / s.sub if s.w else 0); Qn = Q + V; tgt = r + rest; u = np.linspace(1, 0, s.n)[:, None]; Qn = Qn + (tgt - Qn) * (s.st * u ** 1.6 / s.sub); Qn[0] = r
            for _ in range(6):
                for j in range(1, s.n): e = Qn[j] - Qn[j - 1]; Qn[j] = Qn[j - 1] + e / (np.linalg.norm(e) + 1e-9) * L
            V = (Qn - Q) * .96; Q = Qn
            if k % s.sub == 0: s.c[k // s.sub] = Q - r
        return s.c.get(f, Q - r)
