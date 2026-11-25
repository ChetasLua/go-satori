"""Skeleton, poses, in-betweens, cycles, contact IK for db.hero3.
Figure frame: x = the character's right, y = forward, z = up, metres.  Side values in poses are written OUTWARD (x out, y fwd, z up) and mirrored.
Pose = dict (all keys optional, defaults in STAND):
  root (right, fwd, up) pelvis offset, fractions of height H        body (pitch fwd+, roll right+, yaw left+) degrees: the WHOLE figure turned about the pelvis
  pel wst cst nck hed (pitch fwd+, roll right+, yaw left+) degrees   cL cR (up, fwd) extra shoulder girdle, cm
  hL hR wrist target from the shoulder, arm-length units, chest frame (lead='root': root frame)     eL eR elbow pole     sL sR arm stretch (1 = natural)
  wL wR (twist, flex, dev) wrist degrees     gL gR hand shape: fist open flat splay claw cup relax point soft hold flick limp chop v
  fL fR ankle target (out, fwd, up) from the ground point under the root, fractions of H      kL kR knee pole
  tL tR (toe-out yaw, toe-up pitch, toe bend) degrees from the ground     air = 1: feet hang from the shins instead (t = toe-out, ankle angle up+, toe bend)
"""
import numpy as np, math

D2R = math.pi / 180


def Rx(a):
    c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler(p):
    """(pitch fwd+, roll right+, yaw left+) degrees -> matrix"""
    return Rz(p[2] * D2R) @ Rx(-p[0] * D2R) @ Ry(p[1] * D2R)


def unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v); return v / (n + 1e-12)


# ------------------------------------------------------------------ proportions (fractions of H)
BODY = dict(H=1.76, heads=7.0, sw=.120, hw=.050, arm=.170, fore=.149, hand=.112, hipz=.520, kneez=.283, ankz=.045, waistz=.613, chestz=.695, neckz=.820, shz=.786, headz=.860,
            mass=1.0, leg=1.0, arml=1.0, limb=1.0, armm=1.0, legm=1.0, chest=1.0, waist=1.0, neck=1.0, hips=1.0, foot=1.0, muscle=1.0)
CHAR_BODY = dict(
    goku=dict(H=1.76, heads=6.9, sw=.127, limb=1.07, chest=1.04, waist=.98, neck=1.06),
    vegeta=dict(H=1.64, heads=6.35, sw=.132, mass=.98, limb=1.12, chest=1.08, waist=.98, neck=1.10, hw=.052),
    frieza=dict(H=1.60, heads=6.2, sw=.120, limb=.95, armm=.98, waist=.92, chest=.96, neck=.92, hips=.96, muscle=.9, hand=.116, foot=1.0, arm=.170, fore=.152),
    piccolo=dict(H=2.12, heads=7.7, sw=.126, limb=1.06, mass=.97, chest=1.04, waist=.95, hand=.110),
    gohan=dict(H=1.14, heads=4.1, sw=.098, hw=.048, arm=.152, fore=.128, hand=.108, hipz=.430, kneez=.235, ankz=.044, waistz=.515, chestz=.590, neckz=.710, shz=.668, headz=.745, limb=.98, mass=.98, chest=.88, waist=1.04, neck=.72, hips=1.02, muscle=.12, foot=1.15),
    kid=dict(H=1.10, heads=3.6, sw=.098, hw=.050, arm=.150, fore=.124, hand=.112, hipz=.405, kneez=.222, ankz=.044, waistz=.485, chestz=.560, neckz=.672, shz=.632, headz=.705, limb=1.04, mass=1.04, chest=.92, waist=1.06, neck=.74, hips=1.04, muscle=.16, foot=1.2),
)

STAND = dict(root=(0, 0, 0), body=(0, 0, 0), pel=(0, 0, 0), wst=(0, 0, 0), cst=(-2, 0, 0), nck=(6, 0, 0), hed=(-4, 0, 0), cL=(0, 0), cR=(0, 0),
             hL=(.22, .06, -.955), hR=(.22, .06, -.955), eL=(.45, -1, -.1), eR=(.45, -1, -.1), wL=(0, 6, 0), wR=(0, 6, 0), gL='relax', gR='relax',
             fL=(.068, 0, .045), fR=(.068, 0, .045), kL=(.2, 1, 0), kR=(.2, 1, 0), tL=(16, 0, 0), tR=(16, 0, 0), sL=1., sR=1., air=0., lead='chest')


def P(base=None, **kw):
    """new pose: P(fL=..., pel=...) on top of STAND, or P('guard', hR=...)"""
    d = dict(STAND if base is None else (POSES[base] if isinstance(base, str) else base)); d.update(kw); return d


def mirror(p):
    p = POSES[p] if isinstance(p, str) else p
    q = dict(p)
    for k in list(p):
        if k[-1] in 'LR' and k[:-1] in ('h', 'e', 'w', 'g', 'f', 'k', 't', 's', 'c'):
            q[k[:-1] + ('R' if k[-1] == 'L' else 'L')] = p[k]
    for k in ('body', 'pel', 'wst', 'cst', 'nck', 'hed'): a = p.get(k, STAND[k]); q[k] = (a[0], -a[1], -a[2])
    r = p.get('root', (0, 0, 0)); q['root'] = (-r[0], r[1], r[2]); return q


def to_root(p):
    if p.get('lead', 'chest') == 'root': return p
    g = lambda k: np.asarray(p.get(k, STAND[k]), float); Rc = euler(g('pel')) @ euler(g('wst')) @ euler(g('cst')); q = dict(p); q['lead'] = 'root'
    for s, sg in (('L', -1.), ('R', 1.)):
        m = np.array([sg, 1, 1.])
        for k in ('h', 'e'): q[k + s] = (Rc @ (g(k + s) * m)) * m
    return q


def mix_pose(a, b, t):
    """joint-space in-between (t may overshoot to 1.3).  hands travel on arcs round the shoulder, feet on straight lines, angles linear"""
    a = POSES[a] if isinstance(a, str) else a; b = POSES[b] if isinstance(b, str) else b
    if t <= 0: return a
    if abs(t - 1) < 1e-6: return b
    if a.get('lead', 'chest') != b.get('lead', 'chest'): a, b = to_root(a), to_root(b)
    o = {}
    for k, v0 in STAND.items():
        va, vb = a.get(k, v0), b.get(k, v0)
        if isinstance(v0, str): o[k] = va if t < .5 else vb; continue
        x, y = np.asarray(va, float), np.asarray(vb, float)
        if k in ('hL', 'hR'):
            la, lb = np.linalg.norm(x), np.linalg.norm(y); u, w = x / (la + 1e-9), y / (lb + 1e-9)
            om = math.acos(float(np.clip(u @ w, -1, 1)))
            if om > 1.75 and 0 < t < 1:                                       # thrown straight: past the ribs, elbow folded, never over the head
                q = x * (1 - t) + y * t; bias = np.array([.55, .0, -.80]); n_ = np.linalg.norm(q); r_ = .46
                if n_ < r_: h_ = q @ bias; q = q + bias * (-h_ + math.sqrt(max(h_ * h_ + r_ * r_ - n_ * n_, 0.)))
                o[k] = q; continue
            if 1e-3 < om < math.pi - .05: d = (math.sin((1 - t) * om) * u + math.sin(t * om) * w) / math.sin(om)
            else: d = u * (1 - t) + w * t
            o[k] = d * min(la * (1 - t) + lb * t, 1.02)
        elif k in ('eL', 'eR', 'kL', 'kR'): o[k] = unit(unit(x) * (1 - t) + unit(y) * t)
        else: o[k] = x * (1 - t) + y * t
    return o


def add_pose(a, **d):
    """a with numeric offsets added"""
    a = POSES[a] if isinstance(a, str) else a
    o = dict(a)
    for k, v in d.items(): o[k] = np.asarray(a.get(k, STAND[k]), float) + np.asarray(v, float)
    return o


def at(s, x, y, z, b=None):
    """wrist target in H fractions from the CHEST joint (x right, y fwd, z up) -> value for hL / hR"""
    b = b or BODY; sg = -1. if s == 'L' else 1.; A = b['arm'] + b['fore']
    return (float(sg * x - b['sw']) / A, float(y + .006) / A, float(z - (b['shz'] - b['chestz'])) / A)


# ------------------------------------------------------------------ FK / IK
def two_bone(a, t, pole, l1, l2, stretch=1.0):
    """-> mid joint, end joint (clamped), e1 = unit direction the mid joint points to"""
    d = t - a; D = np.linalg.norm(d); m = l1 + l2
    if D > m * .995:
        k = min(D / (m * .995), stretch); l1 *= k; l2 *= k; m = l1 + l2
        if D > m * .998: d = d / D * m * .998; D = m * .998
    D = max(D, abs(l1 - l2) * 1.02 + .18 * min(l1, l2)); d = unit(d)
    x = (l1 * l1 - l2 * l2 + D * D) / (2 * D); h = math.sqrt(max(l1 * l1 - x * x, 0.))
    p = np.asarray(pole, float); p = p - d * (p @ d)
    if np.linalg.norm(p) < 1e-4: p = np.cross(d, [.3, .9, .1])
    p = unit(p)
    return a + d * x + p * h, a + d * D, p


HOLD = dict(wrist=(0, 0, 0), fist=(0, .60, -.04), palm=(0, .34, -.135), finger=(.06, .98, -.02), back=(0, .40, .12), edge=(-.20, .40, 0), cup=(0, .46, -.22))


def rot_to(a, b, mx=math.pi):
    """rotation matrix taking unit a toward unit b by at most mx radians"""
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(a @ b); an = math.atan2(s, c)
    if s < 1e-8: return np.eye(3)
    k = v / s; an = min(an, mx); K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(an) * K + (1 - math.cos(an)) * K @ K


def slerp_R(A, B, t):
    """rotation from A toward B by the part t of the angle between them"""
    if t <= 0: return A
    if t >= 1: return B
    M = B @ A.T; an = math.acos(float(np.clip((np.trace(M) - 1) / 2, -1, 1)))
    if an < 1e-5: return A
    v = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]]); n = np.linalg.norm(v)
    if n < 1e-6: return A
    k = v / n; an *= t; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]); return (np.eye(3) + math.sin(an) * K + (1 - math.cos(an)) * K @ K) @ A


class Skel:
    """joint positions and frames for one pose.  pos: ground point under the root (or the pelvis itself with pivot='pelvis')"""

    def __init__(self, body, pose, pos=(0, 0, 0), heading=0., pitch=0., roll=0., scale=1.0, pivot='ground', reach=None, look=None, lookw=1.0):
        b = dict(BODY); b.update(body); self.b = b; H = b['H'] * scale; self.H = H; self.sc = scale
        pose = POSES[pose] if isinstance(pose, str) else pose
        g = lambda k: np.asarray(pose.get(k, STAND[k]), float)
        bd = g('body'); z0 = b['hipz'] + .02
        Rr = Rz(heading) @ euler((bd[0] + math.degrees(pitch), bd[1] + math.degrees(roll), bd[2]))
        pos = np.asarray(pos, float); rest = np.array([0, 0, z0 * H])
        if pivot == 'pelvis': pv = pos
        else: pv = pos + rest
        self.pv = pv; self.origin = pv - Rr @ rest; self.Rr = Rr; self.heading = heading
        r = g('root') * H
        shift = np.zeros(3)
        for it in range(3 if reach else 1):
            self._solve(b, pose, g, pv + Rr @ r + shift, H, reach or {}, look, lookw)
            if not reach: break
            miss = np.zeros(3); n = 0
            for s in 'LR':
                q = reach.get(s)
                if q is not None and self.short[s] is not None: miss += self.short[s] * q.get('w', 1.) * q.get('body', .9); n += 1
            if n == 0 or np.linalg.norm(miss) < 1e-4: break
            shift = shift + miss / n
            ls = np.linalg.norm(shift); lim = .16 * H
            if ls > lim: shift *= lim / ls

    def _solve(self, b, pose, g, pel, H, reach, look, lookw):
        Rr = self.Rr; self.pel = pel
        Rp = Rr @ euler(g('pel')); Rw = Rp @ euler(g('wst')); Rc = Rw @ euler(g('cst')); Rn = Rc @ euler(g('nck')); Rh = Rn @ euler(g('hed'))
        z0 = b['hipz'] + .02
        self.waist = pel + Rp @ np.array([0, 0, (b['waistz'] - z0) * H]); self.chest = self.waist + Rw @ np.array([0, .004 * H, (b['chestz'] - b['waistz']) * H])
        self.neck = self.chest + Rc @ np.array([0, -.014 * H, (b['neckz'] - b['chestz']) * H])
        if look is not None:
            c = self.neck + Rc @ np.array([0, 0, .09 * H]); d = Rc.T @ unit(np.asarray(look, float) - c)
            # turn and nod are taken from the chest the way a neck does it: never past what a neck can do, never through the floor
            yaw = math.atan2(d[0], d[1]); nod = math.asin(float(np.clip(d[2], -1, 1))); back = abs(yaw); w = lookw * float(np.clip((math.pi - back) / .7, 0, 1)) if back > 2.1 else lookw
            if back > 1.06: yaw = math.copysign(1.06 + .40 * math.tanh((back - 1.06) / .40), yaw)
            nod = float(np.clip(nod, -.95, 1.15))
            sy, cy, sn, cn = math.sin(yaw), math.cos(yaw), math.sin(nod), math.cos(nod); Rl = Rc @ np.array([[cy, sy * cn, -sy * sn], [-sy, cy * cn, -cy * sn], [0, sn, cn]])
            cu = Rc[:, 2]; w_ = float(np.clip((cu[2] - .05) / .55, 0, 1)); w_ = .80 * w_ * w_ * (3 - 2 * w_); cu = unit(cu * (1 - w_) + np.array([0, 0, 1.]) * w_)   # upright people keep their eyes level, flying ones follow the chest
            r0 = cu - Rl[:, 1] * (cu @ Rl[:, 1]); n0 = np.linalg.norm(r0); f_ = float(np.clip((n0 - .22) / .45, 0, 1))
            if f_ > 0:
                r0 = r0 / n0; hr = float(g('hed')[1] + g('nck')[1]) * D2R; t_ = r0 * math.cos(hr) + np.cross(Rl[:, 1], r0) * -math.sin(hr); Rl = rot_to(Rl[:, 2], t_, 1.0 * f_) @ Rl
            Rn = slerp_R(Rn, Rl, .45 * w); Rh = slerp_R(Rh, Rl, w)
        self.Rp, self.Rw, self.Rc, self.Rn, self.Rh = Rp, Rw, Rc, Rn, Rh
        self.headb = self.neck + Rn @ np.array([0, .012 * H, (b['headz'] - b['neckz']) * H])
        hh = H / b['heads']; self.hh = hh; self.u = hh / 2
        self.head = self.headb + Rh @ np.array([0, .014 * H, (1 - hh / H / 2 - b['headz']) * H])
        self.sh, self.el, self.wr, self.hip, self.kn, self.an, self.e1a, self.e1l, self.Rhand, self.Rfoot, self.grip, self.acr = ({} for _ in range(12))
        self.bend = {}; self.lift = {}; self.kbend = {}; self.short = dict(L=None, R=None); self.reached = {}
        hl = b['hand'] * H
        for s, sg in (('L', -1.), ('R', 1.)):
            out = lambda v: np.array([sg * v[0], v[1], v[2]])
            la, lf = b['arm'] * H * b['arml'], b['fore'] * H * b['arml']; st = float(pose.get('s' + s, 1.)); A = la + lf
            tgt = out(g('h' + s)) * A; Rs = Rc if pose.get('lead', 'chest') == 'chest' else Rr
            base0 = self.chest + Rc @ np.array([sg * b['sw'] * H, -.006 * H, (b['shz'] - b['chestz']) * H]); c = g('c' + s)
            q = reach.get(s); pole = Rs @ out(g('e' + s))
            if q is not None and q.get('pole') is not None: pole = np.asarray(q['pole'], float)
            def girdle(tg):
                dirc = unit(Rc.T @ tg); up = max(0., dirc[2] + .25); fw = dirc[1]; ac = -dirc[0] * sg
                far = float(np.clip(np.linalg.norm(tg) / A - .75, 0, .25)) * 4
                bs = base0 + Rc @ np.array([-sg * (.022 * up + .016 * max(0, ac) + .010 * max(0, fw)) * H, (.024 * max(fw, -.5) * (.6 + .4 * far) + .012 * max(0, ac)) * H + c[1] * .01, (.040 * up - .006 * max(0, -dirc[2] - .8)) * H + c[0] * .01])
                return bs, up
            base, up = girdle(Rs @ tgt); want = base + Rs @ tgt
            stretch = max(st, 1.0)
            if st != 1.: want = base + Rs @ tgt * st
            ht = None
            for it in range(5 if q is not None else 1):
                if q is not None:
                    w = q.get('w', 1.); pt = np.asarray(q['p'], float)
                    if ht is None:
                        fd0 = unit(pt - base); wt = pt - fd0 * hl * .5
                    else: wt = pt - ht @ (np.array(HOLD[q.get('part', 'fist')]) * hl)
                    wt = base + (want - base) * (1 - w) + (wt - base) * w if w < 1 else wt
                    base, up = girdle(wt - base0)
                    stretch = max(st, q.get('stretch', 1.14))
                else: wt = want
                el, wr, e1 = two_bone(base, wt, pole, la, lf, stretch)
                fd = unit(wr - el); tw, fl, dv = g('w' + s) * D2R
                e = unit(e1 - fd * (e1 @ fd)); sidev = np.cross(e, fd) * sg
                back = unit(sidev * math.cos(tw) + e * math.sin(tw)); thumb = np.cross(back, fd) * sg
                fd2 = unit(fd * math.cos(fl) - back * math.sin(fl)); back2 = unit(back * math.cos(fl) + fd * math.sin(fl))
                fd3 = unit(fd2 * math.cos(dv) + thumb * math.sin(dv)); th3 = unit(thumb * math.cos(dv) - fd2 * math.sin(dv))
                ht = np.stack([th3, fd3, back2], 1)
                if q is not None and (q.get('palm') is not None or q.get('fingers') is not None):
                    f_ = unit(q['fingers']) if q.get('fingers') is not None else fd3
                    b_ = -unit(q['palm']) if q.get('palm') is not None else back2
                    if q.get('palm') is not None and q.get('fingers') is None: f_ = unit(f_ - b_ * (f_ @ b_))
                    else: b_ = unit(b_ - f_ * (b_ @ f_))
                    ht = np.stack([np.cross(b_, f_) * sg, f_, b_], 1)
                wr_want = wt
            self.sh[s] = base; self.el[s], self.wr[s], self.e1a[s] = el, wr, e1
            self.bend[s] = math.acos(float(np.clip(unit(el - base) @ unit(wr - el), -1, 1))); self.lift[s] = up
            self.Rhand[s] = ht; self.grip[s] = pose.get('g' + s, 'relax')
            if q is not None:
                self.grip[s] = q.get('g', self.grip[s]); self.short[s] = wr_want - wr; self.reached[s] = wr + ht @ (np.array(HOLD[q.get('part', 'fist')]) * hl)
            # legs
            lt, ls = (b['hipz'] - b['kneez']) * H * b['leg'], (b['kneez'] - b['ankz']) * H * b['leg']
            hip = pel + Rp @ np.array([sg * b['hw'] * H, 0, -.02 * H]); self.hip[s] = hip
            ft = self.origin + Rr @ (out(g('f' + s)) * H)
            t = g('t' + s); air = float(pose.get('air', 0.)); kp = Rr @ out(g('k' + s)); qf = reach.get('f' + s)
            if qf is not None:
                w = qf.get('w', 1.); pt = np.asarray(qf['p'], float)
                if qf.get('part', 'sole') == 'sole': pt = pt - np.asarray(qf.get('dir', (0, 0, -1.)), float) * b['ankz'] * H
                ft = ft * (1 - w) + pt * w
                if qf.get('pole') is not None: kp = np.asarray(qf['pole'], float)
            kn, an, k1 = two_bone(hip, ft, kp, lt, ls, 1.04)
            self.kn[s], self.an[s], self.e1l[s] = kn, an, k1
            self.kbend[s] = math.acos(float(np.clip(unit(kn - hip) @ unit(an - kn), -1, 1)))
            Rg = Rr @ Rz(-sg * t[0] * D2R) @ Rx(t[1] * D2R)
            if air > .5:
                sd = unit(an - kn); fy = unit(k1 - sd * (k1 @ sd)); fx = unit(np.cross(fy, -sd)); R0 = np.stack([fx, fy, -sd], 1); Rg = R0 @ Rz(-sg * t[0] * D2R) @ Rx(t[1] * D2R)
            if qf is not None and qf.get('R') is not None: Rg = np.asarray(qf['R'], float)
            self.Rfoot[s] = Rg; self.lift['t' + s] = t[2]
            self.acr[s] = base + Rc @ np.array([0, 0, .026 * H])

    def pt(self, frame, v):
        o, R = dict(pelvis=(self.pel, self.Rp), waist=(self.waist, self.Rw), chest=(self.chest, self.Rc), neck=(self.neck, self.Rn), head=(self.head, self.Rh), root=(self.origin, self.Rr))[frame]
        return o + R @ np.asarray(v, float)

    def spot(self, name):
        """named world points for contact: stomach chest back chin cheekL cheekR forehead wristL wristR elbowL.. fistL fistR palmL palmR shoulderL.. kneeL.. soleL toeL tailbone"""
        H, b = self.H, self.b; hl = b['hand'] * H; m = H / 1.76 * b['mass']
        if name == 'stomach': return self.waist + self.Rw @ np.array([0, .112 * m, .035 * H])
        if name == 'solar': return self.chest + self.Rc @ np.array([0, .125 * m, -.01 * H])
        if name == 'chest': return self.chest + self.Rc @ np.array([0, .150 * m, .045 * H])
        if name == 'back': return self.chest + self.Rc @ np.array([0, -.130 * m, .04 * H])
        if name == 'chin': return self.head + self.Rh @ np.array([0, .62, -.98]) * self.u
        if name in ('cheekL', 'cheekR'): return self.head + self.Rh @ (np.array([.56 * (1 if name[-1] == 'R' else -1), .45, -.45]) * self.u)
        if name == 'forehead': return self.head + self.Rh @ (np.array([0, .84, .42]) * self.u)
        if name == 'face': return self.head + self.Rh @ (np.array([0, .86, -.25]) * self.u)
        if name == 'tailbone': return self.pel + self.Rp @ np.array([0, -.10 * m, .0])
        s = name[-1]; n = name[:-1]
        if n in HOLD or n in ('fist', 'palm', 'finger'): return self.wr[s] + self.Rhand[s] @ (np.array(HOLD[n]) * hl)
        if n == 'forearm': return (self.el[s] + self.wr[s]) / 2
        if n == 'ankle': return self.an[s]
        if n == 'sole': return self.an[s] - self.Rfoot[s][:, 2] * b['ankz'] * H
        if n == 'toe': return self.an[s] + self.Rfoot[s] @ np.array([0, .10 * H, -.6 * b['ankz'] * H])
        return dict(wrist=self.wr, elbow=self.el, shoulder=self.sh, knee=self.kn, hip=self.hip)[n][s]


# ------------------------------------------------------------------ key poses (figure frame)
POSES = {}
_ = P
_F = dict(gL='fist', gR='fist')
POSES['stand'] = P()
POSES['contra'] = P(root=(.018, 0, -.004), pel=(2, 5, -6), wst=(0, -5, 4), cst=(-3, -4, 8), nck=(5, 2, -4), hed=(-2, 3, -6), hL=(.24, .10, -.93), hR=(.25, .02, -.93), fL=(.085, .03, .045), fR=(.065, -.015, .045), tL=(26, 0, 0), tR=(12, 0, 0), kL=(.3, 1, 0))
POSES['fists_down'] = P(hL=(.30, .05, -.90), hR=(.30, .05, -.90), eL=(.7, -1, 0), eR=(.7, -1, 0), cL=(1.5, 0), cR=(1.5, 0), cst=(-6, 0, 0), nck=(8, 0, 0), fL=(.095, 0, .045), fR=(.095, 0, .045), tL=(22, 0, 0), tR=(22, 0, 0), **_F)
POSES['arms_crossed'] = P(cst=(-4, 0, 0), hL=(-.40, .50, -.42), hR=(-.42, .42, -.36), eL=(1, .25, -.55), eR=(1, .25, -.55), gL='relax', gR='relax', wL=(-70, 0, 0), wR=(-70, 0, 0), fL=(.085, 0, .045), fR=(.085, 0, .045), tL=(22, 0, 0), tR=(22, 0, 0))
POSES['lookup'] = P(cst=(-8, 0, 0), nck=(-18, 0, 0), hed=(-26, 0, 0), hL=(.20, .03, -.96), hR=(.20, .03, -.96), fL=(.055, 0, .045), fR=(.055, 0, .045), gL='limp', gR='limp')
POSES['hands_up'] = P(cst=(-8, 0, 0), nck=(-4, 0, 0), hed=(-14, 0, 0), hL=(.30, .10, .92), hR=(.30, .10, .92), eL=(.9, -.5, -.2), eR=(.9, -.5, -.2), gL='open', gR='open', wL=(70, -40, 0), wR=(70, -40, 0), cL=(3, 0), cR=(3, 0))
# ---- fighting on the ground
POSES['guard'] = P(root=(0, -.01, -.085), pel=(8, 0, -34), wst=(8, 0, -6), cst=(8, 0, -4), nck=(6, 0, 18), hed=(-6, 0, 20), cL=(2, 3), cR=(2, 2), hL=(.06, .64, .06), hR=(.12, .36, -.02), eL=(.75, -.1, -1), eR=(1, -.2, -.8), wL=(-10, -6, 0), wR=(10, -6, 0), fL=(.085, .20, .045), fR=(.15, -.17, .054), kL=(.35, 1, 0), kR=(.75, .7, 0), tL=(8, 0, 0), tR=(50, -8, 0), **_F)
POSES['stance_goku'] = P('guard', root=(0, -.015, -.115), pel=(10, 0, -46), wst=(4, 0, -8), cst=(2, 0, -6), nck=(4, 0, 30), hed=(-4, 0, 26), hL=(.18, .72, .02), hR=(.30, .18, .16), eL=(.8, -.1, -1), eR=(1, -.5, -.4), gL='chop', gR='fist', wL=(-30, -20, 0), fL=(.09, .26, .045), fR=(.18, -.22, .050))
POSES['stance_vegeta'] = P('guard', root=(0, -.02, -.10), pel=(6, 0, 34), wst=(2, 0, 6), cst=(0, 0, 6), nck=(4, 0, -22), hed=(-4, 0, -20), hR=(.30, .56, .12), hL=(.40, .05, -.30), eR=(.9, .0, -1), eL=(1, -.8, 0), gL='fist', gR='claw', wR=(-40, -10, 0), fR=(.09, .22, .045), fL=(.17, -.18, .052), kR=(.35, 1, 0), kL=(.75, .7, 0), tR=(8, 0, 0), tL=(50, -8, 0))
POSES['crouch'] = P(root=(0, -.02, -.215), pel=(40, 0, 0), wst=(12, 0, 0), cst=(6, 0, 0), nck=(-20, 0, 0), hed=(-24, 0, 0), hL=(.40, -.20, -.80), hR=(.40, -.20, -.80), eL=(.8, -1, .3), eR=(.8, -1, .3), fL=(.105, .06, .045), fR=(.105, -.10, .060), kL=(.45, 1, 0), kR=(.45, 1, 0), tL=(18, 0, 0), tR=(18, -18, 18), **_F)
POSES['launch'] = P('crouch', lead='root', root=(0, .03, -.225), pel=(52, 0, 0), wst=(10, 0, 0), cst=(2, 0, 0), nck=(-30, 0, 0), hed=(-30, 0, 0), hL=(.42, -.70, -.22), hR=(.42, -.70, -.22), eL=(.6, -.6, 1), eR=(.6, -.6, 1), fL=(.09, .10, .045), fR=(.09, -.26, .085), tL=(10, 0, 0), tR=(10, -46, 40))
POSES['lunge'] = P(lead='root', root=(0, .13, -.17), pel=(36, 0, -20), wst=(6, 0, -8), cst=(2, 0, -10), nck=(-20, 0, 18), hed=(-22, 0, 14), hL=(.18, .60, -.16), hR=(.34, -.66, .10), eL=(1, 0, -.4), eR=(.5, -.6, 1), gL='fist', gR='fist', fL=(.085, .30, .045), fR=(.07, -.42, .095), kL=(.3, 1, 0), kR=(.2, 1, -.5), tL=(8, 0, 0), tR=(20, -52, 40))
POSES['punch_wind'] = P(lead='root', root=(.02, -.085, -.125), pel=(6, 10, -58), wst=(8, 8, -20), cst=(4, 10, -24), nck=(4, -4, 48), hed=(-4, -4, 34), cR=(4, -4), cL=(2, 3), hL=(-.06, .72, .06), hR=(.22, -.62, .18), eL=(.9, -.2, -.5), eR=(1, -.5, .9), wL=(-30, 0, 0), wR=(-40, -18, 0), fL=(.06, .24, .045), fR=(.16, -.22, .052), kL=(.1, 1, 0), kR=(.9, .5, 0), tL=(0, 6, 0), tR=(68, 0, 0), **_F)
POSES['punch_hit'] = P(lead='root', root=(-.03, .13, -.150), pel=(18, -6, 38), wst=(10, -6, 18), cst=(10, -8, 22), nck=(-6, 4, -38), hed=(-8, 4, -36), cR=(4, 7), cL=(2, -2), hR=(-.10, 1.0, .06), hL=(.46, -.42, -.10), eR=(1, 0, -.3), eL=(.5, -1, .2), wR=(-80, -4, 0), fL=(.11, .30, .045), fR=(.06, -.36, .080), kL=(.15, 1, 0), kR=(-.1, 1, -.3), tL=(6, 0, 0), tR=(26, -38, 38), **_F)
POSES['punch_follow'] = P(lead='root', root=(-.05, .18, -.185), pel=(26, -14, 58), wst=(14, -8, 24), cst=(14, -10, 28), nck=(-6, 6, -46), hed=(-6, 6, -40), cR=(3, 8), hR=(-.66, .66, -.16), hL=(.58, -.60, .10), eR=(.9, .5, .2), eL=(.4, -1, .5), wR=(-70, -6, 0), fL=(.12, .30, .045), fR=(.05, -.38, .085), kL=(.2, 1, 0), kR=(0, 1, -.3), tL=(6, 0, 0), tR=(30, -44, 42), **_F)
POSES['jab'] = P('guard', lead='root', root=(0, .04, -.09), pel=(10, 0, -18), wst=(6, 0, -10), cst=(6, 0, -16), nck=(0, 0, 22), hed=(-4, 0, 20), hL=(.04, 1.0, .06), hR=(.14, .40, .06), eL=(.9, 0, -.5), wL=(-80, 0, 0), cL=(3, 6))
POSES['uppercut'] = P(lead='root', root=(-.01, .07, .01), pel=(-6, -8, 30), wst=(-8, -8, 12), cst=(-12, -10, 14), nck=(-4, 4, -24), hed=(-14, 4, -16), cR=(8, 3), hR=(.02, .52, .68), hL=(.50, -.40, -.42), eR=(.8, .6, -1), eL=(.5, -1, -.1), wR=(60, -10, 0), fL=(.06, .10, .065), fR=(.10, -.20, .095), kL=(.1, 1, 0), kR=(.2, 1, 0), tL=(4, -14, 16), tR=(16, -52, 46), **_F)
POSES['gut_punch'] = P(lead='root', root=(-.02, .12, -.215), pel=(24, -10, 30), wst=(14, -8, 14), cst=(8, -10, 16), nck=(-18, 4, -24), hed=(-16, 4, -24), cR=(-1, 6), hR=(-.02, .86, -.10), hL=(.50, -.46, -.24), eR=(.7, -.6, -1), eL=(.5, -1, .2), wR=(40, -6, 0), fL=(.12, .28, .045), fR=(.08, -.32, .080), kL=(.25, 1, 0), kR=(0, 1, -.3), tL=(8, 0, 0), tR=(26, -38, 38), **_F)
POSES['elbow'] = P(lead='root', root=(-.03, .13, -.13), pel=(14, -6, 46), wst=(8, -6, 20), cst=(8, -8, 26), nck=(-6, 4, -44), hed=(-6, 4, -38), cR=(6, 6), hR=(-.30, .36, .12), hL=(.40, .20, -.40), eR=(.35, 1, .25), eL=(.6, -1, 0), wR=(-60, 0, 0), fL=(.10, .28, .045), fR=(.07, -.32, .07), kL=(.2, 1, 0), tL=(6, 0, 0), tR=(26, -30, 30), **_F)
POSES['backfist'] = P(lead='root', root=(0, .06, -.14), pel=(14, -8, 70), wst=(6, -6, 26), cst=(6, -8, 34), nck=(-4, 4, -60), hed=(-4, 4, -40), cR=(3, 4), hR=(.50, .86, .08), hL=(.42, -.44, -.30), eR=(.6, -.6, -.4), eL=(.6, -1, .1), wR=(60, 0, 0), fL=(.16, .20, .045), fR=(.10, -.26, .065), kL=(.4, 1, 0), kR=(.4, 1, 0), tL=(20, 0, 0), tR=(34, -20, 20), **_F)
POSES['palm'] = P(lead='root', root=(-.02, .12, -.12), pel=(14, -4, 30), wst=(8, -4, 14), cst=(8, -4, 18), nck=(-4, 2, -32), hed=(-4, 2, -28), cR=(2, 8), hR=(-.06, .98, .0), hL=(.40, -.30, -.34), eR=(.7, 0, -.8), eL=(.6, -1, 0), gR='splay', gL='claw', wR=(-85, -62, 0), fL=(.10, .28, .045), fR=(.08, -.30, .070), kL=(.2, 1, 0), kR=(.2, 1, -.2), tL=(8, 0, 0), tR=(34, -26, 28))
POSES['shove'] = P('palm', root=(-.03, .20, -.14), pel=(30, -4, 26), hR=(-.04, 1.0, .02), hL=(.34, -.62, .06), fL=(.11, .34, .045), fR=(.07, -.44, .10), tR=(20, -56, 44))
POSES['kick_wind'] = P(root=(-.03, 0, -.03), pel=(-4, -10, 20), wst=(6, -4, 6), cst=(8, -4, 8), nck=(0, 2, -14), hed=(-2, 2, -12), hL=(.16, .50, .14), hR=(.36, .28, -.10), eL=(.8, 0, -1), eR=(1, -.3, -.6), fL=(.02, .02, .045), fR=(.22, .08, .30), kL=(.4, 1, 0), kR=(.8, 1, .4), tL=(40, 0, 0), tR=(20, -50, 0), **_F)
POSES['kick'] = P(root=(-.055, -.035, -.010), pel=(-18, -46, 62), wst=(4, 6, -10), cst=(10, 12, -22), nck=(4, 16, -18), hed=(0, 12, -16), cL=(2, 2), cR=(1, -3), hL=(.10, .48, .10), hR=(.76, -.42, -.36), eL=(.6, -.2, -1), eR=(.4, -1, .2), fL=(.02, -.02, .045), fR=(.06, .50, .685), kL=(.55, 1, 0), kR=(.9, .2, .3), tL=(70, 0, 0), tR=(-60, -64, -10), **_F)
POSES['kick_side'] = P(root=(-.05, -.02, -.01), pel=(0, -38, 70), wst=(2, 8, -14), cst=(6, 12, -20), nck=(0, 14, -22), hed=(0, 10, -22), hL=(.20, .40, .20), hR=(.70, -.10, -.42), eL=(.8, -.2, -1), eR=(.5, -1, .2), fL=(.02, -.02, .045), fR=(.10, .56, .50), kL=(.55, 1, 0), kR=(.9, .3, .5), tL=(70, 0, 0), tR=(-40, -50, 0), **_F)
POSES['kick_spin'] = P(root=(-.04, -.02, .0), pel=(10, -40, 120), wst=(4, 6, -18), cst=(6, 10, -30), nck=(0, 10, -50), hed=(0, 8, -44), hL=(.50, .20, -.10), hR=(.70, -.30, .0), eL=(.8, -.6, -.4), eR=(.6, -1, .2), fL=(.0, .0, .07), fR=(.18, .50, .56), kL=(.4, 1, 0), kR=(1, .1, .4), tL=(60, -20, 20), tR=(-60, -60, -10), **_F)
POSES['knee'] = P(lead='root', root=(-.02, .10, .0), pel=(-14, -6, 24), wst=(6, -2, 6), cst=(14, -2, 6), nck=(0, 2, -16), hed=(4, 2, -14), hL=(.30, .20, .10), hR=(.40, -.50, -.30), eL=(.9, 0, -1), eR=(.5, -1, .2), fL=(.03, -.04, .060), fR=(.04, .10, .34), kL=(.2, 1, 0), kR=(.2, 1, .9), tL=(20, -24, 24), tR=(0, -60, 0), **_F)
POSES['axe_up'] = P(root=(-.04, -.05, .0), pel=(-26, -10, 20), wst=(10, 0, 0), cst=(14, 4, -6), nck=(6, 4, -8), hed=(8, 4, -6), hL=(.40, .30, -.40), hR=(.50, -.40, -.50), eL=(.7, -.8, 0), eR=(.5, -1, 0), fL=(.02, -.02, .045), fR=(.02, .30, .84), kL=(.4, 1, 0), kR=(.2, .2, 1), tL=(40, 0, 0), tR=(0, 30, 0), **_F)
POSES['block_high'] = P(lead='root', root=(0, -.05, -.11), pel=(4, 0, 20), wst=(10, 0, 8), cst=(10, 0, 8), nck=(6, 0, -20), hed=(0, 0, -16), cL=(8, 5), cR=(8, 5), hL=(-.06, .46, .52), hR=(-.16, .40, .40), eL=(.8, .2, -1), eR=(.8, .2, -1), wL=(-60, 0, 0), wR=(-60, 0, 0), fL=(.15, -.15, .052), fR=(.11, .20, .045), kL=(.8, .7, 0), kR=(.35, 1, 0), tL=(44, -8, 0), tR=(10, 0, 0), **_F)
POSES['block_cross'] = P(lead='root', root=(0, -.07, -.13), pel=(-4, 0, 0), wst=(8, 0, 0), cst=(12, 0, 0), nck=(8, 0, 0), hed=(0, 0, 0), cL=(6, 6), cR=(6, 6), hL=(-.46, .52, .16), hR=(-.46, .46, .10), eL=(1, .3, -.8), eR=(1, .3, -.8), wL=(-40, 0, 0), wR=(-40, 0, 0), fL=(.13, .10, .045), fR=(.13, -.14, .052), kL=(.5, 1, 0), kR=(.5, 1, 0), tL=(18, 0, 0), tR=(30, -8, 0), **_F)
POSES['block_back'] = P(lead='root', root=(.02, -.13, -.165), pel=(-12, 8, -30), wst=(8, 8, -14), cst=(10, 8, -16), nck=(2, -4, 34), hed=(4, -6, 24), cL=(8, 5), cR=(2, -3), hL=(-.02, .56, .40), hR=(.56, -.42, -.46), eL=(.8, .3, -1), eR=(.5, -1, .3), gL='fist', gR='claw', wL=(-60, -8, 0), wR=(-20, -20, 0), fL=(.06, .26, .052), fR=(.18, -.26, .052), kL=(.2, 1, .1), kR=(1, .55, 0), tL=(0, 12, 0), tR=(66, 0, 0))
POSES['duck'] = P(root=(0, .02, -.235), pel=(38, 6, -20), wst=(18, 4, -8), cst=(16, 4, -8), nck=(-20, 0, 14), hed=(-22, 0, 12), cL=(3, 4), cR=(3, 2), hL=(.12, .52, .16), hR=(.26, .20, -.24), eL=(.8, 0, -1), eR=(1, -.6, -.4), fL=(.10, .20, .045), fR=(.15, -.22, .060), kL=(.4, 1, 0), kR=(.8, .7, 0), tL=(10, 0, 0), tR=(50, -14, 16), **_F)
POSES['lean_back'] = P(root=(.02, -.10, -.10), pel=(-30, 6, 18), wst=(-8, 4, 6), cst=(-12, 4, 8), nck=(18, 0, -8), hed=(14, 0, -8), hL=(.60, .10, -.50), hR=(.66, -.20, -.40), eL=(.5, -1, 0), eR=(.5, -1, 0), gL='relax', gR='relax', fL=(.10, .16, .045), fR=(.12, -.12, .052), kL=(.3, 1, 0), kR=(.5, 1, 0), tL=(14, 0, 0), tR=(40, 0, 0))
POSES['slip'] = P(root=(.06, .02, -.12), pel=(10, 22, 26), wst=(6, 12, 8), cst=(6, 12, 10), nck=(0, -14, -20), hed=(0, -12, -18), hL=(.40, .30, -.50), hR=(.50, -.10, -.55), eL=(.8, -.6, -.4), eR=(.7, -1, 0), gL='relax', gR='relax', fL=(.12, -.10, .050), fR=(.14, .14, .045), kL=(.4, 1, 0), kR=(.6, 1, 0), tL=(30, -10, 0), tR=(14, 0, 0))
# ---- taking it
POSES['hit_gut'] = P(lead='root', root=(0, -.14, -.02), pel=(-18, 0, 4), wst=(34, 0, 0), cst=(40, 4, 4), nck=(18, 4, 0), hed=(-20, 4, 0), cL=(2, 6), cR=(2, 6), hL=(.34, .82, -.34), hR=(.40, .78, -.20), eL=(.9, 0, .6), eR=(.9, 0, .6), gL='splay', gR='claw', fL=(.11, .16, .075), fR=(.09, .06, .085), kL=(.2, 1, .2), kR=(.2, 1, .2), tL=(8, -40, 10), tR=(8, -44, 10))
POSES['hit_head'] = P(lead='root', root=(.04, -.06, -.02), pel=(-10, 8, 24), wst=(-6, 8, 14), cst=(-10, 10, 18), nck=(-20, 20, 34), hed=(-24, 18, 34), cL=(2, -2), cR=(3, 2), hL=(.56, -.24, -.50), hR=(.62, .40, .10), eL=(.5, -1, .2), eR=(.7, -.4, .4), gL='claw', gR='splay', fL=(.09, .14, .045), fR=(.15, -.16, .075), kL=(.3, 1, 0), kR=(.6, 1, 0), tL=(6, 0, 0), tR=(40, -30, 24))
POSES['hit_up'] = P(lead='root', root=(0, -.05, .08), pel=(-30, 4, 6), wst=(-12, 4, 4), cst=(-16, 4, 6), nck=(-26, 4, 0), hed=(-34, 4, 0), hL=(.60, -.10, -.64), hR=(.66, .10, -.50), eL=(.5, -1, .2), eR=(.6, -1, .4), gL='splay', gR='splay', fL=(.08, .16, .10), fR=(.10, -.02, .16), kL=(.3, 1, .2), kR=(.3, 1, 0), tL=(8, -40, -10), tR=(8, -50, -10))
POSES['hit_spin'] = P(lead='root', body=(30, 40, 0), root=(0, 0, .12), pel=(10, 30, -30), wst=(6, 10, -24), cst=(6, 10, -34), nck=(10, 14, -30), hed=(8, 14, -24), hL=(.76, .10, .30), hR=(.30, .70, -.40), eL=(.5, -1, 0), eR=(.8, .2, -.4), gL='splay', gR='claw', fL=(.18, -.20, .14), fR=(.10, .26, .20), kL=(.3, 1, 0), kR=(.4, 1, .4), tL=(8, -50, -10), tR=(8, -30, 0), air=1)
POSES['thrown'] = P(root=(0, -.06, .06), pel=(-34, -10, -8), wst=(20, -6, -8), cst=(26, -8, -10), nck=(22, -10, -4), hed=(16, -12, -4), hL=(.32, .84, .10), hR=(.48, .74, .22), eL=(.8, 0, -.6), eR=(.8, 0, -.4), gL='splay', gR='claw', fL=(.12, .40, .20), fR=(.06, .30, .08), kL=(.4, 1, .6), kR=(.3, 1, .4), tL=(10, -30, 0), tR=(10, -40, 0))
POSES['swung'] = P(lead='root', body=(70, 0, 0), root=(0, 0, 0), pel=(-6, 0, 10), wst=(-8, 4, 6), cst=(-10, 4, 8), nck=(-10, 0, 0), hed=(-10, 0, 0), hR=(.10, .10, .98), hL=(.40, -.10, .70), eR=(.5, .2, -1), eL=(.6, -.6, 0), gR='fist', gL='splay', fL=(.12, -.22, .06), fR=(.02, -.10, .02), kL=(.3, 1, -.2), kR=(.2, 1, -.2), tL=(8, -50, -14), tR=(8, -50, -14), air=1)
POSES['tumble'] = P(lead='root', body=(150, 20, 0), pel=(30, 10, 0), wst=(24, 6, 6), cst=(26, 6, 8), nck=(20, 0, 0), hed=(14, 0, 0), hL=(.30, .50, -.50), hR=(.50, .30, .10), eL=(.9, 0, -.4), eR=(.9, 0, -.2), gL='claw', gR='splay', fL=(.12, .30, .32), fR=(.08, .12, .24), kL=(.4, 1, .5), kR=(.3, 1, .4), tL=(8, -20, 0), tR=(8, -30, 0), air=1)
POSES['fall'] = P(lead='root', body=(-100, 0, 0), pel=(-6, 0, 0), wst=(10, 0, 0), cst=(10, 0, 0), nck=(-14, 0, 0), hed=(-16, 0, 0), hL=(.60, .62, .20), hR=(.50, .72, .34), eL=(.9, -.3, 0), eR=(.9, -.3, 0), gL='limp', gR='limp', fL=(.10, .20, .10), fR=(.07, .10, .20), kL=(.3, 1, .4), kR=(.3, 1, .4), tL=(8, -30, -8), tR=(8, -34, -8), air=1)
POSES['stagger'] = P(root=(.03, -.04, -.075), pel=(-12, 8, 14), wst=(14, 6, 10), cst=(20, 8, 14), nck=(10, 6, -4), hed=(6, 8, -6), cL=(-1, 3), cR=(2, 1), hL=(.30, .34, -.82), hR=(-.06, .34, -.46), eL=(.5, -1, 0), eR=(1, -.2, -.2), gL='relax', gR='claw', fL=(.12, .10, .045), fR=(.085, -.17, .064), kL=(-.2, 1, 0), kR=(.4, 1, 0), tL=(-8, 0, 0), tR=(34, -20, 24))
# ---- landing, sliding, kneeling
POSES['land'] = P(root=(.015, -.02, -.300), pel=(34, 4, -14), wst=(20, 2, -6), cst=(14, 2, -8), nck=(-24, 0, 14), hed=(-26, 0, 14), cL=(2, 5), cR=(3, -2), hL=(.14, .74, -.60), hR=(.76, -.34, .26), eL=(.8, -.1, .6), eR=(.5, -1, .2), gL='claw', gR='fist', wL=(-60, -30, 0), fL=(.10, .13, .045), fR=(.24, -.16, .066), kL=(.5, 1, 0), kR=(.9, .8, 0), tL=(16, 0, 0), tR=(50, -18, 22))
POSES['land_settle'] = P('crouch', root=(0, -.01, -.14), pel=(22, 0, 0), wst=(8, 0, 0), cst=(4, 0, 0), nck=(-10, 0, 0), hed=(-12, 0, 0), hL=(.40, -.05, -.88), hR=(.40, -.05, -.88), fR=(.105, -.06, .045), tR=(18, 0, 0))
POSES['slide'] = P(lead='root', root=(.03, -.10, -.300), pel=(-12, 12, 30), wst=(18, -4, -8), cst=(14, -6, -14), nck=(-4, 0, -6), hed=(-4, 0, -8), cL=(2, 2), hL=(.70, .48, -.08), hR=(.60, -.60, -.50), eL=(.4, -.6, -1), eR=(.5, -1, .4), gL='splay', gR='claw', wR=(-30, -40, 0), fL=(.12, .36, .045), fR=(.05, .04, .064), kL=(.2, 1, 0), kR=(1, .4, .1), tL=(6, 14, 0), tR=(64, -20, 10))
POSES['brace_back'] = P(lead='root', root=(0, -.12, -.16), pel=(14, 0, -20), wst=(10, 0, -4), cst=(8, 0, -4), nck=(0, 0, 14), hed=(0, 0, 12), hL=(-.02, .80, .10), hR=(-.10, .78, .06), eL=(1, 0, -.7), eR=(1, 0, -.7), gL='splay', gR='splay', wL=(-80, -70, 0), wR=(-80, -70, 0), fL=(.11, .24, .050), fR=(.14, -.28, .055), kL=(.3, 1, 0), kR=(.5, 1, 0), tL=(8, 10, 0), tR=(40, -6, 0))
POSES['kneel'] = P(root=(0, -.04, -.262), pel=(8, 0, 0), wst=(14, 0, 0), cst=(12, 0, 2), nck=(14, 0, 0), hed=(8, 0, 0), hL=(.22, .20, -.82), hR=(.16, .36, -.70), eL=(.6, -1, 0), eR=(.8, -.8, 0), gL='relax', gR='fist', fL=(.085, .18, .045), fR=(.065, -.235, .094), kL=(.25, 1, 0), kR=(.10, 1, -.6), tL=(8, 0, 0), tR=(0, -72, 44))
POSES['kneel_up'] = P('kneel', wst=(6, 0, 0), cst=(4, 0, 0), nck=(0, 0, 0), hed=(-8, 0, 0), hL=(.10, .56, -.52), hR=(.36, -.05, -.86), eL=(.8, -.4, -.3))
POSES['knees'] = P(root=(0, -.03, -.262), pel=(8, 0, 0), wst=(16, 0, 0), cst=(16, 3, 2), nck=(16, 2, 0), hed=(10, 3, 0), cL=(-1, 2), cR=(-1, 2), hL=(.26, .20, -.90), hR=(.24, .14, -.92), eL=(.6, -1, 0), eR=(.6, -1, 0), gL='limp', gR='limp', fL=(.095, -.245, .050), fR=(.085, -.255, .050), kL=(.35, 1, -.5), kR=(.25, 1, -.5), tL=(4, -84, 6), tR=(4, -84, 6))
POSES['knees_fist'] = P('knees', root=(0, -.09, -.275), pel=(34, 0, 0), wst=(22, 0, 0), cst=(16, 0, 0), nck=(0, 0, 0), hed=(-4, 0, 0), hR=(.14, .44, -.86), hL=(.20, .30, -.90), gR='fist', gL='claw')
POSES['cradle'] = P('knees', pel=(4, 0, 0), wst=(10, 0, 4), cst=(10, 2, 6), nck=(18, 4, 8), hed=(12, 4, 6), hL=(-.06, .62, -.46), hR=(.02, .60, -.40), eL=(.8, -.6, -.4), eR=(.8, -.6, -.4), gL='open', gR='open', wL=(150, -10, 0), wR=(150, -10, 0))
POSES['carried'] = P(lead='root', body=(-74, 0, 0), pel=(-4, 0, 0), wst=(10, 2, 0), cst=(6, 4, 4), nck=(-24, 8, 4), hed=(-22, 10, 6), hL=(.34, -.50, -.60), hR=(.30, .20, -.70), eL=(.6, -.4, -1), eR=(.6, .3, -1), gL='limp', gR='limp', fL=(.07, .16, .16), fR=(.06, .10, .10), kL=(.3, 1, .6), kR=(.3, 1, .5), tL=(14, -34, -4), tR=(14, -34, -4), air=1)
POSES['lie'] = P(lead='root', body=(-90, 0, 0), pel=(0, 0, 0), wst=(0, 0, 0), cst=(0, 0, 0), nck=(2, 0, 0), hed=(0, 0, 0), hL=(.80, -.14, -.46), hR=(.86, -.16, -.34), eL=(.3, -1, .2), eR=(.3, -1, .2), gL='open', gR='open', wL=(60, 0, 0), wR=(60, 0, 0), fL=(.10, .0, .04), fR=(.12, .0, .05), kL=(.6, 1, 0), kR=(.6, 1, 0), tL=(40, -24, 0), tR=(40, -24, 0), air=1)
POSES['lotus'] = P(root=(0, -.02, -.455), wst=(6, 0, 0), cst=(0, 0, 0), nck=(8, 0, 0), hed=(0, 0, 0), hL=(.10, .50, -.58), hR=(.10, .50, -.58), eL=(1, -.4, -.3), eR=(1, -.4, -.3), gL='soft', gR='soft', wL=(120, 0, 0), wR=(120, 0, 0), fL=(-.06, .17, .04), fR=(-.04, .20, .075), kL=(1, .30, .22), kR=(1, .30, .22), tL=(-70, -20, 0), tR=(-70, -20, 0))
# ---- energy
POSES['power'] = P(root=(0, -.01, -.115), pel=(6, 0, 0), wst=(8, 0, 0), cst=(-10, 0, 0), nck=(2, 0, 0), hed=(-6, 0, 0), cL=(5, 1), cR=(5, 1), hL=(.40, .30, -.66), hR=(.40, .30, -.66), eL=(1, -.75, .2), eR=(1, -.75, .2), wL=(50, -14, 0), wR=(50, -14, 0), fL=(.165, .02, .045), fR=(.165, -.02, .045), kL=(.7, 1, 0), kR=(.7, 1, 0), tL=(34, 0, 0), tR=(34, 0, 0), **_F)
POSES['power_in'] = P('power', root=(0, 0, -.150), pel=(16, 0, 0), wst=(16, 0, 0), cst=(12, 0, 0), nck=(2, 0, 0), hed=(-4, 0, 0), hL=(.14, .46, -.60), hR=(.14, .46, -.60))
POSES['power_out'] = P('power', pel=(-4, 0, 0), wst=(-6, 0, 0), cst=(-20, 0, 0), nck=(-14, 0, 0), hed=(-26, 0, 0), cL=(6, -3), cR=(6, -3), hL=(.62, .05, -.56), hR=(.62, .05, -.56), eL=(.8, -1, .3), eR=(.8, -1, .3))
POSES['kame_wind'] = P(root=(.015, -.03, -.135), pel=(10, 4, -46), wst=(10, 6, -18), cst=(8, 8, -24), nck=(0, -4, 46), hed=(-4, -4, 38), hR=at('R', .165, .045, -.055), hL=at('L', .115, .060, -.050), eR=(1, -1, .25), eL=(.9, .3, -.3), gL='cup', gR='cup', wR=(60, -30, 0), wL=(-50, -20, 0), cL=(0, 6), cR=(1, -3), fL=(.10, .22, .045), fR=(.17, -.20, .050), kL=(.35, 1, 0), kR=(.8, .7, 0), tL=(10, 0, 0), tR=(52, -4, 0))
POSES['kame_fire'] = P(lead='root', root=(0, .07, -.145), pel=(16, 0, 0), wst=(8, 0, 0), cst=(8, 0, 0), nck=(-8, 0, 0), hed=(-10, 0, 0), cL=(3, 8), cR=(3, 8), hL=(-.36, .90, .04), hR=(-.36, .90, .04), eL=(1, 0, -.5), eR=(1, 0, -.5), gL='cup', gR='cup', wL=(0, -64, 0), wR=(0, -64, 0), fL=(.11, .24, .045), fR=(.13, -.30, .060), kL=(.3, 1, 0), kR=(.5, 1, 0), tL=(10, 0, 0), tR=(34, -14, 10))
POSES['kame_hold'] = P('kame_fire', root=(0, .03, -.180), pel=(22, 0, 0), wst=(12, 0, 0), cst=(6, 0, 0), hL=(-.34, .84, .04), hR=(-.34, .84, .04), fL=(.12, .24, .045), fR=(.15, -.32, .066))
POSES['ff_wide'] = P(root=(0, -.02, -.10), pel=(0, 0, 0), wst=(-4, 0, 0), cst=(-16, 0, 0), nck=(4, 0, 0), hed=(-2, 0, 0), cL=(3, -4), cR=(3, -4), hL=(.98, .0, .08), hR=(.98, .0, .08), eL=(0, -.3, -1), eR=(0, -.3, -1), gL='splay', gR='splay', wL=(80, -50, 0), wR=(80, -50, 0), fL=(.15, .04, .045), fR=(.15, -.06, .045), kL=(.6, 1, 0), kR=(.6, 1, 0), tL=(28, 0, 0), tR=(34, 0, 0))
POSES['ff_fire'] = P('kame_fire', gL='splay', gR='splay', hL=(-.34, .92, .06), hR=(-.34, .92, .06), wL=(-30, -78, 0), wR=(-30, -78, 0))
POSES['beam_one'] = P('palm', root=(-.01, .03, -.10), pel=(4, 0, 34), wst=(2, 0, 10), cst=(0, 0, 14), nck=(0, 0, -30), hed=(-2, 0, -26), hR=(-.02, 1.0, .04), hL=(.34, .0, -.86), gL='fist', fL=(.11, -.14, .050), fR=(.10, .20, .045), kL=(.6, 1, 0), tL=(40, 0, 0), tR=(8, 0, 0))
POSES['finger_up'] = P(root=(.01, 0, 0), pel=(0, 3, -4), cst=(-4, 0, 4), hR=(.30, .34, .52), hL=(.28, .05, -.88), eR=(.9, 0, -1), gR='point', gL='soft', wR=(0, -10, 0), nck=(0, 0, 0), hed=(-4, 4, 4))
POSES['finger_aim'] = P(lead='root', root=(0, 0, 0), pel=(0, 0, 24), wst=(0, 0, 6), cst=(-4, 0, 10), nck=(2, 0, -22), hed=(0, 4, -18), hR=(.02, .98, .06), hL=(.28, .0, -.90), eR=(1, 0, -.6), gR='point', gL='soft', wR=(-60, 0, 0))
POSES['flick'] = P('finger_aim', hR=(-.02, .86, .22), eR=(1, 0, -.5), gR='flick', wR=(60, -30, 0))
POSES['part'] = P(lead='root', root=(0, .02, -.02), pel=(4, 0, 10), cst=(-2, 0, 8), nck=(6, 0, -8), hed=(2, 0, -8), hR=(.10, .86, .20), hL=(.30, .0, -.90), eR=(1, 0, -.8), gR='open', gL='relax', wR=(-70, -50, 0), fL=(.06, .12, .045), fR=(.06, -.12, .065), tR=(10, -24, 20))
POSES['conduct'] = P(cst=(-8, 0, 0), nck=(8, 0, 0), hed=(6, 0, 0), hL=(.74, .22, -.46), hR=(.74, .22, -.46), eL=(.3, -1, -.1), eR=(.3, -1, -.1), gL='soft', gR='soft', wL=(120, -20, 0), wR=(120, -20, 0), fL=(.040, -.01, .045), fR=(.040, -.04, .07), tL=(6, -52, -10), tR=(6, -56, -12), kL=(.1, 1, 0), kR=(.1, 1, 0), air=1)
POSES['moon_up'] = P(cst=(-16, 0, 0), nck=(-6, 0, 0), hed=(-14, 0, 0), hL=(.42, .04, .88), hR=(.42, .04, .88), eL=(.9, -.6, 0), eR=(.9, -.6, 0), gL='splay', gR='splay', wL=(90, -50, 0), wR=(90, -50, 0), cL=(4, 0), cR=(4, 0), fL=(.07, .02, .05), fR=(.06, -.06, .09), tL=(6, -40, -10), tR=(6, -48, -12), kL=(.2, 1, 0), kR=(.2, 1, 0), air=1)
# ---- air (use Fig.fly / body / pitch to aim the whole figure)
_A = dict(air=1, kL=(.15, 1, 0), kR=(.15, 1, 0))
POSES['hover'] = P(root=(0, 0, 0), pel=(-4, 0, 0), wst=(2, 0, 0), cst=(0, 0, 0), nck=(6, 0, 0), hed=(0, 0, 0), hL=(.46, .12, -.84), hR=(.46, .12, -.84), eL=(.7, -1, .1), eR=(.7, -1, .1), fL=(.085, .0, .07), fR=(.085, -.07, .12), tL=(10, -34, -8), tR=(10, -44, -10), **dict(_A, **_F))
POSES['hover_guard'] = P('guard', body=(6, 0, 0), root=(0, 0, 0), fL=(.09, .14, .13), fR=(.12, -.14, .10), tL=(8, -30, -6), tR=(16, -50, -10), air=1)
POSES['fly'] = P(lead='root', body=(62, 0, 0), pel=(0, 0, 0), wst=(-6, 0, 0), cst=(-12, 0, 0), nck=(-24, 0, 0), hed=(-30, 0, 0), cL=(1, -3), cR=(1, -3), hL=(.34, -.22, -.90), hR=(.34, -.22, -.90), eL=(.6, -1, .1), eR=(.6, -1, .1), fL=(.050, -.02, .03), fR=(.055, -.06, .06), tL=(4, -54, -14), tR=(4, -58, -16), **dict(_A, **_F))
POSES['fly_punch'] = P('fly', body=(66, 0, 0), cst=(-6, 0, 14), wst=(-4, 0, 8), hR=(-.02, .20, .97), hL=(.34, -.28, -.86), eR=(.8, -.5, -.2), cR=(6, 2), wR=(-70, 0, 0), nck=(-26, 0, -8), hed=(-32, 0, -6), fL=(.06, -.04, .03), fR=(.06, .10, .16), kR=(.2, 1, .2))
POSES['fly_both'] = P('fly', body=(74, 0, 0), hR=(.08, .16, .97), hL=(.08, .16, .97), eR=(.8, -.5, -.2), eL=(.8, -.5, -.2), cR=(6, 2), cL=(6, 2), wR=(-70, 0, 0), wL=(-70, 0, 0))
POSES['fly_bank'] = P('fly', body=(58, 26, 0), pel=(0, 8, 8), wst=(-4, 6, 8), cst=(-10, 6, 12), hL=(.52, -.10, -.78), hR=(.40, -.40, -.72), fL=(.06, .06, .08), fR=(.06, -.10, .06), kL=(.2, 1, .2))
POSES['dive'] = P('fly', body=(128, 0, 0), nck=(-30, 0, 0), hed=(-30, 0, 0))
POSES['rise'] = P('fly', body=(8, 0, 0), cst=(-14, 0, 0), nck=(-18, 0, 0), hed=(-26, 0, 0), hL=(.30, -.16, -.92), hR=(.30, -.16, -.92), fL=(.05, -.02, .03), fR=(.06, .10, .16), kR=(.2, 1, .3))
POSES['dive_punch'] = P(lead='root', root=(0, .0, .0), body=(36, 0, 0), pel=(6, -10, 34), wst=(4, -6, 18), cst=(2, -8, 24), nck=(-18, 4, -40), hed=(-18, 4, -32), cR=(5, 8), cL=(3, -2), hR=(-.06, .98, -.06), hL=(.56, -.62, .42), eR=(1, 0, .5), eL=(.6, -.6, 1), wR=(-80, -4, 0), fL=(.10, .20, .22), fR=(.05, -.40, .24), kL=(.5, 1, .5), kR=(.2, 1, -.6), tL=(10, -24, 0), tR=(6, -40, -12), **dict(air=1, **_F))
POSES['leap'] = P(root=(0, 0, .02), pel=(14, -6, 18), wst=(-6, -4, 8), cst=(-14, -6, 18), nck=(-10, 2, -20), hed=(-12, 2, -18), cR=(6, -4), cL=(2, 4), hR=(.40, -.46, .52), hL=(.34, .72, .12), eR=(1, -.5, .4), eL=(.6, -.2, -1), gR='fist', gL='claw', wR=(10, -16, 0), fL=(.07, .20, .28), fR=(.06, -.36, .18), kL=(.2, 1, .25), kR=(.12, 1, -.5), tL=(4, -10, 0), tR=(4, -40, -14), air=1)
POSES['tuck'] = P(lead='root', root=(0, 0, 0), pel=(48, 0, 0), wst=(24, 0, 0), cst=(22, 0, 0), nck=(14, 0, 0), hed=(8, 0, 0), hL=(-.06, .50, -.40), hR=(-.04, .48, -.42), eL=(1, .2, -.2), eR=(1, .2, -.2), fL=(.09, .20, .36), fR=(.08, .16, .34), kL=(.3, 1, .7), kR=(.3, 1, .7), tL=(6, -10, 0), tR=(6, -10, 0), **dict(air=1, **_F))
POSES['air_kick'] = P(lead='root', root=(0, 0, 0), body=(-18, -20, 0), pel=(-8, -22, 40), wst=(6, 6, -6), cst=(10, 8, -18), nck=(4, 8, -20), hed=(2, 8, -18), hL=(.30, .40, .10), hR=(.70, -.40, -.30), eL=(.7, -.2, -1), eR=(.4, -1, .2), fL=(.05, -.06, .20), fR=(.14, .62, .46), kL=(.3, 1, .1), kR=(.7, .4, .4), tL=(8, -30, -8), tR=(0, -36, -8), **dict(air=1, **_F))
POSES['hang_back'] = P(lead='root', body=(-30, 0, 0), pel=(-10, 0, 0), wst=(14, 0, 0), cst=(16, 0, 0), nck=(10, 0, 0), hed=(6, 0, 0), hL=(.40, .80, .20), hR=(.50, .70, .30), eL=(.9, -.3, 0), eR=(.9, -.3, 0), gL='splay', gR='claw', fL=(.09, .20, .16), fR=(.06, .10, .08), kL=(.3, 1, .4), kR=(.3, 1, .4), tL=(8, -40, -10), tR=(8, -40, -10), air=1)
# ---- sprint: right-foot contact, down, push, flight (left half = mirror)
POSES['run_contact'] = P(root=(0, 0, -.045), pel=(24, -2, -14), wst=(6, 2, 10), cst=(2, 2, 18), nck=(-12, 0, -8), hed=(-12, 0, -6), cL=(1, 5), cR=(3, -4), hL=(.04, .62, -.02), hR=(.34, -.60, -.18), eL=(.5, -.3, -1), eR=(.35, -1, .45), fR=(.035, .30, .052), fL=(.05, -.38, .30), kR=(.05, 1, 0), kL=(.1, 1, -.6), tR=(4, 14, 0), tL=(4, -86, 0), **_F)
POSES['run_down'] = P(root=(0, 0, -.077), pel=(28, -4, -6), wst=(8, 3, 6), cst=(4, 3, 8), nck=(-14, 0, -4), hed=(-12, 0, -4), cL=(1, 2), cR=(2, -1), hL=(.12, .38, -.40), hR=(.30, -.30, -.50), eL=(.5, -.7, -.7), eR=(.4, -1, .1), fR=(.03, .12, .045), fL=(.05, -.24, .22), kR=(.05, 1, 0), kL=(.1, 1, -.2), tR=(4, 0, 0), tL=(4, -76, 0), **_F)
POSES['run_push'] = P(root=(0, 0, -.004), pel=(26, 3, 16), wst=(4, -2, -12), cst=(0, -2, -20), nck=(-10, 0, 8), hed=(-12, 0, 8), cL=(3, -4), cR=(1, 5), hL=(.34, -.56, -.16), hR=(.04, .64, .04), eL=(.35, -1, .4), eR=(.5, -.3, -1), fR=(.03, -.36, .108), fL=(.045, .26, .33), kR=(.08, 1, -.2), kL=(.1, 1, .35), tR=(6, -56, 46), tL=(4, 6, 0), **_F)
POSES['run_fly'] = P(root=(0, 0, .026), pel=(22, 1, 18), wst=(4, 0, -12), cst=(0, 0, -22), nck=(-10, 0, 8), hed=(-10, 0, 8), cL=(3, -5), cR=(1, 6), hL=(.38, -.62, -.06), hR=(.02, .66, .12), eL=(.35, -1, .5), eR=(.5, -.2, -1), fR=(.05, -.46, .26), fL=(.045, .34, .22), kR=(.1, 1, -.5), kL=(.1, 1, .3), tR=(4, -78, 0), tL=(4, 10, 0), **_F)
# ---- walk (right contact, passing): weight; wrath = walk with clenched fists
POSES['walk_contact'] = P(root=(0, 0, -.012), pel=(2, -1, -7), wst=(0, 1, 4), cst=(-2, 1, 8), nck=(4, 0, -3), hed=(0, 0, -2), hL=(.20, .34, -.90), hR=(.24, -.22, -.93), fR=(.040, .155, .052), fL=(.040, -.165, .080), kR=(.10, 1, 0), kL=(.10, 1, 0), tR=(8, 16, 0), tL=(8, -34, 30))
POSES['walk_down'] = P('walk_contact', root=(.012, 0, -.030), hL=(.22, .24, -.93), hR=(.24, -.14, -.95), fR=(.040, .09, .045), fL=(.040, -.20, .110), kL=(.1, 1, -.1), tR=(8, 0, 0), tL=(8, -54, 40))
POSES['walk_pass'] = P(root=(.012, 0, .004), pel=(2, 3, 0), wst=(0, -2, 0), cst=(-2, -2, 0), nck=(4, 0, 0), hed=(0, 0, 0), hL=(.22, .08, -.95), hR=(.22, .04, -.95), fR=(.030, .0, .045), fL=(.040, .035, .110), kR=(.10, 1, 0), kL=(.12, 1, .1), tR=(8, 0, 0), tL=(8, -28, 0))
POSES['walk_up'] = P('walk_pass', root=(.008, 0, .012), hL=(.22, -.10, -.94), hR=(.22, .22, -.93), fR=(.030, -.09, .052), fL=(.040, .14, .090), tR=(8, -14, 10), tL=(8, 10, 0))


def cycle(keys, u):
    n = len(keys); x = (u % 1.) * n; i = int(x); return mix_pose(keys[i % n], keys[(i + 1) % n], x - i)


def walk(u, wrath=0.):
    """walk cycle, u in strides (1 = two steps), u=0 right heel contact.  wrath 0..1 clenches, lowers the head, widens the arms"""
    k = [POSES[n] for n in ('walk_contact', 'walk_down', 'walk_pass', 'walk_up')]; p = cycle(k + [mirror(q) for q in k], u)
    if wrath > 0:
        p = dict(p); w = wrath
        p = add_pose(p, pel=(3 * w, 0, 0), cst=(-4 * w, 0, 0), nck=(6 * w, 0, 0), hed=(-2 * w, 0, 0), cL=(2 * w, 0), cR=(2 * w, 0)); p['hL'] = np.asarray(p['hL'], float) * (1, .8, .97) + (.12 * w, .02, .03); p['hR'] = np.asarray(p['hR'], float) * (1, .8, .97) + (.12 * w, .02, .03); p['gL'] = p['gR'] = 'fist'
    return p


def sprint(u):
    """sprint cycle, u in strides (1 = two steps).  u=0 right contact"""
    k = [POSES[n] for n in ('run_contact', 'run_down', 'run_push', 'run_fly')]; return cycle(k + [mirror(p) for p in k], u)


def ease(t, e):
    t = float(np.clip(t, 0, 1))
    if e == 'hold': return 0.
    if e == 'lin': return t
    if e == 'io': return t * t * (3 - 2 * t)
    if e == 'out': return 1 - (1 - t) ** 2.4
    if e == 'snap': return 1 - (1 - t) ** 4.
    if e == 'in': return t ** 2.4
    if e == 'over': return 1 + 2.4 * (t - 1) ** 3 + 1.4 * (t - 1) ** 2                    # overshoots ~8 % and settles
    if e == 'spring': return 1 - math.exp(-5.2 * t) * math.cos(9.5 * t)
    return t


def on(f, n=2):
    """hold drawings: on(f, 2) = on twos"""
    return (int(f) // n) * n


def seq(f, keys):
    """held / eased key poses: [(frame, pose or name, ease)] ease = how we LEAVE the key: hold lin in out io snap over spring"""
    g = lambda p: POSES[p] if isinstance(p, str) else p
    if f <= keys[0][0]: return g(keys[0][1])
    for (a, pa, *ea), (b, pb, *_x) in zip(keys[:-1], keys[1:]):
        if f < b: return mix_pose(g(pa), g(pb), ease((f - a) / float(b - a), ea[0] if ea else 'io'))
    return g(keys[-1][1])


def track(f, keys):
    """numbers along keys [(frame, value or vector, ease)]"""
    if f <= keys[0][0]: return np.asarray(keys[0][1], float)
    for (a, va, *ea), (b, vb, *_x) in zip(keys[:-1], keys[1:]):
        if f < b:
            t = ease((f - a) / float(b - a), ea[0] if ea else 'io'); return np.asarray(va, float) * (1 - t) + np.asarray(vb, float) * t
    return np.asarray(keys[-1][1], float)


def path(f, keys):
    """smooth path (Catmull-Rom, no stops) through [(frame, xyz)]"""
    t = np.array([k[0] for k in keys], float); V = np.array([k[1] for k in keys], float)
    if f <= t[0]: return V[0]
    if f >= t[-1]: return V[-1]
    i = int(np.searchsorted(t, f, side='right') - 1); h = t[i + 1] - t[i]; u = (f - t[i]) / h
    m = lambda j: (V[min(j + 1, len(V) - 1)] - V[max(j - 1, 0)]) / (t[min(j + 1, len(V) - 1)] - t[max(j - 1, 0)])
    return (2 * u ** 3 - 3 * u ** 2 + 1) * V[i] + (u ** 3 - 2 * u ** 2 + u) * h * m(i) + (-2 * u ** 3 + 3 * u ** 2) * V[i + 1] + (u ** 3 - u ** 2) * h * m(i + 1)


def idle(p, f, amt=1., seed=0):
    """breath and weight for a held pose: call every frame (or on twos)"""
    p = POSES[p] if isinstance(p, str) else p; s = math.sin(f / 24. * 2 * math.pi / 2.9 + seed); t = math.sin(f / 24. * 2 * math.pi / 5.3 + 1.7 * seed) * amt; s *= amt
    return add_pose(p, cst=(-1.3 * s, .3 * t, 0), wst=(-.5 * s, 0, .3 * t), cL=(.45 * s, 0), cR=(.45 * s, 0), hed=(.8 * s, .4 * t, .5 * t), root=(.002 * t, 0, .0012 * s))


def aim(dirv, up=(0, 0, 1)):
    """heading, pitch, roll for Fig.place / state so that the body axis (feet to head) runs along dirv with the belly toward -up (add the pose's own body pitch: use a pose with body=(0,0,0))"""
    d = unit(dirv); h = math.atan2(-d[0], d[1]) if abs(d[0]) + abs(d[1]) > 1e-6 else 0.; p = math.atan2(math.hypot(d[0], d[1]), d[2]); return h, p, 0.
