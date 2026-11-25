"""HERO3: the fighters. A posed 3-D build (muscles from origin to insertion, cloth shells, plates) re-drawn as an inked anime cel from ANY camera, 100..1600 px tall.

    from db.hero3 import *
    F = Fig('goku').pose('punch_hit').place(0, 6, 0, heading=math.pi)        # metres; heading 0 looks along +Y, pi faces a camera at -Y, -pi/2 looks +X (screen RIGHT for a camera at -Y)
    L = F.draw(cam, f, light=(-.5, .75, .4), pal='night', rim=dict(dir=(.9, .3, -.4), col='#c784ff', thr=.45, amt=.9), expr=dict(expr='shout'))
    img = comp.over(img, L)                                                  # premultiplied (H, W, 4).  draw(..., layer=False) -> (roi, box)

characters   Fig(char, variant):  goku clean|torn|bare|waist|ui|ssj|ssjclean   vegeta clean|shattered|nogear|ssj|ssjshattered   frieza   piccolo clean|nocape|torn   gohan clean|nohat   kid clean|bare
             Fig(char, spec=dict(..)) overrides db/h3_body.SPEC,  scale=, body=dict(H=..) change proportions
state        .pose(name or dict, hR=..)   .place(x, y, z, heading, pitch, roll)   .fly(pelvis_xyz, dir, roll)   
             .look(xyz or None, w=1) the head turns from the chest as a neck can (up to ~84 deg, eyes kept level on upright figures); in anim: look=, lookw=   .reach(R=dict(p=xyz, part='fist', w=1), fL=...)   .blow(vx, vy, vz) wind
animation    F.anim(lambda f: dict(pose=seq(f, KEYS), x=, y=, z=, heading=, pitch=, roll=, pivot='ground'|'pelvis', look=, reach=, wind=, hand=dict(L=1.2)))     F.at(f).draw(cam, f)
             cloth, belt ends, cape, hair, tail lag behind the motion by themselves (pure function of f).   on(f, 2) = on twos
keys         seq(f, [(0, 'guard', 'in'), (6, 'punch_wind', 'out'), (9, 'punch_hit', 'hold'), ...])  eases hold lin in out io snap over spring;  track(f, keys) numbers / vectors;  path(f, keys) smooth 3-D path
             sprint(u) walk(u, wrath=0..1) cycles (u strides), idle(pose, f), mix_pose add_pose mirror P(base, **changes) at('R', x, y, z) hand target from the chest (H fractions)
poses        sorted(POSES): stand contra guard stance_goku stance_vegeta crouch launch lunge punch_wind punch_hit punch_follow jab uppercut gut_punch elbow backfist palm shove kick_wind kick kick_side kick_spin knee axe_up
             block_high block_cross block_back duck lean_back slip | hit_gut hit_head hit_up hit_spin thrown swung tumble fall stagger | land land_settle slide brace_back kneel kneel_up knees knees_fist cradle carried lie lotus lookup
             power power_in power_out kame_wind kame_fire kame_hold ff_wide ff_fire beam_one finger_up finger_aim flick part conduct moon_up hands_up arms_crossed fists_down | hover hover_guard fly fly_punch fly_both fly_bank dive rise dive_punch leap tuck air_kick hang_back
contact      hit(A, 'R', B, 'stomach')  grab(A, 'L', B, 'wristR')  hold(A, 'R', B, 'wristR') (B hangs from A's hand: swings)  touch(A, 'R', B, 'chest')  clash(A, 'R', B, 'R')  brace(A, B)  carry(A, B): solve the IK so that the contact is exact (call after both are posed; skel().spot(name) gives any point)
             reach dict: p=world point, part=wrist|fist|palm|finger|back|edge|cup, w=0..1, palm=world normal the palm faces, fingers=world dir, pole=, g=grip, stretch=1.14, body=.9 (how far the body follows)
light        direction TO the light in camera space (x right, y up, z toward us); key= terminator bias (.1; higher = more shadow)
rim          dict(dir, col, thr=.42, amt=.85, face=.78) or a list of them: hard coloured planes from the form itself
pal          preset name (LIGHTS: day dawn lamp night moon dark fire beam gold rain storm space ki grey) or dict(base='night', mul, shmul, litmul, add, ink, mats={'gi': (light, base, shadow, deep)})
heads        default: the head lane's drawing when the head is big enough (F.lane_min px per unit, 15), else the built-in small head.  head_fn='own' | 'lane' | fn(cel, H, info) | fn(H, info) -> layer
             info = x y s yaw pitch roll R roi f expr light hair wind wet dmg fig pal; expr=dict(expr=, hair=, light=, wet=, dmg=, ...) is passed through to the lane
after draw   F.info: px box hands fists palms wrists elbows shoulders feet soles knees pelvis chest stomach head tail (px), hands3 (world), skel;  F.last() the figure's own layer;  F.mask('handR', 'farmR') part mask;  F.depth() z buffer
two figures  compose([(FA, kwA), (FB, kwB)], cam, f) -> one layer with correct occlusion per pixel
smears       F.smear(cam, f0, f1, 'armR') swept limb (draw under the new pose);  ghosts(F, cam, states, ...) multiples;  after_limb(...);  tails(layer, (dx, dy)) dry-brush speed tails;  stretch(layer, p, dir, k) squash of a finished layer
frieza tail  F.tail = fn(sk, f) -> (n, 3) world points from the tailbone, or tail_curve(sk, f, kind='sway'|'whip'|'wrap'|'block', ...)
qa           tools/h3_qa.py h3_strip.py (a choreographed exchange to copy from) h3_contact.py (every helper) h3_poses.py h3_small.py h3_hero.py -> qa/body/*.jpg
speed        about .5-1.2 s; draw(lod=0|1|2) forces detail, lines=0..1.5 muscle line weight, shadow=False skips cast shadows
"""
import numpy as np, math, cv2
from . import W, H
from .paint import rgb, f32
from .ink import Cel, spl, resample, paste
from . import kit
from .h3_pose import *
from .h3_pose import POSES, Skel, CHAR_BODY, P, mix_pose, unit, HOLD, at, on, seq, track, path, aim, idle, walk, sprint, cycle, mirror, add_pose, ease, rot_to, euler
from .h3_body import Body, MATS, SPEC, VARIANT
from .h3_draw import Draw, Pal, LIGHTS, INK


class Fig:
    def __init__(self, char, variant='clean', spec=None, seed=0, scale=1.0, body=None):
        self.char = char; self.spec = dict(VARIANT.get(char, {}).get(variant, {})); self.spec.update(spec or {}); self.variant = variant
        self.seed = seed; self.scale = scale; self.body = dict(CHAR_BODY[char]); self.body.update(body or {})
        self._pose = P(); self.pos = np.zeros(3); self.heading = 0.; self.pitch = 0.; self.roll = 0.; self.pivot = 'ground'; self._look = None; self._lookw = 1.; self._reach = {}
        self.fn = None; self.wind = (0., 0., 0.); self.info = {}; self._fx = {}; self.tail = None; self.lane_min = 15.; self.light_name = 'day'; self.hand = {}; self._last = None; self.head_info = None

    # ---- state
    def pose(self, p=None, **kw):
        q = POSES[p] if isinstance(p, str) else (p if p is not None else P())
        if kw: q = dict(q); q.update(kw)
        self._pose = q; return self

    def place(self, x=0., y=0., z=0., heading=0., pitch=0., roll=0., pivot='ground'):
        self.pos = np.array([x, y, z], float); self.heading, self.pitch, self.roll, self.pivot = heading, pitch, roll, pivot; return self

    def fly(self, pos, dirv=None, roll=0.):
        """pelvis at pos; dirv = direction of flight: turns the figure (heading) and, for poses without their own body pitch, leans it"""
        self.pos = np.asarray(pos, float); self.pivot = 'pelvis'; self.roll = roll
        if dirv is not None:
            d = unit(dirv); self.heading = math.atan2(-d[0], d[1]) if abs(d[0]) + abs(d[1]) > 1e-6 else self.heading; self.pitch = -math.asin(float(np.clip(d[2], -1, 1)))
        return self

    def look(self, p, w=1.): self._look = None if p is None else np.asarray(p, float); self._lookw = float(w); return self
    def reach(self, **k): self._reach = {a: (b if isinstance(b, dict) else dict(p=b)) for a, b in k.items() if b is not None}; return self
    def blow(self, vx=0., vy=0., vz=0.): self.wind = (vx, vy, vz); return self
    def anim(self, fn): self.fn = fn; return self

    def state(self, f):
        d = dict(pose=self._pose, x=self.pos[0], y=self.pos[1], z=self.pos[2], heading=self.heading, pitch=self.pitch, roll=self.roll, pivot=self.pivot, look=self._look, lookw=self._lookw, reach=self._reach, wind=self.wind, hand=self.hand)
        if self.fn is None: return d
        q = self.fn(f); d['reach'] = {}; d.update(q)            # a frame is a pure function of f: what hit() / reach() did to an earlier frame does not stay
        if 'pos' in q: d['x'], d['y'], d['z'] = q['pos']
        if isinstance(d['pose'], str): d['pose'] = POSES[d['pose']]
        return d

    def set(self, d):
        self._pose = POSES[d['pose']] if isinstance(d['pose'], str) else d['pose']; self.pos = np.array([d['x'], d['y'], d['z']], float); self.heading, self.pitch, self.roll, self.pivot = d['heading'], d['pitch'], d['roll'], d['pivot']
        self._look = d['look']; self._lookw = float(d.get('lookw', 1.)); self._reach = {a: (b if isinstance(b, dict) else dict(p=b)) for a, b in (d['reach'] or {}).items() if b is not None}; self.wind = d['wind']; self.hand = d['hand']; return self

    def at(self, f): return self.set(self.state(f)) if self.fn is not None else self

    def skel_of(self, d):
        rc = {a: (b if isinstance(b, dict) else dict(p=b)) for a, b in (d['reach'] or {}).items() if b is not None}
        return Skel(self.body, d['pose'], (d['x'], d['y'], d['z']), d['heading'], d['pitch'], d['roll'], self.scale, d['pivot'], rc, d['look'], d.get('lookw', 1.))

    def skel(self, f=None):
        return self.skel_of(self.state(f) if f is not None else dict(pose=self._pose, x=self.pos[0], y=self.pos[1], z=self.pos[2], heading=self.heading, pitch=self.pitch, roll=self.roll, pivot=self.pivot, look=self._look, lookw=self._lookw, reach=self._reach))

    def spot(self, name, f=None): return self.skel(f).spot(name)

    # ---- secondary motion: where the anchors were a few frames ago (pure function of f)
    def _lag(self, sk0, hist, key):
        out = np.zeros(3); wt = 0; p0 = key(sk0)
        for k, sk in enumerate(hist, 1):
            w = .66 ** k; out += w * (key(sk) - p0) / k; wt += w
        return out / wt if wt else out

    def make_fx(self, f, sk0):
        fx = dict(self._fx); k = self.scale; wnd = np.asarray(self.wind, float); r = math.sin(f * .55 + self.seed) * .5 + math.sin(f * .9 + 1.7 * self.seed) * .3; gust = wnd * (1 + .35 * r)
        hist = [self.skel(f - q) for q in (1, 2, 3, 4)] if self.fn is not None else []
        cl = lambda v, m: v * min(1., m / (np.linalg.norm(v) + 1e-9))
        pel = self._lag(sk0, hist, lambda s: s.pel); ch = self._lag(sk0, hist, lambda s: s.chest)
        fx.setdefault('cloth', cl(pel * 5.0, 1.0) + gust * .55); fx.setdefault('belt', cl(pel * 7.0, 1.6) + gust * .9); fx.setdefault('cape', cl(ch * 5.0, 1.6) + gust * .9)
        fx.setdefault('hair', cl(self._lag(sk0, hist, lambda s: s.head) * 4.0, .8) + gust * .5); fx.setdefault('down', unit(np.array([0, 0, -1.]) + gust * .30)); fx['f'] = f
        for s, v in (self.hand or {}).items(): fx['handscale' + s] = v
        if self.tail is not None: fx['tail'] = self.tail(sk0, f) if callable(self.tail) else self.tail
        elif self.spec.get('tail', SPEC[self.char].get('tail')) and hist: fx['tail'] = tail_lag(self, sk0, hist, f)
        return fx

    def smear(self, cam, f0, f1, limb='armR', **k):
        a = self.state(f0) if not isinstance(f0, dict) else f0; b = self.state(f1) if not isinstance(f1, dict) else f1
        from .h3_smear import limb_smear
        return limb_smear(self, cam, a, b, limb, **k)

    # ---- draw
    def draw(self, cam, f=0, light=(-.55, .70, .45), pal=None, head_fn=None, expr=None, weight=1.0, rim=None, key=.10, lod=None, head=True, seed=None, layer=True, ink_only=False, lines=1.0, shadow=True):
        self.light_name = pal if isinstance(pal, str) else (pal.get('base', 'day') if isinstance(pal, dict) else 'day'); pal = pal if isinstance(pal, Pal) else Pal(pal)
        sk = self.skel(); self.sk = sk; fx = self.make_fx(f, sk); px = float(cam.scale(sk.pel)) * sk.H
        if not np.isfinite(px) or cam.cam(sk.pel)[2] < .1: px = 900.
        e_ = cam.eye - sk.chest; fx['eyedir'] = e_ / (np.linalg.norm(e_) + 1e-9)
        B = Body(self.char, sk, fx, px if lod is None else (90, 250, 700)[lod], self.spec, self.seed); self.B = B; self._last = None
        D = Draw(self, cam, B, px, pal, light, key, weight, expr, head_fn, head, self.seed if seed is None else seed, fx, ink_only, rim, lines, shadow); out = D.run(); self._last = D; self._cam = cam
        if out is None: self._roi = (np.zeros((1, 1, 4), f32), (0, 0, 1, 1)); return np.zeros((cam.h, cam.w, 4), f32) if layer else self._roi
        img, box = out; self._roi = (img, box)
        if not layer: return img, box
        L = np.zeros((cam.h, cam.w, 4), f32); paste(L, img, box[0], box[1]); return L

    def last(self):
        """the figure alone, as drawn by the last draw() / compose(): full-frame premultiplied layer (for auras, ghosts, lights)"""
        L = np.zeros((self._cam.h, self._cam.w, 4), f32); img, b = self._roi; paste(L, img, b[0], b[1]); return L

    def mask(self, *labels):
        """full-frame float mask of drawn parts after draw(): label prefixes such as 'handR', 'farmR', 'uarm', 'foot', 'torso', 'gi', 'head', 'tail'"""
        D, cam = self._last, self._cam; out = np.zeros((cam.h, cam.w, 1), f32)
        if D is None or not hasattr(D, 'lab'): return out[..., 0]
        ids = [i for i, n in enumerate(D.lname) if any(n.startswith(q) for q in labels)]; m = np.isin(D.lab, ids).astype(f32); x0, y0, x1, y1 = D.bodybox
        paste(out, cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)[..., None], x0, y0); return out[..., 0]

    def depth(self, box=None):
        """z_cam of the last draw (0 = empty; hair takes the head's depth): full frame, or of a box (x0, y0, x1, y1)"""
        D, cam = self._last, self._cam; x0, y0, x1, y1 = box or (0, 0, cam.w, cam.h); out = np.zeros((y1 - y0, x1 - x0, 1), f32)
        if D is None or not hasattr(D, 'Z'): return out[..., 0]
        bx0, by0, bx1, by1 = D.bodybox; paste(out, cv2.resize(D.Z, (bx1 - bx0, by1 - by0), interpolation=cv2.INTER_NEAREST)[..., None], bx0 - x0, by0 - y0); Z = out[..., 0]
        img, b = self._roi; a = np.zeros(out.shape, f32); paste(a, img[..., 3:4], b[0] - x0, b[1] - y0); hole = (a[..., 0] > .02) & (Z <= 0)
        if hole.any():
            h = self.info.get('head'); zh = h['z'] if h else self.info['depth']; Z[hole] = self.info['depth']
            if h:
                yy, xx = np.nonzero(hole); r = np.hypot(xx + x0 - h['x'], yy + y0 - h['y']) < h['s'] * 3.2; Z[yy[r], xx[r]] = zh
        return Z


# ------------------------------------------------------------------ tails
def _tailbase(sk): return sk.pel + sk.Rp @ np.array([0, -.070 * sk.H / 1.76 * sk.b['mass'], -.005 * sk.H / 1.76])


def tail_curve(sk, f, kind='sway', amp=1., target=None, side=1., n=10):
    """Frieza's tail as world points.  sway: idle S;  whip: lashes toward target (world point);  block: rises in a shield curve on `side`;  wrap: coils round target (a world point), low: drags behind"""
    b = _tailbase(sk); back, up, rt = -sk.Rp[:, 1], sk.Rp[:, 2], sk.Rp[:, 0] * side; L = 1.30 * sk.H; u = np.linspace(0, 1, n + 1)[1:, None]; ph = f * 2 * math.pi / 40.
    if kind == 'whip' and target is not None:
        t = np.asarray(target, float); m = (b + t) / 2 + up * .25 * L * .5 + back * .10 * L; return (1 - u) ** 3 * b + 3 * (1 - u) ** 2 * u * (b + back * .22 * L) + 3 * (1 - u) * u * u * m + u ** 3 * t
    if kind == 'block':
        c = [b, b + back * .22 * L - up * .02 * L, b + back * .16 * L + rt * .30 * L + up * .16 * L, b + rt * .34 * L + up * .42 * L + sk.Rp[:, 1] * .06 * L, b + rt * .20 * L + up * .60 * L + sk.Rp[:, 1] * .16 * L]
    elif kind == 'wrap' and target is not None:
        t = np.asarray(target, float); a = u[:, 0] * 2 * math.pi * 1.5; r = .13 * sk.H; d = unit(t - b); e1 = unit(np.cross(d, up) + 1e-6); e2 = np.cross(d, e1); k = np.clip(u * 1.8, 0, 1)
        return b + (t - b) * k + (e1 * np.cos(a)[:, None] + e2 * np.sin(a)[:, None]) * r * k ** 2
    elif kind == 'low': c = [b, b + back * .25 * L - up * .16 * L, b + back * .50 * L - up * .34 * L + rt * .06 * L, b + back * .78 * L - up * .36 * L - rt * .06 * L, b + back * .98 * L - up * .28 * L]
    else:
        s1, s2 = math.sin(ph) * amp, math.sin(ph - .9) * amp
        c = [b, b + back * .16 * L - up * .10 * L, b + back * .32 * L - up * .25 * L + rt * .05 * L * s1, b + back * .50 * L - up * .33 * L + rt * .12 * L * s2, b + back * .67 * L - up * .25 * L + rt * .18 * L * s2, b + back * .74 * L - up * .08 * L + rt * (.15 + .05 * s1) * L, b + back * .66 * L + up * .06 * L + rt * (.09 + .07 * s1) * L, b + back * .58 * L + up * .04 * L + rt * (.06 + .06 * s1) * L]
    c = np.array(c); t = np.linspace(0, 1, len(c)); from .h3_body import _cr3
    return _cr3(c, t, u[:, 0])


def tail_lag(F, sk0, hist, f):
    """default tail of a moving figure: every joint remembers where the body was a little earlier"""
    cur = tail_curve(sk0, f); n = len(cur); old = [tail_curve(s, f) for s in hist]; out = cur.copy()
    for i in range(n):
        d = (i + 1) / n * 3.6; k = min(int(d), len(old) - 1); a = cur if k == 0 else old[k - 1]; b = old[k]; out[i] = a[i] * (1 - (d - k if d - k < 1 else 1)) + b[i] * min(1, d - k)
    v = np.linalg.norm(np.diff(np.vstack([_tailbase(sk0), out]), axis=0), axis=1)
    # keep the length: relax links
    L = 1.30 * sk0.H / n; p = _tailbase(sk0)
    for i in range(n): d = out[i] - p; out[i] = p + d / (np.linalg.norm(d) + 1e-9) * L; p = out[i]
    return out


# ------------------------------------------------------------------ contact between two figures
def hit(A, s, B, spot='stomach', part='fist', w=1., push=.0, dirv=None, **k):
    """A's hand (side s) exactly on B's spot.  push = metres the fist sinks in.  returns the world point"""
    p = B.spot(spot) if isinstance(spot, str) else np.asarray(spot, float); a = A.skel().sh[s]; d = unit(p - a) if dirv is None else unit(dirv); A._reach = dict(A._reach); A._reach[s] = dict(p=p + d * push, part=part, w=w, **k); return p


def touch(A, s, B, spot='chest', w=1., **k):
    """flat palm on B's spot, the palm facing the body"""
    sk = B.skel(); p = sk.spot(spot); c = dict(stomach=sk.waist, solar=sk.chest, chest=sk.chest, back=sk.chest, face=sk.head, forehead=sk.head).get(spot, sk.chest); n = unit(c - p) if spot not in ('back',) else sk.Rc[:, 1]
    if spot in ('chest', 'solar', 'stomach'): n = -sk.Rc[:, 1]
    up = np.array([0, 0, 1.]); A._reach = dict(A._reach); A._reach[s] = dict(p=p, part='palm', w=w, palm=n, fingers=up - n * (up @ n) + 1e-6, g='splay', **k); return p


def grab(A, s, B, spot='wristR', w=1., **k):
    """A's hand closes round B's wrist / ankle / forearm"""
    sk = B.skel(); p = sk.spot(spot); q = spot[-1]; ax = unit(sk.wr[q] - sk.el[q]) if spot.startswith(('wrist', 'forearm')) else unit(sk.an[q] - sk.kn[q]); a = A.skel().sh[s]; d = unit(p - a); f_ = unit(np.cross(ax, d)); n = d - ax * (d @ ax); n = unit(n)
    f_ = unit(np.cross(ax, n)); f_ = f_ if f_[2] > -.2 else -f_; A._reach = dict(A._reach); A._reach[s] = dict(p=p + n * .0, part='cup', w=w, palm=n, fingers=f_, g='hold', **k); return p


def hold(A, s, B, spot='wristR', f=None):
    """B hangs from A's hand: B is moved so that its spot lies in A's closed hand (swings and throws: call after both are posed)"""
    d = A.skel().spot('cup' + s) - B.skel().spot(spot); B.pos = np.asarray(B.pos, float) + d; A._pose = dict(A._pose); A._pose['g' + s] = 'hold'; return d


def clash(A, sa, B, sb, p=None, w=1., **k):
    """two fists meet at one point (default: midway)"""
    if p is None: p = (A.skel().spot('fist' + sa) + B.skel().spot('fist' + sb)) / 2
    p = np.asarray(p, float); A._reach = dict(A._reach); B._reach = dict(B._reach); A._reach[sa] = dict(p=p, part='fist', w=w, **k); B._reach[sb] = dict(p=p, part='fist', w=w, **k); return p


def brace(A, B, s='R', spot='chest'):
    """A catches B's fist in the palm"""
    sk = B.skel(); p = sk.spot('fist' + s); d = unit(sk.wr[s] - sk.el[s]); o = 'L' if s == 'R' else 'R'; A._reach = dict(A._reach); A._reach[o] = dict(p=p, part='palm', palm=-d, fingers=np.array([0, 0, 1.]) - d * d[2], g='hold'); return p


def carry(A, B):
    """A (pose cradle / catch) holds B (pose carried): B's shoulders on A's left arm, knees on A's right"""
    sa = A.skel(); c = sa.chest + sa.Rc[:, 1] * .30 * sa.H / 1.76 - sa.Rc[:, 2] * .10 * sa.H; B.place(*c, heading=A.heading + math.pi / 2, pivot='pelvis'); sb = B.skel(); A._reach = dict(A._reach)
    A._reach['L'] = dict(p=(sb.chest + sb.neck) / 2 - sb.Rc[:, 1] * .12 * sb.H / 1.76, part='palm', palm=sb.Rc[:, 1], g='open', body=.2); A._reach['R'] = dict(p=(sb.kn['L'] + sb.kn['R']) / 2 - np.array([0, 0, .07]), part='palm', palm=np.array([0, 0, 1.]), g='open', body=.2); return c


def compose(items, cam, f=0, layer=True):
    """[(Fig, dict of draw keywords), ...] drawn into one premultiplied layer with per-pixel occlusion between the figures.  After it every Fig keeps its own info / mask() / depth()"""
    Ls = []
    for F, k in items:
        img, box = F.draw(cam, f, layer=False, **k)
        if img.shape[0] > 1: Ls.append((img, box, F))
    out = np.zeros((cam.h, cam.w, 4), f32); Zo = np.full((cam.h, cam.w), 1e9, f32); Ls.sort(key=lambda q: -q[2].info.get('depth', 1.))
    for img, box, F in Ls:
        x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(cam.w, box[2]), min(cam.h, box[3])
        if x1 <= x0 or y1 <= y0: continue
        L = img[y0 - box[1]:y1 - box[1], x0 - box[0]:x1 - box[0]]; Z = F.depth((x0, y0, x1, y1)); o = out[y0:y1, x0:x1]; zo = Zo[y0:y1, x0:x1]; z = np.where(Z > 0, Z, 1e8)
        front = (z < zo + .01) | (o[..., 3] < .01) | (zo > 1e8)
        if (o[..., 3] > .01).any():
            m = cv2.GaussianBlur(front.astype(f32), (0, 0), .8)[..., None]; a = L * m; new = a + o * (1 - a[..., 3:4]); out[y0:y1, x0:x1] = new + L * (1 - m) * (1 - new[..., 3:4])
        else: out[y0:y1, x0:x1] = L
        Zo[y0:y1, x0:x1] = np.where((L[..., 3] > .5) & front, z, zo)
    return out


def ghosts(F, cam, states, pal=None, alphas=None, light=(-.55, .7, .45), tint=None, lod=1, **k):
    """multiples: the figure drawn at several states (frames or state dicts), far to near, with falling opacity -> one layer"""
    keep = F.state(0) if F.fn is None else None; sv = (F._pose, F.pos, F.heading, F.pitch, F.roll, F.pivot, F._look, F._reach, F.wind, F.hand); out = np.zeros((cam.h, cam.w, 4), f32); n = len(states); alphas = alphas or [.25 + .5 * i / max(1, n - 1) for i in range(n)]
    for a, st in zip(alphas, states):
        F.set(F.state(st) if not isinstance(st, dict) else st); L = F.draw(cam, st if not isinstance(st, dict) else 0, light=light, pal=pal, lod=lod, shadow=False, **k)
        if tint is not None: c = rgb(tint) if isinstance(tint, str) else tint; L = L.copy(); L[..., :3] = L[..., :3] * .45 + L[..., 3:4] * c * .55
        L = L * a; out = out * (1 - L[..., 3:4]) + L
    (F._pose, F.pos, F.heading, F.pitch, F.roll, F.pivot, F._look, F._reach, F.wind, F.hand) = sv; return out


def stretch(layer, p, d, k, width=None):
    """smear a finished layer: stretched by k along screen direction d about the point p, squashed across (volume kept)"""
    d = np.asarray(d, float); d = d / (np.hypot(*d) + 1e-9); n = np.array([-d[1], d[0]]); M = np.outer(d, d) * k + np.outer(n, n) / math.sqrt(k); p = np.asarray(p, float); A = np.c_[M, p - M @ p]
    return cv2.warpAffine(layer, A, (layer.shape[1], layer.shape[0]), flags=cv2.INTER_LINEAR)


from .h3_smear import limb_smear, after_limb, tails                # noqa: E402
