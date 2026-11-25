"""g4 LIKE WATER.  Seven evasions, one piano note each (0, 10, .. 60): nothing is wasted, everything misses by centimetres; where he was, an ink wash stays a moment.
The camera goes once round them.  70 an open palm is laid on Frieza's chest; space ripples like water; 73 he is not there any more, the palm stays."""
import numpy as np, cv2, math
from db import W, H as HH, comp, kit, sky, ki
from db.cam import Camera3D
from db.ink import Cel, spl
from db.paint import rgb, f32
from db.hero3 import *
R_, L_ = -math.pi / 2, math.pi / 2
NOTES = (0, 10, 20, 30, 40, 50, 60)
S = lambda u: float(np.clip(u, 0, 1)) ** 2 * (3 - 2 * float(np.clip(u, 0, 1)))

A_ = dict(air=1, fL=(.075, .02, .055), fR=(.070, -.06, .11), kL=(.25, 1, 0), kR=(.25, 1, .1), tL=(12, -32, -8), tR=(12, -42, -10))
REST = P('contra', **dict(A_, root=(0, 0, 0), hL=(.26, .08, -.92), hR=(.28, .02, -.92), gL='relax', gR='relax', cst=(-3, 0, 4), nck=(6, 0, 0), hed=(0, 0, 0)))
K = [(-8, REST, 'io'),
     (-4, REST, 'io'), (1, P(REST, root=(-.012, -.01, 0), wst=(0, -4, 0), cst=(-3, -8, 4), nck=(6, -16, -4), hed=(2, -14, -4)), 'io'),
     (5, P(REST, root=(-.008, -.01, .004), cst=(-5, -5, 6), nck=(4, -10, -2), hed=(0, -8, -2)), 'io'),
     (11, P('lean_back', **dict(A_, root=(.02, -.07, -.02), pel=(-30, 4, 10), wst=(-10, 3, 4), cst=(-16, 3, 6), nck=(10, 0, -6), hed=(10, 0, -6), hL=(.50, -.10, -.66), hR=(.54, -.26, -.58), fL=(.07, .22, .12), fR=(.07, .10, .06))), 'io'),
     (15, P('lean_back', **dict(A_, root=(.01, -.05, -.01), pel=(-20, 6, 10), wst=(-6, 4, 4), cst=(-10, 4, 8), hL=(.46, -.02, -.74), hR=(.48, -.18, -.70), fL=(.07, .16, .10), fR=(.07, .04, .08))), 'io'),
     (20, P('slip', **dict(A_, root=(.05, .0, -.02), pel=(6, 18, 24), wst=(4, 10, 8), cst=(4, 12, 12), nck=(4, -8, -18), hed=(2, -8, -16), hL=(.34, .24, -.70), hR=(.42, -.12, -.72), gL='soft', gR='relax', fL=(.10, -.06, .07), fR=(.10, .12, .12))), 'io'),
     (25, P('slip', **dict(A_, root=(.04, .02, -.01), pel=(4, 12, 30), wst=(2, 8, 10), cst=(2, 8, 14), nck=(6, -4, -22), hed=(2, -4, -18), hL=(.30, .16, -.80), hR=(.40, -.10, -.78), fL=(.10, -.04, .08), fR=(.09, .10, .10))), 'io'),
     (30, P(REST, **dict(A_, root=(0, 0, .04), pel=(12, 0, 8), wst=(8, 0, 4), cst=(4, 0, 4), nck=(4, 0, -8), hed=(4, 0, -6), hL=(.52, .12, -.66), hR=(.56, .06, -.62), gL='open', gR='open', fL=(.10, .16, .34), fR=(.08, .06, .28), kL=(.4, 1, .5), kR=(.3, 1, .5), tL=(12, -26, -8), tR=(12, -30, -8))), 'io'),
     (34, P(REST, **dict(A_, root=(0, 0, .02), pel=(8, 0, 10), hL=(.48, .10, -.72), hR=(.52, .02, -.70), fL=(.09, .10, .22), fR=(.08, .02, .18), kL=(.35, 1, .3), kR=(.3, 1, .3))), 'io'),
     (40, P('duck', **dict(A_, root=(-.01, .02, -.06), pel=(30, -4, -18), wst=(18, -4, -8), cst=(18, -4, -10), nck=(-12, 0, 14), hed=(-16, 0, 14), hL=(.24, .40, -.40), hR=(.40, .06, -.66), gL='soft', gR='relax', fL=(.10, .20, .20), fR=(.12, -.14, .16), kL=(.4, 1, .3), kR=(.6, 1, .2))), 'io'),
     (45, P('duck', **dict(A_, root=(0, .02, -.03), pel=(20, 0, -10), wst=(12, 0, -4), cst=(10, 0, -4), nck=(-6, 0, 10), hed=(-8, 0, 8), hL=(.28, .26, -.62), hR=(.38, .02, -.76), gL='relax', gR='relax', fL=(.09, .12, .14), fR=(.10, -.10, .12))), 'io'),
     (50, P(REST, **dict(A_, root=(0, 0, .01), body=(0, 0, -62), pel=(-4, 0, -10), wst=(-2, 0, -8), cst=(-4, 0, -10), nck=(6, 0, 8), hed=(4, 0, 8), hL=(.06, .10, -.95), hR=(.05, .10, -.95), eL=(.2, -1, 0), eR=(.2, -1, 0), gL='flat', gR='flat', fL=(.02, .02, .05), fR=(.02, -.02, .07))), 'io'),
     (55, P(REST, **dict(A_, root=(0, 0, .0), body=(0, 0, -38), cst=(-2, 0, -6), nck=(6, 0, 8), hed=(2, 0, 8), hL=(.16, .06, -.95), hR=(.18, .0, -.95))), 'io'),
     (60, mirror(P('slip', **dict(A_, root=(-.05, .06, -.05), pel=(16, -16, -20), wst=(8, -8, -8), cst=(8, -8, -10), nck=(-4, 10, 18), hed=(-6, 8, 16), hL=(.36, .30, -.60), hR=(.28, -.22, -.60), gL='soft', gR='open', eR=(.5, -1, .1), fL=(.09, .18, .12), fR=(.10, -.16, .10)))), 'io'),
     (70, mirror(P('palm', **dict(A_, root=(-.01, .05, -.02), pel=(8, -2, 24), wst=(4, -2, 10), cst=(2, -2, 14), nck=(2, 0, -24), hed=(0, 0, -22), hL=(.32, -.06, -.86), gL='relax', fL=(.08, .18, .06), fR=(.07, -.20, .08), wR=(-85, -60, 0)))), 'hold')]
GP = [(-8, (0, 0, 0), 'io'), (9, (0, 0, .0), 'io'), (14, (-.08, 0, .02), 'io'), (20, (-.04, -.10, .02), 'io'), (27, (-.02, -.12, .02), 'io'), (30, (-.02, -.06, .08), 'io'), (36, (-.02, -.02, .05), 'io'), (41, (-.03, 0, -.02), 'io'), (50, (0, .0, 0), 'io'), (57, (.02, .02, 0), 'in'), (61, (.20, .10, -.02), 'io'), (70, (.16, .04, 0), 'hold')]
# Frieza: far, in, punch, tail, kick, far, in, backfist, opened
ZK = [(-8, 'finger_aim', 'hold'), (12, 'finger_aim', 'io'), (15, 'fly_punch', 'io'), (18, 'punch_wind', 'in'), (20, 'punch_hit', 'hold'), (22, 'punch_hit', 'out'), (25, 'punch_follow', 'io'), (29, P('duck', root=(0, 0, -.14)), 'io'), (33, P('duck', root=(0, 0, -.12)), 'io'), (37, 'kick_wind', 'in'), (40, 'kick', 'hold'), (42, 'kick', 'io'), (46, 'kick_spin', 'io'), (48, 'finger_aim', 'hold'), (54, 'finger_aim', 'io'),
      (57, 'fly_punch', 'io'), (58, P('backfist', pel=(6, 0, -20), wst=(4, 0, -10), cst=(2, 0, -24), hR=(-.2, .4, .2)), 'in'), (60, 'backfist', 'hold'), (62, 'backfist', 'out'), (65, P('stagger', hR=(.62, -.34, -.46), hL=(.50, -.2, -.6), cst=(-6, 4, 10), wst=(-4, 0, 6), pel=(-6, 0, 10), nck=(10, 0, -10), hed=(8, 0, -8)), 'io'), (72, P('stagger', hR=(.54, -.30, -.66), hL=(.56, -.1, -.6), cst=(-2, 2, 4), wst=(0, 0, 2), pel=(-4, 0, 4), nck=(14, 0, -4), hed=(10, 0, -2)), 'hold')]
ZP = [(-8, (11., 2.6, .9), 'hold'), (12, (11., 2.6, .9), 'in'), (18, (1.30, -.10, 0), 'in'), (20, (.82, -.02, .02), 'hold'), (22, (.82, -.02, .02), 'io'), (25, (.62, .34, .0), 'io'), (30, (1.06, .12, -.04), 'io'), (36, (1.20, -.10, .05), 'io'), (40, (.96, .02, .02), 'hold'), (42, (.96, .02, .02), 'in'), (47, (8., 4.6, 1.2), 'hold'), (54, (8., 4.6, 1.2), 'in'), (58, (1.26, .20, 0), 'in'), (60, (.92, .04, 0), 'io'), (66, (.98, .02, 0), 'io'), (70, (.95, .02, 0), 'hold')]
ZH = [(-8, L_, 'hold'), (22, L_, 'io'), (25, L_ + .7, 'lin'), (27, L_ + 1.6, 'lin'), (30, L_ + 3.4, 'lin'), (34, L_ + 2 * math.pi - .2, 'out'), (37, L_ + 2 * math.pi, 'io'), (42, L_ + 2 * math.pi, 'in'), (47, L_ + 2 * math.pi - .5, 'hold'), (54, L_ + 2 * math.pi - .5, 'io'), (58, L_ + 2 * math.pi, 'hold')]


AZ = [(-8, -50.), (0, -30.), (10, 20.), (20, 124.), (30, 158.), (40, 200.), (50, 300.), (60, 338.), (70, 358.), (80, 363.)]


def azim(f):
    """degrees the camera has gone round: one turn, swift between the notes, always on the side from which the evasion reads"""
    return float(path(f, [(k, (v, 0., 0.)) for k, v in AZ])[0])


class Shot:
    id = 'g4'; n = 80
    finish = dict(bloom=.26, vig=.16, bloom_thr=.80)
    EVENTS = [(0, 'evade', dict(note='head tilts, a beam parts the hair: piano 1')), (10, 'evade', dict(note='leans back, three beams over the chest: piano 2')), (20, 'evade', dict(note='punch slides past the cheek: piano 3')), (30, 'evade', dict(note='tail sweeps under lifted feet: piano 4')), (40, 'evade', dict(note='ducks the roundhouse: piano 5')),
              (50, 'evade', dict(note='turns edge-on between five beams: piano 6')), (60, 'evade', dict(note='slips inside the backfist: piano 7')), (70, 'palm', dict(size=1., note='open palm on the chest, water ring, held chord')), (73, 'blown', dict(size=1., note='Frieza is gone; silence, ring goes on'))]

    def setup(self):
        cv2.setNumThreads(2)
        G = self.G = Fig('goku', 'ui', seed=3); Z = self.Z = Fig('frieza', seed=5)
        G.anim(lambda f: dict(pose=idle(seq(f, K), f, .8), pos=track(f, GP) + np.array([0, 0, .012 * math.sin(f * .16)]), heading=R_, look=None if 45 <= f <= 57 else self.eyes(f), reach=self.palm(f), wind=(-.10, .05, .06)))
        Z.anim(lambda f: dict(pose=seq(f, ZK), pos=track(f, ZP), heading=float(track(f, ZH)), look=np.array([0, 0, 1.35]) + track(f, GP)))
        Z.tail = self.tail; self.P70 = None
        self.P70 = self.Z.spot('chest', 70); self.beams = self.make()

    def eyes(self, f):
        z = track(f, ZP); return np.array([z[0], z[1] - .75 * S((f - 60) / 5.), z[2] + 1.35])

    def palm(self, f):
        if f < 64 or self.P70 is None: return {}
        p = self.Z.spot('chest', min(f, 72)) if f < 73 else self.P70 + [.02, 0, 0]
        return dict(L=dict(p=p + np.array([-.16 * (1 - S((f - 64) / 6.)), 0, 0]), part='palm', w=S((f - 63) / 6.5), palm=(1., 0, 0), fingers=(-.12, 0, 1.), g='open', body=.5))

    def tail(self, sk, f):
        if 26 <= f <= 35:                                             # one low sweep that meets empty space exactly under him on 30
            a = math.pi + (f - 30) * .62; b = sk.pel + sk.Rp @ np.array([0, -.07, 0]); g = track(30, GP); t = np.array([g[0] + .10 + .95 * math.cos(a) * .0, g[1], .16]); d = np.array([math.cos(a), math.sin(a), 0]); L = 1.30 * sk.H
            c = [b, b + d * .25 * L - [0, 0, .10 * L], b + d * .50 * L - np.array([0, 0, .26 * L]) + np.array([math.sin(a), -math.cos(a), 0]) * .06 * L, b + d * .74 * L - np.array([0, 0, .33 * L]) + np.array([math.sin(a), -math.cos(a), 0]) * .13 * L, b + d * .90 * L - np.array([0, 0, .32 * L]) + np.array([math.sin(a), -math.cos(a), 0]) * .26 * L]
            from db.h3_body import _cr3
            c = np.array(c); return _cr3(c, np.linspace(0, 1, len(c)), np.linspace(0, 1, 11)[1:])
        return tail_curve(sk, f, 'low', 1.) if 16 <= f <= 25 else tail_curve(sk, f, 'sway', 1.2)

    def make(self):
        """thin violet beams as lines in the world: (frame at which the head passes him, A, B, width)"""
        out = []; far = np.array([11., 2.6, 2.1])
        h = self.G.skel(1); q = h.head + h.Rh[:, 0] * h.u * 1.02 + h.Rh[:, 2] * h.u * .1
        def add(t, p, src, w): d = (p - src) / np.linalg.norm(p - src); out.append((t, src + d * 1.5, p + d * 16., w))
        add(1, q, far, 11)
        for k, (t, o) in enumerate(((4, (.0, -.34, 1.20)), (6, (.02, .36, .86)), (7, (.0, -.30, .55)), (3, (0, .5, 1.6)))): add(t, np.array(o), far + [0, .2 * k, .1 * k], 8)
        s = self.G.skel(11)
        for k, (dy, dz) in enumerate(((-.12, .10), (.10, .16), (.0, .30))): add(10 + k * .6, (s.chest + s.Rc[:, 1] * .15) * [1, 1, 0] + [0, dy, s.chest[2] + .16 + dz], far, 11)
        add(13, np.array([-.05, -.25, 1.05]), far, 8); add(15, np.array([0, .34, .7]), far, 8)
        src = np.array([8., 4.6, 2.2]); s = self.G.skel(50); d = np.array([8., 4.6, 0]); d /= np.linalg.norm(d); n = np.array([-d[1], d[0], 0])
        for k, (a, z) in enumerate(((.23, 1.42), (-.23, 1.20), (.25, .92), (-.22, .60), (.20, .34))): add(49 + k * .7, s.pel * [1, 1, 0] + n * a + [0, 0, z], src, 12)
        for k, (a, z) in enumerate(((.8, 1.6), (-.7, 1.0), (.6, .2), (-.9, 1.9), (1.2, .9))): add(47.5 + k * 1.3, n * a + [0, 0, z], src, 7)
        r = np.random.default_rng(8)
        for k in range(46):
            o = np.array([r.normal(0, .6), r.choice([-1, 1]) * r.uniform(.9, 4.2), r.uniform(-.9, 3.2)]); add(r.uniform(-3, 17), o, far + r.normal(0, .25, 3), r.uniform(4, 8))
        for k in range(40):
            o = n * (r.choice([-1, 1]) * r.uniform(.8, 4.0)) + [0, 0, r.uniform(-.9, 3.0)]; add(r.uniform(45, 57), o, src + r.normal(0, .25, 3), r.uniform(4, 8))
        return out

    def camera(self, f):
        a = math.radians(azim(f)); d = float(track(f, [(-8, 3.9, 'io'), (10, 3.6, 'io'), (20, 3.4, 'io'), (32, 4.1, 'io'), (42, 3.5, 'io'), (52, 3.6, 'io'), (62, 3.3, 'io'), (71, 2.75, 'io'), (80, 2.55, 'io')])); ez = float(track(f, [(-8, 1.25, 'io'), (12, 1.75, 'io'), (30, .45, 'io'), (42, 1.5, 'io'), (52, 1.1, 'io'), (70, 1.12, 'io')]))
        cx = float(track(f, [(0, .05, 'io'), (20, .38, 'io'), (45, .34, 'io'), (50, .0, 'io'), (60, .46, 'io'), (80, .52, 'io')])); tz = float(track(f, [(0, 1.0, 'io'), (30, .86, 'io'), (45, .92, 'io'), (60, 1.04, 'io'), (72, 1.14, 'io')]))
        t = np.array([cx, 0, tz]); e = t + np.array([math.sin(a) * d, -math.cos(a) * d, ez - tz]); roll = .10 * math.sin(f * .085 + .5); return Camera3D(e, t, 1500., W, HH, roll=roll)

    def draw_beams(self, cam, f, front):
        sp = []; c0 = self.G.skel().chest
        for t, A, B, w in self.beams:
            q = cam.clip_seg(A, B)
            if q is None: continue
            u = np.clip(((c0 - A) @ (B - A)) / ((B - A) @ (B - A)), 0, 1); C = A + (B - A) * u; nearer = cam.cam(C)[2] < cam.cam(c0)[2]
            if bool(nearer) != front: continue
            if np.hypot(*(q[1] - q[0])) < 60: continue
            dur = 5.; sp.append((t - dur * .52, q[0], q[1], w, dur))
        return ki.volley(sp, f, 'violet', splash=False) if sp else (None, None)

    def frame(self, f):
        az = azim(f); cam = self.camera(f); img, Lg = sky.space(az, f=f, pitch=-7 + 5 * math.sin(f * .05), roll=-math.degrees(.10 * math.sin(f * .085 + .5)), focal=900, aur=.55)
        g = f if any(0 <= f - n + 5 < 8 for n in NOTES) or f >= 62 else on(f, 2); G, Z = self.G, self.Z; G.at(g); Z.at(g)
        l = np.array(Lg['light']); l = np.array([l[0], l[1] + .22, max(l[2], .12)]); l /= np.linalg.norm(l)
        kw = dict(light=tuple(l), rim=[dict(Lg['rim'], thr=.54), dict(Lg['earth'], thr=.64, amt=.55)], pal=dict(base='space', mul='#d4d8ee', shmul='#8286c0', tint=('#6a74b4', .16)), key=.02)
        lay, li = self.draw_beams(cam, f, False)
        if li is not None: img = ki.add(img, li)
        if lay is not None: img = ki.put(img, lay)
        # where he was: ink wash of the drawing before the note
        for n in NOTES + (70,):
            age = f - n
            if 0 <= age < 11:
                pose_f = n - 5; G.at(pose_f); gl = G.draw(cam, pose_f, lod=1, shadow=False, **dict(kw, rim=None)); w = ki.inkwash(gl, age, seed=3 + n, ink='light', drift=(26. * math.cos(n), -14.), life=13., strength=.72)
                if w is not None: img = ki.put(img, w)
        G.at(g); Z.at(g); items = []; e = 'serene' if f < 66 else 'ui'
        zl = None
        if f < 73: items.append((Z, dict(kw, pal=dict(kw['pal'], shmul='#9ca2dc'), expr=dict(expr='contempt' if f < 18 else 'cruel' if f < 41 else 'rage' if f < 66 else 'surprise' if f < 70 else 'disbelief'))))
        items.append((G, dict(kw, expr=dict(expr=e, hair='ui'))))
        if 20 <= f <= 22: sg = G.skel(20); hit(Z, 'R', G, sg.spot('cheekL') - sg.Rh[:, 0] * .11 + sg.Rh[:, 1] * .02, part='fist', body=.5)
        L = compose(items, cam, g)
        sm = np.zeros_like(L)
        if f in (19, 20): sm = Z.smear(cam, f - 1.6, f, 'armR', pal='space', seed=2)
        if f in (39, 40): sm = Z.smear(cam, f - 1.6, f, 'legR', pal='space', seed=3)
        if f in (59, 60): sm = Z.smear(cam, f - 1.6, f, 'armR', pal='space', seed=4)
        if f in (15, 16, 17, 56, 57): sm = tails(L, self.vel(cam, f), n=50, seed=f)
        A = ki.aura(G.last(), f, 'silver', .19, seed=2, up=(0., -1.), face=(G.info['head']['x'], G.info['head']['y'], G.info['head']['s'] * 1.4))
        if A.back is not None: img = A.under(img)
        img = comp.over(img, sm); img = comp.over(img, L)
        if A.back is not None: img = A.over(img)
        if 73 <= f <= 76:
            Z.at(72); zl = Z.draw(cam, 72, **dict(kw, expr=dict(expr='disbelief'))); v = ki.vanish(zl, f - 73, 0., seed=4, stretch=3.4, kind='violet')
            if v is not None: img = ki.put(img, v)
            G.at(g); gl = G.draw(cam, g, **dict(kw, expr=dict(expr=e, hair='ui'))); img = comp.over(img, gl)
        lay, li = self.draw_beams(cam, f, True)
        if li is not None: img = ki.add(img, li, protect=np.clip(L[..., 3], 0, 1) * .6)
        if lay is not None: img = ki.put(img, lay)
        if f >= 70:
            p = cam.project(self.P70)[0]; img = ki.water_ring(img, float(p[0]), float(p[1]), f, 70, speed=44., asp=.60, rot=.06, seed=2, amp=34., rings=5, life=34)
            if f in (70, 71): img = img * 1.0
        return sky.wash(img, amt=.24)

    def vel(self, cam, f):
        a, b = cam.project(track(f - 1, ZP) + [0, 0, 1])[0], cam.project(track(f, ZP) + [0, 0, 1])[0]; v = b - a; n = np.hypot(*v) + 1e-6; return v / n * min(n, 260.)
