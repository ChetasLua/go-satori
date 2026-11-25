"""VEGETA (lead's drawing).  v = Vegeta(cx, cy, s, yaw, pitch, roll, expr='pride', light='night', f=0, wet=0, wind=0); v.layer() -> premultiplied RGBA (head + neck)
   Vegeta.of(H, **k) from a posed head3.Head (head_fn for db/hero3).   s = px from eye line to chin.  LIGHTS / EXPR below."""
import numpy as np, cv2, math
from db import W, H as HH
from db.paint import rgb, f32, smooth, mix
from db.ink import Cel, spl, resample
from db.kit import rim, para
from chars.dbface import Face, blade, side_poly, scalp, neck_poly

PROPS = dict(a=[(-1.0, .14), (-.97, .28), (-.91, .41), (-.78, .525), (-.56, .615), (-.30, .655), (-.05, .675), (.2, .69), (.42, .695), (.62, .67), (.8, .55), (.93, .34), (1.0, .05)],
             f=[(-1.0, .66), (-.92, .74), (-.8, .75), (-.64, .77), (-.46, .77), (-.28, .755), (-.06, .735), (.06, .75), (.2, .80), (.4, .80), (.6, .73), (.8, .55), (.93, .32), (1.0, .05)],
             nose=(.10, -.04, -.43, -.49, .185), ear=(.69, .10, -.40, -.12), socket=.045, cheek=.034)
LIGHTS = {  # skin light, shade, deep, highlight, line; hair base, shade, light; rim colour, rim (dx,dy) px of light travel, key direction (screen x, y up, z)
 'day':   dict(sk='#f7cda6', sh='#cf8c69', dp='#a5634b', hi='#ffe9cf', ln='#24130f', hr='#1c1b26', hs='#0c0b12', hl='#4a5c86', rim=None, key=(-.5, .7, .6), eye='#fbf7f1'),
 'lamp':  dict(sk='#ffcb8f', sh='#c2703f', dp='#8f4a2a', hi='#ffe7b8', ln='#2b120a', hr='#201a20', hs='#0e0a0e', hl='#7a5a4e', rim=None, key=(.1, .8, .55), eye='#fff2d8'),
 'night': dict(sk='#a58fc4', sh='#4b3c78', dp='#2b2150', hi='#d7c4f2', ln='#0e0a1c', hr='#15132a', hs='#08070f', hl='#6a4fb8', rim=('#e2b8ff', 6, 5), key=(-.75, .55, .25), eye='#efe6ff'),
 'beam':  dict(sk='#ffe1a2', sh='#b0662f', dp='#6d3717', hi='#fffbe0', ln='#33150a', hr='#2a1c18', hs='#100a08', hl='#d99a3c', rim=('#fff3b4', -7, 2), key=(.9, .15, .35), eye='#fff8df'),
 'rain':  dict(sk='#c3ac9d', sh='#857470', dp='#594e51', hi='#e6dcd4', ln='#161315', hr='#191a1f', hs='#0b0b0e', hl='#5b6067', rim=('#e8ecec', 4, 7), key=(-.15, .95, .2), eye='#e9e7e4'),
 'dawn':  dict(sk='#ffd9c0', sh='#cb8a80', dp='#9d6266', hi='#fff2e4', ln='#331b1f', hr='#211c28', hs='#0f0c14', hl='#8c6a86', rim=('#fff1da', -6, 4), key=(.65, .5, .5), eye='#fffaf4'),
}
EXPR = {  # eye open, look (x,y), brow (inner dy, outer dy), mouth, jaw, extra
 'pride': dict(eo=.80, look=(0, 0), brow=(-.02, .02), mouth='flat', jaw=0.), 'scorn': dict(eo=.70, look=(.25, .1), brow=(-.03, .05), mouth='smirk', jaw=0.),
 'huff': dict(eo=.0, look=(0, 0), brow=(-.01, 0.), mouth='frown', jaw=0., blush=.8), 'shock': dict(eo=1.35, look=(0, 0), brow=(.10, .07), mouth='o', jaw=.10, tiny=.6),
 'pain': dict(eo=.18, look=(0, 0), brow=(-.02, -.02), mouth='grit', jaw=.05), 'move': dict(eo=1.5, look=(.1, 0), brow=(.07, .06), mouth='part', jaw=.04, tiny=.5),
 'shout': dict(eo=.95, look=(.3, 0), brow=(-.05, .06), mouth='shout', jaw=.20), 'peace': dict(eo=0., look=(0, 0), brow=(.055, -.015), mouth='hint', jaw=0.),
 'ego': dict(eo=.78, look=(0, 0), brow=(-.035, .05), mouth='flat', jaw=0., blank=1.), 'away': dict(eo=.62, look=(.75, -.1), brow=(.01, .0), mouth='frown', jaw=0., blush=1.),
 'smile': dict(eo=.60, look=(.2, 0), brow=(.02, -.005), mouth='smile', jaw=0.), 'fire': dict(eo=.85, look=(.2, 0), brow=(-.045, .06), mouth='grit', jaw=.08),
}
# 3-D hair blades: root, tip, bow, radius (right side; mirrored for x != 0 with a little asymmetry)
TOP = [((0, .80, .34), (.05, 2.42, -.42), (0, .10, .22), .30), ((.22, .80, .32), (.40, 2.22, -.40), (.10, .02, .20), .27), ((-.22, .80, .32), (-.30, 2.26, -.44), (-.12, .02, .20), .27),
       ((.42, .76, .18), (.78, 1.98, -.45), (.14, -.05, .16), .26), ((-.42, .76, .18), (-.70, 2.02, -.48), (-.16, -.05, .16), .26), ((.55, .62, .02), (.95, 1.76, -.52), (.22, -.08, .10), .23), ((-.55, .62, .02), (-.90, 1.80, -.55), (-.22, -.08, .10), .23),
       ((.60, .44, -.14), (1.02, 1.36, -.66), (.22, -.10, .04), .20), ((-.60, .44, -.14), (-1.0, 1.40, -.68), (-.22, -.10, .04), .20), ((.60, .24, -.28), (.94, 1.02, -.80), (.20, -.14, 0), .15), ((-.60, .24, -.28), (-.95, 1.05, -.82), (-.20, -.14, 0), .15)]
BACK = [((0, .78, -.42), (.06, 2.10, -1.22), (0, .12, -.16), .36), ((.34, .66, -.48), (.52, 1.72, -1.34), (.08, .08, -.18), .32), ((-.34, .66, -.48), (-.46, 1.76, -1.36), (-.08, .08, -.18), .32), ((0, .42, -.70), (0, 1.36, -1.66), (0, .04, -.2), .34),
        ((.36, .34, -.62), (.58, 1.12, -1.50), (.1, 0, -.16), .28), ((-.36, .34, -.62), (-.56, 1.16, -1.52), (-.1, 0, -.16), .28), ((.18, .08, -.74), (.24, .62, -1.54), (0, -.04, -.14), .25), ((-.2, .08, -.74), (-.26, .66, -1.52), (0, -.04, -.14), .25), ((0, -.16, -.70), (0, .12, -1.30), (0, -.06, -.1), .2)]
FRONT = [((.0, .56, .66), (.02, 1.72, .30), (0, .05, .18), .15), ((.14, .66, .60), (.24, 1.74, .16), (.05, .02, .17), .15), ((-.14, .66, .60), (-.20, 1.80, .14), (-.05, .02, .17), .15), ((.29, .74, .50), (.46, 1.62, .06), (.08, -.02, .15), .15), ((-.29, .74, .50), (-.42, 1.68, .02), (-.08, -.02, .15), .15)]
LINE_ = [(0, .40), (.035, .46), (.10, .585), (.20, .70), (.31, .765), (.42, .755), (.505, .68), (.555, .57), (.585, .45), (.60, .32), (.615, .18), (.632, .06), (.648, -.06), (.655, -.16)]   # hairline, right half


_HY = np.array([p[1] for p in LINE_[::-1]] + [.80, .90, 1.0]); _HX = np.array([p[0] for p in LINE_[::-1]] + [.10, .02, .0])


def _theta(y):
    i = np.argsort(_HY, kind='stable'); side = [(-.30, 1.46), (-.16, 1.40), (-.05, 1.30), (.06, 1.24), (.18, 1.16), (.32, 1.08), (.45, 1.00), (.57, .90), (.68, .72), (.755, .48), (.80, .25), (.86, .10), (.92, .02), (1.0, 0)]
    return float(np.interp(y, [a for a, _ in side], [b for _, b in side]))


class Vegeta:
    def __init__(self, cx, cy, s, yaw=0., pitch=0., roll=0., expr='pride', light='night', f=0, wet=0., wind=0., scratch=0., lw=None, persp=.10, hair_drop=0., seed=3, **over_):
        self.e = dict(EXPR[expr] if isinstance(expr, str) else expr); self.e.update(over_); self.L = dict(LIGHTS[light] if isinstance(light, str) else light)
        self.F = Face(PROPS, cx, cy, s, yaw, pitch, roll, self.e.get('jaw', 0.), persp); self.s, self.yaw = s, yaw; self.f, self.wet, self.wind, self.scr, self.drop, self.seed = f, wet, wind, scratch, hair_drop, seed
        self.lw = lw or self.F.lw
    @classmethod
    def of(cls, H, **k): return cls(H.cx, H.cy, H.s, H.yaw, H.pitch, H.roll, persp=getattr(H, 'persp', .1), **k)
    def line(self, c, P, w, **k): c.line(P, w * self.lw, k.pop('col', self.L['ln']), seed=self.seed, **k)

    def _blades(self, T):
        out = []; w = self.wind; d = self.drop
        for i, (r, t, b, rad) in enumerate(T):
            t = np.array(t, float); b = np.array(b, float); k = t[1] - r[1]
            t = t + np.array([.0, -.34 * d * k, -.10 * d * k]) + np.array([.0, -.10, -.30]) * w * k * (1 + .25 * math.sin(self.f * .55 + i * 1.3)) + np.array([.02 * math.sin(i * 2.1), .02 * math.cos(i * 1.7), 0])
            out.append(blade(self.F, r, t, b, rad))
        return out

    def hair_blade(self, c, b, inner=True, lo=.0):
        L = self.L; key = L['key']; c.fill(b['poly'], L['hr'])
        sh, lit = side_poly(b, key[0], key[1], .05, .0, 1.); c.shade(sh, L['hs'], b['poly'])
        if b['rad'].max() > 5:   # a hard highlight facet on the lit side
            S = b['S']; E = S + (lit - S) * .86; I = S + (lit - S) * .46; n = len(S); a, z = int(n * .22), int(n * .74)
            tt = np.linspace(0, 1, z - a)[:, None]; In = E[a:z] + (I[a:z] - E[a:z]) * np.sin(np.pi * tt) ** .7
            c.shade(np.vstack([E[a:z], In[::-1]]), L['hl'], b['poly'])
        q = int(len(b['S']) * lo)
        for E, w in ((b['L'], 2.6), (b['R'], 2.6)): self.line(c, E[q:], w, taper=(.35, .12))

    def draw(self, c, neck=True, hair=True):
        F, L, e = self.F, self.L, self.e; lw = self.lw; P, C = F.P, F.C; yaw = self.yaw
        back = sorted(self._blades(BACK), key=lambda b: b['depth']); top = sorted(self._blades(TOP), key=lambda b: b['depth']); front = sorted(self._blades(FRONT), key=lambda b: b['depth'])
        if hair:
            for b in [b for b in back]: self.hair_blade(c, b)
            for b in [b for b in top if b['depth'] < -.25]: self.hair_blade(c, b)
            c.fill(F.H.skull(), L['hr']) if abs(yaw) > 1.9 else None
        fc = F.contour()
        if neck: self.neck(c)
        for sd in (-1, 1):
            if (yaw * sd) < .25: pass
        order = sorted((-1, 1), key=lambda q: F.H.ear(q)['vis'])
        ear = lambda sd: self.ear(c, sd)
        if F.H.ear(order[0])['vis'] > -.15: ear(order[0])
        facing = float(F.H.R[2][2]) > -.10; c.fill(F.H.under(), L['dp']) if facing else None; c.fill(fc, L['sk']) if facing else None
        for p in F.planes(L['key'], .06): c.shade(p, L['sh'], fc)
        for p in F.planes(L['key'], -.42): c.shade(p, L['dp'], fc)
        self.features(c, fc)
        self.outline(c, fc)
        if hair: scalp(F, c, _theta, L['hr'])
        if F.H.ear(order[1])['vis'] > .02: ear(order[1])
        if hair:
            hl = [np.array([(x * sd, y) for x, y in LINE_]) for sd in (-1, 1)]
            ring = np.vstack([hl[0][::-1], hl[1][1:]]); cap = np.vstack([P(spl(ring, False, (), 5)), P([(.66, .5), (.60, .78), (.30, .95), (0, .99), (-.30, .95), (-.60, .78), (-.66, .5)])])
            c.fill(cap, L['hr'])
            for b in [b for b in top if b['depth'] >= -.25]: self.hair_blade(c, b, lo=.12)
            for b in front: self.hair_blade(c, b, lo=.18)
            for sd in (-1, 1):
                h = P(spl(hl[(sd + 1) // 2], False, (), 6)); self.line(c, h, 3.0, taper=(.3, .25))
                if F.vis(.55 * sd, .4) > .1:     # hairline teeth: small wedges that bite into the forehead
                    r = np.random.default_rng(5)
                    for k in range(5):
                        i = 2 + k * 2; a = np.array(LINE_[min(i, 12)]) * (sd, 1); z = np.array(LINE_[min(i + 1, 13)]) * (sd, 1); m = (a + z) / 2 + np.array([-.035 * sd, -.05]) * (.6 + .5 * r.random())
                        c.fill(P([a, m, z]), L['hr'])
        return self

    def outline(self, c, fc):
        self.line(c, fc, 4.2, taper=(0, 0), press=.25) if False else None
        H = self.F.H; jl, jr = H.jaw(-1), H.jaw(1)
        # ink the face contour below the ears only, thicker under the jaw
        ys = fc[:, 1]; up = H.P3(np.array([0, 1., 0])) - H.P3(np.array([0, 0., 0])); up = up / (np.linalg.norm(up) + 1e-6); o = H.P3(np.array([0, .30, .3])); below = ((fc - o) @ up) < 0
        idx = np.where(below)[0]
        if len(idx) < 5: return
        br = np.where(np.diff(idx) > 1)[0]; P = fc[np.r_[idx[br[0] + 1:], idx[:br[0] + 1]]] if len(br) else fc[idx]
        self.line(c, P, 4.6, taper=(.10, .10), press=.35)

    def neck(self, c):
        F, H, L = self.F, self.F.H, self.L; poly, a, b = neck_poly(F); c.fill(poly, L['sk']); fc = F.contour()
        dn = H.P3(np.array([0, -1.45, .0])) - H.P3(np.array([0, -1.0, .0])); k = L['key']; off = dn * .9 + np.array([-k[0], k[1]]) * self.s * .12
        c.shade(np.vstack([fc + off]), L['sh'], poly); c.shade(fc + off * .45, L['dp'], poly)
        for q in (a, b): self.line(c, spl(q, n=6), 4.0, taper=(.05, .2))
        for sd in ((-1, 1) if abs(self.yaw) < .9 else ()): self.line(c, spl(H.P3(np.array([(sd * .36, -.86, .10), (sd * .27, -1.20, .26), (sd * .13, -1.55, .33), (sd * .05, -1.80, .34)])), n=8), 2.2, taper=(.5, .3))

    def ear(self, c, sd):
        E = self.F.H.ear(sd); L = self.L; c.fill(E['outer'], L['sk']); c.shade(E['inner'], L['sh'], E['outer']); self.line(c, E['outer'], 3.2, taper=(.1, .2)); self.line(c, E['inner'], 1.8); self.line(c, E['tragus'], 1.6)

    def features(self, c, fc):
        F, L, e, P, C = self.F, self.L, self.e, self.F.P, self.F.C
        bi, bo = e['brow']; blush = e.get('blush', 0)
        if blush:
            for sd in (-1, 1):
                if F.vis(sd * .4, -.22) > .1: c.shade(C([(sd * .22, -.22), (sd * .40, -.15), (sd * .58, -.20), (sd * .40, -.30)], closed=True), '#e8715c', fc, .45 * blush)
        for sd in (-1, 1):
            if F.vis(sd * .30, .0) < .03: continue
            o = e['eo']; s = np.array([sd, 1.])
            inn, out = np.array([.130, -.075]), np.array([.500, .045])
            up_o = np.array([inn, (.17, -.025), (.25, .020), (.36, .055), (.45, .068), out]); clo = np.array([inn, (.18, -.090), (.27, -.100), (.37, -.085), (.45, -.040), out - (0, .03)])
            lo = np.array([inn, (.19, -.112), (.28, -.128), (.37, -.112), (.45, -.058), out - (0, .012)])
            up = clo + (up_o - clo) * min(o, 1.); up[1:5, 1] += max(0, o - 1) * .10; lo = lo.copy(); lo[1:5, 1] -= max(0, o - 1) * .035
            U, D = C(up * s, 8), C(lo * s, 8)
            # socket shadow under the brow
            c.shade(C(np.array([(.10, -.04), (.16, .07 + bi * .6), (.34, .12 + (bi + bo) * .4), (.56, .20 + bo * .7), (.53, .07), (.36, .05), (.22, .0)]) * s, 6, True), L['sh'], fc)
            if o > .06:
                poly = np.vstack([U, D[::-1]]); c.fill(poly, L['eye'])
                c.shade(np.vstack([U, (U + np.array([0, .055 * self.s]))[::-1]]), mix(rgb(L['eye']), rgb(L['sh']), .6) if not e.get('blank') else L['eye'], poly)
                if not e.get('blank'):
                    k = e.get('tiny', 1.); lx, ly = e['look']; q = np.array([(.305 + lx * .07 * sd, -.018 + ly * .04)]); t = np.linspace(0, 2 * np.pi, 28)
                    ir = C(np.stack([q[0, 0] + .058 * k * np.cos(t), q[0, 1] + .088 * k * np.sin(t)], -1) * s, 2, True); c.shade(ir, L['ln'], poly); hi = C(np.stack([q[0, 0] - .018 + .018 * k * np.cos(t), q[0, 1] + .030 + .022 * k * np.sin(t)], -1) * s, 2, True); c.shade(hi, '#ffffff', poly)
                self.line(c, D[int(len(D) * .28):], 1.7, taper=(.6, .25))
            uw = np.vstack([U]); self.line(c, uw, 4.6 if o > .06 else 3.4, taper=(.22, .12), press=.2)
            if o <= .06: self.line(c, C(np.array([(.20, -.135), (.29, -.150), (.38, -.135)]) * s), 1.4)
            # brow: one heavy blade, joined to the lid near the nose
            br = np.array([(.075, -.005 + bi), (.15, .045 + bi * .9), (.27, .100 + bi * .5 + bo * .2), (.42, .190 + bo * .7), (.585, .300 + bo), (.47, .270 + bo * .9), (.34, .215 + bi * .2 + bo * .6), (.21, .155 + bi * .6), (.115, .095 + bi), (.065, .045 + bi)]) * s
            c.fill(C(br, 6, True, (0, 4)), L['hr']); self.line(c, np.vstack([C(br, 6, True, (0, 4))]), 1.5, taper=(0, 0))
            # frown crease, cheek and under-eye lines
            self.line(c, C(np.array([(.055, .06 + bi), (.04, .13 + bi), (.06, .20 + bi)]) * s), 1.8, taper=(.3, .6))
            self.line(c, C(np.array([(.150, -.130), (.20, -.195), (.27, -.225)]) * s), 1.7, taper=(.3, .7))
            self.line(c, C(np.array([(.52, -.26), (.47, -.40), (.385, -.52)]) * s), 1.6, taper=(.4, .7))
        # nose: long wedge, the far side in shadow
        sd = 1 if (F.yaw + (.001 if self.L['key'][0] < 0 else -.001)) >= 0 else -1; sd = -sd if abs(F.yaw) > .12 and False else sd
        w = .085; dz = lambda y: np.interp(y, [-.50, -.44, -.30, -.04, .06], [.05, .16, .12, .03, .0])
        ridge = np.array([(.0, .03), (.01, -.10), (.012, -.25), (.015, -.38), (.0, -.435)])
        rp = F.H.P3(np.array([(x * sd, y, float(F.H.z(0, y)) + dz(y)) for x, y in ridge])); bp = F.P(np.array([(.10, -.02), (.075, -.15), (.085, -.30), (.115, -.43)]) * (sd, 1)); ub = F.P(np.array([(-.085, -.455), (0, -.49), (.09, -.47)]) * (sd, 1))
        c.shade(np.vstack([rp, ub[1:2], ub[2:3], bp[::-1]]), L['sh'], None); c.shade(np.vstack([rp[-1:], ub]), L['dp'], None)
        self.line(c, spl(np.vstack([rp[1:]]), n=8), 2.6, taper=(.6, .2)); self.line(c, spl(np.vstack([rp[-1:], ub[1:2], ub[2:3]]), n=8), 2.6, taper=(.2, .4)); self.line(c, F.C(np.array([(-.07, -.45), (-.105, -.43), (-.10, -.40)]) * (sd, 1)), 2.0)
        self.mouth(c, fc)
        r = np.random.default_rng(11)
        for k in range(int(8 * self.scr)):
            x, y = [(.36, -.30), (-.42, -.34), (.18, .34), (-.2, .42), (-.30, -.58), (.44, -.12), (.05, -.80), (-.5, .05)][k % 8]
            if F.vis(x, y) > .15:
                a = r.uniform(-.6, .9); v = np.array([math.cos(a), math.sin(a)]) * .07; self.line(c, P([(x - v[0], y - v[1]), (x + v[0], y + v[1])]), 1.7, taper=(.5, .5)); c.fill(C([(x, y - .012), (x + .03, y - .07), (x + .012, y - .012)], 4, True), '#7d1c22') if k % 3 == 0 else None

    def mouth(self, c, fc):
        F, L, e, C = self.F, self.L, self.e, self.F.C; m = e['mouth']; j = e.get('jaw', 0); y = -.665
        if m in ('flat', 'frown', 'smirk', 'hint', 'smile', 'part'):
            pts = {'flat': [(-.17, -.675), (-.08, -.662), (0, -.668), (.08, -.662), (.17, -.678)], 'frown': [(-.17, -.695), (-.08, -.662), (0, -.655), (.08, -.662), (.17, -.695)], 'smirk': [(-.15, -.665), (-.05, -.675), (.06, -.665), (.16, -.640), (.215, -.610)],
                   'hint': [(-.185, -.640), (-.10, -.668), (0, -.678), (.10, -.668), (.185, -.640)], 'smile': [(-.215, -.615), (-.12, -.665), (0, -.685), (.12, -.665), (.215, -.615)], 'part': [(-.15, -.68), (0, -.655), (.15, -.68)]}[m]
            self.line(c, C(pts), 3.0, taper=(.18, .18)); self.line(c, C([(-.07, -.775), (0, -.785), (.07, -.775)]), 1.8, taper=(.5, .5))
            if m in ('smirk',): self.line(c, C([(.225, -.585), (.245, -.62), (.235, -.655)]), 1.6)
            if m in ('smile', 'hint'):
                for s in (-1, 1): self.line(c, C([(s * .205, -.605), (s * .228, -.632), (s * .22, -.66)]), 1.5)
            return
        W_ = {'o': .10, 'grit': .24, 'shout': .25}[m]; t, b = -.615, -.70 - j * .95
        if m == 'grit': t, b = -.600, -.725 - j * .5
        ring = [(-W_, t - .03), (-W_ * .55, t + .012), (0, t - .006), (W_ * .55, t + .012), (W_, t - .03), (W_ * .92, (t + b) / 2 - .01), (W_ * .6, b + .012), (0, b - .004), (-W_ * .6, b + .012), (-W_ * .92, (t + b) / 2 - .01)]
        p = C(ring, 8, True); c.fill(p, '#4a1518')
        if m == 'grit':
            c.fill(p, '#f1e9dc'); c.shade(C([(-W_, t - .02), (0, t + .03), (W_, t - .02), (W_ * .9, t - .055), (0, t - .045), (-W_ * .9, t - .055)], 6, True), L['sh'], p, .55); mid = C([(-W_, (t + b) / 2 - .01), (0, (t + b) / 2 - .02), (W_, (t + b) / 2 - .01)]); self.line(c, mid, 2.0)
            for k in range(-3, 4): self.line(c, F.P([(k * .062, t), (k * .060, b)]), 1.1)
        else:
            c.shade(C([(-W_ * .9, t - .02), (0, t + .02), (W_ * .9, t - .02), (W_ * .7, t - .085), (0, t - .10), (-W_ * .7, t - .085)], 6, True), '#f7f2ea', p)
            c.shade(C([(-W_ * .6, b + .05), (0, b + .14), (W_ * .6, b + .05), (0, b - .02)], 6, True), '#b8484a', p)
        self.line(c, np.vstack([p, p[:2]]), 3.2, taper=(0, 0)); self.line(c, C([(-.09, b - .09), (0, b - .105), (.09, b - .09)]), 2.0, taper=(.5, .5))
        for s in (-1, 1): self.line(c, C([(s * (W_ + .04), t + .06), (s * (W_ + .075), t - .04), (s * (W_ + .06), b + .05)]), 2.0, taper=(.5, .5))

    def layer(self, neck=True, hair=True, ss=2):
        c = Cel(W, HH, ss=ss); self.draw(c, neck, hair); Y = c.layer(); L = self.L
        if L['rim']: col, dx, dy = L['rim']; k = max(.5, self.s / 260.); Z0 = Y.copy(); Y = rim(Y, dx * k, dy * k, col, .85, soft=.8); b = int(max(abs(dx), abs(dy)) * k + 3); m = np.zeros((HH, W), bool); m[:b] = True; m[-b:] = True; m[:, :b] = True; m[:, -b:] = True; Y = np.where(m[..., None], Z0, Y).astype(f32)
        return Y
