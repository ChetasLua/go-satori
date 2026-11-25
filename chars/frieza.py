"""FRIEZA head and neck (lead's drawing).  z = Frieza(cx, cy, s, yaw, pitch, roll, expr='serene', light='moon', f=0); z.layer(); Frieza.of(H, **k)"""
import numpy as np, cv2, math
from db import W, H as HH
from db.paint import rgb, f32, smooth, mix
from db.ink import Cel, spl, resample
from db.kit import rim, para
from chars.dbface import Face, patch, skull_pt, neck_poly

PROPS = dict(a=[(-1.0, .10), (-.965, .20), (-.89, .31), (-.74, .44), (-.54, .545), (-.30, .605), (-.05, .645), (.2, .69), (.42, .73), (.62, .735), (.8, .63), (.93, .40), (1.0, .06)],
             f=[(-1.0, .66), (-.92, .72), (-.8, .73), (-.64, .75), (-.46, .75), (-.28, .745), (-.06, .73), (.06, .75), (.2, .80), (.4, .83), (.6, .78), (.8, .62), (.93, .38), (1.0, .06)],
             nose=(.06, -.14, -.40, -.45, .115), socket=.06, cheek=.045)
LIGHTS = {
 'moon':  dict(sk='#8d86ad', sh='#342d50', dp='#16121f', hi='#f4ecff', ln='#08060e', dm='#5a15a8', ds='#1a0638', dh='#f0dcff', d2='#9a4cf0', eye='#f3edff', iris='#e0202e', lip='#0d0912', rim=('#b765ff', 5, 6), key=(-.35, .6, .35)),
 'back':  dict(sk='#4b4470', sh='#262040', dp='#17132a', hi='#d8c8ff', ln='#050409', dm='#35087a', ds='#12032a', dh='#e6ccff', d2='#7a2fe0', eye='#f3edff', iris='#ff2a35', lip='#050409', rim=('#c98bff', 6, 6), key=(-.2, .3, -.6)),
 'beam':  dict(sk='#fff0c8', sh='#a8763f', dp='#5e3a22', hi='#ffffff', ln='#2c1508', dm='#8a2a6a', ds='#3c0f34', dh='#fff6d0', d2='#e08a8a', eye='#fffdf2', iris='#c51c26', lip='#2c1508', rim=('#fff6c4', 8, 0), key=(-.95, .1, .2)),
 'gold':  dict(sk='#f1dfb4', sh='#8a6a62', dp='#4b3546', hi='#fffbe6', ln='#1d0f14', dm='#5f1a86', ds='#270a44', dh='#fff0c0', d2='#b060c8', eye='#fff9e8', iris='#d21f2b', lip='#1d0f14', rim=('#ffe27a', 7, 2), key=(-.9, .2, .35)),
 'space': dict(sk='#c9c6da', sh='#4a4568', dp='#1e1b30', hi='#ffffff', ln='#08060e', dm='#6a1fc0', ds='#22084a', dh='#f3e4ff', d2='#a65cff', eye='#fbf8ff', iris='#e0202e', lip='#120c1a', rim=('#d9b4ff', -6, 4), key=(.6, .5, .45)),
 'white': dict(sk='#fdf9ff', sh='#d9c8ea', dp='#b49cd0', hi='#ffffff', ln='#6a4b86', dm='#b98ae8', ds='#8d5cc8', dh='#ffffff', d2='#d8b8ff', eye='#ffffff', iris='#d06a80', lip='#7a5a94', rim=('#ffffff', 10, 0), key=(-.9, .0, .5)),
}
EXPR = {'serene': dict(eo=.62, look=(0, -.1), brow=0., mouth='line', jaw=0.), 'polite': dict(eo=.70, look=(0, 0), brow=.01, mouth='smile', jaw=0.), 'cruel': dict(eo=.82, look=(0, 0), brow=-.03, mouth='grin', jaw=.02),
        'contempt': dict(eo=.42, look=(.3, -.5), brow=.0, mouth='smirk', jaw=0.), 'surprise': dict(eo=1.2, look=(0, 0), brow=.06, mouth='part', jaw=.04, tiny=.7), 'fear': dict(eo=1.35, look=(-.1, 0), brow=.05, mouth='grit', jaw=.03, tiny=.5, twitch=1.),
        'rage': dict(eo=1.05, look=(0, 0), brow=-.06, mouth='shout', jaw=.16, tiny=.8), 'disbelief': dict(eo=1.4, look=(0, .1), brow=.08, mouth='part', jaw=.07, tiny=.45), 'empty': dict(eo=.8, look=(0, .3), brow=.09, mouth='line', jaw=0., tiny=.9), 'closed': dict(eo=0., look=(0, 0), brow=0., mouth='line', jaw=0.)}
EDGE = [(0, .335), (.14, .37), (.34, .47), (.58, .555), (.85, .58), (1.15, .53), (1.45, .44), (1.9, .34), (2.5, .28), (3.15, .25)]


class Frieza:
    def __init__(self, cx, cy, s, yaw=0., pitch=0., roll=0., expr='serene', light='moon', f=0, persp=.10, seed=4, glow=0., **over_):
        self.e = dict(EXPR[expr] if isinstance(expr, str) else expr); self.e.update(over_); self.L = dict(LIGHTS[light] if isinstance(light, str) else light)
        self.F = Face(PROPS, cx, cy, s, yaw, pitch, roll, self.e.get('jaw', 0.), persp); self.s, self.f, self.seed, self.lw, self.glow = s, f, seed, self.F.lw, glow
    @classmethod
    def of(cls, H, **k): return cls(H.cx, H.cy, H.s, H.yaw, H.pitch, H.roll, persp=getattr(H, 'persp', .1), **k)
    def line(self, c, P, w, **k): c.line(P, w * self.lw, k.pop('col', self.L['ln']), seed=self.seed, **k)

    def plate(self, c, sd):
        """round ear plate"""
        F, L = self.F, self.L; t = np.linspace(0, 2 * np.pi, 40); vis = float((F.H.R @ np.array([sd, 0, .18]))[2])
        rim_ = F.P3(np.stack([np.full(40, sd * .66), .02 + .255 * np.sin(t), -.08 + .20 * np.cos(t)], -1)); out = F.P3(np.stack([np.full(40, sd * .80), .02 + .215 * np.sin(t), -.10 + .165 * np.cos(t)], -1)); inn = F.P3(np.stack([np.full(40, sd * .81), .02 + .135 * np.sin(t), -.10 + .10 * np.cos(t)], -1))
        hull = cv2.convexHull(np.vstack([rim_, out]).astype(np.float32))[:, 0]; c.fill(hull, L['sh']); self.line(c, np.vstack([hull, hull[:1]]), 3.4, taper=(0, 0))
        if vis > -.2: c.fill(out, L['sk']); self.line(c, np.vstack([out, out[:1]]), 2.2, taper=(0, 0)); c.fill(inn, L['sh']); self.line(c, np.vstack([inn, inn[:1]]), 1.6, taper=(0, 0))

    def draw(self, c, neck=True):
        F, L, e = self.F, self.L, self.e; P, C = F.P, F.C
        if neck:
            poly, a, b = neck_poly(F, .36, .42, -.10, -.16); c.fill(poly, L['sk']); fc0 = F.contour(); dn = F.P3(np.array([0, -1.45, 0.])) - F.P3(np.array([0, -1., 0.]))
            c.shade(fc0 + dn * .95, L['sh'], poly); c.shade(fc0 + dn * .45, L['dp'], poly); [self.line(c, spl(q, n=6), 3.6) for q in (a, b)]
            for sd in (-1, 1): self.line(c, spl(F.P3(np.array([(sd * .28, -.9, .08), (sd * .18, -1.3, .22), (sd * .05, -1.7, .26)])), n=8), 1.8, taper=(.5, .3))
        vis = {sd: float((F.H.R @ np.array([sd, 0, .18]))[2]) for sd in (-1, 1)}; od = sorted((-1, 1), key=lambda q: vis[q])
        self.plate(c, od[0])
        sk = F.H.skull(); fc = F.contour(); c.fill(sk, L['sk']); facing = float(F.H.R[2][2]) > -.10; c.fill(F.H.under(), L['dp']) if facing else None; c.fill(fc, L['sk']) if facing else None
        both = cv2.convexHull(np.vstack([sk, fc]).astype(np.float32))[:, 0]
        # skull shading: far side from the key
        k = L['key']; off = np.array([k[0], -k[1]]) * self.s * .30
        m_in = [sk + off * 1.0]
        for p in F.planes(k, .02, ymax=.62): c.shade(p, L['sh'], fc)
        for p in F.planes(k, -.40, ymax=.62): c.shade(p, L['dp'], fc)
        up = F.P3(np.array([0, 1., 0])) - F.P3(np.array([0, 0., 0])); up = up / (np.linalg.norm(up) + 1e-6); o2 = F.P3(np.array([0, .42, .2]))
        idx = np.where(((sk - o2) @ up) >= -self.s * .30)[0]
        if len(idx) > 4:
            br = np.where(np.diff(idx) > 1)[0]; q = sk[np.r_[idx[br[0] + 1:], idx[:br[0] + 1]]] if len(br) else sk[idx]; self.line(c, q, 4.2, taper=(.05, .05), press=.3)
        # glossy dome
        edge = lambda t: float(np.interp(t, [q[0] for q in EDGE], [q[1] for q in EDGE]))
        dome = patch(F, c, edge, L['dm'], k_out=.985, lift=.25, swell=.055)
        for D in dome[:1]:
            # dark core band, then hard reflections in skull coordinates
            def S(pts, kk=.99): return F.P3(skull_pt(F, [q[0] for q in pts], [q[1] for q in pts], kk, [edge(abs(q[0])) for q in pts], .25, .055))
            yaw = F.yaw; t0 = -yaw
            sx0 = 1 if k[0] < 0 else -1
            band = [(t0 + sx0 * u, .50 + .46 * math.sin((u + .25) * .95)) for u in np.linspace(.05, 1.35, 22)] + [(t0 + sx0 * u, .50 + .18 * math.sin((u - .05) * 1.2)) for u in np.linspace(1.45, .30, 22)]
            c.shade(S(band), L['ds'], D)
            c.shade(S([(t0 + u, edge(abs(t0 + u)) + .015) for u in np.linspace(-1.2, 1.2, 24)] + [(t0 + u, edge(abs(t0 + u)) + .050 + .02 * math.cos(u * 2)) for u in np.linspace(1.1, -1.1, 24)]), L['d2'], D, .6)
            sx = -1 if k[0] < 0 else 1
            for (u, y, a, b_, ro) in ((.46 * sx, .78, .17, .085, .0), (.84 * sx, .60, .05, .075, 0.), (-.30 * sx, .90, .30, .020, 0.)):
                tt = np.linspace(0, 2 * np.pi, 28); c.shade(spl(S([(t0 + u + a * math.cos(q), y + b_ * math.sin(q) + a * .10 * math.cos(q) * sx) for q in tt]), True, (), 2), L['dh'], D)
        for D in dome: self.line(c, np.vstack([D, D[:1]]), 2.6, taper=(0, 0))
        # frame ridge: a second line above the brow, white plate edge
        for sd in (-1, 1):
            if F.vis(sd * .4, .3) > -.05: self.line(c, F.P3(skull_pt(F, np.linspace(.10, 1.0, 14) * sd, [edge(t) - .085 + .02 * t for t in np.linspace(.10, 1.0, 14)], 1.01)), 1.8, taper=(.5, .5))
        self.face(c, fc)
        self.line(c, resample(sk, 4.), 4.2, taper=(0, 0)) if False else None
        # contour ink: union outline
        msk = np.zeros((HH, W), np.uint8)
        ys = fc[:, 1]; up = F.P3(np.array([0, 1., 0])) - F.P3(np.array([0, 0., 0])); up = up / (np.linalg.norm(up) + 1e-6); o = F.P3(np.array([0, .42, .2])); idx = np.where(((fc - o) @ up) < 0)[0]
        if len(idx) > 4:
            br = np.where(np.diff(idx) > 1)[0]; q = fc[np.r_[idx[br[0] + 1:], idx[:br[0] + 1]]] if len(br) else fc[idx]; self.line(c, q, 4.4, taper=(.08, .08), press=.35)
        self.plate(c, od[1]) if vis[od[1]] > -.25 else None
        return self

    def face(self, c, fc):
        F, L, e, P, C = self.F, self.L, self.e, self.F.P, self.F.C; o = e['eo']; b = e['brow']
        for sd in (-1, 1):
            if F.vis(sd * .30, 0) < .02: continue
            s = np.array([sd, 1.]); tw = .03 * e.get('twitch', 0) * (1 if sd > 0 else 0) * (1 if (self.f // 2) % 2 else -1)
            inn, out = np.array([.115, -.085]), np.array([.540, .085]); up_o = np.array([inn, (.165, -.02), (.26, .045), (.37, .085), (.46, .098), out]); clo = np.array([inn, (.18, -.100), (.28, -.106), (.38, -.085), (.47, -.02), out - (0, .04)])
            lo = np.array([inn, (.19, -.125), (.29, -.135), (.39, -.112), (.48, -.040), out - (0, .015)]); up = clo + (up_o - clo) * min(o, 1.); up[1:5, 1] += max(0, o - 1) * .10; lo[1:5, 1] += -max(0, o - 1) * .05 + tw
            U, D = C(up * s, 8), C(lo * s, 8)
            c.shade(C(np.array([(.07, -.10), (.11, .02 + b), (.24, .13 + b * .6), (.42, .20 + b * .4), (.62, .22), (.60, .06), (.50, -.10), (.36, -.20), (.20, -.20)]) * s, 6, True), L['sh'], fc)           # dark eye rim
            if o > .05:
                poly = np.vstack([U, D[::-1]]); c.fill(poly, L['eye']); c.shade(np.vstack([U, (U + np.array([0, .05 * self.s]))[::-1]]), mix(rgb(L['eye']), rgb(L['sh']), .55), poly)
                k = e.get('tiny', 1.); lx, ly = e['look']; cx, cy = .315 + lx * .07 * sd, -.022 + ly * .045; t = np.linspace(0, 2 * np.pi, 30)
                E = lambda r, rx=1., dx=0., dy=0.: C(np.stack([cx + dx + r * rx * np.cos(t), cy + dy + r * 1.28 * np.sin(t)], -1) * s, 2, True)
                c.shade(E(.074 * k), L['ln'], poly); c.shade(E(.062 * k), L['iris'], poly); c.shade(E(.022 * k), L['ln'], poly); c.shade(E(.014, 1., -.022, .028), '#ffffff', poly); self.line(c, D[int(len(D) * .15):], 2.4, taper=(.5, .25))
            self.line(c, np.vstack([U, C(np.array([out, (.585, .12), (.63, .165)]) * s, 6)[1:]]), 5.4, taper=(.25, .4), press=.15)
            # heavy brow edge of the white plate
            br = np.array([(.055, .02 + b), (.12, .085 + b * .8), (.26, .165 + b * .5), (.44, .225 + b * .2), (.64, .25)]); self.line(c, C(br * s, 8), 3.6, taper=(.15, .5))
            c.shade(C(np.vstack([br, [(.64, .215), (.44, .185 + b * .2), (.26, .120 + b * .5), (.13, .045 + b * .8)]]) * s, 6, True), L['dp'], fc)
            # the two cheek lines
            self.line(c, C(np.array([(.30, -.145), (.315, -.26), (.30, -.40), (.265, -.50)]) * s), 2.2, taper=(.2, .6)); self.line(c, C(np.array([(.515, .02), (.53, -.18), (.49, -.36), (.43, -.47)]) * s), 2.2, taper=(.2, .6))
        sd = 1 if F.yaw >= 0 else -1; rp = F.H.P3(np.array([(0, y, float(F.H.z(0, y)) + d) for y, d in ((-.06, .02), (-.2, .05), (-.33, .09), (-.395, .10))])); ub = P(np.array([(-.06, -.425), (0, -.445), (.065, -.43)]) * (sd, 1))
        c.shade(np.vstack([rp[1:], ub[2:3], P(np.array([(.06, -.25)]) * (sd, 1))]), L['sh'], fc); self.line(c, spl(rp[1:], n=8), 1.8, taper=(.7, .2)); self.line(c, spl(ub, n=6), 2.0, taper=(.4, .4))
        self.mouth(c, fc)

    def mouth(self, c, fc):
        F, L, e, C = self.F, self.L, self.e, self.F.C; m = e['mouth']; j = e['jaw']
        lip = lambda pts, th: c.fill(np.vstack([C(pts), C([(x, y - th * (1 - (x / .24) ** 2)) for x, y in pts])[::-1]]), L['lip'])
        if m in ('line', 'smile', 'smirk', 'part'):
            pts = {'line': [(-.17, -.655), (-.07, -.645), (0, -.652), (.07, -.645), (.17, -.655)], 'smile': [(-.215, -.605), (-.10, -.650), (0, -.665), (.10, -.650), (.215, -.605)], 'smirk': [(-.15, -.660), (-.04, -.662), (.08, -.645), (.22, -.590)], 'part': [(-.13, -.665), (0, -.640), (.13, -.665)]}[m]
            if m == 'part': c.fill(C([(-.12, -.665), (0, -.635), (.12, -.665), (.09, -.72 - j), (0, -.74 - j), (-.09, -.72 - j)], 6, True), '#16060c')
            lip([(x, y + .012) for x, y in pts], .028); self.line(c, C(pts), 2.6, taper=(.15, .15)); self.line(c, C([(x * .55, -.715 - j * (m == 'part') + .012 * (x / .2) ** 2) for x in (-.2, -.1, 0, .1, .2)]), 3.0, taper=(.45, .45))
            if m in ('smile', 'smirk'):
                for s in ((-1, 1) if m == 'smile' else (1,)): self.line(c, C([(s * .225, -.575), (s * .255, -.605), (s * .25, -.65)]), 1.8)
        else:
            W_, t, b = {'grin': (.25, -.585, -.69), 'grit': (.20, -.615, -.715), 'shout': (.22, -.60, -.71 - j)}[m]
            cu = .05 if m == 'grin' else -.01; p = C([(-W_, t + cu * .6), (-W_ * .5, t - .02), (0, t - .03), (W_ * .5, t - .02), (W_, t + cu * .6), (W_ * .7, b + .02), (0, b - .015), (-W_ * .7, b + .02)], 8, True); c.fill(p, '#f3eef6' if m != 'shout' else '#1b0610')
            if m == 'shout': c.shade(C([(-W_ * .9, t), (0, t), (W_ * .9, t), (W_ * .7, t - .07), (0, t - .085), (-W_ * .7, t - .07)], 6, True), '#f3eef6', p); c.shade(C([(-W_ * .6, b + .03), (0, b + .11), (W_ * .6, b + .03), (0, b - .02)], 6, True), '#7a1d3c', p)
            else:
                self.line(c, C([(-W_ * .95, (t + b) / 2 + cu * .5), (0, (t + b) / 2 - .015), (W_ * .95, (t + b) / 2 + cu * .5)]), 1.8)
                for k in range(-3, 4): self.line(c, F.P([(k * W_ * .26, t), (k * W_ * .25, b)]), 1.0)
            self.line(c, np.vstack([p, p[:2]]), 4.6, taper=(0, 0), col=L['lip'])

    def layer(self, neck=True, ss=2):
        c = Cel(W, HH, ss=ss); self.draw(c, neck); Y = c.layer(); L = self.L
        if L['rim']: col, dx, dy = L['rim']; k = max(.5, self.s / 240.); Z0 = Y.copy(); Y = rim(Y, dx * k, dy * k, col, .9, soft=.8); b = int(max(abs(dx), abs(dy)) * k + 3); m = np.zeros((HH, W), bool); m[:b] = True; m[-b:] = True; m[:, :b] = True; m[:, -b:] = True; Y = np.where(m[..., None], Z0, Y).astype(f32)
        return Y
