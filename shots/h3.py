"""h3 HALF (80).  0 close: two hands break the bun, steam.  30 the half is offered, a gloved hand takes it (contact 42).  50 two fists meet: CONTACT AT 66, a ring runs through the grass.
Hook: shots/h3_fig.py  fig(f) -> layer drawn behind the hands (frames 30-49: the one who looks the other way; frames 50-79 optional)."""
import numpy as np, cv2, math
from db import W, H, storm, glove as GL, props as PR, hands as HD
from db.paint import rgb, f32, smooth, fbm, grid, ramp, mix
from db.ink import Cel, spl
from db.comp import over, blur
from db.kit import shake

SKIN = dict(HD.PAL['day'], skin='#f6c79f', shade='#c98c78', line='#23161f', nail='#fbdcc0', nail_shade='#d8a08c')
CUT1, CUT2, HIT = 30, 50, 66


def over_rgba(a, b): return a * (1 - b[..., 3:4]) + b


def band(c, g):
    """Goku's blue wristband round the forearm"""
    import cv2 as _cv
    for rings in (g.wrist.values() if isinstance(g.wrist, dict) else [g.wrist]):
        for (y0, y1, col) in ((.42, 1.02, '#2f4f9e'),):
            P = [np.asarray(r['P'], float) for r in rings if y0 <= abs(float(r['y'])) <= y1]
            if len(P) < 2: continue
            hull = _cv.convexHull(np.vstack(P).astype(np.float32))[:, 0]; c.fill(hull, col); m = hull.mean(0); lo = hull[hull[:, 1] > m[1]]
            c.shade(np.vstack([hull[hull[:, 1] > m[1] - 4], [m]]), '#1b2a63', hull) if len(lo) > 2 else None; c.line(np.vstack([hull, hull[:1]]), 4., '#151226', taper=(0, 0))


def bare(preset, x, y, s, side='R', **k):
    po, (a, b, c), dd, pv = GL.POSE[preset]; R = HD.euler(*(k.get('ang') or (a, b, c))) if side == 'R' else HD.euler(-(k.get('ang') or (a, b, c))[0], (k.get('ang') or (a, b, c))[1], -(k.get('ang') or (a, b, c))[2])
    G = GL.Glove(); h = HD.Hand(po, side, R, (0, 0, 0), HD.build(wid=1.12, boxy=.5, knuckle=1.05), pivot=pv, arm=k.get('arm', 4.6)); G.g = HD.render([h], HD.Cam(x, y, s, dd), (.75, .45, .45), res=560, clip=(-300, -300, W + 300, H + 300), thr=.10); c = Cel(W, H); _ink = lambda: None; HD.ink(c, G.g, SKIN, 4.6 * (s / 260.) ** .5, 3, marks=.7); band(c, G.g); G.layer = c.rgba(); G.pts = G.g.pts[0]; G.lw = (s / 260.) ** .55; return G


class Shot:
    id = 'h3'; n = 80
    finish = dict(bloom=.34, vig=.12, soft=.18)
    EVENTS = [(0, 'bird_bed', dict(to=80, size=.2)), (6, 'bun_press', dict(size=.2, note='thumbs sink in')), (11, 'bun_tear', dict(to=22, size=.5, note='soft dough tearing')), (22, 'bun_apart', dict(size=.4)), (30, 'cut', {}), (42, 'take', dict(size=.3, note='glove closes on the half')), (50, 'cut', {}), (57, 'windup', dict(size=.3, note='both fists draw back')), (60, 'swing', dict(size=.5)), (HIT, 'fist_bump', dict(size=1., note='contact, dull and warm')), (HIT, 'grass_ring', dict(to=80, size=.6))]

    def setup(self):
        self.mod = storm.hook('h3'); x, y = grid(); r = np.random.default_rng(3); n = fbm(H, W, (300, 110, 40), seed=4, aspect=3.); n = (n - n.mean()) / n.std(); sky = ramp(np.clip(y / 620 + n * .03, 0, 1), [(0, '#a9c9e8'), (.40, '#dfd3e0'), (.72, '#f7b7a8'), (1, '#ffd9c2')]); c = Cel(W, H)
        for L, col in (([(-20, 560), (160, 500), (340, 524), (560, 470), (760, 516), (980, 494), (1180, 530), (1420, 462), (1650, 516), (1940, 480)], '#b79cb6'), ([(-20, 596), (200, 556), (420, 580), (700, 560), (960, 584), (1300, 560), (1600, 582), (1940, 552)], '#957fa6')): c.fill(np.array(L + [(1940, 700), (-20, 700)]), col)
        img = over(sky, c.rgba()); t = np.clip((y - 600) / 200, 0, 1); lake = ramp(t, [(0, '#f6c2b2'), (.35, '#d9b6c6'), (1, '#8fa9cc')]) + (fbm(H, W, (70, 24), seed=9, aspect=9.) * .05)[..., None]; img = np.where((y > 600)[..., None], lake, img)
        s = np.exp(-(((x - 1480) / 330) ** 2 + ((y - 420) / 330) ** 2)); img = 1 - (1 - img) * (1 - s[..., None] * rgb('#ffe9c8') * .75); self.wide = img.astype(f32); self.soft = blur(self.wide, 16).astype(f32)
        q = lambda G: (G.pts['index', 'dip'] + G.pts['thumb', 'tip']) / 2
        def at(mk, want): G = mk(960, 600); d = np.array(want, float) - q(G); return mk(960 + d[0], 600 + d[1])          # place the pinch point exactly, so a layer only ever moves outward
        self.L = at(lambda x_, y_: bare('bun', x_, y_, 270, 'L'), (792, 620)); self.Rh = at(lambda x_, y_: bare('bun', x_, y_, 270, 'R'), (1128, 620)); self.Ls, self.Rs = GL.split(self.L), GL.split(self.Rh); self.lq, self.rq = q(self.L), q(self.Rh)
        self.of = at(lambda x_, y_: bare('bun', x_, y_, 230, 'L'), (930, 610)); self.ofs = GL.split(self.of); self.oq = q(self.of); self.tk = at(lambda x_, y_: GL.glove('bun', x_, y_, 230, light='dawn', torn=.5, arm=3.4, sleeve=SKIN['skin'], sleeve_s=SKIN['shade']), (1010, 600)); self.tks = GL.split(self.tk); self.tq = q(self.tk)
        ga = GL.glove('bump', 885, 560, 210, light='dawn', torn=.35, arm=5.4, seed=5, sleeve=SKIN['skin'], sleeve_s=SKIN['shade']); gb = bare('bump', 885, 560, 210, 'R', arm=5.4); A = over_rgba(gb.layer, GL.arm_planes(gb, SKIN['shade'], SKIN['line'])); B = GL.mirror(over_rgba(ga.layer, GL.arm_planes(ga, SKIN['shade'], SKIN['line'])), 960); da, db_ = GL.touch(A, B, 5.); self.A, self.B = GL.move(A, da), GL.move(B, db_); o = np.argwhere((self.A[..., 3] > .5) & (self.B[..., 3] > .5)); self.hit = (float(o[:, 1].mean()), float(o[:, 0].mean()))
        gr = np.clip((y - 596) / (H - 596), 0, 1); gnd = ramp(gr, [(0, '#b9b98c'), (.25, '#8fae78'), (.7, '#5f8d62'), (1, '#3f6b50')]) + (fbm(H, W, (120, 40, 14), seed=12, aspect=4.) * .045)[..., None]; self.ground = np.where((y > 596 + 4 * np.sin(x * .004))[..., None], gnd, self.wide).astype(f32); self.field = storm.Field(604, 1150, 1700, 4)

    def frame(self, f):
        if f < CUT1:
            img = self.soft.copy(); u = float(smooth(11, 23, f)); press = math.exp(-((f - 8) / 2.2) ** 2); sep = 150 * u + 26 * float(smooth(23, 26, f)) * math.exp(-(f - 23) / 3.) * (f >= 23); sq = 10 * press; z = 1.03 + .03 * f / 30.
            cx, cy = 960, 640; dl = dr = dy = 0.; off = sep * .62 - sq; bob = 2 * math.sin(f * .3)
            mv = lambda G, d: GL.move(G, d, dy + bob); img = over(img, mv(self.Ls[0], dl - off)); img = over(img, mv(self.Rs[0], dr + off)); c = Cel(W, H); PR.pull(c, cx, cy + 92 + bob, 185 * (1 + .02 * press), float(np.clip(u * 1.0, 0, 1)) if f >= 11 else 0., 'dawn', 2); img = over(img, c.rgba()); img = over(img, mv(self.Ls[1], dl - off)); img = over(img, mv(self.Rs[1], dr + off))
            img = PR.steam(img, cx, cy - 70, 330, 420, f / 24. + .6, 'dawn', .25 + .55 * u, 3, 7); return cv2.warpAffine(img, np.float32([[z, 0, 960 * (1 - z)], [0, z, 560 * (1 - z)]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).astype(f32)
        if f < CUT2:
            img = self.soft.copy(); lay = storm.call(self.mod, 'fig', f); img = over(img, lay) if lay is not None else img; g = f - CUT1; out = float(smooth(0, 7, g)); tin = float(smooth(3, 12, g)); back = float(smooth(14, 20, g)); ox = -260 * (1 - out); c = Cel(W, H); tx, ty = 1010, 600
            hold = self.tq; mvx = 560 * (1 - tin) + 330 * back; mvy = 60 * (1 - tin) + 30 * back; hx, hy = hold[0] + mvx, hold[1] + mvy
            offx, offy = ox, 0.; mine = g >= 12; k_ = float(smooth(10, 14, g)); bx, by = (hx - 34) * k_ + (self.oq[0] + offx + 30) * (1 - k_), (hy + 34) * k_ + (self.oq[1] + offy + 30) * (1 - k_)
            img = over(img, GL.move(self.of.layer if mine else self.ofs[0], offx - 40 * back, offy)); img = over(img, GL.move(self.tks[0], mvx, mvy)); PR.half(c, bx, by + 50, 118, -1, 'dawn', -.12, 2); img = over(img, c.rgba()); img = over(img, GL.move(self.ofs[1], offx - 40 * back, offy)) if not mine else img; img = over(img, GL.move(self.tks[1], mvx, mvy)); img = over(img, lay) if lay is not None and f >= CUT1 else img; return PR.steam(img, bx, by - 40, 120, 170, f / 24., 'dawn', .5, 5).astype(f32)
        img = self.ground.copy(); g = f - HIT; lay = storm.call(self.mod, 'fig', f); ring = None
        if g >= 0: R = 30 + 78 * g * (1 - .010 * g); ring = (self.hit[0], 880, R, math.exp(-g / 11.) * 1.15, .36)
        c = Cel(W, H); self.field.draw(c, f, ring, (0., .45)); img = over(img, c.rgba()); img = over(img, lay) if lay is not None else img
        if f < 57: d = 470 - 250 * float(smooth(CUT2, 57, f))                       # drift in
        elif f < 60: d = 220 + 44 * float(smooth(56, 59, f))                        # three frames back
        elif f < HIT: d = 264 * (1 - ((f - 59) / 7.) ** 2.2)                        # accelerate
        else: d = 0.
        sx, sy = shake(g, 3.0, 9, decay=3.) if 2 <= g < 10 else (0, 0); kick = 4 * math.exp(-(g - 2) / 3.) * math.sin((g - 2) * 1.3) if g >= 2 else 0.
        qx, qy = ((.945, 1.030) if g == 0 else (.970, 1.016)) if 0 <= g < 2 else (1., 1.)      # two frames of squash about the contact point
        def press_(L): hx, hy = self.hit; return cv2.warpAffine(L, np.float32([[qx, 0, hx * (1 - qx)], [0, qy, hy * (1 - qy)]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) if qx != 1 else L
        img = over(img, GL.move(press_(self.B), d + sx + kick, sy)); img = over(img, GL.move(press_(self.A), -d + sx + kick, sy)); c = Cel(W, H)
        if 0 <= g < 9:                                                        # the contact: short drawn ticks and one thin ring with real thickness
            for k in range(14): a = k * .449 + .2; r0, r1 = 54 + 16 * g, 84 + 34 * g * (1 - g / 14); p = np.array(self.hit); e = np.array([math.cos(a), math.sin(a) * .9]); c.line(np.array([p + e * r0, p + e * r1]), 5. * (1 - g / 9), '#fff7e4', (.1, .7), seed=k)
        self.field.draw(c, f, ring, (.45, 1.01)); return over(img, c.rgba()).astype(f32)
