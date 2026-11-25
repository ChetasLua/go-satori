"""d4 EGO (40): a manga page comes alive.  Hook: shots/d4_fig.py  fig(f) -> layer drawn between the letters and the hand (f already held on twos)."""
import numpy as np, cv2, math
from db import W, H, glove as GL, letters as LT, hands as HD, storm
from db.paint import rgb, f32, smooth, fbm, grid, mix
from db.ink import Cel, spl
from db.comp import over, blur
from db.kit import screentone, focus_lines, speed_lines, hatch, shake

PAPER, INK = '#f1eee6', '#0b0a0c'
CX, CY = 1370, 640


class Shot:
    id = 'd4'; n = 40
    finish = dict(bloom=.10, vig=.10, soft=.03, ca=0., bloom_thr=.95)
    EVENTS = [(0, 'cut', dict(note='hard cut colour -> ink, one huge unison hit')), (0, 'letters', dict(note='EGO stamps the page')), (2, 'ink_settle', dict(size=.5)), (6, 'spark_on', dict(size=.2, note='violet spark lights in the palm')),
              (6, 'spark_grow', dict(to=38, note='rising whine, continuous 6..38')), (10, 'fingers', dict(size=.3, note='fingers tighten')), (20, 'fingers', dict(size=.5)), (30, 'fingers', dict(size=.7)), (30, 'surge', dict(to=39, size=1., note='the hand comes at the lens')), (38, 'spark_peak', dict(size=1.))]

    def setup(self):
        self.mod = storm.hook('d4'); po, _, _, _ = GL.POSE['claw']; tight = HD.blend(po, dict(po, **{k: tuple(np.array(po[k]) + (6, 18, 14, 0, 0)) for k in HD.FING}), 1.)
        self.hand = []; self.palm = []
        WRIST = np.array([1165., 885.])                             # the hand's own wrist sits here, behind the palm
        def claw(p, x_, y_): return GL.glove('claw', x_, y_, 352, ink=True, pose=p, dlight=(-.35, .55, .75), lw=1.9, seed=7, res=640, yaw=188, pitch=40, roll=-16, arm=.10)
        G = claw(po, 1370, 742); dw = WRIST - G.pts['wrist']; pos = np.array([1370., 742.]) + dw
        for k in range(10):                                         # keys 0, 3, 6, 9 (one per beat) with two inbetweens each
            q = HD.lerp_pose(po, tight, k / 9.); q = HD.tremble(q, 5 + k // 3, .8) if k % 3 == 0 else q; G = claw(q, pos[0], pos[1]); t = GL.tone(G, 8., dark=.26); m = G.layer * (1 - t[..., 3:4]) + t; self.hand.append(m.astype(np.float16)); self.palm.append(G.pts['palm'])
        self.ax = tuple(WRIST)
        x, y = grid(); tn = fbm(H, W, (260, 90, 30), seed=3); tn = (tn - tn.min()) / (np.ptp(tn) + 1e-6)
        pap = np.zeros((H, W, 3), f32) + rgb(PAPER); pap *= (1 - .035 * fbm(H, W, (3, 1.5), seed=9)[..., None] * .5)
        tone = screentone(np.ones((H, W), f32), 9., dark=np.clip(.10 + .55 * smooth(.35, .95, np.hypot((x - CX) / 1250, (y - CY) / 760)) + .10 * (tn - .5), 0, .75)); pap = pap * (1 - tone[..., None]) + rgb(INK) * tone[..., None]
        h = 590; m = LT.ego_mask((W / 2 - LT.ego_width(h) / 2 + 10, 130, h), wear=.62, seed=3); sh = cv2.warpAffine(m, np.float32([[1, 0, 16], [0, 1, 18]]), (W, H)); pap = pap * (1 - sh[..., None] * .92) + rgb(INK) * sh[..., None] * .92; pap = pap * (1 - m[..., None]) + rgb('#fbf9f3') * m[..., None]
        ed = cv2.dilate(m, np.ones((7, 7), np.uint8)) - cv2.erode(m, np.ones((3, 3), np.uint8)); pap = pap * (1 - np.clip(ed, 0, 1)[..., None]) + rgb(INK) * np.clip(ed, 0, 1)[..., None]; self.page = pap.astype(np.float16)
        self.lines = []
        for k in range(3):
            c = Cel(W, H, ss=1); focus_lines(c, CX, CY, 640, 1500, 150, 8., INK, seed=20 + k, ry=.66); self.lines.append(c.rgba().astype(np.float16))

    def spark(self, img, f, p, k):
        t = max(0, f - 6) / 32.; g = t * t * (3 - 2 * t)
        if f < 6: return img
        r = 5 + 120 * g; rg = np.random.default_rng(300 + f // 2); c = Cel(W, H)
        def star(R, n, amp, seed): q = np.random.default_rng(seed); a = np.linspace(0, 2 * math.pi, n * 2, endpoint=False) + q.uniform(0, .6); rr = np.where(np.arange(n * 2) % 2, R * .34, R * (1 + amp * q.random(n * 2))); return np.c_[p[0] + np.cos(a) * rr, p[1] + np.sin(a) * rr]
        c.fill(star(r * 1.5, 11, .9, 300 + f // 2), '#5a16c8'); c.fill(star(r * 1.05, 9, .6, 340 + f // 2), '#8a2cff'); c.fill(star(r * .66, 8, .4, 380 + f // 2), '#c784ff'); c.fill(star(r * .34, 7, .2, 420 + f // 2), '#fbf7ff')
        for j in range(int(3 + 6 * g)):                                                  # forked static
            a = rg.uniform(0, 6.28); L = r * rg.uniform(1.6, 2.9); P = [p]
            for s in range(5): a += rg.normal(0, .55); P.append(P[-1] + np.array([math.cos(a), math.sin(a)]) * L / 5)
            c.line(np.array(P), 4.5 * (.5 + g), '#8a2cff', (.05, .7), seed=j); c.line(np.array(P), 1.8 * (.5 + g), '#f4e8ff', (.05, .7), seed=j)
        lay = c.rgba(); gl = blur(lay[..., 3], 24 + 40 * g); img = 1 - (1 - img) * (1 - gl[..., None] * rgb('#7a22ff') * (.35 + .5 * g)); return over(img, lay)

    def frame(self, f):
        h2 = f // 2 * 2; beat, r_ = divmod(f, 10); k = min(9, max(0, (beat - 1) * 3 + (1 if r_ < 2 else 2 if r_ < 4 else 3))) if beat else 0; su = max(0., (f - 30) / 9.); z = 1 + .075 * h2 / 38. ; img = self.page.astype(f32); ln = self.lines[(f // 2) % 3].astype(f32); img = over(img, ln, .85)
        lay = storm.call(self.mod, 'fig', h2)
        if lay is not None: img = over(img, lay)
        zh = 1 + .05 * h2 / 38. + .42 * su ** 2.2; dx, dy = shake(h2, 2.2 + 3.5 * (h2 / 38.) ** 2, 41); ax, ay = self.ax; M = np.float32([[zh, 0, ax * (1 - zh) + dx], [0, zh, ay * (1 - zh) + dy]]); hd = cv2.warpAffine(self.hand[k].astype(f32), M, (W, H), flags=cv2.INTER_LINEAR)
        edge = blur((hd[..., 3] > .5).astype(f32), 5); img = img * (1 - np.clip(edge - hd[..., 3], 0, 1)[..., None] * .9) + rgb('#fbf9f3') * np.clip(edge - hd[..., 3], 0, 1)[..., None] * .9      # white keyline
        img = over(img, hd); p = np.array(self.palm[k]) * zh + (ax * (1 - zh) + dx, ay * (1 - zh) + dy); img = self.spark(img, f, p, k)
        img = cv2.warpAffine(img, np.float32([[z, 0, CX * (1 - z)], [0, z, CY * (1 - z)]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        if f == 0: img = np.clip(1 - img, 0, 1) * .98
        if f == 1: img = np.clip(img * 1.25 + .12, 0, 1.1)
        return img.astype(f32)
