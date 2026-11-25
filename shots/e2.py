"""e2 THE RAIN'S POINT OF VIEW (40): straight down.  Hook: shots/e2_fig.py  fig(f) -> layer (seen from above, centred at 960,540);  anchors(f) -> dict(hits=[(x, y), ...]).
Without hits the drops land on the wet rock, weighted to the centre."""
import numpy as np, cv2, math
from db import W, H, storm
from db.paint import rgb, f32, smooth, fbm, grid, ramp
from db.ink import Cel, spl
from db.comp import over, blur
from db.kit import outline


class Shot:
    id = 'e2'; n = 40
    finish = dict(bloom=.16, vig=.30, soft=.14, sat=.95)
    EVENTS = [(0, 'rain_bed', dict(to=40, size=.25, note='heard from inside: muffled, close'))]

    def setup(self):
        self.mod = storm.hook('e2'); x, y = grid(); r = np.random.default_rng(5); n = fbm(H, W, (300, 120, 46, 18), seed=21); n = (n - n.mean()) / n.std(); d = np.hypot((x - 960) / 1100, (y - 540) / 700); g = ramp(np.clip(.62 - .42 * d + n * .06, 0, 1), [(0, '#232629'), (.3, '#3c4043'), (.6, '#606567'), (1, '#8b9092')]); c = Cel(W, H)
        for _ in range(70):                                               # slabs and cracks of the wet cliff top
            a = r.uniform(0, 6.28); p = np.array([r.uniform(-50, W), r.uniform(-50, H)]); P = [p]
            for _k in range(5): a += r.normal(0, .5); P.append(P[-1] + np.array([math.cos(a), math.sin(a)]) * r.uniform(40, 120))
            c.line(np.array(P), r.uniform(1.5, 4.5), '#17191b', (.2, .6), seed=int(r.integers(1 << 20)), alpha=.75); c.line(np.array(P) + (3, 3), 1.2, '#9aa0a2', (.2, .6), seed=2, alpha=.5)
        for px, py, rx, ry in ((250, 200, 270, 120), (1620, 860, 300, 140)): a = np.linspace(0, 6.28, 50); wb = 1 + .20 * np.sin(a * 3 + px) + .13 * np.sin(a * 5 + py) + .08 * np.sin(a * 9 + 1); q = np.c_[px + np.cos(a) * rx * wb, py + np.sin(a) * ry * wb]; c.fill(q, '#555c5f'); c.fill(q * .90 + np.array([px, py]) * .10 - (10, 12), '#6d7578', .8); [c.line(spl([(px - rx * .5 + k * 60, py - ry * .3 + k * 34), (px + k * 60, py - ry * .45 + k * 34), (px + rx * .3 + k * 40, py - ry * .25 + k * 34)], n=8), 3., '#8c9497', (.4, .5), seed=k, alpha=.7) for k in range(3)]; c.line(np.vstack([q, q[:1]]), 1.8, '#1c1f22', (.03, .03), seed=4)
        lay = c.rgba(); self.plate = over(g, lay).astype(f32); self.pud = ((250, 200, 270, 120), (1620, 860, 300, 140)); self.def_hits = np.c_[960 + r.normal(0, 300, 160), 540 + r.normal(0, 210, 160)]

    def frame(self, f):
        z = 1.06 - .05 * f / 39.; rot = .02 * (f / 39. - .5); M = cv2.getRotationMatrix2D((960, 540), math.degrees(rot), z); img = cv2.warpAffine(self.plate, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101); c = Cel(W, H)
        for i, (px, py, rx, ry) in enumerate(self.pud):
            for k in range(4):
                ph = ((f + i * 5 + k * 11) % 26) / 26.; ox, oy = px + (storm._h(i, k, (f + i * 5 + k * 11) // 26) - .5) * rx, py + (storm._h(i, k, (f + i * 5 + k * 11) // 26, 2) - .5) * ry * .8; a = np.linspace(0, 6.28, 34); R = 8 + 74 * ph; c.line(np.c_[ox + np.cos(a) * R, oy + np.sin(a) * R * .8], 2.2 * (1 - ph), '#e6eaeb', (.02, .02), alpha=.7 * (1 - ph), seed=k)
        img = over(img, c.rgba()); lay = storm.call(self.mod, 'fig', f); an = storm.call(self.mod, 'anchors', f) or {}
        if lay is not None: sh = blur(cv2.warpAffine(lay[..., 3], np.float32([[1, 0, 22], [0, 1, 30]]), (W, H)), 16); img = img * (1 - .6 * sh[..., None]); img = over(img, lay)
        hits = np.array(an.get('hits') if an.get('hits') is not None else self.def_hits, float); c = Cel(W, H); b = storm.away(c, f, 84, 3, hits=hits, zhit=8., v=.50); drops = c.rgba(); c = Cel(W, H)
        for i, (hx, hy, age) in enumerate(b): storm.burst_at(c, hx, hy, age, 24, seed=int(hx) * 7 + int(hy))
        img = over(img, c.rgba()); c = Cel(W, H)
        if an.get('tear') is not None and f >= 12: storm.roll(c, np.asarray(an['tear'], float), min(1., (f - 12) / 25.), r=9.); img = over(img, c.rgba()); c = Cel(W, H)
        storm.away(c, f, 5, 77, hits=self.def_hits * 1.25 - (240, 135), zhit=8., v=.50, r=.10, start=.16); img = over(img, drops); return over(img, blur(c.rgba(), 5.)).astype(f32)
