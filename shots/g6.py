"""g6 HANDS.  Four cuts, one per beat (0, 10, 20, 30): Gohan; Piccolo; Bulma, Krillin, Roshi and the turtle; a whole town.  In each: a dip (anticipation, 2-3), the arm whips up and lands
on the half beat (frame 5 of the cut), a light leaves the palm.  The dawn grows cut by cut (rise .16 .42 .68 .95)."""
import numpy as np, math
from db import comp
from chars import friends_hands as H

CUT = (H.g6_gohan, H.g6_piccolo, H.g6_beach, H.g6_town)


class Shot:
    id = 'g6'; n = 40
    finish = dict(bloom=.34, vig=.15, soft=.10, grain=.011)
    EVENTS = [(0, 'cut', dict(note='Gohan')), (2, 'breath_in', dict(size=.3, who='gohan')), (5, 'hand_up', dict(size=.6, who='gohan', note='cloth, small')), (5, 'light', dict(size=.5, note='glass chime')), (10, 'cut', dict(note='Piccolo')), (12, 'cape', dict(size=.3)), (15, 'hand_up', dict(size=.7, who='piccolo')), (15, 'light', dict(size=.55)),
              (20, 'cut', dict(note='beach')), (25, 'hand_up', dict(size=.7, who='krillin')), (25, 'light', dict(size=.55)), (26, 'hand_up', dict(size=.4, who='bulma')), (26, 'light', dict(size=.4)), (27, 'hand_up', dict(size=.4, who='roshi')), (27, 'light', dict(size=.4)),
              (30, 'cut', dict(note='town')), (31, 'crowd_breath', dict(frames=4)), (33, 'hands_wave', dict(frames=5, note='back rows first, front last: hundreds of sleeves')), (34, 'bird', dict(size=.3, note='one wing burst')), (34, 'lights_rise', dict(frames=6, size=1., note='choir of small chimes'))]

    def setup(self): pass

    def frame(self, f):
        c, lf = f // 10, f % 10; img = CUT[c](lf, f); z = 1.012 + .0026 * lf; return comp.cam2d(img.astype(np.float32), 960, 540 - 1.2 * lf, z).astype(np.float32)
