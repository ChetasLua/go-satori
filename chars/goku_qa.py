"""QA sheets of the Goku lane:  python -m chars.goku_qa sheet|hands|small   (run from the project root)"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np, cv2, math, time
cv2.setNumThreads(4)
from db import comp, hands as HD
from db.paint import rgb
from db.ink import Cel
from chars import goku, goku_hands as K
D = math.pi / 180
BG = dict(day='#9fc4dc', lamp='#5a3018', moon='#140c24', rain='#70757a', gold='#1c2a20', ui='#0c0e16', dawn='#f2c8b4')


def cell(e, l, h, y, p, r, wear='gi', w=480, hh=520, s=100, body=True, **k):
    H = goku.GHead().pose(w * .5, hh * .46, s, y * D, p * D, r * D); a = goku.bust(H, e, l, h, seed=3, w=w, h=hh, wear=wear, body=body, **k); return a[..., :3] + (1 - a[..., 3:]) * rgb(BG[l])


def sheet():
    J = [('calm', 'day', 'base', 0, 0, 0), ('smile', 'day', 'base', 32, 2, 3), ('grin', 'day', 'base', -28, 2, -4), ('laugh', 'lamp', 'base', 18, 8, -6), ('serious', 'dawn', 'base', -40, -2, 0), ('determined', 'day', 'base', 20, -10, 3),
         ('smirk', 'dawn', 'base', 50, 0, 4), ('shock', 'lamp', 'base', -18, 2, 0), ('fear', 'moon', 'base', 28, 4, 0), ('grit', 'moon', 'base', -24, -6, 4, 'torn'), ('rage', 'moon', 'base', 10, -4, 0, 'torn'), ('grief', 'rain', 'base', -18, -8, 6, 'torn', dict(wet=.8, dmg=.5)),
         ('griefhid', 'rain', 'base', 30, -20, 6, 'torn', dict(wet=1.)), ('blank', 'gold', 's3', 0, 4, -3, 'torn', dict(hcol='gold')), ('scream', 'gold', 'ssj', -14, 6, 5, 'torn'), ('serious', 'gold', 'ssj', 35, -4, 2, 'torn'), ('serene', 'ui', 'ui', 12, -4, 4, 'bare'), ('ui', 'ui', 'ui', -22, -6, -3, 'bare'),
         ('serious', 'day', 'base', 88, 0, 0), ('calm', 'day', 'base', -88, 0, 0), ('determined', 'moon', 'base', 62, -12, 0), ('calm', 'dawn', 'base', -62, 14, 0), ('serious', 'gold', 'ssj', 80, 0, 0, 'torn'), ('ui', 'ui', 'ui', 65, 0, 0, 'bare'),
         ('calm', 'day', 'base', 20, 33, 0), ('serious', 'day', 'base', -20, -33, 0), ('calm', 'day', 's1', 0, 0, 0), ('grit', 'day', 's2', 0, 0, 0, 'torn'), ('rage', 'gold', 's4', 0, 0, 0, 'torn', dict(hcol='gold')), ('roar', 'gold', 's5', 0, 2, 0, 'torn', dict(hcol='gold'))]
    ims = []
    for j in J:
        k = j[7] if len(j) > 7 else {}; ims.append(cell(*j[:7], **k))
    o = np.vstack([np.hstack(ims[i:i + 6]) for i in range(0, 30, 6)]); return o


def hands():
    W, H = 640, 520
    def one(h, l, bg, s=150, cy=270, dist=6., **k): g, a = K.paint(h, HD.Cam(W / 2, cy, s, dist), l, width=W, height=H, clip=(0, 0, W, H), w=4, res=520, **k); return a[..., :3] + (1 - a[..., 3:]) * rgb(bg)
    ims = [one(K.kame(HD.Ry(90 * D)), 'day', BG['day']), one(K.kame(HD.Rx(62 * D)), 'ui', BG['ui']), one(K.kame(HD.Ry(60 * D) @ HD.Rz(-70 * D)), 'gold', BG['gold']), one(K.father(HD.Rx(20 * D), close=1.), 'day', BG['day'], 120), one(K.father(HD.Rx(20 * D), close=.3), 'day', BG['day'], 120), one(K.puddle(0.), 'rain', BG['rain'], 170), one(K.puddle(.5), 'rain', BG['rain'], 170), one(K.puddle(1.), 'rain', BG['rain'], 170)]
    return np.vstack([np.hstack(ims[:4]), np.hstack(ims[4:])])


if __name__ == '__main__':
    os.makedirs('qa/goku', exist_ok=True)
    for n in sys.argv[1:]: t = time.time(); comp.save(f'qa/goku/{n}.jpg', globals()[n]().astype(np.float32), 90); print(n, '%.1fs' % (time.time() - t))
