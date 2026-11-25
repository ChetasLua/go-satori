import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, cv2, time
from db import comp, W, H
from db.paint import rgb, f32
from chars.chibi_core import *
from chars import chibi_story as ST
def canvas(col, w=W, h=H): return np.ones((h, w, 3), f32) * rgb(col)
if __name__ == '__main__':
    k = sys.argv[1]
    if k == 'line':
        t = time.time(); im = canvas('#f6e6b8'); d = D(lw=4.8, line=ST.LINE)
        for i, w in enumerate(['piccolo', 'vegeta', 'bulma', 'gohan', 'goku', 'krillin', 'roshi']):
            S = 112; ST.fig(d, w, 170 + i * 258, 700, S, ST.walk(w, float(sys.argv[2]) if len(sys.argv) > 2 else .0))
        im = comp.over(im, d.layer(grain=.02)); comp.save('qa/chibi/_line.jpg', im, 92); print(time.time() - t)

    if k == 'walk':
        im = canvas('#f6e6b8'); d = D(lw=2.2, line=ST.LINE); who = ['goku', 'krillin', 'gohan', 'bulma', 'vegeta', 'piccolo', 'roshi']
        sel = sys.argv[2].split(',') if len(sys.argv) > 2 else who
        for j, w in enumerate(sel):
            for i in range(10):
                S = 1080 / len(sel) / 4.6; ST.fig(d, w, 100 + i * 190, (j + 1) * 1080 / len(sel) - 18, S, ST.walk(w, i * 2 / 20.)); 
                d.fl.fill(A([[20 + i * 190, (j + 1) * 1080 / len(sel) - 18], [180 + i * 190, (j + 1) * 1080 / len(sel) - 18], [180 + i * 190, (j + 1) * 1080 / len(sel) - 16], [20 + i * 190, (j + 1) * 1080 / len(sel) - 16]]), '#c9a46a')
        im = comp.over(im, d.layer(grain=.02)); comp.save('qa/chibi/_walk.jpg', im, 92)
    if k == 'dragon':
        from chars import chibi_dragon as DR
        im = canvas('#c6e6ff'); d = D(lw=4.8, line=ST.LINE); f = float(sys.argv[2]) if len(sys.argv) > 2 else 0.
        DR.shenron(d, DR.path(f, 1500, 420), 112, f)
        for i in range(7): DR.bounce_ball(d, 300 + i * 120, 960, 44, f + i * 3, 0, i + 1, seed=i)
        im = comp.over(im, d.layer(grain=.02)); comp.save('qa/chibi/_dragon.jpg', im, 92)
    if k == 'parade':
        from chars import chibi_parade as PR
        t = time.time(); S_ = PR.Parade(); S_.setup(); print('setup', time.time() - t); f = int(sys.argv[2]) if len(sys.argv) > 2 else 4; t = time.time(); im = S_.frame(f); print('frame', time.time() - t)
        im = comp.finish(im, f, **PR.FINISH); comp.save(sys.argv[3] if len(sys.argv) > 3 else 'qa/chibi/parade.jpg', im, 93)
    if k == 'stick':
        from chars import chibi_stick as SK
        t = time.time(); who = sys.argv[2]; f = int(sys.argv[3]) if len(sys.argv) > 3 else 9; im = SK.scene(who, f); print(time.time() - t); comp.save(f'qa/chibi/_s_{who}.jpg', im, 92)
    if k == 'sheet8':
        from chars import chibi_stick as SK
        fr = [int(v) for v in sys.argv[2].split(',')] if len(sys.argv) > 2 else [9]; rows = []
        for f in fr: rows += [cv2.resize(SK.scene(w, f), (480, 270), interpolation=cv2.INTER_AREA) for w in SK.NAMES]
        comp.save('qa/chibi/_sheet8.jpg', np.vstack([np.hstack(rows[i:i + 4]) for i in range(0, len(rows), 4)]), 92)
    if k == 'four':
        from chars import chibi_stick as SK
        f = int(sys.argv[3]); ims = [SK.scene(w, f)[40:1060, 400:1520] for w in sys.argv[2].split(',')]
        comp.save('qa/chibi/_four.jpg', cv2.resize(np.hstack(ims), None, fx=.86, fy=.86, interpolation=cv2.INTER_AREA), 92)

    if k == 'strips':
        from chars import chibi_stick as SK
        who = sys.argv[2].split(',') if len(sys.argv) > 2 else SK.NAMES; rows = []
        for w in who: rows.append(np.hstack([cv2.resize(SK.scene(w, f)[0:1080, 420:1500], (192, 192), interpolation=cv2.INTER_AREA) for f in range(10)]))
        comp.save('qa/chibi/sticker_strips.jpg' if len(who) == 8 else 'qa/chibi/_strips.jpg', np.vstack(rows), 92)

    if k == 'grid':
        from chars import chibi_stick as SK
        t = time.time(); im = SK.grid(int(sys.argv[2]) if len(sys.argv) > 2 else 10, int(sys.argv[3]) if len(sys.argv) > 3 else 0); print(time.time() - t); comp.save(sys.argv[4] if len(sys.argv) > 4 else 'qa/chibi/stickers.jpg', im, 93)

    if k == 'feast':
        from chars import chibi_feast as FE
        t = time.time(); Fz = FE.Feast(); Fz.setup(); print('setup', time.time() - t); t = time.time(); im = Fz.frame_standoff(float(sys.argv[2]) if len(sys.argv) > 2 else 4); print(time.time() - t); im = comp.finish(im, 4, bloom=.30, vig=.12, soft=.14); comp.save('qa/chibi/feast.jpg', im, 93)

    if k == 'gags':
        from chars import chibi_feast as FE
        im = canvas('#b5653a'); d = D(lw=5., line=FE.INK, var=.3)
        FE.gag(d, 'goku', 330, 250, 105, False, dict(state='eat', puff=1., t=3., p=.08)); FE.gag(d, 'vegeta', 900, 250, 105, True, dict(state='eat', puff=1., t=2., p=.30)); FE.gag(d, 'goku', 1420, 250, 105, False, dict(state='stop', puff=1., t=3.))
        d2 = D(lw=4.6, line=ST.LINE); FE.horror(d2, 'goku', 260, 1040, 92, False, 2); FE.horror(d2, 'vegeta', 620, 1040, 92, True, 5)
        d3 = D(lw=4.2, line=FE.NIGHT['line']); FE.sleep_goku(d3, 1120, 820, 105, 6)
        im = comp.over(comp.over(im, d.layer()), d2.layer()); im = comp.over(im, d3.layer(grain=.03)); comp.save('qa/chibi/_gags.jpg', im, 92)

    if k == 'walkfinal':
        im = canvas('#f6e6b8'); d = D(lw=2.0, line=ST.LINE); who = ['roshi', 'krillin', 'goku', 'gohan', 'bulma', 'vegeta', 'piccolo']; H_ = 1080 / 7.; S0 = 33.; V = 2.45
        for j, w in enumerate(who):
            gy = (j + 1) * H_ - 9; d.fl.fill(A([[0, gy], [1920, gy], [1920, gy + 2], [0, gy + 2]]), '#c9a46a')
            for i in range(12):
                f = i * 2 - (0 if i < 20 else 0); P = ST.march(w, f, S0, V, off=.25 if w == 'vegeta' else 0.); x = 86 + i * 157 
                for q in range(4): xx = x - 60 + ((q * 34 - f * V) % 136); d.fl.fill(A([[xx, gy + 2], [xx + 3, gy + 2], [xx + 3, gy + 8], [xx, gy + 8]]), '#a97a4c')
                ST.fig(d, w, x, gy, S0 * (.86 if w == 'piccolo' else 1.), P)
        im = comp.over(im, d.layer(grain=.02)); comp.save('qa/chibi/walk.jpg', im, 93)
