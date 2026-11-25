"""QA harness of the ki lane (stand-in figure + test frames)"""
import sys, time, os
import numpy as np, cv2
cv2.setNumThreads(3)
from .ki_core import *
from . import kit, comp


def stand(cx=960, foot=930, hgt=640, seed=0, dark=False):
    """a quick power-up stance stand-in: cel with shadow planes (NOT a film drawing)"""
    s = hgt / 100.; X = lambda P: np.array([[cx + x * s, foot - y * s] for x, y in P]); c = Cel(); m = lambda P: [(-x, y) for x, y in P]
    gi, gis, sk, sks, bl, bls, hr, ink = ('#f47b20', '#b8481a', '#f2b98c', '#c07a58', '#23407c', '#142552', '#ffd94a', '#2b1410') if not dark else ('#1a1426',) * 8
    leg = [(3, 46), (10, 47), (15, 38), (18, 28), (17.5, 20), (19, 14), (18, 10), (12.5, 10), (11, 18), (8, 28), (2.5, 41)]; boot = [(12, 11), (18.6, 11), (19, 5), (23, 1.5), (23.5, 0), (12.5, 0), (12, 4)]
    tor = [(0, 48), (9, 47), (10, 56), (14, 66), (18, 74), (15.5, 79), (9, 81.5), (0, 82)]; arm = [(15, 78), (20, 76.5), (23, 70), (24, 62), (27, 56), (28.5, 50), (25.5, 47.5), (21.5, 53), (19, 60), (17, 68)]; fist = [(25.6, 51), (29.6, 50.5), (30.8, 46), (28.6, 42.6), (24.6, 43.4), (23.8, 47.5)]; wr = [(24, 53.2), (28.4, 54.4), (29.2, 50.4), (25.2, 49.6)]
    for P, c0, c1 in ((leg, gi, gis), (boot, bl, bls), (arm, sk, sks), (fist, sk, sks), (wr, bl, bls)):
        for q, sg in ((P, 1), (m(P), -1)):
            A = X(spl(q, True, corners=(0, len(q) // 2), n=6)); c.fill(A, c0); sh = X([(x - 2.2 * (1 if sg > 0 else 1) + (4 if sg < 0 else 0), y - .5) for x, y in q]); c.shade(A + [-5 * s * .55, 2], c1, within=A); kit.outline(c, A, 3., ink, seed)
    A = X(spl(tor + m(tor)[::-1][1:-1], True, n=6)); c.fill(A, gi); c.shade(X([(-18, 80), (-4, 70), (-9, 58), (-3, 48), (-20, 46)]), gis, within=A); c.shade(X([(0, 72), (14, 64), (12, 50), (4, 56)]), gis, within=A); kit.outline(c, A, 3., ink, 2)
    c.fill(X([(-6, 82), (0, 70), (6, 82)]), bl); c.fill(X([(-10, 51), (10, 51), (9.4, 46), (-9.4, 46)]), bl); neck = X([(-4, 88), (4, 88), (5.5, 80), (0, 77), (-5.5, 80)]); c.fill(neck, sks)
    face = X(spl([(-5.2, 96), (-5.4, 90), (-3, 85.6), (0, 84.2), (3, 85.6), (5.4, 90), (5.2, 96), (0, 99)], True, n=6)); c.fill(face, sk); c.shade(X([(-6, 98), (-1.5, 92), (-2.4, 84), (-7, 86)]), sks, within=face); kit.outline(c, face, 2.2, ink, 4)
    for sg in (-1, 1): c.fill(X([(sg * 1, 91.4), (sg * 4.6, 93.2), (sg * 4.2, 91.2), (sg * 1.4, 90.4)]), '#12343a'); c.fill(X([(sg * 1.6, 91.2), (sg * 3.8, 92), (sg * 3.4, 91), (sg * 1.8, 90.6)]), '#d8fff4')
    for a, t, b in (((-6, 92), (-13, 101), (-5, 97)), ((-6, 95), (-12, 109), (-2, 99)), ((-4, 98), (-5, 116), (2, 100)), ((0, 99), (5, 114), (5, 98)), ((3, 98), (12, 108), (6.4, 94)), ((5.4, 95), (12, 100), (6, 90.5)), ((-3.4, 99), (-2.6, 92.6), (0, 99)), ((0, 99), (2.6, 94), (3.6, 99))):
        A = X(kit.spike(a, t, b, .10, .03)); c.fill(A, hr); c.shade(A + [-s * 1.3, s * 1.0], '#e08a18' if not dark else hr, within=A); kit.outline(c, A, 2., ink, 7)
    return c.layer(), (cx, foot - 91 * s, 9 * s)


def night(seed=0, hz=700, top='#0b1512', mid='#1d3a34', low='#57705a'):
    """black-green storm field with a wet ground: quick painted ground for tests"""
    x, y = grid(); n = nimg(H, W, 260, 0, seed, 4, 4, 2.2, 1.) ; t = np.clip(y / hz, 0, 1)[..., None]; im = C(top) * (1 - t) + C(mid) * t; im = im * (1 + .35 * np.clip(n, -1, 1)[..., None]); b = (n > .18)[..., None] * C('#2b4a44') * .25; im = im + b
    g = np.clip((y - hz) / (H - hz), 0, 1)[..., None]; gr = C('#38463c') * (1 - g) + C('#0b0f0e') * g; gr = gr * (1 + .25 * nimg(H, W, 40, 0, seed + 4, 3, 2, 6., 1.)[..., None]); return np.where(y[..., None] > hz, gr, im).astype(f32)


def sheet(path, ims, cols=4, w=480):
    h = w * 9 // 16; ims = [cv2.resize(comp.to8(np.clip(i, 0, 1), 0), (w, h), interpolation=cv2.INTER_AREA) for i in ims]; rows = [np.hstack((ims[i:i + cols] + [np.zeros_like(ims[0])] * cols)[:cols]) for i in range(0, len(ims), cols)]; comp.save(path, np.vstack(rows), 90)


def fin(im, f=0, **k): return comp.finish(im, f, **dict(dict(bloom=.28, vig=.14, grain=.010), **k))


OUT = 'qa/ki/'
if __name__ == '__main__':
    import importlib
    from . import ki_aura as KA
    what = sys.argv[1]
    if what == 'aura':
        fig, face = stand(960, 1000, 560); bg = night(3, 790)
        for kind in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['gold']):
            t = time.time(); im = KA.with_aura(bg, fig, 20, kind, 1.0, 1, face); print(kind, '%.2f' % (time.time() - t)); t = time.time(); im = KA.with_aura(bg, fig, 22, kind, 1.0, 1, face); print(kind, '%.2f' % (time.time() - t)); comp.save(OUT + f'w_aura_{kind}.jpg', comp.to8(fin(im)), 92)

    if what == 'astrip':
        fig, face = stand(960, 1000, 560); bg = night(3, 830); ims = []
        for f in range(16, 32):
            ims.append(fin(KA.with_aura(bg, fig, f, sys.argv[2] if len(sys.argv) > 2 else 'gold', 1., 1, face), f))
        sheet(OUT + 'w_astrip.jpg', [i[:, 420:1500][:, :, :] if False else i for i in ims], 4, 560)
    if what == 'bolt':
        from . import ki_bolt as KB
        fig, face = stand(960, 1000, 560); bg = night(3, 830) * .8; ims = []
        st = [(2, (300, -20), (520, 850), 13., 4, dict(forks=5)), (3, (1700, -20), (1500, 840), 10., 9, dict(forks=4)), (6, (960, -20), (1180, 600), 8., 5, dict(forks=4))]
        for f in range(0, 12):
            A = KA.aura(fig, f, 'gold', 1., 1, face=face); b, fr, li = KB.arcs(fig, f, 'blue', 9, 3, face=face); l, g = KB.storm(st, f, 'blue'); im = bg if l is None else put(bg, l) + g; im = A.under(im); im = put(im, b); im = put(im, KA.lit(fig, 'gold', 1., f, 1, face)); im = A.over(im); im = put(im, fr) + li; ims.append(fin(im, f))
        sheet(OUT + 'w_bolt.jpg', ims, 3, 640); comp.save(OUT + 'w_bolt1.jpg', comp.to8(ims[4]), 92)
    if what == 'beam':
        from . import ki_beam as KB
        bg = night(3, 900) * .7
        t = time.time(); ax = KB.Axis2([(-60, 620), (2000, 420)], 60, 190); l, g = KB.beam(ax, 10, 'gold', collars=3); print('beam %.2f' % (time.time() - t))
        comp.save(OUT + 'w_beam_gold.jpg', comp.to8(fin(put(bg, l) + g)), 92)
        t = time.time(); ax = KB.Axis2([(-60, 760), (900, 560), (2000, 300)], 250, 60); l, g = KB.helix(ax, 10); print('helix %.2f' % (time.time() - t))
        comp.save(OUT + 'w_helix.jpg', comp.to8(fin(put(bg, l) + g)), 92)
        t = time.time(); m, ml = KB.muzzle(420, 560, 120, 8, 'gold', kind2='blue', back=math.pi); i, il = KB.impact_front(1400, 500, 130, 8, rot=1.3); print('ends %.2f' % (time.time() - t)); comp.save(OUT + 'w_ends.jpg', comp.to8(fin(put(put(bg, m) + ml, i) + il)), 92)
    if what == 'misc':
        from . import ki_orb as KO, ki_hit as KH, ki_beam as KB
        bg = night(3, 900) * .7; fig, face = stand(960, 1000, 560); ims = []
        T = lambda s, t0: print(s, '%.2f' % (time.time() - t0))
        t = time.time(); l, g = KO.orb(960, 540, 120, 8, 'blue'); c = KO.charge(960, 540, 120, 8); ims.append(fin(put(put(bg, c), l) + g)); T('orb', t)
        t = time.time(); l, g = KO.orb(960, 540, 12, 8, 'blue'); c = KO.charge(960, 540, 40, 8, reach=12); ims.append(fin(put(put(bg, c), l) + g)); T('pearl', t)
        t = time.time(); ims.append(fin(KB.inside(8, 960, 520, 'gold', hole=(960, 560, 150, 330)))); T('inside', t)
        t = time.time(); l, g = KB.clash(960, 540, 130, 8); ax = KB.Axis2([(-60, 620), (960, 540)], 60, 120); b, bl = KB.beam(ax, 8, 'gold'); ay = KB.Axis2([(1980, 460), (960, 540)], 60, 120); b2, bl2 = KB.beam(ay, 8, 'violet'); im = put(put(bg, b) + bl, b2) + bl2; ims.append(fin(put(im, l) + g)); T('clash', t)
        t = time.time(); sp = [(h1(i, 1) * 14 - 6, (2000, 540 + (h1(i, 2) - .5) * 1500), (-100, 560 + (h1(i, 3) - .5) * 700), 5 + 9 * h1(i, 4), 8) for i in range(220)]; l, g = KB.volley(sp, 8, 'violet', bend=KB.deflect(700, 700, 260, 420)); ims.append(fin(put(bg, l) + g)); T('volley', t)
        t = time.time(); im = put(bg, KH.ring(960, 540, 420, 26, kind='base', asp=.4, rot=.2, seed=2)); im = put(im, KH.cone((1500, 300), -.4, 900, .45, 8)); ims.append(fin(im)); T('ring+cone', t)
        for k in range(3): t = time.time(); ims.append(fin(put(bg, KH.vanish(fig, k, .0)))); T('vanish', t)
        t = time.time(); im = C('#06081a') * np.ones((H, W, 3), f32); w = KH.inkwash(fig, 4); im = put(im, w); w = KH.inkwash(np.roll(fig, 420, 1), 8); im = put(im, w); im = put(im, np.roll(fig, -420, 1)); ims.append(fin(im)); T('inkwash', t)
        t = time.time(); x, y = grid(); st = (nimg(H, W, 5, 0, 3, 1, 1) > .93)[..., None] * C('#dfe8ff'); im = C('#06081a') + st * .8 + np.exp(-((x - 1400) / 500) ** 2 - ((y - 900) / 300) ** 2)[..., None] * C('#1d4a9a'); ims.append(fin(KH.water_ring(im.astype(f32), 960, 540, 12))); T('water', t)
        t = time.time(); L, g = KB.line((200, 700), (1800, 420), 8, 0, w=7); ims.append(fin(put(bg, L) + g)); T('line', t)
        sheet(OUT + 'w_misc.jpg', ims, 3, 640)
        A = KA.aura(fig, 8, 'gold', 1., 1, face=face); b0 = A.over(put(A.under(bg), fig)); ims = []
        for n in range(8): t = time.time(); ims.append(KH.impact(b0, n, 960, 620, fig, 3, dirn=.2)); T('impact%d' % n, t)
        sheet(OUT + 'w_impact.jpg', ims, 3, 640)

    if what == 'strips':
        from . import ki; fig, face = stand(960, 1010, 560); bg = night(3, 840)
        S = [fin(ki.with_aura(bg, fig, 8 + i, 'gold', ki.envelope(i, 1, flares=[(5, .5)]), 1, face), i) for i in range(8)]; sheet(OUT + 'strip_aura.jpg', S, 2, 960)
        S = []; ax = ki.Axis2([(330, 640), (900, 560), (2050, 470)], 58, 190)
        for i in range(8):
            l, g = ki.ignite(ax, i, 2, 'blue', 4); im = bg if l is None else put(bg, l) + g; S.append(fin(im, i))
        sheet(OUT + 'strip_beam.jpg', S, 2, 960); S = []
        st = [(1, (520, -20), (760, 850), 15., 4, dict(forks=6)), (2, (1500, -20), (1320, 845), 11., 9, dict(forks=5)), (4, (1000, -20), (300, 852), 10., 15, dict(forks=5))]
        for i in range(8): b, fr, li = ki.arcs(fig, i + 4, 'blue', 8, 3, face=face); l, g = ki.storm(st, i, 'blue'); im = bg * .75; im = im if l is None else put(im, l) + g; im = put(im, b); im = put(put(im, fig), fr) + li; S.append(fin(im, i))
        sheet(OUT + 'strip_bolt.jpg', S, 2, 960)
    if what == 'final':
        from . import ki; fig, face = stand(960, 1010, 560); bg = night(3, 840); im = ki.with_aura(bg, fig, 20, 'gold', 1.15, 1, face, height=.52); b, fr, li = ki.arcs(fig, 20, 'blue', 8, 3, face=face); l, g = ki.storm([(19, (330, -20), (430, 846), 11., 4, dict(forks=6)), (18, (1640, -20), (1500, 842), 9., 9, dict(forks=5))], 20, 'blue'); im = put(im, b); im = put(im, fr) + li; comp.save(OUT + 'aura.jpg', comp.to8(fin(put(im, l * (1 - 0)) + g if False else im)), 94)
        cols = []
        for k in ('gold', 'base', 'violet', 'silver', 'goldblue'): f2, fc = stand(960, 1010, 520); cols.append(fin(ki.with_aura(night(3, 840), f2, 20, k, 1., 1, fc))[:, 640:1280])
        comp.save(OUT + 'kinds.jpg', cv2.resize(comp.to8(np.hstack(cols)), (1920, int(1080 * 1920 / 3200)), interpolation=cv2.INTER_AREA), 92)
