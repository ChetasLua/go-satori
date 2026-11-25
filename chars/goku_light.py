"""Light presets for Goku.  side +1 = key from the viewer's right.  amt: -1 no side shadow, 0 side plane, 1 half the face.
ldir = direction TO the light in view space (x right, y up, z to the viewer).  hdir = the same for the hair if it differs."""
import numpy as np


def _h(root, base, light, hi, shade, deep, line, **k):
    d = dict(root=root, base=base, light=light, hi=hi, shade=shade, deep=deep, line=line); d['dark'] = int(base[1:3], 16) < 70; d.update(k); return d


# hair colour by mood
BLACK = _h('#12111a', '#191824', '#2b2c40', '#565b7a', '#100f18', '#0a0910', '#050508', inmix=.0, hl_thr=.35)
GOLD = _h('#e2982c', '#fbcf4a', '#ffe88a', '#fffbe2', '#d5862a', '#9a5318', '#4a2008', hl_thr=.18)
SILVER = _h('#8b93a8', '#c6cedd', '#e6ebf4', '#ffffff', '#7a8298', '#3d4254', '#14151f', hl_thr=.1)


def _l(**k): return k


LIGHT = dict(
    day=_l(skin=('#f7cda6', '#d8926c', '#ac6448', '#ffe9cf'), blush='#e8805e', white=('#fdfaf4', '#c9c2cc'), hair=BLACK, brow='#14121a', under=('#2b4f9c', '#17305f', '#4c79c8'),
           gi=('#f4832a', '#c2521c', '#ffae58'), line='#1a1015', mouth=('#7c2026', '#d06a66', '#3c0d13'), teeth=('#fdf8ee', '#d4c6c2'), disc='#f6f2ea',
           side=1, amt=-.35, ldir=(.45, .75, .50), nose=.55, rim=[], para=[dict(p0=(0, -1.8), p1=(0, 2.4), stops=[(0, '#fff6e4'), (.5, '#ffffff'), (1, '#d9c6cc')], amt=.6)]),
    lamp=_l(skin=('#ffc488', '#c4743c', '#83401f', '#ffe8b8'), blush='#f0743c', white=('#fff0d2', '#c79b6a'), hair=_h('#150d10', '#20151a', '#4c2c22', '#b0683a', '#150c10', '#0c0709', '#060304', hl_thr=.35), brow='#170c0c', under=('#3a4174', '#1f2248', '#7a6a72'),
            gi=('#f07a20', '#a63f14', '#ffb450'), line='#20100c', mouth=('#6e1a18', '#c85c44', '#360a0a'), teeth=('#fff0cc', '#cca878'), disc='#ffeccb',
            side=1, amt=.0, ldir=(.35, .85, .35), nose=.8, rim=[dict(d=(-7, 5), col='#ffd890', amt=.8)], para=[dict(p0=(.4, -1.8), p1=(-.2, 2.2), stops=[(0, '#fff0cc'), (.4, '#ffffff'), (1, '#a8707a')], amt=.9)]),
    moon=_l(skin=('#a58abf', '#584184', '#2c1e55', '#e6d2ff'), blush='#8a5a9e', white=('#eee4ff', '#8f7cc0'), hair=_h('#0b0814', '#100c1e', '#2a1f52', '#9468e0', '#0a0714', '#06040c', '#030206', hl_thr=.25), brow='#0d0916', under=('#1d2560', '#10123a', '#5a4ab0'),
            gi=('#8a4860', '#472650', '#c07ac0'), line='#0a0614', mouth=('#3a1238', '#84487e', '#1c0820'), teeth=('#e8dcf6', '#9a88c4'), disc='#cfc0ea',
            side=1, amt=.30, ldir=(.70, .62, .1), nose=1., rim=[dict(d=(-9, 4), col='#c784ff', amt=1., over=True)], para=[dict(p0=(1.2, -1.4), p1=(-1., 2.2), stops=[(0, '#f2e4ff'), (.45, '#dcd0f4'), (1, '#57468c')], amt=.9)]),
    rain=_l(skin=('#b9aaa2', '#796c70', '#48424c', '#e6e0dc'), blush='#a0706c', white=('#e8e8ea', '#9a9aa6'), hair=_h('#0c0d12', '#131419', '#2b2f39', '#8f9aa8', '#0c0c11', '#07070a', '#030304', hl_thr=.22), brow='#101015', under=('#2c3f66', '#182440', '#5c7098'),
            gi=('#a8643c', '#6c3c2a', '#d09468'), line='#121016', mouth=('#472026', '#8c565c', '#22090e'), teeth=('#e2e0de', '#a6a2a6'), disc='#cfcfcd',
            side=-1, amt=.20, ldir=(-.35, .90, .12), nose=.9, rim=[dict(d=(5, 7), col='#e2eaf2', amt=.8)], para=[dict(p0=(-.4, -1.8), p1=(.4, 2.2), stops=[(0, '#f4f7fa'), (.45, '#e0e3e8'), (1, '#75788a')], amt=1.)]),
    gold=_l(skin=('#ffe2a4', '#e2923c', '#a1561c', '#fffbe4'), blush='#ff8a3c', white=('#fffdf0', '#e8cf9a'), hair=GOLD, brow='#c07a1c', under=('#4a5a90', '#25386a', '#b8b0a0'),
            gi=('#ff9a30', '#c85a14', '#ffd27a'), line='#3a1808', mouth=('#7a1e14', '#d8663c', '#38080a'), teeth=('#fffdf0', '#e6cfa0'), disc='#fff6da',
            side=1, amt=-.05, ldir=(.30, .80, .45), hdir=(.45, .30, .75), nose=.7, rim=[dict(d=(-7, 5), col='#fffbe0', amt=1., over=True), dict(d=(7, 4), col='#ffe27a', amt=.9)], para=[dict(p0=(0, -1.8), p1=(0, 2.4), stops=[(0, '#fffbe8'), (.5, '#ffffff'), (1, '#e8b070')], amt=.7)]),
    ui=_l(skin=('#efc3a2', '#5d352a', '#241311', '#fff4ea'), blush='#b8604a', white=('#f4f8ff', '#5a5264'), hair=SILVER, brow='#2a2630', under=('#24365f', '#10182f', '#6d86b8'),
          gi=('#b8602a', '#4a2416', '#f0a868'), line='#0d090c', mouth=('#3c1216', '#8a4a44', '#1c070a'), teeth=('#eef0f6', '#8c8894'), disc='#cfd6e4',
          side=1, amt=.92, ldir=(.85, .42, .30), nose=1., rim=[dict(d=(-6, 3), col='#e9f3ff', amt=1., over=True)], para=[dict(p0=(1.3, -.6), p1=(-1.2, .6), stops=[(0, '#ffffff'), (.5, '#e8e0e4'), (1, '#5a4c5c')], amt=.9)]),
    dawn=_l(skin=('#ffd0ac', '#cf8f80', '#966064', '#fff0dc'), blush='#f08a74', white=('#fff8f0', '#cdb8c4'), hair=_h('#17131c', '#211b28', '#3d3448', '#b08a92', '#15111a', '#0d0a11', '#060509', hl_thr=.32), brow='#1a141c', under=('#3a5290', '#23345f', '#7f9acb'),
            gi=('#f28a4a', '#b4563c', '#ffc08a'), line='#1d1218', mouth=('#742630', '#cc7070', '#380e16'), teeth=('#fff8f0', '#d2bcc0'), disc='#fbf0e8',
            side=1, amt=.05, ldir=(.80, .48, .30), nose=.7, rim=[dict(d=(-8, 3), col='#fff0c8', amt=1.)], para=[dict(p0=(1.4, -1.2), p1=(-1.2, 2.), stops=[(0, '#fff0de'), (.45, '#ffffff'), (1, '#a6a8d0')], amt=.8)]),
)
IRIS = dict(base=('#0b0a0d', '#1c181c', '#3a3238'), ssj=('#0a5f5a', '#19b5a4', '#a6f5e2'), ui=('#5d6678', '#b9c4d8', '#ffffff'))


def light(l, side=None, hair=None, **over):
    """light preset (name or dict); side=-1/+1 mirrors the set-up; hair='gold'|'silver'|'black'|dict swaps the hair colours"""
    L = dict(LIGHT[l] if isinstance(l, str) else l)
    if side is not None and side != L['side']:
        L['side'] = side
        for k in ('ldir', 'hdir'):
            if k in L: L[k] = (-L[k][0],) + tuple(L[k][1:])
        L['rim'] = [dict(r, d=(-r['d'][0], r['d'][1])) for r in L['rim']]
        L['para'] = [dict(q, p0=(-q['p0'][0], q['p0'][1]), p1=(-q['p1'][0], q['p1'][1])) for q in L['para']]
    if hair is not None: L['hair'] = dict(dict(gold=GOLD, silver=SILVER, black=BLACK)[hair] if isinstance(hair, str) else hair)
    L.update(over); return L
