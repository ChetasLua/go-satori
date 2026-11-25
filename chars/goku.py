"""GOKU (adult).

    from chars import goku
    H = goku.GHead().pose(cx, cy, s, yaw, pitch, roll)          # eye line centre, s = px eye line -> chin
    layer = goku.bust(H, 'serious', 'day', hair='base', ...)     # premultiplied RGBA: hair, head, neck, shoulders, gi, rim light, gradients
    goku.head(cel, H, expr, light, hair=...)                     # head only on your own Cel (part='back'|'front'|'all')
    goku.head_fn(cel, H, info)                                   # for db/hero3 (info: part expr hair light f seed wind wet dmg)
expr: calm smile grin laugh serious determined smirk shock fear grief griefhid shut serene ui rage grit blank scream_a scream_b scream scream_d roar, (a,b,t), or dict
hair: base s1..s5 ssj ui (colour follows: black / gold / silver; s1..s5 stay black unless hcol='gold')      light: day lamp moon rain gold ui dawn
"""
import numpy as np, math, cv2
from db import W, H as HH, comp
from db.ink import Cel
from db.kit import rim as rim_, para, tint
from db.paint import rgb, f32, smooth
from . import goku_hair as GH, goku_light as GL, goku_head as GF
from .goku_head import GHead, getx, blend, lids, EXPR, Face
from .goku_light import light as light_, LIGHT
from .goku_hair import mixc
A = np.array


def setup(H, expr, light, hair='base', hcol=None, eyes=None, side=None, wet=0., **over):
    X = getx(expr); st = hair if isinstance(hair, str) else 'base'
    hcol = hcol or ('gold' if st == 'ssj' else 'silver' if st == 'ui' else None)
    L = light_(light, side=side, hair=hcol) if isinstance(light, (str, dict)) else light
    if hcol == 'gold': L['brow'] = '#d9a032' if light != 'gold' else L['brow']
    if hcol == 'silver': L['brow'] = '#77809a'
    L.update(over); X = dict(X); X['style'] = eyes or ('ssj' if hcol == 'gold' else 'ui' if hcol == 'silver' else X['style'])
    G = GHead.of(H, X['jaw'], X['jx']); return G, X, L


def head(cel, H, expr='calm', light='day', hair='base', hcol=None, eyes=None, f=0, wind=(0, 0), wet=0., dmg=0., seed=0, part='all', side=None, painterly=False, lines=1., extra=(), drop=(), hair_obj=None, neckless=True, **over):
    G, X, L = setup(H, expr, light, hair, hcol, eyes, side, wet, **over)
    hr = hair_obj or GH.Hair(G, hair, L, f=f, wind=wind, wet=wet, seed=seed, painterly=painterly, extra=extra, drop=drop)
    F = Face(cel, G, X, L, seed, hr, dmg, wet, lines, painterly)
    if part in ('all', 'back'): hr.back(cel)
    if part in ('all', 'front'):
        d = 1 if G.yaw > 0 else -1; ay = abs(G.yaw)
        if ay > 1.80: F.ears([-d, d]); hr.front(cel); return F, hr                      # from behind: no face
        F.ears([q for q in (-1, 1) if ay < .12 or q == d]); F.base(); F.contour()
        F.ears([-d] if ay >= .12 else [])
        F.features(); hr.front(cel)
    return F, hr


def head_fn(cel, H, info):
    g = info.get
    head(cel, H, g('expr', 'calm'), g('light', 'day'), g('hair', 'base'), g('hcol'), g('eyes'), g('f', 0), g('wind', 0.) if np.ndim(g('wind', 0.)) else (g('wind', 0.), 0.), g('wet', 0.), g('dmg', 0.), g('seed', 0), g('part', 'all'), g('side'))


def bust(H, expr='calm', light='day', hair='base', hcol=None, eyes=None, f=0, wind=(0, 0), wet=0., dmg=0., seed=0, side=None, wear='gi', byaw=None, bpitch=0., broll=0., raise_=0., breath=0., painterly=False, lines=1., extra=(), drop=(),
         w=W, h=HH, rims=True, body=True, shift=(0, 0), part='all', hair_obj=None, cel=None, light_pass=True, **over):
    """the whole head and shoulders -> premultiplied RGBA layer (w x h)"""
    G, X, L = setup(H, expr, light, hair, hcol, eyes, side, wet, **over)
    c = cel or Cel(w, h); hr = hair_obj or GH.Hair(G, hair, L, f=f, wind=wind, wet=wet, seed=seed, painterly=painterly, extra=extra, drop=drop)
    F = Face(c, G, X, L, seed, hr, dmg, wet, lines, painterly)
    d = 1 if G.yaw > 0 else -1; ay = abs(G.yaw)
    if part in ('all', 'back'): hr.back(c)
    if body: from .goku_bust import Bust; Bust(c, G, L, byaw, bpitch, broll, raise_, breath, wear, seed, lines, X['tendon'], wet, dmg, shift).paint()
    if part in ('all', 'front'):
        (F.ears([-d, d]) if ay > 1.80 else (F.ears([q for q in (-1, 1) if ay < .12 or q == d]), F.base(), F.contour(), F.ears([-d] if ay >= .12 else []), F.features())); hr.front(c)
    a = c.rgba()
    return finish_layer(a, G, L, rims) if light_pass else a


def finish_layer(a, G, L, rims=True, k=1.):
    for p in L.get('para', []):
        q0, q1 = (G.cx + p['p0'][0] * G.s, G.cy + p['p0'][1] * G.s), (G.cx + p['p1'][0] * G.s, G.cy + p['p1'][1] * G.s); a = para(a, q0, q1, p['stops'], 'mul', p['amt'])
    if rims:
        for r in L.get('rim', []):
            m = max(.35, min(1.6, G.s / 300.)) * k; a = rim_(a, r['d'][0] * m, r['d'][1] * m, r['col'], r['amt'])
    return a
