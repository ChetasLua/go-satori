"""CHIBI lane: one door to both families.
STORYBOOK (soft brown line; b1, e3, h4)
    d = D(lw=4.8, line=LINE);  P = march('goku', f, S, V, off=0.)      walk pose whose feet stick to ground moving V px per frame (20 f per cycle, contacts on beats 0 and 10)
    fig(d, 'goku', x, ground_y, S, P)        who: roshi krillin goku gohan bulma vegeta piccolo.  S = head half-width px (parade 108).  L = d.layer(grain=.03)
      P fields you may override: yaw (-.9..9, + faces right), pitch, roll, lean, hip (dx, dy), fn/ff (ankles), handN/handF (torso units), hkN/hkF fist|open|flat|grip, armTop,
      crossed (Vegeta), staff (Roshi tip), lag (hair drag), ex = dict(eyes dot|happy|sleep|line|blank, mouth open|sing|laugh|smile|flat|frown|pout|o|scream, o=, blink=1, brow=, look=)
    fig(...) returns dict(n, t, h, local): t(pts) maps torso units to px (attach props).  shadow(d, x, gy, S).  ball4(d, node, r, n_star, spin)
    shenron(d, path(f, x0, y0), S, f)        living ribbon dragon.  bounce_ball(d, x, gy, r, f, ...)
    Parade().setup(); .frame(f)              the b1 staging (chibi_parade.py); plates in chibi_world.py (sky, clouds, sea + island, jetty + beach, gull, crab, reflect)
STICKERS (bold line; b2):  sticker(who, f, cx, cy, S, blink=0) die-cut layer;  scene(who, f) full frame;  grid(f, blink) the sheet.  NAMES = order
COMEDY (b4, e3):  Feast().setup(); .frame_standoff(t);  gag(d, who, x, y, S, flip, dict(state='eat'|'stop'|'glare', p=, puff=));  horror(d, who, x, gy, S, flip, t);  stack, bowl, bun, steam, halo, sparks, lamp, light, amber
    sleep_goku(d, x, y, S, t, foot=, bubble=, smile=)"""
from .chibi_core import D, N, Hd, E, Eo, Ei, back, keys, ell, spl, tube, blade2, star, mix, ik, tooth
from .chibi_story import LINE, fig, walk, march, shadow, ball4, hand, PERS, HAIR, speed, foot_at
from .chibi_dragon import shenron, path, bounce_ball
from .chibi_parade import Parade, FINISH
from .chibi_stick import sticker, scene, grid, act, NAMES, GRID, BG, CARD, diecut, puff
from .chibi_feast import Feast, gag, horror, sleep_goku, bowl, stack, bun, steam, halo, sparks, lamp, light, amber, chopsticks, leftovers
