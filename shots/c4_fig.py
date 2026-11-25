"""c4: Piccolo wraps himself round Gohan, seen from behind the child (same axis as the wide).  Keys: k 0 arms open, cape swinging round; .55 closing; 1 shut, hands in cloth and hair; tuck = head pressed down on the child's.
    lay = hug(cx, cy, sc, close=1., tuck=1., spread=1., swing=0., rip=0., turban=True, f=0, light='blast')   -> premultiplied RGBA 1080x1920.  (cx, cy) = the child's head centre, sc = px per Piccolo head unit"""
import numpy as np, math
from chars import fkit as K, piccolo as PI, gohan as G
from chars.fkit import A, cloth, tone, mixc, lw, grade, lover, limb, capsule, outline, Cel, spl, Body, light as light_, hand, band, hair_group, scalp, TAU, W, FH, cv2, f32, shifted
from db import hands as HD

CLING = HD.pose(index=(38, 62, 34, 7, 0), middle=(44, 66, 36, 1, 0), ring=(50, 68, 38, -5, 0), little=(56, 66, 38, -12, 0), thumb=(30, 34, 22, 26, 24), wrist=(8, 4), cup=.3)       # fingers hooked into cloth
HOLD = HD.pose(index=(16, 26, 18, 9, 0), middle=(18, 30, 18, 2, 0), ring=(20, 32, 20, -5, 0), little=(24, 32, 20, -13, 0), thumb=(44, 20, 18, 14, 14), wrist=(14, 0), cup=.35)          # spread on the back of a head
OPEN = HD.pose(index=(6, 14, 8, 10, 0), middle=(4, 12, 8, 2, 0), ring=(6, 14, 8, -6, 0), little=(8, 18, 10, -15, 0), thumb=(52, 22, 16, 8, 8), wrist=(-6, 0), cup=.15)
L2 = lambda a, b, t: A(a, float) * (1 - t) + A(b, float) * t


def forearm(cel, L, sh, el, wr, u, w, sg, seed=0, part='UF'):
    """a real arm: upper arm, point of the elbow, forearm with its muscle belly near the elbow and flat tendon end; the pink patch follows the muscle"""
    sk, pk, ln = cloth(L, PI.SKIN), cloth(L, PI.PINK), tone(L, PI.PINKL, 0); sh, el, wr = [A(v, float) for v in (sh, el, wr)]
    d1, d2 = (el - sh) / np.hypot(*(el - sh)), (wr - el) / np.hypot(*(wr - el)); bis = d1 - d2; bis = bis / (np.hypot(*bis) + 1e-6); n2 = A([-d2[1], d2[0]]); out = 1 if n2 @ bis > 0 else -1; n2 = n2 * out       # n2 = extensor (elbow) side
    U = limb([sh, L2(sh, el, .35), L2(sh, el, .68), el - d1 * .02 * u], [.40 * u, .425 * u, .36 * u, .27 * u])[0]; l = np.hypot(*(wr - el)); t = np.array([0, .10, .24, .42, .62, .82, 1.]); ho = np.array([.24, .315, .355, .325, .255, .20, .185]) * u; hi = np.array([.22, .285, .315, .290, .235, .19, .180]) * u; bow = -.05 * l * np.sin(np.pi * t)
    C = el + d2 * (t * l)[:, None] + n2 * bow[:, None]; Eo, Ei = spl(C + n2 * ho[:, None], n=9), spl(C - n2 * hi[:, None], n=9); tip = el + bis * .30 * u - d2 * .05 * u; F = np.vstack([spl(A([Ei[0], el - d2 * .20 * u - n2 * .10 * u, tip, el + n2 * .30 * u + d2 * .02 * u, Eo[0]]), n=6)[:-1], Eo, Ei[::-1]])
    parts = [q for q, n_ in ((U, 'U'), (F, 'F')) if n_ in part]
    for q in parts: outline(cel, q, w[0] * 1.8, sk[4], seed, tremor=.12)
    for q in parts: cel.fill(q, sk[0])
    # hard shadow: upper arm in two planes, forearm under the muscle edge
    if 'U' in part: cel.clip(U); a, b = limb([sh, L2(sh, el, .35), L2(sh, el, .68), el], [.40 * u, .425 * u, .36 * u, .27 * u])[1:3]; E = a if (a - b)[len(a) // 2, 0] * sg >= 0 else b; c_ = (a + b) / 2; n = len(E); k = np.interp(np.linspace(0, 1, n), [0, .25, .42, .6, .8, 1], [.05, .40, .12, .34, .02, .10])[:, None]; cel.fill(np.vstack([c_ + (E - c_) * k, (c_ + (E - c_) * 1.4)[::-1]]), sk[1]); cel.unclip()
    if 'F' not in part: return d2, .80 * u
    side = Ei if Ei[len(Ei) // 2, 1] > Eo[len(Eo) // 2, 1] else Eo; mid = (Eo + Ei) / 2; n = len(side); k = np.interp(np.linspace(0, 1, n), [0, .16, .30, .48, .60, .80, 1], [.30, .52, .28, .50, .26, .34, .18])[:, None]
    cel.clip(F); cel.fill(np.vstack([mid + (side - mid) * k, (mid + (side - mid) * 1.4)[::-1]]), sk[1])
    # the muscle patch: a leaf that swells over the belly
    s_ = np.linspace(.10, .66, 22); v = (s_ - .10) / .56; Mx = lambda q: np.stack([np.interp(q, np.linspace(0, 1, n), mid[:, i]) for i in (0, 1)], -1); Hx = np.stack([np.interp(s_, np.linspace(0, 1, n), (Eo - mid)[:, i]) for i in (0, 1)], -1); lf = np.sin(np.pi * v ** .8) ** .55; cen = Mx(s_) + Hx * .16; P = np.vstack([cen + Hx * (lf * .62)[:, None], (cen - Hx * (lf * .52)[:, None])[::-1]]); cel.fill(P, pk[0]); cel.clip(P); low = -1 if (Hx[10, 1] > 0) is False else 1; cel.fill(np.vstack([cen + Hx * (lf * (.02 if low > 0 else -.05))[:, None], (cen + Hx * (low * lf * 1.2)[:, None])[::-1]]), pk[1])
    for q in (.20, .36, .52, .68, .83): i = int(q * 21); c0, c1 = cen[i] + Hx[i] * lf[i] * .58, cen[i] - Hx[i] * lf[i] * .48; cel.line(spl(A([c0, (c0 + c1) / 2 + d2 * .07 * u, c1]), n=6), w[2] * 1.2, ln, taper=(.15, .35), seed=seed + i)
    cel.unclip(); cel.line(np.vstack([P, P[:1]]), w[1] * .9, ln, taper=(.02, .02), press=.25, seed=seed + 7); cel.unclip()
    cel.line(spl(A([el + bis * .12 * u + d1 * .02 * u, el + bis * .20 * u, tip - bis * .06 * u]), n=5), w[1], sk[4], taper=(.2, .6), seed=seed + 3); cel.line(spl(A([Mx(.74), Mx(.86) + n2 * .03 * u, Mx(.97) + n2 * .02 * u]), n=5), w[2], sk[4], taper=(.3, .6), seed=seed + 4)      # elbow point, tendon
    band(cel, wr - d2 * .06 * u, -d2, .215 * u, .40 * u, cloth(L, PI.BAND), w, seed + 5, sg=sg); return d2, .80 * u


def wings(cel, L, B, u, wl, f, spread, swing, rip, seed):
    """the cape as a wall behind them: two wings, each its own wave; swing throws the trailing wing round; torn hem"""
    cp = cloth(L, PI.CAPE, PI.CAPEH); r = np.random.default_rng(seed); g = lambda j: .12 * math.sin(f * .9 + j * 1.7)
    for q in (-1, 1):
        k = spread * (1. + .34 * swing * q) * (1.06 if q > 0 else .96); up = .55 * swing * (1 if q > 0 else -.6); E_ = [(q * .45, -.55), (q * 1.5 * k, -1.30 - up * .4 + g(1)), (q * 3.0 * k, -1.70 - up + g(2)), (q * 4.5 * k, -1.30 - up * 1.4 + g(3)), (q * 5.4 * k, -.25 - up * 1.2 + g(4))]; n = 9; edge = []
        for j in range(n):
            v = j / (n - 1); bx = q * (5.4 - 3.4 * v ** 1.15) * k; by = .0 + 4.9 * v ** .92 - up * (1 - v); edge.append((bx + q * .17 * math.sin(j * 2.1 + f * .8), by + g(j + 5)))
            if rip > 0 and j < n - 1: edge.append((bx - q * (.45 + 1.8 * rip * r.uniform(.5, 1.)) * k, by + .30 + .25 * r.random()))
        P = E_ + edge + [(q * .9, 5.2), (q * .4, 3.0), (q * .4, .5)]; nE = len(E_); cor = [0, nE - 1] + (list(range(nE, nE + len(edge))) if rip > 0 else []) + [len(P) - 3]; C = spl(B(P, z=-.4), True, corners=cor, n=7); outline(cel, C, wl[0] * 1.7, cp[4], seed + q, tremor=.15); cel.fill(C, mixc(cp[0], cp[1], .35)); cel.clip(C)
        for j, (a0, l_) in enumerate(((-1.0, 5.4), (-.62, 5.6), (-.30, 5.6), (.02, 5.5), (.34, 5.3), (.66, 5.2), (1.0, 5.1))):
            d = A([q * math.cos(a0 * .9 + .12) * k, math.sin(a0 * .9 + .12) - up * .15]); p0 = A([q * .95, -.10]); Q = [p0 + d * .6, p0 + d * l_ * .45 + A([0, .22 * (-1) ** j + .06 * math.sin(f * 1.1 + j)]), p0 + d * l_ * .98]; c = spl(B(Q, z=-.4), n=9); cel.fill(K.wedge(c[::-1], lambda t: u * (.75 * (1 - t) ** .9) * (1 if j % 2 else .62)) + A([0, .14 * u]), cp[1] if j % 2 else mixc(cp[1], cp[2], .55)); cel.line(c, wl[1] * 1.1, cp[4], taper=(.4, .15), press=.3, seed=seed + j)
        cel.fill(spl(B([(q * .2, -.6), (q * 1.4, -.4), (q * 1.9, .6), (q * 1.7, 2.0), (q * 2.1, 3.4), (q * 1.6, 5.6), (0, 5.6)], z=-.4), True, n=7), mixc(cp[1], cp[2], .6)); cel.unclip()


def child(cx, cy, s, L, lname, f, wind, flinch, seed, pull=None, u=100.):
    """Gohan from behind"""
    H = G.Gohan.make(cx, cy, s, math.pi + .10, -.10 - .16 * flinch, -.06); H = K.FHead.of(H, G.D, 0.); wl = lw(s); cel = Cel(W, FH); o = A([cx - .04 * s, cy + 1.28 * s]); B = Body(o[0], o[1], s, .06, 0., -.03); gg = G.Gohan
    for q in (-1, 1): gg.put_arm(cel, B, L, q, (q * 1.10, .98, -.1), (q * .45, 1.0, -.6), HD.POSES['fist'], HD.euler(0, 0, 0), wl, seed + q, hand_=False)
    gg.torso(cel, B, L, wl, seed, 4.6); g = cloth(L, G.GI, G.GIH); V = lambda P: spl(B(A(P, float)), n=8)
    for j, pts in enumerate(([(.0, .55), (.02, 1.0), (-.03, 1.18)], [(-.52, .50), (-.36, .72), (-.18, .86)], [(.52, .52), (.40, .74), (.22, .90)])): cel.line(V(pts), wl[1], g[4], taper=(.15, .5), press=.3, seed=seed + j)
    if pull is not None:                                      # cloth pulled into the fist: hard wedges of shadow run to the fingertips
        cel.clip(V([(-.92, .18), (.92, .18), (.80, 1.20), (1.0, 2.2), (1.3, 4.5), (-1.3, 4.5), (-1.0, 2.2), (-.80, 1.20)]))
        for j, an in enumerate((-.75, -.25, .35, .95, 1.45, 2.0, 2.6)): d = A([math.cos(an), math.sin(an)]); n_ = A([-d[1], d[0]]); p1 = pull + d * u * (.80 + .32 * (j % 3)); m = (pull + p1) / 2 + n_ * .06 * u * (-1) ** j; cel.fill(np.vstack([pull, m + n_ * .11 * u, p1, m]), g[1]); cel.line(spl(A([pull, m, p1]), n=6), wl[1], g[4], taper=(.05, .7), press=.3, seed=seed + j)
        cel.unclip()
    nk = cloth(L, G.SKIN); cel.fill(spl(A([(cx - .34 * s, cy + .55 * s), (cx + .30 * s, cy + .55 * s), (cx + .34 * s, cy + 1.10 * s), (cx + .50 * s, cy + 1.45 * s), (cx - .56 * s, cy + 1.45 * s), (cx - .38 * s, cy + 1.10 * s)]), True, n=6), nk[1]); cel.line(A([(cx - .34 * s, cy + .6 * s), (cx - .37 * s, cy + 1.0 * s), (cx - .46 * s, cy + 1.3 * s)]), wl[0] * .8, nk[4], taper=(.3, .2), seed=seed); cel.line(A([(cx + .30 * s, cy + .6 * s), (cx + .33 * s, cy + 1.0 * s), (cx + .42 * s, cy + 1.3 * s)]), wl[0] * .8, nk[4], taper=(.3, .2), seed=seed + 1)
    gg.scarf(cel, B, L, wl, seed, wind, f); LK = sorted(G.build_locks(H, f, wind, False), key=lambda l: l.depth); hc = G.hair_cols(L); hair_group(cel, [l for l in LK if l.tag != 'fr'], hc, wl, seed, L['ld'], caps=[scalp(H, lambda t: -.46, 1.07)], line=L['line']); return cel, B, H


def hug(cx, cy, sc, close=1., tuck=1., spread=1., swing=0., rip=0., turban=True, f=0, light='blast', seed=2, flinch=1., expr=None, press=0., lift=0., w=W, h=FH):
    """see the module line.  press 0..1 = how hard he pulls the child in (late in the shot)"""
    if expr is None: expr = 'shield' if tuck < .6 else dict(PI.E['shield'], closed=1, arc=0, dB=-.03)
    L = light_(light); wl = lw(sc); sg = -1 if L['ld'][0] > 0 else 1; s_ch = sc * .80; u = sc; e = close * close * (3 - 2 * close)
    hx = cx + (14 * (1 - tuck) - .16 + .10 * press) * 1. - .06 * sc; hy = cy - (1.62 - .22 * tuck - .16 * press) * sc - .42 * (1 - tuck) * sc; op = max(0., 1 - tuck); H = PI.Piccolo.make(hx - .25 * sc * op, hy + .18 * sc * op, sc, .22 - .16 * tuck, -.20 - .16 * tuck - .10 * press, .05 + .07 * tuck + .03 * press - .20 * op); ox, oy = cx - .02 * sc, cy + .10 * sc - .42 * (1 - tuck) * sc + .06 * press * sc; B = Body(ox - .2 * sc * op, oy, u, .05 + .12 * op, 0., .03 + .02 * press - .15 * op)
    def G_(c, rw=1., rims=True): return grade(c.layer(h, w), L, (cx - 5 * sc, cy - 3 * sc, cx + 5 * sc, cy + 4 * sc), max(.5, sc / 150) * rw, rims=rims)
    def put(lay, al, dy, sof, amt): a = al[..., 3]; lay[..., :3] *= 1 - (np.clip(cv2.GaussianBlur(shifted(a, 0, dy * sc), (0, 0), sof * sc), 0, 1) * (1 - a))[..., None] * amt; return lover(lay, al)
    cel = Cel(w, h); wings(cel, L, B, u, wl, f, spread, swing, rip, seed); PI.Piccolo.neck(cel, H, L, seed); PI.Piccolo.torso(cel, B, L, wl, seed, 5.2); lay = G_(cel)
    cy_ = Cel(w, h); PI.Piccolo.yoke(cy_, B, L, wl, seed, .18 + .26 * tuck + .10 * press + .05 * math.sin(f * .9) + lift, torn=rip); Y = G_(cy_); lay = put(lay, Y, .10, .08, .4)
    c2 = Cel(w, h); PI.Piccolo.head(c2, H, dict(expr=expr, light=light, f=f, seed=seed, turban=turban, part='all')); lay = put(lay, G_(c2), .10, .08, .34)      # his head goes down behind the child's hair
    sw = math.sin(math.pi * close) * .6 * sc; R_ = dict(sh=B([(-1.42, .46, .1)])[0], el=L2((cx - 3.25 * sc, cy + 1.35 * sc), (cx - 2.30 * sc, cy + 1.86 * sc + .06 * sc * press), e), wr=L2((cx - 2.95 * sc, cy + 3.0 * sc), (cx - .98 * sc, cy + 1.66 * sc), e) + A([0, sw])); L_ = dict(sh=B([(1.42, .46, .1)])[0], el=L2((cx + 3.30 * sc, cy + .95 * sc), (cx + 2.34 * sc, cy + 1.30 * sc), e), wr=L2((cx + 3.05 * sc, cy - .95 * sc), (cx + 1.12 * sc, cy + .44 * sc + .05 * sc * press), e) + A([.2 * sw, -.4 * sw]))
    pull = None if e < .6 else R_['wr'] + A([1.12, .04]) * u
    c3, Bc, Hc = child(cx, cy + .06 * sc * press, s_ch, L, light, f, (-.25, .05), flinch, seed + 3, pull, u); lay = put(lay, G_(c3, .55), .06, .10, .45)
    bld = HD.build(len=1.06, wid=1.04, boxy=.5, tip=.2, knuckle=1.1); st = dict(nails='claw', claw=.05, arm=.30); nl = ('#f2efe0', '#c8c4b0', '#8a8674'); yy, xx = np.mgrid[0:h, 0:w].astype(f32)
    for q, a, side in ((-1, R_, 'R'), (1, L_, 'L')):
        c0 = Cel(w, h); forearm(c0, L, a['sh'], a['el'], a['wr'], u, wl, sg, seed + 3 + q, 'U'); au = G_(c0); m = np.clip((1.0 * u - np.hypot(xx - a['sh'][0], yy - a['sh'][1])) / (.12 * u), 0, 1) * Y[..., 3]; au = au * (1 - m)[..., None]; lay = put(lay, au, .10, .08, .42)      # the pad of the cape covers the shoulder
        c = Cel(w, h); d2, hu = forearm(c, L, a['sh'], a['el'], a['wr'], u, wl, sg, seed + 3 + q, 'F'); ang = math.degrees(math.atan2(-d2[1], d2[0])) - 90; hp = HD.blend(OPEN, CLING if q < 0 else HOLD, e); roll = ang + (6 if q < 0 else 4) * e
        hand(c, L, PI.SKIN, hp, a['wr'] + d2 * .02 * u, hu, HD.euler(0, (10 if q < 0 else 16) * e, roll), side, bld, seed=seed + q, style=st, nail=nl); al = G_(c); lay = put(lay, al, .10, .08, .42)
        if q > 0 and e > .97:                                  # fingertips sink: a few tufts close over them
            c = Cel(w, h); hc = G.hair_cols(L); n2 = A([-d2[1], d2[0]]); tp = a['wr'] + d2 * 1.92 * hu
            for j in range(6): x0 = tp + n2 * (-.50 + .20 * j) * u + d2 * (.10 * (j % 2)) * u; t_ = x0 - d2 * (.30 + .10 * (j % 3)) * u + n2 * .06 * (-1) ** j * u; P = kit_spike(x0 - n2 * .11 * u, t_, x0 + n2 * .11 * u); outline(c, P, wl[1] * 1.2, L['line'], seed + j); c.fill(P, hc['base']); c.fill(np.vstack([P[: len(P) // 2], [x0]]), hc['sh'])
            lay = lover(lay, G_(c, rims=False))
    return lay


from db.kit import spike as kit_spike


if __name__ == '__main__':
    from db import comp; from db.paint import rgb; import sys
    bg = np.ones((FH, W, 3), f32) * rgb('#a87af0'); out = []
    for kw in (dict(close=.0, tuck=.0, spread=.62, swing=1., flinch=.2), dict(close=.55, tuck=.5, spread=.85, swing=.5, flinch=.6), dict(close=1., tuck=1., spread=1.), dict(close=1., tuck=1., spread=1., rip=.8, turban=False, press=1., seed=5)): out.append(cv2.resize(comp.over(bg, hug(905, 610, 165, **kw)), (960, 540), interpolation=cv2.INTER_AREA))
    comp.save('qa/phys/_hug.jpg', np.vstack([np.hstack(out[:2]), np.hstack(out[2:])]))
