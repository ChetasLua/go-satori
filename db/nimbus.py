"""KINTO-UN, the golden cloud: a living painted thing.  Lobes in 3-D that breathe and roll, flat gouache tones, a coloured pen, curls, vapour shed behind.

    N = Nimbus(seed=3)
    back, front = N.draw(cx, cy, S, f, yaw=0, pitch=.12, roll=0, pal='dawn', light=(.5, .55, -.45), speed=.2)
    img = over(img, N.vapour(...)); img = over(img, back); img = over(img, rider); img = over(img, front)        # front = the near rim, so the rider sits IN it

cx, cy, S, D   same meaning as Kid.draw: (cx, cy) = the rider's PELVIS, S px per unit, D camera distance.  Cloud: about 5.4 units long, 3.4 deep, top at y = -.55 (a cross-legged boy rests on it).
yaw            0 = it flies to the RIGHT (+x), pi/2 turns its nose away from us, -pi/2 toward us.  pitch > 0 = we look down on it.  R = own 3x3.
speed 0..1.5   rolls the lobes backward, sheds more; stretch > 1 long and thin (dive), < 1 squashed (landing); tail = length of the tail (1 calm, 3 dive); bend = (by, bz) curls the tail along the flight path
sag            [(x, z, w)] riders in cloud coordinates (x forward, z toward us when yaw = 0), w = weight 0..1: it dips under them and bulges round them (h4: many riders)
breath         amplitude of the breathing (1).  N.top(x, z) -> y of the surface under a rider (after sag), so feet press in.
vapour         N.vapour(cx, cy, S, f, ..., n=14, vel=(vx, vy) px per frame of the air going past) : curls of vapour, pure function of f
xmin           drop the lobes behind this x (rider's own eyes: xmin=.6 leaves the nose)
tone           percentiles where light / base / shade end (77, 45, 13); backlit: (88, 64, 24)
pal            dawn day night white, or dict(light, base, shade, deep, line, rim, curl)"""
import numpy as np, cv2, math
from . import W, H
from .clouds import _zsplat, _cast
from .ink import Cel, spl, resample, paste
from .kit import contours
from .paint import rgb, f32, fbm, smooth

pi = math.pi
PAL = dict(dawn=dict(light='#fff0bd', base='#fdcf72', shade='#e99a52', deep='#b96a58', line='#a3512c', rim='#fff8e0', curl='#c87a35', air='#f6c08a'),
           day=dict(light='#fffbe2', base='#ffe08a', shade='#f2b254', deep='#cf8348', line='#a8662a', rim='#ffffff', curl='#cf9038', air='#ffe9b0'),
           white=dict(light='#ffffff', base='#f2f5fb', shade='#c5d3ea', deep='#93a7cf', line='#7c93bd', rim='#ffffff', curl='#a9bcdc', air='#e4ecf8'),
           night=dict(light='#c9b48a', base='#8f7f78', shade='#5e5a74', deep='#3c3c5c', line='#2c2b4a', rim='#e8dcc0', curl='#57546e', air='#6c6a86'))


def E(a):
    y, p, r = a; Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]]); Rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]]); c, n = math.cos(-r), math.sin(-r); return np.array([[c, -n, 0], [n, c, 0], [0, 0, 1]]) @ Rx @ Ry


class Nimbus:
    def __init__(s, seed=3, size=1.0):
        r = np.random.default_rng(seed); s.seed = seed; L = []                        # x, y, z, r, kind (0 core 1 rim 2 crown 3 tail 4 under)
        for x, z, rr in ((1.05, .25, .92), (.15, -.30, 1.02), (-.75, .30, .98), (-1.35, -.25, .80), (1.45, -.35, .74), (.3, .55, .86)): L.append((x, -.58 - rr, z, rr, 0))
        n = 15
        for i in range(n):
            a = 2 * pi * (i + r.uniform(-.2, .2)) / n; rr = r.uniform(.56, .86) * (1.08 if math.cos(a) > .3 else .95); L.append((2.02 * math.cos(a) - .1, -1.10 + r.normal(0, .07), 1.28 * math.sin(a), rr, 1))
        for i in range(26):                                                     # knuckles on the flanks, they break the big forms
            a = 2 * pi * (i + r.uniform(-.4, .4)) / 26; rr = r.uniform(.28, .50); el = r.uniform(-.75, .55); L.append(((2.02 + .52 * math.cos(el)) * math.cos(a) - .1, -1.10 + .58 * math.sin(el), (1.28 + .52 * math.cos(el)) * math.sin(a), rr, 2 if i % 4 else 1))
        for i in range(22):                                                     # cauliflower crowns that make the skyline
            a = 2 * pi * (i + r.uniform(-.35, .35)) / 22; q = r.uniform(.72, .98); rr = r.uniform(.22, .42) * (1.15 if math.sin(a) < 0 else .85); L.append((2.05 * math.cos(a) * q - .1, -.80 + r.uniform(-.06, .14) - (.16 if math.sin(a) > .2 else 0), 1.30 * math.sin(a) * q, rr, 2))
        for i in range(8): a = r.uniform(0, 2 * pi); L.append((1.5 * math.cos(a), -1.78 + r.uniform(-.1, .1), .9 * math.sin(a), r.uniform(.42, .62), 4))
        for i in range(9): u = (i + .5) / 9; L.append((-2.05 - 2.1 * u, -1.12 + .30 * u + r.normal(0, .05), r.normal(0, .12), .74 * (1 - u) ** .85 + .07, 3))
        s.L = np.array(L, float); s.L[:, 1] -= .20; s.L[:, :4] *= size; s.ph = r.uniform(0, 2 * pi, len(L)); s.om = r.uniform(.10, .17, len(L)); s.id = np.arange(len(L))
        s.roll = [(r.uniform(0, 1), r.choice([-1, 1]), r.uniform(-.25, .40), r.uniform(.30, .52)) for _ in range(16)]
        s.curl = [(.95, -1.0, 1.30, .40, 1, .0), (-1.10, -1.30, 1.16, .33, -1, 1.9), (2.10, -1.12, .62, .28, 1, 3.4), (-2.45, -.98, .42, .24, -1, 4.2)]
        s.size = size

    # ---------------------------------------------------------------- deformation
    def _sag(s, P, sag):
        if not sag: return P
        P = P.copy(); tot = sum(w for _, _, w in sag)
        for x, z, w in sag:
            d2 = ((P[:, 0] - x) / .95) ** 2 + ((P[:, 2] - z) / .85) ** 2; P[:, 1] += -w * .55 * np.exp(-d2) * np.clip((P[:, 1] + 2.4) / 1.6, .2, 1) + w * .20 * np.exp(-((np.sqrt(d2) - 1.7) / .55) ** 2)
        P[:, 1] -= .10 * tot * (1 - (P[:, 0] / 3.2) ** 2).clip(-.6, 1); P[:, 0] *= 1 + .035 * min(tot, 6); P[:, 2] *= 1 + .05 * min(tot, 6); return P

    def lobes(s, f, speed=.2, stretch=1., tail=1., bend=(0, 0), sag=(), breath=1., squash=1.):
        L = s.L; t = f; br = np.sin(s.om * t + s.ph) * breath; big = math.sin(t * .105) * .020 * breath
        P = L[:, :3].copy(); r = L[:, 3] * (1 + .045 * br + big); P += .045 * breath * np.c_[np.sin(s.om * .8 * t + s.ph * 1.7), np.sin(s.om * t + s.ph * 2.3), np.cos(s.om * .9 * t + s.ph)] * L[:, 3:4]
        k = L[:, 4]
        # tail: longer, thinner, whipping
        tl = k == 3; u = np.clip((-P[:, 0] - 2.0) / 2.2, 0, 1) * tl; P[:, 0] = np.where(tl, -2.0 + (P[:, 0] + 2.0) * tail, P[:, 0]); r = np.where(tl, r * (1 + .12 * (tail - 1)) * (1 - .10 * (tail - 1) * u).clip(.35, 1), r)
        P[:, 1] += tl * (np.sin(t * .33 - u * 4.2) * .16 * u * (1 + .5 * speed)); P[:, 2] += tl * np.sin(t * .27 - u * 3.6 + 1) * .12 * u
        # rolling lobes: born at the nose, carried back along the flanks, swallowed by the tail
        R_ = []
        for (p0, sd, y, rr) in s.roll:
            q = (p0 + t * (.0028 + .0165 * speed)) % 1.; x = 2.25 - 4.7 * q; w = math.sin(pi * np.clip(q * 1.02, 0, 1)) ** .6; zz = sd * 1.42 * math.sqrt(max(.04, 1 - ((x + .1) / 2.5) ** 2)); R_.append((x, -1.30 + y * .8, zz, rr * w * .98 + .02))
        R_ = np.array(R_); P = np.vstack([P, R_[:, :3]]); r = np.r_[r, R_[:, 3]]; ids = np.r_[s.id, 1000 + np.arange(len(R_))]; kk = np.r_[k, np.full(len(R_), 2)]
        if tail > 1.15:                                                      # extra beads so that a long tail stays one cloud
            m = int(11 * (tail - 1) + 2); ex = []
            for j in range(m): v = (j + .6) / m; ex.append((-2.2 - 2.1 * tail * v, -1.25 + .28 * v + np.sin(t * .33 - v * 4.2) * .16 * v, .10 * math.sin(j * 2.1), (.66 * (1 - v) ** .7 + .06) * (1 + .06 * math.sin(t * .4 + j))))
            ex = np.array(ex); P = np.vstack([P, ex[:, :3]]); r = np.r_[r, ex[:, 3]]; ids = np.r_[ids, 2000 + np.arange(m)]; kk = np.r_[kk, np.full(m, 3)]
        P = s._sag(P, sag)
        back = np.clip(-P[:, 0] - 1.2, 0, None); P[:, 1] += bend[0] * back ** 2 * .12; P[:, 2] += bend[1] * back ** 2 * .12
        sc = np.array([stretch, squash / math.sqrt(stretch), 1 / math.sqrt(stretch) * (2 - squash if squash < 1 else 1)]); c = np.array([0, -1.1, 0]); P = (P - c) * sc + c
        return P, r, ids, kk, sc

    def top(s, x, z, sag=(), f=0, **k):
        P, r, _, kk, sc = s.lobes(f, sag=sag, **k); d2 = (P[:, 0] - x) ** 2 + (P[:, 2] - z) ** 2; m = d2 < (r * .98) ** 2; return float((P[m, 1] + np.sqrt(r[m] ** 2 - d2[m])).max()) if m.any() else -1.

    # ---------------------------------------------------------------- draw
    def draw(s, cx, cy, S, f=0, yaw=0., pitch=.12, roll=0., R=None, D=9., pal='dawn', light=(.5, .6, .35), speed=.2, stretch=1., tail=1., bend=(0, 0), sag=(), breath=1., squash=1., front_z=.80, curls=True, line=1., ss=None, lost=.55, tone=(77, 45, 13), xmin=-99.):
        """-> (full layer, front layer), premultiplied (H, W, 4)"""
        pal = PAL[pal] if isinstance(pal, str) else pal; R = E((yaw, pitch, roll)) if R is None else np.asarray(R, float); Pl, r, ids, kk, sc = s.lobes(f, speed, stretch, tail, bend, sag, breath, squash)
        def proj(P): q = np.asarray(P, float) @ R.T; k = D / np.maximum(D - q[..., 2], .35); return np.stack([cx + q[..., 0] * k * S, cy - q[..., 1] * k * S], -1), q[..., 2], k
        p, z, k = proj(Pl); ax, ay = np.linalg.norm(R[0] * sc), np.linalg.norm(R[1] * sc); rx, ry = r * S * k * ax, r * S * k * ay
        ok = ((D - z) > .5) & (Pl[:, 0] > xmin); p, z, k, rx, ry, ids, kk, Pk = p[ok], z[ok], k[ok], rx[ok], ry[ok], ids[ok], kk[ok], Pl[ok]
        pad = 8; x0, y0 = int(max(-60, (p[:, 0] - rx * 1.15).min() - pad)), int(max(-60, (p[:, 1] - ry * 1.15).min() - pad)); x1, y1 = int(min(W + 60, (p[:, 0] + rx * 1.15).max() + pad)), int(min(H + 60, (p[:, 1] + ry * 1.15).max() + pad)); E0 = np.zeros((H, W, 4), f32)
        if x1 - x0 < 4 or y1 - y0 < 4: return E0, E0.copy()
        w, h = x1 - x0, y1 - y0; ss = (2. if max(w, h) < 900 else 1.25 if max(w, h) < 1500 else 1.) if ss is None else ss; sw, sh = int(w * ss), int(h * ss)
        Z = np.full((sh, sw), -1e6, f32); NX = np.zeros((sh, sw), f32); NY = np.zeros_like(NX); NZ = np.zeros_like(NX); LV = np.full_like(NX, -1)
        _zsplat(Z, NX, NY, NZ, LV, ((p[:, 0] - x0) * ss).astype(f32), ((p[:, 1] - y0) * ss).astype(f32), (z * S * k * ss).astype(f32), (rx * ss).astype(f32), (ry * ss).astype(f32), np.arange(len(p)).astype(f32), (ids * 613 + s.seed * 977).astype(np.int64) & 0xffffff, .062)
        M = Z > -1e5; Mf = M.astype(f32); lt = np.array([light[0], -light[1], light[2]], float); lt /= np.linalg.norm(lt); U = S * ss
        # tones: one mass first, lobes second
        def nb(a, sg):
            q = max(1, int(sg / 3)); aa = cv2.resize(a, (max(2, sw // q), max(2, sh // q)), interpolation=cv2.INTER_AREA); mm = cv2.resize(Mf, (aa.shape[1], aa.shape[0]), interpolation=cv2.INTER_AREA); o = cv2.GaussianBlur(aa, (0, 0), sg / q) / (cv2.GaussianBlur(mm, (0, 0), sg / q) + 1e-3); return cv2.resize(o, (sw, sh), interpolation=cv2.INTER_LINEAR)
        sg = max(2., U * .42); bx, by, bz = nb(NX * Mf, sg), nb(NY * Mf, sg), nb(NZ * Mf, sg); bl = np.sqrt(bx * bx + by * by + bz * bz) + 1e-4; vb = (bx * lt[0] + by * lt[1] + bz * lt[2]) / bl; vl = NX * lt[0] + NY * lt[1] + NZ * lt[2]
        q = 2; Zs = cv2.resize(Z, (max(1, sw // q), max(1, sh // q)), interpolation=cv2.INTER_NEAREST); cs = _cast(Zs, float(lt[0]), float(lt[1]), float(lt[2]), 14, float(max(1.2, U / q * .085)), float(max(1., U / q * .22))); cs = cv2.resize(cv2.GaussianBlur(cs, (0, 0), max(.6, U / q * .035)), (sw, sh), interpolation=cv2.INTER_LINEAR)
        zb = nb(np.where(M, Z, 0), U * .30); cav = np.clip((zb - Z) / (U * .55), -1, 1) * Mf
        ox = int(round((cx - x0) * ss)); oy = int(round((cy - y0) * ss)); wn = _wn(sh, sw, ox, oy, U)
        yy = (np.arange(sh, dtype=f32)[:, None] / ss + y0); base_y = cy + 1.15 * S; up = np.clip((base_y - yy) / (1.6 * S), -1.2, 1)
        v = .44 * vb + .52 * vl + .22 * up - .26 * cs - .10 * np.clip(cav, 0, 1) + .030 * wn; v = np.where(M, v, 0)
        if U < 70: sv = U * .045; v = np.where(M, cv2.GaussianBlur(v, (0, 0), sv) / (cv2.GaussianBlur(Mf, (0, 0), sv) + 1e-3), 0)                # small on screen: fewer, bigger tone shapes
        smp = v[::3, ::3][M[::3, ::3]]
        p_hi, p_lit, p_dk = np.percentile(smp, list(tone)) if smp.size > 30 else (.5, .2, -.2); e = max(.008, (p_hi - p_dk) * .013) * max(1., 1.2 / ss)
        T = [rgb(pal[n]) for n in ('deep', 'shade', 'base', 'light')]; tn = lambda th, a=1.: smooth(th - e * a, th + e * a, v)[..., None]
        c = T[0] + (T[1] - T[0]) * tn(p_dk, 1.6); c = c + (T[2] - c) * tn(p_lit, 1.1); c = c + (T[3] - c) * tn(p_hi, 1.)
        c = c * (1 + .010 * wn[..., None] + .012 * _wn(sh, sw, ox + 71, oy + 31, U * .28)[..., None])
        # the sun catches the crest
        dx, dy = lt[0] * max(1.2, U * .035), lt[1] * max(1.2, U * .035); Ms = cv2.warpAffine(Mf, np.float32([[1, 0, -dx], [0, 1, -dy]]), (sw, sh)); rr = np.clip(Mf - Ms, 0, 1) * smooth(-1.1, .3, wn); c = c + (rgb(pal['rim']) - c) * (cv2.GaussianBlur(rr, (0, 0), .6 * ss) * .85)[..., None]
        a = Mf.copy()
        # lost lower edge: the underside frays into air
        if lost > 0:
            low = smooth(-.35, -1.05, up + .16 * wn); a = a * (1 - lost * low * smooth(.1, .9, 1 - cv2.GaussianBlur(Mf, (0, 0), max(1.5, U * .13)) ** 2 * .9) * .9); c = c + (rgb(pal['air']) - c) * (low * .30 * lost)[..., None]
        lab = cv2.resize(LV, (w, h), interpolation=cv2.INTER_NEAREST) if ss != 1 else LV
        c1 = cv2.resize(c * a[..., None], (w, h), interpolation=cv2.INTER_AREA) if ss != 1 else c * a[..., None]; a1 = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA) if ss != 1 else a
        img = np.dstack([c1, a1]).astype(f32)
        # pen
        cel = Cel(w, h, x0, y0, ss=2); s._ink(cel, S, p, z, rx, ry, k, ids, kk, Z, LV, ss, x0, y0, pal, line, lt, up, yy, Mf)
        if curls: s._curls(cel, proj, S, f, pal, R, line, speed, sc, Z, ss, x0, y0, p, z, rx, ry, ids, kk, LV, lt)
        ln = cel.rgba(); ln = ln * np.clip(a1 * 1.15 + .15, 0, 1)[..., None] if lost > 0 else ln; img = img * (1 - ln[..., 3:4]) + ln
        full = E0; paste(full, img, x0, y0)
        fr = (z > front_z) & (kk != 3); idx = np.nonzero(fr)[0]; fm = np.isin(LV, idx).astype(f32); fm = cv2.resize(fm, (w, h), interpolation=cv2.INTER_AREA) if ss != 1 else fm; lwk = max(1, int(round(.9 + S * .012))); fm = np.maximum(fm, cv2.dilate(fm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * lwk + 1, 2 * lwk + 1))) * (ln[..., 3] > .25)); front = np.zeros((H, W, 4), f32); paste(front, img * fm[..., None], x0, y0)
        return full, front

    def _ink(s, cel, S, p, z, rx, ry, k, ids, kk, Z, LV, ss, x0, y0, pal, line, lt, up, yy, Mf):
        sh, sw = Z.shape; wo = float(np.interp(S, [6, 16, 40, 90, 200], [.0, .7, 1.4, 2.6, 4.6])) * line
        if wo <= .25: return
        lc = pal['line']; a = np.linspace(0, 2 * pi, 97); U = S * ss; L2 = np.array([lt[0], lt[1]]); L2 /= np.hypot(*L2) + 1e-6
        Zf = np.where(Z > -1e5, Z, -U * 9)
        # silhouette: thick where it turns from the light, lifted where the sun eats it, gone where the underside frays
        cs = contours(cv2.GaussianBlur(Mf, (0, 0), .8 * ss), min_len=int(30 * ss))
        for ci, C in enumerate(cs):
            Q = C / ss + [x0, y0]; n = len(Q); T = np.roll(Q, -3, 0) - np.roll(Q, 3, 0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[T[:, 1], -T[:, 0]]
            tst = np.clip(((Q + N * 3 - [x0, y0]) * ss).astype(int), 0, [sw - 1, sh - 1]);
            if Mf[tst[:, 1], tst[:, 0]].mean() > .5: N = -N
            lit = N @ L2; low = N[:, 1] > .72; on = (lit < .60) & ~(low & (np.sin(np.arange(n) * .021 * 60 / max(S, 8)) > -.1)); on = cv2.morphologyEx(on.astype(np.uint8)[None], cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8))[0] > 0
            s._runs(cel, Q, on, wo, lc, lit, ci)
        # lobes over lobes: only the arcs that a hand would draw
        for i in np.argsort(-rx):
            if rx[i] < max(5., S * .20) or kk[i] == 4: continue
            E_ = np.c_[p[i, 0] + rx[i] * .985 * np.cos(a), p[i, 1] + ry[i] * .985 * np.sin(a)]; In = np.c_[p[i, 0] + rx[i] * .86 * np.cos(a), p[i, 1] + ry[i] * .86 * np.sin(a)]; Out = np.c_[p[i, 0] + rx[i] * 1.10 * np.cos(a), p[i, 1] + ry[i] * 1.10 * np.sin(a)]
            g = lambda Q: (np.clip(((Q[:, 0] - x0) * ss).astype(int), 0, sw - 1), np.clip(((Q[:, 1] - y0) * ss).astype(int), 0, sh - 1)); ui, vi = g(In); uo, vo = g(Out); own = LV[vi, ui] == i; ob = (LV[vo, uo] >= 0) & (LV[vo, uo] != i); jump = Zf[vi, ui] - Zf[vo, uo] > U * .16
            on = own & ob & jump & (np.sin(a) < .30)
            on = cv2.morphologyEx(on.astype(np.uint8)[None], cv2.MORPH_OPEN, np.ones((1, 9), np.uint8))[0] > 0
            lit = np.cos(a) * L2[0] + np.sin(a) * L2[1]; s._runs(cel, E_, on, wo * (.62 if kk[i] != 2 else .50), lc, lit, 40 + int(ids[i]), keep=.72)

    def _runs(s, cel, Q, on, w, col, lit, seed, keep=1.):
        n = len(Q)
        if on.all(): cel.line(np.vstack([Q, Q[:1]]), w, col, taper=(.02, .02), press=.3, seed=seed, tremor=.2); return
        if not on.any(): return
        ch = np.nonzero(on != np.roll(on, 1))[0]
        for a, b in zip(ch, np.r_[ch[1:], ch[0] + n]):
            if not on[a % n]: continue
            idx = np.arange(a, b) % n; P = Q[idx]; L = np.hypot(*np.diff(P, axis=0).T).sum()
            if L < w * 6: continue
            if keep < 1: m = len(P); P = P[int(m * (1 - keep) * .4):max(4, int(m * (1 - (1 - keep) * .6)))]
            cel.line(P, w * (1.0 - .30 * float(np.mean(lit[idx]))), col, taper=(.22, .30), press=.35, seed=seed + int(a), tremor=.3, tmin=.10)

    def _curls(s, cel, proj, S, f, pal, R, line, speed, sc, Z, ss, x0, y0, p, z, rx, ry, ids, kk, LV, lt):
        """the signature: a lobe's own edge winds inward and becomes a spiral (with the shade tucked under it)"""
        wo = float(np.interp(S, [6, 16, 40, 90, 200], [.0, .6, 1.4, 2.7, 4.6])) * line
        if wo < .5: return
        sh, sw = Z.shape; got = []
        for i in np.argsort(ids):                                             # fixed order, faded by how much of the lobe shows: nothing pops
            if kk[i] != 1 or ids[i] >= 900 or rx[i] < .30 * S: continue
            u = np.linspace(0, 1, 60); d = 1 if (ids[i] % 2) else -1; th = (.62 * pi if d < 0 else .38 * pi) - d * (u * 2 * pi * 1.22 + f * .006) ; rad = .965 * (1 - u) ** .62 * (1 - .10 * np.sin(u * 5)) + .05
            E_ = np.c_[p[i, 0] + rx[i] * rad * np.cos(th), p[i, 1] + ry[i] * rad * np.sin(th)]; ui = np.clip(((E_[:, 0] - x0) * ss).astype(int), 0, sw - 1); vi = np.clip(((E_[:, 1] - y0) * ss).astype(int), 0, sh - 1); own = LV[vi, ui] == i
            wt = float(smooth(.42, .66, own.mean()))
            if wt <= .02 or any(np.hypot(p[i, 0] - q[0], p[i, 1] - q[1]) < 1.25 * S for q in got): continue
            ow = cv2.morphologyEx(own.astype(np.uint8)[None], cv2.MORPH_CLOSE, np.ones((1, 7), np.uint8))[0] > 0; best = (0, 0); a0 = None
            for q_, v_ in enumerate(list(ow) + [False]):
                if v_ and a0 is None: a0 = q_
                if not v_ and a0 is not None:
                    if q_ - a0 > best[1] - best[0]: best = (a0, q_)
                    a0 = None
            C = E_[best[0]:best[1]]
            if len(C) < 22: continue
            got.append(p[i])
            if len(got) > 3: break
            m = len(C); T = np.gradient(C, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; Nn = np.c_[-T[:, 1], T[:, 0]] * (1 if d > 0 else -1); wd = np.sin(np.pi * np.linspace(0, 1, m)) ** .8 * rx[i] * .10
            cel.fill(np.vstack([C, (C + Nn * wd[:, None])[::-1]]), pal['shade'], .50 * wt)
            cel.line(C, wo * (.45 + .55 * wt), pal['curl'], alpha=wt, taper=(.18, .30), press=.25, seed=int(ids[i]) * 7, tremor=.2, wfun=lambda t: 1.15 - .55 * t)

    # ---------------------------------------------------------------- vapour
    def vapour(s, cx, cy, S, f, vel=(-3., 0.), n=12, life=44, yaw=0., pitch=.12, roll=0., R=None, D=9., pal='dawn', speed=.2, tail=1., op=.9, spread=1., size=1.):
        """curls shed from the tail and the flanks, drifting with the air (vel px per frame, in screen space); each one uncurls, thins and is gone"""
        pal = PAL[pal] if isinstance(pal, str) else pal; R = E((yaw, pitch, roll)) if R is None else np.asarray(R, float); cel = Cel(W, H, ss=2); gap = life / n
        for k in range(int(math.floor((f - life) / gap)), int(math.floor(f / gap)) + 1):
            b = k * gap; a = (f - b) / life
            if not 0 <= a < 1: continue
            r = np.random.default_rng(s.seed * 1009 + (k % 100003)); flank = r.random() < .45
            P = np.array([r.uniform(-2.2, .2), -1.15 + r.uniform(-.2, .25), r.choice([-1, 1]) * 1.45]) if flank else np.array([-2.4 - 1.8 * tail * r.uniform(.3, 1), -.98 + r.normal(0, .14), r.normal(0, .25)]); q = P @ R.T; kk = D / max(D - q[2], .5); p0 = np.array([cx + q[0] * kk * S, cy - q[1] * kk * S])
            t = f - b; p = p0 + np.array(vel) * t * (1 - .25 * a) + np.array([math.sin(t * .09 + k) * .12, -(.16 * a + .10 * math.sin(t * .07 + k * 2))]) * S * spread
            env = math.sin(pi * min(1., a * 1.25 + .06)) ** .7 * (1 - a) ** .55; rr = S * size * r.uniform(.22, .46) * (1. if env > .10 else 0.) + .01 * (.55 + .75 * a); d = r.choice([-1, 1]); th0 = r.uniform(0, 2 * pi) + d * t * .05; turns = (1.35 if S > 46 else .78 + .25 * (S > 30)) * (1 - .50 * a); u = np.linspace(0, 1, 40); th = th0 + d * turns * 2 * pi * (1 - u); rad = rr * (.10 + .9 * u ** .85)
            C = p + np.c_[np.cos(th) * rad, np.sin(th) * rad * .82]; tl = np.array(vel, float); tl = tl / (np.hypot(*tl) + 1e-6); ext = u > .8; C = C + tl * (np.clip(u - .72, 0, 1) ** 1.5 * rr * 5.5)[:, None]
            wd = max(.5, S * size * (.135 if S > 46 else .20) * env); T = np.gradient(C, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; Nn = np.c_[-T[:, 1], T[:, 0]]; pr = np.sin(pi * np.clip(u * .97 + .02, 0, 1)) ** .8 * (1.15 - .5 * u); A, B = C + Nn * (wd * pr)[:, None], C - Nn * (wd * pr)[:, None]
            if env <= .12: continue
            cel.fill(np.vstack([A, B[::-1]]), pal['base'], op); cel.fill(np.vstack([A, (C + Nn * (wd * pr * .05)[:, None])[::-1]]), pal['light'], op)
            if S > 30: cel.line(B, max(.6, S * .018) * env, pal['curl'], taper=(.2, .4), seed=k, alpha=op * .9)
        return cel.layer()


_WN = {}


def _wn(h, w, ox, oy, U):
    """brush-scale noise that moves WITH the cloud (anchored at its centre)"""
    key = max(4, int(round(math.log2(max(U, 2)) * 2)))
    if key not in _WN: u = 2 ** (key / 2.); _WN[key] = fbm(1400, 1800, (u * .9, u * .30, u * .11), (1, .6, .35), 41 + key).astype(f32)
    n = _WN[key]; H_, W_ = n.shape; yy = (np.arange(h) - oy + H_ // 2) % (2 * H_ - 2); xx = (np.arange(w) - ox + W_ // 2) % (2 * W_ - 2); yy = np.where(yy >= H_, 2 * H_ - 2 - yy, yy); xx = np.where(xx >= W_, 2 * W_ - 2 - xx, xx); return n[yy][:, xx]
