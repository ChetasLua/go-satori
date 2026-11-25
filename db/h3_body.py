"""Anatomy and costumes for db.hero3: turns a posed Skel into Clay (muscles from origin to insertion, cloth shells, plates) + fold decals."""
import numpy as np, math
from .h3_sdf import Clay, table, ridge, bump, wrap, prof, TT, PH, NT, NP
from .h3_pose import Skel, BODY, CHAR_BODY, unit, D2R

PI = math.pi

# light, base, shadow, deep
MATS = dict(
    skin=('#ffe8c8', '#f5c79b', '#c48561', '#85503f'), skinv=('#ffe4c6', '#efc096', '#b97d5e', '#7a4a3e'), skink=('#ffedd2', '#f8cfa4', '#d0926c', '#8f5a48'),
    gi=('#ffad52', '#f47b20', '#b64a1c', '#74290f'), gid=('#f29a4a', '#d96a1e', '#9a3d18', '#5e2210'), under=('#4a74c4', '#25447f', '#152a54', '#0b1733'), belt=('#4a74c4', '#25447f', '#152a54', '#0b1733'), band=('#4a74c4', '#25447f', '#152a54', '#0b1733'),
    boot=('#5079c2', '#274680', '#162b55', '#0b1733'), trim=('#ffd98a', '#e2aa4a', '#a06a2c', '#5e3a1a'), trimr=('#e8604a', '#b83a2c', '#7a2220', '#481414'), sole=('#6a4a3a', '#4a3028', '#2e1c18', '#1a100e'),
    suit=('#4a70d0', '#203d8c', '#12275c', '#0a1638'), armw=('#ffffff', '#f0ede6', '#b0b2c8', '#70739a'), armg=('#ffe596', '#e3ab3e', '#9c6626', '#5c381a'), glove=('#ffffff', '#f2efea', '#b2b4ca', '#74779c'), bootw=('#ffffff', '#f0ede6', '#b0b2c8', '#70739a'), gold=('#ffe596', '#e3ab3e', '#9c6626', '#5c381a'),
    fw=('#ffffff', '#f1eff6', '#a5a2c2', '#59557f'), fv=('#c48bff', '#6c2cbc', '#36126c', '#190638'), fnail=('#ffffff', '#e2dcf2', '#8d84b4', '#4c4474'),
    green=('#c6ea86', '#86ba4c', '#4e7f3c', '#2a4e30'), pink=('#ffc4b8', '#dd8a86', '#a35462', '#6a3048'), gip=('#9a74c8', '#5f3e94', '#3a2462', '#20133a'), sashb=('#9fdcf2', '#56a6cc', '#2f6a8c', '#1a3e58'), cape=('#ffffff', '#f0eeea', '#aeb0c6', '#6c6f94'), shoeb=('#c4884e', '#8e5428', '#5a3018', '#341b0f'), bandr=('#f07a6a', '#c23a3c', '#801f2e', '#4c1220'),
    gipk=('#a078cc', '#6a46a0', '#42296c', '#261642'), sashr=('#f87068', '#cc363c', '#88202e', '#521424'), scarf=('#ffffff', '#f3efe9', '#b9b6c6', '#7a7896'), hat=('#f46a5a', '#cf3832', '#8a2028', '#541420'), tail=('#b07a4a', '#7c4e2c', '#4c2d1b', '#2c1910'),
    inside=('#2a2430', '#1c1722', '#120e17', '#0a080e'), insideo=('#a8481c', '#7c2f12', '#4e1c0c', '#301008'), insideb=('#1a2c58', '#101d40', '#0a132c', '#060b1c'), grey=('#a8acb8', '#757a88', '#474b58', '#262933'))

SPEC = dict(
    goku=dict(top='gi', bottom='gi', shoe='goku', skin='skin', cloth='gi', under='under', band='band', belt='belt', hair='goku', disc='悟'),
    vegeta=dict(top='suit', bottom='suit', shoe='vegeta', skin='skinv', cloth='suit', glove='glove', armour=True, hair='vegeta'),
    frieza=dict(top='frieza', bottom='bare', shoe='frieza', skin='fw', cloth='fw', hair='frieza', tail=True),
    piccolo=dict(top='gip', bottom='gi', shoe='piccolo', skin='green', cloth='gip', under=None, band='bandr', belt='sashb', hair='piccolo'),
    gohan=dict(top='gi', bottom='gi', shoe='kid', skin='skink', cloth='gipk', under='gipk', band='bandr', belt='sashr', hair='gohan', scarf=True, shoemat='boot'),
    kid=dict(top='gi', bottom='gi', shoe='kid', skin='skink', cloth='gi', under='under', band='band', belt='belt', hair='kid', disc='亀', mtail=True, shoemat='boot'),
)
# variants: Fig('goku', 'torn') etc.
VARIANT = dict(
    goku=dict(clean={}, torn=dict(torn=True), bare=dict(top='bare'), waist=dict(top='bare', jacket=True), ui=dict(top='bare', hair='ui'), ssj=dict(torn=True, hair='ssj'), ssjclean=dict(hair='ssj')),
    vegeta=dict(clean={}, shattered=dict(armour=False, shattered=True), nogear=dict(armour=False), ssj=dict(hair='vssj'), ssjshattered=dict(armour=False, shattered=True, hair='vssj')),
    frieza=dict(clean={}), piccolo=dict(clean=dict(cape=True, turban=True), nocape=dict(cape=False, turban=False), torn=dict(cape='torn', turban=False)), gohan=dict(clean=dict(hat=True), nohat=dict(hat=False)), kid=dict(clean={}, bare=dict(top='bare')))

# crease ink per muscle id prefix (0 = never drawn)
LINEW = dict(pec=0., pecu=0., clav=.0, trap=.45, trapl=.55, scap=.55, lat=.8, ere=.45, ter=.6, trapm=.5, ab=.0, obl=.0, ser=.0, glu=1., delt=1.0, deltf=.8, deltb=.8, bic=.95, tri=.85, tril=.7, bra=.6, brr=.75, flx=.6, ext=.6, olec=.4, rec=.6, vl=.8, vm=.9, add=.5, hamo=.6, hami=.6, pat=.8, gasm=.8, gasl=.8, sol=.5, tib=.6, scm=.9, neck=0., core=.0, rib=0., bel=0., pel=.5, knu=.45, mal=.3)


def facet(seed, amp, nt=5, np_=7, fade=(0., 0.)):
    """piecewise-linear cloth irregularity: straights and corners instead of hose curves"""
    r = np.random.default_rng(seed); g = r.normal(0, 1, (nt, np_)); g = np.hstack([g, g[:, :1]])
    u = TT * (nt - 1); v = PH / (2 * PI) * np_; i = np.minimum(u.astype(int), nt - 2); j = np.minimum(v.astype(int), np_ - 1); a, b = u - i, v - j
    z = (g[i, j] * (1 - b) + g[i, j + 1] * b) * (1 - a) + (g[i + 1, j] * (1 - b) + g[i + 1, j + 1] * b) * a
    w = np.ones_like(TT)
    if fade[0] > 0: w = w * np.clip(TT / fade[0], 0, 1)
    if fade[1] > 0: w = w * np.clip((1 - TT) / fade[1], 0, 1)
    return amp * z * w


def tear(p, seed, n=7):
    """ragged 0..1 edge over phi"""
    r = np.random.default_rng(seed); k = np.sort(r.uniform(-PI, PI, n)); v = r.uniform(0, 1, n); v[::2] *= .25; kk = np.r_[k - 2 * PI, k, k + 2 * PI]; vv = np.r_[v, v, v]; return np.interp(wrap(p), kk, vv)


def drape(seed, amp, n=5, t=(.1, .9), skew=(.06, .16), w=(.045, .075), ln=(.7, 1.3), sign=None, p=None):
    r = np.random.default_rng(seed); z = 0; sg = sign if sign is not None else (1 if r.random() < .5 else -1)
    for i in range(n):
        t0 = t[0] + (t[1] - t[0]) * (i + r.uniform(.15, .85)) / n; p0 = r.uniform(-PI, PI) if p is None else r.uniform(*p)
        z = z + ridge(TT, PH, t0, p0, sg * r.uniform(*skew), r.uniform(*w), r.uniform(*ln), amp * r.uniform(.55, 1.15))
    return z


class Body:
    """posed, dressed clay + decals"""

    def __init__(self, char, sk, fx, px, spec=None, seed=0):
        self.c = Clay(); self.sk = sk; self.b = sk.b; self.fx = fx; self.px = px; self.deco = []; self.prim = {}; self.char = char; self.discs = []; self.seams = []
        self.sp = dict(SPEC[char]); self.sp.update(spec or {}); self.seed = seed
        self.H = sk.H; self.k = sk.H / 1.76; self.m = self.k * self.b['mass']; self.M = self.b['muscle']
        self.lod = 0 if px < 130 else 1 if px < 320 else 2
        self.build(); self.c.pack()

    # ---- helpers
    def line(self, prim, t, p, w=1., **k): self.deco.append(dict(kind='line', prim=prim, t=np.asarray(t, float), p=np.asarray(p, float), w=w, **k))
    def shard(self, prim, t, p, w=0., **k): self.deco.append(dict(kind='shard', prim=prim, t=np.asarray(t, float), p=np.asarray(p, float), w=w, **k))

    def blade(self, prim, a, b, w=.05, kink=.3, bow=0., pri=0, line=True, lw=1.0, shade=True):
        """cloth fold: a pen line from a=(t, phi) (on the tension point) to b (free end), bowed, with a wedge of shadow hanging from it"""
        a, b = np.asarray(a, float), np.asarray(b, float); d = b - a; L = math.hypot(d[0] * 3.0, d[1]) + 1e-6; n = np.array([d[1] / 3.0, -d[0] * 3.0]) / L
        n = n / (abs(n[0]) + abs(n[1]) / 3 + 1e-6); e = np.array([1., 3.]); o = n * e; u = np.linspace(0, 1, 7); bw = (.10 * L / 3.0 + bow) * (1 if (a[1] + b[1]) > 0 else -1)
        C = a + np.outer(u, d) + np.outer(np.sin(u * PI) * bw * .35, o)
        if shade:
            k = np.array([.0, .55, .85, .8, .55, .25, 0.]) * w * .62; S = np.vstack([C, (C + np.outer(k, o))[::-1][1:-1]]); self.shard(prim, S[:, 0], S[:, 1], pri=pri)
        if line: self.line(prim, C[:, 0], C[:, 1], lw, pri=pri)

    def C(self, x, y, z): return self.sk.chest + self.sk.Rc @ np.array([x * self.m, y * self.m, z * self.k])
    def Wa(self, x, y, z): return self.sk.waist + self.sk.Rw @ np.array([x * self.m, y * self.m, z * self.k])
    def Pv(self, x, y, z): return self.sk.pel + self.sk.Rp @ np.array([x * self.m, y * self.m, z * self.k])

    def trunk_pt(self, x, z, y=0.):
        """point of the trunk in chest coordinates (x right, z up from the chest joint, y fwd), bending with the spine: chest -> pelvis frame"""
        zc = (self.b['chestz'] - self.b['hipz'] - .02) * 1.76; t = float(np.clip((-.02 - z) / (zc - .10), 0, 1)); t = t * t * (3 - 2 * t); return self.C(x, y, z) * (1 - t) + self.Pv(x, y, z + zc) * t, unit(self.sk.Rc[:, 1] * (1 - t) + self.sk.Rp[:, 1] * t), unit(self.sk.Rc[:, 0] * (1 - t) + self.sk.Rp[:, 0] * t)

    def seam(self, pts, w=1., pri=0, back=False, side=0., labels=('torso',), taper=(.35, .55), both=True):
        """pen line on the trunk: [(x, z), ...] dropped from the front (or the back, or sideways) on the finished surface, mirrored"""
        from .ink import spl
        for sg in ((1, -1) if both else (1,)):
            q = spl(np.array(pts, float), n=6); O, D = [], []
            for x, z in q: p, f, r = self.trunk_pt(sg * x, z); d = unit((f if not back else -f) + r * sg * side); O.append(p + d * .40 * self.m); D.append(-d)
            self._sq.append((np.array(O), np.array(D), dict(w=w, pri=pri, labels=labels, taper=taper)))

    def trunk_lines(self):
        if self.sp['top'] == 'frieza':
            S = self.seam; S([(.010, .030), (.040, .012), (.084, .012), (.120, .034), (.140, .064)], 1.2, 0, taper=(.25, .40)); S([(0, .150), (.001, .105), (0, .050)], .7, 0, both=False, taper=(.6, .2)); S([(.066, -.030), (.092, -.066), (.104, -.104)], .7, 0); S([(0, -.130), (0, -.165)], .7, 1, both=False); S([(.040, -.150), (.046, -.185), (.040, -.215)], .6, 1); S([(0, .170), (.002, .080), (0, -.02), (.001, -.130)], .9, 0, back=True, both=False, taper=(.4, .3)); S([(.034, .150), (.074, .120), (.086, .060), (.060, .024)], .8, 0, back=True, taper=(.4, .4)); return
        """the drawn anatomy of a bare (or suited) trunk: few lines, each in its place"""
        S = self.seam; cw = self.b['chest']; ww = self.b['waist']; k = 1.0
        front = (self.sk.Rc[:, 1] @ self.fx.get('eyedir', self.sk.Rc[:, 1])) > -.25; bk = (self.sk.Rc[:, 1] @ self.fx.get('eyedir', self.sk.Rc[:, 1])) < .45
        if front:
            S([(.006, .026), (.036, .006), (.080, .004), (.118, .026), (.142, .060)], 1.35, 0, taper=(.25, .40)); S([(0, .150), (.001, .105), (0, .050), (.002, .024)], .85, 0, both=False, taper=(.6, .2))
            if not self.sp.get('armour'):
                S([(0, -.004), (.001, -.060), (0, -.125), (0, -.150)], .85, 0, both=False, taper=(.3, .5))
                S([(.026, -.002), (.062, -.026), (.096, -.066), (.108, -.100)], .75, 0, taper=(.3, .6)); S([(.006, -.044), (.036, -.049), (.056, -.041)], .8, 0); S([(.006, -.084), (.034, -.089), (.053, -.081)], .75, 1); S([(.006, -.124), (.032, -.128), (.050, -.121)], .7, 1)
                S([(.064, -.045), (.064, -.090), (.058, -.135), (.050, -.165)], .6, 1, taper=(.5, .5)); S([(.112, -.175), (.086, -.205), (.060, -.232)], .85, 0, taper=(.2, .6)); S([(0, -.162), (0, -.172)], 1.0, 1, both=False)
                S([(.116, .010), (.130, -.006), (.126, -.018)], .75, 0, side=.8); S([(.112, -.022), (.126, -.038), (.121, -.050)], .7, 1, side=.8); S([(.106, -.052), (.118, -.068), (.113, -.080)], .65, 1, side=.8)
            S([(.018, .202), (.070, .188), (.120, .196), (.165, .205)], .8, 0, taper=(.15, .5)); S([(.012, .196), (.016, .182), (.010, .172)], .6, 1)
        if bk:
            S([(0, .190), (.002, .100), (0, .0), (.001, -.100), (0, -.170)], 1.0, 0, back=True, both=False, taper=(.4, .3)); S([(.030, .170), (.070, .140), (.092, .090), (.084, .040), (.058, .016)], .9, 0, back=True, taper=(.4, .4)); S([(.150, .060), (.120, .0), (.084, -.060), (.052, -.120)], .85, 0, back=True, taper=(.3, .6))
            S([(.096, .128), (.130, .108), (.150, .084)], .7, 1, back=True); S([(.018, .050), (.040, .010), (.060, -.020)], .6, 1, back=True); S([(.040, -.130), (.046, -.165), (.040, -.190)], .7, 1, back=True)

    def mus(self, P0, P1, n, w, h, mg, G, lab, mat, k=.013, skew=0., box=.12, rest=None, ext=1.06, lift=0., bow=0., s=None, rule=None):
        """muscle belly from origin P0 to insertion P1, bulging along n.  w across, h thick (metres at H=1.76)"""
        s = self.m * self.b['limb'] if s is None else s
        d = P1 - P0; L = np.linalg.norm(d) + 1e-6; ax = d / L; n = np.asarray(n, float); n = n - ax * (n @ ax); nn = np.linalg.norm(n)
        if nn < 1e-5: n = np.cross(ax, [.3, .5, .8]); nn = np.linalg.norm(n)
        n = n / nn; e1 = np.cross(n, ax); q = float(np.clip((rest * self.k / L) ** .5, .86, 1.28)) if rest else 1.
        mk = .45 + .55 * self.M if self.M < 1 else self.M
        return self.c.egg((P0 + P1) / 2 + n * lift * s, ax, e1, (w * s * q, L / 2 * ext, h * s * q * mk), G, lab, mat, k=k * self.k, skew=skew, box=box, bow=bow * s, mg=mg, name=mg, rule=rule)

    # ================================================================== build
    def build(self):
        sp = self.sp; top = sp['top']; self.trunk_core()
        if top in ('bare', 'suit', 'frieza') or sp.get('torn'): self.trunk_muscles(sp['skin'] if top in ('bare', 'frieza') or sp.get('torn') else sp['cloth'])
        self.legs()
        for s in 'LR': self.arm(s); self.hand(s); self.foot(s)
        self.neck_()
        if top in ('gi', 'gip'): self.gi_top()
        if sp.get('belt'): self.belt()
        if sp.get('armour'): self.armour()
        if top == 'frieza': self.frieza_parts()
        if sp.get('scarf'): self.scarf()
        if sp.get('cape'): self.cape(sp['cape'])
        if sp.get('mtail'): self.tail_(False)
        if sp.get('tail'): self.tail_(True)
        if self.M > .3 and self.lod and (top in ('bare', 'frieza') or sp.get('torn') or (top == 'suit')):
            self.c.pack(); self._sq = []; self.trunk_lines(); lvl = self.lod
            self._sq = [q for q in self._sq if q[2]['pri'] <= (0 if self.px < 190 else 1 if self.px < 380 else 2)]
            if self._sq:
                P3, ok = self.c.drop(np.vstack([q[0] for q in self._sq]), np.vstack([q[1] for q in self._sq]), .8, labels=('torso',)); P3 = np.where(ok[:, None], P3, 1e3); i = 0
                for o, d, k_ in self._sq: self.seams.append(dict(k_, Q=P3[i:i + len(o)])); i += len(o)
        sk = self.sk; u = sk.u; R = sk.Rh
        self.c.ball(sk.head + R @ np.array([0, -.03 * u, .02 * u]), (.70 * u, .99 * u, .80 * u), 'head', 'head', sp['skin'], ax=R[:, 2], e1=R[:, 0], k=.0)

    # ------------------------------------------------------------------ trunk
    def trunk_core(self):
        sk, c, sp, m, k = self.sk, self.c, self.sp, self.m, self.k; b = self.b; cw, ww, hp = b['chest'], b['waist'], b['hips']; top = sp['top']
        self.tmat = sp['skin'] if top in ('bare', 'frieza') or sp.get('torn') else sp['cloth'] if top == 'suit' else sp['cloth']
        mat = self.tmat; fwd = sk.Rc[:, 1]; fw2 = unit(sk.Rp[:, 1] + fwd); kid = self.M < .3
        S = [(0, .096, .126, .080, .126), (.25, .108, .140, .088, .140), (.5, .112, .148, .092, .148), (.72, .098, .146, .090, .146), (.88, .072, .104, .080, .104), (1, .050, .066, .060, .066)]
        if kid: S = [(0, .112, .128, .100, .128), (.25, .116, .134, .104, .134), (.5, .112, .138, .104, .138), (.72, .096, .130, .098, .130), (.88, .070, .096, .082, .096), (1, .050, .066, .060, .066)]
        self.prim['ribs'] = c.loft(self.C(0, -.005, -.085), self.C(0, -.02, .235 * (b['neckz'] - b['chestz']) / .125), fwd, table(S, 2.3) * m * cw, 'torso', 'torso', mat, cap=(1.4, 1.3), k=.05 * k, side=-1, name='ribs', mg='rib')
        T = [(0, .096 * hp, .132 * hp, .100 * hp, .132 * hp), (.3, .092, .124 * ww, .090, .124 * ww), (.56, .088, .116 * ww, .082, .116 * ww), (.8, .092 * cw, .122 * cw, .084, .122 * cw), (1, .102 * cw, .134 * cw, .090, .134 * cw)]
        if kid: T = [(0, .104, .128, .108, .128), (.3, .114, .126, .100, .126), (.56, .116, .124, .096, .124), (1, .114, .128, .100, .128)]
        self.prim['belly'] = c.loft(self.Pv(0, 0, -.012), self.C(0, 0, -.03), fw2, table(T, 2.25) * m, 'torso', 'torso', mat, cap=(1.2, 1.2), k=.06 * k, side=-1, name='belly', mg='bel')
        # trapezius slope, lat sheet: keep the shoulder girdle in one piece in any arm pose
        for s, sg in (('L', -1), ('R', 1)):
            rt = sk.Rc[:, 0] * sg; up = sk.Rc[:, 2]
            q0 = sk.neck + up * .012 * k - fwd * .020 * m + rt * .020 * m; q1 = sk.sh[s] + up * .020 * k - rt * .020 * m - fwd * .004 * m
            r0, r1 = (.052, .044) if not kid else (.044, .042)
            c.loft(q0, q1, up, table([(0, r0 * .8, r0, r0 * 1.1, r0), (1, r1 * .9, r1 * 1.1, r1, r1 * 1.1)], 2) * m, 'torso', 'torso', mat, cap=(1, 1), k=.05 * k, name='trap' + s, mg='trap' + s)
            q3 = self.C(sg * .105 * cw, .0, .050); q2 = sk.sh[s] + (sk.el[s] - sk.sh[s]) * .16
            c.loft(q3, q2, fwd, table([(0, .075, .05, .075, .05), (1, .044, .044, .044, .044)], 2) * m * (1 if not kid else .85), 'torso', 'torso', mat, cap=(1, 1), k=.05 * k, name='pit' + s, mg='rib')

    def trunk_muscles(self, mat):
        sk, m, k = self.sk, self.m, self.k; G, lab = 'torso', 'torso'; up, fwd, rt = sk.Rc[:, 2], sk.Rc[:, 1], sk.Rc[:, 0]; ms = m * (.5 + .5 * self.M); lod = self.lod
        if self.M < .3: return
        C, W_, Pv = self.C, self.Wa, self.Pv; cw = self.b['chest']; ww = self.b['waist']
        def mu(*a, **q): q.setdefault('s', m); return self.mus(*a, **q)
        for s, sg in (('L', -1), ('R', 1)):
            r_ = rt * sg; sh, el = sk.sh[s], sk.el[s]; ua = unit(el - sh); e1 = sk.e1a[s]; lata = np.cross(e1, ua) * sg
            ins = sh + ua * .070 * m - e1 * .022 * m + lata * .004 * m - r_ * .012 * m
            # pectorals
            mu(C(sg * .014, .096 * cw, .068), ins, fwd * .9 + r_ * .25, .070, .038, 'pec' + s, G, lab, mat, skew=-.14, box=.46, k=.010, lift=.004)
            mu(C(sg * .022, .084 * cw, .138), ins + up * .012 * k, fwd * .8 + up * .4, .046, .028, 'pecu' + s, G, lab, mat, skew=-.1, box=.2, k=.020)
            if lod: self.c.cap(C(sg * .022, .074, .202), sk.acr[s] - r_ * .022 * m + fwd * .006 * m, .0115 * m, .010 * m, G, lab, mat, k=.014 * k, mg='clav' + s, name='clav')
            if self.sp.get('armour'): continue
            # back
            mu(Pv(sg * .058, -.064, .105), sh + ua * .085 * m + lata * -.012 * m, r_ * .72 - fwd * .60, .072, .046, 'lat' + s, G, lab, mat, skew=.24, box=.12, rest=.36, k=.014)
            mu(C(sg * .046, -.094, .062), sh - r_ * .008 * m - fwd * .034 * m + up * .006 * k, -fwd, .044, .040, 'scap' + s, G, lab, mat, k=.010, skew=.0)
            mu(C(sg * .056, -.094, .030), sh - r_ * .004 * m - fwd * .030 * m - up * .030 * k, -fwd, .030, .036, 'ter' + s, G, lab, mat, k=.009, skew=.0)
            mu(C(sg * .016, -.086, -.010), C(sg * .062, -.088, .170), -fwd + r_ * .2, .038, .034, 'trapl' + s, G, lab, mat, k=.010)
            mu(C(sg * .012, -.082, .120), C(sg * .120, -.050, .185), -fwd + up * .5, .046, .036, 'trapm' + s, G, lab, mat, k=.012)
            mu(Pv(sg * .031, -.076, .020), C(sg * .036, -.080, .030), -fwd, .030, .034, 'ere' + s, G, lab, mat, k=.010)
            if self.sp.get('armour'): continue
            # flank
            mu(C(sg * .118 * cw, .048, -.050), Pv(sg * .112 * self.b['hips'], .040, .060), r_ + fwd * .45, .052, .030, 'obl' + s, G, lab, mat, k=.016, box=.2)
            if lod > 0:
                for q, z in enumerate((.040, .004, -.034)):
                    mu(C(sg * (.142 - .004 * q) * cw, .020, z + .012), C(sg * (.122 - .006 * q) * cw, .082, z - .026), r_ * .9 + fwd * .5, .018, .017, 'ser%d' % q + s, G, lab, mat, k=.008)
            # abs: rows on a curve from the rib arch to the pubis
            A, B_, D_ = C(sg * .0345 * cw, .107 * cw, -.012), W_(sg * .0335 * ww, .099, -.010), Pv(sg * .024, .096, -.005)
            uu = (.08, .30, .52, .80) if lod else (.15, .5, .85)
            for q, u in enumerate(uu):
                p = (1 - u) ** 2 * A + 2 * u * (1 - u) * B_ + u * u * D_; p2 = p + ((1 - u) * (B_ - A) + u * (D_ - B_)) * .01
                ax = unit(p2 - p); n = unit(fwd * (1 - u) + sk.Rp[:, 1] * u + r_ * .25); e1 = np.cross(n, ax)
                ln = (.032, .031, .033, .056)[q] if lod else .045
                if q == 3 and self.sp['bottom'] == 'gi': continue
                self.c.egg(p - n * .002 * m, ax, e1, np.array([.036, ln, .020 * (.5 + .5 * self.M)]) * m * (1 if q < 3 else .92), G, lab, mat, k=.008 * k, box=.55, mg='ab%d' % q + s, name='ab')
            # gluteus
            if self.sp['bottom'] in ('suit', 'bare'):
                mu(Pv(sg * .040, -.070, .075), sk.hip[s] + unit(sk.kn[s] - sk.hip[s]) * .085 * k - sk.Rp[:, 1] * .040 * m + sk.Rp[:, 0] * sg * .010 * m, -sk.Rp[:, 1] + sk.Rp[:, 0] * sg * .2, .076, .062, 'glu' + s, G, lab, self.sp['cloth'] if self.sp['top'] != 'bare' else self.sp['skin'], k=.02, box=.1)

    # ------------------------------------------------------------------ arms
    def arm(self, s):
        sk, c, sp = self.sk, self.c, self.sp; sg = -1. if s == 'L' else 1.; ml = self.m * self.b['limb'] * self.b['armm']; k = self.k; lod = self.lod; kid = self.M < .3
        sh, el, wr = sk.sh[s], sk.el[s], sk.wr[s]; e1 = sk.e1a[s]; f = min(1., sk.bend[s] / 1.7); ua, fa = unit(el - sh), unit(wr - el); LA, LF = np.linalg.norm(el - sh), np.linalg.norm(wr - el)
        lat = np.cross(e1, ua) * sg; fr = -e1; e1f = unit(e1 - fa * (e1 @ fa)); latf = np.cross(e1f, fa) * sg; frf = -e1f; Rc = sk.Rc
        G = 'arm' + s; c.group(G, sh, .20 * self.m, .045 * k); U, F_ = 'uarm' + s, 'farm' + s
        suit = sp['top'] == 'suit'; shat = sp.get('shattered')
        mat = sp['skin'] if not (suit and not shat) else sp['cloth']; matf = mat
        a = lambda t, l, q: sh + ua * t * LA + (lat * l + fr * q) * ml
        d = lambda t, l, q: el + fa * t * LF + (latf * l + frf * q) * ml
        def mu(*z, **q): q.setdefault('s', ml); return self.mus(*z, **q)
        rl = []
        if shat:                                                                                            # ragged stump of a sleeve
            sd = 31 + (s == 'R'); cut = .22 if s == 'L' else .34
            rl = [(lambda t, p, q, sd=sd, cut=cut: ((q - sh) @ ua) < (cut + .16 * tear(p, sd, 9)) * LA, sp['cloth'])]
        pk = []; pkf = []
        if sp['skin'] == 'green':
            pk = [(lambda t, p, q: (((q - a(.56, 0, 0)) @ ua) / (.26 * LA)) ** 2 + (((q - a(.5, 0, 0)) @ lat) / (.050 * ml)) ** 2 < 1, 'pink')]; pkf = [(lambda t, p, q: ((((q - d(.36, 0, 0)) @ fa) / (.24 * LF)) ** 2 + (((q - d(.3, 0, 0)) @ frf - .0) / (.045 * ml)) ** 2 < 1) & (((q - d(.3, 0, 0)) @ latf) > 0), 'pink')]
        if kid:
            c.loft(sh, el, e1, table([(0, .050, .050, .048, .048), (.4, .046, .046, .046, .044), (1, .040, .040, .041, .039)], 2.1) * ml, G, U, mat, cap=(1., 1.1), k=.03 * k, side=-sg, name='ua' + s, mg='core' + s)
            c.loft(el, wr, e1f, table([(0, .041, .041, .041, .040), (.3, .043, .044, .043, .042), (.8, .034, .037, .034, .036), (1, .032, .036, .032, .035)], 2.1) * ml, G, F_, mat, cap=(1., 1.3), k=.03 * k, side=-sg, name='fa' + s, mg='core' + s)
            c.ball(sh + ua * .02 * ml, np.array([.054, .060, .054]) * ml, G, U, mat, ax=ua, e1=e1, k=.04 * k, mg='delt' + s)
        else:
            c.loft(sh, el, e1, table([(0, .046, .046, .044, .042), (.45, .040, .038, .040, .038), (1, .036, .037, .035, .035)], 2.1) * ml, G, U, mat, cap=(1., 1.05), k=.02 * k, side=-sg, name='ua' + s, mg='core' + s, rule=rl)
            up = Rc[:, 2]; ot = Rc[:, 0] * sg; fw = Rc[:, 1]
            # deltoid: three heads from clavicle, acromion, scapula to one point on the outside of the arm
            di = a(.50, .026, .002)
            mu(sh + (ot * .010 + up * .042) * ml, di, lat + up * .35 + ot * .25, .051, .054, 'delt' + s, G, U, mat, k=.009, skew=-.22, rest=.185, box=.1, rule=rl)
            mu(sh + (-ot * .044 + fw * .040 + up * .026) * ml, a(.46, .012, .024), fw * .7 + fr * .4 + lat * .2, .042, .040, 'deltf' + s, G, U, mat, k=.009, skew=-.1, rest=.18, rule=rl)
            mu(sh + (-ot * .044 - fw * .046 + up * .022) * ml, a(.44, .016, -.024), -fw * .7 + e1 * .4 + lat * .2, .042, .040, 'deltb' + s, G, U, mat, k=.009, skew=-.1, rest=.18, rule=rl)
            mu(a(.16, -.004, .022), a(1.0, -.002, .020), fr, .039, .040 + .014 * f, 'bic' + s, G, U, mat, k=.007, skew=.06 - .05 * f, ext=1.0 - .10 * f, lift=.004 + .006 * f, rule=rl + pk)
            mu(a(.08, -.014, -.020), a(.88, -.012, -.022), e1 - lat * .45, .038, .040, 'tri' + s, G, U, mat, k=.007, skew=-.05, rule=rl)
            mu(a(.22, .022, -.012), a(.86, .016, -.020), e1 * .55 + lat * .8, .032, .034, 'tril' + s, G, U, mat, k=.007, skew=-.05, rule=rl)
            if lod: mu(a(.50, .026, .006), d(.10, .016, .010), lat, .020, .020, 'bra' + s, G, U, mat, k=.010, rule=rl)
            c.loft(el, wr, e1f, table([(0, .036, .036, .036, .036), (.28, .038, .040, .040, .038), (.7, .029, .033, .029, .031), (1, .024, .033, .025, .031)], 2.15) * ml, G, F_, matf, cap=(1.05, 1.6), k=.018 * k, side=-sg, name='fa' + s, mg='core' + s, rule=pkf)
            mu(a(.78, .022, .006), d(.62, .022, .010), latf + frf * .5, .032, .035, 'brr' + s, G, F_, matf, k=.008, skew=-.15, rule=pkf)
            mu(d(-.05, -.018, .004), d(.64, -.016, .008), -latf * .8 + frf * .4, .037, .037, 'flx' + s, G, F_, matf, k=.008, skew=-.22, rule=pkf)
            mu(d(.0, .010, -.016), d(.72, .012, -.008), e1f * .8 + latf * .5, .031, .031, 'ext' + s, G, F_, matf, k=.008, skew=-.22, rule=pkf)
            c.ball(el + e1 * .018 * ml - fa * .004 * ml, np.array([.024, .028, .026]) * ml, G, F_, mat, ax=fa, e1=e1, k=.016 * k, mg='olec' + s, name='olec')
        if sp.get('band'):
            self.prim['band' + s] = c.loft(d(.60, 0, 0), d(.985, 0, 0), e1f, table([(0, .041, .044, .042, .043), (.12, .043, .046, .044, .045), (.88, .035, .041, .036, .040), (1, .034, .040, .035, .039)], 2.2) * ml * (1 if not kid else 1.08), 'band' + s, 'band' + s, sp['band'], cap=(0, 0), k=0, side=-sg, name='band' + s)
            if lod > 1: self.line(self.prim['band' + s], [.16, .13, .16], [-1.5, 0, 1.5], .7, pri=1); self.line(self.prim['band' + s], [.86, .89, .86], [-1.5, 0, 1.5], .7, pri=1); self.blade(self.prim['band' + s], (.3, .4), (.6, 1.3), .05, pri=1); self.blade(self.prim['band' + s], (.7, -1.2), (.45, -.3), .05, pri=1)
        if sp.get('glove'):
            g0 = .40; gl = table([(0, .062, .066, .063, .065), (.10, .060, .064, .061, .063), (.14, .055, .059, .056, .058), (.5, .047, .052, .048, .050), (1, .034, .041, .035, .040)], 2.2) * ml
            if lod: gl = gl + facet(61 + (s == 'R'), .0016 * ml, 5, 6) + ridge(TT, PH, .74, .5, .05, .06, 1.6, .004 * ml) + ridge(TT, PH, .86, -1.2, -.05, .06, 1.6, .004 * ml)
            rg = [(lambda t, p, q: t < -.0005, 'inside')]
            if shat and lod: gl = np.where(TT < .16 * tear(PH, 41 + (s == 'R'), 9), gl * .93, gl)
            self.prim['glv' + s] = c.loft(d(g0, 0, 0), wr + fa * .006 * ml, e1f, gl, 'glv' + s, 'glv' + s, sp['glove'], cap=(0, 1.6), k=0, side=-sg, name='glv' + s, rule=rg)
            if lod:
                i = self.prim['glv' + s]; self.line(i, [.135, .125, .135], [-1.7, 0, 1.7], .8, pri=0); self.line(i, [.135, .125, .135], [1.5, PI, -1.5], .8, pri=1)
                self.blade(i, (.95, .6), (.70, -.5), .05, pri=0); self.blade(i, (.90, 2.2), (.62, 1.2), .05, pri=0); self.blade(i, (.2, -1.0), (.46, -.2), .05, pri=1); self.blade(i, (.86, -1.8), (.64, -2.6), .05, pri=1)
        if sp['top'] == 'gi' and not (sp.get('torn') and s == 'L'):                                        # undershirt sleeve: the muscle wears the shirt, the hem stands off
            ce = .52 if not kid else .62; um = sp['under']; hem = lambda t, p, q: ((q - sh) @ ua) < (ce - .03) * LA
            for i, l_ in enumerate(c.lab):
                if l_ == U: c.rule[i] = [(hem, um)] + list(c.rule[i] or [])
            rs_ = Rc[:, 0] * sg; near = lambda Q: (((Q - sh) @ rs_) < .026 * self.m) & (np.linalg.norm(Q - sh, axis=-1) < .090 * self.m)
            for i, l_ in enumerate(c.lab):
                if l_ == U: c.rule[i] = list(c.rule[i] or []) + [(lambda t, p, q: near(q), sp['cloth'])]
            self.relabel = getattr(self, 'relabel', []) + [(U, near, 'gi')]
            mb = 1. if not kid else .82
            R = table([(0, .058, .064, .056, .054), (.45, .061, .066, .060, .056), (1, .066, .070, .064, .059)], 2.1) * ml * mb
            if lod: R = R + facet(71 + (s == 'R') * 3, .0022 * ml, 4, 6, (.2, 0))
            self.prim['slv' + s] = i = c.loft(sh + ua * (ce - .26) * LA, sh + ua * ce * LA, e1, R, 'slv' + s, 'slv' + s, um, cap=(1.1, 0), k=0, side=-sg, name='slv' + s, rule=[(lambda t, p, q: t > 1.0005, 'insideb')])
            if lod: self.line(i, [.80, .83, .80], [-1.6, 0, 1.6], .65, pri=1); self.line(i, [.80, .83, .80], [1.6, PI, -1.6], .65, pri=1); self.blade(i, (.05, 1.3), (.75, .8), .10, pri=0); self.blade(i, (.1, -1.2), (.70, -.6), .10, pri=1)

    # ------------------------------------------------------------------ hands
    HAND = dict(  # per finger (mcp, pip, dip) index..little, spread; thumb joints (thumb side, finger dir, back) in hand lengths
        relax=([(18, 24, 12), (24, 32, 16), (30, 38, 18), (36, 44, 20)], 2, ((.25, .30, -.13), (.28, .45, -.19), (.26, .57, -.22))), limp=([(14, 18, 10), (20, 24, 12), (26, 28, 14), (30, 32, 16)], 3, ((.26, .30, -.12), (.30, .44, -.17), (.31, .56, -.19))),
        fist=([(84, 102, 66), (86, 102, 66), (88, 102, 66), (90, 100, 64)], 0, ((.235, .27, -.15), (.175, .395, -.242), (.02, .415, -.252))), open=([(2, 8, 4), (0, 6, 4), (2, 8, 4), (6, 10, 6)], 8, ((.33, .27, -.07), (.46, .39, -.08), (.56, .49, -.07))),
        flat=([(2, 2, 2)] * 4, 1, ((.25, .30, -.09), (.275, .45, -.10), (.265, .57, -.09))), chop=([(6, 10, 6)] * 4, 0, ((.24, .28, -.14), (.22, .42, -.18), (.16, .52, -.20))), claw=([(-12, 62, 40), (-8, 60, 40), (-8, 62, 40), (-4, 62, 42)], 10, ((.32, .26, -.14), (.42, .39, -.22), (.38, .51, -.29))),
        point=([(0, 4, 2), (84, 98, 58), (86, 100, 58), (88, 100, 58)], 1, ((.24, .30, -.15), (.14, .40, -.262), (-.02, .39, -.268))), v=([(2, 5, 2), (-4, 4, 2), (84, 98, 58), (86, 100, 58)], 12, ((.23, .29, -.15), (.10, .38, -.262), (-.06, .37, -.268))),
        hold=([(58, 70, 36), (60, 72, 36), (62, 74, 36), (64, 76, 36)], 0, ((.25, .30, -.16), (.20, .42, -.30), (.07, .45, -.36))), splay=([(-8, 4, 2), (-10, 2, 2), (-8, 4, 2), (-4, 8, 4)], 14, ((.36, .24, -.05), (.52, .33, -.05), (.64, .41, -.03))),
        soft=([(8, 18, 10), (14, 22, 12), (18, 26, 12), (20, 30, 14)], 4, ((.29, .29, -.11), (.36, .43, -.16), (.37, .55, -.18))), cup=([(16, 34, 20), (18, 36, 20), (20, 36, 22), (22, 38, 24)], 6, ((.33, .26, -.14), (.43, .37, -.20), (.47, .48, -.25))),
        flick=([(34, 96, 50), (10, 14, 8), (14, 20, 10), (20, 28, 14)], 5, ((.28, .30, -.15), (.26, .44, -.26), (.18, .50, -.32))))

    def hand(self, s):
        sk, c = self.sk, self.c; hl = self.b['hand'] * self.H * float(self.fx.get('handscale' + s, 1.)); R = sk.Rhand[s]; th, fd, bk = R[:, 0], R[:, 1], R[:, 2]; o = sk.wr[s]; mat = self.sp.get('glove') or self.sp['skin']
        g = sk.grip[s]; G = 'hand' + s; pxh = self.px / self.H * hl * .9; fat = 1.10 if self.sp.get('glove') else 1.0; kid = self.M < .3
        if kid: fat *= 1.10
        if isinstance(g, str) and '>' in g:                                                              # 'fist>open:.4' = in-between
            a, r = g.split('>'); b, t = r.split(':'); t = float(t); A, B = self.HAND[a], self.HAND[b]
            sh = ([tuple(np.array(x) * (1 - t) + np.array(y) * t) for x, y in zip(A[0], B[0])], A[1] * (1 - t) + B[1] * t, tuple(tuple(np.array(x) * (1 - t) + np.array(y) * t) for x, y in zip(A[2], B[2])))
        else: sh = self.HAND.get(g, self.HAND['relax'])
        pal = table([(0, .115, .150, .115, .150), (.35, .105, .195, .120, .185), (.8, .085, .220, .100, .210), (1, .072, .212, .088, .202)], 2.8) * hl * .98 * (fat * .5 + .5)
        e = o + fd * hl * .50
        c.loft(o - fd * hl * .12, e, bk, pal, G, G, mat, cap=(1.2, 2.6), k=.012, side=1., name='palm' + s, cast=pxh > 26, mg='palm' + s)
        xs = (.165, .055, -.057, -.160); ln = (.40, .44, .41, .33); rr = (.058, .060, .057, .050); fing, spr, thb = sh
        fz = self.sp.get('top') == 'frieza'
        for q in range(4):
            a1, a2, a3 = (v * D2R for v in fing[q]); sp_ = (1.5 - q) * spr * D2R / 1.5
            b0 = o + fd * hl * (.485 - .03 * abs(q - 1.2)) + th * xs[q] * hl + bk * .015 * hl
            fdir = unit(fd * math.cos(sp_) + th * math.sin(sp_)); p = b0; A = 0.
            lab = G if pxh < 30 else G + 'f%d' % q; gr = G if pxh < 16 else lab
            if a1 > .9 and pxh > 24: c.ball(b0 + bk * .022 * hl + fd * .02 * hl, np.array([.062, .060, .055]) * hl * fat, G, G, mat, ax=fd, e1=th, k=.012, mg='knu' + s, name='knu', cast=False)
            for j, (an, lf) in enumerate(zip((a1, a2, a3), (.44, .31, .25))):
                A += an; d = fdir * math.cos(A) - bk * math.sin(A); e_ = p + d * ln[q] * hl * lf
                r0 = rr[q] * hl * fat * (1 - .09 * j); r1 = r0 * (.90 if j < 2 else .74)
                c.cap(p, e_, r0, r1, gr, lab, mat, k=.006, name=G + 'f%d%d' % (q, j), cast=False, mg='fi%d' % q)
                p = e_
            if fz and pxh > 34: c.ball(p + (bk * math.cos(A) + fdir * math.sin(A)) * r1 * .55, np.array([r1 * .75, r1 * .9, r1 * .5]), gr, lab, 'fnail', ax=d, e1=th, k=0, cast=False)
        b0 = o + fd * hl * .12 + th * .13 * hl - bk * .05 * hl; p1, p2, p3 = (o + th * v[0] * hl + fd * v[1] * hl + bk * v[2] * hl for v in thb); lab = G if pxh < 30 else G + 'th'; gr = G if pxh < 16 else lab
        c.cap(b0, p1, .092 * hl * fat, .068 * hl * fat, G, lab, mat, k=.014, name=G + 'th', cast=False, mg='th')
        for a, e_, r0, r1 in ((p1, p2, .066, .060), (p2, p3, .060, .044)): c.cap(a, e_, r0 * hl * fat, r1 * hl * fat, gr, lab, mat, k=.010, name=G + 'th', cast=False, mg='th')

    # ------------------------------------------------------------------ legs
    def legs(self):
        sk, c, sp, k, m = self.sk, self.c, self.sp, self.k, self.m; b = self.b; st = sp['bottom']; ml = m * b['limb'] * b['legm']; hp = b['hips']; kid = self.M < .3; lod = self.lod
        fwd, up = sk.Rp[:, 1], sk.Rp[:, 2]; mat = sp['cloth'] if st in ('gi', 'suit') else sp['skin']
        if sp['top'] == 'bare' and st == 'gi': mat = sp['cloth']
        PG = 'torso' if st != 'gi' else 'pelvis'; PL = 'torso' if st != 'gi' else 'trou'
        ez = 1.0 if st != 'gi' else 1.10
        if st == 'gi': xr = sk.Rp[:, 0]; pl = sk.pel; zr = sk.Rp[:, 2]; zb = .100 * (b['waistz'] - b['hipz']) / .093 * k; self.relabel = [('torso', lambda Q: (Q - pl) @ zr < zb, 'trou', mat), ('gi', lambda Q: (Q - pl) @ zr < zb, 'trou', mat), ('trou', lambda Q: (Q - pl) @ xr >= 0, 'trouR'), ('trou', lambda Q: (Q - pl) @ xr < 0, 'trouL')]
        self.prim['pelv'] = c.loft(self.Pv(0, 0, .050 if st != 'gi' else .112 * (b['waistz'] - b['hipz']) / .093), self.Pv(0, 0, -.075), fwd, table([(0, .092 * hp, .124 * hp, .092 * hp, .124 * hp), (.45, .094 * hp, .134 * hp, .104 * hp, .134 * hp), (1, .074 * hp, .100 * hp, .092 * hp, .100 * hp)] if st != 'gi' else [(0, .112 * hp, .134 * hp, .104 * hp, .134 * hp), (.3, .108 * hp, .140 * hp, .108 * hp, .140 * hp), (.6, .098 * hp, .140 * hp, .108 * hp, .140 * hp), (1, .074 * hp, .100 * hp, .092 * hp, .100 * hp)], 2.35) * m * ez, PG, PL, mat, cap=(1.3 if st != 'gi' else 0, 1.5), k=.045 * k, side=-1, name='pelv', mg='pel')
        for s, sg in (('L', -1.), ('R', 1.)):
            hip, kn, an = sk.hip[s], sk.kn[s], sk.an[s]; e1 = sk.e1l[s]; kb = min(1., sk.kbend[s] / 1.9); G = 'leg' + s; ta, sa = unit(kn - hip), unit(an - kn); LT, LS = np.linalg.norm(kn - hip), np.linalg.norm(an - kn)
            c.group(G, hip, .24 * m, .075 * k)
            e1t = unit(e1 - ta * (e1 @ ta)); lat = np.cross(ta, e1t) * sg; e1s = unit(e1 - sa * (e1 @ sa)); lats = np.cross(sa, e1s) * sg
            if st == 'gi': self.trouser(s, sg, hip, kn, an, e1, ta, sa, LT, LS, kb, ml, mat, G); continue
            t = lambda u, l, f: hip + ta * u * LT + (lat * l + e1t * f) * ml
            d = lambda u, l, f: kn + sa * u * LS + (lats * l + e1s * f) * ml
            rl = []; rs = []
            if sp.get('shattered'):
                h1 = lambda p_, r_, seed: (lambda tt, pp, q: ((((q - p_) @ ta) / r_[0]) ** 2 + (((q - p_) @ lat) / r_[1]) ** 2 + 1.5 * (tear(np.arctan2((q - p_) @ lat / r_[1], (q - p_) @ ta / r_[0]), seed, 13) - .42)) < 1.)
                c0 = t(.45, .02, .06) if s == 'R' else t(.70, -.02, .05); rl = [(h1(c0, (.085 * k, .030 * m), 51 + (s == 'R')), sp['skin'])]
                c1 = d(.22, .0, .04); rs = [(h1(c1, (.055 * k, .022 * m), 53 + (s == 'R')), sp['skin'])] if s == 'R' else []
            if sp['top'] == 'frieza':
                rs = [(lambda tt, pp, q: ((((q - d(.50, 0, 0)) @ sa) / (.30 * LS)) ** 2 + (((q - d(.5, 0, 0)) @ lats) / (.027 * ml)) ** 2 < 1) & (((q - d(.5, 0, 0)) @ e1s) > 0), 'fv')]
            def mu(*z, **q): q.setdefault('s', ml); return self.mus(*z, **q)
            T_, S_ = 'thigh' + s, 'shin' + s
            self.prim['th' + s] = c.loft(hip, kn, e1t, table([(0, .082, .080, .086, .072), (.4, .070, .066, .072, .062), (1, .047, .046, .047, .045)], 2.15) * ml, G, T_, mat, cap=(1, 1.05), k=.03 * k, side=sg, name='th' + s, mg='core' + s, rule=rl)
            mu(t(.04, .004, .046), t(.94, .0, .026), e1t, .046, .038, 'rec' + s, G, T_, mat, k=.009, skew=.0, rule=rl)
            mu(t(.08, .050, .004), t(.96, .032, .014), lat + e1t * .3, .048, .038, 'vl' + s, G, T_, mat, k=.009, rule=rl)
            mu(t(.46, -.026, .026), t(1.02, -.026, .016), -lat * .8 + e1t * .6, .034, .034, 'vm' + s, G, T_, mat, k=.008, skew=.22, rule=rl)
            mu(t(.0, -.034, .0), t(.70, -.032, -.004), -lat, .056, .034, 'add' + s, G, T_, mat, k=.02, rule=rl)
            mu(t(.08, .020, -.046), t(1.02, .022, -.018), -e1t + lat * .35, .042, .038, 'hamo' + s, G, T_, mat, k=.009, skew=-.1, rule=rl)
            mu(t(.08, -.020, -.044), t(1.02, -.024, -.016), -e1t - lat * .35, .042, .036, 'hami' + s, G, T_, mat, k=.009, skew=-.1, rule=rl)
            c.egg(kn + e1 * .034 * ml, unit(ta + sa), np.cross(e1, unit(ta + sa)), np.array([.027, .032, .019]) * ml, G, S_, mat, k=.012 * k, box=.5, mg='pat' + s, name='pat', rule=rs)
            self.prim['sn' + s] = c.loft(kn, an, e1s, table([(0, .044, .043, .043, .041), (.28, .038, .038, .042, .036), (.75, .030, .029, .031, .028), (1, .031, .031, .036, .030)], 2.15) * ml, G, S_, mat, cap=(1, 1.2), k=.02 * k, side=sg, name='sn' + s, mg='core' + s, rule=rs)
            mu(d(-.08, -.016, -.024), d(.58, -.012, -.022), -e1s - lats * .5, .039, .034, 'gasm' + s, G, S_, mat, k=.008, skew=-.05, rule=rs)
            mu(d(-.08, .018, -.024), d(.50, .016, -.020), -e1s + lats * .55, .036, .032, 'gasl' + s, G, S_, mat, k=.008, skew=-.05, rule=rs)
            mu(d(.22, .0, -.020), d(.92, .0, -.026), -e1s, .033, .024, 'sol' + s, G, S_, mat, k=.014, rule=rs)
            mu(d(.04, .020, .014), d(.82, .002, .014), e1s + lats * .7, .026, .022, 'tib' + s, G, S_, mat, k=.009, skew=-.2, rule=rs)

    def trouser(self, s, sg, hip, kn, an, e1, ta, sa, LT, LS, kb, ml, mat, G):
        """baggy gi trousers: hang from the belt, catch on the knee, bag over the boot; gravity and wind move the cloth, not the leg"""
        sk, c, sp, k, m = self.sk, self.c, self.sp, self.k, self.m; lod = self.lod; kid = self.M < .3
        tt = [(0, .094, .090, .106, .078), (.3, .106, .106, .112, .088), (.7, .104, .108, .106, .086), (1, .095, .096, .096, .082)]
        ts = [(0, .095, .096, .096, .082), (.3, .100, .106, .108, .090), (.62, .098, .104, .108, .090), (.84, .086, .090, .096, .080), (1, .058, .060, .064, .056)]
        if kid: tt = [(a, *(x * 1.02 for x in r)) for a, *r in tt]; ts = [(a, *(x * 1.0 for x in r)) for a, *r in ts]
        T1, T2 = table(tt, 2.2) * ml, table(ts, 2.2) * ml
        dn = np.array([0, 0, -1.]); lag = np.asarray(self.fx.get('cloth', np.zeros(3)), float); grav = np.asarray(self.fx.get('down', dn), float)
        def sagger(ax, T, amt, prof_):
            e1_ = unit(e1 - ax * (e1 @ ax)); e2_ = np.cross(ax, e1_) * sg
            dirs = np.cos(PH)[..., None] * e1_ + np.sin(PH)[..., None] * e2_
            g_ = np.maximum(dirs @ grav, -.35); l_ = np.maximum(dirs @ lag, -.5)
            hz = math.sqrt(max(0., 1 - (ax @ dn) ** 2))
            return T + prof_ * (amt * g_ * hz * ml + l_ * .045 * ml)
        b1 = np.sin(np.clip(TT * 1.0, 0, 1) * PI * .75) ; b2 = np.sin(np.clip(.25 + TT * .75, 0, 1) * PI)
        T1 = sagger(ta, T1, .016, b1); T2 = sagger(sa, T2, .030, b2)
        if lod:
            q = (.020 * kb + .004) * ml; fs = 51 + (s == 'R') * 7; fa_ = .0052 * ml
            T1 = T1 + facet(fs, fa_ * .8, 4, 5, (.2, 0)); T2 = T2 + facet(fs + 1, fa_ * .8, 4, 5)
            T1 = T1 + drape(fs + 100, fa_ * 1.6, 3, (.10, .95), (.12, .24), (.08, .13), (1.1, 2.0), sign=sg); T2 = T2 + drape(fs + 101, fa_ * 1.7, 3, (.15, .85), (.10, .22), (.08, .13), (1.1, 2.0), sign=-sg)
            for t0, p0, k_, w in ((.94, PI, .05, 1.), (.83, PI - .6, .10, .7)): T1 = T1 + ridge(TT, PH, t0, p0, k_, .07, 1.4, q * w)
            for t0, p0, k_, w in ((.08, PI, -.06, 1.), (.20, PI + .6, .10, .75)): T2 = T2 + ridge(TT, PH, t0, p0, k_, .07, 1.5, q * w)
            T2 = T2 * (1 + .07 * np.exp(-((TT - .80) / .12) ** 2) * (1 + .35 * np.cos(3 * PH + sg)))
        bt = dict(goku=.60, piccolo=.88, kid=.70).get(sp['shoe'], .7); an2 = kn + sa * bt * LS
        self.prim['th' + s] = u = c.loft(hip, kn, e1, T1, G, 'trou' + s, mat, cap=(1., 1.05), k=.04 * k, side=sg, name='th' + s)
        self.prim['sn' + s] = a = c.loft(kn, an2, e1, T2, G, 'trou' + s, mat, cap=(1., 0.), k=.05 * k, side=sg, name='sn' + s, rule=[(lambda t, p, q: t > 1.0005, 'insideo' if mat == 'gi' else 'inside')])
        c.ball(kn + e1 * .034 * ml, np.array([.060, .066, .060]) * ml, G, 'trou' + s, mat, ax=unit(ta + sa), e1=e1, k=.045 * k, name='knee' + s)
        c.ball(sk.hip[s] - sk.Rp[:, 1] * .040 * m + sk.Rp[:, 2] * .004 * k + sk.Rp[:, 0] * sg * .004, np.array([.078, .084, .074]) * m * self.b['hips'], 'pelvis', 'trou', mat, ax=sk.Rp[:, 2], e1=sk.Rp[:, 1], k=.05 * k)
        # leg inside: only the stretch between hem and boot
        c.loft(kn + sa * LS * .5, an, e1, table([(0, .036, .036, .042, .035), (1, .028, .028, .032, .027)], 2.1) * ml, 'sock' + s, 'sock' + s, dict(piccolo='cape').get(sp['shoe'], 'boot'), cap=(1, 1), k=.02, side=sg, name='sock' + s)
        if lod: self.trouser_folds(s, sg, kb, u, a)

    def trouser_folds(self, s, sg, kb, u, a):
        B = self.blade; k = .35 + .65 * kb; r = np.random.default_rng(7 + (s == 'R')); j = lambda v: v + r.uniform(-.04, .04)
        # from the belt: long drapes that never cross
        B(u, (.03, j(.55)), (.42, j(.34)), .05, 0, 0, 0, True, .9, False); B(u, (.02, j(1.25)), (.58, j(1.05)), .06, 0, 0, 0, True, .9); B(u, (.03, 2.60), (.46, 2.38), .05, 0, 0, 1, True, .85); B(u, (.04, -2.55), (.40, -2.78), .05, 0, 0, 0, True, .85)
        # the crotch pulls
        B(u, (.05, -1.40), (.30, j(-.50)), .055, 0, 0, 0, True, 1.05); B(u, (.15, -1.55), (.46, j(-.95)), .05, 0, 0, 1, True, .9)
        # the knee catches the cloth
        B(u, (.985, .50), (.62, j(.82)), .05, 0, 0, 0, True, 1.0); B(u, (.985, -.42), (.68, j(-.80)), .045, 0, 0, 1, True, .9)
        if kb > .22: B(u, (.975, PI - .40), (.80, PI - 1.75 * k), .055, 0, 0, 0, True, 1.1); B(u, (.92, PI + .45), (.74, PI + 1.45 * k), .05, 0, 0, 1, True, .95); B(a, (.04, PI + .35), (.24, PI + 1.75 * k), .06, 0, 0, 0, True, 1.05)
        B(a, (.03, .48), (.46, j(.80)), .06, 0, 0, 0, True, 1.0); B(a, (.05, -.45), (.40, j(-.82)), .05, 0, 0, 1, True, .9); B(a, (.22, 1.55), (.66, j(1.38)), .05, 0, 0, 1, True, .85, False)
        # gathers at the boot: short, uneven, with shadow
        for q, (p0, dp, t1) in enumerate(((.25, -.12, .72), (1.05, .16, .68), (-.80, .14, .75), (1.85, -.10, .78), (2.6, .10, .70), (-1.6, -.12, .70), (-2.4, .1, .78))): B(a, (.985, p0), (t1 + r.uniform(-.03, .03), p0 + dp), .055, 0, 0, 0 if q in (0, 1, 2, 4, 5) else 1, True, 1.0)

    # ------------------------------------------------------------------ gi
    def gi_top(self):
        sk, c, sp, m, k = self.sk, self.c, self.sp, self.m, self.k; b = self.b; cw, ww = b['chest'], b['waist']; mat = sp['cloth']; up, fwd, rt = sk.Rc[:, 2], sk.Rc[:, 1], sk.Rc[:, 0]; lod = self.lod; kid = self.M < .3; torn = sp.get('torn')
        zt = (b['neckz'] - b['chestz']) / .125
        S = [(0, .118, .140, .104, .140), (.25, .132, .152, .112, .152), (.5, .142, .162, .120, .162), (.70, .124, .160, .120, .160), (.86, .086, .118, .102, .118), (1, .060, .078, .074, .078)]
        if kid: S = [(0, .128, .142, .114, .142), (.25, .130, .146, .116, .146), (.5, .124, .146, .114, .146), (.72, .106, .136, .108, .136), (.88, .080, .104, .092, .104), (1, .060, .076, .070, .076)]
        Rt = table(S, 2.35) * m * cw
        if lod: Rt = Rt + facet(21, .0035 * m, 5, 7, (.1, .3)) + (drape(121, .005 * m, 4, (.05, .6), sign=1 if self.twist() > 0 else -1) if lod > 1 else 0)
        if torn:                                                                     # torn from the neck to the left armpit: the cloth is really gone there
            c.group('gi'); c.carve('gi', self.C(.030, 0, .205), unit(-rt * .64 + up * .77), unit(rt * .77 + up * .64), .011 * m, 13. / m)
            nn_ = unit(-rt * .64 + up * .77); oo_ = self.C(.030, 0, .205); ch_ = sk.chest
            self.relabel = getattr(self, 'relabel', []) + [('torso', lambda Q: ((Q - oo_) @ nn_ < -.014 * m) & ((Q - ch_) @ up < .185 * k), 'gi', mat)]
        rules = []
        if sp.get('under') and not kid or sp.get('disc') or sp['top'] == 'gip':
            o = sk.chest
            def vreg(t, p, q): d = q - o; x, z, y = d @ rt, d @ up, d @ fwd; return (y > -.01 * m) & (np.abs(x) < np.clip((z - .040 * k * zt) / (.170 * k * zt), 0, 1.3) * .070 * m) & (z > .040 * k * zt)
            rules = [(vreg, sp['under'] if sp['top'] == 'gi' else sp['skin'])]
        Gg = 'gi'
        self.prim['gi'] = rb = c.loft(self.C(0, -.004, -.105), self.C(0, -.02, .232 * zt), fwd, Rt, Gg, 'gi', mat, cap=(1.2, 1.2), k=.05 * k, side=-1, name='gi', rule=rules)
        fb = unit(sk.Rp[:, 1] + fwd)
        tb = table([(0, .108, .134 * ww, .102, .134 * ww), (.22, .122, .146 * ww, .112, .146 * ww), (.6, .116, .140 * ww, .106, .140 * ww), (1, .122, .144 * cw, .106, .144 * cw)], 2.3) * m
        if kid: tb = table([(0, .122, .134, .112, .134), (.25, .132, .142, .120, .142), (1, .128, .142, .114, .142)], 2.3) * m
        pit = math.degrees(math.acos(float(np.clip(sk.Rc[:, 2] @ sk.Rp[:, 2], -1, 1)))); lag = np.asarray(self.fx.get('cloth', np.zeros(3)), float)
        if lod:
            tb = tb + facet(22, .004 * m, 4, 7) + drape(122, .006 * m, 4, (.15, .85), sign=1 if self.twist() > 0 else -1); am = .014 * m * min(1, pit / 35)
            if am > .002:
                d = sk.Rp[:, 2] - sk.Rc[:, 2] * (sk.Rp[:, 2] @ sk.Rc[:, 2]); ph0 = math.atan2(-(sk.Rc[:, 2] @ sk.Rp[:, 0]), sk.Rc[:, 2] @ sk.Rp[:, 1])
                for q, t0 in enumerate((.42, .58, .74)): tb = tb + ridge(TT, PH, t0, ph0 + .25 * (-1) ** q, .05 * (-1) ** q, .07, 1.2, am)
        self.prim['gib'] = bl = c.loft(self.Pv(0, 0, .105 * (b['waistz'] - b['hipz']) / .093), self.C(0, 0, -.02), fb, tb, Gg, 'gi', mat, cap=(1.0, 1.2), k=.06 * k, side=-1, name='gib', rule=rules)
        for s, sg in (('L', -1.), ('R', 1.)):
            r_ = rt * sg
            if not kid: c.egg(self.C(sg * .078, .098, .070 if not (torn and s == 'L') else .058), unit(up + r_ * .2), r_, np.array([.074, .070, .050]) * m, Gg, 'gi', mat, k=.04 * k, box=.2, name='pecg', rule=rules)
            if torn and s == 'L': continue
            q0 = sk.neck + up * -.002 * k - fwd * .010 * m + r_ * .040 * m; q1 = sk.sh[s] + r_ * (.030 if not kid else .018) * (1 - .45 * min(1, sk.lift[s])) * m + up * .004 * k
            rr = table([(0, .050, .064, .062, .064), (.5, .060, .076, .070, .076), (1, .064, .077, .070, .077)], 2.2) * m * (1 if not kid else .82)
            if lod: rr = rr + ridge(TT, PH, .62, .4, .12, .12, 1.4, .006 * m)
            sh_ = sk.sh[s]; self.prim['str' + s] = i = c.loft(q0, q1, up, rr, Gg, 'gi', mat, cap=(1, 2.4), k=.035 * k, side=1, name='str' + s, rule=rules + [(lambda t, p, q, sh_=sh_: (t > 1.0005) & (((q - sh_) @ up) < -.020 * m), 'insideo' if mat == 'gi' else 'inside')])
            # cloth between arm and ribs so that a raised arm pulls the vest
            c.loft(self.C(sg * .100 * cw, .0, .060), sk.sh[s] + (sk.el[s] - sk.sh[s]) * .05, fwd, table([(0, .092, .06, .092, .06), (1, .060, .052, .060, .052)], 2) * m * (1 if not kid else .85), Gg, 'gi', mat, cap=(1, 0), k=.05 * k, name='gpit' + s)
            if lod:
                self.line(i, [.90, .88, .90], [-1.2, 0, 1.2], .7, pri=1)
                self.blade(i, (.95, 1.2), (.30, .7), .09, .3, 0., 1); self.blade(i, (.95, -1.4), (.30, -1.0), .09, .3, 0., 1)
        # lapels: folded bands laid on the finished shell
        c.pack()
        def band(pts, w0, w1, lift, label, mt, labs=('gi',)):
            pts = np.asarray(pts, float); t = np.linspace(0, 1, 13); i = np.linspace(0, 1, len(pts)); x, z = np.interp(t, i, pts[:, 0]), np.interp(t, i, pts[:, 1]); dx, dz = np.gradient(x), np.gradient(z); L_ = np.hypot(dx, dz); nx, nz = -dz / L_, dx / L_; w = (w0 + (w1 - w0) * t); R_ = []
            for q in (-.5, 0., .5):
                O_, D_ = [], []
                for x_, z_, nx_, nz_, w_ in zip(x, z, nx, nz, w):
                    zz = z_ + nz_ * w_ * q; tu = float(np.clip((z_ - .150) / .095, 0, 1)); dd = unit(fwd * (1 - tu) + up * tu * 1.15); O_.append(self.C(x_ + nx_ * w_ * q, .10 * (1 - tu) + .010 * tu - .050 * max(0, z_ - .235) / .02, min(zz, .250) * zt) + dd * .30 * m); D_.append(dd)
                O_, D_ = np.array(O_), np.array(D_); P3, ok = c.drop(O_, -D_, .75, labels=labs); R_.append(np.where(ok[:, None], P3 + D_ * (lift * (1. if q else 1.35)) * m, np.nan))
            Q = np.stack(R_, 1); g = ~np.isnan(Q).any((1, 2)); Q = Q[g]
            if len(Q) > 3: c.sheet(Q, label, mt)
        if not kid or sp.get('disc'):
            W_ = 1. if not kid else .92
            Rl = [(.078 * W_, .262), (.080 * W_, .232), (.073 * W_, .190), (.057 * W_, .140), (.029 * W_, .090), (-.006, .038)]; Ll = [(-.078 * W_, .262), (-.080 * W_, .232), (-.073 * W_, .190), (-.057 * W_, .140), (-.029 * W_, .090), (.004, .038), (.044, -.028), (.072, -.095), (.085, -.160)]
            if sp['top'] == 'gip': Ll = Ll[:6]
            band(Rl, .030, .026, .006, 'lapr', mat)
            if not torn: band(Ll, .030, .026, .010, 'lapl', mat)
        if not kid and not torn:
            ph = np.arange(NP) * 2 * PI / NP; rn = .058 * m * b['neck']
            c.ring(sk.neck - up * .010 * k + fwd * .004 * m, unit(up + fwd * .40), fwd, rn * (1.46 + .10 * np.cos(ph)), .013 * m, 0., .8, 'ucol', 'gcol', mat, k=0, name='gcol', rule=[(lambda t, p, q: np.abs(wrap(p)) < 1.35, sp['under'] or sp['skin'])])
        # undershirt collar round the neck base
        if sp.get('under') and sp['top'] == 'gi' and not (torn and False):
            ph = np.arange(NP) * 2 * PI / NP; rn = .058 * m * b['neck'] * (1. if not kid else .9)
            c.ring(sk.neck + up * -.006 * k + fwd * .010 * m, unit(up + fwd * .62), fwd, rn * (1.22 + .16 * np.cos(ph)), .014 * m, 0., .9, 'ucol', 'ucol', sp['under'], k=0, name='ucol')
        if sp.get('disc'): self.discs.append(dict(ch=sp['disc'], frame='chest', x=-.083 * (1 if not kid else .8), z=.088 * zt, r=.036 * (1 if not kid else 1.1), side=1, skip=torn)); self.discs.append(dict(ch=sp['disc'], frame='chest', x=0., z=.075 * zt, r=.094 * (1 if not kid else .85), side=-1, skip=False))
        if lod: self.gi_folds(rb, bl, torn)

    def twist(self):
        sk = self.sk; a = sk.Rp[:, 1]; b = sk.Rc[:, 1]; u = sk.Rp[:, 2]
        a = unit(a - u * (a @ u)); b = unit(b - u * (b @ u)); return math.degrees(math.atan2(np.cross(a, b) @ u, a @ b))

    def gi_folds(self, rb, bl, torn):
        B = self.blade; tw = self.twist(); sk = self.sk
        for s, sg in (('L', -1), ('R', 1)):
            up = min(1.2, sk.lift[s]); k = .75 + .55 * up; p0 = -sg * 1.24; t0 = .66 + .05 * up                 # note: phi runs to the left (side=-1), so right = -phi
            if not (torn and s == 'L'):
                B(rb, (t0, p0), (t0 - .20 * k, p0 + sg * .85 * k), .060, .28, 0., 0); B(rb, (t0 - .14, p0 - sg * .05), (t0 - .46 * k, p0 + sg * .45 * k), .055, .30, 0., 0)
            p1 = -sg * 1.95
            B(rb, (t0 + .03, p1), (t0 - .22, p1 - sg * .70), .060, .28, 0., 0); B(rb, (t0 - .08, p1 - sg * .05), (t0 - .40, p1 - sg * .42), .05, .3, 0., 1)
        d = sk.Rc[:, 2]; hx, hy = -(d @ sk.Rp[:, 0]), d @ sk.Rp[:, 1]; bk = math.hypot(hx, hy); ph0 = math.atan2(hx, hy) if bk > .14 else 1.0; g = .8 + .8 * min(1., bk / .5)
        for q, (t0, side, dt, ln, w) in enumerate(((.80, 1, -.06, .95, .075), (.62, -1, -.10, 1.10, .085), (.44, 1, -.10, 1.15, .085), (.28, -1, -.06, .9, .07))):
            p0 = ph0 + side * 1.0; ln *= g
            B(bl, (t0, p0), (t0 + dt, p0 - side * ln), w * (.8 + .4 * g), .30, 0., 0 if q < 3 else 1)
            B(bl, (t0 - .03, p0 + PI), (t0 + dt * .6, p0 + PI + side * ln * .8), w * .8, .30, 0., 1)
        # gathers rising out of the belt
        for q, (p0, dp, ln) in enumerate(((.35, .10, .34), (-.45, -.16, .40), (1.15, .18, .36), (-1.25, -.12, .42), (2.1, .1, .36), (-2.2, -.1, .40), (2.9, 0, .30))): B(bl, (.015, p0), (ln, p0 + dp), .045, .3, 0., 0 if q < 4 else 1)
        if abs(tw) > 10:
            sgn = 1 if tw > 0 else -1; kk = min(1.3, abs(tw) / 38.)
            for q, (t0, p0) in enumerate(((.06, -.75), (.26, -.25), (.04, .28), (.36, .55))): B(rb, (t0, p0 * sgn), (t0 + .28 * kk + .06, (p0 + .78 * kk) * sgn), .060, .34, 0., 0 if q < 3 else 1)

    def belt(self):
        sk, c, sp, m, k = self.sk, self.c, self.sp, self.m, self.k; b = self.b; ww = b['waist']; bare = sp['top'] == 'bare'; kid = self.M < .3; zz = (b['waistz'] - b['hipz']) / .093
        fb = unit(sk.Rp[:, 1] * 1.4 + sk.Rw[:, 1]); a = self.Pv(0, .0, .078 * zz); e = self.Pv(0, .002, (.126 if sp['belt'] != 'sashb' else .150) * zz)
        q = 1.06 if not bare else 1.04
        R = table([(0, .124, .150 * ww, .118, .150 * ww), (.15, .128, .154 * ww, .121, .154 * ww), (.85, .124, .148 * ww, .116, .148 * ww), (1, .119, .143 * ww, .112, .143 * ww)], 2.3) * m * q * (1 if not kid else 1.04)
        self.prim['belt'] = i = c.loft(a, e, fb, R, 'belt', 'belt', sp['belt'], cap=(0, 0), k=0, side=-1, name='belt', rule=[(lambda t, p, q_: (t > 1.0005) | (t < -.0005), 'inside')])
        if self.lod: self.line(i, [.52, .48, .55], [-2.2, -.6, -.15], .75, pri=0); self.line(i, [.35, .55, .60], [.2, .9, 2.0], .7, pri=1); self.line(i, [.70, .66, .74], [1.8, 2.6, 3.6], .7, pri=1)
        # waist jacket tied (h)
        fr = fb; rt = sk.Rp[:, 0]; kp = (a + e) / 2 + fr * (.126 * m * q + .004) + rt * .024 * m
        if sp['belt'] != 'sashb' or True:
            r = .024 * m
            c.ball(kp, np.array([r * 1.15, r * 1.0, r * .8]), 'knot', 'knot', sp['belt'], ax=sk.Rp[:, 2], e1=rt, k=.0, name='knot')
            c.egg(kp + rt * .026 * m + sk.Rp[:, 2] * .004 * m, rt + sk.Rp[:, 2] * .25, sk.Rp[:, 2], np.array([.016, .024, .012]) * m, 'knot', 'knot', sp['belt'], k=.004, name='knot')
            c.egg(kp - rt * .026 * m + sk.Rp[:, 2] * .002 * m, rt - sk.Rp[:, 2] * .2, sk.Rp[:, 2], np.array([.016, .024, .012]) * m, 'knot', 'knot', sp['belt'], k=.004, name='knot')
            lag = np.asarray(self.fx.get('belt', np.zeros(3)), float); f = self.fx.get('f', 0)
            for q_, (sx, ln, spd) in enumerate(((.012, .30, .26), (-.014, .235, -.30))): self.ribbon(kp + rt * sx * m - sk.Rp[:, 2] * .008 * m + fr * .006 * m, fr, rt, ln * m * (1 if not kid else .75), .026 * m * (1 if not kid else .85), sp['belt'], 'bend%d' % q_, lag, f, q_, body=True, spread=spd)
        if sp.get('jacket'):                                                                # gi top knotted round the hips (h shots)
            base = table([(0, .126, .156, .126, .156), (.3, .138, .176, .146, .176), (1, .120, .190, .160, .190)], 2.2) * m + facet(91, .008 * m, 5, 7) + drape(92, .012 * m, 6)
            self.prim['jk'] = c.loft(self.Pv(0, -.006, .10), self.Pv(0, -.030, -.16), fb, base * np.where(np.cos(PH) > .2, np.clip(1 - .55 * TT * (np.cos(PH) - .2) / .8, .3, 1), 1), 'jk', 'jk', 'gi', cap=(0, 0), k=0, side=-1, name='jk', rule=[(lambda t, p, q_: (t > 1.0005), 'insideo')])
            if self.lod:
                for p0 in (1.2, 1.8, 2.5, 3.1, -2.5, -1.8, -1.2): self.blade(self.prim['jk'], (.05, p0), (.8, p0 + .1), .06, .3)
            for sx in (-1, 1): self.ribbon(kp + rt * sx * .02 * m, fr, rt, .34 * m, .040 * m, 'gi', 'jks%d' % (sx > 0), lag, f, 5 + sx, body=True, spread=.3 * sx, back='gid')

    def ribbon(self, p, fr, sd, ln, w, mat, label, lag, f, q=0, n=12, body=False, spread=0., back=None):
        """hanging strip of doubled cloth: falls, lags behind the motion, flutters, turns its edge to us"""
        c, m, sk = self.c, self.m, self.sk; grav = np.asarray(self.fx.get('down', (0, 0, -1.)), float); d0 = unit(-sk.Rp[:, 2] * .75 + fr * .50 + sd * spread); p0 = np.asarray(p, float); L = float(np.linalg.norm(lag)); C_ = [p0]; D_ = [d0]
        for i in range(n):
            u = i / (n - 1.); wv = math.sin(f * .9 - i * .62 + q * 2.1) * (.06 + .34 * u) * min(1.2, .10 + L)
            dd = unit(d0 * (1 - u) ** 1.3 * (1 - .5 * min(1, L)) + (grav * (.5 + u)) / (1 + L * 1.6) + lag * (1.05 + .6 * u) + sd * wv + fr * wv * .4)
            p1 = p0 + dd * ln / n
            if body:                                                                          # keep it outside the hips and thighs
                for s in 'LR':
                    h_, k_ = sk.hip[s], sk.kn[s]; t = np.clip((p1 - h_) @ (k_ - h_) / ((k_ - h_) @ (k_ - h_)), 0, 1); x = p1 - (h_ + (k_ - h_) * t); r_ = (.118 - .010 * t) * m; nx = np.linalg.norm(x)
                    if nx < r_: p1 = p1 + x / (nx + 1e-6) * (r_ - nx)
            C_.append(p1); D_.append(unit(p1 - p0)); p0 = p1
        C_, D_ = np.array(C_), np.array(D_); D_[0] = D_[1]; u = np.linspace(0, 1, n + 1); rows = []
        for i in range(n + 1):
            tw = .25 * math.sin(f * .7 - i * .5 + q * 1.7) * min(1., .25 + L) * u[i] + (.18 if q % 2 else -.14); e = sd - D_[i] * (sd @ D_[i]); e = unit(e); nn = np.cross(D_[i], e); e2 = e * math.cos(tw) + nn * math.sin(tw); ww = w * (.72 + .40 * min(1., u[i] * 2.2)); cup = nn * ww * .22
            rows.append([C_[i] - e2 * ww, C_[i] - e2 * ww * .35 + cup * .5, C_[i] + e2 * ww * .30 + cup * .4, C_[i] + e2 * ww])
        Q = np.array(rows); Q[-1, :, :] += D_[-1] * (np.array([-.30, .0, .1, .36])[:, None] * w * (1 if q % 2 else -1)); c.sheet(Q, label, mat, back)

    # ------------------------------------------------------------------ armour
    def armour(self):
        sk, c, m, k = self.sk, self.c, self.m, self.k; up, fwd, rt = sk.Rc[:, 2], sk.Rc[:, 1], sk.Rc[:, 0]; lod = self.lod; W_, G_ = 'armw', 'armg'
        S = [(0, .140, .168, .130, .168), (.3, .150, .180, .140, .180), (.62, .156, .186, .144, .186), (1, .136, .174, .134, .174)]
        self.prim['arm'] = i = c.loft(self.C(0, -.006, -.075), self.C(0, -.012, .128), fwd, table(S, 2.5) * m, 'armour', 'armour', W_, cap=(0, 0), k=.02 * k, side=-1, name='armour', rule=[(lambda t, p, q: (t > 1.0005) | (t < -.0005), 'inside')])
        for sg in (-1, 1):
            r_ = rt * sg
            c.egg(self.C(sg * .076, .104, .066), unit(rt + up * .08 * sg), up, np.array([.066, .076, .058]) * m, 'armour', 'armour', W_, k=.016 * k, box=.45, name='apec', mg='apec%d' % sg)
            c.egg(self.C(sg * .074, -.094, .070), unit(rt), up, np.array([.066, .074, .044]) * m, 'armour', 'armour', W_, k=.020 * k, box=.4, name='aback', mg='abk%d' % sg)
            # strap over the shoulder
            P_ = [(.088, .128, .118), (.098, .104, .172), (.104, .050, .212), (.104, -.020, .226), (.102, -.086, .208), (.096, -.130, .160), (.090, -.146, .104)]
            P_ = [self.C(sg * x, y, z) + (sk.sh['R' if sg > 0 else 'L'] - self.C(sg * .2112 * self.b['sw'] / .12 * self.k / self.m, -.0106, .160)) * .45 for x, y, z in P_]
            for a_, b_ in zip(P_[:-1], P_[1:]):
                dd = unit(b_ - a_); out = unit((a_ + b_) / 2 - self.C(sg * .10, 0, .10)); e_ = np.cross(dd, out)
                c.loft(a_ - dd * .004, b_ + dd * .004, e_, table([(0, .030, .013, .030, .013), (1, .030, .013, .030, .013)], 3.0) * m, 'strap%d' % sg, 'strap%d' % sg, W_, cap=(1, 1), k=.012 * k, name='strap')
        # golden abdomen: stacked plates
        fb = unit(sk.Rp[:, 1] + fwd); T = table([(0, .136, .160, .128, .160), (.12, .136, .158, .126, .158), (.5, .128, .150, .116, .150), (1, .138, .164, .122, .164)], 2.35) * m
        if lod:
            for t0 in (.20, .44, .68): T = T - .004 * m * bump(TT - t0, .045)
        rl = [(lambda t, p, q: (np.abs(wrap(p)) < .80 + .10 * np.cos(t * 3)) & (t < 1.0005) & (t > -.0005), G_), (lambda t, p, q: (t > 1.0005) | (t < -.0005), 'inside')]
        self.prim['armab'] = j = c.loft(self.Pv(0, .002, .060), self.C(0, 0, -.060), fb, T, 'armab', 'armab', W_, cap=(0, 0), k=0, side=-1, name='armab', rule=rl)
        if lod:
            for t0 in (.20, .44, .68): self.line(j, [t0 + .02, t0, t0 + .02], [-.80, 0, .80], .9, pri=0); self.line(j, [t0 + .02, t0, t0 + .02], [.95, PI, -.95], .7, pri=1)
            self.line(j, [.03, .5, .97], [0, 0, 0], .7, pri=1, straight=True)
            self.line(i, [.02, .35, .62, .80], [0, .01, 0, 0], 1.0, pri=0)
            self.line(i, [.94, .92, .94], [-1.5, 0, 1.5], .6, pri=2)

    # ------------------------------------------------------------------ neck, scarf, cape
    def neck_(self):
        sk, c, m, k = self.sk, self.c, self.m, self.k; r = .058 * m * self.b['neck']; kid = self.M < .3
        a = sk.neck - sk.Rc[:, 2] * .030 * self.H; e = sk.headb + sk.Rn[:, 2] * .034 * self.H; mat = self.sp['skin']; G = 'neck'; c.group(G, sk.neck, .15 * m, .05 * k if self.sp['top'] in ('bare', 'frieza') or self.sp.get('torn') else 0.)
        if self.sp['top'] == 'suit':                                                         # high collar of the bodysuit
            c.loft(sk.neck - sk.Rc[:, 2] * .035 * k, sk.neck + unit(sk.Rn[:, 2] + sk.Rn[:, 1] * .12) * .030 * k, sk.Rn[:, 1], table([(0, r * 1.42, r * 1.50, r * 1.34, r * 1.50), (.5, r * 1.24, r * 1.30, r * 1.24, r * 1.30), (1, r * 1.16, r * 1.22, r * 1.20, r * 1.22)], 2.1), 'scol', 'scol', self.sp['cloth'], cap=(1, 0), k=0, side=-1, name='scol', rule=[(lambda t, p, q: t > 1.0005, 'insideb')])
        c.loft(a, e, sk.Rn[:, 1], table([(0, r * 1.05, r * 1.15, r * 1.12, r * 1.15), (.5, r * .88, r * .97, r, r * .97), (1, r * .90, r, r * 1.05, r)], 2.1), G, 'neck', mat, cap=(1, 1), k=.02 * k, side=-1, name='neck', mg='neck')
        if not kid and self.sp['top'] != 'suit':
            u = sk.u
            for s, sg in (('L', -1), ('R', 1)):
                c.cap(sk.head + sk.Rh @ np.array([sg * .55, -.10, -.50]) * u, (self.C(sg * .020, .068, .198) if self.sp['top'] in ('bare', 'frieza') or self.sp.get('torn') else sk.neck + sk.Rc @ np.array([sg * .018, .040, -.004]) * m), .0165 * m * self.b['neck'], .0125 * m, G, 'neck', mat, k=.010 * k, mg='scm' + s, name='scm', cast=False)

    def scarf(self):
        sk, c, m, k = self.sk, self.c, self.m, self.k; n0 = sk.neck; up = unit(sk.Rc[:, 2] + sk.Rn[:, 2] * .5); fwd, rt = sk.Rc[:, 1], sk.Rc[:, 0]; ph = np.arange(NP) * 2 * PI / NP
        c.ring(n0 + up * .004 * self.H, unit(up + fwd * .20), fwd, .080 * m * (1 + .10 * np.cos(ph)), .034 * m * (1 + .14 * np.cos(2 * ph + .4)), -.014 * m * np.maximum(np.cos(ph), 0) ** 2, .9, 'scarf', 'scarf1', 'scarf', k=.006, name='scarf1')
        c.ring(n0 + up * .026 * self.H, unit(up + fwd * .05 - rt * .10), fwd, .072 * m * (1 + .10 * np.cos(ph)), .026 * m * (1 + .18 * np.cos(2 * ph + 2.4)), -.008 * m * np.maximum(np.cos(ph), 0) ** 3, .8, 'scarf', 'scarf2', 'scarf', k=.004, name='scarf2')

    def cape(self, kind):
        """white cape: wide padded shoulders and a heavy sheet behind.  kind True | 'torn'"""
        sk, c, m, k = self.sk, self.c, self.m, self.k; up, fwd, rt = sk.Rc[:, 2], sk.Rc[:, 1], sk.Rc[:, 0]; lag = np.asarray(self.fx.get('cape', np.zeros(3)), float); f = self.fx.get('f', 0); grav = np.asarray(self.fx.get('down', (0, 0, -1.)), float); torn = kind == 'torn'; ph = np.arange(NP) * 2 * PI / NP
        c.ring(sk.neck + up * -.004 * k, unit(up + fwd * .30), fwd, .098 * m, .044 * m, 0., .9, 'cape', 'capec', 'cape', k=.02, name='capeng')
        for s, sg in (('L', -1), ('R', 1)):
            if torn and s == 'L': continue
            r_ = rt * sg; a = sk.neck + up * -.006 * k + r_ * .07 * m; e = sk.sh[s] + r_ * .085 * m + up * .012 * k
            c.loft(a, e, up, table([(0, .050, .090, .050, .090), (.6, .062, .112, .050, .112), (1, .052, .112, .034, .112)], 2.4) * m, 'cape', 'capes' + s, 'cape', cap=(1, 1.5), k=.03 * k, name='capes' + s)
        # the sheet: hangs from the shoulder line, folds fan out from the neck, lags behind the body, never enters it
        nv, nu = 15, 33; ln = (1.02 if not torn else .62) * self.H * .60; L = float(np.linalg.norm(lag)); u = np.linspace(-1, 1, nu); Q = np.zeros((nv, nu, 3)); rng = np.random.default_rng(5); fph = rng.uniform(0, 6, 3)
        th = u * 1.28; top = np.array([self.C(.185 * math.sin(t_), -.045 - .085 * math.cos(t_), .205 - .050 * abs(math.sin(t_)) ** 1.5) for t_ in th]); outd = np.array([rt * math.sin(t_) * .55 - fwd * math.cos(t_) for t_ in th]); outd /= np.linalg.norm(outd, axis=1)[:, None]
        fold = np.sin(u * 8.5 + fph[0]) * .6 + np.sin(u * 14. + fph[1]) * .4; fold = np.sign(fold) * np.abs(fold) ** .75; pos = top.copy(); Q[0] = pos
        for i in range(1, nv):
            v = i / (nv - 1.); wv = np.sin(f * .55 - i * .45 + u * 2.2)[:, None] * (.04 + .14 * v) * min(1.4, .30 + L)
            d = outd * (.34 * (1 - v) ** 1.5 + .06) + grav * (.95 + .4 * v) / (1 + 1.3 * L) + lag * (.45 + 1.0 * v) + outd * wv; d /= np.linalg.norm(d, axis=1)[:, None]; pos = pos + d * ln / (nv - 1)
            q_ = pos - sk.pel; y_ = q_ @ sk.Rp[:, 1]; x_ = q_ @ sk.Rp[:, 0]; need = -(.150 * m) * np.sqrt(np.clip(1 - (x_ / (.30 * m)) ** 2, 0, 1)); push = np.maximum(0, y_ - need) * (np.abs(x_) < .30 * m); pos = pos - sk.Rp[:, 1] * push[:, None]
            Q[i] = pos + outd * (fold * (.010 + .060 * v ** .8) * m)[:, None] + rt * ((u * .10 * v ** 1.2) * m)[:, None]
        hole = None
        if torn:
            rr = np.random.default_rng(17); tx = np.interp(u[:-1], np.linspace(-1, 1, 17), rr.uniform(.25, 1., 17)); vv = (np.arange(nv - 1) + .5) / (nv - 1); hole = vv[:, None] > tx[None] * .95 + .05; hole |= (rr.random((nv - 1, nu - 1)) < .04) & (vv[:, None] > .4)
        c.sheet(Q, 'capesh', 'cape', 'cape', hole)
        for j in range(2, nu - 2):
            if (fold[j] < fold[j - 1] and fold[j] < fold[j + 1]) or (fold[j] > fold[j - 1] and fold[j] > fold[j + 1] and j % 2):
                i0 = int(rng.integers(1, 5)); i1 = int(rng.integers(9, nv)); self.seams.append(dict(Q=Q[i0:i1, j], w=1. if fold[j] < 0 else .75, pri=0 if fold[j] < 0 else 1, labels=('capesh',), tol=.05, taper=(.25, .6)))

    # ------------------------------------------------------------------ Frieza
    def frieza_parts(self):
        sk, c, m, k = self.sk, self.c, self.m, self.k; rt, fwd, up = sk.Rc[:, 0], sk.Rc[:, 1], sk.Rc[:, 2]; o = sk.chest
        # breast shield as a material region on the trunk
        def shield(t, p, q):
            d = q - o; x, z, y = np.abs(d @ rt) / m, (d @ up) / k, d @ fwd
            return (y > 0) & (z < .034) & (x < np.interp(z, [-.112, -.085, -.045, .0, .034], [.0, .030, .054, .062, .062]))
        for i, (lb, nm) in enumerate(zip(c.lab, c.name)):
            if lb == 'torso' and (c.mg[i] or '').startswith(('ab', 'rib', 'bel', 'ser', 'obl')): c.rule[i] = (c.rule[i] or []) + [(shield, 'fv')]
        ml = m * self.b['limb']
        for s, sg in (('L', -1.), ('R', 1.)):
            sh, el, wr = sk.sh[s], sk.el[s], sk.wr[s]; ua = unit(el - sh); fa = unit(wr - el); e1 = sk.e1a[s]; lat = np.cross(e1, ua) * sg; e1f = unit(e1 - fa * (e1 @ fa)); latf = np.cross(e1f, fa) * sg
            n = unit(rt * sg * .9 + up * .55 + lat * .5)
            c.ball(sh + n * .016 * m + ua * .010 * m, np.array([.066, .068, .064]) * m, 'dome' + s, 'dome' + s, 'fv', ax=unit(ua + rt * sg * .6), e1=n, k=0, name='dome')
            ph = np.arange(NP) * 2 * PI / NP
            c.ring(sk.wr[s] - fa * .016 * m, fa, e1f, .034 * ml * 1.0, .011 * m, 0., 1., 'cuff' + s, 'cuff' + s, 'fw', k=0, name='cuff')
            LF = np.linalg.norm(wr - el)
            for i, nm in enumerate(c.lab):
                if nm == 'farm' + s: c.rule[i] = (c.rule[i] or []) + [(lambda t, p, q, el=el, fa=fa, e1f=e1f, latf=latf, LF=LF: ((((q - el - fa * LF * .55) @ fa) / (.30 * LF)) ** 2 + (((q - el) @ e1f) / (.020 * ml)) ** 2 < 1) & (((q - el) @ latf) > .0), 'fv')]
            sa = unit(sk.an[s] - sk.kn[s]); e1s = unit(sk.e1l[s] - sa * (sk.e1l[s] @ sa))
            c.ring(sk.an[s] - sa * .030 * k, sa, e1s, .037 * ml, .013 * m, 0., 1., 'cuff' + s, 'acuff' + s, 'fw', k=0, name='cuff')
        # collar plate under the neck
        c.ring(sk.neck + up * .004 * k, unit(up + fwd * .4), fwd, .066 * m, .010 * m, 0., 1., 'fcol', 'fcol', 'fw', k=0, name='fcol') if False else None

    def tail_(self, fr):
        """long tail.  fx['tail'] = (n, 3) world points from the tailbone, or None for the default swaying S"""
        sk, c, m, k = self.sk, self.c, self.m, self.k; P_ = self.fx.get('tail'); base = self.Pv(0, -.070, -.005)
        if P_ is None:
            back, up, rt = -sk.Rp[:, 1], sk.Rp[:, 2], sk.Rp[:, 0]; f = self.fx.get('f', 0); lag = np.asarray(self.fx.get('belt', np.zeros(3)), float); sw = math.sin(f * 2 * PI / 40.) * .5
            if fr:
                from .hero3 import tail_curve
                P_ = np.vstack([base, tail_curve(sk, f)])
            L = (1.30 if fr else .62) * self.H; n = 9 if not fr else 0; p = base.copy(); P_ = [p] if not fr else list(P_); d = unit(back * .9 - up * .45)
            for i in range(n):
                u = i / (n - 1.); turn = up * (-.35 + 1.15 * u if fr else .75 * u - .1) + rt * (sw * (.6 + u) * math.cos(u * 3 - f * .10)) * .55 + lag * (.35 + .6 * u) * .6
                d = unit(d + turn * .55 + back * .10); p = p + d * L / n; P_.append(p)
            P_ = np.array(P_)
        else: P_ = np.vstack([base, np.asarray(P_, float)])
        from .ink import spl
        t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P_, axis=0), axis=1))]; n = 30 if fr else 14; tt = np.linspace(0, t[-1], n + 1)
        Q = _cr3(P_, t, tt); r0, r1 = (.060 * m, .012 * m) if fr else (.021 * m, .018 * m)
        for i in range(n):
            u0, u1 = i / n, (i + 1.) / n; R0, R1 = r0 * (1 - u0) ** .75 + r1, r0 * (1 - u1) ** .75 + r1
            c.cap(Q[i], Q[i + 1], R0, R1, 'tail', 'tail', 'fw' if fr else 'tail', k=.0025 * k, name='tail', mg='tail')
        self.tailpts = Q

    # ------------------------------------------------------------------ feet
    def foot(self, s):
        sk, c, sp = self.sk, self.c, self.sp; k = self.k * self.b['foot'] * (self.b['mass'] ** .3) * 1.02; R = sk.Rfoot[s]; x, y, z = R[:, 0], R[:, 1], R[:, 2]; sg = -1 if s == 'L' else 1
        an = sk.an[s]; G = 'foot' + s; st = sp['shoe']; tb = sk.lift['t' + s] * D2R; lod = self.lod; ml = self.m * self.b['limb'] * self.b['legm']
        mat = dict(goku='boot', vegeta='bootw', frieza='fw', piccolo='shoeb', kid=sp.get('shoemat', 'boot'))[st]; sole = dict(goku='sole', vegeta='bootw', frieza='fw', piccolo='sole', kid='sole')[st]
        h = self.b['ankz'] * self.H; ax0 = -h + .036 * k
        heel = an + R @ np.array([0, -.048 * k, ax0]); ball = an + R @ np.array([sg * .004 * k, .112 * k, ax0 - .002 * k]); dn = .036 * k
        def ft(st_, flat=dn):
            T = table(st_, 2.35) * k; cp = np.cos(PH); lim = np.where(cp < -.05, flat / np.maximum(-cp, .05), 9.); return np.minimum(T, lim)
        hi = dict(frieza=.82, kid=1.).get(st, 1.06)
        rear = ft([(0, .054 * hi, .038, .05, .038), (.3, .062 * hi, .042, .05, .040), (.6, .050 * hi, .048, .05, .044), (1, .036 * hi, .054, .05, .048)])
        low = lambda t, p, q: ((q - an) @ z) < -h + (.016 if st != 'vegeta' else .010) * k
        rl = [(low, sole)] if sole != mat else []
        toecap = []
        if st == 'vegeta': toecap = [(lambda t, p, q: ((q - an) @ y) > .100 * k, 'gold')]
        self.prim['ft' + s] = c.loft(heel, ball, z, rear, G, G, mat, cap=(1.5, 1.2), k=.02, side=sg, name=G + 'r', rule=rl + toecap)
        td = unit(y * math.cos(tb) + z * math.sin(tb)); tz = unit(z * math.cos(tb) - y * math.sin(tb)); tip = ball + td * (.066 if st != 'piccolo' else .080) * k
        toe = ft([(0, .036 * hi, .054, .05, .048), (.5, .033, .050, .05, .045), (1, .024, .036, .05, .034)])
        if st != 'frieza': self.prim['tc' + s] = c.loft(ball, tip, tz, toe, G, G, mat if st != 'vegeta' else 'gold', cap=(1., 1.15), k=.02, side=sg, name=G + 't', rule=rl if st != 'vegeta' else [])
        else:
            for q, xx in enumerate((-.030, .0, .030)):
                o_ = ball + R @ np.array([xx * k, .0, -.004 * k]); d_ = unit(td + x * xx * 5 - z * .06)
                c.cap(o_, o_ + d_ * (.052 - .006 * abs(q - 1)) * k, .022 * k, .016 * k, G, G, mat, k=.010, name=G + 'c', mg='toe%d' % q)
                c.ball(o_ + d_ * .060 * k + z * .006 * k, np.array([.011, .014, .009]) * k, G, G, 'fv', ax=d_, e1=x, k=.0, name=G + 'n')
        up = unit(sk.kn[s] - an); LS = np.linalg.norm(sk.kn[s] - an); e1s = unit(sk.e1l[s] - up * (sk.e1l[s] @ up))
        top = dict(goku=.44, vegeta=.62, frieza=0, piccolo=.14, kid=.34)[st]
        if top > 0:
            rr = dict(goku=[(0, .052, .046, .054, .046), (.25, .046, .045, .050, .044), (.7, .050, .052, .058, .050), (.90, .054, .056, .062, .054), (1, .057, .059, .065, .057)], vegeta=[(0, .054, .048, .056, .048), (.2, .050, .048, .053, .047), (.55, .055, .057, .068, .054), (.90, .062, .064, .074, .060), (1, .068, .069, .078, .066)],
                      piccolo=[(0, .050, .044, .052, .044), (1, .046, .044, .050, .043)], kid=[(0, .052, .046, .054, .046), (1, .050, .050, .056, .048)])[st]
            T = table(rr, 2.2) * ml * (1 if st != 'kid' else .95)
            if lod and st in ('goku', 'vegeta'): T = T + facet(81 + (s == 'R'), .0018 * ml, 5, 6) + ridge(TT, PH, .20, .0, .03, .05, 1.7, .006 * ml) + ridge(TT, PH, .32, PI, -.04, .05, 1.5, .005 * ml)
            rs = [(lambda t, p, q: t > 1.0005, 'inside')]
            if st == 'goku': rs = rs + [(lambda t, p, q: (t > .875) & (t <= 1.0005), 'trim')]
            if sp.get('shattered') and lod: T = np.where(TT > 1 - .12 * tear(PH, 47 + (s == 'R'), 9), T * .93, T)
            self.prim['fs' + s] = j = c.loft(an - z * .030 * k - y * .006 * k, an + up * top * LS, e1s, T, G, G, mat, cap=(1.2, 0), k=.032, side=-sg, name=G + 's', rule=rs)
            if lod:
                if st == 'goku':
                    self.line(j, [.06, .4, .86], [.02, .02, 0.], .8, pri=1); self.line(j, [.10, .14, .10], [-1.1, 0, 1.1], .8, pri=0); self.line(j, [.22, .26, .22], [-1.0, .1, 1.1], .7, pri=1); self.blade(j, (.34, 1.2), (.72, .6), .06, pri=1); self.blade(j, (.16, 2.2), (.30, 1.2), .05, pri=0); self.blade(j, (.36, -1.4), (.74, -.8), .06, pri=1)
                if st == 'vegeta':
                    self.line(j, [.90, .885, .90], [-1.7, 0, 1.7], .8, pri=0); self.line(j, [.90, .885, .90], [1.5, PI, -1.5], .8, pri=1)
                    self.blade(j, (.12, 1.4), (.20, .1), .05, pri=0); self.blade(j, (.24, -1.3), (.30, -.1), .05, pri=0); self.blade(j, (.30, 2.6), (.42, 1.6), .05, pri=1); self.blade(j, (.40, .9), (.80, .6), .05, pri=1)
        else: c.loft(an - z * .030 * k, an + up * .035 * k, y, table([(0, .040, .036, .044, .036), (1, .034, .032, .038, .032)], 2.2) * k, G, G, mat, cap=(1, 1), k=.03, name=G + 'an')
        if lod:
            i = self.prim['ft' + s]
            if st == 'goku':
                for t0 in (.62, .80): self.line(i, [t0 + .06, t0, t0 + .06], [-.62, 0, .62], 1.0, pri=1)
                self.line(i, [.40, .75, 1.], [1.25, 1.22, 1.2], .8, pri=1); self.line(i, [.40, .75, 1.], [-1.25, -1.22, -1.2], .8, pri=1)
            if st == 'vegeta':
                j = self.prim['tc' + s]
                for t0 in (.25, .60): self.line(j, [t0 - .05, t0, t0 - .05], [-1.2, 0, 1.2], .9, pri=0)
                self.line(i, [.52, .48, .52], [-.9, 0, .9], .7, pri=1)


def _cr3(P, t, tt):
    """Catmull-Rom through 3-D points"""
    m = np.zeros_like(P); m[1:-1] = (P[2:] - P[:-2]) / (t[2:] - t[:-2])[:, None]; m[0] = (P[1] - P[0]) / (t[1] - t[0]); m[-1] = (P[-1] - P[-2]) / (t[-1] - t[-2])
    i = np.clip(np.searchsorted(t, tt, side='right') - 1, 0, len(t) - 2); h = (t[i + 1] - t[i])[:, None]; u = ((tt - t[i]) / (t[i + 1] - t[i]))[:, None]
    return (2 * u ** 3 - 3 * u ** 2 + 1) * P[i] + (u ** 3 - 2 * u ** 2 + u) * h * m[i] + (-2 * u ** 3 + 3 * u ** 2) * P[i + 1] + (u ** 3 - u ** 2) * h * m[i + 1]
