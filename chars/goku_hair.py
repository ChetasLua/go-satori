"""GOKU's hair: 3-D curved blades on the skull (same clumps from every angle), cut into planes by a ridge.

    hr = Hair(H, style='base', L=palette dict, f=0, wind=(0,0), wet=0, seed=0, lw=1)
    hr.back(cel)   hr.front(cel)      # behind the neck and face / in front of them
    hr.cast(cel, face_polygon, col)   # the hair's hard shadow on the forehead
styles: 'base', 's1'..'s5' (rising), 'ssj', 'ui', or mix(a, b, t).   painterly=True adds strands and cut-metal planes (close-ups).
"""
import numpy as np, math, cv2, cairo
from db.ink import spl, resample, ribbon, noise1
from db.kit import contours
from db.paint import rgb, f32, smooth
A = np.array
D2R = math.pi / 180


def _c(c): return rgb(c) if isinstance(c, str) else np.asarray(c, f32)
def mixc(a, b, t): return _c(a) * (1 - t) + _c(b) * t


def surf(H, th, y, off=0.):
    """skull shell point at azimuth th (radians; 0 front, + to +x) and height y, pushed out by off; and its normal"""
    th = np.asarray(th, float); y = np.asarray(y, float) * np.ones_like(th)
    a = np.interp(y, H.yy, H.A); f = np.interp(y, H.yy, H.F); b = np.interp(y, H.yy, H.B); p = H.k['p']
    sx, cz = np.sin(th), np.cos(th); r = (np.abs(sx) ** p + np.abs(cz) ** p) ** (-1 / p); d = np.where(cz >= 0, f, b)
    n = np.stack([sx / np.maximum(a, .18), (y - .10) * .95, cz / np.maximum(d, .18)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.stack([a * sx * r, y, d * cz * r], -1) + n * np.asarray(off, float)[..., None] if np.ndim(off) else np.stack([a * sx * r, y, d * cz * r], -1) + n * off, n


# --------------------------------------------------------------------------------------------- the clumps
# name: (th deg, y root, tip xyz, tip direction, root width, k) ; k = dict(out grow along the normal 0..1, k1, k2 handles, pw taper, belly, asym, thick, bias, grp)
def _b(th, y, tip, d, w, **k):
    q = dict(th=th, y=y, tip=A(tip, float), d=A(d, float), w=w, out=.55, k1=.36, k2=.36, pw=1.25, belly=.05, asym=0., thick=.55, bias=0., cut=.2, inset=.04, r0=.85)
    q.update(k); q['d'] = q['d'] / np.linalg.norm(q['d']); return q


# viewer's left is -x.  long front bangs sweep to the viewer's left from a parting right of the middle; big spikes left (one tall), slope and four shorter spikes right, back spikes
BASE = dict(
    # ---- bangs: from a parting right of the middle the long ones sweep to the viewer's left
    F1=_b(2, .88, (-.72, -.28, .80), (.04, -1.0, -.16), .86, out=.50, k1=.50, k2=.56, pw=2.0, belly=.03, asym=.9, thick=.32, bias=.52, cut=.30, r0=.10),
    F2=_b(8, .86, (-.36, -.01, .98), (.22, -1.0, -.14), .54, out=.55, k1=.48, k2=.46, pw=1.8, belly=.03, asym=.8, thick=.32, bias=.50, cut=.30, r0=.10),
    F3=_b(14, .85, (-.06, -.05, 1.03), (.32, -1.0, -.12), .33, out=.58, k1=.46, k2=.44, pw=1.7, belly=.04, asym=.7, thick=.32, bias=.56, cut=.30, r0=.10),
    F4=_b(20, .86, (.36, .05, .95), (-.10, -1.0, -.14), .50, out=.55, k1=.48, k2=.48, pw=1.8, belly=.04, asym=-.7, thick=.32, bias=.50, cut=.30, r0=.10),
    F5=_b(26, .88, (.73, -.14, .70), (-.04, -1.0, -.20), .68, out=.50, k1=.50, k2=.54, pw=1.9, belly=.03, asym=-.9, thick=.32, bias=.46, cut=.30, r0=.10),
    # ---- crown volume that feeds the bangs
    C1=_b(-6, .99, (-1.00, .22, .56), (-.10, -1.0, .00), 1.06, out=.40, k1=.46, k2=.50, pw=1.9, belly=.04, asym=.8, thick=.36, bias=.34, cut=.36, r0=.12),
    C2=_b(34, .99, (1.02, .18, .40), (.12, -1.0, -.10), .98, out=.40, k1=.46, k2=.50, pw=1.9, belly=.04, asym=-.8, thick=.36, bias=.32, cut=.36, r0=.12),
    W1=_b(6, .92, (-.52, 1.50, .52), (-.66, .75, -.05), .62, out=.95, k1=.44, k2=.40, pw=1.25, belly=.05, asym=.5, thick=.5, bias=.30, r0=.35),
    W2=_b(18, .92, (.58, 1.40, .46), (.72, .66, -.10), .56, out=.95, k1=.44, k2=.40, pw=1.25, belly=.05, asym=-.5, thick=.5, bias=.28, r0=.35),
    # ---- viewer's left, big (shark fins: convex above, hollow below)
    L1=_b(-10, .98, (-.66, 2.26, .04), (-.80, .80, -.06), 1.34, out=.98, k1=.46, k2=.42, pw=1.30, belly=.05, asym=.5, thick=.60, bias=.04),
    L2=_b(-44, .84, (-1.76, 1.50, -.10), (-1.0, .02, -.20), 1.30, out=.90, k1=.46, k2=.42, pw=1.40, belly=.06, asym=.6, thick=.58, r0=.42, bias=.10),
    L3=_b(-66, .60, (-2.02, .64, -.18), (-1.0, -.28, -.22), 1.10, out=.88, k1=.44, k2=.42, pw=1.40, belly=.06, asym=.6, thick=.58, r0=.42, bias=.14),
    L4=_b(-84, .34, (-1.64, .06, -.28), (-.90, -.60, -.26), .80, out=.84, k1=.44, k2=.40, pw=1.35, belly=.06, asym=.6, thick=.58, r0=.42, bias=.10),
    L5=_b(-102, .08, (-1.14, -.28, -.40), (-.60, -.86, -.26), .52, out=.76, k1=.42, k2=.40, pw=1.35, belly=.06, asym=.5, thick=.58, r0=.42, bias=.00),
    # ---- top right slope and viewer's right
    T2=_b(20, .99, (.72, 1.84, -.20), (.86, .52, -.24), 1.26, out=.95, k1=.42, k2=.40, pw=1.30, belly=.06, asym=-.5, thick=.60, bias=-.06),
    R1=_b(46, .86, (1.66, 1.36, -.16), (1.0, -.04, -.26), 1.24, out=.90, k1=.44, k2=.42, pw=1.40, belly=.06, asym=-.6, thick=.58, r0=.42, bias=.08),
    R2=_b(67, .62, (1.96, .62, -.22), (1.0, -.34, -.26), 1.04, out=.88, k1=.44, k2=.42, pw=1.40, belly=.06, asym=-.6, thick=.58, r0=.42, bias=.12),
    R3=_b(84, .36, (1.60, .06, -.30), (.88, -.62, -.28), .76, out=.84, k1=.44, k2=.40, pw=1.35, belly=.06, asym=-.6, thick=.58, r0=.42, bias=.09),
    R4=_b(102, .08, (1.12, -.28, -.42), (.60, -.86, -.26), .50, out=.76, k1=.42, k2=.38, pw=1.35, belly=.06, asym=-.5, thick=.58, r0=.42, bias=.0),
    K1=_b(-100, .93, (-.46, 1.66, -.50), (-.30, 1.0, -.50), 1.06, out=.85, k1=.42, k2=.42, pw=1.25, belly=.06, thick=.7, bias=-.24, r0=.7),
    K2=_b(96, .93, (.50, 1.60, -.54), (.34, 1.0, -.55), 1.04, out=.85, k1=.42, k2=.42, pw=1.25, belly=.06, thick=.7, bias=-.26, r0=.7),
    # ---- side cover (lies along the skull, seen in three-quarter and profile)
    SA=_b(-74, .46, (-1.08, .70, -.62), (-.45, .20, -1.0), .74, out=.55, k1=.40, k2=.40, pw=1.4, belly=.07, asym=.5, thick=.45, r0=.42, bias=.02, cut=.12),
    SB=_b(-84, .14, (-1.00, .02, -.80), (-.40, -.22, -1.0), .58, out=.50, k1=.40, k2=.40, pw=1.4, belly=.07, asym=.5, thick=.45, r0=.42, bias=.0, cut=.10),
    SC=_b(-112, .52, (-1.16, .74, -1.12), (-.50, .16, -1.0), .78, out=.60, k1=.40, k2=.40, pw=1.3, belly=.07, thick=.5, r0=.42, bias=-.08, cut=.15),
    SD=_b(74, .46, (1.06, .70, -.64), (.45, .20, -1.0), .74, out=.55, k1=.40, k2=.40, pw=1.4, belly=.07, asym=-.5, thick=.45, r0=.42, bias=.02, cut=.12),
    SE=_b(84, .14, (.98, .02, -.82), (.40, -.22, -1.0), .56, out=.50, k1=.40, k2=.40, pw=1.4, belly=.07, asym=-.5, thick=.45, r0=.42, bias=.0, cut=.10),
    SF=_b(112, .52, (1.14, .72, -1.14), (.50, .16, -1.0), .78, out=.60, k1=.40, k2=.40, pw=1.3, belly=.07, thick=.5, r0=.42, bias=-.08, cut=.15),
    # ---- back
    B1=_b(180, .96, (-.10, 1.94, -.98), (-.06, .72, -.80), 1.06, out=.88, k1=.42, k2=.42, pw=1.12, belly=.08, thick=.66, bias=-.3),
    B2=_b(170, .74, (.20, 1.28, -1.84), (.08, .14, -1.0), 1.00, out=.86, k1=.42, k2=.42, pw=1.12, belly=.08, thick=.66, r0=.42, bias=-.3),
    B3=_b(190, .42, (-.18, .48, -1.96), (-.06, -.22, -1.0), .90, out=.86, k1=.42, k2=.42, pw=1.12, belly=.08, thick=.66, r0=.42, bias=-.3),
    B4=_b(150, .18, (.66, -.04, -1.50), (.30, -.50, -.9), .66, out=.80, k1=.40, k2=.40, pw=1.15, belly=.08, thick=.66, r0=.42, bias=-.2),
    B5=_b(-150, .18, (-.66, .0, -1.52), (-.30, -.48, -.9), .66, out=.80, k1=.40, k2=.40, pw=1.15, belly=.08, thick=.66, r0=.42, bias=-.2),
    B6=_b(-128, .74, (-1.22, 1.34, -1.26), (-.6, .24, -.8), .92, out=.82, k1=.42, k2=.42, pw=1.12, belly=.08, thick=.66, r0=.42, bias=-.2),
    B7=_b(128, .74, (1.16, 1.36, -1.20), (.6, .24, -.8), .90, out=.82, k1=.42, k2=.42, pw=1.12, belly=.08, thick=.66, r0=.42, bias=-.2),
    N1=_b(164, -.18, (.26, -.72, -1.04), (.10, -.9, -.36), .50, out=.55, k1=.40, k2=.40, pw=1.3, belly=.06, thick=.6, r0=.42, bias=-.2),
    N2=_b(-164, -.18, (-.28, -.70, -1.06), (-.12, -.9, -.36), .50, out=.55, k1=.40, k2=.40, pw=1.3, belly=.06, thick=.6, r0=.42, bias=-.2),
    N3=_b(130, -.08, (.80, -.52, -.80), (.36, -.85, -.30), .44, out=.55, k1=.40, k2=.40, pw=1.3, belly=.06, thick=.6, r0=.42, bias=-.1),
    N4=_b(-130, -.08, (-.82, -.50, -.82), (-.36, -.85, -.30), .44, out=.55, k1=.40, k2=.40, pw=1.3, belly=.06, thick=.6, r0=.42, bias=-.1),
)
ORDER = list(BASE)

# Super Saiyan: the same clumps stand up like a flame.  one bang stays and curls on the viewer's left
SSJ = dict(
    F1=_b(-30, .54, (-1.00, 1.30, .58), (-.30, 1.0, .02), .64, out=1.0, k1=.46, k2=.50, pw=1.15, belly=.10, asym=.7, thick=.5, r0=.50, bias=.46),
    F2=_b(-2, .70, (-.40, .17, 1.00), (-.12, -1.0, -.16), .50, out=.72, k1=.46, k2=.48, pw=1.7, belly=.05, asym=.8, thick=.36, bias=.60, cut=.26, r0=.14),
    F3=_b(-9, .56, (-.24, 1.78, .56), (-.06, 1.0, -.10), .66, out=.95, k1=.42, k2=.46, pw=1.12, belly=.10, asym=.5, thick=.5, r0=.50, bias=.46),
    F4=_b(14, .56, (.48, 1.58, .56), (.18, 1.0, -.05), .62, out=.95, k1=.42, k2=.46, pw=1.12, belly=.10, asym=-.5, thick=.5, r0=.50, bias=.48),
    F5=_b(38, .52, (1.12, 1.14, .46), (.38, 1.0, .0), .60, out=1.0, k1=.46, k2=.50, pw=1.15, belly=.10, asym=-.7, thick=.5, r0=.50, bias=.42),
    C1=_b(-26, .90, (-.86, 2.16, .26), (-.16, 1.0, -.05), .90, out=.95, k1=.42, k2=.50, pw=1.1, belly=.10, asym=.6, thick=.55, bias=.24),
    C2=_b(28, .90, (.84, 2.10, .22), (.18, 1.0, -.05), .86, out=.95, k1=.42, k2=.50, pw=1.1, belly=.10, asym=-.6, thick=.55, bias=.22),
    W1=_b(-4, .92, (-.52, 2.06, .40), (-.14, 1.0, -.05), .66, out=.95, k1=.44, k2=.46, pw=1.12, belly=.08, asym=.5, thick=.5, bias=.30, r0=.5),
    W2=_b(22, .92, (.62, 1.92, .38), (.16, 1.0, -.05), .60, out=.95, k1=.44, k2=.46, pw=1.12, belly=.08, asym=-.5, thick=.5, bias=.28, r0=.5),
    L1=_b(-8, .98, (-.14, 2.74, -.02), (-.04, 1.0, -.10), 1.06, out=.9, k1=.40, k2=.46, pw=1.06, belly=.10, asym=.5, thick=.62, bias=.04),
    L2=_b(-50, .82, (-1.50, 2.08, -.10), (-.34, 1.0, -.10), .94, out=1.0, k1=.46, k2=.54, pw=1.08, belly=.10, asym=.7, thick=.60, r0=.45, bias=.08),
    L3=_b(-70, .58, (-1.70, 1.42, -.16), (-.36, 1.0, -.14), .80, out=1.0, k1=.48, k2=.54, pw=1.08, belly=.10, asym=.7, thick=.60, r0=.45, bias=.12),
    L4=_b(-86, .32, (-1.50, .80, -.28), (-.44, .95, -.16), .62, out=1.0, k1=.48, k2=.52, pw=1.1, belly=.10, asym=.7, thick=.60, r0=.45, bias=.10),
    L5=_b(-104, .06, (-1.18, .22, -.40), (-.56, .80, -.20), .44, out=.95, k1=.46, k2=.48, pw=1.15, belly=.08, asym=.6, thick=.6, r0=.45, bias=.0),
    T2=_b(20, .99, (.36, 2.56, -.24), (.10, 1.0, -.20), .98, out=.9, k1=.40, k2=.46, pw=1.06, belly=.10, asym=-.5, thick=.60, bias=-.16),
    R1=_b(50, .84, (1.36, 2.18, -.14), (.30, 1.0, -.12), .94, out=1.0, k1=.46, k2=.54, pw=1.08, belly=.10, asym=-.7, thick=.60, r0=.45, bias=.06),
    R2=_b(70, .58, (1.66, 1.56, -.20), (.34, 1.0, -.16), .78, out=1.0, k1=.48, k2=.54, pw=1.08, belly=.10, asym=-.7, thick=.60, r0=.45, bias=.10),
    R3=_b(86, .32, (1.50, .92, -.30), (.40, .95, -.18), .60, out=1.0, k1=.48, k2=.52, pw=1.1, belly=.10, asym=-.7, thick=.60, r0=.45, bias=.09),
    R4=_b(104, .06, (1.18, .34, -.42), (.50, .85, -.20), .44, out=.95, k1=.46, k2=.48, pw=1.15, belly=.08, asym=-.6, thick=.6, r0=.45, bias=.0),
    K1=_b(-100, .93, (-.48, 2.28, -.44), (-.14, 1.0, -.25), 1.06, out=.9, k1=.42, k2=.46, pw=1.1, belly=.08, thick=.7, bias=-.24, r0=.7),
    K2=_b(96, .93, (.52, 2.24, -.46), (.16, 1.0, -.28), 1.04, out=.9, k1=.42, k2=.46, pw=1.1, belly=.08, thick=.7, bias=-.26, r0=.7),
    SA=_b(-72, .34, (-1.14, 1.02, -.42), (-.25, 1.0, -.30), .66, out=.75, k1=.42, k2=.46, pw=1.25, belly=.08, asym=.5, thick=.45, r0=.45, bias=.04, cut=.10),
    SB=_b(-84, .06, (-1.06, .50, -.60), (-.25, .9, -.50), .54, out=.70, k1=.42, k2=.46, pw=1.25, belly=.08, asym=.5, thick=.45, r0=.45, bias=.02, cut=.10),
    SC=_b(-112, .50, (-1.12, 1.48, -.86), (-.25, 1.0, -.45), .78, out=.80, k1=.42, k2=.46, pw=1.2, belly=.08, thick=.5, r0=.45, bias=-.06, cut=.12),
    SD=_b(72, .34, (1.12, 1.04, -.44), (.25, 1.0, -.30), .66, out=.75, k1=.42, k2=.46, pw=1.25, belly=.08, asym=-.5, thick=.45, r0=.45, bias=.04, cut=.10),
    SE=_b(84, .06, (1.04, .52, -.62), (.25, .9, -.50), .54, out=.70, k1=.42, k2=.46, pw=1.25, belly=.08, asym=-.5, thick=.45, r0=.45, bias=.02, cut=.10),
    SF=_b(112, .50, (1.10, 1.52, -.88), (.25, 1.0, -.45), .78, out=.80, k1=.42, k2=.46, pw=1.2, belly=.08, thick=.5, r0=.45, bias=-.06, cut=.12),
    B1=_b(180, .96, (-.02, 2.56, -.66), (-.02, 1.0, -.25), 1.06, out=.9, k1=.42, k2=.46, pw=1.06, belly=.10, thick=.66, bias=-.3),
    B2=_b(170, .74, (.26, 2.10, -1.28), (.06, 1.0, -.36), 1.00, out=1.0, k1=.46, k2=.50, pw=1.06, belly=.10, thick=.66, r0=.45, bias=-.3),
    B3=_b(190, .42, (-.24, 1.42, -1.62), (-.06, .9, -.46), .90, out=1.0, k1=.46, k2=.52, pw=1.06, belly=.10, thick=.66, r0=.45, bias=-.3),
    B4=_b(150, .16, (.70, .80, -1.46), (.22, .8, -.6), .66, out=.95, k1=.46, k2=.50, pw=1.1, belly=.10, thick=.66, r0=.45, bias=-.2),
    B5=_b(-150, .16, (-.70, .84, -1.46), (-.22, .8, -.6), .66, out=.95, k1=.46, k2=.50, pw=1.1, belly=.10, thick=.66, r0=.45, bias=-.2),
    B6=_b(-128, .74, (-1.06, 2.16, -.90), (-.26, 1.0, -.30), .92, out=1.0, k1=.44, k2=.50, pw=1.08, belly=.10, thick=.66, r0=.45, bias=-.2),
    B7=_b(128, .74, (1.02, 2.22, -.88), (.26, 1.0, -.30), .90, out=1.0, k1=.44, k2=.50, pw=1.08, belly=.10, thick=.66, r0=.45, bias=-.2),
    N1=_b(164, -.18, (.34, -.30, -1.24), (.2, .3, -.9), .50, out=.75, k1=.42, k2=.42, pw=1.2, belly=.07, thick=.6, r0=.45, bias=-.2),
    N2=_b(-164, -.18, (-.36, -.28, -1.26), (-.2, .3, -.9), .50, out=.75, k1=.42, k2=.42, pw=1.2, belly=.07, thick=.6, r0=.45, bias=-.2),
    N3=_b(130, -.08, (.96, -.02, -.96), (.5, .5, -.6), .44, out=.75, k1=.42, k2=.42, pw=1.2, belly=.07, thick=.6, r0=.45, bias=-.1),
    N4=_b(-130, -.08, (-.98, .0, -.98), (-.5, .5, -.6), .44, out=.75, k1=.42, k2=.42, pw=1.2, belly=.07, thick=.6, r0=.45, bias=-.1),
)


def _mixb(a, b, t):
    o = {}
    for k in a:
        o[k] = a[k] * (1 - t) + b[k] * t
    o['d'] = o['d'] / (np.linalg.norm(o['d']) + 1e-9)
    if 0 < t < 1:                                   # the tip swings on an arc round the root: it never shortens on its way up
        r = A([math.sin(o['th'] * D2R) * .7, o['y'], math.cos(o['th'] * D2R) * .75])
        la, lb = np.linalg.norm(a['tip'] - r), np.linalg.norm(b['tip'] - r); v = o['tip'] - r; o['tip'] = r + v / (np.linalg.norm(v) + 1e-9) * (la * (1 - t) + lb * t) * (1 - .06 * math.sin(math.pi * t))
    return o


def mix(a, b, t, stagger=0.):
    """style between two styles; stagger: the crown rises first, the sides drag"""
    a, b = get(a), get(b); o = {}
    for i, k in enumerate(ORDER):
        tt = t
        if stagger: tt = float(np.clip(t + stagger * (.5 - (i * 7 % len(ORDER)) / len(ORDER)), 0, 1.15))
        o[k] = _mixb(a[k], b[k], tt)
    return o


def _ui():
    o = {}
    for k in ORDER:
        q = dict(BASE[k])
        for kk in ('tip', 'd'): q[kk] = q[kk].copy()
        if k[0] in 'LRTB': q['tip'] = q['tip'] * A([1.0, 1.0, 1.0]) + A([0, .16 if k != 'L4' and k != 'R4' else .08, 0]); q['d'] = q['d'] + A([0, .22, 0]); q['w'] *= 1.06; q['belly'] *= 1.2
        o[k] = q
    return o


UI = _ui()
_CACHE = {}


def get(s):
    if isinstance(s, dict): return s
    if s in _CACHE: return _CACHE[s]
    if s == 'base': r = BASE
    elif s == 'ssj': r = SSJ
    elif s == 'ui': r = UI
    elif s[0] == 's' and s[1:].isdigit():
        k = int(s[1:]); r = mix(BASE, SSJ, (0, .14, .34, .58, .80, 1.)[min(k, 5)], stagger=.16 if k < 5 else 0.)
    else: raise KeyError(s)
    _CACHE[s] = r; return r


# --------------------------------------------------------------------------------------------- projected blades
class Blade:
    __slots__ = ('n', 'k', 'q', 'C3', 'C', 'A', 'B', 'R', 'N', 'hw', 'dp', 'L', 'tone', 'P', 'zt', 't', 'front', 'sgn', 'nL', 'nR')


def _bez(p0, p1, p2, p3, n=34):
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u * u * p2 + u ** 3 * p3


def _runs(m):
    out = []; i = 0; n = len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]: j += 1
            out.append((i, j)); i = j + 1
        else: i += 1
    return out


class Hair:
    def __init__(s, H, style='base', L=None, f=0, wind=(0., 0.), wet=0., seed=0, lw=1., painterly=False, part_z=-.12, extra=(), drop=(), lift=0., detail=1., kids=0, flat=False):
        s.H, s.L, s.seed, s.lw, s.wet, s.painterly, s.detail = H, L, int(seed), lw, wet, painterly, detail
        s.kids = kids; s.flat = flat; s.u = H.s; st = get(style); s.pal = L['hair']; s.bl = []
        wx, wy = (wind, 0.) if np.ndim(wind) == 0 else wind
        r = np.random.default_rng(91)
        ph = {k: r.uniform(0, 6.28) for k in ORDER}; st_ = {k: r.uniform(.8, 1.2) for k in ORDER}
        Rinv = H.R.T; g = Rinv @ A([0, -1., 0])
        for i, k in enumerate(list(st) + [None] * 0):
            if k in drop: continue
            s._add(k, st[k], f, wx, wy, wet, ph.get(k, 0.), st_.get(k, 1.), Rinv, g, lift)
        for j, q in enumerate(extra): s._add('X%d' % j, q, f, wx, wy, wet * .3, j * 1.7, 1., Rinv, g, 0.)
        s.bl.sort(key=lambda b: b.dp)
        s.split = part_z
        s._caps()

    def _add(s, k, q, f, wx, wy, wet, ph, stf, Rinv, g, lift):
        H = s.H
        root, n = surf(H, q['th'] * D2R, q['y'], -q['inset'])
        tip = q['tip'].copy(); d = q['d'].copy()
        if wet > 0 and k[0] != 'X':                              # water weight: the clump droops toward screen-down and loses volume
            v = tip - root; Lv = np.linalg.norm(v); gv = g * (.9 if k[0] in 'LRTB' else .55)
            w_ = wet * (.40 if k[0] in 'LRTB' else .22); v2 = v / Lv * (1 - w_) + gv * w_; tip = root + v2 / np.linalg.norm(v2) * Lv * (1 - .10 * wet); d = d * (1 - w_ * 1.2) + g * w_ * 1.2
        if k == 'L1' and lift: tip = tip + A([0, lift, 0])
        L = np.linalg.norm(tip - root) + 1e-6
        if wx or wy:
            sw = math.sin(f * .42 * stf + ph) * .5 + math.sin(f * .23 + ph * 1.7) * .5
            dv = Rinv @ A([wx * (.55 + .45 * sw), wy * (.55 + .45 * sw) + .12 * abs(wx) * math.sin(f * .33 + ph), 0.])
            kk = L * .30 * (1.15 if k[0] == 'F' else 1.); v = tip + dv * kk - root; tip = root + v / np.linalg.norm(v) * L; d = d + dv * .55
        d = d / (np.linalg.norm(d) + 1e-9); u = (tip - root) / L
        gdir = n * q['out'] + u * (1 - q['out']); gdir /= np.linalg.norm(gdir)
        C3 = _bez(root, root + gdir * L * q['k1'], tip - d * L * q['k2'], tip)
        s._blade(k, q, C3, L, wet)

    def _blade(s, k, q, C3, L, wet):
        H = s.H; b = Blade(); b.k, b.q, b.C3, b.L = k, q, C3, L
        t = np.linspace(0, 1, len(C3)); b.t = t
        C = H.P3(C3); T3 = np.gradient(C3, axis=0); T3 /= np.linalg.norm(T3, axis=1)[:, None] + 1e-9
        n0 = C3[6] / (np.linalg.norm(C3[6]) + 1e-9)
        nn = C3 - A([0, .15, -.05]); nn = nn - T3 * np.sum(nn * T3, 1)[:, None]; ln_ = np.linalg.norm(nn, axis=1)
        nn = nn / np.maximum(ln_, 1e-3)[:, None]
        # transport a frame from the root so that radial spikes keep a steady flat axis
        e = np.cross(T3[0], n0); e = e / (np.linalg.norm(e) + 1e-9) if np.linalg.norm(e) > .12 else np.cross(T3[0], A([0, 0, 1.]) if abs(T3[0][2]) < .8 else A([0, 1., 0]))
        E = np.empty_like(C3)
        for i in range(len(C3)): e = e - T3[i] * (e @ T3[i]); e = e / (np.linalg.norm(e) + 1e-9); E[i] = e
        M = np.cross(E, T3)                                        # thickness axis
        M = M * np.where(np.sum(M * nn, 1).mean() < 0, -1, 1)
        T2 = np.gradient(C, axis=0); T2 /= np.hypot(T2[:, 0], T2[:, 1])[:, None] + 1e-9; N = np.c_[-T2[:, 1], T2[:, 0]]
        # smooth normals near the tip
        Ev, Mv = E @ H.R.T, M @ H.R.T                              # view space x right y up z to viewer
        Es, Ms = np.c_[Ev[:, 0], -Ev[:, 1]], np.c_[Mv[:, 0], -Mv[:, 1]]
        a_, b_ = np.sum(Es * N, 1), np.sum(Ms * N, 1)
        pw = q['pw']
        base = (1 - t ** pw) * .5 if pw >= 1 else (1 - t) ** (1 / pw) * .5
        r0_ = q.get('r0', .85); base = base * (np.interp(t, [0, .10, .26, 1], [r0_, .5 + .5 * r0_, 1, 1]) if r0_ > .05 else np.sin(np.minimum(t / .55, 1.) * np.pi / 2) ** 1.2); bel = np.sin(np.pi * t ** .8) * q['belly']
        wid = q['w'] * (base + bel) * (1 - .20 * wet) * H.s
        fs = min(1., float(np.hypot(*np.diff(C, axis=0).T).sum()) / (float(np.linalg.norm(np.diff(C3, axis=0), axis=1).sum()) * H.s + 1e-6)); hw = wid * np.sqrt(a_ ** 2 + (b_ * q['thick']) ** 2 + .02) * (.46 + .54 * fs ** .8)
        ker = np.hanning(9); ker /= ker.sum(); hw = np.convolve(np.r_[[hw[0]] * 4, hw, [hw[-1]] * 4], ker, 'valid'); hw[-1] = 0; hw[-2] *= .72
        sk = q['asym'] * .16 * np.sign(a_.mean() if abs(a_.mean()) > .1 else 1)
        off = np.sin(np.pi * t ** .9) * sk * hw * 1.4
        b.C = C + N * off[:, None]; b.A = b.C + N * hw[:, None]; b.B = b.C - N * hw[:, None]; b.N, b.hw = N, hw
        # ridge: the thick axis seen from here, kept inside the blade
        if np.abs(a_).mean() * 1. >= np.abs(b_).mean() * q['thick']: rv = b_ * q['thick'] * np.sign(Mv[:, 2].mean()) ; zr = np.abs(Mv[:, 2])
        else: rv = a_ * np.sign(Ev[:, 2].mean()); zr = np.abs(Ev[:, 2])
        rr = np.clip(rv, -.42, .42); rr = np.convolve(np.r_[[rr[0]] * 4, rr, [rr[-1]] * 4], ker, 'valid')
        rr = rr + .10 * math.sin(hash_(k) * 6.28); b.R = b.C + N * (hw * np.clip(rr, -.48, .48))[:, None]
        # facet normals in view space (y up)
        Nv = np.c_[N[:, 0], -N[:, 1], np.zeros(len(N))]; Tv = T3 @ H.R.T; Z = A([0, 0, 1.])
        up = Z - Tv * (Tv[:, 2:3]); up /= np.linalg.norm(up, axis=1)[:, None] + 1e-6
        b.nL = Nv * .78 + up * .62; b.nR = -Nv * .78 + up * .62
        b.dp = float((C3[int(len(C3) * .30):] @ H.R.T)[:, 2].mean()) * .75 + float((C3[:8] @ H.R.T)[:, 2].mean()) * .25 + q['bias']
        b.P = np.vstack([b.A, b.B[::-1]]); s.bl.append(b)
        if s.kids and len(k) == 2 and k[0] in 'LRTFCWBS' and q['w'] > .48:      # small tips that break the edge of the big clumps
            for j, sd in enumerate(((1, -1) if hash_(k + 'a') < .5 else (-1, 1))[:s.kids]):
                h_ = hash_(k + str(j)); i0, i1 = int(5 + 4 * h_), int(22 + 6 * hash_(k + 'z' + str(j))); m = i1 - i0; tt = np.linspace(0, 1, m); D3 = C3[i0:i1] + E[i0:i1] * (sd * q['w'] * (.06 + (.40 + .22 * h_) * tt ** 1.4))[:, None]
                if b.hw.max() > .06 * H.s: s._blade(k + 'k%d' % j, dict(q, w=q['w'] * (.30 + .10 * h_), pw=1.3, r0=.0, belly=.06, asym=0., bias=q['bias'] - .015, cut=.45), np.stack([np.interp(np.linspace(0, m - 1, 34), np.arange(m), D3[:, c]) for c in range(3)], -1), L * .4, wet)

    # ------------------------------------------------------------------ skull cap
    HL = A([(-180, -.46), (-150, -.46), (-122, -.30), (-106, .06), (-92, .14), (-80, .02), (-72, -.10), (-66, .10), (-58, .34), (-48, .46), (-36, .56), (-26, .54), (-12, .50), (4, .50), (14, .50), (26, .54), (36, .56), (48, .46), (58, .34), (62, .22), (67, .10), (73, -.10), (81, .02), (93, .14), (106, .06), (122, -.30), (150, -.46), (180, -.46)], float)

    def _caps(s):
        H = s.H; th = np.linspace(-180, 180, 121); v = np.linspace(0, 1, 26)[:, None]; y0 = np.interp(th, s.HL[:, 0], s.HL[:, 1])[None]
        Y = y0 + (.995 - y0) * v ** .85; TH = np.repeat(th[None] * D2R, 26, 0); off = .02 + .20 * np.clip(v * 1.4, 0, 1) ** .8
        P, n = surf(H, TH, Y, np.repeat(off, 121, 1)); k_ = np.clip((v - .55) / .45, 0, 1)[..., None] ** 1.2; n = n * (1 - k_) + A([0, 1., 0]) * k_; n /= np.linalg.norm(n, axis=-1, keepdims=True); P0, _ = surf(H, TH, Y, 0.); P = P0 + n * np.repeat(off, 121, 1)[..., None]
        pole = A([0, 1.0 + float(np.ravel(off[-1])[0]), P[-1, :, 2].mean()]); P = np.concatenate([P, P[-1:] * .5 + pole * .5, np.repeat(pole[None, None], 121, 1)]); n = np.concatenate([n, np.zeros((2, 121, 3)) + A([0, 1., 0])]); Q = H.P3(P); nv = (n @ H.R.T)[..., 2]      # closed over the crown (views from above)
        lo = Q.reshape(-1, 2).min(0) - 6; hi = Q.reshape(-1, 2).max(0) + 6; k = min(1., 900. / max(hi - lo)); w, h = np.ceil((hi - lo) * k).astype(int) + 2
        def poly(sel):
            m = np.zeros((h, w), np.uint8); q = np.round((Q - lo) * k * 16).astype(np.int32)
            for i, j in zip(*np.nonzero(sel)): cv2.fillConvexPoly(m, A([q[i, j], q[i, j + 1], q[i + 1, j + 1], q[i + 1, j]]), 255, cv2.LINE_8, 4)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)); cs = contours(m / 255., min_len=20)
            return [c / k + lo for c in cs if cv2.contourArea(np.float32(c)) > 60]
        nf = (nv[:-1, :-1] + nv[1:, 1:]) / 2
        s.cap_front = poly(nf > .0); s.cap_back = poly(np.ones_like(nf, bool))

    # ------------------------------------------------------------------ paint
    def tone(s, b, side):
        nl = A(s.L.get('hdir', s.L.get('ldir', (.5, .7, .5))), float); nl /= np.linalg.norm(nl)
        return (b.nL if side > 0 else b.nR) @ nl

    def fill_blade(s, cel, b):
        pal = s.pal; t = b.t; n = len(t); r = np.random.default_rng(s.seed * 7 + int(hash_(b.k) * 9999))
        g0, g1 = b.C[0], b.C[int(n * .62)]
        cel.grad(b.P, g0, g1, [(0, pal['root'], 1), (.55, pal['base'], 1), (1, pal['base'], 1)]) if np.hypot(*(g1 - g0)) > 4 else cel.fill(b.P, pal['base'])
        if s.u < 26: return
        big = s.u > 60
        cel.clip(b.P)
        tl, tr = s.tone(b, 1), s.tone(b, -1); ma, mb = float(tl[8:26].mean()), float(tr[8:26].mean())
        lit, lo_ = (b.A, b.B) if ma >= mb else (b.B, b.A)
        a0 = max(0., q_(b, 'cut') * .5)
        # shadow facet: from the ridge to the dark edge, with an angular root end
        w = np.interp(t, [0, .10, .30, .62, .90, 1], [1.0, 1.0, .92, .86, .55, 0.]) * (1.0 if min(ma, mb) < .30 else .55)
        E = lo_ + (b.R - lo_) * w[:, None]
        cel.fill(np.vstack([lo_, E[::-1]]), pal['shade'])
        # lit facet: second plane nearer the edge
        if max(ma, mb) > .18 and big:
            w2 = np.interp(t, [0, a0 + .10, a0 + .22, .62, .86, 1], [0, 0, .62, .58, .34, 0.]); E2 = lit + (b.R - lit) * w2[:, None]; i0 = int(n * (a0 + .08)); cel.fill(np.vstack([lit[i0:], E2[i0:][::-1]]), pal['light'])
        # root occlusion: long pointed tongues of shade that follow the clump
        if big and b.k[0] != 'X':
            k0 = q_(b, 'cut') * .55; pts = [b.A[0]]; m = 3 if b.hw.max() > .10 * s.u else 2; us = np.linspace(1, -1, m * 2 + 1)
            for j, u in enumerate(us):
                tt = k0 + ((.16 + .20 * r.random()) if j % 2 else .01 * r.random()); f_ = tt * (n - 1); pts.append(np.array([np.interp(f_, np.arange(n), b.C[:, c]) + np.interp(f_, np.arange(n), b.N[:, c]) * np.interp(f_, np.arange(n), b.hw) * u for c in (0, 1)]))
            pts.append(b.B[0]); cel.fill(A(pts), pal['deep'] if pal.get('dark') else pal['shade'], .95)
        # highlight: angular cut shards on the lit side of the big clumps
        if big and max(ma, mb) > pal.get('hl_thr', .30) and (not pal.get('dark') or b.k in ('L1', 'L2', 'L3', 'T2', 'R1', 'R2', 'F1', 'F5', 'C1', 'C2', 'SA', 'SD')):
            e_ = lambda tt, u: np.stack([np.interp(tt * (n - 1), np.arange(n), b.R[:, c] + (lit - b.R)[:, c] * u) for c in (0, 1)], -1)
            t0 = .26 + .12 * r.random(); t1 = t0 + .20 + .10 * r.random(); tm = (t0 + t1) / 2; k_ = .55 if pal.get('dark') else 1.
            cel.fill(A([e_(t0, .14), e_(t0 + .07, .14 + .30 * k_), e_(tm, .14 + .34 * k_), e_(t1, .14 + .10 * k_), e_(t1 + .03, .12), e_(t1 - .08, .06), e_(tm, .09)]), pal['hi'])
            if b.hw.max() > .22 * s.u:
                t2 = t1 + .04; cel.fill(A([e_(t2, .16), e_(t2 + .05, .18 + .18 * k_), e_(t2 + .13, .12), e_(t2 + .07, .10)]), pal['hi'])
                if not pal.get('dark'): t3 = t0 + .04; cel.fill(A([e_(t3, .60), e_(t3 + .08, .76), e_(t3 + .20, .70), e_(t3 + .12, .58)]), mixc(pal['hi'], pal['light'], .4))
        cel.unclip()

    def paint_blade(s, cel, b):
        """close-up finish: long cut needles of tone and strands that follow the blade (silver like cut metal, gold like flame)"""
        pal = s.pal; t = b.t; n = len(t); r = np.random.default_rng(int(hash_(b.k) * 99999) + 5); k_ = np.arange(n)
        def P(tt, u):
            f = np.clip(tt, 0, 1) * (n - 1); return np.stack([np.interp(f, k_, b.C[:, j]) + np.interp(f, k_, b.N[:, j]) * np.interp(f, k_, b.hw) * u for j in (0, 1)], -1)
        tl, tr = s.tone(b, 1), s.tone(b, -1); ma, mb = float(tl[8:26].mean()), float(tr[8:26].mean()); sg = 1 if ma >= mb else -1
        ridge = float(np.mean((((b.R - b.C) * b.N).sum(1) / np.maximum(b.hw, 1e-3))[6:24])); big = b.hw.max(); D = s.detail
        cel.clip(b.P)
        # under-painting: the whole blade graded root -> tip on each facet
        for side in (1, -1):
            E = b.A if side > 0 else b.B; m = ma if side > 0 else mb; a, c_ = b.C[0], b.C[-1]
            cel.grad(np.vstack([b.R, E[::-1]]), a, c_, [(0, s.ramp(m - .75), 1), (.28, s.ramp(m - .30), 1), (.62, s.ramp(m), 1), (1, s.ramp(m + .10), 1)])
        # needles
        K = int(np.clip(big / 4.2, 8, 34) * D)
        for j in range(K):
            u0 = r.uniform(-1.02, 1.02); side = 1 if u0 > ridge else -1; m = (ma if side > 0 else mb); lit_ = side == sg; wd = r.uniform(.03, .16) * (1.3 if big < 60 else 1.)
            a0 = r.uniform(-.05, .62); a1 = min(1.02, a0 + r.uniform(.18, .55)); tt = np.linspace(a0, a1, 18); q = np.linspace(0, 1, 18); wv = np.interp(q, [0, .28, .7, 1], [0, 1, .8, 0]) * wd; du = r.normal(0, .07) * q
            v = m + r.normal(0, .34 if not s.flat else .22) + (.18 if lit_ else -.12) - .35 * smooth(.35, .0, (a0 + a1) / 2)
            cel.fill(np.vstack([P(tt, u0 + du - wv * .5), P(tt, u0 + du + wv * .5)[::-1]]), s.ramp(v), r.uniform(.55, .95))
        # ridge shadow: the dark facet starts with one clean cut along the ridge
        lo_ = b.B if sg > 0 else b.A; w = np.interp(t, [0, .08, .3, .7, .95, 1], [1, .95, .80, .86, .5, 0]); E = lo_ + (b.R - lo_) * w[:, None]; ccol = s.ramp(min(ma, mb) - .22)
        cel.fill(np.vstack([b.R, E[::-1]]), ccol, .85)
        # strands
        M = int(np.clip(big / 2.0, 8, 60) * D)
        for j in range(M):
            u = r.uniform(-.98, .98); a0 = r.uniform(-.02, .60); a1 = min(1., a0 + r.uniform(.22, .62)); tt = np.linspace(a0, a1, 16); du = r.normal(0, .07) * np.linspace(0, 1, 16); side = 1 if u > ridge else -1; m = ma if side > 0 else mb
            li = r.random() < (.62 if side == sg else .26); col = mixc(s.ramp(m + .45), pal['hi'], .35 + .5 * r.random()) if li else mixc(s.ramp(m - .55), pal['line'], .55 * r.random())
            cel.line(P(tt, u + du), r.uniform(.7, 2.1) * max(.55, s.u / 420), col, taper=(.40, .65), press=.3, seed=int(r.integers(1 << 30)), alpha=r.uniform(.4, .95), tmin=.02, tremor=.08)
        # cut light
        if max(ma, mb) > .0:
            for q_i in range(1 + int(big / 34)):
                a0 = r.uniform(.14, .62); a1 = a0 + r.uniform(.10, .30); u0 = ridge + sg * r.uniform(.06, .85); wd = r.uniform(.04, .15); tt = np.linspace(a0, min(.98, a1), 10); sk = r.uniform(-.05, .05); wv = np.interp(np.linspace(0, 1, 10), [0, .35, 1], [0, 1, 0])
                cel.fill(np.vstack([P(tt, u0 - wd * wv * .5), P(tt + sk, u0 + wd * wv * .5)[::-1]]), pal['hi'], .95)
        cel.unclip()

    def ramp(s, v):
        pal = s.pal; st = [(-.75, pal['line']), (-.25, pal['deep']), (.14, pal['shade']), (.42, pal['base']), (.64, pal['light']), (.84, pal['hi'])]
        xs = [a for a, _ in st]; cs = [_c(c) for _, c in st]; v = float(np.clip(v, xs[0], xs[-1])); j = int(np.clip(np.searchsorted(xs, v) - 1, 0, len(xs) - 2)); tt = (v - xs[j]) / (xs[j + 1] - xs[j])
        return cs[j] * (1 - tt) + cs[j + 1] * tt

    def ink_blade(s, cel, b):
        pal = s.pal; n = len(b.t); w = s.w_line(); cut = q_(b, 'cut'); inner = mixc(pal['line'], pal['shade'], pal.get('inmix', .0))
        for e, k0, sd in ((b.A, cut, 0), (b.B, cut * .8, 1)):
            i0 = int(n * k0); cel.line(e[i0:], w, pal['line'], taper=(.42, .10), press=.30, seed=s.seed * 17 + sd + int(hash_(b.k) * 997), tremor=.15, tmin=.10)
        if s.u < 60: return
        # interior lines: start at the tip, or at the root
        r = np.random.default_rng(int(hash_(b.k) * 9999) + 3); m = 1 if b.hw.max() < .12 * s.u else 2
        for j in range(m):
            u = (.42, -.36, .1)[j] * (1 if r.random() < .5 else -1); ii = np.arange(int(n * (.46 + .12 * r.random())), int(n * .93)); P = b.C[ii] + b.N[ii] * (b.hw[ii] * u)[:, None]
            cel.line(P, w * .56, inner, taper=(.7, .18), press=.2, seed=s.seed * 23 + j + int(hash_(b.k) * 877), tremor=.12, tmin=.05)
        for j in range(m):
            u = (-.5, .28)[j] * (1 if r.random() < .5 else -1); ii = np.arange(int(n * (cut * .6)), int(n * (cut * .6 + .22 + .10 * r.random()))); P = b.C[ii] + b.N[ii] * (b.hw[ii] * u)[:, None]
            if len(P) > 4: cel.line(P, w * .56, inner, taper=(.3, .6), press=.2, seed=s.seed * 29 + j + int(hash_(b.k) * 787), tremor=.12, tmin=.05)

    def w_line(s): return float(np.interp(math.log(max(s.u, 8)), [math.log(v) for v in (14, 40, 110, 300, 600, 1000)], [.8, 1.3, 1.9, 2.6, 3.4, 4.4])) * s.lw

    def _do(s, cel, bl):
        for b in bl:
            s.fill_blade(cel, b)
            if s.painterly and s.u > 120: s.paint_blade(cel, b)
            s.ink_blade(cel, b)

    def back(s, cel):
        for p in s.cap_back: cel.fill(p, s.pal['deep'])
        s._do(cel, [b for b in s.bl if b.dp < s.split])

    def front(s, cel, only=None):
        for p in s.cap_front:
            cel.fill(p, s.pal.get('cap', s.pal['shade']))
        if s.painterly and s.cap_front and not s.pal.get('dark'): cel.clip(s.cap_front); s.roots(cel, 1.6, s.seed, 150); cel.unclip()
        s._do(cel, [b for b in s.bl if b.dp >= s.split and (only is None or b.k[0] in only)])

    def roots(s, cel, w=1.2, seed=0, n=90, col=None):
        """short strokes that carry the hairline back into the hair"""
        H = s.H; r = np.random.default_rng(seed + 5)
        for j in range(n):
            th = r.uniform(-62, 62); y = float(np.interp(th, s.HL[:, 0], s.HL[:, 1])) - .012; k = r.uniform(.05, .17); dth = r.normal(0, 3) + (th - 8) * -.05
            P, nn = surf(H, A([th, th + dth * .5, th + dth]) * D2R, A([y, y + k * .55, y + k]), A([.0, .03, .07])); v = (nn @ H.R.T)[0, 2]
            if v < .1: continue
            cel.line(spl(H.P3(P), n=6), w * r.uniform(.6, 1.5) * max(.5, s.u / 330), col or (s.pal['line'] if r.random() < .6 else s.pal['shade']), taper=(.35, .6), seed=j, alpha=.8, tmin=.04)

    def cast(s, cel, off, col, alpha=1., keys='FCX', scale=1.):
        """hard shadow of the bangs (call clipped to the face)"""
        for b in s.bl:
            if b.k[0] in keys and b.dp > s.split: cel.fill(b.P + off, col, alpha)
        for p in s.cap_front: cel.fill(p + A(off) * .55, col, alpha)

    def polys(s, front=True): return [b.P for b in s.bl if (b.dp >= s.split) == front]


def q_(b, k): return b.q[k]


def hash_(k):
    h = 2166136261
    for ch in str(k): h = ((h ^ ord(ch)) * 16777619) & 0xffffffff
    return ((h ^ (h >> 15)) & 0xffff) / 65536.
