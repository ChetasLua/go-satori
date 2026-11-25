"""ki stage: small helpers my shots share (hero3 + energy)."""
from .ki_core import *
from .ki_core import _smax, _rib, _vn  # noqa
from . import comp
from .cam import Camera3D
from .hero3 import Fig, compose, hit, clash, grab, touch, brace, ghosts, POSES, P, mix_pose, seq, track, walk, sprint, at, tail_curve, limb_smear, stretch, tails


def safe(F, cam, f, **k):
    """draw; a head too small for its own construction falls back to no head instead of stopping the render"""
    try: return F.draw(cam, f, **k)
    except ValueError:
        k['head'] = False; return F.draw(cam, f, **k)


def face(F, k=1.35):
    h = F.info.get('head') if F.info else None
    return (h['x'], h['y'], h['s'] * k) if h else None


def rooted(pose_fn, foot_fn, n, body_fig, p0, heading, f0=0):
    """root path of a walk whose planted foot never slides.  pose_fn(f) -> pose, foot_fn(f) -> 'L' | 'R' planted.  -> (n, 2) positions"""
    out = np.zeros((n, 2)); F = body_fig; ank = lambda f, s: F.pose(pose_fn(f)).place(0, 0, 0, heading=heading).skel().an[s][:2].copy(); s = foot_fn(f0); out[0] = p0[:2]; pl = out[0] + ank(f0, s)
    for f in range(1, n):
        out[f] = pl - ank(f, s); q = foot_fn(f)
        if q != s: pl = out[f] + ank(f, q); s = q
    return out


def shaken(img, dx, dy, zoom=None):
    zoom = max(1.004, 1.003 + 2.1 * max(abs(dx) / W, abs(dy) / H)) if zoom is None else zoom
    return cv2.warpAffine(img, np.float32([[zoom, 0, dx + (1 - zoom) * W / 2], [0, zoom, dy + (1 - zoom) * H / 2]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def kick(f, hits, amp=9., decay=2.6, seed=1):
    """sum of decaying shakes: hits = [(frame, size)]"""
    d = np.zeros(2)
    for i, (a, s) in enumerate(hits):
        u = f - a
        if 0 <= u < 14: r = np.random.default_rng(seed * 91 + i * 17 + u); d += np.array([r.normal(0, .55), (-1) ** u * (1. + .3 * r.normal())]) * amp * s * math.exp(-u / decay)
    return d


def wet(hz, seed=3):
    """hard horizontal patches of standing water on a ground plane seen at a grazing angle (mask 0..1, below hz)"""
    x, y = grid(); d = np.maximum(y - hz, 1.); o = np.zeros((H, W), f32); wt = np.zeros((H, W), f32)
    for i, (sc, a) in enumerate(((10., 26.), (22., 80.), (44., 200.), (80., 380.))): g = np.exp(-(np.log(d / a)) ** 2 * 1.1); o += g * nimg(H, W, sc, 0, seed + i * 5, 3, 2, 5.5, .30); wt += g
    return np.clip((o / np.maximum(wt, 1e-3) - .02) * 14 + .5, 0, 1) * np.clip((y - hz) / 6., 0, 1)


def mirror(layer, fy, squash=.62, wob=0., t=0., seed=1):
    """layer flipped about the foot line fy (premult RGBA), bent a little by standing water"""
    x, y = grid(); sy = fy - (y - fy) / squash; sx = x + (nimg(H, W, 26, t, seed, 2, 2, 4., .25) * wob if wob else 0); return cv2.remap(layer, sx.astype(f32), np.where(y > fy, sy, -50).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def rain(f, n, seed=1, ang=.10, ln=(40, 150), w=(1., 2.4), col='#cfd6dc', a=.5, speed=95., box=(0, 0, W, H)):
    """long single strokes of falling rain -> layer"""
    F = Field(1)
    for i in range(n):
        z = h1(seed, i, 1); l = ln[0] + (ln[1] - ln[0]) * z; sp = speed * (.5 + .8 * z); y = (h1(seed, i, 2) * 1500 + f * sp) % 1500 - 220; x = h1(seed, i, 3) * 2300 - 190 + y * ang; F.rib(np.array([[x, y], [x + ang * l * .5, y + l * .5], [x + ang * l, y + l]]), [np.array([.15, 1, .3]) * (w[0] + (w[1] - w[0]) * z) * .5], 0.)
    return F.cel((col,)) * a


def lerp(a, b, t): return a + (b - a) * t
def es(t): t = float(np.clip(t, 0, 1)); return t * t * (3 - 2 * t)
def eo(t, p=2.4): return 1 - (1 - float(np.clip(t, 0, 1))) ** p
def ei(t, p=2.4): return float(np.clip(t, 0, 1)) ** p
