"""Toon inker helpers (mask -> drawn contours), from our own JJK figure kit."""
import numpy as np, cv2, math, zlib
from db.paint import rgb, f32
from db.ink import Cel, spl, resample


def _smooth(P, k, closed):
    k = int(k) | 1
    if k < 3 or len(P) < k + 2: return P
    w = np.hanning(k + 2)[1:-1]; w /= w.sum(); h = k // 2
    Q = np.vstack([P[-h:], P, P[:h]]) if closed else np.vstack([2 * P[0] - P[h:0:-1], P, 2 * P[-1] - P[-2:-h - 2:-1]])
    return np.stack([np.convolve(Q[:, i], w, 'valid') for i in (0, 1)], -1)


def _dp(P, eps, closed=False):
    """Douglas-Peucker indices"""
    n = len(P)
    if n < 3: return np.arange(n)
    q = cv2.approxPolyDP(np.round(P * 16).astype(np.int32)[:, None], eps * 16, closed)[:, 0]
    # map back to indices
    d = {}
    R = np.round(P * 16).astype(np.int32)
    idx = []
    j = 0
    key = {}
    for i, p in enumerate(map(tuple, R)): key.setdefault(p, i)
    for p in map(tuple, q):
        if p in key: idx.append(key[p])
    return np.array(sorted(set(idx)))


def draw_contour(P, eps, closed=True, sharp=.75, n=5):
    """straights against curves: simplify, keep corners sharper than `sharp` rad, bend the rest"""
    if len(P) < 8: return P
    idx = _dp(P, eps, closed)
    if len(idx) < 3: return P[[0, len(P) // 2, -1]] if not closed else P
    V = P[idx]
    m = len(V)
    a = np.roll(V, 1, 0) if closed else np.vstack([V[:1], V[:-1]]); b = np.roll(V, -1, 0) if closed else np.vstack([V[1:], V[-1:]])
    t1 = np.arctan2(*(V - a).T[::-1]); t2 = np.arctan2(*(b - V).T[::-1]); turn = np.abs((t2 - t1 + math.pi) % (2 * math.pi) - math.pi)
    c = [i for i in range(m) if turn[i] > sharp and (closed or 0 < i < m - 1)]
    if closed:
        if len(c) == 1: c.append((c[0] + m // 2) % m)
        return spl(V, True, corners=c, n=n)
    return spl(V, False, corners=c, n=n)


def clean_closed(P, eps, sharp=.5, n=6):
    """closed raw contour -> (clean closed polyline, dp index list, n): straights against curves, corners kept.  raw index r maps to clean index via clean_index()"""
    idx = _dp(P, eps, True)
    if len(idx) < 4:
        idx = np.unique(np.linspace(0, len(P) - 1, 5).astype(int)[:-1])
    V = P[idx]; d1 = np.roll(V, -1, 0) - V; L1 = np.hypot(*d1.T) + 1e-6; u1 = d1 / L1[:, None]; u0 = np.roll(u1, 1, 0); L0 = np.roll(L1, 1)
    turn = np.abs((np.arctan2(u1[:, 1], u1[:, 0]) - np.arctan2(u0[:, 1], u0[:, 0]) + math.pi) % (2 * math.pi) - math.pi)
    T = u0 * L1[:, None] + u1 * L0[:, None]; T /= np.hypot(*T.T)[:, None] + 1e-6
    c = (turn > sharp)[:, None]
    to = np.where(c, u1, T); ti = np.where(c, u0, T)
    t = (np.arange(n) / n)[None, :, None]; h00 = 2 * t ** 3 - 3 * t ** 2 + 1; h10 = t ** 3 - 2 * t ** 2 + t; h01 = -2 * t ** 3 + 3 * t ** 2; h11 = t ** 3 - t ** 2
    d = L1[:, None, None] * .92; V1 = np.roll(V, -1, 0)
    Q = h00 * V[:, None] + h10 * to[:, None] * d + h01 * V1[:, None] + h11 * np.roll(ti, -1, 0)[:, None] * d
    return Q.reshape(-1, 2), idx, n


def clean_index(r, idx, n, N):
    """position on the clean contour of raw index r (0..N)"""
    k = np.searchsorted(idx, r, side='right') - 1
    if k < 0: a = idx[-1] - N; b = idx[0]; k = len(idx) - 1; return (k + (r - a) / max(b - a, 1)) * n
    a = idx[k]; b = idx[k + 1] if k + 1 < len(idx) else idx[0] + N
    return (k + (r - a) / max(b - a, 1)) * n


def mask_polys(m, g, box, eps=.6, up=2, min_area=10., smooth=5, sharp=.75, raw=False, sig=.6):
    """uint8/bool mask at G resolution -> list of screen polylines"""
    ys, xs = np.nonzero(m)
    if len(ys) < 3: return []
    y0, y1, x0, x1 = max(0, ys.min() - 3), ys.max() + 4, max(0, xs.min() - 3), xs.max() + 4
    s = m[y0:y1, x0:x1].astype(f32)
    s = cv2.copyMakeBorder(s, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    if sig > 0: s = cv2.GaussianBlur(s, (0, 0), sig)
    s = cv2.resize(s, None, fx=up, fy=up, interpolation=cv2.INTER_LINEAR)
    cs, _ = cv2.findContours((s > .5).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if len(c) < 8: continue
        c = c[:, 0].astype(float)
        if abs(cv2.contourArea(c.astype(f32))) < min_area * up * up: continue
        c = _smooth(c, smooth * up, True)
        c = ((c + .5) / up - 2 + [x0, y0]) / g + [box[0], box[1]]
        out.append(c if raw else draw_contour(c, eps, True, sharp))
    return out


def _hash(*a):
    return zlib.crc32(repr(a).encode()) & 0x7fffffff


def _mixc(a, b, t):
    a = rgb(a) if isinstance(a, str) else np.asarray(a, f32); b = rgb(b) if isinstance(b, str) else np.asarray(b, f32)
    return a * (1 - t) + b * t


def resample2(P, step):
    P = np.asarray(P, float); d = np.hypot(*np.diff(P, axis=0).T); s_ = np.r_[0, np.cumsum(d)]
    n = max(len(P), int(s_[-1] / step) + 2); t = np.linspace(0, s_[-1], n)
    return np.c_[np.interp(t, s_, P[:, 0]), np.interp(t, s_, P[:, 1])]


def _true_runs(ok, n):
    out = []; a = None
    for i, v in enumerate(list(ok) + [False]):
        if v and a is None: a = i
        if not v and a is not None:
            if i - a >= n: out.append((a, i))
            a = None
    return out
