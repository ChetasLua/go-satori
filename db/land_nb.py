"""Compiled kernels of land.py: z-buffer, painted rock, painted water.  (apart, so that editing land.py never recompiles)"""
import numpy as np, math
from numba import njit, prange
from .paint import _vn, _h2


@njit(parallel=True, cache=True, fastmath=True)
def raster(PX, PY, IZ, w, h, ID, ZB, B1, B2, nb):
    """perspective-correct triangles into ID / 1-over-z / barycentrics (ID preset to -1, ZB to 0)"""
    n = PX.shape[0]
    for b in prange(nb):
        r0 = h * b // nb; r1 = h * (b + 1) // nb
        for i in range(n):
            ay = PY[i, 0]; by = PY[i, 1]; cy = PY[i, 2]
            mn = min(ay, min(by, cy)); mx = max(ay, max(by, cy))
            if mx < r0 or mn > r1: continue
            ax = PX[i, 0]; bx = PX[i, 1]; cx = PX[i, 2]
            if max(ax, max(bx, cx)) < 0 or min(ax, min(bx, cx)) > w: continue
            area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
            if abs(area) < 1e-7: continue
            inv = 1.0 / area
            a1 = (cy - ay) * inv; b1 = -(cx - ax) * inv; c1 = -(ax * a1 + ay * b1)
            a2 = -(by - ay) * inv; b2 = (bx - ax) * inv; c2 = -(ax * a2 + ay * b2)
            za = IZ[i, 0]; zb = IZ[i, 1]; zc = IZ[i, 2]
            ya = max(r0, int(math.floor(mn - .5))); yb = min(r1 - 1, int(mx + .5))
            for y in range(ya, yb + 1):
                py = y + 0.5
                lo = 0.0; hi = float(w)
                k1 = b1 * py + c1; k2 = b2 * py + c2
                ok = True
                if a1 > 1e-12: lo = max(lo, -k1 / a1)
                elif a1 < -1e-12: hi = min(hi, -k1 / a1)
                elif k1 < 0: ok = False
                if a2 > 1e-12: lo = max(lo, -k2 / a2)
                elif a2 < -1e-12: hi = min(hi, -k2 / a2)
                elif k2 < 0: ok = False
                a0 = -(a1 + a2); k0 = 1 - k1 - k2
                if a0 > 1e-12: lo = max(lo, -k0 / a0)
                elif a0 < -1e-12: hi = min(hi, -k0 / a0)
                elif k0 < 0: ok = False
                if not ok or lo > hi: continue
                xa = max(0, int(math.ceil(lo - .5))); xb = min(w - 1, int(math.floor(hi - .5)))
                for x in range(xa, xb + 1):
                    px = x + .5
                    w1 = a1 * px + k1; w2 = a2 * px + k2; w0 = 1 - w1 - w2
                    iz = w0 * za + w1 * zb + w2 * zc
                    if iz > ZB[y, x]:
                        ZB[y, x] = iz; ID[y, x] = i
                        B1[y, x] = w1 * zb / iz; B2[y, x] = w2 * zc / iz


@njit(cache=True, fastmath=True)
def _cell(u, v, s):
    """voronoi: crevice distance (d2-d1), cell hash, |cos| of the border normal to u (1 = upright border)"""
    ix = int(math.floor(u)); iy = int(math.floor(v))
    d1 = 9.; d2 = 9.; hid = 0.; ax = 0.; ay = 0.; bx = 0.; by = 0.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            jx = ix + dx; jy = iy + dy
            h = _h2(jx, jy, s); h2 = h * 197.31; h2 -= math.floor(h2)
            px = jx + .5 + (h - .5) * .9 - u; py = jy + .5 + (h2 - .5) * .9 - v
            d = math.sqrt(px * px + py * py)
            if d < d1:
                d2 = d1; bx = ax; by = ay; d1 = d; hid = h; ax = px; ay = py
            elif d < d2: d2 = d; bx = px; by = py
    ex = bx - ax; ey = by - ay
    return d2 - d1, hid, abs(ex) / (math.sqrt(ex * ex + ey * ey) + 1e-6), ax, ay


@njit(cache=True, fastmath=True)
def _ss(a, b, x):
    t = (x - a) / (b - a)
    if t < 0: t = 0.
    if t > 1: t = 1.
    return t * t * (3 - 2 * t)


@njit(cache=True, fastmath=True)
def _brush(u, v, mpp, s, au, av):
    """dry-brush streak noise whose stroke is always ~ the same size on screen: two octaves cross-faded"""
    q = math.log2(max(mpp, 1e-5)); k = math.floor(q); t = q - k
    t = _ss(.15, .85, t)
    f0 = 2. ** (-k)
    a = _vn(u * f0 * au, v * f0 * av, s) * .6 + _vn(u * f0 * au * 2.3, v * f0 * av * 2.1, s + 1) * .4
    b = _vn(u * f0 * .5 * au, v * f0 * .5 * av, s) * .6 + _vn(u * f0 * .5 * au * 2.3, v * f0 * .5 * av * 2.1, s + 1) * .4
    return (a + (b - a) * t) - .5


# par[mat]: 16 canopy on, 17 crown size (m); 0 key gain, 1 ambient floor, 2 sky gain, 3 side/bounce gain, 4 chisel gain, 5 wall stretch, 6 crack amount, 7 strata amount,
#           8 brush amount, 9 edge highlight, 10 ramp sharpness (edge half-width), 11 hue wander, 12 base cell scale, 13 flat-top cell scale,
#           14 wetness (sky gloss on tops), 15 grass amount on tops
@njit(cache=True, fastmath=True)
def _rock(X, Y, Z, fx, fy, fz, nx, ny, nz, ao, mat, sd, mpp, ev, eh, L, shd, par, RAMP, B, vx, vy, vz, GR):
    """-> r g b (before fire, fog).  ev = signed edge value (+ convex light, - crease) already in px terms 0..1"""
    p = par[mat]; NR = RAMP.shape[1]
    flat = abs(fz) > .72
    # smooth normal carries the big form, facet a part of it
    sm = .45 if p[16] <= 0 else .86
    mx = nx * sm + fx * (1 - sm); my = ny * sm + fy * (1 - sm); mz = nz * sm + fz * (1 - sm)
    l = math.sqrt(mx * mx + my * my + mz * mz) + 1e-9; mx /= l; my /= l; mz /= l
    if flat: u = X; v = Y; st = 1.0; cs = p[13]
    else:
        hl = math.sqrt(fx * fx + fy * fy) + 1e-9
        u = (-fy * X + fx * Y) / hl; v = Z; st = p[5]; cs = p[12]
    # warp the painting coordinates a little so that no plane edge is straight
    wu = (_vn(u * .11 + 7, v * .07, sd) - .5) * 3.2 * cs; wv = (_vn(u * .09, v * .11 + 5, sd + 1) - .5) * 3.2 * cs
    tx = 0.; ty = 0.; tz = 0.; crack = 0.; hue = 0.; lipl = 0.; dv = 0.
    cg = p[4]
    if not flat:
        # strata, and upright flutes cut differently in every stratum: the big drawn planes of the cliff
        bh = (7.5 + 5. * _h2(sd, 3, 9)) * cs; q = Z / bh + .42 * _vn(u * .045 / cs, 3., sd + 5) + .16 * _vn(u * .33 / cs, 9., sd + 6); ks = int(math.floor(q)); fs = q - ks
        hs = _h2(ks, sd, 77); dv += (hs - .5) * .12 * p[7]
        ep = min(fs, 1 - fs) * bh / mpp
        if _vn(u / (bh * 1.3), ks * 3.7, sd + 8) > .40:
            if fs < .5 and ep < 2.6: lipl = max(lipl, .7 * p[7])
            elif fs >= .5 and ep < 1.2 + .10 * cs / mpp: crack = max(crack, .85 * p[7])
        if fs > .82: tz -= .30 * p[7] * (fs - .82) / .18        # undercut beneath the next ledge
        wf = (6.0 + 6.5 * hs) * cs
        for o in range(2):
            wg = _ss(5., 12., wf / mpp) * cg
            if wg > 0:
                q = u / wf + (wv * .10 + (_vn(u * .2 / cs, v * .10 / cs, sd + 33 + o) - .5) * .9) + hs * 7.3 * (o + 1); kf = int(math.floor(q)); fr = q - kf
                hf = _h2(kf, ks + 31 * o, sd + 91); h2 = hf * 91.7; h2 -= math.floor(h2)
                sp = .30 + .40 * h2; am = (.10 + .60 * hf * hf) * wg * (.72 - .44 * o)
                sgn = -1. if fr < sp else 1.
                tx += -fy / hl * am * sgn; ty += fx / hl * am * sgn; dv += (hf - .5) * .05 * wg
                e2 = min(fr, 1 - fr) * wf / mpp; gw = min(3.6, .5 + wf * .045 / mpp) * (.2 + 1.5 * _vn(v / wf * .8, kf * 5.1, sd + 35))
                if e2 < gw and h2 > .35: crack = max(crack, min(1., (gw - e2) * .9) * (.9 - .3 * o) * wg)
            wf *= .36
        cg *= .62
    size = (21.0 if not flat else 19.) * cs
    for o in range(5):
        wgt = _ss(6., 15., size / mpp)
        if wgt <= 0: break
        amp = (.34, .30, .25, .20, .16)[o] * cg * wgt
        asp = st if o != 1 else (.75 if not flat else 1.)
        cu = (u + wu * (.5 ** o)) / size; cv = (v + wv * (.5 ** o)) / (size * asp)
        e, h, upb, qx_, qy_ = _cell(cu, cv, sd + o * 17)
        a = h * 6.2831853 * 3.; q = h * 53.7; q -= math.floor(q)
        ta = math.cos(a) * amp; tb = math.sin(a) * amp * .8
        if flat:
            tx += ta * .45; ty += tb * .45
        else:
            tx += -fy / hl * ta; ty += fx / hl * ta; tz += tb
        hue += (q - .5) * wgt * (.5 if o < 3 else .25); dv += (q - .5) * .035 * wgt
        # cracks: some borders only, thin on screen, upright ones on walls, with a lit lip beside them
        if o >= 1 and p[6] > 0:
            epx = e * size / mpp * .8
            wd = min(3.2, .7 + size * .026 / mpp) * (.15 + 1.5 * _vn(u / size * 1.9, v / size * 1.9, sd + 40 + o))
            on = _vn(u / size * .55 + 31, v / size * .4, sd + 60 + o)
            if flat: on += .02
            else: on += (upb - .78) * .5
            if on > .56:
                g = min(1., (on - .56) * 6.)
                c = 1 - epx / max(wd, .6)
                if c > 0: crack = max(crack, min(1., c * 1.6) * g * wgt * (.95 if o < 3 else .6))
                elif epx < wd * 2.2 and h > .5: lipl = max(lipl, g * wgt * .5)
        size *= .38
    if p[16] > 0:
        # forest: crowns as domes in three sizes (crown, bough, leaf clump); u runs along the slope, v up it
        tx *= .25; ty *= .25; tz *= .25; crack = 0.; lipl = 0.
        if flat: ux_ = X; uy_ = Y
        else: ux_ = u; uy_ = v
        sz = p[17]
        for o in range(4):
            wg = _ss(3.5, 9., sz / mpp)
            if wg <= 0: break
            e, h, upb, qx_, qy_ = _cell(ux_ / sz + o * 3.1 + (_vn(ux_ / sz * 1.7, uy_ / sz * 1.7, sd + 71) - .5) * .7 + (_vn(ux_ / sz * 4.6, uy_ / sz * 4.6, sd + 73) - .5) * .42, uy_ / sz * (1.15 if o == 0 else 1.) + (_vn(ux_ / sz * 1.7 + 9, uy_ / sz * 1.7, sd + 72) - .5) * .7 + (_vn(ux_ / sz * 4.6 + 5, uy_ / sz * 4.6, sd + 74) - .5) * .42, sd + 60 + o * 13)
            k = (1.15, .75, .50, .36)[o] * wg
            if flat: tx += -qx_ * k * 1.5; ty += -qy_ * k * 1.5
            else: tx += fy / hl * qx_ * k * 1.5; ty += -fx / hl * qx_ * k * 1.5; tz += -qy_ * k * 1.7 + .12 * k
            dv += (h - .5) * (.14, .08, .06, .04)[o] * wg
            if e < .16: dv -= (.16 - e) * (1.1, .6, .35, .2)[o] * wg
            hue += (h - .5) * (.9, .5, .3, .2)[o] * wg
            sz *= .42
    mx += tx; my += ty; mz += tz
    l = math.sqrt(mx * mx + my * my + mz * mz) + 1e-9; mx /= l; my /= l; mz /= l
    d = mx * L[0] + my * L[1] + mz * L[2]
    key = (.66 * _ss(-.04, .20, d) + .34 * _ss(.70, .84, d)) * shd
    up = .5 + .5 * mz
    side = .75 * _ss(.05, .40, mx * B[0] + my * B[1] + mz * B[2]) + .25 * _ss(.6, .8, mx * B[0] + my * B[1] + mz * B[2])
    val = (p[1] + p[2] * up * up + p[3] * side) * (.35 + .65 * ao) + p[0] * key * (.55 + .45 * ao) + dv * (.5 + .7 * key)
    if flat: dv += (_vn(X * .045 / cs + 3, Y * .045 / cs, 61) * .6 + _vn(X * .16 / cs, Y * .16 / cs + 7, 62) * .4 - .5) * .34
    br = _brush(u + wu, v + wv, mpp, sd + 11, .11, .028) if not flat else _brush(u + wu, v + wv, mpp, sd + 11, .035, .10)
    br2 = _brush(u - wv, v + wu, mpp, sd + 19, .030, .022)
    val += (br * .15 + br2 * .13) * p[8]
    if ev > 0: val += ev * p[9] * (.25 + .75 * shd) * max(0., .35 + d)
    else: val += ev * .28
    val += lipl * .12 * (.3 + .7 * key)
    val -= crack * p[6] * (.10 + .30 * val)
    if val < 0: val = 0.
    if val > .9999: val = .9999
    q = val * (NR - 1); k = int(q); fr = q - k
    fr = _ss(.5 - p[10], .5 + p[10], fr)
    r = RAMP[mat, k, 0] + (RAMP[mat, k + 1, 0] - RAMP[mat, k, 0]) * fr
    g = RAMP[mat, k, 1] + (RAMP[mat, k + 1, 1] - RAMP[mat, k, 1]) * fr
    b = RAMP[mat, k, 2] + (RAMP[mat, k + 1, 2] - RAMP[mat, k, 2]) * fr
    hw = hue * p[11] + br2 * p[11] * .8
    m = 1 + br * .10 * p[8]
    r *= m * (1 + hw); g *= m; b *= m * (1 - hw)
    if p[15] > 0 and fz > .25:                       # turf on the ledges: ragged edge, its own ramp
        n = _vn(X * .16, Y * .16, sd + 3) * .55 + _vn(X * .7, Y * .7, sd + 4) * .3 + _brush(X, Y, mpp, 5, .1, .06) * .5
        t = _ss(-.03, .03, fz - .50 + (n - .42) * .9) * p[15]
        if t > 0:
            q = max(0., min(.9999, val * .85 + .06 + (n - .5) * .22)) * (GR.shape[0] - 1); k = int(q); fr = _ss(.25, .75, q - k)
            r += (GR[k, 0] + (GR[k + 1, 0] - GR[k, 0]) * fr - r) * t; g += (GR[k, 1] + (GR[k + 1, 1] - GR[k, 1]) * fr - g) * t; b += (GR[k, 2] + (GR[k + 1, 2] - GR[k, 2]) * fr - b) * t
    return r, g, b, val, mx, my, mz


# wat: 0 level z, 1 on, 2 wave scale (m), 3 slope, 4 reflect gain, 5 glint gain, 6 foam gain, 7 stroke contrast, 8 time, 9 deep mix far, 10 ripple distort px, 11 glint sigma, 12 rings on, 13 clarity
@njit(cache=True, fastmath=True)
def _water(X, Y, z, dx_, dy_, dz_, rl, mpp, wat, WC, L, REF, px, py, f, hzy, fm, fmm, shd, rings, SKYR):
    """painted water at (X,Y).  WC rows: 0 near deep, 1 far, 2 stroke dark, 3 stroke light, 4 glint, 5 foam, 6 ring light"""
    t = wat[8]
    vz = -dz_ / rl                                    # sin of grazing angle
    fore = max(vz, .01)
    mppy = mpp / fore                                 # metres per px along the view
    ux = X; uy = Y
    n1 = 0.; n2 = 0.; sx = 0.; sy = 0.
    sc = wat[2]
    # wave field: three layers travelling, coarse ones always, fine ones only where they can be drawn
    for o in range(4):
        wgt = _ss(1.6, 5., sc / mppy) if o > 0 else max(.25, _ss(1.2, 4., sc / mppy))
        if wgt > 0:
            a = _vn(ux / (sc * 2.6) + t * .22 * (1 + o * .3), uy / sc - t * .30, 31 + o) - .5
            b = _vn(ux / (sc * 2.1) - t * .17 + 9.3, uy / (sc * .8) + t * .21 + 4.1, 51 + o) - .5
            n1 += a * wgt * (.62 ** o); n2 += b * wgt * (.62 ** o)
        sc *= .42
    far = _ss(120., 2400., z * rl) ** .6
    fres = (1 - vz) ** 3.2 * .96 + .04
    r = WC[0, 0] + (WC[1, 0] - WC[0, 0]) * far; g = WC[0, 1] + (WC[1, 1] - WC[0, 1]) * far; b = WC[0, 2] + (WC[1, 2] - WC[0, 2]) * far
    # rings: pressure fronts / rain rings  rows (x, y, R, width, amp)
    ra = 0.
    for j in range(rings.shape[0]):
        dd = math.sqrt((X - rings[j, 0]) ** 2 + (Y - rings[j, 1]) ** 2) - rings[j, 2]
        w_ = max(rings[j, 3], mpp * .9)
        if abs(dd) < w_ * 3: ra += rings[j, 4] * math.exp(-(dd / w_) ** 2) * (1. if dd < 0 else .9) - rings[j, 4] * .55 * math.exp(-((dd + w_ * 1.8) / w_) ** 2)
    # reflection with ripple distortion
    rx = 0; ry = 0
    if wat[4] > 0:
        dsp = wat[10] * (n1 + n2) * (1.2 + .6 * (1 - far))
        rx = int(px + dsp * .35 + ra * 10); ry = int(py + dsp + ra * 14)
        H_, W_ = REF.shape[0], REF.shape[1]
        if rx < 0: rx = 0
        if rx >= W_: rx = W_ - 1
        if ry < 0: ry = 0
        if ry >= H_: ry = H_ - 1
        k = min(1., fres * wat[4] + wat[13])
        r += (REF[ry, rx, 0] - r) * k; g += (REF[ry, rx, 1] - g) * k; b += (REF[ry, rx, 2] - b) * k
    rlum = 1.
    if wat[4] > 0: rlum = _ss(.11, .20, (REF[ry, rx, 0] + REF[ry, rx, 1] + REF[ry, rx, 2]) * .333)
    # painted strokes: hard-edged lens shapes on the wave field
    s = n1 * 1.0 + n2 * .7
    e = .05 + .10 * far
    kd = _ss(.16 - e, .16 + e, -s) * wat[7]; kl = _ss(.17 - e, .17 + e, s) * wat[7] * (.4 + .6 * shd)
    r += (WC[2, 0] - r) * kd * .55; g += (WC[2, 1] - g) * kd * .55; b += (WC[2, 2] - b) * kd * .55
    r += (WC[3, 0] - r) * kl * .42; g += (WC[3, 1] - g) * kl * .42; b += (WC[3, 2] - b) * kl * .42
    # moon / sun road: slope needed to mirror the light toward the eye
    if wat[5] > 0:
        hx = L[0] - dx_ / rl; hy = L[1] - dy_ / rl; hz = L[2] - dz_ / rl
        if hz > 1e-4:
            qx = hx / hz; qy = hy / hz
            sg = wat[11]
            pr = math.exp(-(qx * qx + qy * qy * .55) / (sg * sg))
            kk = _vn(ux / (wat[2] * 1.1) + t * .9, uy / (wat[2] * .16) - t * .6, 91) * .5 + _vn(ux / max(wat[2] * .34, mpp * 5) - t * 1.1, uy / max(wat[2] * .09, mppy * 1.3) + t * .4, 93) * .5
            th = 1 - pr * (.74 + .20 * far)
            gl = _ss(th - .05, th + .05, kk * .86 + .05) * wat[5] * rlum
            r += (WC[4, 0] - r) * gl; g += (WC[4, 1] - g) * gl; b += (WC[4, 2] - b) * gl
    if ra != 0:
        if ra > 0:
            k = min(1., ra); r += (WC[6, 0] - r) * k; g += (WC[6, 1] - g) * k; b += (WC[6, 2] - b) * k
        else:
            k = min(.7, -ra); r += (WC[2, 0] * .8 - r) * k; g += (WC[2, 1] * .8 - g) * k; b += (WC[2, 2] * .8 - b) * k
    # shore foam from the baked distance map (metres to the nearest rock at water level)
    if wat[6] > 0:
        su = (X - fmm[0]) * fmm[2]; sv = (Y - fmm[1]) * fmm[2]
        if su > 1 and sv > 1 and su < fm.shape[1] - 2 and sv < fm.shape[0] - 2:
            iu = int(su); iv = int(sv); fu = su - iu; fv = sv - iv
            dd = (fm[iv, iu] * (1 - fu) + fm[iv, iu + 1] * fu) * (1 - fv) + (fm[iv + 1, iu] * (1 - fu) + fm[iv + 1, iu + 1] * fu) * fv
            if dd < 16:
                nz = _vn(ux * .22 + t * .3, uy * .22, 71) * .6 + _vn(ux * .9, uy * .9 - t * .5, 73) * .4
                a = 0.
                for j in range(3):
                    c = 1.2 + j * 4.2 + 1.4 * math.sin(t * 1.3 + j * 2.1 + nz * 3) + (nz - .5) * 3.4
                    w_ = max(.55 - .12 * j, mppy * .6) * (.3 + 1.4 * nz)
                    br = _vn(ux * .09 + j * 7, uy * .09 + t * .05, 75)
                    if abs(dd - c) < w_ and br > .36 + .09 * j: a = max(a, (1 - .32 * j))
                if dd < .5 + nz * 1.1: a = 1.
                a *= wat[6]
                r += (WC[5, 0] - r) * a; g += (WC[5, 1] - g) * a; b += (WC[5, 2] - b) * a
    return r, g, b


@njit(parallel=True, cache=True, fastmath=True)
def shade(ID, ZB, B1, B2, w, h, ss, eye, R, f, cx, cy, PX, PY, FN, VN, VA, MAT, SD, EDG, COL, L, SMA, SMmA, smresA, smbiasA, SMB, SMmB, smresB, smbiasB, par, RAMP, B, GR,
          fires, fog, fogc, wat, WC, REF, fm, fmm, rings, GL, out, cov, zout, idout, vout):
    """fires rows: x y z radius r g b.  fog: 0 density 1 start 2 height falloff 3 floor 4 base 5 power 6 max.  fogc: 0 base 1 toward light 2 (dir) 3 low"""
    nf = fires.shape[0]; k2 = 1.0 / (ss * ss)
    for yy in prange(h):
        for xx in range(w):
            rr = 0.; gg = 0.; bb = 0.; cc = 0.; zmin = 1e6; idc = -3; vv = 0.
            lastid = -9; lr = 0.; lg = 0.; lb = 0.; lc = 0.; lz = 1e6; lv = 0.
            # are all samples the same triangle / class?
            same = True
            i0 = ID[yy * ss, xx * ss]
            if ss > 1:
                for sy in range(ss):
                    for sx in range(ss):
                        if ID[yy * ss + sy, xx * ss + sx] != i0: same = False
            m = 1 if same else ss
            for sy in range(m):
                for sx in range(m):
                    if same: fy_ = .5; fx_ = .5; py = yy * ss + ss // 2; px = xx * ss + ss // 2
                    else: fy_ = (sy + .5) / ss; fx_ = (sx + .5) / ss; py = yy * ss + sy; px = xx * ss + sx
                    i = ID[py, px]
                    dx = (xx + fx_ - cx) / f; dy = (yy + fy_ - cy) / f
                    wx = R[0, 0] * dx + R[1, 0] * dy + R[2, 0]; wy = R[0, 1] * dx + R[1, 1] * dy + R[2, 1]; wz = R[0, 2] * dx + R[1, 2] * dy + R[2, 2]
                    rl = math.sqrt(wx * wx + wy * wy + wz * wz)
                    z = 1e6
                    if i >= 0: z = 1.0 / ZB[py, px]
                    tw = -1.
                    if wat[1] > 0 and wz < -1e-5: tw = (wat[0] - eye[2]) / wz
                    iswat = tw > 0 and tw < z
                    if iswat: z = tw
                    if i < 0 and not iswat:
                        lc = 0.; lr = lg = lb = 0.
                        continue
                    X = eye[0] + wx * z; Y = eye[1] + wy * z; Z = eye[2] + wz * z
                    mpp = z / f
                    # cast shadow
                    shd = 1.0
                    for lev in range(2):
                        if lev == 0: smres = smresA; smbias = smbiasA
                        else: smres = smresB; smbias = smbiasB
                        if smres <= 0: continue
                        bxn = 0.; byn = 0.; bzn = 0.
                        if not iswat: bxn = FN[i, 0]; byn = FN[i, 1]; bzn = FN[i, 2]
                        X_ = X + bxn * smbias; Y_ = Y + byn * smbias; Z_ = Z + bzn * smbias
                        M0 = SMmA if lev == 0 else SMmB
                        su = M0[0, 0] * X_ + M0[0, 1] * Y_ + M0[0, 2] * Z_ + M0[0, 3]; sv = M0[1, 0] * X_ + M0[1, 1] * Y_ + M0[1, 2] * Z_ + M0[1, 3]
                        sdp = M0[2, 0] * X_ + M0[2, 1] * Y_ + M0[2, 2] * Z_ + M0[2, 3]
                        if su > 2 and sv > 2 and su < smres - 3 and sv < smres - 3:
                            # hand-cut shadow edge
                            jt = .55 / smbias if lev == 0 and smresB > 0 else 2.4
                            su += (_vn(X * .35 + Z * .2, Y * .35 + Z * .3, 3) - .5) * jt; sv += (_vn(X * .35 + 9, Y * .35 + Z * .33, 4) - .5) * jt
                            if su < 2 or sv < 2 or su > smres - 3 or sv > smres - 3: break
                            iu = int(su - .5); iv = int(sv - .5); fu = su - .5 - iu; fv = sv - .5 - iv; th = sdp + smbias
                            if lev == 0:
                                a00 = 1. if SMA[iv, iu] < th else 0.; a01 = 1. if SMA[iv, iu + 1] < th else 0.; a10 = 1. if SMA[iv + 1, iu] < th else 0.; a11 = 1. if SMA[iv + 1, iu + 1] < th else 0.
                            else:
                                a00 = 1. if SMB[iv, iu] < th else 0.; a01 = 1. if SMB[iv, iu + 1] < th else 0.; a10 = 1. if SMB[iv + 1, iu] < th else 0.; a11 = 1. if SMB[iv + 1, iu + 1] < th else 0.
                            shd = (a00 * (1 - fu) + a01 * fu) * (1 - fv) + (a10 * (1 - fu) + a11 * fu) * fv
                            break
                    nx = 0.; ny = 0.; nz = 1.; val = .3
                    if iswat:
                        r, g, b = _water(X, Y, z, wx, wy, wz, rl, mpp, wat, WC, GL, REF, xx, yy, f, 0., fm, fmm, shd, rings, WC)
                    else:
                        b1 = B1[py, px]; b2 = B2[py, px]; b0 = 1 - b1 - b2
                        fx = FN[i, 0]; fy = FN[i, 1]; fz = FN[i, 2]
                        if fx * wx + fy * wy + fz * wz > 0: fx = -fx; fy = -fy; fz = -fz
                        nx = VN[i, 0, 0] * b0 + VN[i, 1, 0] * b1 + VN[i, 2, 0] * b2; ny = VN[i, 0, 1] * b0 + VN[i, 1, 1] * b1 + VN[i, 2, 1] * b2; nz = VN[i, 0, 2] * b0 + VN[i, 1, 2] * b1 + VN[i, 2, 2] * b2
                        if nx * fx + ny * fy + nz * fz < 0: nx = fx; ny = fy; nz = fz
                        ao = VA[i, 0] * b0 + VA[i, 1] * b1 + VA[i, 2] * b2
                        # drawn edges: px distance to flagged edges, broken along their length
                        ev = 0.
                        ax = PX[i, 0]; ay = PY[i, 0]; bx = PX[i, 1]; by = PY[i, 1]; cx_ = PX[i, 2]; cy_ = PY[i, 2]
                        sxp = (xx + fx_) * ss; syp = (yy + fy_) * ss
                        for j in range(3):
                            eg = EDG[i, j]
                            if eg == 0 or par[MAT[i], 16] > 0: continue
                            if j == 0: ux = bx; uy = by; qx = cx_; qy = cy_; bj = b0
                            elif j == 1: ux = cx_; uy = cy_; qx = ax; qy = ay; bj = b1
                            else: ux = ax; uy = ay; qx = bx; qy = by; bj = b2
                            ex = qx - ux; ey = qy - uy; el = math.sqrt(ex * ex + ey * ey) + 1e-6
                            dist = abs((sxp - ux) * ey - (syp - uy) * ex) / el / ss
                            along = ((sxp - ux) * ex + (syp - uy) * ey) / el
                            tn = _vn((X + Y * .7) * .35 + Z * .3, (Y - X * .6) * .3 + Z * .25, 17) * .6 + _vn(X * 1.7 + Z, Y * 1.7 - Z, 19) * .4
                            wd = min(4.2, .9 + .16 / mpp) * (tn * 2.2 - .55) * min(1., abs(eg) * 1.3)
                            if wd > 0:
                                c = wd - dist
                                if c > 0:
                                    c = min(1., c * 1.2)
                                    if eg > 0: ev = max(ev, c * eg)
                                    else: ev = min(ev, c * eg)
                        r, g, b, val, nx, ny, nz = _rock(X, Y, Z, fx, fy, fz, nx, ny, nz, ao, MAT[i], SD[i], mpp, ev, 0., L, shd, par, RAMP, B, wx, wy, wz, GR)
                        r *= COL[i, 0]; g *= COL[i, 1]; b *= COL[i, 2]
                        # wet / glossy top sheen: mirror of the sky picture
                        wt = par[MAT[i], 14]
                        if wt > 0 and nz > .55:
                            k = wt * (nz - .55) / .45 * ((1 + wz / rl) ** 2) * (.35 + .65 * _ss(.42, .58, _vn(X * .5, Y * .5, 23) * .7 + _vn(X * 2.2, Y * 2.2, 29) * .3))
                            r += (fogc[3, 0] - r) * k; g += (fogc[3, 1] - g) * k; b += (fogc[3, 2] - b) * k
                    for j in range(nf):
                        ex = fires[j, 0] - X; ey = fires[j, 1] - Y; ez = fires[j, 2] - Z
                        d2 = ex * ex + ey * ey + ez * ez; rad = fires[j, 3]
                        if d2 > rad * rad: continue
                        d = math.sqrt(d2) + 1e-6
                        nd = (nx * ex + ny * ey + nz * ez) / d
                        nd = (nd + .18) / 1.18
                        if nd <= 0: continue
                        wn = 1 - d / rad
                        a = wn * wn * nd * 2.2
                        a = a + (_vn(X * .4 + Z * .3, Y * .4 + Z * .2, 47) - .5) * .10 * min(1., a * 4)
                        # three flat steps of firelight
                        a = (.16 * _ss(.06, .10, a) + .30 * _ss(.24, .30, a) + .44 * _ss(.58, .66, a) + .30 * _ss(1.0, 1.1, a))
                        gain = (.16 + .62 * (r + g + b))
                        r += fires[j, 4] * a * gain; g += fires[j, 5] * a * gain; b += fires[j, 6] * a * gain
                    if fog[0] > 0:
                        dist = z * rl
                        hh = 1.0
                        if fog[2] > 0: hh = math.exp(-max(0.0, Z - fog[4]) * fog[2])
                        k = 1 - math.exp(-max(0.0, dist - fog[1]) * fog[0] * (fog[3] + (1 - fog[3]) * hh))
                        if k > fog[6]: k = fog[6]
                        dd = (wx * fogc[2, 0] + wy * fogc[2, 1]) / rl
                        dd = max(0.0, dd) ** fog[5]
                        fr = fogc[0, 0] * (1 - dd) + fogc[1, 0] * dd; fg = fogc[0, 1] * (1 - dd) + fogc[1, 1] * dd; fb = fogc[0, 2] * (1 - dd) + fogc[1, 2] * dd
                        r += (fr - r) * k; g += (fg - g) * k; b += (fb - b) * k
                    rr += r; gg += g; bb += b; cc += 1
                    if z < zmin: zmin = z; idc = -2 if iswat else i; vv = val
            kk = 1. / (m * m)
            out[yy, xx, 0] = rr * kk; out[yy, xx, 1] = gg * kk; out[yy, xx, 2] = bb * kk; cov[yy, xx] = cc * kk; zout[yy, xx] = zmin; idout[yy, xx] = idc; vout[yy, xx] = vv


@njit(cache=True, fastmath=True)
def edges_of(F, FN, nv):
    """F (n,3) vertex ids; signed dihedral per triangle edge (edge j is opposite vertex j): >0 convex, <0 concave, 0 flat/none"""
    n = F.shape[0]
    return n


@njit(parallel=True, cache=True, fastmath=True)
def vnormals(T, FN, key, start, order, crease, VN):
    """angle-thresholded vertex normals.  key (n*3) vertex group id, start/order CSR of corner lists by group"""
    n = T.shape[0]
    for i in prange(n):
        for c in range(3):
            g = key[i * 3 + c]
            x = 0.; y = 0.; z = 0.
            for q in range(start[g], start[g + 1]):
                t = order[q] // 3
                if FN[t, 0] * FN[i, 0] + FN[t, 1] * FN[i, 1] + FN[t, 2] * FN[i, 2] > crease:
                    x += FN[t, 0]; y += FN[t, 1]; z += FN[t, 2]
            l = math.sqrt(x * x + y * y + z * z) + 1e-9
            VN[i, c, 0] = x / l; VN[i, c, 1] = y / l; VN[i, c, 2] = z / l
