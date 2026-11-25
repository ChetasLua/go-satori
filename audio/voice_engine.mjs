// voice_engine.mjs : a throat in the time domain.  Lungs -> windpipe -> two vibrating folds -> 44 tubes with yielding walls, piriform pockets,
// a nose with a sinus -> lips.  Folds and air are coupled; hiss, bursts, breath and devoicing are born where the air jet is fast.  Zero packages.
import { SR, rng, slowNoise, clamp, lerp, smooth } from './voice_dsp.mjs';
import { VOWELS } from './voice_vowels.mjs';
export const NSEC = 44, CR = 500, RHO = .00114, C0 = 35300, RC = RHO * C0, MU = 1.86e-4, KT = 1.1;
export const VI = { a: 0, i: 1, u: 2, e: 3, o: 4 };
const DV = ['a', 'i', 'u', 'e', 'o'].map(v => VOWELS[v].map(A => Float64Array.from(A, Math.sqrt)));
const NASAL = [0, .7, 1.1, 1.8, 2.6, 3.2, 3.5, 3.4, 3.1, 2.8, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.25, 1.1, .95, .85, .8, .9];
const PIRI = 4, VELUM = 22, WALLJ = [7, 14, 20, 28, 34, 40], NW = WALLJ.length, NN = NASAL.length, SUB = 34;

// sqrt-area of the tract at control frame i
export function areaOf(s, i, D) {
  const e = clamp(s.eff[i], 0, 1.999), k = e | 0, u = e - k; D.fill(0);
  for (let v = 0; v < 5; v++) { const w = s.w[v][i]; if (w < 1e-4 && w > -1e-4) continue; const a = DV[v][k], b = DV[v][k + 1]; for (let q = 0; q < NSEC; q++) D[q] += w * (a[q] * (1 - u) + b[q] * u); }
  if (s.lips) { const l = s.lips[i]; if (l !== 0) for (let q = 39; q < NSEC; q++) D[q] *= 1 + l * (q - 38) / 5; }
  for (const g of s.gest) { const m = g.m[i]; if (m < 1e-4) continue; const lo = Math.max(0, Math.ceil(g.loc - g.hw * 2.2)), hi = Math.min(NSEC - 1, Math.floor(g.loc + g.hw * 2.2)), dm = Math.sqrt(g.amin); for (let q = lo; q <= hi; q++) { const x = (q - g.loc) / g.hw, kk = m * Math.exp(-.693 * x * x * (Math.abs(x) < 1 ? x * x * .6 + .4 : 1)), c = Math.min(1, kk); if (D[q] > dm) D[q] = D[q] * (1 - c) + dm * c; } }
  return D;
}
// resample from fs to SR with a windowed sinc (cut a little under 24 kHz)
export function resample(x, n, fs) {
  const ratio = fs / SR, M = Math.ceil(n / ratio), y = new Float32Array(M), half = 22, W = Math.ceil(half * ratio), PH = 256, fc = .94 / ratio, tab = [];
  for (let p = 0; p <= PH; p++) { const t = new Float64Array(2 * W); let s = 0; for (let k = 0; k < 2 * W; k++) { const d = (k - W + 1) - p / PH, z = Math.PI * d * fc, wn = .5 + .5 * Math.cos(Math.PI * clamp(d / W, -1, 1)); t[k] = (Math.abs(z) < 1e-9 ? 1 : Math.sin(z) / z) * wn; s += t[k]; } for (let k = 0; k < 2 * W; k++) t[k] /= s; tab.push(t); }
  for (let i = 0; i < M; i++) { const c = i * ratio, j = Math.floor(c), p = Math.round((c - j) * PH), t = tab[p]; let a = 0; for (let k = 0; k < 2 * W; k++) { const q = j + k - W + 1; if (q >= 0 && q < n) a += x[q] * t[k]; } y[i] = a; }
  return y;
}
// s: control arrays at CR.  o: the body (Lt, fold length, seed ...)
export function sing(s, o = {}) {
  const Lt = o.Lt ?? 17, dx = Lt / NSEC, fs = C0 / dx, dt = 1 / fs, nT = Math.floor(s.n / CR * fs), out = new Float32Array(nT + 8), seed = o.seed ?? 1;
  const R = rng(seed * 7 + 1), fl = [0, 1, 2, 3, 4, 5, 6, 7].map(k => slowNoise(seed * 131 + k * 17 + 5)), flut = o.flutter ?? 1;
  let xs = (seed * 2654435761) >>> 0 || 7; const nz = () => { xs ^= xs << 13; xs >>>= 0; xs ^= xs >>> 17; xs ^= xs << 5; xs >>>= 0; return xs / 2147483648 - 1; };
  const f = new Float64Array(NSEC + 1), b = new Float64Array(NSEC + 1), A = new Float64Array(NSEC).fill(1), A1 = new Float64Array(NSEC).fill(1), dA = new Float64Array(NSEC), r = new Float64Array(NSEC), ls = new Float64Array(NSEC).fill(.998), jt = new Uint8Array(NSEC), D = new Float64Array(NSEC);
  const asc = o.areaScale ?? 1;
  // nose
  const nA = Float64Array.from(NASAL, a => a * (o.noseScale ?? 1)), nf = new Float64Array(NN), nb = new Float64Array(NN), nr = new Float64Array(NN), nl = new Float64Array(NN); nA[0] = .02;
  const snM = .0057, snK = snM * (2 * Math.PI * 540) ** 2, snR = snM * 2 * Math.PI * 540 / 4; let snu = 0, snx = 0;
  const setNose = () => { nA[1] = .5 * (nA[0] + .9); for (let i = 0; i < NN - 1; i++) nr[i] = (nA[i] - nA[i + 1]) / (nA[i] + nA[i + 1]); for (let i = 0; i < NN; i++) nl[i] = 1 - Math.min(.05, .011 * dx / Math.sqrt(Math.max(.05, nA[i]))); }; setNose();
  // piriform pockets: two closed stubs
  const p1 = new Float64Array(Math.max(2, Math.round(2 * 1.6 * (Lt / 17) / dx))), p2 = new Float64Array(Math.max(2, Math.round(2 * 2.05 * (Lt / 17) / dx))), pA1 = .26 * (o.piri ?? 1), pA2 = .22 * (o.piri ?? 1); let pi1 = 0, pi2 = 0;
  // walls
  const wu = new Float64Array(NW), wx = new Float64Array(NW), wM = new Float64Array(NW), wR = new Float64Array(NW), wK = new Float64Array(NW), wIdx = new Int32Array(NSEC).fill(-1); WALLJ.forEach((j, k) => { wIdx[j] = k; jt[j] = 3; }); jt[PIRI] = 1; jt[VELUM] = 2;
  const base = Uint8Array.from(jt);
  // windpipe
  const As = 2.5 * (o.bodyScale ?? 1), sub = new Float64Array(SUB * 2); let si = 0, sLP = 0;
  // folds
  const Lf = o.fold ?? 1.5, NY = 12, NZ = 4, sy = new Float64Array(NY), yy = new Float64Array(NY); for (let k = 0; k < NY; k++) { yy[k] = (k + .5) / NY; sy[k] = Math.sin(Math.PI * yy[k]); } const dy = Lf / NY, cz = new Float64Array(NZ), szz = new Float64Array(NZ);
  let th = 0, thR = 0, thD = 0, jitLP = 0, shLP = 0, jit = 0, sh = 1, subh = 0, np = 0, vib = 0, ptLP = 0, noiF = 0, vent = 0, vph = 0, hy1 = 0, hy2 = 0;
  // noise state
  const NC = 3, cj = new Int32Array(NC), cK = new Float64Array(NC), cL = new Float64Array(NC), cH = new Float64Array(NC), clp = new Float64Array(NC), chp = new Float64Array(NC), cG = new Float64Array(NC), cP = new Uint8Array(NC), clp2 = new Float64Array(NC), cW = new Float64Array(NC), cA = new Float64Array(NC), cD = new Int32Array(NC); let nC = 0;
  let fl1 = 0, br1 = 0, nfl1 = 0, nbr1 = 0, uL0 = 0, uN0 = 0, wSum0 = 0, wLP = 0, ag1 = 0, agS = 0, aspLP = 0, aspHP = 0, uS = 0;
  const lipR = 128 / (9 * Math.PI * Math.PI); let lk = 1, nk = 1;
  const rad = a => 2 * fs * 8 * Math.sqrt(a / Math.PI) / (3 * Math.PI * C0);
  nk = rad(.9 * (o.noseScale ?? 1));
  const BLK = 3; const cha = o.chaos ?? 0; let vRat = o.ventRatio ?? .5; let bn_ = 0, x0j = 0, tlj = 1, shS = 1, x0S = 0, tlS = 1; let cf0 = -1; const Rec = 1700, rcl = Math.PI * MU * MU / (4 * RHO * RHO), K_f = (o.fricK ?? 1) * 9e-4;
  const dbg = o.dbg ? { ag: new Float32Array(nT), u: new Float32Array(nT) } : null;
  const aLP = 1 - Math.exp(-2 * Math.PI * (o.aspLp ?? 5500) / fs), aHP = 1 - Math.exp(-2 * Math.PI * (o.aspHp ?? 900) / fs); const airK = o.air ?? 0, airA = 1 - Math.exp(-2 * Math.PI * 6500 / fs), airB = 1 - Math.exp(-2 * Math.PI * 2600 / fs); let airLP = 0, airHP = 0, air2 = 0; const hfe = o.hfe ?? .22, hb = new Float64Array(12), hf = new Float64Array(12); const aLP2 = 1 - Math.exp(-2 * Math.PI * (o.aspLp2 ?? 4800) / fs); let asp2 = 0; const aFl = o.hissFloor ?? .5, uMk = 1 - Math.exp(-2 * Math.PI * 40 / fs); let uM = 0; const gT = o.gT ?? .3; let vg = 0, tiltK = 1, u1 = 0, u2 = 0, P = 0, f0 = 150, x01 = .02, x02 = .02, xmT = 0, phz = 1.1, asp = 1, rsp = 0, jb = 1, velum = 0, fgain = 1, asy = 0, gate = 0, ventA = 0;
  const cpn = new Float64Array(NC); let nU = 0;
  const hiss = (k, U, a) => { const re2 = U * U / (a * rcl) * .25 * 4 / 1, dp = RHO * U * U / (2 * a * a); if (re2 < Rec * Rec) return; const w = 1 - Rec * Rec / re2, e = K_f * cG[k] * Math.pow(dp, 1.5) * Math.sqrt(a) * w; clp[k] += (nz() - clp[k]) * cL[k]; let z = clp[k]; if (cP[k] > 1) { clp2[k] += (z - clp2[k]) * cL[k]; z = clp2[k] * 1.6; } chp[k] = cH[k] * chp[k] + (1 - cH[k]) * z; cpn[k] = Math.min(e, 40000) * (z - chp[k]); };
  const nose = fin => { let fi = nf[0]; nf[0] = fin * nl[0]; for (let i = 0; i < NN - 1; i++) { const bi = nb[i + 1]; let F, B; if (i === 14) { const pj = (2 * nA[i] * fi + 2 * nA[i + 1] * bi - RC * snu) / (nA[i] + nA[i + 1]); snu += dt / snM * (pj - snR * snu - snK * snx); snx += snu * dt; F = pj - bi; B = pj - fi; } else { const w = nr[i] * (fi - bi); F = fi + w; B = bi + w; } nb[i] = B * nl[i]; fi = nf[i + 1]; nf[i + 1] = F * nl[i + 1]; }
    const a = nA[NN - 1], k = nk, den = (lipR + 1) * k + lipR, br = (((lipR - 1) * k - lipR) * fi - ((lipR - 1) * k + lipR) * nfl1 - (lipR - (lipR + 1) * k) * nbr1) / den; nfl1 = fi; nbr1 = br; nb[NN - 1] = br * nl[NN - 1]; nU = a / RC * (fi - br); };
  const at = (a, c, u) => a[c] * (1 - u) + a[c + 1] * u;
  for (let n = 0; n < nT; n++) {
    if (n % BLK === 0) {
      const ct = n * dt * CR, c = Math.min(s.n - 2, ct | 0), u = ct - c;
      if (c !== cf0) {                    // new control frame: area at its end
        const fresh = cf0 < 0; cf0 = c; areaOf(s, c + 1, D); const t = (c + 1) / CR, f1 = (fl[0](t, 2.3) * .03 + fl[3](t, 6) * .012) * flut, f2 = (fl[1](t, 3.1) * .03 + fl[4](t, 7) * .012) * flut, f3 = (fl[2](t, 1.7) * .03 + fl[5](t, 5) * .012) * flut;
        for (let q = 0; q < NSEC; q++) { const x = q / (NSEC - 1), d = D[q] * (1 + f1 * Math.cos(Math.PI * x) + f2 * Math.cos(2 * Math.PI * x) + f3 * Math.cos(3 * Math.PI * x)); A1[q] = Math.max(.0006, d * d * asc); if (fresh) A[q] = A1[q]; dA[q] = (A1[q] - A[q]) / Math.max(1, fs / CR / BLK); }
        WALLJ.forEach((j, k) => { const S = 2 * Math.sqrt(Math.PI * Math.max(.3, A1[j])) * 1.25 * (NSEC / NW) * dx; wM[k] = 1.6 / S; wR[k] = 1400 / S; wK[k] = 30000 / S; });
        // narrow places that can hiss: local minima below .42 cm2 in the mouth
        for (let q = 0; q < NSEC; q++) jt[q] = base[q]; nC = 0;
        const put = (q, g, w) => { const jq = Math.min(q, NSEC - 2); let k = -1; for (let z = 0; z < nC; z++) if (Math.abs(cj[z] - q) < 2) k = z; if (k >= 0) { if (g && (g.ng ?? 1) > cG[k]) { cG[k] = g.ng ?? 1; cP[k] = g.np ?? 1; cD[k] = Math.min(NSEC - 1, q + (g.nd ?? 1)); cL[k] = 1 - Math.exp(-2 * Math.PI * (g.nf ?? 3500) / fs); cH[k] = Math.exp(-2 * Math.PI * (g.nh ?? 350) / fs); } return; } if (nC >= NC || jt[jq] === 1 || jt[jq] === 2) return; jt[jq] = 10 + nC; cj[nC] = q; cW[nC] = w; cG[nC] = g ? g.ng ?? 1 : .4; cP[nC] = g ? g.np ?? 1 : 1; cD[nC] = Math.min(NSEC - 1, q + (g ? g.nd ?? 1 : 1)); cL[nC] = 1 - Math.exp(-2 * Math.PI * (g ? g.nf ?? 3500 : 2500) / fs); cH[nC] = Math.exp(-2 * Math.PI * (g ? g.nh ?? 350 : 350) / fs); nC++; };
        for (const g of s.gest) if (g.m[c] > .03 || g.m[c + 1] > .03) put(Math.min(NSEC - 1, Math.round(g.loc)), g, 1);
        { let bq = -1; for (let q = 16; q < NSEC - 1; q++) { const a = A1[q]; if (a < .5 && a <= A1[q - 1] && a < A1[q + 1] && (bq < 0 || a < A1[bq])) bq = q; } if (bq >= 0) put(bq, null, 1); }
      }
      for (let q = 0; q < NSEC; q++) { A[q] += dA[q]; if (A[q] < .0006) A[q] = .0006; }
      for (let k = 0; k < nC; k++) { let m = 9; for (let q = Math.max(0, cj[k] - 2); q <= Math.min(NSEC - 1, cj[k] + 2); q++) if (A[q] < m) m = A[q]; cA[k] = m; }
      for (let q = 0; q < NSEC - 1; q++) r[q] = (A[q] - A[q + 1]) / (A[q] + A[q + 1]);
      if ((bn_++ & 3) === 0) for (let q = 0; q < NSEC; q++) ls[q] = 1 - Math.min(.04, (o.loss ?? .0042) * dx / Math.sqrt(Math.max(.05, A[q])));
      lk = rad(Math.max(.02, A[NSEC - 1]));
      velum = at(s.velum, c, u); const vn = velum < .004 ? 1e-6 : velum; if (Math.abs(vn - nA[0]) > 1e-4 || (vn < 1e-5 && nA[0] > 1e-5)) { nA[0] = vn; setNose(); }
      P = at(s.P, c, u); f0 = at(s.f0, c, u); x01 = at(s.x01, c, u); x02 = at(s.x02, c, u); xmT = at(s.xm, c, u); phz = at(s.phz, c, u); asp = at(s.asp, c, u); rsp = at(s.rasp, c, u); jb = at(s.jit, c, u); asy = s.asy ? at(s.asy, c, u) : 0; ventA = s.vent ? at(s.vent, c, u) : 0; tiltK = s.tilt ? 1 - Math.exp(-2 * Math.PI * at(s.tilt, c, u) / fs) : 1;
      for (let z = 0; z < NZ; z++) { const p = phz * z / (NZ - 1); cz[z] = Math.cos(p); szz[z] = Math.sin(p); }
    }
    // ---- folds
    th += 2 * Math.PI * f0 * (1 + jit) * dt; thD += 2 * Math.PI * f0 * asy * dt; thR = th + 1.35 * Math.sin(thD);
    if (th >= 2 * Math.PI) { th -= 2 * Math.PI; np++; jitLP = jitLP * .6 + R.g() * .8; shLP = shLP * .55 + R.g() * .83; const t = n * dt; const sb = np & 1 ? 1 : -1;
      jit = jb * ((o.jitW ?? .0016) * jitLP + (o.jitS ?? .0018) * (fl[3](t, 11) + .8 * fl[4](t, 21))) + rsp * (.010 * sb + .012 * R.g()) + cha * rsp * (.05 * R.g() + (R.u() < .12 ? -.16 : 0)); jit = clamp(jit, -.28, .2);
      x0j = jb * (o.x0j ?? .004) * (R.g() * .6 + fl[7](t, 13)); tlj = Math.exp((o.tlj ?? .16) * jb * (R.g() * .7 + fl[6](t + 3, 9))); sh = 1 + jb * (.012 * shLP + .030 * fl[5](t, 9) + .022 * fl[6](t, 17)) + rsp * (.15 * sb + .12 * R.g()) + cha * rsp * .22 * R.g(); sh = clamp(sh, .15, 1.8); if (ventA > .01 && R.u() < .10 * (1 + cha)) vRat = R.u() < .55 ? .5 : R.u() < .5 ? .3334 : .6667;
      }
    const psi0 = sub[si];                  // arriving from the lungs
    const pe = b[0];
    ptLP += ((2 * psi0 - 2 * pe) - ptLP) * (1 - Math.exp(-dt / .004));
    const q = clamp(ptLP / (.45 * Math.max(P, 1500)), 0, 1.15), want = xmT * Math.sqrt(q) * (ptLP > 1200 ? 1 : Math.max(0, ptLP / 1200));
    vib += (want - vib) * dt / (want > vib ? .012 : .009);
    const sL = Math.sin(th), cLf = Math.cos(th), sR_ = asy ? Math.sin(thR) : sL, cR_ = asy ? Math.cos(thR) : cLf, kS1 = dt / .0016; shS += (sh - shS) * kS1; x0S += (x0j - x0S) * kS1; tlS += (tlj - tlS) * kS1; const xm = vib * shS; let ag = 1e9; const wS = (o.soft ?? .012) * Lf / 1.45;
    for (let z = 0; z < NZ; z++) {
      const zz = z / (NZ - 1), x0 = x0S + x01 + (x02 - x01) * zz - .012 * 4 * zz * (1 - zz), osc = .5 * xm * ((sL * cz[z] - cLf * szz[z]) + (sR_ * cz[z] - cR_ * szz[z])) * (1 - .1 * zz); let a = 0;
      for (let k = 0; k < NY; k++) { const h = x0 * (.35 + 1.0 * yy[k]) + .0015 + osc * sy[k]; a += h <= -wS ? 0 : h >= wS ? h : (h + wS) * (h + wS) / (4 * wS); }
      if (a < ag) ag = a;
    }
    ag = 2 * ag * dy;
    if (ventA > .01) { vph += f0 * (cha ? vRat : o.ventRatio ?? .5) * dt; if (vph > 1) { vph -= 1; vent = .6 + .6 * R.u(); } ag *= 1 - ventA * vent * Math.pow(Math.max(0, Math.sin(Math.PI * 2 * vph)), .7) * .85; }
    const tk = tiltK * tlS > 1 ? 1 : tiltK * tlS; u1 += (ag - u1) * tk; u2 += (u1 - u2) * tk; ag = u2;
    const Ae = A[0], Ast = As * Ae / (As + Ae), dl = psi0 - pe; let ug = 0;
    { const m0 = RHO * gT * fs, q2 = KT * RHO / 2, q1 = m0 + RC * ag / Ast, q0 = 2 * dl + m0 * vg; const v = q0 >= 0 ? 2 * q0 / (q1 + Math.sqrt(q1 * q1 + 4 * q2 * q0)) : 2 * q0 / (q1 + Math.sqrt(q1 * q1 - 4 * q2 * q0)); vg = ag > 1e-5 ? v : 0; ug = ag * vg; }
    const uo = ug;
    // hiss of the jet through the glottis
    let un = 0; uM += (Math.abs(ug) - uM) * uMk; { const x = (1 - aFl) * (Math.abs(ug) - 30) + aFl * .8 * uM; if (x > 0) { aspLP += (nz() - aspLP) * aLP; asp2 += (aspLP - asp2) * aLP2; aspHP += (asp2 - aspHP) * aHP; un = asp * .062 * x * (asp2 - aspHP); } }
    if (dbg) { dbg.ag[n] = ag; dbg.u[n] = ug; }
    const psout = psi0 - RC / As * uo, f_in = pe + RC / Ae * (uo + un + (o.click && n === Math.round(.1 * fs) ? o.click : 0));
    // lungs
    sLP += (psout - sLP) * .45; sub[si] = .9 * (-.55 * sLP) + .775 * P * 1; sub[si] = -.5 * sLP + .75 * P; si = si + 1; if (si >= sub.length) si = 0;
    // ---- tubes
    let fi = f[0]; const b0 = b[0]; f[0] = f_in * ls[0];
    for (let i = 0; i < NSEC - 1; i++) {
      const bi = b[i + 1]; let F, B; const j = jt[i];
      if (j === 0) { const w = r[i] * (fi - bi); F = fi + w; B = bi + w; }
      else if (j === 3) { const k = wIdx[i], As2 = A[i] + A[i + 1], Zp = RC / As2, p0 = 2 * (A[i] * fi + A[i + 1] * bi) / As2, m = dt / wM[k], u = (wu[k] + m * (p0 - wK[k] * wx[k])) / (1 + m * (wR[k] + Zp)), pj = p0 - Zp * u; wu[k] = u; wx[k] += u * dt; F = pj - bi; B = pj - fi; }
      else if (j === 1) { const i1 = p1[pi1], i2 = p2[pi2], pj = 2 * (A[i] * fi + A[i + 1] * bi + pA1 * i1 + pA2 * i2) / (A[i] + A[i + 1] + pA1 + pA2); p1[pi1] = (pj - i1) * .90; p2[pi2] = (pj - i2) * .90; pi1 = (pi1 + 1) % p1.length; pi2 = (pi2 + 1) % p2.length; F = pj - bi; B = pj - fi; }
      else if (j === 2) { const an = nA[0], bn = nb[0], pj = 2 * (A[i] * fi + A[i + 1] * bi + an * bn) / (A[i] + A[i + 1] + an); F = pj - bi; B = pj - fi; nose(pj - bn); }
      else { // jet
        const k = j - 10, aa = cA[k], fade = aa > .5 ? 0 : aa < .2 ? 1 : (.5 - aa) / .3, Z1 = RC / A[i], Z2 = RC / A[i + 1], d = 2 * (fi - bi), kk = fade * RHO / (2 * aa * aa) + 1e-9, z = Z1 + Z2; const U = 2 * Math.abs(d) / (z + Math.sqrt(z * z + 4 * kk * Math.abs(d))) * (d < 0 ? -1 : 1); B = fi - Z1 * U; F = bi + Z2 * U; hiss(k, U, aa); }
      if (hfe > 0 && (i & 3) === 1) { const q = i >> 2, B0 = B, F0 = F; B = B * (1 - hfe) + hb[q] * hfe; hb[q] = B0; F = F * (1 - hfe) + hf[q] * hfe; hf[q] = F0; }
      b[i] = B * ls[i]; fi = f[i + 1]; f[i + 1] = F * ls[i + 1];
    }
    // lips
    { const a = A[NSEC - 1], k = lk, den = (lipR + 1) * k + lipR; const br = (((lipR - 1) * k - lipR) * fi - ((lipR - 1) * k + lipR) * fl1 - (lipR - (lipR + 1) * k) * br1) / den; fl1 = fi; br1 = br; b[NSEC - 1] = br * ls[NSEC - 1]; let uL = a / RC * (fi - br);
      let ws = 0; for (let k2 = 0; k2 < NW; k2++) ws += wu[k2]; wLP += (ws - wLP) * .04;
      out[n] = (uL - uL0) + (nU - uN0) * (o.noseGain ?? .85) + (wLP - wSum0) * (o.wallK ?? .10); if (airK > 0) { airLP += (nz() - airLP) * airA; airHP += (airLP - airHP) * airB; air2 += (airLP - airHP - air2) * airB; const op = a < .05 ? 0 : a > .5 ? 1 : (a - .05) / .45; out[n] += airK * op * Math.max(0, (1 - aFl) * (ug - 20) + aFl * .8 * uM) * ((airLP - airHP) - air2); } uL0 = uL; uN0 = nU; wSum0 = wLP; }
    // turbulence enters at its place
    for (let k = 0; k < nC; k++) if (cpn[k] !== 0) { const j = cD[k]; f[j] += .5 * cpn[k]; if (j > 0) b[j] -= .5 * cpn[k]; cpn[k] = 0; }
  }
  const y = resample(out, nT, fs); if (s.gain) for (let i = 0; i < y.length; i++) { const x = i / SR * CR, k = Math.min(s.n - 2, x | 0); y[i] *= s.gain[k] * (1 - x + k) + s.gain[k + 1] * (x - k); } return { y, dbg, fs };
}
