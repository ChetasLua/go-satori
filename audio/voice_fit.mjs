// voice_fit.mjs : shapes area functions until their resonances land on a tenor's targets.  node audio/voice_fit.mjs -> audio/voice_vowels.mjs   (tract 17.0 cm)
import fs from 'node:fs';
const NSEC = 44, CS = 35300, RHO = .00114, RC = RHO * CS, LT = 17.0, PIRI = 4;
export function formants(A, Lt, n = 6, fmax = 6500) {
  const l = Lt / NSEC, w0 = 2 * Math.PI * 205, out = [];
  const f = fr => { const w = 2 * Math.PI * fr, s0 = Math.sqrt(Math.max(1e-6, 1 - (w0 / w) ** 2)), kk = w * s0 / CS, ga = 1 / s0, x = kk * l, c = Math.cos(x), s = Math.sin(x); let U = 1, p = w * 8 * RHO / (3 * Math.PI * Math.sqrt(Math.PI * Math.max(A[NSEC - 1], .02)));
    for (let i = NSEC - 1; i >= 0; i--) { const a = Math.max(A[i], .0015), p2 = c * p + RC * ga / a * s * U, u2 = -a / (RC * ga) * s * p + c * U; p = p2; U = u2; if (i === PIRI + 1) { U -= p * (.26 * Math.tan(kk * 1.6) + .22 * Math.tan(kk * 2.05)) / (RC * ga); } } return U; };
  let fa = 215, va = f(fa); for (let fb = 225; fb < fmax && out.length < n; fb += 10) { const vb = f(fb); if (va * vb < 0 && Math.abs(va) < 50 && Math.abs(vb) < 50) { let a = fa, b = fb, x = va; for (let i = 0; i < 22; i++) { const m = (a + b) / 2, vm = f(m); if (vm * x < 0) b = m; else { a = m; x = vm; } } if (Math.abs(f((a + b) / 2)) < 5) out.push((a + b) / 2); } fa = fb; va = vb; } return out;
}
const BASE = {
  i: [.45,.40,.38,.40,.68,.80,2.43,3.15,2.86,2.69,3.39,3.80,3.78,4.35,4.50,4.43,4.68,4.52,4.15,4.09,3.51,2.95,2.03,1.66,1.38,1.05,.70,.45,.36,.30,.28,.28,.30,.34,.42,.50,.65,.9,1.4,1.8,1.9,1.7],
  a: [.45,.40,.36,.36,.42,.55,.70,1.05,1.12,.95,.83,.69,.60,.55,.52,.50,.50,.56,.65,.85,1.2,1.45,1.82,2.19,2.66,2.98,3.2,3.5,3.9,4.5,5.1,5.6,6.0,6.2,6.1,5.8,5.3,4.8,4.4,4.3,4.4,4.5,4.6,4.7],
  u: [.45,.42,.38,.43,.60,1.4,2.4,2.6,2.5,2.6,3.2,4.0,4.2,4.0,3.7,3.4,3.0,2.6,2.3,2.0,1.8,1.6,1.4,1.1,.9,.75,.62,.55,.55,.60,.70,.85,1.05,1.3,1.6,2.0,2.3,2.5,2.5,2.2,1.8,1.4,1.1,.95],
  e: [.45,.38,.36,.38,.5,.85,1.35,1.5,1.5,1.5,1.6,1.8,2.1,2.3,2.6,2.80,2.95,3.03,2.9,2.84,2.75,2.46,2.24,2.05,1.88,1.7,1.55,1.4,1.3,1.25,1.3,1.5,1.8,2.1,2.3,2.4,2.5,2.6,2.5,2.2],
  o: [.45,.38,.36,.38,.59,1.2,1.4,1.2,1.0,1.1,1.5,1.7,1.6,1.4,1.25,1.15,1.05,.95,.9,.85,.8,.8,.8,.8,.75,.70,.70,.80,1.0,1.5,2.1,2.9,3.7,4.4,4.9,5.3,5.4,5.0,4.0,2.8,1.9,1.3,.95,.8],
};
export const T = {
  a: [[690, 1170, 2520, 3450, 4250], [760, 1230, 2560, 3350, 4150], [950, 1420, 2800, 3180, 3580]],
  i: [[300, 2180, 2880, 3480, 4250], [370, 2120, 2820, 3400, 4200], [520, 2050, 2820, 3220, 3640]],
  u: [[340, 1250, 2280, 3400, 4200], [400, 1250, 2320, 3320, 4150], [560, 1320, 2560, 3120, 3560]],
  e: [[490, 1890, 2560, 3480, 4250], [615, 1800, 2580, 3380, 4150], [850, 1780, 2750, 3180, 3600]],
  o: [[455, 800, 2520, 3400, 4200], [540, 880, 2560, 3300, 4100], [760, 1090, 2700, 3120, 3560]],
};
let KE = 0; const W = [3.4, 3, 2, 1.2, .5], NB = 16, LIPS = { a: [3.4, 4.2, 6.5], i: [1.6, 2.3, 3.2], u: [.75, 1.0, 1.4], e: [2.2, 3.0, 5.5], o: [.75, 1.05, 1.6] };
const resample = b => Array.from({ length: NSEC }, (_, i) => { const x = i * (b.length - 1) / (NSEC - 1), k = Math.min(b.length - 2, Math.floor(x)), u = x - k; return Math.exp(Math.log(b[k]) * (1 - u) + Math.log(b[k + 1]) * u); });
const soft = (a) => { const lo = .22, hi = 12; return a < .5 ? lo + (a - lo > 0 ? (a - lo) : 0) * 1 + (a < lo ? 0 : 0) : a > 6.5 ? Math.min(hi, 6.5 + (a - 6.5) * .5) : a; };
function shape(base, c) { return base.map((a, i) => { let s = 0; const x = (i + .5) / NSEC; for (let k = 0; k < NB; k++) s += c[k] * Math.cos(Math.PI * k * x); return soft(a * Math.exp(s)); }); }
function resid(base, c, tgt, lip) { const A = shape(base, c), f = formants(A, LT, 5, 5600), r = []; for (let i = 0; i < 5; i++) r.push(W[i] * (KE === 2 ? [1, 1, 1.2, 2, 2.6][i] : 1) * Math.log((f[i] || tgt[i] * 1.6) / tgt[i]) * 12); for (let k = 0; k < NB; k++) r.push(c[k] * (k < 1 ? .3 : .10 + .022 * k)); r.push(.9 * Math.log(A[NSEC - 1] / lip), .6 * Math.log(A[NSEC - 2] / lip), .3 * Math.log(A[NSEC - 3] / lip)); for (const q of [0, 1, 2, 3]) r.push(.5 * Math.log(A[q] / [.80, .62, .33][KE])); let mn = 9; for (let q = 14; q < NSEC; q++) mn = Math.min(mn, A[q]); r.push(mn < .3 ? (.3 - mn) * 8 : 0); return r; }
function solve(M, b) { const n = b.length; for (let i = 0; i < n; i++) { let p = i; for (let j = i + 1; j < n; j++) if (Math.abs(M[j][i]) > Math.abs(M[p][i])) p = j;[M[i], M[p]] = [M[p], M[i]];[b[i], b[p]] = [b[p], b[i]]; for (let j = i + 1; j < n; j++) { const q = M[j][i] / M[i][i]; for (let k = i; k < n; k++) M[j][k] -= q * M[i][k]; b[j] -= q * b[i]; } } for (let i = n - 1; i >= 0; i--) { for (let k = i + 1; k < n; k++) b[i] -= M[i][k] * b[k]; b[i] /= M[i][i]; } return b; }
function fit(base, tgt, c0, lip) {
  let c = c0 ? c0.slice() : new Array(NB).fill(0), lam = .05, r = resid(base, c, tgt, lip), E = r.reduce((s, x) => s + x * x, 0);
  for (let it = 0; it < 90; it++) { const J = []; for (let k = 0; k < NB; k++) { const d = c.slice(); d[k] += .01; const q = resid(base, d, tgt, lip); J.push(q.map((x, i) => (x - r[i]) / .01)); }
    const M = [], b = []; for (let i = 0; i < NB; i++) { M.push([]); for (let j = 0; j < NB; j++) { let s = 0; for (let m = 0; m < r.length; m++) s += J[i][m] * J[j][m]; M[i].push(s + (i === j ? lam * (s + .02) : 0)); } let s = 0; for (let m = 0; m < r.length; m++) s -= J[i][m] * r[m]; b.push(s); }
    const dc = solve(M, b), n = c.map((x, i) => x + Math.max(-.4, Math.min(.4, dc[i]))), q = resid(base, n, tgt, lip), e = q.reduce((s, x) => s + x * x, 0); if (e < E) { if (E - e < 1e-7) break; c = n; r = q; E = e; lam = Math.max(1e-4, lam * .5); } else { lam *= 3; if (lam > 1e5) break; } }
  return c;
}
if (process.argv[1].endsWith('voice_fit.mjs')) {
  let txt = '// voice_vowels.mjs : written by voice_fit.mjs.  44 areas (cm2, glottis to lips) per vowel and effort (0 soft, 1 chest, 2 belt), tract 17.0 cm.\nexport const VOWELS = {\n';
  for (const v of 'aiueo') { const base = resample(BASE[v]); let c = null; txt += ' ' + v + ': [\n'; for (let k = 0; k < 3; k++) { KE = k; c = fit(base, T[v][k], c, LIPS[v][k]); const A = shape(base, c), f = formants(A, LT, 5, 5600); console.log(v, k, T[v][k].join(' '), '->', f.map(x => x.toFixed(0)).join(' '), ' min', Math.min(...A).toFixed(2), 'max', Math.max(...A).toFixed(2), 'lips', A[43].toFixed(2)); txt += '  [' + A.map(x => +x.toFixed(3)).join(',') + '],\n'; } txt += ' ],\n'; }
  txt += '};\n'; fs.writeFileSync(new URL('./voice_vowels.mjs', import.meta.url), txt);
}
