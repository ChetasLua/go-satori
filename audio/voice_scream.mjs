// voice_scream.mjs : the break (f2 f60 .. f3 f20).  Lungs far past singing pressure, folds that stop agreeing with each other, false folds that join in,
// register flips, a breath that runs out.  Written straight onto the muscles.
import { slowNoise, rng, clamp, lerp, smooth, SR } from './voice_dsp.mjs'; import { sing, CR } from './voice_engine.mjs';
const hzm = m => 440 * Math.pow(2, (m - 69) / 12), pw = (T, t) => { let i = 0; while (i < T.length - 2 && t > T[i + 1][0]) i++; const [a, x] = T[i], [b, y] = T[i + 1]; return lerp(x, y, smooth((t - a) / (b - a))); };
// v: {Lt, fold, oct, seed, lag, hard(0..1)}   d: seconds of voice
export function scream(v, d = 1.667, o = {}) {
  const pre = .42, T = pre + d + .75, n = Math.ceil(T * CR), F = x => new Float32Array(n).fill(x), R = rng(v.seed * 31 + 7), N = [0, 1, 2, 3, 4, 5].map(k => slowNoise(v.seed * 97 + k * 13));
  const s = { n, f0: F(0), P: F(0), x01: F(0), x02: F(0), xm: F(0), phz: F(.9), tilt: F(9000), asp: F(1), rasp: F(0), jit: F(1), velum: F(0), eff: F(1), lips: F(0), vent: F(0), asy: F(0), w: [F(0), F(0), F(0), F(0), F(0)], gest: [] };
  // pitch: choked, torn open, the shove up, the octave, a crack into grief, sags, breaks, falls
  const PT = [[-.4, 60], [0, 58], [.05, 64], [.11, 71.4], [.28, 72.0], [.44, 75.5], [.56, 79.5], [.66, 82.8], [.80, 83.4], [.92, 82.6], [.98, 79.5], [1.06, 81.8], [1.20, 79.8], [1.34, 76.2], [1.46, 72], [1.56, 66], [1.66, 60], [2.2, 55]], k = d / 1.667;
  const flips = [[.30, .33, -7], [.545, .575, 5], [.715, .735, -5], [.93, .975, -8], [1.18, 1.205, 3.5], [1.40, 1.45, -5]].map(q => [q[0] + .018 * R.g(), q[1] + .018 * R.g(), q[2]]), crack = [.945 + .02 * R.g(), 1.0 + .02 * R.g()];
  for (let i = 0; i < n; i++) { const t = (i / CR - pre) / k, tt = i / CR; let m = pw(PT, t) + v.oct + .5 * N[0](tt, 4) + .38 * N[1](tt, 11) + .24 * N[2](tt, 26); let fl = 0; for (const [a, b, x] of flips) { const u = smooth((t - a) / .007) * (1 - smooth((t - b) / .010)); m += x * u * v.hard; fl = Math.max(fl, u); }
    const ck = smooth((t - crack[0]) / .012) * (1 - smooth((t - crack[1]) / .03)), on = smooth((t + .02) / .03), off = 1 - smooth((t - 1.52) / .22), strain = smooth((t - .10) / .45) * off, col = smooth((t - 1.30) / .32), choke = 1 - smooth((t - .03) / .07);
    s.f0[i] = hzm(m); s.P[i] = 980 * (on * off * (30 + 20 * smooth(t / .4) + 16 * smooth((t - .4) / .4) - 30 * col) * (1 + .16 * N[3](tt, 7)) + 9 * smooth((t - 1.4) / .1) * (1 - smooth((t - 1.75) / .35)));
    s.x01[i] = s.x02[i] = lerp(.2, -.008 + .010 * (1 - smooth((t - .4) / .3)) - .055 * choke - .012 * strain + .016 * fl + .06 * ck + .11 * col + .012 * N[4](tt, 8), smooth((t + .10) / .06)); s.xm[i] = on * (.11 + .025 * strain - .055 * col) * (1 - .6 * choke) * Math.pow(220 * (v.fk ?? 1) / Math.max(120, s.f0[i]), .25) * (1 - .3 * fl) * (1 - .75 * ck) * (1 - smooth((t - 1.54) / .12));
    s.rasp[i] = v.hard * (.45 * choke + .14 + .55 * strain * (.45 + .7 * Math.abs(N[5](tt, 5))) + .5 * fl + .9 * col * off); s.vent[i] = v.vent * (.35 + .65 * smooth((t - .10) / .35)) * (1 - .55 * smooth((t - .9) / .3)) * off * (.45 + .8 * Math.abs(N[1](tt + 7, 4))); s.asy[i] = t > .3 ? v.asy * smooth((t - .3) / .3) * Math.max(0, .35 + N[2](tt + 3, 2.6)) * off : 0;
    s.asp[i] = .4 + .45 * strain + .5 * (1 - smooth((t - .45) / .25)) * on + .55 * col + .6 * ck; s.jit[i] = 2.5 + 4 * strain; s.phz[i] = lerp(1.0, .5, smooth((t - .3) / .5)); s.tilt[i] = 7800 + 4200 * strain - 5200 * col;
    const e = .55 * (1 - smooth((t - .02) / .13)) + .16 * smooth((t - .5) / .2) * (1 - smooth((t - .9) / .1)), back = .75 * smooth((t - .93) / .28), a = Math.max(0, 1 - e - back); s.w[0][i] = a; s.w[3][i] = e; s.w[4][i] = back * .8; s.w[2][i] = back * .2 + .3 * col * 0; s.eff[i] = clamp(.8 + 1.0 * smooth((t - .04) / .25) + .16 * N[0](tt + 5, 3) - .25 * back - .8 * col, 0, 1.98); s.lips[i] = .5 * strain * (1 - back);
    if (t < -.06) { const u = clamp((i / CR) / (pre - .08), 0, 1); s.P[i] = -980 * 6.5 * Math.pow(Math.max(0, Math.sin(Math.PI * Math.pow(u, 1.9))), .8) * (1 + .25 * N[3](tt, 22)); s.x01[i] = s.x02[i] = .16; s.xm[i] = u > .45 ? .022 : 0; s.f0[i] = 190 + 60 * u; s.asp[i] = .35; s.w[0][i] = .6; s.w[3][i] = .4; s.w[4][i] = 0; s.w[2][i] = 0; } }
  for (const q of ['x01', 'x02', 'P']) { let y = s[q][0]; const g = 1 - Math.exp(-1 / (CR * .005)); for (let i = 0; i < n; i++) { y += (s[q][i] - y) * g; s[q][i] = y; } }
  const r = sing(s, { seed: v.seed, Lt: v.Lt, fold: v.fold, areaScale: v.areaScale ?? 1, soft: .006, air: .0016, hfe: .06, flutter: 2, ventRatio: v.vr ?? .5, loss: .0055, chaos: v.chaos ?? 1, ...(o.eng || {}) });
  // the throat itself overloads: gentle asymmetric folding at the loudest
  let pk = 0; for (const x of r.y) pk = Math.max(pk, Math.abs(x)); const dr = (v.drive ?? 3) / pk; for (let i = 0; i < r.y.length; i++) { const x = r.y[i] * dr + .12; r.y[i] = (Math.tanh(x) - Math.tanh(.12)) / dr * 1.4; } return { y: r.y, pre };
}
export const SCREAMERS = [
  { Lt: 18.0, fold: 1.45, oct: 0, seed: 11, lag: 0, hard: .8, vent: .55, asy: .006, vr: .5, pan: 0, g: 1, fk: 1, chaos: .10, drive: 2.4 },
  { Lt: 19.0, fold: 1.6, oct: -12, seed: 23, lag: .012, hard: 1.2, vent: 1, asy: .012, vr: .5, pan: -.4, g: .50, fk: .8, chaos: .5, drive: 3.2 },
  { Lt: 14.4, fold: 1.1, areaScale: .75, oct: 0, seed: 37, lag: .03, hard: .7, vent: .4, asy: .003, vr: .5, pan: .5, g: .55, fk: 1.5, chaos: .06, drive: 1.8 },
  { Lt: 17.2, fold: 1.4, oct: -5, seed: 53, lag: -.012, hard: 1.1, vent: .95, asy: .02, vr: .3334, pan: .2, g: .42, fk: 1, chaos: .4, drive: 2.8 },
];
