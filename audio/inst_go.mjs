// inst_go.mjs : the instruments that define GO.  Star bells, kalimba, blown lines (shinobue, flute, whistle, clarinet, kazoo), brass sections, choirs, ukulele, toys.  No samples.
import { lerp, SR, TAU, mk, st, rng, bqc, filt, filtS, noise, expEnv, modal, fade, gain, gainS, add, addS, clamp, db, mtof, envAt, bodyIR, convM, convolve2, pan, SVF, peak, norm } from './inst_core.mjs';
import { pluck, IR } from './inst_pluck.mjs';
const S = t => Math.floor(t * SR);
// ------------------------------------------------------------ star bell: a glass sphere with a light in it
// {t,n,v,p (pan),glint (0..1), bloom (0..1), dec}
export function starBell(evs, { n, seed = 'star' }) {
  const R = rng(seed), o = st(n);
  for (const e of evs) {
    const f = mtof(e.n), dec = e.dec ?? 1, L = S(5.2 * dec), x = st(L), p = e.p ?? 0, a0 = S(Math.max(0, e.t));
    // struck glass: near-harmonic pairs that beat slowly, upper modes of a thick shell
    const M = [[1, 1, 4.6], [1.0007, .42, 4.2], [2.0, .16, 2.6], [2.002, .07, 2.2], [2.97, .2, 1.5], [4.02, .13, 1.05], [5.43, .12, .62], [6.9, .05, .42], [8.9, .06, .28], [12.4, .03, .14]];
    for (const [r, a, d] of M) { if (f * r > 15000) continue; const th = clamp(Math.PI / 4 + .24 * R.g(), .2, 1.37), q = { f: f * r * (1 + (r > 2.5 ? .0009 * R.g() : 0)), a, t60: d * dec * Math.pow(800 / f, .25), rise: .0016 }; modal(x.L, [{ ...q, a: a * Math.cos(th) * 1.4142, ph: R.r(-.3, .3) }]); modal(x.R, [{ ...q, a: a * Math.sin(th) * 1.4142, ph: R.r(-.3, .3) }]); }
    // soft felt and the tick of glass
    const tk = noise(S(.03), R); filt(tk, bqc.bp(5200, 1.1)); expEnv(tk, .0022, .0002); for (const c of [x.L, x.R]) add(c, tk, .16 * R.r(.8, 1.2)); const fm = []; for (const q of [f * 4.8, f * 7.3, 3900]) fm.push({ f: Math.min(q, 12000), a: .05, t60: .03 }); modal(x.L, fm); modal(x.R, fm);
    // the light blooming under it: an octave and a twelfth below, swelling late
    const bl = (e.bloom ?? .5) * .2; if (bl > 0) for (const [r, a] of [[.5, 1], [.75, .28], [1, .5]]) { let ph = R.u() * 6; const w = TAU * f * r / SR, trem = R.r(3.1, 4.6); for (let i = 0; i < L; i++) { const t = i / SR, k = Math.sin(Math.PI * Math.pow(Math.min(1, t / (2.6 * dec)), .6)); ph += w; const s = Math.sin(ph) * a * bl * k * k * (1 + .1 * Math.sin(TAU * trem * t)); x.L[i] += s * .9; x.R[i] += s; } }
    // the caustic: a handful of high glints that travel across
    const gl = e.glint ?? .6; if (gl > 0) for (let k = 0; k < 11; k++) { const u = k / 10, fr = f * [4, 5, 6, 8, 9, 10, 12, 8, 6, 5, 9][k] * .5 * (1 + .003 * R.g()); if (fr > 13500 || fr < 1200) continue; const g = mk(S(.5)); modal(g, [{ f: fr, a: 1, t60: R.r(.18, .4), rise: .006 }, { f: fr * 2.76, a: .18, t60: .08 }]); fade(g, .002, .2); pan(x, g, clamp(lerp(-.9, .9, u) * (e.dir ?? 1), -1, 1), gl * .05 * Math.sin(Math.PI * (.15 + .8 * u)), S(.07 + .42 * u + .012 * R.u())); }
    for (const c of [x.L, x.R]) fade(c, .0004, 1.2 * dec);
    const g = e.v * .2, cl = Math.cos((p + 1) * Math.PI / 4) * 1.4142, cr = Math.sin((p + 1) * Math.PI / 4) * 1.4142; add(o.L, x.L, g * cl, a0); add(o.R, x.R, g * cr, a0);
  }
  return o;
}
// ------------------------------------------------------------ kalimba: plucked tine on a small box
export function kalimba(evs, { n, seed = 'kal' }) {
  const R = rng(seed), o = st(n), ir = bodyIR('kal', { len: .1, f0: 160, f1: 5000, n: 70, env: [[160, -2], [300, 4], [520, 3], [1200, 0], [2500, -4], [5000, -14]], t0: .06, t1: .006, strong: [[232, 4, .06], [410, 3, .05]], direct: 1.7 });
  for (const e of evs) { const f = mtof(e.n + .02 * R.g()), v = e.v * R.r(.88, 1.06), x = mk(S(2.4)), t = 1.5 * Math.pow(440 / f, .5) * R.r(.85, 1.1); modal(x, [{ f, a: 1, t60: t, rise: .0012 }, { f: f * 1.003, a: .16, t60: t * .6 }, { f: f * 2.01, a: .12 * v, t60: t * .4 }, { f: f * 5.42 * (1 + .01 * R.g()), a: .22 * v, t60: .09 }, { f: f * 6.3, a: .12 * v, t60: .06 }, { f: f * 12.6, a: .06, t60: .03 }].filter(q => q.f < 14000)); const c = noise(S(.02), R); filt(c, bqc.bp(1500, .7)); expEnv(c, .0032, .0003); add(x, c, .16 * v); fade(x, .0003, .6); pan(o, x, clamp((e.n - 76) / 26 + .1 * R.g(), -.6, .6), v * .24, S(Math.max(0, e.t + .003 * R.g()))); }
  const [l, r] = convolve2(o.L, o.R, ir); return { L: l, R: r };
}
export function glock(evs, { n, seed = 'glk', toy = 0 }) { const R = rng(seed), o = st(n); for (const e of evs) { const f = mtof(e.n + (toy ? .06 * R.g() : 0)), x = mk(S(1.6)), v = e.v * R.r(.88, 1.05); modal(x, [[1, 1, 1.1], [2.756, .34, .42], [5.404, .18, .18], [8.93, .08, .08], ...(toy ? [[1.52, .18, .16], [3.9, .2, .09]] : [])].map(([r, a, t]) => ({ f: f * r, a, t60: t * Math.pow(1200 / f, .3) * (toy ? .5 : 1) })).filter(q => q.f < 15000)); const c = noise(S(.012), R); filt(c, bqc.bp(toy ? 2600 : 4800, .9)); expEnv(c, .0016); add(x, c, toy ? .3 : .2); fade(x, .0002, .4); pan(o, x, e.p ?? clamp((e.n - 84) / 28, -.5, .5), v * .2, S(Math.max(0, e.t + .002 * R.g()))); } return o; }
export function chimesTube(evs, { n, seed = 'wch' }) { const R = rng(seed), o = st(n); for (const e of evs) { const k = e.k ?? 14, pool = e.set || [86, 88, 90, 93, 95, 98, 100, 102]; for (let i = 0; i < k; i++) { const u = i / k, m = pool[Math.floor(R.u() * pool.length)], f = mtof(m) * (1 + .002 * R.g()), x = mk(S(2.2)); modal(x, [{ f, a: 1, t60: 1.6 }, { f: f * 2.76, a: .4, t60: .7 }, { f: f * 5.4, a: .15, t60: .3 }].filter(q => q.f < 15000)); fade(x, .0004, .5); pan(o, x, R.r(-.8, .8), e.v * .06 * R.r(.4, 1), S(e.t + e.dur * Math.pow(u, e.pow ?? 1) + .02 * R.u())); } } return o; }
// ------------------------------------------------------------ one breath, one line: blown instruments.  kind: shinobue | flute | whistle | clarinet | kazoo | slide
// note {t,dur,n,v,slur,port,vib,bend:[[s,c]],grace:[semitones,seconds],sh:'cr'|'dim'|'', air, late}
export function line(notes, kind, { n, seed = 'ln', hum = .004 } = {}) {
  const R = rng(kind + seed), out = mk(n), E = notes.slice().sort((a, b) => a.t - b.t), CR = 32; if (!E.length) return out;
  const phr = []; for (const e of E) { if (e.slur && phr.length) phr[phr.length - 1].push(e); else phr.push([e]); }
  const K = { shinobue: { at: .03, rel: .1, vr: 4.6, vd: 20, va: .34, air: .34, on: .18, sc: -70, st: .035 }, flute: { at: .05, rel: .1, vr: 5.1, vd: 14, va: .12, air: .16, on: .2, sc: -25, st: .03 }, whistle: { at: .03, rel: .06, vr: 5.6, vd: 22, va: .05, air: .12, on: .12, sc: -120, st: .03 }, clarinet: { at: .028, rel: .06, vr: 4.8, vd: 6, va: .05, air: .05, on: .3, sc: -18, st: .02 }, kazoo: { at: .012, rel: .04, vr: 6, vd: 30, va: .1, air: .1, on: .06, sc: -90, st: .03 }, slide: { at: .03, rel: .05, vr: 6, vd: 10, va: .02, air: .16, on: .3, sc: 0, st: .03 } }[kind];
  for (const P of phr) {
    const jit = hum * R.g(), t0 = P[0].t + jit, t1 = P[P.length - 1].t + P[P.length - 1].dur + jit, L = S(t1 - t0 + K.rel + .02), a0 = S(Math.max(0, t0)), x = mk(L);
    let ph = R.u() * TAU, vph = R.u() * TAU, dr = 0, aj = 0, am = 1, cents = 0, f = 440, amp = 0, damp = 0, bright = .5, pulsed = 0, lv = 0; const bps = [1, 2, 3].map(() => new SVF()), bw = new SVF(); let z1 = 0, gl = 0;
    for (let i = 0; i < L; i++) {
      const t = i / SR;
      if (i % CR === 0) {
        let k = 0; while (k < P.length - 1 && t >= P[k + 1].t + jit - t0 - .012) k++; const e = P[k], pv = P[k - 1], te = t - (e.t + jit - t0), iv = pv ? Math.abs(e.n - pv.n) : 0, port = e.port ?? (kind === 'whistle' ? .06 : kind === 'shinobue' ? .035 : .02);
        let m = e.n; if (pv) { const u = clamp((te + .012) / port, 0, 1); m = pv.n + (e.n - pv.n) * u * u * (3 - 2 * u); } if (e.grace && te >= 0 && te < e.grace[1]) m += e.grace[0];
        dr += (R.g() * 5 - dr) * .03; aj += (R.g() * .1 - aj) * .03; vph += TAU * K.vr * CR / SR * (1 + .05 * R.g()); const vd = (e.vib ?? 1) * clamp((te - K.on) / .3, 0, 1);
        cents = dr + K.vd * vd * Math.sin(vph) + (pv ? 0 : K.sc * Math.exp(-t / K.st)) + (e.bend ? envAt(e.bend, Math.max(0, te)) : 0) + (t > t1 - t0 ? -40 * (t - (t1 - t0)) / K.rel : 0);
        f = mtof(m) * Math.pow(2, cents / 1200); const pr = clamp(te / e.dur, 0, 1); let v = e.v * (e.sh === 'cr' ? lerp(.5, 1.1, pr) : e.sh === 'dim' ? lerp(1.05, .45, pr) : .9 + .12 * Math.sin(Math.PI * pr)); if (pv) v *= 1 - .18 * Math.exp(-Math.pow(te / .02, 2)); v *= 1 + aj + K.va * vd * Math.sin(vph + .4) * .5;
        const at = e.at ?? K.at, a = clamp(t / at, 0, 1), r = clamp((t1 - t0 + K.rel - t) / K.rel, 0, 1); const tgt = v * a * a * (3 - 2 * a) * (t > t1 - t0 ? r * r : 1); damp = (tgt - amp) / CR; bright = clamp(e.v, .1, 1); gl = Math.exp(-Math.max(0, pv ? te : t) / .028) * (pv ? .35 : 1);
        bps.forEach((b, q) => b.set(Math.min(f * (q + 1), 14000), 16)); bw.set(Math.min(f * 2.4, 9000), .8);
      }
      amp += damp; ph += TAU * f / SR; if (ph > 6283) ph -= 6283.185307179586; const w = R.u() * 2 - 1; let s = 0;
      if (kind === 'whistle') { s = Math.sin(ph) + .022 * Math.sin(2 * ph) + .008 * Math.sin(3 * ph); bps[0].run(w); bw.run(w); s = s * .5 + bps[0].bp * .035 + bw.bp * (.006 + .03 * gl); }
      else if (kind === 'flute') { const b = bright; s = Math.sin(ph) + .5 * b * Math.sin(2 * ph + .4) + .24 * b * b * Math.sin(3 * ph) + .1 * b * b * Math.sin(4 * ph) + .05 * b * b * b * Math.sin(5 * ph); for (const q of bps) q.run(w); bw.run(w); s = s * .4 + (bps[0].bp * .05 + bps[1].bp * .03) * (1 + gl) + bw.bp * (.02 + .12 * gl); }
      else if (kind === 'shinobue') { const b = bright; s = Math.sin(ph) + .34 * b * Math.sin(2 * ph + .7) + .12 * b * b * Math.sin(3 * ph + .2) + .05 * b * b * Math.sin(4 * ph); for (const q of bps) q.run(w); bw.run(w); am += .0015 * ((1 + .3 * R.g()) - am); s = s * .4 * am + (bps[0].bp * .13 + bps[1].bp * .1 + bps[2].bp * .05) * (1 + 1.4 * gl) * am + bw.bp * (.06 + .3 * gl) * am; }
      else if (kind === 'clarinet') { const c = Math.sin(ph), sh = clamp(.35 + .6 * amp, .2, 1), q = Math.tanh((1.2 + 3.2 * sh) * c + .18 * sh * Math.sin(2 * ph)); z1 += .55 * (w - z1); s = q * .34 + .05 * sh * Math.sin(2 * ph + .3) + .03 * sh * Math.sin(4 * ph) + z1 * (.012 + .07 * gl); }
      else if (kind === 'kazoo') { const u = (ph / TAU) % 1, saw = 2 * u - 1; z1 = Math.max(0, z1 * .995); s = Math.tanh(3.2 * saw + 1.2 * Math.sin(ph)) * .28 + Math.abs(saw) * .18 * (R.u() * .5 + .5) + w * .02; }
      else { s = Math.sin(ph) * .5 + .1 * Math.sin(2 * ph) + .04 * Math.sin(3 * ph); bps[0].run(w); bw.run(w); s += bps[0].bp * .06 + bw.bp * .035; }
      x[i] = s * amp;
    }
    if (kind === 'clarinet') for (const e of P) { const c = noise(S(.012), R); filt(c, bqc.bp(2400, 1.4)); expEnv(c, .002); add(x, c, .02, S(Math.max(0, e.t + jit - t0))); }
    fade(x, .0005, .02); add(out, x, 1, a0);
  }
  if (kind === 'shinobue') filt(out, bqc.hp(420, .7), bqc.hs(5200, -3), bqc.lp(8500, .6)); else if (kind === 'flute') filt(out, bqc.hp(200, .7), bqc.lp(7500, .6)); else if (kind === 'whistle') filt(out, bqc.hp(420, .7), bqc.lp(9000, .7)); else if (kind === 'clarinet') filt(out, bqc.hp(110, .7), bqc.pk(1500, 2, 1), bqc.pk(3200, -2, 1), bqc.lp(5200, .6)); else if (kind === 'kazoo') filt(out, bqc.hp(280, .7), bqc.pk(900, 6, 1.4), bqc.pk(2300, 5, 1.6), bqc.lp(5200, .7)); else filt(out, bqc.hp(300, .7), bqc.lp(8000, .7));
  return out;
}
// ------------------------------------------------------------ brass: closed-form harmonic series whose brightness follows the breath.  kind: trumpet | horn | trombone | tuba
// note {t,dur,n,v,fall,doit,gl:[[s,c]],sh:'sfz'|'cr'|'dim'|'', growl, stac}
const BRS = { // lo, hi: the frequency where the spectrum is 20 dB down, at pp and at ff
  trumpet: { lo: 850, hi: 2900, at: .02, scoop: 55, vd: 12, vr: 5.4, on: .3, bd: k => [bqc.hp(200, .7), bqc.pk(1150 * k, 3, 1), bqc.pk(2450 * k, 2.5, 1.3), bqc.pk(3600 * k, -2, 2), bqc.hs(6500, -3)], g: .34, nz: .035 },
  horn: { lo: 420, hi: 1500, at: .036, scoop: 40, vd: 7, vr: 4.8, on: .45, bd: k => [bqc.hp(90, .7), bqc.pk(430 * k, 2.5, 1), bqc.pk(1250 * k, 2, 1.2), bqc.hs(3500, -4)], g: .36, nz: .02 },
  trombone: { lo: 550, hi: 2200, at: .02, scoop: 60, vd: 2, vr: 5, on: .5, bd: k => [bqc.hp(70, .7), bqc.pk(620 * k, 2.5, 1), bqc.pk(1750 * k, 2, 1.2), bqc.hs(5000, -3)], g: .3, nz: .035 },
  tuba: { lo: 300, hi: 950, at: .026, scoop: 30, vd: 2, vr: 4.6, on: .5, bd: k => [bqc.hp(40, .7), bqc.pk(230 * k, 2, 1), bqc.hs(1500, -3)], g: .42, nz: .035 },
};
export function brass(notes, kind, { n, seed = 'b', players = 3, width = .5, center = 0, hum = .006 } = {}) {
  const C = BRS[kind], o = st(n), P = []; for (let p = 0; p < players; p++) { const R = rng('br' + kind + seed + p); P.push({ R, pan: clamp(center + (players > 1 ? (p / (players - 1) - .5) * 2 * width : 0) + .05 * R.g(), -1, 1), det: 6 * R.g(), late: R.r(-.3, 1) * hum, bri: R.r(.88, 1.1), vr: C.vr * R.r(.9, 1.1), bd: C.bd(R.r(.92, 1.08)) }); }
  for (const e of notes) for (const Q of P) {
    const R = Q.R, rel = e.fall ? .3 : e.doit ? .26 : e.stac ? .06 : .1, dur = e.dur, L = S(dur + rel), a0 = S(Math.max(0, e.t + Q.late + .002 * R.g())), x = mk(L), f0 = mtof(e.n) * Math.pow(2, Q.det / 1200), cap = Math.exp(-5.6 * f0 / 20000); let ph = R.u(), vph = R.u() * TAU, dr = 0, lip = 0, fj = 0, bj = 0, c = 0, inc = f0 / SR, a = .05, da = 0, env = 0, de = 0, nz = 0, n2 = 0, gp = 0, l1 = 0;
    const at = C.at * R.r(.8, 1.3) * (e.sh === 'cr' ? 2.2 : 1) * (1.3 - .5 * e.v), sc = -C.scoop * R.r(.2, 1) * (e.stac ? .5 : 1);
    for (let i = 0; i < L; i++) {
      if ((i & 31) === 0) {
        const t = i / SR; dr += (R.g() * 4.5 - dr) * .03; fj += (R.g() * .12 - fj) * .08; bj += (R.g() * .12 - bj) * .05; lip += (R.g() - lip) * .2; vph += TAU * Q.vr * 32 / SR * (1 + .08 * R.g()); gp += TAU * 31 * 32 / SR;
        c = dr + sc * Math.exp(-t / .022) + C.vd * (e.vib ?? 1) * clamp((t - C.on) / .3, 0, 1) * Math.sin(vph) + (e.gl ? envAt(e.gl, t) : 0) + (t > dur ? (e.fall ? -900 * Math.pow((t - dur) / rel, 1.4) : e.doit ? 700 * Math.pow((t - dur) / rel, 1.6) : 0) : 0) + (e.growl ? 22 * Math.sin(gp) : 0); inc = f0 * Math.pow(2, c / 1200) / SR;
        let v = e.v * (e.sh === 'sfz' ? .42 + .62 * Math.exp(-t / .1) + .5 * Math.pow(clamp((t - .12) / Math.max(.1, dur - .12), 0, 1), 1.6) : e.sh === 'cr' ? .25 + .8 * Math.pow(Math.min(1, t / dur), 1.4) : e.sh === 'dim' ? 1 - .65 * Math.min(1, t / dur) : .82 + .18 * Math.exp(-t / .09)); v *= 1 + fj + (e.growl ? .16 * Math.sin(gp * 1.02) : 0); const u = clamp(t / at, 0, 1); let en = v * u * u * (3 - 2 * u); if (t > dur) en *= Math.pow(clamp(1 - (t - dur) / rel, 0, 1), e.fall || e.doit ? 1.2 : 2); de = (en - env) / 32;
        const cut = (C.lo + (C.hi - C.lo) * Math.pow(clamp(en, 0, 1.1), .9)) * Q.bri * (1 + bj + .02 * lip + .5 * Math.exp(-t / .03) * e.v); da = (Math.min(cap, Math.pow(10, -f0 / cut)) - a) / 32;
      }
      env += de; a += da; ph += inc; if (ph >= 1) ph -= 1; const th = TAU * ph, cs = Math.cos(th), s = a * Math.sin(th) / (1 - 2 * a * cs + a * a) * Math.sqrt(1 - a * a) / Math.max(.2, a); const w = R.u() * 2 - 1; nz += .3 * (w - nz); n2 += .04 * (nz - n2);
      const t = i / SR; x[i] = (s + (nz - n2) * (C.nz * (1 + 2 * Math.abs(s)) + .5 * Math.exp(-t / .008))) * env;
    }
    filt(x, ...Q.bd); pan(o, x, Q.pan, C.g * .62 / Math.sqrt(players), a0);
  }
  return o;
}
// ------------------------------------------------------------ choir pad: many throats, a glottal pulse and five formants each
const VOW = { a: [[760, 0, 90], [1180, -4, 110], [2650, -14, 150], [3300, -16, 200], [4400, -26, 260]], o: [[460, 0, 70], [800, -7, 90], [2550, -24, 150], [3200, -26, 200], [4300, -36, 260]], u: [[330, 0, 60], [680, -13, 80], [2350, -32, 150], [3100, -34, 200], [4200, -44, 260]], e: [[520, 0, 80], [1850, -8, 120], [2550, -13, 150], [3400, -18, 200], [4400, -28, 260]] };
// {t,dur,n,v,vow,at,rel,sh}
export function choirPad(notes, { n, seed = 'cp', players = 6, width = .95 } = {}) {
  const o = st(n);
  notes.forEach((e, ix) => { for (let p = 0; p < players; p++) { const R = rng('cho' + seed + ix + '_' + p), low = e.n < 57, size = (low ? 1 : 1.14) * R.r(.95, 1.06), rel = e.rel ?? .25, at = (e.at ?? .12) * R.r(.8, 1.3), L = S(e.dur + rel), a0 = S(Math.max(0, e.t + (e.sh === undefined && e.at < .03 ? R.r(-.004, .02) : R.r(-.012, .05)))), x = mk(L), f0 = mtof(e.n + .12 * R.g()), cap = Math.exp(-6 * f0 / 20000); let ph = R.u(), vph = R.u() * TAU, dr = 0, aj = 0, inc = f0 / SR, env = 0, de = 0, nz = 0; const vr = R.r(4.3, 6.1), vd = R.r(6, 34) * (e.vib ?? 1), scp = R.r(30, 150), br = R.r(.1, .34); let fl = 0;
    for (let i = 0; i < L; i++) { if ((i & 31) === 0) { const t = i / SR; dr += (R.g() * 11 - dr) * .02; aj += (R.g() * .2 - aj) * .03; fl += (R.g() * 9 - fl) * .3; vph += TAU * vr * 32 / SR * (1 + .12 * R.g()); inc = f0 * Math.pow(2, (dr + vd * clamp((t - .18) / .35, 0, 1) * Math.sin(vph) + fl - scp * Math.exp(-t / .06) + (e.gl ? envAt(e.gl, t) : 0)) / 1200) / SR; let v = e.v * (e.sh === 'cr' ? .25 + .8 * Math.pow(Math.min(1, t / e.dur), 1.3) : e.sh === 'dim' ? 1 - .7 * Math.min(1, t / e.dur) : 1); const u = clamp(t / at, 0, 1); v *= u * u * (3 - 2 * u) * (1 + aj); if (t > e.dur) v *= Math.pow(clamp(1 - (t - e.dur) / rel, 0, 1), 2); de = (v - env) / 32; } env += de; ph += inc; if (ph >= 1) ph -= 1; const a = Math.min(cap, .86), th = TAU * ph, s = a * Math.sin(th) / (1 - 2 * a * Math.cos(th) + a * a); nz += .3 * ((R.u() * 2 - 1) - nz); x[i] = (s + nz * (br + .5 * Math.exp(-i / (.05 * SR)))) * env; }
    const y = mk(L); for (const [fr, g, bw] of VOW[e.vow || 'a']) { const z = Float32Array.from(x); const fc = Math.max(fr * size, fr === VOW[e.vow || 'a'][0][0] ? f0 * 1.02 : 0); filt(z, bqc.bp(fc, fc / (bw * 1.25))); add(y, z, db(g)); } add(y, x, .03); pan(o, y, clamp((p / Math.max(1, players - 1) - .5) * 2 * width + .1 * R.g(), -1, 1), .24 / Math.sqrt(players), a0); } });
  return filtS(o, bqc.hp(90, .7), bqc.lp(7000, .6));
}
// ------------------------------------------------------------ ukulele.  {t,dur,n:[..],v,dir,sp}
export function ukulele(evs, { n, seed = 'uke' }) { const R = rng(seed), x = mk(n), ir = bodyIR('uke', { len: .1, f0: 190, f1: 6500, n: 120, env: [[190, -6], [390, 4], [620, 2], [1100, 1], [2600, 1], [6500, -10]], t0: .07, t1: .005, strong: [[392, 4, .07], [612, 3, .05]], direct: 1 }); for (const e of evs) { const ns = e.dir === 'u' ? e.n.slice().reverse() : e.n, sp = e.sp ?? R.r(.007, .012); ns.forEach((m, k) => { const v = e.v * R.r(.85, 1.1) * (e.dir === 'u' ? .8 : 1); const y = pluck(mtof(m + .05 * R.g()), e.dur, { t60: 1.15, hi: [3600, .3], pick: R.r(.2, .3), pickup: 0, bright: 5200 + 3600 * v, nz: .28, rel: .05, vel: v * .3, soft: .0004, tail: .12, bend: t => 6 * Math.exp(-t / .03) }, R); add(x, y, 1, S(Math.max(0, e.t + k * sp + .002 * R.g()))); }); const nk = noise(S(.02), R); filt(nk, bqc.bp(2600, .7)); fade(nk, .002, .012); add(x, nk, .05 * e.v, S(Math.max(0, e.t - .003))); } const y = convM(x, ir); return filt(y, bqc.hp(170, .7), bqc.hs(3000, 2), bqc.lp(9500, .6)); }
// ------------------------------------------------------------ toys
export function boing(evs, { n, seed = 'bo' }) { const R = rng(seed), x = mk(n); for (const e of evs) { const L = S(e.dur ?? .5), f0 = mtof(e.n), y = mk(L), s = new SVF(); let ph = 0; for (let i = 0; i < L; i++) { const t = i / SR, f = f0 * (1 + .5 * Math.exp(-t / .03)) * (1 + .03 * Math.sin(TAU * 18 * t) * Math.exp(-t / .2)); ph += f / SR; const u = ph % 1; if ((i & 15) === 0) s.set(500 + 1900 * (.5 + .5 * Math.sin(TAU * (9 + 10 * t) * t * 1.0 - 1.2)) * Math.exp(-t / .35), 5); s.run(Math.tanh((2 * u - 1) * 2)); y[i] = (s.bp * .6 + s.lp * .25) * Math.exp(-t / .16) * Math.min(1, t / .002); } fade(y, 0, .05); add(x, y, e.v * .8, S(e.t)); } return x; }
export function pfff(evs, { n, seed = 'pf' }) { const R = rng(seed), o = st(n); for (const e of evs) { const L = S(e.dur), x = noise(L, R); const s = new SVF(); for (let i = 0; i < L; i++) { const u = i / L; if ((i & 31) === 0) s.set(lerp(2600, 500, u), 1.2); s.run(x[i]); x[i] = s.bp * Math.pow(1 - u, 1.4) * Math.min(1, i / 300) * (.6 + .4 * Math.sin(TAU * 22 * i / SR)); } pan(o, x, e.p ?? 0, e.v * .5, S(e.t)); } return o; }
// heart: two soft valves felt through bone
export function heart(evs, { n, seed = 'ht' }) { const R = rng(seed), x = mk(n); for (const e of evs) for (const [dt, f, v, d] of [[0, 46, 1, .1], [.15, 58, .62, .075]]) { const L = S(.5), y = mk(L); let ph = 0; for (let i = 0; i < L; i++) { const t = i / SR, fr = f * (1 + .7 * Math.exp(-t / .016)); ph += TAU * fr / SR; y[i] = Math.sin(ph) * Math.exp(-t / d) * Math.min(1, t / .006); } const nb = noise(L, R); filt(nb, bqc.lp(170, .8), bqc.hp(40, .7)); expEnv(nb, .035, .004); add(y, nb, .7); modal(y, [{ f: 92, a: .2, t60: .12 }, { f: 138, a: .1, t60: .08 }]); fade(y, 0, .1); add(x, y, e.v * v * .6, S(e.t + dt)); } return x; }
// bowed glass: a clear, slowly turning sine stack
export function bowedGlass(evs, { n, seed = 'bg' }) { const R = rng(seed), o = st(n); for (const e of evs) { const at = e.at ?? .35, rel = e.rel ?? .7, L = S(e.dur + rel), f = mtof(e.n); for (const [r, a] of [[1, 1], [1.0013, .5], [2.005, .16], [2.97, .07], [4.1, .03]]) { const x = mk(L), th = R.r(-.7, .7), w = R.r(3.6, 5), p0 = R.u() * 6; let ph = R.u() * 6; for (let i = 0; i < L; i++) { const t = i / SR; ph += TAU * f * r * (1 + .0011 * Math.sin(TAU * w * t + p0)) * (e.gl ? Math.pow(2, envAt(e.gl, t) / 1200) : 1) / SR; const u = clamp(t / at, 0, 1); let k = u * u * (3 - 2 * u); if (t > e.dur) k *= Math.pow(1 - (t - e.dur) / rel, 2); if (e.sh === 'cr') k *= Math.pow(Math.min(1, t / e.dur), 1.5); x[i] = Math.sin(ph) * k * (1 + .12 * Math.sin(TAU * w * .6 * t + p0)); } pan(o, x, clamp((e.p ?? 0) + th * .5, -1, 1), e.v * a * .07, S(Math.max(0, e.t))); } } return o; }
// tinnitus-like ring
export function ring(evs, { n }) { const o = st(n); for (const e of evs) { const L = S(e.dur), x = mk(L), f = mtof(e.n); for (let i = 0; i < L; i++) { const t = i / SR; x[i] = (Math.sin(TAU * f * t) + .5 * Math.sin(TAU * f * 1.0015 * t)) * Math.min(1, t / (e.at ?? .05)) * Math.pow(1 - t / e.dur, 1.5); } pan(o, x, e.p ?? 0, e.v * .05, S(e.t)); } return o; }
