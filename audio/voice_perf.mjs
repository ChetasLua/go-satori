// voice_perf.mjs : reads the score like a singer.  Notes + morae become muscles: lung pressure, fold closure, fold swing, pitch, tongue, lips, velum.
import { rng, slowNoise, clamp, lerp, smooth, ss5, midi } from './voice_dsp.mjs';
import { CR } from './voice_engine.mjs';
export const E8 = 60 / 144 / 2, CM = 980;        // eighth note (s), cm of water in dyn/cm2

export function parseMora(m) {
  if (m === 'N' || m === 'Q') return { t: m };
  const x = /^(sh|ch|ts|[kgsztdnhbpmyrwjf])?(y)?([aiueo])$/.exec(m); if (!x) throw Error('mora ' + m);
  let c = x[1] || '', v = x[3]; if (c === 'h' && v === 'i') c = 'hy'; if (c === 'h' && x[2]) c = 'hy'; if (c === 'h' && v === 'u') c = 'f'; if (!c && x[2]) c = 'y'; if (c === 's' && v === 'i') c = 'sh'; if (c === 't' && v === 'i') c = 'ch'; if (c === 't' && v === 'u') c = 'ts'; if (c === 'z' && v === 'i') c = 'j';
  return { t: 'CV', c, v, pal: !!x[2] || c === 'sh' || c === 'ch' || c === 'j' || c === 'y' };
}
const VOICED = new Set(['g', 'z', 'j', 'd', 'n', 'b', 'm', 'y', 'r', 'w', '', 'N']);
export const CT = {
  k: { cl: .048, vot: .027 }, t: { cl: .050, vot: .023 }, p: { cl: .052, vot: .017 }, d: { cl: .042, vot: 0 }, b: { cl: .050, vot: 0 }, g: { cl: .050, vot: 0 }, n: { cl: .054, vot: 0 }, m: { cl: .062, vot: 0 },
  s: { fr: .096 }, sh: { fr: .096 }, z: { fr: .072 }, j: { cl: .028, fr: .042 }, ch: { cl: .038, fr: .052 }, ts: { cl: .038, fr: .046 }, h: { fr: .060 }, hy: { fr: .072 }, f: { fr: .064 }, r: { cl: .020, vot: 0 }, w: { gl: .090 }, y: { gl: .076 },
};
const unit = { a: [1, 0, 0, 0, 0], i: [0, 1, 0, 0, 0], u: [0, 0, 1, 0, 0], e: [0, 0, 0, 1, 0], o: [0, 0, 0, 0, 1] };
const mix = (a, b, t) => a.map((x, i) => x * (1 - t) + b[i] * t);
const BODY = {
  t: [[0, .25, .25, .5, 0], .30], d: [[0, .25, .25, .5, 0], .30], n: [[0, .25, .25, .5, 0], .28], s: [[0, .3, .3, .4, 0], .30], z: [[0, .3, .3, .4, 0], .30], ts: [[0, .25, .5, .25, 0], .35], r: [[0, .1, .4, .5, 0], .22],
  sh: [[0, .8, .1, .1, 0], .75], ch: [[0, .85, .1, .05, 0], .75], j: [[0, .85, .1, .05, 0], .75], y: [[0, .95, 0, .05, 0], .97], hy: [[0, 1, 0, 0, 0], .7], k: [[.1, .1, .5, .1, .2], .14], g: [[.1, .1, .5, .1, .2], .14], w: [[0, 0, .7, 0, .3], .9], h: [[0, 0, 0, 0, 0], 0], f: [[0, 0, 1, 0, 0], .3], m: [[0, 0, 0, 0, 0], 0], b: [[0, 0, 0, 0, 0], 0],
};
const VELLOC = { a: 28.5, o: 27.5, u: 29, e: 31.5, i: 33 };
// colour of the hiss at each place of articulation (gain, sections downstream, low-pass, high-pass)
const HISS = { s: { ng: 6.5, nd: 2, nf: 8000, nh: 2500 }, sh: { ng: 3.8, nd: 3, nf: 7000, nh: 1200 }, z: { ng: 7, nd: 2, nf: 8000, nh: 2500 }, j: { ng: 4.2, nd: 3, nf: 7000, nh: 1200 }, hy: { ng: 2.2, nd: 2, nf: 5000, nh: 900 }, f: { ng: 1.2, nd: 0, nf: 5000, nh: 300 }, k: { np: 2, ng: 8, nd: 1, nf: 2000, nh: 400 }, K: { np: 2, ng: 10, nd: 1, nf: 2300, nh: 900 }, G: { np: 2, ng: 9, nd: 1, nf: 2300, nh: 800 }, t: { ng: 7.5, nd: 2, nf: 7500, nh: 1600 }, p: { np: 2, ng: 5, nd: 0, nf: 3000, nh: 200 }, g: { np: 2, ng: 12, nd: 1, nf: 1700, nh: 450 }, d: { ng: 4, nd: 2, nf: 7000, nh: 1000 }, b: { np: 2, ng: 2.2, nd: 0, nf: 2500, nh: 200 }, r: { ng: .6, nd: 1, nf: 4000, nh: 500 } };

export function buildLine(notes0, st, o = {}) {
  const R = rng((o.seed ?? 1) * 7919 + 13), nz = [0, 1, 2, 3, 4, 5, 6, 7].map(k => slowNoise((o.seed ?? 1) * 977 + k * 31)), hum = o.hum ?? .007, shift = o.shift ?? 0, e8 = o.e8 ?? E8;
  const notes = notes0.map((n, i) => { const [p0, p1] = n.pitch.split('>'), m = parseMora(n.mora), t = n.t * e8 + (o.lag ?? 0) + (i ? R.g() * hum : R.g() * hum * .5); return { ...m, mora: n.mora, t, d: (n.d - ((n.x || {}).short ?? 0)) * e8, m0: midi(p0) + shift + (n.h ?? 0), m1: p1 ? midi(p1) + shift + (n.h ?? 0) : null, x: n.x || {} }; });
  for (let k = 0; k + 1 < notes.length; k++) { const gap = notes[k + 1].t - (notes[k].t + notes[k].d); if (gap > 0 && gap < (o.joinGap ?? .16) && !notes[k].x.cut) notes[k].d += gap; }
  const T0 = notes[0].t - 1.1, T1 = notes[notes.length - 1].t + notes[notes.length - 1].d + .9, n = Math.ceil((T1 - T0) * CR), F = v => new Float32Array(n).fill(v), idx = t => clamp(Math.round((t - T0) * CR), 0, n - 1), tm = i => T0 + i / CR;
  const s = { n, f0: F(0), P: F(0), x01: F(0), x02: F(0), xm: F(0), phz: F(1.1), tilt: F(9000), asp: F(1), rasp: F(0), jit: F(1), velum: F(0), eff: F(0), lips: F(0), vent: F(0), gain: F(1), w: [F(0), F(0), F(0), F(0), F(0)], gest: [] };
  const kS = (o.fold ?? 1.45) / 1.45, fk = o.f0k ?? 1, gate = F(0), abd = F(0), dipv = F(1), cents = F(0), post = [], volt = [], micro = [];
  const ramp = (t0, t1, t2, t3) => { const a = F(0); for (let i = idx(t0); i <= idx(t3); i++) { const t = tm(i); a[i] = t < t1 ? ss5((t - t0) / Math.max(1e-4, t1 - t0)) : t < t2 ? 1 : 1 - ss5((t - t2) / Math.max(1e-4, t3 - t2)); } return a; };
  const close = (loc, hw, amin, t0, t1, t2, t3, deg = 1, hiss = null) => { const m = ramp(t0, t1, t2, t3); if (deg !== 1) for (let i = 0; i < n; i++) m[i] *= deg; const h = (hiss && HISS[hiss]) || {}; s.gest.push({ loc, hw, amin, m, ...h, ng: (h.ng ?? 1) * (st.hiss ?? 1) * (first ? st.initHiss ?? 1.8 : 1) * hsN * (hiss === 's' || hiss === 'sh' || hiss === 'z' || hiss === 'j' ? st.sib ?? 1 : 1) }); };
  const open = (t0, t1, t2, t3, a) => { const r = ramp(t0, t1, t2, t3); for (let i = idx(t0); i <= idx(t3); i++) abd[i] = Math.max(abd[i], a * r[i]); };
  const dip = (t0, t1, t2, t3, g) => { const r = ramp(t0, t1, t2, t3); for (let i = idx(t0); i <= idx(t3); i++) dipv[i] = Math.min(dipv[i], 1 - (1 - g) * r[i]); };
  const addVel = (t0, t1, t2, t3, a) => { const r = ramp(t0, t1, t2, t3); for (let i = idx(t0); i <= idx(t3); i++) s.velum[i] = Math.max(s.velum[i], a * r[i]); };
  const CS = st.cons ?? .82, WH = !!o.whisper, tau0 = st.tau ?? .015; let cur = null, first = false, hsN = 1;
  const COR = { t: 1, d: 1, n: 1, s: .9, z: .9, ts: .95, r: .85 }, CW = [.42, .25, .42, .25, .46], CB = [0, .32, .14, .54, 0];
  const bodyFor = (c, pv, nv, dur = 1) => { const b = BODY[c], base = mix(pv, nv, .55); if (COR[c]) { let w = 0; for (let v = 0; v < 5; v++) w += base[v] * CW[v]; return mix(base, CB, Math.min(.85, w * COR[c] * (st.loc ?? 1.1) * (dur < .14 ? .6 : 1))); } return b && b[1] > 0 ? mix(base, b[0], b[1]) : base; };
  for (let k = 0; k < notes.length; k++) {
    const nt = notes[k], pr = notes[k - 1], nx = notes[k + 1], gapBefore = !pr || nt.t - (pr.t + pr.d) > .04, tv = nt.t; nt.start = gapBefore; first = gapBefore; hsN = nt.x.hs ?? 1; nt.end = !nx || nx.t - (nt.t + nt.d) > .04;
    if (nt.mora === 'Q') { nt.kind = 'Q'; continue; } if (nt.mora === 'N') { nt.kind = 'N'; if (gapBefore || pr.hum) { nt.hum = true; nt.c = 'm'; cur = mix(unit.o, unit.u, .5); if (gapBefore) { post.push({ t: tv - .5, w: cur, tau: .04 }); volt.push({ on: tv - .02, r0: .06, off: null, r1: .02, phr: true }); } } continue; }
    const nv = unit[nt.v], from = gapBefore ? nv : cur || nv; cur = nv; nt.kind = 'V';
    const avail = gapBefore ? .25 : (pr.kind === 'Q' ? pr.d * .62 : pr.kind === 'N' ? pr.d * .55 : Math.min(pr.d * .52, .13)), q = CT[nt.c], len = q ? (q.cl || 0) + (q.vot || 0) + (q.fr || 0) + (q.gl || 0) : 0, sc = q ? Math.min(nt.x.con > 1.6 ? 2.0 : 1.35, CS * (nt.x.con ?? 1) * (1 + .06 * R.g()), avail / len) : 1, L = x => (q[x] || 0) * sc;
    let c0 = tv, c1 = tv; const c = nt.c, afterQ = pr && pr.kind === 'Q' && !gapBefore, afterN = pr && pr.kind === 'N' && !gapBefore, voiced = VOICED.has(c);
    if (c) {
      if (q.gl) { c0 = tv - L('gl'); c1 = tv - L('gl') * .25; post.push({ t: c0 - .02, w: bodyFor(c, from, nv), tau: tau0 }, { t: c1, w: nv, tau: tau0 * 1.15 }); if (c === 'w') close(43, 2.2, .16, c0 - .03, c0 + .025, c1, tv + .05, .95); else close(33, 4, .30, c0 - .02, c0 + .03, c1, tv + .045, .75); dip(c0 - .02, c0 + .025, c1, tv + .035, .85); }
      else if (c === 'h' || c === 'hy' || c === 'f') {
        c0 = tv - L('fr'); post.push({ t: c0 - .03, w: bodyFor(c, from, nv), tau: tau0 }, { t: c0 + .012, w: nv, tau: tau0 });
        if (c === 'h' && gapBefore) c0 -= .03; if (c === 'h') open(c0 - .035, c0 + .018, tv - .022, tv + .045, st.hOpen ?? .17);
        else if (c === 'hy') { close(33.5, 4.5, .11, c0 - .04, c0, tv - .016, tv + .035, .95, 'hy'); open(c0 - .03, c0 + .015, tv - .020, tv + .034, .17); }
        else { close(43, 2, .11, c0 - .035, c0, tv - .014, tv + .03, .95, 'f'); open(c0 - .03, c0 + .015, tv - .020, tv + .034, .17); }
      }
      else if (q.fr) {
        const cl = L('cl'), f = L('fr'); c0 = tv - cl - f; c1 = tv - .004; if (afterQ) c0 = pr.t - Math.max(.004, Math.min(.04, .16 - pr.d));
        post.push({ t: c0 - .03, w: bodyFor(c, from, nv), tau: tau0 }, { t: tv - .022, w: nv, tau: nt.pal ? tau0 * 1.25 : tau0 });
        const po = c === 'sh' || c === 'ch' || c === 'j', loc = po ? 36.0 : 39.0, hw = po ? 2.6 : 2.0, am = po ? .13 : .085, h = c === 'z' ? 'z' : c === 'j' ? 'j' : po ? 'sh' : 's';
        if (cl) close(loc + (po ? .8 : 0), hw, .0004, c0 - .035 - (gapBefore ? .12 : 0), c0 - (gapBefore ? .12 : 0), tv - f - .004, tv - f + .010, 1, h);
        close(loc, hw, am, c0 - .035 - (gapBefore ? .06 : 0), c0 - .002 - (gapBefore ? .06 : 0), c1 - .010, tv + .030, .985, h);
        if (!voiced) { if (cl) open(tv - f - .028, tv - f + .006, tv - .026, tv + .010, .15); else open(c0 - .030, c0 + .022, tv - .030, tv + .010, .20); } else open(c0 - .01, c0 + .02, tv - .02, tv + .012, .055);
        if (voiced) dip(c0 - .02, c0 + .02, tv - .012, tv + .03, .78);
      }
      else {
        const cl = L('cl') * (gapBefore && voiced && 'gdb'.includes(c) ? 1.5 : 1), vot = L('vot') * (st.vot ?? 1); c0 = tv - cl - vot; c1 = tv - vot; if (afterQ) c0 = pr.t - (st.qEarly ?? 0) - Math.max(.004, Math.min(.04, .16 - pr.d)); if (afterN) c0 = Math.max(pr.t + pr.d * .6, c0);
        const bw = nt.pal || (c === 'n' && nt.v === 'i') ? mix(nv, unit.i, .85) : bodyFor(c, from, nv, nt.d); post.push({ t: c0 - .036, w: bw, tau: tau0 }, { t: c1 - (nt.pal ? .004 : COR[c] ? (st.antic ?? -.002) : .030), w: nv, tau: nt.pal ? tau0 * 1.3 : tau0 });
        const loc = { k: VELLOC[nt.pal ? 'i' : nt.v], g: VELLOC[nt.pal ? 'i' : nt.v], t: 39.8, d: 39.8, n: nt.v === 'i' || nt.pal ? 37.2 : 39.8, r: 39.0, b: 43, p: 43, m: 43 }[c] + (nt.x.front ?? 0), hw = { k: 3.0, g: 3.0, t: 2.2, d: 2.2, n: nt.v === 'i' || nt.pal ? 3.0 : 2.4, r: 1.6, b: 1.8, p: 1.8, m: 1.8 }[c];
        const tin = { k: .042, g: .042, t: .034, d: .030, n: .034, r: .018, b: .034, p: .034, m: .036 }[c], tout = { k: .048, g: nt.x.slow ? nt.x.slow : .052, t: .038, d: .038, n: .040, r: .024, b: .050, p: .050, m: .054 }[c], am = c === 'r' ? .05 : .0004;
        const hold = voiced ? 0 : Math.min(.012, vot * .5), hk = (c === 'k' || c === 'g') && (nt.pal || nt.v === 'i' || nt.v === 'e') ? c.toUpperCase() : c, rq = c === 'n' ? .010 : c === 'r' ? .010 : c === 'g' ? .011 : c === 'k' ? .007 : voiced ? .005 : .003; const ce = gapBefore && 'kgtdbp'.includes(c) ? .12 : 0; close(loc, hw, am, c0 - tin - ce, c0 - ce, c1, c1 + rq + hold * .25, 1, hk); close(loc, hw, .12, c0 - tin - ce, c0 - ce, c1, c1 + tout + hold, .88, hk);
        if (c === 'k') open(c1 - .026, c1 - .002, c1 + vot * .25, tv + .010, gapBefore ? .065 : .075);
        if (c === 't') open(c1 - .030, c1 - .002, c1 + vot * .4, tv + .012, .12);
        if (c === 'p') open(c1 - .030, c1 - .002, c1 + vot * .25, tv + .010, .09);
        if (gapBefore && (c === 'd' || c === 'b' || c === 'g')) open(c1 - .034, c1 - .008, c1 - .001, c1 + .012, st.initOpen ?? .035);
        if (c === 'd' || c === 'b') { dip(c0 - .02, c0 + .012, c1 - .004, c1 + .03, .85); micro.push([c0 + cl * .5, -40, .03]); }
        if (nt.x.soften && (c === 'g' || c === 'd' || c === 'b')) dip(c0 - .02, c0 + .012, c1 + .012, c1 + .012 + .05 * nt.x.soften, 1 - .7 * Math.min(1, nt.x.soften));
        if (c === 'g') { const nas = gapBefore ? 0 : st.gn ?? 0; if (nas) addVel(c0 - .035, c0 + .005, c1 - .012, c1 + .03, .30 * nas); dip(c0 - .02, c0 + .012, c1 - .004, c1 + .03, .85); micro.push([c0 + cl * .5, -30, .03]); }
        if (c === 'n' || c === 'm') { addVel(c0 - .034, c0 + .004, c1 - .004, c1 + .028, .55); micro.push([c0 + cl * .5, -12, .03]); }
        if (c === 'r') { dip(c0 - .018, c0 + .004, c1, c1 + .022, .94); micro.push([c0 + cl * .5, -25, .02]); }
      }
    } else if (!gapBefore) { const same = pr.v === nt.v; post.push({ t: tv - .052, w: nv, tau: tau0 * 1.12 }); if (!nt.x.tie) { if (same) dip(tv - .035, tv - .004, tv + .002, tv + .04, .70); else dip(tv - .04, tv - .005, tv + .004, tv + .045, .92); } }
    if (nt.d > .4 && (st.belt ?? 0)) { const o2 = { a: 'a', i: 'e', u: 'o', e: 'a', o: 'a' }[nt.v], kk = { a: 0, i: .18, u: .12, e: .12, o: .12 }[nt.v] * st.belt; post.push({ t: tv + .12, w: mix(nv, unit[o2], kk * .6), tau: .06 }, { t: tv + nt.d * .45, w: mix(nv, unit[o2], kk), tau: .09 }, { t: tv + nt.d * .85, w: mix(nv, [.2, .1, .3, .3, .1], .14), tau: .06 }); }
    if (gapBefore) post.push({ t: c0 - .45, w: mix(nv, [.35, 0, .2, .45, 0], .6), tau: .04 }, { t: c0 - .10, w: c ? bodyFor(c, nv, nv) : nv, tau: .025 });
    nt.c0 = c0; nt.c1 = c1;
    if (gapBefore || !voiced) volt.push({ on: voiced ? (c === '' ? tv - .010 : c === 'w' && gapBefore ? c0 - .02 : (c === 'g' || c === 'd' || c === 'b') && gapBefore ? (nt.x.hard ? c1 + .006 * nt.x.hard : c0 - .004) : c === 'y' || c === 'w' ? c0 : c0 + (c === 'n' || c === 'm' ? -.015 : .004)) : c === 'h' ? tv - .020 : tv - .006, r0: gapBefore && c === '' ? (st.onset ?? .03) : voiced ? .03 : .018, off: null, r1: .02, phr: gapBefore });
    if (!voiced && volt.length > 1 && volt[volt.length - 2].off === null) { const v = volt[volt.length - 2]; v.off = c === 'h' || c === 'hy' || c === 'f' ? c0 + .010 : c0 + .004; v.r1 = c === 'h' ? .03 : .016; }
    if (nt.end) { const v = volt[volt.length - 1], rel = nt.x.rel ?? st.rel ?? .11; nt.offT = nt.t + nt.d - (nx ? .05 : .02) - rel * .35; if (!nx && o.nextT !== undefined) nt.offT = Math.min(nt.offT, Math.max(nt.t + nt.d * .55, o.nextT - .19 - rel * .7)); v.off = nt.offT; v.r1 = rel; nt.rel = rel; }
  }
  notes.forEach((nt, k) => { if (!nt.hum) return; const pr = notes[k - 1], nx = notes[k + 1]; if (!pr || !pr.hum) { let j = k; while (notes[j + 1] && notes[j + 1].hum) j++; const l = notes[j], more = notes[j + 1] && !notes[j + 1].start, te = l.t + l.d + (more ? -.06 : .35); close(43, 1.4, .0008, nt.t - .34, nt.t - .24, te, te + .05, 1); addVel(nt.t - .4, nt.t - .28, te + .02, te + .08, .75); } if (nt.end) { const v = volt[volt.length - 1], rel = nt.x.rel ?? st.rel ?? .2; nt.offT = nt.t + nt.d - .03 - rel * .3; v.off = nt.offT; v.r1 = rel; nt.rel = rel; } nt.c0 = nt.t - .02; });
  notes.forEach((nt, k) => { const nx = notes[k + 1], pr = notes[k - 1]; if (nt.mora === 'N' && !nt.hum) { const c = nx && !nx.start ? nx.c : '', vel = c === 'g' || c === 'k' || c === '', t0 = nt.t, t1 = nx && !nx.start ? (nx.c1 ?? nx.t) : nt.t + nt.d + .1, lab = c === 'm' || c === 'b' || c === 'p'; close(lab ? 43 : vel ? 28 : 40, lab ? 1.4 : vel ? 3.2 : 2.4, .0008, t0 - .03, t0 + .035, t1 - .01, t1 + .03, 1); const oral = nx && !nx.start && 'bdgkptszjr'.includes((nx.c || 'x')[0]), nasalNext = c === 'm' || c === 'n'; addVel(t0 - .060, t0 + .006, oral ? (nx.c0 ?? t1) - .024 : t1 - .02, oral ? Math.min(t1 - .022, (nx.c0 ?? t1) + .012) : t1 + .020, .65); post.push({ t: t0 - .02, w: mix(unit[pr.v], [0, .3, .3, .4, 0], vel ? .3 : .45), tau: .03 }); if (nt.end) { const v = volt[volt.length - 1], rel = nt.x.rel ?? .12; nt.offT = nt.t + nt.d - .03 - rel * .3; v.off = nt.offT; v.r1 = rel; } } });
  first = false; hsN = 1; post.sort((a, b) => a.t - b.t);
  { let k = 0, tg = post[0].w, tau = post[0].tau; const a = tg.slice(), b = tg.slice(); for (let i = 0; i < n; i++) { const t = tm(i); while (k < post.length && post[k].t <= t) { tg = post[k].w; tau = post[k].tau; k++; } const g = 1 - Math.exp(-1 / (CR * tau)); for (let v = 0; v < 5; v++) { a[v] += (tg[v] - a[v]) * g; b[v] += (a[v] - b[v]) * g; s.w[v][i] = b[v]; } } }
  for (const v of volt) { const off = v.off ?? T1 - .2; for (let i = idx(v.on - v.r0 * .4); i <= idx(off + v.r1); i++) { const t = tm(i), u = smooth((t - (v.on - v.r0 * .4)) / v.r0), d = 1 - smooth((t - off) / v.r1); gate[i] = Math.max(gate[i], u * d); } }

  // ---------- pitch ----------
  const wn = st.wn ?? 44, zeta = st.zeta ?? .66, steps = [];
  notes.forEach((nt, k) => { const pr = notes[k - 1], lead = nt.start ? .30 : clamp(.056 + .006 * Math.abs(nt.m0 - pr.m0), .05, .095); steps.push({ t: nt.t - lead + (nt.x.late ?? 0), m: nt.m0, nt }); if (nt.m1 !== null) steps.push({ t: nt.t + nt.d * (nt.x.glideAt ?? .42), m: nt.m1, nt, slow: 1 }); });
  { let k = 0, x = steps[0].m, v = 0, tg = x, w = wn; for (let i = 0; i < n; i++) { const t = tm(i); while (k < steps.length && steps[k].t <= t) { const s0 = steps[k]; if (s0.nt.start && !s0.slow) { x = s0.m; v = 0; } tg = s0.m; w = s0.slow ? 18 : wn * (1 + .15 * (s0.nt.x.snap ?? 0)); k++; } const dt = 1 / CR; for (let q = 0; q < 2; q++) { const ac = w * w * (tg - x) - 2 * zeta * w * v; v += ac * dt / 2; x += v * dt / 2; } cents[i] = x * 100; } }
  const bump = (tc, a, sg) => { for (let i = idx(tc - 3.2 * sg); i <= idx(tc + 3.2 * sg); i++) { const z = (tm(i) - tc) / sg; cents[i] += a * Math.exp(-.5 * z * z); } };
  for (const [t, a, sg] of micro) bump(t, a * (st.micro ?? 1.3) * (.6 + .8 * R.u()), sg);
  notes.forEach((nt, k) => { bump(nt.t + nt.d * .5, 7 * R.g(), Math.min(.12, nt.d * .32)); if (nt.d > .5) { bump(nt.t + .10, -10 - 8 * R.u(), .07); bump(nt.t + nt.d * .42, 6 * R.g() + 3, .16); bump(nt.t + nt.d * .9, -8 - 10 * R.u(), nt.d * .10); } if (nt.d < .3) bump(nt.t + nt.d * .86, -9 - 6 * R.u(), .03); const pr = notes[k - 1];
    if (pr && !nt.start && nt.m0 - pr.m0 >= 3) bump(nt.t - .10, -(8 + 5 * (nt.m0 - pr.m0)), .026); if (pr && !nt.start && pr.m0 - nt.m0 >= 4) bump(nt.t - .10, 10, .026);
    const sc = nt.x.scoop ?? (nt.start ? (st.scoop0 ?? -70) * (.7 + .6 * R.u()) : R.u() < .6 ? -38 * R.u() : 0); if (sc) { const tau = nt.x.scoopT ?? .055, t0 = nt.start ? nt.c0 ?? nt.t : nt.t - .06; for (let i = idx(t0 - .1); i < idx(nt.t + .5); i++) { const t = tm(i) - t0; cents[i] += sc * (t < 0 ? (nt.start ? 1 : smooth(1 + t / .06)) : Math.exp(-Math.pow(t / tau, 1.5) * .5)); } }
    if (nt.x.fall || (nt.end && st.fall)) { const a = nt.x.fall ?? st.fall, te = (nt.offT ?? nt.t + nt.d) + .02, L = nt.x.fallT ?? .16; for (let i = idx(te - L); i < Math.min(n, idx(te + .3)); i++) { const u = (tm(i) - (te - L)) / L; cents[i] -= a * u * u; } }
    if (nt.x.bend) for (const [t, a, sg] of nt.x.bend) bump(nt.t + t, a, sg); });
  const vibPhase = F(0);
  notes.forEach((nt, k) => { const d = nt.d; if (d < (st.vibMin ?? .42)) return; const depth0 = (nt.x.vib ?? st.vib ?? 32) * (.62 + .62 * R.u()); if (depth0 <= 0) return;
    const on = nt.t + Math.max(.22, (nt.x.vibOn ?? clamp(d * .34, .22, .50)) * (.85 + .4 * R.u())), off = nt.offT ? nt.offT + .05 : nt.t + d - .05, rate = (nt.x.vibRate ?? st.vibRate ?? 5.5) * (.92 + .16 * R.u()); let cyc = 1 + .22 * R.g(), cyc1 = cyc, p = 0, lastc = 0, rr = rate, rr1 = rate; const ske = .10 + .12 * R.u();
    for (let i = idx(on); i <= idx(off); i++) { const t = tm(i), u = smooth((t - on) / (nt.x.vibIn ?? clamp(d * .28, .16, .6))), e = 1 - smooth((t - off + .10) / .10), r2 = rr * (1 + .05 * Math.sin((t - on) * 2.2) + .03 * nz[3](t, 1.3)); p += 2 * Math.PI * r2 / CR; const cc = Math.floor(p / (2 * Math.PI)); if (cc !== lastc) { lastc = cc; cyc1 = clamp(1 + .30 * R.g(), .45, 1.6); rr1 = rate * (1 + .075 * R.g()); } cyc += (cyc1 - cyc) * .03; rr += (rr1 - rr) * .05; const vv = Math.sin(p) + ske * Math.sin(2 * p + .7 + nz[6](t, .8)) + .16 * Math.sin(.5 * p + 1) + .18 * nz[7](t, 7.5); cents[i] += depth0 * u * e * cyc * vv; vibPhase[i] = u * e * Math.sin(p); } });
  const drift = st.drift ?? 1, dtn = o.detune ?? 0; for (let i = 0; i < n; i++) { const t = tm(i); cents[i] += drift * (8 * nz[0](t, 1.1) + 7.5 * nz[1](t, 3.7) + 6 * nz[2](t, 8.5) + 2.5 * nz[7](t, 15)) + dtn + (o.wander ?? 0) * nz[4](t, .5); if (st.tremor) cents[i] += st.tremor * nz[5](t, 6.5) * 10; }

  // ---------- effort, loudness, closure ----------
  const tA = notes[0].t, tB = notes[notes.length - 1].t + notes[notes.length - 1].d, [e0, e1, e2] = st.eff, pk = st.peak ?? .55, eff = F(0);
  for (let i = 0; i < n; i++) { const u = clamp((tm(i) - tA) / (tB - tA), 0, 1); eff[i] = u < pk ? lerp(e0, e1, smooth(u / pk)) : lerp(e1, e2, smooth((u - pk) / (1 - pk))); }
  const evo = F(0), eadd = F(0), raspE = F(0), gdb = F(0), xadd = F(0), vfade = F(1), aspAdd = F(0), fryE = F(0), gx = F(0);
  notes.forEach((nt, k) => {
    const pr = notes[k - 1], nx = notes[k + 1], e8t = Math.round((nt.t) / e8), acc = nt.x.acc ?? ((e8t % 8 === 0) ? 1 : (e8t % 4 === 0) ? .6 : (e8t % 2 === 0) ? .25 : 0) * (st.accent ?? 1), end = nx && !nt.end ? nx.t : nt.t + nt.d, hi = pr ? clamp((nt.m0 - pr.m0) / 12, -.4, .4) : 0, rnd = R.g() * .7;
    for (let i = idx(nt.t - .04); i < idx(end + .02); i++) { const t = tm(i) - nt.t, w = smooth((t + .04) / .05) * (1 - smooth((tm(i) - end) / .02)), uu = clamp(t / Math.min(nt.d, .42), 0, 1), arc = -(st.arc ?? 2) * (Math.pow(1 - smooth(Math.min(1, t / .07)), 2) * .8 + .9 * smooth((uu - .4) / .6) * (nt.d > .5 ? .45 : 1)); gdb[i] += w * (acc * 1.9 * Math.exp(-Math.max(0, t) / .16) + 1.4 * hi + rnd + arc); eadd[i] += w * (acc * .06 * Math.exp(-Math.max(0, t) / .15) + .10 * hi); }
    { const k2 = nt.d > .35 ? 1 : .55; for (let i = idx(nt.t - .02); i < idx(end + .06); i++) { const u = clamp((tm(i) - nt.t) / nt.d, 0, 1.15); evo[i] += k2 * ((.26 + .08 * rnd) * Math.sin(Math.PI * Math.min(1, u * 1.1)) - .14); } }
    if (nt.d > .4) { const bl = st.belt ?? 0; for (let i = idx(nt.t - .02); i < idx(end + .12); i++) { const t = tm(i) - nt.t, u = clamp(t / nt.d, 0, 1.2), z = (u - .30) / .26, g = Math.exp(-z * z), tail = smooth((u - .74) / .26); gdb[i] += (2.4 + .6 * bl) * g - (nt.x.cresc ? .4 : nt.end ? 4.4 : 2.4) * tail + 1.2 * nz[4](tm(i) + 11, 3.1); xadd[i] += -(.004 + .004 * bl) * g + (nt.x.cresc ? .004 : nt.end ? .030 : .014) * tail + .005 * nz[5](tm(i) + 5, 2.6); evo[i] += .16 * bl * smooth(t / .22); if (bl) { const q = nz[6](tm(i) + 3, 3.3); if (q > .55 && u < .7) raspE[i] = Math.max(raspE[i], (st.rasp ?? 0) * 1.1 * clamp((q - .55) * 2, 0, 1) * smooth(t / .1)); } } }
    if (nt.x.airTail) for (let i = idx(nt.t + nt.d * .42); i < n; i++) { const u = smooth((tm(i) - nt.t - nt.d * .42) / (nt.d * .42)); vfade[i] = Math.min(vfade[i], 1 - .96 * u * nt.x.airTail); xadd[i] += .09 * u; }
    if (nt.start && st.slowAtk) for (let i = idx(nt.t - .3); i < idx(nt.t + st.slowAtk); i++) gdb[i] -= 10 * (1 - smooth((tm(i) - nt.t + .05) / st.slowAtk));
    if (nt.x.cresc) for (let i = idx(nt.t); i < idx(end + .3); i++) { const u = clamp((tm(i) - nt.t) / nt.d, 0, 1); eadd[i] += nt.x.cresc * ss5(u); }
    else if (nt.d > .6 && (st.swell ?? 1)) for (let i = idx(nt.t); i < idx(end + .3); i++) { const u = clamp((tm(i) - nt.t) / nt.d, 0, 1); eadd[i] += (st.swell ?? 1) * (.07 * Math.sin(Math.PI * Math.min(1, u * 1.15)) - .05 * u * u); }
    if (nt.x.g) for (let i = idx(nt.t - .03); i < idx(end + .25); i++) gx[i] += nt.x.g * smooth((tm(i) - nt.t + .03) / .03) * (nt.end ? 1 : 1 - smooth((tm(i) - end + .03) / .04));
    { const cg = nt.x.cg ?? st.cg ?? 0; if (cg && nt.c && nt.c0 !== undefined) { const a = nt.c0 - .012, b = nt.t + (nt.c === 'm' || nt.c === 'n' ? -.006 : nt.c === 'k' || nt.c === 't' || nt.c === 'h' ? .004 : -.002); for (let i = idx(a - .02); i < idx(b + .02); i++) gx[i] += cg * smooth((tm(i) - a + .012) / .016) * (1 - smooth((tm(i) - b) / .014)); } }
    if (nt.x.air) for (let i = idx(nt.t - .10); i < idx(end + .3); i++) { const w = smooth((tm(i) - nt.t + .10) / .08); aspAdd[i] += nt.x.air * w; xadd[i] += .012 * nt.x.air * w; }
    if (nt.x.eff) for (let i = idx(nt.t - .03); i < idx(end); i++) eadd[i] += nt.x.eff * smooth((tm(i) - nt.t + .03) / .06) * (1 - smooth((tm(i) - end + .05) / .05));
    const rs = nt.x.rasp ?? (acc >= .6 ? (st.rasp ?? 0) : (st.rasp ?? 0) * .25); if (rs) for (let i = idx(nt.t - .02); i < idx(nt.t + Math.min(nt.d, .6)); i++) { const t = tm(i) - nt.t; raspE[i] = Math.max(raspE[i], rs * smooth((t + .02) / .04) * Math.exp(-Math.max(0, t - .03) / (nt.x.raspT ?? .11)) * (.75 + .5 * Math.abs(nz[6](tm(i), 14)))); }
    if (nt.x.flip) for (const [t0, t1, cst, xo] of nt.x.flip) for (let i = idx(nt.t + t0 - .03); i < idx(nt.t + t1 + .05); i++) { const u = smooth((tm(i) - nt.t - t0 + .012) / .024) * (1 - smooth((tm(i) - nt.t - t1) / .035)); cents[i] += cst * u; xadd[i] += xo * u; gdb[i] -= 3 * u; }
    if (nt.x.fry) { const L = nt.x.fry; for (let i = idx(nt.t - L - .02); i < idx(nt.t + .03); i++) { const t = tm(i) - nt.t, u = smooth((t + L + .02) / .02) * (1 - smooth((t + .012) / .036)); cents[i] = lerp(cents[i], 100 * (midi('C2') + 1.5 * nz[6](tm(i), 20)), u); gate[i] = Math.max(gate[i], .55 * u); s.jit[i] += 8 * u; xadd[i] -= .02 * u; fryE[i] = Math.max(fryE[i], u); } }
    if (nt.start && !nt.c && (st.hard ?? .5)) for (let i = idx(nt.t - .09); i < idx(nt.t + .05); i++) { const t = tm(i) - nt.t, z = (t + .02) / .03; xadd[i] -= .018 * (st.hard ?? .5) * Math.exp(-z * z); }
  });
  // breath before each phrase
  const endE = F(0); notes.forEach(nt => { if (nt.end) for (let i = idx((nt.offT ?? nt.t + nt.d) - .15); i < idx(nt.t + nt.d + .4); i++) endE[i] = 1; });
  const brE = F(0), brs = []; notes.forEach((nt, k) => { if (!nt.start) return; const pr = notes[k - 1], room = pr ? nt.c0 - (pr.offT + .10) : o.prevEnd !== undefined ? Math.min(.95, nt.c0 - o.prevEnd - .04) : .95, L = Math.min(st.brLen ?? .38, room - .05); if (L > .12 && (st.breathIn ?? 1) > 0 && !nt.x.nobreath && !o.nobreath) { const gp = nt.c === '' ? .085 : .04; brs.push([nt.c0 - L - gp, nt.c0 - gp, st.breathIn ?? 1]); } });
  for (const [a, b, g] of brs) for (let i = idx(a); i <= idx(b); i++) { const u = clamp((tm(i) - a) / (b - a), 0, 1); brE[i] = g * Math.pow(Math.max(0, Math.sin(Math.PI * Math.pow(u, 1.35))), .8) * (.82 + .18 * nz[2](tm(i), 9)); }
  // pressure: on before the first sound, off with the release
  const pg = F(0), adg = F(0); { let k = 0; const ph = []; notes.forEach((nt, i) => { if (nt.start) ph.push({ a: (nt.c0 ?? nt.t), b: null }); if (nt.end) { ph[ph.length - 1].b = nt.offT ?? nt.t + nt.d; ph[ph.length - 1].r = nt.x.rel ?? st.rel ?? .11; } }); for (const p of ph) { const fst = notes.find(q => q.start && (q.c0 ?? q.t) === p.a), vow = fst && !fst.c, vl = fst && fst.c && !VOICED.has(fst.c), a0 = vl ? (fst.c1 > p.a ? fst.c1 : fst.t) - .022 : p.a - .15, al = vl ? .030 : .05, on = vow ? .045 : .085, lead = vow ? .048 : .10; for (let i = idx(p.a - .15); i <= idx(p.b + p.r + .25); i++) adg[i] = Math.max(adg[i], smooth((tm(i) - a0) / al) * (1 - smooth((tm(i) - p.b - p.r - .1) / .15))); for (let i = idx(p.a - .16); i <= idx(p.b + p.r + .16); i++) { const t = tm(i), u = smooth((t - p.a + lead) / (st.pOn ?? on)), d = 1 - smooth((t - p.b + .02) / (p.r + .10)); pg[i] = Math.max(pg[i], u * d); } } }
  const life = st.life ?? 1, ab0 = x => x;
  for (let i = 0; i < n; i++) {
    const t = tm(i), e = clamp(eff[i] + eadd[i], .04, 1.5), g = gate[i], f0 = 440 * Math.pow(2, (cents[i] / 100 - 69) / 12), m = cents[i] / 100, hv = smooth((m - (st.reg ?? 64)) / 10);
    const dbv = gdb[i] + life * (.9 * nz[0](t + 50, 2.3) + .7 * nz[1](t + 70, 5.5)) + .9 * vibPhase[i], Pcm = (4.2 + 4.5 * e + 13 * e * e) * Math.pow(10, dbv / 34) * Math.pow(2, clamp((m - 57) / 12, -1, 1.4) * .30) * (st.pK ?? 1);
    s.f0[i] = f0; s.eff[i] = clamp((f0 / fk - 200) / 215 * .95 + .07 * vibPhase[Math.max(0, i - 20)] + (e - .5) * .55 + (st.open ?? 0) + .16 * life * nz[2](t + 30, 1.9) + evo[i], 0, 1.98);
    const ab = WH ? 0 : abd[i], br = st.breath ?? .4, x0 = (st.x0 ?? .030) + .040 * br - .045 * smooth((e - .25) / .95) + xadd[i] + .012 * hv * (st.head ?? 1) + .010 * life * nz[5](t, 2.1) * .5 + (1 - Math.min(1, g)) * (st.softOn ?? .012 + .033 * endE[i]) * pg[i] ;
    const lv = (.060 + .052 * Math.min(1.1, e)) * Math.pow(170 * fk / f0, .28) * Math.pow(10, clamp(gdb[i], -8, 5) / 60);
    s.xm[i] = WH ? 0 : lv * g * dipv[i] * vfade[i] * (o.airy ? .25 : 1) * (1 - .55 * smooth(ab / .16));
    s.x01[i] = x0 + ab * .9 + (o.airy ? .05 : 0); s.x02[i] = x0 + ab + (o.airy ? .06 : 0);
    s.P[i] = Pcm * CM * pg[i]; s.phz[i] = lerp(1.25, .45, hv); s.rasp[i] = raspE[i]; s.jit[i] *= (st.jit ?? 1) * (1 + .5 * (1 - g)); s.asp[i] = (st.asp ?? 1) * (o.airy ? 1.6 : 1) + .9 * smooth(ab / .15) + aspAdd[i]; s.tilt[i] = (st.tiltK ?? 1) * (2300 + 5200 * Math.min(e, 1.2)) * (1 - .25 * hv) * Math.pow(10, clamp(gdb[i], -8, 5) / 40);
    if (WH) { s.x01[i] = .026 + abd[i] * .6; s.x02[i] = .042 + abd[i] * .75; s.xm[i] = (st.murmur ?? .022) * g * dipv[i] * (1 - smooth(ab0(abd[i]) / .08)); s.P[i] *= 1.5; }
    if (brE[i] > 0.01 && pg[i] < .02) { s.P[i] = -(st.brP ?? 2.2) * CM * brE[i]; s.x01[i] = s.x02[i] = .30; s.xm[i] = 0; s.asp[i] = (st.brAsp ?? 1.2); }
    else if (adg[i] < .999) { s.x01[i] = lerp(.2, s.x01[i], adg[i]); s.x02[i] = lerp(.2, s.x02[i], adg[i]); }
  }
  if (kS !== 1) for (let i = 0; i < n; i++) { s.xm[i] *= kS; s.x01[i] *= kS; s.x02[i] *= kS; }
  { const G = [0, 5.5, 4.5, 1.2, 1.8], kb = st.balance ?? .55; for (let i = 0; i < n; i++) { let g = 0, sw = 1e-6; for (let v = 0; v < 5; v++) { const w = Math.max(0, s.w[v][i]); g += w * G[v]; sw += w; } s.gain[i] = Math.pow(10, (kb * g / sw + gx[i]) / 20); } }
  // the folds do not teleport: 12 ms muscles
  for (const k of ['x01', 'x02']) { let y = s[k][0], z = y; const g = 1 - Math.exp(-1 / (CR * .006)); for (let i = 0; i < n; i++) { y += (s[k][i] - y) * g; z += (y - z) * g; s[k][i] = z; } }
  { const g = 1 - Math.exp(-1 / (CR * .008)); let y = 0; for (let i = 0; i < n; i++) { y += (s.P[i] - y) * g; s.P[i] = y; } }
  return { s, t0: T0, notes };
}
