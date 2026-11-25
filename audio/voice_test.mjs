// node voice_test.mjs v1|v2 out.wav [styleJSON] [engineJSON] [phrase]
import { SR, writeWav, todb } from './voice_dsp.mjs';
import * as fx from './voice_fx.mjs';
import { STYLE0 } from './voice_test_styles.mjs';
const [, , eng, out, sj, ej, ph = 'kumo'] = process.argv, style = JSON.parse(sj || '{}'), eo = JSON.parse(ej || '{}');
export const PH = {
  kumo: 'ku A3 0 1|mo B3 1 1|ni D4 2 2|hi E4 4 1|to D4 5 1|ri B3 6 2|ho A3 8 1|shi B3 9 1|wo D4 10 2|mi F#4 12 1|te E4 13 1|ta D4 14 2.6',
  ame: 'a F#3 0 1.5|me G3 1.5 .5|ga A3 2 2|na B3 4 1|ma A3 5 1|e F#3 6 2|wo E3 8 2|yo F#3 10 2|bu D3 12 4',
  hawk: 'ko A3 0 1|bu C4 1 1|shi D4 2 1|de A3 3 1|ka F4 4 1|ta E4 5 1|re D4 6 4',
  go: 'go B4 0 4|o G#4 4 1|ku E4 6 1|mo F#4 7 1|wo G#4 8 2|nu B4 10 1|ke A4 11 1|te G#4 12 5',
};
const notes = PH[ph].split('|').map(q => { const [mora, pitch, t, d] = q.split(' '); return { mora, pitch, t: +t + 8, d: +d }; });
const P = eng === 'v1' ? await import('./v1/voice_perf.mjs') : await import('./voice_perf.mjs'); const st = { ...STYLE0[eng], ...style }; let y, start;
if (eng === 'v1') { const { synth } = await import('./v1/voice_core.mjs'), { Tract } = await import('./v1/voice_tract.mjs'), k = (60 / 144 / 2) / P.E8, b = P.buildLine(notes.map(n => ({ ...n, t: n.t * k, d: n.d * k })), st, { seed: 1 }), r = synth(b.s, { tract: new Tract(eo.Lt ?? 18.2), seed: 1009, hfj: .3, hfs: .5 }); y = r.y; }
else { const { sing } = await import('./voice_engine.mjs'), b = P.buildLine(notes, st, { seed: eo.seed ?? 1 }), t0 = Date.now(), r = sing(b.s, { seed: 5, ...eo }); y = r.y; console.error('ms', Date.now() - t0); }
let x = fx.eq(y, [['hp', 75, .7, 0], ...JSON.parse(process.env.EQ || '[]')]); let pk = 0; for (const v of x) pk = Math.max(pk, Math.abs(v)); x = x.map(v => v / pk * .7); if (process.env.DRY !== '1') { const rm = fx.room(x, 5, { n: 34, lp: 3500, len: .036 }), pl = fx.plate(x, { decay: .5, pre: .02, damp: 3000, hicut: 4600 }); x = x.map((v, i) => v + .05 * rm[0][i] + .12 * pl[0][i]); } pk = 0; for (const v of x) pk = Math.max(pk, Math.abs(v));
writeWav(out, [x.map(v => v * .8 / pk)], { float: false });
