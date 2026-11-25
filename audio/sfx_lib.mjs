// The effects library of GO: original synthesis, no samples.   import { SFX, render } from './sfx_lib.mjs'
//   render(name, { seed, size, pan, dist, level, dur, v, chord, ... }) -> St { l, r, n, sync }     reference level: -18 LUFS in the loudest 100 ms, peak <= -1.5 dBFS
//   `sync` = seconds from the first sample to the moment that must sit on the picture's frame
//   CLI:  node audio/sfx_lib.mjs list | demo [a,b,c] | one NAME out.wav '{"seed":2}'
export { SFX, render, CH, def } from './sfx_core.mjs';
import { SFX, render } from './sfx_core.mjs';
import './sfx_1.mjs';
import './sfx_2.mjs';
import './sfx_3.mjs';
import './sfx_4.mjs';
import './sfx_5.mjs';
import './sfx_6.mjs';
import { SR, N, St, writeWav, db } from './sfx_dsp.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url)), out = path.join(here, '..', 'out');
// open tops: physical things are allowed their splinters (the effects themselves keep it sparse: grit, cracks, sparks; never a steady hiss)
const TOP = { hit: 12500, kick: 12500, block: 12500, palm_catch: 12000, palm_shove: 11000, launch: 12000, clash_far: 9500, armour_shatter: 14500, steel_crack: 13000, steel_tip: 9000, gone: 10000, trench: 9500, pillar_break: 12500, weight_drop: 12000, boot_plant: 10000, wrath_step: 11500, storm_ring: 10000, blast_arrive: 13500, the_blow: 13500, deck_boom: 9500, moon_crack: 13500, column_fall: 9500, column_rise: 9000, water_skim: 10000, spray_fan: 10000, cape_shred: 9500, ground_plates: 10500, plates_lift: 10500, rain_cone: 10000, flick: 12000, cloud_burst: 9500, sea_dish: 8000, throw_off: 8000, tail_whip: 8500, pass_lens: 9000, splash: 9000, stones_fall: 9500, beam_step: 11000, ground_tremor: 5200, steel_fall: 9500, electric: 14000, water_splash: 9500 };
for (const [k, v] of Object.entries(TOP)) if (SFX[k]) SFX[k].hi = v;
export const VAR = { chibi_step: [0, 1, 2, 3, 4, 5, 6].map(shoe => ({ shoe })), sticker_pop: [0, 1, 2, 3, 4, 5, 6, 7].map(v => ({ v })), hit: [0, 1, 2, 3, 4, 5].map(w => ({ w })), kick: [0, 1, 2, 3, 4, 5].map(w => ({ w })), rain: [0, 1, 2, 3, 4].map(v => ({ v })), boot_plant: [0, 1].map(v => ({ v })), beam_hold: ['goku', 'vegeta', 'frieza', 'both'].map(who => ({ who })), vanish: ['goku', 'vegeta', 'frieza'].map(who => ({ who })), block: [{ v: 0 }, { v: 1 }, { v: 2 }, { v: 3 }], aura_loop: ['goku', 'vegeta'].map(who => ({ who })), stomach: [0, 1].map(v => ({ v })), star_ignite: [0, 3].map(step => ({ step })), aura_flash: [0, 4].map(step => ({ step })), deck_boom: [0, .5, 1].map(alt => ({ alt })), hands_shimmer: [0, 3].map(step => ({ step })) };
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [cmd, a, b, c] = process.argv.slice(2);
  if (cmd === 'list') { for (const m of Object.values(SFX)) console.log(m.name.padEnd(17), String(m.lv).padStart(4), m.d); console.log(Object.keys(SFX).length, 'effects'); }
  else if (cmd === 'one') { const s = render(a, c ? JSON.parse(c) : {}); writeWav(b, s.l, s.r, 16); console.log(a, (s.n / SR).toFixed(2) + 's'); }
  else if (cmd === 'demo') {
    const all = [], rows = []; let t = .5; const names = a ? a.split(',') : Object.keys(SFX); for (const name of names) (VAR[name] ?? [{}]).forEach((o, i, L) => { const s = render(name, o); all.push([t, s]); rows.push(`${(t + s.sync).toFixed(2).padStart(7)}  ${(L.length > 1 ? `${name} #${i}` : name).padEnd(20)} ${SFX[name].hear}`); t += s.n / SR + .45; });
    const st = new St(t + .5, Float32Array); for (const [at, s] of all) st.addSt(s, { at, g: 1 }); fs.mkdirSync(out, { recursive: true }); const f = b ?? path.join(out, 'sfx_demo.wav'); writeWav(f, st.l, st.r, 16); fs.writeFileSync(f.replace(/wav$/, 'txt'), 'seconds  effect (the time is the moment that sits on the picture frame)\n' + rows.join('\n') + '\n'); console.log(rows.length, 'entries,', t.toFixed(1), 's');
  }
}
