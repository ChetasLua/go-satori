// Chord of any moment (Hz, low to high) for effects that tune themselves.  Reads audio/score.json (harm: { t, d, sym } in eighths); falls back to a key map.
import fs from 'node:fs';
const P = new URL('./score.json', import.meta.url), SC = fs.existsSync(P) ? JSON.parse(fs.readFileSync(P)) : null, E8 = 60 / 144 / 2;
const PC = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 }, hz = m => 440 * Math.pow(2, (m - 69) / 12);
const FB = [[0, 'D'], [56, 'G'], [128, 'Dphr'], [176, 'Dm'], [256, 'Bm'], [312, 'D5'], [344, 'B'], [360, 'E'], [488, 'D']];
export function parse(sym) { const m = /^([A-G])([#b]?)(.*?)(?:\/.*)?$/.exec(sym); if (!m) return null; const r = (PC[m[1]] + (m[2] === '#' ? 1 : m[2] === 'b' ? -1 : 0) + 12) % 12, q = m[3], third = /phr/.test(q) ? 1 : /^m(?!aj)/.test(q) ? 3 : /sus4/.test(q) ? 5 : /sus2/.test(q) ? 2 : /^5/.test(q) ? 7 : 4, base = r <= 4 ? 48 + r : 36 + r; return [base, base + third, base + 7]; }
export function symAt(sec) { const e = sec / E8 + 1e-6; let s = null; if (SC?.harm) { for (const h of SC.harm) if (h.t <= e) { if (h.sym !== 'N') s = h.sym; } else break; } if (!s) for (const [t, q] of FB) if (t <= e) s = q; return s; }
export function chordAt(sec) { const n = parse(symAt(sec)) ?? [50, 54, 57]; return n.map(hz).sort((a, b) => a - b); }
export const note = p => { const m = /^([A-G])([#b]?)(-?\d)$/.exec(p); return hz(12 * (+m[3] + 1) + PC[m[1]] + (m[2] === '#' ? 1 : m[2] === 'b' ? -1 : 0)); };
export const BELLS = (SC?.bells ?? []).reduce((o, b) => ((o[b.shot] ??= []).push(note(b.pitch)), o), {});
export const SCORE = SC;
