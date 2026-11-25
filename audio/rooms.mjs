// Scene rooms of GO: synthetic impulse responses (early reflections placed from the set + a diffuse tail per octave band).   getRoom(name) -> { l, r }
import { room } from './sfx_dsp.mjs';
export const ROOMS = {
  glass:   { t60: .55, bands: { 63: .4, 125: .5, 2000: 1.1, 4000: .7 }, pre: .001, hicut: 6400, locut: 160, early: [[.9, .3, -.5], [1.7, .24, .5], [2.9, .18, -.2], [4.6, .12, .3]] },      // a1: inside a sphere the size of a fist
  air:     { t60: 1.4, tail: .4, bands: { 63: .6, 125: .7, 4000: .4, 8000: .2 }, pre: .06, build: .25, hicut: 3000, locut: 140, early: [[410, .04, -.6, 1200], [620, .03, .6, 1000]] },     // a2-a4, h4: above cloud, nothing near
  beach:   { t60: .35, tail: .25, bands: { 4000: .5 }, pre: .006, hicut: 4200, early: [[5, .16, -.4, 3600], [11, .1, .5, 3000]] },                                                        // b1
  toy:     { t60: .3, bands: { 125: .6 }, pre: .002, hicut: 5200, early: [[2.8, .24, -.5], [5.1, .2, .5], [8, .12, 0]] },                                                                   // b2
  lake:    { t60: .9, tail: .3, bands: { 4000: .4 }, pre: .03, hicut: 3400, early: [[3, .12, 0, 3600], [290, .08, -.5, 1500], [430, .06, .6, 1200], [610, .035, -.2, 900]] },              // b3: hills answer
  kitchen: { t60: .36, bands: { 125: .9, 4000: .7 }, pre: .003, hicut: 5400, early: [[3.4, .28, -.5], [5.2, .26, .5], [8.1, .2, 0], [12.4, .14, -.2]] },                                      // b4: a wooden room
  yard:    { t60: .5, tail: .3, bands: { 4000: .5 }, pre: .01, hicut: 3600, early: [[9, .16, -.4, 3000], [24, .1, .6, 2400]] },                                                             // b5
  night:   { t60: 4.4, bands: { 63: 1.2, 125: 1.2, 1000: .75, 2000: .5, 4000: .3, 8000: .15 }, pre: .06, build: .2, hicut: 2800, early: [[120, .06, -.5, 1400], [210, .05, .6, 1100]] },    // c1: black sea, sky lid
  skull:   { t60: .22, bands: { 63: .6, 2000: .4, 4000: .2 }, pre: .001, hicut: 1400, early: [[1.3, .3, -.3, 1200], [2.2, .25, .3, 1000]] },                                                 // c2, c3, g3-g5: inside a head
  coast:   { t60: 2.0, bands: { 63: 1.1, 125: 1.15, 2000: .65, 4000: .4, 8000: .2 }, pre: .03, build: .04, hicut: 3600, early: [[38, .2, -.8, 2600], [61, .18, .8, 2400], [133, .12, -.4, 1600], [224, .09, .5, 1200]] },   // c4, d1, d2, d5: mesas
  sky:     { t60: 2.6, tail: .7, bands: { 63: 1.1, 4000: .35, 8000: .2 }, pre: .09, build: .3, hicut: 2800, early: [[310, .12, -.6, 1300], [470, .1, .6, 1100], [700, .07, -.2, 800]] },     // d3, g1, g2: clouds above a sea
  page:    { t60: .8, bands: { 125: 1.2, 4000: .4 }, pre: .006, hicut: 3600, early: [[7, .24, -.6, 3200], [12, .2, .6, 2800], [20, .14, 0, 2400]] },                                         // d4
  waste:   { t60: 2.6, bands: { 63: 1.2, 125: 1.2, 1000: .8, 2000: .5, 4000: .3 }, pre: .05, build: .1, hicut: 2600, early: [[71, .14, -.7, 1800], [118, .12, .7, 1500], [205, .08, 0, 1100]] }, // d6: grey waste, slowed time
  cliff:   { t60: .6, tail: .2, bands: { 4000: .4 }, pre: .02, hicut: 3000, early: [[3, .12, 0, 3000], [150, .04, .6, 1100]] },                                                              // e1, e2
  memory:  { t60: .55, bands: { 63: .6, 125: .9, 2000: .7, 4000: .45 }, pre: .004, hicut: 3000, early: [[4, .24, -.5, 2400], [7, .22, .5, 2200], [13, .15, 0, 1800]] },                       // e3
  crater:  { t60: 1.1, bands: { 125: 1.1, 2000: .7, 4000: .4 }, pre: .018, hicut: 3400, early: [[17, .18, -.6, 2800], [29, .15, .6, 2500], [52, .1, 0, 1800]] },                                // e4, f1
  storm:   { t60: 2.2, bands: { 63: 1.1, 125: 1.15, 2000: .6, 4000: .35, 8000: .2 }, pre: .026, build: .05, hicut: 3600, early: [[29, .18, -.7, 2600], [47, .16, .7, 2400], [99, .1, .1, 1600]] }, // f2-f4, g7, g8
  light:   { t60: 5.5, bands: { 63: .6, 125: .8, 4000: .6, 8000: .3 }, pre: .05, build: .5, hicut: 4200, locut: 180 },                                                                         // g9, f1: the suspended world
  town:    { t60: .9, tail: .4, bands: { 4000: .5 }, pre: .015, hicut: 3600, early: [[19, .14, -.6, 2600], [33, .12, .6, 2400]] },                                                            // g6
  dawn:    { t60: .8, tail: .25, bands: { 4000: .45 }, pre: .025, hicut: 3400, early: [[4, .1, 0, 3400], [240, .06, -.5, 1400], [360, .05, .5, 1100]] },                                       // h1-h3: crater rim far off
  hall:    { t60: 2.9, bands: { 63: 1.05, 125: 1.1, 250: 1.05, 2000: .75, 4000: .5, 8000: .28 }, pre: .012, build: .03, hicut: 6500, early: [[23, .12, -.6, 4000], [37, .1, .6, 3600]] },      // the last chord rings out
  paper:   { t60: .4, bands: { 125: .8, 4000: .8 }, pre: .003, hicut: 5200, early: [[4.1, .28, .4], [7.3, .2, -.5], [11.9, .14, .1]] },                                                        // h5
};
const cache = new Map();
export function getRoom(name) { if (!ROOMS[name]) throw new Error('no room ' + name); if (!cache.has(name)) { if (cache.size > 3) cache.delete(cache.keys().next().value); cache.set(name, room({ seed: name.length * 7 + name.charCodeAt(0), ...ROOMS[name] })); } return cache.get(name); }
