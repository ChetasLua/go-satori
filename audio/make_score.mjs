// make_score.mjs : the song of GO.  node audio/make_score.mjs  -> audio/score.json, then prints the checks (range, breath, chord collisions).
// 144 BPM, 70 bars; t and d are in 8th notes (1 bar = 8, 1 eighth = 5 frames = .208333 s).  Melody, harmony and words are ours.
// THE SEED: two notes that fall a step (吾, "I"), answered by the same two notes a fourth higher (忄, the heart).
//   question (four stars)  mi re LA SOL  = F# E B A      hangs on the fifth
//   answer   (GO)          la sol RE DO  = B A E D       comes home.    In the chorus (E major): C# B F# E.
// EGO (d4) is only the first two notes, stuck.  行け (d6) is the leap of a fifth that the ego never made.  か め は め = the four stars as four chords, は = home.
import fs from 'node:fs';
const url = new URL('./score.json', import.meta.url);
const PC = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const midi = s => { const m = /^([A-G])([#b]?)(-?\d)$/.exec(s); if (!m) throw Error('pitch ' + s); return PC[m[1]] + (m[2] === '#' ? 1 : m[2] === 'b' ? -1 : 0) + 12 * (+m[3] + 1); };
// ---------------------------------------------------------------- harmony: one string per bar, "sym:eighths"
const H = [
  /*0 a1*/ 'Dpent:8', 'Dpent:8', /*2 a2*/ 'D:8', 'Gmaj7:4 D/F#:4', /*4 a3*/ 'Em7:4 A7sus4:4', 'Bb:1.2 C:1.2 A7:.8 D:4.8', /*6 a4*/ 'G:4 D7:4',
  /*7 b1*/ 'G:4 C6:2 G:2', 'G:2 C6:2 D7:2 G:2', /*9 b2*/ 'G:2 Cm:2 F:2 A7:2', 'D7:2 N:2 E7:2 G:2', /*11 b3*/ 'Cmaj7:6 D6:2', 'G:4 A7:1.2 N:2.8', /*13 b4*/ 'G:4 Em:4', 'N:2 Em:2 B7:1.2 N:2.8', /*15 b5*/ 'G:3 C6:5',
  /*16 c1*/ 'Dphr:8', 'Dphr:8', /*18 c2*/ 'N:8', /*19 c3*/ 'Ebmaj7/D:8', /*20 c4*/ 'N:8', 'N:4 Dm/Eb:4',
  /*22 d1*/ 'D5:8', /*23 d2*/ 'Dm:4 C:4', 'Bb:4 C:4', /*25 d3*/ 'Dm:4 C/E:2 F:2', 'Dm:4 A5:2 N:2', /*27 d4*/ 'N:8', /*28 d5*/ 'Bb/D:8', 'Gm/D:4 N:4', /*30 d6*/ 'Dm:8', 'D5:4 N:4',
  /*32 e1*/ 'Bm11:8', 'Gmaj7:8', /*34 e2*/ 'D/F#:4 A6:2 G:2', /*35 e3*/ 'D:2 G:2 A7:2 D:2', 'G:4 D/A:4', /*37 e4*/ 'Gmaj7:4 Em:2 Bm/D:2', 'Gmaj7:4 Bm:4',
  /*39 f1*/ 'N:8', /*40 f2*/ 'B5:4 C5:4', 'C#5:2 D5:2 D#5:4', /*42 f3*/ 'Em:4 C:2 D:2', /*43 f4*/ 'Em:4 Am:4', 'B:4 Am:2 B7:1.2 N:.8',
  /*45 g1*/ 'N:4 E:4', 'A:4 Bsus4:2 C#m:2', /*47 g2*/ 'C#m:4 E/G#:4', 'A6:4 E/B:2 B:2', /*49 g3*/ 'E:8', 'Amaj7#11:8', /*51 g4*/ 'C#m9:4 F#m7:4', 'Amaj9:8', /*53 g5*/ 'Am:4 D:2 Am:2', /*54 g6*/ 'C#m:2 E/G#:2 A:2 B:2',
  /*55 g7*/ 'E:4 D6:4', 'A:4 Bsus4:4', /*57 g8*/ 'C:4 D6:4', 'E:6 B:2', /*59 g9*/ 'E:8', 'N:8',
  /*61 h1*/ 'N:8', 'N:4 Gmaj7:2 A7sus4:2', /*63 h2*/ 'Dpent:8', 'G:4 A7sus4:4', /*65 h3*/ 'D:4 G:2 D:2', 'Bm:4 G:2 A:2', /*67 h4*/ 'G:4 D/F#:4', 'Em7:2 A7sus4:2 D:4', /*69 h5*/ 'D:8'];
if (H.length !== 70) throw Error('bars ' + H.length);
const harm = []; H.forEach((s, b) => { let t = b * 8; for (const q of s.split(' ')) { const [sym, d] = q.split(':'); harm.push({ t, d: +d, sym }); t += +d; } if (Math.abs(t - (b + 1) * 8) > 1e-6) throw Error('bar ' + b + ' length ' + (t - b * 8)); });
// ---------------------------------------------------------------- lines: id, bar, voice, style, ja, en, "mora:pitch:eighths"  (r = rest, Q = small tsu, N = moraic n or hum, A>B = glide, trailing {..} = divisi below)
const L = [
  // A  the boy
  ['a1', 1, 'lead', 'hum p child', 'ん…', '', 'r:2 N:B4:2 N:A4:2 N:E5:2 N:D5:6 r:2'],
  ['a2', 3, 'lead', 'p child, close, alone; falls away at the end', 'くもに ひとり', 'Alone upon a cloud', 'ku:B4:1 mo:A4:1 ni:E5:2 hi:D5:1 to:B4:1 ri:A4:1.6 r:.4'],
  ['a3', 4, 'lead', 'mp child, grin; the last note tips over the edge with the Nimbus', 'ほしを みてた', 'I was watching a star', 'ho:B4:1 shi:A4:1 o:F#5:2 mi:E5:1 te:D5:1 ta:A4>D4:3.5 r:6.5'],
  ['a4', 6, 'lead', 'f whoop, slow start, fast rise', 'おー！', 'Whoa!', 'o:D5>A5:3.2 r:4.8'],
  // B  the family
  ['b1a', 7, 'kids', 'mf marching, swung, cheeky', 'てくてく いこう', 'Trot, trot, here we go', 'te:B4:1.333 ku:A4:.667 te:E5:1.333 ku:D5:.667 i:B4:1.333 ko:A4:.667 u:G4:2'],
  ['b1b', 8, 'kids', 'f marching, swung, proud last note', 'みんなが いるぞ', 'Everyone is here', 'mi:B4:1.333 N:A4:.667 na:E5:1.333 ga:D5:.667 i:F#5:1.333 ru:A5:.667 zo:G5:1.8 r:.2'],
  ['b2', 10, 'kids', 'f shouted more than sung; "te" lands on the grid (b2 f70) and is cut by the cold note', 'ほら わらって！', 'Come on, smile!', 'r:1 ho:D5:1 ra:B4:1.5 r:.5 wa:B4:1 ra:D5:.6 Q:.4 te:G5:1.2 r:.8'],
  ['b3a', 11, 'lead', 'mp warm chest, a father', 'おおきな てと', 'A big hand and', 'o:B3:1 o:A3:1 ki:E4:1 na:D4:1 te:D4:1 to:B3:1 r:2'],
  ['b3b', 11, 'kids', 'p one child (solo), same tune an octave up', 'ちいさな て', 'a small hand', 'r:6 chi:B4:1 i:A4:1 sa:E5:1 na:D5:1 te:G5:1.8 r:4.2'],
  ['b4a', 13, 'lead', 'mf comic, conspiratorial, clipped', 'さいごの ひとつ', 'The very last one', 'r:1 sa:B4:1 i:B4:1 go:C5:1 no:B4:1 hi:G4:1 to:F#4:1 tsu:E4:1'],
  ['b4b', 14, 'lead', 'mp sly, one eyebrow; cut dead when the lamp goes (b4 f66)', 'だれの もの', 'Whose is it?', 'r:2 da:E4:.5 re:G4:.5 no:B4:1 mo:B4:.5 no:C5:.7 r:2.8'],
  ['b5', 15, 'lead', 'pp, slowing; cut dead at b5 f24', 'また あした', 'See you tomorrow', 'ma:B4:.75 ta:A4:.75 a:E5:.9 shi:D5:.95 ta:A4:1.4 r:3.25'],
  // C  fear
  ['c1', 16, 'lead', 'pp low, no vibrato, almost no air', 'そらが とじる', 'The sky is closing', 'r:1 so:D4:1 ra:D4:1 ga:Eb4:3 r:1 to:C4:1 ji:Bb3:1 ru:A3:4.5 r:2.5'],
  ['c2', 18, 'whisper', 'ppp between two heartbeats', 'しずかだ', 'It is so quiet', 'r:2.6 shi:D4:.6 zu:D4:.6 ka:D4:.6 da:D4:1.2 r:2.4'],
  // D  clash
  ['d1', 22, 'lead', 'f through the teeth, then open', 'ひざよ わらうな', 'Knees, stop shaking', 'r:4 hi:A4:1 za:A4:1 yo:D5:2 wa:C5:1 ra:D5:1 u:F5:2 na:E5:3 r:1'],
  ['d2', 24, 'lead', 'ff', 'まえへ まえへ', 'Forward! Forward!', 'ma:D5:1 e:C5:1 e:G5:2 ma:F5:1 e:E5:1 e:G5:1.7 r:.3'],
  ['d3', 25, 'lead', 'ff every eighth is a clash, last note held into the flurry', 'こぶしで かたれ', 'Let your fists do the talking', 'ko:F5:1 bu:F5:1 shi:G5:1 de:F5:1 ka:E5:1 ta:F5:1 re:A5:6.8 r:3.2'],
  ['d4', 27, 'lead', 'fff belt, pride; unison with the whole orchestra', 'おれは おれだ', 'I am I!', 'o:F5:1 re:E5:1 wa:E5:1.8 r:.2 o:F5:1 re:E5:1 da:E5:1.8 r:.2'],
  ['d6', 31, 'lead', 'fff one long cry, cut dead at d6 f60', 'いけ！', 'Go!', 'r:1.2 i:D5:.5 ke:A5:2.3 r:4'],
  // E  sadness
  ['e1', 32, 'lead', 'pp close, small, almost spoken', 'あめが なまえを よぶ', 'The rain is calling a name', 'r:2 a:F#4:1 me:F#4:1 ga:E4:2 na:D4:1 ma:E4:1 e:D4:1 o:C#4:1 yo:D4:1 bu:B3:2.4 r:2.6'],
  ['e2', 33, 'lead', 'pp', 'たったまま ねむる ひと', 'A man asleep on his feet', 'r:6 ta:B3:.5 Q:.5 ta:D4:1 ma:E4:1 ma:F#4:1 ne:A4:1 mu:G4:1 ru:F#4:1 hi:E4:1 to:D4:1.8 r:.2'],
  ['e3', 35, 'lead', 'p tender, a lullaby: the bun tune of b4 in major', 'さいごの ひとつ くれた ひと', 'The man who gave me the last one', 'sa:A4:1 i:A4:1 go:B4:1 no:A4:1 hi:F#4:1 to:E4:1 tsu:D4:1.7 r:.3 ku:G4:1 re:A4:1 ta:B4:2 hi:A4:1 to:F#4:1.8 r:1.2'],
  ['e4a', 37, 'lead', 'pp breaking', 'きづかなかった', 'I never noticed', 'ki:F#4:1 zu:F#4:1 ka:G4:1 na:F#4:1 ka:E4:1 Q:.6 ta:D4:2.1 r:.3'],
  ['e4b', 38, 'lead', 'ppp breaking, last note nearly without voice', 'ごめんな', "I'm sorry", 'r:1 go:D4:1 me:E4:1 N:D4:1 na:B3:3 r:1'],
  // F  wrath
  ['f2a', 40, 'lead', 'spoken-sung, low, shaking, between the flashes', 'ゆるさない', "I won't forgive you", 'r:1 yu:B3:.8 ru:B3:.8 sa:C4:1 r:1.4 na:C4:.8 i:B3:1.6 r:.6'],
  ['f2s', 41, 'scream', 'fff vocal-tract scream, from f2 f60 through f3 f20', '', '', 'r:4 a:B4>B5:4 a:B5>E5:4 r:4'],
  ['f4', 43, 'lead', 'ff heavy, one step per beat', 'いっぽ いっぽ', 'One step. One step.', 'i:E4:1 Q:1 po:G4:1.8 r:.2 i:E4:1 Q:1 po:A4:1.8 r:.2 i:B4:1 Q:1 po:B4:1.8 r:.2 i:C5:1 Q:1 po:D#5:1.2 r:.8'],
  // G  triumph
  ['g1', 45, 'lead', 'fff, the hook', 'ゴー！ くもを ぬけて', 'GO! Break through the clouds', 'r:4 go:E5:3.8 r:.2 ku:C#5:1 mo:B4:1 o:F#5:2 nu:E5:1 ke:F#5:1 te:G#5:5.6 r:.4'],
  ['g2', 47, 'lead', 'fff, the hook, higher', 'ゴー！ ほしを こえて', 'GO! Go beyond the stars', 'r:4 go:G#5:3.8 r:.2 ho:C#5:1 shi:B4:1 o:F#5:2 ko:E5:1 e:G#5:1 te:B5:2'],
  ['g3', 49, 'lead', 'ppp breath and voice, one mora on each bell', 'ちからを ぬけば', 'Let all your strength go', 'r:2 chi:G#4:2 ka:F#4:2 ra:C#5:2 o:B4:2 nu:E5:1 ke:D#5:1 ba:B4:3.6 r:.4'],
  ['g4', 51, 'lead', 'pp head voice, weightless', 'かぜに なる', 'and you become the wind', 'ka:C#5:2 ze:B4:2 ni:F#5:4 na:E5:2 ru:E5:4 r:2'],
  ['g5', 53, 'lead', 'f dark, weight', 'つきが おちる', 'The moon is falling', 'tsu:C5:1 ki:C5:1 ga:E5:2 o:D5:1 chi:C5:1 ru:A4:1.8 r:.2'],
  ['g6', 54, 'lead', 'mf rising to ff', 'てを あげろ', 'Raise your hands', 'te:E5:1 o:F#5:1 a:G#5:2 ge:A5:1 ro:B5:2.8 r:.2'],
  ['g7', 55, 'lead', 'ff with everyone, each syllable louder', 'か　め　は　め', 'KA  ME  HA  ME', 'ka:G#4:3.7 r:.3 me:F#4:3.7 r:.3 ha:C#5:3.7 r:.3 me:B4:3.8 r:.2'],
  ['g8', 57, 'lead', 'fff the longest note of the film, with everyone', 'は—', 'HAAA—', 'ha:E5:8 a:G#5:6 a:B5:6 r:4'],
  // H  dawn
  ['h1', 62, 'lead', 'spoken, comic, flat on the back', 'はらへった', "I'm hungry.", 'r:1 ha:E4:.6 ra:F#4:.6 he:F#4:.6 Q:.4 ta:D4:1.2 r:3.6'],
  ['h2', 63, 'kids', 'p one child (solo), one number on each bell', 'ひとつ ふたつ みっつ よっつ', 'One... two... three... four.', 'r:2 hi:F#4:.6 to:F#4:.6 tsu:F#4:.7 r:.1 fu:E4:.6 ta:E4:.6 tsu:E4:.7 r:.1 mi:B4:.6 Q:.5 tsu:B4:.8 r:.1 yo:A4:.6 Q:.5 tsu:A4:2.4 r:4.5'],
  ['h3', 65, 'lead', 'mp smiling: the bun tune again, and this time it goes up', 'はんぶんこ', 'Half for you, half for me', 'r:2 ha:A4:1 N:A4:1 bu:B4:1 N:A4:1 ko:D5:4 r:6'],
  ['h4a', 67, 'lead', 'mf everybody: the tune of a2, and this time it goes up', 'くもに みんな', 'Everyone upon a cloud', 'ku:B4:1 mo:A4:1 ni:E5:2 mi:D5:1 N:E5:1 na:F#5:1.8 r:.2'],
  ['h4b', 68, 'lead', 'mf laughing: the answer, home', 'わらって た', 'and we were laughing', 'wa:B4:1 ra:A4:.5 Q:.5 te:E5:1 ta:D5:2.6 r:2.4'],
  ['h5', 68, 'lead', 'ppp hum, then one breathed "go" as the needle leaves the sheet (h5 f46)', '悟', 'GO', 'r:6 N:B4:1.9 r:.1 N:A4:1.9 r:3.3 go:D4:2.6 r:.2'],
  // choir and kids around the singer.  {..} = the other voices of the same chord
  ['b1k', 7, 'kids', 'second part, thirds and sixths under the tune', 'てくてく いこう みんなが いるぞ', '', 'te:G4:1.333 ku:F#4:.667 te:C5:1.333 ku:B4:.667 i:G4:1.333 ko:F#4:.667 u:D4:2 mi:G4:1.333 N:F#4:.667 na:C5:1.333 ga:B4:.667 i:D5:1.333 ru:F#5:.667 zo:D5:1.8 r:.2'],
  ['c1c', 16, 'choir', 'ppp low "o", no vibrato, basses; enters like weather', 'お', '', 'o:D3:16{A2}'],
  ['c3c', 19, 'choir', 'pp "o" cluster, cut dead at bar 20', 'お', '', 'o:D3:8{Eb3,A3}'],
  ['c4c', 21, 'choir', 'fff one blow at c4 f60', 'あ', '', 'r:4 a:D4:4{Eb3,A3,Ab4,D5}'],
  ['d2c', 24, 'choir', 'ff shouted hits on the three pillars and the throw', 'は', '', 'ha:D4:.6{D3,F3} r:1.4 ha:D4:.6{D3,G3} r:1.4 ha:C4:.6{C3,G3} r:1.4 ha:C4:1.6{C3,G3,E4} r:.4'],
  ['d3c', 25, 'choir', 'f held, swelling into the caught fists, cut dead at d3 f70', 'あ', '', 'a:D4:4{F3,A3} a:C4:2{G3,E4} a:C4:2{A3,F4} a:D4:4{A3,F4} a:E4:2{A3,A4} r:2'],
  ['d4c', 27, 'choir', 'fff octaves with the lead, men', 'おれは おれだ', '', 'o:F4:1{F3} re:E4:1{E3} wa:E4:1.8{E3} r:.2 o:F4:1{F3} re:E4:1{E3} da:E4:1.8{E3} r:.2'],
  ['d5c', 28, 'choir', 'ff "a" inside the beam: held to d5 f30, then each of his two steps (f30, f40) takes a bite out of it; gone at d5 f50', 'あ', '', 'a:D4:5.8{Bb2,F3,Bb3,F4} r:.2 a:D4:1.8{Bb2,F3,Bb3} r:.2 a:D4:1.8{G2,G3} r:6.2'],
  ['e3c', 35, 'kids', 'ppp far away, "lu"-like u, the memory', 'う', '', 'u:F#4:2 u:G4:2 u:E4:2 u:F#4:2 u:D5:4 u:D5:2 r:2'],
  ['f2c', 40, 'choir', 'clusters climbing by semitones, one on each flash (f2 f0 20 40 50 60): each louder', 'あ', '', 'a:B3:3.6{B2,C3,F#3} r:.4 a:C4:3.6{C3,C#3,G3} r:.4 a:C#4:1.8{C#3,D3,G#3,C#4} r:.2 a:D4:1.8{D3,Eb3,A3,D4} r:.2 a:D#4:4{D#3,E3,A#3,D#4,F#4}'],
  ['f3c', 42, 'choir', 'fff full', 'あ', '', 'a:E4:4{E3,G3,B3,G4,B4} a:E4:2{C3,G3,C4,G4} a:D4:1.8{D3,A3,F#4,A4} r:.2'],
  ['f4c', 43, 'choir', 'ff men, "ho", one giant step per beat', 'ほ', '', 'ho:E3:1.6{E2,B2} r:.4 ho:E3:1.6{E2,B2} r:.4 ho:A3:1.6{A2,E3} r:.4 ho:A3:1.6{A2,E3} r:.4 ho:B3:1.6{B2,F#3} r:.4 ho:B3:1.6{B2,F#3} r:.4 ho:C4:1.6{A2,E3} r:.4 ho:B3:1.2{B2,F#3,D#4} r:.8'],
  ['g1c', 45, 'choir', 'fff', 'ゴー', '', 'r:4 go:B4:3.8{E3,B3,E4,G#4} r:.2 a:C#5:4{A3,E4,A4} a:B4:2{B3,E4,F#4} a:C#5:2{C#4,G#4,E4} a:C#5:3.6{C#4,G#4,E4} r:.4'],
  ['g2c', 47, 'choir', 'fff', 'ゴー', '', 'r:4 go:B4:3.8{G#3,B3,E4,G#4} r:.2 a:C#5:4{A3,E4,F#4} a:B4:2{B3,E4,G#4} a:B4:2{B3,D#4,F#4}'],
  ['g4c', 51, 'choir', 'ppp "u" like air', 'う', '', 'u:E4:4{G#4,B4} u:E4:4{A4,C#5} u:E4:6{A4,C#5,G#4} r:2'],
  ['g5c', 53, 'choir', 'f low "o", weight', 'お', '', 'o:A3:4{A2,E3,C4} o:F#3:2{D3,A3,D4} o:A3:2{A2,E3,C4}'],
  ['g6c1', 54, 'choir', 'one voice: mp, enters on the first hand', 'あ', '', 'a:C#4:2 a:B3:2 a:C#4:2 a:D#4:2'],
  ['g6c2', 54, 'choir', 'enters on the second hand (beat 2), mf', 'あ', '', 'r:2 a:E4:2 a:E4:2 a:F#4:2'],
  ['g6c3', 54, 'kids', 'enters on the third hand (beat 3), f', 'あ', '', 'r:4 a:A4:2 a:B4:2'],
  ['g6c4', 54, 'choir', 'everyone (beat 4), ff', 'あ', '', 'r:6 a:B4:2{B2,F#3,D#4}'],
  ['g7c', 55, 'choir', 'ff everyone in the world, four chords', 'か　め　は　め', '', 'ka:G#4:3.7{E3,B3,E4,B4} r:.3 me:F#4:3.7{D3,A3,D4,B4} r:.3 ha:C#5:3.7{A3,E4,A4} r:.3 me:B4:3.8{B3,E4,F#4} r:.2'],
  ['g7k', 55, 'kids', 'ff Gohan and the children an octave above', 'か　め　は　め', '', 'ka:G#5:3.7 r:.3 me:F#5:3.7 r:.3 ha:E5:3.7 r:.3 me:F#5:3.8 r:.2'],
  ['g8c', 57, 'choir', 'fff', 'は', '', 'ha:G4:4{C3,G3,C4,E4,C5} a:F#4:4{D3,A3,D4,B4} a:G#4:6{E3,B3,E4,B4} a:F#4:2{B2,F#3,D#4,B4} a:G#4:4{E3,B3,E4,B4} r:4'],
  ['g8k', 57, 'kids', 'fff', 'は', '', 'ha:G5:4 a:F#5:4 a:B5:6 a:D#5:2 a:E5:4 r:4'],
  ['h4c', 67, 'choir', 'mf everyone, happy, a little out of tune', 'くもに みんな わらってた', '', 'ku:G4:1{D4,B3} mo:G4:1{D4,B3} ni:G4:2{D4,B3} mi:A4:1{F#4,D4} N:A4:1{F#4,D4} na:A4:1.8{F#4,D4} r:.2 wa:G4:1{E4,B3} ra:G4:.5{E4,B3} Q:.5 te:A4:1{E4,D4} ta:F#4:2.6{D4,A3} r:2.4'],
  ['h4k', 67, 'kids', 'f everyone', 'くもに みんな わらってた', '', 'ku:B4:1 mo:A4:1 ni:E5:2 mi:D5:1 N:E5:1 na:F#5:1.8 r:.2 wa:B4:1 ra:A4:.5 Q:.5 te:E5:1 ta:D5:2.6 r:2.4'],
];
const out = [];
for (const [id, bar, voice, style, ja, en, str] of L) {
  let t = 0; const ns = [];
  for (const tok of str.split(' ')) {
    const m = /^(r|[A-Za-z]+):(?:([A-G][#b]?\d(?:>[A-G][#b]?\d)?):)?([\d.]+)(?:\{([^}]*)\})?$/.exec(tok); if (!m) throw Error(id + ' token ' + tok);
    const d = +m[3]; if (m[1] !== 'r') { const n = { mora: m[1], pitch: m[2] || '', t: +(bar * 8 + t).toFixed(4), d }; if (m[4]) n.div = m[4].split(','); ns.push(n); } t += d;
  }
  if (Math.abs(t / 8 - Math.round(t / 8)) > .002 && Math.abs(t % 4) > .01 && Math.abs(t % 4 - 4) > .01) throw Error(id + ' ends at ' + t);
  const last = ns[ns.length - 1], f0 = ns[0].t * 5, f1 = (last.t + last.d) * 5;
  out.push({ id, bar, voice, style, ja, en, f0: +f0.toFixed(1), f1: +f1.toFixed(1), notes: ns });
}
// a small-tsu borrows the pitch of the note it closes
for (const l of out) l.notes.forEach((n, i) => { if (!n.pitch) n.pitch = l.notes[i - 1].pitch.split('>').pop(); });
const bells = [];
const bell = (shot, f0, frames, ps) => frames.forEach((f, i) => bells.push({ shot, f: f0 + f, t: +((f0 + f) / 5).toFixed(3), pitch: ps[i] }));
bell('a1', 0, [10, 20, 30, 40], ['F#5', 'E5', 'B5', 'A5']); bell('g3', 1960, [10, 20, 30, 40], ['G#5', 'F#5', 'C#6', 'B5']); bell('h2', 2520, [10, 20, 30, 40], ['F#5', 'E5', 'B5', 'A5']); bell('h5', 2740, [10, 20, 34, 50], ['B5', 'A5', 'E6', 'D6']);
const SEC = [['A', 0, 'wonder'], ['B', 7, 'warmth and comedy'], ['C', 16, 'fear'], ['D', 22, 'clash'], ['E', 32, 'sadness'], ['F', 39, 'wrath'], ['G', 45, 'triumph'], ['H', 61, 'dawn']];
const chords = H.map(s => s.split(' ')[0].split(':')[0]);
fs.writeFileSync(url, JSON.stringify({ title: 'GO (吾 / 悟)', bpm: 144, eighth_s: 60 / 144 / 2, bar_s: 60 / 144 * 4, n_bars: 70, seconds: 70 * 60 / 144 * 4, frames: 2800, fps: 24, frames_per_eighth: 5, swing: 'b1 (bars 7-8) is written out swung, 1.333 + .667', motif: { question: ['F#5', 'E5', 'B5', 'A5'], answer: ['B5', 'A5', 'E6', 'D6'], chorus: ['C#5', 'B4', 'F#5', 'E5'] }, bells, sections: SEC, chords, harm, lines: out }, null, 1));
// ---------------------------------------------------------------- checks
const CH = { '': [0, 4, 7], m: [0, 3, 7], 5: [0, 7], 6: [0, 4, 7, 9], m7: [0, 3, 7, 10], maj7: [0, 4, 7, 11], 7: [0, 4, 7, 10], '7sus4': [0, 5, 7, 10], sus4: [0, 5, 7], m9: [0, 3, 7, 10, 2], maj9: [0, 4, 7, 11, 2], m11: [0, 3, 7, 10, 2, 5], 'maj7#11': [0, 4, 7, 11, 6, 2], pent: [0, 2, 4, 7, 9], phr: [0, 1, 3, 7, 8, 10] };
const pcs = sym => { if (sym === 'N') return null; const [h, b] = sym.split('/'), m = /^([A-G][#b]?)(.*)$/.exec(h), r = PC[m[1][0]] + (m[1][1] === '#' ? 1 : m[1][1] === 'b' ? -1 : 0); if (!CH[m[2]]) throw Error('chord ' + sym); const s = new Set(CH[m[2]].map(x => (x + r + 12) % 12)); if (b) s.add((PC[b[0]] + (b[1] === '#' ? 1 : b[1] === 'b' ? -1 : 0) + 12) % 12); return s; };
const chordAt = t => { let c = harm[0]; for (const h of harm) if (h.t <= t + 1e-6) c = h; return c; };
let bad = 0; const say = s => { console.log('  ' + s); };
console.log('lines', out.length, 'notes', out.reduce((s, l) => s + l.notes.length, 0));
for (const l of out) {
  const ms = l.notes.flatMap(n => n.pitch.split('>').map(midi)), lo = Math.min(...ms), hi = Math.max(...ms), R = { lead: [57, 84], kids: [60, 84], choir: [40, 84], whisper: [40, 90], scream: [57, 90] }[l.voice];
  if (lo < R[0] || hi > R[1]) { bad++; say(`${l.id} range ${lo}..${hi}`); }
  l.notes.forEach((n, i) => {
    const nx = l.notes[i + 1], pv = l.notes[i - 1], m = midi(n.pitch.split('>').pop());
    if (n.d < .5 && n.mora !== 'Q') { bad++; say(`${l.id} "${n.mora}" very short ${n.d}`); }
    if (pv) { const j = Math.abs(midi(n.pitch.split('>')[0]) - midi(pv.pitch.split('>').pop())); if (j > 9) { bad++; say(`${l.id} "${pv.mora}-${n.mora}" leap of ${j}`); } }
    // any overlap with harmony: walk the chords under the note
    const z = n.t + n.d; let t = n.t; while (t < z - .05) { const c = chordAt(t), e = Math.min(z, c.t + c.d), s = pcs(c.sym), len = e - t; if (s && len >= .9 && l.voice !== 'whisper' && l.voice !== 'scream') for (const q of [m, ...(n.div || []).map(midi)]) { const p = ((q % 12) + 12) % 12; if (!s.has(p)) { const clash = s.has((p + 11) % 12) ? 'A SEMITONE ABOVE A CHORD TONE' : s.has((p + 1) % 12) ? 'a semitone below a chord tone' : ''; if (clash || len >= 2) { const hard = clash.startsWith('A SEM') && len >= 1.8; if (hard) bad++; say(`${hard ? '!!' : '  '} ${l.id} "${n.mora}" ${q === m ? n.pitch : q} over ${c.sym} for ${len.toFixed(1)} ${clash}`); } } } t = c.t + c.d; if (c === harm[harm.length - 1]) break; }
  });
  const last = l.notes[l.notes.length - 1]; if (last.d < 1.2 && l.voice === 'lead' && !/cut|comic/.test(l.style)) say(`${l.id} ends short (${last.d})`);
}
const lead = out.filter(l => l.voice === 'lead' || l.voice === 'whisper' || l.voice === 'scream').sort((a, b) => a.notes[0].t - b.notes[0].t);
for (let i = 1; i < lead.length; i++) { const a = lead[i - 1].notes.at(-1), b = lead[i].notes[0], g = b.t - a.t - a.d; if (g < .28) { if (g < -.01) bad++; say(`${g < -.01 ? '!!' : ''} breath before ${lead[i].id}: ${g.toFixed(2)}`); } }
console.log('problems', bad);
