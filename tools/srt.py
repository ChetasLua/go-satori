"""audio/score.json -> dist/go.en.srt: one English line per sung line, sorted, never overlapping"""
import json
L = [l for l in json.load(open('audio/score.json'))['lines'] if (l.get('en') or '').strip() and l.get('notes')]
E = sorted([(l['notes'][0]['t'] * .2083333 - .05, max(n['t'] + n['d'] for n in l['notes']) * .2083333 + .25, l['en'].strip()) for l in L]); out = []
for a, b, t in E:
    a = max(a, 0.)
    if out and (t == out[-1][2] or a - out[-1][0] < .45): out[-1][1] = max(out[-1][1], b); continue
    out.append([a, b, t])
for x, y in zip(out, out[1:]): x[1] = min(x[1], y[0] - .05)
out = [x for x in out if x[1] - x[0] > .25]; out[-1][1] = min(out[-1][1], 118.4)
ts = lambda s: '%02d:%02d:%06.3f' % (s // 3600, s % 3600 // 60, s % 60)
open('dist/go.en.srt', 'w').write('\n'.join('%d\n%s --> %s\n%s\n' % (i + 1, ts(a).replace('.', ','), ts(b).replace('.', ','), t) for i, (a, b, t) in enumerate(out))); print(len(out), 'lines', out[0], out[-1])
