"""h5 satori.  The sun of h4 is behind the paper.  A brush writes the I (seven strokes, done at 24), one held beat, then the heart (34-46): to awaken.  Seal of the four-star ball on 50, a small line, 60 f."""
from db.letters_brush import Satori

NAMES = ['top line', 'leaning stem', 'across and down', 'the long arm', 'mouth, left', 'mouth, over and down', 'mouth, floor', 'heart: left drop', 'heart: right drop', 'heart: the spine, head to needle']


class Shot:
    id = 'h5'; n = 60
    finish = dict(bloom=.08, vig=.10, soft=.04, grain=.008, halation='#fff0dc', lift=.0)
    EVENTS = ([(0, 'paper_room', dict(note='match cut: the sun of h4 glows through the sheet at (960, 486) r 208'))]
              + [(int(round(a)), 'brush_write', dict(stroke=k + 1, dur=max(1, int(round(b - a))), size=(.5, .4, .6, .9, .35, .6, .4, .5, .5, 1.)[k], pan=(0., .0, .1, .1, 0., .15, .05, -.35, -.2, -.3)[k], note=NAMES[k])) for k, (a, b) in enumerate(Satori.TIMES)]
              + [(24, 'brush_lift', dict(size=.3, note='the brush leaves; one held beat, 24-34, nothing but paper')), (31, 'brush_return', dict(size=.2)), (46, 'go_breath', dict(note='the needle leaves the sheet: the breathed "go"; the paper warms')), (Satori.STAMP, 'seal_stamp', dict(size=.9, pan=.4, bell=4, note='the fourth bell')), (52, 'pen_letters', dict(dur=4, size=.2, note='two small lines write themselves, done at 56; the picture is then still and can be held'))])

    def setup(self): self.s = Satori(); self.s.setup()
    def frame(self, f): return self.s.frame(f)
