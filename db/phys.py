"""PHYS: matter that answers force, drawn as anime (cel tones, ink, hard shadow), moving by real physics.   from db.phys import *
World: +X right, +Y into the picture, +Z up, metres; t in seconds (f / 24).  Most things draw into an ink.Cel you own:  cel = Cel(); ...; img = over(img, cel.layer())
CAMERA   cam = stage(hy=430, gx=960, gy=800, ppm=150)   level camera: horizon at y 430, world (0,0,0) lands on (960,800), 150 px per metre there.  cam = flat()  world X, -Z = screen px (for effects over a flat drawing)
         any cam.Camera3D works everywhere except Sea (level camera only; look from above with hy far off the top)
PALETTES ROCK / DUST / WATER keys: night fire grey gold dawn day mono (DUST also violet storm steam, WATER also lake blue).  Pass a key or your own dict.

ROCK (phys_rock)            light = direction TO the key light, rim_dir = direction to a second hard light (orange fires, aura), shadow = '#hex' ground shadow
  rb = Rubble(1); rb.burst((0,0,.1), n=80, speed=(4,14), up=.9, size=(.1,1.1), seed=3); rb.bake(96)          # in setup: rigid bodies, real contact, friction, rest.  g=number or g(t) (negative = debris that stops obeying gravity)
  rb.draw(cel, cam, f, 'night', step=2, light=(.5,.2,.8), rim_dir=(-.8,-.2,.1), shadow='#151230', trails=.8); grit(cel, cam, t, (0,0,0), 150)
  fr = Fracture((0,0), rad=4, seed=3, thick=.4, tilt=.95, speed=36); fr.draw(cel, cam, t, 'night', glow=None, lift=lambda u,t: 0)   # plates with thickness + radial cracks.  hold=False: they fall back; lift: they float (f3)
  p = Pillar((4.6,8.5), h=13, r=1.8, seed=4).hit(z=6, dirn=(1,0,.1), t0=.4, speed=24).bake(80); p.draw(cel, cam, f, 'night')        # a body goes through a pillar: Voronoi shards, the top loses its support
  tr = Trench(path_xy, width=2.6, depth=1.1, bank=.8, seed=2); front = lambda t: 95*(1-math.exp(-1.9*t)); tr.draw(cel, cam, front(t), 'night'); rb = tr.rubble(front, 96, 150)
  draw_bodies(cel, cam, [(chunk(seed,'block'|'shard'|'slab').scaled(m), xyz, axis_m(axis, angle), seed)], 'grey')                   # place any rock by hand (a pebble that lifts: f1)
DUST (phys_dust)            returns lobe rows; add lists together, split them by row[1] (depth) to sandwich a figure
  L = dust_ring(t, (0,0), r0=1, v0=10, minor=.7, n=40) + plume(t, (0,0,0), (0,0,1), v0=12, n=40, r=(.3,1.2), spread=.4) + dust_trail(t, tr.at, front, tr.L)
  img = over(img, lobes_layer(cam, L, 'night', light=(.5,-.75,.45), rim=(.7,-.6)))      # light = SCREEN vector (x right, y down, z to us); ss=1.25 fast, 2 close-up; wisps(cel, cam, t, (0,0,0)) = last curls
WATER (phys_water)
  sea = Sea(waves(amp=.3, lam=9, seed=2)); rgb, mask, buf = sea.draw(cam, t, 'night', light=(0,1,.2), road=1, road_w=.4, ring=None, dish=None)   # light = direction to what glitters; dish=(x,y,r,depth,rim) (f3, d3 dents); ring = pressure front
  splash(cel, cam, t, (x,y), v0=4.6, r0=.4, n=17, rad=.08, pal='day', dirn=None); ripple(cel, cam, (x,y), t, pal='grey'); column(cel, cam, (x,y), z0, z1, r, t, 'lake'); column_fall(cel, cam, t, (x,y), z_top, z_head, r, length, 'lake')
  wake(cel, cam, t, lambda e: (x(e), y(e)), 'day', side=1.1, up=2.4)                     # a3 hand cuts the sea: spray walls in true arcs, rooster tail, V wake (path in world metres, e seconds)
  squeeze(cel, t, [(x, y, angle), ...], 'grey')                                          # e4 fist, px;  pull(cel, cam, t, (x,y,z), [(x,y),...], 'blue')  water climbs to an orb (g7);  drop(cel, cam, xyz, r, WATER['day'], v)
RAIN (phys_rain)
  rn = Rain(560, box=(15,22,11), seed=2, wind=(1,0)); img = over(img, rn.draw(cam, f/24, 1., pal='grey'))        # world rain, right for any camera (e2: Camera3D looking straight down)
  tau, v = stop_clock(f, f_stop, f_rise, spread); rn.draw(cam, tau, v, bg=img, bead_r=.016, hero=15)               # f1 stops, hangs as lens beads, rises one by one.  lens_drop(layer, img, x, y, r) = one hero drop
  warp = cone_warp(apex, dirn, half, t); rn.draw(cam, tau, 1., warp=warp); cone_blast(cel, cam, apex, dirn, half, t)   # g1
  hits(cel, f, lambda u, w: (x, y, scale), n=60); runoff(cel, f, [polyline,...]); drip(cel, f, x, y, period=26)
WAVE (phys_wave)            c4 and any big blast: Ridge(...).still / .breakup(cel, t), dome(cel, cx, cy, r, t, 'violet'|'gold'|'blue'), condensation(cel, cx, cy, rx, ry, t), wall(cel, cam, centre, R, h, t), wall_lobes(cam, centre, R, h, t), fireball(cam, t, (x,y,z), r)
SOFT (phys_soft)
  ab = ArmourBreak([(poly_px, '#e9e6f2', 46), ...], hit=(x,y), speed=2600); ab.draw(cel, t)                        # d5: cracks race, white-hot front, plates with thickness leave one after another
  rib = Ribbons(roots_px, dirs, 380, 54, 2).bake(50, wind=lambda f: (-2600,-240), release=lambda i: 14+i*2); rib.draw(cel, f, dict(lit,mid,dark,ink))
  layer, light = flakes(f, 120, morph=0..1 or fn(u)); bend = grass_ring(t, (0,.6)); grass_field(cel, cam, bend)
DRAWING  claw_edge / claw_band = rows of curved claws on any edge (waves, flames); thin, wedge = fast tapered strokes; zigzag = crack path; noise, h1(seed, i...) = stable randoms
Self-tests and worked scenes: tools/phys_qa.py break trench wave rain rainstrip splash extras"""
from .phys_core import *
from .phys_rock import *
from .phys_dust import *
from .phys_water import *
from .phys_rain import *
from .phys_soft import *
from .phys_wave import *
