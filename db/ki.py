"""KI: drawn energy for the whole film.  `from db import ki`.  Everything is a pure function of the frame f (int), seeded, premultiplied float32 RGBA layers (H,W,4)
and additive RGB lights (H,W,3).  Compose:  img = ki.put(img, layer);  img = img + light   (hold light off faces: ki.add(img, light, protect=mask)).
kinds: 'gold' SSJ furnace | 'blue' | 'base' blue-white | 'violet' Frieza | 'silver' UI | 'red' | 'white' gold-white | aura also 'goldblue'

AURA        img = ki.with_aura(img, fig_layer, f, 'gold', power=ki.envelope(f, on=10, off=70, flares=[(30, .6)]), face=(x, y, r))          # back flame + lit figure + front wisps + light
  by hand   A = ki.aura(fig_layer, f, 'gold', 1., seed=1, height=.46, up=(0,-1), face=(x,y,r));  if hasattr(A,'haze'): img = A.haze(img)     (silver bends the background)
            img = A.under(img); img = ki.put(img, ki.lit(fig_layer, 'gold', 1., f, face=face)); img = A.over(img)
            flying: up = -direction of travel, ground=False.  height .3 calm .. .9 furnace; width 1.3 wide; face is kept clear; keep=[(x,y,r),..] = circles (fists, face) round which the flame thins; fringe, sharp = small tips, tip sharpness.  lit(..., dirs=(dx,dy)) = one-sided light
  pillar    back, front, light = ki.pillar3(cam, f, P0=(0,0,0), R0=1.35, height=10, hole=(x,y,rx,ry))                                        # 3-D, orbit it
LIGHTNING   layer, light = ki.storm([(start, p0, p1, width, seed, dict(forks=4))], f, 'gold')      back, front, light = ki.arcs(fig_layer, f, 'blue', n=7, face=face)
SPHERE      layer, light = ki.orb(x, y, r, f, 'blue');  under = ki.charge(x, y, r, f, 'blue', reach=8, water=True);  hands = ki.light_on(hand_layer, x, y, r, 'blue')     r<12 = pearl
BEAM        ax = ki.Axis2([(x0,y0),...,(x1,y1)], r0, r1)  or Axis2(P, lambda v: radius)  or ki.Axis3(cam, A, B, R0)     (m = distance along it in beam radii, 0..ax.M)
            layer, light = ki.beam(ax, f, 'gold', collars=4, dens=1.5)        Final Flash / Kamehameha;  head= tail= metric ends
            layer, light = ki.helix(ax, f)                                    gold-blue double helix
            layer, light = ki.ignite(ax, f, t0, 'gold', frames=7)             gather, flash, head runs, muzzle (style='helix');   ki.die(ax, f, t0, 'gold')
            ki.muzzle(x,y,r,f,'gold',kind2='blue',back=math.pi)   ki.impact_front(x,y,r,f,('gold','blue'),asp=.5,rot=0)   ki.clash(x,y,r,f,('gold','violet'))
  parted    ax = ki.Axis2(P, r0, r1, warp=ki.parting(m0, .4));  beam as usual;  ki.wake(plain_axis, m0, .25, f, 'gold')      ki.inside(f, cx, cy, 'gold', hole=(x,y,rx,ry)) -> RGB, we look along it
  thin      ki.line(p0, p1, f, t0, 'violet', w=6)    ki.volley([(t0,p0,p1,width,frames)], f, 'violet', bend=ki.deflect(cx,cy,rx,ry))    ki.splashes([(frame,x,y,r,dir)], f)
TRAILS      F = ki.trail(px_polyline, 'gold', w);  ki.trail_cel(F,'gold');  ki.head(x,y,r,'gold',f,ang=flight)  ki.clash_star(x,y,r,age)  ki.sea_dent(cam,X,age,size)  ki.punch(cloud_layer,cx,cy,r)
HITS        ki.impact(img, n 0..7, cx, cy, subject=layer, dirn=a) -> RGB impact frame (all different)    ki.ring(x,y,rad,thick,asp=,rot=)  ki.ring3(cam,P,axis,rad,thick)
            ki.refract_ring(img,x,y,rad,width,amp)   ki.cone(tip, dirn, length, half, f)   ki.vanish(fig_layer, k 0..3, ang)   
GHOSTS      ki.inkwash(fig_layer, age, ink='light'|'dark', drift=(dx,dy))     ki.water_ring(img, x, y, f, t0)
SCENE       layer = ki.cumulus(ki.heap(cx,base,w,h,seed), light, (deep,shadow,half,lit,top))   ki.puffs(Field, heap, light)  ki.rock(cel,P,seed,...)  ki.chunk(...)  ki.moon(x,y,r)  ki.sea_waves(U,V,T,t)
Tongues of your own: F = ki.Field(4); ki.lick(F, ki.spine(x,y,L,angle), half_width); layer = F.cel(ki.PAL['gold']['c'], lw=(2,0,1,0), ink=(ki.PAL['gold']['ink'],None,ki.PAL['gold']['c'][0],None))
Run aura on ones or twos (step=2 default), beams and lightning on ones.  Cost (busy machine): aura .45 s, goldblue .9, beam .6, helix 1.3, pillar3 1.5, ignite .7, others < .45.
"""
from .ki_core import *
from .ki_aura import Hug, Aura, aura, aura_ui, envelope, lit, with_aura, pillar3
from .ki_bolt import bolt, bolt_pts, bolt_cel, storm, arcs
from .ki_orb import orb, charge, light_on
from .ki_beam import Axis2, Axis3, beam, body_field, streams, swells, collar, helix, petals, disc, rays, muzzle, impact_front, parting, wake, volley, deflect, splashes, line, inside, clash, ignite, die
from .ki_hit import ring, ring3, ring_strokes, circle3, ellipse_pts, refract_ring, cone, vanish, impact, inkwash, water_ring
from .ki_trail import trail, trail_cel, head, clash_star, sea_dent, punch
from .ki_scene import puffs, cumulus, heap, rock, chunk, moon, sea_uv, sea_waves
