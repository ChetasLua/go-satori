#!/bin/zsh
# tools/rough.sh [out.mp4]  -> a playable cut from whatever is finished: finished shots from out/<id>.mkv, others a grey slate; sound = out/soundtrack.wav, else out/band_mix.wav, else silence.  ffmpeg only.
cd "$(dirname "$0")/.."; O=${1:-out/rough.mp4}; FF=.mm/env/bin/ffmpeg; FP=.mm/env/bin/ffprobe; mkdir -p out/rough; : > out/rough/list.txt; miss=()
python3 -c "
import sys; sys.path.insert(0,'.')
import film
for s,a,n in film.SHOTS: print(s,n)" | while read s n; do
  p=out/rough/$s.mp4; src=out/$s.mkv; k=0
  [ -f $src ] && k=$($FP -v error -count_packets -select_streams v -show_entries stream=nb_read_packets -of csv=p=0 $src | tr -d ,)
  if [ "$k" = "$n" ]; then
    if [ ! -f $p ] || [ $src -nt $p ]; then nice -n 10 $FF -nostdin -y -v error -threads 3 -i $src -vf "scale=1280:720:flags=lanczos:in_range=tv:out_range=tv,format=yuv420p" -c:v libx264 -preset veryfast -crf 20 -r 24 $p; fi
  else miss+=($s); nice -n 10 $FF -nostdin -y -v error -f lavfi -i "color=c=0x2a2c33:s=1280x720:r=24" -frames:v $n -c:v libx264 -preset ultrafast -crf 30 -pix_fmt yuv420p $p; touch -t 202001010000 $p; fi
  echo "file '$PWD/$p'" >> out/rough/list.txt
done
A=out/soundtrack.wav; [ -f $A ] || A=out/band_mix.wav
if [ -f $A ]; then nice -n 10 $FF -nostdin -y -v error -f concat -safe 0 -i out/rough/list.txt -i $A -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart $O
else nice -n 10 $FF -nostdin -y -v error -f concat -safe 0 -i out/rough/list.txt -c copy -movflags +faststart $O; fi
ls -la $O; echo "missing: $miss"
