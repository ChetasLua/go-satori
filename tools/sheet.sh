#!/bin/zsh
# tools/sheet.sh out.jpg shot [shot ...]  -> 5 frames of every complete shot, one row each (ffmpeg only, no render slot)
cd "$(dirname "$0")/.."; O=$1; shift; FF=.mm/env/bin/ffmpeg; FP=.mm/env/bin/ffprobe; T=$(mktemp -d); rows=()
for s in $@; do
  n=$(python3 -c "import sys; sys.path.insert(0,'.'); import film; print(film.LEN['$s'])"); k=$($FP -v error -count_packets -select_streams v -show_entries stream=nb_read_packets -of csv=p=0 out/$s.mkv 2>/dev/null | tr -d ,)
  [ "$k" = "$n" ] || { echo "skip $s ($k/$n)"; continue; }
  e=""; for q in 8 28 50 72 94; do e="$e+eq(n\,$((n*q/100)))"; done
  $FF -nostdin -y -v error -i out/$s.mkv -vf "select='${e#+}',scale=400:225:out_range=pc,tile=5x1" -frames:v 1 -fps_mode vfr $T/$s.png && rows+=(-i $T/$s.png)
done
c=$((${#rows}/2)); [ $c -gt 1 ] && $FF -nostdin -y -v error $rows -filter_complex "vstack=inputs=$c" -q:v 3 $O || cp $T/*.png $O; echo "$c rows -> $O"; rm -r $T
