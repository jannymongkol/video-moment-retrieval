for f in data/Charades_v1_480/*.mp4; do
  ffmpeg -i "$f" -r 16 "data/Charades_v1_480_16/$(basename "$f")";
done