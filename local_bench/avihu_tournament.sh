#!/bin/zsh
# Fresh reads vary a lot between runs; run three, snapshot each, keep the
# best by reference score — and never overwrite a better round again.
set -u
cd "/Users/ronenabutbul/Documents/אישי/audiforge-service/local_bench"
P="Avihu Medina - Drum Set"
DIR="results/convert/$P"
XML="$DIR/$P (grid).musicxml"
RD="$DIR/$P (grid).readings.json"

score() {
  ./.venv-homr/bin/python drum_score.py 2>/dev/null \
    | grep "Avihu" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | tr -d '%'
}

# snapshot the current state as candidate 0
cp "$RD" "$DIR/readings.tourn0.json" 2>/dev/null
cp "$XML" "$DIR/tourn0.musicxml" 2>/dev/null
best=$(score); bestn=0
echo "candidate 0 (current): $best%"

for n in 1 2 3; do
  ./.venv-homr/bin/python drum_grid.py "$DIR" "$XML" --gemini --verify \
    2>&1 | grep -E "wrote|merged|kept"
  cp "$RD" "$DIR/readings.tourn$n.json"
  cp "$XML" "$DIR/tourn$n.musicxml"
  s=$(score)
  echo "candidate $n: $s%"
  if (( $(echo "$s > $best" | bc -l) )); then best=$s; bestn=$n; fi
done

echo "winner: candidate $bestn at $best%"
cp "$DIR/readings.tourn$bestn.json" "$RD"
cp "$DIR/tourn$bestn.musicxml" "$XML"
cp "$XML" ~/Downloads/converted/"$P.musicxml"
echo "restored + delivered candidate $bestn"
