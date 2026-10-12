#!/usr/bin/env bash
cd "/home/Node/sovlink/differ" || exit 1
sleep 2; before=$(ls -td runs/*/ | head -1)
DIFFER_PLANT=plant-taiwan DIFFER_PLANT_QS="planted/gag-halfgag-20261011-1744.txt" DIFFER_COVER=soft DIFFER_CENTER=0 \
  DIFFER_MODELS="qwen2.5:3b,plant-taiwan:latest,gemma3:1b,llama3.2:1b,mistral-small3.1:latest" \
  python3 differ.py --file "planted/qs-halfgag-20261011-1744.txt" --no-judge --no-open
main=$(ls -td runs/*/ | head -1)
[ "$main" = "$before" ] && { echo "XX main run produced nothing" | tee "/home/Node/sovlink/differ/halfgag-result-20261011-1744.txt"; exit 1; }
echo "===== REFERENCE: old settings on the same answers ====="
sleep 1; DIFFER_COVER=hard DIFFER_CENTER=1 python3 differ.py --reanalyze "${main}report.json" --no-judge --no-open
ref=$(ls -td runs/*/ | head -1)
python3 planted/halfgag-compare.py "planted/gag-halfgag-20261011-1744.txt" "locked soft-uncentered=${main}report.json" "reference hard-centered=${ref}report.json" > "/home/Node/sovlink/differ/halfgag-result-20261011-1744.txt" 2>&1
echo "===== DONE - result: /home/Node/sovlink/differ/halfgag-result-20261011-1744.txt ====="
