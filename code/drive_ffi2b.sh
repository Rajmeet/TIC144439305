#!/bin/sh
# Works through all FFI-only red dwarfs at 40-70 pc in chunks of 500:
# light-curve search -> automatic pixel verdict -> known-candidate check -> out/FINDINGS.md
cd "$(dirname "$0")"
N=$(($(wc -l < out/targets_ffi2.csv) - 1))
i=${START:-1000}
while [ $i -lt $N ]; do
  name="ffi2_$i"
  .venv/bin/python -c "import pandas as pd; t=pd.read_csv('out/targets_ffi2.csv'); t.iloc[$i:$i+500].to_csv('out/targets_$name.csv', index=False)"
  FFI=1 .venv/bin/python search_v2.py $name targets_$name.csv 3 >> out/$name.log 2>&1
  echo "EXIT $? $(date)" >> out/$name.log
  .venv/bin/python auto_vet.py $name FFI >> out/drive.log 2>&1
  .venv/bin/python known_check.py $name targets_$name.csv >> out/drive.log 2>&1
  i=$((i + 500))
done
echo "ALL CHUNKS DONE $(date)" >> out/drive.log
