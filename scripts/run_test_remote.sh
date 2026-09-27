#!/usr/bin/env bash
# Finish the test-set run on a Linux machine (e.g. AWS EC2, 16+ vCPU, 64+ GB RAM).
#
# Expected layout in the directory you run this from (~/work):
#   colab_code.zip                 (project code + model)
#   dataset/test/test_source{1,2,3}.tsv     (competition dataset, at least the test/ folder)
#   test_cache/chunk_*.parquet     (finished chunks copied from Google Drive; may be empty)
#
#   bash run_test_remote.sh            # or: WORKERS=12 bash run_test_remote.sh
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
NCPU=$(nproc)
WORKERS=${WORKERS:-$(( NCPU > 14 ? 12 : (NCPU > 4 ? NCPU - 2 : 2) ))}

echo "== setup =="
rm -rf run && mkdir -p run/student_resource && cd run
unzip -q ../colab_code.zip
ln -s "$HERE/dataset" student_resource/dataset
mkdir -p "$HERE/test_cache"
python3 -m pip -q install pandas numpy scipy scikit-learn rapidfuzz xgboost pyyaml psutil pyarrow

# cache on the persistent test_cache dir, smaller matmul batches (lower peak memory; same results)
sed -i "s#cache_dir: *[^ ]*#cache_dir: $HERE/test_cache#" configs/baseline.yaml
sed -i "0,/batch_size: *[0-9]*/s//batch_size: 100/" configs/baseline.yaml
grep -n "cache_dir\|batch_size" configs/baseline.yaml
python3 -c "import sklearn; print('sklearn', sklearn.__version__, '(model trained with 1.6.1)')"
echo "cores=$NCPU  workers=$WORKERS  chunks already cached: $(ls "$HERE/test_cache" | grep -c _matches.parquet || true)"
free -g | head -2

echo "== run (retries with fewer workers; finished chunks are reused) =="
ok=0
for w in $WORKERS $(( WORKERS / 2 )) 3; do
  echo ">>> attempt with $w workers"
  if python3 -u -m src.pipeline --config configs/baseline.yaml --mode test --allow-full-test --resume \
        --threshold 0.65 --workers "$w" --chunk-size 10000 --no-log 2>&1 | tee -a "$HERE/test_run.log"; then
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then ok=1; break; fi
  fi
  sleep 5
done
[ "$ok" -eq 1 ] || { echo "RUN INCOMPLETE - do not use the outputs. Re-run this script (finished chunks are reused)."; exit 2; }

echo "== validate =="
python3 utils/validate_submission.py --matching outputs/test/matching_results.tsv \
  --candidate outputs/test/candidate_pairs.tsv --test-dir student_resource/dataset/test
mkdir -p "$HERE/final" && cp outputs/test/matching_results.tsv outputs/test/candidate_pairs.tsv "$HERE/final/"
echo "DONE. Copy $HERE/final/matching_results.tsv (and candidate_pairs.tsv) back to your machine."
