#!/bin/bash
#PBS -N llama_embeddings
#PBS -l select=1:ncpus=8:mem=64gb:scratch_local=64gb:ngpus=1:gpu_mem=30gb
#PBS -l walltime=3:00:00
#PBS -j oe
#PBS -o llama_embeddings16.out
#PBS -e llama_embeddings16.err

DATADIR=/storage/brno2/home/ariuka/tense
LAYERS=16

echo "$PBS_JOBID is running on node $(hostname -f) in scratch directory $SCRATCHDIR" >> $DATADIR/sae/jobs_info.txt

test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

cp -r $DATADIR/all_sentences_train.csv $SCRATCHDIR || { echo >&2 "Error while copying JSON input!"; exit 2; }
cp -r $DATADIR/all_sentences_test.csv $SCRATCHDIR || { echo >&2 "Error while copying JSON input!"; exit 2; }
cp $DATADIR/sae/run_model.py $SCRATCHDIR || { echo >&2 "Error copying run_model.py"; exit 3; }

cd $SCRATCHDIR

# Load modules (adjust names/versions as needed)
module load cuda python/3.10

# use venv Python directly (do not rely on relative paths)
PYTHON_EXE=$DATADIR/Language-Model-SAEs/.venv/bin/python3
test -x "$PYTHON_EXE" || { echo >&2 "Python executable not found at $PYTHON_EXE"; exit 5; }

# Export your HF token so transformers can authenticate
export HF_TOKEN="..."

# Loop over layer indices
for LAYER in $LAYERS; do
  echo "=== Extracting embeddings from layer $LAYER ==="
  python3 run_model.py \
    --train-csv   all_sentences_train.csv \
    --test-csv    all_sentences_test.csv \
    --hf-token    $HF_TOKEN \
    --torch-dtype float16 \
    --batch-size 16 \
    --model-name  meta-llama/Llama-3.1-8B \
    --layer-idx   $LAYER \
    --out-dir     ./model_outputs
done

clean_scratch

echo "Embedding job completed."
