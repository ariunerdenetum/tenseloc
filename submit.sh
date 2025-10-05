#!/bin/bash
#PBS -N causal_experiment
#PBS -l select=1:ncpus=8:mem=64gb:scratch_local=64gb:ngpus=1:gpu_mem=20gb
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -o causal_fr.out
#PBS -e causal_fr.err

DATADIR=/storage/brno2/home/ariuka/tense
LANG_FILE=translated_prompts_fr.json

echo "$PBS_JOBID is running on node $(hostname -f) in scratch directory $SCRATCHDIR" >> $DATADIR/causal/jobs_info.txt

test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

cp -r $DATADIR/causal/causal_prompts $SCRATCHDIR || { echo >&2 "Error while copying JSON input!"; exit 2; }
cp $DATADIR/causal/run_causal.py $SCRATCHDIR || { echo >&2 "Error copying run_causal.py"; exit 3; }

cd $SCRATCHDIR

# Load modules (adjust names/versions as needed)
module load cuda python/3.10

# use venv Python directly (do not rely on relative paths)
PYTHON_EXE=$DATADIR/venv/bin/python3
test -x "$PYTHON_EXE" || { echo >&2 "Python executable not found at $PYTHON_EXE"; exit 5; }

# Export your HF token so transformers can authenticate
export HF_TOKEN="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"

JSON_PATH="$SCRATCHDIR/causal_prompts/$LANG_FILE"

# $PYTHON_EXE run_causal.py \
#   --hf_token $HF_TOKEN \
#   --json_file "$SCRATCHDIR/causal_prompts/$LANG_FILE" \
#   --m_noise 5 \
#   --window 3 || { echo >&2 "Python script failed (exit code $?)"; exit 6; }

# # Copy all CSV results back
# cp *.csv $DATADIR/causal/ || { echo >&2 "Result file(s) copying failed (exit code $?)"; exit 4; }

# Determine number of entries in JSON
NUM_ENTRIES=$($PYTHON_EXE - <<EOF
import pandas as pd
df = pd.read_json("$JSON_PATH")
print(len(df))
EOF
)
if [ -z "$NUM_ENTRIES" ]; then
    echo "Failed to read number of entries" >&2
    exit 6
fi

echo "=== Causal tracing: processing $NUM_ENTRIES entries ==="

for (( idx=0; idx<NUM_ENTRIES; idx++ )); do
    echo "Processing entry $idx out of $NUM_ENTRIES"
    $PYTHON_EXE run_causal.py \
        --hf_token $HF_TOKEN \
        --json_file "$JSON_PATH" \
        --m_noise 5 \
        --window 3 \
        --entry_idx $idx || { echo >&2 "run_causal.py failed on entry $idx"; exit 7; }

    # Copy CSV immediately
    PID=$(ls *.csv | tail -n 1)
    cp "$PID" $DATADIR/causal/ || { echo >&2 "Failed copying $PID"; exit 8; }
    rm "$PID"
done

clean_scratch
