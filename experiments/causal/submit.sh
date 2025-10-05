#!/bin/bash
#PBS -N causal_experiment
#PBS -l select=1:ncpus=8:mem=64gb:scratch_local=64gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o causal_fr.out
#PBS -e causal_fr.err

cd $PBS_O_WORKDIR

# Load modules (adjust names/versions as needed)
module load cuda python/3.10

# Activate your virtual environment
source ../venv/bin/activate

# Export your HF token so transformers can authenticate
export HF_TOKEN="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"

echo "Working directory is: $(pwd)"
echo "Listing contents:" ls -R .
echo "=== Causal tracing ==="
python3 run_causal.py \
  --hf_token $HF_TOKEN \
  --json_file "./causal_prompts/translated_prompts_fr.json" \
  --m_noise 5 \
  --window 3

echo "ROME job completed."
