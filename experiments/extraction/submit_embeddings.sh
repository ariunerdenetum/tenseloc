#!/bin/bash
#PBS -N llama_embeddings
#PBS -l select=1:ncpus=4:mem=32gb:scratch_local=10gb:ngpus=1
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o llama_embeddings.out
#PBS -e llama_embeddings.err

cd $PBS_O_WORKDIR

# Load modules (adjust names/versions as needed)
module load cuda python/3.10

# Activate your virtual environment
source ~/venv/bin/activate

# Export your HF token so transformers can authenticate
export HF_TOKEN="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"

# Run the embedding script
python run_embeddings.py \
  --train-csv all_sentences_train.csv \
  --test-csv  all_sentences_test.csv \
  --hf-token  $HF_TOKEN \
  --torch-dtype float16 \
  --model-name meta-llama/Llama-3.1-8b \
  --out-dir    ./results

echo "Embedding job completed."

