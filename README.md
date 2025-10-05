# TenseLoC: Tense Localization and Control in a Multilingual LLM

## Overview

While multilingual language models excel across diverse tasks, little is known about how they encode fundamental grammatical categories such as tense. We examine how decoder-only transformers encode tense, focusing on the mechanisms underlying representation, transfer, and control across languages. To identify tense-specific subspaces and their transferability, we cover eight typologically diverse languages: English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai. We construct a synthetic tense-annotated dataset and combine probing, causal analysis, feature disentanglement, and model steering to LLama-3.1 8B, a state-of-the-art multilingual model.

Key findings include the clear emergence of tense representation in early layers and significant transfer within the same language family in LLama-3.1 8B. Causal tracing further shows that attention outputs around layer 16 consistently carry cross-lingually transferable tense information. By effectively utilizing sparse autoencoders, we identify and steer English tense-related features, thereby enhancing target language tense prediction accuracy by up to 11\% in a downstream cloze task.

## Instructions for installation

### Virtual environment

1. Create & Activate virtual environment
```shell
python3 -m venv venv
source venv/bin/activate

pip install <package> --no-cache-dir

# 2. Install Dependencies
pip install -r requirements.txt --no-cache-dir
```

2. Running Makefile
```shell
make action argument=something

### Structure

```bash
/ (root)
├── README.md                # Overview, motivation, and instructions for installation and usage
├── requirements.txt         # Python dependencies
├── src/                     % Source code
│   │
    |── download_ud_data.py
    |── global_scramble.py
    |── local_scramble.py
    |── merge_sentences.py 
│   └── ...           
│
├── experiments/             % Notebooks and experiment-specific scripts
│   │
│   └── notebook files
│
└── results/                 % Final aggregated results, figures, and reports
    ├── figures/             % Plots and visualization outputs
    ├── tables/              % Summary tables of results
```

### Model

```bash
---------------------- MODEL ----------------------
LlamaModel(
  (embed_tokens): Embedding(128256, 4096)                                           # <= Embedding layer
  (layers): ModuleList(
    (0-31): 32 x LlamaDecoderLayer(
      (self_attn): LlamaAttention(
        (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
        (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
        (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (mlp): LlamaMLP(
        (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
        (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
        (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
        (act_fn): SiLU()
      )
      (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
    )
  )
  (norm): LlamaRMSNorm((4096,), eps=1e-05)
  (rotary_emb): LlamaRotaryEmbedding()
)
---------------------------------------------------
Model 'meta-llama/Llama-3.1-8b' has 32 hidden layers (excluding embeddings layer).
```
