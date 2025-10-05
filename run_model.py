#!/usr/bin/env python3
"""
Script to extract Llama-3.1 8B activations from attention, MLP, and residual streams
at specified hidden layers, using HookedTransformer, and save as Parquet files.
Processes first 240 examples per tense label in train set and 60 per label in test set.
"""
import os
import argparse
import pandas as pd
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


STREAM_HOOKS = {
    'attention': 'blocks.{layer}.hook_attn_out',
    'mlp': 'blocks.{layer}.hook_mlp_out',
    'residual': 'blocks.{layer}.hook_resid_post'
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', type=str, required=True)
    parser.add_argument('--test-csv', type=str, required=True)
    parser.add_argument('--model-name', type=str,
                        default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--layers', type=int, nargs='+', required=True,
                        help='List of layer indices to extract')
    parser.add_argument('--out-dir', type=str, default='./results')
    parser.add_argument('--hf-token', type=str, required=True)
    parser.add_argument('--torch-dtype', type=str, default='bfloat16')
    parser.add_argument('--batch-size', type=int, default=16)
    return parser.parse_args()


def load_data(path, n_per_label):
    df = pd.read_csv(path, encoding='utf-8-sig')
    frames = []
    for label in ['past', 'present', 'future']:
        subset = df[df['tense'] == label].head(n_per_label)
        frames.append(subset)
    return pd.concat(frames, ignore_index=True)


def extract_stream(tokenized, verb_indices, hook_name, model):
    input_ids = tokenized['input_ids'].to(model.cfg.device)
    attention_mask = tokenized['attention_mask'].to(model.cfg.device)
    # Use correct signature: tokens tensor and attention_mask keyword
    _, cache = model.run_with_cache(input_ids, attention_mask=attention_mask)
    acts = cache[hook_name]
    out = []
    for i, vidx in enumerate(verb_indices):
        word_ids = tokenized.word_ids(batch_index=i)
        positions = [pos for pos, w in enumerate(word_ids) if w == vidx]
        if not positions:
            raise RuntimeError(f"No token for verb_index={vidx} in example {i}")
        sub_embs = acts[i, positions, :]
        emb = sub_embs.mean(dim=0).cpu().numpy()
        out.append(emb)
    return torch.tensor(out)


def process_split(df, model, tokenizer, layers, split_name, n_per_label, batch_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Prepare dataframes per batch
    sentences = df['sentence'].tolist()
    verbs = df['verb_index'].tolist()
    metadata = df[['language','sentence','main_verb','verb_index','tense']]

    for layer in layers:
        for stream_name, hook_fmt in STREAM_HOOKS.items():
            hook_name = hook_fmt.format(layer=layer)
            records = []
            # iterate in batches
            for start in range(0, len(sentences), batch_size):
                batch_sent = sentences[start:start+batch_size]
                batch_verbs = verbs[start:start+batch_size]
                # tokenize with word alignment
                tokenized = tokenizer(
                    [s.split() for s in batch_sent],
                    is_split_into_words=True,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                )
                # extract activations
                emb_batch = extract_stream(tokenized, batch_verbs, hook_name, model)
                # collect records
                for i, emb in enumerate(emb_batch.numpy()):
                    rec = {f'{stream_name}_{j}': float(v)
                           for j, v in enumerate(emb)}
                    md = metadata.iloc[start + i].to_dict()
                    rec.update({'layer': layer, 'stream': stream_name, **md})
                    records.append(rec)

            df_out = pd.DataFrame.from_records(records)
            out_file = os.path.join(
                out_dir,
                f'llama_{split_name}_layer{layer}_{stream_name}.parquet'
            )
            df_out.to_parquet(out_file, index=False)
            print(f'Saved {out_file}: {df_out.shape}')


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = getattr(torch, args.torch_dtype)

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
        padding_side='left',
        token=hf_token
    )

    # ensure pad_token exists for batch padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model_hf = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map='auto',
        token=args.hf_token
    )

    # resize embeddings if pad_token added
    model_hf.resize_token_embeddings(len(tokenizer))

    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name,
        hf_model=model_hf,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype
    ).eval()

    df_train = load_data(args.train_csv, n_per_label=240)
    df_test = load_data(args.test_csv, n_per_label=60)

    process_split(df_train, model, tokenizer,
                  args.layers, 'train', 240,
                  args.batch_size, args.out_dir)
    process_split(df_test, model, tokenizer,
                  args.layers, 'test', 60,
                  args.batch_size, args.out_dir)


if __name__ == '__main__':
    main()
