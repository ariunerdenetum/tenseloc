#!/usr/bin/env python3
"""
Script to extract Llama-3.1 8B embeddings for sentences at a specified hidden layer and save them as CSV, using batching for efficiency.
"""
import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from a specified hidden layer using Llama-3.1 8B with batching"
    )
    parser.add_argument(
        "--train-csv", type=str, required=True, help="Path to the training CSV file"
    )
    parser.add_argument(
        "--test-csv", type=str, required=True, help="Path to the test CSV file"
    )
    parser.add_argument(
        "--hf-token", type=str, required=True, help="Hugging Face authentication token"
    )
    parser.add_argument(
        "--torch-dtype", type=str, default="float16", help="Torch dtype (e.g., float16, float32)"
    )
    parser.add_argument(
        "--model-name", type=str, default="meta-llama/Llama-3.1-8b", help="Pretrained model name or path"
    )
    parser.add_argument(
        "--layer-idx", type=int, required=True, help="Hidden layer index to extract embeddings from (0 = embeddings, 1 = first layer, etc.)"
    )
    parser.add_argument(
        "--out-dir", type=str, default="./results", help="Directory to save output CSVs"
    )
    return parser.parse_args()

def extract_embeddings(df, model, tokenizer, device, layer_idx, batch_size):
    records = []
    sentences = df['sentence'].tolist()
    languages = df['language'].tolist()
    verbs = df['main_verb'].tolist()
    verb_indices = df['verb_index'].tolist()
    labels = df['tense'].tolist()

    for start in range(0, len(sentences), batch_size):
        end = start + batch_size
        batch_sent = sentences[start:end]
        batch_indices = verb_indices[start:end]
        tokenized = tokenizer(
            [sent.split() for sent in batch_sent],
            is_split_into_words=True,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokenized, output_hidden_states=True)

        hidden = outputs.hidden_states[layer_idx]
        for i in range(hidden.size(0)):
            word_ids = tokenized.word_ids(batch_index=i)
            verb_idx = int(batch_indices[i])
            token_idxs = [j for j, w in enumerate(word_ids) if w == verb_idx]
            if not token_idxs:
                raise RuntimeError(f"No tokens for verb index={verb_idx} in batch item {i}")
            emb = hidden[i, token_idxs, :].mean(dim=0).cpu().numpy()
            rec = {f'hidden_{k}': float(val) for k, val in enumerate(emb)}
            rec.update({
                'language': languages[start + i],
                'sentence': batch_sent[i],
                'main_verb': verbs[start + i],
                'verb_index': verb_idx,
                'label': labels[start + i]
            })
            records.append(rec)
    return pd.DataFrame.from_records(records)

def main():
    args = parse_args()
    os.environ['HF_TOKEN'] = args.hf_token
    os.environ['TORCH_DTYPE'] = args.torch_dtype

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = getattr(torch, args.torch_dtype)
    batch_size = 16

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=args.hf_token,
        padding_side='left'
    )
    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map='auto',
        token=args.hf_token
    )
    model.eval()  # disable dropout and set evaluation mode
    # freeze model parameters to save memory and prevent gradient computation
    for param in model.parameters():
        param.requires_grad = False

    try:
        total_layers = model.config.num_hidden_layers
        print(f"Model '{args.model_name}' has {total_layers} hidden layers (excluding embeddings layer).")
    except AttributeError:
        print("Warning: Could not retrieve number of hidden layers from model config.")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    os.makedirs(args.out_dir, exist_ok=True)

    df_train = pd.read_csv(args.train_csv, encoding='utf-8-sig')
    df_test = pd.read_csv(args.test_csv, encoding='utf-8-sig')

    features_train = extract_embeddings(df_train, model, tokenizer, device, args.layer_idx, batch_size)
    features_test = extract_embeddings(df_test, model, tokenizer, device, args.layer_idx, batch_size)

    train_out = os.path.join(args.out_dir, f'llama_train_layer{args.layer_idx}_features.csv')
    test_out = os.path.join(args.out_dir, f'llama_test_layer{args.layer_idx}_features.csv')
    features_train.to_csv(train_out, index=False, encoding='utf-8-sig')
    features_test.to_csv(test_out, index=False, encoding='utf-8-sig')

    print("Saved train features to", train_out)
    print("Saved test features to", test_out)

if __name__ == '__main__':
    main()
