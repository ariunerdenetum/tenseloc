#!/usr/bin/env python3
"""
Script to extract Llama-3.1 8B embeddings for sentences and save them as CSV.
"""
import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract embeddings using Llama-3.1 8B"
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="Path to the training CSV file"
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        required=True,
        help="Path to the test CSV file"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        required=True,
        help="Hugging Face authentication token"
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float16",
        help="Torch dtype (e.g., float16, float32)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8b",
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./results",
        help="Directory to save output CSVs"
    )
    return parser.parse_args()

def extract_embeddings(df, model, tokenizer, device):
    records = []
    for idx, row in df.iterrows():
        words = row['sentence'].split()
        inputs = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1].squeeze(0)

        word_ids = inputs.word_ids(batch_index=0)
        verb_idx = int(row['verb_index'])
        token_idxs = [i for i, w in enumerate(word_ids) if w == verb_idx]
        if not token_idxs:
            raise RuntimeError(f"No tokens for verb index={verb_idx} at row {idx}")

        verb_emb = last_hidden[token_idxs].mean(dim=0).cpu().numpy()
        rec = {f'hidden_{i}': float(val) for i, val in enumerate(verb_emb)}
        rec.update({
            'language': row['language'],
            'sentence': row['sentence'],
            'main_verb': row['main_verb'],
            'verb_index': verb_idx,
            'label': row['tense']
        })
        records.append(rec)

    return pd.DataFrame.from_records(records)


def main():
    args = parse_args()
    os.environ['HF_TOKEN'] = args.hf_token
    os.environ['TORCH_DTYPE'] = args.torch_dtype

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = getattr(torch, args.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_auth_token=args.hf_token,
        padding_side='left'
    )
    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map='auto',
        use_auth_token=args.hf_token
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    os.makedirs(args.out_dir, exist_ok=True)

    df_train = pd.read_csv(args.train_csv, encoding='utf-8-sig')
    df_test = pd.read_csv(args.test_csv, encoding='utf-8-sig')

    features_train = extract_embeddings(df_train, model, tokenizer, device)
    features_test = extract_embeddings(df_test, model, tokenizer, device)

    train_out = os.path.join(args.out_dir, 'llama_train_features.csv')
    test_out = os.path.join(args.out_dir, 'llama_test_features.csv')
    features_train.to_csv(train_out, index=False, encoding='utf-8-sig')
    features_test.to_csv(test_out, index=False, encoding='utf-8-sig')

    print("Saved train features to", train_out)
    print("Saved test features to", test_out)

if __name__ == '__main__':
    main()
    