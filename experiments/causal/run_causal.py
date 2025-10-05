# run_causal.py
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from itertools import chain
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyvene import (
    ConstantSourceIntervention,
    LocalistRepresentationIntervention,
    IntervenableConfig,
    RepresentationConfig,
    IntervenableModel,
    VanillaIntervention,
)

# Global placeholders for device and model type
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = None

def get_gold_joint_original(model, tokenizer, prompt: str, device: torch.device, gold_ids: list[int]) -> float:
    gold_joint = 1.0
    curr_prompt = prompt
    for gold_id in gold_ids:
        inputs = tokenizer(curr_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[0, -1]
        probs = F.softmax(logits, dim=-1)
        p_gold = probs[gold_id].item()
        gold_joint *= p_gold
        subtok = tokenizer.decode([gold_id], clean_up_tokenization_spaces=False)
        curr_prompt += subtok
    return gold_joint

def get_gold_joint_intervened(
    intervenable,
    tokenizer,
    prompt: str,
    device: torch.device,
    sources,
    unit_locations,
    gold_ids: list[int],
) -> float:
    gold_joint = 1.0
    curr_prompt = prompt
    for gold_id in gold_ids:
        inputs = tokenizer(curr_prompt, return_tensors="pt").to(device)
        _, out = intervenable(inputs, sources, unit_locations=unit_locations)
        logits = out.logits[0, -1]
        probs = F.softmax(logits, dim=-1)
        p_gold = probs[gold_id].item()
        gold_joint *= p_gold
        subtok = tokenizer.decode([gold_id], clean_up_tokenization_spaces=False)
        curr_prompt += subtok
    return gold_joint

def get_restore_positions(few_shot: str, target: str, tokenizer) -> list[int]:
    lines = few_shot.split("\n")
    char_starts = []
    cum = 0
    for line in lines:
        char_starts.append(cum)
        cum += len(line) + 1
    first_span = (char_starts[0], char_starts[0] + len(lines[0]))
    last_span = (char_starts[-1], char_starts[-1] + len(lines[-1]))
    enc = tokenizer(few_shot, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    token_ids = enc["input_ids"]
    candidates = []
    for i, (st, en) in enumerate(offsets):
        in_first = first_span[0] <= st and en <= first_span[1]
        in_last = last_span[0] <= st and en <= last_span[1]
        if not (in_first or in_last):
            continue
        candidates.append(i)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    L = len(target_ids)
    pos_restore = []
    for i in candidates:
        window = token_ids[i : i + L]
        if window == target_ids:
            pos_restore.extend(range(i + 1, i + L + 1))
    return sorted(set(pos_restore))

class NoiseIntervention(ConstantSourceIntervention, LocalistRepresentationIntervention):
    def __init__(self, embed_dim=None, seed: int = 1, **kwargs):
        super().__init__()
        embed_dim = embed_dim or kwargs.get("latent_dim")
        if embed_dim is None:
            raise ValueError(f"No latent_dim in kwargs: {list(kwargs)}")
        self.interchange_dim = embed_dim
        rs = np.random.RandomState(seed)
        prng = lambda *shape: rs.randn(*shape)
        self.noise = torch.from_numpy(prng(1, 1, embed_dim)).to(DEVICE)
        self.noise_level = 0.13462981581687927

    def forward(self, base, source=None, subspaces=None):
        base[..., : self.interchange_dim] += self.noise * self.noise_level
        return base

def corrupted_config(model_type, layer, seed: int = 1) -> IntervenableConfig:
    return IntervenableConfig(
        model_type=model_type,
        representations=[RepresentationConfig(layer, "block_input")],
        intervention_types=NoiseIntervention,
        intervention_additional_kwargs=[{"seed": seed}],
    )

def restore_corrupted_with_interval_config(
    intervened_layer: int,
    restore_layer: int,
    stream: str = "mlp",
    window: int = 3,
    num_layers: int = 33,
    seed: int = 1,
) -> IntervenableConfig:
    half = window // 2
    start = max(0, restore_layer - half)
    end = min(num_layers, restore_layer + half + 1)
    reps = []
    types = []
    kwargs_list = [{"seed": seed}]
    reps.append(RepresentationConfig(0, "block_input"))
    types.append(NoiseIntervention)
    for L in range(start, end):
        reps.append(RepresentationConfig(L, stream))
        types.append(VanillaIntervention)
        kwargs_list.append({})
    return IntervenableConfig(
        model_type=MODEL_TYPE,
        representations=reps,
        intervention_types=types,
        intervention_additional_kwargs=kwargs_list,
    )

def main():
    global MODEL_TYPE
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8b")
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--m_noise", type=int, default=10)
    parser.add_argument("--window", type=int, default=1)
    args = parser.parse_args()

    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.float16, token=args.hf_token
    )
    model.eval()
    
    MODEL_TYPE = type(model)

    df = pd.read_json(args.json_file)
    streams = ["block_output", "attention_output", "mlp_activation", "mlp_output"]
    num_layers = model.config.num_hidden_layers

    for entry in df.itertuples():
        rows = []
        pid, prompt, restore_words, gold = entry.prompt_id, entry.prompt, entry.words_restore, entry.gold
        tense, lang = pid.split("_")
        lang = lang[:2]
        gold_ids = tokenizer.encode(f" {gold}", add_special_tokens=False)
        p_clean = get_gold_joint_original(model, tokenizer, prompt, device, gold_ids)
        pos_restore = sorted(
            set(chain.from_iterable(get_restore_positions(prompt, w, tokenizer) for w in restore_words))
        )
        base = tokenizer(prompt, return_tensors="pt").to(device)
        restoration_positions = [0]
        if pos_restore and min(pos_restore) > 0:
            restoration_positions.append(min(pos_restore) - 1)
        restoration_positions.extend(pos_restore)
        final_token_pos = base.input_ids.size(1) - 1
        restoration_positions.append(final_token_pos)
        restoration_positions = sorted(set(restoration_positions))

        for seed in range(args.m_noise):
            config_corrupt = corrupted_config(MODEL_TYPE, layer=0, seed=seed)
            intervenable_corrupt = IntervenableModel(config_corrupt, model)
            p_corrupt = get_gold_joint_intervened(
                intervenable_corrupt,
                tokenizer,
                prompt,
                device,
                sources=None,
                unit_locations={"base": [[pos_restore]]},
                gold_ids=gold_ids,
            )
            for stream in streams:
                for restore_layer in range(num_layers):
                    for pos in restoration_positions:
                        cfg = restore_corrupted_with_interval_config(
                            intervened_layer=0,
                            restore_layer=restore_layer,
                            stream=stream,
                            window=args.window,
                            num_layers=num_layers,
                            seed=seed,
                        )
                        interv = IntervenableModel(cfg, model)
                        n_restores = len(cfg.representations) - 1
                        sources = [None] + [base] * n_restores
                        unit_locations = {
                            "sources->base": (
                                [None] + [[[pos]]] * n_restores,
                                [[pos_restore]] + [[[pos]]] * n_restores,
                            )
                        }
                        p_restored = get_gold_joint_intervened(
                            intervenable=interv,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            device=device,
                            sources=sources,
                            unit_locations=unit_locations,
                            gold_ids=gold_ids,
                        )
                        rows.append(
                            {
                                "language": lang,
                                "tense": tense,
                                "stream": stream,
                                "prompt_id": pid,
                                "pos": pos,
                                "noise_seed": seed,
                                "restore_layer": restore_layer,
                                "gold": gold,
                                "p_clean": p_clean,
                                "p_corrupt": p_corrupt,
                                "p_restored": p_restored,
                                "delta_corrupt": p_clean - p_corrupt,
                                "delta_restored": p_restored - p_corrupt,
                            }
                        )
        pd.DataFrame(rows).to_csv(f"{pid}.csv", index=False)
        print(f"Saved {pid}.csv")

if __name__ == "__main__":
    main()
