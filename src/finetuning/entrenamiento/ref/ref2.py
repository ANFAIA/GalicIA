# pip install -U "trl>=0.21.0" "transformers>=4.42" peft datasets accelerate bitsandbytes

import re
import math
import random
from typing import List, Dict, Any

import torch
from datasets import load_from_disk, Dataset, DatasetDict

from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# ----------------------------
# Heurística de métrica poética
# ----------------------------
VOWELS = set("aeiouáéíóúàèìòùâêîôûäëïöüAEIOUÁÉÍÓÚÂÊÎÔÛÄËÏÖÜ")

def approx_syllables(word: str) -> int:
    if not word:
        return 0
    prev_vowel = False
    count = 0
    for ch in word:
        is_vowel = ch in VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(count, 1)

def approx_line_syllables(line: str) -> int:
    return sum(approx_syllables(w) for w in line.strip().split())

def rhyme_key(line: str) -> str:
    tokens = [t for t in ''.join(ch if ch.isalnum() else ' ' for ch in line.lower()).split() if t]
    if not tokens:
        return ""
    last = tokens[-1]
    return last[-3:] if len(last) >= 3 else last

def score_poem(text: str, structure) -> float:
    stanzas = [s for s in text.strip().split("\n\n") if s.strip()]
    target_stanzas = len(structure)
    score = -abs(len(stanzas) - target_stanzas) * 1.0

    for i, st in enumerate(stanzas[:target_stanzas]):
        lines = [l for l in st.strip().split("\n") if l.strip()]
        target_syllables = structure[i].get("syllables", [])
        target_rhymes = structure[i].get("rhyme", [])
        score -= abs(len(lines) - len(target_syllables)) * 0.5

        rhyme_map: Dict[str, List[str]] = {}
        for j, line in enumerate(lines[:len(target_syllables)]):
            syl = approx_line_syllables(line)
            score -= abs(syl - target_syllables[j]) * 0.2
            if j < len(target_rhymes):
                lab = target_rhymes[j]
                rhyme_map.setdefault(lab, []).append(rhyme_key(line))

        for _, keys in rhyme_map.items():
            if len(keys) > 1:
                base = keys[0]
                mismatches = sum(1 for k in keys[1:] if k != base)
                score -= mismatches * 0.5

    if any(tok in text.lower() for tok in ["non", "que", "coa", "noite", "lúa", "auga", "terra", "vento"]):
        score += 0.3
    return float(score)

# ----------------------------
# Dataset → formato conversacional + structure
# ----------------------------
def build_conversational_ds(raw: Dataset | DatasetDict) -> DatasetDict:
    if isinstance(raw, Dataset):
        tmp = raw.train_test_split(test_size=0.1, seed=42)
        raw = DatasetDict(train=tmp["train"], eval=tmp["test"])

    def to_messages(ex: Dict[str, Any]):
        # prompt original del user
        user_prompt = ex["conversations"]["content"][0]
        # pista de estrutura (opcional)
        patt = ex.get("structure") or []
        patt_str = "; ".join(
            f"{'/'.join(map(str, p.get('syllables', [])))} con rima {''.join(p.get('rhyme', []))}"
            for p in patt
        )
        hint = ""
        if patt_str:
            hint = (
                "\n\nSegue esta estrutura: " + patt_str +
                ". Escribe en galego. Responde só co poema final, "
                "sen explicacións nin etiquetas nin pensamento."
            )
        messages = [{"role": "user", "content": user_prompt + hint}]
        return {"messages": messages, "structure": patt}

    out = {}
    for split in raw.keys():
        out[split] = raw[split].map(to_messages, remove_columns=raw[split].column_names)
    return DatasetDict(out)

# ----------------------------
# Recompensa (sen razonamento)
# ----------------------------
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

def poem_reward(completions, structure, **kwargs):
    """
    completions: list[list[{"role": "assistant", "content": "..."}]]
    structure:   list[ list[dict] ]  (tal cual ven do dataset)
    """
    rewards = []
    for comp, patt in zip(completions, structure):
        # extrae o texto da resposta do assistant
        text = comp[0]["content"] if isinstance(comp, list) else str(comp)
        # elimina calquera razonamento oculto antes de puntuar
        clean = THINK_BLOCK_RE.sub("", text).strip()
        r = score_poem(clean, patt)

        # penalización forte se aparecen etiquetas <think> (non queremos razonamento)
        if "<think>" in text.lower() or "</think>" in text.lower():
            r -= 1.0

        # capping suave para estabilidade de GRPO
        r = max(min(r, 5.0), -5.0)
        rewards.append(float(r))
    return rewards

# ----------------------------
# Main
# ----------------------------
def main():
    model_id = "galicIA-v1"  # <- tu adaptador LoRA en el Hub o ruta local

    # Carga tokenizer (left padding recomendado en GRPO)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Carga política desde un checkpoint PEFT (LoRA)
    # AutoPeftModel localiza el base_model y aplica el adaptador automáticamente.
    policy = AutoPeftModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    # Dataset
    raw = load_from_disk("poemas_GalicIA_est")  # tu dataset original
    ds = build_conversational_ds(raw)
    train_ds = ds["train"]
    eval_ds = ds.get("eval")

    # Config GRPO (DeepSeek-like): varias generaciones por prompt, reward verificable
    args = GRPOConfig(
        output_dir="galicIA-v1-grpo",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_train_epochs=1.0,
        logging_steps=10,
        save_steps=200,
        remove_unused_columns=False,   # <- necesitamos 'structure' en la reward
        max_prompt_length=512,
        max_completion_length=180,
        num_generations=4,             # G: nº de completions por prompt
        temperature=0.8,
        top_p=0.95,
        # Por defecto beta=0.0 (sen KL), e scale_rewards=True; para evitar sesgo por
        # dificultade ao nivel de prompt, moitas veces pónse False (Dr.GRPO).
        scale_rewards=False,
        log_completions=True,
        report_to=None,                # wandb/… se queres logging externo
    )

    trainer = GRPOTrainer(
        model=policy,                  # pasamos o modelo PEFT xa cargado (evita double-PEFT)
        processing_class=tok,          # tokenizer (left padding, pad_token)
        reward_funcs=poem_reward,      # función de recompensa verificable
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
