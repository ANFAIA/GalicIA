import random
import torch
from typing import Dict, Any

from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)

# ------------------------------------------------------------
# Utilidades mínimas para construir preferencias (heurístico)
# ------------------------------------------------------------
VOWELS = set("aeiouáéíóúàèìòùâêîôûäëïöüAEIOUÁÉÍÓÚÂÊÎÔÛÄËÏÖÜ")

def approx_syllables(word: str) -> int:
    # Heurística simple: contar grupos vocálicos (no exacto, pero estable)
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
    # Clave de rima muy simple: últimas 3 letras alfanuméricas del último token
    tokens = [t for t in ''.join(ch if ch.isalnum() else ' ' for ch in line.lower()).split() if t]
    if not tokens:
        return ""
    last = tokens[-1]
    return last[-3:] if len(last) >= 3 else last

def score_poem(text: str, structure) -> float:
    """
    text: poema completo (con saltos de línea).
    structure: lista de estrofas; cada una con {'syllables': [7,5], 'rhyme': ['A','B']} etc.
    Recompensa = ajuste de sílabas - penalizaciones por desviación de rima/formato.
    """
    stanzas = [s for s in text.strip().split("\n\n") if s.strip()]
    target_stanzas = len(structure)
    score = -abs(len(stanzas) - target_stanzas) * 1.0

    for i, st in enumerate(stanzas[:target_stanzas]):
        lines = [l for l in st.strip().split("\n") if l.strip()]
        target_syllables = structure[i].get("syllables", [])
        target_rhymes = structure[i].get("rhyme", [])
        score -= abs(len(lines) - len(target_syllables)) * 0.5

        rhyme_map = {}
        for j, line in enumerate(lines[:len(target_syllables)]):
            syl = approx_line_syllables(line)
            score -= abs(syl - target_syllables[j]) * 0.2  # tolerante
            if j < len(target_rhymes):
                lab = target_rhymes[j]
                rhyme_map.setdefault(lab, []).append(rhyme_key(line))

        for lab, keys in rhyme_map.items():
            if len(keys) > 1:
                base = keys[0]
                mismatches = sum(1 for k in keys[1:] if k != base)
                score -= mismatches * 0.5

    # Bonus pequeño por vocabulario común en gl
    if any(tok in text.lower() for tok in ["non", "que", "coa", "noite", "lúa", "auga", "terra", "vento"]):
        score += 0.3
    return float(score)

def messages_from_example(ex: Dict[str, Any], tokenizer):
    """
    Construye mensajes de chat a partir del ejemplo de tu dataset:
    {
      'conversations': {'role': ['user'], 'content': ['...']},
      'structure': [...]
    }
    """
    user_prompt = ex["conversations"]["content"][0]
    structure_hint = ""
    try:
        patt = ex.get("structure") or []
        patt_str = "; ".join(
            f"{'/'.join(map(str, p.get('syllables', [])))} con rima {''.join(p.get('rhyme', []))}"
            for p in patt
        )
        if patt_str:
            structure_hint = f"\n\nSegue esta estrutura: {patt_str}. Escribe en galego."
    except Exception:
        pass

    messages = [
        {"role": "user", "content": user_prompt + structure_hint}
    ]
    return messages

def generate_two_candidates(ex, gen_model, gen_tok, max_new_tokens=180, temperature=0.9, top_p=0.95):
    messages = messages_from_example(ex, gen_tok)
    prompt = gen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = gen_tok(prompt, return_tensors="pt")
    inputs = {k: v.to(gen_model.device) for k, v in inputs.items()}

    pad_id = gen_tok.pad_token_id
    if pad_id is None and gen_tok.eos_token_id is not None:
        pad_id = gen_tok.eos_token_id

    out1 = gen_model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=pad_id,
    )
    out2 = gen_model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=pad_id,
    )

    gen_start = inputs["input_ids"].shape[1]
    txt1 = gen_tok.decode(out1[0][gen_start:], skip_special_tokens=True).strip()
    txt2 = gen_tok.decode(out2[0][gen_start:], skip_special_tokens=True).strip()
    return txt1, txt2

def build_preference_dataset(raw_ds: DatasetDict, gen_model_name: str, trust_remote_code: bool = True) -> DatasetDict:
    """
    Toma tu DatasetDict con columnas:
      - id
      - conversations: {'role': [...], 'content': [...]}
      - structure: [...]
    y devuelve un DatasetDict con columnas:
      - chosen
      - rejected
    usando una heurística de recompensa sobre estructura/rima/sílabas.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_tok = AutoTokenizer.from_pretrained(gen_model_name, trust_remote_code=trust_remote_code, use_fast=True)
    gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name, trust_remote_code=trust_remote_code).to(device)

    # Alinear plantilla de chat si no existe
    if gen_tok.chat_template is None:
        gen_model, gen_tok = setup_chat_format(gen_model, gen_tok)

    def map_fn(example):
        cand1, cand2 = generate_two_candidates(example, gen_model, gen_tok)
        sc1 = score_poem(cand1, example.get("structure") or [])
        sc2 = score_poem(cand2, example.get("structure") or [])
        if sc1 == sc2:
            # desempate aleatorio suave
            if random.random() < 0.5:
                sc1 += 1e-6
            else:
                sc2 += 1e-6
        if sc1 > sc2:
            chosen, rejected = cand1, cand2
        else:
            chosen, rejected = cand2, cand1
        return {"chosen": chosen, "rejected": rejected}

    out = {}
    for split in raw_ds.keys():
        cols = raw_ds[split].column_names
        out[split] = raw_ds[split].map(map_fn, remove_columns=cols)
    return DatasetDict(out)

if __name__ == "__main__":
    # -----------------------------
    # Configuración fija (sin CLI)
    # -----------------------------
    script_args = ScriptArguments(
        dataset_name="poemas_GalicIA_est",
        dataset_config=None,
    )

    training_args = RewardConfig(
        output_dir="galicIA-v1-ref",
        per_device_train_batch_size=8,
        num_train_epochs=1,
        gradient_checkpointing=True,
        learning_rate=1.0e-4,
        eval_strategy="steps",
        eval_steps=50,
        max_length=512,
        report_to=None,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
    )
    # Evitar warning de PyTorch reentrant con checkpointing
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    model_args = ModelConfig(
        model_name_or_path="pajon1/galicIA-v1",
        trust_remote_code=True,
        torch_dtype="auto",
        # Forzamos LoRA dentro del código:
        use_peft=False,
        lora_r=700,
        lora_alpha=1400,
        lora_dropout=0.0,
        lora_task_type="SEQ_CLS",  # IMPORTANTE para reward modeling
        # load_in_4bit=True,
    )

    # -----------------------------
    # Modelo y Tokenizer (reward)
    # -----------------------------
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,

    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=1, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    # Alinear padding tokens
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Si el modelo base no trae plantilla de chat, aplicar ChatML por defecto
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # -----------------------------
    # Dataset
    # -----------------------------
    raw_dataset = load_from_disk(script_args.dataset_name)
    # Cargar solo el 10% de forma aleatoria
    raw_dataset = raw_dataset.shuffle(seed=42).select(range(int(0.00005 * len(raw_dataset))))

    # Normalizar a DatasetDict con 'train' y opcionalmente 'test'
    if isinstance(raw_dataset, Dataset):
        if training_args.eval_strategy != "no":
            tmp = raw_dataset.train_test_split(test_size=0.1, seed=42)
            raw_dataset = DatasetDict(train=tmp["train"], test=tmp["test"])
            script_args.dataset_train_split = "train"
            script_args.dataset_test_split = "test"
        else:
            raw_dataset = DatasetDict(train=raw_dataset)
            script_args.dataset_train_split = "train"
            script_args.dataset_test_split = None
    else:
        # Ya es DatasetDict; aseguramos que existen los splits requeridos
        if training_args.eval_strategy != "no" and script_args.dataset_test_split not in raw_dataset:
            # Si no hay test, crear uno 10%
            tmp = raw_dataset[script_args.dataset_train_split].train_test_split(test_size=0.1, seed=42)
            raw_dataset = DatasetDict(train=tmp["train"], test=tmp["test"])
            script_args.dataset_train_split = "train"
            script_args.dataset_test_split = "test"

    # Si el dataset NO trae columnas 'chosen'/'rejected', construimos preferencias on-the-fly
    train_cols = set(raw_dataset[script_args.dataset_train_split].column_names)
    needs_prefs = not {"chosen", "rejected"}.issubset(train_cols)
    if needs_prefs:
        pref_ds = build_preference_dataset(
            raw_dataset,
            gen_model_name=model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
        )
        dataset = pref_ds
    else:
        dataset = raw_dataset

    # Preparar eval_dataset de forma segura
    eval_ds = None
    if training_args.eval_strategy != "no" and script_args.dataset_test_split is not None:
        if script_args.dataset_test_split in dataset:
            eval_ds = dataset[script_args.dataset_test_split]

    # -----------------------------
    # Entrenamiento
    # -----------------------------
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_ds,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()

    # -----------------------------
    # Guardado y evaluación
    # -----------------------------
    trainer.save_model(training_args.output_dir)

    if training_args.eval_strategy != "no" and eval_ds is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Push opcional si configuras push_to_hub=True en RewardConfig:
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)