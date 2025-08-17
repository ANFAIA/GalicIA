"""
Este script demuestra cómo continuar el entrenamiento de un modelo de recompensa
utilizando TRL y adaptadores LoRA. Parte de una base de datos que contiene
mensajes de usuario y especificaciones de estructura poética y construye un
conjunto de preferencias heurísticas para formar pares ``chosen``/``rejected``.

El modelo de partida es un LoRA ya entrenado (``pajon1/galicIA-v1``) con base
``unsloth/Qwen3-0.6B``. A partir de la configuración de este adaptador se
extraen los parámetros principales (rango, alfa, dropout y módulos objetivo)
observados en su archivo ``adapter_config.json``【642958274612479†L56-L124】. Para
que el adaptador siga evolucionando en una tarea de clasificación (reward
modeling) se carga el modelo base como un ``AutoModelForSequenceClassification``
con una única etiqueta y posteriormente se inyecta el adaptador LoRA con
``is_trainable=True``. De esta forma, sólo se actualizan las matrices de bajo
rango del adaptador sin modificar el modelo completo.

El entrenamiento se realiza con ``RewardTrainer`` de TRL. Se usa una fracción
pequeña del dataset para ahorrar recursos y demostrar el flujo. La función
``score_poem`` define una recompensa heurística basada en el número de sílabas
y rimas de cada línea, con pequeños ajustes para vocabulario gallego. La
función ``build_preference_dataset`` toma cada ejemplo del dataset y genera
dos candidatos de poema usando el modelo generativo (no mostrado aquí) y
selecciona el mejor según la heurística para construir los pares
``chosen``/``rejected``.
"""

import random
from typing import Dict, Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
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
    get_quantization_config,
    setup_chat_format,
)
from peft import PeftModel

# -----------------------------------------------------------------------------
# Heurísticas para medir la calidad de un poema respecto a su estructura
# -----------------------------------------------------------------------------
VOWELS = set("aeiouáéíóúàèìòùâêîôûäëïöüAEIOUÁÉÍÓÚÂÊÎÔÛÄËÏÖÜ")


def approx_syllables(word: str) -> int:
    """
    Cuenta grupos vocálicos como aproximación del número de sílabas.

    Esta heurística no es exacta pero ofrece una medida estable sin necesidad
    de listas de diccionario.
    """
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
    """
    Devuelve el número aproximado de sílabas de una línea sumando las sílabas
    de cada palabra.
    """
    return sum(approx_syllables(w) for w in line.strip().split())


def rhyme_key(line: str) -> str:
    """
    Genera una clave de rima simple tomando las últimas tres letras del último
    token alfanumérico de la línea. Se utiliza para comparar rimas entre
    diferentes líneas.
    """
    tokens = [t for t in ''.join(ch if ch.isalnum() else ' ' for ch in line.lower()).split() if t]
    if not tokens:
        return ""
    last = tokens[-1]
    return last[-3:] if len(last) >= 3 else last


def score_poem(text: str, structure) -> float:
    """
    Calcula una puntuación heurística de un poema comparando su estructura con
    la especificada.

    La puntuación penaliza desviaciones en el número de estrofas, número de
    líneas por estrofa, número de sílabas por línea y rimas. También añade un
    pequeño bonus si aparecen palabras comunes en gallego.

    Args:
        text: El poema generado, con estrofas separadas por una línea en blanco.
        structure: Lista de diccionarios, cada uno con claves 'syllables' y 'rhyme'
            para la estrofa correspondiente.

    Returns:
        Puntuación numérica (float) que cuanto mayor, mejor ajuste tiene.
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
        for j, line in enumerate(lines[: len(target_syllables)]):
            syl = approx_line_syllables(line)
            score -= abs(syl - target_syllables[j]) * 0.2
            if j < len(target_rhymes):
                lab = target_rhymes[j]
                rhyme_map.setdefault(lab, []).append(rhyme_key(line))

        for lab, keys in rhyme_map.items():
            if len(keys) > 1:
                base = keys[0]
                mismatches = sum(1 for k in keys[1:] if k != base)
                score -= mismatches * 0.5

    # Bonus por vocabulario común en gallego
    if any(tok in text.lower() for tok in [
        "non",
        "que",
        "coa",
        "noite",
        "lúa",
        "auga",
        "terra",
        "vento",
    ]):
        score += 0.3
    return float(score)


def messages_from_example(ex: Dict[str, Any], tokenizer) -> list:
    """
    Construye una conversación de chat a partir de un ejemplo del dataset.
    Inyecta una pista de la estructura a seguir (sílabas y rima) para ayudar al
    modelo generativo a producir poemas ajustados a las especificaciones.
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
        {"role": "user", "content": user_prompt + structure_hint},
    ]
    return messages


def generate_two_candidates(
    ex: Dict[str, Any],
    gen_model,
    gen_tok,
    max_new_tokens: int = 180,
    temperature: float = 0.9,
    top_p: float = 0.95,
) -> tuple[str, str]:
    """
    Genera dos candidatos distintos para un poema a partir de un ejemplo usando
    muestreo estocástico (top_p/temperature). La función devuelve los textos
    generados sin el prompt inicial para que se puedan puntuar y comparar.
    """
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


def build_preference_dataset(
    raw_ds: DatasetDict, gen_model_name: str, trust_remote_code: bool = True
) -> DatasetDict:
    """
    Construye un nuevo DatasetDict con columnas 'chosen' y 'rejected' a partir
    de un DatasetDict original. Para cada ejemplo se generan dos poemas y se
    selecciona el mejor según ``score_poem``.

    Args:
        raw_ds: DatasetDict original con columnas 'id', 'conversations' y
            'structure'.
        gen_model_name: Nombre del modelo generativo a usar para producir
            candidatos.
        trust_remote_code: Permitir código remoto al cargar el modelo generativo.

    Returns:
        DatasetDict con los mismos splits que ``raw_ds`` pero con columnas
        'chosen' y 'rejected'.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_tok = AutoTokenizer.from_pretrained(gen_model_name, trust_remote_code=trust_remote_code, use_fast=True)
    # Cargar modelo generativo (causal) para producir poemas. Usamos
    # AutoModelForCausalLM para que ``generate`` funcione correctamente.
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_name, trust_remote_code=trust_remote_code
    ).to(device)

    # Ajustar plantilla de chat
    if gen_tok.chat_template is None:
        gen_model, gen_tok = setup_chat_format(gen_model, gen_tok)

    def map_fn(example: Dict[str, Any]) -> Dict[str, str]:
        cand1, cand2 = generate_two_candidates(example, gen_model, gen_tok)
        sc1 = score_poem(cand1, example.get("structure") or [])
        sc2 = score_poem(cand2, example.get("structure") or [])
        # Desempate aleatorio si empatan
        if sc1 == sc2:
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
    # -------------------------------------------------------------------------
    # Parámetros fijos para el experimento
    # -------------------------------------------------------------------------
    # Nombre del dataset cargado con ``load_from_disk``. Debe contener
    # 'conversations' y 'structure'.
    script_args = ScriptArguments(dataset_name="poemas_GalicIA_est", dataset_config=None)

    # Configuración de entrenamiento para RewardTrainer. Se utilizan pocos pasos
    # y un tamaño de lote reducido para esta prueba de concepto.
    training_args = RewardConfig(
        output_dir="galicIA-v1-ref",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        gradient_checkpointing=True,
        learning_rate=1.0e-4,
        eval_strategy="steps",
        eval_steps=20,
        max_length=512,
        report_to=None,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=20,
    )
    # Evitar advertencia de PyTorch con checkpointing reentrante
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # -------------------------------------------------------------------------
    # Configuración del adaptador LoRA preexistente
    # -------------------------------------------------------------------------
    # Los siguientes valores se extrajeron del ``adapter_config.json`` de
    # ``pajon1/galicIA-v1``【642958274612479†L56-L124】. Mantener estos valores permite
    # continuar entrenando el mismo adaptador en una tarea diferente (SEQ_CLS).
    LORA_TARGET_MODULES = [
        "down_proj",
        "k_proj",
        "up_proj",
        "gate_proj",
        "q_proj",
        "o_proj",
        "v_proj",
    ]
    LORA_R = 700
    LORA_ALPHA = 1400
    LORA_DROPOUT = 0.0

    # Ruta del adaptador LoRA a cargar y ruta del modelo base original
    ADAPTER_NAME = "pajon1/galicIA-v1"
    BASE_MODEL_NAME = "unsloth/Qwen3-0.6B"

    # Creamos un ModelConfig con LoRA activado y parámetros del adaptador
    model_args = ModelConfig(
        model_name_or_path=ADAPTER_NAME,
        trust_remote_code=True,
        torch_dtype="auto",
        use_peft=True,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_target_modules=LORA_TARGET_MODULES,
        lora_task_type="SEQ_CLS",
    )

    # -------------------------------------------------------------------------
    # Carga del modelo base y del adaptador LoRA
    # -------------------------------------------------------------------------
    # Configuración de cuantización y distribución del modelo
    quantization_config = get_quantization_config(model_args)
    device_map = get_kbit_device_map() if quantization_config is not None else None
    model_kwargs = dict(
        revision=model_args.model_revision,
        device_map=device_map,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=(
            getattr(torch, model_args.torch_dtype)
            if isinstance(model_args.torch_dtype, str) and model_args.torch_dtype not in ["auto", None]
            else None
        ),
    )

    # Tokenizador y modelo base de clasificación
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, num_labels=1, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    # Alinear token de relleno
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # Añadir plantilla de chat si fuera necesario
    if tokenizer.chat_template is None:
        base_model, tokenizer = setup_chat_format(base_model, tokenizer)

    # Convertir el modelo base en un modelo PEFT cargando el adaptador
    # Se especifica ``is_trainable=True`` para que las matrices LoRA puedan
    # actualizarse durante el entrenamiento.
    model = PeftModel.from_pretrained(base_model, ADAPTER_NAME, is_trainable=True)

    # -------------------------------------------------------------------------
    # Carga y preparación del dataset
    # -------------------------------------------------------------------------
    raw_dataset = load_from_disk(script_args.dataset_name)
    # Usamos una fracción muy pequeña para pruebas rápidas
    sample_size = max(1, int(0.00005 * len(raw_dataset)))
    raw_dataset = raw_dataset.shuffle(seed=42).select(range(sample_size))

    # Normalizar a un DatasetDict con splits 'train' y 'test'
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
        # Asegurarse de que existen los splits requeridos
        if training_args.eval_strategy != "no" and script_args.dataset_test_split not in raw_dataset:
            tmp = raw_dataset[script_args.dataset_train_split].train_test_split(test_size=0.1, seed=42)
            raw_dataset = DatasetDict(train=tmp["train"], test=tmp["test"])
            script_args.dataset_train_split = "train"
            script_args.dataset_test_split = "test"

    # Construir dataset de preferencias si es necesario
    train_cols = set(raw_dataset[script_args.dataset_train_split].column_names)
    if not {"chosen", "rejected"}.issubset(train_cols):
        pref_ds = build_preference_dataset(
            raw_dataset,
            gen_model_name=ADAPTER_NAME,
            trust_remote_code=model_args.trust_remote_code,
        )
        dataset = pref_ds
    else:
        dataset = raw_dataset

    # Preparar dataset de evaluación si corresponde
    eval_ds = None
    if training_args.eval_strategy != "no" and script_args.dataset_test_split is not None:
        if script_args.dataset_test_split in dataset:
            eval_ds = dataset[script_args.dataset_test_split]

    # -------------------------------------------------------------------------
    # Entrenamiento con RewardTrainer
    # -------------------------------------------------------------------------
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_ds,
        # No pasamos ``peft_config`` porque el modelo ya es un PeftModel
    )
    trainer.train()

    # Guardar el modelo entrenado
    trainer.save_model(training_args.output_dir)

    # Evaluar y registrar métricas si hay conjunto de validación
    if training_args.eval_strategy != "no" and eval_ds is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Subir al Hub opcionalmente si se habilita ``push_to_hub`` en RewardConfig
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)