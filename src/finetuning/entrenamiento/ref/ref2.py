import random
import torch
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple

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

# Importar las funciones de evaluación métrica más precisas
from src.extractor_métrica.procesar_poema import rango_silabas, rima_consonante, rima_asonante


# Configurar logging
def setup_logging(output_dir: str):
    """Configurar archivo de log para el entrenamiento"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training_log.txt")

    # Escribir encabezado del log
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"GALICIA RL TRAINING LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

    return log_file


def log_training_step(log_file: str, step: int, example: Dict, prompt: str,
                      cand1: str, cand2: str, sc1: float, sc2: float,
                      chosen: str, rejected: str, structure: List[Dict]):
    """Registrar detalles de cada paso del entrenamiento"""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"TRAINING STEP {step}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 60}\n\n")

        # Prompt original del usuario (sin modificaciones)
        f.write("PROMPT ORIGINAL:\n")
        f.write("-" * 40 + "\n")
        original_prompt = example.get("conversations", {}).get("content", [""])[0]
        f.write(f"{original_prompt}\n\n")

        # Estructura métrica
        f.write("ESTRUCTURA MÉTRICA:\n")
        f.write("-" * 40 + "\n")
        if structure:
            for i, stanza in enumerate(structure):
                syllables = stanza.get("syllables", [])
                rhymes = stanza.get("rhyme", [])
                f.write(f"Estrofa {i + 1}:\n")
                f.write(f"  Sílabas: {syllables}\n")
                f.write(f"  Rimas: {rhymes}\n")
        else:
            f.write("No hay estructura definida\n")
        f.write("\n")

        # Prompt completo usado para generación (debería ser igual al original ahora)
        f.write("PROMPT USADO PARA GENERACIÓN:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{prompt}\n\n")

        # Candidato 1
        f.write("CANDIDATO 1:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Puntuación: {sc1:.3f}\n")
        f.write(f"Texto:\n{cand1}\n\n")

        # Candidato 2
        f.write("CANDIDATO 2:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Puntuación: {sc2:.3f}\n")
        f.write(f"Texto:\n{cand2}\n\n")

        # Resultado de la comparación
        f.write("RESULTADO:\n")
        f.write("-" * 40 + "\n")
        winner = "CANDIDATO 1" if sc1 > sc2 else "CANDIDATO 2"
        f.write(f"Ganador: {winner} (diff: {abs(sc1 - sc2):.3f})\n")
        f.write(f"Elegido: {chosen[:100]}...\n")
        f.write(f"Rechazado: {rejected[:100]}...\n\n")


def extract_syllable_count(verso: str) -> int:
    """Extraer número de sílabas de un verso usando rango_silabas y tomando el valor medio"""
    mn, mx = rango_silabas(verso)
    return (mn + mx) // 2  # Tomar el valor medio del rango


def extract_rhyme_pattern(verso: str, rhyme_type: str = "asonante") -> str:
    """Extraer patrón de rima de un verso"""
    if not verso.strip():
        return ""

    last_word = verso.split()[-1] if verso.split() else ""
    if not last_word:
        return ""

    if rhyme_type == "consonante":
        return rima_consonante(last_word)
    else:
        return rima_asonante(last_word)


def score_poem_advanced(text: str, structure: List[Dict]) -> float:
    """
    Función de puntuación usando evaluación métrica (sin bonus léxico gallego ni penalización por longitud).

    Args:
        text: poema completo (con saltos de línea)
        structure: lista de estrofas con estructura esperada

    Returns:
        float: puntuación del poema (mayor es mejor)
    """
    if not text.strip():
        return -10.0

    # Validar y limpiar estructura
    if not structure or not isinstance(structure, list):
        return -5.0

    valid_structure = []
    for stanza_info in structure:
        if not isinstance(stanza_info, dict):
            continue

        syllables = stanza_info.get("syllables", [])
        rhymes = stanza_info.get("rhyme", [])

        # Filtrar valores None y validar tipos
        if syllables and isinstance(syllables, list):
            syllables = [s for s in syllables if s is not None and isinstance(s, (int, float))]
        else:
            syllables = []

        if rhymes and isinstance(rhymes, list):
            rhymes = [r for r in rhymes if r is not None and isinstance(r, str)]
        else:
            rhymes = []

        if syllables or rhymes:  # Incluir si tiene al menos sílabas o rimas válidas
            valid_structure.append({
                "syllables": syllables,
                "rhyme": rhymes
            })

    if not valid_structure:
        return -5.0

    # Dividir en estrofas
    stanzas = [s.strip() for s in text.strip().split("\n\n") if s.strip()]
    expected_stanzas = len(valid_structure)

    # Penalización por número incorrecto de estrofas
    score = -abs(len(stanzas) - expected_stanzas) * 2.0

    # Evaluar cada estrofa
    for i, stanza in enumerate(stanzas[:expected_stanzas]):
        if i >= len(valid_structure):
            break

        lines = [line.strip() for line in stanza.split("\n") if line.strip() or line == ""]
        expected_structure = valid_structure[i]
        expected_syllables = expected_structure.get("syllables", [])
        expected_rhymes = expected_structure.get("rhyme", [])

        # Si no hay estructura válida para esta estrofa, seguir sin modificar score
        if not expected_syllables and not expected_rhymes:
            continue

        # Penalización por número incorrecto de versos (solo si tenemos syllables esperadas)
        if expected_syllables:
            score -= abs(len([l for l in lines if l.strip()]) - len(expected_syllables)) * 1.5

        # Evaluar sílabas y rimas de cada verso
        rhyme_groups = {}

        for j, line in enumerate(lines):
            if not line.strip():
                score -= 2.0
                continue

            # Evaluar sílabas
            if j < len(expected_syllables):
                try:
                    actual_syllables = extract_syllable_count(line)
                    expected_syl = expected_syllables[j]
                    syllable_diff = abs(actual_syllables - expected_syl)

                    # Penalización escalada por diferencia de sílabas
                    if syllable_diff == 0:
                        score += 1.0  # Bonus por sílabas exactas
                    elif syllable_diff <= 1:
                        score -= 0.3  # Penalización leve por 1 sílaba de diferencia
                    elif syllable_diff <= 2:
                        score -= 0.8  # Penalización moderada
                    else:
                        score -= syllable_diff * 0.5  # Penalización fuerte
                except Exception as e:
                    print(f"Error evaluating syllables for line '{line}': {e}")
                    score -= 1.0  # Penalización por error

            # Evaluar rimas (etiquetas iguales deben rimar de forma consistente)
            if j < len(expected_rhymes) and expected_rhymes[j] is not None:
                rhyme_label = expected_rhymes[j]
                try:
                    rhyme_pattern = extract_rhyme_pattern(line, "asonante")
                    if rhyme_label not in rhyme_groups:
                        rhyme_groups[rhyme_label] = []
                    rhyme_groups[rhyme_label].append(rhyme_pattern)
                except Exception as e:
                    print(f"Error evaluating rhyme for line '{line}': {e}")
                    score -= 0.5  # Penalización menor por error de rima

        # Evaluar consistencia de rimas dentro de cada grupo
        for rhyme_label, patterns in rhyme_groups.items():
            if len(patterns) <= 1:
                continue

            unique_patterns = set(p for p in patterns if p)  # Excluir patrones vacíos

            if len(unique_patterns) == 0:
                score -= 1.0  # Sin patrón de rima
            elif len(unique_patterns) == 1:
                score += len(patterns) * 0.8  # Rima consistente
            else:
                inconsistency_penalty = (len(unique_patterns) - 1) * 0.7
                score -= inconsistency_penalty  # Rima inconsistente

    return float(score)



def messages_from_example(ex: Dict[str, Any], tokenizer):
    """
    Construye mensajes de chat a partir del ejemplo del dataset
    """
    # Usar SOLO el prompt original del dataset, sin modificaciones
    user_prompt = ex["conversations"]["content"][0]

    messages = [
        {"role": "user", "content": user_prompt}
    ]
    return messages


def generate_two_candidates(ex, gen_model, gen_tok, max_new_tokens=150, temperature=0.9, top_p=0.95):
    """Generar dos candidatos de poemas para comparación"""
    messages = messages_from_example(ex, gen_tok)

    prompt = gen_tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # Tokenizar el prompt
    inputs = gen_tok(
        prompt,
        return_tensors="pt"
    )
    inputs = {k: v.to(gen_model.device) for k, v in inputs.items()}

    pad_id = gen_tok.pad_token_id
    if pad_id is None and gen_tok.eos_token_id is not None:
        pad_id = gen_tok.eos_token_id

    # Generar dos candidatos con diferentes seeds para mayor diversidad
    torch.manual_seed(random.randint(0, 10000))
    out1 = gen_model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=pad_id,
        repetition_penalty=1.6,  # Añadir repetition_penalty como en tu ejemplo
    )

    torch.manual_seed(random.randint(0, 10000))
    out2 = gen_model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=pad_id,
        repetition_penalty=1.6,  # Añadir repetition_penalty como en tu ejemplo
    )

    gen_start = inputs["input_ids"].shape[1]
    txt1 = gen_tok.decode(out1[0][gen_start:], skip_special_tokens=True).strip()
    txt2 = gen_tok.decode(out2[0][gen_start:], skip_special_tokens=True).strip()

    return prompt, txt1, txt2


def build_preference_dataset(raw_ds: DatasetDict, gen_model_name: str, output_dir: str,
                             trust_remote_code: bool = True) -> DatasetDict:
    """
    Construir dataset de preferencias usando evaluación métrica avanzada
    """
    # Configurar logging
    log_file = setup_logging(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_tok = AutoTokenizer.from_pretrained(gen_model_name, trust_remote_code=trust_remote_code, use_fast=True)
    gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name, trust_remote_code=trust_remote_code).to(device)

    # Alinear plantilla de chat si no existe
    if gen_tok.chat_template is None:
        gen_model, gen_tok = setup_chat_format(gen_model, gen_tok)

    step_counter = 0

    def map_fn(example):
        nonlocal step_counter
        step_counter += 1

        try:
            prompt, cand1, cand2 = generate_two_candidates(example, gen_model, gen_tok)
            structure = example.get("structure", [])

            sc1 = score_poem_advanced(cand1, structure)
            sc2 = score_poem_advanced(cand2, structure)

            # Desempate con pequeña aleatoriedad
            if abs(sc1 - sc2) < 1e-6:
                if random.random() < 0.5:
                    sc1 += 1e-6
                else:
                    sc2 += 1e-6

            if sc1 > sc2:
                chosen, rejected = cand1, cand2
                score_diff = sc1 - sc2
            else:
                chosen, rejected = cand2, cand1
                score_diff = sc2 - sc1

            print(f"Step {step_counter} - Scores: {sc1:.2f} vs {sc2:.2f} (diff: {score_diff:.2f})")

            # Log detallado
            log_training_step(
                log_file, step_counter, example, prompt,
                cand1, cand2, sc1, sc2, chosen, rejected, structure
            )

            return {"chosen": chosen, "rejected": rejected}

        except Exception as e:
            print(f"Error processing example: {e}")
            # Log del error
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\nERROR en step {step_counter}: {str(e)}\n")

            # Fallback en caso de error
            return {"chosen": "Error en generación", "rejected": "Error en generación"}

    out = {}
    for split in raw_ds.keys():
        print(f"Processing split: {split}")
        cols = raw_ds[split].column_names
        out[split] = raw_ds[split].map(map_fn, remove_columns=cols)

    # Log final
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"TRAINING COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total steps processed: {step_counter}\n")
        f.write(f"{'=' * 80}\n")

    return DatasetDict(out)


if __name__ == "__main__":
    # -----------------------------
    # Configuración
    # -----------------------------
    script_args = ScriptArguments(
        dataset_name="poemas_GalicIA_est",
        dataset_config=None,
    )

    training_args = RewardConfig(
        output_dir="galicIA-v1-ref-advanced",
        per_device_train_batch_size=4,  # Reducido para mayor estabilidad
        num_train_epochs=1,
        gradient_checkpointing=True,
        learning_rate=5.0e-5,  # Learning rate más conservativo
        eval_strategy="steps",
        eval_steps=100,
        max_length=512,
        report_to=None,
        logging_strategy="steps",
        logging_steps=20,
        save_strategy="steps",
        save_steps=100,
        warmup_steps=50,  # Añadir warmup
    )
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    model_args = ModelConfig(
        model_name_or_path="pajon1/galicIA-v1",
        trust_remote_code=True,
        torch_dtype="auto",
        use_peft=False,
        lora_r=700,
        lora_alpha=1400,
        lora_dropout=0.05,  # Añadir algo de dropout
        lora_task_type="SEQ_CLS",
    )

    # -----------------------------
    # Modelo y Tokenizer
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

    # Plantilla de chat
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # -----------------------------
    # Dataset
    # -----------------------------
    raw_dataset = load_from_disk(script_args.dataset_name)
    # Usar una muestra más pequeña para pruebas (0.005% del dataset)
    sample_size = int(0.00005 * len(raw_dataset))
    raw_dataset = raw_dataset.shuffle(seed=42).select(range(min(sample_size, 100)))  # Max 100 ejemplos para pruebas

    print(f"Using {len(raw_dataset)} examples for training")

    # Normalizar a DatasetDict
    if isinstance(raw_dataset, Dataset):
        if training_args.eval_strategy != "no":
            tmp = raw_dataset.train_test_split(test_size=0.2, seed=42)  # 20% para eval
            raw_dataset = DatasetDict(train=tmp["train"], test=tmp["test"])
            script_args.dataset_train_split = "train"
            script_args.dataset_test_split = "test"
        else:
            raw_dataset = DatasetDict(train=raw_dataset)
            script_args.dataset_train_split = "train"
            script_args.dataset_test_split = None

    # Construir dataset de preferencias
    train_cols = set(raw_dataset[script_args.dataset_train_split].column_names)
    needs_prefs = not {"chosen", "rejected"}.issubset(train_cols)

    if needs_prefs:
        print("Building preference dataset with advanced metric evaluation...")
        pref_ds = build_preference_dataset(
            raw_dataset,
            gen_model_name=model_args.model_name_or_path,
            output_dir=training_args.output_dir,
            trust_remote_code=model_args.trust_remote_code,
        )
        dataset = pref_ds
    else:
        dataset = raw_dataset

    # Preparar eval_dataset
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

    print("Starting training...")
    trainer.train()

    # -----------------------------
    # Guardado y evaluación
    # -----------------------------
    trainer.save_model(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")

    if training_args.eval_strategy != "no" and eval_ds is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        print("Evaluation metrics:", metrics)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)