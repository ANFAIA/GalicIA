import random
import torch
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple

from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import (
    DPOTrainer,
    DPOConfig,
    setup_chat_format,
)

# Importar las funciones de evaluación métrica
from src.extractor_métrica.procesar_poema import rango_silabas, rima_consonante, rima_asonante


# ==========================================
# FUNCIONES DE EVALUACIÓN MÉTRICA
# ==========================================

def extract_syllable_count(verso: str) -> int:
    """Extraer número de sílabas de un verso"""
    mn, mx = rango_silabas(verso)
    return min,mx


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


def score_poem(text: str, structure: List[Dict]) -> float:
    """
    Función simplificada de puntuación para evaluar poemas.
    Mayor puntuación = mejor adherencia a la estructura.
    """
    if not text.strip():
        return -10.0

    if not structure or not isinstance(structure, list):
        return 0.0

    score = 0.0
    stanzas = [s.strip() for s in text.strip().split("\n\n") if s.strip()]

    # Bonus/penalización por número correcto de estrofas
    expected_stanzas = len(structure)
    score -= abs(len(stanzas) - expected_stanzas) * 2.0

    # Evaluar cada estrofa
    for i, stanza in enumerate(stanzas[:expected_stanzas]):
        if i >= len(structure):
            break

        lines = [line.strip() for line in stanza.split("\n") if line.strip()]
        expected_syllables = structure[i].get("syllables", [])
        expected_rhymes = structure[i].get("rhyme", [])

        # Validar estructura
        if expected_syllables:
            expected_syllables = [s for s in expected_syllables
                                  if s is not None and isinstance(s, (int, float))]
        if expected_rhymes:
            expected_rhymes = [r for r in expected_rhymes
                               if r is not None and isinstance(r, str)]

        # Penalización por número incorrecto de versos
        if expected_syllables:
            score -= abs(len(lines) - len(expected_syllables)) * 1.5

        # Evaluar cada verso
        rhyme_groups = {}
        for j, line in enumerate(lines):
            if not line.strip():
                score -= 1.0
                continue

            # Evaluar sílabas
            if j < len(expected_syllables):
                try:
                    min_sil,max_sil = extract_syllable_count(line)
                    expected_syl = expected_syllables[j]
                    actual_syllables=-1
                    if min_sil>expected_syl:
                        actual_syllables=min_sil
                    if max_sil<expected_syl:
                        actual_syllables=max_sil
                    if min_sil < expected_syl < max_sil:
                        actual_syllables=expected_syllables
                    syllable_diff = abs(actual_syllables - expected_syl)

                    if syllable_diff == 0:
                        score += 2.0  # Bonus por exactitud
                    elif syllable_diff <= 1:
                        score += 0.5  # Pequeño bonus por estar cerca
                    else:
                        score -= syllable_diff * 0.5
                except:
                    score -= 0.5

            # Recopilar rimas para evaluación posterior
            if j < len(expected_rhymes) and expected_rhymes[j]:
                rhyme_label = expected_rhymes[j]
                try:
                    rhyme_pattern = extract_rhyme_pattern(line, "asonante")
                    if rhyme_label not in rhyme_groups:
                        rhyme_groups[rhyme_label] = []
                    rhyme_groups[rhyme_label].append(rhyme_pattern)
                except:
                    pass

        # Evaluar consistencia de rimas
        for rhyme_label, patterns in rhyme_groups.items():
            if len(patterns) <= 1:
                continue
            unique_patterns = set(p for p in patterns if p)
            if len(unique_patterns) == 1:
                score += len(patterns) * 1.0  # Bonus por rima consistente
            else:
                score -= (len(unique_patterns) - 1) * 0.5

    return float(score)


# ==========================================
# GENERACIÓN DE DATASET DE PREFERENCIAS
# ==========================================

def generate_candidates(
        prompt: str,
        model,
        tokenizer,
        num_candidates: int = 4,
        max_new_tokens: int = 150,
        temperature: float = 0.9,
        top_p: float = 0.95
) -> List[str]:
    """Generar múltiples candidatos para un prompt"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    candidates = []
    for i in range(num_candidates):
        torch.manual_seed(random.randint(0, 100000))

        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        candidates.append(generated_text)

    return candidates


def create_preference_dataset(
        dataset: Dataset,
        model_name: str,
        num_candidates: int = 4,
        sample_size: int = None
) -> Dataset:
    """
    Crear dataset de preferencias generando y evaluando candidatos.
    Genera múltiples candidatos y selecciona el mejor/peor para mayor señal de aprendizaje.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Cargar modelo y tokenizer para generación
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    # Configurar chat template si es necesario
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Limitar tamaño del dataset si se especifica
    if sample_size:
        dataset = dataset.shuffle(seed=42).select(range(min(sample_size, len(dataset))))

    preference_data = []

    for idx, example in enumerate(dataset):
        print(f"Processing example {idx + 1}/{len(dataset)}")

        try:
            # Obtener el prompt del usuario
            user_prompt = example["conversations"]["content"][0]
            structure = example.get("structure", [])

            # Crear prompt con formato de chat
            messages = [{"role": "user", "content": user_prompt}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Generar múltiples candidatos
            candidates = generate_candidates(
                prompt, model, tokenizer,
                num_candidates=num_candidates
            )

            # Evaluar cada candidato
            scored_candidates = []
            for candidate in candidates:
                score = score_poem(candidate, structure)
                scored_candidates.append((score, candidate))

            # Ordenar por puntuación
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            # Seleccionar el mejor y el peor para máximo contraste
            best_score, best_poem = scored_candidates[0]
            worst_score, worst_poem = scored_candidates[-1]

            # Solo agregar si hay diferencia significativa
            if best_score - worst_score > 1.0:
                preference_data.append({
                    "prompt": prompt,
                    "chosen": best_poem,
                    "rejected": worst_poem,
                    "score_diff": best_score - worst_score,
                    "structure": structure
                })

                print(f"  Score diff: {best_score - worst_score:.2f}")
            else:
                print(f"  Skipped: insufficient score difference")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"Created {len(preference_data)} preference pairs from {len(dataset)} examples")

    return Dataset.from_list(preference_data)


# ==========================================
# ENTRENAMIENTO DPO
# ==========================================

def train_dpo_model(
        base_model_name: str,
        dataset_path: str,
        output_dir: str = "galicia-dpo-model",
        num_train_epochs: int = 2,
        per_device_train_batch_size: int = 2,
        learning_rate: float = 5e-7,
        beta: float = 0.1,  # Parámetro DPO que controla cuánto desviarse del modelo de referencia
        max_samples: int = 1000,  # Limitar samples para pruebas
):
    """
    Entrenar modelo con DPO para seguir estructuras métricas.

    DPO es más eficiente que PPO porque:
    - No necesita un reward model separado
    - Es más estable en el entrenamiento
    - Requiere menos memoria
    - Converge más rápido
    """

    print("=" * 60)
    print("INICIANDO ENTRENAMIENTO DPO")
    print("=" * 60)

    # Cargar dataset
    print("\n1. Cargando dataset...")
    raw_dataset = load_from_disk(dataset_path)

    # Crear dataset de preferencias
    print("\n2. Creando dataset de preferencias...")
    pref_dataset = create_preference_dataset(
        raw_dataset,
        base_model_name,
        num_candidates=4,  # Generar 4 candidatos por ejemplo
        sample_size=max_samples
    )

    # Dividir en train/eval
    print("\n3. Dividiendo dataset...")
    split = pref_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"  Train: {len(train_dataset)} ejemplos")
    print(f"  Eval: {len(eval_dataset)} ejemplos")

    # Cargar modelo y tokenizer
    print("\n4. Cargando modelo base...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configurar argumentos de entrenamiento DPO
    print("\n5. Configurando entrenamiento...")
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=4,  # Para simular batch más grande
        gradient_checkpointing=True,  # Ahorrar memoria
        learning_rate=learning_rate,
        beta=beta,  # Parámetro específico de DPO
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        bf16=torch.cuda.is_available(),  # Usar bf16 si hay GPU
        report_to="none",  # Cambiar a "wandb" si quieres logging
        remove_unused_columns=False,
        max_length=512,
        max_prompt_length=256,
    )

    # Crear trainer DPO
    print("\n6. Creando DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # Cambiado de 'tokenizer' a 'processing_class'
        # El modelo de referencia se crea automáticamente como una copia del modelo inicial
    )

    # Entrenar
    print("\n7. Iniciando entrenamiento...")
    print("=" * 60)
    trainer.train()

    # Guardar modelo final
    print("\n8. Guardando modelo...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Mostrar métricas finales
    print("\n9. Evaluación final...")
    metrics = trainer.evaluate()
    print(f"Métricas finales: {metrics}")

    print("\n" + "=" * 60)
    print(f"ENTRENAMIENTO COMPLETADO")
    print(f"Modelo guardado en: {output_dir}")
    print("=" * 60)

    return trainer


# ==========================================
# FUNCIÓN DE INFERENCIA PARA PROBAR
# ==========================================

def generate_poem_with_structure(
        model_path: str,
        prompt: str,
        structure: List[Dict],
        temperature: float = 0.7,
        max_new_tokens: int = 150
) -> str:
    """
    Generar un poema usando el modelo entrenado con DPO.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cargar modelo entrenado
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Crear prompt con formato de chat
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generar
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Evaluar el poema generado
    score = score_poem(generated_text, structure)

    print(f"Poema generado (puntuación: {score:.2f}):")
    print("-" * 40)
    print(generated_text)
    print("-" * 40)

    return generated_text


# ==========================================
# MAIN - EJEMPLO DE USO
# ==========================================

if __name__ == "__main__":
    # Configuración
    BASE_MODEL = "galicIA-full-FIM"  # Tu modelo base
    DATASET_PATH = "poemas_GalicIA_est"  # Tu dataset con estructuras
    OUTPUT_DIR = "galicia-dpo-structured"  # Donde guardar el modelo entrenado

    # Entrenar modelo con DPO
    trainer = train_dpo_model(
        base_model_name=BASE_MODEL,
        dataset_path=DATASET_PATH,
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        learning_rate=5e-7,
        beta=0.1,  # Controla cuánto puede desviarse del modelo original
        max_samples=2,  # Usar 500 ejemplos para entrenamiento rápido
    )