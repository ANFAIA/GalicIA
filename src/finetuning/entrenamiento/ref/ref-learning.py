import random
import torch
from typing import Dict, List, Tuple, Optional

from datasets import load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)
from trl import (
    DPOTrainer,
    DPOConfig,
    setup_chat_format,
)

# Importar las funciones de evaluación métrica
from src.extractor_métrica.procesar_poema import rango_silabas, rima_consonante, rima_asonante


# ==========================================
# UTILIDADES
# ==========================================

def pick_dtype():
    """Elige dtype estable según la GPU."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32bit



class NanInfGuard(LogitsProcessor):
    """Protege los logits de NaN/Inf y desbordes absurdos."""
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        return scores.clamp(min=-1e4, max=1e4)


def ensure_tokens(tokenizer):
    """Asegura pad/eos token ids válidos."""
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.eos_token_id is None and tokenizer.pad_token_id is not None:
        tokenizer.eos_token_id = tokenizer.pad_token_id
    # Si faltan ambos, crea un token especial de pad
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is None:
        if "<pad>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = tokenizer.pad_token_id


# ==========================================
# FUNCIONES DE EVALUACIÓN MÉTRICA
# ==========================================

def extract_syllable_count(verso: str) -> Tuple[int, int]:
    """Extraer rango de número de sílabas de un verso"""
    try:
        mn, mx = rango_silabas(verso)
        return mn, mx
    except Exception as e:
        print(f"Error calculating syllables for '{verso}': {e}")
        return 0, 0


def extract_rhyme_pattern(verso: str, rhyme_type: str = "asonante") -> str:
    """Extraer patrón de rima de un verso"""
    if not verso.strip():
        return ""

    words = verso.split()
    if not words:
        return ""

    last_word = words[-1].rstrip('.,!?;:')  # Remove punctuation
    if not last_word:
        return ""

    try:
        if rhyme_type == "consonante":
            return rima_consonante(last_word)
        else:
            return rima_asonante(last_word)
    except Exception as e:
        print(f"Error extracting rhyme from '{last_word}': {e}")
        return ""


def score_poem(text: str, structure: List[Dict]) -> float:
    """
    Función de puntuación para evaluar poemas.
    Mayor puntuación = mejor adherencia a la estructura.
    """
    if not text.strip():
        return -10.0

    if not structure or not isinstance(structure, list):
        return 0.0

    score = 0.0
    stanzas = [s.strip() for s in text.strip().split("\n\n") if s.strip()]

    # Penalización por número incorrecto de estrofas
    expected_stanzas = len(structure)
    score -= abs(len(stanzas) - expected_stanzas) * 2.0

    # Evaluar cada estrofa
    for i, stanza in enumerate(stanzas[:expected_stanzas]):
        if i >= len(structure):
            break

        lines = [line.strip() for line in stanza.split("\n") if line.strip()]
        expected_syllables = structure[i].get("syllables", [])
        expected_rhymes = structure[i].get("rhyme", [])

        # Validar y filtrar estructuras
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
                    min_sil, max_sil = extract_syllable_count(line)
                    expected_syl = int(expected_syllables[j])

                    # Lógica para determinar sílabas reales
                    if min_sil <= expected_syl <= max_sil:
                        actual_syllables = expected_syl  # Perfecto
                    elif expected_syl < min_sil:
                        actual_syllables = min_sil  # Muy pocas sílabas esperadas
                    else:  # expected_syl > max_sil
                        actual_syllables = max_sil  # Demasiadas sílabas esperadas

                    syllable_diff = abs(actual_syllables - expected_syl)

                    if syllable_diff == 0:
                        score += 2.0  # Bonus por exactitud
                    elif syllable_diff <= 1:
                        score += 0.5  # Pequeño bonus por estar cerca
                    else:
                        score -= syllable_diff * 0.5

                except Exception as e:
                    print(f"Error evaluating syllables in line '{line}': {e}")
                    score -= 0.5

            # Recopilar rimas para evaluación posterior
            if j < len(expected_rhymes) and expected_rhymes[j]:
                rhyme_label = expected_rhymes[j]
                try:
                    rhyme_pattern = extract_rhyme_pattern(line, "asonante")
                    if rhyme_pattern:  # Solo agregar si hay patrón
                        if rhyme_label not in rhyme_groups:
                            rhyme_groups[rhyme_label] = []
                        rhyme_groups[rhyme_label].append(rhyme_pattern)
                except Exception as e:
                    print(f"Error processing rhyme in line '{line}': {e}")

        # Evaluar consistencia de rimas
        for rhyme_label, patterns in rhyme_groups.items():
            if len(patterns) <= 1:
                continue
            unique_patterns = set(p for p in patterns if p)
            if len(unique_patterns) == 1 and unique_patterns:
                score += len(patterns) * 1.0  # Bonus por rima consistente
            elif len(unique_patterns) > 1:
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
        top_p: float = 0.95,
        top_k: int = 50,
) -> List[str]:
    """Generar múltiples candidatos para un prompt con guardas de estabilidad."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        candidates = []
        logits_processors = LogitsProcessorList([NanInfGuard()])

        for _ in range(num_candidates):
            # Usar seed diferente para cada candidato
            torch.manual_seed(random.randint(0, 100000))

            # Parámetros robustos
            gen_kwargs = dict(
                **inputs,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-5),
                top_p=min(max(top_p, 1e-6), 0.9999),
                top_k=max(int(top_k), 0),
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                logits_processor=logits_processors,
                renormalize_logits=True,
                use_cache=True,
            )

            with torch.no_grad():
                try:
                    outputs = model.generate(**gen_kwargs)
                except RuntimeError as e:
                    # Fallback determinista (greedy) si falló el muestreo
                    print(f"[WARN] Sampling failed, falling back to greedy: {e}")
                    outputs = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )

            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            candidates.append(generated_text)

        return candidates

    except Exception as e:
        print(f"Error generating candidates: {e}")
        return []


def create_preference_dataset_with_model(
        dataset: Dataset,
        model,
        tokenizer,
        num_candidates: int = 4,
        sample_size: Optional[int] = None
) -> Dataset:
    """
    Crear dataset de preferencias usando modelo ya cargado.
    """
    print(f"Creating preference dataset with {len(dataset)} examples")

    # Limitar tamaño del dataset si se especifica
    if sample_size:
        dataset = dataset.shuffle(seed=42).select(range(min(sample_size, len(dataset))))
        print(f"Limited to {len(dataset)} examples")

    preference_data = []
    successful_examples = 0

    for idx, example in enumerate(dataset):
        print(f"Processing example {idx + 1}/{len(dataset)}")

        try:
            # Obtener el prompt del usuario
            conversations = example.get("conversations", {})
            if "content" not in conversations or not conversations["content"]:
                print(f"  Skipped: no content in conversations")
                continue

            user_prompt = conversations["content"][0]
            structure = example.get("structure", [])

            if not user_prompt.strip():
                print(f"  Skipped: empty user prompt")
                continue

            # Crear prompt con formato de chat
            messages = [{"role": "user", "content": user_prompt}]

            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"  Error applying chat template: {e}")
                # Fallback a formato simple
                prompt = f"User: {user_prompt}\nAssistant:"

            # Generar múltiples candidatos
            candidates = generate_candidates(
                prompt, model, tokenizer,
                num_candidates=num_candidates
            )

            if len(candidates) < 2:
                print(f"  Skipped: insufficient candidates generated")
                continue

            # Evaluar cada candidato
            scored_candidates = []
            for candidate in candidates:
                if candidate.strip():  # Solo evaluar candidatos no vacíos
                    score = score_poem(candidate, structure)
                    scored_candidates.append((score, candidate))

            if len(scored_candidates) < 2:
                print(f"  Skipped: insufficient valid candidates")
                continue

            # Ordenar por puntuación
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            # Seleccionar el mejor y el peor para máximo contraste
            best_score, best_poem = scored_candidates[0]
            worst_score, worst_poem = scored_candidates[-1]

            # Solo agregar si hay diferencia significativa
            score_diff = best_score - worst_score
            if score_diff > 1.0:
                preference_data.append({
                    "prompt": prompt,
                    "chosen": best_poem,
                    "rejected": worst_poem,
                    "score_diff": score_diff,
                    "structure": structure,
                    "best_score": best_score,
                    "worst_score": worst_score
                })

                successful_examples += 1
                print(f"  ✅ Score diff: {score_diff:.2f} (best: {best_score:.2f}, worst: {worst_score:.2f})")
            else:
                print(f"  ❌ Skipped: insufficient score difference ({score_diff:.2f})")

        except Exception as e:
            print(f"  ❌ Error processing example {idx}: {e}")
            continue

    print(f"\n📊 Created {len(preference_data)} preference pairs from {len(dataset)} examples")
    print(f"📊 Success rate: {successful_examples}/{len(dataset)} ({100 * successful_examples / len(dataset):.1f}%)")

    if not preference_data:
        raise ValueError("No preference pairs were created. Check your data and scoring function.")

    return Dataset.from_list(preference_data)


# ==========================================
# ENTRENAMIENTO DPO OPTIMIZADO
# ==========================================

def train_dpo_model(
        base_model_name: str,
        dataset_path: str,
        output_dir: str = "galicia-dpo-model",
        num_train_epochs: int = 2,
        per_device_train_batch_size: int = 2,
        learning_rate: float = 5e-7,
        beta: float = 0.1,
        max_samples: int = 1000,
):
    """
    Entrenar modelo con DPO de forma optimizada (una sola carga del modelo).
    """
    print("=" * 60)
    print("INICIANDO ENTRENAMIENTO DPO OPTIMIZADO")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Estabilidad matmul en GPU
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

    # 1. Cargar modelo UNA SOLA VEZ
    print("\n1. Cargando modelo base...")
    dtype = pick_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    # Ajustes de tokenizer y modelo
    ensure_tokens(tokenizer)
    if hasattr(model, "resize_token_embeddings"):
        # Por si se añadió un <pad> nuevo
        model.resize_token_embeddings(len(tokenizer))

    # Configurar chat template si es necesario
    if getattr(tokenizer, "chat_template", None) is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    model.eval()  # importante para estabilidad en generación

    # 2. Cargar dataset
    print("\n2. Cargando dataset...")
    try:
        raw_dataset = load_from_disk(dataset_path)
        print(f"  Dataset loaded: {len(raw_dataset)} examples")
    except Exception as e:
        raise ValueError(f"Error loading dataset from {dataset_path}: {e}")

    # 3. Crear dataset de preferencias usando el mismo modelo
    print("\n3. Creando dataset de preferencias...")
    pref_dataset = create_preference_dataset_with_model(
        raw_dataset,
        model,
        tokenizer,
        num_candidates=4,
        sample_size=max_samples
    )

    if len(pref_dataset) == 0:
        raise ValueError("No preference pairs created. Cannot proceed with training.")

    # 4. Dividir en train/eval
    print("\n4. Dividiendo dataset...")
    split = pref_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"  Train: {len(train_dataset)} ejemplos")
    print(f"  Eval: {len(eval_dataset)} ejemplos")

    # 5. Configurar argumentos de entrenamiento DPO
    print("\n5. Configurando entrenamiento...")
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        beta=beta,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="steps",           # deja igual que tu script original
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        bf16=bf16_ok,                    # solo si está soportado
        fp16=(torch.cuda.is_available() and not bf16_ok),
        report_to="none",
        remove_unused_columns=False,
        max_length=512,
        max_prompt_length=256,
        dataloader_pin_memory=False,     # Optimización de memoria
        dataloader_num_workers=0,        # Evitar problemas de multiprocessing
        save_total_limit=2,              # Limitar checkpoints guardados
    )

    # 6. Crear trainer DPO
    print("\n6. Creando DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Permite optimizaciones internas
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # 7. Entrenar
    print("\n7. Iniciando entrenamiento...")
    print("=" * 60)

    try:
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        raise

    # 8. Guardar modelo final
    print("\n8. Guardando modelo...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 9. Mostrar métricas finales
    print("\n9. Evaluación final...")
    try:
        metrics = trainer.evaluate()
        print(f"Métricas finales: {metrics}")
    except Exception as e:
        print(f"Error in final evaluation: {e}")

    print("\n" + "=" * 60)
    print(f"ENTRENAMIENTO COMPLETADO")
    print(f"Modelo guardado en: {output_dir}")
    print("=" * 60)

    return trainer


# ==========================================
# FUNCIÓN DE INFERENCIA MEJORADA
# ==========================================

def generate_poem_with_structure(
        model_path: str,
        prompt: str,
        structure: List[Dict],
        temperature: float = 0.7,
        max_new_tokens: int = 150
) -> Tuple[str, float]:
    """
    Generar un poema usando el modelo entrenado con DPO.
    Returns: (generated_poem, score)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Cargar modelo entrenado
        dtype = pick_dtype()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        ensure_tokens(tokenizer)
        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))
        model.eval()

        # Crear prompt con formato de chat
        messages = [{"role": "user", "content": prompt}]

        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            formatted_prompt = f"User: {prompt}\nAssistant:"

        # Generar
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        logits_processors = LogitsProcessorList([NanInfGuard()])

        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=max(temperature, 1e-5),
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    logits_processor=logits_processors,
                    renormalize_logits=True,
                    use_cache=True,
                )
            except RuntimeError as e:
                print(f"[WARN] Sampling failed in inference, falling back to greedy: {e}")
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
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

        return generated_text, score

    except Exception as e:
        print(f"Error generating poem: {e}")
        return "", -999.0


# ==========================================
# MAIN - EJEMPLO DE USO
# ==========================================

if __name__ == "__main__":
    # Configuración
    BASE_MODEL = "galicIA-full-FIM"
    DATASET_PATH = "poemas_GalicIA_est"
    OUTPUT_DIR = "galicia-dpo-structured"

    try:
        # Entrenar modelo con DPO
        trainer = train_dpo_model(
            base_model_name=BASE_MODEL,
            dataset_path=DATASET_PATH,
            output_dir=OUTPUT_DIR,
            num_train_epochs=2,
            per_device_train_batch_size=2,
            learning_rate=5e-7,
            beta=0.1,
            max_samples=400,  # Reducido para pruebas
        )

        print("✅ Entrenamiento completado exitosamente!")

    except Exception as e:
        print(f"❌ Error en el entrenamiento: {e}")
        raise
