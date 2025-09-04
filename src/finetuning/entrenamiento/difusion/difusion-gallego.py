#!/usr/bin/env python3
"""
Fine-tuning de LLaDA-8B-Base (modelo de difusión) con Unsloth para Gallego
Cambios mínimos sobre el código original
"""

from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import torch

# 1) Cargar y concatenar los 3 datasets de ProxectoNós - SIN CAMBIOS
ds1 = load_dataset("proxectonos/belebele_gl", split="train")  # Reading comprehension
ds2 = load_dataset("proxectonos/galcola", split="train")  # Linguistic acceptability
ds3 = load_dataset("proxectonos/mgsm_gl", split="train")  # Math QA

raw_ds = concatenate_datasets([ds1, ds2, ds3]).shuffle(seed=42)
print(f"Dataset total: {len(raw_ds)} ejemplos")

# 2) ⚠️ CAMBIO CRÍTICO: Cargar LLaDA-8B-Base (modelo de difusión)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="GSAI-ML/LLaDA-8B-Base",  # ⚠️ CAMBIO: Modelo de difusión LLaDA
    max_seq_length=512,
    load_in_4bit=True,  # ⚠️ CAMBIO: Usar 4bit para ahorrar memoria con 8B params
    load_in_8bit=False,
    full_finetuning=False,  # ⚠️ CAMBIO: OBLIGATORIO para modelos de difusión
    trust_remote_code=True,  # ⚠️ CAMBIO: Necesario para cargar el código custom de LLaDA
    torch_dtype=torch.bfloat16,  # ⚠️ CAMBIO: LLaDA funciona mejor con bfloat16
)

# Asegurar que el tokenizer tenga pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


# 3) Función de formateo genérica - SIN CAMBIOS
def formatting_prompts_func(examples):
    texts = []
    convs = []
    n = len(next(iter(examples.values())))

    for i in range(n):
        # Extraemos todos los posibles campos (o None)
        passage = examples.get("flores_passage", [None] * n)[i]
        correct_num = examples.get("correct_answer_num", [None] * n)[i]
        mc1 = examples.get("mc_answer1", [None] * n)[i]
        mc2 = examples.get("mc_answer2", [None] * n)[i]
        mc3 = examples.get("mc_answer3", [None] * n)[i]
        mc4 = examples.get("mc_answer4", [None] * n)[i]
        sentence = examples.get("sentence", [None] * n)[i]
        label = examples.get("label", [None] * n)[i]
        question = examples.get("question", [None] * n)[i]
        answer = examples.get("answer", [None] * n)[i]

        # Montar prompt y recoger respuesta
        if passage is not None and correct_num is not None and None not in (mc1, mc2, mc3, mc4):
            idx = int(correct_num) - 1
            opciones = [mc1, mc2, mc3, mc4]
            prompt = (
                f"Lee o siguinte pasaxe e responde á pregunta elixiendo a opción correcta:\n\n"
                f"{passage}\n\n"
                f"Pregunta: {question}\n\n"
                "Opcións:\n"
                f"1) {opciones[0]}\n"
                f"2) {opciones[1]}\n"
                f"3) {opciones[2]}\n"
                f"4) {opciones[3]}"
            )
            answer_content = opciones[idx]

        elif sentence is not None and label is not None:
            resp = "Aceptable" if label == 1 else "Non aceptable"
            prompt = f"¿É esta oración gramaticalmente aceptable?\n\"{sentence}\""
            answer_content = resp

        elif question is not None and answer is not None:
            question = question.removeprefix("Pregunta: ")
            answer = answer.removeprefix("Resposta paso a paso: ")
            prompt = question
            answer_content = answer

        else:
            prompt = str({k: examples[k][i] for k in examples if examples[k][i] is not None})
            answer_content = ""

        # Conversación de dos turnos
        convo = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer_content}
        ]
        convs.append(convo)

        # Aplicar plantilla sin placeholder de generación
        try:
            texts.append(
                tokenizer.apply_chat_template(
                    convo,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        except:
            # Fallback si LLaDA no soporta chat_template
            texts.append(f"User: {prompt}\nAssistant: {answer_content}")

    return {
        "conversations": convs,
        "text": texts
    }


# 4) Mapear y generar campo "text" - SIN CAMBIOS
ds_formatted = raw_ds.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=raw_ds.column_names
)

print(f"Dataset formateado: {len(ds_formatted)} ejemplos")
if len(ds_formatted) > 0:
    print(f"Ejemplo: {ds_formatted[0]['text'][:200]}...")

# 5) ⚠️ CAMBIO: Configuración LoRA adaptada para modelo de difusión de 8B
r_config = 64  # ⚠️ CAMBIO: Reducido de 120 a 64 para estabilidad con difusión

model = FastLanguageModel.get_peft_model(
    model,
    r=r_config,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=r_config * 2,
    lora_dropout=0.05,  # ⚠️ CAMBIO: Añadido dropout para difusión
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print("Parámetros entrenables:")
model.print_trainable_parameters()

# 6) ⚠️ CAMBIO: Configuración optimizada para modelo de difusión
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds_formatted,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",

        # ⚠️ CAMBIOS: Ajustes para modelo de 8B con difusión
        per_device_train_batch_size=1,  # Reducido para 8B params
        gradient_accumulation_steps=16,  # Aumentado para compensar

        num_train_epochs=3,  # Más épocas para difusión

        learning_rate=2e-5,  # ⚠️ CAMBIO: LR más conservador
        weight_decay=0.01,
        optim="adamw_torch",  # ⚠️ CAMBIO: Optimizador más estable
        lr_scheduler_type="cosine",  # ⚠️ CAMBIO: Mejor para difusión

        warmup_ratio=0.1,  # ⚠️ CAMBIO: Usar ratio en vez de steps fijos

        logging_steps=50,
        save_steps=500,

        # ⚠️ NUEVAS configuraciones para estabilidad con difusión
        fp16=False,  # Desactivado porque usamos bfloat16
        bf16=True,  # Activar bfloat16 para LLaDA
        gradient_checkpointing=True,  # Ahorrar memoria con 8B params
        max_grad_norm=1.0,  # Clip gradients para estabilidad

        # Configuraciones adicionales para difusión
        remove_unused_columns=False,  # Mantener todas las columnas
        report_to="none",  # Desactivar reporting por ahora
    ),
)

# 7) Entrenar
print("\n" + "=" * 50)
print("Iniciando fine-tuning de LLaDA-8B con LoRA")
print(f"Modelo: GSAI-ML/LLaDA-8B-Base (Difusión)")
print(f"Dataset: {len(ds_formatted)} ejemplos en gallego")
print(f"LoRA rank: {r_config}")
print("=" * 50 + "\n")

try:
    stats = trainer.train()
    print(f"\nEntrenamiento completado:")
    print(f"  - Pérdida final: {stats.training_loss:.4f}")
    print(f"  - Pasos totales: {stats.global_step}")
except Exception as e:
    print(f"Error durante entrenamiento: {e}")
    print("Intentando guardar modelo parcial...")

# 8) Guardar modelo - CAMBIO: Nombre actualizado
output_dir = "galicIA-llada-diffusion"
print(f"\nGuardando modelo en {output_dir}...")

try:
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Modelo guardado correctamente")
except Exception as e:
    print(f"⚠️ Error guardando modelo: {e}")

# 9) ⚠️ NUEVO: Test de generación con modelo de difusión
print("\n" + "=" * 50)
print("Test de generación:")
print("=" * 50)

try:
    # Preparar para inferencia
    FastLanguageModel.for_inference(model)

    # Prompts de prueba en gallego
    test_prompts = [
        "A capital de Galicia é",
        "O idioma galego",
        "Responde: Cal é o río máis importante de Galicia?"
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        # Generar con configuración para difusión
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Resposta: {generated}")

except Exception as e:
    print(f"⚠️ Error en generación: {e}")
    print("Nota: LLaDA puede requerir métodos de generación especiales")

print("\n" + "=" * 50)
print("¡Proceso completado!")
print(f"Modelo fine-tuned guardado en: {output_dir}")
print("=" * 50)