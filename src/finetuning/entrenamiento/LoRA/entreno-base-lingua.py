from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# 1) Cargar y concatenar los 3 datasets de ProxectoNós
ds1 = load_dataset("proxectonos/belebele_gl", split="train")    # Reading comprehension :contentReference[oaicite:3]{index=3}
ds2 = load_dataset("proxectonos/galcola", split="train")        # Linguistic acceptability :contentReference[oaicite:4]{index=4}
ds3 = load_dataset("proxectonos/mgsm_gl", split="train")        # Math QA :contentReference[oaicite:5]{index=5}

raw_ds = concatenate_datasets([ds1, ds2, ds3]).shuffle(seed=42)

# 2) Cargar modelo y tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-0.6B",
    max_seq_length=512,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
)
#tokenizer = get_chat_template(tokenizer, chat_template="unsloth")

# 3) Función de formateo genérica
def formatting_prompts_func(examples):
    texts = []
    # número de ejemplos en este batch
    n = len(examples[next(iter(examples))])

    for i in range(n):
        # 1) Detección de tipo de dataset

        # --- Reading comprehension (ds1) ---
        if (
                "flores_passage" in examples
                and all(f"mc_answer{j}" in examples for j in range(1, 5))
                and "correct_answer_num" in examples
        ):
            passage = examples["flores_passage"][i]
            q = examples["question"][i]
            # recoge las 4 opciones
            opciones = [examples[f"mc_answer{j}"][i] for j in range(1, 5)]
            # índice (cadena) a entero para elegir la respuesta correcta
            idx_corr = int(examples["correct_answer_num"][i]) - 1

            prompt = (
                "### Instrucción:\n"
                "Lee el siguiente pasaje y responde la pregunta eligiendo la opción correcta:\n\n"
                f"{passage}\n\n"
                f"Pregunta: {q}\n\n"
                "Opciones:\n"
                f"1) {opciones[0]}\n"
                f"2) {opciones[1]}\n"
                f"3) {opciones[2]}\n"
                f"4) {opciones[3]}\n\n"
                "### Respuesta:\n"
                f"{opciones[idx_corr]}"
            )

        # --- Linguistic acceptability (ds2) ---
        elif "sentence" in examples and "label" in examples:
            s = examples["sentence"][i]
            lbl = "Aceptable" if examples["label"][i] == 1 else "No aceptable"

            prompt = (
                "### Instrucción:\n"
                "¿Es esta oración gramaticalmente aceptable en gallego?\n"
                f"\"{s}\"\n\n"
                "### Respuesta:\n"
                f"{lbl}"
            )

        # --- Math QA (ds3) ---
        elif "question" in examples and "answer" in examples:
            q = examples["question"][i]
            a = examples["answer"][i]

            prompt = (
                "### Instrucción:\n"
                "Responde a la siguiente pregunta:\n"
                f"{q}\n\n"
                "### Respuesta:\n"
                f"{a}"
            )

        # --- Fallback genérico ---
        else:
            prompt = str({k: examples[k][i] for k in examples})

        # 2) Envuelve en lista de mensajes
        convo = [{"role": "user", "content": prompt}]

        # 3) Aplica plantilla de chat
        texts.append(
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        )

    return {"text": texts}

# 4) Mapear y generar campo "text"
ds_formatted = raw_ds.map(formatting_prompts_func, batched=True, remove_columns=raw_ds.column_names)

# 5) Envolver en PEFT/LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=700,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
    ],
    lora_alpha=1400,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# 6) Preparar para inferencia
model = FastLanguageModel.for_inference(model)

# 7) Configurar y lanzar trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds_formatted,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        # batch efectivo = 2 × 8 = 16
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,

        # o bien en lugar de max_steps, si tu versión de TRL lo soporta:
        # num_train_epochs=3,

        # total_steps = 804 pasos/época × 3 épocas
        max_steps=2412,

        learning_rate=8e-5,
        weight_decay=0.01,
        optim="adamw_8bit",
        lr_scheduler_type="linear",

        # warm‑up ≈ 10% de 2412
        warmup_steps=241,

        # cada ~100 pasos
        logging_steps=100,
        # un checkpoint al final de cada época
        save_steps=804,
    ),
)


stats = trainer.train()
print(stats)

# 8) Guardar modelo
model.save_pretrained("galicIA-basev1.1")
tokenizer.save_pretrained("galicIA-basev1.1")
