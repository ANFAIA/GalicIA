from datasets import load_from_disk
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

def inspeccionar_adapters(model):
    # ¿Es un modelo PEFT?
    if not hasattr(model, "peft_config"):
        print("➡️ El modelo no tiene adapters (no es PeftModel).")
        return []

    # Nombres de adapters cargados
    adapters = list(model.peft_config.keys())
    print(f"➡️ Adapters encontrados: {adapters}")

    # Adapter activo (si aplica)
    active = getattr(model, "active_adapter", None)
    print(f"➡️ Adapter activo: {active}")

    # Info básica de cada adapter (tipo y si está en modo inferencia = congelado)
    for name, cfg in model.peft_config.items():
        peft_type = getattr(cfg, "peft_type", type(cfg).__name__)
        frozen = getattr(cfg, "inference_mode", None)  # True => congelado, False => entrenable
        print(f"   - {name}: peft_type={peft_type}, inference_mode={frozen}")

    # Devolver la lista por si quieres usarla en merge_and_unload(...)
    return adapters

# 1) Cargar y preprocesar dataset
raw_ds = load_from_disk("poemas_GalicIA_norima")
print(raw_ds)

# 2) Cargar modelo y tokenizer (base)
model_og, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-0.6B",
    max_seq_length=512,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
)

# 3) Convertir a texto con prompt de generación
def formatting_prompts_func(examples):
    return {
        "text": [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for convo in examples["conversations"]
        ]
    }

ds_formatted = raw_ds.map(formatting_prompts_func, batched=True)
print(ds_formatted)
print(ds_formatted[0])

# 5) LoRA: cargar "idioma" congelado + fusionarlo temporalmente + añadir "tarea" entrenable
from peft import PeftModel, LoraConfig, get_peft_model

# 5a) Cargar LoRA de idioma (congelado)
model = PeftModel.from_pretrained(
    model_og,
    "galicIA-base",         # <- tu LoRA de idioma
    adapter_name="lang",
    is_trainable=False,     # <- congelado
)

# 5b) Activar y fusionar 'lang' en el base (reversible)
#model.set_adapter("lang")
model = model.merge_and_unload()

# 5c) Añadir LoRA de tarea (entrenable) y activarlo
task_cfg = LoraConfig(
    r=250,
    lora_alpha=500,
    lora_dropout=0,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
    ],
    task_type="CAUSAL_LM",
)
#model.add_adapter("task", task_cfg)
#model.set_adapter("task")
model = get_peft_model(model, task_cfg)
inspeccionar_adapters(model)

# 6) Preparación de entreno (GC, grads en inputs)
model.config.use_cache = False
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

# 7) Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds_formatted,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=8e-5,
        weight_decay=0.01,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        warmup_steps=28,
        logging_steps=30,
        save_steps=140,
    ),
)

stats = trainer.train()
print(stats)

# Preparar para inferencia con Unsloth (manteniendo adapters)
model = FastLanguageModel.for_inference(model)

full = model.merge_and_unload()

# Guarda
full.save_pretrained("galicIA-full-noFIM")
tokenizer.save_pretrained("galicIA-full-noFIM")
