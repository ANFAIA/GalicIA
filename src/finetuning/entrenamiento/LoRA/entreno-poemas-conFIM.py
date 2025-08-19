from datasets import load_from_disk, concatenate_datasets
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import random

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
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-0.6B",
    max_seq_length=512,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
)

# === FIM: añadir tokens si faltan y ajustar embeddings ===
FIM_TOKENS = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"]
missing = [t for t in FIM_TOKENS if t not in tokenizer.get_vocab()]
if missing:
    tokenizer.add_special_tokens({"additional_special_tokens": FIM_TOKENS})
    model.resize_token_embeddings(len(tokenizer))

def _first_assistant_text(conversations):
    for turn in conversations:
        if turn.get("role") == "assistant":
            return turn.get("content", "")
    return ""

# 3) Convertir a texto con prompt de generación (mezcla normal + FIM con cambios mínimos)
def formatting_prompts_func(examples, fim_ratio: float = 0.3):
    out = []
    if fim_ratio>1:
        prev = formatting_prompts_func(examples, fim_ratio - 1)
        out.extend(prev["text"])  # <--- aplanar en lugar de append
        fim_ratio = 1.0  # última pasada completa (prob=1.0)
    for conv in examples["conversations"]:
        make_fim = random.random() < fim_ratio
        if make_fim:
            asst = _first_assistant_text(conv)
            if asst:
                # Cortes en límites naturales (por líneas no vacías)
                lines = [l for l in asst.splitlines() if l.strip()]
                if len(lines) >= 6:
                    i = max(1, len(lines) // 3)
                    j = max(i + 1, (2 * len(lines)) // 3)
                    prefix = "\n".join(lines[:i])
                    middle = "\n".join(lines[i:j])
                    suffix = "\n".join(lines[j:])
                    # Formato FIM estándar
                    out.append(f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}")
                    continue
        # Caso normal (chat template) si no se usa FIM o no es viable
        out.append(
            tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    return {"text": out}

# === CAMBIO MÍNIMO: duplicar datos concatenando una pasada chat + una pasada FIM ===
ds_chat = raw_ds.map(
    formatting_prompts_func,
    batched=True,
    fn_kwargs={"fim_ratio": 0.0},
)
ds_fim = raw_ds.map(
    formatting_prompts_func,
    batched=True,
    fn_kwargs={"fim_ratio": 1.0},
)
print(ds_fim[12])
ds_formatted = concatenate_datasets([ds_chat, ds_fim])

print(ds_formatted)
#print(ds_formatted[0])

# 5) LoRA: cargar "idioma" congelado + fusionarlo temporalmente + añadir "tarea" entrenable
from peft import PeftModel, LoraConfig, get_peft_model

# 5a) Cargar LoRA de idioma (congelado)
model = PeftModel.from_pretrained(
    model,
    "galicIA-base",
    adapter_name="lang",
    is_trainable=False,   # lang congelado
)
model = model.merge_and_unload()

# 5c) Añadir LoRA de tarea NUEVO (entrenable) sobre el base ya fusionado
task_cfg = LoraConfig(
    r=250,
    lora_alpha=500,
    lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
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
        learning_rate=8e-5,
        weight_decay=0.01,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        warmup_steps=28,
        num_train_epochs=2,
        logging_steps=30,
        save_steps=140,
    ),
)

stats = trainer.train()
print(stats)


# Preparar para inferencia con Unsloth (manteniendo adapters)
model = FastLanguageModel.for_inference(model)

full = model.merge_and_unload()

# (Opcional) si quieres el parche de Unsloth para inferencia rápida:
# full = FastLanguageModel.for_inference(full)

# Guarda
full.save_pretrained("galicIA-full-FIM")
tokenizer.save_pretrained("galicIA-full-FIM")
