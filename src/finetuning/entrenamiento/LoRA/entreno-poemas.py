from datasets import load_from_disk
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# 1) Cargar y preprocesar dataset
raw_ds = load_from_disk("bases_movimientos/poemas_GalicIA_trovadorismo")
print(raw_ds)


# 2) Cargar modelo y tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="galicIA-v1",
    max_seq_length=512,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
)

#tokenizer = get_chat_template(tokenizer, chat_template="chatml")

# 3) Convertir a texto con prompt de generación
def formatting_prompts_func(examples):
    return {
        "text": [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            for convo in examples["conversations"]
        ]
    }

ds_formatted = raw_ds.map(formatting_prompts_func,batched=True,)

# 4) Envolver en HF dataset
#hf_dataset = HFDataset.from_dict({"text": ds_formatted["text"]})
print(ds_formatted)
print(ds_formatted[0])

# 5) Ajustar PEFT/LoRA
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

# 8) Guardar modelo
model.save_pretrained("galicIA-v1-trovadorismo")
tokenizer.save_pretrained("galicIA-v1-trovadorismo")
