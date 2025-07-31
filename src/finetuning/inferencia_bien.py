from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading

# Carga de checkpoint y tokenizer/model
checkpoint = "galicIA-v1"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
model     = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True)

# Mensaje de entrada
messages = [
    {"role": "user", "content": "Xera un poema sobre voar nun avión"}
]

# Prepara los tensores con el template de chat
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=False
)

# Crea el streamer
streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

# Función target para generate, lanzada en un hilo
def generate_stream():
    model.generate(
        input_ids=inputs,
        max_new_tokens=400,
        repetition_penalty=1.6,
        streamer=streamer
    )

# Arranca el hilo de generación
thread = threading.Thread(target=generate_stream)
thread.start()

# Consume el streamer e imprime en tiempo real
for token in streamer:
    print(token, end="", flush=True)

# Espera a que termine el hilo (opcional)
thread.join()
