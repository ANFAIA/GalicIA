from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading

# Carga de checkpoint y tokenizer/model
#checkpoint = "Qwen/Qwen3-3B"
checkpoint="pajon1/galicIA-v1"
#checkpoint="galicIA-full-FIM"
tokenizer = AutoTokenizer.from_pretrained(checkpoint,)
model     = AutoModelForCausalLM.from_pretrained(checkpoint)

# Mensaje de entrada
messages = [
    {"role": "user", "content": "Dame un poema que exprese a dor e a nostalxia dun emigrante que escoita ao lonxe as campaniñas da súa aldea"}
]

# Prepara los tensores con el template de chat
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=False
)
print(inputs)

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
        max_new_tokens=150,
        repetition_penalty=1.6,
        streamer=streamer,
        #stop = ["\n\n"]
    )

# Arranca el hilo de generación
thread = threading.Thread(target=generate_stream)
thread.start()

# Consume el streamer e imprime en tiempo real
for token in streamer:
    print(token, end="", flush=True)

# Espera a que termine el hilo (opcional)
thread.join()