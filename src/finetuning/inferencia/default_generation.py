from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading

# ------------------------------
# Carga de modelo y tokenizer
# ------------------------------
def load_model_and_tokenizer(checkpoint: str):
    """
    Carga y devuelve (tokenizer, model).
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    return tokenizer, model
checkpoint = "galicIA-full-FIM"
tokenizer, model = load_model_and_tokenizer(checkpoint)
# ------------------------------
# Helper: preparar los inputs
# ------------------------------
def _prepare_inputs(tokenizer, messages):
    """
    Aplica el template de chat tal y como lo hacías.
    Devuelve los input_ids listos para generate().
    """
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,   # lo mantengo igual que en tu código
    )
    return inputs

# ------------------------------
# 1) Streaming por stdout (no devuelve nada)
# ------------------------------
def stream_chat_to_stdout(messages,
                          max_new_tokens=150,
                          repetition_penalty=1.6,
                          **generate_kwargs):
    """
    Genera texto en tiempo real e imprime por pantalla (stdout).
    No retorna nada.
    """
    inputs = _prepare_inputs(tokenizer, messages)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    def _worker():
        model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
            **generate_kwargs
        )

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    for token in streamer:
        print(token, end="", flush=True)

    thread.join()
    # Opcional: salto de línea final
    print()

# ------------------------------
# 2) Generación que devuelve el texto completo al final
# ------------------------------
def generate_chat_text(messages,
                       max_new_tokens=150,
                       repetition_penalty=1.6,
                       **generate_kwargs) -> str:
    """
    Genera el texto completo y lo devuelve como string (sin imprimir).
    """
    inputs = _prepare_inputs(tokenizer, messages)

    output_ids = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        **generate_kwargs
    )

    # Decodificar sólo lo nuevo (sin el prompt)
    generated_ids = output_ids[0, inputs.shape[-1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text

# ------------------------------
# Ejemplo de uso
# ------------------------------
if __name__ == "__main__":
    # checkpoint = "Qwen/Qwen3-0.6B"
    # checkpoint = "galicIA-base"




    messages = [
        {"role": "user", "content": "Faime un poema sobre a guerra"}
    ]

    print("=== STREAMING (stdout) ===")
    stream_chat_to_stdout(messages, max_new_tokens=150, repetition_penalty=1.6)

    #print("\n=== TEXTO COMPLETO (return) ===")
    #full_text = generate_chat_text(messages, max_new_tokens=150, repetition_penalty=1.6)
    #print(full_text)
