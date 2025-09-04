from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, \
    StoppingCriteriaList
import torch
import threading
import re

# === Carga del modelo ===
ckpt = "pajon1/galicIA-v1"
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    ckpt, torch_dtype="auto", device_map="auto", trust_remote_code=True
)

# Asegurar que el tokenizer tiene pad_token
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def _eos_ids():
    ids = []
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)

    # Tokens especiales comunes
    special_tokens = ["<|im_end|>", "</s>", "<|endoftext|>"]
    for token in special_tokens:
        try:
            token_id = tok.convert_tokens_to_ids(token)
            if token_id is not None and token_id != tok.unk_token_id and token_id not in ids:
                ids.append(token_id)
        except:
            pass

    return ids if ids else [tok.eos_token_id]


def _prepare_inputs(messages, force_continue_after_think=True):
    # Aplicar chat template normal
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking=False)

    # SOLUCIÓN: Si queremos forzar al modelo a continuar después de <think>
    if force_continue_after_think:
        # Opción 1: Agregar contenido después del prompt para "empujar" la generación
        prompt = prompt + "\n<think>\n"  # Iniciamos el tag think nosotros
        print(f"[DEBUG] Prompt modificado: {prompt[:300]}...")
    else:
        print(f"[DEBUG] Prompt original: {prompt[:300]}...")

    # Tokenizar
    inputs = tok(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Mover a dispositivo
    return {k: v.to(model.device) for k, v in inputs.items()}


# Stopping criteria personalizado que NO para en </think>
class SelectiveStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, keywords_to_stop):
        self.tokenizer = tokenizer
        self.keywords = keywords_to_stop
        self.key_ids = []
        for k in keywords_to_stop:
            tokens = tokenizer(k, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            if len(tokens) > 0:
                self.key_ids.append(tokens)

    def __call__(self, input_ids, scores, **kwargs):
        # Solo parar en keywords específicos, NO en </think>
        for kid in self.key_ids:
            if len(kid) <= input_ids.shape[1]:
                if torch.equal(input_ids[0, -len(kid):].cpu(), kid):
                    return True
        return False


def clean_output(text):
    """Limpia el output removiendo tags de think si es necesario"""
    # Remover tags <think> y </think> vacíos
    text = re.sub(r'<think>\s*</think>', '', text)
    # Si hay contenido dentro de think, extraerlo
    text = re.sub(r'<think>(.*?)</think>', r'\1', text, flags=re.DOTALL)
    return text.strip()


def stream_chat_to_stdout(messages,
                          max_new_tokens=500,  # Aumentado para dar más espacio
                          temperature=0.7,
                          top_p=0.9,
                          repetition_penalty=1.1,
                          no_repeat_ngram_size=3,
                          force_continue=True):
    print("\n=== Iniciando generación ===")
    inputs = _prepare_inputs(messages, force_continue_after_think=force_continue)
    eos_ids = _eos_ids()
    print(f"[DEBUG] EOS IDs: {eos_ids}")
    print(f"[DEBUG] Input shape: {inputs['input_ids'].shape}")

    # Solo parar en errores reales, no en </think>
    stop_keywords = ["Traceback", "Error:", "RuntimeError", "ValueError"]
    stops = StoppingCriteriaList([SelectiveStoppingCriteria(tok, stop_keywords)])

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)

    generated_text = []

    def _worker():
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=50,  # Forzar generación mínima
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=50,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_token_id=eos_ids,
                pad_token_id=tok.pad_token_id,
                stopping_criteria=stops,
                streamer=streamer,
                length_penalty=1.0,  # No penalizar longitud
            )

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    print("\nRespuesta del modelo (raw):\n")
    for token in streamer:
        print(token, end="", flush=True)
        generated_text.append(token)

    t.join()

    # Limpiar y mostrar output procesado
    full_text = "".join(generated_text)
    cleaned_text = clean_output(full_text)

    print("\n\n=== Respuesta limpia ===")
    print(cleaned_text)
    print("\n=== Generación completada ===\n")
    return cleaned_text


# === SOLUCIÓN ALTERNATIVA: Bypass directo ===
def generate_direct(prompt_text, max_new_tokens=300):
    """Generación directa sin chat template"""
    print(f"\n=== Generación directa ===")
    print(f"Prompt: {prompt_text}")

    inputs = tok(prompt_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    response = tok.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    cleaned = clean_output(response)
    print(f"\nRespuesta: {cleaned}\n")
    return cleaned


# === PRUEBAS ===
if __name__ == "__main__":
    promt="¿?"
    print("=" * 60)
    print("PRUEBA 1: Con chat template modificado")
    print("=" * 60)
    messages = [
        {"role": "user", "content": f"{promt}"}
    ]
    stream_chat_to_stdout(messages, force_continue=True)

    print("\n" + "=" * 60)
    print("PRUEBA 2: Generación directa sin template")
    print("=" * 60)

    # Probar diferentes formatos de prompt directo
    prompts = [
        # Formato 1: Simple
        f"Usuario: {promt}\nAsistente:",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Formato {i} ---")
        generate_direct(prompt, max_new_tokens=200)

    print("\n" + "=" * 60)
    print("PRUEBA 3: Pregunta simple")
    print("=" * 60)
    messages = [
        {"role": "user", "content": f"{promt}"}
    ]
    stream_chat_to_stdout(messages, temperature=0.3)