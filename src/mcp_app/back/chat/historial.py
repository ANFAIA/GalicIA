from collections import defaultdict
from transformers import AutoTokenizer
import configparser
import os

config = configparser.ConfigParser()
config.read('config.ini')
base_dir = config["paths"]["base_dir"]
llm_model = config["model"]["llm_token"]
llm_subdir = llm_model.replace("/", "_")
llm_dir = os.path.join(base_dir, "llm", llm_subdir)

_tokenizer = AutoTokenizer.from_pretrained(llm_dir, trust_remote_code=True, use_fast=True)
def count_tokens(
    text: str,
    *,
    max_tokens: int | None = None   # opcional, por si aún quisieras limitar
) -> int:
    """
    Devuelve el número de tokens que produce el tokenizer de Qwen-3
    para `text`.  Si `max_tokens` se indica, el conteo se trunca a
    ese valor (útil para comprobar longitudes máximas).
    """
    # Codificamos sin añadir tokens especiales
    token_ids = _tokenizer.encode(text, add_special_tokens=False)

    # Si se requiere un tope de tokens para el conteo
    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]

    return len(token_ids)

memory: dict[str, list[dict]] = defaultdict(list)

def save_message(conv_id: str, role: str, content: str):
    memory[conv_id].append(
        {"role": role, "content": content}
    )

def get_history(conv_id: str, max_tokens=2000) -> list[dict]:
    """Devuelve los mensajes más recientes sin pasarse del límite."""
    hist = memory[conv_id]
    #print(hist,flush=True)
    # recortamos por el final (los más antiguos primero)
    total = 0
    trimmed = []
    for msg in reversed(hist):               # empieza por el último
        total += count_tokens(msg["content"])  # tu propia función o Tiktoken
        if total > max_tokens:
            break
        trimmed.append(msg)
    return list(reversed(trimmed))