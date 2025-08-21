
from datetime import datetime
import configparser

from src.finetuning.inferencia.default_generation import generate_chat_text


def generar_poema(promt: str) -> str:
    """Implementación local de get_docs usando RAG."""
    return generate_chat_text(promt, max_new_tokens=150, repetition_penalty=1.6)



