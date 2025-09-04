
from datetime import datetime
import configparser

from src.finetuning.inferencia.default_generation import stream_chat_to_stdout


def generar_poema(promt: str) -> str:
    """Implementación local de get_docs usando RAG."""
    messages = [
        {"role": "user", "content": f"{promt}"}
    ]
    result = stream_chat_to_stdout(messages, temperature=0.3)
    print(result)
    return result



