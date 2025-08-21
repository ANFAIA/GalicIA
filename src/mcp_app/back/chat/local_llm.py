import asyncio
from qwen_agent.agents import Assistant
import configparser
import threading

from src.mcp_app.back.chat.historial import get_history, save_message

config = configparser.ConfigParser()
config.read('config.ini')


def recortar_despues_de_think(texto):
    marcador = "</think>"
    if marcador in texto:
        return texto.split(marcador, 1)[1]
    return texto

def init_agent_service():
    config.read('config.ini')
    model = config['model']['llm_tool']
    url = config['model']['url']
    key = config['model']['key']
    llm_cfg = {
        'model': model,
        'model_server': url,
        # 'api_key': 'key',
        'max_tokens': 32768,
    }

    system = ("""Eres un LLM que enruta a unha funcion que crea poemas en galego, o unico que debes facer é pasar o poema ao usuario final, non escribas nada mais en o poema devolto""")
    tools = [{
        "mcpServers": {
            "rag_llm": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
    }]
    bot = Assistant(
        llm=llm_cfg,
        name='GalicIA',
        description='chatbot en galego para poesía',
        system_message=system,
        function_list=tools,
    )

    return bot

def run(query_text: str, conversation_id: str) -> str:
    # 1) Inicializa el agente
    bot = init_agent_service()

    # 1️⃣ Recupera y recorta el historial
    history = get_history(conversation_id, max_tokens=4000)

    # 2️⃣ Añade el turno actual del usuario
    history.append({'role': 'user', 'content': query_text})
    save_message(conversation_id, "user", query_text)

    # 4) Ejecuta en streaming, acumulando la respuesta
    response_chunks = []
    response_final=None
    for chunk in bot.run(
        messages=history,
        thread_id=conversation_id,
    ):
        # Cada chunk puede ser una lista de dicts o un str directo
        if isinstance(chunk, list):
            # Extrae 'content' de cada dict y concatena
            for msg in chunk:
                if isinstance(msg, dict) and 'content' in msg:
                    response_chunks.append(msg['content'])
                    response_final=msg['content']
                else:
                    response_chunks.append(str(msg))
                    response_final=str(msg)
        elif isinstance(chunk, dict):
            # Caso en que directamente recibes un dict
            response_chunks.append(chunk.get('content', str(chunk)))
            response_final=chunk.get('content', str(chunk))
        else:
            # Por si alguna vez devuelve un str directamente
            response_chunks.append(str(chunk))
            response_final=str(chunk)
    response_final=recortar_despues_de_think(response_final)
    # 5) Une y devuelve la respuesta completa
    save_message(conversation_id, "assistant", response_final)
    return response_final


# Crear un nuevo bucle de eventos
loop = asyncio.new_event_loop()

# Función que inicia el bucle de eventos en un hilo separado
def start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

# Iniciar el hilo que ejecutará el bucle de eventos
t = threading.Thread(target=start_loop, args=(loop,))
t.start()

# Función para ejecutar corutinas en el bucle de eventos
def run_coroutine(coroutine):
    return asyncio.run_coroutine_threadsafe(coroutine, loop).result()

# Función query_llm que utiliza el bucle de eventos en el hilo separado
def query_llm(query_text: str, conversation_id: str) -> str:
    #return run_coroutine(run(query_text))#, conversation_id))
    return run(query_text,conversation_id)
