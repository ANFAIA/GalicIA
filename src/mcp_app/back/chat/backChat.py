# backChat.py
import configparser
import json
from openai import OpenAI
from openai.types.chat.completion_create_params import Function

from src.mcp_app.back.chat.historial import get_history, save_message
from src.mcp_app.services.tools.tools import generar_poema as generar_poema_service

# Carga configuración
config = configparser.ConfigParser()
config.read('config.ini')

# Definición de funciones usando objetos Function
FUNCTIONS = [
    Function(
        name="generar_poema",
        description="Dada unha orden de tipo:'Faime un poema sobre X' devole o poema",
        parameters={
            "type": "object",
            "properties": {
                "promt": {
                    "type": "string",
                    "description": "Orden de tipo :'Faime un poema sobre X'"
                }
            },
            "required": ["promt"]
        }
    ),
]

def generar_poema(promt: str) -> str:
    """
    Dado un promt genera un poema
    """
    return generar_poema_service(promt)



def recortar_despues_de_think(texto):
    marcador = "</think>"
    if marcador in texto:
        return texto.split(marcador, 1)[1]
    return texto

def query_llm(query_text: str, conversation_id: str) -> str:

    config.read('config.ini')
    model = config['model']['llm_tool']
    url = config['model']['url']
    key = config['model']['key']
    # Cliente OpenAI
    client = OpenAI(base_url=url
    ,api_key=key)

    # 1) Prompt system (solo una vez)
    system_msg = {
        "role": "system",
        "content": (
            "Eres un LLM que enruta a unha funcion que crea poemas en galego, o unico que debes facer é pasar o poema ao usuario final, non escribas nada mais en o poema devolto"
        )
    }

    # 2) Historial previo con sus roles originales
    history = get_history(conversation_id)  # p.ej. [{'role':'user', ...},
    #        {'role':'assistant', ...},
    #        {'role':'function', ...}]

    # 3) Turno actual del usuario
    user_msg = {"role": "user", "content": query_text}

    # 4) Contexto final
    messages = [system_msg] + history + [user_msg]
    save_message(conversation_id,"user",query_text)
    #print(history,flush=True)

    # 1ª llamada, permitiendo Function Calls
    resp1 = client.chat.completions.create(
        model=model,
        messages=messages,
        functions=FUNCTIONS,
        function_call="auto"
    )
    msg1 = resp1.choices[0].message

    if msg1.function_call:
        fname = msg1.function_call.name
        fargs = json.loads(msg1.function_call.arguments or "{}")

        # Ejecutamos la función
        if fname == "generar_poema":
            func_resp = generar_poema(**fargs)
        else:
            func_resp = f"Error: función desconocida «{fname}»."

        # Añadimos la llamada y la respuesta de la función
        messages.append({
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": fname,
                "arguments": msg1.function_call.arguments
            }
        })
        messages.append({
            "role": "function",
            "name": fname,
            "content": func_resp
        })

        # 2ª llamada para obtener la respuesta final
        resp2 = client.chat.completions.create(
            model=model,
            messages=messages
        )
        respuesta_final = recortar_despues_de_think(resp2.choices[0].message.content)
        save_message(conversation_id, "assistant", respuesta_final)
        return respuesta_final

    # Si no se invocó función, devolvemos directamente
    respuesta_final=recortar_despues_de_think(msg1.content)
    save_message(conversation_id, "assistant", respuesta_final)
    return respuesta_final
