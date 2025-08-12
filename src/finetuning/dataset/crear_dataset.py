import re
from pathlib import Path
from datasets import Dataset, Features, Value
from openai import OpenAI, OpenAIError
import requests
from src.extractor_métrica.procesar_poema import rango_silabas, rima_consonante, rima_asonante


promt_generator="""
Eres un xenerador de promts para un modelo llm, dado un poema, un autor e un movemento en galego.

Debes seguir este formato:
Autor: Rosalía de castro
Movemento: Rexurdimento
Rima:
5/8  rima='oe'
7/8  rima='eo'
7/9  rima='oo'
8/9  rima='eo'
Poema:
Adios rios, adios fontes,
Adios regatos pequenos,
Adios vista dos meus ollos
Non sei cando nos veremos.

Miña terra, miña terra,
Terra donde m' eu criey,
Ortiña que quero tanto,
Figueiriñas que prantey.

Prados, rios, arboredas,
Pinares que move ó vento,
Paxariños piadores,
Casiña dó meu contento.

Muhiño d' os castañares,
Noites craras de luar,
Campaniñas trimbadoras
Dá igrexiña dó lugar.

Amoriñas d' ás silveiras
Qu' eu lle dab' ó meu amor,
Camiñiños antr' ó millo,
Adios para sempr' adios!

Adios groria! adios contento!
Deixo á casa onde nacin,
Deixo á aldea que conoço,
Por un mundo que non vin!

Deixo amigos por estraños,
Deixo á veiga pó lo mar,
Deixo, en fin, canto ben quero...
¡Que pudera non deixar!...

Mais son prob' e mal pecado,
A miña terra n' é miña,
Qu' hastra lle dán de prestado
A veira por que camiña
O que naceu desdichado.

Téñovos pois que deixar,
Hortiña que tanto amei,
Fogueiriña dó meu lár,
Arboriños que prantei,
Fontiña do cabañar.

Adios, adios que me vou,
Herbiñas do camposanto,
Donde meu pay s' enterrou,
Herbiñas que biquey tanto,
Terriña que vos criou.

Adios tamén, queridiña...
Adios por sempre quizais!...
Dígoch' este adios chorando
Desd' á veiriña do mar.
Non m' olvides, queridiña,
Si morro de soidás...
Tantas legoas mar adentro...
¡Miña casiña! meu lar!

Devolverás: 
Dame un poema que trate da despedida nostálxica e dolorosa dun emigrante que abandona a sua tierra natal, expresando a añoranza e o sufrimento que supón a separación do seu fogar, seus paisaxes e seus seres queridos. Falo no estilo de Rosalía de castro no Rexurdimento con un esquema métrico 8– 8a 8– 8a de rima asonante.

8– 8a 8– 8a de rima asonante es el ÚNICO formato de indicar la métrica.
Adapta elnúmero y la letra de la rima al poema dado, la letra es una forma de poner una misma rima pero en lugar po poner una rima pones una letra que la identifica: de 'ao'->'A'o 'B' o 'C' como un diccionario. 
Cuidado podría haber versos libres '-'.
NO pongas la rima original si no su 'letra del diccionario'.
"""

def extraer_primera_estrofa(poema: str) -> str:
    """
    Dado un string con varias estrofas separadas por líneas en blanco,
    devuelve la primera estrofa.
    """
    # Eliminamos espacios al inicio y final
    texto = poema.strip()
    # Separamos por doble nueva línea (estanza en blanco)
    estrofas = texto.split("\n\n")
    # Devolvemos la primera
    return estrofas[0]

# 1) Regex para extraer autor, período y texto
POEM_REGEX = re.compile(
    r"\[(?P<author>[^\]]+)\]"      # [Autor]
    r"\[(?P<period>[^\]]+)\]\s*"   # [Período]
    r"\[inicio\]\s*"               # [inicio]
    r"(?P<text>.*?)"               # cuerpo del poema
    r"\s*\[fin\]",                 # [fin]
    re.DOTALL
)

def parse_poem_file(path: Path):
    """Lee un archivo de poema y extrae autor, período y texto."""
    text = path.read_text(encoding="utf-8")
    m = POEM_REGEX.search(text)
    if not m:
        print(f"⚠️ No se pudo parsear {path}")
        return None
    data = m.groupdict()
    data["text"] = data["text"].strip()
    return data

def limpiar_string(s):
    # 1) eliminar todos los dígitos
    sin_digitos = re.sub(r'\d+', '', s)
    # 2) reemplazar '_' por espacio
    resultado = sin_digitos.replace('_', ' ')
    return resultado

def analizar_estrofa(poema):
    """
    Analiza cada verso del poema y devuelve un string con los resultados
    de recuento silábico y rima asonante.
    """
    resultados = []
    for verso in poema.splitlines():
        verso = verso.strip()
        if not verso:
            # Saltamos líneas vacías
            continue

        # Contamos sílabas
        mn, mx = rango_silabas(verso)

        # Tomamos la última “palabra” y le quitamos puntuación final
        ultima = verso.split()[-1].rstrip(".,;:?!¡¿\"'")

        # Calculamos rima asonante
        r = rima_asonante(ultima)

        resultados.append(f"{mn}/{mx}  rima='{r}'")

    return "\n".join(resultados)


def get_promt(autor: str, movemento: str, poema: str,rima:str,model: str = "gpt-4.1", max_chars: int = 400_000):
    """
    Descarga el HTML de `url`, lo trunca a `max_chars`
    y lo envía al endpoint /chat/completions junto con `prompt`.
    """
    print(f"Procesado autor: {autor}\n")
    client = OpenAI()

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": promt_generator},
                        {"type": "text", "text": f"""Devolver só o a frase do promt.\nAutor: {limpiar_string(autor)}\nMovemento: {movemento}\nPoema: {poema}\nRima: {rima}Devolverás:"""}
                    ],
                }
            ],
        )
        print(completion.choices[0].message.content)
        return completion.choices[0].message.content

    except (requests.RequestException, OpenAIError) as e:
        raise RuntimeError(f"Error interno: {e}")

# 2) Construcción de records con claves role/content
ROOT_DIR = Path("data_norm")
records = []

for txt_path in ROOT_DIR.rglob("*.txt"):
    parsed = parse_poem_file(txt_path)
    if not parsed:
        continue

    completion = parsed["text"]
    estrofa=extraer_primera_estrofa(parsed["text"])
    rima=analizar_estrofa(estrofa)
    # Claves y roles según lo que espera apply_chat_template
    if True:#parsed['period']=="Trovadorismo":
        conv = [
            {"role": "user",      "content": get_promt(parsed['author'],parsed['period'],parsed["text"],rima)},
            {"role": "assistant", "content": completion}
        ]
        records.append({"conversations": conv})

# 3) Definición de features (lista de dicts, sin Sequence)
features = Features({
    "conversations": [
        {"role": Value("string"), "content": Value("string")}
    ]
})

# 4) Crear y guardar el dataset
dataset = Dataset.from_list(records, features=features)
dataset.save_to_disk("bases_movimientos/poemas_GalicIA_trovadorismo")
print(dataset)