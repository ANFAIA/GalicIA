from openai import OpenAI, OpenAIError
import requests
import os
import re
from concurrent.futures import ThreadPoolExecutor

# ── imports de tu proyecto ─────────────────────────────────────────────────────
from src.scrapping.galiciana.procesar_json import extraer_urls
from src.scrapping.wikisource.get_poem_url import extraer_urls_poemas
from src.scrapping.colección_poesia.extractor_autores_url import (
    fetch_html_from_url,
    extract_author_urls_from_html,
)
from src.scrapping.colección_poesia.extractor_poemas_url import extract_poem_urls_from_html
from src.scrapping.cantigas.urls_cantigas import get_cantigas_urls
# ───────────────────────────────────────────────────────────────────────────────

# ── PROMPT ─────────────────────────────────────────────────────────────────────
prompt = """
Eres un extractor de poemas.
Si el HTML contiene varios poemas, escribe **cada poema** con el formato:

```Texto de la poesía```
```Título```
```Autor```

y deja una línea que contenga solo tres guiones
---
entre un poema y el siguiente.

Ejemplo (dos poemas):

```Verso 1
Verso 2```
```Título A```
```Autor A```
---
```Verso 1
Verso 2```
```Título B```
```Autor B```
"""
# ───────────────────────────────────────────────────────────────────────────────


def _fs_name(txt: str) -> str:
    """Convierte un texto en un nombre válido para el sistema de archivos."""
    txt = txt.strip().replace(" ", "_")
    return re.sub(r"[^\w\-]", "_", txt, flags=re.UNICODE)


def save_poem(raw: str, base_dir: str = "./prueba") -> str:
    """
    Espera un string con tres bloques delimitados por ```:
      1. Poema   2. Título   3. Autor
    Guarda el poema en base_dir/Autor/Título.txt y devuelve la ruta.
    """
    blocks = re.findall(r"```(.*?)```", raw, flags=re.S)
    if len(blocks) < 3:
        raise ValueError(
            "Se necesitan tres bloques delimitados por ``` ... ```: poema, título y autor."
        )

    poem, title, author = [b.strip() for b in blocks[:3]]

    dir_path = os.path.join(base_dir, _fs_name(author))
    file_path = os.path.join(dir_path, f"{_fs_name(title)}.txt")

    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        if not poem.endswith("\n"):
            poem += "\n"
        f.write(poem)

    return file_path


def save_poems(raw: str, base_dir: str = "./galiciana_data"):
    """
    Divide la salida del modelo por la línea '---'
    y guarda cada poema individual con save_poem().
    """
    for block in re.split(r"\n\s*---+\s*\n", raw):
        if block.strip():
            try:
                ruta = save_poem(block, base_dir)
                print(f"✔ Guardado en {ruta}", flush=True)
            except ValueError as e:
                print(f"⚠ Bloque ignorado: {e}", flush=True)


def html_to_chat(url: str, model: str = "gpt-4.1-mini", max_chars: int = 400_000):
    """
    Descarga el HTML de `url`, lo trunca a `max_chars`
    y lo envía al endpoint /chat/completions junto con `prompt`.
    """
    client = OpenAI()

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        html = resp.text[:max_chars]

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": html},
                    ],
                }
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content

    except (requests.RequestException, OpenAIError) as e:
        raise RuntimeError(f"Error interno: {e}")


def process_url(url):
    respuesta = html_to_chat(url)
    save_poems(respuesta)
    print(f"Fin {url}", flush=True)


if __name__ == "__main__":

    #print(html_to_chat('https://bvg.udc.es/paxina.jsp?id_obra=14Na19-11&alias=Sebasti%E1n+Mart%EDnez-Risco&id_edicion=14Na19-11001&formato=texto&pagina=2&cabecera=%3Ca+href%3D%22ficha_obra.jsp%3Fid%3D14Na19-11%26alias%3DSebasti%E1n+Mart%EDnez-Risco%22+class%3D%22nombreObraPaxina%22%3E14+Nadali%F1as%3A+1958+-+1973%3C%2Fa%3E&maxpagina=2&minpagina=1'))

    # ── Galiciana ───────────────────────────────────────────────────────────────
    #'''
    archivo = "galiciana/poemas_texto_links.jsonl"
    enlaces = extraer_urls(archivo)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_url, url): url for url in enlaces}
    #'''
    # ── Otros ejemplos (descomenta el que necesites) ───────────────────────────
    """
    # WikiSource
    url = 'https://gl.wikisource.org/wiki/Categoría:Poesía'
    poemas_url = extraer_urls_poemas(url)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_url, url): url for url in poemas_url}

    # Colección Poesía Galega
    html_content = fetch_html_from_url("https://coleccionpoesiagalega.blogspot.com/")
    urls_auth = extract_author_urls_from_html(html_content)
    for url_ath in urls_auth:
        html_label = fetch_html_from_url(url_ath)
        poemas_url = extract_poem_urls_from_html(html_label)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(process_url, url): url for url in poemas_url}

    # Cantigas medievales
    urls_cantigas = get_cantigas_urls()
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_url, url): url for url in urls_cantigas}
    """
