from openai import OpenAI, OpenAIError
import requests
import os,re
from wikisource.get_poem_url import extraer_urls_poemas
from concurrent.futures import ThreadPoolExecutor
from colección_poesia.extractor_autores_url import fetch_html_from_url, extract_author_urls_from_html
from colección_poesia.extractor_poemas_url import extract_poem_urls_from_html
from cantigas.urls_cantigas import get_cantigas_urls
prompt="""Eres un esctractor de poema, dado este html saca el poema, pon un salto de linea entre los versos y uno doble entre estrofas.
Basate en la estructura del poema que se te da.
En el texto de la poesía no incluyas ni el titulo, ni el nombre del autor, ni fechas.
Sigue este formato:
```Texto de Poesía```
```Titulo de la poesía```
```Autor de la poesía```
Como:
```Aracana naçaon, máis venturosa,
máis que quantas hoge ha de gloria dina,
pois na prosperidade e na ruína
sempre envexadas estás, nunca envexosa.

Se enresta o ilustre Afonso a temerosa
lança, se arranca a espada que fulmina,
creio que xulgareis que determina
só o conquistar a terra belicosa.```

```Neboeiro```

```Antón Losada Diégue```
"""


def _fs_name(txt: str) -> str:
    """Devuelve nombre apto para el sistema de archivos."""
    txt = txt.strip().replace(' ', '_')
    return re.sub(r'[^\w\-]', '_', txt, flags=re.UNICODE)


def save_poem(raw: str, base_dir: str = './prueba') -> str:
    """
    Espera un único string con tres bloques delimitados por ``` ... ```:
    1. Poema
    2. Título
    3. Autor
    Guarda el poema en .data/Autor/Título.txt y devuelve la ruta.
    """
    # Extraer bloques entre ``` ... ```
    blocks = re.findall(r'```(.*?)```', raw, flags=re.S)
    if len(blocks) < 3:
        raise ValueError(
            "Se necesitan tres bloques delimitados por ``` ... ```: poema, título y autor."
        )

    poem, title, author = [b.strip() for b in blocks[:3]]

    dir_path = os.path.join(base_dir, _fs_name(author))
    file_path = os.path.join(dir_path, f'{_fs_name(title)}.txt')

    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        # Aseguramos salto de línea al final
        if not poem.endswith('\n'):
            poem += '\n'
        f.write(poem)

    return file_path



def html_to_chat(url: str, model: str = "gpt-4.1-mini", max_chars: int = 400_000):
    """
    Descarga el HTML de `url`, lo trunca a `max_chars` caracteres
    y lo envía al endpoint /chat/completions junto con `prompt`.

    Devuelve el string con la respuesta del modelo.
    """
    client = OpenAI()

    try:
        # 1 · Descargar HTML
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        html = resp.text[:max_chars]

        # 2 · Llamada a Chat Completions
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {  # prompt del usuario
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": html}
                    ]
                }
            ],
            temperature=0.2
        )
        return completion.choices[0].message.content

    except (requests.RequestException, OpenAIError) as e:
        raise RuntimeError(f"Error interno: {e}")

def process_url(url):
    save_poem(html_to_chat(url))
    print(f"Guardado {url}",flush=True)

if __name__ == '__main__':

    print(html_to_chat('https://bvg.udc.es/paxina.jsp?id_obra=14Na19-11&alias=Sebasti%E1n+Mart%EDnez-Risco&id_edicion=14Na19-11001&formato=texto&pagina=2&cabecera=%3Ca+href%3D%22ficha_obra.jsp%3Fid%3D14Na19-11%26alias%3DSebasti%E1n+Mart%EDnez-Risco%22+class%3D%22nombreObraPaxina%22%3E14+Nadali%F1as%3A+1958+-+1973%3C%2Fa%3E&maxpagina=2&minpagina=1'))
    # wikisource
    '''
    url = 'https://gl.wikisource.org/wiki/Categoría:Poesía'
    poemas_url = extraer_urls_poemas(url)
    print(poemas_url)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Envía todas las tareas al pool
        futures = {executor.submit(process_url, url): url for url in poemas_url}
    '''

    #coleccion
    '''
    html_content = fetch_html_from_url("https://coleccionpoesiagalega.blogspot.com/")
    urls_auth = extract_author_urls_from_html(html_content)
    for url_ath in urls_auth:
        html_label = fetch_html_from_url(url_ath)
        poemas_url = extract_poem_urls_from_html(html_label)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Envía todas las tareas al pool
            futures = {executor.submit(process_url, url): url for url in poemas_url}
    '''

    #cantigas medievales
    '''
    urls_cantigas = get_cantigas_urls()
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Envía todas las tareas al pool
        futures = {executor.submit(process_url, url): url for url in urls_cantigas}
    '''
