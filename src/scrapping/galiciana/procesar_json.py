import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import List

def extraer_urls(path: str | Path, *, únicas: bool = True) -> List[str]:
    """
    Devuelve una lista con todas las URL encontradas en el campo ``dato``
    de un fichero JSON Lines.

    Parámetros
    ----------
    path : str | Path
        Ruta al fichero .jsonl
    únicas : bool, opcional (por defecto True)
        Si es True elimina duplicados preservando el orden de aparición.

    Retorna
    -------
    List[str]
        Lista de URL encontradas.
    """
    url_regex = re.compile(r"https?://[^\s\"'<>]+")
    urls: list[str] = []

    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Línea malformada → se ignora
                continue

            dato = obj.get("dato", "")
            # El campo puede ser una cadena o una lista; normalizamos a lista
            items = dato if isinstance(dato, list) else [dato]

            for item in items:
                urls.extend(url_regex.findall(str(item)))

    if únicas:
        # OrderedDict.fromkeys preserva el orden y elimina duplicados
        urls = list(OrderedDict.fromkeys(urls))

    return urls


# Ejemplo de uso
if __name__ == "__main__":
    archivo = "poemas_texto_links.jsonl"
    enlaces = extraer_urls(archivo)
    print(f"Se encontraron {len(enlaces)} URL s")
    print(enlaces[:5])        # muestra las primeras 5
