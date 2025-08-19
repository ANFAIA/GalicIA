#!/usr/bin/env python3
"""
Extrae todas las URL con texto disponible de la sección de poesía de la BVG
Autor: …
"""

import re, time, sys
from urllib.parse import urljoin
from pathlib import Path

import requests
from bs4 import BeautifulSoup
try:
    from tqdm import tqdm        # Solo para barra de progreso bonita
except ImportError:
    tqdm = lambda x, **k: x      # Fallback silencioso


START_URL = "https://bvg.udc.es/busqueda_obras.jsp?opcion=todos&categoria=poesia"
BASE      = "https://bvg.udc.es"
OUT_FILE  = Path("poemas_con_texto.txt")

HEADERS = {
    "User-Agent": "poetry-scraper/1.0 (+https://github.com/tu-usuario)"
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def paginate(url: str):
    """Generador que recorre todas las páginas de resultados"""
    while url:
        resp = SESSION.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        yield soup

        nxt = soup.select_one('a[title*="Seguinte"], a:contains("Seguinte")')
        url = urljoin(url, nxt["href"]) if nxt else None


def work_links(listing_soup):
    """Extrae las URL de cada obra (ficha) en una página de listado"""
    for a in listing_soup.select('a[href*="ficha_obra.jsp"]'):
        yield urljoin(BASE, a["href"])


def has_text_format(ficha_soup):
    """
    Determina si la ficha declara formato 'Texto'.
    La BVG lista los formatos dentro de un <div class="formato"> o similar.
    """
    return bool(ficha_soup.find(string=re.compile(r'\bTexto\b', re.I)))


def text_url(ficha_soup):
    """
    Devuelve la URL directa al visor de texto, si existe.
    Suele presentarse como algo del tipo visor_obra_texto.jsp?id_obra=...
    """
    link = ficha_soup.select_one('a[href*="visor_obra_texto.jsp"]')
    return urljoin(BASE, link["href"]) if link else None


def scrape(start_url=START_URL):
    resultados = []

    for page in paginate(start_url):
        for ficha in tqdm(list(work_links(page)), desc="Obras", unit="obra"):
            try:
                r = SESSION.get(ficha, timeout=30)
                r.raise_for_status()
                fsoup = BeautifulSoup(r.text, "html.parser")
                if has_text_format(fsoup):
                    resultados.append(text_url(fsoup) or ficha)
            except Exception as e:
                print(f"⚠️  No se pudo procesar {ficha}: {e}", file=sys.stderr)
            finally:
                time.sleep(0.2)          # Sé amable con el servidor

    return resultados


def main():
    urls = scrape()
    OUT_FILE.write_text("\n".join(urls), encoding="utf-8")
    print(f"\n✅  Encontradas {len(urls)} obras con texto. "
          f"Se han guardado en {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
