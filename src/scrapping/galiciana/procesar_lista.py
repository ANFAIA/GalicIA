#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lee poemas_con_texto.txt (una URL por línea) y genera poemas_texto_links.jsonl
Detecta automáticamente:
  - POEMA  → devuelve 1 URL
  - INDICE → devuelve lista de URLs (uno por poema)
  - ERROR  → algo fue mal
Dependencias: requests, beautifulsoup4, chardet
"""

from pathlib import Path
from urllib.parse import urljoin, urlparse
import json, re, sys, requests, chardet
from bs4 import BeautifulSoup

INPUT_FILE  = Path("poemas_con_texto.txt")
OUTPUT_FILE = Path("poemas_texto_links.jsonl")

################################################################################
# Utilidades de red y parsing HTML
################################################################################
def fetch_soup(url: str, timeout: int = 20) -> BeautifulSoup | None:
    try:
        r = requests.get(url, timeout=timeout)
        enc = chardet.detect(r.content)["encoding"] or "latin-1"
        r.encoding = enc
        return BeautifulSoup(r.text, "html.parser")
    except Exception as exc:
        print(f"⚠️  Error al acceder a {url}: {exc}", file=sys.stderr)
        return None

def enlace_texto(ficha_url: str) -> str | None:
    """Localiza el <a> «Texto» dentro de una ficha de obra y devuelve la URL absoluta."""
    soup = fetch_soup(ficha_url)
    if not soup:
        return None
    a = soup.find("a", string=lambda s: s and s.strip().lower() == "texto")
    return urljoin(ficha_url, a["href"]) if a and a.get("href") else None

################################################################################
# Heurísticas de clasificación
################################################################################
_re_letter = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]")

def _contiene_letras(s: str) -> bool:
    return bool(_re_letter.search(s))

def clasificar_indice(url: str, soup: BeautifulSoup) -> str:
    """
    Decide si un indice_paxinas.jsp es
      - 'INDICE'  (colección de poemas)
      - 'POEMA'   (poema largo paginado)
    """
    # TODOS los enlaces a páginas internas que acaba de encontrar
    links = [a for a in soup.find_all("a", href=lambda h: h and "paxina.jsp" in h)]
    if not links:
        return "POEMA"          # rarísimo, pero mejor tratarlo como poema único

    textos = [a.get_text(strip=True) for a in links]
    con_letras = sum(_contiene_letras(t) for t in textos)

    # ≥25 % con letras  → títulos de poemas  → colección
    return "INDICE" if con_letras / len(textos) >= 0.25 else "POEMA"

def poemas_de_indice(indice_url: str, soup: BeautifulSoup) -> list[str]:
    """Extrae URLs de cada poema desde un verdadero índice de colección."""
    poemas = []
    for a in soup.find_all("a", href=lambda h: h and "paxina.jsp" in h):
        poemas.append(urljoin(indice_url, a["href"]))
    # quitar duplicados conservando orden
    visto, uniq = set(), []
    for u in poemas:
        if u not in visto:
            uniq.append(u)
            visto.add(u)
    return uniq

################################################################################
# Bucle principal
################################################################################
def main() -> None:
    with INPUT_FILE.open(encoding="utf-8") as fh, \
         OUTPUT_FILE.open("w", encoding="utf-8") as out:

        for ficha in map(str.strip, fh):
            if not ficha:
                continue

            texto_url = enlace_texto(ficha)
            if not texto_url:
                json.dump({"ficha": ficha, "tipo": "ERROR", "dato": None}, out, ensure_ascii=False)
                out.write("\n")
                continue

            path = urlparse(texto_url).path

            # Caso 1 ────────────────────────────────────────────────────────────
            if "paxina.jsp" in path:
                json.dump({"ficha": ficha, "tipo": "POEMA", "dato": texto_url},
                          out, ensure_ascii=False)
                out.write("\n")
                continue

            # Caso 2: indice_paxinas.jsp ───────────────────────────────────────
            soup = fetch_soup(texto_url)
            if not soup:
                json.dump({"ficha": ficha, "tipo": "ERROR", "dato": None}, out, ensure_ascii=False)
                out.write("\n")
                continue

            tipo = clasificar_indice(texto_url, soup)
            if tipo == "INDICE":
                lista = poemas_de_indice(texto_url, soup)
                dato = lista or None
            else:  # poema largo paginado
                dato = texto_url

            json.dump({"ficha": ficha, "tipo": tipo, "dato": dato},
                      out, ensure_ascii=False)
            out.write("\n")

    total = sum(1 for _ in INPUT_FILE.open(encoding="utf-8"))
    print(f"✅  Procesadas {total} fichas — resultado en {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
