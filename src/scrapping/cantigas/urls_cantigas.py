from typing import List
from urllib.parse import urljoin
import string
import requests
from bs4 import BeautifulSoup

def get_cantigas_urls() -> List[str]:
    """
    Devuelve la lista (ordenada) de URLs de todas las cantigas publicadas en
    https://cantigas.fcsh.unl.pt/listacantigas.asp
    """
    BASE       = "https://cantigas.fcsh.unl.pt/"
    LIST_PAGE  = urljoin(BASE, "listacantigas.asp")

    def _page_links(letter: str | None = None, session: requests.Session | None = None) -> set[str]:
        params  = {"letra": letter} if letter else {}
        s       = session or requests.Session()
        r       = s.get(LIST_PAGE, params=params)
        r.encoding = "windows-1252"       # el HTML viene en cp1252
        soup    = BeautifulSoup(r.text, "html.parser")

        # En la primera celda de cada fila (<td width="300">) siempre aparece
        # el enlace único a la cantiga (cantiga.asp?cdcant=…) :contentReference[oaicite:0]{index=0}
        return {
            urljoin(BASE, a["href"])
            for a in soup.select('td[width="300"] > a[href*="cantiga.asp"]')
        }

    session = requests.Session()
    links   = set()

    # índice general + todas las letras A-Z
    links.update(_page_links(None, session))
    for letter in string.ascii_uppercase:
        links.update(_page_links(letter, session))

    return sorted(links)

if __name__ == "__main__":
    urls = get_cantigas_urls()
    print(f"Total: {len(urls)} cantigas")
    for u in urls[:5]:
        print(u)
