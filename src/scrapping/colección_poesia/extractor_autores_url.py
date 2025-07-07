import sys
import requests
from bs4 import BeautifulSoup

# URL fija del sitio de poesía galega
default_url = "https://coleccionpoesiagalega.blogspot.com/"


def fetch_html_from_url(url):
    """
    Descarga el contenido HTML de la URL dada.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        sys.exit(f"Error al descargar la URL {url}: {e}")


def extract_author_urls_from_html(html_content):
    """
    Extrae las URLs de autores de un contenido HTML.
    Busca todos los enlaces de etiquetas de autor (/search/label/) y devuelve un conjunto de URLs.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    urls = set()
    # Buscar enlaces que apunten a etiquetas de autor en Blogger
    for link in soup.find_all('a', href=True):
        href = link['href']
        # Filtrar URLs de etiquetas de autor
        if "/search/label/" in href:
            urls.add(href)
    return urls


if __name__ == '__main__':
    print(f"Descargando y procesando: {default_url}")
    html_content = fetch_html_from_url(default_url)

    # Extraer y mostrar URLs de autor
    urls = extract_author_urls_from_html(html_content)
    if urls:
        print("URLs de etiquetas de autor encontradas:")
        for url in sorted(urls):
            print(url)
    else:
        print("No se encontraron URLs de autor en la página.")
