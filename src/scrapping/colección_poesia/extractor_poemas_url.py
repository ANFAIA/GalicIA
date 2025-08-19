import sys
import requests
from bs4 import BeautifulSoup

# URLs fijas del sitio de poesía galega
default_label_url = (
    "https://coleccionpoesiagalega.blogspot.com/search/label/Ferm%C3%ADn%20Bouza-Brey"
)


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

def extract_poem_urls_from_html(html_content):
    """
    Extrae las URLs de los poemas listados en una página de etiqueta.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    poem_urls = set()
    # Cada entrada de poema está en h3.post-title.entry-title > a
    for a in soup.select('h3.post-title.entry-title a'):
        href = a.get('href')
        if href:
            poem_urls.add(href)
    return poem_urls


if __name__ == '__main__':
    # 2) Extraer URLs de poemas para Rosalía de Castro
    print(f"\nDescargando lista de poemas desde: {default_label_url}")
    html_label = fetch_html_from_url(default_label_url)
    poem_urls = extract_poem_urls_from_html(html_label)
    if poem_urls:
        print("\nURLs de poemas encontrados en la etiqueta 'Rosalía de Castro':")
        for url in sorted(poem_urls):
            print(url)
    else:
        print("No se encontraron URLs de poemas en la etiqueta especificada.")
