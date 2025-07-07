import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

def extraer_urls_poemas(url_categoria):
    # Hacer la petición HTTP
    resp = requests.get(url_categoria)
    resp.raise_for_status()

    # Parsear el HTML
    soup = BeautifulSoup(resp.text, 'html.parser')

    # Buscar el contenedor de las páginas de la categoría
    contenedor = soup.find(id='mw-pages')
    if not contenedor:
        print("No se encontró la sección de páginas.")
        return []

    # Extraer todos los enlaces dentro del contenedor
    enlaces = contenedor.find_all('a', href=True)
    urls_poemas = []
    for a in enlaces:
        href = a['href']
        # Ignorar enlaces que no sean a artículos (p. ej. navegación interna)
        if href.startswith('/wiki/'):
            full_url = urljoin(url_categoria, href)
            urls_poemas.append(full_url)

    return urls_poemas

if __name__ == '__main__':
    url = 'https://gl.wikisource.org/wiki/Categoría:Poesía'
    poemas = extraer_urls_poemas(url)
    for u in poemas:
        print(u)
