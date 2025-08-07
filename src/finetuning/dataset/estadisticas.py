#!/usr/bin/env python3

from pathlib import Path
import re
from collections import Counter

# Ruta fija a tu carpeta raíz:
RAIZ = Path('data_norm')

# Buscamos todos los *.txt bajo RAIZ (incluye subcarpetas)
txts = list(RAIZ.rglob('*.txt'))
print(f"🔍 Encontrados {len(txts)} archivos .txt bajo «{RAIZ}»\n")

# Regex para sacar lo que hay dentro del segundo par de corchetes
pat = re.compile(r'\[.*?\]\[\s*(.*?)\s*\]', re.DOTALL)

contador = Counter()
sin_coincidencia = []

for path in txts:
    # Abrimos probando utf-8-sig y latin-1
    contenido = None
    for enc in ('utf-8-sig', 'latin-1'):
        try:
            contenido = path.read_text(encoding=enc)
            break
        except Exception:
            continue
    if contenido is None:
        print(f"⚠️ No he podido leer «{path}» en utf-8 ni latin-1")
        continue

    m = pat.search(contenido)
    if m:
        etiqueta = m.group(1).strip()
        contador[etiqueta] += 1
    else:
        sin_coincidencia.append(path)

# Imprimo resultados
print("🏷  Frecuencia de segunda etiqueta:")
print(f"{'Etiqueta':25} | Cantidad")
print("-"*40)
for etiqueta, cnt in contador.most_common():
    print(f"{etiqueta:25} | {cnt}")

# Archivos donde no ha encontrado etiqueta (hasta 10)
if sin_coincidencia:
    print(f"\n❗ {len(sin_coincidencia)} archivos SIN segunda etiqueta reconocida (muestro hasta 10):")
    for p in sin_coincidencia[:10]:
        print("  -", p)
    if len(sin_coincidencia) > 10:
        print(f"  … y {len(sin_coincidencia)-10} más")
