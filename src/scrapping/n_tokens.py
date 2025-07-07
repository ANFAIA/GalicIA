import os
from pathlib import Path
import csv
import tiktoken

# Cambia esta ruta por la raíz de tus archivos .txt
ROOT = Path("./galiciana_data")

enc = tiktoken.get_encoding("cl100k_base")   # mismo modelo que usa GPT-4/3.5-turbo
totales = 0
filas_csv = []

for fichero in ROOT.rglob("*.txt"):          # recorre subcarpetas recursivamente
    with fichero.open("r", encoding="utf-8", errors="ignore") as f:
        texto = f.read()
    n_tokens = len(enc.encode(texto))
    print(f"{fichero.relative_to(ROOT)} → {n_tokens} tokens")
    filas_csv.append([str(fichero.relative_to(ROOT)), n_tokens])
    totales += n_tokens

print(f"\nTOTAL carpeta → {totales} tokens")

