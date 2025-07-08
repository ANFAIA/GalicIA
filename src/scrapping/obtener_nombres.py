#!/usr/bin/env python3
"""
Crea un archivo de texto con el nombre de todas las subcarpetas
inmediatas de la carpeta indicada en SRC_DIR.
"""

from pathlib import Path

# ── Configuración ──────────────────────────────────────────────────────────────
SRC_DIR = Path("galiciana_data").expanduser().resolve()  # ← Cambia esta ruta
OUT_FILE = Path("subcarpetas_galiciana.txt").resolve()                 # ← Cambia el destino
# ───────────────────────────────────────────────────────────────────────────────

# Comprobamos que la ruta exista y sea un directorio
if not SRC_DIR.is_dir():
    raise NotADirectoryError(f"'{SRC_DIR}' no es una carpeta válida")

# Obtenemos solo las subcarpetas directas (no recursivo) y sus nombres
subcarpetas = [p.name for p in SRC_DIR.iterdir() if p.is_dir()]

# Guardamos la lista, un nombre por línea
OUT_FILE.write_text("\n".join(subcarpetas) + "\n", encoding="utf-8")

print(f"Se han listado {len(subcarpetas)} subcarpetas en '{OUT_FILE}'.")
