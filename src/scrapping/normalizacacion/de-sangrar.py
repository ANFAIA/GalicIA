#!/usr/bin/env python3
"""
Elimina cualquier espacio o tabulación al principio de cada línea con texto
(dentro de *1.txt) en todo un árbol de directorios.

Por defecto reescribe cada archivo *in place* creando primero un backup .bak.
"""

from pathlib import Path
import fileinput
import shutil
import sys

ROOT = Path(r"../data_norm")  # ⇦ cámbialo
BACKUP_EXT = ".bak"  # pon None si NO quieres backups


def cleanup_txt(path: Path):
    """Quita sangrado por la izquierda en líneas no vacías."""
    backup = None
    if BACKUP_EXT:
        backup = path.with_suffix(path.suffix + BACKUP_EXT)
        shutil.copy2(path, backup)  # copia de seguridad

    with fileinput.FileInput(path, inplace=True, backup=None) as f:
        for line in f:
            # Si la línea tiene algo distinto de espacios en blanco, la limpiamos
            if line.strip():
                print(line.lstrip(), end="")  # lstrip() borra espacios/tabs a la izquierda
            else:
                print(line, end="")  # líneas vacías intactas


def main():
    txts = list(ROOT.rglob("*1.txt"))
    if not txts:
        sys.exit("No se encontraron 1.txt en " + str(ROOT))

    for txt in txts:
        print(f"Limpiando → {txt}")
        cleanup_txt(txt)


if __name__ == "__main__":
    main()
