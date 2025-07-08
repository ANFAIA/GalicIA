#!/usr/bin/env python3
"""
Mueve cada subcarpeta de SOURCE_DIR a DEST_DIR conservando todo su contenido.
Si ya existe una carpeta con el mismo nombre en DEST_DIR, añade un sufijo numérico.
"""

from pathlib import Path
import shutil

SOURCE_DIR = Path(r"../galiciana_data")   # ⇦ carpeta que contiene las subcarpetas
DEST_DIR   = Path(r"../data_norm")  # ⇦ donde quieres llevarlas

def unique_path(dest: Path) -> Path:
    """Genera un nombre único si la carpeta ya existe."""
    if not dest.exists():
        return dest
    i = 1
    while True:
        candidate = dest.with_name(f"{dest.name}_{i}")
        if not candidate.exists():
            return candidate
        i += 1

def main():
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"{SOURCE_DIR} no es una carpeta.")
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    for sub in SOURCE_DIR.iterdir():
        if sub.is_dir():
            target = unique_path(DEST_DIR / sub.name)
            print(f"→ {sub}  ➜  {target}")
            shutil.move(str(sub), str(target))

if __name__ == "__main__":
    main()
