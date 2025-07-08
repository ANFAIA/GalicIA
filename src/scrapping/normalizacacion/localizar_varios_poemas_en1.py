#!/usr/bin/env python3
"""
Encuentra *1.txt que contengan una línea 'I' y otra línea 'II' (cada una por separado).
Imprime las rutas relativas al directorio raíz indicado.
"""

from pathlib import Path
import sys
import re

ROOT = Path(r"../data_norm")   # ⇦ cambia esto o pásalo por CLI

# Regex: línea compuesta solo por I o II (se permiten espacios en blanco alrededor)
RX_I  = re.compile(r'^\s*I\s*$',  re.IGNORECASE)
RX_II = re.compile(r'^\s*II\s*$', re.IGNORECASE)

def matches(path: Path) -> bool:
    """Devuelve True si el archivo contiene ambas líneas."""
    has_I = has_II = False
    try:
        with path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not has_I  and RX_I .match(line): has_I  = True
                if not has_II and RX_II.match(line): has_II = True
                if has_I and has_II:
                    return True
    except OSError as e:
        print(f"⚠️  No se pudo leer {path}: {e}", file=sys.stderr)
    return False

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
    for txt in root.rglob("*1.txt"):
        if matches(txt):
            print(txt.relative_to(root))

if __name__ == "__main__":
    main()
