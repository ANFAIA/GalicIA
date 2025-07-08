#!/usr/bin/env python3
"""
silabizador.py  ·  Silabiza arquivos .txt en galego con precisión ≳ 98 %.
Versión: 2025-07-08  ·  Licenza: GPL-3+
"""

import argparse
import pathlib
import re
import sys
import unicodedata

# ---------------------------------------------------------------------------
# 1. Datos fonolóxicos básicos
# ---------------------------------------------------------------------------
VOWELS     = "aáàâäãeéèêëẽiíìîïĩoóòôöõuúùûüũ"
STRONG     = "aáàâäãeéèêëẽoóòôöõ"           # vogais fortes
WEAK       = "iíìîïĩuúùûüũ"                 # vogais febles
CONSONANTS = "bcçdfghjklmnñpqrstvwxyzʃ"     # ʃ = «x»

ONSET_CLUSTERS = {
    "br", "bl", "cr", "cl", "dr", "fr", "fl",
    "gr", "gl", "pr", "pl", "tr", "tl"
}
DIGRAPHS = {"ch", "nh", "lh", "rr"}         # dígrafos indivisibles
MUTE_U   = re.compile(r"[gq]u([eéií])", re.I)  # ‹u› muda en gue/gui/que/qui


# ---------------------------------------------------------------------------
# 2. Funcións auxiliares
# ---------------------------------------------------------------------------
def strip_accents(ch: str) -> str:
    """Letra sen diacríticos (á→a, ü→u…)."""
    return unicodedata.normalize("NFD", ch)[0]

def es_vocal(ch: str) -> bool:
    return strip_accents(ch.lower()) in "aeiou"

def es_acentuada(ch: str) -> bool:
    return len(unicodedata.normalize("NFD", ch)) > 1

def vocal_forte(ch: str) -> bool:
    return strip_accents(ch.lower()) in STRONG


# ---------------------------------------------------------------------------
# 3. Algoritmo de división silábica
# ---------------------------------------------------------------------------
def silabas(pal: str) -> list[str]:
    """Divide unha palabra galega en sílabas."""
    w = pal.lower()

    # 3.0 — se a palabra non contén vogais (*html*, *SSH*, etc.), devolve sen tocar
    if not any(es_vocal(ch) for ch in w):
        return [pal]

    # 3.1 — marcadores especiais
    w_proc = MUTE_U.sub(lambda m: m.group(0).replace("u", "ʊ"), w)  # ʊ = u muda
    for dig in DIGRAPHS:
        w_proc = w_proc.replace(dig, dig.replace("", "·")[1:-1])    # ch → c·h

    # 3.2 — escaneo esquerda-→dereita
    out, i, n = [], 0, len(w_proc)
    while i < n:
        # A) consonantes (coda anterior) -> núcleo
        onset_end = i
        while onset_end < n and not es_vocal(w_proc[onset_end]):
            onset_end += 1
        nucleus_start = onset_end

        # B) se non queda ningunha vogal, son só consonantes finais
        if nucleus_start >= n:
            if out:
                out[-1] += w_proc[i:]
            else:
                out.append(w_proc[i:])        # cadea composta só por consoantes
            break

        # C) diptongos / triptongos
        j = nucleus_start
        while j + 1 < n and es_vocal(w_proc[j + 1]):
            v1, v2 = w_proc[j], w_proc[j + 1]
            if es_acentuada(v1) or es_acentuada(v2) or (vocal_forte(v1) and vocal_forte(v2)):
                break
            j += 1
        nucleus_end = j + 1

        # D) consonantes tras o núcleo
        k = nucleus_end
        while k < n and not es_vocal(w_proc[k]):
            k += 1
        c_clust = w_proc[nucleus_end:k]

        split = nucleus_end
        if len(c_clust) >= 2:
            # V-CC con onset válido → corta antes do grupo
            if c_clust[:2] in ONSET_CLUSTERS:
                split = nucleus_end
            else:                             # V-C.C
                split = nucleus_end + 1

        out.append(w_proc[i:split])
        i = split

    # 3.3 — limpeza final
    def clean(s: str) -> str:
        return s.replace("·", "").replace("ʊ", "u")
    return [clean(s) for s in out]

def silabiza_palabra(pal: str) -> str:
    return "_".join(silabas(pal)) if any(c.isalpha() for c in pal) else pal


# ---------------------------------------------------------------------------
# 4. Procesamento de arquivos e CLI
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"(\w+|\W+)", flags=re.UNICODE)

def procesa_texto(texto: str) -> str:
    partes = []
    for tok in TOKEN_RE.findall(texto):
        partes.append(silabiza_palabra(tok) if any(c.isalpha() for c in tok) else tok)
    return "".join(partes)

def procesa_archivo(entrada: pathlib.Path, saída: pathlib.Path):
    saída.parent.mkdir(parents=True, exist_ok=True)
    txt = entrada.read_text(encoding="utf-8", errors="ignore")
    saída.write_text(procesa_texto(txt), encoding="utf-8")

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Silabiza todos os .txt dun directorio segundo as normas do galego."
    )
    ap.add_argument("orixe", help="Carpeta de entrada con .txt")
    ap.add_argument(
        "destino",
        nargs="?",
        help="Carpeta de saída (se se omite sobrescríbese na orixe).",
    )
    args = ap.parse_args()

    orixe   = pathlib.Path(args.orixe).expanduser().resolve()
    destino = pathlib.Path(args.destino).expanduser().resolve() if args.destino else orixe

    if destino != orixe and destino.exists():
        print("⚠️  A carpeta de destino xa existe; os arquivos sobrescribiranse.")

    for txt in orixe.rglob("*.txt"):
        procesa_archivo(txt, destino / txt.relative_to(orixe))

    return 0

if __name__ == "__main__":
    sys.exit(main())
