# -*- coding: utf-8 -*-
"""
rangos_y_rima.py

Calcula para cada verso:
  • El rango de sílabas (mínimo/máximo) considerando:
    – sinalefa entre palabras
    – dialefa (no sinalefa)
    – sinéresis (fusionar sólo hiatos sin tilde)
    – diéresis (romper cualquier diptongo)
    – ajuste clásico final (+1 agudo, –1 esdrújulo)
  • La rima asonante correcta (elimina la 'u' muda tras g/q y saca solo vocales).
"""

import re
import unicodedata
from typing import List, Tuple

import pyphen

# ─── constantes ──────────────────────────────────────────────────────────────

DIC      = pyphen.Pyphen(lang="gl")
VOWELS   = "aeiouáéíóúü"
DIP_RE = re.compile(
    r"(?<![gq])"                     # no hay g o q inmediatamente antes
    r"(?:[iuüy][aeo]|[aeo][iuüy]|[iuüy]{2}|ai|au|ei|eu|oi|ou)",
    re.I
)
# ─── utilidades ──────────────────────────────────────────────────────────────

def _strip_acc(t: str) -> str:
    """Quita tildes de la cadena."""
    return "".join(
        c for c in unicodedata.normalize("NFD", t)
        if unicodedata.category(c) != "Mn"
    )

def _silabear(pal: str) -> List[str]:
    """Silabea con Pyphen, reúne diptongos y separa hiatos con tilde."""
    parts = DIC.inserted(pal).split('-')
    # 1) volver a unir diptongos que Pyphen separa
    tmp = []
    for s in parts:
        if tmp and tmp[-1][-1] in VOWELS and s and s[0] in "iuuy":
            tmp[-1] += s
        else:
            tmp.append(s)
    # 2) separar hiatos marcados por tilde
    res = []
    for syl in tmp:
        frag = ""
        for i, ch in enumerate(syl):
            frag += ch
            if ch in "áéíóú" and i + 1 < len(syl) and syl[i+1] in VOWELS:
                res.append(frag)
                frag = ""
        if frag:
            res.append(frag)
    return res or [pal]

def _ajuste_final(pal: str) -> int:
    """
    Ajuste de sílaba final:
      +1 si la sílaba tónica es la última  (agudo),
       0 si es la penúltima                (llano),
      -1 si está antes de la penúltima     (esdrújulo).
    Esto también cubre automáticamente los monosílabos
    (su única sílaba es la última → +1).
    """
    syls    = _silabear(pal)
    idx_char= _tonic_index(pal, syls)

    # determinar índice de sílaba tónica (0‑based)
    pos = 0
    for i, syl in enumerate(syls):
        if idx_char < pos + len(syl):
            stressed = i
            break
        pos += len(syl)
    else:
        stressed = len(syls) - 1

    n = len(syls)
    if stressed == n - 1:
        return 1   # agudo
    if stressed == n - 2:
        return 0   # llano
    return -1      # esdrújulo

# ─── funciones principales ───────────────────────────────────────────────────

def rango_silabas(verso: str) -> Tuple[int, int]:
    """
    Devuelve (mín, máx) de sílabas poéticas para el verso dado,
    ignorando todo lo que no sea letra o espacio.
    """
    # ————————— Limpieza: solo letras y espacios —————————
    verso = re.sub(r"[^A-Za-zÁÉÍÓÚáéíóúÜüÑñ ]+", "", verso)

    if not verso.strip():
        return 0, 0

    # Normalizar palabras
    toks = [re.sub(r"[^\wáéíóúüÿ]", "", w) for w in verso.split()]
    syls = [_silabear(t) for t in toks]

    # 1) contar todas las sílabas + sinalefa entre palabras
    total = 0
    prev_v = False
    for sl in syls:
        first = True
        for s in sl:
            if first and prev_v and s[0] in VOWELS:
                total -= 1
            total += 1
            first = False
        prev_v = sl[-1][-1] in VOWELS
    min_s = total
    max_s = sum(len(sl) for sl in syls)

    # 2) licencias internas (sinéresis + diéresis)
    sin_tot = 0
    dier_tot = 0
    for sl in syls:
        # diéresis: romper diptongos
        dier_tot += len(DIP_RE.findall("".join(sl)))
        # sinéresis: fusionar hiato SÓLO si ninguna lleva tilde
        for i in range(len(sl) - 1):
            v1, v2 = sl[i][-1], sl[i+1][0]
            if (v1 in VOWELS and v2 in VOWELS
                and v1 not in "áéíóú" and v2 not in "áéíóú"):
                sin_tot += 1

    min_s -= sin_tot
    max_s += dier_tot

    # 3) ajuste clásico final
    adj = _ajuste_final(toks[-1])
    min_s += adj
    max_s += adj

    return min_s, max_s

def _tonic_index(pal: str, _unused=None) -> int:
    """
    Índice 0‑based de la vocal tónica:

    • Si hay tilde, devuelve esa posición.
    • Sin tilde:
        – si acaba en vocal/n/s/y  → penúltimo grupo de vocales,
        – si no                     → último grupo de vocales.
      Dentro del grupo se elige la primera vocal fuerte (a/e/o),
      o la primera vocal si no hay fuerte.
    """
    # 1) tilde explícita
    for i, c in enumerate(pal):
        if c in "áéíóú":
            return i

    # 2) grupos de vocales (de izquierda a derecha)
    groups = [(m.start(), m.group()) for m in re.finditer(r"[aeiouü]+", pal)]
    if not groups:                                # sin vocal (muy raro)
        return len(pal) - 1

    # 3) elegir el grupo tónico
    grp_idx = -2 if re.search(r"[nsaeiouy]$", pal) and len(groups) > 1 else -1
    start, box = groups[grp_idx]                  # posición + secuencia de vocales

    # 4) dentro del grupo: primera fuerte o primera
    for j, ch in enumerate(box):
        if ch in "aeo":
            return start + j
    return start                                  # cae en i/u


def rima_consonante(pal: str) -> str:
    """
    Extrae a rima consonante correcta.

    • Normaliza: minúsculas, quita non‑letras, elimina 'h' e 'u' muda (gue/gui, que/qui).
    • Localiza a vocal tónica (_tonic_index).
    • Desde esa posición ata o final:
        – elimina tildes (e diaereses se as houbera)
        – devolve todo o sufixo (vogais e consoantes).
    """
    # 1) limpeza básica
    w = re.sub(r"[^\wáéíóúüÿ]", "", pal.lower()).replace("h", "")
    # eliminar 'u' muda en gue/gui, que/qui
    w = re.sub(r"([gq])u(?=[ie])", r"\1", w)
    if not w:
        return ""

    # 2) localizar vocal tónica
    syls = _silabear(w)
    idx  = _tonic_index(w, syls)

    # 3) sufixo desde a tónica, sen tildes nin diaeresis
    suf  = _strip_acc(w[idx:])
    # opcional: se _strip_acc non elimina 'ü' ou 'ÿ', podes facer:
    # suf = suf.replace("ü", "u").replace("ÿ", "y")

    # 4) devolve todo o sufixo
    return suf

def rima_asonante(pal: str) -> str:
    """
    Extrae la rima asonante correcta.

    • Normaliza: minusculas, quita no‑letras, elimina 'h' y 'u' muda (gue/gui, que/qui).
    • Localiza la vocal tónica (_tonic_index).
    • Desde esa vocal hasta el final:
        – elimina tildes
        – si hay una i/u inicial y **la siguiente letra también es vocal fuerte**
          (a/e/o) **sin consonante entremedias**, la descarta (diptongo real).
        – devuelve solo las vocales resultantes.
    """
    # 1) limpieza básica
    w = re.sub(r"[^\wáéíóúüÿ]", "", pal.lower()).replace("h", "")
    w = re.sub(r"([gq])u(?=[ie])", r"\1", w)   # 'u' muda
    if not w:
        return ""

    # 2) localizar vocal tónica
    syls = _silabear(w)
    #print(syls)
    idx  = _tonic_index(w, syls)

    # 3) sufijo desde la tónica
    suf  = _strip_acc(w[idx:])                 # sin tildes
    #print(suf)
    # 4) descartar semivocal i/u inicial SÓLO si forma diptongo inmediato
    #if (len(suf) >= 2 and suf[0] in "iu" and suf[1] in "aeo"):
    #    suf = suf[1:]
    #print(suf)
    # 5) quedarse sólo con vocales
    return "".join(c for c in suf if c in "aeiou")

# ─── demostración ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    poema = """
Adios rios, adios fontes,
Adios regatos pequenos,
Adios vista dos meus ollos
Non sei cando nos veremos.

"""

    for verso in poema.strip().splitlines():
        mn, mx = rango_silabas(verso)
        r = rima_asonante(verso.split()[-1])
        print(f"{mn}/{mx}  rima='{r}'")

    print(rango_silabas(verso))
    print(rima_consonante(verso.split()[-1]))