# Generador de dataset con sílabas MIXTAS por verso.
# - INPUT solo en: {"conversations": [{"role":"user","content":"..."}]}
# - Estructura: se repite por estrofa, con cada verso como {"syllables": int, "rhyme": str|None}
# - Control de combinatoria con muestreo (cap por nº de versos)

import json
import random
from pathlib import Path
from itertools import product

# -------------------- Configuración --------------------
# Conjunto de sílabas posibles para construir secuencias mixtas por verso:
SYLLABLE_POOL = [5,6,7,8,9,10,11,12,13,14]   # incluye 14 para teus esquemas longos
STANZA_COUNTS = [1,2,3,4,5]                  # nº de estrofas a repetir
RANDOM_SEED = 1234
random.seed(RANDOM_SEED)

# Patrones de rima por nº de versos (None = verso sen rima / libre)
PATTERNS = {
    2: [
        ["A","A"], ["A","B"], [None,None]
    ],
    3: [
        ["A","A","A"], ["A","A","B"], ["A","B","A"], ["A","B","B"],
        [None,"A",None], ["A",None,"A"], [None,None,None]
    ],
    4: [
        ["A","A","A","A"], ["A","A","B","B"], ["A","B","A","B"], ["A","B","B","A"],
        ["A","B","C","B"], [None,"A",None,"A"], [None,"B",None,"B"], [None,None,None,None],
        ["A","B","C","D"]
    ],
    5: [
        ["A","A","B","B","A"], ["A","B","A","B","A"], ["A","B","B","B","A"], ["A","B","C","B","A"],
        ["A","B","C","D","E"], [None,"A","A",None,"A"]
    ],
    6: [
        ["A","B","A","B","C","C"], ["A","A","B","B","C","C"], ["A","B","C","C","B","A"],
        ["A","B","C","A","B","C"], ["A","B","C","D","E","F"], [None,"A",None,"A",None,"A"]
    ]
}

# Plantillas de prompt (galego)
TEMPLATES = [
    "Xera un poema con estrutura {scheme} de {n_est} estrofas.",
    "Compoñe un poema seguindo o patrón {scheme} e {n_est} estrofas.",
    "Crea un poema de {n_est} estrofas co esquema {scheme}.",
    "Elabora un poema (métrica e rima) {scheme}, {n_est} estrofas."
]
NUM_GL = {1:"unha", 2:"dúas", 3:"tres", 4:"catro", 5:"cinco"}

# Control de combinatoria:
MODE = "sample"            # "all" para TODAS as combinacións; "sample" para mostra
CAP_PER_LENGTH = 2000      # máximo de secuencias de sílabas por nº de versos (se MODE="sample")
TWO_TEXT_VARIANTS = True   # 2 redaccións por combinación

# Si queres excluir casos uniformes (todas as liñas co mesmo nº de sílabas):
MIXED_ONLY = False         # True -> só secuencias con sílabas distintas nalgunha liña

# -------------------- Utilidades --------------------
def scheme_string(syll_seq, label_seq):
    # Ex.: [7,8,10] + ["A",None,"B"] -> "7a 8- 10b"
    parts = []
    for s, lab in zip(syll_seq, label_seq):
        parts.append(f"{s}-" if lab is None else f"{s}{lab.lower()}")
    return " ".join(parts)

def build_structure(syll_seq, label_seq, n_stanzas):
    stanza = [{"syllables": int(s), "rhyme": (lab if lab is not None else None)}
              for s, lab in zip(syll_seq, label_seq)]
    return [list(stanza) for _ in range(n_stanzas)]

def generate_syllable_sequences(n_lines, pool, mode="sample", cap=2000, mixed_only=False):
    """
    Devuelve listas de longitud n_lines con sílabas por verso.
    mode="all": todas las combinaciones (len(pool)**n_lines) -> ¡peligroso!
    mode="sample": muestrea hasta 'cap' combinaciones diferentes.
    mixed_only=True: descarta secuencias uniformes (todos los valores iguales).
    """
    if mode == "all":
        combos = list(product(pool, repeat=n_lines))
        if mixed_only:
            combos = [c for c in combos if len(set(c)) > 1]
        return [list(c) for c in combos]

    # sample
    seen = set()
    seqs = []
    trials = max(cap * 20, 5000)  # intentos para lograr diversidad
    for _ in range(trials):
        c = tuple(random.choice(pool) for _ in range(n_lines))
        if mixed_only and len(set(c)) == 1:
            continue
        if c not in seen:
            seen.add(c)
            seqs.append(list(c))
            if len(seqs) >= cap:
                break
    return seqs

# -------------------- Construcción del dataset --------------------
def build_record(idx, syll_seq, labels, n_st, variant_idx=0):
    sch = scheme_string(syll_seq, labels)
    t = TEMPLATES[variant_idx % len(TEMPLATES)]
    n_est_text = NUM_GL.get(n_st, str(n_st))
    user_text = t.format(scheme=sch, n_est=n_est_text)

    return {
        "id": f"gl_poetry_mixed_{idx:08d}",
        "conversations": [
            {"role": "user", "content": user_text}
        ],
        "structure": build_structure(syll_seq, labels, n_st),
        "metadata": {
            "stanza_count": n_st,
            "lines_per_stanza": len(labels),
            "syllables_per_line": syll_seq,
            "pattern_labels": labels,
            "language": "gl",
            "task": "poetry_generation_with_structure_mixed"
        }
    }

def main():
    records = []
    idx = 0

    for n_lines, label_patterns in PATTERNS.items():
        # generar secuencias de sílabas (mixtas o non) para este nº de versos
        syll_seq_list = generate_syllable_sequences(
            n_lines, SYLLABLE_POOL, mode=MODE, cap=CAP_PER_LENGTH, mixed_only=MIXED_ONLY
        )

        for labels in label_patterns:
            for syll_seq in syll_seq_list:
                for n_st in STANZA_COUNTS:
                    if TWO_TEXT_VARIANTS:
                        for v in range(2):
                            records.append(build_record(idx, syll_seq, labels, n_st, variant_idx=v))
                            idx += 1
                    else:
                        records.append(build_record(idx, syll_seq, labels, n_st, variant_idx=0))
                        idx += 1

    # Añade tu ejemplo específico (como pediste) para garantizar su presencia
    example_text = ("Dame un poema que exprese o pesar dunha muller pola traizón do seu amigo/amado, "
                    "mostrando o dor e o desexo de vinganza sentimental, ao modo dunha cantiga de amigo "
                    "do Trovadorismo, ao estilo de FERNÁN_VELHO, cun esquema métrico "
                    "7– 8A 10– 11B 7– 11B 9– 12A 10– 11C 11– 14C de rima asonante.")
    ex_syll = [7,8,10,11,7,11,9,12,10,11,11,14]
    ex_labels = [None,"A",None,"B",None,"B",None,"A",None,"C",None,"C"]  # "-" -> None
    ex_stanzas = 2
    records.append({
        "id": f"gl_poetry_mixed_example_{idx:08d}",
        "conversations": [{"role": "user", "content": example_text}],
        "structure": build_structure(ex_syll, ex_labels, ex_stanzas),
        "metadata": {
            "stanza_count": ex_stanzas,
            "lines_per_stanza": len(ex_labels),
            "syllables_per_line": ex_syll,
            "pattern_labels": ex_labels,
            "rhyme_type": "asonante",
            "language": "gl",
            "task": "poetry_generation_with_structure_mixed"
        }
    })
    idx += 1

    # Guardar a JSONL
    out_path = Path("poetry_structure_dataset_gl_MIXED.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Guardado JSONL en: {out_path}")
    print(f"Total filas: {len(records)}")

    # ---------- (Opcional) Guardar en formato Hugging Face ----------
    # Evita errores de Arrow gracias a que structure ya está en dicts tipados.
    try:
        from datasets import Dataset, Features, Sequence, Value
        features = Features({
            "id": Value("string"),
            "conversations": Sequence({
                "role": Value("string"),
                "content": Value("string"),
            }),
            "structure": Sequence(
                Sequence({
                    "syllables": Value("int32"),
                    "rhyme": Value("string"),  # admite null
                })
            ),
            "metadata": {
                "stanza_count": Value("int32"),
                "lines_per_stanza": Value("int32"),
                "syllables_per_line": Sequence(Value("int32")),
                "pattern_labels": Sequence(Value("string")),
                "rhyme_type": Value("string"),
                "language": Value("string"),
                "task": Value("string"),
            }
        })
        ds = Dataset.from_list(records, features=features)
        ds.save_to_disk("poemas_GalicIA_est")
        print("HF Dataset gardado en: poemas_GalicIA_est")
    except Exception as e:
        print("Aviso: non se gardou en formato HF. Mensaxe:", e)

if __name__ == "__main__":
    main()
