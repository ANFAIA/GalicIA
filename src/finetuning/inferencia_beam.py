import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.extractor_métrica.procesar_poema import rango_silabas, rima_asonante

# ——— Configuración inicial —————————————————————————————
checkpoint = "galicIA+++"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True)
model.eval()

initial_prompt = "Xera un poema sobre un can e un gato"
tokens = tokenizer.apply_chat_template(
    [{"role": "user", "content": initial_prompt}],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=False
)
init_history_ids = tokens["input_ids"] if isinstance(tokens, dict) else tokens
init_mask = tokens.get("attention_mask", None) if isinstance(tokens, dict) else None
prefix_len = init_history_ids.shape[-1]

# ——— Generación de un verso con beam search —————————————————————
def generate_verse_with_beams(
    history_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_syl: int,
    target_rhyme: str = None,
    num_beams: int = 8,
    max_length: int = 100
):
    eos_id = tokenizer.eos_token_id

    # Generamos num_beams candidatos completos (incluyendo el prefijo)
    sequences = model.generate(
        input_ids=history_ids,
        attention_mask=attention_mask,
        max_length=prefix_len + max_length,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        eos_token_id=eos_id,
        pad_token_id=tokenizer.pad_token_id,
        early_stopping=True,
    )  # shape: (num_beams, prefix_len + gen_len)

    best_score = float('-inf')
    best = None

    for seq in sequences:
        # Extraemos sólo la parte generada (sin el prefijo)
        gen_ids = seq[prefix_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        # Tomamos sólo la primera línea (hasta el primer salto)
        verse = text.split('\n', 1)[0].strip()

        # Métrica
        min_s, max_s = rango_silabas(verse)
        metric_ok = (min_s <= target_syl <= max_s)
        metric_score = -abs((min_s + max_s) / 2 - target_syl)

        # Rima
        last_word = verse.split()[-1] if verse else ''
        rhyme = rima_asonante(last_word)
        rhyme_ok = (target_rhyme is None or rhyme == target_rhyme)
        rhyme_score = 1.0 if rhyme_ok else 0.0

        # Score combinado: priorizamos rima, luego métrica
        total_score = 10 * rhyme_score + metric_score
        if total_score > best_score:
            best_score = total_score
            best = (verse, metric_ok, rhyme_ok, rhyme, min_s, max_s)

    if best is None:
        raise RuntimeError("No se generó ningún candidato de verso válido")

    return best  # (verse, metric_ok, rhyme_ok, rhyme_assigned, min_s, max_s)

# ——— Generación del poema completo ——————————————————————————
def generate_structured_poem_beam(
    structure,
    num_beams: int = 8,
    max_attempts: int = 3
):
    poem_lines = []

    for stanza_idx, stanza in enumerate(structure, start=1):
        rhyme_map = {}
        print(f"=== Estrofa {stanza_idx} ===")

        for verse_idx, (target_syl, letter) in enumerate(stanza, start=1):
            print(f"-- Verso {verse_idx}: {target_syl} sílabas, rima '{letter}'")
            history = init_history_ids.clone()
            mask = init_mask
            target_rhyme = rhyme_map.get(letter)

            verse = None
            for attempt in range(1, max_attempts + 1):
                cand, met_ok, rhy_ok, rhy, min_s, max_s = generate_verse_with_beams(
                    history,
                    mask,
                    target_syl,
                    target_rhyme,
                    num_beams=num_beams
                )
                print(f"  Intento {attempt}: '{cand}' [{min_s}-{max_s} sílabas, rima ok={rhy_ok}]")
                if met_ok and rhy_ok:
                    verse = cand
                    if letter and letter not in rhyme_map:
                        rhyme_map[letter] = rhy
                        print(f"  → Asignada rima '{rhy}' para letra '{letter}'")
                    break

            if verse is None:
                raise RuntimeError(
                    f"Verso {verse_idx} de la estrofa {stanza_idx} falló: "
                    f"sílabas≈{min_s}-{max_s}, rima ok={rhy_ok}"
                )

            poem_lines.append(verse)
        poem_lines.append("")  # separador de estrofas

    return "\n".join(poem_lines).strip()

# ——— Punto de entrada ——————————————————————————————————————
if __name__ == "__main__":
    # Ejemplo: dos estrofas de cuatro versos octosílabos, esquema AABB
    structure = [
        [(8, "A"), (8, "A"), (8, "B"), (8, "B")],
        [(8, "A"), (8, "A"), (8, "B"), (8, "B")],
    ]

    poema = generate_structured_poem_beam(structure)
    print("\n=== POEMA FINAL ===\n")
    print(poema)

    with open("poema_beam.txt", "w", encoding="utf-8") as f:
        f.write(poema)
    print("\n[✓] Poema guardado en 'poema_beam.txt'")
