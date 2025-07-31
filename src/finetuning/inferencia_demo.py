import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.extractor_métrica.procesar_poema import rango_silabas, rima_consonante, rima_asonante

checkpoint = "galicIA-v1"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True)
model.eval()

# Prompt inicial
initial_prompt = "Xera un poema sobre un gato."

# Prepara inputs para history
tokens = tokenizer.apply_chat_template(
    [{"role": "user", "content": initial_prompt}],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=False
)
newline_id = tokenizer.convert_tokens_to_ids("\n")
question_end = tokenizer.convert_tokens_to_ids("?")
exclamation_end=tokenizer.convert_tokens_to_ids("!")
init_history_ids = tokens["input_ids"] if isinstance(tokens, dict) else tokens
init_mask = tokens.get("attention_mask", None) if isinstance(tokens, dict) else None

# Procesadores de logits
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
rep_proc    = RepetitionPenaltyLogitsProcessor(penalty=1.6)
temp_warper = TemperatureLogitsWarper(temperature=0.8)
topk_warper = TopKLogitsWarper(top_k=50)
topp_warper = TopPLogitsWarper(top_p=0.95)

# Heurística de finalización: booster EOS cuando estamos cerca del límite

def adjust_for_ending(logits, history_ids, target_syl, current_syl,inte,exc):
    """
    Booster EOS y '\n' sólo cuando estamos en el umbral;
    si no, los penaliza para que no terminen prematuramente.
    """
    eos_id = tokenizer.eos_token_id
    #print(current_syl)
    # Si aún no llegamos a la sílaba objetivo, penalizamos EOS y newline
    if current_syl < target_syl:
        logits[:, eos_id]   -= 99.0
        logits[:, newline_id] -= 99.0
    else:
        # Cerca o pasados de target_syl, permitimos cerrar el verso
        logits[:, eos_id]   += 18.0
        logits[:, newline_id] += 18.0
        if inte>0:
            logits[:, question_end] += 10.0
        else:
            logits[:, question_end] -= 99.0
        if exc > 0:
            logits[:, exclamation_end] += 10.0
        else:
            logits[:, exclamation_end] -= 99.0
    return logits

# Función para heurística de rima: boost tokens que terminan con rima deseada

def adjust_for_rhyme(logits, target_rhyme, current_syl, target_syl, tokenizer):
    """
    Si estamos cerca del final del verso (últimas 2 sílabas), aumentar logits
    de tokens cuya cadena termina con el patrón de rima.
    """

    if target_rhyme and current_syl >= target_syl - 2:
        print(f"  [Heurística Rima] Cerca fin, boost tokens con rima '{target_rhyme}'")
        # Recorremos topk vocab ids para boost (por eficiencia)
        for token_id in range(logits.size(-1)):
            token_str = tokenizer.decode(token_id)
            if token_str.endswith(target_rhyme):
                logits[:, token_id] += 2.0
    return logits

# Muestreo token a token con logging y heurística

def sample_token(history_ids, past_kv, mask, target_syl, current_syl, inte,exc,target_rhyme=None):
    outputs = model(
        input_ids=history_ids,
        attention_mask=mask,
        past_key_values=past_kv,
        use_cache=True,
    )
    logits = outputs.logits[:, -1, :]
    past_kv = outputs.past_key_values

    # Aplicar procesadores
    l = rep_proc(history_ids, logits)
    l = temp_warper(history_ids, l)
    l = adjust_for_ending(l, history_ids, target_syl, current_syl,inte,exc)
    # Ajustar logits según heurística de rima
    #l = adjust_for_rhyme(l, target_rhyme, current_syl, target_syl, tokenizer)
    l = topk_warper(history_ids, l)
    l = topp_warper(history_ids, l)

    # Debug top-10
    probs = torch.softmax(l, dim=-1)
    vals, idxs = torch.topk(probs, 10, dim=-1)
    entries = [f"{tokenizer.decode(int(i)).replace('\n','↵')}:{float(v)*100:5.2f}%" for i, v in zip(idxs[0], vals[0])]
    #print("    ▶ Top-10:", ", ".join(entries))

    # Muestreo final
    next_token = torch.multinomial(probs, num_samples=1)
    tok_str = tokenizer.decode(next_token[0])
    if tok_str!='\n':
        print(f"{tok_str}", end="")
    return next_token, past_kv

# Generación de poema con métrica, rima, heurística y backtracking parcial

def generate_structured_poem(structure, max_attempts=8, max_steps=100):
    poem_lines = []

    for stanza_idx, stanza in enumerate(structure, start=1):
        rhyme_map = {}
        print(f"\n=== Estrofa {stanza_idx} ===")

        for verse_idx, (target_syl, letter) in enumerate(stanza, start=1):
            inte=0
            exc=0
            print(f"\n-- Verso {verse_idx}: objetivo {target_syl} sílabas, rima {letter}")
            success = False

            for attempt in range(1, max_attempts + 1):
                print(f"  * Intento {attempt}")
                history = init_history_ids.clone()
                past_kv = None
                mask = init_mask
                words = []
                current_syl = 0

                for step in range(1, max_steps + 1):
                    # Genera siguiente token
                    next_token, past_kv = sample_token(history, past_kv, mask, target_syl, current_syl,inte,exc)
                    mask = None
                    history = next_token if past_kv else torch.cat([history, next_token], dim=1)
                    w = tokenizer.decode(next_token[0])

                    # Sólo consideramos fin de verso si hemos alcanzado la métrica
                    if (w == tokenizer.eos_token or "\n" in w) and current_syl >= target_syl:
                        print("    - Fin de verso detectado")
                        break

                    # Si sale '\n' antes de tiempo, lo tratamos como parte del verso
                    if "\n" in w and current_syl < target_syl:
                        w = w.replace("\n", "")
                    if "¿" in w and current_syl < target_syl:
                        inte=inte+1
                    if "¡" in w and current_syl < target_syl:
                        exc=exc+1
                    if "?" in w and current_syl < target_syl:
                        inte=inte-1
                    if "!" in w and current_syl < target_syl:
                        exc=exc-1

                    words.append(w)
                    text = ''.join(words)
                    min_s, max_s = rango_silabas(text)
                    current_syl = (min_s + max_s) // 2
                    #print(f"    - Parcial: '{text}', sílabas aprox. {current_syl}")

                verse = ''.join(words).strip()
                print(f"  Generado: '{verse}'")

                # Verificación de métrica
                min_s, max_s = rango_silabas(verse)
                metric_ok = (min_s <= target_syl <= max_s)

                # Verificación de rima
                if letter:
                    last_word = verse.split()[-1] if verse else ''
                    r = rima_asonante(last_word)
                    if letter not in rhyme_map:
                        rhyme_map[letter] = r
                        print(f"  [Rima] Asignada letra '{letter}' -> '{r}'")
                    rhyme_ok = (r == rhyme_map[letter])
                else:
                    rhyme_ok = True

                # Decisión final
                if metric_ok and rhyme_ok:
                    print(f"  ✔ ACEPTADO: '{verse}' [{min_s}-{max_s} sílabas, rima='{rhyme_map.get(letter)}']")
                    poem_lines.append(verse)
                    success = True
                    break
                else:
                    reasons = []
                    if not metric_ok:
                        reasons.append(f"sílabas {min_s}-{max_s}≠{target_syl}")
                    if letter and not rhyme_ok:
                        reasons.append(f"rima '{r}'≠'{rhyme_map[letter]}'")
                    print(f"  ✗ RECHAZADO: {'; '.join(reasons)}")

                    # Si falla rima, recortar y reintentar sin contar intento
                    if letter and not rhyme_ok and len(words) > 3:
                        print("    - Recorte: quitando últimas 3 palabras, reintentando")
                        words = words[:-3]
                        partial = initial_prompt + ' ' + ''.join(words)
                        inp = tokenizer(partial, return_tensors="pt")
                        history = inp.input_ids
                        mask = inp.attention_mask
                        # recalcular current_syl
                        text = ''.join(words)
                        smin, smax = rango_silabas(text)
                        current_syl = (smin + smax) // 2
                        continue

            if not success:
                raise RuntimeError(f"Verso {verse_idx} de estrofa {stanza_idx} no válido")

        poem_lines.append("")  # separador de estrofas

    return "\n".join(poem_lines).strip()

if __name__ == '__main__':
    # Estructura de ejemplo: estrofas con versos (sílabas, letra rima)
    structure = [[(10, None), (8, None), (10, None), (8, None)],
                 [(8, None), (8, None), (8, None), (8, None)]]
    try:
        poem = generate_structured_poem(structure)
    except RuntimeError as e:
        print("Error en generación:", e)
        exit(1)

    print("\n=== POEMA FINAL ===\n", poem)
    with open('poema.txt', 'w', encoding='utf-8') as f:
        f.write(poem)
    print("\n[✓] Poema guardado en 'poema.txt'")
