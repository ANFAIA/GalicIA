import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.extractor_métrica.procesar_poema import rango_silabas, rima_consonante, rima_asonante

checkpoint = "galicIA-v1"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True)
model.eval()

# Prompt inicial
def get_initial_history(prompt):
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False
    )
    history_ids = tokens["input_ids"] if isinstance(tokens, dict) else tokens
    mask = tokens.get("attention_mask", None) if isinstance(tokens, dict) else None
    return history_ids, mask

initial_prompt = "Xera un poema sobre xogar ao parchís."
init_history_ids, init_mask = get_initial_history(initial_prompt)
newline_id = tokenizer.convert_tokens_to_ids("\n")
question_end = tokenizer.convert_tokens_to_ids("?")
exclamation_end = tokenizer.convert_tokens_to_ids("!")

# Logits processors
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

# Heurística de finalización: booster EOS/newline cuando cerca del límite
def adjust_for_ending(logits, history_ids, target_syl, current_syl, inte, exc):
    eos_id = tokenizer.eos_token_id
    if current_syl < target_syl:
        logits[:, eos_id]    -= 99.0
        logits[:, newline_id] -= 99.0
    else:
        logits[:, eos_id]    += 10.0
        logits[:, newline_id] += 10.0
        if inte > 0:
            logits[:, question_end] += 10.0
        if exc > 0:
            logits[:, exclamation_end] += 19.0
    return logits

# Muestreo token a token (sin rima/ métrica en beam search)
def sample_token(history_ids, past_kv, mask, target_syl, current_syl, inte, exc):
    outputs = model(
        input_ids=history_ids,
        attention_mask=mask,
        past_key_values=past_kv,
        use_cache=True,
    )
    logits = outputs.logits[:, -1, :]
    past_kv = outputs.past_key_values

    l = rep_proc(history_ids, logits)
    l = temp_warper(history_ids, l)
    l = adjust_for_ending(l, history_ids, target_syl, current_syl, inte, exc)
    l = topk_warper(history_ids, l)
    l = topp_warper(history_ids, l)

    probs = torch.softmax(l, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token, past_kv

# Generación de poema con métrica, rima y beam search exhaustivo
def generate_structured_poem(structure, max_attempts=8, max_steps=100):
    poem_lines = []
    for stanza_idx, stanza in enumerate(structure, start=1):
        rhyme_map = {}
        print(f"\n=== Estrofa {stanza_idx} ===")

        for verse_idx, (target_syl, letter) in enumerate(stanza, start=1):
            inte = exc = 0
            print(f"\n-- Verso {verse_idx}: objetivo {target_syl} sílabas, rima {letter}")
            success = False

            for attempt in range(1, max_attempts+1):
                print(f"  * Intento {attempt}")
                history, mask = init_history_ids.clone(), init_mask
                past_kv = None
                words = []
                current_syl = 0

                for step in range(1, max_steps+1):
                    # Check para usar beam search
                    if letter and current_syl >= target_syl - 6:
                        print("    - Iniciando beam search para finales con rima requerida")
                        beam_out = model.generate(
                            history,
                            attention_mask=mask,
                            max_length=history.size(1) + 50,
                            num_beams=5,
                            early_stopping=True,
                            output_scores=True,
                            return_dict_in_generate=True
                        )
                        seqs = beam_out.sequences
                        scores = beam_out.scores
                        # Mostrar avance por token de cada beam
                        print("      Avance beam por token:")
                        for i, seq in enumerate(seqs, start=1):
                            partial = []
                            print(f"        Beam {i}:")
                            for t, score_t in enumerate(scores, start=1):
                                token_id = seq[history.size(1) + t - 1].item()
                                tok = tokenizer.decode(token_id)
                                partial.append(tok)
                                phrase = ''.join(words) + ''.join(partial)
                                print(f"          Step {t}: token='{tok}' frase='{phrase.strip()}' score={score_t[i-1]:.2f}")
                        # Mostrar candidatos finales
                        print("      Candidatos finales:")
                        for i, seq in enumerate(seqs, start=1):
                            tail = seq[history.size(1):]
                            end_line = tokenizer.decode(tail, skip_special_tokens=True).split("\n")[0]
                            candidate = ''.join(words) + end_line
                            smin, smax = rango_silabas(candidate)
                            beam_rhyme = rima_asonante(end_line.strip().split()[-1] if end_line.strip() else '')
                            print(f"        Beam {i}: '{end_line.strip()}' [{smin}-{smax} sílabas], rima '{beam_rhyme}'")
                        # Seleccionar primer beam válido
                        for i, seq in enumerate(seqs, start=1):
                            tail = seq[history.size(1):]
                            end_line = tokenizer.decode(tail, skip_special_tokens=True).split("\n")[0]
                            candidate = ''.join(words) + end_line
                            smin, smax = rango_silabas(candidate)
                            if not (smin <= target_syl <= smax):
                                continue
                            last = end_line.strip().split()[-1] if end_line.strip() else ''
                            beam_rhyme = rima_asonante(last)
                            if letter not in rhyme_map:
                                rhyme_map[letter] = beam_rhyme
                            if beam_rhyme == rhyme_map[letter]:
                                print(f"      -> Seleccionado Beam {i}: '{end_line.strip()}' [{smin}-{smax} sílabas], rima '{beam_rhyme}'")
                                words.append(end_line)
                                history = seq.unsqueeze(0)
                                break
                        break

                    # Generar token normal
                    next_token, past_kv = sample_token(history, past_kv, mask, target_syl, current_syl, inte, exc)
                    mask = None
                    history = next_token if past_kv else torch.cat([history, next_token], dim=1)
                    tok = tokenizer.decode(next_token[0])

                    # Actualizar palabras y sílabas
                    if "\n" in tok:
                        tok = tok.replace("\n", "")
                    words.append(tok)
                    text = ''.join(words)
                    min_s, max_s = rango_silabas(text)
                    current_syl = (min_s + max_s) // 2
                    print(f"    - Parcial: '{text}', sílabas aprox. {current_syl}")

                verse = ''.join(words).strip()
                print(f"  Generado: '{verse}'")

                min_s, max_s = rango_silabas(verse)
                metric_ok = (min_s <= target_syl <= max_s)
                rhyme_ok = True
                if letter:
                    last = verse.split()[-1] if verse else ''
                    r = rima_asonante(last)
                    if letter not in rhyme_map:
                        rhyme_map[letter] = r
                        print(f"  [Rima] Asignada letra '{letter}' -> '{r}'")
                    rhyme_ok = (r == rhyme_map[letter])

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
                    if letter and not rhyme_ok and len(words) > 3:
                        print("    - Recorte: quitando últimas 3 palabras, reintentando")
                        words = words[:-3]
                        partial = initial_prompt + ' ' + ''.join(words)
                        history, mask = get_initial_history(partial)
                        smin, smax = rango_silabas(''.join(words))
                        current_syl = (smin + smax) // 2
                        continue

            if not success:
                raise RuntimeError(f"Verso {verse_idx} de estrofa {stanza_idx} no válido")

        poem_lines.append("")
    return "\n".join(poem_lines).strip()

if __name__ == '__main__':
    structure = [[(8, 'A'), (8, None), (8, 'A'), (8, None)]]
    try:
        poem = generate_structured_poem(structure)
    except RuntimeError as e:
        print("Error en generación:", e)
        exit(1)

    print("\n=== POEMA FINAL ===\n", poem)
    with open('poema.txt', 'w', encoding='utf-8') as f:
        f.write(poem)
    print("\n[✓] Poema guardado en 'poema.txt'")
