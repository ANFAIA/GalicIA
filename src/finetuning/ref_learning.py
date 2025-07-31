"""
Entrenamiento por refuerzo (PPO) de un modelo causal‑LM para poesía galega.
-----------------------------------------------------------------------
• Librerías usadas (versiones probadas):
    transformers==4.40.*
    trl==0.6.*            # API estable con `.step()` y `PPOConfig`
    accelerate>=0.26      # gestionado internamente por TRL
• GPU recomendada ≥12 GB; si solo tienes 8 GB usa BATCH_SIZE=2.
• El código entrena línea a línea (verso) pero a partir de
  prompts que describen el poema completo — patrón común en la comunidad.
"""
from __future__ import annotations
import os
import itertools
import logging
from typing import List, Tuple

import torch
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. Compatibilidad TRL -------------------------------------------------
#   TRL<=0.6 espera `top_k_top_p_filtering` en transformers; desde 4.41 cambió de módulo.
try:
    from transformers import top_k_top_p_filtering  # type: ignore
except ImportError:  # ≥4.41.0
    from transformers.generation.utils import top_k_top_p_filtering  # type: ignore
    transformers.top_k_top_p_filtering = top_k_top_p_filtering  # type: ignore

from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead  # noqa: E402

# --- 2. Logging ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# --- 3. Configuración global ----------------------------------------------
BASE_CKPT = "galicIA-v1"
OUTPUT_DIR = "./poema_rl_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS: int = 3
BATCH_SIZE: int = 2        # Baja a 2 si tienes <10 GB de VRAM
MAX_UPDATES: int = 5_000   # Paso PPO = forward+update
MAX_SCHEME_LEN: int = 6
SYLLABLE_OPTS: List[int] = list(range(8, 23, 2))  # 8,10,…,22
STANZA_COUNTS: List[int] = [1, 2]

# --- 4. Funciones métrica y rima ------------------------------------------
from src.extractor_métrica.procesar_poema import rango_silabas, rima_asonante  # type: ignore

def reward_line(line: str, target_syl: int, target_rhyme: str | None) -> float:
    """Reward más alto (≈0) cuanto mejor la métrica y la rima."""
    if not line.strip():
        return -5.0  # línea vacía ⇒ fuerte penalización
    min_s, max_s = rango_silabas(line)
    syl_pen = abs(target_syl - (min_s + max_s) / 2)
    rhyme_pen = 0.0
    if target_rhyme is not None:
        last = line.split()[-1]
        rhyme_pen = 0.0 if rima_asonante(last) == target_rhyme else 1.0
    return - (0.5 * syl_pen + 1.5 * rhyme_pen)

# --- 5. Carga de modelos ---------------------------------------------------
log.info("Cargando modelo base y tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(BASE_CKPT, local_files_only=True)
# Referencia (no necesita GPU)
ref_model = AutoModelForCausalLM.from_pretrained(BASE_CKPT, local_files_only=True)
# Modelo con value‑head; este sí en GPU/CPU
model = AutoModelForCausalLMWithValueHead.from_pretrained(BASE_CKPT, local_files_only=True).to(device)
model.config.pad_token_id = tokenizer.eos_token_id

# --- 6. PPO config ---------------------------------------------------------
ppo_cfg = PPOConfig(
    steps=MAX_UPDATES,
    batch_size=BATCH_SIZE,
    mini_batch_size=1,
    learning_rate=1e-5,
    ppo_epochs=1,
    cliprange=0.2,
    vf_coef=0.1,
    gamma=1.0,
    lam=0.95,
)
trainer = PPOTrainer(
    config=ppo_cfg,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# --- 7. Dataset ------------------------------------------------------------

def gen_schemes(max_len: int) -> List[List[str | None]]:
    schemes: List[List[str | None]] = [[None] * L for L in range(1, max_len + 1)]
    schemes += [
        ['A', 'A'], ['A', 'B'],               # pareados
        ['A', 'B', 'A'],                      # terceto
        ['A', 'B', 'A', 'B'], ['A', 'A', 'B', 'B'], ['A', 'B', 'B', 'A'],
        ['A', 'B', 'A', 'B', 'C', 'C'], ['A', 'A', 'B', 'B', 'C', 'C'],
    ]
    return schemes

schemes = gen_schemes(MAX_SCHEME_LEN)

dataset: list[dict] = []
for stanza_n in STANZA_COUNTS:
    for scheme in schemes:
        for syl_seq in itertools.product(SYLLABLE_OPTS, repeat=len(scheme)):
            base_structure: List[Tuple[int, str | None]] = list(zip(syl_seq, scheme))
            full_structure = base_structure * stanza_n
            prompt_parts = [f"{s}-{r}" if r else f"{s}" for s, r in base_structure]
            prompt = f"Xera un poema con estrutura {' '.join(prompt_parts)} de {stanza_n} estrofas"
            for idx, (syl, rhyme) in enumerate(full_structure):
                dataset.append({
                    "prompt": prompt,
                    "target_syl": syl,
                    "target_rhyme": rhyme,
                    "line_idx": idx,
                })
log.info("Prompts generados: %d", len(dataset))

# --- 8. Entrenamiento ------------------------------------------------------
update_ct = 0
for epoch in range(1, EPOCHS + 1):
    np.random.shuffle(dataset)
    log.info("=== Epoch %d/%d ===", epoch, EPOCHS)
    for start in range(0, len(dataset), BATCH_SIZE):
        if update_ct >= MAX_UPDATES:
            break
        batch = dataset[start:start + BATCH_SIZE]
        prompts = [b['prompt'] for b in batch]
        toks = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
        outs = model.generate(**toks, max_new_tokens=60, pad_token_id=tokenizer.eos_token_id)
        responses = [tokenizer.decode(o[len(inp):], skip_special_tokens=True) for o, inp in zip(outs, toks['input_ids'])]
        rewards: List[float] = []
        for resp, meta in zip(responses, batch):
            lines = [l for l in resp.splitlines() if l.strip()]
            line = lines[meta['line_idx']] if meta['line_idx'] < len(lines) else ""
            rewards.append(reward_line(line, meta['target_syl'], meta['target_rhyme']))
        stats = trainer.step(prompts, responses, rewards)
        update_ct += 1
        if update_ct % 50 == 0:
            log.info("Step %d | mean_r %.3f | kl %.4f", update_ct, np.mean(rewards), stats.get('ppo/approx_kl', 0))
    if update_ct >= MAX_UPDATES:
        log.warning("Máximo de actualizaciones alcanzado (%d)", MAX_UPDATES)
        break

# --- 9. Guardado -----------------------------------------------------------
save_dir = os.path.join(OUTPUT_DIR, 'rl_poema_model')
os.makedirs(save_dir, exist_ok=True)
trainer.model.save_pretrained(save_dir)
trainer.tokenizer.save_pretrained(save_dir)
log.info("Modelo RL guardado en %s", save_dir)

# --- 10. Inferencia helper -------------------------------------------------

def infer(prompt: str, ckpt: str = save_dir, max_new: int = 120) -> str:
    tk = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    mdl = AutoModelForCausalLM.from_pretrained(ckpt, local_files_only=True).to(device)
    inps = tk(prompt, return_tensors='pt').to(device)
    gen = mdl.generate(**inps, max_new_tokens=max_new, pad_token_id=tk.eos_token_id)
    return tk.decode(gen[0][inps['input_ids'].shape[-1]:], skip_special_tokens=True)

if __name__ == "__main__":
    demo_prompt = "Xera un poema con estrutura 8-A 8-A 8-B 8-B de 2 estrofas"
    print(infer(demo_prompt))
