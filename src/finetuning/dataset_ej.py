from datasets import load_dataset, load_from_disk

# 1. Cargamos el dataset
#non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset=load_from_disk("poemas_GalicIA")

# 2. Guardamos en disco (se crea la carpeta "FineTome-100k-train")
#dataset.save_to_disk("FineTome-100k-train")

# 3. (Opcional) Para recargarlo más adelante:
#reloaded_dataset = load_from_disk("FineTome-100k-train")
print(dataset)
print(dataset[0])

from unsloth.chat_templates import CHAT_TEMPLATES
print(list(CHAT_TEMPLATES.keys()))