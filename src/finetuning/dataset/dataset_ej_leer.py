from datasets import load_dataset, load_from_disk

# 1. Cargamos el dataset
#non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split="train")
ds1 = load_dataset("pajon1/poemas_galicIA", split="train")    # Reading comprehension :contentReference[oaicite:3]{index=3}
#ds2 = load_dataset("proxectonos/galcola", split="train")        # Linguistic acceptability :contentReference[oaicite:4]{index=4}
#ds3 = load_dataset("proxectonos/mgsm_gl", split="train")        # Math QA :contentReference[oaicite:5]{index=5}
print(ds1[0])
#print(ds2[0])
#print(ds3[0])

# 2. Guardamos en disco (se crea la carpeta "FineTome-100k-train")
ds1.save_to_disk("asdfasdfasdf")

# 3. (Opcional) Para recargarlo más adelante:
#reloaded_dataset = load_from_disk("FineTome-100k-train")
#print(dataset)


#from unsloth.chat_templates import CHAT_TEMPLATES
#print(list(CHAT_TEMPLATES.keys()))