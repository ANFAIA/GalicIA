import os
import configparser
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import torch
import multiprocessing as mp
N_CORES = mp.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
torch.set_num_threads(N_CORES)          # intra-op :contentReference[oaicite:4]{index=4}
torch.set_num_interop_threads(N_CORES)

# Leer la configuración desde config.ini
config = configparser.ConfigParser()
config.read("config.ini")





def download_sentence_transformer(model_name, save_dir):
    """
    Descarga y guarda un modelo de SentenceTransformer en la carpeta especificada.
    """
    print(f"Descargando SentenceTransformer: {model_name}")
    try:
        model = SentenceTransformer(model_name, cache_folder=save_dir)
        # Guardar el modelo en la carpeta especificada
        model.save(save_dir)
        print(f"Modelo guardado en {save_dir}")
    except Exception as e:
        print(f"Error al descargar el modelo {model_name}: {e}")

def download_llm(model_name: str, save_dir: str):
    """
    Descarga y guarda un modelo de lenguaje (LLM) y su tokenizer en la carpeta especificada
    sin cargar el modelo en memoria para reducir el uso de RAM.

    :param model_name: Nombre del modelo en Hugging Face Hub.
    :param save_dir: Directorio donde se guardará el modelo y el tokenizer.
    """
    print(f"Descargando LLM: {model_name}")
    try:
        # Descargar el modelo y el tokenizer usando snapshot_download
        snapshot_download(repo_id=model_name, cache_dir=save_dir, local_dir=save_dir, local_dir_use_symlinks=False)
        print(f"LLM y tokenizer guardados en {save_dir}")
    except Exception as e:
        print(f"Error al descargar el modelo LLM {model_name}: {e}")

def innit_model():
    base_dir = config["paths"]["base_dir"]
    # Crear el directorio base si no existe
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Directorio base creado en {base_dir}")
    else:
        print(f"El directorio base ya existe en {base_dir}")

    # Descargar el modelo LLM

    llm_model = config['model']['llm_token']
    llm_subdir = llm_model.replace('/', '_')
    llm_dir = os.path.join(base_dir, "llm", llm_subdir)
    if not os.path.exists(llm_dir):
        os.makedirs(llm_dir)
        download_llm(llm_model, llm_dir)
    else:
        print(f"El modelo LLM ya existe en {llm_dir}")

    #quantize_llm(llm_dir)


if __name__ == "__main__":
    innit_model()
