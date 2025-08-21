import os
import socket
import subprocess
import sys
import configparser
import pathlib
import logging
import json
logging.getLogger("vllm").setLevel(logging.ERROR)
config = configparser.ConfigParser()


def is_port_in_use(port: int, host: str = "localhost") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        try:
            sock.connect((host, port))
            return True
        except (ConnectionRefusedError, socket.timeout):
            return False
import ctypes, signal

libc = ctypes.CDLL("libc.so.6")

def set_pdeathsig(sig=signal.SIGTERM):
    return lambda: libc.prctl(1, sig)

def init():
    """
    Lanza el servidor vLLM con el modelo cuantizado ya existente.
    """
    #env = os.environ.copy()
    config.read('config.ini')
    port = int(config['model']['llm_port'])
    base_dir = config["paths"]["base_dir"]
    llm_model = config['model']['llm_rag']
    llm_subdir = llm_model.replace('/', '_')
    llm_dir = os.path.join(base_dir, "llm", llm_subdir)

    if not os.path.isdir(llm_dir):
        raise FileNotFoundError(
            f"'{llm_dir}' no existe. Ejecuta primero quantize_llm().")

    if is_port_in_use(port):
        print(f"⚠️  Puerto {port} ocupado; la API parece activa.")
        return

    print(f"🎯 Iniciando API en puerto {port} usando '{llm_dir}'…")

    # --- 1. FLASHINFER 0.2.5 + ENV VAR ------------------------------------------
    env = os.environ.copy()
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "1"  # fuerza el sampler rápido

    # --- 2. KERNEL-TUNING · FP8 W8A8 --------------------------------------------
    site_pkgs = pathlib.Path(sys.executable).parent.parent / "lib/python3.12/site-packages"
    config_dir = site_pkgs / "vllm/model_executor/layers/quantization/utils/configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    # --- 3. ARRANQUE DE vLLM -----------------------------------------------------
    subprocess.Popen([
        "vllm", "serve", llm_model,
        #"--quantization", "fp8",
        #"--tensor-parallel-size", "4",
        #"--enable-expert-parallel",
        "--dtype", "auto",  # elige BF16/FP16 donde convenga
        "--max-model-len", "28000",  # ventana razonable para 24 GB
        "--gpu-memory-utilization", "0.96",
        #"--cpu-offload-gb", "2",
        #"--max-num-batched-tokens", "2048",
        #"--cpu-offload-gb", "10",
        "--enable-prefix-caching",
        #"--enable-reasoning", "--reasoning-parser", "deepseek_r1",
        "--port", str(port),
    ], env=env)

    print(f"Servidor OpenAI listo en http://localhost:{port}/v1 …")




# ────────────────────────────
# 3) Ejecución manual (ejemplo)
# ────────────────────────────
if __name__ == "__main__":
    # Llama explícitamente a la fase que necesites:
    # quantize_llm()   # ← solo la primera vez
    init()       # ← cada vez que quieras levantar la API
