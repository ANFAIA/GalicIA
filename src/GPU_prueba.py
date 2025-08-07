# fichero: check_attention_impl.py
import torch

print(f"Torch   : {torch.__version__}")
print(f"CUDA    available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA SM? {torch.cuda.get_device_name(0)}")
print()

# 1) ¿xFormers?
try:
    import xformers.ops as xops
    print("xFormers: INSTALADO")
    print("  • memory_efficient_attention disponible?", hasattr(xops, "memory_efficient_attention"))
except ImportError:
    print("xFormers: NO instalado")

# 2) ¿flash-attn?
try:
    import flash_attn
    print(f"flash-attn: INSTALADO (versión {flash_attn.__version__})")
except ImportError:
    print("flash-attn: NO instalado")

print()

# 3) ¿Qué kernels de atención tiene PyTorch registrados?
dispatch_keys = ["Meta", "Triton", "CUDA", "CPU"]
# probamos dos posibles nombres de operador:
op_names = ["scaled_dot_product_attention", "aten::scaled_dot_product_attention"]

print("Comprobando kernels de scaled_dot_product_attention:")
for key in dispatch_keys:
    has_kernel = False
    for op in op_names:
        try:
            has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key(op, key)
            if has_kernel:
                break
        except RuntimeError:
            # el nombre de op no existe, seguimos con el siguiente
            continue
    print(f" Dispatch key '{key}': {has_kernel}")
