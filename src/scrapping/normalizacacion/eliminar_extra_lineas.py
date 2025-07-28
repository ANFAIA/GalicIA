import os
import tempfile
import shutil

# Directorio raíz donde están las subcarpetas con los archivos .txt
base_dir = "../data_norm"
# Cadena a filtrar
filtro = "[...]"

for root, dirs, files in os.walk(base_dir):
    for nombre_archivo in files:
        if nombre_archivo.lower().endswith(".txt"):
            ruta_original = os.path.join(root, nombre_archivo)
            # Crear un archivo temporal
            fd, ruta_temp = tempfile.mkstemp(text=True)
            with os.fdopen(fd, 'w', encoding='utf-8') as archivo_temp, \
                 open(ruta_original, 'r', encoding='utf-8') as archivo_orig:
                for linea in archivo_orig:
                    # Escribir solo las líneas que NO contienen la cadena de filtro
                    if filtro not in linea:
                        archivo_temp.write(linea)
            # Reemplazar el archivo original con el filtrado
            shutil.move(ruta_temp, ruta_original)

print("Procesado completado: se han eliminado líneas con la cadena '[...]' en todos los .txt.")
