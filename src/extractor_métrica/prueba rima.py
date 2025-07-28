import re
import unicodedata
import pyphen

dic = pyphen.Pyphen(lang='es')

def normaliza(pal):
    pal = pal.lower()
    pal = ''.join(c for c in unicodedata.normalize('NFD', pal)
                  if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-zñü]', '', pal)

def acento(pal):
    # posición de la sílaba tónica (simplificado)
    if re.search(r'[áéíóú]', pal):
        return max(i for i, c in enumerate(pal) if c in "áéíóú")  # índice de vocal tónica
    # agudas/graves/esdrújulas → aquí simplificamos a 'grave'
    # lo ideal es silabificar y aplicar reglas
    return len(pal.rstrip('ns')) - 2  # aproximado

def cola_asonante(pal):
    p = normaliza(pal)
    idx = acento(p)
    return ''.join(v for v in p[idx:] if v in 'aeiou')  # solo vocales

print(cola_asonante('igual'), cola_asonante('despertar'))  # 'a' 'a'  -> riman
print(cola_asonante('rienda'),  cola_asonante('puedas'))    # 'io' 'io' -> riman
