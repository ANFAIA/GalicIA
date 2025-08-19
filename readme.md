# GalicIA: Preservación y Generación de Poesía Gallega mediante IA

## Resumen Ejecutivo

GalicIA es un proyecto *open source* centrado en la creación de datasets y modelos de lenguaje especializados en poesía gallega, abarcando tanto texto como audio. El proyecto busca preservar y potenciar la rica tradición poética gallega mediante tecnologías de IA.

## Objetivos Principales

1. Crear el mayor dataset público de poesía gallega (texto y audio)  
2. Desarrollar modelos de lenguaje especializados en gallego poético  
3. Implementar herramientas de transcripción audio-texto  
4. Preservar características únicas de la poesía gallega  

## Elementos del proyecto

```text
├── src/                            # Código fuente
│   ├── finetuning/                 # Crea el LLM y hace la inferencia con una base de datos
│   │   ├── dataset/                # Archivos para dataset de poemas
│   │   │   └── crear_dataset.py    # genera datos sintéticos mediante prompts
│   │   ├── entrenamiento/          # Entrena el modelo y adapta con LoRA
│   │   │   └── LoRA/
│   │   │       ├── entreno-base-lingua.py      # preentrena el modelo base en gallego
│   │   │       ├── entreno-poemas.py           # fine-tuning poético con LoRA
│   │   │       └── entreno-poemas-conFIM.py    # fine-tuning con Fill-In-the-Middle (FIM)
│   │   └── inferencia/             # Generación de poemas
│   │       ├── inferencia_estructura.py        # generación métrica (respeta sílabas y rimas)
│   │       └── default_generation.py           # generación básica del modelo LLM
│   ├── extractor-métrica/          # Módulo de métricas
│   │   └── procesar_poema.py       # extrae número de sílabas y rima de cada verso
│   └── scrapping/                  # Obtención de datos de internet
│       ├── cantigas/               # Scrapers para cantigas medievales
│       ├── colección_poesia/       # Scrapers para el blog Colección Poesía Galega
│       ├── galiciana/              # Obtiene URLs de Galiciana
│       ├── wikisource/             # Obtiene URLs de Wikisource
│       └── llm_extract_poem.py     # extrae poemas de HTML con un modelo LLM
├── README.md                       # Este archivo
└── requirements.txt                # Dependencias
```
## Ejecución del proyecto y descripción de los scripts

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Configurar la clave de OpenAI (variable de entorno `OPENAI_API_KEY`).

3. Ejecutar cada script con Python:

| Tarea                                              | Script                                                     | Comando                                                                       |
|----------------------------------------------------|------------------------------------------------------------|-------------------------------------------------------------------------------|
| Extracción de más de 1000 poemas                   | src/scrapping/llm_extract_poem.py                          | python src/scrapping/llm_extract_poem.py                                      |
| Normalizar el texto de los poemas                  | src/scrapping/normalizacacion                              | Para cada script de la carpeta python src/scrapping/normalizacacion/script.py |
| Creación de un modelo base preentrenado en gallego | src/finetuning/entrenamiento/LoRA/entreno-base-lingua.py   | python src/finetuning/entrenamiento/LoRA/entreno-base-lingua.py               |
| Fine-tuning poético con LoRA                       | src/finetuning/entrenamiento/LoRA/entreno-poemas.py        | python src/finetuning/entrenamiento/LoRA/entreno-poemas.py                    |
| Data augmentation con Fill-In-the-Middle (FIM)     | src/finetuning/entrenamiento/LoRA/entreno-poemas-conFIM.py | python src/finetuning/entrenamiento/LoRA/entreno-poemas-conFIM.py             |
| Generación de datos sintéticos mediante prompts    | src/finetuning/dataset/crear_dataset.py                    | python src/finetuning/dataset/crear_dataset.py                                |
| Generación de datos para RL                        | src/finetuning/dataset/crear_dataset_ref.py                | python src/finetuning/dataset/crear_dataset_fef.py                            |
| Módulo de extracción silábica y rima               | src/extractor-métrica/procesar_poema.py                    | from src.extractor-métrica.procesar_poema import rango_silabas, rima_asonante |
| Generación métrica que respeta sílabas y rima      | src/finetuning/inferencia/inferencia_estructura.py         | python src/finetuning/inferencia/inferencia_estructura.py                     |
| Generación de poemas con un modelo LLM básico      | src/finetuning/inferencia/default_generation.py            | python src/finetuning/inferencia/default_generation.py                        |

