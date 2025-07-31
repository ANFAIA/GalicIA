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
├── .idea/                          
├── ejemplos/                       
├── src/                            # Código fuente
│   ├── finetuning/                 # Crea el LLM y hace la inferecia con una base de datos
│   │   ├── dataset/                # Crea el data ser de poemas para el entrenamiento
│   │   ├── entrenamiento/          # Hace el entrenamiento, tanto en LoRA como en otros métodos
│   │   └──inferencia/              # Utiliza diferentes métodos para hacer la inferencia
│   │
│   ├── extractor-métrica/ 
│   │   └── procesar_poema.py       # script de sacar características de verso
│   └── scraping/                   # Saca los datos de intenet
│      
├── README.md                       # Este archivo
└── requirements.txt                # Dependencias
