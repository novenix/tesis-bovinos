Pipeline de Etiquetado Automatizado para Ganado Bovino
1. Contexto del Dataset y Escala
Actualmente se cuenta con un volumen masivo de datos ya curados que alcanza las 173,956 imágenes (aproximadamente 160 GB), extraídas a 2 frames por segundo de archivos .dav y alojadas en /dataset_curado/train/images. Debido a esta escala masiva, la anotación manual tradicional (dibujar cajas delimitadoras o máscaras de segmentación píxel por píxel) es computacional, temporal y económicamente inviable. El enfoque requiere un pipeline de "Motor de Datos" (Data Engine) automatizado.

2. Estrategia Base: Modelos Fundacionales de Visión (SAM 3)
Para la generación rápida de la verdad fundamental (ground truth) de los bovinos, la herramienta óptima del estado del arte son los Modelos Fundacionales de Visión, específicamente Segment Anything 3 (SAM 3).

SAM 3 actúa como un modelo de Segmentación Conceptual Prompteable (Promptable Concept Segmentation) capaz de funcionar en un entorno zero-shot. El flujo de trabajo para el etiquetado se define así:   

Auto-etiquetado por Lotes (Zero-Shot): Se inyectan las imágenes al modelo SAM 3 utilizando prompts (indicaciones) de texto en lenguaje natural, como "cow", "cattle" o partes específicas si se requiere ("cow head", "cow leg"). El modelo rastrea la semántica del texto y proyecta automáticamente máscaras de segmentación y cajas delimitadoras precisas sobre cada animal que coincida con el texto.   

Revisión Human-in-the-Loop: La tarea humana se desplaza de la creación manual a la simple supervisión. Los investigadores o el equipo de anotación revisan las propuestas generadas por la IA, aceptando las correctas y corrigiendo únicamente los casos difíciles (por ejemplo, oclusiones severas o mala iluminación).   

3. Mitigación de Falsos Positivos: El Problema de los Humanos
Como se evidenció en los procesos de curaduría inteligente (específicamente en el Test 5.2 con YOLO-World v2), los modelos de vocabulario abierto tienden a clasificar a las personas (operarios del corral) como "vacas" si se les restringe a buscar únicamente ganado. Para resolver esto de raíz desde el etiquetado hasta el entrenamiento final, se deben seguir dos directrices:

A. Prompting Negativo y Clases Explícitas
En la fase de auto-etiquetado o filtrado de frames, es fundamental incluir la clase "person" de forma explícita en el vocabulario del modelo. Al brindar esta categoría, la IA prefiere asociar las características del humano a "person" con una confianza alta. El algoritmo de curaduría debe estar programado para que, si detecta la clase "person" con una confianza mayor a 0.5 (como en el script 5.2), esa detección se ignore o el frame se descarte si el humano es el objeto principal, evitando el ruido en las etiquetas finales.

B. Uso de Imágenes de Fondo (Background Images)
Para garantizar que el modelo definitivo (el que se entrenará posteriormente usando estas etiquetas) no detecte humanos ni objetos estáticos del corral como vacas, se debe incluir explícitamente entre un 5% y un 10% de "imágenes de fondo" en el dataset de entrenamiento. Estas son imágenes que no contienen vacas (por ejemplo, fotografías donde solo salen personas trabajando, o el corral completamente vacío) y se introducen en el dataset sin ningún tipo de etiqueta. Esta técnica obliga a la red neuronal a aprender qué es el fondo y a reducir activamente los falsos positivos en entornos de producción.   

