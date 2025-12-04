Perfecto, acá tenés un `README.md` listo para pegar y commitear.

````markdown
# Chatbot Normativo FCyT – Baseline 2025

Este proyecto implementa un **chatbot normativo** para la Facultad de Ciencias y Tecnologías (FCyT – UNCA), que permite hacer consultas sobre distintos reglamentos y documentos institucionales a partir de un corpus de archivos PDF.

La versión actual es un **baseline**: ya incluye extracción de texto, fragmentación, construcción de un índice TF-IDF, backend web con FastAPI e interfaz mínima en el navegador.  
El objetivo en el examen / hackathon es **mejorar y extender** este baseline, no reconstruirlo desde cero.

---

## 1. Requisitos

- **Python 3.11** (recomendado)
- Conexión a internet solo para la **primera instalación de dependencias** (`pip install`)

Los comandos que se muestran a continuación asumen un entorno Windows PowerShell / Git Bash.  
Adaptar según sistema operativo si fuera necesario.

---

## 2. Clonado del repositorio

```bash
git clone https://github.com/hectorpyco/fcyt-chatbot-normativo.git
cd fcyt-chatbot-normativo
````

---

## 3. Creación y activación del entorno virtual

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> Si PowerShell bloquea la ejecución de scripts, usar:
>
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```

### Linux / macOS (bash/zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 4. Instalación de dependencias

Con el entorno virtual activado:

```bash
pip install -r requirements.txt
```

Esto instalará:

* `fastapi`
* `uvicorn`
* `pypdf`
* `numpy`
* `scikit-learn`
* `pydantic`

---

## 5. Estructura del proyecto

```text
fcyt-chatbot-normativo/
├─ app.py                # Backend FastAPI + interfaz web mínima
├─ chatbot.py            # Versión modo consola (opcional)
├─ procesar_pdfs.py      # Script para procesar PDFs y construir el índice TF-IDF
├─ requirements.txt      # Dependencias del proyecto
├─ docs/                 # Carpeta con los PDFs normativos (corpus)
└─ .gitignore
```

La carpeta `docs/` contiene los PDFs digitales provistos (reglamentos, planes, ley, etc.).
Solo se procesan PDFs **legibles** (con texto), no escaneados puros.

---

## 6. Generación del índice (procesar los PDFs)

Antes de usar el chatbot por primera vez, es necesario construir el índice TF-IDF a partir de los PDFs de `docs/`.

Con el entorno virtual activado:

```bash
python procesar_pdfs.py
```

Si todo sale bien, verás algo como:

```text
Buscando PDFs en la carpeta: docs

Procesando PDF: ...
Procesando PDF: ...

Total de fragmentos: XXX

Generando matriz TF-IDF...
Dimensión del espacio vectorial: YYYY

Guardando índice...

✔ ¡Proceso completado!
  - indice_tfidf.pkl generado
Listo para usar con el chatbot.
```

Esto genera el archivo:

* `indice_tfidf.pkl` → contiene:

  * los fragmentos de texto,
  * el vectorizador TF-IDF entrenado,
  * la matriz de embeddings TF-IDF.

> Cada vez que se agreguen o cambien PDFs en `docs/`, se debe volver a ejecutar
> `python procesar_pdfs.py` para regenerar el índice.

---

## 7. Uso del chatbot en modo consola (opcional)

Para probar la lógica de búsqueda sin interfaz web:

```bash
python chatbot.py
```

Verás:

```text
=== Chatbot normativo FCyT (modo consola) ===
Escribe tu pregunta sobre reglamentos. Escribe 'salir' para terminar.
```

Ejemplos de preguntas:

* `¿Cuál es la función del docente de la materia PFG?`
* `¿Cuál es el plazo máximo para concluir la carrera?`
* `¿Qué establece el reglamento de investigación sobre los proyectos?`

El programa devolverá los fragmentos más relevantes, indicando el PDF fuente y un score de similitud.

---

## 8. Uso del chatbot vía web (FastAPI + navegador)

Para levantar el servidor web:

```bash
uvicorn app:app --reload --port 8000
```

Luego, abrir en el navegador:

```text
http://127.0.0.1:8000/
```

La interfaz permite:

1. Escribir una pregunta en un cuadro de texto.
2. Pulsar “Consultar”.
3. Ver una lista de fragmentos relevantes con:

   * nombre del documento,
   * score de similitud,
   * porción del texto encontrado.

Para detener el servidor, usar `CTRL + C` en la terminal donde corre `uvicorn`.

---

## 9. Notas para el examen / hackathon

* Este repositorio constituye el **baseline oficial**:
  ya están implementados el procesamiento de PDFs, el índice TF-IDF y la API `/ask`.

* El trabajo de los equipos consistirá en:

  * mejorar la relevancia de las respuestas,
  * organizar mejor las normativas (por tipo de documento),
  * mejorar la interfaz,
  * y/o incorporar técnicas adicionales (embeddings, re-ranking, generación de respuestas, etc.).

* No se espera que los estudiantes reescriban desde cero el pipeline básico, sino que **lo entiendan, lo usen y lo extiendan**.

---

## 10. Problemas frecuentes

* **Error: índice no encontrado (`indice_tfidf.pkl`)**
  → Ejecutar antes: `python procesar_pdfs.py`

* **El servidor levanta, pero no hay resultados útiles**
  → Verificar que `docs/` contiene PDFs legibles y que el índice se generó sin errores.

* **No reconoce el comando `uvicorn`**
  → Asegurarse de haber activado el entorno virtual y corrido `pip install -r requirements.txt`.

---

Cualquier mejora o extensión debe respetar este flujo básico:

> PDFs → extracción de texto → fragmentación → índice → API `/ask` → interfaz de usuario.

