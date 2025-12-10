
# Chatbot Normativo FCyT – Baseline 2025

Este proyecto implementa un **chatbot normativo** para la Facultad de Ciencias y Tecnologías (FCyT – UNCA), que permite realizar consultas sobre reglamentos y documentos institucionales a partir de archivos PDF.  

El objetivo de esta versión es proporcionar un **baseline funcional y extensible** para que los estudiantes puedan comprender la arquitectura, ejecutarla localmente y mejorarla en el marco del examen final o hackathon académico.

---

## 🧭 ¿Qué hace este sistema?

Este proyecto permite realizar búsquedas inteligentes dentro de los documentos normativos de la FCyT utilizando preguntas en lenguaje natural. Para ello, el sistema realiza los siguientes pasos:

1. **Carga de documentos:** El sistema **espera** a que el usuario proporcione los archivos PDF.
2. **Extracción de texto:** Se extrae el texto completo de cada PDF cargado.
3. **Fragmentación:** El contenido se divide en fragmentos (chunks) para facilitar la indexación y la recuperación.
4. **Indexación mixta (híbrida):**

   * Se mantiene una representación basada en **TF-IDF** (útil para coincidencias exactas y búsquedas por términos/claves).
   * Paralelamente, se generan **embeddings densos** para cada fragmento usando un modelo preentrenado (`paraphrase-multilingual-MiniLM-L12-v2`).
   * Ambos tipos de representaciones conforman un índice local híbrido, **sin depender de servicios externos**.
5. **Búsqueda y recuperación:** Cuando llega una consulta:

   * La pregunta se transforma tanto a TF-IDF como a embedding.
   * Se calculan **scores TF-IDF** y **scores densos (embeddings)** por similitud (p. ej. coseno).
   * Se combina un score híbrido que pondera TF-IDF y embeddings según el tipo de consulta (p. ej. más peso a TF-IDF para definiciones o búsquedas de palabras clave, más peso a embeddings para búsquedas semánticas generales).
   * Se selecciona un Top-K inicial según ese score combinado y se aplican reglas adicionales por tipo de contenido:

     * **Definiciones:** devolver el bloque o párrafo completo (máximo contexto).
     * **Procedimientos:** expandir con oraciones contiguas relevantes (más contexto operativo).
     * **Búsqueda general:** seleccionar el párrafo más relevante y, si es necesario, recortar para limitar longitud.
   * Además se aplican *boosts* basados en metadatos del documento (tipo de documento, etiquetas, prioridad institucional, etc.) antes de ordenar los candidatos finales.
6. **Respuesta:** El sistema devuelve fragmentos textuales extraídos del corpus, indicando la fuente y metadatos asociados.

### Garantías y límites

* **No inventa información:** todas las respuestas provienen directamente del texto de los documentos cargados.
* **Offline:** funciona localmente una vez instalados los modelos y dependencias.
* **Extensible:** arquitectura pensada como baseline para mejorar la recuperación semántica, ajustar ponderaciones, añadir UI o servicios de QA más avanzados.

---

## 🧩 Requisitos

### ✔ Python 3.11 (recomendado)

Descarga oficial:
- Windows 64-bit:  
  https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

Página oficial:  
https://www.python.org/downloads/release/python-3119/

> Importante: durante la instalación, marcar **“Add Python to PATH”**.

### ✔ Conexión a internet  
Solo necesaria para instalar dependencias la primera vez.

---

## 📥 1. Clonar el repositorio

```bash
git clone https://github.com/hectorpyco/fcyt-chatbot-normativo.git
cd fcyt-chatbot-normativo
````

---

## 🐍 2. Crear y activar el entorno virtual

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si aparece un error de permisos:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```
---
## 📦 3. Instalar dependencias

```bash
pip install -r requirements.txt
```
Esto instala los requerimientos detallados dentro del archivo

---

## 📚 4. Estructura del proyecto

```
fcyt-chatbot-normativo/
├─ app.py
├─ requirements.txt
├─ templates/
└─ .gitignore
```

---

## 🌐 5. Servidor web, carga de documentos y uso del chatbot

Para iniciar el sistema, primero se debe levantar el servidor web con FastAPI:

```bash
uvicorn app:app --reload --port 8000
```

Luego abrir en el navegador:

```
http://127.0.0.1:8000/
```

Desde esta interfaz web se realizan **todas las operaciones principales del sistema**, tanto la administración de documentos como el uso del chatbot.

---

### 📥 5.1 Carga de documentos y generación del índice

Antes de realizar consultas, el usuario debe cargar los PDF normativos.
Esto se hace desde la sección **“Manage PDFs”** disponible en la interfaz.

El flujo es el siguiente:

1. **Subir archivos PDF:**
   El usuario selecciona uno o varios archivos.
   Al procesarse, el sistema:

   * extrae el texto,
   * fragmenta el contenido,
   * genera embeddings densos,
   * calcula representaciones TF-IDF,
   * y finalmente construye un **índice híbrido**.

   Este índice se guarda en el archivo:
   **`indice_tfidf.pkl`**

2. **Visualización del índice:**
   La interfaz muestra la lista de documentos cargados con:

   * nombre del archivo,
   * tamaño,
   * estado en el índice,
   * opción para eliminarlos individualmente.

3. **Limpieza del índice:**
   Existe un botón para borrar todo el índice y comenzar desde cero.

---

### 💬 5.2 Uso del chatbot desde la interfaz

Una vez generado el índice, se puede acceder a la sección principal del chatbot.

En este apartado, el usuario puede:

1. **Ingresar una pregunta en lenguaje natural**,
2. **Enviar la consulta**,
3. **Recibir el resultado del sistema**, que incluye:

   * el **fragmento más relevante**,
   * el **documento de origen**,
   * el **score o confianza** de la coincidencia,
   * y los metadatos relevantes.

Las respuestas provienen **exclusivamente del contenido de los PDFs cargados**, garantizando fidelidad normativa.

---

Para detener el servidor:
`CTRL + C`

---


## 🧪 6. Objetivo académico del baseline

Este proyecto funciona como una base práctica para que los estudiantes:

* comprendan cómo funciona un **sistema de búsqueda híbrido** que combina TF-IDF y embeddings semánticos,
* experimenten con técnicas de recuperación de información (IR) aplicadas a documentos normativos reales,
* practiquen la **carga, indexación y administración** de documentos desde una interfaz web,
* entiendan cómo se construyen índices locales sin depender de servicios externos,
* modifiquen la lógica de **clasificación de preguntas**, ponderación de scores y estrategias diferenciadas (definiciones, procedimientos, búsquedas generales),
* mejoren la interfaz del chatbot y la gestión de documentos,
* incorporen nuevos modelos de embeddings o integrar modelos externos/locales para extender las capacidades del sistema,
* optimicen la calidad de las respuestas, agreguen visualizaciones o creen nuevas funcionalidades para el examen final o hackathon académico.

---

## 📄 Licencia y uso académico

Este proyecto está diseñado para fines educativos dentro de la FCyT – UNCA.
Puede ser adaptado libremente durante el hackathon o en prácticas de laboratorio.
