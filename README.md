Perfecto.
# Chatbot Normativo FCyT – Baseline 2025

Este proyecto implementa un **chatbot normativo** para la Facultad de Ciencias y Tecnologías (FCyT – UNCA), que permite realizar consultas sobre reglamentos y documentos institucionales a partir de archivos PDF.  

El objetivo de esta versión es proporcionar un **baseline funcional y extensible** para que los estudiantes puedan comprender la arquitectura, ejecutarla localmente y mejorarla en el marco del examen final o hackathon académico.

---

## 🧭 ¿Qué hace este sistema?

El proyecto permite consultar documentos normativos de la FCyT utilizando preguntas en lenguaje natural. Para lograrlo, el sistema:

1. **Carga todos los PDFs** ubicados en la carpeta `docs/`.
2. **Extrae el texto** de cada documento.
3. **Divide el contenido en fragmentos** (chunks) manejables.
4. **Convierte cada fragmento en un vector numérico** mediante la técnica TF-IDF (Term Frequency – Inverse Document Frequency).
5. **Construye un índice de búsqueda local**, sin depender de servicios externos.
6. Cuando el usuario realiza una consulta:
   - La pregunta se vectoriza.
   - Se calcula la similitud entre la pregunta y cada fragmento del corpus.
   - Se devuelven los fragmentos más relevantes, indicando el documento de origen.

Este enfoque garantiza que el sistema:

- **Nunca inventa información**,  
- **Siempre responde con texto real proveniente de los documentos**,  
- **Funciona completamente offline** una vez instalado,  
- Y constituye una base sólida para futuras mejoras en búsqueda semántica, interfaces y asistentes inteligentes.

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

Esto instala:

* fastapi
* uvicorn
* pypdf
* numpy
* scikit-learn
* pydantic

---

## 📚 4. Estructura del proyecto

```
fcyt-chatbot-normativo/
├─ app.py
├─ chatbot.py
├─ procesar_pdfs.py
├─ requirements.txt
├─ docs/                  # PDFs normativos
└─ .gitignore
```

---

## 🏗 5. Procesar los PDFs (generar el índice)

Antes de hacer cualquier consulta, generar el índice TF-IDF:

```bash
python procesar_pdfs.py
```

Esto produce un archivo:

```
indice_tfidf.pkl
```

que contiene:

* fragmentos de texto,
* vectorizador TF-IDF,
* matriz de similitudes.

> Cada vez que se agreguen o cambien PDFs en `docs/`, se debe ejecutar nuevamente este comando.

---

## 💬 6. Uso del chatbot en modo consola

```bash
python chatbot.py
```

Ejemplo de diálogo:

```
=== Chatbot normativo FCyT ===
Pregunta: ¿Cuál es la función del docente de la materia PFG?
```

El sistema devolverá los fragmentos más relevantes y el documento correspondiente.

---

## 🌐 7. Servidor web con FastAPI

Levantar el servidor:

```bash
uvicorn app:app --reload --port 8000
```

Abrir en el navegador:

```
http://127.0.0.1:8000/
```

La interfaz permite:

* ingresar una pregunta,
* enviarla al backend,
* ver los fragmentos recuperados.

Para detener el servidor:
`CTRL + C`

---

## 🧪 8. Objetivo académico del baseline

Este proyecto sirve como punto de partida para que los estudiantes:

* comprendan los conceptos básicos de recuperación de información (IR),
* experimenten con TF-IDF y búsqueda vectorial,
* agreguen nuevos documentos normativos,
* exploren técnicas más avanzadas de extracción,
* mejoren la interfaz de usuario,
* integren modelos locales o remotos para enriquecer las respuestas,
* transformen el prototipo en una herramienta más inteligente y completa.

---

## 🛠 9. Problemas frecuentes y soluciones

* **Error: `indice_tfidf.pkl` no encontrado**
  → Ejecutar `python procesar_pdfs.py`.

* **El sistema no devuelve respuestas útiles**
  → Verificar que los PDFs sean digitales y no escaneados.
  → Regenerar el índice.

* **`uvicorn` no se reconoce**
  → El entorno virtual no está activado.
  → Verificar instalación con `pip install -r requirements.txt`.

---

## 📄 Licencia y uso académico

Este proyecto está diseñado para fines educativos dentro de la FCyT – UNCA.
Puede ser adaptado libremente durante el hackathon o en prácticas de laboratorio.
