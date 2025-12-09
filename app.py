import pickle
import numpy as np
import re
import os
import io
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import pypdf

# Nombre del archivo del índice
INDICE_PATH = "indice_tfidf.pkl"

# Variables globales para el índice (se cargarán/actualizarán)
documentos = []
vectorizer = None
embeddings = None
index_loaded = False


# ==== FUNCIONES DE PROCESAMIENTO DE PDF (Tomadas de procesar_pdf.py) ====

def extraer_texto(pdf_file: io.BytesIO, filename: str):
    """Extrae texto de un PDF en memoria página por página."""
    try:
        reader = pypdf.PdfReader(pdf_file)
        texto = ""
        for page in reader.pages:
            try:
                texto += page.extract_text() + "\n"
            except Exception:
                pass
        return texto
    except Exception as e:
        print(f"Error al leer PDF {filename}: {e}")
        return ""


def dividir_en_chunks(texto, max_chars=500):
    """Divide texto largo en bloques (~500 caracteres aprox)."""
    palabras = texto.split()
    chunks = []
    actual = []
    count = 0

    for p in palabras:
        actual.append(p)
        count += len(p) + 1
        if count > max_chars:
            chunks.append(" ".join(actual))
            actual = []
            count = 0

    if actual:
        chunks.append(" ".join(actual))

    return chunks


def generar_y_guardar_indice(archivos_subidos: list[UploadFile]):
    """Procesa los archivos subidos, genera el nuevo índice TF-IDF y lo guarda."""
    global documentos, vectorizer, embeddings, index_loaded
    
    documentos_nuevos = []
    print("\n--- INICIANDO GENERACIÓN DE ÍNDICE ---")
    
    # 1. Extracción y Fragmentación
    for archivo in archivos_subidos:
        print(f"Procesando: {archivo.filename}")
        # Leer el contenido del archivo en memoria
        content = archivo.file.read()
        pdf_file = io.BytesIO(content)
        
        texto = extraer_texto(pdf_file, archivo.filename)
        chunks = dividir_en_chunks(texto)

        for chunk in chunks:
            documentos_nuevos.append({"texto": chunk, "fuente": archivo.filename})
            
    if not documentos_nuevos:
        print("No se extrajo texto de los documentos subidos. Índice no actualizado.")
        # Retornar False si no se pudo generar el índice
        return False

    print(f"Total de fragmentos: {len(documentos_nuevos)}")
    texts = [d["texto"] for d in documentos_nuevos]

    # 2. Crear Matriz TF-IDF
    print("Generando matriz TF-IDF...")
    vectorizer_nuevo = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words=None
    )

    X = vectorizer_nuevo.fit_transform(texts)
    embeddings_nuevos = X.toarray().astype("float32")
    
    print("Dimensión del espacio vectorial:", embeddings_nuevos.shape[1])

    # 3. Guardar y Actualizar Globales
    print("Guardando índice...")
    data = {
        "documentos": documentos_nuevos,
        "vectorizer": vectorizer_nuevo,
        "embeddings": embeddings_nuevos,
    }

    with open(INDICE_PATH, "wb") as f:
        pickle.dump(data, f)
        
    # Actualizar las variables globales para el chatbot
    documentos = documentos_nuevos
    vectorizer = vectorizer_nuevo
    embeddings = embeddings_nuevos
    index_loaded = True
    
    print("✔ ¡Índice generado y cargado correctamente!")
    return True

# ==== CARGA INICIAL DEL ÍNDICE ====

def cargar_indice_inicial():
    """Carga el índice TF-IDF al iniciar la app si existe."""
    global documentos, vectorizer, embeddings, index_loaded
    if Path(INDICE_PATH).exists():
        print(f"Cargando índice desde {INDICE_PATH}...")
        try:
            with open(INDICE_PATH, "rb") as f:
                data = pickle.load(f)
            documentos = data["documentos"]
            vectorizer = data["vectorizer"]
            embeddings = data["embeddings"]
            index_loaded = True
            print(f"Índice cargado con {len(documentos)} fragmentos.")
        except Exception as e:
            print(f"Error al cargar el índice: {e}. Inicie la app sin índice.")
    else:
        print("Índice no encontrado. Por favor, sube PDFs para crearlo.")

cargar_indice_inicial()

# ==== LÓGICA DE BÚSQUEDA (sin cambios) ====

def buscar_respuesta(pregunta: str, k: int = 3):
    """Devuelve los k fragmentos más similares a la pregunta."""
    # ... (El código de la función buscar_respuesta es el mismo que tienes) ...
    
    if not index_loaded:
        return [{"texto": "El índice de documentos no está cargado. Por favor, sube un PDF para generar el índice.", "fuente": "Sistema", "score": 0.0}]

    q_vec = vectorizer.transform([pregunta]).toarray().astype("float32")
    sims = cosine_similarity(q_vec, embeddings)[0]

    idxs = np.argsort(sims)[::-1][:k]

    resultados = []
    for idx in idxs:
        doc = documentos[idx]
        texto = doc.get("texto", "")

        # Dividir en oraciones (también separa por saltos de línea)
        oraciones = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', texto) if s.strip()]

        snippet = texto  # fallback: todo el texto si algo falla
        snippet_score = float(sims[idx])

        if oraciones:
            try:
                sent_vecs = vectorizer.transform(oraciones).toarray().astype("float32")
                sent_sims = cosine_similarity(q_vec, sent_vecs)[0]
                best_i = int(np.argmax(sent_sims))
                best_sentence = oraciones[best_i].strip()

                # Añadir la siguiente oración como contexto si la frase es muy corta
                if len(best_sentence) < 100 and best_i + 1 < len(oraciones):
                    best_sentence = best_sentence + " " + oraciones[best_i + 1].strip()

                snippet = best_sentence
                snippet_score = float(sent_sims[best_i])
            except Exception:
                snippet = texto
                snippet_score = float(sims[idx])

        resultados.append(
            {
                "score": snippet_score,
                "texto": snippet,
                "fuente": doc.get("fuente", ""),
            }
        )

    # Ordenar los resultados por score descendente (score más alto primero)
    resultados.sort(key=lambda r: r["score"], reverse=True)

    return resultados


# ==== MODELOS DE ENTRADA / SALIDA (sin cambios) ====

class Question(BaseModel):
    question: str


# ==== RUTAS ====

app = FastAPI(title="Chatbot normativo FCyT")


@app.get("/", response_class=HTMLResponse)
def home():
    # Servir el HTML desde templates/index.html
    path = Path(__file__).parent / "templates" / "index.html"
    if path.exists():
        return HTMLResponse(content=path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<p>Index not found</p>", status_code=404)


@app.post("/ask")
def ask(q: Question):
    resultados = buscar_respuesta(q.question, k=3)
    return {"resultados": resultados}


@app.post("/upload-pdfs")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """Ruta para subir archivos PDF y regenerar el índice."""
    if not files:
        raise HTTPException(status_code=400, detail="No se subió ningún archivo.")

    # Filtrar solo PDFs
    pdfs = [f for f in files if f.filename and f.filename.lower().endswith(".pdf")]
    
    if not pdfs:
        raise HTTPException(status_code=400, detail="No se subió ningún archivo PDF válido.")

    try:
        # Esto procesa los archivos subidos, guarda el nuevo índice
        # y actualiza las variables globales (documentos, vectorizer, embeddings)
        if generar_y_guardar_indice(pdfs):
            return JSONResponse(
                content={"message": f"PDFs subidos"},
                status_code=200
            )
        else:
            raise HTTPException(status_code=500, detail="Los archivos se subieron, pero falló la generación del índice (posiblemente por no extraer texto).")
            
    except Exception as e:
        # Asegúrate de capturar cualquier error durante el procesamiento
        print(f"Error en la subida y procesamiento: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno al procesar los archivos: {str(e)}")