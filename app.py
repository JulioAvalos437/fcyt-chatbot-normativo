import numpy as np
import re
import io
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import pypdf

# Nombre de los archivos del índice
INDICE_PATH = "indice_tfidf.pkl" 

# Variables globales para el índice
documentos_activos = []
vectorizer = None
embeddings = None
index_loaded = False

app = FastAPI(title="Chatbot normativo FCyT")

# ==== FUNCIONES DE PROCESAMIENTO DE PDF ====
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

# ==== GESTIÓN DEL ÍNDICE ====

def generar_y_cargar_indice_en_memoria(current_documents: list):
    """Genera el nuevo índice TF-IDF basado en la lista de documentos y LO CARGA EN MEMORIA."""
    global documentos_activos, vectorizer, embeddings, index_loaded
    
    documentos_activos = current_documents

    if not documentos_activos:
        print("La lista de documentos está vacía. Índice no generado.")
        vectorizer = None
        embeddings = None
        index_loaded = False
        return True

    print(f"\nGenerando índice con {len(documentos_activos)} fragmentos...")
    
    texts = [d["texto"] for d in documentos_activos]

    vectorizer_nuevo = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words=None
    )

    X = vectorizer_nuevo.fit_transform(texts)
    embeddings_nuevos = X.toarray().astype("float32")
    
    vectorizer = vectorizer_nuevo
    embeddings = embeddings_nuevos
    index_loaded = True
    
    print("✔ Índice generado y cargado EN MEMORIA correctamente!")
    return True

# ==== LÓGICA DE BÚSQUEDA ====
def buscar_respuesta(pregunta: str, k: int = 3):
    """Devuelve los k fragmentos más similares a la pregunta."""
    if not index_loaded:
        return [{"texto": "El índice no está cargado. Por favor, carga un PDF en Gestionar PDFs.", "fuente": "Sistema", "score": 0.0}]
    
    q_vec = vectorizer.transform([pregunta]).toarray().astype("float32")
    sims = cosine_similarity(q_vec, embeddings)[0]

    idxs = np.argsort(sims)[::-1][:k]

    resultados = []
    for idx in idxs:
        doc = documentos_activos[idx]
        texto = doc.get("texto", "")

        oraciones = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', texto) if s.strip()]

        snippet = texto
        snippet_score = float(sims[idx])

        if oraciones and vectorizer:
            try:
                sent_vecs = vectorizer.transform(oraciones).toarray().astype("float32")
                sent_sims = cosine_similarity(q_vec, sent_vecs)[0]
                best_i = int(np.argmax(sent_sims))
                best_sentence = oraciones[best_i].strip()

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

    resultados.sort(key=lambda r: r["score"], reverse=True)

    return resultados

# ==== MODELOS ====
class Question(BaseModel):
    question: str
    
class DocumentToDelete(BaseModel):
    filename: str

# ==== RUTAS ====

@app.get("/", response_class=HTMLResponse)
def home():
    """Página principal del chatbot."""
    path = Path(__file__).parent / "templates" / "index.html"
    if path.exists():
        return HTMLResponse(content=path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<p>Index not found</p>", status_code=404)


@app.get("/manage-pdfs", response_class=HTMLResponse)
def manage_pdfs():
    """Página de gestión de PDFs."""
    path = Path(__file__).parent / "templates" / "manage-pdfs.html"
    if path.exists():
        return HTMLResponse(content=path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<p>Manage PDFs page not found</p>", status_code=404)


@app.post("/ask")
def ask(q: Question):
    resultados = buscar_respuesta(q.question, k=3)
    return {"resultados": resultados}

@app.get("/list-pdfs")
def list_pdfs():
    """Devuelve la lista única de nombres de archivos cargados."""
    if not documentos_activos:
        return {"files": []}
    
    fuentes_unicas = sorted(list(set(d["fuente"] for d in documentos_activos)))
    return {"files": fuentes_unicas}


@app.post("/upload-pdfs")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """Sube nuevos PDFs y regenera el índice."""
    if not files:
        raise HTTPException(status_code=400, detail="No se subió ningún archivo.")

    pdfs_procesados = []
    
    for archivo in files:
        if archivo.filename and archivo.filename.lower().endswith(".pdf"):
            print(f"Añadiendo PDF: {archivo.filename}")
            content = archivo.file.read()
            pdf_file = io.BytesIO(content)
            
            texto = extraer_texto(pdf_file, archivo.filename)
            chunks = dividir_en_chunks(texto)

            for chunk in chunks:
                pdfs_procesados.append({"texto": chunk, "fuente": archivo.filename})
    
    if not pdfs_procesados:
        raise HTTPException(status_code=400, detail="No se subió ningún archivo PDF válido.")

    nueva_lista_documentos = documentos_activos + pdfs_procesados
    
    if generar_y_cargar_indice_en_memoria(nueva_lista_documentos):
        return JSONResponse(
            content={"message": f"Se subieron {len(pdfs_procesados)} fragmentos nuevos. Total: {len(documentos_activos)} fragmentos indexados."},
            status_code=200
        )
    else:
        raise HTTPException(status_code=500, detail="Error al generar el índice.")


@app.post("/delete-pdf")
def delete_pdf(data: DocumentToDelete):
    """Elimina todos los fragmentos de un PDF y regenera el índice."""
    filename_to_delete = data.filename
    
    if not filename_to_delete:
        raise HTTPException(status_code=400, detail="Se requiere el nombre del archivo.")

    documentos_restantes = [d for d in documentos_activos if d["fuente"] != filename_to_delete]
    
    if len(documentos_restantes) == len(documentos_activos):
        raise HTTPException(status_code=404, detail=f"Archivo '\''{filename_to_delete}'\'' no encontrado.")

    if generar_y_cargar_indice_en_memoria(documentos_restantes):
        return JSONResponse(
            content={"message": f"PDF '\''{filename_to_delete}'\'' eliminado. Índice regenerado con {len(documentos_restantes)} fragmentos."},
            status_code=200
        )
    else:
        raise HTTPException(status_code=500, detail="Error al regenerar el índice.")

@app.post("/clear-index")
def clear_index():
    """Borra todos los documentos de la memoria RAM."""
    global documentos_activos, vectorizer, embeddings, index_loaded
    
    documentos_activos = []
    vectorizer = None
    embeddings = None
    index_loaded = False
    
    print("✔ Índice en memoria borrado.")
    
    return JSONResponse(
        content={"message": "Índice vaciado correctamente. Listo para nuevos documentos."},
        status_code=200
    )