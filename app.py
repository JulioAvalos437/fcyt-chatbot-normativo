import pickle
import numpy as np
import re
import os
import io
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import pypdf

# Nombre de los archivos del índice (Se mantiene por referencia, pero NO se usa para guardar/cargar)
INDICE_PATH = "indice_tfidf.pkl" 

# Variables globales para el índice (Inician vacías y se vacían al reiniciar Uvicorn)
documentos_activos = [] # Lista de fragmentos (chunks) {texto, fuente}
vectorizer = None
embeddings = None
index_loaded = False

app = FastAPI(title="Chatbot normativo FCyT")

# ==== FUNCIONES DE PROCESAMIENTO DE PDF (Sin cambios) ====
# (extraer_texto y dividir_en_chunks permanecen iguales)
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

# ==== GESTIÓN Y GENERACIÓN DEL ÍNDICE (SOLO EN MEMORIA) ====

# Renombramos la función y ELIMINAMOS el guardado en disco
def generar_y_cargar_indice_en_memoria(current_documents: list):
    """Genera el nuevo índice TF-IDF basado en la lista de documentos y LO CARGA EN MEMORIA."""
    global documentos_activos, vectorizer, embeddings, index_loaded
    
    documentos_activos = current_documents # Actualizar la lista activa
    
    if not documentos_activos:
        print("La lista de documentos está vacía. Índice no generado.")
        vectorizer = None
        embeddings = None
        index_loaded = False
        return True

    print(f"\nGenerando índice con un total de {len(documentos_activos)} fragmentos...")
    
    texts = [d["texto"] for d in documentos_activos]

    # Crear Matriz TF-IDF
    vectorizer_nuevo = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words=None
    )

    X = vectorizer_nuevo.fit_transform(texts)
    embeddings_nuevos = X.toarray().astype("float32")
    
    # *** ELIMINAMOS AQUÍ LA LÓGICA DE GUARDADO EN DISCO (pickle.dump) ***
    
    # Actualizar las variables globales
    vectorizer = vectorizer_nuevo
    embeddings = embeddings_nuevos
    index_loaded = True
    
    print("✔ ¡Índice generado y cargado EN MEMORIA correctamente!")
    return True

# ==== ELIMINAMOS LA CARGA INICIAL ====
# Eliminamos completamente la función cargar_indice_inicial() y su llamada para asegurar
# que el índice esté vacío al iniciar el servidor.


# ==== LÓGICA DE BÚSQUEDA (sin cambios en la lógica) ====
def buscar_respuesta(pregunta: str, k: int = 3):
    """Devuelve los k fragmentos más similares a la pregunta."""
    if not index_loaded:
        return [{"texto": "El índice de documentos no está cargado. Por favor, sube un PDF para generar el índice.", "fuente": "Sistema", "score": 0.0}]
    
    # [ ... Resto de la lógica de búsqueda sin cambios ... ]
    
    q_vec = vectorizer.transform([pregunta]).toarray().astype("float32")
    sims = cosine_similarity(q_vec, embeddings)[0]

    idxs = np.argsort(sims)[::-1][:k]

    resultados = []
    for idx in idxs:
        doc = documentos_activos[idx] # Usar la lista activa
        texto = doc.get("texto", "")

        # Dividir en oraciones (el mismo código de segmentación de oraciones)
        oraciones = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', texto) if s.strip()]

        snippet = texto
        snippet_score = float(sims[idx])

        if oraciones and vectorizer: # Aseguramos que el vectorizer no es None
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

# ==== MODELOS DE ENTRADA / SALIDA (sin cambios) ====
class Question(BaseModel):
    question: str
    
class DocumentToDelete(BaseModel):
    filename: str

# ==== RUTAS DE GESTIÓN Y CHATBOT ====

@app.get("/", response_class=HTMLResponse)
def home():
    """
    Ruta raíz que borra el índice de la memoria y sirve el HTML.
    Esto asegura que al recargar la página, el índice siempre esté vacío.
    """
    global documentos_activos, vectorizer, embeddings, index_loaded
    
    # *** LÓGICA DE BORRADO AUTOMÁTICO AL CARGAR LA PÁGINA ***
    if index_loaded:
        documentos_activos = []
        vectorizer = None
        embeddings = None
        index_loaded = False
        print("Índice en memoria vaciado automáticamente al cargar la página.")

    # Servir el HTML desde templates/index.html
    path = Path(__file__).parent / "templates" / "index.html"
    if path.exists():
        return HTMLResponse(content=path.read_text(encoding="utf-8"))
    
    return HTMLResponse(content="<p>Index not found</p>", status_code=404)


@app.post("/ask")
def ask(q: Question):
    resultados = buscar_respuesta(q.question, k=3)
    return {"resultados": resultados}

@app.get("/list-pdfs")
def list_pdfs():
    """Devuelve la lista única de nombres de archivos cargados."""
    if not documentos_activos:
        return {"files": []}
    
    # Obtener nombres de archivo únicos de todos los fragmentos
    fuentes_unicas = sorted(list(set(d["fuente"] for d in documentos_activos)))
    return {"files": fuentes_unicas}


@app.post("/upload-pdfs")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """Sube nuevos PDFs, los añade a la lista activa y regenera el índice."""
    if not files:
        raise HTTPException(status_code=400, detail="No se subió ningún archivo.")

    pdfs_procesados = []
    
    # 1. Procesar nuevos archivos y extraer fragmentos
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
        raise HTTPException(status_code=400, detail="No se subió ningún archivo PDF válido o no se pudo extraer texto.")

    # 2. Combinar con los documentos activos existentes
    nueva_lista_documentos = documentos_activos + pdfs_procesados
    
    # 3. Generar y cargar el nuevo índice completo (¡USAMOS LA FUNCIÓN EN MEMORIA!)
    if generar_y_cargar_indice_en_memoria(nueva_lista_documentos):
        return JSONResponse(
            content={"message": f"Éxito: Se subieron {len(pdfs_procesados)} fragmentos nuevos y se regeneró el índice con {len(documentos_activos)} fragmentos totales."},
            status_code=200
        )
    else:
        raise HTTPException(status_code=500, detail="Fallo la generación del índice.")


@app.post("/delete-pdf")
def delete_pdf(data: DocumentToDelete):
    """Elimina todos los fragmentos asociados a un nombre de archivo y regenera el índice."""
    filename_to_delete = data.filename
    
    if not filename_to_delete:
        raise HTTPException(status_code=400, detail="Se requiere el nombre del archivo a eliminar.")

    # 1. Filtrar los documentos activos, excluyendo los del archivo a eliminar
    documentos_restantes = [d for d in documentos_activos if d["fuente"] != filename_to_delete]
    
    if len(documentos_restantes) == len(documentos_activos):
        raise HTTPException(status_code=404, detail=f"Archivo '{filename_to_delete}' no encontrado en el índice.")

    # 2. Generar y cargar el nuevo índice con los documentos restantes (¡USAMOS LA FUNCIÓN EN MEMORIA!)
    if generar_y_cargar_indice_en_memoria(documentos_restantes):
        return JSONResponse(
            content={"message": f"Éxito: Archivo '{filename_to_delete}' eliminado. Índice regenerado con {len(documentos_restantes)} fragmentos."},
            status_code=200
        )
    else:
        raise HTTPException(status_code=500, detail="Fallo la regeneración del índice después de la eliminación.")
    
    # ... (Después de las otras rutas) ...

@app.post("/clear-index")
def clear_index():
    """Borra todos los documentos, el vectorizador y el índice de la memoria RAM."""
    global documentos_activos, vectorizer, embeddings, index_loaded
    
    # Limpiar todas las variables globales
    documentos_activos = []
    vectorizer = None
    embeddings = None
    index_loaded = False
    
    print("✔ Índice en memoria borrado exitosamente.")
    
    return JSONResponse(
        content={"message": "Índice en memoria vaciado y listo para nuevos documentos."},
        status_code=200
    )