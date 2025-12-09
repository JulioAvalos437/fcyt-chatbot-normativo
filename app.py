import numpy as np
import re
import io
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import pypdf
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict
import unicodedata
from datetime import datetime

# ==== CONFIGURACIÓN ====
INDICE_PATH = "indice_tfidf.pkl"
PDF_DIR = "docs"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Crear carpeta docs si no existe
Path(PDF_DIR).mkdir(exist_ok=True)

# Variables globales para el índice
documentos_activos = []
vectorizer = None
tfidf_embeddings = None
dense_embeddings = None
embedding_model = None
index_loaded = False

app = FastAPI(title="Chatbot normativo FCyT")

# ==== CARGA DEL MODELO DE EMBEDDINGS ====


def cargar_modelo_embeddings():
    """Carga el modelo de Sentence Transformers."""
    global embedding_model
    if embedding_model is None:
        print("Cargando modelo de embeddings...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("✔ Modelo de embeddings cargado")
    return embedding_model

# ==== NORMALIZACIÓN DE TEXTO ====


def normalizar_texto(texto: str) -> str:
    """Normaliza texto: lowercase, elimina acentos."""
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    return texto


def limpiar_texto_busqueda(texto: str) -> str:
    """Limpieza más agresiva para búsqueda."""
    texto = normalizar_texto(texto)
    texto = re.sub(r'[^a-z0-9\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# ==== DETECCIÓN DE TIPO DE PREGUNTA ====


def detectar_tipo_pregunta(pregunta: str) -> Dict:
    """Detecta el tipo de pregunta y extrae palabras clave."""
    pregunta_lower = pregunta.lower()

    tipo = {
        "es_definicion": False,
        "es_procedimiento": False,
        "es_requisito": False,
        "es_funcion": False,
        "palabras_clave": [],
        "entidades": []
    }

    # Detectar preguntas de definición
    patrones_definicion = [
        r'\bqu[eé] es\b',
        r'\bdefin[ae]\b',
        r'\bconcepto de\b',
        r'\bsignifica\b',
        r'\bqu[eé] significa\b',
        r'\ba qu[eé] se refiere\b'
    ]

    for patron in patrones_definicion:
        if re.search(patron, pregunta_lower):
            tipo["es_definicion"] = True
            break

    # Detectar preguntas de procedimiento
    if re.search(r'\bc[oó]mo\b|\bprocedimiento\b|\bpasos\b|\bproceso\b', pregunta_lower):
        tipo["es_procedimiento"] = True

    # Detectar preguntas de requisitos
    if re.search(r'\brequisitos?\b|\bcondiciones?\b|\bnecesario\b', pregunta_lower):
        tipo["es_requisito"] = True

    # Detectar preguntas de función/rol
    if re.search(r'\bfunci[oó]n\b|\brol\b|\btarea\b|\bresponsabilidad', pregunta_lower):
        tipo["es_funcion"] = True

    # Extraer entidades importantes (palabras en mayúscula, siglas)
    entidades = re.findall(
        r'\b[A-Z]{2,}\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', pregunta)
    tipo["entidades"] = list(set(entidades))

    # Extraer palabras clave (excluir stopwords comunes)
    stopwords = {'que', 'es', 'el', 'la', 'de', 'del', 'en', 'y', 'a', 'para', 'por',
                 'con', 'un', 'una', 'los', 'las', 'se', 'al', 'como', 'cual', 'cuales'}
    palabras = re.findall(r'\b\w+\b', pregunta_lower)
    tipo["palabras_clave"] = [
        p for p in palabras if p not in stopwords and len(p) > 3]

    return tipo

# ==== FUNCIONES DE PERSISTENCIA ====


def cargar_indice_del_disco():
    """Carga el índice desde disco si existe."""
    global documentos_activos, vectorizer, tfidf_embeddings, dense_embeddings, index_loaded

    if Path(INDICE_PATH).exists():
        try:
            print(f"Cargando índice desde {INDICE_PATH}...")
            with open(INDICE_PATH, "rb") as f:
                data = pickle.load(f)

            documentos_activos = data.get("documentos", [])
            vectorizer = data.get("vectorizer")
            tfidf_embeddings = data.get("tfidf_embeddings")
            dense_embeddings = data.get("dense_embeddings")
            index_loaded = True

            print(f"✔ Índice cargado: {len(documentos_activos)} fragmentos")
            return True
        except Exception as e:
            print(f"Error cargando índice: {e}")
            return False
    else:
        print("No hay índice guardado. Iniciando con índice vacío.")
        return False


def guardar_indice_en_disco():
    """Guarda el índice actual en disco."""
    try:
        data = {
            "documentos": documentos_activos,
            "vectorizer": vectorizer,
            "tfidf_embeddings": tfidf_embeddings,
            "dense_embeddings": dense_embeddings,
        }

        with open(INDICE_PATH, "wb") as f:
            pickle.dump(data, f)

        print(f"✔ Índice guardado en {INDICE_PATH}")
        return True
    except Exception as e:
        print(f"Error guardando índice: {e}")
        return False


# Cargar índice al iniciar la aplicación
cargar_indice_del_disco()

# ==== CHUNKING INTELIGENTE ====


def extraer_metadatos(texto: str, filename: str) -> Dict:
    """Extrae metadatos del texto (capítulos, artículos, secciones)."""
    metadatos = {
        "capitulo": None,
        "articulo": None,
        "seccion": None,
        "tipo": "general",
        "es_definicion": False,
        "es_procedimiento": False
    }

    texto_lower = texto.lower()

    # Detectar si es una definición
    if re.search(r'\bes\b.*\bdefin|concept|significa', texto_lower) or texto_lower.count(':') > 0:
        metadatos["es_definicion"] = True
        metadatos["tipo"] = "definicion"

    # Detectar si es un procedimiento
    if re.search(r'\b(pasos?|etapas?|fases?|procedimiento|proceso)\b', texto_lower):
        metadatos["es_procedimiento"] = True
        metadatos["tipo"] = "procedimiento"

    # Buscar capítulo
    match_cap = re.search(
        r'cap[ií]tulo\s+([IVXLCDM]+|\d+)', texto, re.IGNORECASE)
    if match_cap:
        metadatos["capitulo"] = match_cap.group(1)
        if metadatos["tipo"] == "general":
            metadatos["tipo"] = "capitulo"

    # Buscar artículo
    match_art = re.search(r'art[ií]culo\s+(\d+)', texto, re.IGNORECASE)
    if match_art:
        metadatos["articulo"] = match_art.group(1)
        if metadatos["tipo"] == "general":
            metadatos["tipo"] = "articulo"

    # Buscar sección
    match_sec = re.search(r'secci[óo]n\s+(\d+)', texto, re.IGNORECASE)
    if match_sec:
        metadatos["seccion"] = match_sec.group(1)

    return metadatos


def dividir_en_chunks_inteligente(texto: str, filename: str, min_chars=200, max_chars=800):
    """Divide texto en chunks respetando estructura del documento."""
    chunks = []

    # NUEVO: Guardar el texto COMPLETO como referencia para búsquedas de definiciones
    texto_completo_referencia = texto

    # Primero dividir por secciones grandes
    secciones = re.split(r'\n\s*\n+', texto)

    for seccion in secciones:
        seccion = seccion.strip()
        if not seccion or len(seccion) < 50:
            continue

        # Si la sección es pequeña, mantenerla completa
        if len(seccion) <= max_chars:
            metadatos = extraer_metadatos(seccion, filename)
            # NUEVO: agregar referencia al texto completo
            metadatos["texto_completo_pdf"] = texto_completo_referencia
            chunks.append({
                "texto": seccion,
                "metadatos": metadatos
            })
        else:
            # Dividir por oraciones pero manteniendo más contexto
            oraciones = re.split(r'(?<=[.!?])\s+', seccion)
            chunk_actual = ""
            metadatos_base = extraer_metadatos(seccion[:500], filename)
            # NUEVO: agregar referencia al texto completo
            metadatos_base["texto_completo_pdf"] = texto_completo_referencia

            for i, oracion in enumerate(oraciones):
                nueva_longitud = len(chunk_actual) + len(oracion)

                if nueva_longitud <= max_chars:
                    chunk_actual += " " + oracion if chunk_actual else oracion
                else:
                    # Solo guardar si tiene contenido suficiente
                    if len(chunk_actual) >= min_chars:
                        chunks.append({
                            "texto": chunk_actual.strip(),
                            "metadatos": metadatos_base
                        })
                    chunk_actual = oracion
                    # Actualizar metadatos si cambia el contexto
                    if i < len(oraciones) - 1:
                        contexto = " ".join(
                            oraciones[i:min(i+3, len(oraciones))])
                        metadatos_base = extraer_metadatos(contexto, filename)
                        # NUEVO: agregar referencia al texto completo
                        metadatos_base["texto_completo_pdf"] = texto_completo_referencia

            if len(chunk_actual) >= min_chars:
                chunks.append({
                    "texto": chunk_actual.strip(),
                    "metadatos": metadatos_base
                })

    return chunks

# ==== PROCESAMIENTO DE PDF ====


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

# ==== GENERACIÓN DE ÍNDICE HÍBRIDO ====


def generar_indice_hibrido(documents_list: List[Dict]):
    """Genera índice híbrido: TF-IDF + embeddings densos."""
    global vectorizer, tfidf_embeddings, dense_embeddings, embedding_model

    if not documents_list:
        return None, None, None

    texts = [d["texto"] for d in documents_list]
    texts_normalized = [limpiar_texto_busqueda(t) for t in texts]

    # 1. TF-IDF
    print("Generando índice TF-IDF...")
    vectorizer_nuevo = TfidfVectorizer(
        max_features=20000,
        # Aumentado a 4-gramas para capturar frases más largas
        ngram_range=(1, 4),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True  # Mejor para documentos largos
    )
    X_tfidf = vectorizer_nuevo.fit_transform(texts_normalized)
    tfidf_emb = X_tfidf.toarray().astype("float32")

    # 2. Embeddings densos
    print("Generando embeddings densos...")
    if embedding_model is None:
        cargar_modelo_embeddings()

    dense_emb = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True  # Normalizar para mejor comparación
    ).astype("float32")

    return vectorizer_nuevo, tfidf_emb, dense_emb


def actualizar_indice(documents_list: List[Dict]):
    """Actualiza el índice en memoria y lo guarda en disco."""
    global documentos_activos, vectorizer, tfidf_embeddings, dense_embeddings, index_loaded

    print(f"\nActualizando índice con {len(documents_list)} fragmentos...")

    documentos_activos = documents_list

    if not documentos_activos:
        vectorizer = None
        tfidf_embeddings = None
        dense_embeddings = None
        index_loaded = False
        print("Índice vaciado (sin documentos)")
        guardar_indice_en_disco()
        return True

    vec_nuevo, tfidf_emb, dense_emb = generar_indice_hibrido(
        documentos_activos)

    if vec_nuevo is None:
        print("Error: no se pudo generar el índice")
        return False

    vectorizer = vec_nuevo
    tfidf_embeddings = tfidf_emb
    dense_embeddings = dense_emb
    index_loaded = True

    if guardar_indice_en_disco():
        print("✔ Índice híbrido actualizado y persistido!")
        return True
    else:
        print("Error al guardar índice en disco")
        return False

# ==== BÚSQUEDA HÍBRIDA MEJORADA ====


def calcular_boost_score(doc: Dict, tipo_pregunta: Dict, score_base: float) -> float:
    """Calcula un boost basado en la coincidencia del tipo de documento con la pregunta."""
    boost = 1.0
    metadatos = doc.get("metadatos", {})

    # Boost para definiciones
    if tipo_pregunta["es_definicion"] and metadatos.get("es_definicion"):
        boost *= 1.5

    # Boost para procedimientos
    if tipo_pregunta["es_procedimiento"] and metadatos.get("es_procedimiento"):
        boost *= 1.4

    # Boost si contiene palabras clave exactas
    texto_lower = doc["texto"].lower()
    palabras_encontradas = sum(
        1 for p in tipo_pregunta["palabras_clave"] if p in texto_lower)
    if palabras_encontradas > 0:
        boost *= (1.0 + 0.1 * palabras_encontradas)

    # Boost si contiene entidades exactas
    for entidad in tipo_pregunta["entidades"]:
        if entidad in doc["texto"]:
            boost *= 1.3

    return score_base * boost


def buscar_respuesta_hibrida(pregunta: str, k_inicial: int = 20):
    """
    Búsqueda híbrida mejorada con análisis de tipo de pregunta.
    Para definiciones, busca por coincidencia EXACTA de frases primero.
    """
    if not index_loaded:
        return {
            "texto": "El índice no está cargado. Por favor, carga un PDF en Gestionar PDFs.",
            "fuente": "Sistema",
            "score": 0.0,
            "metadatos": {}
        }

    # Cargar modelo de embeddings si no está cargado
    cargar_modelo_embeddings()

    # Analizar tipo de pregunta
    tipo_pregunta = detectar_tipo_pregunta(pregunta)
    print(f"\n🔍 Tipo de pregunta detectado: {tipo_pregunta}")

    pregunta_norm = limpiar_texto_busqueda(pregunta)

    # ========== BÚSQUEDA ESPECIAL PARA DEFINICIONES ==========
    if tipo_pregunta["es_definicion"]:
        print(f"📖 Modo búsqueda por DEFINICIÓN activado")

        # Extraer el TÉRMINO a definir
        termino_match = re.search(
            r'qu[eé]\s+es\s+([^?]+)|defin[ae].*?\s+([^?]+)|concepto\s+(?:de\s+)?([^?]+)',
            pregunta,
            re.IGNORECASE
        )

        termino_a_buscar = ""
        if termino_match:
            termino_a_buscar = next(
                (g for g in termino_match.groups() if g), "").strip()

        if termino_a_buscar:
            print(f"  📝 Término a buscar: '{termino_a_buscar}'")

        # ESTRATEGIA 1: Buscar por PATRÓN DE SECCIÓN en el TEXTO COMPLETO del PDF
        print(f"  🔎 Buscando por patrón de sección en texto completo...")
        fragmentos_con_patron = []
        pdfs_visitados = set()  # Para evitar duplicados

        for idx, doc in enumerate(documentos_activos):
            metadatos = doc.get("metadatos", {})
            # Obtener el texto completo del PDF desde los metadatos
            texto_completo = metadatos.get("texto_completo_pdf", "")
            fuente = doc.get("fuente", "")

            # Evitar procesar el mismo PDF varias veces
            if not texto_completo or fuente in pdfs_visitados:
                continue

            pdfs_visitados.add(fuente)

            # Buscar líneas que contengan "N.N. ALGO SIMILAR AL TÉRMINO"
            patron_seccion = r'^\s*\d+\.\d+\.\s+(.+?)(?:\n|$)'
            matches = re.findall(
                patron_seccion, texto_completo, re.MULTILINE | re.IGNORECASE)

            for match_titulo in matches:
                # Comparar el título de la sección con el término buscado
                titulo_lower = match_titulo.lower()
                termino_lower = termino_a_buscar.lower()

                # Contar cuántas palabras coinciden
                palabras_titulo = set(re.findall(r'\b\w+\b', titulo_lower))
                palabras_termino = set(re.findall(r'\b\w+\b', termino_lower))
                palabras_comunes = palabras_titulo & palabras_termino

                # Si hay coincidencia significativa de palabras
                if len(palabras_comunes) >= len(palabras_termino) * 0.5:
                    puntuacion = len(palabras_comunes) / \
                        max(len(palabras_termino), 1)

                    # Extraer la SECCIÓN COMPLETA desde el texto completo
                    # Buscar desde el título hasta la siguiente sección o fin
                    patron_completo = (
                        r'^\s*\d+\.\d+\.\s+' + re.escape(match_titulo) +
                        r'(.+?)(?=^\s*\d+\.\d+\.|$)'
                    )
                    match_seccion = re.search(
                        patron_completo,
                        texto_completo,
                        re.MULTILINE | re.IGNORECASE | re.DOTALL
                    )

                    if match_seccion:
                        contenido_seccion = match_seccion.group(1).strip()

                        # CAMBIO: Buscar desde la línea del título hasta la siguiente sección
                        # sin usar non-greedy matching que corta rápido
                        pos_titulo = texto_completo.find(f"{match_titulo}")
                        if pos_titulo != -1:
                            # Buscar desde después del título
                            pos_inicio = pos_titulo + len(match_titulo)

                            # Buscar la siguiente sección
                            siguiente_seccion = re.search(
                                r'\n\s*\d+\.\d+\.',
                                texto_completo[pos_inicio:]
                            )

                            if siguiente_seccion:
                                pos_final = pos_inicio + siguiente_seccion.start()
                            else:
                                pos_final = len(texto_completo)

                            # Extraer el contenido completo entre el título y siguiente sección
                            contenido_seccion = texto_completo[pos_inicio:pos_final].strip(
                            )

                        # Limitar a máximo 350 palabras
                        palabras = contenido_seccion.split()
                        if len(palabras) > 220:
                            contenido_seccion = " ".join(
                                palabras[:220]) + "..."

                        seccion_completa = (
                            match_titulo + "\n" + contenido_seccion).strip()
                        fragmentos_con_patron.append(
                            (idx, puntuacion, match_titulo, seccion_completa, fuente))
                        print(
                            f"    ✅ Fragmento {idx} ('{fuente}'): '{match_titulo}' - score {puntuacion:.2f} - {len(seccion_completa)} chars")

        if fragmentos_con_patron:
            # Usar el fragmento con mejor coincidencia de patrón
            idx_mejor, puntuacion, titulo, seccion_completa, fuente = max(
                fragmentos_con_patron, key=lambda x: x[1])

            print(f"  ✅ MATCH POR PATRÓN: fragmento {idx_mejor}")
            print(f"     Sección encontrada: '{titulo}'")

            # Limitar longitud
            MAX_CHARS = 3000  # Aumentado para secciones completas
            if len(seccion_completa) > MAX_CHARS:
                seccion_completa = seccion_completa[:MAX_CHARS].rsplit(" ", 1)[
                    0] + "..."

            return {
                "texto": seccion_completa,
                "fuente": fuente,
                "score": 0.99,
                "metadatos": documentos_activos[idx_mejor].get("metadatos", {})
            }

        # ESTRATEGIA 2: Si no encontró por patrón, usar TF-IDF puro
        print(f"  ⚠️ No se encontró por patrón. Usando TF-IDF...")

    # ========== BÚSQUEDA ESTÁNDAR HÍBRIDA ==========

    # 1. TF-IDF (mejor para coincidencias exactas)
    q_tfidf = vectorizer.transform([pregunta_norm]).toarray().astype("float32")
    scores_tfidf = cosine_similarity(q_tfidf, tfidf_embeddings)[0]

    # 2. Embeddings densos (mejor para semántica)
    q_dense = embedding_model.encode(
        [pregunta],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    scores_dense = cosine_similarity(q_dense, dense_embeddings)[0]

    # Ajustar pesos según tipo de pregunta
    if tipo_pregunta["es_definicion"]:
        alpha = 0.2  # Máximo peso a TF-IDF para definiciones
    elif tipo_pregunta["palabras_clave"] and len(tipo_pregunta["palabras_clave"]) > 3:
        alpha = 0.35
    else:
        alpha = 0.6

    # Combinar scores
    scores_hibridos = alpha * scores_dense + (1 - alpha) * scores_tfidf

    # Top-K inicial
    top_k_indices = np.argsort(scores_hibridos)[::-1][:k_inicial]

    # 2. Procesar resultados
    candidatos = []

    for idx in top_k_indices:
        doc = documentos_activos[idx]
        texto_original = doc["texto"]
        metadatos = doc.get("metadatos", {})

        # Calcular boost basado en tipo de documento
        score_boosted = calcular_boost_score(
            doc, tipo_pregunta, scores_hibridos[idx])

        # ========== ESTRATEGIA DIFERENCIADA ==========

        # Para DEFINICIONES: devolver el párrafo/bloque COMPLETO
        if tipo_pregunta["es_definicion"] or metadatos.get("es_definicion"):
            resultado_texto = texto_original.strip()
            resultado_score = float(score_boosted)

        # Para PROCEDIMIENTOS: expandir con más contexto
        elif tipo_pregunta["es_procedimiento"] or metadatos.get("es_procedimiento"):
            oraciones = re.split(r'(?<=[.!?])\s+', texto_original)
            oraciones = [s.strip() for s in oraciones if len(s.strip()) > 20]

            if len(oraciones) > 1:
                oraciones_embeddings = embedding_model.encode(
                    oraciones,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                scores_oraciones = cosine_similarity(
                    q_dense, oraciones_embeddings)[0]
                mejor_idx = np.argmax(scores_oraciones)

                start = max(0, mejor_idx - 1)
                end = min(len(oraciones), mejor_idx + 3)
                resultado_texto = " ".join(oraciones[start:end])
                resultado_score = float(
                    scores_oraciones[mejor_idx]) * 0.6 + float(score_boosted) * 0.4
            else:
                resultado_texto = texto_original.strip()
                resultado_score = float(score_boosted)

        # Para BÚSQUEDAS GENERALES: segmentar por párrafos
        else:
            parrafos = re.split(r'\n\s*\n+', texto_original)
            parrafos = [p.strip() for p in parrafos if len(p.strip()) > 50]

            if len(parrafos) > 1:
                parrafos_embeddings = embedding_model.encode(
                    parrafos,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                scores_parrafos = cosine_similarity(
                    q_dense, parrafos_embeddings)[0]
                mejor_idx = np.argmax(scores_parrafos)
                resultado_texto = parrafos[mejor_idx]
                resultado_score = float(
                    scores_parrafos[mejor_idx]) * 0.7 + float(score_boosted) * 0.3
            else:
                resultado_texto = texto_original.strip()
                resultado_score = float(score_boosted)

        # Limitar longitud
        MAX_CHARS = 2000
        if len(resultado_texto) > MAX_CHARS:
            resultado_texto = resultado_texto[:MAX_CHARS].rsplit(" ", 1)[
                0] + "..."

        candidatos.append({
            "texto": resultado_texto,
            "fuente": doc.get("fuente", ""),
            "metadatos": metadatos,
            "score": resultado_score,
            "score_original": float(scores_hibridos[idx])
        })

    # 3. Ordenar y seleccionar el mejor
    if not candidatos:
        return {
            "texto": "No se encontró información relevante para tu consulta.",
            "fuente": "Sistema",
            "score": 0.0,
            "metadatos": {}
        }

    candidatos.sort(key=lambda x: x["score"], reverse=True)
    mejor_resultado = candidatos[0]

    # Log de debug
    print(
        f"✅ Mejor resultado: score={mejor_resultado['score']:.3f}, fuente={mejor_resultado['fuente']}")
    print(f"   Metadatos: {mejor_resultado['metadatos']}")
    print(f"   Texto preview: {mejor_resultado['texto'][:150]}...")

    return mejor_resultado

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
    """Endpoint de búsqueda - retorna el mejor resultado único."""
    resultado = buscar_respuesta_hibrida(q.question)
    return {"resultado": resultado}


@app.get("/list-pdfs")
def list_pdfs():
    """Devuelve información de cada PDF: fecha de carga, tamaño y cantidad de fragmentos."""
    if not documentos_activos:
        return {
            "files": [],
            "total_fragments": 0,
            "tamano_indice_bytes": 0
        }

    info_por_pdf = {}

    for doc in documentos_activos:
        fuente = doc.get("fuente")
        fecha = doc.get("fecha_carga")  # agregado
        size = doc.get("tamano_pdf")    # agregado

        # inicializar si no existe
        if fuente not in info_por_pdf:
            info_por_pdf[fuente] = {
                "filename": fuente,
                "fecha_carga": fecha,
                "tamano_pdf_bytes": size,
                "fragments": 0
            }

        info_por_pdf[fuente]["fragments"] += 1

    # tamaño total del índice en disco
    tamano_indice = os.path.getsize(
        INDICE_PATH) if Path(INDICE_PATH).exists() else 0

    return {
        "files": list(info_por_pdf.values()),
        "total_fragments": len(documentos_activos),
        "tamano_indice_bytes": tamano_indice
    }


@app.post("/upload-pdfs")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """Sube nuevos PDFs, procesa con chunking inteligente y actualiza índice."""
    if not files:
        raise HTTPException(
            status_code=400, detail="No se subió ningún archivo.")

    pdfs_procesados = []
    fecha_carga = datetime.now().isoformat()

    cargar_modelo_embeddings()

    for archivo in files:
        if archivo.filename and archivo.filename.lower().endswith(".pdf"):
            print(f"\nProcesando PDF: {archivo.filename}")
            content = archivo.file.read()
            pdf_file = io.BytesIO(content)
            pdf_size = len(content)  

            pdf_path = os.path.join(PDF_DIR, archivo.filename)
            with open(pdf_path, "wb") as f:
                f.write(content)

            print(f"  - PDF guardado en {pdf_path}")

            texto = extraer_texto(pdf_file, archivo.filename)
            chunks = dividir_en_chunks_inteligente(texto, archivo.filename)

            for chunk_data in chunks:
                pdfs_procesados.append({
                    "texto": chunk_data["texto"],
                    "fuente": archivo.filename,
                    "fecha_carga": fecha_carga,
                    "tamano_pdf": pdf_size,
                    "metadatos": chunk_data["metadatos"]
                })

            print(f"  - {len(chunks)} fragmentos extraídos")

    if not pdfs_procesados:
        raise HTTPException(
            status_code=400, detail="No se subió ningún archivo PDF válido.")

    nueva_lista_documentos = documentos_activos + pdfs_procesados

    if actualizar_indice(nueva_lista_documentos):
        return JSONResponse(
            content={
                "message": f"Se indexaron {len(pdfs_procesados)} fragmentos nuevos. Total: {len(documentos_activos)} fragmentos."},
            status_code=200
        )
    else:
        raise HTTPException(
            status_code=500, detail="Error al generar el índice.")


@app.post("/delete-pdf")
def delete_pdf(data: DocumentToDelete):
    """Elimina un PDF de disco y del índice."""
    filename_to_delete = data.filename

    if not filename_to_delete:
        raise HTTPException(
            status_code=400, detail="Se requiere el nombre del archivo.")

    pdf_path = os.path.join(PDF_DIR, filename_to_delete)
    if Path(pdf_path).exists():
        try:
            os.remove(pdf_path)
            print(f"✔ PDF eliminado de disco: {pdf_path}")
        except Exception as e:
            print(f"Error al eliminar PDF: {e}")

    documentos_restantes = [
        d for d in documentos_activos if d["fuente"] != filename_to_delete]

    if len(documentos_restantes) == len(documentos_activos):
        raise HTTPException(
            status_code=404, detail=f"Archivo '{filename_to_delete}' no encontrado en el índice.")

    if actualizar_indice(documentos_restantes):
        return JSONResponse(
            content={
                "message": f"PDF '{filename_to_delete}' eliminado. Índice regenerado con {len(documentos_restantes)} fragmentos."},
            status_code=200
        )
    else:
        raise HTTPException(
            status_code=500, detail="Error al regenerar el índice.")


@app.post("/clear-index")
def clear_index():
    """Borra todos los documentos del índice y disco."""
    global documentos_activos, vectorizer, tfidf_embeddings, dense_embeddings, index_loaded

    if Path(PDF_DIR).exists():
        try:
            for archivo in os.listdir(PDF_DIR):
                if archivo.lower().endswith(".pdf"):
                    pdf_path = os.path.join(PDF_DIR, archivo)
                    os.remove(pdf_path)
                    print(f"✔ PDF eliminado: {pdf_path}")
        except Exception as e:
            print(f"Error eliminando PDFs: {e}")

    documentos_activos = []
    vectorizer = None
    tfidf_embeddings = None
    dense_embeddings = None
    index_loaded = False

    if Path(INDICE_PATH).exists():
        try:
            os.remove(INDICE_PATH)
            print(f"✔ Archivo de índice eliminado: {INDICE_PATH}")
        except Exception as e:
            print(f"Error eliminando índice: {e}")

    print("✔ Índice y PDFs limpiados completamente.")

    return JSONResponse(
        content={
            "message": "Índice y PDFs eliminados correctamente. Listo para nuevos documentos."},
        status_code=200
    )
