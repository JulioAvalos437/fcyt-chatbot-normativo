import pickle
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from pathlib import Path

# ==== CARGA DEL ÍNDICE ====

with open("indice_tfidf.pkl", "rb") as f:
    data = pickle.load(f)

documentos = data["documentos"]    # lista de dicts {texto, fuente}
vectorizer = data["vectorizer"]    # TfidfVectorizer entrenado
embeddings = data["embeddings"]    # matriz (n_docs x dim)

app = FastAPI(title="Chatbot normativo FCyT")


def buscar_respuesta(pregunta: str, k: int = 3):
    """Devuelve los k fragmentos más similares a la pregunta.
    Para cada documento seleccionado, extrae la oración más relevante
    (evitando traer texto anterior no relacionado).
    """
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


# ==== MODELOS DE ENTRADA / SALIDA ====

class Question(BaseModel):
    question: str


# ==== RUTAS ====

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