import os
import re
import tempfile
import uuid
from hashlib import sha256
from time import perf_counter
from typing import Any

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

DB_PATH = os.getenv("RAG_CHROMA_PATH", "./ma_base_vectorielle")
COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "index_bge_m3")
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
LLM_MODEL_NAME = os.getenv("RAG_LLM_MODEL", "llama-3.3-70b-versatile")
TOP_K = 3
NO_ANSWER_MSG = "Je ne trouve pas la reponse dans les documents fournis"


def inject_styles() -> None:
    st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Fraunces:opsz,wght@9..144,500;9..144,700&display=swap');

:root {
    --ink: #172136;
    --muted: #405070;
    --teal: #0f766e;
    --teal-soft: #ccfbf1;
    --gold: #f59e0b;
    --sky: #e0f2fe;
    --card: rgba(255, 255, 255, 0.86);
}

.stApp {
    background:
        radial-gradient(1200px 420px at -8% -8%, #e0f2fe 0%, transparent 55%),
        radial-gradient(980px 420px at 108% -4%, #fef3c7 0%, transparent 60%),
        linear-gradient(165deg, #f8fafc 0%, #edf7f3 55%, #f5fbff 100%);
    color: var(--ink);
    font-family: "Sora", sans-serif;
}

h1, h2, h3 {
    font-family: "Fraunces", serif;
    letter-spacing: -0.01em;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f5fffb 0%, #f4f7ff 100%);
    border-right: 1px solid rgba(23, 33, 54, 0.08);
}

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 18px;
    padding: 1.15rem 1.2rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(23, 33, 54, 0.08);
    background: linear-gradient(120deg, rgba(15,118,110,0.11), rgba(245,158,11,0.13));
    box-shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
    animation: liftIn 0.5s ease;
}

.hero .title {
    margin: 0;
    font-size: 1.6rem;
    color: var(--ink);
}

.hero .subtitle {
    margin: 0.3rem 0 0;
    color: var(--muted);
    font-size: 0.95rem;
}

.hero::after {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    top: -120px;
    right: -80px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(245,158,11,0.34) 0%, rgba(245,158,11,0) 70%);
}

.stat-card {
    border-radius: 14px;
    padding: 0.75rem 0.8rem;
    background: var(--card);
    border: 1px solid rgba(23, 33, 54, 0.08);
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.06);
}

.stat-label {
    color: var(--muted);
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stat-value {
    color: var(--ink);
    font-size: 0.95rem;
    font-weight: 700;
    margin-top: 0.15rem;
}

[data-testid="stChatMessage"] {
    border-radius: 16px;
    border: 1px solid rgba(23, 33, 54, 0.08);
    background: rgba(255, 255, 255, 0.85);
    box-shadow: 0 12px 24px rgba(15, 23, 42, 0.05);
    padding: 0.35rem 0.45rem;
}

[data-testid="stChatInputTextArea"] textarea {
    border-radius: 14px;
    border: 1px solid rgba(15, 118, 110, 0.35);
    background: #ffffff;
}

.stButton button {
    border-radius: 999px;
    border: 1px solid rgba(15, 118, 110, 0.35);
    color: #0f4f4a;
    background: linear-gradient(180deg, #ffffff, #f1fffb);
    font-weight: 600;
}

.stButton button:hover {
    border-color: rgba(15, 118, 110, 0.55);
    transform: translateY(-1px);
}

@keyframes liftIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 768px) {
    .hero {
        padding: 0.95rem;
        margin-bottom: 0.85rem;
    }

    .hero .title {
        font-size: 1.25rem;
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
<div class="hero">
    <h1 class="title">RAG Chat Studio</h1>
    <p class="subtitle">Pose ta question, recois une reponse claire en sections, puis verifie les preuves dans les sources.</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_cards(collection_count: int | None) -> None:
    col_1, col_2, col_3, col_4 = st.columns(4)
    cards = [
        ("Embedding", EMBEDDING_MODEL_NAME),
        ("Collection", COLLECTION_NAME),
        ("Top-K", str(TOP_K)),
        ("Chunks indexes", str(collection_count) if collection_count is not None else "-"),
    ]

    for col, (label, value) in zip([col_1, col_2, col_3, col_4], cards):
        with col:
            st.markdown(
                f"""
<div class="stat-card">
    <div class="stat-label">{label}</div>
    <div class="stat-value">{value}</div>
</div>
                """,
                unsafe_allow_html=True,
            )


def normalize_assistant_response(raw_text: str) -> str:
    text = (raw_text or "").strip()

    if not text:
        return (
            "### Reponse finale\n"
            f"Reponse: {NO_ANSWER_MSG}\n"
            "Contexte: aucune information explicite dans les extraits recuperes.\n"
            "Limite: reformule la question ou enrichis les documents indexes."
        )

    if text == NO_ANSWER_MSG:
        return (
            "### Reponse finale\n"
            f"Reponse: {NO_ANSWER_MSG}\n"
            "Contexte: la reponse n'apparait pas explicitement dans les passages retrouves.\n"
            "Limite: aucun ajout externe n'est autorise."
        )

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"```(?:markdown|md)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    if "### Reponse finale" in text:
        body = text.split("### Reponse finale", 1)[1].strip()
    else:
        body = text

    lines = []
    for line in body.splitlines():
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        if cleaned_line.startswith("#"):
            continue
        cleaned_line = re.sub(r"^[-*]\s*", "", cleaned_line)
        if cleaned_line:
            lines.append(cleaned_line)

    normalized_body = "\n".join(lines).strip()

    def _clean_field(value: str) -> str:
        value = (value or "").strip()
        value = re.sub(r"\s+", " ", value)
        value = re.sub(
            r"^(?:(?:Reponse|Contexte|Limite)\s*:\s*)+",
            "",
            value,
            flags=re.IGNORECASE,
        )
        return value.strip()

    response_match = re.search(
        r"(?is)\bReponse\s*:\s*(.*?)(?=\bContexte\s*:|\bLimite\s*:|$)",
        normalized_body,
    )
    context_match = re.search(
        r"(?is)\bContexte\s*:\s*(.*?)(?=\bLimite\s*:|$)",
        normalized_body,
    )
    limit_match = re.search(
        r"(?is)\bLimite\s*:\s*(.*?)(?=\bContexte\s*:|\bReponse\s*:|$)",
        normalized_body,
    )

    if response_match or context_match or limit_match:
        response_value = _clean_field(response_match.group(1) if response_match else "")
        context_value = _clean_field(context_match.group(1) if context_match else "")
        limit_value = _clean_field(limit_match.group(1) if limit_match else "")

        if not response_value:
            response_value = NO_ANSWER_MSG
        if not context_value:
            context_value = "cette sortie est strictement basee sur les sources recuperees."
        if not limit_value:
            limit_value = "verifier les passages dans la section sources."
    else:
        merged = _clean_field(normalized_body)
        if not merged:
            merged = NO_ANSWER_MSG

        response_value = merged
        context_value = "cette sortie est strictement basee sur les sources recuperees."
        limit_value = "verifier les passages dans la section sources."

    return (
        "### Reponse finale\n"
        f"Reponse: {response_value}\n"
        f"Contexte: {context_value}\n"
        f"Limite: {limit_value}"
    )


def render_answer(answer_text: str, distances: list[float], retrieval_ms: float) -> None:
    with st.container(border=True):
        st.markdown(answer_text)


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_collection() -> Any:
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def clear_collection(collection: Any) -> None:
    total = collection.count()
    if total <= 0:
        return

    ids = collection.get(limit=total).get("ids") or []
    if ids:
        collection.delete(ids=ids)


def infer_current_cv_name(collection: Any) -> str | None:
    total = collection.count()
    if total <= 0:
        return None

    sample_size = min(total, 50)
    payload = collection.get(limit=sample_size, include=["metadatas"])
    metadatas = payload.get("metadatas") or []

    source_names: list[str] = []
    for metadata in metadatas:
        if isinstance(metadata, dict):
            source_name = metadata.get("source")
            if isinstance(source_name, str) and source_name.strip():
                source_names.append(source_name.strip())

    if not source_names:
        return None

    return max(set(source_names), key=source_names.count)


def reload_vector_store(uploaded_file: Any, collection: Any, model_embedding: SentenceTransformer) -> int:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = text_splitter.split_documents(docs)

        texts = [doc.page_content for doc in chunks]
        if not texts:
            return 0

        metadatas = []
        for doc in chunks:
            metadata = dict(doc.metadata or {})
            metadata["source"] = uploaded_file.name
            metadatas.append(metadata)

        embeddings = model_embedding.encode(texts).tolist()
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        return len(texts)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@st.cache_resource(show_spinner=False)
def get_llm_client() -> OpenAI:
    api_key = (
        os.getenv("Groq_API_Key")
    )
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set OPENAI_API_KEY, BLAZE_API_KEY, Blaze_API_Key, or Groq_API_Key in your .env"
        )

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def build_strict_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(
        [f"[SOURCE {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)]
    )
    return f"""Tu es un assistant RAG strict.

Regles obligatoires:
1. Utilise uniquement les faits presents dans CONTEXT.
2. N'invente aucune information externe.
3. Si la reponse ne figure pas clairement dans CONTEXT, renvoie exactement:
"{NO_ANSWER_MSG}"

Format de sortie obligatoire (une seule reponse, un seul bloc):
### Reponse finale
Reponse: <la reponse directe, claire et concise>
Contexte: <sur quoi la reponse se base dans CONTEXT>
Limite: <ce qui manque ou ce qui reste incertain>

Interdits:
- Ne genere pas d'autres titres Markdown.
- Ne genere pas plusieurs sections separees.
- Ne renvoie pas plusieurs reponses alternatives.

CONTEXT:
{context}

QUESTION:
{question}

REPONSE:
"""


def retrieve_relevant_chunks(question: str, k: int = TOP_K) -> tuple[list[str], list[float], list[dict[str, Any]]]:
    embedding_model = get_embedding_model()
    collection = get_collection()

    question_embedding = embedding_model.encode([question])[0]
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=k,
    )

    documents = (results.get("documents") or [[]])[0] or []
    distances_raw = (results.get("distances") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []

    distances = [float(value) for value in distances_raw]
    return documents, distances, metadatas


def call_llm(prompt: str) -> str:
    client = get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def render_sources(sources: list[dict[str, Any]]) -> None:
    with st.expander("Sources utilisees", expanded=False):
        if not sources:
            st.caption("Aucune source a afficher.")
            return

        for i, src in enumerate(sources, start=1):
            with st.container(border=True):
                st.markdown(f"**Source {i}**")

                distance = src.get("distance")
                metadata = src.get("metadata") or {}

                source_name = metadata.get("source")
                page = metadata.get("page")

                meta_line = []
                if distance is not None:
                    meta_line.append(f"distance={distance:.4f}")
                if source_name is not None:
                    meta_line.append(f"fichier={source_name}")
                if page is not None:
                    meta_line.append(f"page={page}")

                if meta_line:
                    st.caption(" | ".join(meta_line))

                full_text = (src.get("text") or "").strip()
                preview = full_text[:700] + ("..." if len(full_text) > 700 else "")
                st.write(preview)


def main() -> None:
    st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="wide")
    inject_styles()
    render_hero()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_cv_hash" not in st.session_state:
        st.session_state.current_cv_hash = None
    if "current_cv_name" not in st.session_state:
        st.session_state.current_cv_name = None

    quick_question = None
    uploaded_file = None
    with st.sidebar:
        st.subheader("Configuration")
        st.write(f"Collection: {COLLECTION_NAME}")
        st.write(f"DB path: {DB_PATH}")
        st.write(f"LLM model: {LLM_MODEL_NAME}")
        st.write(f"Top-k retrieval: {TOP_K}")

        st.markdown("### Raccourcis")

        st.markdown("### Chargement dynamique")
        uploaded_file = st.file_uploader(
            "Ajouter un nouveau CV (PDF)",
            type=["pdf"],
            accept_multiple_files=False,
            help="Le fichier remplace l'index actuel et reinitialise l'historique de conversation.",
        )
        st.write(f"CV actif: {st.session_state.current_cv_name or (uploaded_file.name if uploaded_file is not None else 'Aucun')}")

        if st.button("Vider l'historique", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    collection_count = None
    try:
        collection = get_collection()

        if not st.session_state.current_cv_name:
            inferred_cv_name = infer_current_cv_name(collection)
            if inferred_cv_name:
                st.session_state.current_cv_name = inferred_cv_name

        if uploaded_file is not None:
            uploaded_hash = sha256(uploaded_file.getvalue()).hexdigest()
            if uploaded_hash != st.session_state.current_cv_hash:
                with st.sidebar:
                    with st.spinner("Indexation du nouveau CV en cours..."):
                        clear_collection(collection)
                        st.session_state.messages = []
                        added_chunks = reload_vector_store(
                            uploaded_file=uploaded_file,
                            collection=collection,
                            model_embedding=get_embedding_model(),
                        )

                st.session_state.current_cv_hash = uploaded_hash
                st.session_state.current_cv_name = uploaded_file.name

                if added_chunks > 0:
                    st.sidebar.success(
                        f"{uploaded_file.name} indexe avec succes ({added_chunks} chunks ajoutes). Historique reinitialise."
                    )
                else:
                    st.sidebar.warning(
                        f"{uploaded_file.name} charge mais aucun texte exploitable n'a ete detecte. Historique reinitialise."
                    )
            else:
                if not st.session_state.current_cv_name:
                    st.session_state.current_cv_name = uploaded_file.name
                st.sidebar.info("Ce CV est deja charge.")

        collection_count = collection.count()
        render_stat_cards(collection_count)
        st.caption(f"CV actuellement charge: {st.session_state.current_cv_name or 'Aucun'}")
        if collection_count == 0:
            st.warning(
                "The ChromaDB collection is empty. Index your chunks before asking questions."
            )
    except Exception as exc:
        st.error(f"Unable to initialize ChromaDB collection: {exc}")
        return

    st.markdown("### Conversation")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer(
                    message["content"],
                    message.get("distances", []),
                    message.get("retrieval_ms", 0.0),
                )
            else:
                st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])

    user_question = st.chat_input("Pose ta question sur les documents indexes...")
    if quick_question:
        user_question = quick_question

    if not user_question:
        return

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    assistant_text = ""
    sources_payload: list[dict[str, Any]] = []
    distances_payload: list[float] = []
    retrieval_ms = 0.0

    with st.chat_message("assistant"):
        try:
            with st.spinner("Calcul embedding + recherche vectorielle..."):
                retrieval_start = perf_counter()
                chunks, distances, metadatas = retrieve_relevant_chunks(user_question, TOP_K)
                retrieval_ms = (perf_counter() - retrieval_start) * 1000
                distances_payload = distances

            if not chunks:
                assistant_text = normalize_assistant_response(NO_ANSWER_MSG)
                render_answer(assistant_text, [], retrieval_ms)
            else:
                strict_prompt = build_strict_prompt(user_question, chunks)

                with st.spinner("Generation de la reponse..."):
                    raw_answer = call_llm(strict_prompt)

                assistant_text = normalize_assistant_response(raw_answer)

                render_answer(assistant_text, distances_payload, retrieval_ms)

                for idx, chunk in enumerate(chunks):
                    sources_payload.append(
                        {
                            "text": chunk,
                            "distance": distances[idx] if idx < len(distances) else None,
                            "metadata": metadatas[idx] if idx < len(metadatas) else {},
                        }
                    )

                render_sources(sources_payload)

        except Exception as exc:
            assistant_text = "Une erreur est survenue pendant la generation de la reponse."
            st.error(str(exc))
            render_answer(assistant_text, [], retrieval_ms)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_text,
            "sources": sources_payload,
            "distances": distances_payload,
            "retrieval_ms": retrieval_ms,
        }
    )


if __name__ == "__main__":
    main()
