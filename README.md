# Tp_Rag Streamlit Chat

This project now includes a Streamlit conversational UI for your RAG pipeline.

## Run

1. Install dependencies:

```bash
uv sync
```

2. Ensure your vector DB is already indexed in `./ma_base_vectorielle` with collection `index_bge_m3`.

3. Set environment variables in `.env` (example):

```env
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
RAG_LLM_MODEL=llama-3.3-70b-versatile
```

4. Start the app:

```bash
uv run streamlit run app.py
```

## Features

- `st.chat_input` + `st.chat_message` conversational UX
- Message history with `st.session_state`
- Question embedding with `BAAI/bge-m3`
- Top-3 retrieval from ChromaDB
- Strict anti-hallucination prompt construction
- LLM generation via `openai` SDK
- Source chunks displayed in an expander under each assistant answer
