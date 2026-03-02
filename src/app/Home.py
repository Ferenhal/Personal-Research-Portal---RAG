import streamlit as st

st.set_page_config(
    page_title="Personal Research Portal",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Personal Research Portal")

st.markdown(
    """
Welcome to my Personal Research Portal!

The intent of this project is stablish and analyze a corpus which can answer the question:

How reliable are LLMs as assistants in business analytics workflows (querying + summarizing + synthesizing), and what prompt constraints measurably improve groundedness and usability?

Use the pages in the left sidebar:
- **Ask**: ask a question, see evidence + citations, and save it to a thread
- **Threads**: browse past runs grouped by `thread_id`, export to Markdown
- **Evaluation**: run `data/eval/queries.jsonl` set and view summary metrics

**Assumptions Prior to Running the UI**
*(if you encounter any errors, double-check these first!)*

- You already ran: `make ingest`, `make chunk`, `make index`
- Ollama is running locally (default: `http://localhost:11434`) and the model is pulled
    """
)

st.info(
    "Tip: Start with **Ask**. If you haven't built the index yet, run the Makefile targets first.",
    icon="💡",
)
