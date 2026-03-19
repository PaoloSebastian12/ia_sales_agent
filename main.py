import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from src.engine import build_automerging_index, create_sales_agent, build_email_engine
from src.evaluator import evaluate_rag
import math

if "eval_result" not in st.session_state:
    st.session_state.eval_result = None

st.set_page_config(page_title="AI Sales Intelligence",
                   page_icon="🚀", layout="wide")
load_dotenv()

st.markdown("""
    <style>
    /* Estilo para los botones principales */
    .stButton > button {
        border-radius: 20px;      /* Bordes más redondeados */
        width: auto;             /* No tan anchos por defecto */
        padding: 10px 25px;       /* Espaciado interno (arriba/abajo, lados) */
        background-color: #006666; /* Color del logo */
        color: white;             /* Texto blanco para buen contraste */
        font-weight: bold;        /* Texto en negrita (opcional) */
        border: none;             /* Quitamos el borde por defecto de streamlit */
        transition: background-color 0.3s ease; /* Transición suave del color */
        margin-top: 10px;          /* Espaciado superior */
        display: inline-block;     /* Asegura que 'width: auto' funcione bien */
    }

    /* Efecto 'hover' al pasar el mouse encima */
    .stButton > button:hover {
        background-color: #1E90FF; /* Cambia a azul mar */
        color: white;             /* Mantiene el texto blanco */
    }
    </style>
""", unsafe_allow_html=True)

if "GOOGLE_API_KEY" in os.environ:
    Settings.llm = Gemini(model="models/gemini-2.5-flash")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
else:
    st.error("⚠️ Please configure your GOOGLE_API_KEY in the .env file.")

if "index" not in st.session_state:
    st.session_state.index = None

with st.sidebar:
    try:
        st.image("src/LOGO.png", width=200,)
    except FileExistsError:
        st.warning("Error")
    st.info("This agent analyzes Contracts and Technical Manuals using Advanced RAG (Auto-merging Retrieval).")
    use_existing_index = False
    if os.path.exists("./merging_index"):
        use_existing_index = st.radio(
            "Existing index detected. What would you like to do?",
            [
                "Use existing index (fast)",
                "Rebuild index from documents"
            ]
        )
    if st.button("Load Documents", use_container_width=True):
        force_reindex = use_existing_index == "Rebuild index from documents"
        if force_reindex:
            st.cache_resource.clear()
        with st.spinner("Preparing knowledge base..."):
            force_reindex = use_existing_index == "Rebuild index from documents"
            tech_engine = build_automerging_index(
                "./data/technical", "./merging_index", force_reindex)
            email_engine = build_email_engine("./data/emails/emails.csv")
            contract_engine = build_automerging_index(
                "./data/contracts", "./merging_index_contracts", force_reindex)
            st.session_state.index = create_sales_agent(
                tech_engine, email_engine, contract_engine)
            st.success("Full Knowledge Base Active!")
    st.caption(
                "Rebuilding the index may take 5–15 minutes depending on document size.")

st.title("AI Sales & Compliance Agent")
st.subheader("Intelligent Sales Viability & Technical Analyst")
col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.write("### 💬 Consultant Inquiry")
    query = st.text_input(
        "Ask a technical or legal question about the deal:",
        placeholder="e.g., Do we comply with ISO 27001 standards?"
    )
    start = st.button("Start Inquiry", use_container_width=True)
    if query and start:
        if st.session_state.index:
            query_engine = st.session_state.index
            with st.spinner("Generating answer and evaluating with RAGAS..."):
                response = query_engine.query(query)
                st.markdown("#### 🧠 Tool Used")
                tools_used = set()
                if hasattr(response, "metadata") and response.metadata:
                    source = response.metadata.get("source", "")
                    if source == "email":
                        tools_used.add("📧 Customer Emails")
                    elif source == "technical":
                        tools_used.add("⚙️ Technical Manuals")
                    elif source == "contract":
                        tools_used.add("📄 Legal Contracts")
                for node in getattr(response, "source_nodes", []):
                    metadata = node.metadata
                    if "technical" in str(metadata).lower():
                        tools_used.add("⚙️ Technical Manuals")
                    elif "contract" in str(metadata).lower():
                        tools_used.add("📄 Legal Contracts")
                if tools_used:
                    for tool in tools_used:
                        st.success(tool)
                else:
                    st.warning("No tool detected")
                source_nodes = response.source_nodes
                st.session_state.eval_result = evaluate_rag(
                    query, response.response, source_nodes)
                st.markdown("#### Analyst's Response:")
                st.info(response)
        else:
            st.warning(
                "👈 Please index the documents in the sidebar before proceeding.")

with col2:
    st.write("### 📊 System Status and RAGAS Quality Metrics")
    if st.session_state.index:
        st.success("✅ RAG Engine Active")
        if st.session_state.eval_result is not None:
            f_score = st.session_state.eval_result['faithfulness'].iloc[0]
            a_score = st.session_state.eval_result['answer_relevancy'].iloc[0]
            if f_score is None or math.isnan(f_score):
                st.warning("No faithfulness detected")
            else:
                st.metric("Faithfulness (Anti-Hallucination)",
                          f"{f_score:.2f}")
                st.progress(float(f_score))
            if a_score is None or math.isnan(a_score):
                st.warning("No Anser Relevance detected")
            else:
                st.metric("Answer Relevance", f"{a_score:.2f}")
                st.progress(float(a_score))

        else:
            st.info("Run a query to generate evaluation metrics.")
    else:
        st.error("❌ RAG Engine Inactive")
        st.write("Please run the Load Documents to start.")

st.divider()
st.caption("⚠️ AI Engineering Prototype - Focused on Hallucination Mitigation & Hierarchical Context Retrieval.")
