import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_hugging_face_embeddings

# 1. Page Configuration (Premium Dark Theme Layout)
st.set_page_config(
    page_title="MediBot AI Pro", 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Injecting CSS for Advanced UI Styling (Glassmorphism & Neon accents)
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .stApp { background-color: #0b0f19; color: #f1f5f9; }
    
    /* Sidebar Custom Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937;
    }
    
    /* Custom Cards for Health Tips */
    .tip-card {
        background: rgba(31, 41, 55, 0.7);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #0ea5e9;
        margin-bottom: 12px;
    }
    
    /* Premium Headers */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0ea5e9, #22c55e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    
    /* Subtle subtitle */
    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    
    /* Chat inputs styling */
    .stChatInput {
        border-radius: 20px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# 2. Caching RAG Core System
@st.cache_resource
def initialize_rag_system():
    embeddings = download_hugging_face_embeddings()
    index_name = "medical-chatbot"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model_name="llama-3.1-8b-instant", 
        temperature=0.2 # Lower temperature for higher medical accuracy
    )
    
    system_prompt = (
        "You are a Medical expert AI medical assistant. Use the following pieces of retrieved context "
        "to answer the question. If you don't know the answer, say that you don't know.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

try:
    rag_chain = initialize_rag_system()
except Exception as e:
    st.error(f"Initialization Failed: {e}")
    st.stop()

# ==================== SIDEBAR DESIGN ====================
with st.sidebar:
    st.markdown("<h2 style='color: #0ea5e9; text-align:center;'>⚙️ MediBot Panel</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 💡 Quick Health Tips")
    st.markdown("""
        <div class="tip-card">
            <strong>Stay Hydrated:</strong> Drinking water regulates body temperature and keeps organs functioning properly.
        </div>
        <div class="tip-card">
            <strong>Check Symptoms:</strong> This AI utilizes certified encyclopedias to fetch contextual background data.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔒 Disclaimer")
    st.caption("MediBot is an AI assistant built for educational reference using standard medical literature. Always consult a certified medical practitioner for professional clinical diagnosis.")
    
    # Simple Reset Chat Button
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Hello Saif! System reset done. How can I assist you now?"}]
        st.rerun()

# ==================== MAIN CHAT INTERFACE ====================

# Layout split for spacing
st.markdown("<div class='main-title'>🧬 MediBot AI Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Clinical Knowledge Base & Retrieval System</div>", unsafe_allow_html=True)

# Session State for Chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello Saif! I have mapped the medical encyclopedia index. Drop your symptoms or queries below."}
    ]

# Render chat content cleanly
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt ingestion
if user_query := st.chat_input("Ask about symptoms, conditions, treatments..."):
    
    # Display user input
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Spinner wrapper with custom text
    with st.chat_message("assistant"):
        with st.spinner("Searching secure database chunks & compiling response..."):
            try:
                response = rag_chain.invoke({"input": user_query})
                bot_reply = response["answer"]
                st.markdown(bot_reply)
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            except Exception as e:
                st.error("Error connecting to LLM cluster. Please try again.")