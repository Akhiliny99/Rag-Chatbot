__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
import pypdf
import time
import hashlib

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap');
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }
    .title-text {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle-text {
        color: #64748b; font-size: 0.95rem;
        margin-top: 0.2rem; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e2330; border: 1px solid #2d3748;
        border-radius: 12px; padding: 1rem 1.2rem; text-align: center;
    }
    .metric-label { color: #64748b; font-size: 0.75rem;
        letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.3rem; }
    .metric-value { color: #e2e8f0; font-size: 1.4rem; font-weight: 700; }
    .chat-user {
        background: linear-gradient(135deg, #1e3a5f, #1e2d5f);
        border: 1px solid #3b82f6;
        border-radius: 12px 12px 4px 12px;
        padding: 1rem 1.2rem; margin: 0.8rem 0;
        color: #e2e8f0;
    }
    .chat-bot {
        background: linear-gradient(135deg, #1a1f2e, #1e2330);
        border: 1px solid #2d3748;
        border-radius: 12px 12px 12px 4px;
        padding: 1rem 1.2rem; margin: 0.8rem 0;
        color: #e2e8f0;
    }
    .chat-label-user { color: #60a5fa; font-size: 0.75rem;
        font-weight: 700; margin-bottom: 0.4rem; }
    .chat-label-bot  { color: #a78bfa; font-size: 0.75rem;
        font-weight: 700; margin-bottom: 0.4rem; }
    .source-box {
        background: #13161e; border: 1px solid #2d3748;
        border-radius: 8px; padding: 0.6rem 0.8rem;
        margin: 0.3rem 0; font-size: 0.8rem; color: #64748b;
    }
    .status-ready {
        background: rgba(74,222,128,0.1); border: 1px solid #4ade80;
        border-radius: 8px; padding: 0.5rem 1rem;
        color: #4ade80; font-size: 0.85rem; text-align: center;
    }
    .status-empty {
        background: rgba(251,146,60,0.1); border: 1px solid #fb923c;
        border-radius: 8px; padding: 0.5rem 1rem;
        color: #fb923c; font-size: 0.85rem; text-align: center;
    }
</style>
""", unsafe_allow_html=True)


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'docs_loaded' not in st.session_state:
    st.session_state.docs_loaded = False
if 'doc_count' not in st.session_state:
    st.session_state.doc_count = 0
if 'chunk_count' not in st.session_state:
    st.session_state.chunk_count = 0
if 'doc_names' not in st.session_state:
    st.session_state.doc_names = []



@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_chroma_client():
    return chromadb.Client()

embedding_model = load_embedding_model()
chroma_client   = get_chroma_client()



def extract_text_from_pdf(uploaded_file):
    reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n[Page {page_num+1}]\n{page_text}"
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def add_documents_to_db(texts, doc_name, collection):
    all_chunks = []
    all_ids    = []
    all_metas  = []
    for i, chunk in enumerate(texts):
        chunk_id = hashlib.md5(f"{doc_name}_{i}_{chunk[:50]}".encode()).hexdigest()
        all_chunks.append(chunk)
        all_ids.append(chunk_id)
        all_metas.append({'source': doc_name, 'chunk': i})
    embeddings = embedding_model.encode(all_chunks).tolist()
    collection.upsert(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metas
    )
    return len(all_chunks)


def retrieve_relevant_chunks(query, collection, n_results=4):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count())
    )
    chunks   = results['documents'][0] if results['documents'] else []
    metadata = results['metadatas'][0]  if results['metadatas'] else []
    return chunks, metadata


def generate_answer(query, context_chunks, chat_history, api_key):
    
    client  = Groq(api_key=api_key)
    context = "\n\n---\n\n".join(context_chunks)

    history_text = ""
    for msg in chat_history[-4:]:
        history_text += f"User: {msg['user']}\nAssistant: {msg['bot']}\n\n"

    prompt = f"""You are a helpful AI assistant that answers questions 
based on the provided document context.

DOCUMENT CONTEXT:
{context}

PREVIOUS CONVERSATION:
{history_text}

CURRENT QUESTION: {query}

Instructions:
- Answer based primarily on the document context provided
- If the answer is not in the documents, say so clearly
- Be concise but comprehensive
- Mention which part of the document your answer comes from
- For follow-up questions, consider the conversation history

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    return response.choices[0].message.content



st.sidebar.markdown("## ⚙️ Configuration")

api_key = st.sidebar.text_input(
    "🔑 Groq API Key",
    type="password",
    placeholder="gsk_...",
    help="Get free key at console.groq.com"
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 📄 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=['pdf', 'txt'],
    accept_multiple_files=True,
    help="Upload any documents you want to chat with"
)

process_btn = st.sidebar.button(
    "🔄 Process Documents",
    use_container_width=True,
    type="primary"
)

if process_btn and uploaded_files and api_key:
    with st.spinner("Processing documents..."):
        try:
            chroma_client.delete_collection("documents")
        except:
            pass
        collection   = chroma_client.create_collection("documents")
        total_chunks = 0
        doc_names    = []

        for uploaded_file in uploaded_files:
            st.sidebar.write(f"📄 Processing: {uploaded_file.name}")
            if uploaded_file.name.endswith('.pdf'):
                text = extract_text_from_pdf(uploaded_file)
            else:
                text = uploaded_file.read().decode('utf-8')
            chunks = chunk_text(text, chunk_size=500, overlap=50)
            n      = add_documents_to_db(chunks, uploaded_file.name, collection)
            total_chunks += n
            doc_names.append(uploaded_file.name)
            st.sidebar.success(f"✅ {uploaded_file.name}: {n} chunks")

        st.session_state.docs_loaded  = True
        st.session_state.doc_count    = len(uploaded_files)
        st.session_state.chunk_count  = total_chunks
        st.session_state.doc_names    = doc_names
        st.session_state.collection   = collection
        st.session_state.chat_history = []

elif process_btn and not api_key:
    st.sidebar.error("⚠️ Please enter your Groq API key first!")
elif process_btn and not uploaded_files:
    st.sidebar.error("⚠️ Please upload at least one document!")

if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 How to Use")
st.sidebar.markdown("""
1. Enter your Groq API key
2. Upload PDF or TXT documents
3. Click **Process Documents**
4. Ask questions about your documents!

**Example questions:**
- "What is this document about?"
- "Summarize the main points"
- "What does it say about [topic]?"
- "List the key findings"
""")



st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="title-text">🤖 RAG Document Chatbot</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Ask questions about your documents · '
            'Powered by Groq LLaMA 3.1 + ChromaDB + Sentence Transformers</p>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-label">LLM</div>'
                '<div class="metric-value">LLaMA 3.1</div></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-label">Vector DB</div>'
                '<div class="metric-value">ChromaDB</div></div>',
                unsafe_allow_html=True)
with c3:
    doc_count = st.session_state.doc_count
    st.markdown(f'<div class="metric-card"><div class="metric-label">Documents</div>'
                f'<div class="metric-value">{doc_count}</div></div>',
                unsafe_allow_html=True)
with c4:
    chunk_count = st.session_state.chunk_count
    st.markdown(f'<div class="metric-card"><div class="metric-label">Chunks</div>'
                f'<div class="metric-value">{chunk_count}</div></div>',
                unsafe_allow_html=True)

st.markdown("---")

if st.session_state.docs_loaded:
    docs_str = ", ".join(st.session_state.doc_names)
    st.markdown(f'<div class="status-ready">✅ Ready — {st.session_state.doc_count} document(s) loaded: {docs_str}</div>',
                unsafe_allow_html=True)
else:
    st.markdown('<div class="status-empty">⚠️ No documents loaded — Upload documents in the sidebar to get started</div>',
                unsafe_allow_html=True)

st.markdown("")


chat_col, info_col = st.columns([2, 1], gap="large")

with chat_col:
    with st.container():
        if not st.session_state.chat_history:
            st.markdown("""
            <div style='background:#1e2330; border:1px dashed #2d3748;
                 border-radius:12px; padding:2rem; text-align:center; margin:1rem 0'>
                <div style='font-size:2.5rem'>💬</div>
                <div style='color:#64748b; margin-top:0.8rem'>
                    Upload documents and ask your first question!<br>
                    <span style='color:#60a5fa'>The AI will answer based on YOUR documents.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                st.markdown(f"""
                <div class="chat-user">
                    <div class="chat-label-user">👤 YOU</div>
                    {msg['user']}
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="chat-bot">
                    <div class="chat-label-bot">🤖 AI ASSISTANT</div>
                    {msg['bot']}
                </div>
                """, unsafe_allow_html=True)
                if msg.get('sources'):
                    with st.expander("📚 View Sources Used", expanded=False):
                        for src in msg['sources']:
                            st.markdown(f'<div class="source-box">📄 {src}</div>',
                                        unsafe_allow_html=True)

    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question:",
            placeholder="e.g. What is the main topic of this document?",
            label_visibility="collapsed"
        )
        col1, col2 = st.columns([4, 1])
        with col1:
            submitted = st.form_submit_button("Send ➤",
                            use_container_width=True, type="primary")
        with col2:
            st.form_submit_button("Clear 🗑️", use_container_width=True)

    if submitted and question.strip():
        if not api_key:
            st.error("⚠️ Please enter your Groq API key in the sidebar!")
        elif not st.session_state.docs_loaded:
            st.error("⚠️ Please upload and process documents first!")
        else:
            with st.spinner("🔍 Searching documents and generating answer..."):
                try:
                    collection    = st.session_state.collection
                    chunks, metas = retrieve_relevant_chunks(
                        question, collection, n_results=4)
                    time.sleep(1)
                    answer  = generate_answer(
                        question, chunks,
                        st.session_state.chat_history, api_key)
                    sources = list(set([
                        f"{m['source']} (chunk {m['chunk']})"
                        for m in metas]))
                    st.session_state.chat_history.append({
                        'user':    question,
                        'bot':     answer,
                        'sources': sources
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Check your API key at console.groq.com and try again.")

with info_col:
    st.markdown("### 🧠 How RAG Works")
    st.markdown("""
    <div style='background:#1e2330; border-radius:12px; padding:1.2rem;'>
        <div style='color:#60a5fa; font-weight:700; margin-bottom:0.8rem'>
            Step 1 — Document Processing</div>
        <div style='color:#94a3b8; font-size:0.85rem; margin-bottom:1rem'>
            Documents split into 500-word chunks,
            converted to vectors and stored in ChromaDB.
        </div>
        <div style='color:#a78bfa; font-weight:700; margin-bottom:0.8rem'>
            Step 2 — Semantic Search</div>
        <div style='color:#94a3b8; font-size:0.85rem; margin-bottom:1rem'>
            Question converted to vector.
            ChromaDB finds most similar chunks
            using cosine similarity.
        </div>
        <div style='color:#4ade80; font-weight:700; margin-bottom:0.8rem'>
            Step 3 — AI Generation</div>
        <div style='color:#94a3b8; font-size:0.85rem'>
            Relevant chunks + question sent to
            LLaMA 3.1 via Groq. Answers based on
            YOUR documents only.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Session Stats")
    total_questions = len(st.session_state.chat_history)
    st.markdown(f"""
    <div style='background:#1e2330; border-radius:12px; padding:1.2rem;'>
        <div style='display:flex; justify-content:space-between; margin-bottom:0.5rem'>
            <span style='color:#64748b'>Questions asked</span>
            <span style='color:#e2e8f0; font-weight:700'>{total_questions}</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:0.5rem'>
            <span style='color:#64748b'>Documents loaded</span>
            <span style='color:#e2e8f0; font-weight:700'>{st.session_state.doc_count}</span>
        </div>
        <div style='display:flex; justify-content:space-between'>
            <span style='color:#64748b'>Chunks indexed</span>
            <span style='color:#e2e8f0; font-weight:700'>{st.session_state.chunk_count}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.doc_names:
        st.markdown("### 📁 Loaded Documents")
        for name in st.session_state.doc_names:
            st.markdown(f"""
            <div style='background:#1e2330; border-radius:8px;
                 padding:0.5rem 0.8rem; margin:0.3rem 0;
                 border-left:3px solid #60a5fa;
                 color:#94a3b8; font-size:0.85rem'>
                📄 {name}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 💡 Sample Questions")
    for q in ["What is this document about?",
              "Summarize the key points",
              "What are the main findings?",
              "What methodology was used?",
              "List the conclusions"]:
        st.markdown(f"""
        <div style='background:#13161e; border:1px solid #2d3748;
             border-radius:8px; padding:0.4rem 0.8rem; margin:0.3rem 0;
             color:#64748b; font-size:0.82rem'>
            💬 {q}
        </div>
        """, unsafe_allow_html=True)


st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#334155; font-size:0.8rem; padding:1rem 0'>
    Built with ❤️ 
    <a href='https://github.com/Akhiliny99/Rag-Chatbot' style='color:#60a5fa'>View on GitHub</a>
</div>

""", unsafe_allow_html=True)
