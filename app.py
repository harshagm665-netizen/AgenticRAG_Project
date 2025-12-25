# app.py
import streamlit as st
import os
import time
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Import our custom modules
from safety import check_safety
from rag_tools import rag_retriever_tool, filter_tool, elaborator_tool

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Agentic RAG Benchmark", layout="wide")
st.title("🛡️ Agentic RAG System with Guardrails")

# --- SESSION STATE INITIALIZATION ---
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "docs" not in st.session_state:
    st.session_state.docs = []
if "logs" not in st.session_state:
    st.session_state.logs = []

# --- TABS LAYOUT ---
tab1, tab2, tab3, tab4 = st.tabs(["📂 Data Setup", "⚙️ Chunking", "🤖 Agent Interface", "📊 Logs & Safety"])

# ==========================
# TAB 1: DATA LOADING
# ==========================
with tab1:
    st.header("1. Load Documents")
    uploaded_files = st.file_uploader("Upload PDF, TXT, or MD files", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process & Load Documents"):
            loaded_docs = []
            
            # Temporary directory for processing
            if not os.path.exists("temp_data"):
                os.makedirs("temp_data")
            
            total_files = 0
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join("temp_data", uploaded_file.name)
                
                # Save file to disk temporarily
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Select appropriate loader
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif uploaded_file.name.endswith(".md"):
                    loader = UnstructuredMarkdownLoader(file_path)
                else:
                    loader = TextLoader(file_path)
                
                # Load and enforce metadata
                try:
                    docs = loader.load()
                    for doc in docs:
                        # normalize filename in metadata for filtering
                        doc.metadata["source"] = uploaded_file.name
                    loaded_docs.extend(docs)
                    total_files += 1
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {e}")

            st.session_state.docs = loaded_docs
            st.success(f"Successfully loaded {total_files} files ({len(loaded_docs)} pages/documents).")
            
            # Display File Stats
            st.subheader("File Summary")
            file_stats = {}
            for doc in loaded_docs:
                name = doc.metadata.get("source", "unknown")
                file_stats[name] = file_stats.get(name, 0) + len(doc.page_content.split())
            
            st.table([{"File Name": k, "Word Count (Approx)": v} for k, v in file_stats.items()])

# ==========================
# TAB 2: CHUNKING & EMBEDDING
# ==========================
with tab2:
    st.header("2. Chunking Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        strategy = st.radio("Chunking Strategy", ["Recursive Character", "Fixed Size (Character)"])
    with col2:
        chunk_size = st.slider("Chunk Size", 100, 2000, 500)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 50)
    
    if st.button("Preview Chunks"):
        if not st.session_state.docs:
            st.error("No documents loaded yet.")
        else:
            if strategy == "Recursive Character":
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            else:
                splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            splits = splitter.split_documents(st.session_state.docs[:1]) # Preview first doc only
            st.write(f"**Preview (First 3 chunks of first document):**")
            for i, split in enumerate(splits[:3]):
                st.info(f"Chunk {i+1} (Source: {split.metadata['source']}):\n\n{split.page_content[:200]}...")

    st.divider()
    
    if st.button("⚡ Build Vector Store (FAISS)"):
        if not st.session_state.docs:
            st.error("Please load documents first.")
        else:
            with st.spinner("Chunking and Embedding..."):
                # 1. Split
                if strategy == "Recursive Character":
                    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                else:
                    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                
                all_splits = splitter.split_documents(st.session_state.docs)
                
                # 2. Embed with custom endpoint
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    st.error("OPENAI_API_KEY not found. Please set it in your .env file.")
                    
                
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    api_key=api_key,
                    base_url="https://openai.vocareum.com/v1"
                )
                vectorstore = FAISS.from_documents(all_splits, embeddings)
                
                st.session_state.vectors = vectorstore
                st.success(f"Vector Store Ready! Indexed {len(all_splits)} chunks.")

# ==========================
# TAB 3: AGENT INTERFACE
# ==========================
with tab3:
    st.header("3. Query the Agent")
    st.markdown("Use natural language. Try keywords like **'carefully'** for elaboration, or **'from filename.pdf'** to filter.")
    
    user_query = st.text_input("Enter your query:")
    
    if st.button("🚀 Run Agent"):
        if not st.session_state.vectors:
            st.error("Vector store not initialized. Go to 'Chunking' tab.")
        elif not user_query:
            st.warning("Please enter a query.")
        else:
            # --- STEP 1: INPUT GUARDRAIL ---
            is_safe_in, word_in = check_safety(user_query)
            if not is_safe_in:
                st.error(f"🛑 Output generation stopped due to policy violation (Guardrail triggered by: '{word_in}').")
                st.session_state.logs.append({"timestamp": time.strftime("%H:%M:%S"), "tool": "Guardrail (Input)", "status": "BLOCKED", "trigger": word_in})
            else:
                start_time = time.time()
                tool_used = "Unknown"
                response_text = ""
                
                # --- STEP 2: AGENT ROUTING LOGIC ---
                query_lower = user_query.lower()
                
                # Logic: Elaborator -> Filter -> Retriever
                if any(x in query_lower for x in ["carefully", "detailed", "thoroughly", "explain"]):
                    tool_used = "Elaborator Tool"
                    st.info(f"🤖 Agent: Detected complex intent. Using **{tool_used}**.")
                    response_text = elaborator_tool(user_query, st.session_state.vectors)
                    
                elif "from " in query_lower or "file:" in query_lower:
                    tool_used = "Filter Tool"
                    # Basic extraction logic
                    try:
                        if "from " in query_lower:
                            # Split by 'from ', take the right side, take the first word
                            target_file = user_query.split("from ")[1].strip().split(" ")[0]
                        else:
                            target_file = user_query.split("file:")[1].strip().split(" ")[0]
                        
                        st.info(f"🤖 Agent: Detected file constraint. Using **{tool_used}** on '{target_file}'.")
                        response_text = filter_tool(user_query, st.session_state.vectors, target_file)
                    except:
                        response_text = "Error: Could not parse filename. Use format 'from filename.pdf'."
                        
                else:
                    tool_used = "RAG Retriever Tool"
                    st.info(f"🤖 Agent: Standard query. Using **{tool_used}**.")
                    response_text = rag_retriever_tool(user_query, st.session_state.vectors)

                # --- STEP 3: OUTPUT GUARDRAIL ---
                is_safe_out, word_out = check_safety(response_text)
                elapsed = round(time.time() - start_time, 2)
                
                if not is_safe_out:
                    st.error(f"🛑 Output generation stopped due to policy violation (Guardrail triggered by output: '{word_out}').")
                    st.session_state.logs.append({"timestamp": time.strftime("%H:%M:%S"), "tool": tool_used, "status": "BLOCKED (Output)", "trigger": word_out, "time": elapsed})
                else:
                    st.markdown("### Answer:")
                    st.write(response_text)
                    st.session_state.logs.append({"timestamp": time.strftime("%H:%M:%S"), "tool": tool_used, "status": "SUCCESS", "trigger": "-", "time": elapsed})

# ==========================
# TAB 4: LOGS
# ==========================
with tab4:
    st.header("4. System Logs & Safety Monitor")
    if st.session_state.logs:
        st.table(st.session_state.logs)
    else:
        st.write("No activity recorded yet.")