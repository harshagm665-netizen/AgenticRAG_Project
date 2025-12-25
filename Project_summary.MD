# Agentic RAG System - Project Summary

## 📋 Overview

This project implements an intelligent Retrieval-Augmented Generation (RAG) system with three specialized agent tools and comprehensive safety guardrails. The system demonstrates agentic behavior through automatic query routing and maintains security through dual-stage content filtering.

---

## 📁 Project Deliverables

### 1. **Project Report (3-4 pages)**
   - **File:** `Agentic_RAG_Project_Report.docx`
   - **Contents:**
     - Section 1: System Architecture & Core Components
     - Section 2: Detailed Tool Flow Description
     - Section 3: Challenges and Key Learnings
     - Section 4: Conclusion and Future Work

### 2. **Architecture Diagram**
   - **File:** `architecture_diagram.png`
   - **Description:** Visual representation of system layers, data flow, and component interactions

### 3. **Source Code**
   - `app.py` - Main Streamlit application
   - `rag_tools.py` - Three agent tools implementation
   - `safety.py` - Guardrail system
   - `requirements.txt` - Dependencies
   - `.env` - API configuration template

### 4. **Documentation**
   - `DEMO_SCRIPT.md` - Step-by-step demo guide (~13 minutes)
   - `QUICKSTART.md` - Setup instructions
   - `TROUBLESHOOTING.md` - Common issues and solutions
   - `SETUP.md` - Detailed configuration guide

### 5. **Test Data**
   - `secret_project.txt` - Sample file for safety testing

---

## 🏗️ System Architecture

### **Three-Layer Design**

1. **Data Ingestion Layer**
   - Multi-format document loading (PDF, TXT, MD)
   - Configurable chunking strategies
   - Metadata normalization for source tracking

2. **Agent Orchestration Layer**
   - Keyword-based query routing
   - Three specialized tools:
     - **Standard RAG Retriever**: Direct semantic search
     - **Elaborator Tool**: Query refinement for vague questions
     - **Filter Tool**: Source-specific document retrieval

3. **Safety & Logging Layer**
   - Input/output guardrails with keyword filtering
   - Complete audit trail of all operations
   - Real-time transparency dashboard

---

## 🛠️ Key Technologies

- **LLM:** GPT-4o-mini (Vocareum endpoint)
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector Store:** FAISS
- **Framework:** LangChain (LCEL syntax)
- **Interface:** Streamlit
- **Language:** Python 3.8+

---

## 🎯 Agent Tools

### 1. Standard RAG Retriever
**Purpose:** Direct question answering  
**Trigger:** Default (no special keywords)  
**Process:** Query → Vector Search (k=3) → LLM Generation

### 2. Elaborator Tool
**Purpose:** Handle vague/underspecified queries  
**Trigger Keywords:** carefully, detailed, thoroughly, explain  
**Process:** Query Refinement → Enhanced Search → Comprehensive Answer

### 3. Filter Tool
**Purpose:** Source-specific document retrieval  
**Trigger Patterns:** "from [filename]" or "file:[filename]"  
**Process:** Metadata Filter → Scoped Search → Attributed Answer

---

## 🛡️ Safety Guardrails

### Dual-Stage Filtering

**Input Guardrail:**
- Validates user queries before processing
- Blocks malicious or policy-violating inputs
- Prevents resource waste on unsafe queries

**Output Guardrail:**
- Validates generated responses
- Prevents leakage of sensitive information
- Catches context-based violations

### Banned Keywords
- confidential
- secret
- classified
- internal use only
- restricted

---

## 📊 Key Features

✅ **Intelligent Routing** - Automatic tool selection based on query characteristics  
✅ **Safety First** - Dual-stage content filtering with audit logs  
✅ **Source Attribution** - Metadata-based filtering for compliance  
✅ **Transparency** - Complete operation logging and monitoring  
✅ **Flexibility** - Configurable chunking and retrieval parameters  
✅ **Production Ready** - Custom endpoint support, error handling, user feedback

---

## 🚀 Demo Flow

1. **Data Setup** (2 min) - Upload documents and view statistics
2. **Chunking** (2 min) - Configure and preview chunk strategies
3. **Tool Testing** (5 min) - Test all three agent tools
4. **Safety Demo** (2 min) - Trigger guardrails and show blocking
5. **Logs Review** (2 min) - Review audit trail and transparency

**Total Demo Time:** ~13 minutes

---

## 💡 Key Learnings

### 1. **Framework Evolution**
LangChain's rapid changes required migration to stable LCEL syntax and core APIs rather than high-level abstractions.

### 2. **Custom Endpoints**
Vocareum integration taught the importance of checking current parameter names and explicit string passing for API keys.

### 3. **Metadata Management**
Normalizing metadata during ingestion (not relying on loader defaults) ensures predictable filtering behavior.

### 4. **Defense in Depth**
Multiple security checkpoints (input + output) provide redundancy and catch different violation types.

### 5. **Chunking Trade-offs**
500 characters with 50-character overlap balances context preservation and retrieval precision for technical documents.

---

## 🔮 Future Enhancements

1. **LLM-Powered Routing** - Replace keyword detection with intent classification
2. **Semantic Guardrails** - Move beyond keywords to similarity-based content filtering
3. **Conversation Memory** - Support multi-turn interactions with context
4. **Production Scale** - Migrate to Pinecone/Weaviate for large-scale deployments
5. **Hybrid Search** - Combine dense and sparse retrieval methods
6. **Citation Tracking** - Show which chunks informed each answer
7. **ML-Based Safety** - Develop nuanced content filtering beyond keywords

---

## 📈 Real-World Applications

- **Enterprise Knowledge Management** - Compliance-aware document search
- **Legal Analysis** - Source-specific research with audit trails
- **Research Paper Exploration** - Academic literature review and synthesis
- **Customer Support** - Policy-grounded answer generation
- **Regulatory Compliance** - Verification and audit trail generation

---

## 📝 Files Overview

```
AgenticRAG_Project/
├── app.py                              # Main application
├── rag_tools.py                        # Agent tools
├── safety.py                           # Guardrail system
├── requirements.txt                    # Dependencies
├── .env                                # API configuration
├── Agentic_RAG_Project_Report.docx    # Full report (3-4 pages)
├── architecture_diagram.png            # Visual architecture
├── secret_project.txt                  # Test data
├── DEMO_SCRIPT.md                      # Demo guide
├── QUICKSTART.md                       # Setup instructions
├── TROUBLESHOOTING.md                  # Issue resolution
└── SETUP.md                            # Configuration guide
```

---

## ⚡ Quick Start

```bash
# 1. Set API key
echo "OPENAI_API_KEY=your_key" > .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run application
streamlit run app.py
```

---

## 🏆 Project Highlights

- ✅ **Complete System** - End-to-end RAG with safety controls
- ✅ **Agentic Behavior** - Intelligent routing and tool selection
- ✅ **Production Quality** - Error handling, logging, user feedback
- ✅ **Well Documented** - Comprehensive guides and demo script
- ✅ **Extensible Architecture** - Modular design for easy customization

---

## 📧 Contact & Support

For questions or issues:
1. Review `TROUBLESHOOTING.md` for common problems
2. Check `DEMO_SCRIPT.md` for usage examples
3. Consult `SETUP.md` for configuration details

---

**Last Updated:** December 2025  
**Version:** 1.0  
**Status:** Production Ready ✅