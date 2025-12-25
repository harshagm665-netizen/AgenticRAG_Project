# rag_tools.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables or .env file")

# Initialize LLM with custom endpoint
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key,
    base_url="https://openai.vocareum.com/v1"
)

# Define a prompt template for QA
qa_template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:"""

qa_prompt = ChatPromptTemplate.from_template(qa_template)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def rag_retriever_tool(query, vectorstore):
    """
    Tool 1: Standard RAG Retriever.
    Searches the vector store and returns an answer based on context.
    """
    if not vectorstore:
        return "Error: Vector store not initialized."
    
    try:
        # Get retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Retrieve documents using invoke
        docs = retriever.invoke(query)
        
        if not docs:
            return "No relevant documents found."
        
        # Format context
        context = format_docs(docs)
        
        # Create the chain
        chain = qa_prompt | llm | StrOutputParser()
        
        # Get answer
        answer = chain.invoke({"context": context, "question": query})
        
        return answer
    except Exception as e:
        return f"Error during retrieval: {str(e)}"

def filter_tool(query, vectorstore, target_filename):
    """
    Tool 2: Filter Tool.
    Applies a metadata filter to search ONLY within a specific file.
    """
    if not vectorstore:
        return "Error: Vector store not initialized."
    
    try:
        # Metadata filter dictionary
        filter_dict = {"source": target_filename}
        
        # Get retriever with filter
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3, "filter": filter_dict}
        )
        
        # Retrieve documents using invoke
        docs = retriever.invoke(query)
        
        if not docs:
            return f"No documents found from file: {target_filename}"
        
        # Format context
        context = format_docs(docs)
        
        # Create the chain
        chain = qa_prompt | llm | StrOutputParser()
        
        # Get answer
        answer = chain.invoke({"context": context, "question": query})
        
        return answer
    except Exception as e:
        return f"Error during filtered retrieval: {str(e)}"

def elaborator_tool(query, vectorstore):
    """
    Tool 3: Elaborator Tool.
    Uses the LLM to expand a vague query into a detailed one before searching.
    """
    try:
        # Step 1: Refine the query
        refine_prompt = f"Refine the following vague user query to be more technical, precise, and detailed for a semantic search engine. Only output the refined query, nothing else: '{query}'"
        refined_query = llm.invoke(refine_prompt).content
        
        # Step 2: Use the refined query for standard retrieval
        answer = rag_retriever_tool(refined_query, vectorstore)
        
        return f"**Refined Query:** {refined_query}\n\n**Answer:** {answer}"
    except Exception as e:
        return f"Error during elaboration: {str(e)}"