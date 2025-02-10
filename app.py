import os
import tempfile
from typing import List, Dict, Literal, Optional
from dataclasses import dataclass

import streamlit as st

# Document loading and splitting
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Vector store and free open source embeddings (using HuggingFace)
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

# Import Groq LLM using the correct ChatGroq as per Groq documentation
from langchain_groq import ChatGroq

# Qdrant client for managing collections
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# For query routing using a simple agent
from phi.agent import Agent

# For fallback web research and retrieval chain
from langchain.llms.base import BaseLLM  # Updated import for base language model
from langchain.schema import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

# For constructing a QA chain using context
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.chat import ChatPromptTemplate

# For creating a ReAct agent fallback using the current LangChain agents API
from langchain.agents import initialize_agent, AgentType, Tool

def init_session_state():
    """Initialize session state variables"""
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = ""
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = ""
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = ""
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'databases' not in st.session_state:
        st.session_state.databases = {}

init_session_state()

# Define available database types
DatabaseType = Literal["products", "Support", "finance"]
PERSIST_DIRECTORY = "db_storage"

@dataclass
class CollectionConfig:
    name: str
    description: str
    collection_name: str  # Used as the Qdrant collection name

# Collection configurations for routing queries
COLLECTIONS: Dict[DatabaseType, CollectionConfig] = {
    "products": CollectionConfig(
        name="Product Information",
        description="Product details, specifications, and features",
        collection_name="products_collection"
    ),
    "Support": CollectionConfig(
        name="Customer Support & FAQ",
        description="Customer support information, frequently asked questions, and guides",
        collection_name="support_collection"
    ),
    "finance": CollectionConfig(
        name="Financial Information",
        description="Financial data, revenue, costs, liabilities, and reports",
        collection_name="finance_collection"
    )
}

def initialize_models() -> bool:
    """Initialize ChatGroq LLM, free open source embeddings, and Qdrant client"""
    if (st.session_state.groq_api_key and 
        st.session_state.qdrant_url and 
        st.session_state.qdrant_api_key):
        
        os.environ["GROQ_API_KEY"] = st.session_state.groq_api_key
        
        # Use Hugging Face’s free open source embeddings model
        st.session_state.embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large"
        )
        
        # Initialize ChatGroq as the LLM (using your desired model name and temperature)
        st.session_state.llm = ChatGroq(
            model_name="mixtral-8x7b-32768",
            temperature=0
        )
        
        try:
            # Initialize Qdrant client with provided credentials
            client = QdrantClient(
                url=st.session_state.qdrant_url,
                api_key=st.session_state.qdrant_api_key
            )
            
            # Test connection by retrieving collections
            client.get_collections()
            vector_size = 768
            st.session_state.databases = {}
            for d_type, config in COLLECTIONS.items():
                try:
                    client.get_collection(config.collection_name)
                except Exception:
                    # Create the collection if it does not exist
                    client.create_collection(
                        collection_name=config.collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                    )
                
                st.session_state.databases[d_type] = Qdrant(
                    client=client,
                    collection_name=config.collection_name,
                    embeddings=st.session_state.embeddings
                )
            
            return True
        except Exception as e:
            st.error(f"Failed to connect to Qdrant: {str(e)}")
            return False
    return False

def process_document(file) -> List[Document]:
    """Process an uploaded PDF document, split it into chunks, and return as a list of Document objects"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
            
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Remove the temporary file
        os.unlink(tmp_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        return texts
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return []

def create_routing_agent() -> Agent:
    """Creates a query routing agent using ChatGroq to decide which database to route a question to"""
    return Agent(
        model=ChatGroq(
            model_name="mixtral-8x7b-32768",
            api_key=st.session_state.groq_api_key
        ),
        tools=[],
        description=(
            "You are a query routing expert. Your only job is to analyze questions and determine "
            "which database they should be routed to. You must respond with exactly one of these three "
            "options: 'products', 'Support', or 'finance'. The user's question is: {question}"
        ),
        instructions=[
            "Follow these rules strictly:",
            "1. For questions about product, features, specifications, or item details, or product manuals → return 'products'",
            "2. For questions about help, guidance, troubleshooting, customer service, FAQ, or guides → return 'Support'",
            "3. For questions about costs, revenue, pricing, financial data, financial reports, or investment → return 'finance'",
            "4. Return ONLY the database name, no other text or explanation",
            "5. If you're not confident about the routing, return an empty response"
        ],
        markdown=False,
        show_tool_calls=False
    )

def route_query(question: str) -> Optional[DatabaseType]:
    """Route query by comparing similarity scores from each database; falls back to LLM routing if necessary."""
    try:
        best_score = -1
        best_db_type = None
        
        # Evaluate similarity score across each database
        for db_type, db in st.session_state.databases.items():
            results = db.similarity_search_with_score(question, k=3)
            if results:
                avg_score = sum(score for _, score in results) / len(results)
                if avg_score > best_score:
                    best_score = avg_score
                    best_db_type = db_type
        
        confidence_threshold = 0.5
        if best_score >= confidence_threshold and best_db_type:
            st.success(f"Using vector similarity routing: {best_db_type} (confidence: {best_score:.3f})")
            return best_db_type
            
        st.warning(f"Low confidence scores (below {confidence_threshold}), falling back to LLM routing")
        
        # Fallback: use LLM routing decision via the routing agent
        routing_agent = create_routing_agent()
        response = routing_agent.run(question)
        db_type = (response.content.strip().lower().translate(str.maketrans('', '', '`\'"')))
        
        if db_type in COLLECTIONS:
            st.success(f"Using LLM routing decision: {db_type}")
            return db_type
            
        st.warning("No suitable database found, will use web search fallback")
        return None
        
    except Exception as e:
        st.error(f"Routing error: {str(e)}")
        return None

def create_fallback_agent(chat_model: BaseLLM):
    """Create a ReAct agent for fallback web research."""
    def web_research(query: str) -> str:
        """Perform a DuckDuckGo search and format the results."""
        try:
            search = DuckDuckGoSearchRun(num_results=5)
            results = search.run(query)
            return results
        except Exception as e:
            return f"Search failed: {str(e)}. Providing answer based on general knowledge."
    
    tool = Tool(
        name="WebSearch",
        func=web_research,
        description="Useful for answering questions that require up-to-date information via web search."
    )
    tools = [tool]
    agent = initialize_agent(
        tools=tools,
        llm=chat_model,
        agent=AgentType.REACT,
        verbose=False
    )
    return agent

def query_database(db: Qdrant, question: str) -> tuple[str, list]:
    """Query the selected database, retrieve relevant documents, and generate an answer based on context."""
    try:
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        relevant_docs = retriever.get_relevant_documents(question)
        
        if relevant_docs:
            # Create a prompt template to instruct the LLM how to answer using the provided context
            retrieval_qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant that answers questions based on provided context. Always be direct and concise in your responses. If the context doesn't contain enough information to fully answer the question, acknowledge this limitation. Base your answers strictly on the provided context and avoid making assumptions."),
                ("human", "Here is the context:\n{context}"),
                ("human", "Question: {input}"),
                ("assistant", "I'll help answer your question based on the context provided."),
                ("human", "Please provide your answer:")
            ])
            
            # Create the QA chain using the "stuff" method to combine document chunks
            chain = load_qa_chain(
                st.session_state.llm,
                chain_type="stuff",
                prompt=retrieval_qa_prompt
            )
            response = chain({"input_documents": relevant_docs, "input": question})
            return response["output_text"], relevant_docs
        
        raise ValueError("No relevant documents found in database")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "I encountered an error. Please try rephrasing your question.", []

def _handle_web_fallback(question: str) -> tuple[str, list]:
    st.info("No relevant documents found. Searching the web...")
    fallback_agent = create_fallback_agent(st.session_state.llm)
    
    with st.spinner('Researching...'):
        try:
            answer = fallback_agent.run(f"Research and provide a detailed answer for: '{question}'")
            return f"Web Search Result:\n{answer}", []
        except Exception:
            fallback_response = st.session_state.llm.invoke(question).content
            return f"Web search unavailable. General response: {fallback_response}", []
    
def main():
    """Main application function for the RAG Agent with Database Routing."""
    st.set_page_config(page_title="RAG Agent with Database Routing", page_icon="📚")
    st.title("📠 RAG Agent with Database Routing")
    
    # Sidebar configuration for API keys and Qdrant settings
    with st.sidebar:
        st.header("Configuration")
        
        groq_api_key = st.text_input(
            "Enter Groq API Key:",
            type="password",
            value=st.session_state.groq_api_key,
            key="groq_api_key_input"
        )
        qdrant_url = st.text_input(
            "Enter Qdrant URL:",
            value=st.session_state.qdrant_url,
            help="Example: https://your-cluster.qdrant.tech"
        )
        qdrant_api_key = st.text_input(
            "Enter Qdrant API Key:",
            type="password",
            value=st.session_state.qdrant_api_key
        )
        
        # Update session state with credentials
        if groq_api_key:
            st.session_state.groq_api_key = groq_api_key
        if qdrant_url:
            st.session_state.qdrant_url = qdrant_url
        if qdrant_api_key:
            st.session_state.qdrant_api_key = qdrant_api_key
            
        # Initialize models once all credentials are provided
        if (st.session_state.groq_api_key and 
            st.session_state.qdrant_url and 
            st.session_state.qdrant_api_key):
            if initialize_models():
                st.success("Connected to Groq and Qdrant successfully!")
            else:
                st.error("Failed to initialize. Please check your credentials.")
        else:
            st.warning("Please enter all required credentials to continue")
            st.stop()
        
        st.markdown("---")
    
    st.header("Document Upload")
    st.info("Upload documents to populate the databases. Each tab corresponds to a different database.")
    tabs = st.tabs([collection_config.name for collection_config in COLLECTIONS.values()])
    
    for (collection_type, collection_config), tab in zip(COLLECTIONS.items(), tabs):
        with tab:
            st.write(collection_config.description)
            uploaded_files = st.file_uploader(
                f"Upload PDF documents to {collection_config.name}",
                type="pdf",
                key=f"upload_{collection_type}",
                accept_multiple_files=True
            )
            if uploaded_files:
                with st.spinner('Processing documents...'):
                    all_texts = []
                    for uploaded_file in uploaded_files:
                        texts = process_document(uploaded_file)
                        all_texts.extend(texts)
                    
                    if all_texts:
                        db = st.session_state.databases[collection_type]
                        db.add_documents(all_texts)
                        st.success("Documents processed and added to the database!")
    
    # Query section for user questions
    st.header("Ask Questions")
    st.info("Enter your question below to find answers from the relevant database.")
    question = st.text_input("Enter your question:")
    
    if question:
        with st.spinner('Finding answer...'):
            # Determine the best database to query based on the question
            collection_type = route_query(question)
            if collection_type is None:
                # Use web search fallback if no appropriate database is found
                answer, _ = _handle_web_fallback(question)
                st.write("### Answer (from web search)")
                st.write(answer)
            else:
                st.info(f"Routing question to: {COLLECTIONS[collection_type].name}")
                db = st.session_state.databases[collection_type]
                answer, _ = query_database(db, question)
                st.write("### Answer")
                st.write(answer)

if __name__ == "__main__":
    main()
