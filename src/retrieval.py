from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os

class RagSystem:
    """
    RAG system for querying documentation and code from various file types.
    """
    
    def __init__(self, db_path, llm_provider="anthropic", model_name="claude-3-7-sonnet-20250219", 
                api_key=None, system_prompt=None):
        """
        Initialize the RAG system with a path to the vector database.
        
        Args:
            db_path: Path to the Chroma DB for embeddings
            llm_provider: LLM provider to use ('anthropic' or 'openai')
            model_name: Name of the model to use
            api_key: Optional API key (otherwise uses environment variables)
            system_prompt: Optional system prompt for the LLM
        """
        # Load embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load vector database
        self.db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embedding_model
        )
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = "You are an AI assistant."
        
        # Initialize LLM based on provider
        if llm_provider.lower() == "anthropic":
            # Get API key from args or environment
            anthropic_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                raise ValueError("No Anthropic API key provided. Set it with api_key parameter or ANTHROPIC_API_KEY environment variable.")
            
            self.llm = ChatAnthropic(
                temperature=0,
                model=model_name,
                anthropic_api_key=anthropic_api_key,
                system=system_prompt
            )
        elif llm_provider.lower() == "openai":
            # Get API key from args or environment
            openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("No OpenAI API key provided. Set it with api_key parameter or OPENAI_API_KEY environment variable.")
            
            self.llm = ChatOpenAI(
                temperature=0,
                model=model_name,
                openai_api_key=openai_api_key,
                model_kwargs={"messages": [{"role": "system", "content": system_prompt}]}
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}. Use 'anthropic' or 'openai'.")
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Use the provided context to answer the user's question.
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer:
            """
        )
    
    def query(self, question, k=5):
        """
        Query the RAG system with a question.
        
        Args:
            question: User's question
            k: Number of documents to retrieve from the database
            
        Returns:
            Answer from the LLM
        """
        # Retrieve relevant documents
        results = self.db.similarity_search_with_score(question, k=k)
        
        # Extract documents
        context_docs = [doc for doc, _ in results]
        
        # Build context string
        context_str = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')} | Type: {doc.metadata.get('file_type', 'Unknown')}]\n{doc.page_content}"
            for doc in context_docs
        ])
        
        # Create chain
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": k}),
            chain_type_kwargs={"prompt": self.prompt_template}
        )
        
        # Run chain
        result = chain.invoke({
            "query": question,
            "context": context_str
        })
        
        return result["result"]
    
    def interactive_mode(self):
        """
        Start an interactive session for querying the RAG system.
        """
        print("RAG System - Interactive Mode")
        print("Type 'exit' to quit")
        
        while True:
            question = input("\nEnter your question: ")
            
            if question.lower() == 'exit':
                break
                
            try:
                answer = self.query(question)
                print("\nAnswer:")
                print(answer)
            except Exception as e:
                print(f"Error: {e}")

# For backward compatibility
class RPackageRagSystem(RagSystem):
    """
    Legacy class for backward compatibility. Use RagSystem instead.
    """
    
    def __init__(self, docs_db_path, code_db_path, anthropic_api_key=None):
        """
        Initialize the RAG system with paths to the vector databases.
        
        Args:
            docs_db_path: Path to the Chroma DB for markdown documentation
            code_db_path: Path to the Chroma DB for R code
            anthropic_api_key: Optional Anthropic API key
        """
        print("Warning: RPackageRagSystem is deprecated. Use RagSystem instead.")
        
        # Use the main db path (they should be the same in new version, but respect legacy behavior)
        super().__init__(
            db_path=docs_db_path,
            llm_provider="anthropic", 
            model_name="claude-3-7-sonnet-20250219",
            api_key=anthropic_api_key,
            system_prompt="You are an expert R programmer and data scientist. Use the provided context about R packages to answer the user's question."
        )

if __name__ == "__main__":
    # Example usage
    db_path = "./chroma_db/knowledge_base"
    
    # Initialize RAG system
    rag = RagSystem(db_path)
    
    # Start interactive mode
    rag.interactive_mode()