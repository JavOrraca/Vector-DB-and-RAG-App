#!/usr/bin/env python3
"""
Vector DB Wrapper Script

This script provides a wrapper around the vector database functions,
designed to be called from R via system2() rather than reticulate.
"""

import sys
import os
import json
import traceback
from pathlib import Path

# Add the src directory to sys.path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import our functions
from ingestion import ingest_files

def create_vector_db(input_dir, output_dir, collection_name, chunk_size, chunk_overlap):
    """
    Create a vector database from the files in input_dir.
    
    Args:
        input_dir: Directory containing files to ingest
        output_dir: Directory to store the vector database
        collection_name: Name for the collection
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
        
    Returns:
        JSON string with result information
    """
    try:
        print(f"Creating vector DB with parameters:")
        print(f"- Input directory: {input_dir}")
        print(f"- Output directory: {output_dir}")
        print(f"- Collection name: {collection_name}")
        print(f"- Chunk size: {chunk_size}")
        print(f"- Chunk overlap: {chunk_overlap}")
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Call the ingest_files function
        result_db, persist_dir = ingest_files(
            directory_path=input_dir,
            collection_name=collection_name,
            output_dir=output_dir,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap)
        )
        
        return json.dumps({
            "success": True,
            "db_path": persist_dir
        })
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback_str}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback_str
        })

def retrieve_context(db_path, query, k=5):
    """
    Retrieve context from the vector database for a given query.
    
    Args:
        db_path: Path to the vector database
        query: Query string
        k: Number of documents to retrieve
        
    Returns:
        JSON string with context information
    """
    try:
        print(f"Retrieving context for query: {query}")
        print(f"Vector DB path: {db_path}")
        print(f"Number of documents to retrieve: {k}")
        
        # Import required modules
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Load embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load vector database
        db = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_model
        )
        
        # Retrieve relevant documents
        results = db.similarity_search_with_score(query, k=int(k))
        
        # Extract and format context
        context_list = []
        for doc, score in results:
            # Pre-escape the content to handle Markdown code blocks with backticks
            # This ensures backticks and other special characters don't break the JSON structure
            escaped_content = json.dumps(doc.page_content)[1:-1]  # Remove the quotes added by dumps
            
            context_list.append({
                "source": doc.metadata.get("source", "Unknown"),
                "file_type": doc.metadata.get("file_type", "Unknown"),
                "content": escaped_content,
                "score": float(score)
            })
        
        return json.dumps({
            "success": True,
            "context": context_list
        })
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback_str}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback_str
        })

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else None
    
    if command == "create_vector_db":
        if len(sys.argv) < 7:
            print("Usage: python vector_db_wrapper.py create_vector_db input_dir output_dir collection_name chunk_size chunk_overlap")
            sys.exit(1)
        
        input_dir = sys.argv[2]
        output_dir = sys.argv[3]
        collection_name = sys.argv[4]
        chunk_size = sys.argv[5]
        chunk_overlap = sys.argv[6]
        
        print(create_vector_db(input_dir, output_dir, collection_name, chunk_size, chunk_overlap))
    
    elif command == "retrieve_context":
        if len(sys.argv) < 4:
            print("Usage: python vector_db_wrapper.py retrieve_context db_path query [k]")
            sys.exit(1)
        
        db_path = sys.argv[2]
        query = sys.argv[3]
        k = sys.argv[4] if len(sys.argv) > 4 else 5
        
        print(retrieve_context(db_path, query, k))
    
    else:
        print("Unknown command. Available commands: create_vector_db, retrieve_context")
        sys.exit(1)