import os
import argparse
from ingestion import ingest_files
from retrieval import RagSystem

def main():
    parser = argparse.ArgumentParser(description="Vector DB and RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest files")
    ingest_parser.add_argument("--content-dir", required=True, help="Directory containing files to ingest")
    ingest_parser.add_argument("--output-dir", default="./chroma_db", help="Directory to store vector database")
    ingest_parser.add_argument("--collection-name", default="knowledge_base", help="Name for the vector database collection")
    ingest_parser.add_argument("--chunk-size", type=int, default=1000, help="Size of text chunks for splitting")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("--db-path", required=True, help="Path to vector database")
    query_parser.add_argument("--question", help="Question to ask (if not provided, enters interactive mode)")
    query_parser.add_argument("--api-key", help="LLM API key (if not set as env var)")
    query_parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"], help="LLM provider to use")
    query_parser.add_argument("--model", default="claude-3-7-sonnet-20250219", help="Model name to use")
    query_parser.add_argument("--system-prompt", default="You are an AI assistant.", help="System prompt for the LLM")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, args.collection_name), exist_ok=True)
        
        db_path = os.path.join(args.output_dir, args.collection_name)
        
        print(f"Ingesting files from {args.content_dir}...")
        db, persist_dir = ingest_files(
            directory_path=args.content_dir, 
            collection_name=args.collection_name,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        print(f"Ingestion complete. Vector database stored in: {persist_dir}")
        
    elif args.command == "query":
        # Initialize RAG system
        rag = RagSystem(
            db_path=args.db_path,
            llm_provider=args.provider,
            model_name=args.model,
            api_key=args.api_key,
            system_prompt=args.system_prompt
        )
        
        if args.question:
            # Single question mode
            answer = rag.query(args.question)
            print(answer)
        else:
            # Interactive mode
            rag.interactive_mode()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()