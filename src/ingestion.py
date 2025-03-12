import os
import glob
import json
import tempfile
import pandas as pd
import numpy as np
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, 
    CSVLoader, 
    JSONLoader, 
    BSHTMLLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    NotebookLoader
)
import pickle
try:
    import pyreadr
except ImportError:
    pyreadr = None

def ingest_files(directory_path, collection_name="knowledge_base", output_dir="./chroma_db", 
                chunk_size=1000, chunk_overlap=200):
    """
    Ingest files from a single directory, split into appropriate chunks, 
    and store in vector database.
    
    Args:
        directory_path: Path to directory containing files to ingest
        collection_name: Name of the collection in the vector database
        output_dir: Base directory to store the vector database
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
    """
    # Initialize splitters
    headers_to_split_on = [
        ("#", "header1"),
        ("##", "header2"),
        ("###", "header3"),
        ("####", "header4")
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    code_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Process all files
    documents = []
    
    # Find and process files by type
    file_types = {
        # Text and Markdown
        "markdown": glob.glob(os.path.join(directory_path, "**/*.md"), recursive=True),
        "r": glob.glob(os.path.join(directory_path, "**/*.R"), recursive=True),
        "rmd": glob.glob(os.path.join(directory_path, "**/*.Rmd"), recursive=True),
        "qmd": glob.glob(os.path.join(directory_path, "**/*.qmd"), recursive=True),
        
        # Python files
        "python": glob.glob(os.path.join(directory_path, "**/*.py"), recursive=True),
        "ipynb": glob.glob(os.path.join(directory_path, "**/*.ipynb"), recursive=True),
        
        # Web files
        "html": glob.glob(os.path.join(directory_path, "**/*.html"), recursive=True),
        "css": glob.glob(os.path.join(directory_path, "**/*.css"), recursive=True),
        "scss": glob.glob(os.path.join(directory_path, "**/*.scss"), recursive=True),
        "js": glob.glob(os.path.join(directory_path, "**/*.js"), recursive=True),
        
        # Data files
        "csv": glob.glob(os.path.join(directory_path, "**/*.csv"), recursive=True),
        "json": glob.glob(os.path.join(directory_path, "**/*.json"), recursive=True),
        "parquet": glob.glob(os.path.join(directory_path, "**/*.parquet"), recursive=True),
        "sqlite": glob.glob(os.path.join(directory_path, "**/*.db"), recursive=True) + 
                 glob.glob(os.path.join(directory_path, "**/*.sqlite"), recursive=True) + 
                 glob.glob(os.path.join(directory_path, "**/*.sqlite3"), recursive=True),
        
        # R specific data formats
        "rds": glob.glob(os.path.join(directory_path, "**/*.rds"), recursive=True),
        "rdata": glob.glob(os.path.join(directory_path, "**/*.RData"), recursive=True) + 
                glob.glob(os.path.join(directory_path, "**/*.rdata"), recursive=True),
        
        # Python specific data formats
        "pickle": glob.glob(os.path.join(directory_path, "**/*.pkl"), recursive=True) + 
                 glob.glob(os.path.join(directory_path, "**/*.pickle"), recursive=True),
        
        # Office documents
        "pdf": glob.glob(os.path.join(directory_path, "**/*.pdf"), recursive=True),
        "excel": glob.glob(os.path.join(directory_path, "**/*.xlsx"), recursive=True) + 
                glob.glob(os.path.join(directory_path, "**/*.xls"), recursive=True),
        "word": glob.glob(os.path.join(directory_path, "**/*.docx"), recursive=True) + 
               glob.glob(os.path.join(directory_path, "**/*.doc"), recursive=True),
        "powerpoint": glob.glob(os.path.join(directory_path, "**/*.pptx"), recursive=True) + 
                     glob.glob(os.path.join(directory_path, "**/*.ppt"), recursive=True)
    }
    
    # Process markdown files
    print(f"Processing {len(file_types['markdown'])} markdown (.md) files...")
    for file_path in file_types['markdown']:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                content = f.read()
            except UnicodeDecodeError:
                print(f"Warning: Could not read {file_path} due to encoding issues. Skipping.")
                continue
        
        # Get relative path for metadata
        rel_path = os.path.relpath(file_path, directory_path)
        
        # Split by headers first
        md_docs = markdown_splitter.split_text(content)
        for doc in md_docs:
            doc.metadata["source"] = rel_path
            doc.metadata["file_type"] = "markdown"
        
        # Further split by size if needed
        docs = text_splitter.split_documents(md_docs)
        documents.extend(docs)
    
    # Process R files
    print(f"Processing {len(file_types['r'])} R (.R) files...")
    for file_path in file_types['r']:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                content = f.read()
            except UnicodeDecodeError:
                print(f"Warning: Could not read {file_path} due to encoding issues. Skipping.")
                continue
        
        # Get relative path for metadata
        rel_path = os.path.relpath(file_path, directory_path)
        
        # Split text into chunks
        chunks = code_splitter.create_documents(
            texts=[content],
            metadatas=[{"source": rel_path, "file_type": "R", "language": "R"}]
        )
        documents.extend(chunks)
    
    # Process Rmd files
    print(f"Processing {len(file_types['rmd'])} R Markdown (.Rmd) files...")
    for file_path in file_types['rmd']:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                content = f.read()
            except UnicodeDecodeError:
                print(f"Warning: Could not read {file_path} due to encoding issues. Skipping.")
                continue
        
        # Get relative path for metadata
        rel_path = os.path.relpath(file_path, directory_path)
        
        # Try to split by headers first (since Rmd is markdown-based)
        try:
            md_docs = markdown_splitter.split_text(content)
            for doc in md_docs:
                doc.metadata["source"] = rel_path
                doc.metadata["file_type"] = "Rmd"
            
            # Further split by size if needed
            docs = text_splitter.split_documents(md_docs)
            documents.extend(docs)
        except Exception as e:
            # Fallback to regular splitting if header parsing fails
            print(f"Warning: Markdown parsing failed for {file_path}, using regular chunking")
            chunks = text_splitter.create_documents(
                texts=[content],
                metadatas=[{"source": rel_path, "file_type": "Rmd"}]
            )
            documents.extend(chunks)
    
    # Process Quarto files
    print(f"Processing {len(file_types['qmd'])} Quarto (.qmd) files...")
    for file_path in file_types['qmd']:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                content = f.read()
            except UnicodeDecodeError:
                print(f"Warning: Could not read {file_path} due to encoding issues. Skipping.")
                continue
        
        # Get relative path for metadata
        rel_path = os.path.relpath(file_path, directory_path)
        
        # Try to split by headers first (since qmd is markdown-based)
        try:
            md_docs = markdown_splitter.split_text(content)
            for doc in md_docs:
                doc.metadata["source"] = rel_path
                doc.metadata["file_type"] = "qmd"
            
            # Further split by size if needed
            docs = text_splitter.split_documents(md_docs)
            documents.extend(docs)
        except Exception as e:
            # Fallback to regular splitting if header parsing fails
            print(f"Warning: Markdown parsing failed for {file_path}, using regular chunking")
            chunks = text_splitter.create_documents(
                texts=[content],
                metadatas=[{"source": rel_path, "file_type": "qmd"}]
            )
            documents.extend(chunks)
            
    # Process Python files
    print(f"Processing {len(file_types['python'])} Python (.py) files...")
    for file_path in file_types['python']:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                content = f.read()
            except UnicodeDecodeError:
                print(f"Warning: Could not read {file_path} due to encoding issues. Skipping.")
                continue
        
        # Get relative path for metadata
        rel_path = os.path.relpath(file_path, directory_path)
        
        # Split text into chunks
        chunks = code_splitter.create_documents(
            texts=[content],
            metadatas=[{"source": rel_path, "file_type": "Python", "language": "Python"}]
        )
        documents.extend(chunks)
        
    # Process Jupyter Notebook files
    print(f"Processing {len(file_types['ipynb'])} Jupyter Notebook (.ipynb) files...")
    for file_path in file_types['ipynb']:
        try:
            loader = NotebookLoader(file_path, include_outputs=True)
            notebook_docs = loader.load()
            
            # Get relative path for metadata
            rel_path = os.path.relpath(file_path, directory_path)
            
            # Add metadata
            for doc in notebook_docs:
                doc.metadata["source"] = rel_path
                doc.metadata["file_type"] = "Jupyter Notebook"
                
            # Split into chunks
            docs = text_splitter.split_documents(notebook_docs)
            documents.extend(docs)
        except Exception as e:
            print(f"Warning: Could not process Jupyter notebook {file_path}: {e}")
            continue
    
    # Process HTML files
    print(f"Processing {len(file_types['html'])} HTML (.html) files...")
    for file_path in file_types['html']:
        try:
            loader = BSHTMLLoader(file_path)
            html_docs = loader.load()
            
            # Get relative path for metadata
            rel_path = os.path.relpath(file_path, directory_path)
            
            # Add metadata
            for doc in html_docs:
                doc.metadata["source"] = rel_path
                doc.metadata["file_type"] = "HTML"
                
            # Split into chunks
            docs = text_splitter.split_documents(html_docs)
            documents.extend(docs)
        except Exception as e:
            print(f"Warning: Could not process HTML file {file_path}: {e}")
            continue
    
    # Process CSS/SCSS/JS files (treat as code)
    for file_type, file_list in [
        ("CSS", file_types['css']), 
        ("SCSS", file_types['scss']), 
        ("JavaScript", file_types['js'])
    ]:
        print(f"Processing {len(file_list)} {file_type} files...")
        for file_path in file_list:
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    content = f.read()
                except UnicodeDecodeError:
                    print(f"Warning: Could not read {file_path} due to encoding issues. Skipping.")
                    continue
            
            # Get relative path for metadata
            rel_path = os.path.relpath(file_path, directory_path)
            
            # Split text into chunks
            chunks = code_splitter.create_documents(
                texts=[content],
                metadatas=[{"source": rel_path, "file_type": file_type, "language": file_type}]
            )
            documents.extend(chunks)
    
    # Process CSV files
    print(f"Processing {len(file_types['csv'])} CSV files...")
    for file_path in file_types['csv']:
        try:
            loader = CSVLoader(file_path)
            csv_docs = loader.load()
            
            # Get relative path for metadata
            rel_path = os.path.relpath(file_path, directory_path)
            
            # Add metadata
            for doc in csv_docs:
                doc.metadata["source"] = rel_path
                doc.metadata["file_type"] = "CSV"
                
            # Split into chunks
            docs = text_splitter.split_documents(csv_docs)
            documents.extend(docs)
        except Exception as e:
            print(f"Warning: Could not process CSV file {file_path}: {e}")
            
            # Fallback using pandas if the standard loader fails
            try:
                df = pd.read_csv(file_path)
                content = df.to_string(index=False)
                
                rel_path = os.path.relpath(file_path, directory_path)
                
                chunks = text_splitter.create_documents(
                    texts=[content],
                    metadatas=[{"source": rel_path, "file_type": "CSV"}]
                )
                documents.extend(chunks)
            except Exception as e2:
                print(f"Warning: Fallback for CSV file {file_path} also failed: {e2}")
                continue
    
    # Process JSON files
    print(f"Processing {len(file_types['json'])} JSON files...")
    for file_path in file_types['json']:
        try:
            # Define a simple extraction function for JSON
            def extract_data(data, json_path="$."):
                if isinstance(data, dict):
                    return " ".join([f"{json_path}{k}: {v}" for k, v in data.items()])
                elif isinstance(data, list):
                    return " ".join([str(item) for item in data])
                else:
                    return str(data)
            
            loader = JSONLoader(
                file_path=file_path,
                jq_schema=".",
                content_key=None,
                text_content=False,
                json_lines=False,
                metadata_func=lambda meta: {"source": os.path.relpath(file_path, directory_path)}
            )
            
            json_docs = loader.load()
            
            # Add metadata
            for doc in json_docs:
                doc.metadata["file_type"] = "JSON"
                
            # Split into chunks
            docs = text_splitter.split_documents(json_docs)
            documents.extend(docs)
        except Exception as e:
            print(f"Warning: Could not process JSON file {file_path} with loader: {e}")
            
            # Fallback to manual JSON parsing
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    json_data = json.loads(content)
                    
                    # Convert to string representation
                    if isinstance(json_data, dict):
                        content = json.dumps(json_data, indent=2)
                    elif isinstance(json_data, list):
                        content = "\n".join([json.dumps(item, indent=2) for item in json_data])
                    else:
                        content = str(json_data)
                    
                    rel_path = os.path.relpath(file_path, directory_path)
                    
                    chunks = text_splitter.create_documents(
                        texts=[content],
                        metadatas=[{"source": rel_path, "file_type": "JSON"}]
                    )
                    documents.extend(chunks)
            except Exception as e2:
                print(f"Warning: Fallback for JSON file {file_path} also failed: {e2}")
                continue
    
    # Process Parquet files
    print(f"Processing {len(file_types['parquet'])} Parquet files...")
    for file_path in file_types['parquet']:
        try:
            # Read parquet file to pandas DataFrame
            df = pd.read_parquet(file_path)
            content = df.to_string(index=False)
            
            rel_path = os.path.relpath(file_path, directory_path)
            
            # Create document chunks
            chunks = text_splitter.create_documents(
                texts=[content],
                metadatas=[{"source": rel_path, "file_type": "Parquet"}]
            )
            documents.extend(chunks)
        except Exception as e:
            print(f"Warning: Could not process Parquet file {file_path}: {e}")
            continue
    
    # Process SQLite databases
    print(f"Processing {len(file_types['sqlite'])} SQLite database files...")
    for file_path in file_types['sqlite']:
        try:
            # Connect to SQLite database
            conn = sqlite3.connect(file_path)
            
            # Get list of tables
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            rel_path = os.path.relpath(file_path, directory_path)
            
            # Process each table
            for table in tables:
                table_name = table[0]
                
                # Read table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                schema = cursor.fetchall()
                schema_text = f"Table: {table_name}\nSchema: " + ", ".join([f"{col[1]} ({col[2]})" for col in schema])
                
                # Read sample data (first 100 rows)
                df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 100", conn)
                data_text = df.to_string(index=False)
                
                # Combine schema and data
                content = f"{schema_text}\n\nSample data:\n{data_text}"
                
                # Create document chunks
                chunks = text_splitter.create_documents(
                    texts=[content],
                    metadatas=[{
                        "source": rel_path,
                        "file_type": "SQLite",
                        "table": table_name
                    }]
                )
                documents.extend(chunks)
            
            conn.close()
        except Exception as e:
            print(f"Warning: Could not process SQLite database {file_path}: {e}")
            continue
    
    # Process R data formats
    if pyreadr is not None:
        # Process RDS files
        print(f"Processing {len(file_types['rds'])} RDS files...")
        for file_path in file_types['rds']:
            try:
                # Read RDS file (single R object)
                result = pyreadr.read_r(file_path)
                
                # Convert to string representation
                if isinstance(result, pd.DataFrame):
                    content = result.to_string(index=False)
                else:
                    content = str(result)
                
                rel_path = os.path.relpath(file_path, directory_path)
                
                # Create document chunks
                chunks = text_splitter.create_documents(
                    texts=[content],
                    metadatas=[{"source": rel_path, "file_type": "RDS"}]
                )
                documents.extend(chunks)
            except Exception as e:
                print(f"Warning: Could not process RDS file {file_path}: {e}")
                continue
                
        # Process RData files
        print(f"Processing {len(file_types['rdata'])} RData files...")
        for file_path in file_types['rdata']:
            try:
                # Read RData file (multiple R objects)
                result = pyreadr.read_r(file_path)
                
                rel_path = os.path.relpath(file_path, directory_path)
                
                # Process each object in the RData file
                for obj_name, obj_data in result.items():
                    if isinstance(obj_data, pd.DataFrame):
                        content = f"Object: {obj_name}\n" + obj_data.to_string(index=False)
                    else:
                        content = f"Object: {obj_name}\n" + str(obj_data)
                    
                    # Create document chunks
                    chunks = text_splitter.create_documents(
                        texts=[content],
                        metadatas=[{
                            "source": rel_path,
                            "file_type": "RData",
                            "object": obj_name
                        }]
                    )
                    documents.extend(chunks)
            except Exception as e:
                print(f"Warning: Could not process RData file {file_path}: {e}")
                continue
    else:
        print("Warning: pyreadr package not available. Skipping RDS and RData files.")
    
    # Process Python pickle files
    print(f"Processing {len(file_types['pickle'])} pickle files...")
    for file_path in file_types['pickle']:
        try:
            # Read pickle file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Convert to string representation
            if isinstance(data, pd.DataFrame):
                content = data.to_string(index=False)
            elif isinstance(data, np.ndarray):
                content = str(data)
            elif isinstance(data, dict):
                content = "\n".join([f"{k}: {v}" for k, v in data.items()])
            elif isinstance(data, list):
                content = "\n".join([str(item) for item in data])
            else:
                content = str(data)
            
            rel_path = os.path.relpath(file_path, directory_path)
            
            # Create document chunks
            chunks = text_splitter.create_documents(
                texts=[content],
                metadatas=[{"source": rel_path, "file_type": "Pickle"}]
            )
            documents.extend(chunks)
        except Exception as e:
            print(f"Warning: Could not process pickle file {file_path}: {e}")
            continue
    
    # Process PDF files
    print(f"Processing {len(file_types['pdf'])} PDF files...")
    for file_path in file_types['pdf']:
        try:
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()
            
            # Get relative path for metadata
            rel_path = os.path.relpath(file_path, directory_path)
            
            # Add metadata
            for doc in pdf_docs:
                doc.metadata["source"] = rel_path
                doc.metadata["file_type"] = "PDF"
                
            # Split into chunks
            docs = text_splitter.split_documents(pdf_docs)
            documents.extend(docs)
        except Exception as e:
            print(f"Warning: Could not process PDF file {file_path}: {e}")
            continue
    
    # Process Excel files
    print(f"Processing {len(file_types['excel'])} Excel files...")
    for file_path in file_types['excel']:
        try:
            loader = UnstructuredExcelLoader(file_path)
            excel_docs = loader.load()
            
            # Get relative path for metadata
            rel_path = os.path.relpath(file_path, directory_path)
            
            # Add metadata
            for doc in excel_docs:
                doc.metadata["source"] = rel_path
                doc.metadata["file_type"] = "Excel"
                
            # Split into chunks
            docs = text_splitter.split_documents(excel_docs)
            documents.extend(docs)
        except Exception as e:
            print(f"Warning: Could not process Excel file {file_path} with loader: {e}")
            
            # Fallback to pandas
            try:
                df_dict = pd.read_excel(file_path, sheet_name=None)
                
                rel_path = os.path.relpath(file_path, directory_path)
                
                # Process each sheet
                for sheet_name, df in df_dict.items():
                    content = f"Sheet: {sheet_name}\n" + df.to_string(index=False)
                    
                    # Create document chunks
                    chunks = text_splitter.create_documents(
                        texts=[content],
                        metadatas=[{
                            "source": rel_path,
                            "file_type": "Excel",
                            "sheet": sheet_name
                        }]
                    )
                    documents.extend(chunks)
            except Exception as e2:
                print(f"Warning: Fallback for Excel file {file_path} also failed: {e2}")
                continue
    
    # Process Word files
    print(f"Processing {len(file_types['word'])} Word files...")
    for file_path in file_types['word']:
        try:
            loader = UnstructuredWordDocumentLoader(file_path)
            word_docs = loader.load()
            
            # Get relative path for metadata
            rel_path = os.path.relpath(file_path, directory_path)
            
            # Add metadata
            for doc in word_docs:
                doc.metadata["source"] = rel_path
                doc.metadata["file_type"] = "Word"
                
            # Split into chunks
            docs = text_splitter.split_documents(word_docs)
            documents.extend(docs)
        except Exception as e:
            print(f"Warning: Could not process Word file {file_path}: {e}")
            continue
    
    # Process PowerPoint files
    print(f"Processing {len(file_types['powerpoint'])} PowerPoint files...")
    for file_path in file_types['powerpoint']:
        try:
            loader = UnstructuredPowerPointLoader(file_path)
            ppt_docs = loader.load()
            
            # Get relative path for metadata
            rel_path = os.path.relpath(file_path, directory_path)
            
            # Add metadata
            for doc in ppt_docs:
                doc.metadata["source"] = rel_path
                doc.metadata["file_type"] = "PowerPoint"
                
            # Split into chunks
            docs = text_splitter.split_documents(ppt_docs)
            documents.extend(docs)
        except Exception as e:
            print(f"Warning: Could not process PowerPoint file {file_path}: {e}")
            continue
    
    # Store in vector database
    print(f"Creating vector database with {len(documents)} document chunks...")
    # Use os.path.join for proper path handling
    persist_dir = os.path.join(output_dir, collection_name)
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    
    print(f"Vector database created and persisted to {persist_dir}")
    return db, persist_dir

# For backward compatibility
def ingest_markdown_files(directory_path, collection_name="r_packages_docs", output_dir="./chroma_db"):
    """
    Legacy function. Use ingest_files instead.
    """
    print("Warning: This function is deprecated. Use ingest_files instead.")
    
    # Call the new function with default parameters
    db, persist_dir = ingest_files(directory_path, collection_name, output_dir)
    return db

# For backward compatibility
def ingest_r_files(directory_path, collection_name="r_packages_code", output_dir="./chroma_db"):
    """
    Legacy function. Use ingest_files instead.
    """
    print("Warning: This function is deprecated. Use ingest_files instead.")
    
    # Call the new function with default parameters
    db, persist_dir = ingest_files(directory_path, collection_name, output_dir)
    return db

# For backward compatibility
def ingest_all_r_files(directory_path, collection_name="r_knowledge_base", output_dir="./chroma_db"):
    """
    Legacy function. Use ingest_files instead.
    """
    print("Warning: This function is deprecated. Use ingest_files instead.")
    
    # Call the new function with default parameters
    db, persist_dir = ingest_files(directory_path, collection_name, output_dir)
    return db

if __name__ == "__main__":
    # Example usage
    # Replace with your actual path
    content_path = "../data"
    output_dir = "./chroma_db"
    
    # Use the new unified function
    db, persist_dir = ingest_files(content_path, output_dir=output_dir)
    
    print(f"Ingested all files into Chroma DB at {persist_dir}")