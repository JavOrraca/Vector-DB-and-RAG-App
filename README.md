# Vector DB and RAG App

A comprehensive vector database and Retrieval-Augmented Generation (RAG) system developed with Python and R Shiny. This enterprise-ready solution can process a wide variety of file types to create a vector database for enhanced LLM responses based on your own data.

## Overview

This system offers:

1. **Versatile File Processing**: Ingest various file types including code, documents, data files, and more
2. **Vector Database Creation**: Create embeddings using state-of-the-art models
3. **User-Friendly Interface**: An intuitive R Shiny app for uploading files and creating vector databases
4. **Chat Interface**: Query your data with RAG-enhanced responses from your choice of LLM
5. **Multiple LLM Support**: Works with both Anthropic and OpenAI models
6. **Customizable Parameters**: Control chunking parameters, system prompts, and more

## Command Line Interface

The Python CLI offers two primary functions:

1. **Ingest**: Process various file types into a vector database
2. **Query**: Interact with the vector database through a CLI interface

### Requirements

- Python 3.8+
- R 4.1+ (for the Shiny interface)
- Required Python packages (key dependencies):
  - langchain and extensions (anthropic, openai, huggingface, chroma)
  - chromadb (vector database)
  - sentence-transformers (embedding model)
  - anthropic and openai (LLM APIs)
  - unstructured (document processing)
  - Various file format libraries (pypdf, pyreadr, openpyxl, etc.)
- Required R packages:
  - Core: shiny, bslib, shinyjs, jsonlite
  - GitHub packages: shinychat, ellmer (installed automatically)

> **Note**: The application uses a unique architecture where the R Shiny interface communicates with Python through command-line calls rather than direct integration via reticulate. This design makes the application more robust across different environments.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Vector-DB-and-RAG-App.git
cd Vector-DB-and-RAG-App
```

2. Install the required Python dependencies:

   **Fast Install (Recommended):**
   ```bash
   # This uses uv - a much faster alternative to pip (up to 10-50x faster)
   python install_deps.py
   
   # Activate the virtual environment
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate  # On Windows
   ```

   **Alternative (Standard pip):**
   ```bash
   pip install -r requirements.txt
   ```

3. Install the required R dependencies (if using the Shiny app):
```bash
Rscript app/install_dependencies.R
```

Note: This will install GitHub packages like `shinychat` and `ellmer` which are required for the chat interface. You may need to install the `devtools` package first if it's not already installed.

4. Set your API keys (optional):
```bash
# For Anthropic
export ANTHROPIC_API_KEY=your-anthropic-api-key

# For OpenAI
export OPENAI_API_KEY=your-openai-api-key
```

You can also set these keys directly in the Shiny app interface or pass them as parameters to the Python CLI.

> **Note about Python dependencies**: The installation now uses [uv](https://github.com/astral-sh/uv), a very fast Python package installer and resolver. It can be 10-50x faster than pip for installing packages and creating virtual environments. The script works on all major platforms (Linux, macOS, and Windows).

## Usage

### Python CLI

#### Ingesting Files

Process files from a directory into a vector database:

```bash
python src/main.py ingest --content-dir ./data --output-dir ./chroma_db --chunk-size 1000 --chunk-overlap 200
```

Parameters:
- `--content-dir`: Directory containing files to ingest
- `--output-dir`: Directory to store the vector database (default: `./chroma_db`)
- `--collection-name`: Name for the vector database collection (default: `knowledge_base`)
- `--chunk-size`: Size of text chunks for splitting (default: 1000)
- `--chunk-overlap`: Overlap between chunks (default: 200)

#### Querying the System

After ingestion, query the system:

```bash
# Interactive mode
python src/main.py query --db-path ./chroma_db/knowledge_base

# Single query mode with specific LLM settings
python src/main.py query --db-path ./chroma_db/knowledge_base --question "How do I...?" --provider anthropic --model claude-3-7-sonnet-20250219
```

Parameters:
- `--db-path`: Path to vector database
- `--question`: Question to ask (optional; if not provided, enters interactive mode)
- `--api-key`: LLM API key (if not set as environment variable)
- `--provider`: LLM provider to use (`anthropic` or `openai`, default: `anthropic`)
- `--model`: Model name to use (default: `claude-3-7-sonnet-20250219` for Anthropic)
- `--system-prompt`: System prompt for the LLM (default: "You are an AI assistant.")
- `--k`: Number of most relevant documents to retrieve (default: 4)

### Shiny App

The Shiny app provides a user-friendly interface for:
1. Uploading files
2. Creating vector databases
3. Chatting with your data using RAG-enhanced responses

To launch the app:

```bash
cd app
Rscript app.R
```

Features of the Shiny app:
- Upload individual files or entire folders
- Configure chunking parameters via sliders
- Choose between Anthropic or OpenAI models
- Switch between dark and light themes
- Download your vector database for later use
- Chat interface with context-aware responses

## Supported File Types

The system can ingest and process numerous file types:

- **Text and Markdown**: `.md`, `.R`, `.Rmd`, `.qmd`, `.py`, `.js`, `.html`, `.css`, `.scss`
- **Notebooks**: `.ipynb` (Jupyter Notebooks)
- **Data Files**: `.csv`, `.json`, `.parquet`, `.db`, `.sqlite`, `.sqlite3`
- **R-specific Formats**: `.rds`, `.RData`
- **Python-specific Formats**: `.pkl`, `.pickle`
- **Office Documents**: `.pdf`, `.xlsx`, `.docx`, `.pptx`

Each file type is processed with specialized loaders:
- Text files are processed with appropriate text splitters
- Structured data files (CSV, JSON, etc.) are processed with tabular handling
- Office documents use the `unstructured` library for content extraction
- Database files have their schema and data extracted properly

The system uses the LangChain framework's document loaders and text splitters for consistent processing across file types.

## System Components

### Architecture

The system uses a clean separation between Python and R components:

1. **Python Backend**:
   - Handles all data processing, embedding creation, and vector database operations
   - Exposes functionality through a command-line wrapper script
   - Returns structured JSON responses for reliable interoperability
   - The `vector_db_wrapper.py` module acts as a bridge for R-Python communication

2. **R Shiny Frontend**:
   - Provides the user interface and manages user interactions
   - Communicates with Python backend through subprocess calls
   - Parses JSON responses for display and interaction
   - Uses GitHub packages `shinychat` and `ellmer` for enhanced UI capabilities

This architecture makes the system more robust across different environments and easier to debug since it doesn't rely on direct language integration that can be fragile across different OS configurations.

### Ingestion

The ingestion pipeline:
1. Recursively finds all supported files in the specified directory
2. Processes each file type appropriately with specialized loaders for different formats
3. Creates vector embeddings for each chunk
4. Stores the embeddings in a unified Chroma vector database

### Retrieval

The retrieval system:
1. Takes a user question
2. Generates an embedding for the question
3. Searches the vector database using `similarity_search_with_score` to find relevant documents
4. Selects the top k most similar documents (default k=4)
5. Combines the results into a formatted context
6. Sends the question and context to an LLM (Anthropic Claude or OpenAI GPT) with a system prompt
7. Returns the LLM's response in JSON format with the context sources

The system uses a scoring mechanism to ensure only highly relevant documents are included, and provides transparency by including the sources in the response.

### Shiny Interface

The Shiny app provides:
1. A file upload interface with chunking parameter controls
2. Vector database creation with progress indicators
3. LLM configuration options
4. A chat interface for interacting with your data
5. Database download capabilities

## Customization

### Embedding Models

By default, the system uses HuggingFaceEmbeddings with the `sentence-transformers/all-MiniLM-L6-v2` model, which provides a good balance of quality and performance. This model creates embeddings with 384 dimensions that are used by Chroma for similarity search.

### Vector Database

The system exclusively uses Chroma as the vector database, which provides:
- Persistent storage of document embeddings
- Efficient similarity search with scoring
- Metadata storage for document tracking

The system leverages Chroma's `similarity_search_with_score` method to return the most relevant documents based on embedding similarity.

### LLM Providers

The system supports both Anthropic and OpenAI:
- **Anthropic**: Claude models (default: claude-3-7-sonnet-20250219)
- **OpenAI**: GPT models (default: gpt-4o)

API keys can be provided in three ways:
1. As environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`)
2. As command-line parameters (`--api-key`)
3. Through the Shiny interface (API key input field)

### UI Customization

The Shiny app features:
- Retro dark theme (default) with a light theme option
- Google's Press Start 2P font for headings
- Modern card-based layout using the bslib package
- Responsive design for various screen sizes