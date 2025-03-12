# Load required libraries
library(shiny)
library(bslib)
library(shinyjs)
library(htmltools)
library(httr2)
library(shinychat)
library(ellmer)
library(jsonlite)
library(fs)
library(future)
library(promises)

# Configure maximum upload size (2GB)
options(shiny.maxRequestSize = 2 * 1024^3)

# Set up asynchronous processing
plan(multisession)

# Helper function to check if running on Windows
is_windows <- function() {
  .Platform$OS.type == "windows"
}

# Find src directory
src_dir <- normalizePath(file.path(getwd(), "..", "src"))
wrapper_script <- file.path(src_dir, "vector_db_wrapper.py")
cat("Source directory:", src_dir, "\n")
cat("Wrapper script:", wrapper_script, "\n")

# Ensure wrapper script is executable
if (is_windows() == FALSE) {
  system2("chmod", c("+x", wrapper_script))
}
# Function to create vector DB from uploaded files - uses direct system call
create_vector_db <- function(file_paths, temp_dir, chunk_size = 1000, chunk_overlap = 200) {
  # Create output directory
  output_dir <- file.path(temp_dir, "chroma_db")
  collection_name <- "knowledge_base"
  
  cat("Creating vector DB with parameters:\n")
  cat("- Output directory:", output_dir, "\n")
  cat("- Collection name:", collection_name, "\n")
  cat("- Chunk size:", chunk_size, "\n")
  cat("- Chunk overlap:", chunk_overlap, "\n")
  cat("- Number of files:", length(file_paths), "\n")
  
  # Create input directory with symlinks to uploaded files
  input_dir <- file.path(temp_dir, "input")
  dir.create(input_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Create symlinks to the uploaded files
  for (file_path in file_paths) {
    file_name <- basename(file_path)
    file.symlink(file_path, file.path(input_dir, file_name))
  }
  cat("Input directory prepared:", input_dir, "\n")
  
  # Call Python wrapper script
  tryCatch({
    cat("Calling vector_db_wrapper.py script...\n")
    
    python_cmd <- Sys.which("python3")
    if (python_cmd == "") {
      python_cmd <- Sys.which("python")
    }
    
    # Run the command and capture output
    result <- system2(
      python_cmd,
      args = c(
        wrapper_script,
        "create_vector_db",
        input_dir,
        output_dir,
        collection_name,
        as.character(chunk_size),
        as.character(chunk_overlap)
      ),
      stdout = TRUE,
      stderr = TRUE
    )
    
    # Look for the JSON line in the output
    json_line <- NULL
    for (line in result) {
      if (grepl('^\\{"success":', line)) {
        json_line <- line
        break
      }
    }
    
    if (is.null(json_line)) {
      cat("Could not find JSON in output. Raw output:", paste(result, collapse="\n"), "\n")
      db_path <- file.path(output_dir, collection_name)
      cat("Falling back to default path:", db_path, "\n")
      return(db_path)
    }
    
    # Try to parse as JSON
    tryCatch({
      parsed <- fromJSON(json_line)
      if (parsed$success) {
        cat("Vector DB created successfully at:", parsed$db_path, "\n")
        return(parsed$db_path)
      } else {
        cat("Error creating vector DB:", parsed$error, "\n")
        if (is.null(parsed$traceback) == FALSE) {
          cat("Traceback:", parsed$traceback, "\n")
        }
        stop(parsed$error)
      }
    }, error = function(e) {
      cat("Could not parse JSON. Raw line:", json_line, "\n")
      cat("Full output:", paste(result, collapse="\n"), "\n")
      db_path <- file.path(output_dir, collection_name)
      cat("Falling back to default path:", db_path, "\n")
      return(db_path)
    })
  }, error = function(e) {
    cat("Error executing vector_db_wrapper.py:", conditionMessage(e), "\n")
    stop(paste("Error creating vector DB:", conditionMessage(e)))
  })
}
# Function to zip the vector database for download
create_vector_db_zip <- function(db_path, temp_dir) {
  zip_file <- file.path(temp_dir, "vector_db.zip")
  
  cat("Creating zip file for vector DB\n")
  cat("Source DB path:", db_path, "\n")
  cat("Target zip file:", zip_file, "\n")
  
  # Check if the source exists
  if (dir.exists(db_path) == FALSE) {
    cat("Warning: Vector DB directory does not exist at", db_path, "\n")
    # Create a dummy file explaining the error
    error_file <- file.path(temp_dir, "vector_db_error.txt")
    cat("Vector database does not exist at:", db_path, file = error_file)
    zip_cmd <- paste("cd", shQuote(temp_dir), "&&", 
                     "zip -j", shQuote(zip_file), shQuote("vector_db_error.txt"))
    system(zip_cmd)
    return(zip_file)
  }
  
  # Check if zip command is available
  if (Sys.which("zip") == "") {
    cat("Warning: zip command not found, using R's zip function\n")
    
    # Use R's built-in zip function
    files_to_zip <- list.files(db_path, full.names = TRUE, recursive = TRUE)
    
    if (length(files_to_zip) == 0) {
      cat("Warning: No files found in vector DB directory\n")
      error_file <- file.path(temp_dir, "empty_db.txt")
      cat("Vector database directory is empty", file = error_file)
      files_to_zip <- error_file
    }
    
    zip(zip_file, files_to_zip, flags = "-r9X")
  } else {
    # Use system zip command
    cat("Using system zip command\n")
    zip_cmd <- paste("cd", shQuote(dirname(db_path)), "&&", 
                     "zip -r", shQuote(zip_file), shQuote(basename(db_path)))
    cat("Executing:", zip_cmd, "\n")
    system(zip_cmd)
  }
  
  # Verify zip file was created
  if (file.exists(zip_file) == FALSE) {
    cat("Warning: Zip file was not created\n")
    # Create a fallback zip file
    error_file <- file.path(temp_dir, "zip_error.txt")
    cat("Failed to create zip file for vector database", file = error_file)
    zip_file <- file.path(temp_dir, "error.zip")
    zip(zip_file, error_file)
  }
  
  cat("Zip file created:", zip_file, "\n")
  return(zip_file)
}

# Function to create a chat completion using LLM API
create_chat_completion <- function(prompt, context, provider, model, api_key, system_prompt = "You are an AI assistant.") {
  response <- if (provider == "anthropic") {
    # Create Anthropic API request
    req <- request("https://api.anthropic.com/v1/messages") %>%
      req_headers(
        "x-api-key" = api_key,
        "anthropic-version" = "2023-06-01",
        "content-type" = "application/json"
      ) %>%
      req_body_json(list(
        model = model,
        max_tokens = 4000,
        messages = list(
          list(role = "user", content = paste0("Context: ", context, "\n\nQuestion: ", prompt))
        ),
        system = system_prompt
      ))
    
    # Send request
    resp <- req %>% req_perform()
    content <- resp %>% resp_body_json()
    content$content[[1]]$text
  } else {
    # Create OpenAI API request
    req <- request("https://api.openai.com/v1/chat/completions") %>%
      req_headers(
        "Authorization" = paste("Bearer", api_key),
        "Content-Type" = "application/json"
      ) %>%
      req_body_json(list(
        model = model,
        messages = list(
          list(role = "system", content = system_prompt),
          list(role = "user", content = paste0("Context: ", context, "\n\nQuestion: ", prompt))
        ),
        max_tokens = 4000
      ))
    
    # Send request
    resp <- req %>% req_perform()
    content <- resp %>% resp_body_json()
    content$choices[[1]]$message$content
  }
  
  return(response)
}
# Function to get relevant context from vector database
get_context <- function(query, db_path, k = 5) {
  cat("Getting context for query:", query, "\n")
  cat("Vector DB path:", db_path, "\n")
  
  # Create a temp file for JSON output
  json_file <- tempfile(fileext = ".json")
  on.exit(unlink(json_file), add = TRUE)
  
  # Call Python wrapper script for context retrieval
  tryCatch({
    cat("Calling vector_db_wrapper.py script for context retrieval...\n")
    
    python_cmd <- Sys.which("python3")
    if (python_cmd == "") {
      python_cmd <- Sys.which("python")
    }
    
    # Run the command and capture output
    result <- system2(
      python_cmd,
      args = c(
        wrapper_script,
        "retrieve_context",
        db_path,
        shQuote(query),
        as.character(k)
      ),
      stdout = TRUE,
      stderr = TRUE
    )
    
    # Get combined output
    combined_output <- paste(result, collapse = "\n")
    
    # Create a temp script to extract just the JSON
    temp_script <- tempfile(fileext = ".py")
    cat(paste0(
      "import re, json\n",
      "with open('", gsub("\\\\", "/", json_file), "', 'w') as f:\n",
      "    text = \"\"\"", gsub("\"", "\\\"", combined_output), "\"\"\"\n",
      "    match = re.search(r'{.*\"success\".*?\"context\".*?\\[.*?\\].*?}', text, re.DOTALL)\n",
      "    if match:\n",
      "        json_text = match.group(0)\n",
      "        f.write(json_text)\n",
      "        print('JSON extracted')\n",
      "    else:\n",
      "        f.write('{\"success\": false, \"error\": \"Could not extract JSON\"}')\n",
      "        print('No JSON found')\n"
    ), file = temp_script)
    
    # Run the extraction script
    system2(python_cmd, args = c(temp_script))
    
    # Check if JSON file was created and has content
    if (file.exists(json_file) && file.info(json_file)$size > 0) {
      # Read and parse JSON
      json_content <- readLines(json_file, warn = FALSE)
      json_content <- paste(json_content, collapse = "\n")
      
      parsed <- tryCatch({
        fromJSON(json_content)
      }, error = function(e) {
        cat("Error parsing JSON:", conditionMessage(e), "\n")
        return(NULL)
      })
      
      if (is.null(parsed) == FALSE && is.null(parsed$success) == FALSE && parsed$success) {
        # Process context
        if (length(parsed$context) == 0) {
          return("No relevant context found.")
        }
        
        # Build context string
        context_parts <- lapply(1:length(parsed$context), function(i) {
          item <- parsed$context[[i]]
          if (is.list(item) && is.null(item$source) == FALSE && is.null(item$file_type) == FALSE && is.null(item$content) == FALSE) {
            return(paste0("[Source: ", item$source, " | Type: ", item$file_type, "]\n", item$content))
          } else {
            return(NULL)
          }
        })
        
        # Remove NULL entries and combine
        null_flags <- sapply(context_parts, is.null)
        context_parts <- context_parts[null_flags == FALSE]
        context_str <- paste(context_parts, collapse = "\n\n")
        
        if (nchar(context_str) > 0) {
          return(context_str)
        } else {
          return("Extracted context was empty.")
        }
      } else {
        if (is.null(parsed) == FALSE && is.null(parsed$error) == FALSE) {
          return(paste("Error retrieving context:", parsed$error))
        } else {
          return("Error: Could not parse context response.")
        }
      }
    } else {
      return("Error: No JSON data was extracted from the response.")
    }
  }, error = function(e) {
    return(paste("Error in context retrieval:", conditionMessage(e)))
  })
}

# Set up theme options
theme_options <- list(
  light = bs_theme(
    version = 5,
    bootswatch = "flatly",
    base_font = font_google("Roboto"),
    heading_font = font_google("Press Start 2P")
  ),
  dark = bs_theme(
    version = 5,
    bootswatch = "darkly",
    base_font = font_google("Roboto"),
    heading_font = font_google("Press Start 2P"),
    bg = "#121212",
    fg = "#f8f9fa"
  )
)
