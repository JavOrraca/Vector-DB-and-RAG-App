server <- function(input, output, session) {
  # Enable shinyjs
  useShinyjs()
  
  # State variables
  uploaded_files <- reactiveVal(list())
  temp_dir <- reactiveVal(tempdir())
  vector_db_path <- reactiveVal(NULL)
  chat_history <- reactiveVal(list())
  current_theme <- reactiveVal("dark")
  
  # Update theme when toggle button is clicked
  observeEvent(input$theme_toggle, {
    new_theme <- if (current_theme() == "dark") "light" else "dark"
    current_theme(new_theme)
    session$setCurrentTheme(theme_options[[new_theme]])
  })
  
  # Handle file uploads
  observeEvent(input$file_upload, {
    new_files <- input$file_upload
    
    # Add to uploaded files list
    current_files <- uploaded_files()
    for (i in 1:nrow(new_files)) {
      file_info <- list(
        name = new_files$name[i],
        size = format(new_files$size[i] / 1024, digits = 2),
        type = tools::file_ext(new_files$name[i]),
        path = new_files$datapath[i]
      )
      current_files[[length(current_files) + 1]] <- file_info
    }
    
    uploaded_files(current_files)
  })
  
  # Upload folder button (opens a modal dialog with instructions)
  observeEvent(input$upload_folder, {
    showModal(modalDialog(
      title = "Upload Folder",
      p("To upload an entire folder:"),
      tags$ol(
        tags$li("Select all files in the folder"),
        tags$li("Drag and drop them onto the file upload area")
      ),
      p("Note: Some browsers have limitations with folder uploads. For best results, use Chrome."),
      easyClose = TRUE,
      footer = modalButton("Close")
    ))
  })
  
  # Display file count
  output$file_count <- renderText({
    length(uploaded_files())
  })
  
  # Display vector DB status
  output$db_status <- renderText({
    if (is.null(vector_db_path())) {
      "Not Created"
    } else {
      "Created"
    }
  })
  
  # Set DB created status
  output$db_created <- reactive({
    !is.null(vector_db_path())
  })
  outputOptions(output, "db_created", suspendWhenHidden = FALSE)
  
  # Display uploaded files
  output$file_list <- renderTable({
    files <- uploaded_files()
    if (length(files) == 0) {
      return(data.frame(
        Name = character(0),
        Type = character(0),
        Size = character(0)
      ))
    }
    
    data.frame(
      Name = sapply(files, function(f) f$name),
      Type = sapply(files, function(f) toupper(f$type)),
      Size = sapply(files, function(f) paste0(f$size, " KB"))
    )
  })
  
  # Create vector DB when run button is clicked
  observeEvent(input$run_button, {
    req(length(uploaded_files()) > 0)
    
    # Show progress
    withProgress(message = "Creating Vector Database", value = 0, {
      incProgress(0.1, detail = "Preparing files...")
      
      # Get all file paths
      file_paths <- sapply(uploaded_files(), function(f) f$path)
      
      # Create a new temporary directory for this run
      run_temp_dir <- file.path(tempdir(), paste0("vector_db_", format(Sys.time(), "%Y%m%d%H%M%S")))
      dir.create(run_temp_dir, showWarnings = FALSE, recursive = TRUE)
      temp_dir(run_temp_dir)
      
      # Create vector DB (this will take some time)
      incProgress(0.2, detail = "Processing files...")
      
      # Get values outside of future context
      chunk_size_val <- input$chunk_size
      chunk_overlap_val <- input$chunk_overlap
      
      # Run the ingestion asynchronously
      future_promise({
        tryCatch({
          db_path <- create_vector_db(
            file_paths = file_paths,
            temp_dir = run_temp_dir,
            chunk_size = chunk_size_val,
            chunk_overlap = chunk_overlap_val
          )
          list(success = TRUE, db_path = db_path)
        }, error = function(e) {
          list(success = FALSE, error = as.character(e))
        })
      }) %...>% 
        (function(result) {
          if (result$success) {
            vector_db_path(result$db_path)
            showNotification("Vector database created successfully!", type = "message")
          } else {
            showNotification(paste("Error creating vector database:", result$error), type = "warning")
          }
        })
    })
  })
  
  # Download handler for the vector DB
  output$download_db <- downloadHandler(
    filename = function() {
      "vector_db.zip"
    },
    content = function(file) {
      # Package the vector DB as a zip file
      zip_file <- create_vector_db_zip(vector_db_path(), temp_dir())
      file.copy(zip_file, file)
    },
    contentType = "application/zip"
  )
  
  # Chat UI
  # Render chat messages
  output$chat_messages <- renderUI({
    history <- chat_history()
    if (length(history) == 0) {
      return(div(
        p("Ask a question about your uploaded files. The system will use your vector database to find relevant context before sending to the LLM."),
        style = "color: #6c757d; font-style: italic; text-align: center; margin-top: 20px;"
      ))
    }
    
    message_elements <- lapply(1:length(history), function(i) {
      msg <- history[[i]]
      if (msg$sender == "user") {
        div(
          div(msg$text,
              style = "background-color: #007bff; color: white; padding: 10px 15px; border-radius: 15px; display: inline-block; max-width: 80%; margin-bottom: 10px;"),
          style = "text-align: right; margin-bottom: 10px;"
        )
      } else {
        div(
          div(HTML(markdown::markdownToHTML(text = msg$text, fragment.only = TRUE)),
              style = "background-color: #f8f9fa; color: #212529; padding: 10px 15px; border-radius: 15px; display: inline-block; max-width: 80%; margin-bottom: 10px;"),
          style = "text-align: left; margin-bottom: 10px;"
        )
      }
    })
    
    do.call(tagList, message_elements)
  })
  
  # Handle sending messages
  observeEvent(input$send_message, {
    req(input$chat_input, vector_db_path())
    
    # Get the user's message
    user_message <- input$chat_input
    
    # Clear the input
    updateTextInput(session, "chat_input", value = "")
    
    # Add the user message to chat history
    current_history <- chat_history()
    current_history[[length(current_history) + 1]] <- list(
      sender = "user",
      text = user_message
    )
    chat_history(current_history)
    
    # Add a placeholder for the assistant's response
    current_history <- chat_history()
    current_history[[length(current_history) + 1]] <- list(
      sender = "assistant",
      text = "_Thinking..._"
    )
    chat_history(current_history)
    
    # Get and store values outside of future context
    api_key <- input$api_key
    if (input$api_key == "****") {
      api_key <- if (input$llm_provider == "anthropic") {
        Sys.getenv("ANTHROPIC_API_KEY")
      } else {
        Sys.getenv("OPENAI_API_KEY")
      }
    }
    
    # Store other reactive values
    current_provider <- input$llm_provider
    current_model <- input$model_name
    current_system_prompt <- input$system_prompt
    current_db_path <- vector_db_path()
    
    # Process the message asynchronously
    future_promise({
      # Get relevant context from vector DB
      context <- get_context(user_message, current_db_path)
      
      # Get response from LLM
      response <- create_chat_completion(
        prompt = user_message,
        context = context,
        provider = current_provider,
        model = current_model,
        api_key = api_key,
        system_prompt = current_system_prompt
      )
      
      response
    }) %...>%
      (function(response) {
        # Update the assistant's response in the chat history
        current_history <- chat_history()
        current_history[[length(current_history)]]$text <- response
        chat_history(current_history)
        
        # Scroll to bottom of chat
        runjs("document.getElementById('chat_container').scrollTop = document.getElementById('chat_container').scrollHeight;")
      })
    
    # Scroll to bottom of chat
    runjs("document.getElementById('chat_container').scrollTop = document.getElementById('chat_container').scrollHeight;")
  })
  
  # Also send message when Enter key is pressed in the input
  observeEvent(input$chat_input, {
    if (input$chat_input != "" && substr(input$chat_input, nchar(input$chat_input), nchar(input$chat_input)) == "\n") {
      # Remove the newline character
      updateTextInput(session, "chat_input", value = substr(input$chat_input, 1, nchar(input$chat_input) - 1))
      # Send the message
      click("send_message")
    }
  })
  
  # Update LLM config based on provider selection
  observeEvent(input$llm_provider, {
    if (input$llm_provider == "anthropic") {
      updateTextInput(session, "model_name", value = "claude-3-7-sonnet-20250219")
      if (Sys.getenv("ANTHROPIC_API_KEY") != "") {
        updateTextInput(session, "api_key", value = "****")
      } else {
        updateTextInput(session, "api_key", value = "")
      }
    } else {
      updateTextInput(session, "model_name", value = "gpt-4o")
      if (Sys.getenv("OPENAI_API_KEY") != "") {
        updateTextInput(session, "api_key", value = "****")
      } else {
        updateTextInput(session, "api_key", value = "")
      }
    }
  })
}