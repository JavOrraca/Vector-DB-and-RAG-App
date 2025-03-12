ui <- page_navbar(
  title = "Vector DB and RAG App",
  theme = theme_options$dark,  # Default to dark theme
  sidebar = sidebar(
    width = 350,
    title = "Configuration",
    
    # File Upload Section
    card(
      card_header("Upload Files"),
      fileInput("file_upload", "Choose Files", multiple = TRUE, accept = NULL, 
                buttonLabel = "Browse...", placeholder = "No files selected"),
      actionButton("upload_folder", "Upload Folder", class = "btn-secondary"),
      hr(),
      
      # Chunking Parameters
      sliderInput("chunk_size", "Chunk Size", min = 100, max = 2000, value = 1000, step = 100),
      sliderInput("chunk_overlap", "Chunk Overlap", min = 0, max = 500, value = 200, step = 50),
      
      # LLM Configuration
      hr(),
      selectInput("llm_provider", "LLM Provider", 
                  choices = c("Anthropic" = "anthropic", "OpenAI" = "openai"),
                  selected = "anthropic"),
      textInput("model_name", "Model Name", value = "claude-3-7-sonnet-20250219"),
      passwordInput("api_key", "API Key", value = if (Sys.getenv("ANTHROPIC_API_KEY") != "") "****" else ""),
      textAreaInput("system_prompt", "System Role", value = "You are an AI assistant.", 
                    rows = 3, resize = "vertical"),
      
      # Run Button
      hr(),
      actionButton("run_button", "Create Vector DB", class = "btn-primary", width = "100%"),
      conditionalPanel(
        condition = "output.db_created == true",
        downloadButton("download_db", "Download Vector DB", class = "btn-success", width = "100%", 
                      style = "margin-top: 10px;")
      )
    )
  ),
  
  # Main Navigation Tabs
  nav_panel(
    title = "Files & Chat",
    layout_columns(
      fill = FALSE,
      value_box(
        title = "Uploaded Files",
        value = textOutput("file_count"),
        showcase = bsicons::bs_icon("file-earmark"),
        theme = "primary",
        full_screen = TRUE,
        fill = TRUE
      ),
      value_box(
        title = "Vector DB Status",
        value = textOutput("db_status"),
        showcase = bsicons::bs_icon("database"),
        theme = "secondary", 
        full_screen = TRUE,
        fill = TRUE
      )
    ),
    
    # Uploaded Files List
    card(
      card_header("Uploaded Files"),
      div(style = "max-height: 300px; overflow-y: auto;",
          tableOutput("file_list")
      )
    ),
    
    # Chat Interface
    conditionalPanel(
      condition = "output.db_created == true",
      card(
        card_header("RAG Chat"),
        div(
          id = "chat_container",
          style = "height: 400px; overflow-y: auto; padding: 15px; margin-bottom: 15px;",
          uiOutput("chat_messages")
        ),
        fluidRow(
          column(10,
                 textInput("chat_input", NULL, width = "100%", placeholder = "Type your message here...")
          ),
          column(2,
                 actionButton("send_message", "Send", width = "100%", class = "btn-primary")
          )
        )
      )
    )
  ),
  
  # About Tab
  nav_panel(
    title = "About",
    card(
      card_header("About Vector DB and RAG App"),
      card_body(
        h3("Overview"),
        p("This application allows you to create a vector database from various file types and use it for Retrieval-Augmented Generation (RAG) with your choice of LLM provider."),
        
        h3("Features"),
        tags$ul(
          tags$li("Upload multiple files or folders"),
          tags$li("Process various file types including documents, code, data files, and more"),
          tags$li("Configure chunking parameters for optimal vector DB creation"),
          tags$li("Chat with your data using either Anthropic or OpenAI models"),
          tags$li("Download your vector database for later use")
        ),
        
        h3("Supported File Types"),
        p("The Vector DB can process many different file types including:"),
        tags$ul(
          tags$li("Text and Markdown: .md, .R, .Rmd, .qmd"),
          tags$li("Python files: .py, .ipynb"),
          tags$li("Web files: .html, .css, .scss, .js"),
          tags$li("Data files: .csv, .json, .parquet, .db, .sqlite"),
          tags$li("R specific data: .rds, .RData"),
          tags$li("Python data: .pkl, .pickle"),
          tags$li("Office documents: .pdf, .xlsx, .docx, .pptx")
        )
      )
    )
  ),
  
  # Theme switcher in the navbar
  nav_spacer(),
  nav_item(
    actionButton("theme_toggle", "Toggle Theme", class = "btn-sm")
  )
)