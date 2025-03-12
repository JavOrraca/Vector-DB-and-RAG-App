# Main app file that sources ui.R, server.R, and global.R

# Source necessary files
source("global.R")
source("ui.R")
source("server.R")

# Create and launch Shiny app
shinyApp(ui, server)