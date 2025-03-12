# Install required R packages for the Vector DB and RAG App

# List of required packages
required_packages <- c(
  "shiny",
  "bslib",
  "shinyjs",
  "htmltools",
  "httr2",
  "shinychat",
  "ellmer",
  "reticulate",
  "jsonlite",
  "fs",
  "future",
  "promises",
  "markdown",
  "tools"
)

# Find which packages are not installed
not_installed <- required_packages[!required_packages %in% installed.packages()[,"Package"]]

# Install missing packages
if (length(not_installed) > 0) {
  cat("Installing missing packages:", paste(not_installed, collapse = ", "), "\n")
  install.packages(not_installed, repos = "https://cloud.r-project.org/")
} else {
  cat("All required packages are already installed.\n")
}

# Install development packages from GitHub if needed
github_packages <- c(
  "posit-dev/shinychat",
  "tidyverse/ellmer"
)

# Install or update GitHub packages
for (package in github_packages) {
  pkg_name <- basename(package)
  if (!requireNamespace(pkg_name, quietly = TRUE)) {
    cat("Installing", pkg_name, "from GitHub\n")
    devtools::install_github(package)
  }
}

cat("All dependencies installed successfully!\n")