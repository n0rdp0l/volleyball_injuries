# ui.R

library(shiny)
library(shinyWidgets)

# Define UI for data upload app
shinyUI(fluidPage(
  
  # App title
  titlePanel("Volleyball Training Data Analysis"),
  
  # Sidebar layout with input and output definitions
  sidebarLayout(
    
    # Sidebar panel for inputs
    sidebarPanel(
      fileInput("ExerciseTrainingData", "Upload 'ExerciseTrainingData.csv'", accept = ".csv"),
      fileInput("Jumps", "Upload 'Jumps.csv'", accept = ".csv"),
      fileInput("Wellness", "Upload 'Wellness.csv'", accept = ".csv"),
      fileInput("PlayerTrainingData", "Upload 'PlayerTrainingData.csv'", accept = ".csv"),
      fileInput("StrengthTraining", "Upload 'StrengthTraining.csv'", accept = ".csv"),
      actionButton("run_analysis", "Run Analysis")
    ),
    
    # Main panel for displaying outputs
    mainPanel(
      div(id = "r_output",
          h3("Model Coefficients"),
          textOutput("coefficients")
      ),
      div(id = "py_output",
          h3("Predicted Injury Score"),
          progressBar(id = "injury_score", value = 0, total = 10, display_pct = TRUE, status = "success")
      )
    )
  )
))
