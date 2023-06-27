# ui.R

library(shiny)

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
      textOutput("r_output"),
      textOutput("py_output")
    )
  )
))
