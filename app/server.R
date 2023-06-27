# server.R

library(shiny)
library(reticulate)
library(shinyWidgets)

# Define server logic required to summarize and view the selected file
shinyServer(function(input, output, session) {
  
  # R script execution
  observeEvent(input$run_analysis, {
    req(input$ExerciseTrainingData, input$Jumps, input$Wellness, input$PlayerTrainingData, input$StrengthTraining)
    
    
    coefficients <- source('volleyball.R')$value
    output$coefficients <- renderPrint({coefficients})
  })
  
  # Python script execution
  observeEvent(input$run_analysis, {
    req(input$ExerciseTrainingData, input$Jumps, input$Wellness, input$PlayerTrainingData, input$StrengthTraining)
    
    
    predicted_scores <- py_run_file('xgboost.py')$value
    injury_score <- predicted_scores[1, "score"]
    
    status <- ifelse(injury_score <= 5, "success", "danger")
    updateProgressBar(session, "injury_score", value = injury_score, status = status)
  })
})
