# server.R

library(shiny)
library(reticulate)

# Define server logic required to summarize and view the selected file
shinyServer(function(input, output) {
  
  # R script execution
  observeEvent(input$run_analysis, {
    req(input$ExerciseTrainingData, input$Jumps, input$Wellness, input$PlayerTrainingData, input$StrengthTraining)
    
    source('volleyball.R')
    output$r_output <- renderText({"R Script executed successfully!"})
  })
  
  # Python script execution
  observeEvent(input$run_analysis, {
    req(input$ExerciseTrainingData, input$Jumps, input$Wellness, input$PlayerTrainingData, input$StrengthTraining)
    
    py_run_file('xgboost.py')
    output$py_output <- renderText({"Python Script executed successfully!"})
  })
})
