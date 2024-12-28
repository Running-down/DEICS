
library(shiny)
library(caret)
library(xgboost)
library(shinydashboard)

# Read the XGBoost model from the RDS file
model_xgbtree <- readRDS("model_xgbtree.rds")

# Create a reactiveVal to store the prediction result
prediction_reactive <- reactiveVal(NULL)

# Create UI interface
ui <- fluidPage(
  titlePanel("Diagnostic Model of Early Infection after Cardiovascular Surgery (DEICS)"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("age", "Age", min = 18, max = 80, value = 56),
      selectInput("gender", "Gender", choices = c("male", "female"), selected = "female"),
      numericInput("BMI", "BMI (kg/m2)", value = 20.2),
      selectInput("nyha", "NYHA class ≥ III", choices = c("YES", "NO"), selected = "NO"),
      selectInput("history_of_smoke", "Smoke history", choices = c("YES", "NO"), selected = "NO"),
      selectInput("history_of_drink", "Drink history", choices = c("YES", "NO"), selected = "NO"),
      selectInput("hypertension", "Hypertension", choices = c("YES", "NO"), selected = "NO"),
      selectInput("diabetes", "Diabetes with insulin", choices = c("YES", "NO"), selected = "NO"),
      selectInput("stroke", "Stroke", choices = c("YES", "NO"), selected = "NO"),
      selectInput("CAD", "CAD", choices = c("YES", "NO"), selected = "YES"),
      selectInput("MI", "MI", choices = c("YES", "NO"), selected = "NO"),
      selectInput("eGFE60", "eGFR < 60 ml/min", choices = c("YES", "NO"), selected = "NO"),
      selectInput("COPD", "COPD", choices = c("YES", "NO"), selected = "NO"),
      selectInput("Peripheral_arterial_disease", "Peripheral arterial disease", choices = c("YES", "NO"), selected = "NO"),
      numericInput("LVEF", "LVEF (%)", value = 59),
      selectInput("SOFA_1", "SOFA score ≥ 1", choices = c("YES", "NO"), selected = "NO"),
      numericInput("ALB_0", "Albumin (g/L)", value = 41.8),
      selectInput("history_of_cardiac_srgery", "History of cardiac surgery", choices = c("YES", "NO"), selected = "NO"),
      selectInput("emergency", "Emergency surgery", choices = c("YES", "NO"), selected = "NO"),
      selectInput("Procedure_name", "Procedure name", choices = c("Isolated CABG", "AVR or MVR or TVR","AVR+MVR","Valve + CABG surgery","Thoracic aortic surgery","Others"), selected = "Thoracic aortic surgery"),
      numericInput("CPB", "CPB (min)", value = 73),
      numericInput("ACC", "ACC (min)", value = 52),
      numericInput("DHCA", "DHCA (min)", value = 0),
      selectInput("incision", "Minimally invasive", choices = c("YES", "NO"), selected = "NO"),
      numericInput("RBCs", "RBCs (u)", value = 3),
      numericInput("plasma", "Plasma (ml)", value = 600),
      selectInput("platelet", "Platelet", choices = c("YES", "NO"), selected = "NO"),
      selectInput("cryoprecipitate", "Cryoprecipitate", choices = c("YES", "NO"), selected = "NO"),
      selectInput("EorCorI", "Use of ECMO or IABP or CRRT", choices = c("YES", "NO"), selected = "NO"),
      selectInput("X_Ray", "CXR Exudation", choices = c("Mild", "Moderate", "Severe"), selected = "Moderate"),
      numericInput("MV", "Duration of MV (h)", value = 6.9),
      numericInput("PCT", "Procalcitonin on POD3 (ng/ml)", value = 1.12),
      numericInput("CRP", "C-reactive protein on POD3 (ng/ml)", value = 89.5),
      numericInput("IL6", "Interleukin-6 on POD3 (ng/ml)", value = 17.8),
      # Button to trigger prediction
      actionButton("predict", "Predict")
    ),
    mainPanel(
      # Display the prediction result
      verbatimTextOutput("prediction")
    )
  )
)


# Server logic
server <- function(input, output) {
  
  observeEvent(input$predict, {
    tryCatch({
      # Map procedure names to numerical values
      map_procedure_name <- c(
        "Isolated CABG" = 1,
        "AVR or MVR or TVR" = 2,
        "AVR+MVR" = 3,
        "Valve + CABG surgery" = 4,
        "Thoracic aortic surgery" = 5,
        "Others" = 6
      )
      
      # Extract user-inputted feature values
      features <- data.frame(
        age = input$age,
        gender = ifelse(input$gender == "male", 1, 0),
        BMI = input$BMI,
        nyha = ifelse(input$nyha == "YES", 1, ifelse(input$nyha == "NO", 0, 0)),
        history_of_smoke = ifelse(input$history_of_smoke == "YES", 1, 0),
        history_of_drink = ifelse(input$history_of_drink == "YES", 1, 0),
        hypertension = ifelse(input$hypertension == "YES", 1, 0),
        diabetes = ifelse(input$diabetes == "YES", 1, 0),
        stroke = ifelse(input$stroke == "YES", 1, 0),
        CAD = ifelse(input$CAD == "YES", 1, 0),
        MI = ifelse(input$MI == "YES", 1, 0),
        eGFE60 = ifelse(input$eGFE60 == "YES", 1, 0),
        COPD = ifelse(input$COPD == "YES", 1, 0),
        Peripheral_arterial_disease = ifelse(input$Peripheral_arterial_disease == "YES", 1, 0),
        LVEF = input$LVEF,
        SOFA_1 = ifelse(input$SOFA_1 == "YES", 1, 0),
        ALB_0 = input$ALB_0,
        history_of_cardiac_srgery = ifelse(input$history_of_cardiac_srgery == "YES", 1, 0),
        emergency = ifelse(input$emergency == "YES", 1, 0),
        Procedure_name = map_procedure_name[input$Procedure_name],
        CPB = input$CPB,
        ACC = input$ACC,
        DHCA = input$DHCA,
        incision = ifelse(input$incision == "YES", 1, 0),
        RBCs = input$RBCs,
        plasma = input$plasma,
        platelet = ifelse(input$platelet == "YES", 1, 0),
        cryoprecipitate = ifelse(input$cryoprecipitate == "YES", 1, 0),
        EorCorI = ifelse(input$EorCorI == "YES", 1, 0),
        X_Ray = match(input$X_Ray, c("Mild", "Moderate", "Severe")) - 1,
        MV = input$MV,
        PCT = input$PCT,
        CRP = input$CRP,
        IL6 = input$IL6
      )
      
      features_df <- as.data.frame(features)  
      
      # Convert logical inputs to numeric
      logical_inputs <- c("nyha", "history_of_smoke","history_of_drink", "hypertension", "diabetes", "stroke", "CAD", 
                          "MI", "eGFE60", "COPD", "Peripheral_arterial_disease", "SOFA_1", 
                          "history_of_cardiac_srgery", "emergency", "incision", "platelet", 
                          "cryoprecipitate", "EorCorI")
      
      # Check if column names exist
      invalid_columns <- setdiff(logical_inputs, colnames(features_df))
      if (length(invalid_columns) > 0) {
        stop(paste("Error: The following column names do not exist in the data frame:", paste(invalid_columns, collapse = ", ")))
      }
      
      # Convert to numeric
      features_df[logical_inputs] <- lapply(features_df[logical_inputs], as.numeric)
      
      # Reorder columns to match model's feature_names
      features_df <- features_df[, model_xgbtree$finalModel$feature_names]
      
      # Check if column names match
      if (!identical(colnames(features_df), model_xgbtree$finalModel$feature_names)) {
        stop("错误：重新排列列后的 features_df 列名与模型的 feature_names 不一致")
      }
      
      features_matrix_data <- as.matrix(features_df)
      
      features_matrix <- xgb.DMatrix(data = features_matrix_data)
      
      # Use XGBoost model for prediction (obtain predictions for continuous variable)
      prediction <- predict(model_xgbtree$finalModel, newdata = features_matrix, type = "response")
      
      # Round the prediction result to six decimal places
      prediction_val <- round(1 - prediction, 6)
      
      # Save the prediction result to reactiveVal
      prediction_reactive(prediction_val)
      
    }, error = function(e) {
      # Handle errors, e.g., print error messages
      cat("Error message:", conditionMessage(e), "\n")
      cat("Error trace:", conditionCall(e), "\n")
      cat("Error type:", conditionClass(e), "\n")
      
      # Return NULL or other appropriate default value in case of an error
      prediction_reactive(NULL)
    })
  })
  
  # In the render function for output$prediction, return the prediction value instead of NULL
  output$prediction <- renderPrint({
    # Get the value from prediction_reactive
    prediction_val <- prediction_reactive()
    
    # If prediction_val is not NULL, display the prediction result
    if (!is.null(prediction_val)) {
      paste("Predicted Result：", prediction_val)
    } else {
      "Predicted Result: NA"
    }
  })
}
# Launch the Shiny application
shinyApp(ui, server)
