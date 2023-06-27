library(readr)
library(dplyr)
library(ggplot2)
library(visdat)
library(naniar)
library(tidyverse)
library(zoo)
library(lubridate)
library(caret)
library(ordinal)
library(MASS)
library(glmnet) 
library(ranger)
library(car)
library(broom)
library(xtable)

ExerciseTrainingData <- read_csv2('ExerciseTrainingData.csv', col_types = cols())
Jumps <- read_csv2('Jumps.csv', col_types = cols())
Wellness <- read_csv2('Wellness.csv', col_types = cols())
PlayerTrainingData <- read_csv('PlayerTrainingData.csv', col_types = cols())
StrengthTraining <- read_csv2('StrengthTraining.csv', col_types = cols())

# Convert date columns to date type for all data frames
ExerciseTrainingData$Date <- as.Date(ExerciseTrainingData$Date, format = "%d-%m-%Y")
Jumps$Date <- as.Date(Jumps$Date, format = "%d-%m-%Y")
StrengthTraining$Date <- as.Date(StrengthTraining$Date, format = "%d-%m-%Y")
Wellness$Date <- as.Date(Wellness$Date, format = "%d-%m-%Y")

# merge the data frames by the "date" columns
df <- Reduce(function(x, y) merge(x, y, by = c("Date", "PlayerID"), all = TRUE),
                    list(ExerciseTrainingData, Jumps, StrengthTraining, Wellness))
df <- merge(df, PlayerTrainingData, by =  c("TrainingID", "PlayerID"))

df <- df[, -which(names(df) == "DateTime")]

## adress duration variables 
df$Duration.player <- as.character(df$Duration.y)
df$Duration.exercise <- as.character(df$Duration.x)
df <- df[, -which(names(df) == "Duration.y")]
df <- df[, -which(names(df) == "Duration.x")]

df$Duration.player <-sapply(df$Duration.player, function(x) {
  # Split by space
  x <- strsplit(x, " ")[[1]]
  # Handle NA values
  if(length(x) == 1 && x[1] == "NA") {
    return(NA)
  }
  # Remove "days" and rejoin
  x <- x[-(1:2)]
  # Remove milliseconds
  x[length(x)] <- strsplit(x[length(x)], "\\.")[[1]][1]
  # Combine back into a time string
  x <- paste(x, collapse = ":")
  return(x)
})


df$Duration.exercise <- sapply(df$Duration.exercise, function(x) {
  # Split by space
  x <- strsplit(x, " ")[[1]]
  # Handle NA values
  if(length(x) == 1 && x[1] == "NA") {
    return(NA)
  }
  # Remove "days" and rejoin
  x <- x[-(1:2)]
  # Remove milliseconds
  x[length(x)] <- strsplit(x[length(x)], "\\.")[[1]][1]
  # Combine back into a time string
  x <- paste(x, collapse = ":")
  return(x)
})


df$Duration.player <- hms(df$Duration.player)
df$Duration.exercise <- hms(df$Duration.exercise)

# Convert to numeric
df$Duration.player <- as.numeric(df$Duration.player, units="secs")
df$Duration.exercise <- as.numeric(df$Duration.exercise, units="secs")

df <- df[, -which(names(df) == "DateEndTime")]
df <- df[, -which(names(df) == "DateStartTime")]




# Create a list of data frames, each containing a sequence of all dates within the date range for each player
date_sequences <- lapply(unique(df$PlayerID), function(id) {
  date_range <- range(df$Date[df$PlayerID == id])
  data.frame(Date = seq(date_range[1], date_range[2], by = "day"),
             PlayerID = id)
})

# Combine the data frames in the list into a single data frame
all_dates <- do.call(rbind, date_sequences)

# Join the sequence of all dates with the original data frame
df <- full_join(all_dates, df, by = c("Date", "PlayerID"))

df_no.lags <- df



df$Date <- as.Date(df$Date)

# a list of column names to create lagged versions of
numeric_vars <- names(df)[sapply(df, is.numeric)]
# Remove 'PlayerID' and 'injury' from the list
numeric_vars <- numeric_vars[!(numeric_vars %in% c("PlayerID", "injury"))]

# List to store names of newly created lagged variables
lagged_vars <- c()

# Step 1:  lagged variables for all variables (except for PlayerID, date, and injury)
for (var in names(df)[!(names(df) %in% c("PlayerID", "Date", "injury"))]) {
  for (lag in 1:3) {
    lag_var_name <- paste(var, "lag", lag, sep = "_")
    df <- df %>%
      group_by(PlayerID) %>%
      mutate(!!lag_var_name := lag(!!sym(var), lag))
    # Append newly created lagged variable name to the list
    lagged_vars <- c(lagged_vars, lag_var_name)
  }
}

# Step 2: Delete original non-lagged, non-numeric variables
df <- df %>%
  dplyr::select(c("PlayerID", "Date", "Injury", numeric_vars, lagged_vars))

# Step 3: Create rolling sum variables for the original numeric variables
for (var in numeric_vars) {
  for (roll in c(3, 7)) {
    roll_var_name <- paste(var, "roll_sum", roll, sep = "_")
    df <- df %>%
      group_by(PlayerID) %>%
      arrange(Date) %>%
      mutate(!!roll_var_name := rollapplyr(!!sym(var), width = roll+1, FUN = function(x) sum(x, na.rm = TRUE), fill = NA, align = "right"))
  }
}

# Step 4: Delete original non-lagged numeric variables (except for 'injury')
df_final <- df %>%
  dplyr::select(-setdiff(numeric_vars, "Injury"))
rm(df)

# Total number of missing values in the data frame
total_na <- sort(sum(is.na(df_final)))

# Number of missing values in each column
col_na <- sort(colSums(is.na(df_final)))


# Sort the data by date
df_final$Date <- as.Date(df_final$Date, format = "%Y-%m-%d")

df_final[, sapply(df_final, is.character)] <- lapply(df_final[, sapply(df_final, is.character)], as.factor)
df_final <- df_final %>% arrange(Date)
df_final$Injury <- as.factor(df_final$Injury)



# Identify the numeric variables (excluding the outcome variable and date)
numeric_vars <- setdiff(names(df_final)[sapply(df_final, is.numeric)], c("PlayerID", "Date"))

# Create a preProcess object for scaling the numeric variables
preproc <- preProcess(df_final[, numeric_vars], method = "scale")

# Scale the numeric variables in the training and testing sets
df_final[, numeric_vars] <- predict(preproc, newdata = df_final[, numeric_vars])

# Determine the date that splits the data into roughly an 80/20 split
split_date <- df_final$Date[round(0.8 * nrow(df_final))]

# Create train and test sets
train <- df_final %>% filter(Date <= split_date)
test <- df_final %>% filter(Date > split_date)

dim(df_no.lags)
df_no.lags <- na.omit(df_no.lags)
dim(df_no.lags)

df_no.lags[, sapply(df_no.lags, is.character)] <- lapply(df_no.lags[, sapply(df_no.lags, is.character)], as.factor)
df_no.lags <- df_no.lags %>% arrange(Date)
#df_no.lags$Injury <- as.factor(df_no.lags$Injury)
df_no.lags <- df_no.lags[, -which(names(df_no.lags) == "Focus")]


# Determine the date that splits the data into roughly an 80/20 split
split_date <- df_no.lags$Date[round(0.7 * nrow(df_no.lags))]

# Create train and test sets
train <- df_no.lags %>% filter(Date <= split_date)
test <- df_no.lags %>% filter(Date > split_date)

train$Date <- as.factor(train$Date)
test$Date <- as.factor(test$Date)


# Prepare data for lasso model
x_train <- model.matrix(Injury ~ .-1, data = train) 
y_train <- train$Injury 

# Fit lasso model
cv.lasso <- cv.glmnet(x_train, y_train, alpha = 1, family = "gaussian") # alpha = 1 for lasso

# Get optimal lambda value
lambda_optimal <- cv.lasso$lambda.min

# Fit lasso model with optimal lambda
lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = lambda_optimal)

# Print lasso model
#lasso_model


# Get coefficients from the lasso model
coefficients <- coef(lasso_model)

# Create a data frame of variable names and their corresponding coefficients
coef_df <- data.frame(Variable = rownames(coefficients), Coefficient = as.numeric(coefficients))

# Remove the intercept
coef_df <- coef_df[coef_df$Variable != "(Intercept)", ]

# Get the variables with non-zero coefficients
final_variables <- coef_df[coef_df$Coefficient != 0, ]$Variable

# Print the final variables
print(final_variables)


rm(df_final)
rm(df_no.lags)


# Prepare data for random forest
# Convert factors to dummy variables for random forest
predictors_rf <- model.matrix(~.-1, data = train[ , !(names(train) %in% "Injury")])

# Create a new data frame for the random forest model
train_set_rf <- data.frame(Injury = train$Injury, predictors_rf)

# Fit random forest model
rf_model <- ranger(Injury ~ ., data = train_set_rf, importance = 'impurity')




train$Injury <- as.factor(train$Injury)
test$Injury <- as.factor(test$Injury)
train$ExerciseID <- as.factor(train$ExerciseID)
test$ExerciseID <- as.factor(test$ExerciseID)
train$TrainingID <- as.factor(train$TrainingID)
test$TrainingID <- as.factor(test$TrainingID)
train$Date <- as.factor(train$Date)
test$Date <- as.factor(test$Date)



# Prepare data for clmm model
variables <- c("TrainingID", "Recovered", "Wellness", "Hours of sleep", "RPE", "Duration.player", "ExerciseID", "Duration_s", "Prct", "Date", "Injury")
train_clmm <- train[, variables]
test_clmm <- test[, variables]

# Select numeric variables, excluding 'TrainingID', 'PlayerID', 'ExerciseID'
numeric_vars <- sapply(train_clmm, is.numeric)
exclude_vars <- c("TrainingID", "PlayerID", "ExerciseID")
numeric_vars[exclude_vars] <- FALSE

# Compute scaling parameters on the training set
preProcValues <- preProcess(train_clmm[, numeric_vars], method = c("center", "scale"))

# Apply the scaling transformation to the training set and test set
train_clmm[, numeric_vars] <- predict(preProcValues, train_clmm[, numeric_vars])
test_clmm[, numeric_vars] <- predict(preProcValues, test_clmm[, numeric_vars])


# Fit clmm model
clmm_model <- clmm(Injury ~ Recovered + `Hours of sleep` + RPE + Duration.player + Wellness + Duration_s + Prct + (1|Date), data = train_clmm)

# Print clmm model
print(summary(clmm_model))

# Predict on test data
predictions <- predict(clmm_model, newdata = test_clmm, type = "class")

# Evaluate model
confusionMatrix(predictions, test_clmm$Injury)

