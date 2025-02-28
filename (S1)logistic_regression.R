# ------------------------------------------------------------------------------
# Logistic Regression Overview:
# ------------------------------------------------------------------------------
# Logistic Regression is one of the fundamental algorithms in machine learning
# for classification problems. Unlike Linear Regression, which predicts continuous
# values, Logistic Regression estimates the probability that a given input
# belongs to a particular class (e.g., Spam or Not Spam).
#
# ------------------------------------------------------------------------------
# Why use Logistic Regression?
# ------------------------------------------------------------------------------
# - Simple and easy to interpret.
# - Works well for linearly separable datasets.
# - Outputs probabilities, allowing for decision-making with a threshold (e.g., 0.5).
# - Serves as a baseline model before trying more complex classification algorithms
#   like Random Forest or Neural Networks.
#
# ------------------------------------------------------------------------------
# Common Applications:
# ------------------------------------------------------------------------------
# - Email Classification (Spam vs. Not Spam)
# - Medical Diagnosis (Disease Present vs. Not Present)
# - Credit Scoring (Loan Default vs. No Default)
# - Marketing Campaigns (Will a customer buy the product?)
#
# ------------------------------------------------------------------------------
# Logistic Regression Formula:
# ------------------------------------------------------------------------------
#     ŷ = 1 / (1 + e^-(β0 + β1x1 + β2x2 + ... + βnxn))
#
# Where:
# - ŷ (Predicted Probability) → The estimated probability that the target value is 1.
# - β0 (Intercept) → The bias term, shifting the decision boundary.
# - β1, β2, ..., βn (Coefficients) → The weights assigned to each feature.
# - x1, x2, ..., xn → Feature values used for prediction.
# - e (Euler’s Number) → The base of the natural logarithm (~2.718).
# - ε (Error Term) → Represents model uncertainty and randomness in real-world data.
#
# ------------------------------------------------------------------------------
# Key Insights:
# ------------------------------------------------------------------------------
# - Logistic Regression is a statistical model used for binary classification (e.g., spam vs. not spam).
# - The sigmoid function transforms predictions into probabilities between 0 and 1.
# - Unlike Linear Regression, Logistic Regression does not predict continuous values
#   but rather the likelihood of an event.
# - The decision boundary is set by selecting a probability threshold (e.g., 0.5).
# - Works well with interpretable, linearly separable problems and is the foundation
#   of many ML models.
# - Logistic Regression is simple yet effective, making it a great choice for classification problems.
# ------------------------------------------------------------------------------


# Load required libraries
library(caret)        # Machine learning utilities
library(e1071)        # Support Vector Machines (SVM)
library(randomForest) # Random Forest model

# Load dataset
df <- read.csv("Social_Network_Ads.csv")  # Read dataset from CSV file

# Convert target variable to factor (for classification)
df$Purchased <- as.factor(df$Purchased)  # Convert to factor to ensure classification

# Split data into training and testing sets (70% train, 30% test)
set.seed(123)  # Set random seed for reproducibility
train_index <- createDataPartition(df$Purchased, p = 0.7, list = FALSE)  # Generate indices for training set
train_data <- df[train_index, ]  # Create training set
test_data <- df[-train_index, ]  # Create test set

# ---- Logistic Regression ----
log_model <- glm(Purchased ~ Age + EstimatedSalary, data = train_data, family = binomial)  # Train logistic regression model
prob_log <- predict(log_model, test_data, type = "response")  # Predict probabilities
pred_log <- ifelse(prob_log > 0.5, 1, 0)  # Convert probabilities to binary classes
log_cm <- confusionMatrix(as.factor(pred_log), test_data$Purchased)  # Compute confusion matrix

# ---- SVM ----
svm_model <- svm(Purchased ~ Age + EstimatedSalary, data = train_data, kernel = "linear")  # Train SVM model
pred_svm <- predict(svm_model, test_data)  # Get predictions
svm_cm <- confusionMatrix(pred_svm, test_data$Purchased)  # Compute confusion matrix

# ---- Random Forest ----
rf_model <- randomForest(Purchased ~ Age + EstimatedSalary, data = train_data, ntree = 100)  # Train Random Forest model
pred_rf <- predict(rf_model, test_data)  # Get predictions
rf_cm <- confusionMatrix(pred_rf, test_data$Purchased)  # Compute confusion matrix

# ---- Function to plot decision boundary with background colors ----
plot_decision_boundary <- function(model, model_name) {
  # Generate grid points for visualization
  X1 <- seq(min(df$Age) - 1, max(df$Age) + 1, by = 0.1)  # Age range
  X2 <- seq(min(df$EstimatedSalary) - 1000, max(df$EstimatedSalary) + 1000, by = 500)  # Salary range
  grid_set <- expand.grid(X1, X2)  # Create grid of all possible values
  colnames(grid_set) <- c("Age", "EstimatedSalary")  # Rename columns
  
  # Predict values for each point in the grid
  if ("randomForest" %in% class(model)) {
    y_grid <- predict(model, grid_set, type = "class")  # Predictions for Random Forest
  } else if ("glm" %in% class(model)) {
    prob_grid <- predict(model, grid_set, type = "response")  # Logistic Regression probabilities
    y_grid <- ifelse(prob_grid > 0.5, 1, 0)  # Convert probabilities to binary classes
  } else if ("svm" %in% class(model)) {
    y_grid <- predict(model, grid_set)  # Predictions for SVM
  }
  
  y_grid <- as.numeric(as.character(y_grid))  # Convert predictions to numeric
  
  # Plot decision boundary with background colors
  filled.contour(x = X1, y = X2, z = matrix(y_grid, length(X1), length(X2)),
                 color.palette = function(n) c("tomato", "springgreen3"),  # Red for 0, green for 1
                 plot.title = title(main = paste(model_name, "Decision Boundary"),
                                    xlab = "Age", ylab = "Estimated Salary"),
                 key.title = title(main = "Class"),
                 plot.axes = {
                   axis(1)  # X-axis
                   axis(2)  # Y-axis
                   # Plot test data points
                   points(test_data[, c("Age", "EstimatedSalary")], pch = 21, 
                          bg = ifelse(test_data$Purchased == 1, "green", "red"), col = "black")  # Green for purchased, red for not purchased
                 })
}

# ---- Plot all decision boundaries ----
par(mfrow = c(1, 1))  # Reset plotting layout
plot_decision_boundary(log_model, "Logistic Regression")
plot_decision_boundary(svm_model, "SVM")
plot_decision_boundary(rf_model, "Random Forest")

## ---- Create Performance Table ----
perf_results <- data.frame(
  Model = c("Logistic Regression", "SVM", "Random Forest"),
  Precision = c(
    log_cm$byClass["Precision"],
    svm_cm$byClass["Precision"],
    rf_cm$byClass["Precision"]
  ),
  Recall = c(
    log_cm$byClass["Recall"],
    svm_cm$byClass["Recall"],
    rf_cm$byClass["Recall"]
  ),
  Accuracy = c(
    log_cm$overall["Accuracy"],
    svm_cm$overall["Accuracy"],
    rf_cm$overall["Accuracy"]
  ),
  F1_Score = c(
    2 * ((log_cm$byClass["Precision"] * log_cm$byClass["Recall"]) / (log_cm$byClass["Precision"] + log_cm$byClass["Recall"])),
    2 * ((svm_cm$byClass["Precision"] * svm_cm$byClass["Recall"]) / (svm_cm$byClass["Precision"] + svm_cm$byClass["Recall"])),
    2 * ((rf_cm$byClass["Precision"] * rf_cm$byClass["Recall"]) / (rf_cm$byClass["Precision"] + rf_cm$byClass["Recall"]))
  )
)

# ---- Display Performance Table in Viewer ----
datatable(perf_results, options = list(pageLength = 5), rownames = TRUE)  # Show performance metrics in interactive table

# Function to visualize confusion matrix
plot_confusion_matrix <- function(cm, model_name) {
  cm_table <- as.data.frame(cm$table)  # Convert confusion matrix to data frame
  colnames(cm_table) <- c("Prediction", "Reference", "Freq")  # Rename columns
  
  cm_table$Prediction <- as.factor(cm_table$Prediction)  # Convert to factor
  cm_table$Reference <- as.factor(cm_table$Reference)  # Convert to factor
  
  ggplot(cm_table, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile() +  # Create a heatmap
    geom_text(aes(label = Freq), color = "white", size = 5) +  # Add frequency text
    scale_fill_gradient(low = "blue", high = "red") +  # Color scale
    labs(title = paste("Confusion Matrix -", model_name),
         x = "Predicted", y = "Actual") +
    theme_minimal()
}

# Generate confusion matrices for all models
cm1 <- plot_confusion_matrix(log_cm, "Logistic Regression")
cm2 <- plot_confusion_matrix(rf_cm, "Random Forest")
cm3 <- plot_confusion_matrix(svm_cm, "SVM")

# Display all confusion matrices in one row
grid.arrange(cm1, cm2, cm3, ncol = 3)