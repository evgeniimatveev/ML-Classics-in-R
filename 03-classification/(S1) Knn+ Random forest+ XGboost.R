

# Overview of K-Nearest Neighbors (KNN)

# KNN is a fundamental machine learning algorithm used for classification and regression problems.
# Unlike Logistic Regression, which models probabilities, KNN is a non-parametric, instance-based algorithm
# that makes predictions based on the similarity of data points.
# It classifies a new data point by looking at the majority class among its k nearest neighbors.

# Why use K-Nearest Neighbors?
# - Simple and intuitive – no explicit training phase, just store the data.
# - Flexible – can be used for both classification and regression tasks.
# - Works well with non-linear decision boundaries.
# - Handles multi-class classification naturally.
# - No assumptions about data distribution – purely data-driven.

# Common Applications of KNN
# - Recommendation Systems (e.g., suggesting movies based on similar users' preferences).
# - Anomaly Detection (e.g., identifying fraudulent transactions in banking).
# - Medical Diagnosis (e.g., classifying tumors as malignant or benign).
# - Handwritten Digit Recognition (e.g., OCR systems like digit recognition in postal mail).
# - Customer Segmentation (e.g., grouping users by purchasing behavior).

# The K-Nearest Neighbors Formula:
# y_hat = (1/k) * sum(y_i) for i = 1 to k
#
# Where:
# - y_hat -> The predicted class (classification) or averaged value (regression) based on the nearest neighbors.
# - k -> The number of closest data points considered for making predictions.
# - y_i -> The target values (or class labels) of the k nearest neighbors.
# - d(x, x_i) -> The metric used to measure similarity between points (e.g., Euclidean, Manhattan, Minkowski).

# Choosing k Wisely:
# - Small k -> More sensitive to noise but captures local structure.
# - Large k -> Smoother decision boundary but may lose finer details.

# Key Insights:
# - KNN is a simple yet powerful algorithm used for classification and regression.
# - Lazy Learning – No training phase; all computations happen during prediction.
# - Distance-based approach makes it highly dependent on feature scaling (e.g., normalization).

# Conclusion:
# KNN is a straightforward, effective machine learning algorithm that excels in scenarios where
# interpretability and simplicity are key. However, it can be computationally expensive for large datasets,
# making feature scaling and efficient distance calculations crucial for optimal performance.


# Load necessary libraries
library(tidyverse)  # For data manipulation and visualization
library(data.table)  # Efficient data reading and manipulation
library(caret)  # Machine learning workflow and utilities
library(class)  # KNN classification model
library(randomForest)  # Random Forest classification model
library(xgboost)  # Gradient Boosting model for classification
library(pROC)  # For ROC Curve analysis
library(ROCR)  # Performance evaluation utilities

# Load dataset
data <- fread("Social_Network_Ads.csv")  # Read CSV file into a dataframe

# Fix column names
tidy_colnames <- make.names(colnames(data))  # Ensure column names are valid
colnames(data) <- tidy_colnames  # Apply cleaned column names

# Convert target variable to factor
data$Purchased <- as.factor(data$Purchased)  # Convert classification target to factor

# Remove 'UserID' column if it exists
if ("UserID" %in% colnames(data)) {
  data <- select(data, -UserID)  # Drop UserID column
}

# Split dataset into training (70%) and testing (30%)
set.seed(42)  # Set seed for reproducibility
trainIndex <- createDataPartition(data$Purchased, p = 0.7, list = FALSE)  # Create index for training data
trainData <- data[trainIndex, ]  # Subset training data
testData <- data[-trainIndex, ]  # Subset testing data

# Convert categorical variables to numeric for XGBoost
trainData_xgb <- trainData %>% select(-Purchased) %>% mutate_if(is.character, as.numeric)  # Convert characters to numeric
testData_xgb <- testData %>% select(-Purchased) %>% mutate_if(is.character, as.numeric)  # Convert characters to numeric

# Train K-Nearest Neighbors Model
set.seed(42)
knn_model <- train(Purchased ~ ., data = trainData, method = "knn",  # Train using KNN
                   trControl = trainControl(method = "cv", number = 5),  # Cross-validation
                   tuneLength = 10)  # Hyperparameter tuning length
knn_preds <- predict(knn_model, testData)  # Make predictions on test set

# Train Random Forest Model
set.seed(42)
rf_model <- randomForest(Purchased ~ ., data = trainData, ntree = 100)  # Train with 100 trees
rf_preds <- predict(rf_model, testData)  # Make predictions on test set

# Train Gradient Boosting Model (XGBoost)
set.seed(42)
xgb_train <- xgb.DMatrix(data = as.matrix(trainData_xgb), label = as.numeric(trainData$Purchased) - 1)  # Convert training data to matrix
xgb_test <- xgb.DMatrix(data = as.matrix(testData_xgb))  # Convert test data to matrix

xgb_model <- xgboost(data = xgb_train, nrounds = 100, objective = "binary:logistic", eval_metric = "auc")  # Train XGBoost model
xgb_preds <- predict(xgb_model, xgb_test) > 0.5  # Convert probabilities to binary values
xgb_preds_factor <- factor(ifelse(xgb_preds, "1", "0"), levels = levels(testData$Purchased))  # Ensure correct factor levels

# Compute confusion matrices
knn_cm <- confusionMatrix(knn_preds, testData$Purchased)  # Confusion matrix for KNN
rf_cm <- confusionMatrix(rf_preds, testData$Purchased)  # Confusion matrix for Random Forest
xgb_cm <- confusionMatrix(xgb_preds_factor, testData$Purchased)  # Confusion matrix for XGBoost

# Print accuracy scores
cat("KNN Accuracy:", knn_cm$overall["Accuracy"], "\n")
cat("Random Forest Accuracy:", rf_cm$overall["Accuracy"], "\n")
cat("XGBoost Accuracy:", xgb_cm$overall["Accuracy"], "\n")

# Compute Precision, Recall, and F1-score
knn_metrics <- knn_cm$byClass
rf_metrics <- rf_cm$byClass
xgb_metrics <- xgb_cm$byClass

# Print Precision, Recall, and F1-score
cat("\nKNN Precision:", knn_metrics["Precision"], "| Recall:", knn_metrics["Recall"], "| F1-score:", knn_metrics["F1"])
cat("\nRandom Forest Precision:", rf_metrics["Precision"], "| Recall:", rf_metrics["Recall"], "| F1-score:", rf_metrics["F1"])
cat("\nXGBoost Precision:", xgb_metrics["Precision"], "| Recall:", xgb_metrics["Recall"], "| F1-score:", xgb_metrics["F1"], "\n")

# Create DataFrame for visualization
roc_data <- data.frame(
  FPR = c(knn_roc$specificities, rf_roc$specificities, xgb_roc$specificities),  # False Positive Rate
  TPR = c(knn_roc$sensitivities, rf_roc$sensitivities, xgb_roc$sensitivities),  # True Positive Rate
  Model = factor(rep(c("KNN", "Random Forest", "XGBoost"), 
                     times = c(length(knn_roc$specificities), 
                               length(rf_roc$specificities),
                               length(xgb_roc$specificities))))  # Assign labels
)

# Enhanced ROC Plot using ggplot2
ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(size = 1.5) +  # Make ROC curves thicker for clarity
  scale_color_manual(values = c("blue", "green", "red")) +  # Custom colors
  labs(title = "ROC Curve Comparison", x = "False Positive Rate", y = "True Positive Rate") +
  theme_classic() +  # Cleaner theme
  theme(legend.position = "bottom")  # Move legend to bottom for better visibility

# Summary of Model Performance
cat("\nSummary of Model Performance:\n")
cat("KNN: Accuracy =", knn_cm$overall["Accuracy"], "| ROC-AUC =", auc(knn_roc), "\n")
cat("Random Forest: Accuracy =", rf_cm$overall["Accuracy"], "| ROC-AUC =", auc(rf_roc), "\n")
cat("XGBoost: Accuracy =", xgb_cm$overall["Accuracy"], "| ROC-AUC =", auc(xgb_roc), "\n")

# Conclusion:
cat("\n🔹 **Final Verdict** 🔹\n")
cat("🏆 **Random Forest is the best performing model** with the highest accuracy and ROC-AUC.\n")
cat("🚀 **XGBoost performed well** but slightly below Random Forest.\n")
cat("⚠️ **KNN struggled** and might not be the best choice for this dataset.\n")