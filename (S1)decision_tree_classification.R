# ==============================================
# 🧠 Decision Tree Classifier - Theory & Implementation
# ==============================================

# 📌 Overview:
# Decision Trees (DT) are supervised learning algorithms used for classification and regression.
# They split the dataset into smaller subsets based on feature conditions, forming a tree-like structure.
# DTs are popular due to their **simplicity** and **interpretability**.

# ✅ Easy to interpret and visualize.
# ✅ Requires little data preprocessing (no need for feature scaling or normalization).
# ✅ Handles both numerical and categorical data.

# 📌 Limitations:
# ⚠️ Prone to overfitting, especially with deep trees.
# ⚠️ Can be sensitive to small changes in data.

# ===============================
# 📥 Load necessary libraries
# ===============================

library(rpart)       # Decision Tree Model
library(ggplot2)     # Visualization
library(caret)       # Data Processing & Evaluation
library(dplyr)       # Data Manipulation
library(pROC)        # ROC-AUC Calculation
library(reshape2)    # Data reshaping for visualization

# ===============================
# 📂 Load and preprocess dataset
# ===============================

social_data <- read.csv("Social_Network_Ads.csv")  # Load dataset

# Remove 'User.ID' column if it exists
if ("User.ID" %in% colnames(social_data)) {
  social_data <- select(social_data, -User.ID)  # Drop 'User.ID' column
}

# Convert categorical variables to factors
social_data$Purchased <- as.factor(social_data$Purchased)  # Convert target variable to factor

# Handle 'Gender' column (convert to factor if exists)
if ("Gender" %in% colnames(social_data)) {
  social_data$Gender <- as.factor(social_data$Gender)  # Convert Gender column to factor
}

# Set random seed for reproducibility
set.seed(42)  

# Split dataset into 70% training and 30% testing
train_index <- createDataPartition(social_data$Purchased, p = 0.7, list = FALSE)  # Create partition index
train_set <- social_data[train_index, ]  # Training set
test_set <- social_data[-train_index, ]  # Test set

# ===============================
# 🌳 Train Decision Tree Model
# ===============================

dt_model <- rpart(Purchased ~ ., data = train_set, method = "class")  # Train Decision Tree model

# Predict class labels on the test set
y_pred <- predict(dt_model, test_set, type = "class")  

# ===============================
# 📊 Confusion Matrix & Performance Metrics
# ===============================

cm <- confusionMatrix(y_pred, test_set$Purchased)  # Compute confusion matrix

# Extract key performance metrics
metrics <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"),  # Define metric names
  Value = c(
    cm$overall["Accuracy"],  # Extract accuracy
    cm$byClass["Precision"],  # Extract precision
    cm$byClass["Recall"],  # Extract recall
    cm$byClass["F1"],  # Extract F1-score
    auc(as.numeric(test_set$Purchased), as.numeric(y_pred))  # Compute ROC-AUC
  )
)

View(metrics)  # Display metrics in RStudio Viewer

# Convert confusion matrix to dataframe for visualization
cm_matrix <- as.data.frame(cm$table)  
colnames(cm_matrix) <- c("Actual", "Predicted", "Freq")  # Rename columns
View(cm_matrix)  # Display confusion matrix in Viewer

# ===============================
# 📊 Visualizing Confusion Matrix
# ===============================

ggplot(cm_matrix, aes(x = Actual, y = Predicted, fill = Freq)) +  # Set up heatmap
  geom_tile(color = "black") +  # Create heatmap with black borders
  geom_text(aes(label = Freq), size = 6, color = "white") +  # Add labels
  scale_fill_gradient(low = "blue", high = "red") +  # Color gradient
  labs(title = "Confusion Matrix", x = "Actual Label", y = "Predicted Label") +  # Set labels
  theme_minimal()  # Use minimal theme for better readability

# ===============================
# 📉 Decision Boundary Visualization
# ===============================

plot_decision_boundary <- function(model, data, title) {  
  # Create a grid of values for Age and Estimated Salary
  grid_x <- seq(min(data$Age) - 2, max(data$Age) + 2, by = 0.1)  # Generate Age values
  grid_y <- seq(min(data$EstimatedSalary) - 2000, max(data$EstimatedSalary) + 2000, by = 500)  # Generate Salary values
  grid <- expand.grid(Age = grid_x, EstimatedSalary = grid_y)  # Create a combination of Age & Salary
  
  # Handle 'Gender' column if it exists in training data
  if ("Gender" %in% colnames(train_set)) {
    grid$Gender <- factor(sample(levels(train_set$Gender), nrow(grid), replace = TRUE))  # Randomly assign Gender values
  }
  
  # Predict class labels for the grid
  grid$Purchased <- predict(model, grid, type = "class")  # Predict classes
  grid$Purchased <- as.factor(grid$Purchased)  # Convert predictions to factor
  
  # Plot decision boundary using ggplot2
  ggplot(data, aes(x = Age, y = EstimatedSalary, color = Purchased)) +  
    geom_point(size = 3, alpha = 0.8) +  # Plot actual data points
    geom_tile(data = grid, aes(fill = Purchased, color = NULL), alpha = 0.3) +  # Plot decision boundary
    scale_color_manual(values = c("blue", "red")) +  # Define colors for points
    scale_fill_manual(values = c("lightblue", "lightcoral")) +  # Define colors for boundary
    labs(title = title, x = "Age", y = "Estimated Salary") +  # Set title and labels
    theme_minimal()  # Apply minimal theme
}

# ===============================
# 📍 Plot Decision Boundaries for Training & Test Sets
# ===============================

plot_decision_boundary(dt_model, train_set, "Decision Boundary (Training Data)")  # Training Data
plot_decision_boundary(dt_model, test_set, "Decision Boundary (Test Data)")  # Test Data