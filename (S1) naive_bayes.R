

# ==============================================
# 🧠 Naïve Bayes Classifier - Theory & Implementation
# ==============================================

# 📌 Overview:
# Naïve Bayes (NB) is a family of probabilistic algorithms used for classification based on Bayes' theorem.
# It assumes that the features are conditionally independent given the class label.
# Despite this assumption, Naïve Bayes is widely used in text classification, spam filtering, sentiment analysis, 
# and medical diagnosis due to its simplicity and efficiency.

# 📌 What is Naïve Bayes?
# NB classifiers apply Bayes' theorem to calculate the probability of a class given a set of features.
# Despite the "naïve" assumption of feature independence, NB performs well in various applications, especially with high-dimensional data.

# 📌 Popular Naïve Bayes Variants:
# - **Gaussian Naïve Bayes (GNB) 🔵**: Assumes continuous features follow a normal distribution.
# - **Multinomial Naïve Bayes (MNB) 🟠**: Best suited for text classification based on word frequencies.
# - **Bernoulli Naïve Bayes (BNB) 🟢**: Suitable for binary feature classification (presence/absence).
# - **Complement Naïve Bayes (CNB) 🔵**: A variation of MNB, often better for imbalanced datasets.

# ✅ **Key Strengths**: Fast training, effective for large datasets, works well with noisy data.
# ⚠️ **Limitations**: Assumes feature independence, struggles with correlated variables.

# 📌 The Naïve Bayes Formula:
# P(C_k | X) = (P(X | C_k) * P(C_k)) / P(X)
# Where:
# - P(C_k | X): Posterior probability - probability of class C_k given input data X.
# - P(X | C_k): Likelihood - probability of observing X given C_k.
# - P(C_k): Prior probability - initial belief about class C_k before seeing X.
# - P(X): Evidence - total probability of X across all possible classes.

# 📌 Choosing the Right Naïve Bayes Variant:
# - Gaussian Naïve Bayes (GNB) 🟡 - Assumes that numerical features follow a normal distribution.
# - Multinomial Naïve Bayes (MNB) 🔵 - Best for text classification (e.g., spam detection, sentiment analysis).
# - Bernoulli Naïve Bayes (BNB) 🟢 - Works with binary data (e.g., presence/absence of words).
# - Complement Naïve Bayes (CNB) 🔵 - A variation of MNB, better for imbalanced datasets.

# 📌 Key Insights:
# ✅ NB is computationally efficient and effective for large-scale classification tasks.
# ✅ Works well with small datasets and high-dimensional data.
# ✅ Assumes feature independence, which may be a limitation for correlated variables.
# ✅ Frequently used in NLP applications like spam filtering and text classification.
# ✅ Despite assumptions, NB is a strong baseline and often outperforms complex models in text-based applications.


# ===============================
# 📥 Load necessary libraries
# ===============================

library(e1071)      # Naïve Bayes model
library(ggplot2)    # Visualization
library(caret)      # Data processing & confusion matrix
library(dplyr)      # Data manipulation
library(pROC)       # ROC-AUC calculation

# ===============================
# 📂 Load and preprocess dataset
# ===============================

# Load dataset
social_data <- read.csv("Social_Network_Ads.csv")

# Remove 'User.ID' if it exists
if ("User.ID" %in% colnames(social_data)) {
  social_data <- select(social_data, -User.ID)
}

# Convert 'Purchased' column to factor
social_data$Purchased <- as.factor(social_data$Purchased)

# Split dataset into training (70%) and testing (30%)
set.seed(42) 
train_index <- createDataPartition(social_data$Purchased, p = 0.7, list = FALSE)
train_set <- social_data[train_index, ]
test_set <- social_data[-train_index, ]

# ===============================
# 🔥 Train Naïve Bayes model
# ===============================

nb_model <- naiveBayes(Purchased ~ ., data = train_set, laplace = 1)

# Predict on the test set
y_pred <- predict(nb_model, test_set)

# ===============================
# 📊 Confusion Matrix & Performance Metrics
# ===============================

# Compute confusion matrix
cm <- confusionMatrix(y_pred, test_set$Purchased)

# Extract performance metrics
metrics <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"),
  Value = c(
    cm$overall["Accuracy"],
    cm$byClass["Precision"],
    cm$byClass["Recall"],
    cm$byClass["F1"],
    auc(as.numeric(test_set$Purchased), as.numeric(y_pred))
  )
)

# Display metrics in Viewer
View(metrics)

# Extract TP, FP, FN, TN
TP <- cm$table[2,2]  # True Positives
FP <- cm$table[2,1]  # False Positives
FN <- cm$table[1,2]  # False Negatives
TN <- cm$table[1,1]  # True Negatives

# Convert confusion matrix to dataframe and display in Viewer
cm_matrix <- as.data.frame(cm$table)
colnames(cm_matrix) <- c("Actual", "Predicted", "Freq")
View(cm_matrix)

# ===============================
# 📉 Decision Boundary Visualization
# ===============================

plot_decision_boundary <- function(model, data, title) {
  # Create a grid of values for Age and Estimated Salary
  grid_x <- seq(min(data$Age) - 2, max(data$Age) + 2, by = 0.1)
  grid_y <- seq(min(data$EstimatedSalary) - 2000, max(data$EstimatedSalary) + 2000, by = 500)
  grid <- expand.grid(Age = grid_x, EstimatedSalary = grid_y)
  
  # Predict class labels for the grid
  grid$Purchased <- predict(model, grid, type = "class")
  grid$Purchased <- as.factor(grid$Purchased)
  
  # Plot decision boundary
  ggplot(data, aes(x = Age, y = EstimatedSalary, color = Purchased)) +
    geom_point(size = 3, alpha = 0.8) +
    geom_tile(data = grid, aes(fill = Purchased, color = NULL), alpha = 0.3) +
    scale_color_manual(values = c("blue", "red")) +
    scale_fill_manual(values = c("lightblue", "lightcoral")) +
    labs(title = title, x = "Age", y = "Estimated Salary") +
    theme_minimal()
}

# Plot decision boundary
plot_decision_boundary(nb_model, test_set, "Decision Boundary: Naïve Bayes")